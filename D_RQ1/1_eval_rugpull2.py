#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import importlib.util
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Config (edit here)
# ============================================================
END_BLOCK_CAP = 19777901
MONTH_BLOCKS = 210000

SCENARIOS = {
    "s1": {"trust": 20, "rugpull": 20, "unknown": 20},
    "s2": {"trust": 20, "rugpull": 20, "unknown": "all"},
}

LABEL_RANDOM_SEED = 20260226
TOPK_LIST = [5, 10, 20]
OUTPUT_DIR = Path("output/D/1_eval_rugpull2")
SAVE_INTERMEDIATE_CONTRACTS = False
STAGEA_REUSE = True

LABEL_CSV = Path("output/C/3_collectionlabel.csv")
TRP_SCRIPT = Path("D_RQ1/0_rugpullTRP.py")
FIG_DPI = 140


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_label_df(label_df: pd.DataFrame) -> pd.DataFrame:
    df = label_df.copy()
    need = {"contract_address", "collection_type", "source_tx_file", "create_block_number"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in label csv: {miss}")

    df["contract_address"] = df["contract_address"].astype(str).str.lower()
    df["collection_type"] = df["collection_type"].astype(str).str.lower()
    df["source_tx_file"] = df["source_tx_file"].astype(str).str.strip()
    df["create_block_number"] = pd.to_numeric(df["create_block_number"], errors="coerce")
    df = df[df["source_tx_file"] != ""].copy()
    return df


def build_scenario_universe(
    label_df: pd.DataFrame,
    scenario_cfg: dict,
    seed: int,
) -> tuple[list[Path], pd.DataFrame, dict, int]:
    rng = random.Random(seed)
    df = normalize_label_df(label_df)
    pool = df[df["collection_type"].isin(["trust", "rugpull", "unknown"])].copy()

    picked_parts = []
    sample_desc = []
    for t in ["trust", "rugpull", "unknown"]:
        sub = pool[pool["collection_type"] == t].copy()
        n_cfg = scenario_cfg.get(t, 0)
        if isinstance(n_cfg, str) and n_cfg.lower() == "all":
            pick = sub
            sample_desc.append(f"{t}:all/{len(sub)}")
        else:
            n_take = min(int(n_cfg), len(sub))
            idx = list(sub.index)
            rng.shuffle(idx)
            pick = sub.loc[idx[:n_take]].copy()
            sample_desc.append(f"{t}:{n_take}/{len(sub)}")
        picked_parts.append(pick)

    picked = pd.concat(picked_parts, ignore_index=True)
    if picked.empty:
        raise ValueError("Scenario selected empty universe.")

    # de-dup by address
    picked = picked.sort_values(["contract_address", "source_tx_file"], kind="mergesort")
    picked = picked.drop_duplicates(subset=["contract_address"], keep="first").reset_index(drop=True)

    files = []
    for p in picked["source_tx_file"].tolist():
        fp = Path(p)
        if fp.exists():
            files.append(fp)
    if not files:
        raise FileNotFoundError("No existing source_tx_file for selected scenario.")

    type_map = picked[["contract_address", "collection_type"]].copy()
    start_block_min = int(pd.to_numeric(picked["create_block_number"], errors="coerce").dropna().min())

    meta = {
        "sample_desc": ";".join(sample_desc),
        "selected_contracts": int(type_map["contract_address"].nunique()),
        "selected_files": int(len(files)),
        "start_block_min": int(start_block_min),
    }
    return files, type_map, meta, start_block_min


def build_monthly_end_blocks(start_block_min: int, month_blocks: int, end_cap: int) -> list[int]:
    if month_blocks <= 0:
        raise ValueError("MONTH_BLOCKS must be > 0")
    ends = []
    cur = int(start_block_min) + int(month_blocks)
    while cur < int(end_cap):
        ends.append(int(cur))
        cur += int(month_blocks)
    if not ends or ends[-1] != int(end_cap):
        ends.append(int(end_cap))
    return sorted(set(ends))


def run_scenario_snapshots(
    trp,
    scenario_name: str,
    files: list[Path],
    type_map: pd.DataFrame,
    start_block_min: int,
    end_blocks: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_all = trp.load_events_from_files(files)
    if df_all.empty:
        raise ValueError(f"[{scenario_name}] no events loaded.")

    panel_rows = []
    rug_rows = []
    all_brief_rows = []

    stagea_cache = None
    # force faster reuse for monthly snapshots
    if STAGEA_REUSE:
        trp.STAGEA_REFRESH_HOURS = None
        trp.STAGEA_REFRESH_BLOCKS = None
        trp.FORCE_STAGEA_REBUILD = False

    for idx, end_block in enumerate(end_blocks, 1):
        df_hist = df_all[df_all["block_number"] <= int(end_block)].copy()
        df_win = df_all[(df_all["block_number"] >= int(start_block_min)) & (df_all["block_number"] <= int(end_block))].copy()
        if df_win.empty:
            continue

        end_time = trp.block_to_time(df_all, int(end_block))
        rebuild, reason = trp.should_rebuild(stagea_cache, int(end_block), end_time)
        if stagea_cache is None or rebuild:
            stagea_contract, stagea_user = trp.run_stage_a(df_hist)
            stagea_cache = {
                "end_block": int(end_block),
                "end_time": end_time,
                "stagea_contract": stagea_contract,
                "stagea_user": stagea_user,
            }
        stagea_contract = stagea_cache["stagea_contract"]
        stagea_user = stagea_cache["stagea_user"]

        u0_full = trp.run_stage_b(df_hist, stagea_contract, stagea_user)
        contracts_out, _users_out, _cmeta = trp.run_stage_c(
            df_window=df_win,
            u0_full=u0_full,
            stagea_contract=stagea_contract,
            tau_end=int(end_block),
            contract_type_map=type_map,
        )
        if contracts_out.empty:
            continue

        contracts_out = contracts_out.copy()
        contracts_out["collection_type"] = contracts_out["collection_type"].fillna("unknown").astype(str).str.lower()
        contracts_out["scenario"] = scenario_name
        contracts_out["snapshot_idx"] = int(idx)
        contracts_out["end_block"] = int(end_block)
        contracts_out["start_block"] = int(start_block_min)

        # brief all-contract output
        all_brief = contracts_out[["scenario", "snapshot_idx", "start_block", "end_block", "contract_id", "contract_address", "collection_type", "rank_C", "cC"]].copy()
        all_brief_rows.append(all_brief)

        rug = contracts_out[contracts_out["collection_type"] == "rugpull"].copy()
        if not rug.empty:
            rug_brief = rug[["scenario", "snapshot_idx", "start_block", "end_block", "contract_id", "contract_address", "rank_C", "cC"]].copy()
            rug_rows.append(rug_brief)
            n_total = int(len(contracts_out))

            panel_rows.append(
                {
                    "scenario": scenario_name,
                    "snapshot_idx": int(idx),
                    "end_block": int(end_block),
                    "rugpull_n": int(len(rug)),
                    "rank_mean": float(rug["rank_C"].mean()),
                    "rank_median": float(rug["rank_C"].median()),
                    "rank_p25": float(rug["rank_C"].quantile(0.25)),
                    "rank_p75": float(rug["rank_C"].quantile(0.75)),
                    "rank_min": float(rug["rank_C"].min()),
                    "rank_max": float(rug["rank_C"].max()),
                    "cC_mean": float(rug["cC"].mean()),
                    "cC_median": float(rug["cC"].median()),
                    # "top{k}_rate" is defined as reverse-top-k (i.e., bottom-k by rank).
                    **{
                        f"top{k}_rate": float((rug["rank_C"] >= max(1, n_total - int(k) + 1)).mean())
                        for k in TOPK_LIST
                    },
                }
            )
        else:
            panel_rows.append(
                {
                    "scenario": scenario_name,
                    "snapshot_idx": int(idx),
                    "end_block": int(end_block),
                    "rugpull_n": 0,
                    "rank_mean": np.nan,
                    "rank_median": np.nan,
                    "rank_p25": np.nan,
                    "rank_p75": np.nan,
                    "rank_min": np.nan,
                    "rank_max": np.nan,
                    "cC_mean": np.nan,
                    "cC_median": np.nan,
                    **{f"top{k}_rate": np.nan for k in TOPK_LIST},
                }
            )

        if SAVE_INTERMEDIATE_CONTRACTS:
            outp = OUTPUT_DIR / f"{scenario_name}_contracts_{int(end_block)}.csv"
            contracts_out.to_csv(outp, index=False, encoding="utf-8-sig")

    panel_df = pd.DataFrame(panel_rows).sort_values(["snapshot_idx", "end_block"]).reset_index(drop=True)
    rug_df = pd.concat(rug_rows, ignore_index=True) if rug_rows else pd.DataFrame(
        columns=["scenario", "snapshot_idx", "start_block", "end_block", "contract_id", "contract_address", "rank_C", "cC"]
    )
    all_brief_df = pd.concat(all_brief_rows, ignore_index=True) if all_brief_rows else pd.DataFrame(
        columns=["scenario", "snapshot_idx", "start_block", "end_block", "contract_id", "contract_address", "collection_type", "rank_C", "cC"]
    )

    summary_cols = ["scenario", "snapshot_idx", "end_block", "rugpull_n", "rank_mean", "rank_median", "rank_p25", "rank_p75"] + [f"top{k}_rate" for k in TOPK_LIST]
    summary_df = panel_df[summary_cols].copy() if not panel_df.empty else pd.DataFrame(columns=summary_cols)
    return panel_df, rug_df, summary_df, all_brief_df


def plot_rank_trends(scenario_panels: dict[str, pd.DataFrame], scenario_rugs: dict[str, pd.DataFrame]) -> None:
    # Per-scenario median and quantiles
    for s, panel in scenario_panels.items():
        if panel.empty:
            continue
        x = panel["snapshot_idx"]

        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.plot(x, panel["rank_median"], label="median", linewidth=2.0)
        ax.fill_between(x, panel["rank_p25"], panel["rank_p75"], alpha=0.25, label="P25-P75")
        ax.set_title(f"{s.upper()} Rugpull Rank Over Snapshots")
        ax.set_xlabel("Snapshot Index")
        ax.set_ylabel("Rank_C (lower is better)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{s}_rugpull_rank_quantiles_over_time.png", dpi=FIG_DPI)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8.8, 4.8))
        ax2.plot(x, panel["rank_median"], linewidth=2.1, color="#0b6")
        ax2.set_title(f"{s.upper()} Rugpull Median Rank Over Snapshots")
        ax2.set_xlabel("Snapshot Index")
        ax2.set_ylabel("Median rank_C")
        ax2.grid(alpha=0.25)
        fig2.tight_layout()
        fig2.savefig(OUTPUT_DIR / f"{s}_rugpull_rank_median_over_time.png", dpi=FIG_DPI)
        plt.close(fig2)

    # Compare median curves
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for s, panel in scenario_panels.items():
        if panel.empty:
            continue
        ax.plot(panel["snapshot_idx"], panel["rank_median"], marker="o", markersize=3, linewidth=1.8, label=s.upper())
    ax.set_title("Scenario Compare: Rugpull Median Rank")
    ax.set_xlabel("Snapshot Index")
    ax.set_ylabel("Median rank_C")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scenario_compare_rugpull_rank_median.png", dpi=FIG_DPI)
    plt.close(fig)

    # Last snapshot boxplot compare
    box_data = []
    box_labels = []
    for s, rug in scenario_rugs.items():
        if rug.empty:
            continue
        last_idx = int(rug["snapshot_idx"].max())
        vals = rug[rug["snapshot_idx"] == last_idx]["rank_C"].dropna().to_numpy()
        if len(vals) > 0:
            box_data.append(vals)
            box_labels.append(f"{s.upper()}@last")
    if box_data:
        fig2, ax2 = plt.subplots(figsize=(7.2, 4.8))
        ax2.boxplot(box_data, labels=box_labels, showfliers=True)
        ax2.set_title("Scenario Compare: Rugpull Rank Boxplot (Last Snapshot)")
        ax2.set_ylabel("rank_C")
        ax2.grid(alpha=0.2, axis="y")
        fig2.tight_layout()
        fig2.savefig(OUTPUT_DIR / "scenario_compare_rugpull_rank_box_last_snapshot.png", dpi=FIG_DPI)
        plt.close(fig2)

    # ECDF for last snapshot
    fig3, ax3 = plt.subplots(figsize=(8.4, 4.8))
    has_any = False
    for s, rug in scenario_rugs.items():
        if rug.empty:
            continue
        last_idx = int(rug["snapshot_idx"].max())
        vals = np.sort(rug[rug["snapshot_idx"] == last_idx]["rank_C"].dropna().to_numpy())
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / float(len(vals))
        ax3.step(vals, y, where="post", label=f"{s.upper()}@last")
        has_any = True
    if has_any:
        ax3.set_title("Rugpull Rank ECDF (Last Snapshot)")
        ax3.set_xlabel("rank_C")
        ax3.set_ylabel("ECDF")
        ax3.grid(alpha=0.25)
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig(OUTPUT_DIR / "rugpull_rank_ecdf_last_snapshot.png", dpi=FIG_DPI)
    plt.close(fig3)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not LABEL_CSV.exists():
        raise FileNotFoundError(f"Label csv not found: {LABEL_CSV.resolve()}")

    trp = load_module(TRP_SCRIPT, "trp_mod_for_eval2")
    label_df = pd.read_csv(LABEL_CSV)

    compare_rows = []
    scenario_panels = {}
    scenario_rugs = {}

    for i, (sname, scfg) in enumerate(SCENARIOS.items()):
        files, type_map, smeta, start_block_min = build_scenario_universe(
            label_df=label_df,
            scenario_cfg=scfg,
            seed=LABEL_RANDOM_SEED + i,
        )
        end_blocks = build_monthly_end_blocks(start_block_min, MONTH_BLOCKS, END_BLOCK_CAP)

        panel_df, rug_df, summary_df, all_brief_df = run_scenario_snapshots(
            trp=trp,
            scenario_name=sname,
            files=files,
            type_map=type_map,
            start_block_min=start_block_min,
            end_blocks=end_blocks,
        )

        panel_df.to_csv(OUTPUT_DIR / f"scenario_{sname}_panel.csv", index=False, encoding="utf-8-sig")
        rug_df.to_csv(OUTPUT_DIR / f"scenario_{sname}_rugpull_snapshot_ranks.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(OUTPUT_DIR / f"scenario_{sname}_rank_summary_by_snapshot.csv", index=False, encoding="utf-8-sig")
        all_brief_df.to_csv(OUTPUT_DIR / f"scenario_{sname}_all_contract_scores_brief.csv", index=False, encoding="utf-8-sig")

        scenario_panels[sname] = panel_df
        scenario_rugs[sname] = rug_df

        compare_rows.append(
            {
                "scenario": sname,
                "sample_desc": smeta["sample_desc"],
                "selected_contracts": smeta["selected_contracts"],
                "selected_files": smeta["selected_files"],
                "start_block_min": start_block_min,
                "n_snapshots": len(end_blocks),
                "first_end_block": int(end_blocks[0]) if end_blocks else np.nan,
                "last_end_block": int(end_blocks[-1]) if end_blocks else np.nan,
                "rugpull_rows_total": int(len(rug_df)),
                "final_median_rank": float(summary_df["rank_median"].dropna().iloc[-1]) if not summary_df.empty and summary_df["rank_median"].notna().any() else np.nan,
                **{
                    f"final_top{k}_rate": float(summary_df[f"top{k}_rate"].dropna().iloc[-1])
                    if (not summary_df.empty and f"top{k}_rate" in summary_df.columns and summary_df[f"top{k}_rate"].notna().any())
                    else np.nan
                    for k in TOPK_LIST
                },
            }
        )

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(OUTPUT_DIR / "scenario_compare_rank_summary.csv", index=False, encoding="utf-8-sig")

    plot_rank_trends(scenario_panels, scenario_rugs)

    run_meta = pd.DataFrame(
        [
            {"key": "end_block_cap", "value": END_BLOCK_CAP},
            {"key": "month_blocks", "value": MONTH_BLOCKS},
            {"key": "label_random_seed", "value": LABEL_RANDOM_SEED},
            {"key": "save_intermediate_contracts", "value": int(1 if SAVE_INTERMEDIATE_CONTRACTS else 0)},
            {"key": "stagea_reuse", "value": int(1 if STAGEA_REUSE else 0)},
            {"key": "topk_list", "value": "|".join(str(k) for k in TOPK_LIST)},
            {"key": "scenarios", "value": str(SCENARIOS)},
        ]
    )
    run_meta.to_csv(OUTPUT_DIR / "run_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'scenario_s1_rugpull_snapshot_ranks.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'scenario_s2_rugpull_snapshot_ranks.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'scenario_compare_rank_summary.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'run_meta.csv').resolve()}")
    print(f"[Saved] figures in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
