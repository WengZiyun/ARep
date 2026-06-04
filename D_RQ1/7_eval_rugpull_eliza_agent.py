#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import importlib.util
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================
LABEL_CSV = Path("output/C/3_collectionlabel.csv")
TRP_SCRIPT = Path("D_RQ1/0_rugpullTRP.py")
BASE_EVAL_SCRIPT = Path("D_RQ1/1_eval_rugpull3.py")
OUTPUT_DIR = Path("output/D/7_eval_rugpull_eliza_agent")

SEVEN_DAYS_SECONDS = 604800
TOP_PERCENT_LIST = [0.10, 0.20, 0.30]
FIG_DPI = 140

# Eliza agent settings
ELIZA_TOP_USER_PCT = 0.05
ELIZA_TOP_USER_MIN = 20
ELIZA_TOP_USER_MAX = 300
ELIZA_AGENT_MIX = 0.35
ELIZA_EVENT_VALUE_LOG_SCALE = 1.0
STAGEA_REUSE = True


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def format_seconds(sec: float) -> str:
    sec_i = int(max(0, round(float(sec))))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def normalize_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if x.empty:
        return x
    xmin = float(x.min())
    xmax = float(x.max())
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    if xmax - xmin <= 1e-18:
        return pd.Series(np.ones(len(x), dtype=float), index=x.index)
    return (x - xmin) / (xmax - xmin)


def build_eliza_user_core(users_out: pd.DataFrame) -> pd.DataFrame:
    if users_out.empty:
        return pd.DataFrame(columns=["user", "uC", "rank_C_user", "eliza_user_weight"])

    work = users_out.copy()
    work["user"] = work["user"].astype(str).str.strip().str.lower()
    work["uC"] = pd.to_numeric(work["uC"], errors="coerce").fillna(0.0)
    work = work.sort_values(["uC", "user"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    n_users = int(len(work))
    take_n = int(np.ceil(float(n_users) * float(ELIZA_TOP_USER_PCT)))
    take_n = max(int(ELIZA_TOP_USER_MIN), take_n)
    take_n = min(int(ELIZA_TOP_USER_MAX), take_n, n_users)
    core = work.head(take_n).copy()

    mass = float(core["uC"].sum())
    if mass > 0:
        core["eliza_user_weight"] = core["uC"] / mass
    else:
        core["eliza_user_weight"] = 1.0 / float(max(1, len(core)))
    return core[["user", "uC", "rank_C_user", "eliza_user_weight"]]


def build_contract_agent_support(df_win: pd.DataFrame, eliza_core: pd.DataFrame) -> pd.DataFrame:
    if df_win.empty or eliza_core.empty:
        return pd.DataFrame(columns=["contract_id", "eliza_support_raw", "eliza_support_norm", "eliza_core_users_on_contract"])

    buy_df = df_win[df_win["tx_type"].isin({"mint", "trade"})].copy()
    if buy_df.empty:
        return pd.DataFrame(columns=["contract_id", "eliza_support_raw", "eliza_support_norm", "eliza_core_users_on_contract"])

    buy_df["buyer"] = buy_df["buyer"].fillna("").astype(str).str.lower().str.strip()
    buy_df["price"] = pd.to_numeric(buy_df["price"], errors="coerce").fillna(0.0).astype(float)
    buy_df = buy_df[buy_df["buyer"] != ""].copy()
    if buy_df.empty:
        return pd.DataFrame(columns=["contract_id", "eliza_support_raw", "eliza_support_norm", "eliza_core_users_on_contract"])

    buy_df["event_weight"] = 1.0 + float(ELIZA_EVENT_VALUE_LOG_SCALE) * np.log1p(np.maximum(0.0, buy_df["price"].to_numpy(dtype=float)))
    exposure = (
        buy_df.groupby(["contract_id", "buyer"], as_index=False)
        .agg(
            user_contract_events=("buyer", "size"),
            user_contract_value_usd=("price", "sum"),
            event_weight=("event_weight", "sum"),
        )
        .rename(columns={"buyer": "user"})
    )

    merged = exposure.merge(eliza_core, on="user", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["contract_id", "eliza_support_raw", "eliza_support_norm", "eliza_core_users_on_contract"])

    merged["agent_contrib"] = merged["event_weight"] * merged["eliza_user_weight"]
    out = (
        merged.groupby("contract_id", as_index=False)
        .agg(
            eliza_support_raw=("agent_contrib", "sum"),
            eliza_core_users_on_contract=("user", "nunique"),
        )
        .sort_values(["eliza_support_raw", "contract_id"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    out["eliza_support_norm"] = normalize_series(out["eliza_support_raw"])
    return out


def apply_eliza_feedback(contracts_out: pd.DataFrame, support_df: pd.DataFrame) -> pd.DataFrame:
    if contracts_out.empty:
        return contracts_out.copy()

    out = contracts_out.copy()
    out["cC"] = pd.to_numeric(out["cC"], errors="coerce").fillna(0.0)
    out = out.merge(support_df, on="contract_id", how="left")
    out["eliza_support_raw"] = pd.to_numeric(out["eliza_support_raw"], errors="coerce").fillna(0.0)
    out["eliza_support_norm"] = pd.to_numeric(out["eliza_support_norm"], errors="coerce").fillna(0.0)
    out["eliza_core_users_on_contract"] = pd.to_numeric(out["eliza_core_users_on_contract"], errors="coerce").fillna(0).astype(int)

    out["cC_base_norm"] = normalize_series(out["cC"])
    out["cC_eliza"] = (1.0 - float(ELIZA_AGENT_MIX)) * out["cC_base_norm"] + float(ELIZA_AGENT_MIX) * out["eliza_support_norm"]
    out = out.sort_values(["cC_eliza", "contract_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["rank_C_eliza"] = np.arange(1, len(out) + 1, dtype=np.int64)
    out["rank_pct_eliza"] = out["rank_C_eliza"] / float(max(1, len(out)))
    out["rank_delta_eliza"] = pd.to_numeric(out["rank_C"], errors="coerce").fillna(0).astype(int) - out["rank_C_eliza"]
    return out


def summarize_rugpull(snapshot_idx: int, start_block: int, end_block: int, adjusted: pd.DataFrame) -> dict:
    rug = adjusted[adjusted["collection_type"] == "rugpull"].copy()
    if rug.empty:
        return {
            "snapshot_idx": snapshot_idx,
            "start_block": start_block,
            "end_block": end_block,
            "rugpull_count": 0,
            "base_rank_median": np.nan,
            "eliza_rank_median": np.nan,
            "median_rank_gain": np.nan,
            "base_top10_rate": np.nan,
            "eliza_top10_rate": np.nan,
            "base_top20_rate": np.nan,
            "eliza_top20_rate": np.nan,
            "base_top30_rate": np.nan,
            "eliza_top30_rate": np.nan,
            "mean_eliza_support_rugpull": np.nan,
        }

    return {
        "snapshot_idx": snapshot_idx,
        "start_block": start_block,
        "end_block": end_block,
        "rugpull_count": int(len(rug)),
        "base_rank_median": float(pd.to_numeric(rug["rank_C"], errors="coerce").median()),
        "eliza_rank_median": float(pd.to_numeric(rug["rank_C_eliza"], errors="coerce").median()),
        "median_rank_gain": float(pd.to_numeric(rug["rank_delta_eliza"], errors="coerce").median()),
        "base_top10_rate": float((pd.to_numeric(rug["rank_pct"], errors="coerce") <= TOP_PERCENT_LIST[0]).mean()),
        "eliza_top10_rate": float((pd.to_numeric(rug["rank_pct_eliza"], errors="coerce") <= TOP_PERCENT_LIST[0]).mean()),
        "base_top20_rate": float((pd.to_numeric(rug["rank_pct"], errors="coerce") <= TOP_PERCENT_LIST[1]).mean()),
        "eliza_top20_rate": float((pd.to_numeric(rug["rank_pct_eliza"], errors="coerce") <= TOP_PERCENT_LIST[1]).mean()),
        "base_top30_rate": float((pd.to_numeric(rug["rank_pct"], errors="coerce") <= TOP_PERCENT_LIST[2]).mean()),
        "eliza_top30_rate": float((pd.to_numeric(rug["rank_pct_eliza"], errors="coerce") <= TOP_PERCENT_LIST[2]).mean()),
        "mean_eliza_support_rugpull": float(pd.to_numeric(rug["eliza_support_norm"], errors="coerce").mean()),
    }


def plot_outputs(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    x = pd.to_numeric(summary_df["snapshot_idx"], errors="coerce")

    fig1, ax1 = plt.subplots(figsize=(8.8, 4.8))
    ax1.plot(x, summary_df["base_rank_median"], label="Base median rank", linewidth=2.0, color="#4C78A8")
    ax1.plot(x, summary_df["eliza_rank_median"], label="Eliza-adjusted median rank", linewidth=2.0, color="#F58518")
    ax1.set_title("Rugpull Median Rank: Base vs Eliza Agent")
    ax1.set_xlabel("Snapshot Index")
    ax1.set_ylabel("Median rank (lower is better)")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / "rugpull_rank_median_base_vs_eliza.png", dpi=FIG_DPI)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.8, 4.8))
    ax2.plot(x, summary_df["base_top10_rate"], label="Base Top10%", linewidth=1.8, linestyle="--", color="#4C78A8")
    ax2.plot(x, summary_df["eliza_top10_rate"], label="Eliza Top10%", linewidth=1.8, color="#4C78A8")
    ax2.plot(x, summary_df["base_top20_rate"], label="Base Top20%", linewidth=1.8, linestyle="--", color="#54A24B")
    ax2.plot(x, summary_df["eliza_top20_rate"], label="Eliza Top20%", linewidth=1.8, color="#54A24B")
    ax2.plot(x, summary_df["base_top30_rate"], label="Base Top30%", linewidth=1.8, linestyle="--", color="#E45756")
    ax2.plot(x, summary_df["eliza_top30_rate"], label="Eliza Top30%", linewidth=1.8, color="#E45756")
    ax2.set_title("Rugpull Top-K Rate: Base vs Eliza Agent")
    ax2.set_xlabel("Snapshot Index")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.25)
    ax2.legend(ncol=2)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "rugpull_top_percent_base_vs_eliza.png", dpi=FIG_DPI)
    plt.close(fig2)


def run_snapshots(trp, files: list[Path], type_map: pd.DataFrame, plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df_all = trp.load_events_from_files(files)
    if df_all.empty:
        raise ValueError("No events loaded from selected files")

    stagea_cache = None
    if STAGEA_REUSE:
        trp.STAGEA_REFRESH_HOURS = None
        trp.STAGEA_REFRESH_BLOCKS = None
        trp.FORCE_STAGEA_REBUILD = False

    contract_rows = []
    eliza_rows = []
    summary_rows = []
    progress_meta = {
        "n_snapshots_planned": int(len(plan)),
        "n_snapshots_processed": 0,
        "eta_total_seconds_after_first_snapshot": np.nan,
        "snapshot_loop_elapsed_seconds": np.nan,
    }

    start_block = int(plan["start_block"].iloc[0])
    loop_t0 = perf_counter()
    total_snapshots = int(len(plan))
    print(f"[Progress] snapshots planned: {total_snapshots}")

    for r in plan.itertuples(index=False):
        snapshot_idx = int(r.snapshot_idx)
        end_block = int(r.end_block)

        df_hist = df_all[df_all["block_number"] <= end_block].copy()
        df_win = df_all[(df_all["block_number"] >= start_block) & (df_all["block_number"] <= end_block)].copy()
        if df_win.empty:
            continue

        end_time = trp.block_to_time(df_all, end_block)
        rebuild, _reason = trp.should_rebuild(stagea_cache, end_block, end_time)
        if stagea_cache is None or rebuild:
            stagea_contract, stagea_user = trp.run_stage_a(df_hist)
            stagea_cache = {
                "end_block": end_block,
                "end_time": end_time,
                "stagea_contract": stagea_contract,
                "stagea_user": stagea_user,
            }

        u0_full = trp.run_stage_b(df_hist, stagea_cache["stagea_contract"], stagea_cache["stagea_user"])
        contracts_out, users_out, _meta = trp.run_stage_c(
            df_window=df_win,
            u0_full=u0_full,
            stagea_contract=stagea_cache["stagea_contract"],
            tau_end=end_block,
            contract_type_map=type_map,
        )
        if contracts_out.empty or users_out.empty:
            continue

        base = contracts_out.copy()
        n_contracts = int(len(base))
        base["snapshot_idx"] = snapshot_idx
        base["start_block"] = start_block
        base["end_block"] = end_block
        base["n_contracts_snapshot"] = n_contracts
        base["rank_pct"] = pd.to_numeric(base["rank_C"], errors="coerce") / float(max(1, n_contracts))

        eliza_core = build_eliza_user_core(users_out)
        support_df = build_contract_agent_support(df_win, eliza_core)
        adjusted = apply_eliza_feedback(base, support_df)
        adjusted["snapshot_idx"] = snapshot_idx
        adjusted["start_block"] = start_block
        adjusted["end_block"] = end_block
        adjusted["n_contracts_snapshot"] = n_contracts

        core_u_mass = float(pd.to_numeric(eliza_core["uC"], errors="coerce").sum()) if not eliza_core.empty else 0.0
        summary_rows.append(summarize_rugpull(snapshot_idx, start_block, end_block, adjusted))
        eliza_rows.append(
            pd.DataFrame(
                [
                    {
                        "snapshot_idx": snapshot_idx,
                        "start_block": start_block,
                        "end_block": end_block,
                        "eliza_core_user_count": int(len(eliza_core)),
                        "eliza_core_mass_uC": core_u_mass,
                        "eliza_support_contract_count": int(support_df["contract_id"].nunique()) if not support_df.empty else 0,
                        "eliza_support_raw_sum": float(pd.to_numeric(support_df["eliza_support_raw"], errors="coerce").sum()) if not support_df.empty else 0.0,
                    }
                ]
            )
        )
        contract_rows.append(
            adjusted[
                [
                    "snapshot_idx",
                    "start_block",
                    "end_block",
                    "contract_id",
                    "contract_address",
                    "collection_type",
                    "cC",
                    "rank_C",
                    "rank_pct",
                    "cC_base_norm",
                    "eliza_support_raw",
                    "eliza_support_norm",
                    "eliza_core_users_on_contract",
                    "cC_eliza",
                    "rank_C_eliza",
                    "rank_pct_eliza",
                    "rank_delta_eliza",
                    "n_contracts_snapshot",
                ]
            ]
        )

        processed = snapshot_idx
        elapsed = perf_counter() - loop_t0
        avg_per_snapshot = elapsed / float(max(1, processed))
        remain = max(0, total_snapshots - processed)
        eta_left = avg_per_snapshot * float(remain)
        eta_total = avg_per_snapshot * float(total_snapshots)
        if processed == 1:
            progress_meta["eta_total_seconds_after_first_snapshot"] = float(eta_total)
        print(
            "[Progress] "
            f"{processed}/{total_snapshots} "
            f"({processed / float(max(1, total_snapshots)) * 100:.1f}%) "
            f"elapsed={format_seconds(elapsed)} "
            f"eta_left={format_seconds(eta_left)} "
            f"eta_total={format_seconds(eta_total)}"
        )

    contracts_df = pd.concat(contract_rows, ignore_index=True) if contract_rows else pd.DataFrame()
    eliza_df = pd.concat(eliza_rows, ignore_index=True) if eliza_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values("snapshot_idx").reset_index(drop=True) if summary_rows else pd.DataFrame()
    progress_meta["n_snapshots_processed"] = int(summary_df["snapshot_idx"].nunique()) if not summary_df.empty else 0
    progress_meta["snapshot_loop_elapsed_seconds"] = float(perf_counter() - loop_t0)
    return contracts_df, eliza_df, summary_df, progress_meta


def main() -> None:
    t0 = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LABEL_CSV.exists():
        raise FileNotFoundError(f"Label file not found: {LABEL_CSV.resolve()}")

    trp = load_module(TRP_SCRIPT, "trp_mod_for_eliza")
    eval3 = load_module(BASE_EVAL_SCRIPT, "base_eval3_mod_for_eliza")
    label_df = pd.read_csv(LABEL_CSV)

    selected, files, type_map, start_block, end_block, universe_meta = eval3.build_fixed_universe(label_df)
    bps = eval3.estimate_blocks_per_second(label_df)
    step_blocks = int(round(SEVEN_DAYS_SECONDS * bps))
    step_blocks = max(1, step_blocks)
    plan = eval3.build_snapshot_plan(start_block=start_block, end_block=end_block, step_blocks=step_blocks)

    contracts_df, eliza_df, summary_df, progress_meta = run_snapshots(trp=trp, files=files, type_map=type_map, plan=plan)

    selected[
        [
            "contract_address",
            "collection_type",
            "create_block_number",
            "detected_block_number",
            "daily_activity",
            "source_tx_file",
        ]
    ].to_csv(OUTPUT_DIR / "selected_universe.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(OUTPUT_DIR / "snapshot_plan.csv", index=False, encoding="utf-8-sig")
    contracts_df.to_csv(OUTPUT_DIR / "snapshot_contract_scores_eliza.csv", index=False, encoding="utf-8-sig")
    eliza_df.to_csv(OUTPUT_DIR / "snapshot_eliza_agent.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "snapshot_summary_eliza.csv", index=False, encoding="utf-8-sig")

    plot_outputs(summary_df)

    t1 = perf_counter()
    run_meta = pd.DataFrame(
        [
            {"key": "selected_total", "value": universe_meta["selected_total"]},
            {"key": "selected_rugpull", "value": universe_meta["selected_rugpull"]},
            {"key": "selected_unknown", "value": universe_meta["selected_unknown"]},
            {"key": "selected_trust", "value": universe_meta["selected_trust"]},
            {"key": "selected_files", "value": universe_meta["selected_files"]},
            {"key": "start_block", "value": start_block},
            {"key": "end_block", "value": end_block},
            {"key": "blocks_per_second_median", "value": bps},
            {"key": "step_blocks_7d", "value": step_blocks},
            {"key": "top_percent_list", "value": "|".join(str(x) for x in TOP_PERCENT_LIST)},
            {"key": "eliza_top_user_pct", "value": ELIZA_TOP_USER_PCT},
            {"key": "eliza_top_user_min", "value": ELIZA_TOP_USER_MIN},
            {"key": "eliza_top_user_max", "value": ELIZA_TOP_USER_MAX},
            {"key": "eliza_agent_mix", "value": ELIZA_AGENT_MIX},
            {"key": "eliza_event_value_log_scale", "value": ELIZA_EVENT_VALUE_LOG_SCALE},
            {"key": "stagea_reuse", "value": int(1 if STAGEA_REUSE else 0)},
            {"key": "n_snapshots", "value": len(plan)},
            {"key": "n_snapshots_processed", "value": progress_meta["n_snapshots_processed"]},
            {"key": "eta_total_seconds_after_first_snapshot", "value": progress_meta["eta_total_seconds_after_first_snapshot"]},
            {"key": "snapshot_loop_elapsed_seconds", "value": progress_meta["snapshot_loop_elapsed_seconds"]},
            {"key": "elapsed_seconds", "value": round(t1 - t0, 3)},
        ]
    )
    run_meta.to_csv(OUTPUT_DIR / "run_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'snapshot_contract_scores_eliza.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'snapshot_eliza_agent.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'snapshot_summary_eliza.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'run_meta.csv').resolve()}")
    print(f"[Saved] figures in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
