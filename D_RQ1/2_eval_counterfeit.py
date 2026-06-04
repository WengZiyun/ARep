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
# Config (edit here)
# ============================================================
LABEL_CSV = Path("output/C/3_collectionlabel.csv")
TRP_SCRIPT = Path("D_RQ1/0_rugpullTRP.py")
OUTPUT_DIR = Path("output/D/2_eval_counterfeit")

UNKNOWN_TOP_N = 50
TRUST_EARLIEST_N = 20

END_BLOCK_CAP = 19777901
SEVEN_DAYS_SECONDS = 604800

SAVE_INTERMEDIATE_CONTRACTS = False
STAGEA_REUSE = True
TOP_PERCENT_LIST = [0.10, 0.20, 0.30]
FIG_DPI = 140
TARGET_TYPE = "counterfeit"


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


def normalize_label_df(label_df: pd.DataFrame) -> pd.DataFrame:
    df = label_df.copy()
    required = {
        "contract_address",
        "collection_type",
        "source_tx_file",
        "create_block_number",
        "detected_block_number",
        "daily_activity",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in label csv: {missing}")

    df["contract_address"] = df["contract_address"].astype(str).str.lower().str.strip()
    df["collection_type"] = df["collection_type"].astype(str).str.lower().str.strip()
    df["source_tx_file"] = df["source_tx_file"].astype(str).str.strip()
    df["create_block_number"] = pd.to_numeric(df["create_block_number"], errors="coerce")
    df["detected_block_number"] = pd.to_numeric(df["detected_block_number"], errors="coerce")
    df["daily_activity"] = pd.to_numeric(df["daily_activity"], errors="coerce")

    df = df[(df["contract_address"] != "") & (df["source_tx_file"] != "")].copy()
    return df


def estimate_blocks_per_second(label_df: pd.DataFrame) -> float:
    work = label_df.copy()
    work["first_timestamp"] = pd.to_datetime(work["first_timestamp"], errors="coerce", utc=True)
    work["last_timestamp"] = pd.to_datetime(work["last_timestamp"], errors="coerce", utc=True)
    work = work.dropna(subset=["create_block_number", "max_block_number", "first_timestamp", "last_timestamp"]).copy()

    if work.empty:
        raise ValueError("Cannot estimate block speed: no valid rows")

    block_span = (pd.to_numeric(work["max_block_number"], errors="coerce") - pd.to_numeric(work["create_block_number"], errors="coerce")).astype(float)
    sec_span = (work["last_timestamp"] - work["first_timestamp"]).dt.total_seconds().astype(float)
    good = (block_span > 0) & (sec_span > 0)
    work = work[good].copy()
    if work.empty:
        raise ValueError("Cannot estimate block speed: no positive span rows")

    bps = (pd.to_numeric(work["max_block_number"], errors="coerce") - pd.to_numeric(work["create_block_number"], errors="coerce")) / (
        work["last_timestamp"] - work["first_timestamp"]
    ).dt.total_seconds()
    bps = pd.to_numeric(bps, errors="coerce").dropna()
    if bps.empty:
        raise ValueError("Cannot estimate block speed: empty series")
    return float(np.median(bps.values))


def build_fixed_universe(label_df: pd.DataFrame) -> tuple[pd.DataFrame, list[Path], pd.DataFrame, int, int, dict]:
    df = normalize_label_df(label_df)

    target = (
        df[df["collection_type"] == TARGET_TYPE]
        .dropna(subset=["create_block_number", "detected_block_number"]) 
        .copy()
    )
    target = target[target["detected_block_number"] <= int(END_BLOCK_CAP)].copy()
    if target.empty:
        raise ValueError(
            f"No valid {TARGET_TYPE} rows after requiring create/detected blocks and detected<=END_BLOCK_CAP"
        )

    unknown = df[df["collection_type"] == "unknown"].dropna(subset=["daily_activity"]).copy()
    unknown = unknown.sort_values(["daily_activity", "create_block_number"], ascending=[False, True], kind="mergesort").head(UNKNOWN_TOP_N)

    trust = df[df["collection_type"] == "trust"].dropna(subset=["create_block_number"]).copy()
    trust = trust.sort_values(["create_block_number", "daily_activity"], ascending=[True, False], kind="mergesort").head(TRUST_EARLIEST_N)

    selected = pd.concat([target, unknown, trust], ignore_index=True)
    selected = selected.sort_values(["contract_address", "source_tx_file"], kind="mergesort")
    selected = selected.drop_duplicates(subset=["contract_address"], keep="first").reset_index(drop=True)

    start_block = int(pd.to_numeric(target["create_block_number"], errors="coerce").min())
    end_block = int(pd.to_numeric(target["detected_block_number"], errors="coerce").max())
    end_block = min(end_block, int(END_BLOCK_CAP))

    files = []
    for p in selected["source_tx_file"].tolist():
        fp = Path(p)
        if fp.exists():
            files.append(fp)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No existing source files in selected universe")

    type_map = selected[["contract_address", "collection_type"]].copy()

    meta = {
        "selected_total": int(len(selected)),
        f"selected_{TARGET_TYPE}": int((selected["collection_type"] == TARGET_TYPE).sum()),
        "selected_unknown": int((selected["collection_type"] == "unknown").sum()),
        "selected_trust": int((selected["collection_type"] == "trust").sum()),
        "selected_files": int(len(files)),
        "start_block": start_block,
        "end_block": end_block,
    }
    return selected, files, type_map, start_block, end_block, meta


def build_snapshot_plan(start_block: int, end_block: int, step_blocks: int) -> pd.DataFrame:
    if step_blocks <= 0:
        raise ValueError("step_blocks must be positive")
    ends = []
    cur = int(start_block) + int(step_blocks)
    while cur < int(end_block):
        ends.append(int(cur))
        cur += int(step_blocks)
    if not ends or ends[-1] != int(end_block):
        ends.append(int(end_block))

    plan = pd.DataFrame(
        {
            "snapshot_idx": np.arange(1, len(ends) + 1, dtype=np.int64),
            "start_block": int(start_block),
            "end_block": ends,
            "step_blocks_7d": int(step_blocks),
        }
    )
    return plan


def run_snapshots(trp, files: list[Path], type_map: pd.DataFrame, plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df_all = trp.load_events_from_files(files)
    if df_all.empty:
        raise ValueError("No events loaded from selected files")

    stagea_cache = None
    if STAGEA_REUSE:
        trp.STAGEA_REFRESH_HOURS = None
        trp.STAGEA_REFRESH_BLOCKS = None
        trp.FORCE_STAGEA_REBUILD = False

    scores_rows = []
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
                f"eta_total={format_seconds(eta_total)} "
                f"(empty window skipped)"
            )
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
        contracts_out, _users_out, _meta = trp.run_stage_c(
            df_window=df_win,
            u0_full=u0_full,
            stagea_contract=stagea_cache["stagea_contract"],
            tau_end=end_block,
            contract_type_map=type_map,
        )
        if contracts_out.empty:
            continue

        n_total = int(len(contracts_out))
        out = contracts_out.copy()
        out["collection_type"] = out["collection_type"].fillna("unknown").astype(str).str.lower()
        out["snapshot_idx"] = snapshot_idx
        out["start_block"] = start_block
        out["end_block"] = end_block
        out["n_contracts_snapshot"] = n_total
        out["rank_pct"] = out["rank_C"] / float(max(1, n_total))

        scores_rows.append(
            out[
                [
                    "snapshot_idx",
                    "start_block",
                    "end_block",
                    "contract_id",
                    "contract_address",
                    "collection_type",
                    "cC",
                    "rank_C",
                    "n_contracts_snapshot",
                    "rank_pct",
                ]
            ]
        )

        tgt = out[out["collection_type"] == TARGET_TYPE].copy()
        if tgt.empty:
            summary_rows.append(
                {
                    "snapshot_idx": snapshot_idx,
                    "start_block": start_block,
                    "end_block": end_block,
                    f"{TARGET_TYPE}_count": 0,
                    "rank_mean": np.nan,
                    "rank_median": np.nan,
                    "rank_p25": np.nan,
                    "rank_p75": np.nan,
                    f"{TARGET_TYPE}_in_top10pct_rate": np.nan,
                    f"{TARGET_TYPE}_in_top20pct_rate": np.nan,
                    f"{TARGET_TYPE}_in_top30pct_rate": np.nan,
                }
            )
        else:
            summary_rows.append(
                {
                    "snapshot_idx": snapshot_idx,
                    "start_block": start_block,
                    "end_block": end_block,
                    f"{TARGET_TYPE}_count": int(len(tgt)),
                    "rank_mean": float(tgt["rank_C"].mean()),
                    "rank_median": float(tgt["rank_C"].median()),
                    "rank_p25": float(tgt["rank_C"].quantile(0.25)),
                    "rank_p75": float(tgt["rank_C"].quantile(0.75)),
                    f"{TARGET_TYPE}_in_top10pct_rate": float((tgt["rank_pct"] <= TOP_PERCENT_LIST[0]).mean()),
                    f"{TARGET_TYPE}_in_top20pct_rate": float((tgt["rank_pct"] <= TOP_PERCENT_LIST[1]).mean()),
                    f"{TARGET_TYPE}_in_top30pct_rate": float((tgt["rank_pct"] <= TOP_PERCENT_LIST[2]).mean()),
                }
            )

        if SAVE_INTERMEDIATE_CONTRACTS:
            (OUTPUT_DIR / "intermediate").mkdir(parents=True, exist_ok=True)
            out.to_csv(OUTPUT_DIR / "intermediate" / f"contracts_snapshot_{snapshot_idx:03d}_{end_block}.csv", index=False, encoding="utf-8-sig")

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

    if scores_rows:
        scores = pd.concat(scores_rows, ignore_index=True)
    else:
        scores = pd.DataFrame(
            columns=[
                "snapshot_idx",
                "start_block",
                "end_block",
                "contract_id",
                "contract_address",
                "collection_type",
                "cC",
                "rank_C",
                "n_contracts_snapshot",
                "rank_pct",
            ]
        )

    target_only = scores[scores["collection_type"] == TARGET_TYPE].copy() if not scores.empty else pd.DataFrame(columns=scores.columns)
    summary = pd.DataFrame(summary_rows).sort_values("snapshot_idx").reset_index(drop=True) if summary_rows else pd.DataFrame(
        columns=[
            "snapshot_idx",
            "start_block",
            "end_block",
            f"{TARGET_TYPE}_count",
            "rank_mean",
            "rank_median",
            "rank_p25",
            "rank_p75",
            f"{TARGET_TYPE}_in_top10pct_rate",
            f"{TARGET_TYPE}_in_top20pct_rate",
            f"{TARGET_TYPE}_in_top30pct_rate",
        ]
    )

    progress_meta["n_snapshots_processed"] = int(summary["snapshot_idx"].nunique()) if not summary.empty else 0
    progress_meta["snapshot_loop_elapsed_seconds"] = float(perf_counter() - loop_t0)
    return scores, target_only, summary, progress_meta


def plot_outputs(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    x = summary["snapshot_idx"]

    fig1, ax1 = plt.subplots(figsize=(8.8, 4.8))
    ax1.plot(x, summary["rank_median"], linewidth=2.0, color="#0b6")
    ax1.set_title(f"{TARGET_TYPE.capitalize()} Median Rank Over 7-Day Snapshots")
    ax1.set_xlabel("Snapshot Index")
    ax1.set_ylabel("Median rank_C")
    ax1.grid(alpha=0.25)
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / f"{TARGET_TYPE}_rank_median_over_snapshots.png", dpi=FIG_DPI)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.8, 4.8))
    ax2.plot(x, summary["rank_median"], label="P50", linewidth=2.0)
    ax2.fill_between(x, summary["rank_p25"], summary["rank_p75"], alpha=0.25, label="P25-P75")
    ax2.set_title(f"{TARGET_TYPE.capitalize()} Rank Quantiles Over 7-Day Snapshots")
    ax2.set_xlabel("Snapshot Index")
    ax2.set_ylabel("rank_C")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / f"{TARGET_TYPE}_rank_quantiles_over_snapshots.png", dpi=FIG_DPI)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8.8, 4.8))
    ax3.plot(x, summary[f"{TARGET_TYPE}_in_top10pct_rate"], label="Top10%", linewidth=1.9)
    ax3.plot(x, summary[f"{TARGET_TYPE}_in_top20pct_rate"], label="Top20%", linewidth=1.9)
    ax3.plot(x, summary[f"{TARGET_TYPE}_in_top30pct_rate"], label="Top30%", linewidth=1.9)
    ax3.set_title(f"{TARGET_TYPE.capitalize()} In Top-% Rate Over 7-Day Snapshots")
    ax3.set_xlabel("Snapshot Index")
    ax3.set_ylabel("Rate")
    ax3.set_ylim(0, 1)
    ax3.grid(alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / f"{TARGET_TYPE}_top_percent_rate_over_snapshots.png", dpi=FIG_DPI)
    plt.close(fig3)


def main() -> None:
    t0 = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LABEL_CSV.exists():
        raise FileNotFoundError(f"Label file not found: {LABEL_CSV.resolve()}")

    trp = load_module(TRP_SCRIPT, "trp_mod_for_counterfeit_eval")
    label_df = pd.read_csv(LABEL_CSV)

    selected, files, type_map, start_block, end_block, universe_meta = build_fixed_universe(label_df)

    bps = estimate_blocks_per_second(label_df)
    step_blocks = int(round(SEVEN_DAYS_SECONDS * bps))
    step_blocks = max(1, step_blocks)

    plan = build_snapshot_plan(start_block=start_block, end_block=end_block, step_blocks=step_blocks)

    scores, target_only, summary, progress_meta = run_snapshots(trp=trp, files=files, type_map=type_map, plan=plan)

    selected_out = selected[
        [
            "contract_address",
            "collection_type",
            "create_block_number",
            "detected_block_number",
            "daily_activity",
            "source_tx_file",
        ]
    ].copy()
    selected_out.to_csv(OUTPUT_DIR / "selected_universe.csv", index=False, encoding="utf-8-sig")

    selected_meta = pd.DataFrame(
        [
            {"key": "selected_total", "value": universe_meta["selected_total"]},
            {"key": f"selected_{TARGET_TYPE}", "value": universe_meta[f"selected_{TARGET_TYPE}"]},
            {"key": "selected_unknown", "value": universe_meta["selected_unknown"]},
            {"key": "selected_trust", "value": universe_meta["selected_trust"]},
            {"key": "selected_files", "value": universe_meta["selected_files"]},
            {"key": "start_block", "value": universe_meta["start_block"]},
            {"key": "end_block", "value": universe_meta["end_block"]},
            {"key": "end_block_cap", "value": END_BLOCK_CAP},
            {"key": "unknown_top_n", "value": UNKNOWN_TOP_N},
            {"key": "trust_earliest_n", "value": TRUST_EARLIEST_N},
        ]
    )
    selected_meta.to_csv(OUTPUT_DIR / "selected_universe_meta.csv", index=False, encoding="utf-8-sig")

    plan.to_csv(OUTPUT_DIR / "snapshot_plan.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(OUTPUT_DIR / "snapshot_contract_scores.csv", index=False, encoding="utf-8-sig")
    target_only.to_csv(OUTPUT_DIR / f"snapshot_{TARGET_TYPE}_only.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "snapshot_summary.csv", index=False, encoding="utf-8-sig")

    plot_outputs(summary)

    t1 = perf_counter()
    run_meta = pd.DataFrame(
        [
            {"key": "target_type", "value": TARGET_TYPE},
            {"key": "end_block_cap", "value": END_BLOCK_CAP},
            {"key": "seven_days_seconds", "value": SEVEN_DAYS_SECONDS},
            {"key": "blocks_per_second_median", "value": bps},
            {"key": "step_blocks_7d", "value": step_blocks},
            {"key": "start_block", "value": start_block},
            {"key": "end_block", "value": end_block},
            {"key": "n_snapshots", "value": len(plan)},
            {"key": "n_snapshots_processed", "value": progress_meta["n_snapshots_processed"]},
            {"key": "n_snapshot_contract_rows", "value": len(scores)},
            {"key": f"n_snapshot_{TARGET_TYPE}_rows", "value": len(target_only)},
            {"key": "save_intermediate_contracts", "value": int(1 if SAVE_INTERMEDIATE_CONTRACTS else 0)},
            {"key": "stagea_reuse", "value": int(1 if STAGEA_REUSE else 0)},
            {"key": "top_percent_list", "value": "|".join(str(x) for x in TOP_PERCENT_LIST)},
            {"key": "eta_total_seconds_after_first_snapshot", "value": progress_meta["eta_total_seconds_after_first_snapshot"]},
            {"key": "snapshot_loop_elapsed_seconds", "value": progress_meta["snapshot_loop_elapsed_seconds"]},
            {"key": "elapsed_seconds", "value": round(t1 - t0, 3)},
        ]
    )
    run_meta.to_csv(OUTPUT_DIR / "run_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'selected_universe.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'snapshot_plan.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'snapshot_contract_scores.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / f'snapshot_{TARGET_TYPE}_only.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'snapshot_summary.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'run_meta.csv').resolve()}")
    print(f"[Saved] figures in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
