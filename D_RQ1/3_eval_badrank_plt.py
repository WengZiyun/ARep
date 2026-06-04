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
TRP_SCRIPT = Path("D_RQ1/0_rugpullTRP1.py")
SNAPSHOT_USER_SCORES_CSV = Path("output/D/3_eval_wash2/snapshot_user_scores.csv")
SNAPSHOT_PLAN_CSV = Path("output/D/3_eval_wash2/snapshot_plan.csv")
SNAPSHOT_RUN_META_CSV = Path("output/D/3_eval_wash2/run_meta.csv")

AUTHOR_TRACE_DIR = Path("washtarding/nft_ownership_traces")
INPUT_DIR_MAIN = Path("output/B/2_tranding")
INPUT_DIR_NORMAL = Path("output/B/2_4normalTranding")
OUTPUT_DIR = Path("output/D/3_eval_badrank_plt")

MAIN_TOP_N = 50
NORMAL_TOP_N = 50
CHUNK_SIZE = 400_000
FIG_DPI = 140
BLOCKS_PER_DAY = 7200
SHORT_HOLD_BLOCKS = BLOCKS_PER_DAY
RATE_SCALE_BLOCKS = 10_000.0

RANK_BIN_EDGES = [i / 10.0 for i in range(11)]
RANK_BIN_LABELS = [f"[{i/10:.1f},{(i+1)/10:.1f}{')' if i < 9 else ']'}" for i in range(10)]


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_author_end_block() -> int:
    files = sorted(AUTHOR_TRACE_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv found in {AUTHOR_TRACE_DIR.resolve()}")
    mx = -1
    for fp in files:
        for chunk in pd.read_csv(fp, usecols=["block_no"], chunksize=CHUNK_SIZE, low_memory=False):
            b = pd.to_numeric(chunk["block_no"], errors="coerce").dropna()
            if not b.empty:
                cur = int(b.max())
                if cur > mx:
                    mx = cur
    if mx < 0:
        raise ValueError("Failed to infer end_block from author trace dataset")
    return int(mx)


def file_stats_until_end(csv_path: Path, end_block: int) -> tuple[int, int]:
    cnt = 0
    min_block = None
    for chunk in pd.read_csv(csv_path, usecols=["block_number"], chunksize=CHUNK_SIZE, low_memory=False):
        b = pd.to_numeric(chunk["block_number"], errors="coerce")
        valid = b[b <= int(end_block)].dropna()
        if valid.empty:
            continue
        cnt += int(len(valid))
        cur_min = int(valid.min())
        if min_block is None or cur_min < min_block:
            min_block = cur_min
    if min_block is None:
        min_block = 2**31 - 1
    return cnt, int(min_block)


def select_input_files_same_as_wash2(end_block: int) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    files_main = sorted(INPUT_DIR_MAIN.glob("*.csv"))
    if not files_main:
        raise FileNotFoundError(f"No csv found in {INPUT_DIR_MAIN.resolve()}")

    files_normal = sorted(INPUT_DIR_NORMAL.glob("*.csv"))
    if not files_normal:
        raise FileNotFoundError(f"No csv found in {INPUT_DIR_NORMAL.resolve()}")

    rows_main = []
    for fp in files_main:
        act, create_blk = file_stats_until_end(fp, end_block=end_block)
        rows_main.append(
            {
                "file": str(fp),
                "file_name": fp.name,
                "activity_before_end_block": int(act),
                "create_block_before_end_block": int(create_blk),
            }
        )
    main_df = pd.DataFrame(rows_main).sort_values(
        ["activity_before_end_block", "create_block_before_end_block", "file_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    main_top_df = main_df.head(int(MAIN_TOP_N)).copy()
    main_files = [Path(p) for p in main_top_df["file"].tolist()]

    rows_normal = []
    for fp in files_normal:
        act, _ = file_stats_until_end(fp, end_block=end_block)
        rows_normal.append(
            {
                "file": str(fp),
                "file_name": fp.name,
                "activity_before_end_block": int(act),
            }
        )
    normal_df = pd.DataFrame(rows_normal).sort_values(
        ["activity_before_end_block", "file_name"], ascending=[False, True]
    ).reset_index(drop=True)
    normal_top_df = normal_df.head(int(NORMAL_TOP_N)).copy()
    normal_files = [Path(p) for p in normal_top_df["file"].tolist()]

    files = sorted(set(main_files + normal_files))
    return files, main_top_df, normal_top_df


def build_user_rank_bins(snapshot_user_scores_df: pd.DataFrame) -> pd.DataFrame:
    df = snapshot_user_scores_df.copy()
    need = ["snapshot_idx", "start_block", "end_block", "user", "rank_pct", "n_users_snapshot"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in snapshot_user_scores: {miss}")

    df["snapshot_idx"] = pd.to_numeric(df["snapshot_idx"], errors="coerce")
    df["start_block"] = pd.to_numeric(df["start_block"], errors="coerce")
    df["end_block"] = pd.to_numeric(df["end_block"], errors="coerce")
    df["rank_pct"] = pd.to_numeric(df["rank_pct"], errors="coerce")
    df["n_users_snapshot"] = pd.to_numeric(df["n_users_snapshot"], errors="coerce")
    df["user"] = df["user"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["snapshot_idx", "start_block", "end_block", "rank_pct", "n_users_snapshot"]).copy()

    df["snapshot_idx"] = df["snapshot_idx"].astype(int)
    df["start_block"] = df["start_block"].astype(int)
    df["end_block"] = df["end_block"].astype(int)

    # Left-closed right-open for first 9 bins, last bin right-closed.
    bins = pd.cut(df["rank_pct"], bins=RANK_BIN_EDGES, labels=RANK_BIN_LABELS, include_lowest=True, right=True)
    df["rank_bin"] = bins.astype(str)

    # Keep one row per (snapshot, user)
    df = df.sort_values(["snapshot_idx", "user", "rank_pct"], kind="mergesort").drop_duplicates(["snapshot_idx", "user"], keep="first")
    return df[["snapshot_idx", "start_block", "end_block", "user", "rank_pct", "rank_bin", "n_users_snapshot"]]


def build_all_holding_periods(df_all: pd.DataFrame, start_block: int, max_end_block: int) -> pd.DataFrame:
    req = ["contract_id", "token_id", "buyer", "seller", "block_number", "tx_index_in_block"]
    miss = [c for c in req if c not in df_all.columns]
    if miss:
        raise ValueError(f"Missing columns in events for holding calc: {miss}")

    work = df_all.copy()
    work["block_number"] = pd.to_numeric(work["block_number"], errors="coerce")
    work["tx_index_in_block"] = pd.to_numeric(work["tx_index_in_block"], errors="coerce").fillna(0)
    work = work.dropna(subset=["block_number"]).copy()
    work["block_number"] = work["block_number"].astype(np.int64)

    work = work[(work["block_number"] >= int(start_block)) & (work["block_number"] <= int(max_end_block))].copy()
    work["buyer"] = work["buyer"].astype(str).str.lower().str.strip()
    work["seller"] = work["seller"].astype(str).str.lower().str.strip()
    work["contract_id"] = work["contract_id"].astype(str)
    work["token_id"] = work["token_id"].astype(str)

    work = work.sort_values(["contract_id", "token_id", "block_number", "tx_index_in_block"], kind="mergesort")

    rows = []
    for (_cid, _tid), g in work.groupby(["contract_id", "token_id"], sort=False):
        hold_buy_block: dict[str, int] = {}
        for r in g[["seller", "buyer", "block_number"]].itertuples(index=False):
            seller = str(r.seller)
            buyer = str(r.buyer)
            blk = int(r.block_number)

            if seller in hold_buy_block:
                buy_blk = hold_buy_block.pop(seller)
                d = blk - int(buy_blk)
                if d >= 0:
                    rows.append(
                        {
                            "user": seller,
                            "buy_block": int(buy_blk),
                            "sell_block": int(blk),
                            "holding_blocks": int(d),
                            "is_short_hold": int(1 if d <= SHORT_HOLD_BLOCKS else 0),
                        }
                    )

            hold_buy_block[buyer] = int(blk)

    if not rows:
        return pd.DataFrame(columns=["user", "buy_block", "sell_block", "holding_blocks", "is_short_hold"])
    return pd.DataFrame(rows)


def build_user_event_appearances(df_all: pd.DataFrame, start_block: int, max_end_block: int) -> pd.DataFrame:
    req = ["buyer", "seller", "block_number"]
    miss = [c for c in req if c not in df_all.columns]
    if miss:
        raise ValueError(f"Missing columns in events for user appearance calc: {miss}")

    work = df_all.copy()
    work["block_number"] = pd.to_numeric(work["block_number"], errors="coerce")
    work = work.dropna(subset=["block_number"]).copy()
    work["block_number"] = work["block_number"].astype(np.int64)
    work = work[(work["block_number"] >= int(start_block)) & (work["block_number"] <= int(max_end_block))].copy()
    work["buyer"] = work["buyer"].astype(str).str.lower().str.strip()
    work["seller"] = work["seller"].astype(str).str.lower().str.strip()

    buyers = work[["buyer", "block_number"]].rename(columns={"buyer": "user"})
    sellers = work[["seller", "block_number"]].rename(columns={"seller": "user"})
    out = pd.concat([buyers, sellers], ignore_index=True)
    out = out[out["user"] != ""].copy()
    return out.sort_values(["block_number", "user"], kind="mergesort").reset_index(drop=True)


def compute_user_holding_blocks(periods_df: pd.DataFrame, appearances_df: pd.DataFrame, start_block: int, end_block_t: int) -> pd.DataFrame:
    if periods_df.empty:
        return pd.DataFrame(
            columns=[
                "user",
                "holding_blocks_user_median",
                "holding_obs_count",
                "short_hold_ratio",
                "holds_per_10k_blocks",
                "events_per_10k_blocks",
                "speculator_score",
            ]
        )

    sub = periods_df[periods_df["sell_block"] <= int(end_block_t)].copy()
    app = appearances_df[appearances_df["block_number"] <= int(end_block_t)].copy() if not appearances_df.empty else pd.DataFrame(columns=["user"])
    if sub.empty and app.empty:
        return pd.DataFrame(
            columns=[
                "user",
                "holding_blocks_user_median",
                "holding_obs_count",
                "short_hold_ratio",
                "holds_per_10k_blocks",
                "events_per_10k_blocks",
                "speculator_score",
            ]
        )

    span_blocks = max(1.0, float(int(end_block_t) - int(start_block)))

    hold_out = (
        sub.groupby("user", as_index=False)
        .agg(
            holding_blocks_user_median=("holding_blocks", "median"),
            holding_obs_count=("holding_blocks", "count"),
            short_hold_ratio=("is_short_hold", "mean"),
        )
    )
    event_out = app.groupby("user", as_index=False).agg(user_event_count=("block_number", "count")) if not app.empty else pd.DataFrame(columns=["user", "user_event_count"])

    out = hold_out.merge(event_out, on="user", how="left")
    out["user_event_count"] = pd.to_numeric(out["user_event_count"], errors="coerce").fillna(0).astype(int)
    out["short_hold_ratio"] = pd.to_numeric(out["short_hold_ratio"], errors="coerce").fillna(0.0)
    out["holds_per_10k_blocks"] = out["holding_obs_count"].astype(float) / span_blocks * RATE_SCALE_BLOCKS
    out["events_per_10k_blocks"] = out["user_event_count"].astype(float) / span_blocks * RATE_SCALE_BLOCKS
    out["speculator_score"] = (
        np.log1p(out["events_per_10k_blocks"].astype(float))
        * (1.0 + 2.0 * out["short_hold_ratio"].astype(float))
        / np.log1p(out["holding_blocks_user_median"].astype(float) + 1.0)
    )
    out = out.sort_values(["holding_obs_count", "holding_blocks_user_median", "user"], ascending=[False, True, True])
    return out[
        [
            "user",
            "holding_blocks_user_median",
            "holding_obs_count",
            "short_hold_ratio",
            "holds_per_10k_blocks",
            "events_per_10k_blocks",
            "speculator_score",
        ]
    ]


def merge_bin_and_holding(user_bin_df: pd.DataFrame, user_holding_df: pd.DataFrame, snapshot_idx: int, end_block_t: int) -> pd.DataFrame:
    cur_bin = user_bin_df[user_bin_df["snapshot_idx"] == int(snapshot_idx)].copy()
    if cur_bin.empty:
        return pd.DataFrame(
            columns=[
                "snapshot_idx",
                "end_block",
                "rank_bin",
                "user",
                "holding_blocks_user_median",
                "holding_obs_count",
                "short_hold_ratio",
                "holds_per_10k_blocks",
                "events_per_10k_blocks",
                "speculator_score",
            ]
        )

    merged = cur_bin.merge(user_holding_df, on="user", how="inner")
    # Remove users without complete holding cycles in this window.
    merged = merged[merged["holding_obs_count"] > 0].copy()
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "snapshot_idx",
                "end_block",
                "rank_bin",
                "user",
                "holding_blocks_user_median",
                "holding_obs_count",
                "short_hold_ratio",
                "holds_per_10k_blocks",
                "events_per_10k_blocks",
                "speculator_score",
            ]
        )

    merged["snapshot_idx"] = int(snapshot_idx)
    merged["end_block"] = int(end_block_t)
    return merged[
        [
            "snapshot_idx",
            "end_block",
            "rank_bin",
            "user",
            "holding_blocks_user_median",
            "holding_obs_count",
            "short_hold_ratio",
            "holds_per_10k_blocks",
            "events_per_10k_blocks",
            "speculator_score",
        ]
    ]


def summarize_for_boxplot(window_level_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if window_level_df.empty:
        return pd.DataFrame(columns=["snapshot_idx", "end_block", "rank_bin", "metric", "n_users", "box_q1", "box_median", "box_q3", "whisker_low", "whisker_high"])

    rows = []
    for r in window_level_df.groupby(["snapshot_idx", "end_block", "rank_bin"], sort=True):
        (snap, endb, rank_bin), g = r
        x = pd.to_numeric(g[metric_col], errors="coerce").dropna()
        if x.empty:
            continue
        q1 = float(x.quantile(0.25))
        q2 = float(x.quantile(0.50))
        q3 = float(x.quantile(0.75))
        iqr = q3 - q1
        low = float(max(x.min(), q1 - 1.5 * iqr))
        high = float(min(x.max(), q3 + 1.5 * iqr))
        rows.append(
            {
                "snapshot_idx": int(snap),
                "end_block": int(endb),
                "rank_bin": str(rank_bin),
                "metric": metric_col,
                "n_users": int(len(x)),
                "box_q1": q1,
                "box_median": q2,
                "box_q3": q3,
                "whisker_low": low,
                "whisker_high": high,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["snapshot_idx", "rank_bin"], kind="mergesort").reset_index(drop=True)


def plot_boxplot(window_level_df: pd.DataFrame, metric_col: str, out_png: Path, y_label: str, use_log: bool) -> None:
    if window_level_df.empty:
        return

    data = []
    for b in RANK_BIN_LABELS:
        x = pd.to_numeric(
            window_level_df[window_level_df["rank_bin"] == b][metric_col],
            errors="coerce",
        ).dropna()
        data.append(x.values if len(x) > 0 else np.array([np.nan]))

    plt.figure(figsize=(11.5, 5.5))
    plt.boxplot(data, labels=RANK_BIN_LABELS, showfliers=False)
    if use_log:
        plt.yscale("log")
    plt.title(f"{metric_col} by Rank Bin (All Snapshots Combined)")
    plt.xlabel("Rank Bin (higher bin = worse rank)")
    plt.ylabel(y_label)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI)
    plt.close()


def main() -> None:
    t0 = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_exists(SNAPSHOT_USER_SCORES_CSV)
    ensure_exists(SNAPSHOT_PLAN_CSV)
    ensure_exists(SNAPSHOT_RUN_META_CSV)
    ensure_exists(TRP_SCRIPT)

    snapshot_users = pd.read_csv(SNAPSHOT_USER_SCORES_CSV, low_memory=False)
    snapshot_plan = pd.read_csv(SNAPSHOT_PLAN_CSV, low_memory=False)
    run_meta_prev = pd.read_csv(SNAPSHOT_RUN_META_CSV, low_memory=False)

    user_bin_df = build_user_rank_bins(snapshot_users)

    # Validate S consistency from plan.
    if "start_block" not in snapshot_plan.columns or "end_block" not in snapshot_plan.columns or "snapshot_idx" not in snapshot_plan.columns:
        raise ValueError("snapshot_plan.csv missing required columns: snapshot_idx/start_block/end_block")
    snapshot_plan = snapshot_plan.copy()
    snapshot_plan["snapshot_idx"] = pd.to_numeric(snapshot_plan["snapshot_idx"], errors="coerce")
    snapshot_plan["start_block"] = pd.to_numeric(snapshot_plan["start_block"], errors="coerce")
    snapshot_plan["end_block"] = pd.to_numeric(snapshot_plan["end_block"], errors="coerce")
    snapshot_plan = snapshot_plan.dropna(subset=["snapshot_idx", "start_block", "end_block"]).copy()
    snapshot_plan["snapshot_idx"] = snapshot_plan["snapshot_idx"].astype(int)
    snapshot_plan["start_block"] = snapshot_plan["start_block"].astype(int)
    snapshot_plan["end_block"] = snapshot_plan["end_block"].astype(int)
    snapshot_plan = snapshot_plan.sort_values("snapshot_idx", kind="mergesort")

    s_unique = snapshot_plan["start_block"].unique().tolist()
    if len(s_unique) != 1:
        raise ValueError("snapshot_plan start_block is not unique across snapshots; cannot use fixed S")
    S = int(s_unique[0])

    # Reuse same input selection as 3_eval_wash2.
    author_end_block = get_author_end_block()
    files, main_top50_df, normal_top50_df = select_input_files_same_as_wash2(end_block=author_end_block)

    trp = load_module(TRP_SCRIPT, "trp_mod_badrank")
    df_all = trp.load_events_from_files(files)
    df_all = df_all[(pd.to_numeric(df_all["block_number"], errors="coerce") >= S) & (pd.to_numeric(df_all["block_number"], errors="coerce") <= int(snapshot_plan["end_block"].max()))].copy()

    periods_df = build_all_holding_periods(df_all=df_all, start_block=S, max_end_block=int(snapshot_plan["end_block"].max()))
    appearances_df = build_user_event_appearances(df_all=df_all, start_block=S, max_end_block=int(snapshot_plan["end_block"].max()))

    # Build window-level result for each snapshot using cumulative [S, end_block_t].
    window_rows = []
    for r in snapshot_plan.itertuples(index=False):
        snap = int(r.snapshot_idx)
        endb = int(r.end_block)
        uh = compute_user_holding_blocks(periods_df=periods_df, appearances_df=appearances_df, start_block=S, end_block_t=endb)
        merged = merge_bin_and_holding(user_bin_df=user_bin_df, user_holding_df=uh, snapshot_idx=snap, end_block_t=endb)
        if not merged.empty:
            window_rows.append(merged)

    window_level_df = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame(
        columns=[
            "snapshot_idx",
            "end_block",
            "rank_bin",
            "user",
            "holding_blocks_user_median",
            "holding_obs_count",
            "short_hold_ratio",
            "holds_per_10k_blocks",
            "events_per_10k_blocks",
            "speculator_score",
        ]
    )

    metric_specs = [
        ("holding_blocks_user_median", "Median Holding Blocks per User (log scale)", True, "holding_blocks_boxplot_by_rankbin.png"),
        ("short_hold_ratio", "Short-Hold Ratio (<=1 day)", False, "short_hold_ratio_boxplot_by_rankbin.png"),
        ("holds_per_10k_blocks", "Completed Holds per 10k Blocks", False, "holds_per_10k_blocks_boxplot_by_rankbin.png"),
        ("events_per_10k_blocks", "Events per 10k Blocks", False, "events_per_10k_blocks_boxplot_by_rankbin.png"),
        ("speculator_score", "Speculator Score", False, "speculator_score_boxplot_by_rankbin.png"),
    ]

    summary_parts = []
    for metric_col, y_label, use_log, fig_name in metric_specs:
        summary_parts.append(summarize_for_boxplot(window_level_df, metric_col=metric_col))
        plot_boxplot(window_level_df, metric_col=metric_col, out_png=OUTPUT_DIR / fig_name, y_label=y_label, use_log=use_log)
    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()

    # Save outputs
    window_fp = OUTPUT_DIR / "rankbin_holding_window_level.csv"
    summary_fp = OUTPUT_DIR / "rankbin_holding_summary.csv"
    meta_fp = OUTPUT_DIR / "run_meta.csv"

    window_level_df.to_csv(window_fp, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_fp, index=False, encoding="utf-8-sig")

    # Quality checks for acceptance metrics.
    n_bins_found = int(window_level_df["rank_bin"].nunique()) if not window_level_df.empty else 0
    start_block_consistent = int(1 if len(s_unique) == 1 else 0)
    check_snapshot_user = (
        window_level_df.groupby("snapshot_idx", as_index=False)["user"].nunique().rename(columns={"user": "n_users_in_bins"})
        if not window_level_df.empty
        else pd.DataFrame(columns=["snapshot_idx", "n_users_in_bins"])
    )
    base_snapshot_users = user_bin_df.groupby("snapshot_idx", as_index=False)["user"].nunique().rename(columns={"user": "n_users_snapshot"})
    if not check_snapshot_user.empty:
        merged_chk = check_snapshot_user.merge(base_snapshot_users, on="snapshot_idx", how="left")
        no_overflow = int((merged_chk["n_users_in_bins"] <= merged_chk["n_users_snapshot"]).all())
    else:
        no_overflow = 1

    t1 = perf_counter()
    run_meta = pd.DataFrame(
        [
            {"key": "rank_bin_mode", "value": "exclusive_10_bins"},
            {"key": "rank_bin_labels", "value": "|".join(RANK_BIN_LABELS)},
            {"key": "window_mode", "value": "cumulative_S_to_end_block"},
            {"key": "y_metric", "value": "median_holding_blocks"},
            {"key": "holding_definition", "value": "buy_block_to_next_sell_block"},
            {"key": "population_scope", "value": "all_users"},
            {"key": "start_block_S", "value": int(S)},
            {"key": "author_end_block_for_selection", "value": int(author_end_block)},
            {"key": "n_input_files_total", "value": int(len(files))},
            {"key": "n_input_files_main_top50", "value": int(len(main_top50_df))},
            {"key": "n_input_files_normal_top50", "value": int(len(normal_top50_df))},
            {"key": "n_snapshot_rows_input", "value": int(len(user_bin_df))},
            {"key": "n_snapshots_plan", "value": int(snapshot_plan["snapshot_idx"].nunique())},
            {"key": "n_holding_period_rows", "value": int(len(periods_df))},
            {"key": "n_user_appearance_rows", "value": int(len(appearances_df))},
            {"key": "n_window_level_rows", "value": int(len(window_level_df))},
            {"key": "n_summary_rows", "value": int(len(summary_df))},
            {"key": "short_hold_blocks_threshold", "value": int(SHORT_HOLD_BLOCKS)},
            {"key": "rate_scale_blocks", "value": RATE_SCALE_BLOCKS},
            {"key": "extra_metrics", "value": "short_hold_ratio|holds_per_10k_blocks|events_per_10k_blocks|speculator_score"},
            {"key": "accept_bins_found", "value": n_bins_found},
            {"key": "accept_start_block_consistent", "value": start_block_consistent},
            {"key": "accept_n_users_in_bins_no_overflow", "value": int(no_overflow)},
            {"key": "input_snapshot_run_meta_exists", "value": int(1 if not run_meta_prev.empty else 0)},
            {"key": "elapsed_seconds", "value": round(t1 - t0, 3)},
        ]
    )
    run_meta.to_csv(meta_fp, index=False, encoding="utf-8-sig")

    print(f"[Saved] {window_fp.resolve()}")
    print(f"[Saved] {summary_fp.resolve()}")
    for _metric_col, _y_label, _use_log, fig_name in metric_specs:
        print(f"[Saved] {(OUTPUT_DIR / fig_name).resolve()}")
    print(f"[Saved] {meta_fp.resolve()}")


if __name__ == "__main__":
    main()
