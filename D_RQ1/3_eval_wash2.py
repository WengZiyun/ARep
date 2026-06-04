#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import importlib.util
from time import perf_counter

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================
TRP_SCRIPT = Path("D_RQ1/0_rugpullTRP1.py")
WASH_LABEL_GLOBAL_CSV = Path("output/B/5_2washcheck/wash_addresses_label_verified_global.csv")
AUTHOR_TRACE_DIR = Path("washtarding/nft_ownership_traces")
INPUT_DIR_MAIN = Path("output/B/2_tranding")
INPUT_DIR_NORMAL = Path("output/B/2_4normalTranding")
OUTPUT_DIR = Path("output/D/3_eval_wash2")

MAIN_TOP_N = 50
NORMAL_TOP_N = 50
CHUNK_SIZE = 400_000
SEVEN_DAYS_SECONDS = 604800
TOP_PERCENT_LIST = [0.10, 0.20, 0.30]
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


def load_wash_addresses() -> set[str]:
    if not WASH_LABEL_GLOBAL_CSV.exists():
        raise FileNotFoundError(f"Wash label file not found: {WASH_LABEL_GLOBAL_CSV.resolve()}")
    df = pd.read_csv(WASH_LABEL_GLOBAL_CSV)
    if "address" not in df.columns:
        raise ValueError(f"Missing column 'address' in {WASH_LABEL_GLOBAL_CSV}")
    return set(df["address"].astype(str).str.strip().str.lower().tolist())


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


def select_input_files(end_block: int) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
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
        act, _create_blk = file_stats_until_end(fp, end_block=end_block)
        rows_normal.append(
            {
                "file": str(fp),
                "file_name": fp.name,
                "activity_before_end_block": int(act),
            }
        )
    normal_df = pd.DataFrame(rows_normal).sort_values(["activity_before_end_block", "file_name"], ascending=[False, True]).reset_index(drop=True)
    normal_top_df = normal_df.head(int(NORMAL_TOP_N)).copy()
    normal_files = [Path(p) for p in normal_top_df["file"].tolist()]

    files = sorted(set(main_files + normal_files))
    return files, main_top_df, normal_top_df


def estimate_blocks_per_second(df_all: pd.DataFrame) -> float:
    work = df_all[["block_number", "timestamp"]].copy()
    work = work.dropna(subset=["block_number", "timestamp"]).copy()
    work["block_number"] = pd.to_numeric(work["block_number"], errors="coerce")
    work = work.dropna(subset=["block_number"]).copy()
    if work.empty:
        return 1.0 / 12.0
    work = work.sort_values(["timestamp", "block_number"], kind="mergesort").drop_duplicates(subset=["timestamp"], keep="first")
    if len(work) < 2:
        return 1.0 / 12.0

    blk = work["block_number"].to_numpy(dtype=float)
    ts = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    sec = ts.view("int64") / 1e9
    d_blk = np.diff(blk)
    d_sec = np.diff(sec)
    good = (d_blk > 0) & (d_sec > 0)
    if not np.any(good):
        return 1.0 / 12.0
    bps = d_blk[good] / d_sec[good]
    if len(bps) == 0:
        return 1.0 / 12.0
    return float(np.median(bps))


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

    return pd.DataFrame(
        {
            "snapshot_idx": np.arange(1, len(ends) + 1, dtype=np.int64),
            "start_block": int(start_block),
            "end_block": ends,
            "step_blocks_7d": int(step_blocks),
        }
    )


def run_snapshots(trp, df_all: pd.DataFrame, plan: pd.DataFrame, wash_addresses: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    stagea_cache = None
    if STAGEA_REUSE:
        trp.STAGEA_REFRESH_HOURS = None
        trp.STAGEA_REFRESH_BLOCKS = None
        trp.FORCE_STAGEA_REBUILD = False

    users_rows = []
    wash_rows = []
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
        empty_type_map = pd.DataFrame(columns=["contract_address", "collection_type"])
        _contracts_out, users_out, _meta = trp.run_stage_c(
            df_window=df_win,
            u0_full=u0_full,
            stagea_contract=stagea_cache["stagea_contract"],
            tau_end=end_block,
            contract_type_map=empty_type_map,
        )

        if users_out.empty:
            continue

        out = users_out.copy()
        out["user"] = out["user"].astype(str).str.strip().str.lower()
        n_users = int(len(out))
        out["snapshot_idx"] = snapshot_idx
        out["start_block"] = start_block
        out["end_block"] = end_block
        out["n_users_snapshot"] = n_users
        out["rank_pct"] = out["rank_C_user"] / float(max(1, n_users))
        out["is_wash"] = out["user"].isin(wash_addresses).astype(int)

        users_rows.append(
            out[
                [
                    "snapshot_idx",
                    "start_block",
                    "end_block",
                    "user",
                    "uC",
                    "rank_C_user",
                    "n_users_snapshot",
                    "rank_pct",
                    "is_wash",
                ]
            ]
        )

        w = out[out["is_wash"] == 1].copy()
        if not w.empty:
            wash_rows.append(
                w[
                    [
                        "snapshot_idx",
                        "start_block",
                        "end_block",
                        "user",
                        "uC",
                        "rank_C_user",
                        "n_users_snapshot",
                        "rank_pct",
                        "is_wash",
                    ]
                ]
            )

        summary_rows.append(
            {
                "snapshot_idx": snapshot_idx,
                "start_block": start_block,
                "end_block": end_block,
                "n_users_total": n_users,
                "n_wash_users": int(len(w)),
                "wash_ratio": float(len(w) / n_users) if n_users > 0 else np.nan,
                "wash_rank_mean": float(w["rank_C_user"].mean()) if len(w) > 0 else np.nan,
                "wash_rank_median": float(w["rank_C_user"].median()) if len(w) > 0 else np.nan,
                "wash_rank_p25": float(w["rank_C_user"].quantile(0.25)) if len(w) > 0 else np.nan,
                "wash_rank_p75": float(w["rank_C_user"].quantile(0.75)) if len(w) > 0 else np.nan,
                "wash_in_bottom10pct_rate": float((w["rank_pct"] >= 1 - TOP_PERCENT_LIST[0]).mean()) if len(w) > 0 else np.nan,
                "wash_in_bottom20pct_rate": float((w["rank_pct"] >= 1 - TOP_PERCENT_LIST[1]).mean()) if len(w) > 0 else np.nan,
                "wash_in_bottom30pct_rate": float((w["rank_pct"] >= 1 - TOP_PERCENT_LIST[2]).mean()) if len(w) > 0 else np.nan,
                "wash_in_top10pct_rate": float((w["rank_pct"] <= TOP_PERCENT_LIST[0]).mean()) if len(w) > 0 else np.nan,
                "wash_in_top20pct_rate": float((w["rank_pct"] <= TOP_PERCENT_LIST[1]).mean()) if len(w) > 0 else np.nan,
                "wash_in_top30pct_rate": float((w["rank_pct"] <= TOP_PERCENT_LIST[2]).mean()) if len(w) > 0 else np.nan,
            }
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

    users_scores = pd.concat(users_rows, ignore_index=True) if users_rows else pd.DataFrame(
        columns=["snapshot_idx", "start_block", "end_block", "user", "uC", "rank_C_user", "n_users_snapshot", "rank_pct", "is_wash"]
    )
    wash_only = pd.concat(wash_rows, ignore_index=True) if wash_rows else pd.DataFrame(columns=users_scores.columns)
    summary = pd.DataFrame(summary_rows).sort_values("snapshot_idx").reset_index(drop=True) if summary_rows else pd.DataFrame()

    progress_meta["n_snapshots_processed"] = int(summary["snapshot_idx"].nunique()) if not summary.empty else 0
    progress_meta["snapshot_loop_elapsed_seconds"] = float(perf_counter() - loop_t0)
    return users_scores, wash_only, summary, progress_meta


def main():
    t0 = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trp = load_module(TRP_SCRIPT, "trp_mod_wash_eval2")
    wash_addresses = load_wash_addresses()

    author_end_block = get_author_end_block()
    files, main_top50_df, normal_top50_df = select_input_files(end_block=author_end_block)

    df_all = trp.load_events_from_files(files)
    if df_all.empty:
        raise ValueError("No events loaded from selected files")
    df_all = df_all[df_all["block_number"] <= int(author_end_block)].copy()
    if df_all.empty:
        raise ValueError("No events remain after applying author-derived end_block")

    start_block = int(df_all["block_number"].min())
    end_block = int(df_all["block_number"].max())

    bps = estimate_blocks_per_second(df_all)
    step_blocks = int(round(SEVEN_DAYS_SECONDS * bps))
    step_blocks = max(1, step_blocks)
    plan = build_snapshot_plan(start_block=start_block, end_block=end_block, step_blocks=step_blocks)

    user_scores, wash_only, summary, progress_meta = run_snapshots(
        trp=trp,
        df_all=df_all,
        plan=plan,
        wash_addresses=wash_addresses,
    )

    # Save outputs
    plan.to_csv(OUTPUT_DIR / "snapshot_plan.csv", index=False, encoding="utf-8-sig")
    user_scores.to_csv(OUTPUT_DIR / "snapshot_user_scores.csv", index=False, encoding="utf-8-sig")
    wash_only.to_csv(OUTPUT_DIR / "snapshot_wash_only_users.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "snapshot_summary.csv", index=False, encoding="utf-8-sig")
    main_top50_df.to_csv(OUTPUT_DIR / "main_top50_by_activity_then_create_before_end_block.csv", index=False, encoding="utf-8-sig")
    normal_top50_df.to_csv(OUTPUT_DIR / "normal_top50_by_activity_before_end_block.csv", index=False, encoding="utf-8-sig")

    t1 = perf_counter()
    run_meta = pd.DataFrame(
        [
            {"key": "trp_script", "value": str(TRP_SCRIPT)},
            {"key": "wash_label_global", "value": str(WASH_LABEL_GLOBAL_CSV)},
            {"key": "author_trace_dir", "value": str(AUTHOR_TRACE_DIR)},
            {"key": "author_end_block", "value": author_end_block},
            {"key": "input_dir_main", "value": str(INPUT_DIR_MAIN)},
            {"key": "input_dir_normal", "value": str(INPUT_DIR_NORMAL)},
            {"key": "main_top_n", "value": MAIN_TOP_N},
            {"key": "normal_top_n", "value": NORMAL_TOP_N},
            {"key": "n_input_files_total", "value": len(files)},
            {"key": "n_input_files_main_top50", "value": len(main_top50_df)},
            {"key": "n_input_files_normal_top50", "value": len(normal_top50_df)},
            {"key": "start_block", "value": start_block},
            {"key": "end_block", "value": end_block},
            {"key": "blocks_per_second_median", "value": bps},
            {"key": "step_blocks_7d", "value": step_blocks},
            {"key": "n_snapshots", "value": len(plan)},
            {"key": "n_snapshots_processed", "value": progress_meta["n_snapshots_processed"]},
            {"key": "n_snapshot_user_rows", "value": len(user_scores)},
            {"key": "n_snapshot_wash_rows", "value": len(wash_only)},
            {"key": "stagea_reuse", "value": int(1 if STAGEA_REUSE else 0)},
            {"key": "top_percent_list", "value": "|".join(str(x) for x in TOP_PERCENT_LIST)},
            {"key": "eta_total_seconds_after_first_snapshot", "value": progress_meta["eta_total_seconds_after_first_snapshot"]},
            {"key": "snapshot_loop_elapsed_seconds", "value": progress_meta["snapshot_loop_elapsed_seconds"]},
            {"key": "elapsed_seconds", "value": round(t1 - t0, 3)},
        ]
    )
    run_meta.to_csv(OUTPUT_DIR / "run_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'snapshot_plan.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'snapshot_user_scores.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'snapshot_wash_only_users.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'snapshot_summary.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'run_meta.csv').resolve()}")


if __name__ == "__main__":
    main()
