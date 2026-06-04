#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import importlib.util

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
OUTPUT_DIR = Path("output/D/3_eval_wash")
MAIN_TOP_N = 50
NORMAL_TOP_N = 50
CHUNK_SIZE = 400_000


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def select_input_files(end_block: int) -> tuple[list[Path], pd.DataFrame]:
    files_main = sorted(INPUT_DIR_MAIN.glob("*.csv"))
    if not files_main:
        raise FileNotFoundError(f"No csv found in {INPUT_DIR_MAIN.resolve()}")

    files_normal = sorted(INPUT_DIR_NORMAL.glob("*.csv"))
    if not files_normal:
        raise FileNotFoundError(f"No csv found in {INPUT_DIR_NORMAL.resolve()}")

    # Main collections: select top MAIN_TOP_N by activity before end_block,
    # then earlier create block first as tie-break.
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

    rows = []
    for fp in files_normal:
        act, _create_blk = file_stats_until_end(fp, end_block=end_block)
        rows.append({"file": str(fp), "file_name": fp.name, "activity_before_end_block": int(act)})
    act_df = pd.DataFrame(rows).sort_values(["activity_before_end_block", "file_name"], ascending=[False, True]).reset_index(drop=True)
    top_df = act_df.head(int(NORMAL_TOP_N)).copy()
    top_files = [Path(p) for p in top_df["file"].tolist()]

    files = sorted(set(main_files + top_files))
    return files, top_df, main_top_df


def rank_summary(users_scored: pd.DataFrame) -> pd.DataFrame:
    wash = users_scored[users_scored["is_wash"] == 1].copy()
    all_n = int(len(users_scored))
    wash_n = int(len(wash))
    out = {
        "n_users_total": all_n,
        "n_wash_users": wash_n,
        "wash_ratio": float(wash_n / all_n) if all_n > 0 else np.nan,
        "wash_rank_mean": float(wash["rank_C_user"].mean()) if wash_n > 0 else np.nan,
        "wash_rank_median": float(wash["rank_C_user"].median()) if wash_n > 0 else np.nan,
        "wash_rank_p25": float(wash["rank_C_user"].quantile(0.25)) if wash_n > 0 else np.nan,
        "wash_rank_p75": float(wash["rank_C_user"].quantile(0.75)) if wash_n > 0 else np.nan,
        "wash_in_bottom10pct_rate": float((wash["rank_pct"] >= 0.90).mean()) if wash_n > 0 else np.nan,
        "wash_in_bottom20pct_rate": float((wash["rank_pct"] >= 0.80).mean()) if wash_n > 0 else np.nan,
        "wash_in_bottom30pct_rate": float((wash["rank_pct"] >= 0.70).mean()) if wash_n > 0 else np.nan,
        "wash_in_top10pct_rate": float((wash["rank_pct"] <= 0.10).mean()) if wash_n > 0 else np.nan,
        "wash_in_top20pct_rate": float((wash["rank_pct"] <= 0.20).mean()) if wash_n > 0 else np.nan,
        "wash_in_top30pct_rate": float((wash["rank_pct"] <= 0.30).mean()) if wash_n > 0 else np.nan,
    }
    return pd.DataFrame([out])


def main():
    trp = load_module(TRP_SCRIPT, "trp_mod_wash_eval")
    wash_addresses = load_wash_addresses()
    end_block = get_author_end_block()
    files, normal_top50_df, main_top50_df = select_input_files(end_block=end_block)

    df_all = trp.load_events_from_files(files)
    if df_all.empty:
        raise ValueError("No events loaded from selected files")

    df_all = df_all[df_all["block_number"] <= int(end_block)].copy()
    if df_all.empty:
        raise ValueError("No events remain after applying author-derived end_block")
    start_block = int(df_all["block_number"].min())
    df_hist = df_all.copy()
    df_win = df_all[(df_all["block_number"] >= start_block) & (df_all["block_number"] <= end_block)].copy()

    stagea_contract, stagea_user = trp.run_stage_a(df_hist)
    u0_full = trp.run_stage_b(df_hist, stagea_contract, stagea_user)
    empty_type_map = pd.DataFrame(columns=["contract_address", "collection_type"])
    _contracts_out, users_out, cmeta = trp.run_stage_c(
        df_window=df_win,
        u0_full=u0_full,
        stagea_contract=stagea_contract,
        tau_end=end_block,
        contract_type_map=empty_type_map,
    )

    users = users_out.copy()
    users["user"] = users["user"].astype(str).str.strip().str.lower()
    n_users = int(len(users))
    users["rank_pct"] = users["rank_C_user"] / float(max(1, n_users))
    users["is_wash"] = users["user"].isin(wash_addresses).astype(int)

    users = users.sort_values(["rank_C_user", "user"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    wash_only = users[users["is_wash"] == 1].copy().sort_values(["rank_C_user", "user"], ascending=[True, True], kind="mergesort")
    summary = rank_summary(users)
    summary["start_block"] = start_block
    summary["end_block"] = end_block
    summary["events"] = int(cmeta["events"])
    summary["nnz_wrec"] = int(cmeta["nnz_wrec"])
    summary["iters_used"] = int(cmeta["iters_used"])
    summary["final_delta"] = float(cmeta["final_delta"])
    summary["n_input_files"] = int(len(files))
    summary["n_wash_labels_input"] = int(len(wash_addresses))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    users_fp = OUTPUT_DIR / "wash_user_rank_all.csv"
    wash_fp = OUTPUT_DIR / "wash_user_rank_only.csv"
    summary_fp = OUTPUT_DIR / "wash_user_rank_summary.csv"
    meta_fp = OUTPUT_DIR / "wash_user_rank_meta.csv"
    main_top50_fp = OUTPUT_DIR / "main_top50_by_activity_then_create_before_end_block.csv"
    normal_top50_fp = OUTPUT_DIR / "normal_top50_by_activity_before_end_block.csv"

    users.to_csv(users_fp, index=False, encoding="utf-8-sig")
    wash_only.to_csv(wash_fp, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_fp, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "key": [
                "trp_script",
                "wash_label_global",
                "author_trace_dir",
                "input_dir_main",
                "input_dir_normal",
                "main_top_n",
                "normal_top_n",
                "author_end_block",
                "n_input_files_main",
                "n_input_files_main_top50",
                "n_input_files_normal_top50",
            ],
            "value": [
                str(TRP_SCRIPT),
                str(WASH_LABEL_GLOBAL_CSV),
                str(AUTHOR_TRACE_DIR),
                str(INPUT_DIR_MAIN),
                str(INPUT_DIR_NORMAL),
                str(MAIN_TOP_N),
                str(NORMAL_TOP_N),
                str(end_block),
                str(len(list(INPUT_DIR_MAIN.glob('*.csv')))),
                str(len(main_top50_df)),
                str(len(normal_top50_df)),
            ],
        }
    ).to_csv(meta_fp, index=False, encoding="utf-8-sig")
    main_top50_df.to_csv(main_top50_fp, index=False, encoding="utf-8-sig")
    normal_top50_df.to_csv(normal_top50_fp, index=False, encoding="utf-8-sig")

    print(f"[Saved] {users_fp.resolve()}")
    print(f"[Saved] {wash_fp.resolve()}")
    print(f"[Saved] {summary_fp.resolve()}")
    print(f"[Saved] {meta_fp.resolve()}")
    print(f"[Saved] {main_top50_fp.resolve()}")
    print(f"[Saved] {normal_top50_fp.resolve()}")
    print(f"[Done] users={len(users)}, wash_users={len(wash_only)}, wash_labels_input={len(wash_addresses)}")


if __name__ == "__main__":
    main()
