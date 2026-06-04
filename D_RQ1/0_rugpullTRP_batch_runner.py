#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import importlib.util

import pandas as pd


# ============================================================
# Config (edit here)
# ============================================================
PLAN_CSV = Path("output/D/1_eval_rugpull_metrics_single/rerun_block_windows_plan.csv")
COHORT_CSV = Path("output/D/1_eval_rugpull_metrics_single/rugpull_cohorts.csv")
LABEL_CSV = Path("output/C/3_collectionlabel.csv")
TRP_SCRIPT = Path("D_RQ1/0_rugpullTRP.py")

# Optional filters
TARGET_COHORT_IDS: set[str] = {"0x4923017f3b7fac4e096b46e401c0662f0b7e393f_15136287"}
MAX_WINDOWS = 120  # 0 => all
WINDOW_STRIDE = 24  # keep every N-th window after sorting; 1 means no downsampling
SKIP_EXISTING_CONTRACTS = True
USE_COHORT_UNIVERSE = True  # use union(member_address) from cohort csv to build event universe
WRITE_USERS_AND_META = False  # False => write only contracts_{end_block}.csv for faster batch
FORCE_STAGEA_REUSE = True  # True => avoid frequent Stage A rebuilds in hourly stepping
BATCH_OUTPUT_DIR = Path("output/D/0_rugpullTRP_single")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_windows(plan_csv: Path) -> pd.DataFrame:
    if not plan_csv.exists():
        raise FileNotFoundError(f"Plan CSV not found: {plan_csv.resolve()}")
    df = pd.read_csv(plan_csv)
    need = {"start_block", "end_block"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in plan csv: {miss}")

    if TARGET_COHORT_IDS:
        if "cohort_id" not in df.columns:
            raise ValueError("TARGET_COHORT_IDS is set but plan csv has no cohort_id column.")
        df = df[df["cohort_id"].astype(str).isin(TARGET_COHORT_IDS)].copy()

    df["start_block"] = pd.to_numeric(df["start_block"], errors="coerce")
    df["end_block"] = pd.to_numeric(df["end_block"], errors="coerce")
    df = df.dropna(subset=["start_block", "end_block"]).copy()
    df["start_block"] = df["start_block"].astype(int)
    df["end_block"] = df["end_block"].astype(int)
    df = df[df["start_block"] <= df["end_block"]].copy()
    df = df.drop_duplicates(subset=["start_block", "end_block"]).sort_values(["end_block", "start_block"]).reset_index(drop=True)

    if WINDOW_STRIDE and int(WINDOW_STRIDE) > 1 and len(df) > 0:
        stride = int(WINDOW_STRIDE)
        keep_idx = list(range(0, len(df), stride))
        if keep_idx[-1] != len(df) - 1:
            keep_idx.append(len(df) - 1)
        df = df.iloc[keep_idx].copy().reset_index(drop=True)

    if MAX_WINDOWS and MAX_WINDOWS > 0:
        df = df.head(int(MAX_WINDOWS)).copy()
    return df


def build_cohort_address_universe(cohort_csv: Path, label_csv: Path, cohort_filter: set[str]) -> tuple[list[Path], pd.DataFrame, dict]:
    if not cohort_csv.exists():
        raise FileNotFoundError(f"Cohort CSV not found: {cohort_csv.resolve()}")
    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv.resolve()}")

    cdf = pd.read_csv(cohort_csv)
    need_c = {"member_address", "cohort_id"}
    miss_c = [c for c in need_c if c not in cdf.columns]
    if miss_c:
        raise ValueError(f"Missing columns in cohort csv: {miss_c}")
    if cohort_filter:
        cdf = cdf[cdf["cohort_id"].astype(str).isin(cohort_filter)].copy()
    cdf["member_address"] = cdf["member_address"].astype(str).str.lower()
    addresses = sorted(set(cdf["member_address"].tolist()))
    if not addresses:
        raise ValueError("No addresses selected from cohort csv.")

    ldf = pd.read_csv(label_csv)
    need_l = {"contract_address", "source_tx_file", "collection_type"}
    miss_l = [c for c in need_l if c not in ldf.columns]
    if miss_l:
        raise ValueError(f"Missing columns in label csv: {miss_l}")
    ldf["contract_address"] = ldf["contract_address"].astype(str).str.lower()
    ldf["source_tx_file"] = ldf["source_tx_file"].astype(str).str.strip()
    ldf["collection_type"] = ldf["collection_type"].astype(str).str.lower()
    ldf = ldf[ldf["source_tx_file"] != ""].copy()
    ldf = ldf.drop_duplicates(subset=["contract_address"], keep="first")

    pick = ldf[ldf["contract_address"].isin(addresses)].copy()
    files = [Path(p) for p in pick["source_tx_file"].tolist() if Path(p).exists()]
    if not files:
        raise ValueError("No source files found for selected cohort addresses.")

    type_map = pick[["contract_address", "collection_type"]].copy()
    meta = {
        "label_filter_mode": "cohort_universe",
        "label_include_types": "|".join(sorted(type_map["collection_type"].dropna().unique().tolist())),
        "label_selected_collections": int(type_map["contract_address"].nunique()),
        "label_selected_rows": int(len(type_map)),
        "label_selected_files": int(len(files)),
        "label_source_field": "source_tx_file",
        "label_random_sampling": "",
    }
    return files, type_map, meta


def main() -> None:
    trp = load_module(TRP_SCRIPT, "rugpull_trp_mod")
    if FORCE_STAGEA_REUSE:
        trp.STAGEA_REFRESH_HOURS = None
        trp.STAGEA_REFRESH_BLOCKS = None
        trp.FORCE_STAGEA_REBUILD = False
        trp.SAVE_STAGEA_SNAPSHOT_ON_REBUILD = False
    windows_df = build_windows(PLAN_CSV)
    if windows_df.empty:
        raise ValueError("No valid windows to run from plan.")

    if USE_COHORT_UNIVERSE:
        files, contract_type_map, label_meta = build_cohort_address_universe(
            cohort_csv=COHORT_CSV, label_csv=LABEL_CSV, cohort_filter=TARGET_COHORT_IDS
        )
        df_all = trp.load_events_from_files(files)
    else:
        df_all, label_meta, contract_type_map = trp.load_events()
    if df_all.empty:
        raise ValueError("No events after input selection.")

    output_dir = BATCH_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stagea_cache = None

    print(f"[Batch] total_windows={len(windows_df):,}, universe_contracts={label_meta['label_selected_collections']:,}")
    for i, r in enumerate(windows_df.itertuples(index=False), 1):
        start_block = int(r.start_block)
        end_block = int(r.end_block)

        contracts_fp = output_dir / f"contracts_{end_block}.csv"
        if SKIP_EXISTING_CONTRACTS and contracts_fp.exists():
            print(f"[Batch] {i}/{len(windows_df)} skip existing end_block={end_block}")
            continue

        df_hist = df_all[df_all["block_number"] <= end_block].copy()
        df_win = df_all[(df_all["block_number"] >= start_block) & (df_all["block_number"] <= end_block)].copy()
        if df_win.empty:
            print(f"[Batch] {i}/{len(windows_df)} empty window [{start_block}, {end_block}]")
            continue

        end_time = trp.block_to_time(df_all, end_block)
        rebuild, reason = trp.should_rebuild(stagea_cache, end_block, end_time)
        stagea_snapshot = None
        if rebuild:
            stagea_contract, stagea_user = trp.run_stage_a(df_hist)
            stagea_cache = {
                "end_block": int(end_block),
                "end_time": end_time,
                "stagea_contract": stagea_contract,
                "stagea_user": stagea_user,
            }
            stagea_snapshot = stagea_contract.copy()

        stagea_contract = stagea_cache["stagea_contract"]
        stagea_user = stagea_cache["stagea_user"]
        u0_full = trp.run_stage_b(df_hist, stagea_contract, stagea_user)
        contracts_out, users_out, cmeta = trp.run_stage_c(
            df_win, u0_full, stagea_contract, tau_end=end_block, contract_type_map=contract_type_map
        )

        meta = {
            "start_block": start_block,
            "end_block": end_block,
            "stagea_rebuilt": int(1 if rebuild else 0),
            "stagea_rebuild_reason": reason,
            "label_filter_mode": label_meta["label_filter_mode"],
            "label_include_types": label_meta["label_include_types"],
            "label_selected_collections": label_meta["label_selected_collections"],
            "label_selected_files": label_meta["label_selected_files"],
            "label_random_sampling": label_meta["label_random_sampling"],
            "iters_used": cmeta["iters_used"],
            "final_delta": cmeta["final_delta"],
            "events": cmeta["events"],
            "nnz_wrec": cmeta["nnz_wrec"],
        }
        if WRITE_USERS_AND_META:
            trp.save_outputs(end_block, contracts_out, users_out, meta, stagea_snapshot)
        else:
            contracts_out.to_csv(contracts_fp, index=False, encoding="utf-8-sig")
            print(f"[Saved] {contracts_fp.resolve()}")
        print(f"[Batch] {i}/{len(windows_df)} done [{start_block}, {end_block}]")

    print("[Batch] finished.")


if __name__ == "__main__":
    main()

