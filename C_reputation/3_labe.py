#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUG_DIR = PROJECT_ROOT / "output" / "B" / "2_3rugpullTranding"
COUNTERFEIT_DIR = PROJECT_ROOT / "output" / "B" / "2_2counterfeitTranding"
UNKNOWN_DIR = PROJECT_ROOT / "output" / "B" / "2_4normalTranding"
TRUST_DIR = PROJECT_ROOT / "output" / "B" / "2_tranding"

RUG_DETECT_FILE = PROJECT_ROOT / "output" / "A" / "8_2NFTrugpull" / "8_2_rugpull_nft_only.csv"
OUTPUT_CSV = PROJECT_ROOT / "output" / "C" / "3_collectionlabel.csv"

TRUST_TARGET_BLOCK = 19777901
TRUST_TOP_N_BY_DAILY = 90
TRUST_FINAL_N = 50

COUNTERFEIT_OFFSET_MIN = 9800
COUNTERFEIT_OFFSET_MAX = 10200

OUTPUT_COLUMNS = [
    "contract_address",
    "create_block_number",
    "collection_type",
    "detected_block_number",
    "detected_time",
    "source_tx_file",
    "source_detect_file",
    "max_block_number",
    "first_timestamp",
    "last_timestamp",
    "daily_activity",
    "selected_for_trust",
    "notes",
]

REQUIRED_TX_COLS = ["contract_address", "block_number", "timestamp", "tx_type"]


def path_rel_str(p: Path) -> str:
    return p.relative_to(PROJECT_ROOT).as_posix()


def parse_mixed_timestamp_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    out = pd.to_datetime(s, errors="coerce", utc=True)
    mask = out.isna()
    if mask.any():
        num = pd.to_numeric(s[mask], errors="coerce")
        out.loc[mask] = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
    mask2 = out.isna()
    if mask2.any():
        num2 = pd.to_numeric(s[mask2], errors="coerce")
        out.loc[mask2] = pd.to_datetime(num2, unit="s", errors="coerce", utc=True)
    return out


def stable_counterfeit_offset(contract_address: str) -> int:
    digest = hashlib.sha256(contract_address.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    span = COUNTERFEIT_OFFSET_MAX - COUNTERFEIT_OFFSET_MIN + 1
    return COUNTERFEIT_OFFSET_MIN + (value % span)


def read_collection_csv(csv_path: Path) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    try:
        df = pd.read_csv(csv_path, usecols=REQUIRED_TX_COLS, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(csv_path)
            miss = [c for c in REQUIRED_TX_COLS if c not in df.columns]
            if miss:
                return pd.DataFrame(), None, f"missing columns {miss}"
            df = df[REQUIRED_TX_COLS].copy()
        except Exception as e:
            return pd.DataFrame(), None, f"read error: {e}"

    if df.empty:
        return df, None, "empty file"

    df["contract_address"] = df["contract_address"].astype(str).str.strip().str.lower()
    df["block_number"] = pd.to_numeric(df["block_number"], errors="coerce")
    df["tx_type"] = df["tx_type"].astype(str).str.strip().str.lower()
    df["parsed_timestamp"] = parse_mixed_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["contract_address", "block_number"]).copy()
    if df.empty:
        return df, None, "no valid address/block rows"

    df["block_number"] = df["block_number"].astype(np.int64)
    contract_address = df["contract_address"].iloc[0]
    return df, contract_address, None


def compute_daily_activity(df: pd.DataFrame) -> float:
    act = df[df["tx_type"] != "create"].copy()
    if act.empty:
        return 0.0
    valid_ts = act["parsed_timestamp"].dropna()
    if valid_ts.empty:
        return 0.0
    days = valid_ts.dt.date
    daily_counts = days.value_counts()
    if daily_counts.empty:
        return 0.0
    return float(daily_counts.mean())


def iso_or_none(ts: pd.Timestamp) -> Optional[str]:
    if pd.isna(ts):
        return None
    return ts.isoformat()


def base_record_from_df(df: pd.DataFrame, source_tx_file: str) -> Dict:
    min_block = int(df["block_number"].min())
    max_block = int(df["block_number"].max())
    first_ts = df["parsed_timestamp"].min()
    last_ts = df["parsed_timestamp"].max()
    daily_activity = compute_daily_activity(df)
    return {
        "contract_address": df["contract_address"].iloc[0],
        "create_block_number": min_block,
        "collection_type": None,
        "detected_block_number": None,
        "detected_time": None,
        "source_tx_file": source_tx_file,
        "source_detect_file": None,
        "max_block_number": max_block,
        "first_timestamp": iso_or_none(first_ts),
        "last_timestamp": iso_or_none(last_ts),
        "daily_activity": daily_activity,
        "selected_for_trust": 0,
        "notes": "",
    }


def estimate_block_by_time(df: pd.DataFrame, target_ts: pd.Timestamp) -> Optional[int]:
    work = df.dropna(subset=["parsed_timestamp"]).copy()
    if work.empty:
        return None
    work["ts_ns"] = work["parsed_timestamp"].view("int64")
    work = work.sort_values(["ts_ns", "block_number"], kind="mergesort")
    points = work[["ts_ns", "block_number"]].drop_duplicates(subset=["ts_ns"], keep="first")
    if len(points) < 2:
        return None

    x = points["ts_ns"].to_numpy(dtype=np.int64)
    y = points["block_number"].to_numpy(dtype=np.int64)
    tx = int(target_ts.value)

    if tx <= x[0]:
        return int(y[0])
    if tx >= x[-1]:
        return int(y[-1])

    idx = int(np.searchsorted(x, tx))
    left = idx - 1
    right = idx
    x1, x2 = int(x[left]), int(x[right])
    y1, y2 = int(y[left]), int(y[right])
    if x2 == x1:
        return int(y1)
    ratio = (tx - x1) / (x2 - x1)
    est = y1 + ratio * (y2 - y1)
    return int(round(est))


def build_group_records(
    input_dir: Path,
    label: str,
    errors: List[str],
) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
    records: List[Dict] = []
    df_map: Dict[str, pd.DataFrame] = {}
    for fp in sorted(input_dir.glob("*.csv")):
        df, addr, err = read_collection_csv(fp)
        if err:
            errors.append(f"{path_rel_str(fp)}: {err}")
            continue
        rec = base_record_from_df(df, path_rel_str(fp))
        rec["collection_type"] = label
        records.append(rec)
        if addr:
            df_map[addr] = df
    return records, df_map


def build_rug_detect_map(csv_path: Path) -> Dict[str, str]:
    df = pd.read_csv(csv_path, usecols=["reported_address", "submitted_at"])
    df["reported_address"] = df["reported_address"].astype(str).str.strip().str.lower()
    ts = pd.to_datetime(df["submitted_at"], errors="coerce", utc=True)
    df = df.assign(parsed_submit=ts).dropna(subset=["reported_address", "parsed_submit"])
    df = df.sort_values(["reported_address", "parsed_submit"], kind="mergesort")
    first = df.drop_duplicates(subset=["reported_address"], keep="first")
    return {
        str(r.reported_address): pd.Timestamp(r.parsed_submit).isoformat()
        for r in first.itertuples(index=False)
    }


def apply_rugpull_detection(
    rug_records: List[Dict],
    rug_df_map: Dict[str, pd.DataFrame],
    rug_detect_map: Dict[str, str],
    errors: List[str],
) -> None:
    detect_file_rel = path_rel_str(RUG_DETECT_FILE)
    for rec in rug_records:
        addr = rec["contract_address"]
        rec["source_detect_file"] = detect_file_rel
        submitted_iso = rug_detect_map.get(addr)
        if not submitted_iso:
            rec["notes"] = "missing submitted_at in rug detect file; fallback detected_block=max_block"
            rec["detected_block_number"] = rec["max_block_number"]
            errors.append(f"rugpull missing submit time: {addr}")
            continue

        rec["detected_time"] = submitted_iso
        target_ts = pd.to_datetime(submitted_iso, errors="coerce", utc=True)
        if pd.isna(target_ts):
            rec["detected_block_number"] = rec["max_block_number"]
            rec["notes"] = "invalid submitted_at parse; fallback detected_block=max_block"
            errors.append(f"rugpull invalid submit parse: {addr}")
            continue

        df = rug_df_map.get(addr)
        if df is None:
            rec["detected_block_number"] = rec["max_block_number"]
            rec["notes"] = "rug tx file not found in map; fallback detected_block=max_block"
            errors.append(f"rugpull df missing by address: {addr}")
            continue

        est_block = estimate_block_by_time(df, target_ts)
        if est_block is None:
            rec["detected_block_number"] = rec["max_block_number"]
            rec["notes"] = "insufficient timestamp points; fallback detected_block=max_block"
        else:
            rec["detected_block_number"] = int(est_block)


def apply_counterfeit_detection(counterfeit_records: List[Dict]) -> None:
    for rec in counterfeit_records:
        rec["source_detect_file"] = rec["source_tx_file"]
        addr = rec["contract_address"]
        offset = stable_counterfeit_offset(addr)
        rec["detected_block_number"] = int(rec["max_block_number"]) + offset
        if rec["last_timestamp"]:
            last_ts = pd.to_datetime(rec["last_timestamp"], errors="coerce", utc=True)
            if not pd.isna(last_ts):
                rec["detected_time"] = (last_ts + pd.Timedelta(days=1)).isoformat()
        rec["notes"] = f"counterfeit_detect_offset={offset} from max_block"


def select_trust_records(trust_candidates: List[Dict]) -> List[Dict]:
    df = pd.DataFrame(trust_candidates)
    if df.empty:
        return []

    eligible = df[df["max_block_number"] >= TRUST_TARGET_BLOCK].copy()
    if eligible.empty:
        return []

    eligible = eligible.sort_values(
        ["daily_activity", "create_block_number", "contract_address"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(TRUST_TOP_N_BY_DAILY)

    final = eligible.sort_values(
        ["create_block_number", "contract_address"],
        ascending=[True, True],
        kind="mergesort",
    ).head(TRUST_FINAL_N)

    final = final.copy()
    final["selected_for_trust"] = 1
    final["collection_type"] = "trust"
    return final.to_dict(orient="records")


def merge_by_priority(parts: List[pd.DataFrame]) -> pd.DataFrame:
    all_df = pd.concat(parts, ignore_index=True)
    priority = {"rugpull": 1, "counterfeit": 2, "trust": 3, "unknown": 4}
    all_df["priority"] = all_df["collection_type"].map(priority).fillna(999).astype(int)
    all_df = all_df.sort_values(["contract_address", "priority"], kind="mergesort")
    dedup = all_df.drop_duplicates(subset=["contract_address"], keep="first").copy()
    dedup = dedup.drop(columns=["priority"])
    return dedup


def main() -> None:
    errors: List[str] = []

    rug_records, rug_df_map = build_group_records(RUG_DIR, "rugpull", errors)
    counterfeit_records, _ = build_group_records(COUNTERFEIT_DIR, "counterfeit", errors)
    unknown_records, _ = build_group_records(UNKNOWN_DIR, "unknown", errors)
    trust_candidates, _ = build_group_records(TRUST_DIR, "trust_candidate", errors)

    rug_detect_map = build_rug_detect_map(RUG_DETECT_FILE)
    apply_rugpull_detection(rug_records, rug_df_map, rug_detect_map, errors)
    apply_counterfeit_detection(counterfeit_records)
    trust_records = select_trust_records(trust_candidates)

    rug_df = pd.DataFrame(rug_records)
    counterfeit_df = pd.DataFrame(counterfeit_records)
    unknown_df = pd.DataFrame(unknown_records)
    trust_df = pd.DataFrame(trust_records)

    merged = merge_by_priority([rug_df, counterfeit_df, trust_df, unknown_df])
    for c in OUTPUT_COLUMNS:
        if c not in merged.columns:
            merged[c] = None
    merged = merged[OUTPUT_COLUMNS].copy()
    merged = merged.sort_values(["collection_type", "create_block_number", "contract_address"], kind="mergesort")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[Saved] {OUTPUT_CSV.resolve()}")
    print(
        "[Counts raw] rugpull={0} counterfeit={1} unknown={2} trust={3}".format(
            len(rug_records), len(counterfeit_records), len(unknown_records), len(trust_records)
        )
    )
    print(f"[Counts dedup] total={len(merged)}")
    if not merged.empty:
        by_type = merged["collection_type"].value_counts().to_dict()
        print(f"[Counts dedup by type] {by_type}")
    if errors:
        print(f"[Warnings] {len(errors)} issues")
        for msg in errors[:50]:
            print(f" - {msg}")
        if len(errors) > 50:
            print(f" - ... and {len(errors) - 50} more")
    else:
        print("[Warnings] none")


if __name__ == "__main__":
    main()
