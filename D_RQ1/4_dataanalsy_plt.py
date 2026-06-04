#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# Config
# ============================================================
LABEL_CSV = Path("output/C/3_collectionlabel.csv")
OUTPUT_DIR = Path("output/D/4_dataanalsy")

# If None, malicious types are auto-detected as all types except trust/unknown.
MALICIOUS_TYPES: list[str] | None = ["rugpull", "counterfeit"]


def _normalize_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def _parse_mixed_timestamp_col(s: pd.Series) -> pd.Series:
    # 1) direct parse for ISO-like strings
    dt = pd.to_datetime(s, errors="coerce", utc=True)

    # 2) fill unresolved entries with numeric epoch parsing (ms first, then s)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        # Typical Ethereum dataset often stores epoch in ms around 1e12+
        num_ms = num.where(num >= 1e12)
        num_s = num.where((num > 0) & (num < 1e12))
        dt_ms = pd.to_datetime(num_ms, unit="ms", errors="coerce", utc=True)
        dt_s = pd.to_datetime(num_s, unit="s", errors="coerce", utc=True)
        dt = dt.fillna(dt_ms).fillna(dt_s)
    return dt


def _compute_last_peak_activity_time(tx_file: Path, detected_time: pd.Timestamp) -> tuple[pd.Timestamp, int, int]:
    if not tx_file.exists():
        return pd.NaT, 0, 0

    tx = pd.read_csv(tx_file, usecols=["timestamp"], low_memory=False)
    if tx.empty or "timestamp" not in tx.columns:
        return pd.NaT, 0, 0

    ts = _parse_mixed_timestamp_col(tx["timestamp"])
    ts = ts.dropna()
    ts = ts[ts <= detected_time]
    if ts.empty:
        return pd.NaT, 0, 0

    day = ts.dt.floor("D")
    day_counts = day.value_counts().sort_index()
    peak_cnt = int(day_counts.max())
    peak_days = day_counts[day_counts == peak_cnt].index
    last_peak_day = peak_days.max()
    peak_ts = ts[day == last_peak_day].max()
    return peak_ts, peak_cnt, int(len(ts))


def main() -> None:
    if not LABEL_CSV.exists():
        raise FileNotFoundError(f"Label file not found: {LABEL_CSV.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LABEL_CSV, low_memory=False)
    need_cols = [
        "contract_address",
        "collection_type",
        "detected_time",
        "last_timestamp",
        "first_timestamp",
        "source_tx_file",
    ]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in label csv: {missing}")

    work = df.copy()
    work["contract_address"] = _normalize_text(work["contract_address"])
    work["collection_type"] = _normalize_text(work["collection_type"])
    work["detected_time"] = pd.to_datetime(work["detected_time"], errors="coerce", utc=True)
    work["last_timestamp"] = pd.to_datetime(work["last_timestamp"], errors="coerce", utc=True)
    work["first_timestamp"] = pd.to_datetime(work["first_timestamp"], errors="coerce", utc=True)

    if MALICIOUS_TYPES is None:
        malicious_types = sorted(
            t for t in work["collection_type"].dropna().unique().tolist() if t not in {"trust", "unknown", ""}
        )
    else:
        malicious_types = [str(t).strip().lower() for t in MALICIOUS_TYPES]

    mal = work[work["collection_type"].isin(malicious_types)].copy()
    mal = mal.dropna(subset=["detected_time", "last_timestamp", "first_timestamp"]).copy()
    if mal.empty:
        raise ValueError("No valid malicious rows after filtering and datetime parsing.")

    # Time lag between reported time and last transaction time.
    mal["lag_seconds"] = (mal["detected_time"] - mal["last_timestamp"]).dt.total_seconds()
    mal["lag_hours"] = mal["lag_seconds"] / 3600.0
    mal["lag_days"] = mal["lag_seconds"] / 86400.0
    mal["active_seconds_before_inactive"] = (mal["last_timestamp"] - mal["first_timestamp"]).dt.total_seconds()
    mal["active_days_before_inactive"] = mal["active_seconds_before_inactive"] / 86400.0
    mal["total_seconds_first_to_detect"] = (mal["detected_time"] - mal["first_timestamp"]).dt.total_seconds()
    mal["inactive_start_ratio"] = mal["active_seconds_before_inactive"] / mal["total_seconds_first_to_detect"].replace(0, pd.NA)

    # Calendar view: when inactivity starts
    mal["inactive_start_date_utc"] = mal["last_timestamp"].dt.date
    mal["inactive_start_month_utc"] = mal["last_timestamp"].dt.to_period("M").astype(str)

    # Peak activity time before report (behavioral "most active" anchor)
    peak_rows = []
    for r in mal.itertuples(index=False):
        fp = Path(str(r.source_tx_file))
        peak_ts, peak_cnt, tx_n = _compute_last_peak_activity_time(fp, r.detected_time)
        peak_rows.append(
            {
                "contract_address": str(r.contract_address),
                "peak_active_time": peak_ts,
                "peak_daily_tx_count": peak_cnt,
                "tx_count_before_detect": tx_n,
            }
        )
    peak_df = pd.DataFrame(peak_rows)
    peak_df["contract_address"] = _normalize_text(peak_df["contract_address"])
    mal = mal.merge(peak_df, on="contract_address", how="left")
    mal["peak_active_time"] = pd.to_datetime(mal["peak_active_time"], errors="coerce", utc=True)
    mal["detect_minus_peak_seconds"] = (mal["detected_time"] - mal["peak_active_time"]).dt.total_seconds()
    mal["detect_minus_peak_days"] = mal["detect_minus_peak_seconds"] / 86400.0

    by_type = (
        mal.groupby("collection_type", as_index=False)
        .agg(
            n=("contract_address", "count"),
            lag_mean_seconds=("lag_seconds", "mean"),
            lag_median_seconds=("lag_seconds", "median"),
            lag_mean_hours=("lag_hours", "mean"),
            lag_median_hours=("lag_hours", "median"),
            lag_mean_days=("lag_days", "mean"),
            lag_median_days=("lag_days", "median"),
            lag_min_seconds=("lag_seconds", "min"),
            lag_max_seconds=("lag_seconds", "max"),
            active_days_before_inactive_mean=("active_days_before_inactive", "mean"),
            active_days_before_inactive_median=("active_days_before_inactive", "median"),
            inactive_start_ratio_mean=("inactive_start_ratio", "mean"),
            inactive_start_ratio_median=("inactive_start_ratio", "median"),
            detect_minus_peak_days_mean=("detect_minus_peak_days", "mean"),
            detect_minus_peak_days_median=("detect_minus_peak_days", "median"),
        )
        .sort_values("collection_type", kind="mergesort")
        .reset_index(drop=True)
    )

    overall = pd.DataFrame(
        [
            {
                "scope": "malicious_all",
                "types": "|".join(malicious_types),
                "n": int(len(mal)),
                "lag_mean_seconds": float(mal["lag_seconds"].mean()),
                "lag_median_seconds": float(mal["lag_seconds"].median()),
                "lag_mean_hours": float(mal["lag_hours"].mean()),
                "lag_median_hours": float(mal["lag_hours"].median()),
                "lag_mean_days": float(mal["lag_days"].mean()),
                "lag_median_days": float(mal["lag_days"].median()),
                "lag_min_seconds": float(mal["lag_seconds"].min()),
                "lag_max_seconds": float(mal["lag_seconds"].max()),
                "active_days_before_inactive_mean": float(mal["active_days_before_inactive"].mean()),
                "active_days_before_inactive_median": float(mal["active_days_before_inactive"].median()),
                "inactive_start_ratio_mean": float(mal["inactive_start_ratio"].mean()),
                "inactive_start_ratio_median": float(mal["inactive_start_ratio"].median()),
                "detect_minus_peak_days_mean": float(mal["detect_minus_peak_days"].mean()),
                "detect_minus_peak_days_median": float(mal["detect_minus_peak_days"].median()),
            }
        ]
    )

    inactive_start_month_dist = (
        mal.groupby("inactive_start_month_utc", as_index=False)
        .agg(n=("contract_address", "count"))
        .sort_values("inactive_start_month_utc", kind="mergesort")
        .reset_index(drop=True)
    )

    details_out = mal[
        [
            "contract_address",
            "collection_type",
            "first_timestamp",
            "last_timestamp",
            "detected_time",
            "lag_seconds",
            "lag_hours",
            "lag_days",
            "active_days_before_inactive",
            "inactive_start_ratio",
            "peak_active_time",
            "peak_daily_tx_count",
            "tx_count_before_detect",
            "detect_minus_peak_seconds",
            "detect_minus_peak_days",
            "inactive_start_date_utc",
            "inactive_start_month_utc",
        ]
    ].copy()

    details_out.to_csv(OUTPUT_DIR / "malicious_detect_minus_lasttx_details.csv", index=False, encoding="utf-8-sig")
    by_type.to_csv(OUTPUT_DIR / "malicious_detect_minus_lasttx_by_type.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUTPUT_DIR / "malicious_detect_minus_lasttx_overall.csv", index=False, encoding="utf-8-sig")
    inactive_start_month_dist.to_csv(OUTPUT_DIR / "malicious_inactive_start_month_distribution.csv", index=False, encoding="utf-8-sig")

    row = overall.iloc[0]
    print(f"[Types] {'|'.join(malicious_types)}")
    print(f"[N] {int(row['n'])}")
    print(
        "[Average lag] "
        f"{row['lag_mean_seconds']:.2f} sec | "
        f"{row['lag_mean_hours']:.4f} h | "
        f"{row['lag_mean_days']:.6f} days"
    )
    print(
        "[Median lag] "
        f"{row['lag_median_seconds']:.2f} sec | "
        f"{row['lag_median_hours']:.4f} h | "
        f"{row['lag_median_days']:.6f} days"
    )
    print(
        "[Inactive start timing] "
        f"active_before_inactive_mean={row['active_days_before_inactive_mean']:.4f} days, "
        f"active_before_inactive_median={row['active_days_before_inactive_median']:.4f} days, "
        f"inactive_start_ratio_mean={row['inactive_start_ratio_mean']:.4f}, "
        f"inactive_start_ratio_median={row['inactive_start_ratio_median']:.4f}"
    )
    print(
        "[Report - last peak active time] "
        f"mean={row['detect_minus_peak_days_mean']:.4f} days, "
        f"median={row['detect_minus_peak_days_median']:.4f} days"
    )
    print(f"[Saved] {(OUTPUT_DIR / 'malicious_detect_minus_lasttx_overall.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'malicious_detect_minus_lasttx_by_type.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'malicious_detect_minus_lasttx_details.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'malicious_inactive_start_month_distribution.csv').resolve()}")


if __name__ == "__main__":
    main()
