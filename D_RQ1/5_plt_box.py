#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# Config
# ============================================================
COLLECTION_LABEL_CSV = Path("output/C/3_collectionlabel.csv")
WASH_LABEL_CSV = Path("output/B/5_2washcheck/wash_addresses_label_verified_global.csv")
WASH_TX_DIRS = [Path("output/B/2_tranding"), Path("output/B/2_4normalTranding")]
OUTPUT_DIR = Path("output/D/5_split_stats")

CHUNK_SIZE = 200_000


def _normalize_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def _parse_mixed_ts(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601")
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        dt_ms = pd.to_datetime(num.where(num >= 1e12), unit="ms", errors="coerce", utc=True)
        dt_s = pd.to_datetime(num.where((num > 0) & (num < 1e12)), unit="s", errors="coerce", utc=True)
        dt = dt.fillna(dt_ms).fillna(dt_s)
    return dt


def _to_utc_ts(x):
    t = pd.to_datetime(x, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    # normalize to UTC-aware for safe comparison
    if getattr(t, "tzinfo", None) is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _metric_stats(s: pd.Series, prefix: str) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {
            f"{prefix}_mean_days": float("nan"),
            f"{prefix}_median_days": float("nan"),
            f"{prefix}_p25_days": float("nan"),
            f"{prefix}_p75_days": float("nan"),
            f"{prefix}_p90_days": float("nan"),
            f"{prefix}_min_days": float("nan"),
            f"{prefix}_max_days": float("nan"),
        }
    return {
        f"{prefix}_mean_days": float(s.mean()),
        f"{prefix}_median_days": float(s.median()),
        f"{prefix}_p25_days": float(s.quantile(0.25)),
        f"{prefix}_p75_days": float(s.quantile(0.75)),
        f"{prefix}_p90_days": float(s.quantile(0.90)),
        f"{prefix}_min_days": float(s.min()),
        f"{prefix}_max_days": float(s.max()),
    }


def summarize_behavior(df: pd.DataFrame, behavior_name: str) -> pd.DataFrame:
    sub = df[df["collection_type"] == behavior_name].copy()
    row = {"behavior": behavior_name, "n": int(len(sub))}
    row.update(_metric_stats(sub["lag_last_tx_days"], "last_tx"))
    row.update(_metric_stats(sub["lag_peak_days"], "peak"))
    return pd.DataFrame([row])


def _last_tx_and_peak_from_ts(ts: pd.Series, detected_time: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    hit = ts[ts <= detected_time]
    if hit.empty:
        return pd.NaT, pd.NaT, 0
    last_tx = hit.max()
    day = hit.dt.floor("D")
    day_counts = day.value_counts()
    peak_cnt = int(day_counts.max())
    peak_days = day_counts[day_counts == peak_cnt].index
    peak_day = peak_days.max()
    peak_ts = hit[day == peak_day].max()
    return last_tx, peak_ts, peak_cnt


def build_rugpull_counterfeit_details() -> pd.DataFrame:
    if not COLLECTION_LABEL_CSV.exists():
        raise FileNotFoundError(f"Missing file: {COLLECTION_LABEL_CSV.resolve()}")
    df = pd.read_csv(COLLECTION_LABEL_CSV, low_memory=False)
    need = ["contract_address", "collection_type", "detected_time", "source_tx_file"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {COLLECTION_LABEL_CSV}: {miss}")

    d = df.copy()
    d["contract_address"] = _normalize_text(d["contract_address"])
    d["collection_type"] = _normalize_text(d["collection_type"])
    d["detected_time"] = _parse_mixed_ts(d["detected_time"])
    d = d[d["collection_type"].isin(["rugpull", "counterfeit"])].copy()
    d = d.dropna(subset=["detected_time", "source_tx_file"]).copy()

    # compute last tx and peak-active time <= detected_time from source tx file
    ts_cache: dict[str, pd.Series] = {}
    last_list, peak_list, peak_cnt_list = [], [], []
    for r in d.itertuples(index=False):
        fp = Path(str(r.source_tx_file))
        det_t = _to_utc_ts(r.detected_time)
        if pd.isna(det_t) or not fp.exists():
            last_list.append(pd.NaT)
            peak_list.append(pd.NaT)
            peak_cnt_list.append(0)
            continue

        key = str(fp.resolve())
        if key not in ts_cache:
            tx = pd.read_csv(fp, usecols=["timestamp"], low_memory=False)
            ts = _parse_mixed_ts(tx["timestamp"]).dropna().sort_values(kind="mergesort").reset_index(drop=True)
            ts_cache[key] = ts
        ts = ts_cache[key]
        last_tx, peak_ts, peak_cnt = _last_tx_and_peak_from_ts(ts, det_t)
        last_list.append(last_tx)
        peak_list.append(peak_ts)
        peak_cnt_list.append(peak_cnt)

    d["last_timestamp"] = last_list
    d["peak_active_time"] = peak_list
    d["peak_daily_tx_count"] = peak_cnt_list
    d = d.dropna(subset=["last_timestamp", "peak_active_time"]).copy()
    d["lag_last_tx_days"] = (d["detected_time"] - d["last_timestamp"]).dt.total_seconds() / 86400.0
    d["lag_peak_days"] = (d["detected_time"] - d["peak_active_time"]).dt.total_seconds() / 86400.0
    return d[
        [
            "contract_address",
            "collection_type",
            "last_timestamp",
            "peak_active_time",
            "peak_daily_tx_count",
            "detected_time",
            "lag_last_tx_days",
            "lag_peak_days",
        ]
    ].copy()


def build_wash_details() -> pd.DataFrame:
    if not WASH_LABEL_CSV.exists():
        raise FileNotFoundError(f"Missing file: {WASH_LABEL_CSV.resolve()}")
    wash = pd.read_csv(WASH_LABEL_CSV, low_memory=False)
    need = ["address", "detected_time"]
    miss = [c for c in need if c not in wash.columns]
    if miss:
        raise ValueError(f"Missing columns in {WASH_LABEL_CSV}: {miss}")

    wash = wash.copy()
    wash["address"] = _normalize_text(wash["address"])
    wash["detected_time"] = pd.to_datetime(wash["detected_time"], errors="coerce", utc=True)
    wash = wash.dropna(subset=["address", "detected_time"]).copy()
    # keep earliest detected time per address
    wash = wash.sort_values(["address", "detected_time"], kind="mergesort").drop_duplicates("address", keep="first")
    wash = wash.reset_index(drop=True)

    det_map = {a: _to_utc_ts(t) for a, t in zip(wash["address"].tolist(), wash["detected_time"].tolist())}
    last_map: dict[str, pd.Timestamp] = {a: pd.NaT for a in det_map.keys()}
    peak_count_map: dict[str, dict[pd.Timestamp, int]] = defaultdict(dict)
    peak_last_ts_map: dict[str, dict[pd.Timestamp, pd.Timestamp]] = defaultdict(dict)

    usecols = ["timestamp", "seller", "buyer"]
    tx_files: list[Path] = []
    for d in WASH_TX_DIRS:
        if d.exists():
            tx_files.extend(sorted(d.glob("*.csv")))
    if not tx_files:
        raise FileNotFoundError("No wash tx files found in configured dirs.")

    wash_set = set(det_map.keys())
    for fp in tx_files:
        for chunk in pd.read_csv(fp, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
            if chunk.empty:
                continue
            chunk["seller"] = _normalize_text(chunk["seller"])
            chunk["buyer"] = _normalize_text(chunk["buyer"])
            ts = _parse_mixed_ts(chunk["timestamp"])

            involved = chunk["seller"].isin(wash_set) | chunk["buyer"].isin(wash_set)
            if not involved.any():
                continue
            c = chunk.loc[involved, ["seller", "buyer"]].copy()
            c["ts"] = ts.loc[involved].values
            c = c.dropna(subset=["ts"]).copy()
            if c.empty:
                continue

            # update per matched address
            for role in ["seller", "buyer"]:
                part = c[[role, "ts"]].rename(columns={role: "addr"}).copy()
                part = part[part["addr"].isin(wash_set)].copy()
                if part.empty:
                    continue
                part["det_t"] = part["addr"].map(det_map)
                part["ts"] = part["ts"].map(_to_utc_ts)
                part = part.dropna(subset=["det_t", "ts"]).copy()
                part = part[part["ts"] <= part["det_t"]].copy()
                if part.empty:
                    continue

                # last tx before detect
                g_last = part.groupby("addr", as_index=False)["ts"].max()
                for r in g_last.itertuples(index=False):
                    addr = str(r.addr)
                    t = _to_utc_ts(r.ts)
                    prev = last_map.get(addr, pd.NaT)
                    if pd.isna(prev) or t > prev:
                        last_map[addr] = t

                # daily activity count before detect for peak day
                part["day"] = part["ts"].dt.floor("D")
                g_day = part.groupby(["addr", "day"], as_index=False).agg(cnt=("ts", "size"), last_ts=("ts", "max"))
                for r in g_day.itertuples(index=False):
                    addr = str(r.addr)
                    day = _to_utc_ts(r.day)
                    cnt = int(r.cnt)
                    last_ts = _to_utc_ts(r.last_ts)
                    prev_cnt = int(peak_count_map[addr].get(day, 0))
                    peak_count_map[addr][day] = prev_cnt + cnt
                    prev_last = peak_last_ts_map[addr].get(day, pd.NaT)
                    if pd.isna(prev_last) or (not pd.isna(last_ts) and last_ts > prev_last):
                        peak_last_ts_map[addr][day] = last_ts

    out = wash[["address", "detected_time"]].copy()
    out["last_timestamp"] = out["address"].map(last_map)
    peak_ts_list, peak_cnt_list = [], []
    for addr in out["address"].tolist():
        day_cnt = peak_count_map.get(addr, {})
        if not day_cnt:
            peak_ts_list.append(pd.NaT)
            peak_cnt_list.append(0)
            continue
        max_cnt = max(day_cnt.values())
        best_days = [d for d, c in day_cnt.items() if c == max_cnt]
        best_day = max(best_days)
        peak_ts_list.append(peak_last_ts_map.get(addr, {}).get(best_day, pd.NaT))
        peak_cnt_list.append(int(max_cnt))
    out["peak_active_time"] = peak_ts_list
    out["peak_daily_tx_count"] = peak_cnt_list
    out["collection_type"] = "wash"
    out["lag_last_tx_days"] = (out["detected_time"] - out["last_timestamp"]).dt.total_seconds() / 86400.0
    out["lag_peak_days"] = (out["detected_time"] - out["peak_active_time"]).dt.total_seconds() / 86400.0
    out = out.dropna(subset=["last_timestamp", "peak_active_time", "lag_last_tx_days", "lag_peak_days"]).copy()
    out = out.rename(columns={"address": "contract_address"})
    return out[
        [
            "contract_address",
            "collection_type",
            "last_timestamp",
            "peak_active_time",
            "peak_daily_tx_count",
            "detected_time",
            "lag_last_tx_days",
            "lag_peak_days",
        ]
    ].copy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rc = build_rugpull_counterfeit_details()
    wash = build_wash_details()

    all_details = pd.concat([rc, wash], ignore_index=True)

    summary = pd.concat(
        [
            summarize_behavior(all_details, "rugpull"),
            summarize_behavior(all_details, "counterfeit"),
            summarize_behavior(all_details, "wash"),
        ],
        ignore_index=True,
    )

    all_details.to_csv(OUTPUT_DIR / "three_behavior_time_lag_details.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "three_behavior_time_lag_summary.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'three_behavior_time_lag_summary.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'three_behavior_time_lag_details.csv').resolve()}")
    for r in summary.itertuples(index=False):
        print(
            f"[{r.behavior}] n={int(r.n)}, "
            f"report-last_tx mean={r.last_tx_mean_days:.4f}d median={r.last_tx_median_days:.4f}d, "
            f"report-peak mean={r.peak_mean_days:.4f}d median={r.peak_median_days:.4f}d"
        )


if __name__ == "__main__":
    main()
