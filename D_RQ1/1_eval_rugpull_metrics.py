#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Config (edit here, no argparse)
# ============================================================
INPUT_LABEL_CSV = Path("output/C/3_collectionlabel.csv")
INPUT_CONTRACTS_DIR = Path("output/D/0_rugpullTRP_single")
OUTPUT_DIR = Path("output/D/1_eval_rugpull_metrics_single")

# Cohort mode:
# For each run: sample 1 rugpull target + trust 20 + unknown 20, repeat 10 times.
COHORT_REPEAT_N = 1
COHORT_RUGPULL_PER_RUN = 1  # fixed to 1 by design
FIXED_TARGET_ADDRESSES = {"0x4923017f3b7fac4e096b46e401c0662f0b7e393f"}  # empty set => random sample

RUGPULL_SAMPLE_N = 1
TRUST_N = 20
UNKNOWN_N = 20
COHORT_RANDOM_SEED = 20260226

TOPK_LIST = [1, 3, 5, 10]
DELTA_HOURS = [1, 24, 168]  # 1h / 1d / 7d
TRACK_END_EXTRA_HOURS = 24  # detected + 1 day
SCORE_MODE = "one_minus_minmax_cC"
HOURLY_STEP_HOURS = 1
END_BLOCK_CAP = 19777901
USE_EXPECTED_HOURLY_GRID = False  # set True only when contracts_{tau} are produced hourly without downsampling

LOW_COVERAGE_MIN_POINTS = 5
FIG_DPI = 140


def safe_name(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text))
    return out.strip("_") or "unknown"


def parse_tau_from_filename(path: Path) -> int | None:
    m = re.match(r"^contracts_(\d+)\.csv$", path.name)
    if not m:
        return None
    return int(m.group(1))


def load_contract_time_slices(input_dir: Path) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    files = sorted(input_dir.glob("contracts_*.csv"), key=lambda p: parse_tau_from_filename(p) or -1)
    if not files:
        raise FileNotFoundError(f"No contracts_*.csv in {input_dir.resolve()}")

    slices: dict[int, pd.DataFrame] = {}
    rows = []
    for fp in files:
        tau = parse_tau_from_filename(fp)
        if tau is None:
            continue
        df = pd.read_csv(fp)
        need = {"contract_address", "contract_id", "cC"}
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"Missing columns in {fp.name}: {miss}")
        sub = df.copy()
        sub["contract_address"] = sub["contract_address"].fillna("").astype(str).str.lower()
        sub["contract_id"] = sub["contract_id"].fillna("").astype(str)
        sub["cC"] = pd.to_numeric(sub["cC"], errors="coerce")

        if SCORE_MODE == "one_minus_minmax_cC":
            cmin = float(sub["cC"].min())
            cmax = float(sub["cC"].max())
            if not np.isfinite(cmin) or not np.isfinite(cmax):
                sub["risk_score"] = np.nan
                score_note = "non_finite_cC"
            elif cmax - cmin <= 1e-18:
                sub["risk_score"] = 0.5
                score_note = "flat_cC_to_0.5"
            else:
                sub["risk_score"] = 1.0 - (sub["cC"] - cmin) / (cmax - cmin)
                score_note = ""
        else:
            raise ValueError("Unsupported SCORE_MODE")

        slices[tau] = sub
        rows.append(
            {
                "tau": tau,
                "file": fp.name,
                "n_contracts": int(len(sub)),
                "score_note": score_note,
            }
        )
    index_df = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    return slices, index_df


def estimate_blocks_per_second(label_df: pd.DataFrame) -> float:
    work = label_df.copy()
    work["create_block_number"] = pd.to_numeric(work["create_block_number"], errors="coerce")
    work["max_block_number"] = pd.to_numeric(work["max_block_number"], errors="coerce")
    work["first_timestamp"] = pd.to_datetime(work["first_timestamp"], errors="coerce", utc=True)
    work["last_timestamp"] = pd.to_datetime(work["last_timestamp"], errors="coerce", utc=True)
    work = work.dropna(subset=["create_block_number", "max_block_number", "first_timestamp", "last_timestamp"]).copy()
    if work.empty:
        raise ValueError("Cannot estimate block speed from label file (no valid rows).")

    block_span = (work["max_block_number"] - work["create_block_number"]).astype(float)
    sec_span = (work["last_timestamp"] - work["first_timestamp"]).dt.total_seconds().astype(float)
    valid = (block_span > 0) & (sec_span > 0)
    work = work[valid].copy()
    if work.empty:
        raise ValueError("Cannot estimate block speed: no positive block/time span rows.")
    bps = (work["max_block_number"] - work["create_block_number"]) / (
        work["last_timestamp"] - work["first_timestamp"]
    ).dt.total_seconds()
    bps = pd.to_numeric(bps, errors="coerce").dropna()
    if bps.empty:
        raise ValueError("Cannot estimate block speed: empty bps series.")
    return float(np.median(bps.values))


def build_cohorts(label_df: pd.DataFrame, repeat_n: int) -> tuple[pd.DataFrame, list[dict]]:
    work = label_df.copy()
    work["collection_type"] = work["collection_type"].astype(str).str.lower()
    work["contract_address"] = work["contract_address"].astype(str).str.lower()
    work["create_block_number"] = pd.to_numeric(work["create_block_number"], errors="coerce")
    work["detected_block_number"] = pd.to_numeric(work["detected_block_number"], errors="coerce")

    rugs = work[work["collection_type"] == "rugpull"].dropna(subset=["create_block_number", "detected_block_number"]).copy()
    trusts = work[work["collection_type"] == "trust"]["contract_address"].dropna().astype(str).str.lower().unique().tolist()
    unknowns = work[work["collection_type"] == "unknown"]["contract_address"].dropna().astype(str).str.lower().unique().tolist()
    create_map = (
        work.dropna(subset=["create_block_number"])
        .drop_duplicates(subset=["contract_address"], keep="first")
        .set_index("contract_address")["create_block_number"]
        .to_dict()
    )

    if rugs.empty:
        raise ValueError("No rugpull rows with valid create/detected blocks.")
    rng = random.Random(COHORT_RANDOM_SEED)
    rug_rows = rugs.to_dict("records")
    rng.shuffle(rug_rows)
    if FIXED_TARGET_ADDRESSES:
        fixed = {str(x).lower() for x in FIXED_TARGET_ADDRESSES}
        target_rows = [r for r in rug_rows if str(r.get("contract_address", "")).lower() in fixed]
        if not target_rows:
            raise ValueError("FIXED_TARGET_ADDRESSES set, but none found in rugpull label rows.")
        target_rows = target_rows[: min(repeat_n, len(target_rows))]
    else:
        target_rows = rug_rows[: min(repeat_n, len(rug_rows))]

    cohort_rows = []
    cohorts = []
    for row in target_rows:
        target_addr = str(row["contract_address"]).lower()
        target_create = int(row["create_block_number"])
        target_detect = int(row["detected_block_number"])

        trust_pool = [x for x in trusts if x != target_addr]
        unknown_pool = [x for x in unknowns if x != target_addr]
        rng.shuffle(trust_pool)
        rng.shuffle(unknown_pool)

        trust_pick = trust_pool[: min(TRUST_N, len(trust_pool))]
        unknown_pick = unknown_pool[: min(UNKNOWN_N, len(unknown_pool))]
        cohort_addrs = [target_addr] + trust_pick + unknown_pick
        member_creates = [int(create_map.get(a, target_create)) for a in cohort_addrs]
        start_block = int(min(member_creates)) if member_creates else int(target_create)

        cohort_id = safe_name(f"{target_addr}_{target_detect}")
        cohorts.append(
            {
                "cohort_id": cohort_id,
                "target_address": target_addr,
                "target_create_block": target_create,
                "target_detect_block": target_detect,
                "start_block": start_block,
                "trust_list": trust_pick,
                "unknown_list": unknown_pick,
                "cohort_addrs": set(cohort_addrs),
            }
        )

        for addr in [target_addr]:
            cohort_rows.append(
                {
                    "cohort_id": cohort_id,
                    "target_address": target_addr,
                    "member_address": addr,
                    "member_type": "target_rugpull",
                    "target_create_block": target_create,
                    "target_detect_block": target_detect,
                    "start_block": start_block,
                    "desired_trust_n": TRUST_N,
                    "desired_unknown_n": UNKNOWN_N,
                    "actual_trust_n": len(trust_pick),
                    "actual_unknown_n": len(unknown_pick),
                    "seed": COHORT_RANDOM_SEED,
                }
            )
        for addr in trust_pick:
            cohort_rows.append(
                {
                    "cohort_id": cohort_id,
                    "target_address": target_addr,
                    "member_address": addr,
                    "member_type": "trust",
                    "target_create_block": target_create,
                    "target_detect_block": target_detect,
                    "start_block": start_block,
                    "desired_trust_n": TRUST_N,
                    "desired_unknown_n": UNKNOWN_N,
                    "actual_trust_n": len(trust_pick),
                    "actual_unknown_n": len(unknown_pick),
                    "seed": COHORT_RANDOM_SEED,
                }
            )
        for addr in unknown_pick:
            cohort_rows.append(
                {
                    "cohort_id": cohort_id,
                    "target_address": target_addr,
                    "member_address": addr,
                    "member_type": "unknown",
                    "target_create_block": target_create,
                    "target_detect_block": target_detect,
                    "start_block": start_block,
                    "desired_trust_n": TRUST_N,
                    "desired_unknown_n": UNKNOWN_N,
                    "actual_trust_n": len(trust_pick),
                    "actual_unknown_n": len(unknown_pick),
                    "seed": COHORT_RANDOM_SEED,
                }
            )
    return pd.DataFrame(cohort_rows), cohorts


def evaluate_panel(
    slices: dict[int, pd.DataFrame],
    cohorts: list[dict],
    delta_blocks: dict[str, int],
    step_blocks: int,
    end_block_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    taus_available = sorted(slices.keys())
    tau_set = set(taus_available)
    panel_rows = []
    missing_rows = []

    for c in cohorts:
        cohort_id = c["cohort_id"]
        target = c["target_address"]
        detect = int(c["target_detect_block"])
        start = int(c["start_block"])
        target_create = int(c["target_create_block"])
        end = int(min(detect + delta_blocks["1d"], int(end_block_cap)))
        addrs = c["cohort_addrs"]
        if target_create > end:
            missing_rows.append({"cohort_id": cohort_id, "target_address": target, "tau": target_create, "reason": "target_create_above_end_cap"})
            continue

        if USE_EXPECTED_HOURLY_GRID:
            expected_taus = list(range(target_create, end + 1, max(1, int(step_blocks))))
            if expected_taus[-1] != end:
                expected_taus.append(end)
        else:
            expected_taus = [t for t in taus_available if target_create <= t <= end]
            if not expected_taus:
                missing_rows.append({"cohort_id": cohort_id, "target_address": target, "tau": target_create, "reason": "no_available_taus_in_range"})
                continue

        for tau in expected_taus:
            if tau not in tau_set:
                missing_rows.append({"cohort_id": cohort_id, "target_address": target, "tau": tau, "reason": "contracts_slice_missing"})
                continue
            s = slices[tau]
            sub = s[s["contract_address"].isin(addrs)].copy()
            if sub.empty:
                missing_rows.append({"cohort_id": cohort_id, "target_address": target, "tau": tau, "reason": "empty_subset"})
                continue
            if not sub["contract_address"].eq(target).any():
                missing_rows.append({"cohort_id": cohort_id, "target_address": target, "tau": tau, "reason": "target_missing"})
                continue

            sub = sub.sort_values(["risk_score", "contract_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
            sub["rank"] = np.arange(1, len(sub) + 1)
            trow = sub[sub["contract_address"] == target].iloc[0]
            rank = int(trow["rank"])
            risk_score = float(trow["risk_score"]) if pd.notna(trow["risk_score"]) else np.nan

            y_map = {}
            for k, db in delta_blocks.items():
                y_map[f"y_{k}"] = int((tau < detect) and (detect <= tau + int(db)))

            row = {
                "cohort_id": cohort_id,
                "target_address": target,
                "tau": int(tau),
                "target_detect_block": detect,
                "target_create_block": target_create,
                "start_block": start,
                "track_end_block": end,
                "rank": rank,
                "risk_score": risk_score,
                "n_candidates_in_tau": int(len(sub)),
            }
            row.update(y_map)
            for k in TOPK_LIST:
                row[f"hit_at_{k}"] = int(rank <= int(k))
            panel_rows.append(row)

    panel_df = pd.DataFrame(panel_rows)
    if not panel_df.empty:
        panel_df = panel_df.sort_values(["cohort_id", "tau"]).reset_index(drop=True)
    else:
        panel_df = pd.DataFrame(
            columns=[
                "cohort_id",
                "target_address",
                "tau",
                "target_detect_block",
                "target_create_block",
                "start_block",
                "track_end_block",
                "rank",
                "risk_score",
                "n_candidates_in_tau",
            ]
            + [f"y_{k}" for k in delta_blocks.keys()]
            + [f"hit_at_{k}" for k in TOPK_LIST]
        )
    missing_df = pd.DataFrame(missing_rows)
    if not missing_df.empty:
        missing_df = missing_df.sort_values(["cohort_id", "tau"]).reset_index(drop=True)
    else:
        missing_df = pd.DataFrame(columns=["cohort_id", "target_address", "tau", "reason"])
    return panel_df, missing_df


def summarize(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    delta_keys = [f"{h}h" if h < 24 else ("1d" if h == 24 else "7d") for h in DELTA_HOURS]
    by_target_rows = []
    for (cohort_id, target), g in panel_df.groupby(["cohort_id", "target_address"], sort=True):
        rec = {
            "cohort_id": cohort_id,
            "target_address": target,
            "n_taus": int(len(g)),
            "tau_min": int(g["tau"].min()),
            "tau_max": int(g["tau"].max()),
            "avg_rank": float(g["rank"].mean()),
            "median_rank": float(g["rank"].median()),
            "low_coverage": int(len(g) < LOW_COVERAGE_MIN_POINTS),
        }
        for k in TOPK_LIST:
            rec[f"hit_rate_at_{k}"] = float(g[f"hit_at_{k}"].mean())

        for d in delta_keys:
            ycol = f"y_{d}"
            pos = g[g[ycol] == 1]
            rec[f"pos_steps_{d}"] = int(len(pos))
            for k in TOPK_LIST:
                hcol = f"hit_at_{k}"
                if len(pos) > 0:
                    rec[f"precision_at_{k}_{d}"] = float(pos[hcol].mean())
                    rec[f"recall_at_{k}_{d}"] = float(pos[hcol].mean())  # single-target setting
                else:
                    rec[f"precision_at_{k}_{d}"] = np.nan
                    rec[f"recall_at_{k}_{d}"] = np.nan
        by_target_rows.append(rec)

    by_target = pd.DataFrame(by_target_rows).sort_values(["avg_rank", "target_address"]).reset_index(drop=True)
    num = by_target.select_dtypes(include=[np.number]).columns.tolist()
    overall_stats = by_target[num].agg(["mean", "median", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    overall_stats.insert(0, "n_targets", len(by_target))
    return by_target, overall_stats


def build_rerun_plan_rows(cohorts: list[dict], delta_blocks: dict[str, int], step_blocks: int, end_block_cap: int) -> pd.DataFrame:
    rows = []
    for c in cohorts:
        start_block = int(c["start_block"])
        target_create = int(c["target_create_block"])
        detect = int(c["target_detect_block"])
        end_block = int(min(detect + delta_blocks["1d"], int(end_block_cap)))
        if target_create > end_block:
            continue
        ends = list(range(target_create, end_block + 1, max(1, int(step_blocks))))
        if ends[-1] != end_block:
            ends.append(end_block)
        for i, e in enumerate(ends, 1):
            rows.append(
                {
                    "cohort_id": c["cohort_id"],
                    "target_address": c["target_address"],
                    "start_block": start_block,
                    "end_block": int(e),
                    "target_create_block": target_create,
                    "target_detect_block": detect,
                    "track_end_block": end_block,
                    "step_idx": i,
                }
            )
    return pd.DataFrame(rows)


def plot_rank_curves(panel_df: pd.DataFrame, by_target: pd.DataFrame) -> None:
    if panel_df.empty:
        return
    for row in by_target.itertuples(index=False):
        cohort_id = row.cohort_id
        target = row.target_address
        g = panel_df[(panel_df["cohort_id"] == cohort_id) & (panel_df["target_address"] == target)].sort_values("tau")
        if g.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.plot(g["tau"], g["rank"], linewidth=2.0, color="#0b6")
        ax.scatter(g["tau"], g["rank"], s=14, color="#0b6", alpha=0.75)

        c0 = int(g["target_create_block"].iloc[0])
        d0 = int(g["target_detect_block"].iloc[0])
        e0 = int(g["track_end_block"].iloc[0])
        ax.axvline(c0, color="#666", linestyle="--", linewidth=1.2, label="create_block")
        ax.axvline(d0, color="#d33", linestyle="--", linewidth=1.2, label="detected_block")
        ax.axvline(e0, color="#36c", linestyle="--", linewidth=1.2, label="detected+1d")

        ymax = max(1, int(g["n_candidates_in_tau"].max()))
        ax.set_ylim(ymax + 1, 0.5)  # rank 1 on top
        ax.set_xlabel("tau (block)")
        ax.set_ylabel("rank (1 is best)")
        ax.set_title(f"Rugpull Rank Curve: {target[:10]}...{target[-6:]}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        out = OUTPUT_DIR / f"rank_curve_{safe_name(target)}.png"
        fig.savefig(out, dpi=FIG_DPI)
        plt.close(fig)


def plot_hit_rate_by_delta(panel_df: pd.DataFrame) -> None:
    if panel_df.empty:
        return
    delta_keys = [f"{h}h" if h < 24 else ("1d" if h == 24 else "7d") for h in DELTA_HOURS]
    rows = []
    for d in delta_keys:
        ycol = f"y_{d}"
        pos = panel_df[panel_df[ycol] == 1]
        for k in TOPK_LIST:
            if len(pos) == 0:
                v = np.nan
            else:
                v = float(pos[f"hit_at_{k}"].mean())
            rows.append({"delta": d, "K": k, "hit_rate": v})
    m = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    width = 0.22
    x = np.arange(len(TOPK_LIST), dtype=float)
    for i, d in enumerate(delta_keys):
        vals = m[m["delta"] == d].sort_values("K")["hit_rate"].to_numpy()
        ax.bar(x + (i - (len(delta_keys) - 1) / 2.0) * width, vals, width=width, label=d)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in TOPK_LIST])
    ax.set_xlabel("K")
    ax.set_ylabel("Positive-step hit rate")
    ax.set_title("Hit Rate by Delta Window")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hit_rate_by_delta.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_avg_rank_over_time(panel_df: pd.DataFrame) -> None:
    if panel_df.empty:
        return
    g = panel_df.groupby("tau", as_index=False).agg(avg_rank=("rank", "mean"), std_rank=("rank", "std"), n=("rank", "size"))
    g["std_rank"] = g["std_rank"].fillna(0.0)
    g["se"] = g["std_rank"] / np.sqrt(np.maximum(g["n"], 1))
    g["low"] = g["avg_rank"] - 1.96 * g["se"]
    g["high"] = g["avg_rank"] + 1.96 * g["se"]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(g["tau"], g["avg_rank"], color="#06c", linewidth=2.0, label="avg rank")
    ax.fill_between(g["tau"], g["low"], g["high"], color="#06c", alpha=0.2, label="95% CI")
    ymax = max(1.0, float((panel_df["n_candidates_in_tau"]).max()))
    ax.set_ylim(ymax + 1, 0.5)
    ax.set_xlabel("tau (block)")
    ax.set_ylabel("average rank (1 is best)")
    ax.set_title("Average Rugpull Rank Over Time")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "avg_rank_over_time.png", dpi=FIG_DPI)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_LABEL_CSV.exists():
        raise FileNotFoundError(f"Label file not found: {INPUT_LABEL_CSV.resolve()}")

    label_df = pd.read_csv(INPUT_LABEL_CSV)
    try:
        slices, slice_index = load_contract_time_slices(INPUT_CONTRACTS_DIR)
    except FileNotFoundError:
        slices, slice_index = {}, pd.DataFrame(columns=["tau", "file", "n_contracts", "score_note"])
    slice_index.to_csv(OUTPUT_DIR / "contracts_slice_index.csv", index=False, encoding="utf-8-sig")

    bps = estimate_blocks_per_second(label_df)
    delta_blocks = {
        "1h": int(round(3600 * bps)),
        "1d": int(round(86400 * bps)),
        "7d": int(round(7 * 86400 * bps)),
    }
    step_blocks = int(round(float(HOURLY_STEP_HOURS) * 3600.0 * bps))
    step_blocks = max(1, step_blocks)

    # Requested evaluation mode: rugpull random 1 each run, repeated N times.
    # Equivalent to N cohorts each with 1 target rugpull + trust 20 + unknown 20.
    repeat_n = int(COHORT_REPEAT_N if COHORT_REPEAT_N > 0 else RUGPULL_SAMPLE_N)
    cohort_df, cohorts = build_cohorts(label_df, repeat_n=repeat_n)
    cohort_df.to_csv(OUTPUT_DIR / "rugpull_cohorts.csv", index=False, encoding="utf-8-sig")

    rerun_plan = build_rerun_plan_rows(
        cohorts=cohorts, delta_blocks=delta_blocks, step_blocks=step_blocks, end_block_cap=END_BLOCK_CAP
    )
    rerun_plan.to_csv(OUTPUT_DIR / "rerun_block_windows_plan.csv", index=False, encoding="utf-8-sig")

    panel_df, missing_df = evaluate_panel(
        slices=slices, cohorts=cohorts, delta_blocks=delta_blocks, step_blocks=step_blocks, end_block_cap=END_BLOCK_CAP
    )
    panel_df.to_csv(OUTPUT_DIR / "rugpull_panel.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(OUTPUT_DIR / "rugpull_missing_taus.csv", index=False, encoding="utf-8-sig")

    by_target, overall = summarize(panel_df)
    by_target.to_csv(OUTPUT_DIR / "rugpull_summary_by_target.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUTPUT_DIR / "rugpull_summary_overall.csv", index=False, encoding="utf-8-sig")

    meta = pd.DataFrame(
        [
            {"key": "blocks_per_second_median", "value": bps},
            {"key": "delta_1h_blocks", "value": delta_blocks["1h"]},
            {"key": "delta_1d_blocks", "value": delta_blocks["1d"]},
            {"key": "delta_7d_blocks", "value": delta_blocks["7d"]},
            {"key": "step_blocks", "value": step_blocks},
            {"key": "hourly_step_hours", "value": HOURLY_STEP_HOURS},
            {"key": "end_block_cap", "value": END_BLOCK_CAP},
            {"key": "n_contract_slices", "value": len(slices)},
            {"key": "n_cohorts", "value": len(cohorts)},
            {"key": "n_rerun_plan_rows", "value": len(rerun_plan)},
            {"key": "n_panel_rows", "value": len(panel_df)},
            {"key": "n_missing_rows", "value": len(missing_df)},
            {"key": "score_mode", "value": SCORE_MODE},
            {"key": "trust_n", "value": TRUST_N},
            {"key": "unknown_n", "value": UNKNOWN_N},
            {"key": "rugpull_sample_n", "value": RUGPULL_SAMPLE_N},
            {"key": "cohort_seed", "value": COHORT_RANDOM_SEED},
        ]
    )
    meta.to_csv(OUTPUT_DIR / "meta.csv", index=False, encoding="utf-8-sig")

    plot_rank_curves(panel_df, by_target)
    plot_hit_rate_by_delta(panel_df)
    plot_avg_rank_over_time(panel_df)

    print(f"[Saved] {(OUTPUT_DIR / 'rugpull_cohorts.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'rugpull_panel.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'rerun_block_windows_plan.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'rugpull_summary_by_target.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'rugpull_summary_overall.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'meta.csv').resolve()}")
    print(f"[Saved] rank curves + hit/avg charts in {OUTPUT_DIR.resolve()}")
    if len(slices) <= 1:
        print("[Warn] Only one contracts_{tau}.csv found; time-series analysis will be limited.")


if __name__ == "__main__":
    main()
