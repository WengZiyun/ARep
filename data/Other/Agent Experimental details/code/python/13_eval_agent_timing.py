#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PANEL_CSV = Path("output/D/7_agent_panel/contract_window_panel.csv")
INPUT_DECISIONS_CSV = Path("output/D/8_agent_decisions/agent_decisions.csv")
OUTPUT_DIR = Path("output/D/13_agent_timing")
MALICIOUS_TYPES = {"rugpull", "counterfeit", "wash"}


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def compute_consecutive(group: pd.DataFrame, action: str, first_idx: int) -> int:
    g = group.sort_values("snapshot_idx", kind="mergesort").reset_index(drop=True)
    sub = g[pd.to_numeric(g["snapshot_idx"], errors="coerce") >= first_idx].copy()
    count = 0
    for row in sub.itertuples(index=False):
        if str(row.decision) == action:
            count += 1
        else:
            break
    return count


def evaluate_one(group: pd.DataFrame, angle: str) -> dict:
    g = group.sort_values("snapshot_idx", kind="mergesort").reset_index(drop=True)
    action = "SELL" if angle == "sell" else "AVOID_BUY"
    risk_rows = g[g["decision"] == action].copy()
    has_signal = int(not risk_rows.empty)
    first_row = risk_rows.iloc[0] if has_signal else None

    has_buy_event = int((pd.to_numeric(g["window_last_trade_exists"], errors="coerce").fillna(0) == 1).any())
    never_bought_before_detect = 1 if has_buy_event == 0 else 0

    if has_signal:
        first_risk_window = int(first_row["snapshot_idx"])
        lead_windows = pd.to_numeric(first_row.get("lead_windows_to_detect"), errors="coerce")
        consecutive_risk_windows = compute_consecutive(g, action=action, first_idx=first_risk_window)
    else:
        first_risk_window = np.nan
        lead_windows = np.nan
        consecutive_risk_windows = 0

    never_bought_and_no_avoid_signal = 1 if angle == "buy" and never_bought_before_detect and not has_signal else 0

    return {
        "decision_mode": str(g["decision_mode"].iloc[0]),
        "contract_id": str(g["contract_id"].iloc[0]),
        "collection_type": str(g["collection_type"].iloc[0]),
        "low_rep_threshold_pct": float(g["low_rep_threshold_pct"].iloc[0]),
        "trigger_k": int(g["trigger_k"].iloc[0]),
        "angle": angle,
        "detected_window_proxy": float(pd.to_numeric(g["snapshot_idx"], errors="coerce").max()),
        "first_sell_window": first_risk_window if angle == "sell" else np.nan,
        "first_avoid_buy_window": first_risk_window if angle == "buy" else np.nan,
        "first_risk_window": first_risk_window,
        "lead_windows": lead_windows,
        "consecutive_risk_windows": int(consecutive_risk_windows),
        "has_risk_signal_before_detect": int(has_signal),
        "never_bought_before_detect": int(never_bought_before_detect),
        "never_bought_and_no_avoid_signal": int(never_bought_and_no_avoid_signal),
        "prompt_tokens_total": float(pd.to_numeric(g.get("prompt_tokens"), errors="coerce").fillna(0).sum()) if "prompt_tokens" in g.columns else 0.0,
        "completion_tokens_total": float(pd.to_numeric(g.get("completion_tokens"), errors="coerce").fillna(0).sum()) if "completion_tokens" in g.columns else 0.0,
        "total_tokens_total": float(pd.to_numeric(g.get("total_tokens"), errors="coerce").fillna(0).sum()) if "total_tokens" in g.columns else 0.0,
    }


def main() -> None:
    ensure_exists(INPUT_PANEL_CSV)
    ensure_exists(INPUT_DECISIONS_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(INPUT_PANEL_CSV, low_memory=False)
    decisions = pd.read_csv(INPUT_DECISIONS_CSV, low_memory=False)

    panel = panel[panel["collection_type"].astype(str).isin(MALICIOUS_TYPES)].copy()
    decisions = decisions[decisions["collection_type"].astype(str).isin(MALICIOUS_TYPES)].copy()
    if panel.empty or decisions.empty:
        raise ValueError("No malicious rows found in panel or decisions")

    merged = panel.merge(
        decisions[
            [
                "decision_mode",
                "contract_id",
                "snapshot_idx",
                "low_rep_threshold_pct",
                "trigger_k",
                "holding_state",
                "decision",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ]
        ],
        on=["contract_id", "snapshot_idx", "low_rep_threshold_pct"],
        how="inner",
        suffixes=("_panel", ""),
    )
    if "holding_state_panel" in merged.columns:
        merged = merged.drop(columns=["holding_state_panel"])
    if merged.empty:
        raise ValueError("Merged panel and decisions are empty for malicious rows")

    rows = []
    for _, group in merged.groupby(["decision_mode", "contract_id", "low_rep_threshold_pct", "trigger_k", "holding_state"], sort=False):
        angle = "sell" if str(group["holding_state"].iloc[0]) == "holding" else "buy"
        rows.append(evaluate_one(group, angle=angle))

    timing = pd.DataFrame(rows)
    summary = (
        timing.groupby(["decision_mode", "collection_type", "angle"], as_index=False)
        .agg(
            contracts=("contract_id", "nunique"),
            signal_coverage=("has_risk_signal_before_detect", "mean"),
            mean_lead_windows=("lead_windows", "mean"),
            median_lead_windows=("lead_windows", "median"),
            mean_consecutive_risk_windows=("consecutive_risk_windows", "mean"),
            no_signal_rate=("has_risk_signal_before_detect", lambda s: 1.0 - float(pd.Series(s).mean())),
            never_bought_no_signal_rate=("never_bought_and_no_avoid_signal", "mean"),
            mean_total_tokens=("total_tokens_total", "mean"),
        )
        .sort_values(["collection_type", "angle", "decision_mode"], kind="mergesort")
        .reset_index(drop=True)
    )

    timing.to_csv(OUTPUT_DIR / "timing_contract_level.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "timing_summary.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'timing_contract_level.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'timing_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
