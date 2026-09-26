#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================
INPUT_PANEL_CSV = Path("output/D/7_agent_panel/contract_window_panel.csv")
INPUT_DECISIONS_CSV = Path("output/D/8_agent_decisions/agent_decisions.csv")
OUTPUT_DIR = Path("output/D/9_agent_returns")


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def first_valid_price(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.iloc[0]) if not x.empty else np.nan


def last_valid_price(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.iloc[-1]) if not x.empty else np.nan


def terminal_exit_row(group: pd.DataFrame) -> pd.Series:
    g = group.copy()
    detected_rows = g[pd.notna(g["detected_block_number"]) & (pd.to_numeric(g["end_block"], errors="coerce") >= pd.to_numeric(g["detected_block_number"], errors="coerce"))]
    if not detected_rows.empty:
        return detected_rows.iloc[0]
    return g.iloc[-1]


def evaluate_one_contract(group: pd.DataFrame) -> list[dict]:
    g = group.sort_values(["snapshot_idx"], kind="mergesort").reset_index(drop=True)
    results: list[dict] = []
    decision_mode = str(g["decision_mode"].iloc[0]) if "decision_mode" in g.columns else "rule_agent"
    holding_state = str(g["holding_state"].iloc[0]) if "holding_state" in g.columns else "flat"

    terminal_row = terminal_exit_row(g)
    baseline_exit_price = first_valid_price(pd.Series([terminal_row.get("window_last_trade_price_usd"), terminal_row.get("next_window_first_trade_price_usd")]))
    baseline_entry_candidates = g[pd.to_numeric(g["window_last_trade_exists"], errors="coerce").fillna(0).astype(int) == 1].copy()
    baseline_entry_row = baseline_entry_candidates.iloc[0] if not baseline_entry_candidates.empty else None
    baseline_entry_price = np.nan if baseline_entry_row is None else float(pd.to_numeric(pd.Series([baseline_entry_row.get("window_last_trade_price_usd")]), errors="coerce").dropna().iloc[0])

    sell_rows = g[g["decision"] == "SELL"].copy()
    avoid_rows = g[g["decision"] == "AVOID_BUY"].copy()
    total_tokens = float(pd.to_numeric(g.get("total_tokens"), errors="coerce").fillna(0).sum()) if "total_tokens" in g.columns else 0.0
    prompt_tokens = float(pd.to_numeric(g.get("prompt_tokens"), errors="coerce").fillna(0).sum()) if "prompt_tokens" in g.columns else 0.0
    completion_tokens = float(pd.to_numeric(g.get("completion_tokens"), errors="coerce").fillna(0).sum()) if "completion_tokens" in g.columns else 0.0
    mean_latency_ms = float(pd.to_numeric(g.get("latency_ms"), errors="coerce").dropna().mean()) if "latency_ms" in g.columns else np.nan

    if holding_state == "holding":
        if not sell_rows.empty:
            sell_row = sell_rows.iloc[0]
            agent_exit_price = first_valid_price(pd.Series([sell_row.get("next_window_first_trade_price_usd"), sell_row.get("window_last_trade_price_usd")]))
            holding_loss_avoided = agent_exit_price - baseline_exit_price if np.isfinite(agent_exit_price) and np.isfinite(baseline_exit_price) else np.nan
            results.append(
                {
                    "decision_mode": decision_mode,
                    "scenario": "already_holding",
                    "contract_id": str(g["contract_id"].iloc[0]),
                    "contract_address": str(g["contract_address"].iloc[0]),
                    "collection_type": str(g["collection_type"].iloc[0]),
                    "low_rep_threshold_pct": float(g["low_rep_threshold_pct"].iloc[0]),
                    "trigger_k": int(g["trigger_k"].iloc[0]),
                    "decision_window_idx": int(sell_row["snapshot_idx"]),
                    "lead_windows": pd.to_numeric(sell_row.get("lead_windows_to_detect"), errors="coerce"),
                    "agent_decision": "SELL",
                    "baseline_exit_price_usd": baseline_exit_price,
                    "agent_exit_price_usd": agent_exit_price,
                    "holding_loss_avoided_usd": holding_loss_avoided,
                    "buy_loss_avoided_usd": np.nan,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "mean_latency_ms": mean_latency_ms,
                }
            )
        else:
            results.append(
                {
                    "decision_mode": decision_mode,
                    "scenario": "already_holding",
                    "contract_id": str(g["contract_id"].iloc[0]),
                    "contract_address": str(g["contract_address"].iloc[0]),
                    "collection_type": str(g["collection_type"].iloc[0]),
                    "low_rep_threshold_pct": float(g["low_rep_threshold_pct"].iloc[0]),
                    "trigger_k": int(g["trigger_k"].iloc[0]),
                    "decision_window_idx": np.nan,
                    "lead_windows": np.nan,
                    "agent_decision": "HOLD",
                    "baseline_exit_price_usd": baseline_exit_price,
                    "agent_exit_price_usd": baseline_exit_price,
                    "holding_loss_avoided_usd": 0.0 if np.isfinite(baseline_exit_price) else np.nan,
                    "buy_loss_avoided_usd": np.nan,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "mean_latency_ms": mean_latency_ms,
                }
            )
    else:
        if baseline_entry_row is not None and not avoid_rows.empty:
            avoid_before_entry = avoid_rows[pd.to_numeric(avoid_rows["snapshot_idx"], errors="coerce") <= int(baseline_entry_row["snapshot_idx"])].copy()
            if not avoid_before_entry.empty and np.isfinite(baseline_entry_price) and np.isfinite(baseline_exit_price):
                avoid_row = avoid_before_entry.iloc[0]
                buy_loss_avoided = baseline_entry_price - baseline_exit_price
                results.append(
                    {
                        "decision_mode": decision_mode,
                        "scenario": "not_holding",
                        "contract_id": str(g["contract_id"].iloc[0]),
                        "contract_address": str(g["contract_address"].iloc[0]),
                        "collection_type": str(g["collection_type"].iloc[0]),
                        "low_rep_threshold_pct": float(g["low_rep_threshold_pct"].iloc[0]),
                        "trigger_k": int(g["trigger_k"].iloc[0]),
                        "decision_window_idx": int(avoid_row["snapshot_idx"]),
                        "lead_windows": pd.to_numeric(avoid_row.get("lead_windows_to_detect"), errors="coerce"),
                        "agent_decision": "AVOID_BUY",
                        "baseline_entry_price_usd": baseline_entry_price,
                        "baseline_exit_price_usd": baseline_exit_price,
                        "agent_exit_price_usd": np.nan,
                        "holding_loss_avoided_usd": np.nan,
                        "buy_loss_avoided_usd": buy_loss_avoided,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "mean_latency_ms": mean_latency_ms,
                    }
                )
            else:
                results.append(
                    {
                        "decision_mode": decision_mode,
                        "scenario": "not_holding",
                        "contract_id": str(g["contract_id"].iloc[0]),
                        "contract_address": str(g["contract_address"].iloc[0]),
                        "collection_type": str(g["collection_type"].iloc[0]),
                        "low_rep_threshold_pct": float(g["low_rep_threshold_pct"].iloc[0]),
                        "trigger_k": int(g["trigger_k"].iloc[0]),
                        "decision_window_idx": np.nan,
                        "lead_windows": np.nan,
                        "agent_decision": "ALLOW_BUY",
                        "baseline_entry_price_usd": baseline_entry_price,
                        "baseline_exit_price_usd": baseline_exit_price,
                        "agent_exit_price_usd": baseline_exit_price,
                        "holding_loss_avoided_usd": np.nan,
                        "buy_loss_avoided_usd": 0.0 if np.isfinite(baseline_entry_price) and np.isfinite(baseline_exit_price) else np.nan,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "mean_latency_ms": mean_latency_ms,
                    }
                )
        else:
            results.append(
                {
                    "decision_mode": decision_mode,
                    "scenario": "not_holding",
                    "contract_id": str(g["contract_id"].iloc[0]),
                    "contract_address": str(g["contract_address"].iloc[0]),
                    "collection_type": str(g["collection_type"].iloc[0]),
                    "low_rep_threshold_pct": float(g["low_rep_threshold_pct"].iloc[0]),
                    "trigger_k": int(g["trigger_k"].iloc[0]),
                    "decision_window_idx": np.nan,
                    "lead_windows": np.nan,
                    "agent_decision": "NO_BASELINE_ENTRY",
                    "baseline_entry_price_usd": baseline_entry_price,
                    "baseline_exit_price_usd": baseline_exit_price,
                    "agent_exit_price_usd": np.nan,
                    "holding_loss_avoided_usd": np.nan,
                    "buy_loss_avoided_usd": np.nan,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "mean_latency_ms": mean_latency_ms,
                }
            )

    return results


def main() -> None:
    ensure_exists(INPUT_PANEL_CSV)
    ensure_exists(INPUT_DECISIONS_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(INPUT_PANEL_CSV, low_memory=False)
    decisions = pd.read_csv(INPUT_DECISIONS_CSV, low_memory=False)
    if panel.empty or decisions.empty:
        raise ValueError("Panel or decisions input is empty")

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
                "confidence",
                "reason",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency_ms",
            ]
        ],
        on=["contract_id", "snapshot_idx", "low_rep_threshold_pct"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Merged panel and decisions are empty; check thresholds and snapshot keys")
    if "holding_state_y" in merged.columns and "holding_state" not in merged.columns:
        merged = merged.rename(columns={"holding_state_y": "holding_state"})
    if "holding_state_x" in merged.columns:
        merged = merged.drop(columns=["holding_state_x"])

    results = []
    for _, group in merged.groupby(["decision_mode", "contract_id", "low_rep_threshold_pct", "trigger_k", "holding_state"], sort=False):
        results.extend(evaluate_one_contract(group))

    returns_df = pd.DataFrame(results)
    summary = (
        returns_df.groupby(["decision_mode", "scenario", "low_rep_threshold_pct", "trigger_k"], as_index=False)
        .agg(
            contracts=("contract_id", "nunique"),
            sell_count=("agent_decision", lambda s: int((s == "SELL").sum())),
            avoid_buy_count=("agent_decision", lambda s: int((s == "AVOID_BUY").sum())),
            mean_holding_loss_avoided_usd=("holding_loss_avoided_usd", "mean"),
            median_holding_loss_avoided_usd=("holding_loss_avoided_usd", "median"),
            mean_buy_loss_avoided_usd=("buy_loss_avoided_usd", "mean"),
            median_buy_loss_avoided_usd=("buy_loss_avoided_usd", "median"),
            mean_lead_windows=("lead_windows", "mean"),
            median_lead_windows=("lead_windows", "median"),
            mean_total_tokens=("total_tokens", "mean"),
            total_tokens=("total_tokens", "sum"),
            mean_latency_ms=("mean_latency_ms", "mean"),
        )
        .sort_values(["decision_mode", "scenario", "low_rep_threshold_pct", "trigger_k"], kind="mergesort")
        .reset_index(drop=True)
    )

    returns_df.to_csv(OUTPUT_DIR / "position_log.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "return_summary.csv", index=False, encoding="utf-8-sig")

    lead_curve = (
        returns_df[pd.notna(returns_df["lead_windows"])].groupby(["decision_mode", "scenario", "trigger_k"], as_index=False).agg(
            mean_lead_windows=("lead_windows", "mean"),
            mean_holding_loss_avoided_usd=("holding_loss_avoided_usd", "mean"),
            mean_buy_loss_avoided_usd=("buy_loss_avoided_usd", "mean"),
            mean_total_tokens=("total_tokens", "mean"),
        )
    )
    lead_curve.to_csv(OUTPUT_DIR / "lead_window_return_curve.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'position_log.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'return_summary.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'lead_window_return_curve.csv').resolve()}")


if __name__ == "__main__":
    main()
