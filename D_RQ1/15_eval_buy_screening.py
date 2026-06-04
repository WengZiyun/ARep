#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd


INPUT_PANEL_CSV = Path("output/D/7_agent_panel/contract_window_panel.csv")
LABEL_CSV = Path("output/C/3_collectionlabel.csv")
OUTPUT_DIR = Path("output/D/15_buy_screening")
SERVER_BASE_URL = "http://127.0.0.1:3000"
PLUGIN_NAME = "nft-risk-plugin"

DECISION_MODES = [
    x.strip()
    for x in os.getenv(
        "BUY_SCREEN_DECISION_MODES",
        "rule_agent,llm_base,llm_base_rank,llm_base_rank_compact,llm_base_rank_mini,llm_rank",
    ).split(",")
    if x.strip()
]
REQUEST_TIMEOUT_SECONDS = int(os.getenv("BUY_SCREEN_REQUEST_TIMEOUT", "90"))
TRIGGER_K = int(os.getenv("BUY_SCREEN_TRIGGER_K", "3"))
THRESHOLD_PCT = float(os.getenv("BUY_SCREEN_THRESHOLD_PCT", "0.8"))
MALICIOUS_SAMPLE_SIZE = int(os.getenv("BUY_SCREEN_MALICIOUS_N", "100"))
GOOD_SAMPLE_SIZE = int(os.getenv("BUY_SCREEN_GOOD_N", "100"))
RANDOM_SEED = int(os.getenv("BUY_SCREEN_RANDOM_SEED", "42"))


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def get_json(url: str, timeout_sec: int) -> dict | list:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def post_json(url: str, payload: dict, timeout_sec: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Agent response is not a JSON object")
    return data


def resolve_agent_endpoint(base_url: str, timeout_sec: int) -> tuple[str, str]:
    agents_url = f"{base_url.rstrip('/')}/api/agents"
    payload = get_json(agents_url, timeout_sec=timeout_sec)

    if isinstance(payload, dict):
        inner = payload.get("data", payload)
        if isinstance(inner, dict) and "agents" in inner:
            candidates = inner.get("agents")
        else:
            candidates = payload.get("agents", inner)
    else:
        candidates = payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"No agents found from {agents_url}")

    first = candidates[0]
    if not isinstance(first, dict):
        raise ValueError("Invalid /api/agents response shape")

    agent_id = str(first.get("id") or first.get("agentId") or first.get("uuid") or "").strip()
    if not agent_id:
        raise ValueError("Could not resolve agent id from /api/agents")

    endpoint = (
        f"{base_url.rstrip('/')}/api/agents/{agent_id}/plugins/"
        f"{urllib.parse.quote(PLUGIN_NAME)}/risk-evaluate?agentId={urllib.parse.quote(agent_id)}"
    )
    return agent_id, endpoint


def build_payload(row: pd.Series, decision_mode: str) -> dict:
    return {
        "decision_mode": str(decision_mode),
        "contract_id": str(row["contract_id"]),
        "contract_address": str(row["contract_address"]),
        "collection_type": str(row["collection_type"]),
        "window_idx": int(row["snapshot_idx"]),
        "start_block": int(row["start_block"]),
        "end_block": int(row["end_block"]),
        "rep_score": float(row["rep_score"]),
        "rep_rank": int(row["rep_rank"]),
        "rep_rank_pct": float(row["rep_rank_pct"]),
        "low_rep_threshold_pct": float(row["low_rep_threshold_pct"]),
        "is_low_rep": int(row["is_low_rep"]),
        "low_rep_streak": int(row["low_rep_streak"]),
        "lead_windows_to_detect": None if pd.isna(row["lead_windows_to_detect"]) else float(row["lead_windows_to_detect"]),
        "holding_state": "flat",
        "can_buy_baseline": int(row["can_buy_baseline"]),
        "window_last_trade_price_usd": None
        if pd.isna(row["window_last_trade_price_usd"])
        else float(row["window_last_trade_price_usd"]),
        "next_window_first_trade_price_usd": None
        if pd.isna(row["next_window_first_trade_price_usd"])
        else float(row["next_window_first_trade_price_usd"]),
        "next_trade_exists": int(row["next_trade_exists"]),
        "trigger_k": int(TRIGGER_K),
        "window_trade_count": float(pd.to_numeric(row["window_trade_count"], errors="coerce") or 0.0),
        "window_trade_value_total_usd": float(pd.to_numeric(row["window_trade_value_total_usd"], errors="coerce") or 0.0),
        "window_unique_buyers": float(pd.to_numeric(row["window_unique_buyers"], errors="coerce") or 0.0),
        "window_unique_trade_users": float(pd.to_numeric(row["window_unique_trade_users"], errors="coerce") or 0.0),
        "window_mint_cnt": float(pd.to_numeric(row["window_mint_cnt"], errors="coerce") or 0.0),
        "current_mint_supply_count": float(pd.to_numeric(row.get("current_mint_supply_count", 0), errors="coerce") or 0.0),
        "window_gas_total_usd": float(pd.to_numeric(row["window_gas_total_usd"], errors="coerce") or 0.0),
        "price_change_vs_prev_window_pct": None
        if pd.isna(row["price_change_vs_prev_window_pct"])
        else float(row["price_change_vs_prev_window_pct"]),
        "trade_count_change_vs_prev_window_pct": None
        if pd.isna(row["trade_count_change_vs_prev_window_pct"])
        else float(row["trade_count_change_vs_prev_window_pct"]),
        "liquidity_flag": int(pd.to_numeric(row["liquidity_flag"], errors="coerce") or 0),
    }


def sample_malicious_rows(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel[
        (panel["collection_type"].astype(str) == "rugpull")
        & (pd.to_numeric(panel["low_rep_threshold_pct"], errors="coerce") == THRESHOLD_PCT)
    ].copy()
    work["lead"] = pd.to_numeric(work["lead_windows_to_detect"], errors="coerce")
    work = work[work["lead"].notna() & (work["lead"] >= 0)].copy()
    if work.empty:
        raise ValueError("No rugpull rows with valid lead_windows_to_detect found")

    # Keep only the single window immediately before detection for each contract.
    work = (
        work.sort_values(["contract_id", "lead", "snapshot_idx"], kind="mergesort")
        .groupby("contract_id", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    return work.sample(n=MALICIOUS_SAMPLE_SIZE, replace=True, random_state=RANDOM_SEED).reset_index(drop=True)


def sample_good_rows(panel: pd.DataFrame) -> pd.DataFrame:
    ensure_exists(LABEL_CSV)
    labels = pd.read_csv(LABEL_CSV, low_memory=False)
    labels["contract_address"] = labels["contract_address"].astype(str).str.lower().str.strip()
    labels["selected_for_trust"] = pd.to_numeric(labels["selected_for_trust"], errors="coerce").fillna(0).astype(int)
    trust_addr = set(labels.loc[labels["selected_for_trust"] == 1, "contract_address"].astype(str))
    work = panel[
        (panel["collection_type"].astype(str) == "trust")
        & (pd.to_numeric(panel["low_rep_threshold_pct"], errors="coerce") == THRESHOLD_PCT)
        & (panel["contract_address"].astype(str).str.lower().isin(trust_addr))
    ].copy()
    if work.empty:
        raise ValueError("No selected_for_trust rows found for good-product evaluation")
    return work.sample(n=GOOD_SAMPLE_SIZE, replace=len(work) < GOOD_SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)


def main() -> None:
    ensure_exists(INPUT_PANEL_CSV)
    ensure_exists(LABEL_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(INPUT_PANEL_CSV, low_memory=False)
    if panel.empty:
        raise ValueError("Input panel is empty")

    malicious_rows = sample_malicious_rows(panel)
    good_rows = sample_good_rows(panel)

    agent_id, endpoint = resolve_agent_endpoint(SERVER_BASE_URL, timeout_sec=REQUEST_TIMEOUT_SECONDS)

    records: list[dict] = []
    errors: list[dict] = []
    for decision_mode in DECISION_MODES:
        for sample_group, sample_df in [("malicious_pre_report", malicious_rows), ("good_trust", good_rows)]:
            for idx, row in sample_df.iterrows():
                payload = build_payload(row, decision_mode=decision_mode)
                try:
                    response = post_json(endpoint, payload, timeout_sec=REQUEST_TIMEOUT_SECONDS)
                    result = response.get("data", response)
                    records.append(
                        {
                            "decision_mode": decision_mode,
                            "model_name": str(result.get("model_name", "")),
                            "sample_group": sample_group,
                            "sample_idx": int(idx),
                            "contract_id": payload["contract_id"],
                            "collection_type": payload["collection_type"],
                            "snapshot_idx": payload["window_idx"],
                            "lead_windows_to_detect": payload["lead_windows_to_detect"],
                            "risk_label": str(result.get("risk_label", "")),
                            "market_view": str(result.get("market_view", "")),
                            "reputation_view": str(result.get("reputation_view", "")),
                            "decision": str(result.get("decision", "")),
                            "confidence": pd.to_numeric(result.get("confidence"), errors="coerce"),
                            "reason": str(result.get("reason", "")),
                            "prompt_tokens": int(pd.to_numeric(result.get("prompt_tokens"), errors="coerce") or 0),
                            "completion_tokens": int(pd.to_numeric(result.get("completion_tokens"), errors="coerce") or 0),
                            "total_tokens": int(pd.to_numeric(result.get("total_tokens"), errors="coerce") or 0),
                            "latency_ms": int(pd.to_numeric(result.get("latency_ms"), errors="coerce") or 0),
                            "parse_success": int(pd.to_numeric(result.get("parse_success"), errors="coerce") or 0),
                        }
                    )
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                    errors.append(
                        {
                            "decision_mode": decision_mode,
                            "sample_group": sample_group,
                            "sample_idx": int(idx),
                            "contract_id": payload["contract_id"],
                            "snapshot_idx": payload["window_idx"],
                            "error": str(exc),
                        }
                    )

    decisions = pd.DataFrame(records)
    decision_errors = pd.DataFrame(errors)
    if decisions.empty:
        raise ValueError("No decision results were collected")

    summary_rows = []
    for decision_mode, sub in decisions.groupby("decision_mode", sort=False):
        mal = sub[sub["sample_group"] == "malicious_pre_report"].copy()
        good = sub[sub["sample_group"] == "good_trust"].copy()
        summary_rows.append(
            {
                "decision_mode": decision_mode,
                "model_name": "|".join(sorted(set(sub["model_name"].dropna().astype(str)))) if "model_name" in sub.columns else "",
                "malicious_n": int(len(mal)),
                "good_n": int(len(good)),
                "malicious_prevent_buy_rate": float((mal["decision"] == "AVOID_BUY").mean()) if not mal.empty else float("nan"),
                "good_false_reject_rate": float((good["decision"] == "AVOID_BUY").mean()) if not good.empty else float("nan"),
                "mean_total_tokens": float(sub["total_tokens"].mean()) if not sub.empty else float("nan"),
                "mean_prompt_tokens": float(sub["prompt_tokens"].mean()) if not sub.empty else float("nan"),
                "mean_completion_tokens": float(sub["completion_tokens"].mean()) if not sub.empty else float("nan"),
                "mean_latency_ms": float(sub["latency_ms"].mean()) if not sub.empty else float("nan"),
                "parse_success_rate": float(sub["parse_success"].mean()) if not sub.empty else float("nan"),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("decision_mode", kind="mergesort").reset_index(drop=True)

    decisions.to_csv(OUTPUT_DIR / "buy_screening_decisions.csv", index=False, encoding="utf-8-sig")
    decision_errors.to_csv(OUTPUT_DIR / "buy_screening_errors.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "buy_screening_summary.csv", index=False, encoding="utf-8-sig")

    meta = pd.DataFrame(
        [
            {"key": "decision_modes", "value": "|".join(DECISION_MODES)},
            {"key": "malicious_sample_size", "value": MALICIOUS_SAMPLE_SIZE},
            {"key": "good_sample_size", "value": GOOD_SAMPLE_SIZE},
            {"key": "threshold_pct", "value": THRESHOLD_PCT},
            {"key": "trigger_k", "value": TRIGGER_K},
            {"key": "agent_id", "value": agent_id},
            {"key": "agent_endpoint", "value": endpoint},
            {"key": "records_rows", "value": int(len(decisions))},
            {"key": "errors_rows", "value": int(len(decision_errors))},
        ]
    )
    meta.to_csv(OUTPUT_DIR / "buy_screening_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'buy_screening_decisions.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'buy_screening_errors.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'buy_screening_summary.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'buy_screening_meta.csv').resolve()}")


if __name__ == "__main__":
    main()
