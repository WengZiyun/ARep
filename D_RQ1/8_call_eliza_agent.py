#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import os
import urllib.request
import urllib.error
import urllib.parse

import pandas as pd


# ============================================================
# Config
# ============================================================
INPUT_PANEL_CSV = Path("output/D/7_agent_panel/contract_window_panel.csv")
OUTPUT_DIR = Path("output/D/8_agent_decisions")
SERVER_BASE_URL = "http://127.0.0.1:3000"
PLUGIN_NAME = "nft-risk-plugin"
TRIGGER_K_LIST = [int(x) for x in os.getenv("ELIZA_TRIGGER_K_LIST", "1,2,3,4").split(",") if x.strip()]
DECISION_MODES = [x.strip() for x in os.getenv("ELIZA_DECISION_MODES", "rule_agent,llm_base,llm_base_rank,llm_rank").split(",") if x.strip()]
HOLDING_STATES = [x.strip() for x in os.getenv("ELIZA_HOLDING_STATES", "flat,holding").split(",") if x.strip()]
REQUEST_TIMEOUT_SECONDS = int(os.getenv("ELIZA_AGENT_REQUEST_TIMEOUT", "90"))

# Limit LLM rows by default to avoid accidental high token cost.
# Set to 0 or a negative number to disable the cap.
MAX_ROWS_BY_MODE = {
    "rule_agent": int(os.getenv("RULE_AGENT_MAX_ROWS", "0")),
    "llm_base": int(os.getenv("LLM_BASE_MAX_ROWS", "200")),
    "llm_base_rank": int(os.getenv("LLM_BASE_RANK_MAX_ROWS", os.getenv("LLM_RANK_MAX_ROWS", "200"))),
    "llm_base_rank_compact": int(os.getenv("LLM_BASE_RANK_COMPACT_MAX_ROWS", "200")),
    "llm_base_rank_mini": int(os.getenv("LLM_BASE_RANK_MINI_MAX_ROWS", "200")),
    "llm_rank": int(os.getenv("LLM_RANK_ONLY_MAX_ROWS", "200")),
    "llm_rank_compact": int(os.getenv("LLM_RANK_COMPACT_MAX_ROWS", "200")),
    "llm_rank_mini": int(os.getenv("LLM_RANK_MINI_MAX_ROWS", "200")),
}
GLOBAL_MAX_ROWS = int(os.getenv("ELIZA_GLOBAL_MAX_ROWS", "0"))
COLLECTION_TYPES_FILTER = [x.strip() for x in os.getenv("ELIZA_COLLECTION_TYPES", "").split(",") if x.strip()]
MAX_CONTRACTS = int(os.getenv("ELIZA_MAX_CONTRACTS", "0"))
MAX_WINDOWS_PER_CONTRACT = int(os.getenv("ELIZA_MAX_WINDOWS_PER_CONTRACT", "0"))


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


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


def get_json(url: str, timeout_sec: int) -> dict | list:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


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


def build_payload(row: pd.Series, trigger_k: int, decision_mode: str, holding_state: str) -> dict:
    payload = {
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
        "holding_state": str(holding_state),
        "can_buy_baseline": int(row["can_buy_baseline"]),
        "window_last_trade_price_usd": None
        if pd.isna(row["window_last_trade_price_usd"])
        else float(row["window_last_trade_price_usd"]),
        "next_window_first_trade_price_usd": None
        if pd.isna(row["next_window_first_trade_price_usd"])
        else float(row["next_window_first_trade_price_usd"]),
        "next_trade_exists": int(row["next_trade_exists"]),
        "trigger_k": int(trigger_k),
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
    return payload


def iter_work_rows(panel: pd.DataFrame, decision_mode: str) -> pd.DataFrame:
    work = panel.copy()
    if COLLECTION_TYPES_FILTER:
        work = work[work["collection_type"].astype(str).isin(COLLECTION_TYPES_FILTER)].copy()
    work = work.sort_values(["contract_id", "low_rep_threshold_pct", "snapshot_idx"], kind="mergesort").reset_index(drop=True)
    if MAX_CONTRACTS > 0:
        keep_contracts = work["contract_id"].drop_duplicates().head(MAX_CONTRACTS).tolist()
        work = work[work["contract_id"].isin(keep_contracts)].copy()
    if MAX_WINDOWS_PER_CONTRACT > 0:
        work = (
            work.groupby(["contract_id", "low_rep_threshold_pct"], group_keys=False, sort=False)
            .head(MAX_WINDOWS_PER_CONTRACT)
            .reset_index(drop=True)
        )
    if GLOBAL_MAX_ROWS > 0:
        work = work.head(GLOBAL_MAX_ROWS).copy()
    max_rows = int(MAX_ROWS_BY_MODE.get(decision_mode, 0) or 0)
    if max_rows > 0:
        work = work.head(max_rows).copy()
    return work


def main() -> None:
    ensure_exists(INPUT_PANEL_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(INPUT_PANEL_CSV, low_memory=False)
    if panel.empty:
        raise ValueError("Input panel is empty")

    agent_id, agent_endpoint = resolve_agent_endpoint(SERVER_BASE_URL, timeout_sec=REQUEST_TIMEOUT_SECONDS)

    rows = []
    errors = []
    for decision_mode in DECISION_MODES:
        work = iter_work_rows(panel, decision_mode)
        for holding_state in HOLDING_STATES:
            for trigger_k in TRIGGER_K_LIST:
                for row in work.itertuples(index=False):
                    row_s = pd.Series(row._asdict())
                    payload = build_payload(
                        row_s,
                        trigger_k=trigger_k,
                        decision_mode=decision_mode,
                        holding_state=holding_state,
                    )
                    try:
                        response = post_json(agent_endpoint, payload, timeout_sec=REQUEST_TIMEOUT_SECONDS)
                        result = response.get("data", response)
                        rows.append(
                            {
                                "decision_mode": str(result.get("decision_mode", decision_mode)),
                                "model_name": str(result.get("model_name", "")),
                                "contract_id": payload["contract_id"],
                                "contract_address": payload["contract_address"],
                                "collection_type": payload["collection_type"],
                                "snapshot_idx": payload["window_idx"],
                                "start_block": payload["start_block"],
                                "end_block": payload["end_block"],
                                "low_rep_threshold_pct": payload["low_rep_threshold_pct"],
                                "trigger_k": int(trigger_k),
                                "holding_state": str(result.get("holding_state", holding_state)),
                                "decision": str(result.get("decision", "")),
                                "confidence": pd.to_numeric(result.get("confidence"), errors="coerce"),
                                "reason": str(result.get("reason", "")),
                                "is_low_rep": int(result.get("is_low_rep", payload["is_low_rep"])),
                                "low_rep_streak": int(result.get("low_rep_streak", payload["low_rep_streak"])),
                                "next_trade_exists": int(bool(result.get("next_trade_exists", payload["next_trade_exists"]))),
                                "prompt_tokens": int(pd.to_numeric(result.get("prompt_tokens"), errors="coerce") or 0),
                                "completion_tokens": int(pd.to_numeric(result.get("completion_tokens"), errors="coerce") or 0),
                                "total_tokens": int(pd.to_numeric(result.get("total_tokens"), errors="coerce") or 0),
                                "latency_ms": int(pd.to_numeric(result.get("latency_ms"), errors="coerce") or 0),
                                "parse_success": int(pd.to_numeric(result.get("parse_success"), errors="coerce") or 0),
                                "raw_decision_text": str(result.get("raw_decision_text", "")),
                            }
                        )
                    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                        errors.append(
                            {
                                "decision_mode": decision_mode,
                                "holding_state": holding_state,
                                "contract_id": payload["contract_id"],
                                "snapshot_idx": payload["window_idx"],
                                "low_rep_threshold_pct": payload["low_rep_threshold_pct"],
                                "trigger_k": int(trigger_k),
                                "error": str(exc),
                            }
                        )

    decisions = pd.DataFrame(rows)
    decision_errors = pd.DataFrame(errors)
    decisions.to_csv(OUTPUT_DIR / "agent_decisions.csv", index=False, encoding="utf-8-sig")

    with (OUTPUT_DIR / "agent_decisions.jsonl").open("w", encoding="utf-8") as fh:
        for item in decisions.to_dict(orient="records"):
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    decision_errors.to_csv(OUTPUT_DIR / "agent_decision_errors.csv", index=False, encoding="utf-8-sig")
    meta = pd.DataFrame(
        [
            {"key": "rows_input", "value": int(len(panel))},
            {"key": "trigger_k_list", "value": "|".join(str(x) for x in TRIGGER_K_LIST)},
            {"key": "decision_modes", "value": "|".join(DECISION_MODES)},
            {"key": "holding_states", "value": "|".join(HOLDING_STATES)},
            {"key": "collection_types_filter", "value": "|".join(COLLECTION_TYPES_FILTER)},
            {"key": "global_max_rows", "value": GLOBAL_MAX_ROWS},
            {"key": "max_contracts", "value": MAX_CONTRACTS},
            {"key": "max_windows_per_contract", "value": MAX_WINDOWS_PER_CONTRACT},
            {"key": "rule_agent_max_rows", "value": MAX_ROWS_BY_MODE["rule_agent"]},
            {"key": "llm_base_max_rows", "value": MAX_ROWS_BY_MODE["llm_base"]},
            {"key": "llm_rank_max_rows", "value": MAX_ROWS_BY_MODE["llm_rank"]},
            {"key": "decisions_rows", "value": int(len(decisions))},
            {"key": "errors_rows", "value": int(len(decision_errors))},
            {"key": "agent_id", "value": agent_id},
            {"key": "agent_endpoint", "value": agent_endpoint},
        ]
    )
    meta.to_csv(OUTPUT_DIR / "decision_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'agent_decisions.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'agent_decisions.jsonl').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'agent_decision_errors.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'decision_meta.csv').resolve()}")


if __name__ == "__main__":
    main()
