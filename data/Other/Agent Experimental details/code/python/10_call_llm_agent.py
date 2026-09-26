#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import time
import urllib.error
import urllib.request

import pandas as pd


# ============================================================
# Config
# ============================================================
INPUT_PANEL_CSV = Path("output/D/7_agent_panel/contract_window_panel.csv")
OUTPUT_DIR = Path("output/D/10_llm_agent")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.uniapi.io/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-oss-120b").strip()

LLM_METHODS = ["llm_base", "llm_rank"]
TRIGGER_K_LIST = [1, 2, 3, 4]
REQUEST_TIMEOUT_SECONDS = 90
TEMPERATURE = 0.0
MAX_COMPLETION_TOKENS = 180

# Safety guard to avoid accidental large bills.
# Set env LLM_AGENT_MAX_ROWS_OVERRIDE to override.
MAX_ROWS_PER_METHOD = int(os.getenv("LLM_AGENT_MAX_ROWS_OVERRIDE", "500"))


SYSTEM_PROMPT = """You are an NFT risk decision agent.

Your task is to evaluate one NFT contract-window record and output exactly one action:
SELL, AVOID_BUY, HOLD, or ALLOW_BUY.

Decision objective:
- reduce downside risk,
- avoid unnecessary losses,
- avoid overly aggressive reactions when evidence is weak.

Rules:
- use only the provided fields,
- do not assume hidden labels or future outcomes,
- do not invent missing values,
- if evidence is insufficient, prefer the more conservative valid action only when justified,
- return strict JSON only.

Output format:
{
  "decision": "SELL" | "AVOID_BUY" | "HOLD" | "ALLOW_BUY",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}
""".strip()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def require_api_key() -> None:
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set. Set it in the environment before running.")


def fmt_num(value) -> str:
    if pd.isna(value):
        return "null"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num) >= 1000 or num.is_integer():
        return str(int(num)) if num.is_integer() else f"{num:.2f}"
    return f"{num:.6f}".rstrip("0").rstrip(".")


def build_user_prompt(row: pd.Series, method: str, trigger_k: int) -> str:
    lines = [
        "Evaluate the following NFT contract-window record.",
        "",
        "Current position state:",
        f"- holding_state: {row['holding_state']}",
        "",
        "Current market execution context:",
        f"- current_window_last_trade_price_usd: {fmt_num(row['window_last_trade_price_usd'])}",
        f"- next_trade_exists: {int(pd.to_numeric(row['next_trade_exists'], errors='coerce') or 0)}",
        f"- liquidity_flag: {int(pd.to_numeric(row['liquidity_flag'], errors='coerce') or 0)}",
        "",
        "Current market activity:",
        f"- window_trade_count: {fmt_num(row['window_trade_count'])}",
        f"- window_trade_value_total_usd: {fmt_num(row['window_trade_value_total_usd'])}",
        f"- window_unique_buyers: {fmt_num(row['window_unique_buyers'])}",
        f"- window_unique_trade_users: {fmt_num(row['window_unique_trade_users'])}",
        f"- window_mint_count: {fmt_num(row['window_mint_cnt'])}",
        f"- window_gas_total_usd: {fmt_num(row['window_gas_total_usd'])}",
        "",
        "Recent market change:",
        f"- price_change_vs_prev_window_pct: {fmt_num(row['price_change_vs_prev_window_pct'])}",
        f"- trade_count_change_vs_prev_window_pct: {fmt_num(row['trade_count_change_vs_prev_window_pct'])}",
    ]
    if method == "llm_rank":
        lines.extend(
            [
                "",
                "Reputation signals:",
                f"- rep_score: {fmt_num(row['rep_score'])}",
                f"- rep_rank: {fmt_num(row['rep_rank'])}",
                f"- rep_rank_pct: {fmt_num(row['rep_rank_pct'])}",
                f"- is_low_rep: {fmt_num(row['is_low_rep'])}",
                f"- low_rep_streak: {fmt_num(row['low_rep_streak'])}",
            ]
        )
    lines.extend(
        [
            "",
            "Decision reminder:",
            f"- trigger_k: {trigger_k}",
            "- SELL is only valid when currently holding.",
            "- AVOID_BUY is only valid when currently flat.",
            "- HOLD means keep holding.",
            "- ALLOW_BUY means buying is allowed but not mandatory.",
            "",
            "Return strict JSON only.",
        ]
    )
    return "\n".join(lines)


def call_openai_compatible(system_prompt: str, user_prompt: str) -> dict:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def extract_json_object(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_decision(parsed: dict | None) -> tuple[str, float | None, str, int]:
    if not parsed:
        return "PARSE_FAILED", None, "", 0
    decision = str(parsed.get("decision", "")).strip().upper()
    if decision not in {"SELL", "AVOID_BUY", "HOLD", "ALLOW_BUY"}:
        return "PARSE_FAILED", None, str(parsed.get("reason", "")), 0
    confidence = pd.to_numeric(parsed.get("confidence"), errors="coerce")
    reason = str(parsed.get("reason", "")).strip()
    return decision, (None if pd.isna(confidence) else float(confidence)), reason, 1


def iter_panel_rows(panel: pd.DataFrame):
    work = panel.copy()
    work = work.sort_values(["contract_id", "low_rep_threshold_pct", "snapshot_idx"], kind="mergesort").reset_index(drop=True)
    if MAX_ROWS_PER_METHOD > 0:
        work = work.head(MAX_ROWS_PER_METHOD).copy()
    return work


def main() -> None:
    ensure_exists(INPUT_PANEL_CSV)
    require_api_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(INPUT_PANEL_CSV, low_memory=False)
    if panel.empty:
        raise ValueError("Input panel is empty")

    rows = []
    errors = []
    started = time.time()

    for method in LLM_METHODS:
        for trigger_k in TRIGGER_K_LIST:
            for row in iter_panel_rows(panel).itertuples(index=False):
                row_s = pd.Series(row._asdict())
                user_prompt = build_user_prompt(row_s, method=method, trigger_k=trigger_k)
                t0 = time.time()
                try:
                    response = call_openai_compatible(SYSTEM_PROMPT, user_prompt)
                    latency_ms = int(round((time.time() - t0) * 1000.0))
                    choice = ((response.get("choices") or [{}])[0] or {})
                    message = choice.get("message") or {}
                    raw_text = str(message.get("content") or "")
                    parsed = extract_json_object(raw_text)
                    decision, confidence, reason, parse_success = normalize_decision(parsed)
                    usage = response.get("usage") or {}
                    rows.append(
                        {
                            "method": method,
                            "model_name": OPENAI_MODEL,
                            "contract_id": str(row_s["contract_id"]),
                            "contract_address": str(row_s["contract_address"]),
                            "collection_type": str(row_s["collection_type"]),
                            "snapshot_idx": int(row_s["snapshot_idx"]),
                            "start_block": int(row_s["start_block"]),
                            "end_block": int(row_s["end_block"]),
                            "low_rep_threshold_pct": float(row_s["low_rep_threshold_pct"]),
                            "trigger_k": int(trigger_k),
                            "decision": decision,
                            "confidence": confidence,
                            "reason": reason,
                            "parse_success": int(parse_success),
                            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                            "total_tokens": int(usage.get("total_tokens", 0) or 0),
                            "latency_ms": latency_ms,
                            "raw_decision_text": raw_text,
                        }
                    )
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                    errors.append(
                        {
                            "method": method,
                            "contract_id": str(row_s["contract_id"]),
                            "snapshot_idx": int(row_s["snapshot_idx"]),
                            "low_rep_threshold_pct": float(row_s["low_rep_threshold_pct"]),
                            "trigger_k": int(trigger_k),
                            "error": str(exc),
                        }
                    )

    decisions = pd.DataFrame(rows)
    error_df = pd.DataFrame(errors)
    decisions.to_csv(OUTPUT_DIR / "llm_agent_decisions.csv", index=False, encoding="utf-8-sig")
    error_df.to_csv(OUTPUT_DIR / "llm_agent_errors.csv", index=False, encoding="utf-8-sig")

    meta = pd.DataFrame(
        [
            {"key": "model_name", "value": OPENAI_MODEL},
            {"key": "base_url", "value": OPENAI_BASE_URL},
            {"key": "methods", "value": "|".join(LLM_METHODS)},
            {"key": "trigger_k_list", "value": "|".join(str(x) for x in TRIGGER_K_LIST)},
            {"key": "max_rows_per_method", "value": MAX_ROWS_PER_METHOD},
            {"key": "decision_rows", "value": int(len(decisions))},
            {"key": "error_rows", "value": int(len(error_df))},
            {"key": "elapsed_seconds", "value": round(time.time() - started, 3)},
        ]
    )
    meta.to_csv(OUTPUT_DIR / "llm_agent_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'llm_agent_decisions.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'llm_agent_errors.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'llm_agent_meta.csv').resolve()}")


if __name__ == "__main__":
    main()
