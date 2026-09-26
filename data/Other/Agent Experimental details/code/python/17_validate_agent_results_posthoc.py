from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path("output/D/15_buy_screening")
PANEL_CSV = Path("output/D/7_agent_panel/contract_window_panel.csv")
ARCHIVED_SUMMARY_CSV = INPUT_DIR / "all_models_summary.csv"
OUTPUT_DIR = Path("output/D/17_agent_posthoc_validation")

BOOTSTRAP_ITERATIONS = 10_000
RANDOM_SEED = 42
MANUAL_AUDIT_ROWS_PER_STRATUM = 5
PAPER_TABLE_MODELS = {
    "gpt-5.4-nano",
    "grok-4-1-fast",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash-nothinking",
}
PAPER_TABLE_MODES = {"llm_base", "llm_base_rank", "llm_rank"}

VALID_DECISIONS = {"AVOID_BUY", "ALLOW_BUY"}
VALID_RISK_LABELS = {"RISK", "NO_RISK"}
VALID_VIEWS = {"positive", "neutral", "negative"}


@dataclass(frozen=True)
class SourceSpec:
    filename: str
    model_name: str
    include_mode: str | None = None
    mode_map: tuple[tuple[str, str], ...] = ()
    evidence_scope: str = "decision_reason_confidence"
    note: str = ""


SOURCES = (
    SourceSpec(
        "toptrust_100x2_decisions_end.csv",
        "gpt-5.4-nano",
        evidence_scope="decision_reason_risk_views",
        note=(
            "Complete later 100x2 row-level run. It is not the unidentified GPT run "
            "reported as 'recorded run' in all_models_summary.csv."
        ),
    ),
    SourceSpec("claude_rule_llm_base_decisions.csv", "claude-sonnet-4-6", "llm_base"),
    SourceSpec(
        "claude_llm_rank_decisions.csv",
        "claude-sonnet-4-6",
        mode_map=(("llm_rank", "llm_base_rank"),),
    ),
    SourceSpec(
        "claude_llm_rank_compact_decisions.csv",
        "claude-sonnet-4-6",
        mode_map=(("llm_rank_compact", "llm_base_rank_compact"),),
    ),
    SourceSpec(
        "claude_llm_rank_mini_decisions.csv",
        "claude-sonnet-4-6",
        mode_map=(("llm_rank_mini", "llm_base_rank_mini"),),
    ),
    SourceSpec("deepseek_rule_llm_base_decisions.csv", "deepseek-v3.2", "llm_base"),
    SourceSpec(
        "deepseek_llm_rank_decisions.csv",
        "deepseek-v3.2",
        mode_map=(("llm_rank", "llm_base_rank"),),
    ),
    SourceSpec(
        "deepseek_llm_rank_compact_decisions.csv",
        "deepseek-v3.2",
        mode_map=(("llm_rank_compact", "llm_base_rank_compact"),),
    ),
    SourceSpec(
        "deepseek_llm_rank_mini_decisions.csv",
        "deepseek-v3.2",
        mode_map=(("llm_rank_mini", "llm_base_rank_mini"),),
    ),
    SourceSpec("gemini_rule_llm_base_decisions.csv", "gemini-3-flash-preview", "llm_base"),
    SourceSpec(
        "gemini_llm_rank_decisions.csv",
        "gemini-3-flash-preview",
        mode_map=(("llm_rank", "llm_base_rank"),),
    ),
    SourceSpec(
        "gemini_llm_rank_compact_decisions.csv",
        "gemini-3-flash-preview",
        mode_map=(("llm_rank_compact", "llm_base_rank_compact"),),
    ),
    SourceSpec(
        "gemini_llm_rank_mini_decisions.csv",
        "gemini-3-flash-preview",
        mode_map=(("llm_rank_mini", "llm_base_rank_mini"),),
    ),
    SourceSpec(
        "toptrust_100x2_grok41fast_llm_base_decisions.csv",
        "grok-4-1-fast",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run used by the paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_grok41fast_llm_base_rank_decisions.csv",
        "grok-4-1-fast",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run used by the paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_grok41fast_llm_base_rank_mini_decisions.csv",
        "grok-4-1-fast",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run; not displayed in the three-mode paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_grok41fast_llm_rank_decisions.csv",
        "grok-4-1-fast",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run used by the paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_claudehaiku_llm_base_decisions.csv",
        "claude-haiku-4-5-20251001",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run used by the paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_claudehaiku_llm_base_rank_decisions.csv",
        "claude-haiku-4-5-20251001",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run used by the paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_claudehaiku_llm_base_rank_mini_decisions.csv",
        "claude-haiku-4-5-20251001",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run; not displayed in the three-mode paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_claudehaiku_llm_rank_decisions.csv",
        "claude-haiku-4-5-20251001",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run used by the paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_gemini25flashnothinking_llm_base_decisions.csv",
        "gemini-2.5-flash-nothinking",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run used by the paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_gemini25flashnothinking_llm_base_rank_decisions.csv",
        "gemini-2.5-flash-nothinking",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run used by the paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_gemini25flashnothinking_llm_base_rank_mini_decisions.csv",
        "gemini-2.5-flash-nothinking",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run; not displayed in the three-mode paper table.",
    ),
    SourceSpec(
        "toptrust_100x2_gemini25flashnothinking_llm_rank_decisions.csv",
        "gemini-2.5-flash-nothinking",
        evidence_scope="decision_reason_risk_views",
        note="Complete archived top-trust row-level run used by the paper table.",
    ),
)

RULE_SOURCE = SourceSpec(
    "claude_rule_llm_base_decisions.csv",
    "rule_agent",
    include_mode="rule_agent",
    evidence_scope="deterministic_rule_output",
    note="One archived copy of the deterministic rule baseline; duplicate copies are omitted.",
)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") + RANDOM_SEED) % (2**32)


def canonical_index(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_group(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"top_trust", "good_trust", "trust"}:
        return "good_trust"
    if text in {"malicious_pre_report", "risk", "malicious"}:
        return "malicious_pre_report"
    return text


def load_source(spec: SourceSpec) -> pd.DataFrame:
    path = INPUT_DIR / spec.filename
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    frame["source_row"] = np.arange(len(frame), dtype=int) + 2
    if spec.include_mode is not None:
        frame = frame.loc[frame["decision_mode"].astype(str) == spec.include_mode].copy()
    else:
        frame = frame.loc[frame["decision_mode"].astype(str) != "rule_agent"].copy()

    mapping = dict(spec.mode_map)
    frame["decision_mode_original"] = frame["decision_mode"].astype(str)
    frame["decision_mode"] = frame["decision_mode_original"].replace(mapping)
    frame["model_name"] = spec.model_name
    frame["source_file"] = spec.filename
    frame["evidence_scope"] = spec.evidence_scope
    frame["source_note"] = spec.note
    frame["sample_group"] = frame["sample_group"].map(normalize_group)
    return frame


def prepare_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_CSV)
    panel["contract_id"] = panel["contract_id"].astype(str)
    panel["snapshot_key"] = panel["snapshot_idx"].map(canonical_index)
    panel = panel.drop_duplicates(["contract_id", "snapshot_key"], keep="last")
    keep = [
        "contract_id",
        "snapshot_key",
        "collection_type",
        "window_last_trade_price_usd",
        "window_trade_count",
        "window_trade_value_total_usd",
        "window_unique_buyers",
        "window_unique_trade_users",
        "window_mint_cnt",
        "current_mint_supply_count",
        "window_gas_total_usd",
        "price_change_vs_prev_window_pct",
        "trade_count_change_vs_prev_window_pct",
        "rep_score",
        "rep_rank",
        "rep_rank_pct",
        "is_low_rep",
        "low_rep_streak",
        "lead_windows_to_detect",
    ]
    panel = panel[keep].copy()
    return panel.rename(
        columns={name: f"current_panel_{name}" for name in keep if name not in {"contract_id", "snapshot_key"}}
    )


def present_and_valid(series: pd.Series, valid_values: set[str]) -> pd.Series:
    return series.fillna("").astype(str).str.strip().isin(valid_values)


def add_validation_fields(frame: pd.DataFrame, rule_reasons: set[str]) -> pd.DataFrame:
    result = frame.copy()
    result["contract_id"] = result["contract_id"].astype(str)
    result["snapshot_key"] = result["snapshot_idx"].map(canonical_index)
    result["parse_valid"] = pd.to_numeric(result.get("parse_success"), errors="coerce").fillna(0).eq(1)
    result["decision_valid"] = present_and_valid(result["decision"], VALID_DECISIONS)
    result["reason_nonempty"] = result["reason"].fillna("").astype(str).str.strip().ne("")

    if "confidence" in result.columns:
        confidence_available = result["confidence"].notna() & result["confidence"].astype(str).str.strip().ne("")
        confidence = pd.to_numeric(result["confidence"], errors="coerce")
        result["confidence_valid"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        result.loc[confidence_available, "confidence_valid"] = confidence.loc[confidence_available].between(
            0, 1, inclusive="both"
        )
    else:
        result["confidence_valid"] = pd.Series(pd.NA, index=result.index, dtype="boolean")

    if "risk_label" in result.columns:
        risk_available = result["risk_label"].notna() & result["risk_label"].astype(str).str.strip().ne("")
        result["risk_label_valid"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        result.loc[risk_available, "risk_label_valid"] = present_and_valid(
            result.loc[risk_available, "risk_label"], VALID_RISK_LABELS
        )
        expected = result["risk_label"].astype(str).map({"RISK": "AVOID_BUY", "NO_RISK": "ALLOW_BUY"})
        result["decision_risk_consistent"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        result.loc[risk_available, "decision_risk_consistent"] = expected.loc[risk_available].eq(
            result.loc[risk_available, "decision"].astype(str)
        )
    else:
        result["risk_label_valid"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        result["decision_risk_consistent"] = pd.Series(pd.NA, index=result.index, dtype="boolean")

    if {"market_view", "reputation_view"}.issubset(result.columns):
        views_available = (
            result["market_view"].notna()
            & result["market_view"].astype(str).str.strip().ne("")
            & result["reputation_view"].notna()
            & result["reputation_view"].astype(str).str.strip().ne("")
        )
        result["evidence_views_valid"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        result.loc[views_available, "evidence_views_valid"] = (
            present_and_valid(result.loc[views_available, "market_view"], VALID_VIEWS)
            & present_and_valid(result.loc[views_available, "reputation_view"], VALID_VIEWS)
        )
    else:
        result["evidence_views_valid"] = pd.Series(pd.NA, index=result.index, dtype="boolean")

    result["prompt_rule_consistent"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    logic_available = result["risk_label_valid"].eq(True) & result["evidence_views_valid"].eq(True)
    has_negative = result["market_view"].eq("negative") | result["reputation_view"].eq("negative")
    has_positive = result["market_view"].eq("positive") | result["reputation_view"].eq("positive")
    expected_risk_label = pd.Series("NO_RISK", index=result.index)
    expected_risk_label.loc[has_negative & ~has_positive] = "RISK"
    result["prompt_rule_expected_risk_label"] = ""
    result.loc[logic_available, "prompt_rule_expected_risk_label"] = expected_risk_label.loc[logic_available]
    result.loc[logic_available, "prompt_rule_consistent"] = result.loc[logic_available, "risk_label"].eq(
        expected_risk_label.loc[logic_available]
    )

    checks = result["parse_valid"] & result["decision_valid"] & result["reason_nonempty"]
    for column in ("confidence_valid", "risk_label_valid", "decision_risk_consistent", "evidence_views_valid"):
        available = result[column].notna()
        checks &= ~available | result[column].fillna(False).astype(bool)
    result["saved_fields_valid"] = checks

    normalized_reason = result["reason"].fillna("").astype(str).str.strip()
    is_llm = result["model_name"].ne("rule_agent")
    result["fallback_suspected"] = (~result["parse_valid"]) | (is_llm & normalized_reason.isin(rule_reasons))
    result["included_in_valid_metrics"] = result["saved_fields_valid"] & ~result["fallback_suspected"]
    result["exclusion_reason"] = ""
    result.loc[~result["parse_valid"], "exclusion_reason"] = "parse_failed_rule_fallback_likely"
    result.loc[
        result["parse_valid"] & ~result["saved_fields_valid"], "exclusion_reason"
    ] = "saved_output_fields_invalid"
    result.loc[
        result["parse_valid"] & result["saved_fields_valid"] & result["fallback_suspected"],
        "exclusion_reason",
    ] = "rule_fallback_signature"
    result["raw_response_available"] = (
        result["raw_decision_text"].fillna("").astype(str).str.strip().ne("")
        if "raw_decision_text" in result.columns
        else False
    )
    return result


def numeric_value(row: pd.Series, name: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def automated_claim_check(row: pd.Series) -> tuple[str, str, str]:
    if not bool(row.get("included_in_valid_metrics", False)):
        return "not_checked", "", "output_excluded_before_claim_check"

    reason = str(row.get("reason", ""))
    lower = reason.lower()
    checked: list[str] = []
    flags: list[str] = []

    # Only fields stored with the archived decision are treated as exact historical input.
    # The current panel is used for record location only because it has changed since these runs.
    rank_pct = numeric_value(row, "archived_rep_rank_pct")
    if rank_pct is not None:
        expected = int(round((1 - rank_pct) * 100))
        claims = [int(value) for value in re.findall(r"(?<![\d.])(\d{1,3})\s*/\s*100", lower)]
        claims += [
            int(value)
            for value in re.findall(r"(?<![\d.])(\d{1,3})(?:st|nd|rd|th)?\s+percentile", lower)
        ]
        if claims:
            checked.append("credibility_percentile")
            if any(abs(value - expected) > 1 for value in claims):
                flags.append(f"credibility_claim_mismatch(expected={expected},claimed={claims})")

    streak = numeric_value(row, "archived_low_rep_streak")
    if streak is not None:
        streak_claims = [
            int(value)
            for value in re.findall(
                r"(?<![\d.])(\d+)\s*(?:-|\s)(?:consecutive\s+)?windows?(?:\s+low[- ]reputation)?\s+streak",
                lower,
            )
        ]
        streak_claims += [
            int(value)
            for value in re.findall(r"(?<![\d.])(\d+)\s+consecutive\s+windows", lower)
        ]
        if streak_claims:
            checked.append("low_rep_streak")
            if any(value != int(round(streak)) for value in streak_claims):
                flags.append(f"streak_claim_mismatch(expected={int(round(streak))},claimed={streak_claims})")

    if flags:
        return "flagged_for_human_review", "|".join(sorted(set(checked))), "|".join(flags)
    if checked:
        return "no_mismatch_detected", "|".join(sorted(set(checked))), ""
    if rank_pct is not None or streak is not None:
        return "no_checkable_numeric_claim", "", ""
    return "no_exact_archived_input_fields_to_check", "", ""


def build_group_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["model_name", "decision_mode", "sample_group"]
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        valid = group.loc[group["included_in_valid_metrics"]].copy()
        rows.append(
            {
                "model_name": key[0],
                "decision_mode": key[1],
                "sample_group": key[2],
                "attempted_rows": len(group),
                "valid_rows": len(valid),
                "valid_output_coverage": len(valid) / len(group) if len(group) else np.nan,
                "unique_contracts_valid": valid["contract_id"].nunique(),
                "unique_contract_snapshots_valid": valid[["contract_id", "snapshot_key"]].drop_duplicates().shape[0],
                "avoid_buy_rate_valid_only": (valid["decision"] == "AVOID_BUY").mean() if len(valid) else np.nan,
                "mean_total_tokens_valid_only": pd.to_numeric(valid.get("total_tokens"), errors="coerce").mean(),
                "current_panel_identifier_join_rate": group["current_panel_row_found"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_model_summary(group_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, mode), group in group_metrics.groupby(["model_name", "decision_mode"], sort=False):
        row: dict[str, object] = {"model_name": model, "decision_mode": mode}
        for sample_group, prefix in (
            ("malicious_pre_report", "malicious"),
            ("good_trust", "good"),
        ):
            match = group.loc[group["sample_group"] == sample_group]
            if match.empty:
                continue
            item = match.iloc[0]
            row[f"{prefix}_attempted_rows"] = item["attempted_rows"]
            row[f"{prefix}_valid_rows"] = item["valid_rows"]
            row[f"{prefix}_valid_output_coverage"] = item["valid_output_coverage"]
            row[f"{prefix}_unique_contracts"] = item["unique_contracts_valid"]
            row[f"{prefix}_unique_contract_snapshots"] = item["unique_contract_snapshots_valid"]
            metric = "malicious_prevent_buy_rate_valid_only" if prefix == "malicious" else "good_false_reject_rate_valid_only"
            row[metric] = item["avoid_buy_rate_valid_only"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_contract_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame.loc[frame["included_in_valid_metrics"]].copy()
    valid["avoid_buy"] = valid["decision"].eq("AVOID_BUY").astype(float)

    # Repeated sampling of the same contract-window is averaged before contract aggregation.
    window_level = (
        valid.groupby(
            ["model_name", "decision_mode", "sample_group", "contract_id", "snapshot_key"],
            as_index=False,
            sort=False,
        )["avoid_buy"]
        .mean()
    )
    contract_level = (
        window_level.groupby(
            ["model_name", "decision_mode", "sample_group", "contract_id"],
            as_index=False,
            sort=False,
        )
        .agg(unique_snapshots=("snapshot_key", "nunique"), avoid_buy_rate=("avoid_buy", "mean"))
    )

    summary_rows: list[dict[str, object]] = []
    for key, group in contract_level.groupby(["model_name", "decision_mode", "sample_group"], sort=False):
        values = group["avoid_buy_rate"].to_numpy(dtype=float)
        rng = np.random.default_rng(stable_seed(*key))
        if len(values):
            draws = rng.choice(values, size=(BOOTSTRAP_ITERATIONS, len(values)), replace=True).mean(axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975])
            estimate = values.mean()
        else:
            estimate = lower = upper = np.nan
        summary_rows.append(
            {
                "model_name": key[0],
                "decision_mode": key[1],
                "sample_group": key[2],
                "unique_contracts": len(values),
                "contract_macro_avoid_buy_rate": estimate,
                "cluster_bootstrap_ci_lower": lower,
                "cluster_bootstrap_ci_upper": upper,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "small_cluster_warning": int(len(values) < 10),
            }
        )
    return contract_level, pd.DataFrame(summary_rows)


def build_manual_audit_sample(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[
        frame["included_in_valid_metrics"] & frame["model_name"].ne("rule_agent")
    ].copy()
    valid = valid.drop_duplicates(
        ["model_name", "decision_mode", "sample_group", "contract_id", "snapshot_key", "reason"]
    )
    parts: list[pd.DataFrame] = [
        valid.loc[valid["automated_claim_check_status"] == "flagged_for_human_review"]
    ]
    for key, group in valid.groupby(["model_name", "decision_mode", "sample_group"], sort=False):
        n = min(MANUAL_AUDIT_ROWS_PER_STRATUM, len(group))
        parts.append(group.sample(n=n, random_state=stable_seed(*key)))
    sample = pd.concat(parts, ignore_index=True) if parts else valid.head(0)
    sample = sample.drop_duplicates(
        ["model_name", "decision_mode", "sample_group", "contract_id", "snapshot_key", "reason"]
    )
    sample["human_evidence_supported"] = ""
    sample["human_no_unsupported_claims"] = ""
    sample["human_decision_reason_consistent"] = ""
    sample["human_auditor"] = ""
    sample["human_audit_notes"] = ""
    preferred = [
        "model_name",
        "decision_mode",
        "sample_group",
        "contract_id",
        "snapshot_idx",
        "decision",
        "risk_label",
        "market_view",
        "reputation_view",
        "confidence",
        "reason",
        "automated_claim_check_status",
        "automated_claim_flags",
        "prompt_rule_expected_risk_label",
        "prompt_rule_consistent",
        "archived_rep_rank_pct",
        "archived_low_rep_streak",
        "historical_input_scope",
        "current_panel_alignment",
        "current_panel_window_last_trade_price_usd",
        "current_panel_window_trade_count",
        "current_panel_window_trade_value_total_usd",
        "current_panel_window_unique_buyers",
        "current_panel_window_unique_trade_users",
        "current_panel_window_mint_cnt",
        "current_panel_current_mint_supply_count",
        "current_panel_window_gas_total_usd",
        "current_panel_price_change_vs_prev_window_pct",
        "current_panel_trade_count_change_vs_prev_window_pct",
        "current_panel_rep_score",
        "current_panel_rep_rank",
        "current_panel_rep_rank_pct",
        "current_panel_low_rep_streak",
        "source_file",
        "source_row",
        "human_evidence_supported",
        "human_no_unsupported_claims",
        "human_decision_reason_consistent",
        "human_auditor",
        "human_audit_notes",
    ]
    return sample[[name for name in preferred if name in sample.columns]]


def build_source_provenance() -> pd.DataFrame:
    rows = [
        {
            "artifact": "all_models_summary.csv: GPT-5.4-nano recorded-run rows",
            "status": "summary_only_not_row_level_validatable",
            "use": "Retain as historical record only; do not present as post-hoc validated.",
            "limitation": "No matching 100x2 row-level file with the same reported metrics was identified.",
        },
        {
            "artifact": "toptrust_100x2_decisions_end.csv",
            "status": "row_level_available_alternate_gpt_run",
            "use": "Use as the auditable GPT run, reported separately from the historical recorded-run rows.",
            "limitation": "Raw provider response and exact rendered prompts were not saved.",
        },
        {
            "artifact": "Claude decision CSV files",
            "status": "row_level_available_partial_output_fields",
            "use": "Use after parse and saved-field validation.",
            "limitation": "Risk/evidence-view fields and raw provider responses were not retained.",
        },
        {
            "artifact": "DeepSeek decision CSV files",
            "status": "row_level_available_partial_output_fields",
            "use": "Use after parse and saved-field validation; report missing attempts.",
            "limitation": "Risk/evidence-view fields and raw provider responses were not retained.",
        },
        {
            "artifact": "Gemini decision CSV files",
            "status": "row_level_available_heavy_parse_failure",
            "use": "Use only parse-valid non-fallback rows as conditional descriptive evidence.",
            "limitation": "Most rows used rule fallback; valid-only rows may be selection-biased.",
        },
        {
            "artifact": "8_agent_decisions/agent_decisions.csv",
            "status": "raw_decision_text_available_separate_experiment",
            "use": "Suitable for a separate full raw-text audit of the earlier experiment.",
            "limitation": "Not the buy-screening experiment summarized here.",
        },
    ]
    return pd.DataFrame(rows)


def build_archived_summary_audit() -> pd.DataFrame:
    archived = pd.read_csv(ARCHIVED_SUMMARY_CSV)
    archived["posthoc_status"] = "row_level_source_available"
    archived["recommended_use"] = "recompute_from_valid_row_level_outputs"
    gpt = archived["model_name"].eq("gpt-5.4-nano") & archived["decision_mode"].ne("rule_agent")
    archived.loc[gpt, "posthoc_status"] = "summary_only_no_matching_row_level_source"
    archived.loc[gpt, "recommended_use"] = "historical_record_only"
    gemini = archived["model_name"].eq("gemini-3-flash-preview") & archived["decision_mode"].ne("rule_agent")
    archived.loc[gemini, "posthoc_status"] = "fallback_contaminated"
    archived.loc[gemini, "recommended_use"] = "do_not_use_original_performance_metric"
    return archived


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame[columns].copy()
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.3f}")
    header = "| " + " | ".join(data.columns) + " |"
    separator = "|" + "|".join(["---"] * len(data.columns)) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in data.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_report(
    validation_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    claim_summary: pd.DataFrame,
    logic_summary: pd.DataFrame,
    contract_summary: pd.DataFrame,
    panel_alignment_summary: pd.DataFrame,
) -> None:
    model_counts = (
        validation_summary.groupby("model_name", as_index=False)
        .agg(attempted=("rows", "sum"), parsed=("parse_valid_rows", "sum"), eligible=("eligible_rows", "sum"))
    )
    model_counts["eligible_rate"] = model_counts["eligible"] / model_counts["attempted"]

    report = f"""# Post-hoc Validation of Archived LLM-Agent Results

Generated: {datetime.now(timezone.utc).isoformat()}

## Purpose

This audit salvages the archived buy-screening results without calling the discontinued model endpoints. Original files are not modified. The audit separates parse-valid LLM outputs from likely rule fallbacks, validates all retained output fields, checks exact historical fields that were saved beside decisions, and reports contract-level uncertainty.

## Recoverable Output Coverage

{markdown_table(model_counts, ['model_name', 'attempted', 'parsed', 'eligible', 'eligible_rate'])}

Interpretation:

- GPT-5.4-nano uses the complete later `toptrust_100x2_decisions_end.csv` row-level run. It must not be presented as the same run as the unmatched GPT `recorded run` rows in `all_models_summary.csv`.
- Claude and DeepSeek have high row-level recoverability, but only the output fields saved in their legacy CSV files can be validated.
- Gemini parse-failed rows are excluded because the stored decisions are rule fallbacks. Parse-valid Gemini rows are conditional descriptive evidence only because successful parsing may be selective.

## Recomputed Valid-only Metrics

{markdown_table(model_summary, [
    'model_name', 'decision_mode',
    'malicious_valid_rows', 'malicious_valid_output_coverage', 'malicious_prevent_buy_rate_valid_only',
    'good_valid_rows', 'good_valid_output_coverage', 'good_false_reject_rate_valid_only'
])}

## Automated Claim Screening

{markdown_table(claim_summary, ['model_name', 'automated_claim_check_status', 'rows'])}

Automated claim checks are conservative screening rules for numeric percentile and streak claims when those inputs were saved with the archived decision. A pass is not proof that a reason is hallucination-free; flagged rows require human review in `automated_claim_flags.csv` and `manual_audit_sample.csv`.

## Prompt-rule Consistency

{markdown_table(logic_summary, ['model_name', 'decision_mode', 'checked_rows', 'consistent_rows', 'prompt_rule_consistency_rate'])}

This check asks whether the saved `market_view` and `reputation_view` imply the saved `risk_label` under the prompt's explicit aggregation rule. It measures internal decision consistency, not whether either evidence view is factually grounded.

## Current-panel Alignment

{markdown_table(panel_alignment_summary, ['current_panel_alignment', 'rows'])}

Matching identifiers do not prove that the current panel is the exact historical prompt input. Archived per-row fields take precedence whenever they are available.

## Contract-level Validation

- Metrics in `contract_macro_bootstrap.csv` first collapse repeated copies of the same contract-window, then average within contracts.
- Confidence intervals resample contracts, not rows.
- `small_cluster_warning=1` marks strata with fewer than 10 independent contracts.
- The archived 100-row sets contain substantially fewer independent contracts, so row-level percentages must be described as descriptive rather than as 100 independent project observations.

Contract-level strata generated: {len(contract_summary)}.

## Evidence Limits That Cannot Be Recovered

- Exact rendered prompts, complete provider responses, request IDs, and exact backend revisions were not retained for the buy-screening runs.
- The current `contract_window_panel.csv` has drifted from archived per-row values for some matching identifiers. It is retained as contextual data, not treated as exact historical prompt evidence.
- A parse-valid JSON object is not by itself proof of semantic correctness.
- The later auditable GPT run cannot retroactively validate the different GPT summary-only run.
- Single-run archives cannot establish repeated-run stability.
- Human review is still required for semantic claims not covered by deterministic checks.

## Recommended Reporting Language

The archived results were subjected to post-hoc validity checks. Performance metrics were recomputed only from parse-valid, structurally valid, non-fallback outputs. Claims were checked only against exact historical fields retained beside each decision; the current panel was used as contextual data where version drift prevented exact reconstruction. Contract-clustered uncertainty and a stratified human evidence audit were added. These checks improve traceability but do not establish full reproducibility of discontinued model backends or eliminate all semantic hallucination risk.
"""
    (OUTPUT_DIR / "POSTHOC_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")


def write_manual_audit_protocol() -> None:
    protocol = """# Manual Semantic Audit Protocol

Use `manual_audit_sample.csv` without consulting `sample_group` or collection labels when judging the explanation itself.

## Allowed labels

- `PASS`: the saved evidence supports the statement.
- `FAIL`: the statement contradicts saved evidence or introduces an unsupported factual claim.
- `UNVERIFIABLE`: the exact historical input needed to judge the statement was not archived.

## Columns to complete

- `human_evidence_supported`: Judge factual support. Use `UNVERIFIABLE` when `historical_input_scope=identifiers_and_output_only`.
- `human_no_unsupported_claims`: Mark `FAIL` if the reason introduces facts absent from the archived evidence.
- `human_decision_reason_consistent`: Judge whether the action follows from the reason, independently of whether the reason is factually supported.
- `human_auditor`: Stable reviewer identifier.
- `human_audit_notes`: Briefly identify the conflicting or missing evidence.

## Procedure

1. Hide `sample_group` and ground-truth outcome labels during the first-pass review.
2. Review the saved decision, reason, and exact archived fields first.
3. Treat `current_panel_*` fields as context only when `current_panel_alignment` is not `archived_fields_match`.
4. Do not convert missing evidence into either a pass or a hallucination; use `UNVERIFIABLE`.
5. Prefer two independent reviewers and adjudicate disagreements. Report agreement before adjudication.
6. Report semantic-error rates with the number of verifiable rows as the denominator.

The audit cannot reconstruct fields that were never saved, so its strongest defensible result is a measured error rate on verifiable claims plus an explicit unverifiable fraction.
"""
    (OUTPUT_DIR / "MANUAL_AUDIT_PROTOCOL.md").write_text(protocol, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rule_frame = load_source(RULE_SOURCE)
    rule_reasons = set(rule_frame["reason"].fillna("").astype(str).str.strip())
    frames = [load_source(spec) for spec in SOURCES]
    combined = pd.concat([rule_frame, *frames], ignore_index=True, sort=False)
    combined = add_validation_fields(combined, rule_reasons)

    panel = prepare_panel()
    combined = combined.merge(panel, on=["contract_id", "snapshot_key"], how="left", validate="many_to_one")
    combined["current_panel_row_found"] = combined["current_panel_collection_type"].notna()

    combined["archived_rep_rank_pct"] = pd.to_numeric(combined.get("rep_rank_pct"), errors="coerce")
    combined["archived_low_rep_streak"] = pd.to_numeric(combined.get("low_rep_streak"), errors="coerce")
    combined["historical_input_scope"] = "identifiers_and_output_only"
    combined.loc[
        combined["archived_rep_rank_pct"].notna() & combined["archived_low_rep_streak"].notna(),
        "historical_input_scope",
    ] = "archived_rep_rank_and_streak"
    current_rank = pd.to_numeric(combined.get("current_panel_rep_rank_pct"), errors="coerce")
    current_streak = pd.to_numeric(combined.get("current_panel_low_rep_streak"), errors="coerce")
    comparable = combined["archived_rep_rank_pct"].notna() & combined["archived_low_rep_streak"].notna()
    rank_match = np.isclose(combined["archived_rep_rank_pct"], current_rank, equal_nan=False)
    streak_match = combined["archived_low_rep_streak"].eq(current_streak)
    combined["current_panel_alignment"] = "not_comparable_no_archived_fields"
    combined.loc[comparable & rank_match & streak_match, "current_panel_alignment"] = "archived_fields_match"
    combined.loc[comparable & ~(rank_match & streak_match), "current_panel_alignment"] = "archived_fields_mismatch"

    claim_results = combined.apply(automated_claim_check, axis=1, result_type="expand")
    claim_results.columns = [
        "automated_claim_check_status",
        "automated_claims_checked",
        "automated_claim_flags",
    ]
    combined = pd.concat([combined, claim_results], axis=1)

    validation_summary = (
        combined.groupby(["model_name", "decision_mode", "sample_group", "evidence_scope"], as_index=False, sort=False)
        .agg(
            rows=("decision", "size"),
            parse_valid_rows=("parse_valid", "sum"),
            saved_fields_valid_rows=("saved_fields_valid", "sum"),
            fallback_suspected_rows=("fallback_suspected", "sum"),
            eligible_rows=("included_in_valid_metrics", "sum"),
            current_panel_row_found=("current_panel_row_found", "sum"),
            current_panel_mismatch_rows=(
                "current_panel_alignment",
                lambda values: int((values == "archived_fields_mismatch").sum()),
            ),
            raw_response_available_rows=("raw_response_available", "sum"),
            logic_check_available_rows=("prompt_rule_consistent", "count"),
            logic_consistent_rows=("prompt_rule_consistent", "sum"),
        )
    )
    validation_summary["parse_valid_rate"] = validation_summary["parse_valid_rows"] / validation_summary["rows"]
    validation_summary["eligible_rate"] = validation_summary["eligible_rows"] / validation_summary["rows"]
    validation_summary["current_panel_identifier_join_rate"] = (
        validation_summary["current_panel_row_found"] / validation_summary["rows"]
    )
    validation_summary["prompt_rule_consistency_rate"] = (
        validation_summary["logic_consistent_rows"] / validation_summary["logic_check_available_rows"]
    )

    group_metrics = build_group_metrics(combined)
    model_summary = build_model_summary(group_metrics)
    contract_level, contract_summary = build_contract_metrics(combined)
    manual_sample = build_manual_audit_sample(combined)
    claim_summary = (
        combined.loc[combined["included_in_valid_metrics"] & combined["model_name"].ne("rule_agent")]
        .groupby(["model_name", "automated_claim_check_status"], as_index=False, sort=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    logic_rows = combined.loc[
        combined["included_in_valid_metrics"]
        & combined["model_name"].ne("rule_agent")
        & combined["prompt_rule_consistent"].notna()
    ].copy()
    logic_summary = (
        logic_rows.groupby(["model_name", "decision_mode"], as_index=False, sort=False)
        .agg(
            checked_rows=("prompt_rule_consistent", "size"),
            consistent_rows=("prompt_rule_consistent", "sum"),
        )
    )
    logic_summary["prompt_rule_consistency_rate"] = (
        logic_summary["consistent_rows"] / logic_summary["checked_rows"]
    )
    panel_alignment_summary = (
        combined.groupby("current_panel_alignment", as_index=False, sort=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    claim_flags = combined.loc[
        combined["automated_claim_check_status"] == "flagged_for_human_review"
    ].copy()

    combined.to_csv(OUTPUT_DIR / "validated_decisions.csv", index=False, encoding="utf-8-sig")
    validation_summary.to_csv(OUTPUT_DIR / "validation_coverage_summary.csv", index=False, encoding="utf-8-sig")
    group_metrics.to_csv(OUTPUT_DIR / "valid_only_group_metrics.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(OUTPUT_DIR / "validated_model_summary.csv", index=False, encoding="utf-8-sig")
    contract_level.to_csv(OUTPUT_DIR / "contract_level_metrics.csv", index=False, encoding="utf-8-sig")
    contract_summary.to_csv(OUTPUT_DIR / "contract_macro_bootstrap.csv", index=False, encoding="utf-8-sig")
    claim_summary.to_csv(OUTPUT_DIR / "automated_claim_check_summary.csv", index=False, encoding="utf-8-sig")
    logic_summary.to_csv(OUTPUT_DIR / "decision_logic_consistency_summary.csv", index=False, encoding="utf-8-sig")
    claim_flags.to_csv(OUTPUT_DIR / "automated_claim_flags.csv", index=False, encoding="utf-8-sig")
    panel_alignment_summary.to_csv(OUTPUT_DIR / "current_panel_alignment_summary.csv", index=False, encoding="utf-8-sig")
    manual_sample.to_csv(OUTPUT_DIR / "manual_audit_sample.csv", index=False, encoding="utf-8-sig")
    paper_table_manual_sample = manual_sample.loc[
        manual_sample["model_name"].isin(PAPER_TABLE_MODELS)
        & manual_sample["decision_mode"].isin(PAPER_TABLE_MODES)
    ].copy()
    paper_table_manual_sample.to_csv(
        OUTPUT_DIR / "manual_audit_sample_paper_table.csv", index=False, encoding="utf-8-sig"
    )
    build_source_provenance().to_csv(OUTPUT_DIR / "source_provenance.csv", index=False, encoding="utf-8-sig")
    build_archived_summary_audit().to_csv(OUTPUT_DIR / "archived_summary_audit.csv", index=False, encoding="utf-8-sig")
    write_report(
        validation_summary,
        model_summary,
        claim_summary,
        logic_summary,
        contract_summary,
        panel_alignment_summary,
    )
    write_manual_audit_protocol()

    print(f"[Saved] {OUTPUT_DIR.resolve()}")
    print(f"[Rows] validated_decisions={len(combined)}, manual_audit_sample={len(manual_sample)}")


if __name__ == "__main__":
    main()
