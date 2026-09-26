from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path("output/D/15_buy_screening")
OUTPUT_DIR = Path("output/D/18_agent_logic_validation")
Z_95 = 1.959963984540054

VALID_VIEWS = {"positive", "neutral", "negative"}
VALID_RISK_LABELS = {"RISK", "NO_RISK"}
VALID_DECISIONS = {"AVOID_BUY", "ALLOW_BUY"}
TABLE_MODES = {"llm_base", "llm_base_rank", "llm_rank"}


@dataclass(frozen=True)
class SourceSpec:
    filename: str
    model_name: str
    mode: str | None = None


SOURCES = (
    SourceSpec("toptrust_100x2_decisions_end.csv", "gpt-5.4-nano"),
    SourceSpec("toptrust_100x2_grok41fast_llm_base_decisions.csv", "grok-4-1-fast", "llm_base"),
    SourceSpec("toptrust_100x2_grok41fast_llm_base_rank_decisions.csv", "grok-4-1-fast", "llm_base_rank"),
    SourceSpec("toptrust_100x2_grok41fast_llm_rank_decisions.csv", "grok-4-1-fast", "llm_rank"),
    SourceSpec(
        "toptrust_100x2_claudehaiku_llm_base_decisions.csv",
        "claude-haiku-4-5-20251001",
        "llm_base",
    ),
    SourceSpec(
        "toptrust_100x2_claudehaiku_llm_base_rank_decisions.csv",
        "claude-haiku-4-5-20251001",
        "llm_base_rank",
    ),
    SourceSpec(
        "toptrust_100x2_claudehaiku_llm_rank_decisions.csv",
        "claude-haiku-4-5-20251001",
        "llm_rank",
    ),
    SourceSpec(
        "toptrust_100x2_gemini25flashnothinking_llm_base_decisions.csv",
        "gemini-2.5-flash-nothinking",
        "llm_base",
    ),
    SourceSpec(
        "toptrust_100x2_gemini25flashnothinking_llm_base_rank_decisions.csv",
        "gemini-2.5-flash-nothinking",
        "llm_base_rank",
    ),
    SourceSpec(
        "toptrust_100x2_gemini25flashnothinking_llm_rank_decisions.csv",
        "gemini-2.5-flash-nothinking",
        "llm_rank",
    ),
)


def expected_risk_label(market_view: str, reputation_view: str) -> str:
    """Apply the explicit evidence-combination rule from the archived prompt."""
    views = {market_view, reputation_view}
    if "negative" in views and "positive" not in views:
        return "RISK"
    return "NO_RISK"


def expected_buy_decision(risk_label: str) -> str:
    return "AVOID_BUY" if risk_label == "RISK" else "ALLOW_BUY"


def wilson_interval(errors: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = errors / total
    denominator = 1 + Z_95**2 / total
    center = (p + Z_95**2 / (2 * total)) / denominator
    half_width = Z_95 * np.sqrt(p * (1 - p) / total + Z_95**2 / (4 * total**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def load_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in SOURCES:
        path = INPUT_DIR / spec.filename
        frame = pd.read_csv(path)
        frame["source_file"] = spec.filename
        frame["source_row"] = np.arange(len(frame), dtype=int) + 2
        frame["model_name"] = spec.model_name
        if spec.mode is not None:
            frame = frame.loc[frame["decision_mode"].astype(str) == spec.mode].copy()
        else:
            frame = frame.loc[frame["decision_mode"].astype(str).isin(TABLE_MODES)].copy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def validate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["market_view"] = result["market_view"].fillna("").astype(str).str.strip().str.lower()
    result["reputation_view"] = result["reputation_view"].fillna("").astype(str).str.strip().str.lower()
    result["risk_label"] = result["risk_label"].fillna("").astype(str).str.strip().str.upper()
    result["decision"] = result["decision"].fillna("").astype(str).str.strip().str.upper()

    result["fields_valid"] = (
        result["market_view"].isin(VALID_VIEWS)
        & result["reputation_view"].isin(VALID_VIEWS)
        & result["risk_label"].isin(VALID_RISK_LABELS)
        & result["decision"].isin(VALID_DECISIONS)
    )
    result["expected_risk_label"] = result.apply(
        lambda row: expected_risk_label(row["market_view"], row["reputation_view"])
        if row["fields_valid"]
        else "INVALID_INPUT",
        axis=1,
    )
    result["expected_decision_from_risk_label"] = result["risk_label"].map(
        {"RISK": "AVOID_BUY", "NO_RISK": "ALLOW_BUY"}
    ).fillna("INVALID_INPUT")
    result["expected_decision_from_views"] = result["expected_risk_label"].map(
        {"RISK": "AVOID_BUY", "NO_RISK": "ALLOW_BUY"}
    ).fillna("INVALID_INPUT")

    result["view_to_risk_consistent"] = result["fields_valid"] & result["risk_label"].eq(
        result["expected_risk_label"]
    )
    result["risk_to_decision_consistent"] = result["fields_valid"] & result["decision"].eq(
        result["expected_decision_from_risk_label"]
    )
    result["end_to_end_consistent"] = result["fields_valid"] & result["decision"].eq(
        result["expected_decision_from_views"]
    )
    result["logic_violation"] = ~(
        result["fields_valid"]
        & result["view_to_risk_consistent"]
        & result["risk_to_decision_consistent"]
    )

    result["violation_type"] = ""
    result.loc[~result["fields_valid"], "violation_type"] = "invalid_or_missing_enum"
    result.loc[
        result["fields_valid"] & ~result["view_to_risk_consistent"], "violation_type"
    ] = "evidence_views_do_not_match_risk_label"
    result.loc[
        result["fields_valid"]
        & result["view_to_risk_consistent"]
        & ~result["risk_to_decision_consistent"],
        "violation_type",
    ] = "risk_label_does_not_match_decision"
    result.loc[
        result["fields_valid"]
        & ~result["view_to_risk_consistent"]
        & ~result["risk_to_decision_consistent"],
        "violation_type",
    ] = "both_rule_stages_inconsistent"
    return result


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, sort=False, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        total = len(group)
        errors = int(group["logic_violation"].sum())
        lower, upper = wilson_interval(errors, total)
        rate = errors / total if total else np.nan
        row = dict(zip(group_columns, key_values))
        row.update(
            {
                "n": total,
                "invalid_field_rows": int((~group["fields_valid"]).sum()),
                "view_to_risk_violations": int((~group["view_to_risk_consistent"]).sum()),
                "risk_to_decision_violations": int((~group["risk_to_decision_consistent"]).sum()),
                "logic_violation_rows": errors,
                "logic_violation_rate": rate,
                "logic_consistency_rate": 1 - rate if total else np.nan,
                "logic_violation_ci95_lower": lower,
                "logic_violation_ci95_upper": upper,
                "logic_violation_ci95_half_range": (upper - lower) / 2 if total else np.nan,
                "logic_error_table_display": f"{rate:.3f} [{lower:.3f}, {upper:.3f}]",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_rule_matrix() -> None:
    rows = []
    for market_view in ("positive", "neutral", "negative"):
        for reputation_view in ("positive", "neutral", "negative"):
            risk_label = expected_risk_label(market_view, reputation_view)
            rows.append(
                {
                    "market_view": market_view,
                    "reputation_view": reputation_view,
                    "expected_risk_label": risk_label,
                    "expected_decision": expected_buy_decision(risk_label),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "logic_rule_matrix.csv", index=False, encoding="utf-8-sig")


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Agent Logic-Violation Audit",
        "",
        "This audit checks deterministic consistency among `market_view`, `reputation_view`, `risk_label`, and `decision`.",
        "It is an internal-logic audit, not a factual hallucination detector.",
        "",
        "## Rule",
        "",
        "- If at least one evidence view is `negative` and neither view is `positive`, expect `RISK` and `AVOID_BUY`.",
        "- Otherwise, expect `NO_RISK` and `ALLOW_BUY`.",
        "- Independently, `RISK` must map to `AVOID_BUY`, and `NO_RISK` must map to `ALLOW_BUY`.",
        "",
        "## Table-ready Results",
        "",
        "| Model | Mode | N | Logic violations | Logic error [95% Wilson CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {row.decision_mode} | {row.n} | "
            f"{row.logic_violation_rows} | {row.logic_error_table_display} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A logic violation means that the four saved output fields contradict the explicit prompt rule. "
            "It may indicate instruction-following failure or internal inconsistency, but it does not prove that "
            "the underlying market or ARep statements are factually hallucinated.",
        ]
    )
    (OUTPUT_DIR / "LOGIC_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    decisions = validate_rows(load_rows())
    summary = summarize(decisions, ["model_name", "decision_mode"])
    summary_by_group = summarize(decisions, ["model_name", "decision_mode", "sample_group"])

    decisions.to_csv(OUTPUT_DIR / "logic_validation_rows.csv", index=False, encoding="utf-8-sig")
    decisions.loc[decisions["logic_violation"]].to_csv(
        OUTPUT_DIR / "logic_violation_examples.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(OUTPUT_DIR / "logic_violation_summary.csv", index=False, encoding="utf-8-sig")
    summary_by_group.to_csv(
        OUTPUT_DIR / "logic_violation_summary_by_group.csv", index=False, encoding="utf-8-sig"
    )
    write_rule_matrix()
    write_report(summary)

    print(f"[Saved] {OUTPUT_DIR.resolve()}")
    print(f"[Rows] total={len(decisions)}, violations={int(decisions['logic_violation'].sum())}")


if __name__ == "__main__":
    main()
