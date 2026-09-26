from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AUDIT_DIR = Path("output/D/19_manual_hallucination_audit")
INPUT_CSV = AUDIT_DIR / "manual_audit_120_to_fill.csv"
OUTPUT_ROWS_CSV = AUDIT_DIR / "manual_audit_120_classified.csv"
OUTPUT_SUMMARY_CSV = AUDIT_DIR / "manual_hallucination_summary.csv"
OUTPUT_REPORT_MD = AUDIT_DIR / "MANUAL_HALLUCINATION_REPORT.md"

VALID_LABELS = {"PASS", "FAIL", "UNVERIFIABLE"}
Z_95 = 1.959963984540054


def normalize_label(value: object) -> str:
    text = str(value).strip().upper()
    return text if text in VALID_LABELS else ""


def classify_semantic_hallucination(row: pd.Series) -> str:
    evidence = normalize_label(row.get("human_evidence_supported"))
    unsupported = normalize_label(row.get("human_no_unsupported_claims"))
    if not evidence or not unsupported:
        return "NOT_REVIEWED"
    if evidence == "FAIL" or unsupported == "FAIL":
        return "HALLUCINATION"
    if evidence == "PASS" and unsupported == "PASS":
        return "NO_HALLUCINATION"
    return "UNVERIFIABLE"


def wilson_interval(errors: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = errors / total
    denominator = 1 + Z_95**2 / total
    center = (p + Z_95**2 / (2 * total)) / denominator
    half_width = Z_95 * np.sqrt(p * (1 - p) / total + Z_95**2 / (4 * total**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def summarize_group(group: pd.DataFrame) -> dict[str, object]:
    counts = group["semantic_hallucination_class"].value_counts()
    hallucination = int(counts.get("HALLUCINATION", 0))
    no_hallucination = int(counts.get("NO_HALLUCINATION", 0))
    unverifiable = int(counts.get("UNVERIFIABLE", 0))
    not_reviewed = int(counts.get("NOT_REVIEWED", 0))
    verifiable = hallucination + no_hallucination
    rate = hallucination / verifiable if verifiable else np.nan
    lower, upper = wilson_interval(hallucination, verifiable)
    margin = max(rate - lower, upper - rate) if verifiable else np.nan
    total = len(group)
    return {
        "total_rows": total,
        "verifiable_rows": verifiable,
        "hallucination_rows": hallucination,
        "no_hallucination_rows": no_hallucination,
        "unverifiable_rows": unverifiable,
        "not_reviewed_rows": not_reviewed,
        "hallucination_rate_verifiable": rate,
        "hallucination_ci95_lower": lower,
        "hallucination_ci95_upper": upper,
        "hallucination_pm95_conservative_margin": margin,
        "hallucination_rate_pm95_display": (
            f"{rate:.3f} ± {margin:.3f}" if verifiable else "NA"
        ),
        "hallucination_rate_ci95_display": (
            f"{rate:.3f} [{lower:.3f}, {upper:.3f}]" if verifiable else "NA"
        ),
        "unverifiable_rate_all_rows": unverifiable / total if total else np.nan,
        "minimum_hallucination_rate_all_rows": hallucination / total if total else np.nan,
        "maximum_hallucination_rate_all_rows": (
            (hallucination + unverifiable + not_reviewed) / total if total else np.nan
        ),
    }


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append({"summary_level": "overall", "model_name": "ALL", "decision_mode": "ALL", **summarize_group(frame)})

    for model_name, group in frame.groupby("model_name", sort=False):
        rows.append(
            {
                "summary_level": "model",
                "model_name": model_name,
                "decision_mode": "ALL",
                **summarize_group(group),
            }
        )

    for (model_name, decision_mode), group in frame.groupby(["model_name", "decision_mode"], sort=False):
        rows.append(
            {
                "summary_level": "model_mode",
                "model_name": model_name,
                "decision_mode": decision_mode,
                **summarize_group(group),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame) -> None:
    overall = summary.loc[summary["summary_level"] == "overall"].iloc[0]
    report = f"""# Manual Hallucination Audit Summary

## Completion

- Total sampled rows: {int(overall['total_rows'])}
- Verifiable reviewed rows: {int(overall['verifiable_rows'])}
- Unverifiable rows: {int(overall['unverifiable_rows'])}
- Not-reviewed rows: {int(overall['not_reviewed_rows'])}

## Overall Estimate

- Hallucination rows: {int(overall['hallucination_rows'])}
- Hallucination rate among verifiable rows: {overall['hallucination_rate_ci95_display']}
- Optional symmetric table display: {overall['hallucination_rate_pm95_display']}
- Unverifiable rate: {overall['unverifiable_rate_all_rows']:.3f}
- All-row identification bounds: [{overall['minimum_hallucination_rate_all_rows']:.3f}, {overall['maximum_hallucination_rate_all_rows']:.3f}]

The preferred report is the estimate with the Wilson 95% interval. The symmetric `±` value is supplied only for table formatting and uses the larger distance from the estimate to either Wilson bound. Per-model-mode estimates use only 10 sampled rows and should be treated as exploratory.
"""
    OUTPUT_REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Audit template not found: {INPUT_CSV}")
    frame = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    required = {"human_evidence_supported", "human_no_unsupported_claims"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required audit columns: {sorted(missing)}")

    frame["semantic_hallucination_class"] = frame.apply(classify_semantic_hallucination, axis=1)
    summary = build_summary(frame)
    frame.to_csv(OUTPUT_ROWS_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    write_report(summary)

    overall = summary.loc[summary["summary_level"] == "overall"].iloc[0]
    print(f"[Saved] {AUDIT_DIR.resolve()}")
    print(
        "[Audit] "
        f"verifiable={int(overall['verifiable_rows'])}, "
        f"unverifiable={int(overall['unverifiable_rows'])}, "
        f"not_reviewed={int(overall['not_reviewed_rows'])}"
    )


if __name__ == "__main__":
    main()
