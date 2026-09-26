# Manual Semantic Audit Protocol

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
