# Manual Hallucination Audit

Edit `manual_audit_120_to_fill.csv`. For every row, fill:

- `human_evidence_supported`
- `human_no_unsupported_claims`
- `human_decision_reason_consistent`
- `human_auditor`
- `human_audit_notes`

Use only `PASS`, `FAIL`, or `UNVERIFIABLE` in the first three fields.

## Semantic hallucination coding

- `HALLUCINATION`: either evidence support or no-unsupported-claims is `FAIL`.
- `NO_HALLUCINATION`: both fields are `PASS`.
- `UNVERIFIABLE`: neither field fails, but at least one is `UNVERIFIABLE`.
- Blank or unrecognized values remain `NOT_REVIEWED`.

Do not use the ground-truth sample class to judge the explanation. When exact historical evidence is unavailable, use `UNVERIFIABLE`; do not treat missing evidence as a pass.

After all rows are labeled, run:

```powershell
D:\envs\CUDA118COPY\python.exe D_RQ1/19_summarize_manual_hallucination_audit.py
```

The preferred result is `hallucination_rate_ci95_display` in `manual_hallucination_summary.csv`. A symmetric `hallucination_rate_pm95_display` is also generated for a table that requires `estimate \pm margin`, but the Wilson interval is statistically preferable, especially when the observed rate is zero.

Each model-mode subgroup contains only 10 manually sampled rows. Use the 120-row overall estimate as the main manual hallucination rate; subgroup estimates are exploratory unless more rows are reviewed.
