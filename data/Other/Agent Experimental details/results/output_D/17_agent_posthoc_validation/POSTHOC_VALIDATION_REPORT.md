# Post-hoc Validation of Archived LLM-Agent Results

Generated: 2026-09-25T09:09:50.157251+00:00

## Purpose

This audit salvages the archived buy-screening results without calling the discontinued model endpoints. Original files are not modified. The audit separates parse-valid LLM outputs from likely rule fallbacks, validates all retained output fields, checks exact historical fields that were saved beside decisions, and reports contract-level uncertainty.

## Recoverable Output Coverage

| model_name | attempted | parsed | eligible | eligible_rate |
|---|---|---|---|---|
| claude-haiku-4-5-20251001 | 800 | 800 | 800 | 1.000 |
| claude-sonnet-4-6 | 800 | 800 | 800 | 1.000 |
| deepseek-v3.2 | 798 | 798 | 798 | 1.000 |
| gemini-2.5-flash-nothinking | 800 | 800 | 800 | 1.000 |
| gemini-3-flash-preview | 800 | 174 | 174 | 0.217 |
| gpt-5.4-nano | 800 | 800 | 800 | 1.000 |
| grok-4-1-fast | 800 | 800 | 800 | 1.000 |
| rule_agent | 200 | 200 | 200 | 1.000 |

Interpretation:

- GPT-5.4-nano uses the complete later `toptrust_100x2_decisions_end.csv` row-level run. It must not be presented as the same run as the unmatched GPT `recorded run` rows in `all_models_summary.csv`.
- Claude and DeepSeek have high row-level recoverability, but only the output fields saved in their legacy CSV files can be validated.
- Gemini parse-failed rows are excluded because the stored decisions are rule fallbacks. Parse-valid Gemini rows are conditional descriptive evidence only because successful parsing may be selective.

## Recomputed Valid-only Metrics

| model_name | decision_mode | malicious_valid_rows | malicious_valid_output_coverage | malicious_prevent_buy_rate_valid_only | good_valid_rows | good_valid_output_coverage | good_false_reject_rate_valid_only |
|---|---|---|---|---|---|---|---|
| rule_agent | rule_agent | 100 | 1.000 | 0.300 | 100 | 1.000 | 0.010 |
| gpt-5.4-nano | llm_base | 100 | 1.000 | 0.000 | 100 | 1.000 | 0.000 |
| gpt-5.4-nano | llm_base_rank | 100 | 1.000 | 0.570 | 100 | 1.000 | 0.290 |
| gpt-5.4-nano | llm_base_rank_mini | 100 | 1.000 | 0.550 | 100 | 1.000 | 0.030 |
| gpt-5.4-nano | llm_rank | 100 | 1.000 | 0.620 | 100 | 1.000 | 0.000 |
| claude-sonnet-4-6 | llm_base | 100 | 1.000 | 0.470 | 100 | 1.000 | 0.150 |
| claude-sonnet-4-6 | llm_base_rank | 100 | 1.000 | 0.290 | 100 | 1.000 | 0.000 |
| claude-sonnet-4-6 | llm_base_rank_compact | 100 | 1.000 | 0.220 | 100 | 1.000 | 0.020 |
| claude-sonnet-4-6 | llm_base_rank_mini | 100 | 1.000 | 0.220 | 100 | 1.000 | 0.060 |
| deepseek-v3.2 | llm_base | 100 | 1.000 | 0.500 | 100 | 1.000 | 0.010 |
| deepseek-v3.2 | llm_base_rank | 99 | 1.000 | 0.586 | 100 | 1.000 | 0.040 |
| deepseek-v3.2 | llm_base_rank_compact | 100 | 1.000 | 0.570 | 99 | 1.000 | 0.030 |
| deepseek-v3.2 | llm_base_rank_mini | 100 | 1.000 | 0.570 | 100 | 1.000 | 0.030 |
| gemini-3-flash-preview | llm_base | 0 | 0.000 | NA | 0 | 0.000 | NA |
| gemini-3-flash-preview | llm_base_rank | 20 | 0.200 | 0.600 | 33 | 0.330 | 0.000 |
| gemini-3-flash-preview | llm_base_rank_compact | 28 | 0.280 | 0.286 | 8 | 0.080 | 0.125 |
| gemini-3-flash-preview | llm_base_rank_mini | 47 | 0.470 | 0.383 | 38 | 0.380 | 0.000 |
| grok-4-1-fast | llm_base | 100 | 1.000 | 1.000 | 100 | 1.000 | 0.000 |
| grok-4-1-fast | llm_base_rank | 100 | 1.000 | 1.000 | 100 | 1.000 | 0.000 |
| grok-4-1-fast | llm_base_rank_mini | 100 | 1.000 | 0.450 | 100 | 1.000 | 0.040 |
| grok-4-1-fast | llm_rank | 100 | 1.000 | 0.640 | 100 | 1.000 | 0.230 |
| claude-haiku-4-5-20251001 | llm_base | 100 | 1.000 | 1.000 | 100 | 1.000 | 0.000 |
| claude-haiku-4-5-20251001 | llm_base_rank | 100 | 1.000 | 1.000 | 100 | 1.000 | 0.000 |
| claude-haiku-4-5-20251001 | llm_base_rank_mini | 100 | 1.000 | 0.650 | 100 | 1.000 | 0.000 |
| claude-haiku-4-5-20251001 | llm_rank | 100 | 1.000 | 0.970 | 100 | 1.000 | 0.000 |
| gemini-2.5-flash-nothinking | llm_base | 100 | 1.000 | 0.520 | 100 | 1.000 | 0.000 |
| gemini-2.5-flash-nothinking | llm_base_rank | 100 | 1.000 | 0.980 | 100 | 1.000 | 0.010 |
| gemini-2.5-flash-nothinking | llm_base_rank_mini | 100 | 1.000 | 0.650 | 100 | 1.000 | 0.020 |
| gemini-2.5-flash-nothinking | llm_rank | 100 | 1.000 | 0.710 | 100 | 1.000 | 1.000 |

## Automated Claim Screening

| model_name | automated_claim_check_status | rows |
|---|---|---|
| gpt-5.4-nano | no_checkable_numeric_claim | 354 |
| gpt-5.4-nano | no_mismatch_detected | 446 |
| claude-sonnet-4-6 | no_exact_archived_input_fields_to_check | 800 |
| deepseek-v3.2 | no_exact_archived_input_fields_to_check | 798 |
| gemini-3-flash-preview | no_exact_archived_input_fields_to_check | 174 |
| grok-4-1-fast | no_exact_archived_input_fields_to_check | 800 |
| claude-haiku-4-5-20251001 | no_exact_archived_input_fields_to_check | 800 |
| gemini-2.5-flash-nothinking | no_exact_archived_input_fields_to_check | 800 |

Automated claim checks are conservative screening rules for numeric percentile and streak claims when those inputs were saved with the archived decision. A pass is not proof that a reason is hallucination-free; flagged rows require human review in `automated_claim_flags.csv` and `manual_audit_sample.csv`.

## Prompt-rule Consistency

| model_name | decision_mode | checked_rows | consistent_rows | prompt_rule_consistency_rate |
|---|---|---|---|---|
| gpt-5.4-nano | llm_base | 200 | 183 | 0.915 |
| gpt-5.4-nano | llm_base_rank | 200 | 184 | 0.920 |
| gpt-5.4-nano | llm_base_rank_mini | 200 | 200 | 1.000 |
| gpt-5.4-nano | llm_rank | 200 | 200 | 1.000 |
| grok-4-1-fast | llm_base | 200 | 200 | 1.000 |
| grok-4-1-fast | llm_base_rank | 200 | 197 | 0.985 |
| grok-4-1-fast | llm_base_rank_mini | 200 | 186 | 0.930 |
| grok-4-1-fast | llm_rank | 200 | 200 | 1.000 |
| claude-haiku-4-5-20251001 | llm_base | 200 | 200 | 1.000 |
| claude-haiku-4-5-20251001 | llm_base_rank | 200 | 200 | 1.000 |
| claude-haiku-4-5-20251001 | llm_base_rank_mini | 200 | 179 | 0.895 |
| claude-haiku-4-5-20251001 | llm_rank | 200 | 200 | 1.000 |
| gemini-2.5-flash-nothinking | llm_base | 200 | 145 | 0.725 |
| gemini-2.5-flash-nothinking | llm_base_rank | 200 | 200 | 1.000 |
| gemini-2.5-flash-nothinking | llm_base_rank_mini | 200 | 198 | 0.990 |
| gemini-2.5-flash-nothinking | llm_rank | 200 | 200 | 1.000 |

This check asks whether the saved `market_view` and `reputation_view` imply the saved `risk_label` under the prompt's explicit aggregation rule. It measures internal decision consistency, not whether either evidence view is factually grounded.

## Current-panel Alignment

| current_panel_alignment | rows |
|---|---|
| not_comparable_no_archived_fields | 4998 |
| archived_fields_match | 724 |
| archived_fields_mismatch | 76 |

Matching identifiers do not prove that the current panel is the exact historical prompt input. Archived per-row fields take precedence whenever they are available.

## Contract-level Validation

- Metrics in `contract_macro_bootstrap.csv` first collapse repeated copies of the same contract-window, then average within contracts.
- Confidence intervals resample contracts, not rows.
- `small_cluster_warning=1` marks strata with fewer than 10 independent contracts.
- The archived 100-row sets contain substantially fewer independent contracts, so row-level percentages must be described as descriptive rather than as 100 independent project observations.

Contract-level strata generated: 56.

## Evidence Limits That Cannot Be Recovered

- Exact rendered prompts, complete provider responses, request IDs, and exact backend revisions were not retained for the buy-screening runs.
- The current `contract_window_panel.csv` has drifted from archived per-row values for some matching identifiers. It is retained as contextual data, not treated as exact historical prompt evidence.
- A parse-valid JSON object is not by itself proof of semantic correctness.
- The later auditable GPT run cannot retroactively validate the different GPT summary-only run.
- Single-run archives cannot establish repeated-run stability.
- Human review is still required for semantic claims not covered by deterministic checks.

## Recommended Reporting Language

The archived results were subjected to post-hoc validity checks. Performance metrics were recomputed only from parse-valid, structurally valid, non-fallback outputs. Claims were checked only against exact historical fields retained beside each decision; the current panel was used as contextual data where version drift prevented exact reconstruction. Contract-clustered uncertainty and a stratified human evidence audit were added. These checks improve traceability but do not establish full reproducibility of discontinued model backends or eliminate all semantic hallucination risk.
