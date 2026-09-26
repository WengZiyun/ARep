# Agent Experimental Details

This directory archives the code, prompts, notes, inputs, and generated results used in the NFT risk-agent experiments.

## Directory layout

- `code/python/`: Python scripts for panel construction, Eliza/LLM calls, evaluation, and plotting.
- `code/eliza-nft-risk-agent/`: Eliza agent source code and reproducible project configuration.
- `documents/`: Prompt templates, experiment notes, and the Eliza integration plan.
- `results/output_D/`: Existing agent input panels, decisions, evaluation tables, and figures.
- `results/reference_inputs/3_collectionlabel.csv`: Labels used by the evaluation scripts.
- `SHA256SUMS.csv`: Relative file paths and SHA-256 hashes for integrity checks.

## Available result sets

- `7_agent_panel`: Structured contract-window inputs supplied to the agents.
- `8_agent_decisions`: Eliza decision records, raw decision text, errors, and run metadata.
- `9_agent_returns`: Return-based evaluation outputs.
- `11_agent_ablation_plots`: Agent ablation summaries and plots.
- `13_agent_timing`: Timing evaluation tables.
- `14_agent_timing_plots`: Timing figures and supporting summaries.
- `15_buy_screening`: Buy-screening decisions, errors, summaries, and metadata.
- `17_agent_posthoc_validation`: Post-hoc parse/fallback validation, valid-only metrics, contract-clustered uncertainty, source provenance, and a manual semantic-audit sample.
- `18_agent_logic_validation`: Deterministic four-field logic audit for market view, ARep view, risk label, and final buy decision, including row-level violations and Wilson 95% intervals.
- `19_manual_hallucination_audit`: A 120-row human-review template and summarizer for hallucination rate, unverifiable rate, and 95% uncertainty intervals.

For the discontinued-model runs, start with `results/output_D/17_agent_posthoc_validation/POSTHOC_VALIDATION_REPORT.md`. The original result files are preserved unchanged. The post-hoc outputs distinguish row-level auditable runs from summary-only records and do not treat the current input panel as exact historical prompt evidence where archived fields show version drift.

No generated output directories currently exist for `10_call_llm_agent.py` or the older `7_eval_rugpull_eliza_agent.py`; their source code is included.

## Security and size exclusions

The archive intentionally excludes `.env`, `node_modules`, `dist`, `.eliza`, local logs, and build cache files. API credentials must be supplied through environment variables when rerunning the experiments.
