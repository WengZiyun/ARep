# Buy-Side Screening Experiment Note

## Scope
This note records the current buy-side experiment only.

Current target:
- compare different decision modes on whether the agent should `AVOID_BUY` or `ALLOW_BUY`
- compare malicious-product filtering ability
- compare false rejection on high-quality products
- compare token cost

This note is intentionally separated from the earlier sell/hold experiments.

## Current Runtime
- Agent runtime: Eliza plugin route
- Plugin file:
  - [plugin.ts](e:/1_Hot/0_ACode/25-10-25_NFT声誉体系构建/D_RQ1/agent_eliza/nft-risk-agent/src/plugin.ts)
- Caller:
  - [15_eval_buy_screening.py](e:/1_Hot/0_ACode/25-10-25_NFT声誉体系构建/D_RQ1/15_eval_buy_screening.py)
- Input panel:
  - [contract_window_panel.csv](e:/1_Hot/0_ACode/25-10-25_NFT声誉体系构建/output/D/7_agent_panel/contract_window_panel.csv)

## Current Models
- Rule baseline:
  - `rule_agent`
- Current LLM backend under test:
  - `gpt-5.4-nano`

Later tests may replace the LLM backend while keeping the same prompt structure and evaluation table.

## Current Buy-Side Action Space
- `AVOID_BUY`
- `ALLOW_BUY`

Although the plugin still supports `SELL/HOLD`, the current buy-screening experiment only evaluates flat-position purchase decisions.

## Current Prompt Design

### System Prompt Logic
The current system prompt follows these principles:
- use only provided fields
- do not assume future outcomes
- consider two evidence groups with equal importance:
  - market evidence
  - reputation evidence
- first judge each evidence group separately
- then combine them into a final decision
- for buy-side:
  - `AVOID_BUY` when both evidence groups are negative, or when one is clearly negative and the other is at least neutral
  - `ALLOW_BUY` when both are not negative, or when one is negative but the other is clearly positive
- malicious-product risk includes not only price loss, but also the possibility that the product may later be delisted, frozen, or become practically unsellable

### Output Format
Current LLM output is expected to be:

```json
{
  "market_view": "positive|neutral|negative",
  "reputation_view": "positive|neutral|negative",
  "decision": "AVOID_BUY|ALLOW_BUY|SELL|HOLD",
  "confidence": 0.0,
  "reason": "one short sentence"
}
```

For the current buy-side experiment, only `AVOID_BUY` and `ALLOW_BUY` are relevant.

## Current Input Features

### Market evidence
Current prompt uses some or all of the following, depending on the mode:
- `window_last_trade_price_usd`
- `window_trade_count`
- `window_trade_value_total_usd`
- `window_unique_buyers`
- `window_unique_trade_users`
- `window_mint_cnt`
- `current_mint_supply_count`
- `window_gas_total_usd`
- `price_change_vs_prev_window_pct`
- `trade_count_change_vs_prev_window_pct`

Notes:
- `current_mint_supply_count` is currently a proxy for current collection size
- it is implemented as cumulative mint count up to the current window

### Reputation evidence
Current prompt does not expose the raw rank fields directly as naked numbers only.
Instead, it explains the rank signal as:
- a long-term market credibility summary
- derived from contract behavior, user activity, cost commitment, mint/supply behavior, and sustained engagement across windows

Current wording concept:
- `credibility percentile rank: X/100, where a higher value means higher credibility`
- whether the collection has stayed in a low-reputation region for several consecutive windows

This is intended to make the LLM interpret rank as a long-term trust signal rather than as an unexplained risk scalar.

## Current Test Protocol

### Malicious side
- target type:
  - `rugpull`
- threshold:
  - `low_rep_threshold_pct = 0.8`
- sample construction:
  - for each rugpull contract, keep only the single window immediately before the report/detection point
- evaluation sample size:
  - 100 malicious samples

### Good-product side
- use only high-quality original-like products selected by:
  - `selected_for_trust = 1`
- do not use broad `normal` products because their quality is ambiguous
- evaluation sample size:
  - 100 good samples

### Main output table
- [buy_screening_summary.csv](e:/1_Hot/0_ACode/25-10-25_NFT声誉体系构建/output/D/15_buy_screening/buy_screening_summary.csv)

Main fields:
- `malicious_prevent_buy_rate`
- `good_false_reject_rate`
- `mean_total_tokens`
- `mean_prompt_tokens`
- `mean_completion_tokens`
- `mean_latency_ms`
- `parse_success_rate`

## Current Naming Problem
The current internal mode names are convenient for coding, but not yet ideal for final write-up.

### Current code names
- `llm_base`
- `llm_base_rank`
- `llm_base_rank_compact`
- `llm_base_rank_mini`
- `llm_rank`

### Recommended conceptual names for paper/note
- `llm_base`
  - market-only LLM baseline
- `llm_base_rank`
  - full market + rank mode
- `llm_base_rank_compact`
  - compressed market + rank mode
- `llm_base_rank_mini`
  - very small market block + rank mode
- `llm_rank`
  - true rank-only buy-side prompt

### New ablation now added
We now explicitly distinguish:
- `llm_base_rank`
  - base market block + rank block
- `llm_rank`
  - rank-only prompt without the base market feature block

This is important for fairness, because the previous naming mixed the base-plus-rank setting with the true rank-only setting.

## Current Conclusions
Based on the latest `gpt-5.4-nano` run:

- `llm_base`
  - malicious filtering remains weak
  - market information alone is insufficient for this buy-side task

- `llm_base_rank` (current code: `llm_rank`)
  - currently shows the strongest malicious-product filtering ability among LLM modes
  - while keeping false rejection on high-quality products very low
  - this is currently the best full LLM setting

- `llm_base_rank_compact`
  - gives a strong cost-performance tradeoff
  - slightly lower malicious filtering than full rank mode
  - still useful as a compressed prompt candidate

- `llm_base_rank_mini`
  - cheaper than full rank mode
  - still competitive, but slightly weaker and less stable than the full rank version

Overall interpretation:
- explicit explanation of rank as a long-term credibility signal materially improves the usefulness of the rank-enhanced prompt
- prompt wording strongly changes LLM behavior
- rank is currently helpful only when the LLM understands what the rank actually measures

## Next Planned Experiments

### 1. Add true rank-only prompt
Need to add a separate buy-side mode:
- recommended note name: `llm_rank`
- idea:
  - only reputation explanation
  - no explicit market feature block

This will test whether the long-term rank signal alone can screen malicious products.

### 2. Continue cross-model comparison
Keep the same prompt and evaluation table, but replace the backend model, for example:
- `gpt-5.4-nano`
- other smaller / larger models later

### 3. Keep one unified comparison table
Recommended final comparison dimensions:
- model
- prompt mode
- malicious prevent-buy rate
- good false-reject rate
- mean total tokens

This should remain the primary summary format.

## Gemini Run Record

### Runtime status
We attempted to reproduce the same five-group buy-side comparison under a Gemini backend.

Because the currently running Eliza service still does not recognize the newly renamed modes:
- `llm_base_rank`
- `llm_base_rank_compact`
- `llm_base_rank_mini`

the Gemini run was executed with the old mode names and then mapped back in the summary:
- `llm_rank` -> `llm_base_rank`
- `llm_rank_compact` -> `llm_base_rank_compact`
- `llm_rank_mini` -> `llm_base_rank_mini`

### Gemini backend
- model:
  - `gemini-3-flash-preview`

### Gemini 5-mode mapped summary
- output:
  - [gemini_5mode_summary_mapped.csv](e:/1_Hot/0_ACode/25-10-25_NFT声誉体系构建/output/D/15_buy_screening/gemini_5mode_summary_mapped.csv)

Current mapped result:

| decision_mode | model_name | malicious_prevent_buy_rate | good_false_reject_rate | mean_total_tokens | parse_success_rate |
|---|---|---:|---:|---:|---:|
| `rule_agent` | `rule_agent` | `0.30` | `0.01` | `0.00` | `1.00` |
| `llm_base` | `gemini-3-flash-preview` | `0.30` | `0.01` | `1282.87` | `0.00` |
| `llm_base_rank` | `gemini-3-flash-preview` | `0.30` | `0.01` | `1198.05` | `0.265` |
| `llm_base_rank_compact` | `gemini-3-flash-preview` | `0.30` | `0.02` | `1260.44` | `0.18` |
| `llm_base_rank_mini` | `gemini-3-flash-preview` | `0.30` | `0.01` | `1209.42` | `0.425` |

### Interpretation of the Gemini run
This run should be treated with caution.

Reasons:
- the four LLM modes show nearly identical malicious filtering rates
- `parse_success_rate` is very low for Gemini in the current OpenAI-compatible route
- the `llm_base` parse success rate is effectively `0.00`

This strongly suggests that the current Gemini integration is not yet as stable or trustworthy as the earlier GPT-based runs.

Therefore:
- the Gemini run is useful as an operational record
- but it should not yet be treated as a strong comparative conclusion
- before using Gemini results in the main paper table, the parsing and response-format stability should be improved

### Operational note
To finish the Gemini comparison, the experiment was run mode-by-mode and then merged:
- `rule_agent + llm_base`
- `llm_rank`
- `llm_rank_compact`
- `llm_rank_mini`

This was necessary because full five-mode batch execution under Gemini was too slow for a single long run.

## Claude Sonnet Run Record

### Runtime status
We also reproduced the same five-group buy-side comparison under a Claude backend.

As with Gemini, the currently running Eliza service did not expose the renamed modes directly, so the run was executed with the old mode names and then mapped back in the summary:
- `llm_rank` -> `llm_base_rank`
- `llm_rank_compact` -> `llm_base_rank_compact`
- `llm_rank_mini` -> `llm_base_rank_mini`

### Claude backend
- model:
  - `claude-sonnet-4-6`

### Claude 5-mode mapped summary
- output:
  - [claude_5mode_summary_mapped.csv](e:/1_Hot/0_ACode/25-10-25_NFT声誉体系构建/output/D/15_buy_screening/claude_5mode_summary_mapped.csv)

Current mapped result:

| decision_mode | model_name | malicious_prevent_buy_rate | good_false_reject_rate | mean_total_tokens | parse_success_rate |
|---|---|---:|---:|---:|---:|
| `rule_agent` | `rule_agent` | `0.30` | `0.01` | `0.00` | `1.00` |
| `llm_base` | `claude-sonnet-4-6` | `0.47` | `0.15` | `1044.23` | `1.00` |
| `llm_base_rank` | `claude-sonnet-4-6` | `0.29` | `0.00` | `976.91` | `1.00` |
| `llm_base_rank_compact` | `claude-sonnet-4-6` | `0.22` | `0.02` | `1036.88` | `1.00` |
| `llm_base_rank_mini` | `claude-sonnet-4-6` | `0.22` | `0.06` | `1000.58` | `1.00` |

### Interpretation of the Claude run
Compared with the Gemini run, Claude is much more stable in the current OpenAI-compatible route:
- all LLM groups achieved `parse_success_rate = 1.00`
- the outputs are therefore much more usable for comparison

Current interpretation:
- `llm_base`
  - strongest malicious filtering among Claude LLM modes
  - but also the highest false-reject rate on good products
- `llm_base_rank`
  - lower malicious filtering than `llm_base`
  - but the false-reject rate drops to `0.00`
  - this is currently the cleanest precision-oriented Claude setting
- `llm_base_rank_compact`
  - weaker malicious filtering than the full rank setting
  - slightly higher false reject than full rank
- `llm_base_rank_mini`
  - similar malicious filtering to compact
  - but a higher false-reject rate than compact

Overall interpretation:
- under Claude, the market-only prompt is the most aggressive at blocking malicious products, but it also over-rejects good products
- adding the long-term rank explanation appears to improve buy-side precision substantially
- under the current Claude setting, `llm_base_rank` is the most balanced reputation-enhanced option

### Operational note
To finish the Claude comparison, the experiment was also run mode-by-mode and then merged:
- `rule_agent + llm_base`
- `llm_rank`
- `llm_rank_compact`
- `llm_rank_mini`

This stepwise execution was used to avoid losing intermediate results during long-running model calls.

## DeepSeek Run Record

### Runtime status
We also reproduced the same five-group buy-side comparison under a DeepSeek backend.

As with Gemini and Claude, the currently running Eliza service still exposed the legacy working mode names more reliably, so the run was executed with the old mode names and then mapped back in the summary:
- `llm_rank` -> `llm_base_rank`
- `llm_rank_compact` -> `llm_base_rank_compact`
- `llm_rank_mini` -> `llm_base_rank_mini`

### DeepSeek backend
- model:
  - `deepseek-v3.2`

### DeepSeek 5-mode mapped summary
- output:
  - [deepseek_5mode_summary_mapped.csv](e:/1_Hot/0_ACode/25-10-25_NFT声誉体系构建/output/D/15_buy_screening/deepseek_5mode_summary_mapped.csv)

Current mapped result:

| decision_mode | model_name | malicious_prevent_buy_rate | good_false_reject_rate | mean_total_tokens | parse_success_rate |
|---|---|---:|---:|---:|---:|
| `rule_agent` | `rule_agent` | `0.30` | `0.01` | `0.00` | `1.00` |
| `llm_base` | `deepseek-v3.2` | `0.50` | `0.01` | `917.71` | `1.00` |
| `llm_base_rank` | `deepseek-v3.2` | `0.5859` | `0.04` | `868.40` | `1.00` |
| `llm_base_rank_compact` | `deepseek-v3.2` | `0.57` | `0.0303` | `916.76` | `1.00` |
| `llm_base_rank_mini` | `deepseek-v3.2` | `0.57` | `0.03` | `886.52` | `1.00` |

### Interpretation of the DeepSeek run
DeepSeek is substantially more stable than Gemini under the current OpenAI-compatible route:
- all tested LLM groups achieved `parse_success_rate = 1.00`
- outputs remained structured and directly usable for comparison

Current interpretation:
- `llm_base`
  - already shows strong malicious-product filtering
  - while keeping false rejection on good products low
- `llm_base_rank`
  - further improves malicious filtering beyond `llm_base`
  - with only a small increase in false rejection
  - and uses fewer tokens than the market-only `llm_base`
- `llm_base_rank_compact`
  - remains very close to the full rank setting in malicious filtering
  - with slightly lower false rejection than the full rank setting
- `llm_base_rank_mini`
  - keeps nearly the same malicious filtering as compact
  - while remaining one of the cheaper DeepSeek rank-enhanced variants

Overall interpretation:
- under DeepSeek, the long-term rank explanation is clearly helpful
- unlike the Claude run, the rank-enhanced variants improve malicious filtering rather than mainly trading recall for precision
- under the current DeepSeek setting, `llm_base_rank` is the strongest full setting, while `llm_base_rank_mini` is a strong low-cost alternative

### Operational note
To finish the DeepSeek comparison, the experiment was again run mode-by-mode and then merged:
- `rule_agent + llm_base`
- `llm_rank`
- `llm_rank_compact`
- `llm_rank_mini`

During this run, the `llm_rank_mini` result file was missing after a crash and had to be rerun once, then copied into:
- [deepseek_llm_rank_mini_summary.csv](e:/1_Hot/0_ACode/25-10-25_NFT声誉体系构建/output/D/15_buy_screening/deepseek_llm_rank_mini_summary.csv)

This confirms that the stepwise execution pattern remains the safest way to preserve intermediate results for long-running external model tests.
