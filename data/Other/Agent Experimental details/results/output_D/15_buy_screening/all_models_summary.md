# All Buy-Side Model Summary

Combined table across all tested backends for the buy-side screening task.

- malicious sample: 100
- good sample: 100
- decision modes reported with paper-facing names
- `llm_base_rank*` rows for Gemini / Claude / DeepSeek are mapped from legacy runtime names:
  - `llm_rank`
  - `llm_rank_compact`
  - `llm_rank_mini`

## Summary Table

| model_name | decision_mode | malicious_prevent_buy_rate | good_false_reject_rate | mean_total_tokens | parse_success_rate |
|---|---|---:|---:|---:|---:|
| `gpt-5.4-nano` | `rule_agent` | `0.30` | `0.01` | `0.00` | `1.00` |
| `gpt-5.4-nano` | `llm_base` | `0.07` | `0.00` | `885.55` | `NA` |
| `gpt-5.4-nano` | `llm_base_rank` | `0.49` | `0.00` | `963.22` | `NA` |
| `gpt-5.4-nano` | `llm_base_rank_compact` | `0.39` | `0.02` | `891.50` | `NA` |
| `gpt-5.4-nano` | `llm_base_rank_mini` | `0.44` | `0.06` | `864.89` | `NA` |
| `gemini-3-flash-preview` | `rule_agent` | `0.30` | `0.01` | `0.00` | `1.00` |
| `gemini-3-flash-preview` | `llm_base` | `0.30` | `0.01` | `1282.87` | `0.00` |
| `gemini-3-flash-preview` | `llm_base_rank` | `0.30` | `0.01` | `1198.05` | `0.265` |
| `gemini-3-flash-preview` | `llm_base_rank_compact` | `0.30` | `0.02` | `1260.44` | `0.18` |
| `gemini-3-flash-preview` | `llm_base_rank_mini` | `0.30` | `0.01` | `1209.42` | `0.425` |
| `claude-sonnet-4-6` | `rule_agent` | `0.30` | `0.01` | `0.00` | `1.00` |
| `claude-sonnet-4-6` | `llm_base` | `0.47` | `0.15` | `1044.23` | `1.00` |
| `claude-sonnet-4-6` | `llm_base_rank` | `0.29` | `0.00` | `976.91` | `1.00` |
| `claude-sonnet-4-6` | `llm_base_rank_compact` | `0.22` | `0.02` | `1036.88` | `1.00` |
| `claude-sonnet-4-6` | `llm_base_rank_mini` | `0.22` | `0.06` | `1000.58` | `1.00` |
| `deepseek-v3.2` | `rule_agent` | `0.30` | `0.01` | `0.00` | `1.00` |
| `deepseek-v3.2` | `llm_base` | `0.50` | `0.01` | `917.71` | `1.00` |
| `deepseek-v3.2` | `llm_base_rank` | `0.5859` | `0.04` | `868.40` | `1.00` |
| `deepseek-v3.2` | `llm_base_rank_compact` | `0.57` | `0.0303` | `916.76` | `1.00` |
| `deepseek-v3.2` | `llm_base_rank_mini` | `0.57` | `0.03` | `886.52` | `1.00` |

## Quick Read

- `DeepSeek` is currently the strongest overall backend in malicious-product filtering while remaining stable.
- `Claude` is stable and shows a clear precision/recall trade-off between `llm_base` and `llm_base_rank`.
- `Gemini` is not stable enough under the current compatibility route because parse success is too low.
- `gpt-5.4-nano` produced promising rank-enhanced results, but the final parse-success column was not preserved in the mapped summary stage, so it should be treated as a recorded run rather than a fully archived mapped file.
