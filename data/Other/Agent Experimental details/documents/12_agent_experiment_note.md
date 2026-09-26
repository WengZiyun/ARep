# Agent Experiment Note

## Goal
Compare how different agent decision modes use market context and reputation/rank signals to make NFT risk decisions, and evaluate their token cost and downstream return impact.

## Shared Setup
- Same contract-window input panel: `output/D/7_agent_panel/contract_window_panel.csv`
- Same action space:
  - `SELL`
  - `AVOID_BUY`
  - `HOLD`
  - `ALLOW_BUY`
- Same Eliza runtime and same evaluation pipeline
- Same historical execution/replay logic in return evaluation

## Decision Modes

### 1. `rule_agent`
Pure rule baseline.

Uses:
- `rep_rank_pct`
- `is_low_rep`
- `low_rep_streak`
- `holding_state`
- `next_trade_exists`
- `trigger_k`

Purpose:
- zero-token baseline
- highly interpretable baseline

### 2. `llm_base`
LLM with market context only, no rank/reputation.

Uses:
- `holding_state`
- `window_last_trade_price_usd`
- `next_trade_exists`
- `liquidity_flag`
- `window_trade_count`
- `window_trade_value_total_usd`
- `window_unique_buyers`
- `window_unique_trade_users`
- `window_mint_cnt`
- `window_gas_total_usd`
- `price_change_vs_prev_window_pct`
- `trade_count_change_vs_prev_window_pct`

Purpose:
- pure LLM baseline without our method

### 3. `llm_rank`
Full LLM + rank context.

Uses:
- all `llm_base` fields
- `rep_score`
- `rep_rank`
- `rep_rank_pct`
- `is_low_rep`
- `low_rep_streak`

Purpose:
- main full-information reputation-enhanced LLM

### 4. `llm_rank_compact`
Compressed LLM + rank context.

Uses:
- `holding_state`
- `window_last_trade_price_usd`
- `next_trade_exists`
- `liquidity_flag`
- `price_change_vs_prev_window_pct`
- `trade_count_change_vs_prev_window_pct`
- `rep_score`
- `rep_rank_pct`
- `is_low_rep`
- `low_rep_streak`

Purpose:
- test whether part of the verbose market context can be removed

### 5. `llm_rank_only`
Minimal execution context + rank.

Uses:
- `holding_state`
- `next_trade_exists`
- `rep_score`
- `rep_rank_pct`
- `is_low_rep`
- `low_rep_streak`

Purpose:
- test whether rank can support decisions with very little extra context

### 6. `llm_rank_pure`
Pure rank-only setting.

Uses:
- `holding_state`
- `rep_score`
- `rep_rank_pct`
- `is_low_rep`
- `low_rep_streak`

Purpose:
- cleanest rank-only ablation

### 7. `llm_rank_mini`
Rank + the two most critical market signals.

Uses:
- `holding_state`
- `next_trade_exists`
- `price_change_vs_prev_window_pct`
- `rep_score`
- `rep_rank_pct`
- `is_low_rep`
- `low_rep_streak`

Purpose:
- low-token practical setting
- candidate for the best cost-performance tradeoff

## Comparison Logic

### Main comparison
- `rule_agent` vs `llm_base`
- `llm_base` vs `llm_rank`
- `rule_agent` vs `llm_rank`

Questions:
- Does LLM outperform pure rules?
- Does rank improve LLM decisions?
- Does full LLM+rank outperform the rule baseline?

### Ablation / compression comparison
- `llm_rank` vs `llm_rank_compact`
- `llm_rank` vs `llm_rank_only`
- `llm_rank` vs `llm_rank_pure`
- `llm_rank` vs `llm_rank_mini`

Questions:
- Can rank replace part of the market context?
- How much context can be removed before decision quality drops?
- Can token cost be reduced while preserving risk sensitivity?

## Metrics

### Decision behavior
- `SELL` rate in holding scenario
- `AVOID_BUY` rate in flat scenario
- `HOLD` / `SELL` distribution

### Cost
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `avg_total_tokens`
- `latency_ms`

### Return impact
- `holding_loss_avoided_usd`
- `buy_loss_avoided_usd`
- later extension:
  - `net_pnl_improvement_usd`
  - `missed_upside_usd`
  - `drawdown_reduction_usd`

## Interpretation Focus
- If `llm_rank` > `llm_base`, rank improves LLM decision quality.
- If `llm_rank_compact` or `llm_rank_mini` stays close to `llm_rank`, rank can compress context.
- If `llm_rank_only` or `llm_rank_pure` remains strong, rank itself is information-dense.
- If compressed modes reduce token usage substantially with similar decision behavior, the method is cost-effective.
