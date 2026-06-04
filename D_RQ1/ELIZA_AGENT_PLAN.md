# Eliza Agent Integration Plan For D_RQ1

## Goal

Use a real Eliza-style agent runtime for risk decisions, while keeping pricing, execution, and return evaluation in local reproducible scripts.

The agent should:

- observe weekly NFT reputation signals from our pipeline
- detect consecutive low-reputation windows
- output one of: `SELL`, `AVOID_BUY`, `HOLD`, `ALLOW_BUY`

The backtester should:

- use historical NFT trade prices as execution prices
- measure avoided loss and reduced drawdown
- evaluate how many consecutive early-warning windows improve returns


## Why Not Let Eliza Do Everything

Eliza is an agent runtime, not a market backtesting engine.

Recommended split:

- Python scripts in `D_RQ1`: data panel building, price extraction, simulation, return evaluation
- Eliza runtime: decision layer only


## Proposed Local Structure

### In This Repo

- `D_RQ1/7_build_agent_window_panel.py`
  - build weekly contract-window panel
  - output reputation, price, streak-ready features

- `D_RQ1/8_eval_agent_returns.py`
  - run local backtest
  - execute SELL / AVOID_BUY decisions
  - compute avoided loss metrics

- `D_RQ1/agent_io/`
  - contract-window JSONL or CSV files
  - messages sent to the agent
  - structured agent outputs

### Separate Eliza Project

- `agent_eliza/character.ts`
- `agent_eliza/plugins/nft-risk-plugin.ts`

Reason:

- Eliza requires Node/Bun tooling
- our current repo is Python-heavy
- separation keeps experiments reproducible


## Agent Inputs Per Window

Each contract-window record should include:

- `contract_id`
- `contract_address`
- `window_idx`
- `window_start_block`
- `window_end_block`
- `rep_score`
- `rep_rank_pct`
- `rep_is_low`
- `low_rep_streak`
- `holding_state`
- `window_last_trade_price_usd`
- `next_exec_price_usd`
- `is_tradeable_next_window`
- `is_rugpull_target`


## Agent Output Schema

The agent should return strict JSON:

```json
{
  "contract_id": "CoolCats",
  "window_idx": 12,
  "decision": "SELL",
  "confidence": 0.91,
  "reason": "reputation stayed below threshold for 3 consecutive windows"
}
```

Allowed decisions:

- `SELL`
- `AVOID_BUY`
- `HOLD`
- `ALLOW_BUY`


## Execution Rules

- If `SELL` while holding:
  - execute at the next window's first observed trade price
- If `AVOID_BUY` while flat:
  - skip the baseline buy
- If no next-window trade exists:
  - mark the action as unfilled
  - keep a separate liquidity flag


## Return Metrics

- `holding_loss_avoided = strategy_exit_price - baseline_exit_price`
- `buy_loss_avoided = baseline_entry_price - terminal_price`
- `lead_windows`
- `consecutive_lead_windows`
- `max_drawdown_reduction`


## Environment Status On This Machine

Current findings on this Windows machine:

- `node` exists but is `v20.17.0`
- official Eliza requirement is newer
- `bun` is not installed
- `npm` currently has a path/permission issue
- `wsl --status` returned an access/service error

This means:

- we should not assume Eliza can be installed cleanly in the current shell
- the most likely working path is `WSL2 + Ubuntu + Node 23.3+ + Bun`


## Recommended Next Step

1. Fix or bypass the local Node toolchain problem.
2. Create a minimal Eliza project outside the Python pipeline.
3. First validate one contract-window message roundtrip.
4. Then connect batch simulation to D_RQ1 outputs.
