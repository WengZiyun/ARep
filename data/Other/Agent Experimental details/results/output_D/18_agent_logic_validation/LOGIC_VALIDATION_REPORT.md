# Agent Logic-Violation Audit

This audit checks deterministic consistency among `market_view`, `reputation_view`, `risk_label`, and `decision`.
It is an internal-logic audit, not a factual hallucination detector.

## Rule

- If at least one evidence view is `negative` and neither view is `positive`, expect `RISK` and `AVOID_BUY`.
- Otherwise, expect `NO_RISK` and `ALLOW_BUY`.
- Independently, `RISK` must map to `AVOID_BUY`, and `NO_RISK` must map to `ALLOW_BUY`.

## Table-ready Results

| Model | Mode | N | Logic violations | Logic error [95% Wilson CI] |
|---|---|---:|---:|---:|
| gpt-5.4-nano | llm_base | 200 | 17 | 0.085 [0.054, 0.132] |
| gpt-5.4-nano | llm_base_rank | 200 | 16 | 0.080 [0.050, 0.126] |
| gpt-5.4-nano | llm_rank | 200 | 0 | 0.000 [0.000, 0.019] |
| grok-4-1-fast | llm_base | 200 | 0 | 0.000 [0.000, 0.019] |
| grok-4-1-fast | llm_base_rank | 200 | 3 | 0.015 [0.005, 0.043] |
| grok-4-1-fast | llm_rank | 200 | 0 | 0.000 [0.000, 0.019] |
| claude-haiku-4-5-20251001 | llm_base | 200 | 0 | 0.000 [0.000, 0.019] |
| claude-haiku-4-5-20251001 | llm_base_rank | 200 | 0 | 0.000 [0.000, 0.019] |
| claude-haiku-4-5-20251001 | llm_rank | 200 | 0 | 0.000 [0.000, 0.019] |
| gemini-2.5-flash-nothinking | llm_base | 200 | 55 | 0.275 [0.218, 0.341] |
| gemini-2.5-flash-nothinking | llm_base_rank | 200 | 0 | 0.000 [0.000, 0.019] |
| gemini-2.5-flash-nothinking | llm_rank | 200 | 0 | 0.000 [0.000, 0.019] |

## Interpretation

A logic violation means that the four saved output fields contradict the explicit prompt rule. It may indicate instruction-following failure or internal inconsistency, but it does not prove that the underlying market or ARep statements are factually hallucinated.
