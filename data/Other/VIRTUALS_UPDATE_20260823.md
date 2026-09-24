# Virtuals Registry Update (2026-08-23)

## 1. Snapshot comparison

| Metric | 2026-05-06 | 2026-08-23 |
|---|---:|---:|
| Capture time (UTC) | 12:34:26 | 04:05:43 |
| Public agents | 42,061 | 44,051 |

The registry grew by 1,990 agents (4.73%). All 42,061 previously observed IDs remained visible; the new IDs range from 42,195 to 44,184.

## 2. Earlier data collection

The May dataset covered the public agent catalog and sampled recent details, metrics, engagements, ratings, feedback, and interactions for each agent. Per-agent limits mean it is not a complete interaction history. The main raw file is `data/bundles.jsonl`.

A review of the raw bundles found 25,377 engagement requests that ended with HTTP 429. The `error_count_this_run=0` in `data/meta.json` describes only the final resumed run, not the complete collection.

## 3. August data collection

The August snapshot used read-only requests to the public `/api/agents` endpoint, with 100 agents per page sorted by `publishedAt:desc`. All 441 pages returned HTTP 200. Raw pages are stored in `output/0_datainit/1_current_registry_20260823/raw/registry_pages/`; `compare_to_20260506/5_raw_page_manifest.csv` records request and file checks.

## 4. Agent cards and offerings

Among the 42,061 shared agents, 18 names, 29 descriptions, 41 job payloads, and 558 `lastActiveAt` values changed. No wallet or owner addresses changed.

The earlier `records.csv.offering_count` understated offerings because it missed `details.jobs`. Counts recalculated from the raw bundles are:

| Metric | May | August | Change |
|---|---:|---:|---:|
| Agents with a job or service offering | 19,899 | 19,950 | +51 |
| Job and legacy offering records | 37,508 | 37,669 | +161 |

Of the 1,990 new agents, 52 (2.61%) published a total of 109 jobs. None had a positive platform-reported successful job, unique buyer, transaction, or revenue count.

## 5. Activity among shared agents

| Platform-reported metric | Comparable agents | Agents with an increase | Total change |
|---|---:|---:|---:|
| Successful jobs | 41,241 | 19 | +2,440 |
| Unique buyers | 39,141 | 18 | +1,089 |
| Transactions | 39,090 | 19 | +6,535 |
| Gross agentic amount | 39,075 | 19 | +317,768.31 |
| Revenue | 38,216 | 17 | +133.12 |

No comparable agent showed a decrease in these metrics. Synapse Robotics Network, Degen Claw, and Capminal accounted for 82.83% of the successful-job increase and 88.89% of the unique-buyer increase. Degen Claw alone accounted for about 98.96% of the gross-agentic-amount increase. Registry growth therefore did not translate into broadly distributed activity growth.
