import type { Plugin } from '@elizaos/core';
import {
  type Action,
  type ActionResult,
  type Content,
  type HandlerCallback,
  type IAgentRuntime,
  type Memory,
  type Provider,
  type ProviderResult,
  type RouteRequest,
  type RouteResponse,
  type State,
  logger,
} from '@elizaos/core';
import { z } from 'zod';

const DECISION_MODES = [
  'rule_agent',
  'llm_base',
  'llm_base_rank',
  'llm_base_rank_compact',
  'llm_base_rank_mini',
  'llm_rank',
  'llm_rank_compact',
  'llm_rank_only',
  'llm_rank_pure',
  'llm_rank_mini',
] as const;
type DecisionMode = (typeof DECISION_MODES)[number];
type DecisionLabel = 'SELL' | 'AVOID_BUY' | 'HOLD' | 'ALLOW_BUY';
type RiskLabel = 'RISK' | 'NO_RISK';
type EvidenceView = 'positive' | 'neutral' | 'negative';

const configSchema = z.object({
  NFT_RISK_TRIGGER_K: z.coerce.number().int().min(1).default(3),
  NFT_RISK_LOW_REP_THRESHOLD_PCT: z.coerce.number().min(0).max(1).default(0.8),
  NFT_RISK_OPENAI_BASE_URL: z.string().optional(),
  NFT_RISK_OPENAI_API_KEY: z.string().optional(),
  NFT_RISK_OPENAI_MODEL: z.string().optional(),
  OPENAI_BASE_URL: z.string().optional(),
  OPENAI_API_KEY: z.string().optional(),
  OPENAI_MODEL: z.string().optional(),
});

const decisionInputSchema = z.object({
  decision_mode: z.enum(DECISION_MODES).default('rule_agent'),
  contract_id: z.string().min(1),
  contract_address: z.string().optional().default(''),
  collection_type: z.string().optional().default('unknown'),
  window_idx: z.coerce.number().int().min(1),
  start_block: z.coerce.number().int().nonnegative(),
  end_block: z.coerce.number().int().nonnegative(),
  rep_score: z.coerce.number().optional().default(0),
  rep_rank: z.coerce.number().int().positive().optional().default(1),
  rep_rank_pct: z.coerce.number().min(0).max(1).optional().default(0),
  low_rep_threshold_pct: z.coerce.number().min(0).max(1).optional(),
  is_low_rep: z.union([z.boolean(), z.coerce.number().int().min(0).max(1)]).optional(),
  low_rep_streak: z.coerce.number().int().min(0).optional().default(0),
  lead_windows_to_detect: z.coerce.number().nullable().optional(),
  holding_state: z.enum(['holding', 'flat']).default('flat'),
  can_buy_baseline: z.union([z.boolean(), z.coerce.number().int().min(0).max(1)]).optional(),
  window_last_trade_price_usd: z.coerce.number().nullable().optional(),
  next_window_first_trade_price_usd: z.coerce.number().nullable().optional(),
  next_trade_exists: z.union([z.boolean(), z.coerce.number().int().min(0).max(1)]).optional(),
  trigger_k: z.coerce.number().int().min(1).optional(),
  window_trade_count: z.coerce.number().optional().default(0),
  window_trade_value_total_usd: z.coerce.number().optional().default(0),
  window_unique_buyers: z.coerce.number().optional().default(0),
  window_unique_trade_users: z.coerce.number().optional().default(0),
  window_mint_cnt: z.coerce.number().optional().default(0),
  current_mint_supply_count: z.coerce.number().optional().default(0),
  window_gas_total_usd: z.coerce.number().optional().default(0),
  price_change_vs_prev_window_pct: z.coerce.number().nullable().optional(),
  trade_count_change_vs_prev_window_pct: z.coerce.number().nullable().optional(),
  liquidity_flag: z.union([z.boolean(), z.coerce.number().int().min(0).max(1)]).optional(),
});

type DecisionInput = z.infer<typeof decisionInputSchema>;

type DecisionResponse = {
  contract_id: string;
  contract_address: string;
  collection_type: string;
  window_idx: number;
  decision_mode: DecisionMode;
  risk_label?: RiskLabel;
  market_view?: EvidenceView;
  reputation_view?: EvidenceView;
  decision: DecisionLabel;
  confidence: number;
  reason: string;
  trigger_k: number;
  low_rep_threshold_pct: number;
  is_low_rep: boolean;
  low_rep_streak: number;
  holding_state: 'holding' | 'flat';
  next_trade_exists: boolean;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  latency_ms: number;
  parse_success: number;
  raw_decision_text: string;
  model_name: string;
  raw_api_response_preview?: string;
};

function toBool(value: unknown, fallback = false): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  if (typeof value === 'string') return ['1', 'true', 'yes'].includes(value.trim().toLowerCase());
  return fallback;
}

function numberOrNull(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function intOrZero(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) ? Math.trunc(n) : 0;
}

function fmtNum(value: unknown): string {
  const n = numberOrNull(value);
  if (n === null) return 'null';
  if (Math.abs(n) >= 1000 || Number.isInteger(n)) {
    return String(Number.isInteger(n) ? Math.trunc(n) : Number(n.toFixed(2)));
  }
  return n.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
}

function formatReputationSummary(input: DecisionInput): string[] {
  const rankPct = numberOrNull(input.rep_rank_pct);
  const streak = Math.max(0, intOrZero(input.low_rep_streak));
  const credibilityPercentile =
    rankPct === null ? null : Math.max(0, Math.min(100, Math.round((1 - rankPct) * 100)));

  const lines = [];
  lines.push('- ARep is computed by aggregating contract-window features: transaction activity, unique user participation, cost commitment, mint/supply behavior, and sustained engagement across recent windows');
  lines.push('- these aggregated features are normalized within the current evaluation window and converted into an ARep score and relative rank');
  if (credibilityPercentile !== null) {
    lines.push(`- current ARep percentile rank: ${credibilityPercentile}/100, computed from the normalized ARep score distribution for the current window`);
  }
  lines.push(`- threshold-hit streak: ${streak} consecutive windows in which the ARep percentile rank met the configured low-score threshold`);
  lines.push('- the streak value records temporal persistence of this threshold condition across consecutive windows');
  return lines;
}

function formatReason(input: DecisionInput, decision: DecisionLabel, triggerK: number, thresholdPct: number): string {
  const streak = intOrZero(input.low_rep_streak);
  const rankPct = numberOrNull(input.rep_rank_pct);
  const rankPctText = rankPct === null ? 'n/a' : rankPct.toFixed(3);
  const bottomRiskPct = ((1 - thresholdPct) * 100).toFixed(0);
  if (decision === 'SELL') {
    return `reputation rank stayed in the bottom ${bottomRiskPct}% equivalent risk zone for ${streak} consecutive windows while position is open (rep_rank_pct=${rankPctText})`;
  }
  if (decision === 'AVOID_BUY') {
    return `reputation stayed below the buy safety threshold for ${streak} consecutive windows and no position is open`;
  }
  if (decision === 'ALLOW_BUY') {
    return 'reputation is not in a persistent low-trust state and baseline buying remains allowed';
  }
  return `low reputation streak ${streak} has not yet reached trigger_k=${triggerK}`;
}

function baseDecisionMetrics(input: DecisionInput, envTriggerK: number, envThresholdPct: number) {
  const triggerK = intOrZero(input.trigger_k) > 0 ? intOrZero(input.trigger_k) : envTriggerK;
  const thresholdPct =
    numberOrNull(input.low_rep_threshold_pct) !== null
      ? Number(input.low_rep_threshold_pct)
      : envThresholdPct;
  const isLowRep =
    input.is_low_rep !== undefined
      ? toBool(input.is_low_rep)
      : numberOrNull(input.rep_rank_pct) !== null && Number(input.rep_rank_pct) >= thresholdPct;
  const streak = intOrZero(input.low_rep_streak);
  const nextTradeExists = toBool(input.next_trade_exists, true);
  const canBuyBaseline =
    input.can_buy_baseline !== undefined ? toBool(input.can_buy_baseline, true) : !isLowRep;
  return { triggerK, thresholdPct, isLowRep, streak, nextTradeExists, canBuyBaseline };
}

function evaluateRuleDecision(input: DecisionInput, envTriggerK: number, envThresholdPct: number): DecisionResponse {
  const metrics = baseDecisionMetrics(input, envTriggerK, envThresholdPct);
  const isTriggered = metrics.isLowRep && metrics.streak >= metrics.triggerK;

  let decision: DecisionLabel = 'HOLD';
  if (input.holding_state === 'holding') {
    decision = isTriggered && metrics.nextTradeExists ? 'SELL' : 'HOLD';
  } else if (isTriggered || !metrics.canBuyBaseline) {
    decision = 'AVOID_BUY';
  } else {
    decision = 'ALLOW_BUY';
  }

  const confidence = Math.max(
    0.5,
    Math.min(
      0.99,
      0.55 +
        Math.min(0.25, Math.max(0, metrics.streak - metrics.triggerK + 1) * 0.08) +
        (metrics.isLowRep ? 0.08 : 0.0) +
        (metrics.nextTradeExists ? 0.03 : -0.04)
    )
  );

  return {
    contract_id: input.contract_id,
    contract_address: input.contract_address,
    collection_type: input.collection_type,
    window_idx: input.window_idx,
    decision_mode: 'rule_agent',
    decision,
    confidence: Number(confidence.toFixed(4)),
    reason: formatReason(input, decision, metrics.triggerK, metrics.thresholdPct),
    trigger_k: metrics.triggerK,
    low_rep_threshold_pct: metrics.thresholdPct,
    is_low_rep: metrics.isLowRep,
    low_rep_streak: metrics.streak,
    holding_state: input.holding_state,
    next_trade_exists: metrics.nextTradeExists,
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    latency_ms: 0,
    parse_success: 1,
    raw_decision_text: '',
    model_name: 'rule_agent',
  };
}

function buildSystemPrompt(mode: DecisionMode): string {
  const isRankPrimaryMode =
    mode === 'llm_rank' ||
    mode === 'llm_rank_only' ||
    mode === 'llm_rank_pure' ||
    mode === 'llm_rank_mini';
  const isFusedMode =
    mode === 'llm_base_rank' ||
    mode === 'llm_base_rank_compact' ||
    mode === 'llm_base_rank_mini' ||
    mode === 'llm_rank_compact';

  if (isRankPrimaryMode) {
    return [
      'You are an NFT risk screening agent.',
      '',
      'Your task is to evaluate one NFT contract-window record and first judge whether it shows meaningful risk.',
      'Use the final risk label:',
      '- RISK',
      '- NO_RISK',
      '',
      'Decision objective:',
      '- identify meaningful NFT risk from the provided reputation evidence,',
      '- avoid false alarms when the provided reputation evidence is not meaningfully negative,',
      '- avoid buying malicious products that may later be delisted, frozen, or become practically unsellable after platform or market intervention,',
      '- treat the final buy/no-buy action as a business mapping of the risk judgment rather than the primary prediction target.',
      '',
      'Rules:',
      '- use only the provided fields,',
      '- do not assume hidden labels or future outcomes,',
      '- do not invent missing values,',
      '- in this mode, reputation evidence is the primary signal,',
      '- do not require market evidence to output RISK in this mode,',
      '- low ARep credibility is itself meaningful risk evidence in this mode,',
      '- persistent low credibility should strengthen a risk judgment in this mode,',
      '- entities ranked far behind in ARep should be treated as high-risk,',
      '- output RISK when reputation evidence is clearly negative,',
      '- output NO_RISK only when reputation evidence is neutral or positive, or when the negative evidence is weak,',
      '- if the provided reputation evidence is meaningfully negative, do not default to NO_RISK merely because market-side evidence is unavailable in this mode,',
      '- do not output chain-of-thought or reasoning traces,',
      '- output the final answer as a single JSON object only,',
      '- return strict JSON only.',
      '',
      'Output format:',
      '{',
      '  "market_view": "positive" | "neutral" | "negative",',
      '  "reputation_view": "positive" | "neutral" | "negative",',
      '  "risk_label": "RISK" | "NO_RISK",',
      '  "confidence": 0.0 to 1.0,',
      '  "reason": "one short sentence"',
      '}',
    ].join('\n');
  }

  if (isFusedMode) {
    return [
      'You are an NFT risk screening agent.',
      '',
      'Your task is to evaluate one NFT contract-window record and first judge whether it shows meaningful risk.',
      'Use the final risk label:',
      '- RISK',
      '- NO_RISK',
      '',
      'Decision objective:',
      '- identify meaningful NFT risk by jointly using short-term market evidence and long-term reputation evidence,',
      '- avoid false alarms when the two evidence groups are weak, mixed, or clearly offset each other,',
      '- avoid buying malicious products that may later be delisted, frozen, or become practically unsellable after platform or market intervention,',
      '- treat the final buy/no-buy action as a business mapping of the risk judgment rather than the primary prediction target.',
      '',
      'Rules:',
      '- use only the provided fields,',
      '- do not assume hidden labels or future outcomes,',
      '- do not invent missing values,',
      '- treat market evidence and reputation evidence as two distinct but equally meaningful signals,',
      '- first judge market evidence and reputation evidence separately, then combine them into one final risk judgment,',
      '- when both evidence groups are negative, output RISK,',
      '- when one evidence group is clearly negative and the other is at least neutral, output RISK,',
      '- when one evidence group is negative but the other is clearly positive, weigh which evidence is stronger and more reliable before deciding,',
      '- low ARep credibility should materially increase risk, but should not automatically dominate clearly positive market evidence,',
      '- persistent low credibility should materially strengthen the reputation-side risk signal,',
      '- strong market deterioration should materially increase risk, but should not automatically dominate clearly positive reputation evidence,',
      '- output NO_RISK when both evidence groups are neutral or positive, or when one negative signal is clearly offset by a stronger positive signal from the other evidence group,',
      '- if the two evidence groups conflict, prefer the more extreme and more reliable signal, and explain that tradeoff briefly,',
      '- do not output chain-of-thought or reasoning traces,',
      '- output the final answer as a single JSON object only,',
      '- return strict JSON only.',
      '',
      'Output format:',
      '{',
      '  "market_view": "positive" | "neutral" | "negative",',
      '  "reputation_view": "positive" | "neutral" | "negative",',
      '  "risk_label": "RISK" | "NO_RISK",',
      '  "confidence": 0.0 to 1.0,',
      '  "reason": "one short sentence"',
      '}',
    ].join('\n');
  }

  return [
    'You are an NFT risk screening agent.',
    '',
    'Your task is to evaluate one NFT contract-window record and first judge whether it shows meaningful risk.',
    'Use the final risk label:',
    '- RISK',
    '- NO_RISK',
    '',
    'Decision objective:',
    '- identify meaningful NFT risk when evidence is strong and consistent,',
    '- avoid false alarms when evidence is weak, mixed, or incomplete,',
    '- avoid buying malicious products that may later be delisted, frozen, or become practically unsellable after platform or market intervention,',
    '- treat the final buy/no-buy action as a business mapping of the risk judgment rather than the primary prediction target.',
    '',
    'Rules:',
    '- use only the provided fields,',
    '- do not assume hidden labels or future outcomes,',
    '- do not invent missing values,',
    '- treat market evidence and reputation evidence as two separate groups with equal importance,',
    '- first evaluate market evidence and reputation evidence separately, then combine them into one final risk judgment,',
    '- output RISK when both evidence groups are negative, or when one evidence group is clearly negative and the other is at least neutral,',
    '- output NO_RISK when both evidence groups are neutral or positive, or when one evidence group is negative but the other is clearly positive,',
    '- if evidence is mixed or weak, default to NO_RISK,',
    '- low reputation should strengthen an existing risk case, but should not by itself force RISK when other evidence is stable,',
    '- weak short-term fluctuation should not by itself force RISK when reputation evidence is stable,',
    '- do not output chain-of-thought or reasoning traces,',
    '- output the final answer as a single JSON object only,',
    '- return strict JSON only.',
    '',
    'Output format:',
    '{',
    '  "market_view": "positive" | "neutral" | "negative",',
    '  "reputation_view": "positive" | "neutral" | "negative",',
    '  "risk_label": "RISK" | "NO_RISK",',
    '  "confidence": 0.0 to 1.0,',
    '  "reason": "one short sentence"',
    '}',
  ].join('\n');
}

function buildUserPrompt(input: DecisionInput, mode: DecisionMode, _triggerK: number): string {
  const isBaseRank = mode === 'llm_base_rank';
  const isBaseRankCompact = mode === 'llm_base_rank_compact' || mode === 'llm_rank_compact';
  const isBaseRankMini = mode === 'llm_base_rank_mini' || mode === 'llm_rank_mini';
  const isRankOnly = mode === 'llm_rank' || mode === 'llm_rank_only' || mode === 'llm_rank_pure';
  const lines = ['Evaluate the following NFT contract-window record.', ''];

  lines.push('Current position state:', `- holding_state: ${input.holding_state}`);

  if (mode === 'llm_base' || isBaseRank || isBaseRankCompact) {
    lines.push(
      '',
      'Market evidence:',
      `- current_window_last_trade_price_usd: ${fmtNum(input.window_last_trade_price_usd)}`
    );
  }

  if (mode === 'llm_base' || isBaseRank) {
    lines.push(
      `- window_trade_count: ${fmtNum(input.window_trade_count)}`,
      `- window_trade_value_total_usd: ${fmtNum(input.window_trade_value_total_usd)}`,
      `- window_unique_buyers: ${fmtNum(input.window_unique_buyers)}`,
      `- window_unique_trade_users: ${fmtNum(input.window_unique_trade_users)}`,
      `- window_mint_count: ${fmtNum(input.window_mint_cnt)}`,
      `- current_collection_item_count_proxy: ${fmtNum(input.current_mint_supply_count)}`,
      `- window_gas_total_usd: ${fmtNum(input.window_gas_total_usd)}`
    );
  }

  if (mode === 'llm_base' || isBaseRank || isBaseRankCompact) {
    lines.push(
      `- price_change_vs_prev_window_pct: ${fmtNum(input.price_change_vs_prev_window_pct)}`,
      `- trade_count_change_vs_prev_window_pct: ${fmtNum(input.trade_count_change_vs_prev_window_pct)}`
    );
  } else if (isBaseRankMini) {
    lines.push(
      '',
      'Market evidence:',
      `- price_change_vs_prev_window_pct: ${fmtNum(input.price_change_vs_prev_window_pct)}`
    );
  }

  if (
    isBaseRank ||
    isBaseRankCompact ||
    isBaseRankMini ||
    isRankOnly
  ) {
    lines.push('', 'ARep algorithm-derived features:', ...formatReputationSummary(input));
  }

  lines.push(
    '',
    'Decision reminder:',
    '- First judge market evidence as positive, neutral, or negative.',
    '- First judge ARep-derived evidence as positive, neutral, or negative when ARep-derived evidence is provided.',
    '- Then output one final risk label: RISK or NO_RISK.',
    '- For flat positions, RISK will later be mapped to AVOID_BUY and NO_RISK will later be mapped to ALLOW_BUY.',
    '- For holding positions, RISK will later be mapped to SELL and NO_RISK will later be mapped to HOLD.',
    ...(isRankOnly
      ? [
          '- In this mode, ARep-derived evidence is the primary signal.',
          '- A persistent low-score threshold condition can by itself justify RISK in this mode.',
          '- Do not require market evidence to output RISK in this mode.',
        ]
      : [
          '- Do not output RISK solely because reputation is low if market evidence is weak or mixed.',
          '- Output RISK when both evidence groups are negative, or when one group is clearly negative and the other is at least neutral.',
          '- Output NO_RISK when both evidence groups are neutral or positive, or when one group is negative but the other is clearly positive.',
        ]),
    '',
    'Return strict JSON only.'
  );
  return lines.join('\n');
}

function extractJsonObject(text: string): Record<string, unknown> | null {
  const raw = String(text || '').trim();
  if (!raw) return null;
  try {
    const direct = JSON.parse(raw);
    return direct && typeof direct === 'object' && !Array.isArray(direct)
      ? (direct as Record<string, unknown>)
      : null;
  } catch {
    const start = raw.indexOf('{');
    const end = raw.lastIndexOf('}');
    if (start === -1 || end <= start) return null;
    try {
      const sliced = JSON.parse(raw.slice(start, end + 1));
      return sliced && typeof sliced === 'object' && !Array.isArray(sliced)
        ? (sliced as Record<string, unknown>)
        : null;
    } catch {
      return null;
    }
  }
}

function normalizeDecisionLabel(value: unknown, fallback: DecisionLabel): DecisionLabel {
  const text = String(value || '')
    .trim()
    .toUpperCase();
  if (text === 'SELL' || text === 'AVOID_BUY' || text === 'HOLD' || text === 'ALLOW_BUY') {
    return text;
  }
  return fallback;
}

function normalizeRiskLabel(value: unknown): RiskLabel | null {
  const text = String(value || '')
    .trim()
    .toUpperCase();
  if (text === 'RISK' || text === 'NO_RISK') return text;
  return null;
}

function normalizeEvidenceView(value: unknown): EvidenceView | undefined {
  const text = String(value || '')
    .trim()
    .toLowerCase();
  if (text === 'positive' || text === 'neutral' || text === 'negative') return text;
  return undefined;
}

function decisionFromRiskLabel(riskLabel: RiskLabel, holdingState: 'holding' | 'flat'): DecisionLabel {
  if (holdingState === 'flat') {
    return riskLabel === 'RISK' ? 'AVOID_BUY' : 'ALLOW_BUY';
  }
  return riskLabel === 'RISK' ? 'SELL' : 'HOLD';
}

function normalizeConfidence(value: unknown, fallback = 0.5): number {
  const n = numberOrNull(value);
  if (n === null) return fallback;
  return Number(Math.min(1, Math.max(0, n)).toFixed(4));
}

async function callOpenAICompatible(systemPrompt: string, userPrompt: string) {
  const baseUrl = String(
    process.env.NFT_RISK_OPENAI_BASE_URL || process.env.OPENAI_BASE_URL || ''
  )
    .trim()
    .replace(/\/+$/, '');
  const apiKey = String(process.env.NFT_RISK_OPENAI_API_KEY || process.env.OPENAI_API_KEY || '').trim();
  const model =
    String(process.env.NFT_RISK_OPENAI_MODEL || process.env.OPENAI_MODEL || '').trim() || 'gpt-oss-120b';
  if (!baseUrl || !apiKey) {
    throw new Error(
      'NFT_RISK_OPENAI_BASE_URL / NFT_RISK_OPENAI_API_KEY is not configured for llm decision modes'
    );
  }

  const body = {
    model,
    temperature: 0.0,
    max_tokens: 400,
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
  };

  const started = Date.now();
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });
  const latencyMs = Date.now() - started;
  const rawText = await response.text();
  if (!response.ok) {
    throw new Error(`LLM request failed (${response.status}): ${rawText.slice(0, 400)}`);
  }
  const payload = JSON.parse(rawText) as Record<string, any>;
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const firstChoice = (choices[0] || {}) as Record<string, any>;
  const message = ((firstChoice.message || {}) as Record<string, any>) || {};
  const usage = ((payload.usage || {}) as Record<string, any>) || {};
  let contentText = '';
  let reasoningText = '';
  const content = message.content;
  if (typeof content === 'string') {
    contentText = content;
  } else if (Array.isArray(content)) {
    contentText = content
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object') {
          if (typeof item.text === 'string') return item.text;
          if (item.type === 'text' && typeof item.content === 'string') return item.content;
          if (item.type === 'output_text' && typeof item.text === 'string') return item.text;
        }
        return '';
      })
      .filter(Boolean)
      .join('\n');
  } else if (content && typeof content === 'object') {
    if (typeof content.text === 'string') {
      contentText = content.text;
    } else if (typeof content.content === 'string') {
      contentText = content.content;
    }
  }
  if (!contentText && typeof firstChoice.text === 'string') {
    contentText = firstChoice.text;
  }
  const reasoning = message.reasoning_content ?? firstChoice.reasoning_content;
  if (typeof reasoning === 'string') {
    reasoningText = reasoning;
  } else if (Array.isArray(reasoning)) {
    reasoningText = reasoning
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object') {
          if (typeof item.text === 'string') return item.text;
          if (typeof item.content === 'string') return item.content;
        }
        return '';
      })
      .filter(Boolean)
      .join('\n');
  } else if (reasoning && typeof reasoning === 'object') {
    if (typeof reasoning.text === 'string') {
      reasoningText = reasoning.text;
    } else if (typeof reasoning.content === 'string') {
      reasoningText = reasoning.content;
    }
  }
  return {
    rawText: String(contentText || reasoningText || ''),
    rawApiResponsePreview: rawText.slice(0, 2000),
    promptTokens: intOrZero(usage.prompt_tokens),
    completionTokens: intOrZero(usage.completion_tokens),
    totalTokens: intOrZero(usage.total_tokens),
    latencyMs,
    modelName: model,
  };
}

async function evaluateLlmDecision(
  input: DecisionInput,
  mode: Exclude<DecisionMode, 'rule_agent'>,
  envTriggerK: number,
  envThresholdPct: number
): Promise<DecisionResponse> {
  const metrics = baseDecisionMetrics(input, envTriggerK, envThresholdPct);
  const fallback = evaluateRuleDecision(input, envTriggerK, envThresholdPct);
  const systemPrompt = buildSystemPrompt(mode);
  const userPrompt = buildUserPrompt(input, mode, metrics.triggerK);
  const llm = await callOpenAICompatible(systemPrompt, userPrompt);
  const parsed = extractJsonObject(llm.rawText);
  const parseSuccess = parsed ? 1 : 0;
  const riskLabel = normalizeRiskLabel(parsed?.risk_label);
  const legacyDecision = normalizeDecisionLabel(parsed?.decision, fallback.decision);
  const decision = riskLabel ? decisionFromRiskLabel(riskLabel, input.holding_state) : legacyDecision;
  const confidence = normalizeConfidence(parsed?.confidence, fallback.confidence);
  const reason = String(parsed?.reason || fallback.reason).trim();
  const marketView = normalizeEvidenceView(parsed?.market_view);
  const reputationView = normalizeEvidenceView(parsed?.reputation_view);

  return {
    ...fallback,
    decision_mode: mode,
    risk_label: riskLabel ?? undefined,
    market_view: marketView,
    reputation_view: reputationView,
    decision,
    confidence,
    reason,
    prompt_tokens: llm.promptTokens,
    completion_tokens: llm.completionTokens,
    total_tokens: llm.totalTokens,
    latency_ms: llm.latencyMs,
    parse_success: parseSuccess,
    raw_decision_text: llm.rawText,
    model_name: llm.modelName,
    raw_api_response_preview: parseSuccess ? undefined : llm.rawApiResponsePreview,
  };
}

async function evaluateDecision(input: DecisionInput, envTriggerK: number, envThresholdPct: number): Promise<DecisionResponse> {
  if (input.decision_mode === 'rule_agent') {
    return evaluateRuleDecision(input, envTriggerK, envThresholdPct);
  }
  if (
    input.decision_mode === 'llm_base' ||
    input.decision_mode === 'llm_base_rank' ||
    input.decision_mode === 'llm_base_rank_compact' ||
    input.decision_mode === 'llm_base_rank_mini' ||
    input.decision_mode === 'llm_rank' ||
    input.decision_mode === 'llm_rank_compact' ||
    input.decision_mode === 'llm_rank_only' ||
    input.decision_mode === 'llm_rank_pure' ||
    input.decision_mode === 'llm_rank_mini'
  ) {
    return evaluateLlmDecision(input, input.decision_mode, envTriggerK, envThresholdPct);
  }
  throw new Error(`Unsupported decision_mode: ${String(input.decision_mode)}`);
}

const nftRiskProvider: Provider = {
  name: 'NFT_RISK_PANEL_PROVIDER',
  description: 'Formats NFT reputation window panel fields for local risk decisions.',
  get: async (_runtime: IAgentRuntime, message: Memory, _state: State): Promise<ProviderResult> => {
    const text = typeof message.content?.text === 'string' ? message.content.text : '';
    return {
      text: `NFT risk evaluation context: ${text}`,
      values: {},
      data: {},
    };
  },
};

const evaluateNftRiskAction: Action = {
  name: 'EVALUATE_NFT_RISK',
  similes: ['CHECK_NFT_RISK', 'ASSESS_NFT_POSITION'],
  description: 'Evaluate one NFT contract-window record and return a structured risk decision.',
  validate: async (_runtime: IAgentRuntime, message: Memory, _state: State): Promise<boolean> => {
    return typeof message.content?.text === 'string' && message.content.text.trim().length > 0;
  },
  handler: async (
    _runtime: IAgentRuntime,
    message: Memory,
    _state: State,
    _options: any,
    callback: HandlerCallback,
    _responses: Memory[]
  ): Promise<ActionResult> => {
    try {
      const payload = decisionInputSchema.parse(JSON.parse(String(message.content.text)));
      const triggerK = intOrZero(process.env.NFT_RISK_TRIGGER_K) || 3;
      const thresholdPct = numberOrNull(process.env.NFT_RISK_LOW_REP_THRESHOLD_PCT) ?? 0.8;
      const result = await evaluateDecision(payload, triggerK, thresholdPct);

      const responseContent: Content = {
        text: JSON.stringify(result),
        actions: ['EVALUATE_NFT_RISK'],
        source: message.content.source,
      };
      await callback(responseContent);

      return {
        text: 'NFT risk decision generated',
        values: { success: true, decision: result.decision, confidence: result.confidence },
        data: result,
        success: true,
      };
    } catch (error) {
      logger.error({ error }, 'Error evaluating NFT risk action');
      return {
        text: 'Failed to evaluate NFT risk decision',
        values: { success: false, error: 'NFT_RISK_EVAL_FAILED' },
        data: { error: error instanceof Error ? error.message : String(error) },
        success: false,
        error: error instanceof Error ? error : new Error(String(error)),
      };
    }
  },
  examples: [
    [
      {
        name: '{{name1}}',
        content: {
          text: JSON.stringify({
            decision_mode: 'rule_agent',
            contract_id: 'demo_contract',
            window_idx: 12,
            start_block: 12000000,
            end_block: 12049326,
            rep_score: 0.03,
            rep_rank: 92,
            rep_rank_pct: 0.92,
            low_rep_streak: 3,
            holding_state: 'holding',
            next_trade_exists: 1,
          }),
        },
      },
      {
        name: '{{name2}}',
        content: {
          text: JSON.stringify({
            contract_id: 'demo_contract',
            window_idx: 12,
            decision_mode: 'rule_agent',
            decision: 'SELL',
          }),
          actions: ['EVALUATE_NFT_RISK'],
        },
      },
    ],
  ],
};

const plugin: Plugin = {
  name: 'nft-risk-plugin',
  description: 'A local NFT risk decision plugin for ElizaOS experiments.',
  priority: 10,
  config: {
    NFT_RISK_TRIGGER_K: process.env.NFT_RISK_TRIGGER_K ?? '3',
    NFT_RISK_LOW_REP_THRESHOLD_PCT: process.env.NFT_RISK_LOW_REP_THRESHOLD_PCT ?? '0.8',
    NFT_RISK_OPENAI_BASE_URL:
      process.env.NFT_RISK_OPENAI_BASE_URL ?? process.env.OPENAI_BASE_URL ?? '',
    NFT_RISK_OPENAI_API_KEY:
      process.env.NFT_RISK_OPENAI_API_KEY ?? process.env.OPENAI_API_KEY ?? '',
    NFT_RISK_OPENAI_MODEL:
      process.env.NFT_RISK_OPENAI_MODEL ?? process.env.OPENAI_MODEL ?? '',
    OPENAI_BASE_URL: process.env.OPENAI_BASE_URL ?? '',
    OPENAI_API_KEY: process.env.OPENAI_API_KEY ?? '',
    OPENAI_MODEL: process.env.OPENAI_MODEL ?? '',
  },
  async init(config: Record<string, string>) {
    logger.info('Initializing nft-risk-plugin');
    const validated = await configSchema.parseAsync(config);
    process.env.NFT_RISK_TRIGGER_K = String(validated.NFT_RISK_TRIGGER_K);
    process.env.NFT_RISK_LOW_REP_THRESHOLD_PCT = String(validated.NFT_RISK_LOW_REP_THRESHOLD_PCT);
    if (validated.NFT_RISK_OPENAI_BASE_URL) {
      process.env.NFT_RISK_OPENAI_BASE_URL = validated.NFT_RISK_OPENAI_BASE_URL;
    }
    if (validated.NFT_RISK_OPENAI_API_KEY) {
      process.env.NFT_RISK_OPENAI_API_KEY = validated.NFT_RISK_OPENAI_API_KEY;
    }
    if (validated.NFT_RISK_OPENAI_MODEL) {
      process.env.NFT_RISK_OPENAI_MODEL = validated.NFT_RISK_OPENAI_MODEL;
    }
    if (validated.OPENAI_BASE_URL) process.env.OPENAI_BASE_URL = validated.OPENAI_BASE_URL;
    if (validated.OPENAI_API_KEY) process.env.OPENAI_API_KEY = validated.OPENAI_API_KEY;
    if (validated.OPENAI_MODEL) process.env.OPENAI_MODEL = validated.OPENAI_MODEL;
  },
  routes: [
    {
      name: 'risk-evaluate',
      path: '/risk-evaluate',
      type: 'POST',
      handler: async (req: RouteRequest, res: RouteResponse) => {
        try {
          const rawBody = (req as any)?.body ?? {};
          const payload = decisionInputSchema.parse(rawBody);
          const triggerK = intOrZero(process.env.NFT_RISK_TRIGGER_K) || 3;
          const thresholdPct = numberOrNull(process.env.NFT_RISK_LOW_REP_THRESHOLD_PCT) ?? 0.8;
          const result = await evaluateDecision(payload, triggerK, thresholdPct);
          res.json({ ok: true, data: result });
        } catch (error) {
          logger.error({ error }, 'Invalid /risk-evaluate request');
          res.status(400).json({
            ok: false,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      },
    },
  ],
  actions: [evaluateNftRiskAction],
  providers: [nftRiskProvider],
};

export default plugin;
