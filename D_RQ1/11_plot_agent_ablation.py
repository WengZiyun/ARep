#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Config
# ============================================================
INPUT_DECISIONS_CSV = Path("output/D/8_agent_decisions/agent_decisions.csv")
OUTPUT_DIR = Path("output/D/11_agent_ablation_plots")

MODE_ORDER = [
    "llm_base",
    "llm_rank",
    "llm_rank_compact",
    "llm_rank_only",
]

MODE_LABEL = {
    "llm_base": "LLM-Base",
    "llm_rank": "LLM+Rank",
    "llm_rank_compact": "LLM+Rank Compact",
    "llm_rank_only": "LLM+Rank Only",
}

MODE_COLOR = {
    "llm_base": "#4C6EF5",
    "llm_rank": "#E8590C",
    "llm_rank_compact": "#2B8A3E",
    "llm_rank_only": "#C2255C",
}


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def load_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_exists(INPUT_DECISIONS_CSV)
    decisions = pd.read_csv(INPUT_DECISIONS_CSV, low_memory=False)
    decisions = decisions[decisions["decision_mode"].isin(MODE_ORDER)].copy()
    if decisions.empty:
        raise ValueError("No ablation decision rows found in agent_decisions.csv")

    token_summary = (
        decisions.groupby("decision_mode", as_index=False)
        .agg(
            rows=("decision_mode", "size"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_completion_tokens=("completion_tokens", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
        )
    )

    holding = decisions[decisions["holding_state"] == "holding"].copy()
    holding_summary = (
        holding.groupby("decision_mode", as_index=False)
        .agg(
            rows=("decision_mode", "size"),
            sell_count=("decision", lambda s: int((s == "SELL").sum())),
            hold_count=("decision", lambda s: int((s == "HOLD").sum())),
        )
    )
    holding_summary["sell_rate"] = holding_summary["sell_count"] / holding_summary["rows"]

    summary = token_summary.merge(holding_summary, on="decision_mode", how="left")
    summary["mode_label"] = summary["decision_mode"].map(MODE_LABEL)
    summary["color"] = summary["decision_mode"].map(MODE_COLOR)
    summary = summary.set_index("decision_mode").loc[MODE_ORDER].reset_index()
    return decisions, summary


def plot_avg_tokens(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(
        summary["mode_label"],
        summary["avg_total_tokens"],
        color=summary["color"],
        edgecolor="black",
        linewidth=1.0,
    )
    for x, y in zip(summary["mode_label"], summary["avg_total_tokens"]):
        ax.text(x, y + 5, f"{y:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Average Total Tokens per Decision")
    ax.set_title("Ablation: Token Cost by Decision Mode")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUTPUT_DIR / "ablation_avg_total_tokens.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sell_rate(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(
        summary["mode_label"],
        summary["sell_rate"],
        color=summary["color"],
        edgecolor="black",
        linewidth=1.0,
    )
    for x, y in zip(summary["mode_label"], summary["sell_rate"]):
        ax.text(x, y + 0.015, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("SELL Rate in Holding Scenario")
    ax.set_title("Ablation: Risk Exit Tendency by Decision Mode")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUTPUT_DIR / "ablation_holding_sell_rate.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_token_vs_sell_rate(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for row in summary.itertuples(index=False):
        ax.scatter(
            row.avg_total_tokens,
            row.sell_rate,
            s=180,
            color=row.color,
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )
        ax.text(
            row.avg_total_tokens + 3,
            row.sell_rate + 0.01,
            row.mode_label,
            fontsize=10,
        )
    ax.set_xlabel("Average Total Tokens per Decision")
    ax.set_ylabel("SELL Rate in Holding Scenario")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.set_title("Ablation: Decision Sensitivity vs Token Cost")
    fig.tight_layout()
    out = OUTPUT_DIR / "ablation_token_vs_sell_rate.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, summary = load_summary()
    summary.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False, encoding="utf-8-sig")

    p1 = plot_avg_tokens(summary)
    p2 = plot_sell_rate(summary)
    p3 = plot_token_vs_sell_rate(summary)

    print(f"[Saved] {(OUTPUT_DIR / 'ablation_summary.csv').resolve()}")
    print(f"[Saved] {p1.resolve()}")
    print(f"[Saved] {p2.resolve()}")
    print(f"[Saved] {p3.resolve()}")


if __name__ == "__main__":
    main()
