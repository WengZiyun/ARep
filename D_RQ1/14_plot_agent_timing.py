#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_SUMMARY_CSV = Path("output/D/13_agent_timing/timing_summary.csv")
INPUT_CONTRACT_CSV = Path("output/D/13_agent_timing/timing_contract_level.csv")
OUTPUT_DIR = Path("output/D/14_agent_timing_plots")

MODE_ORDER = [
    "rule_agent",
    "llm_base",
    "llm_rank",
    "llm_rank_compact",
    "llm_rank_mini",
    "llm_rank_only",
    "llm_rank_pure",
]

MODE_LABEL = {
    "rule_agent": "Rule",
    "llm_base": "LLM-Base",
    "llm_rank": "LLM+Rank",
    "llm_rank_compact": "Rank Compact",
    "llm_rank_mini": "Rank Mini",
    "llm_rank_only": "Rank Only",
    "llm_rank_pure": "Rank Pure",
}

MODE_COLOR = {
    "rule_agent": "#6C757D",
    "llm_base": "#4C6EF5",
    "llm_rank": "#E8590C",
    "llm_rank_compact": "#2B8A3E",
    "llm_rank_mini": "#0B7285",
    "llm_rank_only": "#C2255C",
    "llm_rank_pure": "#7B2CBF",
}


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def prepare(angle: str) -> pd.DataFrame:
    ensure_exists(INPUT_SUMMARY_CSV)
    df = pd.read_csv(INPUT_SUMMARY_CSV, low_memory=False)
    df = df[df["angle"] == angle].copy()
    df["mode_label"] = df["decision_mode"].map(MODE_LABEL)
    df["color"] = df["decision_mode"].map(MODE_COLOR)
    df = df.set_index("decision_mode").reindex([m for m in MODE_ORDER if m in df["decision_mode"].values]).reset_index()
    return df


def prepare_contract(angle: str) -> pd.DataFrame:
    ensure_exists(INPUT_CONTRACT_CSV)
    df = pd.read_csv(INPUT_CONTRACT_CSV, low_memory=False)
    df = df[df["angle"] == angle].copy()
    df["mode_label"] = df["decision_mode"].map(MODE_LABEL)
    df["color"] = df["decision_mode"].map(MODE_COLOR)
    df = df[df["decision_mode"].isin(MODE_ORDER)].copy()
    return df


def plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, out_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar(df["mode_label"], df[metric], color=df["color"], edgecolor="black", linewidth=1.0)
    for x, y in zip(df["mode_label"], df[metric]):
        if pd.notna(y):
            ax.text(x, y + (0.02 if y <= 1 else 0.2), f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_lead_boxplot(df: pd.DataFrame, out_name: str) -> Path:
    plot_df = df[df["has_risk_signal_before_detect"] == 1].copy()
    order = [m for m in MODE_ORDER if m in plot_df["decision_mode"].unique()]
    data = [plot_df.loc[plot_df["decision_mode"] == mode, "lead_windows"].dropna().tolist() for mode in order]
    labels = [MODE_LABEL[mode] for mode in order]
    colors = [MODE_COLOR[mode] for mode in order]

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_title("Timing: Lead-Window Distribution Before Detection (SELL)")
    ax.set_ylabel("Lead Windows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_first_sell_scatter(df: pd.DataFrame, out_name: str) -> Path:
    plot_df = df[df["has_risk_signal_before_detect"] == 1].copy()
    order = [m for m in MODE_ORDER if m in plot_df["decision_mode"].unique()]
    x_map = {mode: idx for idx, mode in enumerate(order)}

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    for mode in order:
        sub = plot_df[plot_df["decision_mode"] == mode].copy().reset_index(drop=True)
        x = np.full(len(sub), x_map[mode], dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(sub)) if len(sub) > 1 else np.array([0.0])
        ax.scatter(
            x + jitter,
            sub["first_sell_window"],
            s=70,
            color=MODE_COLOR[mode],
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            label=MODE_LABEL[mode],
        )

    ax.set_xticks(list(x_map.values()), [MODE_LABEL[m] for m in order])
    ax.set_ylabel("First SELL Window")
    ax.set_title("Timing: Contract-Level First SELL Window (SELL)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sell = prepare("sell")
    buy = prepare("buy")
    sell_contract = prepare_contract("sell")

    p1 = plot_metric(sell, "mean_lead_windows", "Timing: Mean Lead Windows Before Detection (SELL)", "Mean Lead Windows", "sell_mean_lead_windows.png")
    p2 = plot_metric(sell, "signal_coverage", "Timing: SELL Signal Coverage Before Detection", "Coverage", "sell_signal_coverage.png")
    p3 = plot_metric(sell, "mean_consecutive_risk_windows", "Timing: Consecutive SELL Windows Before Detection", "Mean Consecutive Risk Windows", "sell_consecutive_windows.png")
    p4 = plot_metric(buy, "never_bought_no_signal_rate", "Buy-Side Special Case: Never Bought and No Avoid Signal", "Rate", "buy_never_bought_no_signal_rate.png")
    p5 = plot_lead_boxplot(sell_contract, "sell_lead_windows_boxplot.png")
    p6 = plot_first_sell_scatter(sell_contract, "sell_first_window_scatter.png")

    print(f"[Saved] {p1.resolve()}")
    print(f"[Saved] {p2.resolve()}")
    print(f"[Saved] {p3.resolve()}")
    print(f"[Saved] {p4.resolve()}")
    print(f"[Saved] {p5.resolve()}")
    print(f"[Saved] {p6.resolve()}")


if __name__ == "__main__":
    main()
