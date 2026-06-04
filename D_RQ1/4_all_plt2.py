#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ============================================================
# Config
# ============================================================
RUG_METRICS_CSV = Path("output/D/1_eval_rugpull_plt2/metrics_mean_by_delta.csv")
RUG_AUPRC_CSV = Path("output/D/1_eval_rugpull_plt2/auprc_mean_by_delta.csv")

COUNTERFEIT_METRICS_CSV = Path("output/D/2_eval_counterfeit_plt/metrics_mean_by_delta.csv")
COUNTERFEIT_AUPRC_CSV = Path("output/D/2_eval_counterfeit_plt/auprc_mean_by_delta.csv")

WASH_METRICS_CSV = Path("output/D/3_eval_wash_plt2/metrics_mean_by_delta.csv")
WASH_AUPRC_CSV = Path("output/D/3_eval_wash_plt2/auprc_mean_by_delta.csv")

OUTPUT_DIR = Path("output/D/4_all_plt")
OUTPUT_FIG = OUTPUT_DIR / "all_in_one_vertical_3x2.png"
OUTPUT_FIG_PDF = OUTPUT_DIR / "all_in_one_vertical_3x2.pdf"
FIG_DPI = 220
FONT_SCALE = 2

# Use all available delta windows by default.
MAX_DELTA: int | None = None

# Smooth curve to reduce high-frequency jitter.
SMOOTH_WINDOW = 3
SMOOTH_CENTER = True
SMOOTH_EWMA_SPAN = 3
SPIKE_SIGMA = 3.0
MARKER_EVERY = 5

LEFT_COL_XMAX = 51
RIGHT_COL_XMAX = 31
LEFT_XPAD = 1.0

COLOR_K10 = "#8c2d04"
COLOR_K20 = "#d95f0e"
COLOR_K30 = "#2171b5"


def fs(size: float) -> float:
    return float(size) * float(FONT_SCALE)


def set_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def load_one(metrics_csv: Path, auprc_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_exists(metrics_csv)
    ensure_exists(auprc_csv)
    m = pd.read_csv(metrics_csv, low_memory=False)
    _ = pd.read_csv(auprc_csv, low_memory=False)  # reserved for compatibility with existing outputs

    need_m = {"delta_windows", "top_percent", "recall_mean", "precision_mean"}
    miss_m = [c for c in need_m if c not in m.columns]
    if miss_m:
        raise ValueError(f"Missing columns in {metrics_csv}: {miss_m}")

    m["delta_windows"] = pd.to_numeric(m["delta_windows"], errors="coerce")
    m["top_percent"] = pd.to_numeric(m["top_percent"], errors="coerce")
    m["recall_mean"] = pd.to_numeric(m["recall_mean"], errors="coerce")
    m["precision_mean"] = pd.to_numeric(m["precision_mean"], errors="coerce")
    m = m.dropna(subset=["delta_windows", "top_percent", "recall_mean", "precision_mean"]).copy()

    if MAX_DELTA is not None:
        m = m[m["delta_windows"] <= int(MAX_DELTA)].copy()
    return m, pd.DataFrame()


def series_by_top_percent(
    metrics: pd.DataFrame, metric_col: str, top_percent: float
) -> tuple[pd.Series, pd.Series]:
    sub = metrics[metrics["top_percent"] == float(top_percent)].sort_values("delta_windows", kind="mergesort")
    x = sub["delta_windows"].reset_index(drop=True)
    y = (sub[metric_col] * 100.0).astype(float).reset_index(drop=True)

    # Despike: replace abrupt isolated points by local median trend.
    y_med = y.rolling(window=5, min_periods=1, center=True).median()
    resid = (y - y_med).abs()
    mad = float(np.nanmedian((resid - np.nanmedian(resid)).__abs__()))
    scale = 1.4826 * mad if mad > 0 else float(np.nanstd(resid))
    if np.isfinite(scale) and scale > 0:
        spike_mask = resid > (float(SPIKE_SIGMA) * scale)
        y = y.where(~spike_mask, y_med)

    if SMOOTH_WINDOW > 1:
        y = y.rolling(window=int(SMOOTH_WINDOW), min_periods=1, center=bool(SMOOTH_CENTER)).mean()
    if SMOOTH_EWMA_SPAN > 1:
        y = y.ewm(span=int(SMOOTH_EWMA_SPAN), adjust=False).mean()
    return x, y


def plot_panel(
    ax,
    title: str,
    panel_tag: str,
    metrics: pd.DataFrame,
    mode: str,
    xlim: tuple[float, float],
    xticks: list[int],
    ylim: tuple[float, float],
) -> None:
    x10, y10 = series_by_top_percent(metrics, mode, 0.10)
    x20, y20 = series_by_top_percent(metrics, mode, 0.20)
    x30, y30 = series_by_top_percent(metrics, mode, 0.30)

    keep10 = (x10 >= xlim[0]) & (x10 <= xlim[1])
    keep20 = (x20 >= xlim[0]) & (x20 <= xlim[1])
    keep30 = (x30 >= xlim[0]) & (x30 <= xlim[1])

    ax.plot(
        x10[keep10],
        y10[keep10],
        color=COLOR_K10,
        marker="o",
        markersize=3.0,
        markevery=int(MARKER_EVERY),
        linewidth=1.4,
        label="K=10%",
    )
    ax.plot(
        x20[keep20],
        y20[keep20],
        color=COLOR_K20,
        marker="o",
        markersize=3.0,
        markevery=int(MARKER_EVERY),
        linewidth=1.4,
        label="K=20%",
    )
    ax.plot(
        x30[keep30],
        y30[keep30],
        color=COLOR_K30,
        marker="o",
        markersize=3.0,
        markevery=int(MARKER_EVERY),
        linewidth=1.4,
        label="K=30%",
    )
    ax.set_title(f"{panel_tag} {title}", fontsize=fs(10))
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(*ylim)
    ax.grid(True, which="major", linewidth=0.8, color="0.80", alpha=0.55)
    ax.tick_params(labelsize=fs(9))
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_alpha(0.95)


def add_zoom_inset(
    ax,
    metrics: pd.DataFrame,
    mode: str,
    xlim: tuple[float, float],
    xticks: list[int],
) -> None:
    inset = inset_axes(ax, width="43%", height="43%", loc="upper right", borderpad=0.9)
    series = [
        (0.10, COLOR_K10),
        (0.20, COLOR_K20),
        (0.30, COLOR_K30),
    ]

    y_values: list[float] = []
    for top_percent, color in series:
        x, y = series_by_top_percent(metrics, mode, top_percent)
        keep = (x >= xlim[0]) & (x <= xlim[1])
        y_values.extend(y[keep].dropna().astype(float).tolist())
        inset.plot(
            x[keep],
            y[keep],
            color=color,
            marker="o",
            markersize=2.2,
            markevery=int(MARKER_EVERY),
            linewidth=1.1,
        )

    if y_values:
        y_max = float(np.nanmax(y_values))
        y_top = max(1.0, y_max * 1.35)
    else:
        y_top = 1.0

    inset.set_xlim(*xlim)
    inset.set_xticks(xticks)
    inset.set_ylim(0.0, y_top)
    inset.set_yticks(np.linspace(0.0, y_top, 3))
    inset.grid(True, which="major", linewidth=0.6, color="0.82", alpha=0.55)
    inset.tick_params(labelsize=fs(6.5), width=0.8, length=2.5)
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.95)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.35", lw=0.9, alpha=0.9)


def infer_axis_ranges(*metrics_list: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    x_vals = []
    y_vals_recall = []
    y_vals_precision = []
    for m in metrics_list:
        x_vals.extend(pd.to_numeric(m["delta_windows"], errors="coerce").dropna().tolist())
        y_vals_recall.extend((pd.to_numeric(m["recall_mean"], errors="coerce").dropna() * 100.0).tolist())
        y_vals_precision.extend((pd.to_numeric(m["precision_mean"], errors="coerce").dropna() * 100.0).tolist())

    if not x_vals:
        return (-0.5, 10.5), (0.0, 100.0), (0.0, 40.0)
    x_min = float(min(x_vals))
    x_max = float(max(x_vals))
    x_pad = max(0.5, (x_max - x_min) * 0.04)

    def _infer_y(y_vals: list[float], default_hi: float) -> tuple[float, float]:
        if not y_vals:
            return (0.0, default_hi)
        y_min = float(np.nanmin(y_vals))
        y_max = float(np.nanmax(y_vals))
        y_low = max(0.0, y_min - max(1.5, (y_max - y_min) * 0.10))
        y_high = y_max + max(2.5, (y_max - y_min) * 0.15)
        if y_high <= y_low:
            return (0.0, default_hi)
        return (y_low, y_high)

    return (x_min - x_pad, x_max + x_pad), _infer_y(y_vals_recall, 80.0), _infer_y(y_vals_precision, 80.0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    set_pub_style()

    rug_m, rug_a = load_one(RUG_METRICS_CSV, RUG_AUPRC_CSV)
    cnt_m, cnt_a = load_one(COUNTERFEIT_METRICS_CSV, COUNTERFEIT_AUPRC_CSV)
    wash_m, wash_a = load_one(WASH_METRICS_CSV, WASH_AUPRC_CSV)
    _xlim, _ylim_recall, _ylim_precision = infer_axis_ranges(rug_m, cnt_m, wash_m)

    fig, axes = plt.subplots(3, 2, figsize=(9.2, 10.6), sharex=False, sharey="col")

    xlim_left = (0.0 - float(LEFT_XPAD), float(LEFT_COL_XMAX))
    xlim_right = (0.0 - float(LEFT_XPAD), float(RIGHT_COL_XMAX))
    xticks_left = [0, 15, 30, 45]
    xticks_right = [0, 10, 20, 30]
    ylim_recall = (0.0, 110.0)
    ylim_precision = (0.0, 40.0)

    # Rows: datasets. Columns: Recall@K, Precision@K.
    plot_panel(axes[0, 0], "Rug Pull Recall@K", "(a)", rug_m, "recall_mean", xlim_left, xticks_left, ylim_recall)
    plot_panel(axes[0, 1], "Rug Pull Precision@K", "(b)", rug_m, "precision_mean", xlim_left, xticks_left, ylim_precision)

    plot_panel(axes[1, 0], "Counterfeit Recall@K", "(c)", cnt_m, "recall_mean", xlim_left, xticks_left, ylim_recall)
    plot_panel(axes[1, 1], "Counterfeit Precision@K", "(d)", cnt_m, "precision_mean", xlim_left, xticks_left, ylim_precision)

    plot_panel(axes[2, 0], "Wash Trading Recall@K", "(e)", wash_m, "recall_mean", xlim_right, xticks_right, ylim_recall)
    plot_panel(axes[2, 1], "Wash Trading Precision@K", "(f)", wash_m, "precision_mean", xlim_right, xticks_right, ylim_precision)
    add_zoom_inset(axes[2, 1], wash_m, "precision_mean", xlim_right, xticks_right)

    for ax in axes[:, 0]:
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    for ax in axes[:, 1]:
        ax.set_yticks([0, 10, 20, 30, 40])

    axes[0, 0].set_ylabel("Recall@K (%)", fontsize=fs(11))
    axes[1, 0].set_ylabel("Recall@K (%)", fontsize=fs(11))
    axes[2, 0].set_ylabel("Recall@K (%)", fontsize=fs(11))
    axes[0, 1].set_ylabel("Precision@K (%)", fontsize=fs(11))
    axes[1, 1].set_ylabel("Precision@K (%)", fontsize=fs(11))
    axes[2, 1].set_ylabel("Precision@K (%)", fontsize=fs(11))
    for j in range(2):
        axes[2, j].set_xlabel("Delta windows", fontsize=fs(10))

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        frameon=True,
        fontsize=fs(10),
    )

    fig.subplots_adjust(left=0.10, right=0.985, top=0.955, bottom=0.14, wspace=0.28, hspace=0.34)
    fig.savefig(OUTPUT_FIG, dpi=FIG_DPI)
    fig.savefig(OUTPUT_FIG_PDF)
    plt.close(fig)

    print(f"[Saved] {OUTPUT_FIG.resolve()}")
    print(f"[Saved] {OUTPUT_FIG_PDF.resolve()}")


if __name__ == "__main__":
    main()
