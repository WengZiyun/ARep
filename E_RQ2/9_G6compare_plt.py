from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "output" / "E" / "9_G6compare_v9_sybil"
DEFAULT_SUMMARY_CSV = DEFAULT_RESULT_DIR / "g6_compare_plot_summary.csv"
DEFAULT_EXTRA_PLOT_DIR = DEFAULT_RESULT_DIR / "extra_replot"

METHOD_ORDER = ["our", "our_no_u0", "birank", "pagerank", "eigentrust"]
METHOD_LABELS = {
    "our": "Our",
    "our_no_u0": "Our-NoU0",
    "birank": "BiRank",
    "pagerank": "PageRank",
    "eigentrust": "EigenTrust",
}
METHOD_COLORS = {
    "our": "#8c2d04",
    "our_no_u0": "#e6550d",
    "birank": "#d95f0e",
    "pagerank": "#2171b5",
    "eigentrust": "#08306b",
}
K_STYLES = ["-", "--", "-.", ":"]
K_MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


def _load_plot_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing plot summary CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required_base = {
        "method",
        "k_ratio",
        "attack_mode",
        "attack_multiplier",
        "contract_precision_at_k_mean",
        "contract_precision_at_k_ci95",
        "contract_recall_at_k_mean",
        "contract_recall_at_k_ci95",
        "user_precision_at_k_mean",
        "user_precision_at_k_ci95",
        "user_recall_at_k_mean",
        "user_recall_at_k_ci95",
    }
    missing = sorted(required_base - set(df.columns))
    if missing:
        raise ValueError(f"Plot summary CSV missing columns: {missing}")
    return df.copy()


def _x_levels(sub_df: pd.DataFrame) -> tuple[list[float], list[str], str]:
    levels = sorted(sub_df["attack_multiplier"].astype(float).dropna().unique().tolist())
    mode = str(sub_df["attack_mode"].iloc[0]) if len(sub_df) else "attack1"
    if mode == "attack2":
        labels = [f"{int(round(v * 100))}%" for v in levels]
        xlabel = "Drop Ratio"
    else:
        labels = [f"{v:g}x" for v in levels]
        xlabel = "Outlier Multiplier"
    return levels, labels, xlabel


def _plot_metric_by_method_panels(
    plot_df: pd.DataFrame,
    attack_mode: str,
    mean_col: str,
    ci_col: str,
    ylabel: str,
    title: str,
    out_png: Path,
    out_svg: Path,
) -> None:
    sub_df = plot_df[plot_df["attack_mode"].astype(str).eq(str(attack_mode))].copy()
    if sub_df.empty:
        return
    methods = [m for m in METHOD_ORDER if m in sub_df["method"].astype(str).unique().tolist()]
    if not methods:
        return
    levels, labels, xlabel = _x_levels(sub_df)
    x_map = {float(v): i for i, v in enumerate(levels)}
    ratios = sorted(sub_df["k_ratio"].astype(float).dropna().unique().tolist())

    n = len(methods)
    ncols = 2
    nrows = int(math.ceil(float(n) / float(ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13.2, 4.1 * nrows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for idx, method in enumerate(methods):
        r = idx // ncols
        c = idx % ncols
        ax = axes_arr[r, c]
        method_df = sub_df[sub_df["method"].astype(str).eq(method)].copy()
        for ki, ratio in enumerate(ratios):
            line = method_df[np.isclose(method_df["k_ratio"].astype(float), float(ratio))].sort_values("attack_multiplier")
            if line.empty:
                continue
            x = line["attack_multiplier"].astype(float).map(x_map).to_numpy(dtype=float)
            y = line[mean_col].astype(float).to_numpy()
            ci = line[ci_col].fillna(0.0).astype(float).to_numpy()
            k_pct = int(round(float(ratio) * 100.0))
            ax.plot(
                x,
                y,
                color=METHOD_COLORS.get(method, "#444444"),
                linestyle=K_STYLES[ki % len(K_STYLES)],
                marker=K_MARKERS[ki % len(K_MARKERS)],
                linewidth=2.2,
                markersize=4.8,
                label=f"K={k_pct}%",
            )
            ax.fill_between(x, y - ci, y + ci, color=METHOD_COLORS.get(method, "#444444"), alpha=0.10)
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.grid(True, alpha=0.25)
        ax.set_xticks(np.arange(len(labels), dtype=float))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=2)

    for j in range(len(methods), nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes_arr[r, c].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_svg, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_scalar_metric_by_method_panels(
    plot_df: pd.DataFrame,
    attack_mode: str,
    mean_col: str,
    ci_col: str,
    ylabel: str,
    title: str,
    out_png: Path,
    out_svg: Path,
) -> None:
    sub_df = plot_df[plot_df["attack_mode"].astype(str).eq(str(attack_mode))].copy()
    if sub_df.empty or mean_col not in sub_df.columns or ci_col not in sub_df.columns:
        return
    methods = [m for m in METHOD_ORDER if m in sub_df["method"].astype(str).unique().tolist()]
    if not methods:
        return
    # AUPRC does not depend on K; collapse duplicated K rows.
    sub_df = (
        sub_df.sort_values(["method", "attack_multiplier", "k_ratio"])
        .groupby(["method", "attack_mode", "attack_multiplier"], dropna=False, as_index=False)
        .first()
    )
    levels, labels, xlabel = _x_levels(sub_df)
    x_map = {float(v): i for i, v in enumerate(levels)}

    n = len(methods)
    ncols = 2
    nrows = int(math.ceil(float(n) / float(ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13.2, 4.1 * nrows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for idx, method in enumerate(methods):
        r = idx // ncols
        c = idx % ncols
        ax = axes_arr[r, c]
        method_df = sub_df[sub_df["method"].astype(str).eq(method)].sort_values("attack_multiplier")
        if method_df.empty:
            continue
        x = method_df["attack_multiplier"].astype(float).map(x_map).to_numpy(dtype=float)
        y = method_df[mean_col].astype(float).to_numpy()
        ci = method_df[ci_col].fillna(0.0).astype(float).to_numpy()
        ax.plot(
            x,
            y,
            color=METHOD_COLORS.get(method, "#444444"),
            linestyle="-",
            marker="o",
            linewidth=2.2,
            markersize=4.8,
            label=METHOD_LABELS.get(method, method),
        )
        ax.fill_between(x, y - ci, y + ci, color=METHOD_COLORS.get(method, "#444444"), alpha=0.10)
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.grid(True, alpha=0.25)
        ax.set_xticks(np.arange(len(labels), dtype=float))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    for j in range(len(methods), nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes_arr[r, c].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_svg, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_extra_plots(plot_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(outdir / "g6_compare_plot_summary_reloaded.csv", index=False)
    metrics: list[tuple[str, str, str, str]] = [
        ("contract_precision_at_k_mean", "contract_precision_at_k_ci95", "Contract Precision@K (Bottom-K)", "contract_precision_at_k"),
        ("contract_recall_at_k_mean", "contract_recall_at_k_ci95", "Contract Recall@K (Bottom-K)", "contract_recall_at_k"),
        ("user_precision_at_k_mean", "user_precision_at_k_ci95", "User Precision@K (Bottom-K)", "user_precision_at_k"),
        ("user_recall_at_k_mean", "user_recall_at_k_ci95", "User Recall@K (Bottom-K)", "user_recall_at_k"),
    ]
    optional_metrics = [
        (
            "contract_precision_at_k_excl_zombie_mean",
            "contract_precision_at_k_excl_zombie_ci95",
            "Contract Precision@K Excluding Zombie (Bottom-K)",
            "contract_precision_at_k_excl_zombie",
        ),
        (
            "contract_recall_at_k_excl_zombie_mean",
            "contract_recall_at_k_excl_zombie_ci95",
            "Contract Recall@K Excluding Zombie (Bottom-K)",
            "contract_recall_at_k_excl_zombie",
        ),
    ]
    for mean_col, ci_col, ylabel, short in optional_metrics:
        if mean_col in plot_df.columns and ci_col in plot_df.columns:
            metrics.append((mean_col, ci_col, ylabel, short))
    attack_modes = sorted(plot_df["attack_mode"].astype(str).dropna().unique().tolist())
    for attack_mode in attack_modes:
        mode_tag = str(attack_mode).lower()
        for mean_col, ci_col, ylabel, short in metrics:
            _plot_metric_by_method_panels(
                plot_df=plot_df,
                attack_mode=attack_mode,
                mean_col=mean_col,
                ci_col=ci_col,
                ylabel=ylabel,
                title=f"G6 {attack_mode} {ylabel} vs Intensity (Replot, Per-Method Panels)",
                out_png=outdir / f"g6_{mode_tag}_{short}_vs_intensity_replot_panels.png",
                out_svg=outdir / f"g6_{mode_tag}_{short}_vs_intensity_replot_panels.svg",
            )
        _plot_scalar_metric_by_method_panels(
            plot_df=plot_df,
            attack_mode=attack_mode,
            mean_col="contract_auprc_excl_zombie_mean",
            ci_col="contract_auprc_excl_zombie_ci95",
            ylabel="Contract AUPRC Excluding Zombie",
            title=f"G6 {attack_mode} Contract AUPRC Excluding Zombie vs Intensity (Replot)",
            out_png=outdir / f"g6_{mode_tag}_contract_auprc_excl_zombie_vs_intensity_replot_panels.png",
            out_svg=outdir / f"g6_{mode_tag}_contract_auprc_excl_zombie_vs_intensity_replot_panels.svg",
        )
        _plot_scalar_metric_by_method_panels(
            plot_df=plot_df,
            attack_mode=attack_mode,
            mean_col="user_auprc_mean",
            ci_col="user_auprc_ci95",
            ylabel="User AUPRC",
            title=f"G6 {attack_mode} User AUPRC vs Intensity (Replot)",
            out_png=outdir / f"g6_{mode_tag}_user_auprc_vs_intensity_replot_panels.png",
            out_svg=outdir / f"g6_{mode_tag}_user_auprc_vs_intensity_replot_panels.svg",
        )


def main() -> None:
    plot_df = _load_plot_summary(DEFAULT_SUMMARY_CSV)
    save_extra_plots(plot_df, DEFAULT_EXTRA_PLOT_DIR)
    print(f"[g6_plt] done | src={DEFAULT_SUMMARY_CSV} | out={DEFAULT_EXTRA_PLOT_DIR}")


if __name__ == "__main__":
    main()
