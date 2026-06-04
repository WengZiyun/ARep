from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "output" / "E" / "6_G3Acompare2_methodsplit_v1" / "g3a_bucket_compare_summary.csv"
OUTDIR = ROOT / "output" / "E" / "6_G3Acompare2_plt_methodsplit_v2"

PRIMARY_PARAM = {
    "our": 0.7,
    "birank": 0.7,
    "pagerank": 0.7,
    "eigentrust": 0.7,
}

TARGET_BUCKET = (70, 80)
METHOD_ORDER = ["our", "birank", "pagerank", "eigentrust"]
METHOD_LABELS = {
    "our": "Our",
    "birank": "BiRank",
    "pagerank": "PageRank",
    "eigentrust": "EigenTrust",
}
METHOD_COLORS = {
    "our": "#8c2d04",
    "birank": "#d95f0e",
    "pagerank": "#2171b5",
    "eigentrust": "#08306b",
}
FONT_SCALE = 3.4
MONTH_ORDER = ["m00_pre", "m01_attack", "m02_post1", "m03_post2", "m04_post3", "m05_post4", "m06_post5", "m07_post6"]
MONTH_LABELS = {
    "m00_pre": "Pre",
    "m01_attack": "Attack",
    "m02_post1": "Post1",
    "m03_post2": "Post2",
    "m04_post3": "Post3",
    "m05_post4": "Post4",
    "m06_post5": "Post5",
    "m07_post6": "Post6",
}


def fs(size: float) -> float:
    return float(size) * float(FONT_SCALE)


def set_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": fs(12),
            "axes.titlesize": fs(13),
            "axes.labelsize": fs(12),
            "xtick.labelsize": fs(11),
            "ytick.labelsize": fs(11),
            "legend.fontsize": fs(11),
            "axes.linewidth": 1.0,
        }
    )


def _filter_primary(df: pd.DataFrame) -> pd.DataFrame:
    keep_mask = np.zeros(len(df), dtype=bool)
    for method, param_value in PRIMARY_PARAM.items():
        keep_mask |= (df["method"] == method) & np.isclose(df["param_value"], float(param_value))
    return df.loc[keep_mask].copy().reset_index(drop=True)


def _ci95(std: np.ndarray, n: np.ndarray) -> np.ndarray:
    safe_n = np.maximum(n.astype(float), 1.0)
    return 1.96 * std.astype(float) / np.sqrt(safe_n)


def load_bucket_summary(summary_csv: Path, bucket_start: int, bucket_end: int) -> pd.DataFrame:
    summary_df = _filter_primary(pd.read_csv(summary_csv))
    summary_df = summary_df[
        (summary_df["bucket_from_bottom_pct_start"].astype(int) == int(bucket_start))
        & (summary_df["bucket_from_bottom_pct_end"].astype(int) == int(bucket_end))
    ].copy()
    summary_df["month_idx_plot"] = summary_df["month_tag"].map({tag: idx for idx, tag in enumerate(MONTH_ORDER)})
    return summary_df.sort_values(["method", "month_idx_plot"]).reset_index(drop=True)


def _plot_month_ci(
    ax,
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    invert_y: bool = False,
) -> None:
    for method in METHOD_ORDER:
        sub = df[df["method"] == method].sort_values("month_idx_plot")
        if sub.empty:
            continue
        x = sub["month_idx_plot"].to_numpy(dtype=float)
        mean = sub[mean_col].to_numpy(dtype=float)
        std = sub[std_col].fillna(0.0).to_numpy(dtype=float)
        n = sub["n_repeats_actual"].fillna(1).to_numpy(dtype=float)
        ci = _ci95(std, n)
        ax.plot(x, mean, marker="o", linewidth=2.2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, mean - ci, mean + ci, color=METHOD_COLORS[method], alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(MONTH_ORDER), dtype=float))
    ax.set_xticklabels([MONTH_LABELS[tag] for tag in MONTH_ORDER], rotation=20, ha="right")
    # Keep small horizontal gaps on both sides for publication aesthetics.
    ax.set_xlim(-0.15, float(len(MONTH_ORDER) - 1) + 0.15)
    ax.margins(x=0.0)
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)


def plot_bucket_confidence_panels(summary_df: pd.DataFrame, outdir: Path, bucket_start: int, bucket_end: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(22.0, 14.0))
    bucket_text = f"{bucket_start}-{bucket_end}% from bottom"
    _plot_month_ci(
        axes[0, 0],
        summary_df,
        "malicious_contract_rank_mean",
        "malicious_contract_rank_std",
        f"(a) Malicious Contract Rank\n({bucket_text}, 95% CI)",
        "Rank",
    )
    _plot_month_ci(
        axes[0, 1],
        summary_df,
        "malicious_contract_score_mean",
        "malicious_contract_score_std",
        f"(b) Malicious Contract Score\n({bucket_text}, 95% CI)",
        "Score",
    )
    _plot_month_ci(
        axes[1, 0],
        summary_df,
        "rebel_user_rank_mean",
        "rebel_user_rank_std",
        f"(c) Rebel User Rank\n({bucket_text}, 95% CI)",
        "Rank",
    )
    _plot_month_ci(
        axes[1, 1],
        summary_df,
        "rebel_user_score_mean",
        "rebel_user_score_std",
        f"(d) Rebel User Score\n({bucket_text}, 95% CI)",
        "Score",
    )
    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.tick_params(width=1.0, length=4)
    legend_handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], marker="o", linewidth=2.2, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.95, bottom=0.20, wspace=0.28, hspace=0.46)
    out_path = outdir / f"g3a_bucket_{bucket_start:02d}_{bucket_end:02d}_trajectory_ci95.png"
    out_pdf = outdir / f"g3a_bucket_{bucket_start:02d}_{bucket_end:02d}_trajectory_ci95.pdf"
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_bucket_ab_horizontal_inverted(summary_df: pd.DataFrame, outdir: Path, bucket_start: int, bucket_end: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(22.0, 8.0))
    bucket_text = f"{bucket_start}-{bucket_end}% from bottom"
    _plot_month_ci(
        axes[0],
        summary_df,
        "malicious_contract_rank_mean",
        "malicious_contract_rank_std",
        f"(a) Malicious ASA Rank\n({bucket_text}, 95% CI)",
        "Rank",
        invert_y=True,
    )
    _plot_month_ci(
        axes[1],
        summary_df,
        "rebel_user_rank_mean",
        "rebel_user_rank_std",
        f"(b) Rebel FMA Rank\n({bucket_text}, 95% CI)",
        "Rank",
        invert_y=True,
    )

    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.tick_params(width=1.0, length=4)

    legend_handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], marker="o", linewidth=2.2, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.30, wspace=0.24)

    out_path = outdir / f"g3a_bucket_{bucket_start:02d}_{bucket_end:02d}_ab_horizontal_inverted_ci95.png"
    out_pdf = outdir / f"g3a_bucket_{bucket_start:02d}_{bucket_end:02d}_ab_horizontal_inverted_ci95.pdf"
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary csv: {SUMMARY_CSV}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_pub_style()
    bucket_start, bucket_end = TARGET_BUCKET
    summary_df = load_bucket_summary(SUMMARY_CSV, bucket_start, bucket_end)
    if summary_df.empty:
        raise ValueError(f"No rows found for bucket {bucket_start}-{bucket_end} in {SUMMARY_CSV}.")
    saved_path = plot_bucket_confidence_panels(summary_df, OUTDIR, bucket_start, bucket_end)
    saved_path_ab = plot_bucket_ab_horizontal_inverted(summary_df, OUTDIR, bucket_start, bucket_end)
    print(f"[Saved] {saved_path}")
    print(f"[Saved] {saved_path_ab}")


if __name__ == "__main__":
    main()
