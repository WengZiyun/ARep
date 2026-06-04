from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "output" / "E" / "6_G3compare_fourway_v1" / "g3_scale_compare_summary.csv"
OUTDIR = ROOT / "output" / "E" / "6_G3compare_plt"

PRIMARY_PARAM = {
    "our": 0.7,
    "birank": 0.7,
    "pagerank": 0.7,
    "eigentrust": 0.7,
}

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


def load_primary_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    keep_mask = np.zeros(len(df), dtype=bool)
    for method, param_value in PRIMARY_PARAM.items():
        keep_mask |= (df["method"] == method) & np.isclose(df["param_value"], float(param_value))
    return df.loc[keep_mask].sort_values(["rebel_ratio", "method"]).reset_index(drop=True)


def _plot_metric(ax, summary_df: pd.DataFrame, mean_col: str, std_col: str, title: str, ylabel: str) -> None:
    for method in METHOD_ORDER:
        sub = summary_df[summary_df["method"] == method].sort_values("rebel_ratio")
        if sub.empty:
            continue
        x = sub["rebel_ratio"].to_numpy(dtype=float)
        mean = sub[mean_col].to_numpy(dtype=float)
        std = sub[std_col].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, mean - std, mean + std, color=METHOD_COLORS[method], alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel("rebel_ratio")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary_df = load_primary_summary(SUMMARY_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_metric(
        axes[0],
        summary_df,
        "target_contract_mean_norm_rank_mean",
        "target_contract_mean_norm_rank_std",
        "G3 Target Contract Mean Normalized Rank",
        "Mean normalized rank",
    )
    _plot_metric(
        axes[1],
        summary_df,
        "g3_user_mean_norm_rank_mean",
        "g3_user_mean_norm_rank_std",
        "G3 Rebel User Mean Normalized Rank",
        "Mean normalized rank",
    )
    fig.tight_layout()
    out_path = OUTDIR / "g3_mean_norm_rank_primary_0.7.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_metric(
        axes[0],
        summary_df,
        "target_contract_best_rank_mean",
        "target_contract_best_rank_std",
        "Best G3 Target Contract Rank",
        "Best raw rank",
    )
    _plot_metric(
        axes[1],
        summary_df,
        "g3_user_best_rank_mean",
        "g3_user_best_rank_std",
        "Best G3 Rebel User Rank",
        "Best raw rank",
    )
    fig.tight_layout()
    out_path = OUTDIR / "g3_best_rank_primary_0.7.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_metric(
        axes[0],
        summary_df,
        "g3_user_rank_deterioration_mean",
        "g3_user_rank_deterioration_std",
        "G3 Rebel User Rank Deterioration",
        "Post rank - Pre rank",
    )
    _plot_metric(
        axes[1],
        summary_df,
        "g3_user_score_drop_mean",
        "g3_user_score_drop_std",
        "G3 Rebel User Score Drop",
        "Pre score - Post score",
    )
    fig.tight_layout()
    out_path = OUTDIR / "g3_rebel_penalty_primary_0.7.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
