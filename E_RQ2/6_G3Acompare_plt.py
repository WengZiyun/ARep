from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "output" / "E" / "6_G3Acompare_fourway_v4_bucketwide_p070" / "g3a_bucket_compare_summary.csv"
DELTA_CSV = ROOT / "output" / "E" / "6_G3Acompare_fourway_v4_bucketwide_p070" / "g3a_attack_month_delta_summary.csv"
OUTDIR = ROOT / "output" / "E" / "6_G3Acompare_plt_v4_bucketwide_p070"

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


def _filter_primary(df: pd.DataFrame) -> pd.DataFrame:
    keep_mask = np.zeros(len(df), dtype=bool)
    for method, param_value in PRIMARY_PARAM.items():
        keep_mask |= (df["method"] == method) & np.isclose(df["param_value"], float(param_value))
    return df.loc[keep_mask].copy().reset_index(drop=True)


def load_primary_tables(summary_csv: Path, delta_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = _filter_primary(pd.read_csv(summary_csv))
    delta_df = _filter_primary(pd.read_csv(delta_csv))
    summary_df["month_index"] = summary_df["month_index"].astype(int)
    summary_df["bucket_mid"] = (
        summary_df["bucket_from_bottom_pct_start"].astype(float) + summary_df["bucket_from_bottom_pct_end"].astype(float)
    ) / 2.0
    delta_df["bucket_mid"] = (
        delta_df["bucket_from_bottom_pct_start"].astype(float) + delta_df["bucket_from_bottom_pct_end"].astype(float)
    ) / 2.0
    return summary_df, delta_df


def _plot_bucket_series(ax, df: pd.DataFrame, mean_col: str, std_col: str, title: str, ylabel: str) -> None:
    for method in METHOD_ORDER:
        sub = df[df["method"] == method].sort_values("bucket_from_bottom_pct_start")
        if sub.empty:
            continue
        x = sub["bucket_mid"].to_numpy(dtype=float)
        mean = sub[mean_col].to_numpy(dtype=float)
        std = sub[std_col].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, mean - std, mean + std, color=METHOD_COLORS[method], alpha=0.14)
    ax.set_title(title)
    ax.set_xlabel("Rebel reputation bucket from bottom (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(5.0, 100.0, 10.0))
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_month_series(ax, df: pd.DataFrame, mean_col: str, std_col: str, title: str, ylabel: str) -> None:
    df = df.copy()
    df["month_idx_plot"] = df["month_tag"].map({tag: idx for idx, tag in enumerate(MONTH_ORDER)})
    for method in METHOD_ORDER:
        sub = df[df["method"] == method].sort_values("month_idx_plot")
        if sub.empty:
            continue
        x = sub["month_idx_plot"].to_numpy(dtype=float)
        mean = sub[mean_col].to_numpy(dtype=float)
        std = sub[std_col].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, mean - std, mean + std, color=METHOD_COLORS[method], alpha=0.14)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(MONTH_ORDER), dtype=float))
    ax.set_xticklabels([MONTH_LABELS[tag] for tag in MONTH_ORDER])
    ax.grid(True, alpha=0.3)
    ax.legend()


def _pick_bucket_examples(summary_df: pd.DataFrame) -> list[tuple[int, int]]:
    bucket_pairs = (
        summary_df[["bucket_from_bottom_pct_start", "bucket_from_bottom_pct_end"]]
        .drop_duplicates()
        .sort_values(["bucket_from_bottom_pct_start", "bucket_from_bottom_pct_end"])
        .itertuples(index=False, name=None)
    )
    pairs = list(bucket_pairs)
    if len(pairs) <= 3:
        return pairs
    return [pairs[0], pairs[len(pairs) // 2], pairs[-1]]


def plot_attack_month_bucket_impact(delta_df: pd.DataFrame, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    _plot_bucket_series(
        axes[0],
        delta_df,
        "malicious_contract_rank_delta_from_pre_mean",
        "malicious_contract_rank_delta_from_pre_std",
        "Attack-Month Malicious Contract Rank Delta",
        "Rank delta from pre",
    )
    _plot_bucket_series(
        axes[1],
        delta_df,
        "rebel_user_rank_delta_from_pre_mean",
        "rebel_user_rank_delta_from_pre_std",
        "Attack-Month Rebel User Rank Delta",
        "Rank delta from pre",
    )
    fig.tight_layout()
    out_path = outdir / "g3a_attack_month_bucket_impact_primary.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_attack_month_bucket_scores(delta_df: pd.DataFrame, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    _plot_bucket_series(
        axes[0],
        delta_df,
        "malicious_contract_score_delta_from_pre_mean",
        "malicious_contract_score_delta_from_pre_std",
        "Attack-Month Malicious Contract Score Delta",
        "Score delta from pre",
    )
    _plot_bucket_series(
        axes[1],
        delta_df,
        "rebel_user_score_delta_from_pre_mean",
        "rebel_user_score_delta_from_pre_std",
        "Attack-Month Rebel User Score Delta",
        "Score delta from pre",
    )
    fig.tight_layout()
    out_path = outdir / "g3a_attack_month_bucket_scores_primary.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_bucket_trajectory_panels(summary_df: pd.DataFrame, outdir: Path) -> list[Path]:
    saved: list[Path] = []
    for bucket_start, bucket_end in _pick_bucket_examples(summary_df):
        sub = summary_df[
            (summary_df["bucket_from_bottom_pct_start"] == int(bucket_start))
            & (summary_df["bucket_from_bottom_pct_end"] == int(bucket_end))
        ].copy()
        if sub.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
        _plot_month_series(
            axes[0, 0],
            sub,
            "malicious_contract_rank_mean",
            "malicious_contract_rank_std",
            f"Malicious Contract Rank ({bucket_start}-{bucket_end}% from bottom)",
            "Rank",
        )
        _plot_month_series(
            axes[0, 1],
            sub,
            "malicious_contract_score_mean",
            "malicious_contract_score_std",
            f"Malicious Contract Score ({bucket_start}-{bucket_end}% from bottom)",
            "Score",
        )
        _plot_month_series(
            axes[1, 0],
            sub,
            "rebel_user_rank_mean",
            "rebel_user_rank_std",
            f"Rebel User Rank ({bucket_start}-{bucket_end}% from bottom)",
            "Rank",
        )
        _plot_month_series(
            axes[1, 1],
            sub,
            "rebel_user_score_mean",
            "rebel_user_score_std",
            f"Rebel User Score ({bucket_start}-{bucket_end}% from bottom)",
            "Score",
        )
        fig.tight_layout()
        out_path = outdir / f"g3a_bucket_{bucket_start:02d}_{bucket_end:02d}_trajectory_primary.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
    return saved


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary csv: {SUMMARY_CSV}")
    if not DELTA_CSV.exists():
        raise FileNotFoundError(f"Missing delta csv: {DELTA_CSV}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary_df, delta_df = load_primary_tables(SUMMARY_CSV, DELTA_CSV)

    saved = [
        plot_attack_month_bucket_impact(delta_df, OUTDIR),
        plot_attack_month_bucket_scores(delta_df, OUTDIR),
    ]
    saved.extend(plot_bucket_trajectory_panels(summary_df, OUTDIR))

    for path in saved:
        print(f"[Saved] {path}")


if __name__ == "__main__":
    main()
