from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "E" / "10_G1G2_plt"

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
FONT_SCALE = 3


def fs(size: float) -> float:
    return float(size) * float(FONT_SCALE)


def set_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": fs(12),
            "axes.titlesize": fs(14),
            "axes.labelsize": fs(13),
            "xtick.labelsize": fs(11),
            "ytick.labelsize": fs(11),
            "legend.fontsize": fs(12),
            "axes.linewidth": 1.0,
        }
    )


METHOD_PARAM_META = {
    "our": {"suffix": "beta", "score_prefix": "4_stageC", "contract_rank_col": "rank_C_contract"},
    "birank": {"suffix": "eta", "score_prefix": "1_birank", "contract_rank_col": "rank_contract"},
    "pagerank": {"suffix": "gamma", "score_prefix": "2_pagerank", "contract_rank_col": "rank_contract"},
    "eigentrust": {"suffix": "alpha", "score_prefix": "3_eigentrust", "contract_rank_col": "rank_contract"},
}

G1_SUMMARY_CANDIDATES = [
    ROOT / "output" / "E" / "4_G1compare_fourway_v3" / "g1_scale_compare_summary.csv",
    ROOT / "output" / "E" / "4_G1compare_fourway_v2" / "g1_scale_compare_summary.csv",
    ROOT / "output" / "E" / "4_G1compare_fourway" / "g1_scale_compare_summary.csv",
    ROOT / "output" / "E" / "4_G1compare" / "g1_scale_compare_summary.csv",
]
G2_SUMMARY_CANDIDATES = [
    ROOT / "output" / "E" / "5_G2compare_fourway_v1" / "g2_scale_compare_summary.csv",
]
MAX_COLUMNS_PER_PANEL = 6


def _pick_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No file found in candidates: {candidates}")


def _limit_scales(scales: list[int], max_columns: int = MAX_COLUMNS_PER_PANEL) -> list[int]:
    if len(scales) <= max_columns:
        return scales
    idx = np.linspace(0, len(scales) - 1, max_columns)
    chosen = sorted({scales[int(round(i))] for i in idx})
    if len(chosen) > max_columns:
        chosen = chosen[:max_columns]
    return chosen


def _load_primary_summary(summary_csv: Path, scale_col: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)

    keep_mask = np.zeros(len(df), dtype=bool)
    for method, p in PRIMARY_PARAM.items():
        keep_mask |= (df["method"].astype(str) == method) & np.isclose(pd.to_numeric(df["param_value"], errors="coerce"), float(p))

    out = df.loc[keep_mask].copy()
    out[scale_col] = pd.to_numeric(out[scale_col], errors="coerce")
    out = out.dropna(subset=[scale_col]).copy()
    out[scale_col] = out[scale_col].astype(int)
    return out


def _g1_scale_tag(n_sybil: int) -> str:
    return f"n{int(n_sybil):04d}"


def _g2_scale_tag(fake_trades_per_attacker: int) -> str:
    return f"t{int(fake_trades_per_attacker):04d}"


def _contract_score_path(result_root: Path, method: str, scale_tag: str, param_value: float) -> Path:
    meta = METHOD_PARAM_META[method]
    suffix = meta["suffix"]
    prefix = meta["score_prefix"]
    outdir = result_root / f"{method}_{scale_tag}"
    return outdir / f"{prefix}_contract_scores_{suffix}{float(param_value):.2f}.csv"


def _compute_g1_no_zombie_mean_rank(g1_primary: pd.DataFrame, g1_result_root: Path) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for row in g1_primary.itertuples(index=False):
        method = str(row.method)
        n_sybil = int(row.n_sybil)
        param_value = float(row.param_value)
        score_path = _contract_score_path(g1_result_root, method, _g1_scale_tag(n_sybil), param_value)
        df_c = pd.read_csv(score_path)

        df_eval = df_c[~df_c["contract_id"].astype(str).str.startswith("Zombie")].copy()
        rank_col = METHOD_PARAM_META[method]["contract_rank_col"]
        df_eval = df_eval.sort_values(rank_col, ascending=True).reset_index(drop=True)
        df_eval["rank_no_zombie"] = np.arange(1, len(df_eval) + 1)
        sybil_eval = df_eval[df_eval["contract_id"].astype(str).str.startswith("Sybil")].copy()
        records.append(
            {
                "method": method,
                "n_sybil": n_sybil,
                "n_eval": int(len(df_eval)),
                "mean_rank_no_zombie": float(sybil_eval["rank_no_zombie"].mean()) if len(sybil_eval) else np.nan,
            }
        )
    return pd.DataFrame(records).sort_values(["n_sybil", "method"]).reset_index(drop=True)


def _compute_g2_no_zombie_mean_rank(g2_primary: pd.DataFrame, g2_result_root: Path) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for row in g2_primary.itertuples(index=False):
        method = str(row.method)
        scale = int(row.fake_trades_per_attacker)
        param_value = float(row.param_value)
        score_path = _contract_score_path(g2_result_root, method, _g2_scale_tag(scale), param_value)
        df_c = pd.read_csv(score_path)

        df_eval = df_c[~df_c["contract_id"].astype(str).str.startswith("Zombie")].copy()
        rank_col = METHOD_PARAM_META[method]["contract_rank_col"]
        df_eval = df_eval.sort_values(rank_col, ascending=True).reset_index(drop=True)
        df_eval["rank_no_zombie"] = np.arange(1, len(df_eval) + 1)
        target_eval = df_eval[df_eval["contract_id"].astype(str).str.startswith("Sybil")].copy()
        records.append(
            {
                "method": method,
                "fake_trades_per_attacker": scale,
                "n_eval": int(len(df_eval)),
                "mean_rank_no_zombie": float(target_eval["rank_no_zombie"].mean()) if len(target_eval) else np.nan,
            }
        )
    return pd.DataFrame(records).sort_values(["fake_trades_per_attacker", "method"]).reset_index(drop=True)


def _plot_grouped_raw_rank_bar(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    scale_col: str,
    metric_col: str,
    ylabel: str,
    title: str,
    xlabel: str,
) -> None:
    plot_df = summary_df.copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col]).copy()

    scales = sorted(int(x) for x in plot_df[scale_col].unique())
    scales = _limit_scales(scales)
    if not scales:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", color="crimson", transform=ax.transAxes)
        ax.set_title(title, fontsize=fs(12))
        ax.axis("off")
        return

    x = np.arange(len(scales), dtype=float)
    width = 0.18

    for idx, method in enumerate(METHOD_ORDER):
        sub = plot_df[plot_df["method"] == method].sort_values(scale_col)
        values = []
        for s in scales:
            m = sub[sub[scale_col] == s]
            values.append(float(m[metric_col].iloc[0]) if not m.empty else np.nan)

        offset = (idx - (len(METHOD_ORDER) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_title(title, fontsize=fs(12))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scales], rotation=0)
    ax.grid(True, axis="y", linestyle=(0, (6, 6)), linewidth=0.8, color="0.35", alpha=0.8, zorder=0)
    y_max = float(np.nanmax(plot_df[metric_col].to_numpy(dtype=float)))
    ax.set_ylim(0.0, y_max * 1.12 if y_max > 0 else 1.0)


def _plot_grouped_inverted_rank_bar(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    scale_col: str,
    rank_col: str,
    n_eval_col: str,
    ylabel: str,
    title: str,
    xlabel: str,
    drop_scales: set[int] | None = None,
    relabel_zero_to_one: bool = False,
    force_include_scale_zero: bool = False,
) -> None:
    plot_df = summary_df.copy()
    plot_df[rank_col] = pd.to_numeric(plot_df[rank_col], errors="coerce")
    plot_df[n_eval_col] = pd.to_numeric(plot_df[n_eval_col], errors="coerce")
    if force_include_scale_zero:
        zero_mask = plot_df[scale_col].astype(int) == 0
        plot_df.loc[zero_mask & plot_df[rank_col].isna(), rank_col] = plot_df.loc[zero_mask & plot_df[rank_col].isna(), n_eval_col]
    plot_df = plot_df.dropna(subset=[rank_col, n_eval_col]).copy()
    if drop_scales:
        plot_df = plot_df[~plot_df[scale_col].isin(drop_scales)].copy()

    scales = sorted(int(x) for x in plot_df[scale_col].unique())
    scales = _limit_scales(scales)
    if not scales:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", color="crimson", transform=ax.transAxes)
        ax.set_title(title, fontsize=fs(12))
        ax.axis("off")
        return

    x = np.arange(len(scales), dtype=float)
    width = 0.18
    rank_max = float(plot_df[n_eval_col].max())
    rank_min = 1.0

    for idx, method in enumerate(METHOD_ORDER):
        sub = plot_df[plot_df["method"] == method].sort_values(scale_col)
        ranks = []
        for s in scales:
            m = sub[sub[scale_col] == s]
            ranks.append(float(m[rank_col].iloc[0]) if not m.empty else np.nan)
        heights = [rank_max - r if pd.notna(r) else np.nan for r in ranks]

        offset = (idx - (len(METHOD_ORDER) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            heights,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_title(title, fontsize=fs(12))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    if relabel_zero_to_one:
        labels = ["1" if int(s) == 0 else str(int(s)) for s in scales]
    else:
        labels = [str(int(s)) for s in scales]
    ax.set_xticklabels(labels, rotation=0)
    if rank_max > 10000:
        tick_step = 2000
    elif rank_max > 5000:
        tick_step = 1000
    elif rank_max > 1000:
        tick_step = 200
    elif rank_max > 200:
        tick_step = 50
    else:
        tick_step = 20
    tick_labels = np.arange(rank_max, rank_min - 1e-9, -tick_step, dtype=float)
    if tick_labels[-1] != rank_min:
        tick_labels = np.append(tick_labels, rank_min)
    tick_positions = rank_max - tick_labels
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{int(t)}" for t in tick_labels])
    ax.set_ylim(0, rank_max - rank_min)
    ax.grid(True, axis="y", linestyle=(0, (6, 6)), linewidth=0.8, color="0.35", alpha=0.8, zorder=0)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_pub_style()

    g1_summary_path = _pick_existing(G1_SUMMARY_CANDIDATES)
    g2_summary_path = _pick_existing(G2_SUMMARY_CANDIDATES)
    g1_result_root = g1_summary_path.parent
    g2_result_root = g2_summary_path.parent

    g1_primary = _load_primary_summary(g1_summary_path, scale_col="n_sybil")
    g2_primary = _load_primary_summary(g2_summary_path, scale_col="fake_trades_per_attacker")
    g1_no_zombie = _compute_g1_no_zombie_mean_rank(g1_primary, g1_result_root)
    g2_no_zombie = _compute_g2_no_zombie_mean_rank(g2_primary, g2_result_root)

    # Larger canvas + extra margins to avoid clipping after font scaling.
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))

    _plot_grouped_inverted_rank_bar(
        axes[0, 0],
        summary_df=g1_no_zombie,
        scale_col="n_sybil",
        rank_col="mean_rank_no_zombie",
        n_eval_col="n_eval",
        ylabel="Mean rank ",
        title="(a) G1 Malicious ASA Mean Rank",
        xlabel="",
        drop_scales={0},
        relabel_zero_to_one=False,
    )
    _plot_grouped_inverted_rank_bar(
        axes[0, 1],
        summary_df=g2_no_zombie,
        scale_col="fake_trades_per_attacker",
        rank_col="mean_rank_no_zombie",
        n_eval_col="n_eval",
        ylabel="Mean rank ",
        title="(b) G2 Malicious ASA Mean Rank",
        xlabel="",
        drop_scales={0, 1, 269},
        relabel_zero_to_one=False,
    )

    _plot_grouped_inverted_rank_bar(
        axes[1, 0],
        summary_df=g1_primary,
        scale_col="n_sybil",
        rank_col="g1_user_mean_rank",
        n_eval_col="n_users_total",
        ylabel="Mean rank",
        title="(c) G1 Malicious FMA Mean Rank",
        xlabel="Number of Sybil accounts",
        drop_scales={0},
        relabel_zero_to_one=False,
    )
    _plot_grouped_inverted_rank_bar(
        axes[1, 1],
        summary_df=g2_primary,
        scale_col="fake_trades_per_attacker",
        rank_col="g2_user_mean_rank",
        n_eval_col="n_users_total",
        ylabel="Mean rank",
        title="(d) G2 Malicious FMA Mean Rank",
        xlabel="Forged interactions per attacker",
        drop_scales={0, 1, 269},
        relabel_zero_to_one=False,
    )

    g1_bottom = g1_primary.copy()
    g1_bottom["n_users_total"] = pd.to_numeric(g1_bottom["n_users_total"], errors="coerce")
    g1_bottom = g1_bottom[g1_bottom["n_sybil"] != 0]
    g2_bottom = g2_primary.copy()
    g2_bottom["n_users_total"] = pd.to_numeric(g2_bottom["n_users_total"], errors="coerce")
    g2_bottom = g2_bottom[~g2_bottom["fake_trades_per_attacker"].isin([0, 1, 269])]
    bottom_rank_max = float(
        np.nanmax(
            np.concatenate(
                [
                    g1_bottom["n_users_total"].dropna().to_numpy(dtype=float),
                    g2_bottom["n_users_total"].dropna().to_numpy(dtype=float),
                ]
            )
        )
    )
    bottom_rank_min = 1.0
    bottom_ylim_max = bottom_rank_max - bottom_rank_min
    shared_pos = np.linspace(0.0, bottom_ylim_max, 7)
    shared_labels = [int(round(bottom_rank_max - p)) for p in shared_pos]
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_ylim(0.0, bottom_ylim_max)
        ax.set_yticks(shared_pos)
        ax.set_yticklabels([str(x) for x in shared_labels])

    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.tick_params(width=1.0, length=4)

    legend_handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor="black", label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.03))

    fig.subplots_adjust(left=0.12, right=0.99, top=0.93, bottom=0.20, wspace=0.30, hspace=0.30)

    out_path = OUTDIR / "g1_g2_2x2_overview.png"
    out_pdf = OUTDIR / "g1_g2_2x2_overview.pdf"
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {out_path}")
    print(f"[Saved] {out_pdf}")
    print(f"[Used] G1 summary: {g1_summary_path}")
    print(f"[Used] G2 summary: {g2_summary_path}")


if __name__ == "__main__":
    main()
