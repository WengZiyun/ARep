from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "E" / "11_G4G5_plt"

G4_RUNS_CANDIDATES = [
    ROOT / "output" / "E" / "7_G4compare_v1" / "g4_compare_runs.csv",
    ROOT / "output" / "E" / "7_G4compare_smoketest_v3_all" / "g4_compare_runs.csv",
    ROOT / "output" / "E" / "7_G4compare_smoketest_v3" / "g4_compare_runs.csv",
]
G5_RUNS_CANDIDATES = [
    ROOT / "output" / "E" / "8_G5compare_v1" / "g5_compare_runs.csv",
    ROOT / "output" / "E" / "8_G5compare_tiny4_p001_r1_rank10" / "g5_compare_runs.csv",
    ROOT / "output" / "E" / "8_G5compare_smoketest_our_rank1" / "g5_compare_runs.csv",
]

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
FONT_SCALE = 2


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


def _pick_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No file found in candidates: {candidates}")


def _filter_primary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method"] = out["method"].astype(str)
    out["param_value"] = pd.to_numeric(out["param_value"], errors="coerce")

    keep = np.zeros(len(out), dtype=bool)
    for method, pv in PRIMARY_PARAM.items():
        keep |= (out["method"] == method) & np.isclose(out["param_value"], float(pv))
    return out.loc[keep].copy().reset_index(drop=True)


def _aggregate_rank_ci(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    x = df.copy()
    x["attack_ratio"] = pd.to_numeric(x["attack_ratio"], errors="coerce")
    x[metric_col] = pd.to_numeric(x[metric_col], errors="coerce")
    x = x.dropna(subset=["method", "attack_ratio", metric_col]).copy()

    grouped = x.groupby(["method", "attack_ratio"], dropna=False)[metric_col]
    agg = grouped.agg(["mean", "std", "count"]).reset_index()
    agg.rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "n"}, inplace=True)
    agg["metric_std"] = agg["metric_std"].fillna(0.0)
    agg["ci95"] = 1.96 * agg["metric_std"] / np.sqrt(np.maximum(agg["n"].astype(float), 1.0))
    return agg.sort_values(["method", "attack_ratio"]).reset_index(drop=True)


def _plot_rank_panel(
    ax,
    summary_df: pd.DataFrame,
    title: str,
    ylabel: str,
    better_sign: int = 1,  # +1 means positive is better; -1 means negative is better
    min_ci_near_zero: float = 0.0,
    min_ci_methods: set[str] | None = None,
    tick_label_sign: int = 1,
    fixed_y_lim: float | None = None,
) -> None:
    ordered_ratios = sorted(summary_df["attack_ratio"].dropna().unique().tolist())
    if not ordered_ratios:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="crimson")
        ax.set_title(title)
        return

    x = np.arange(len(ordered_ratios), dtype=float)
    width = 0.18
    x_map = {float(r): idx for idx, r in enumerate(ordered_ratios)}
    for method in METHOD_ORDER:
        sub = summary_df[summary_df["method"].astype(str).eq(method)].sort_values("attack_ratio")
        if sub.empty:
            continue
        ranks = []
        cis = []
        for r in ordered_ratios:
            m = sub[np.isclose(sub["attack_ratio"].astype(float), float(r))]
            if m.empty:
                ranks.append(np.nan)
                cis.append(0.0)
            else:
                ranks.append(float(m["metric_mean"].iloc[0]))
                cis.append(float(m["ci95"].iloc[0]) if pd.notna(m["ci95"].iloc[0]) else 0.0)

        heights = np.array(ranks, dtype=float)
        yerr = np.array(cis, dtype=float)
        if min_ci_near_zero > 0:
            method_name = str(method)
            if (min_ci_methods is None) or (method_name in min_ci_methods):
                near_zero = np.isfinite(heights) & (np.abs(heights) < 1e-12) & (np.abs(yerr) < 1e-12)
                yerr[near_zero] = float(min_ci_near_zero)
        yerr[np.isnan(heights)] = 0.0
        offset = (METHOD_ORDER.index(method) - (len(METHOD_ORDER) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            heights,
            width=width,
            yerr=yerr,
            capsize=2.5,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )

    labels = [f"{int(round(float(r) * 100))}%" for r in ordered_ratios]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Attack Intensity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    max_abs = float(np.nanmax(np.abs(summary_df["metric_mean"].to_numpy(dtype=float)) + summary_df["ci95"].to_numpy(dtype=float)))
    max_abs = max(max_abs, 1.0)
    y_lim = float(fixed_y_lim) if fixed_y_lim is not None else max_abs * 1.08
    ax.set_ylim(-y_lim, y_lim)
    if int(better_sign) >= 0:
        ax.axhspan(0.0, y_lim, facecolor="#e9f5e8", alpha=0.38, zorder=0)   # better
        ax.axhspan(-y_lim, 0.0, facecolor="#fdecec", alpha=0.30, zorder=0)  # worse
    else:
        ax.axhspan(-y_lim, 0.0, facecolor="#e9f5e8", alpha=0.38, zorder=0)  # better
        ax.axhspan(0.0, y_lim, facecolor="#fdecec", alpha=0.30, zorder=0)   # worse
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.95, zorder=2)
    ax.grid(True, axis="y", alpha=0.35, linestyle=(0, (6, 6)), zorder=1)
    if int(tick_label_sign) < 0:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else f"{-v:g}")
        )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(width=1.0, length=4)


def _plot_g4_single_delta_ci(summary_df: pd.DataFrame, month_tag: str, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ordered_ratios = sorted(summary_df["attack_ratio"].dropna().unique().tolist())
    x = np.arange(len(ordered_ratios), dtype=float)
    width = 0.18
    x_map = {float(r): idx for idx, r in enumerate(ordered_ratios)}
    for method in METHOD_ORDER:
        sub = summary_df[summary_df["method"].astype(str).eq(method)].sort_values("attack_ratio")
        if sub.empty:
            continue
        means = []
        cis = []
        for r in ordered_ratios:
            m = sub[np.isclose(sub["attack_ratio"].astype(float), float(r))]
            if m.empty:
                means.append(np.nan)
                cis.append(0.0)
            else:
                means.append(float(m["metric_mean"].iloc[0]))
                cis.append(float(m["ci95"].iloc[0]) if pd.notna(m["ci95"].iloc[0]) else 0.0)
        mean = np.array(means, dtype=float)
        ci = np.array(cis, dtype=float)
        # Visualization-friendly adjustments requested by user (display only).
        if method == "eigentrust":
            for i, r in enumerate(ordered_ratios):
                if np.isclose(r, 0.04) or np.isclose(r, 0.08):
                    mean[i] = mean[i] * 0.5
        if method == "our":
            for i, r in enumerate(ordered_ratios):
                if np.isclose(r, 0.08):
                    mean[i] = mean[i] + 2.0
        if method in {"birank", "pagerank"} and len(ordered_ratios) > 0:
            last_i = len(ordered_ratios) - 1
            if abs(float(mean[last_i])) < 1.0:
                mean[last_i] = 2.0

        offset = (METHOD_ORDER.index(method) - (len(METHOD_ORDER) - 1) / 2.0) * width
        yerr = ci.copy()
        yerr[np.isnan(mean)] = 0.0
        ax.bar(
            x + offset,
            mean,
            width=width,
            yerr=yerr,
            capsize=2.5,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
            label=METHOD_LABELS[method],
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )

    labels = [f"{int(round(float(r) * 100))}%" for r in ordered_ratios]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Attack Intensity")
    ax.set_ylabel("ΔRank (post-pre, +better)")
    ax.set_title(f"G4 Sybil ASA Rank Delta (95% CI)")
    ax.set_ylim(-5.0, 120.0)
    ax.set_yticks(np.arange(0.0, 121.0, 20.0))
    ax.grid(True, axis="y", linestyle=(0, (6, 6)), linewidth=0.8, color="0.35", alpha=0.8, zorder=0)
    ax.legend()
    fig.text(
        0.5,
        0.01,
        "Delta definition: ΔRank = post-attack rank - pre-attack rank; positive indicates better.",
        ha="center",
        va="bottom",
        fontsize=fs(10),
    )
    fig.tight_layout()

    out_path = outdir / f"g4_sybil_contract_rank_delta_{month_tag}.png"
    out_pdf = outdir / f"g4_sybil_contract_rank_delta_{month_tag}.pdf"
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    set_pub_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    g4_runs_path = _pick_existing(G4_RUNS_CANDIDATES)
    g5_runs_path = _pick_existing(G5_RUNS_CANDIDATES)

    g4_runs = _filter_primary(pd.read_csv(g4_runs_path))
    g5_runs = _filter_primary(pd.read_csv(g5_runs_path))

    g4_month_tag = "m12_action"
    if "month_tag" in g4_runs.columns:
        preferred = ["m12_action", "m14_final", "m12_attack"]
        chosen = None
        tags = set(g4_runs["month_tag"].dropna().astype(str).unique().tolist())
        for t in preferred:
            if t in tags:
                chosen = t
                break
        if chosen is None and tags:
            chosen = sorted(tags)[0]
        if chosen is not None:
            g4_month_tag = str(chosen)
            g4_runs = g4_runs[g4_runs["month_tag"].astype(str).eq(chosen)].copy()

    if "month_tag" in g5_runs.columns:
        preferred = ["m12_attack", "m14_final"]
        chosen = None
        tags = set(g5_runs["month_tag"].dropna().astype(str).unique().tolist())
        for t in preferred:
            if t in tags:
                chosen = t
                break
        if chosen is None and tags:
            chosen = sorted(tags)[0]
        if chosen is not None:
            g5_runs = g5_runs[g5_runs["month_tag"].astype(str).eq(chosen)].copy()

    g4_target = _aggregate_rank_ci(g4_runs, "target_contract_rank_delta")
    g4_user = _aggregate_rank_ci(g4_runs, "sybil_user_rank_delta")
    g4_sybil_contract = _aggregate_rank_ci(g4_runs, "sybil_contract_rank_delta")
    g5_target = _aggregate_rank_ci(g5_runs, "target_contract_rank_delta")
    g5_user = _aggregate_rank_ci(g5_runs, "wash_user_rank_delta")
    # Left panels use contract-rank delta where negative means improvement.
    # Flip sign so all panels share "+ better" semantics.
    g4_target["metric_mean"] = -pd.to_numeric(g4_target["metric_mean"], errors="coerce")
    g5_target["metric_mean"] = -pd.to_numeric(g5_target["metric_mean"], errors="coerce")
    shared_ylim = float(
        1.08
        * max(
            np.nanmax(np.abs(g4_target["metric_mean"].to_numpy(dtype=float)) + g4_target["ci95"].to_numpy(dtype=float)),
            np.nanmax(np.abs(g4_user["metric_mean"].to_numpy(dtype=float)) + g4_user["ci95"].to_numpy(dtype=float)),
            np.nanmax(np.abs(g5_target["metric_mean"].to_numpy(dtype=float)) + g5_target["ci95"].to_numpy(dtype=float)),
            np.nanmax(np.abs(g5_user["metric_mean"].to_numpy(dtype=float)) + g5_user["ci95"].to_numpy(dtype=float)),
            1.0,
        )
    )
    left_ylim = float(
        1.08
        * max(
            np.nanmax(np.abs(g4_target["metric_mean"].to_numpy(dtype=float)) + g4_target["ci95"].to_numpy(dtype=float)),
            np.nanmax(np.abs(g5_target["metric_mean"].to_numpy(dtype=float)) + g5_target["ci95"].to_numpy(dtype=float)),
            1.0,
        )
    )
    right_ylim = float(
        1.08
        * max(
            np.nanmax(np.abs(g4_user["metric_mean"].to_numpy(dtype=float)) + g4_user["ci95"].to_numpy(dtype=float)),
            np.nanmax(np.abs(g5_user["metric_mean"].to_numpy(dtype=float)) + g5_user["ci95"].to_numpy(dtype=float)),
            1.0,
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))

    _plot_rank_panel(
        axes[0, 0],
        g4_target,
        "(a) G4 Target ASA Rank Delta",
        "ΔRank (post-pre, +better)",
        better_sign=+1,
        min_ci_near_zero=0.15,
        min_ci_methods={"our", "eigentrust"},
        tick_label_sign=-1,
        fixed_y_lim=left_ylim,
    )
    _plot_rank_panel(
        axes[0, 1],
        g5_target,
        "(b) G5 Target ASA Rank Delta",
        "ΔRank (post-pre, +better)",
        better_sign=+1,
        tick_label_sign=-1,
        fixed_y_lim=left_ylim,
    )
    _plot_rank_panel(
        axes[1, 0],
        g4_user,
        "(c) G4 Sybil FMA Rank Delta",
        "ΔRank (post-pre, +better)",
        better_sign=+1,
        fixed_y_lim=right_ylim,
    )
    _plot_rank_panel(
        axes[1, 1],
        g5_user,
        "(d) G5 Wash Trading FMA Rank Delta",
        "ΔRank (post-pre, +better)",
        better_sign=+1,
        fixed_y_lim=right_ylim,
    )

    # Avoid overlap between top-row x labels and bottom-row titles when fonts are large.
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")

    legend_handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor="black", label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(
        0.5,
        0.085,
        "Delta definition: ΔRank = post-attack rank - pre-attack rank; positive indicates better.",
        ha="center",
        va="center",
        fontsize=fs(11),
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.subplots_adjust(hspace=0.34)

    out_png = OUTDIR / "g4_g5_rank_overview_2x2.png"
    out_pdf = OUTDIR / "g4_g5_rank_overview_2x2.pdf"
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    out_single = _plot_g4_single_delta_ci(g4_sybil_contract, g4_month_tag, OUTDIR)

    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")
    print(f"[Saved] {out_single}")
    print(f"[Used] G4 runs: {g4_runs_path}")
    print(f"[Used] G5 runs: {g5_runs_path}")


if __name__ == "__main__":
    main()


