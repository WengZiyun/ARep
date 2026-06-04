from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "output" / "E" / "4_G1compare_fourway_v2" / "g1_scale_compare_summary.csv"
RESULT_ROOT = ROOT / "output" / "E" / "4_G1compare_fourway_v2"
OUTDIR = ROOT / "output" / "E" / "4_G1compare_plt"

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

EXCLUDE_SCALES = {13720}

METHOD_PARAM_META = {
    "our": {"suffix": "beta", "score_prefix": "4_stageC", "contract_rank_col": "rank_C_contract"},
    "birank": {"suffix": "eta", "score_prefix": "1_birank", "contract_rank_col": "rank_contract"},
    "pagerank": {"suffix": "gamma", "score_prefix": "2_pagerank", "contract_rank_col": "rank_contract"},
    "eigentrust": {"suffix": "alpha", "score_prefix": "3_eigentrust", "contract_rank_col": "rank_contract"},
}


def load_primary_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)

    keep_mask = np.zeros(len(df), dtype=bool)
    for method, param_value in PRIMARY_PARAM.items():
        keep_mask |= (df["method"] == method) & np.isclose(df["param_value"], float(param_value))

    out = df.loc[keep_mask].copy()
    out = out[~out["n_sybil"].isin(EXCLUDE_SCALES)].copy()
    out = out.sort_values(["n_sybil", "method"]).reset_index(drop=True)
    return out


def _scale_tag(n_sybil: int) -> str:
    return f"n{int(n_sybil):04d}"


def _contract_score_path(method: str, n_sybil: int, param_value: float) -> Path:
    meta = METHOD_PARAM_META[method]
    suffix = meta["suffix"]
    prefix = meta["score_prefix"]
    outdir = RESULT_ROOT / f"{method}_{_scale_tag(n_sybil)}"
    return outdir / f"{prefix}_contract_scores_{suffix}{float(param_value):.2f}.csv"


def compute_sybil_mean_rank_excluding_zombie(summary_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for row in summary_df.itertuples(index=False):
        method = str(row.method)
        n_sybil = int(row.n_sybil)
        param_value = float(row.param_value)
        score_path = _contract_score_path(method, n_sybil, param_value)
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
                "param_value": param_value,
                "n_contracts_eval": int(len(df_eval)),
                "sybil_contract_mean_rank_no_zombie": float(sybil_eval["rank_no_zombie"].mean()) if len(sybil_eval) else np.nan,
            }
        )

    return pd.DataFrame(records).sort_values(["n_sybil", "method"]).reset_index(drop=True)


def plot_sybil_mean_rank_bar(summary_df: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)

    scales = [int(x) for x in sorted(summary_df["n_sybil"].unique())]
    x = np.arange(len(scales), dtype=float)
    width = 0.18
    rank_max = float(summary_df["n_contracts_eval"].max())
    rank_min = 1.0

    fig, ax = plt.subplots(figsize=(12, 5.8))

    for idx, method in enumerate(METHOD_ORDER):
        sub = summary_df[summary_df["method"] == method].sort_values("n_sybil")
        ranks = []
        for scale in scales:
            matched = sub[sub["n_sybil"] == scale]
            ranks.append(float(matched["sybil_contract_mean_rank_no_zombie"].iloc[0]) if not matched.empty else np.nan)

        heights = [rank_max - rank if pd.notna(rank) else np.nan for rank in ranks]

        offset = (idx - (len(METHOD_ORDER) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            heights,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_title("Sybil Mean Rank without Zombie Collections (Primary = 0.7)")
    ax.set_xlabel("n_sybil")
    ax.set_ylabel("Mean sybil rank (Zombie excluded)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(scale) for scale in scales], rotation=0)
    tick_step = 50 if rank_max > 200 else 20
    tick_labels = np.arange(rank_max, rank_min - 1e-9, -tick_step, dtype=float)
    if tick_labels[-1] != rank_min:
        tick_labels = np.append(tick_labels, rank_min)
    tick_positions = rank_max - tick_labels
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{int(tick)}" for tick in tick_labels])
    ax.set_ylim(0, rank_max - rank_min)
    ax.grid(True, axis="y", linestyle=(0, (6, 6)), linewidth=0.8, color="0.35", alpha=0.8, zorder=0)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    out_path = outdir / "g1_scale_sybil_mean_rank_bar_primary_0.7_no_zombie.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    summary_df = load_primary_summary(SUMMARY_CSV)
    summary_no_zombie = compute_sybil_mean_rank_excluding_zombie(summary_df)
    out_path = plot_sybil_mean_rank_bar(summary_no_zombie, OUTDIR)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
