#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import importlib.util
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
#热力图的横轴表示用户累计买入投入的分位组，纵轴表示用户中位持有期的分位组。每个网格对应一类同时具有特定持有行为和投入水平的用户群体，
# 网格中的数值为该群体的平均 rank_pct，数值越小表示声誉越高。结果显示，从左上角的“短持有、低投入”到右下角的“长持有、高投入”，平均 rank_pct 整体呈下降趋势，
# 说明用户声誉受到持有稳定性与资本投入的共同影响。
#其中，左上角单元格代表持有期和投入均处于最低分位的用户，通常对应最低声誉群体；
# 右下角单元格代表持有期和投入均处于最高分位的用户，通常对应最高声誉群体。


# ============================================================
# Config
# ============================================================
BADRANK_PLOT_SCRIPT = Path("D_RQ1/3_eval_badrank_plt.py")
TRP_SCRIPT = Path("D_RQ1/0_rugpullTRP1.py")
SNAPSHOT_USER_SCORES_CSV = Path("output/D/3_eval_wash2/snapshot_user_scores.csv")
SNAPSHOT_PLAN_CSV = Path("output/D/3_eval_wash2/snapshot_plan.csv")
SNAPSHOT_RUN_META_CSV = Path("output/D/3_eval_wash2/run_meta.csv")
OUTPUT_DIR = Path("output/D/3_eval_badrank_hold_invest2")
FAST_PLOT_FROM_EXISTING_JOINT = True
PRECOMPUTED_JOINT_CSV = OUTPUT_DIR / "joint_user_hold_invest_snapshot_level.csv"

REPRESENTATIVE_SNAPSHOT_IDX = 40
QUANTILE_BINS = 5
FIG_DPI = 140
SAVE_PDF = True
FONT_SCALE = 2.0
HOLDING_DAY_BINS = [0, 1, 3, 7, 14, 30, 90, 180, 365, np.inf]
HOLDING_DAY_BIN_LABELS = [
    "[0,1)",
    "[1,3)",
    "[3,7)",
    "[7,14)",
    "[14,30)",
    "[30,90)",
    "[90,180)",
    "[180,365)",
    "[365,+)",
]


def fs(size: float) -> float:
    return float(size) * float(FONT_SCALE)


def set_pub_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": fs(12),
            "axes.labelsize": fs(11),
            "xtick.labelsize": fs(10),
            "ytick.labelsize": fs(10),
            "legend.fontsize": fs(10),
            "axes.linewidth": 0.9,
        }
    )


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_user_buy_events(df_all: pd.DataFrame, start_block: int, max_end_block: int) -> pd.DataFrame:
    req = ["buyer", "block_number", "tx_type", "price"]
    miss = [c for c in req if c not in df_all.columns]
    if miss:
        raise ValueError(f"Missing columns in events for investment calc: {miss}")

    work = df_all.copy()
    work["block_number"] = pd.to_numeric(work["block_number"], errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce").fillna(0.0)
    work["tx_type"] = work["tx_type"].astype(str).str.lower().str.strip()
    work["buyer"] = work["buyer"].fillna("").astype(str).str.lower().str.strip()
    work = work.dropna(subset=["block_number"]).copy()
    work["block_number"] = work["block_number"].astype(np.int64)

    work = work[
        (work["block_number"] >= int(start_block))
        & (work["block_number"] <= int(max_end_block))
        & (work["tx_type"].isin({"mint", "trade"}))
        & (work["buyer"] != "")
    ].copy()

    return work[["buyer", "block_number", "price"]].sort_values(
        ["block_number", "buyer"], kind="mergesort"
    ).reset_index(drop=True)


def compute_user_investment(
    buy_events_df: pd.DataFrame,
    start_block: int,
    end_block_t: int,
) -> pd.DataFrame:
    if buy_events_df.empty:
        return pd.DataFrame(columns=["user", "buy_total_usd", "buy_count", "buy_avg_usd"])

    sub = buy_events_df[buy_events_df["block_number"] <= int(end_block_t)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["user", "buy_total_usd", "buy_count", "buy_avg_usd"])

    out = (
        sub.groupby("buyer", as_index=False)
        .agg(
            buy_total_usd=("price", "sum"),
            buy_count=("price", "count"),
            buy_avg_usd=("price", "mean"),
        )
        .rename(columns={"buyer": "user"})
    )
    out["buy_total_usd"] = pd.to_numeric(out["buy_total_usd"], errors="coerce").fillna(0.0)
    out["buy_count"] = pd.to_numeric(out["buy_count"], errors="coerce").fillna(0).astype(int)
    out["buy_avg_usd"] = pd.to_numeric(out["buy_avg_usd"], errors="coerce").fillna(0.0)

    span_blocks = max(1.0, float(int(end_block_t) - int(start_block)))
    out["buy_events_per_10k_blocks"] = out["buy_count"].astype(float) / span_blocks * 10_000.0
    return out[
        ["user", "buy_total_usd", "buy_count", "buy_avg_usd", "buy_events_per_10k_blocks"]
    ].sort_values(["buy_total_usd", "buy_count", "user"], ascending=[False, False, True])


def assign_quantile_labels(
    s: pd.Series,
    prefix: str,
    q: int,
) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    valid = x.notna()
    out = pd.Series(index=s.index, dtype="object")
    if int(valid.sum()) == 0:
        return out
    ranked = x[valid].rank(method="first")
    bins = pd.qcut(ranked, q=int(q), labels=[f"{prefix}{i}" for i in range(1, int(q) + 1)])
    out.loc[valid] = bins.astype(str)
    return out


def make_heatmap_table(df: pd.DataFrame, value_col: str, count_col_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    work["holding_q"] = assign_quantile_labels(work["holding_days"], "H", QUANTILE_BINS)
    work["invest_q"] = assign_quantile_labels(work["buy_total_usd"], "I", QUANTILE_BINS)
    work = work.dropna(subset=["holding_q", "invest_q"]).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    value_pivot = work.pivot_table(
        index="holding_q",
        columns="invest_q",
        values=value_col,
        aggfunc="mean",
    )
    count_pivot = work.pivot_table(
        index="holding_q",
        columns="invest_q",
        values="user",
        aggfunc="count",
    ).rename_axis(index=None, columns=None)

    holding_order = [f"H{i}" for i in range(1, QUANTILE_BINS + 1)]
    invest_order = [f"I{i}" for i in range(1, QUANTILE_BINS + 1)]
    value_pivot = value_pivot.reindex(index=holding_order, columns=invest_order)
    count_pivot = count_pivot.reindex(index=holding_order, columns=invest_order)
    count_pivot.name = count_col_name
    return value_pivot, count_pivot


def plot_heatmap(
    value_pivot: pd.DataFrame,
    count_pivot: pd.DataFrame,
    out_png: Path,
    title: str,
    total_users: int | None = None,
) -> None:
    if value_pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    mat = value_pivot.astype(float).values
    vmin = float(np.nanmin(mat)) if np.isfinite(np.nanmin(mat)) else None
    vmax = float(np.nanmax(mat)) if np.isfinite(np.nanmax(mat)) else None
    # Keep high-contrast publication map; rightmost-column text is forced to black below.
    im = ax.imshow(mat, cmap="cividis_r", aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Rank % (lower better)")
    cbar.ax.tick_params(length=3.5, width=0.8, labelsize=fs(10))

    ax.set_xticks(range(value_pivot.shape[1]))
    ax.set_xticklabels(value_pivot.columns.tolist())
    ax.set_yticks(range(value_pivot.shape[0]))
    ax.set_yticklabels(value_pivot.index.tolist())
    ax.set_xlabel("Investment Quantile (I5 = higher buy amount)")
    ax.set_ylabel("Holding Quantile (H5 = longer holding)")
    if total_users is None:
        ax.set_title(title, pad=10)
    else:
        ax.set_title(f"{title}\nN={int(total_users):,}", pad=10)

    # Cell boundaries for publication-style readability.
    ax.set_xticks(np.arange(-0.5, value_pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, value_pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(value_pivot.shape[0]):
        for j in range(value_pivot.shape[1]):
            val = value_pivot.iloc[i, j]
            cnt = count_pivot.iloc[i, j] if not count_pivot.empty else np.nan
            txt = "NA" if pd.isna(val) else f"{float(val):.3f}\n(n={int(cnt) if pd.notna(cnt) else 0})"
            if j == value_pivot.shape[1] - 1:
                color = "black"
            elif pd.isna(val):
                color = "black"
            else:
                # Adaptive text color for contrast.
                norm = (float(val) - vmin) / (vmax - vmin) if (vmin is not None and vmax is not None and vmax > vmin) else 0.5
                color = "white" if norm < 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=fs(8.5), linespacing=1.1)

    plt.tight_layout()
    plt.savefig(out_png, dpi=max(FIG_DPI, 300), bbox_inches="tight")
    if SAVE_PDF:
        out_pdf = out_png.with_suffix(".pdf")
        plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def build_holding_days_segment_table(
    s_holding_days: pd.Series,
) -> pd.DataFrame:
    x = pd.to_numeric(s_holding_days, errors="coerce")
    x = x[x.notna()].astype(float)
    if x.empty:
        return pd.DataFrame(columns=["holding_segment", "n_users", "ratio"])
    x = x.clip(lower=0.0)
    seg = pd.cut(
        x,
        bins=HOLDING_DAY_BINS,
        labels=HOLDING_DAY_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    count = seg.value_counts(sort=False).reindex(HOLDING_DAY_BIN_LABELS).fillna(0).astype(int)
    out = pd.DataFrame(
        {
            "holding_segment": HOLDING_DAY_BIN_LABELS,
            "n_users": count.values,
        }
    )
    n_total = int(out["n_users"].sum())
    out["ratio"] = out["n_users"].astype(float) / float(max(1, n_total))
    return out


def plot_holding_days_segment_bar(
    seg_df: pd.DataFrame,
    out_png: Path,
    title: str,
) -> None:
    if seg_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(seg_df), dtype=float)
    y = seg_df["n_users"].to_numpy(dtype=float)
    bars = ax.bar(
        x,
        y,
        width=0.76,
        color="#2a6f97",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.92,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(seg_df["holding_segment"].tolist(), rotation=30, ha="right")
    ax.set_xlabel("Holding Duration Segment (days)")
    ax.set_ylabel("User Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ymax = float(np.nanmax(y)) if len(y) else 0.0
    ax.set_ylim(0.0, ymax * 1.12 if ymax > 0 else 1.0)

    for i, b in enumerate(bars):
        cnt = int(seg_df.iloc[i]["n_users"])
        ratio = float(seg_df.iloc[i]["ratio"]) * 100.0
        ax.text(
            b.get_x() + b.get_width() * 0.5,
            b.get_height(),
            f"{cnt}\n({ratio:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=fs(8.5),
            color="black",
            linespacing=1.05,
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=max(FIG_DPI, 300), bbox_inches="tight")
    if SAVE_PDF:
        out_pdf = out_png.with_suffix(".pdf")
        plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def ols_fit(df: pd.DataFrame) -> pd.DataFrame:
    need = ["rank_pct", "log_holding_days", "log_buy_total_usd", "interaction"]
    work = df.copy()
    for c in need:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=need).copy()
    if work.empty:
        return pd.DataFrame(
            columns=["term", "coef", "std_err", "t_value", "p_value", "ci_low", "ci_high", "n_obs", "r2", "adj_r2"]
        )

    y = work["rank_pct"].astype(float).to_numpy()
    X = np.column_stack(
        [
            np.ones(len(work), dtype=float),
            work["log_holding_days"].astype(float).to_numpy(),
            work["log_buy_total_usd"].astype(float).to_numpy(),
            work["interaction"].astype(float).to_numpy(),
        ]
    )
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat

    n = int(len(y))
    k = int(X.shape[1])
    dof = max(1, n - k)
    sse = float(resid.T @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    sigma2 = sse / dof
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_vals = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=dof)
    crit = float(stats.t.ppf(0.975, df=dof))
    ci_low = beta - crit * se
    ci_high = beta + crit * se

    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / dof) if dof > 0 and pd.notna(r2) else float("nan")

    rows = []
    terms = ["Intercept", "log_holding_days", "log_buy_total_usd", "interaction"]
    for i, term in enumerate(terms):
        rows.append(
            {
                "term": term,
                "coef": float(beta[i]),
                "std_err": float(se[i]),
                "t_value": float(t_vals[i]),
                "p_value": float(p_vals[i]),
                "ci_low": float(ci_low[i]),
                "ci_high": float(ci_high[i]),
                "n_obs": n,
                "r2": r2,
                "adj_r2": adj_r2,
            }
        )
    return pd.DataFrame(rows)


def summarize_joint_bins(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "snapshot_idx",
                "holding_q",
                "invest_q",
                "n_users",
                "mean_rank_pct",
                "median_rank_pct",
                "median_holding_days",
                "median_buy_total_usd",
            ]
        )

    work = df.copy()
    work["holding_q"] = assign_quantile_labels(work["holding_days"], "H", QUANTILE_BINS)
    work["invest_q"] = assign_quantile_labels(work["buy_total_usd"], "I", QUANTILE_BINS)
    work = work.dropna(subset=["holding_q", "invest_q"]).copy()
    if work.empty:
        return pd.DataFrame()

    out = (
        work.groupby(["snapshot_idx", "holding_q", "invest_q"], as_index=False)
        .agg(
            n_users=("user", "size"),
            mean_rank_pct=("rank_pct", "mean"),
            median_rank_pct=("rank_pct", "median"),
            median_holding_days=("holding_days", "median"),
            median_buy_total_usd=("buy_total_usd", "median"),
        )
        .sort_values(["snapshot_idx", "holding_q", "invest_q"], kind="mergesort")
    )
    return out


def main() -> None:
    t0 = perf_counter()
    set_pub_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_mode = "full_recompute"
    source_joint_csv = ""
    run_meta_prev = pd.DataFrame()
    S: int | None = None
    author_end_block: int | None = None
    n_input_files_total: int | None = None
    n_input_files_main_top50: int | None = None
    n_input_files_normal_top50: int | None = None
    n_snapshots_plan: int | None = None

    if FAST_PLOT_FROM_EXISTING_JOINT and PRECOMPUTED_JOINT_CSV.exists():
        joint_df = pd.read_csv(PRECOMPUTED_JOINT_CSV, low_memory=False)
        run_mode = "fast_from_joint_csv"
        source_joint_csv = str(PRECOMPUTED_JOINT_CSV.resolve())
    else:
        ensure_exists(BADRANK_PLOT_SCRIPT)
        ensure_exists(TRP_SCRIPT)
        ensure_exists(SNAPSHOT_USER_SCORES_CSV)
        ensure_exists(SNAPSHOT_PLAN_CSV)
        ensure_exists(SNAPSHOT_RUN_META_CSV)

        badrank = load_module(BADRANK_PLOT_SCRIPT, "badrank_hold_invest_mod")
        trp = load_module(TRP_SCRIPT, "trp_hold_invest_mod")

        snapshot_users = pd.read_csv(SNAPSHOT_USER_SCORES_CSV, low_memory=False)
        snapshot_plan = pd.read_csv(SNAPSHOT_PLAN_CSV, low_memory=False)
        run_meta_prev = pd.read_csv(SNAPSHOT_RUN_META_CSV, low_memory=False)

        user_bin_df = badrank.build_user_rank_bins(snapshot_users)

        snapshot_plan = snapshot_plan.copy()
        snapshot_plan["snapshot_idx"] = pd.to_numeric(snapshot_plan["snapshot_idx"], errors="coerce")
        snapshot_plan["start_block"] = pd.to_numeric(snapshot_plan["start_block"], errors="coerce")
        snapshot_plan["end_block"] = pd.to_numeric(snapshot_plan["end_block"], errors="coerce")
        snapshot_plan = snapshot_plan.dropna(subset=["snapshot_idx", "start_block", "end_block"]).copy()
        snapshot_plan["snapshot_idx"] = snapshot_plan["snapshot_idx"].astype(int)
        snapshot_plan["start_block"] = snapshot_plan["start_block"].astype(int)
        snapshot_plan["end_block"] = snapshot_plan["end_block"].astype(int)
        snapshot_plan = snapshot_plan.sort_values("snapshot_idx", kind="mergesort")
        n_snapshots_plan = int(snapshot_plan["snapshot_idx"].nunique())

        s_unique = snapshot_plan["start_block"].unique().tolist()
        if len(s_unique) != 1:
            raise ValueError("snapshot_plan start_block is not unique across snapshots; cannot use fixed S")
        S = int(s_unique[0])

        author_end_block = badrank.get_author_end_block()
        files, main_top50_df, normal_top50_df = badrank.select_input_files_same_as_wash2(end_block=author_end_block)
        n_input_files_total = int(len(files))
        n_input_files_main_top50 = int(len(main_top50_df))
        n_input_files_normal_top50 = int(len(normal_top50_df))

        df_all = trp.load_events_from_files(files)
        df_all = df_all[
            (pd.to_numeric(df_all["block_number"], errors="coerce") >= S)
            & (pd.to_numeric(df_all["block_number"], errors="coerce") <= int(snapshot_plan["end_block"].max()))
        ].copy()

        periods_df = badrank.build_all_holding_periods(
            df_all=df_all,
            start_block=S,
            max_end_block=int(snapshot_plan["end_block"].max()),
        )
        appearances_df = badrank.build_user_event_appearances(
            df_all=df_all,
            start_block=S,
            max_end_block=int(snapshot_plan["end_block"].max()),
        )
        buy_events_df = build_user_buy_events(
            df_all=df_all,
            start_block=S,
            max_end_block=int(snapshot_plan["end_block"].max()),
        )

        joint_rows = []
        for r in snapshot_plan.itertuples(index=False):
            snap = int(r.snapshot_idx)
            endb = int(r.end_block)

            user_holding_df = badrank.compute_user_holding_blocks(
                periods_df=periods_df,
                appearances_df=appearances_df,
                start_block=S,
                end_block_t=endb,
            )
            user_invest_df = compute_user_investment(
                buy_events_df=buy_events_df,
                start_block=S,
                end_block_t=endb,
            )
            cur_bin = user_bin_df[user_bin_df["snapshot_idx"] == snap].copy()
            if cur_bin.empty or user_holding_df.empty:
                continue

            merged = cur_bin.merge(user_holding_df, on="user", how="inner")
            merged = merged[merged["holding_obs_count"] > 0].copy()
            if merged.empty:
                continue

            merged = merged.merge(user_invest_df, on="user", how="left")
            merged["buy_total_usd"] = pd.to_numeric(merged["buy_total_usd"], errors="coerce").fillna(0.0)
            merged["buy_count"] = pd.to_numeric(merged["buy_count"], errors="coerce").fillna(0).astype(int)
            merged["buy_avg_usd"] = pd.to_numeric(merged["buy_avg_usd"], errors="coerce").fillna(0.0)
            merged["buy_events_per_10k_blocks"] = pd.to_numeric(
                merged["buy_events_per_10k_blocks"], errors="coerce"
            ).fillna(0.0)

            merged["snapshot_idx"] = snap
            merged["end_block"] = endb
            merged["holding_days"] = merged["holding_blocks_user_median"].astype(float) / float(badrank.BLOCKS_PER_DAY)
            merged["log_holding_days"] = np.log1p(merged["holding_days"].clip(lower=0.0))
            merged["log_buy_total_usd"] = np.log1p(merged["buy_total_usd"].clip(lower=0.0))
            merged["interaction"] = merged["log_holding_days"] * merged["log_buy_total_usd"]
            joint_rows.append(
                merged[
                    [
                        "snapshot_idx",
                        "start_block",
                        "end_block",
                        "user",
                        "rank_pct",
                        "rank_bin",
                        "holding_blocks_user_median",
                        "holding_days",
                        "holding_obs_count",
                        "short_hold_ratio",
                        "events_per_10k_blocks",
                        "speculator_score",
                        "buy_total_usd",
                        "buy_count",
                        "buy_avg_usd",
                        "buy_events_per_10k_blocks",
                        "log_holding_days",
                        "log_buy_total_usd",
                        "interaction",
                    ]
                ].copy()
            )

        joint_df = (
            pd.concat(joint_rows, ignore_index=True)
            if joint_rows
            else pd.DataFrame(
                columns=[
                    "snapshot_idx",
                    "start_block",
                    "end_block",
                    "user",
                    "rank_pct",
                    "rank_bin",
                    "holding_blocks_user_median",
                    "holding_days",
                    "holding_obs_count",
                    "short_hold_ratio",
                    "events_per_10k_blocks",
                    "speculator_score",
                    "buy_total_usd",
                    "buy_count",
                    "buy_avg_usd",
                    "buy_events_per_10k_blocks",
                    "log_holding_days",
                    "log_buy_total_usd",
                    "interaction",
                ]
            )
        )

    if "holding_days" not in joint_df.columns:
        raise ValueError("joint_df missing required column: holding_days")
    if "buy_total_usd" not in joint_df.columns:
        raise ValueError("joint_df missing required column: buy_total_usd")
    if "rank_pct" not in joint_df.columns:
        raise ValueError("joint_df missing required column: rank_pct")
    if "snapshot_idx" in joint_df.columns:
        joint_df["snapshot_idx"] = pd.to_numeric(joint_df["snapshot_idx"], errors="coerce").astype("Int64")
    joint_df["holding_days"] = pd.to_numeric(joint_df["holding_days"], errors="coerce")
    joint_df["buy_total_usd"] = pd.to_numeric(joint_df["buy_total_usd"], errors="coerce")
    joint_df["rank_pct"] = pd.to_numeric(joint_df["rank_pct"], errors="coerce")
    if "log_holding_days" not in joint_df.columns:
        joint_df["log_holding_days"] = np.log1p(joint_df["holding_days"].clip(lower=0.0))
    else:
        joint_df["log_holding_days"] = pd.to_numeric(joint_df["log_holding_days"], errors="coerce")
    if "log_buy_total_usd" not in joint_df.columns:
        joint_df["log_buy_total_usd"] = np.log1p(joint_df["buy_total_usd"].clip(lower=0.0))
    else:
        joint_df["log_buy_total_usd"] = pd.to_numeric(joint_df["log_buy_total_usd"], errors="coerce")
    if "interaction" not in joint_df.columns:
        joint_df["interaction"] = joint_df["log_holding_days"] * joint_df["log_buy_total_usd"]
    else:
        joint_df["interaction"] = pd.to_numeric(joint_df["interaction"], errors="coerce")

    summary_df = summarize_joint_bins(joint_df)
    reg_df = ols_fit(joint_df)

    heat_all_val, heat_all_cnt = make_heatmap_table(
        joint_df,
        value_col="rank_pct",
        count_col_name="n_users",
    )
    plot_heatmap(
        heat_all_val,
        heat_all_cnt,
        OUTPUT_DIR / "heatmap_mean_rank_pct_all_snapshots.png",
        "Mean rank percentile by holding and investment quantiles",
        total_users=int(np.nansum(heat_all_cnt.values)) if not heat_all_cnt.empty else 0,
    )

    rep_df = joint_df[joint_df["snapshot_idx"] == int(REPRESENTATIVE_SNAPSHOT_IDX)].copy()
    heat_rep_val, heat_rep_cnt = make_heatmap_table(
        rep_df,
        value_col="rank_pct",
        count_col_name="n_users",
    )
    plot_heatmap(
        heat_rep_val,
        heat_rep_cnt,
        OUTPUT_DIR / f"heatmap_mean_rank_pct_snapshot_{REPRESENTATIVE_SNAPSHOT_IDX}.png",
        f"Mean Rank %: Holding vs Investment (Snapshot {REPRESENTATIVE_SNAPSHOT_IDX})",
        total_users=int(np.nansum(heat_rep_cnt.values)) if not heat_rep_cnt.empty else 0,
    )
    holding_seg_df = build_holding_days_segment_table(joint_df["holding_days"])
    plot_holding_days_segment_bar(
        holding_seg_df,
        OUTPUT_DIR / "holding_days_segment_distribution_all_snapshots.png",
        "Holding Duration Segment Distribution (All Snapshots)",
    )

    joint_fp = OUTPUT_DIR / "joint_user_hold_invest_snapshot_level.csv"
    summary_fp = OUTPUT_DIR / "joint_hold_invest_quantile_summary.csv"
    reg_fp = OUTPUT_DIR / "ols_rank_pct_hold_invest_interaction.csv"
    holding_seg_fp = OUTPUT_DIR / "holding_days_segment_distribution.csv"
    meta_fp = OUTPUT_DIR / "run_meta.csv"

    joint_df.to_csv(joint_fp, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_fp, index=False, encoding="utf-8-sig")
    reg_df.to_csv(reg_fp, index=False, encoding="utf-8-sig")
    holding_seg_df.to_csv(holding_seg_fp, index=False, encoding="utf-8-sig")

    rep_n = int(len(rep_df))
    rep_rank_min = float(rep_df["rank_pct"].min()) if not rep_df.empty else float("nan")
    rep_rank_max = float(rep_df["rank_pct"].max()) if not rep_df.empty else float("nan")

    t1 = perf_counter()
    run_meta = pd.DataFrame(
        [
            {"key": "representative_snapshot_idx", "value": int(REPRESENTATIVE_SNAPSHOT_IDX)},
            {"key": "quantile_bins", "value": int(QUANTILE_BINS)},
            {"key": "window_mode", "value": "cumulative_S_to_end_block"},
            {"key": "run_mode", "value": run_mode},
            {"key": "source_joint_csv", "value": source_joint_csv},
            {"key": "holding_metric", "value": "holding_blocks_user_median"},
            {"key": "investment_metric", "value": "buy_total_usd_from_price_usd"},
            {"key": "regression_formula", "value": "rank_pct ~ log1p(holding_days) + log1p(buy_total_usd) + interaction"},
            {"key": "start_block_S", "value": int(S) if S is not None else ""},
            {"key": "author_end_block_for_selection", "value": int(author_end_block) if author_end_block is not None else ""},
            {"key": "n_input_files_total", "value": int(n_input_files_total) if n_input_files_total is not None else ""},
            {"key": "n_input_files_main_top50", "value": int(n_input_files_main_top50) if n_input_files_main_top50 is not None else ""},
            {"key": "n_input_files_normal_top50", "value": int(n_input_files_normal_top50) if n_input_files_normal_top50 is not None else ""},
            {"key": "n_snapshots_plan", "value": int(n_snapshots_plan) if n_snapshots_plan is not None else ""},
            {"key": "n_joint_rows", "value": int(len(joint_df))},
            {"key": "n_summary_rows", "value": int(len(summary_df))},
            {"key": "n_regression_rows", "value": int(len(reg_df))},
            {"key": "n_holding_segment_rows", "value": int(len(holding_seg_df))},
            {"key": "n_rep_snapshot_rows", "value": rep_n},
            {"key": "rep_snapshot_rank_pct_min", "value": rep_rank_min},
            {"key": "rep_snapshot_rank_pct_max", "value": rep_rank_max},
            {"key": "accept_heatmap_all_nonempty", "value": int(0 if heat_all_val.empty else 1)},
            {"key": "accept_heatmap_rep_nonempty", "value": int(0 if heat_rep_val.empty else 1)},
            {"key": "accept_holding_segment_nonempty", "value": int(0 if holding_seg_df.empty else 1)},
            {"key": "input_snapshot_run_meta_exists", "value": int(1 if not run_meta_prev.empty else 0)},
            {"key": "elapsed_seconds", "value": round(t1 - t0, 3)},
        ]
    )
    run_meta.to_csv(meta_fp, index=False, encoding="utf-8-sig")

    print(f"[Saved] {joint_fp.resolve()}")
    print(f"[Saved] {summary_fp.resolve()}")
    print(f"[Saved] {reg_fp.resolve()}")
    print(f"[Saved] {holding_seg_fp.resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'heatmap_mean_rank_pct_all_snapshots.png').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / f'heatmap_mean_rank_pct_snapshot_{REPRESENTATIVE_SNAPSHOT_IDX}.png').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'holding_days_segment_distribution_all_snapshots.png').resolve()}")
    print(f"[Saved] {meta_fp.resolve()}")


if __name__ == "__main__":
    main()
