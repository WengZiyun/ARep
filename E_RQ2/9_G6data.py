from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "BASE_CLEAN_CSV": ROOT / "output" / "A" / "18_RQ2compare_G6_v1" / "_base_clean" / "RQdataG6.csv",
    "OUTDIR": ROOT / "output" / "E" / "9_G6data_analysis_v2",
    "TARGET_MONTH": "2025-12",
    "RANDOM_SEED": 11,
    "CLUSTER_SAMPLE_SIZE": 8000,
    "N_CLUSTERS": 5,
    "CLUSTER_MAX_ITER": 40,
    "CLUSTER_OUTLIER_Q": 0.99,
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def load_base_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base clean CSV not found: {path}")
    df = pd.read_csv(path)
    if "calendar_year_month" not in df.columns and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["calendar_year_month"] = ts.dt.to_period("M").astype(str)
    return df


def build_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    trades = df[df["tx_type"].astype(str).eq("trade")].copy()
    trades["price_num"] = pd.to_numeric(trades["price"], errors="coerce").fillna(0.0)
    trades["gas_num"] = pd.to_numeric(trades["gas"], errors="coerce").fillna(0.0)
    trades["log_price"] = np.log1p(trades["price_num"].clip(lower=0.0))
    trades["log_gas"] = np.log1p(trades["gas_num"].clip(lower=0.0))
    trades["pair_key"] = (
        trades["seller"].astype(str) + "->" + trades["buyer"].astype(str) + "|" + trades["contract_id"].astype(str)
    )
    trades["pair_count"] = trades.groupby("pair_key")["pair_key"].transform("count").astype(float)
    trades["contract_activity"] = trades.groupby("contract_id")["contract_id"].transform("count").astype(float)
    trades["buyer_activity"] = trades.groupby("buyer")["buyer"].transform("count").astype(float)
    trades["seller_activity"] = trades.groupby("seller")["seller"].transform("count").astype(float)
    trades["log_pair_count"] = np.log1p(trades["pair_count"])
    trades["log_contract_activity"] = np.log1p(trades["contract_activity"])
    trades["log_buyer_activity"] = np.log1p(trades["buyer_activity"])
    trades["log_seller_activity"] = np.log1p(trades["seller_activity"])
    return trades.reset_index(drop=True)


def _hist_prob(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, density=False)
    p = hist.astype(float) + 1e-12
    return p / p.sum()


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def compute_monthly_js_divergence(trades: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    months = sorted(trades["calendar_year_month"].astype(str).dropna().unique().tolist())
    for param in params:
        global_vals = pd.to_numeric(trades[param], errors="coerce").dropna().to_numpy(dtype=float)
        if len(global_vals) < 20:
            continue
        lo = float(np.nanpercentile(global_vals, 1))
        hi = float(np.nanpercentile(global_vals, 99))
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 50)
        p_global = _hist_prob(np.clip(global_vals, lo, hi), bins)
        for ym in months:
            local_vals = pd.to_numeric(
                trades.loc[trades["calendar_year_month"].astype(str).eq(ym), param], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            if len(local_vals) < 20:
                continue
            p_local = _hist_prob(np.clip(local_vals, lo, hi), bins)
            records.append(
                {
                    "calendar_year_month": ym,
                    "param": param,
                    "js_divergence": _js_divergence(p_local, p_global),
                    "n_local": int(len(local_vals)),
                }
            )
    return pd.DataFrame(records)


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(x, axis=0, keepdims=True)
    sigma = np.where(sigma <= 1e-12, 1.0, sigma)
    return (x - mu) / sigma


def _kmeans_numpy(x: np.ndarray, n_clusters: int, max_iter: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    init_idx = rng.choice(n, size=min(n_clusters, n), replace=False)
    centers = x[init_idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(centers.shape[0]):
            sub = x[labels == k]
            if len(sub) > 0:
                centers[k] = sub.mean(axis=0)
    return labels, centers


def run_clustering(trades: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    target_month = str(cfg["TARGET_MONTH"])
    sub = trades[trades["calendar_year_month"].astype(str).eq(target_month)].copy()
    if sub.empty:
        return pd.DataFrame()
    sample_n = min(int(cfg["CLUSTER_SAMPLE_SIZE"]), len(sub))
    sub = sub.sample(n=sample_n, random_state=int(cfg["RANDOM_SEED"])).copy()
    feats = sub[["log_price", "log_gas", "log_pair_count", "log_contract_activity"]].to_numpy(dtype=float)
    feats_std = _standardize(feats)
    labels, centers = _kmeans_numpy(
        feats_std,
        n_clusters=int(cfg["N_CLUSTERS"]),
        max_iter=int(cfg["CLUSTER_MAX_ITER"]),
        seed=int(cfg["RANDOM_SEED"]),
    )
    d2 = np.sum((feats_std - centers[labels]) ** 2, axis=1)
    sub["cluster_id"] = labels.astype(int)
    sub["cluster_distance"] = d2.astype(float)
    thr = float(np.quantile(sub["cluster_distance"], float(cfg["CLUSTER_OUTLIER_Q"])))
    sub["cluster_outlier"] = sub["cluster_distance"] >= thr
    return sub


def compute_parameter_potential(
    trades: pd.DataFrame,
    js_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    target_month = str(cfg["TARGET_MONTH"])
    params = [
        "log_price",
        "log_gas",
        "log_pair_count",
        "log_contract_activity",
        "log_buyer_activity",
        "log_seller_activity",
    ]
    records: list[dict[str, Any]] = []
    for p in params:
        x = pd.to_numeric(trades[p], errors="coerce").dropna().astype(float)
        if len(x) < 20:
            continue
        p50 = float(np.percentile(x, 50))
        p95 = float(np.percentile(x, 95))
        spread_ratio = (p95 - p50) / max(abs(p50), 1e-6)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        if mad <= 1e-12:
            outlier_rate = 0.0
        else:
            z = 0.67448975 * (x - med) / mad
            outlier_rate = float(np.mean(np.abs(z) >= 3.5))
        target_js_rows = js_df[js_df["param"].astype(str).eq(p) & js_df["calendar_year_month"].astype(str).eq(target_month)]
        target_js = float(target_js_rows["js_divergence"].iloc[0]) if len(target_js_rows) else 0.0
        if p == "log_gas" and len(cluster_df):
            outlier_vals = cluster_df.loc[cluster_df["cluster_outlier"], "log_gas"]
            base_vals = cluster_df["log_gas"]
            cluster_sep = float(outlier_vals.mean() - base_vals.mean()) if len(outlier_vals) else 0.0
        else:
            cluster_sep = 0.0
        records.append(
            {
                "param": p,
                "spread_ratio": spread_ratio,
                "outlier_rate": outlier_rate,
                "target_month_js": target_js,
                "cluster_sep": cluster_sep,
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    for c in ["spread_ratio", "outlier_rate", "target_month_js", "cluster_sep"]:
        s = out[c].astype(float)
        lo, hi = float(s.min()), float(s.max())
        out[f"{c}_n"] = 0.0 if hi <= lo else (s - lo) / (hi - lo)
    out["potential_score"] = (
        0.35 * out["target_month_js_n"]
        + 0.25 * out["spread_ratio_n"]
        + 0.25 * out["outlier_rate_n"]
        + 0.15 * out["cluster_sep_n"]
    )
    return out.sort_values("potential_score", ascending=False).reset_index(drop=True)


def plot_monthly_js(js_df: pd.DataFrame, outdir: Path) -> None:
    if js_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    months = sorted(js_df["calendar_year_month"].astype(str).unique().tolist())
    x_map = {m: i for i, m in enumerate(months)}
    for param, sub in js_df.groupby("param"):
        sub = sub.sort_values("calendar_year_month")
        x = [x_map[m] for m in sub["calendar_year_month"].astype(str).tolist()]
        y = sub["js_divergence"].astype(float).to_numpy()
        ax.plot(x, y, marker="o", linewidth=2.0, label=str(param))
    ax.set_xticks(list(x_map.values()))
    ax.set_xticklabels(list(x_map.keys()), rotation=45, ha="right")
    ax.set_ylabel("JS Divergence vs Global")
    ax.set_xlabel("Month")
    ax.set_title("Monthly Distribution Divergence by Parameter")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "g6_monthly_js_divergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_scatter(cluster_df: pd.DataFrame, outdir: Path) -> None:
    if cluster_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        cluster_df["log_price"],
        cluster_df["log_gas"],
        c=cluster_df["cluster_id"],
        s=14,
        alpha=0.6,
        cmap="tab10",
    )
    out = cluster_df[cluster_df["cluster_outlier"]]
    if len(out):
        ax.scatter(
            out["log_price"],
            out["log_gas"],
            s=34,
            facecolors="none",
            edgecolors="red",
            linewidths=0.9,
            label="Cluster Outlier (top 1%)",
        )
    ax.set_xlabel("log1p(price)")
    ax.set_ylabel("log1p(gas)")
    ax.set_title("Target-Month Trade Clusters in (price, gas) Space")
    ax.grid(True, alpha=0.25)
    if len(out):
        ax.legend(loc="best")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("cluster_id")
    fig.tight_layout()
    fig.savefig(outdir / "g6_target_month_cluster_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_potential(param_df: pd.DataFrame, outdir: Path) -> None:
    if param_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(param_df), dtype=float)
    ax.bar(x, param_df["potential_score"].astype(float).to_numpy(), color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(param_df["param"].astype(str).tolist(), rotation=30, ha="right")
    ax.set_ylabel("Potential Score")
    ax.set_title("Parameter Potential for Singularity Injection")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "g6_parameter_potential_score.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analysis(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)
    outdir = Path(cfg["OUTDIR"])
    _ensure_dir(outdir)

    raw = load_base_clean(Path(cfg["BASE_CLEAN_CSV"]))
    trades = build_trade_features(raw)
    params = ["log_price", "log_gas", "log_pair_count", "log_contract_activity", "log_buyer_activity", "log_seller_activity"]
    js_df = compute_monthly_js_divergence(trades, params=params)
    cluster_df = run_clustering(trades, cfg)
    potential_df = compute_parameter_potential(trades, js_df, cluster_df, cfg)

    js_df.to_csv(outdir / "g6_monthly_js_divergence.csv", index=False)
    cluster_df.to_csv(outdir / "g6_target_month_cluster_points.csv", index=False)
    potential_df.to_csv(outdir / "g6_parameter_potential.csv", index=False)
    if len(cluster_df):
        cluster_df[cluster_df["cluster_outlier"]].sort_values("cluster_distance", ascending=False).head(500).to_csv(
            outdir / "g6_target_month_cluster_outliers_top500.csv", index=False
        )

    plot_monthly_js(js_df, outdir)
    plot_cluster_scatter(cluster_df, outdir)
    plot_parameter_potential(potential_df, outdir)

    summary = {
        "target_month": str(cfg["TARGET_MONTH"]),
        "n_trade_rows": int(len(trades)),
        "n_js_points": int(len(js_df)),
        "n_cluster_points": int(len(cluster_df)),
        "n_cluster_outliers": int(cluster_df["cluster_outlier"].sum()) if len(cluster_df) else 0,
        "top_parameter": potential_df.head(1).to_dict(orient="records"),
        "output_dir": str(outdir),
    }
    _write_json(outdir / "g6_visual_analysis_summary.json", summary)
    return summary


if __name__ == "__main__":
    summary_obj = run_analysis()
    print(
        f"[G6 data] done | trades={summary_obj['n_trade_rows']} | cluster_points={summary_obj['n_cluster_points']} | "
        f"cluster_outliers={summary_obj['n_cluster_outliers']}",
        flush=True,
    )
