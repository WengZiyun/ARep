
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from math import ceil
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Config (edit here)
# ============================================================
SCORES_CSV = Path("output/D/2_eval_counterfeit/snapshot_contract_scores.csv")
PLAN_CSV = Path("output/D/2_eval_counterfeit/snapshot_plan.csv")
LABEL_CSV = Path("output/C/3_collectionlabel.csv")
UNIVERSE_CSV = Path("output/D/2_eval_counterfeit/selected_universe.csv")  # optional filter
OUTPUT_DIR = Path("output/D/2_eval_counterfeit_plt")

TOP_PERCENT_LIST = [0.10, 0.20, 0.30]
DELTA_WINDOWS = list(range(0, 61))
POSITIVE_TYPE = "counterfeit"
USE_BOTTOM_RANK = True
FIG_DPI = 140


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def require_columns(df: pd.DataFrame, need: list[str], name: str) -> None:
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {name}: {miss}")


def normalize_addr(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.strip()


def average_precision_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, str]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=float)
    pos = int(y.sum())
    n = int(y.size)
    if n == 0:
        return float("nan"), "empty_candidates"
    if pos == 0:
        return float("nan"), "all_negative"
    if pos == n:
        return float("nan"), "all_positive"

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / (tp + fp)
    recall = tp / float(pos)

    ap = 0.0
    prev_recall = 0.0
    for i in range(n):
        if y_sorted[i] == 1:
            ap += precision[i] * (recall[i] - prev_recall)
            prev_recall = recall[i]
    return float(ap), ""


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str] | None]:
    ensure_exists(SCORES_CSV)
    ensure_exists(PLAN_CSV)
    ensure_exists(LABEL_CSV)

    scores = pd.read_csv(SCORES_CSV, low_memory=False)
    plan = pd.read_csv(PLAN_CSV, low_memory=False)
    label = pd.read_csv(LABEL_CSV, low_memory=False)

    require_columns(
        scores,
        ["snapshot_idx", "start_block", "end_block", "contract_address", "collection_type", "rank_C", "n_contracts_snapshot"],
        "snapshot_contract_scores.csv",
    )
    require_columns(plan, ["snapshot_idx", "start_block", "end_block"], "snapshot_plan.csv")
    require_columns(label, ["contract_address", "collection_type", "create_block_number", "detected_block_number"], "3_collectionlabel.csv")

    scores = scores.copy()
    scores["snapshot_idx"] = pd.to_numeric(scores["snapshot_idx"], errors="coerce").astype("Int64")
    scores["start_block"] = pd.to_numeric(scores["start_block"], errors="coerce").astype("Int64")
    scores["end_block"] = pd.to_numeric(scores["end_block"], errors="coerce").astype("Int64")
    scores["rank_C"] = pd.to_numeric(scores["rank_C"], errors="coerce")
    scores["n_contracts_snapshot"] = pd.to_numeric(scores["n_contracts_snapshot"], errors="coerce")
    scores["contract_address"] = normalize_addr(scores["contract_address"])
    scores["collection_type"] = normalize_addr(scores["collection_type"])
    scores = scores.dropna(subset=["snapshot_idx", "start_block", "end_block", "rank_C", "n_contracts_snapshot"]).copy()
    scores["snapshot_idx"] = scores["snapshot_idx"].astype(int)
    scores["start_block"] = scores["start_block"].astype(int)
    scores["end_block"] = scores["end_block"].astype(int)

    plan = plan.copy()
    plan["snapshot_idx"] = pd.to_numeric(plan["snapshot_idx"], errors="coerce").astype("Int64")
    plan["start_block"] = pd.to_numeric(plan["start_block"], errors="coerce").astype("Int64")
    plan["end_block"] = pd.to_numeric(plan["end_block"], errors="coerce").astype("Int64")
    plan = plan.dropna(subset=["snapshot_idx", "start_block", "end_block"]).copy()
    plan["snapshot_idx"] = plan["snapshot_idx"].astype(int)
    plan["start_block"] = plan["start_block"].astype(int)
    plan["end_block"] = plan["end_block"].astype(int)
    plan = plan.sort_values("snapshot_idx", kind="mergesort").drop_duplicates("snapshot_idx", keep="last")

    label = label.copy()
    label["contract_address"] = normalize_addr(label["contract_address"])
    label["collection_type"] = normalize_addr(label["collection_type"])
    label["create_block_number"] = pd.to_numeric(label["create_block_number"], errors="coerce")
    label["detected_block_number"] = pd.to_numeric(label["detected_block_number"], errors="coerce")

    universe_set = None
    if UNIVERSE_CSV.exists():
        uni = pd.read_csv(UNIVERSE_CSV, low_memory=False)
        if "contract_address" in uni.columns:
            universe_set = set(normalize_addr(uni["contract_address"]).tolist())

    return scores, plan, label, universe_set


def build_anchor_map(plan: pd.DataFrame, label: pd.DataFrame, universe_set: set[str] | None) -> tuple[dict[int, set[str]], dict]:
    positives = label[label["collection_type"] == POSITIVE_TYPE].copy()
    create_missing = int(positives["create_block_number"].isna().sum())
    detect_missing = int(positives["detected_block_number"].isna().sum())
    positives = positives.dropna(subset=["detected_block_number"]).copy()
    positives["detected_block_number"] = positives["detected_block_number"].astype(np.int64)
    if universe_set is not None:
        positives = positives[positives["contract_address"].isin(universe_set)].copy()

    positives = positives.sort_values(["contract_address", "detected_block_number"], kind="mergesort")
    positives = positives.drop_duplicates("contract_address", keep="first")

    anchor_map: dict[int, set[str]] = {}
    not_mapped_detect = 0
    not_mapped_create = 0
    lifespan_windows: list[int] = []
    for r in positives.itertuples(index=False):
        addr = str(r.contract_address)
        db = int(r.detected_block_number)
        detect_hit = plan[(plan["start_block"] <= db) & (plan["end_block"] >= db)]
        if detect_hit.empty:
            not_mapped_detect += 1
            continue
        detect_snapshot = int(detect_hit.iloc[0]["snapshot_idx"])
        anchor_map.setdefault(detect_snapshot, set()).add(addr)

        cb = pd.to_numeric(getattr(r, "create_block_number", np.nan), errors="coerce")
        if pd.isna(cb):
            continue
        create_hit = plan[(plan["start_block"] <= int(cb)) & (plan["end_block"] >= int(cb))]
        if create_hit.empty:
            not_mapped_create += 1
            continue
        create_snapshot = int(create_hit.iloc[0]["snapshot_idx"])
        lifespan_windows.append(max(0, detect_snapshot - create_snapshot))

    max_lifespan_delta = int(max(lifespan_windows)) if lifespan_windows else 0

    meta = {
        "positive_create_missing": create_missing,
        "positive_detect_missing": detect_missing,
        "positive_anchor_mapped": int(sum(len(v) for v in anchor_map.values())),
        "positive_not_mapped_detect_window": int(not_mapped_detect),
        "positive_not_mapped_create_window": int(not_mapped_create),
        "positive_lifespan_pairs_mapped": int(len(lifespan_windows)),
        "max_lifespan_delta_windows": max_lifespan_delta,
        "n_anchor_windows": int(len(anchor_map)),
    }
    return anchor_map, meta


def evaluate(
    scores: pd.DataFrame,
    plan: pd.DataFrame,
    anchor_map: dict[int, set[str]],
    delta_windows: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_snapshot = {
        int(k): v.copy()
        for k, v in scores.sort_values(["snapshot_idx", "rank_C"], kind="mergesort").groupby("snapshot_idx", sort=True)
    }
    bounds = {int(r.snapshot_idx): (int(r.start_block), int(r.end_block)) for r in plan.itertuples(index=False)}
    n_max = int(plan["snapshot_idx"].max()) if not plan.empty else 0

    rows_metrics = []
    rows_auprc = []

    for w in sorted(anchor_map.keys()):
        m_set = set(anchor_map.get(w, set()))
        m = int(len(m_set))
        if m <= 0:
            continue

        for delta in delta_windows:
            t = int(w - int(delta))
            if t < 1 or t > n_max:
                continue
            if t not in by_snapshot or t not in bounds:
                continue

            cur = by_snapshot[t].copy()
            if cur.empty:
                continue

            keep = (cur["collection_type"] != POSITIVE_TYPE) | (cur["contract_address"].isin(m_set))
            cur = cur[keep].copy()
            if cur.empty:
                continue

            n = int(len(cur))
            if USE_BOTTOM_RANK:
                cur = cur.sort_values(["rank_C", "contract_address"], ascending=[False, True], kind="mergesort")
            else:
                cur = cur.sort_values(["rank_C", "contract_address"], ascending=[True, True], kind="mergesort")
            cur = cur.reset_index(drop=True)

            cand_addrs = cur["contract_address"].tolist()
            cand_set = set(cand_addrs)
            pos_set = m_set.intersection(cand_set)
            n_pos = m
            n_pos_eval = int(len(pos_set))

            y_true = np.array([1 if a in m_set else 0 for a in cand_addrs], dtype=np.int64)
            risk_score = (cur["rank_C"] / cur["n_contracts_snapshot"].replace(0, np.nan)).fillna(0.0).astype(float).values
            auprc, reason = average_precision_score_binary(y_true, risk_score)

            rows_auprc.append(
                {
                    "anchor_snapshot_idx": int(w),
                    "lookback_snapshot_idx": int(t),
                    "delta_windows": int(delta),
                    "anchor_start_block": int(bounds[w][0]),
                    "anchor_end_block": int(bounds[w][1]),
                    "lookback_start_block": int(bounds[t][0]),
                    "lookback_end_block": int(bounds[t][1]),
                    "n_candidates": int(n),
                    "n_positives": int(n_pos),
                    "n_positives_eval_candidates": int(n_pos_eval),
                    "auprc": auprc,
                    "auprc_reason": reason,
                }
            )

            for p in TOP_PERCENT_LIST:
                k = int(ceil(float(p) * float(n)))
                k = max(1, min(k, n))
                topk = set(cand_addrs[:k])
                hit = int(len(topk.intersection(pos_set)))
                recall_strict = float(hit / n_pos) if n_pos > 0 else np.nan
                recall_detectable = float(hit / n_pos_eval) if n_pos_eval > 0 else np.nan
                precision_strict = float(hit / k)
                precision_detectable = float(hit / k) if n_pos_eval > 0 else np.nan

                rows_metrics.append(
                    {
                        "anchor_snapshot_idx": int(w),
                        "lookback_snapshot_idx": int(t),
                        "delta_windows": int(delta),
                        "top_percent": float(p),
                        "anchor_start_block": int(bounds[w][0]),
                        "anchor_end_block": int(bounds[w][1]),
                        "lookback_start_block": int(bounds[t][0]),
                        "lookback_end_block": int(bounds[t][1]),
                        "n_candidates": int(n),
                        "k": int(k),
                        "n_positives_anchor_window": int(n_pos),
                        "n_positives_eval_candidates": int(n_pos_eval),
                        "hits": int(hit),
                        "recall_at_k": recall_detectable,
                        "recall_at_k_detectable": recall_detectable,
                        "recall_at_k_strict": recall_strict,
                        "precision_at_k": precision_detectable,
                        "precision_at_k_detectable": precision_detectable,
                        "precision_at_k_strict": precision_strict,
                    }
                )

    return pd.DataFrame(rows_metrics), pd.DataFrame(rows_auprc)


def plot_metrics(metrics_mean: pd.DataFrame, auprc_mean: pd.DataFrame, delta_windows: list[int]) -> None:
    x_min = int(min(delta_windows)) if delta_windows else 0
    x_max = int(max(delta_windows)) if delta_windows else 0

    if not metrics_mean.empty:
        plt.figure(figsize=(8.8, 4.8))
        for p in TOP_PERCENT_LIST:
            sub = metrics_mean[metrics_mean["top_percent"] == float(p)].sort_values("delta_windows")
            if sub.empty:
                continue
            plt.plot(sub["delta_windows"], sub["recall_mean"], marker="o", linewidth=2.0, label=f"Top{int(round(p*100))}%")
        plt.title("Recall@K vs Delta (Forward Label)")
        plt.xlabel("Delta (windows)")
        plt.ylabel("Mean Detectable Recall@K")
        plt.ylim(0, 1)
        plt.xlim(x_min, x_max)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "recall_at_k_vs_delta.png", dpi=FIG_DPI)
        plt.close()

        plt.figure(figsize=(8.8, 4.8))
        for p in TOP_PERCENT_LIST:
            sub = metrics_mean[metrics_mean["top_percent"] == float(p)].sort_values("delta_windows")
            if sub.empty:
                continue
            plt.plot(sub["delta_windows"], sub["precision_mean"], marker="o", linewidth=2.0, label=f"Top{int(round(p*100))}%")
        plt.title("Precision@K vs Delta (Forward Label)")
        plt.xlabel("Delta (windows)")
        plt.ylabel("Mean Detectable Precision@K")
        plt.ylim(0, 1)
        plt.xlim(x_min, x_max)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "precision_at_k_vs_delta.png", dpi=FIG_DPI)
        plt.close()

    if not auprc_mean.empty:
        sub = auprc_mean.sort_values("delta_windows")
        x = sub["delta_windows"].values
        y = sub["auprc_mean"].values
        ys = sub["auprc_std"].fillna(0.0).values
        low = np.maximum(0.0, y - ys)
        high = np.minimum(1.0, y + ys)

        plt.figure(figsize=(8.8, 4.8))
        plt.plot(x, y, marker="o", linewidth=2.0, label="Mean AUPRC")
        plt.fill_between(x, low, high, alpha=0.2, label="±1 std")
        plt.title("AUPRC vs Delta (Forward Label)")
        plt.xlabel("Delta (windows)")
        plt.ylabel("Mean AUPRC")
        plt.ylim(0, 1)
        plt.xlim(x_min, x_max)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "auprc_vs_delta.png", dpi=FIG_DPI)
        plt.close()


def main() -> None:
    t0 = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores, plan, label, universe_set = load_inputs()
    anchor_map, anchor_meta = build_anchor_map(plan=plan, label=label, universe_set=universe_set)
    delta_windows = DELTA_WINDOWS

    metrics_df, auprc_df = evaluate(scores=scores, plan=plan, anchor_map=anchor_map, delta_windows=delta_windows)
    if metrics_df.empty:
        raise ValueError("No metric rows generated. Check inputs and anchor mapping.")

    metrics_mean = (
        metrics_df.groupby(["delta_windows", "top_percent"], as_index=False)
        .agg(
            recall_mean=("recall_at_k_detectable", "mean"),
            recall_count_non_nan=("recall_at_k_detectable", lambda s: int(pd.Series(s).notna().sum())),
            recall_strict_mean=("recall_at_k_strict", "mean"),
            recall_strict_count_non_nan=("recall_at_k_strict", lambda s: int(pd.Series(s).notna().sum())),
            precision_mean=("precision_at_k_detectable", "mean"),
            precision_count_non_nan=("precision_at_k_detectable", lambda s: int(pd.Series(s).notna().sum())),
            precision_strict_mean=("precision_at_k_strict", "mean"),
            precision_strict_count=("precision_at_k_strict", "count"),
            n_anchor_windows=("anchor_snapshot_idx", "nunique"),
        )
        .sort_values(["delta_windows", "top_percent"], kind="mergesort")
        .reset_index(drop=True)
    )

    if auprc_df.empty:
        auprc_mean = pd.DataFrame(columns=["delta_windows", "auprc_mean", "auprc_std", "auprc_count_non_nan", "n_anchor_windows"])
    else:
        auprc_mean = (
            auprc_df.groupby(["delta_windows"], as_index=False)
            .agg(
                auprc_mean=("auprc", "mean"),
                auprc_std=("auprc", "std"),
                auprc_count_non_nan=("auprc", lambda s: int(pd.Series(s).notna().sum())),
                n_anchor_windows=("anchor_snapshot_idx", "nunique"),
            )
            .sort_values(["delta_windows"], kind="mergesort")
            .reset_index(drop=True)
        )

    metrics_df.to_csv(OUTPUT_DIR / "metrics_window_level.csv", index=False, encoding="utf-8-sig")
    metrics_mean.to_csv(OUTPUT_DIR / "metrics_mean_by_delta.csv", index=False, encoding="utf-8-sig")
    auprc_df.to_csv(OUTPUT_DIR / "auprc_window_level.csv", index=False, encoding="utf-8-sig")
    auprc_mean.to_csv(OUTPUT_DIR / "auprc_mean_by_delta.csv", index=False, encoding="utf-8-sig")

    plot_metrics(metrics_mean=metrics_mean, auprc_mean=auprc_mean, delta_windows=delta_windows)

    t1 = perf_counter()
    run_meta = pd.DataFrame(
        [
            {"key": "evaluation_mode", "value": "forward_label_anchor_window"},
            {"key": "delta_windows", "value": "|".join(str(x) for x in delta_windows)},
            {"key": "max_delta_windows", "value": int(max(delta_windows) if delta_windows else 0)},
            {"key": "max_delta_source", "value": "fixed_0_to_60"},
            {"key": "top_percent_list", "value": "|".join(str(x) for x in TOP_PERCENT_LIST)},
            {"key": "recall_mean_mode", "value": "detectable_only_skip_no_positive_eval_candidates"},
            {"key": "precision_mean_mode", "value": "detectable_only_skip_no_positive_eval_candidates"},
            {"key": "positive_type", "value": POSITIVE_TYPE},
            {"key": "use_bottom_rank", "value": int(1 if USE_BOTTOM_RANK else 0)},
            {"key": "n_input_scores_rows", "value": int(len(scores))},
            {"key": "n_input_snapshots", "value": int(plan["snapshot_idx"].nunique())},
            {"key": "n_anchor_windows", "value": int(anchor_meta["n_anchor_windows"])},
            {"key": "n_anchor_positive_events", "value": int(anchor_meta["positive_anchor_mapped"])},
            {"key": "n_positive_create_missing", "value": int(anchor_meta["positive_create_missing"])},
            {"key": "n_positive_detect_missing", "value": int(anchor_meta["positive_detect_missing"])},
            {"key": "n_positive_not_mapped_detect_window", "value": int(anchor_meta["positive_not_mapped_detect_window"])},
            {"key": "n_positive_not_mapped_create_window", "value": int(anchor_meta["positive_not_mapped_create_window"])},
            {"key": "n_positive_lifespan_pairs_mapped", "value": int(anchor_meta["positive_lifespan_pairs_mapped"])},
            {"key": "n_metrics_rows", "value": int(len(metrics_df))},
            {"key": "n_auprc_rows", "value": int(len(auprc_df))},
            {"key": "elapsed_seconds", "value": round(t1 - t0, 3)},
        ]
    )
    run_meta.to_csv(OUTPUT_DIR / "run_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'metrics_window_level.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'metrics_mean_by_delta.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'auprc_window_level.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'auprc_mean_by_delta.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'run_meta.csv').resolve()}")
    print(f"[Saved] figures in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
