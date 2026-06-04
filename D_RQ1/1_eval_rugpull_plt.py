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
SCORES_CSV = Path("output/D/1_eval_rugpull3/snapshot_contract_scores.csv")
PLAN_CSV = Path("output/D/1_eval_rugpull3/snapshot_plan.csv")
LABEL_CSV = Path("output/C/3_collectionlabel.csv")
UNIVERSE_CSV = Path("output/D/1_eval_rugpull3/selected_universe.csv")
OUTPUT_DIR = Path("output/D/1_eval_rugpull_plt")

TOP_PERCENT_LIST = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
DELTA_WINDOWS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MAX_DELTA = 10
POSITIVE_TYPE = "rugpull"
USE_BOTTOM_RANK = True
FIG_DPI = 140
PRECISION_TARGET_LIST = [0.02, 0.05, 0.10]


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    for p in [SCORES_CSV, PLAN_CSV, LABEL_CSV, UNIVERSE_CSV]:
        ensure_exists(p)

    scores = pd.read_csv(SCORES_CSV, low_memory=False)
    plan = pd.read_csv(PLAN_CSV, low_memory=False)
    label = pd.read_csv(LABEL_CSV, low_memory=False)
    universe = pd.read_csv(UNIVERSE_CSV, low_memory=False)

    require_columns(
        scores,
        ["snapshot_idx", "end_block", "contract_address", "collection_type", "rank_C", "n_contracts_snapshot"],
        "snapshot_contract_scores.csv",
    )
    require_columns(plan, ["snapshot_idx", "start_block", "end_block"], "snapshot_plan.csv")
    require_columns(label, ["contract_address", "collection_type", "detected_block_number"], "3_collectionlabel.csv")
    require_columns(universe, ["contract_address"], "selected_universe.csv")

    scores = scores.copy()
    scores["contract_address"] = normalize_addr(scores["contract_address"])
    scores["collection_type"] = normalize_addr(scores["collection_type"])
    scores["snapshot_idx"] = pd.to_numeric(scores["snapshot_idx"], errors="coerce").astype("Int64")
    scores["end_block"] = pd.to_numeric(scores["end_block"], errors="coerce").astype("Int64")
    scores["rank_C"] = pd.to_numeric(scores["rank_C"], errors="coerce")
    scores["n_contracts_snapshot"] = pd.to_numeric(scores["n_contracts_snapshot"], errors="coerce")
    scores = scores.dropna(subset=["snapshot_idx", "end_block", "rank_C", "n_contracts_snapshot"]).copy()
    scores["snapshot_idx"] = scores["snapshot_idx"].astype(int)
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
    label["detected_block_number"] = pd.to_numeric(label["detected_block_number"], errors="coerce")

    universe_set = set(normalize_addr(universe["contract_address"]).tolist())
    return scores, plan, label, universe_set


def build_window_bounds(plan: pd.DataFrame) -> dict[int, tuple[int, int]]:
    bounds = {}
    for r in plan.itertuples(index=False):
        bounds[int(r.snapshot_idx)] = (int(r.start_block), int(r.end_block))
    return bounds


def map_detect_to_anchor_snapshot(
    label: pd.DataFrame,
    universe_set: set[str],
    bounds: dict[int, tuple[int, int]],
) -> tuple[dict[int, set[str]], dict[str, int], dict[str, int], dict[str, int]]:
    rugs = label[label["collection_type"] == POSITIVE_TYPE].copy()
    detect_missing = int(rugs["detected_block_number"].isna().sum())
    create_missing = int(pd.to_numeric(rugs.get("create_block_number", np.nan), errors="coerce").isna().sum()) if "create_block_number" in rugs.columns else int(len(rugs))
    rugs = rugs.dropna(subset=["detected_block_number"]).copy()
    rugs["detected_block_number"] = rugs["detected_block_number"].astype(np.int64)
    rugs["create_block_number"] = pd.to_numeric(rugs.get("create_block_number", np.nan), errors="coerce")
    rugs = rugs[rugs["contract_address"].isin(universe_set)].copy()
    rugs = rugs.sort_values(["contract_address", "detected_block_number"], kind="mergesort")
    rugs = rugs.drop_duplicates("contract_address", keep="first")

    anchor_pos: dict[int, set[str]] = {}
    addr_to_anchor: dict[str, int] = {}
    create_block_map: dict[str, int] = {}
    not_in_any_window = 0

    items = [(int(k), v[0], v[1]) for k, v in bounds.items()]
    items.sort(key=lambda x: x[0])

    for r in rugs.itertuples(index=False):
        addr = str(r.contract_address)
        db = int(r.detected_block_number)
        anchor = None
        for sidx, sb, eb in items:
            if sb <= db <= eb:
                anchor = sidx
                break
        if anchor is None:
            not_in_any_window += 1
            continue

        anchor_pos.setdefault(anchor, set()).add(addr)
        addr_to_anchor[addr] = anchor
        cblk = r.create_block_number
        if pd.notna(cblk):
            create_block_map[addr] = int(cblk)

    meta = {
        "rugpull_detect_missing": detect_missing,
        "rugpull_create_missing": create_missing,
        "rugpull_not_in_any_window": not_in_any_window,
        "rugpull_anchor_mapped": int(len(addr_to_anchor)),
    }
    return anchor_pos, addr_to_anchor, meta, create_block_map


def evaluate_anchor_lookback(
    scores: pd.DataFrame,
    bounds: dict[int, tuple[int, int]],
    anchor_pos: dict[int, set[str]],
    addr_to_anchor: dict[str, int],
    create_block_map: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_metrics = []
    rows_auprc = []

    by_snapshot: dict[int, pd.DataFrame] = {
        int(k): v.copy()
        for k, v in scores.sort_values(["snapshot_idx", "rank_C"], kind="mergesort").groupby("snapshot_idx", sort=True)
    }

    anchor_windows = sorted(anchor_pos.keys())
    for w in anchor_windows:
        m_set = anchor_pos.get(w, set())
        m_size_total = int(len(m_set))
        if m_size_total <= 0:
            continue

        for delta in DELTA_WINDOWS:
            t = int(w - int(delta))
            if t < 1:
                continue
            if t not in by_snapshot or t not in bounds:
                continue

            cands_raw = by_snapshot[t].copy()
            if cands_raw.empty:
                continue

            n_raw = int(len(cands_raw))
            if n_raw <= 0:
                continue

            # Remove rugpulls already discovered in previous windows (< t),
            # but keep K based on raw window size as requested.
            prev_discovered = {a for a, aw in addr_to_anchor.items() if int(aw) < int(t)}
            cands = cands_raw[~cands_raw["contract_address"].isin(prev_discovered)].copy()
            n_removed_prev = int(n_raw - len(cands))
            if cands.empty:
                continue

            if USE_BOTTOM_RANK:
                cands = cands.sort_values(["rank_C", "contract_address"], ascending=[False, True], kind="mergesort")
            else:
                cands = cands.sort_values(["rank_C", "contract_address"], ascending=[True, True], kind="mergesort")
            cands = cands.reset_index(drop=True)

            cand_addrs = cands["contract_address"].tolist()
            cand_set = set(cand_addrs)
            n_eval = int(len(cands))
            if n_eval <= 0:
                continue

            pos_set_eval = m_set.intersection(cand_set)
            n_pos_eval = int(len(pos_set_eval))
            lookback_end_block = int(bounds[t][1])
            pos_set_detectable = {a for a in pos_set_eval if int(create_block_map.get(a, 10**30)) <= lookback_end_block}
            n_pos_detectable = int(len(pos_set_detectable))

            risk_score = (cands["rank_C"] / cands["n_contracts_snapshot"].replace(0, np.nan)).fillna(0.0).astype(float).values
            y_true = np.array([1 if a in pos_set_eval else 0 for a in cand_addrs], dtype=np.int64)
            auprc, auprc_reason = average_precision_score_binary(y_true=y_true, y_score=risk_score)

            rows_auprc.append(
                {
                    "anchor_snapshot_idx": int(w),
                    "lookback_snapshot_idx": int(t),
                    "delta_windows": int(delta),
                    "anchor_start_block": int(bounds[w][0]),
                    "anchor_end_block": int(bounds[w][1]),
                    "lookback_start_block": int(bounds[t][0]),
                    "lookback_end_block": int(bounds[t][1]),
                    "n_candidates_raw": n_raw,
                    "n_removed_prev_discovered": n_removed_prev,
                    "n_candidates_eval": n_eval,
                    "n_positives_anchor_window": m_size_total,
                    "n_positives_eval_candidates": n_pos_eval,
                    "n_positives_detectable": n_pos_detectable,
                    "auprc": auprc,
                    "auprc_reason": auprc_reason,
                }
            )

            for p in TOP_PERCENT_LIST:
                k = int(ceil(float(p) * float(n_raw)))
                k = max(1, min(k, n_eval))
                topk = set(cand_addrs[:k])
                hits = int(len(topk.intersection(pos_set_eval)))
                hits_detectable = int(len(topk.intersection(pos_set_detectable)))
                recall_strict = float(hits / m_size_total) if m_size_total > 0 else np.nan
                recall_detectable = float(hits_detectable / n_pos_detectable) if n_pos_detectable > 0 else np.nan
                precision = float(hits / k)

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
                        "n_candidates_raw": n_raw,
                        "n_removed_prev_discovered": n_removed_prev,
                        "n_candidates_eval": n_eval,
                        "k": k,
                        "n_positives_anchor_window": m_size_total,
                        "n_positives_eval_candidates": n_pos_eval,
                        "n_positives_detectable": n_pos_detectable,
                        "hits": hits,
                        "hits_detectable": hits_detectable,
                        "recall_at_k_strict": recall_strict,
                        "recall_at_k_detectable": recall_detectable,
                        "recall_at_k": recall_strict,
                        "precision_at_k": precision,
                    }
                )

    return pd.DataFrame(rows_metrics), pd.DataFrame(rows_auprc)


def plot_recall_precision(mean_df: pd.DataFrame) -> None:
    if mean_df.empty:
        return

    plt.figure(figsize=(8.8, 4.8))
    for p in TOP_PERCENT_LIST:
        sub = mean_df[mean_df["top_percent"] == float(p)].sort_values("delta_windows")
        if sub.empty:
            continue
        plt.plot(sub["delta_windows"], sub["recall_mean"], linewidth=2.0, marker="o", label=f"Top{int(round(p*100))}%")
    plt.title("Recall@K vs Delta (Anchor-based Lookback Evaluation)")
    plt.xlabel("Delta (windows, 1 window = 7 days)")
    plt.ylabel("Mean Recall@K")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "recall_at_k_vs_delta.png", dpi=FIG_DPI)
    plt.close()

    plt.figure(figsize=(8.8, 4.8))
    for p in TOP_PERCENT_LIST:
        sub = mean_df[mean_df["top_percent"] == float(p)].sort_values("delta_windows")
        if sub.empty:
            continue
        plt.plot(sub["delta_windows"], sub["precision_mean"], linewidth=2.0, marker="o", label=f"Top{int(round(p*100))}%")
    plt.title("Precision@K vs Delta (Anchor-based Lookback Evaluation)")
    plt.xlabel("Delta (windows, 1 window = 7 days)")
    plt.ylabel("Mean Precision@K")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_at_k_vs_delta.png", dpi=FIG_DPI)
    plt.close()


def plot_auprc(auprc_mean_df: pd.DataFrame) -> None:
    if auprc_mean_df.empty:
        return

    sub = auprc_mean_df.sort_values("delta_windows")
    x = sub["delta_windows"].values
    y = sub["auprc_mean"].values
    y_std = sub["auprc_std"].fillna(0.0).values
    y_low = np.maximum(0.0, y - y_std)
    y_high = np.minimum(1.0, y + y_std)

    plt.figure(figsize=(8.8, 4.8))
    plt.plot(x, y, linewidth=2.0, marker="o", label="Mean AUPRC")
    plt.fill_between(x, y_low, y_high, alpha=0.2, label="±1 std")
    plt.title("AUPRC vs Delta (Anchor-based Lookback Evaluation)")
    plt.xlabel("Delta (windows, 1 window = 7 days)")
    plt.ylabel("Mean AUPRC")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "auprc_vs_delta.png", dpi=FIG_DPI)
    plt.close()


def plot_threshold_sensitivity(metrics_mean_df: pd.DataFrame) -> None:
    if metrics_mean_df.empty:
        return

    deltas_to_plot = [0, 1, 3, 5, 10]
    deltas_to_plot = [d for d in deltas_to_plot if d in set(metrics_mean_df["delta_windows"].tolist())]
    if not deltas_to_plot:
        return

    plt.figure(figsize=(8.8, 4.8))
    for d in deltas_to_plot:
        sub = metrics_mean_df[metrics_mean_df["delta_windows"] == int(d)].sort_values("top_percent")
        if sub.empty:
            continue
        plt.plot(sub["top_percent"] * 100.0, sub["recall_mean"], marker="o", linewidth=1.8, label=f"Δ={d}")
    plt.title("Recall@K vs Top-% Threshold")
    plt.xlabel("Top-% Threshold")
    plt.ylabel("Mean Recall@K")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "recall_vs_top_percent.png", dpi=FIG_DPI)
    plt.close()

    plt.figure(figsize=(8.8, 4.8))
    for d in deltas_to_plot:
        sub = metrics_mean_df[metrics_mean_df["delta_windows"] == int(d)].sort_values("top_percent")
        if sub.empty:
            continue
        plt.plot(sub["top_percent"] * 100.0, sub["precision_mean"], marker="o", linewidth=1.8, label=f"Δ={d}")
    plt.title("Precision@K vs Top-% Threshold")
    plt.xlabel("Top-% Threshold")
    plt.ylabel("Mean Precision@K")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_vs_top_percent.png", dpi=FIG_DPI)
    plt.close()


def select_best_recall_under_precision(metrics_mean_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in sorted(set(metrics_mean_df["delta_windows"].tolist())):
        sub_d = metrics_mean_df[metrics_mean_df["delta_windows"] == int(d)].copy()
        for pmin in PRECISION_TARGET_LIST:
            feasible = sub_d[sub_d["precision_mean"] >= float(pmin)].copy()
            if feasible.empty:
                rows.append(
                    {
                        "delta_windows": int(d),
                        "precision_target": float(pmin),
                        "best_top_percent": np.nan,
                        "best_recall_mean": np.nan,
                        "best_precision_mean": np.nan,
                    }
                )
                continue
            feasible = feasible.sort_values(["recall_mean", "precision_mean", "top_percent"], ascending=[False, False, True], kind="mergesort")
            best = feasible.iloc[0]
            rows.append(
                {
                    "delta_windows": int(d),
                    "precision_target": float(pmin),
                    "best_top_percent": float(best["top_percent"]),
                    "best_recall_mean": float(best["recall_mean"]),
                    "best_precision_mean": float(best["precision_mean"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    t0 = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores, plan, label, universe_set = load_inputs()
    bounds = build_window_bounds(plan)
    anchor_pos, addr_to_anchor, flag_meta, create_block_map = map_detect_to_anchor_snapshot(
        label=label,
        universe_set=universe_set,
        bounds=bounds,
    )

    metrics_df, auprc_df = evaluate_anchor_lookback(
        scores=scores,
        bounds=bounds,
        anchor_pos=anchor_pos,
        addr_to_anchor=addr_to_anchor,
        create_block_map=create_block_map,
    )
    if metrics_df.empty:
        raise ValueError("No valid metric rows generated under anchor-lookback mode.")

    metrics_mean_df = (
        metrics_df.groupby(["delta_windows", "top_percent"], as_index=False)
        .agg(
            recall_mean=("recall_at_k", "mean"),
            recall_count_non_nan=("recall_at_k", lambda s: int(pd.Series(s).notna().sum())),
            recall_strict_mean=("recall_at_k_strict", "mean"),
            recall_strict_count_non_nan=("recall_at_k_strict", lambda s: int(pd.Series(s).notna().sum())),
            precision_mean=("precision_at_k", "mean"),
            precision_count=("precision_at_k", "count"),
            n_anchor_windows=("anchor_snapshot_idx", "nunique"),
        )
        .sort_values(["delta_windows", "top_percent"], kind="mergesort")
        .reset_index(drop=True)
    )

    if auprc_df.empty:
        auprc_mean_df = pd.DataFrame(columns=["delta_windows", "auprc_mean", "auprc_std", "auprc_count_non_nan", "n_anchor_windows"])
    else:
        auprc_mean_df = (
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
    metrics_mean_df.to_csv(OUTPUT_DIR / "metrics_mean_by_delta.csv", index=False, encoding="utf-8-sig")
    auprc_df.to_csv(OUTPUT_DIR / "auprc_window_level.csv", index=False, encoding="utf-8-sig")
    auprc_mean_df.to_csv(OUTPUT_DIR / "auprc_mean_by_delta.csv", index=False, encoding="utf-8-sig")
    best_threshold_df = select_best_recall_under_precision(metrics_mean_df)
    best_threshold_df.to_csv(OUTPUT_DIR / "best_threshold_by_precision_target.csv", index=False, encoding="utf-8-sig")

    plot_recall_precision(metrics_mean_df)
    plot_auprc(auprc_mean_df)
    plot_threshold_sensitivity(metrics_mean_df)

    t1 = perf_counter()
    run_meta = pd.DataFrame(
        [
            {"key": "evaluation_mode", "value": "anchor_lookback"},
            {"key": "delta_range", "value": "0..10"},
            {"key": "history_shortage_policy", "value": "skip"},
            {"key": "recall_denominator_mode", "value": "strict_anchor_window"},
            {"key": "window_boundary_rule", "value": "start_block<=detected_block<=end_block"},
            {"key": "scores_csv", "value": str(SCORES_CSV)},
            {"key": "plan_csv", "value": str(PLAN_CSV)},
            {"key": "label_csv", "value": str(LABEL_CSV)},
            {"key": "universe_csv", "value": str(UNIVERSE_CSV)},
            {"key": "top_percent_list", "value": "|".join(str(x) for x in TOP_PERCENT_LIST)},
            {"key": "precision_target_list", "value": "|".join(str(x) for x in PRECISION_TARGET_LIST)},
            {"key": "delta_windows", "value": "|".join(str(x) for x in DELTA_WINDOWS)},
            {"key": "positive_type", "value": POSITIVE_TYPE},
            {"key": "use_bottom_rank", "value": int(1 if USE_BOTTOM_RANK else 0)},
            {"key": "n_input_scores_rows", "value": int(len(scores))},
            {"key": "n_input_snapshots", "value": int(plan["snapshot_idx"].nunique())},
            {"key": "n_universe_contracts", "value": int(len(universe_set))},
            {"key": "n_anchor_windows", "value": int(len(anchor_pos))},
            {"key": "n_anchor_rugpull_events", "value": int(len(addr_to_anchor))},
            {"key": "n_rugpull_detect_missing", "value": int(flag_meta["rugpull_detect_missing"])},
            {"key": "n_rugpull_create_missing", "value": int(flag_meta["rugpull_create_missing"])},
            {"key": "n_rugpull_not_in_any_window", "value": int(flag_meta["rugpull_not_in_any_window"])},
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
    print(f"[Saved] {(OUTPUT_DIR / 'best_threshold_by_precision_target.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'run_meta.csv').resolve()}")
    print(f"[Saved] figures in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
