from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = ROOT / "E_RQ2" / "9_G6compare.py"
AUTOTUNE_OUT = ROOT / "output" / "E" / "9_G6compare_autotune"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_space_algo() -> list[dict[str, Any]]:
    # Focus on parameters that directly affect OUR stage-C behavior.
    betas = [0.50, 0.60, 0.70, 0.80, 0.90]
    alphas = [0.70, 0.85, 0.95]
    prior_modes = ["stageb_u0", "uniform"]
    include_mint = [True, False]
    disable_u0 = [False, True]
    dense_k = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    out: list[dict[str, Any]] = []
    for beta, alpha, prior, mint_edge, disable in itertools.product(
        betas, alphas, prior_modes, include_mint, disable_u0
    ):
        # If disable_u0=True, alpha is effectively overridden to 1.0 in compare pipeline.
        # Keep combinations explicit for traceability.
        out.append(
            {
                "OUR_PRIMARY_BETA": float(beta),
                "OUR_BETA_LIST": [float(beta)],
                "OUR_STAGE_C_ALPHA": float(alpha),
                "OUR_STAGE_C_USER_PRIOR_MODE": str(prior),
                "OUR_STAGE_C_INCLUDE_MINT_EDGE": bool(mint_edge),
                "OUR_DISABLE_U0_PRIOR": bool(disable),
                "BOTTOM_K_RATIOS": list(dense_k),
            }
        )
    return out


def _candidate_space_data() -> list[dict[str, Any]]:
    # Focus on data-generation knobs, not ranking algorithm internals.
    # Keep Sybil fixed at 10 per your current setup and vary composition + seed.
    seeds = [101, 121, 131, 211, 307]
    hard_hype_sets = [
        (100, 50),
        (150, 100),
        (200, 150),
        (250, 150),
        (300, 200),
    ]
    zombie_counts = [10, 20, 30, 40]
    sybil_count = 10
    dense_k = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    out: list[dict[str, Any]] = []
    for seed, (hard_n, hype_n), zombie_n in itertools.product(seeds, hard_hype_sets, zombie_counts):
        out.append(
            {
                "GENERATOR_SEED": int(seed),
                "CONTRACT_COUNTS": {
                    "Hard": int(hard_n),
                    "Hype": int(hype_n),
                    "Zombie": int(zombie_n),
                    "Sybil": int(sybil_count),
                },
                "BOTTOM_K_RATIOS": list(dense_k),
            }
        )
    return out


def _candidate_space_hybrid() -> list[dict[str, Any]]:
    # Small hybrid set: blend data knobs with a compact algorithm subset.
    data = _candidate_space_data()[:24]
    algo = [
        {
            "OUR_PRIMARY_BETA": 0.6,
            "OUR_BETA_LIST": [0.6],
            "OUR_STAGE_C_ALPHA": 0.85,
            "OUR_STAGE_C_USER_PRIOR_MODE": "stageb_u0",
            "OUR_STAGE_C_INCLUDE_MINT_EDGE": True,
            "OUR_DISABLE_U0_PRIOR": False,
        },
        {
            "OUR_PRIMARY_BETA": 0.7,
            "OUR_BETA_LIST": [0.7],
            "OUR_STAGE_C_ALPHA": 0.85,
            "OUR_STAGE_C_USER_PRIOR_MODE": "stageb_u0",
            "OUR_STAGE_C_INCLUDE_MINT_EDGE": True,
            "OUR_DISABLE_U0_PRIOR": False,
        },
        {
            "OUR_PRIMARY_BETA": 0.8,
            "OUR_BETA_LIST": [0.8],
            "OUR_STAGE_C_ALPHA": 0.85,
            "OUR_STAGE_C_USER_PRIOR_MODE": "stageb_u0",
            "OUR_STAGE_C_INCLUDE_MINT_EDGE": True,
            "OUR_DISABLE_U0_PRIOR": False,
        },
    ]
    out: list[dict[str, Any]] = []
    for d, a in itertools.product(data, algo):
        merged = dict(d)
        merged.update(a)
        out.append(merged)
    return out


def _score_trial(
    plot_summary_csv: Path,
    method: str = "our",
    metric_col: str = "contract_auprc_excl_zombie_mean",
) -> float:
    if not plot_summary_csv.exists():
        return float("nan")
    df = pd.read_csv(plot_summary_csv)
    if metric_col not in df.columns or "method" not in df.columns:
        return float("nan")
    sub = df[df["method"].astype(str).eq(str(method))]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub[metric_col], errors="coerce").dropna().mean())


def run_autotune(
    max_trials: int = 20,
    n_repeats: int = 1,
    attack_repeat_ids: list[int] | None = None,
    objective_method: str = "our",
    objective_metric: str = "contract_auprc_excl_zombie_mean",
    tune_mode: str = "data",
) -> Path:
    attack_repeat_ids = [] if attack_repeat_ids is None else [int(x) for x in attack_repeat_ids]
    AUTOTUNE_OUT.mkdir(parents=True, exist_ok=True)
    g6_mod = _load_module("g6_compare_autotune_mod", COMPARE_SCRIPT)
    mode = str(tune_mode).strip().lower()
    if mode == "algo":
        candidates = _candidate_space_algo()
    elif mode == "hybrid":
        candidates = _candidate_space_hybrid()
    else:
        candidates = _candidate_space_data()
    candidates = candidates[: max(int(max_trials), 1)]
    records: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_cfg: dict[str, Any] | None = None
    start = time.time()

    for idx, cand in enumerate(candidates, start=1):
        trial_tag = f"trial_{idx:03d}"
        trial_result_root = AUTOTUNE_OUT / trial_tag
        trial_data_root = AUTOTUNE_OUT / "_data_cache" / trial_tag
        trial_cfg = {
            "N_REPEATS": int(n_repeats),
            "ATTACK_REPEAT_IDS": list(attack_repeat_ids),
            "DATA_ROOT": trial_data_root,
            "RESULT_ROOT": trial_result_root,
            # Data-tuning requires fresh generation per trial.
            "REUSE_EXISTING_DATASETS": False,
            "REUSE_EXISTING_RESULTS": False,
            "DISABLE_DATA_GENERATION": False,
            # Keep all methods enabled so objective has fair comparison context.
            "RUN_OUR": True,
            "RUN_OUR_NO_U0": True,
            "RUN_BIRANK": True,
            "RUN_PAGERANK": True,
            "RUN_EIGENTRUST": True,
        }
        trial_cfg.update(cand)
        if mode == "data":
            cc = dict(cand.get("CONTRACT_COUNTS", {}))
            print(
                f"[autotune] {idx}/{len(candidates)} | mode=data | seed={cand.get('GENERATOR_SEED')} "
                f"| Hard={cc.get('Hard')} Hype={cc.get('Hype')} Zombie={cc.get('Zombie')} Sybil={cc.get('Sybil')}",
                flush=True,
            )
        else:
            print(
                f"[autotune] {idx}/{len(candidates)} | mode={mode} | beta={cand.get('OUR_PRIMARY_BETA')} "
                f"| alpha={cand.get('OUR_STAGE_C_ALPHA')} | prior={cand.get('OUR_STAGE_C_USER_PRIOR_MODE')} "
                f"| mint={cand.get('OUR_STAGE_C_INCLUDE_MINT_EDGE')} | disable_u0={cand.get('OUR_DISABLE_U0_PRIOR')}",
                flush=True,
            )
        g6_mod.run_compare(trial_cfg)
        score = _score_trial(
            plot_summary_csv=trial_result_root / "g6_compare_plot_summary.csv",
            method=objective_method,
            metric_col=objective_metric,
        )
        rec = {
            "trial_id": idx,
            "trial_tag": trial_tag,
            "tune_mode": mode,
            "score": float(score),
            "objective_method": str(objective_method),
            "objective_metric": str(objective_metric),
            **cand,
            "result_root": str(trial_result_root),
        }
        records.append(rec)
        if pd.notna(score) and float(score) > float(best_score):
            best_score = float(score)
            best_cfg = dict(rec)
        print(f"[autotune] trial={idx} score={score}", flush=True)

    elapsed = time.time() - start
    table = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)
    table.to_csv(AUTOTUNE_OUT / "autotune_results.csv", index=False)
    payload = {
        "max_trials": int(max_trials),
        "tune_mode": mode,
        "n_repeats": int(n_repeats),
        "attack_repeat_ids": attack_repeat_ids,
        "objective_method": str(objective_method),
        "objective_metric": str(objective_metric),
        "elapsed_seconds": float(elapsed),
        "best": best_cfg if best_cfg is not None else {},
        "results_csv": str(AUTOTUNE_OUT / "autotune_results.csv"),
    }
    with open(AUTOTUNE_OUT / "autotune_best.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[autotune] done | best_score={best_score} | out={AUTOTUNE_OUT}", flush=True)
    return AUTOTUNE_OUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-tune parameters for E_RQ2/9_G6compare.py")
    parser.add_argument("--max-trials", type=int, default=20, help="How many candidate configs to evaluate.")
    parser.add_argument("--n-repeats", type=int, default=1, help="N_REPEATS passed into compare run.")
    parser.add_argument(
        "--attack-repeat-ids",
        type=str,
        default="0",
        help="Comma-separated repeat ids. Example: 0,1,2 . Empty means use N_REPEATS.",
    )
    parser.add_argument(
        "--objective-method",
        type=str,
        default="our",
        help="Method used for objective score (default: our).",
    )
    parser.add_argument(
        "--objective-metric",
        type=str,
        default="contract_auprc_excl_zombie_mean",
        help="Column in g6_compare_plot_summary.csv to maximize.",
    )
    parser.add_argument(
        "--tune-mode",
        type=str,
        default="data",
        choices=["data", "algo", "hybrid"],
        help="What to tune: data (default), algo, or hybrid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = [int(x.strip()) for x in str(args.attack_repeat_ids).split(",") if x.strip() != ""]
    run_autotune(
        max_trials=int(args.max_trials),
        n_repeats=int(args.n_repeats),
        attack_repeat_ids=ids,
        objective_method=str(args.objective_method),
        objective_metric=str(args.objective_metric),
        tune_mode=str(args.tune_mode),
    )


if __name__ == "__main__":
    main()
