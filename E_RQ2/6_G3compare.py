from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    "RATIO_SCALES": [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
    "N_REPEATS": 10,
    "DATA_ROOT": ROOT / "output" / "A" / "13_RQ2compare_G3_v1",
    "RESULT_ROOT": ROOT / "output" / "E" / "6_G3compare_fourway_v2",
    "GENERATOR_SCRIPT": ROOT / "A_dataget" / "11_RQ2data_G3.py",
    "OUR_SCRIPT": ROOT / "E_RQ2" / "0_our.py",
    "BIRANK_SCRIPT": ROOT / "E_RQ2" / "1_birank1.py",
    "PAGERANK_SCRIPT": ROOT / "E_RQ2" / "2_pagerank2.py",
    "EIGENTRUST_SCRIPT": ROOT / "E_RQ2" / "3_eigentrust3.py",
    "GENERATOR_SEED": 11,
    "REUSE_EXISTING_DATASETS": True,
    "REUSE_EXISTING_RESULTS": False,
    "BASELINE_TAG": "_base_no_g3_sybil_trade",
    "CONTRACT_COUNTS": {"Hard": 80, "Hype": 20, "Zombie": 50, "Sybil": 10},
    "G3_TARGET_CONTRACT_LIMIT": 10,
    "G3_ATTACK_WINDOW_DAYS": 30,
    "G3_ATTACKER_GROUP": "g3_rebel",
    "RUN_OUR": True,
    "RUN_BIRANK": True,
    "RUN_PAGERANK": True,
    "RUN_EIGENTRUST": True,
    "OUR_BETA_LIST": [0.7],
    "BIRANK_ETA_LIST": [0.7],
    "PAGERANK_GAMMA_LIST": [0.7],
    "EIGENTRUST_ALPHA_LIST": [0.7],
    "OUR_PRIMARY_BETA": 0.7,
    "BIRANK_PRIMARY_ETA": 0.7,
    "PAGERANK_PRIMARY_GAMMA": 0.7,
    "EIGENTRUST_PRIMARY_ALPHA": 0.7,
}


METHOD_PARAM_META = {
    "our": {"param_name": "beta", "suffix": "beta", "score_prefix": "3_stageC"},
    "birank": {"param_name": "eta", "suffix": "eta", "score_prefix": "1_birank"},
    "pagerank": {"param_name": "gamma", "suffix": "gamma", "score_prefix": "2_pagerank"},
    "eigentrust": {"param_name": "alpha", "suffix": "alpha", "score_prefix": "3_eigentrust"},
}


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN_MOD = _load_module("g3_generator_compare", CONFIG["GENERATOR_SCRIPT"])
OUR_MOD = _load_module("our_pipeline_g3_compare", CONFIG["OUR_SCRIPT"])
BIRANK_MOD = _load_module("birank_g3_compare", CONFIG["BIRANK_SCRIPT"])
PAGERANK_MOD = _load_module("pagerank_g3_compare", CONFIG["PAGERANK_SCRIPT"])
EIGENTRUST_MOD = _load_module("eigentrust_g3_compare", CONFIG["EIGENTRUST_SCRIPT"])


def _ratio_tag(rebel_ratio: float) -> str:
    return f"p{int(round(float(rebel_ratio) * 100)):03d}"


def _dataset_tag(rebel_ratio: float, repeat_id: int) -> str:
    return f"{_ratio_tag(rebel_ratio)}_rep{int(repeat_id):02d}"


def _dataset_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "data": data_dir / "RQdataG3.csv",
        "users": data_dir / "RQdataG3_users.csv",
        "contracts": data_dir / "RQdataG3_contracts.csv",
        "report": data_dir / "g3_report.json",
        "config": data_dir / "g3_config.json",
    }


def _all_exist(paths: dict[str, Path], keys: list[str]) -> bool:
    return all(paths[k].exists() for k in keys)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


def _apply_contract_counts(contract_profiles: list[Any], cfg: dict[str, Any]) -> list[Any]:
    count_map = {str(k): int(v) for k, v in dict(cfg.get("CONTRACT_COUNTS", {})).items()}
    for profile in contract_profiles:
        category = str(getattr(profile, "category", ""))
        if category in count_map:
            profile.num_contracts = count_map[category]
    return contract_profiles


def _count_sybil_trade_rows(data_csv: Path) -> int:
    df = pd.read_csv(data_csv, low_memory=False, usecols=["contract_id", "tx_type"])
    return int(
        (
            df["contract_id"].astype(str).str.startswith("Sybil")
            & df["tx_type"].astype(str).str.lower().eq("trade")
        ).sum()
    )


def _baseline_dataset_is_valid(dataset_paths: dict[str, Path]) -> bool:
    required = ["data", "users", "contracts", "report", "config"]
    if not _all_exist(dataset_paths, required):
        return False
    return _count_sybil_trade_rows(dataset_paths["data"]) == 0


def _sample_seed(rebel_ratio: float, repeat_id: int) -> int:
    return int(CONFIG["GENERATOR_SEED"]) * 100_000 + int(round(rebel_ratio * 10_000)) * 100 + int(repeat_id)


def _ratio_dataset_is_valid(dataset_paths: dict[str, Path], rebel_ratio: float, repeat_id: int) -> bool:
    required = ["data", "users", "contracts", "report", "config"]
    if not _all_exist(dataset_paths, required):
        return False
    report = _read_json(dataset_paths["report"])
    summary = dict(report.get("summary") or {})
    checks = dict(report.get("checks") or {})
    return (
        int(summary.get("repeat_id", -1)) == int(repeat_id)
        and abs(float(summary.get("rebel_ratio", -1.0)) - float(rebel_ratio)) < 1e-12
        and int(summary.get("sample_seed", -1)) == _sample_seed(rebel_ratio, repeat_id)
        and bool(checks.get("all_attacks_after_base", False))
        and bool(checks.get("one_trade_per_rebel", False))
    )


def _method_param_values(cfg: dict[str, Any], method: str) -> list[float]:
    if method == "our":
        return [float(x) for x in cfg["OUR_BETA_LIST"]]
    if method == "birank":
        return [float(x) for x in cfg["BIRANK_ETA_LIST"]]
    if method == "pagerank":
        return [float(x) for x in cfg["PAGERANK_GAMMA_LIST"]]
    if method == "eigentrust":
        return [float(x) for x in cfg["EIGENTRUST_ALPHA_LIST"]]
    raise ValueError(f"Unknown method: {method}")


def _score_file_paths(method: str, outdir: Path, param_value: float | None = None) -> tuple[Path, Path]:
    meta = METHOD_PARAM_META[method]
    prefix = meta["score_prefix"]
    if param_value is None:
        return outdir / f"{prefix}_contract_scores.csv", outdir / f"{prefix}_user_scores.csv"
    suffix_tag = f"{meta['suffix']}{float(param_value):.2f}"
    return (
        outdir / f"{prefix}_contract_scores_{suffix_tag}.csv",
        outdir / f"{prefix}_user_scores_{suffix_tag}.csv",
    )


def _baseline_result_dir(method: str) -> Path:
    return Path(CONFIG["RESULT_ROOT"]) / f"{method}_{_dataset_tag(0.0, 0)}"


def _result_exists(method: str, outdir: Path, param_values: list[float]) -> bool:
    required = list(_score_file_paths(method, outdir, None))
    for param_value in param_values:
        required.extend(_score_file_paths(method, outdir, float(param_value)))
    return all(path.exists() for path in required)


def ensure_baseline_dataset(base_dir: Path) -> dict[str, Path]:
    dataset_paths = _dataset_paths(base_dir)
    if bool(CONFIG["REUSE_EXISTING_DATASETS"]) and _baseline_dataset_is_valid(dataset_paths):
        print(f"[G3Compare] reuse baseline dataset: {base_dir}")
        return dataset_paths

    base_dir.mkdir(parents=True, exist_ok=True)
    contract_profiles, user_profiles, _, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, CONFIG)
    GEN_MOD.generate_g3_baseline(
        contract_profiles=contract_profiles,
        user_profiles=user_profiles,
        save_dir=str(base_dir),
        out_name="RQdataG3.csv",
        horizon_months=36,
    )
    return dataset_paths


def generate_dataset_for_ratio_repeat(rebel_ratio: float, repeat_id: int, data_dir: Path) -> dict[str, Path]:
    dataset_paths = _dataset_paths(data_dir)
    if bool(CONFIG["REUSE_EXISTING_DATASETS"]) and _ratio_dataset_is_valid(dataset_paths, rebel_ratio, repeat_id):
        print(f"[G3Compare] reuse dataset for ratio={rebel_ratio:.4f}, repeat={repeat_id}: {data_dir}")
        return dataset_paths

    base_dir = Path(CONFIG["DATA_ROOT"]) / str(CONFIG["BASELINE_TAG"])
    base_paths = ensure_baseline_dataset(base_dir)
    if float(rebel_ratio) == 0.0:
        return base_paths

    data_dir.mkdir(parents=True, exist_ok=True)
    contract_profiles, user_profiles, _, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, CONFIG)
    GEN_MOD.generate_g3_dataset_from_base(
        base_data_csv=str(base_paths["data"]),
        base_users_csv=str(base_paths["users"]),
        base_contracts_csv=str(base_paths["contracts"]),
        contract_profiles=contract_profiles,
        user_profiles=user_profiles,
        save_dir=str(data_dir),
        out_name="RQdataG3.csv",
        g3_cfg={
            "enable": True,
            "rebel_ratio": float(rebel_ratio),
            "repeat_id": int(repeat_id),
            "sample_seed": _sample_seed(rebel_ratio, repeat_id),
            "target_contract_limit": int(CONFIG["G3_TARGET_CONTRACT_LIMIT"]),
            "attack_window_days": int(CONFIG["G3_ATTACK_WINDOW_DAYS"]),
            "max_trades_per_rebel": 1,
        },
        save_reports=True,
    )
    return dataset_paths


def run_our_on_dataset(dataset_paths: dict[str, Path], outdir: Path, compare_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(OUR_MOD.CONFIG)
    cfg.update(
        {
            "RAW_DATA_CSV": dataset_paths["data"],
            "USERS_META_CSV": dataset_paths["users"],
            "CONTRACTS_META_CSV": dataset_paths["contracts"],
            "OUTDIR": outdir,
            "STAGE_C_BETA": float(compare_cfg["OUR_PRIMARY_BETA"]),
        }
    )
    return OUR_MOD.run_pipeline(cfg)


def run_birank_on_dataset(our_outdir: Path, outdir: Path, compare_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(BIRANK_MOD.CONFIG)
    cfg.update(
        {
            "INPUT_DIR": our_outdir,
            "OUTDIR": outdir,
            "ETA_LIST": [float(x) for x in compare_cfg["BIRANK_ETA_LIST"]],
            "PRIMARY_ETA": float(compare_cfg["BIRANK_PRIMARY_ETA"]),
        }
    )
    return BIRANK_MOD.run(cfg)


def run_pagerank_on_dataset(our_outdir: Path, outdir: Path, compare_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(PAGERANK_MOD.CONFIG)
    cfg.update(
        {
            "INPUT_DIR": our_outdir,
            "OUTDIR": outdir,
            "GAMMA_LIST": [float(x) for x in compare_cfg["PAGERANK_GAMMA_LIST"]],
            "PRIMARY_GAMMA": float(compare_cfg["PAGERANK_PRIMARY_GAMMA"]),
        }
    )
    return PAGERANK_MOD.run(cfg)


def run_eigentrust_on_dataset(dataset_paths: dict[str, Path], our_outdir: Path, outdir: Path, compare_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(EIGENTRUST_MOD.CONFIG)
    cfg.update(
        {
            "INPUT_DIR": our_outdir,
            "RAW_DATA_CSV": dataset_paths["data"],
            "CONTRACTS_META_CSV": dataset_paths["contracts"],
            "OUTDIR": outdir,
            "ALPHA_LIST": [float(x) for x in compare_cfg["EIGENTRUST_ALPHA_LIST"]],
            "PRIMARY_ALPHA": float(compare_cfg["EIGENTRUST_PRIMARY_ALPHA"]),
        }
    )
    return EIGENTRUST_MOD.run(cfg)


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else float("nan")


def _normalized_rank(series: pd.Series, total_n: int) -> pd.Series:
    if total_n <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series.astype(float) - 1.0) / float(total_n - 1)


def collect_method_metrics(method: str, dataset_paths: dict[str, Path], outdir: Path, rebel_ratio: float, repeat_id: int, param_value: float | None = None) -> dict[str, Any]:
    if method == "our":
        contract_rank_col, user_rank_col = "rank_C_contract", "rank_C_user"
        contract_score_col, user_score_col = "cC", "uC"
    elif method == "birank":
        contract_rank_col, user_rank_col = "rank_contract", "rank_user"
        contract_score_col, user_score_col = "c_birank", "u_birank"
    elif method == "pagerank":
        contract_rank_col, user_rank_col = "rank_contract", "rank_user"
        contract_score_col, user_score_col = "c_pagerank", "u_pagerank"
    elif method == "eigentrust":
        contract_rank_col, user_rank_col = "rank_contract", "rank_user"
        contract_score_col, user_score_col = "c_eigentrust", "u_eigentrust"
    else:
        raise ValueError(f"Unknown method: {method}")

    contract_path, user_path = _score_file_paths(method, outdir, param_value=param_value)
    base_contract_path, base_user_path = _score_file_paths(method, _baseline_result_dir(method), param_value=param_value)
    df_c = pd.read_csv(contract_path)
    df_u = pd.read_csv(user_path)
    base_df_u = pd.read_csv(base_user_path) if base_user_path.exists() else pd.DataFrame(columns=["user", user_rank_col, user_score_col])
    report = _read_json(dataset_paths["report"])
    summary = dict(report.get("summary") or {})
    sampled_rebels = set((report.get("samples") or {}).get("sampled_rebels") or [])

    target_contracts = set(summary.get("target_contract_ids") or [cid for cid in df_c["contract_id"].astype(str).tolist() if cid.startswith("Sybil")])
    target_c = df_c[df_c["contract_id"].astype(str).isin(target_contracts)].copy()
    g3_u = df_u[df_u["user"].astype(str).isin(sampled_rebels)].copy()
    base_g3_u = base_df_u[base_df_u["user"].astype(str).isin(sampled_rebels)].copy()

    target_c["norm_rank"] = _normalized_rank(target_c[contract_rank_col], len(df_c))
    g3_u["norm_rank"] = _normalized_rank(g3_u[user_rank_col], len(df_u))
    base_g3_u["norm_rank"] = _normalized_rank(base_g3_u[user_rank_col], len(base_df_u)) if len(base_df_u) else pd.Series(dtype=float)
    param_meta = METHOD_PARAM_META[method]

    post_rank_mean = _safe_mean(g3_u[user_rank_col])
    pre_rank_mean = _safe_mean(base_g3_u[user_rank_col])
    post_score_mean = _safe_mean(g3_u[user_score_col])
    pre_score_mean = _safe_mean(base_g3_u[user_score_col])

    return {
        "method": method,
        "param_name": param_meta["param_name"],
        "param_value": float(param_value) if param_value is not None else float("nan"),
        "rebel_ratio": float(rebel_ratio),
        "repeat_id": int(repeat_id),
        "sample_seed": int(summary.get("sample_seed", _sample_seed(rebel_ratio, repeat_id))),
        "dataset_tag": _dataset_tag(rebel_ratio, repeat_id),
        "n_rebels": int(len(sampled_rebels)),
        "n_contracts_total": int(len(df_c)),
        "n_users_total": int(len(df_u)),
        "target_contract_mean_rank": _safe_mean(target_c[contract_rank_col]),
        "target_contract_mean_norm_rank": _safe_mean(target_c["norm_rank"]),
        "target_contract_best_rank": float(target_c[contract_rank_col].min()) if len(target_c) else float("nan"),
        "target_contract_score_mean": _safe_mean(target_c[contract_score_col]),
        "target_contract_top20_count": int((target_c[contract_rank_col] <= 20).sum()),
        "g3_user_mean_rank": post_rank_mean,
        "g3_user_mean_norm_rank": _safe_mean(g3_u["norm_rank"]),
        "g3_user_best_rank": float(g3_u[user_rank_col].min()) if len(g3_u) else float("nan"),
        "g3_user_score_mean": post_score_mean,
        "g3_user_top200_count": int((g3_u[user_rank_col] <= 200).sum()),
        "g3_user_count_observed": int(len(g3_u)),
        "g3_user_pre_mean_rank": pre_rank_mean,
        "g3_user_pre_mean_norm_rank": _safe_mean(base_g3_u["norm_rank"]),
        "g3_user_pre_best_rank": float(base_g3_u[user_rank_col].min()) if len(base_g3_u) else float("nan"),
        "g3_user_pre_score_mean": pre_score_mean,
        "g3_user_rank_deterioration": post_rank_mean - pre_rank_mean if pd.notna(post_rank_mean) and pd.notna(pre_rank_mean) else float("nan"),
        "g3_user_norm_rank_deterioration": _safe_mean(g3_u["norm_rank"]) - _safe_mean(base_g3_u["norm_rank"]) if len(g3_u) and len(base_g3_u) else float("nan"),
        "g3_user_score_drop": pre_score_mean - post_score_mean if pd.notna(post_score_mean) and pd.notna(pre_score_mean) else float("nan"),
    }


def aggregate_repeat_metrics(runs_df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "n_rebels",
        "target_contract_mean_rank",
        "target_contract_mean_norm_rank",
        "target_contract_best_rank",
        "g3_user_mean_rank",
        "g3_user_mean_norm_rank",
        "g3_user_best_rank",
        "g3_user_score_mean",
        "g3_user_pre_mean_rank",
        "g3_user_pre_mean_norm_rank",
        "g3_user_pre_best_rank",
        "g3_user_pre_score_mean",
        "g3_user_rank_deterioration",
        "g3_user_norm_rank_deterioration",
        "g3_user_score_drop",
    ]
    grouped = runs_df.groupby(["method", "param_name", "param_value", "rebel_ratio"], dropna=False)
    records = []
    for key, sub in grouped:
        method, param_name, param_value, rebel_ratio = key
        record = {
            "method": method,
            "param_name": param_name,
            "param_value": float(param_value),
            "rebel_ratio": float(rebel_ratio),
            "n_repeats_actual": int(sub["repeat_id"].nunique()),
        }
        for col in value_cols:
            record[f"{col}_mean"] = float(sub[col].mean()) if len(sub) else float("nan")
            record[f"{col}_std"] = float(sub[col].std(ddof=0)) if len(sub) else float("nan")
        records.append(record)
    return pd.DataFrame(records).sort_values(["method", "param_value", "rebel_ratio"]).reset_index(drop=True)


def run_compare(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    compare_cfg = dict(CONFIG)
    if config:
        compare_cfg.update(config)

    result_root = Path(compare_cfg["RESULT_ROOT"])
    result_root.mkdir(parents=True, exist_ok=True)
    base_dir = Path(compare_cfg["DATA_ROOT"]) / str(compare_cfg["BASELINE_TAG"])
    ensure_baseline_dataset(base_dir)

    ratio_scales = [float(x) for x in compare_cfg["RATIO_SCALES"]]
    n_repeats = int(compare_cfg["N_REPEATS"])
    print(f"[G3Compare] ratios={ratio_scales} | repeats={n_repeats}")

    run_records: list[dict[str, Any]] = []
    for rebel_ratio in ratio_scales:
        for repeat_id in range(n_repeats):
            dataset_tag = _dataset_tag(rebel_ratio, repeat_id)
            data_dir = Path(compare_cfg["DATA_ROOT"]) / dataset_tag
            print(f"[G3Compare] ratio={rebel_ratio:.4f}, repeat={repeat_id}")
            dataset_paths = generate_dataset_for_ratio_repeat(rebel_ratio, repeat_id, data_dir)

            our_outdir = result_root / f"our_{dataset_tag}"
            our_params = _method_param_values(compare_cfg, "our")
            if bool(compare_cfg["RUN_OUR"]):
                if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("our", our_outdir, our_params):
                    print(f"[G3Compare] reuse OUR result: {our_outdir}")
                else:
                    run_our_on_dataset(dataset_paths, our_outdir, compare_cfg)
                for param_value in our_params:
                    run_records.append(collect_method_metrics("our", dataset_paths, our_outdir, rebel_ratio, repeat_id, param_value))

            birank_outdir = result_root / f"birank_{dataset_tag}"
            birank_params = _method_param_values(compare_cfg, "birank")
            if bool(compare_cfg["RUN_BIRANK"]):
                if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("birank", birank_outdir, birank_params):
                    print(f"[G3Compare] reuse BiRank result: {birank_outdir}")
                else:
                    run_birank_on_dataset(our_outdir, birank_outdir, compare_cfg)
                for param_value in birank_params:
                    run_records.append(collect_method_metrics("birank", dataset_paths, birank_outdir, rebel_ratio, repeat_id, param_value))

            pagerank_outdir = result_root / f"pagerank_{dataset_tag}"
            pagerank_params = _method_param_values(compare_cfg, "pagerank")
            if bool(compare_cfg["RUN_PAGERANK"]):
                if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("pagerank", pagerank_outdir, pagerank_params):
                    print(f"[G3Compare] reuse PageRank result: {pagerank_outdir}")
                else:
                    run_pagerank_on_dataset(our_outdir, pagerank_outdir, compare_cfg)
                for param_value in pagerank_params:
                    run_records.append(collect_method_metrics("pagerank", dataset_paths, pagerank_outdir, rebel_ratio, repeat_id, param_value))

            eigentrust_outdir = result_root / f"eigentrust_{dataset_tag}"
            eigentrust_params = _method_param_values(compare_cfg, "eigentrust")
            if bool(compare_cfg["RUN_EIGENTRUST"]):
                if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("eigentrust", eigentrust_outdir, eigentrust_params):
                    print(f"[G3Compare] reuse EigenTrust result: {eigentrust_outdir}")
                else:
                    run_eigentrust_on_dataset(dataset_paths, our_outdir, eigentrust_outdir, compare_cfg)
                for param_value in eigentrust_params:
                    run_records.append(collect_method_metrics("eigentrust", dataset_paths, eigentrust_outdir, rebel_ratio, repeat_id, param_value))

    runs_df = pd.DataFrame(run_records).sort_values(["method", "param_value", "rebel_ratio", "repeat_id"]).reset_index(drop=True)
    summary_df = aggregate_repeat_metrics(runs_df)
    runs_path = result_root / "g3_scale_compare_runs.csv"
    summary_path = result_root / "g3_scale_compare_summary.csv"
    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[G3Compare] runs saved: {runs_path}")
    print(f"[G3Compare] summary saved: {summary_path}")
    print("[G3Compare] finished")
    return runs_df, summary_df


if __name__ == "__main__":
    run_compare()
