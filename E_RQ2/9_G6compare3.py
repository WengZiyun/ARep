from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "N_REPEATS": 10,
    "ATTACK_REPEAT_IDS": [],
    "ATTACK_GAS_MULTIPLIERS_BASE": [2.0, 4.0, 10.0, 16.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0],
    "ATTACK_CUSTOM_MULTIPLIERS": [],
    "ATTACK_INCLUDE_CANONICAL_MULTIPLIERS": True,
    "ATTACK_GAS_DELTA_TARGET": 1000.0,
    "ATTACK_GAS_MAX_MULTIPLIER_CAP": 1e12,
    "ATTACK2_DROP_RATIOS": [0.01, 0.03, 0.05, 0.10, 0.20],
    "ATTACK4_DROP_RATIOS": [0.02, 0.05, 0.10],
    "ATTACK5_DROP_RATIOS": [0.02, 0.05, 0.10],
    "ATTACK6_DROP_RATIOS": [0.02, 0.05, 0.10],
    "ATTACK7_DROP_RATIOS": [0.02, 0.05, 0.10],
    "ATTACK8_DROP_RATIOS": [0.02, 0.05, 0.10],
    "ATTACK9_DROP_RATIOS": [0.02, 0.05, 0.10],
    "DATA_ROOT": ROOT / "output" / "A" / "18_RQ2compare_G6_v9_sybil",
    "RESULT_ROOT": ROOT / "output" / "E" / "9_G6compare_v9_sybil",
    "GENERATOR_SCRIPT": ROOT / "A_dataget" / "11_RQ2data_G6.py",
    "OUR_SCRIPT": ROOT / "E_RQ2" / "0_our.py",
    "BIRANK_SCRIPT": ROOT / "E_RQ2" / "1_birank1.py",
    "PAGERANK_SCRIPT": ROOT / "E_RQ2" / "2_pagerank2.py",
    "EIGENTRUST_SCRIPT": ROOT / "E_RQ2" / "3_eigentrust3.py",
    "GENERATOR_SEED": 12371,
    "BASELINE_TAG": "_base_clean_sybil_only",
    "CALENDAR_BASE_START_YM": "2023-01",
    "CALENDAR_PRE_END_YM": "2025-12",
    "REUSE_EXISTING_DATASETS": True,
    "REUSE_EXISTING_RESULTS": True,
    "DISABLE_DATA_GENERATION": False,
    "SAVE_TRANSACTIONS_ALL": False,
    "SAVE_MONTHLY_PARTITIONS": False,
    "CONTRACT_COUNTS": {"Hard": 100, "Hype": 50, "Zombie": 20, "Sybil": 10},
    "INCLUDE_SYBIL_CONTRACTS": True,
    "ENABLE_SYBIL_USER_INJECTION": True,
    "SYBIL_USER_COUNT": 1000,
    "SYBIL_USER_GROUP": "g6_sybil_only",
    "SYBIL_USER_ONLY_CONTRACT_PREF": {"Hard": 0.0, "Hype": 0.0, "Zombie": 0.0, "Sybil": 1.0},
    "BOTTOM_K_RATIOS": [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00],
    "LIGHTWEIGHT_MAX_ONLY": False,
    "LIGHTWEIGHT_METHODS": ["our", "birank", "pagerank", "eigentrust"],
    "LIGHTWEIGHT_INDEX_COLS": ["contract_spearman_rho_mean", "user_spearman_rho_mean"],
    "RUN_OUR": True,
    "RUN_OUR_NO_U0": True,
    "RUN_OUR_SENSITIVE": False,
    "RUN_OUR_ROBUST": False,
    "RUN_BIRANK": True,
    "RUN_PAGERANK": True,
    "RUN_EIGENTRUST": True,
    "OUR_STAGE_C_INCLUDE_MINT_EDGE": True,
    "OUR_STAGE_C_ALPHA": 0.85,
    "OUR_STAGE_C_USER_PRIOR_MODE": "stageb_u0",
    "OUR_DISABLE_U0_PRIOR": False,
    "OUR_SENSITIVE_STAGE_C_ALPHA": 0.95,
    "OUR_SENSITIVE_STAGE_C_BETA": 0.85,
    "OUR_SENSITIVE_STAGE_C_USER_PRIOR_MODE": "stageb_u0",
    "OUR_SENSITIVE_INCLUDE_MINT_EDGE": True,
    "OUR_ROBUST_STAGE_C_ALPHA": 0.65,
    "OUR_ROBUST_STAGE_C_BETA": 0.55,
    "OUR_ROBUST_STAGE_C_USER_PRIOR_MODE": "uniform",
    "OUR_ROBUST_INCLUDE_MINT_EDGE": False,
    "OUR_BETA_LIST": [0.7],
    "BIRANK_ETA_LIST": [0.7],
    "PAGERANK_GAMMA_LIST": [0.7],
    "EIGENTRUST_ALPHA_LIST": [0.7],
    "OUR_PRIMARY_BETA": 0.7,
    "BIRANK_PRIMARY_ETA": 0.7,
    "PAGERANK_PRIMARY_GAMMA": 0.7,
    "EIGENTRUST_PRIMARY_ALPHA": 0.7,
    "ATTACK_MODE": "attack2",
    "ATTACK_VARIANT": "single_outlier_trade",
    "ATTACK_MODE_NOTES": {
        "attack1": "gas变化：随机单条记录gas放大",
        "attack2": "数据缺失：每个月随机丢弃部分记录",
        "attack6": "Hub inflation：按月增加高连接Sybil用户交易",
        "attack7": "Recent burst：在最近月份增加Sybil交易突发",
        "attack8": "Collusive ring：增加Sybil用户到Sybil合约的协同行为",
        "attack9": "Bridge isolation：按月删除普通用户到Sybil合约桥接边",
    },
    "ATTACK_TARGET_SCOPE": "contract_event",
    "ATTACK_PARAMS": {
        "target_month": "2025-12",
        "target_event_selector": "random_trade",
        "modify_fields": ["gas"],
        "target_user_rank": 1,
        "exclude_buyer_groups": [],
        "exclude_users": [],
        "exclude_user_prefixes": [],
        "outlier_multiplier": 1.0,
        "copy_rows": 0,
        "append_rows": 0,
        "random_seed_offset": 7000,
    },
    "ATTACK2_PARAMS": {
        "target_months": "all",
        "drop_scope": "trade_only",
        "random_seed_offset": 9100,
    },
    "ATTACK4_PARAMS": {
        "target_months": "all",
        "drop_scope": "trade_only",
        "ranking_field": "gas",
    },
    "ATTACK5_PARAMS": {
        "target_months": "all",
        "drop_scope": "trade_only",
    },
    "ATTACK6_PARAMS": {
        "target_months": "all",
        "add_scope": "trade_only",
    },
    "ATTACK7_PARAMS": {
        "target_months": "last_3",
        "add_scope": "trade_only",
    },
    "ATTACK8_PARAMS": {
        "target_months": "all",
        "add_scope": "trade_only",
    },
    "ATTACK9_PARAMS": {
        "target_months": "all",
        "drop_scope": "trade_only",
    },
    "ATTACK3_PARAMS": {
        "target_month": "2025-12",
        "target_event_selector": "random_trade",
        "target_tx_type": "trade",
        "price_field": "price_usd",
        "random_seed_offset": 9300,
    },
}

METHOD_PARAM_META = {
    "our": {"param_name": "beta", "suffix": "beta", "score_prefix": "3_stageC"},
    "our_no_u0": {"param_name": "beta", "suffix": "beta", "score_prefix": "3_stageC"},
    "our_sensitive": {"param_name": "beta", "suffix": "beta", "score_prefix": "3_stageC"},
    "our_robust": {"param_name": "beta", "suffix": "beta", "score_prefix": "3_stageC"},
    "birank": {"param_name": "eta", "suffix": "eta", "score_prefix": "1_birank"},
    "pagerank": {"param_name": "gamma", "suffix": "gamma", "score_prefix": "2_pagerank"},
    "eigentrust": {"param_name": "alpha", "suffix": "alpha", "score_prefix": "3_eigentrust"},
}

METHOD_ORDER = ["our", "our_no_u0", "our_sensitive", "our_robust", "birank", "pagerank", "eigentrust"]
METHOD_LABELS = {
    "our": "Our",
    "our_no_u0": "Our-NoU0",
    "our_sensitive": "Our-Sensitive",
    "our_robust": "Our-Robust",
    "birank": "BiRank",
    "pagerank": "PageRank",
    "eigentrust": "EigenTrust",
}
METHOD_COLORS = {
    "our": "#8c2d04",
    "our_no_u0": "#e6550d",
    "our_sensitive": "#fd8d3c",
    "our_robust": "#31a354",
    "birank": "#d95f0e",
    "pagerank": "#2171b5",
    "eigentrust": "#08306b",
}


def _format_minutes(seconds: float) -> str:
    return f"{max(0.0, float(seconds)) / 60.0:.1f}m"


def _progress_message(
    done: int,
    total: int,
    start_ts: float,
    repeat_id: int,
    attack_mode: str,
    attack_multiplier: float,
    method: str,
    status: str,
) -> str:
    elapsed = max(0.0, time.time() - float(start_ts))
    rate = elapsed / max(int(done), 1)
    remaining = rate * max(int(total) - int(done), 0)
    pct = 100.0 * float(done) / max(int(total), 1)
    if str(attack_mode) in {"attack2", "attack4", "attack5", "attack6", "attack7", "attack8", "attack9"}:
        level = f"{100.0 * float(attack_multiplier):.1f}%"
    else:
        level = f"{float(attack_multiplier):g}x"
    return (
        f"[G6 progress] {done}/{total} ({pct:5.1f}%) | "
        f"elapsed={_format_minutes(elapsed)} | eta={_format_minutes(remaining)} | "
        f"rep={int(repeat_id) + 1} | level={level} | method={method} | status={status}"
    )


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEN_MOD = _load_module("g6_generator_compare", CONFIG["GENERATOR_SCRIPT"])
OUR_MOD = _load_module("our_pipeline_g6_compare", CONFIG["OUR_SCRIPT"])
BIRANK_MOD = _load_module("birank_g6_compare", CONFIG["BIRANK_SCRIPT"])
PAGERANK_MOD = _load_module("pagerank_g6_compare", CONFIG["PAGERANK_SCRIPT"])
EIGENTRUST_MOD = _load_module("eigentrust_g6_compare", CONFIG["EIGENTRUST_SCRIPT"])


def _g6_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "data": data_dir / "RQdataG6.csv",
        "users": data_dir / "RQdataG6_users.csv",
        "contracts": data_dir / "RQdataG6_contracts.csv",
        "role_map": data_dir / "RQdataG6_user_role_map.csv",
        "report": data_dir / "g6_report.json",
        "config": data_dir / "g6_config.json",
    }


def _all_exist(paths: dict[str, Path], keys: list[str]) -> bool:
    return all(paths[k].exists() for k in keys)


def _dataset_view_exists(data_dir: Path) -> bool:
    paths = _g6_paths(data_dir)
    return _all_exist(paths, ["data", "report", "config"])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _dataset_tag(repeat_id: int) -> str:
    return f"rep{int(repeat_id):02d}_attacked"


def _score_file_paths(method: str, outdir: Path, param_value: float | None = None) -> tuple[Path, Path]:
    meta = METHOD_PARAM_META[method]
    prefix = meta["score_prefix"]
    if method in {"our", "our_no_u0", "our_sensitive", "our_robust"}:
        return outdir / f"{prefix}_contract_scores.csv", outdir / f"{prefix}_user_scores.csv"
    if param_value is None:
        return outdir / f"{prefix}_contract_scores.csv", outdir / f"{prefix}_user_scores.csv"
    suffix_tag = f"{meta['suffix']}{float(param_value):.2f}"
    return outdir / f"{prefix}_contract_scores_{suffix_tag}.csv", outdir / f"{prefix}_user_scores_{suffix_tag}.csv"


def _method_param_values(cfg: dict[str, Any], method: str) -> list[float]:
    if method == "our":
        return [float(x) for x in cfg["OUR_BETA_LIST"]]
    if method == "our_no_u0":
        return [float(x) for x in cfg["OUR_BETA_LIST"]]
    if method == "our_sensitive":
        return [float(x) for x in cfg["OUR_BETA_LIST"]]
    if method == "our_robust":
        return [float(x) for x in cfg["OUR_BETA_LIST"]]
    if method == "birank":
        return [float(x) for x in cfg["BIRANK_ETA_LIST"]]
    if method == "pagerank":
        return [float(x) for x in cfg["PAGERANK_GAMMA_LIST"]]
    if method == "eigentrust":
        return [float(x) for x in cfg["EIGENTRUST_ALPHA_LIST"]]
    raise ValueError(f"Unknown method: {method}")


def _result_exists(method: str, outdir: Path, param_values: list[float], compare_cfg: dict[str, Any]) -> bool:
    required = list(_score_file_paths(method, outdir, None))
    if method not in {"our", "our_no_u0", "our_sensitive", "our_robust"}:
        for param_value in param_values:
            required.extend(_score_file_paths(method, outdir, float(param_value)))
    if not all(path.exists() for path in dict.fromkeys(required)):
        return False
    if method in {"our", "our_no_u0", "our_sensitive", "our_robust"}:
        return _our_like_result_matches_config(method=method, outdir=outdir, compare_cfg=compare_cfg)
    return True


def _our_like_result_matches_config(method: str, outdir: Path, compare_cfg: dict[str, Any]) -> bool:
    pipeline_meta_path = outdir / "0_pipeline_meta.json"
    stagec_meta_path = outdir / "3_stageC_meta.json"
    if not pipeline_meta_path.exists() or not stagec_meta_path.exists():
        return False
    pipeline_meta = _read_json(pipeline_meta_path)
    stagec_meta = _read_json(stagec_meta_path)
    expected = _our_method_runtime_config(method, compare_cfg)
    expected_alpha = float(expected["alpha"])
    expected_beta = float(expected["beta"])
    expected_prior_mode = str(expected["prior_mode"])
    got_prior_mode = str(pipeline_meta.get("stage_c_user_prior_mode", "")).strip().lower()
    got_alpha = float(stagec_meta.get("alpha", float("nan")))
    got_beta = float(stagec_meta.get("beta", float("nan")))
    if got_prior_mode != str(expected_prior_mode).strip().lower():
        return False
    if not np.isfinite(got_alpha) or abs(got_alpha - expected_alpha) > 1e-12:
        return False
    if not np.isfinite(got_beta) or abs(got_beta - expected_beta) > 1e-12:
        return False
    return True


def _method_columns(method: str) -> tuple[str, str]:
    if method in {"our", "our_no_u0", "our_sensitive", "our_robust"}:
        return "rank_C", "rank_C"
    if method in {"birank", "pagerank", "eigentrust"}:
        return "rank_contract", "rank_user"
    raise ValueError(f"Unknown method: {method}")


def _method_score_columns(method: str) -> tuple[str, str]:
    if method in {"our", "our_no_u0", "our_sensitive", "our_robust"}:
        return "cC", "uC"
    if method == "birank":
        return "c_birank", "u_birank"
    if method == "pagerank":
        return "c_pagerank", "u_pagerank"
    if method == "eigentrust":
        return "c_eigentrust", "u_eigentrust"
    raise ValueError(f"Unknown method: {method}")


def _apply_contract_counts(contract_profiles: list[Any], cfg: dict[str, Any]) -> list[Any]:
    count_map = {str(k): int(v) for k, v in dict(cfg.get("CONTRACT_COUNTS", {})).items()}
    for profile in contract_profiles:
        category = str(getattr(profile, "category", ""))
        if category in count_map:
            profile.num_contracts = count_map[category]
    return contract_profiles


def _active_methods(compare_cfg: dict[str, Any]) -> list[str]:
    methods: list[str] = []
    if bool(compare_cfg.get("RUN_OUR", False)):
        methods.append("our")
    if bool(compare_cfg.get("RUN_OUR_NO_U0", False)):
        methods.append("our_no_u0")
    if bool(compare_cfg.get("RUN_OUR_SENSITIVE", False)):
        methods.append("our_sensitive")
    if bool(compare_cfg.get("RUN_OUR_ROBUST", False)):
        methods.append("our_robust")
    if bool(compare_cfg.get("RUN_BIRANK", False)):
        methods.append("birank")
    if bool(compare_cfg.get("RUN_PAGERANK", False)):
        methods.append("pagerank")
    if bool(compare_cfg.get("RUN_EIGENTRUST", False)):
        methods.append("eigentrust")
    return methods


def ensure_clean_dataset(base_dir: Path, compare_cfg: dict[str, Any]) -> dict[str, Path]:
    dataset_paths = _g6_paths(base_dir)
    needed = ["data", "users", "contracts", "role_map", "report", "config"]
    if bool(compare_cfg["REUSE_EXISTING_DATASETS"]) and _all_exist(dataset_paths, needed):
        return dataset_paths
    if bool(compare_cfg.get("DISABLE_DATA_GENERATION", False)):
        raise FileNotFoundError(
            "G6 clean dataset generation is disabled and required files are missing. "
            f"Expected under: {base_dir}"
        )
    contract_profiles, user_profiles, _, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, compare_cfg)
    GEN_MOD.run_g6_baseline(
        contract_profiles=contract_profiles,
        user_profiles=user_profiles,
        save_dir=base_dir,
        horizon_months=GEN_MOD._months_between_inclusive(compare_cfg["CALENDAR_BASE_START_YM"], compare_cfg["CALENDAR_PRE_END_YM"]),
        end_date=GEN_MOD._month_end_dt(compare_cfg["CALENDAR_PRE_END_YM"]),
        seed=int(compare_cfg["GENERATOR_SEED"]),
        include_sybil_contracts=bool(compare_cfg.get("INCLUDE_SYBIL_CONTRACTS", True)),
    )
    return dataset_paths


def _our_method_runtime_config(method: str, compare_cfg: dict[str, Any]) -> dict[str, Any]:
    if method == "our":
        alpha = 1.0 if bool(compare_cfg.get("OUR_DISABLE_U0_PRIOR", False)) else float(compare_cfg.get("OUR_STAGE_C_ALPHA", 0.85))
        return {
            "alpha": float(alpha),
            "beta": float(compare_cfg.get("OUR_PRIMARY_BETA", 0.7)),
            "prior_mode": str(compare_cfg.get("OUR_STAGE_C_USER_PRIOR_MODE", "stageb_u0")),
            "include_non_acquisition": bool(compare_cfg.get("OUR_STAGE_C_INCLUDE_MINT_EDGE", False)),
        }
    if method == "our_no_u0":
        alpha = 1.0 if bool(compare_cfg.get("OUR_DISABLE_U0_PRIOR", False)) else float(compare_cfg.get("OUR_STAGE_C_ALPHA", 0.85))
        return {
            "alpha": float(alpha),
            "beta": float(compare_cfg.get("OUR_PRIMARY_BETA", 0.7)),
            "prior_mode": "uniform",
            "include_non_acquisition": bool(compare_cfg.get("OUR_STAGE_C_INCLUDE_MINT_EDGE", False)),
        }
    if method == "our_sensitive":
        return {
            "alpha": float(compare_cfg.get("OUR_SENSITIVE_STAGE_C_ALPHA", 0.95)),
            "beta": float(compare_cfg.get("OUR_SENSITIVE_STAGE_C_BETA", compare_cfg.get("OUR_PRIMARY_BETA", 0.7))),
            "prior_mode": str(compare_cfg.get("OUR_SENSITIVE_STAGE_C_USER_PRIOR_MODE", "stageb_u0")),
            "include_non_acquisition": bool(compare_cfg.get("OUR_SENSITIVE_INCLUDE_MINT_EDGE", True)),
        }
    if method == "our_robust":
        return {
            "alpha": float(compare_cfg.get("OUR_ROBUST_STAGE_C_ALPHA", 0.65)),
            "beta": float(compare_cfg.get("OUR_ROBUST_STAGE_C_BETA", 0.55)),
            "prior_mode": str(compare_cfg.get("OUR_ROBUST_STAGE_C_USER_PRIOR_MODE", "uniform")),
            "include_non_acquisition": bool(compare_cfg.get("OUR_ROBUST_INCLUDE_MINT_EDGE", False)),
        }
    raise ValueError(f"Unknown our-like method: {method}")


def run_our_on_dataset(dataset_paths: dict[str, Path], outdir: Path, compare_cfg: dict[str, Any], method: str = "our") -> dict[str, Any]:
    cfg = dict(OUR_MOD.CONFIG)
    method_cfg = _our_method_runtime_config(method, compare_cfg)
    cfg.update(
        {
            "RAW_DATA_CSV": dataset_paths["data"],
            "USERS_META_CSV": dataset_paths["users"],
            "CONTRACTS_META_CSV": dataset_paths["contracts"],
            "OUTDIR": outdir,
            "STAGE_C_BETA": float(method_cfg["beta"]),
            "STAGE_C_ALPHA": float(method_cfg["alpha"]),
            "STAGE_C_USER_PRIOR_MODE": str(method_cfg["prior_mode"]),
            "STAGE_C_INCLUDE_NON_ACQUISITION": bool(method_cfg["include_non_acquisition"]),
        }
    )
    return OUR_MOD.run_pipeline(cfg)


def run_our_no_u0_on_dataset(dataset_paths: dict[str, Path], outdir: Path, compare_cfg: dict[str, Any]) -> dict[str, Any]:
    return run_our_on_dataset(dataset_paths, outdir, compare_cfg, method="our_no_u0")


def run_our_sensitive_on_dataset(dataset_paths: dict[str, Path], outdir: Path, compare_cfg: dict[str, Any]) -> dict[str, Any]:
    return run_our_on_dataset(dataset_paths, outdir, compare_cfg, method="our_sensitive")


def run_our_robust_on_dataset(dataset_paths: dict[str, Path], outdir: Path, compare_cfg: dict[str, Any]) -> dict[str, Any]:
    return run_our_on_dataset(dataset_paths, outdir, compare_cfg, method="our_robust")


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


def _run_method(method: str, dataset_paths: dict[str, Path], our_outdir: Path, outdir: Path, compare_cfg: dict[str, Any]) -> bool:
    params = _method_param_values(compare_cfg, method)
    if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists(method, outdir, params, compare_cfg):
        return False
    if method == "our":
        run_our_on_dataset(dataset_paths, outdir, compare_cfg, method="our")
        return True
    if method == "our_no_u0":
        run_our_no_u0_on_dataset(dataset_paths, outdir, compare_cfg)
        return True
    if method == "our_sensitive":
        run_our_sensitive_on_dataset(dataset_paths, outdir, compare_cfg)
        return True
    if method == "our_robust":
        run_our_robust_on_dataset(dataset_paths, outdir, compare_cfg)
        return True
    if method == "birank":
        run_birank_on_dataset(our_outdir, outdir, compare_cfg)
        return True
    if method == "pagerank":
        run_pagerank_on_dataset(our_outdir, outdir, compare_cfg)
        return True
    if method == "eigentrust":
        run_eigentrust_on_dataset(dataset_paths, our_outdir, outdir, compare_cfg)
        return True
    raise ValueError(f"Unknown method: {method}")


def _write_dataset_view(
    data_dir: Path,
    data_df: pd.DataFrame,
    users_path: Path,
    contracts_path: Path,
    role_map_path: Path,
    report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = _g6_paths(data_dir)
    data_df.to_csv(paths["data"], index=False)
    _write_json(paths["report"], report)
    _write_json(paths["config"], config)
    return {
        "data": paths["data"],
        "users": Path(users_path),
        "contracts": Path(contracts_path),
        "role_map": Path(role_map_path),
        "report": paths["report"],
        "config": paths["config"],
    }


def _select_attack_anchor(
    clean_df: pd.DataFrame,
    compare_cfg: dict[str, Any],
    repeat_id: int,
    attack_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = dict(compare_cfg.get("ATTACK_PARAMS", {}))
    if attack_params:
        params.update(dict(attack_params))
    target_month = str(params.get("target_month", compare_cfg["CALENDAR_PRE_END_YM"]))
    selector_name = str(params.get("target_event_selector", "random_trade"))
    target_tx_type = str(params.get("target_tx_type", "trade"))
    df = clean_df.copy()
    if "calendar_year_month" not in df.columns:
        df = GEN_MOD.G4_MOD._attach_month_columns(df)

    exclude_buyer_groups = {str(x).strip().lower() for x in params.get("exclude_buyer_groups", []) if str(x).strip()}
    exclude_users = {str(x).strip() for x in params.get("exclude_users", []) if str(x).strip()}
    exclude_user_prefixes = [str(x).strip() for x in params.get("exclude_user_prefixes", []) if str(x).strip()]

    def _is_excluded_user(u: str) -> bool:
        su = str(u)
        if su in exclude_users:
            return True
        return any(su.startswith(pref) for pref in exclude_user_prefixes)

    scoped_df = df.copy()
    if target_tx_type:
        scoped_df = scoped_df[scoped_df["tx_type"].astype(str).eq(target_tx_type)].copy()
    if exclude_buyer_groups and "buyer_group" in scoped_df.columns:
        scoped_df = scoped_df[~scoped_df["buyer_group"].astype(str).str.lower().isin(exclude_buyer_groups)].copy()
    if exclude_users or exclude_user_prefixes:
        buyer_ex = scoped_df["buyer"].astype(str).map(_is_excluded_user) if "buyer" in scoped_df.columns else False
        seller_ex = scoped_df["seller"].astype(str).map(_is_excluded_user) if "seller" in scoped_df.columns else False
        scoped_df = scoped_df[~(buyer_ex | seller_ex)].copy()
    month_df = scoped_df[scoped_df["calendar_year_month"].astype(str).eq(target_month)].copy()
    month_df["price_num"] = pd.to_numeric(month_df["price"], errors="coerce")
    month_df["gas_num"] = pd.to_numeric(month_df["gas"], errors="coerce")
    candidates = month_df.copy()
    if candidates.empty:
        # Fallback: if target month has no row, use any month for this tx_type scope.
        scoped_df = scoped_df.copy()
        scoped_df["price_num"] = pd.to_numeric(scoped_df["price"], errors="coerce")
        scoped_df["gas_num"] = pd.to_numeric(scoped_df["gas"], errors="coerce")
        candidates = scoped_df.copy()
    if candidates.empty:
        raise ValueError(f"No candidates found for target_month={target_month}, target_tx_type={target_tx_type}")
    if selector_name == "max_price_trade":
        candidates = candidates.sort_values(
            ["price_num", "block_number", "tx_index_in_block", "token_id"],
            ascending=[False, True, True, True],
        ).reset_index()
        row = candidates.iloc[min(max(int(repeat_id), 0), len(candidates) - 1)]
    elif selector_name in {"random_trade", "random_sale"}:
        seed = int(compare_cfg["GENERATOR_SEED"]) * 100_000 + int(params.get("random_seed_offset", 0)) + int(repeat_id)
        rng = np.random.default_rng(seed)
        row = candidates.iloc[int(rng.integers(0, len(candidates)))]
    else:
        raise ValueError(f"Unsupported target_event_selector: {selector_name}")
    effective_month = str(row.get("calendar_year_month", target_month))
    month_for_stats = scoped_df[scoped_df["calendar_year_month"].astype(str).eq(effective_month)].copy()
    month_for_stats["price_num"] = pd.to_numeric(month_for_stats["price"], errors="coerce")
    month_for_stats["gas_num"] = pd.to_numeric(month_for_stats["gas"], errors="coerce")
    month_price_avg = float(month_for_stats["price_num"].fillna(0.0).mean()) if len(month_for_stats) else float(row["price_num"])
    month_gas_avg = float(month_for_stats["gas_num"].fillna(0.0).mean()) if len(month_for_stats) else float(row["gas_num"])
    return {
        "selector_name": selector_name,
        "target_tx_type": target_tx_type,
        "target_month": effective_month,
        "target_row_index": int(getattr(row, "name")),
        "target_contract_id": str(row["contract_id"]),
        "target_token_id": int(pd.to_numeric(pd.Series([row["token_id"]]), errors="coerce").fillna(-1).iloc[0]),
        "target_user_buyer": str(row.get("buyer", "")),
        "target_user_seller": str(row.get("seller", "")),
        "old_price": float(row["price_num"]),
        "old_gas": float(pd.to_numeric(pd.Series([row["gas_num"]]), errors="coerce").fillna(0.0).iloc[0]),
        "month_avg_price": float(month_price_avg),
        "month_avg_gas": float(month_gas_avg),
    }


def _apply_attack_variant(
    clean_df: pd.DataFrame,
    anchor_meta: dict[str, Any],
    compare_cfg: dict[str, Any],
    repeat_id: int,
    attack_multiplier: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attack_variant = str(compare_cfg.get("ATTACK_VARIANT", "single_outlier_trade"))
    attack_params = dict(compare_cfg.get("ATTACK_PARAMS", {}))
    modify_fields = [str(x) for x in attack_params.get("modify_fields", ["price"])]
    attacked_df = clean_df.copy()
    row_idx = int(anchor_meta["target_row_index"])
    outlier_multiplier = float(attack_multiplier)
    base_price = max(float(anchor_meta.get("month_avg_price", 0.0)), 1e-12)
    old_price = float(anchor_meta["old_price"])
    if outlier_multiplier <= 0.0:
        new_price = float(old_price)
    else:
        new_price = base_price * outlier_multiplier
    old_gas = float(anchor_meta["old_gas"])
    new_gas = old_gas
    if attack_variant != "single_outlier_trade":
        raise ValueError(f"Unsupported ATTACK_VARIANT: {attack_variant}")
    if "price" in modify_fields:
        attacked_df.at[row_idx, "price"] = float(new_price)
    if "gas" in modify_fields:
        base_gas = max(float(anchor_meta.get("month_avg_gas", 0.0)), 1e-12)
        if outlier_multiplier <= 0.0:
            new_gas = float(old_gas)
        else:
            new_gas = base_gas * outlier_multiplier
        attacked_df.at[row_idx, "gas"] = float(new_gas)
    rows_modified = int(
        ("price" in modify_fields and abs(float(new_price) - float(old_price)) > 1e-18)
        or ("gas" in modify_fields and abs(float(new_gas) - float(old_gas)) > 1e-18)
    )
    attack_meta = {
        "attack_variant": attack_variant,
        "repeat_id": int(repeat_id),
        "target_month": str(anchor_meta["target_month"]),
        "target_row_index": int(row_idx),
        "target_contract_id": str(anchor_meta["target_contract_id"]),
        "target_token_id": int(anchor_meta["target_token_id"]),
        "target_user_buyer": str(anchor_meta["target_user_buyer"]),
        "target_user_seller": str(anchor_meta["target_user_seller"]),
        "modified_fields": modify_fields,
        "old_price": float(old_price),
        "new_price": float(new_price),
        "price_multiplier_realized": float(new_price) / max(float(old_price), 1e-12),
        "old_gas": float(old_gas),
        "new_gas": float(new_gas),
        "gas_multiplier_realized": float(new_gas) / max(float(old_gas), 1e-12),
        "attack_multiplier": float(outlier_multiplier),
        "rows_added": 0,
        "rows_modified": int(rows_modified),
        "rows_deleted": 0,
        "selector_name": str(anchor_meta["selector_name"]),
        "target_tx_type": str(anchor_meta.get("target_tx_type", "trade")),
    }
    return attacked_df, attack_meta


def _apply_attack3_modify_sale_fee(
    clean_df: pd.DataFrame,
    compare_cfg: dict[str, Any],
    repeat_id: int,
    attack_multiplier: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attack3_params = dict(compare_cfg.get("ATTACK3_PARAMS", {}))
    target_month = str(attack3_params.get("target_month", compare_cfg["CALENDAR_PRE_END_YM"]))
    target_tx_type = str(attack3_params.get("target_tx_type", "trade"))
    selector_name = str(attack3_params.get("target_event_selector", "random_trade"))
    selector_params = {
        "target_month": target_month,
        "target_event_selector": selector_name,
        "target_tx_type": target_tx_type,
        "random_seed_offset": int(attack3_params.get("random_seed_offset", 9300)),
    }
    anchor_meta = _select_attack_anchor(clean_df, compare_cfg, repeat_id, attack_params=selector_params)
    attacked_df = clean_df.copy()
    row_idx = int(anchor_meta["target_row_index"])
    requested_price_field = str(attack3_params.get("price_field", "price_usd"))
    price_field = requested_price_field if requested_price_field in attacked_df.columns else "price"
    if price_field not in attacked_df.columns:
        raise ValueError("attack3 requires a price field (price_usd or price) in dataset.")
    old_price = float(pd.to_numeric(pd.Series([attacked_df.at[row_idx, price_field]]), errors="coerce").fillna(0.0).iloc[0])
    base_price = max(float(anchor_meta.get("month_avg_price", 0.0)), 1e-12)
    if float(attack_multiplier) <= 0.0:
        new_price = float(old_price)
    else:
        new_price = base_price * float(attack_multiplier)
    attacked_df.at[row_idx, price_field] = float(new_price)
    old_gas = float(anchor_meta["old_gas"])
    new_gas = float(old_gas)
    attack_meta = {
        "attack_variant": "single_trade_price_amplify",
        "repeat_id": int(repeat_id),
        "target_month": str(anchor_meta["target_month"]),
        "target_row_index": int(row_idx),
        "target_contract_id": str(anchor_meta["target_contract_id"]),
        "target_token_id": int(anchor_meta["target_token_id"]),
        "target_user_buyer": str(anchor_meta["target_user_buyer"]),
        "target_user_seller": str(anchor_meta["target_user_seller"]),
        "modified_fields": [str(price_field)],
        "old_price": float(old_price),
        "new_price": float(new_price),
        "price_multiplier_realized": float(new_price) / max(float(old_price), 1e-12),
        "old_gas": float(old_gas),
        "new_gas": float(new_gas),
        "gas_multiplier_realized": float(new_gas) / max(float(old_gas), 1e-12),
        "attack_multiplier": float(attack_multiplier),
        "rows_added": 0,
        "rows_modified": int(abs(float(new_price) - float(old_price)) > 1e-18),
        "rows_deleted": 0,
        "selector_name": str(anchor_meta["selector_name"]),
        "target_tx_type": str(anchor_meta.get("target_tx_type", target_tx_type)),
        "price_field": str(price_field),
        "price_field_requested": str(requested_price_field),
    }
    return attacked_df, attack_meta


def _apply_attack2_monthly_random_drop(
    clean_df: pd.DataFrame,
    compare_cfg: dict[str, Any],
    repeat_id: int,
    drop_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attack2_params = dict(compare_cfg.get("ATTACK2_PARAMS", {}))
    if "calendar_year_month" not in clean_df.columns:
        clean_df = GEN_MOD.G4_MOD._attach_month_columns(clean_df)
    scope = str(attack2_params.get("drop_scope", "trade_only"))
    if scope == "trade_only":
        work = clean_df[clean_df["tx_type"].astype(str).eq("trade")].copy()
    elif scope == "all_events":
        work = clean_df.copy()
    else:
        raise ValueError(f"Unsupported ATTACK2 drop_scope: {scope}")
    months_cfg = attack2_params.get("target_months", "all")
    if isinstance(months_cfg, str) and months_cfg == "all":
        months = sorted(work["calendar_year_month"].astype(str).dropna().unique().tolist())
    else:
        months = [str(x) for x in list(months_cfg)]
    drop_idx: list[int] = []
    base_seed = int(compare_cfg["GENERATOR_SEED"]) * 100_000 + int(attack2_params.get("random_seed_offset", 9100)) + int(repeat_id)
    for mi, ym in enumerate(months):
        month_rows = work[work["calendar_year_month"].astype(str).eq(str(ym))]
        n = len(month_rows)
        if n <= 0:
            continue
        k = int(np.floor(float(drop_ratio) * n))
        if float(drop_ratio) > 0 and k <= 0:
            k = 1
        if k <= 0:
            continue
        rng = np.random.default_rng(base_seed + mi * 17)
        pick = rng.choice(month_rows.index.to_numpy(), size=min(k, n), replace=False)
        drop_idx.extend([int(x) for x in pick.tolist()])
    attacked_df = clean_df.drop(index=drop_idx).reset_index(drop=True)
    attack_meta = {
        "attack_variant": "monthly_random_drop",
        "repeat_id": int(repeat_id),
        "target_month": "all",
        "target_row_index": -1,
        "target_contract_id": "",
        "target_token_id": -1,
        "target_user_buyer": "",
        "target_user_seller": "",
        "modified_fields": ["row_presence"],
        "old_price": float("nan"),
        "new_price": float("nan"),
        "price_multiplier_realized": float("nan"),
        "old_gas": float("nan"),
        "new_gas": float("nan"),
        "gas_multiplier_realized": float("nan"),
        "attack_multiplier": float(drop_ratio),
        "rows_added": 0,
        "rows_modified": 0,
        "rows_deleted": int(len(drop_idx)),
        "selector_name": "monthly_random_drop",
        "drop_scope": scope,
        "target_month_count": int(len(months)),
    }
    return attacked_df, attack_meta


def _apply_attack4_monthly_targeted_drop(
    clean_df: pd.DataFrame,
    compare_cfg: dict[str, Any],
    repeat_id: int,
    drop_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attack4_params = dict(compare_cfg.get("ATTACK4_PARAMS", {}))
    if "calendar_year_month" not in clean_df.columns:
        clean_df = GEN_MOD.G4_MOD._attach_month_columns(clean_df)
    scope = str(attack4_params.get("drop_scope", "trade_only"))
    if scope == "trade_only":
        work = clean_df[clean_df["tx_type"].astype(str).eq("trade")].copy()
    elif scope == "all_events":
        work = clean_df.copy()
    else:
        raise ValueError(f"Unsupported ATTACK4 drop_scope: {scope}")
    ranking_field = str(attack4_params.get("ranking_field", "gas"))
    if ranking_field not in work.columns:
        raise ValueError(f"ATTACK4 ranking_field not found: {ranking_field}")
    months_cfg = attack4_params.get("target_months", "all")
    if isinstance(months_cfg, str) and months_cfg == "all":
        months = sorted(work["calendar_year_month"].astype(str).dropna().unique().tolist())
    else:
        months = [str(x) for x in list(months_cfg)]
    drop_idx: list[int] = []
    for ym in months:
        month_rows = work[work["calendar_year_month"].astype(str).eq(str(ym))].copy()
        n = len(month_rows)
        if n <= 0:
            continue
        month_rows["_rank_score"] = pd.to_numeric(month_rows[ranking_field], errors="coerce").fillna(0.0)
        month_rows = month_rows.sort_values(["_rank_score", "block_number", "tx_index_in_block"], ascending=[False, True, True])
        k = int(np.floor(float(drop_ratio) * n))
        if float(drop_ratio) > 0 and k <= 0:
            k = 1
        if k <= 0:
            continue
        drop_idx.extend([int(x) for x in month_rows.head(min(k, n)).index.tolist()])
    attacked_df = clean_df.drop(index=drop_idx).reset_index(drop=True)
    attack_meta = {
        "attack_variant": "monthly_targeted_drop",
        "repeat_id": int(repeat_id),
        "target_month": "all",
        "target_row_index": -1,
        "target_contract_id": "",
        "target_token_id": -1,
        "target_user_buyer": "",
        "target_user_seller": "",
        "modified_fields": ["row_presence"],
        "old_price": float("nan"),
        "new_price": float("nan"),
        "price_multiplier_realized": float("nan"),
        "old_gas": float("nan"),
        "new_gas": float("nan"),
        "gas_multiplier_realized": float("nan"),
        "attack_multiplier": float(drop_ratio),
        "rows_added": 0,
        "rows_modified": 0,
        "rows_deleted": int(len(drop_idx)),
        "selector_name": "monthly_targeted_drop",
        "drop_scope": scope,
        "target_month_count": int(len(months)),
        "ranking_field": ranking_field,
    }
    return attacked_df, attack_meta


def _apply_attack5_bridge_edge_drop(
    clean_df: pd.DataFrame,
    compare_cfg: dict[str, Any],
    repeat_id: int,
    drop_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attack5_params = dict(compare_cfg.get("ATTACK5_PARAMS", {}))
    if "calendar_year_month" not in clean_df.columns:
        clean_df = GEN_MOD.G4_MOD._attach_month_columns(clean_df)
    scope = str(attack5_params.get("drop_scope", "trade_only"))
    if scope == "trade_only":
        work = clean_df[clean_df["tx_type"].astype(str).eq("trade")].copy()
    elif scope == "all_events":
        work = clean_df.copy()
    else:
        raise ValueError(f"Unsupported ATTACK5 drop_scope: {scope}")
    months_cfg = attack5_params.get("target_months", "all")
    if isinstance(months_cfg, str) and months_cfg == "all":
        months = sorted(work["calendar_year_month"].astype(str).dropna().unique().tolist())
    else:
        months = [str(x) for x in list(months_cfg)]
    drop_idx: list[int] = []
    for ym in months:
        month_rows = work[work["calendar_year_month"].astype(str).eq(str(ym))].copy()
        n = len(month_rows)
        if n <= 0:
            continue
        bdeg = month_rows.groupby("buyer").size().rename("bdeg")
        sdeg = month_rows.groupby("seller").size().rename("sdeg")
        month_rows = month_rows.join(bdeg, on="buyer").join(sdeg, on="seller")
        month_rows["bridge_score"] = (
            pd.to_numeric(month_rows["bdeg"], errors="coerce").fillna(0.0)
            + pd.to_numeric(month_rows["sdeg"], errors="coerce").fillna(0.0)
        )
        month_rows = month_rows.sort_values(["bridge_score", "block_number", "tx_index_in_block"], ascending=[False, True, True])
        k = int(np.floor(float(drop_ratio) * n))
        if float(drop_ratio) > 0 and k <= 0:
            k = 1
        if k <= 0:
            continue
        drop_idx.extend([int(x) for x in month_rows.head(min(k, n)).index.tolist()])
    attacked_df = clean_df.drop(index=drop_idx).reset_index(drop=True)
    attack_meta = {
        "attack_variant": "bridge_edge_targeted_drop",
        "repeat_id": int(repeat_id),
        "target_month": "all",
        "target_row_index": -1,
        "target_contract_id": "",
        "target_token_id": -1,
        "target_user_buyer": "",
        "target_user_seller": "",
        "modified_fields": ["row_presence"],
        "old_price": float("nan"),
        "new_price": float("nan"),
        "price_multiplier_realized": float("nan"),
        "old_gas": float("nan"),
        "new_gas": float("nan"),
        "gas_multiplier_realized": float("nan"),
        "attack_multiplier": float(drop_ratio),
        "rows_added": 0,
        "rows_modified": 0,
        "rows_deleted": int(len(drop_idx)),
        "selector_name": "bridge_edge_targeted_drop",
        "drop_scope": scope,
        "target_month_count": int(len(months)),
    }
    return attacked_df, attack_meta


def _resolve_target_months(work_df: pd.DataFrame, months_cfg: Any) -> list[str]:
    all_months = sorted(work_df["calendar_year_month"].astype(str).dropna().unique().tolist())
    if isinstance(months_cfg, str):
        mode = str(months_cfg).strip().lower()
        if mode == "all":
            return all_months
        if mode.startswith("last_"):
            try:
                n = max(int(mode.replace("last_", "")), 1)
            except ValueError:
                n = 1
            return all_months[-n:]
    return [str(x) for x in list(months_cfg)]


def _ratio_to_count(ratio: float, n: int) -> int:
    k = int(np.floor(float(ratio) * int(n)))
    if float(ratio) > 0.0 and int(n) > 0 and k <= 0:
        k = 1
    return max(0, min(int(n), int(k)))


def _is_sybil_contract(series: pd.Series) -> pd.Series:
    return series.astype(str).str.contains("sybil", case=False, na=False)


def _is_sybil_user(series: pd.Series) -> pd.Series:
    return series.astype(str).str.contains("sybil", case=False, na=False)


def _apply_attack6_hub_inflation(
    clean_df: pd.DataFrame,
    compare_cfg: dict[str, Any],
    repeat_id: int,
    add_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attack6_params = dict(compare_cfg.get("ATTACK6_PARAMS", {}))
    if "calendar_year_month" not in clean_df.columns:
        clean_df = GEN_MOD.G4_MOD._attach_month_columns(clean_df)
    scope = str(attack6_params.get("add_scope", "trade_only"))
    if scope == "trade_only":
        work = clean_df[clean_df["tx_type"].astype(str).eq("trade")].copy()
    elif scope == "all_events":
        work = clean_df.copy()
    else:
        raise ValueError(f"Unsupported ATTACK6 add_scope: {scope}")
    months = _resolve_target_months(work, attack6_params.get("target_months", "all"))
    add_rows: list[pd.DataFrame] = []
    for ym in months:
        month_rows = work[work["calendar_year_month"].astype(str).eq(str(ym))].copy()
        n = len(month_rows)
        if n <= 0:
            continue
        sybil_mask = _is_sybil_user(month_rows["buyer"])
        if not sybil_mask.any():
            continue
        k = _ratio_to_count(float(add_ratio), n)
        if k <= 0:
            continue
        sybil_rows = month_rows[sybil_mask].copy()
        bdeg = sybil_rows.groupby("buyer").size().rename("bdeg")
        sybil_rows = sybil_rows.join(bdeg, on="buyer")
        sybil_rows = sybil_rows.sort_values(["bdeg", "block_number", "tx_index_in_block"], ascending=[False, True, True])
        take = sybil_rows.head(min(k, len(sybil_rows))).copy()
        if take.empty:
            continue
        take["gas"] = pd.to_numeric(take["gas"], errors="coerce").fillna(0.0) * 0.5
        add_rows.append(take[clean_df.columns].copy())
    attacked_df = clean_df.copy()
    if add_rows:
        attacked_df = pd.concat([attacked_df] + add_rows, axis=0, ignore_index=True)
    attack_meta = {
        "attack_variant": "hub_inflation_add",
        "repeat_id": int(repeat_id),
        "target_month": "all",
        "target_row_index": -1,
        "target_contract_id": "",
        "target_token_id": -1,
        "target_user_buyer": "",
        "target_user_seller": "",
        "modified_fields": ["row_presence", "gas"],
        "old_price": float("nan"),
        "new_price": float("nan"),
        "price_multiplier_realized": float("nan"),
        "old_gas": float("nan"),
        "new_gas": float("nan"),
        "gas_multiplier_realized": float("nan"),
        "attack_multiplier": float(add_ratio),
        "rows_added": int(sum(len(x) for x in add_rows)),
        "rows_modified": 0,
        "rows_deleted": 0,
        "selector_name": "hub_inflation_add",
        "drop_scope": scope,
        "target_month_count": int(len(months)),
    }
    return attacked_df.reset_index(drop=True), attack_meta


def _apply_attack7_recent_burst(
    clean_df: pd.DataFrame,
    compare_cfg: dict[str, Any],
    repeat_id: int,
    add_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attack7_params = dict(compare_cfg.get("ATTACK7_PARAMS", {}))
    if "calendar_year_month" not in clean_df.columns:
        clean_df = GEN_MOD.G4_MOD._attach_month_columns(clean_df)
    scope = str(attack7_params.get("add_scope", "trade_only"))
    if scope == "trade_only":
        work = clean_df[clean_df["tx_type"].astype(str).eq("trade")].copy()
    elif scope == "all_events":
        work = clean_df.copy()
    else:
        raise ValueError(f"Unsupported ATTACK7 add_scope: {scope}")
    months = _resolve_target_months(work, attack7_params.get("target_months", "last_3"))
    add_rows: list[pd.DataFrame] = []
    for ym in months:
        month_rows = work[work["calendar_year_month"].astype(str).eq(str(ym))].copy()
        n = len(month_rows)
        if n <= 0:
            continue
        sybil_contract_mask = _is_sybil_contract(month_rows["contract_id"])
        candidates = month_rows[sybil_contract_mask].copy()
        if candidates.empty:
            continue
        k = _ratio_to_count(float(add_ratio), n)
        if k <= 0:
            continue
        candidates = candidates.sort_values(["block_number", "tx_index_in_block"], ascending=[False, False])
        take = candidates.head(min(k, len(candidates))).copy()
        if take.empty:
            continue
        take["block_number"] = pd.to_numeric(take["block_number"], errors="coerce").fillna(0).astype(int) + 1
        add_rows.append(take[clean_df.columns].copy())
    attacked_df = clean_df.copy()
    if add_rows:
        attacked_df = pd.concat([attacked_df] + add_rows, axis=0, ignore_index=True)
    attack_meta = {
        "attack_variant": "recent_burst_add",
        "repeat_id": int(repeat_id),
        "target_month": "recent",
        "target_row_index": -1,
        "target_contract_id": "",
        "target_token_id": -1,
        "target_user_buyer": "",
        "target_user_seller": "",
        "modified_fields": ["row_presence", "block_number"],
        "old_price": float("nan"),
        "new_price": float("nan"),
        "price_multiplier_realized": float("nan"),
        "old_gas": float("nan"),
        "new_gas": float("nan"),
        "gas_multiplier_realized": float("nan"),
        "attack_multiplier": float(add_ratio),
        "rows_added": int(sum(len(x) for x in add_rows)),
        "rows_modified": 0,
        "rows_deleted": 0,
        "selector_name": "recent_burst_add",
        "drop_scope": scope,
        "target_month_count": int(len(months)),
    }
    return attacked_df.reset_index(drop=True), attack_meta


def _apply_attack8_collusive_ring(
    clean_df: pd.DataFrame,
    compare_cfg: dict[str, Any],
    repeat_id: int,
    add_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attack8_params = dict(compare_cfg.get("ATTACK8_PARAMS", {}))
    if "calendar_year_month" not in clean_df.columns:
        clean_df = GEN_MOD.G4_MOD._attach_month_columns(clean_df)
    scope = str(attack8_params.get("add_scope", "trade_only"))
    if scope == "trade_only":
        work = clean_df[clean_df["tx_type"].astype(str).eq("trade")].copy()
    elif scope == "all_events":
        work = clean_df.copy()
    else:
        raise ValueError(f"Unsupported ATTACK8 add_scope: {scope}")
    months = _resolve_target_months(work, attack8_params.get("target_months", "all"))
    add_rows: list[pd.DataFrame] = []
    for ym in months:
        month_rows = work[work["calendar_year_month"].astype(str).eq(str(ym))].copy()
        n = len(month_rows)
        if n <= 0:
            continue
        sybil_contract_rows = month_rows[_is_sybil_contract(month_rows["contract_id"])].copy()
        if sybil_contract_rows.empty:
            continue
        sybil_users = month_rows.loc[_is_sybil_user(month_rows["buyer"]), "buyer"].astype(str).drop_duplicates().tolist()
        if not sybil_users:
            continue
        k = _ratio_to_count(float(add_ratio), n)
        if k <= 0:
            continue
        base = sybil_contract_rows.sort_values(["block_number", "tx_index_in_block"], ascending=[False, False]).head(min(k, len(sybil_contract_rows))).copy()
        if base.empty:
            continue
        base = base.reset_index(drop=True)
        for i in range(len(base)):
            base.at[i, "buyer"] = sybil_users[i % len(sybil_users)]
        add_rows.append(base[clean_df.columns].copy())
    attacked_df = clean_df.copy()
    if add_rows:
        attacked_df = pd.concat([attacked_df] + add_rows, axis=0, ignore_index=True)
    attack_meta = {
        "attack_variant": "collusive_ring_add",
        "repeat_id": int(repeat_id),
        "target_month": "all",
        "target_row_index": -1,
        "target_contract_id": "",
        "target_token_id": -1,
        "target_user_buyer": "",
        "target_user_seller": "",
        "modified_fields": ["row_presence", "buyer"],
        "old_price": float("nan"),
        "new_price": float("nan"),
        "price_multiplier_realized": float("nan"),
        "old_gas": float("nan"),
        "new_gas": float("nan"),
        "gas_multiplier_realized": float("nan"),
        "attack_multiplier": float(add_ratio),
        "rows_added": int(sum(len(x) for x in add_rows)),
        "rows_modified": 0,
        "rows_deleted": 0,
        "selector_name": "collusive_ring_add",
        "drop_scope": scope,
        "target_month_count": int(len(months)),
    }
    return attacked_df.reset_index(drop=True), attack_meta


def _apply_attack9_bridge_isolation(
    clean_df: pd.DataFrame,
    compare_cfg: dict[str, Any],
    repeat_id: int,
    drop_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    attack9_params = dict(compare_cfg.get("ATTACK9_PARAMS", {}))
    if "calendar_year_month" not in clean_df.columns:
        clean_df = GEN_MOD.G4_MOD._attach_month_columns(clean_df)
    scope = str(attack9_params.get("drop_scope", "trade_only"))
    if scope == "trade_only":
        work = clean_df[clean_df["tx_type"].astype(str).eq("trade")].copy()
    elif scope == "all_events":
        work = clean_df.copy()
    else:
        raise ValueError(f"Unsupported ATTACK9 drop_scope: {scope}")
    months = _resolve_target_months(work, attack9_params.get("target_months", "all"))
    drop_idx: list[int] = []
    for ym in months:
        month_rows = work[work["calendar_year_month"].astype(str).eq(str(ym))].copy()
        if month_rows.empty:
            continue
        bridge_mask = _is_sybil_contract(month_rows["contract_id"]) & (~_is_sybil_user(month_rows["buyer"]))
        candidates = month_rows[bridge_mask].copy()
        n = len(candidates)
        if n <= 0:
            continue
        k = _ratio_to_count(float(drop_ratio), n)
        if k <= 0:
            continue
        candidates = candidates.sort_values(["block_number", "tx_index_in_block"], ascending=[False, False])
        drop_idx.extend([int(x) for x in candidates.head(min(k, n)).index.tolist()])
    attacked_df = clean_df.drop(index=drop_idx).reset_index(drop=True)
    attack_meta = {
        "attack_variant": "bridge_isolation_drop",
        "repeat_id": int(repeat_id),
        "target_month": "all",
        "target_row_index": -1,
        "target_contract_id": "",
        "target_token_id": -1,
        "target_user_buyer": "",
        "target_user_seller": "",
        "modified_fields": ["row_presence"],
        "old_price": float("nan"),
        "new_price": float("nan"),
        "price_multiplier_realized": float("nan"),
        "old_gas": float("nan"),
        "new_gas": float("nan"),
        "gas_multiplier_realized": float("nan"),
        "attack_multiplier": float(drop_ratio),
        "rows_added": 0,
        "rows_modified": 0,
        "rows_deleted": int(len(drop_idx)),
        "selector_name": "bridge_isolation_drop",
        "drop_scope": scope,
        "target_month_count": int(len(months)),
    }
    return attacked_df, attack_meta


def _resolve_attack_multipliers(anchor_meta: dict[str, Any], compare_cfg: dict[str, Any]) -> list[float]:
    custom = [float(x) for x in compare_cfg.get("ATTACK_CUSTOM_MULTIPLIERS", []) if float(x) >= 0.0]
    if custom:
        multipliers = sorted({float(x) for x in custom})
        return multipliers if multipliers else [1.0]
    base = [float(x) for x in compare_cfg.get("ATTACK_GAS_MULTIPLIERS_BASE", [2.0, 4.0, 10.0, 16.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0])]
    canonical = [10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0] if bool(compare_cfg.get("ATTACK_INCLUDE_CANONICAL_MULTIPLIERS", True)) else []
    multipliers = sorted({float(x) for x in (base + canonical + [0.0, 1.0]) if float(x) >= 0.0})
    if not multipliers:
        multipliers = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
    return sorted({float(x) for x in multipliers})


def _attack_mode_note(compare_cfg: dict[str, Any]) -> str:
    mode = str(compare_cfg.get("ATTACK_MODE", "attack1"))
    note_map = dict(compare_cfg.get("ATTACK_MODE_NOTES", {}))
    if mode == "attack3":
        return str(note_map.get(mode, "amplify one December trade price in-memory"))
    return str(note_map.get(mode, ""))


def _resolve_attack_levels(
    clean_dataset_paths: dict[str, Path],
    compare_cfg: dict[str, Any],
    repeat_id: int,
) -> list[float]:
    mode = str(compare_cfg.get("ATTACK_MODE", "attack1"))
    if mode in {"attack1", "attack3"}:
        clean_df_for_anchor = pd.read_csv(clean_dataset_paths["data"])
        if "calendar_year_month" not in clean_df_for_anchor.columns:
            clean_df_for_anchor = GEN_MOD.G4_MOD._attach_month_columns(clean_df_for_anchor)
        if mode == "attack1":
            selector_params = dict(compare_cfg.get("ATTACK_PARAMS", {}))
            selector_params["target_month"] = str(compare_cfg["CALENDAR_PRE_END_YM"])
            selector_params["target_tx_type"] = "trade"
        else:
            selector_params = dict(compare_cfg.get("ATTACK3_PARAMS", {}))
            selector_params["target_month"] = str(compare_cfg["CALENDAR_PRE_END_YM"])
            selector_params["target_tx_type"] = str(selector_params.get("target_tx_type", "trade"))
        anchor_meta = _select_attack_anchor(clean_df_for_anchor, compare_cfg, int(repeat_id), attack_params=selector_params)
        return _resolve_attack_multipliers(anchor_meta, compare_cfg)
    if mode == "attack2":
        ratios = [float(x) for x in compare_cfg.get("ATTACK2_DROP_RATIOS", [0.01, 0.05, 0.10])]
        ratios = sorted({max(0.0, min(0.95, float(x))) for x in ratios if float(x) > 0})
        if not ratios:
            raise ValueError("ATTACK2_DROP_RATIOS is empty.")
        return ratios
    if mode == "attack4":
        ratios = [float(x) for x in compare_cfg.get("ATTACK4_DROP_RATIOS", [0.02, 0.05, 0.10])]
        ratios = sorted({max(0.0, min(0.95, float(x))) for x in ratios if float(x) > 0})
        if not ratios:
            raise ValueError("ATTACK4_DROP_RATIOS is empty.")
        return ratios
    if mode == "attack5":
        ratios = [float(x) for x in compare_cfg.get("ATTACK5_DROP_RATIOS", [0.02, 0.05, 0.10])]
        ratios = sorted({max(0.0, min(0.95, float(x))) for x in ratios if float(x) > 0})
        if not ratios:
            raise ValueError("ATTACK5_DROP_RATIOS is empty.")
        return ratios
    if mode == "attack6":
        ratios = [float(x) for x in compare_cfg.get("ATTACK6_DROP_RATIOS", [0.02, 0.05, 0.10])]
        ratios = sorted({max(0.0, min(0.95, float(x))) for x in ratios if float(x) > 0})
        if not ratios:
            raise ValueError("ATTACK6_DROP_RATIOS is empty.")
        return ratios
    if mode == "attack7":
        ratios = [float(x) for x in compare_cfg.get("ATTACK7_DROP_RATIOS", [0.02, 0.05, 0.10])]
        ratios = sorted({max(0.0, min(0.95, float(x))) for x in ratios if float(x) > 0})
        if not ratios:
            raise ValueError("ATTACK7_DROP_RATIOS is empty.")
        return ratios
    if mode == "attack8":
        ratios = [float(x) for x in compare_cfg.get("ATTACK8_DROP_RATIOS", [0.02, 0.05, 0.10])]
        ratios = sorted({max(0.0, min(0.95, float(x))) for x in ratios if float(x) > 0})
        if not ratios:
            raise ValueError("ATTACK8_DROP_RATIOS is empty.")
        return ratios
    if mode == "attack9":
        ratios = [float(x) for x in compare_cfg.get("ATTACK9_DROP_RATIOS", [0.02, 0.05, 0.10])]
        ratios = sorted({max(0.0, min(0.95, float(x))) for x in ratios if float(x) > 0})
        if not ratios:
            raise ValueError("ATTACK9_DROP_RATIOS is empty.")
        return ratios
    raise ValueError(f"Unsupported ATTACK_MODE: {mode}")


def _build_attack_report(
    clean_df: pd.DataFrame,
    attacked_df: pd.DataFrame,
    attack_meta: dict[str, Any],
    compare_cfg: dict[str, Any],
) -> dict[str, Any]:
    mode = str(compare_cfg.get("ATTACK_MODE", "attack1"))
    note = _attack_mode_note(compare_cfg)
    target_price_increased = (
        float(attack_meta["new_price"]) > float(attack_meta["old_price"])
        if np.isfinite(float(attack_meta.get("new_price", float("nan"))))
        and np.isfinite(float(attack_meta.get("old_price", float("nan"))))
        else True
    )
    target_gas_increased = (
        float(attack_meta["new_gas"]) > float(attack_meta["old_gas"])
        if np.isfinite(float(attack_meta.get("new_gas", float("nan"))))
        and np.isfinite(float(attack_meta.get("old_gas", float("nan"))))
        else True
    )
    additive_modes = {"attack6", "attack7", "attack8"}
    multiplicative_modes = {"attack1", "attack3"}
    return {
        "summary": {
            "scenario": "g6_attacked_view",
            "attack_mode": mode,
            "attack_mode_note": note,
            "attack_variant": str(attack_meta["attack_variant"]),
            "attack_multiplier": float(attack_meta.get("attack_multiplier", 1.0)),
            "target_month": str(attack_meta["target_month"]),
            "rows_clean": int(len(clean_df)),
            "rows_attacked": int(len(attacked_df)),
            "rows_added": int(attack_meta["rows_added"]),
            "rows_modified": int(attack_meta["rows_modified"]),
            "rows_deleted": int(attack_meta["rows_deleted"]),
            "target_contract_id": str(attack_meta["target_contract_id"]),
            "target_token_id": int(attack_meta["target_token_id"]),
            "target_user_buyer": str(attack_meta["target_user_buyer"]),
            "target_user_seller": str(attack_meta["target_user_seller"]),
            "old_price": float(attack_meta["old_price"]),
            "new_price": float(attack_meta["new_price"]),
            "price_multiplier_realized": float(attack_meta["price_multiplier_realized"]),
            "old_gas": float(attack_meta["old_gas"]),
            "new_gas": float(attack_meta["new_gas"]),
            "gas_multiplier_realized": float(attack_meta.get("gas_multiplier_realized", float("nan"))),
            "selector_name": str(attack_meta["selector_name"]),
            "target_tx_type": str(attack_meta.get("target_tx_type", "")),
            "price_field": str(attack_meta.get("price_field", "")),
            "drop_scope": str(attack_meta.get("drop_scope", "")),
            "target_month_count": int(attack_meta.get("target_month_count", 0)),
        },
        "checks": {
            "row_count_unchanged": int(len(clean_df)) == int(len(attacked_df))
            if mode in multiplicative_modes
            else (int(len(clean_df)) <= int(len(attacked_df)) if mode in additive_modes else int(len(clean_df)) >= int(len(attacked_df))),
            "at_least_one_row_modified": int(attack_meta["rows_modified"]) >= 1
            if mode in multiplicative_modes
            else (int(attack_meta["rows_added"]) >= 1 if mode in additive_modes else int(attack_meta["rows_deleted"]) >= 1),
            "target_price_increased": target_price_increased,
            "target_gas_increased": target_gas_increased,
        },
        "samples": {
            "modified_fields": [str(x) for x in attack_meta["modified_fields"]],
            "attack_target_scope": str(compare_cfg.get("ATTACK_TARGET_SCOPE", "")),
        },
    }


def build_attacked_view_from_clean(
    clean_dataset_paths: dict[str, Path],
    compare_cfg: dict[str, Any],
    repeat_id: int,
    attack_multiplier: float,
) -> dict[str, Any]:
    clean_df = pd.read_csv(clean_dataset_paths["data"])
    if "calendar_year_month" not in clean_df.columns:
        clean_df = GEN_MOD.G4_MOD._attach_month_columns(clean_df)
    mode = str(compare_cfg.get("ATTACK_MODE", "attack1"))
    if mode == "attack1":
        attack1_params = dict(compare_cfg.get("ATTACK_PARAMS", {}))
        attack1_params["target_month"] = str(compare_cfg["CALENDAR_PRE_END_YM"])
        attack1_params["target_tx_type"] = "trade"
        anchor_meta = _select_attack_anchor(clean_df, compare_cfg, repeat_id, attack_params=attack1_params)
        attacked_df, attack_meta = _apply_attack_variant(clean_df, anchor_meta, compare_cfg, repeat_id, float(attack_multiplier))
    elif mode == "attack2":
        attacked_df, attack_meta = _apply_attack2_monthly_random_drop(
            clean_df=clean_df,
            compare_cfg=compare_cfg,
            repeat_id=int(repeat_id),
            drop_ratio=float(attack_multiplier),
        )
    elif mode == "attack4":
        attacked_df, attack_meta = _apply_attack4_monthly_targeted_drop(
            clean_df=clean_df,
            compare_cfg=compare_cfg,
            repeat_id=int(repeat_id),
            drop_ratio=float(attack_multiplier),
        )
    elif mode == "attack5":
        attacked_df, attack_meta = _apply_attack5_bridge_edge_drop(
            clean_df=clean_df,
            compare_cfg=compare_cfg,
            repeat_id=int(repeat_id),
            drop_ratio=float(attack_multiplier),
        )
    elif mode == "attack6":
        attacked_df, attack_meta = _apply_attack6_hub_inflation(
            clean_df=clean_df,
            compare_cfg=compare_cfg,
            repeat_id=int(repeat_id),
            add_ratio=float(attack_multiplier),
        )
    elif mode == "attack7":
        attacked_df, attack_meta = _apply_attack7_recent_burst(
            clean_df=clean_df,
            compare_cfg=compare_cfg,
            repeat_id=int(repeat_id),
            add_ratio=float(attack_multiplier),
        )
    elif mode == "attack8":
        attacked_df, attack_meta = _apply_attack8_collusive_ring(
            clean_df=clean_df,
            compare_cfg=compare_cfg,
            repeat_id=int(repeat_id),
            add_ratio=float(attack_multiplier),
        )
    elif mode == "attack9":
        attacked_df, attack_meta = _apply_attack9_bridge_isolation(
            clean_df=clean_df,
            compare_cfg=compare_cfg,
            repeat_id=int(repeat_id),
            drop_ratio=float(attack_multiplier),
        )
    elif mode == "attack3":
        attacked_df, attack_meta = _apply_attack3_modify_sale_fee(
            clean_df=clean_df,
            compare_cfg=compare_cfg,
            repeat_id=int(repeat_id),
            attack_multiplier=float(attack_multiplier),
        )
    else:
        raise ValueError(f"Unsupported ATTACK_MODE: {mode}")
    report = _build_attack_report(clean_df, attacked_df, attack_meta, compare_cfg)
    config = {
        "stage": "attacked_view",
        "repeat_id": int(repeat_id),
        "attack_multiplier": float(attack_multiplier),
        "attack_mode": mode,
        "attack_mode_note": _attack_mode_note(compare_cfg),
        "attack_variant": str(compare_cfg["ATTACK_VARIANT"]),
        "attack_target_scope": str(compare_cfg["ATTACK_TARGET_SCOPE"]),
        "attack_params": dict(compare_cfg.get("ATTACK_PARAMS", {})),
        "attack2_params": dict(compare_cfg.get("ATTACK2_PARAMS", {})),
        "attack4_params": dict(compare_cfg.get("ATTACK4_PARAMS", {})),
        "attack5_params": dict(compare_cfg.get("ATTACK5_PARAMS", {})),
        "attack6_params": dict(compare_cfg.get("ATTACK6_PARAMS", {})),
        "attack7_params": dict(compare_cfg.get("ATTACK7_PARAMS", {})),
        "attack8_params": dict(compare_cfg.get("ATTACK8_PARAMS", {})),
        "attack9_params": dict(compare_cfg.get("ATTACK9_PARAMS", {})),
        "attack3_params": dict(compare_cfg.get("ATTACK3_PARAMS", {})),
    }
    return {
        "attacked_df": attacked_df,
        "attack_meta": attack_meta,
        "report": report,
        "config": config,
    }


def _build_runtime_attacked_dataset(
    clean_dataset_paths: dict[str, Path],
    runtime_dir: Path,
    compare_cfg: dict[str, Any],
    repeat_id: int,
    attack_multiplier: float,
) -> tuple[dict[str, Path], dict[str, Any]]:
    payload = build_attacked_view_from_clean(clean_dataset_paths, compare_cfg, repeat_id, float(attack_multiplier))
    attacked_dir = runtime_dir / "attacked_dataset"
    paths = _write_dataset_view(
        data_dir=attacked_dir,
        data_df=payload["attacked_df"],
        users_path=clean_dataset_paths["users"],
        contracts_path=clean_dataset_paths["contracts"],
        role_map_path=clean_dataset_paths["role_map"],
        report=payload["report"],
        config=payload["config"],
    )
    return paths, payload["report"]


def _align_rankings(
    clean_rank_df: pd.DataFrame,
    attacked_rank_df: pd.DataFrame,
    entity_col: str,
    rank_col: str,
) -> pd.DataFrame:
    left = clean_rank_df[[entity_col, rank_col]].copy().rename(columns={rank_col: "rank_clean"})
    right = attacked_rank_df[[entity_col, rank_col]].copy().rename(columns={rank_col: "rank_attacked"})
    left[entity_col] = left[entity_col].astype(str)
    right[entity_col] = right[entity_col].astype(str)
    merged = left.merge(right, on=entity_col, how="inner")
    merged["rank_clean"] = pd.to_numeric(merged["rank_clean"], errors="coerce")
    merged["rank_attacked"] = pd.to_numeric(merged["rank_attacked"], errors="coerce")
    return merged.dropna(subset=["rank_clean", "rank_attacked"]).sort_values(entity_col).reset_index(drop=True)


def _kendall_tau_from_ranks(aligned_df: pd.DataFrame) -> float:
    n = int(len(aligned_df))
    if n < 2:
        return float("nan")
    clean = aligned_df["rank_clean"].to_numpy(dtype=float)
    attacked = aligned_df["rank_attacked"].to_numpy(dtype=float)
    discordant = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n - 1):
        diff_clean = clean[i] - clean[i + 1 :]
        diff_attacked = attacked[i] - attacked[i + 1 :]
        discordant += int(np.sum(diff_clean * diff_attacked < 0))
    return 1.0 - (2.0 * float(discordant) / float(total_pairs))


def _spearman_rho_from_ranks(aligned_df: pd.DataFrame) -> float:
    n = int(len(aligned_df))
    if n < 2:
        return float("nan")
    rank_delta_sq_sum = float(
        np.square(aligned_df["rank_clean"].to_numpy(dtype=float) - aligned_df["rank_attacked"].to_numpy(dtype=float)).sum()
    )
    return 1.0 - (6.0 * rank_delta_sq_sum / float(n * (n * n - 1)))


def compute_ranking_correlation(
    clean_rank_df: pd.DataFrame,
    attacked_rank_df: pd.DataFrame,
    entity_col: str,
    rank_col: str,
) -> dict[str, float]:
    aligned = _align_rankings(clean_rank_df, attacked_rank_df, entity_col, rank_col)
    return {
        "n_entities": float(len(aligned)),
        "kendall_tau": float(_kendall_tau_from_ranks(aligned)),
        "spearman_rho": float(_spearman_rho_from_ranks(aligned)),
    }


def _get_sybil_contract_ids(contracts_meta_df: pd.DataFrame, raw_df: pd.DataFrame | None = None) -> set[str]:
    ids: set[str] = set()
    if "contract_id" in contracts_meta_df.columns:
        if "category" in contracts_meta_df.columns:
            mask = contracts_meta_df["category"].astype(str).str.lower().eq("sybil")
            ids |= {str(x) for x in contracts_meta_df.loc[mask, "contract_id"].astype(str).tolist()}
        ids |= {
            str(x)
            for x in contracts_meta_df["contract_id"].astype(str).tolist()
            if str(x).strip().lower().startswith("sybil_")
        }
    if raw_df is not None and "contract_id" in raw_df.columns:
        if "category" in raw_df.columns:
            mask = raw_df["category"].astype(str).str.lower().eq("sybil")
            ids |= {str(x) for x in raw_df.loc[mask, "contract_id"].astype(str).tolist()}
        ids |= {
            str(x)
            for x in raw_df["contract_id"].astype(str).tolist()
            if str(x).strip().lower().startswith("sybil_")
        }
    return ids


def _get_zombie_contract_ids(contracts_meta_df: pd.DataFrame) -> set[str]:
    if "contract_id" not in contracts_meta_df.columns or "category" not in contracts_meta_df.columns:
        return set()
    mask = contracts_meta_df["category"].astype(str).str.lower().eq("zombie")
    return {str(x) for x in contracts_meta_df.loc[mask, "contract_id"].astype(str).tolist()}


def _get_sybil_user_ids(users_meta_df: pd.DataFrame, compare_cfg: dict[str, Any]) -> set[str]:
    if "user" not in users_meta_df.columns:
        return set()
    group_cols = [c for c in ["group", "base_group"] if c in users_meta_df.columns]
    if not group_cols:
        return set()
    requested_count = int(compare_cfg.get("SYBIL_USER_COUNT", 0) or 0)
    sybil_group_cfg = str(compare_cfg.get("SYBIL_USER_GROUP", "")).strip().lower()
    for gcol in group_cols:
        series = users_meta_df[gcol].astype(str)
        if sybil_group_cfg and series.str.lower().eq(sybil_group_cfg).any():
            users = sorted({str(x) for x in users_meta_df.loc[series.str.lower().eq(sybil_group_cfg), "user"].astype(str).tolist()})
            if requested_count > 0:
                users = users[: min(requested_count, len(users))]
            return set(users)
    for gcol in group_cols:
        series = users_meta_df[gcol].astype(str).str.lower()
        mask = series.str.contains("sybil", na=False)
        if mask.any():
            users = sorted({str(x) for x in users_meta_df.loc[mask, "user"].astype(str).tolist()})
            if requested_count > 0:
                users = users[: min(requested_count, len(users))]
            return set(users)
    # In this synthetic setup, injected sybil-only users are often in "normal" group.
    for gcol in group_cols:
        series = users_meta_df[gcol].astype(str).str.lower()
        if series.eq("normal").any():
            users = sorted({str(x) for x in users_meta_df.loc[series.eq("normal"), "user"].astype(str).tolist()})
            if requested_count > 0:
                users = users[: min(requested_count, len(users))]
            return set(users)
    return set()


def _bottom_k_hits(
    rank_df: pd.DataFrame,
    id_col: str,
    rank_col: str,
    target_ids: set[str],
    k_ratio: float,
    universe_ids: set[str] | None = None,
) -> dict[str, float]:
    if id_col not in rank_df.columns or rank_col not in rank_df.columns:
        raise ValueError(f"Missing required ranking columns: id_col={id_col}, rank_col={rank_col}")
    work = rank_df[[id_col, rank_col]].copy()
    work[id_col] = work[id_col].astype(str)
    work[rank_col] = pd.to_numeric(work[rank_col], errors="coerce")
    if universe_ids is not None:
        allowed = {str(x) for x in set(universe_ids)}
        work = work[work[id_col].astype(str).isin(allowed)].copy()
    work = work.dropna(subset=[rank_col]).sort_values([rank_col, id_col], ascending=[True, True]).reset_index(drop=True)
    n_entities = int(len(work))
    if n_entities <= 0:
        return {
            "k": 0.0,
            "hits": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "target_total": float(len(target_ids)),
            "n_entities": 0.0,
        }
    k = max(1, int(math.ceil(float(k_ratio) * float(n_entities))))
    tail = work.tail(k)
    hits = int(tail[id_col].astype(str).isin(set(target_ids)).sum())
    precision = float(hits) / float(max(k, 1))
    recall = float(hits) / float(max(len(target_ids), 1))
    return {
        "k": float(k),
        "hits": float(hits),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "target_total": float(len(target_ids)),
        "n_entities": float(n_entities),
    }


def _average_precision_from_bottom_rank(
    rank_df: pd.DataFrame,
    id_col: str,
    rank_col: str,
    target_ids: set[str],
    universe_ids: set[str] | None = None,
) -> float:
    if id_col not in rank_df.columns or rank_col not in rank_df.columns:
        return float("nan")
    work = rank_df[[id_col, rank_col]].copy()
    work[id_col] = work[id_col].astype(str)
    work[rank_col] = pd.to_numeric(work[rank_col], errors="coerce")
    if universe_ids is not None:
        allowed = {str(x) for x in set(universe_ids)}
        work = work[work[id_col].isin(allowed)].copy()
    work = work.dropna(subset=[rank_col]).sort_values([rank_col, id_col], ascending=[False, True]).reset_index(drop=True)
    if work.empty:
        return float("nan")
    y_true = work[id_col].isin(set(target_ids)).astype(int).to_numpy(dtype=int)
    n_pos = int(y_true.sum())
    if n_pos <= 0:
        return float("nan")
    hits = 0
    precisions: list[float] = []
    for i, y in enumerate(y_true, start=1):
        if int(y) == 1:
            hits += 1
            precisions.append(float(hits) / float(i))
    if not precisions:
        return 0.0
    return float(np.mean(np.array(precisions, dtype=float)))


def _average_precision_from_scores(
    score_df: pd.DataFrame,
    id_col: str,
    score_col: str,
    target_ids: set[str],
    universe_ids: set[str] | None = None,
) -> float:
    if id_col not in score_df.columns or score_col not in score_df.columns:
        return float("nan")
    work = score_df[[id_col, score_col]].copy()
    work[id_col] = work[id_col].astype(str)
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    if universe_ids is not None:
        allowed = {str(x) for x in set(universe_ids)}
        work = work[work[id_col].isin(allowed)].copy()
    work = work.dropna(subset=[score_col]).reset_index(drop=True)
    if work.empty:
        return float("nan")
    y_true = work[id_col].isin(set(target_ids)).astype(int).to_numpy(dtype=int)
    n_pos = int(y_true.sum())
    if n_pos <= 0:
        return float("nan")
    # Lower reputation score => higher Sybil risk.
    risk = -work[score_col].to_numpy(dtype=float)
    order = np.argsort(-risk, kind="mergesort")
    y_sorted = y_true[order]
    hits = 0
    precisions: list[float] = []
    for i, y in enumerate(y_sorted, start=1):
        if int(y) == 1:
            hits += 1
            precisions.append(float(hits) / float(i))
    if not precisions:
        return 0.0
    return float(np.mean(np.array(precisions, dtype=float)))


def _auprc_from_pr_points(df: pd.DataFrame, precision_col: str, recall_col: str) -> float:
    if precision_col not in df.columns or recall_col not in df.columns:
        return float("nan")
    work = df[[precision_col, recall_col]].copy()
    work[precision_col] = pd.to_numeric(work[precision_col], errors="coerce")
    work[recall_col] = pd.to_numeric(work[recall_col], errors="coerce")
    work = work.dropna(subset=[precision_col, recall_col]).sort_values([recall_col, precision_col]).reset_index(drop=True)
    if work.empty:
        return float("nan")
    p = np.clip(work[precision_col].to_numpy(dtype=float), 0.0, 1.0)
    r = np.clip(work[recall_col].to_numpy(dtype=float), 0.0, 1.0)
    p_pad = np.concatenate(([float(p[0])], p))
    r_pad = np.concatenate(([0.0], r))
    return float(np.trapz(p_pad, r_pad))


def add_auprc_columns(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return runs_df.copy()
    group_cols = [
        "method",
        "param_name",
        "param_value",
        "repeat_id",
        "attack_mode",
        "attack_variant",
        "attack_multiplier",
    ]
    records: list[dict[str, Any]] = []
    for key, sub in runs_df.groupby(group_cols, dropna=False):
        method, param_name, param_value, repeat_id, attack_mode, attack_variant, attack_multiplier = key
        records.append(
            {
                "method": method,
                "param_name": param_name,
                "param_value": float(param_value),
                "repeat_id": int(repeat_id),
                "attack_mode": str(attack_mode),
                "attack_variant": str(attack_variant),
                "attack_multiplier": float(attack_multiplier),
                "contract_auprc_excl_zombie": _auprc_from_pr_points(
                    sub,
                    precision_col="contract_precision_at_k_excl_zombie",
                    recall_col="contract_recall_at_k_excl_zombie",
                ),
                "user_auprc": _auprc_from_pr_points(
                    sub,
                    precision_col="user_precision_at_k",
                    recall_col="user_recall_at_k",
                ),
            }
        )
    auprc_df = pd.DataFrame(records)
    return runs_df.merge(auprc_df, on=group_cols, how="left")


def collect_method_metrics(
    method: str,
    clean_outdir: Path,
    attacked_outdir: Path,
    repeat_id: int,
    attack_multiplier: float,
    attack_report: dict[str, Any],
    sybil_user_ids: set[str],
    sybil_contract_ids: set[str],
    zombie_contract_ids: set[str],
    k_ratios: list[float],
    param_value: float | None = None,
) -> list[dict[str, Any]]:
    contract_rank_col, user_rank_col = _method_columns(method)
    contract_score_col, user_score_col = _method_score_columns(method)
    clean_contract_path, clean_user_path = _score_file_paths(method, clean_outdir, param_value)
    attacked_contract_path, attacked_user_path = _score_file_paths(method, attacked_outdir, param_value)
    clean_df_c = pd.read_csv(clean_contract_path)
    clean_df_u = pd.read_csv(clean_user_path)
    attacked_df_c = pd.read_csv(attacked_contract_path)
    attacked_df_u = pd.read_csv(attacked_user_path)
    summary = dict(attack_report.get("summary") or {})
    contract_metrics = compute_ranking_correlation(clean_df_c, attacked_df_c, "contract_id", contract_rank_col)
    user_metrics = compute_ranking_correlation(clean_df_u, attacked_df_u, "user", user_rank_col)
    contract_auprc_excl_zombie = _average_precision_from_scores(
        attacked_df_c,
        id_col="contract_id",
        score_col=contract_score_col,
        target_ids=sybil_contract_ids,
        universe_ids=({str(x) for x in attacked_df_c["contract_id"].astype(str).tolist()} - set(zombie_contract_ids)),
    )
    contract_auprc_all = _average_precision_from_scores(
        attacked_df_c,
        id_col="contract_id",
        score_col=contract_score_col,
        target_ids=sybil_contract_ids,
        universe_ids=None,
    )
    user_auprc = _average_precision_from_scores(
        attacked_df_u,
        id_col="user",
        score_col=user_score_col,
        target_ids=sybil_user_ids,
        universe_ids=None,
    )
    records: list[dict[str, Any]] = []
    non_zombie_universe = {str(x) for x in attacked_df_c["contract_id"].astype(str).tolist()} - set(zombie_contract_ids)
    for k_ratio in [float(x) for x in k_ratios]:
        contract_bottom_k = _bottom_k_hits(attacked_df_c, "contract_id", contract_rank_col, sybil_contract_ids, float(k_ratio))
        contract_bottom_k_excl_zombie = _bottom_k_hits(
            attacked_df_c,
            "contract_id",
            contract_rank_col,
            sybil_contract_ids,
            float(k_ratio),
            universe_ids=non_zombie_universe,
        )
        user_bottom_k = _bottom_k_hits(attacked_df_u, "user", user_rank_col, sybil_user_ids, float(k_ratio))
        records.append(
            {
                "method": method,
                "param_name": METHOD_PARAM_META[method]["param_name"],
                "param_value": float(param_value) if param_value is not None else float("nan"),
                "repeat_id": int(repeat_id),
                "k_ratio": float(k_ratio),
                "attack_mode": str(summary.get("attack_mode", "")),
                "attack_mode_note": str(summary.get("attack_mode_note", "")),
                "attack_multiplier": float(attack_multiplier),
                "attack_variant": str(summary.get("attack_variant", "")),
                "target_month": str(summary.get("target_month", "")),
                "rows_modified": int(summary.get("rows_modified", 0)),
                "rows_added": int(summary.get("rows_added", 0)),
                "rows_deleted": int(summary.get("rows_deleted", 0)),
                "contract_n_entities": int(contract_metrics["n_entities"]),
                "contract_kendall_tau": float(contract_metrics["kendall_tau"]),
                "contract_spearman_rho": float(contract_metrics["spearman_rho"]),
                "contract_target_sybil_total": int(contract_bottom_k["target_total"]),
                "contract_k": int(contract_bottom_k["k"]),
                "contract_hits_sybil": int(contract_bottom_k["hits"]),
                "contract_precision_at_k": float(contract_bottom_k["precision_at_k"]),
                "contract_recall_at_k": float(contract_bottom_k["recall_at_k"]),
                "contract_n_entities_excl_zombie": int(contract_bottom_k_excl_zombie["n_entities"]),
                "contract_k_excl_zombie": int(contract_bottom_k_excl_zombie["k"]),
                "contract_hits_sybil_excl_zombie": int(contract_bottom_k_excl_zombie["hits"]),
                "contract_precision_at_k_excl_zombie": float(contract_bottom_k_excl_zombie["precision_at_k"]),
                "contract_recall_at_k_excl_zombie": float(contract_bottom_k_excl_zombie["recall_at_k"]),
                "contract_auprc_all": float(contract_auprc_all) if np.isfinite(contract_auprc_all) else float("nan"),
                "contract_auprc_excl_zombie": float(contract_auprc_excl_zombie) if np.isfinite(contract_auprc_excl_zombie) else float("nan"),
                "user_n_entities": int(user_metrics["n_entities"]),
                "user_kendall_tau": float(user_metrics["kendall_tau"]),
                "user_spearman_rho": float(user_metrics["spearman_rho"]),
                "user_target_sybil_total": int(user_bottom_k["target_total"]),
                "user_k": int(user_bottom_k["k"]),
                "user_hits_sybil": int(user_bottom_k["hits"]),
                "user_precision_at_k": float(user_bottom_k["precision_at_k"]),
                "user_recall_at_k": float(user_bottom_k["recall_at_k"]),
                "user_auprc": float(user_auprc) if np.isfinite(user_auprc) else float("nan"),
                "target_contract_id": str(summary.get("target_contract_id", "")),
                "target_token_id": int(summary.get("target_token_id", -1)),
                "target_user_buyer": str(summary.get("target_user_buyer", "")),
                "target_user_seller": str(summary.get("target_user_seller", "")),
                "target_tx_type": str(summary.get("target_tx_type", "")),
                "old_price": float(summary.get("old_price", 0.0)),
                "new_price": float(summary.get("new_price", 0.0)),
                "price_multiplier_realized": float(summary.get("price_multiplier_realized", float("nan"))),
                "old_gas": float(summary.get("old_gas", 0.0)),
                "new_gas": float(summary.get("new_gas", 0.0)),
                "gas_multiplier_realized": float(summary.get("gas_multiplier_realized", float("nan"))),
            }
        )
    return records


def aggregate_repeat_metrics(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()
    metric_cols = [
        "rows_modified",
        "rows_added",
        "rows_deleted",
        "contract_n_entities",
        "contract_kendall_tau",
        "contract_spearman_rho",
        "contract_target_sybil_total",
        "contract_k",
        "contract_hits_sybil",
        "contract_precision_at_k",
        "contract_recall_at_k",
        "contract_n_entities_excl_zombie",
        "contract_k_excl_zombie",
        "contract_hits_sybil_excl_zombie",
        "contract_precision_at_k_excl_zombie",
        "contract_recall_at_k_excl_zombie",
        "contract_auprc_all",
        "contract_auprc_excl_zombie",
        "user_n_entities",
        "user_kendall_tau",
        "user_spearman_rho",
        "user_target_sybil_total",
        "user_k",
        "user_hits_sybil",
        "user_precision_at_k",
        "user_recall_at_k",
        "user_auprc",
        "old_price",
        "new_price",
        "price_multiplier_realized",
        "old_gas",
        "new_gas",
        "gas_multiplier_realized",
    ]
    grouped = runs_df.groupby(
        ["method", "param_name", "param_value", "attack_mode", "attack_variant", "attack_multiplier"],
        dropna=False,
    )
    records: list[dict[str, Any]] = []
    for key, sub in grouped:
        method, param_name, param_value, attack_mode, attack_variant, attack_multiplier = key
        record = {
            "method": method,
            "param_name": param_name,
            "param_value": float(param_value),
            "k_ratio": float(pd.to_numeric(sub["k_ratio"], errors="coerce").iloc[0]) if "k_ratio" in sub.columns else float("nan"),
            "attack_mode": str(attack_mode),
            "attack_mode_note": str(sub["attack_mode_note"].mode().iloc[0]) if len(sub["attack_mode_note"].mode()) else "",
            "attack_variant": str(attack_variant),
            "attack_multiplier": float(attack_multiplier),
            "target_month": str(sub["target_month"].mode().iloc[0]) if len(sub["target_month"].mode()) else "",
            "n_repeats_actual": int(sub["repeat_id"].nunique()),
        }
        for col in metric_cols:
            mean = float(sub[col].mean()) if len(sub) else float("nan")
            std = float(sub[col].std(ddof=0)) if len(sub) else float("nan")
            record[f"{col}_mean"] = mean
            record[f"{col}_std"] = std
            record[f"{col}_ci95"] = 1.96 * std / math.sqrt(max(record["n_repeats_actual"], 1))
        records.append(record)
    return pd.DataFrame(records).sort_values(["method", "attack_mode", "attack_variant", "attack_multiplier"]).reset_index(drop=True)


def build_plot_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()
    grouped = runs_df.groupby(
        ["method", "param_name", "param_value", "attack_mode", "attack_variant", "attack_multiplier"],
        dropna=False,
    )
    records: list[dict[str, Any]] = []
    for key, sub in grouped:
        method, param_name, param_value, attack_mode, attack_variant, attack_multiplier = key
        record = {
            "method": method,
            "param_name": param_name,
            "param_value": float(param_value),
            "k_ratio": float(pd.to_numeric(sub["k_ratio"], errors="coerce").iloc[0]) if "k_ratio" in sub.columns else float("nan"),
            "attack_mode": str(attack_mode),
            "attack_variant": str(attack_variant),
            "attack_multiplier": float(attack_multiplier),
            "n_repeats_actual": int(sub["repeat_id"].nunique()),
        }
        for col in [
            "contract_kendall_tau",
            "contract_spearman_rho",
            "user_kendall_tau",
            "user_spearman_rho",
            "contract_precision_at_k",
            "contract_recall_at_k",
            "contract_precision_at_k_excl_zombie",
            "contract_recall_at_k_excl_zombie",
            "contract_auprc_all",
            "contract_auprc_excl_zombie",
            "user_precision_at_k",
            "user_recall_at_k",
            "user_auprc",
        ]:
            mean = float(sub[col].mean()) if len(sub) else float("nan")
            std = float(sub[col].std(ddof=0)) if len(sub) else float("nan")
            record[f"{col}_mean"] = mean
            record[f"{col}_std"] = std
            record[f"{col}_ci95"] = 1.96 * std / math.sqrt(max(record["n_repeats_actual"], 1))
        records.append(record)
    return pd.DataFrame(records).sort_values(["method", "attack_mode", "attack_variant", "attack_multiplier"]).reset_index(drop=True)


def _plot_metric_vs_intensity(
    plot_summary_df: pd.DataFrame,
    attack_mode: str,
    mean_col: str,
    ci_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    sub_df = plot_summary_df[plot_summary_df["attack_mode"].astype(str).eq(str(attack_mode))].copy()
    if sub_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ordered_levels = sub_df.sort_values("attack_multiplier")["attack_multiplier"].drop_duplicates().astype(float).tolist()
    x_map = {float(v): i for i, v in enumerate(ordered_levels)}
    for method in METHOD_ORDER:
        method_sub = sub_df[sub_df["method"].astype(str).eq(method)].sort_values("attack_multiplier")
        if method_sub.empty:
            continue
        x = method_sub["attack_multiplier"].astype(float).map(x_map).to_numpy(dtype=float)
        mean = method_sub[mean_col].astype(float).to_numpy()
        ci = method_sub[ci_col].fillna(0.0).astype(float).to_numpy()
        ax.plot(x, mean, marker="o", linewidth=2.2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, mean - ci, mean + ci, color=METHOD_COLORS[method], alpha=0.16)

    if str(attack_mode) in {"attack2", "attack4", "attack5", "attack6", "attack7", "attack8", "attack9"}:
        labels = [f"{int(round(v * 100))}%" for v in ordered_levels]
        ax.set_xlabel("Intensity Ratio")
    else:
        labels = [f"{v:g}x" for v in ordered_levels]
        ax.set_xlabel("Outlier Multiplier")
    ax.set_xticks(np.arange(len(labels), dtype=float))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_plots(plot_summary_df: pd.DataFrame, result_root: Path) -> None:
    if plot_summary_df.empty:
        return
    plot_df = plot_summary_df.copy()
    if "k_ratio" in plot_df.columns and plot_df["k_ratio"].notna().any():
        primary_k = float(pd.to_numeric(plot_df["k_ratio"], errors="coerce").dropna().max())
        plot_df = plot_df[pd.to_numeric(plot_df["k_ratio"], errors="coerce").eq(primary_k)].copy()
    else:
        primary_k = float("nan")
    for attack_mode in sorted(plot_summary_df["attack_mode"].astype(str).dropna().unique().tolist()):
        mode_tag = str(attack_mode).lower()
        _plot_metric_vs_intensity(
            plot_summary_df=plot_df,
            attack_mode=attack_mode,
            mean_col="contract_kendall_tau_mean",
            ci_col="contract_kendall_tau_ci95",
            ylabel="Contract Kendall Tau",
            title=f"G6 {attack_mode} Contract Kendall Tau vs Intensity (k={primary_k:.2f})" if np.isfinite(primary_k) else f"G6 {attack_mode} Contract Kendall Tau vs Intensity",
            out_path=result_root / f"g6_{mode_tag}_contract_kendall_tau_vs_intensity.png",
        )
        _plot_metric_vs_intensity(
            plot_summary_df=plot_df,
            attack_mode=attack_mode,
            mean_col="contract_spearman_rho_mean",
            ci_col="contract_spearman_rho_ci95",
            ylabel="Contract Spearman Rho",
            title=f"G6 {attack_mode} Contract Spearman Rho vs Intensity (k={primary_k:.2f})" if np.isfinite(primary_k) else f"G6 {attack_mode} Contract Spearman Rho vs Intensity",
            out_path=result_root / f"g6_{mode_tag}_contract_spearman_rho_vs_intensity.png",
        )
        _plot_metric_vs_intensity(
            plot_summary_df=plot_df,
            attack_mode=attack_mode,
            mean_col="user_kendall_tau_mean",
            ci_col="user_kendall_tau_ci95",
            ylabel="User Kendall Tau",
            title=f"G6 {attack_mode} User Kendall Tau vs Intensity (k={primary_k:.2f})" if np.isfinite(primary_k) else f"G6 {attack_mode} User Kendall Tau vs Intensity",
            out_path=result_root / f"g6_{mode_tag}_user_kendall_tau_vs_intensity.png",
        )
        _plot_metric_vs_intensity(
            plot_summary_df=plot_df,
            attack_mode=attack_mode,
            mean_col="user_spearman_rho_mean",
            ci_col="user_spearman_rho_ci95",
            ylabel="User Spearman Rho",
            title=f"G6 {attack_mode} User Spearman Rho vs Intensity (k={primary_k:.2f})" if np.isfinite(primary_k) else f"G6 {attack_mode} User Spearman Rho vs Intensity",
            out_path=result_root / f"g6_{mode_tag}_user_spearman_rho_vs_intensity.png",
        )


def _resolve_attack1_target_user(clean_our_outdir: Path, compare_cfg: dict[str, Any]) -> str:
    target_rank = max(int(compare_cfg.get("ATTACK_PARAMS", {}).get("target_user_rank", 1)), 1)
    _, user_score_path = _score_file_paths("our", clean_our_outdir, None)
    if not user_score_path.exists():
        raise FileNotFoundError(f"Missing clean our user score file: {user_score_path}")
    user_df = pd.read_csv(user_score_path)
    if "rank_C" not in user_df.columns or "user" not in user_df.columns:
        raise ValueError("Our user score file must contain columns: user, rank_C")
    chosen = user_df.sort_values("rank_C", ascending=False).iloc[min(target_rank - 1, len(user_df) - 1)]
    return str(chosen["user"])


def _resolve_attack3_target_user(clean_our_outdir: Path, compare_cfg: dict[str, Any]) -> str:
    target_rank = max(int(compare_cfg.get("ATTACK3_PARAMS", {}).get("target_user_rank", 1)), 1)
    _, user_score_path = _score_file_paths("our", clean_our_outdir, None)
    if not user_score_path.exists():
        raise FileNotFoundError(f"Missing clean our user score file: {user_score_path}")
    user_df = pd.read_csv(user_score_path)
    if "rank_C" not in user_df.columns or "user" not in user_df.columns:
        raise ValueError("Our user score file must contain columns: user, rank_C")
    chosen = user_df.sort_values("rank_C", ascending=False).iloc[min(target_rank - 1, len(user_df) - 1)]
    return str(chosen["user"])


def _safe_level_tag(x: float) -> str:
    try:
        return format(float(x), ".3e").replace("+", "")
    except Exception:
        return "nan"


def run_compare(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    compare_cfg = dict(CONFIG)
    if config:
        compare_cfg.update(config)
    data_root = Path(compare_cfg["DATA_ROOT"])
    result_root = Path(compare_cfg["RESULT_ROOT"])
    data_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    base_dir = data_root / str(compare_cfg["BASELINE_TAG"])
    clean_dataset_paths = ensure_clean_dataset(base_dir, compare_cfg)
    clean_data_df = pd.read_csv(clean_dataset_paths["data"])
    users_meta_df = pd.read_csv(clean_dataset_paths["users"])
    contracts_meta_df = pd.read_csv(clean_dataset_paths["contracts"])
    sybil_user_ids = _get_sybil_user_ids(users_meta_df, compare_cfg)
    sybil_contract_ids = _get_sybil_contract_ids(contracts_meta_df, clean_data_df)
    zombie_contract_ids = _get_zombie_contract_ids(contracts_meta_df)
    bottom_k_ratios = sorted(
        {max(0.0, min(1.0, float(x))) for x in compare_cfg.get("BOTTOM_K_RATIOS", [0.01, 0.05, 0.10]) if float(x) > 0.0}
    )
    if not bottom_k_ratios:
        raise ValueError("BOTTOM_K_RATIOS is empty.")

    clean_our_outdir = result_root / "our_base_clean"
    _run_method("our", clean_dataset_paths, clean_our_outdir, clean_our_outdir, compare_cfg)
    clean_method_outdirs = {"our": clean_our_outdir}
    active_methods = _active_methods(compare_cfg)
    if bool(compare_cfg.get("LIGHTWEIGHT_MAX_ONLY", False)):
        preferred = [str(x) for x in compare_cfg.get("LIGHTWEIGHT_METHODS", ["our", "birank", "pagerank", "eigentrust"])]
        active_methods = [m for m in preferred if m in set(active_methods)]
    if not active_methods:
        raise ValueError("No active methods enabled for G6 comparison.")
    for method in [m for m in active_methods if m != "our"]:
        method_outdir = result_root / f"{method}_base_clean"
        _run_method(method, clean_dataset_paths, clean_our_outdir, method_outdir, compare_cfg)
        clean_method_outdirs[method] = method_outdir

    run_records: list[dict[str, Any]] = []
    repeat_ids = [int(x) for x in compare_cfg.get("ATTACK_REPEAT_IDS", [])]
    if not repeat_ids:
        n_repeats = max(int(compare_cfg.get("N_REPEATS", 1)), 1)
        repeat_ids = list(range(n_repeats))
    per_repeat_multipliers: dict[int, list[float]] = {}
    for repeat_id in repeat_ids:
        levels = _resolve_attack_levels(clean_dataset_paths, compare_cfg, int(repeat_id))
        if bool(compare_cfg.get("LIGHTWEIGHT_MAX_ONLY", False)) and levels:
            levels = [float(max(levels))]
        per_repeat_multipliers[int(repeat_id)] = levels
    total_tasks = sum(len(per_repeat_multipliers[rid]) for rid in repeat_ids) * len(active_methods)
    completed_tasks = 0
    start_ts = time.time()
    print(
        f"[G6 progress] total_tasks={total_tasks} | repeats={len(repeat_ids)} | "
        f"methods={len(active_methods)} | mode={str(compare_cfg.get('ATTACK_MODE', 'attack1'))}",
        flush=True,
    )
    for repeat_id in repeat_ids:
        for attack_multiplier in per_repeat_multipliers[int(repeat_id)]:
            level_tag = _safe_level_tag(float(attack_multiplier))
            with tempfile.TemporaryDirectory(prefix=f"g6_r{int(repeat_id):02d}_m{level_tag}_") as tmp:
                runtime_root = Path(tmp)
                attacked_dataset_paths, attack_report = _build_runtime_attacked_dataset(
                    clean_dataset_paths=clean_dataset_paths,
                    runtime_dir=runtime_root,
                    compare_cfg=compare_cfg,
                    repeat_id=int(repeat_id),
                    attack_multiplier=float(attack_multiplier),
                )
                attacked_our_outdir = runtime_root / "our_attacked"
                _run_method("our", attacked_dataset_paths, attacked_our_outdir, attacked_our_outdir, compare_cfg)
                attacked_method_outdirs = {"our": attacked_our_outdir}
                for method in [m for m in active_methods if m != "our"]:
                    method_outdir = runtime_root / f"{method}_attacked"
                    _run_method(method, attacked_dataset_paths, attacked_our_outdir, method_outdir, compare_cfg)
                    attacked_method_outdirs[method] = method_outdir

                for method in active_methods:
                    param_value = _method_param_values(compare_cfg, method)[0]
                    run_records.extend(
                        collect_method_metrics(
                            method=method,
                            clean_outdir=clean_method_outdirs[method],
                            attacked_outdir=attacked_method_outdirs[method],
                            repeat_id=int(repeat_id),
                            attack_multiplier=float(attack_multiplier),
                            attack_report=attack_report,
                            sybil_user_ids=sybil_user_ids,
                            sybil_contract_ids=sybil_contract_ids,
                            zombie_contract_ids=zombie_contract_ids,
                            k_ratios=bottom_k_ratios,
                            param_value=param_value,
                        )
                    )
                    completed_tasks += 1
                    print(
                        _progress_message(
                            done=completed_tasks,
                            total=total_tasks,
                            start_ts=start_ts,
                            repeat_id=int(repeat_id),
                            attack_mode=str(compare_cfg.get("ATTACK_MODE", "attack1")),
                            attack_multiplier=float(attack_multiplier),
                            method=str(method),
                            status="done",
                        ),
                        flush=True,
                    )

    runs_df = pd.DataFrame(run_records).sort_values(["method", "repeat_id", "attack_multiplier", "k_ratio"]).reset_index(drop=True)
    summary_df = aggregate_repeat_metrics(runs_df)
    plot_summary_df = build_plot_summary(runs_df)
    runs_df.to_csv(result_root / "g6_compare_runs.csv", index=False)
    summary_df.to_csv(result_root / "g6_compare_summary.csv", index=False)
    plot_summary_df.to_csv(result_root / "g6_compare_plot_summary.csv", index=False)
    if bool(compare_cfg.get("LIGHTWEIGHT_MAX_ONLY", False)) and not summary_df.empty:
        idx_cols = [str(x) for x in compare_cfg.get("LIGHTWEIGHT_INDEX_COLS", ["contract_spearman_rho_mean", "user_spearman_rho_mean"])]
        metric_cols = [c for c in idx_cols if c in summary_df.columns]
        lightweight_cols = ["attack_mode", "attack_multiplier", "method"] + metric_cols
        lightweight_df = summary_df[lightweight_cols].copy().sort_values(["attack_mode", "attack_multiplier", "method"]).reset_index(drop=True)
        lightweight_df.to_csv(result_root / "g6_lightweight_max_metrics.csv", index=False)
    save_plots(plot_summary_df, result_root)
    print(f"[G6 progress] finished | runs={len(runs_df)} | summary={len(summary_df)}", flush=True)
    return runs_df, summary_df


if __name__ == "__main__":
    run_compare()
