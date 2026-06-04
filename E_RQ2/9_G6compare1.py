from __future__ import annotations

import importlib.util
import json
import math
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
    "N_REPEATS": 3,
    "ATTACK_REPEAT_IDS": [],
    "ATTACK_GAS_MULTIPLIERS_BASE": [2.0, 4.0, 16.0, 1000.0],
    "ATTACK_GAS_DELTA_TARGET": 1000.0,
    "ATTACK_GAS_MAX_MULTIPLIER_CAP": 1e12,
    "ATTACK2_DROP_RATIOS": [0.01, 0.03, 0.05, 0.10, 0.20],
    "DATA_ROOT": ROOT / "output" / "A" / "18_RQ2compare_G6_v1",
    "RESULT_ROOT": ROOT / "output" / "E" / "9_G6compare_v1",
    "GENERATOR_SCRIPT": ROOT / "A_dataget" / "11_RQ2data_G6.py",
    "OUR_SCRIPT": ROOT / "E_RQ2" / "0_our.py",
    "BIRANK_SCRIPT": ROOT / "E_RQ2" / "1_birank1.py",
    "PAGERANK_SCRIPT": ROOT / "E_RQ2" / "2_pagerank2.py",
    "EIGENTRUST_SCRIPT": ROOT / "E_RQ2" / "3_eigentrust3.py",
    "GENERATOR_SEED": 121,
    "BASELINE_TAG": "_base_clean",
    "CALENDAR_BASE_START_YM": "2023-01",
    "CALENDAR_PRE_END_YM": "2025-12",
    "REUSE_EXISTING_DATASETS": True,
    "REUSE_EXISTING_RESULTS": True,
    "DISABLE_DATA_GENERATION": True,
    "SAVE_TRANSACTIONS_ALL": False,
    "SAVE_MONTHLY_PARTITIONS": False,
    "CONTRACT_COUNTS": {"Hard": 100, "Hype": 50, "Zombie": 30, "Sybil": 0},
    "RUN_OUR": True,
    "RUN_BIRANK": True,
    "RUN_PAGERANK": True,
    "RUN_EIGENTRUST": True,
    "OUR_STAGE_C_INCLUDE_MINT_EDGE": True,
    "OUR_STAGE_C_ALPHA": 0.85,
    "OUR_STAGE_C_USER_PRIOR_MODE": "stageb_u0",
    "OUR_DISABLE_U0_PRIOR": False,
    "OUR_BETA_LIST": [0.7],
    "BIRANK_ETA_LIST": [0.7],
    "PAGERANK_GAMMA_LIST": [0.7],
    "EIGENTRUST_ALPHA_LIST": [0.7],
    "OUR_PRIMARY_BETA": 0.7,
    "BIRANK_PRIMARY_ETA": 0.7,
    "PAGERANK_PRIMARY_GAMMA": 0.7,
    "EIGENTRUST_PRIMARY_ALPHA": 0.7,
    "ATTACK_MODE": "attack1",
    "ATTACK_VARIANT": "single_outlier_trade",
    "ATTACK_MODE_NOTES": {
        "attack1": "gas变化：随机单条记录gas放大",
        "attack2": "数据缺失：每个月随机丢弃部分记录",
    },
    "ATTACK_TARGET_SCOPE": "contract_event",
    "ATTACK_PARAMS": {
        "target_month": "2025-12",
        "target_event_selector": "random_trade",
        "modify_fields": ["gas"],
        "target_user_rank": 1,
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
    "birank": {"param_name": "eta", "suffix": "eta", "score_prefix": "1_birank"},
    "pagerank": {"param_name": "gamma", "suffix": "gamma", "score_prefix": "2_pagerank"},
    "eigentrust": {"param_name": "alpha", "suffix": "alpha", "score_prefix": "3_eigentrust"},
}

METHOD_ORDER = ["our", "birank", "pagerank", "eigentrust"]
METHOD_LABELS = {"our": "Our", "birank": "BiRank", "pagerank": "PageRank", "eigentrust": "EigenTrust"}
METHOD_COLORS = {"our": "#8c2d04", "birank": "#d95f0e", "pagerank": "#2171b5", "eigentrust": "#08306b"}


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
    if str(attack_mode) == "attack2":
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
    if method == "our":
        return outdir / f"{prefix}_contract_scores.csv", outdir / f"{prefix}_user_scores.csv"
    if param_value is None:
        return outdir / f"{prefix}_contract_scores.csv", outdir / f"{prefix}_user_scores.csv"
    suffix_tag = f"{meta['suffix']}{float(param_value):.2f}"
    return outdir / f"{prefix}_contract_scores_{suffix_tag}.csv", outdir / f"{prefix}_user_scores_{suffix_tag}.csv"


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


def _result_exists(method: str, outdir: Path, param_values: list[float]) -> bool:
    required = list(_score_file_paths(method, outdir, None))
    if method != "our":
        for param_value in param_values:
            required.extend(_score_file_paths(method, outdir, float(param_value)))
    return all(path.exists() for path in dict.fromkeys(required))


def _method_columns(method: str) -> tuple[str, str]:
    if method == "our":
        return "rank_C", "rank_C"
    if method in {"birank", "pagerank", "eigentrust"}:
        return "rank_contract", "rank_user"
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
        include_sybil_contracts=False,
    )
    return dataset_paths


def run_our_on_dataset(dataset_paths: dict[str, Path], outdir: Path, compare_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(OUR_MOD.CONFIG)
    stage_c_alpha = 1.0 if bool(compare_cfg.get("OUR_DISABLE_U0_PRIOR", False)) else float(compare_cfg.get("OUR_STAGE_C_ALPHA", 0.85))
    cfg.update(
        {
            "RAW_DATA_CSV": dataset_paths["data"],
            "USERS_META_CSV": dataset_paths["users"],
            "CONTRACTS_META_CSV": dataset_paths["contracts"],
            "OUTDIR": outdir,
            "STAGE_C_BETA": float(compare_cfg["OUR_PRIMARY_BETA"]),
            "STAGE_C_ALPHA": float(stage_c_alpha),
            "STAGE_C_USER_PRIOR_MODE": str(compare_cfg.get("OUR_STAGE_C_USER_PRIOR_MODE", "uniform")),
            "STAGE_C_INCLUDE_NON_ACQUISITION": bool(compare_cfg.get("OUR_STAGE_C_INCLUDE_MINT_EDGE", False)),
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


def _run_method(method: str, dataset_paths: dict[str, Path], our_outdir: Path, outdir: Path, compare_cfg: dict[str, Any]) -> bool:
    params = _method_param_values(compare_cfg, method)
    if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists(method, outdir, params):
        return False
    if method == "our":
        run_our_on_dataset(dataset_paths, outdir, compare_cfg)
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
    target_user = str(params.get("target_user", "")).strip()
    df = clean_df.copy()
    if "calendar_year_month" not in df.columns:
        df = GEN_MOD.G4_MOD._attach_month_columns(df)
    month_df = df[df["calendar_year_month"].astype(str).eq(target_month)].copy()
    if target_tx_type:
        month_df = month_df[month_df["tx_type"].astype(str).eq(target_tx_type)].copy()
    if target_user:
        month_df = month_df[month_df["buyer"].astype(str).eq(target_user)].copy()
    month_df["price_num"] = pd.to_numeric(month_df["price"], errors="coerce")
    month_df["gas_num"] = pd.to_numeric(month_df["gas"], errors="coerce")
    candidates = month_df[month_df["gas_num"].fillna(0.0) > 0].copy()
    if candidates.empty:
        raise ValueError(f"No positive-gas candidates found for target_month={target_month}, target_tx_type={target_tx_type}, target_user={target_user}")
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
    month_positive = month_df.loc[month_df["price_num"].fillna(0.0) > 0, "price_num"]
    month_median_price = float(month_positive.median()) if len(month_positive) else float(row["price_num"])
    return {
        "selector_name": selector_name,
        "target_tx_type": target_tx_type,
        "target_user": target_user,
        "target_month": target_month,
        "target_row_index": int(getattr(row, "name")),
        "target_contract_id": str(row["contract_id"]),
        "target_token_id": int(pd.to_numeric(pd.Series([row["token_id"]]), errors="coerce").fillna(-1).iloc[0]),
        "target_user_buyer": str(row.get("buyer", "")),
        "target_user_seller": str(row.get("seller", "")),
        "old_price": float(row["price_num"]),
        "old_gas": float(pd.to_numeric(pd.Series([row["gas_num"]]), errors="coerce").fillna(0.0).iloc[0]),
        "month_median_price": float(month_median_price),
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
    base_price = max(float(anchor_meta["month_median_price"]), float(anchor_meta["old_price"]), 1e-12)
    new_price = base_price * outlier_multiplier
    old_gas = float(anchor_meta["old_gas"])
    new_gas = old_gas
    if attack_variant != "single_outlier_trade":
        raise ValueError(f"Unsupported ATTACK_VARIANT: {attack_variant}")
    if "price" in modify_fields:
        attacked_df.at[row_idx, "price"] = float(new_price)
    if "gas" in modify_fields:
        new_gas = max(old_gas, 1e-12) * outlier_multiplier
        attacked_df.at[row_idx, "gas"] = float(new_gas)
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
        "old_price": float(anchor_meta["old_price"]),
        "new_price": float(new_price),
        "price_multiplier_realized": float(new_price) / max(float(anchor_meta["old_price"]), 1e-12),
        "old_gas": float(old_gas),
        "new_gas": float(new_gas),
        "gas_multiplier_realized": float(new_gas) / max(float(old_gas), 1e-12),
        "attack_multiplier": float(outlier_multiplier),
        "rows_added": 0,
        "rows_modified": 1,
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
    new_price = max(old_price, 1e-12) * float(attack_multiplier)
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
        "rows_modified": 1,
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


def _resolve_attack_multipliers(anchor_meta: dict[str, Any], compare_cfg: dict[str, Any]) -> list[float]:
    attack_params = dict(compare_cfg.get("ATTACK_PARAMS", {}))
    base = [float(x) for x in compare_cfg.get("ATTACK_GAS_MULTIPLIERS_BASE", [2.0, 4.0, 16.0, 1000.0])]
    base = sorted({x for x in base if x > 1.0})
    if not base:
        base = [2.0]
    old_gas = max(float(anchor_meta.get("old_gas", 0.0)), 1e-12)
    target_delta = float(compare_cfg.get("ATTACK_GAS_DELTA_TARGET", 1000.0))
    cap = float(compare_cfg.get("ATTACK_GAS_MAX_MULTIPLIER_CAP", 1e12))
    required_multiplier = 1.0 + target_delta / old_gas
    multipliers = list(base)
    cur = multipliers[-1]
    while cur < required_multiplier and cur < cap:
        cur = min(cur * 10.0, cap)
        multipliers.append(float(cur))
    if multipliers[-1] < required_multiplier:
        multipliers.append(float(required_multiplier))
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
            selector_params["target_user"] = str(compare_cfg.get("_ATTACK1_TARGET_USER", "")).strip()
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
            if mode in {"attack1", "attack3"}
            else int(len(clean_df)) >= int(len(attacked_df)),
            "at_least_one_row_modified": int(attack_meta["rows_modified"]) >= 1
            if mode in {"attack1", "attack3"}
            else int(attack_meta["rows_deleted"]) >= 1,
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
        attack1_params["target_user"] = str(compare_cfg.get("_ATTACK1_TARGET_USER", "")).strip()
        anchor_meta = _select_attack_anchor(clean_df, compare_cfg, repeat_id, attack_params=attack1_params)
        attacked_df, attack_meta = _apply_attack_variant(clean_df, anchor_meta, compare_cfg, repeat_id, float(attack_multiplier))
    elif mode == "attack2":
        attacked_df, attack_meta = _apply_attack2_monthly_random_drop(
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


def collect_method_metrics(
    method: str,
    clean_outdir: Path,
    attacked_outdir: Path,
    repeat_id: int,
    attack_multiplier: float,
    attack_report: dict[str, Any],
    param_value: float | None = None,
) -> dict[str, Any]:
    contract_rank_col, user_rank_col = _method_columns(method)
    clean_contract_path, clean_user_path = _score_file_paths(method, clean_outdir, param_value)
    attacked_contract_path, attacked_user_path = _score_file_paths(method, attacked_outdir, param_value)
    clean_df_c = pd.read_csv(clean_contract_path)
    clean_df_u = pd.read_csv(clean_user_path)
    attacked_df_c = pd.read_csv(attacked_contract_path)
    attacked_df_u = pd.read_csv(attacked_user_path)
    summary = dict(attack_report.get("summary") or {})
    contract_metrics = compute_ranking_correlation(clean_df_c, attacked_df_c, "contract_id", contract_rank_col)
    user_metrics = compute_ranking_correlation(clean_df_u, attacked_df_u, "user", user_rank_col)
    return {
        "method": method,
        "param_name": METHOD_PARAM_META[method]["param_name"],
        "param_value": float(param_value) if param_value is not None else float("nan"),
        "repeat_id": int(repeat_id),
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
        "user_n_entities": int(user_metrics["n_entities"]),
        "user_kendall_tau": float(user_metrics["kendall_tau"]),
        "user_spearman_rho": float(user_metrics["spearman_rho"]),
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
        "user_n_entities",
        "user_kendall_tau",
        "user_spearman_rho",
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

    if str(attack_mode) == "attack2":
        labels = [f"{int(round(v * 100))}%" for v in ordered_levels]
        ax.set_xlabel("Drop Ratio")
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
    for attack_mode in sorted(plot_summary_df["attack_mode"].astype(str).dropna().unique().tolist()):
        mode_tag = str(attack_mode).lower()
        _plot_metric_vs_intensity(
            plot_summary_df=plot_summary_df,
            attack_mode=attack_mode,
            mean_col="contract_kendall_tau_mean",
            ci_col="contract_kendall_tau_ci95",
            ylabel="Contract Kendall Tau",
            title=f"G6 {attack_mode} Contract Kendall Tau vs Intensity",
            out_path=result_root / f"g6_{mode_tag}_contract_kendall_tau_vs_intensity.png",
        )
        _plot_metric_vs_intensity(
            plot_summary_df=plot_summary_df,
            attack_mode=attack_mode,
            mean_col="contract_spearman_rho_mean",
            ci_col="contract_spearman_rho_ci95",
            ylabel="Contract Spearman Rho",
            title=f"G6 {attack_mode} Contract Spearman Rho vs Intensity",
            out_path=result_root / f"g6_{mode_tag}_contract_spearman_rho_vs_intensity.png",
        )
        _plot_metric_vs_intensity(
            plot_summary_df=plot_summary_df,
            attack_mode=attack_mode,
            mean_col="user_kendall_tau_mean",
            ci_col="user_kendall_tau_ci95",
            ylabel="User Kendall Tau",
            title=f"G6 {attack_mode} User Kendall Tau vs Intensity",
            out_path=result_root / f"g6_{mode_tag}_user_kendall_tau_vs_intensity.png",
        )
        _plot_metric_vs_intensity(
            plot_summary_df=plot_summary_df,
            attack_mode=attack_mode,
            mean_col="user_spearman_rho_mean",
            ci_col="user_spearman_rho_ci95",
            ylabel="User Spearman Rho",
            title=f"G6 {attack_mode} User Spearman Rho vs Intensity",
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
    chosen = user_df.sort_values("rank_C", ascending=True).iloc[min(target_rank - 1, len(user_df) - 1)]
    return str(chosen["user"])


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

    clean_our_outdir = result_root / "our_base_clean"
    _run_method("our", clean_dataset_paths, clean_our_outdir, clean_our_outdir, compare_cfg)
    if str(compare_cfg.get("ATTACK_MODE", "attack1")) == "attack1":
        compare_cfg["_ATTACK1_TARGET_USER"] = _resolve_attack1_target_user(clean_our_outdir, compare_cfg)
    clean_method_outdirs = {"our": clean_our_outdir}
    active_methods = _active_methods(compare_cfg)
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
        per_repeat_multipliers[int(repeat_id)] = _resolve_attack_levels(clean_dataset_paths, compare_cfg, int(repeat_id))
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
            with tempfile.TemporaryDirectory(prefix=f"g6_rep{int(repeat_id):02d}_mul{int(attack_multiplier):04d}_") as tmp:
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
                    run_records.append(
                        collect_method_metrics(
                            method=method,
                            clean_outdir=clean_method_outdirs[method],
                            attacked_outdir=attacked_method_outdirs[method],
                            repeat_id=int(repeat_id),
                            attack_multiplier=float(attack_multiplier),
                            attack_report=attack_report,
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

    runs_df = pd.DataFrame(run_records).sort_values(["method", "repeat_id", "attack_multiplier"]).reset_index(drop=True)
    summary_df = aggregate_repeat_metrics(runs_df)
    plot_summary_df = build_plot_summary(runs_df)
    runs_df.to_csv(result_root / "g6_compare_runs.csv", index=False)
    summary_df.to_csv(result_root / "g6_compare_summary.csv", index=False)
    plot_summary_df.to_csv(result_root / "g6_compare_plot_summary.csv", index=False)
    save_plots(plot_summary_df, result_root)
    print(f"[G6 progress] finished | runs={len(runs_df)} | summary={len(summary_df)}", flush=True)
    return runs_df, summary_df


if __name__ == "__main__":
    run_compare()
