from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "ATTACK_RATIO_SCALES": [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
    "TARGET_RANKS": list(range(1, 11)),
    "N_REPEATS": 10,
    "WASH_USER_GROUP_SIZE": 32,
    "DATA_ROOT": ROOT / "output" / "A" / "17_RQ2compare_G5_v1",
    "RESULT_ROOT": ROOT / "output" / "E" / "8_G5compare_v1",
    "GENERATOR_SCRIPT": ROOT / "A_dataget" / "11_RQ2data_G5.py",
    "OUR_SCRIPT": ROOT / "E_RQ2" / "0_our.py",
    "BIRANK_SCRIPT": ROOT / "E_RQ2" / "1_birank1.py",
    "PAGERANK_SCRIPT": ROOT / "E_RQ2" / "2_pagerank2.py",
    "EIGENTRUST_SCRIPT": ROOT / "E_RQ2" / "3_eigentrust3.py",
    "BASELINE_TAG": "_base_no_g5_attack",
    "GENERATOR_SEED": 11,
    "CALENDAR_BASE_START_YM": "2023-01",
    "CALENDAR_PRE_END_YM": "2025-11",
    "CALENDAR_ATTACK_YM": "2025-12",
    "CALENDAR_POST_YM": "2025-12",
    "REUSE_EXISTING_DATASETS": True,
    "REUSE_EXISTING_RESULTS": False,
    "CONTRACT_COUNTS": {"Hard": 100, "Hype": 50, "Zombie": 30, "Sybil": 0},
    "RUN_OUR": True,
    "RUN_BIRANK": True,
    "RUN_PAGERANK": True,
    "RUN_EIGENTRUST": True,
    "OUR_STAGE_C_INCLUDE_MINT_EDGE": True,
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

METHOD_ORDER = ["our", "birank", "pagerank", "eigentrust"]
METHOD_LABELS = {"our": "Our", "birank": "BiRank", "pagerank": "PageRank", "eigentrust": "EigenTrust"}
METHOD_COLORS = {"our": "#8c2d04", "birank": "#d95f0e", "pagerank": "#2171b5", "eigentrust": "#08306b"}


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEN_MOD = _load_module("g5_generator_compare", CONFIG["GENERATOR_SCRIPT"])
OUR_MOD = _load_module("our_pipeline_g5_compare", CONFIG["OUR_SCRIPT"])
BIRANK_MOD = _load_module("birank_g5_compare", CONFIG["BIRANK_SCRIPT"])
PAGERANK_MOD = _load_module("pagerank_g5_compare", CONFIG["PAGERANK_SCRIPT"])
EIGENTRUST_MOD = _load_module("eigentrust_g5_compare", CONFIG["EIGENTRUST_SCRIPT"])


def _g5_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "data": data_dir / "RQdataG5.csv",
        "users": data_dir / "RQdataG5_users.csv",
        "contracts": data_dir / "RQdataG5_contracts.csv",
        "role_map": data_dir / "RQdataG5_user_role_map.csv",
        "report": data_dir / "g5_report.json",
        "config": data_dir / "g5_config.json",
    }


def _all_exist(paths: dict[str, Path], keys: list[str]) -> bool:
    return all(paths[k].exists() for k in keys)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


def _ratio_tag(attack_ratio: float) -> str:
    return f"p{int(round(float(attack_ratio) * 100)):03d}"


def _dataset_tag(attack_ratio: float, repeat_id: int) -> str:
    return f"{_ratio_tag(attack_ratio)}_rep{int(repeat_id):02d}"


def _selection_tag(method: str, target_rank: int) -> str:
    return f"{method}_rank{int(target_rank):02d}"


def _sample_seed(attack_ratio: float, repeat_id: int) -> int:
    return int(CONFIG["GENERATOR_SEED"]) * 100_000 + int(round(float(attack_ratio) * 10_000)) * 100 + int(repeat_id)


def _selection_seed(attack_ratio: float, repeat_id: int, method: str, target_rank: int) -> int:
    method_offset = METHOD_ORDER.index(method) + 1
    return _sample_seed(attack_ratio, repeat_id) * 1000 + method_offset * 100 + int(target_rank)


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


def _method_columns(method: str) -> tuple[str, str, str, str]:
    if method == "our":
        return "rank_C", "rank_C", "cC", "uC"
    if method == "birank":
        return "rank_contract", "rank_user", "c_birank", "u_birank"
    if method == "pagerank":
        return "rank_contract", "rank_user", "c_pagerank", "u_pagerank"
    if method == "eigentrust":
        return "rank_contract", "rank_user", "c_eigentrust", "u_eigentrust"
    raise ValueError(f"Unknown method: {method}")


def _normalized_rank(series: pd.Series, total_n: int) -> pd.Series:
    if total_n <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series.astype(float) - 1.0) / float(total_n - 1)


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else float("nan")


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


def ensure_baseline_dataset(base_dir: Path, compare_cfg: dict[str, Any]) -> dict[str, Path]:
    dataset_paths = _g5_paths(base_dir)
    needed = ["data", "users", "contracts", "role_map", "report", "config"]
    if bool(compare_cfg["REUSE_EXISTING_DATASETS"]) and _all_exist(dataset_paths, needed):
        return dataset_paths
    contract_profiles, user_profiles, _, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, compare_cfg)
    GEN_MOD.run_g5_baseline(
        contract_profiles=contract_profiles,
        user_profiles=user_profiles,
        save_dir=base_dir,
        horizon_months=GEN_MOD._months_between_inclusive(compare_cfg["CALENDAR_BASE_START_YM"], compare_cfg["CALENDAR_POST_YM"]),
        end_date=GEN_MOD._month_end_dt(compare_cfg["CALENDAR_POST_YM"]),
        seed=int(compare_cfg["GENERATOR_SEED"]),
        include_sybil_contracts=False,
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


def _run_method(method: str, dataset_paths: dict[str, Path], our_outdir: Path, outdir: Path, compare_cfg: dict[str, Any]) -> None:
    params = _method_param_values(compare_cfg, method)
    if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists(method, outdir, params):
        return
    if method == "our":
        run_our_on_dataset(dataset_paths, outdir, compare_cfg)
        return
    if method == "birank":
        run_birank_on_dataset(our_outdir, outdir, compare_cfg)
        return
    if method == "pagerank":
        run_pagerank_on_dataset(our_outdir, outdir, compare_cfg)
        return
    if method == "eigentrust":
        run_eigentrust_on_dataset(dataset_paths, our_outdir, outdir, compare_cfg)
        return
    raise ValueError(f"Unknown method: {method}")


def _eligible_benign_users(users_df: pd.DataFrame) -> list[str]:
    blocked_prefixes = ("g1_", "g2_", "g3_", "g3a_", "g4_", "g5_", "attacker")
    return [
        str(user)
        for user in users_df["user"].astype(str).tolist()
        if not any(str(user).startswith(prefix) for prefix in blocked_prefixes)
    ]


def _write_pre_snapshot(prepared: dict[str, Any], pre_dir: Path, compare_cfg: dict[str, Any]) -> dict[str, Path]:
    paths = _g5_paths(pre_dir)
    needed = ["data", "users", "contracts", "role_map", "report", "config"]
    if bool(compare_cfg["REUSE_EXISTING_DATASETS"]) and _all_exist(paths, needed):
        return paths
    pre_df = prepared["pre_data_df"].copy()
    report = GEN_MOD._build_report(
        data_df=pre_df,
        meta=prepared,
        wash_users=[],
        target_contract_id=None,
    )
    config = {
        "stage": "m11_pre",
        "attack_ratio": float(prepared["attack_ratio"]),
        "repeat_id": int(prepared["repeat_id"]),
        "sample_seed": int(prepared["sample_seed"]),
        "calendar_year_month": str(compare_cfg["CALENDAR_PRE_END_YM"]),
    }
    return GEN_MOD._save_snapshot(
        save_dir=pre_dir,
        data_df=pre_df,
        users_df=prepared["users_df"],
        contracts_df=prepared["contracts_df"],
        role_map_df=prepared["role_map_df"],
        transactions_all_df=prepared["transactions_all_df"],
        report=report,
        config=config,
    )


def _select_wash_users(
    method: str,
    pre_dataset_paths: dict[str, Path],
    pre_outdir: Path,
    sample_seed: int,
    compare_cfg: dict[str, Any],
    param_value: float | None = None,
) -> dict[str, Any]:
    _, user_rank_col, _, _ = _method_columns(method)
    _, user_path = _score_file_paths(method, pre_outdir, param_value)
    df_u = pd.read_csv(user_path)
    users_meta = pd.read_csv(pre_dataset_paths["users"])
    eligible = set(_eligible_benign_users(users_meta))
    ranked = df_u[df_u["user"].astype(str).isin(eligible)].copy()
    ranked = ranked.sort_values([user_rank_col, "user"], ascending=[True, True]).reset_index(drop=True)
    if ranked.empty:
        raise ValueError(f"No eligible benign users in pre ranking for method={method}.")
    pool_size = max(int(math.ceil(len(ranked) * 0.05)), int(compare_cfg["WASH_USER_GROUP_SIZE"]))
    candidate_pool = ranked.tail(pool_size).copy()
    if len(candidate_pool) < int(compare_cfg["WASH_USER_GROUP_SIZE"]):
        raise ValueError(
            f"Wash candidate pool too small for method={method}: "
            f"pool={len(candidate_pool)}, required={int(compare_cfg['WASH_USER_GROUP_SIZE'])}"
        )
    rng = np.random.default_rng(int(sample_seed))
    picked = rng.choice(
        candidate_pool["user"].astype(str).to_numpy(dtype=object),
        size=int(compare_cfg["WASH_USER_GROUP_SIZE"]),
        replace=False,
    )
    return {
        "wash_users": sorted(str(x) for x in picked.tolist()),
        "wash_candidate_pool_size": int(len(candidate_pool)),
    }


def _select_target_contract(
    method: str,
    prepared: dict[str, Any],
    pre_dataset_paths: dict[str, Path],
    pre_outdir: Path,
    requested_target_rank: int,
    wash_users: list[str],
    param_value: float | None = None,
) -> dict[str, Any]:
    contract_rank_col, _, contract_score_col, _ = _method_columns(method)
    contract_path, _ = _score_file_paths(method, pre_outdir, param_value)
    df_c = pd.read_csv(contract_path)
    contracts_meta = pd.read_csv(pre_dataset_paths["contracts"])
    merged = df_c.merge(contracts_meta[["contract_id", "category", "quality"]], on="contract_id", how="left")
    merged["contract_id"] = merged["contract_id"].astype(str)
    merged["category"] = merged["category"].fillna("").astype(str)
    merged = merged.sort_values([contract_rank_col, "contract_id"], ascending=[True, True]).reset_index(drop=True)
    start_idx = max(0, int(requested_target_rank) - 1)
    for row in merged.iloc[start_idx:].itertuples(index=False):
        if str(row.category) == "Sybil":
            continue
        holder_rows = GEN_MOD._latest_non_wash_holders(prepared["pre_data_df"], str(row.contract_id), list(wash_users))
        if not holder_rows:
            continue
        return {
            "target_contract_id": str(row.contract_id),
            "target_contract_category": str(row.category),
            "target_contract_quality": int(getattr(row, "quality")),
            "requested_target_rank": int(requested_target_rank),
            "resolved_target_rank": int(getattr(row, contract_rank_col)),
            "rank_selection_fallback_used": bool(int(getattr(row, contract_rank_col)) != int(requested_target_rank)),
            "target_contract_score_pre": float(getattr(row, contract_score_col)),
        }
    raise ValueError(
        f"No eligible target contract with tradable token for method={method}, requested_target_rank={requested_target_rank}"
    )


def _write_post_snapshot(
    prepared: dict[str, Any],
    wash_meta: dict[str, Any],
    target_meta: dict[str, Any],
    post_dir: Path,
    selection_method: str,
    compare_cfg: dict[str, Any],
    attack_ratio: float,
) -> dict[str, Path]:
    paths = _g5_paths(post_dir)
    needed = ["data", "users", "contracts", "role_map", "report", "config"]
    if bool(compare_cfg["REUSE_EXISTING_DATASETS"]) and _all_exist(paths, needed):
        return paths
    payload = GEN_MOD.build_g5_attack_snapshot_from_pre(
        prepared=prepared,
        wash_users=[str(x) for x in wash_meta["wash_users"]],
        target_contract_id=str(target_meta["target_contract_id"]),
        requested_target_rank=int(target_meta["requested_target_rank"]),
        resolved_target_rank=int(target_meta["resolved_target_rank"]),
        wash_candidate_pool_size=int(wash_meta["wash_candidate_pool_size"]),
        selection_method=str(selection_method),
        attack_ratio=float(attack_ratio),
        save_dir=post_dir,
    )
    return payload["paths"]


def collect_method_metrics(
    method: str,
    pre_dataset_paths: dict[str, Path],
    pre_outdir: Path,
    post_dataset_paths: dict[str, Path],
    post_outdir: Path,
    attack_ratio: float,
    repeat_id: int,
    wash_meta: dict[str, Any],
    target_meta: dict[str, Any],
    param_value: float | None = None,
) -> dict[str, Any]:
    contract_rank_col, user_rank_col, contract_score_col, user_score_col = _method_columns(method)
    pre_contract_path, pre_user_path = _score_file_paths(method, pre_outdir, param_value)
    post_contract_path, post_user_path = _score_file_paths(method, post_outdir, param_value)
    pre_df_c = pd.read_csv(pre_contract_path)
    pre_df_u = pd.read_csv(pre_user_path)
    post_df_c = pd.read_csv(post_contract_path)
    post_df_u = pd.read_csv(post_user_path)
    post_report = _read_json(post_dataset_paths["report"])
    post_summary = dict(post_report.get("summary") or {})

    wash_users = set((post_summary.get("wash_users") or wash_meta.get("wash_users") or []))
    target_contract_id = str(target_meta["target_contract_id"])
    pre_wash_users = pre_df_u[pre_df_u["user"].astype(str).isin(wash_users)].copy()
    post_wash_users = post_df_u[post_df_u["user"].astype(str).isin(wash_users)].copy()
    pre_target_contract = pre_df_c[pre_df_c["contract_id"].astype(str).eq(target_contract_id)].copy()
    post_target_contract = post_df_c[post_df_c["contract_id"].astype(str).eq(target_contract_id)].copy()

    pre_wash_users["norm_rank"] = _normalized_rank(pre_wash_users[user_rank_col], len(pre_df_u))
    post_wash_users["norm_rank"] = _normalized_rank(post_wash_users[user_rank_col], len(post_df_u))
    pre_target_contract["norm_rank"] = _normalized_rank(pre_target_contract[contract_rank_col], len(pre_df_c))
    post_target_contract["norm_rank"] = _normalized_rank(post_target_contract[contract_rank_col], len(post_df_c))

    wash_user_rank_pre = _safe_mean(pre_wash_users[user_rank_col])
    wash_user_rank_post = _safe_mean(post_wash_users[user_rank_col])
    target_contract_rank_pre = _safe_mean(pre_target_contract[contract_rank_col])
    target_contract_rank_post = _safe_mean(post_target_contract[contract_rank_col])
    wash_user_score_pre = _safe_mean(pre_wash_users[user_score_col])
    wash_user_score_post = _safe_mean(post_wash_users[user_score_col])
    target_contract_score_pre = _safe_mean(pre_target_contract[contract_score_col])
    target_contract_score_post = _safe_mean(post_target_contract[contract_score_col])

    return {
        "method": method,
        "param_name": METHOD_PARAM_META[method]["param_name"],
        "param_value": float(param_value) if param_value is not None else float("nan"),
        "attack_ratio": float(attack_ratio),
        "repeat_id": int(repeat_id),
        "target_rank": int(target_meta["requested_target_rank"]),
        "requested_target_rank": int(target_meta["requested_target_rank"]),
        "resolved_target_rank": int(target_meta["resolved_target_rank"]),
        "wash_user_group_size": int(post_summary.get("wash_user_group_size", len(wash_users))),
        "month_tag": "m12_attack",
        "calendar_year_month": str(post_summary.get("calendar_year_month", "")),
        "sample_seed": int(post_summary.get("sample_seed", _sample_seed(attack_ratio, repeat_id))),
        "selection_method": str(method),
        "target_contract_id": target_contract_id,
        "target_contract_category": str(target_meta["target_contract_category"]),
        "rank_selection_fallback_used": bool(target_meta["rank_selection_fallback_used"]),
        "wash_user_rank_pre_mean": wash_user_rank_pre,
        "wash_user_rank_post_mean": wash_user_rank_post,
        "wash_user_rank_delta": wash_user_rank_post - wash_user_rank_pre,
        "wash_user_norm_rank_delta": _safe_mean(post_wash_users["norm_rank"]) - _safe_mean(pre_wash_users["norm_rank"]),
        "wash_user_score_delta": wash_user_score_post - wash_user_score_pre,
        "target_contract_rank_pre": target_contract_rank_pre,
        "target_contract_rank_post": target_contract_rank_post,
        "target_contract_rank_delta": target_contract_rank_post - target_contract_rank_pre,
        "target_contract_norm_rank_delta": _safe_mean(post_target_contract["norm_rank"]) - _safe_mean(pre_target_contract["norm_rank"]),
        "target_contract_score_delta": target_contract_score_post - target_contract_score_pre,
        "baseline_dec_events": int(post_summary.get("baseline_dec_events", 0)),
        "target_wash_trade_count": int(post_summary.get("target_wash_trade_count", 0)),
        "actual_wash_trade_count": int(post_summary.get("actual_wash_trade_count", 0)),
        "entry_price": float(post_summary.get("entry_price", 0.0)),
        "final_exit_price": float(post_summary.get("final_exit_price", 0.0)),
        "all_wash_users_participated": bool(post_summary.get("all_wash_users_participated", False)),
        "wash_candidate_pool_size": int(post_summary.get("wash_candidate_pool_size", wash_meta["wash_candidate_pool_size"])),
    }


def aggregate_repeat_metrics(runs_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "wash_user_group_size",
        "rank_selection_fallback_used",
        "wash_user_rank_delta",
        "wash_user_norm_rank_delta",
        "wash_user_score_delta",
        "target_contract_rank_delta",
        "target_contract_norm_rank_delta",
        "target_contract_score_delta",
        "baseline_dec_events",
        "target_wash_trade_count",
        "actual_wash_trade_count",
        "entry_price",
        "final_exit_price",
        "all_wash_users_participated",
        "wash_candidate_pool_size",
    ]
    grouped = runs_df.groupby(["method", "param_name", "param_value", "attack_ratio", "target_rank"], dropna=False)
    records: list[dict[str, Any]] = []
    for key, sub in grouped:
        method, param_name, param_value, attack_ratio, target_rank = key
        record = {
            "method": method,
            "param_name": param_name,
            "param_value": float(param_value),
            "attack_ratio": float(attack_ratio),
            "target_rank": int(target_rank),
            "calendar_year_month": str(sub["calendar_year_month"].iloc[0]),
            "n_repeats_actual": int(sub["repeat_id"].nunique()),
            "target_contract_category_mode": str(sub["target_contract_category"].mode().iloc[0]) if len(sub["target_contract_category"].mode()) else "",
        }
        for col in metric_cols:
            mean = float(sub[col].mean()) if len(sub) else float("nan")
            std = float(sub[col].std(ddof=0)) if len(sub) else float("nan")
            record[f"{col}_mean"] = mean
            record[f"{col}_std"] = std
            record[f"{col}_ci95"] = 1.96 * std / math.sqrt(max(record["n_repeats_actual"], 1))
        records.append(record)
    return pd.DataFrame(records).sort_values(["method", "attack_ratio", "target_rank"]).reset_index(drop=True)


def build_plot_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    per_repeat = (
        runs_df.groupby(["method", "param_name", "param_value", "attack_ratio", "repeat_id"], dropna=False)[
            [
                "wash_user_rank_delta",
                "target_contract_rank_delta",
                "all_wash_users_participated",
                "actual_wash_trade_count",
            ]
        ]
        .mean()
        .reset_index()
    )
    grouped = per_repeat.groupby(["method", "param_name", "param_value", "attack_ratio"], dropna=False)
    records: list[dict[str, Any]] = []
    for key, sub in grouped:
        method, param_name, param_value, attack_ratio = key
        record = {
            "method": method,
            "param_name": param_name,
            "param_value": float(param_value),
            "attack_ratio": float(attack_ratio),
            "n_repeats_actual": int(sub["repeat_id"].nunique()),
            "calendar_year_month": str(CONFIG["CALENDAR_ATTACK_YM"]),
        }
        for col in [
            "wash_user_rank_delta",
            "target_contract_rank_delta",
            "all_wash_users_participated",
            "actual_wash_trade_count",
        ]:
            mean = float(sub[col].mean()) if len(sub) else float("nan")
            std = float(sub[col].std(ddof=0)) if len(sub) else float("nan")
            record[f"{col}_mean"] = mean
            record[f"{col}_std"] = std
            record[f"{col}_ci95"] = 1.96 * std / math.sqrt(max(record["n_repeats_actual"], 1))
        records.append(record)
    return pd.DataFrame(records).sort_values(["method", "attack_ratio"]).reset_index(drop=True)


def _plot_monthly_delta(
    plot_summary_df: pd.DataFrame,
    attack_ratio: float,
    mean_col: str,
    ci_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    sub_df = plot_summary_df[np.isclose(plot_summary_df["attack_ratio"].astype(float), float(attack_ratio))].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    x = np.array([0.0, 1.0], dtype=float)
    for method in METHOD_ORDER:
        sub = sub_df[sub_df["method"].astype(str).eq(method)]
        if sub.empty:
            continue
        mean_val = float(sub[mean_col].iloc[0])
        ci_val = float(sub[ci_col].fillna(0.0).iloc[0])
        y = np.array([0.0, mean_val], dtype=float)
        lower = np.array([0.0, mean_val - ci_val], dtype=float)
        upper = np.array([0.0, mean_val + ci_val], dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, lower, upper, color=METHOD_COLORS[method], alpha=0.16)
    ax.set_xticks(x)
    ax.set_xticklabels(["2025-11", "2025-12"])
    ax.set_xlabel("Calendar Month")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_attack_ratio(
    plot_summary_df: pd.DataFrame,
    mean_col: str,
    ci_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ordered_ratios = plot_summary_df.sort_values("attack_ratio")["attack_ratio"].drop_duplicates().tolist()
    x_map = {float(r): idx for idx, r in enumerate(ordered_ratios)}
    for method in METHOD_ORDER:
        sub = plot_summary_df[plot_summary_df["method"].astype(str).eq(method)].sort_values("attack_ratio")
        if sub.empty:
            continue
        x = sub["attack_ratio"].astype(float).map(x_map).to_numpy(dtype=float)
        mean = sub[mean_col].to_numpy(dtype=float)
        ci = sub[ci_col].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", linewidth=2.2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, mean - ci, mean + ci, color=METHOD_COLORS[method], alpha=0.16)
    labels = [f"{int(round(float(x) * 100))}%" for x in ordered_ratios]
    ax.set_xticks(np.arange(len(labels), dtype=float))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Attack Intensity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_plots(plot_summary_df: pd.DataFrame, result_root: Path) -> None:
    _plot_monthly_delta(
        plot_summary_df=plot_summary_df,
        attack_ratio=0.01,
        mean_col="wash_user_rank_delta_mean",
        ci_col="wash_user_rank_delta_ci95",
        ylabel="Wash User Rank Delta",
        title="G5 Wash User Rank Delta (1% Attack Intensity, 95% CI)",
        out_path=result_root / "g5_wash_user_rank_delta_monthly_p001.png",
    )
    _plot_monthly_delta(
        plot_summary_df=plot_summary_df,
        attack_ratio=0.01,
        mean_col="target_contract_rank_delta_mean",
        ci_col="target_contract_rank_delta_ci95",
        ylabel="Target Contract Rank Delta",
        title="G5 Target Contract Rank Delta (1% Attack Intensity, 95% CI)",
        out_path=result_root / "g5_target_contract_rank_delta_monthly_p001.png",
    )
    _plot_attack_ratio(
        plot_summary_df=plot_summary_df,
        mean_col="wash_user_rank_delta_mean",
        ci_col="wash_user_rank_delta_ci95",
        ylabel="Wash User Rank Delta",
        title="G5 Wash User Rank Delta vs Attack Intensity",
        out_path=result_root / "g5_attack_intensity_overview_wash_user.png",
    )
    _plot_attack_ratio(
        plot_summary_df=plot_summary_df,
        mean_col="target_contract_rank_delta_mean",
        ci_col="target_contract_rank_delta_ci95",
        ylabel="Target Contract Rank Delta",
        title="G5 Target Contract Rank Delta vs Attack Intensity",
        out_path=result_root / "g5_attack_intensity_overview_target_contract.png",
    )


def run_compare(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compare_cfg = dict(CONFIG)
    if config:
        compare_cfg.update(config)

    data_root = Path(compare_cfg["DATA_ROOT"])
    result_root = Path(compare_cfg["RESULT_ROOT"])
    data_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    base_dir = data_root / str(compare_cfg["BASELINE_TAG"])
    base_paths = ensure_baseline_dataset(base_dir, compare_cfg)

    run_records: list[dict[str, Any]] = []
    active_methods = _active_methods(compare_cfg)
    for attack_ratio in [float(x) for x in compare_cfg["ATTACK_RATIO_SCALES"]]:
        for repeat_id in range(int(compare_cfg["N_REPEATS"])):
            dataset_tag = _dataset_tag(attack_ratio, repeat_id)
            dataset_root = data_root / dataset_tag
            pre_dir = dataset_root / "m11_pre"
            prepared = GEN_MOD.prepare_g5_pre_attack_from_base(
                base_data_csv=base_paths["data"],
                base_users_csv=base_paths["users"],
                base_contracts_csv=base_paths["contracts"],
                attack_ratio=float(attack_ratio),
                repeat_id=int(repeat_id),
                sample_seed=int(_sample_seed(attack_ratio, repeat_id)),
                calendar_pre_ym=str(compare_cfg["CALENDAR_PRE_END_YM"]),
                calendar_attack_ym=str(compare_cfg["CALENDAR_ATTACK_YM"]),
            )
            pre_dataset_paths = _write_pre_snapshot(prepared, pre_dir, compare_cfg)

            pre_our_outdir = result_root / f"our_{dataset_tag}_m11_pre"
            _run_method("our", pre_dataset_paths, pre_our_outdir, pre_our_outdir, compare_cfg)
            pre_method_outdirs = {"our": pre_our_outdir}
            for method in [m for m in active_methods if m != "our"]:
                method_outdir = result_root / f"{method}_{dataset_tag}_m11_pre"
                _run_method(method, pre_dataset_paths, pre_our_outdir, method_outdir, compare_cfg)
                pre_method_outdirs[method] = method_outdir

            for method in active_methods:
                param_value = _method_param_values(compare_cfg, method)[0]
                wash_meta = _select_wash_users(
                    method=method,
                    pre_dataset_paths=pre_dataset_paths,
                    pre_outdir=pre_method_outdirs[method],
                    sample_seed=int(_selection_seed(attack_ratio, repeat_id, method, 0)),
                    compare_cfg=compare_cfg,
                    param_value=param_value,
                )
                for target_rank in [int(x) for x in compare_cfg["TARGET_RANKS"]]:
                    target_meta = _select_target_contract(
                        method=method,
                        prepared=prepared,
                        pre_dataset_paths=pre_dataset_paths,
                        pre_outdir=pre_method_outdirs[method],
                        requested_target_rank=int(target_rank),
                        wash_users=[str(x) for x in wash_meta["wash_users"]],
                        param_value=param_value,
                    )
                    post_dir = dataset_root / _selection_tag(method, target_rank) / "m12_attack"
                    post_dataset_paths = _write_post_snapshot(
                        prepared=prepared,
                        wash_meta=wash_meta,
                        target_meta=target_meta,
                        post_dir=post_dir,
                        selection_method=method,
                        compare_cfg=compare_cfg,
                        attack_ratio=float(attack_ratio),
                    )
                    post_our_outdir = result_root / f"our_{dataset_tag}_{_selection_tag(method, target_rank)}_m12_attack"
                    if method == "our":
                        _run_method("our", post_dataset_paths, post_our_outdir, post_our_outdir, compare_cfg)
                        post_outdir = post_our_outdir
                    else:
                        _run_method("our", post_dataset_paths, post_our_outdir, post_our_outdir, compare_cfg)
                        post_outdir = result_root / f"{method}_{dataset_tag}_{_selection_tag(method, target_rank)}_m12_attack"
                        _run_method(method, post_dataset_paths, post_our_outdir, post_outdir, compare_cfg)

                    run_records.append(
                        collect_method_metrics(
                            method=method,
                            pre_dataset_paths=pre_dataset_paths,
                            pre_outdir=pre_method_outdirs[method],
                            post_dataset_paths=post_dataset_paths,
                            post_outdir=post_outdir,
                            attack_ratio=float(attack_ratio),
                            repeat_id=int(repeat_id),
                            wash_meta=wash_meta,
                            target_meta=target_meta,
                            param_value=param_value,
                        )
                    )

    runs_df = pd.DataFrame(run_records).sort_values(
        ["method", "attack_ratio", "target_rank", "repeat_id"]
    ).reset_index(drop=True)
    summary_df = aggregate_repeat_metrics(runs_df)
    plot_summary_df = build_plot_summary(runs_df)
    runs_df.to_csv(result_root / "g5_compare_runs.csv", index=False)
    summary_df.to_csv(result_root / "g5_compare_summary.csv", index=False)
    plot_summary_df.to_csv(result_root / "g5_compare_plot_summary.csv", index=False)
    save_plots(plot_summary_df, result_root)
    return runs_df, summary_df, plot_summary_df


if __name__ == "__main__":
    run_compare()
