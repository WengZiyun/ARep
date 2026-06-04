from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    "REPUTATION_BUCKETS": [(70, 80)],
    "N_REPEATS": 10,
    "MONTH_TAGS": ["m00_pre", "m01_attack", "m02_post1", "m03_post2", "m04_post3", "m05_post4", "m06_post5", "m07_post6"],
    "EVAL_MONTH_TAGS": ["m00_pre", "m01_attack", "m02_post1", "m04_post3", "m07_post6"],
    "CALENDAR_BASE_START_YM": "2023-01",
    "CALENDAR_PRE_END_YM": "2025-05",
    "CALENDAR_ATTACK_YM": "2025-06",
    "CALENDAR_FINAL_YM": "2025-12",
    "CALENDAR_MONTH_TAG_MAP": {
        "m00_pre": "2025-05",
        "m01_attack": "2025-06",
        "m02_post1": "2025-07",
        "m03_post2": "2025-08",
        "m04_post3": "2025-09",
        "m05_post4": "2025-10",
        "m06_post5": "2025-11",
        "m07_post6": "2025-12",
    },
    "DATA_ROOT": ROOT / "output" / "A" / "15_RQ2compare_G3A2_v3",
    "RESULT_ROOT": ROOT / "output" / "E" / "6_G3Acompare2_methodsplit_v1",
    "GENERATOR_SCRIPT": ROOT / "A_dataget" / "11_RQ2data_G3A2.py",
    "GENERATOR_MODE": "g3a2",
    "OUR_SCRIPT": ROOT / "E_RQ2" / "0_our.py",
    "BIRANK_SCRIPT": ROOT / "E_RQ2" / "1_birank1.py",
    "PAGERANK_SCRIPT": ROOT / "E_RQ2" / "2_pagerank2.py",
    "EIGENTRUST_SCRIPT": ROOT / "E_RQ2" / "3_eigentrust3.py",
    "BASELINE_TAG": "_base_no_g3a_sybil_trade",
    "GENERATOR_SEED": 11,
    "G3A_TARGET_CONTRACT_LIMIT": 10,
    "G3A_POST_MONTHS": 6,
    "SNAPSHOT_INTERVAL_DAYS": 30,
    "REUSE_EXISTING_DATASETS": True,
    "REUSE_EXISTING_RESULTS": False,
    "CONTRACT_COUNTS": {"Hard": 100, "Hype": 50, "Zombie": 30, "Sybil": 0},
    "RUN_OUR": True,
    "OUR_STAGE_C_INCLUDE_MINT_EDGE": True,
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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEN_MOD = _load_module("g3a_generator_compare_v2", CONFIG["GENERATOR_SCRIPT"])
OUR_MOD = _load_module("our_pipeline_g3a_compare_v2", CONFIG["OUR_SCRIPT"])
BIRANK_MOD = _load_module("birank_g3a_compare_v2", CONFIG["BIRANK_SCRIPT"])
PAGERANK_MOD = _load_module("pagerank_g3a_compare_v2", CONFIG["PAGERANK_SCRIPT"])
EIGENTRUST_MOD = _load_module("eigentrust_g3a_compare_v2", CONFIG["EIGENTRUST_SCRIPT"])


def _dataset_paths(data_dir: Path) -> dict[str, Path]:
    if str(CONFIG.get("GENERATOR_MODE", "g3a")).lower() == "g3a2":
        return {
            "data": data_dir / "RQdataG3A2.csv",
            "users": data_dir / "RQdataG3A2_users.csv",
            "contracts": data_dir / "RQdataG3A2_contracts.csv",
            "report": data_dir / "g3a2_report.json",
            "config": data_dir / "g3a2_config.json",
        }
    return {
        "data": data_dir / "RQdataG3A.csv",
        "users": data_dir / "RQdataG3A_users.csv",
        "contracts": data_dir / "RQdataG3A_contracts.csv",
        "report": data_dir / "g3a_report.json",
        "config": data_dir / "g3a_config.json",
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
    if method == "our":
        return outdir / f"{prefix}_contract_scores.csv", outdir / f"{prefix}_user_scores.csv"
    if param_value is None:
        return outdir / f"{prefix}_contract_scores.csv", outdir / f"{prefix}_user_scores.csv"
    suffix_tag = f"{meta['suffix']}{float(param_value):.2f}"
    return (
        outdir / f"{prefix}_contract_scores_{suffix_tag}.csv",
        outdir / f"{prefix}_user_scores_{suffix_tag}.csv",
    )


def _result_exists(method: str, outdir: Path, param_values: list[float]) -> bool:
    required = list(_score_file_paths(method, outdir, None))
    for param_value in param_values:
        required.extend(_score_file_paths(method, outdir, float(param_value)))
    unique_required = list(dict.fromkeys(required))
    return all(path.exists() for path in unique_required)


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


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else float("nan")


def _normalized_rank(series: pd.Series, total_n: int) -> pd.Series:
    if total_n <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series.astype(float) - 1.0) / float(total_n - 1)


def _bucket_label(bucket_start: int, bucket_end: int) -> str:
    return f"bottom_{int(bucket_start):02d}_{int(bucket_end):02d}"


def _bucket_dataset_tag(bucket_start: int, bucket_end: int, repeat_id: int) -> str:
    return f"{_bucket_label(bucket_start, bucket_end)}_rep{int(repeat_id):02d}"


def _bucket_sample_seed(bucket_start: int, bucket_end: int, repeat_id: int) -> int:
    return int(CONFIG["GENERATOR_SEED"]) * 100_000 + int(bucket_start) * 1_000 + int(bucket_end) * 10 + int(repeat_id)


def _result_tag(bucket_start: int, bucket_end: int, repeat_id: int, month_tag: str) -> str:
    return f"{_bucket_dataset_tag(bucket_start, bucket_end, repeat_id)}_{month_tag}"


def _month_index(month_tag: str) -> int:
    return int(CONFIG["MONTH_TAGS"].index(month_tag))


def _calendar_year_month(compare_cfg: dict[str, Any], month_tag: str) -> str:
    return str(dict(compare_cfg["CALENDAR_MONTH_TAG_MAP"])[month_tag])


def _calendar_month_index_from_attack(compare_cfg: dict[str, Any], month_tag: str) -> int:
    return _month_index(month_tag) - 1


def _eval_month_tags(compare_cfg: dict[str, Any]) -> list[str]:
    raw = compare_cfg.get("EVAL_MONTH_TAGS", compare_cfg["MONTH_TAGS"])
    tags = [str(tag) for tag in raw]
    valid = set(str(tag) for tag in compare_cfg["MONTH_TAGS"])
    return [tag for tag in tags if tag in valid]


def _eligible_normal_users(user_meta_df: pd.DataFrame) -> list[str]:
    users = []
    for user in user_meta_df["user"].astype(str).tolist():
        if user == "attacker":
            continue
        if user.startswith("g1_sybil_") or user.startswith("g2_") or user.startswith("g3_") or user.startswith("g3a_"):
            continue
        users.append(user)
    return users


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
    report = _read_json(dataset_paths["report"])
    config = _read_json(dataset_paths["config"])
    return (
        _count_sybil_trade_rows(dataset_paths["data"]) == 0
        and bool((report.get("checks") or {}).get("no_sybil_trade_in_base", False))
        and str(config.get("calendar_base_start_ym", "")) == str(CONFIG["CALENDAR_BASE_START_YM"])
        and str(config.get("calendar_final_ym", "")) == str(CONFIG["CALENDAR_FINAL_YM"])
    )


def _sequence_is_valid(
    data_root: Path,
    bucket_start: int,
    bucket_end: int,
    repeat_id: int,
    month_tags: list[str],
    compare_cfg: dict[str, Any],
    selected_user: str | None = None,
) -> bool:
    sampled_rebels = None
    for month_tag in month_tags:
        paths = _dataset_paths(data_root / month_tag)
        if not _all_exist(paths, ["data", "users", "contracts", "report", "config"]):
            return False
        report = _read_json(paths["report"])
        summary = dict(report.get("summary") or {})
        checks = dict(report.get("checks") or {})
        samples = dict(report.get("samples") or {})
        if int(summary.get("repeat_id", -1)) != int(repeat_id):
            return False
        if int(summary.get("reputation_bucket_from_bottom_pct_start", -1)) != int(bucket_start):
            return False
        if int(summary.get("reputation_bucket_from_bottom_pct_end", -1)) != int(bucket_end):
            return False
        if int(summary.get("sample_seed", -1)) != _bucket_sample_seed(bucket_start, bucket_end, repeat_id):
            return False
        if str(summary.get("month_tag", "")) != month_tag:
            return False
        if str(summary.get("calendar_year_month", "")) != _calendar_year_month(compare_cfg, month_tag):
            return False
        if selected_user is not None and str(summary.get("selected_rebel_user", "")) != str(selected_user):
            return False
        if not bool(checks.get("no_sybil_trade_in_base", False)):
            return False
        if month_tag != "m00_pre" and not bool(checks.get("target_contract_has_no_trade", True)):
            return False
        if month_tag == "m00_pre" and _count_sybil_trade_rows(paths["data"]) != 0:
            return False
        current_rebels = list(samples.get("sampled_rebels") or [])
        if sampled_rebels is None:
            sampled_rebels = current_rebels
        elif current_rebels != sampled_rebels:
            return False
    return True


def _reference_pre_dir(compare_cfg: dict[str, Any]) -> Path:
    return Path(compare_cfg["DATA_ROOT"]) / "_bucket_reference_pre"


def _reference_pre_result_dir(compare_cfg: dict[str, Any], method: str) -> Path:
    return Path(compare_cfg["RESULT_ROOT"]) / f"{str(method)}__bucket_reference_pre"


def _selection_dataset_tag(selection_method: str, bucket_start: int, bucket_end: int, repeat_id: int) -> str:
    return f"sel_{str(selection_method)}_{_bucket_dataset_tag(bucket_start, bucket_end, repeat_id)}"


def _selection_result_tag(selection_method: str, bucket_start: int, bucket_end: int, repeat_id: int, month_tag: str) -> str:
    return f"{_selection_dataset_tag(selection_method, bucket_start, bucket_end, repeat_id)}_{month_tag}"


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


def _primary_param_value(compare_cfg: dict[str, Any], method: str) -> float:
    if method == "our":
        return float(compare_cfg["OUR_PRIMARY_BETA"])
    if method == "birank":
        return float(compare_cfg["BIRANK_PRIMARY_ETA"])
    if method == "pagerank":
        return float(compare_cfg["PAGERANK_PRIMARY_GAMMA"])
    if method == "eigentrust":
        return float(compare_cfg["EIGENTRUST_PRIMARY_ALPHA"])
    raise ValueError(f"Unknown method: {method}")


def _format_progress(current: int, total: int, start_ts: float) -> str:
    elapsed = max(0.0, time.perf_counter() - start_ts)
    rate = elapsed / current if current > 0 else 0.0
    remaining = max(0.0, rate * (total - current))
    return f"[G3ACompare] progress {current}/{total} ({current / total:.1%}) | elapsed {elapsed / 60:.1f}m | eta {remaining / 60:.1f}m"


def ensure_baseline_dataset(base_dir: Path, compare_cfg: dict[str, Any]) -> dict[str, Path]:
    dataset_paths = _dataset_paths(base_dir)
    if bool(compare_cfg["REUSE_EXISTING_DATASETS"]) and _baseline_dataset_is_valid(dataset_paths):
        return dataset_paths
    base_dir.mkdir(parents=True, exist_ok=True)
    contract_profiles, user_profiles, _, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, compare_cfg)
    if str(compare_cfg.get("GENERATOR_MODE", "g3a")).lower() == "g3a2":
        GEN_MOD.run_g3a2(
            contract_profiles=contract_profiles,
            user_profiles=user_profiles,
            save_dir=str(base_dir),
            horizon_months=int(GEN_MOD._months_between_inclusive(compare_cfg["CALENDAR_BASE_START_YM"], compare_cfg["CALENDAR_FINAL_YM"])),
            end_date=GEN_MOD._month_end_dt(compare_cfg["CALENDAR_FINAL_YM"]),
            seed=int(compare_cfg["GENERATOR_SEED"]),
            include_sybil_contracts=False,
        )
        return dataset_paths
    GEN_MOD.generate_g3a_baseline(
        contract_profiles=contract_profiles,
        user_profiles=user_profiles,
        save_dir=str(base_dir),
        out_name="RQdataG3A.csv",
        horizon_months=int(GEN_MOD._months_between_inclusive(compare_cfg["CALENDAR_BASE_START_YM"], compare_cfg["CALENDAR_FINAL_YM"])),
        end_date=GEN_MOD._month_end_dt(compare_cfg["CALENDAR_FINAL_YM"]),
    )
    return dataset_paths


def _build_g3a2_reference_report(
    month_tag: str,
    compare_cfg: dict[str, Any],
    data_df: pd.DataFrame,
    selected_user: str | None = None,
    repeat_id: int = -1,
    sample_seed: int = -1,
    bucket_start: int = 0,
    bucket_end: int = 100,
    target_contract_ids: list[str] | None = None,
    creation_mode: str = "baseline_only",
) -> dict[str, Any]:
    calendar_ym = _calendar_year_month(compare_cfg, month_tag)
    selected_rebel = "" if selected_user is None else str(selected_user)
    target_contract_ids = [] if target_contract_ids is None else [str(x) for x in target_contract_ids]
    target_trade_mask = data_df["contract_id"].astype(str).isin(target_contract_ids) & data_df["tx_type"].astype(str).str.lower().eq("trade")
    return {
        "summary": {
            "month_tag": str(month_tag),
            "month_index": int(_month_index(month_tag)),
            "calendar_year_month": str(calendar_ym),
            "sample_seed": int(sample_seed),
            "repeat_id": int(repeat_id),
            "selected_rebel_user": selected_rebel,
            "sampled_rebel_count": 0 if not selected_rebel else 1,
            "target_contract_creation_mode": str(creation_mode),
            "reputation_bucket_label": _bucket_label(bucket_start, bucket_end),
            "reputation_bucket_from_bottom_pct_start": int(bucket_start),
            "reputation_bucket_from_bottom_pct_end": int(bucket_end),
            "n_rows": int(len(data_df)),
            "target_contract_ids": target_contract_ids,
        },
        "checks": {
            "no_sybil_trade_in_base": bool(
                not (
                    data_df["contract_id"].astype(str).str.startswith("Sybil")
                    & data_df["tx_type"].astype(str).str.lower().eq("trade")
                ).any()
            ),
            "target_contract_has_no_trade": bool(not target_trade_mask.any()),
        },
        "samples": {
            "sampled_rebels": [] if not selected_rebel else [selected_rebel],
            "target_contract_ids": target_contract_ids,
        },
    }


def _build_g3a2_attack_contract_id(selected_user: str, bucket_start: int, bucket_end: int, repeat_id: int) -> str:
    user_tail = str(selected_user).split("_")[-1]
    return f"G3A2Mal_{int(bucket_start):02d}_{int(bucket_end):02d}_{int(repeat_id):02d}_{user_tail}"


def _inject_g3a2_betrayal(
    source_paths: dict[str, Path],
    compare_cfg: dict[str, Any],
    selected_user: str,
    bucket_start: int,
    bucket_end: int,
    repeat_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    data_df = pd.read_csv(source_paths["data"])
    contracts_df = pd.read_csv(source_paths["contracts"])

    selected_user = str(selected_user)
    attack_ym = str(compare_cfg["CALENDAR_ATTACK_YM"])
    silent_start_ym = "2025-07"
    attack_contract_id = _build_g3a2_attack_contract_id(selected_user, bucket_start, bucket_end, repeat_id)

    month_mask = data_df["calendar_year_month"].astype(str)
    user_mask = (data_df["buyer"].astype(str) == selected_user) | (data_df["seller"].astype(str) == selected_user)
    remove_mask = user_mask & (month_mask >= attack_ym)
    data_df = data_df.loc[~remove_mask].copy()

    attack_month_rows = data_df.loc[month_mask == attack_ym, ["block_number", "tx_index_in_block"]].copy()
    if len(attack_month_rows) == 0:
        raise ValueError(f"No attack-month rows found for {attack_ym} in G3A2 baseline.")
    attack_block = int(attack_month_rows["block_number"].max()) + 1

    quality = 0
    new_contract = {
        "contract_id": attack_contract_id,
        "contract_address": f"0xmal{abs(hash(attack_contract_id)) % 10**34:034d}",
        "category": "Malicious",
        "quality": int(quality),
        "birth_block": int(attack_block),
        "active_end_block": int(attack_block + 10),
        "creator_address": f"creator::{selected_user}",
        "price_dist": "normal",
        "p_min": 0.0,
        "p_max": 0.0,
        "mint_price_mode": "gas_only",
        "trading_curve": "low_steady",
        "intent_density": 0.0,
        "creator_group": "rebel",
        "new_node_only": 0,
        "max_supply": 1,
        "minted_supply": 1,
        "mint_bias": 0.0,
        "secondary_bias": 0.0,
    }
    contracts_df = pd.concat([contracts_df, pd.DataFrame([new_contract])], ignore_index=True)

    buyer_group = ""
    user_meta_df = pd.read_csv(source_paths["users"], usecols=["user", "group"])
    matched = user_meta_df.loc[user_meta_df["user"].astype(str) == selected_user, "group"]
    if len(matched):
        buyer_group = str(matched.iloc[0])

    mint_row = {
        "block_number": int(attack_block),
        "tx_type": "mint",
        "contract_id": attack_contract_id,
        "category": "Malicious",
        "contract_quality": int(quality),
        "token_id": 0,
        "seller": f"creator::{selected_user}",
        "buyer": selected_user,
        "price": 0.0,
        "gas": 0.006,
        "timestamp": f"{attack_ym}-28 12:00:00",
        "tx_index_in_block": 0,
        "buyer_group": buyer_group,
        "calendar_year_month": attack_ym,
    }
    data_df = pd.concat([data_df, pd.DataFrame([mint_row])], ignore_index=True)
    data_df["block_number"] = pd.to_numeric(data_df["block_number"], errors="coerce").fillna(0).astype(int)
    data_df["tx_index_in_block"] = pd.to_numeric(data_df["tx_index_in_block"], errors="coerce").fillna(0).astype(int)
    data_df = data_df.sort_values(["block_number", "tx_index_in_block", "contract_id", "token_id"]).reset_index(drop=True)
    data_df["tx_index_in_block"] = data_df.groupby("block_number").cumcount().astype(int)

    return data_df, contracts_df, attack_contract_id


def _write_g3a2_cumulative_snapshot(
    data_df: pd.DataFrame,
    users_df: pd.DataFrame,
    contracts_df: pd.DataFrame,
    target_dir: Path,
    compare_cfg: dict[str, Any],
    month_tag: str,
    selected_user: str | None = None,
    repeat_id: int = -1,
    sample_seed: int = -1,
    bucket_start: int = 0,
    bucket_end: int = 100,
    target_contract_ids: list[str] | None = None,
    source_data_csv: str | None = None,
    creation_mode: str = "baseline_only",
) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    out_paths = _dataset_paths(target_dir)
    cutoff_ym = _calendar_year_month(compare_cfg, month_tag)

    cut_df = data_df[data_df["calendar_year_month"].astype(str) <= str(cutoff_ym)].copy()
    cut_df.to_csv(out_paths["data"], index=False)
    users_df.to_csv(out_paths["users"], index=False)
    contracts_df.to_csv(out_paths["contracts"], index=False)

    config = {
        "generator_mode": "g3a2",
        "calendar_base_start_ym": str(compare_cfg["CALENDAR_BASE_START_YM"]),
        "calendar_pre_end_ym": str(compare_cfg["CALENDAR_PRE_END_YM"]),
        "calendar_attack_ym": str(compare_cfg["CALENDAR_ATTACK_YM"]),
        "calendar_final_ym": str(compare_cfg["CALENDAR_FINAL_YM"]),
        "month_tag": str(month_tag),
        "calendar_year_month": str(cutoff_ym),
        "source_data_csv": "" if source_data_csv is None else str(source_data_csv),
        "selected_rebel_user": "" if selected_user is None else str(selected_user),
    }
    with open(out_paths["config"], "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    report = _build_g3a2_reference_report(
        month_tag=month_tag,
        compare_cfg=compare_cfg,
        data_df=cut_df,
        selected_user=selected_user,
        repeat_id=repeat_id,
        sample_seed=sample_seed,
        bucket_start=bucket_start,
        bucket_end=bucket_end,
        target_contract_ids=target_contract_ids,
        creation_mode=creation_mode,
    )
    with open(out_paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return out_paths


def _build_g3a2_reference_sequence(
    source_paths: dict[str, Path],
    save_dir: Path,
    compare_cfg: dict[str, Any],
    selected_user: str | None = None,
    repeat_id: int = -1,
    sample_seed: int = -1,
    bucket_start: int = 0,
    bucket_end: int = 100,
    target_contract_ids: list[str] | None = None,
    creation_mode: str = "baseline_only",
) -> dict[str, dict[str, Path]]:
    data_df = pd.read_csv(source_paths["data"])
    users_df = pd.read_csv(source_paths["users"])
    contracts_df = pd.read_csv(source_paths["contracts"])
    month_paths: dict[str, dict[str, Path]] = {}
    for month_tag in [str(tag) for tag in compare_cfg["MONTH_TAGS"]]:
        month_dir = save_dir / month_tag
        month_paths[month_tag] = _write_g3a2_cumulative_snapshot(
            data_df=data_df,
            users_df=users_df,
            contracts_df=contracts_df,
            target_dir=month_dir,
            compare_cfg=compare_cfg,
            month_tag=month_tag,
            selected_user=selected_user,
            repeat_id=repeat_id,
            sample_seed=sample_seed,
            bucket_start=bucket_start,
            bucket_end=bucket_end,
            target_contract_ids=target_contract_ids,
            source_data_csv=str(source_paths["data"]),
            creation_mode=creation_mode,
        )
    return month_paths


def ensure_reference_pre_sequence(compare_cfg: dict[str, Any]) -> dict[str, dict[str, Path]]:
    if str(compare_cfg.get("GENERATOR_MODE", "g3a")).lower() == "g3a2":
        ref_dir = _reference_pre_dir(compare_cfg)
        month_tags = [str(tag) for tag in compare_cfg["MONTH_TAGS"]]
        if bool(compare_cfg["REUSE_EXISTING_DATASETS"]) and _sequence_is_valid(ref_dir, 0, 100, -1, month_tags, compare_cfg):
            return {month_tag: _dataset_paths(ref_dir / month_tag) for month_tag in month_tags}
        base_dir = Path(compare_cfg["DATA_ROOT"]) / str(compare_cfg["BASELINE_TAG"])
        base_paths = ensure_baseline_dataset(base_dir, compare_cfg)
        return _build_g3a2_reference_sequence(base_paths, ref_dir, compare_cfg)
    ref_dir = _reference_pre_dir(compare_cfg)
    month_tags = _eval_month_tags(compare_cfg)
    if bool(compare_cfg["REUSE_EXISTING_DATASETS"]) and _sequence_is_valid(ref_dir, 0, 100, -1, month_tags, compare_cfg):
        return {month_tag: _dataset_paths(ref_dir / month_tag) for month_tag in month_tags}

    base_dir = Path(compare_cfg["DATA_ROOT"]) / str(compare_cfg["BASELINE_TAG"])
    base_paths = ensure_baseline_dataset(base_dir, compare_cfg)
    contract_profiles, user_profiles, _, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, compare_cfg)
    GEN_MOD.generate_g3a_sequence_from_base(
        base_data_csv=str(base_paths["data"]),
        base_users_csv=str(base_paths["users"]),
        base_contracts_csv=str(base_paths["contracts"]),
        contract_profiles=contract_profiles,
        user_profiles=user_profiles,
        save_dir=str(ref_dir),
        g3a_cfg={
            "enable": True,
            "rebel_ratio": 0.0,
            "n_rebels": 0,
            "repeat_id": -1,
            "sample_seed": _bucket_sample_seed(0, 100, -1),
            "post_months": int(compare_cfg["G3A_POST_MONTHS"]),
            "calendar_base_start_ym": str(compare_cfg["CALENDAR_BASE_START_YM"]),
            "calendar_pre_end_ym": str(compare_cfg["CALENDAR_PRE_END_YM"]),
            "calendar_attack_ym": str(compare_cfg["CALENDAR_ATTACK_YM"]),
            "calendar_final_ym": str(compare_cfg["CALENDAR_FINAL_YM"]),
            "inject_contract_mode": "mint_only_no_trade",
            "rebel_silent_after_attack": True,
            "reputation_bucket_label": _bucket_label(0, 100),
            "reputation_bucket_from_bottom_pct_start": 0,
            "reputation_bucket_from_bottom_pct_end": 100,
        },
        save_reports=True,
    )
    return {month_tag: _dataset_paths(ref_dir / month_tag) for month_tag in month_tags}


def ensure_reference_pre_scores(compare_cfg: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Path]]:
    ref_sequence = ensure_reference_pre_sequence(compare_cfg)
    pre_dataset_paths = ref_sequence["m00_pre"]
    outdirs = {method: _reference_pre_result_dir(compare_cfg, method) for method in _active_methods(compare_cfg)}

    our_outdir = outdirs["our"]
    our_params = _method_param_values(compare_cfg, "our")
    if not (bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("our", our_outdir, our_params)):
        run_our_on_dataset(pre_dataset_paths, our_outdir, compare_cfg)

    if "birank" in outdirs:
        birank_params = _method_param_values(compare_cfg, "birank")
        if not (bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("birank", outdirs["birank"], birank_params)):
            run_birank_on_dataset(our_outdir, outdirs["birank"], compare_cfg)
    if "pagerank" in outdirs:
        pagerank_params = _method_param_values(compare_cfg, "pagerank")
        if not (bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("pagerank", outdirs["pagerank"], pagerank_params)):
            run_pagerank_on_dataset(our_outdir, outdirs["pagerank"], compare_cfg)
    if "eigentrust" in outdirs:
        eigentrust_params = _method_param_values(compare_cfg, "eigentrust")
        if not (bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("eigentrust", outdirs["eigentrust"], eigentrust_params)):
            run_eigentrust_on_dataset(pre_dataset_paths, our_outdir, outdirs["eigentrust"], compare_cfg)
    return pre_dataset_paths, outdirs


def build_bucket_user_map(compare_cfg: dict[str, Any]) -> dict[str, dict[tuple[int, int], list[dict[str, Any]]]]:
    pre_dataset_paths, outdirs = ensure_reference_pre_scores(compare_cfg)
    eligible_users = set(_eligible_normal_users(pd.read_csv(pre_dataset_paths["users"])))
    bucket_map_by_method: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = {}
    for method in _active_methods(compare_cfg):
        user_scores_path = _score_file_paths(method, outdirs[method], _primary_param_value(compare_cfg, method))[1]
        df_u = pd.read_csv(user_scores_path)
        _, user_rank_col, _, _ = _method_columns(method)
        df_u = df_u[df_u["user"].astype(str).isin(eligible_users)].copy()
        df_u = df_u.sort_values([user_rank_col, "user"], ascending=[False, True]).reset_index(drop=True)
        total_n = len(df_u)
        method_bucket_map: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for bucket_start, bucket_end in compare_cfg["REPUTATION_BUCKETS"]:
            start_idx = int(np.floor(total_n * (float(bucket_start) / 100.0)))
            end_idx = int(np.ceil(total_n * (float(bucket_end) / 100.0)))
            end_idx = max(start_idx + 1, min(total_n, end_idx))
            bucket_df = df_u.iloc[start_idx:end_idx].copy()
            method_bucket_map[(int(bucket_start), int(bucket_end))] = bucket_df.to_dict("records")
        bucket_map_by_method[method] = method_bucket_map
    return bucket_map_by_method


def _sample_bucket_user(bucket_rows: list[dict[str, Any]], bucket_start: int, bucket_end: int, repeat_id: int) -> dict[str, Any]:
    if not bucket_rows:
        raise ValueError(f"No eligible users found in bucket {bucket_start}-{bucket_end}.")
    rng = np.random.default_rng(_bucket_sample_seed(bucket_start, bucket_end, repeat_id))
    idx = int(rng.integers(0, len(bucket_rows)))
    return dict(bucket_rows[idx])


def generate_dataset_sequence_for_bucket_repeat(
    bucket_start: int,
    bucket_end: int,
    repeat_id: int,
    bucket_dir: Path,
    selected_user: str,
    compare_cfg: dict[str, Any],
) -> dict[str, dict[str, Path]]:
    if str(compare_cfg.get("GENERATOR_MODE", "g3a")).lower() == "g3a2":
        month_tags = [str(tag) for tag in compare_cfg["MONTH_TAGS"]]
        if bool(compare_cfg["REUSE_EXISTING_DATASETS"]) and _sequence_is_valid(
            bucket_dir,
            bucket_start,
            bucket_end,
            repeat_id,
            month_tags,
            compare_cfg,
            selected_user=selected_user,
        ):
            return {month_tag: _dataset_paths(bucket_dir / month_tag) for month_tag in month_tags}

        base_dir = Path(compare_cfg["DATA_ROOT"]) / str(compare_cfg["BASELINE_TAG"])
        base_paths = ensure_baseline_dataset(base_dir, compare_cfg)
        bucket_dir.mkdir(parents=True, exist_ok=True)
        injected_df, injected_contracts_df, attack_contract_id = _inject_g3a2_betrayal(
            source_paths=base_paths,
            compare_cfg=compare_cfg,
            selected_user=selected_user,
            bucket_start=bucket_start,
            bucket_end=bucket_end,
            repeat_id=repeat_id,
        )
        users_df = pd.read_csv(base_paths["users"])
        month_paths: dict[str, dict[str, Path]] = {}
        sample_seed = _bucket_sample_seed(bucket_start, bucket_end, repeat_id)
        for month_tag in month_tags:
            month_dir = bucket_dir / month_tag
            source_df = injected_df if month_tag != "m00_pre" else pd.read_csv(base_paths["data"])
            source_contracts = injected_contracts_df if month_tag != "m00_pre" else pd.read_csv(base_paths["contracts"])
            target_contract_ids = [] if month_tag == "m00_pre" else [attack_contract_id]
            creation_mode = "baseline_only" if month_tag == "m00_pre" else "mint_only_no_trade"
            month_paths[month_tag] = _write_g3a2_cumulative_snapshot(
                data_df=source_df,
                users_df=users_df,
                contracts_df=source_contracts,
                target_dir=month_dir,
                compare_cfg=compare_cfg,
                month_tag=month_tag,
                selected_user=selected_user,
                repeat_id=repeat_id,
                sample_seed=sample_seed,
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                target_contract_ids=target_contract_ids,
                source_data_csv=str(base_paths["data"]),
                creation_mode=creation_mode,
            )
        return month_paths
    month_tags = _eval_month_tags(compare_cfg)
    if bool(compare_cfg["REUSE_EXISTING_DATASETS"]) and _sequence_is_valid(
        bucket_dir,
        bucket_start,
        bucket_end,
        repeat_id,
        month_tags,
        compare_cfg,
        selected_user=selected_user,
    ):
        return {month_tag: _dataset_paths(bucket_dir / month_tag) for month_tag in month_tags}

    base_dir = Path(compare_cfg["DATA_ROOT"]) / str(compare_cfg["BASELINE_TAG"])
    base_paths = ensure_baseline_dataset(base_dir, compare_cfg)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    contract_profiles, user_profiles, _, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, compare_cfg)
    GEN_MOD.generate_g3a_sequence_from_base(
        base_data_csv=str(base_paths["data"]),
        base_users_csv=str(base_paths["users"]),
        base_contracts_csv=str(base_paths["contracts"]),
        contract_profiles=contract_profiles,
        user_profiles=user_profiles,
        save_dir=str(bucket_dir),
        g3a_cfg={
            "enable": True,
            "rebel_ratio": 0.0,
            "selected_rebel_user": str(selected_user),
            "n_rebels": 1,
            "repeat_id": int(repeat_id),
            "sample_seed": _bucket_sample_seed(bucket_start, bucket_end, repeat_id),
            "target_contract_limit": int(compare_cfg["G3A_TARGET_CONTRACT_LIMIT"]),
            "post_months": int(compare_cfg["G3A_POST_MONTHS"]),
            "calendar_base_start_ym": str(compare_cfg["CALENDAR_BASE_START_YM"]),
            "calendar_pre_end_ym": str(compare_cfg["CALENDAR_PRE_END_YM"]),
            "calendar_attack_ym": str(compare_cfg["CALENDAR_ATTACK_YM"]),
            "calendar_final_ym": str(compare_cfg["CALENDAR_FINAL_YM"]),
            "inject_contract_mode": "mint_only_no_trade",
            "rebel_silent_after_attack": True,
            "reputation_bucket_label": _bucket_label(bucket_start, bucket_end),
            "reputation_bucket_from_bottom_pct_start": int(bucket_start),
            "reputation_bucket_from_bottom_pct_end": int(bucket_end),
        },
        save_reports=True,
    )
    return {month_tag: _dataset_paths(bucket_dir / month_tag) for month_tag in month_tags}


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


def collect_method_metrics(
    method: str,
    dataset_paths: dict[str, Path],
    outdir: Path,
    pre_dataset_paths: dict[str, Path],
    pre_outdir: Path,
    bucket_start: int,
    bucket_end: int,
    repeat_id: int,
    month_tag: str,
    compare_cfg: dict[str, Any],
    selection_method: str | None = None,
    param_value: float | None = None,
) -> dict[str, Any]:
    contract_rank_col, user_rank_col, contract_score_col, user_score_col = _method_columns(method)
    contract_path, user_path = _score_file_paths(method, outdir, param_value=param_value)
    pre_contract_path, pre_user_path = _score_file_paths(method, pre_outdir, param_value=param_value)
    df_c = pd.read_csv(contract_path)
    df_u = pd.read_csv(user_path)
    pre_df_c = pd.read_csv(pre_contract_path)
    pre_df_u = pd.read_csv(pre_user_path)
    report = _read_json(dataset_paths["report"])
    pre_report = _read_json(pre_dataset_paths["report"])
    summary = dict(report.get("summary") or {})
    sampled_rebels = set((report.get("samples") or {}).get("sampled_rebels") or [])
    target_contracts = set(summary.get("target_contract_ids") or [cid for cid in df_c["contract_id"].astype(str).tolist() if cid.startswith("Sybil")])

    target_c = df_c[df_c["contract_id"].astype(str).isin(target_contracts)].copy()
    pre_target_c = pre_df_c[pre_df_c["contract_id"].astype(str).isin(target_contracts)].copy()
    g3a_u = df_u[df_u["user"].astype(str).isin(sampled_rebels)].copy()
    pre_g3a_u = pre_df_u[pre_df_u["user"].astype(str).isin(sampled_rebels)].copy()
    target_c["norm_rank"] = _normalized_rank(target_c[contract_rank_col], len(df_c))
    pre_target_c["norm_rank"] = _normalized_rank(pre_target_c[contract_rank_col], len(pre_df_c))
    g3a_u["norm_rank"] = _normalized_rank(g3a_u[user_rank_col], len(df_u))
    pre_g3a_u["norm_rank"] = _normalized_rank(pre_g3a_u[user_rank_col], len(pre_df_u))

    target_contract_rank = _safe_mean(target_c[contract_rank_col])
    target_contract_rank_pre = _safe_mean(pre_target_c[contract_rank_col])
    target_contract_score = _safe_mean(target_c[contract_score_col])
    target_contract_score_pre = _safe_mean(pre_target_c[contract_score_col])
    rebel_user_rank = _safe_mean(g3a_u[user_rank_col])
    rebel_user_rank_pre = _safe_mean(pre_g3a_u[user_rank_col])
    rebel_user_score = _safe_mean(g3a_u[user_score_col])
    rebel_user_score_pre = _safe_mean(pre_g3a_u[user_score_col])

    return {
        "method": method,
        "param_name": METHOD_PARAM_META[method]["param_name"],
        "param_value": float(param_value) if param_value is not None else float("nan"),
        "bucket_label": str(summary.get("reputation_bucket_label") or _bucket_label(bucket_start, bucket_end)),
        "bucket_from_bottom_pct_start": int(summary.get("reputation_bucket_from_bottom_pct_start", bucket_start)),
        "bucket_from_bottom_pct_end": int(summary.get("reputation_bucket_from_bottom_pct_end", bucket_end)),
        "repeat_id": int(repeat_id),
        "sample_seed": int(summary.get("sample_seed", _bucket_sample_seed(bucket_start, bucket_end, repeat_id))),
        "dataset_tag": _selection_dataset_tag(selection_method or method, bucket_start, bucket_end, repeat_id),
        "selection_method": str(selection_method or method),
        "month_tag": month_tag,
        "month_index": int(summary.get("month_index", _month_index(month_tag))),
        "calendar_year_month": str(summary.get("calendar_year_month") or _calendar_year_month(compare_cfg, month_tag)),
        "calendar_month_index_from_attack": int(_calendar_month_index_from_attack(compare_cfg, month_tag)),
        "target_contract_creation_mode": str(summary.get("target_contract_creation_mode") or "mint_only_no_trade"),
        "stage_c_include_mint_edge": bool(compare_cfg.get("OUR_STAGE_C_INCLUDE_MINT_EDGE", False)),
        "selected_rebel_user": str(summary.get("selected_rebel_user") or (next(iter(sampled_rebels)) if sampled_rebels else "")),
        "n_rebels": int(summary.get("sampled_rebel_count", len(sampled_rebels))),
        "malicious_contract_id": str(next(iter(target_contracts)) if target_contracts else ""),
        "malicious_contract_rank": target_contract_rank,
        "malicious_contract_norm_rank": _safe_mean(target_c["norm_rank"]),
        "malicious_contract_best_rank": float(target_c[contract_rank_col].min()) if len(target_c) else float("nan"),
        "malicious_contract_score": target_contract_score,
        "malicious_contract_rank_delta_from_pre": target_contract_rank - target_contract_rank_pre,
        "malicious_contract_score_delta_from_pre": target_contract_score - target_contract_score_pre,
        "rebel_user_rank": rebel_user_rank,
        "rebel_user_norm_rank": _safe_mean(g3a_u["norm_rank"]),
        "rebel_user_best_rank": float(g3a_u[user_rank_col].min()) if len(g3a_u) else float("nan"),
        "rebel_user_score": rebel_user_score,
        "rebel_user_rank_delta_from_pre": rebel_user_rank - rebel_user_rank_pre,
        "rebel_user_norm_rank_delta_from_pre": _safe_mean(g3a_u["norm_rank"]) - _safe_mean(pre_g3a_u["norm_rank"]),
        "rebel_user_score_delta_from_pre": rebel_user_score - rebel_user_score_pre,
        "pre_dataset_month_tag": str((pre_report.get("summary") or {}).get("month_tag", "m00_pre")),
    }


def aggregate_repeat_metrics(runs_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "n_rebels",
        "malicious_contract_rank",
        "malicious_contract_norm_rank",
        "malicious_contract_best_rank",
        "malicious_contract_score",
        "malicious_contract_rank_delta_from_pre",
        "malicious_contract_score_delta_from_pre",
        "rebel_user_rank",
        "rebel_user_norm_rank",
        "rebel_user_best_rank",
        "rebel_user_score",
        "rebel_user_rank_delta_from_pre",
        "rebel_user_norm_rank_delta_from_pre",
        "rebel_user_score_delta_from_pre",
    ]
    records = []
    grouped = runs_df.groupby(
        ["method", "selection_method", "param_name", "param_value", "bucket_label", "bucket_from_bottom_pct_start", "bucket_from_bottom_pct_end", "month_tag"],
        dropna=False,
    )
    for key, sub in grouped:
        method, selection_method, param_name, param_value, bucket_label, bucket_start, bucket_end, month_tag = key
        record = {
            "method": method,
            "selection_method": str(selection_method),
            "param_name": param_name,
            "param_value": float(param_value),
            "bucket_label": str(bucket_label),
            "bucket_from_bottom_pct_start": int(bucket_start),
            "bucket_from_bottom_pct_end": int(bucket_end),
            "month_tag": month_tag,
            "month_index": int(sub["month_index"].iloc[0]),
            "calendar_year_month": str(sub["calendar_year_month"].iloc[0]),
            "calendar_month_index_from_attack": int(sub["calendar_month_index_from_attack"].iloc[0]),
            "target_contract_creation_mode": str(sub["target_contract_creation_mode"].iloc[0]),
            "stage_c_include_mint_edge": bool(sub["stage_c_include_mint_edge"].iloc[0]),
            "n_repeats_actual": int(sub["repeat_id"].nunique()),
        }
        for col in metric_cols:
            record[f"{col}_mean"] = float(sub[col].mean()) if len(sub) else float("nan")
            record[f"{col}_std"] = float(sub[col].std(ddof=0)) if len(sub) else float("nan")
        records.append(record)
    return pd.DataFrame(records).sort_values(["method", "selection_method", "param_value", "bucket_from_bottom_pct_start", "month_index"]).reset_index(drop=True)


def build_attack_month_delta_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    attack_df = runs_df.loc[runs_df["month_tag"].astype(str).eq("m01_attack")].copy()
    if attack_df.empty:
        return pd.DataFrame()
    metric_cols = [
        "malicious_contract_rank_delta_from_pre",
        "malicious_contract_score_delta_from_pre",
        "rebel_user_rank_delta_from_pre",
        "rebel_user_norm_rank_delta_from_pre",
        "rebel_user_score_delta_from_pre",
    ]
    records = []
    grouped = attack_df.groupby(
        ["method", "selection_method", "param_name", "param_value", "bucket_label", "bucket_from_bottom_pct_start", "bucket_from_bottom_pct_end"],
        dropna=False,
    )
    for key, sub in grouped:
        method, selection_method, param_name, param_value, bucket_label, bucket_start, bucket_end = key
        record = {
            "method": method,
            "selection_method": str(selection_method),
            "param_name": param_name,
            "param_value": float(param_value),
            "bucket_label": str(bucket_label),
            "bucket_from_bottom_pct_start": int(bucket_start),
            "bucket_from_bottom_pct_end": int(bucket_end),
            "month_tag": "m01_attack",
            "calendar_year_month": str(sub["calendar_year_month"].iloc[0]),
            "calendar_month_index_from_attack": int(sub["calendar_month_index_from_attack"].iloc[0]),
            "target_contract_creation_mode": str(sub["target_contract_creation_mode"].iloc[0]),
            "stage_c_include_mint_edge": bool(sub["stage_c_include_mint_edge"].iloc[0]),
            "n_repeats_actual": int(sub["repeat_id"].nunique()),
        }
        for col in metric_cols:
            record[f"{col}_mean"] = float(sub[col].mean()) if len(sub) else float("nan")
            record[f"{col}_std"] = float(sub[col].std(ddof=0)) if len(sub) else float("nan")
        records.append(record)
    return pd.DataFrame(records).sort_values(["method", "selection_method", "param_value", "bucket_from_bottom_pct_start"]).reset_index(drop=True)


def run_compare(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compare_cfg = dict(CONFIG)
    if config:
        compare_cfg.update(config)

    result_root = Path(compare_cfg["RESULT_ROOT"])
    result_root.mkdir(parents=True, exist_ok=True)

    base_dir = Path(compare_cfg["DATA_ROOT"]) / str(compare_cfg["BASELINE_TAG"])
    ensure_baseline_dataset(base_dir, compare_cfg)
    bucket_user_map = build_bucket_user_map(compare_cfg)

    buckets = [(int(lo), int(hi)) for lo, hi in compare_cfg["REPUTATION_BUCKETS"]]
    n_repeats = int(compare_cfg["N_REPEATS"])
    month_tags = _eval_month_tags(compare_cfg)
    active_methods = _active_methods(compare_cfg)
    total_tasks = len(buckets) * n_repeats * len(active_methods)
    finished_tasks = 0
    start_ts = time.perf_counter()
    print(f"[G3ACompare] start | method_bucket_tasks={total_tasks} | months_per_task={len(month_tags)}")

    run_records: list[dict[str, Any]] = []
    for bucket_start, bucket_end in buckets:
        for repeat_id in range(n_repeats):
            for selection_method in active_methods:
                bucket_rows = bucket_user_map[selection_method][(bucket_start, bucket_end)]
                selected_user_row = _sample_bucket_user(bucket_rows, bucket_start, bucket_end, repeat_id)
                selected_user = str(selected_user_row["user"])
                bucket_dir = Path(compare_cfg["DATA_ROOT"]) / _selection_dataset_tag(selection_method, bucket_start, bucket_end, repeat_id)
                month_dataset_paths = generate_dataset_sequence_for_bucket_repeat(
                    bucket_start,
                    bucket_end,
                    repeat_id,
                    bucket_dir,
                    selected_user,
                    compare_cfg,
                )

                for month_tag in month_tags:
                    dataset_paths = month_dataset_paths[month_tag]
                    pre_dataset_paths = month_dataset_paths["m00_pre"]
                    result_tag = _selection_result_tag(selection_method, bucket_start, bucket_end, repeat_id, month_tag)

                    our_outdir = result_root / f"our_{result_tag}"
                    our_params = _method_param_values(compare_cfg, "our")
                    if not (bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("our", our_outdir, our_params)):
                        run_our_on_dataset(dataset_paths, our_outdir, compare_cfg)

                    if selection_method == "our":
                        pre_our_outdir = result_root / f"our_{_selection_result_tag(selection_method, bucket_start, bucket_end, repeat_id, 'm00_pre')}"
                        for param_value in our_params:
                            run_records.append(
                                collect_method_metrics(
                                    "our",
                                    dataset_paths,
                                    our_outdir,
                                    pre_dataset_paths,
                                    pre_our_outdir,
                                    bucket_start,
                                    bucket_end,
                                    repeat_id,
                                    month_tag,
                                    compare_cfg,
                                    selection_method=selection_method,
                                    param_value=param_value,
                                )
                            )

                    if selection_method == "birank":
                        birank_outdir = result_root / f"birank_{result_tag}"
                        birank_params = _method_param_values(compare_cfg, "birank")
                        if not (bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("birank", birank_outdir, birank_params)):
                            run_birank_on_dataset(our_outdir, birank_outdir, compare_cfg)
                        pre_birank_outdir = result_root / f"birank_{_selection_result_tag(selection_method, bucket_start, bucket_end, repeat_id, 'm00_pre')}"
                        for param_value in birank_params:
                            run_records.append(
                                collect_method_metrics(
                                    "birank",
                                    dataset_paths,
                                    birank_outdir,
                                    pre_dataset_paths,
                                    pre_birank_outdir,
                                    bucket_start,
                                    bucket_end,
                                    repeat_id,
                                    month_tag,
                                    compare_cfg,
                                    selection_method=selection_method,
                                    param_value=param_value,
                                )
                            )

                    if selection_method == "pagerank":
                        pagerank_outdir = result_root / f"pagerank_{result_tag}"
                        pagerank_params = _method_param_values(compare_cfg, "pagerank")
                        if not (bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("pagerank", pagerank_outdir, pagerank_params)):
                            run_pagerank_on_dataset(our_outdir, pagerank_outdir, compare_cfg)
                        pre_pagerank_outdir = result_root / f"pagerank_{_selection_result_tag(selection_method, bucket_start, bucket_end, repeat_id, 'm00_pre')}"
                        for param_value in pagerank_params:
                            run_records.append(
                                collect_method_metrics(
                                    "pagerank",
                                    dataset_paths,
                                    pagerank_outdir,
                                    pre_dataset_paths,
                                    pre_pagerank_outdir,
                                    bucket_start,
                                    bucket_end,
                                    repeat_id,
                                    month_tag,
                                    compare_cfg,
                                    selection_method=selection_method,
                                    param_value=param_value,
                                )
                            )

                    if selection_method == "eigentrust":
                        eigentrust_outdir = result_root / f"eigentrust_{result_tag}"
                        eigentrust_params = _method_param_values(compare_cfg, "eigentrust")
                        if not (bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("eigentrust", eigentrust_outdir, eigentrust_params)):
                            run_eigentrust_on_dataset(dataset_paths, our_outdir, eigentrust_outdir, compare_cfg)
                        pre_eigentrust_outdir = result_root / f"eigentrust_{_selection_result_tag(selection_method, bucket_start, bucket_end, repeat_id, 'm00_pre')}"
                        for param_value in eigentrust_params:
                            run_records.append(
                                collect_method_metrics(
                                    "eigentrust",
                                    dataset_paths,
                                    eigentrust_outdir,
                                    pre_dataset_paths,
                                    pre_eigentrust_outdir,
                                    bucket_start,
                                    bucket_end,
                                    repeat_id,
                                    month_tag,
                                    compare_cfg,
                                    selection_method=selection_method,
                                    param_value=param_value,
                                )
                            )

                finished_tasks += 1
                print(_format_progress(finished_tasks, total_tasks, start_ts))

    runs_df = pd.DataFrame(run_records).sort_values(
        ["method", "param_value", "bucket_from_bottom_pct_start", "repeat_id", "month_index"]
    ).reset_index(drop=True)
    summary_df = aggregate_repeat_metrics(runs_df)
    delta_df = build_attack_month_delta_summary(runs_df)
    runs_path = result_root / "g3a_bucket_compare_runs.csv"
    summary_path = result_root / "g3a_bucket_compare_summary.csv"
    delta_path = result_root / "g3a_attack_month_delta_summary.csv"
    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    print(f"[G3ACompare] done | outputs={result_root}")
    return runs_df, summary_df, delta_df


if __name__ == "__main__":
    run_compare()
