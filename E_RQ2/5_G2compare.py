from __future__ import annotations

import importlib.util
import json
import random
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    "SCALES": None,
    "INCLUDE_ZERO_SCALE": True,
    "AUTO_SCALE_START": 1,
    "AUTO_SCALE_MULTIPLIER": 2.0,
    "DATA_ROOT": ROOT / "output" / "A" / "12_RQ2compare_G2_v1",
    "RESULT_ROOT": ROOT / "output" / "E" / "5_G2compare_fourway_v1",
    "GENERATOR_SCRIPT": ROOT / "A_dataget" / "11_RQ2data_G2.py",
    "OUR_SCRIPT": ROOT / "E_RQ2" / "0_our_old.py",
    "BIRANK_SCRIPT": ROOT / "E_RQ2" / "1_birank.py",
    "PAGERANK_SCRIPT": ROOT / "E_RQ2" / "2_pagerank.py",
    "EIGENTRUST_SCRIPT": ROOT / "E_RQ2" / "3_eigentrust.py",
    "GENERATOR_SEED": 11,
    "RUN_OUR": True,
    "RUN_BIRANK": True,
    "RUN_PAGERANK": True,
    "RUN_EIGENTRUST": True,
    "OUR_BETA_LIST": [0.1, 0.3, 0.5, 0.7, 0.9],
    "BIRANK_ETA_LIST": [0.1, 0.3, 0.5, 0.7, 0.9],
    "PAGERANK_GAMMA_LIST": [0.1, 0.3, 0.5, 0.7, 0.9],
    "EIGENTRUST_ALPHA_LIST": [0.1, 0.3, 0.5, 0.7, 0.9],
    "OUR_PRIMARY_BETA": 0.7,
    "BIRANK_PRIMARY_ETA": 0.7,
    "PAGERANK_PRIMARY_GAMMA": 0.7,
    "EIGENTRUST_PRIMARY_ALPHA": 0.7,
    "REUSE_EXISTING_DATASETS": True,
    "REUSE_EXISTING_RESULTS": False,
    "BASELINE_TAG": "_base_no_g2_sybil_trade",
    "CONTRACT_COUNTS": {"Hard": 80, "Hype": 20, "Zombie": 50, "Sybil": 10},
    "G2_ATTACKER_COUNT": 500,
    "G2_TARGET_CONTRACT_LIMIT": 10,
    "G2_TAU_START_RATIO": 0.30,
    "G2_TAU_END_RATIO": 0.60,
    "G2_ATTACKER_GROUP": "g2_ctrl",
    "G2_PRICE_MULTIPLIER_RANGE": (0.90, 1.10),
    "G2_MAX_TRIALS_PER_TRADE": 12,
}


METHOD_PARAM_META = {
    "our": {"param_name": "beta", "suffix": "beta", "score_prefix": "4_stageC"},
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


GEN_MOD = _load_module("g2_generator_compare", CONFIG["GENERATOR_SCRIPT"])
OUR_MOD = _load_module("our_pipeline_g2_compare", CONFIG["OUR_SCRIPT"])
BIRANK_MOD = _load_module("birank_g2_compare", CONFIG["BIRANK_SCRIPT"])
PAGERANK_MOD = _load_module("pagerank_g2_compare", CONFIG["PAGERANK_SCRIPT"])
EIGENTRUST_MOD = _load_module("eigentrust_g2_compare", CONFIG["EIGENTRUST_SCRIPT"])


def _scale_tag(fake_trades_per_attacker: int) -> str:
    return f"t{int(fake_trades_per_attacker):04d}"


def _dataset_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "data": data_dir / "RQdataG2.csv",
        "users": data_dir / "RQdataG2_users.csv",
        "contracts": data_dir / "RQdataG2_contracts.csv",
        "report": data_dir / "g2_report.json",
        "config": data_dir / "g2_config.json",
    }


def _legacy_g1_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "data": data_dir / "RQdataG1.csv",
        "users": data_dir / "RQdataG1_users.csv",
        "contracts": data_dir / "RQdataG1_contracts.csv",
        "report": data_dir / "g1_report.json",
        "config": data_dir / "g1_config.json",
    }


def _all_exist(paths: dict[str, Path], keys: list[str]) -> bool:
    return all(paths[k].exists() for k in keys)


def _ensure_g2_named_outputs(data_dir: Path) -> dict[str, Path]:
    target_paths = _dataset_paths(data_dir)
    legacy_paths = _legacy_g1_paths(data_dir)
    for key, dst in target_paths.items():
        src = legacy_paths[key]
        if (not dst.exists()) and src.exists():
            shutil.copy2(str(src), str(dst))
    return target_paths


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


def _result_exists(method: str, outdir: Path, param_values: list[float]) -> bool:
    required = list(_score_file_paths(method, outdir, None))
    for param_value in param_values:
        required.extend(_score_file_paths(method, outdir, float(param_value)))
    return all(path.exists() for path in required)


def _apply_contract_counts(contract_profiles: list[Any], cfg: dict[str, Any]) -> list[Any]:
    count_map = {str(k): int(v) for k, v in dict(cfg.get("CONTRACT_COUNTS", {})).items()}
    for profile in contract_profiles:
        category = str(getattr(profile, "category", ""))
        if category in count_map:
            profile.num_contracts = count_map[category]
    return contract_profiles


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


def _count_nonattack_sybil_trades(data_csv: Path) -> int:
    if not data_csv.exists():
        return -1
    df = pd.read_csv(data_csv, low_memory=False)
    sybil_trade_mask = (
        df["contract_id"].astype(str).str.startswith("Sybil")
        & df["tx_type"].astype(str).str.lower().eq("trade")
    )
    if "attack_type" not in df.columns:
        return int(sybil_trade_mask.sum())
    organic_mask = sybil_trade_mask & (~df["attack_type"].astype(str).eq("G2"))
    return int(organic_mask.sum())


def _baseline_dataset_is_valid(dataset_paths: dict[str, Path]) -> bool:
    required = ["data", "users", "contracts", "report", "config"]
    if not _all_exist(dataset_paths, required):
        return False
    return _count_nonattack_sybil_trades(dataset_paths["data"]) == 0


def _scale_dataset_is_valid(dataset_paths: dict[str, Path], fake_trades_per_attacker: int) -> bool:
    required = ["data", "users", "contracts", "report", "config"]
    if not _all_exist(dataset_paths, required):
        return False
    if _count_nonattack_sybil_trades(dataset_paths["data"]) != 0:
        return False

    report = _read_json(dataset_paths["report"])
    summary = dict(report.get("summary") or {})
    expected_total = int(CONFIG["G2_ATTACKER_COUNT"]) * max(0, int(fake_trades_per_attacker))
    return (
        int(summary.get("trades_per_attacker", -1)) == int(fake_trades_per_attacker)
        and int(summary.get("realized_total_fake_trades", -1)) == expected_total
    )


def _base_trade_count(dataset_paths: dict[str, Path]) -> int:
    df = pd.read_csv(dataset_paths["data"], low_memory=False, usecols=["tx_type"])
    return int(df["tx_type"].astype(str).str.lower().eq("trade").sum())


def resolve_scales(compare_cfg: dict[str, Any]) -> list[int]:
    manual_scales = compare_cfg.get("SCALES")
    if manual_scales:
        return sorted({max(0, int(x)) for x in manual_scales})

    base_dir = Path(compare_cfg["DATA_ROOT"]) / str(compare_cfg["BASELINE_TAG"])
    base_paths = ensure_baseline_dataset(base_dir)
    base_trade_count = _base_trade_count(base_paths)
    n_attackers = max(1, int(compare_cfg["G2_ATTACKER_COUNT"]))
    stop_scale = max(1, int(base_trade_count // 2 // n_attackers))

    scales: list[int] = [0] if bool(compare_cfg.get("INCLUDE_ZERO_SCALE", True)) else []
    current = max(1, int(compare_cfg.get("AUTO_SCALE_START", 1)))
    multiplier = float(compare_cfg.get("AUTO_SCALE_MULTIPLIER", 2.0))
    if multiplier <= 1.0:
        raise ValueError("AUTO_SCALE_MULTIPLIER must be > 1.0.")

    while current < stop_scale:
        scales.append(int(current))
        next_value = int(np.ceil(current * multiplier))
        if next_value <= current:
            next_value = current + 1
        current = next_value

    if stop_scale not in scales:
        scales.append(int(stop_scale))

    return sorted({int(x) for x in scales})


def _build_g2_cfg(fake_trades_per_attacker: int) -> dict[str, Any]:
    trades_per_attacker = max(0, int(fake_trades_per_attacker))
    return {
        "enable": trades_per_attacker > 0,
        "n_attackers": int(CONFIG["G2_ATTACKER_COUNT"]),
        "trades_per_attacker": trades_per_attacker,
        "target_contract_limit": int(CONFIG["G2_TARGET_CONTRACT_LIMIT"]),
        "target_contract_ids": [],
        "tau_start_ratio": float(CONFIG["G2_TAU_START_RATIO"]),
        "tau_end_ratio": float(CONFIG["G2_TAU_END_RATIO"]),
        "attacker_group": str(CONFIG["G2_ATTACKER_GROUP"]),
        "price_multiplier_range": tuple(float(x) for x in CONFIG["G2_PRICE_MULTIPLIER_RANGE"]),
        "max_trials_per_trade": int(CONFIG["G2_MAX_TRIALS_PER_TRADE"]),
    }


def _choose_g2_target_contract_ids(contract_df: pd.DataFrame, g2_cfg: dict[str, Any]) -> list[str]:
    if g2_cfg["target_contract_ids"]:
        valid = set(contract_df["contract_id"].astype(str))
        picked = [cid for cid in g2_cfg["target_contract_ids"] if cid in valid]
        if picked:
            return picked
    limit = max(1, int(g2_cfg["target_contract_limit"]))
    sybil = contract_df[contract_df["category"].astype(str) == "Sybil"]["contract_id"].astype(str).tolist()
    if sybil:
        return sybil[:limit]
    return contract_df["contract_id"].astype(str).tolist()[:limit]


def _sample_g2_price(gen: Any, cp: Any, g2_cfg: dict[str, Any]) -> float:
    base_price = float(gen._sample_trade_price(cp))
    lo, hi = g2_cfg["price_multiplier_range"]
    multiplier = random.uniform(lo, max(lo, hi))
    price = base_price * multiplier
    return float(max(float(cp.p_min), min(float(cp.p_max), price)))


def _pick_g2_token_from_state(
    token_owner: dict[tuple[str, int], str],
    cid: str,
    buyer: str,
    attacker_set: set[str],
) -> tuple[int | None, str | None]:
    creator = f"creator_{cid}"
    attacker_owned: list[tuple[int, str]] = []
    normal_secondary: list[tuple[int, str]] = []
    creator_owned: list[tuple[int, str]] = []
    for (cid2, tid), owner in token_owner.items():
        if cid2 != cid or owner == buyer:
            continue
        owner_str = str(owner)
        if owner_str in attacker_set:
            attacker_owned.append((tid, owner_str))
        elif owner_str == creator:
            creator_owned.append((tid, owner_str))
        else:
            normal_secondary.append((tid, owner_str))
    for pool in (attacker_owned, normal_secondary, creator_owned):
        if pool:
            return random.choice(pool)
    return None, None


def ensure_baseline_dataset(base_dir: Path) -> dict[str, Path]:
    dataset_paths = _ensure_g2_named_outputs(base_dir)
    if bool(CONFIG["REUSE_EXISTING_DATASETS"]) and _baseline_dataset_is_valid(dataset_paths):
        print(f"[G2Compare] reuse baseline dataset: {base_dir}")
        return dataset_paths

    base_dir.mkdir(parents=True, exist_ok=True)
    gen = GEN_MOD.Data3BehaviorGenerator(
        chain_end_block=20_000_000,
        block_time_sec=12,
        seed=int(CONFIG["GENERATOR_SEED"]),
    )
    contract_profiles, user_profiles, creator_trade_cfg, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, CONFIG)
    gen.generate_data3(
        contract_profiles=contract_profiles,
        user_profiles=user_profiles,
        horizon_months=36,
        save_dir=str(base_dir),
        out_name="RQdataG2.csv",
        creator_trade_cfg=creator_trade_cfg,
        g1_cfg={"enable": False},
    )
    dataset_paths = _ensure_g2_named_outputs(base_dir)

    df = pd.read_csv(dataset_paths["data"], low_memory=False)
    df = GEN_MOD.remove_sybil_trade_rows(df)
    gen._finalize_event_frame(df, gen._infer_end_date_from_events(df)).to_csv(dataset_paths["data"], index=False)

    report = _read_json(dataset_paths["report"])
    report["g2_enabled"] = False
    report["summary"] = {
        "n_attackers": int(CONFIG["G2_ATTACKER_COUNT"]),
        "trades_per_attacker": 0,
        "requested_total_fake_trades": 0,
        "realized_total_fake_trades": 0,
        "distinct_target_contracts_hit": 0,
    }
    report["checks"] = {"no_organic_sybil_trade_in_base": bool(_count_nonattack_sybil_trades(dataset_paths["data"]) == 0)}
    with open(dataset_paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    with open(dataset_paths["config"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": int(CONFIG["GENERATOR_SEED"]),
                "generator_script": str(CONFIG["GENERATOR_SCRIPT"]),
                "base_mode": "sybil_contracts_without_sybil_trades",
                "g2_cfg": _build_g2_cfg(0),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return dataset_paths


def generate_dataset_for_scale(fake_trades_per_attacker: int, data_dir: Path) -> dict[str, Path]:
    dataset_paths = _dataset_paths(data_dir)
    if bool(CONFIG["REUSE_EXISTING_DATASETS"]) and _scale_dataset_is_valid(dataset_paths, fake_trades_per_attacker):
        print(f"[G2Compare] reuse dataset for scale={fake_trades_per_attacker}: {data_dir}")
        return dataset_paths

    base_dir = Path(CONFIG["DATA_ROOT"]) / str(CONFIG["BASELINE_TAG"])
    base_paths = ensure_baseline_dataset(base_dir)
    g2_cfg = _build_g2_cfg(int(fake_trades_per_attacker))
    if not g2_cfg["enable"]:
        return base_paths

    data_dir.mkdir(parents=True, exist_ok=True)

    gen = GEN_MOD.Data3BehaviorGenerator(
        chain_end_block=20_000_000,
        block_time_sec=12,
        seed=int(CONFIG["GENERATOR_SEED"]),
    )
    contract_profiles, _, _, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, CONFIG)
    contract_profile_map = {
        f"{cp.category}_{k:03d}": cp
        for cp in contract_profiles
        for k in range(cp.num_contracts)
    }

    df = pd.read_csv(base_paths["data"], low_memory=False)
    user_meta_df = pd.read_csv(base_paths["users"])
    contract_df = pd.read_csv(base_paths["contracts"])
    base_report = _read_json(base_paths["report"])
    end_date = gen._infer_end_date_from_events(df)

    target_contract_ids = _choose_g2_target_contract_ids(contract_df, g2_cfg)
    if not target_contract_ids:
        raise ValueError("No target contracts found for G2 dataset generation.")

    contract_lookup = contract_df.set_index("contract_id").to_dict("index")
    token_owner, next_token_id = gen._rebuild_owner_state(df)
    attacker_group = str(g2_cfg["attacker_group"])
    attacker_users = [f"{attacker_group}_{gen.seed}_{i:06d}" for i in range(int(g2_cfg["n_attackers"]))]
    attacker_set = set(attacker_users)

    injected_user_rows = [
        {
            "user": uid,
            "group": attacker_group,
            "budget_eth": 0.0,
            "n_buys_target": int(g2_cfg["trades_per_attacker"]),
            "avg_ticket_eth": 0.0,
            "sell_prob": 0.0,
            "hold_blocks_mean": 0,
            "hold_blocks_std": 0,
            "early_scale_blocks": 0,
            "secondary_buy_prob": 0.0,
            "category_pref": "{}",
        }
        for uid in attacker_users
    ]

    records: list[dict[str, Any]] = []
    tau_values: list[float] = []
    failed_trades = 0
    bootstrap_trades = 0
    attacker_to_attacker_trades = 0

    for _ in range(int(g2_cfg["trades_per_attacker"])):
        buyers_this_round = attacker_users.copy()
        random.shuffle(buyers_this_round)
        for buyer in buyers_this_round:
            realized = False
            for _attempt in range(int(g2_cfg["max_trials_per_trade"])):
                cid = random.choice(target_contract_ids)
                meta = contract_lookup[cid]
                cp = contract_profile_map[cid]

                life = max(1, int(meta["active_end_block"]) - int(meta["birth_block"]))
                tau_start = int(meta["birth_block"] + life * float(g2_cfg["tau_start_ratio"]))
                tau_end = int(meta["birth_block"] + life * float(g2_cfg["tau_end_ratio"]))
                tau_start = GEN_MOD.clip_int(
                    tau_start,
                    int(meta["birth_block"]),
                    max(int(meta["birth_block"]), int(meta["active_end_block"]) - 1),
                )
                tau_end = GEN_MOD.clip_int(max(tau_start + 1, tau_end), tau_start + 1, int(meta["active_end_block"]))

                span = max(1, tau_end - tau_start)
                if random.random() < 0.80:
                    offset = int(np.random.exponential(scale=max(1, span / 5)))
                    trade_block = GEN_MOD.clip_int(tau_start + offset, tau_start, tau_end - 1)
                else:
                    trade_block = random.randint(tau_start, tau_end - 1)

                tid, seller = _pick_g2_token_from_state(token_owner, cid, buyer, attacker_set)
                if tid is None:
                    creator = f"creator_{cid}"
                    tid = next_token_id[cid]
                    next_token_id[cid] += 1
                    token_owner[(cid, tid)] = creator
                    seller = creator
                    gen._append_event(
                        records,
                        trade_block,
                        "mint",
                        cid,
                        meta["category"],
                        meta["quality"],
                        tid,
                        creator,
                        creator,
                        0.0,
                        "creator",
                        is_injected=False,
                        attack_type=None,
                        note="g2_inject_mint",
                    )

                if seller == buyer:
                    continue

                gen._append_event(
                    records,
                    trade_block,
                    "trade",
                    cid,
                    meta["category"],
                    meta["quality"],
                    tid,
                    seller,
                    buyer,
                    _sample_g2_price(gen, cp, g2_cfg),
                    attacker_group,
                    is_injected=True,
                    attack_type="G2",
                    note="g2_fabricated_trade",
                )
                token_owner[(cid, tid)] = buyer
                tau_values.append((trade_block - int(meta["birth_block"])) / float(life))
                bootstrap_trades += int(str(seller).startswith("creator_"))
                attacker_to_attacker_trades += int((str(seller) in attacker_set) and (buyer in attacker_set))
                realized = True
                break

            if not realized:
                failed_trades += 1

    if records:
        df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)
    if injected_user_rows:
        user_meta_df = pd.concat([user_meta_df, pd.DataFrame(injected_user_rows)], ignore_index=True)

    df = gen._finalize_event_frame(df, end_date)
    g2_df = df[df["attack_type"] == "G2"].copy()
    buy_counts = g2_df["buyer"].value_counts() if len(g2_df) else pd.Series(dtype=int)
    involvements = (
        pd.concat([g2_df["buyer"].rename("addr"), g2_df["seller"].rename("addr")], axis=0).value_counts()
        if len(g2_df)
        else pd.Series(dtype=int)
    )

    summary = {
        "enabled": True,
        "target_contract_ids": target_contract_ids,
        "n_attackers": int(g2_cfg["n_attackers"]),
        "trades_per_attacker": int(g2_cfg["trades_per_attacker"]),
        "requested_total_fake_trades": int(g2_cfg["n_attackers"]) * int(g2_cfg["trades_per_attacker"]),
        "realized_total_fake_trades": int(len(g2_df)),
        "failed_trade_slots": int(failed_trades),
        "distinct_target_contracts_hit": int(g2_df["contract_id"].nunique()) if len(g2_df) else 0,
        "bootstrap_from_creator_trades": int(bootstrap_trades),
        "attacker_to_attacker_trades": int(attacker_to_attacker_trades),
        "tau_ratio_min": round(float(min(tau_values)), 6) if tau_values else None,
        "tau_ratio_max": round(float(max(tau_values)), 6) if tau_values else None,
    }

    report = {
        "g2_enabled": True,
        "baseline_report_snapshot": {"creator_trade_summary": base_report.get("creator_trade_summary")},
        "summary": {
            **summary,
            "distinct_g2_addresses_observed": int(len(set(g2_df["buyer"]))) if len(g2_df) else 0,
            "max_fake_buys_per_g2_addr": int(buy_counts.max()) if not buy_counts.empty else 0,
            "avg_fake_buys_per_g2_addr": round(float(buy_counts.mean()), 4) if not buy_counts.empty else 0.0,
            "max_total_involvement_per_addr": int(involvements.max()) if not involvements.empty else 0,
        },
        "checks": {
            "all_injected_rows_flagged": bool(len(g2_df) == int(df["is_injected"].sum())),
            "only_target_contracts": bool(len(g2_df) == 0 or g2_df["contract_id"].isin(set(target_contract_ids)).all()),
            "no_self_trade": bool(len(g2_df) == 0 or (g2_df["seller"] != g2_df["buyer"]).all()),
            "tau_within_config_window": bool(
                (not tau_values)
                or (
                    min(tau_values) >= float(g2_cfg["tau_start_ratio"]) - 1e-9
                    and max(tau_values) <= float(g2_cfg["tau_end_ratio"]) + 1e-9
                )
            ),
            "realized_requested_trade_budget": bool(int(len(g2_df)) == int(summary["requested_total_fake_trades"])),
            "buy_count_per_attacker_matches_scale": bool(
                (not buy_counts.empty)
                and int(buy_counts.min()) == int(g2_cfg["trades_per_attacker"])
                and int(buy_counts.max()) == int(g2_cfg["trades_per_attacker"])
            ),
        },
        "samples": {
            "buy_count_per_g2_addr_top10": buy_counts.head(10).to_dict(),
            "involvement_per_addr_top10": involvements.head(10).to_dict(),
            "tau_ratio_first10": [round(float(x), 6) for x in tau_values[:10]],
            "target_contract_ids": target_contract_ids,
        },
    }

    dataset_paths = _dataset_paths(data_dir)
    df.to_csv(dataset_paths["data"], index=False)
    user_meta_df.to_csv(dataset_paths["users"], index=False)
    contract_df.to_csv(dataset_paths["contracts"], index=False)
    with open(dataset_paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    with open(dataset_paths["config"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": gen.seed,
                "base_data_csv": str(base_paths["data"]),
                "base_users_csv": str(base_paths["users"]),
                "base_contracts_csv": str(base_paths["contracts"]),
                "g2_cfg": g2_cfg,
                "g2_summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(
        f"[G2Compare] generated scale={fake_trades_per_attacker} | "
        f"requested={summary['requested_total_fake_trades']} | realized={summary['realized_total_fake_trades']}"
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
            "STAGE_C_BETA_LIST": [float(x) for x in compare_cfg["OUR_BETA_LIST"]],
            "STAGE_C_PRIMARY_BETA": float(compare_cfg["OUR_PRIMARY_BETA"]),
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


def run_eigentrust_on_dataset(
    dataset_paths: dict[str, Path],
    our_outdir: Path,
    outdir: Path,
    compare_cfg: dict[str, Any],
) -> dict[str, Any]:
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


def collect_method_metrics(
    method: str,
    fake_trades_per_attacker: int,
    outdir: Path,
    param_value: float | None = None,
) -> dict[str, Any]:
    param_meta = METHOD_PARAM_META[method]
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
    df_c = pd.read_csv(contract_path)
    df_u = pd.read_csv(user_path)

    attacked_c = df_c[df_c["contract_id"].astype(str).str.startswith("Sybil")].copy()
    g2_u = df_u[df_u["user"].astype(str).str.startswith(f"{CONFIG['G2_ATTACKER_GROUP']}_")].copy()
    attacked_c["norm_rank"] = _normalized_rank(attacked_c[contract_rank_col], len(df_c))
    g2_u["norm_rank"] = _normalized_rank(g2_u[user_rank_col], len(df_u))

    return {
        "method": method,
        "param_name": param_meta["param_name"],
        "param_value": float(param_value) if param_value is not None else float("nan"),
        "fake_trades_per_attacker": int(fake_trades_per_attacker),
        "n_contracts_total": int(len(df_c)),
        "n_users_total": int(len(df_u)),
        "target_contract_mean_rank": _safe_mean(attacked_c[contract_rank_col]),
        "target_contract_mean_norm_rank": _safe_mean(attacked_c["norm_rank"]),
        "target_contract_best_rank": float(attacked_c[contract_rank_col].min()) if len(attacked_c) else float("nan"),
        "target_contract_score_mean": _safe_mean(attacked_c[contract_score_col]),
        "target_contract_top20_count": int((attacked_c[contract_rank_col] <= 20).sum()),
        "g2_user_mean_rank": _safe_mean(g2_u[user_rank_col]),
        "g2_user_mean_norm_rank": _safe_mean(g2_u["norm_rank"]),
        "g2_user_best_rank": float(g2_u[user_rank_col].min()) if len(g2_u) else float("nan"),
        "g2_user_score_mean": _safe_mean(g2_u[user_score_col]),
        "g2_user_top200_count": int((g2_u[user_rank_col] <= 200).sum()),
        "g2_user_count_observed": int(len(g2_u)),
    }


def plot_results(summary_df: pd.DataFrame, outdir: Path, cfg: dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    primary_df = summary_df[
        (
            ((summary_df["method"] == "our") & np.isclose(summary_df["param_value"], float(cfg["OUR_PRIMARY_BETA"])))
            | ((summary_df["method"] == "birank") & np.isclose(summary_df["param_value"], float(cfg["BIRANK_PRIMARY_ETA"])))
            | ((summary_df["method"] == "pagerank") & np.isclose(summary_df["param_value"], float(cfg["PAGERANK_PRIMARY_GAMMA"])))
            | ((summary_df["method"] == "eigentrust") & np.isclose(summary_df["param_value"], float(cfg["EIGENTRUST_PRIMARY_ALPHA"])))
        )
    ].copy()
    primary_tag = (
        f"our{float(cfg['OUR_PRIMARY_BETA']):.1f}_"
        f"birank{float(cfg['BIRANK_PRIMARY_ETA']):.1f}_"
        f"pagerank{float(cfg['PAGERANK_PRIMARY_GAMMA']):.1f}_"
        f"eigentrust{float(cfg['EIGENTRUST_PRIMARY_ALPHA']):.1f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, sub in primary_df.groupby("method"):
        sub = sub.sort_values("fake_trades_per_attacker")
        axes[0].plot(sub["fake_trades_per_attacker"], sub["target_contract_mean_norm_rank"], marker="o", label=method)
        axes[1].plot(sub["fake_trades_per_attacker"], sub["g2_user_mean_norm_rank"], marker="o", label=method)
    axes[0].set_title("G2 Target Contract Mean Normalized Rank")
    axes[0].set_xlabel("fake_trades_per_attacker")
    axes[0].set_ylabel("Mean normalized rank")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_title("G2 Controller Mean Normalized Rank")
    axes[1].set_xlabel("fake_trades_per_attacker")
    axes[1].set_ylabel("Mean normalized rank")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / f"g2_mean_norm_rank_{primary_tag}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, sub in primary_df.groupby("method"):
        sub = sub.sort_values("fake_trades_per_attacker")
        axes[0].plot(sub["fake_trades_per_attacker"], sub["target_contract_best_rank"], marker="o", label=method)
        axes[1].plot(sub["fake_trades_per_attacker"], sub["g2_user_best_rank"], marker="o", label=method)
    axes[0].set_title("Best G2 Target Contract Rank")
    axes[0].set_xlabel("fake_trades_per_attacker")
    axes[0].set_ylabel("Best raw rank")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_title("Best G2 Controller Rank")
    axes[1].set_xlabel("fake_trades_per_attacker")
    axes[1].set_ylabel("Best raw rank")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / f"g2_best_rank_{primary_tag}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_compare(config: dict[str, Any] | None = None) -> pd.DataFrame:
    compare_cfg = dict(CONFIG)
    if config:
        compare_cfg.update(config)
    result_root = Path(compare_cfg["RESULT_ROOT"])
    result_root.mkdir(parents=True, exist_ok=True)

    scales = resolve_scales(compare_cfg)
    base_dir = Path(compare_cfg["DATA_ROOT"]) / str(compare_cfg["BASELINE_TAG"])
    base_paths = ensure_baseline_dataset(base_dir)
    base_trade_count = _base_trade_count(base_paths)
    half_trade_count = base_trade_count // 2
    print(
        f"[G2Compare] scales={scales} | "
        f"n_attackers={int(compare_cfg['G2_ATTACKER_COUNT'])} | "
        f"target_contract_limit={int(compare_cfg['G2_TARGET_CONTRACT_LIMIT'])} | "
        f"base_trade_count={base_trade_count} | "
        f"half_trade_count={half_trade_count} | "
        f"terminal_fake_trades={scales[-1] * int(compare_cfg['G2_ATTACKER_COUNT']) if scales else 0}"
    )

    records: list[dict[str, Any]] = []
    for fake_trades_per_attacker in scales:
        tag = _scale_tag(int(fake_trades_per_attacker))
        data_dir = Path(compare_cfg["DATA_ROOT"]) / tag
        print(f"[G2Compare] scale={fake_trades_per_attacker}")
        dataset_paths = generate_dataset_for_scale(int(fake_trades_per_attacker), data_dir)

        our_outdir = result_root / f"our_{tag}"
        our_params = _method_param_values(compare_cfg, "our")
        if bool(compare_cfg["RUN_OUR"]):
            if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("our", our_outdir, our_params):
                print(f"[G2Compare] reuse OUR result: {our_outdir}")
            else:
                run_our_on_dataset(dataset_paths, our_outdir, compare_cfg)
            for param_value in our_params:
                records.append(collect_method_metrics("our", int(fake_trades_per_attacker), our_outdir, param_value))

        birank_outdir = result_root / f"birank_{tag}"
        birank_params = _method_param_values(compare_cfg, "birank")
        if bool(compare_cfg["RUN_BIRANK"]):
            if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("birank", birank_outdir, birank_params):
                print(f"[G2Compare] reuse BiRank result: {birank_outdir}")
            else:
                run_birank_on_dataset(our_outdir, birank_outdir, compare_cfg)
            for param_value in birank_params:
                records.append(collect_method_metrics("birank", int(fake_trades_per_attacker), birank_outdir, param_value))

        pagerank_outdir = result_root / f"pagerank_{tag}"
        pagerank_params = _method_param_values(compare_cfg, "pagerank")
        if bool(compare_cfg["RUN_PAGERANK"]):
            if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("pagerank", pagerank_outdir, pagerank_params):
                print(f"[G2Compare] reuse PageRank result: {pagerank_outdir}")
            else:
                run_pagerank_on_dataset(our_outdir, pagerank_outdir, compare_cfg)
            for param_value in pagerank_params:
                records.append(collect_method_metrics("pagerank", int(fake_trades_per_attacker), pagerank_outdir, param_value))

        eigentrust_outdir = result_root / f"eigentrust_{tag}"
        eigentrust_params = _method_param_values(compare_cfg, "eigentrust")
        if bool(compare_cfg["RUN_EIGENTRUST"]):
            if bool(compare_cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("eigentrust", eigentrust_outdir, eigentrust_params):
                print(f"[G2Compare] reuse EigenTrust result: {eigentrust_outdir}")
            else:
                run_eigentrust_on_dataset(dataset_paths, our_outdir, eigentrust_outdir, compare_cfg)
            for param_value in eigentrust_params:
                records.append(collect_method_metrics("eigentrust", int(fake_trades_per_attacker), eigentrust_outdir, param_value))

    summary_df = pd.DataFrame(records).sort_values(["method", "param_value", "fake_trades_per_attacker"]).reset_index(drop=True)
    summary_path = result_root / "g2_scale_compare_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    plot_results(summary_df, result_root, compare_cfg)
    print(f"[G2Compare] summary saved: {summary_path}")
    print("[G2Compare] finished")
    return summary_df


if __name__ == "__main__":
    run_compare()
