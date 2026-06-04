from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    "SCALES": None,
    "EXCLUDE_SCALES": [13720],
    "INCLUDE_ZERO_SCALE": True,
    "AUTO_SCALE_START": 50,
    "AUTO_SCALE_MULTIPLIER": 2.0,
    "AUTO_SCALE_MAX_STEPS": 12,
    "STOP_AT_SYBIL_SHARE": 0.50,
    "DATA_ROOT": ROOT / "output" / "A" / "11_RQ2compare_v2",
    "RESULT_ROOT": ROOT / "output" / "E" / "4_G1compare_fourway_v3",
    "GENERATOR_SCRIPT": ROOT / "A_dataget" / "11_RQ2data_G1_2.py",
    "OUR_SCRIPT": ROOT / "E_RQ2" / "0_our_old.py",
    "BIRANK_SCRIPT": ROOT / "E_RQ2" / "1_birank.py",
    "PAGERANK_SCRIPT": ROOT / "E_RQ2" / "2_pagerank.py",
    "EIGENTRUST_SCRIPT": ROOT / "E_RQ2" / "3_eigentrust.py",
    "GENERATOR_SEED": 11,
    "RUN_OUR": True,
    "RUN_BIRANK": True,
    "RUN_PAGERANK": True,
    "RUN_EIGENTRUST": True,
    "PARAM_GRID": [0.1, 0.3, 0.5, 0.7, 0.9],
    "OUR_BETA_LIST": [0.1, 0.3, 0.5, 0.7, 0.9],
    "BIRANK_ETA_LIST": [0.1, 0.3, 0.5, 0.7, 0.9],
    "PAGERANK_GAMMA_LIST": [0.1, 0.3, 0.5, 0.7, 0.9],
    "EIGENTRUST_ALPHA_LIST": [0.1, 0.3, 0.5, 0.7, 0.9],
    "OUR_PRIMARY_BETA": 0.7,
    "BIRANK_PRIMARY_ETA": 0.7,
    "PAGERANK_PRIMARY_GAMMA": 0.7,
    "EIGENTRUST_PRIMARY_ALPHA": 0.7,
    "USE_INCREMENTAL_G1": True,
    "REUSE_EXISTING_DATASETS": True,
    "REUSE_EXISTING_RESULTS": False,
    "BASELINE_TAG": "_base_no_g1",
    "CONTRACT_COUNTS": {
        "Hard": 80,
        "Hype": 20,
        "Zombie": 50,
        "Sybil": 10,
    },
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


GEN_MOD = _load_module("g1_generator_compare", CONFIG["GENERATOR_SCRIPT"])
OUR_MOD = _load_module("our_pipeline_compare", CONFIG["OUR_SCRIPT"])
BIRANK_MOD = _load_module("birank_compare", CONFIG["BIRANK_SCRIPT"])
PAGERANK_MOD = _load_module("pagerank_compare", CONFIG["PAGERANK_SCRIPT"])
EIGENTRUST_MOD = _load_module("eigentrust_compare", CONFIG["EIGENTRUST_SCRIPT"])


def _scale_tag(n_sybil: int) -> str:
    return f"n{int(n_sybil):04d}"


def _dataset_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "data": data_dir / "RQdataG1.csv",
        "users": data_dir / "RQdataG1_users.csv",
        "contracts": data_dir / "RQdataG1_contracts.csv",
        "report": data_dir / "g1_report.json",
        "config": data_dir / "g1_config.json",
    }


def _all_exist(paths: dict[str, Path], keys: list[str]) -> bool:
    return all(paths[k].exists() for k in keys)


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
    if method not in METHOD_PARAM_META:
        raise ValueError(f"Unknown method: {method}")

    meta = METHOD_PARAM_META[method]
    prefix = meta["score_prefix"]
    if param_value is None:
        return (
            outdir / f"{prefix}_contract_scores.csv",
            outdir / f"{prefix}_user_scores.csv",
        )

    suffix = meta["suffix"]
    suffix_tag = f"{suffix}{float(param_value):.2f}"
    return (
        outdir / f"{prefix}_contract_scores_{suffix_tag}.csv",
        outdir / f"{prefix}_user_scores_{suffix_tag}.csv",
    )


def _result_exists(method: str, outdir: Path, param_values: list[float]) -> bool:
    required = list(_score_file_paths(method, outdir, None))
    for param_value in param_values:
        required.extend(_score_file_paths(method, outdir, float(param_value)))
    return all(path.exists() for path in required)


def _build_g1_cfg(n_sybil: int) -> dict[str, Any]:
    _, _, _, g1_cfg = GEN_MOD.build_default_profiles()
    if int(n_sybil) <= 0:
        return {"enable": False}

    g1_cfg = dict(g1_cfg)
    g1_cfg["enable"] = True
    g1_cfg["n_sybil"] = int(n_sybil)
    edges = int(g1_cfg.get("edges_per_sybil", 1))
    sell_back = int(bool(g1_cfg.get("sell_back", False)))
    g1_cfg["budget_caps"] = dict(g1_cfg.get("budget_caps") or {})
    g1_cfg["budget_caps"]["max_total_trades"] = int(n_sybil) * edges * (1 + sell_back)
    return g1_cfg


def _apply_contract_counts(contract_profiles: list[Any], cfg: dict[str, Any]) -> list[Any]:
    count_map = {str(k): int(v) for k, v in dict(cfg.get("CONTRACT_COUNTS", {})).items()}
    for profile in contract_profiles:
        category = str(getattr(profile, "category", ""))
        if category in count_map:
            profile.num_contracts = count_map[category]
    return contract_profiles


def _load_creator_trade_summary(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return dict(report.get("creator_trade_summary") or {})


def _base_market_user_count() -> int:
    _, user_profiles, _, _ = GEN_MOD.build_default_profiles()
    return int(sum(int(profile.n_users) for profile in user_profiles))


def resolve_scales(cfg: dict[str, Any]) -> list[int]:
    manual_scales = cfg.get("SCALES")
    exclude_scales = {int(x) for x in cfg.get("EXCLUDE_SCALES", [])}
    if manual_scales:
        scales = sorted({int(x) for x in manual_scales if int(x) >= 0 and int(x) not in exclude_scales})
        return scales

    base_users = _base_market_user_count()
    stop_share = float(cfg["STOP_AT_SYBIL_SHARE"])
    if not (0 < stop_share <= 1):
        raise ValueError("STOP_AT_SYBIL_SHARE must be in (0, 1].")

    stop_n_sybil = int(np.ceil((stop_share * base_users) / max(1e-12, 1.0 - stop_share)))
    start = max(1, int(cfg["AUTO_SCALE_START"]))
    multiplier = float(cfg["AUTO_SCALE_MULTIPLIER"])
    max_steps = max(1, int(cfg["AUTO_SCALE_MAX_STEPS"]))
    if multiplier <= 1.0:
        raise ValueError("AUTO_SCALE_MULTIPLIER must be > 1.0.")

    scales: list[int] = [0] if bool(cfg.get("INCLUDE_ZERO_SCALE", True)) else []
    current = start
    for _ in range(max_steps):
        scales.append(int(current))
        if current >= stop_n_sybil:
            break
        next_value = int(np.ceil(current * multiplier))
        if next_value <= current:
            next_value = current + 1
        current = next_value

    if stop_n_sybil not in scales:
        scales.append(int(stop_n_sybil))

    return sorted({int(x) for x in scales if int(x) not in exclude_scales})


def ensure_baseline_dataset(base_dir: Path) -> dict[str, Path]:
    dataset_paths = _dataset_paths(base_dir)
    required = ["data", "users", "contracts", "report", "config"]
    if bool(CONFIG["REUSE_EXISTING_DATASETS"]) and _all_exist(dataset_paths, required):
        print(f"[G1Compare] reuse baseline dataset: {base_dir}")
        return dataset_paths

    base_dir.mkdir(parents=True, exist_ok=True)

    gen = GEN_MOD.Data3BehaviorGenerator(
        chain_end_block=20_000_000,
        block_time_sec=12,
        seed=int(CONFIG["GENERATOR_SEED"]),
    )
    contract_profiles, user_profiles, creator_trade_cfg, g1_cfg = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, CONFIG)

    g1_cfg = {"enable": False}

    gen.generate_data3(
        contract_profiles=contract_profiles,
        user_profiles=user_profiles,
        horizon_months=36,
        save_dir=str(base_dir),
        out_name="RQdataG1.csv",
        creator_trade_cfg=creator_trade_cfg,
        g1_cfg=g1_cfg,
    )
    return dataset_paths


def generate_dataset_for_scale(n_sybil: int, data_dir: Path) -> dict[str, Path]:
    dataset_paths = _dataset_paths(data_dir)
    required = ["data", "users", "contracts", "report", "config"]
    if bool(CONFIG["REUSE_EXISTING_DATASETS"]) and _all_exist(dataset_paths, required):
        print(f"[G1Compare] reuse dataset for scale={n_sybil}: {data_dir}")
        return dataset_paths

    base_dir = Path(CONFIG["DATA_ROOT"]) / str(CONFIG["BASELINE_TAG"])
    base_paths = ensure_baseline_dataset(base_dir)
    if int(n_sybil) <= 0:
        return base_paths

    data_dir.mkdir(parents=True, exist_ok=True)

    gen = GEN_MOD.Data3BehaviorGenerator(
        chain_end_block=20_000_000,
        block_time_sec=12,
        seed=int(CONFIG["GENERATOR_SEED"]),
    )
    contract_profiles, user_profiles, _, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, CONFIG)
    creator_trade_summary = _load_creator_trade_summary(base_paths["report"])

    if bool(CONFIG["USE_INCREMENTAL_G1"]):
        gen.generate_g1_dataset_from_base(
            base_data_csv=str(base_paths["data"]),
            base_users_csv=str(base_paths["users"]),
            base_contracts_csv=str(base_paths["contracts"]),
            contract_profiles=contract_profiles,
            user_profiles=user_profiles,
            save_dir=str(data_dir),
            out_name="RQdataG1.csv",
            g1_cfg=_build_g1_cfg(int(n_sybil)),
            creator_trade_summary=creator_trade_summary,
        )
        return dataset_paths

    contract_profiles, user_profiles, creator_trade_cfg, _ = GEN_MOD.build_default_profiles()
    contract_profiles = _apply_contract_counts(contract_profiles, CONFIG)
    gen.generate_data3(
        contract_profiles=contract_profiles,
        user_profiles=user_profiles,
        horizon_months=36,
        save_dir=str(data_dir),
        out_name="RQdataG1.csv",
        creator_trade_cfg=creator_trade_cfg,
        g1_cfg=_build_g1_cfg(int(n_sybil)),
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


def collect_method_metrics(method: str, n_sybil: int, outdir: Path, param_value: float | None = None) -> dict[str, Any]:
    param_meta = METHOD_PARAM_META[method]
    if method == "our":
        contract_rank_col = "rank_C_contract"
        user_rank_col = "rank_C_user"
        contract_score_col = "cC"
        user_score_col = "uC"
    elif method == "birank":
        contract_rank_col = "rank_contract"
        user_rank_col = "rank_user"
        contract_score_col = "c_birank"
        user_score_col = "u_birank"
    elif method == "pagerank":
        contract_rank_col = "rank_contract"
        user_rank_col = "rank_user"
        contract_score_col = "c_pagerank"
        user_score_col = "u_pagerank"
    elif method == "eigentrust":
        contract_rank_col = "rank_contract"
        user_rank_col = "rank_user"
        contract_score_col = "c_eigentrust"
        user_score_col = "u_eigentrust"
    else:
        raise ValueError(f"Unknown method: {method}")

    contract_path, user_path = _score_file_paths(method, outdir, param_value=param_value)
    df_c = pd.read_csv(contract_path)
    df_u = pd.read_csv(user_path)

    sybil_c = df_c[df_c["contract_id"].astype(str).str.startswith("Sybil")].copy()
    g1_u = df_u[df_u["user"].astype(str).str.startswith("g1_sybil_")].copy()

    sybil_c["norm_rank"] = _normalized_rank(sybil_c[contract_rank_col], len(df_c))
    g1_u["norm_rank"] = _normalized_rank(g1_u[user_rank_col], len(df_u))

    topk_contract = int((sybil_c[contract_rank_col] <= 20).sum())
    topk_user = int((g1_u[user_rank_col] <= 200).sum())

    return {
        "method": method,
        "param_name": param_meta["param_name"],
        "param_value": float(param_value) if param_value is not None else float("nan"),
        "n_sybil": int(n_sybil),
        "n_contracts_total": int(len(df_c)),
        "n_users_total": int(len(df_u)),
        "sybil_contract_mean_rank": _safe_mean(sybil_c[contract_rank_col]),
        "sybil_contract_mean_norm_rank": _safe_mean(sybil_c["norm_rank"]),
        "sybil_contract_best_rank": float(sybil_c[contract_rank_col].min()) if len(sybil_c) else float("nan"),
        "sybil_contract_score_mean": _safe_mean(sybil_c[contract_score_col]),
        "sybil_contract_top20_count": topk_contract,
        "g1_user_mean_rank": _safe_mean(g1_u[user_rank_col]),
        "g1_user_mean_norm_rank": _safe_mean(g1_u["norm_rank"]),
        "g1_user_best_rank": float(g1_u[user_rank_col].min()) if len(g1_u) else float("nan"),
        "g1_user_score_mean": _safe_mean(g1_u[user_score_col]),
        "g1_user_top200_count": topk_user,
        "g1_user_count_observed": int(len(g1_u)),
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
        sub = sub.sort_values("n_sybil")
        axes[0].plot(
            sub["n_sybil"],
            sub["sybil_contract_mean_norm_rank"],
            marker="o",
            label=method,
        )
        axes[1].plot(
            sub["n_sybil"],
            sub["g1_user_mean_norm_rank"],
            marker="o",
            label=method,
        )

    axes[0].set_title("Sybil Contract Mean Normalized Rank (Primary=0.7)")
    axes[0].set_xlabel("n_sybil")
    axes[0].set_ylabel("Mean normalized rank")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("G1 User Mean Normalized Rank (Primary=0.7)")
    axes[1].set_xlabel("n_sybil")
    axes[1].set_ylabel("Mean normalized rank")
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "g1_scale_rank_compare.png", dpi=200, bbox_inches="tight")
    fig.savefig(outdir / f"g1_scale_rank_compare_primary_{primary_tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, sub in primary_df.groupby("method"):
        sub = sub.sort_values("n_sybil")
        axes[0].plot(sub["n_sybil"], sub["sybil_contract_best_rank"], marker="o", label=method)
        axes[1].plot(sub["n_sybil"], sub["g1_user_best_rank"], marker="o", label=method)

    axes[0].set_title("Best Sybil Contract Rank (Primary=0.7)")
    axes[0].set_xlabel("n_sybil")
    axes[0].set_ylabel("Best raw rank")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Best G1 User Rank (Primary=0.7)")
    axes[1].set_xlabel("n_sybil")
    axes[1].set_ylabel("Best raw rank")
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "g1_scale_best_rank_compare.png", dpi=200, bbox_inches="tight")
    fig.savefig(outdir / f"g1_scale_best_rank_compare_primary_{primary_tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, sub in primary_df.groupby("method"):
        sub = sub.sort_values("n_sybil")
        axes[0].plot(sub["n_sybil"], sub["sybil_contract_mean_rank"], marker="o", label=method)
        axes[1].plot(sub["n_sybil"], sub["g1_user_mean_rank"], marker="o", label=method)

    axes[0].set_title("Sybil Contract Mean Raw Rank (Primary=0.7)")
    axes[0].set_xlabel("n_sybil")
    axes[0].set_ylabel("Mean raw rank")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("G1 User Mean Raw Rank (Primary=0.7)")
    axes[1].set_xlabel("n_sybil")
    axes[1].set_ylabel("Mean raw rank")
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "g1_scale_mean_rank_compare.png", dpi=200, bbox_inches="tight")
    fig.savefig(outdir / f"g1_scale_mean_rank_compare_primary_{primary_tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_map = {
        "our": axes[0, 0],
        "birank": axes[0, 1],
        "pagerank": axes[1, 0],
        "eigentrust": axes[1, 1],
    }
    for method, ax in axes_map.items():
        method_df = summary_df[summary_df["method"] == method].copy()
        for param_value, sub in method_df.groupby("param_value"):
            sub = sub.sort_values("n_sybil")
            ax.plot(
                sub["n_sybil"],
                sub["sybil_contract_mean_norm_rank"],
                marker="o",
                label=f"{METHOD_PARAM_META[method]['param_name']}={float(param_value):.1f}",
            )
        ax.set_title(f"{method}: Sybil contract mean normalized rank")
        ax.set_xlabel("n_sybil")
        ax.set_ylabel("Mean normalized rank")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / "g1_scale_param_sweep_contracts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_map = {
        "our": axes[0, 0],
        "birank": axes[0, 1],
        "pagerank": axes[1, 0],
        "eigentrust": axes[1, 1],
    }
    for method, ax in axes_map.items():
        method_df = summary_df[summary_df["method"] == method].copy()
        for param_value, sub in method_df.groupby("param_value"):
            sub = sub.sort_values("n_sybil")
            ax.plot(
                sub["n_sybil"],
                sub["g1_user_mean_norm_rank"],
                marker="o",
                label=f"{METHOD_PARAM_META[method]['param_name']}={float(param_value):.1f}",
            )
        ax.set_title(f"{method}: G1 user mean normalized rank")
        ax.set_xlabel("n_sybil")
        ax.set_ylabel("Mean normalized rank")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / "g1_scale_param_sweep_users.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_compare(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)
    if cfg.get("PARAM_GRID"):
        grid = [float(x) for x in cfg["PARAM_GRID"]]
        cfg["OUR_BETA_LIST"] = [float(x) for x in cfg.get("OUR_BETA_LIST", grid)]
        cfg["BIRANK_ETA_LIST"] = [float(x) for x in cfg.get("BIRANK_ETA_LIST", grid)]
        cfg["PAGERANK_GAMMA_LIST"] = [float(x) for x in cfg.get("PAGERANK_GAMMA_LIST", grid)]
        cfg["EIGENTRUST_ALPHA_LIST"] = [float(x) for x in cfg.get("EIGENTRUST_ALPHA_LIST", grid)]

    data_root = Path(cfg["DATA_ROOT"])
    result_root = Path(cfg["RESULT_ROOT"])
    data_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    scales = resolve_scales(cfg)
    base_users = _base_market_user_count()
    print(
        f"[G1Compare] scales={scales} | base_users={base_users} | "
        f"stop_share={float(cfg['STOP_AT_SYBIL_SHARE']):.2f}"
    )

    records: list[dict[str, Any]] = []

    for n_sybil in scales:
        tag = _scale_tag(int(n_sybil))
        data_dir = data_root / tag
        our_outdir = result_root / f"our_{tag}"
        birank_outdir = result_root / f"birank_{tag}"
        pagerank_outdir = result_root / f"pagerank_{tag}"
        eigentrust_outdir = result_root / f"eigentrust_{tag}"

        print("\n" + "=" * 90)
        print(f"[G1Compare] scale={n_sybil}")
        print("=" * 90)

        dataset_paths = generate_dataset_for_scale(int(n_sybil), data_dir)

        if cfg["RUN_OUR"]:
            if bool(cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("our", our_outdir, _method_param_values(cfg, "our")):
                print(f"[G1Compare] reuse OUR result: {our_outdir}")
            else:
                run_our_on_dataset(dataset_paths, our_outdir, cfg)
            for param_value in _method_param_values(cfg, "our"):
                records.append(collect_method_metrics("our", int(n_sybil), our_outdir, param_value=param_value))

        if cfg["RUN_BIRANK"]:
            if bool(cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("birank", birank_outdir, _method_param_values(cfg, "birank")):
                print(f"[G1Compare] reuse BiRank result: {birank_outdir}")
            else:
                run_birank_on_dataset(our_outdir, birank_outdir, cfg)
            for param_value in _method_param_values(cfg, "birank"):
                records.append(collect_method_metrics("birank", int(n_sybil), birank_outdir, param_value=param_value))

        if cfg["RUN_PAGERANK"]:
            if bool(cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("pagerank", pagerank_outdir, _method_param_values(cfg, "pagerank")):
                print(f"[G1Compare] reuse PageRank result: {pagerank_outdir}")
            else:
                run_pagerank_on_dataset(our_outdir, pagerank_outdir, cfg)
            for param_value in _method_param_values(cfg, "pagerank"):
                records.append(collect_method_metrics("pagerank", int(n_sybil), pagerank_outdir, param_value=param_value))

        if cfg["RUN_EIGENTRUST"]:
            if bool(cfg["REUSE_EXISTING_RESULTS"]) and _result_exists("eigentrust", eigentrust_outdir, _method_param_values(cfg, "eigentrust")):
                print(f"[G1Compare] reuse EigenTrust result: {eigentrust_outdir}")
            else:
                run_eigentrust_on_dataset(dataset_paths, our_outdir, eigentrust_outdir, cfg)
            for param_value in _method_param_values(cfg, "eigentrust"):
                records.append(collect_method_metrics("eigentrust", int(n_sybil), eigentrust_outdir, param_value=param_value))

    summary_df = pd.DataFrame(records).sort_values(["method", "param_value", "n_sybil"]).reset_index(drop=True)
    summary_path = result_root / "g1_scale_compare_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_results(summary_df, result_root, cfg)

    print("\n" + "=" * 90)
    print("[G1Compare] finished")
    print("=" * 90)
    print(f"summary: {summary_path}")
    print(f"plot   : {result_root / 'g1_scale_rank_compare.png'}")
    print(f"plot   : {result_root / 'g1_scale_best_rank_compare.png'}")

    return summary_df


if __name__ == "__main__":
    run_compare()
