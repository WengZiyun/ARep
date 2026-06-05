from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    "RAW_DATA_CSV": ROOT / "output" / "A" / "11_RQ2data" / "RQdataG1.csv",
    "USERS_META_CSV": ROOT / "output" / "A" / "11_RQ2data" / "RQdataG1_users.csv",
    "CONTRACTS_META_CSV": ROOT / "output" / "A" / "11_RQ2data" / "RQdataG1_contracts.csv",
    "OUTDIR": ROOT / "output" / "E" / "0_our",
    "STAGE_A_ALPHA": 0.7,
    "STAGE_A_BETA": 0.85,
    "STAGE_A_AGE_POWER": 2.0,
    "STAGE_B_TRUSTED_RATIO": 0.03,
    "STAGE_B_TAU_S": 200_000,
    "STAGE_B_TAU_H": 50_000,
    "STAGE_C_MODE": "decay_full",
    "STAGE_C_WINDOW_BLOCKS": 216_000,
    "STAGE_C_TAU_DECAY": 216_000,
    "STAGE_C_MU": 1.0,
    "STAGE_C_W_GAS": 1.0,
    "STAGE_C_W_VAL": 0.0,
    "STAGE_C_INCLUDE_MINT_GAS": True,
    "STAGE_C_INCLUDE_MINT_EDGE": False,
    "STAGE_C_TRADE_ONLY": True,
    "STAGE_C_ALPHA": 0.85,
    "STAGE_C_BETA_LIST": [0.7, 0.6, 0.5, 0.4, 0.1],
    "STAGE_C_PRIMARY_BETA": 0.7,
    "STAGE_C_NORM": "random_walk",
    "STAGE_C_MAX_ITER": 80,
    "STAGE_C_TOL": 1e-10,
    "RUN_LABEL_CHECK": True,
}


def _load_module(name: str, relative_path: str) -> Any:
    file_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE_AB = _load_module("arep_stage_ab", "4_ARep/2_3 copy.py")
STAGE_PREP = _load_module("arep_stage_prep", "4_ARep/3_ copy.py")
STAGE_C = _load_module("arep_stage_c", "4_ARep/4_copy.py")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_parent(path)
    df.to_csv(path, index=False)


def _save_lines(items: list[str], path: Path) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def _output_paths(outdir: Path) -> dict[str, Path]:
    return {
        "stage_a_contract_scores": outdir / "2_stageA_contract_scores.csv",
        "stage_b_user_scores": outdir / "2_stageB_user_scores.csv",
        "stage_b_trusted_contracts": outdir / "2_stageB_trusted_contracts.txt",
        "stage_b_norm_scores": outdir / "3_stageB_user_scores_norm.csv",
        "stage_c_u0": outdir / "3_stageC_u0_aligned.npy",
        "stage_c_w": outdir / "3_stageC_W_trade.npz",
        "stage_c_users": outdir / "3_stageC_W_users.csv",
        "stage_c_contracts": outdir / "3_stageC_W_contracts.csv",
        "stage_c_window_meta": outdir / "3_stageC_window_meta.csv",
        "stage_c_user_scores": outdir / "4_stageC_user_scores.csv",
        "stage_c_contract_scores": outdir / "4_stageC_contract_scores.csv",
    }


def _stage_c_output_paths(outdir: Path, beta: float, primary_beta: float) -> dict[str, Path]:
    base_u = outdir / "4_stageC_user_scores.csv"
    base_c = outdir / "4_stageC_contract_scores.csv"
    suf_u = outdir / f"4_stageC_user_scores_beta{beta:.2f}.csv"
    suf_c = outdir / f"4_stageC_contract_scores_beta{beta:.2f}.csv"
    return {
        "base_u": base_u,
        "base_c": base_c,
        "suf_u": suf_u,
        "suf_c": suf_c,
        "is_primary": abs(beta - primary_beta) < 1e-12,
    }


def _save_stage_c_scores(
    outdir: Path,
    users: list[str],
    contracts: list[str],
    uC: np.ndarray,
    cC: np.ndarray,
    beta: float,
    primary_beta: float,
    meta: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)

    df_u = pd.DataFrame({"user": users, "uC": uC}).sort_values("uC", ascending=False).reset_index(drop=True)
    df_u["rank_C_user"] = np.arange(1, len(df_u) + 1)

    df_c = pd.DataFrame({"contract_id": contracts, "cC": cC}).sort_values("cC", ascending=False).reset_index(drop=True)
    df_c["rank_C_contract"] = np.arange(1, len(df_c) + 1)

    p = _stage_c_output_paths(outdir, beta, primary_beta)
    df_u.to_csv(p["suf_u"], index=False)
    df_c.to_csv(p["suf_c"], index=False)
    if p["is_primary"]:
        df_u.to_csv(p["base_u"], index=False)
        df_c.to_csv(p["base_c"], index=False)

    print("\n" + "=" * 90)
    print(f"[Stage C] Personalized BiRank finished [OK] (alpha={CONFIG['STAGE_C_ALPHA']:.2f}, beta={beta:.2f}, norm={CONFIG['STAGE_C_NORM']})")
    print("=" * 90)
    if meta:
        meta_show = " | ".join(
            [f"{k}: {meta.get(k)}" for k in ["window_blocks", "start_block", "end_block", "nnz", "n_rows", "n_cols", "mode"] if k in meta]
        )
        if meta_show:
            print(meta_show)

    print("\nTop contracts by Stage C:")
    print(df_c.head(int(STAGE_C.CONFIG["SHOW_TOP_CONTRACTS"])).to_string(index=False))
    print("\nTop users by Stage C:")
    print(df_u.head(int(STAGE_C.CONFIG["SHOW_TOP_USERS"])).to_string(index=False))
    print(f"\n[Saved] contracts -> {p['suf_c']}")
    print(f"[Saved] users     -> {p['suf_u']}")
    if p["is_primary"]:
        print(f"[Saved:PRIMARY] contracts -> {p['base_c']}")
        print(f"[Saved:PRIMARY] users     -> {p['base_u']}")

    return df_u, df_c


def run_stage_ab(cfg: dict[str, Any]) -> dict[str, Any]:
    data_csv = Path(cfg["RAW_DATA_CSV"])
    users_csv = Path(cfg["USERS_META_CSV"])
    contracts_csv = Path(cfg["CONTRACTS_META_CSV"])
    outdir = Path(cfg["OUTDIR"])
    out_paths = _output_paths(outdir)

    df = pd.read_csv(data_csv)
    users_meta = pd.read_csv(users_csv)
    contracts_meta = pd.read_csv(contracts_csv)

    if "tx_index_in_block" not in df.columns:
        df["tx_index_in_block"] = 0

    df = STAGE_AB.enrich_hold_and_age(df, contracts_meta)

    contract_scores_A = STAGE_AB.compute_A_contract_scores(
        df=df,
        contracts_meta=contracts_meta,
        alpha_A=float(cfg["STAGE_A_ALPHA"]),
        beta_A=float(cfg["STAGE_A_BETA"]),
        age_power=float(cfg["STAGE_A_AGE_POWER"]),
    )
    contract_scores_A["rank_A"] = np.arange(1, len(contract_scores_A) + 1)
    _save_csv(contract_scores_A, out_paths["stage_a_contract_scores"])

    scores_B, trusted = STAGE_AB.compute_B_user_scores(
        df=df,
        users_meta=users_meta,
        contract_scores_A=contract_scores_A,
        trusted_ratio=float(cfg["STAGE_B_TRUSTED_RATIO"]),
        tau_s=int(cfg["STAGE_B_TAU_S"]),
        tau_h=int(cfg["STAGE_B_TAU_H"]),
    )
    _save_csv(scores_B, out_paths["stage_b_user_scores"])
    _save_lines(trusted, out_paths["stage_b_trusted_contracts"])

    return {
        "df": df,
        "users_meta": users_meta,
        "contracts_meta": contracts_meta,
        "contract_scores_A": contract_scores_A,
        "scores_B": scores_B,
        "trusted_contracts": trusted,
        "out_paths": out_paths,
    }


def run_stage_c_prepare(cfg: dict[str, Any], stage_ab_ctx: dict[str, Any]) -> dict[str, Any]:
    outdir = Path(cfg["OUTDIR"])
    out_paths = stage_ab_ctx["out_paths"]

    STAGE_PREP.CONFIG["STAGEB_USER_SCORES_CSV"] = str(out_paths["stage_b_user_scores"])
    STAGE_PREP.CONFIG["DATA_CSV"] = str(cfg["RAW_DATA_CSV"])
    STAGE_PREP.CONFIG["OUTDIR"] = str(outdir)
    STAGE_PREP.CONFIG["MODE"] = str(cfg["STAGE_C_MODE"])
    STAGE_PREP.CONFIG["WINDOW_BLOCKS"] = int(cfg["STAGE_C_WINDOW_BLOCKS"])
    STAGE_PREP.CONFIG["TAU_DECAY"] = float(cfg["STAGE_C_TAU_DECAY"])
    STAGE_PREP.CONFIG["TAU_S"] = float(cfg["STAGE_B_TAU_S"])
    STAGE_PREP.CONFIG["TAU_H"] = float(cfg["STAGE_B_TAU_H"])
    STAGE_PREP.CONFIG["MU"] = float(cfg["STAGE_C_MU"])
    STAGE_PREP.CONFIG["W_GAS"] = float(cfg["STAGE_C_W_GAS"])
    STAGE_PREP.CONFIG["W_VAL"] = float(cfg["STAGE_C_W_VAL"])
    STAGE_PREP.CONFIG["INCLUDE_MINT_GAS"] = bool(cfg["STAGE_C_INCLUDE_MINT_GAS"])
    STAGE_PREP.CONFIG["INCLUDE_MINT_EDGE"] = bool(cfg.get("STAGE_C_INCLUDE_MINT_EDGE", False))
    STAGE_PREP.CONFIG["TRADE_ONLY"] = bool(cfg["STAGE_C_TRADE_ONLY"])

    dfB_norm, p0_map = STAGE_PREP.load_stageB_and_make_p0(
        str(out_paths["stage_b_user_scores"]),
        score_col=STAGE_PREP.CONFIG["STAGEB_SCORE_COL"],
        user_col=STAGE_PREP.CONFIG["STAGEB_USER_COL"],
    )
    W_trade, users_W, contracts_W, meta = STAGE_PREP.build_Wtrade_stageA_like(
        data_csv=str(cfg["RAW_DATA_CSV"]),
        mu=float(cfg["STAGE_C_MU"]),
        w_gas=float(cfg["STAGE_C_W_GAS"]),
        w_val=float(cfg["STAGE_C_W_VAL"]),
        include_mint_gas=bool(cfg["STAGE_C_INCLUDE_MINT_GAS"]),
    )
    u0 = STAGE_PREP.align_user_prior_to_W(users_W, p0_map)
    STAGE_PREP.save_stageC_inputs(str(outdir), dfB_norm, W_trade, users_W, contracts_W, u0, meta)

    return {
        "dfB_norm": dfB_norm,
        "W_trade": W_trade,
        "users_W": users_W,
        "contracts_W": contracts_W,
        "u0": u0,
        "meta": meta,
    }


def run_stage_c(cfg: dict[str, Any]) -> dict[str, Any]:
    outdir = Path(cfg["OUTDIR"])

    STAGE_C.CONFIG["OUTDIR"] = str(outdir)
    STAGE_C.CONFIG["DATA_CSV"] = str(cfg["RAW_DATA_CSV"])
    STAGE_C.CONFIG["RUN_LABEL_CHECK"] = bool(cfg["RUN_LABEL_CHECK"])
    STAGE_C.CONFIG["ALPHA_C"] = float(cfg["STAGE_C_ALPHA"])
    STAGE_C.CONFIG["BETA_LIST"] = list(cfg["STAGE_C_BETA_LIST"])
    STAGE_C.CONFIG["PRIMARY_BETA"] = float(cfg["STAGE_C_PRIMARY_BETA"])
    STAGE_C.CONFIG["NORM"] = str(cfg["STAGE_C_NORM"])
    STAGE_C.CONFIG["MAX_ITER"] = int(cfg["STAGE_C_MAX_ITER"])
    STAGE_C.CONFIG["TOL"] = float(cfg["STAGE_C_TOL"])
    STAGE_C.CONFIG["STAGEA_CONTRACT_SCORES_CSV"] = "2_stageA_contract_scores.csv"

    W_trade, users_W, contracts_W, u0_aligned, meta = STAGE_C.load_stageC_inputs(str(outdir))
    d_c_raw = np.array(W_trade.sum(axis=0)).flatten()

    if STAGE_C.CONFIG.get("RUN_LABEL_CHECK", False):
        df_lc = STAGE_C.label_check_from_data(str(cfg["RAW_DATA_CSV"]), contracts_in_W=set(contracts_W))
        if df_lc is not None:
            _save_csv(df_lc, outdir / "4_stageC_label_check_report.csv")

    STAGE_C.diagnose_sybil_recency(str(outdir), W_trade, users_W, contracts_W, norm=str(cfg["STAGE_C_NORM"]), sybil_prefix="Sybil_")
    STAGE_C.diagnose_u0_sybil_association(
        str(outdir),
        W_trade,
        users_W,
        contracts_W,
        u0_aligned,
        norm=str(cfg["STAGE_C_NORM"]),
        sybil_prefix="Sybil_",
        topk_contrib=int(STAGE_C.CONFIG.get("DIAG_TOPK_CONTRIB", 200)),
    )

    last_user_df = None
    last_contract_df = None
    for beta in cfg["STAGE_C_BETA_LIST"]:
        uC, cC = STAGE_C.birank_personalized(
            W=W_trade,
            u0=u0_aligned,
            alpha=float(cfg["STAGE_C_ALPHA"]),
            beta=float(beta),
            norm=str(cfg["STAGE_C_NORM"]),
            c0=None,
            max_iter=int(cfg["STAGE_C_MAX_ITER"]),
            tol=float(cfg["STAGE_C_TOL"]),
            contracts=contracts_W,
            outdir=str(outdir),
        )
        df_uC, df_cC = _save_stage_c_scores(
            outdir=outdir,
            users=users_W,
            contracts=contracts_W,
            uC=uC,
            cC=cC,
            beta=float(beta),
            primary_beta=float(cfg["STAGE_C_PRIMARY_BETA"]),
            meta=meta,
        )
        STAGE_C.diagnostics(W_trade, users_W, contracts_W, u0_aligned, uC, cC, d_c_raw, beta=float(beta), norm=str(cfg["STAGE_C_NORM"]))
        STAGE_C.compare_stageA_stageC(str(outdir), df_cC, topk=int(STAGE_C.CONFIG["COMPARE_TOPK"]))
        last_user_df = df_uC
        last_contract_df = df_cC

    return {
        "W_trade": W_trade,
        "users_W": users_W,
        "contracts_W": contracts_W,
        "u0_aligned": u0_aligned,
        "meta": meta,
        "last_user_df": last_user_df,
        "last_contract_df": last_contract_df,
    }


def run_pipeline(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(CONFIG if cfg is None else cfg)
    Path(cfg["OUTDIR"]).mkdir(parents=True, exist_ok=True)

    stage_ab_ctx = run_stage_ab(cfg)
    stage_c_prep_ctx = run_stage_c_prepare(cfg, stage_ab_ctx)
    stage_c_ctx = run_stage_c(cfg)

    return {
        "config": cfg,
        "stage_ab": stage_ab_ctx,
        "stage_c_prepare": stage_c_prep_ctx,
        "stage_c": stage_c_ctx,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("\n" + "=" * 90)
    print("[0_our] Pipeline finished")
    print("=" * 90)
    print(f"raw data   : {result['config']['RAW_DATA_CSV']}")
    print(f"output dir : {result['config']['OUTDIR']}")
    print("saved core outputs:")
    for name, path in _output_paths(Path(result["config"]["OUTDIR"])).items():
        print(f"  {name}: {path}")
