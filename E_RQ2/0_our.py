from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "output" / "A" / "15_RQ2compare_G3A2_v3"
OUTDIR = ROOT / "output" / "E" / "0_our_paper"

CONFIG = {
    "RAW_DATA_CSV": DATA_ROOT / "RQdataG3A2.csv",
    "USERS_META_CSV": DATA_ROOT / "RQdataG3A2_users.csv",
    "CONTRACTS_META_CSV": DATA_ROOT / "RQdataG3A2_contracts.csv",
    "OUTDIR": OUTDIR,
    "STAGE_A_ANCHOR_RATIO": 0.03,
    "STAGE_A_ETA": 0.85,
    "STAGE_A_MAX_ITER": 200,
    "STAGE_A_TOL": 1e-10,
    "STAGE_B_TAU_S": 200_000.0,
    "STAGE_B_TAU_H": 50_000.0,
    "STAGE_C_TAU_DECAY": 216_000.0,
    "STAGE_C_ALPHA": 0.85,
    "STAGE_C_USER_PRIOR_MODE": "stageb_u0",
    "STAGE_C_BETA": 0.7,
    "STAGE_C_MAX_ITER": 200,
    "STAGE_C_TOL": 1e-10,
    "STAGE_C_INCLUDE_NON_ACQUISITION": True,
}


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE_A = _load_module("paper_stage_a", ROOT / "E_RQ2" / "0_stageA.py")
STAGE_B = _load_module("paper_stage_b", ROOT / "E_RQ2" / "0_stageB.py")
STAGE_C = _load_module("paper_stage_c", ROOT / "E_RQ2" / "0_stageC.py")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(data: dict[str, Any], path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_sparse(matrix: sparse.csr_matrix, path: Path) -> None:
    _ensure_dir(path.parent)
    sparse.save_npz(path, matrix)


def run_pipeline(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(CONFIG if cfg is None else cfg)
    outdir = Path(cfg["OUTDIR"])
    _ensure_dir(outdir)

    df = pd.read_csv(cfg["RAW_DATA_CSV"])
    users_meta = pd.read_csv(cfg["USERS_META_CSV"])
    contracts_meta = pd.read_csv(cfg["CONTRACTS_META_CSV"])
    tau_end = int(pd.to_numeric(df["block_number"], errors="coerce").fillna(0).max()) if len(df) else 0

    stage_a = STAGE_A.run_stage_a(
        df=df,
        anchor_ratio=float(cfg["STAGE_A_ANCHOR_RATIO"]),
        eta=float(cfg["STAGE_A_ETA"]),
        max_iter=int(cfg["STAGE_A_MAX_ITER"]),
        tol=float(cfg["STAGE_A_TOL"]),
        include_mint_gas=True,
    )
    stage_a.user_scores.to_csv(outdir / "1_stageA_user_scores.csv", index=False)
    stage_a.contract_scores.to_csv(outdir / "1_stageA_contract_scores.csv", index=False)
    pd.Series(stage_a.anchors, name="contract_id").to_csv(outdir / "1_stageA_anchors.csv", index=False)
    _save_sparse(stage_a.W_hist, outdir / "1_stageA_W_hist.npz")
    _save_json(stage_a.meta, outdir / "1_stageA_meta.json")

    stage_b = STAGE_B.run_stage_b(
        df=df,
        contract_scores_a=stage_a.contract_scores,
        anchors=stage_a.anchors,
        users_meta=users_meta,
        tau_end=tau_end,
        tau_s=float(cfg["STAGE_B_TAU_S"]),
        tau_h=float(cfg["STAGE_B_TAU_H"]),
    )
    stage_b.user_scores.to_csv(outdir / "2_stageB_user_scores.csv", index=False)
    stage_b.event_scores.to_csv(outdir / "2_stageB_event_scores.csv", index=False)
    _save_json(stage_b.meta, outdir / "2_stageB_meta.json")

    stage_c_user_scores = stage_b.user_scores.copy()
    stage_c_user_prior_mode = str(cfg.get("STAGE_C_USER_PRIOR_MODE", "stageb_u0")).strip().lower()
    if stage_c_user_prior_mode == "uniform":
        if "u0" not in stage_c_user_scores.columns:
            raise ValueError("Stage-B user scores do not contain 'u0' required for Stage-C prior override.")
        n_users = max(len(stage_c_user_scores), 1)
        stage_c_user_scores["u0"] = 1.0 / float(n_users)
    elif stage_c_user_prior_mode != "stageb_u0":
        raise ValueError("STAGE_C_USER_PRIOR_MODE must be 'stageb_u0' or 'uniform'.")

    stage_c = STAGE_C.run_stage_c(
        df=df,
        stage_b_user_scores=stage_c_user_scores,
        stage_a_contract_scores=stage_a.contract_scores,
        tau_end=tau_end,
        tau_decay=float(cfg["STAGE_C_TAU_DECAY"]),
        tau_s=float(cfg["STAGE_B_TAU_S"]),
        tau_h=float(cfg["STAGE_B_TAU_H"]),
        alpha=float(cfg["STAGE_C_ALPHA"]),
        beta=float(cfg["STAGE_C_BETA"]),
        max_iter=int(cfg["STAGE_C_MAX_ITER"]),
        tol=float(cfg["STAGE_C_TOL"]),
        include_non_acquisition=bool(cfg["STAGE_C_INCLUDE_NON_ACQUISITION"]),
    )
    stage_c.user_scores.to_csv(outdir / "3_stageC_user_scores.csv", index=False)
    stage_c.contract_scores.to_csv(outdir / "3_stageC_contract_scores.csv", index=False)
    stage_c.event_scores.to_csv(outdir / "3_stageC_event_scores.csv", index=False)
    _save_sparse(stage_c.W_rec, outdir / "3_stageC_W_rec.npz")
    _save_json(stage_c.meta, outdir / "3_stageC_meta.json")

    pipeline_meta = {
        "raw_data_csv": str(cfg["RAW_DATA_CSV"]),
        "users_meta_csv": str(cfg["USERS_META_CSV"]),
        "contracts_meta_csv": str(cfg["CONTRACTS_META_CSV"]),
        "outdir": str(outdir),
        "tau_end": tau_end,
        "n_events": int(len(df)),
        "n_users_meta": int(len(users_meta)),
        "n_contracts_meta": int(len(contracts_meta)),
        "stage_c_user_prior_mode": stage_c_user_prior_mode,
    }
    _save_json(pipeline_meta, outdir / "0_pipeline_meta.json")

    return {
        "config": cfg,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "stage_c": stage_c,
        "pipeline_meta": pipeline_meta,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("[our.py] paper-aligned pipeline finished")
    print(f"outdir: {result['pipeline_meta']['outdir']}")
