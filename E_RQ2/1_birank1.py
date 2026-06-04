from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


EPS = 1e-12
ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    "INPUT_DIR": ROOT / "output" / "E" / "0_our_paper",
    "OUTDIR": ROOT / "output" / "E" / "1_birank",
    "W_REC_NPZ": "3_stageC_W_rec.npz",
    "EVENT_SCORES_CSV": "3_stageC_event_scores.csv",
    "USER_SCORES_CSV": "3_stageC_user_scores.csv",
    "CONTRACT_SCORES_CSV": "3_stageC_contract_scores.csv",
    "STAGEC_META_JSON": "3_stageC_meta.json",
    "W_TRADE_NPZ": "3_stageC_W_trade.npz",
    "W_USERS_CSV": "3_stageC_W_users.csv",
    "W_CONTRACTS_CSV": "3_stageC_W_contracts.csv",
    "WINDOW_META_CSV": "3_stageC_window_meta.csv",
    "ETA_LIST": [0.1, 0.3, 0.5, 0.7, 0.9],
    "PRIMARY_ETA": 0.7,
    "USER_PRIOR_MODE": "uniform",
    "CONTRACT_PRIOR_MODE": "uniform",
    "MAX_ITER": 80,
    "TOL": 1e-10,
    "SHOW_TOP_USERS": 20,
    "SHOW_TOP_CONTRACTS": 20,
    "OUT_USER_SCORES_CSV": "1_birank_user_scores.csv",
    "OUT_CONTRACT_SCORES_CSV": "1_birank_contract_scores.csv",
}


def _load_new_layout(input_dir: Path, cfg: dict) -> tuple[sparse.csr_matrix, list[str], list[str], dict | None]:
    w_path = input_dir / cfg["W_REC_NPZ"]
    event_scores_path = input_dir / cfg["EVENT_SCORES_CSV"]
    meta_path = input_dir / cfg["STAGEC_META_JSON"]
    if not w_path.exists():
        raise FileNotFoundError(f"Missing Stage-C matrix artifact: {w_path}")
    if not event_scores_path.exists():
        raise FileNotFoundError(f"Missing Stage-C event artifact: {event_scores_path}")

    W = sparse.load_npz(w_path).tocsr()
    event_scores = pd.read_csv(event_scores_path, usecols=["user", "contract_id"])
    users = sorted(event_scores["user"].astype(str).unique().tolist())
    contracts = sorted(event_scores["contract_id"].astype(str).unique().tolist())
    if W.shape != (len(users), len(contracts)):
        raise ValueError(f"Stage-C W_rec shape mismatch: W={W.shape}, users={len(users)}, contracts={len(contracts)}")

    meta = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = dict(json.load(f))
    return W, users, contracts, meta


def _load_legacy_layout(input_dir: Path, cfg: dict) -> tuple[sparse.csr_matrix, list[str], list[str], dict | None]:
    w_path = input_dir / cfg["W_TRADE_NPZ"]
    users_path = input_dir / cfg["W_USERS_CSV"]
    contracts_path = input_dir / cfg["W_CONTRACTS_CSV"]
    meta_path = input_dir / cfg["WINDOW_META_CSV"]

    for path in [w_path, users_path, contracts_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing WRec artifact: {path}")

    W = sparse.load_npz(w_path).tocsr()
    users = pd.read_csv(users_path)["user"].astype(str).tolist()
    contracts = pd.read_csv(contracts_path)["contract_id"].astype(str).tolist()
    meta = pd.read_csv(meta_path).iloc[0].to_dict() if meta_path.exists() else None
    if W.shape != (len(users), len(contracts)):
        raise ValueError(f"WRec shape mismatch: W={W.shape}, users={len(users)}, contracts={len(contracts)}")
    return W, users, contracts, meta


def load_wrec(input_dir: Path, cfg: dict):
    if (input_dir / cfg["W_REC_NPZ"]).exists():
        return _load_new_layout(input_dir, cfg)
    return _load_legacy_layout(input_dir, cfg)


def build_symmetric_operator(W: sparse.csr_matrix):
    d_u = np.array(W.sum(axis=1)).flatten()
    d_c = np.array(W.sum(axis=0)).flatten()
    d_u[d_u <= 0] = 1.0
    d_c[d_c <= 0] = 1.0

    Su = sparse.diags(1.0 / np.sqrt(d_u))
    Sc = sparse.diags(1.0 / np.sqrt(d_c))
    S = Su @ W @ Sc
    return S, S.T


def build_uniform_prior(n_nodes: int, mode: str, label: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode != "uniform":
        raise ValueError(f"Unsupported {label} prior mode: {mode}. Only 'uniform' is allowed in baseline BiRank.")
    return np.ones(n_nodes, dtype=float) / max(n_nodes, 1)


def birank_uniform(
    W: sparse.csr_matrix,
    eta: float,
    max_iter: int,
    tol: float,
    user_prior_mode: str,
    contract_prior_mode: str,
):
    n_u, n_c = W.shape
    S, ST = build_symmetric_operator(W)

    u0 = build_uniform_prior(n_u, user_prior_mode, "user")
    c0 = build_uniform_prior(n_c, contract_prior_mode, "contract")
    u = u0.reshape(-1, 1)
    c = c0.reshape(-1, 1)
    u0v = u0.reshape(-1, 1)
    c0v = c0.reshape(-1, 1)

    for _ in range(max_iter):
        c_new = eta * (ST @ u) + (1.0 - eta) * c0v
        u_new = eta * (S @ c_new) + (1.0 - eta) * u0v
        c_new = c_new / (c_new.sum() + EPS)
        u_new = u_new / (u_new.sum() + EPS)
        if np.max(np.abs(c_new - c)) < tol and np.max(np.abs(u_new - u)) < tol:
            c, u = c_new, u_new
            break
        c, u = c_new, u_new

    return u.flatten(), c.flatten()


def save_scores(outdir: Path, users: list[str], contracts: list[str], u: np.ndarray, c: np.ndarray, eta: float, cfg: dict):
    outdir.mkdir(parents=True, exist_ok=True)

    df_u = pd.DataFrame({"user": users, "u_birank": u}).sort_values("u_birank", ascending=False).reset_index(drop=True)
    df_u["rank_user"] = np.arange(1, len(df_u) + 1)

    df_c = pd.DataFrame({"contract_id": contracts, "c_birank": c}).sort_values("c_birank", ascending=False).reset_index(drop=True)
    df_c["rank_contract"] = np.arange(1, len(df_c) + 1)

    base_u = outdir / cfg["OUT_USER_SCORES_CSV"]
    base_c = outdir / cfg["OUT_CONTRACT_SCORES_CSV"]
    suf_u = outdir / f"1_birank_user_scores_eta{eta:.2f}.csv"
    suf_c = outdir / f"1_birank_contract_scores_eta{eta:.2f}.csv"

    df_u.to_csv(suf_u, index=False)
    df_c.to_csv(suf_c, index=False)
    if abs(eta - float(cfg["PRIMARY_ETA"])) < 1e-12:
        df_u.to_csv(base_u, index=False)
        df_c.to_csv(base_c, index=False)

    return df_u, df_c


def summarize_contracts(df_c: pd.DataFrame):
    out = df_c.copy()
    out["category"] = out["contract_id"].astype(str).str.split("_").str[0]
    sybil = out[out["category"] == "Sybil"][["contract_id", "c_birank", "rank_contract"]].copy()
    cat_mean = out.groupby("category")["rank_contract"].mean().sort_values()
    return sybil, cat_mean


def run(config: dict | None = None):
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)

    input_dir = Path(cfg["INPUT_DIR"])
    outdir = Path(cfg["OUTDIR"])

    W, users, contracts, meta = load_wrec(input_dir, cfg)

    print("=" * 90)
    print("[1_birank1] Baseline on WRec with symmetric normalization and uniform priors")
    print("=" * 90)
    print(f"input dir : {input_dir}")
    print(f"output dir: {outdir}")
    print(f"user prior: {cfg['USER_PRIOR_MODE']}")
    print(f"contract prior: {cfg['CONTRACT_PRIOR_MODE']}")
    if meta is not None:
        meta_show = " | ".join(
            [
                f"{k}: {meta.get(k)}"
                for k in ["window_blocks", "start_block", "end_block", "nnz", "n_users", "n_contracts", "mode", "tau_end", "tau_decay"]
                if k in meta
            ]
        )
        if meta_show:
            print(meta_show)

    last_user_df = None
    last_contract_df = None
    for eta in cfg["ETA_LIST"]:
        u, c = birank_uniform(
            W,
            eta=float(eta),
            max_iter=int(cfg["MAX_ITER"]),
            tol=float(cfg["TOL"]),
            user_prior_mode=str(cfg["USER_PRIOR_MODE"]),
            contract_prior_mode=str(cfg["CONTRACT_PRIOR_MODE"]),
        )
        df_u, df_c = save_scores(outdir, users, contracts, u, c, eta=float(eta), cfg=cfg)
        sybil_df, cat_mean = summarize_contracts(df_c)

        print("\n" + "=" * 90)
        print(f"[1_birank1] eta={eta:.2f}")
        print("=" * 90)
        print("\nTop contracts:")
        print(df_c.head(cfg["SHOW_TOP_CONTRACTS"]).to_string(index=False))
        print("\nTop users:")
        print(df_u.head(cfg["SHOW_TOP_USERS"]).to_string(index=False))
        print("\nSybil ranks:")
        if len(sybil_df) == 0:
            print("No Sybil contracts found.")
        else:
            print(sybil_df.to_string(index=False))
        print("\nMean contract rank by category:")
        print(cat_mean.to_string())

        last_user_df = df_u
        last_contract_df = df_c

    return {
        "users": users,
        "contracts": contracts,
        "meta": meta,
        "last_user_df": last_user_df,
        "last_contract_df": last_contract_df,
    }


if __name__ == "__main__":
    run()
