from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


EPS = 1e-12
ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    "INPUT_DIR": ROOT / "output" / "E" / "0_our",
    "OUTDIR": ROOT / "output" / "E" / "2_pagerank",
    "W_TRADE_NPZ": "3_stageC_W_trade.npz",
    "W_USERS_CSV": "3_stageC_W_users.csv",
    "W_CONTRACTS_CSV": "3_stageC_W_contracts.csv",
    "WINDOW_META_CSV": "3_stageC_window_meta.csv",
    "U0_ALIGNED_NPY": "3_stageC_u0_aligned.npy",
    "GAMMA_LIST": [0.7, 0.6, 0.5, 0.4, 0.1],
    "PRIMARY_GAMMA": 0.7,
    "RESTART_MODE": "uniform",
    "RESTART_USER_MASS": 0.5,
    "RESTART_CONTRACT_MASS": 0.5,
    "MAX_ITER": 80,
    "TOL": 1e-10,
    "SHOW_TOP_USERS": 20,
    "SHOW_TOP_CONTRACTS": 20,
    "OUT_USER_SCORES_CSV": "2_pagerank_user_scores.csv",
    "OUT_CONTRACT_SCORES_CSV": "2_pagerank_contract_scores.csv",
}


def load_wrec(input_dir: Path):
    w_path = input_dir / CONFIG["W_TRADE_NPZ"]
    users_path = input_dir / CONFIG["W_USERS_CSV"]
    contracts_path = input_dir / CONFIG["W_CONTRACTS_CSV"]
    meta_path = input_dir / CONFIG["WINDOW_META_CSV"]
    u0_path = input_dir / CONFIG["U0_ALIGNED_NPY"]

    for path in [w_path, users_path, contracts_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing WRec artifact: {path}")

    W = sparse.load_npz(w_path).tocsr()
    users = pd.read_csv(users_path)["user"].astype(str).tolist()
    contracts = pd.read_csv(contracts_path)["contract_id"].astype(str).tolist()
    meta = pd.read_csv(meta_path).iloc[0].to_dict() if meta_path.exists() else None
    u0 = np.load(u0_path).astype(float) if u0_path.exists() else None

    if W.shape != (len(users), len(contracts)):
        raise ValueError(f"WRec shape mismatch: W={W.shape}, users={len(users)}, contracts={len(contracts)}")
    if u0 is not None and len(u0) != len(users):
        raise ValueError(f"u0 shape mismatch: u0={len(u0)}, users={len(users)}")
    return W, users, contracts, meta, u0


def build_bipartite_transition(W: sparse.csr_matrix) -> sparse.csr_matrix:
    d_u = np.array(W.sum(axis=1)).flatten()
    d_c = np.array(W.sum(axis=0)).flatten()
    d_u[d_u <= 0] = 1.0
    d_c[d_c <= 0] = 1.0

    p_uc = sparse.diags(1.0 / d_u) @ W
    p_cu = sparse.diags(1.0 / d_c) @ W.T

    zeros_uu = sparse.csr_matrix((W.shape[0], W.shape[0]))
    zeros_cc = sparse.csr_matrix((W.shape[1], W.shape[1]))
    return sparse.bmat([[zeros_uu, p_uc], [p_cu, zeros_cc]], format="csr")


def build_restart_vector(cfg: dict, n_u: int, n_c: int, u0: np.ndarray | None) -> np.ndarray:
    mode = str(cfg["RESTART_MODE"]).strip().lower()

    if mode == "uniform":
        r = np.ones(n_u + n_c, dtype=float)
        return r / (r.sum() + EPS)

    if mode == "user_prior":
        if u0 is None:
            raise ValueError("RESTART_MODE='user_prior' requires U0_ALIGNED_NPY to exist.")
        user_mass = float(cfg["RESTART_USER_MASS"])
        contract_mass = float(cfg["RESTART_CONTRACT_MASS"])
        if user_mass < 0 or contract_mass < 0 or (user_mass + contract_mass) <= 0:
            raise ValueError("Restart masses must be non-negative and not both zero.")

        u0v = np.asarray(u0, dtype=float).flatten()
        u0v[u0v <= 0] = EPS
        u0v = u0v / (u0v.sum() + EPS)

        c0v = np.ones(n_c, dtype=float) / max(n_c, 1)
        r = np.concatenate([user_mass * u0v, contract_mass * c0v])
        return r / (r.sum() + EPS)

    raise ValueError(f"Unknown RESTART_MODE={cfg['RESTART_MODE']}. Use 'uniform' or 'user_prior'.")


def pagerank_bipartite_block(
    W: sparse.csr_matrix,
    gamma: float,
    restart: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_u, n_c = W.shape
    P = build_bipartite_transition(W)
    pi = restart.reshape(-1, 1).copy()
    rv = restart.reshape(-1, 1)

    for _ in range(max_iter):
        pi_new = gamma * (P.T @ pi) + (1.0 - gamma) * rv
        pi_new = pi_new / (pi_new.sum() + EPS)
        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new

    pi = pi.flatten()
    return pi[:n_u], pi[n_u:]


def save_scores(outdir: Path, users: list[str], contracts: list[str], u: np.ndarray, c: np.ndarray, gamma: float):
    outdir.mkdir(parents=True, exist_ok=True)

    df_u = pd.DataFrame({"user": users, "u_pagerank": u}).sort_values("u_pagerank", ascending=False).reset_index(drop=True)
    df_u["rank_user"] = np.arange(1, len(df_u) + 1)

    df_c = pd.DataFrame({"contract_id": contracts, "c_pagerank": c}).sort_values("c_pagerank", ascending=False).reset_index(drop=True)
    df_c["rank_contract"] = np.arange(1, len(df_c) + 1)

    base_u = outdir / CONFIG["OUT_USER_SCORES_CSV"]
    base_c = outdir / CONFIG["OUT_CONTRACT_SCORES_CSV"]
    suf_u = outdir / f"2_pagerank_user_scores_gamma{gamma:.2f}.csv"
    suf_c = outdir / f"2_pagerank_contract_scores_gamma{gamma:.2f}.csv"

    df_u.to_csv(suf_u, index=False)
    df_c.to_csv(suf_c, index=False)
    if abs(gamma - float(CONFIG["PRIMARY_GAMMA"])) < 1e-12:
        df_u.to_csv(base_u, index=False)
        df_c.to_csv(base_c, index=False)

    return df_u, df_c


def summarize_contracts(df_c: pd.DataFrame):
    out = df_c.copy()
    out["category"] = out["contract_id"].astype(str).str.split("_").str[0]
    sybil = out[out["category"] == "Sybil"][["contract_id", "c_pagerank", "rank_contract"]].copy()
    cat_mean = out.groupby("category")["rank_contract"].mean().sort_values()
    return sybil, cat_mean


def run(config: dict | None = None):
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)

    input_dir = Path(cfg["INPUT_DIR"])
    outdir = Path(cfg["OUTDIR"])

    W, users, contracts, meta, u0 = load_wrec(input_dir)
    restart = build_restart_vector(cfg, len(users), len(contracts), u0=u0)

    print("=" * 90)
    print("[2_pagerank] Bipartite PageRank on WRec block transition")
    print("=" * 90)
    print(f"input dir : {input_dir}")
    print(f"output dir: {outdir}")
    print(f"restart   : {cfg['RESTART_MODE']}")
    if meta is not None:
        meta_show = " | ".join(
            [f"{k}: {meta.get(k)}" for k in ["window_blocks", "start_block", "end_block", "nnz", "n_rows", "n_cols", "mode"] if k in meta]
        )
        if meta_show:
            print(meta_show)

    last_user_df = None
    last_contract_df = None
    for gamma in cfg["GAMMA_LIST"]:
        u, c = pagerank_bipartite_block(
            W=W,
            gamma=float(gamma),
            restart=restart,
            max_iter=int(cfg["MAX_ITER"]),
            tol=float(cfg["TOL"]),
        )
        df_u, df_c = save_scores(outdir, users, contracts, u, c, gamma=float(gamma))
        sybil_df, cat_mean = summarize_contracts(df_c)

        print("\n" + "=" * 90)
        print(f"[2_pagerank] gamma={gamma:.2f}")
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
        "restart_mode": cfg["RESTART_MODE"],
        "last_user_df": last_user_df,
        "last_contract_df": last_contract_df,
    }


if __name__ == "__main__":
    run()
