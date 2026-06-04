from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


EPS = 1e-12
ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    "INPUT_DIR": ROOT / "output" / "E" / "0_our_paper",
    "RAW_DATA_CSV": ROOT / "output" / "A" / "11_RQ2data" / "RQdataG1.csv",
    "CONTRACTS_META_CSV": ROOT / "output" / "A" / "11_RQ2data" / "RQdataG1_contracts.csv",
    "OUTDIR": ROOT / "output" / "E" / "3_eigentrust",
    "STAGEC_META_JSON": "3_stageC_meta.json",
    "WINDOW_META_CSV": "3_stageC_window_meta.csv",
    "MODE": "decay_full",
    "WINDOW_BLOCKS": 216_000,
    "TAU_DECAY": 216_000,
    "TRADE_ONLY": False,
    "INCLUDE_MINT_GAS": True,
    "W_GAS": 1.0,
    "ALPHA_LIST": [0.7, 0.6, 0.5, 0.4, 0.1],
    "PRIMARY_ALPHA": 0.7,
    "PRETRUST_MODE": "uniform",
    "MAX_ITER": 80,
    "TOL": 1e-10,
    "SHOW_TOP_USERS": 20,
    "SHOW_TOP_CONTRACTS": 20,
    "OUT_USER_SCORES_CSV": "3_eigentrust_user_scores.csv",
    "OUT_CONTRACT_SCORES_CSV": "3_eigentrust_contract_scores.csv",
}


def load_window_cfg(cfg: dict) -> dict:
    out = dict(cfg)
    meta_json_path = Path(cfg["INPUT_DIR"]) / cfg["STAGEC_META_JSON"]
    meta_csv_path = Path(cfg["INPUT_DIR"]) / cfg["WINDOW_META_CSV"]
    if meta_json_path.exists():
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = dict(json.load(f))
        out["MODE"] = "decay_full"
        if "tau_decay" in meta:
            out["TAU_DECAY"] = float(meta["tau_decay"])
        return out
    if meta_csv_path.exists():
        meta = pd.read_csv(meta_csv_path).iloc[0].to_dict()
        if "mode" in meta and pd.notna(meta["mode"]):
            out["MODE"] = str(meta["mode"])
        if "window_blocks" in meta and pd.notna(meta["window_blocks"]):
            out["WINDOW_BLOCKS"] = int(meta["window_blocks"])
        if "tau_decay" in meta and pd.notna(meta["tau_decay"]):
            out["TAU_DECAY"] = float(meta["tau_decay"])
    return out


def load_filtered_events(cfg: dict) -> tuple[pd.DataFrame, dict]:
    data_csv = Path(cfg["RAW_DATA_CSV"])
    if not data_csv.exists():
        raise FileNotFoundError(f"Missing raw data file: {data_csv}")

    df = pd.read_csv(data_csv, low_memory=False)
    if "block_number" not in df.columns:
        raise ValueError("RAW_DATA_CSV must contain block_number.")

    df["block_number"] = pd.to_numeric(df["block_number"], errors="coerce").fillna(0).astype(int)
    df["tx_type"] = df["tx_type"].astype(str)
    df["buyer"] = df["buyer"].astype(str)
    df["contract_id"] = df["contract_id"].astype(str)
    df["gas"] = pd.to_numeric(df["gas"], errors="coerce").fillna(0.0)

    if bool(cfg["TRADE_ONLY"]):
        allowed_tx_types = {"trade"}
    else:
        allowed_tx_types = {"trade"}
        if bool(cfg["INCLUDE_MINT_GAS"]):
            allowed_tx_types.add("mint")
    df = df[df["tx_type"].isin(allowed_tx_types)].copy()

    if len(df) == 0:
        raise ValueError("No events remain after applying TRADE_ONLY / INCLUDE_MINT_GAS filtering.")

    end_block = int(df["block_number"].max())
    mode = str(cfg["MODE"]).strip().lower()

    if mode == "cutoff":
        start_block = max(0, end_block - int(cfg["WINDOW_BLOCKS"]))
        wdf = df[(df["block_number"] >= start_block) & (df["block_number"] <= end_block)].copy()
        wdf["time_w"] = 1.0
    elif mode == "decay_full":
        tau = float(cfg["TAU_DECAY"])
        if tau <= 0:
            raise ValueError("TAU_DECAY must be > 0 for decay_full.")
        wdf = df.copy()
        wdf["time_w"] = wdf["block_number"].apply(lambda t: math.exp(-(end_block - int(t)) / tau))
        start_block = int(wdf["block_number"].min())
    else:
        raise ValueError("MODE must be cutoff or decay_full.")

    if len(wdf) == 0:
        raise ValueError("No events remain after window / decay filtering.")

    meta = {
        "mode": mode,
        "start_block": int(start_block),
        "end_block": int(end_block),
        "window_blocks": int(cfg["WINDOW_BLOCKS"]),
        "tau_decay": float(cfg["TAU_DECAY"]),
        "n_events_used": int(len(wdf)),
    }
    return wdf, meta


def build_user_creator_matrix(wdf: pd.DataFrame, contracts_meta_csv: Path, w_gas: float) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    contracts_meta = pd.read_csv(contracts_meta_csv)
    contracts_meta["contract_id"] = contracts_meta["contract_id"].astype(str)

    creator_by_contract = {cid: f"creator_{cid}" for cid in contracts_meta["contract_id"].tolist()}
    wdf = wdf[wdf["contract_id"].isin(set(creator_by_contract.keys()))].copy()
    if len(wdf) == 0:
        raise ValueError("No events remain after aligning with contracts meta.")

    wdf["creator"] = wdf["contract_id"].map(creator_by_contract)
    wdf["weight"] = pd.to_numeric(wdf["gas"], errors="coerce").fillna(0.0) * pd.to_numeric(wdf["time_w"], errors="coerce").fillna(1.0) * float(w_gas)
    agg = wdf.groupby(["buyer", "creator"], as_index=False)["weight"].sum()
    agg = agg[agg["weight"] > 0].copy()
    if len(agg) == 0:
        raise ValueError("All EigenTrust edge weights are zero after aggregation.")

    users = sorted(set(agg["buyer"].astype(str)).union(set(agg["creator"].astype(str))))
    creators = sorted(set(agg["creator"].astype(str)))
    u_map = {u: i for i, u in enumerate(users)}

    rows = agg["buyer"].astype(str).map(u_map).to_numpy()
    cols = agg["creator"].astype(str).map(u_map).to_numpy()
    data = agg["weight"].astype(float).to_numpy()

    C = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(users)))
    return C, users, creators


def normalize_local_trust(C: sparse.csr_matrix) -> sparse.csr_matrix:
    row_sum = np.array(C.sum(axis=1)).flatten()
    row_sum[row_sum <= 0] = 1.0
    return sparse.diags(1.0 / row_sum) @ C


def build_pretrust(users: list[str], creators: list[str], mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    p = np.zeros(len(users), dtype=float)

    if mode == "uniform":
        p[:] = 1.0
    elif mode == "creator_uniform":
        creator_set = set(creators)
        for i, user in enumerate(users):
            if user in creator_set:
                p[i] = 1.0
    else:
        raise ValueError(f"Unknown PRETRUST_MODE={mode}. Use 'uniform' or 'creator_uniform'.")

    if p.sum() <= 0:
        p[:] = 1.0
    return p / (p.sum() + EPS)


def eigentrust(C_norm: sparse.csr_matrix, alpha: float, pretrust: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
    t = pretrust.reshape(-1, 1).copy()
    pv = pretrust.reshape(-1, 1)

    for _ in range(max_iter):
        t_new = (1.0 - alpha) * (C_norm.T @ t) + alpha * pv
        t_new = t_new / (t_new.sum() + EPS)
        if np.max(np.abs(t_new - t)) < tol:
            t = t_new
            break
        t = t_new

    return t.flatten()


def save_scores(
    outdir: Path,
    user_scores: np.ndarray,
    users: list[str],
    contracts_meta_csv: Path,
    alpha: float,
    cfg: dict,
):
    outdir.mkdir(parents=True, exist_ok=True)

    df_u = pd.DataFrame({"user": users, "u_eigentrust": user_scores}).sort_values("u_eigentrust", ascending=False).reset_index(drop=True)
    df_u["rank_user"] = np.arange(1, len(df_u) + 1)

    user_score_map = dict(zip(df_u["user"], df_u["u_eigentrust"]))
    contracts_meta = pd.read_csv(contracts_meta_csv)
    contracts_meta["contract_id"] = contracts_meta["contract_id"].astype(str)
    contracts_meta["creator"] = contracts_meta["contract_id"].apply(lambda cid: f"creator_{cid}")
    contracts_meta["c_eigentrust"] = contracts_meta["creator"].map(user_score_map).fillna(0.0)

    df_c = contracts_meta[["contract_id", "c_eigentrust"]].sort_values("c_eigentrust", ascending=False).reset_index(drop=True)
    df_c["rank_contract"] = np.arange(1, len(df_c) + 1)

    base_u = outdir / cfg["OUT_USER_SCORES_CSV"]
    base_c = outdir / cfg["OUT_CONTRACT_SCORES_CSV"]
    suf_u = outdir / f"3_eigentrust_user_scores_alpha{alpha:.2f}.csv"
    suf_c = outdir / f"3_eigentrust_contract_scores_alpha{alpha:.2f}.csv"

    df_u.to_csv(suf_u, index=False)
    df_c.to_csv(suf_c, index=False)
    if abs(alpha - float(cfg["PRIMARY_ALPHA"])) < 1e-12:
        df_u.to_csv(base_u, index=False)
        df_c.to_csv(base_c, index=False)

    return df_u, df_c


def summarize_contracts(df_c: pd.DataFrame):
    out = df_c.copy()
    out["category"] = out["contract_id"].astype(str).str.split("_").str[0]
    sybil = out[out["category"] == "Sybil"][["contract_id", "c_eigentrust", "rank_contract"]].copy()
    cat_mean = out.groupby("category")["rank_contract"].mean().sort_values()
    return sybil, cat_mean


def run(config: dict | None = None):
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)
    cfg = load_window_cfg(cfg)

    outdir = Path(cfg["OUTDIR"])
    contracts_meta_csv = Path(cfg["CONTRACTS_META_CSV"])
    if not contracts_meta_csv.exists():
        raise FileNotFoundError(f"Missing contracts meta file: {contracts_meta_csv}")

    wdf, meta = load_filtered_events(cfg)
    C, users, creators = build_user_creator_matrix(wdf, contracts_meta_csv=contracts_meta_csv, w_gas=float(cfg["W_GAS"]))
    C_norm = normalize_local_trust(C)
    pretrust = build_pretrust(users, creators, mode=str(cfg["PRETRUST_MODE"]))

    meta.update(
        {
            "n_rows": int(C.shape[0]),
            "n_cols": int(C.shape[1]),
            "nnz": int(C.nnz),
            "n_creators": int(len(creators)),
        }
    )

    print("=" * 90)
    print("[3_eigentrust3] User-to-creator EigenTrust on Stage-C-consistent decayed weights")
    print("=" * 90)
    print(f"raw data   : {cfg['RAW_DATA_CSV']}")
    print(f"output dir : {outdir}")
    print(f"pretrust   : {cfg['PRETRUST_MODE']}")
    meta_show = " | ".join(
        [f"{k}: {meta.get(k)}" for k in ["window_blocks", "start_block", "end_block", "nnz", "n_rows", "n_creators", "mode", "tau_decay"]]
    )
    print(meta_show)

    last_user_df = None
    last_contract_df = None
    for alpha in cfg["ALPHA_LIST"]:
        trust = eigentrust(
            C_norm=C_norm,
            alpha=float(alpha),
            pretrust=pretrust,
            max_iter=int(cfg["MAX_ITER"]),
            tol=float(cfg["TOL"]),
        )
        df_u, df_c = save_scores(outdir, trust, users, contracts_meta_csv=contracts_meta_csv, alpha=float(alpha), cfg=cfg)
        sybil_df, cat_mean = summarize_contracts(df_c)

        print("\n" + "=" * 90)
        print(f"[3_eigentrust3] alpha={alpha:.2f}")
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
        "creators": creators,
        "meta": meta,
        "last_user_df": last_user_df,
        "last_contract_df": last_contract_df,
    }


if __name__ == "__main__":
    run()
