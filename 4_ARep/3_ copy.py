import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

EPS = 1e-12

CONFIG = {
    "STAGEB_USER_SCORES_CSV": "../output/4/2_stageB_user_scores.csv",
    "DATA_CSV": "../data/4/data3.csv",
    "OUTDIR": "../output/4/",
    "MODE": "decay_full",
    "WINDOW_BLOCKS": 216_000,
    "TAU_DECAY": 216_000,
    "TAU_S": 200_000,
    "TAU_H": 50_000,
    "MU": 1.0,
    "W_GAS": 1.0,
    "W_VAL": 0.0,
    "INCLUDE_MINT_GAS": True,
    "INCLUDE_MINT_EDGE": False,
    "STAGEB_USER_COL": "user",
    "STAGEB_SCORE_COL": "score_B",
    "TRADE_ONLY": True,
}


def load_stageB_and_make_p0(stageB_csv: str, score_col: str, user_col: str):
    if not os.path.exists(stageB_csv):
        raise FileNotFoundError(f"Stage B file not found: {stageB_csv}")

    dfB = pd.read_csv(stageB_csv)
    if user_col not in dfB.columns or score_col not in dfB.columns:
        raise ValueError(f"Stage B file must contain {user_col} and {score_col}; got {list(dfB.columns)}")

    dfB = dfB[[user_col, score_col]].copy()
    dfB[score_col] = pd.to_numeric(dfB[score_col], errors="coerce").fillna(0.0)

    smax = float(dfB[score_col].max())
    if smax <= 0:
        dfB["score01"] = EPS
    else:
        dfB["score01"] = dfB[score_col] / (smax + EPS)
        dfB.loc[dfB["score01"] <= 0, "score01"] = EPS

    denom = float(dfB["score01"].sum())
    if denom <= 0:
        denom = EPS * max(len(dfB), 1)

    dfB["p0"] = dfB["score01"] / denom
    p0_map = dict(zip(dfB[user_col].astype(str), dfB["p0"]))
    return dfB, p0_map


def _supported_event_mask(df: pd.DataFrame) -> pd.Series:
    if bool(CONFIG.get("TRADE_ONLY", True)):
        if bool(CONFIG.get("INCLUDE_MINT_EDGE", False)):
            return df["tx_type"].astype(str).isin(["trade", "mint"])
        return df["tx_type"].astype(str).eq("trade")
    return df["tx_type"].astype(str).isin(["trade", "mint"])

def build_weighted_event_matrix(
    df: pd.DataFrame,
    tau_decay: float,
    tau_s: float,
    tau_h: float,
    include_mint_gas: bool,
) -> tuple[sparse.csr_matrix, list[str], list[str], dict]:
    required = {"block_number", "tx_type", "buyer", "contract_id", "gas"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"data missing required columns for Stage C weighting: {missing}")

    work = df.copy()
    work["block_number"] = pd.to_numeric(work["block_number"], errors="coerce").fillna(0).astype(int)
    work["gas"] = pd.to_numeric(work["gas"], errors="coerce").fillna(0.0)
    work["buyer"] = work["buyer"].astype(str)
    work["contract_id"] = work["contract_id"].astype(str)
    work["tx_type"] = work["tx_type"].astype(str)
    if "tx_index_in_block" not in work.columns:
        work["tx_index_in_block"] = 0

    work = work.loc[_supported_event_mask(work)].copy()
    if len(work) == 0:
        raise ValueError("no usable Stage C events remain after filtering")

    end_block = int(work["block_number"].max())

    users = pd.unique(work["buyer"])
    users = users[(users != "") & (~pd.isna(users))]
    contracts = pd.unique(work["contract_id"])
    u_map = {u: i for i, u in enumerate(users)}
    c_map = {c: i for i, c in enumerate(contracts)}
    weight = defaultdict(float)

    event_weight_sum = 0.0
    event_rows = 0
    mint_rows = 0
    trade_rows = 0

    for row in work.sort_values(["block_number", "tx_index_in_block"]).itertuples(index=False):
        buyer = str(row.buyer)
        cid = str(row.contract_id)
        t = int(row.block_number)
        tx_type = str(row.tx_type)
        gas = float(row.gas) if bool(include_mint_gas or tx_type != "mint") else 0.0
        decay = math.exp(-(end_block - t) / max(float(tau_decay), EPS))
        w_event = gas * decay
        if w_event > 0 and buyer in u_map:
            weight[(u_map[buyer], c_map[cid])] += float(w_event)
            event_weight_sum += float(w_event)
            event_rows += 1
        if tx_type == "trade":
            trade_rows += 1
        elif tx_type == "mint":
            mint_rows += 1

    rows = [k[0] for k in weight.keys()]
    cols = [k[1] for k in weight.keys()]
    data = [v for v in weight.values()]
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(contracts)))
    meta = {
        "n_events_used": int(len(work)),
        "n_weighted_events": int(event_rows),
        "mint_rows_used": int(mint_rows),
        "trade_rows_used": int(trade_rows),
        "event_weight_sum": float(event_weight_sum),
        "weight_rule": "gas_times_time_decay",
    }
    return matrix, users.tolist(), contracts.tolist(), meta


def build_Wtrade_stageA_like(data_csv: str, mu: float, w_gas: float, w_val: float, include_mint_gas: bool):
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"data file not found: {data_csv}")

    df = pd.read_csv(data_csv)
    if "block_number" not in df.columns:
        raise ValueError("data must contain block_number")

    df["block_number"] = pd.to_numeric(df["block_number"], errors="coerce").fillna(0).astype(int)
    end_block = int(df["block_number"].max()) if len(df) else 0
    mode = str(CONFIG["MODE"]).strip().lower()

    if mode == "cutoff":
        start_block = max(0, end_block - int(CONFIG["WINDOW_BLOCKS"]))
        wdf = df[(df["block_number"] >= start_block) & (df["block_number"] <= end_block)].copy()
    elif mode == "decay_full":
        wdf = df.copy()
        start_block = int(wdf["block_number"].min()) if len(wdf) else 0
    else:
        raise ValueError("MODE must be cutoff or decay_full")

    if len(wdf) == 0:
        raise ValueError("window/decay processing left no usable rows")

    W_trade, users_W, contracts_W, meta_extra = build_weighted_event_matrix(
        wdf,
        tau_decay=float(CONFIG["TAU_DECAY"]),
        tau_s=float(CONFIG["TAU_S"]),
        tau_h=float(CONFIG["TAU_H"]),
        include_mint_gas=include_mint_gas,
    )
    meta = {
        "mode": mode,
        "start_block": int(start_block),
        "end_block": int(end_block),
        "window_blocks": int(CONFIG["WINDOW_BLOCKS"]),
        "tau_decay": float(CONFIG["TAU_DECAY"]),
        "tau_s": float(CONFIG["TAU_S"]),
        "tau_h": float(CONFIG["TAU_H"]),
        "n_rows": int(W_trade.shape[0]),
        "n_cols": int(W_trade.shape[1]),
        "nnz": int(W_trade.nnz),
    }
    meta.update(meta_extra)
    return W_trade, users_W, contracts_W, meta


def align_user_prior_to_W(users_W, p0_map: dict):
    u0 = np.array([float(p0_map.get(str(u), 0.0)) for u in users_W], dtype=float)
    u0[u0 <= 0] = EPS
    u0 = u0 / (u0.sum() + EPS)
    return u0


def save_stageC_inputs(outdir: str, dfB_norm: pd.DataFrame, W_trade, users_W, contracts_W, u0_vec, meta: dict):
    os.makedirs(outdir, exist_ok=True)
    dfB_norm.to_csv(os.path.join(outdir, "3_stageB_user_scores_norm.csv"), index=False)
    np.save(os.path.join(outdir, "3_stageC_u0_aligned.npy"), u0_vec)
    sparse.save_npz(os.path.join(outdir, "3_stageC_W_trade.npz"), W_trade)
    pd.Series(users_W, name="user").to_csv(os.path.join(outdir, "3_stageC_W_users.csv"), index=False)
    pd.Series(contracts_W, name="contract_id").to_csv(os.path.join(outdir, "3_stageC_W_contracts.csv"), index=False)
    pd.DataFrame([meta]).to_csv(os.path.join(outdir, "3_stageC_window_meta.csv"), index=False)


if __name__ == "__main__":
    stageB_csv = CONFIG["STAGEB_USER_SCORES_CSV"]
    data_csv = CONFIG["DATA_CSV"]
    outdir = CONFIG["OUTDIR"]
    mu = float(CONFIG["MU"])
    w_gas = float(CONFIG["W_GAS"])
    w_val = float(CONFIG["W_VAL"])
    include_mint_gas = bool(CONFIG["INCLUDE_MINT_GAS"])
    user_col = CONFIG["STAGEB_USER_COL"]
    score_col = CONFIG["STAGEB_SCORE_COL"]

    dfB_norm, p0_map = load_stageB_and_make_p0(stageB_csv, score_col=score_col, user_col=user_col)
    W_trade, users_W, contracts_W, meta = build_Wtrade_stageA_like(
        data_csv=data_csv,
        mu=mu,
        w_gas=w_gas,
        w_val=w_val,
        include_mint_gas=include_mint_gas,
    )
    u0 = align_user_prior_to_W(users_W, p0_map)
    save_stageC_inputs(outdir, dfB_norm, W_trade, users_W, contracts_W, u0, meta)

    print("[OK] Stage C inputs saved")
    print(f"  outdir: {outdir}")
    print(f"  mode: {meta['mode']}")
    print(f"  range: [{meta['start_block']}, {meta['end_block']}]")
    print(f"  W_trade: shape=({meta['n_rows']},{meta['n_cols']}), nnz={meta['nnz']}, weighted_events={meta['n_weighted_events']}")
