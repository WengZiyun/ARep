#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage C: Time Adaptation (paper-aligned)

Implements:
  1) Build time-decayed endorsement matrix W^Rec at tau_end.
  2) Build row-stochastic transitions:
       P_{U->C} = D_U^{-1} W^Rec
       P_{C->U} = D_C^{-1} (W^Rec)^T
  3) Personalized bipartite propagation with priors (u0, c0):
       c_{k+1} = beta * P_{U->C}^T u_k + (1-beta) c0
       u_{k+1} = alpha * P_{C->U}^T c_{k+1} + (1-alpha) u0
  4) Stop by L1 convergence or max iterations.

Outputs (prefix 2_):
  - output/C/2_stageC_contract_scores.csv
  - output/C/2_stageC_user_scores.csv
  - output/C/2_stageC_W_rec.npz
  - output/C/2_stageC_W_users.csv
  - output/C/2_stageC_W_contracts.csv
  - output/C/2_stageC_meta.csv
"""

from pathlib import Path
from collections import defaultdict
import math

import numpy as np
import pandas as pd
from scipy import sparse


# ============================================================
# 0) Defaults (edit here; no argparse)
# ============================================================
INPUT_DIR = Path("output/B/2_tranding")

STAGEA_CONTRACT_CSV = Path("output/C/0_stageA_contract_scores.csv")
STAGEB_U0_FULL_CSV = Path("output/C/1_stageB_u0_full.csv")
STAGEB_USER_PRIOR_CSV = Path("output/C/1_stageB_user_prior.csv")

OUTPUT_CONTRACT_CSV = Path("output/C/2_stageC_contract_scores.csv")
OUTPUT_USER_CSV = Path("output/C/2_stageC_user_scores.csv")
OUTPUT_WREC_NPZ = Path("output/C/2_stageC_W_rec.npz")
OUTPUT_W_USERS_CSV = Path("output/C/2_stageC_W_users.csv")
OUTPUT_W_CONTRACTS_CSV = Path("output/C/2_stageC_W_contracts.csv")
OUTPUT_META_CSV = Path("output/C/2_stageC_meta.csv")

# Stage C hyperparameters
ALPHA = 1
BETA = 0.4
TAU_D = 300_000.0
TAU_S = 200_000.0
TAU_H = 50_000.0
N_REF = None
EPS = 1e-12
MAX_ITER = 100
TOL = 1e-10

# Fair window policy
# - near_latest_min_contract_last_block: keep contracts whose last_block is close to global latest,
#   then choose the minimum last_block inside this near-latest cohort.
# - min_contract_last_block: use min over all contracts (strict, may be too conservative)
# - global_last_block: use global max observed block
TAU_END_MODE = "near_latest_min_contract_last_block"
# Near-latest threshold in blocks (used by near_latest_min_contract_last_block).
# Contracts with last_block >= (global_last_block - TAU_END_NEAR_GAP_BLOCKS) are kept.
TAU_END_NEAR_GAP_BLOCKS = 800_000

# Event policy
# acquisition in this dataset: mint/trade (sale normalized to trade)
ACQUISITION_TYPES = {"mint", "trade"}
# can include non-acquisition interactions if present in source data (e.g., transfer/deploy)
INCLUDE_NON_ACQUISITION = True
NON_ACQUISITION_TYPES = {"transfer", "deploy"}

# For non-acquisition events: set v_e=0, only cost/holding-related factors
# cost := gas for non-acquisition


# ============================================================
# 1) Load events
# ============================================================
REQUIRED_COLS = [
    "block_number", "timestamp", "tx_type",
    "contract_address", "contract_id", "token_id",
    "seller", "buyer",
    "price_usd", "gas_usd",
]


def load_events(input_dir: Path) -> pd.DataFrame:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files under: {input_dir.resolve()}")

    frames = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {fp.name}: {missing}")

        use_cols = REQUIRED_COLS + (["tx_index_in_block"] if "tx_index_in_block" in df.columns else [])
        sub = df[use_cols].copy()
        if "tx_index_in_block" not in sub.columns:
            sub["tx_index_in_block"] = 0

        sub = sub.rename(columns={"price_usd": "price", "gas_usd": "gas"})
        sub["tx_type"] = sub["tx_type"].astype(str).str.lower().str.strip().replace({"sale": "trade"})

        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)

    out["block_number"] = pd.to_numeric(out["block_number"], errors="coerce").fillna(0).astype(np.int64)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["tx_index_in_block"] = pd.to_numeric(out["tx_index_in_block"], errors="coerce").fillna(0).astype(np.int64)

    out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0.0).astype(float)
    out["gas"] = pd.to_numeric(out["gas"], errors="coerce").fillna(0.0).astype(float)

    out["token_id"] = out["token_id"].astype(str)
    out["contract_id"] = out["contract_id"].astype(str)
    out["buyer"] = out["buyer"].fillna("").astype(str).str.lower()
    out["seller"] = out["seller"].fillna("").astype(str).str.lower()

    if INCLUDE_NON_ACQUISITION:
        keep_types = ACQUISITION_TYPES.union(NON_ACQUISITION_TYPES)
    else:
        keep_types = set(ACQUISITION_TYPES)

    out = out[out["tx_type"].isin(keep_types)].copy()
    out = out[out["buyer"] != ""].copy()
    out = out.sort_values(["block_number", "tx_index_in_block"], kind="mergesort").reset_index(drop=True)
    return out


def choose_tau_end(df: pd.DataFrame, mode: str, near_gap_blocks: int) -> tuple[int, dict]:
    if df.empty:
        return 0, {"global_last_block": 0, "cohort_contracts": 0, "all_contracts": 0}

    global_last = int(df["block_number"].max())
    per_contract_last = df.groupby("contract_id", as_index=False)["block_number"].max()
    all_contracts = int(len(per_contract_last))

    if mode == "global_last_block":
        return global_last, {
            "global_last_block": global_last,
            "cohort_contracts": all_contracts,
            "all_contracts": all_contracts,
            "near_gap_blocks": int(near_gap_blocks),
        }
    if mode == "min_contract_last_block":
        tau_end = int(per_contract_last["block_number"].min())
        return tau_end, {
            "global_last_block": global_last,
            "cohort_contracts": all_contracts,
            "all_contracts": all_contracts,
            "near_gap_blocks": int(near_gap_blocks),
        }
    if mode == "near_latest_min_contract_last_block":
        gap = max(0, int(near_gap_blocks))
        lower = global_last - gap
        cohort = per_contract_last[per_contract_last["block_number"] >= lower].copy()
        if cohort.empty:
            cohort = per_contract_last
        tau_end = int(cohort["block_number"].min())
        return tau_end, {
            "global_last_block": global_last,
            "cohort_contracts": int(len(cohort)),
            "all_contracts": all_contracts,
            "near_gap_blocks": gap,
            "near_lower_bound": int(lower),
        }
    raise ValueError(
        "TAU_END_MODE must be one of "
        "'near_latest_min_contract_last_block', 'min_contract_last_block', 'global_last_block'."
    )


def compute_contract_stats(df_window: pd.DataFrame) -> pd.DataFrame:
    if df_window.empty:
        return pd.DataFrame(
            columns=[
                "contract_id",
                "window_first_block",
                "window_last_block",
                "window_mint_cnt",
                "window_trade_cnt",
                "window_gas_total_usd",
                "window_trade_value_total_usd",
                "window_unique_trade_users",
                "window_unique_buyers",
            ]
        )

    base = df_window.groupby("contract_id", as_index=False).agg(
        window_first_block=("block_number", "min"),
        window_last_block=("block_number", "max"),
        window_mint_cnt=("tx_type", lambda s: int((s == "mint").sum())),
        window_trade_cnt=("tx_type", lambda s: int((s == "trade").sum())),
        window_gas_total_usd=("gas", "sum"),
    )

    trade_df = df_window[df_window["tx_type"] == "trade"].copy()
    trade_sum = trade_df.groupby("contract_id", as_index=False)["price"].sum().rename(
        columns={"price": "window_trade_value_total_usd"}
    )

    buyers = trade_df[["contract_id", "buyer"]].rename(columns={"buyer": "user"})
    sellers = trade_df[["contract_id", "seller"]].rename(columns={"seller": "user"})
    trade_users = pd.concat([buyers, sellers], ignore_index=True)
    trade_users = trade_users[trade_users["user"].astype(str) != ""]
    uniq_trade_users = trade_users.groupby("contract_id", as_index=False)["user"].nunique().rename(
        columns={"user": "window_unique_trade_users"}
    )

    uniq_buyers = df_window.groupby("contract_id", as_index=False)["buyer"].nunique().rename(
        columns={"buyer": "window_unique_buyers"}
    )

    out = base.merge(trade_sum, on="contract_id", how="left")
    out = out.merge(uniq_trade_users, on="contract_id", how="left")
    out = out.merge(uniq_buyers, on="contract_id", how="left")

    out["window_trade_value_total_usd"] = out["window_trade_value_total_usd"].fillna(0.0)
    out["window_unique_trade_users"] = out["window_unique_trade_users"].fillna(0).astype(int)
    out["window_unique_buyers"] = out["window_unique_buyers"].fillna(0).astype(int)
    return out


def compute_contract_address_map(df_window: pd.DataFrame) -> pd.DataFrame:
    if df_window.empty:
        return pd.DataFrame(columns=["contract_id", "contract_address"])

    addr_df = df_window[["contract_id", "contract_address"]].copy()
    addr_df["contract_address"] = addr_df["contract_address"].fillna("").astype(str).str.strip()
    addr_df = addr_df[addr_df["contract_address"] != ""]
    if addr_df.empty:
        return pd.DataFrame(columns=["contract_id", "contract_address"])

    # If one contract_id appears with multiple addresses, keep the most frequent one.
    addr_df = (
        addr_df.groupby(["contract_id", "contract_address"], as_index=False)
        .size()
        .sort_values(["contract_id", "size", "contract_address"], ascending=[True, False, True], kind="mergesort")
    )
    out = addr_df.drop_duplicates(subset=["contract_id"], keep="first")[["contract_id", "contract_address"]]
    return out


# ============================================================
# 2) Build event factors for Stage C
# ============================================================
def g_early_from_rank(r_rank: int, N_current: int, n_ref: int, eps: float) -> float:
    if N_current <= 1:
        return 1.0

    D = (N_current - 1.0) + float(n_ref)
    q = (r_rank - 1.0) / (D + eps)
    q_max = (N_current - 1.0) / (D + eps)

    denom = math.log1p(q_max)
    if denom <= eps:
        return 1.0

    val = 1.0 - (math.log1p(max(0.0, q)) / denom)
    return max(0.0, min(1.0, val))


def build_stagec_event_table(
    df: pd.DataFrame,
    tau_s: float,
    tau_h: float,
    n_ref: int | None,
    eps: float,
    tau_end: int,
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 1

    ev = df.copy()
    ev["event_id"] = np.arange(len(ev), dtype=np.int64)
    ev["is_acq"] = ev["tx_type"].isin(ACQUISITION_TYPES).astype(np.int8)

    # v_e policy
    ev["v_e"] = np.where(ev["is_acq"] == 1, ev["price"], 0.0)
    ev["cost"] = (ev["v_e"] + ev["gas"]).clip(lower=0.0)

    # relinquish/next event block on same token path, else tau_end (window end)
    ev["next_block"] = ev.groupby(["contract_id", "token_id"], sort=False)["block_number"].shift(-1)
    ev["tau_rel"] = ev["next_block"].fillna(tau_end).astype(np.int64)
    ev["delta_t"] = (ev["tau_rel"] - ev["block_number"]).clip(lower=0).astype(np.int64)

    # N before event by contract (online unique buyers)
    seen_buyers_by_contract = defaultdict(set)
    n_prev = np.zeros(len(ev), dtype=np.int64)
    for i, r in enumerate(ev.itertuples(index=False)):
        cid = r.contract_id
        b = r.buyer
        n_prev[i] = len(seen_buyers_by_contract[cid])
        seen_buyers_by_contract[cid].add(b)
    ev["N_prev"] = n_prev
    ev["N_current"] = (ev["N_prev"] + 1).clip(lower=1).astype(np.int64)

    # adopter rank by first acquisition order (mint/trade only)
    acq_first = ev[ev["is_acq"] == 1].groupby(["contract_id", "buyer"], as_index=False)["block_number"].min()
    acq_first = acq_first.rename(columns={"block_number": "first_acq_block"})
    acq_first = acq_first.sort_values(["contract_id", "first_acq_block", "buyer"], kind="mergesort")
    acq_first["r_rank"] = acq_first.groupby("contract_id", sort=False).cumcount() + 1
    ev = ev.merge(acq_first[["contract_id", "buyer", "r_rank"]], on=["contract_id", "buyer"], how="left")
    ev["r_rank"] = ev["r_rank"].fillna(1).astype(np.int64)

    if n_ref is None:
        uniq_vals = ev.groupby("contract_id")["buyer"].nunique().values
        n_ref_val = int(np.median(uniq_vals)) if len(uniq_vals) else 1
        n_ref_val = max(1, n_ref_val)
    else:
        n_ref_val = max(1, int(n_ref))

    cap = defaultdict(float)
    last_t = {}

    cap_before = np.zeros(len(ev), dtype=float)
    f1_arr = np.zeros(len(ev), dtype=float)
    ghold_arr = np.zeros(len(ev), dtype=float)
    gearly_arr = np.zeros(len(ev), dtype=float)
    f2_arr = np.zeros(len(ev), dtype=float)
    w_arr = np.zeros(len(ev), dtype=float)
    decay_arr = np.zeros(len(ev), dtype=float)

    for i, r in enumerate(ev.itertuples(index=False)):
        user = r.buyer
        t = int(r.block_number)
        cost = float(r.cost)

        if user in last_t:
            dt = t - int(last_t[user])
            if dt > 0:
                cap[user] *= math.exp(-dt / float(tau_s))
        last_t[user] = t

        cap_u = float(cap[user])
        cap_before[i] = cap_u

        f1 = cost / (cost + cap_u + eps) if cost > 0 else 0.0
        g_hold = 1.0 - math.exp(-float(r.delta_t) / float(max(tau_h, eps)))
        g_early = g_early_from_rank(int(r.r_rank), int(r.N_current), n_ref_val, eps)
        f2 = 0.5 * (g_hold + g_early)

        w_e = f1 * f2
        decay = math.exp(-(tau_end - t) / float(max(TAU_D, eps)))

        f1_arr[i] = f1
        ghold_arr[i] = g_hold
        gearly_arr[i] = g_early
        f2_arr[i] = f2
        w_arr[i] = w_e
        decay_arr[i] = decay

        cap[user] += cost

    ev["Cap_u"] = cap_before
    ev["f1"] = f1_arr
    ev["g_hold"] = ghold_arr
    ev["g_early"] = gearly_arr
    ev["f2"] = f2_arr
    ev["w_e"] = w_arr
    ev["decay"] = decay_arr
    ev["w_rec"] = ev["w_e"] * ev["decay"]

    return ev, n_ref_val


# ============================================================
# 3) Build W^Rec and transitions
# ============================================================
def build_wrec_matrix(ev: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    if ev.empty:
        return sparse.csr_matrix((0, 0), dtype=np.float64), np.array([], dtype=object), np.array([], dtype=object)

    users = ev["buyer"].dropna().astype(str).unique()
    contracts = ev["contract_id"].dropna().astype(str).unique()

    u_map = {u: i for i, u in enumerate(users)}
    c_map = {c: j for j, c in enumerate(contracts)}

    agg = ev.groupby(["buyer", "contract_id"], as_index=False)["w_rec"].sum()
    agg = agg[agg["w_rec"] > 0].copy()

    if agg.empty:
        return sparse.csr_matrix((len(users), len(contracts)), dtype=np.float64), users, contracts

    rows = agg["buyer"].map(u_map).to_numpy(dtype=np.int64)
    cols = agg["contract_id"].map(c_map).to_numpy(dtype=np.int64)
    data = agg["w_rec"].to_numpy(dtype=np.float64)

    W = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(contracts)), dtype=np.float64)
    W.eliminate_zeros()
    return W, users, contracts


def row_stochastic(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    if mat.shape[0] == 0:
        return mat.copy()
    row_sum = np.asarray(mat.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sum, dtype=np.float64)
    nz = row_sum > 0
    inv[nz] = 1.0 / row_sum[nz]
    return sparse.diags(inv) @ mat


# ============================================================
# 4) Priors and Stage C iteration
# ============================================================
def load_u0_prior(users: np.ndarray) -> np.ndarray:
    if len(users) == 0:
        return np.array([], dtype=np.float64)

    src = STAGEB_U0_FULL_CSV if STAGEB_U0_FULL_CSV.exists() else STAGEB_USER_PRIOR_CSV
    if not src.exists():
        return np.full(len(users), 1.0 / len(users), dtype=np.float64)

    df = pd.read_csv(src)
    if "user" not in df.columns:
        return np.full(len(users), 1.0 / len(users), dtype=np.float64)

    if "u0" in df.columns:
        val_col = "u0"
    elif "p0" in df.columns:
        val_col = "p0"
    else:
        return np.full(len(users), 1.0 / len(users), dtype=np.float64)

    prior_map = dict(zip(df["user"].astype(str).str.lower(), pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)))
    u0 = np.array([float(prior_map.get(str(u).lower(), 0.0)) for u in users], dtype=np.float64)

    s = float(u0.sum())
    if s <= 0:
        u0[:] = 1.0 / len(u0)
    else:
        u0 /= s
    return u0


def load_c0_prior(contracts: np.ndarray) -> np.ndarray:
    if len(contracts) == 0:
        return np.array([], dtype=np.float64)

    if not STAGEA_CONTRACT_CSV.exists():
        return np.full(len(contracts), 1.0 / len(contracts), dtype=np.float64)

    df = pd.read_csv(STAGEA_CONTRACT_CSV)
    if "contract_id" not in df.columns:
        return np.full(len(contracts), 1.0 / len(contracts), dtype=np.float64)

    if "cA" in df.columns:
        val_col = "cA"
    else:
        return np.full(len(contracts), 1.0 / len(contracts), dtype=np.float64)

    c_map = dict(zip(df["contract_id"].astype(str), pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)))
    c0 = np.array([float(c_map.get(str(c), 0.0)) for c in contracts], dtype=np.float64)

    s = float(c0.sum())
    if s <= 0:
        c0[:] = 1.0 / len(c0)
    else:
        c0 /= s
    return c0


def stagec_iterate(
    P_u2c: sparse.csr_matrix,
    P_c2u: sparse.csr_matrix,
    u0: np.ndarray,
    c0: np.ndarray,
    alpha: float,
    beta: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0):
        raise ValueError("ALPHA and BETA must be in [0,1].")

    if len(u0) == 0 or len(c0) == 0:
        return u0.copy(), c0.copy(), 0, 0.0

    u = u0.copy()
    c = c0.copy()

    last_delta = 0.0
    for it in range(1, max_iter + 1):
        c_new = beta * (P_u2c.T @ u) + (1.0 - beta) * c0
        u_new = alpha * (P_c2u.T @ c_new) + (1.0 - alpha) * u0

        c_new = np.maximum(c_new, 0.0)
        u_new = np.maximum(u_new, 0.0)

        sc = float(c_new.sum())
        su = float(u_new.sum())
        if sc > 0:
            c_new /= sc
        if su > 0:
            u_new /= su

        delta = float(np.abs(u_new - u).sum() + np.abs(c_new - c).sum())
        u, c = u_new, c_new
        last_delta = delta
        if delta < tol:
            return u, c, it, delta

    return u, c, max_iter, last_delta


# ============================================================
# 5) Save outputs
# ============================================================
def save_outputs(
    W_rec: sparse.csr_matrix,
    users: np.ndarray,
    contracts: np.ndarray,
    user_scores: pd.DataFrame,
    contract_scores: pd.DataFrame,
    meta: pd.DataFrame,
) -> None:
    OUTPUT_CONTRACT_CSV.parent.mkdir(parents=True, exist_ok=True)

    contract_scores.to_csv(OUTPUT_CONTRACT_CSV, index=False, encoding="utf-8-sig")
    user_scores.to_csv(OUTPUT_USER_CSV, index=False, encoding="utf-8-sig")
    sparse.save_npz(OUTPUT_WREC_NPZ, W_rec)

    pd.DataFrame({
        "user_index": np.arange(len(users), dtype=np.int64),
        "user": users,
    }).to_csv(OUTPUT_W_USERS_CSV, index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "contract_index": np.arange(len(contracts), dtype=np.int64),
        "contract_id": contracts,
    }).to_csv(OUTPUT_W_CONTRACTS_CSV, index=False, encoding="utf-8-sig")

    meta.to_csv(OUTPUT_META_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# 6) Main
# ============================================================
def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {INPUT_DIR.resolve()}")

    print(f"[Load] {INPUT_DIR.resolve()}")
    df = load_events(INPUT_DIR)
    print(f"[Data] rows={len(df):,} contracts={df['contract_id'].nunique():,} buyers={df['buyer'].nunique():,} tx_types={df['tx_type'].nunique()}")

    tau_end_global = int(df["block_number"].max()) if len(df) else 0
    tau_end_fair, tau_stats = choose_tau_end(df, TAU_END_MODE, TAU_END_NEAR_GAP_BLOCKS)
    df_window = df[df["block_number"] <= tau_end_fair].copy()
    print(
        f"[Window] mode={TAU_END_MODE} tau_end_fair={tau_end_fair:,} "
        f"tau_end_global={tau_end_global:,} cohort={tau_stats.get('cohort_contracts', 0)}/{tau_stats.get('all_contracts', 0)} "
        f"rows={len(df_window):,}"
    )

    ev, n_ref_val = build_stagec_event_table(
        df=df_window,
        tau_s=TAU_S,
        tau_h=TAU_H,
        n_ref=N_REF,
        eps=EPS,
        tau_end=tau_end_fair,
    )
    print(f"[Events] rows={len(ev):,} tau_end={tau_end_fair:,} N_ref={n_ref_val}")

    W_rec, users, contracts = build_wrec_matrix(ev)
    print(f"[Graph] |U|={len(users):,} |C|={len(contracts):,} nnz(W_rec)={W_rec.nnz:,}")

    P_u2c = row_stochastic(W_rec)
    P_c2u = row_stochastic(W_rec.T.tocsr())

    u0 = load_u0_prior(users)
    c0 = load_c0_prior(contracts)

    uC, cC, n_iter, final_delta = stagec_iterate(
        P_u2c=P_u2c,
        P_c2u=P_c2u,
        u0=u0,
        c0=c0,
        alpha=ALPHA,
        beta=BETA,
        max_iter=MAX_ITER,
        tol=TOL,
    )

    user_scores = pd.DataFrame({"user": users, "uC": uC})
    user_scores = user_scores.sort_values(["uC", "user"], ascending=[False, True]).reset_index(drop=True)
    user_scores["rank_C_user"] = np.arange(1, len(user_scores) + 1)

    contract_scores = pd.DataFrame({"contract_id": contracts, "cC": cC})
    contract_scores = contract_scores.sort_values(["cC", "contract_id"], ascending=[False, True]).reset_index(drop=True)
    contract_scores["rank_C"] = np.arange(1, len(contract_scores) + 1)
    contract_addr_map = compute_contract_address_map(df_window)
    contract_scores = contract_scores.merge(contract_addr_map, on="contract_id", how="left")
    contract_stats = compute_contract_stats(df_window)
    contract_scores = contract_scores.merge(contract_stats, on="contract_id", how="left")

    meta = pd.DataFrame([
        {"key": "tau_end", "value": tau_end_fair},
        {"key": "tau_end_mode", "value": TAU_END_MODE},
        {"key": "tau_end_global", "value": tau_end_global},
        {"key": "tau_end_near_gap_blocks", "value": TAU_END_NEAR_GAP_BLOCKS},
        {"key": "tau_end_cohort_contracts", "value": tau_stats.get("cohort_contracts", 0)},
        {"key": "tau_end_all_contracts", "value": tau_stats.get("all_contracts", 0)},
        {"key": "tau_end_near_lower_bound", "value": tau_stats.get("near_lower_bound", 0)},
        {"key": "tau_d", "value": TAU_D},
        {"key": "tau_s", "value": TAU_S},
        {"key": "tau_h", "value": TAU_H},
        {"key": "n_ref", "value": n_ref_val},
        {"key": "alpha", "value": ALPHA},
        {"key": "beta", "value": BETA},
        {"key": "max_iter", "value": MAX_ITER},
        {"key": "tol", "value": TOL},
        {"key": "iters_used", "value": n_iter},
        {"key": "final_delta_l1", "value": final_delta},
        {"key": "users", "value": len(users)},
        {"key": "contracts", "value": len(contracts)},
        {"key": "events", "value": len(ev)},
        {"key": "events_in_window_raw", "value": len(df_window)},
        {"key": "nnz_wrec", "value": int(W_rec.nnz)},
    ])

    save_outputs(
        W_rec=W_rec,
        users=users,
        contracts=contracts,
        user_scores=user_scores,
        contract_scores=contract_scores,
        meta=meta,
    )

    print(f"[Saved] {OUTPUT_CONTRACT_CSV.resolve()}")
    print(f"[Saved] {OUTPUT_USER_CSV.resolve()}")
    print(f"[Saved] {OUTPUT_WREC_NPZ.resolve()}")
    print(f"[Saved] {OUTPUT_W_USERS_CSV.resolve()}")
    print(f"[Saved] {OUTPUT_W_CONTRACTS_CSV.resolve()}")
    print(f"[Saved] {OUTPUT_META_CSV.resolve()}")

    print("[Top 15 contracts]")
    print(contract_scores.head(15)[["rank_C", "contract_id", "contract_address", "cC"]].to_string(index=False))


if __name__ == "__main__":
    main()
