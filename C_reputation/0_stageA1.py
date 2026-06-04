#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage A (Route-A): Anchor Contract Scoring with Uniform Restart

Implements:
  - Build W^Hist = G + V from on-chain trading events (mint/trade).
  - Symmetric normalization S = D_U^{-1/2} W D_C^{-1/2}.
  - Standard bipartite propagation (BiRank-style) with uniform restart priors:
        c^{k+1} = eta * S^T u^k + (1-eta) * c0
        u^{k+1} = eta * S   c^{k+1} + (1-eta) * u0
    where u0 and c0 are uniform distributions.
  - Normalize (L1) each iteration for numerical stability.

Notes:
  - No age/maturity prior is used (Route-A).
  - Works with sparse matrices for scalability.
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------
# IO & preprocessing
# ---------------------------

def load_trading_events(input_dir: Path) -> pd.DataFrame:
    """
    Load per-contract CSV files and convert to Stage-A schema.

    Required columns (per CSV):
      block_number, timestamp, tx_type, contract_address, contract_id,
      token_id, seller, buyer, price_usd, gas_usd
    Optional:
      tx_index_in_block

    Output columns:
      block_number (int64), timestamp (datetime64),
      tx_type in {"mint","trade"},
      contract_id (str), token_id (str),
      buyer (str), seller (str),
      price (float), gas (float),
      tx_index_in_block (int64)
    """
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {input_dir}")

    required = [
        "block_number",
        "timestamp",
        "tx_type",
        "contract_address",
        "contract_id",
        "token_id",
        "seller",
        "buyer",
        "price_usd",
        "gas_usd",
    ]

    frames = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"Missing columns in {fp.name}: {miss}")

        cols = required + (["tx_index_in_block"] if "tx_index_in_block" in df.columns else [])
        sub = df[cols].copy()
        if "tx_index_in_block" not in sub.columns:
            sub["tx_index_in_block"] = 0

        sub = sub.rename(columns={"price_usd": "price", "gas_usd": "gas"})
        sub["tx_type"] = sub["tx_type"].astype(str).str.lower()

        # Stage A uses mint + trade (you can extend if needed)
        sub = sub[sub["tx_type"].isin(["mint", "trade", "sale"])].copy()
        sub["tx_type"] = sub["tx_type"].replace({"sale": "trade"})

        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)

    out["block_number"] = pd.to_numeric(out["block_number"], errors="coerce").fillna(0).astype(np.int64)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    out["tx_index_in_block"] = pd.to_numeric(out["tx_index_in_block"], errors="coerce").fillna(0).astype(np.int64)

    out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0.0).astype(float)
    out["gas"] = pd.to_numeric(out["gas"], errors="coerce").fillna(0.0).astype(float)

    out["token_id"] = out["token_id"].astype(str)
    out["contract_id"] = out["contract_id"].astype(str)

    out["buyer"] = out["buyer"].fillna("").astype(str)
    out["seller"] = out["seller"].fillna("").astype(str)

    # Keep stable ordering for reproducibility
    out = out.sort_values(["block_number", "tx_index_in_block"]).reset_index(drop=True)
    return out


def compute_contract_output_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Optional: simple descriptive stats per contract for debugging/reporting."""
    base = (
        df.groupby("contract_id", as_index=False)
        .agg(
            contract_address=("contract_address", "first"),
            birth_time=("timestamp", "first"),
            last_time=("timestamp", "last"),
        )
    )

    trade = df[df["tx_type"] == "trade"].copy()
    if trade.empty:
        out = base.copy()
        out["trade_count"] = 0
        out["trade_price_mean_usd"] = 0.0
        out["active_days"] = 1.0
        out["trade_frequency_per_day"] = 0.0
        out["unique_users_count"] = 0
        return out

    trade_basic = (
        trade.groupby("contract_id", as_index=False)
        .agg(trade_count=("price", "size"),
             trade_price_mean_usd=("price", "mean"))
    )

    buyers = trade[["contract_id", "buyer"]].rename(columns={"buyer": "user"})
    sellers = trade[["contract_id", "seller"]].rename(columns={"seller": "user"})
    all_users = pd.concat([buyers, sellers], ignore_index=True)
    all_users = all_users[all_users["user"].astype(str) != ""]
    user_cnt = all_users.groupby("contract_id", as_index=False)["user"].nunique()
    user_cnt = user_cnt.rename(columns={"user": "unique_users_count"})

    out = base.merge(trade_basic, on="contract_id", how="left").merge(user_cnt, on="contract_id", how="left")
    out["trade_count"] = out["trade_count"].fillna(0).astype(int)
    out["trade_price_mean_usd"] = out["trade_price_mean_usd"].fillna(0.0)
    out["unique_users_count"] = out["unique_users_count"].fillna(0).astype(int)

    delta_days = (out["last_time"] - out["birth_time"]).dt.total_seconds() / 86400.0
    out["active_days"] = delta_days.fillna(0.0).clip(lower=1.0)
    out["trade_frequency_per_day"] = out["trade_count"] / out["active_days"]
    return out


# ---------------------------
# Build W^Hist = G + V
# ---------------------------

def build_whist_W(
    df: pd.DataFrame,
    mu: float = 1.0,
    w_gas: float = 1.0,
    w_val: float = 1.0,
    include_mint_gas: bool = True,
):
    """
    Build historical endorsement matrix W^Hist \in R^{|U| x |C|}.

    Gas term:
      G_{u,c} = sum_{e: buyer=u, contract=c} gas(e)  (mint included optionally)

    Value term (peak-anchored retention, simplified per-token ledger):
      For each (contract, token):
        - track last_holder, last_price
        - track peak_holder, peak_price
      At end:
        - credit last_holder with last_price
        - if peak_price > last_price, credit peak_holder with mu*(peak_price-last_price)

    Returns:
      W (csr_matrix), users (np.array of str), contracts (np.array of str)
    """
    # node sets
    users = pd.unique(pd.concat([df["buyer"], df["seller"]], ignore_index=True))
    users = users[(users != "") & (~pd.isna(users))]
    contracts = df["contract_id"].dropna().unique()

    if len(users) == 0 or len(contracts) == 0:
        raise ValueError("Empty user/contract set after preprocessing.")

    u_map = {u: i for i, u in enumerate(users)}
    c_map = {c: i for i, c in enumerate(contracts)}

    weight = defaultdict(float)

    def add(u: str, c: str, v: float):
        if v == 0.0:
            return
        if u not in u_map or c not in c_map:
            return
        weight[(u_map[u], c_map[c])] += float(v)

    # --- Gas component (buyer pays gas) ---
    gdf = df if include_mint_gas else df[df["tx_type"] != "mint"]
    gas_agg = gdf.groupby(["buyer", "contract_id"], as_index=False)["gas"].sum()
    for r in gas_agg.itertuples(index=False):
        add(r.buyer, r.contract_id, w_gas * float(r.gas))

    # --- Value component (peak-anchored, per token) ---
    token_state = {}  # (contract_id, token_id) -> dict
    for r in df.itertuples(index=False):
        key = (r.contract_id, r.token_id)
        st = token_state.get(
            key,
            {"peak_price": 0.0, "peak_holder": None, "last_price": 0.0, "last_holder": None},
        )

        if r.tx_type == "mint":
            # mint treated as acquisition at price 0 (value part),
            # buyer becomes last_holder; peak_holder initialized if absent
            st["last_price"] = 0.0
            st["last_holder"] = r.buyer
            if st["peak_holder"] is None:
                st["peak_holder"] = r.buyer
        else:
            # trade
            price = float(r.price)
            st["last_price"] = price
            st["last_holder"] = r.buyer
            if price > st["peak_price"]:
                st["peak_price"] = price
                st["peak_holder"] = r.buyer

        token_state[key] = st

    for (cid, _tid), st in token_state.items():
        peak_p = float(st["peak_price"])
        last_p = float(st["last_price"])
        last_h = st["last_holder"]
        peak_h = st["peak_holder"]

        if last_h is not None:
            add(last_h, cid, w_val * last_p)
        if peak_h is not None and peak_p > last_p:
            add(peak_h, cid, w_val * mu * (peak_p - last_p))

    if not weight:
        raise ValueError("No valid edges built for W^Hist (all weights zero).")

    rows = [k[0] for k in weight.keys()]
    cols = [k[1] for k in weight.keys()]
    data = [weight[k] for k in weight.keys()]
    W = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(contracts)))
    return W, users, contracts


# ---------------------------
# Stage A: BiRank with uniform restart
# ---------------------------

def birank_uniform_restart(
    W: sparse.csr_matrix,
    eta: float = 0.85,
    max_iter: int = 80,
    tol: float = 1e-10,
):
    """
    BiRank-style bipartite propagation with uniform restart priors (Route-A).

    Steps:
      S = D_U^{-1/2} W D_C^{-1/2}
      u0 = uniform over users, c0 = uniform over contracts
      iterate:
        c <- eta * S^T u + (1-eta) * c0
        u <- eta * S   c + (1-eta) * u0
      L1-normalize u and c each step.
    """
    if not (0.0 < eta < 1.0):
        raise ValueError("eta must be in (0,1).")

    n_u, n_c = W.shape
    if n_u == 0 or n_c == 0:
        raise ValueError("W has empty shape.")

    # degrees
    d_u = np.asarray(W.sum(axis=1)).ravel()
    d_c = np.asarray(W.sum(axis=0)).ravel()

    # avoid division by zero
    d_u = np.where(d_u > 0, d_u, 1.0)
    d_c = np.where(d_c > 0, d_c, 1.0)

    # symmetric normalization
    Du_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_u))
    Dc_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_c))
    S = Du_inv_sqrt @ W @ Dc_inv_sqrt

    # uniform restart priors
    u0 = np.full(n_u, 1.0 / n_u, dtype=float)
    c0 = np.full(n_c, 1.0 / n_c, dtype=float)

    u = u0.copy()
    c = c0.copy()

    for _ in range(max_iter):
        c_new = eta * (S.T @ u) + (1.0 - eta) * c0
        u_new = eta * (S @ c_new) + (1.0 - eta) * u0

        # L1 normalize for numerical stability
        c_new = c_new / (c_new.sum() + 1e-15)
        u_new = u_new / (u_new.sum() + 1e-15)

        # convergence check (infinity norm)
        if np.max(np.abs(c_new - c)) < tol and np.max(np.abs(u_new - u)) < tol:
            c, u = c_new, u_new
            break

        c, u = c_new, u_new

    return u, c


def compute_stageA_contract_scores(
    df: pd.DataFrame,
    eta: float = 0.85,
    mu: float = 1.0,
    w_gas: float = 1.0,
    w_val: float = 1.0,
    include_mint_gas: bool = True,
    max_iter: int = 80,
    tol: float = 1e-10,
):
    """
    End-to-end Stage A:
      - Build W^Hist (CSR)
      - Run BiRank with uniform restart
      - Return contract ranking DataFrame
    """
    W, users, contracts = build_whist_W(
        df=df,
        mu=mu,
        w_gas=w_gas,
        w_val=w_val,
        include_mint_gas=include_mint_gas,
    )

    _, cA = birank_uniform_restart(W=W, eta=eta, max_iter=max_iter, tol=tol)

    out = pd.DataFrame({"contract_id": contracts, "cA": cA})
    out = out.sort_values("cA", ascending=False).reset_index(drop=True)
    out["rank_A"] = np.arange(1, len(out) + 1)
    return out


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage A (Route-A) contract scoring with uniform restart.")
    parser.add_argument("--input_dir", type=str, default="output/B/2_tranding")
    parser.add_argument("--output_file", type=str, default="output/C/0_stageA_contract_scores.csv")

    # W^Hist parameters
    parser.add_argument("--mu", type=float, default=1.0, help="retention coefficient mu in (0,1], used in V term")
    parser.add_argument("--w_gas", type=float, default=1.0, help="weight for gas component")
    parser.add_argument("--w_val", type=float, default=1.0, help="weight for value component")
    parser.add_argument("--include_mint_gas", action="store_true", help="include mint gas in G term")

    # BiRank parameters
    parser.add_argument("--eta", type=float, default=0.85, help="damping factor eta in (0,1)")
    parser.add_argument("--max_iter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-10)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    print(f"[Load] reading CSVs from: {input_dir}")
    df = load_trading_events(input_dir)

    print(
        f"[Data] rows={len(df):,}, contracts={df['contract_id'].nunique():,}, "
        f"buyers={df['buyer'].nunique():,}, sellers={df['seller'].nunique():,}"
    )
    print(f"[Data] tx_type counts: {df['tx_type'].value_counts().to_dict()}")

    print("[Stage A] building W^Hist and computing contract scores (uniform restart)...")
    scores = compute_stageA_contract_scores(
        df=df,
        eta=args.eta,
        mu=args.mu,
        w_gas=args.w_gas,
        w_val=args.w_val,
        include_mint_gas=args.include_mint_gas,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    extra = compute_contract_output_stats(df)
    scores = scores.merge(extra, on="contract_id", how="left")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_path, index=False)

    print(f"[Saved] {out_path}")
    print("[Top10]")
    print(scores.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
