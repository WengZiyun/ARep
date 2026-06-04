#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage A (Route-A): Value Accounting + Symmetric Normalization + Bipartite Propagation (BiRank-style)

Paper-aligned choices:
  - W^{Hist} = G + V
      * G: sunk-cost endorsements (gas); strictly additive, never decreases.
      * V: peak-anchored value endorsements from trades (per-token conservative approx).
        Only V is affected by peak/rollback retention; G is untouched.
  - Only mint + trade/sale events are used (transfer excluded).
  - Symmetric normalization: S = D_U^{-1/2} W D_C^{-1/2}
  - Bipartite propagation with one damping factor η:
      c^{k+1} = η S^T u^{k} + (1-η) c0
      u^{k+1} = η S  c^{k+1} + (1-η) u0
    with uniform restart distributions u0,c0 to guarantee convergence / avoid rank sinks.

Extra diagnostics (requested):
  - Print a sample W column for one selected contract:
      * show nnz edges, top-k edges, and G vs V decomposition on that column
      * report simple connectivity diagnostics
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse


# ============================================================
# 0) Defaults (edit here; no argparse)
# ============================================================
INPUT_DIR = Path("output/B/2_tranding")              # folder of per-contract CSVs
OUTPUT_CSV = Path("output/C/0_stageA_contract_scores.csv")
OUTPUT_USER_CSV = Path("output/C/0_stageA_user_scores.csv")
OUTPUT_S_NPZ = Path("output/C/0_stageA_W_sym_norm.npz")
OUTPUT_S_USERS_CSV = Path("output/C/0_stageA_W_sym_norm_users.csv")
OUTPUT_S_CONTRACTS_CSV = Path("output/C/0_stageA_W_sym_norm_contracts.csv")

# Stage A hyperparameters
ETA = 0.5
MAX_ITER = 80
TOL = 1e-10

# Value Accounting hyperparameters
MU = 1.0          # retention coefficient for V rollback (mu in (0,1])
W_GAS = 1.0       # weight on G
W_VAL = 1.0       # weight on V
INCLUDE_MINT_GAS = True

# Optional winsorization (robustness if USD conversion has spikes)
WINSOR_Q = 0.0    # set e.g. 0.999; 0 disables

# Debug / inspection
DEBUG_CONTRACT_ID = None   # set to a specific contract_id str; None => auto choose 1st
DEBUG_TOPK_EDGES = 30      # print top edges for the debug contract
DEBUG_MAX_USERS_PRINT = 40 # show at most this many rows if you print a dense slice


# ============================================================
# 1) I/O: load & normalize schema
# ============================================================
REQUIRED_COLS = [
    "block_number", "timestamp", "tx_type",
    "contract_address", "contract_id", "token_id",
    "seller", "buyer",
    "price_usd", "gas_usd",
]

def load_trading_events(input_dir: Path) -> pd.DataFrame:
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
        sub["tx_type"] = sub["tx_type"].astype(str).str.lower().str.strip()
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
    out["contract_address"] = out["contract_address"].astype(str)
    out["buyer"] = out["buyer"].fillna("").astype(str)
    out["seller"] = out["seller"].fillna("").astype(str)

    # deterministic token-state evolution
    out = out.sort_values(["block_number", "tx_index_in_block"], kind="mergesort").reset_index(drop=True)
    return out


# ============================================================
# 2) Stage A: build W^{Hist} = G + V (and expose G,V separately)
# ============================================================
def _winsorize(df: pd.DataFrame, q: float) -> pd.DataFrame:
    if q <= 0:
        return df
    if not (0.5 < q < 1.0):
        raise ValueError("WINSOR_Q must be in (0.5,1.0) or 0 to disable.")
    df = df.copy()
    p_cap = float(df["price"].quantile(q))
    g_cap = float(df["gas"].quantile(q))
    df["price"] = np.minimum(df["price"].values, p_cap)
    df["gas"] = np.minimum(df["gas"].values, g_cap)
    return df


def build_W_hist_GV(
    df: pd.DataFrame,
    mu: float,
    w_gas: float,
    w_val: float,
    include_mint_gas: bool,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    """
    Returns:
      W = G + V,
      plus G and V separately for auditing.

    Important invariant (what you requested):
      - G is additive only; no rollback, no subtraction.
      - Only V uses peak-anchored rollback/retention.
    """
    # Universe
    users = pd.unique(pd.concat([df["buyer"], df["seller"]], ignore_index=True))
    users = users[(users != "") & (~pd.isna(users))]
    contracts = df["contract_id"].dropna().unique()
    if len(users) == 0 or len(contracts) == 0:
        raise ValueError("No valid users/contracts after preprocessing.")

    u_map = {u: i for i, u in enumerate(users)}
    c_map = {c: i for i, c in enumerate(contracts)}

    # Use dict accumulators for sparse construction
    wG = defaultdict(float)
    wV = defaultdict(float)

    def add(dct, u: str, c: str, v: float) -> None:
        if v <= 0:
            return
        ui = u_map.get(u, None)
        cj = c_map.get(c, None)
        if ui is None or cj is None:
            return
        dct[(ui, cj)] += float(v)

    # ---- G: gas endorsements (buyer pays in your current schema)
    gdf = df if include_mint_gas else df[df["tx_type"] != "mint"]
    gas_agg = gdf.groupby(["buyer", "contract_id"], as_index=False)["gas"].sum()
    for r in gas_agg.itertuples(index=False):
        if r.buyer:
            add(wG, r.buyer, r.contract_id, w_gas * float(r.gas))

    # ---- V: peak-anchored conservative per-token accounting (no subtraction on gas)
    token_state = {}
    for r in df.itertuples(index=False):
        key = (r.contract_id, r.token_id)
        st = token_state.get(
            key,
            {"peak_price": 0.0, "peak_holder": "", "last_price": 0.0, "last_holder": ""}
        )

        if r.tx_type == "mint":
            st["last_price"] = 0.0
            st["last_holder"] = r.buyer or ""
            if not st["peak_holder"]:
                st["peak_holder"] = st["last_holder"]
        else:
            price = float(r.price)
            st["last_price"] = price
            st["last_holder"] = r.buyer or ""
            if price > st["peak_price"]:
                st["peak_price"] = price
                st["peak_holder"] = st["last_holder"]
        token_state[key] = st

    for (cid, _tid), st in token_state.items():
        peak_p = float(st["peak_price"])
        last_p = float(st["last_price"])
        last_h = st["last_holder"]
        peak_h = st["peak_holder"]

        # buyer gets execution price (current owner)
        if last_h:
            add(wV, last_h, cid, w_val * last_p)

        # retention (rollback) ONLY affects V, never touches G
        if peak_h and peak_p > last_p:
            add(wV, peak_h, cid, w_val * mu * (peak_p - last_p))

    # Build sparse matrices
    def to_csr(dct: dict, shape: tuple[int, int]) -> sparse.csr_matrix:
        if not dct:
            return sparse.csr_matrix(shape, dtype=np.float64)
        rows = np.fromiter((k[0] for k in dct.keys()), dtype=np.int64)
        cols = np.fromiter((k[1] for k in dct.keys()), dtype=np.int64)
        data = np.fromiter((dct[k] for k in dct.keys()), dtype=np.float64)
        return sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float64)

    shape = (len(users), len(contracts))
    G = to_csr(wG, shape)
    V = to_csr(wV, shape)
    W = (G + V).tocsr()
    W.eliminate_zeros()
    return W, G, V, users, contracts


# ============================================================
# 3) Symmetric normalization + propagation
# ============================================================
def stageA_propagation(W: sparse.csr_matrix, eta: float, max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < eta < 1.0):
        raise ValueError("ETA must be in (0,1).")

    n_u, n_c = W.shape
    S = build_symmetric_normalized_W(W)

    u0 = np.full(n_u, 1.0 / n_u, dtype=np.float64)
    c0 = np.full(n_c, 1.0 / n_c, dtype=np.float64)

    u = u0.copy()
    c = c0.copy()

    for _ in range(max_iter):
        c_new = eta * (S.T @ u) + (1.0 - eta) * c0
        u_new = eta * (S @ c_new) + (1.0 - eta) * u0

        c_new = c_new / (c_new.sum() + 1e-18)
        u_new = u_new / (u_new.sum() + 1e-18)

        if max(np.max(np.abs(c_new - c)), np.max(np.abs(u_new - u))) < tol:
            c, u = c_new, u_new
            break
        c, u = c_new, u_new

    return u, c


def build_symmetric_normalized_W(W: sparse.csr_matrix) -> sparse.csr_matrix:
    n_u, n_c = W.shape
    if n_u == 0 or n_c == 0:
        return sparse.csr_matrix(W.shape, dtype=np.float64)

    d_u = np.asarray(W.sum(axis=1)).reshape(-1)
    d_c = np.asarray(W.sum(axis=0)).reshape(-1)
    d_u = np.maximum(d_u, 1e-12)
    d_c = np.maximum(d_c, 1e-12)
    return (sparse.diags(1.0 / np.sqrt(d_u)) @ W @ sparse.diags(1.0 / np.sqrt(d_c))).tocsr()


def save_symmetric_normalized_W(
    S: sparse.csr_matrix,
    users: np.ndarray,
    contracts: np.ndarray,
    s_npz_path: Path,
    users_csv_path: Path,
    contracts_csv_path: Path,
) -> None:
    s_npz_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(s_npz_path, S)

    pd.DataFrame({
        "user_index": np.arange(len(users), dtype=np.int64),
        "user": users
    }).to_csv(users_csv_path, index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "contract_index": np.arange(len(contracts), dtype=np.int64),
        "contract_id": contracts
    }).to_csv(contracts_csv_path, index=False, encoding="utf-8-sig")


# ============================================================
# 4) Diagnostics: inspect W for one contract
# ============================================================
def debug_print_W_for_one_contract(
    W: sparse.csr_matrix,
    G: sparse.csr_matrix,
    V: sparse.csr_matrix,
    users: np.ndarray,
    contracts: np.ndarray,
    contract_id: str,
    topk: int = 30,
) -> None:
    c_map = {c: j for j, c in enumerate(contracts)}
    if contract_id not in c_map:
        raise ValueError(f"DEBUG_CONTRACT_ID not found: {contract_id}")

    j = c_map[contract_id]

    # Column vectors
    w_col = W[:, j].tocoo()
    g_col = G[:, j].tocoo()
    v_col = V[:, j].tocoo()

    nnz_w = w_col.nnz
    nnz_g = g_col.nnz
    nnz_v = v_col.nnz

    sum_w = float(w_col.data.sum()) if nnz_w else 0.0
    sum_g = float(g_col.data.sum()) if nnz_g else 0.0
    sum_v = float(v_col.data.sum()) if nnz_v else 0.0

    print("\n" + "=" * 72)
    print(f"[DEBUG] Contract = {contract_id}")
    print(f"[DEBUG] Column nnz: W={nnz_w}, G={nnz_g}, V={nnz_v}")
    print(f"[DEBUG] Column sums: W={sum_w:.6g}, G={sum_g:.6g}, V={sum_v:.6g}  (check: W≈G+V)")
    print(f"[DEBUG] Sanity: |W-(G+V)|_1 = {abs(sum_w - (sum_g + sum_v)):.6g}")

    # Top-k edges by W weight
    if nnz_w == 0:
        print("[DEBUG] This contract has no incident user edges in W. (check your event filtering / IDs)")
        return

    idx = np.argsort(-w_col.data)[: min(topk, nnz_w)]
    print(f"\n[DEBUG] Top-{len(idx)} user->contract edges by W weight (showing G and V parts):")
    print("  rank | user_address (truncated) |   W    |   G    |   V")
    # build quick lookup for per-user G,V on this contract
    g_lookup = {int(i): float(x) for i, x in zip(g_col.row, g_col.data)}
    v_lookup = {int(i): float(x) for i, x in zip(v_col.row, v_col.data)}

    for rnk, t in enumerate(idx, 1):
        ui = int(w_col.row[t])
        wv = float(w_col.data[t])
        gv = g_lookup.get(ui, 0.0)
        vv = v_lookup.get(ui, 0.0)
        uaddr = str(users[ui])
        uaddr_short = (uaddr[:10] + "..." + uaddr[-6:]) if len(uaddr) > 20 else uaddr
        print(f"  {rnk:>4d} | {uaddr_short:<22s} | {wv:>6.3g} | {gv:>6.3g} | {vv:>6.3g}")

    # Connectivity hints
    # - how many users have any edge at all
    deg_u = np.asarray(W.sum(axis=1)).reshape(-1)
    active_users = int(np.sum(deg_u > 0))
    print(f"\n[DEBUG] Graph activity: active_users={active_users}/{len(users)} ({active_users/len(users):.2%})")
    # contract degree (how many distinct users touch this contract)
    distinct_users_on_c = nnz_w
    print(f"[DEBUG] Contract degree (distinct users with W>0): {distinct_users_on_c}")
    print("=" * 72 + "\n")


# ============================================================
# 5) Export helpers
# ============================================================
def compute_contract_stats(df: pd.DataFrame) -> pd.DataFrame:
    base = df.groupby("contract_id", as_index=False).agg(
        contract_address=("contract_address", "first"),
        first_block=("block_number", "min"),
        last_block=("block_number", "max"),
        mint_cnt=("tx_type", lambda s: int((s == "mint").sum())),
        trade_cnt=("tx_type", lambda s: int((s == "trade").sum())),
        gas_total_usd=("gas", "sum"),
        trade_value_total_usd=("price", lambda s: float(s[df.loc[s.index, "tx_type"].eq("trade")].sum())),
    )

    trade_df = df[df["tx_type"] == "trade"]
    buyers = trade_df[["contract_id", "buyer"]].rename(columns={"buyer": "user"})
    sellers = trade_df[["contract_id", "seller"]].rename(columns={"seller": "user"})
    all_users = pd.concat([buyers, sellers], ignore_index=True)
    all_users = all_users[all_users["user"].astype(str) != ""]
    uniq = all_users.groupby("contract_id", as_index=False)["user"].nunique().rename(columns={"user": "unique_trade_users"})

    out = base.merge(uniq, on="contract_id", how="left")
    out["unique_trade_users"] = out["unique_trade_users"].fillna(0).astype(int)
    return out


def save_user_scores(users: np.ndarray, uA: np.ndarray, output_csv: Path) -> None:
    user_scores = pd.DataFrame({"user": users, "uA": uA})
    user_scores = user_scores.sort_values(["uA", "user"], ascending=[False, True]).reset_index(drop=True)
    user_scores["rank_A_user"] = np.arange(1, len(user_scores) + 1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    user_scores.to_csv(output_csv, index=False, encoding="utf-8-sig")


# ============================================================
# 6) Main (no arguments)
# ============================================================
def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {INPUT_DIR.resolve()}")

    print(f"[Load] {INPUT_DIR.resolve()}")
    df = load_trading_events(INPUT_DIR)
    print(f"[Data] rows={len(df):,} contracts={df['contract_id'].nunique():,} "
          f"buyers={df['buyer'].nunique():,} sellers={df['seller'].nunique():,}")
    print(f"[Data] tx_type counts: {df['tx_type'].value_counts().to_dict()}")

    df = _winsorize(df, WINSOR_Q)

    W, G, V, users, contracts = build_W_hist_GV(
        df=df, mu=MU, w_gas=W_GAS, w_val=W_VAL, include_mint_gas=INCLUDE_MINT_GAS
    )
    print(f"[Graph] |U|={len(users):,} |C|={len(contracts):,} nnz(W)={W.nnz:,} "
          f"nnz(G)={G.nnz:,} nnz(V)={V.nnz:,}")

    # Export symmetric-normalized W for downstream inspection/use.
    S = build_symmetric_normalized_W(W)
    save_symmetric_normalized_W(
        S=S,
        users=users,
        contracts=contracts,
        s_npz_path=OUTPUT_S_NPZ,
        users_csv_path=OUTPUT_S_USERS_CSV,
        contracts_csv_path=OUTPUT_S_CONTRACTS_CSV,
    )
    print(f"[Saved] {OUTPUT_S_NPZ.resolve()}")
    print(f"[Saved] {OUTPUT_S_USERS_CSV.resolve()}")
    print(f"[Saved] {OUTPUT_S_CONTRACTS_CSV.resolve()}")

    # Debug: show one contract's W column decomposition
    debug_c = DEBUG_CONTRACT_ID if DEBUG_CONTRACT_ID is not None else str(contracts[0])
    debug_print_W_for_one_contract(W, G, V, users, contracts, debug_c, topk=DEBUG_TOPK_EDGES)

    # Stage A propagation
    print("[Stage A] propagation...")
    uA, cA = stageA_propagation(W, eta=ETA, max_iter=MAX_ITER, tol=TOL)
    save_user_scores(users=users, uA=uA, output_csv=OUTPUT_USER_CSV)
    print(f"[Saved] {OUTPUT_USER_CSV.resolve()}")

    # Export scores + diagnostics
    scores = pd.DataFrame({"contract_id": contracts, "cA": cA})
    scores = scores.sort_values(["cA", "contract_id"], ascending=[False, True]).reset_index(drop=True)
    scores["rank_A"] = np.arange(1, len(scores) + 1)

    stats = compute_contract_stats(df)
    out = scores.merge(stats, on="contract_id", how="left")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[Saved] {OUTPUT_CSV.resolve()}")

    print("[Top 15]")
    print(out.head(15)[
        ["rank_A", "contract_id", "cA", "mint_cnt", "trade_cnt",
         "gas_total_usd", "trade_value_total_usd", "unique_trade_users"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
