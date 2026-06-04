import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse


def load_trading_events(input_dir: Path) -> pd.DataFrame:
    """
    Load per-contract CSV files from output/B/2_tranding and convert to Stage-A schema.
    Required output columns:
      block_number, timestamp, tx_type, contract_address, contract_id,
      token_id, seller, buyer, price, gas, tx_index_in_block
    """
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {input_dir}")

    frames = []
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

    for fp in csv_files:
        df = pd.read_csv(fp)
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"Missing columns in {fp.name}: {miss}")

        sub = df[required + (["tx_index_in_block"] if "tx_index_in_block" in df.columns else [])].copy()
        if "tx_index_in_block" not in sub.columns:
            sub["tx_index_in_block"] = 0

        sub = sub.rename(columns={"price_usd": "price", "gas_usd": "gas"})
        sub["tx_type"] = sub["tx_type"].astype(str).str.lower()
        # Keep only events used by Stage A logic.
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
    return out


def build_contracts_meta(df: pd.DataFrame) -> pd.DataFrame:
    birth = (
        df.groupby("contract_id", as_index=False)["block_number"]
        .min()
        .rename(columns={"block_number": "birth_block"})
    )
    return birth


def compute_contract_output_stats(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.sort_values(["block_number", "tx_index_in_block"])
        .groupby("contract_id", as_index=False)
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
        .agg(trade_count=("price", "size"), trade_price_mean_usd=("price", "mean"))
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


def build_whist_W(
    df: pd.DataFrame,
    mu: float = 1.0,
    w_gas: float = 1.0,
    w_val: float = 1.0,
    include_mint_gas: bool = True,
):
    users = pd.unique(pd.concat([df["buyer"], df["seller"]], ignore_index=True))
    users = users[(users != "") & (~pd.isna(users))]
    contracts = df["contract_id"].dropna().unique()

    u_map = {u: i for i, u in enumerate(users)}
    c_map = {c: i for i, c in enumerate(contracts)}

    weight = defaultdict(float)

    def add(u, c, v):
        if v == 0 or u not in u_map or c not in c_map:
            return
        weight[(u_map[u], c_map[c])] += float(v)

    gdf = df if include_mint_gas else df[df["tx_type"] != "mint"]
    gas_agg = gdf.groupby(["buyer", "contract_id"], as_index=False)["gas"].sum()
    for r in gas_agg.itertuples(index=False):
        add(r.buyer, r.contract_id, w_gas * r.gas)

    sdf = df.sort_values(["block_number", "tx_index_in_block"]).reset_index(drop=True)
    token_state = {}
    for r in sdf.itertuples(index=False):
        key = (r.contract_id, r.token_id)
        st = token_state.get(
            key,
            {"peak_price": 0.0, "peak_holder": None, "last_price": 0.0, "last_holder": None},
        )

        if r.tx_type == "mint":
            st["last_price"] = 0.0
            st["last_holder"] = r.buyer
            if st["peak_holder"] is None:
                st["peak_holder"] = r.buyer
        else:
            price = float(r.price)
            st["last_price"] = price
            st["last_holder"] = r.buyer
            if price > st["peak_price"]:
                st["peak_price"] = price
                st["peak_holder"] = r.buyer

        token_state[key] = st

    for (cid, _tid), st in token_state.items():
        peak_p = st["peak_price"]
        last_p = st["last_price"]
        if st["last_holder"] is not None:
            add(st["last_holder"], cid, w_val * last_p)
        if st["peak_holder"] is not None and peak_p > last_p:
            add(st["peak_holder"], cid, w_val * mu * (peak_p - last_p))

    if not weight:
        raise ValueError("No valid edges built for W_Hist.")

    rows = [k[0] for k in weight]
    cols = [k[1] for k in weight]
    data = [weight[k] for k in weight]
    W = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(contracts)))
    return W, users, contracts


def birank_with_contract_prior(
    W: sparse.csr_matrix,
    alpha: float,
    beta: float,
    c0: np.ndarray,
    max_iter: int = 60,
    tol: float = 1e-8,
):
    n_u, n_c = W.shape
    d_u = np.array(W.sum(axis=1)).flatten()
    d_c = np.array(W.sum(axis=0)).flatten()
    d_u[d_u == 0] = 1.0
    d_c[d_c == 0] = 1.0

    S = sparse.diags(1.0 / np.sqrt(d_u)) @ W @ sparse.diags(1.0 / np.sqrt(d_c))

    u0 = np.ones(n_u) / n_u
    c0 = c0 / (c0.sum() + 1e-12)
    u = u0.reshape(-1, 1)
    c = c0.reshape(-1, 1)
    u0v = u.copy()
    c0v = c.copy()

    for _ in range(max_iter):
        c_new = alpha * (S.T @ u) + (1.0 - alpha) * c0v
        u_new = beta * (S @ c_new) + (1.0 - beta) * u0v

        c_new = c_new / (c_new.sum() + 1e-12)
        u_new = u_new / (u_new.sum() + 1e-12)

        if np.max(np.abs(c_new - c)) < tol and np.max(np.abs(u_new - u)) < tol:
            c, u = c_new, u_new
            break
        c, u = c_new, u_new

    return u.flatten(), c.flatten()


def compute_A_contract_scores(
    df: pd.DataFrame,
    contracts_meta: pd.DataFrame,
    alpha_A: float = 0.7,
    beta_A: float = 0.85,
    age_power: float = 2.0,
):
    W, users, contracts = build_whist_W(df, mu=1.0, include_mint_gas=True)
    if len(users) == 0 or len(contracts) == 0:
        raise ValueError("Empty users/contracts after preprocessing.")

    birth_map = dict(zip(contracts_meta["contract_id"], contracts_meta["birth_block"]))
    eval_block = int(df["block_number"].max())
    age_blocks = np.array(
        [max(1, eval_block - int(birth_map.get(cid, eval_block))) for cid in contracts],
        dtype=float,
    )

    c0 = np.log1p(age_blocks) ** age_power
    c0 = c0 / (c0.sum() + 1e-12)

    _, cA = birank_with_contract_prior(W=W, alpha=alpha_A, beta=beta_A, c0=c0)
    out = pd.DataFrame({"contract_id": contracts, "cA": cA})
    out = out.sort_values("cA", ascending=False).reset_index(drop=True)
    out["rank_A"] = np.arange(1, len(out) + 1)
    return out


def main():
    parser = argparse.ArgumentParser(description="Stage A contract scoring on output/B/2_tranding data.")
    parser.add_argument("--input_dir", type=str, default="output/B/2_tranding")
    parser.add_argument("--output_file", type=str, default="output/C/0_stageA_contract_scores.csv")
    parser.add_argument("--alpha_A", type=float, default=0.7)
    parser.add_argument("--beta_A", type=float, default=0.85)
    parser.add_argument("--age_power", type=float, default=2.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    print(f"[Load] reading CSVs from: {input_dir}")
    df = load_trading_events(input_dir)
    contracts_meta = build_contracts_meta(df)
    print(
        f"[Data] rows={len(df):,}, contracts={df['contract_id'].nunique():,}, "
        f"buyers={df['buyer'].nunique():,}, sellers={df['seller'].nunique():,}"
    )
    print(f"[Data] tx_type counts: {df['tx_type'].value_counts().to_dict()}")

    print("[Stage A] computing contract scores...")
    scores = compute_A_contract_scores(
        df=df,
        contracts_meta=contracts_meta,
        alpha_A=args.alpha_A,
        beta_A=args.beta_A,
        age_power=args.age_power,
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