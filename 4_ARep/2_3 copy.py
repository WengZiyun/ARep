import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import sparse

import math
from collections import defaultdict

# 合约自适应 tau_h_j：在线均值 (MLE)
hold_sum_by_c = defaultdict(float)   # Σ hold
hold_cnt_by_c = defaultdict(int)     # count
global_hold_sum = 0.0
global_hold_cnt = 0

def get_tau_h_j(contract_id: str, eps: float = 1e-12) -> float:
    """返回合约 j 的 tau_h_j；若样本不足，用全局均值兜底。"""
    if hold_cnt_by_c[contract_id] > 0:
        return hold_sum_by_c[contract_id] / (hold_cnt_by_c[contract_id] + eps)
    # 合约没历史样本：用全局均值
    if global_hold_cnt > 0:
        return global_hold_sum / (global_hold_cnt + eps)
    # 仍然没有：给一个很小但非零的兜底，避免除零
    return 1.0

def update_tau_stats(contract_id: str, hold_blocks: float):
    """用当前事件的 hold 更新合约/全局统计（必须在用完 tau_h_j 之后再更新，避免泄漏）。"""
    global global_hold_sum, global_hold_cnt
    hold_sum_by_c[contract_id] += hold_blocks
    hold_cnt_by_c[contract_id] += 1
    global_hold_sum += hold_blocks
    global_hold_cnt += 1

# -------------------------
# Utils
# -------------------------
def g_early_log_norm(rank_r: int, N_unique: int, N_ref: int, eps: float = 1e-12) -> float:
    """
    g_early = 1 - ln(1+q) / ln(1+q_max)
      q     = (r-1) / ((N-1)+N_ref)
      q_max = (N-1) / ((N-1)+N_ref)

    性质：
      - r 越小(越早)，g_early 越大
      - 边际扣分递减：|dg/dr| = 1 / (ln(1+q_max) * (D + r - 1))，前面扣得快，后面扣得慢
    注意：math.log / log1p 是自然对数 ln
    """
    if N_unique <= 1:
        return 1.0  # 只有1个独立买家时，不惩罚“晚入场”

    D = (N_unique - 1) + float(N_ref)

    q = (rank_r - 1.0) / (D + eps)
    q_max = (N_unique - 1.0) / (D + eps)

    denom = math.log1p(q_max)  # ln(1+q_max)
    if denom <= 0:
        return 1.0

    val = 1.0 - math.log1p(q) / denom
    return max(0.0, min(1.0, val))



def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def auc_manual(y_true, scores) -> float:
    """
    AUC via rank statistic (handles ties by average rank).
    y_true: 1 pos, 0 neg
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(scores).rank(method="average").values
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# -------------------------
# A Stage: Whist(HWM+Gas) -> BiRank with contract prior
# -------------------------
def build_whist_W(
    df: pd.DataFrame,
    mu: float = 1.0,
    w_gas: float = 1.0,
    w_val: float = 1.0,
    include_mint_gas: bool = True,
):
    """
    Build W_Hist (users x contracts) with:
      - gas accumulation (buyer -> contract)
      - HWM-PnL style: last holder keeps last_price; peak holder keeps (peak-last)
    Note: token_id is used to track each NFT path.
    """
    # user universe: both buyers and sellers (credit can accrue to holders)
    users = pd.unique(pd.concat([df["buyer"], df["seller"]], ignore_index=True))
    users = users[(users != "") & (~pd.isna(users))]
    contracts = df["contract_id"].unique()

    u_map = {u: i for i, u in enumerate(users)}
    c_map = {c: i for i, c in enumerate(contracts)}

    weight = defaultdict(float)

    def add(u, c, v):
        if v == 0:
            return
        if u not in u_map:
            return
        weight[(u_map[u], c_map[c])] += float(v)

    # gas part
    gdf = df if include_mint_gas else df[df["tx_type"] != "mint"]
    gas_agg = gdf.groupby(["buyer", "contract_id"])["gas"].sum().reset_index()
    for r in gas_agg.itertuples(index=False):
        add(r.buyer, r.contract_id, w_gas * r.gas)

    # token state for HWM-PnL
    sdf = df.sort_values(["block_number", "tx_index_in_block"]).reset_index(drop=True)
    token_state = {}  # (contract, token_id) -> dict

    for r in sdf.itertuples(index=False):
        cid = r.contract_id
        tid = int(r.token_id)
        key = (cid, tid)

        st = token_state.get(
            key,
            {"peak_price": 0.0, "peak_holder": None, "last_price": 0.0, "last_holder": None},
        )

        if str(r.tx_type) == "mint":
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

    # project token states back to (user, contract)
    for (cid, _tid), st in token_state.items():
        peak_p = st["peak_price"]
        last_p = st["last_price"]

        if st["last_holder"] is not None:
            add(st["last_holder"], cid, w_val * last_p)

        if st["peak_holder"] is not None and peak_p > last_p:
            add(st["peak_holder"], cid, w_val * mu * (peak_p - last_p))

    rows = [k[0] for k in weight.keys()]
    cols = [k[1] for k in weight.keys()]
    data = [v for v in weight.values()]

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
    """
    BiRank with personalized contract prior c0:
      c <- alpha S^T u + (1-alpha) c0
      u <- beta  S  c + (1-beta)  u0
    """
    n_u, n_c = W.shape
    d_u = np.array(W.sum(axis=1)).flatten()
    d_c = np.array(W.sum(axis=0)).flatten()
    d_u[d_u == 0] = 1.0
    d_c[d_c == 0] = 1.0

    S = sparse.diags(1.0 / np.sqrt(d_u)) @ W @ sparse.diags(1.0 / np.sqrt(d_c))

    u0 = np.ones(n_u) / n_u
    u = u0.reshape(-1, 1)
    c = c0.reshape(-1, 1)

    u0v = u0.reshape(-1, 1)
    c0v = c0.reshape(-1, 1)

    for _ in range(max_iter):
        c_new = alpha * (S.T @ u) + (1 - alpha) * c0v
        u_new = beta * (S @ c_new) + (1 - beta) * u0v

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
    """
    A stage: compute time-tested contract hardness score cA.
    age prior is the key to suppress "young isolated sybil" dominating A.
    """
    W, users, contracts = build_whist_W(df, mu=1.0, include_mint_gas=True)

    birth_map = dict(zip(contracts_meta["contract_id"], contracts_meta["birth_block"]))
    eval_block = int(df["block_number"].max())

    age_blocks = np.array(
        [max(1, eval_block - int(birth_map.get(cid, eval_block))) for cid in contracts],
        dtype=float,
    )
    # strong age prior
    c0 = (np.log1p(age_blocks) ** age_power)
    c0 = c0 / c0.sum()

    _u, cA = birank_with_contract_prior(W, alpha=alpha_A, beta=beta_A, c0=c0)
    scores = pd.DataFrame({"contract_id": contracts, "cA": cA}).sort_values("cA", ascending=False).reset_index(drop=True)
    return scores


# -------------------------
# B Stage: user p0 from trusted contracts only
# -------------------------
def enrich_hold_and_age(df: pd.DataFrame, contracts_meta: pd.DataFrame):
    """
    For each trade (buy) event:
      - age_at_buy_blocks = buy_block - birth_block
      - hold_blocks = next_trade_block - buy_block, else eval_block - buy_block
    Mint rows keep hold=0/age=0 (ignored by B anyway).
    """
    df = df.sort_values(["block_number", "tx_index_in_block"]).reset_index(drop=True).copy()
    df["hold_blocks"] = 0
    df["age_at_buy_blocks"] = 0

    birth_map = dict(zip(contracts_meta["contract_id"], contracts_meta["birth_block"]))
    eval_block = int(df["block_number"].max())

    last_buy_idx = {}  # (cid, tid) -> last trade row index

    for idx, r in enumerate(df.itertuples(index=False)):
        if str(r.tx_type) != "trade":
            continue

        cid = r.contract_id
        tid = int(r.token_id)
        buy_block = int(r.block_number)

        birth_block = int(birth_map.get(cid, buy_block))
        df.at[idx, "age_at_buy_blocks"] = max(0, buy_block - birth_block)

        key = (cid, tid)
        if key in last_buy_idx:
            prev_idx = last_buy_idx[key]
            prev_buy_block = int(df.at[prev_idx, "block_number"])
            df.at[prev_idx, "hold_blocks"] = max(0, buy_block - prev_buy_block)

        last_buy_idx[key] = idx

    for _key, idx in last_buy_idx.items():
        if df.at[idx, "hold_blocks"] == 0:
            df.at[idx, "hold_blocks"] = max(0, eval_block - int(df.at[idx, "block_number"]))

    return df


def compute_B_user_scores(
    df: pd.DataFrame,
    users_meta: pd.DataFrame,
    contract_scores_A: pd.DataFrame,
    trusted_ratio: float = 0.01,
    tau_s: int = 200_000,   # EWMA scale (blocks)
    tau_h: int = 50_000,    # fallback holding scale (blocks) - 仅在无历史时兜底
):
    """
    w_e = f1(e) * f2(e)
      f1 = share, share = cost / (cost + Cap_i + eps)
      f2 = (g_hold + g_early)/2
         g_hold  = 1 - exp(-hold/tau_h_j)                      # 合约自适应(已实现)
         g_early = 1 / (1 + ln(1 + q * ln(1+N_unique_buyers)))  # <-- 替换为方案A (ln版)
           q = (r-1)/((N-1)+N_ref)
           N = unique buyers up to current t (online)
           r = adopter rank (buyer first adoption order in this contract)
    Only trade events on trusted contracts contribute to user score.
    """

    # -----------------------------
    # Reset adaptive tau stats (避免跨次调用污染)
    # -----------------------------
    global global_hold_sum, global_hold_cnt
    hold_sum_by_c.clear()
    hold_cnt_by_c.clear()
    global_hold_sum = 0.0
    global_hold_cnt = 0

    eps = 1e-12

    # -----------------------------
    # Trusted contracts by A
    # -----------------------------
    cs = contract_scores_A.copy()
    m = len(cs)
    k = max(1, int(round(trusted_ratio * m)))
    trusted = cs.head(k)["contract_id"].tolist()
    trusted_set = set(trusted)

    # normalize trusted contract scores
    c_map = dict(zip(cs["contract_id"], cs["cA"]))
    denom = sum(c_map.get(cid, 0.0) for cid in trusted) + eps
    for cid in trusted:
        c_map[cid] = c_map.get(cid, 0.0) / denom

    # -----------------------------
    # Compute N_ref from data (unique buyers per contract)
    #   N_ref = median_j N_j  (统计标尺，不是调参)
    # -----------------------------
    trade_df = df[df["tx_type"].astype(str) == "trade"]
    if len(trade_df) > 0:
        uniq_by_contract = trade_df.groupby("contract_id")["buyer"].nunique()
        N_ref = int(np.median(uniq_by_contract.values))
    else:
        N_ref = 1
    N_ref = max(1, N_ref)  # avoid zero

    # -----------------------------
    # Online trackers for adopter rank and unique buyers
    # -----------------------------
    buyers_seen = defaultdict(set)     # cid -> set(buyers seen so far)
    adopter_rank = defaultdict(dict)   # cid -> {buyer: rank}
    unique_count = defaultdict(int)    # cid -> current N_j(t)

    # -----------------------------
    # EWMA Cap_i (fairness normalization)
    # -----------------------------
    cap = defaultdict(float)
    last_t = {}
    user_score = defaultdict(float)

    df = df.sort_values(["block_number", "tx_index_in_block"]).reset_index(drop=True)

    for r in df.itertuples(index=False):
        if str(r.tx_type) != "trade":
            continue

        buyer = r.buyer
        cid = r.contract_id
        t = int(r.block_number)

        cost = float(r.price) + float(r.gas)

        # decay cap to time t (recursive EWMA)
        if buyer in last_t:
            dt = t - last_t[buyer]
            if dt > 0:
                cap[buyer] *= math.exp(-dt / float(tau_s))
        last_t[buyer] = t

        cap_before = cap[buyer]
        share = cost / (cost + cap_before + eps) if cost > 0 else 0.0
        f1 = share

        hold = int(r.hold_blocks)

        # -----------------------------
        # g_hold: contract-adaptive tau_h_j (use history, then update)
        # -----------------------------
        tau_h_j = get_tau_h_j(cid)
        if tau_h_j <= 0:
            tau_h_j = float(tau_h) if (tau_h is not None and tau_h > 0) else 1.0
        g_hold = 1.0 - math.exp(-hold / float(tau_h_j))

        # -----------------------------
        # g_early: Scheme A (ln version) using unique buyers + adopter rank
        #   N = unique buyers so far
        #   r = adopter rank (first adoption order)
        #   q = (r-1)/((N-1)+N_ref)
        #   chi = ln(1+N)
        #   g_early = 1 / (1 + ln(1 + q * chi))
        # -----------------------------
        if buyer not in buyers_seen[cid]:
            buyers_seen[cid].add(buyer)
            unique_count[cid] += 1
            adopter_rank[cid][buyer] = unique_count[cid]

        N = unique_count[cid]
        r_rank = adopter_rank[cid][buyer]

        # q = (r_rank - 1.0) / ((max(N - 1, 0)) + float(N_ref))   # in [0,1) and smoothed
        # chi = math.log1p(N)                                     # ln(1+N)
        # g_early = 1.0 / (1.0 + math.log1p(q * chi))             # ln(1 + q*chi)
        # D = (N-1)+N_ref 的平滑仍保留；但采用“前陡后缓”的归一化 log 方案
        g_early = g_early_log_norm(rank_r=r_rank, N_unique=N, N_ref=N_ref)


        # fuse
        f2 = (g_hold + g_early) / 2.0
        w = f1 * f2

        # accumulate score only on trusted contracts
        if cid in trusted_set:
            user_score[buyer] += w * c_map.get(cid, 0.0)

        # update cap with current spending (all spending matters for fairness)
        cap[buyer] += cost

        # update tau statistics AFTER using current tau (avoid leakage)
        update_tau_stats(cid, hold)

    scores = pd.DataFrame({"user": list(user_score.keys()), "score_B": list(user_score.values())})
    scores = scores.merge(users_meta[["user", "group"]], on="user", how="left")
    scores["group"] = scores["group"].fillna("unknown")
    scores = scores.sort_values("score_B", ascending=False).reset_index(drop=True)
    scores["rank_B"] = np.arange(1, len(scores) + 1)

    return scores, trusted


# -------------------------
# Evaluation
# -------------------------
def evaluate_B(scores_B: pd.DataFrame, users_meta: pd.DataFrame, topKs=(50, 100, 200)):
    full = users_meta[["user", "group"]].copy()
    full = full.merge(scores_B[["user", "score_B"]], on="user", how="left")
    full["score_B"] = full["score_B"].fillna(0.0)

    full = full.sort_values("score_B", ascending=False).reset_index(drop=True)
    full["rank_B"] = np.arange(1, len(full) + 1)

    # define "love vs others" (you can adjust)
    full["is_love"] = full["group"].isin(["whale_love", "retail_love"]).astype(int)

    auc_all = auc_manual(full["is_love"].values, full["score_B"].values)

    precision_at_k = {}
    composition_at_k = {}
    for K in topKs:
        top = full.head(K)
        precision_at_k[K] = float(top["is_love"].mean())
        composition_at_k[K] = top["group"].value_counts(normalize=True).to_dict()

    group_stats = full.groupby("group").agg(
        n=("user", "size"),
        mean_score=("score_B", "mean"),
        median_score=("score_B", "median"),
        mean_rank=("rank_B", "mean"),
        median_rank=("rank_B", "median"),
        share_top200=("rank_B", lambda r: float((r <= 200).mean())),
    ).sort_values("mean_rank")

    # whales-only AUC (often更能体现“热爱 vs 炒作”)
    whales = full[full["group"].isin(["whale_love", "whale_flip"])].copy()
    whales["y"] = whales["group"].eq("whale_love").astype(int)
    auc_whales = auc_manual(whales["y"].values, whales["score_B"].values)

    return {
        "auc_all": auc_all,
        "auc_whales": auc_whales,
        "precision_at_k": precision_at_k,
        "composition_at_k": composition_at_k,
        "group_stats": group_stats,
        "full_ranked": full,
    }

# -------------------------
# Output I/O helpers (NEW)
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def load_csv_if_exists(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return None


def save_list_txt(items, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(str(x) + "\n")

# -------------------------
# Main (REPLACE)
# -------------------------
if __name__ == "__main__":
    # -----------------------------
    # Inputs
    # -----------------------------
    data_dir = "../data/4/"  # 改成你的路径
    f_data = os.path.join(data_dir, "data3.csv")
    f_users = os.path.join(data_dir, "data3_users.csv")
    f_contracts = os.path.join(data_dir, "data3_contracts.csv")

    df = pd.read_csv(f_data)
    users_meta = pd.read_csv(f_users)
    contracts_meta = pd.read_csv(f_contracts)

    # enforce ordering fields
    if "tx_index_in_block" not in df.columns:
        df["tx_index_in_block"] = 0

    # B needs hold/age
    df = enrich_hold_and_age(df, contracts_meta)

    # -----------------------------
    # Outputs
    # -----------------------------
    out_dir = os.path.join("..", "output", "4")
    ensure_dir(out_dir)

    f_out_A = os.path.join(out_dir, "2_stageA_contract_scores.csv")
    f_out_B = os.path.join(out_dir, "2_stageB_user_scores.csv")
    f_out_B_trusted = os.path.join(out_dir, "2_stageB_trusted_contracts.txt")

    # -----------------------------
    # Stage A: compute OR load cache
    # -----------------------------
    contract_scores_A = load_csv_if_exists(f_out_A)
    if contract_scores_A is None:
        print(f"[Stage A] cache not found -> computing, then saving to: {f_out_A}")
        contract_scores_A = compute_A_contract_scores(
            df=df,
            contracts_meta=contracts_meta,
            alpha_A=0.7,
            beta_A=0.85,
            age_power=2.0,
        )
        # add rank for convenience
        contract_scores_A = contract_scores_A.copy()
        contract_scores_A["rank_A"] = np.arange(1, len(contract_scores_A) + 1)
        save_csv(contract_scores_A, f_out_A)
    else:
        print(f"[Stage A] cache found -> loading from: {f_out_A}")
        # 若历史文件没 rank_A，这里补上（不改变 cA）
        if "rank_A" not in contract_scores_A.columns:
            contract_scores_A = contract_scores_A.sort_values("cA", ascending=False).reset_index(drop=True)
            contract_scores_A["rank_A"] = np.arange(1, len(contract_scores_A) + 1)

    # -----------------------------
    # Stage B: compute (uses Stage A) and save
    # -----------------------------
    print(f"[Stage B] computing using Stage A results...")
    scores_B, trusted = compute_B_user_scores(
        df=df,
        users_meta=users_meta,
        contract_scores_A=contract_scores_A,
        trusted_ratio=0.1,   # 你可以改成 0.05 / 0.1
        tau_s=200_000,
        tau_h=50_000,
    )
    save_csv(scores_B, f_out_B)
    save_list_txt(trusted, f_out_B_trusted)

    # -----------------------------
    # Eval (optional) + print
    # -----------------------------
    report = evaluate_B(scores_B, users_meta, topKs=(50, 100, 200))

    print("\n" + "=" * 90)
    print(f"[B阶段] Trusted contracts (top by A): {trusted}")
    print("=" * 90)

    print(f"AUC (all users, love vs others):   {report['auc_all']:.4f}")
    print(f"AUC (whales only, love vs flip):   {report['auc_whales']:.4f}\n")

    print("Precision@K:")
    for K, v in report["precision_at_k"].items():
        print(f"  P@{K}: {v:.3f}")

    print("\nTopK composition:")
    for K, comp in report["composition_at_k"].items():
        top_items = sorted(comp.items(), key=lambda x: -x[1])[:5]
        print(f"  Top{K}: {top_items}")

    print("\nGroup stats (mean rank smaller is better):")
    print(report["group_stats"].to_string())

    print("\nTop-20 users by B score:")
    print(report["full_ranked"][["user", "group", "score_B", "rank_B"]].head(20).to_string(index=False))

    print("\n" + "-" * 90)
    print("[Saved]")
    print(f"  Stage A: {f_out_A}")
    print(f"  Stage B: {f_out_B}")
    print(f"  Stage B trusted list: {f_out_B_trusted}")
    print("-" * 90)