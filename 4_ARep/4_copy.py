import os
import numpy as np
import pandas as pd
from scipy import sparse

EPS = 1e-12
# ==========================
# 在 CONFIG 里新增这些（可选）
# ==========================
# 说明：
# - W_TRADE_COUNT_NPZ：不做衰减的“计数矩阵”（同 shape），若你能在 prepare_stageC_inputs 阶段额外导出就最好
# - TRADE_EVENTS_CSV：事件级明细（至少含 user, contract_id, block 或 timestamp），可选
# - 下面诊断会“有则用，无则跳过”
#
# CONFIG.update({
#     "W_TRADE_COUNT_NPZ": "3_stageC_W_trade_count.npz",     # optional
#     "TRADE_EVENTS_CSV": "3_stageC_trade_events.csv",       # optional
#     "DIAG_LAST_BLOCKS": 500_000,                           # optional: 最近多少 blocks 视为“近期”
#     "DIAG_TOPK_CONTRIB": 200,                              # optional
#     "OUT_RECENCY_REPORT_CSV": "4_stageC_recency_report.csv",
#     "OUT_U0_ASSOC_REPORT_CSV": "4_stageC_u0_sybil_assoc.csv",
# })


def _safe_load_npz(path: str):
    if path and os.path.exists(path):
        return sparse.load_npz(path).tocsr()
    return None


def build_S_matrix(W: sparse.csr_matrix, norm: str):
    """
    norm:
      - "random_walk": S = D_u^{-1} W
      - "symmetric"  : S = D_u^{-1/2} W D_c^{-1/2}
    """
    d_u = np.array(W.sum(axis=1)).flatten()
    d_c = np.array(W.sum(axis=0)).flatten()
    d_u[d_u <= 0] = 1.0
    d_c[d_c <= 0] = 1.0

    if norm == "random_walk":
        Su = sparse.diags(1.0 / d_u)
        S = Su @ W
        return S, S.T, d_u, d_c
    elif norm == "symmetric":
        Su = sparse.diags(1.0 / np.sqrt(d_u))
        Sc = sparse.diags(1.0 / np.sqrt(d_c))
        P_uc = Su @ W @ Sc
        P_uc_T = P_uc.T
        P_cu_T = P_uc
        return S, S.T, d_u, d_c
    else:
        raise ValueError(f"Unknown norm={norm}, expected 'random_walk' or 'symmetric'.")


def contract_basic_stats(W: sparse.csr_matrix, contracts: list):
    """
    输出每个合约列的：
      - d_c: 列和（总权重）
      - nnz_c: 非零边数量（连接了多少用户）
      - mean_w: d_c/nnz_c（代理：均值权重）
      - top_share_20: top20 贡献者占比（代理：集中度）
      - hhi_approx: 近似 HHI（用列向量归一化后平方和）
    """
    W_csc = W.tocsc()
    d_c = np.array(W.sum(axis=0)).flatten()
    nnz_c = np.diff(W_csc.indptr)

    mean_w = d_c / np.maximum(nnz_c, 1)

    # top20 share（对每列单独取 top20，计算占比）
    top_share_20 = np.zeros(len(contracts), dtype=float)
    hhi = np.zeros(len(contracts), dtype=float)

    for j in range(W_csc.shape[1]):
        col = W_csc.getcol(j)
        if col.nnz == 0 or d_c[j] <= 0:
            continue
        data = col.data.astype(float)
        # HHI
        p = data / (data.sum() + EPS)
        hhi[j] = float(np.sum(p * p))
        # top20 share
        if data.size <= 20:
            top_share_20[j] = float(data.sum() / (d_c[j] + EPS))
        else:
            top_share_20[j] = float(np.sort(data)[-20:].sum() / (d_c[j] + EPS))

    df = pd.DataFrame({
        "contract_id": contracts,
        "d_c": d_c,
        "nnz_c": nnz_c,
        "mean_w": mean_w,
        "top20_share": top_share_20,
        "hhi": hhi,
    }).sort_values("d_c", ascending=False).reset_index(drop=True)
    return df


def diagnose_sybil_recency(
    outdir: str,
    W_decay: sparse.csr_matrix,
    users: list,
    contracts: list,
    norm: str,
    sybil_prefix: str = "Sybil_",
):
    """
    诊断目标：
    1) 仅凭 W_decay 的代理指标：Sybil 列是否“均值更高 / 更集中”
    2) 若存在 W_count：计算 avg_decay_factor = sum(decayed)/sum(count)（更接近 1 => 越近期）
    """
    print("\n" + "="*90)
    print("[Diag-1] Sybil recency / decay intensity")
    print("="*90)

    # 代理指标
    df_stats = contract_basic_stats(W_decay, contracts)
    df_stats["is_sybil"] = df_stats["contract_id"].astype(str).str.startswith(sybil_prefix)

    # 汇总对比
    agg = df_stats.groupby("is_sybil")[["d_c","nnz_c","mean_w","top20_share","hhi"]].agg(["mean","median","max"])
    print("\n[Proxy] column stats summary (Sybil vs non-Sybil):")
    print(agg)

    # 打印 Sybil 与 Hype top若干
    df_sybil = df_stats[df_stats["is_sybil"]].copy()
    df_non = df_stats[~df_stats["is_sybil"]].copy()

    print("\nTop Sybil contracts by d_c:")
    print(df_sybil.head(10).to_string(index=False))
    print("\nTop non-Sybil contracts by d_c:")
    print(df_non.head(10).to_string(index=False))

    # 尝试加载 W_count（若存在）
    W_count = None
    if "W_TRADE_COUNT_NPZ" in CONFIG:
        W_count_path = os.path.join(outdir, CONFIG["W_TRADE_COUNT_NPZ"])
        W_count = _safe_load_npz(W_count_path)

    if W_count is not None:
        if W_count.shape != W_decay.shape:
            print(f"\n[Warn] W_count shape mismatch: {W_count.shape} vs W_decay {W_decay.shape}, skip exact decay factor.")
        else:
            count_c = np.array(W_count.sum(axis=0)).flatten()
            decay_c = np.array(W_decay.sum(axis=0)).flatten()
            avg_decay = decay_c / np.maximum(count_c, 1e-9)  # 每次交互的平均衰减权重（越接近1越近期）
            df_stats["count_c"] = count_c
            df_stats["avg_decay_factor"] = avg_decay
            print("\n[Exact] avg_decay_factor (=sum(decayed)/sum(count)) summary (Sybil vs non-Sybil):")
            agg2 = df_stats.groupby("is_sybil")[["avg_decay_factor","count_c"]].agg(["mean","median","max","min"])
            print(agg2)

    # 保存报告
    out_csv = os.path.join(outdir, CONFIG.get("OUT_RECENCY_REPORT_CSV", "4_stageC_recency_report.csv"))
    df_stats.to_csv(out_csv, index=False)
    print(f"\n[Saved] recency proxy report -> {out_csv}")

    return df_stats


def diagnose_u0_sybil_association(
    outdir: str,
    W: sparse.csr_matrix,
    users: list,
    contracts: list,
    u0: np.ndarray,
    norm: str,
    sybil_prefix: str = "Sybil_",
    topk_contrib: int = 200,
):
    """
    诊断目标：Sybil 合约是否与 StageB prior(u0) 强关联

    输出：
    - u0-weighted outflow to Sybil (在 u0 分布下，用户外流到 Sybil 的比例)
    - c_from_u0 = S^T u0 (仅由 u0 投影得到的合约得分)，Sybil 占比
    - Top-u0 用户与 Sybil top contributors 的重合（每个 Sybil 合约）
    """
    print("\n" + "="*90)
    print("[Diag-2] Association between Sybil contracts and Stage-B prior u0")
    print("="*90)

    u0 = np.asarray(u0, dtype=float).flatten()
    u0[u0 <= 0] = EPS
    u0 = u0 / (u0.sum() + EPS)

    # sybil columns
    sybil_cols = [j for j,c in enumerate(contracts) if str(c).startswith(sybil_prefix)]
    if len(sybil_cols) == 0:
        print("[Info] No Sybil contracts found by prefix, skip.")
        return None

    # build S (consistent with your Stage C normalization)
    S, ST, d_u, d_c = build_S_matrix(W, norm=norm)

    # (1) u0-weighted outflow to sybil
    # row_sybil_share_i = sum_{c in sybil} S[i,c]
    Sy = S[:, sybil_cols]                      # (n_u, n_sybil)
    row_sybil_share = np.array(Sy.sum(axis=1)).flatten()
    u0_sybil_outflow = float(np.dot(u0, row_sybil_share))  # probability mass to sybil in one step
    print(f"\n(1) u0-weighted 1-step outflow to Sybil = {u0_sybil_outflow:.6f}")

    # (2) c_from_u0 = S^T u0
    c_from_u0 = (ST @ u0.reshape(-1,1)).flatten()
    c_from_u0 = c_from_u0 / (c_from_u0.sum() + EPS)

    sybil_mass_from_u0 = float(c_from_u0[sybil_cols].sum())
    print(f"(2) Sybil mass in c_from_u0 (=S^T u0) = {sybil_mass_from_u0:.6f}")

    # show top contracts by c_from_u0
    df_cproj = pd.DataFrame({"contract_id": contracts, "c_from_u0": c_from_u0})
    df_cproj["is_sybil"] = df_cproj["contract_id"].astype(str).str.startswith(sybil_prefix)
    df_cproj = df_cproj.sort_values("c_from_u0", ascending=False).reset_index(drop=True)
    print("\nTop-20 contracts by c_from_u0:")
    print(df_cproj.head(20).to_string(index=False))

    # (3) overlap: top-u0 users vs sybil top contributors
    # define top-u0 set (e.g., top 1% users)
    n_u = len(users)
    top_pcts = [0.1, 0.5, 1.0, 5.0]  # %
    u0_rank = np.argsort(-u0)

    W_csc = W.tocsc()
    rows = np.array(users, dtype=object)

    records = []
    for pct in top_pcts:
        k = max(int(np.ceil(n_u * pct / 100.0)), 1)
        top_u0_idx = set(u0_rank[:k])

        for j in sybil_cols:
            col = W_csc.getcol(j)
            if col.nnz == 0:
                continue
            # top contributors by W[u,sybil]
            data = col.data
            idx = col.indices
            order = np.argsort(-data)
            idx_top = idx[order[:min(topk_contrib, len(order))]]
            set_top_contrib = set(idx_top.tolist())

            overlap = len(top_u0_idx.intersection(set_top_contrib))
            jacc = overlap / max(len(top_u0_idx.union(set_top_contrib)), 1)

            # u0 mass on these contributors
            u0_mass_on_contrib = float(u0[list(set_top_contrib)].sum())
            # u0-weighted inflow specifically to this sybil contract (using W or S?)
            # here we use S (consistent with ranking step):
            inflow = float((ST[j, :] @ u0.reshape(-1,1))[0,0])  # contract j receives from u0
            records.append({
                "pct_top_u0": pct,
                "top_u0_k": k,
                "sybil_contract": contracts[j],
                "sybil_inflow_from_u0": inflow,
                "u0_mass_on_sybil_top_contrib": u0_mass_on_contrib,
                "overlap_count": overlap,
                "jaccard": jacc,
                "sybil_top_contrib_size": len(set_top_contrib),
            })

    df_assoc = pd.DataFrame(records).sort_values(["pct_top_u0","sybil_inflow_from_u0"], ascending=[True, False])
    print("\n(3) Overlap between Top-u0 users and Sybil top contributors (by W):")
    print(df_assoc.to_string(index=False))

    out_csv = os.path.join(outdir, CONFIG.get("OUT_U0_ASSOC_REPORT_CSV", "4_stageC_u0_sybil_assoc.csv"))
    df_assoc.to_csv(out_csv, index=False)
    print(f"\n[Saved] u0-sybil association report -> {out_csv}")

    return df_assoc
# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "OUTDIR": "../output/4/",

    # Stage C 输入工件（来自 prepare_stageC_inputs.py 或你的 decay_full 版本）
    "W_TRADE_NPZ": "3_stageC_W_trade.npz",
    "W_USERS_CSV": "3_stageC_W_users.csv",
    "W_CONTRACTS_CSV": "3_stageC_W_contracts.csv",
    "U0_ALIGNED_NPY": "3_stageC_u0_aligned.npy",
    "WINDOW_META_CSV": "3_stageC_window_meta.csv",

    # ✅ 如果要做“标签/行为一致性检查”，需要原始 data
    "RUN_LABEL_CHECK": True,
    "DATA_CSV": "../data/4/data3.csv",  # <- 改成你的真实路径
    "DATA_CHUNKSIZE": 800_000,          # 数据大时用 chunk 读

    # （可选）Stage A 合约得分文件：用于对比 A vs C 的 TopK 重合
    "STAGEA_CONTRACT_SCORES_CSV": "2_stageA_contract_scores.csv",  # 需含 contract_id, cA
    "COMPARE_TOPK": 50,
    "W_TRADE_COUNT_NPZ": "3_stageC_W_trade_count.npz",     # optional
    "TRADE_EVENTS_CSV": "3_stageC_trade_events.csv",       # optional
    "DIAG_LAST_BLOCKS": 500_000,                           # optional: 最近多少 blocks 视为“近期”
    "DIAG_TOPK_CONTRIB": 200,                              # optional
    "OUT_RECENCY_REPORT_CSV": "4_stageC_recency_report.csv",
    "OUT_U0_ASSOC_REPORT_CSV": "4_stageC_u0_sybil_assoc.csv",



    # ========================================================
    # ✅ Stage C BiRank 参数
    # 你要求：alpha = 1；beta 扫描
    # ========================================================
    "ALPHA_C": 0.85,
    "BETA_LIST": [0.7, 0.6, 0.5, 0.4, 0.1],
    "PRIMARY_BETA": 0.7,
    "MAX_ITER": 80,
    "TOL": 1e-10,
    "CONTRACT_RELIABILITY_ANCHOR_RATIO": 0.03,

    # ✅ 归一化方式
    # - "symmetric": S = D_u^{-1/2} W D_c^{-1/2}
    # - "random_walk": S = D_u^{-1} W   （用户侧行随机游走）
    "NORM": "random_walk",

    # 输出文件名（PRIMARY_BETA 会写这两个）
    "OUT_USER_SCORES_CSV": "4_stageC_user_scores.csv",
    "OUT_CONTRACT_SCORES_CSV": "4_stageC_contract_scores.csv",

    # 展示 TopN
    "SHOW_TOP_USERS": 20,
    "SHOW_TOP_CONTRACTS": 20,

    # ========================================================
    # ✅ “攻击者/女巫”识别与合约标签检查规则
    # ========================================================
    "ATTACKER_USER_PREFIXES": ["attacker_u_", "attacker_"],  # 你模拟数据的 attacker 前缀
    "SYBIL_CONTRACT_KEYWORD": "Sybil",

    # 行为判定阈值（用于“行为像 Sybil”的合约识别）
    "LABEL_CHECK_MIN_TX": 200,              # 交易太少不判定
    "ATTACKER_TX_RATIO_TH": 0.50,           # 攻击者参与交易占比阈值
    "TOP10_TRADER_TX_SHARE_TH": 0.60,       # Top10 交易者集中度阈值（越高越像刷）
    "REPORT_MISMATCH_TOPN": 30,
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _is_attacker(user: str) -> bool:
    u = str(user)
    for p in CONFIG["ATTACKER_USER_PREFIXES"]:
        if u.startswith(p):
            return True
    return False

def _user_group(user: str) -> str:
    # 你日志里 attacker_u_* 被标成 unknown，说明原来的 group 识别没生效；
    # 这里强制用前缀修复。
    return "attacker" if _is_attacker(user) else "non_attacker"

def _output_paths(outdir: str, beta: float):
    base_u = os.path.join(outdir, CONFIG["OUT_USER_SCORES_CSV"])
    base_c = os.path.join(outdir, CONFIG["OUT_CONTRACT_SCORES_CSV"])
    suf_u = os.path.join(outdir, f"4_stageC_user_scores_beta{beta:.2f}.csv")
    suf_c = os.path.join(outdir, f"4_stageC_contract_scores_beta{beta:.2f}.csv")
    return base_u, base_c, suf_u, suf_c

# ------------------------------------------------------------
# (1) Load artifacts
# ------------------------------------------------------------
def load_stageC_inputs(outdir: str):
    w_npz = os.path.join(outdir, CONFIG["W_TRADE_NPZ"])
    u_csv = os.path.join(outdir, CONFIG["W_USERS_CSV"])
    c_csv = os.path.join(outdir, CONFIG["W_CONTRACTS_CSV"])
    u0_npy = os.path.join(outdir, CONFIG["U0_ALIGNED_NPY"])
    meta_csv = os.path.join(outdir, CONFIG["WINDOW_META_CSV"])

    for p in [w_npz, u_csv, c_csv, u0_npy]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少 Stage C 输入工件: {p}")

    W = sparse.load_npz(w_npz).tocsr()
    users = pd.read_csv(u_csv)["user"].astype(str).tolist()
    contracts = pd.read_csv(c_csv)["contract_id"].astype(str).tolist()
    u0 = np.load(u0_npy).astype(float)

    meta = None
    if os.path.exists(meta_csv):
        meta = pd.read_csv(meta_csv).iloc[0].to_dict()

    if W.shape != (len(users), len(contracts)):
        raise ValueError(f"W 维度与 users/contracts 不一致: W={W.shape}, users={len(users)}, contracts={len(contracts)}")

    if len(u0) != len(users):
        raise ValueError(f"u0 长度与 users 不一致: u0={len(u0)}, users={len(users)}")

    u0[u0 <= 0] = EPS
    u0 = u0 / (u0.sum() + EPS)

    return W, users, contracts, u0, meta


def build_contract_reliability(contracts: list[str], d_c_raw: np.ndarray, outdir: str) -> np.ndarray:
    stagea_name = str(CONFIG.get("STAGEA_CONTRACT_SCORES_CSV", "2_stageA_contract_scores.csv"))
    stagea_path = os.path.join(outdir, stagea_name)
    d_c_raw = np.asarray(d_c_raw, dtype=float).reshape(-1)
    if len(contracts) == 0:
        return np.ones(0, dtype=float)

    gas_map = {str(cid): float(val) for cid, val in zip(contracts, d_c_raw)}

    def _fallback() -> np.ndarray:
        max_gas = float(np.max(d_c_raw)) if len(d_c_raw) else 0.0
        if max_gas <= 0:
            return np.zeros(len(contracts), dtype=float)
        return np.clip(d_c_raw / max_gas, 0.0, 1.0)

    if not os.path.exists(stagea_path):
        return _fallback()

    df_a = pd.read_csv(stagea_path)
    if "contract_id" not in df_a.columns:
        return _fallback()

    if "is_anchor" in df_a.columns:
        hard_contracts = df_a.loc[
            pd.to_numeric(df_a["is_anchor"], errors="coerce").fillna(0).astype(int).eq(1),
            "contract_id",
        ].astype(str).tolist()
    else:
        top_ratio = float(CONFIG.get("CONTRACT_RELIABILITY_ANCHOR_RATIO", 0.03))
        topk = max(1, int(np.ceil(len(df_a) * top_ratio)))
        hard_contracts = df_a.head(topk)["contract_id"].astype(str).tolist()

    hard_gas = [gas_map[cid] for cid in hard_contracts if cid in gas_map]
    median_hard_gas = float(np.median(hard_gas)) if hard_gas else 0.0
    if median_hard_gas <= 0:
        return _fallback()

    rel = np.array(
        [min(1.0, float(gas_map.get(str(cid), 0.0)) / (median_hard_gas + EPS)) for cid in contracts],
        dtype=float,
    )
    return np.clip(rel, 0.0, 1.0)

# ------------------------------------------------------------
# (2) Build normalized operator S
# ------------------------------------------------------------
def build_S(W: sparse.csr_matrix, norm: str):
    n_u, n_c = W.shape
    d_u = np.array(W.sum(axis=1)).flatten()
    d_c = np.array(W.sum(axis=0)).flatten()
    d_u[d_u <= 0] = 1.0
    d_c[d_c <= 0] = 1.0

    if norm == "symmetric":
        Su = sparse.diags(1.0 / np.sqrt(d_u))
        Sc = sparse.diags(1.0 / np.sqrt(d_c))
        P_uc = Su @ W @ Sc
        P_uc_T = P_uc.T
        P_cu_T = P_uc
    elif norm == "random_walk":
        # 用户侧行随机游走：每个用户的总“外出贡献”归一为 1
        Su = sparse.diags(1.0 / d_u)
        Sc_inv = sparse.diags(1.0 / d_c)
        P_uc = Su @ W
        P_uc_T = P_uc.T
        P_cu_T = W @ Sc_inv
    else:
        raise ValueError(f"Unknown NORM={norm}, use symmetric|random_walk")

    return P_uc, P_uc_T, P_cu_T, d_u, d_c

# ------------------------------------------------------------
# (3) Personalized BiRank
# ------------------------------------------------------------
def birank_personalized(W, u0, alpha, beta, norm="random_walk", c0=None, max_iter=80, tol=1e-10, contracts=None, outdir=None, contract_reliability=None):
    n_u, n_c = W.shape
    P_uc, P_uc_T, P_cu_T, _, _ = build_S(W, norm=norm)

    u0 = u0.reshape(-1, 1)
    if c0 is None:
        c0 = np.ones(n_c, dtype=float) / float(n_c)
    c0 = np.asarray(c0, dtype=float)
    c0[c0 <= 0] = EPS
    c0 = (c0 / (c0.sum() + EPS)).reshape(-1, 1)

    if contract_reliability is None:
        if contracts is not None and outdir is not None:
            d_c_raw = np.array(W.sum(axis=0)).flatten()
            contract_reliability = build_contract_reliability(list(contracts), d_c_raw, str(outdir))
        else:
            contract_reliability = np.ones(n_c, dtype=float)
    contract_reliability = np.asarray(contract_reliability, dtype=float).reshape(-1, 1)
    contract_reliability = np.clip(contract_reliability, 0.0, 1.0)
    if contract_reliability.shape[0] != n_c:
        raise ValueError(f"contract_reliability length mismatch: expected {n_c}, got {contract_reliability.shape[0]}")

    u = u0.copy()
    c = c0.copy()

    for _ in range(max_iter):
        # alpha=1 => 不注入合约先验
        c_new = beta * (P_uc_T @ u) + (1.0 - beta) * c0
        u_new = alpha * (P_cu_T @ (c_new * contract_reliability)) + (1.0 - alpha) * u0

        c_new = c_new / (c_new.sum() + EPS)
        u_new = u_new / (u_new.sum() + EPS)

        if (np.max(np.abs(c_new - c)) < tol) and (np.max(np.abs(u_new - u)) < tol):
            c, u = c_new, u_new
            break
        c, u = c_new, u_new

    return u.flatten(), c.flatten()

# ------------------------------------------------------------
# (4) Diagnostics: why Sybil dominates?
# ------------------------------------------------------------
def diagnostics(W, users, contracts, u0, uC, cC, d_c_raw, beta, norm):
    print("\n" + "=" * 90)
    print(f"[Diagnostics] A/B/C  (beta={beta:.2f}, normalization={norm})")
    print("=" * 90)

    # A: raw column sum
    df_dc = pd.DataFrame({"contract_id": contracts, "d_c": d_c_raw})
    sybil_kw = CONFIG["SYBIL_CONTRACT_KEYWORD"]
    df_sybil = df_dc[df_dc["contract_id"].str.contains(sybil_kw, na=False)].sort_values("d_c", ascending=False)
    df_hype = df_dc[df_dc["contract_id"].str.contains("Hype", na=False)].sort_values("d_c", ascending=False)

    print("\n[A] Column weight sum d_c = sum_u W[u,c]  (raw W, before S-normalization)")
    if len(df_sybil) > 0:
        print(f"    Sybil count={len(df_sybil)} | top {min(5, len(df_sybil))} by d_c:")
        for r in df_sybil.head(5).itertuples(index=False):
            print(f"      {r.contract_id:<12} d_c={r.d_c:.6g}")
    if len(df_hype) > 0:
        print(f"    Hype count={len(df_hype)} | top {min(10, len(df_hype))} by d_c:")
        for r in df_hype.head(10).itertuples(index=False):
            print(f"      {r.contract_id:<12} d_c={r.d_c:.6g}")

    # B: top contributors to sybil contracts
    print("\n[B] Top-20 contributors to each Sybil contract (by W[u,sybil])")
    for cid in df_sybil.head(5)["contract_id"].tolist():
        j = contracts.index(cid)
        col = W.getcol(j).tocoo()
        if col.nnz == 0:
            continue
        idx = np.argsort(-col.data)[:20]
        top_rows = col.row[idx]
        top_vals = col.data[idx]
        tmp = pd.DataFrame({
            "user": [users[i] for i in top_rows],
            "group": [_user_group(users[i]) for i in top_rows],
            "W_u_sybil": top_vals
        })
        print(f"\n    Contract: {cid} | nnz={col.nnz} | top contributors:")
        print(tmp.to_string(index=False))
        comp = tmp["group"].value_counts().to_dict()
        print(f"    group composition among top contributors: {comp}")

    # C: prior drift
    u0v = u0.flatten()
    uCv = uC.flatten()
    # Pearson
    u0m = u0v - u0v.mean()
    uCm = uCv - uCv.mean()
    denom = (np.sqrt((u0m*u0m).sum()) * np.sqrt((uCm*uCm).sum()) + EPS)
    pearson = float((u0m*uCm).sum() / denom)

    def _kl(p, q):
        p = np.clip(p, EPS, 1.0)
        q = np.clip(q, EPS, 1.0)
        return float(np.sum(p * np.log(p/q)))

    kl_u0_uC = _kl(u0v, uCv)
    kl_uC_u0 = _kl(uCv, u0v)
    m = 0.5*(u0v+uCv)
    js = 0.5*_kl(u0v, m) + 0.5*_kl(uCv, m)

    print("\n[C] Prior drift: similarity between u0 and uC")
    print(f"    Pearson corr(u0,uC) = {pearson:.6f}")
    print(f"    KL(u0||uC) = {kl_u0_uC:.6f}")
    print(f"    KL(uC||u0) = {kl_uC_u0:.6f}")
    print(f"    JS(u0,uC)  = {js:.6f}")

    # group mass (only attacker vs non_attacker)
    dfu = pd.DataFrame({"user": users, "u0": u0v, "uC": uCv})
    dfu["group"] = dfu["user"].apply(_user_group)
    gm = dfu.groupby("group")[["u0", "uC"]].sum().rename(columns={"u0": "mass_u0", "uC": "mass_uC"}).sort_values("mass_uC", ascending=False)
    print("\n    Group mass (sum of probabilities) sorted by mass_uC:")
    print(gm.to_string())

# ------------------------------------------------------------
# (5) Label check on raw data: label vs behavior
# ------------------------------------------------------------
def label_check_from_data(data_csv: str, contracts_in_W: set):
    if not os.path.exists(data_csv):
        print(f"\n[Warn] DATA_CSV not found: {data_csv}, skip label check.")
        return None

    min_tx = int(CONFIG["LABEL_CHECK_MIN_TX"])
    atk_th = float(CONFIG["ATTACKER_TX_RATIO_TH"])
    top10_th = float(CONFIG["TOP10_TRADER_TX_SHARE_TH"])
    sybil_kw = CONFIG["SYBIL_CONTRACT_KEYWORD"]

    # accumulators
    tx_count = {}
    atk_tx_count = {}
    min_block = {}
    max_block = {}

    # trader count per contract (approx) + top10 concentration needs per-(contract,trader) counts
    trader_counts = {}  # (contract, trader) -> n

    def _upd_block(cid, b):
        if cid not in min_block:
            min_block[cid] = b
            max_block[cid] = b
        else:
            if b < min_block[cid]:
                min_block[cid] = b
            if b > max_block[cid]:
                max_block[cid] = b

    # chunked reading
    chunksize = int(CONFIG["DATA_CHUNKSIZE"])
    usecols = ["buyer", "seller", "contract_id", "block_number"]
    for chunk in pd.read_csv(data_csv, usecols=usecols, chunksize=chunksize):
        chunk["contract_id"] = chunk["contract_id"].astype(str)
        # 只统计出现在 W 的合约（减少成本）
        chunk = chunk[chunk["contract_id"].isin(contracts_in_W)]
        if len(chunk) == 0:
            continue

        chunk["buyer"] = chunk["buyer"].astype(str)
        chunk["seller"] = chunk["seller"].astype(str)
        chunk["block_number"] = pd.to_numeric(chunk["block_number"], errors="coerce").fillna(0).astype(int)

        # tx-level
        for r in chunk.itertuples(index=False):
            cid = r.contract_id
            tx_count[cid] = tx_count.get(cid, 0) + 1
            _upd_block(cid, int(r.block_number))

            is_atk = _is_attacker(r.buyer) or _is_attacker(r.seller)
            if is_atk:
                atk_tx_count[cid] = atk_tx_count.get(cid, 0) + 1

            # traders (count-based, buyer+seller)
            keyb = (cid, r.buyer)
            keys = (cid, r.seller)
            trader_counts[keyb] = trader_counts.get(keyb, 0) + 1
            trader_counts[keys] = trader_counts.get(keys, 0) + 1

    # build stats df
    rows = []
    for cid, n in tx_count.items():
        atk_n = atk_tx_count.get(cid, 0)
        atk_ratio = atk_n / max(n, 1)
        span = int(max_block[cid] - min_block[cid]) if cid in min_block else 0

        # top10 trader share
        # collect counts for this contract
        arr = [v for (c, _t), v in trader_counts.items() if c == cid]
        total = float(sum(arr)) if len(arr) else 0.0
        if total <= 0:
            top10_share = 0.0
            uniq_traders = 0
        else:
            uniq_traders = len(arr)
            arr.sort(reverse=True)
            top10_share = float(sum(arr[:10]) / total)

        label_sybil = 1 if sybil_kw in cid else 0
        behavior_sybil_like = 1 if (n >= min_tx and (atk_ratio >= atk_th or top10_share >= top10_th)) else 0
        mismatch = 1 if (label_sybil != behavior_sybil_like) else 0

        rows.append({
            "contract_id": cid,
            "tx_count": n,
            "span_blocks": span,
            "atk_tx_ratio": atk_ratio,
            "top10_trader_tx_share": top10_share,
            "uniq_trader_count": uniq_traders,
            "label_sybil": label_sybil,
            "behavior_sybil_like": behavior_sybil_like,
            "label_mismatch": mismatch
        })

    df = pd.DataFrame(rows).sort_values(["label_mismatch", "tx_count"], ascending=[False, False]).reset_index(drop=True)
    return df

# ------------------------------------------------------------
# (6) Save + show
# ------------------------------------------------------------
def save_and_show(outdir: str, users, contracts, uC, cC, beta: float, meta=None):
    os.makedirs(outdir, exist_ok=True)

    df_u = pd.DataFrame({"user": users, "uC": uC}).sort_values("uC", ascending=False).reset_index(drop=True)
    df_u["rank_C_user"] = np.arange(1, len(df_u) + 1)

    df_c = pd.DataFrame({"contract_id": contracts, "cC": cC}).sort_values("cC", ascending=False).reset_index(drop=True)
    df_c["rank_C_contract"] = np.arange(1, len(df_c) + 1)

    base_u, base_c, suf_u, suf_c = _output_paths(outdir, beta)

    df_u.to_csv(suf_u, index=False)
    df_c.to_csv(suf_c, index=False)

    if abs(beta - float(CONFIG["PRIMARY_BETA"])) < 1e-12:
        df_u.to_csv(base_u, index=False)
        df_c.to_csv(base_c, index=False)

    print("\n" + "=" * 90)
    print(f"[Stage C] Personalized BiRank finished ✅  (alpha={CONFIG['ALPHA_C']:.2f}, beta={beta:.2f}, norm={CONFIG['NORM']})")
    print("=" * 90)
    if meta is not None:
        # meta 字段可能来自“全生命周期 + 衰减”，不强行解释为“截断窗口”
        meta_show = " | ".join([f"{k}: {meta.get(k)}" for k in ["window_blocks","start_block","end_block","nnz","n_rows","n_cols","mode"] if k in meta])
        if meta_show:
            print(meta_show)

    print("\nTop contracts by Stage C:")
    print(df_c.head(CONFIG["SHOW_TOP_CONTRACTS"]).to_string(index=False))

    print("\nTop users by Stage C:")
    print(df_u.head(CONFIG["SHOW_TOP_USERS"]).to_string(index=False))

    print(f"\n[Saved] contracts -> {suf_c}")
    print(f"[Saved] users     -> {suf_u}")
    if abs(beta - float(CONFIG["PRIMARY_BETA"])) < 1e-12:
        print(f"[Saved:PRIMARY] contracts -> {base_c}")
        print(f"[Saved:PRIMARY] users     -> {base_u}")

    return df_u, df_c

# ------------------------------------------------------------
# (7) Compare StageA vs StageC contract topK overlap (optional)
# ------------------------------------------------------------
def compare_stageA_stageC(outdir: str, df_c_stageC: pd.DataFrame, topk: int = 50):
    fA = os.path.join(outdir, CONFIG["STAGEA_CONTRACT_SCORES_CSV"])
    if not os.path.exists(fA):
        print("\n[Warn] 未发现 Stage A 合约得分文件，跳过 A vs C 对比。")
        print(f"       expected: {fA}")
        return

    dfA = pd.read_csv(fA)
    if "contract_id" not in dfA.columns:
        print("\n[Warn] Stage A 文件缺少 contract_id 列，跳过对比。")
        return

    if "cA" in dfA.columns:
        topA = dfA.sort_values("cA", ascending=False).head(topk)["contract_id"].astype(str).tolist()
    else:
        # 退化：取第二列当 score
        score_col = [c for c in dfA.columns if c != "contract_id"][0]
        topA = dfA.sort_values(score_col, ascending=False).head(topk)["contract_id"].astype(str).tolist()

    topC = df_c_stageC.head(topk)["contract_id"].astype(str).tolist()

    setA, setC = set(topA), set(topC)
    inter = setA.intersection(setC)

    jaccard = len(inter) / max(len(setA.union(setC)), 1)
    recallA = len(inter) / max(len(setA), 1)
    recallC = len(inter) / max(len(setC), 1)

    print("\n" + "-" * 90)
    print(f"[A vs C contract overlap @Top{topk}]")
    print(f"  |A∩C| = {len(inter)}")
    print(f"  Jaccard(A,C) = {jaccard:.4f}")
    print(f"  Recall(A->C) = {recallA:.4f}")
    print(f"  Recall(C->A) = {recallC:.4f}")
    print(f"  Intersection samples: {list(sorted(inter))[:10]}")
    print("-" * 90)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    outdir = CONFIG["OUTDIR"]

    W_trade, users_W, contracts_W, u0_aligned, meta = load_stageC_inputs(outdir)

    # raw degrees for diagnostics
    d_c_raw = np.array(W_trade.sum(axis=0)).flatten()

    # optional: label check from raw data
    if CONFIG.get("RUN_LABEL_CHECK", False):
        df_lc = label_check_from_data(CONFIG["DATA_CSV"], contracts_in_W=set(contracts_W))
        if df_lc is not None:
            f_lc = os.path.join(outdir, "4_stageC_label_check_report.csv")
            df_lc.to_csv(f_lc, index=False)
            print("\n" + "=" * 90)
            print("[LabelCheck] label(Sybil name) vs behavior(attacker ratio / concentration)")
            print("=" * 90)
            mism = df_lc[df_lc["label_mismatch"] == 1].head(CONFIG["REPORT_MISMATCH_TOPN"])
            if len(mism) == 0:
                print("No mismatches under current thresholds.")
            else:
                print(mism[[
                    "contract_id","label_sybil","behavior_sybil_like","tx_count",
                    "span_blocks","atk_tx_ratio","top10_trader_tx_share","uniq_trader_count"
                ]].to_string(index=False))
            print(f"\n[Saved] label check report -> {f_lc}")

    alpha = float(CONFIG["ALPHA_C"])
    beta_list = CONFIG.get("BETA_LIST", [float(CONFIG.get("BETA_C", 0.85))])
    norm = str(CONFIG.get("NORM", "random_walk"))
    norm_used = CONFIG.get("NORM", "random_walk")  # 若你已有 norm 参数来源，改成你实际变量
    diagnose_sybil_recency(outdir, W_trade, users_W, contracts_W, norm=norm_used, sybil_prefix="Sybil_")
    diagnose_u0_sybil_association(outdir, W_trade, users_W, contracts_W, u0_aligned, norm=norm_used, sybil_prefix="Sybil_", topk_contrib=CONFIG.get("DIAG_TOPK_CONTRIB", 200))
   

    for beta in beta_list:
        beta = float(beta)

        uC, cC = birank_personalized(
            W=W_trade,
            u0=u0_aligned,
            alpha=alpha,
            beta=beta,
            norm=norm,
            c0=None,
            max_iter=int(CONFIG["MAX_ITER"]),
            tol=float(CONFIG["TOL"]),
        )

        df_uC, df_cC = save_and_show(outdir, users_W, contracts_W, uC, cC, beta=beta, meta=meta)

        diagnostics(W_trade, users_W, contracts_W, u0_aligned, uC, cC, d_c_raw, beta=beta, norm=norm)

        compare_stageA_stageC(outdir, df_cC, topk=int(CONFIG["COMPARE_TOPK"]))
        # ==========================
        # MAIN 里加两行调用即可
        # ==========================
        # 在你 load 完 W_trade, users_W, contracts_W, u0_aligned, meta 之后，
        # 在 run birank 之前插入：
        #
        # norm_used = CONFIG.get("NORM", "random_walk")  # 若你已有 norm 参数来源，改成你实际变量
        # diagnose_sybil_recency(outdir, W_trade, users_W, contracts_W, norm=norm_used, sybil_prefix="Sybil_")
        # diagnose_u0_sybil_association(outdir, W_trade, users_W, contracts_W, u0_aligned, norm=norm_used, sybil_prefix="Sybil_", topk_contrib=CONFIG.get("DIAG_TOPK_CONTRIB", 200))
        import pandas as pd

        df = pd.read_csv("../output/4/4_stageC_contract_scores.csv")
        syb = df[df["contract_id"].str.startswith("Sybil")].sort_values("rank_C_contract")
        print(syb[["contract_id","cC","rank_C_contract"]])
