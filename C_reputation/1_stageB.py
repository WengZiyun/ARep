#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage B: Credit Attribution (paper-aligned implementation)

Given Stage A contract scores cA and raw trading events, this script builds an
exogenous user prior p0(u) and normalized u0 for Stage C.

Core equations implemented:
  w_e = f1(e) * f2(e)
  f1(e) = cost_e / (cost_e + Cap_u(t_e) + eps)
  Cap_u(t_e) = sum_{k<t_e} cost_k * exp(-(t_e-t_k)/tau_s)
  g_hold(e) = 1 - exp(-Delta_t_e / tau_h)
  g_early(e)= 1 - ln(1+q(r))/ln(1+q_max)
    q(r)   = (r-1)/((N-1)+N_ref)
    q_max  = (N-1)/((N-1)+N_ref)
  f2(e) = 0.5 * (g_early + g_hold)
  p0(u) = sum_{e: u(e)=u, c(e) in C_hard} w_e * cA_tilde(c(e))
  u0    = Normalize(p0)

Outputs (all prefixed with 1_):
  - output/C/1_stageB_anchor_contracts.csv
  - output/C/1_stageB_event_scores.csv
  - output/C/1_stageB_user_prior.csv
  - output/C/1_stageB_u0_full.csv
"""

from pathlib import Path
from collections import defaultdict
import math

import numpy as np
import pandas as pd


# ============================================================
# 0) Defaults (edit here; no argparse)
# ============================================================
INPUT_DIR = Path("output/B/2_tranding")
STAGEA_CONTRACT_CSV = Path("output/C/0_stageA_contract_scores.csv")
STAGEA_USER_CSV = Path("output/C/0_stageA_user_scores.csv")

OUTPUT_ANCHORS_CSV = Path("output/C/1_stageB_anchor_contracts.csv")
OUTPUT_EVENT_SCORES_CSV = Path("output/C/1_stageB_event_scores.csv")
OUTPUT_USER_PRIOR_CSV = Path("output/C/1_stageB_user_prior.csv")
OUTPUT_U0_FULL_CSV = Path("output/C/1_stageB_u0_full.csv")

# Anchor set selection: C_hard from Stage A ranking
HARD_TOP_RATIO = 0.10  # used when HARD_TOP_K is None
HARD_TOP_K = None      # set int to override ratio

# Stage B hyperparameters
TAU_S = 200_000.0
TAU_H = 50_000.0
N_REF = None           # None => median unique owners across anchors
EPS = 1e-12

# Require irreversible user-initiated cost (exclude unsolicited airdrop-like)
REQUIRE_POSITIVE_COST = True


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
        sub["tx_type"] = sub["tx_type"].astype(str).str.lower().str.strip().replace({"sale": "trade"})
        sub = sub[sub["tx_type"].isin(["mint", "trade"])].copy()

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

    # deterministic ordering
    out = out.sort_values(["block_number", "tx_index_in_block"], kind="mergesort").reset_index(drop=True)
    return out


# ============================================================
# 2) Anchor contracts from Stage A
# ============================================================
def load_anchor_set(stagea_contract_csv: Path, hard_top_ratio: float, hard_top_k: int | None) -> pd.DataFrame:
    if not stagea_contract_csv.exists():
        raise FileNotFoundError(f"Stage A contract scores not found: {stagea_contract_csv.resolve()}")

    cs = pd.read_csv(stagea_contract_csv)
    need = {"contract_id", "cA", "rank_A"}
    miss = [c for c in need if c not in cs.columns]
    if miss:
        raise ValueError(f"Missing columns in {stagea_contract_csv.name}: {miss}")

    cs = cs.sort_values(["rank_A", "contract_id"], ascending=[True, True]).reset_index(drop=True)
    m = len(cs)
    if m == 0:
        raise ValueError("Stage A score table is empty.")

    if hard_top_k is None:
        if not (0 < hard_top_ratio <= 1):
            raise ValueError("HARD_TOP_RATIO must be in (0,1].")
        k = max(1, int(round(m * hard_top_ratio)))
    else:
        k = max(1, min(int(hard_top_k), m))

    anchors = cs.head(k).copy()
    denom = float(anchors["cA"].sum())
    if denom <= 0:
        anchors["cA_tilde"] = 1.0 / len(anchors)
    else:
        anchors["cA_tilde"] = anchors["cA"] / denom
    anchors["is_anchor"] = 1
    return anchors


# ============================================================
# 3) Build acquisition events with holding duration and adopter rank stats
# ============================================================
def build_stageb_events(df: pd.DataFrame, anchor_set: set[str]) -> pd.DataFrame:
    # acquisition events on anchors only: buyer acquires ownership via mint/trade
    ev = df[df["contract_id"].isin(anchor_set)].copy()
    if ev.empty:
        return ev

    ev["event_id"] = np.arange(len(ev), dtype=np.int64)
    ev["cost"] = (ev["price"] + ev["gas"]).clip(lower=0.0)

    # relinquish block = next acquisition on same (contract, token), else tau_end
    tau_end = int(df["block_number"].max())
    ev["next_block"] = ev.groupby(["contract_id", "token_id"], sort=False)["block_number"].shift(-1)
    ev["tau_rel"] = ev["next_block"].fillna(tau_end).astype(np.int64)
    ev["delta_t"] = (ev["tau_rel"] - ev["block_number"]).clip(lower=0).astype(np.int64)

    # N before event and adopter rank r on contract by first acquisition order
    # N_prev: number of unique buyers before current event (online O(n))
    seen_buyers_by_contract = defaultdict(set)
    n_prev = np.zeros(len(ev), dtype=np.int64)
    for i, r in enumerate(ev.itertuples(index=False)):
        cid = r.contract_id
        b = r.buyer
        n_prev[i] = len(seen_buyers_by_contract[cid])
        seen_buyers_by_contract[cid].add(b)
    ev["N_prev"] = n_prev

    first_buy_block = (
        ev.groupby(["contract_id", "buyer"], as_index=False)["block_number"]
        .min()
        .rename(columns={"block_number": "first_buy_block"})
    )
    first_buy_block = first_buy_block.sort_values(["contract_id", "first_buy_block", "buyer"], kind="mergesort")
    first_buy_block["r_rank"] = first_buy_block.groupby("contract_id", sort=False).cumcount() + 1

    ev = ev.merge(first_buy_block[["contract_id", "buyer", "r_rank"]], on=["contract_id", "buyer"], how="left")
    ev["r_rank"] = ev["r_rank"].fillna(1).astype(np.int64)

    # N_current used in q_max formula from paper: N = N_prev + 1
    ev["N_current"] = (ev["N_prev"] + 1).clip(lower=1).astype(np.int64)
    return ev


# ============================================================
# 4) Factor computation and p0 aggregation
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


def compute_stageb_user_prior(
    ev: pd.DataFrame,
    cA_tilde_map: dict[str, float],
    tau_s: float,
    tau_h: float,
    n_ref: int | None,
    eps: float,
    require_positive_cost: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ev.empty:
        empty_users = pd.DataFrame(columns=["user", "p0", "u0", "rank_B", "event_cnt_anchor"])
        empty_ev = pd.DataFrame(columns=[
            "event_id", "user", "contract_id", "tx_type", "block_number", "tau_rel", "delta_t",
            "price", "gas", "cost", "Cap_u", "f1", "g_hold", "g_early", "f2", "w_e", "cA_tilde", "credit"
        ])
        return empty_users, empty_ev

    if n_ref is None:
        # paper's size-adaptive smoothing scale baseline from market size statistics
        n_ref_val = int(np.median(ev.groupby("contract_id")["buyer"].nunique().values))
        n_ref_val = max(1, n_ref_val)
    else:
        n_ref_val = max(1, int(n_ref))

    ev = ev.sort_values(["block_number", "tx_index_in_block", "event_id"], kind="mergesort").reset_index(drop=True)

    cap = defaultdict(float)
    last_t = {}
    user_p0 = defaultdict(float)
    user_event_cnt = defaultdict(int)

    cap_before = np.zeros(len(ev), dtype=float)
    f1_arr = np.zeros(len(ev), dtype=float)
    ghold_arr = np.zeros(len(ev), dtype=float)
    gearly_arr = np.zeros(len(ev), dtype=float)
    f2_arr = np.zeros(len(ev), dtype=float)
    w_arr = np.zeros(len(ev), dtype=float)
    credit_arr = np.zeros(len(ev), dtype=float)
    ctilde_arr = np.zeros(len(ev), dtype=float)

    for i, r in enumerate(ev.itertuples(index=False)):
        user = r.buyer
        t = int(r.block_number)
        cost = float(r.cost)

        # EWMA capacity decay to current event block
        if user in last_t:
            dt = t - int(last_t[user])
            if dt > 0:
                cap[user] *= math.exp(-dt / float(tau_s))
        last_t[user] = t

        cap_u = float(cap[user])
        cap_before[i] = cap_u

        if require_positive_cost and cost <= 0.0:
            f1 = 0.0
        else:
            f1 = cost / (cost + cap_u + eps) if cost > 0 else 0.0

        g_hold = 1.0 - math.exp(-float(r.delta_t) / float(max(tau_h, eps)))
        g_early = g_early_from_rank(int(r.r_rank), int(r.N_current), n_ref_val, eps)

        f2 = 0.5 * (g_hold + g_early)
        w_e = f1 * f2

        c_tilde = float(cA_tilde_map.get(r.contract_id, 0.0))
        credit = w_e * c_tilde

        f1_arr[i] = f1
        ghold_arr[i] = g_hold
        gearly_arr[i] = g_early
        f2_arr[i] = f2
        w_arr[i] = w_e
        ctilde_arr[i] = c_tilde
        credit_arr[i] = credit

        user_p0[user] += credit
        user_event_cnt[user] += 1

        # all spending updates capacity (fairness baseline)
        cap[user] += cost

    ev_out = ev[[
        "event_id", "buyer", "contract_id", "tx_type", "block_number", "tau_rel", "delta_t",
        "price", "gas", "cost"
    ]].copy()
    ev_out = ev_out.rename(columns={"buyer": "user"})
    ev_out["Cap_u"] = cap_before
    ev_out["f1"] = f1_arr
    ev_out["g_hold"] = ghold_arr
    ev_out["g_early"] = gearly_arr
    ev_out["f2"] = f2_arr
    ev_out["w_e"] = w_arr
    ev_out["cA_tilde"] = ctilde_arr
    ev_out["credit"] = credit_arr

    users = pd.DataFrame({"user": list(user_p0.keys()), "p0": list(user_p0.values())})
    users["event_cnt_anchor"] = users["user"].map(user_event_cnt).fillna(0).astype(int)

    p0_sum = float(users["p0"].sum())
    if p0_sum > 0:
        users["u0"] = users["p0"] / p0_sum
    else:
        users["u0"] = 0.0

    users = users.sort_values(["u0", "p0", "user"], ascending=[False, False, True]).reset_index(drop=True)
    users["rank_B"] = np.arange(1, len(users) + 1)
    return users, ev_out


# ============================================================
# 5) Save helpers
# ============================================================
def save_outputs(
    anchors: pd.DataFrame,
    event_scores: pd.DataFrame,
    user_prior: pd.DataFrame,
    stagea_user_csv: Path,
) -> None:
    OUTPUT_ANCHORS_CSV.parent.mkdir(parents=True, exist_ok=True)

    anchors.to_csv(OUTPUT_ANCHORS_CSV, index=False, encoding="utf-8-sig")
    event_scores.to_csv(OUTPUT_EVENT_SCORES_CSV, index=False, encoding="utf-8-sig")
    user_prior.to_csv(OUTPUT_USER_PRIOR_CSV, index=False, encoding="utf-8-sig")

    # align with Stage A user universe if available (for Stage C ingestion)
    if stagea_user_csv.exists():
        uA = pd.read_csv(stagea_user_csv)
        if "user" not in uA.columns:
            raise ValueError(f"Missing column 'user' in {stagea_user_csv.name}")

        full = uA[["user"]].drop_duplicates().copy()
        full = full.merge(user_prior[["user", "p0", "u0", "rank_B", "event_cnt_anchor"]], on="user", how="left")
        full["p0"] = full["p0"].fillna(0.0)
        full["u0"] = full["u0"].fillna(0.0)
        full["event_cnt_anchor"] = full["event_cnt_anchor"].fillna(0).astype(int)

        # rank only among non-zero users, keep zero users unranked as 0
        nz = full["u0"] > 0
        full.loc[nz, "rank_B"] = full.loc[nz, "u0"].rank(method="first", ascending=False).astype(int)
        full.loc[~nz, "rank_B"] = 0
        full["rank_B"] = full["rank_B"].astype(int)

        full = full.sort_values(["rank_B", "user"], ascending=[True, True]).reset_index(drop=True)
        full.to_csv(OUTPUT_U0_FULL_CSV, index=False, encoding="utf-8-sig")
    else:
        user_prior.to_csv(OUTPUT_U0_FULL_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# 6) Main
# ============================================================
def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {INPUT_DIR.resolve()}")

    print(f"[Load] events from {INPUT_DIR.resolve()}")
    df = load_trading_events(INPUT_DIR)
    print(f"[Data] rows={len(df):,} contracts={df['contract_id'].nunique():,} buyers={df['buyer'].nunique():,}")

    anchors = load_anchor_set(STAGEA_CONTRACT_CSV, HARD_TOP_RATIO, HARD_TOP_K)
    anchor_set = set(anchors["contract_id"].astype(str))
    cA_tilde_map = dict(zip(anchors["contract_id"].astype(str), anchors["cA_tilde"].astype(float)))
    print(f"[Anchors] |C_hard|={len(anchor_set):,} sum(cA_tilde)={anchors['cA_tilde'].sum():.6f}")

    ev = build_stageb_events(df, anchor_set)
    print(f"[Events] anchor acquisitions={len(ev):,}")

    user_prior, event_scores = compute_stageb_user_prior(
        ev=ev,
        cA_tilde_map=cA_tilde_map,
        tau_s=TAU_S,
        tau_h=TAU_H,
        n_ref=N_REF,
        eps=EPS,
        require_positive_cost=REQUIRE_POSITIVE_COST,
    )

    save_outputs(anchors=anchors, event_scores=event_scores, user_prior=user_prior, stagea_user_csv=STAGEA_USER_CSV)

    print(f"[Saved] {OUTPUT_ANCHORS_CSV.resolve()}")
    print(f"[Saved] {OUTPUT_EVENT_SCORES_CSV.resolve()}")
    print(f"[Saved] {OUTPUT_USER_PRIOR_CSV.resolve()}")
    print(f"[Saved] {OUTPUT_U0_FULL_CSV.resolve()}")

    if len(user_prior) > 0:
        print("[Top 15 users by u0]")
        print(user_prior.head(15)[["rank_B", "user", "u0", "p0", "event_cnt_anchor"]].to_string(index=False))
    else:
        print("[Top 15 users by u0] empty")


if __name__ == "__main__":
    main()
