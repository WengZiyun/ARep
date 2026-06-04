from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


EPS = 1e-12


@dataclass
class StageCResult:
    W_rec: sparse.csr_matrix
    users: list[str]
    contracts: list[str]
    user_scores: pd.DataFrame
    contract_scores: pd.DataFrame
    event_scores: pd.DataFrame
    meta: dict[str, Any]


def _prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    required = {
        "block_number",
        "tx_type",
        "contract_id",
        "token_id",
        "seller",
        "buyer",
        "price",
        "gas",
    }
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Stage C data missing required columns: {sorted(missing)}")

    if "tx_index_in_block" not in work.columns:
        work["tx_index_in_block"] = 0

    work["block_number"] = pd.to_numeric(work["block_number"], errors="coerce").fillna(0).astype(int)
    work["tx_index_in_block"] = pd.to_numeric(work["tx_index_in_block"], errors="coerce").fillna(0).astype(int)
    work["price"] = pd.to_numeric(work["price"], errors="coerce").fillna(0.0)
    work["gas"] = pd.to_numeric(work["gas"], errors="coerce").fillna(0.0)
    work["token_id"] = pd.to_numeric(work["token_id"], errors="coerce").fillna(-1).astype(int)
    work["seller"] = work["seller"].fillna("").astype(str)
    work["buyer"] = work["buyer"].fillna("").astype(str)
    work["contract_id"] = work["contract_id"].fillna("").astype(str)
    work["tx_type"] = work["tx_type"].fillna("").astype(str)
    return work.sort_values(["block_number", "tx_index_in_block"]).reset_index(drop=True)


def _is_supported_stage_c_event(row: Any, include_non_acquisition: bool) -> bool:
    tx_type = str(row.tx_type)
    if tx_type in {"trade", "mint"}:
        return bool(str(row.buyer))
    return include_non_acquisition and bool(str(row.buyer))


def _compute_relinquish_map(work: pd.DataFrame, tau_end: int) -> dict[int, int]:
    next_block_by_idx: dict[int, int] = {}
    last_owner_idx: dict[tuple[str, int], int] = {}
    for idx, row in enumerate(work.itertuples(index=False)):
        if str(row.tx_type) not in {"trade", "mint"}:
            continue
        key = (str(row.contract_id), int(row.token_id))
        if key in last_owner_idx:
            next_block_by_idx[last_owner_idx[key]] = int(row.block_number)
        last_owner_idx[key] = idx
    for idx in last_owner_idx.values():
        next_block_by_idx[idx] = int(tau_end)
    return next_block_by_idx


def _g_early(rank_r: int, n_unique: int, n_ref: int) -> float:
    rank = max(int(rank_r), 1)
    return 1.0 / float(rank * rank)


def score_stage_c_events(
    df: pd.DataFrame,
    tau_end: int | None = None,
    tau_decay: float = 216_000.0,
    tau_s: float = 200_000.0,
    tau_h: float = 50_000.0,
    include_non_acquisition: bool = True,
) -> pd.DataFrame:
    work = _prepare_events(df)
    if tau_end is None:
        tau_end = int(work["block_number"].max()) if len(work) else 0
    next_block_by_idx = _compute_relinquish_map(work, tau_end=tau_end)

    trade_df = work[work["tx_type"].eq("trade")]
    n_ref = int(np.median(trade_df.groupby("contract_id")["buyer"].nunique().values)) if len(trade_df) else 1
    n_ref = max(n_ref, 1)

    cap = defaultdict(float)
    last_t = {}
    adopters = defaultdict(dict)
    owners_seen = defaultdict(set)
    rows: list[dict[str, Any]] = []

    for idx, row in enumerate(work.itertuples(index=False)):
        if not _is_supported_stage_c_event(row, include_non_acquisition=include_non_acquisition):
            continue

        user = str(row.buyer)
        cid = str(row.contract_id)
        t = int(row.block_number)
        tx_type = str(row.tx_type)
        v_e = float(row.price) if tx_type == "trade" else 0.0
        gas_e = float(row.gas)

        if user in last_t:
            dt = max(0, t - last_t[user])
            if dt > 0:
                cap[user] *= float(np.exp(-dt / max(tau_s, EPS)))
        cap_before = float(cap[user])
        last_t[user] = t

        if tx_type == "trade":
            if user not in owners_seen[cid]:
                owners_seen[cid].add(user)
                adopters[cid][user] = len(owners_seen[cid])
            n_unique = len(owners_seen[cid])
            rank_r = adopters[cid][user]
            tau_rel = int(next_block_by_idx.get(idx, tau_end))
            delta_t = max(0, min(tau_rel, tau_end) - t)
            g_early = _g_early(rank_r=rank_r, n_unique=n_unique, n_ref=n_ref)
            g_hold = 1.0 - float(np.exp(-delta_t / max(tau_h, EPS)))
            cost = v_e + gas_e
            f1 = cost / (cost + cap_before + EPS) if cost > 0 else 0.0
            f2 = 0.5 * (g_early + g_hold)
            w_e = f1 * f2
        elif tx_type == "mint":
            tau_rel = int(t)
            delta_t = 0
            g_early = 0.0
            g_hold = 0.0
            cost = gas_e
            f1 = cost / (cost + cap_before + EPS) if cost > 0 else 0.0
            f2 = 1.0
            w_e = f1
        else:
            tau_rel = int(tau_end)
            delta_t = 0
            g_early = 0.0
            g_hold = 1.0 - float(np.exp(-gas_e / max(tau_h, EPS)))
            cost = v_e + gas_e
            f1 = cost / (cost + cap_before + EPS) if cost > 0 else 0.0
            f2 = 0.5 * (g_early + g_hold)
            w_e = f1 * f2
        decay = float(np.exp(-(tau_end - t) / max(tau_decay, EPS)))
        w_decay = w_e * decay

        rows.append(
            {
                "event_index": idx,
                "user": user,
                "contract_id": cid,
                "tx_type": tx_type,
                "block_number": t,
                "tau_rel": tau_rel,
                "delta_t": delta_t,
                "v_e": v_e,
                "gas_e": gas_e,
                "cap_before": cap_before,
                "f1": f1,
                "g_early": g_early,
                "g_hold": g_hold,
                "f2": f2,
                "w_e": w_e,
                "decay": decay,
                "w_decay": w_decay,
            }
        )
        cap[user] += cost

    return pd.DataFrame(rows)


def build_recent_matrix(event_scores: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    if event_scores.empty:
        return sparse.csr_matrix((0, 0)), [], []

    users = sorted(event_scores["user"].astype(str).unique().tolist())
    contracts = sorted(event_scores["contract_id"].astype(str).unique().tolist())
    user_index = {user: idx for idx, user in enumerate(users)}
    contract_index = {cid: idx for idx, cid in enumerate(contracts)}

    grouped = event_scores.groupby(["user", "contract_id"], as_index=False)["w_decay"].sum()
    rows = [user_index[str(row.user)] for row in grouped.itertuples(index=False)]
    cols = [contract_index[str(row.contract_id)] for row in grouped.itertuples(index=False)]
    data = [float(row.w_decay) for row in grouped.itertuples(index=False)]
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(contracts)))
    return matrix, users, contracts


def build_contract_reliability(
    event_scores: pd.DataFrame,
    contracts: list[str],
    stage_a_contract_scores: pd.DataFrame | None = None,
) -> dict[str, float]:
    if event_scores.empty:
        return {str(contract_id): 1.0 for contract_id in contracts}

    gas_by_contract = (
        event_scores.groupby("contract_id", as_index=False)["gas_e"]
        .sum()
        .rename(columns={"gas_e": "gas_sum"})
    )
    gas_map = dict(zip(gas_by_contract["contract_id"].astype(str), gas_by_contract["gas_sum"].astype(float)))

    hard_contracts: list[str] = []
    if stage_a_contract_scores is not None and "contract_id" in stage_a_contract_scores.columns:
        if "is_anchor" in stage_a_contract_scores.columns:
            hard_contracts = (
                stage_a_contract_scores.loc[stage_a_contract_scores["is_anchor"].astype(int).eq(1), "contract_id"]
                .astype(str)
                .tolist()
            )
        else:
            hard_contracts = stage_a_contract_scores["contract_id"].astype(str).tolist()

    hard_gas = [float(gas_map[cid]) for cid in hard_contracts if cid in gas_map]
    if not hard_gas:
        hard_gas = [float(x) for x in gas_by_contract["gas_sum"].astype(float).tolist() if float(x) > 0]
    median_hard_gas = float(np.median(hard_gas)) if hard_gas else 0.0

    reliability: dict[str, float] = {}
    for contract_id in contracts:
        gas_sum = float(gas_map.get(str(contract_id), 0.0))
        if median_hard_gas <= 0:
            reliability[str(contract_id)] = 1.0 if gas_sum > 0 else 0.0
        else:
            reliability[str(contract_id)] = float(np.clip(gas_sum / (median_hard_gas + EPS), 0.0, 1.0))
    return reliability


def _align_prior(names: list[str], value_map: dict[str, float], default_uniform: bool = True) -> np.ndarray:
    vec = np.array([float(value_map.get(name, 0.0)) for name in names], dtype=float)
    total = float(vec.sum())
    if total <= 0 and default_uniform:
        vec = np.ones(len(names), dtype=float) / max(len(names), 1)
    else:
        vec = vec / (total + EPS)
    return vec


def personalized_birank(
    W_rec: sparse.csr_matrix,
    users: list[str],
    contracts: list[str],
    u0_map: dict[str, float],
    c0_map: dict[str, float] | None = None,
    contract_reliability_map: dict[str, float] | None = None,
    alpha: float = 1.0,
    beta: float = 0.7,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_users, n_contracts = W_rec.shape
    if n_users == 0 or n_contracts == 0:
        return np.array([]), np.array([]), 0

    d_u = np.asarray(W_rec.sum(axis=1)).reshape(-1)
    d_c = np.asarray(W_rec.sum(axis=0)).reshape(-1)
    d_u[d_u <= 0] = 1.0
    d_c[d_c <= 0] = 1.0

    p_uc = sparse.diags(1.0 / d_u) @ W_rec
    p_cu = sparse.diags(1.0 / d_c) @ W_rec.T

    u0 = _align_prior(users, u0_map, default_uniform=True).reshape(-1, 1)
    c0 = _align_prior(contracts, c0_map or {}, default_uniform=True).reshape(-1, 1)
    contract_reliability = np.array(
        [float((contract_reliability_map or {}).get(contract_id, 1.0)) for contract_id in contracts],
        dtype=float,
    ).reshape(-1, 1)
    contract_reliability = np.clip(contract_reliability, 0.0, 1.0)
    if float(contract_reliability.sum()) <= 0:
        contract_reliability = np.ones((n_contracts, 1), dtype=float)
    u = u0.copy()
    c = c0.copy()

    for iteration in range(1, max_iter + 1):
        c_new = beta * (p_uc.T @ u) + (1.0 - beta) * c0
        c_reliable = c_new * contract_reliability
        u_new = alpha * (p_cu.T @ c_reliable) + (1.0 - alpha) * u0
        c_new = c_new / (float(c_new.sum()) + EPS)
        u_new = u_new / (float(u_new.sum()) + EPS)
        delta = float(np.abs(c_new - c).sum() + np.abs(u_new - u).sum())
        c, u = c_new, u_new
        if delta < tol:
            return u.ravel(), c.ravel(), iteration

    return u.ravel(), c.ravel(), max_iter


def run_stage_c(
    df: pd.DataFrame,
    stage_b_user_scores: pd.DataFrame,
    stage_a_contract_scores: pd.DataFrame | None = None,
    tau_end: int | None = None,
    tau_decay: float = 216_000.0,
    tau_s: float = 200_000.0,
    tau_h: float = 50_000.0,
    alpha: float = 1.0,
    beta: float = 0.7,
    max_iter: int = 200,
    tol: float = 1e-10,
    include_non_acquisition: bool = True,
) -> StageCResult:
    event_scores = score_stage_c_events(
        df=df,
        tau_end=tau_end,
        tau_decay=tau_decay,
        tau_s=tau_s,
        tau_h=tau_h,
        include_non_acquisition=include_non_acquisition,
    )
    W_rec, users, contracts = build_recent_matrix(event_scores)
    u0_map = dict(zip(stage_b_user_scores["user"].astype(str), stage_b_user_scores["u0"].astype(float)))
    c0_map = None
    if stage_a_contract_scores is not None:
        c0_map = dict(zip(stage_a_contract_scores["contract_id"].astype(str), stage_a_contract_scores["cA"].astype(float)))
    contract_reliability_map = build_contract_reliability(
        event_scores=event_scores,
        contracts=contracts,
        stage_a_contract_scores=stage_a_contract_scores,
    )

    u_scores, c_scores, iterations = personalized_birank(
        W_rec=W_rec,
        users=users,
        contracts=contracts,
        u0_map=u0_map,
        c0_map=c0_map,
        contract_reliability_map=contract_reliability_map,
        alpha=alpha,
        beta=beta,
        max_iter=max_iter,
        tol=tol,
    )

    user_df = pd.DataFrame({"user": users, "uC": u_scores}).sort_values("uC", ascending=False).reset_index(drop=True)
    user_df["rank_C"] = np.arange(1, len(user_df) + 1)
    contract_df = pd.DataFrame({"contract_id": contracts, "cC": c_scores}).sort_values("cC", ascending=False).reset_index(drop=True)
    contract_df["rank_C"] = np.arange(1, len(contract_df) + 1)

    meta = {
        "tau_end": int(tau_end if tau_end is not None else (df["block_number"].max() if len(df) else 0)),
        "tau_decay": float(tau_decay),
        "tau_s": float(tau_s),
        "tau_h": float(tau_h),
        "alpha": float(alpha),
        "beta": float(beta),
        "iterations": int(iterations),
        "n_users": len(users),
        "n_contracts": len(contracts),
        "nnz": int(W_rec.nnz),
        "n_scored_events": int(len(event_scores)),
        "n_contracts_reliability_zero": int(sum(1 for v in contract_reliability_map.values() if float(v) <= 0)),
    }
    return StageCResult(
        W_rec=W_rec,
        users=users,
        contracts=contracts,
        user_scores=user_df,
        contract_scores=contract_df,
        event_scores=event_scores,
        meta=meta,
    )
