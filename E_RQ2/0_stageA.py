from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


EPS = 1e-12


@dataclass
class StageAResult:
    W_hist: sparse.csr_matrix
    S_hist: sparse.csr_matrix
    users: list[str]
    contracts: list[str]
    contract_scores: pd.DataFrame
    user_scores: pd.DataFrame
    anchors: list[str]
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
        raise ValueError(f"Stage A data missing required columns: {sorted(missing)}")

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


def _add_weight(weight: dict[tuple[str, str], float], user: str, contract_id: str, value: float) -> None:
    if not user or value <= 0:
        return
    weight[(user, contract_id)] += float(value)


def build_historical_matrix(
    df: pd.DataFrame,
    include_mint_gas: bool = True,
) -> tuple[sparse.csr_matrix, list[str], list[str], dict[str, Any]]:
    work = _prepare_events(df)

    gas_weight: dict[tuple[str, str], float] = defaultdict(float)
    value_weight: dict[tuple[str, str], float] = defaultdict(float)
    peak_by_contract: dict[str, float] = defaultdict(float)
    token_state: dict[tuple[str, int], dict[str, Any]] = {}

    gas_df = work if include_mint_gas else work[work["tx_type"] != "mint"]
    for row in gas_df.itertuples(index=False):
        _add_weight(gas_weight, str(row.buyer), str(row.contract_id), float(row.gas))

    for row in work.itertuples(index=False):
        cid = str(row.contract_id)
        tid = int(row.token_id)
        tx_type = str(row.tx_type)
        buyer = str(row.buyer)
        seller = str(row.seller)
        price = float(row.price)

        if tx_type == "mint":
            token_state[(cid, tid)] = {"current_holder": buyer, "paid_holders": []}
            continue

        if tx_type != "trade":
            continue

        peak_old = float(peak_by_contract[cid])
        peak_new = max(peak_old, price)
        current = token_state.get((cid, tid), {"current_holder": seller, "paid_holders": []})
        paid_holders = [holder for holder in current.get("paid_holders", []) if holder]

        if price > 0 and buyer:
            _add_weight(value_weight, buyer, cid, price)

        shortfall = max(0.0, peak_old - price)
        if shortfall > 0 and paid_holders:
            remaining = shortfall
            for holder in reversed(paid_holders):
                if remaining <= EPS:
                    break
                _add_weight(value_weight, holder, cid, remaining)
                remaining = 0.0

        if peak_new > peak_old:
            paid_holders = []

        if buyer:
            paid_holders.append(buyer)

        token_state[(cid, tid)] = {
            "current_holder": buyer or current.get("current_holder", ""),
            "paid_holders": paid_holders,
        }
        peak_by_contract[cid] = peak_new

    users = sorted({user for user, _ in gas_weight.keys()} | {user for user, _ in value_weight.keys()})
    contracts = sorted(set(work["contract_id"].tolist()))
    user_index = {user: idx for idx, user in enumerate(users)}
    contract_index = {cid: idx for idx, cid in enumerate(contracts)}

    total_weight: dict[tuple[int, int], float] = defaultdict(float)
    for mapping in (gas_weight, value_weight):
        for (user, cid), value in mapping.items():
            if user in user_index and cid in contract_index and value > 0:
                total_weight[(user_index[user], contract_index[cid])] += float(value)

    rows = [key[0] for key in total_weight]
    cols = [key[1] for key in total_weight]
    data = [total_weight[key] for key in total_weight]
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(contracts)))

    value_sum_by_contract = defaultdict(float)
    for (_, cid), value in value_weight.items():
        value_sum_by_contract[cid] += float(value)

    meta = {
        "n_users": len(users),
        "n_contracts": len(contracts),
        "nnz": int(matrix.nnz),
        "gas_total": float(sum(gas_weight.values())),
        "value_total": float(sum(value_weight.values())),
        "peak_sum": float(sum(peak_by_contract.values())),
        "value_minus_peak_sum": float(
            sum(value_sum_by_contract.get(cid, 0.0) - peak_by_contract.get(cid, 0.0) for cid in contracts)
        ),
    }
    return matrix, users, contracts, meta


def symmetric_normalize(W_hist: sparse.csr_matrix) -> sparse.csr_matrix:
    d_u = np.asarray(W_hist.sum(axis=1)).reshape(-1)
    d_c = np.asarray(W_hist.sum(axis=0)).reshape(-1)
    d_u[d_u <= 0] = 1.0
    d_c[d_c <= 0] = 1.0
    su = sparse.diags(1.0 / np.sqrt(d_u))
    sc = sparse.diags(1.0 / np.sqrt(d_c))
    return su @ W_hist @ sc


def run_stage_a_birank(
    S_hist: sparse.csr_matrix,
    eta: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-10,
    u0: np.ndarray | None = None,
    c0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_users, n_contracts = S_hist.shape
    if u0 is None:
        u0 = np.ones(n_users, dtype=float) / max(n_users, 1)
    if c0 is None:
        c0 = np.ones(n_contracts, dtype=float) / max(n_contracts, 1)

    u = np.asarray(u0, dtype=float).reshape(-1, 1)
    c = np.asarray(c0, dtype=float).reshape(-1, 1)
    u0v = u.copy()
    c0v = c.copy()

    for iteration in range(1, max_iter + 1):
        c_new = eta * (S_hist.T @ u) + (1.0 - eta) * c0v
        u_new = eta * (S_hist @ c) + (1.0 - eta) * u0v
        c_new = c_new / (float(c_new.sum()) + EPS)
        u_new = u_new / (float(u_new.sum()) + EPS)
        delta = float(np.abs(c_new - c).sum() + np.abs(u_new - u).sum())
        c, u = c_new, u_new
        if delta < tol:
            return u.ravel(), c.ravel(), iteration

    return u.ravel(), c.ravel(), max_iter


def run_stage_a(
    df: pd.DataFrame,
    anchor_ratio: float = 0.03,
    eta: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-10,
    include_mint_gas: bool = True,
) -> StageAResult:
    W_hist, users, contracts, meta = build_historical_matrix(df=df, include_mint_gas=include_mint_gas)
    S_hist = symmetric_normalize(W_hist)
    u_scores, c_scores, iterations = run_stage_a_birank(S_hist=S_hist, eta=eta, max_iter=max_iter, tol=tol)

    contract_df = pd.DataFrame({"contract_id": contracts, "cA": c_scores}).sort_values("cA", ascending=False).reset_index(drop=True)
    contract_df["rank_A"] = np.arange(1, len(contract_df) + 1)

    user_df = pd.DataFrame({"user": users, "uA": u_scores}).sort_values("uA", ascending=False).reset_index(drop=True)
    user_df["rank_A"] = np.arange(1, len(user_df) + 1)

    topk = max(1, int(np.ceil(anchor_ratio * max(len(contract_df), 1))))
    anchors = contract_df.head(topk)["contract_id"].tolist()
    contract_df["is_anchor"] = contract_df["contract_id"].isin(anchors).astype(int)

    meta = dict(meta)
    meta.update(
        {
            "eta": float(eta),
            "anchor_ratio": float(anchor_ratio),
            "iterations": int(iterations),
            "n_anchors": len(anchors),
        }
    )

    return StageAResult(
        W_hist=W_hist,
        S_hist=S_hist,
        users=users,
        contracts=contracts,
        contract_scores=contract_df,
        user_scores=user_df,
        anchors=anchors,
        meta=meta,
    )
