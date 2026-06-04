from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class StageBResult:
    user_scores: pd.DataFrame
    event_scores: pd.DataFrame
    u0: np.ndarray
    users: list[str]
    normalized_anchor_scores: dict[str, float]
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
        raise ValueError(f"Stage B data missing required columns: {sorted(missing)}")

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


def _normalize_anchor_scores(contract_scores: pd.DataFrame, anchors: list[str]) -> dict[str, float]:
    anchor_df = contract_scores[contract_scores["contract_id"].isin(set(anchors))][["contract_id", "cA"]].copy()
    total = float(anchor_df["cA"].sum())
    if total <= 0:
        anchor_df["cA_norm"] = 1.0 / max(len(anchor_df), 1)
    else:
        anchor_df["cA_norm"] = anchor_df["cA"] / total
    return dict(zip(anchor_df["contract_id"], anchor_df["cA_norm"]))


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


def _is_acquisition_event(row: Any) -> bool:
    tx_type = str(row.tx_type)
    if tx_type == "trade":
        return bool(str(row.buyer))
    if tx_type == "mint":
        return bool(str(row.buyer)) and (float(row.gas) > 0 or float(row.price) > 0)
    return False


def _g_early(rank_r: int, n_unique: int, n_ref: int) -> float:
    rank = max(int(rank_r), 1)
    return 1.0 / float(rank * rank)


def build_stage_b_event_scores(
    df: pd.DataFrame,
    anchor_scores: dict[str, float],
    tau_end: int | None = None,
    tau_s: float = 200_000.0,
    tau_h: float = 50_000.0,
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
        if not _is_acquisition_event(row):
            continue
        cid = str(row.contract_id)
        if cid not in anchor_scores:
            continue

        user = str(row.buyer)
        t = int(row.block_number)
        cost = float(row.price) + float(row.gas)

        if user in last_t:
            dt = max(0, t - last_t[user])
            if dt > 0:
                cap[user] *= float(np.exp(-dt / max(tau_s, EPS)))
        cap_before = float(cap[user])
        last_t[user] = t

        if user not in owners_seen[cid]:
            owners_seen[cid].add(user)
            adopters[cid][user] = len(owners_seen[cid])

        n_unique = len(owners_seen[cid])
        rank_r = adopters[cid][user]
        tau_rel = int(next_block_by_idx.get(idx, tau_end))
        delta_t = max(0, min(tau_rel, tau_end) - t)

        f1 = cost / (cost + cap_before + EPS) if cost > 0 else 0.0
        g_hold = 1.0 - float(np.exp(-delta_t / max(tau_h, EPS)))
        g_early = _g_early(rank_r=rank_r, n_unique=n_unique, n_ref=n_ref)
        f2 = 0.5 * (g_hold + g_early)
        w_e = f1 * f2
        contribution = w_e * float(anchor_scores[cid])

        rows.append(
            {
                "event_index": idx,
                "user": user,
                "contract_id": cid,
                "tx_type": str(row.tx_type),
                "block_number": t,
                "tau_rel": tau_rel,
                "delta_t": delta_t,
                "cost": cost,
                "cap_before": cap_before,
                "f1": f1,
                "g_hold": g_hold,
                "g_early": g_early,
                "f2": f2,
                "w_e": w_e,
                "cA_norm": float(anchor_scores[cid]),
                "contribution": contribution,
            }
        )

        cap[user] += cost

    return pd.DataFrame(rows)


def aggregate_stage_b_users(
    event_scores: pd.DataFrame,
    users_meta: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if event_scores.empty:
        if users_meta is None:
            return pd.DataFrame(columns=["user", "p0_raw", "u0", "rank_B"]), np.array([]), []
        users = users_meta["user"].astype(str).tolist()
        base = pd.DataFrame({"user": users, "p0_raw": 0.0})
        base["u0"] = 1.0 / max(len(base), 1)
        base["rank_B"] = np.arange(1, len(base) + 1)
        return base, base["u0"].to_numpy(dtype=float), users

    user_df = event_scores.groupby("user", as_index=False)["contribution"].sum().rename(columns={"contribution": "p0_raw"})
    if users_meta is not None and "user" in users_meta.columns:
        base_cols = ["user"] + [col for col in ["group", "base_group"] if col in users_meta.columns]
        user_df = users_meta[base_cols].astype({"user": str}).merge(user_df, on="user", how="left")
        user_df["p0_raw"] = user_df["p0_raw"].fillna(0.0)

    total = float(user_df["p0_raw"].sum())
    if total <= 0:
        user_df["u0"] = 1.0 / max(len(user_df), 1)
    else:
        user_df["u0"] = user_df["p0_raw"] / total

    user_df = user_df.sort_values("u0", ascending=False).reset_index(drop=True)
    user_df["rank_B"] = np.arange(1, len(user_df) + 1)
    return user_df, user_df["u0"].to_numpy(dtype=float), user_df["user"].astype(str).tolist()


def run_stage_b(
    df: pd.DataFrame,
    contract_scores_a: pd.DataFrame,
    anchors: list[str],
    users_meta: pd.DataFrame | None = None,
    tau_end: int | None = None,
    tau_s: float = 200_000.0,
    tau_h: float = 50_000.0,
) -> StageBResult:
    anchor_scores = _normalize_anchor_scores(contract_scores=contract_scores_a, anchors=anchors)
    event_scores = build_stage_b_event_scores(
        df=df,
        anchor_scores=anchor_scores,
        tau_end=tau_end,
        tau_s=tau_s,
        tau_h=tau_h,
    )
    user_scores, u0, users = aggregate_stage_b_users(event_scores=event_scores, users_meta=users_meta)
    meta = {
        "tau_s": float(tau_s),
        "tau_h": float(tau_h),
        "tau_end": int(tau_end if tau_end is not None else (df["block_number"].max() if len(df) else 0)),
        "n_anchor_contracts": len(anchor_scores),
        "n_scored_events": int(len(event_scores)),
        "n_users": len(users),
    }
    return StageBResult(
        user_scores=user_scores,
        event_scores=event_scores,
        u0=u0,
        users=users,
        normalized_anchor_scores=anchor_scores,
        meta=meta,
    )
