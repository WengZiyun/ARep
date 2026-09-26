#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================
LABEL_CSV = Path("output/C/3_collectionlabel.csv")
TRP_SCRIPT = Path("D_RQ1/0_rugpullTRP.py")
BASE_EVAL_SCRIPT = Path("D_RQ1/1_eval_rugpull3.py")
BASE_OUTPUT_DIR = Path("output/D/1_eval_rugpull3")
OUTPUT_DIR = Path("output/D/7_agent_panel")

LOW_REP_THRESHOLD_PCT = 0.80
LOW_REP_THRESHOLD_LIST = [0.80, 0.90]
TARGET_COLLECTION_TYPES = {"rugpull", "trust", "unknown"}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def build_trade_panel(df_all: pd.DataFrame) -> pd.DataFrame:
    trades = df_all[df_all["tx_type"] == "trade"].copy()
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "contract_id",
                "block_number",
                "timestamp",
                "price",
                "buyer",
                "seller",
            ]
        )
    trades["contract_id"] = trades["contract_id"].astype(str)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce").fillna(0.0).astype(float)
    trades = trades.sort_values(["contract_id", "block_number", "tx_index_in_block"], kind="mergesort").reset_index(drop=True)
    return trades[["contract_id", "block_number", "timestamp", "price", "buyer", "seller"]]


def build_price_lookup(trades: pd.DataFrame) -> dict[str, dict[str, list]]:
    lookup: dict[str, dict[str, list]] = {}
    if trades.empty:
        return lookup
    for contract_id, sub in trades.groupby("contract_id", sort=False):
        lookup[str(contract_id)] = {
            "blocks": sub["block_number"].astype(np.int64).tolist(),
            "prices": sub["price"].astype(float).tolist(),
            "times": sub["timestamp"].tolist(),
        }
    return lookup


def build_window_market_feature_table(df_all: pd.DataFrame, score_windows: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty or score_windows.empty:
        return pd.DataFrame(
            columns=[
                "contract_id",
                "snapshot_idx",
                "window_trade_count",
                "window_trade_value_total_usd",
                "window_unique_buyers",
                "window_unique_trade_users",
                "window_mint_cnt",
                "window_gas_total_usd",
            ]
        )

    windows = (
        score_windows[["snapshot_idx", "start_block", "end_block"]]
        .drop_duplicates()
        .sort_values(["snapshot_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    ends = windows["end_block"].astype(np.int64).to_numpy()
    starts = windows["start_block"].astype(np.int64).to_numpy()
    snap_ids = windows["snapshot_idx"].astype(np.int64).to_numpy()

    ev = df_all.copy()
    ev["block_number"] = pd.to_numeric(ev["block_number"], errors="coerce").fillna(-1).astype(np.int64)
    lo = int(starts.min())
    hi = int(ends.max())
    ev = ev[(ev["block_number"] >= lo) & (ev["block_number"] <= hi)].copy()
    if ev.empty:
        return pd.DataFrame(
            columns=[
                "contract_id",
                "snapshot_idx",
                "window_trade_count",
                "window_trade_value_total_usd",
                "window_unique_buyers",
                "window_unique_trade_users",
                "window_mint_cnt",
                "window_gas_total_usd",
            ]
        )

    pos = np.searchsorted(ends, ev["block_number"].to_numpy(dtype=np.int64), side="left")
    valid = pos < len(ends)
    ev = ev.loc[valid].copy()
    pos = pos[valid]
    ev["snapshot_idx"] = snap_ids[pos]
    ev["window_start_block_effective"] = starts[pos]
    ev = ev[ev["block_number"] >= ev["window_start_block_effective"]].copy()
    if ev.empty:
        return pd.DataFrame(
            columns=[
                "contract_id",
                "snapshot_idx",
                "window_trade_count",
                "window_trade_value_total_usd",
                "window_unique_buyers",
                "window_unique_trade_users",
                "window_mint_cnt",
                "window_gas_total_usd",
            ]
        )

    ev["contract_id"] = ev["contract_id"].astype(str)
    ev["tx_type"] = ev["tx_type"].astype(str).str.lower().str.strip()
    ev["buyer"] = ev["buyer"].fillna("").astype(str).str.lower().str.strip()
    ev["seller"] = ev["seller"].fillna("").astype(str).str.lower().str.strip()
    ev["price"] = pd.to_numeric(ev["price"], errors="coerce").fillna(0.0).astype(float)
    ev["gas"] = pd.to_numeric(ev["gas"], errors="coerce").fillna(0.0).astype(float)

    base = ev.groupby(["contract_id", "snapshot_idx"], as_index=False).agg(
        window_mint_cnt=("tx_type", lambda s: int((s == "mint").sum())),
        window_gas_total_usd=("gas", "sum"),
    )

    trades = ev[ev["tx_type"] == "trade"].copy()
    if trades.empty:
        base["window_trade_count"] = 0
        base["window_trade_value_total_usd"] = 0.0
        base["window_unique_buyers"] = 0
        base["window_unique_trade_users"] = 0
        return base

    trade_sum = trades.groupby(["contract_id", "snapshot_idx"], as_index=False).agg(
        window_trade_count=("tx_type", "size"),
        window_trade_value_total_usd=("price", "sum"),
        window_unique_buyers=("buyer", lambda s: int(s[s != ""].nunique())),
    )

    buyers = trades[["contract_id", "snapshot_idx", "buyer"]].rename(columns={"buyer": "user"})
    sellers = trades[["contract_id", "snapshot_idx", "seller"]].rename(columns={"seller": "user"})
    trade_users = pd.concat([buyers, sellers], ignore_index=True)
    trade_users = trade_users[trade_users["user"] != ""].copy()
    uniq_trade_users = trade_users.groupby(["contract_id", "snapshot_idx"], as_index=False).agg(
        window_unique_trade_users=("user", "nunique")
    )

    out = base.merge(trade_sum, on=["contract_id", "snapshot_idx"], how="left")
    out = out.merge(uniq_trade_users, on=["contract_id", "snapshot_idx"], how="left")
    out["window_trade_count"] = out["window_trade_count"].fillna(0).astype(int)
    out["window_trade_value_total_usd"] = out["window_trade_value_total_usd"].fillna(0.0).astype(float)
    out["window_unique_buyers"] = out["window_unique_buyers"].fillna(0).astype(int)
    out["window_unique_trade_users"] = out["window_unique_trade_users"].fillna(0).astype(int)
    return out


def latest_trade_before_or_at(lookup: dict[str, dict[str, list]], contract_id: str, end_block: int):
    info = lookup.get(str(contract_id))
    if not info:
        return np.nan, pd.NaT, 0
    blocks = info["blocks"]
    idx = np.searchsorted(blocks, int(end_block), side="right") - 1
    if idx < 0:
        return np.nan, pd.NaT, 0
    return float(info["prices"][idx]), info["times"][idx], 1


def first_trade_after(lookup: dict[str, dict[str, list]], contract_id: str, end_block: int):
    info = lookup.get(str(contract_id))
    if not info:
        return np.nan, pd.NaT, 0
    blocks = info["blocks"]
    idx = np.searchsorted(blocks, int(end_block), side="right")
    if idx >= len(blocks):
        return np.nan, pd.NaT, 0
    return float(info["prices"][idx]), info["times"][idx], 1


def add_low_rep_flags(panel: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    out = panel.copy()
    out["low_rep_threshold_pct"] = float(threshold_pct)
    out["is_low_rep"] = (pd.to_numeric(out["rep_rank_pct"], errors="coerce") >= float(threshold_pct)).astype(int)
    out["low_rep_streak"] = 0

    for contract_id, idx in out.groupby("contract_id", sort=False).groups.items():
        streak = 0
        for pos in list(idx):
            if int(out.at[pos, "is_low_rep"]) == 1:
                streak += 1
            else:
                streak = 0
            out.at[pos, "low_rep_streak"] = streak
    return out


def build_agent_panel(
    scores: pd.DataFrame,
    selected_universe: pd.DataFrame,
    label_df: pd.DataFrame,
    price_lookup: dict[str, dict[str, list]],
    market_features: pd.DataFrame,
) -> pd.DataFrame:
    work = scores.copy()
    work["contract_id"] = work["contract_id"].astype(str)
    work["contract_address"] = work["contract_address"].fillna("").astype(str).str.lower().str.strip()
    work["collection_type"] = work["collection_type"].fillna("unknown").astype(str).str.lower().str.strip()
    work = work[work["collection_type"].isin(TARGET_COLLECTION_TYPES)].copy()
    work["snapshot_idx"] = pd.to_numeric(work["snapshot_idx"], errors="coerce").astype(np.int64)
    work["start_block"] = pd.to_numeric(work["start_block"], errors="coerce").astype(np.int64)
    work["end_block"] = pd.to_numeric(work["end_block"], errors="coerce").astype(np.int64)
    work["rep_score"] = pd.to_numeric(work["cC"], errors="coerce").fillna(0.0).astype(float)
    work["rep_rank"] = pd.to_numeric(work["rank_C"], errors="coerce").astype(np.int64)
    work["rep_rank_pct"] = pd.to_numeric(work["rank_pct"], errors="coerce").fillna(1.0).astype(float)

    uni = selected_universe.copy()
    uni["contract_address"] = uni["contract_address"].astype(str).str.lower().str.strip()
    uni["create_block_number"] = pd.to_numeric(uni["create_block_number"], errors="coerce")
    uni["detected_block_number"] = pd.to_numeric(uni["detected_block_number"], errors="coerce")
    uni = uni.drop_duplicates(subset=["contract_address"], keep="first")

    lbl = label_df.copy()
    lbl["contract_address"] = lbl["contract_address"].astype(str).str.lower().str.strip()
    lbl["detected_time"] = pd.to_datetime(lbl["detected_time"], errors="coerce", utc=True)
    lbl = lbl.drop_duplicates(subset=["contract_address"], keep="first")

    out = work.merge(
        uni[["contract_address", "create_block_number", "detected_block_number", "daily_activity", "source_tx_file"]],
        on="contract_address",
        how="left",
    )
    out = out.merge(lbl[["contract_address", "detected_time"]], on="contract_address", how="left")
    out = out.merge(market_features, on=["contract_id", "snapshot_idx"], how="left")
    out["window_trade_count"] = pd.to_numeric(out["window_trade_count"], errors="coerce").fillna(0).astype(int)
    out["window_trade_value_total_usd"] = pd.to_numeric(out["window_trade_value_total_usd"], errors="coerce").fillna(0.0).astype(float)
    out["window_unique_buyers"] = pd.to_numeric(out["window_unique_buyers"], errors="coerce").fillna(0).astype(int)
    out["window_unique_trade_users"] = pd.to_numeric(out["window_unique_trade_users"], errors="coerce").fillna(0).astype(int)
    out["window_mint_cnt"] = pd.to_numeric(out["window_mint_cnt"], errors="coerce").fillna(0).astype(int)
    out["window_gas_total_usd"] = pd.to_numeric(out["window_gas_total_usd"], errors="coerce").fillna(0.0).astype(float)

    out["holding_state"] = "flat"
    out["can_buy_baseline"] = (out["rep_rank_pct"] < LOW_REP_THRESHOLD_PCT).astype(int)
    out["lead_windows_to_detect"] = np.where(
        pd.notna(out["detected_block_number"]) & (out["detected_block_number"] >= out["end_block"]),
        ((out["detected_block_number"] - out["end_block"]) / out["end_block"].map(lambda _: 1)).astype(float),  # placeholder; replaced below
        np.nan,
    )

    step_by_contract = (
        out.groupby("contract_id", as_index=False)
        .agg(step_blocks=("end_block", lambda s: float(np.median(np.diff(sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist()))))) if len(set(s)) > 1 else np.nan))
    )
    out = out.merge(step_by_contract, on="contract_id", how="left")
    out["lead_windows_to_detect"] = np.where(
        pd.notna(out["detected_block_number"]) & pd.notna(out["step_blocks"]) & (out["step_blocks"] > 0),
        (out["detected_block_number"] - out["end_block"]) / out["step_blocks"],
        np.nan,
    )

    last_prices = []
    last_times = []
    last_exists = []
    next_prices = []
    next_times = []
    next_exists = []
    for row in out.itertuples(index=False):
        last_price, last_time, has_last = latest_trade_before_or_at(price_lookup, str(row.contract_id), int(row.end_block))
        next_price, next_time, has_next = first_trade_after(price_lookup, str(row.contract_id), int(row.end_block))
        last_prices.append(last_price)
        last_times.append(last_time)
        last_exists.append(has_last)
        next_prices.append(next_price)
        next_times.append(next_time)
        next_exists.append(has_next)

    out["window_last_trade_price_usd"] = last_prices
    out["window_last_trade_time"] = last_times
    out["window_last_trade_exists"] = last_exists
    out["next_window_first_trade_price_usd"] = next_prices
    out["next_window_first_trade_time"] = next_times
    out["next_trade_exists"] = next_exists
    out["baseline_buy_price_usd"] = out["window_last_trade_price_usd"]
    out["baseline_exit_price_usd"] = np.where(
        pd.notna(out["detected_block_number"]) & (out["detected_block_number"] <= out["end_block"]),
        out["window_last_trade_price_usd"],
        out["next_window_first_trade_price_usd"],
    )
    out = out.sort_values(["contract_id", "snapshot_idx"], kind="mergesort").reset_index(drop=True)
    # Cumulative minted items up to the current window as a proxy for current collection size.
    out["current_mint_supply_count"] = out.groupby("contract_id", sort=False)["window_mint_cnt"].cumsum().astype(int)
    out["prev_window_last_trade_price_usd"] = out.groupby("contract_id", sort=False)["window_last_trade_price_usd"].shift(1)
    out["prev_window_trade_count"] = out.groupby("contract_id", sort=False)["window_trade_count"].shift(1)
    out["price_change_vs_prev_window_pct"] = np.where(
        pd.notna(out["prev_window_last_trade_price_usd"]) & (pd.to_numeric(out["prev_window_last_trade_price_usd"], errors="coerce") > 0),
        (pd.to_numeric(out["window_last_trade_price_usd"], errors="coerce") - pd.to_numeric(out["prev_window_last_trade_price_usd"], errors="coerce"))
        / pd.to_numeric(out["prev_window_last_trade_price_usd"], errors="coerce"),
        np.nan,
    )
    out["trade_count_change_vs_prev_window_pct"] = np.where(
        pd.notna(out["prev_window_trade_count"]) & (pd.to_numeric(out["prev_window_trade_count"], errors="coerce") > 0),
        (pd.to_numeric(out["window_trade_count"], errors="coerce") - pd.to_numeric(out["prev_window_trade_count"], errors="coerce"))
        / pd.to_numeric(out["prev_window_trade_count"], errors="coerce"),
        np.nan,
    )
    out["liquidity_flag"] = (
        (pd.to_numeric(out["window_last_trade_exists"], errors="coerce").fillna(0).astype(int) > 0)
        | (pd.to_numeric(out["next_trade_exists"], errors="coerce").fillna(0).astype(int) > 0)
    ).astype(int)
    out["agent_input_version"] = "v1"

    frames = [add_low_rep_flags(out, threshold) for threshold in LOW_REP_THRESHOLD_LIST]
    final = pd.concat(frames, ignore_index=True)
    final = final.sort_values(
        ["contract_id", "low_rep_threshold_pct", "snapshot_idx"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    final["lead_windows_to_detect"] = pd.to_numeric(final["lead_windows_to_detect"], errors="coerce")
    return final[
        [
            "contract_id",
            "contract_address",
            "collection_type",
            "snapshot_idx",
            "start_block",
            "end_block",
            "create_block_number",
            "detected_block_number",
            "detected_time",
            "daily_activity",
            "source_tx_file",
            "window_trade_count",
            "window_trade_value_total_usd",
            "window_unique_buyers",
            "window_unique_trade_users",
            "window_mint_cnt",
            "current_mint_supply_count",
            "window_gas_total_usd",
            "prev_window_last_trade_price_usd",
            "prev_window_trade_count",
            "price_change_vs_prev_window_pct",
            "trade_count_change_vs_prev_window_pct",
            "liquidity_flag",
            "rep_score",
            "rep_rank",
            "rep_rank_pct",
            "low_rep_threshold_pct",
            "is_low_rep",
            "low_rep_streak",
            "lead_windows_to_detect",
            "holding_state",
            "can_buy_baseline",
            "window_last_trade_price_usd",
            "window_last_trade_time",
            "window_last_trade_exists",
            "next_window_first_trade_price_usd",
            "next_window_first_trade_time",
            "next_trade_exists",
            "baseline_buy_price_usd",
            "baseline_exit_price_usd",
            "agent_input_version",
        ]
    ]


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in df.to_dict(orient="records"):
            normalized = {}
            for key, value in row.items():
                if pd.isna(value):
                    normalized[key] = None
                elif isinstance(value, pd.Timestamp):
                    normalized[key] = value.isoformat()
                else:
                    normalized[key] = value
            fh.write(json.dumps(normalized, ensure_ascii=False) + "\n")


def main() -> None:
    ensure_exists(LABEL_CSV)
    ensure_exists(BASE_OUTPUT_DIR / "snapshot_contract_scores.csv")
    ensure_exists(BASE_OUTPUT_DIR / "selected_universe.csv")

    trp = load_module(TRP_SCRIPT, "trp_mod_for_agent_panel")
    eval3 = load_module(BASE_EVAL_SCRIPT, "eval3_mod_for_agent_panel")

    label_df = pd.read_csv(LABEL_CSV, low_memory=False)
    selected_universe = pd.read_csv(BASE_OUTPUT_DIR / "selected_universe.csv", low_memory=False)
    scores = pd.read_csv(BASE_OUTPUT_DIR / "snapshot_contract_scores.csv", low_memory=False)

    selected, files, _type_map, _start_block, _end_block, _meta = eval3.build_fixed_universe(label_df)
    df_all = trp.load_events_from_files(files)
    trades = build_trade_panel(df_all)
    price_lookup = build_price_lookup(trades)
    market_features = build_window_market_feature_table(df_all=df_all, score_windows=scores)

    panel = build_agent_panel(
        scores=scores,
        selected_universe=selected_universe,
        label_df=label_df,
        price_lookup=price_lookup,
        market_features=market_features,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_DIR / "contract_window_panel.csv", index=False, encoding="utf-8-sig")
    write_jsonl(panel, OUTPUT_DIR / "contract_window_panel.jsonl")

    summary = pd.DataFrame(
        [
            {"key": "rows", "value": int(len(panel))},
            {"key": "contracts", "value": int(panel["contract_id"].nunique()) if not panel.empty else 0},
            {"key": "thresholds", "value": "|".join(str(x) for x in LOW_REP_THRESHOLD_LIST)},
            {"key": "rugpull_rows", "value": int((panel["collection_type"] == "rugpull").sum()) if not panel.empty else 0},
            {"key": "rows_with_next_trade", "value": int(panel["next_trade_exists"].sum()) if not panel.empty else 0},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "panel_meta.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(OUTPUT_DIR / 'contract_window_panel.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'contract_window_panel.jsonl').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'panel_meta.csv').resolve()}")


if __name__ == "__main__":
    main()
