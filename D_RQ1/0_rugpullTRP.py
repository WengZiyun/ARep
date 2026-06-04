#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import importlib.util
import random

import numpy as np
import pandas as pd
from scipy import sparse

# ==========================
# Config (edit in script)
# ==========================
COLLECTION_LABEL_CSV = Path("output/C/3_collectionlabel.csv")
USE_COLLECTION_LABEL = True
LABEL_FILTER_MODE = "collection_type"  # collection_type / selected_for_trust / all
LABEL_INCLUDE_TYPES = {"trust", "rugpull","unknown"}
LABEL_SOURCE_FIELD = "source_tx_file"
LABEL_REQUIRE_EXIST = True
COLLECTION_MERGE_MODE = "merge_once"

INPUT_DIR = Path("output/B/2_tranding")

WINDOW_MODE = "blocks"  # blocks / time / both
BLOCK_WINDOWS = []  # optional explicit [(start, end), ...]
BLOCK_END_BLOCKS = [19777901]  # when BLOCK_WINDOWS is empty, start uses selected-data min block
TIME_WINDOWS = []
TIMEZONE = "UTC"

# Optional per-type random sampling (only under LABEL_FILTER_MODE="collection_type").
# Set value >0 to sample N collections for that type.
LABEL_RANDOM_SAMPLES_BY_TYPE = {"trust": 20, "rugpull": 20, "unknown": 20}
LABEL_RANDOM_SEED = 20260226

STAGEA_TOP_K = 2
STAGEA_REFRESH_MODE = "auto_reuse"
STAGEA_REFRESH_HOURS = 24
STAGEA_REFRESH_BLOCKS = None
FORCE_STAGEA_REBUILD = False
SAVE_STAGEA_SNAPSHOT_ON_REBUILD = True

OUTPUT_DIR = Path("output/D/0_rugpullTRP")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stageA = _load_module(Path("C_reputation/0_stageA2.py"), "stageA_mod")
stageB = _load_module(Path("C_reputation/1_stageB.py"), "stageB_mod")
stageC = _load_module(Path("C_reputation/2_stageC.py"), "stageC_mod")


def select_input_files_from_label() -> tuple[list[Path], dict, pd.DataFrame]:
    df = pd.read_csv(COLLECTION_LABEL_CSV)
    if LABEL_SOURCE_FIELD not in df.columns:
        raise ValueError(f"Missing {LABEL_SOURCE_FIELD} in {COLLECTION_LABEL_CSV}")
    if LABEL_FILTER_MODE == "collection_type":
        keep = {str(x).strip().lower() for x in LABEL_INCLUDE_TYPES}
        work = df[df["collection_type"].astype(str).str.lower().isin(keep)].copy()
    elif LABEL_FILTER_MODE == "selected_for_trust":
        work = df[pd.to_numeric(df["selected_for_trust"], errors="coerce").fillna(0).astype(int) == 1].copy()
    elif LABEL_FILTER_MODE == "all":
        work = df.copy()
    else:
        raise ValueError("LABEL_FILTER_MODE must be collection_type / selected_for_trust / all")
    if work.empty:
        raise ValueError("No collections selected from 3_collectionlabel.csv")

    work["collection_type"] = work["collection_type"].astype(str).str.lower()
    rng = random.Random(LABEL_RANDOM_SEED)
    sampled_parts = []
    sampled_info = []
    for t in sorted(set(work["collection_type"].tolist())):
        sub = work[work["collection_type"] == t].copy()
        n_cfg = int(LABEL_RANDOM_SAMPLES_BY_TYPE.get(t, 0))
        if LABEL_FILTER_MODE == "collection_type" and n_cfg > 0:
            n_take = min(n_cfg, len(sub))
            idx = list(sub.index)
            rng.shuffle(idx)
            sub = sub.loc[idx[:n_take]].copy()
            sampled_info.append(f"{t}:{n_take}/{len(work[work['collection_type']==t])}")
        sampled_parts.append(sub)
    if sampled_parts:
        work = pd.concat(sampled_parts, ignore_index=True)
    if work.empty:
        raise ValueError("No collections remain after random sampling.")

    files = [Path(p) for p in work[LABEL_SOURCE_FIELD].astype(str).str.strip().unique().tolist() if str(p).strip()]
    miss = [p for p in files if not p.exists()]
    if miss and LABEL_REQUIRE_EXIST:
        raise FileNotFoundError(f"Missing source files (sample): {', '.join(str(x) for x in miss[:5])}")
    files = [p for p in files if p.exists()]
    type_map = work[["contract_address", "collection_type"]].copy() if "contract_address" in work.columns else pd.DataFrame(columns=["contract_address", "collection_type"])
    type_map["contract_address"] = type_map["contract_address"].astype(str).str.lower()
    type_map = type_map.drop_duplicates(subset=["contract_address"], keep="first").reset_index(drop=True)

    return files, {
        "label_filter_mode": LABEL_FILTER_MODE,
        "label_include_types": "|".join(sorted(LABEL_INCLUDE_TYPES)),
        "label_selected_collections": int(work["contract_address"].nunique()) if "contract_address" in work.columns else int(len(work)),
        "label_selected_rows": int(len(work)),
        "label_selected_files": int(len(files)),
        "label_source_field": LABEL_SOURCE_FIELD,
        "label_random_sampling": ";".join(sampled_info),
    }, type_map


def load_events_from_files(files: list[Path]) -> pd.DataFrame:
    # Use low_memory=False to avoid mixed-type chunk inference warnings on wide CSVs.
    frames = [pd.read_csv(p, low_memory=False) for p in files]
    df = pd.concat(frames, ignore_index=True)
    req = stageC.REQUIRED_COLS
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in merged input: {miss}")
    cols = req + (["tx_index_in_block"] if "tx_index_in_block" in df.columns else [])
    out = df[cols].copy()
    if "tx_index_in_block" not in out.columns:
        out["tx_index_in_block"] = 0
    out = out.rename(columns={"price_usd": "price", "gas_usd": "gas"})
    out["tx_type"] = out["tx_type"].astype(str).str.lower().str.strip().replace({"sale": "trade"})
    keep_types = stageC.ACQUISITION_TYPES.union(stageC.NON_ACQUISITION_TYPES)
    out = out[out["tx_type"].isin(keep_types)].copy()
    out["block_number"] = pd.to_numeric(out["block_number"], errors="coerce").fillna(0).astype(np.int64)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True, format="mixed")
    out["tx_index_in_block"] = pd.to_numeric(out["tx_index_in_block"], errors="coerce").fillna(0).astype(np.int64)
    out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0.0).astype(float)
    out["gas"] = pd.to_numeric(out["gas"], errors="coerce").fillna(0.0).astype(float)
    out["token_id"] = out["token_id"].astype(str)
    out["contract_id"] = out["contract_id"].astype(str)
    out["contract_address"] = out["contract_address"].fillna("").astype(str).str.lower()
    out["buyer"] = out["buyer"].fillna("").astype(str).str.lower()
    out["seller"] = out["seller"].fillna("").astype(str).str.lower()
    out = out[out["buyer"] != ""].copy()
    return out.sort_values(["block_number", "tx_index_in_block"], kind="mergesort").reset_index(drop=True)


def load_events() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    if USE_COLLECTION_LABEL:
        if COLLECTION_MERGE_MODE != "merge_once":
            raise ValueError("Only COLLECTION_MERGE_MODE='merge_once' supported")
        files, meta, type_map = select_input_files_from_label()
    else:
        files = sorted(INPUT_DIR.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No csv in {INPUT_DIR.resolve()}")
        meta = {
            "label_filter_mode": "disabled", "label_include_types": "", "label_selected_collections": 0,
            "label_selected_rows": 0, "label_selected_files": int(len(files)), "label_source_field": "input_dir",
            "label_random_sampling": "",
        }
        type_map = pd.DataFrame(columns=["contract_address", "collection_type"])
    return load_events_from_files(files), meta, type_map


def nearest_block_for_time(df_ts: pd.DataFrame, ts: pd.Timestamp) -> int:
    s = df_ts["timestamp"]
    if ts <= s.iloc[0]:
        return int(df_ts["block_number"].iloc[0])
    if ts >= s.iloc[-1]:
        return int(df_ts["block_number"].iloc[-1])
    i = int(s.searchsorted(ts, side="left"))
    l, r = i - 1, i
    dl = abs((ts - s.iloc[l]).total_seconds())
    dr = abs((s.iloc[r] - ts).total_seconds())
    return int(df_ts["block_number"].iloc[l if dl <= dr else r])


def resolve_windows(df: pd.DataFrame) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start_auto = int(df["block_number"].min())
    if WINDOW_MODE in {"blocks", "both"}:
        if BLOCK_WINDOWS:
            for s, e in BLOCK_WINDOWS:
                if int(s) <= int(e):
                    out.append((int(s), int(e)))
                else:
                    print(f"[Window] skip invalid block window ({s},{e})")
        else:
            for e in BLOCK_END_BLOCKS:
                end_block = int(e)
                if start_auto <= end_block:
                    out.append((start_auto, end_block))
                else:
                    print(f"[Window] skip invalid end block {end_block} (< start_auto {start_auto})")
    if WINDOW_MODE in {"time", "both"}:
        ts_df = df[["timestamp", "block_number"]].dropna().sort_values(["timestamp", "block_number"], kind="mergesort")
        for s, e in TIME_WINDOWS:
            sb = nearest_block_for_time(ts_df, pd.Timestamp(s, tz=TIMEZONE))
            eb = nearest_block_for_time(ts_df, pd.Timestamp(e, tz=TIMEZONE))
            if sb <= eb:
                out.append((sb, eb))
            else:
                print(f"[Window] skip invalid time window {s}~{e} -> ({sb},{eb})")
    if not out:
        out = [(int(df["block_number"].min()), int(df["block_number"].max()))]
    return sorted(set(out), key=lambda x: (x[1], x[0]))


def block_to_time(df: pd.DataFrame, block_number: int):
    s = df[df["block_number"] <= int(block_number)]["timestamp"]
    return None if s.empty else s.max()


def run_stage_a(df_hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Keep A consistent with original 0_stageA2.py: mint/trade only (sale already normalized to trade).
    df_ab = df_hist[df_hist["tx_type"].isin(["mint", "trade"])].copy()
    dfa = stageA._winsorize(df_ab, stageA.WINSOR_Q if hasattr(stageA, "WINSOR_Q") else 0.0)
    W, _G, _V, users, contracts = stageA.build_W_hist_GV(
        dfa, mu=stageA.MU, w_gas=stageA.W_GAS, w_val=stageA.W_VAL, include_mint_gas=stageA.INCLUDE_MINT_GAS
    )
    uA, cA = stageA.stageA_propagation(W, eta=stageA.ETA, max_iter=stageA.MAX_ITER, tol=stageA.TOL)
    cdf = pd.DataFrame({"contract_id": contracts, "cA": cA}).sort_values(["cA", "contract_id"], ascending=[False, True]).reset_index(drop=True)
    cdf["rank_A"] = np.arange(1, len(cdf) + 1, dtype=np.int64)
    udf = pd.DataFrame({"user": users, "uA": uA}).sort_values(["uA", "user"], ascending=[False, True]).reset_index(drop=True)
    udf["rank_A_user"] = np.arange(1, len(udf) + 1, dtype=np.int64)
    return cdf, udf


def run_stage_b(df_hist: pd.DataFrame, stagea_contract: pd.DataFrame, stagea_user: pd.DataFrame) -> pd.DataFrame:
    # Keep B consistent with original 1_stageB.py: mint/trade only.
    df_ab = df_hist[df_hist["tx_type"].isin(["mint", "trade"])].copy()
    m = len(stagea_contract)
    if m == 0:
        return stagea_user.assign(p0=0.0, u0=0.0, rank_B=0, event_cnt_anchor=0)[["user", "p0", "u0", "rank_B", "event_cnt_anchor"]]
    k = max(1, min(int(STAGEA_TOP_K), m))
    anchors = stagea_contract.head(k).copy()
    denom = float(anchors["cA"].sum())
    anchors["cA_tilde"] = (anchors["cA"] / denom) if denom > 0 else (1.0 / len(anchors))
    c_map = dict(zip(anchors["contract_id"].astype(str), anchors["cA_tilde"].astype(float)))
    ev = stageB.build_stageb_events(df_ab, set(anchors["contract_id"].astype(str)))
    prior, _ = stageB.compute_stageb_user_prior(
        ev=ev, cA_tilde_map=c_map, tau_s=stageB.TAU_S, tau_h=stageB.TAU_H,
        n_ref=stageB.N_REF, eps=stageB.EPS, require_positive_cost=stageB.REQUIRE_POSITIVE_COST
    )
    full = stagea_user[["user"]].drop_duplicates().merge(
        prior[["user", "p0", "u0", "rank_B", "event_cnt_anchor"]], on="user", how="left"
    )
    full["p0"] = full["p0"].fillna(0.0)
    full["u0"] = full["u0"].fillna(0.0)
    full["event_cnt_anchor"] = full["event_cnt_anchor"].fillna(0).astype(int)
    nz = full["u0"] > 0
    full.loc[nz, "rank_B"] = full.loc[nz, "u0"].rank(method="first", ascending=False).astype(int)
    full.loc[~nz, "rank_B"] = 0
    return full[["user", "p0", "u0", "rank_B", "event_cnt_anchor"]].sort_values(["rank_B", "user"]).reset_index(drop=True)


def align_prior(keys: np.ndarray, df: pd.DataFrame, key_col: str, val_col: str) -> np.ndarray:
    if len(keys) == 0:
        return np.array([], dtype=np.float64)
    if df.empty or key_col not in df.columns or val_col not in df.columns:
        return np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
    m = dict(zip(df[key_col].astype(str), pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)))
    arr = np.array([float(m.get(str(k), 0.0)) for k in keys], dtype=np.float64)
    s = float(arr.sum())
    return (arr / s) if s > 0 else np.full(len(keys), 1.0 / len(keys), dtype=np.float64)


def run_stage_c(
    df_window: pd.DataFrame,
    u0_full: pd.DataFrame,
    stagea_contract: pd.DataFrame,
    tau_end: int,
    contract_type_map: pd.DataFrame,
):
    ev, n_ref_val = stageC.build_stagec_event_table(df_window, tau_s=stageC.TAU_S, tau_h=stageC.TAU_H, n_ref=stageC.N_REF, eps=stageC.EPS, tau_end=tau_end)
    W_rec, users, contracts = stageC.build_wrec_matrix(ev)
    P_u2c = stageC.row_stochastic(W_rec)
    P_c2u = stageC.row_stochastic(W_rec.T.tocsr())
    u0 = align_prior(users, u0_full, "user", "u0")
    c0 = align_prior(contracts, stagea_contract, "contract_id", "cA")
    uC, cC, iters, delta = stageC.stagec_iterate(
        P_u2c=P_u2c, P_c2u=P_c2u, u0=u0, c0=c0,
        alpha=stageC.ALPHA, beta=stageC.BETA, max_iter=stageC.MAX_ITER, tol=stageC.TOL
    )
    users_out = pd.DataFrame({"user": users, "uC": uC}).sort_values(["uC", "user"], ascending=[False, True]).reset_index(drop=True)
    users_out["rank_C_user"] = np.arange(1, len(users_out) + 1, dtype=np.int64)
    contracts_out = pd.DataFrame({"contract_id": contracts, "cC": cC}).sort_values(["cC", "contract_id"], ascending=[False, True]).reset_index(drop=True)
    contracts_out["rank_C"] = np.arange(1, len(contracts_out) + 1, dtype=np.int64)
    contracts_out = contracts_out.merge(stageC.compute_contract_address_map(df_window), on="contract_id", how="left")
    if not contract_type_map.empty:
        ctm = contract_type_map.rename(columns={"contract_address": "contract_address_lc", "collection_type": "collection_type"}).copy()
        contracts_out["contract_address_lc"] = contracts_out["contract_address"].fillna("").astype(str).str.lower()
        contracts_out = contracts_out.merge(ctm[["contract_address_lc", "collection_type"]], on="contract_address_lc", how="left")
        contracts_out = contracts_out.drop(columns=["contract_address_lc"])
    else:
        contracts_out["collection_type"] = np.nan
    contracts_out = contracts_out.merge(stageC.compute_contract_stats(df_window), on="contract_id", how="left")
    return contracts_out, users_out, {"iters_used": int(iters), "final_delta": float(delta), "n_ref": int(n_ref_val), "events": int(len(ev)), "nnz_wrec": int(W_rec.nnz)}


def should_rebuild(cache: dict | None, end_block: int, end_time):
    if FORCE_STAGEA_REBUILD:
        return True, "force"
    if cache is None:
        return True, "first_run"
    if STAGEA_REFRESH_MODE != "auto_reuse":
        return False, "reuse"
    if STAGEA_REFRESH_BLOCKS is not None and int(end_block) - int(cache["end_block"]) >= int(STAGEA_REFRESH_BLOCKS):
        return True, "by_blocks"
    if STAGEA_REFRESH_HOURS is not None and end_time is not None and cache.get("end_time") is not None:
        if (end_time - cache["end_time"]).total_seconds() >= float(STAGEA_REFRESH_HOURS) * 3600.0:
            return True, "by_hours"
    return False, "reuse"


def save_outputs(end_block: int, contracts_out: pd.DataFrame, users_out: pd.DataFrame, meta: dict, stagea_snapshot: pd.DataFrame | None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfp = OUTPUT_DIR / f"contracts_{end_block}.csv"
    ufp = OUTPUT_DIR / f"users_{end_block}.csv"
    mfp = OUTPUT_DIR / f"meta_{end_block}.csv"
    contracts_out.to_csv(cfp, index=False, encoding="utf-8-sig")
    users_out.to_csv(ufp, index=False, encoding="utf-8-sig")
    pd.DataFrame({"key": list(meta.keys()), "value": list(meta.values())}).to_csv(mfp, index=False, encoding="utf-8-sig")
    if stagea_snapshot is not None and SAVE_STAGEA_SNAPSHOT_ON_REBUILD:
        sfp = OUTPUT_DIR / f"stagea_{end_block}.csv"
        stagea_snapshot.to_csv(sfp, index=False, encoding="utf-8-sig")
    print(f"[Saved] {cfp.resolve()}")
    print(f"[Saved] {ufp.resolve()}")
    print(f"[Saved] {mfp.resolve()}")


def main():
    df_all, label_meta, contract_type_map = load_events()
    if df_all.empty:
        raise ValueError("No events after input selection")
    windows = resolve_windows(df_all)
    print(f"[Data] rows={len(df_all):,}, windows={len(windows)}")
    cache = None
    for i, (start_block, end_block) in enumerate(windows, 1):
        print(f"[Run] {i}/{len(windows)} [{start_block}, {end_block}]")
        df_hist = df_all[df_all["block_number"] <= end_block].copy()
        df_win = df_all[(df_all["block_number"] >= start_block) & (df_all["block_number"] <= end_block)].copy()
        if df_win.empty:
            print("[Run] skip empty window")
            continue
        end_time = block_to_time(df_all, end_block)
        rebuild, reason = should_rebuild(cache, end_block, end_time)
        stagea_snapshot = None
        if rebuild:
            stagea_contract, stagea_user = run_stage_a(df_hist)
            cache = {"end_block": int(end_block), "end_time": end_time, "stagea_contract": stagea_contract, "stagea_user": stagea_user}
            stagea_snapshot = stagea_contract.copy()
        stagea_contract = cache["stagea_contract"]
        stagea_user = cache["stagea_user"]
        u0_full = run_stage_b(df_hist, stagea_contract, stagea_user)
        contracts_out, users_out, cmeta = run_stage_c(
            df_win, u0_full, stagea_contract, tau_end=int(end_block), contract_type_map=contract_type_map
        )
        meta = {
            "start_block": int(start_block),
            "end_block": int(end_block),
            "stagea_rebuilt": int(1 if rebuild else 0),
            "stagea_rebuild_reason": reason,
            "label_filter_mode": label_meta["label_filter_mode"],
            "label_include_types": label_meta["label_include_types"],
            "label_selected_collections": label_meta["label_selected_collections"],
            "label_selected_files": label_meta["label_selected_files"],
            "label_random_sampling": label_meta["label_random_sampling"],
            "iters_used": cmeta["iters_used"],
            "final_delta": cmeta["final_delta"],
            "events": cmeta["events"],
            "nnz_wrec": cmeta["nnz_wrec"],
        }
        save_outputs(int(end_block), contracts_out, users_out, meta, stagea_snapshot)
    print("[Done]")


if __name__ == "__main__":
    main()
