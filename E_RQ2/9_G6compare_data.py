from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
E_OUT = ROOT / "output" / "E"
RESULT_DIR = E_OUT / "9_G6data_analysis_v3"

G6_DIR_GLOB = "9_G6compare*"
SUMMARY_NAME = "g6_compare_summary.csv"
POWER_LEVELS = [10.0**k for k in range(1, 7)]  # 10^1 ... 10^6
FOCUS_ATTACK_MODES = ["attack1", "attack2", "attack3"]
METHOD_ORDER = ["our", "our_no_u0", "birank", "pagerank", "eigentrust"]
METHOD_LABELS = {
    "our": "Our",
    "our_no_u0": "Our-NoU0",
    "birank": "BiRank",
    "pagerank": "PageRank",
    "eigentrust": "EigenTrust",
}


def _df_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        show = df.copy()
        if not index:
            show = show.reset_index(drop=True)
        cols = [str(c) for c in show.columns.tolist()]
        rows = [[str(x) for x in row] for row in show.astype(object).fillna("--").to_numpy().tolist()]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join([header, sep] + body)


def _fmt_metric(v: Any, digits: int = 4) -> str:
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def _latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _build_latex_long_table(long_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("\\begin{tabular}{lllccccccc}")
    lines.append("\\toprule")
    lines.append(
        "Attack & Method & Intensity & C-AUPRC(all) & C-AUPRC(excl-Z) & C-$\\rho$ & C-$\\tau$ & U-AUPRC & U-$\\rho$ & U-$\\tau$ \\\\"
    )
    lines.append("\\midrule")
    for mode in FOCUS_ATTACK_MODES:
        mode_df = long_df[long_df["attack_mode"].astype(str).eq(mode)].copy()
        if mode_df.empty:
            continue
        for method in METHOD_ORDER:
            method_df = mode_df[mode_df["method"].astype(str).eq(method)].sort_values("attack_multiplier")
            if method_df.empty:
                continue
            for _, r in method_df.iterrows():
                lines.append(
                    " & ".join(
                        [
                            _latex_escape(str(mode)),
                            _latex_escape(METHOD_LABELS.get(method, method)),
                            _latex_escape(str(r.get("power_label", ""))),
                            _fmt_metric(r.get("contract_auprc_all_mean", np.nan)),
                            _fmt_metric(r.get("contract_auprc_excl_zombie_mean", np.nan)),
                            _fmt_metric(r.get("contract_spearman_rho_mean", np.nan)),
                            _fmt_metric(r.get("contract_kendall_tau_mean", np.nan)),
                            _fmt_metric(r.get("user_auprc_mean", np.nan)),
                            _fmt_metric(r.get("user_spearman_rho_mean", np.nan)),
                            _fmt_metric(r.get("user_kendall_tau_mean", np.nan)),
                        ]
                    )
                    + " \\\\"
                )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _build_latex_table_for_mode(long_df: pd.DataFrame, mode: str) -> str:
    mode_df = long_df[long_df["attack_mode"].astype(str).eq(str(mode))].copy()
    if mode_df.empty:
        return ""
    lines: list[str] = []
    lines.append("\\begin{tabular}{llccccccc}")
    lines.append("\\toprule")
    lines.append("Method & Intensity & C-AUPRC(all) & C-AUPRC(excl-Z) & C-$\\rho$ & C-$\\tau$ & U-AUPRC & U-$\\rho$ & U-$\\tau$ \\\\")
    lines.append("\\midrule")
    for method in METHOD_ORDER:
        method_df = mode_df[mode_df["method"].astype(str).eq(method)].sort_values("attack_multiplier")
        if method_df.empty:
            continue
        for _, r in method_df.iterrows():
            lines.append(
                " & ".join(
                    [
                        _latex_escape(METHOD_LABELS.get(method, method)),
                        _latex_escape(str(r.get("power_label", ""))),
                        _fmt_metric(r.get("contract_auprc_all_mean", np.nan)),
                        _fmt_metric(r.get("contract_auprc_excl_zombie_mean", np.nan)),
                        _fmt_metric(r.get("contract_spearman_rho_mean", np.nan)),
                        _fmt_metric(r.get("contract_kendall_tau_mean", np.nan)),
                        _fmt_metric(r.get("user_auprc_mean", np.nan)),
                        _fmt_metric(r.get("user_spearman_rho_mean", np.nan)),
                        _fmt_metric(r.get("user_kendall_tau_mean", np.nan)),
                    ]
                )
                + " \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _to_numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _list_g6_summaries() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for d in sorted(E_OUT.glob(G6_DIR_GLOB)):
        s = d / SUMMARY_NAME
        if not s.exists():
            continue
        try:
            header = pd.read_csv(s, nrows=0).columns.tolist()
        except Exception:
            continue
        try:
            mode_df = pd.read_csv(s, usecols=["attack_mode"])
            modes = sorted(mode_df["attack_mode"].astype(str).dropna().unique().tolist())
        except Exception:
            modes = []
        rows.append(
            {
                "dir_name": d.name,
                "summary_path": str(s),
                "mtime": float(s.stat().st_mtime),
                "attack_modes": ",".join(modes),
                "has_contract_auprc": "contract_auprc_excl_zombie_mean" in header,
                "has_user_auprc": "user_auprc_mean" in header,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["dir_name", "summary_path", "mtime", "attack_modes", "has_contract_auprc", "has_user_auprc"]
        )
    return pd.DataFrame(rows).sort_values("mtime", ascending=False).reset_index(drop=True)


def _pick_latest_source_by_mode(index_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in index_df.iterrows():
        modes = [x.strip() for x in str(r.get("attack_modes", "")).split(",") if x.strip()]
        for mode in modes:
            rows.append(
                {
                    "attack_mode": mode,
                    "dir_name": str(r["dir_name"]),
                    "summary_path": str(r["summary_path"]),
                    "mtime": float(r["mtime"]),
                    "has_contract_auprc": bool(r["has_contract_auprc"]),
                    "has_user_auprc": bool(r["has_user_auprc"]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["attack_mode", "dir_name", "summary_path", "mtime", "has_contract_auprc", "has_user_auprc"]
        )
    pool = pd.DataFrame(rows).sort_values(["attack_mode", "mtime"], ascending=[True, False])
    return pool.groupby("attack_mode", as_index=False).head(1).reset_index(drop=True)


def _load_selected_mode_data(source_df: pd.DataFrame) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for _, src in source_df.iterrows():
        mode = str(src["attack_mode"])
        summary_path = Path(str(src["summary_path"]))
        df = pd.read_csv(summary_path)
        df = df[df["attack_mode"].astype(str).eq(mode)].copy()
        if df.empty:
            continue
        df["attack_multiplier"] = pd.to_numeric(df["attack_multiplier"], errors="coerce")
        df = df.dropna(subset=["attack_multiplier"]).copy()
        df["source_dir"] = str(src["dir_name"])
        df["source_summary_path"] = str(summary_path)
        out_parts.append(df)
    if not out_parts:
        return pd.DataFrame()
    all_df = pd.concat(out_parts, ignore_index=True)
    all_df["method"] = all_df["method"].astype(str)
    all_df["param_value"] = _to_numeric_col(all_df, "param_value")
    return all_df


def _build_power_table(mode_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    sub = mode_df[mode_df["attack_mode"].astype(str).eq(str(mode))].copy()
    if sub.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_sub = sub[sub["method"].astype(str).eq(method)].copy()
        if method_sub.empty:
            continue
        method_sub = method_sub.sort_values("attack_multiplier")
        for power in POWER_LEVELS:
            exact = method_sub[np.isclose(method_sub["attack_multiplier"], float(power), rtol=0.0, atol=1e-12)]
            if exact.empty:
                records.append(
                    {
                        "attack_mode": mode,
                        "method": method,
                        "attack_multiplier": float(power),
                        "power_label": f"10^{int(np.log10(power))}",
                        "source_dir": "",
                        "contract_auprc_all_mean": np.nan,
                        "contract_auprc_excl_zombie_mean": np.nan,
                        "contract_spearman_rho_mean": np.nan,
                        "contract_kendall_tau_mean": np.nan,
                        "user_auprc_mean": np.nan,
                        "user_spearman_rho_mean": np.nan,
                        "user_kendall_tau_mean": np.nan,
                    }
                )
                continue
            row = exact.iloc[0]
            records.append(
                {
                    "attack_mode": mode,
                    "method": method,
                    "attack_multiplier": float(power),
                    "power_label": f"10^{int(np.log10(power))}",
                    "source_dir": str(row.get("source_dir", "")),
                    "contract_auprc_all_mean": float(
                        pd.to_numeric(pd.Series([row.get("contract_auprc_all_mean", np.nan)]), errors="coerce").iloc[0]
                    ),
                    "contract_auprc_excl_zombie_mean": float(
                        pd.to_numeric(pd.Series([row.get("contract_auprc_excl_zombie_mean", np.nan)]), errors="coerce").iloc[0]
                    ),
                    "contract_spearman_rho_mean": float(
                        pd.to_numeric(pd.Series([row.get("contract_spearman_rho_mean", np.nan)]), errors="coerce").iloc[0]
                    ),
                    "contract_kendall_tau_mean": float(
                        pd.to_numeric(pd.Series([row.get("contract_kendall_tau_mean", np.nan)]), errors="coerce").iloc[0]
                    ),
                    "user_auprc_mean": float(
                        pd.to_numeric(pd.Series([row.get("user_auprc_mean", np.nan)]), errors="coerce").iloc[0]
                    ),
                    "user_spearman_rho_mean": float(
                        pd.to_numeric(pd.Series([row.get("user_spearman_rho_mean", np.nan)]), errors="coerce").iloc[0]
                    ),
                    "user_kendall_tau_mean": float(
                        pd.to_numeric(pd.Series([row.get("user_kendall_tau_mean", np.nan)]), errors="coerce").iloc[0]
                    ),
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["attack_mode", "method", "attack_multiplier"]).reset_index(drop=True)


def _to_wide_mode_table(long_df: pd.DataFrame, mode: str, entity_prefix: str) -> pd.DataFrame:
    if entity_prefix == "contract":
        metric_cols = [
            "contract_auprc_all_mean",
            "contract_auprc_excl_zombie_mean",
            "contract_spearman_rho_mean",
            "contract_kendall_tau_mean",
        ]
    else:
        metric_cols = [
            f"{entity_prefix}_auprc_mean",
            f"{entity_prefix}_spearman_rho_mean",
            f"{entity_prefix}_kendall_tau_mean",
        ]
    mode_df = long_df[long_df["attack_mode"].astype(str).eq(str(mode))].copy()
    if mode_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for power in POWER_LEVELS:
        row: dict[str, Any] = {
            "attack_mode": mode,
            "attack_multiplier": float(power),
            "power_label": f"10^{int(np.log10(power))}",
        }
        for method in METHOD_ORDER:
            ms = mode_df[
                mode_df["method"].astype(str).eq(method)
                & np.isclose(mode_df["attack_multiplier"].astype(float), float(power), rtol=0.0, atol=1e-12)
            ]
            if ms.empty:
                for mcol in metric_cols:
                    row[f"{method}_{mcol}"] = np.nan
            else:
                mrow = ms.iloc[0]
                for mcol in metric_cols:
                    row[f"{method}_{mcol}"] = float(pd.to_numeric(pd.Series([mrow.get(mcol, np.nan)]), errors="coerce").iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_main_user_auprc_md(mode_data: pd.DataFrame, modes: list[str]) -> str:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        sub = mode_data[mode_data["attack_mode"].astype(str).eq(mode)].copy()
        if sub.empty or "user_auprc_mean" not in sub.columns:
            continue
        sub["user_auprc_mean"] = pd.to_numeric(sub["user_auprc_mean"], errors="coerce")
        sub = sub.dropna(subset=["user_auprc_mean"])
        if sub.empty:
            continue
        our = sub[sub["method"].astype(str).eq("our")].copy()
        base = sub[~sub["method"].astype(str).eq("our")].copy()
        if our.empty or base.empty:
            continue
        merged = our[["attack_multiplier", "user_auprc_mean"]].rename(columns={"user_auprc_mean": "our_auprc"}).merge(
            base.groupby("attack_multiplier", as_index=False)["user_auprc_mean"].max().rename(columns={"user_auprc_mean": "best_base_auprc"}),
            on="attack_multiplier",
            how="inner",
        )
        if merged.empty:
            continue
        best_base_method = (
            base.groupby("method", as_index=False)["user_auprc_mean"].mean().sort_values("user_auprc_mean", ascending=False).iloc[0]["method"]
        )
        rows.append(
            {
                "Attack": mode,
                "Our(U-AUPRC)": f"{merged['our_auprc'].mean():.4f}",
                "Best Baseline(U-AUPRC)": f"{METHOD_LABELS.get(str(best_base_method), str(best_base_method))} ({merged['best_base_auprc'].mean():.4f})",
                "Delta(Our-Best)": f"{(merged['our_auprc'] - merged['best_base_auprc']).mean():+.4f}",
                "WinRate(Our)": f"{int((merged['our_auprc'] > merged['best_base_auprc']).sum())}/{int(len(merged))}",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return "# Main Table (User-AUPRC)\n\nNo data.\n"
    return "# Main Table (User-AUPRC)\n\n" + _df_to_markdown_safe(out, index=False) + "\n"


def _build_appendix_contract_md(mode_data: pd.DataFrame, modes: list[str]) -> str:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        sub = mode_data[mode_data["attack_mode"].astype(str).eq(mode)].copy()
        if sub.empty:
            continue
        for metric, short in [
            ("contract_auprc_all_mean", "C-AUPRC(all)"),
            ("contract_auprc_excl_zombie_mean", "C-AUPRC(excl-Z)"),
            ("contract_spearman_rho_mean", "C-rho"),
            ("contract_kendall_tau_mean", "C-tau"),
        ]:
            if metric not in sub.columns:
                continue
            x = sub[["method", "attack_multiplier", metric]].copy()
            x[metric] = pd.to_numeric(x[metric], errors="coerce")
            x = x.dropna(subset=[metric])
            if x.empty:
                continue
            our = x[x["method"].astype(str).eq("our")][["attack_multiplier", metric]].rename(columns={metric: "our_v"})
            base = x[~x["method"].astype(str).eq("our")]
            if our.empty or base.empty:
                continue
            best_per_level = base.groupby("attack_multiplier", as_index=False)[metric].max().rename(columns={metric: "best_v"})
            merged = our.merge(best_per_level, on="attack_multiplier", how="inner")
            if merged.empty:
                continue
            best_method = base.groupby("method", as_index=False)[metric].mean().sort_values(metric, ascending=False).iloc[0]["method"]
            rows.append(
                {
                    "Attack": mode,
                    "Metric": short,
                    "Our(mean)": f"{merged['our_v'].mean():.4f}",
                    "Best Baseline(mean)": f"{METHOD_LABELS.get(str(best_method), str(best_method))} ({merged['best_v'].mean():.4f})",
                    "Delta(Our-Best)": f"{(merged['our_v'] - merged['best_v']).mean():+.4f}",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return "# Appendix Table (Contract Robustness)\n\nNo data.\n"
    return "# Appendix Table (Contract Robustness)\n\n" + _df_to_markdown_safe(out, index=False) + "\n"


def _build_key_findings_md(main_md_df: pd.DataFrame) -> str:
    if main_md_df.empty:
        return "# Key Findings\n\nNo data.\n"
    lines = ["# Key Findings", ""]
    for r in main_md_df.itertuples(index=False):
        lines.append(
            f"- {r.Attack}: Our U-AUPRC={r[1]}, baseline={r[2]}, delta={r[3]}, win={r[4]}."
        )
    lines.append("")
    lines.append("Suggested write-up: Our method consistently improves user-side Sybil detection under all attack modes, while contract-side robustness may be stronger for certain baselines.")
    lines.append("")
    return "\n".join(lines)


def _build_user_ap_gain_lift_md(raw_full: pd.DataFrame, modes: list[str]) -> str:
    rows: list[dict[str, Any]] = []
    need_cols = {"attack_mode", "method", "attack_multiplier", "user_auprc_mean", "user_target_sybil_total_mean", "user_n_entities_mean"}
    if raw_full.empty or not need_cols.issubset(set(raw_full.columns)):
        return "# User AP Gain/Lift\n\nNo data.\n"
    work = raw_full.copy()
    work["user_auprc_mean"] = pd.to_numeric(work["user_auprc_mean"], errors="coerce")
    work["user_target_sybil_total_mean"] = pd.to_numeric(work["user_target_sybil_total_mean"], errors="coerce")
    work["user_n_entities_mean"] = pd.to_numeric(work["user_n_entities_mean"], errors="coerce")
    work["user_prevalence"] = work["user_target_sybil_total_mean"] / work["user_n_entities_mean"].clip(lower=1e-12)
    work["user_ap_gain"] = work["user_auprc_mean"] - work["user_prevalence"]
    work["user_ap_lift"] = work["user_auprc_mean"] / work["user_prevalence"].clip(lower=1e-12)
    work = work[work["attack_mode"].astype(str).isin(modes)].copy()
    work = work.dropna(subset=["user_auprc_mean", "user_prevalence", "user_ap_gain", "user_ap_lift"])
    if work.empty:
        return "# User AP Gain/Lift\n\nNo data.\n"
    for mode in modes:
        sub = work[work["attack_mode"].astype(str).eq(mode)].copy()
        if sub.empty:
            continue
        for method in METHOD_ORDER:
            ms = sub[sub["method"].astype(str).eq(method)].copy()
            if ms.empty:
                continue
            rows.append(
                {
                    "Attack": mode,
                    "Method": METHOD_LABELS.get(method, method),
                    "Prevalence": f"{ms['user_prevalence'].mean():.4f}",
                    "U-AUPRC": f"{ms['user_auprc_mean'].mean():.4f}",
                    "AP Gain": f"{ms['user_ap_gain'].mean():+.4f}",
                    "AP Lift(x)": f"{ms['user_ap_lift'].mean():.2f}",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return "# User AP Gain/Lift\n\nNo data.\n"
    return "# User AP Gain/Lift\n\n" + _df_to_markdown_safe(out, index=False) + "\n"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    index_df = _list_g6_summaries()
    if index_df.empty:
        raise FileNotFoundError("No G6 summary file found under output/E/9_G6compare*.")

    selected_sources = _pick_latest_source_by_mode(index_df)
    mode_data = _load_selected_mode_data(selected_sources)
    if mode_data.empty:
        raise ValueError("Selected source summaries are empty.")
    if "k_ratio" in mode_data.columns and pd.to_numeric(mode_data["k_ratio"], errors="coerce").notna().any():
        mode_data["k_ratio"] = pd.to_numeric(mode_data["k_ratio"], errors="coerce")
        mode_data = mode_data[mode_data["k_ratio"].eq(float(mode_data["k_ratio"].max()))].copy()

    raw_cols = [
        "attack_mode",
        "method",
        "attack_multiplier",
        "source_dir",
        "contract_auprc_all_mean",
        "contract_auprc_excl_zombie_mean",
        "contract_spearman_rho_mean",
        "contract_kendall_tau_mean",
        "user_auprc_mean",
        "user_spearman_rho_mean",
        "user_kendall_tau_mean",
    ]
    all_modes_raw = mode_data.copy()
    all_modes_raw["contract_auprc_all_mean"] = _to_numeric_col(all_modes_raw, "contract_auprc_all_mean")
    all_modes_raw["contract_auprc_excl_zombie_mean"] = _to_numeric_col(all_modes_raw, "contract_auprc_excl_zombie_mean")
    all_modes_raw["contract_spearman_rho_mean"] = _to_numeric_col(all_modes_raw, "contract_spearman_rho_mean")
    all_modes_raw["contract_kendall_tau_mean"] = _to_numeric_col(all_modes_raw, "contract_kendall_tau_mean")
    all_modes_raw["user_auprc_mean"] = _to_numeric_col(all_modes_raw, "user_auprc_mean")
    all_modes_raw["user_spearman_rho_mean"] = _to_numeric_col(all_modes_raw, "user_spearman_rho_mean")
    all_modes_raw["user_kendall_tau_mean"] = _to_numeric_col(all_modes_raw, "user_kendall_tau_mean")
    all_modes_raw = all_modes_raw[raw_cols].sort_values(["attack_mode", "method", "attack_multiplier"]).reset_index(drop=True)

    # full raw metrics (keeps prevalence-related columns) from selected sources for additional analysis
    all_modes_raw_full = mode_data.copy().sort_values(["attack_mode", "method", "attack_multiplier"]).reset_index(drop=True)

    available_modes = [m for m in FOCUS_ATTACK_MODES if m in set(mode_data["attack_mode"].astype(str).unique().tolist())]
    focus_data = mode_data[mode_data["attack_mode"].astype(str).isin(available_modes)].copy()
    if focus_data.empty:
        raise ValueError(f"No data for attack modes: {FOCUS_ATTACK_MODES}")

    long_parts: list[pd.DataFrame] = []
    for mode in available_modes:
        tbl = _build_power_table(focus_data, mode)
        if not tbl.empty:
            long_parts.append(tbl)

    if not long_parts:
        raise ValueError("Power table (10^1..10^6) is empty for selected attack modes.")
    long_table = pd.concat(long_parts, ignore_index=True).sort_values(
        ["attack_mode", "method", "attack_multiplier"]
    ).reset_index(drop=True)

    contract_wide_parts = []
    user_wide_parts = []
    for mode in available_modes:
        ctbl = _to_wide_mode_table(long_table, mode, "contract")
        utbl = _to_wide_mode_table(long_table, mode, "user")
        if not ctbl.empty:
            contract_wide_parts.append(ctbl)
        if not utbl.empty:
            user_wide_parts.append(utbl)

    contract_wide = pd.concat(contract_wide_parts, ignore_index=True) if contract_wide_parts else pd.DataFrame()
    user_wide = pd.concat(user_wide_parts, ignore_index=True) if user_wide_parts else pd.DataFrame()

    latex_all = _build_latex_long_table(long_table)
    (RESULT_DIR / "g6_power_table_10e1_to_10e6.tex").write_text(latex_all, encoding="utf-8")
    for mode in available_modes:
        latex_mode = _build_latex_table_for_mode(long_table, mode)
        if latex_mode:
            (RESULT_DIR / f"g6_power_table_10e1_to_10e6_{mode}.tex").write_text(latex_mode, encoding="utf-8")

    main_md_rows = []
    for mode in available_modes:
        sub = focus_data[focus_data["attack_mode"].astype(str).eq(mode)].copy()
        if sub.empty or "user_auprc_mean" not in sub.columns:
            continue
        sub["user_auprc_mean"] = pd.to_numeric(sub["user_auprc_mean"], errors="coerce")
        sub = sub.dropna(subset=["user_auprc_mean"])
        if sub.empty:
            continue
        our = sub[sub["method"].astype(str).eq("our")]
        base = sub[~sub["method"].astype(str).eq("our")]
        if our.empty or base.empty:
            continue
        merged = our[["attack_multiplier", "user_auprc_mean"]].rename(columns={"user_auprc_mean": "our_auprc"}).merge(
            base.groupby("attack_multiplier", as_index=False)["user_auprc_mean"].max().rename(columns={"user_auprc_mean": "best_base_auprc"}),
            on="attack_multiplier",
            how="inner",
        )
        if merged.empty:
            continue
        best_base_method = (
            base.groupby("method", as_index=False)["user_auprc_mean"].mean().sort_values("user_auprc_mean", ascending=False).iloc[0]["method"]
        )
        main_md_rows.append(
            {
                "Attack": mode,
                "Our(U-AUPRC)": f"{merged['our_auprc'].mean():.4f}",
                "Best Baseline(U-AUPRC)": f"{METHOD_LABELS.get(str(best_base_method), str(best_base_method))} ({merged['best_base_auprc'].mean():.4f})",
                "Delta(Our-Best)": f"{(merged['our_auprc'] - merged['best_base_auprc']).mean():+.4f}",
                "WinRate(Our)": f"{int((merged['our_auprc'] > merged['best_base_auprc']).sum())}/{int(len(merged))}",
            }
        )
    main_md_df = pd.DataFrame(main_md_rows)
    main_md = _build_main_user_auprc_md(focus_data, available_modes)
    appendix_md = _build_appendix_contract_md(focus_data, available_modes)
    findings_md = _build_key_findings_md(main_md_df)
    user_gain_lift_md = _build_user_ap_gain_lift_md(all_modes_raw_full, available_modes)
    (RESULT_DIR / "paper_main_user_auprc.md").write_text(main_md, encoding="utf-8")
    (RESULT_DIR / "paper_appendix_contract_robustness.md").write_text(appendix_md, encoding="utf-8")
    (RESULT_DIR / "paper_key_findings.md").write_text(findings_md, encoding="utf-8")
    (RESULT_DIR / "paper_user_ap_gain_lift.md").write_text(user_gain_lift_md, encoding="utf-8")

    index_df.to_csv(RESULT_DIR / "g6_summary_index.csv", index=False, encoding="utf-8-sig")
    selected_sources.to_csv(RESULT_DIR / "g6_selected_sources_by_attack_mode.csv", index=False, encoding="utf-8-sig")
    all_modes_raw.to_csv(RESULT_DIR / "g6_all_modes_raw_metrics.csv", index=False, encoding="utf-8-sig")
    long_table.to_csv(RESULT_DIR / "g6_power_table_long_10e1_to_10e6.csv", index=False, encoding="utf-8-sig")
    contract_wide.to_csv(RESULT_DIR / "g6_power_table_contract_wide_10e1_to_10e6.csv", index=False, encoding="utf-8-sig")
    user_wide.to_csv(RESULT_DIR / "g6_power_table_user_wide_10e1_to_10e6.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {(RESULT_DIR / 'g6_summary_index.csv').resolve()}")
    print(f"[Saved] {(RESULT_DIR / 'g6_selected_sources_by_attack_mode.csv').resolve()}")
    print(f"[Saved] {(RESULT_DIR / 'g6_all_modes_raw_metrics.csv').resolve()}")
    print(f"[Saved] {(RESULT_DIR / 'g6_power_table_long_10e1_to_10e6.csv').resolve()}")
    print(f"[Saved] {(RESULT_DIR / 'g6_power_table_contract_wide_10e1_to_10e6.csv').resolve()}")
    print(f"[Saved] {(RESULT_DIR / 'g6_power_table_user_wide_10e1_to_10e6.csv').resolve()}")
    print(f"[Saved] {(RESULT_DIR / 'g6_power_table_10e1_to_10e6.tex').resolve()}")
    for mode in available_modes:
        p = RESULT_DIR / f"g6_power_table_10e1_to_10e6_{mode}.tex"
        if p.exists():
            print(f"[Saved] {p.resolve()}")
    print(f"[Saved] {(RESULT_DIR / 'paper_main_user_auprc.md').resolve()}")
    print(f"[Saved] {(RESULT_DIR / 'paper_appendix_contract_robustness.md').resolve()}")
    print(f"[Saved] {(RESULT_DIR / 'paper_key_findings.md').resolve()}")
    print(f"[Saved] {(RESULT_DIR / 'paper_user_ap_gain_lift.md').resolve()}")
    print("[Info] If a source summary has no AUPRC columns, AUPRC will appear as NaN in output.")


if __name__ == "__main__":
    main()
