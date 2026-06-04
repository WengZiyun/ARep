#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import math

import pandas as pd


# ============================================================
# Config
# ============================================================
RUG_CSV = Path("output/D/1_eval_rugpull_plt2/auprc_mean_by_delta.csv")
COUNTERFEIT_CSV = Path("output/D/2_eval_counterfeit_plt/auprc_mean_by_delta.csv")
WASH_CSV = Path("output/D/3_eval_wash_plt2/auprc_mean_by_delta.csv")

OUTPUT_DIR = Path("output/D/6_auprc_table")

# Follow your example checkpoints
DELTA_POINTS = [0, 4, 10, 15, 20, 25, 40]
DAYS_PER_DELTA = 7


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")


def _days_label(days: int) -> str:
    if days == 0:
        return "0 (爆发期)"
    if days == 28:
        return "28 (1个月前)"
    if days == 70:
        return "70 (约2个月前)"
    if days == 105:
        return "105 (约3.5个月前)"
    if days == 140:
        return "140 (约4.5个月前)"
    if days == 175:
        return "175 (约6个月前)"
    if days == 280:
        return "280 (约9个月前)"
    return str(days)


def _format_mean_sem(mean_val: float, sem_val: float) -> str:
    if pd.isna(mean_val) or pd.isna(sem_val):
        return "NA"
    if abs(mean_val) < 0.1 or abs(sem_val) < 0.1:
        return f"{mean_val:.3f}$\\pm${sem_val:.3f}"
    return f"{mean_val:.2f}$\\pm${sem_val:.2f}"


def _build_one_behavior_table(behavior: str, csv_path: Path) -> pd.DataFrame:
    _ensure_exists(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    need = ["delta_windows", "auprc_mean", "auprc_std", "auprc_count_non_nan"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {csv_path}: {miss}")

    df["delta_windows"] = pd.to_numeric(df["delta_windows"], errors="coerce")
    df["auprc_mean"] = pd.to_numeric(df["auprc_mean"], errors="coerce")
    df["auprc_std"] = pd.to_numeric(df["auprc_std"], errors="coerce")
    df["auprc_count_non_nan"] = pd.to_numeric(df["auprc_count_non_nan"], errors="coerce")
    df = df.dropna(subset=["delta_windows"]).copy()
    df["delta_windows"] = df["delta_windows"].astype(int)

    rows = []
    for dlt in DELTA_POINTS:
        days = int(dlt * DAYS_PER_DELTA)
        hit = df[df["delta_windows"] == int(dlt)]
        if hit.empty:
            rows.append(
                {
                    "behavior": behavior,
                    "days_label": _days_label(days),
                    "days": days,
                    "delta": int(dlt),
                    "mean": math.nan,
                    "sem": math.nan,
                    "mean_pm_sem": "NA",
                    "n": 0,
                }
            )
            continue

        r = hit.iloc[0]
        mean_val = float(r["auprc_mean"])
        std_val = float(r["auprc_std"])
        n_val = int(r["auprc_count_non_nan"]) if pd.notna(r["auprc_count_non_nan"]) else 0
        sem_val = float(std_val / math.sqrt(n_val)) if n_val > 0 else math.nan
        rows.append(
            {
                "behavior": behavior,
                "days_label": _days_label(days),
                "days": days,
                "delta": int(dlt),
                "mean": mean_val,
                "sem": sem_val,
                "mean_pm_sem": _format_mean_sem(mean_val, sem_val),
                "n": n_val,
            }
        )
    return pd.DataFrame(rows)


def _to_markdown_table(df: pd.DataFrame, title: str) -> str:
    show = df[["days_label", "delta", "mean_pm_sem"]].copy()
    show.columns = ["回溯天数 (Days)", "回溯窗口 (Δ)", "平均 AUPRC (Mean ± SEM)"]
    md = [f"### {title}", show.to_markdown(index=False), ""]
    return "\n".join(md)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rug = _build_one_behavior_table("rugpull", RUG_CSV)
    cnt = _build_one_behavior_table("counterfeit", COUNTERFEIT_CSV)
    wash = _build_one_behavior_table("wash", WASH_CSV)

    combined = pd.concat([rug, cnt, wash], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "auprc_delta_table_three_behaviors.csv", index=False, encoding="utf-8-sig")

    rug.to_csv(OUTPUT_DIR / "auprc_delta_table_rugpull.csv", index=False, encoding="utf-8-sig")
    cnt.to_csv(OUTPUT_DIR / "auprc_delta_table_counterfeit.csv", index=False, encoding="utf-8-sig")
    wash.to_csv(OUTPUT_DIR / "auprc_delta_table_wash.csv", index=False, encoding="utf-8-sig")

    md_text = "\n".join(
        [
            "## Table X: TRep 识别性能随回溯窗口（Delta）的变化情况",
            _to_markdown_table(rug, "Rugpull"),
            _to_markdown_table(cnt, "Counterfeit"),
            _to_markdown_table(wash, "Wash"),
        ]
    )
    (OUTPUT_DIR / "auprc_delta_table_three_behaviors.md").write_text(md_text, encoding="utf-8")

    print(f"[Saved] {(OUTPUT_DIR / 'auprc_delta_table_three_behaviors.csv').resolve()}")
    print(f"[Saved] {(OUTPUT_DIR / 'auprc_delta_table_three_behaviors.md').resolve()}")


if __name__ == "__main__":
    main()
