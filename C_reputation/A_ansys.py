#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("output/A/4_collection_time_span.csv")
OUTPUT_DIR = Path("output/C")
OUTPUT_TABLE = OUTPUT_DIR / "A_collection_block_span_summary.csv"
OUTPUT_FIG = OUTPUT_DIR / "A_collection_block_span_gantt.png"


def load_and_prepare(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv.resolve()}")

    df = pd.read_csv(input_csv)
    required = ["name", "min_block_number", "max_block_number", "block_span"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {input_csv.name}: {miss}")

    out = df[["name", "min_block_number", "max_block_number", "block_span"]].copy()
    out["min_block_number"] = pd.to_numeric(out["min_block_number"], errors="coerce")
    out["max_block_number"] = pd.to_numeric(out["max_block_number"], errors="coerce")
    out["block_span"] = pd.to_numeric(out["block_span"], errors="coerce")
    out = out.dropna(subset=["min_block_number", "max_block_number"]).copy()

    out["min_block_number"] = out["min_block_number"].astype(np.int64)
    out["max_block_number"] = out["max_block_number"].astype(np.int64)
    out["block_span"] = (out["max_block_number"] - out["min_block_number"]).astype(np.int64)

    out = out.sort_values(["min_block_number", "max_block_number", "name"], ascending=[True, True, True]).reset_index(drop=True)
    out["collection_index"] = np.arange(len(out), dtype=np.int64)
    return out


def save_table(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "collection_index",
        "name",
        "min_block_number",
        "max_block_number",
        "block_span",
    ]
    df[cols].to_csv(output_csv, index=False, encoding="utf-8-sig")


def plot_gantt_lines(df: pd.DataFrame, output_fig: Path) -> None:
    output_fig.parent.mkdir(parents=True, exist_ok=True)

    n = len(df)
    fig_h = max(5.0, min(18.0, n * 0.06))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    y = df["collection_index"].to_numpy(dtype=float)
    x0 = df["min_block_number"].to_numpy(dtype=float)
    x1 = df["max_block_number"].to_numpy(dtype=float)

    ax.hlines(y, x0, x1, colors="#1f77b4", linewidth=1.0, alpha=0.9)

    ax.set_xlabel("Block Number")
    ax.set_ylabel("Collections")
    ax.set_title("Collection Time Spans by Block Height")

    # Do not show per-collection names on y-axis.
    ax.set_yticks([])

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_ylim(-1, n)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(output_fig, dpi=220)
    plt.close(fig)


def main() -> None:
    df = load_and_prepare(INPUT_CSV)
    save_table(df, OUTPUT_TABLE)
    plot_gantt_lines(df, OUTPUT_FIG)

    print(f"[Saved] {OUTPUT_TABLE.resolve()}")
    print(f"[Saved] {OUTPUT_FIG.resolve()}")
    print(f"[Data] collections={len(df):,} min_block={df['min_block_number'].min():,} max_block={df['max_block_number'].max():,}")


if __name__ == "__main__":
    main()
