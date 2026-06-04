from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = ROOT / "E_RQ2" / "9_G6compare3.py"
SUMMARY_SCRIPT = ROOT / "E_RQ2" / "9_G6compare_data.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_one(
    compare_mod: Any,
    mode: str,
    outdir_name: str,
    n_repeats: int,
    force_regen_dataset: bool,
    extra_conf: dict[str, Any] | None = None,
) -> None:
    conf = {
        "ATTACK_MODE": mode,
        "RESULT_ROOT": compare_mod.ROOT / "output" / "E" / outdir_name,
        "N_REPEATS": int(n_repeats),
        "REUSE_EXISTING_RESULTS": False,
        # 为了确保真的有 Sybil contract，建议首次全量跑时关闭复用数据集
        "REUSE_EXISTING_DATASETS": not bool(force_regen_dataset),
        "INCLUDE_SYBIL_CONTRACTS": True,
        "BASELINE_TAG": "_base_clean_sybil_contracts_v11",
    }
    if extra_conf:
        conf.update(dict(extra_conf))
    print(f"[Run] mode={mode} | outdir={conf['RESULT_ROOT']} | repeats={n_repeats}")
    compare_mod.run_compare(conf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G6 compare (attack1/2/3) and refresh summary tables.")
    parser.add_argument("--repeats", type=int, default=10, help="N_REPEATS for each attack mode (default: 10)")
    parser.add_argument(
        "--attack-modes",
        type=str,
        default="attack1,attack2,attack3",
        help="Comma-separated attack modes to run (e.g. attack1 or attack1,attack3).",
    )
    parser.add_argument(
        "--regen-dataset",
        action="store_true",
        help="Force regenerate baseline dataset (recommended at least once after code/data-rule updates).",
    )
    parser.add_argument(
        "--custom-multipliers",
        type=str,
        default="",
        help="Comma-separated multipliers for attack1/attack3 (e.g. 1e6,1e7,1e8).",
    )
    parser.add_argument(
        "--lightweight-max-only",
        action="store_true",
        help="Run only the maximum multiplier per repeat and only four methods (Our/BiRank/PageRank/EigenTrust).",
    )
    parser.add_argument(
        "--lightweight-index-cols",
        type=str,
        default="contract_spearman_rho_mean,user_spearman_rho_mean",
        help="Two (or more) summary columns exported to g6_lightweight_max_metrics.csv.",
    )
    parser.add_argument(
        "--lightweight-methods",
        type=str,
        default="",
        help="Comma-separated method list for lightweight mode (e.g. our,our_no_u0,our_sensitive,our_robust,birank,pagerank,eigentrust).",
    )
    parser.add_argument(
        "--our-ablation-suite",
        action="store_true",
        help="Enable Our ablation suite (Our/NoU0/Sensitive/Robust).",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip running E_RQ2/9_G6compare_data.py after all attack modes finish.",
    )
    parser.add_argument(
        "--sybil-user-count",
        type=int,
        default=None,
        help="Override SYBIL_USER_COUNT in compare config.",
    )
    args = parser.parse_args()

    compare_mod = _load_module(COMPARE_SCRIPT, "g6_compare3_runner")

    modes = [x.strip() for x in str(args.attack_modes).split(",") if x.strip()]
    if not modes:
        raise ValueError("No attack mode selected.")
    custom_multipliers = [float(x) for x in str(args.custom_multipliers).split(",") if str(x).strip()]
    idx_cols = [str(x).strip() for x in str(args.lightweight_index_cols).split(",") if str(x).strip()]
    lw_methods = [str(x).strip() for x in str(args.lightweight_methods).split(",") if str(x).strip()]
    extra_conf: dict[str, Any] = {
        "LIGHTWEIGHT_MAX_ONLY": bool(args.lightweight_max_only),
        "LIGHTWEIGHT_INDEX_COLS": idx_cols,
    }
    if lw_methods:
        extra_conf["LIGHTWEIGHT_METHODS"] = lw_methods
    if args.sybil_user_count is not None:
        extra_conf["SYBIL_USER_COUNT"] = int(args.sybil_user_count)
    if bool(args.our_ablation_suite):
        extra_conf.update(
            {
                "RUN_OUR": True,
                "RUN_OUR_NO_U0": True,
                "RUN_OUR_SENSITIVE": True,
                "RUN_OUR_ROBUST": True,
            }
        )
    if custom_multipliers:
        extra_conf["ATTACK_CUSTOM_MULTIPLIERS"] = custom_multipliers
    for mode in modes:
        mode_tag = str(mode).lower()
        outdir_name = f"9_G6compare_v11_{mode_tag}_auprc"
        _run_one(compare_mod, mode_tag, outdir_name, args.repeats, bool(args.regen_dataset and mode_tag == "attack1"), extra_conf=extra_conf)

    if not args.skip_summary:
        summary_mod = _load_module(SUMMARY_SCRIPT, "g6_summary_runner")
        print("[Run] summary export")
        summary_mod.main()

    print("[Done] all tasks finished.")


if __name__ == "__main__":
    main()
