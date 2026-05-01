#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cluster1_batadal_tuning import (  # noqa: E402
    DEFAULT_PROPOSED_CONFIG_PATH,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    DEFAULT_TUNED_OUTPUT_ROOT,
    run_tuning_workflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cluster 1 P_C1_BATADAL threshold and hyperparameter tuning.")
    parser.add_argument(
        "--source-output-root",
        default=str(DEFAULT_SOURCE_OUTPUT_ROOT),
        help="Existing Cluster 1 BATADAL output root containing A/B/P predictions.",
    )
    parser.add_argument(
        "--tuned-output-root",
        default=str(DEFAULT_TUNED_OUTPUT_ROOT),
        help="Output root for tuned P_C1_BATADAL variants.",
    )
    parser.add_argument(
        "--proposed-config",
        default=str(DEFAULT_PROPOSED_CONFIG_PATH),
        help="Cluster 1 BATADAL proposed config.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for P_C1_BATADAL tuned runs.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Retrain tuned variants even when their prediction artifacts already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_tuning_workflow(
        source_output_root=args.source_output_root,
        tuned_output_root=args.tuned_output_root,
        proposed_config_path=args.proposed_config,
        seed=args.seed,
        force_rerun=args.force_rerun,
    )
    print(f"Threshold sweep: {artifacts.threshold_sweep_csv_path}")
    print(f"Threshold calibration report: {artifacts.threshold_report_path}")
    print(f"Tuning comparison CSV: {artifacts.comparison_csv_path}")
    print(f"Tuning comparison report: {artifacts.comparison_report_path}")
    print(f"Selected final P_C1_BATADAL: {artifacts.final_row['label']}")
    for run_root in artifacts.executed_run_roots:
        print(f"Tuned run root: {run_root}")


if __name__ == "__main__":
    main()

