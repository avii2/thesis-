#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cluster1_batadal_calibration import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROPOSED_CONFIG_PATH,
    POSITIVE_CLASS_WEIGHT_SCALES,
    THRESHOLD_SELECTION_MODES,
    run_calibration_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P_C1_BATADAL threshold calibration and positive-class-weight ablation."
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Cluster 1 BATADAL output root containing P_C1_BATADAL predictions.",
    )
    parser.add_argument(
        "--proposed-config",
        default=str(DEFAULT_PROPOSED_CONFIG_PATH),
        help="Cluster 1 BATADAL proposed config used for class-weight ablation runs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed suffix for saved prediction files.")
    parser.add_argument(
        "--positive-class-weight-scale",
        dest="positive_class_weight_scales",
        action="append",
        type=float,
        choices=POSITIVE_CLASS_WEIGHT_SCALES,
        help="Positive class weight scale to run. May be repeated. Defaults to 0.25, 0.50, and 1.00.",
    )
    parser.add_argument(
        "--threshold-mode",
        dest="threshold_modes",
        action="append",
        choices=THRESHOLD_SELECTION_MODES,
        help="Validation threshold mode to report. May be repeated. Defaults to all calibration modes.",
    )
    parser.add_argument(
        "--force-rerun-class-weight",
        action="store_true",
        help="Rerun class-weight variants even if prediction artifacts already exist.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke-test defaults for class-weight runs.")
    parser.add_argument("--rounds", type=int, help="Optional FL round override for class-weight runs.")
    parser.add_argument("--local-epochs", type=int, help="Optional local epoch override for class-weight runs.")
    parser.add_argument("--batch-size", type=int, help="Optional batch-size override for class-weight runs.")
    parser.add_argument(
        "--max-train-examples-per-client",
        type=int,
        help="Optional cap for class-weight ablation training splits.",
    )
    parser.add_argument(
        "--max-eval-examples-per-client",
        type=int,
        help="Optional cap for class-weight ablation validation/test splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_calibration_analysis(
        output_root=args.output_root,
        proposed_config_path=args.proposed_config,
        seed=args.seed,
        positive_class_weight_scales=tuple(args.positive_class_weight_scales or POSITIVE_CLASS_WEIGHT_SCALES),
        threshold_modes=tuple(args.threshold_modes or THRESHOLD_SELECTION_MODES),
        force_rerun_class_weight=args.force_rerun_class_weight,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        smoke_test=args.smoke_test,
        max_train_examples_per_client=args.max_train_examples_per_client,
        max_eval_examples_per_client=args.max_eval_examples_per_client,
    )
    print(f"Threshold sweep: {artifacts.threshold_sweep_csv_path}")
    print(f"Threshold calibration report: {artifacts.threshold_report_path}")
    print(f"Class-weight ablation: {artifacts.class_weight_ablation_csv_path}")
    print(f"Calibration report: {artifacts.report_path}")
    for run_root in artifacts.class_weight_run_roots:
        print(f"Class-weight run root: {run_root}")


if __name__ == "__main__":
    main()
