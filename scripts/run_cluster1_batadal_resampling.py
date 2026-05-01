#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cluster1_batadal_resampling import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROPOSED_CONFIG_PATH,
    POSITIVE_CLASS_WEIGHT_SCALES,
    run_resampling_workflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P_C1_BATADAL with training-only positive-window oversampling."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for resampling runs.")
    parser.add_argument("--proposed-config", default=str(DEFAULT_PROPOSED_CONFIG_PATH), help="P_C1_BATADAL config.")
    parser.add_argument("--seed", type=int, default=42, help="Run seed.")
    parser.add_argument(
        "--positive-class-weight-scale",
        action="append",
        dest="scales",
        type=float,
        choices=POSITIVE_CLASS_WEIGHT_SCALES,
        help="Scale to run. May be repeated. Defaults to 0.10, 0.25, 0.50.",
    )
    parser.add_argument("--rounds", type=int, help="Optional FL round override.")
    parser.add_argument("--local-epochs", type=int, help="Optional local epoch override.")
    parser.add_argument("--batch-size", type=int, help="Optional batch size override.")
    parser.add_argument("--learning-rate", type=float, help="Optional learning-rate override.")
    parser.add_argument("--dropout", type=float, help="Optional CNN1D-BN dropout override.")
    parser.add_argument("--force-rerun", action="store_true", help="Retrain even when artifacts already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_resampling_workflow(
        output_root=args.output_root,
        proposed_config_path=args.proposed_config,
        scales=tuple(args.scales or POSITIVE_CLASS_WEIGHT_SCALES),
        seed=args.seed,
        force_rerun=args.force_rerun,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
    )
    print(f"Selection CSV: {artifacts.comparison_csv_path}")
    print(f"Selection report: {artifacts.comparison_report_path}")
    print(f"Selected run: {artifacts.selected_row['run_id']}")
    for run_root in artifacts.run_roots:
        print(f"Run root: {run_root}")


if __name__ == "__main__":
    main()

