#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.cluster1_batadal import DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_ROOT, prepare_cluster1_batadal  # noqa: E402
from src.train import run_experiments  # noqa: E402


BATADAL_EXPERIMENT_IDS = ("A_C1_BATADAL", "B_C1_BATADAL", "P_C1_BATADAL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BATADAL and run active Cluster 1 BATADAL experiments.")
    parser.add_argument(
        "--experiment-id",
        action="append",
        dest="experiment_ids",
        choices=BATADAL_EXPERIMENT_IDS,
        help="Experiment to run. May be repeated. Defaults to all Cluster 1 BATADAL experiments.",
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse existing BATADAL prepared metadata.")
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke-test training defaults.")
    parser.add_argument("--rounds", type=int, help="Optional FL round override.")
    parser.add_argument("--local-epochs", type=int, help="Optional local epoch override.")
    parser.add_argument("--batch-size", type=int, help="Optional batch-size override.")
    parser.add_argument("--seed", type=int, help="Optional seed override.")
    parser.add_argument(
        "--max-train-examples-per-client",
        type=int,
        help="Optional cap for smoke-sized train splits.",
    )
    parser.add_argument(
        "--max-eval-examples-per-client",
        type=int,
        help="Optional cap for smoke-sized validation/test splits.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for Cluster 1 BATADAL experiment artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_prepare:
        prepare_cluster1_batadal(DEFAULT_CONFIG_PATH, output_root=args.output_root)

    selected = tuple(args.experiment_ids or BATADAL_EXPERIMENT_IDS)
    result = run_experiments(
        experiment_ids=selected,
        smoke_test=args.smoke_test,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        max_train_examples_per_client=args.max_train_examples_per_client,
        max_eval_examples_per_client=args.max_eval_examples_per_client,
        output_root=Path(args.output_root),
    )
    for artifact in result.experiments:
        print(f"{artifact.experiment_id}: wrote {artifact.metrics_csv_path}")
    if result.summary_csv_path is not None:
        print(f"Cluster 1 BATADAL summary: {result.summary_csv_path}")


if __name__ == "__main__":
    main()
