from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.fl.maincluster import FlatMainClusterRunResult, run_flat_maincluster_experiment


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def _resolve_path(base_config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()

    repo_root_candidate = base_config_path.parents[1] / configured_path
    if repo_root_candidate.exists():
        return repo_root_candidate.resolve()

    sibling_candidate = base_config_path.parent / configured_path
    if sibling_candidate.exists():
        return sibling_candidate.resolve()

    return repo_root_candidate.resolve()


def run_flat_baseline_experiments(
    baseline_config_path: str | Path = "configs/baseline_flat.yaml",
    *,
    experiment_ids: Sequence[str] | None = None,
    smoke_test: bool = False,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float = 0.05,
    seed: int | None = None,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
    output_root: str | Path = "outputs",
) -> list[FlatMainClusterRunResult]:
    baseline_config_path = Path(baseline_config_path).resolve()
    baseline = _load_yaml(baseline_config_path)

    defaults_key = "smoke_test_defaults" if smoke_test else "training_defaults"
    defaults = baseline.get(defaults_key)
    if not isinstance(defaults, Mapping):
        raise ValueError(f"Baseline config is missing {defaults_key}.")

    configured_rounds = int(rounds if rounds is not None else defaults["rounds"])
    configured_local_epochs = int(local_epochs if local_epochs is not None else defaults["local_epochs"])
    configured_batch_size = int(batch_size if batch_size is not None else defaults["batch_size"])
    if seed is not None:
        configured_seed = int(seed)
    elif smoke_test:
        configured_seed = int(defaults["seed"])
    else:
        seeds = defaults.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("training_defaults.seeds must contain at least one seed.")
        configured_seed = int(seeds[0])

    selected_experiment_ids = set(experiment_ids or [])
    cluster_entries = baseline.get("clusters")
    if not isinstance(cluster_entries, list) or not cluster_entries:
        raise ValueError("baseline_flat config must contain at least one cluster entry.")

    results: list[FlatMainClusterRunResult] = []
    for cluster_entry in cluster_entries:
        if not isinstance(cluster_entry, Mapping):
            raise ValueError("Each baseline_flat cluster entry must be a mapping.")
        experiment_id = str(cluster_entry["experiment_id"])
        if selected_experiment_ids and experiment_id not in selected_experiment_ids:
            continue

        if str(cluster_entry.get("hierarchy")) != "flat":
            raise ValueError(f"{experiment_id}: Baseline A must remain flat.")
        if str(cluster_entry.get("clustering_method")) != "none":
            raise ValueError(f"{experiment_id}: Baseline A must not use clustering.")
        if str(cluster_entry.get("model_family")) != "cnn1d":
            raise ValueError(f"{experiment_id}: Baseline A must use cnn1d.")
        if str(cluster_entry.get("fl_method")) != "FedAvg":
            raise ValueError(f"{experiment_id}: Baseline A must use FedAvg.")

        cluster_config_path = _resolve_path(
            baseline_config_path,
            str(cluster_entry["cluster_config"]),
        )
        results.append(
            run_flat_maincluster_experiment(
                experiment_id=experiment_id,
                cluster_config_path=cluster_config_path,
                output_root=output_root,
                rounds=configured_rounds,
                local_epochs=configured_local_epochs,
                batch_size=configured_batch_size,
                learning_rate=learning_rate,
                seed=configured_seed,
                max_train_examples_per_client=max_train_examples_per_client,
                max_eval_examples_per_client=max_eval_examples_per_client,
            )
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Baseline A flat per-cluster FL experiments.")
    parser.add_argument(
        "--baseline-config",
        default="configs/baseline_flat.yaml",
        help="Path to configs/baseline_flat.yaml.",
    )
    parser.add_argument(
        "--experiment-id",
        action="append",
        dest="experiment_ids",
        help="Optional experiment id filter such as A_C1.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke_test_defaults from baseline config.")
    parser.add_argument("--rounds", type=int, help="Optional override for FL rounds.")
    parser.add_argument("--local-epochs", type=int, help="Optional override for local epochs.")
    parser.add_argument("--batch-size", type=int, help="Optional override for local batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Local SGD learning rate.")
    parser.add_argument("--seed", type=int, help="Optional override for the run seed.")
    parser.add_argument(
        "--max-train-examples-per-client",
        type=int,
        help="Optional cap applied after split adaptation for smoke-sized training.",
    )
    parser.add_argument(
        "--max-eval-examples-per-client",
        type=int,
        help="Optional cap applied after split adaptation for validation/test evaluation.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output directory. Defaults to outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_flat_baseline_experiments(
        baseline_config_path=args.baseline_config,
        experiment_ids=args.experiment_ids,
        smoke_test=args.smoke_test,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_train_examples_per_client=args.max_train_examples_per_client,
        max_eval_examples_per_client=args.max_eval_examples_per_client,
        output_root=args.output_root,
    )
    for result in results:
        print(f"{result.experiment_id}: wrote {result.metrics_csv_path}")


if __name__ == "__main__":
    main()
