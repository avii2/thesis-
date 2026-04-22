from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.fl.aggregators import aggregate_subcluster_updates_to_maincluster
from src.fl.maincluster import (
    _evaluate_cluster_split,
    _load_yaml,
    _round_row,
    _write_round_metrics,
    _write_summary_metrics,
    build_flat_federated_clients,
)
from src.fl.subcluster import group_clients_by_subcluster, load_frozen_membership, run_subcluster_round
from src.models.cnn1d import CNN1DClassifier


@dataclass(frozen=True)
class HierarchicalBaselineRunResult:
    experiment_id: str
    cluster_id: int
    dataset: str
    output_dir: Path
    summary_path: Path
    round_metrics_path: Path
    metrics_csv_path: Path
    summary: Mapping[str, Any]


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


def run_hierarchical_baseline_experiment(
    *,
    experiment_id: str,
    cluster_config_path: str | Path,
    membership_file: str | Path,
    expected_n_subclusters: int,
    output_root: str | Path = "outputs",
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
) -> HierarchicalBaselineRunResult:
    cluster_yaml = _load_yaml(cluster_config_path)
    cluster_section = cluster_yaml.get("cluster")
    if not isinstance(cluster_section, Mapping):
        raise ValueError(f"Cluster config {cluster_config_path} is missing cluster metadata.")

    cluster_id = int(cluster_section["id"])
    clients, model_config, data_summary = build_flat_federated_clients(
        cluster_config_path,
        max_train_examples_per_client=max_train_examples_per_client,
        max_eval_examples_per_client=max_eval_examples_per_client,
    )
    membership = load_frozen_membership(
        membership_file,
        expected_cluster_id=cluster_id,
        expected_n_subclusters=expected_n_subclusters,
        expected_client_ids=[client.client_id for client in clients],
    )
    clients_by_subcluster = group_clients_by_subcluster(clients, membership)

    output_root = Path(output_root)
    run_dir = output_root / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = output_root / "metrics" / f"{experiment_id}_metrics.csv"
    round_metrics_path = run_dir / "round_metrics.csv"
    summary_path = run_dir / "run_summary.json"

    global_model = CNN1DClassifier(model_config, seed=seed)
    global_state = global_model.state_dict()
    parameter_bytes = global_model.parameter_bytes()
    membership_contents_before = membership.membership_file.read_text(encoding="utf-8")

    started_at = time.perf_counter()
    round_rows: list[dict[str, Any]] = []
    best_round_index = 1
    best_validation_f1 = float("-inf")
    best_test_metrics: Mapping[str, Any] | None = None
    best_validation_metrics: Mapping[str, Any] | None = None
    best_train_metrics: Mapping[str, Any] | None = None
    best_subcluster_sample_counts: Mapping[str, int] | None = None
    best_subcluster_client_counts: Mapping[str, int] | None = None

    for round_index in range(1, rounds + 1):
        subcluster_updates = []
        subcluster_losses: list[float] = []
        subcluster_sample_counts: dict[str, int] = {}
        subcluster_client_counts: dict[str, int] = {}

        for subcluster_index, subcluster in enumerate(membership.subclusters):
            result = run_subcluster_round(
                cluster_id=cluster_id,
                subcluster_id=subcluster.subcluster_id,
                clients=clients_by_subcluster[subcluster.subcluster_id],
                parent_state=global_state,
                model_config=model_config,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed + round_index * 1000 + subcluster_index * 100,
            )
            subcluster_updates.append(result.to_weighted_state())
            subcluster_losses.append(result.mean_local_loss)
            subcluster_sample_counts[result.subcluster_id] = result.num_train_samples
            subcluster_client_counts[result.subcluster_id] = result.num_clients

        global_state = aggregate_subcluster_updates_to_maincluster(
            subcluster_updates,
            expected_cluster_id=cluster_id,
        )

        train_metrics = _evaluate_cluster_split(clients, global_state, model_config, split_name="train")
        validation_metrics = _evaluate_cluster_split(clients, global_state, model_config, split_name="validation")
        test_metrics = _evaluate_cluster_split(clients, global_state, model_config, split_name="test")
        communication_cost_bytes = int(parameter_bytes * 2 * (len(clients) + membership.n_subclusters))

        round_rows.append(
            _round_row(
                round_index,
                float(np.mean(subcluster_losses)),
                train_metrics,
                validation_metrics,
                test_metrics,
                communication_cost_bytes,
            )
        )

        validation_f1 = validation_metrics["f1"]
        if validation_f1 is not None and float(validation_f1) >= best_validation_f1:
            best_validation_f1 = float(validation_f1)
            best_round_index = round_index
            best_train_metrics = dict(train_metrics)
            best_validation_metrics = dict(validation_metrics)
            best_test_metrics = dict(test_metrics)
            best_subcluster_sample_counts = dict(subcluster_sample_counts)
            best_subcluster_client_counts = dict(subcluster_client_counts)

    elapsed_seconds = float(time.perf_counter() - started_at)
    if best_test_metrics is None or best_validation_metrics is None or best_train_metrics is None:
        raise ValueError(f"{experiment_id}: unable to determine best validation round.")
    if best_subcluster_sample_counts is None or best_subcluster_client_counts is None:
        raise ValueError(f"{experiment_id}: missing best-round subcluster statistics.")

    membership_contents_after = membership.membership_file.read_text(encoding="utf-8")
    if membership_contents_before != membership_contents_after:
        raise ValueError(f"{experiment_id}: frozen membership file changed during hierarchical baseline execution.")

    total_communication_cost_bytes = int(sum(row["communication_cost_bytes"] for row in round_rows))
    summary = {
        "experiment_id": experiment_id,
        "cluster_id": cluster_id,
        "dataset": str(cluster_section["dataset_name"]),
        "cluster_config_path": str(cluster_config_path),
        "hierarchy": "hierarchical_fixed",
        "subcluster_layer_used": True,
        "membership_file_used": str(membership.membership_file),
        "membership_hash": membership.membership_hash,
        "membership_file_changed": False,
        "model_family": "cnn1d",
        "fl_method": "FedAvg",
        "aggregation": "weighted_arithmetic_mean",
        "clustering_method": "agglomerative",
        "input_adapter": data_summary["input_adapter"],
        "input_channels": data_summary["input_channels"],
        "input_length": data_summary["input_length"],
        "num_leaf_clients": len(clients),
        "n_subclusters": membership.n_subclusters,
        "subcluster_client_counts": {
            subcluster.subcluster_id: len(subcluster.client_ids)
            for subcluster in membership.subclusters
        },
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "model_parameter_bytes": parameter_bytes,
        "communication_cost_per_round_bytes": round_rows[-1]["communication_cost_bytes"],
        "total_communication_cost_bytes": total_communication_cost_bytes,
        "best_validation_round": best_round_index,
        "best_validation_f1": best_validation_metrics["f1"],
        "test_f1_at_best_validation_round": best_test_metrics["f1"],
        "wall_clock_training_seconds": elapsed_seconds,
        "data_summary": data_summary,
        "best_round_train_metrics": best_train_metrics,
        "best_round_validation_metrics": best_validation_metrics,
        "best_round_test_metrics": best_test_metrics,
        "best_round_subcluster_train_sample_counts": best_subcluster_sample_counts,
        "best_round_subcluster_client_counts": best_subcluster_client_counts,
        "round_metrics_path": str(round_metrics_path),
        "metrics_csv_path": str(metrics_csv_path),
    }

    summary_row = {
        "experiment_id": experiment_id,
        "cluster_id": cluster_id,
        "dataset": str(cluster_section["dataset_name"]),
        "hierarchy": "hierarchical_fixed",
        "model_family": "cnn1d",
        "fl_method": "FedAvg",
        "aggregation": "weighted_arithmetic_mean",
        "clustering_method": "agglomerative",
        "membership_hash": membership.membership_hash,
        "input_adapter": data_summary["input_adapter"],
        "num_leaf_clients": len(clients),
        "n_subclusters": membership.n_subclusters,
        "rounds": rounds,
        "best_validation_round": best_round_index,
        "best_validation_f1": best_validation_metrics["f1"],
        "test_accuracy": best_test_metrics["accuracy"],
        "test_precision": best_test_metrics["precision"],
        "test_recall": best_test_metrics["recall"],
        "test_f1": best_test_metrics["f1"],
        "test_auroc": best_test_metrics["auroc"],
        "test_pr_auc": best_test_metrics["pr_auc"],
        "test_fpr": best_test_metrics["fpr"],
        "test_confusion_matrix": json.dumps(best_test_metrics["confusion_matrix"]),
        "communication_cost_per_round_bytes": round_rows[-1]["communication_cost_bytes"],
        "total_communication_cost_bytes": total_communication_cost_bytes,
        "wall_clock_training_seconds": elapsed_seconds,
    }

    _write_round_metrics(round_metrics_path, round_rows)
    _write_summary_metrics(metrics_csv_path, summary_row)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return HierarchicalBaselineRunResult(
        experiment_id=experiment_id,
        cluster_id=cluster_id,
        dataset=str(cluster_section["dataset_name"]),
        output_dir=run_dir,
        summary_path=summary_path,
        round_metrics_path=round_metrics_path,
        metrics_csv_path=metrics_csv_path,
        summary=summary,
    )


def run_hierarchical_baseline_experiments(
    baseline_config_path: str | Path = "configs/baseline_hierarchical.yaml",
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
) -> list[HierarchicalBaselineRunResult]:
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
        raise ValueError("baseline_hierarchical config must contain at least one cluster entry.")

    results: list[HierarchicalBaselineRunResult] = []
    for cluster_entry in cluster_entries:
        if not isinstance(cluster_entry, Mapping):
            raise ValueError("Each baseline_hierarchical cluster entry must be a mapping.")
        experiment_id = str(cluster_entry["experiment_id"])
        if selected_experiment_ids and experiment_id not in selected_experiment_ids:
            continue

        if str(cluster_entry.get("hierarchy")) != "hierarchical_fixed":
            raise ValueError(f"{experiment_id}: Baseline B must remain hierarchical_fixed.")
        if str(cluster_entry.get("clustering_method")) != "agglomerative":
            raise ValueError(f"{experiment_id}: Baseline B must use frozen agglomerative memberships.")
        if str(cluster_entry.get("model_family")) != "cnn1d":
            raise ValueError(f"{experiment_id}: Baseline B must use cnn1d.")
        if str(cluster_entry.get("fl_method")) != "FedAvg":
            raise ValueError(f"{experiment_id}: Baseline B must use FedAvg.")

        cluster_config_path = _resolve_path(
            baseline_config_path,
            str(cluster_entry["cluster_config"]),
        )
        membership_path = _resolve_path(
            baseline_config_path,
            str(cluster_entry["membership_file"]),
        )
        results.append(
            run_hierarchical_baseline_experiment(
                experiment_id=experiment_id,
                cluster_config_path=cluster_config_path,
                membership_file=membership_path,
                expected_n_subclusters=int(cluster_entry["n_subclusters"]),
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
    parser = argparse.ArgumentParser(description="Run Baseline B uniform hierarchical FCFL experiments.")
    parser.add_argument(
        "--baseline-config",
        default="configs/baseline_hierarchical.yaml",
        help="Path to configs/baseline_hierarchical.yaml.",
    )
    parser.add_argument(
        "--experiment-id",
        action="append",
        dest="experiment_ids",
        help="Optional experiment id filter such as B_C1.",
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
    results = run_hierarchical_baseline_experiments(
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
