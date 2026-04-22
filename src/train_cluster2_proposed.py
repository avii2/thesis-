from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.fl.aggregators import WeightedState, aggregate_leaf_updates_to_subcluster, aggregate_subcluster_updates_to_maincluster
from src.fl.fedprox import predict_split_fedprox, train_fedprox_client
from src.fl.maincluster import (
    _load_yaml,
    _round_row,
    _split_metrics,
    _write_round_metrics,
    _write_summary_metrics,
    build_flat_federated_clients,
)
from src.fl.subcluster import group_clients_by_subcluster, load_frozen_membership
from src.models.mlp import CompactMLPClassifier, MLPConfig


@dataclass(frozen=True)
class Cluster2ProposedRunResult:
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


def _load_cluster2_proposed_entry(config_path: str | Path) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    clusters = config.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError(f"{config_path}: proposed config must contain clusters.")

    for cluster_entry in clusters:
        if not isinstance(cluster_entry, Mapping):
            continue
        if str(cluster_entry.get("experiment_id")) != "P_C2":
            continue
        if str(cluster_entry.get("model_family")) != "compact_mlp":
            raise ValueError("P_C2 must use model_family=compact_mlp.")
        if str(cluster_entry.get("fl_method")) != "FedProx":
            raise ValueError("P_C2 must use fl_method=FedProx.")
        if str(cluster_entry.get("aggregation")) != "weighted_arithmetic_mean":
            raise ValueError("P_C2 must use aggregation=weighted_arithmetic_mean.")
        return config_path, config, cluster_entry

    raise ValueError(f"{config_path}: could not find P_C2 entry in proposed config.")


def _evaluate_cluster2_split(
    clients: Sequence[Any],
    *,
    state: Mapping[str, np.ndarray],
    model_config: MLPConfig,
    split_name: str,
) -> dict[str, Any]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for client in clients:
        split = getattr(client, split_name)
        split_probabilities, _ = predict_split_fedprox(state, model_config, split)
        if split_probabilities.size == 0:
            continue
        probabilities.append(split_probabilities)
        labels.append(split.labels.astype(np.int8, copy=False))

    if not probabilities:
        return _split_metrics(np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32))
    return _split_metrics(np.concatenate(labels), np.concatenate(probabilities))


def run_cluster2_proposed(
    proposed_config_path: str | Path = "configs/proposed.yaml",
    *,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
    mu: float = 0.01,
    seed: int | None = None,
    smoke_test: bool = False,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
    output_root: str | Path = "outputs",
) -> Cluster2ProposedRunResult:
    resolved_proposed_config_path, proposed_config, cluster_entry = _load_cluster2_proposed_entry(proposed_config_path)

    defaults_key = "smoke_test_defaults" if smoke_test else "training_defaults"
    defaults = proposed_config.get(defaults_key)
    if not isinstance(defaults, Mapping):
        raise ValueError(f"{resolved_proposed_config_path}: missing {defaults_key}.")

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

    cluster_config_path = _resolve_path(resolved_proposed_config_path, str(cluster_entry["cluster_config"]))
    membership_path = _resolve_path(resolved_proposed_config_path, str(cluster_entry["membership_file"]))
    cluster_yaml = _load_yaml(cluster_config_path)
    cluster_section = cluster_yaml.get("cluster")
    if not isinstance(cluster_section, Mapping):
        raise ValueError(f"{cluster_config_path}: cluster metadata is missing.")
    cluster_id = int(cluster_section["id"])
    if cluster_id != 2:
        raise ValueError("run_cluster2_proposed only supports Cluster 2.")

    clients, _, data_summary = build_flat_federated_clients(
        cluster_config_path,
        max_train_examples_per_client=max_train_examples_per_client,
        max_eval_examples_per_client=max_eval_examples_per_client,
    )
    if data_summary["input_adapter"] != "feature_vector_as_sequence":
        raise ValueError("Cluster 2 proposed path requires feature-vector sequence inputs before MLP flattening.")
    if int(data_summary["input_channels"]) != 1:
        raise ValueError(
            f"Cluster 2 proposed path requires input_channels=1 before MLP flattening. "
            f"Observed {data_summary['input_channels']}."
        )

    membership = load_frozen_membership(
        membership_path,
        expected_cluster_id=2,
        expected_n_subclusters=int(cluster_entry["n_subclusters"]),
        expected_client_ids=[client.client_id for client in clients],
    )
    clients_by_subcluster = group_clients_by_subcluster(clients, membership)

    model_config = MLPConfig(input_dim=int(data_summary["input_length"]))
    global_model = CompactMLPClassifier(model_config, seed=configured_seed)
    global_state = global_model.state_dict()
    parameter_bytes = global_model.parameter_bytes()

    output_root = Path(output_root)
    run_dir = output_root / "runs" / "P_C2"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = output_root / "metrics" / "P_C2_metrics.csv"
    round_metrics_path = run_dir / "round_metrics.csv"
    summary_path = run_dir / "run_summary.json"
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

    for round_index in range(1, configured_rounds + 1):
        subcluster_updates: list[WeightedState] = []
        subcluster_losses: list[float] = []
        subcluster_sample_counts: dict[str, int] = {}
        subcluster_client_counts: dict[str, int] = {}

        for subcluster_index, subcluster in enumerate(membership.subclusters):
            local_updates: list[WeightedState] = []
            local_losses: list[float] = []
            subcluster_clients = clients_by_subcluster[subcluster.subcluster_id]

            for client_index, client in enumerate(subcluster_clients):
                result = train_fedprox_client(
                    client,
                    global_state,
                    model_config,
                    local_epochs=configured_local_epochs,
                    batch_size=configured_batch_size,
                    learning_rate=learning_rate,
                    mu=mu,
                    seed=configured_seed + round_index * 1000 + subcluster_index * 100 + client_index,
                )
                local_updates.append(result.to_weighted_state(cluster_id=cluster_id))
                local_losses.append(result.train_loss)

            aggregated_subcluster_state = aggregate_leaf_updates_to_subcluster(
                local_updates,
                expected_cluster_id=cluster_id,
            )
            subcluster_updates.append(
                WeightedState(
                    cluster_id=cluster_id,
                    contributor_id=subcluster.subcluster_id,
                    num_samples=sum(client.train.num_samples for client in subcluster_clients),
                    state=aggregated_subcluster_state,
                )
            )
            subcluster_losses.append(float(np.mean(local_losses)))
            subcluster_sample_counts[subcluster.subcluster_id] = sum(
                client.train.num_samples for client in subcluster_clients
            )
            subcluster_client_counts[subcluster.subcluster_id] = len(subcluster_clients)

        global_state = aggregate_subcluster_updates_to_maincluster(
            subcluster_updates,
            expected_cluster_id=cluster_id,
        )

        train_metrics = _evaluate_cluster2_split(clients, state=global_state, model_config=model_config, split_name="train")
        validation_metrics = _evaluate_cluster2_split(
            clients,
            state=global_state,
            model_config=model_config,
            split_name="validation",
        )
        test_metrics = _evaluate_cluster2_split(clients, state=global_state, model_config=model_config, split_name="test")
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
        raise ValueError("P_C2: unable to determine best validation round.")
    if best_subcluster_sample_counts is None or best_subcluster_client_counts is None:
        raise ValueError("P_C2: missing best-round subcluster statistics.")

    membership_contents_after = membership.membership_file.read_text(encoding="utf-8")
    if membership_contents_before != membership_contents_after:
        raise ValueError("P_C2: frozen membership file changed during proposed Cluster 2 execution.")

    total_communication_cost_bytes = int(sum(row["communication_cost_bytes"] for row in round_rows))
    summary = {
        "experiment_id": "P_C2",
        "cluster_id": cluster_id,
        "dataset": str(cluster_section["dataset_name"]),
        "cluster_config_path": str(cluster_config_path),
        "hierarchy": "hierarchical_fixed",
        "subcluster_layer_used": True,
        "membership_file_used": str(membership.membership_file),
        "membership_hash": membership.membership_hash,
        "membership_file_changed": False,
        "model_family": "compact_mlp",
        "fl_method": "FedProx",
        "aggregation": "weighted_arithmetic_mean",
        "clustering_method": "agglomerative",
        "input_adapter": data_summary["input_adapter"],
        "model_input_layout": "batch_x_features",
        "input_channels": data_summary["input_channels"],
        "input_length": data_summary["input_length"],
        "model_input_dim": model_config.input_dim,
        "num_leaf_clients": len(clients),
        "n_subclusters": membership.n_subclusters,
        "subcluster_client_counts": {
            subcluster.subcluster_id: len(subcluster.client_ids)
            for subcluster in membership.subclusters
        },
        "rounds": configured_rounds,
        "local_epochs": configured_local_epochs,
        "batch_size": configured_batch_size,
        "learning_rate": learning_rate,
        "mu": mu,
        "optimizer": "Adam",
        "seed": configured_seed,
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
        "experiment_id": "P_C2",
        "cluster_id": cluster_id,
        "dataset": str(cluster_section["dataset_name"]),
        "hierarchy": "hierarchical_fixed",
        "model_family": "compact_mlp",
        "fl_method": "FedProx",
        "aggregation": "weighted_arithmetic_mean",
        "clustering_method": "agglomerative",
        "membership_hash": membership.membership_hash,
        "input_adapter": data_summary["input_adapter"],
        "model_input_layout": "batch_x_features",
        "num_leaf_clients": len(clients),
        "n_subclusters": membership.n_subclusters,
        "rounds": configured_rounds,
        "mu": mu,
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

    return Cluster2ProposedRunResult(
        experiment_id="P_C2",
        cluster_id=cluster_id,
        dataset=str(cluster_section["dataset_name"]),
        output_dir=run_dir,
        summary_path=summary_path,
        round_metrics_path=round_metrics_path,
        metrics_csv_path=metrics_csv_path,
        summary=summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the proposed Cluster 2 TON IoT + compact MLP + FedProx experiment."
    )
    parser.add_argument(
        "--proposed-config",
        default="configs/proposed.yaml",
        help="Path to configs/proposed.yaml.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke_test_defaults from proposed config.")
    parser.add_argument("--rounds", type=int, help="Optional override for FL rounds.")
    parser.add_argument("--local-epochs", type=int, help="Optional override for local epochs.")
    parser.add_argument("--batch-size", type=int, help="Optional override for local batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Local Adam learning rate.")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient.")
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
    result = run_cluster2_proposed(
        proposed_config_path=args.proposed_config,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mu=args.mu,
        seed=args.seed,
        smoke_test=args.smoke_test,
        max_train_examples_per_client=args.max_train_examples_per_client,
        max_eval_examples_per_client=args.max_eval_examples_per_client,
        output_root=args.output_root,
    )
    print(f"{result.experiment_id}: wrote {result.metrics_csv_path}")


if __name__ == "__main__":
    main()
