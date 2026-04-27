from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.fl.aggregators import WeightedState
from src.fl.maincluster import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    _load_yaml,
    _round_row,
    _split_metrics,
    _write_round_metrics,
    _write_summary_metrics,
    build_flat_federated_clients,
    compute_cluster_positive_class_weight,
    evaluate_round_with_validation_threshold,
    write_prediction_outputs,
)
from src.fl.subcluster import group_clients_by_subcluster, load_frozen_membership
from src.fl.fedbn import (
    aggregate_fedbn_leaf_updates,
    aggregate_fedbn_subcluster_updates,
    merge_global_non_bn_with_local_bn,
    non_bn_parameter_bytes,
    predict_split_fedbn,
    train_fedbn_client,
)
from src.models.tcn import TCNClassifier, TCNConfig


@dataclass(frozen=True)
class Cluster1ProposedRunResult:
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


def _evaluate_cluster_split_with_local_bn(
    clients: Sequence[Any],
    *,
    global_state: Mapping[str, np.ndarray],
    client_local_states: Mapping[str, Mapping[str, np.ndarray]],
    model_config: TCNConfig,
    split_name: str,
) -> dict[str, Any]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for client in clients:
        split = getattr(client, split_name)
        eval_state = merge_global_non_bn_with_local_bn(global_state, client_local_states[client.client_id])
        split_probabilities, _ = predict_split_fedbn(eval_state, model_config, split)
        if split_probabilities.size == 0:
            continue
        probabilities.append(split_probabilities)
        labels.append(split.labels.astype(np.int8, copy=False))

    if not probabilities:
        return _split_metrics(np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32))
    return _split_metrics(np.concatenate(labels), np.concatenate(probabilities))


def _load_cluster1_proposed_entry(config_path: str | Path) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    clusters = config.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError(f"{config_path}: proposed config must contain clusters.")

    for cluster_entry in clusters:
        if not isinstance(cluster_entry, Mapping):
            continue
        if str(cluster_entry.get("experiment_id")) != "P_C1":
            continue
        if str(cluster_entry.get("model_family")) != "tcn":
            raise ValueError("P_C1 must use model_family=tcn.")
        if str(cluster_entry.get("fl_method")) != "FedBN":
            raise ValueError("P_C1 must use fl_method=FedBN.")
        if str(cluster_entry.get("aggregation")) != "weighted_non_bn_mean":
            raise ValueError("P_C1 must use aggregation=weighted_non_bn_mean.")
        return config_path, config, cluster_entry

    raise ValueError(f"{config_path}: could not find P_C1 entry in proposed config.")


def _optional_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"P_C1 {key} must be a mapping when provided.")
    return value


def _resolve_block_channels(value: Sequence[int] | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    channels = tuple(int(channel) for channel in value)
    if len(channels) != 3:
        raise ValueError("P_C1 TCN block_channels must contain exactly three integers.")
    if any(channel <= 0 for channel in channels):
        raise ValueError("P_C1 TCN block_channels must be positive.")
    return channels


def _resolve_tcn_hyperparameters(
    cluster_entry: Mapping[str, Any],
    *,
    tcn_block_channels: Sequence[int] | None,
    tcn_hidden_dim: int | None,
    tcn_dropout: float | None,
) -> dict[str, Any]:
    model_hyperparameters = _optional_mapping(cluster_entry, "model_hyperparameters")
    block_channels = (
        _resolve_block_channels(tcn_block_channels)
        if tcn_block_channels is not None
        else _resolve_block_channels(model_hyperparameters.get("block_channels"))
    )
    hidden_dim_value = tcn_hidden_dim if tcn_hidden_dim is not None else model_hyperparameters.get("hidden_dim")
    dropout_value = tcn_dropout if tcn_dropout is not None else model_hyperparameters.get("dropout")

    return {
        "block_channels": block_channels if block_channels is not None else (32, 64, 64),
        "hidden_dim": int(hidden_dim_value) if hidden_dim_value is not None else 32,
        "dropout": float(dropout_value) if dropout_value is not None else 0.1,
    }


def _resolve_positive_class_weight_scale(
    cluster_entry: Mapping[str, Any],
    positive_class_weight_scale: float | None,
) -> float:
    training_hyperparameters = _optional_mapping(cluster_entry, "training_hyperparameters")
    configured_scale = (
        positive_class_weight_scale
        if positive_class_weight_scale is not None
        else training_hyperparameters.get("positive_class_weight_scale", 1.0)
    )
    scale = float(configured_scale)
    if scale <= 0.0:
        raise ValueError("P_C1 positive_class_weight_scale must be positive.")
    return scale


def run_cluster1_proposed(
    proposed_config_path: str | Path = "configs/proposed.yaml",
    *,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float = 0.05,
    seed: int | None = None,
    smoke_test: bool = False,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
    output_root: str | Path = "outputs",
    tcn_block_channels: Sequence[int] | None = None,
    tcn_hidden_dim: int | None = None,
    tcn_dropout: float | None = None,
    positive_class_weight_scale: float | None = None,
) -> Cluster1ProposedRunResult:
    resolved_proposed_config_path, proposed_config, cluster_entry = _load_cluster1_proposed_entry(proposed_config_path)

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
    if cluster_id != 1:
        raise ValueError("run_cluster1_proposed only supports Cluster 1.")

    clients, _, data_summary = build_flat_federated_clients(
        cluster_config_path,
        max_train_examples_per_client=max_train_examples_per_client,
        max_eval_examples_per_client=max_eval_examples_per_client,
    )
    computed_positive_class_weight = compute_cluster_positive_class_weight(clients)
    resolved_positive_class_weight_scale = _resolve_positive_class_weight_scale(
        cluster_entry,
        positive_class_weight_scale,
    )
    positive_class_weight = computed_positive_class_weight * resolved_positive_class_weight_scale
    if data_summary["input_adapter"] != "sliding_window_feature_channels":
        raise ValueError("Cluster 1 proposed path requires sliding-window inputs.")

    membership = load_frozen_membership(
        membership_path,
        expected_cluster_id=1,
        expected_n_subclusters=int(cluster_entry["n_subclusters"]),
        expected_client_ids=[client.client_id for client in clients],
    )
    clients_by_subcluster = group_clients_by_subcluster(clients, membership)

    resolved_tcn_hyperparameters = _resolve_tcn_hyperparameters(
        cluster_entry,
        tcn_block_channels=tcn_block_channels,
        tcn_hidden_dim=tcn_hidden_dim,
        tcn_dropout=tcn_dropout,
    )
    model_config = TCNConfig(
        input_channels=int(data_summary["input_channels"]),
        input_length=int(data_summary["input_length"]),
        block_channels=resolved_tcn_hyperparameters["block_channels"],
        hidden_dim=resolved_tcn_hyperparameters["hidden_dim"],
        dropout=resolved_tcn_hyperparameters["dropout"],
    )
    global_model = TCNClassifier(model_config, seed=configured_seed)
    global_state = global_model.state_dict()
    client_local_states = {
        client.client_id: global_model.state_dict()
        for client in clients
    }
    subcluster_states = {
        subcluster.subcluster_id: global_model.state_dict()
        for subcluster in membership.subclusters
    }
    communicated_parameter_bytes = non_bn_parameter_bytes(global_state)

    output_root = Path(output_root)
    run_dir = output_root / "runs" / "P_C1"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = output_root / "metrics" / "P_C1_metrics.csv"
    round_metrics_path = run_dir / "round_metrics.csv"
    summary_path = run_dir / "run_summary.json"
    membership_contents_before = membership.membership_file.read_text(encoding="utf-8")

    started_at = time.perf_counter()
    round_rows: list[dict[str, Any]] = []
    best_round_index = 1
    best_validation_f1 = float("-inf")
    best_test_metrics: Mapping[str, Any] | None = None
    best_test_metrics_default_threshold: Mapping[str, Any] | None = None
    best_validation_metrics: Mapping[str, Any] | None = None
    best_validation_metrics_default_threshold: Mapping[str, Any] | None = None
    best_train_metrics: Mapping[str, Any] | None = None
    best_subcluster_sample_counts: Mapping[str, int] | None = None
    best_subcluster_client_counts: Mapping[str, int] | None = None
    best_selected_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD
    best_validation_labels: np.ndarray | None = None
    best_validation_probabilities: np.ndarray | None = None
    best_test_labels: np.ndarray | None = None
    best_test_probabilities: np.ndarray | None = None

    for round_index in range(1, configured_rounds + 1):
        subcluster_updates: list[WeightedState] = []
        subcluster_losses: list[float] = []
        subcluster_sample_counts: dict[str, int] = {}
        subcluster_client_counts: dict[str, int] = {}

        for subcluster_index, subcluster in enumerate(membership.subclusters):
            subcluster_parent_state = merge_global_non_bn_with_local_bn(
                global_state,
                subcluster_states[subcluster.subcluster_id],
            )
            local_updates: list[WeightedState] = []
            local_losses: list[float] = []

            for client_index, client in enumerate(clients_by_subcluster[subcluster.subcluster_id]):
                result = train_fedbn_client(
                    client,
                    subcluster_parent_state,
                    client_local_states[client.client_id],
                    model_config,
                    local_epochs=configured_local_epochs,
                    batch_size=configured_batch_size,
                    learning_rate=learning_rate,
                    seed=configured_seed + round_index * 1000 + subcluster_index * 100 + client_index,
                    positive_class_weight=positive_class_weight,
                )
                client_local_states[client.client_id] = {
                    key: np.asarray(value, dtype=np.float32).copy()
                    for key, value in result.updated_state.items()
                }
                local_updates.append(result.to_weighted_state(cluster_id=cluster_id))
                local_losses.append(result.train_loss)

            updated_subcluster_state = aggregate_fedbn_leaf_updates(
                local_updates,
                cluster_id=cluster_id,
                reference_state=subcluster_parent_state,
            )
            subcluster_states[subcluster.subcluster_id] = updated_subcluster_state
            subcluster_updates.append(
                WeightedState(
                    cluster_id=cluster_id,
                    contributor_id=subcluster.subcluster_id,
                    num_samples=sum(client.train.num_samples for client in clients_by_subcluster[subcluster.subcluster_id]),
                    state=updated_subcluster_state,
                )
            )
            subcluster_losses.append(float(np.mean(local_losses)))
            subcluster_sample_counts[subcluster.subcluster_id] = sum(
                client.train.num_samples for client in clients_by_subcluster[subcluster.subcluster_id]
            )
            subcluster_client_counts[subcluster.subcluster_id] = len(clients_by_subcluster[subcluster.subcluster_id])

        global_state = aggregate_fedbn_subcluster_updates(
            subcluster_updates,
            cluster_id=cluster_id,
            reference_state=global_state,
        )

        round_evaluation = evaluate_round_with_validation_threshold(
            clients,
            predictor=lambda client, split: predict_split_fedbn(
                merge_global_non_bn_with_local_bn(global_state, client_local_states[client.client_id]),
                model_config,
                split,
            )[0],
        )
        train_metrics = round_evaluation["train_metrics"]
        validation_metrics = round_evaluation["validation_metrics"]
        validation_metrics_default_threshold = round_evaluation["validation_metrics_default_threshold"]
        test_metrics = round_evaluation["test_metrics"]
        test_metrics_default_threshold = round_evaluation["test_metrics_default_threshold"]
        communication_cost_bytes = int(communicated_parameter_bytes * 2 * (len(clients) + membership.n_subclusters))

        round_rows.append(
            _round_row(
                round_index,
                float(np.mean(subcluster_losses)),
                train_metrics,
                validation_metrics,
                test_metrics,
                communication_cost_bytes,
                validation_metrics_default_threshold=validation_metrics_default_threshold,
                test_metrics_default_threshold=test_metrics_default_threshold,
            )
        )

        validation_f1 = validation_metrics["f1"]
        if validation_f1 is not None and float(validation_f1) >= best_validation_f1:
            best_validation_f1 = float(validation_f1)
            best_round_index = round_index
            best_train_metrics = dict(train_metrics)
            best_validation_metrics = dict(validation_metrics)
            best_validation_metrics_default_threshold = dict(validation_metrics_default_threshold)
            best_test_metrics = dict(test_metrics)
            best_test_metrics_default_threshold = dict(test_metrics_default_threshold)
            best_subcluster_sample_counts = dict(subcluster_sample_counts)
            best_subcluster_client_counts = dict(subcluster_client_counts)
            best_selected_threshold = float(round_evaluation["selected_threshold"])
            best_validation_labels = round_evaluation["validation_labels"].copy()
            best_validation_probabilities = round_evaluation["validation_probabilities"].copy()
            best_test_labels = round_evaluation["test_labels"].copy()
            best_test_probabilities = round_evaluation["test_probabilities"].copy()

    elapsed_seconds = float(time.perf_counter() - started_at)
    if (
        best_test_metrics is None
        or best_test_metrics_default_threshold is None
        or best_validation_metrics is None
        or best_validation_metrics_default_threshold is None
        or best_train_metrics is None
        or best_validation_labels is None
        or best_validation_probabilities is None
        or best_test_labels is None
        or best_test_probabilities is None
    ):
        raise ValueError("P_C1: unable to determine best validation round.")
    if best_subcluster_sample_counts is None or best_subcluster_client_counts is None:
        raise ValueError("P_C1: missing best-round subcluster statistics.")

    membership_contents_after = membership.membership_file.read_text(encoding="utf-8")
    if membership_contents_before != membership_contents_after:
        raise ValueError("P_C1: frozen membership file changed during proposed Cluster 1 execution.")

    total_communication_cost_bytes = int(sum(row["communication_cost_bytes"] for row in round_rows))
    prediction_outputs = write_prediction_outputs(
        output_root=output_root,
        experiment_id="P_C1",
        validation_labels=best_validation_labels,
        validation_probabilities=best_validation_probabilities,
        test_labels=best_test_labels,
        test_probabilities=best_test_probabilities,
        selected_threshold=best_selected_threshold,
        seed=configured_seed,
    )
    summary = {
        "experiment_id": "P_C1",
        "cluster_id": cluster_id,
        "dataset": str(cluster_section["dataset_name"]),
        "cluster_config_path": str(cluster_config_path),
        "hierarchy": "hierarchical_fixed",
        "subcluster_layer_used": True,
        "membership_file_used": str(membership.membership_file),
        "membership_hash": membership.membership_hash,
        "membership_file_changed": False,
        "model_family": "tcn",
        "fl_method": "FedBN",
        "aggregation": "weighted_non_bn_mean",
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
        "rounds": configured_rounds,
        "local_epochs": configured_local_epochs,
        "batch_size": configured_batch_size,
        "learning_rate": learning_rate,
        "seed": configured_seed,
        "tcn_block_channels": list(model_config.block_channels),
        "tcn_hidden_dim": model_config.hidden_dim,
        "tcn_dropout": model_config.dropout,
        "computed_positive_class_weight": computed_positive_class_weight,
        "positive_class_weight_scale": resolved_positive_class_weight_scale,
        "positive_class_weight": positive_class_weight,
        "communicated_non_bn_parameter_bytes": communicated_parameter_bytes,
        "communication_cost_per_round_bytes": round_rows[-1]["communication_cost_bytes"],
        "total_communication_cost_bytes": total_communication_cost_bytes,
        "best_validation_round": best_round_index,
        "best_validation_f1": best_validation_metrics["f1"],
        "best_validation_f1_default_threshold": best_validation_metrics_default_threshold["f1"],
        "threshold_used": best_selected_threshold,
        "test_f1_at_best_validation_round": best_test_metrics["f1"],
        "wall_clock_training_seconds": elapsed_seconds,
        "data_summary": data_summary,
        "best_round_train_metrics": best_train_metrics,
        "best_round_validation_metrics_default_threshold": best_validation_metrics_default_threshold,
        "best_round_validation_metrics": best_validation_metrics,
        "best_round_test_metrics_default_threshold": best_test_metrics_default_threshold,
        "best_round_test_metrics": best_test_metrics,
        "best_round_subcluster_train_sample_counts": best_subcluster_sample_counts,
        "best_round_subcluster_client_counts": best_subcluster_client_counts,
        "prediction_outputs": prediction_outputs,
        "round_metrics_path": str(round_metrics_path),
        "metrics_csv_path": str(metrics_csv_path),
    }

    summary_row = {
        "experiment_id": "P_C1",
        "cluster_id": cluster_id,
        "dataset": str(cluster_section["dataset_name"]),
        "hierarchy": "hierarchical_fixed",
        "model_family": "tcn",
        "fl_method": "FedBN",
        "aggregation": "weighted_non_bn_mean",
        "clustering_method": "agglomerative",
        "membership_hash": membership.membership_hash,
        "input_adapter": data_summary["input_adapter"],
        "num_leaf_clients": len(clients),
        "n_subclusters": membership.n_subclusters,
        "rounds": configured_rounds,
        "tcn_block_channels": json.dumps(list(model_config.block_channels)),
        "tcn_hidden_dim": model_config.hidden_dim,
        "tcn_dropout": model_config.dropout,
        "computed_positive_class_weight": computed_positive_class_weight,
        "positive_class_weight_scale": resolved_positive_class_weight_scale,
        "positive_class_weight": positive_class_weight,
        "best_validation_round": best_round_index,
        "best_validation_f1": best_validation_metrics["f1"],
        "best_validation_f1_default_threshold": best_validation_metrics_default_threshold["f1"],
        "threshold_used": best_selected_threshold,
        "test_accuracy": best_test_metrics["accuracy"],
        "test_precision": best_test_metrics["precision"],
        "test_recall": best_test_metrics["recall"],
        "test_f1": best_test_metrics["f1"],
        "test_auroc": best_test_metrics["auroc"],
        "test_pr_auc": best_test_metrics["pr_auc"],
        "test_fpr": best_test_metrics["fpr"],
        "test_confusion_matrix": json.dumps(best_test_metrics["confusion_matrix"]),
        "test_accuracy_default_threshold": best_test_metrics_default_threshold["accuracy"],
        "test_precision_default_threshold": best_test_metrics_default_threshold["precision"],
        "test_recall_default_threshold": best_test_metrics_default_threshold["recall"],
        "test_f1_default_threshold": best_test_metrics_default_threshold["f1"],
        "test_auroc_default_threshold": best_test_metrics_default_threshold["auroc"],
        "test_pr_auc_default_threshold": best_test_metrics_default_threshold["pr_auc"],
        "test_fpr_default_threshold": best_test_metrics_default_threshold["fpr"],
        "test_confusion_matrix_default_threshold": json.dumps(
            best_test_metrics_default_threshold["confusion_matrix"]
        ),
        "communication_cost_per_round_bytes": round_rows[-1]["communication_cost_bytes"],
        "total_communication_cost_bytes": total_communication_cost_bytes,
        "wall_clock_training_seconds": elapsed_seconds,
    }

    _write_round_metrics(round_metrics_path, round_rows)
    _write_summary_metrics(metrics_csv_path, summary_row)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return Cluster1ProposedRunResult(
        experiment_id="P_C1",
        cluster_id=cluster_id,
        dataset=str(cluster_section["dataset_name"]),
        output_dir=run_dir,
        summary_path=summary_path,
        round_metrics_path=round_metrics_path,
        metrics_csv_path=metrics_csv_path,
        summary=summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the proposed Cluster 1 HAI + TCN + FedBN experiment.")
    parser.add_argument(
        "--proposed-config",
        default="configs/proposed.yaml",
        help="Path to configs/proposed.yaml.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke_test_defaults from proposed config.")
    parser.add_argument("--rounds", type=int, help="Optional override for FL rounds.")
    parser.add_argument("--local-epochs", type=int, help="Optional override for local epochs.")
    parser.add_argument("--batch-size", type=int, help="Optional override for local batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Local SGD learning rate.")
    parser.add_argument(
        "--tcn-block-channels",
        nargs=3,
        type=int,
        metavar=("C1", "C2", "C3"),
        help="Optional TCN block channel override for P_C1.",
    )
    parser.add_argument("--tcn-hidden-dim", type=int, help="Optional TCN hidden dimension override for P_C1.")
    parser.add_argument("--tcn-dropout", type=float, help="Optional TCN dropout override for P_C1.")
    parser.add_argument(
        "--positive-class-weight-scale",
        type=float,
        help="Scale applied to the Cluster 1 training-label positive-class weight.",
    )
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
    result = run_cluster1_proposed(
        proposed_config_path=args.proposed_config,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        smoke_test=args.smoke_test,
        max_train_examples_per_client=args.max_train_examples_per_client,
        max_eval_examples_per_client=args.max_eval_examples_per_client,
        output_root=args.output_root,
        tcn_block_channels=args.tcn_block_channels,
        tcn_hidden_dim=args.tcn_hidden_dim,
        tcn_dropout=args.tcn_dropout,
        positive_class_weight_scale=args.positive_class_weight_scale,
    )
    print(f"{result.experiment_id}: wrote {result.metrics_csv_path}")


if __name__ == "__main__":
    main()
