from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data.partitions import PartitionBuildResult, build_candidate_leaf_clients
from src.data.preprocess import RawPreparedDataset, prepare_training_dataset
from src.data.transforms import CategoricalTransformArtifacts, NumericTransformArtifacts
from src.fl.aggregators import WeightedState, aggregate_subcluster_updates_to_maincluster
from src.fl.client import ClientSplit, FlatClientDataset, predict_split, train_flat_client
from src.models.cnn1d import CNN1DClassifier, CNN1DConfig


UNAVAILABLE_METRIC = "metric_unavailable_single_class"


@dataclass(frozen=True)
class FlatMainClusterRunResult:
    experiment_id: str
    cluster_id: int
    dataset: str
    output_dir: Path
    summary_path: Path
    round_metrics_path: Path
    metrics_csv_path: Path
    summary: Mapping[str, Any]


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def _fit_feature_artifacts(
    prepared: RawPreparedDataset,
    partition_result: PartitionBuildResult,
) -> tuple[NumericTransformArtifacts, CategoricalTransformArtifacts]:
    ordered_numeric = prepared.numeric_matrix[partition_result.ordered_row_indices]
    ordered_categorical = prepared.categorical_matrix[partition_result.ordered_row_indices]

    numeric_train_chunks: list[np.ndarray] = []
    categorical_train_chunks: list[np.ndarray] = []
    for client in partition_result.metadata.clients:
        train_range = client.train
        numeric_train_chunks.append(ordered_numeric[train_range.start_index : train_range.end_index])
        categorical_train_chunks.append(ordered_categorical[train_range.start_index : train_range.end_index])

    if prepared.numeric_input_feature_names:
        cluster_train_numeric = np.vstack(numeric_train_chunks)
        numeric_artifacts = NumericTransformArtifacts.fit(
            prepared.numeric_input_feature_names,
            cluster_train_numeric,
        )
    else:
        numeric_artifacts = NumericTransformArtifacts.fit(
            prepared.numeric_input_feature_names,
            np.empty((partition_result.metadata.total_ordered_samples, 0), dtype=np.float32),
        )

    if prepared.categorical_input_feature_names:
        cluster_train_categorical = np.vstack(categorical_train_chunks)
        categorical_artifacts = CategoricalTransformArtifacts.fit(
            prepared.categorical_input_feature_names,
            cluster_train_categorical,
        )
    else:
        categorical_artifacts = CategoricalTransformArtifacts.fit(
            prepared.categorical_input_feature_names,
            np.empty((partition_result.metadata.total_ordered_samples, 0), dtype=object),
        )

    return numeric_artifacts, categorical_artifacts


def _transform_rows(
    numeric_artifacts: NumericTransformArtifacts,
    categorical_artifacts: CategoricalTransformArtifacts,
    numeric_rows: np.ndarray,
    categorical_rows: np.ndarray,
) -> np.ndarray:
    numeric_output = numeric_artifacts.transform(numeric_rows)
    categorical_output = categorical_artifacts.transform(categorical_rows)
    if numeric_output.size and categorical_output.size:
        return np.hstack([numeric_output, categorical_output])
    if numeric_output.size:
        return numeric_output
    if categorical_output.size:
        return categorical_output
    return np.empty((numeric_rows.shape[0], 0), dtype=np.float32)


def _cap_split(split: ClientSplit, limit: int | None) -> ClientSplit:
    if limit is None or split.num_samples <= limit:
        return split
    return ClientSplit(
        inputs=split.inputs[:limit].copy(),
        labels=split.labels[:limit].copy(),
    )


def _sliding_window_split(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    window_length: int,
    stride: int,
    label_rule: str,
    client_id: str,
    split_name: str,
) -> ClientSplit:
    if rows.shape[0] < window_length:
        raise ValueError(
            f"{client_id} {split_name}: requires at least {window_length} rows for Cluster 1 sliding windows, "
            f"observed {rows.shape[0]}."
        )
    if label_rule != "any_positive_row":
        raise ValueError(f"Unsupported Cluster 1 window_label_rule: {label_rule}")

    feature_windows = np.lib.stride_tricks.sliding_window_view(
        rows,
        window_shape=window_length,
        axis=0,
    )[::stride]
    label_windows = np.lib.stride_tricks.sliding_window_view(
        labels,
        window_shape=window_length,
        axis=0,
    )[::stride]
    window_labels = (label_windows.max(axis=1) > 0).astype(np.int8, copy=False)
    return ClientSplit(
        inputs=feature_windows.astype(np.float32, copy=False),
        labels=window_labels,
    )


def _feature_vector_split(rows: np.ndarray, labels: np.ndarray) -> ClientSplit:
    return ClientSplit(
        inputs=rows[:, None, :].astype(np.float32, copy=False),
        labels=labels.astype(np.int8, copy=False),
    )


def _cluster_input_adapter(
    cluster_id: int,
    cluster_yaml: Mapping[str, Any],
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    client_id: str,
    split_name: str,
) -> tuple[ClientSplit, str]:
    if cluster_id == 1:
        preprocessing = cluster_yaml.get("preprocessing")
        if not isinstance(preprocessing, Mapping):
            raise ValueError("Cluster 1 config must define preprocessing window settings.")
        window_length = int(preprocessing.get("window_length", 0))
        stride = int(preprocessing.get("stride", 0))
        label_rule = str(preprocessing.get("window_label_rule", ""))
        if window_length <= 0 or stride <= 0:
            raise ValueError("Cluster 1 window_length and stride must be positive integers.")
        return (
            _sliding_window_split(
                rows,
                labels,
                window_length=window_length,
                stride=stride,
                label_rule=label_rule,
                client_id=client_id,
                split_name=split_name,
            ),
            "sliding_window_feature_channels",
        )

    return _feature_vector_split(rows, labels), "feature_vector_as_sequence"


def build_flat_federated_clients(
    cluster_config_path: str | Path,
    *,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
) -> tuple[list[FlatClientDataset], CNN1DConfig, Mapping[str, Any]]:
    cluster_yaml = _load_yaml(cluster_config_path)
    prepared = prepare_training_dataset(cluster_config_path)
    partition_result = build_candidate_leaf_clients(cluster_config_path)
    numeric_artifacts, categorical_artifacts = _fit_feature_artifacts(prepared, partition_result)

    ordered_numeric = prepared.numeric_matrix[partition_result.ordered_row_indices]
    ordered_categorical = prepared.categorical_matrix[partition_result.ordered_row_indices]
    ordered_labels = prepared.labels[partition_result.ordered_row_indices]

    clients: list[FlatClientDataset] = []
    input_shape: tuple[int, int] | None = None
    adapter_name: str | None = None

    for client_meta in partition_result.metadata.clients:
        def split_rows(split_meta: Any) -> tuple[np.ndarray, np.ndarray]:
            numeric_rows = ordered_numeric[split_meta.start_index : split_meta.end_index]
            categorical_rows = ordered_categorical[split_meta.start_index : split_meta.end_index]
            labels = ordered_labels[split_meta.start_index : split_meta.end_index]
            features = _transform_rows(
                numeric_artifacts,
                categorical_artifacts,
                numeric_rows,
                categorical_rows,
            )
            return features, labels

        train_rows, train_labels = split_rows(client_meta.train)
        val_rows, val_labels = split_rows(client_meta.validation)
        test_rows, test_labels = split_rows(client_meta.test)

        train_split, current_adapter = _cluster_input_adapter(
            prepared.inspection.config.cluster_id,
            cluster_yaml,
            train_rows,
            train_labels,
            client_id=client_meta.client_id,
            split_name="train",
        )
        val_split, _ = _cluster_input_adapter(
            prepared.inspection.config.cluster_id,
            cluster_yaml,
            val_rows,
            val_labels,
            client_id=client_meta.client_id,
            split_name="validation",
        )
        test_split, _ = _cluster_input_adapter(
            prepared.inspection.config.cluster_id,
            cluster_yaml,
            test_rows,
            test_labels,
            client_id=client_meta.client_id,
            split_name="test",
        )

        train_split = _cap_split(train_split, max_train_examples_per_client)
        val_split = _cap_split(val_split, max_eval_examples_per_client)
        test_split = _cap_split(test_split, max_eval_examples_per_client)

        if train_split.num_samples <= 0:
            raise ValueError(f"{client_meta.client_id}: flat baseline requires at least one train sample.")

        current_shape = (train_split.inputs.shape[1], train_split.inputs.shape[2])
        if input_shape is None:
            input_shape = current_shape
            adapter_name = current_adapter
        elif current_shape != input_shape:
            raise ValueError(
                f"{prepared.inspection.config.dataset_name}: inconsistent input shapes across clients. "
                f"Observed {current_shape} vs expected {input_shape}."
            )

        clients.append(
            FlatClientDataset(
                cluster_id=prepared.inspection.config.cluster_id,
                client_id=client_meta.client_id,
                train=train_split,
                validation=val_split,
                test=test_split,
                input_adapter=current_adapter,
            )
        )

    assert input_shape is not None
    assert adapter_name is not None
    model_config = CNN1DConfig(
        input_channels=input_shape[0],
        input_length=input_shape[1],
    )

    data_summary = {
        "cluster_id": prepared.inspection.config.cluster_id,
        "dataset": prepared.inspection.config.dataset_name,
        "num_clients": len(clients),
        "input_adapter": adapter_name,
        "input_channels": model_config.input_channels,
        "input_length": model_config.input_length,
        "retained_input_feature_columns": list(prepared.retained_input_feature_names),
        "model_feature_columns": list(
            numeric_artifacts.kept_features + categorical_artifacts.output_features
        ),
        "dropped_all_missing_feature_columns": list(numeric_artifacts.dropped_all_missing_features),
        "dropped_constant_feature_columns": list(
            numeric_artifacts.dropped_constant_features + categorical_artifacts.dropped_constant_features
        ),
        "dropped_high_cardinality_categorical_columns": list(
            categorical_artifacts.dropped_high_cardinality_features
        ),
        "max_train_examples_per_client": max_train_examples_per_client,
        "max_eval_examples_per_client": max_eval_examples_per_client,
        "client_train_sample_counts": {
            client.client_id: client.train.num_samples for client in clients
        },
        "client_validation_sample_counts": {
            client.client_id: client.validation.num_samples for client in clients
        },
        "client_test_sample_counts": {
            client.client_id: client.test.num_samples for client in clients
        },
    }
    return clients, model_config, data_summary


def _split_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    if labels.size == 0:
        return {
            "loss": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auroc": UNAVAILABLE_METRIC,
            "pr_auc": UNAVAILABLE_METRIC,
            "fpr": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "support": 0,
        }

    predictions = (probabilities >= 0.5).astype(np.int8, copy=False)
    labels = labels.astype(np.int8, copy=False)
    loss = float(
        -np.mean(
            labels * np.log(np.clip(probabilities, 1e-6, 1.0 - 1e-6))
            + (1 - labels) * np.log(np.clip(1.0 - probabilities, 1e-6, 1.0 - 1e-6))
        )
    )
    accuracy = float(accuracy_score(labels, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0

    unique_labels = set(labels.tolist())
    if unique_labels == {0, 1}:
        auroc: float | str = float(roc_auc_score(labels, probabilities))
        pr_auc: float | str = float(average_precision_score(labels, probabilities))
    else:
        auroc = UNAVAILABLE_METRIC
        pr_auc = UNAVAILABLE_METRIC

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "confusion_matrix": matrix.astype(int).tolist(),
        "support": int(labels.shape[0]),
        "label_counts": {key: int(value) for key, value in sorted(Counter(labels.tolist()).items())},
    }


def _evaluate_cluster_split(
    clients: Sequence[FlatClientDataset],
    state: Mapping[str, np.ndarray],
    model_config: CNN1DConfig,
    *,
    split_name: str,
) -> dict[str, Any]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for client in clients:
        split = getattr(client, split_name)
        split_probabilities, _ = predict_split(state, model_config, split)
        if split_probabilities.size == 0:
            continue
        probabilities.append(split_probabilities)
        labels.append(split.labels.astype(np.int8, copy=False))

    if not probabilities:
        return _split_metrics(np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32))

    return _split_metrics(np.concatenate(labels), np.concatenate(probabilities))


def _round_row(
    round_index: int,
    train_loss: float,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    communication_cost_bytes: int,
) -> dict[str, Any]:
    return {
        "round": round_index,
        "train_loss_local_mean": train_loss,
        "train_accuracy": train_metrics["accuracy"],
        "train_f1": train_metrics["f1"],
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_f1": validation_metrics["f1"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "communication_cost_bytes": communication_cost_bytes,
    }


def _write_round_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Round metrics rows must not be empty.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_metrics(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def run_flat_maincluster_experiment(
    *,
    experiment_id: str,
    cluster_config_path: str | Path,
    output_root: str | Path = "outputs",
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
) -> FlatMainClusterRunResult:
    cluster_yaml = _load_yaml(cluster_config_path)
    cluster_section = cluster_yaml.get("cluster")
    if not isinstance(cluster_section, Mapping):
        raise ValueError(f"Cluster config {cluster_config_path} is missing cluster metadata.")

    clients, model_config, data_summary = build_flat_federated_clients(
        cluster_config_path,
        max_train_examples_per_client=max_train_examples_per_client,
        max_eval_examples_per_client=max_eval_examples_per_client,
    )

    output_root = Path(output_root)
    run_dir = output_root / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = output_root / "metrics" / f"{experiment_id}_metrics.csv"
    round_metrics_path = run_dir / "round_metrics.csv"
    summary_path = run_dir / "run_summary.json"

    global_model = CNN1DClassifier(model_config, seed=seed)
    global_state = global_model.state_dict()
    parameter_bytes = global_model.parameter_bytes()

    started_at = time.perf_counter()
    round_rows: list[dict[str, Any]] = []
    best_round_index = 1
    best_validation_f1 = float("-inf")
    best_test_metrics: Mapping[str, Any] | None = None
    best_validation_metrics: Mapping[str, Any] | None = None
    best_train_metrics: Mapping[str, Any] | None = None

    for round_index in range(1, rounds + 1):
        local_updates: list[WeightedState] = []
        local_losses: list[float] = []
        for client_index, client in enumerate(clients):
            result = train_flat_client(
                client,
                global_state,
                model_config,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed + round_index * 1000 + client_index,
            )
            local_updates.append(result.to_weighted_state(cluster_id=client.cluster_id))
            local_losses.append(result.train_loss)

        global_state = aggregate_subcluster_updates_to_maincluster(
            local_updates,
            expected_cluster_id=int(cluster_section["id"]),
        )
        train_metrics = _evaluate_cluster_split(clients, global_state, model_config, split_name="train")
        validation_metrics = _evaluate_cluster_split(clients, global_state, model_config, split_name="validation")
        test_metrics = _evaluate_cluster_split(clients, global_state, model_config, split_name="test")
        communication_cost_bytes = int(parameter_bytes * len(clients) * 2)

        round_rows.append(
            _round_row(
                round_index,
                float(np.mean(local_losses)),
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

    elapsed_seconds = float(time.perf_counter() - started_at)
    if best_test_metrics is None or best_validation_metrics is None or best_train_metrics is None:
        raise ValueError(f"{experiment_id}: unable to determine best validation round.")

    total_communication_cost_bytes = int(sum(row["communication_cost_bytes"] for row in round_rows))
    summary = {
        "experiment_id": experiment_id,
        "cluster_id": int(cluster_section["id"]),
        "dataset": str(cluster_section["dataset_name"]),
        "cluster_config_path": str(cluster_config_path),
        "hierarchy": "flat",
        "subcluster_layer_used": False,
        "membership_file_used": None,
        "model_family": "cnn1d",
        "fl_method": "FedAvg",
        "aggregation": "weighted_arithmetic_mean",
        "input_adapter": data_summary["input_adapter"],
        "input_channels": data_summary["input_channels"],
        "input_length": data_summary["input_length"],
        "num_leaf_clients": len(clients),
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
        "round_metrics_path": str(round_metrics_path),
        "metrics_csv_path": str(metrics_csv_path),
    }

    summary_row = {
        "experiment_id": experiment_id,
        "cluster_id": int(cluster_section["id"]),
        "dataset": str(cluster_section["dataset_name"]),
        "hierarchy": "flat",
        "model_family": "cnn1d",
        "fl_method": "FedAvg",
        "aggregation": "weighted_arithmetic_mean",
        "input_adapter": data_summary["input_adapter"],
        "num_leaf_clients": len(clients),
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

    return FlatMainClusterRunResult(
        experiment_id=experiment_id,
        cluster_id=int(cluster_section["id"]),
        dataset=str(cluster_section["dataset_name"]),
        output_dir=run_dir,
        summary_path=summary_path,
        round_metrics_path=round_metrics_path,
        metrics_csv_path=metrics_csv_path,
        summary=summary,
    )
