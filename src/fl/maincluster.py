from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5


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


def compute_positive_class_weight_from_labels(labels: np.ndarray) -> float:
    flattened = np.asarray(labels, dtype=np.int8).reshape(-1)
    positives = int(np.sum(flattened == 1))
    negatives = int(np.sum(flattened == 0))
    if positives <= 0 or negatives <= 0:
        raise ValueError(
            "Positive-class weighting requires both classes to be present in training labels. "
            f"Observed positives={positives}, negatives={negatives}."
        )
    return float(negatives / positives)


def compute_cluster_positive_class_weight(clients: Sequence[FlatClientDataset]) -> float:
    if not clients:
        raise ValueError("Positive-class weighting requires at least one client.")
    train_labels = [
        client.train.labels.astype(np.int8, copy=False)
        for client in clients
        if client.train.num_samples > 0
    ]
    if not train_labels:
        raise ValueError("Positive-class weighting requires at least one non-empty client train split.")
    return compute_positive_class_weight_from_labels(np.concatenate(train_labels))


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
    *,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
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
            "threshold_used": float(threshold),
        }

    predictions = (probabilities >= threshold).astype(np.int8, copy=False)
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
        "threshold_used": float(threshold),
    }


def _collect_split_outputs(
    clients: Sequence[Any],
    *,
    split_name: str,
    predictor: Callable[[Any, Any], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for client in clients:
        split = getattr(client, split_name)
        split_probabilities = predictor(client, split)
        if split_probabilities.size == 0:
            continue
        probabilities.append(split_probabilities.astype(np.float32, copy=False))
        labels.append(split.labels.astype(np.int8, copy=False))

    if not probabilities:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32)
    return np.concatenate(labels), np.concatenate(probabilities)


def select_threshold_maximizing_validation_f1(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    default_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> float:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if labels.size == 0 or probabilities.size == 0:
        return float(default_threshold)

    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives <= 0 or negatives <= 0:
        return float(default_threshold)

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    cumulative_true_positives = np.cumsum(sorted_labels == 1)
    cumulative_false_positives = np.cumsum(sorted_labels == 0)
    unique_last_indices = np.flatnonzero(
        np.r_[sorted_probabilities[1:] != sorted_probabilities[:-1], True]
    )

    tp = cumulative_true_positives[unique_last_indices].astype(np.float64, copy=False)
    fp = cumulative_false_positives[unique_last_indices].astype(np.float64, copy=False)
    fn = float(positives) - tp
    denominator = 2.0 * tp + fp + fn
    f1_scores = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator > 0.0,
    )
    thresholds = sorted_probabilities[unique_last_indices].astype(np.float64, copy=False)

    best_f1 = float(np.max(f1_scores))
    candidate_indices = np.flatnonzero(np.isclose(f1_scores, best_f1))
    candidate_thresholds = thresholds[candidate_indices]
    default_distance = np.abs(candidate_thresholds - float(default_threshold))
    best_index = candidate_indices[int(np.argmin(default_distance))]
    return float(thresholds[best_index])


def evaluate_round_with_validation_threshold(
    clients: Sequence[Any],
    *,
    predictor: Callable[[Any, Any], np.ndarray],
) -> dict[str, Any]:
    train_labels, train_probabilities = _collect_split_outputs(
        clients,
        split_name="train",
        predictor=predictor,
    )
    validation_labels, validation_probabilities = _collect_split_outputs(
        clients,
        split_name="validation",
        predictor=predictor,
    )
    selected_threshold = select_threshold_maximizing_validation_f1(
        validation_labels,
        validation_probabilities,
    )
    test_labels, test_probabilities = _collect_split_outputs(
        clients,
        split_name="test",
        predictor=predictor,
    )

    return {
        "train_metrics": _split_metrics(
            train_labels,
            train_probabilities,
            threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        ),
        "validation_metrics_default_threshold": _split_metrics(
            validation_labels,
            validation_probabilities,
            threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        ),
        "validation_metrics": _split_metrics(
            validation_labels,
            validation_probabilities,
            threshold=selected_threshold,
        ),
        "test_metrics_default_threshold": _split_metrics(
            test_labels,
            test_probabilities,
            threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        ),
        "test_metrics": _split_metrics(
            test_labels,
            test_probabilities,
            threshold=selected_threshold,
        ),
        "selected_threshold": float(selected_threshold),
        "validation_labels": validation_labels,
        "validation_probabilities": validation_probabilities,
        "test_labels": test_labels,
        "test_probabilities": test_probabilities,
    }


def _prediction_file_suffix(seed: int | None) -> str:
    return f"_seed_{int(seed)}" if seed is not None else ""


def write_prediction_outputs(
    *,
    output_root: str | Path,
    experiment_id: str,
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    selected_threshold: float,
    seed: int | None = None,
) -> dict[str, str]:
    prediction_dir = Path(output_root) / "predictions" / experiment_id
    prediction_dir.mkdir(parents=True, exist_ok=True)
    suffix = _prediction_file_suffix(seed)
    validation_path = prediction_dir / f"validation_predictions{suffix}.npz"
    test_path = prediction_dir / f"test_predictions{suffix}.npz"
    threshold_path = prediction_dir / f"selected_threshold{suffix}.json"

    np.savez_compressed(
        validation_path,
        probabilities=np.asarray(validation_probabilities, dtype=np.float32),
        labels=np.asarray(validation_labels, dtype=np.int8),
    )
    np.savez_compressed(
        test_path,
        probabilities=np.asarray(test_probabilities, dtype=np.float32),
        labels=np.asarray(test_labels, dtype=np.int8),
    )
    threshold_payload = {
        "experiment_id": experiment_id,
        "seed": int(seed) if seed is not None else None,
        "threshold_selected_on": "validation",
        "selection_metric": "f1",
        "default_threshold": float(DEFAULT_CLASSIFICATION_THRESHOLD),
        "selected_threshold": float(selected_threshold),
    }
    threshold_path.write_text(json.dumps(threshold_payload, indent=2) + "\n", encoding="utf-8")
    return {
        "prediction_dir": str(prediction_dir),
        "validation_predictions_path": str(validation_path),
        "test_predictions_path": str(test_path),
        "selected_threshold_path": str(threshold_path),
    }


def _evaluate_cluster_split(
    clients: Sequence[FlatClientDataset],
    state: Mapping[str, np.ndarray],
    model_config: CNN1DConfig,
    *,
    split_name: str,
) -> dict[str, Any]:
    labels, probabilities = _collect_split_outputs(
        clients,
        split_name=split_name,
        predictor=lambda _client, split: predict_split(state, model_config, split)[0],
    )
    return _split_metrics(labels, probabilities, threshold=DEFAULT_CLASSIFICATION_THRESHOLD)


def _round_row(
    round_index: int,
    train_loss: float,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    communication_cost_bytes: int,
    *,
    validation_metrics_default_threshold: Mapping[str, Any] | None = None,
    test_metrics_default_threshold: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "round": round_index,
        "train_loss_local_mean": train_loss,
        "train_accuracy": train_metrics["accuracy"],
        "train_f1": train_metrics["f1"],
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "validation_f1": validation_metrics["f1"],
        "validation_auroc": validation_metrics["auroc"],
        "validation_pr_auc": validation_metrics["pr_auc"],
        "validation_fpr": validation_metrics["fpr"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_auroc": test_metrics["auroc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_fpr": test_metrics["fpr"],
        "threshold_used": test_metrics["threshold_used"],
        "communication_cost_bytes": communication_cost_bytes,
    }
    if validation_metrics_default_threshold is not None:
        row.update(
            {
                "validation_accuracy_default_threshold": validation_metrics_default_threshold["accuracy"],
                "validation_precision_default_threshold": validation_metrics_default_threshold["precision"],
                "validation_recall_default_threshold": validation_metrics_default_threshold["recall"],
                "validation_f1_default_threshold": validation_metrics_default_threshold["f1"],
                "validation_auroc_default_threshold": validation_metrics_default_threshold["auroc"],
                "validation_pr_auc_default_threshold": validation_metrics_default_threshold["pr_auc"],
                "validation_fpr_default_threshold": validation_metrics_default_threshold["fpr"],
            }
        )
    if test_metrics_default_threshold is not None:
        row.update(
            {
                "test_accuracy_default_threshold": test_metrics_default_threshold["accuracy"],
                "test_precision_default_threshold": test_metrics_default_threshold["precision"],
                "test_recall_default_threshold": test_metrics_default_threshold["recall"],
                "test_f1_default_threshold": test_metrics_default_threshold["f1"],
                "test_auroc_default_threshold": test_metrics_default_threshold["auroc"],
                "test_pr_auc_default_threshold": test_metrics_default_threshold["pr_auc"],
                "test_fpr_default_threshold": test_metrics_default_threshold["fpr"],
            }
        )
    return row


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
    positive_class_weight = compute_cluster_positive_class_weight(clients)

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
    best_test_metrics_default_threshold: Mapping[str, Any] | None = None
    best_validation_metrics: Mapping[str, Any] | None = None
    best_validation_metrics_default_threshold: Mapping[str, Any] | None = None
    best_train_metrics: Mapping[str, Any] | None = None
    best_selected_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD
    best_validation_labels: np.ndarray | None = None
    best_validation_probabilities: np.ndarray | None = None
    best_test_labels: np.ndarray | None = None
    best_test_probabilities: np.ndarray | None = None

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
                positive_class_weight=positive_class_weight,
            )
            local_updates.append(result.to_weighted_state(cluster_id=client.cluster_id))
            local_losses.append(result.train_loss)

        global_state = aggregate_subcluster_updates_to_maincluster(
            local_updates,
            expected_cluster_id=int(cluster_section["id"]),
        )
        round_evaluation = evaluate_round_with_validation_threshold(
            clients,
            predictor=lambda _client, split: predict_split(global_state, model_config, split)[0],
        )
        train_metrics = round_evaluation["train_metrics"]
        validation_metrics = round_evaluation["validation_metrics"]
        validation_metrics_default_threshold = round_evaluation["validation_metrics_default_threshold"]
        test_metrics = round_evaluation["test_metrics"]
        test_metrics_default_threshold = round_evaluation["test_metrics_default_threshold"]
        communication_cost_bytes = int(parameter_bytes * len(clients) * 2)

        round_rows.append(
            _round_row(
                round_index,
                float(np.mean(local_losses)),
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
        raise ValueError(f"{experiment_id}: unable to determine best validation round.")

    total_communication_cost_bytes = int(sum(row["communication_cost_bytes"] for row in round_rows))
    prediction_outputs = write_prediction_outputs(
        output_root=output_root,
        experiment_id=experiment_id,
        validation_labels=best_validation_labels,
        validation_probabilities=best_validation_probabilities,
        test_labels=best_test_labels,
        test_probabilities=best_test_probabilities,
        selected_threshold=best_selected_threshold,
        seed=seed,
    )
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
        "positive_class_weight": positive_class_weight,
        "model_parameter_bytes": parameter_bytes,
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
        "prediction_outputs": prediction_outputs,
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
