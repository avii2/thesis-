from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.fl.aggregators import WeightedState, aggregate_leaf_updates_to_subcluster, aggregate_subcluster_updates_to_maincluster
from src.fl.client import ClientSplit, FlatClientDataset, LocalTrainingResult
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
from src.models.mlp import CompactMLPClassifier, MLPConfig, STATE_KEYS
from src.models.tcn import TCNClassifier, TCNConfig
from src.train_hierarchical_baseline import run_hierarchical_baseline_experiment


@dataclass(frozen=True)
class AblationRunResult:
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


def _configured_defaults(
    config: Mapping[str, Any],
    *,
    smoke_test: bool,
    rounds: int | None,
    local_epochs: int | None,
    batch_size: int | None,
    seed: int | None,
) -> tuple[int, int, int, int]:
    defaults_key = "smoke_test_defaults" if smoke_test else "training_defaults"
    defaults = config.get(defaults_key)
    if not isinstance(defaults, Mapping):
        raise ValueError(f"Ablation config is missing {defaults_key}.")

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
    return configured_rounds, configured_local_epochs, configured_batch_size, configured_seed


def _load_custom_control_entry(
    config_path: str | Path,
    *,
    expected_experiment_id: str,
    expected_run_source: str,
    expected_cluster_id: int,
    expected_model_family: str,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    resolved_config_path = Path(config_path).resolve()
    config = _load_yaml(resolved_config_path)
    comparisons = config.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 1:
        raise ValueError(f"{resolved_config_path}: custom ablation config must contain exactly one comparison.")
    comparison = comparisons[0]
    if not isinstance(comparison, Mapping):
        raise ValueError(f"{resolved_config_path}: comparison entry must be a mapping.")

    control = comparison.get("control")
    if not isinstance(control, Mapping):
        raise ValueError(f"{resolved_config_path}: comparison.control must be a mapping.")

    if str(control.get("experiment_id")) != expected_experiment_id:
        raise ValueError(
            f"{resolved_config_path}: expected control experiment_id={expected_experiment_id!r}, "
            f"observed {control.get('experiment_id')!r}."
        )
    if str(control.get("run_source")) != expected_run_source:
        raise ValueError(
            f"{resolved_config_path}: expected control run_source={expected_run_source!r}, "
            f"observed {control.get('run_source')!r}."
        )
    if int(control.get("cluster_id")) != expected_cluster_id:
        raise ValueError(
            f"{resolved_config_path}: expected cluster_id={expected_cluster_id}, observed {control.get('cluster_id')!r}."
        )
    if str(control.get("model_family")) != expected_model_family:
        raise ValueError(
            f"{resolved_config_path}: expected model_family={expected_model_family!r}, "
            f"observed {control.get('model_family')!r}."
        )

    return resolved_config_path, config, comparison, control


def _train_tcn_fedavg_client(
    client: FlatClientDataset,
    parent_state: Mapping[str, np.ndarray],
    model_config: TCNConfig,
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    positive_class_weight: float = 1.0,
) -> LocalTrainingResult:
    if client.num_train_samples <= 0:
        raise ValueError(f"{client.client_id}: train split must contain at least one sample.")

    model = TCNClassifier.from_state(model_config, parent_state, seed=seed)
    rng = np.random.default_rng(seed)
    epoch_losses: list[float] = []
    for _ in range(local_epochs):
        epoch_losses.append(
            model.train_epoch(
                client.train.inputs,
                client.train.labels,
                batch_size=batch_size,
                learning_rate=learning_rate,
                rng=rng,
                positive_class_weight=positive_class_weight,
            )
        )
    return LocalTrainingResult(
        client_id=client.client_id,
        num_train_samples=client.num_train_samples,
        train_loss=float(np.mean(epoch_losses)),
        updated_state=model.state_dict(),
    )


def _predict_tcn_split(
    state: Mapping[str, np.ndarray],
    model_config: TCNConfig,
    split: ClientSplit,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    if split.num_samples == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int8)

    model = TCNClassifier.from_state(model_config, state)
    probabilities = model.predict_proba(split.inputs)
    predictions = (probabilities >= threshold).astype(np.int8, copy=False)
    return probabilities, predictions


def _evaluate_tcn_cluster_split(
    clients: list[FlatClientDataset],
    *,
    state: Mapping[str, np.ndarray],
    model_config: TCNConfig,
    split_name: str,
) -> dict[str, Any]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for client in clients:
        split = getattr(client, split_name)
        split_probabilities, _ = _predict_tcn_split(state, model_config, split)
        if split_probabilities.size == 0:
            continue
        probabilities.append(split_probabilities)
        labels.append(split.labels.astype(np.int8, copy=False))
    if not probabilities:
        return _split_metrics(np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32))
    return _split_metrics(np.concatenate(labels), np.concatenate(probabilities))


def _as_tabular_inputs(split: ClientSplit) -> np.ndarray:
    inputs = split.inputs.astype(np.float32, copy=False)
    if inputs.ndim == 2:
        return inputs
    if inputs.ndim == 3 and inputs.shape[1] == 1:
        return inputs[:, 0, :]
    raise ValueError(
        "Compact MLP FedAvg ablation expects inputs with shape (batch, features) or "
        f"(batch, 1, features). Observed {inputs.shape}."
    )


def _train_mlp_fedavg_client(
    client: FlatClientDataset,
    parent_state: Mapping[str, np.ndarray],
    model_config: MLPConfig,
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    positive_class_weight: float = 1.0,
) -> LocalTrainingResult:
    if client.num_train_samples <= 0:
        raise ValueError(f"{client.client_id}: train split must contain at least one sample.")

    model = CompactMLPClassifier.from_state(model_config, parent_state, seed=seed)
    optimizer_state: dict[str, dict[str, np.ndarray] | int] = {"t": 0}
    rng = np.random.default_rng(seed)
    train_inputs = _as_tabular_inputs(client.train)
    train_labels = client.train.labels.astype(np.float32, copy=False)
    batch_losses: list[float] = []

    for _ in range(local_epochs):
        indices = rng.permutation(train_inputs.shape[0])
        for start in range(0, train_inputs.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_inputs = train_inputs[batch_indices]
            batch_labels = train_labels[batch_indices]
            loss, gradients = model.loss_and_gradients(
                batch_inputs,
                batch_labels,
                rng=rng,
                positive_class_weight=positive_class_weight,
            )
            model.apply_adam_gradients(
                {
                    key: np.asarray(gradients[key], dtype=np.float32)
                    for key in STATE_KEYS
                },
                optimizer_state,
                learning_rate=learning_rate,
            )
            batch_losses.append(loss)

    return LocalTrainingResult(
        client_id=client.client_id,
        num_train_samples=client.num_train_samples,
        train_loss=float(np.mean(batch_losses)),
        updated_state=model.state_dict(),
    )


def _predict_mlp_split(
    state: Mapping[str, np.ndarray],
    model_config: MLPConfig,
    split: ClientSplit,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    if split.num_samples == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int8)

    model = CompactMLPClassifier.from_state(model_config, state)
    probabilities = model.predict_proba(_as_tabular_inputs(split))
    predictions = (probabilities >= threshold).astype(np.int8, copy=False)
    return probabilities, predictions


def _evaluate_mlp_cluster_split(
    clients: list[FlatClientDataset],
    *,
    state: Mapping[str, np.ndarray],
    model_config: MLPConfig,
    split_name: str,
) -> dict[str, Any]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for client in clients:
        split = getattr(client, split_name)
        split_probabilities, _ = _predict_mlp_split(state, model_config, split)
        if split_probabilities.size == 0:
            continue
        probabilities.append(split_probabilities)
        labels.append(split.labels.astype(np.int8, copy=False))
    if not probabilities:
        return _split_metrics(np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32))
    return _split_metrics(np.concatenate(labels), np.concatenate(probabilities))


def run_cluster1_fedavg_tcn_ablation(
    ablation_config_path: str | Path = "configs/ablation_cluster1_fedbn.yaml",
    *,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    smoke_test: bool = False,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
    output_root: str | Path = "outputs",
) -> AblationRunResult:
    resolved_config_path, config, comparison, control = _load_custom_control_entry(
        ablation_config_path,
        expected_experiment_id="AB_C1_FEDAVG_TCN",
        expected_run_source="custom_cluster1_fedavg_tcn",
        expected_cluster_id=1,
        expected_model_family="tcn",
    )
    configured_rounds, configured_local_epochs, configured_batch_size, configured_seed = _configured_defaults(
        config,
        smoke_test=smoke_test,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    learning_rate = float(control["learning_rate"])
    cluster_config_path = _resolve_path(resolved_config_path, str(control["cluster_config"]))
    membership_path = _resolve_path(resolved_config_path, str(control["membership_file"]))

    cluster_yaml = _load_yaml(cluster_config_path)
    cluster_section = cluster_yaml.get("cluster")
    if not isinstance(cluster_section, Mapping):
        raise ValueError(f"{cluster_config_path}: cluster metadata is missing.")
    cluster_id = int(cluster_section["id"])
    if cluster_id != 1:
        raise ValueError("AB_C1_FEDAVG_TCN only supports Cluster 1.")

    clients, _, data_summary = build_flat_federated_clients(
        cluster_config_path,
        max_train_examples_per_client=max_train_examples_per_client,
        max_eval_examples_per_client=max_eval_examples_per_client,
    )
    positive_class_weight = compute_cluster_positive_class_weight(clients)
    if data_summary["input_adapter"] != "sliding_window_feature_channels":
        raise ValueError("AB_C1_FEDAVG_TCN requires sliding-window Cluster 1 inputs.")

    membership = load_frozen_membership(
        membership_path,
        expected_cluster_id=1,
        expected_n_subclusters=int(control["n_subclusters"]),
        expected_client_ids=[client.client_id for client in clients],
    )
    clients_by_subcluster = group_clients_by_subcluster(clients, membership)

    model_config = TCNConfig(
        input_channels=int(data_summary["input_channels"]),
        input_length=int(data_summary["input_length"]),
    )
    global_model = TCNClassifier(model_config, seed=configured_seed)
    global_state = global_model.state_dict()
    parameter_bytes = global_model.parameter_bytes()

    output_root = Path(output_root)
    run_dir = output_root / "runs" / "AB_C1_FEDAVG_TCN"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = output_root / "metrics" / "AB_C1_FEDAVG_TCN_metrics.csv"
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
            local_updates: list[WeightedState] = []
            local_losses: list[float] = []
            subcluster_clients = clients_by_subcluster[subcluster.subcluster_id]

            for client_index, client in enumerate(subcluster_clients):
                result = _train_tcn_fedavg_client(
                    client,
                    global_state,
                    model_config,
                    local_epochs=configured_local_epochs,
                    batch_size=configured_batch_size,
                    learning_rate=learning_rate,
                    seed=configured_seed + round_index * 1000 + subcluster_index * 100 + client_index,
                    positive_class_weight=positive_class_weight,
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
        round_evaluation = evaluate_round_with_validation_threshold(
            clients,
            predictor=lambda _client, split: _predict_tcn_split(
                global_state,
                model_config,
                split,
            )[0],
        )
        train_metrics = round_evaluation["train_metrics"]
        validation_metrics = round_evaluation["validation_metrics"]
        validation_metrics_default_threshold = round_evaluation["validation_metrics_default_threshold"]
        test_metrics = round_evaluation["test_metrics"]
        test_metrics_default_threshold = round_evaluation["test_metrics_default_threshold"]
        communication_cost_bytes = int(parameter_bytes * 2 * (len(clients) + membership.n_subclusters))

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
        raise ValueError("AB_C1_FEDAVG_TCN: unable to determine best validation round.")
    if best_subcluster_sample_counts is None or best_subcluster_client_counts is None:
        raise ValueError("AB_C1_FEDAVG_TCN: missing best-round subcluster statistics.")

    membership_contents_after = membership.membership_file.read_text(encoding="utf-8")
    if membership_contents_before != membership_contents_after:
        raise ValueError("AB_C1_FEDAVG_TCN: frozen membership file changed during ablation execution.")

    total_communication_cost_bytes = int(sum(row["communication_cost_bytes"] for row in round_rows))
    prediction_outputs = write_prediction_outputs(
        output_root=output_root,
        experiment_id="AB_C1_FEDAVG_TCN",
        validation_labels=best_validation_labels,
        validation_probabilities=best_validation_probabilities,
        test_labels=best_test_labels,
        test_probabilities=best_test_probabilities,
        selected_threshold=best_selected_threshold,
        seed=configured_seed,
    )
    summary = {
        "experiment_id": "AB_C1_FEDAVG_TCN",
        "comparison_id": str(comparison["comparison_id"]),
        "cluster_id": cluster_id,
        "dataset": str(cluster_section["dataset_name"]),
        "cluster_config_path": str(cluster_config_path),
        "hierarchy": "hierarchical_fixed",
        "subcluster_layer_used": True,
        "membership_file_used": str(membership.membership_file),
        "membership_hash": membership.membership_hash,
        "membership_file_changed": False,
        "model_family": "tcn",
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
        "rounds": configured_rounds,
        "local_epochs": configured_local_epochs,
        "batch_size": configured_batch_size,
        "learning_rate": learning_rate,
        "optimizer_style": "sgd",
        "seed": configured_seed,
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
        "best_round_subcluster_train_sample_counts": best_subcluster_sample_counts,
        "best_round_subcluster_client_counts": best_subcluster_client_counts,
        "prediction_outputs": prediction_outputs,
        "round_metrics_path": str(round_metrics_path),
        "metrics_csv_path": str(metrics_csv_path),
    }

    summary_row = {
        "experiment_id": "AB_C1_FEDAVG_TCN",
        "cluster_id": cluster_id,
        "dataset": str(cluster_section["dataset_name"]),
        "hierarchy": "hierarchical_fixed",
        "model_family": "tcn",
        "fl_method": "FedAvg",
        "aggregation": "weighted_arithmetic_mean",
        "clustering_method": "agglomerative",
        "membership_hash": membership.membership_hash,
        "input_adapter": data_summary["input_adapter"],
        "num_leaf_clients": len(clients),
        "n_subclusters": membership.n_subclusters,
        "rounds": configured_rounds,
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

    return AblationRunResult(
        experiment_id="AB_C1_FEDAVG_TCN",
        cluster_id=cluster_id,
        dataset=str(cluster_section["dataset_name"]),
        output_dir=run_dir,
        summary_path=summary_path,
        round_metrics_path=round_metrics_path,
        metrics_csv_path=metrics_csv_path,
        summary=summary,
    )


def run_cluster2_fedavg_mlp_ablation(
    ablation_config_path: str | Path = "configs/ablation_cluster2_fedprox.yaml",
    *,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    smoke_test: bool = False,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
    output_root: str | Path = "outputs",
) -> AblationRunResult:
    resolved_config_path, config, comparison, control = _load_custom_control_entry(
        ablation_config_path,
        expected_experiment_id="AB_C2_FEDAVG_MLP",
        expected_run_source="custom_cluster2_fedavg_mlp",
        expected_cluster_id=2,
        expected_model_family="compact_mlp",
    )
    configured_rounds, configured_local_epochs, configured_batch_size, configured_seed = _configured_defaults(
        config,
        smoke_test=smoke_test,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    learning_rate = float(control["learning_rate"])
    cluster_config_path = _resolve_path(resolved_config_path, str(control["cluster_config"]))
    membership_path = _resolve_path(resolved_config_path, str(control["membership_file"]))

    cluster_yaml = _load_yaml(cluster_config_path)
    cluster_section = cluster_yaml.get("cluster")
    if not isinstance(cluster_section, Mapping):
        raise ValueError(f"{cluster_config_path}: cluster metadata is missing.")
    cluster_id = int(cluster_section["id"])
    if cluster_id != 2:
        raise ValueError("AB_C2_FEDAVG_MLP only supports Cluster 2.")

    clients, _, data_summary = build_flat_federated_clients(
        cluster_config_path,
        max_train_examples_per_client=max_train_examples_per_client,
        max_eval_examples_per_client=max_eval_examples_per_client,
    )
    if data_summary["input_adapter"] != "feature_vector_as_sequence":
        raise ValueError("AB_C2_FEDAVG_MLP requires feature-vector sequence inputs before MLP flattening.")
    if int(data_summary["input_channels"]) != 1:
        raise ValueError(
            f"AB_C2_FEDAVG_MLP requires input_channels=1 before flattening. "
            f"Observed {data_summary['input_channels']}."
        )

    membership = load_frozen_membership(
        membership_path,
        expected_cluster_id=2,
        expected_n_subclusters=int(control["n_subclusters"]),
        expected_client_ids=[client.client_id for client in clients],
    )
    clients_by_subcluster = group_clients_by_subcluster(clients, membership)

    model_config = MLPConfig(input_dim=int(data_summary["input_length"]))
    global_model = CompactMLPClassifier(model_config, seed=configured_seed)
    global_state = global_model.state_dict()
    parameter_bytes = global_model.parameter_bytes()

    output_root = Path(output_root)
    run_dir = output_root / "runs" / "AB_C2_FEDAVG_MLP"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = output_root / "metrics" / "AB_C2_FEDAVG_MLP_metrics.csv"
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
            local_updates: list[WeightedState] = []
            local_losses: list[float] = []
            subcluster_clients = clients_by_subcluster[subcluster.subcluster_id]

            for client_index, client in enumerate(subcluster_clients):
                result = _train_mlp_fedavg_client(
                    client,
                    global_state,
                    model_config,
                    local_epochs=configured_local_epochs,
                    batch_size=configured_batch_size,
                    learning_rate=learning_rate,
                    seed=configured_seed + round_index * 1000 + subcluster_index * 100 + client_index,
                    positive_class_weight=positive_class_weight,
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
        round_evaluation = evaluate_round_with_validation_threshold(
            clients,
            predictor=lambda _client, split: _predict_mlp_split(
                global_state,
                model_config,
                split,
            )[0],
        )
        train_metrics = round_evaluation["train_metrics"]
        validation_metrics = round_evaluation["validation_metrics"]
        validation_metrics_default_threshold = round_evaluation["validation_metrics_default_threshold"]
        test_metrics = round_evaluation["test_metrics"]
        test_metrics_default_threshold = round_evaluation["test_metrics_default_threshold"]
        communication_cost_bytes = int(parameter_bytes * 2 * (len(clients) + membership.n_subclusters))

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
        raise ValueError("AB_C2_FEDAVG_MLP: unable to determine best validation round.")
    if best_subcluster_sample_counts is None or best_subcluster_client_counts is None:
        raise ValueError("AB_C2_FEDAVG_MLP: missing best-round subcluster statistics.")

    membership_contents_after = membership.membership_file.read_text(encoding="utf-8")
    if membership_contents_before != membership_contents_after:
        raise ValueError("AB_C2_FEDAVG_MLP: frozen membership file changed during ablation execution.")

    total_communication_cost_bytes = int(sum(row["communication_cost_bytes"] for row in round_rows))
    prediction_outputs = write_prediction_outputs(
        output_root=output_root,
        experiment_id="AB_C2_FEDAVG_MLP",
        validation_labels=best_validation_labels,
        validation_probabilities=best_validation_probabilities,
        test_labels=best_test_labels,
        test_probabilities=best_test_probabilities,
        selected_threshold=best_selected_threshold,
        seed=configured_seed,
    )
    summary = {
        "experiment_id": "AB_C2_FEDAVG_MLP",
        "comparison_id": str(comparison["comparison_id"]),
        "cluster_id": cluster_id,
        "dataset": str(cluster_section["dataset_name"]),
        "cluster_config_path": str(cluster_config_path),
        "hierarchy": "hierarchical_fixed",
        "subcluster_layer_used": True,
        "membership_file_used": str(membership.membership_file),
        "membership_hash": membership.membership_hash,
        "membership_file_changed": False,
        "model_family": "compact_mlp",
        "fl_method": "FedAvg",
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
        "optimizer": "Adam",
        "seed": configured_seed,
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
        "best_round_subcluster_train_sample_counts": best_subcluster_sample_counts,
        "best_round_subcluster_client_counts": best_subcluster_client_counts,
        "prediction_outputs": prediction_outputs,
        "round_metrics_path": str(round_metrics_path),
        "metrics_csv_path": str(metrics_csv_path),
    }

    summary_row = {
        "experiment_id": "AB_C2_FEDAVG_MLP",
        "cluster_id": cluster_id,
        "dataset": str(cluster_section["dataset_name"]),
        "hierarchy": "hierarchical_fixed",
        "model_family": "compact_mlp",
        "fl_method": "FedAvg",
        "aggregation": "weighted_arithmetic_mean",
        "clustering_method": "agglomerative",
        "membership_hash": membership.membership_hash,
        "input_adapter": data_summary["input_adapter"],
        "model_input_layout": "batch_x_features",
        "num_leaf_clients": len(clients),
        "n_subclusters": membership.n_subclusters,
        "rounds": configured_rounds,
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

    return AblationRunResult(
        experiment_id="AB_C2_FEDAVG_MLP",
        cluster_id=cluster_id,
        dataset=str(cluster_section["dataset_name"]),
        output_dir=run_dir,
        summary_path=summary_path,
        round_metrics_path=round_metrics_path,
        metrics_csv_path=metrics_csv_path,
        summary=summary,
    )


def run_cluster3_fedavg_cnn1d_ablation(
    ablation_config_path: str | Path = "configs/ablation_cluster3_scaffold.yaml",
    *,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    smoke_test: bool = False,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
    output_root: str | Path = "outputs",
) -> AblationRunResult:
    resolved_config_path, config, comparison, control = _load_custom_control_entry(
        ablation_config_path,
        expected_experiment_id="AB_C3_FEDAVG_CNN1D",
        expected_run_source="custom_cluster3_fedavg_cnn1d",
        expected_cluster_id=3,
        expected_model_family="cnn1d",
    )
    configured_rounds, configured_local_epochs, configured_batch_size, configured_seed = _configured_defaults(
        config,
        smoke_test=smoke_test,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    learning_rate = float(control["learning_rate"])
    cluster_config_path = _resolve_path(resolved_config_path, str(control["cluster_config"]))
    membership_path = _resolve_path(resolved_config_path, str(control["membership_file"]))

    result = run_hierarchical_baseline_experiment(
        experiment_id="AB_C3_FEDAVG_CNN1D",
        cluster_config_path=cluster_config_path,
        membership_file=membership_path,
        expected_n_subclusters=int(control["n_subclusters"]),
        output_root=output_root,
        rounds=configured_rounds,
        local_epochs=configured_local_epochs,
        batch_size=configured_batch_size,
        learning_rate=learning_rate,
        seed=configured_seed,
        max_train_examples_per_client=max_train_examples_per_client,
        max_eval_examples_per_client=max_eval_examples_per_client,
    )

    summary = dict(result.summary)
    summary["comparison_id"] = str(comparison["comparison_id"])
    summary["ablation_config_path"] = str(resolved_config_path)
    result.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return AblationRunResult(
        experiment_id="AB_C3_FEDAVG_CNN1D",
        cluster_id=result.cluster_id,
        dataset=result.dataset,
        output_dir=result.output_dir,
        summary_path=result.summary_path,
        round_metrics_path=result.round_metrics_path,
        metrics_csv_path=result.metrics_csv_path,
        summary=summary,
    )
