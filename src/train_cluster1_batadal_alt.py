from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.fl.aggregators import WeightedState
from src.fl.fedyogi import (
    FedYogiConfig,
    apply_fedyogi_delta,
    initialize_fedyogi_state,
    parameter_bytes,
    predict_split_tcn_gn,
    state_delta,
    train_tcn_gn_client,
    weighted_average_delta,
)
from src.fl.maincluster import (
    _load_yaml,
    _split_metrics,
    build_flat_federated_clients,
    compute_cluster_positive_class_weight,
    validation_threshold_sweep,
)
from src.fl.subcluster import group_clients_by_subcluster, load_frozen_membership
from src.models.tcn_gn import TCNGNClassifier, TCNGNConfig


EXPERIMENT_ID = "P_C1_BATADAL_TCN_FEDYOGI"
DEFAULT_CONFIG_PATH = Path("configs/proposed_cluster1_batadal_tcn_fedyogi.yaml")
THRESHOLD_SELECTION_MODES = (
    "max_validation_f1",
    "max_validation_f1_with_validation_fpr_le_0.10",
    "max_validation_f1_with_validation_fpr_le_0.05",
)
THRESHOLD_MODE_FPR_LIMITS: Mapping[str, float | None] = {
    "max_validation_f1": None,
    "max_validation_f1_with_validation_fpr_le_0.10": 0.10,
    "max_validation_f1_with_validation_fpr_le_0.05": 0.05,
}


@dataclass(frozen=True)
class Cluster1BatadalAltResult:
    experiment_id: str
    output_root: Path
    metrics_csv_path: Path
    report_path: Path
    rows: tuple[Mapping[str, Any], ...]


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


def _optional_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping when provided.")
    return value


def _require_alt_entry(config_path: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    clusters = config.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError(f"{config_path}: alternative config must contain clusters.")
    for entry in clusters:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("experiment_id")) != EXPERIMENT_ID:
            continue
        expected = {
            "model_family": "tcn_gn",
            "fl_method": "FedYogi",
            "aggregation": "hierarchical_fedyogi_delta",
            "hierarchy": "hierarchical_fixed",
        }
        for key, expected_value in expected.items():
            observed = str(entry.get(key))
            if observed != expected_value:
                raise ValueError(f"{EXPERIMENT_ID}: expected {key}={expected_value}, observed {observed}.")
        if int(entry.get("n_subclusters", 0)) != 2:
            raise ValueError(f"{EXPERIMENT_ID}: n_subclusters must remain 2.")
        return entry
    raise ValueError(f"{config_path}: could not find {EXPERIMENT_ID}.")


def _resolve_training_defaults(
    config: Mapping[str, Any],
    *,
    rounds: int | None,
    local_epochs: int | None,
    batch_size: int | None,
    seed: int | None,
) -> dict[str, int]:
    defaults = config.get("training_defaults")
    if not isinstance(defaults, Mapping):
        raise ValueError("Alternative config must define training_defaults.")
    configured_seed = int(seed if seed is not None else defaults.get("seed", 42))
    return {
        "rounds": int(rounds if rounds is not None else defaults["rounds"]),
        "local_epochs": int(local_epochs if local_epochs is not None else defaults["local_epochs"]),
        "batch_size": int(batch_size if batch_size is not None else defaults["batch_size"]),
        "seed": configured_seed,
    }


def _resolve_model_config(entry: Mapping[str, Any], *, input_channels: int, input_length: int) -> TCNGNConfig:
    model_hyperparameters = _optional_mapping(entry, "model_hyperparameters")
    dilations_raw = model_hyperparameters.get("dilations", [1, 2, 4, 8, 16])
    dilations = tuple(int(value) for value in dilations_raw)
    return TCNGNConfig(
        input_channels=input_channels,
        input_length=input_length,
        channels=int(model_hyperparameters.get("channels", 64)),
        dilations=dilations,
        kernel_size=int(model_hyperparameters.get("kernel_size", 3)),
        groups=int(model_hyperparameters.get("groups", 8)),
        hidden_dim=int(model_hyperparameters.get("hidden_dim", 32)),
        dropout=float(model_hyperparameters.get("dropout", 0.15)),
    )


def _resolve_fedyogi_config(entry: Mapping[str, Any]) -> FedYogiConfig:
    fedyogi = _optional_mapping(entry, "fedyogi")
    config = FedYogiConfig(
        server_lr=float(fedyogi.get("server_lr", 0.01)),
        beta1=float(fedyogi.get("beta1", 0.9)),
        beta2=float(fedyogi.get("beta2", 0.99)),
        tau=float(fedyogi.get("tau", 1e-3)),
    )
    config.validate()
    return config


def _resolve_positive_class_weight_scales(
    entry: Mapping[str, Any],
    overrides: Sequence[float] | None,
) -> tuple[float, ...]:
    if overrides is not None:
        values = tuple(float(value) for value in overrides)
    else:
        training_hyperparameters = _optional_mapping(entry, "training_hyperparameters")
        raw_values = training_hyperparameters.get("positive_class_weight_scales", [0.25, 0.50, 1.00])
        values = tuple(float(value) for value in raw_values)
    if not values:
        raise ValueError("positive_class_weight_scales must not be empty.")
    if any(value < 0.0 for value in values):
        raise ValueError("positive_class_weight_scales must be non-negative.")
    return values


def _resolve_threshold_modes(entry: Mapping[str, Any]) -> tuple[str, ...]:
    raw_modes = entry.get("threshold_selection_modes", THRESHOLD_SELECTION_MODES)
    modes = tuple(str(value) for value in raw_modes)
    unsupported = [mode for mode in modes if mode not in THRESHOLD_MODE_FPR_LIMITS]
    if unsupported:
        raise ValueError(f"Unsupported threshold selection modes: {unsupported}")
    return modes


def _numeric_metric(value: Any, *, missing_value: float = float("-inf")) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return missing_value
    return missing_value


def select_validation_threshold_row(
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    *,
    mode: str,
) -> Mapping[str, Any]:
    if mode not in THRESHOLD_MODE_FPR_LIMITS:
        raise ValueError(f"Unsupported threshold selection mode: {mode}")
    sweep = validation_threshold_sweep(validation_labels, validation_probabilities)
    if not sweep:
        raise ValueError("No validation thresholds are available; test data was not used.")
    fpr_limit = THRESHOLD_MODE_FPR_LIMITS[mode]
    eligible = [
        row
        for row in sweep
        if fpr_limit is None or float(row["validation_fpr"]) <= fpr_limit + 1e-12
    ]
    if not eligible and fpr_limit is not None:
        fallback_threshold = 1.0
        fallback_metrics = _split_metrics(
            validation_labels,
            validation_probabilities,
            threshold=fallback_threshold,
        )
        if float(fallback_metrics["fpr"]) <= fpr_limit + 1e-12:
            eligible = [
                {
                    "threshold": fallback_threshold,
                    "validation_precision": fallback_metrics["precision"],
                    "validation_recall": fallback_metrics["recall"],
                    "validation_f1": fallback_metrics["f1"],
                    "validation_pr_auc": fallback_metrics["pr_auc"],
                    "validation_fpr": fallback_metrics["fpr"],
                    "selected": True,
                }
            ]
    if not eligible:
        raise ValueError(f"No validation threshold satisfies {mode}; test data was not used for fallback tuning.")
    return max(
        eligible,
        key=lambda row: (
            _numeric_metric(row["validation_f1"]),
            _numeric_metric(row["validation_pr_auc"]),
            -_numeric_metric(row["validation_fpr"], missing_value=float("inf")),
            _numeric_metric(row["threshold"]),
        ),
    )


def _collect_split_outputs(
    clients: Sequence[Any],
    *,
    state: Mapping[str, np.ndarray],
    model_config: TCNGNConfig,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for client in clients:
        split = getattr(client, split_name)
        split_probabilities, _ = predict_split_tcn_gn(state, model_config, split)
        if split_probabilities.size == 0:
            continue
        probabilities.append(split_probabilities.astype(np.float32, copy=False))
        labels.append(split.labels.astype(np.int8, copy=False))
    if not probabilities:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32)
    return np.concatenate(labels), np.concatenate(probabilities)


def _confusion_parts(metrics: Mapping[str, Any]) -> dict[str, int]:
    matrix = metrics["confusion_matrix"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _best_key(validation_metrics: Mapping[str, Any], threshold: float) -> tuple[float, float, float]:
    return (
        _numeric_metric(validation_metrics.get("f1")),
        -_numeric_metric(validation_metrics.get("fpr"), missing_value=float("inf")),
        float(threshold),
    )


def _state_copy(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=np.float32).copy() for key, value in state.items()}


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_run_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    metrics_csv_path: Path,
    membership_hash: str,
    data_summary: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_by_validation = max(
        rows,
        key=lambda row: (
            _numeric_metric(row["validation_f1"]),
            -_numeric_metric(row["validation_fpr"], missing_value=float("inf")),
            _numeric_metric(row["selected_threshold"]),
        ),
    )
    lines = [
        "# P_C1_BATADAL_TCN_FEDYOGI Report",
        "",
        "This is a Cluster 1 BATADAL alternative ablation. The existing P_C1_BATADAL experiment is unchanged.",
        "",
        "## Protocol",
        "",
        "- Dataset: BATADAL",
        "- Model: TCN-GN",
        "- FL method: FedYogi",
        "- Aggregation: hierarchical_fedyogi_delta",
        "- Leaf clients: 12",
        "- Fixed sub-clusters: 2",
        f"- Frozen membership hash: `{membership_hash}`",
        "- Thresholds are selected using validation predictions only.",
        "- Held-out test predictions are evaluated only after validation threshold selection.",
        "- No Cluster 2, Cluster 3, or ledger logic is modified by this ablation.",
        "",
        "## Data Guardrails",
        "",
        f"- Input adapter: `{data_summary.get('input_adapter')}`",
        f"- Input channels: {data_summary.get('input_channels')}",
        f"- Input length: {data_summary.get('input_length')}",
        f"- Test used for training: {data_summary.get('test_dataset_used_for_training', False)}",
        f"- Test used for validation: {data_summary.get('test_dataset_used_for_validation', False)}",
        f"- Test used for threshold tuning: {data_summary.get('test_dataset_used_for_threshold_tuning', False)}",
        f"- Test used for preprocessing fit: {data_summary.get('test_dataset_used_for_preprocessing_fit', False)}",
        f"- Test used for clustering: {data_summary.get('test_dataset_used_for_clustering', False)}",
        f"- Test used for descriptor computation: {data_summary.get('test_dataset_used_for_descriptor_computation', False)}",
        "- Class-weight scales are reported as an ablation; any preferred scale must be chosen from validation metrics only.",
        "",
        "## Best Validation Row",
        "",
        f"- positive_class_weight_scale: {best_by_validation['positive_class_weight_scale']}",
        f"- threshold_selection_mode: `{best_by_validation['threshold_selection_mode']}`",
        f"- best_validation_round: {best_by_validation['best_validation_round']}",
        f"- selected_threshold: {best_by_validation['selected_threshold']}",
        f"- validation_f1: {best_by_validation['validation_f1']}",
        f"- validation_fpr: {best_by_validation['validation_fpr']}",
        f"- test_f1: {best_by_validation['test_f1']}",
        f"- test_fpr: {best_by_validation['test_fpr']}",
        "",
        "## Metrics CSV",
        "",
        f"`{metrics_csv_path}`",
        "",
        "## Results",
        "",
        "| scale | threshold mode | best round | threshold | test accuracy | test precision | test recall | test F1 | test AUROC | test PR-AUC | test FPR | confusion matrix | communication bytes | training seconds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scale} | `{mode}` | {round_} | {threshold:.6f} | {test_accuracy} | {test_precision} | {test_recall} | {test_f1} | {test_auroc} | {test_pr_auc} | {test_fpr} | `{confusion_matrix}` | {communication_cost} | {training_time:.3f} |".format(
                scale=float(row["positive_class_weight_scale"]),
                mode=row["threshold_selection_mode"],
                round_=int(row["best_validation_round"]),
                threshold=float(row["selected_threshold"]),
                test_accuracy=_format_report_metric(row["test_accuracy"]),
                test_precision=_format_report_metric(row["test_precision"]),
                test_recall=_format_report_metric(row["test_recall"]),
                test_f1=_format_report_metric(row["test_f1"]),
                test_auroc=_format_report_metric(row["test_auroc"]),
                test_pr_auc=_format_report_metric(row["test_pr_auc"]),
                test_fpr=_format_report_metric(row["test_fpr"]),
                confusion_matrix=str(row["test_confusion_matrix"]),
                communication_cost=int(row["total_communication_cost_bytes"]),
                training_time=float(row["training_time_seconds"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_metric(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _format_report_metric(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _evaluate_selected_test(
    *,
    clients: Sequence[Any],
    state: Mapping[str, np.ndarray],
    model_config: TCNGNConfig,
    threshold: float,
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray]:
    test_labels, test_probabilities = _collect_split_outputs(
        clients,
        state=state,
        model_config=model_config,
        split_name="test",
    )
    return (
        _split_metrics(test_labels, test_probabilities, threshold=threshold),
        test_labels,
        test_probabilities,
    )


def run_cluster1_batadal_alt(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    client_lr: float | None = None,
    seed: int | None = None,
    output_root: str | Path | None = None,
    positive_class_weight_scales: Sequence[float] | None = None,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
) -> Cluster1BatadalAltResult:
    resolved_config_path = Path(config_path).resolve()
    config = _load_yaml(resolved_config_path)
    entry = _require_alt_entry(resolved_config_path, config)
    defaults = _resolve_training_defaults(
        config,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    training_hyperparameters = _optional_mapping(entry, "training_hyperparameters")
    configured_client_lr = float(client_lr if client_lr is not None else training_hyperparameters.get("client_lr", 0.001))
    if configured_client_lr <= 0.0:
        raise ValueError("client_lr must be positive.")

    cluster_config_path = _resolve_path(resolved_config_path, str(entry["cluster_config"]))
    membership_path = _resolve_path(resolved_config_path, str(entry["membership_file"]))
    cluster_yaml = _load_yaml(cluster_config_path)
    cluster_section = cluster_yaml.get("cluster")
    if not isinstance(cluster_section, Mapping) or int(cluster_section.get("id", 0)) != 1:
        raise ValueError(f"{EXPERIMENT_ID}: cluster_config must resolve to Cluster 1.")

    clients, _, data_summary = build_flat_federated_clients(
        cluster_config_path,
        max_train_examples_per_client=max_train_examples_per_client,
        max_eval_examples_per_client=max_eval_examples_per_client,
    )
    if len(clients) != 12:
        raise ValueError(f"{EXPERIMENT_ID}: expected 12 Cluster 1 leaf clients, observed {len(clients)}.")
    if data_summary.get("input_adapter") != "sliding_window_feature_channels":
        raise ValueError(f"{EXPERIMENT_ID}: BATADAL alternative requires sliding-window inputs.")
    if any(client.cluster_id != 1 for client in clients):
        raise ValueError(f"{EXPERIMENT_ID}: all clients must remain inside Cluster 1.")

    membership = load_frozen_membership(
        membership_path,
        expected_cluster_id=1,
        expected_n_subclusters=int(entry["n_subclusters"]),
        expected_client_ids=[client.client_id for client in clients],
    )
    if membership.n_subclusters != 2:
        raise ValueError(f"{EXPERIMENT_ID}: expected 2 frozen sub-clusters.")
    clients_by_subcluster = group_clients_by_subcluster(clients, membership)

    model_config = _resolve_model_config(
        entry,
        input_channels=int(data_summary["input_channels"]),
        input_length=int(data_summary["input_length"]),
    )
    fedyogi_config = _resolve_fedyogi_config(entry)
    threshold_modes = _resolve_threshold_modes(entry)
    scales = _resolve_positive_class_weight_scales(entry, positive_class_weight_scales)
    computed_positive_class_weight = compute_cluster_positive_class_weight(clients)

    resolved_output_root = Path(output_root if output_root is not None else entry.get("output_root", "outputs_c1_batadal_alt"))
    metrics_csv_path = resolved_output_root / "metrics" / f"{EXPERIMENT_ID}_metrics.csv"
    report_path = resolved_output_root / "reports" / f"{EXPERIMENT_ID}_report.md"
    membership_contents_before = membership.membership_file.read_text(encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for scale in scales:
        positive_class_weight = 1.0 if scale == 0.0 else computed_positive_class_weight * scale
        initial_model = TCNGNClassifier(model_config, seed=defaults["seed"])
        main_state = initial_model.state_dict()
        main_optimizer = initialize_fedyogi_state(main_state)
        subcluster_optimizers = {
            subcluster.subcluster_id: initialize_fedyogi_state(main_state)
            for subcluster in membership.subclusters
        }
        communicated_parameter_bytes = parameter_bytes(main_state)
        communication_cost_per_round = int(communicated_parameter_bytes * 2 * (len(clients) + membership.n_subclusters))
        best_by_mode: dict[str, dict[str, Any]] = {}
        scale_round_rows: list[dict[str, Any]] = []
        started_at = time.perf_counter()

        for round_index in range(1, defaults["rounds"] + 1):
            subcluster_delta_updates: list[WeightedState] = []
            local_losses: list[float] = []
            for subcluster_index, subcluster in enumerate(membership.subclusters):
                parent_state = _state_copy(main_state)
                client_delta_updates: list[WeightedState] = []
                subcluster_losses: list[float] = []
                for client_index, client in enumerate(clients_by_subcluster[subcluster.subcluster_id]):
                    result = train_tcn_gn_client(
                        client,
                        parent_state,
                        model_config,
                        local_epochs=defaults["local_epochs"],
                        batch_size=defaults["batch_size"],
                        learning_rate=configured_client_lr,
                        seed=defaults["seed"] + round_index * 1000 + subcluster_index * 100 + client_index,
                        positive_class_weight=positive_class_weight,
                    )
                    client_delta_updates.append(
                        WeightedState(
                            cluster_id=1,
                            contributor_id=result.client_id,
                            num_samples=result.num_train_samples,
                            state=state_delta(parent_state, result.updated_state),
                        )
                    )
                    subcluster_losses.append(result.train_loss)
                averaged_client_delta = weighted_average_delta(
                    client_delta_updates,
                    expected_cluster_id=1,
                    aggregation_scope=f"{EXPERIMENT_ID}:{subcluster.subcluster_id}:leaf_to_subcluster",
                )
                subcluster_state, subcluster_optimizers[subcluster.subcluster_id] = apply_fedyogi_delta(
                    parent_state,
                    averaged_client_delta,
                    subcluster_optimizers[subcluster.subcluster_id],
                    fedyogi_config,
                )
                subcluster_delta_updates.append(
                    WeightedState(
                        cluster_id=1,
                        contributor_id=subcluster.subcluster_id,
                        num_samples=sum(client.train.num_samples for client in clients_by_subcluster[subcluster.subcluster_id]),
                        state=state_delta(main_state, subcluster_state),
                    )
                )
                local_losses.extend(subcluster_losses)

            averaged_subcluster_delta = weighted_average_delta(
                subcluster_delta_updates,
                expected_cluster_id=1,
                aggregation_scope=f"{EXPERIMENT_ID}:subcluster_to_main",
            )
            main_state, main_optimizer = apply_fedyogi_delta(
                main_state,
                averaged_subcluster_delta,
                main_optimizer,
                fedyogi_config,
            )

            validation_labels, validation_probabilities = _collect_split_outputs(
                clients,
                state=main_state,
                model_config=model_config,
                split_name="validation",
            )
            for mode in threshold_modes:
                threshold_row = select_validation_threshold_row(
                    validation_labels,
                    validation_probabilities,
                    mode=mode,
                )
                threshold = float(threshold_row["threshold"])
                validation_metrics = _split_metrics(
                    validation_labels,
                    validation_probabilities,
                    threshold=threshold,
                )
                current_key = _best_key(validation_metrics, threshold)
                best = best_by_mode.get(mode)
                if best is None or current_key > best["best_key"]:
                    best_by_mode[mode] = {
                        "best_key": current_key,
                        "round": round_index,
                        "state": _state_copy(main_state),
                        "threshold": threshold,
                        "validation_metrics": dict(validation_metrics),
                        "validation_labels": validation_labels.copy(),
                        "validation_probabilities": validation_probabilities.copy(),
                        "train_loss_local_mean": float(np.mean(local_losses)),
                    }
            scale_round_rows.append(
                {
                    "round": round_index,
                    "positive_class_weight_scale": scale,
                    "train_loss_local_mean": float(np.mean(local_losses)),
                    "communication_cost_bytes": communication_cost_per_round,
                }
            )

        elapsed_seconds = float(time.perf_counter() - started_at)
        total_communication_cost = int(communication_cost_per_round * defaults["rounds"])
        prediction_dir = resolved_output_root / "predictions" / EXPERIMENT_ID / f"scale_{scale:.2f}".replace(".", "")
        prediction_dir.mkdir(parents=True, exist_ok=True)

        for mode in threshold_modes:
            best = best_by_mode[mode]
            threshold = float(best["threshold"])
            test_metrics, test_labels, test_probabilities = _evaluate_selected_test(
                clients=clients,
                state=best["state"],
                model_config=model_config,
                threshold=threshold,
            )
            validation_metrics = best["validation_metrics"]
            validation_confusion = _confusion_parts(validation_metrics)
            test_confusion = _confusion_parts(test_metrics)
            suffix = mode.replace("max_validation_f1", "valf1").replace(".", "")
            np.savez_compressed(
                prediction_dir / f"validation_predictions_{suffix}.npz",
                labels=best["validation_labels"].astype(np.int8, copy=False),
                probabilities=best["validation_probabilities"].astype(np.float32, copy=False),
            )
            np.savez_compressed(
                prediction_dir / f"test_predictions_{suffix}.npz",
                labels=test_labels.astype(np.int8, copy=False),
                probabilities=test_probabilities.astype(np.float32, copy=False),
            )
            row = {
                "experiment_id": EXPERIMENT_ID,
                "cluster_id": 1,
                "dataset": "BATADAL",
                "hierarchy": "hierarchical_fixed",
                "num_leaf_clients": len(clients),
                "n_subclusters": membership.n_subclusters,
                "clustering_method": "existing_frozen_agglomerative_membership",
                "membership_file": str(membership.membership_file),
                "membership_hash": membership.membership_hash,
                "model_family": "tcn_gn",
                "fl_method": "FedYogi",
                "aggregation": "hierarchical_fedyogi_delta",
                "rounds": defaults["rounds"],
                "local_epochs": defaults["local_epochs"],
                "batch_size": defaults["batch_size"],
                "client_optimizer": "Adam",
                "client_lr": configured_client_lr,
                "server_lr": fedyogi_config.server_lr,
                "beta1": fedyogi_config.beta1,
                "beta2": fedyogi_config.beta2,
                "tau": fedyogi_config.tau,
                "seed": defaults["seed"],
                "positive_class_weight_scale": scale,
                "computed_positive_class_weight": computed_positive_class_weight,
                "positive_class_weight": positive_class_weight,
                "threshold_selection_mode": mode,
                "validation_fpr_limit": THRESHOLD_MODE_FPR_LIMITS[mode]
                if THRESHOLD_MODE_FPR_LIMITS[mode] is not None
                else "",
                "threshold_selected_on": "validation",
                "test_dataset_used_for_threshold_tuning": False,
                "selected_threshold": threshold,
                "best_validation_round": int(best["round"]),
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_f1": validation_metrics["f1"],
                "validation_auroc": validation_metrics["auroc"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "validation_fpr": validation_metrics["fpr"],
                "validation_support": validation_metrics["support"],
                "validation_tn": validation_confusion["tn"],
                "validation_fp": validation_confusion["fp"],
                "validation_fn": validation_confusion["fn"],
                "validation_tp": validation_confusion["tp"],
                "validation_confusion_matrix": _json_metric(validation_metrics["confusion_matrix"]),
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_auroc": test_metrics["auroc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_fpr": test_metrics["fpr"],
                "test_support": test_metrics["support"],
                "test_tn": test_confusion["tn"],
                "test_fp": test_confusion["fp"],
                "test_fn": test_confusion["fn"],
                "test_tp": test_confusion["tp"],
                "test_confusion_matrix": _json_metric(test_metrics["confusion_matrix"]),
                "parameter_bytes": communicated_parameter_bytes,
                "communication_cost_per_round_bytes": communication_cost_per_round,
                "total_communication_cost_bytes": total_communication_cost,
                "training_time_seconds": elapsed_seconds,
                "output_root": str(resolved_output_root),
            }
            rows.append(row)
        run_summary_path = (
            resolved_output_root
            / "runs"
            / EXPERIMENT_ID
            / f"scale_{scale:.2f}".replace(".", "")
            / "run_summary.json"
        )
        _write_run_json(
            run_summary_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "positive_class_weight_scale": scale,
                "rounds": defaults["rounds"],
                "local_epochs": defaults["local_epochs"],
                "batch_size": defaults["batch_size"],
                "client_lr": configured_client_lr,
                "fedyogi": {
                    "server_lr": fedyogi_config.server_lr,
                    "beta1": fedyogi_config.beta1,
                    "beta2": fedyogi_config.beta2,
                    "tau": fedyogi_config.tau,
                },
                "model_config": {
                    "input_channels": model_config.input_channels,
                    "input_length": model_config.input_length,
                    "channels": model_config.channels,
                    "dilations": list(model_config.dilations),
                    "kernel_size": model_config.kernel_size,
                    "groups": model_config.groups,
                    "hidden_dim": model_config.hidden_dim,
                    "dropout": model_config.dropout,
                },
                "data_summary": data_summary,
                "round_rows": scale_round_rows,
            },
        )

    membership_contents_after = membership.membership_file.read_text(encoding="utf-8")
    if membership_contents_before != membership_contents_after:
        raise ValueError(f"{EXPERIMENT_ID}: frozen membership file changed during execution.")

    _write_rows(metrics_csv_path, rows)
    _write_report(
        report_path,
        rows=rows,
        metrics_csv_path=metrics_csv_path,
        membership_hash=membership.membership_hash,
        data_summary=data_summary,
    )
    return Cluster1BatadalAltResult(
        experiment_id=EXPERIMENT_ID,
        output_root=resolved_output_root,
        metrics_csv_path=metrics_csv_path,
        report_path=report_path,
        rows=tuple(rows),
    )


def _parse_float_list(value: str | None) -> tuple[float, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Cluster 1 BATADAL TCN-GN FedYogi alternative ablation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--client-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--positive-class-weight-scales", default=None)
    parser.add_argument("--max-train-examples-per-client", type=int, default=None)
    parser.add_argument("--max-eval-examples-per-client", type=int, default=None)
    args = parser.parse_args(argv)

    result = run_cluster1_batadal_alt(
        args.config,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        client_lr=args.client_lr,
        seed=args.seed,
        output_root=args.output_root,
        positive_class_weight_scales=_parse_float_list(args.positive_class_weight_scales),
        max_train_examples_per_client=args.max_train_examples_per_client,
        max_eval_examples_per_client=args.max_eval_examples_per_client,
    )
    print(f"Wrote metrics: {result.metrics_csv_path}")
    print(f"Wrote report: {result.report_path}")


if __name__ == "__main__":
    main()
