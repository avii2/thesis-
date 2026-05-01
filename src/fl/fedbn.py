from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.fl.aggregators import (
    WeightedState,
    aggregate_leaf_updates_to_subcluster_non_bn,
    aggregate_subcluster_updates_to_maincluster_non_bn,
    is_batch_norm_key,
)
from src.fl.client import ClientSplit, FlatClientDataset, LocalTrainingResult
from src.fl.sampling import positive_window_oversample_indices
from src.models.cnn1d_bn import CNN1DBNClassifier, CNN1DBNConfig


@dataclass(frozen=True)
class FedBNStateSplit:
    bn_state: Mapping[str, np.ndarray]
    non_bn_state: Mapping[str, np.ndarray]


def split_state_by_batch_norm(state: Mapping[str, np.ndarray]) -> FedBNStateSplit:
    bn_state = {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in state.items()
        if is_batch_norm_key(key)
    }
    non_bn_state = {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in state.items()
        if not is_batch_norm_key(key)
    }
    return FedBNStateSplit(bn_state=bn_state, non_bn_state=non_bn_state)


def merge_global_non_bn_with_local_bn(
    shared_state: Mapping[str, np.ndarray],
    local_state: Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    local_lookup = dict(local_state or {})
    for key, value in shared_state.items():
        if is_batch_norm_key(key):
            source = local_lookup.get(key, value)
        else:
            source = value
        merged[key] = np.asarray(source, dtype=np.float32).copy()
    return merged


def replace_non_bn_state(
    reference_state: Mapping[str, np.ndarray],
    non_bn_updates: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for key, value in reference_state.items():
        source = value if is_batch_norm_key(key) else non_bn_updates.get(key, value)
        merged[key] = np.asarray(source, dtype=np.float32).copy()
    return merged


def non_bn_parameter_bytes(state: Mapping[str, np.ndarray]) -> int:
    return int(
        sum(
            np.asarray(value, dtype=np.float32).nbytes
            for key, value in state.items()
            if not is_batch_norm_key(key)
        )
    )


def _model_from_state(
    model_config: CNN1DBNConfig,
    state: Mapping[str, Any],
    *,
    seed: int = 42,
) -> CNN1DBNClassifier:
    if isinstance(model_config, CNN1DBNConfig):
        return CNN1DBNClassifier.from_state(model_config, state, seed=seed)
    raise TypeError(f"Unsupported FedBN model config type: {type(model_config).__name__}.")


def train_fedbn_client(
    client: FlatClientDataset,
    shared_state: Mapping[str, np.ndarray],
    local_state: Mapping[str, np.ndarray] | None,
    model_config: CNN1DBNConfig,
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    positive_class_weight: float = 1.0,
    training_resampling: str | None = None,
    target_positive_fraction: float = 0.5,
) -> LocalTrainingResult:
    if client.num_train_samples <= 0:
        raise ValueError(f"{client.client_id}: train split must contain at least one sample.")

    initial_state = merge_global_non_bn_with_local_bn(shared_state, local_state)
    model = _model_from_state(model_config, initial_state, seed=seed)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for _ in range(local_epochs):
        train_inputs = client.train.inputs
        train_labels = client.train.labels
        if training_resampling == "positive_window_oversampling":
            sample_indices = positive_window_oversample_indices(
                train_labels,
                rng=rng,
                target_positive_fraction=target_positive_fraction,
            )
            train_inputs = train_inputs[sample_indices]
            train_labels = train_labels[sample_indices]
        elif training_resampling not in {None, "none"}:
            raise ValueError(f"Unsupported FedBN training_resampling strategy: {training_resampling}")
        losses.append(
            model.train_epoch(
                train_inputs,
                train_labels,
                batch_size=batch_size,
                learning_rate=learning_rate,
                rng=rng,
                positive_class_weight=positive_class_weight,
            )
        )
    return LocalTrainingResult(
        client_id=client.client_id,
        num_train_samples=client.num_train_samples,
        train_loss=float(np.mean(losses)),
        updated_state=model.state_dict(),
    )


def predict_split_fedbn(
    state: Mapping[str, np.ndarray],
    model_config: CNN1DBNConfig,
    split: ClientSplit,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    if split.num_samples == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int8)

    model = _model_from_state(model_config, state)
    probabilities = model.predict_proba(split.inputs)
    predictions = (probabilities >= threshold).astype(np.int8, copy=False)
    return probabilities, predictions


def aggregate_fedbn_leaf_updates(
    updates: list[WeightedState],
    *,
    cluster_id: int,
    reference_state: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    aggregated_non_bn = aggregate_leaf_updates_to_subcluster_non_bn(
        updates,
        expected_cluster_id=cluster_id,
    )
    return replace_non_bn_state(reference_state, aggregated_non_bn)


def aggregate_fedbn_subcluster_updates(
    updates: list[WeightedState],
    *,
    cluster_id: int,
    reference_state: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    aggregated_non_bn = aggregate_subcluster_updates_to_maincluster_non_bn(
        updates,
        expected_cluster_id=cluster_id,
    )
    return replace_non_bn_state(reference_state, aggregated_non_bn)
