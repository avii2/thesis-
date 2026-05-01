from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.fl.aggregators import AggregationError, CrossClusterAggregationError, WeightedState
from src.fl.client import ClientSplit, FlatClientDataset, LocalTrainingResult
from src.models.tcn_gn import TCNGNClassifier, TCNGNConfig


@dataclass(frozen=True)
class FedYogiConfig:
    server_lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 1e-3

    def validate(self) -> None:
        if self.server_lr <= 0.0:
            raise ValueError("FedYogiConfig.server_lr must be positive.")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("FedYogiConfig.beta1 must be in [0, 1).")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("FedYogiConfig.beta2 must be in [0, 1).")
        if self.tau <= 0.0:
            raise ValueError("FedYogiConfig.tau must be positive.")


@dataclass(frozen=True)
class FedYogiServerState:
    first_moment: Mapping[str, np.ndarray]
    second_moment: Mapping[str, np.ndarray]
    step: int = 0


def initialize_fedyogi_state(reference_state: Mapping[str, np.ndarray]) -> FedYogiServerState:
    return FedYogiServerState(
        first_moment={
            key: np.zeros_like(np.asarray(value, dtype=np.float32), dtype=np.float32)
            for key, value in reference_state.items()
        },
        second_moment={
            key: np.zeros_like(np.asarray(value, dtype=np.float32), dtype=np.float32)
            for key, value in reference_state.items()
        },
        step=0,
    )


def _validate_matching_state_keys(
    reference_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    *,
    context: str,
) -> None:
    reference_keys = set(reference_state.keys())
    candidate_keys = set(candidate_state.keys())
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        extra = sorted(candidate_keys - reference_keys)
        raise AggregationError(f"{context}: incompatible parameter keys. Missing={missing or '[]'} Extra={extra or '[]'}.")
    for key in reference_state:
        reference_shape = np.asarray(reference_state[key]).shape
        candidate_shape = np.asarray(candidate_state[key]).shape
        if reference_shape != candidate_shape:
            raise AggregationError(
                f"{context}: parameter {key!r} shape mismatch. "
                f"Observed {candidate_shape}, expected {reference_shape}."
            )


def state_delta(
    reference_state: Mapping[str, np.ndarray],
    updated_state: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    _validate_matching_state_keys(reference_state, updated_state, context="fedyogi_state_delta")
    return {
        key: (
            np.asarray(updated_state[key], dtype=np.float32)
            - np.asarray(reference_state[key], dtype=np.float32)
        ).astype(np.float32, copy=False)
        for key in reference_state
    }


def weighted_average_delta(
    updates: Sequence[WeightedState],
    *,
    expected_cluster_id: int,
    aggregation_scope: str,
) -> dict[str, np.ndarray]:
    if not updates:
        raise AggregationError(f"{aggregation_scope}: at least one weighted delta is required.")
    observed_cluster_ids = {update.cluster_id for update in updates}
    if observed_cluster_ids != {expected_cluster_id}:
        raise CrossClusterAggregationError(
            f"{aggregation_scope}: cross-cluster averaging is forbidden. "
            f"Expected cluster_id={expected_cluster_id}; observed={sorted(observed_cluster_ids)}."
        )

    reference_state = updates[0].state
    reference_keys = set(reference_state.keys())
    for update in updates:
        if not isinstance(update.num_samples, int) or update.num_samples <= 0:
            raise AggregationError(
                f"{aggregation_scope}: contributor {update.contributor_id!r} must have positive integer num_samples."
            )
        if set(update.state.keys()) != reference_keys:
            missing = sorted(reference_keys - set(update.state.keys()))
            extra = sorted(set(update.state.keys()) - reference_keys)
            raise AggregationError(
                f"{aggregation_scope}: contributor {update.contributor_id!r} has incompatible delta keys. "
                f"Missing={missing or '[]'} Extra={extra or '[]'}."
            )

    total_samples = sum(update.num_samples for update in updates)
    averaged: dict[str, np.ndarray] = {}
    for key in reference_state:
        weighted_sum = None
        for update in updates:
            contribution = np.asarray(update.state[key], dtype=np.float32) * (update.num_samples / total_samples)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        averaged[key] = np.asarray(weighted_sum, dtype=np.float32)
    return averaged


def apply_fedyogi_delta(
    current_state: Mapping[str, np.ndarray],
    averaged_delta: Mapping[str, np.ndarray],
    optimizer_state: FedYogiServerState,
    config: FedYogiConfig,
) -> tuple[dict[str, np.ndarray], FedYogiServerState]:
    config.validate()
    _validate_matching_state_keys(current_state, averaged_delta, context="fedyogi_apply_delta")
    _validate_matching_state_keys(current_state, optimizer_state.first_moment, context="fedyogi_first_moment")
    _validate_matching_state_keys(current_state, optimizer_state.second_moment, context="fedyogi_second_moment")

    next_state: dict[str, np.ndarray] = {}
    next_first: dict[str, np.ndarray] = {}
    next_second: dict[str, np.ndarray] = {}
    for key, value in current_state.items():
        delta = np.asarray(averaged_delta[key], dtype=np.float32)
        first = (
            config.beta1 * np.asarray(optimizer_state.first_moment[key], dtype=np.float32)
            + (1.0 - config.beta1) * delta
        ).astype(np.float32, copy=False)
        previous_second = np.asarray(optimizer_state.second_moment[key], dtype=np.float32)
        delta_squared = (delta * delta).astype(np.float32, copy=False)
        second = (
            previous_second
            - (1.0 - config.beta2) * np.sign(previous_second - delta_squared) * delta_squared
        ).astype(np.float32, copy=False)
        second = np.maximum(second, 0.0).astype(np.float32, copy=False)
        update = config.server_lr * first / (np.sqrt(second) + config.tau)
        next_state[key] = (np.asarray(value, dtype=np.float32) + update).astype(np.float32, copy=False)
        next_first[key] = first
        next_second[key] = second

    return next_state, FedYogiServerState(
        first_moment=next_first,
        second_moment=next_second,
        step=optimizer_state.step + 1,
    )


def train_tcn_gn_client(
    client: FlatClientDataset,
    parent_state: Mapping[str, np.ndarray],
    model_config: TCNGNConfig,
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    positive_class_weight: float = 1.0,
) -> LocalTrainingResult:
    if client.num_train_samples <= 0:
        raise ValueError(f"{client.client_id}: train split must contain at least one sample.")
    if local_epochs <= 0:
        raise ValueError("FedYogi local_epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("FedYogi batch_size must be positive.")

    model = TCNGNClassifier.from_state(model_config, parent_state, seed=seed)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for _ in range(local_epochs):
        losses.append(
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
        train_loss=float(np.mean(losses)),
        updated_state=model.state_dict(),
    )


def predict_split_tcn_gn(
    state: Mapping[str, np.ndarray],
    model_config: TCNGNConfig,
    split: ClientSplit,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    if split.num_samples == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int8)

    model = TCNGNClassifier.from_state(model_config, state)
    probabilities = model.predict_proba(split.inputs)
    predictions = (probabilities >= threshold).astype(np.int8, copy=False)
    return probabilities, predictions


def parameter_bytes(state: Mapping[str, np.ndarray]) -> int:
    return int(sum(np.asarray(value, dtype=np.float32).nbytes for value in state.values()))
