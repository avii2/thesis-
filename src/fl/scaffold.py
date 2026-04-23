from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.fl.aggregators import WeightedState
from src.fl.client import ClientSplit, FlatClientDataset
from src.models.cnn1d import CNN1DClassifier, CNN1DConfig, STATE_KEYS


@dataclass(frozen=True)
class ScaffoldTrainingResult:
    client_id: str
    num_train_samples: int
    train_loss: float
    updated_state: Mapping[str, np.ndarray]
    updated_control_variate: Mapping[str, np.ndarray]
    delta_control_variate: Mapping[str, np.ndarray]
    local_steps: int

    def to_weighted_state(self, *, cluster_id: int) -> WeightedState:
        return WeightedState(
            cluster_id=cluster_id,
            contributor_id=self.client_id,
            num_samples=self.num_train_samples,
            state=self.updated_state,
        )


def zero_control_variate_like(reference_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    _validate_state_keys(reference_state)
    return {
        key: np.zeros_like(np.asarray(reference_state[key], dtype=np.float32))
        for key in STATE_KEYS
    }


def control_variate_bytes(control_variate: Mapping[str, Any]) -> int:
    _validate_state_keys(control_variate)
    return int(sum(np.asarray(control_variate[key], dtype=np.float32).nbytes for key in STATE_KEYS))


def state_l2_norm(state: Mapping[str, Any]) -> float:
    _validate_state_keys(state)
    squared_norm = 0.0
    for key in STATE_KEYS:
        squared_norm += float(np.sum(np.square(np.asarray(state[key], dtype=np.float32))))
    return float(np.sqrt(squared_norm))


def sum_control_variate_states(states: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not states:
        raise ValueError("SCAFFOLD requires at least one control-variate state to sum.")

    reference = states[0]
    _validate_state_keys(reference)
    summed = {
        key: np.asarray(reference[key], dtype=np.float32).copy()
        for key in STATE_KEYS
    }
    for state in states[1:]:
        _validate_compatible_state(reference, state, state_name="control_variate_state")
        for key in STATE_KEYS:
            summed[key] = (summed[key] + np.asarray(state[key], dtype=np.float32)).astype(np.float32, copy=False)
    return summed


def update_server_control_variate(
    server_control_variate: Mapping[str, Any],
    delta_control_variates: Sequence[Mapping[str, Any]],
    *,
    total_leaf_clients: int,
) -> dict[str, np.ndarray]:
    if total_leaf_clients <= 0:
        raise ValueError(f"SCAFFOLD total_leaf_clients must be positive. Observed {total_leaf_clients}.")
    _validate_state_keys(server_control_variate)

    if not delta_control_variates:
        return {
            key: np.asarray(server_control_variate[key], dtype=np.float32).copy()
            for key in STATE_KEYS
        }

    delta_sum = sum_control_variate_states(delta_control_variates)
    updated = {}
    for key in STATE_KEYS:
        updated[key] = (
            np.asarray(server_control_variate[key], dtype=np.float32)
            + delta_sum[key] / float(total_leaf_clients)
        ).astype(np.float32, copy=False)
    return updated


def train_scaffold_client(
    client: FlatClientDataset,
    parent_state: Mapping[str, np.ndarray],
    server_control_variate: Mapping[str, np.ndarray],
    client_control_variate: Mapping[str, np.ndarray],
    model_config: CNN1DConfig,
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    positive_class_weight: float = 1.0,
) -> ScaffoldTrainingResult:
    if client.num_train_samples <= 0:
        raise ValueError(f"{client.client_id}: train split must contain at least one sample.")
    if learning_rate <= 0.0:
        raise ValueError(f"SCAFFOLD learning_rate must be positive. Observed {learning_rate}.")
    if local_epochs <= 0:
        raise ValueError(f"SCAFFOLD local_epochs must be positive. Observed {local_epochs}.")
    if batch_size <= 0:
        raise ValueError(f"SCAFFOLD batch_size must be positive. Observed {batch_size}.")

    _validate_state_keys(parent_state)
    _validate_compatible_state(parent_state, server_control_variate, state_name="server_control_variate")
    _validate_compatible_state(parent_state, client_control_variate, state_name="client_control_variate")

    model = CNN1DClassifier.from_state(model_config, parent_state, seed=seed)
    rng = np.random.default_rng(seed)
    batch_losses: list[float] = []
    local_steps = 0

    for _ in range(local_epochs):
        indices = rng.permutation(client.train.inputs.shape[0])
        for start in range(0, client.train.inputs.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_inputs = client.train.inputs[batch_indices]
            batch_labels = client.train.labels[batch_indices].astype(np.float32, copy=False)
            loss, gradients = model._loss_and_gradients(
                batch_inputs,
                batch_labels,
                positive_class_weight=positive_class_weight,
            )
            batch_losses.append(loss)

            for key in STATE_KEYS:
                corrected_gradient = (
                    np.asarray(gradients[key], dtype=np.float32)
                    - np.asarray(client_control_variate[key], dtype=np.float32)
                    + np.asarray(server_control_variate[key], dtype=np.float32)
                ).astype(np.float32, copy=False)
                model._state[key] = (
                    np.asarray(model._state[key], dtype=np.float32) - learning_rate * corrected_gradient
                ).astype(np.float32, copy=False)
            local_steps += 1

    if local_steps <= 0:
        raise ValueError(f"{client.client_id}: SCAFFOLD requires at least one local update step.")

    updated_state = model.state_dict()
    scale = float(1.0 / (local_steps * learning_rate))
    updated_control_variate: dict[str, np.ndarray] = {}
    delta_control_variate: dict[str, np.ndarray] = {}
    for key in STATE_KEYS:
        updated_control_variate[key] = (
            np.asarray(client_control_variate[key], dtype=np.float32)
            - np.asarray(server_control_variate[key], dtype=np.float32)
            + (
                np.asarray(parent_state[key], dtype=np.float32)
                - np.asarray(updated_state[key], dtype=np.float32)
            ) * scale
        ).astype(np.float32, copy=False)
        delta_control_variate[key] = (
            updated_control_variate[key] - np.asarray(client_control_variate[key], dtype=np.float32)
        ).astype(np.float32, copy=False)

    return ScaffoldTrainingResult(
        client_id=client.client_id,
        num_train_samples=client.num_train_samples,
        train_loss=float(np.mean(batch_losses)),
        updated_state=updated_state,
        updated_control_variate=updated_control_variate,
        delta_control_variate=delta_control_variate,
        local_steps=local_steps,
    )


def predict_split_scaffold(
    state: Mapping[str, np.ndarray],
    model_config: CNN1DConfig,
    split: ClientSplit,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    if split.num_samples == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int8)

    model = CNN1DClassifier.from_state(model_config, state)
    probabilities = model.predict_proba(split.inputs)
    predictions = (probabilities >= threshold).astype(np.int8, copy=False)
    return probabilities, predictions


def _validate_state_keys(state: Mapping[str, Any]) -> None:
    missing = [key for key in STATE_KEYS if key not in state]
    extra = [key for key in state.keys() if key not in STATE_KEYS]
    if missing or extra:
        raise ValueError(
            f"SCAFFOLD state keys mismatch. Missing={missing or '[]'} Extra={extra or '[]'}."
        )


def _validate_compatible_state(
    reference_state: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    state_name: str,
) -> None:
    _validate_state_keys(state)
    for key in STATE_KEYS:
        reference_value = np.asarray(reference_state[key], dtype=np.float32)
        current_value = np.asarray(state[key], dtype=np.float32)
        if reference_value.shape != current_value.shape:
            raise ValueError(
                f"SCAFFOLD {state_name} {key!r} has shape {current_value.shape}, expected {reference_value.shape}."
            )
