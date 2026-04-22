from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.fl.aggregators import WeightedState
from src.fl.client import FlatClientDataset, ClientSplit
from src.models.mlp import CompactMLPClassifier, MLPConfig, STATE_KEYS


@dataclass(frozen=True)
class FedProxLossComponents:
    base_loss: float
    proximal_penalty: float

    @property
    def total_loss(self) -> float:
        return float(self.base_loss + self.proximal_penalty)


@dataclass(frozen=True)
class FedProxTrainingResult:
    client_id: str
    num_train_samples: int
    train_loss: float
    mean_base_loss: float
    mean_proximal_penalty: float
    updated_state: Mapping[str, np.ndarray]

    def to_weighted_state(self, *, cluster_id: int) -> WeightedState:
        return WeightedState(
            cluster_id=cluster_id,
            contributor_id=self.client_id,
            num_samples=self.num_train_samples,
            state=self.updated_state,
        )


def fedprox_loss_components(
    *,
    base_loss: float,
    state: Mapping[str, Any],
    reference_state: Mapping[str, Any],
    mu: float,
) -> FedProxLossComponents:
    proximal_penalty = compute_proximal_penalty(state, reference_state, mu=mu)
    return FedProxLossComponents(
        base_loss=float(base_loss),
        proximal_penalty=proximal_penalty,
    )


def compute_proximal_penalty(
    state: Mapping[str, Any],
    reference_state: Mapping[str, Any],
    *,
    mu: float,
) -> float:
    _validate_mu(mu)
    _validate_state_keys(state, reference_state)
    squared_norm = 0.0
    for key in STATE_KEYS:
        delta = np.asarray(state[key], dtype=np.float32) - np.asarray(reference_state[key], dtype=np.float32)
        squared_norm += float(np.sum(np.square(delta)))
    return float((mu / 2.0) * squared_norm)


def compute_proximal_gradients(
    state: Mapping[str, Any],
    reference_state: Mapping[str, Any],
    *,
    mu: float,
) -> dict[str, np.ndarray]:
    _validate_mu(mu)
    _validate_state_keys(state, reference_state)
    return {
        key: (mu * (np.asarray(state[key], dtype=np.float32) - np.asarray(reference_state[key], dtype=np.float32))).astype(
            np.float32,
            copy=False,
        )
        for key in STATE_KEYS
    }


def train_fedprox_client(
    client: FlatClientDataset,
    parent_state: Mapping[str, np.ndarray],
    model_config: MLPConfig,
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    mu: float,
    seed: int,
) -> FedProxTrainingResult:
    if client.num_train_samples <= 0:
        raise ValueError(f"{client.client_id}: train split must contain at least one sample.")
    _validate_mu(mu)

    train_inputs = _as_tabular_inputs(client.train)
    train_labels = client.train.labels.astype(np.float32, copy=False)
    model = CompactMLPClassifier.from_state(model_config, parent_state, seed=seed)
    reference_state = {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in parent_state.items()
    }
    optimizer_state: dict[str, dict[str, np.ndarray] | int] = {"t": 0}
    rng = np.random.default_rng(seed)

    total_losses: list[float] = []
    base_losses: list[float] = []
    proximal_penalties: list[float] = []

    for _ in range(local_epochs):
        indices = rng.permutation(train_inputs.shape[0])
        for start in range(0, train_inputs.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_inputs = train_inputs[batch_indices]
            batch_labels = train_labels[batch_indices]

            base_loss, gradients = model.loss_and_gradients(batch_inputs, batch_labels, rng=rng)
            current_state = model.state_dict()
            prox_components = fedprox_loss_components(
                base_loss=base_loss,
                state=current_state,
                reference_state=reference_state,
                mu=mu,
            )
            proximal_gradients = compute_proximal_gradients(current_state, reference_state, mu=mu)
            combined_gradients = {
                key: (np.asarray(gradients[key], dtype=np.float32) + proximal_gradients[key]).astype(np.float32, copy=False)
                for key in STATE_KEYS
            }
            model.apply_adam_gradients(
                combined_gradients,
                optimizer_state,
                learning_rate=learning_rate,
            )
            total_losses.append(prox_components.total_loss)
            base_losses.append(prox_components.base_loss)
            proximal_penalties.append(prox_components.proximal_penalty)

    return FedProxTrainingResult(
        client_id=client.client_id,
        num_train_samples=client.num_train_samples,
        train_loss=float(np.mean(total_losses)),
        mean_base_loss=float(np.mean(base_losses)),
        mean_proximal_penalty=float(np.mean(proximal_penalties)),
        updated_state=model.state_dict(),
    )


def predict_split_fedprox(
    state: Mapping[str, np.ndarray],
    model_config: MLPConfig,
    split: ClientSplit,
) -> tuple[np.ndarray, np.ndarray]:
    if split.num_samples == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int8)

    model = CompactMLPClassifier.from_state(model_config, state)
    probabilities = model.predict_proba(_as_tabular_inputs(split))
    predictions = (probabilities >= 0.5).astype(np.int8, copy=False)
    return probabilities, predictions


def _as_tabular_inputs(split: ClientSplit) -> np.ndarray:
    inputs = split.inputs.astype(np.float32, copy=False)
    if inputs.ndim == 2:
        return inputs
    if inputs.ndim == 3 and inputs.shape[1] == 1:
        return inputs[:, 0, :]
    raise ValueError(
        "FedProx Cluster 2 path expects inputs with shape (batch, features) or "
        f"(batch, 1, features). Observed {inputs.shape}."
    )


def _validate_mu(mu: float) -> None:
    if mu < 0.0:
        raise ValueError(f"FedProx mu must be non-negative. Observed {mu}.")


def _validate_state_keys(state: Mapping[str, Any], reference_state: Mapping[str, Any]) -> None:
    state_keys = set(state.keys())
    reference_keys = set(reference_state.keys())
    expected_keys = set(STATE_KEYS)
    if state_keys != expected_keys:
        missing = sorted(expected_keys - state_keys)
        extra = sorted(state_keys - expected_keys)
        raise ValueError(
            f"FedProx state mismatch. Missing={missing or '[]'} Extra={extra or '[]'}."
        )
    if reference_keys != expected_keys:
        missing = sorted(expected_keys - reference_keys)
        extra = sorted(reference_keys - expected_keys)
        raise ValueError(
            f"FedProx reference state mismatch. Missing={missing or '[]'} Extra={extra or '[]'}."
        )
