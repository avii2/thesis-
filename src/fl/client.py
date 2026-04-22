from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.fl.aggregators import WeightedState
from src.models.cnn1d import CNN1DClassifier, CNN1DConfig


@dataclass(frozen=True)
class ClientSplit:
    inputs: np.ndarray
    labels: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class FlatClientDataset:
    cluster_id: int
    client_id: str
    train: ClientSplit
    validation: ClientSplit
    test: ClientSplit
    input_adapter: str

    @property
    def num_train_samples(self) -> int:
        return self.train.num_samples


@dataclass(frozen=True)
class LocalTrainingResult:
    client_id: str
    num_train_samples: int
    train_loss: float
    updated_state: Mapping[str, np.ndarray]

    def to_weighted_state(self, *, cluster_id: int) -> WeightedState:
        return WeightedState(
            cluster_id=cluster_id,
            contributor_id=self.client_id,
            num_samples=self.num_train_samples,
            state=self.updated_state,
        )


def train_flat_client(
    client: FlatClientDataset,
    global_state: Mapping[str, np.ndarray],
    model_config: CNN1DConfig,
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> LocalTrainingResult:
    if client.num_train_samples <= 0:
        raise ValueError(f"{client.client_id}: train split must contain at least one sample.")

    model = CNN1DClassifier.from_state(model_config, global_state, seed=seed)
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
            )
        )

    return LocalTrainingResult(
        client_id=client.client_id,
        num_train_samples=client.num_train_samples,
        train_loss=float(np.mean(epoch_losses)),
        updated_state=model.state_dict(),
    )


def predict_split(
    state: Mapping[str, np.ndarray],
    model_config: CNN1DConfig,
    split: ClientSplit,
) -> tuple[np.ndarray, np.ndarray]:
    if split.num_samples == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int8)

    model = CNN1DClassifier.from_state(model_config, state)
    probabilities = model.predict_proba(split.inputs)
    predictions = (probabilities >= 0.5).astype(np.int8, copy=False)
    return probabilities, predictions
