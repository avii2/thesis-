from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.fl.aggregators import WeightedState, aggregate_leaf_updates_to_subcluster
from src.fl.client import FlatClientDataset, LocalTrainingResult, train_flat_client
from src.models.cnn1d import CNN1DConfig


@dataclass(frozen=True)
class FrozenSubclusterMembership:
    subcluster_id: str
    client_ids: tuple[str, ...]


@dataclass(frozen=True)
class FrozenMembership:
    cluster_id: int
    dataset: str
    membership_file: Path
    membership_hash: str
    n_subclusters: int
    fixed_subcluster_ids: tuple[str, ...]
    subclusters: tuple[FrozenSubclusterMembership, ...]
    client_to_subcluster: Mapping[str, str]
    frozen: bool


@dataclass(frozen=True)
class SubclusterRoundResult:
    cluster_id: int
    subcluster_id: str
    num_train_samples: int
    num_clients: int
    mean_local_loss: float
    aggregated_state: Mapping[str, np.ndarray]

    def to_weighted_state(self) -> WeightedState:
        return WeightedState(
            cluster_id=self.cluster_id,
            contributor_id=self.subcluster_id,
            num_samples=self.num_train_samples,
            state=self.aggregated_state,
        )


def load_frozen_membership(
    membership_path: str | Path,
    *,
    expected_cluster_id: int,
    expected_n_subclusters: int,
    expected_client_ids: Sequence[str] | None = None,
) -> FrozenMembership:
    path = Path(membership_path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen membership file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Frozen membership file must contain a JSON object: {path}")

    cluster_id = int(raw.get("cluster_id"))
    if cluster_id != expected_cluster_id:
        raise ValueError(
            f"{path}: expected cluster_id={expected_cluster_id}, observed {cluster_id}."
        )

    frozen = bool(raw.get("frozen"))
    if not frozen:
        raise ValueError(f"{path}: hierarchical baseline requires frozen memberships.")

    n_subclusters = int(raw.get("n_subclusters"))
    if n_subclusters != expected_n_subclusters:
        raise ValueError(
            f"{path}: expected n_subclusters={expected_n_subclusters}, observed {n_subclusters}."
        )

    fixed_subcluster_ids = tuple(str(value) for value in raw.get("fixed_subcluster_ids", []))
    if len(fixed_subcluster_ids) != expected_n_subclusters:
        raise ValueError(
            f"{path}: fixed_subcluster_ids length {len(fixed_subcluster_ids)} does not match "
            f"expected n_subclusters={expected_n_subclusters}."
        )

    subcluster_entries = raw.get("subclusters")
    client_entries = raw.get("clients")
    if not isinstance(subcluster_entries, list) or not isinstance(client_entries, list):
        raise ValueError(f"{path}: missing subclusters or clients entries.")

    subclusters: list[FrozenSubclusterMembership] = []
    subcluster_id_lookup: set[str] = set()
    client_to_subcluster: dict[str, str] = {}
    for subcluster in subcluster_entries:
        if not isinstance(subcluster, dict):
            raise ValueError(f"{path}: each subcluster entry must be an object.")
        subcluster_id = str(subcluster.get("subcluster_id"))
        client_ids = tuple(str(client_id) for client_id in subcluster.get("client_ids", []))
        if not client_ids:
            raise ValueError(f"{path}: subcluster {subcluster_id!r} is empty.")
        if subcluster_id in subcluster_id_lookup:
            raise ValueError(f"{path}: duplicate subcluster_id {subcluster_id!r}.")
        subcluster_id_lookup.add(subcluster_id)
        for client_id in client_ids:
            if client_id in client_to_subcluster:
                raise ValueError(f"{path}: client {client_id!r} appears in multiple subclusters.")
            client_to_subcluster[client_id] = subcluster_id
        subclusters.append(
            FrozenSubclusterMembership(
                subcluster_id=subcluster_id,
                client_ids=client_ids,
            )
        )

    if tuple(subcluster_id_lookup) and set(fixed_subcluster_ids) != subcluster_id_lookup:
        raise ValueError(
            f"{path}: subcluster ids in fixed_subcluster_ids do not match subclusters payload."
        )

    declared_clients = {
        str(entry.get("client_id")): str(entry.get("subcluster_id"))
        for entry in client_entries
        if isinstance(entry, dict)
    }
    if declared_clients != client_to_subcluster:
        raise ValueError(f"{path}: client membership table does not match subcluster table.")

    if expected_client_ids is not None:
        expected_set = set(expected_client_ids)
        observed_set = set(client_to_subcluster)
        if observed_set != expected_set:
            missing = sorted(expected_set - observed_set)
            extra = sorted(observed_set - expected_set)
            raise ValueError(
                f"{path}: frozen membership client set mismatch. Missing={missing or '[]'} Extra={extra or '[]'}."
            )

    reuse_groups = raw.get("reuse_for_experiment_groups", [])
    required_reuse_groups = (
        "baseline_uniform_hierarchical",
        "proposed_specialized_hierarchical",
    )
    missing_reuse_groups = [
        group for group in required_reuse_groups if group not in reuse_groups
    ]
    if missing_reuse_groups:
        raise ValueError(
            f"{path}: frozen membership file is missing required reuse groups {missing_reuse_groups}."
        )

    return FrozenMembership(
        cluster_id=cluster_id,
        dataset=str(raw.get("dataset")),
        membership_file=path,
        membership_hash=str(raw.get("membership_hash")),
        n_subclusters=n_subclusters,
        fixed_subcluster_ids=fixed_subcluster_ids,
        subclusters=tuple(subclusters),
        client_to_subcluster=client_to_subcluster,
        frozen=frozen,
    )


def group_clients_by_subcluster(
    clients: Sequence[FlatClientDataset],
    membership: FrozenMembership,
) -> dict[str, tuple[FlatClientDataset, ...]]:
    by_client_id = {client.client_id: client for client in clients}
    grouped: dict[str, tuple[FlatClientDataset, ...]] = {}
    for subcluster in membership.subclusters:
        grouped[subcluster.subcluster_id] = tuple(
            by_client_id[client_id] for client_id in subcluster.client_ids
        )
    return grouped


def run_subcluster_round(
    *,
    cluster_id: int,
    subcluster_id: str,
    clients: Sequence[FlatClientDataset],
    parent_state: Mapping[str, np.ndarray],
    model_config: CNN1DConfig,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    positive_class_weight: float = 1.0,
) -> SubclusterRoundResult:
    if not clients:
        raise ValueError(f"{subcluster_id}: hierarchical baseline requires at least one leaf client.")

    local_updates: list[WeightedState] = []
    local_losses: list[float] = []
    for client_index, client in enumerate(clients):
        result: LocalTrainingResult = train_flat_client(
            client,
            parent_state,
            model_config,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + client_index,
            positive_class_weight=positive_class_weight,
        )
        local_updates.append(result.to_weighted_state(cluster_id=cluster_id))
        local_losses.append(result.train_loss)

    aggregated_state = aggregate_leaf_updates_to_subcluster(
        local_updates,
        expected_cluster_id=cluster_id,
    )
    return SubclusterRoundResult(
        cluster_id=cluster_id,
        subcluster_id=subcluster_id,
        num_train_samples=sum(client.train.num_samples for client in clients),
        num_clients=len(clients),
        mean_local_loss=float(np.mean(local_losses)),
        aggregated_state=aggregated_state,
    )
