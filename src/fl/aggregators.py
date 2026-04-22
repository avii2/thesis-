from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


BN_EXCLUSION_PATTERNS = (
    "bn",
    "batchnorm",
    "running_mean",
    "running_var",
    "num_batches_tracked",
)


class AggregationError(ValueError):
    """Raised when weighted aggregation inputs are invalid."""


class CrossClusterAggregationError(AggregationError):
    """Raised when aggregation inputs span multiple main clusters."""


@dataclass(frozen=True)
class WeightedState:
    cluster_id: int
    contributor_id: str
    num_samples: int
    state: Mapping[str, Any]


def is_batch_norm_key(key: str) -> bool:
    lowered = key.lower()
    return any(pattern in lowered for pattern in BN_EXCLUSION_PATTERNS)


def aggregate_leaf_updates_to_subcluster(
    updates: Sequence[WeightedState],
    *,
    expected_cluster_id: int | None = None,
) -> dict[str, Any]:
    return _aggregate_weighted_states(
        updates,
        expected_cluster_id=expected_cluster_id,
        aggregation_scope="leaf_to_subcluster",
        excluded_keys=(),
    )


def aggregate_subcluster_updates_to_maincluster(
    updates: Sequence[WeightedState],
    *,
    expected_cluster_id: int | None = None,
) -> dict[str, Any]:
    return _aggregate_weighted_states(
        updates,
        expected_cluster_id=expected_cluster_id,
        aggregation_scope="subcluster_to_maincluster",
        excluded_keys=(),
    )


def aggregate_leaf_updates_to_subcluster_non_bn(
    updates: Sequence[WeightedState],
    *,
    expected_cluster_id: int | None = None,
) -> dict[str, Any]:
    return _aggregate_weighted_states(
        updates,
        expected_cluster_id=expected_cluster_id,
        aggregation_scope="leaf_to_subcluster_non_bn",
        excluded_keys=tuple(key for key in _reference_keys(updates) if is_batch_norm_key(key)),
    )


def aggregate_subcluster_updates_to_maincluster_non_bn(
    updates: Sequence[WeightedState],
    *,
    expected_cluster_id: int | None = None,
) -> dict[str, Any]:
    return _aggregate_weighted_states(
        updates,
        expected_cluster_id=expected_cluster_id,
        aggregation_scope="subcluster_to_maincluster_non_bn",
        excluded_keys=tuple(key for key in _reference_keys(updates) if is_batch_norm_key(key)),
    )


def _reference_keys(updates: Sequence[WeightedState]) -> tuple[str, ...]:
    _require_non_empty_updates(updates, aggregation_scope="reference_keys")
    first = updates[0]
    return tuple(first.state.keys())


def _aggregate_weighted_states(
    updates: Sequence[WeightedState],
    *,
    expected_cluster_id: int | None,
    aggregation_scope: str,
    excluded_keys: Sequence[str],
) -> dict[str, Any]:
    _require_non_empty_updates(updates, aggregation_scope=aggregation_scope)
    cluster_id = _require_same_cluster(updates, expected_cluster_id=expected_cluster_id, aggregation_scope=aggregation_scope)
    del cluster_id

    reference_keys = set(updates[0].state.keys())
    for update in updates[1:]:
        current_keys = set(update.state.keys())
        if current_keys != reference_keys:
            missing = sorted(reference_keys - current_keys)
            extra = sorted(current_keys - reference_keys)
            raise AggregationError(
                f"{aggregation_scope}: contributor {update.contributor_id!r} has incompatible parameter keys. "
                f"Missing={missing or '[]'} Extra={extra or '[]'}."
            )

    excluded_lookup = set(excluded_keys)
    keys_to_aggregate = [key for key in updates[0].state if key not in excluded_lookup]
    if not keys_to_aggregate:
        raise AggregationError(
            f"{aggregation_scope}: no parameters remain after applying exclusion rules."
        )

    total_weight = 0
    for update in updates:
        if not isinstance(update.num_samples, int) or update.num_samples <= 0:
            raise AggregationError(
                f"{aggregation_scope}: contributor {update.contributor_id!r} must have positive integer num_samples."
            )
        total_weight += update.num_samples

    aggregated: dict[str, Any] = {}
    for key in keys_to_aggregate:
        weighted_sum = None
        for update in updates:
            contribution = update.state[key] * (update.num_samples / total_weight)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        aggregated[key] = weighted_sum

    return aggregated


def _require_non_empty_updates(
    updates: Sequence[WeightedState],
    *,
    aggregation_scope: str,
) -> None:
    if not updates:
        raise AggregationError(f"{aggregation_scope}: at least one weighted state is required.")


def _require_same_cluster(
    updates: Sequence[WeightedState],
    *,
    expected_cluster_id: int | None,
    aggregation_scope: str,
) -> int:
    cluster_ids = {update.cluster_id for update in updates}
    if len(cluster_ids) != 1:
        observed = sorted(cluster_ids)
        raise CrossClusterAggregationError(
            f"{aggregation_scope}: cross-cluster averaging is forbidden. Observed cluster_ids={observed}."
        )

    cluster_id = next(iter(cluster_ids))
    if expected_cluster_id is not None and cluster_id != expected_cluster_id:
        raise CrossClusterAggregationError(
            f"{aggregation_scope}: expected cluster_id={expected_cluster_id} but observed cluster_id={cluster_id}."
        )

    return cluster_id
