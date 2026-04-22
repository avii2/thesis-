from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fl.aggregators import (  # noqa: E402
    WeightedState,
    aggregate_leaf_updates_to_subcluster,
    aggregate_leaf_updates_to_subcluster_non_bn,
    aggregate_subcluster_updates_to_maincluster,
)


class WeightedAggregationTests(unittest.TestCase):
    def test_leaf_to_subcluster_weighted_mean_matches_hand_computation(self) -> None:
        updates = [
            WeightedState(
                cluster_id=2,
                contributor_id="client_a",
                num_samples=2,
                state={
                    "weight": np.array([1.0, 3.0], dtype=np.float64),
                    "bias": np.array([2.0], dtype=np.float64),
                },
            ),
            WeightedState(
                cluster_id=2,
                contributor_id="client_b",
                num_samples=6,
                state={
                    "weight": np.array([5.0, 7.0], dtype=np.float64),
                    "bias": np.array([10.0], dtype=np.float64),
                },
            ),
        ]

        aggregated = aggregate_leaf_updates_to_subcluster(updates, expected_cluster_id=2)

        np.testing.assert_allclose(aggregated["weight"], np.array([4.0, 6.0], dtype=np.float64), atol=1e-6)
        np.testing.assert_allclose(aggregated["bias"], np.array([8.0], dtype=np.float64), atol=1e-6)

    def test_subcluster_to_maincluster_weighted_mean_matches_hand_computation(self) -> None:
        updates = [
            WeightedState(
                cluster_id=3,
                contributor_id="W1",
                num_samples=3,
                state={"weight": np.array([1.0, 2.0], dtype=np.float64)},
            ),
            WeightedState(
                cluster_id=3,
                contributor_id="W2",
                num_samples=1,
                state={"weight": np.array([5.0, 10.0], dtype=np.float64)},
            ),
            WeightedState(
                cluster_id=3,
                contributor_id="W3",
                num_samples=2,
                state={"weight": np.array([7.0, 1.0], dtype=np.float64)},
            ),
        ]

        aggregated = aggregate_subcluster_updates_to_maincluster(updates, expected_cluster_id=3)

        expected = ((3 / 6) * np.array([1.0, 2.0])) + ((1 / 6) * np.array([5.0, 10.0])) + ((2 / 6) * np.array([7.0, 1.0]))
        np.testing.assert_allclose(aggregated["weight"], expected, atol=1e-6)

    def test_non_bn_helper_excludes_batchnorm_related_keys(self) -> None:
        updates = [
            WeightedState(
                cluster_id=1,
                contributor_id="client_a",
                num_samples=1,
                state={
                    "conv.weight": np.array([1.0, 2.0], dtype=np.float64),
                    "bn1.weight": np.array([3.0], dtype=np.float64),
                    "bn1.running_mean": np.array([4.0], dtype=np.float64),
                    "num_batches_tracked": np.array([5.0], dtype=np.float64),
                },
            ),
            WeightedState(
                cluster_id=1,
                contributor_id="client_b",
                num_samples=3,
                state={
                    "conv.weight": np.array([5.0, 6.0], dtype=np.float64),
                    "bn1.weight": np.array([30.0], dtype=np.float64),
                    "bn1.running_mean": np.array([40.0], dtype=np.float64),
                    "num_batches_tracked": np.array([50.0], dtype=np.float64),
                },
            ),
        ]

        aggregated = aggregate_leaf_updates_to_subcluster_non_bn(updates, expected_cluster_id=1)

        self.assertEqual(set(aggregated), {"conv.weight"})
        np.testing.assert_allclose(aggregated["conv.weight"], np.array([4.0, 5.0], dtype=np.float64), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
