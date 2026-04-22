from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fl.aggregators import (  # noqa: E402
    AggregationError,
    CrossClusterAggregationError,
    WeightedState,
    aggregate_leaf_updates_to_subcluster,
    aggregate_subcluster_updates_to_maincluster,
)


class NoCrossClusterAveragingTests(unittest.TestCase):
    def test_leaf_aggregation_rejects_cross_cluster_inputs(self) -> None:
        updates = [
            WeightedState(
                cluster_id=1,
                contributor_id="C1_L001",
                num_samples=2,
                state={"weight": np.array([1.0], dtype=np.float64)},
            ),
            WeightedState(
                cluster_id=2,
                contributor_id="C2_L001",
                num_samples=2,
                state={"weight": np.array([3.0], dtype=np.float64)},
            ),
        ]

        with self.assertRaises(CrossClusterAggregationError) as context:
            aggregate_leaf_updates_to_subcluster(updates)

        self.assertIn("cross-cluster averaging is forbidden", str(context.exception))

    def test_maincluster_aggregation_rejects_expected_cluster_mismatch(self) -> None:
        updates = [
            WeightedState(
                cluster_id=3,
                contributor_id="W1",
                num_samples=2,
                state={"weight": np.array([1.0], dtype=np.float64)},
            ),
            WeightedState(
                cluster_id=3,
                contributor_id="W2",
                num_samples=2,
                state={"weight": np.array([3.0], dtype=np.float64)},
            ),
        ]

        with self.assertRaises(CrossClusterAggregationError) as context:
            aggregate_subcluster_updates_to_maincluster(updates, expected_cluster_id=2)

        self.assertIn("expected cluster_id=2", str(context.exception))

    def test_aggregation_rejects_incompatible_parameter_keys(self) -> None:
        updates = [
            WeightedState(
                cluster_id=2,
                contributor_id="T1",
                num_samples=2,
                state={"weight": np.array([1.0], dtype=np.float64)},
            ),
            WeightedState(
                cluster_id=2,
                contributor_id="T2",
                num_samples=2,
                state={"other_weight": np.array([3.0], dtype=np.float64)},
            ),
        ]

        with self.assertRaises(AggregationError) as context:
            aggregate_leaf_updates_to_subcluster(updates, expected_cluster_id=2)

        self.assertIn("incompatible parameter keys", str(context.exception))


if __name__ == "__main__":
    unittest.main()
