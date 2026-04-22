from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fl.aggregators import WeightedState, is_batch_norm_key  # noqa: E402
from src.fl.fedbn import (  # noqa: E402
    aggregate_fedbn_leaf_updates,
    aggregate_fedbn_subcluster_updates,
    merge_global_non_bn_with_local_bn,
    split_state_by_batch_norm,
)


class FedBNExclusionTests(unittest.TestCase):
    def test_merge_reuses_shared_non_bn_and_local_bn_state(self) -> None:
        shared_state = {
            "conv_weight": np.array([10.0], dtype=np.float32),
            "block_bn_weight": np.array([1.0], dtype=np.float32),
            "block_bn_running_mean": np.array([0.5], dtype=np.float32),
        }
        local_state = {
            "conv_weight": np.array([99.0], dtype=np.float32),
            "block_bn_weight": np.array([7.0], dtype=np.float32),
            "block_bn_running_mean": np.array([3.0], dtype=np.float32),
        }

        merged = merge_global_non_bn_with_local_bn(shared_state, local_state)

        self.assertEqual(float(merged["conv_weight"][0]), 10.0)
        self.assertEqual(float(merged["block_bn_weight"][0]), 7.0)
        self.assertEqual(float(merged["block_bn_running_mean"][0]), 3.0)

    def test_leaf_aggregation_excludes_bn_keys_and_preserves_reference_bn(self) -> None:
        reference_state = {
            "conv_weight": np.array([0.0], dtype=np.float32),
            "dense_weight": np.array([0.0], dtype=np.float32),
            "block_bn_weight": np.array([11.0], dtype=np.float32),
            "block_bn_running_mean": np.array([22.0], dtype=np.float32),
        }
        updates = [
            WeightedState(
                cluster_id=1,
                contributor_id="C1_L001",
                num_samples=1,
                state={
                    "conv_weight": np.array([1.0], dtype=np.float32),
                    "dense_weight": np.array([3.0], dtype=np.float32),
                    "block_bn_weight": np.array([101.0], dtype=np.float32),
                    "block_bn_running_mean": np.array([201.0], dtype=np.float32),
                },
            ),
            WeightedState(
                cluster_id=1,
                contributor_id="C1_L002",
                num_samples=3,
                state={
                    "conv_weight": np.array([5.0], dtype=np.float32),
                    "dense_weight": np.array([7.0], dtype=np.float32),
                    "block_bn_weight": np.array([301.0], dtype=np.float32),
                    "block_bn_running_mean": np.array([401.0], dtype=np.float32),
                },
            ),
        ]

        aggregated = aggregate_fedbn_leaf_updates(
            updates,
            cluster_id=1,
            reference_state=reference_state,
        )

        self.assertAlmostEqual(float(aggregated["conv_weight"][0]), 4.0)
        self.assertAlmostEqual(float(aggregated["dense_weight"][0]), 6.0)
        self.assertEqual(float(aggregated["block_bn_weight"][0]), 11.0)
        self.assertEqual(float(aggregated["block_bn_running_mean"][0]), 22.0)

    def test_maincluster_aggregation_excludes_bn_keys_and_bn_key_detection_matches_patterns(self) -> None:
        reference_state = {
            "linear_weight": np.array([0.0], dtype=np.float32),
            "tcn_block1_bn_running_var": np.array([9.0], dtype=np.float32),
        }
        updates = [
            WeightedState(
                cluster_id=1,
                contributor_id="H1",
                num_samples=2,
                state={
                    "linear_weight": np.array([2.0], dtype=np.float32),
                    "tcn_block1_bn_running_var": np.array([101.0], dtype=np.float32),
                },
            ),
            WeightedState(
                cluster_id=1,
                contributor_id="H2",
                num_samples=6,
                state={
                    "linear_weight": np.array([10.0], dtype=np.float32),
                    "tcn_block1_bn_running_var": np.array([201.0], dtype=np.float32),
                },
            ),
        ]

        aggregated = aggregate_fedbn_subcluster_updates(
            updates,
            cluster_id=1,
            reference_state=reference_state,
        )
        split = split_state_by_batch_norm(aggregated)

        self.assertAlmostEqual(float(aggregated["linear_weight"][0]), 8.0)
        self.assertEqual(float(aggregated["tcn_block1_bn_running_var"][0]), 9.0)
        self.assertTrue(is_batch_norm_key("tcn_block1_bn_running_var"))
        self.assertTrue(is_batch_norm_key("BatchNorm1d.weight"))
        self.assertEqual(set(split.bn_state.keys()), {"tcn_block1_bn_running_var"})
        self.assertEqual(set(split.non_bn_state.keys()), {"linear_weight"})


if __name__ == "__main__":
    unittest.main()
