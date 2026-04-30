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
from src.models.cnn1d_bn import CNN1DBNClassifier, CNN1DBNConfig  # noqa: E402


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
            "block1_bn_running_var": np.array([9.0], dtype=np.float32),
        }
        updates = [
            WeightedState(
                cluster_id=1,
                contributor_id="H1",
                num_samples=2,
                state={
                    "linear_weight": np.array([2.0], dtype=np.float32),
                    "block1_bn_running_var": np.array([101.0], dtype=np.float32),
                },
            ),
            WeightedState(
                cluster_id=1,
                contributor_id="H2",
                num_samples=6,
                state={
                    "linear_weight": np.array([10.0], dtype=np.float32),
                    "block1_bn_running_var": np.array([201.0], dtype=np.float32),
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
        self.assertEqual(float(aggregated["block1_bn_running_var"][0]), 9.0)
        self.assertTrue(is_batch_norm_key("block1_bn_running_var"))
        self.assertTrue(is_batch_norm_key("BatchNorm1d.weight"))
        self.assertEqual(set(split.bn_state.keys()), {"block1_bn_running_var"})
        self.assertEqual(set(split.non_bn_state.keys()), {"linear_weight"})

    def test_cnn1d_bn_model_keys_are_detected_and_excluded(self) -> None:
        model = CNN1DBNClassifier(CNN1DBNConfig(input_channels=3, input_length=8), seed=7)
        state = model.state_dict()
        split = split_state_by_batch_norm(state)

        self.assertIn("block1_bn_weight", split.bn_state)
        self.assertIn("block1_bn_running_mean", split.bn_state)
        self.assertIn("block1_bn_running_var", split.bn_state)
        self.assertIn("block1_bn_num_batches_tracked", split.bn_state)
        self.assertIn("block1_conv_weight", split.non_bn_state)
        self.assertFalse(is_batch_norm_key("block1_conv_weight"))

        updates = [
            WeightedState(cluster_id=1, contributor_id="C1_L001", num_samples=1, state=state),
            WeightedState(
                cluster_id=1,
                contributor_id="C1_L002",
                num_samples=1,
                state={
                    key: (
                        np.asarray(value, dtype=np.float32) + np.float32(10.0)
                        if not is_batch_norm_key(key)
                        else np.asarray(value, dtype=np.float32) + np.float32(99.0)
                    )
                    for key, value in state.items()
                },
            ),
        ]
        aggregated = aggregate_fedbn_leaf_updates(
            updates,
            cluster_id=1,
            reference_state=state,
        )

        np.testing.assert_allclose(aggregated["block1_bn_running_mean"], state["block1_bn_running_mean"])
        np.testing.assert_allclose(aggregated["block1_conv_bias"], state["block1_conv_bias"] + np.float32(5.0))


if __name__ == "__main__":
    unittest.main()
