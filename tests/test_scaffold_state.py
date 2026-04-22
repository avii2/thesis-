from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fl.client import ClientSplit, FlatClientDataset  # noqa: E402
from src.fl.scaffold import train_scaffold_client, update_server_control_variate, zero_control_variate_like  # noqa: E402
from src.models.cnn1d import CNN1DClassifier, CNN1DConfig, STATE_KEYS  # noqa: E402


class ScaffoldStateTests(unittest.TestCase):
    def test_server_control_variate_update_matches_average_increment_over_total_leaf_clients(self) -> None:
        config = CNN1DConfig(input_channels=1, input_length=4)
        model = CNN1DClassifier(config, seed=3)
        reference_state = model.state_dict()
        server_control = zero_control_variate_like(reference_state)

        delta_one = zero_control_variate_like(reference_state)
        delta_two = zero_control_variate_like(reference_state)
        delta_one["conv_bias"] = np.array([1.0] * config.num_filters, dtype=np.float32)
        delta_two["conv_bias"] = np.array([3.0] * config.num_filters, dtype=np.float32)
        delta_one["dense_bias"] = np.asarray(2.0, dtype=np.float32)
        delta_two["dense_bias"] = np.asarray(6.0, dtype=np.float32)

        updated = update_server_control_variate(
            server_control,
            [delta_one, delta_two],
            total_leaf_clients=4,
        )

        self.assertTrue(np.allclose(updated["conv_bias"], np.array([1.0] * config.num_filters, dtype=np.float32)))
        self.assertAlmostEqual(float(updated["dense_bias"]), 2.0, places=6)

    def test_client_control_variate_update_matches_spec_formula_when_initial_controls_are_zero(self) -> None:
        config = CNN1DConfig(input_channels=1, input_length=4)
        parent_model = CNN1DClassifier(config, seed=9)
        parent_state = parent_model.state_dict()
        zero_control = zero_control_variate_like(parent_state)

        inputs = np.asarray(
            [
                [[0.0, 1.0, 2.0, 3.0]],
                [[1.0, 2.0, 3.0, 4.0]],
                [[2.0, 3.0, 4.0, 5.0]],
                [[3.0, 4.0, 5.0, 6.0]],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        empty_inputs = np.empty((0, 1, 4), dtype=np.float32)
        empty_labels = np.empty(0, dtype=np.int8)
        client = FlatClientDataset(
            cluster_id=3,
            client_id="C3_L001",
            train=ClientSplit(inputs=inputs, labels=labels),
            validation=ClientSplit(inputs=empty_inputs, labels=empty_labels),
            test=ClientSplit(inputs=empty_inputs, labels=empty_labels),
            input_adapter="feature_vector_as_sequence",
        )

        result = train_scaffold_client(
            client,
            parent_state,
            zero_control,
            zero_control,
            config,
            local_epochs=1,
            batch_size=2,
            learning_rate=0.01,
            seed=42,
        )

        self.assertGreater(result.local_steps, 0)
        self.assertGreater(result.train_loss, 0.0)
        for key in STATE_KEYS:
            expected = (
                np.asarray(parent_state[key], dtype=np.float32) - np.asarray(result.updated_state[key], dtype=np.float32)
            ) / float(result.local_steps * 0.01)
            self.assertTrue(np.allclose(result.updated_control_variate[key], expected, atol=1e-5))
            self.assertTrue(np.allclose(result.delta_control_variate[key], expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
