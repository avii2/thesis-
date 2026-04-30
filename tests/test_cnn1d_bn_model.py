from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.cnn1d_bn import CNN1DBNClassifier, CNN1DBNConfig  # noqa: E402


class CNN1DBNModelTests(unittest.TestCase):
    def test_forward_pass_accepts_cluster1_windowed_input(self) -> None:
        model = CNN1DBNClassifier(
            CNN1DBNConfig(input_channels=5, input_length=32),
            seed=42,
        )
        inputs = np.random.default_rng(123).normal(size=(4, 5, 32)).astype(np.float32)

        logits = model.predict_logits(inputs)
        probabilities = model.predict_proba(inputs)

        self.assertEqual(logits.shape, (4,))
        self.assertEqual(probabilities.shape, (4,))
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))

    def test_state_dict_round_trip_preserves_predictions(self) -> None:
        config = CNN1DBNConfig(input_channels=3, input_length=16)
        model = CNN1DBNClassifier(config, seed=7)
        inputs = np.random.default_rng(9).normal(size=(3, 3, 16)).astype(np.float32)
        before = model.predict_logits(inputs)

        restored = CNN1DBNClassifier(config, seed=11)
        restored.load_state_dict(model.state_dict())
        after = restored.predict_logits(inputs)

        np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)

    def test_batch_norm_state_keys_are_explicit(self) -> None:
        model = CNN1DBNClassifier(CNN1DBNConfig(input_channels=2, input_length=8), seed=42)
        keys = set(model.state_dict())

        self.assertIn("block1_bn_weight", keys)
        self.assertIn("block1_bn_bias", keys)
        self.assertIn("block1_bn_running_mean", keys)
        self.assertIn("block1_bn_running_var", keys)
        self.assertIn("block1_bn_num_batches_tracked", keys)
        self.assertIn("block1_conv_weight", keys)
        self.assertIn("linear1_weight", keys)


if __name__ == "__main__":
    unittest.main()
