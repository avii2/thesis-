from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fl.fedprox import compute_proximal_gradients, fedprox_loss_components  # noqa: E402
from src.models.mlp import CompactMLPClassifier, MLPConfig  # noqa: E402


class FedProxLossTests(unittest.TestCase):
    def test_fedprox_loss_equals_base_loss_plus_proximal_term(self) -> None:
        config = MLPConfig(input_dim=3)
        model = CompactMLPClassifier(config, seed=7)
        state = model.state_dict()
        reference = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in state.items()
        }
        reference["linear1_weight"][0, 0] = reference["linear1_weight"][0, 0] + 2.0
        reference["linear3_bias"] = np.asarray(float(reference["linear3_bias"]) - 1.0, dtype=np.float32)

        components = fedprox_loss_components(
            base_loss=0.75,
            state=state,
            reference_state=reference,
            mu=0.01,
        )

        expected_penalty = 0.5 * 0.01 * ((2.0 ** 2) + (1.0 ** 2))
        self.assertAlmostEqual(components.base_loss, 0.75)
        self.assertAlmostEqual(components.proximal_penalty, expected_penalty, places=6)
        self.assertAlmostEqual(components.total_loss, 0.75 + expected_penalty, places=6)

    def test_fedprox_gradients_are_zero_when_state_matches_reference(self) -> None:
        config = MLPConfig(input_dim=4)
        model = CompactMLPClassifier(config, seed=11)
        state = model.state_dict()

        gradients = compute_proximal_gradients(
            state,
            state,
            mu=0.01,
        )

        for gradient in gradients.values():
            self.assertTrue(np.allclose(gradient, 0.0))


if __name__ == "__main__":
    unittest.main()
