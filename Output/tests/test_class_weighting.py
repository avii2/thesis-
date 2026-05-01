from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fl.client import ClientSplit, FlatClientDataset  # noqa: E402
from src.fl.maincluster import compute_cluster_positive_class_weight  # noqa: E402
from src.models.losses import binary_cross_entropy_with_logits  # noqa: E402
from src.models.cnn1d import CNN1DClassifier, CNN1DConfig  # noqa: E402


def _dummy_split(labels: list[int]) -> ClientSplit:
    inputs = np.zeros((len(labels), 1, 4), dtype=np.float32)
    return ClientSplit(inputs=inputs, labels=np.asarray(labels, dtype=np.int8))


class ClassWeightingTests(unittest.TestCase):
    def test_positive_class_weight_uses_training_labels_only(self) -> None:
        clients = [
            FlatClientDataset(
                cluster_id=1,
                client_id="C1",
                train=_dummy_split([0, 0, 1]),
                validation=_dummy_split([1, 1, 1, 1]),
                test=_dummy_split([1, 1, 1, 1]),
                input_adapter="sliding_window_feature_channels",
            ),
            FlatClientDataset(
                cluster_id=1,
                client_id="C2",
                train=_dummy_split([0, 1]),
                validation=_dummy_split([1, 1]),
                test=_dummy_split([1, 1]),
                input_adapter="sliding_window_feature_channels",
            ),
        ]

        weight = compute_cluster_positive_class_weight(clients)

        # Train labels only: negatives=3, positives=2.
        self.assertAlmostEqual(weight, 1.5)

    def test_weighted_binary_cross_entropy_changes_positive_loss_contribution(self) -> None:
        model = CNN1DClassifier(CNN1DConfig(input_channels=1, input_length=4), seed=7)
        inputs = np.ones((2, 1, 4), dtype=np.float32)
        labels = np.asarray([1, 0], dtype=np.int8)

        unweighted_loss = model.binary_cross_entropy(inputs, labels, positive_class_weight=1.0)
        weighted_loss = model.binary_cross_entropy(inputs, labels, positive_class_weight=3.0)

        self.assertGreater(weighted_loss, unweighted_loss)

    def test_bce_with_logits_pos_weight_matches_closed_form(self) -> None:
        logits = np.asarray([-2.0, 0.0, 2.0], dtype=np.float32)
        labels = np.asarray([1, 0, 1], dtype=np.float32)
        pos_weight = 4.0

        loss = binary_cross_entropy_with_logits(
            logits,
            labels,
            positive_class_weight=pos_weight,
        )

        expected = np.mean(
            pos_weight * labels * np.log1p(np.exp(-logits))
            + (1.0 - labels) * np.log1p(np.exp(logits))
        )
        self.assertAlmostEqual(loss, float(expected), places=6)

    def test_model_loss_uses_bce_with_logits_pos_weight(self) -> None:
        model = CNN1DClassifier(CNN1DConfig(input_channels=1, input_length=4), seed=11)
        inputs = np.ones((3, 1, 4), dtype=np.float32)
        labels = np.asarray([1, 0, 1], dtype=np.int8)
        pos_weight = 2.5

        model_loss = model.binary_cross_entropy(
            inputs,
            labels,
            positive_class_weight=pos_weight,
        )
        direct_loss = binary_cross_entropy_with_logits(
            model.predict_logits(inputs),
            labels,
            positive_class_weight=pos_weight,
        )

        self.assertAlmostEqual(model_loss, direct_loss, places=7)


if __name__ == "__main__":
    unittest.main()
