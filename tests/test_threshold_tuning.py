from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fl.client import ClientSplit, FlatClientDataset  # noqa: E402
from src.fl.maincluster import (  # noqa: E402
    evaluate_round_with_validation_threshold,
    select_threshold_maximizing_validation_f1,
)


def _split(labels: list[int]) -> ClientSplit:
    return ClientSplit(
        inputs=np.zeros((len(labels), 1, 4), dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int8),
    )


class ThresholdTuningTests(unittest.TestCase):
    def test_select_threshold_maximizing_validation_f1(self) -> None:
        labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
        probabilities = np.asarray([0.9, 0.4, 0.6, 0.1], dtype=np.float32)

        threshold = select_threshold_maximizing_validation_f1(labels, probabilities)

        self.assertAlmostEqual(threshold, 0.4)

    def test_test_split_is_not_used_for_threshold_selection(self) -> None:
        client = FlatClientDataset(
            cluster_id=1,
            client_id="C1",
            train=_split([0, 1, 0, 1]),
            validation=_split([1, 1, 0, 0]),
            test=_split([1, 1, 0, 0]),
            input_adapter="sliding_window_feature_channels",
        )

        validation_probabilities = np.asarray([0.9, 0.4, 0.6, 0.1], dtype=np.float32)
        test_probabilities = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)

        def predictor(current_client: FlatClientDataset, split: ClientSplit) -> np.ndarray:
            if split is current_client.train:
                return np.asarray([0.2, 0.8, 0.3, 0.7], dtype=np.float32)
            if split is current_client.validation:
                return validation_probabilities
            if split is current_client.test:
                return test_probabilities
            raise AssertionError("Unexpected split")

        evaluation = evaluate_round_with_validation_threshold([client], predictor=predictor)

        validation_threshold = select_threshold_maximizing_validation_f1(
            client.validation.labels,
            validation_probabilities,
        )
        test_optimal_threshold = select_threshold_maximizing_validation_f1(
            client.test.labels,
            test_probabilities,
        )

        self.assertAlmostEqual(evaluation["selected_threshold"], validation_threshold)
        self.assertNotAlmostEqual(evaluation["selected_threshold"], test_optimal_threshold)
        self.assertAlmostEqual(
            evaluation["test_metrics"]["threshold_used"],
            validation_threshold,
        )


if __name__ == "__main__":
    unittest.main()
