from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fl.client import ClientSplit, FlatClientDataset  # noqa: E402
from src.fl.maincluster import (  # noqa: E402
    evaluate_round_with_validation_threshold,
    select_threshold_maximizing_validation_f1,
    write_prediction_outputs,
)
from src.train import _write_confusion_matrices_report  # noqa: E402


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
        self.assertEqual(
            evaluation["test_metrics_default_threshold"]["threshold_used"],
            0.5,
        )

    def test_prediction_outputs_save_labels_probabilities_and_threshold(self) -> None:
        validation_labels = np.asarray([1, 0, 1], dtype=np.int8)
        validation_probabilities = np.asarray([0.9, 0.2, 0.7], dtype=np.float32)
        test_labels = np.asarray([0, 1], dtype=np.int8)
        test_probabilities = np.asarray([0.4, 0.8], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_prediction_outputs(
                output_root=tmpdir,
                experiment_id="TEST_C1",
                validation_labels=validation_labels,
                validation_probabilities=validation_probabilities,
                test_labels=test_labels,
                test_probabilities=test_probabilities,
                selected_threshold=0.7,
                seed=42,
            )

            with np.load(outputs["validation_predictions_path"]) as validation_npz:
                saved_validation_labels = validation_npz["labels"].copy()
                saved_validation_probabilities = validation_npz["probabilities"].copy()
            with np.load(outputs["test_predictions_path"]) as test_npz:
                saved_test_labels = test_npz["labels"].copy()
                saved_test_probabilities = test_npz["probabilities"].copy()
            threshold_payload = json.loads(Path(outputs["selected_threshold_path"]).read_text())

        np.testing.assert_array_equal(saved_validation_labels, validation_labels)
        np.testing.assert_allclose(saved_validation_probabilities, validation_probabilities)
        np.testing.assert_array_equal(saved_test_labels, test_labels)
        np.testing.assert_allclose(saved_test_probabilities, test_probabilities)
        self.assertEqual(threshold_payload["threshold_selected_on"], "validation")
        self.assertEqual(threshold_payload["selection_metric"], "f1")
        self.assertAlmostEqual(threshold_payload["selected_threshold"], 0.7)
        self.assertAlmostEqual(threshold_payload["default_threshold"], 0.5)

    def test_confusion_matrix_report_exports_tn_fp_fn_tp(self) -> None:
        row = {
            "experiment_id": "TEST_C1",
            "cluster_id": 1,
            "dataset": "synthetic",
            "run_category": "baseline_flat",
            "model_family": "cnn1d",
            "fl_method": "FedAvg",
            "aggregation": "weighted_arithmetic_mean",
            "threshold_used": 0.7,
            "test_confusion_matrix": json.dumps([[8, 2], [1, 9]]),
            "test_confusion_matrix_default_threshold": json.dumps([[7, 3], [4, 6]]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = _write_confusion_matrices_report([row], Path(tmpdir))
            payload = json.loads(report_path.read_text())

        self.assertEqual(payload["confusion_matrix_layout"], "[[TN, FP], [FN, TP]]")
        entry = payload["experiments"]["TEST_C1"]
        self.assertEqual(entry["threshold_selected_on"], "validation")
        self.assertEqual(entry["tuned_threshold"]["tn"], 8)
        self.assertEqual(entry["tuned_threshold"]["fp"], 2)
        self.assertEqual(entry["tuned_threshold"]["fn"], 1)
        self.assertEqual(entry["tuned_threshold"]["tp"], 9)
        self.assertEqual(entry["default_threshold_metrics"]["tn"], 7)
        self.assertEqual(entry["default_threshold_metrics"]["fp"], 3)


if __name__ == "__main__":
    unittest.main()
