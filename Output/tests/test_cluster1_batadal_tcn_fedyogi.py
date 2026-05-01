from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fl.aggregators import CrossClusterAggregationError, WeightedState  # noqa: E402
from src.fl.fedyogi import (  # noqa: E402
    FedYogiConfig,
    apply_fedyogi_delta,
    initialize_fedyogi_state,
    weighted_average_delta,
)
from src.models.tcn_gn import TCNGNClassifier, TCNGNConfig  # noqa: E402
from src.train_cluster1_batadal_alt import (  # noqa: E402
    EXPERIMENT_ID,
    _write_report,
    select_validation_threshold_row,
)


class Cluster1BatadalTcnFedYogiTests(unittest.TestCase):
    def test_alternative_config_is_cluster1_only_and_does_not_replace_current_proposed(self) -> None:
        config = yaml.safe_load(
            (REPO_ROOT / "configs/proposed_cluster1_batadal_tcn_fedyogi.yaml").read_text(encoding="utf-8")
        )
        entry = config["clusters"][0]

        self.assertEqual(entry["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(entry["cluster_id"], 1)
        self.assertEqual(entry["dataset"], "BATADAL")
        self.assertEqual(entry["num_leaf_clients"], 12)
        self.assertEqual(entry["n_subclusters"], 2)
        self.assertEqual(entry["membership_file"], "outputs_c1_batadal/clustering/cluster1_memberships.json")
        self.assertEqual(entry["model_family"], "tcn_gn")
        self.assertEqual(entry["fl_method"], "FedYogi")
        self.assertEqual(entry["aggregation"], "hierarchical_fedyogi_delta")
        self.assertEqual(entry["output_root"], "outputs_c1_batadal_alt")
        self.assertTrue(config["runtime_validation"]["do_not_replace_existing_p_c1_batadal"])

    def test_tcn_gn_forward_state_keys_and_training_step(self) -> None:
        config = TCNGNConfig(
            input_channels=3,
            input_length=8,
            channels=8,
            dilations=(1, 2),
            groups=2,
            hidden_dim=4,
            dropout=0.0,
        )
        model = TCNGNClassifier(config, seed=42)
        inputs = np.random.default_rng(7).normal(size=(6, 3, 8)).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        before = model.state_dict()

        probabilities = model.predict_proba(inputs)
        loss = model.train_epoch(
            inputs,
            labels,
            batch_size=3,
            learning_rate=0.001,
            rng=np.random.default_rng(9),
            positive_class_weight=1.0,
        )
        after = model.state_dict()

        self.assertEqual(probabilities.shape, (6,))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertTrue(np.isfinite(loss))
        self.assertIn("tcn_block1_gn_weight", after)
        self.assertNotIn("batch", " ".join(after).lower())
        self.assertTrue(any(not np.allclose(before[key], after[key]) for key in before))

    def test_fedyogi_weighted_delta_rejects_cross_cluster_inputs(self) -> None:
        updates = [
            WeightedState(cluster_id=1, contributor_id="C1_L001", num_samples=1, state={"w": np.array([1.0])}),
            WeightedState(cluster_id=2, contributor_id="C2_L001", num_samples=1, state={"w": np.array([3.0])}),
        ]

        with self.assertRaises(CrossClusterAggregationError):
            weighted_average_delta(
                updates,
                expected_cluster_id=1,
                aggregation_scope="test_leaf_to_subcluster",
            )

    def test_fedyogi_update_moves_state_in_delta_direction(self) -> None:
        current = {"w": np.array([0.0], dtype=np.float32)}
        updates = [
            WeightedState(cluster_id=1, contributor_id="C1_L001", num_samples=1, state={"w": np.array([1.0])}),
            WeightedState(cluster_id=1, contributor_id="C1_L002", num_samples=3, state={"w": np.array([3.0])}),
        ]
        averaged = weighted_average_delta(
            updates,
            expected_cluster_id=1,
            aggregation_scope="test_subcluster_to_main",
        )
        optimizer_state = initialize_fedyogi_state(current)
        next_state, next_optimizer_state = apply_fedyogi_delta(
            current,
            averaged,
            optimizer_state,
            FedYogiConfig(server_lr=0.01, beta1=0.9, beta2=0.99, tau=1e-3),
        )

        self.assertAlmostEqual(float(averaged["w"][0]), 2.5)
        self.assertGreater(float(next_state["w"][0]), 0.0)
        self.assertEqual(next_optimizer_state.step, 1)

    def test_threshold_selection_mode_respects_validation_fpr_constraint(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        probabilities = np.array([0.90, 0.10, 0.80, 0.20], dtype=np.float32)

        selected = select_validation_threshold_row(
            labels,
            probabilities,
            mode="max_validation_f1_with_validation_fpr_le_0.05",
        )

        self.assertLessEqual(float(selected["validation_fpr"]), 0.05)

    def test_threshold_selection_has_validation_only_predict_none_fallback(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        probabilities = np.array([0.995, 0.994, 0.993, 0.992], dtype=np.float32)

        selected = select_validation_threshold_row(
            labels,
            probabilities,
            mode="max_validation_f1_with_validation_fpr_le_0.05",
        )

        self.assertEqual(float(selected["threshold"]), 1.0)
        self.assertEqual(float(selected["validation_fpr"]), 0.0)

    def test_report_writer_handles_unavailable_curve_metrics(self) -> None:
        rows = [
            {
                "positive_class_weight_scale": 0.25,
                "threshold_selection_mode": "max_validation_f1",
                "best_validation_round": 1,
                "selected_threshold": 0.5,
                "validation_f1": 0.1,
                "validation_fpr": 0.0,
                "test_accuracy": None,
                "test_precision": 0.0,
                "test_recall": 0.0,
                "test_f1": 0.0,
                "test_auroc": "metric_unavailable_single_class",
                "test_pr_auc": "metric_unavailable_single_class",
                "test_fpr": 0.0,
                "test_confusion_matrix": "[[1, 0], [0, 0]]",
                "total_communication_cost_bytes": 1,
                "training_time_seconds": 0.01,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.md"
            _write_report(
                report_path,
                rows=rows,
                metrics_csv_path=Path("metrics.csv"),
                membership_hash="abc",
                data_summary={
                    "input_adapter": "sliding_window_feature_channels",
                    "input_channels": 1,
                    "input_length": 2,
                },
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("metric_unavailable_single_class", report)
        self.assertIn("communication bytes", report)


if __name__ == "__main__":
    unittest.main()
