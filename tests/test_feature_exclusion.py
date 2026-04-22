from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_dataset_for_training  # noqa: E402
from src.data.schema_validation import DatasetConfigError  # noqa: E402
from src.data.preprocess import fit_preprocessor_for_training  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_config(
    path: Path,
    raw_dir: Path,
    *,
    excluded_columns: list[str],
    exclude_if_present: list[str],
) -> None:
    config = {
        "config_version": 1,
        "cluster": {
            "id": 77,
            "key": "T77",
            "dataset_key": "TEST_FEATURE_EXCLUSION",
            "dataset_name": "Feature Exclusion Test",
            "audit_report": "outputs/reports/test_feature_exclusion.json",
        },
        "data": {
            "data_root_env_var": "FCFL_DATA_ROOT",
            "default_data_root": str(raw_dir.parent),
            "current_raw_input_dir": str(raw_dir),
            "current_raw_files": ["sample.csv"],
            "training_input_mode": "raw_csv_glob",
            "training_input_glob": None,
            "training_input_path": None,
            "expected_processed_input_path": None,
            "label_column": "attack",
            "label_column_confirmed_from_audit": True,
            "candidate_label_columns_present": ["attack"],
            "timestamp_or_order_columns": ["time"],
            "excluded_columns": excluded_columns,
            "exclude_if_present": exclude_if_present,
        },
        "runtime_validation": {
            "require_training_input_to_exist": True,
            "require_label_column_to_exist": True,
            "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


class FeatureExclusionTests(unittest.TestCase):
    def test_loader_reports_present_excluded_and_retained_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "sample.csv",
                [
                    ["time", "sensor", "attack", "attack_P1", "AttackType"],
                    ["2026-01-01", "1.0", "0", "0", "none"],
                    ["2026-01-02", "2.0", "1", "1", "spoof"],
                ],
            )
            config_path = tmp_path / "cluster.yaml"
            write_config(
                config_path,
                raw_dir,
                excluded_columns=["attack", "time", "attack_P1"],
                exclude_if_present=["AttackType"],
            )

            inspection = load_dataset_for_training(config_path)
            file_inspection = inspection.file_inspections[0]

            self.assertEqual(file_inspection.present_excluded_columns, ("time", "attack", "attack_P1", "AttackType"))
            self.assertEqual(file_inspection.retained_feature_columns, ("sensor",))

    def test_preprocessor_excludes_label_timestamp_and_leakage_from_output_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "sample.csv",
                [
                    ["time", "sensor", "attack", "attack_P1", "AttackType"],
                    ["2026-01-01", "1.0", "0", "0", "none"],
                    ["2026-01-02", "2.0", "1", "1", "spoof"],
                ],
            )
            config_path = tmp_path / "cluster.yaml"
            write_config(
                config_path,
                raw_dir,
                excluded_columns=["attack", "time", "attack_P1"],
                exclude_if_present=["AttackType"],
            )

            result = fit_preprocessor_for_training(config_path)

            self.assertEqual(result.preprocessor.output_feature_names, ("sensor",))
            self.assertEqual(result.transformed_training_matrix.shape, (2, 1))

    def test_config_rejects_label_not_present_in_excluded_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "sample.csv",
                [
                    ["time", "sensor", "attack"],
                    ["2026-01-01", "1.0", "0"],
                    ["2026-01-02", "2.0", "1"],
                ],
            )
            config_path = tmp_path / "cluster.yaml"
            write_config(
                config_path,
                raw_dir,
                excluded_columns=["time"],
                exclude_if_present=[],
            )

            with self.assertRaises(DatasetConfigError) as context:
                load_dataset_for_training(config_path)

            self.assertIn("label column 'attack' must appear in excluded_columns", str(context.exception))


if __name__ == "__main__":
    unittest.main()
