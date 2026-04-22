from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_dataset_config, load_dataset_for_profile, load_dataset_for_training  # noqa: E402
from src.data.schema_validation import DatasetSchemaError, resolve_configured_path  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_cluster_config(
    path: Path,
    *,
    raw_dir: Path,
    raw_files: list[str],
    label_column: str = "attack",
    excluded_columns: list[str] | None = None,
    exclude_if_present: list[str] | None = None,
    timestamp_or_order_columns: list[str] | None = None,
    training_input_mode: str = "raw_csv_glob",
    training_input_path: str | None = None,
    expected_processed_input_path: str | None = None,
) -> None:
    config = {
        "config_version": 1,
        "cluster": {
            "id": 99,
            "key": "T99",
            "dataset_key": "TEST_DATASET",
            "dataset_name": "Test Dataset",
            "audit_report": "outputs/reports/test_dataset.json",
        },
        "data": {
            "data_root_env_var": "FCFL_DATA_ROOT",
            "default_data_root": str(raw_dir.parent),
            "current_raw_input_dir": str(raw_dir),
            "current_raw_files": raw_files,
            "training_input_mode": training_input_mode,
            "training_input_glob": None,
            "training_input_path": training_input_path,
            "expected_processed_input_path": expected_processed_input_path,
            "label_column": label_column,
            "label_column_confirmed_from_audit": True,
            "candidate_label_columns_present": [label_column],
            "timestamp_or_order_columns": timestamp_or_order_columns or [],
            "excluded_columns": excluded_columns or [label_column],
            "exclude_if_present": exclude_if_present or [],
        },
        "runtime_validation": {
            "require_training_input_to_exist": True,
            "require_label_column_to_exist": True,
            "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
            "error_on_missing_expected_processed_input": "EXPECTED_PROCESSED_INPUT_REQUIRED",
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


class DataContractTests(unittest.TestCase):
    def test_profile_loader_uses_only_configured_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()

            write_csv(
                raw_dir / "configured.csv",
                [
                    ["time", "sensor", "attack"],
                    ["2026-01-01", "1.0", "0"],
                    ["2026-01-02", "2.0", "1"],
                ],
            )
            write_csv(
                raw_dir / "extra.csv",
                [
                    ["time", "sensor", "attack"],
                    ["2026-01-01", "9.0", "0"],
                    ["2026-01-02", "8.0", "1"],
                ],
            )

            config_path = tmp_path / "cluster.yaml"
            write_cluster_config(
                config_path,
                raw_dir=raw_dir,
                raw_files=["configured.csv"],
                excluded_columns=["attack", "time"],
                timestamp_or_order_columns=["time"],
            )

            inspection = load_dataset_for_profile(config_path)

            self.assertEqual(len(inspection.file_inspections), 1)
            self.assertEqual(inspection.file_inspections[0].path.name, "configured.csv")

    def test_training_loader_reports_available_columns_when_label_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()

            write_csv(
                raw_dir / "sample.csv",
                [
                    ["time", "sensor", "wrong_label"],
                    ["2026-01-01", "1.0", "0"],
                    ["2026-01-02", "2.0", "1"],
                ],
            )

            config_path = tmp_path / "cluster.yaml"
            write_cluster_config(
                config_path,
                raw_dir=raw_dir,
                raw_files=["sample.csv"],
                label_column="attack",
                excluded_columns=["attack", "time"],
                timestamp_or_order_columns=["time"],
            )

            with self.assertRaises(DatasetSchemaError) as context:
                load_dataset_for_training(config_path)

            message = str(context.exception)
            self.assertIn("Available columns: time, sensor, wrong_label", message)
            self.assertIn("Refusing to guess alternative label names.", message)

    def test_cluster2_training_loader_uses_configured_combined_file_when_present(self) -> None:
        config_path = REPO_ROOT / "configs/cluster2_ton_iot.yaml"
        config = load_dataset_config(config_path)
        combined_path = resolve_configured_path(config, config.expected_processed_input_path or "")
        if not combined_path.exists():
            self.skipTest(f"Cluster 2 combined telemetry CSV is not built yet: {combined_path}")

        inspection = load_dataset_for_training(config_path)

        self.assertEqual(inspection.file_paths, (combined_path,))
        self.assertEqual(
            inspection.file_inspections[0].retained_feature_columns,
            (
                "fridge_temperature",
                "temp_condition",
                "latitude",
                "longitude",
                "door_state",
                "sphone_signal",
            ),
        )


if __name__ == "__main__":
    unittest.main()
