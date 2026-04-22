from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.build_ton_iot_combined import build_ton_iot_combined  # noqa: E402
from src.data.loaders import load_dataset_for_training  # noqa: E402
from src.data.schema_validation import DatasetSchemaError  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_cluster2_config(path: Path, raw_dir: Path, output_path: Path) -> None:
    config = {
        "config_version": 1,
        "cluster": {
            "id": 2,
            "key": "C2",
            "dataset_key": "TON_IoT_combined_telemetry",
            "dataset_name": "TON IoT combined telemetry",
            "audit_report": "outputs/reports/test_ton_iot.json",
        },
        "data": {
            "data_root_env_var": "FCFL_DATA_ROOT",
            "default_data_root": str(raw_dir.parent),
            "current_raw_input_dir": str(raw_dir),
            "current_raw_files": [
                "Train_Test_IoT_Fridge.csv",
                "Train_Test_IoT_GPS_Tracker.csv",
                "Train_Test_IoT_Garage_Door.csv",
            ],
            "training_input_mode": "combined_processed_csv_required",
            "training_input_glob": None,
            "training_input_path": None,
            "expected_processed_input_path": str(output_path),
            "label_column": "label",
            "label_column_confirmed_from_audit": True,
            "candidate_label_columns_present": ["label", "type"],
            "timestamp_or_order_columns": ["date", "time"],
            "excluded_columns": ["label", "type", "date", "time"],
            "exclude_if_present": ["source"],
        },
        "runtime_validation": {
            "require_training_input_to_exist": True,
            "require_label_column_to_exist": True,
            "error_on_missing_expected_processed_input": "TON_IOT_COMBINED_TELEMETRY_REQUIRED",
            "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


class TonIotCombinedSchemaTests(unittest.TestCase):
    def test_builder_creates_deterministic_combined_schema_and_training_loader_accepts_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()

            write_csv(
                raw_dir / "Train_Test_IoT_Fridge.csv",
                [
                    ["date", "time", "fridge_temperature", "temp_condition", "label", "type"],
                    ["25-Apr-19", " 19:19:40 ", "9", "high", "1", "ddos"],
                    ["25-Apr-19", " 19:19:45 ", "12.65", "low", "0", "normal"],
                ],
            )
            write_csv(
                raw_dir / "Train_Test_IoT_GPS_Tracker.csv",
                [
                    ["date", "time", "latitude", "longitude", "label", "type"],
                    ["25-Apr-19", " 18:31:39 ", "116.5", "132.1", "1", "ddos"],
                    ["25-Apr-19", " 18:31:41 ", "121.7", "135.0", "0", "normal"],
                ],
            )
            write_csv(
                raw_dir / "Train_Test_IoT_Garage_Door.csv",
                [
                    ["date", "time", "door_state", "sphone_signal", "label", "type"],
                    ["25-Apr-19", " 14:42:33 ", "closed", "0", "1", "ddos"],
                    ["25-Apr-19", " 14:42:38 ", "open", "true", "0", "normal"],
                ],
            )

            output_path = tmp_path / "combined.csv"
            report_path = tmp_path / "report.json"
            config_path = tmp_path / "cluster2.yaml"
            write_cluster2_config(config_path, raw_dir, output_path)

            summary = build_ton_iot_combined(config_path, report_path=report_path)

            self.assertEqual(
                summary["output_columns"],
                [
                    "date",
                    "time",
                    "source",
                    "fridge_temperature",
                    "temp_condition",
                    "latitude",
                    "longitude",
                    "door_state",
                    "sphone_signal",
                    "label",
                    "type",
                ],
            )
            self.assertEqual(summary["kept_feature_columns"], [
                "fridge_temperature",
                "temp_condition",
                "latitude",
                "longitude",
                "door_state",
                "sphone_signal",
            ])
            self.assertTrue(summary["pure_intersection_after_exclusions_would_be_empty"])

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["source"], "fridge")
            self.assertEqual(rows[0]["time"], "19:19:40")
            self.assertEqual(rows[0]["latitude"], "")
            self.assertEqual(rows[2]["source"], "gps_tracker")
            self.assertEqual(rows[2]["fridge_temperature"], "")
            self.assertEqual(rows[4]["source"], "garage_door")
            self.assertEqual(rows[4]["door_state"], "closed")
            self.assertEqual(rows[4]["temp_condition"], "")

            inspection = load_dataset_for_training(config_path)
            self.assertEqual(len(inspection.file_paths), 1)
            self.assertEqual(inspection.file_paths[0], output_path)
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

    def test_builder_fails_when_required_shared_columns_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()

            write_csv(
                raw_dir / "Train_Test_IoT_Fridge.csv",
                [
                    ["date", "time", "fridge_temperature", "label", "type"],
                    ["25-Apr-19", " 19:19:40 ", "9", "1", "ddos"],
                ],
            )
            write_csv(
                raw_dir / "Train_Test_IoT_GPS_Tracker.csv",
                [
                    ["date", "latitude", "longitude", "label", "type"],
                    ["25-Apr-19", "116.5", "132.1", "0", "normal"],
                ],
            )
            write_csv(
                raw_dir / "Train_Test_IoT_Garage_Door.csv",
                [
                    ["date", "time", "door_state", "label", "type"],
                    ["25-Apr-19", " 14:42:33 ", "closed", "1", "ddos"],
                ],
            )

            output_path = tmp_path / "combined.csv"
            config_path = tmp_path / "cluster2.yaml"
            write_cluster2_config(config_path, raw_dir, output_path)

            with self.assertRaises(DatasetSchemaError) as context:
                build_ton_iot_combined(config_path)

            self.assertIn("missing required shared columns", str(context.exception))


if __name__ == "__main__":
    unittest.main()
