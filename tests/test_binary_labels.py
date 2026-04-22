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
from src.data.schema_validation import DatasetSchemaError  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_config(path: Path, raw_dir: Path) -> None:
    config = {
        "config_version": 1,
        "cluster": {
            "id": 88,
            "key": "T88",
            "dataset_key": "TEST_BINARY",
            "dataset_name": "Binary Label Test",
            "audit_report": "outputs/reports/test_binary.json",
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
            "timestamp_or_order_columns": [],
            "excluded_columns": ["attack"],
            "exclude_if_present": [],
        },
        "runtime_validation": {
            "require_training_input_to_exist": True,
            "require_label_column_to_exist": True,
            "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


class BinaryLabelTests(unittest.TestCase):
    def test_loader_maps_binary_string_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "sample.csv",
                [
                    ["sensor", "attack"],
                    ["1.0", "normal"],
                    ["2.0", "attack"],
                ],
            )
            config_path = tmp_path / "cluster.yaml"
            write_config(config_path, raw_dir)

            inspection = load_dataset_for_training(config_path)

            self.assertEqual(inspection.file_inspections[0].binary_label_counts, {"0": 1, "1": 1})

    def test_loader_rejects_unknown_label_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "sample.csv",
                [
                    ["sensor", "attack"],
                    ["1.0", "normal"],
                    ["2.0", "suspicious"],
                ],
            )
            config_path = tmp_path / "cluster.yaml"
            write_config(config_path, raw_dir)

            with self.assertRaises(DatasetSchemaError) as context:
                load_dataset_for_training(config_path)

            message = str(context.exception)
            self.assertIn("Observed raw values: normal, suspicious", message)
            self.assertIn("Unknown values after binary mapping: suspicious", message)

    def test_loader_rejects_single_class_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "sample.csv",
                [
                    ["sensor", "attack"],
                    ["1.0", "0"],
                    ["2.0", "0"],
                ],
            )
            config_path = tmp_path / "cluster.yaml"
            write_config(config_path, raw_dir)

            with self.assertRaises(DatasetSchemaError) as context:
                load_dataset_for_training(config_path)

            self.assertIn("must contain both binary classes {0,1}", str(context.exception))


if __name__ == "__main__":
    unittest.main()
