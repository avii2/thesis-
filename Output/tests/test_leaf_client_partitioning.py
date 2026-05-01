from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.partitions import build_candidate_leaf_clients  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_config(
    path: Path,
    *,
    cluster_id: int,
    dataset_name: str,
    raw_dir: Path,
    raw_files: list[str],
    label_column: str,
    excluded_columns: list[str],
    timestamp_or_order_columns: list[str],
    training_input_mode: str = "raw_csv_glob",
    training_input_path: str | None = None,
    expected_processed_input_path: str | None = None,
    candidate_leaf_clients: int = 3,
    split_ratios: dict[str, float] | None = None,
) -> None:
    config = {
        "config_version": 1,
        "cluster": {
            "id": cluster_id,
            "key": f"C{cluster_id}",
            "dataset_key": f"TEST_CLUSTER_{cluster_id}",
            "dataset_name": dataset_name,
            "audit_report": f"outputs/reports/test_cluster_{cluster_id}.json",
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
            "timestamp_or_order_columns": timestamp_or_order_columns,
            "excluded_columns": excluded_columns,
            "exclude_if_present": [],
        },
        "partitioning": {
            "candidate_leaf_clients": candidate_leaf_clients,
        },
        "runtime_validation": {
            "require_training_input_to_exist": True,
            "require_label_column_to_exist": True,
            "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
            "error_on_missing_expected_processed_input": "EXPECTED_PROCESSED_INPUT_REQUIRED",
        },
    }
    if split_ratios is not None:
        config["partitioning"]["split_ratios"] = split_ratios
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


class LeafClientPartitioningTests(unittest.TestCase):
    def test_partitioner_sorts_by_timestamp_creates_contiguous_shards_and_logs_single_class_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "sample.csv",
                [
                    ["time", "sensor", "attack"],
                    ["2026-01-01 00:00:05", "5", "1"],
                    ["2026-01-01 00:00:01", "1", "0"],
                    ["2026-01-01 00:00:06", "6", "1"],
                    ["2026-01-01 00:00:02", "2", "0"],
                    ["2026-01-01 00:00:07", "7", "1"],
                    ["2026-01-01 00:00:03", "3", "0"],
                    ["2026-01-01 00:00:08", "8", "1"],
                    ["2026-01-01 00:00:04", "4", "0"],
                    ["2026-01-01 00:00:09", "9", "0"],
                    ["2026-01-01 00:00:10", "10", "1"],
                    ["2026-01-01 00:00:11", "11", "0"],
                    ["2026-01-01 00:00:12", "12", "1"],
                ],
            )
            config_path = tmp_path / "cluster.yaml"
            write_config(
                config_path,
                cluster_id=1,
                dataset_name="Sorted Partition Test",
                raw_dir=raw_dir,
                raw_files=["sample.csv"],
                label_column="attack",
                excluded_columns=["attack", "time"],
                timestamp_or_order_columns=["time"],
                candidate_leaf_clients=3,
                split_ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
            )

            output_path = tmp_path / "clients.json"
            result = build_candidate_leaf_clients(config_path, output_path=output_path)

            self.assertEqual(result.metadata.ordering_mode, "sorted_by_configured_timestamp_or_order_columns")
            self.assertEqual(result.metadata.ordering_columns_used, ("time",))
            self.assertEqual(result.ordered_row_indices.tolist(), [1, 3, 5, 7, 0, 2, 4, 6, 8, 9, 10, 11])
            self.assertEqual(result.metadata.num_leaf_clients, 3)
            self.assertEqual([client.num_total_samples for client in result.metadata.clients], [4, 4, 4])
            self.assertEqual([client.train.num_samples for client in result.metadata.clients], [2, 2, 2])
            self.assertEqual(result.metadata.clients[0].single_class_splits, ("train", "validation", "test"))
            self.assertTrue(any(item["client_id"] == "C1_L001" for item in result.metadata.single_class_local_partitions))

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["num_leaf_clients"], 3)
            self.assertEqual(saved["clients"][1]["ordered_start_index"], 4)
            self.assertEqual(saved["cluster_split_label_counts"]["train"], {"0": 3, "1": 3})

    def test_partitioner_preserves_file_order_when_no_timestamp_columns_are_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            write_csv(
                raw_dir / "part1.csv",
                [
                    ["sensor", "attack"],
                    ["10", "0"],
                    ["11", "1"],
                    ["12", "0"],
                ],
            )
            write_csv(
                raw_dir / "part2.csv",
                [
                    ["sensor", "attack"],
                    ["20", "1"],
                    ["21", "0"],
                    ["22", "1"],
                ],
            )
            config_path = tmp_path / "cluster.yaml"
            write_config(
                config_path,
                cluster_id=2,
                dataset_name="File Order Partition Test",
                raw_dir=raw_dir,
                raw_files=["part1.csv", "part2.csv"],
                label_column="attack",
                excluded_columns=["attack"],
                timestamp_or_order_columns=[],
                candidate_leaf_clients=2,
            )

            result = build_candidate_leaf_clients(config_path, output_path=tmp_path / "clients.json")

            self.assertEqual(result.metadata.ordering_mode, "preserve_file_order")
            self.assertEqual(result.ordered_row_indices.tolist(), [0, 1, 2, 3, 4, 5])
            self.assertEqual([client.num_total_samples for client in result.metadata.clients], [3, 3])
            self.assertEqual(result.metadata.clients[0].train.num_samples, 2)
            self.assertEqual(result.metadata.clients[0].validation.num_samples, 0)
            self.assertEqual(result.metadata.clients[0].test.num_samples, 1)
            self.assertEqual(result.metadata.cluster_split_label_counts["train"], {"0": 2, "1": 2})


if __name__ == "__main__":
    unittest.main()
