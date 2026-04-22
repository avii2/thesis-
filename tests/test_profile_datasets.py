from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import profile_datasets as profiler  # noqa: E402


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


class ProfileDatasetsTests(unittest.TestCase):
    def test_profile_csv_confirms_hai_attack_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / "hai.csv"
            write_csv(
                csv_path,
                [
                    ["time", "sensor", "attack", "attack_P1"],
                    ["2020-01-01 00:00:00", "1.0", "0", "0"],
                    ["2020-01-01 00:00:01", "2.0", "1", "1"],
                ],
            )

            config = profiler.ClusterConfig(
                cluster_id=1,
                dataset_name="HAI 21.03",
                source_paths=(str(tmp_path),),
                report_path=str(tmp_path / "report.json"),
                candidate_label_columns=("attack",),
                confirmed_label_column="attack",
                timestamp_columns=("time",),
                leakage_or_id_columns=("attack_P1",),
                expected_layout_note="test",
            )

            profile = profiler.profile_csv(csv_path, config)

            self.assertEqual(profile["row_count"], 2)
            self.assertEqual(profile["candidate_label_columns"], ["attack"])
            self.assertEqual(profile["timestamp_or_order_columns"], ["time"])
            self.assertEqual(profile["obvious_leakage_or_id_columns"], ["attack_P1"])
            self.assertEqual(profile["confirmed_label_mapped_counts"], {"0": 1, "1": 1})
            self.assertEqual(profile["candidate_retained_columns"], ["sensor"])

    def test_build_cluster_report_flags_schema_mismatch_for_ton(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            write_csv(
                tmp_path / "device_a.csv",
                [
                    ["date", "time", "temperature", "label", "type"],
                    ["2026-01-01", "00:00:00", "12", "0", "normal"],
                ],
            )
            write_csv(
                tmp_path / "device_b.csv",
                [
                    ["date", "time", "latitude", "label", "type"],
                    ["2026-01-01", "00:00:01", "10.0", "1", "ddos"],
                ],
            )

            config = profiler.ClusterConfig(
                cluster_id=2,
                dataset_name="TON IoT",
                source_paths=(str(tmp_path),),
                report_path=str(tmp_path / "report.json"),
                candidate_label_columns=("label", "type"),
                confirmed_label_column=None,
                timestamp_columns=("date", "time"),
                leakage_or_id_columns=("device",),
                expected_layout_note="test",
                combined_telemetry_required=True,
            )

            report = profiler.build_cluster_report(config)

            self.assertFalse(report["schema_consistent_across_files"])
            self.assertIn("CSV schemas differ across files in this cluster.", report["audit_blockers"])
            self.assertTrue(any("combined telemetry" in blocker.lower() for blocker in report["audit_blockers"]))
            self.assertEqual(report["candidate_label_columns_present"], ["label", "type"])


if __name__ == "__main__":
    unittest.main()
