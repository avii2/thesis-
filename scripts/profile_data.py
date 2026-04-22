#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import DEFAULT_CLUSTER_CONFIG_PATHS, load_dataset_for_profile  # noqa: E402
from src.data.schema_validation import DatasetInspection, resolve_training_file_paths  # noqa: E402


def build_profile_report(inspection: DatasetInspection) -> dict:
    config = inspection.config
    report = {
        "cluster_id": config.cluster_id,
        "dataset": config.dataset_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config.config_path),
        "source_kind": inspection.source_kind,
        "source_paths": [str(path) for path in inspection.file_paths],
        "csv_file_count": len(inspection.file_inspections),
        "total_rows": sum(file_inspection.row_count for file_inspection in inspection.file_inspections),
        "label_column": config.label_column,
        "label_column_confirmed_from_audit": config.label_column_confirmed_from_audit,
        "schema_consistent_across_files": inspection.schema_consistent_across_files,
        "union_columns": list(inspection.union_columns),
        "intersection_columns": list(inspection.intersection_columns),
        "timestamp_or_order_columns": list(config.timestamp_or_order_columns),
        "configured_excluded_columns": list(config.excluded_columns),
        "configured_exclude_if_present": list(config.exclude_if_present),
        "audit_blockers": list(inspection.audit_blockers),
        "files": [],
    }

    training_input_status = {"available": True}
    try:
        training_paths = resolve_training_file_paths(config)
    except Exception as exc:  # noqa: BLE001
        training_input_status = {"available": False, "error": str(exc)}
    else:
        training_input_status["paths"] = [str(path) for path in training_paths]
    report["training_input_status"] = training_input_status

    for file_inspection in inspection.file_inspections:
        report["files"].append(
            {
                "file_name": file_inspection.path.name,
                "relative_path": str(file_inspection.path),
                "row_count": file_inspection.row_count,
                "column_count": len(file_inspection.normalized_columns),
                "raw_columns": list(file_inspection.raw_columns),
                "normalized_columns": list(file_inspection.normalized_columns),
                "available_columns": list(file_inspection.normalized_columns),
                "missing_value_counts": dict(file_inspection.missing_value_counts),
                "numeric_columns": list(file_inspection.numeric_columns),
                "categorical_columns": list(file_inspection.categorical_columns),
                "present_excluded_columns": list(file_inspection.present_excluded_columns),
                "retained_feature_columns": list(file_inspection.retained_feature_columns),
                "label_raw_value_counts": dict(file_inspection.label_raw_value_counts),
                "binary_label_counts": dict(file_inspection.binary_label_counts),
                "header_issues": list(file_inspection.header_issues),
            }
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset profile reports from cluster configs.")
    parser.add_argument(
        "--config",
        dest="config_paths",
        action="append",
        help="Optional cluster config path. If omitted, profile all three default cluster configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = args.config_paths or [str(path) for path in DEFAULT_CLUSTER_CONFIG_PATHS.values()]

    for config_path in config_paths:
        inspection = load_dataset_for_profile(config_path)
        report = build_profile_report(inspection)
        output_path = REPO_ROOT / inspection.config.audit_report_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
