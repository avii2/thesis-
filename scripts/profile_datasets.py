#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}


@dataclass(frozen=True)
class ClusterConfig:
    cluster_id: int
    dataset_name: str
    source_paths: tuple[str, ...]
    report_path: str
    candidate_label_columns: tuple[str, ...]
    confirmed_label_column: str | None
    timestamp_columns: tuple[str, ...]
    leakage_or_id_columns: tuple[str, ...]
    expected_layout_note: str
    combined_telemetry_required: bool = False


CLUSTER_CONFIGS = (
    ClusterConfig(
        cluster_id=1,
        dataset_name="HAI 21.03",
        source_paths=("data/raw/hai_2103/hai-21.03",),
        report_path="outputs/reports/data_profile_cluster1.json",
        candidate_label_columns=("attack", "Attack", "label", "Label", "target", "Target"),
        confirmed_label_column="attack",
        timestamp_columns=("time", "Time", "timestamp", "Timestamp", "date", "Date"),
        leakage_or_id_columns=("attack_P1", "attack_P2", "attack_P3", "attack_P4"),
        expected_layout_note="Cluster 1 is expected to read the eight HAI 21.03 CSVs under data/raw/hai_2103/hai-21.03/.",
    ),
    ClusterConfig(
        cluster_id=2,
        dataset_name="TON IoT",
        source_paths=("data/raw/ ton_iot",),
        report_path="outputs/reports/data_profile_cluster2.json",
        candidate_label_columns=("label", "Label", "attack", "Attack", "target", "Target", "type", "Type"),
        confirmed_label_column=None,
        timestamp_columns=("date", "Date", "time", "Time", "timestamp", "Timestamp"),
        leakage_or_id_columns=("device", "Device", "device_id", "Device_ID", "source", "Source", "src", "dst", "id", "ID"),
        expected_layout_note=(
            "Cluster 2 currently has only per-device CSVs under data/raw/ ton_iot/. "
            "No combined_telemetry directory exists in the audited local repo."
        ),
        combined_telemetry_required=True,
    ),
    ClusterConfig(
        cluster_id=3,
        dataset_name="WUSTL-IIOT-2021",
        source_paths=("data/raw/ wustl_iiot_2021",),
        report_path="outputs/reports/data_profile_cluster3.json",
        candidate_label_columns=("label", "Label", "class", "Class", "target", "Target", "traffic", "Traffic", "attack", "Attack"),
        confirmed_label_column=None,
        timestamp_columns=("StartTime", "LastTime", "time", "Time", "timestamp", "Timestamp", "date", "Date"),
        leakage_or_id_columns=("StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId", "attack_type", "AttackType", "traffic_class", "TrafficClass"),
        expected_layout_note="Cluster 3 currently has one CSV under data/raw/ wustl_iiot_2021/.",
    ),
)


def normalize_header(name: str) -> str:
    return name.lstrip("\ufeff")


def normalize_value(value: str) -> str:
    return value.strip()


def is_missing(value: str) -> bool:
    return normalize_value(value).lower() in MISSING_TOKENS


def is_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def label_value_to_binary(value: str) -> str | None:
    normalized = normalize_value(value)
    lowered = normalized.lower()
    if lowered in {"0", "normal", "benign", "false"}:
        return "0"
    if lowered in {"1", "attack", "malicious", "anomaly", "true"}:
        return "1"
    return None


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def load_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def profile_csv(path: Path, config: ClusterConfig) -> dict:
    raw_columns = load_csv_header(path)
    normalized_columns = [normalize_header(column) for column in raw_columns]
    candidate_label_lookup = {normalize_header(column).lower() for column in config.candidate_label_columns}
    timestamp_lookup = {normalize_header(column).lower() for column in config.timestamp_columns}
    leakage_lookup = {normalize_header(column).lower() for column in config.leakage_or_id_columns}

    candidate_label_columns = [
        column for column in normalized_columns if normalize_header(column).lower() in candidate_label_lookup
    ]
    timestamp_columns = [
        column for column in normalized_columns if normalize_header(column).lower() in timestamp_lookup
    ]
    leakage_columns = [
        column for column in normalized_columns if normalize_header(column).lower() in leakage_lookup
    ]

    missing_counts = {column: 0 for column in normalized_columns}
    numeric_flags = {column: True for column in normalized_columns}
    non_missing_counts = {column: 0 for column in normalized_columns}
    candidate_value_counts = {column: Counter() for column in candidate_label_columns}
    header_issues: list[str] = []
    row_count = 0
    row_length_mismatches = 0

    if raw_columns and raw_columns[0].startswith("\ufeff"):
        header_issues.append("UTF-8 BOM detected on the first header cell.")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            row_count += 1
            if len(row) != len(normalized_columns):
                row_length_mismatches += 1
                if len(row) < len(normalized_columns):
                    row = row + [""] * (len(normalized_columns) - len(row))
                else:
                    row = row[: len(normalized_columns)]

            for column, value in zip(normalized_columns, row):
                normalized_value = normalize_value(value)
                if is_missing(normalized_value):
                    missing_counts[column] += 1
                    continue
                non_missing_counts[column] += 1
                if numeric_flags[column] and not is_numeric(normalized_value):
                    numeric_flags[column] = False
                if column in candidate_value_counts:
                    candidate_value_counts[column][normalized_value] += 1

    if row_length_mismatches:
        header_issues.append(f"{row_length_mismatches} row(s) had a column-count mismatch and were padded or truncated during profiling.")

    numeric_columns = [column for column in normalized_columns if numeric_flags[column] and non_missing_counts[column] > 0]
    categorical_columns = [column for column in normalized_columns if column not in numeric_columns]
    recommended_excluded_columns = sorted(set(candidate_label_columns + timestamp_columns + leakage_columns))
    candidate_retained_columns = [column for column in normalized_columns if column not in recommended_excluded_columns]

    profile = {
        "file_name": path.name,
        "relative_path": str(path),
        "row_count": row_count,
        "column_count": len(normalized_columns),
        "raw_columns": raw_columns,
        "normalized_columns": normalized_columns,
        "candidate_label_columns": candidate_label_columns,
        "timestamp_or_order_columns": timestamp_columns,
        "obvious_leakage_or_id_columns": leakage_columns,
        "missing_value_counts": missing_counts,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "candidate_label_value_counts": {column: sorted_counter(counter) for column, counter in candidate_value_counts.items()},
        "recommended_excluded_columns": recommended_excluded_columns,
        "candidate_retained_columns": candidate_retained_columns,
        "header_issues": header_issues,
    }

    if config.confirmed_label_column and config.confirmed_label_column in candidate_value_counts:
        mapped_counts = Counter()
        unmapped_values = Counter()
        for raw_value, count in candidate_value_counts[config.confirmed_label_column].items():
            mapped = label_value_to_binary(raw_value)
            if mapped is None:
                unmapped_values[raw_value] += count
            else:
                mapped_counts[mapped] += count
        profile["confirmed_label_column"] = config.confirmed_label_column
        profile["confirmed_label_mapped_counts"] = sorted_counter(mapped_counts)
        if unmapped_values:
            profile["confirmed_label_unmapped_values"] = sorted_counter(unmapped_values)

    return profile


def expand_csv_paths(source_paths: Iterable[str]) -> list[Path]:
    csv_paths: list[Path] = []
    for source in source_paths:
        base = Path(source)
        if base.is_file() and base.suffix.lower() == ".csv":
            csv_paths.append(base)
            continue
        if base.is_dir():
            csv_paths.extend(sorted(path for path in base.glob("*.csv") if path.is_file()))
    return sorted(csv_paths)


def build_cluster_report(config: ClusterConfig) -> dict:
    csv_paths = expand_csv_paths(config.source_paths)
    file_profiles = [profile_csv(path, config) for path in csv_paths]

    normalized_column_sets = [tuple(profile["normalized_columns"]) for profile in file_profiles]
    unique_schemas = sorted({schema for schema in normalized_column_sets})
    schema_consistent = len(unique_schemas) <= 1

    union_columns = sorted({column for profile in file_profiles for column in profile["normalized_columns"]})
    if file_profiles:
        intersection = set(file_profiles[0]["normalized_columns"])
        for profile in file_profiles[1:]:
            intersection &= set(profile["normalized_columns"])
        intersection_columns = sorted(intersection)
    else:
        intersection_columns = []

    candidate_label_columns = sorted({column for profile in file_profiles for column in profile["candidate_label_columns"]})
    timestamp_columns = sorted({column for profile in file_profiles for column in profile["timestamp_or_order_columns"]})
    leakage_columns = sorted({column for profile in file_profiles for column in profile["obvious_leakage_or_id_columns"]})

    blockers: list[str] = []
    notes: list[str] = [config.expected_layout_note]
    source_path_notes = [source for source in config.source_paths if "/ " in source]
    if source_path_notes:
        notes.append(
            "One or more dataset directories include a leading space after data/raw/. "
            f"Observed paths: {', '.join(source_path_notes)}"
        )
    if not csv_paths:
        blockers.append("No CSV files were found for this cluster.")
    if not schema_consistent:
        blockers.append("CSV schemas differ across files in this cluster.")
    if config.combined_telemetry_required and csv_paths:
        blockers.append(
            "The combined telemetry training table is not present. A deterministic build step is required before Cluster 2 training."
        )
    if config.confirmed_label_column:
        missing_confirmed_label = [
            profile["file_name"]
            for profile in file_profiles
            if config.confirmed_label_column not in profile["candidate_label_columns"]
        ]
        if missing_confirmed_label:
            blockers.append(
                f"The expected label column {config.confirmed_label_column!r} is missing from: {', '.join(missing_confirmed_label)}"
            )
            label_status = "missing_configured_label"
            confirmed_label = None
            confirmed_label_counts = {}
        else:
            label_status = "confirmed_from_schema"
            confirmed_label = config.confirmed_label_column
            confirmed_label_counts_counter = Counter()
            for profile in file_profiles:
                confirmed_label_counts_counter.update(profile.get("confirmed_label_mapped_counts", {}))
            confirmed_label_counts = sorted_counter(confirmed_label_counts_counter)
    else:
        label_status = "requires_user_confirmation"
        confirmed_label = None
        confirmed_label_counts = {}

    cluster_candidate_value_counts = {
        column: Counter() for column in candidate_label_columns
    }
    for profile in file_profiles:
        for column, counts in profile["candidate_label_value_counts"].items():
            cluster_candidate_value_counts[column].update(counts)

    return {
        "cluster_id": config.cluster_id,
        "dataset": config.dataset_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "missing_value_policy": "Counts treat blank strings and the tokens na, n/a, nan, null, none (case-insensitive) as missing.",
        "source_paths": list(config.source_paths),
        "csv_file_count": len(csv_paths),
        "total_rows": sum(profile["row_count"] for profile in file_profiles),
        "label_status": label_status,
        "confirmed_label_column": confirmed_label,
        "confirmed_label_counts": confirmed_label_counts,
        "schema_consistent_across_files": schema_consistent,
        "schema_variants": [list(schema) for schema in unique_schemas],
        "union_columns": union_columns,
        "intersection_columns": intersection_columns,
        "candidate_label_columns_present": candidate_label_columns,
        "timestamp_or_order_columns_present": timestamp_columns,
        "obvious_leakage_or_id_columns_present": leakage_columns,
        "cluster_candidate_label_value_counts": {
            column: sorted_counter(counter) for column, counter in cluster_candidate_value_counts.items()
        },
        "audit_notes": notes,
        "audit_blockers": blockers,
        "files": file_profiles,
    }


def build_audit_note(cluster_reports: list[dict]) -> str:
    lines = [
        "# Dataset Audit Reality Check",
        "",
        f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Confirmed",
        "",
    ]
    for report in cluster_reports:
        lines.append(
            f"- Cluster {report['cluster_id']} ({report['dataset']}): {report['csv_file_count']} CSV file(s), "
            f"{report['total_rows']} row(s), schema_consistent={str(report['schema_consistent_across_files']).lower()}."
        )
        if report["confirmed_label_column"]:
            lines.append(
                f"- Cluster {report['cluster_id']} confirmed label column: `{report['confirmed_label_column']}` "
                f"with mapped counts {report['confirmed_label_counts']}."
            )
    lines.extend(
        [
            "",
            "## Requires User Confirmation / Follow-Up",
            "",
        ]
    )
    for report in cluster_reports:
        if report["audit_blockers"]:
            for blocker in report["audit_blockers"]:
                lines.append(f"- Cluster {report['cluster_id']}: {blocker}")
        if report["label_status"] == "requires_user_confirmation":
            lines.append(
                f"- Cluster {report['cluster_id']}: label column still requires user confirmation. "
                f"Observed candidates: {', '.join(report['candidate_label_columns_present'])}."
            )
    lines.extend(
        [
            "",
            "## Local Layout Notes",
            "",
            "- The audited repo uses repo-local data under `data/raw/`, not an external `desktop/thesis/data/` root.",
            "- The current local TON and WUSTL directories have leading spaces in their names: `data/raw/ ton_iot/` and `data/raw/ wustl_iiot_2021/`.",
            "- Cluster 2 does not yet have a `combined_telemetry/` directory or a unified combined telemetry CSV in the audited local repo.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    cluster_reports = [build_cluster_report(config) for config in CLUSTER_CONFIGS]
    report_paths = [Path(config.report_path) for config in CLUSTER_CONFIGS]
    for path in report_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    for report, path in zip(cluster_reports, report_paths, strict=True):
        path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    audit_note_path = Path("outputs/reports/reality_check_audit.md")
    audit_note_path.write_text(build_audit_note(cluster_reports), encoding="utf-8")


if __name__ == "__main__":
    main()
