from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .loaders import DEFAULT_CLUSTER_CONFIG_PATHS, load_dataset_config, load_dataset_for_profile
from .schema_validation import DatasetConfig, DatasetSchemaError, normalize_header, normalize_value, resolve_configured_path


REQUIRED_SHARED_COLUMNS = ("date", "time", "label", "type")
SOURCE_COLUMN = "source"
DEFAULT_REPORT_PATH = Path("outputs/reports/ton_iot_combined_schema.json")


@dataclass(frozen=True)
class TonIotSourceSchema:
    path: Path
    source_name: str
    row_count: int
    raw_columns: tuple[str, ...]
    normalized_columns: tuple[str, ...]
    telemetry_columns: tuple[str, ...]


def _derive_source_name(path: Path) -> str:
    stem = path.stem
    prefix = "Train_Test_IoT_"
    if not stem.startswith(prefix):
        raise DatasetSchemaError(
            f"TON IoT combined telemetry builder expected filenames starting with {prefix!r}. "
            f"Observed: {path.name}"
        )
    return stem[len(prefix) :].lower()


def _validate_cluster2_config(config: DatasetConfig) -> None:
    if config.cluster_id != 2:
        raise DatasetSchemaError(
            f"TON IoT combined telemetry builder expects the Cluster 2 config. "
            f"Observed cluster_id={config.cluster_id} for {config.dataset_name!r}."
        )

    if not config.expected_processed_input_path:
        raise DatasetSchemaError(
            f"{config.dataset_name}: expected_processed_input_path must be configured for the combined builder."
        )


def inspect_ton_iot_raw_schemas(
    config_path: str | Path = DEFAULT_CLUSTER_CONFIG_PATHS[2],
) -> tuple[DatasetConfig, tuple[TonIotSourceSchema, ...], tuple[str, ...]]:
    config = load_dataset_config(config_path)
    _validate_cluster2_config(config)
    inspection = load_dataset_for_profile(config_path)

    intersection_lookup = set(inspection.intersection_columns)
    missing_shared = [column for column in REQUIRED_SHARED_COLUMNS if column not in intersection_lookup]
    if missing_shared:
        raise DatasetSchemaError(
            f"{config.dataset_name}: raw TON IoT files are missing required shared columns "
            f"{missing_shared}. Observed intersection: {list(inspection.intersection_columns)}."
        )

    telemetry_union: list[str] = []
    source_schemas: list[TonIotSourceSchema] = []

    for file_inspection in inspection.file_inspections:
        telemetry_columns = tuple(
            column
            for column in file_inspection.normalized_columns
            if column not in REQUIRED_SHARED_COLUMNS
        )
        if not telemetry_columns:
            raise DatasetSchemaError(
                f"{config.dataset_name}: {file_inspection.path.name} has no device-specific telemetry columns "
                "after removing shared metadata columns."
            )

        for column in telemetry_columns:
            if column not in telemetry_union:
                telemetry_union.append(column)

        source_schemas.append(
            TonIotSourceSchema(
                path=file_inspection.path,
                source_name=_derive_source_name(file_inspection.path),
                row_count=file_inspection.row_count,
                raw_columns=file_inspection.raw_columns,
                normalized_columns=file_inspection.normalized_columns,
                telemetry_columns=telemetry_columns,
            )
        )

    return config, tuple(source_schemas), tuple(telemetry_union)


def _resolve_output_path(config: DatasetConfig, output_path: str | Path | None) -> Path:
    if output_path is not None:
        return Path(output_path)
    return resolve_configured_path(config, config.expected_processed_input_path or "")


def _normalized_row_lookup(fieldnames: Sequence[str]) -> dict[str, str]:
    return {normalize_header(fieldname): fieldname for fieldname in fieldnames}


def _build_summary(
    *,
    config: DatasetConfig,
    source_schemas: Sequence[TonIotSourceSchema],
    telemetry_union: Sequence[str],
    output_path: Path,
    output_columns: Sequence[str],
    total_rows: int,
) -> dict[str, Any]:
    source_names = [source.source_name for source in source_schemas]
    harmonized_columns = []
    for column in telemetry_union:
        present_in = [source.source_name for source in source_schemas if column in source.telemetry_columns]
        blank_for = [name for name in source_names if name not in present_in]
        harmonized_columns.append(
            {
                "column": column,
                "present_in_sources": present_in,
                "blank_for_sources": blank_for,
                "fill_value_when_not_applicable": "",
            }
        )

    summary = {
        "dataset": config.dataset_name,
        "status": "ok",
        "build_strategy": "deterministic_explicit_union_with_blank_fill_for_non_applicable_device_fields",
        "config_path": str(config.config_path),
        "input_directory": config.current_raw_input_dir,
        "input_files": [
            {
                "path": str(source.path),
                "source_name": source.source_name,
                "row_count": source.row_count,
                "raw_columns": list(source.raw_columns),
                "telemetry_columns": list(source.telemetry_columns),
            }
            for source in source_schemas
        ],
        "required_shared_columns": list(REQUIRED_SHARED_COLUMNS),
        "kept_feature_columns": list(telemetry_union),
        "metadata_columns_retained": ["date", "time", SOURCE_COLUMN, "label", "type"],
        "source_column_added": SOURCE_COLUMN,
        "dropped_columns": [],
        "harmonized_columns": harmonized_columns,
        "pure_intersection_after_exclusions_would_be_empty": True,
        "output_columns": list(output_columns),
        "output_path": str(output_path),
        "output_row_count": total_rows,
    }
    return summary


def build_ton_iot_combined(
    config_path: str | Path = DEFAULT_CLUSTER_CONFIG_PATHS[2],
    *,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> Mapping[str, Any]:
    config, source_schemas, telemetry_union = inspect_ton_iot_raw_schemas(config_path)
    final_output_path = _resolve_output_path(config, output_path)
    final_report_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_report_path.parent.mkdir(parents=True, exist_ok=True)

    output_columns = ("date", "time", SOURCE_COLUMN) + telemetry_union + ("label", "type")
    total_rows = 0

    with final_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()

        for source in source_schemas:
            with source.path.open("r", encoding="utf-8-sig", newline="") as source_handle:
                reader = csv.DictReader(source_handle)
                if reader.fieldnames is None:
                    raise DatasetSchemaError(f"{config.dataset_name}: empty source CSV file: {source.path}")

                row_lookup = _normalized_row_lookup(reader.fieldnames)
                missing_columns = [
                    column
                    for column in REQUIRED_SHARED_COLUMNS + source.telemetry_columns
                    if column not in row_lookup
                ]
                if missing_columns:
                    raise DatasetSchemaError(
                        f"{config.dataset_name}: {source.path.name} is missing expected column(s) during build: "
                        f"{missing_columns}. Available columns: {list(row_lookup)}."
                    )

                for row in reader:
                    combined_row = {column: "" for column in output_columns}
                    combined_row["date"] = normalize_value(row[row_lookup["date"]])
                    combined_row["time"] = normalize_value(row[row_lookup["time"]])
                    combined_row[SOURCE_COLUMN] = source.source_name
                    combined_row["label"] = normalize_value(row[row_lookup["label"]])
                    combined_row["type"] = normalize_value(row[row_lookup["type"]])

                    for column in source.telemetry_columns:
                        combined_row[column] = normalize_value(row[row_lookup[column]])

                    writer.writerow(combined_row)
                    total_rows += 1

    summary = _build_summary(
        config=config,
        source_schemas=source_schemas,
        telemetry_union=telemetry_union,
        output_path=final_output_path,
        output_columns=output_columns,
        total_rows=total_rows,
    )
    final_report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic TON IoT combined telemetry CSV.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CLUSTER_CONFIG_PATHS[2]),
        help="Cluster 2 config path. Defaults to configs/cluster2_ton_iot.yaml.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional combined CSV output path. Defaults to the Cluster 2 config expected_processed_input_path.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="Optional schema report JSON path. Defaults to outputs/reports/ton_iot_combined_schema.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_ton_iot_combined(
        args.config,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
