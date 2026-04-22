from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}


class DatasetConfigError(ValueError):
    """Raised when a dataset config is missing required fields or is internally inconsistent."""


class DatasetSchemaError(RuntimeError):
    """Raised when a dataset file or schema violates the data contract."""


@dataclass(frozen=True)
class DatasetConfig:
    config_path: Path
    cluster_id: int
    cluster_key: str
    dataset_key: str
    dataset_name: str
    audit_report_path: str
    data_root_env_var: str
    default_data_root: str
    current_raw_input_dir: str
    current_raw_files: tuple[str, ...]
    training_input_mode: str
    training_input_glob: str | None
    training_input_path: str | None
    expected_processed_input_path: str | None
    label_column: str
    label_column_confirmed_from_audit: bool
    candidate_label_columns_present: tuple[str, ...]
    timestamp_or_order_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    exclude_if_present: tuple[str, ...]
    runtime_validation: Mapping[str, Any]


@dataclass(frozen=True)
class CsvInspection:
    path: Path
    raw_columns: tuple[str, ...]
    normalized_columns: tuple[str, ...]
    row_count: int
    missing_value_counts: Mapping[str, int]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    present_excluded_columns: tuple[str, ...]
    retained_feature_columns: tuple[str, ...]
    label_raw_value_counts: Mapping[str, int]
    binary_label_counts: Mapping[str, int]
    header_issues: tuple[str, ...]


@dataclass(frozen=True)
class DatasetInspection:
    config: DatasetConfig
    source_kind: str
    file_inspections: tuple[CsvInspection, ...]
    file_paths: tuple[Path, ...]
    schema_consistent_across_files: bool
    union_columns: tuple[str, ...]
    intersection_columns: tuple[str, ...]
    audit_blockers: tuple[str, ...]


def normalize_header(value: str) -> str:
    return value.lstrip("\ufeff")


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
    if lowered in {"0", "normal", "benign"}:
        return "0"
    if lowered in {"1", "attack", "malicious", "anomaly"}:
        return "1"

    try:
        numeric_value = float(normalized)
    except ValueError:
        return None

    if numeric_value == 0.0:
        return "0"
    if numeric_value == 1.0:
        return "1"
    return None


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise DatasetConfigError(f"Missing or invalid mapping for {key!r}.")
    return value


def _require_string(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigError(f"Missing or invalid string for {key!r}.")
    return value


def _optional_string(parent: Mapping[str, Any], key: str) -> str | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigError(f"Invalid optional string for {key!r}.")
    return value


def _require_bool(parent: Mapping[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise DatasetConfigError(f"Missing or invalid boolean for {key!r}.")
    return value


def _require_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int):
        raise DatasetConfigError(f"Missing or invalid integer for {key!r}.")
    return value


def _require_string_list(parent: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    value = parent.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DatasetConfigError(f"Missing or invalid list for {key!r}.")

    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DatasetConfigError(f"Invalid string item in list {key!r}.")
        items.append(item)

    if not allow_empty and not items:
        raise DatasetConfigError(f"List {key!r} must not be empty.")

    return tuple(items)


def load_cluster_config(config_path: str | Path) -> DatasetConfig:
    path = Path(config_path)
    if not path.exists():
        raise DatasetConfigError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise DatasetConfigError(f"Config file must contain a mapping at top level: {path}")

    cluster = _require_mapping(data, "cluster")
    dataset = _require_mapping(data, "data")
    runtime_validation = _require_mapping(data, "runtime_validation")

    config = DatasetConfig(
        config_path=path,
        cluster_id=_require_int(cluster, "id"),
        cluster_key=_require_string(cluster, "key"),
        dataset_key=_require_string(cluster, "dataset_key"),
        dataset_name=_require_string(cluster, "dataset_name"),
        audit_report_path=_require_string(cluster, "audit_report"),
        data_root_env_var=_require_string(dataset, "data_root_env_var"),
        default_data_root=_require_string(dataset, "default_data_root"),
        current_raw_input_dir=_require_string(dataset, "current_raw_input_dir"),
        current_raw_files=_require_string_list(dataset, "current_raw_files"),
        training_input_mode=_require_string(dataset, "training_input_mode"),
        training_input_glob=_optional_string(dataset, "training_input_glob"),
        training_input_path=_optional_string(dataset, "training_input_path"),
        expected_processed_input_path=_optional_string(dataset, "expected_processed_input_path"),
        label_column=_require_string(dataset, "label_column"),
        label_column_confirmed_from_audit=_require_bool(dataset, "label_column_confirmed_from_audit"),
        candidate_label_columns_present=_require_string_list(dataset, "candidate_label_columns_present"),
        timestamp_or_order_columns=_require_string_list(dataset, "timestamp_or_order_columns", allow_empty=True),
        excluded_columns=_require_string_list(dataset, "excluded_columns"),
        exclude_if_present=_require_string_list(dataset, "exclude_if_present", allow_empty=True),
        runtime_validation=runtime_validation,
    )

    validate_exclusion_configuration(config)
    validate_training_mode_configuration(config)
    return config


def validate_exclusion_configuration(config: DatasetConfig) -> None:
    excluded = set(config.excluded_columns)
    optional = set(config.exclude_if_present)

    if config.label_column not in excluded and config.label_column not in optional:
        raise DatasetConfigError(
            f"{config.dataset_name}: label column {config.label_column!r} must appear in excluded_columns "
            "or exclude_if_present."
        )

    for column in config.timestamp_or_order_columns:
        if column not in excluded and column not in optional:
            raise DatasetConfigError(
                f"{config.dataset_name}: timestamp/order column {column!r} must appear in excluded_columns "
                "or exclude_if_present."
            )


def validate_training_mode_configuration(config: DatasetConfig) -> None:
    valid_modes = {"raw_csv_glob", "single_csv", "combined_processed_csv_required"}
    if config.training_input_mode not in valid_modes:
        raise DatasetConfigError(
            f"{config.dataset_name}: unsupported training_input_mode {config.training_input_mode!r}. "
            f"Expected one of: {', '.join(sorted(valid_modes))}."
        )

    if config.training_input_mode == "raw_csv_glob" and not (config.training_input_glob or config.current_raw_files):
        raise DatasetConfigError(
            f"{config.dataset_name}: raw_csv_glob mode requires training_input_glob or current_raw_files."
        )

    if config.training_input_mode == "single_csv" and not config.training_input_path:
        raise DatasetConfigError(f"{config.dataset_name}: single_csv mode requires training_input_path.")

    if (
        config.training_input_mode == "combined_processed_csv_required"
        and not config.expected_processed_input_path
    ):
        raise DatasetConfigError(
            f"{config.dataset_name}: combined_processed_csv_required mode requires expected_processed_input_path."
        )


def get_data_root(config: DatasetConfig) -> Path:
    return Path(os.getenv(config.data_root_env_var, config.default_data_root))


def resolve_configured_path(config: DatasetConfig, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path

    default_root = Path(config.default_data_root)
    actual_root = get_data_root(config)

    if path == default_root:
        return actual_root

    if path.parts[: len(default_root.parts)] == default_root.parts:
        relative_parts = path.parts[len(default_root.parts) :]
        return actual_root.joinpath(*relative_parts)

    return path


def resolve_profile_file_paths(config: DatasetConfig) -> list[Path]:
    base_dir = resolve_configured_path(config, config.current_raw_input_dir)
    if not base_dir.exists():
        raise DatasetSchemaError(
            f"{config.dataset_name}: configured raw input directory does not exist: {base_dir}"
        )

    file_paths: list[Path] = []
    for filename in config.current_raw_files:
        candidate = base_dir / filename
        if not candidate.exists():
            raise DatasetSchemaError(
                f"{config.dataset_name}: configured raw file is missing: {candidate}"
            )
        if candidate.suffix.lower() != ".csv":
            raise DatasetSchemaError(
                f"{config.dataset_name}: configured raw file is not a CSV: {candidate}"
            )
        file_paths.append(candidate)
    return file_paths


def resolve_training_file_paths(config: DatasetConfig) -> list[Path]:
    if config.training_input_mode == "raw_csv_glob":
        if config.current_raw_files:
            return resolve_profile_file_paths(config)

        if not config.training_input_glob:
            raise DatasetConfigError(
                f"{config.dataset_name}: training_input_glob is required for raw_csv_glob mode."
            )
        pattern = resolve_configured_path(config, config.training_input_glob)
        matched = sorted(pattern.parent.glob(pattern.name))
        if not matched:
            raise DatasetSchemaError(
                f"{config.dataset_name}: no files matched configured training glob: {pattern}"
            )
        return matched

    if config.training_input_mode == "single_csv":
        assert config.training_input_path is not None
        path = resolve_configured_path(config, config.training_input_path)
        if not path.exists():
            raise DatasetSchemaError(f"{config.dataset_name}: configured training file is missing: {path}")
        return [path]

    assert config.training_input_mode == "combined_processed_csv_required"
    assert config.expected_processed_input_path is not None
    expected_path = resolve_configured_path(config, config.expected_processed_input_path)
    if not expected_path.exists():
        error_code = config.runtime_validation.get(
            "error_on_missing_expected_processed_input",
            "EXPECTED_PROCESSED_INPUT_REQUIRED",
        )
        raise DatasetSchemaError(
            f"{error_code}: {config.dataset_name} requires configured processed input file "
            f"{expected_path}. Refusing to fall back to current raw per-device files."
        )
    return [expected_path]


def _missing_label_error(config: DatasetConfig, path: Path, columns: Sequence[str]) -> DatasetSchemaError:
    candidates = ", ".join(config.candidate_label_columns_present)
    available = ", ".join(columns)
    return DatasetSchemaError(
        f"{config.dataset_name}: configured label column {config.label_column!r} is missing in {path}. "
        f"Available columns: {available}. Refusing to guess alternative label names. "
        f"Configured audit candidates: {candidates}."
    )


def _binary_label_error(config: DatasetConfig, path: Path, raw_counts: Counter[str], unknown_values: Counter[str]) -> DatasetSchemaError:
    observed = ", ".join(sorted(raw_counts))
    unknown = ", ".join(sorted(unknown_values))
    return DatasetSchemaError(
        f"{config.dataset_name}: label column {config.label_column!r} in {path} is not safely binary. "
        f"Observed raw values: {observed}. Unknown values after binary mapping: {unknown}."
    )


def validate_binary_label_counts(
    config: DatasetConfig,
    path: Path,
    raw_counts: Counter[str],
    *,
    require_both_classes: bool,
) -> dict[str, int]:
    binary_counts: Counter[str] = Counter()
    unknown_values: Counter[str] = Counter()

    for raw_value, count in raw_counts.items():
        mapped_value = label_value_to_binary(raw_value)
        if mapped_value is None:
            unknown_values[raw_value] += count
        else:
            binary_counts[mapped_value] += count

    if unknown_values:
        raise _binary_label_error(config, path, raw_counts, unknown_values)

    if require_both_classes and set(binary_counts) != {"0", "1"}:
        observed = ", ".join(f"{key}:{value}" for key, value in sorted(binary_counts.items()))
        raise DatasetSchemaError(
            f"{config.dataset_name}: label column {config.label_column!r} in {path} must contain both "
            f"binary classes {{0,1}} after mapping. Observed counts: {observed or 'none'}."
        )

    return sorted_counter(binary_counts)


def inspect_csv_file(config: DatasetConfig, path: Path, *, require_label: bool = True) -> CsvInspection:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_columns = next(reader)
        except StopIteration as exc:
            raise DatasetSchemaError(f"{config.dataset_name}: empty CSV file: {path}") from exc

        normalized_columns = [normalize_header(column) for column in raw_columns]
        if require_label and config.label_column not in normalized_columns:
            raise _missing_label_error(config, path, normalized_columns)

        missing_value_counts = {column: 0 for column in normalized_columns}
        non_missing_counts = {column: 0 for column in normalized_columns}
        numeric_flags = {column: True for column in normalized_columns}
        label_raw_counts: Counter[str] = Counter()
        row_count = 0
        header_issues: list[str] = []

        if raw_columns and raw_columns[0].startswith("\ufeff"):
            header_issues.append("UTF-8 BOM detected on the first header cell.")

        label_index = normalized_columns.index(config.label_column) if require_label else None

        for row in reader:
            row_count += 1
            if len(row) != len(normalized_columns):
                raise DatasetSchemaError(
                    f"{config.dataset_name}: row {row_count + 1} in {path} has {len(row)} value(s), "
                    f"but the header has {len(normalized_columns)} column(s)."
                )

            for column, value in zip(normalized_columns, row):
                normalized = normalize_value(value)
                if is_missing(normalized):
                    missing_value_counts[column] += 1
                    continue
                non_missing_counts[column] += 1
                if numeric_flags[column] and not is_numeric(normalized):
                    numeric_flags[column] = False

            if label_index is not None:
                label_raw_counts[normalize_value(row[label_index])] += 1

    binary_label_counts = (
        validate_binary_label_counts(
            config,
            path,
            label_raw_counts,
            require_both_classes=False,
        )
        if require_label
        else {}
    )
    exclusion_lookup = set(config.excluded_columns) | set(config.exclude_if_present)
    present_excluded_columns = tuple(column for column in normalized_columns if column in exclusion_lookup)
    retained_feature_columns = tuple(column for column in normalized_columns if column not in exclusion_lookup)
    numeric_columns = tuple(
        column for column in normalized_columns if numeric_flags[column] and non_missing_counts[column] > 0
    )
    categorical_columns = tuple(column for column in normalized_columns if column not in numeric_columns)

    return CsvInspection(
        path=path,
        raw_columns=tuple(raw_columns),
        normalized_columns=tuple(normalized_columns),
        row_count=row_count,
        missing_value_counts=missing_value_counts,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        present_excluded_columns=present_excluded_columns,
        retained_feature_columns=retained_feature_columns,
        label_raw_value_counts=sorted_counter(label_raw_counts),
        binary_label_counts=binary_label_counts,
        header_issues=tuple(header_issues),
    )


def inspect_dataset_files(
    config: DatasetConfig,
    file_paths: Sequence[Path],
    *,
    source_kind: str,
    require_consistent_schema: bool,
) -> DatasetInspection:
    if not file_paths:
        raise DatasetSchemaError(f"{config.dataset_name}: no configured CSV files were provided.")

    file_inspections = tuple(inspect_csv_file(config, path) for path in file_paths)
    dataset_binary_counts: Counter[str] = Counter()
    for inspection in file_inspections:
        dataset_binary_counts.update(inspection.binary_label_counts)

    if set(dataset_binary_counts) != {"0", "1"}:
        observed = ", ".join(f"{key}:{value}" for key, value in sorted(dataset_binary_counts.items()))
        file_list = ", ".join(path.name for path in file_paths)
        raise DatasetSchemaError(
            f"{config.dataset_name}: configured dataset files must contain both binary classes {{0,1}} after mapping. "
            f"Observed aggregate counts across [{file_list}]: {observed or 'none'}."
        )

    schema_variants = {inspection.normalized_columns for inspection in file_inspections}
    schema_consistent = len(schema_variants) == 1

    if require_consistent_schema and not schema_consistent:
        details = "; ".join(
            f"{inspection.path.name}: {', '.join(inspection.normalized_columns)}"
            for inspection in file_inspections
        )
        raise DatasetSchemaError(
            f"{config.dataset_name}: configured files do not share one consistent schema. {details}"
        )

    union_columns = tuple(
        sorted({column for inspection in file_inspections for column in inspection.normalized_columns})
    )
    intersection = set(file_inspections[0].normalized_columns)
    for inspection in file_inspections[1:]:
        intersection &= set(inspection.normalized_columns)

    audit_blockers: list[str] = []
    if not schema_consistent:
        audit_blockers.append("CSV schemas differ across configured files.")
    if config.training_input_mode == "combined_processed_csv_required":
        expected_path = resolve_configured_path(config, config.expected_processed_input_path or "")
        if not expected_path.exists():
            audit_blockers.append(
                "Configured combined telemetry training file is not present yet."
            )

    return DatasetInspection(
        config=config,
        source_kind=source_kind,
        file_inspections=file_inspections,
        file_paths=tuple(file_paths),
        schema_consistent_across_files=schema_consistent,
        union_columns=union_columns,
        intersection_columns=tuple(sorted(intersection)),
        audit_blockers=tuple(audit_blockers),
    )
