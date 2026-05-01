from __future__ import annotations

import csv
import hashlib
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from src.data.schema_validation import (
    DatasetConfigError,
    DatasetSchemaError,
    MISSING_TOKENS,
    label_value_to_binary,
    load_cluster_config,
    resolve_configured_path,
)
from src.data.transforms import NumericTransformArtifacts


DEFAULT_CONFIG_PATH = Path("configs/cluster1_batadal.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs_c1_batadal")
BATADAL_FILES = ("training_dataset_1.csv", "training_dataset_2.csv", "test_dataset.csv")
TIMESTAMP_COLUMN = "DATETIME"
LABEL_COLUMN = "ATT_FLAG"
DATETIME_FORMAT = "%d/%m/%y %H"
DEFAULT_VALIDATION_CUTOFF = datetime(2016, 11, 29, 5, 0)


@dataclass(frozen=True)
class BatadalRawFile:
    file_name: str
    path: Path
    header: tuple[str, ...]
    feature_names: tuple[str, ...]
    timestamps: tuple[datetime, ...]
    feature_matrix: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class BatadalSegment:
    segment_key: str
    file_name: str
    split_name: str
    timestamps: tuple[datetime, ...]
    feature_matrix: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class BatadalWindowRef:
    segment_key: str
    file_name: str
    split_name: str
    start: int
    stop: int
    label: int
    sequence: int
    end_timestamp: datetime


@dataclass(frozen=True)
class BatadalRuntimeClient:
    client_id: str
    train_inputs: np.ndarray
    train_labels: np.ndarray
    validation_inputs: np.ndarray
    validation_labels: np.ndarray
    test_inputs: np.ndarray
    test_labels: np.ndarray


@dataclass(frozen=True)
class BatadalRuntimeData:
    clients: tuple[BatadalRuntimeClient, ...]
    input_channels: int
    input_length: int
    data_summary: Mapping[str, Any]


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise DatasetConfigError(f"Expected mapping in YAML file: {path}")
    return data


def _label_counts(labels: Sequence[int] | np.ndarray) -> dict[str, int]:
    counter = Counter(str(int(label)) for label in np.asarray(labels, dtype=np.int8).tolist())
    return {key: counter[key] for key in sorted(counter)}


def _window_counts(windows: Sequence[BatadalWindowRef]) -> dict[str, int]:
    return _label_counts([window.label for window in windows])


def _balanced_counts(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def _parse_datetime(value: str, *, path: Path, row_number: int) -> datetime:
    text = str(value).strip()
    try:
        return datetime.strptime(text, DATETIME_FORMAT)
    except ValueError as exc:
        raise DatasetSchemaError(
            f"BATADAL: could not parse {TIMESTAMP_COLUMN}={text!r} in {path} row {row_number}. "
            f"Expected day-first format like 04/07/16 00."
        ) from exc


def _parse_label(value: str, *, path: Path, row_number: int) -> int:
    mapped = label_value_to_binary(str(value))
    if mapped is None:
        raise DatasetSchemaError(
            f"BATADAL: {LABEL_COLUMN} must be binary 0/1. "
            f"Observed {value!r} in {path} row {row_number}."
        )
    return int(mapped)


def _parse_feature(value: str) -> float:
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return float("nan")
    return float(text)


def _feature_group(feature_name: str) -> str:
    if feature_name.startswith("L_"):
        return "tank_level"
    if feature_name.startswith("F_"):
        return "flow"
    if feature_name.startswith("S_"):
        return "status"
    if feature_name.startswith("P_"):
        return "pressure"
    raise DatasetSchemaError(
        f"BATADAL: unexpected telemetry feature {feature_name!r}. "
        "Expected features to start with L_, F_, S_, or P_."
    )


def _validate_feature_schema(header: Sequence[str]) -> tuple[str, ...]:
    if TIMESTAMP_COLUMN not in header:
        raise DatasetSchemaError(f"BATADAL requires timestamp column {TIMESTAMP_COLUMN!r}.")
    if LABEL_COLUMN not in header:
        raise DatasetSchemaError(f"BATADAL requires binary label column {LABEL_COLUMN!r}.")
    feature_names = tuple(column for column in header if column not in {TIMESTAMP_COLUMN, LABEL_COLUMN})
    if len(feature_names) != 43:
        raise DatasetSchemaError(
            f"BATADAL requires 43 telemetry feature columns after excluding "
            f"{TIMESTAMP_COLUMN!r} and {LABEL_COLUMN!r}; observed {len(feature_names)}."
        )
    for feature_name in feature_names:
        _feature_group(feature_name)
    return feature_names


def _read_batadal_file(path: Path) -> BatadalRawFile:
    if not path.exists():
        raise DatasetSchemaError(f"BATADAL required raw file is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if not header:
            raise DatasetSchemaError(f"BATADAL raw CSV is empty: {path}")
        feature_names = _validate_feature_schema(header)

        timestamps: list[datetime] = []
        features: list[list[float]] = []
        labels: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            timestamps.append(_parse_datetime(row[TIMESTAMP_COLUMN], path=path, row_number=row_number))
            labels.append(_parse_label(row[LABEL_COLUMN], path=path, row_number=row_number))
            try:
                features.append([_parse_feature(row[name]) for name in feature_names])
            except ValueError as exc:
                raise DatasetSchemaError(
                    f"BATADAL: non-numeric telemetry value in {path} row {row_number}."
                ) from exc

    order = sorted(range(len(timestamps)), key=lambda index: (timestamps[index], index))
    ordered_timestamps = tuple(timestamps[index] for index in order)
    feature_matrix = np.asarray([features[index] for index in order], dtype=np.float32)
    label_array = np.asarray([labels[index] for index in order], dtype=np.int8)
    return BatadalRawFile(
        file_name=path.name,
        path=path,
        header=header,
        feature_names=feature_names,
        timestamps=ordered_timestamps,
        feature_matrix=feature_matrix,
        labels=label_array,
    )


def _load_raw_files(config_path: str | Path) -> tuple[Mapping[str, Any], Any, dict[str, BatadalRawFile]]:
    dataset_config = load_cluster_config(config_path)
    raw_config = _load_yaml(config_path)
    if dataset_config.cluster_id != 1:
        raise DatasetConfigError("BATADAL builder is restricted to Cluster 1.")
    if dataset_config.label_column != LABEL_COLUMN:
        raise DatasetConfigError(f"BATADAL Cluster 1 config must use label_column={LABEL_COLUMN!r}.")

    raw_dir = resolve_configured_path(dataset_config, dataset_config.current_raw_input_dir)
    files = tuple(dataset_config.current_raw_files)
    if files != BATADAL_FILES:
        raise DatasetConfigError(f"BATADAL Cluster 1 config must list current_raw_files={BATADAL_FILES!r}.")

    loaded = {file_name: _read_batadal_file(raw_dir / file_name) for file_name in files}
    first_header = loaded[files[0]].header
    mismatched = [file_name for file_name, raw_file in loaded.items() if raw_file.header != first_header]
    if mismatched:
        raise DatasetSchemaError(f"BATADAL requires identical schema across all three CSV files. Mismatch: {mismatched}")
    return raw_config, dataset_config, loaded


def _segment_from_mask(
    raw_file: BatadalRawFile,
    *,
    split_name: str,
    mask: np.ndarray,
) -> BatadalSegment:
    if mask.dtype != bool or mask.shape[0] != raw_file.labels.shape[0]:
        raise ValueError("BATADAL segment mask must be boolean and row-aligned.")
    indices = np.flatnonzero(mask)
    segment_key = f"{raw_file.file_name}::{split_name}"
    return BatadalSegment(
        segment_key=segment_key,
        file_name=raw_file.file_name,
        split_name=split_name,
        timestamps=tuple(raw_file.timestamps[index] for index in indices),
        feature_matrix=raw_file.feature_matrix[indices].copy(),
        labels=raw_file.labels[indices].copy(),
    )


def _build_raw_segments(
    raw_files: Mapping[str, BatadalRawFile],
    *,
    validation_cutoff: datetime,
) -> tuple[list[BatadalSegment], list[BatadalSegment], list[BatadalSegment]]:
    training_1 = raw_files["training_dataset_1.csv"]
    training_2 = raw_files["training_dataset_2.csv"]
    test = raw_files["test_dataset.csv"]

    train_segments = [
        _segment_from_mask(
            training_1,
            split_name="train",
            mask=np.ones(training_1.labels.shape[0], dtype=bool),
        )
    ]
    train_mask_2 = np.asarray([timestamp < validation_cutoff for timestamp in training_2.timestamps], dtype=bool)
    validation_mask_2 = ~train_mask_2
    train_segments.append(_segment_from_mask(training_2, split_name="train", mask=train_mask_2))
    validation_segments = [_segment_from_mask(training_2, split_name="validation", mask=validation_mask_2)]
    test_segments = [
        _segment_from_mask(
            test,
            split_name="test",
            mask=np.ones(test.labels.shape[0], dtype=bool),
        )
    ]
    for split_name, segments in (
        ("train", train_segments),
        ("validation", validation_segments),
        ("test", test_segments),
    ):
        if sum(segment.labels.shape[0] for segment in segments) == 0:
            raise DatasetSchemaError(f"BATADAL raw {split_name} split has zero rows.")
    return train_segments, validation_segments, test_segments


def _build_windows(
    segments: Sequence[BatadalSegment],
    *,
    window_length: int,
    stride: int,
    start_sequence: int = 0,
) -> list[BatadalWindowRef]:
    windows: list[BatadalWindowRef] = []
    sequence = start_sequence
    for segment in segments:
        if segment.labels.shape[0] < window_length:
            continue
        for start in range(0, segment.labels.shape[0] - window_length + 1, stride):
            stop = start + window_length
            windows.append(
                BatadalWindowRef(
                    segment_key=segment.segment_key,
                    file_name=segment.file_name,
                    split_name=segment.split_name,
                    start=start,
                    stop=stop,
                    label=int(segment.labels[stop - 1]),
                    sequence=sequence,
                    end_timestamp=segment.timestamps[stop - 1],
                )
            )
            sequence += 1
    return windows


def _validate_both_window_classes(split_name: str, windows: Sequence[BatadalWindowRef]) -> None:
    counts = _window_counts(windows)
    if set(counts) != {"0", "1"}:
        raise DatasetSchemaError(
            f"BATADAL {split_name} windows must contain both classes for supervised IDS evaluation. "
            f"Observed {counts}."
        )


def batadal_training_client_assignment(
    windows: Sequence[BatadalWindowRef],
    *,
    num_clients: int,
) -> list[list[BatadalWindowRef]]:
    positives = [window for window in windows if window.label == 1]
    negatives = [window for window in windows if window.label == 0]
    clients: list[list[BatadalWindowRef]] = [[] for _ in range(num_clients)]

    for index, window in enumerate(positives):
        clients[index % num_clients].append(window)

    targets = _balanced_counts(len(windows), num_clients)
    negative_index = 0
    for client_index, target in enumerate(targets):
        deficit = max(0, target - len(clients[client_index]))
        if deficit:
            clients[client_index].extend(negatives[negative_index : negative_index + deficit])
            negative_index += deficit

    client_index = 0
    while negative_index < len(negatives):
        clients[client_index % num_clients].append(negatives[negative_index])
        negative_index += 1
        client_index += 1

    return [sorted(client_windows, key=lambda window: window.sequence) for client_windows in clients]


def _contiguous_window_assignment(
    windows: Sequence[BatadalWindowRef],
    *,
    num_clients: int,
) -> list[list[BatadalWindowRef]]:
    targets = _balanced_counts(len(windows), num_clients)
    clients: list[list[BatadalWindowRef]] = []
    cursor = 0
    for target in targets:
        clients.append(list(windows[cursor : cursor + target]))
        cursor += target
    return clients


def _fit_training_preprocessor(
    train_segments: Sequence[BatadalSegment],
    feature_names: Sequence[str],
) -> NumericTransformArtifacts:
    training_rows = np.vstack([segment.feature_matrix for segment in train_segments])
    artifacts = NumericTransformArtifacts.fit(feature_names, training_rows)
    if not artifacts.kept_features:
        raise DatasetSchemaError("BATADAL preprocessing dropped all feature columns.")
    return artifacts


def _transform_segments(
    segments: Sequence[BatadalSegment],
    artifacts: NumericTransformArtifacts,
) -> dict[str, np.ndarray]:
    return {
        segment.segment_key: artifacts.transform(segment.feature_matrix)
        for segment in segments
    }


def _windows_to_arrays(
    transformed_by_segment: Mapping[str, np.ndarray],
    windows: Sequence[BatadalWindowRef],
    *,
    input_channels: int,
    window_length: int,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = list(windows)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        return (
            np.empty((0, input_channels, window_length), dtype=np.float32),
            np.empty(0, dtype=np.int8),
        )

    inputs = np.empty((len(selected), input_channels, window_length), dtype=np.float32)
    labels = np.empty(len(selected), dtype=np.int8)
    for index, window in enumerate(selected):
        inputs[index] = transformed_by_segment[window.segment_key][window.start : window.stop].T
        labels[index] = int(window.label)
    return inputs, labels


def _nan_stat(values: np.ndarray, op: str) -> float:
    if values.size == 0 or np.isnan(values).all():
        return 0.0
    if op == "mean":
        return float(np.nanmean(values))
    if op == "std":
        return float(np.nanstd(values))
    raise ValueError(f"Unsupported descriptor op: {op}")


def _client_descriptor(
    *,
    feature_names: Sequence[str],
    segments_by_key: Mapping[str, BatadalSegment],
    windows: Sequence[BatadalWindowRef],
    include_attack_ratio: bool,
) -> np.ndarray:
    if not windows:
        raise DatasetSchemaError("BATADAL descriptor computation requires non-empty client training windows.")

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for window in windows:
        segment = segments_by_key[window.segment_key]
        rows.append(segment.feature_matrix[window.start : window.stop])
        labels.append(window.label)
    matrix = np.vstack(rows)

    descriptor_values: list[float] = []
    for feature_index, feature_name in enumerate(feature_names):
        group = _feature_group(feature_name)
        values = matrix[:, feature_index].astype(np.float64, copy=False)
        if group in {"tank_level", "flow", "pressure"}:
            descriptor_values.append(_nan_stat(values, "mean"))
            descriptor_values.append(_nan_stat(values, "std"))
        elif group == "status":
            descriptor_values.append(_nan_stat(values, "mean"))

    if include_attack_ratio:
        descriptor_values.append(float(np.mean(np.asarray(labels, dtype=np.float32))))
    return np.asarray(descriptor_values, dtype=np.float32)


def _descriptor_names(feature_names: Sequence[str], *, include_attack_ratio: bool) -> tuple[str, ...]:
    names: list[str] = []
    for feature_name in feature_names:
        group = _feature_group(feature_name)
        if group in {"tank_level", "flow", "pressure"}:
            names.extend((f"{feature_name}_mean", f"{feature_name}_std"))
        elif group == "status":
            names.append(f"{feature_name}_duty_cycle")
    if include_attack_ratio:
        names.append("training_window_attack_ratio")
    return tuple(names)


def _agglomerative_model(n_subclusters: int) -> AgglomerativeClustering:
    try:
        return AgglomerativeClustering(n_clusters=n_subclusters, linkage="ward", metric="euclidean")
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_subclusters, linkage="ward", affinity="euclidean")


def _client_sequence(client_id: str) -> int:
    return int(client_id.rsplit("L", 1)[1])


def _normalize_assignments(
    client_ids: Sequence[str],
    raw_labels: Sequence[int],
    fixed_subcluster_ids: Sequence[str],
) -> dict[str, str]:
    members_by_label: dict[int, list[str]] = {}
    for client_id, raw_label in zip(client_ids, raw_labels, strict=True):
        members_by_label.setdefault(int(raw_label), []).append(client_id)
    if len(members_by_label) != len(fixed_subcluster_ids):
        raise DatasetSchemaError(
            f"BATADAL agglomerative clustering produced {len(members_by_label)} non-empty subclusters, "
            f"expected {len(fixed_subcluster_ids)}."
        )
    sorted_labels = sorted(
        members_by_label,
        key=lambda label: min(_client_sequence(client_id) for client_id in members_by_label[label]),
    )
    raw_to_fixed = {raw_label: fixed_subcluster_ids[index] for index, raw_label in enumerate(sorted_labels)}
    return {
        client_id: raw_to_fixed[int(raw_label)]
        for client_id, raw_label in zip(client_ids, raw_labels, strict=True)
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _segment_profile(segment: BatadalSegment) -> dict[str, Any]:
    return {
        "segment_key": segment.segment_key,
        "file_name": segment.file_name,
        "split": segment.split_name,
        "row_count": int(segment.labels.shape[0]),
        "label_counts": _label_counts(segment.labels),
        "start_timestamp": segment.timestamps[0].isoformat() if segment.timestamps else None,
        "end_timestamp": segment.timestamps[-1].isoformat() if segment.timestamps else None,
    }


def _raw_profile(raw_file: BatadalRawFile) -> dict[str, Any]:
    return {
        "file_name": raw_file.file_name,
        "path": str(raw_file.path),
        "row_count": int(raw_file.labels.shape[0]),
        "label_counts": _label_counts(raw_file.labels),
        "start_timestamp": raw_file.timestamps[0].isoformat() if raw_file.timestamps else None,
        "end_timestamp": raw_file.timestamps[-1].isoformat() if raw_file.timestamps else None,
        "feature_count": len(raw_file.feature_names),
    }


def _client_entry(
    client_id: str,
    train_windows: Sequence[BatadalWindowRef],
    validation_windows: Sequence[BatadalWindowRef],
    test_windows: Sequence[BatadalWindowRef],
) -> dict[str, Any]:
    train_counts = _window_counts(train_windows)
    validation_counts = _window_counts(validation_windows)
    test_counts = _window_counts(test_windows)
    return {
        "client_id": client_id,
        "num_train_samples": len(train_windows),
        "num_val_samples": len(validation_windows),
        "num_test_samples": len(test_windows),
        "train_label_counts": train_counts,
        "val_label_counts": validation_counts,
        "test_label_counts": test_counts,
        "train_positive_windows": int(train_counts.get("1", 0)),
        "validation_positive_windows": int(validation_counts.get("1", 0)),
        "test_positive_windows": int(test_counts.get("1", 0)),
        "notes": ["controlled_federated_emulation_positive_round_robin_negative_chronological_fill"],
    }


def _subcluster_profile_lines(
    *,
    client_ids: Sequence[str],
    assignments: Mapping[str, str],
    descriptor_names: Sequence[str],
    descriptor_matrix: np.ndarray,
    fixed_subcluster_ids: Sequence[str],
) -> list[str]:
    lines = [
        "# Cluster 1 BATADAL Sub-Cluster Profile",
        "",
        "H1 and H2 are fixed identifier labels assigned after one offline AgglomerativeClustering run. "
        "The table reports descriptor differences only; it does not assign physical meanings to H1 or H2.",
        "",
        "## Membership",
        "",
        "| sub-cluster | clients |",
        "|---|---|",
    ]
    for subcluster_id in fixed_subcluster_ids:
        members = [client_id for client_id in client_ids if assignments[client_id] == subcluster_id]
        lines.append(f"| {subcluster_id} | {', '.join(members)} |")

    centroids: dict[str, np.ndarray] = {}
    for subcluster_id in fixed_subcluster_ids:
        indices = [index for index, client_id in enumerate(client_ids) if assignments[client_id] == subcluster_id]
        centroids[subcluster_id] = descriptor_matrix[indices].mean(axis=0)

    if len(fixed_subcluster_ids) == 2:
        left, right = fixed_subcluster_ids
        delta = centroids[right] - centroids[left]
        top_indices = np.argsort(np.abs(delta))[::-1][:12]
        lines.extend(
            [
                "",
                "## Largest Descriptor Centroid Differences",
                "",
                f"Positive delta means `{right}` is higher than `{left}`.",
                "",
                "| descriptor | delta |",
                "|---|---:|",
            ]
        )
        for index in top_indices:
            lines.append(f"| {descriptor_names[index]} | {float(delta[index]):.6f} |")
    return lines


def _write_file_split_report(path: Path, profile: Mapping[str, Any]) -> None:
    lines = [
        "# Cluster 1 BATADAL File Split Report",
        "",
        "Rows are sorted chronologically inside each source file before splitting. "
        "Raw train/validation/test boundaries are applied before sliding-window generation.",
        "",
        f"- Validation cutoff for `training_dataset_2.csv`: `{profile['validation_cutoff']}`",
        "- Training: all `training_dataset_1.csv` rows plus `training_dataset_2.csv` rows before the cutoff.",
        "- Validation: `training_dataset_2.csv` rows at or after the cutoff.",
        "- Held-out test: all `test_dataset.csv` rows.",
        "",
        "| split | source segment | rows | ATT_FLAG=0 | ATT_FLAG=1 | start | end |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for split_name in ("train", "validation", "test"):
        for segment in profile["segments"][split_name]:
            counts = segment["label_counts"]
            lines.append(
                f"| {split_name} | {segment['segment_key']} | {segment['row_count']} | "
                f"{counts.get('0', 0)} | {counts.get('1', 0)} | "
                f"{segment['start_timestamp']} | {segment['end_timestamp']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_client_balance_report(path: Path, clients: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Cluster 1 BATADAL Client Balance Report",
        "",
        "This is controlled federated emulation. BATADAL is one water-distribution SCADA dataset; "
        "the 12 leaf clients are deterministic emulated clients, not physical independent water utilities.",
        "",
        "Positive training windows are assigned round-robin across clients. Negative windows are then used to "
        "fill client quotas while preserving chronological order as much as practical.",
        "",
        "| client | train 0 | train 1 | validation 0 | validation 1 | test 0 | test 1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for client in clients:
        train_counts = client["train_label_counts"]
        validation_counts = client["val_label_counts"]
        test_counts = client["test_label_counts"]
        lines.append(
            f"| {client['client_id']} | {train_counts.get('0', 0)} | {train_counts.get('1', 0)} | "
            f"{validation_counts.get('0', 0)} | {validation_counts.get('1', 0)} | "
            f"{test_counts.get('0', 0)} | {test_counts.get('1', 0)} |"
        )
    clients_with_positive_train = sum(1 for client in clients if int(client["train_label_counts"].get("1", 0)) > 0)
    lines.extend(
        [
            "",
            f"Clients with at least one positive training window: `{clients_with_positive_train}/{len(clients)}`",
            f"Every client has positive training windows: `{'YES' if clients_with_positive_train == len(clients) else 'NO'}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_feature_drop_report(
    path: Path,
    *,
    artifacts: NumericTransformArtifacts,
) -> None:
    lines = [
        "# Cluster 1 BATADAL Feature Drop Report",
        "",
        f"Input telemetry features: `{len(artifacts.input_features)}`",
        f"Retained model features: `{len(artifacts.kept_features)}`",
        "",
        "| reason | features |",
        "|---|---|",
        f"| all missing in training rows | {', '.join(artifacts.dropped_all_missing_features) or 'none'} |",
        f"| constant in training rows | {', '.join(artifacts.dropped_constant_features) or 'none'} |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_summary_report(
    path: Path,
    *,
    profile: Mapping[str, Any],
    clients: Sequence[Mapping[str, Any]],
    active_model_family: str,
) -> None:
    clients_with_positive_train = sum(1 for client in clients if int(client["train_label_counts"].get("1", 0)) > 0)
    lines = [
        "# Cluster 1 BATADAL Validation Summary",
        "",
        f"- Train row counts and label counts: `{profile['row_counts']['train']}`, `{profile['row_label_counts']['train']}`",
        f"- Validation row counts and label counts: `{profile['row_counts']['validation']}`, `{profile['row_label_counts']['validation']}`",
        f"- Test row counts and label counts: `{profile['row_counts']['test']}`, `{profile['row_label_counts']['test']}`",
        f"- Train window counts and label counts: `{profile['window_counts']['train_total']}`, `{profile['window_label_counts']['train']}`",
        f"- Validation window counts and label counts: `{profile['window_counts']['validation_total']}`, `{profile['window_label_counts']['validation']}`",
        f"- Test window counts and label counts: `{profile['window_counts']['test_total']}`, `{profile['window_label_counts']['test']}`",
        f"- Number of leaf clients: `{profile['num_leaf_clients']}`",
        f"- Number of sub-clusters: `{profile['num_subclusters']}`",
        f"- Every client has positive training windows: `{'YES' if clients_with_positive_train == len(clients) else 'NO'}`",
        "- Test leakage confirmation: `test_dataset.csv` is never used for training, validation, scaler fitting, "
        "imputation fitting, clustering, descriptor computation, threshold tuning, or hyperparameter selection.",
        f"- Active model family used for P_C1_BATADAL: `{active_model_family}`",
        "",
        "Primary thesis metric for BATADAL Cluster 1 is held-out test F1. Accuracy is reported only as a secondary metric.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_count(segments: Sequence[BatadalSegment]) -> int:
    return int(sum(segment.labels.shape[0] for segment in segments))


def _row_label_counts(segments: Sequence[BatadalSegment]) -> dict[str, int]:
    if not segments:
        return {}
    return _label_counts(np.concatenate([segment.labels for segment in segments]))


def _assemble(
    config_path: str | Path,
    *,
    output_root: str | Path | None,
    write_outputs: bool,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
) -> dict[str, Any]:
    raw_config, dataset_config, raw_files = _load_raw_files(config_path)
    partitioning = raw_config.get("partitioning")
    preprocessing = raw_config.get("preprocessing")
    clustering = raw_config.get("clustering")
    experiment_defaults = raw_config.get("experiment_defaults", {})
    if not isinstance(partitioning, Mapping) or not isinstance(preprocessing, Mapping) or not isinstance(clustering, Mapping):
        raise DatasetConfigError("BATADAL config must define partitioning, preprocessing, and clustering sections.")
    if not isinstance(experiment_defaults, Mapping):
        raise DatasetConfigError("BATADAL config experiment_defaults must be a mapping when provided.")

    num_clients = int(partitioning.get("candidate_leaf_clients", 0))
    if num_clients != 12:
        raise DatasetConfigError("Cluster 1 BATADAL must keep exactly 12 leaf clients.")
    if str(partitioning.get("strategy")) != "batadal_controlled_emulation":
        raise DatasetConfigError("Cluster 1 BATADAL requires partitioning.strategy=batadal_controlled_emulation.")

    window_length = int(preprocessing.get("window_length", 0))
    stride = int(preprocessing.get("stride", 0))
    if window_length <= 0 or stride <= 0:
        raise DatasetConfigError("BATADAL window_length and stride must be positive.")
    if str(preprocessing.get("window_label_rule")) != "last_row":
        raise DatasetConfigError("BATADAL default thesis protocol requires window_label_rule=last_row.")

    fixed_subcluster_ids = tuple(str(item) for item in clustering.get("fixed_subcluster_ids", []))
    n_subclusters = int(clustering.get("fixed_subclusters", 0))
    if n_subclusters != 2 or fixed_subcluster_ids != ("H1", "H2"):
        raise DatasetConfigError("Cluster 1 BATADAL must keep fixed sub-clusters H1 and H2.")

    cutoff_text = str(preprocessing.get("validation_cutoff", "2016-11-29 05:00"))
    validation_cutoff = datetime.strptime(cutoff_text, "%Y-%m-%d %H:%M")
    root = Path(output_root) if output_root is not None else Path(raw_config.get("data", {}).get("batadal_output_root", DEFAULT_OUTPUT_ROOT))

    feature_names = raw_files["training_dataset_1.csv"].feature_names
    train_segments, validation_segments, test_segments = _build_raw_segments(
        raw_files,
        validation_cutoff=validation_cutoff,
    )
    train_windows = _build_windows(train_segments, window_length=window_length, stride=stride, start_sequence=0)
    validation_windows = _build_windows(
        validation_segments,
        window_length=window_length,
        stride=stride,
        start_sequence=len(train_windows),
    )
    test_windows = _build_windows(
        test_segments,
        window_length=window_length,
        stride=stride,
        start_sequence=len(train_windows) + len(validation_windows),
    )
    if not train_windows or not validation_windows or not test_windows:
        raise DatasetSchemaError("BATADAL produced zero windows for at least one split.")
    _validate_both_window_classes("training", train_windows)
    _validate_both_window_classes("validation", validation_windows)
    _validate_both_window_classes("test", test_windows)

    training_clients = batadal_training_client_assignment(train_windows, num_clients=num_clients)
    validation_clients = _contiguous_window_assignment(validation_windows, num_clients=num_clients)
    test_clients = _contiguous_window_assignment(test_windows, num_clients=num_clients)

    numeric_artifacts = _fit_training_preprocessor(train_segments, feature_names)
    all_segments = train_segments + validation_segments + test_segments
    transformed_by_segment = _transform_segments(all_segments, numeric_artifacts)

    client_ids = [f"C1_L{index + 1:03d}" for index in range(num_clients)]
    runtime_clients: list[BatadalRuntimeClient] = []
    client_entries: list[dict[str, Any]] = []
    for index, client_id in enumerate(client_ids):
        train_inputs, train_labels = _windows_to_arrays(
            transformed_by_segment,
            training_clients[index],
            input_channels=len(numeric_artifacts.kept_features),
            window_length=window_length,
            limit=max_train_examples_per_client,
        )
        validation_inputs, validation_labels = _windows_to_arrays(
            transformed_by_segment,
            validation_clients[index],
            input_channels=len(numeric_artifacts.kept_features),
            window_length=window_length,
            limit=max_eval_examples_per_client,
        )
        test_inputs, test_labels = _windows_to_arrays(
            transformed_by_segment,
            test_clients[index],
            input_channels=len(numeric_artifacts.kept_features),
            window_length=window_length,
            limit=max_eval_examples_per_client,
        )
        if train_labels.size == 0:
            raise DatasetSchemaError(f"{client_id}: BATADAL requires at least one training window.")
        runtime_clients.append(
            BatadalRuntimeClient(
                client_id=client_id,
                train_inputs=train_inputs,
                train_labels=train_labels,
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                test_inputs=test_inputs,
                test_labels=test_labels,
            )
        )
        client_entries.append(_client_entry(client_id, training_clients[index], validation_clients[index], test_clients[index]))

    include_attack_ratio = bool(clustering.get("include_attack_ratio_in_descriptor", True))
    segments_by_key = {segment.segment_key: segment for segment in train_segments}
    descriptor_names = _descriptor_names(feature_names, include_attack_ratio=include_attack_ratio)
    descriptor_matrix = np.vstack(
        [
            _client_descriptor(
                feature_names=feature_names,
                segments_by_key=segments_by_key,
                windows=training_clients[index],
                include_attack_ratio=include_attack_ratio,
            )
            for index in range(num_clients)
        ]
    )
    descriptor_scaler = StandardScaler()
    standardized_descriptor_matrix = descriptor_scaler.fit_transform(descriptor_matrix).astype(np.float32, copy=False)
    raw_labels = _agglomerative_model(n_subclusters).fit_predict(standardized_descriptor_matrix).tolist()
    assignments = _normalize_assignments(client_ids, raw_labels, fixed_subcluster_ids)

    clients_for_membership = [
        {"client_id": client_id, "subcluster_id": assignments[client_id]}
        for client_id in client_ids
    ]
    membership_hash = hashlib.sha256(json.dumps(clients_for_membership, sort_keys=True).encode("utf-8")).hexdigest()
    reports_dir = root / "reports"
    preprocessing_dir = root / "preprocessing"
    clients_dir = root / "clients"
    clustering_dir = root / "clustering"
    client_metadata_path = clients_dir / "cluster1_leaf_clients.json"
    descriptor_scaler_path = clustering_dir / "cluster1_descriptor_scaler.pkl"
    membership_path = clustering_dir / "cluster1_memberships.json"

    subclusters = [
        {
            "subcluster_id": subcluster_id,
            "client_ids": [client_id for client_id in client_ids if assignments[client_id] == subcluster_id],
        }
        for subcluster_id in fixed_subcluster_ids
    ]
    membership = {
        "cluster_id": 1,
        "dataset": "BATADAL",
        "variant": "cluster1_batadal",
        "status": "ok",
        "clustering_method": "AgglomerativeClustering",
        "linkage": "ward",
        "metric": "euclidean",
        "descriptor": "batadal_group_mean_std_status_duty_cycle_attack_ratio"
        if include_attack_ratio
        else "batadal_group_mean_std_status_duty_cycle",
        "descriptor_dim": int(descriptor_matrix.shape[1]),
        "descriptor_source_split": "train",
        "descriptor_attack_ratio_included": include_attack_ratio,
        "n_subclusters": n_subclusters,
        "fixed_subcluster_ids": list(fixed_subcluster_ids),
        "frozen": True,
        "membership_hash": membership_hash,
        "client_metadata_path": str(client_metadata_path),
        "descriptor_scaler_path": str(descriptor_scaler_path),
        "membership_file": str(membership_path),
        "reuse_for_experiment_groups": [
            "baseline_uniform_hierarchical",
            "proposed_specialized_hierarchical",
        ],
        "subclusters": subclusters,
        "clients": clients_for_membership,
    }

    row_counts = {
        "train": _row_count(train_segments),
        "validation": _row_count(validation_segments),
        "test": _row_count(test_segments),
    }
    row_label_counts = {
        "train": _row_label_counts(train_segments),
        "validation": _row_label_counts(validation_segments),
        "test": _row_label_counts(test_segments),
    }
    window_label_counts = {
        "train": _window_counts(train_windows),
        "validation": _window_counts(validation_windows),
        "test": _window_counts(test_windows),
    }
    client_train_label_counts = {
        client.client_id: _label_counts(client.train_labels) for client in runtime_clients
    }
    client_validation_label_counts = {
        client.client_id: _label_counts(client.validation_labels) for client in runtime_clients
    }
    client_test_label_counts = {
        client.client_id: _label_counts(client.test_labels) for client in runtime_clients
    }
    active_model_family = str(experiment_defaults.get("proposed_model_family", "cnn1d_bn"))
    profile = {
        "cluster_id": 1,
        "dataset": "BATADAL",
        "dataset_description": "BATADAL water-distribution SCADA telemetry",
        "variant": "cluster1_batadal",
        "status": "ok",
        "config_path": str(config_path),
        "output_root": str(root),
        "raw_files": [_raw_profile(raw_files[file_name]) for file_name in BATADAL_FILES],
        "validation_cutoff": cutoff_text,
        "segments": {
            "train": [_segment_profile(segment) for segment in train_segments],
            "validation": [_segment_profile(segment) for segment in validation_segments],
            "test": [_segment_profile(segment) for segment in test_segments],
        },
        "row_counts": row_counts,
        "row_label_counts": row_label_counts,
        "window_counts": {
            "train_total": len(train_windows),
            "validation_total": len(validation_windows),
            "test_total": len(test_windows),
        },
        "window_label_counts": window_label_counts,
        "feature_columns_before_preprocessing": list(feature_names),
        "feature_columns_after_preprocessing": list(numeric_artifacts.kept_features),
        "dropped_all_missing_feature_columns": list(numeric_artifacts.dropped_all_missing_features),
        "dropped_constant_feature_columns": list(numeric_artifacts.dropped_constant_features),
        "input_type": "multivariate_time_series",
        "window_length": window_length,
        "stride": stride,
        "window_label_rule": "last_row",
        "num_leaf_clients": num_clients,
        "num_subclusters": n_subclusters,
        "client_metadata_path": str(client_metadata_path),
        "membership_path": str(membership_path),
        "preprocessing_artifact_paths": {
            "imputer": str(preprocessing_dir / "cluster1_batadal_imputer.pkl"),
            "scaler": str(preprocessing_dir / "cluster1_batadal_scaler.pkl"),
            "preprocessor": str(preprocessing_dir / "cluster1_batadal_preprocessor.pkl"),
        },
        "test_leakage_prevention": {
            "test_dataset_used_for_training": False,
            "test_dataset_used_for_validation": False,
            "test_dataset_used_for_scaler_fit": False,
            "test_dataset_used_for_imputation_fit": False,
            "test_dataset_used_for_descriptor_computation": False,
            "test_dataset_used_for_clustering": False,
            "test_dataset_used_for_threshold_tuning": False,
            "test_dataset_used_for_hyperparameter_selection": False,
        },
        "active_proposed_model_family": active_model_family,
    }
    client_metadata = {
        "cluster_id": 1,
        "dataset": "BATADAL",
        "variant": "cluster1_batadal",
        "num_leaf_clients": num_clients,
        "partitioning_strategy": "controlled_federated_emulation_positive_round_robin_negative_chronological_fill",
        "controlled_emulation_note": (
            "BATADAL is partitioned into 12 deterministic emulated leaf clients for simulation; "
            "these clients are not physical independent water utilities."
        ),
        "source_paths": [str(raw_files[file_name].path) for file_name in ("training_dataset_1.csv", "training_dataset_2.csv")],
        "heldout_test_source_paths": [str(raw_files["test_dataset.csv"].path)],
        "descriptor_source_split": "train",
        "cluster_split_label_counts": {
            "train": window_label_counts["train"],
            "validation": window_label_counts["validation"],
            "test": window_label_counts["test"],
        },
        "clients": client_entries,
    }
    data_summary = {
        "cluster_id": 1,
        "dataset": "BATADAL",
        "dataset_description": "BATADAL water-distribution SCADA telemetry",
        "variant": "cluster1_batadal",
        "num_clients": len(runtime_clients),
        "input_adapter": "sliding_window_feature_channels",
        "input_channels": len(numeric_artifacts.kept_features),
        "input_length": window_length,
        "retained_input_feature_columns": list(feature_names),
        "model_feature_columns": list(numeric_artifacts.kept_features),
        "dropped_all_missing_feature_columns": list(numeric_artifacts.dropped_all_missing_features),
        "dropped_constant_feature_columns": list(numeric_artifacts.dropped_constant_features),
        "dropped_high_cardinality_categorical_columns": [],
        "max_train_examples_per_client": max_train_examples_per_client,
        "max_eval_examples_per_client": max_eval_examples_per_client,
        "train_source_files": ["training_dataset_1.csv", "training_dataset_2.csv"],
        "validation_source_files": ["training_dataset_2.csv"],
        "heldout_test_source_files": ["test_dataset.csv"],
        "test_dataset_used_for_training": False,
        "test_dataset_used_for_validation": False,
        "test_dataset_used_for_preprocessing_fit": False,
        "test_dataset_used_for_clustering": False,
        "test_dataset_used_for_descriptor_computation": False,
        "test_dataset_used_for_threshold_tuning": False,
        "client_train_sample_counts": {
            client.client_id: client.train_labels.shape[0] for client in runtime_clients
        },
        "client_validation_sample_counts": {
            client.client_id: client.validation_labels.shape[0] for client in runtime_clients
        },
        "client_test_sample_counts": {
            client.client_id: client.test_labels.shape[0] for client in runtime_clients
        },
        "client_train_label_counts": client_train_label_counts,
        "client_validation_label_counts": client_validation_label_counts,
        "client_test_label_counts": client_test_label_counts,
        "cluster_split_label_counts": {
            "train": window_label_counts["train"],
            "validation": window_label_counts["validation"],
            "test": window_label_counts["test"],
        },
        "window_label_rule": "last_row",
        "preprocessing_fit_scope": "cluster1_training_rows_only",
    }

    if write_outputs:
        _write_json(client_metadata_path, client_metadata)
        _write_json(membership_path, membership)
        _write_pickle(descriptor_scaler_path, descriptor_scaler)
        _write_pickle(
            preprocessing_dir / "cluster1_batadal_imputer.pkl",
            {
                "variant": "cluster1_batadal",
                "fit_scope": "cluster1_training_rows_only",
                "input_features": numeric_artifacts.input_features,
                "kept_features": numeric_artifacts.kept_features,
                "medians": numeric_artifacts.medians.tolist(),
            },
        )
        _write_pickle(
            preprocessing_dir / "cluster1_batadal_scaler.pkl",
            {
                "variant": "cluster1_batadal",
                "fit_scope": "cluster1_training_rows_only",
                "input_features": numeric_artifacts.input_features,
                "kept_features": numeric_artifacts.kept_features,
                "means": numeric_artifacts.means.tolist(),
                "scales": numeric_artifacts.scales.tolist(),
            },
        )
        _write_pickle(
            preprocessing_dir / "cluster1_batadal_preprocessor.pkl",
            {
                "variant": "cluster1_batadal",
                "label_column": LABEL_COLUMN,
                "numeric_artifacts": numeric_artifacts,
                "output_feature_names": numeric_artifacts.kept_features,
                "window_length": window_length,
                "stride": stride,
                "window_label_rule": "last_row",
            },
        )
        _write_json(reports_dir / "data_profile_cluster1_batadal.json", profile)
        _write_file_split_report(reports_dir / "file_split_cluster1_batadal.md", profile)
        _write_client_balance_report(reports_dir / "client_balance_cluster1_batadal.md", client_entries)
        subcluster_lines = _subcluster_profile_lines(
            client_ids=client_ids,
            assignments=assignments,
            descriptor_names=descriptor_names,
            descriptor_matrix=descriptor_matrix,
            fixed_subcluster_ids=fixed_subcluster_ids,
        )
        (reports_dir / "subcluster_profile_cluster1_batadal.md").parent.mkdir(parents=True, exist_ok=True)
        (reports_dir / "subcluster_profile_cluster1_batadal.md").write_text(
            "\n".join(subcluster_lines) + "\n",
            encoding="utf-8",
        )
        _write_validation_summary_report(
            reports_dir / "cluster1_batadal_validation_summary.md",
            profile=profile,
            clients=client_entries,
            active_model_family=active_model_family,
        )
        _write_feature_drop_report(
            reports_dir / "feature_drop_report_cluster1_batadal.md",
            artifacts=numeric_artifacts,
        )

    return {
        "runtime": BatadalRuntimeData(
            clients=tuple(runtime_clients),
            input_channels=len(numeric_artifacts.kept_features),
            input_length=window_length,
            data_summary=data_summary,
        ),
        "profile": profile,
        "client_metadata": client_metadata,
        "membership": membership,
        "paths": {
            "data_profile": str(reports_dir / "data_profile_cluster1_batadal.json"),
            "file_split_report": str(reports_dir / "file_split_cluster1_batadal.md"),
            "client_balance_report": str(reports_dir / "client_balance_cluster1_batadal.md"),
            "subcluster_profile": str(reports_dir / "subcluster_profile_cluster1_batadal.md"),
            "validation_summary": str(reports_dir / "cluster1_batadal_validation_summary.md"),
            "client_metadata": str(client_metadata_path),
            "membership": str(membership_path),
        },
    }


def prepare_cluster1_batadal(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    return _assemble(config_path, output_root=output_root, write_outputs=True)


def build_batadal_runtime(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
) -> BatadalRuntimeData:
    result = _assemble(
        config_path,
        output_root=None,
        write_outputs=False,
        max_train_examples_per_client=max_train_examples_per_client,
        max_eval_examples_per_client=max_eval_examples_per_client,
    )
    runtime = result["runtime"]
    if not isinstance(runtime, BatadalRuntimeData):
        raise TypeError("BATADAL runtime assembly returned an unexpected payload.")
    return runtime
