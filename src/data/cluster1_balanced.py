from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data.schema_validation import DatasetSchemaError, normalize_header


DEFAULT_RAW_DIR = Path("data/raw/hai_2103/hai-21.03")
DEFAULT_OUTPUT_ROOT = Path("outputs_c1_balanced_train")
DEFAULT_VARIANT_ROOT = Path("data/variants/hai_2103_balanced_train")
DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_DOC_PATH = Path("docs/CLUSTER1_BALANCED_TRAINING_VARIANT.md")

EXPECTED_FILE_ORDER = (
    "train1.csv",
    "train2.csv",
    "train3.csv",
    "test1.csv",
    "test2.csv",
    "test3.csv",
    "test4.csv",
    "test5.csv",
)
EXPECTED_HAI_COUNTS: dict[str, tuple[int, int, int]] = {
    "train1.csv": (216001, 216001, 0),
    "train2.csv": (226801, 226801, 0),
    "train3.csv": (478801, 478801, 0),
    "test1.csv": (43201, 42572, 629),
    "test2.csv": (118801, 115352, 3449),
    "test3.csv": (108001, 106466, 1535),
    "test4.csv": (39601, 38444, 1157),
    "test5.csv": (92401, 90224, 2177),
}
TRAIN_SOURCE_FILES = ("train1.csv", "train2.csv", "train3.csv", "test1.csv", "test2.csv")
VALIDATION_SOURCE_FILES = ("test3.csv",)
HELDOUT_TEST_FILES = ("test4.csv", "test5.csv")
RATIO_SPECS: tuple[tuple[str, int], ...] = (
    ("ratio_1to1", 1),
    ("ratio_2to1", 2),
    ("ratio_3to1", 3),
    ("ratio_5to1", 5),
    ("ratio_8to1", 8),
    ("ratio_10to1", 10),
)
LEAKAGE_COLUMNS = {"time", "attack", "attack_p1", "attack_p2", "attack_p3", "attack_p4"}
TIMESTAMP_OR_ORDER_COLUMNS = {"time", "timestamp", "date"}
LABEL_COLUMN = "attack"
WINDOW_LENGTH = 32
STRIDE = 8
SEED = 42
NUM_CLIENTS = 12


@dataclass(frozen=True)
class ResolvedRawFile:
    logical_name: str
    path: Path
    read_from: str
    compression: str | None


@dataclass(frozen=True)
class WindowRecord:
    source_file: str
    start_row: int
    end_row: int
    label: int
    attack_row_count_inside_window: int
    sequence: int
    client_id: str | None = None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _read_initial_text(path: Path, limit: int = 256) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return handle.read(limit)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read(limit)


def _resolve_raw_file(raw_dir: Path, logical_name: str) -> ResolvedRawFile:
    csv_path = raw_dir / logical_name
    gz_path = raw_dir / f"{logical_name}.gz"
    if csv_path.exists():
        path = csv_path
        read_from = ".csv"
        compression = None
    elif gz_path.exists():
        path = gz_path
        read_from = ".csv.gz"
        compression = "gzip"
    else:
        raise FileNotFoundError(f"Missing required HAI file: {csv_path} or {gz_path}")

    initial_text = _read_initial_text(path)
    if initial_text.startswith("version https://git-lfs.github.com/spec/v1"):
        raise DatasetSchemaError(f"{path} is a Git LFS pointer, not the HAI dataset.")

    return ResolvedRawFile(
        logical_name=logical_name,
        path=path,
        read_from=read_from,
        compression=compression,
    )


def _read_header(path: Path) -> list[str]:
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        try:
            return [normalize_header(column) for column in next(reader)]
        except StopIteration as exc:
            raise DatasetSchemaError(f"Empty HAI CSV file: {path}") from exc


def _load_labels(path: Path, label_index: int) -> np.ndarray:
    labels = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=(label_index,),
        dtype=np.int8,
        encoding="utf-8",
        autostrip=True,
        invalid_raise=True,
    )
    labels = np.atleast_1d(labels).astype(np.int8, copy=False)
    invalid = labels[(labels != 0) & (labels != 1)]
    if invalid.size:
        raise DatasetSchemaError(f"Unexpected non-binary attack labels in {path}: {sorted(set(invalid.tolist()))}")
    return labels


def _load_feature_matrix(path: Path, feature_indices: Sequence[int]) -> np.ndarray:
    matrix = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=tuple(feature_indices),
        dtype=np.float32,
        encoding="utf-8",
        autostrip=True,
        missing_values=["", "na", "n/a", "nan", "null", "none"],
        filling_values=np.nan,
        invalid_raise=True,
    )
    if matrix.size == 0:
        return np.empty((0, len(feature_indices)), dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, len(feature_indices))
    return matrix.astype(np.float32, copy=False)


def _label_counts(labels: Sequence[int] | np.ndarray) -> dict[str, int]:
    counter = Counter(str(int(label)) for label in np.asarray(labels, dtype=np.int8).reshape(-1).tolist())
    return {key: int(counter[key]) for key in sorted(counter)}


def _window_counts(records: Sequence[WindowRecord]) -> dict[str, int]:
    return _label_counts([record.label for record in records])


def _client_sequence(client_id: str) -> int:
    return int(client_id.rsplit("L", 1)[1])


def _client_ids(num_clients: int) -> list[str]:
    return [f"C1_L{index + 1:03d}" for index in range(num_clients)]


def _is_leakage_column(column: str) -> bool:
    lower = column.lower()
    return lower in LEAKAGE_COLUMNS or lower in TIMESTAMP_OR_ORDER_COLUMNS or "attack" in lower


def _numeric_feature_columns(
    *,
    headers: Mapping[str, Sequence[str]],
    raw_files: Mapping[str, ResolvedRawFile],
    label_column: str,
) -> tuple[list[str], list[int]]:
    first_header = list(headers[EXPECTED_FILE_ORDER[0]])
    common = set(first_header)
    for header in headers.values():
        common &= set(header)
    if label_column not in common:
        raise DatasetSchemaError(f"HAI balanced variant requires label column {label_column!r} in every file.")

    candidate_features = [column for column in first_header if column in common and not _is_leakage_column(column)]
    feature_indices = [first_header.index(column) for column in candidate_features]
    if not feature_indices:
        raise DatasetSchemaError("No non-leakage HAI telemetry feature columns remain.")

    numeric_features: list[str] = []
    numeric_indices: list[int] = []
    for column, index in zip(candidate_features, feature_indices, strict=True):
        for resolved in raw_files.values():
            header = list(headers[resolved.logical_name])
            file_index = header.index(column)
            try:
                sample = np.genfromtxt(
                    resolved.path,
                    delimiter=",",
                    skip_header=1,
                    usecols=(file_index,),
                    dtype=np.float32,
                    encoding="utf-8",
                    autostrip=True,
                    max_rows=1024,
                    missing_values=["", "na", "n/a", "nan", "null", "none"],
                    filling_values=np.nan,
                    invalid_raise=True,
                )
            except ValueError:
                sample = np.asarray([], dtype=np.float32)
            if sample.size == 0 and EXPECTED_HAI_COUNTS.get(resolved.logical_name, (1, 0, 0))[0] > 0:
                break
        else:
            numeric_features.append(column)
            numeric_indices.append(index)

    if not numeric_features:
        raise DatasetSchemaError("No numeric HAI telemetry feature columns remain after leakage exclusion.")
    return numeric_features, numeric_indices


def _audit_raw_files(
    *,
    raw_dir: Path,
    output_root: Path,
    expected_counts: Mapping[str, tuple[int, int, int]] | None,
) -> tuple[dict[str, Any], dict[str, ResolvedRawFile], dict[str, list[str]], dict[str, np.ndarray]]:
    raw_files = {name: _resolve_raw_file(raw_dir, name) for name in EXPECTED_FILE_ORDER}
    headers = {name: _read_header(resolved.path) for name, resolved in raw_files.items()}

    union_columns = sorted({column for header in headers.values() for column in header})
    common_columns = sorted(set(headers[EXPECTED_FILE_ORDER[0]]).intersection(*(set(headers[name]) for name in EXPECTED_FILE_ORDER[1:])))
    label_arrays: dict[str, np.ndarray] = {}
    files: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for name in EXPECTED_FILE_ORDER:
        resolved = raw_files[name]
        header = headers[name]
        if LABEL_COLUMN not in header:
            raise DatasetSchemaError(f"{name}: missing required label column {LABEL_COLUMN!r}.")
        labels = _load_labels(resolved.path, header.index(LABEL_COLUMN))
        label_arrays[name] = labels
        total = int(labels.shape[0])
        attack_0 = int(np.sum(labels == 0))
        attack_1 = int(np.sum(labels == 1))
        expected = expected_counts.get(name) if expected_counts is not None else None
        matches_expected = expected is None or (total, attack_0, attack_1) == tuple(expected)
        if not matches_expected:
            mismatches.append(
                {
                    "file": name,
                    "expected": {
                        "rows": int(expected[0]),
                        "attack_0": int(expected[1]),
                        "attack_1": int(expected[2]),
                    },
                    "observed": {"rows": total, "attack_0": attack_0, "attack_1": attack_1},
                }
            )
        files.append(
            {
                "file": name,
                "path": str(resolved.path),
                "read_from": resolved.read_from,
                "compression": resolved.compression,
                "row_count": total,
                "attack_0_count": attack_0,
                "attack_1_count": attack_1,
                "columns_present": list(header),
                "missing_columns_compared_with_union": [column for column in union_columns if column not in header],
                "columns_not_in_common_schema": [column for column in header if column not in common_columns],
                "matches_expected_counts": matches_expected,
            }
        )

    status = "ok" if not mismatches else "count_mismatch"
    audit = {
        "status": status,
        "label_column": LABEL_COLUMN,
        "raw_dir": str(raw_dir),
        "files": files,
        "union_columns": union_columns,
        "common_columns": common_columns,
        "schema_consistent_across_files": all(not item["missing_columns_compared_with_union"] for item in files),
        "count_mismatches": mismatches,
    }
    reports_dir = output_root / "reports"
    _write_json(reports_dir / "hai_file_audit.json", audit)
    _write_raw_audit_md(reports_dir / "hai_file_audit.md", audit)
    return audit, raw_files, headers, label_arrays


def _write_raw_audit_md(path: Path, audit: Mapping[str, Any]) -> None:
    lines = [
        "# HAI 21.03 Raw File Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "| file | read_from | rows | attack=0 | attack=1 | missing vs union | not in common |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for entry in audit["files"]:
        lines.append(
            f"| {entry['file']} | {entry['read_from']} | {entry['row_count']} | "
            f"{entry['attack_0_count']} | {entry['attack_1_count']} | "
            f"{entry['missing_columns_compared_with_union']} | {entry['columns_not_in_common_schema']} |"
        )
    lines.extend(["", "## Columns", ""])
    for entry in audit["files"]:
        lines.append(f"### {entry['file']}")
        lines.append("")
        lines.append(", ".join(f"`{column}`" for column in entry["columns_present"]))
        lines.append("")
    if audit["count_mismatches"]:
        lines.extend(["## Count Mismatches", "", "Generation stopped because at least one raw file count did not match."])
    _write_markdown(path, lines)


def _build_windows_for_labels(labels: np.ndarray, source_file: str, *, start_sequence: int) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    sequence = start_sequence
    if labels.shape[0] < WINDOW_LENGTH:
        return records
    for start in range(0, labels.shape[0] - WINDOW_LENGTH + 1, STRIDE):
        stop = start + WINDOW_LENGTH
        attack_count = int(np.sum(labels[start:stop] == 1))
        records.append(
            WindowRecord(
                source_file=source_file,
                start_row=int(start),
                end_row=int(stop - 1),
                label=int(attack_count > 0),
                attack_row_count_inside_window=attack_count,
                sequence=sequence,
            )
        )
        sequence += 1
    return records


def _build_window_audit(
    *,
    labels_by_file: Mapping[str, np.ndarray],
    feature_columns: Sequence[str],
    output_root: Path,
) -> tuple[dict[str, list[WindowRecord]], dict[str, Any]]:
    windows_by_file: dict[str, list[WindowRecord]] = {}
    sequence = 0
    files: list[dict[str, Any]] = []
    for name in EXPECTED_FILE_ORDER:
        records = _build_windows_for_labels(labels_by_file[name], name, start_sequence=sequence)
        sequence += len(records)
        windows_by_file[name] = records
        counts = _window_counts(records)
        files.append(
            {
                "file": name,
                "window_count": len(records),
                "positive_windows": int(counts.get("1", 0)),
                "negative_windows": int(counts.get("0", 0)),
            }
        )
    audit = {
        "status": "ok",
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "label_rule": "any_positive_row",
        "no_window_crosses_file_boundary": True,
        "files": files,
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
    }
    reports_dir = output_root / "reports"
    _write_json(reports_dir / "window_audit.json", audit)
    _write_window_audit_md(reports_dir / "window_audit.md", audit)
    return windows_by_file, audit


def _write_window_audit_md(path: Path, audit: Mapping[str, Any]) -> None:
    lines = [
        "# HAI 21.03 Window Audit",
        "",
        f"Window length: `{audit['window_length']}`",
        f"Stride: `{audit['stride']}`",
        f"Label rule: `{audit['label_rule']}`",
        "",
        "| file | windows | positive | negative |",
        "|---|---:|---:|---:|",
    ]
    for entry in audit["files"]:
        lines.append(
            f"| {entry['file']} | {entry['window_count']} | {entry['positive_windows']} | {entry['negative_windows']} |"
        )
    lines.extend(
        [
            "",
            f"Feature count: `{audit['feature_count']}`",
            "",
            "## Final Feature Columns",
            "",
            ", ".join(f"`{column}`" for column in audit["feature_columns"]),
        ]
    )
    _write_markdown(path, lines)


def _split_records(windows_by_file: Mapping[str, Sequence[WindowRecord]]) -> tuple[list[WindowRecord], list[WindowRecord], list[WindowRecord]]:
    train_records = [record for name in TRAIN_SOURCE_FILES for record in windows_by_file[name]]
    val_records = [record for name in VALIDATION_SOURCE_FILES for record in windows_by_file[name]]
    test_records = [record for name in HELDOUT_TEST_FILES for record in windows_by_file[name]]
    return train_records, val_records, test_records


def _assign_clients(
    records: Sequence[WindowRecord],
    *,
    num_clients: int,
    seed: int,
    shuffle_each_client: bool,
) -> list[WindowRecord]:
    clients: list[list[WindowRecord]] = [[] for _ in range(num_clients)]
    positives = [record for record in records if record.label == 1]
    negatives = [record for record in records if record.label == 0]
    client_ids = _client_ids(num_clients)

    for index, record in enumerate(positives):
        client_index = index % num_clients
        clients[client_index].append(replace(record, client_id=client_ids[client_index]))
    for index, record in enumerate(negatives):
        client_index = index % num_clients
        clients[client_index].append(replace(record, client_id=client_ids[client_index]))

    if shuffle_each_client:
        for client_index, client_records in enumerate(clients):
            rng = np.random.default_rng(seed + client_index)
            order = rng.permutation(len(client_records))
            clients[client_index] = [client_records[int(item)] for item in order]

    return [record for client_records in clients for record in client_records]


def _records_to_arrays(records: Sequence[WindowRecord]) -> dict[str, np.ndarray]:
    return {
        "labels": np.asarray([record.label for record in records], dtype=np.int8),
        "source_file": np.asarray([record.source_file for record in records]),
        "start_row": np.asarray([record.start_row for record in records], dtype=np.int64),
        "end_row": np.asarray([record.end_row for record in records], dtype=np.int64),
        "attack_row_count_inside_window": np.asarray(
            [record.attack_row_count_inside_window for record in records],
            dtype=np.int16,
        ),
        "client_id": np.asarray([record.client_id or "" for record in records]),
    }


def _assemble_inputs(
    records: Sequence[WindowRecord],
    *,
    raw_files: Mapping[str, ResolvedRawFile],
    headers: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str],
) -> np.ndarray:
    if not records:
        return np.empty((0, len(feature_columns), WINDOW_LENGTH), dtype=np.float32)

    output = np.empty((len(records), len(feature_columns), WINDOW_LENGTH), dtype=np.float32)
    records_by_file: dict[str, list[tuple[int, WindowRecord]]] = {}
    for output_index, record in enumerate(records):
        records_by_file.setdefault(record.source_file, []).append((output_index, record))

    for source_file, indexed_records in records_by_file.items():
        header = list(headers[source_file])
        feature_indices = [header.index(column) for column in feature_columns]
        feature_matrix = _load_feature_matrix(raw_files[source_file].path, feature_indices)
        for output_index, record in indexed_records:
            output[output_index] = feature_matrix[record.start_row : record.end_row + 1].T

    return output


def _flatten_for_preprocessing(inputs: np.ndarray) -> np.ndarray:
    if inputs.size == 0:
        return np.empty((0, inputs.shape[1] if inputs.ndim == 3 else 0), dtype=np.float32)
    return inputs.transpose(0, 2, 1).reshape(-1, inputs.shape[1])


def _transform_inputs(
    inputs: np.ndarray,
    *,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    if inputs.size == 0:
        return inputs.astype(np.float32, copy=True)
    flat = _flatten_for_preprocessing(inputs)
    transformed = scaler.transform(imputer.transform(flat)).astype(np.float32, copy=False)
    return transformed.reshape(inputs.shape[0], WINDOW_LENGTH, inputs.shape[1]).transpose(0, 2, 1)


def _save_window_npz(path: Path, *, inputs: np.ndarray, records: Sequence[WindowRecord]) -> None:
    arrays = _records_to_arrays(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        inputs=inputs.astype(np.float32, copy=False),
        labels=arrays["labels"],
        source_file=arrays["source_file"],
        start_row=arrays["start_row"],
        end_row=arrays["end_row"],
        attack_row_count_inside_window=arrays["attack_row_count_inside_window"],
        client_id=arrays["client_id"],
    )


def _save_client_metadata(
    *,
    ratio_name: str,
    output_root: Path,
    train_records: Sequence[WindowRecord],
    val_records: Sequence[WindowRecord],
    test_records: Sequence[WindowRecord],
    num_clients: int,
) -> dict[str, Any]:
    clients: list[dict[str, Any]] = []
    single_class: list[dict[str, Any]] = []
    for client_id in _client_ids(num_clients):
        client_train = [record for record in train_records if record.client_id == client_id]
        client_val = [record for record in val_records if record.client_id == client_id]
        client_test = [record for record in test_records if record.client_id == client_id]
        train_counts = _window_counts(client_train)
        val_counts = _window_counts(client_val)
        test_counts = _window_counts(client_test)
        warning = "attack-free training client" if train_counts.get("1", 0) == 0 else ""
        entry = {
            "client_id": client_id,
            "train_window_count": len(client_train),
            "train_positive_count": int(train_counts.get("1", 0)),
            "train_negative_count": int(train_counts.get("0", 0)),
            "val_window_count": len(client_val),
            "val_positive_count": int(val_counts.get("1", 0)),
            "val_negative_count": int(val_counts.get("0", 0)),
            "test_window_count": len(client_test),
            "test_positive_count": int(test_counts.get("1", 0)),
            "test_negative_count": int(test_counts.get("0", 0)),
            "source_files_represented": sorted({record.source_file for record in client_train + client_val + client_test}),
            "train_source_files": sorted({record.source_file for record in client_train}),
            "val_source_files": sorted({record.source_file for record in client_val}),
            "test_source_files": sorted({record.source_file for record in client_test}),
            "warning": warning,
        }
        if warning:
            single_class.append({"client_id": client_id, "split": "train", "label_counts": train_counts})
        clients.append(entry)

    payload = {
        "cluster_id": 1,
        "dataset": "HAI 21.03 balanced-training variant",
        "variant": "cluster1_balanced_training",
        "ratio": ratio_name,
        "num_leaf_clients": num_clients,
        "train_source_files": list(TRAIN_SOURCE_FILES),
        "validation_source_files": list(VALIDATION_SOURCE_FILES),
        "heldout_test_source_files": list(HELDOUT_TEST_FILES),
        "partitioning_strategy": "balanced_window_round_robin_positive_then_negative",
        "validation_partitioning_strategy": "natural_validation_windows_round_robin_by_label",
        "heldout_test_partitioning_strategy": "natural_heldout_windows_round_robin_by_label",
        "cluster_split_label_counts": {
            "train": _window_counts(train_records),
            "validation": _window_counts(val_records),
            "test": _window_counts(test_records),
        },
        "single_class_local_partitions": single_class,
        "clients": clients,
    }
    _write_json(output_root / ratio_name / "clients" / "cluster1_leaf_clients.json", payload)
    return payload


def _agglomerative_model(n_subclusters: int) -> AgglomerativeClustering:
    try:
        return AgglomerativeClustering(n_clusters=n_subclusters, linkage="ward", metric="euclidean")
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_subclusters, linkage="ward", affinity="euclidean")


def _normalize_memberships(client_ids: Sequence[str], raw_labels: Sequence[int], fixed_ids: Sequence[str]) -> dict[str, str]:
    by_label: dict[int, list[str]] = {}
    for client_id, raw_label in zip(client_ids, raw_labels, strict=True):
        by_label.setdefault(int(raw_label), []).append(client_id)
    sorted_labels = sorted(by_label, key=lambda label: min(_client_sequence(client_id) for client_id in by_label[label]))
    return {
        client_id: fixed_ids[sorted_labels.index(int(raw_label))]
        for client_id, raw_label in zip(client_ids, raw_labels, strict=True)
    }


def _write_clustering(
    *,
    ratio_name: str,
    output_root: Path,
    train_inputs: np.ndarray,
    train_records: Sequence[WindowRecord],
    feature_columns: Sequence[str],
    num_clients: int,
) -> dict[str, Any]:
    client_ids = _client_ids(num_clients)
    descriptors: list[np.ndarray] = []
    descriptor_payload: dict[str, Any] = {
        "cluster_id": 1,
        "dataset": "HAI 21.03 balanced-training variant",
        "variant": "cluster1_balanced_training",
        "ratio": ratio_name,
        "descriptor": "feature_wise_mean_std",
        "descriptor_source_split": "train",
        "feature_columns": list(feature_columns),
        "clients": [],
    }
    record_client_ids = np.asarray([record.client_id for record in train_records])
    for client_id in client_ids:
        mask = record_client_ids == client_id
        if not bool(np.any(mask)):
            raise DatasetSchemaError(f"{ratio_name}: {client_id} has no training windows for clustering.")
        client_inputs = train_inputs[mask]
        mean = client_inputs.mean(axis=(0, 2), dtype=np.float64)
        std = client_inputs.std(axis=(0, 2), dtype=np.float64)
        descriptor = np.concatenate([mean, std]).astype(np.float32, copy=False)
        descriptors.append(descriptor)
        descriptor_payload["clients"].append(
            {
                "client_id": client_id,
                "train_window_count": int(mask.sum()),
                "descriptor": descriptor.astype(float).tolist(),
            }
        )

    descriptor_matrix = np.vstack(descriptors)
    raw_labels = _agglomerative_model(2).fit_predict(descriptor_matrix).tolist()
    assignments = _normalize_memberships(client_ids, raw_labels, ("H1", "H2"))
    clients_for_membership = [
        {"client_id": client_id, "subcluster_id": assignments[client_id]}
        for client_id in client_ids
    ]
    membership_hash = hashlib.sha256(json.dumps(clients_for_membership, sort_keys=True).encode("utf-8")).hexdigest()
    membership = {
        "cluster_id": 1,
        "dataset": "HAI 21.03 balanced-training variant",
        "variant": "cluster1_balanced_training",
        "ratio": ratio_name,
        "status": "ok",
        "clustering_method": "AgglomerativeClustering",
        "linkage": "ward",
        "metric": "euclidean",
        "descriptor": "feature_wise_mean_std",
        "descriptor_dim": int(descriptor_matrix.shape[1]),
        "descriptor_source_split": "train",
        "n_subclusters": 2,
        "fixed_subcluster_ids": ["H1", "H2"],
        "frozen": True,
        "membership_hash": membership_hash,
        "client_metadata_path": str(output_root / ratio_name / "clients" / "cluster1_leaf_clients.json"),
        "membership_file": str(output_root / ratio_name / "clustering" / "cluster1_memberships.json"),
        "reuse_for_experiment_groups": [
            "baseline_uniform_hierarchical",
            "proposed_specialized_hierarchical",
        ],
        "subclusters": [
            {
                "subcluster_id": subcluster_id,
                "client_ids": [client_id for client_id in client_ids if assignments[client_id] == subcluster_id],
            }
            for subcluster_id in ("H1", "H2")
        ],
        "clients": clients_for_membership,
    }
    clustering_dir = output_root / ratio_name / "clustering"
    _write_json(clustering_dir / "client_descriptors.json", descriptor_payload)
    _write_json(clustering_dir / "cluster1_memberships.json", membership)
    return membership


def _write_configs(
    *,
    config_dir: Path,
    output_root: Path,
    variant_root: Path,
    ratios: Sequence[tuple[str, int]],
) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    for ratio_name, ratio_value in ratios:
        short = ratio_name.removeprefix("ratio_")
        upper = short.upper()
        cluster_config = {
            "config_version": 1,
            "cluster": {
                "id": 1,
                "key": "C1",
                "dataset_key": f"HAI_2103_BALANCED_{upper}",
                "dataset_name": f"HAI 21.03 balanced-training variant {short}",
                "audit_report": str(output_root / "reports" / "hai_file_audit.json"),
            },
            "data": {
                "data_root_env_var": "FCFL_DATA_ROOT",
                "default_data_root": "data",
                "current_raw_input_dir": "data/raw/hai_2103/hai-21.03",
                "current_raw_files": list(EXPECTED_FILE_ORDER),
                "balanced_variant_dir": str(variant_root / ratio_name),
                "balanced_output_root": str(output_root / ratio_name),
                "training_input_mode": "raw_csv_glob",
                "training_input_glob": "data/raw/hai_2103/hai-21.03/*.csv",
                "schema_consistent_across_files": True,
                "label_column": LABEL_COLUMN,
                "label_column_confirmed_from_audit": True,
                "candidate_label_columns_present": [LABEL_COLUMN],
                "timestamp_or_order_columns": ["time"],
                "excluded_columns": ["attack", "time", "attack_P1", "attack_P2", "attack_P3"],
                "exclude_if_present": [
                    "attack_P4",
                    "Attack",
                    "label",
                    "Label",
                    "target",
                    "Target",
                    "timestamp",
                    "Timestamp",
                    "Time",
                    "date",
                    "Date",
                ],
            },
            "partitioning": {
                "candidate_leaf_clients": NUM_CLIENTS,
                "strategy": "balanced_window_npz",
                "train_source_files": list(TRAIN_SOURCE_FILES),
                "validation_source_files": list(VALIDATION_SOURCE_FILES),
                "heldout_test_files": list(HELDOUT_TEST_FILES),
                "negative_to_positive_ratio": ratio_value,
            },
            "clustering": {
                "fixed_subclusters": 2,
                "fixed_subcluster_ids": ["H1", "H2"],
                "membership_file": str(output_root / ratio_name / "clustering" / "cluster1_memberships.json"),
            },
            "preprocessing": {
                "input_type": "multivariate_time_series",
                "window_length": WINDOW_LENGTH,
                "stride": STRIDE,
                "window_label_rule": "any_positive_row",
                "preprocessed_npz": True,
            },
            "runtime_validation": {
                "require_training_input_to_exist": True,
                "require_schema_consistency_across_files": True,
                "require_label_column_to_exist": True,
                "error_on_missing_label_column": "CONFIGURED_LABEL_COLUMN_MISSING",
                "heldout_test_files_must_not_be_used_for_training": True,
                "heldout_test_files_must_not_be_used_for_validation": True,
                "fit_preprocessing_on_training_windows_only": True,
            },
        }
        (config_dir / f"cluster1_hai_balanced_{short}.yaml").write_text(
            yaml.safe_dump(cluster_config, sort_keys=False),
            encoding="utf-8",
        )

        proposed_config = _experiment_config(
            experiment_group="proposed_specialized_hierarchical",
            description=f"Cluster 1 balanced-training variant {short}: proposed CNN1D-BN + FedBN run.",
            experiment_id=f"P_C1_BAL_{upper}",
            cluster_config=f"configs/cluster1_hai_balanced_{short}.yaml",
            membership_file=str(output_root / ratio_name / "clustering" / "cluster1_memberships.json"),
            model_family="cnn1d_bn",
            fl_method="FedBN",
            aggregation="weighted_non_bn_mean",
            output_root=str(output_root / ratio_name),
        )
        (config_dir / f"proposed_cluster1_balanced_{short}.yaml").write_text(
            yaml.safe_dump(proposed_config, sort_keys=False),
            encoding="utf-8",
        )

        baseline_config = _experiment_config(
            experiment_group="baseline_uniform_hierarchical",
            description=f"Cluster 1 balanced-training variant {short}: hierarchical CNN1D + FedAvg baseline.",
            experiment_id=f"B_C1_BAL_{upper}",
            cluster_config=f"configs/cluster1_hai_balanced_{short}.yaml",
            membership_file=str(output_root / ratio_name / "clustering" / "cluster1_memberships.json"),
            model_family="cnn1d",
            fl_method="FedAvg",
            aggregation="weighted_arithmetic_mean",
            output_root=str(output_root / ratio_name),
        )
        (config_dir / f"baseline_hierarchical_cluster1_balanced_{short}.yaml").write_text(
            yaml.safe_dump(baseline_config, sort_keys=False),
            encoding="utf-8",
        )


def _experiment_config(
    *,
    experiment_group: str,
    description: str,
    experiment_id: str,
    cluster_config: str,
    membership_file: str,
    model_family: str,
    fl_method: str,
    aggregation: str,
    output_root: str,
) -> dict[str, Any]:
    return {
        "config_version": 1,
        "experiment_group": experiment_group,
        "description": description,
        "source_of_truth": {
            "architecture_contract": "docs/ARCHITECTURE_CONTRACT.md",
            "data_contract": "docs/DATA_CONTRACT.md",
            "balanced_training_variant": "docs/CLUSTER1_BALANCED_TRAINING_VARIANT.md",
        },
        "ledger": {"metadata_only": True},
        "training_defaults": {
            "rounds": 50,
            "local_epochs": 1,
            "batch_size": 128,
            "seeds": [42],
        },
        "smoke_test_defaults": {
            "rounds": 2,
            "local_epochs": 1,
            "batch_size": 64,
            "seed": 42,
        },
        "clustering_defaults": {
            "method": "AgglomerativeClustering",
            "linkage": "ward",
            "metric": "euclidean",
            "descriptor": "feature_wise_mean_std",
            "reuse_frozen_memberships": True,
        },
        "clusters": [
            {
                "experiment_id": experiment_id,
                "cluster_config": cluster_config,
                "hierarchy": "hierarchical_fixed",
                "clustering_method": "agglomerative",
                "n_subclusters": 2,
                "descriptor": "feature_wise_mean_std",
                "membership_file": membership_file,
                "model_family": model_family,
                "fl_method": fl_method,
                "aggregation": aggregation,
                "output_root": output_root,
            }
        ],
        "runtime_validation": {
            "require_cluster_configs_to_exist": True,
            "require_frozen_membership_files_before_training": True,
            "require_ledger_metadata_only": True,
            "require_balanced_cluster1_outputs": True,
        },
    }


def _write_initial_doc(doc_path: Path) -> None:
    lines = [
        "# Cluster 1 Balanced-Training Variant",
        "",
        "This document is updated after the balanced Cluster 1 runs complete.",
        "",
        "The variant keeps the fixed FCFL architecture unchanged and changes only Cluster 1 dataset construction, balanced training-window sampling, preprocessing, validation, and threshold selection.",
        "",
        "Key leakage controls:",
        "- `test3.csv` is validation and threshold/ratio selection only.",
        "- `test4.csv` and `test5.csv` are held-out test only.",
        "- Validation and held-out test windows are not balanced.",
        "- Imputation and scaling are fitted only on balanced training windows.",
        "- Held-out test labels and predictions are not used for threshold or ratio selection.",
    ]
    _write_markdown(doc_path, lines)


def prepare_cluster1_balanced_train(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant_root: str | Path = DEFAULT_VARIANT_ROOT,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    doc_path: str | Path = DEFAULT_DOC_PATH,
    ratios: Sequence[tuple[str, int]] = RATIO_SPECS,
    expected_counts: Mapping[str, tuple[int, int, int]] | None = EXPECTED_HAI_COUNTS,
    num_clients: int = NUM_CLIENTS,
    seed: int = SEED,
    write_configs: bool = True,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    output_root = Path(output_root)
    variant_root = Path(variant_root)
    config_dir = Path(config_dir)
    doc_path = Path(doc_path)

    output_root.mkdir(parents=True, exist_ok=True)
    variant_root.mkdir(parents=True, exist_ok=True)

    raw_audit, raw_files, headers, labels_by_file = _audit_raw_files(
        raw_dir=raw_dir,
        output_root=output_root,
        expected_counts=expected_counts,
    )
    if raw_audit["status"] != "ok":
        return {"status": "count_mismatch", "raw_audit": raw_audit}

    feature_columns, _ = _numeric_feature_columns(
        headers=headers,
        raw_files=raw_files,
        label_column=LABEL_COLUMN,
    )
    windows_by_file, window_audit = _build_window_audit(
        labels_by_file=labels_by_file,
        feature_columns=feature_columns,
        output_root=output_root,
    )
    train_pool, val_pool, test_pool = _split_records(windows_by_file)
    train_positive = [record for record in train_pool if record.label == 1]
    train_negative = [record for record in train_pool if record.label == 0]
    if not train_positive:
        raise DatasetSchemaError("Cluster 1 balanced variant found no positive training windows in test1/test2.")
    if not train_negative:
        raise DatasetSchemaError("Cluster 1 balanced variant found no negative training windows.")

    ratio_results: dict[str, Any] = {}
    skipped_ratios: list[dict[str, Any]] = []
    for ratio_name, ratio_value in ratios:
        needed_negatives = ratio_value * len(train_positive)
        if needed_negatives > len(train_negative):
            skipped_ratios.append(
                {
                    "ratio": ratio_name,
                    "needed_negative_windows": needed_negatives,
                    "available_negative_windows": len(train_negative),
                }
            )
            continue

        rng = np.random.default_rng(seed)
        sampled_negative_indices = rng.choice(len(train_negative), size=needed_negatives, replace=False)
        sampled_negatives = [train_negative[int(index)] for index in sorted(sampled_negative_indices.tolist())]
        selected_train = train_positive + sampled_negatives
        train_records = _assign_clients(
            selected_train,
            num_clients=num_clients,
            seed=seed,
            shuffle_each_client=True,
        )
        val_records = _assign_clients(
            val_pool,
            num_clients=num_clients,
            seed=seed,
            shuffle_each_client=False,
        )
        test_records = _assign_clients(
            test_pool,
            num_clients=num_clients,
            seed=seed,
            shuffle_each_client=False,
        )

        raw_train_inputs = _assemble_inputs(
            train_records,
            raw_files=raw_files,
            headers=headers,
            feature_columns=feature_columns,
        )
        raw_val_inputs = _assemble_inputs(
            val_records,
            raw_files=raw_files,
            headers=headers,
            feature_columns=feature_columns,
        )
        raw_test_inputs = _assemble_inputs(
            test_records,
            raw_files=raw_files,
            headers=headers,
            feature_columns=feature_columns,
        )

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_flat = _flatten_for_preprocessing(raw_train_inputs)
        imputed_train_flat = imputer.fit_transform(train_flat)
        scaler.fit(imputed_train_flat)
        train_inputs = scaler.transform(imputed_train_flat).astype(np.float32, copy=False).reshape(
            raw_train_inputs.shape[0],
            WINDOW_LENGTH,
            raw_train_inputs.shape[1],
        ).transpose(0, 2, 1)
        val_inputs = _transform_inputs(raw_val_inputs, imputer=imputer, scaler=scaler)
        test_inputs = _transform_inputs(raw_test_inputs, imputer=imputer, scaler=scaler)

        ratio_variant_dir = variant_root / ratio_name
        _save_window_npz(ratio_variant_dir / "train_windows.npz", inputs=train_inputs, records=train_records)
        _save_window_npz(ratio_variant_dir / "val_windows.npz", inputs=val_inputs, records=val_records)
        _save_window_npz(ratio_variant_dir / "test_windows.npz", inputs=test_inputs, records=test_records)

        preprocessing_dir = output_root / ratio_name / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        imputer_path = preprocessing_dir / "imputer.joblib"
        scaler_path = preprocessing_dir / "scaler.joblib"
        joblib.dump(imputer, imputer_path)
        joblib.dump(scaler, scaler_path)

        train_counts = _window_counts(train_records)
        val_counts = _window_counts(val_records)
        test_counts = _window_counts(test_records)
        positive_class_weight = float(train_counts.get("0", 0) / train_counts.get("1", 1))
        preprocessing_summary = {
            "cluster_id": 1,
            "dataset": "HAI 21.03 balanced-training variant",
            "variant": "cluster1_balanced_training",
            "ratio": ratio_name,
            "status": "ok",
            "fit_scope": "balanced_training_windows_only",
            "imputer": "SimpleImputer(strategy='median')",
            "scaler": "StandardScaler",
            "fit_shape": [int(train_flat.shape[0]), int(train_flat.shape[1])],
            "input_tensor_layout": "N,C,T",
            "window_length": WINDOW_LENGTH,
            "feature_count": len(feature_columns),
            "feature_columns": list(feature_columns),
            "positive_class_weight": positive_class_weight,
            "artifact_paths": {
                "imputer": str(imputer_path),
                "scaler": str(scaler_path),
            },
            "leakage_prevention": {
                "fit_on_validation_windows": False,
                "fit_on_heldout_test_windows": False,
                "threshold_tuning_on_heldout_test": False,
            },
        }
        _write_json(preprocessing_dir / "preprocessing_summary.json", preprocessing_summary)
        clients = _save_client_metadata(
            ratio_name=ratio_name,
            output_root=output_root,
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            num_clients=num_clients,
        )
        membership = _write_clustering(
            ratio_name=ratio_name,
            output_root=output_root,
            train_inputs=train_inputs,
            train_records=train_records,
            feature_columns=feature_columns,
            num_clients=num_clients,
        )

        metadata = {
            "cluster_id": 1,
            "dataset": "HAI 21.03",
            "variant": "cluster1_balanced_training",
            "ratio": ratio_name,
            "negative_to_positive_ratio": ratio_value,
            "seed": seed,
            "window_length": WINDOW_LENGTH,
            "stride": STRIDE,
            "label_rule": "any_positive_row",
            "input_tensor_layout": "N,C,T",
            "preprocessing_applied_to_npz_inputs": True,
            "train_source_files": list(TRAIN_SOURCE_FILES),
            "validation_source_files": list(VALIDATION_SOURCE_FILES),
            "heldout_test_source_files": list(HELDOUT_TEST_FILES),
            "class_counts": {
                "train": train_counts,
                "validation": val_counts,
                "test": test_counts,
            },
            "positive_class_weight": positive_class_weight,
            "feature_count": len(feature_columns),
            "feature_columns": list(feature_columns),
            "paths": {
                "train_windows": str(ratio_variant_dir / "train_windows.npz"),
                "val_windows": str(ratio_variant_dir / "val_windows.npz"),
                "test_windows": str(ratio_variant_dir / "test_windows.npz"),
                "client_metadata": str(output_root / ratio_name / "clients" / "cluster1_leaf_clients.json"),
                "membership": str(output_root / ratio_name / "clustering" / "cluster1_memberships.json"),
                "preprocessing_summary": str(preprocessing_dir / "preprocessing_summary.json"),
            },
            "leakage_prevention": {
                "test3_used_for_training": False,
                "test4_test5_used_for_training": False,
                "test4_test5_used_for_validation": False,
                "test4_test5_used_for_scaling_or_imputation": False,
                "test4_test5_used_for_clustering": False,
                "validation_balanced": False,
                "heldout_test_balanced": False,
            },
        }
        _write_json(ratio_variant_dir / "metadata.json", metadata)
        ratio_results[ratio_name] = {
            "metadata": metadata,
            "clients": clients,
            "membership": membership,
            "preprocessing_summary": preprocessing_summary,
        }

    if write_configs:
        _write_configs(
            config_dir=config_dir,
            output_root=output_root,
            variant_root=variant_root,
            ratios=ratios,
        )
        _write_initial_doc(doc_path)

    summary = {
        "status": "ok",
        "raw_audit_path": str(output_root / "reports" / "hai_file_audit.json"),
        "window_audit_path": str(output_root / "reports" / "window_audit.json"),
        "variant_root": str(variant_root),
        "output_root": str(output_root),
        "ratios_generated": sorted(ratio_results),
        "ratios_skipped": skipped_ratios,
        "train_pool_counts_before_balancing": _window_counts(train_pool),
        "validation_counts": _window_counts(val_pool),
        "heldout_test_counts": _window_counts(test_pool),
        "feature_count": len(feature_columns),
    }
    _write_json(output_root / "reports" / "balanced_variant_generation_summary.json", summary)
    return {
        "status": "ok",
        "raw_audit": raw_audit,
        "window_audit": window_audit,
        "summary": summary,
        "ratios": ratio_results,
    }
