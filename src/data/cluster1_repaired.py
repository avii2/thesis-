from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from src.data.schema_validation import (
    DatasetConfigError,
    DatasetSchemaError,
    label_value_to_binary,
    load_cluster_config,
    normalize_header,
    resolve_configured_path,
)
from src.data.transforms import NumericTransformArtifacts


DEFAULT_CONFIG_PATH = Path("configs/cluster1_hai_repaired.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs_c1_repaired")


@dataclass(frozen=True)
class WindowRef:
    file_name: str
    start: int
    stop: int
    label: int
    sequence: int


@dataclass(frozen=True)
class LoadedFile:
    file_name: str
    path: Path
    feature_matrix: np.ndarray
    labels: np.ndarray


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise DatasetConfigError(f"Expected mapping in YAML file: {path}")
    return data


def _require_string_list(parent: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = parent.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise DatasetConfigError(f"Cluster 1 repaired config requires non-empty data.{key}.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DatasetConfigError(f"Cluster 1 repaired config has invalid data.{key} item.")
        result.append(item)
    return tuple(result)


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return [normalize_header(column) for column in next(reader)]
        except StopIteration as exc:
            raise DatasetSchemaError(f"Cluster 1 repaired: empty CSV file: {path}") from exc


def _mapped_label_array(path: Path, label_index: int) -> np.ndarray:
    raw_labels = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=(label_index,),
        dtype=str,
        encoding="utf-8",
        autostrip=True,
    )
    raw_labels = np.atleast_1d(raw_labels)
    if raw_labels.size == 0:
        return np.empty(0, dtype=np.int8)

    mapped = np.empty(raw_labels.shape[0], dtype=np.int8)
    for index, raw_label in enumerate(raw_labels):
        value = label_value_to_binary(str(raw_label))
        if value is None:
            raise DatasetSchemaError(
                f"Cluster 1 repaired: unexpected label value {raw_label!r} in {path}."
            )
        mapped[index] = int(value)
    return mapped


def _load_feature_matrix(path: Path, feature_indices: Sequence[int]) -> np.ndarray:
    features = np.genfromtxt(
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
    if features.size == 0:
        return np.empty((0, len(feature_indices)), dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, len(feature_indices))
    return features.astype(np.float32, copy=False)


def _label_counts(labels: Sequence[int] | np.ndarray) -> dict[str, int]:
    counter = Counter(str(int(label)) for label in np.asarray(labels, dtype=np.int8).tolist())
    return {key: counter[key] for key in sorted(counter)}


def _file_label_counts(path: Path, label_column: str) -> dict[str, Any]:
    header = _read_header(path)
    if label_column not in header:
        raise DatasetSchemaError(f"Cluster 1 repaired: missing {label_column!r} in {path}.")
    labels = _mapped_label_array(path, header.index(label_column))
    return {
        "path": str(path),
        "row_count": int(labels.shape[0]),
        "label_counts": _label_counts(labels),
        "contains_both_classes": set(labels.tolist()) == {0, 1},
    }


def _build_windows(files: Sequence[LoadedFile], *, window_length: int, stride: int) -> list[WindowRef]:
    windows: list[WindowRef] = []
    sequence = 0
    for loaded in files:
        if loaded.labels.shape[0] < window_length:
            continue
        for start in range(0, loaded.labels.shape[0] - window_length + 1, stride):
            stop = start + window_length
            label = int(loaded.labels[start:stop].max() > 0)
            windows.append(
                WindowRef(
                    file_name=loaded.file_name,
                    start=start,
                    stop=stop,
                    label=label,
                    sequence=sequence,
                )
            )
            sequence += 1
    return windows


def _balanced_counts(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def attack_aware_client_assignment(
    windows: Sequence[WindowRef],
    *,
    num_clients: int,
) -> list[list[WindowRef]]:
    positives = [window for window in windows if window.label == 1]
    negatives = [window for window in windows if window.label == 0]
    clients: list[list[WindowRef]] = [[] for _ in range(num_clients)]

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


def split_client_train_validation(
    client_windows: Sequence[WindowRef],
    *,
    validation_ratio: float,
) -> tuple[list[WindowRef], list[WindowRef]]:
    positives = [window for window in client_windows if window.label == 1]
    negatives = [window for window in client_windows if window.label == 0]

    def split_group(group: list[WindowRef]) -> tuple[list[WindowRef], list[WindowRef]]:
        if len(group) <= 1:
            return group, []
        val_count = int(round(len(group) * validation_ratio))
        val_count = max(1, val_count)
        val_count = min(val_count, len(group) - 1)
        return group[:-val_count], group[-val_count:]

    train_pos, val_pos = split_group(positives)
    train_neg, val_neg = split_group(negatives)
    train = sorted(train_pos + train_neg, key=lambda window: window.sequence)
    validation = sorted(val_pos + val_neg, key=lambda window: window.sequence)
    return train, validation


def _contiguous_client_assignment(windows: Sequence[WindowRef], *, num_clients: int) -> list[list[WindowRef]]:
    targets = _balanced_counts(len(windows), num_clients)
    clients: list[list[WindowRef]] = []
    cursor = 0
    for target in targets:
        clients.append(list(windows[cursor : cursor + target]))
        cursor += target
    return clients


def _window_counts(windows: Sequence[WindowRef]) -> dict[str, int]:
    return _label_counts([window.label for window in windows])


def _validate_both_classes(name: str, windows: Sequence[WindowRef]) -> None:
    counts = _window_counts(windows)
    if set(counts) != {"0", "1"}:
        observed = ", ".join(f"{key}:{value}" for key, value in counts.items())
        raise DatasetSchemaError(f"{name} must contain both window classes. Observed {observed}.")


def _mark_training_rows(
    train_windows_by_client: Sequence[Sequence[WindowRef]],
    loaded_files: Mapping[str, LoadedFile],
) -> dict[str, np.ndarray]:
    masks = {
        file_name: np.zeros(loaded.labels.shape[0], dtype=bool)
        for file_name, loaded in loaded_files.items()
    }
    for client_windows in train_windows_by_client:
        for window in client_windows:
            masks[window.file_name][window.start : window.stop] = True
    return masks


def _transform_window(
    transformed_by_file: Mapping[str, np.ndarray],
    window: WindowRef,
) -> np.ndarray:
    rows = transformed_by_file[window.file_name][window.start : window.stop]
    return rows.T.reshape(-1)


def _client_descriptor(
    transformed_by_file: Mapping[str, np.ndarray],
    train_windows: Sequence[WindowRef],
) -> np.ndarray:
    if not train_windows:
        raise DatasetSchemaError("Cluster 1 repaired: client has no training windows for descriptor computation.")

    first = _transform_window(transformed_by_file, train_windows[0]).astype(np.float64)
    total = np.zeros_like(first, dtype=np.float64)
    total_sq = np.zeros_like(first, dtype=np.float64)
    for window in train_windows:
        vector = _transform_window(transformed_by_file, window).astype(np.float64, copy=False)
        total += vector
        total_sq += vector * vector
    count = float(len(train_windows))
    mean = total / count
    variance = np.maximum((total_sq / count) - (mean * mean), 0.0)
    std = np.sqrt(variance)
    return np.concatenate([mean, std]).astype(np.float32, copy=False)


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
            f"Cluster 1 repaired: agglomerative clustering produced {len(members_by_label)} "
            f"non-empty subclusters, expected {len(fixed_subcluster_ids)}."
        )
    sorted_labels = sorted(
        members_by_label,
        key=lambda label: min(_client_sequence(client_id) for client_id in members_by_label[label]),
    )
    label_to_fixed = {
        raw_label: fixed_subcluster_ids[index] for index, raw_label in enumerate(sorted_labels)
    }
    return {
        client_id: label_to_fixed[int(raw_label)]
        for client_id, raw_label in zip(client_ids, raw_labels, strict=True)
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _write_file_split_report(path: Path, profile: Mapping[str, Any]) -> None:
    lines = [
        "# Cluster 1 Repaired File Split Report",
        "",
        "This report is for the exploratory Cluster 1 repaired supervised variant only.",
        "",
        "## Train/Validation Source Files",
        "",
        "| file | rows | attack=0 | attack=1 |",
        "|---|---:|---:|---:|",
    ]
    for entry in profile["train_validation_files"]:
        counts = entry["label_counts"]
        lines.append(f"| {entry['file_name']} | {entry['row_count']} | {counts.get('0', 0)} | {counts.get('1', 0)} |")
    lines.extend(
        [
            "",
            "## Held-Out Test Source Files",
            "",
            "| file | rows | attack=0 | attack=1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for entry in profile["heldout_test_files"]:
        counts = entry["label_counts"]
        lines.append(f"| {entry['file_name']} | {entry['row_count']} | {counts.get('0', 0)} | {counts.get('1', 0)} |")
    lines.extend(
        [
            "",
            "## Window Counts",
            "",
            f"- Train/validation source windows: `{profile['train_validation_window_counts']}`",
            f"- Held-out test windows: `{profile['heldout_test_window_counts']}`",
            "",
            "`test4.csv` and `test5.csv` are not used for training, validation, preprocessing-fit, or descriptor computation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_client_balance_report(path: Path, clients: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Cluster 1 Repaired Client Balance Report",
        "",
        "Positive windows are distributed round-robin across the 12 candidate clients before negative windows are used to fill deterministic client quotas.",
        "",
        "| client | train 0 | train 1 | validation 0 | validation 1 | held-out test 0 | held-out test 1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for client in clients:
        train_counts = client["train_label_counts"]
        val_counts = client["val_label_counts"]
        test_counts = client["test_label_counts"]
        lines.append(
            f"| {client['client_id']} | {train_counts.get('0', 0)} | {train_counts.get('1', 0)} | "
            f"{val_counts.get('0', 0)} | {val_counts.get('1', 0)} | "
            f"{test_counts.get('0', 0)} | {test_counts.get('1', 0)} |"
        )
    clients_with_positive = sum(
        1
        for client in clients
        if client["train_label_counts"].get("1", 0) + client["val_label_counts"].get("1", 0) > 0
    )
    lines.extend(
        [
            "",
            f"Clients with at least one positive train-or-validation window: `{clients_with_positive}/{len(clients)}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_summary_report(
    path: Path,
    *,
    profile: Mapping[str, Any],
    clients: Sequence[Mapping[str, Any]],
) -> None:
    train_validation_counts = profile["train_validation_window_counts"]
    heldout_counts = profile["heldout_test_window_counts"]
    clients_with_positive_train = [
        str(client["client_id"])
        for client in clients
        if int(client["train_label_counts"].get("1", 0)) > 0
    ]
    attack_free_training_clients = [
        str(client["client_id"])
        for client in clients
        if int(client["train_label_counts"].get("1", 0)) == 0
    ]
    all_clients_have_positive_train = len(attack_free_training_clients) == 0

    lines = [
        "# Cluster 1 Repaired Validation Summary",
        "",
        "This summary validates only the exploratory repaired Cluster 1 dataset-construction variant.",
        "",
        "## Window Counts",
        "",
        f"- Repaired training/validation source windows: `{train_validation_counts}`",
        f"- Repaired held-out test windows: `{heldout_counts}`",
        "",
        "## Training Pool Class Counts",
        "",
        f"- Negative windows: `{int(train_validation_counts.get('0', 0))}`",
        f"- Positive windows: `{int(train_validation_counts.get('1', 0))}`",
        "",
        "## Held-Out Test Pool Class Counts",
        "",
        f"- Negative windows: `{int(heldout_counts.get('0', 0))}`",
        f"- Positive windows: `{int(heldout_counts.get('1', 0))}`",
        "",
        "## Client Attack Coverage",
        "",
        f"- Clients with at least one positive training window: `{len(clients_with_positive_train)}/{len(clients)}`",
        f"- Every client received at least one positive training window: `{'YES' if all_clients_have_positive_train else 'NO'}`",
        f"- Attack-free training clients: `{len(attack_free_training_clients)}`",
    ]
    if attack_free_training_clients:
        lines.append(f"- Attack-free training client IDs: `{attack_free_training_clients}`")
    else:
        lines.append("- Attack-free training client IDs: `[]`")
    lines.extend(
        [
            "",
            "## Leakage Check",
            "",
            "- `test4.csv` and `test5.csv` are used only for held-out test-window construction.",
            "- They are not used for training, validation, preprocessing fit, or descriptor computation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_cluster1_repaired(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    dataset_config = load_cluster_config(config_path)
    raw_config = _load_yaml(config_path)
    data_config = raw_config.get("data")
    if not isinstance(data_config, Mapping):
        raise DatasetConfigError("Cluster 1 repaired config must contain a data section.")
    if dataset_config.cluster_id != 1:
        raise DatasetConfigError("The repaired dataset builder is restricted to Cluster 1.")

    partitioning = raw_config.get("partitioning")
    preprocessing = raw_config.get("preprocessing")
    clustering = raw_config.get("clustering")
    if not isinstance(partitioning, Mapping) or not isinstance(preprocessing, Mapping) or not isinstance(clustering, Mapping):
        raise DatasetConfigError("Cluster 1 repaired config must define partitioning, preprocessing, and clustering.")

    train_validation_files = _require_string_list(data_config, "train_validation_files")
    heldout_test_files = _require_string_list(data_config, "heldout_test_files")
    num_clients = int(partitioning.get("candidate_leaf_clients", 0))
    if num_clients <= 0:
        raise DatasetConfigError("Cluster 1 repaired partitioning.candidate_leaf_clients must be positive.")
    validation_ratio = float(partitioning.get("validation_ratio", 0.15))
    if not 0.0 < validation_ratio < 1.0:
        raise DatasetConfigError("Cluster 1 repaired partitioning.validation_ratio must be in (0, 1).")

    window_length = int(preprocessing.get("window_length", 0))
    stride = int(preprocessing.get("stride", 0))
    if window_length <= 0 or stride <= 0:
        raise DatasetConfigError("Cluster 1 repaired window_length and stride must be positive.")
    if str(preprocessing.get("window_label_rule")) != "any_positive_row":
        raise DatasetConfigError("Cluster 1 repaired only supports any_positive_row window labels.")

    fixed_subcluster_ids = tuple(str(item) for item in clustering.get("fixed_subcluster_ids", []))
    n_subclusters = int(clustering.get("fixed_subclusters", 0))
    if n_subclusters != 2 or len(fixed_subcluster_ids) != 2:
        raise DatasetConfigError("Cluster 1 repaired must keep K1=2 fixed subclusters.")

    root = Path(output_root) if output_root is not None else Path(data_config.get("repaired_output_root", DEFAULT_OUTPUT_ROOT))
    raw_dir = resolve_configured_path(dataset_config, dataset_config.current_raw_input_dir)
    all_files = train_validation_files + heldout_test_files
    file_paths = {file_name: raw_dir / file_name for file_name in all_files}
    missing = [str(path) for path in file_paths.values() if not path.exists()]
    if missing:
        raise DatasetSchemaError(f"Cluster 1 repaired missing raw file(s): {missing}")

    first_header = _read_header(file_paths[train_validation_files[0]])
    label_column = dataset_config.label_column
    if label_column not in first_header:
        raise DatasetSchemaError(f"Cluster 1 repaired missing label column {label_column!r}.")
    exclusion_lookup = set(dataset_config.excluded_columns) | set(dataset_config.exclude_if_present)
    feature_names = tuple(column for column in first_header if column not in exclusion_lookup)
    feature_indices = [first_header.index(column) for column in feature_names]
    label_index = first_header.index(label_column)

    file_profiles: dict[str, dict[str, Any]] = {}
    for file_name, path in file_paths.items():
        profile = _file_label_counts(path, label_column)
        profile["file_name"] = file_name
        file_profiles[file_name] = profile
        header = _read_header(path)
        if header != first_header:
            raise DatasetSchemaError(f"Cluster 1 repaired requires consistent HAI schema. Mismatch in {path}.")

    def load_file(file_name: str) -> LoadedFile:
        path = file_paths[file_name]
        labels = _mapped_label_array(path, label_index)
        features = _load_feature_matrix(path, feature_indices)
        if features.shape[0] != labels.shape[0]:
            raise DatasetSchemaError(f"Cluster 1 repaired feature/label row mismatch in {path}.")
        return LoadedFile(file_name=file_name, path=path, feature_matrix=features, labels=labels)

    train_validation_loaded = [load_file(file_name) for file_name in train_validation_files]
    heldout_loaded = [load_file(file_name) for file_name in heldout_test_files]
    loaded_by_name = {loaded.file_name: loaded for loaded in train_validation_loaded + heldout_loaded}

    train_validation_windows = _build_windows(train_validation_loaded, window_length=window_length, stride=stride)
    heldout_windows = _build_windows(heldout_loaded, window_length=window_length, stride=stride)
    if not train_validation_windows:
        raise DatasetSchemaError("Cluster 1 repaired train/validation pool produced zero windows.")
    if not heldout_windows:
        raise DatasetSchemaError("Cluster 1 repaired held-out pool produced zero windows.")
    _validate_both_classes("Cluster 1 repaired train/validation windows", train_validation_windows)
    _validate_both_classes("Cluster 1 repaired held-out windows", heldout_windows)

    assigned_clients = attack_aware_client_assignment(train_validation_windows, num_clients=num_clients)
    heldout_clients = _contiguous_client_assignment(heldout_windows, num_clients=num_clients)

    train_by_client: list[list[WindowRef]] = []
    validation_by_client: list[list[WindowRef]] = []
    for client_windows in assigned_clients:
        train_windows, validation_windows = split_client_train_validation(
            client_windows,
            validation_ratio=validation_ratio,
        )
        train_by_client.append(train_windows)
        validation_by_client.append(validation_windows)

    cluster_train_counts = Counter()
    cluster_validation_counts = Counter()
    cluster_test_counts = Counter()
    client_entries: list[dict[str, Any]] = []
    for index in range(num_clients):
        client_id = f"C1_L{index + 1:03d}"
        train_counts = _window_counts(train_by_client[index])
        validation_counts = _window_counts(validation_by_client[index])
        test_counts = _window_counts(heldout_clients[index])
        cluster_train_counts.update(train_counts)
        cluster_validation_counts.update(validation_counts)
        cluster_test_counts.update(test_counts)
        client_entries.append(
            {
                "client_id": client_id,
                "ordered_start_index": 0,
                "ordered_end_index": len(assigned_clients[index]),
                "num_total_samples": len(assigned_clients[index]) + len(heldout_clients[index]),
                "num_train_samples": len(train_by_client[index]),
                "num_val_samples": len(validation_by_client[index]),
                "num_test_samples": len(heldout_clients[index]),
                "train_label_counts": train_counts,
                "val_label_counts": validation_counts,
                "test_label_counts": test_counts,
                "train_range": {
                    "start_index": 0,
                    "end_index": len(train_by_client[index]),
                    "num_samples": len(train_by_client[index]),
                    "label_counts": train_counts,
                    "single_class": len(train_counts) == 1,
                },
                "val_range": {
                    "start_index": len(train_by_client[index]),
                    "end_index": len(train_by_client[index]) + len(validation_by_client[index]),
                    "num_samples": len(validation_by_client[index]),
                    "label_counts": validation_counts,
                    "single_class": len(validation_counts) == 1,
                },
                "test_range": {
                    "start_index": 0,
                    "end_index": len(heldout_clients[index]),
                    "num_samples": len(heldout_clients[index]),
                    "label_counts": test_counts,
                    "single_class": len(test_counts) == 1,
                },
                "single_class_splits": [
                    split_name
                    for split_name, counts in (
                        ("train", train_counts),
                        ("validation", validation_counts),
                        ("test", test_counts),
                    )
                    if len(counts) == 1
                ],
                "notes": ["repaired_cluster1_window_first_attack_aware_partition"],
            }
        )

    for split_name, counts in (
        ("train", cluster_train_counts),
        ("validation", cluster_validation_counts),
        ("held-out test", cluster_test_counts),
    ):
        if set(counts) != {"0", "1"}:
            raise DatasetSchemaError(
                f"Cluster 1 repaired cluster-level {split_name} split lacks both classes: {dict(counts)}"
            )

    train_masks = _mark_training_rows(train_by_client, {loaded.file_name: loaded for loaded in train_validation_loaded})
    training_feature_rows = np.vstack(
        [
            loaded.feature_matrix[train_masks[loaded.file_name]]
            for loaded in train_validation_loaded
            if train_masks[loaded.file_name].any()
        ]
    )
    numeric_artifacts = NumericTransformArtifacts.fit(feature_names, training_feature_rows)
    if not numeric_artifacts.kept_features:
        raise DatasetSchemaError("Cluster 1 repaired preprocessing dropped all feature columns.")

    transformed_by_file = {
        loaded.file_name: numeric_artifacts.transform(loaded.feature_matrix)
        for loaded in train_validation_loaded
    }

    client_ids = [entry["client_id"] for entry in client_entries]
    descriptor_rows = [
        _client_descriptor(transformed_by_file, train_by_client[index])
        for index in range(num_clients)
    ]
    raw_descriptor_matrix = np.vstack(descriptor_rows)
    descriptor_scaler = StandardScaler()
    standardized_descriptor_matrix = descriptor_scaler.fit_transform(raw_descriptor_matrix).astype(np.float32, copy=False)
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

    client_metadata = {
        "cluster_id": 1,
        "dataset": "HAI 21.03 repaired supervised variant",
        "variant": "cluster1_repaired",
        "num_leaf_clients": num_clients,
        "total_ordered_samples": len(train_validation_windows),
        "split_ratios": {"train": 1.0 - validation_ratio, "validation": validation_ratio, "test": 0.0},
        "ordering_mode": "window_first_file_order_then_attack_aware_round_robin",
        "ordering_columns_used": [],
        "source_paths": [str(file_paths[file_name]) for file_name in train_validation_files],
        "heldout_test_source_paths": [str(file_paths[file_name]) for file_name in heldout_test_files],
        "cluster_label_counts": _window_counts(train_validation_windows),
        "cluster_split_label_counts": {
            "train": {key: cluster_train_counts[key] for key in sorted(cluster_train_counts)},
            "validation": {key: cluster_validation_counts[key] for key in sorted(cluster_validation_counts)},
            "test": {key: cluster_test_counts[key] for key in sorted(cluster_test_counts)},
        },
        "single_class_local_partitions": [
            {
                "client_id": client["client_id"],
                "split": split_name,
                "label_counts": client[f"{'val' if split_name == 'validation' else split_name}_label_counts"],
            }
            for client in client_entries
            for split_name in client["single_class_splits"]
        ],
        "descriptor_source_split": "train",
        "partitioning_strategy": "window_first_attack_aware_round_robin_positive_then_negative_quota_fill",
        "clients": client_entries,
    }
    _write_json(client_metadata_path, client_metadata)

    subclusters = [
        {
            "subcluster_id": subcluster_id,
            "client_ids": [client_id for client_id in client_ids if assignments[client_id] == subcluster_id],
        }
        for subcluster_id in fixed_subcluster_ids
    ]
    membership = {
        "cluster_id": 1,
        "dataset": "HAI 21.03 repaired supervised variant",
        "variant": "cluster1_repaired",
        "status": "ok",
        "clustering_method": "AgglomerativeClustering",
        "linkage": "ward",
        "metric": "euclidean",
        "descriptor": "feature_mean_std_window_flattened",
        "descriptor_dim": int(raw_descriptor_matrix.shape[1]),
        "descriptor_source_split": "train",
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
    _write_json(membership_path, membership)
    _write_pickle(descriptor_scaler_path, descriptor_scaler)

    imputer_path = preprocessing_dir / "cluster1_hai_repaired_imputer.pkl"
    scaler_path = preprocessing_dir / "cluster1_hai_repaired_scaler.pkl"
    preprocessor_path = preprocessing_dir / "cluster1_hai_repaired_preprocessor.pkl"
    _write_pickle(
        imputer_path,
        {
            "variant": "cluster1_repaired",
            "fit_scope": "repaired_training_windows_only",
            "input_features": numeric_artifacts.input_features,
            "kept_features": numeric_artifacts.kept_features,
            "medians": numeric_artifacts.medians.tolist(),
        },
    )
    _write_pickle(
        scaler_path,
        {
            "variant": "cluster1_repaired",
            "fit_scope": "repaired_training_windows_only",
            "input_features": numeric_artifacts.input_features,
            "kept_features": numeric_artifacts.kept_features,
            "means": numeric_artifacts.means.tolist(),
            "scales": numeric_artifacts.scales.tolist(),
        },
    )
    _write_pickle(
        preprocessor_path,
        {
            "variant": "cluster1_repaired",
            "label_column": label_column,
            "numeric_artifacts": numeric_artifacts,
            "output_feature_names": numeric_artifacts.kept_features,
            "window_length": window_length,
            "stride": stride,
            "window_label_rule": "any_positive_row",
        },
    )

    profile = {
        "cluster_id": 1,
        "dataset": "HAI 21.03",
        "variant": "cluster1_repaired",
        "status": "ok",
        "config_path": str(config_path),
        "output_root": str(root),
        "label_column": label_column,
        "train_validation_source_files": list(train_validation_files),
        "heldout_test_source_files": list(heldout_test_files),
        "train_validation_files": [file_profiles[file_name] for file_name in train_validation_files],
        "heldout_test_files": [file_profiles[file_name] for file_name in heldout_test_files],
        "feature_columns_before_preprocessing": list(feature_names),
        "feature_columns_after_preprocessing": list(numeric_artifacts.kept_features),
        "dropped_all_missing_feature_columns": list(numeric_artifacts.dropped_all_missing_features),
        "dropped_constant_feature_columns": list(numeric_artifacts.dropped_constant_features),
        "window_length": window_length,
        "stride": stride,
        "window_label_rule": "any_positive_row",
        "train_validation_window_counts": _window_counts(train_validation_windows),
        "heldout_test_window_counts": _window_counts(heldout_windows),
        "cluster_split_label_counts": client_metadata["cluster_split_label_counts"],
        "clients_with_positive_train_or_validation": sum(
            1
            for client in client_entries
            if client["train_label_counts"].get("1", 0) + client["val_label_counts"].get("1", 0) > 0
        ),
        "num_leaf_clients": num_clients,
        "membership_path": str(membership_path),
        "client_metadata_path": str(client_metadata_path),
        "preprocessing_artifact_paths": {
            "imputer": str(imputer_path),
            "scaler": str(scaler_path),
            "preprocessor": str(preprocessor_path),
        },
        "test_leakage_prevention": {
            "heldout_test_files_used_for_training": False,
            "heldout_test_files_used_for_validation": False,
            "heldout_test_files_used_for_preprocessing_fit": False,
            "heldout_test_files_used_for_descriptors": False,
        },
    }
    _write_json(reports_dir / "data_profile_cluster1_repaired.json", profile)
    _write_json(
        reports_dir / "preprocessing_summary_cluster1_repaired.json",
        {
            "cluster_id": 1,
            "dataset": "HAI 21.03",
            "variant": "cluster1_repaired",
            "status": "ok",
            "fit_scope": "repaired_training_windows_only",
            "source_paths": [str(file_paths[file_name]) for file_name in train_validation_files],
            "heldout_test_source_paths": [str(file_paths[file_name]) for file_name in heldout_test_files],
            "label_column": label_column,
            "label_counts": _window_counts(train_validation_windows),
            "output_feature_columns": list(numeric_artifacts.kept_features),
            "output_feature_count": len(numeric_artifacts.kept_features),
            "windowing": "window_first",
            "artifact_paths": profile["preprocessing_artifact_paths"],
        },
    )
    _write_json(
        reports_dir / "label_summary_cluster1_repaired.json",
        {
            "cluster_id": 1,
            "dataset": "HAI 21.03",
            "variant": "cluster1_repaired",
            "label_column": label_column,
            "train_validation_window_counts": _window_counts(train_validation_windows),
            "heldout_test_window_counts": _window_counts(heldout_windows),
            "cluster_split_label_counts": client_metadata["cluster_split_label_counts"],
        },
    )
    _write_file_split_report(reports_dir / "cluster1_repaired_file_split_report.md", profile)
    _write_client_balance_report(reports_dir / "cluster1_repaired_client_balance_report.md", client_entries)
    _write_validation_summary_report(
        reports_dir / "cluster1_repaired_validation_summary.md",
        profile=profile,
        clients=client_entries,
    )

    return {
        "profile": profile,
        "client_metadata": client_metadata,
        "membership": membership,
        "paths": {
            "data_profile": str(reports_dir / "data_profile_cluster1_repaired.json"),
            "file_split_report": str(reports_dir / "cluster1_repaired_file_split_report.md"),
            "client_balance_report": str(reports_dir / "cluster1_repaired_client_balance_report.md"),
            "validation_summary": str(reports_dir / "cluster1_repaired_validation_summary.md"),
            "client_metadata": str(client_metadata_path),
            "membership": str(membership_path),
        },
    }
