from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .leaf_clients import (
    ClusterLeafClientMetadata,
    LeafClientMetadata,
    SplitMetadata,
    save_cluster_leaf_client_metadata,
)
from .loaders import DEFAULT_CLUSTER_CONFIG_PATHS, load_dataset_for_training
from .schema_validation import DatasetConfig, DatasetConfigError, DatasetInspection, DatasetSchemaError, label_value_to_binary, load_cluster_config


DEFAULT_SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


@dataclass(frozen=True)
class PartitioningConfig:
    dataset_config: DatasetConfig
    candidate_leaf_clients: int
    split_ratios: dict[str, float]


@dataclass(frozen=True)
class PartitionBuildResult:
    metadata: ClusterLeafClientMetadata
    output_path: Path
    ordered_row_indices: np.ndarray
    ordered_labels: np.ndarray


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _metadata_path_for_cluster(cluster_id: int) -> Path:
    return Path(f"outputs/clients/cluster{cluster_id}_leaf_clients.json")


def load_partitioning_config(config_path: str | Path) -> PartitioningConfig:
    dataset_config = load_cluster_config(config_path)
    raw_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise DatasetConfigError(f"Partition config must be a mapping: {config_path}")

    partitioning = raw_config.get("partitioning")
    if not isinstance(partitioning, dict):
        raise DatasetConfigError(f"{dataset_config.dataset_name}: missing partitioning config section.")

    candidate_leaf_clients = partitioning.get("candidate_leaf_clients")
    if not isinstance(candidate_leaf_clients, int) or candidate_leaf_clients <= 0:
        raise DatasetConfigError(
            f"{dataset_config.dataset_name}: partitioning.candidate_leaf_clients must be a positive integer."
        )

    split_ratios = dict(DEFAULT_SPLIT_RATIOS)
    configured_ratios = partitioning.get("split_ratios")
    if configured_ratios is not None:
        if not isinstance(configured_ratios, dict):
            raise DatasetConfigError(
                f"{dataset_config.dataset_name}: partitioning.split_ratios must be a mapping if provided."
            )

        ratio_keys = {
            "train": configured_ratios.get("train"),
            "validation": configured_ratios.get("validation", configured_ratios.get("val")),
            "test": configured_ratios.get("test"),
        }
        for split_name, value in ratio_keys.items():
            if not isinstance(value, (int, float)):
                raise DatasetConfigError(
                    f"{dataset_config.dataset_name}: split ratio {split_name!r} must be numeric."
                )
            split_ratios[split_name] = float(value)

    total_ratio = sum(split_ratios.values())
    if not math.isclose(total_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise DatasetConfigError(
            f"{dataset_config.dataset_name}: split ratios must sum to 1.0. Observed {total_ratio}."
        )
    for split_name, value in split_ratios.items():
        if value < 0.0 or value > 1.0:
            raise DatasetConfigError(
                f"{dataset_config.dataset_name}: split ratio {split_name!r} must be in [0, 1]."
            )

    return PartitioningConfig(
        dataset_config=dataset_config,
        candidate_leaf_clients=candidate_leaf_clients,
        split_ratios=split_ratios,
    )


def _load_string_column(path: Path, column_index: int) -> np.ndarray:
    values = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        usecols=(column_index,),
        dtype=str,
        encoding="utf-8",
        autostrip=True,
    )
    if values.size == 0:
        return np.empty(0, dtype=str)
    return np.atleast_1d(values)


def _load_mapped_labels(inspection: DatasetInspection) -> np.ndarray:
    first_file = inspection.file_inspections[0]
    label_index = first_file.normalized_columns.index(inspection.config.label_column)
    mapped_labels: list[np.ndarray] = []

    for path in inspection.file_paths:
        raw_labels = _load_string_column(path, label_index)
        mapped = np.empty(raw_labels.shape[0], dtype=np.int8)
        for index, raw_label in enumerate(raw_labels):
            label = label_value_to_binary(str(raw_label))
            if label is None:
                raise DatasetSchemaError(
                    f"{inspection.config.dataset_name}: unexpected label value {raw_label!r} while building leaf-client splits."
                )
            mapped[index] = int(label)
        mapped_labels.append(mapped)

    return np.concatenate(mapped_labels) if mapped_labels else np.empty(0, dtype=np.int8)


def _select_order_columns(inspection: DatasetInspection) -> tuple[str, ...]:
    first_file = inspection.file_inspections[0]
    missing_counts = first_file.missing_value_counts
    available = []
    for column in inspection.config.timestamp_or_order_columns:
        if column in first_file.normalized_columns and missing_counts.get(column, 0) == 0:
            available.append(column)
    return tuple(available)


def _looks_like_dd_mon_yy(value: str) -> bool:
    if len(value) != 9:
        return False
    try:
        datetime.strptime(value, "%d-%b-%y")
    except ValueError:
        return False
    return True


def _is_numeric_array(values: np.ndarray) -> bool:
    try:
        values.astype(np.float64)
    except ValueError:
        return False
    return True


def _coerce_order_key(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values

    if _is_numeric_array(values):
        return values.astype(np.float64)

    sample = str(values[0]).strip()
    if _looks_like_dd_mon_yy(sample):
        unique_values = sorted(set(str(value).strip() for value in values))
        mapping = {
            value: datetime.strptime(value, "%d-%b-%y").strftime("%Y-%m-%d")
            for value in unique_values
        }
        return np.asarray([mapping[str(value).strip()] for value in values], dtype=str)

    return np.asarray([str(value).strip() for value in values], dtype=str)


def _load_order_columns(
    inspection: DatasetInspection,
    order_columns: Sequence[str],
) -> list[np.ndarray]:
    if not order_columns:
        return []

    first_file = inspection.file_inspections[0]
    column_indices = [first_file.normalized_columns.index(column) for column in order_columns]
    loaded_columns = [[] for _ in order_columns]

    for path in inspection.file_paths:
        for list_index, column_index in enumerate(column_indices):
            loaded_columns[list_index].append(_load_string_column(path, column_index))

    return [_coerce_order_key(np.concatenate(column_values)) for column_values in loaded_columns]


def _build_ordered_indices(
    inspection: DatasetInspection,
    total_rows: int,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    order_columns = _select_order_columns(inspection)
    if not order_columns:
        return np.arange(total_rows, dtype=np.int64), "preserve_file_order", ()

    order_keys = _load_order_columns(inspection, order_columns)
    original_indices = np.arange(total_rows, dtype=np.int64)
    ordered_indices = np.lexsort(tuple([original_indices] + list(reversed(order_keys))))
    return ordered_indices.astype(np.int64, copy=False), "sorted_by_configured_timestamp_or_order_columns", order_columns


def _shard_boundaries(total_rows: int, num_clients: int) -> list[tuple[int, int]]:
    return [
        (client_index * total_rows // num_clients, (client_index + 1) * total_rows // num_clients)
        for client_index in range(num_clients)
    ]


def _split_bounds(num_samples: int, split_ratios: Mapping[str, float]) -> dict[str, tuple[int, int]]:
    if num_samples <= 0:
        return {
            "train": (0, 0),
            "validation": (0, 0),
            "test": (0, 0),
        }

    train_end = max(1, int(num_samples * split_ratios["train"]))
    train_end = min(train_end, num_samples)

    validation_end = int(num_samples * (split_ratios["train"] + split_ratios["validation"]))
    validation_end = max(train_end, validation_end)
    validation_end = min(validation_end, num_samples)

    return {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, num_samples),
    }


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    counter = Counter(str(int(label)) for label in labels.tolist())
    return _sorted_counter(counter)


def _single_class(split_labels: np.ndarray) -> bool:
    if split_labels.size == 0:
        return False
    return len(set(split_labels.tolist())) == 1


def _build_split_metadata(
    ordered_start_index: int,
    local_bounds: tuple[int, int],
    shard_labels: np.ndarray,
) -> SplitMetadata:
    local_start, local_end = local_bounds
    labels = shard_labels[local_start:local_end]
    return SplitMetadata(
        start_index=ordered_start_index + local_start,
        end_index=ordered_start_index + local_end,
        num_samples=int(labels.shape[0]),
        label_counts=_label_counts(labels),
        single_class=_single_class(labels),
    )


def _validate_cluster_split_coverage(
    dataset_name: str,
    cluster_split_label_counts: dict[str, Counter[str]],
) -> None:
    for split_name, label_counts in cluster_split_label_counts.items():
        if not label_counts:
            continue
        if set(label_counts) != {"0", "1"}:
            observed = ", ".join(f"{key}:{value}" for key, value in sorted(label_counts.items()))
            raise DatasetSchemaError(
                f"{dataset_name}: cluster-level {split_name} split must contain both labels {{0,1}}. "
                f"Observed counts: {observed or 'none'}."
            )


def build_candidate_leaf_clients(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> PartitionBuildResult:
    partitioning_config = load_partitioning_config(config_path)
    inspection = load_dataset_for_training(config_path)
    labels = _load_mapped_labels(inspection)
    total_rows = int(labels.shape[0])

    if total_rows < partitioning_config.candidate_leaf_clients:
        raise DatasetSchemaError(
            f"{inspection.config.dataset_name}: {total_rows} rows are insufficient for "
            f"{partitioning_config.candidate_leaf_clients} candidate leaf clients."
        )

    ordered_indices, ordering_mode, order_columns_used = _build_ordered_indices(inspection, total_rows)
    ordered_labels = labels[ordered_indices]

    boundaries = _shard_boundaries(total_rows, partitioning_config.candidate_leaf_clients)
    cluster_split_label_counts = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    single_class_local_partitions: list[dict[str, Any]] = []
    client_entries: list[LeafClientMetadata] = []

    for client_offset, (start_index, end_index) in enumerate(boundaries, start=1):
        shard_labels = ordered_labels[start_index:end_index]
        if shard_labels.size == 0:
            raise DatasetSchemaError(
                f"{inspection.config.dataset_name}: candidate leaf client {client_offset} would be empty."
            )

        local_bounds = _split_bounds(int(shard_labels.shape[0]), partitioning_config.split_ratios)
        train_metadata = _build_split_metadata(start_index, local_bounds["train"], shard_labels)
        validation_metadata = _build_split_metadata(start_index, local_bounds["validation"], shard_labels)
        test_metadata = _build_split_metadata(start_index, local_bounds["test"], shard_labels)

        if train_metadata.num_samples == 0:
            raise DatasetSchemaError(
                f"{inspection.config.dataset_name}: client C{inspection.config.cluster_id}_L{client_offset:03d} has no train split."
            )

        split_metadata_lookup = {
            "train": train_metadata,
            "validation": validation_metadata,
            "test": test_metadata,
        }

        single_class_splits: list[str] = []
        notes: list[str] = []
        for split_name, metadata in split_metadata_lookup.items():
            cluster_split_label_counts[split_name].update(metadata.label_counts)
            if metadata.num_samples == 0:
                notes.append(f"{split_name} split is empty.")
            if metadata.single_class:
                single_class_splits.append(split_name)
                single_class_local_partitions.append(
                    {
                        "client_id": f"C{inspection.config.cluster_id}_L{client_offset:03d}",
                        "split": split_name,
                        "label_counts": dict(metadata.label_counts),
                    }
                )

        client_entries.append(
            LeafClientMetadata(
                client_id=f"C{inspection.config.cluster_id}_L{client_offset:03d}",
                ordered_start_index=start_index,
                ordered_end_index=end_index,
                num_total_samples=int(shard_labels.shape[0]),
                train=train_metadata,
                validation=validation_metadata,
                test=test_metadata,
                single_class_splits=tuple(single_class_splits),
                notes=tuple(notes),
            )
        )

    _validate_cluster_split_coverage(inspection.config.dataset_name, cluster_split_label_counts)

    metadata = ClusterLeafClientMetadata(
        cluster_id=inspection.config.cluster_id,
        dataset=inspection.config.dataset_name,
        num_leaf_clients=partitioning_config.candidate_leaf_clients,
        total_ordered_samples=total_rows,
        split_ratios=partitioning_config.split_ratios,
        ordering_mode=ordering_mode,
        ordering_columns_used=tuple(order_columns_used),
        source_paths=tuple(str(path) for path in inspection.file_paths),
        cluster_label_counts=_label_counts(ordered_labels),
        cluster_split_label_counts={
            split_name: _sorted_counter(counts)
            for split_name, counts in cluster_split_label_counts.items()
        },
        single_class_local_partitions=tuple(single_class_local_partitions),
        descriptor_source_split="train",
        clients=tuple(client_entries),
    )

    resolved_output_path = Path(output_path) if output_path is not None else _metadata_path_for_cluster(metadata.cluster_id)
    saved_path = save_cluster_leaf_client_metadata(metadata, resolved_output_path)
    return PartitionBuildResult(
        metadata=metadata,
        output_path=saved_path,
        ordered_row_indices=ordered_indices,
        ordered_labels=ordered_labels,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate leaf-client partitions.")
    parser.add_argument(
        "--config",
        dest="config_paths",
        action="append",
        help="Optional cluster config path. If omitted, all default cluster configs are processed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = args.config_paths or [str(path) for path in DEFAULT_CLUSTER_CONFIG_PATHS.values()]
    for config_path in config_paths:
        result = build_candidate_leaf_clients(config_path)
        print(f"{result.metadata.dataset}: wrote {result.output_path}")


if __name__ == "__main__":
    main()
