from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SplitMetadata:
    start_index: int
    end_index: int
    num_samples: int
    label_counts: Mapping[str, int]
    single_class: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "num_samples": self.num_samples,
            "label_counts": dict(self.label_counts),
            "single_class": self.single_class,
        }


@dataclass(frozen=True)
class LeafClientMetadata:
    client_id: str
    ordered_start_index: int
    ordered_end_index: int
    num_total_samples: int
    train: SplitMetadata
    validation: SplitMetadata
    test: SplitMetadata
    single_class_splits: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "ordered_start_index": self.ordered_start_index,
            "ordered_end_index": self.ordered_end_index,
            "num_total_samples": self.num_total_samples,
            "num_train_samples": self.train.num_samples,
            "num_val_samples": self.validation.num_samples,
            "num_test_samples": self.test.num_samples,
            "train_label_counts": dict(self.train.label_counts),
            "val_label_counts": dict(self.validation.label_counts),
            "test_label_counts": dict(self.test.label_counts),
            "train_range": self.train.to_dict(),
            "val_range": self.validation.to_dict(),
            "test_range": self.test.to_dict(),
            "single_class_splits": list(self.single_class_splits),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ClusterLeafClientMetadata:
    cluster_id: int
    dataset: str
    num_leaf_clients: int
    total_ordered_samples: int
    split_ratios: Mapping[str, float]
    ordering_mode: str
    ordering_columns_used: tuple[str, ...]
    source_paths: tuple[str, ...]
    cluster_label_counts: Mapping[str, int]
    cluster_split_label_counts: Mapping[str, Mapping[str, int]]
    single_class_local_partitions: tuple[Mapping[str, Any], ...]
    descriptor_source_split: str
    clients: tuple[LeafClientMetadata, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "dataset": self.dataset,
            "num_leaf_clients": self.num_leaf_clients,
            "total_ordered_samples": self.total_ordered_samples,
            "split_ratios": dict(self.split_ratios),
            "ordering_mode": self.ordering_mode,
            "ordering_columns_used": list(self.ordering_columns_used),
            "source_paths": list(self.source_paths),
            "cluster_label_counts": dict(self.cluster_label_counts),
            "cluster_split_label_counts": {
                split_name: dict(label_counts)
                for split_name, label_counts in self.cluster_split_label_counts.items()
            },
            "single_class_local_partitions": [dict(item) for item in self.single_class_local_partitions],
            "descriptor_source_split": self.descriptor_source_split,
            "clients": [client.to_dict() for client in self.clients],
        }


def metadata_path_for_cluster(cluster_id: int) -> Path:
    return Path(f"outputs/clients/cluster{cluster_id}_leaf_clients.json")


def save_cluster_leaf_client_metadata(
    metadata: ClusterLeafClientMetadata,
    path: str | Path | None = None,
) -> Path:
    output_path = Path(path) if path is not None else metadata_path_for_cluster(metadata.cluster_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output_path
