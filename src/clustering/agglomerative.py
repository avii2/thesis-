from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from sklearn.cluster import AgglomerativeClustering

from src.data.descriptors import ClusterDescriptorResult, build_cluster_descriptors
from src.data.loaders import DEFAULT_CLUSTER_CONFIG_PATHS
from src.data.schema_validation import DatasetConfigError, DatasetSchemaError, load_cluster_config


MEMBERSHIP_PATHS = {
    1: Path("outputs/clustering/cluster1_memberships.json"),
    2: Path("outputs/clustering/cluster2_memberships.json"),
    3: Path("outputs/clustering/cluster3_memberships.json"),
}


@dataclass(frozen=True)
class AgglomerativeRuntimeConfig:
    cluster_id: int
    dataset: str
    config_path: Path
    n_subclusters: int
    fixed_subcluster_ids: tuple[str, ...]
    linkage: str
    metric: str


@dataclass(frozen=True)
class AgglomerativeResult:
    membership_path: Path
    summary: Mapping[str, Any]
    reused_existing_membership: bool


def _default_membership_path(cluster_id: int) -> Path:
    if cluster_id in MEMBERSHIP_PATHS:
        return MEMBERSHIP_PATHS[cluster_id]
    return Path(f"outputs/clustering/cluster{cluster_id}_memberships.json")


def load_agglomerative_runtime_config(config_path: str | Path) -> AgglomerativeRuntimeConfig:
    dataset_config = load_cluster_config(config_path)
    raw_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise DatasetConfigError(f"Clustering config must be a mapping: {config_path}")

    clustering = raw_config.get("clustering")
    if not isinstance(clustering, dict):
        raise DatasetConfigError(f"{dataset_config.dataset_name}: missing clustering config section.")

    fixed_subclusters = clustering.get("fixed_subclusters")
    fixed_subcluster_ids = clustering.get("fixed_subcluster_ids")
    if not isinstance(fixed_subclusters, int) or fixed_subclusters <= 0:
        raise DatasetConfigError(
            f"{dataset_config.dataset_name}: clustering.fixed_subclusters must be a positive integer."
        )
    if (
        not isinstance(fixed_subcluster_ids, list)
        or len(fixed_subcluster_ids) != fixed_subclusters
        or any(not isinstance(item, str) or not item.strip() for item in fixed_subcluster_ids)
    ):
        raise DatasetConfigError(
            f"{dataset_config.dataset_name}: clustering.fixed_subcluster_ids must contain "
            f"{fixed_subclusters} non-empty strings."
        )

    return AgglomerativeRuntimeConfig(
        cluster_id=dataset_config.cluster_id,
        dataset=dataset_config.dataset_name,
        config_path=Path(config_path),
        n_subclusters=fixed_subclusters,
        fixed_subcluster_ids=tuple(fixed_subcluster_ids),
        linkage="ward",
        metric="euclidean",
    )


def _client_sequence(client_id: str) -> int:
    return int(client_id.rsplit("L", 1)[1])


def _build_model(n_subclusters: int) -> AgglomerativeClustering:
    try:
        return AgglomerativeClustering(
            n_clusters=n_subclusters,
            linkage="ward",
            metric="euclidean",
        )
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=n_subclusters,
            linkage="ward",
            affinity="euclidean",
        )


def _normalize_cluster_labels(
    client_ids: tuple[str, ...],
    raw_labels: list[int],
    fixed_subcluster_ids: tuple[str, ...],
) -> dict[str, str]:
    members_by_raw_label: dict[int, list[str]] = {}
    for client_id, raw_label in zip(client_ids, raw_labels, strict=True):
        members_by_raw_label.setdefault(int(raw_label), []).append(client_id)

    if len(members_by_raw_label) != len(fixed_subcluster_ids):
        raise DatasetSchemaError(
            f"Agglomerative clustering produced {len(members_by_raw_label)} non-empty sub-cluster(s), "
            f"expected {len(fixed_subcluster_ids)}."
        )

    sorted_raw_labels = sorted(
        members_by_raw_label,
        key=lambda raw_label: min(_client_sequence(client_id) for client_id in members_by_raw_label[raw_label]),
    )
    raw_to_fixed = {
        raw_label: fixed_subcluster_ids[index]
        for index, raw_label in enumerate(sorted_raw_labels)
    }
    return {
        client_id: raw_to_fixed[int(raw_label)]
        for client_id, raw_label in zip(client_ids, raw_labels, strict=True)
    }


def _membership_summary(
    runtime_config: AgglomerativeRuntimeConfig,
    descriptor_result: ClusterDescriptorResult,
    assignments: Mapping[str, str],
    membership_path: Path,
) -> dict[str, Any]:
    clients = [
        {
            "client_id": client_id,
            "subcluster_id": assignments[client_id],
        }
        for client_id in descriptor_result.client_ids
    ]
    subclusters = [
        {
            "subcluster_id": subcluster_id,
            "client_ids": [client["client_id"] for client in clients if client["subcluster_id"] == subcluster_id],
        }
        for subcluster_id in runtime_config.fixed_subcluster_ids
    ]

    membership_hash = hashlib.sha256(
        json.dumps(clients, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "cluster_id": runtime_config.cluster_id,
        "dataset": runtime_config.dataset,
        "status": "ok",
        "clustering_method": "AgglomerativeClustering",
        "linkage": runtime_config.linkage,
        "metric": runtime_config.metric,
        "descriptor": "feature_mean_std",
        "descriptor_dim": descriptor_result.descriptor_dim,
        "descriptor_source_split": "train",
        "n_subclusters": runtime_config.n_subclusters,
        "fixed_subcluster_ids": list(runtime_config.fixed_subcluster_ids),
        "frozen": True,
        "membership_hash": membership_hash,
        "client_metadata_path": str(descriptor_result.client_metadata_path),
        "descriptor_scaler_path": str(descriptor_result.descriptor_scaler_path),
        "membership_file": str(membership_path),
        "reuse_for_experiment_groups": [
            "baseline_uniform_hierarchical",
            "proposed_specialized_hierarchical",
        ],
        "subclusters": subclusters,
        "clients": clients,
    }


def run_offline_agglomerative_clustering(
    config_path: str | Path,
    *,
    membership_path: str | Path | None = None,
    client_metadata_path: str | Path | None = None,
    descriptor_scaler_path: str | Path | None = None,
    force_recompute: bool = False,
) -> AgglomerativeResult:
    runtime_config = load_agglomerative_runtime_config(config_path)
    resolved_membership_path = Path(membership_path) if membership_path is not None else _default_membership_path(runtime_config.cluster_id)

    if resolved_membership_path.exists() and not force_recompute:
        summary = json.loads(resolved_membership_path.read_text(encoding="utf-8"))
        return AgglomerativeResult(
            membership_path=resolved_membership_path,
            summary=summary,
            reused_existing_membership=True,
        )

    descriptor_result = build_cluster_descriptors(
        config_path,
        client_metadata_path=client_metadata_path,
        descriptor_scaler_path=descriptor_scaler_path,
    )
    model = _build_model(runtime_config.n_subclusters)
    raw_labels = model.fit_predict(descriptor_result.standardized_descriptor_matrix).tolist()
    assignments = _normalize_cluster_labels(
        descriptor_result.client_ids,
        raw_labels,
        runtime_config.fixed_subcluster_ids,
    )

    summary = _membership_summary(
        runtime_config,
        descriptor_result,
        assignments,
        resolved_membership_path,
    )
    resolved_membership_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_membership_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return AgglomerativeResult(
        membership_path=resolved_membership_path,
        summary=summary,
        reused_existing_membership=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-time offline Agglomerative Clustering.")
    parser.add_argument(
        "--config",
        dest="config_paths",
        action="append",
        help="Optional cluster config path. If omitted, all default cluster configs are processed.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute membership files even if frozen membership files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = args.config_paths or [str(path) for path in DEFAULT_CLUSTER_CONFIG_PATHS.values()]
    for config_path in config_paths:
        result = run_offline_agglomerative_clustering(
            config_path,
            force_recompute=args.force_recompute,
        )
        action = "reused" if result.reused_existing_membership else "wrote"
        print(f"{result.summary['dataset']}: {action} {result.membership_path}")


if __name__ == "__main__":
    main()
