from __future__ import annotations

from pathlib import Path

from .schema_validation import (
    DatasetConfig,
    DatasetInspection,
    load_cluster_config,
    inspect_dataset_files,
    resolve_profile_file_paths,
    resolve_training_file_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CLUSTER_CONFIG_PATHS = {
    1: REPO_ROOT / "configs/cluster1_hai.yaml",
    2: REPO_ROOT / "configs/cluster2_ton_iot.yaml",
    3: REPO_ROOT / "configs/cluster3_wustl.yaml",
}


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    return load_cluster_config(config_path)


def load_dataset_for_profile(config_path: str | Path) -> DatasetInspection:
    config = load_cluster_config(config_path)
    file_paths = resolve_profile_file_paths(config)
    return inspect_dataset_files(
        config,
        file_paths,
        source_kind="profile",
        require_consistent_schema=False,
    )


def load_dataset_for_training(config_path: str | Path) -> DatasetInspection:
    config = load_cluster_config(config_path)
    file_paths = resolve_training_file_paths(config)
    return inspect_dataset_files(
        config,
        file_paths,
        source_kind="training",
        require_consistent_schema=True,
    )


def load_cluster1_dataset_for_training(config_path: str | Path = DEFAULT_CLUSTER_CONFIG_PATHS[1]) -> DatasetInspection:
    return load_dataset_for_training(config_path)


def load_cluster2_dataset_for_training(config_path: str | Path = DEFAULT_CLUSTER_CONFIG_PATHS[2]) -> DatasetInspection:
    return load_dataset_for_training(config_path)


def load_cluster3_dataset_for_training(config_path: str | Path = DEFAULT_CLUSTER_CONFIG_PATHS[3]) -> DatasetInspection:
    return load_dataset_for_training(config_path)


def load_cluster1_dataset_for_profile(config_path: str | Path = DEFAULT_CLUSTER_CONFIG_PATHS[1]) -> DatasetInspection:
    return load_dataset_for_profile(config_path)


def load_cluster2_dataset_for_profile(config_path: str | Path = DEFAULT_CLUSTER_CONFIG_PATHS[2]) -> DatasetInspection:
    return load_dataset_for_profile(config_path)


def load_cluster3_dataset_for_profile(config_path: str | Path = DEFAULT_CLUSTER_CONFIG_PATHS[3]) -> DatasetInspection:
    return load_dataset_for_profile(config_path)
