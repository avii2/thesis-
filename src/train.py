from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import yaml
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.ledger.metadata_schema import LedgerRecord, canonical_sha256, model_version_for_round
from src.ledger.mock_ledger import JSONLMockLedger, MAIN_CLUSTER_HEAD_ROLE
from src.train_ablation import (
    run_cluster1_fedavg_tcn_ablation,
    run_cluster2_fedavg_mlp_ablation,
    run_cluster3_fedavg_cnn1d_ablation,
)
from src.train_cluster1_proposed import run_cluster1_proposed
from src.train_cluster2_proposed import run_cluster2_proposed
from src.train_cluster3_proposed import run_cluster3_proposed
from src.train_flat_baseline import run_flat_baseline_experiments
from src.train_hierarchical_baseline import run_hierarchical_baseline_experiments

matplotlib.use("Agg")


SUPPORTED_EXPERIMENT_IDS = (
    "A_C1",
    "A_C2",
    "A_C3",
    "B_C1",
    "B_C2",
    "B_C3",
    "P_C1",
    "P_C2",
    "P_C3",
    "AB_C1_FEDAVG_TCN",
    "AB_C2_FEDAVG_MLP",
    "AB_C3_FEDAVG_CNN1D",
)

DEFAULT_RUN_ALL_EXPERIMENT_IDS = (
    "A_C1",
    "A_C2",
    "A_C3",
    "B_C1",
    "B_C2",
    "B_C3",
    "P_C1",
    "P_C2",
    "P_C3",
)

MATRIX_REQUIRED_FIELDS = (
    "experiment_id",
    "run_category",
    "cluster_id",
    "dataset",
    "model",
    "fl_method",
    "aggregation",
    "hierarchy",
    "clustering_method",
    "n_subclusters",
    "descriptor",
    "run_repeats",
    "notes",
)

DEFAULT_MATRIX_PATH = "docs/EXPERIMENT_MATRIX.csv"
DEFAULT_BASELINE_FLAT_CONFIG = "configs/baseline_flat.yaml"
DEFAULT_BASELINE_HIERARCHICAL_CONFIG = "configs/baseline_hierarchical.yaml"
DEFAULT_PROPOSED_CONFIG = "configs/proposed.yaml"
DEFAULT_ABLATION_CLUSTER1_CONFIG = "configs/ablation_cluster1_fedbn.yaml"
DEFAULT_ABLATION_CLUSTER2_CONFIG = "configs/ablation_cluster2_fedprox.yaml"
DEFAULT_ABLATION_CLUSTER3_CONFIG = "configs/ablation_cluster3_scaffold.yaml"
PLOT_FILE_EXTENSION = ".svg"

MULTISEED_REQUIRED_METRICS = (
    ("test_accuracy", "Accuracy"),
    ("test_precision", "Precision"),
    ("test_recall", "Recall"),
    ("test_f1", "F1"),
    ("test_auroc", "AUROC"),
    ("test_pr_auc", "PR-AUC"),
    ("test_fpr", "FPR"),
    ("wall_clock_training_seconds", "Wall-clock Time"),
    ("total_communication_cost_bytes", "Communication Cost"),
)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    run_category: str
    cluster_id: int
    dataset: str
    model: str
    fl_method: str
    aggregation: str
    hierarchy: str
    clustering_method: str
    n_subclusters: int
    descriptor: str
    run_repeats: int
    notes: str


@dataclass(frozen=True)
class ConfigExperiment:
    experiment_id: str
    experiment_group: str
    config_path: Path
    cluster_config_path: Path
    cluster_id: int
    dataset_name: str
    entry: Mapping[str, Any]


@dataclass(frozen=True)
class ExperimentRunArtifacts:
    experiment_id: str
    cluster_id: int
    output_dir: Path
    summary_path: Path
    round_metrics_path: Path
    metrics_csv_path: Path
    model_manifest_path: Path
    ledger_path: Path
    convergence_plot_path: Path
    summary: Mapping[str, Any]
    seed: int | None = None


@dataclass(frozen=True)
class FailedSeedRun:
    experiment_id: str
    seed: int
    error: str
    failure_marker_path: Path | None = None


@dataclass(frozen=True)
class ExperimentBatchResult:
    experiments: list[ExperimentRunArtifacts]
    summary_csv_path: Path | None
    comparison_plot_paths: list[Path] = field(default_factory=list)
    mean_std_summary_csv_path: Path | None = None
    mean_std_results_csv_path: Path | None = None
    mean_std_markdown_path: Path | None = None
    failed_seed_runs: list[FailedSeedRun] = field(default_factory=list)


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return data


def _resolve_path(base_config_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()

    repo_root_candidate = base_config_path.parents[1] / configured_path
    if repo_root_candidate.exists():
        return repo_root_candidate.resolve()

    sibling_candidate = base_config_path.parent / configured_path
    if sibling_candidate.exists():
        return sibling_candidate.resolve()

    return repo_root_candidate.resolve()


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _coerce_csv_value(value: str) -> Any:
    text = str(value).strip()
    if not text:
        return None
    if text == "metric_unavailable_single_class":
        return text
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _metric_as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped == "metric_unavailable_single_class":
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _ensure_sha256_hash(value: Any, *, fallback_payload: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.startswith("sha256:") and len(normalized) == len("sha256:") + 64:
            return normalized
        if len(normalized) == 64 and all(character in "0123456789abcdefABCDEF" for character in normalized):
            return f"sha256:{normalized.lower()}"
    return canonical_sha256(fallback_payload)


def _load_round_metrics(round_metrics_path: str | Path) -> list[dict[str, Any]]:
    round_metrics_path = Path(round_metrics_path)
    if not round_metrics_path.exists():
        raise FileNotFoundError(f"Missing round metrics file: {round_metrics_path}")
    with round_metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _coerce_csv_value(value) for key, value in row.items()} for row in reader]


def _load_metrics_row(metrics_csv_path: str | Path) -> dict[str, Any]:
    metrics_csv_path = Path(metrics_csv_path)
    if not metrics_csv_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV file: {metrics_csv_path}")
    with metrics_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{metrics_csv_path}: expected exactly one metrics row, observed {len(rows)}.")
    return {key: _coerce_csv_value(value) for key, value in rows[0].items()}


def load_experiment_matrix(matrix_path: str | Path = DEFAULT_MATRIX_PATH) -> dict[str, ExperimentSpec]:
    matrix_path = Path(matrix_path).resolve()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Experiment matrix file does not exist: {matrix_path}")

    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        missing_fields = [field for field in MATRIX_REQUIRED_FIELDS if field not in header]
        if missing_fields:
            raise ValueError(f"{matrix_path}: experiment matrix is missing fields {missing_fields}.")

        experiments: dict[str, ExperimentSpec] = {}
        for row in reader:
            experiment_id = str(row["experiment_id"]).strip()
            if not experiment_id:
                raise ValueError(f"{matrix_path}: encountered a blank experiment_id row.")
            if experiment_id in experiments:
                raise ValueError(f"{matrix_path}: duplicate experiment_id {experiment_id!r}.")
            experiments[experiment_id] = ExperimentSpec(
                experiment_id=experiment_id,
                run_category=str(row["run_category"]).strip(),
                cluster_id=int(str(row["cluster_id"]).strip()),
                dataset=str(row["dataset"]).strip(),
                model=str(row["model"]).strip(),
                fl_method=str(row["fl_method"]).strip(),
                aggregation=str(row["aggregation"]).strip(),
                hierarchy=str(row["hierarchy"]).strip(),
                clustering_method=str(row["clustering_method"]).strip(),
                n_subclusters=int(str(row["n_subclusters"]).strip()),
                descriptor=str(row["descriptor"]).strip(),
                run_repeats=int(str(row["run_repeats"]).strip()),
                notes=str(row["notes"]).strip(),
            )
    return experiments


def load_config_registry(
    *,
    baseline_flat_config_path: str | Path = DEFAULT_BASELINE_FLAT_CONFIG,
    baseline_hierarchical_config_path: str | Path = DEFAULT_BASELINE_HIERARCHICAL_CONFIG,
    proposed_config_path: str | Path = DEFAULT_PROPOSED_CONFIG,
    ablation_cluster1_config_path: str | Path = DEFAULT_ABLATION_CLUSTER1_CONFIG,
    ablation_cluster2_config_path: str | Path = DEFAULT_ABLATION_CLUSTER2_CONFIG,
    ablation_cluster3_config_path: str | Path = DEFAULT_ABLATION_CLUSTER3_CONFIG,
) -> dict[str, ConfigExperiment]:
    config_paths = (
        Path(baseline_flat_config_path).resolve(),
        Path(baseline_hierarchical_config_path).resolve(),
        Path(proposed_config_path).resolve(),
    )

    registry: dict[str, ConfigExperiment] = {}
    for config_path in config_paths:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        config = _load_yaml(config_path)
        experiment_group = str(config.get("experiment_group", "")).strip()
        if not experiment_group:
            raise ValueError(f"{config_path}: missing experiment_group.")
        clusters = config.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            raise ValueError(f"{config_path}: expected a non-empty clusters list.")

        for entry in clusters:
            if not isinstance(entry, Mapping):
                raise ValueError(f"{config_path}: each cluster entry must be a mapping.")
            experiment_id = str(entry.get("experiment_id", "")).strip()
            if not experiment_id:
                raise ValueError(f"{config_path}: found cluster entry without experiment_id.")
            if experiment_id in registry:
                raise ValueError(f"Duplicate experiment_id {experiment_id!r} across configs.")
            cluster_config_path = _resolve_path(config_path, str(entry["cluster_config"]))
            if not cluster_config_path.exists():
                raise FileNotFoundError(
                    f"{config_path}: cluster_config for {experiment_id} does not exist: {cluster_config_path}"
                )
            cluster_yaml = _load_yaml(cluster_config_path)
            cluster_section = cluster_yaml.get("cluster")
            if not isinstance(cluster_section, Mapping):
                raise ValueError(f"{cluster_config_path}: missing cluster metadata.")
            registry[experiment_id] = ConfigExperiment(
                experiment_id=experiment_id,
                experiment_group=experiment_group,
                config_path=config_path,
                cluster_config_path=cluster_config_path,
                cluster_id=int(cluster_section["id"]),
                dataset_name=str(cluster_section["dataset_name"]),
                entry=entry,
            )

    ablation_config_paths = (
        Path(ablation_cluster1_config_path).resolve(),
        Path(ablation_cluster2_config_path).resolve(),
        Path(ablation_cluster3_config_path).resolve(),
    )
    for config_path in ablation_config_paths:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        config = _load_yaml(config_path)
        comparisons = config.get("comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            raise ValueError(f"{config_path}: expected a non-empty comparisons list.")

        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                raise ValueError(f"{config_path}: each comparison entry must be a mapping.")
            control = comparison.get("control")
            if not isinstance(control, Mapping):
                raise ValueError(f"{config_path}: each comparison must define a control mapping.")
            experiment_id = str(control.get("experiment_id", "")).strip()
            if experiment_id not in SUPPORTED_EXPERIMENT_IDS or not experiment_id.startswith("AB_"):
                continue
            if experiment_id in registry:
                raise ValueError(f"Duplicate experiment_id {experiment_id!r} across configs.")
            cluster_config_path = _resolve_path(config_path, str(control["cluster_config"]))
            if not cluster_config_path.exists():
                raise FileNotFoundError(
                    f"{config_path}: cluster_config for {experiment_id} does not exist: {cluster_config_path}"
                )
            cluster_yaml = _load_yaml(cluster_config_path)
            cluster_section = cluster_yaml.get("cluster")
            if not isinstance(cluster_section, Mapping):
                raise ValueError(f"{cluster_config_path}: missing cluster metadata.")
            registry[experiment_id] = ConfigExperiment(
                experiment_id=experiment_id,
                experiment_group="ablation_fl_method",
                config_path=config_path,
                cluster_config_path=cluster_config_path,
                cluster_id=int(control["cluster_id"]),
                dataset_name=str(cluster_section["dataset_name"]),
                entry=control,
            )
    return registry


def validate_experiment_registry(
    matrix: Mapping[str, ExperimentSpec],
    registry: Mapping[str, ConfigExperiment],
) -> None:
    missing_matrix = [exp_id for exp_id in SUPPORTED_EXPERIMENT_IDS if exp_id not in matrix]
    if missing_matrix:
        raise ValueError(f"Experiment matrix is missing supported experiments: {missing_matrix}.")

    missing_configs = [exp_id for exp_id in SUPPORTED_EXPERIMENT_IDS if exp_id not in registry]
    if missing_configs:
        raise ValueError(f"Config registry is missing supported experiments: {missing_configs}.")

    for experiment_id in SUPPORTED_EXPERIMENT_IDS:
        spec = matrix[experiment_id]
        config_entry = registry[experiment_id]
        entry = config_entry.entry

        if spec.run_category != config_entry.experiment_group:
            raise ValueError(
                f"{experiment_id}: matrix run_category={spec.run_category!r} does not match "
                f"config experiment_group={config_entry.experiment_group!r}."
            )
        if spec.cluster_id != config_entry.cluster_id:
            raise ValueError(
                f"{experiment_id}: matrix cluster_id={spec.cluster_id} does not match "
                f"cluster config cluster.id={config_entry.cluster_id}."
            )
        if spec.hierarchy != str(entry.get("hierarchy", "")).strip():
            raise ValueError(
                f"{experiment_id}: matrix hierarchy={spec.hierarchy!r} does not match "
                f"config hierarchy={entry.get('hierarchy')!r}."
            )
        if spec.clustering_method != str(entry.get("clustering_method", "")).strip():
            raise ValueError(
                f"{experiment_id}: matrix clustering_method={spec.clustering_method!r} does not match "
                f"config clustering_method={entry.get('clustering_method')!r}."
            )
        if spec.n_subclusters != int(entry.get("n_subclusters", 0)):
            raise ValueError(
                f"{experiment_id}: matrix n_subclusters={spec.n_subclusters} does not match "
                f"config n_subclusters={entry.get('n_subclusters')!r}."
            )
        if spec.model != str(entry.get("model_family", "")).strip():
            raise ValueError(
                f"{experiment_id}: matrix model={spec.model!r} does not match "
                f"config model_family={entry.get('model_family')!r}."
            )
        if spec.fl_method != str(entry.get("fl_method", "")).strip():
            raise ValueError(
                f"{experiment_id}: matrix fl_method={spec.fl_method!r} does not match "
                f"config fl_method={entry.get('fl_method')!r}."
            )
        if spec.aggregation != str(entry.get("aggregation", "")).strip():
            raise ValueError(
                f"{experiment_id}: matrix aggregation={spec.aggregation!r} does not match "
                f"config aggregation={entry.get('aggregation')!r}."
            )
        config_descriptor = str(entry.get("descriptor", "none")).strip()
        if spec.descriptor != config_descriptor:
            raise ValueError(
                f"{experiment_id}: matrix descriptor={spec.descriptor!r} does not match "
                f"config descriptor={config_descriptor!r}."
            )


def _ensure_output_directories(output_root: str | Path) -> dict[str, Path]:
    output_root = Path(output_root).resolve()
    paths = {
        "root": output_root,
        "runs": output_root / "runs",
        "metrics": output_root / "metrics",
        "models": output_root / "models",
        "plots": output_root / "plots",
        "ledgers": output_root / "ledgers",
        "predictions": output_root / "predictions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _normalize_seeds(seeds: Sequence[int] | None) -> list[int]:
    if seeds is None:
        return []
    normalized: list[int] = []
    seen: set[int] = set()
    for value in seeds:
        seed = int(value)
        if seed in seen:
            continue
        seen.add(seed)
        normalized.append(seed)
    if not normalized:
        raise ValueError("At least one seed must be provided when using multiseed mode.")
    return normalized


def _seed_label(seed: int | None) -> str | None:
    if seed is None:
        return None
    return f"seed_{int(seed)}"


def _artifact_destination_paths(
    *,
    experiment_id: str,
    output_root: Path,
    seed: int | None = None,
) -> dict[str, Path]:
    seed_label = _seed_label(seed)
    run_dir = output_root / "runs" / experiment_id
    model_dir = output_root / "models" / experiment_id
    if seed_label is not None:
        run_dir = run_dir / seed_label
        model_dir = model_dir / seed_label
        metrics_name = f"{experiment_id}_{seed_label}_metrics.csv"
        ledger_name = f"{experiment_id}_{seed_label}_ledger.jsonl"
        plot_name = f"convergence_{experiment_id}_{seed_label}{PLOT_FILE_EXTENSION}"
    else:
        metrics_name = f"{experiment_id}_metrics.csv"
        ledger_name = f"{experiment_id}_ledger.jsonl"
        plot_name = f"convergence_{experiment_id}{PLOT_FILE_EXTENSION}"
    return {
        "output_dir": run_dir,
        "summary_path": run_dir / "run_summary.json",
        "round_metrics_path": run_dir / "round_metrics.csv",
        "metrics_csv_path": output_root / "metrics" / metrics_name,
        "ledger_path": output_root / "ledgers" / ledger_name,
        "convergence_plot_path": output_root / "plots" / plot_name,
        "model_manifest_path": model_dir / "model_manifest.json",
        "failure_marker_path": run_dir / "FAILED.json",
    }


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _copy_prediction_outputs(
    *,
    experiment_id: str,
    summary: Mapping[str, Any],
    output_root: Path,
) -> dict[str, str] | None:
    raw_prediction_outputs = summary.get("prediction_outputs")
    if not isinstance(raw_prediction_outputs, Mapping):
        return None

    prediction_dir = output_root / "predictions" / experiment_id
    prediction_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {"prediction_dir": str(prediction_dir)}
    for key in (
        "validation_predictions_path",
        "test_predictions_path",
        "selected_threshold_path",
    ):
        configured_source = raw_prediction_outputs.get(key)
        if not configured_source:
            continue
        source_path = Path(str(configured_source)).resolve()
        if not source_path.exists():
            continue
        destination_path = prediction_dir / source_path.name
        copied_path = _copy_file(source_path, destination_path)
        copied[key] = str(copied_path)
    return copied


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_single_row_csv(path: Path, row: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def _write_failed_seed_marker(
    *,
    experiment_id: str,
    seed: int,
    error: str,
    output_root: Path,
) -> Path:
    failure_marker_path = _artifact_destination_paths(
        experiment_id=experiment_id,
        output_root=output_root,
        seed=seed,
    )["failure_marker_path"]
    payload = {
        "experiment_id": experiment_id,
        "seed": int(seed),
        "status": "FAILED",
        "error": str(error),
    }
    return _write_json(failure_marker_path, payload)


def _selected_experiment_ids(experiment_ids: Sequence[str] | None, *, run_all: bool) -> list[str]:
    if experiment_ids:
        selected = _dedupe_preserve_order(experiment_ids)
    else:
        selected = list(DEFAULT_RUN_ALL_EXPERIMENT_IDS) if run_all or experiment_ids is None else []
    if not selected:
        raise ValueError("No experiment ids selected.")
    unsupported = [exp_id for exp_id in selected if exp_id not in SUPPORTED_EXPERIMENT_IDS]
    if unsupported:
        raise ValueError(
            "This experiment runner currently supports the required A/B/P experiments plus the dedicated AB_* ablations. "
            f"Unsupported selections: {unsupported}."
        )
    return selected


def _dispatch_experiment(
    experiment_id: str,
    *,
    registry: Mapping[str, ConfigExperiment],
    smoke_test: bool,
    rounds: int | None,
    local_epochs: int | None,
    batch_size: int | None,
    seed: int | None,
    max_train_examples_per_client: int | None,
    max_eval_examples_per_client: int | None,
    output_root: Path,
) -> Any:
    config_entry = registry[experiment_id]
    if experiment_id.startswith("A_"):
        results = run_flat_baseline_experiments(
            baseline_config_path=config_entry.config_path,
            experiment_ids=[experiment_id],
            smoke_test=smoke_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
        if len(results) != 1:
            raise ValueError(f"{experiment_id}: expected one flat-baseline result, observed {len(results)}.")
        return results[0]

    if experiment_id.startswith("B_"):
        results = run_hierarchical_baseline_experiments(
            baseline_config_path=config_entry.config_path,
            experiment_ids=[experiment_id],
            smoke_test=smoke_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
        if len(results) != 1:
            raise ValueError(f"{experiment_id}: expected one hierarchical-baseline result, observed {len(results)}.")
        return results[0]

    if experiment_id == "P_C1":
        return run_cluster1_proposed(
            proposed_config_path=config_entry.config_path,
            smoke_test=smoke_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
    if experiment_id == "P_C2":
        return run_cluster2_proposed(
            proposed_config_path=config_entry.config_path,
            smoke_test=smoke_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
    if experiment_id == "P_C3":
        return run_cluster3_proposed(
            proposed_config_path=config_entry.config_path,
            smoke_test=smoke_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
    if experiment_id == "AB_C1_FEDAVG_TCN":
        return run_cluster1_fedavg_tcn_ablation(
            ablation_config_path=config_entry.config_path,
            smoke_test=smoke_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
    if experiment_id == "AB_C2_FEDAVG_MLP":
        return run_cluster2_fedavg_mlp_ablation(
            ablation_config_path=config_entry.config_path,
            smoke_test=smoke_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
    if experiment_id == "AB_C3_FEDAVG_CNN1D":
        return run_cluster3_fedavg_cnn1d_ablation(
            ablation_config_path=config_entry.config_path,
            smoke_test=smoke_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )

    raise ValueError(f"{experiment_id}: unsupported experiment dispatch.")


def _synthetic_flat_membership(summary: Mapping[str, Any]) -> dict[str, list[str]]:
    data_summary = summary.get("data_summary")
    if not isinstance(data_summary, Mapping):
        return {"MAIN": []}
    counts = data_summary.get("client_train_sample_counts")
    if not isinstance(counts, Mapping):
        return {"MAIN": []}
    return {"MAIN": sorted(str(client_id) for client_id in counts.keys())}


def _ledger_subcluster_metadata(
    spec: ExperimentSpec,
    summary: Mapping[str, Any],
) -> tuple[int, int, int, str, str | None]:
    if bool(summary.get("subcluster_layer_used", False)):
        subcluster_count = int(summary.get("n_subclusters", spec.n_subclusters))
        train_counts = summary.get("best_round_subcluster_train_sample_counts")
        if isinstance(train_counts, Mapping):
            effective_sample_count = int(sum(int(value) for value in train_counts.values()))
        else:
            effective_sample_count = 0
        participant_count = subcluster_count
        membership_hash = _ensure_sha256_hash(
            summary.get("membership_hash"),
            fallback_payload={
                "membership_file_used": summary.get("membership_file_used"),
                "subcluster_client_counts": summary.get("subcluster_client_counts"),
                "best_round_subcluster_client_counts": summary.get("best_round_subcluster_client_counts"),
            },
        )
        subcluster_digest = canonical_sha256(
            summary.get("best_round_subcluster_client_counts")
            or summary.get("subcluster_client_counts")
            or {}
        )
        return subcluster_count, effective_sample_count, participant_count, membership_hash, subcluster_digest

    synthetic_membership = _synthetic_flat_membership(summary)
    data_summary = summary.get("data_summary")
    if isinstance(data_summary, Mapping):
        train_counts = data_summary.get("client_train_sample_counts")
        if isinstance(train_counts, Mapping):
            effective_sample_count = int(sum(int(value) for value in train_counts.values()))
            participant_count = int(len(train_counts))
        else:
            effective_sample_count = 0
            participant_count = 0
    else:
        effective_sample_count = 0
        participant_count = 0
    return 1, effective_sample_count, participant_count, canonical_sha256(synthetic_membership), canonical_sha256(
        synthetic_membership
    )


def _clustering_configuration_hash(spec: ExperimentSpec, summary: Mapping[str, Any]) -> str:
    payload = {
        "hierarchy": spec.hierarchy,
        "clustering_method": spec.clustering_method,
        "descriptor": spec.descriptor,
        "n_subclusters": spec.n_subclusters,
        "membership_file_used": summary.get("membership_file_used"),
        "membership_hash": summary.get("membership_hash"),
    }
    return canonical_sha256(payload)


def _write_ledger_for_experiment(
    *,
    experiment_id: str,
    spec: ExperimentSpec,
    summary: Mapping[str, Any],
    round_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    seed: int | None = None,
) -> Path:
    ledger_path = _artifact_destination_paths(
        experiment_id=experiment_id,
        output_root=output_root,
        seed=seed,
    )["ledger_path"]
    if ledger_path.exists():
        ledger_path.unlink()
    ledger = JSONLMockLedger(ledger_path)
    cluster_head_id = f"MC{spec.cluster_id}_HEAD"
    previous_hash: str | None = None
    subcluster_count, effective_sample_count, participant_count, membership_hash, subcluster_digest = (
        _ledger_subcluster_metadata(spec, summary)
    )
    clustering_configuration_hash = _clustering_configuration_hash(spec, summary)
    export_base_time = datetime.now(timezone.utc)

    for export_index, round_row in enumerate(round_rows):
        round_id = int(round_row.get("round", export_index + 1))
        timestamp_start = (export_base_time + timedelta(milliseconds=export_index * 2)).isoformat()
        timestamp_end = (export_base_time + timedelta(milliseconds=export_index * 2 + 1)).isoformat()
        new_hash = canonical_sha256(
            {
                "experiment_id": experiment_id,
                "cluster_id": spec.cluster_id,
                "round_id": round_id,
                "previous_main_model_hash": previous_hash,
                "round_metrics": round_row,
                "summary_digest": {
                    "model_family": summary.get("model_family"),
                    "fl_method": summary.get("fl_method"),
                    "aggregation": summary.get("aggregation"),
                },
            }
        )
        record = LedgerRecord(
            round_id=round_id,
            cluster_id=spec.cluster_id,
            cluster_head_id=cluster_head_id,
            model_version=model_version_for_round(spec.cluster_id, round_id),
            previous_main_model_hash=previous_hash,
            new_main_model_hash=new_hash,
            clustering_method=spec.clustering_method,
            clustering_configuration_hash=clustering_configuration_hash,
            subcluster_count=subcluster_count,
            subcluster_membership_hash=membership_hash,
            fl_method=str(summary.get("fl_method", spec.fl_method)),
            aggregation_rule=str(summary.get("aggregation", spec.aggregation)),
            effective_sample_count=effective_sample_count,
            participant_count=participant_count,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            submitter_identity=cluster_head_id,
            subcluster_digest=subcluster_digest,
        )
        ledger.append_record(
            record,
            actor_role=MAIN_CLUSTER_HEAD_ROLE,
            actor_identity=cluster_head_id,
        )
        previous_hash = new_hash

    return ledger_path


def _write_model_manifest(
    *,
    experiment_id: str,
    spec: ExperimentSpec,
    summary: Mapping[str, Any],
    output_root: Path,
    seed: int | None = None,
) -> Path:
    manifest_path = _artifact_destination_paths(
        experiment_id=experiment_id,
        output_root=output_root,
        seed=seed,
    )["model_manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    model_hash = canonical_sha256(
        {
            "experiment_id": experiment_id,
            "cluster_id": spec.cluster_id,
            "model_family": summary.get("model_family", spec.model),
            "best_validation_round": summary.get("best_validation_round"),
            "best_round_validation_metrics": summary.get("best_round_validation_metrics"),
            "best_round_test_metrics": summary.get("best_round_test_metrics"),
            "seed": seed,
        }
    )
    manifest = {
        "experiment_id": experiment_id,
        "cluster_id": spec.cluster_id,
        "dataset": summary.get("dataset", spec.dataset),
        "model_family": summary.get("model_family", spec.model),
        "model_version": model_version_for_round(
            spec.cluster_id,
            int(summary.get("best_validation_round", 0)),
        ),
        "model_hash": model_hash,
        "artifact_type": "metadata_manifest",
        "weights_emitted_by_runner": False,
        "seed": seed,
        "summary_path": str(summary.get("summary_path", "")),
        "round_metrics_path": str(summary.get("round_metrics_path", "")),
        "metrics_csv_path": str(summary.get("metrics_csv_path", "")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_note_svg(path: Path, *, title: str, note: str) -> Path:
    content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="840" height="460" viewBox="0 0 840 460">
  <rect width="840" height="460" fill="#f8fafc"/>
  <text x="32" y="56" font-size="24" font-family="Helvetica, Arial, sans-serif" fill="#0f172a">{_svg_escape(title)}</text>
  <text x="32" y="98" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#334155">{_svg_escape(note)}</text>
</svg>
"""
    path.write_text(content, encoding="utf-8")
    return path


def _experiment_color(experiment_id: str) -> str:
    if experiment_id.startswith("A_"):
        return "#2563eb"
    if experiment_id.startswith("B_"):
        return "#f59e0b"
    if experiment_id.startswith("P_"):
        return "#059669"
    if experiment_id.startswith("AB_"):
        return "#7c3aed"
    return "#475569"


def _metric_label(metric_key: str) -> str:
    return {
        "best_validation_f1": "Best Validation F1",
        "test_f1": "Test F1",
        "train_f1": "Train F1",
        "validation_f1": "Validation F1",
    }.get(metric_key, metric_key.replace("_", " ").title())


def _polyline_points(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    chart_left: float,
    chart_top: float,
    chart_width: float,
    chart_height: float,
) -> str:
    if not xs:
        return ""
    min_x = min(xs)
    max_x = max(xs)
    x_span = max(max_x - min_x, 1.0)
    points: list[str] = []
    for x_value, y_value in zip(xs, ys):
        x = chart_left + ((x_value - min_x) / x_span) * chart_width
        y = chart_top + (1.0 - y_value) * chart_height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _write_convergence_plot(
    experiment_id: str,
    round_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    seed: int | None = None,
) -> Path:
    plot_path = _artifact_destination_paths(
        experiment_id=experiment_id,
        output_root=output_root,
        seed=seed,
    )["convergence_plot_path"]
    if not round_rows:
        return _write_note_svg(
            plot_path,
            title=f"{experiment_id} convergence",
            note="No round metrics were available to plot.",
        )

    xs = [float(row["round"]) for row in round_rows]
    train_ys = [_metric_as_float(row.get("train_f1")) for row in round_rows]
    validation_ys = [_metric_as_float(row.get("validation_f1")) for row in round_rows]
    test_ys = [_metric_as_float(row.get("test_f1")) for row in round_rows]
    series = [
        ("train_f1", "#2563eb", train_ys),
        ("validation_f1", "#dc2626", validation_ys),
        ("test_f1", "#059669", test_ys),
    ]
    available_series = [
        (name, color, [value for value in values if value is not None], values)
        for name, color, values in series
        if any(value is not None for value in values)
    ]
    if not available_series:
        return _write_note_svg(
            plot_path,
            title=f"{experiment_id} convergence",
            note="Round metrics were present, but no numeric F1 values were available to plot.",
        )
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    for name, color, _, raw_values in available_series:
        filtered_pairs = [(x_value, y_value) for x_value, y_value in zip(xs, raw_values) if y_value is not None]
        filtered_xs = [pair[0] for pair in filtered_pairs]
        filtered_ys = [pair[1] for pair in filtered_pairs]
        ax.plot(
            filtered_xs,
            filtered_ys,
            marker="o",
            linewidth=2.5,
            markersize=5,
            color=color,
            label=_metric_label(name),
        )

    ax.set_title(f"{experiment_id} Convergence", fontsize=16, pad=14)
    ax.set_xlabel("Round")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if len(set(xs)) == 1:
        only_round = int(xs[0])
        ax.set_xticks([only_round])
        ax.set_xlim(only_round - 0.5, only_round + 0.5)
    ax.legend(frameon=False, loc="best")
    fig.savefig(plot_path, format="svg")
    plt.close(fig)
    return plot_path


def _write_comparison_plot(
    *,
    plot_name: str,
    title: str,
    metric_key: str,
    summary_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> Path:
    plot_path = output_root / "plots" / f"{plot_name}{PLOT_FILE_EXTENSION}"
    ordered_rows = [(str(row["experiment_id"]), _metric_as_float(row.get(metric_key))) for row in summary_rows]
    ordered_rows = [(experiment_id, value) for experiment_id, value in ordered_rows if value is not None]
    if not ordered_rows:
        return _write_note_svg(
            plot_path,
            title=title,
            note=f"No numeric {metric_key} values were available for comparison.",
        )

    experiment_ids = [experiment_id for experiment_id, _ in ordered_rows]
    values = [float(value) for _, value in ordered_rows]
    colors = [_experiment_color(experiment_id) for experiment_id in experiment_ids]

    plt.close("all")
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    bars = ax.bar(experiment_ids, values, color=colors, edgecolor="#0f172a", linewidth=0.4)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel("Experiment ID")
    ax.set_ylabel(_metric_label(metric_key))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(value + 0.02, 0.99),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0f172a",
        )

    fig.savefig(plot_path, format="svg")
    plt.close(fig)
    return plot_path


def _write_summary_csv(summary_rows: Sequence[Mapping[str, Any]], output_root: Path) -> Path:
    summary_path = output_root / "metrics" / "summary_all_experiments.csv"
    preferred_columns = [
        "experiment_id",
        "cluster_id",
        "dataset",
        "run_category",
        "hierarchy",
        "model_family",
        "fl_method",
        "aggregation",
        "clustering_method",
        "rounds",
        "best_validation_round",
        "best_validation_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auroc",
        "test_pr_auc",
        "test_fpr",
        "communication_cost_per_round_bytes",
        "total_communication_cost_bytes",
        "wall_clock_training_seconds",
        "summary_path",
        "round_metrics_path",
        "metrics_csv_path",
        "model_manifest_path",
        "ledger_path",
        "convergence_plot_path",
    ]
    extra_columns = sorted(
        {
            key
            for row in summary_rows
            for key in row.keys()
            if key not in preferred_columns
        }
    )
    fieldnames = preferred_columns + extra_columns
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return summary_path


def _mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    numeric_values = [float(value) for value in values]
    return mean(numeric_values), pstdev(numeric_values)


def _comma_join(items: Sequence[int]) -> str:
    if not items:
        return ""
    return ",".join(str(int(item)) for item in items)


def _seed_artifact_state(
    *,
    experiment_id: str,
    seed: int,
    output_root: Path,
) -> tuple[str, dict[str, Any] | None, str]:
    artifact_paths = _artifact_destination_paths(
        experiment_id=experiment_id,
        output_root=output_root,
        seed=seed,
    )
    failure_marker_path = artifact_paths["failure_marker_path"]
    metrics_csv_path = artifact_paths["metrics_csv_path"]
    if failure_marker_path.exists():
        failure_payload = json.loads(failure_marker_path.read_text(encoding="utf-8"))
        return "FAILED", None, str(failure_payload.get("error", "seed run failed"))
    if metrics_csv_path.exists():
        return "COMPLETE", _load_metrics_row(metrics_csv_path), ""
    related_artifacts = [
        artifact_paths["summary_path"],
        artifact_paths["round_metrics_path"],
        artifact_paths["ledger_path"],
        artifact_paths["convergence_plot_path"],
        artifact_paths["model_manifest_path"],
    ]
    if any(path.exists() for path in related_artifacts):
        return "PARTIAL", None, "Seed-specific outputs are incomplete."
    return "MISSING", None, "Seed-specific outputs were not found."


def write_multiseed_reports(
    *,
    experiment_ids: Sequence[str],
    seeds: Sequence[int],
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
    output_root: str | Path = "outputs",
    markdown_output_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    normalized_seeds = _normalize_seeds(seeds)
    output_root = Path(output_root).resolve()
    matrix = load_experiment_matrix(matrix_path)

    metrics_output_path = output_root / "metrics" / "summary_all_experiments_mean_std.csv"
    reports_output_path = output_root / "reports" / "results_mean_std_table.csv"
    if markdown_output_path is None:
        markdown_output_path = Path(__file__).resolve().parents[1] / "docs" / "RESULTS_SUMMARY_MEAN_STD.md"
    else:
        markdown_output_path = Path(markdown_output_path).resolve()
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    reports_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    markdown_lines: list[str] = ["# RESULTS SUMMARY MEAN STD", ""]

    coverage_lines = [
        "## 1. Seed coverage",
        "| experiment_id | status | successful_seeds | missing_seeds | failed_seeds | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    mean_std_lines = [
        "",
        "## 2. Mean ± std across seeds",
        "| experiment_id | Accuracy | Precision | Recall | F1 | AUROC | PR-AUC | FPR | wall-clock time | communication cost | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    issue_lines = ["", "## 3. Missing or failed seeds"]
    issue_count = 0

    for experiment_id in _dedupe_preserve_order(experiment_ids):
        if experiment_id not in matrix:
            raise ValueError(f"{experiment_id}: missing from experiment matrix.")
        spec = matrix[experiment_id]
        successful_metrics: list[dict[str, Any]] = []
        successful_seeds: list[int] = []
        missing_seeds: list[int] = []
        failed_seeds: list[int] = []
        partial_seeds: list[int] = []
        notes: list[str] = []

        for seed in normalized_seeds:
            seed_state, metrics_row, note = _seed_artifact_state(
                experiment_id=experiment_id,
                seed=seed,
                output_root=output_root,
            )
            if seed_state == "COMPLETE" and metrics_row is not None:
                successful_metrics.append(metrics_row)
                successful_seeds.append(seed)
            elif seed_state == "FAILED":
                failed_seeds.append(seed)
                notes.append(f"seed {seed} FAILED: {note}")
            elif seed_state == "PARTIAL":
                partial_seeds.append(seed)
                notes.append(f"seed {seed} PARTIAL: {note}")
            else:
                missing_seeds.append(seed)
                notes.append(f"seed {seed} MISSING")

        if len(successful_seeds) == len(normalized_seeds):
            status = "COMPLETE"
        elif successful_seeds:
            status = "PARTIAL"
        elif failed_seeds or partial_seeds:
            status = "FAILED"
        else:
            status = "MISSING"

        aggregate_row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": status,
            "cluster_id": spec.cluster_id,
            "dataset": spec.dataset,
            "model": spec.model,
            "fl_method": spec.fl_method,
            "aggregation": spec.aggregation,
            "hierarchy_type": spec.hierarchy,
            "clustering_type": spec.clustering_method,
            "expected_seeds": _comma_join(normalized_seeds),
            "successful_seeds": _comma_join(successful_seeds),
            "missing_seeds": _comma_join(missing_seeds),
            "failed_seeds": _comma_join(failed_seeds),
            "partial_seeds": _comma_join(partial_seeds),
            "successful_seed_count": len(successful_seeds),
            "notes": "; ".join(notes),
        }

        for metric_key, _ in MULTISEED_REQUIRED_METRICS:
            values = [
                _metric_as_float(metrics_row.get(metric_key))
                for metrics_row in successful_metrics
                if _metric_as_float(metrics_row.get(metric_key)) is not None
            ]
            metric_mean, metric_std = _mean_std(values)
            aggregate_row[f"{metric_key}_mean"] = metric_mean
            aggregate_row[f"{metric_key}_std"] = metric_std

        summary_rows.append(aggregate_row)
        coverage_lines.append(
            f"| {experiment_id} | {status} | {_comma_join(successful_seeds) or 'none'} | "
            f"{_comma_join(missing_seeds) or 'none'} | {_comma_join(failed_seeds) or 'none'} | "
            f"{aggregate_row['notes'] or 'All requested seeds completed.'} |"
        )

        display_row = {
            "experiment_id": experiment_id,
            "cluster_id": spec.cluster_id,
            "dataset": spec.dataset,
            "model": spec.model,
            "fl_method": spec.fl_method,
            "aggregation": spec.aggregation,
            "hierarchy_type": spec.hierarchy,
            "clustering_type": spec.clustering_method,
            "status": status,
            "expected_seeds": aggregate_row["expected_seeds"],
            "successful_seeds": aggregate_row["successful_seeds"],
            "missing_seeds": aggregate_row["missing_seeds"],
            "failed_seeds": aggregate_row["failed_seeds"],
            "notes": aggregate_row["notes"],
        }
        for metric_key, label in MULTISEED_REQUIRED_METRICS:
            metric_mean = aggregate_row[f"{metric_key}_mean"]
            metric_std = aggregate_row[f"{metric_key}_std"]
            display_row[label] = (
                f"{metric_mean:.4f} ± {metric_std:.4f}"
                if metric_mean is not None and metric_std is not None
                else "NOT AVAILABLE"
            )
        table_rows.append(display_row)
        mean_std_lines.append(
            f"| {experiment_id} | {display_row['Accuracy']} | {display_row['Precision']} | "
            f"{display_row['Recall']} | {display_row['F1']} | {display_row['AUROC']} | "
            f"{display_row['PR-AUC']} | {display_row['FPR']} | {display_row['Wall-clock Time']} | "
            f"{display_row['Communication Cost']} | {status} |"
        )

        if status != "COMPLETE":
            issue_count += 1
            issue_lines.append(f"- `{experiment_id}`: {aggregate_row['notes'] or 'One or more seeds are unavailable.'}")

    if issue_count == 0:
        issue_lines.append("- None. All requested seeds are present for the selected experiments.")

    summary_fieldnames = [
        "experiment_id",
        "status",
        "cluster_id",
        "dataset",
        "model",
        "fl_method",
        "aggregation",
        "hierarchy_type",
        "clustering_type",
        "expected_seeds",
        "successful_seeds",
        "missing_seeds",
        "failed_seeds",
        "partial_seeds",
        "successful_seed_count",
    ]
    for metric_key, _ in MULTISEED_REQUIRED_METRICS:
        summary_fieldnames.extend((f"{metric_key}_mean", f"{metric_key}_std"))
    summary_fieldnames.append("notes")
    with metrics_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in summary_fieldnames})

    table_fieldnames = [
        "experiment_id",
        "cluster_id",
        "dataset",
        "model",
        "fl_method",
        "aggregation",
        "hierarchy_type",
        "clustering_type",
        "status",
        "expected_seeds",
        "successful_seeds",
        "missing_seeds",
        "failed_seeds",
    ] + [label for _, label in MULTISEED_REQUIRED_METRICS] + ["notes"]
    with reports_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fieldnames)
        writer.writeheader()
        for row in table_rows:
            writer.writerow({field: row.get(field, "") for field in table_fieldnames})

    markdown_lines.extend(coverage_lines)
    markdown_lines.extend(mean_std_lines)
    markdown_lines.extend(issue_lines)
    markdown_lines.extend(
        [
            "",
            "## 4. Notes",
            f"- Expected seeds: `{_comma_join(normalized_seeds)}`.",
            "- Means and standard deviations are computed from the available seed-specific metrics files only.",
            "- Missing or failed seeds are reported explicitly and excluded from the aggregated statistics.",
        ]
    )
    markdown_output_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return metrics_output_path, reports_output_path, markdown_output_path


def _finalize_run_outputs(
    *,
    experiment_id: str,
    spec: ExperimentSpec,
    raw_result: Any,
    output_root: Path,
    seed: int | None = None,
) -> ExperimentRunArtifacts:
    raw_output_dir = Path(raw_result.output_dir).resolve()
    raw_summary_path = Path(raw_result.summary_path).resolve()
    raw_round_metrics_path = Path(raw_result.round_metrics_path).resolve()
    raw_metrics_csv_path = Path(raw_result.metrics_csv_path).resolve()
    if not raw_output_dir.exists():
        raise FileNotFoundError(f"{experiment_id}: missing run output directory {raw_output_dir}")
    if not raw_summary_path.exists():
        raise FileNotFoundError(f"{experiment_id}: missing summary file {raw_summary_path}")
    if not raw_round_metrics_path.exists():
        raise FileNotFoundError(f"{experiment_id}: missing round metrics file {raw_round_metrics_path}")
    if not raw_metrics_csv_path.exists():
        raise FileNotFoundError(f"{experiment_id}: missing metrics CSV file {raw_metrics_csv_path}")

    destination_paths = _artifact_destination_paths(
        experiment_id=experiment_id,
        output_root=output_root,
        seed=seed,
    )
    output_dir = destination_paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    round_metrics_path = _copy_file(raw_round_metrics_path, destination_paths["round_metrics_path"])

    metrics_row = _load_metrics_row(raw_metrics_csv_path)
    if seed is not None:
        metrics_row["seed"] = int(seed)
    metrics_csv_path = _write_single_row_csv(destination_paths["metrics_csv_path"], metrics_row)

    summary = dict(raw_result.summary)
    summary["summary_path"] = str(destination_paths["summary_path"])
    summary["round_metrics_path"] = str(round_metrics_path)
    summary["metrics_csv_path"] = str(metrics_csv_path)
    copied_prediction_outputs = _copy_prediction_outputs(
        experiment_id=experiment_id,
        summary=summary,
        output_root=output_root,
    )
    if copied_prediction_outputs is not None:
        summary["prediction_outputs"] = copied_prediction_outputs
    if seed is not None:
        summary["seed"] = int(seed)
    summary_path = _write_json(destination_paths["summary_path"], summary)
    failure_marker_path = destination_paths["failure_marker_path"]
    if failure_marker_path.exists():
        failure_marker_path.unlink()

    round_rows = _load_round_metrics(round_metrics_path)
    model_manifest_path = _write_model_manifest(
        experiment_id=experiment_id,
        spec=spec,
        summary=summary,
        output_root=output_root,
        seed=seed,
    )
    ledger_path = _write_ledger_for_experiment(
        experiment_id=experiment_id,
        spec=spec,
        summary=summary,
        round_rows=round_rows,
        output_root=output_root,
        seed=seed,
    )
    convergence_plot_path = _write_convergence_plot(experiment_id, round_rows, output_root, seed=seed)

    return ExperimentRunArtifacts(
        experiment_id=experiment_id,
        cluster_id=spec.cluster_id,
        output_dir=output_dir,
        summary_path=summary_path,
        round_metrics_path=round_metrics_path,
        metrics_csv_path=metrics_csv_path,
        model_manifest_path=model_manifest_path,
        ledger_path=ledger_path,
        convergence_plot_path=convergence_plot_path,
        summary=summary,
        seed=seed,
    )


def run_experiments(
    *,
    experiment_ids: Sequence[str] | None = None,
    run_all: bool = False,
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
    baseline_flat_config_path: str | Path = DEFAULT_BASELINE_FLAT_CONFIG,
    baseline_hierarchical_config_path: str | Path = DEFAULT_BASELINE_HIERARCHICAL_CONFIG,
    proposed_config_path: str | Path = DEFAULT_PROPOSED_CONFIG,
    smoke_test: bool = False,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
    output_root: str | Path = "outputs",
) -> ExperimentBatchResult:
    normalized_seeds = _normalize_seeds(seeds)
    if seed is not None and normalized_seeds:
        raise ValueError("Use either seed or seeds, not both.")
    selected_experiments = _selected_experiment_ids(experiment_ids, run_all=run_all)
    matrix = load_experiment_matrix(matrix_path)
    registry = load_config_registry(
        baseline_flat_config_path=baseline_flat_config_path,
        baseline_hierarchical_config_path=baseline_hierarchical_config_path,
        proposed_config_path=proposed_config_path,
    )
    validate_experiment_registry(matrix, registry)
    missing_in_matrix = [exp_id for exp_id in selected_experiments if exp_id not in matrix]
    if missing_in_matrix:
        raise ValueError(f"Selected experiments are missing from the matrix: {missing_in_matrix}.")
    missing_in_registry = [exp_id for exp_id in selected_experiments if exp_id not in registry]
    if missing_in_registry:
        raise ValueError(f"Selected experiments are missing from the config registry: {missing_in_registry}.")

    output_paths = _ensure_output_directories(output_root)
    if normalized_seeds:
        artifacts: list[ExperimentRunArtifacts] = []
        failed_seed_runs: list[FailedSeedRun] = []
        for experiment_id in selected_experiments:
            for run_seed in normalized_seeds:
                try:
                    with tempfile.TemporaryDirectory(prefix=f"{experiment_id}_{run_seed}_") as staging_dir:
                        raw_result = _dispatch_experiment(
                            experiment_id,
                            registry=registry,
                            smoke_test=smoke_test,
                            rounds=rounds,
                            local_epochs=local_epochs,
                            batch_size=batch_size,
                            seed=run_seed,
                            max_train_examples_per_client=max_train_examples_per_client,
                            max_eval_examples_per_client=max_eval_examples_per_client,
                            output_root=Path(staging_dir),
                        )
                        artifacts.append(
                            _finalize_run_outputs(
                                experiment_id=experiment_id,
                                spec=matrix[experiment_id],
                                raw_result=raw_result,
                                output_root=output_paths["root"],
                                seed=run_seed,
                            )
                        )
                except Exception as exc:  # pragma: no cover - exercised via summary generation
                    failure_marker_path = _write_failed_seed_marker(
                        experiment_id=experiment_id,
                        seed=run_seed,
                        error=str(exc),
                        output_root=output_paths["root"],
                    )
                    failed_seed_runs.append(
                        FailedSeedRun(
                            experiment_id=experiment_id,
                            seed=run_seed,
                            error=str(exc),
                            failure_marker_path=failure_marker_path,
                        )
                    )

        mean_std_summary_csv_path, mean_std_results_csv_path, mean_std_markdown_path = write_multiseed_reports(
            experiment_ids=selected_experiments,
            seeds=normalized_seeds,
            matrix_path=matrix_path,
            output_root=output_paths["root"],
        )
        return ExperimentBatchResult(
            experiments=artifacts,
            summary_csv_path=None,
            comparison_plot_paths=[],
            mean_std_summary_csv_path=mean_std_summary_csv_path,
            mean_std_results_csv_path=mean_std_results_csv_path,
            mean_std_markdown_path=mean_std_markdown_path,
            failed_seed_runs=failed_seed_runs,
        )

    artifacts: list[ExperimentRunArtifacts] = []
    for experiment_id in selected_experiments:
        raw_result = _dispatch_experiment(
            experiment_id,
            registry=registry,
            smoke_test=smoke_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_paths["root"],
        )
        artifacts.append(
            _finalize_run_outputs(
                experiment_id=experiment_id,
                spec=matrix[experiment_id],
                raw_result=raw_result,
                output_root=output_paths["root"],
            )
        )

    summary_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        spec = matrix[artifact.experiment_id]
        metrics_row = _load_metrics_row(artifact.metrics_csv_path)
        metrics_row["run_category"] = spec.run_category
        metrics_row["summary_path"] = str(artifact.summary_path)
        metrics_row["round_metrics_path"] = str(artifact.round_metrics_path)
        metrics_row["metrics_csv_path"] = str(artifact.metrics_csv_path)
        metrics_row["model_manifest_path"] = str(artifact.model_manifest_path)
        metrics_row["ledger_path"] = str(artifact.ledger_path)
        metrics_row["convergence_plot_path"] = str(artifact.convergence_plot_path)
        summary_rows.append(metrics_row)

    summary_csv_path = _write_summary_csv(summary_rows, output_paths["root"])
    comparison_plot_paths = [
        _write_comparison_plot(
            plot_name="comparison_best_validation_f1",
            title="Best Validation F1 Comparison",
            metric_key="best_validation_f1",
            summary_rows=summary_rows,
            output_root=output_paths["root"],
        ),
        _write_comparison_plot(
            plot_name="comparison_test_f1",
            title="Test F1 Comparison",
            metric_key="test_f1",
            summary_rows=summary_rows,
            output_root=output_paths["root"],
        ),
    ]
    return ExperimentBatchResult(
        experiments=artifacts,
        summary_csv_path=summary_csv_path,
        comparison_plot_paths=comparison_plot_paths,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FCFL experiment matrix across the supported A/B/P experiments."
    )
    parser.add_argument(
        "--experiment-id",
        action="append",
        dest="experiment_ids",
        help="Experiment id to run. Repeatable, for example --experiment-id A_C1 --experiment-id P_C3.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all supported A/B/P experiments from docs/EXPERIMENT_MATRIX.csv.",
    )
    parser.add_argument(
        "--matrix",
        default=DEFAULT_MATRIX_PATH,
        help="Path to docs/EXPERIMENT_MATRIX.csv.",
    )
    parser.add_argument(
        "--baseline-flat-config",
        default=DEFAULT_BASELINE_FLAT_CONFIG,
        help="Path to configs/baseline_flat.yaml.",
    )
    parser.add_argument(
        "--baseline-hierarchical-config",
        default=DEFAULT_BASELINE_HIERARCHICAL_CONFIG,
        help="Path to configs/baseline_hierarchical.yaml.",
    )
    parser.add_argument(
        "--proposed-config",
        default=DEFAULT_PROPOSED_CONFIG,
        help="Path to configs/proposed.yaml.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke-test defaults from the underlying config.")
    parser.add_argument("--rounds", type=int, help="Optional override for FL rounds.")
    parser.add_argument("--local-epochs", type=int, help="Optional override for local epochs.")
    parser.add_argument("--batch-size", type=int, help="Optional override for local batch size.")
    parser.add_argument("--seed", type=int, help="Optional override for the run seed.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Run seed-specific experiments without touching legacy single-seed outputs, for example --seeds 42 123 2025.",
    )
    parser.add_argument(
        "--max-train-examples-per-client",
        type=int,
        help="Optional cap applied after split adaptation for smoke-sized training.",
    )
    parser.add_argument(
        "--max-eval-examples-per-client",
        type=int,
        help="Optional cap applied after split adaptation for validation/test evaluation.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output directory. Defaults to outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seed is not None and args.seeds:
        raise SystemExit("Use either --seed or --seeds, not both.")
    batch = run_experiments(
        experiment_ids=args.experiment_ids,
        run_all=args.all,
        matrix_path=args.matrix,
        baseline_flat_config_path=args.baseline_flat_config,
        baseline_hierarchical_config_path=args.baseline_hierarchical_config,
        proposed_config_path=args.proposed_config,
        smoke_test=args.smoke_test,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        seeds=args.seeds,
        max_train_examples_per_client=args.max_train_examples_per_client,
        max_eval_examples_per_client=args.max_eval_examples_per_client,
        output_root=args.output_root,
    )
    for artifact in batch.experiments:
        seed_suffix = f" seed={artifact.seed}" if artifact.seed is not None else ""
        print(f"{artifact.experiment_id}{seed_suffix}: wrote {artifact.metrics_csv_path}")
        print(f"{artifact.experiment_id}{seed_suffix}: wrote {artifact.ledger_path}")
        print(f"{artifact.experiment_id}{seed_suffix}: wrote {artifact.convergence_plot_path}")
    if batch.summary_csv_path is not None:
        print(f"summary: wrote {batch.summary_csv_path}")
    for path in batch.comparison_plot_paths:
        print(f"comparison_plot: wrote {path}")
    if batch.mean_std_summary_csv_path is not None:
        print(f"mean_std_summary: wrote {batch.mean_std_summary_csv_path}")
    if batch.mean_std_results_csv_path is not None:
        print(f"mean_std_table: wrote {batch.mean_std_results_csv_path}")
    if batch.mean_std_markdown_path is not None:
        print(f"mean_std_markdown: wrote {batch.mean_std_markdown_path}")
    for failed_seed_run in batch.failed_seed_runs:
        print(
            f"{failed_seed_run.experiment_id} seed={failed_seed_run.seed}: FAILED "
            f"({failed_seed_run.error})"
        )


if __name__ == "__main__":
    main()
