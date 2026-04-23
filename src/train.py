from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import yaml
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.ledger.metadata_schema import LedgerRecord, canonical_sha256, model_version_for_round
from src.ledger.mock_ledger import JSONLMockLedger, MAIN_CLUSTER_HEAD_ROLE
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
PLOT_FILE_EXTENSION = ".svg"


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


@dataclass(frozen=True)
class ExperimentBatchResult:
    experiments: list[ExperimentRunArtifacts]
    summary_csv_path: Path
    comparison_plot_paths: list[Path]


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
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _selected_experiment_ids(experiment_ids: Sequence[str] | None, *, run_all: bool) -> list[str]:
    if experiment_ids:
        selected = _dedupe_preserve_order(experiment_ids)
    else:
        selected = list(SUPPORTED_EXPERIMENT_IDS) if run_all or experiment_ids is None else []
    if not selected:
        raise ValueError("No experiment ids selected.")
    unsupported = [exp_id for exp_id in selected if exp_id not in SUPPORTED_EXPERIMENT_IDS]
    if unsupported:
        raise ValueError(
            "This experiment runner currently supports only the required A/B/P experiments. "
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
) -> Path:
    ledger_path = output_root / "ledgers" / f"{experiment_id}_ledger.jsonl"
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
) -> Path:
    model_dir = output_root / "models" / experiment_id
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "model_manifest.json"
    model_hash = canonical_sha256(
        {
            "experiment_id": experiment_id,
            "cluster_id": spec.cluster_id,
            "model_family": summary.get("model_family", spec.model),
            "best_validation_round": summary.get("best_validation_round"),
            "best_round_validation_metrics": summary.get("best_round_validation_metrics"),
            "best_round_test_metrics": summary.get("best_round_test_metrics"),
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


def _write_convergence_plot(experiment_id: str, round_rows: Sequence[Mapping[str, Any]], output_root: Path) -> Path:
    plot_path = output_root / "plots" / f"convergence_{experiment_id}{PLOT_FILE_EXTENSION}"
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


def _finalize_run_outputs(
    *,
    experiment_id: str,
    spec: ExperimentSpec,
    raw_result: Any,
    output_root: Path,
) -> ExperimentRunArtifacts:
    output_dir = Path(raw_result.output_dir).resolve()
    summary_path = Path(raw_result.summary_path).resolve()
    round_metrics_path = Path(raw_result.round_metrics_path).resolve()
    metrics_csv_path = Path(raw_result.metrics_csv_path).resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"{experiment_id}: missing run output directory {output_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"{experiment_id}: missing summary file {summary_path}")
    if not round_metrics_path.exists():
        raise FileNotFoundError(f"{experiment_id}: missing round metrics file {round_metrics_path}")
    if not metrics_csv_path.exists():
        raise FileNotFoundError(f"{experiment_id}: missing metrics CSV file {metrics_csv_path}")

    summary = dict(raw_result.summary)
    summary["summary_path"] = str(summary_path)
    summary["round_metrics_path"] = str(round_metrics_path)
    summary["metrics_csv_path"] = str(metrics_csv_path)
    round_rows = _load_round_metrics(round_metrics_path)
    model_manifest_path = _write_model_manifest(
        experiment_id=experiment_id,
        spec=spec,
        summary=summary,
        output_root=output_root,
    )
    ledger_path = _write_ledger_for_experiment(
        experiment_id=experiment_id,
        spec=spec,
        summary=summary,
        round_rows=round_rows,
        output_root=output_root,
    )
    convergence_plot_path = _write_convergence_plot(experiment_id, round_rows, output_root)

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
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
    output_root: str | Path = "outputs",
) -> ExperimentBatchResult:
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
        max_train_examples_per_client=args.max_train_examples_per_client,
        max_eval_examples_per_client=args.max_eval_examples_per_client,
        output_root=args.output_root,
    )
    for artifact in batch.experiments:
        print(f"{artifact.experiment_id}: wrote {artifact.metrics_csv_path}")
        print(f"{artifact.experiment_id}: wrote {artifact.ledger_path}")
        print(f"{artifact.experiment_id}: wrote {artifact.convergence_plot_path}")
    print(f"summary: wrote {batch.summary_csv_path}")
    for path in batch.comparison_plot_paths:
        print(f"comparison_plot: wrote {path}")


if __name__ == "__main__":
    main()
