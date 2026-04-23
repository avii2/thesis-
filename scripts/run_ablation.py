from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train import (  # noqa: E402
    _dispatch_experiment,
    _ensure_output_directories,
    _finalize_run_outputs,
    _load_metrics_row,
    _load_yaml,
    _write_comparison_plot,
    load_config_registry,
    load_experiment_matrix,
    validate_experiment_registry,
)
from src.train_ablation import (  # noqa: E402
    run_cluster1_fedavg_tcn_ablation,
    run_cluster2_fedavg_mlp_ablation,
    run_cluster3_fedavg_cnn1d_ablation,
)


DEFAULT_ABLATION_CONFIGS = (
    "configs/ablation_hierarchy.yaml",
    "configs/ablation_cluster1_fedbn.yaml",
    "configs/ablation_cluster2_fedprox.yaml",
    "configs/ablation_cluster3_scaffold.yaml",
)


@dataclass(frozen=True)
class AblationComparisonArtifact:
    ablation_id: str
    comparison_id: str
    metrics_csv_path: Path
    plot_path: Path
    row: Mapping[str, Any]


def _resolve_selected_configs(
    config_paths: Sequence[str] | None,
    *,
    run_all: bool,
) -> list[Path]:
    if config_paths:
        selected = [Path(path).resolve() for path in config_paths]
    elif run_all or config_paths is None:
        selected = [(REPO_ROOT / path).resolve() for path in DEFAULT_ABLATION_CONFIGS]
    else:
        selected = []
    if not selected:
        raise ValueError("No ablation config files selected.")
    for path in selected:
        if not path.exists():
            raise FileNotFoundError(f"Ablation config file does not exist: {path}")
    return selected


def _active_run_settings(
    config: Mapping[str, Any],
    *,
    smoke_test: bool,
    rounds: int | None,
    local_epochs: int | None,
    batch_size: int | None,
    seed: int | None,
) -> tuple[int, int, int, int]:
    defaults_key = "smoke_test_defaults" if smoke_test else "training_defaults"
    defaults = config.get(defaults_key)
    if not isinstance(defaults, Mapping):
        raise ValueError(f"Ablation config is missing {defaults_key}.")
    configured_rounds = int(rounds if rounds is not None else defaults["rounds"])
    configured_local_epochs = int(local_epochs if local_epochs is not None else defaults["local_epochs"])
    configured_batch_size = int(batch_size if batch_size is not None else defaults["batch_size"])
    if seed is not None:
        configured_seed = int(seed)
    elif smoke_test:
        configured_seed = int(defaults["seed"])
    else:
        seeds = defaults.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("training_defaults.seeds must contain at least one seed.")
        configured_seed = int(seeds[0])
    return configured_rounds, configured_local_epochs, configured_batch_size, configured_seed


def _write_ablation_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"{path}: ablation rows must not be empty.")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _comparison_summary_row(
    *,
    ablation_id: str,
    comparison: Mapping[str, Any],
    control_row: Mapping[str, Any],
    treatment_row: Mapping[str, Any],
) -> dict[str, Any]:
    control_report_id = str(comparison["control"].get("report_as_experiment_id") or comparison["control"]["experiment_id"])
    treatment_report_id = str(comparison["treatment"].get("report_as_experiment_id") or comparison["treatment"]["experiment_id"])

    def metric_delta(key: str) -> float | None:
        control_value = control_row.get(key)
        treatment_value = treatment_row.get(key)
        if isinstance(control_value, (int, float)) and isinstance(treatment_value, (int, float)):
            return float(treatment_value) - float(control_value)
        return None

    return {
        "ablation_id": ablation_id,
        "comparison_id": str(comparison["comparison_id"]),
        "title": str(comparison["title"]),
        "control_experiment_id": control_report_id,
        "control_source_experiment_id": str(comparison["control"]["experiment_id"]),
        "treatment_experiment_id": treatment_report_id,
        "treatment_source_experiment_id": str(comparison["treatment"]["experiment_id"]),
        "cluster_id": treatment_row.get("cluster_id", control_row.get("cluster_id")),
        "dataset": treatment_row.get("dataset", control_row.get("dataset")),
        "control_best_validation_f1": control_row.get("best_validation_f1"),
        "treatment_best_validation_f1": treatment_row.get("best_validation_f1"),
        "delta_best_validation_f1": metric_delta("best_validation_f1"),
        "control_test_accuracy": control_row.get("test_accuracy"),
        "treatment_test_accuracy": treatment_row.get("test_accuracy"),
        "delta_test_accuracy": metric_delta("test_accuracy"),
        "control_test_f1": control_row.get("test_f1"),
        "treatment_test_f1": treatment_row.get("test_f1"),
        "delta_test_f1": metric_delta("test_f1"),
        "control_test_auroc": control_row.get("test_auroc"),
        "treatment_test_auroc": treatment_row.get("test_auroc"),
        "control_total_communication_cost_bytes": control_row.get("total_communication_cost_bytes"),
        "treatment_total_communication_cost_bytes": treatment_row.get("total_communication_cost_bytes"),
        "delta_total_communication_cost_bytes": metric_delta("total_communication_cost_bytes"),
        "control_metrics_csv_path": control_row.get("metrics_csv_path"),
        "treatment_metrics_csv_path": treatment_row.get("metrics_csv_path"),
    }


def run_ablation_configs(
    *,
    config_paths: Sequence[str] | None = None,
    run_all: bool = False,
    smoke_test: bool = False,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
    output_root: str | Path = "outputs",
) -> list[AblationComparisonArtifact]:
    selected_configs = _resolve_selected_configs(config_paths, run_all=run_all)
    output_root = Path(output_root).resolve()
    _ensure_output_directories(output_root)

    matrix = load_experiment_matrix(REPO_ROOT / "docs" / "EXPERIMENT_MATRIX.csv")
    registry = load_config_registry(
        baseline_flat_config_path=REPO_ROOT / "configs" / "baseline_flat.yaml",
        baseline_hierarchical_config_path=REPO_ROOT / "configs" / "baseline_hierarchical.yaml",
        proposed_config_path=REPO_ROOT / "configs" / "proposed.yaml",
    )
    validate_experiment_registry(matrix, registry)

    loaded_configs = [(path, _load_yaml(path)) for path in selected_configs]
    run_settings = {
        _active_run_settings(
            config,
            smoke_test=smoke_test,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
        )
        for _, config in loaded_configs
    }
    if len(run_settings) != 1:
        raise ValueError(
            "Selected ablation configs resolve to inconsistent run settings. "
            "Pass explicit CLI overrides to unify rounds/local_epochs/batch_size/seed."
        )
    active_rounds, active_local_epochs, active_batch_size, active_seed = next(iter(run_settings))

    standard_experiment_ids: list[str] = []
    need_custom_cluster1 = False
    need_custom_cluster2 = False
    need_custom_cluster3 = False
    for _, config in loaded_configs:
        comparisons = config.get("comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            raise ValueError("Each ablation config must contain a non-empty comparisons list.")
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                raise ValueError("Each ablation comparison must be a mapping.")
            for arm_name in ("control", "treatment"):
                arm = comparison.get(arm_name)
                if not isinstance(arm, Mapping):
                    raise ValueError(f"Ablation comparison {comparison.get('comparison_id')!r} is missing {arm_name}.")
                run_source = str(arm.get("run_source"))
                experiment_id = str(arm["experiment_id"])
                if run_source == "standard":
                    standard_experiment_ids.append(experiment_id)
                elif run_source == "custom_cluster1_fedavg_tcn":
                    need_custom_cluster1 = True
                elif run_source == "custom_cluster2_fedavg_mlp":
                    need_custom_cluster2 = True
                elif run_source == "custom_cluster3_fedavg_cnn1d":
                    need_custom_cluster3 = True
                else:
                    raise ValueError(f"Unsupported ablation run_source {run_source!r} for {experiment_id}.")

    artifact_by_experiment_id: dict[str, Any] = {}
    for experiment_id in dict.fromkeys(standard_experiment_ids):
        raw_result = _dispatch_experiment(
            experiment_id,
            registry=registry,
            smoke_test=smoke_test,
            rounds=active_rounds,
            local_epochs=active_local_epochs,
            batch_size=active_batch_size,
            seed=active_seed,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
        artifact_by_experiment_id[experiment_id] = _finalize_run_outputs(
            experiment_id=experiment_id,
            spec=matrix[experiment_id],
            raw_result=raw_result,
            output_root=output_root,
        )

    if need_custom_cluster1:
        raw_result = run_cluster1_fedavg_tcn_ablation(
            ablation_config_path=REPO_ROOT / "configs" / "ablation_cluster1_fedbn.yaml",
            rounds=active_rounds,
            local_epochs=active_local_epochs,
            batch_size=active_batch_size,
            seed=active_seed,
            smoke_test=smoke_test,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
        artifact_by_experiment_id["AB_C1_FEDAVG_TCN"] = _finalize_run_outputs(
            experiment_id="AB_C1_FEDAVG_TCN",
            spec=matrix["AB_C1_FEDAVG_TCN"],
            raw_result=raw_result,
            output_root=output_root,
        )

    if need_custom_cluster2:
        raw_result = run_cluster2_fedavg_mlp_ablation(
            ablation_config_path=REPO_ROOT / "configs" / "ablation_cluster2_fedprox.yaml",
            rounds=active_rounds,
            local_epochs=active_local_epochs,
            batch_size=active_batch_size,
            seed=active_seed,
            smoke_test=smoke_test,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
        artifact_by_experiment_id["AB_C2_FEDAVG_MLP"] = _finalize_run_outputs(
            experiment_id="AB_C2_FEDAVG_MLP",
            spec=matrix["AB_C2_FEDAVG_MLP"],
            raw_result=raw_result,
            output_root=output_root,
        )

    if need_custom_cluster3:
        raw_result = run_cluster3_fedavg_cnn1d_ablation(
            ablation_config_path=REPO_ROOT / "configs" / "ablation_cluster3_scaffold.yaml",
            rounds=active_rounds,
            local_epochs=active_local_epochs,
            batch_size=active_batch_size,
            seed=active_seed,
            smoke_test=smoke_test,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
            output_root=output_root,
        )
        artifact_by_experiment_id["AB_C3_FEDAVG_CNN1D"] = _finalize_run_outputs(
            experiment_id="AB_C3_FEDAVG_CNN1D",
            spec=matrix["AB_C3_FEDAVG_CNN1D"],
            raw_result=raw_result,
            output_root=output_root,
        )

    comparison_artifacts: list[AblationComparisonArtifact] = []
    for config_path, config in loaded_configs:
        ablation_id = str(config["ablation_id"])
        comparisons = config["comparisons"]
        ablation_rows: list[dict[str, Any]] = []
        for comparison in comparisons:
            control_experiment_id = str(comparison["control"]["experiment_id"])
            treatment_experiment_id = str(comparison["treatment"]["experiment_id"])
            if control_experiment_id not in artifact_by_experiment_id:
                raise ValueError(f"{config_path}: missing executed control experiment {control_experiment_id}.")
            if treatment_experiment_id not in artifact_by_experiment_id:
                raise ValueError(f"{config_path}: missing executed treatment experiment {treatment_experiment_id}.")

            control_artifact = artifact_by_experiment_id[control_experiment_id]
            treatment_artifact = artifact_by_experiment_id[treatment_experiment_id]
            control_row = _load_metrics_row(control_artifact.metrics_csv_path)
            treatment_row = _load_metrics_row(treatment_artifact.metrics_csv_path)
            control_row["experiment_id"] = str(
                comparison["control"].get("report_as_experiment_id", control_experiment_id)
            )
            control_row["metrics_csv_path"] = str(control_artifact.metrics_csv_path)
            treatment_row["experiment_id"] = str(
                comparison["treatment"].get("report_as_experiment_id", treatment_experiment_id)
            )
            treatment_row["metrics_csv_path"] = str(treatment_artifact.metrics_csv_path)

            comparison_row = _comparison_summary_row(
                ablation_id=ablation_id,
                comparison=comparison,
                control_row=control_row,
                treatment_row=treatment_row,
            )
            plot_path = _write_comparison_plot(
                plot_name=str(comparison["comparison_id"]),
                title=str(comparison["title"]),
                metric_key="test_f1",
                summary_rows=[control_row, treatment_row],
                output_root=output_root,
            )
            comparison_row["plot_path"] = str(plot_path)
            ablation_rows.append(comparison_row)
            comparison_artifacts.append(
                AblationComparisonArtifact(
                    ablation_id=ablation_id,
                    comparison_id=str(comparison["comparison_id"]),
                    metrics_csv_path=output_root / "metrics" / f"{ablation_id}.csv",
                    plot_path=plot_path,
                    row=comparison_row,
                )
            )

        _write_ablation_csv(output_root / "metrics" / f"{ablation_id}.csv", ablation_rows)

    return comparison_artifacts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the required FCFL ablation suites.")
    parser.add_argument(
        "--config",
        action="append",
        dest="config_paths",
        help="Path to an ablation config YAML file. Repeatable.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all required ablation configs.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke_test_defaults from the ablation configs.")
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
    results = run_ablation_configs(
        config_paths=args.config_paths,
        run_all=args.all,
        smoke_test=args.smoke_test,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        max_train_examples_per_client=args.max_train_examples_per_client,
        max_eval_examples_per_client=args.max_eval_examples_per_client,
        output_root=args.output_root,
    )
    for result in results:
        print(f"{result.comparison_id}: wrote {result.metrics_csv_path}")
        print(f"{result.comparison_id}: wrote {result.plot_path}")


if __name__ == "__main__":
    main()
