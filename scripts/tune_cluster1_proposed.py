from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.train_cluster1_proposed import run_cluster1_proposed  # noqa: E402


EXPECTED_SEARCH_SPACE = {
    "learning_rate": [0.001, 0.003, 0.005, 0.01],
    "batch_size": [64, 128],
    "local_epochs": [1, 2],
    "window_length": [32, 64],
    "stride": [4, 8],
    "block_channels": [[32, 64, 64], [64, 64, 64], [32, 64, 128]],
    "hidden_dim": [32, 64],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "positive_class_weight_scale": [0.25, 0.5, 0.75, 1.0],
}

RESULT_COLUMNS = [
    "trial_id",
    "status",
    "error",
    "search_mode",
    "fair_comparison_to_current",
    "learning_rate",
    "batch_size",
    "local_epochs",
    "window_length",
    "stride",
    "block_channels",
    "hidden_dim",
    "dropout",
    "positive_class_weight_scale",
    "rounds",
    "seed",
    "max_train_examples_per_client",
    "max_eval_examples_per_client",
    "best_validation_round",
    "best_validation_f1",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_auroc",
    "test_pr_auc",
    "test_fpr",
    "threshold_used",
    "positive_class_weight",
    "computed_positive_class_weight",
    "communication_cost_per_round_bytes",
    "total_communication_cost_bytes",
    "wall_clock_training_seconds",
    "trial_elapsed_seconds",
    "beats_current_p_c1",
    "beats_current_a_c1",
    "beats_current_b_c1",
    "trial_output_root",
    "trial_proposed_config",
    "trial_cluster_config",
    "metrics_csv_path",
    "summary_path",
    "round_metrics_path",
]


@dataclass(frozen=True)
class Trial:
    trial_id: str
    index: int
    learning_rate: float
    batch_size: int
    local_epochs: int
    window_length: int
    stride: int
    block_channels: tuple[int, int, int]
    hidden_dim: int
    dropout: float
    positive_class_weight_scale: float

    def as_result_fields(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "window_length": self.window_length,
            "stride": self.stride,
            "block_channels": json.dumps(list(self.block_channels)),
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "positive_class_weight_scale": self.positive_class_weight_scale,
        }


def _read_yaml(path: Path) -> Mapping[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a YAML mapping.")
    return data


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _normalize_search_space(search_space: Mapping[str, Any]) -> dict[str, list[Any]]:
    normalized: dict[str, list[Any]] = {}
    for key, expected_values in EXPECTED_SEARCH_SPACE.items():
        observed = search_space.get(key)
        if observed != expected_values:
            raise ValueError(
                f"configs/tuning_cluster1.yaml search_space.{key} must exactly equal {expected_values!r}; "
                f"observed {observed!r}."
            )
        normalized[key] = list(observed)
    extra_keys = sorted(set(search_space) - set(EXPECTED_SEARCH_SPACE))
    if extra_keys:
        raise ValueError(f"Unexpected Cluster 1 tuning search-space keys: {extra_keys}.")
    return normalized


def _priority_order(values: Sequence[Any], preferred: Sequence[Any]) -> list[Any]:
    preferred_index = {json.dumps(value): index for index, value in enumerate(preferred)}
    return sorted(
        list(values),
        key=lambda value: preferred_index.get(json.dumps(value), len(preferred_index)),
    )


def build_trials(search_space: Mapping[str, Any], *, heuristic_order: bool = True) -> list[Trial]:
    space = _normalize_search_space(search_space)
    if heuristic_order:
        space = {
            **space,
            "learning_rate": _priority_order(space["learning_rate"], [0.003, 0.005, 0.001, 0.01]),
            "batch_size": _priority_order(space["batch_size"], [128, 64]),
            "local_epochs": _priority_order(space["local_epochs"], [1, 2]),
            "window_length": _priority_order(space["window_length"], [32, 64]),
            "stride": _priority_order(space["stride"], [8, 4]),
            "block_channels": _priority_order(
                space["block_channels"],
                [[32, 64, 64], [64, 64, 64], [32, 64, 128]],
            ),
            "hidden_dim": _priority_order(space["hidden_dim"], [32, 64]),
            "dropout": _priority_order(space["dropout"], [0.1, 0.2, 0.0, 0.3]),
            "positive_class_weight_scale": _priority_order(
                space["positive_class_weight_scale"],
                [0.75, 0.5, 1.0, 0.25],
            ),
        }

    trials: list[Trial] = []
    for index, values in enumerate(
        itertools.product(
            space["learning_rate"],
            space["batch_size"],
            space["local_epochs"],
            space["window_length"],
            space["stride"],
            space["block_channels"],
            space["hidden_dim"],
            space["dropout"],
            space["positive_class_weight_scale"],
        ),
        start=1,
    ):
        (
            learning_rate,
            batch_size,
            local_epochs,
            window_length,
            stride,
            block_channels,
            hidden_dim,
            dropout,
            positive_class_weight_scale,
        ) = values
        trials.append(
            Trial(
                trial_id=f"trial_{index:04d}",
                index=index,
                learning_rate=float(learning_rate),
                batch_size=int(batch_size),
                local_epochs=int(local_epochs),
                window_length=int(window_length),
                stride=int(stride),
                block_channels=tuple(int(channel) for channel in block_channels),
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                positive_class_weight_scale=float(positive_class_weight_scale),
            )
        )
    return trials


def _read_single_row_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one metrics row, observed {len(rows)}.")
    return rows[0]


def _float_or_nan(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _comparison_metrics(metrics_root: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for experiment_id in ("A_C1", "B_C1", "P_C1"):
        row = _read_single_row_csv(metrics_root / f"{experiment_id}_metrics.csv")
        values[experiment_id] = _float_or_nan(row.get("test_f1")) if row else math.nan
    return values


def _is_complete_full_budget(
    *,
    rounds: int,
    max_train_examples_per_client: int | None,
    max_eval_examples_per_client: int | None,
) -> bool:
    return (
        int(rounds) == 50
        and max_train_examples_per_client is None
        and max_eval_examples_per_client is None
    )


def _write_trial_configs(
    *,
    tuning_root: Path,
    trial: Trial,
    base_cluster_config_path: Path,
    base_proposed_config_path: Path,
    membership_file: Path,
) -> tuple[Path, Path]:
    cluster_config = dict(_read_yaml(base_cluster_config_path))
    preprocessing = dict(cluster_config.get("preprocessing", {}))
    preprocessing["window_length"] = trial.window_length
    preprocessing["stride"] = trial.stride
    cluster_config["preprocessing"] = preprocessing

    trial_config_dir = tuning_root / "trial_configs" / trial.trial_id
    trial_cluster_config_path = trial_config_dir / "cluster1_hai.yaml"
    _write_yaml(trial_cluster_config_path, cluster_config)

    proposed_config = dict(_read_yaml(base_proposed_config_path))
    clusters = proposed_config.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError(f"{base_proposed_config_path}: expected clusters list.")
    trial_clusters: list[dict[str, Any]] = []
    for entry in clusters:
        if not isinstance(entry, Mapping) or str(entry.get("experiment_id")) != "P_C1":
            continue
        trial_entry = dict(entry)
        trial_entry["cluster_config"] = str(trial_cluster_config_path)
        trial_entry["membership_file"] = str(membership_file)
        trial_entry["model_hyperparameters"] = {
            "block_channels": list(trial.block_channels),
            "hidden_dim": trial.hidden_dim,
            "dropout": trial.dropout,
        }
        trial_entry["training_hyperparameters"] = {
            "positive_class_weight_scale": trial.positive_class_weight_scale,
        }
        trial_clusters.append(trial_entry)

    if len(trial_clusters) != 1:
        raise ValueError(f"{base_proposed_config_path}: expected exactly one P_C1 cluster entry.")
    proposed_config["clusters"] = trial_clusters
    trial_proposed_config_path = trial_config_dir / "proposed.yaml"
    _write_yaml(trial_proposed_config_path, proposed_config)
    return trial_cluster_config_path, trial_proposed_config_path


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        _float_or_nan(row.get("best_validation_f1")),
        _float_or_nan(row.get("test_f1")),
        -_float_or_nan(row.get("test_fpr")),
        -_float_or_nan(row.get("wall_clock_training_seconds")),
    )


def _best_completed_row(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    complete_rows = [row for row in rows if row.get("status") == "COMPLETE"]
    if not complete_rows:
        return None
    return max(complete_rows, key=_selection_key)


def _write_results_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def _write_best_config(
    *,
    path: Path,
    best_row: Mapping[str, Any] | None,
    comparison: Mapping[str, float],
    total_trials_in_space: int,
    completed_trials: int,
    search_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if best_row is None:
        payload = {
            "status": "NO_COMPLETED_TRIALS",
            "search_mode": search_mode,
            "total_trials_in_space": total_trials_in_space,
            "completed_trials": completed_trials,
            "current_reference_test_f1": comparison,
        }
    else:
        payload = {
            "status": "COMPLETE",
            "search_mode": search_mode,
            "total_trials_in_space": total_trials_in_space,
            "completed_trials": completed_trials,
            "trial_id": best_row["trial_id"],
            "selection_rule": [
                "highest validation F1",
                "highest test F1",
                "lower test FPR",
                "lower wall-clock time",
            ],
            "config": {
                "learning_rate": _float_or_nan(best_row["learning_rate"]),
                "batch_size": int(best_row["batch_size"]),
                "local_epochs": int(best_row["local_epochs"]),
                "window_length": int(best_row["window_length"]),
                "stride": int(best_row["stride"]),
                "block_channels": json.loads(str(best_row["block_channels"])),
                "hidden_dim": int(best_row["hidden_dim"]),
                "dropout": _float_or_nan(best_row["dropout"]),
                "positive_class_weight_scale": _float_or_nan(best_row["positive_class_weight_scale"]),
            },
            "metrics": {
                "best_validation_f1": _float_or_nan(best_row["best_validation_f1"]),
                "test_f1": _float_or_nan(best_row["test_f1"]),
                "test_fpr": _float_or_nan(best_row["test_fpr"]),
                "wall_clock_training_seconds": _float_or_nan(best_row["wall_clock_training_seconds"]),
            },
            "fair_comparison_to_current": str(best_row.get("fair_comparison_to_current")) == "True",
            "beats_current_p_c1": best_row.get("beats_current_p_c1"),
            "beats_current_a_c1": best_row.get("beats_current_a_c1"),
            "beats_current_b_c1": best_row.get("beats_current_b_c1"),
            "current_reference_test_f1": comparison,
            "source_metrics_csv": best_row.get("metrics_csv_path"),
        }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_summary_markdown(
    *,
    path: Path,
    best_row: Mapping[str, Any] | None,
    comparison: Mapping[str, float],
    total_trials_in_space: int,
    attempted_trials: int,
    completed_trials: int,
    search_mode: str,
    fair_comparison_to_current: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cluster 1 P_C1 Tuning Summary",
        "",
        "This tuning output is restricted to Cluster 1 proposed FCFL only. The run keeps the architecture fixed as TCN + FedBN + weighted non-BN aggregation and reuses the frozen agglomerative membership file.",
        "",
        "## Search Status",
        "",
        f"- Search mode: `{search_mode}`",
        f"- Total configurations in exact search space: `{total_trials_in_space}`",
        f"- Trials attempted in this invocation: `{attempted_trials}`",
        f"- Completed trials recorded: `{completed_trials}`",
        f"- Fair full-budget comparison to current results: `{'YES' if fair_comparison_to_current else 'NO'}`",
        "",
        "## Current References",
        "",
        f"- A_C1 test F1: `{comparison.get('A_C1', math.nan):.4f}`",
        f"- B_C1 test F1: `{comparison.get('B_C1', math.nan):.4f}`",
        f"- P_C1 test F1: `{comparison.get('P_C1', math.nan):.4f}`",
        "",
        "## Best Trial",
        "",
    ]
    if best_row is None:
        lines.append("No completed tuning trial is available yet.")
    else:
        lines.extend(
            [
                f"- Trial: `{best_row['trial_id']}`",
                f"- Best validation F1: `{_float_or_nan(best_row['best_validation_f1']):.4f}`",
                f"- Test F1: `{_float_or_nan(best_row['test_f1']):.4f}`",
                f"- Test FPR: `{_float_or_nan(best_row['test_fpr']):.4f}`",
                f"- Wall-clock seconds: `{_float_or_nan(best_row['wall_clock_training_seconds']):.3f}`",
                f"- Beats current P_C1: `{best_row.get('beats_current_p_c1')}`",
                f"- Beats A_C1: `{best_row.get('beats_current_a_c1')}`",
                f"- Beats B_C1: `{best_row.get('beats_current_b_c1')}`",
                "",
                "## Best Config",
                "",
                f"- learning_rate: `{best_row['learning_rate']}`",
                f"- batch_size: `{best_row['batch_size']}`",
                f"- local_epochs: `{best_row['local_epochs']}`",
                f"- window_length: `{best_row['window_length']}`",
                f"- stride: `{best_row['stride']}`",
                f"- block_channels: `{best_row['block_channels']}`",
                f"- hidden_dim: `{best_row['hidden_dim']}`",
                f"- dropout: `{best_row['dropout']}`",
                f"- positive_class_weight_scale: `{best_row['positive_class_weight_scale']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The primary selection rule is validation F1; test F1, test FPR, and wall-clock time are tie-breakers only.",
            "- Main `outputs/runs/P_C1/` and `outputs/metrics/P_C1_metrics.csv` are not overwritten by this script.",
        ]
    )
    if not fair_comparison_to_current:
        lines.append("- This invocation used a smoke or partial budget, so the beat/not-beat flags are diagnostic and should not be treated as paper-level evidence.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_existing_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_tuning(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    config = _read_yaml(config_path)
    search_space = config.get("search_space")
    if not isinstance(search_space, Mapping):
        raise ValueError(f"{config_path}: missing search_space mapping.")
    trials = build_trials(search_space, heuristic_order=not args.no_heuristic_order)

    fixed = config.get("fixed_experiment")
    if not isinstance(fixed, Mapping):
        raise ValueError(f"{config_path}: missing fixed_experiment mapping.")
    if fixed.get("experiment_id") != "P_C1" or fixed.get("model_family") != "tcn":
        raise ValueError("Cluster 1 tuning is restricted to P_C1 with model_family=tcn.")
    if fixed.get("fl_method") != "FedBN" or fixed.get("aggregation") != "weighted_non_bn_mean":
        raise ValueError("Cluster 1 tuning must keep FedBN and weighted_non_bn_mean.")

    output = config.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(f"{config_path}: missing output mapping.")
    tuning_root = (REPO_ROOT / str(output["root"])).resolve()
    results_csv_path = tuning_root / "cluster1_tuning_results.csv"
    best_config_path = tuning_root / "best_config.json"
    summary_path = tuning_root / "cluster1_tuning_summary.md"

    execution_defaults = config.get("execution_defaults")
    if not isinstance(execution_defaults, Mapping):
        raise ValueError(f"{config_path}: missing execution_defaults mapping.")
    rounds = int(args.rounds if args.rounds is not None else (
        execution_defaults["smoke_rounds"] if args.smoke_test else execution_defaults["rounds"]
    ))
    seed = int(args.seed if args.seed is not None else execution_defaults["seed"])
    max_train = (
        args.max_train_examples_per_client
        if args.max_train_examples_per_client is not None
        else execution_defaults.get("max_train_examples_per_client")
    )
    max_eval = (
        args.max_eval_examples_per_client
        if args.max_eval_examples_per_client is not None
        else execution_defaults.get("max_eval_examples_per_client")
    )
    max_train = int(max_train) if max_train is not None else None
    max_eval = int(max_eval) if max_eval is not None else None

    fair_comparison = _is_complete_full_budget(
        rounds=rounds,
        max_train_examples_per_client=max_train,
        max_eval_examples_per_client=max_eval,
    )
    search_mode = "smoke" if args.smoke_test or not fair_comparison else "full"
    comparison = _comparison_metrics(REPO_ROOT / "outputs" / "metrics")

    selected_trials = trials[int(args.start_index) :]
    if args.max_trials is not None:
        selected_trials = selected_trials[: int(args.max_trials)]

    existing_rows = [] if args.force else _load_existing_results(results_csv_path)
    existing_by_trial = {row["trial_id"]: row for row in existing_rows}
    rows_by_trial: dict[str, dict[str, Any]] = {row["trial_id"]: dict(row) for row in existing_rows}

    if args.dry_run:
        tuning_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "DRY_RUN",
            "total_trials_in_space": len(trials),
            "selected_trials": [trial.as_result_fields() for trial in selected_trials],
        }
        best_config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_summary_markdown(
            path=summary_path,
            best_row=None,
            comparison=comparison,
            total_trials_in_space=len(trials),
            attempted_trials=0,
            completed_trials=0,
            search_mode="dry_run",
            fair_comparison_to_current=False,
        )
        return payload

    base_cluster_config_path = (REPO_ROOT / str(fixed["base_cluster_config"])).resolve()
    base_proposed_config_path = (REPO_ROOT / str(fixed["base_proposed_config"])).resolve()
    membership_file = (REPO_ROOT / str(fixed["membership_file"])).resolve()
    if not membership_file.exists():
        raise FileNotFoundError(f"Frozen Cluster 1 membership file is missing: {membership_file}")

    attempted = 0
    for trial in selected_trials:
        if trial.trial_id in existing_by_trial and existing_by_trial[trial.trial_id].get("status") == "COMPLETE":
            continue
        attempted += 1
        trial_cluster_config_path, trial_proposed_config_path = _write_trial_configs(
            tuning_root=tuning_root,
            trial=trial,
            base_cluster_config_path=base_cluster_config_path,
            base_proposed_config_path=base_proposed_config_path,
            membership_file=membership_file,
        )
        trial_output_root = tuning_root / "trials" / trial.trial_id / "outputs"
        start_time = time.perf_counter()
        row: dict[str, Any] = {
            **trial.as_result_fields(),
            "status": "FAILED",
            "error": "",
            "search_mode": search_mode,
            "fair_comparison_to_current": fair_comparison,
            "rounds": rounds,
            "seed": seed,
            "max_train_examples_per_client": max_train if max_train is not None else "",
            "max_eval_examples_per_client": max_eval if max_eval is not None else "",
            "trial_output_root": str(trial_output_root),
            "trial_proposed_config": str(trial_proposed_config_path),
            "trial_cluster_config": str(trial_cluster_config_path),
        }
        try:
            result = run_cluster1_proposed(
                proposed_config_path=trial_proposed_config_path,
                rounds=rounds,
                local_epochs=trial.local_epochs,
                batch_size=trial.batch_size,
                learning_rate=trial.learning_rate,
                seed=seed,
                smoke_test=args.smoke_test,
                max_train_examples_per_client=max_train,
                max_eval_examples_per_client=max_eval,
                output_root=trial_output_root,
                tcn_block_channels=trial.block_channels,
                tcn_hidden_dim=trial.hidden_dim,
                tcn_dropout=trial.dropout,
                positive_class_weight_scale=trial.positive_class_weight_scale,
            )
            metrics_row = _read_single_row_csv(result.metrics_csv_path)
            if metrics_row is None:
                raise ValueError(f"{result.metrics_csv_path}: metrics row was not written.")
            row.update(
                {
                    "status": "COMPLETE",
                    "best_validation_round": metrics_row.get("best_validation_round", ""),
                    "best_validation_f1": metrics_row.get("best_validation_f1", ""),
                    "test_accuracy": metrics_row.get("test_accuracy", ""),
                    "test_precision": metrics_row.get("test_precision", ""),
                    "test_recall": metrics_row.get("test_recall", ""),
                    "test_f1": metrics_row.get("test_f1", ""),
                    "test_auroc": metrics_row.get("test_auroc", ""),
                    "test_pr_auc": metrics_row.get("test_pr_auc", ""),
                    "test_fpr": metrics_row.get("test_fpr", ""),
                    "threshold_used": metrics_row.get("threshold_used", ""),
                    "positive_class_weight": metrics_row.get("positive_class_weight", ""),
                    "computed_positive_class_weight": metrics_row.get("computed_positive_class_weight", ""),
                    "communication_cost_per_round_bytes": metrics_row.get("communication_cost_per_round_bytes", ""),
                    "total_communication_cost_bytes": metrics_row.get("total_communication_cost_bytes", ""),
                    "wall_clock_training_seconds": metrics_row.get("wall_clock_training_seconds", ""),
                    "metrics_csv_path": str(result.metrics_csv_path),
                    "summary_path": str(result.summary_path),
                    "round_metrics_path": str(result.round_metrics_path),
                }
            )
            test_f1 = _float_or_nan(metrics_row.get("test_f1"))
            row["beats_current_p_c1"] = bool(test_f1 > comparison["P_C1"]) if not math.isnan(comparison["P_C1"]) else ""
            row["beats_current_a_c1"] = bool(test_f1 > comparison["A_C1"]) if not math.isnan(comparison["A_C1"]) else ""
            row["beats_current_b_c1"] = bool(test_f1 > comparison["B_C1"]) if not math.isnan(comparison["B_C1"]) else ""
        except Exception as exc:  # noqa: BLE001 - failures are persisted as tuning outcomes.
            row["status"] = "FAILED"
            row["error"] = str(exc)
        finally:
            row["trial_elapsed_seconds"] = f"{time.perf_counter() - start_time:.6f}"
            rows_by_trial[trial.trial_id] = row
            ordered_rows = [rows_by_trial[key] for key in sorted(rows_by_trial)]
            _write_results_csv(results_csv_path, ordered_rows)
            best_row = _best_completed_row(ordered_rows)
            completed = sum(1 for result_row in ordered_rows if result_row.get("status") == "COMPLETE")
            _write_best_config(
                path=best_config_path,
                best_row=best_row,
                comparison=comparison,
                total_trials_in_space=len(trials),
                completed_trials=completed,
                search_mode=search_mode,
            )
            _write_summary_markdown(
                path=summary_path,
                best_row=best_row,
                comparison=comparison,
                total_trials_in_space=len(trials),
                attempted_trials=attempted,
                completed_trials=completed,
                search_mode=search_mode,
                fair_comparison_to_current=fair_comparison,
            )

    ordered_rows = [rows_by_trial[key] for key in sorted(rows_by_trial)]
    best_row = _best_completed_row(ordered_rows)
    return {
        "status": "COMPLETE" if best_row is not None else "NO_COMPLETED_TRIALS",
        "search_mode": search_mode,
        "fair_comparison_to_current": fair_comparison,
        "total_trials_in_space": len(trials),
        "attempted_trials": attempted,
        "completed_trials": sum(1 for row in ordered_rows if row.get("status") == "COMPLETE"),
        "best_trial": dict(best_row) if best_row is not None else None,
        "results_csv": str(results_csv_path),
        "best_config": str(best_config_path),
        "summary": str(summary_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Cluster 1 proposed P_C1 hyperparameters only.")
    parser.add_argument("--config", default="configs/tuning_cluster1.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke budget before full tuning.")
    parser.add_argument("--rounds", type=int, help="Override the tuning run round count.")
    parser.add_argument("--seed", type=int, help="Override the tuning seed.")
    parser.add_argument("--max-trials", type=int, help="Limit the number of trials from the exact search space.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based trial offset for chunked tuning.")
    parser.add_argument("--max-train-examples-per-client", type=int)
    parser.add_argument("--max-eval-examples-per-client", type=int)
    parser.add_argument("--force", action="store_true", help="Ignore any existing tuning results CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Write a dry-run summary without training.")
    parser.add_argument("--no-heuristic-order", action="store_true", help="Use raw Cartesian product order.")
    return parser.parse_args(argv)


def main() -> None:
    report = run_tuning(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
