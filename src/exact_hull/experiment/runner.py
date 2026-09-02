"""Configuration loading, job expansion, and isolated GAMS execution."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyomo
from pyomo.common.collections import ComponentMap
from pyomo.common.enums import SolverAPIVersion
from pyomo.core.plugins.transform.logical_to_linear import update_boolean_vars_from_binary
from pyomo.environ import (
    BooleanVar,
    ConcreteModel,
    Constraint,
    ConstraintList,
    Objective,
    SolverFactory,
    TransformationFactory,
    Var,
    value,
)
from pyomo.gdp import Disjunct
from pyomo.gdp.util import clone_without_expression_components
from pyomo.opt import SolverStatus, TerminationCondition

from exact_hull.benchmarks import BENCHMARKS, INSTANCE_PARAMETERS
from exact_hull.benchmarks.base import BenchmarkCase
from exact_hull.experiment.logparse import (
    parse_solver_bounds,
    parse_solver_metadata,
    solver_timed_out,
)
from exact_hull.experiment.results import (
    RUN_MODES,
    RunRecord,
    is_valid_result,
    validate_manifest,
    write_record_atomic,
)
from exact_hull.experiment.solvers import TOLS, options_for
from exact_hull.experiment.structure import structural_counts
from exact_hull.transformations import TRANSFORMATIONS

RANGE_PARAMETERS = {
    "random_quadratic": {"coeff_range", "constraint_margin", "x_range"},
    "kmeans": {"coord_range"},
}
INSTANCE_DEFAULTS = {
    "random_quadratic": {
        "n_dimensions": 3,
        "n_disjunctions": 2,
        "n_disjuncts_per_disjunction": 3,
        "n_constraints_per_disjunct": 2,
        "n_feasible_regions": 2,
        "ensure_positive_definite": False,
        "objective_positive_definite": None,
        "replicate": 1,
        "sparsity_factor": 0.0,
    },
    "kmeans": {"replicate": 1},
    "cstr": {"NT": 5},
    "clay": {"instance": "CLay0203", "metric": "l1"},
}
INT_PARAMETERS = {
    "random_quadratic": {
        "n_dimensions",
        "n_disjunctions",
        "n_disjuncts_per_disjunction",
        "n_constraints_per_disjunct",
        "n_feasible_regions",
        "replicate",
    },
    "kmeans": {"n_dimensions", "n_clusters", "n_points", "replicate"},
    "cstr": {"NT"},
    "clay": set(),
}
FLOAT_PARAMETERS = {"random_quadratic": {"sparsity_factor"}}
BOOL_PARAMETERS = {
    "random_quadratic": {"ensure_positive_definite", "objective_positive_definite"}
}
RANGE_DEFAULTS = {
    "random_quadratic": {
        "coeff_range": [-1.0, 1.0],
        "constraint_margin": [0.0, 0.01],
        "x_range": [-1.0, 1.0],
    },
    "kmeans": {"coord_range": [-1.0, 1.0]},
}


@dataclass(frozen=True)
class Job:
    run_id: str
    benchmark: str
    instance_id: str
    params: dict[str, Any]
    seed: int
    strategy: str
    label: str
    transformation_options: dict[str, Any]
    solver: str
    subsolver: str
    variant: str | None
    mode: str
    time_limit: float


@dataclass(frozen=True)
class _ExecutionTask:
    job: Job
    output_directory: Path
    versions: dict[str, str | None]


def _validate_instance_config(benchmark: str, instances: Any) -> None:
    if not isinstance(instances, dict):
        raise ValueError("instances must be a table")
    fixed = instances.get("fixed", {})
    if not isinstance(fixed, dict):
        raise ValueError("instances.fixed must be a table")
    axis_keys = set(instances) - {"fixed"}
    overlap = axis_keys & set(fixed)
    if overlap:
        raise ValueError(f"Instance keys cannot be both axes and fixed: {sorted(overlap)}")
    for key in sorted(axis_keys | set(fixed)):
        if key not in INSTANCE_PARAMETERS[benchmark]:
            raise ValueError(f"Unknown instance parameter for {benchmark}: {key}")
    for key in sorted(RANGE_PARAMETERS.get(benchmark, set())):
        if key in axis_keys:
            raise ValueError(f"Tuple-valued instance parameter {key} must be in instances.fixed")
        if key in fixed and (
            not isinstance(fixed[key], list)
            or len(fixed[key]) != 2
            or not all(type(value) in {int, float} for value in fixed[key])
            or not all(math.isfinite(value) for value in fixed[key])
            or fixed[key][0] >= fixed[key][1]
        ):
            raise ValueError(f"instances.fixed.{key} must be a finite, increasing two-number list")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalize_option_numbers(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, list):
        return [_normalize_option_numbers(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_option_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_option_numbers(item) for key, item in value.items()}
    return value


def _normalize_strategy_options(name: str, options: dict[str, Any]) -> dict[str, Any]:
    try:
        config = TransformationFactory(name).CONFIG(options)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid options for transformation {name}: {error}") from error
    normalized = {key: _normalize_option_numbers(config[key]) for key in config}
    # mbigm's solver option is a Pyomo solver object; the manifest stores its name.
    if name == "gdp.mbigm" and hasattr(config["solver"], "name"):
        normalized["solver"] = config["solver"].name
    return normalized


def _normalize_integer_parameter(benchmark: str, key: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_integer_parameter(benchmark, key, item) for item in value]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Instance parameter {benchmark}.{key} must contain integers")
    if not math.isfinite(value) or not float(value).is_integer():
        raise ValueError(f"Instance parameter {benchmark}.{key} must contain integers")
    normalized = int(value)
    if key == "replicate" and normalized <= 0:
        raise ValueError(f"Instance parameter {benchmark}.replicate must be positive")
    return normalized


def _normalize_bool_parameter(benchmark: str, key: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_bool_parameter(benchmark, key, item) for item in value]
    if not isinstance(value, bool):
        raise ValueError(f"Instance parameter {benchmark}.{key} must contain booleans")
    return value


def _normalize_instances(benchmark: str, instances: dict[str, Any]) -> dict[str, Any]:
    axes = {key: value for key, value in instances.items() if key != "fixed"}
    supplied_fixed = dict(instances.get("fixed", {}))
    merged = dict(axes)
    merged.update(supplied_fixed)
    defaults = {**INSTANCE_DEFAULTS[benchmark], **RANGE_DEFAULTS.get(benchmark, {})}
    for key, default in defaults.items():
        merged.setdefault(key, default)

    if (
        benchmark == "random_quadratic"
        and merged["objective_positive_definite"] is None
        and not isinstance(merged["ensure_positive_definite"], list)
    ):
        merged["objective_positive_definite"] = merged["ensure_positive_definite"]

    for key, parameter_value in list(merged.items()):
        if key in INT_PARAMETERS[benchmark]:
            merged[key] = _normalize_integer_parameter(benchmark, key, parameter_value)
        elif key in BOOL_PARAMETERS.get(benchmark, set()) and parameter_value is not None:
            merged[key] = _normalize_bool_parameter(benchmark, key, parameter_value)
        elif key in FLOAT_PARAMETERS.get(benchmark, set()) or key in RANGE_PARAMETERS.get(
            benchmark, set()
        ):
            merged[key] = _normalize_option_numbers(parameter_value)

    fixed_keys = set(supplied_fixed) | set(RANGE_DEFAULTS.get(benchmark, {}))
    normalized = {key: value for key, value in merged.items() if key not in fixed_keys}
    if fixed_keys:
        normalized["fixed"] = {key: merged[key] for key in merged if key in fixed_keys}
    return normalized


def _time_limit(experiment: dict[str, Any], name: str, default: float) -> float:
    raw = experiment.get(name, default)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"experiment.{name} must be a positive finite number")
    limit = float(raw)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(f"experiment.{name} must be a positive finite number")
    return limit


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    allowed_top_level = {"experiment", "instances", "strategies", "solvers"}
    unknown_top_level = set(config) - allowed_top_level
    if unknown_top_level:
        key = sorted(unknown_top_level)[0]
        raise ValueError(f"Unknown top-level config key: {key}")
    experiment = config.get("experiment", {})
    allowed_experiment = {
        "benchmark",
        "base_seed",
        "time_limit",
        "modes",
        "root_time_limit",
        "relaxation_time_limit",
    }
    unknown_experiment = set(experiment) - allowed_experiment
    if unknown_experiment:
        key = sorted(unknown_experiment)[0]
        raise ValueError(f"Unknown experiment config key: {key}")
    benchmark = experiment.get("benchmark")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    raw_instances = config.get("instances", {})
    _validate_instance_config(benchmark, raw_instances)
    if not config.get("strategies") or not config.get("solvers"):
        raise ValueError("Config must define at least one strategy and solver")
    known = set(TRANSFORMATIONS) | {
        "gdp.hull",
        "gdp.bigm",
        "gdp.mbigm",
        "gdp.binary_multiplication",
    }
    strategies = []
    identities = set()
    labels = {}
    for index, raw_strategy in enumerate(config["strategies"], start=1):
        unknown_strategy = set(raw_strategy) - {"name", "label", "options"}
        if unknown_strategy:
            key = sorted(unknown_strategy)[0]
            label = raw_strategy.get("label", raw_strategy.get("name", "unknown"))
            raise ValueError(
                f"Unknown strategy config key in entry {index} ({label!r}): {key}"
            )
        name = raw_strategy.get("name")
        if name not in known:
            raise ValueError(f"Unknown transformation: {name}")
        options = _normalize_strategy_options(name, dict(raw_strategy.get("options", {})))
        identity = _canonical({"transformation": name, "options": options})
        label = raw_strategy.get("label", name)
        if identity in identities:
            raise ValueError(f"Duplicate computational strategy: {name} {options}")
        if label in labels and labels[label] != identity:
            raise ValueError(f"Strategy label {label!r} refers to multiple strategies")
        identities.add(identity)
        labels[label] = identity
        strategies.append({"name": name, "label": label, "options": options})
    time_limit = _time_limit(experiment, "time_limit", 3600)
    modes = experiment.get("modes", ["solve"])
    if (
        not isinstance(modes, list)
        or not modes
        or any(not isinstance(mode, str) or mode not in RUN_MODES for mode in modes)
    ):
        raise ValueError(f"experiment.modes must be a nonempty list drawn from {sorted(RUN_MODES)}")
    if len(modes) != len(set(modes)):
        raise ValueError("experiment.modes must not contain duplicates")
    root_time_limit = _time_limit(experiment, "root_time_limit", time_limit)
    relaxation_time_limit = _time_limit(experiment, "relaxation_time_limit", time_limit)
    solvers = []
    for index, raw_solver in enumerate(config["solvers"], start=1):
        unknown_solver = set(raw_solver) - {"name", "subsolver", "variant"}
        if unknown_solver:
            key = sorted(unknown_solver)[0]
            label = raw_solver.get("subsolver", "unknown")
            raise ValueError(
                f"Unknown solver config key in entry {index} ({label!r}): {key}"
            )
        solver_name = raw_solver.get("name", "gams")
        if solver_name != "gams":
            raise ValueError("Only the GAMS solver interface is supported")
        subsolver = raw_solver["subsolver"]
        variant = raw_solver.get("variant")
        for mode in modes:
            options_for(subsolver, time_limit, variant, mode)
        solvers.append({"name": solver_name, "subsolver": subsolver, "variant": variant})
    instances = _normalize_instances(benchmark, raw_instances)
    return {
        "experiment": {
            "benchmark": benchmark,
            "base_seed": int(experiment.get("base_seed", 0)),
            "time_limit": time_limit,
            "modes": modes,
            "root_time_limit": root_time_limit,
            "relaxation_time_limit": relaxation_time_limit,
        },
        "instances": instances,
        "strategies": strategies,
        "solvers": solvers,
    }


def expand_jobs(config: dict[str, Any]) -> list[Job]:
    experiment = config["experiment"]
    benchmark_name = experiment["benchmark"]
    limits = {
        "solve": float(experiment.get("time_limit", 3600)),
        "root": float(experiment.get("root_time_limit", experiment.get("time_limit", 3600))),
        "relaxation": float(
            experiment.get("relaxation_time_limit", experiment.get("time_limit", 3600))
        ),
    }
    cases = BENCHMARKS[benchmark_name].cases(
        config.get("instances", {}), int(experiment.get("base_seed", 0))
    )
    jobs = []
    seen = set()
    for case in cases:
        for strategy in config["strategies"]:
            for solver in config["solvers"]:
                for mode in experiment.get("modes", ["solve"]):
                    variant = solver.get("variant")
                    time_limit = limits[mode]
                    solve_options = options_for(solver["subsolver"], time_limit, variant, mode)
                    identity = {
                        "benchmark": benchmark_name,
                        "instance_params": case.params,
                        "seed": case.seed,
                        "transformation": strategy["name"],
                        "transformation_options": strategy["options"],
                        "solver": solver["name"],
                        "subsolver": solver["subsolver"],
                        "variant": variant,
                        "mode": mode,
                        "time_limit": time_limit,
                        "solver_options": solve_options,
                    }
                    fingerprint = _canonical(identity)
                    run_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
                    if run_id in seen:
                        raise ValueError(f"Duplicate planned job fingerprint: {run_id}")
                    seen.add(run_id)
                    jobs.append(
                        Job(
                            run_id,
                            benchmark_name,
                            case.instance_id,
                            case.params,
                            case.seed,
                            strategy["name"],
                            strategy.get("label", strategy["name"]),
                            strategy.get("options", {}),
                            solver.get("name", "gams"),
                            solver["subsolver"],
                            variant,
                            mode,
                            time_limit,
                        )
                    )
    return jobs


def build_manifest(
    config: dict[str, Any],
    jobs: list[Job],
    execution_limit: int | None = None,
    versions: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Return the normalized campaign plan used by execution and reporting."""
    instances = []
    seen_instances = set()
    for job in jobs:
        identity = (job.instance_id, _canonical(job.params), job.seed)
        if identity in seen_instances:
            continue
        seen_instances.add(identity)
        instances.append({"instance_id": job.instance_id, "params": job.params, "seed": job.seed})
    return {
        "schema_version": 1,
        "config": config,
        "planned_jobs": [asdict(job) for job in jobs],
        "instances": instances,
        "execution": {"initial_limit": execution_limit},
        "versions": _versions() if versions is None else versions,
    }


def _first_difference(existing: Any, expected: Any, path: str) -> str | None:
    if type(existing) is not type(expected):
        return path
    if isinstance(existing, dict):
        for key in sorted(set(existing) | set(expected)):
            child = f"{path}.{key}" if path else key
            if key not in existing or key not in expected:
                return child
            difference = _first_difference(existing[key], expected[key], child)
            if difference:
                return difference
        return None
    if isinstance(existing, list):
        if len(existing) != len(expected):
            return path
        for index, (old, new) in enumerate(zip(existing, expected, strict=True)):
            difference = _first_difference(old, new, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if existing == expected else path


def _manifest_mismatch(existing: dict[str, Any], expected: dict[str, Any]) -> str | None:
    ignored = {"execution", "versions"}
    existing_core = {key: value for key, value in existing.items() if key not in ignored}
    expected_core = {key: value for key, value in expected.items() if key not in ignored}
    if existing_core == expected_core:
        return None
    config_field = _first_difference(
        existing_core.get("config"), expected_core.get("config"), "config"
    )
    if config_field:
        return f"{config_field} differs"
    old_jobs = {job["run_id"] for job in existing.get("planned_jobs", [])}
    new_jobs = {job["run_id"] for job in expected.get("planned_jobs", [])}
    if old_jobs != new_jobs:
        return (
            f"planned job set differs: {len(new_jobs - old_jobs)} added, "
            f"{len(old_jobs - new_jobs)} removed"
        )
    field = _first_difference(existing_core, expected_core, "") or "manifest"
    return f"{field} differs"


def _environment_differences(existing: dict[str, Any], current: dict[str, str | None]) -> list[str]:
    recorded = existing.get("versions", {})
    return [
        f"{name}: manifest={recorded.get(name)!r}, current={current.get(name)!r}"
        for name in ("python", "pyomo", "gams", "git_sha")
        if recorded.get(name) != current.get(name)
    ]


def _validate_existing_manifest(
    existing: dict[str, Any], expected: dict[str, Any], strict_env: bool
) -> None:
    validate_manifest(existing)
    mismatch = _manifest_mismatch(existing, expected)
    if mismatch:
        raise ValueError(f"Cannot resume: {mismatch}")
    differences = _environment_differences(existing, expected.get("versions", {}))
    if differences:
        message = "Resume environment differs from manifest: " + "; ".join(differences)
        if strict_env:
            raise ValueError(message)
        print(f"WARNING: {message}", file=sys.stderr)


def _prepare_manifest(
    output_directory: Path,
    manifest: dict[str, Any],
    resume: bool,
    strict_env: bool = False,
) -> None:
    manifest_path = output_directory / "manifest.json"
    stale_temporary = output_directory / ".manifest.json.tmp"
    entries = (
        [
            path
            for path in output_directory.iterdir()
            if path != stale_temporary
            and not (path.name.startswith(".manifest.") and path.name.endswith(".tmp"))
        ]
        if output_directory.exists()
        else []
    )
    nonempty = bool(entries)
    if nonempty:
        if not resume and entries == [manifest_path]:
            try:
                existing = json.loads(manifest_path.read_text())
                validate_manifest(existing)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid campaign manifest: {error}") from error
            if _manifest_mismatch(existing, manifest) is None:
                return
        if not resume:
            raise ValueError(
                f"Output directory is not empty: {output_directory}; use --resume to continue"
            )
        if not manifest_path.exists():
            raise ValueError(f"Cannot resume without manifest: {manifest_path}")
        try:
            existing = json.loads(manifest_path.read_text())
            _validate_existing_manifest(existing, manifest, strict_env)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid campaign manifest: {error}") from error
        if stale_temporary.exists():
            stale_temporary.unlink()
        return
    if stale_temporary.exists():
        stale_temporary.unlink()
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=output_directory, prefix=".manifest.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, manifest_path)
        except FileExistsError:
            try:
                existing = json.loads(manifest_path.read_text())
                _validate_existing_manifest(existing, manifest, strict_env)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as validation_error:
                raise ValueError(
                    f"Concurrent campaign manifest validation failed: {validation_error}"
                ) from validation_error
    finally:
        temporary.unlink(missing_ok=True)


def _repository_git_sha() -> str | None:
    for directory in Path(__file__).resolve().parents:
        if not (directory / ".git").exists():
            continue
        try:
            return subprocess.run(
                ["git", "-C", str(directory), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return None


def _parse_gams_version(output: str) -> str | None:
    """Extract the stable release token, excluding the banner timestamp."""
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", output)
    return match.group(1) if match else None


def _versions() -> dict[str, str | None]:
    gams_version = None
    gams = shutil.which("gams")
    if gams:
        try:
            output = subprocess.run(
                [gams, "-version"], capture_output=True, text=True, check=False, timeout=10
            )
            gams_version = _parse_gams_version(output.stdout or output.stderr)
        except (OSError, subprocess.TimeoutExpired, IndexError):
            pass
    return {
        "python": platform.python_version(),
        "pyomo": pyomo.version.__version__,
        "gams": gams_version,
        # GAMS/Pyomo does not expose the selected subsolver version reliably.
        "solver": None,
        "git_sha": _repository_git_sha(),
    }


def _as_float(candidate) -> float | None:
    try:
        result = float(candidate)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_bound(candidate) -> float | None:
    result = _as_float(candidate)
    return result if result is not None and abs(result) < 1e19 else None


def _as_int(candidate) -> int | None:
    result = _as_float(candidate)
    if result is None or result < 0 or not result.is_integer():
        return None
    return int(result)


def _status(
    result,
    log_text: str,
    subsolver: str,
    time_limit: float,
    solver_time_sec: float | None,
    wall_time_sec: float,
    rel_gap: float | None,
    mode: str = "solve",
    lower_bound: float | None = None,
) -> str:
    termination = result.solver.termination_condition
    solver_status = result.solver.status
    if solver_status in {SolverStatus.error, SolverStatus.aborted}:
        return "solver_error"
    if solver_status == SolverStatus.warning and mode != "root":
        return "solver_error"
    if (
        termination
        in {
            TerminationCondition.globallyOptimal,
            TerminationCondition.optimal,
            TerminationCondition.feasible,
        }
        and rel_gap is not None
        and rel_gap <= (TOLS["rel_gap"] if mode == "root" else 1e-12)
        and (
            mode != "root"
            or termination
            in {TerminationCondition.globallyOptimal, TerminationCondition.optimal}
        )
        and solver_status == SolverStatus.ok
    ):
        return (
            "globally_optimal" if termination == TerminationCondition.globallyOptimal else "optimal"
        )
    if solver_timed_out(log_text, subsolver):
        return "timeout"
    # Pyomo collapses GAMS modelstat 1 and 8, while Gurobi uses solvestat 4
    # for node limits, so only the solver log distinguishes this stop from an optimum.
    if mode == "root" and "node limit reached" in log_text.lower():
        return "node_limit"
    if termination in {
        TerminationCondition.maxTimeLimit,
        TerminationCondition.maxIterations,
        TerminationCondition.maxEvaluations,
    }:
        return "timeout"
    if solver_status == SolverStatus.warning:
        if mode == "root" and termination in {
            TerminationCondition.globallyOptimal,
            TerminationCondition.optimal,
            TerminationCondition.feasible,
            TerminationCondition.other,
        }:
            return "feasible"
        return "solver_error"
    effective_time = solver_time_sec if solver_time_sec is not None else wall_time_sec
    if (
        termination
        in {
            TerminationCondition.globallyOptimal,
            TerminationCondition.optimal,
            TerminationCondition.feasible,
        }
        and effective_time >= time_limit
        and (rel_gap is None or rel_gap > 1e-12)
    ):
        return "timeout"
    if (
        mode == "root"
        and termination
        in {
            TerminationCondition.globallyOptimal,
            TerminationCondition.optimal,
            TerminationCondition.feasible,
            TerminationCondition.other,
        }
        and (rel_gap is None or rel_gap > 1e-12)
    ):
        return "feasible"
    if termination == TerminationCondition.globallyOptimal and solver_status == SolverStatus.ok:
        return "globally_optimal"
    if termination == TerminationCondition.optimal and solver_status == SolverStatus.ok:
        return "optimal"
    if termination == TerminationCondition.locallyOptimal:
        return "locally_optimal"
    if termination in {TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded}:
        return "infeasible"
    if termination in {TerminationCondition.feasible, TerminationCondition.other}:
        return "feasible"
    return "solver_error"


def _solver_diagnostic(result) -> str | None:
    from pyomo.opt.results.container import UndefinedData

    for name in ("message", "termination_message"):
        message = getattr(result.solver, name, None)
        if message is not None and not isinstance(message, UndefinedData) and str(message).strip():
            return str(message)
    return (
        f"solver status={result.solver.status}; termination={result.solver.termination_condition}"
    )


def _base_record(
    job: Job,
    status: str,
    started: float,
    versions: dict[str, str | None],
    error: str | None = None,
) -> RunRecord:
    return RunRecord(
        run_id=job.run_id,
        benchmark=job.benchmark,
        instance_id=job.instance_id,
        instance_params=job.params,
        seed=job.seed,
        strategy=job.label,
        transformation=job.strategy,
        transformation_options=job.transformation_options,
        solver=job.solver,
        subsolver=job.subsolver,
        variant=job.variant,
        mode=job.mode,
        time_limit=job.time_limit,
        duration_sec=time.perf_counter() - started,
        solver_time_sec=None,
        status=status,
        timestamp=datetime.now(UTC).isoformat(),
        versions=dict(versions),
        error=error,
    )


def _describe(error: BaseException) -> str:
    """Error type and message; ``repr`` of OSError omits the offending path."""
    return f"{type(error).__name__}: {error}"


def _attach_transform_counts(record: RunRecord, model, counts: dict[str, int]) -> None:
    for name, count in counts.items():
        setattr(record, name, count)
    path_counts = getattr(model, "_exact_hull_path_counts", None)
    if path_counts is not None:
        for name, count in path_counts.items():
            setattr(record, name, count)
    for name, transform_value in getattr(model, "_exact_hull_transform_stats", {}).items():
        setattr(record, name, transform_value)


def _attach_log_metadata(record: RunRecord, log_path: Path, subsolver: str) -> str:
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    if log_text:
        metadata = parse_solver_metadata(log_text, subsolver)
        record.versions["solver"] = metadata.pop("version")
        for name, metadata_value in metadata.items():
            setattr(record, name, metadata_value)
    return log_text


def _load_solution(model, result) -> bool:
    """Load the solver's solution into ``model``; False when no usable incumbent exists.

    GAMS may return loadable zero variable levels when no incumbent exists, so a finite
    reported upper bound is required in addition to a loadable solution.
    """
    problem = getattr(result, "problem", None)
    if problem is None or _as_bound(getattr(problem, "upper_bound", None)) is None:
        return False
    solutions = getattr(result, "solution", None)
    if solutions is None:  # result objects without Pyomo's solution container
        return True
    if len(solutions) == 0:
        return False
    try:
        model.solutions.load_from(result)
    except (ValueError, TypeError, KeyError):
        return False
    return True


class _GamsMbigmSolver:
    """V1-marked adapter allowing Pyomo's mbigm estimator to use GAMS."""

    name = "gams/gurobi"
    options: dict[str, Any] = {}

    def __init__(self):
        self.m_estimation_time_sec = 0.0
        self.m_estimation_subsolves = 0

    @staticmethod
    def api_version():
        return SolverAPIVersion.V1

    def solve(self, model, **kwargs):
        kwargs.pop("keepfiles", None)
        standalone = ConcreteModel()
        expressions = [
            objective.expr
            for objective in model.component_data_objects(Objective, active=True)
        ] + [
            constraint.expr
            for constraint in model.component_data_objects(Constraint, active=True)
        ]
        variables = []
        seen = set()
        from pyomo.core.expr import identify_variables

        for expression in expressions:
            for variable in identify_variables(expression, include_fixed=True):
                if id(variable) not in seen:
                    variables.append(variable)
                    seen.add(id(variable))
        substitute = {}
        for index, variable in enumerate(variables):
            clone = Var(domain=variable.domain, bounds=(variable.lb, variable.ub))
            standalone.add_component(f"v_{index}", clone)
            if variable.fixed:
                clone.fix(value(variable))
            substitute[id(variable)] = clone
        standalone.constraints = ConstraintList()
        for constraint in model.component_data_objects(Constraint, active=True):
            body = clone_without_expression_components(constraint.body, substitute=substitute)
            lower = clone_without_expression_components(constraint.lower, substitute=substitute)
            upper = clone_without_expression_components(constraint.upper, substitute=substitute)
            standalone.constraints.add((lower, body, upper))
        objective = next(model.component_data_objects(Objective, active=True))
        standalone.objective = Objective(
            expr=clone_without_expression_components(objective.expr, substitute=substitute),
            sense=objective.sense,
        )
        mbigm_options = options_for("gurobi", 30)
        mbigm_options.insert(mbigm_options.index("$offecho"), "DualReductions 0")
        # Gurobi must stop strictly inside GAMS's optcr; a stop exactly at the
        # shared 1e-6 gap can be reclassified as non-optimal on readback.
        mbigm_options[mbigm_options.index(f"MIPGap {TOLS['rel_gap']:g}")] = "MIPGap 1e-7"
        solve_started = time.perf_counter()
        self.m_estimation_subsolves += 1
        try:
            result = SolverFactory("gams").solve(
                standalone,
                solver="gurobi",
                add_options=mbigm_options,
                **kwargs,
            )
        finally:
            self.m_estimation_time_sec += time.perf_counter() - solve_started
        lower = _as_bound(getattr(result.problem, "lower_bound", None))
        upper = _as_bound(getattr(result.problem, "upper_bound", None))
        close_bound = (
            lower is not None
            and upper is not None
            and abs(upper - lower)
            <= TOLS["rel_gap"] * max(abs(lower), abs(upper), 1.0)
        )
        termination = result.solver.termination_condition
        if result.solver.status != SolverStatus.ok:
            # mbigm treats infeasible as a proof that it may prune a disjunct.  Only
            # pass recognized success/proof terminations from a clean solver run.
            result.solver.termination_condition = TerminationCondition.error
        elif termination == TerminationCondition.globallyOptimal or (
            termination == TerminationCondition.feasible and close_bound
        ):
            result.solver.termination_condition = TerminationCondition.optimal
        elif termination not in {
            TerminationCondition.optimal,
            TerminationCondition.infeasible,
        }:
            result.solver.termination_condition = TerminationCondition.error
        return result


def _mbigm_options_fingerprint(options: dict[str, Any]) -> str:
    solver = options.get("solver")
    if isinstance(solver, _GamsMbigmSolver):
        solver_name = "gurobi"
    elif isinstance(solver, str) or solver is None:
        solver_name = solver
    else:
        solver_name = getattr(solver, "name", str(solver))
    normalized = {
        **{key: value for key, value in options.items() if key != "solver"},
        "solver": solver_name,
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_mbigm_cache(
    model, path: Path, options_fingerprint: str
) -> tuple[ComponentMap, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 3:
        raise ValueError(f"M-value cache {path} has unsupported schema version")
    if payload.get("options_fingerprint") != options_fingerprint:
        raise ValueError(
            f"M-value cache {path} was created with different mbigm options"
        )
    values = ComponentMap()
    for row in payload["values"]:
        constraint = model.find_component(row["constraint"])
        disjunct = model.find_component(row["disjunct"])
        if constraint is None or disjunct is None:
            raise ValueError(f"M-value cache {path} does not match the rebuilt instance")
        values[constraint, disjunct] = tuple(row["value"])
    return values, payload


def _write_mbigm_cache(path: Path, data: dict[str, Any]) -> None:
    """Atomically publish one valid M set, accepting an identical race winner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            existing = json.loads(path.read_text())
            if existing.get("options_fingerprint") != data["options_fingerprint"]:
                raise ValueError(
                    f"M-value cache {path} was concurrently created with different mbigm options"
                ) from error
            if existing.get("values") != data.get("values"):
                raise ValueError(
                    f"M-value cache {path} was concurrently created with diverging M values; "
                    "rerun this job"
                ) from error
    finally:
        temporary.unlink(missing_ok=True)


def transform_model(
    model,
    strategy: str,
    options: dict[str, Any],
    mode: str,
    *,
    mbigm_cache_path: Path | None = None,
    mbigm_provenance: dict[str, Any] | None = None,
):
    """Apply the same logical, GDP, structural-count, and mode steps used by jobs."""
    original_variables = list(model.component_data_objects(Var, active=None, descend_into=True))
    original_booleans = list(
        model.component_data_objects(BooleanVar, active=None, descend_into=True)
    )
    original_disjuncts = list(
        model.component_data_objects(Disjunct, active=None, descend_into=True)
    )
    transform_started = time.perf_counter()
    adapter = None
    cache_hit = False
    cache_payload = None
    try:
        TransformationFactory("core.logical_to_linear").apply_to(model)
        applied_options = dict(options)
        options_fingerprint = _mbigm_options_fingerprint(applied_options)
        cache_hit = (
            strategy == "gdp.mbigm"
            and mbigm_cache_path is not None
            and mbigm_cache_path.exists()
        )
        if strategy == "gdp.mbigm" and applied_options.get("solver") == "gurobi":
            adapter = _GamsMbigmSolver()
            applied_options["solver"] = adapter
            applied_options["threads"] = 1
        if cache_hit:
            applied_options["bigM"], cache_payload = _read_mbigm_cache(
                model, mbigm_cache_path, options_fingerprint
            )
        transformation = TransformationFactory(strategy)
        transformation.apply_to(model, **applied_options)
        if strategy == "gdp.mbigm" and mbigm_cache_path is not None and not cache_hit:
            rows = []
            for (constraint, disjunct), bounds in transformation.get_all_M_values(model).items():
                parent = constraint.parent_block()
                while parent is not None and not isinstance(parent, Disjunct):
                    parent = parent.parent_block()
                if parent is disjunct:
                    continue
                rows.append(
                    {
                        "constraint": constraint.name,
                        "disjunct": disjunct.name,
                        "value": [
                            _as_float(bound) if bound is not None else None
                            for bound in bounds
                        ],
                    }
                )
            rows.sort(key=lambda row: (row["constraint"], row["disjunct"]))
            _write_mbigm_cache(
                mbigm_cache_path,
                {
                    "schema_version": 3,
                    "options_fingerprint": options_fingerprint,
                    "m_estimation_time_total_sec": (
                        adapter.m_estimation_time_sec if adapter is not None else None
                    ),
                    "provenance": mbigm_provenance or {},
                    "values": rows,
                },
            )
            cache_payload = json.loads(mbigm_cache_path.read_text())
        counts = structural_counts(model)
        if mode == "relaxation":
            TransformationFactory("core.relax_integer_vars").apply_to(model)
    finally:
        model._exact_hull_transform_stats = {
            "transform_sec": time.perf_counter() - transform_started,
            "m_estimation_time_sec": (
                adapter.m_estimation_time_sec if strategy == "gdp.mbigm" and adapter else 0.0
            )
            if strategy == "gdp.mbigm"
            else None,
            "m_estimation_time_total_sec": (
                _as_float(cache_payload.get("m_estimation_time_total_sec"))
                if cache_payload is not None
                else (adapter.m_estimation_time_sec if adapter is not None else 0.0)
            )
            if strategy == "gdp.mbigm"
            else None,
            "m_estimation_cache_hit": cache_hit if strategy == "gdp.mbigm" else None,
            "m_estimation_subsolves": (
                adapter.m_estimation_subsolves if strategy == "gdp.mbigm" and adapter else 0
            )
            if strategy == "gdp.mbigm"
            else None,
        }
    return counts, original_variables, original_booleans, original_disjuncts


def run_job(
    job: Job,
    output_directory: Path,
    versions: dict[str, str | None] | None = None,
) -> RunRecord:
    started = time.perf_counter()
    versions = _versions() if versions is None else versions
    benchmark = BENCHMARKS[job.benchmark]
    try:
        model = benchmark.build(BenchmarkCase(job.instance_id, job.params, job.seed))
    except Exception as error:  # every planned job produces a result
        return _base_record(job, "build_error", started, versions, _describe(error))
    objectives = list(model.component_data_objects(Objective, active=True))
    if len(objectives) != 1 or not objectives[0].is_minimizing():
        return _base_record(
            job,
            "build_error",
            started,
            versions,
            "The model must have exactly one active minimization objective",
        )
    job_directory = output_directory / "jobs" / job.run_id
    scratch = job_directory / "scratch"
    log_path = job_directory / "solver.log"
    try:
        counts, original_variables, original_booleans, original_disjuncts = transform_model(
            model,
            job.strategy,
            job.transformation_options,
            job.mode,
            mbigm_cache_path=(
                output_directory / "mbigm" / f"{job.instance_id}.json"
                if job.strategy == "gdp.mbigm"
                else None
            ),
            mbigm_provenance={
                "benchmark": job.benchmark,
                "instance_id": job.instance_id,
                "instance_params": job.params,
                "seed": job.seed,
                "pyomo": versions.get("pyomo"),
                "solver": "gams/gurobi",
                "time_limit_sec": 30,
                "dual_reductions": 0,
                "threads": 1,
            },
        )
    except Exception as error:
        record = _base_record(job, "transform_error", started, versions, _describe(error))
        _attach_transform_counts(record, model, {})
        return record
    try:
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)
        solver = SolverFactory("gams")
        # keepfiles=True so Pyomo never tries to delete a primal GDX that GAMS did not
        # write (no-incumbent timeouts); the scratch directory is removed below. Solutions
        # are loaded manually after checking that GAMS reported an incumbent objective.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = solver.solve(
                    model,
                    solver=job.subsolver,
                    add_options=options_for(job.subsolver, job.time_limit, job.variant, job.mode),
                    tee=False,
                    logfile=str(log_path),
                    keepfiles=True,
                    load_solutions=False,
                    tmpdir=str(scratch),
                )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        has_solution = _load_solution(model, result)
        record = _base_record(job, "solver_error", started, versions)
        _attach_transform_counts(record, model, counts)
        log_text = _attach_log_metadata(record, log_path, job.subsolver)
        record.solver_time_sec = _as_float(getattr(result.solver, "user_time", None))
        record.solver_status = str(result.solver.status)
        record.termination = str(result.solver.termination_condition)
        problem = result.problem
        record.lower_bound = _as_bound(getattr(problem, "lower_bound", None))
        record.upper_bound = _as_bound(getattr(problem, "upper_bound", None))
        record.num_variables = _as_int(getattr(problem, "number_of_variables", None))
        record.num_constraints = _as_int(getattr(problem, "number_of_constraints", None))
        record.num_nonzeros = _as_int(getattr(problem, "number_of_nonzeros", None))
        record.num_discrete_variables = _as_int(
            getattr(problem, "number_of_integer_variables", None)
        )
        if log_text:
            parsed_lower, parsed_upper = parse_solver_bounds(log_text, job.subsolver)
            record.lower_bound = (
                record.lower_bound if record.lower_bound is not None else _as_bound(parsed_lower)
            )
            record.upper_bound = (
                record.upper_bound if record.upper_bound is not None else _as_bound(parsed_upper)
            )
        if not has_solution:
            record.upper_bound = None
            record.objective = None
        if record.lower_bound is not None and record.upper_bound is not None:
            record.abs_gap = abs(record.upper_bound - record.lower_bound)
            scale = max(abs(record.lower_bound), abs(record.upper_bound), 1e-12)
            record.rel_gap = record.abs_gap / scale
        if has_solution and objectives:
            record.objective = _as_float(value(objectives[0], exception=False))
        record.solution = {}
        if has_solution:
            if job.mode == "solve":
                try:
                    update_boolean_vars_from_binary(model)
                except ValueError:
                    # Preserve the numeric payload. Verification will classify
                    # fractional or contradictory indicator representations.
                    pass
            try:
                record.solution = benchmark.solution(model)
                indicator_variable_ids = {
                    id(disjunct.binary_indicator_var) for disjunct in original_disjuncts
                }
                record.solution["variables"] = {
                    variable.name: _as_float(value(variable, exception=False))
                    for variable in original_variables
                    if id(variable) not in indicator_variable_ids
                }
                record.solution["indicators"] = {
                    disjunct.name: _as_float(
                        value(disjunct.binary_indicator_var, exception=False)
                    )
                    for disjunct in original_disjuncts
                }
                record.solution["booleans"] = {
                    boolean.name: value(boolean, exception=False)
                    for boolean in original_booleans
                }
            except (KeyError, ValueError, TypeError):
                record.solution = {}
        record.duration_sec = time.perf_counter() - started
        record.status = _status(
            result,
            log_text,
            job.subsolver,
            job.time_limit,
            record.solver_time_sec,
            record.duration_sec,
            record.rel_gap,
            job.mode,
            record.lower_bound,
        )
        if (
            job.mode == "relaxation"
            and record.num_discrete_variables is not None
            and record.num_discrete_variables != 0
        ):
            record.status = "transform_error"
            record.error = (
                "Integrality relaxation left "
                f"{record.num_discrete_variables!r} discrete variables in the solved model"
            )
        elif record.status == "solver_error":
            record.error = _solver_diagnostic(result)
        record.timestamp = datetime.now(UTC).isoformat()
        return record
    except Exception as error:
        record = _base_record(job, "solver_error", started, versions, _describe(error))
        _attach_transform_counts(record, model, counts)
        try:
            _attach_log_metadata(record, log_path, job.subsolver)
        except Exception:
            pass
        return record


def _prepare_execution_tasks(
    config_path: Path,
    output_directory: Path,
    limit: int | None,
    resume: bool,
    rerun_statuses: set[str] | None = None,
    shard: tuple[int, int] | None = None,
    strict_env: bool = False,
    versions: dict[str, str | None] | None = None,
) -> list[_ExecutionTask]:
    config = load_config(config_path)
    planned_jobs = expand_jobs(config)
    versions = _versions() if versions is None else versions
    _prepare_manifest(
        output_directory,
        build_manifest(config, planned_jobs, execution_limit=limit, versions=versions),
        resume,
        strict_env,
    )
    jobs = planned_jobs
    if shard is not None:
        shard_number, shard_count = shard
        jobs = [job for index, job in enumerate(jobs) if index % shard_count == shard_number - 1]
    jobs = jobs[:limit] if limit is not None else jobs
    tasks = []
    for job in jobs:
        destination = output_directory / "jobs" / job.run_id / "result.json"
        if resume and destination.exists() and is_valid_result(
            destination, job.run_id, asdict(job)
        ):
            existing_status = RunRecord.from_dict(json.loads(destination.read_text())).status
            if not rerun_statuses or existing_status not in rerun_statuses:
                continue
        tasks.append(_ExecutionTask(job, output_directory, versions))
    return tasks


def _parallel_phases(
    tasks: list[_ExecutionTask],
) -> tuple[list[_ExecutionTask], dict[tuple[Path, str], list[_ExecutionTask]]]:
    phase_one = []
    gated: dict[tuple[Path, str], list[_ExecutionTask]] = {}
    mbigm_gates = set()
    for task in tasks:
        if task.job.strategy != "gdp.mbigm":
            phase_one.append(task)
            continue
        key = (task.output_directory, task.job.instance_id)
        if key not in mbigm_gates:
            mbigm_gates.add(key)
            phase_one.append(task)
        else:
            gated.setdefault(key, []).append(task)
    return phase_one, gated


def _run_job_worker(task: _ExecutionTask, concurrency: int) -> RunRecord:
    record = run_job(task.job, task.output_directory, task.versions)
    record.concurrency = concurrency
    destination = task.output_directory / "jobs" / task.job.run_id / "result.json"
    write_record_atomic(record, destination)
    return record


def _execute_tasks(tasks: list[_ExecutionTask], jobs: int) -> list[RunRecord]:
    if jobs == 1:
        return [_run_job_worker(task, jobs) for task in tasks]

    phase_one, gated = _parallel_phases(tasks)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    executor = ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=multiprocessing.get_context("spawn"),
    )
    records = []
    futures = {}
    try:
        for task in phase_one:
            futures[executor.submit(_run_job_worker, task, jobs)] = task
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                task = futures.pop(future)
                records.append(future.result())
                if task.job.strategy == "gdp.mbigm":
                    key = (task.output_directory, task.job.instance_id)
                    remaining = gated.get(key, [])
                    cache_path = (
                        task.output_directory / "mbigm" / f"{task.job.instance_id}.json"
                    )
                    if cache_path.exists():
                        released = gated.pop(key, [])
                    elif remaining:
                        released = [remaining.pop(0)]
                        if not remaining:
                            gated.pop(key)
                    else:
                        released = []
                    for gated_task in released:
                        futures[executor.submit(_run_job_worker, gated_task, jobs)] = gated_task
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()
    return records


def _validate_jobs(jobs: int) -> None:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs <= 0:
        raise ValueError("jobs must be a positive integer")


def run(
    config_path: Path,
    output_directory: Path,
    limit=None,
    resume=False,
    rerun_statuses: set[str] | None = None,
    shard: tuple[int, int] | None = None,
    strict_env: bool = False,
    jobs: int = 1,
) -> list[RunRecord]:
    _validate_jobs(jobs)
    tasks = _prepare_execution_tasks(
        config_path,
        output_directory,
        limit,
        resume,
        rerun_statuses,
        shard,
        strict_env,
    )
    return _execute_tasks(tasks, jobs)


def run_campaigns(
    config_paths: list[Path],
    output_root: Path,
    *,
    resume: bool = False,
    rerun_statuses: set[str] | None = None,
    strict_env: bool = False,
    jobs: int = 1,
) -> list[RunRecord]:
    if len(config_paths) < 2:
        raise ValueError("run_campaigns requires at least two config paths")
    _validate_jobs(jobs)
    stems = [path.stem for path in config_paths]
    if any(stem in {".", ".."} for stem in stems):
        raise ValueError("Config file stems '.' and '..' are not allowed")
    if len(set(stems)) != len(stems):
        raise ValueError("Config file stems must be unique")

    versions = _versions()
    tasks = []
    for config_path in config_paths:
        output_directory = output_root / config_path.stem
        tasks.extend(
            _prepare_execution_tasks(
                config_path,
                output_directory,
                None,
                resume,
                rerun_statuses,
                None,
                strict_env,
                versions,
            )
        )
    return _execute_tasks(tasks, jobs)
