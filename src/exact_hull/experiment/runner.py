"""Configuration loading, job expansion, and isolated GAMS execution."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import platform
import shutil
import subprocess
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyomo
from pyomo.environ import Objective, SolverFactory, TransformationFactory, value
from pyomo.opt import SolverStatus, TerminationCondition

from exact_hull.benchmarks import BENCHMARKS, INSTANCE_PARAMETERS
from exact_hull.benchmarks.base import BenchmarkCase
from exact_hull.experiment.logparse import (
    parse_gams_solvestat,
    parse_root_relaxation,
    parse_solver_bounds,
    solver_timed_out,
)
from exact_hull.experiment.results import (
    RunRecord,
    is_valid_result,
    validate_manifest,
    write_json_atomic,
    write_record_atomic,
)
from exact_hull.experiment.solvers import options_for
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
        "sparsity_factor": 0.0,
    },
    "kmeans": {},
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
    },
    "kmeans": {"n_dimensions", "n_clusters", "n_points"},
    "cstr": {"NT"},
    "clay": set(),
}
FLOAT_PARAMETERS = {"random_quadratic": {"sparsity_factor"}}
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
    time_limit: float


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
    return {key: _normalize_option_numbers(config[key]) for key in config}


def _normalize_integer_parameter(benchmark: str, key: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_integer_parameter(benchmark, key, item) for item in value]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Instance parameter {benchmark}.{key} must contain integers")
    if not math.isfinite(value) or not float(value).is_integer():
        raise ValueError(f"Instance parameter {benchmark}.{key} must contain integers")
    return int(value)


def _normalize_instances(benchmark: str, instances: dict[str, Any]) -> dict[str, Any]:
    axes = {key: value for key, value in instances.items() if key != "fixed"}
    supplied_fixed = dict(instances.get("fixed", {}))
    merged = dict(axes)
    merged.update(supplied_fixed)
    defaults = {**INSTANCE_DEFAULTS[benchmark], **RANGE_DEFAULTS.get(benchmark, {})}
    for key, default in defaults.items():
        merged.setdefault(key, default)

    for key, parameter_value in list(merged.items()):
        if key in INT_PARAMETERS[benchmark]:
            merged[key] = _normalize_integer_parameter(benchmark, key, parameter_value)
        elif key in FLOAT_PARAMETERS.get(benchmark, set()) or key in RANGE_PARAMETERS.get(
            benchmark, set()
        ):
            merged[key] = _normalize_option_numbers(parameter_value)

    fixed_keys = set(supplied_fixed) | set(RANGE_DEFAULTS.get(benchmark, {}))
    normalized = {key: value for key, value in merged.items() if key not in fixed_keys}
    if fixed_keys:
        normalized["fixed"] = {key: merged[key] for key in merged if key in fixed_keys}
    return normalized


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    experiment = config.get("experiment", {})
    benchmark = experiment.get("benchmark")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    raw_instances = config.get("instances", {})
    _validate_instance_config(benchmark, raw_instances)
    if not config.get("strategies") or not config.get("solvers"):
        raise ValueError("Config must define at least one strategy and solver")
    known = set(TRANSFORMATIONS) | {"gdp.hull", "gdp.bigm"}
    strategies = []
    identities = set()
    labels = {}
    for raw_strategy in config["strategies"]:
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
    solvers = []
    time_limit = float(experiment.get("time_limit", 3600))
    for raw_solver in config["solvers"]:
        solver_name = raw_solver.get("name", "gams")
        if solver_name != "gams":
            raise ValueError("Only the GAMS solver interface is supported")
        subsolver = raw_solver["subsolver"]
        variant = raw_solver.get("variant")
        options_for(subsolver, time_limit, variant)
        solvers.append({"name": solver_name, "subsolver": subsolver, "variant": variant})
    instances = _normalize_instances(benchmark, raw_instances)
    return {
        "experiment": {
            "benchmark": benchmark,
            "base_seed": int(experiment.get("base_seed", 0)),
            "time_limit": time_limit,
        },
        "instances": instances,
        "strategies": strategies,
        "solvers": solvers,
    }


def expand_jobs(config: dict[str, Any]) -> list[Job]:
    experiment = config["experiment"]
    benchmark_name = experiment["benchmark"]
    time_limit = float(experiment.get("time_limit", 3600))
    cases = BENCHMARKS[benchmark_name].cases(
        config.get("instances", {}), int(experiment.get("base_seed", 0))
    )
    jobs = []
    seen = set()
    for case in cases:
        for strategy in config["strategies"]:
            for solver in config["solvers"]:
                variant = solver.get("variant")
                solve_options = options_for(solver["subsolver"], time_limit, variant)
                identity = {
                    "benchmark": benchmark_name,
                    "instance_params": case.params,
                    "seed": case.seed,
                    "transformation": strategy["name"],
                    "transformation_options": strategy["options"],
                    "solver": solver["name"],
                    "subsolver": solver["subsolver"],
                    "variant": variant,
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
                        time_limit,
                    )
                )
    return jobs


def build_manifest(
    config: dict[str, Any], jobs: list[Job], execution_limit: int | None = None
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
    existing_core = {key: value for key, value in existing.items() if key != "execution"}
    expected_core = {key: value for key, value in expected.items() if key != "execution"}
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


def _prepare_manifest(output_directory: Path, manifest: dict[str, Any], resume: bool) -> None:
    manifest_path = output_directory / "manifest.json"
    stale_temporary = output_directory / ".manifest.json.tmp"
    entries = (
        [path for path in output_directory.iterdir() if path != stale_temporary]
        if output_directory.exists()
        else []
    )
    nonempty = bool(entries)
    if nonempty:
        if not resume:
            raise ValueError(
                f"Output directory is not empty: {output_directory}; use --resume to continue"
            )
        if not manifest_path.exists():
            raise ValueError(f"Cannot resume without manifest: {manifest_path}")
        try:
            existing = json.loads(manifest_path.read_text())
            validate_manifest(existing)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid campaign manifest: {error}") from error
        mismatch = _manifest_mismatch(existing, manifest)
        if mismatch:
            raise ValueError(f"Cannot resume: {mismatch}")
        if stale_temporary.exists():
            stale_temporary.unlink()
        return
    if stale_temporary.exists():
        stale_temporary.unlink()
    write_json_atomic(manifest, manifest_path)


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


def _versions() -> dict[str, str | None]:
    gams_version = None
    gams = shutil.which("gams")
    if gams:
        try:
            output = subprocess.run(
                [gams, "-version"], capture_output=True, text=True, check=False, timeout=10
            )
            gams_version = (output.stdout or output.stderr).splitlines()[0].strip() or None
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


def _status(
    result,
    log_text: str,
    subsolver: str,
    time_limit: float,
    solver_time_sec: float | None,
    wall_time_sec: float,
    rel_gap: float | None,
) -> str:
    termination = result.solver.termination_condition
    solvestat = parse_gams_solvestat(log_text)
    if result.solver.status in {SolverStatus.error, SolverStatus.aborted, SolverStatus.warning}:
        return "solver_error"
    if (
        termination
        in {
            TerminationCondition.globallyOptimal,
            TerminationCondition.optimal,
            TerminationCondition.feasible,
        }
        and rel_gap is not None
        and rel_gap <= 1e-12
    ):
        return (
            "globally_optimal" if termination == TerminationCondition.globallyOptimal else "optimal"
        )
    if solvestat == 3 or solver_timed_out(log_text, subsolver):
        return "timeout"
    if termination in {
        TerminationCondition.maxTimeLimit,
        TerminationCondition.maxIterations,
        TerminationCondition.maxEvaluations,
    }:
        return "timeout"
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
    if termination == TerminationCondition.globallyOptimal:
        return "globally_optimal"
    if termination == TerminationCondition.optimal:
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


def _load_solution(model, result) -> bool:
    """Load the solver's solution into ``model``; False when no usable incumbent exists.

    GAMS reports ``NA`` variable levels when a solve ends without an incumbent (model
    status 14). Pyomo's GAMS plugin still inserts a solution object, and loading it fails
    on the NaN levels of indicator variables, so failure to load is the no-solution signal.
    """
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
    try:
        TransformationFactory("core.logical_to_linear").apply_to(model)
        TransformationFactory(job.strategy).apply_to(model, **job.transformation_options)
    except Exception as error:
        return _base_record(job, "transform_error", started, versions, _describe(error))
    job_directory = output_directory / "jobs" / job.run_id
    scratch = job_directory / "scratch"
    log_path = job_directory / "solver.log"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        solver = SolverFactory("gams")
        # keepfiles=True so Pyomo never tries to delete a primal GDX that GAMS did not
        # write (no-incumbent timeouts); the scratch directory is removed below. Solutions
        # are loaded manually because Pyomo crashes loading NaN levels into indicators.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = solver.solve(
                    model,
                    solver=job.subsolver,
                    add_options=options_for(job.subsolver, job.time_limit, job.variant),
                    tee=False,
                    logfile=str(log_path),
                    keepfiles=True,
                    load_solutions=False,
                    tmpdir=str(scratch),
                )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        has_solution = _load_solution(model, result)
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        record = _base_record(job, "solver_error", started, versions)
        record.solver_time_sec = _as_float(getattr(result.solver, "user_time", None))
        problem = result.problem
        record.lower_bound = _as_float(getattr(problem, "lower_bound", None))
        record.upper_bound = _as_float(getattr(problem, "upper_bound", None))
        if log_text:
            record.root_relaxation = parse_root_relaxation(log_text, job.subsolver)
            parsed_lower, parsed_upper = parse_solver_bounds(log_text, job.subsolver)
            record.lower_bound = (
                record.lower_bound if record.lower_bound is not None else parsed_lower
            )
            record.upper_bound = (
                record.upper_bound if record.upper_bound is not None else parsed_upper
            )
        if record.lower_bound is not None and record.upper_bound is not None:
            record.abs_gap = abs(record.upper_bound - record.lower_bound)
            scale = max(abs(record.lower_bound), abs(record.upper_bound), 1e-12)
            record.rel_gap = record.abs_gap / scale
        objectives = list(model.component_data_objects(Objective, active=True))
        if has_solution and objectives:
            record.objective = _as_float(value(objectives[0], exception=False))
        record.solution = {}
        if has_solution:
            try:
                record.solution = benchmark.solution(model)
            except (ValueError, TypeError):
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
        )
        record.error = _solver_diagnostic(result) if record.status == "solver_error" else None
        record.timestamp = datetime.now(UTC).isoformat()
        return record
    except Exception as error:
        return _base_record(job, "solver_error", started, versions, _describe(error))


def run(config_path: Path, output_directory: Path, limit=None, resume=False) -> list[RunRecord]:
    config = load_config(config_path)
    planned_jobs = expand_jobs(config)
    _prepare_manifest(
        output_directory,
        build_manifest(config, planned_jobs, execution_limit=limit),
        resume,
    )
    jobs = planned_jobs[:limit] if limit is not None else planned_jobs
    versions = _versions()
    records = []
    for job in jobs:
        destination = output_directory / "jobs" / job.run_id / "result.json"
        if (
            resume
            and destination.exists()
            and is_valid_result(destination, job.run_id, asdict(job))
        ):
            continue
        record = run_job(job, output_directory, versions)
        write_record_atomic(record, destination)
        records.append(record)
    return records
