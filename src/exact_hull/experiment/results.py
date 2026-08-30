"""Stable per-job result records and aggregate exports."""

from __future__ import annotations

import json
import math
import numbers
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import pandas as pd

RunStatus = Literal[
    "optimal",
    "globally_optimal",
    "locally_optimal",
    "feasible",
    "timeout",
    "node_limit",
    "infeasible",
    "solver_error",
    "build_error",
    "transform_error",
]
RUN_STATUSES = frozenset(
    {
        "optimal",
        "globally_optimal",
        "locally_optimal",
        "feasible",
        "timeout",
        "node_limit",
        "infeasible",
        "solver_error",
        "build_error",
        "transform_error",
    }
)
VERIFIED_OPTIMAL_STATUSES = frozenset({"optimal", "globally_optimal"})
RUN_MODES = frozenset({"solve", "root", "relaxation"})


@dataclass
class RunRecord:
    run_id: str
    benchmark: str
    instance_id: str
    instance_params: dict[str, Any]
    seed: int
    strategy: str
    transformation: str
    transformation_options: dict[str, Any]
    solver: str
    subsolver: str
    variant: str | None
    mode: str
    time_limit: float
    duration_sec: float | None
    solver_time_sec: float | None
    status: RunStatus
    objective: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    abs_gap: float | None = None
    rel_gap: float | None = None
    num_variables: int | None = None
    num_constraints: int | None = None
    num_nonzeros: int | None = None
    num_discrete_variables: int | None = None
    solver_status: str | None = None
    termination: str | None = None
    solution: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    versions: dict[str, str | None] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        if not isinstance(data, dict):
            raise TypeError("Run record must be a JSON object")
        record = cls(**data)
        if not isinstance(record.run_id, str) or not record.run_id:
            raise ValueError("Run record run_id must be a nonempty string")
        if record.status not in RUN_STATUSES:
            raise ValueError(f"Unknown run status: {record.status}")
        if record.mode not in RUN_MODES:
            raise ValueError(f"Unknown run mode: {record.mode}")
        for name in (
            "benchmark",
            "instance_id",
            "strategy",
            "transformation",
            "solver",
            "subsolver",
            "timestamp",
        ):
            if not isinstance(getattr(record, name), str):
                raise TypeError(f"Run record {name} must be a string")
        if record.variant is not None and not isinstance(record.variant, str):
            raise TypeError("Run record variant must be a string or null")
        if record.error is not None and not isinstance(record.error, str):
            raise TypeError("Run record error must be a string or null")
        for name in ("solver_status", "termination"):
            value = getattr(record, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"Run record {name} must be a string or null")
        for name in ("instance_params", "transformation_options", "solution", "versions"):
            if not isinstance(getattr(record, name), dict):
                raise TypeError(f"Run record {name} must be an object")
        if isinstance(record.seed, bool) or not isinstance(record.seed, int):
            raise TypeError("Run record seed must be an integer")
        _validate_number("time_limit", record.time_limit, optional=False, nonnegative=True)
        for name in ("duration_sec", "solver_time_sec", "abs_gap", "rel_gap"):
            _validate_number(name, getattr(record, name), optional=True, nonnegative=True)
        for name in ("objective", "lower_bound", "upper_bound"):
            _validate_number(name, getattr(record, name), optional=True)
        for name in (
            "num_variables",
            "num_constraints",
            "num_nonzeros",
            "num_discrete_variables",
        ):
            value = getattr(record, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise TypeError(f"Run record {name} must be a nonnegative integer or null")
        return record


def _validate_number(name: str, value: Any, *, optional: bool, nonnegative: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        qualifier = "a real number or null" if optional else "a real number"
        raise TypeError(f"Run record {name} must be {qualifier}")
    if not math.isfinite(value):
        raise ValueError(f"Run record {name} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"Run record {name} must be nonnegative")


def write_json_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_record_atomic(record: RunRecord, path: Path) -> None:
    write_json_atomic(asdict(record), path)


def _load_record(
    path: Path, planned_jobs: dict[str, dict[str, Any]] | None = None
) -> tuple[RunRecord | None, str | None]:
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise TypeError("JSON root is not an object")
        run_id = data.get("run_id")
        if run_id != path.parent.name:
            raise ValueError(
                f"embedded run_id {run_id!r} does not match directory {path.parent.name!r}"
            )
        if planned_jobs is not None and run_id not in planned_jobs:
            raise ValueError(f"run_id {run_id!r} is not in the campaign manifest")
        record = RunRecord.from_dict(data)
        if planned_jobs is not None:
            _validate_record_identity(record, planned_jobs[run_id])
        return record, None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        return None, str(error)


def _validate_record_identity(record: RunRecord, planned_job: dict[str, Any]) -> None:
    expected = {
        "benchmark": planned_job["benchmark"],
        "instance_id": planned_job["instance_id"],
        "strategy": planned_job["label"],
        "transformation": planned_job["strategy"],
        "solver": planned_job["solver"],
        "subsolver": planned_job["subsolver"],
        "variant": planned_job["variant"],
        "mode": planned_job["mode"],
    }
    for name, value in expected.items():
        if getattr(record, name) != value:
            raise ValueError(
                f"record {name} {getattr(record, name)!r} does not match planned value {value!r}"
            )


def validate_manifest(manifest: Any) -> dict[str, dict[str, Any]]:
    """Validate the campaign fields required by execution and analysis."""
    if not isinstance(manifest, dict):
        raise TypeError("manifest root must be an object")
    if not isinstance(manifest.get("config"), dict):
        raise TypeError("manifest config must be an object")
    planned_jobs = manifest.get("planned_jobs")
    if not isinstance(planned_jobs, list):
        raise TypeError("manifest planned_jobs must be a list")
    required = {
        "run_id",
        "benchmark",
        "instance_id",
        "label",
        "strategy",
        "solver",
        "subsolver",
        "variant",
        "mode",
    }
    indexed = {}
    for index, job in enumerate(planned_jobs):
        if not isinstance(job, dict):
            raise TypeError(f"manifest planned_jobs[{index}] must be an object")
        missing = sorted(required - set(job))
        if missing:
            raise ValueError(f"manifest planned_jobs[{index}] is missing {missing}")
        run_id = job["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise TypeError(f"manifest planned_jobs[{index}].run_id must be a nonempty string")
        if run_id in indexed:
            raise ValueError(f"manifest contains duplicate run_id {run_id!r}")
        for name in required - {"run_id", "variant"}:
            if not isinstance(job[name], str):
                raise TypeError(f"manifest planned_jobs[{index}].{name} must be a string")
        if job["variant"] is not None and not isinstance(job["variant"], str):
            raise TypeError(f"manifest planned_jobs[{index}].variant must be a string or null")
        if job["mode"] not in RUN_MODES:
            raise ValueError(f"manifest planned_jobs[{index}].mode is invalid")
        indexed[run_id] = job
    instances = manifest.get("instances")
    if not isinstance(instances, list) or not all(
        isinstance(instance, dict) and isinstance(instance.get("instance_id"), str)
        for instance in instances
    ):
        raise TypeError("manifest instances must be a list of instance objects")
    return indexed


def is_valid_result(
    path: Path, expected_run_id: str, planned_job: dict[str, Any] | None = None
) -> bool:
    """Return whether a result is complete, current-schema, and in the expected directory."""
    jobs = {expected_run_id: planned_job} if planned_job is not None else None
    record, _ = _load_record(path, jobs)
    if record is not None and record.run_id != expected_run_id:
        return False
    return record is not None


def read_campaign(run_directory: Path) -> tuple[dict[str, Any], list[RunRecord]]:
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Campaign manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
        planned = validate_manifest(manifest)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid campaign manifest: {error}") from error
    records = []
    skipped = []
    for path in sorted(run_directory.glob("jobs/*/result.json")):
        record, reason = _load_record(path, planned)
        if record is None:
            skipped.append(f"{path.parent.name}: {reason}")
            continue
        records.append(record)
    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} invalid or foreign result file(s): " + "; ".join(skipped),
            stacklevel=2,
        )
    return manifest, records


def aggregate(
    run_directory: Path, xlsx: bool = False, records: list[RunRecord] | None = None
) -> Path:
    from exact_hull.analysis.outcomes import ground_truth, is_correct

    if records is None:
        _, records = read_campaign(run_directory)
    truth = ground_truth(records)
    rows = []
    for record in records:
        row = asdict(record)
        row["ground_truth"] = truth.get(record.instance_id)
        row["correct"] = (
            record.mode == "solve"
            and record.status in VERIFIED_OPTIMAL_STATUSES
            and is_correct(record.objective, row["ground_truth"])
        )
        for field_name in ("instance_params", "transformation_options", "solution", "versions"):
            row[field_name] = json.dumps(row[field_name], sort_keys=True)
        rows.append(row)
    columns = [field.name for field in fields(RunRecord)] + ["ground_truth", "correct"]
    frame = pd.DataFrame(rows, columns=columns)
    destination = run_directory / ("results.xlsx" if xlsx else "results.csv")
    if xlsx:
        frame.to_excel(destination, index=False)
    else:
        frame.to_csv(destination, index=False)
    return destination
