"""Independent feasibility and objective checks on original GDP models."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path

from pyomo.core.base.componentuid import ComponentUID
from pyomo.environ import BooleanVar, Constraint, LogicalConstraint, Objective, Var, value
from pyomo.gdp import Disjunct, Disjunction

from exact_hull.analysis.outcomes import is_correct
from exact_hull.benchmarks import BENCHMARKS
from exact_hull.benchmarks.base import BenchmarkCase
from exact_hull.experiment.results import RunRecord, read_campaign
from exact_hull.experiment.solvers import TOLS


def _constraint_violation(constraint) -> float:
    body = value(constraint.body, exception=False)
    if body is None or not math.isfinite(body):
        return math.inf
    violations = [0.0]
    if constraint.lower is not None:
        lower = value(constraint.lower, exception=False)
        if lower is None or not math.isfinite(lower):
            return math.inf
        violations.append(abs(body - lower) if constraint.equality else max(0.0, lower - body))
    if constraint.upper is not None and not constraint.equality:
        upper = value(constraint.upper, exception=False)
        if upper is None or not math.isfinite(upper):
            return math.inf
        violations.append(max(0.0, body - upper))
    return max(violations)


def _result(record: RunRecord, status: str, **values) -> dict:
    return {
        "run_id": record.run_id,
        "instance_id": record.instance_id,
        "verification_status": status,
        "max_residual": values.get("max_residual"),
        "recomputed_objective": values.get("recomputed_objective"),
    }


def _binary_value(value_: object) -> int | None:
    try:
        candidate = float(value_)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    rounded = round(candidate)
    if rounded not in {0, 1} or abs(candidate - rounded) > TOLS["int"]:
        raise ArithmeticError("fractional indicator")
    return rounded


def _consistent_binary_sources(values: list[object]) -> object | None:
    present = [value_ for value_ in values if value_ is not None]
    if not present:
        return None
    numeric = []
    for value_ in present:
        candidate = float(value_)
        if not math.isfinite(candidate):
            return None
        numeric.append(candidate)
    if max(numeric) - min(numeric) > TOLS["int"]:
        raise ValueError("conflicting representations of one binary variable")
    return present[0]


def verify_record(
    record: RunRecord, tolerance: float = 1e-5, model=None
) -> dict:
    """Rebuild and evaluate one stored solution against the original GDP."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Verification tolerance must be finite and positive")
    variables = record.solution.get("variables") if record.solution else None
    indicators = record.solution.get("indicators") if record.solution else None
    booleans = record.solution.get("booleans", {}) if record.solution else {}
    if record.mode != "solve" or not record.solution:
        return _result(record, "no_solution")
    if not isinstance(variables, dict) or not isinstance(indicators, dict):
        return _result(record, "not_verifiable")
    try:
        if model is None:
            model = BENCHMARKS[record.benchmark].build(
                BenchmarkCase(record.instance_id, record.instance_params, record.seed)
            )
        for name, stored in variables.items():
            variable = ComponentUID(name).find_component_on(model)
            if variable is None or stored is None or not math.isfinite(float(stored)):
                raise ValueError(f"missing variable {name}")
            variable.fix(float(stored))
        disjuncts = list(
            model.component_data_objects(Disjunct, active=None, descend_into=True)
        )
        indicator_owners = {
            id(disjunct.binary_indicator_var): disjunct for disjunct in disjuncts
        }
        for disjunct in disjuncts:
            if disjunct.name not in indicators:
                raise ValueError(f"missing indicator {disjunct.name}")
            stored = _consistent_binary_sources(
                [
                    indicators[disjunct.name],
                    booleans.get(disjunct.indicator_var.name),
                    variables.get(disjunct.binary_indicator_var.name),
                ]
            )
            selected = _binary_value(stored)
            if selected is None:
                raise ValueError(f"missing indicator {disjunct.name}")
            disjunct.indicator_var.fix(bool(selected))
        for boolean in model.component_data_objects(
            BooleanVar, active=None, descend_into=True
        ):
            parent = boolean.parent_block()
            if isinstance(parent, Disjunct) and boolean is parent.indicator_var:
                # A disjunct Boolean is resolved only through its indicators entry;
                # aliases were checked for consistency in the disjunct loop above.
                continue
            associated = boolean.get_associated_binary()
            indicator_owner = (
                indicator_owners.get(id(associated)) if associated is not None else None
            )
            stored = _consistent_binary_sources(
                [
                    booleans.get(boolean.name),
                    variables.get(associated.name) if associated is not None else None,
                    indicators.get(indicator_owner.name)
                    if indicator_owner is not None
                    else None,
                ]
            )
            selected = _binary_value(stored)
            if selected is None:
                raise ValueError(f"missing Boolean variable {boolean.name}")
            boolean.fix(bool(selected))
            if associated is not None and indicator_owner is None:
                associated.fix(selected)
        for variable in model.component_data_objects(Var, active=True, descend_into=True):
            variable_value = value(variable, exception=False)
            if (
                not variable.fixed
                or variable_value is None
                or not math.isfinite(float(variable_value))
            ):
                raise ValueError(f"missing finite fixed variable {variable.name}")
    except ArithmeticError:
        return _result(record, "fractional_indicators")
    except (KeyError, TypeError, ValueError, AttributeError):
        return _result(record, "not_verifiable")

    residuals = []
    for variable in model.component_data_objects(Var, active=None, descend_into=True):
        variable_value = value(variable, exception=False)
        if variable.lb is not None:
            residuals.append(max(0.0, value(variable.lb) - variable_value))
        if variable.ub is not None:
            residuals.append(max(0.0, variable_value - value(variable.ub)))
    # Block traversal does not descend into Disjunct blocks, so these are precisely
    # the active global constraints; selected-disjunct constraints are checked below.
    for constraint in model.component_data_objects(Constraint, active=True, descend_into=True):
        residuals.append(_constraint_violation(constraint))
    try:
        for disjunction in model.component_data_objects(
            Disjunction, active=True, descend_into=True
        ):
            selected = [
                disjunct
                for disjunct in disjunction.disjuncts
                if float(indicators[disjunct.name]) >= 0.5
            ]
            residuals.append(abs(sum(float(indicators[d.name]) for d in disjunction.disjuncts) - 1))
            if len(selected) != 1:
                residuals.append(1.0)
                continue
            for constraint in selected[0].component_data_objects(
                Constraint, active=True, descend_into=True
            ):
                residuals.append(_constraint_violation(constraint))
        for logical in model.component_data_objects(
            LogicalConstraint, active=True, descend_into=True
        ):
            logical_value = value(logical.body, exception=False)
            residuals.append(0.0 if logical_value is True else 1.0)
    except (KeyError, TypeError, ValueError):
        return _result(record, "not_verifiable")
    objectives = list(model.component_data_objects(Objective, active=True))
    recomputed = value(objectives[0], exception=False) if len(objectives) == 1 else None
    maximum = max(residuals, default=0.0)
    if not math.isfinite(maximum) or maximum > tolerance:
        status = "infeasible_point"
    elif not is_correct(recomputed, record.objective):
        status = "objective_mismatch"
    else:
        status = "verified_feasible"
    return _result(
        record, status, max_residual=maximum, recomputed_objective=recomputed
    )


def verify_run(
    run_directory: Path, tolerance: float = 1e-5, reverify: bool = False
) -> list[dict]:
    """Incrementally verify current solve records and rewrite ``verification.csv``."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Verification tolerance must be finite and positive")
    _, records = read_campaign(run_directory)
    destination = run_directory / "verification.csv"
    existing = {}
    if destination.exists() and not reverify:
        try:
            with destination.open(newline="") as stream:
                existing = {row["run_id"]: row for row in csv.DictReader(stream)}
        except (OSError, KeyError, csv.Error):
            existing = {}
    cache = {}
    rows = []
    for record in records:
        if record.mode != "solve":
            continue
        previous = existing.get(record.run_id)
        try:
            previous_tolerance = float(previous.get("verification_tolerance", "nan"))
        except (AttributeError, TypeError, ValueError):
            previous_tolerance = math.nan
        if (
            previous is not None
            and previous.get("record_timestamp") == record.timestamp
            and previous_tolerance == tolerance
            and previous.get("verification_status")
            in {
                "verified_feasible",
                "infeasible_point",
                "fractional_indicators",
                "objective_mismatch",
                "no_solution",
                "not_verifiable",
            }
        ):
            rows.append(previous)
            continue
        try:
            key = (record.benchmark, record.instance_id)
            if key not in cache:
                cache[key] = BENCHMARKS[record.benchmark].build(
                    BenchmarkCase(record.instance_id, record.instance_params, record.seed)
                )
            row = verify_record(record, tolerance, cache[key].clone())
        except Exception:
            row = _result(record, "not_verifiable")
        row["record_timestamp"] = record.timestamp
        row["verification_tolerance"] = tolerance
        rows.append(row)
    columns = [
        "run_id",
        "instance_id",
        "verification_status",
        "max_residual",
        "recomputed_objective",
        "record_timestamp",
        "verification_tolerance",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", dir=destination.parent, prefix=".verification.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return rows
