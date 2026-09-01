"""Certified reference-objective derivation."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable
from pathlib import Path

from pyomo.environ import Objective, SolverFactory, TransformationFactory, value
from pyomo.gdp import Disjunction
from pyomo.opt import SolverStatus, TerminationCondition

from exact_hull.analysis.outcomes import is_correct
from exact_hull.analysis.verify import verify_run
from exact_hull.benchmarks import BENCHMARKS
from exact_hull.benchmarks.base import BenchmarkCase
from exact_hull.experiment.results import (
    VERIFIED_OPTIMAL_STATUSES,
    read_campaign,
    write_json_atomic,
)
from exact_hull.experiment.runner import _load_solution, _versions
from exact_hull.experiment.solvers import options_for


def _solve_fixed_selection(model, time_limit: float = 300) -> float | str | None:
    TransformationFactory("core.logical_to_linear").apply_to(model)
    TransformationFactory("gdp.fix_disjuncts").apply_to(model)
    options = options_for("scip", time_limit)
    options = [
        "option optcr=1e-8;" if option.startswith("option optcr=") else
        "option optca=1e-10;" if option.startswith("option optca=") else
        "limits/gap = 1e-8" if option.startswith("limits/gap =") else
        "limits/absgap = 1e-10" if option.startswith("limits/absgap =") else option
        for option in options
    ]
    result = SolverFactory("gams").solve(
        model,
        solver="scip",
        add_options=options,
        tee=False,
        load_solutions=False,
    )
    if (
        result.solver.status == SolverStatus.ok
        and result.solver.termination_condition == TerminationCondition.infeasible
    ):
        return "infeasible"
    if result.solver.status != SolverStatus.ok or result.solver.termination_condition not in {
        TerminationCondition.optimal,
        TerminationCondition.globallyOptimal,
    }:
        return None
    if not _load_solution(model, result):
        return None
    objective = next(model.component_data_objects(Objective, active=True))
    candidate = value(objective, exception=False)
    return float(candidate) if candidate is not None and math.isfinite(candidate) else None


def enumerate_reference(
    benchmark_name: str,
    case: BenchmarkCase,
    cap: int = 4096,
    solve_selection: Callable[[object], float | str | None] = _solve_fixed_selection,
) -> tuple[float | None, dict]:
    """Enumerate disjunct selections when their Cartesian product fits the cap."""
    prototype = BENCHMARKS[benchmark_name].build(case)
    disjunctions = list(prototype.component_data_objects(Disjunction, active=True))
    sizes = [len(disjunction.disjuncts) for disjunction in disjunctions]
    selection_count = math.prod(sizes)
    if selection_count > cap:
        return None, {"route": "enumeration", "selection_count": selection_count, "skipped": True}
    best = math.inf
    solved = 0
    failures = 0
    errors = []
    for selection in itertools.product(*(range(size) for size in sizes)):
        model = BENCHMARKS[benchmark_name].build(case)
        current = list(model.component_data_objects(Disjunction, active=True))
        for disjunction, selected in zip(current, selection, strict=True):
            for index, disjunct in enumerate(disjunction.disjuncts):
                disjunct.indicator_var.fix(index == selected)
        try:
            candidate = solve_selection(model)
        except Exception as error:
            failures += 1
            errors.append(f"{type(error).__name__}: {error}")
            continue
        if candidate == "infeasible":
            solved += 1
        elif isinstance(candidate, int | float) and math.isfinite(candidate):
            solved += 1
            best = min(best, candidate)
    provenance = {"route": "enumeration", "selection_count": selection_count, "solved": solved}
    if failures:
        provenance["failures"] = failures
        provenance["errors"] = errors
    return ((best if solved == selection_count and math.isfinite(best) else None), provenance)


def derive_references(
    run_directory: Path,
    cap: int = 4096,
    solve_selection: Callable[[object], float | str | None] = _solve_fixed_selection,
    reverify: bool = False,
    tolerance: float = 1e-5,
) -> dict:
    """Apply enumeration first, then verified cross-subsolver agreement."""
    manifest, records = read_campaign(run_directory)
    verification_rows = verify_run(
        run_directory, tolerance=tolerance, reverify=reverify
    )
    verified = {row["run_id"]: row["verification_status"] for row in verification_rows}
    benchmark_name = manifest["config"]["experiment"]["benchmark"]
    destination = run_directory / "references.json"
    if destination.exists():
        try:
            entries = json.loads(destination.read_text()).get("references", {})
        except (OSError, json.JSONDecodeError, TypeError):
            entries = {}
    else:
        entries = {}
    for instance in manifest["instances"]:
        case = BenchmarkCase(instance["instance_id"], instance["params"], instance["seed"])
        existing = entries.get(case.instance_id, {})
        existing_route = existing.get("provenance", {}).get("route")
        if existing.get("status") == "certified" and existing_route == "enumeration":
            continue
        objective, provenance = enumerate_reference(
            benchmark_name, case, cap=cap, solve_selection=solve_selection
        )
        if objective is None:
            feasible_records = [
                record
                for record in records
                if record.instance_id == case.instance_id
                and record.mode == "solve"
                and record.variant != "convex"
                and verified.get(record.run_id) == "verified_feasible"
                and record.objective is not None
                and math.isfinite(record.objective)
            ]
            candidates = [
                record
                for record in feasible_records
                if record.status in VERIFIED_OPTIMAL_STATUSES
            ]
            pairs = [
                (left, right)
                for left, right in itertools.combinations(candidates, 2)
                if left.subsolver != right.subsolver
                and is_correct(
                    max(left.objective, right.objective),
                    min(left.objective, right.objective),
                )
            ]
            pair = min(
                pairs,
                key=lambda item: (
                    min(item[0].objective, item[1].objective),
                    tuple(sorted((item[0].run_id, item[1].run_id))),
                ),
                default=None,
            )
            if pair is not None:
                pair = tuple(sorted(pair, key=lambda record: record.run_id))
                agreed_objective = min(pair[0].objective, pair[1].objective)
                provenance = {
                    "route": "agreement",
                    "run_ids": [pair[0].run_id, pair[1].run_id],
                    "subsolvers": [pair[0].subsolver, pair[1].subsolver],
                }
                conflict = min(
                    (
                        record
                        for record in feasible_records
                        if record.objective < agreed_objective
                        and not is_correct(record.objective, agreed_objective)
                    ),
                    key=lambda record: (record.objective, record.run_id),
                    default=None,
                )
                if conflict is None:
                    objective = agreed_objective
                else:
                    provenance["conflict"] = {
                        "run_id": conflict.run_id,
                        "objective": conflict.objective,
                        "agreed_objective": agreed_objective,
                    }
        if provenance.get("route") == "enumeration" and not provenance.get("skipped"):
            provenance.update(
                {
                    "time_limit_sec": 300,
                    "relative_gap_tolerance": 1e-8,
                    "absolute_gap_tolerance": 1e-10,
                }
            )
        entries[case.instance_id] = {
            "status": "certified" if objective is not None else "reference_unknown",
            "objective": objective,
            "provenance": provenance,
        }
        payload = {"schema_version": 1, "versions": _versions(), "references": entries}
        write_json_atomic(payload, destination)
    payload = {"schema_version": 1, "versions": _versions(), "references": entries}
    write_json_atomic(payload, destination)
    return payload
