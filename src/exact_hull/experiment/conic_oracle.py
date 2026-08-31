"""Independent direct-QCP CEHR relaxation oracle."""

from __future__ import annotations

import csv
import importlib.util
import math
import tempfile
from pathlib import Path

from pyomo.environ import Objective, SolverFactory, Var, value
from pyomo.opt import SolverStatus, TerminationCondition

from exact_hull.benchmarks import BENCHMARKS
from exact_hull.experiment.results import write_json_atomic
from exact_hull.experiment.runner import load_config, transform_model


def build_oracle_model(benchmark_name: str, case):
    """Build a binary-free CEHR relaxation and a backend-independent descriptor."""
    model = BENCHMARKS[benchmark_name].build(case)
    counts, _, _, _ = transform_model(
        model, "gdp.hull_exact_conic_no_cholesky", {}, "relaxation"
    )
    path_counts = getattr(model, "_exact_hull_path_counts", {})
    binary_count = sum(
        variable.is_binary() for variable in model.component_data_objects(Var, descend_into=True)
    )
    descriptor = {
        **counts,
        **path_counts,
        "n_binary_variables": binary_count,
        "has_rotated_cone_structure": path_counts.get("n_cone_rows", 0) > 0,
    }
    return model, descriptor


def _solve_gurobi(model) -> tuple[float | None, str]:
    import gurobipy as gp

    with tempfile.TemporaryDirectory(prefix="exact-hull-conic-oracle-") as temporary:
        model_path = Path(temporary) / "oracle.mps"
        model.write(str(model_path), io_options={"symbolic_solver_labels": True})
        oracle = gp.read(str(model_path))
        if oracle.NumIntVars:
            raise RuntimeError("Oracle export unexpectedly contains integer variables")
        # Deliberately do not set NonConvex: acceptance is part of the recognition datum.
        oracle.optimize()
        if oracle.Status != gp.GRB.OPTIMAL:
            status_names = {
                getattr(gp.GRB, name): name
                for name in (
                    "LOADED", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF",
                    "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT", "SOLUTION_LIMIT",
                    "INTERRUPTED", "NUMERIC", "SUBOPTIMAL", "USER_OBJ_LIMIT",
                )
                if hasattr(gp.GRB, name)
            }
            return None, status_names.get(oracle.Status, f"status_{oracle.Status}")
        candidate = float(oracle.ObjVal)
        return (candidate if math.isfinite(candidate) else None), "OPTIMAL"


def _solve_mosek(model) -> tuple[float | None, str]:
    solver = SolverFactory("mosek_direct")
    result = solver.solve(model, tee=False)
    status = result.solver.status
    termination = result.solver.termination_condition
    if status != SolverStatus.ok or termination not in {
        TerminationCondition.optimal,
        TerminationCondition.globallyOptimal,
    }:
        return None, f"{status}/{termination}"
    candidate = getattr(result.problem, "lower_bound", None)
    try:
        bound = float(candidate)
    except (TypeError, ValueError):
        bound = None
    if bound is None or not math.isfinite(bound):
        bound = None
    if bound is None:
        objective = next(model.component_data_objects(Objective, active=True), None)
        candidate = value(objective, exception=False) if objective is not None else None
        try:
            bound = float(candidate)
        except (TypeError, ValueError):
            bound = None
        if bound is not None and not math.isfinite(bound):
            bound = None
    return bound, str(termination)


def _write_rows(rows: list[dict], output_directory: Path) -> None:
    write_json_atomic(rows, output_directory / "conic-bounds.json")
    fieldnames = sorted({key for row in rows for key in row}) or ["instance_id"]
    with (output_directory / "conic-bounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def conic_bounds(config_path: Path, output_directory: Path) -> list[dict]:
    """Solve each instance's CEHR relaxation through a direct solver API."""
    config = load_config(config_path)
    benchmark_name = config["experiment"]["benchmark"]
    if benchmark_name == "cstr" or (
        benchmark_name == "random_quadratic"
        and config["instances"].get("ensure_positive_definite") is not True
    ):
        raise ValueError("conic-bound requires a convex-family configuration")
    cases = BENCHMARKS[benchmark_name].cases(
        config["instances"], config["experiment"]["base_seed"]
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    gurobi_available = importlib.util.find_spec("gurobipy") is not None
    mosek_available = importlib.util.find_spec("mosek") is not None
    backend = "gurobipy" if gurobi_available else "mosek" if mosek_available else None
    if backend is None:
        raise RuntimeError(
            "No independent conic backend is available; install gurobipy or Pyomo+MOSEK"
        )
    for case in cases:
        descriptor = {}
        bound = None
        try:
            model, descriptor = build_oracle_model(benchmark_name, case)
            if backend == "gurobipy":
                bound, status = _solve_gurobi(model)
            else:
                bound, status = _solve_mosek(model)
        except Exception as error:
            status = f"{type(error).__name__}: {error}"
        rows.append(
            {
                "instance_id": case.instance_id,
                "oracle_bound": bound,
                "oracle_status": status,
                "backend": backend,
                **descriptor,
            }
        )
        _write_rows(rows, output_directory)
    return rows
