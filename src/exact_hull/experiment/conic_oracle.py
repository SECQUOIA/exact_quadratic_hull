"""Independent direct-QCP CEHR relaxation oracle."""

from __future__ import annotations

import csv
import importlib.util
import math
import tempfile
from pathlib import Path

from pyomo.environ import Var

from exact_hull.benchmarks import BENCHMARKS
from exact_hull.experiment.results import write_json_atomic
from exact_hull.experiment.runner import load_config, transform_model


def build_oracle_model(benchmark_name: str, case):
    """Build a binary-free CEHR relaxation and a backend-independent descriptor."""
    model = BENCHMARKS[benchmark_name].build(case)
    counts, _, _, _ = transform_model(
        model, "gdp.hull_exact_conic_original", {}, "relaxation"
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


def _finite(candidate) -> float | None:
    try:
        result = float(candidate)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _solve_gurobi(model) -> dict:
    import gurobipy as gp

    with tempfile.TemporaryDirectory(prefix="exact-hull-conic-oracle-") as temporary:
        model_path = Path(temporary) / "oracle.mps"
        model.write(str(model_path), io_options={"symbolic_solver_labels": True})
        oracle = gp.read(str(model_path))
        if oracle.NumIntVars:
            raise RuntimeError("Oracle export unexpectedly contains integer variables")
        oracle.setParam("TimeLimit", 600)
        oracle.setParam("Threads", 1)
        oracle.optimize()
        status_names = {
            getattr(gp.GRB, name): name
            for name in (
                "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF",
                "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT", "SOLUTION_LIMIT",
                "INTERRUPTED", "NUMERIC", "SUBOPTIMAL", "USER_OBJ_LIMIT",
            )
            if hasattr(gp.GRB, name)
        }
        primal = _finite(oracle.ObjVal) if oracle.SolCount else None
        try:
            dual = _finite(oracle.ObjBound)
        except Exception:
            dual = None
        return {
            "status": status_names.get(oracle.Status, f"status_{oracle.Status}"),
            "optimal": oracle.Status == gp.GRB.OPTIMAL and primal is not None,
            "primal_objective": primal,
            "dual_bound": dual,
            "runtime_sec": _finite(oracle.Runtime),
        }


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
    if benchmark_name == "cstr":
        raise ValueError("conic-bound requires a convex-family configuration")
    cases = BENCHMARKS[benchmark_name].cases(
        config["instances"], config["experiment"]["base_seed"]
    )
    if benchmark_name == "random_quadratic" and any(
        case.params["ensure_positive_definite"] is not True
        or case.params["objective_positive_definite"] is not True
        for case in cases
    ):
        raise ValueError("conic-bound requires a convex-family configuration")
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    if importlib.util.find_spec("gurobipy") is None:
        raise RuntimeError("The conic oracle requires gurobipy; install gurobipy")
    for case in cases:
        descriptor = {}
        backend = None
        result = {}
        bound = None
        try:
            model, descriptor = build_oracle_model(benchmark_name, case)
            if descriptor.get("n_fallback_rows", 0) > 0:
                status = (
                    "refused: "
                    f"n_fallback_rows={descriptor['n_fallback_rows']}; "
                    "oracle requires a pure factorized CEHR relaxation"
                )
            else:
                backend = "gurobipy"
                result = _solve_gurobi(model)
                status = result["status"]
                if result["optimal"]:
                    bound = result["primal_objective"]
        except Exception as error:
            status = f"{type(error).__name__}: {error}"
        rows.append(
            {
                "instance_id": case.instance_id,
                "oracle_bound": bound,
                "primal_objective": result.get("primal_objective"),
                "dual_bound": result.get("dual_bound"),
                "runtime_sec": result.get("runtime_sec"),
                "oracle_status": status,
                "backend": backend,
                **descriptor,
            }
        )
        _write_rows(rows, output_directory)
    return rows
