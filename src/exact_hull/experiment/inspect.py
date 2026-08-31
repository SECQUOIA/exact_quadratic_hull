"""Transformation and optional solver-presolve inspection."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path

from exact_hull.benchmarks import BENCHMARKS
from exact_hull.benchmarks.base import BenchmarkCase
from exact_hull.experiment.runner import load_config, transform_model


def _solver_presolve(path: Path) -> tuple[str | None, dict]:
    if importlib.util.find_spec("gurobipy") is not None:
        import gurobipy as gp

        model = gp.read(str(path))
        presolved = model.presolve()
        has_binary_square = False
        has_binary_bilinear = False
        has_epigraph = any("conic_aux_t" in variable.VarName for variable in presolved.getVars())
        has_rotated_cone = False
        for constraint in presolved.getQConstrs():
            expression = presolved.getQCRow(constraint)
            has_epigraph_term = False
            has_binary_product = False
            for index in range(expression.size()):
                variable_i = expression.getVar1(index)
                variable_j = expression.getVar2(index)
                binary_i = variable_i.VType == gp.GRB.BINARY
                binary_j = variable_j.VType == gp.GRB.BINARY
                if variable_i is variable_j and binary_i:
                    has_binary_square = True
                elif binary_i or binary_j:
                    has_binary_bilinear = True
                    has_binary_product = True
                if "conic_aux_t" in variable_i.VarName or "conic_aux_t" in variable_j.VarName:
                    has_epigraph_term = True
            has_rotated_cone = has_rotated_cone or (
                has_epigraph_term and has_binary_product
            )
        return "gurobipy", {
            "solver_presolved_num_variables": presolved.NumVars,
            "solver_presolved_num_constraints": presolved.NumConstrs,
            "solver_presolved_num_quadratic_constraints": presolved.NumQConstrs,
            "solver_survives_binary_square": has_binary_square,
            "solver_survives_binary_bilinear": has_binary_bilinear,
            "solver_survives_epigraph": has_epigraph,
            "solver_survives_rotated_cone": has_rotated_cone,
        }
    if importlib.util.find_spec("pyscipopt") is not None:
        from pyscipopt import Model

        model = Model()
        model.readProblem(str(path))
        model.presolve()
        variables = model.getVars(transformed=True)
        constraints = model.getConss(transformed=True)
        binary_names = {
            variable.name for variable in variables if variable.vtype() == "BINARY"
        }
        expression_text = []
        if hasattr(model, "getExprNonlinear"):
            for constraint in constraints:
                try:
                    expression_text.append(str(model.getExprNonlinear(constraint)))
                except Exception:  # not every constraint has a nonlinear expression
                    pass
        nonlinear_text = "\n".join(expression_text)
        has_binary_square = any(
            re.search(rf"\b{re.escape(name)}\b\s*(?:\^\s*2|\*\s*{re.escape(name)}\b)",
                      nonlinear_text)
            for name in binary_names
        )
        has_binary_bilinear = any(
            re.search(rf"\b{re.escape(name)}\b\s*\*", nonlinear_text)
            or re.search(rf"\*\s*\b{re.escape(name)}\b", nonlinear_text)
            for name in binary_names
        )
        has_epigraph = any("conic_aux_t" in variable.name for variable in variables)
        has_rotated_cone = has_epigraph and any(
            "conic_constraint" in constraint.name for constraint in constraints
        )
        return "pyscipopt", {
            "solver_presolved_num_variables": model.getNVars(transformed=True),
            "solver_presolved_num_constraints": model.getNConss(transformed=True),
            "solver_presolved_num_quadratic_constraints": None,
            "solver_survives_binary_square": has_binary_square,
            "solver_survives_binary_bilinear": has_binary_bilinear,
            "solver_survives_epigraph": has_epigraph,
            "solver_survives_rotated_cone": has_rotated_cone,
        }
    return None, {}


def inspect_config(
    config_path: Path, output_directory: Path, all_instances: bool = False
) -> tuple[list[dict], bool]:
    """Inspect each strategy on the first or every configured instance."""
    config = load_config(config_path)
    experiment = config["experiment"]
    benchmark = BENCHMARKS[experiment["benchmark"]]
    cases = benchmark.cases(config["instances"], experiment["base_seed"])
    if not all_instances:
        cases = cases[:1]
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    used_backend = False
    for strategy in config["strategies"]:
        for case in cases:
            base = {
                "instance_id": case.instance_id,
                "strategy": strategy["label"],
                "transformation": strategy["name"],
            }
            try:
                model = benchmark.build(BenchmarkCase(case.instance_id, case.params, case.seed))
                counts, _, _, _ = transform_model(
                    model, strategy["name"], strategy["options"], "solve"
                )
            except Exception as error:
                rows.append(
                    {
                        **base,
                        "transform_error": f"{type(error).__name__}: {error}",
                        "model_path": None,
                    }
                )
                continue
            path_counts = getattr(model, "_exact_hull_path_counts", {})
            stem = f"{case.instance_id}-{strategy['label']}".replace("/", "-")
            model_path = output_directory / f"{stem}.lp"
            export_error = None
            temporary_model_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=output_directory,
                    prefix=f".{stem}.",
                    suffix=".lp",
                    delete=False,
                ) as stream:
                    temporary_model_path = Path(stream.name)
                model.write(
                    str(temporary_model_path),
                    io_options={"symbolic_solver_labels": True},
                )
                os.replace(temporary_model_path, model_path)
            except Exception as error:
                export_error = f"{type(error).__name__}: {error}"
                if temporary_model_path is not None:
                    temporary_model_path.unlink(missing_ok=True)
                model_path.unlink(missing_ok=True)
            backend = None
            solver_counts = {}
            presolve_error = None
            if export_error is None:
                try:
                    backend, solver_counts = _solver_presolve(model_path)
                    used_backend = used_backend or backend is not None
                except Exception as error:  # backend APIs vary across supported versions
                    presolve_error = f"{type(error).__name__}: {error}"
            rows.append(
                {
                    **base,
                    **counts,
                    **path_counts,
                    **solver_counts,
                    "solver_backend": backend,
                    "model_path": str(model_path) if export_error is None else None,
                    "export_error": export_error,
                    "presolve_error": presolve_error,
                    "transform_error": None,
                }
            )
    json_path = output_directory / "inspection.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    csv_path = output_directory / "inspection.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows, used_backend
