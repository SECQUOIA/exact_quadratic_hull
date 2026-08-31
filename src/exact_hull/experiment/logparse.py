"""Solver-specific parsers for GAMS output logs."""

from __future__ import annotations

import re

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _last(pattern: str, text: str, flags: int = re.IGNORECASE) -> float | None:
    matches = re.findall(pattern, text, flags=flags)
    return float(matches[-1]) if matches else None


def parse_solver_bounds(text: str, subsolver: str) -> tuple[float | None, float | None]:
    """Extract final lower and upper bounds using solver-specific summary lines."""
    subsolver = subsolver.lower()
    if subsolver == "gurobi":
        matches = re.findall(
            rf"best objective[ \t]+({_NUMBER}|-),[ \t]*best bound[ \t]+({_NUMBER}|-)",
            text,
            flags=re.IGNORECASE,
        )
        if not matches:
            return None, None
        upper, lower = matches[-1]
        return (
            float(lower) if lower != "-" else None,
            float(upper) if upper != "-" else None,
        )
    if subsolver == "scip":
        lower = _last(rf"Dual Bound[ \t]*:[ \t]*({_NUMBER})", text)
        upper = _last(rf"Primal Bound[ \t]*:[ \t]*({_NUMBER})", text)
        return lower, upper
    raise ValueError(f"Unsupported log parser subsolver: {subsolver}")


def solver_timed_out(text: str, subsolver: str) -> bool:
    """Detect a time limit from the selected solver's emitted log text."""
    patterns = {
        "gurobi": r"\bTime limit reached\b",
        "scip": r"\btime limit reached\b",
    }
    try:
        pattern = patterns[subsolver.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported log parser subsolver: {subsolver}") from error
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def parse_solver_metadata(text: str, subsolver: str) -> dict[str, str | int | None]:
    """Extract solver version and available post-presolve model statistics."""
    metadata: dict[str, str | int | None] = {
        "version": None,
        "presolved_num_variables": None,
        "presolved_num_constraints": None,
        "presolved_num_nonzeros": None,
        "presolved_num_soc": None,
        "presolved_num_bilinear": None,
        "presolved_num_quadratic": None,
        "presolved_num_nonlinear": None,
    }
    subsolver = subsolver.lower()
    if subsolver == "gurobi":
        version = re.findall(r"Gurobi Optimizer version\s+([^\s]+)", text, re.IGNORECASE)
        if version:
            metadata["version"] = version[-1]
        presolved = re.findall(
            r"Presolved:\s*(\d+) rows,\s*(\d+) columns,\s*(\d+) nonzeros",
            text,
            re.IGNORECASE,
        )
        if presolved:
            rows, columns, nonzeros = presolved[-1]
            metadata.update(
                presolved_num_constraints=int(rows),
                presolved_num_variables=int(columns),
                presolved_num_nonzeros=int(nonzeros),
            )
        patterns = {
            "presolved_num_soc": r"Presolved model has\s+(\d+) second-order cone constraints?",
            "presolved_num_bilinear": (
                r"Presolved model has\s+(\d+) bilinear constraint(?:\(s\)|s)?"
            ),
            "presolved_num_quadratic": (
                r"Presolved model has\s+(\d+) quadratic constraint(?:\(s\)|s)?"
            ),
        }
    elif subsolver == "scip":
        version = re.findall(r"SCIP version\s+([^\s]+)", text, re.IGNORECASE)
        if version:
            metadata["version"] = version[-1]
        presolved = re.findall(
            r"presolved problem has\s+(\d+) variables[\s\S]*?and\s+(\d+) constraints",
            text,
            re.IGNORECASE,
        )
        if presolved:
            variables, constraints = presolved[-1]
            metadata.update(
                presolved_num_variables=int(variables),
                presolved_num_constraints=int(constraints),
            )
        patterns = {
            "presolved_num_soc": r"(\d+) constraints? of type <(?:soc|secondordercone)>",
            "presolved_num_bilinear": r"(\d+) constraints? of type <bilinear>",
            "presolved_num_quadratic": r"(\d+) constraints? of type <quadratic>",
            "presolved_num_nonlinear": r"(\d+) constraints? of type <nonlinear>",
        }
    else:
        raise ValueError(f"Unsupported log parser subsolver: {subsolver}")
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            metadata[key] = int(matches[-1])
    return metadata
