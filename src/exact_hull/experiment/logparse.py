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
