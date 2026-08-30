"""Solver-specific parsers for GAMS output logs."""

from __future__ import annotations

import re

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _last(pattern: str, text: str, flags: int = re.IGNORECASE) -> float | None:
    matches = re.findall(pattern, text, flags=flags)
    return float(matches[-1]) if matches else None


def _baron_root(text: str) -> float | None:
    header = re.search(
        r"^\s*Iteration\s+Time[^\n]*Lower bound\s+Upper bound\s+Progress\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if header:
        for line in text[header.end() :].splitlines():
            if not line.strip() or set(line.strip()) <= {"-", "="}:
                continue
            line = line.lstrip()
            if line.startswith("*"):
                line = line[1:].lstrip()
            fields = line.split()
            numeric_fields = []
            for field in fields:
                try:
                    numeric_fields.append(float(field.rstrip("+")))
                except ValueError:
                    pass
            if len(numeric_fields) >= 3:
                return numeric_fields[2]
            break
    if re.search(r"Problem solved during preprocessing", text, flags=re.IGNORECASE):
        value = _last(rf"Best possible[ \t]*=[ \t]*({_NUMBER})", text)
        if value is not None:
            return value
        return _last(rf"Lower bound is[ \t]+({_NUMBER})", text)
    return None


def _gurobi_root(text: str) -> float | None:
    value = _last(rf"Root relaxation:[ \t]*objective[ \t]+({_NUMBER})", text)
    if value is not None:
        return value
    if re.search(r"Root relaxation:[ \t]*cutoff", text, flags=re.IGNORECASE):
        table = re.search(
            r"Nodes[ \t]+\|[ \t]+Current Node[ \t]+\|[ \t]+Objective Bounds",
            text,
            flags=re.IGNORECASE,
        )
        if table:
            for line in text[table.end() :].splitlines():
                if not re.match(r"^\s+\d+\s+\d+", line):
                    continue
                fields = line.replace("|", " ").split()
                numbers = []
                for field in fields:
                    try:
                        numbers.append(float(field.rstrip("%")))
                    except ValueError:
                        pass
                if len(numbers) >= 2:
                    return numbers[-2]
                break
    if re.search(r"Explored[ \t]+\d+[ \t]+nodes?", text, flags=re.IGNORECASE):
        value = _last(rf"best bound[ \t]+({_NUMBER})", text)
        if value is not None:
            return value
    return _last(rf"best objective[ \t]+{_NUMBER},[ \t]*best bound[ \t]+({_NUMBER})", text)


def _scip_root(text: str) -> float | None:
    if re.search(r"problem is solved by trivial preprocessing", text, flags=re.IGNORECASE):
        value = _last(rf"objective value[ \t]*:[ \t]*({_NUMBER})", text)
        if value is not None:
            return value
    value = _last(
        rf"LP0[ \t]+\(\d+r,[ \t]*\d+c\)[ \t]*:[ \t]*"
        rf"(?:opt\.|infeas\.|unbounded)[ \t]*\[({_NUMBER})",
        text,
    )
    if value is not None:
        return value
    match = re.search(
        rf"root node[\s\S]*?dual bound[ \t]*:[ \t]*({_NUMBER})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return _last(rf"Dual Bound[ \t]*:[ \t]*({_NUMBER})", text)


def parse_root_relaxation(text: str, subsolver: str) -> float | None:
    """Extract the root relaxation using the selected solver's real grammar."""
    parser = {
        "gurobi": _gurobi_root,
        "baron": _baron_root,
        "scip": _scip_root,
    }.get(subsolver.lower())
    if parser is None:
        raise ValueError(f"Unsupported log parser subsolver: {subsolver}")
    return parser(text)


def _baron_bounds(text: str) -> tuple[float | None, float | None]:
    lower = _last(rf"Best possible[ \t]*=[ \t]*({_NUMBER})", text)
    if lower is None:
        lower = _last(rf"Lower bound is[ \t]+({_NUMBER})", text)
    if lower is None:
        lower = _baron_root(text)
    upper = _last(rf"Best solution[ \t]*=[ \t]*({_NUMBER})", text)
    if upper is None:
        upper = _last(rf"Solution[ \t]*=[ \t]*({_NUMBER})[ \t]+found at node", text)
    return lower, upper


def parse_solver_bounds(text: str, subsolver: str) -> tuple[float | None, float | None]:
    """Extract final lower and upper bounds using solver-specific summary lines."""
    subsolver = subsolver.lower()
    if subsolver == "gurobi":
        matches = re.findall(
            rf"best objective[ \t]+({_NUMBER}),[ \t]*best bound[ \t]+({_NUMBER})",
            text,
            flags=re.IGNORECASE,
        )
        if not matches:
            return None, None
        upper, lower = matches[-1]
        return float(lower), float(upper)
    if subsolver == "baron":
        return _baron_bounds(text)
    if subsolver == "scip":
        lower = _last(rf"Dual Bound[ \t]*:[ \t]*({_NUMBER})", text)
        upper = _last(rf"Primal Bound[ \t]*:[ \t]*({_NUMBER})", text)
        return lower, upper
    raise ValueError(f"Unsupported log parser subsolver: {subsolver}")


def parse_gams_solvestat(text: str) -> int | None:
    """Return the numeric GAMS SOLVER STATUS code when present in output."""
    match = re.search(
        r"^\s*\*{0,4}\s*SOLVER STATUS\s+(\d+)", text, flags=re.IGNORECASE | re.MULTILINE
    )
    return int(match.group(1)) if match else None


def solver_timed_out(text: str, subsolver: str) -> bool:
    """Detect a time limit from the selected solver's emitted log text."""
    patterns = {
        "gurobi": r"\bTime limit reached\b",
        "scip": r"\btime limit reached\b",
        "baron": r"Max\. allowable time exceeded",
    }
    try:
        pattern = patterns[subsolver.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported log parser subsolver: {subsolver}") from error
    return re.search(pattern, text, flags=re.IGNORECASE) is not None
