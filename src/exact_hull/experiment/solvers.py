"""GAMS solver settings shared by every benchmark."""

from __future__ import annotations

from collections.abc import Callable

TOLS = {
    "rel_gap": 1e-6,
    "abs_gap": 1e-10,
    "feas": 1e-6,
    "opt": 1e-6,
    "int": 1e-5,
}


def _common(time_limit: float) -> list[str]:
    return [
        f"option reslim={time_limit:g};",
        "option threads=1;",
        f"option optcr={TOLS['rel_gap']:g};",
        f"option optca={TOLS['abs_gap']:g};",
    ]


def gurobi_options(time_limit: float, variant: str | None = None) -> list[str]:
    if variant is not None:
        raise ValueError(f"Unknown Gurobi variant: {variant}")
    return _common(time_limit) + [
        "$onecho > gurobi.opt",
        "NonConvex 2",
        "Threads 1",
        f"MIPGap {TOLS['rel_gap']:g}",
        f"MIPGapAbs {TOLS['abs_gap']:g}",
        f"FeasibilityTol {TOLS['feas']:g}",
        f"OptimalityTol {TOLS['opt']:g}",
        f"IntFeasTol {TOLS['int']:g}",
        "$offecho",
        "GAMS_MODEL.optfile=1;",
    ]


def scip_options(time_limit: float, variant: str | None = None) -> list[str]:
    if variant not in {None, "convex"}:
        raise ValueError(f"Unknown SCIP variant: {variant}")
    options = _common(time_limit) + [
        "$onecho > scip.opt",
        f"limits/time = {time_limit:g}",
        "parallel/maxnthreads = 1",
        f"limits/gap = {TOLS['rel_gap']:g}",
        f"limits/absgap = {TOLS['abs_gap']:g}",
        f"numerics/feastol = {TOLS['feas']:g}",
        f"numerics/dualfeastol = {TOLS['opt']:g}",
        f"numerics/sumepsilon = {TOLS['feas']:g}",
        "display/verblevel = 4",
    ]
    if variant == "convex":
        options.append("constraints/nonlinear/assumeconvex = TRUE")
    return options + ["$offecho", "GAMS_MODEL.optfile=1;"]


OPTION_BUILDERS: dict[str, Callable[[float, str | None], list[str]]] = {
    "gurobi": gurobi_options,
    "scip": scip_options,
}


def options_for(
    subsolver: str,
    time_limit: float,
    variant: str | None = None,
    mode: str = "solve",
) -> list[str]:
    if mode not in {"solve", "root", "relaxation"}:
        raise ValueError(f"Unknown job mode: {mode}")
    try:
        options = OPTION_BUILDERS[subsolver](time_limit, variant)
    except KeyError as error:
        raise ValueError(f"Unsupported GAMS subsolver: {subsolver}") from error
    if mode != "root":
        return options
    insertion = options.index("$offecho")
    if subsolver == "gurobi":
        # Empirically, NodeLimit=1 lets Gurobi finish node 0 (including root
        # cuts) while reporting one explored node and no processed child.
        root_options = ["NodeLimit 1"]
    else:
        root_options = [
            "limits/nodes = 1",
            "limits/totalnodes = 1",
            "limits/restarts = 0",
        ]
    return options[:insertion] + root_options + options[insertion:]
