"""Register all exact-hull transformations on import."""

from exact_hull.transformations.conic import (
    ExactHullConicNoCholesky,
    ExactHullConicNoSqrtExtraVar,
    ExactHullConicNoSqrtNoExtraVar,
    ExactHullConicOriginal,
    ExactHullConicSqrtExtraVar,
    ExactHullConicSqrtNoExtraVar,
)
from exact_hull.transformations.experimental import ExactHullExtraVar, ExactHullExtraVarInequal
from exact_hull.transformations.general import ExactHull

TRANSFORMATIONS = {
    "gdp.hull_exact": ExactHull,
    "gdp.hull_exact_conic_no_cholesky": ExactHullConicNoCholesky,
    "gdp.hull_exact_conic_original": ExactHullConicOriginal,
    "gdp.hull_exact_conic_no_sqrt_no_extra_var": ExactHullConicNoSqrtNoExtraVar,
    "gdp.hull_exact_conic_no_sqrt_extra_var": ExactHullConicNoSqrtExtraVar,
    "gdp.hull_exact_conic_sqrt_no_extra_var": ExactHullConicSqrtNoExtraVar,
    "gdp.hull_exact_conic_sqrt_extra_var": ExactHullConicSqrtExtraVar,
    "gdp.hull_exact_extra_var": ExactHullExtraVar,
    "gdp.hull_exact_extra_var_inequal": ExactHullExtraVarInequal,
}

__all__ = ["TRANSFORMATIONS"]
