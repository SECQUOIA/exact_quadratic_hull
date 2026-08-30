"""Benchmark registry."""

from exact_hull.benchmarks.clay import BENCHMARK as clay
from exact_hull.benchmarks.cstr import BENCHMARK as cstr
from exact_hull.benchmarks.kmeans import BENCHMARK as kmeans
from exact_hull.benchmarks.random_quadratic import BENCHMARK as random_quadratic

BENCHMARKS = {
    "random_quadratic": random_quadratic,
    "kmeans": kmeans,
    "clay": clay,
    "cstr": cstr,
}

INSTANCE_PARAMETERS = {
    "random_quadratic": {
        "n_dimensions",
        "n_disjunctions",
        "n_disjuncts_per_disjunction",
        "n_constraints_per_disjunct",
        "n_feasible_regions",
        "coeff_range",
        "constraint_margin",
        "x_range",
        "ensure_positive_definite",
        "sparsity_factor",
    },
    "kmeans": {"n_dimensions", "n_clusters", "n_points", "coord_range"},
    "cstr": {"NT"},
    "clay": {"instance", "metric"},
}


__all__ = ["BENCHMARKS", "INSTANCE_PARAMETERS"]
