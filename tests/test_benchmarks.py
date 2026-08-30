import numpy as np
from pyomo.environ import value

from exact_hull.benchmarks.cstr import BEST_KNOWN_NT5_OBJECTIVE
from exact_hull.benchmarks.cstr import build_model as build_cstr
from exact_hull.benchmarks.kmeans import BENCHMARK as kmeans_benchmark
from exact_hull.benchmarks.kmeans import build_model as build_kmeans
from exact_hull.benchmarks.random_quadratic import build_model as build_random


def test_random_quadratic_is_deterministic_and_local_rng_only():
    np.random.seed(44)
    before = np.random.get_state()
    first = build_random(seed=12)
    second = build_random(seed=12)
    different = build_random(seed=13)
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    np.testing.assert_allclose(
        [value(first.Q_objective[index]) for index in first.Q_objective],
        [value(second.Q_objective[index]) for index in second.Q_objective],
    )
    assert not np.allclose(
        [value(first.Q_objective[index]) for index in first.Q_objective],
        [value(different.Q_objective[index]) for index in different.Q_objective],
    )


def test_kmeans_seed_and_tight_distance_bound():
    first = build_kmeans(3, 2, 5, (-2, 4), seed=9)
    second = build_kmeans(3, 2, 5, (-2, 4), seed=9)
    different = build_kmeans(3, 2, 5, (-2, 4), seed=10)
    np.testing.assert_allclose(first._point_array, second._point_array)
    assert not np.allclose(first._point_array, different._point_array)
    assert first.distance[1].ub == 3 * (4 - (-2)) ** 2


def test_kmeans_campaign_has_96_instances():
    cases = kmeans_benchmark.cases(
        {
            "n_dimensions": [2, 3, 4, 5],
            "n_clusters": [3, 4, 5],
            "n_points": list(range(10, 18)),
        },
        5,
    )
    assert len(cases) == 96


def test_cstr_port_and_reference_objective_constant():
    model = build_cstr(5)
    assert len(model.N) == 5
    assert BEST_KNOWN_NT5_OBJECTIVE == 3.06181298849707
