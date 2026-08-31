import numpy as np
import pytest
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


def test_random_quadratic_can_separate_objective_and_constraint_convexity():
    model = build_random(
        n_dimensions=4,
        n_disjunctions=2,
        n_disjuncts_per_disjunction=3,
        n_constraints_per_disjunct=3,
        ensure_positive_definite=False,
        objective_positive_definite=True,
        seed=19,
    )
    objective_matrix = np.array(
        [
            [value(model.Q_objective[i, j]) for j in model.dimensions]
            for i in model.dimensions
        ]
    )
    assert np.linalg.eigvalsh(objective_matrix).min() >= 0.1 - 1e-10
    assert any(
        np.linalg.eigvalsh(matrix).min() < 0
        for matrix, _, _ in model._coefficient_arrays.values()
    )


def test_kmeans_seed_and_tight_distance_bound():
    first = build_kmeans(3, 2, 5, (-2, 4), seed=9)
    second = build_kmeans(3, 2, 5, (-2, 4), seed=9)
    different = build_kmeans(3, 2, 5, (-2, 4), seed=10)
    np.testing.assert_allclose(first._point_array, second._point_array)
    assert not np.allclose(first._point_array, different._point_array)
    assert first.distance[1].ub == 3 * (4 - (-2)) ** 2


def test_new_generator_arguments_and_seed_are_keyword_only():
    with pytest.raises(TypeError):
        build_kmeans(3, 2, 5, (-2, 4), 9)
    with pytest.raises(TypeError):
        build_random(
            3,
            2,
            3,
            2,
            2,
            (-1.0, 1.0),
            (0.0, 0.01),
            (-1.0, 1.0),
            False,
            0.0,
            9,
        )


def test_kmeans_campaign_has_48_instances():
    cases = kmeans_benchmark.cases(
        {
            "n_dimensions": [2, 3, 5],
            "n_clusters": [3, 5],
            "n_points": [12, 15, 17, 20],
            "replicate": [1, 2],
        },
        5,
    )
    assert len(cases) == 48


def test_cstr_port_and_reference_objective_constant():
    model = build_cstr(5)
    assert len(model.N) == 5
    assert BEST_KNOWN_NT5_OBJECTIVE == 3.06181298849707
