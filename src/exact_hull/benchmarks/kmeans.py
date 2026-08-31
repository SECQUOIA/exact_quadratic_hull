"""Seeded minimum-sum-of-squares clustering GDP benchmark."""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction

from exact_hull.benchmarks.base import (
    BenchmarkCase,
    content_instance_id,
    grid_rows,
    stable_seed,
    validate_case_ids,
)


def build_model(
    n_dimensions: int,
    n_clusters: int,
    n_points: int,
    coord_range: tuple[float, float] = (-1.0, 1.0),
    seed: int | None = None,
) -> pyo.ConcreteModel:
    rng = np.random.default_rng(seed)
    points = rng.uniform(*coord_range, size=(n_points, n_dimensions))
    model = pyo.ConcreteModel(name="kmeans")
    model.dimensions = pyo.RangeSet(n_dimensions)
    model.clusters = pyo.RangeSet(n_clusters)
    model.points = pyo.RangeSet(n_points)
    model.points_coordinates = pyo.Param(
        model.points,
        model.dimensions,
        initialize=lambda _, point, dimension: float(points[point - 1, dimension - 1]),
    )
    model.center_coordinates = pyo.Var(model.clusters, model.dimensions, bounds=coord_range)
    distance_bound = n_dimensions * (coord_range[1] - coord_range[0]) ** 2
    model.distance = pyo.Var(model.points, bounds=(0, distance_bound))
    model.symmetry_breaking = pyo.Constraint(
        model.clusters,
        rule=lambda model, cluster: (
            pyo.Constraint.Skip
            if cluster == 1
            else model.center_coordinates[cluster - 1, 1] <= model.center_coordinates[cluster, 1]
        ),
    )

    def assignment_rule(disjunct, cluster, point):
        disjunct.distance = pyo.Constraint(
            expr=model.distance[point]
            >= sum(
                (
                    model.points_coordinates[point, dimension]
                    - model.center_coordinates[cluster, dimension]
                )
                ** 2
                for dimension in model.dimensions
            )
        )

    model.assignment_disjunct = Disjunct(model.clusters, model.points, rule=assignment_rule)
    model.assignment = Disjunction(
        model.points,
        rule=lambda model, point: [
            model.assignment_disjunct[cluster, point] for cluster in model.clusters
        ],
    )
    model.obj = pyo.Objective(expr=sum(model.distance.values()))
    model._point_array = points
    return model


class KMeansBenchmark:
    def cases(self, instance_config, base_seed):
        return validate_case_ids([
            BenchmarkCase(
                content_instance_id("kmeans", "kmeans", base_seed, params),
                params,
                stable_seed(base_seed, params, "kmeans"),
            )
            for params in grid_rows(instance_config)
        ])

    def build(self, case):
        params = dict(case.params)
        if "coord_range" in params:
            params["coord_range"] = tuple(params["coord_range"])
        return build_model(seed=case.seed, **params)

    def solution(self, model):
        return {
            "centers": {
                f"{cluster},{dimension}": pyo.value(model.center_coordinates[cluster, dimension])
                for cluster in model.clusters
                for dimension in model.dimensions
            }
        }


BENCHMARK = KMeansBenchmark()
