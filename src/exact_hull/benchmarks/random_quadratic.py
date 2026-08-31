"""Seeded random quadratic GDP benchmark."""

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


def generate_quadratic_function(
    n_dimensions: int,
    coeff_range: tuple[float, float],
    ensure_positive_definite: bool = False,
    sparsity_factor: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate coefficients without modifying NumPy's process-global RNG."""
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(*coeff_range, size=(n_dimensions, n_dimensions))
    matrix = (matrix + matrix.T) / 2
    if sparsity_factor:
        # Preserve the original generator's sparsity-factor convention.
        mask = np.triu(rng.random((n_dimensions, n_dimensions)) > (1 - sparsity_factor))
        matrix *= np.logical_or(mask, mask.T)
    if ensure_positive_definite:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues += max(0.0, 0.1 - float(eigenvalues.min()))
        matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    linear = rng.uniform(*coeff_range, size=n_dimensions)
    constant = float(rng.uniform(*coeff_range))
    return matrix, linear, constant


def build_model(
    n_dimensions: int = 3,
    n_disjunctions: int = 2,
    n_disjuncts_per_disjunction: int = 3,
    n_constraints_per_disjunct: int = 2,
    n_feasible_regions: int = 2,
    coeff_range: tuple[float, float] = (-1.0, 1.0),
    constraint_margin: tuple[float, float] = (0.0, 0.01),
    x_range: tuple[float, float] = (-1.0, 1.0),
    ensure_positive_definite: bool = False,
    sparsity_factor: float = 0.0,
    *,
    objective_positive_definite: bool | None = None,
    replicate: int = 1,
    seed: int | None = None,
) -> pyo.ConcreteModel:
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate <= 0:
        raise ValueError("replicate must be a positive integer")
    if n_feasible_regions > n_disjuncts_per_disjunction:
        raise ValueError("n_feasible_regions cannot exceed disjuncts per disjunction")
    rng = np.random.default_rng(seed)
    model = pyo.ConcreteModel(name="random_quadratic")
    model.dimensions = pyo.RangeSet(n_dimensions)
    model.disjunctions = pyo.RangeSet(n_disjunctions)
    model.disjuncts = pyo.RangeSet(n_disjuncts_per_disjunction)
    model.x = pyo.Var(model.dimensions, bounds=x_range)
    objective_q, objective_c, objective_d = generate_quadratic_function(
        n_dimensions,
        coeff_range,
        (
            ensure_positive_definite
            if objective_positive_definite is None
            else objective_positive_definite
        ),
        sparsity_factor,
        int(rng.integers(0, 2**63)),
    )
    model.Q_objective = pyo.Param(
        model.dimensions,
        model.dimensions,
        initialize=lambda _, i, j: float(objective_q[i - 1, j - 1]),
    )
    model.c_objective = pyo.Param(
        model.dimensions, initialize=lambda _, i: float(objective_c[i - 1])
    )
    model.d_objective = pyo.Param(initialize=objective_d)
    model.obj = pyo.Objective(
        expr=sum(
            model.Q_objective[i, j] * model.x[i] * model.x[j]
            for i in model.dimensions
            for j in model.dimensions
        )
        + sum(model.c_objective[i] * model.x[i] for i in model.dimensions)
        + model.d_objective
    )
    feasible_points = rng.uniform(*x_range, size=(n_feasible_regions, n_dimensions))
    coefficient_store = {}

    def disjunct_rule(disjunct, disjunction_index, disjunct_index):
        def constraint_rule(_, constraint_index):
            matrix, linear, constant = generate_quadratic_function(
                n_dimensions,
                coeff_range,
                ensure_positive_definite,
                sparsity_factor,
                int(rng.integers(0, 2**63)),
            )
            if disjunct_index <= n_feasible_regions:
                point = feasible_points[disjunct_index - 1]
                value = float(point @ matrix @ point + linear @ point + constant)
                constant -= value + float(rng.uniform(*constraint_margin))
            coefficient_store[disjunction_index, disjunct_index, constraint_index] = (
                matrix,
                linear,
                constant,
            )
            return (
                sum(
                    float(matrix[i - 1, j - 1]) * model.x[i] * model.x[j]
                    for i in model.dimensions
                    for j in model.dimensions
                )
                + sum(float(linear[i - 1]) * model.x[i] for i in model.dimensions)
                + constant
                <= 0
            )

        disjunct.constraints = pyo.Constraint(
            pyo.RangeSet(n_constraints_per_disjunct), rule=constraint_rule
        )

    model.disjunct_blocks = Disjunct(model.disjunctions, model.disjuncts, rule=disjunct_rule)
    model.choice = Disjunction(
        model.disjunctions,
        rule=lambda model, index: [
            model.disjunct_blocks[index, disjunct] for disjunct in model.disjuncts
        ],
    )
    model._coefficient_arrays = coefficient_store
    return model


class RandomQuadraticBenchmark:
    def cases(self, instance_config, base_seed):
        cases = []
        for params in grid_rows(instance_config):
            params.setdefault("replicate", 1)
            if params.get("objective_positive_definite") is None:
                params["objective_positive_definite"] = params.get(
                    "ensure_positive_definite", False
                )
            seed = stable_seed(base_seed, params, "random_quadratic")
            instance_id = content_instance_id("rq", "random_quadratic", base_seed, params)
            cases.append(BenchmarkCase(instance_id, params, seed))
        return validate_case_ids(cases)

    def build(self, case):
        return build_model(seed=case.seed, **case.params)

    def solution(self, model):
        return {"x": {str(index): pyo.value(model.x[index]) for index in model.x}}


BENCHMARK = RandomQuadraticBenchmark()
