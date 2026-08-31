"""Conic exact quadratic hull reformulations (CEHR)."""

from __future__ import annotations

import numpy as np
from pyomo.common.modeling import unique_component_name
from pyomo.environ import Constraint, NonNegativeReals, Var, sqrt
from pyomo.repn.standard_repn import generate_standard_repn

from exact_hull.transformations.base import ExactHullBase, increment_path_count, registered
from exact_hull.transformations.general import homogeneous_quadratic, quadratic_terms

EIGENVALUE_RELATIVE_TOL = 1e-9


def _add_auxiliary(block, source, constraint_map, stem, component):
    name = unique_component_name(block, f"_{stem}_{source.local_name}")
    block.add_component(name, component)
    if isinstance(component, Constraint):
        constraint_map.transformed_constraints[source].append(component)
        constraint_map.src_constraint[component] = source
    return component


def _quadratic_data(constraint, substitute_map, sign):
    repn = generate_standard_repn(constraint.body)
    variables = []
    seen = set()
    for var_i, var_j in repn.quadratic_vars:
        for variable in (var_i, var_j):
            if id(variable) not in seen:
                variables.append(variable)
                seen.add(id(variable))
    positions = {id(variable): position for position, variable in enumerate(variables)}
    matrix = np.zeros((len(variables), len(variables)))
    for coefficient, (var_i, var_j) in zip(repn.quadratic_coefs, repn.quadratic_vars, strict=True):
        row, column = positions[id(var_i)], positions[id(var_j)]
        if row == column:
            matrix[row, column] += sign * float(coefficient)
        else:
            matrix[row, column] += sign * float(coefficient) / 2
            matrix[column, row] += sign * float(coefficient) / 2
    disaggregated = [substitute_map.get(id(variable), variable) for variable in variables]
    return repn, matrix, disaggregated


class ConicExactHullBase(ExactHullBase):
    """Use a conic representation when a quadratic inequality is convex."""

    representation = "factorized"

    def _quadratic_relations(self, constraint, disjunct, substitute_map, constraint_map):
        indicator = disjunct.binary_indicator_var
        if constraint.equality:
            increment_path_count(constraint.model(), "n_fallback_rows")
            increment_path_count(constraint.model(), "n_equality_fallback_rows")
            expression = homogeneous_quadratic(constraint, indicator, substitute_map)
            return {"eq": expression == constraint.lower * indicator**2}

        relations = {}
        for side, bound, sign in (
            ("lb", constraint.lower, -1),
            ("ub", constraint.upper, 1),
        ):
            if bound is None:
                continue
            repn, matrix, variables = _quadratic_data(constraint, substitute_map, sign)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            scale = max(1.0, abs(float(eigenvalues[-1]))) if len(eigenvalues) else 1.0
            if len(eigenvalues) and float(eigenvalues[0]) < -EIGENVALUE_RELATIVE_TOL * scale:
                increment_path_count(constraint.model(), "n_fallback_rows")
                expression = homogeneous_quadratic(constraint, indicator, substitute_map)
                relations[side] = (
                    expression >= bound * indicator**2
                    if side == "lb"
                    else expression <= bound * indicator**2
                )
                continue

            block = disjunct._transformation_block()
            auxiliary = _add_auxiliary(
                block,
                constraint,
                constraint_map,
                f"conic_aux_t_{side}",
                Var(domain=NonNegativeReals),
            )
            increment_path_count(constraint.model(), "n_epigraph_vars")
            increment_path_count(constraint.model(), "n_cone_rows")
            linear = auxiliary
            for coefficient, variable in zip(
                repn.linear_coefs or (), repn.linear_vars or (), strict=True
            ):
                linear += sign * coefficient * substitute_map.get(id(variable), variable)
            if repn.constant:
                linear += sign * repn.constant * indicator
            relations[side] = linear <= sign * bound * indicator

            positive = eigenvalues > 1e-10
            factor = eigenvectors[:, positive] @ np.diag(np.sqrt(eigenvalues[positive]))
            factor_expressions = [
                sum(factor[row, column] * variables[row] for row in range(len(variables)))
                for column in range(factor.shape[1])
            ]
            if self.representation == "no_cholesky":
                cone_quadratic = quadratic_terms(constraint, substitute_map, sign)
                cone_relation = cone_quadratic <= auxiliary * indicator
            elif self.representation == "factorized":
                cone_relation = (
                    sum(value**2 for value in factor_expressions) <= auxiliary * indicator
                )
            elif self.representation in {"no_sqrt", "sqrt"}:
                left = sum((2 * value) ** 2 for value in factor_expressions)
                left += (auxiliary - indicator) ** 2
                right = auxiliary + indicator
                cone_relation = (
                    left <= right**2 if self.representation == "no_sqrt" else sqrt(left) <= right
                )
            else:
                z = _add_auxiliary(
                    block,
                    constraint,
                    constraint_map,
                    f"conic_aux_z_{side}",
                    Var(range(len(factor_expressions))),
                )
                w = _add_auxiliary(block, constraint, constraint_map, f"conic_aux_w_{side}", Var())
                s = _add_auxiliary(
                    block,
                    constraint,
                    constraint_map,
                    f"conic_aux_s_{side}",
                    Var(domain=NonNegativeReals),
                )
                for position, value in enumerate(factor_expressions):
                    _add_auxiliary(
                        block,
                        constraint,
                        constraint_map,
                        f"conic_z_def_{side}_{position}",
                        Constraint(expr=z[position] == 2 * value),
                    )
                _add_auxiliary(
                    block,
                    constraint,
                    constraint_map,
                    f"conic_w_def_{side}",
                    Constraint(expr=w == auxiliary - indicator),
                )
                _add_auxiliary(
                    block,
                    constraint,
                    constraint_map,
                    f"conic_s_def_{side}",
                    Constraint(expr=s == auxiliary + indicator),
                )
                left = sum(z[position] ** 2 for position in range(len(factor_expressions))) + w**2
                cone_relation = left <= s**2 if self.representation == "extra" else sqrt(left) <= s
            _add_auxiliary(
                block,
                constraint,
                constraint_map,
                f"conic_constraint_{side}",
                Constraint(expr=cone_relation),
            )
        return relations


@registered(
    "gdp.hull_exact_conic_no_cholesky",
    "CEHR without matrix factorization (default conic formulation)",
)
class ExactHullConicNoCholesky(ConicExactHullBase):
    representation = "no_cholesky"


@registered("gdp.hull_exact_conic_original", "CEHR using only a factorized rotated cone")
class ExactHullConicOriginal(ConicExactHullBase):
    representation = "factorized"


@registered("gdp.hull_exact_conic_no_sqrt_no_extra_var", "CEHR squared SOC without extra variables")
class ExactHullConicNoSqrtNoExtraVar(ConicExactHullBase):
    representation = "no_sqrt"


@registered("gdp.hull_exact_conic_no_sqrt_extra_var", "CEHR squared SOC with extra variables")
class ExactHullConicNoSqrtExtraVar(ConicExactHullBase):
    representation = "extra"


@registered(
    "gdp.hull_exact_conic_sqrt_no_extra_var", "CEHR square-root SOC without extra variables"
)
class ExactHullConicSqrtNoExtraVar(ConicExactHullBase):
    representation = "sqrt"


@registered("gdp.hull_exact_conic_sqrt_extra_var", "CEHR square-root SOC with extra variables")
class ExactHullConicSqrtExtraVar(ConicExactHullBase):
    representation = "sqrt_extra"
