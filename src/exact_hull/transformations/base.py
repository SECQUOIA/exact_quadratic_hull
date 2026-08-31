"""Shared control flow for exact quadratic hull transformations."""

from __future__ import annotations

import logging
from abc import abstractmethod

import pyomo.core.expr as EXPR
from pyomo.core import TransformationFactory
from pyomo.core.base.block import Block
from pyomo.gdp.plugins.hull import Hull_Reformulation
from pyomo.gdp.util import clone_without_expression_components

logger = logging.getLogger("pyomo.gdp.hull")


def increment_path_count(model, name: str, amount: int = 1) -> None:
    counts = getattr(model, "_exact_hull_path_counts", None)
    if counts is None:
        counts = {
            "n_cone_rows": 0,
            "n_fallback_rows": 0,
            "n_equality_fallback_rows": 0,
            "n_epigraph_vars": 0,
        }
        model._exact_hull_path_counts = counts
    counts[name] += amount


class ExactHullBase(Hull_Reformulation):
    """Pyomo hull transformation with an exact-quadratic emitter hook."""

    @abstractmethod
    def _quadratic_relations(self, constraint, disjunct, substitute_map, constraint_map):
        """Return relational expressions keyed by ``eq``, ``lb``, or ``ub``."""

    def _add_mapped_constraint(
        self, container, name, original_index, side, relation, source, constraint_map
    ):
        key = (
            (name, original_index, side) if source.parent_component().is_indexed() else (name, side)
        )
        container.add(key, relation)
        transformed = container[key]
        constraint_map.transformed_constraints[source].append(transformed)
        constraint_map.src_constraint[transformed] = source

    def _transform_constraint(self, obj, disjunct, var_substitute_map, zero_substitute_map):
        relaxation_block = disjunct._transformation_block()
        # Pyomo 6.10 guards private_data() by the caller's module, so an external
        # subclass must initialize the same private namespace directly.
        if relaxation_block._private_data is None:
            relaxation_block._private_data = {}
        constraint_map = relaxation_block._private_data.setdefault(
            "pyomo.gdp", Block._private_data_initializers["pyomo.gdp"]()
        )
        transformed = relaxation_block.transformedConstraints

        # Keep the source index distinct from all emitter loop variables. This fixes
        # the indexed-constraint shadowing bug in the original forks.
        for original_index in sorted(obj.keys()):
            constraint = obj[original_index]
            if not constraint.active:
                continue
            name = f"{constraint.local_name}_{len(transformed)}"
            degree = constraint.body.polynomial_degree()
            if degree == 2:
                relations = self._quadratic_relations(
                    constraint, disjunct, var_substitute_map, constraint_map
                )
                for side in ("eq", "lb", "ub"):
                    if side in relations:
                        self._add_mapped_constraint(
                            transformed,
                            name,
                            original_index,
                            side,
                            relations[side],
                            constraint,
                            constraint_map,
                        )
                continue

            nonlinear = degree not in (0, 1)
            eps = self._config.EPS
            mode = self._config.perspective_function
            if not nonlinear or mode == "FurmanSawayaGrossmann":
                value_at_zero = clone_without_expression_components(
                    constraint.body, substitute=zero_substitute_map
                )
            indicator = disjunct.binary_indicator_var
            if nonlinear:
                if mode == "LeeGrossmann":
                    sub_expr = clone_without_expression_components(
                        constraint.body,
                        substitute={
                            var: sub / indicator for var, sub in var_substitute_map.items()
                        },
                    )
                    expression = sub_expr * indicator
                elif mode == "GrossmannLee":
                    sub_expr = clone_without_expression_components(
                        constraint.body,
                        substitute={
                            var: sub / (indicator + eps) for var, sub in var_substitute_map.items()
                        },
                    )
                    expression = (indicator + eps) * sub_expr
                elif mode == "FurmanSawayaGrossmann":
                    denominator = (1 - eps) * indicator + eps
                    sub_expr = clone_without_expression_components(
                        constraint.body,
                        substitute={
                            var: sub / denominator for var, sub in var_substitute_map.items()
                        },
                    )
                    expression = denominator * sub_expr - eps * value_at_zero * (1 - indicator)
                else:
                    raise RuntimeError(f"Unknown nonlinear hull mode: {mode}")
            else:
                expression = clone_without_expression_components(
                    constraint.body, substitute=var_substitute_map
                )

            if constraint.equality:
                if nonlinear:
                    relation = expression == constraint.lower * indicator
                else:
                    variables = list(EXPR.identify_variables(expression))
                    if len(variables) == 1 and not constraint.lower:
                        variables[0].fix(0)
                        constraint_map.transformed_constraints[constraint].append(variables[0])
                        constraint_map.src_constraint[variables[0]] = constraint
                        continue
                    relation = (
                        expression - (1 - indicator) * value_at_zero == constraint.lower * indicator
                    )
                self._add_mapped_constraint(
                    transformed,
                    name,
                    original_index,
                    "eq",
                    relation,
                    constraint,
                    constraint_map,
                )
                continue

            if constraint.lower is not None:
                if self._generate_debug_messages:
                    logger.debug("GDP(Hull): Transforming constraint '%s'", constraint.name)
                relation = (
                    expression >= constraint.lower * indicator
                    if nonlinear
                    else expression - (1 - indicator) * value_at_zero
                    >= constraint.lower * indicator
                )
                self._add_mapped_constraint(
                    transformed,
                    name,
                    original_index,
                    "lb",
                    relation,
                    constraint,
                    constraint_map,
                )
            if constraint.upper is not None:
                relation = (
                    expression <= constraint.upper * indicator
                    if nonlinear
                    else expression - (1 - indicator) * value_at_zero
                    <= constraint.upper * indicator
                )
                self._add_mapped_constraint(
                    transformed,
                    name,
                    original_index,
                    "ub",
                    relation,
                    constraint,
                    constraint_map,
                )
        obj.deactivate()


def registered(name: str, doc: str):
    """Register only package-owned names, never Pyomo's deprecated ``gdp.chull``."""

    return TransformationFactory.register(name, doc=doc)
