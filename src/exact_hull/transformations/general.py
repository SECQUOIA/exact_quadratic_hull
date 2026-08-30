"""General exact quadratic hull reformulation (GEHR)."""

from pyomo.repn.standard_repn import generate_standard_repn

from exact_hull.transformations.base import ExactHullBase, registered


def homogeneous_quadratic(constraint, indicator, substitute_map, sign=1):
    """Return ``sign * (v'Qv + c'v*y + d*y**2)``."""
    repn = generate_standard_repn(constraint.body)
    if not repn.is_quadratic():
        raise RuntimeError("Recognized quadratic constraint, but representation is not quadratic")
    expression = 0
    for coefficient, (var_i, var_j) in zip(repn.quadratic_coefs, repn.quadratic_vars, strict=True):
        disaggregated_i = substitute_map.get(id(var_i), var_i)
        disaggregated_j = substitute_map.get(id(var_j), var_j)
        if var_i is var_j:
            expression += sign * coefficient * disaggregated_i**2
        else:
            expression += sign * coefficient * disaggregated_i * disaggregated_j
    for coefficient, variable in zip(repn.linear_coefs or (), repn.linear_vars or (), strict=True):
        disaggregated = substitute_map.get(id(variable), variable)
        expression += sign * coefficient * disaggregated * indicator
    if repn.constant:
        expression += sign * repn.constant * indicator**2
    return expression


def quadratic_terms(constraint, substitute_map, sign=1):
    """Return only ``sign * v'Qv``, preserving powers for diagonal terms."""
    repn = generate_standard_repn(constraint.body)
    expression = 0
    for coefficient, (var_i, var_j) in zip(repn.quadratic_coefs, repn.quadratic_vars, strict=True):
        disaggregated_i = substitute_map.get(id(var_i), var_i)
        disaggregated_j = substitute_map.get(id(var_j), var_j)
        expression += (
            sign * coefficient * disaggregated_i**2
            if var_i is var_j
            else sign * coefficient * disaggregated_i * disaggregated_j
        )
    return expression


@registered("gdp.hull_exact", "Exact quadratic hull reformulation (GEHR)")
class ExactHull(ExactHullBase):
    def _quadratic_relations(self, constraint, disjunct, substitute_map, constraint_map):
        del constraint_map
        indicator = disjunct.binary_indicator_var
        expression = homogeneous_quadratic(constraint, indicator, substitute_map)
        if constraint.equality:
            return {"eq": expression == constraint.lower * indicator**2}
        relations = {}
        if constraint.lower is not None:
            relations["lb"] = expression >= constraint.lower * indicator**2
        if constraint.upper is not None:
            relations["ub"] = expression <= constraint.upper * indicator**2
        return relations
