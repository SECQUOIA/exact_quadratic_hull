"""Experimental exact-hull variants retained from the original experiments."""

from pyomo.common.modeling import unique_component_name
from pyomo.environ import Constraint, Reals, Var
from pyomo.repn.standard_repn import generate_standard_repn

from exact_hull.transformations.base import ExactHullBase, registered
from exact_hull.transformations.general import quadratic_terms


class ExtraVariableExactHull(ExactHullBase):
    equality_links = True

    def _quadratic_relations(self, constraint, disjunct, substitute_map, constraint_map):
        indicator = disjunct.binary_indicator_var
        repn = generate_standard_repn(constraint.body)
        block = disjunct._transformation_block()
        if constraint.equality:
            quadratic = quadratic_terms(constraint, substitute_map)
            linear = 0
            for coefficient, variable in zip(
                repn.linear_coefs or (), repn.linear_vars or (), strict=True
            ):
                linear += coefficient * substitute_map.get(id(variable), variable)
            auxiliary = Var(domain=Reals)
            block.add_component(
                unique_component_name(block, f"_aux_t_{constraint.local_name}_eq"), auxiliary
            )
            link = Constraint(
                expr=auxiliary + linear + ((repn.constant or 0) - constraint.lower) * indicator == 0
            )
            block.add_component(
                unique_component_name(block, f"_t_def_{constraint.local_name}_eq"), link
            )
            constraint_map.transformed_constraints[constraint].append(link)
            constraint_map.src_constraint[link] = constraint
            return {"eq": quadratic == auxiliary * indicator}
        relations = {}
        for side, bound, sign in (
            ("lb", constraint.lower, -1),
            ("ub", constraint.upper, 1),
        ):
            if bound is None:
                continue
            quadratic = quadratic_terms(constraint, substitute_map, sign)
            linear = 0
            for coefficient, variable in zip(
                repn.linear_coefs or (), repn.linear_vars or (), strict=True
            ):
                linear += sign * coefficient * substitute_map.get(id(variable), variable)
            constant = sign * (repn.constant or 0)
            transformed_bound = sign * bound
            auxiliary = Var(domain=Reals)
            block.add_component(
                unique_component_name(block, f"_aux_t_{constraint.local_name}_{side}"), auxiliary
            )
            link_body = auxiliary + linear + (constant - transformed_bound) * indicator
            link = Constraint(expr=link_body == 0 if self.equality_links else link_body <= 0)
            block.add_component(
                unique_component_name(block, f"_t_def_{constraint.local_name}_{side}"), link
            )
            constraint_map.transformed_constraints[constraint].append(link)
            constraint_map.src_constraint[link] = constraint
            relation = quadratic <= auxiliary * indicator
            relations[side] = relation
        return relations


@registered("gdp.hull_exact_extra_var", "Experimental exact hull with equality-linked auxiliary")
class ExactHullExtraVar(ExtraVariableExactHull):
    pass


@registered(
    "gdp.hull_exact_extra_var_inequal",
    "Experimental exact hull with inequality-linked auxiliary",
)
class ExactHullExtraVarInequal(ExtraVariableExactHull):
    equality_links = False
