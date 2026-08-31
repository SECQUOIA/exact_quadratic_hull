from __future__ import annotations

import inspect

import pytest
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    TransformationFactory,
    Var,
    value,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.plugins.hull import Hull_Reformulation
from pyomo.util.calc_var_value import calculate_variable_from_constraint

import exact_hull


def quadratic_gdp(indices=(3, 7)):
    model = ConcreteModel()
    model.x = Var(bounds=(-2, 2))
    model.disjunct = Disjunct([1, 2])
    for disjunct in model.disjunct.values():
        disjunct.quadratic = Constraint(
            indices, rule=lambda _, index: model.x**2 + index * model.x <= 10
        )
    model.choice = Disjunction(expr=[model.disjunct[1], model.disjunct[2]])
    return model


@pytest.mark.parametrize("name", list(exact_hull.TRANSFORMATIONS))
def test_registered_transformations_preserve_original_indices(name):
    model = quadratic_gdp()
    transformation = TransformationFactory(name)
    transformation.apply_to(model)
    disaggregated = transformation.get_disaggregated_var(model.x, model.disjunct[1])
    assert disaggregated is not model.x
    for original_index in (3, 7):
        mapped = transformation.get_transformed_constraints(
            model.disjunct[1].quadratic[original_index]
        )
        assert mapped
        indexed = [item for item in mapped if item.index() is not None]
        assert indexed
        assert all(item.index()[-2] == original_index for item in indexed)


def test_pyomo_private_method_canary():
    assert hasattr(Hull_Reformulation, "_transform_constraint")
    assert list(inspect.signature(Hull_Reformulation._transform_constraint).parameters) == [
        "self",
        "obj",
        "disjunct",
        "var_substitute_map",
        "zero_substitute_map",
    ]


def test_import_does_not_replace_pyomo_chull_alias():
    assert type(TransformationFactory("gdp.chull")).__module__ == "pyomo.gdp.plugins.hull"


def _feasible(constraint, tolerance=1e-8):
    body = value(constraint.body)
    return (constraint.lower is None or body >= value(constraint.lower) - tolerance) and (
        constraint.upper is None or body <= value(constraint.upper) + tolerance
    )


def _set_conic_auxiliaries(block, point, indicator):
    for variable in block.component_data_objects(Var):
        name = variable.name
        if "conic_aux_t" in name:
            variable.set_value(point**2)
        elif "conic_aux_z" in name:
            variable.set_value(2 * point)
        elif "conic_aux_w" in name:
            variable.set_value(point**2 - indicator)
        elif "conic_aux_s" in name:
            variable.set_value(point**2 + indicator)


@pytest.mark.parametrize(
    "name",
    ["gdp.hull_exact", *[name for name in exact_hull.TRANSFORMATIONS if "conic" in name]],
)
@pytest.mark.parametrize(("point", "expected"), [(0.5, True), (1.5, False)])
def test_quadratic_representations_have_same_fixed_indicator_feasible_set(name, point, expected):
    model = ConcreteModel()
    model.x = Var(bounds=(-2, 2))
    model.disjunct = Disjunct([1, 2])
    for disjunct in model.disjunct.values():
        disjunct.quadratic = Constraint(expr=model.x**2 <= 1)
    model.choice = Disjunction(expr=[model.disjunct[1], model.disjunct[2]])
    transformation = TransformationFactory(name)
    transformation.apply_to(model)
    model.x.set_value(point)
    for number, disjunct in model.disjunct.items():
        disjunct.binary_indicator_var.set_value(1 if number == 1 else 0)
        transformation.get_disaggregated_var(model.x, disjunct).set_value(
            point if number == 1 else 0
        )
    for number, block in model._pyomo_gdp_hull_reformulation.relaxedDisjuncts.items():
        _set_conic_auxiliaries(block, point if number == 0 else 0, 1 if number == 0 else 0)
    constraints = list(model.component_data_objects(Constraint, active=True))
    assert all(_feasible(item) for item in constraints) is expected


def test_fixing_indicator_recovers_original_constraint():
    model = quadratic_gdp(indices=(1,))
    transformation = TransformationFactory("gdp.hull_exact")
    transformation.apply_to(model)
    model.disjunct[1].binary_indicator_var.fix(1)
    model.disjunct[2].binary_indicator_var.fix(0)
    model.x.set_value(1)
    transformation.get_disaggregated_var(model.x, model.disjunct[1]).set_value(1)
    transformation.get_disaggregated_var(model.x, model.disjunct[2]).set_value(0)
    transformed = transformation.get_transformed_constraints(model.disjunct[1].quadratic[1])
    assert all(_feasible(item) for item in transformed)


def _rich_quadratic_gdp():
    model = ConcreteModel()
    model.x = Var([1, 2], bounds=(-3, 3))
    model.disjunct = Disjunct([1, 2])
    for disjunct in model.disjunct.values():
        quadratic = (
            model.x[1] ** 2
            + 2 * model.x[1] * model.x[2]
            + 2 * model.x[2] ** 2
            + model.x[1]
            - 2 * model.x[2]
            + 0.5
        )
        disjunct.upper = Constraint(expr=quadratic <= 2)
        disjunct.lower = Constraint(expr=-quadratic >= -2)
        disjunct.ranged = Constraint(expr=(-1, quadratic, 2))
        disjunct.equality = Constraint(expr=quadratic == 0.5)
    model.choice = Disjunction(expr=[model.disjunct[1], model.disjunct[2]])
    return model


def _set_rich_point(model, transformation, point):
    model.x[1].set_value(point[0])
    model.x[2].set_value(point[1])
    for weight, disjunct in zip((0.4, 0.6), model.disjunct.values(), strict=True):
        disjunct.binary_indicator_var.set_value(weight, skip_validation=True)
        for index in (1, 2):
            transformation.get_disaggregated_var(model.x[index], disjunct).set_value(
                weight * point[index - 1]
            )
        block = disjunct._transformation_block()
        quadratic = weight * (point[0] ** 2 + 2 * point[0] * point[1] + 2 * point[1] ** 2)
        for variable in block.component_data_objects(Var):
            if "conic_aux_t" in variable.name:
                variable.set_value(quadratic)
        for marker in ("conic_z_def", "conic_w_def", "conic_s_def"):
            for constraint in block.component_data_objects(Constraint, active=True):
                if marker not in constraint.name:
                    continue
                auxiliary = next(
                    variable
                    for variable in identify_variables(constraint.body)
                    if f"conic_aux_{marker.removeprefix('conic_')[0]}" in variable.name
                )
                calculate_variable_from_constraint(auxiliary, constraint)


def _rich_feasibility(name, point):
    model = _rich_quadratic_gdp()
    transformation = TransformationFactory(name)
    transformation.apply_to(model)
    _set_rich_point(model, transformation, point)
    return {
        source.local_name: all(
            _feasible(constraint)
            for constraint in transformation.get_transformed_constraints(source)
        )
        for source in model.disjunct[1].component_data_objects(Constraint)
    }


@pytest.mark.parametrize("point", [(0, 0), (-2, 1.5), (1, 1)])
def test_all_quadratic_emitters_agree_at_fractional_indicators(point):
    names = ["gdp.hull_exact", *[name for name in exact_hull.TRANSFORMATIONS if "conic" in name]]
    results = [_rich_feasibility(name, point) for name in names]
    assert all(result == results[0] for result in results[1:])


def test_ranged_constraint_lower_side_uses_correct_sign():
    result = _rich_feasibility("gdp.hull_exact", (-2, 1.5))
    assert result["upper"]
    assert not result["ranged"]


@pytest.mark.parametrize(
    "name",
    ["gdp.hull_exact_conic_no_sqrt_extra_var", "gdp.hull_exact_conic_sqrt_extra_var"],
)
def test_extra_variable_soc_radius_is_nonnegative(name):
    model = _rich_quadratic_gdp()
    TransformationFactory(name).apply_to(model)
    radii = [
        variable for variable in model.component_data_objects(Var) if "conic_aux_s" in variable.name
    ]
    assert radii
    assert all(variable.domain is NonNegativeReals for variable in radii)


@pytest.mark.parametrize("name", ["gdp.hull_exact_extra_var", "gdp.hull_exact_extra_var_inequal"])
def test_experimental_equality_retains_lifted_t_form(name):
    model = _rich_quadratic_gdp()
    transformation = TransformationFactory(name)
    transformation.apply_to(model)
    block = model.disjunct[1]._transformation_block()
    auxiliaries = [
        variable
        for variable in block.component_data_objects(Var)
        if "_aux_t_equality_eq" in variable.name
    ]
    assert len(auxiliaries) == 1
    mapped = transformation.get_transformed_constraints(model.disjunct[1].equality)
    assert len(mapped) == 2
    assert any(auxiliaries[0].name in str(constraint.expr) for constraint in mapped)


@pytest.mark.parametrize(
    "name",
    ["gdp.hull_exact", "gdp.hull_exact_extra_var", "gdp.hull_exact_extra_var_inequal"],
)
def test_diagonal_inequality_terms_use_power_expression(name):
    model = quadratic_gdp(indices=(1,))
    transformation = TransformationFactory(name)
    transformation.apply_to(model)
    transformed = transformation.get_transformed_constraints(model.disjunct[1].quadratic[1])
    assert any("**2" in str(constraint.body) for constraint in transformed)


def test_cehr_records_cone_fallback_equality_and_epigraph_paths():
    model = ConcreteModel()
    model.x = Var(bounds=(-2, 2))
    model.disjunct = Disjunct([1, 2])
    for disjunct in model.disjunct.values():
        disjunct.convex = Constraint(expr=model.x**2 <= 1)
        disjunct.nonconvex = Constraint(expr=-(model.x**2) <= 1)
        disjunct.equality = Constraint(expr=model.x**2 == 1)
    model.choice = Disjunction(expr=list(model.disjunct.values()))
    TransformationFactory("gdp.hull_exact_conic_no_cholesky").apply_to(model)
    assert model._exact_hull_path_counts == {
        "n_cone_rows": 2,
        "n_fallback_rows": 4,
        "n_equality_fallback_rows": 2,
        "n_epigraph_vars": 2,
    }


def test_cehr_eigenvalue_test_is_relative_to_the_largest_eigenvalue():
    model = ConcreteModel()
    model.x = Var([1, 2], bounds=(-2, 2))
    model.disjunct = Disjunct([1, 2])
    for disjunct in model.disjunct.values():
        disjunct.nearly_psd = Constraint(
            expr=1e6 * model.x[1] ** 2 - 5e-4 * model.x[2] ** 2 <= 1
        )
    model.choice = Disjunction(expr=list(model.disjunct.values()))
    TransformationFactory("gdp.hull_exact_conic_no_cholesky").apply_to(model)
    assert model._exact_hull_path_counts["n_cone_rows"] == 2
    assert model._exact_hull_path_counts["n_fallback_rows"] == 0
