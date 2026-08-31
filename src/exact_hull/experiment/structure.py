"""Transform-time structural statistics for Pyomo models."""

from __future__ import annotations

import numbers
from collections import Counter
from dataclasses import dataclass, field

from pyomo.environ import Constraint, Var

Factor = tuple[str, bool]
Monomial = tuple[Factor, ...]
CoefficientMap = dict[Monomial, float]
BoundaryKey = tuple[object, ...]
BoundaryMap = dict[BoundaryKey, CoefficientMap]
StructuralKey = tuple[BoundaryKey | None, Monomial]
StructuralMap = dict[StructuralKey, float]
_CANCELLATION_TOLERANCE = 1e-12


@dataclass
class _Terms:
    polynomial: CoefficientMap = field(default_factory=dict)
    # Quotient numerators combine only when their canonical denominators match.
    boundary: BoundaryMap = field(default_factory=dict)
    denominators: BoundaryMap = field(default_factory=dict)
    # Collected terms are beneath an opaque denominator or intrinsic boundary.
    collected: CoefficientMap = field(default_factory=dict)


def _canonical(factors: tuple[Factor, ...]) -> Monomial:
    return tuple(sorted(factors))


def _add(target: CoefficientMap, source: CoefficientMap, scale: float = 1.0) -> None:
    for monomial, coefficient in source.items():
        merged = target.get(monomial, 0.0) + scale * coefficient
        if abs(merged) <= _CANCELLATION_TOLERANCE:
            target.pop(monomial, None)
        else:
            target[monomial] = merged


def _scaled(source: CoefficientMap, scale: float) -> CoefficientMap:
    result: CoefficientMap = {}
    _add(result, source, scale)
    return result


def _add_boundaries(
    target: BoundaryMap, source: BoundaryMap, scale: float = 1.0
) -> None:
    for key, monomials in source.items():
        destination = target.setdefault(key, {})
        _add(destination, monomials, scale)
        if not destination:
            target.pop(key)


def _scaled_boundaries(source: BoundaryMap, scale: float) -> BoundaryMap:
    result: BoundaryMap = {}
    _add_boundaries(result, source, scale)
    return result


def _merge_denominators(target: BoundaryMap, source: BoundaryMap) -> None:
    for key, monomials in source.items():
        target.setdefault(key, dict(monomials))


def _multiply(left: CoefficientMap, right: CoefficientMap) -> CoefficientMap:
    result: CoefficientMap = {}
    for left_term, left_coefficient in left.items():
        for right_term, right_coefficient in right.items():
            _add(
                result,
                {_canonical(left_term + right_term): left_coefficient * right_coefficient},
            )
    return result


def _power(source: CoefficientMap, exponent: int) -> CoefficientMap:
    result: CoefficientMap = {(): 1.0}
    for _ in range(exponent):
        result = _multiply(result, source)
    return result


def _constant_value(expression) -> float | None:
    if isinstance(expression, numbers.Number):
        return float(expression)
    if hasattr(expression, "is_constant") and expression.is_constant():
        candidate = getattr(expression, "value", None)
        if isinstance(candidate, numbers.Number):
            return float(candidate)
    return None


def _constant_integer(expression) -> int | None:
    candidate = _constant_value(expression)
    if candidate is not None and candidate.is_integer():
        return int(candidate)
    return None


def _expression_key(expression) -> BoundaryKey:
    constant = _constant_value(expression)
    if constant is not None:
        return ("constant", constant)
    if expression.is_variable_type():
        return ("variable", expression.name)
    if not expression.is_expression_type():
        return (expression.__class__.__name__, str(expression))
    name = expression.__class__.__name__.removeprefix("NPV_")
    arguments = tuple(_expression_key(argument) for argument in expression.args)
    if name in {"SumExpression", "LinearExpression", "ProductExpression"}:
        arguments = tuple(sorted(arguments, key=repr))
    if name == "UnaryFunctionExpression":
        return (name, expression.getname(), *arguments)
    return (name, *arguments)


def _same_coefficients(left: CoefficientMap, right: CoefficientMap) -> bool:
    return set(left) == set(right) and all(
        abs(left[key] - right[key]) <= _CANCELLATION_TOLERANCE for key in left
    )


def _scalar(terms: _Terms) -> float | None:
    if terms.boundary or terms.collected or set(terms.polynomial) - {()}:
        return None
    return terms.polynomial.get((), 0.0)


def _merge_terms(arguments) -> _Terms:
    result = _Terms()
    for argument in arguments:
        terms = _structural_terms(argument)
        _add(result.polynomial, terms.polynomial)
        _add_boundaries(result.boundary, terms.boundary)
        _merge_denominators(result.denominators, terms.denominators)
        _add(result.collected, terms.collected)
    return result


def _multiply_boundary(
    monomials: CoefficientMap,
    polynomial: CoefficientMap,
    denominator: CoefficientMap,
) -> CoefficientMap:
    scalar = polynomial.get(()) if set(polynomial) <= {()} else None
    if scalar is not None:
        return _scaled(monomials, scalar)
    if _same_coefficients(polynomial, denominator):
        variable_terms = {
            monomial: coefficient
            for monomial, coefficient in denominator.items()
            if monomial
        }
        if variable_terms:
            result: CoefficientMap = {}
            for monomial, coefficient in monomials.items():
                if len(monomial) < 2:
                    _add(result, {monomial: coefficient})
                else:
                    _add(
                        result,
                        _multiply({monomial: coefficient}, variable_terms),
                    )
            return result
    return dict(monomials)


def _product(left: _Terms, right: _Terms) -> _Terms:
    polynomial = _multiply(left.polynomial, right.polynomial)
    boundary: BoundaryMap = {}
    denominators: BoundaryMap = {}
    collected: CoefficientMap = {}
    left_scalar = _scalar(left)
    right_scalar = _scalar(right)
    for key in left.boundary.keys() | right.boundary.keys():
        if key in left.boundary and key in right.boundary:
            boundary[key] = _multiply(left.boundary[key], right.boundary[key])
        elif key in left.boundary:
            boundary[key] = _multiply_boundary(
                left.boundary[key], right.polynomial, left.denominators[key]
            )
        else:
            boundary[key] = _multiply_boundary(
                right.boundary[key], left.polynomial, right.denominators[key]
            )
    _merge_denominators(denominators, left.denominators)
    _merge_denominators(denominators, right.denominators)
    _add(collected, left.collected, right_scalar if right_scalar is not None else 1.0)
    _add(collected, right.collected, left_scalar if left_scalar is not None else 1.0)
    return _Terms(polynomial, boundary, denominators, collected)


def _structural_terms(expression) -> _Terms:
    """Return coefficient-aware monomials without probing variable values."""
    constant = _constant_value(expression)
    if constant is not None:
        return _Terms({(): constant})
    if expression is None:
        return _Terms({(): 0.0})
    if expression.is_variable_type():
        return _Terms({((expression.name, expression.is_binary()),): 1.0})
    if not expression.is_potentially_variable():
        return _Terms({(): 1.0})
    if not expression.is_expression_type():
        return _Terms()
    name = expression.__class__.__name__
    if name in {"SumExpression", "LinearExpression", "NPV_SumExpression"}:
        return _merge_terms(expression.args)
    if name in {"ProductExpression", "MonomialTermExpression", "NPV_ProductExpression"}:
        result = _Terms({(): 1.0})
        for argument in expression.args:
            result = _product(result, _structural_terms(argument))
        return result
    if name in {"NegationExpression", "NPV_NegationExpression"}:
        terms = _structural_terms(expression.args[0])
        return _Terms(
            _scaled(terms.polynomial, -1.0),
            _scaled_boundaries(terms.boundary, -1.0),
            dict(terms.denominators),
            _scaled(terms.collected, -1.0),
        )
    if name in {"PowExpression", "NPV_PowExpression"}:
        exponent = _constant_integer(expression.args[1])
        base = _structural_terms(expression.args[0])
        if exponent is None or exponent < 0 or exponent > 4:
            collected: CoefficientMap = {}
            _add(collected, base.polynomial)
            for monomials in base.boundary.values():
                _add(collected, monomials)
            _add(collected, base.collected)
            return _Terms(collected=collected)
        return _Terms(
            polynomial=_power(base.polynomial, exponent),
            boundary={
                key: _power(monomials, exponent)
                for key, monomials in base.boundary.items()
            },
            denominators=dict(base.denominators),
            collected=dict(base.collected),
        )
    if name in {"DivisionExpression", "NPV_DivisionExpression"}:
        numerator = _structural_terms(expression.args[0])
        denominator = expression.args[1]
        if (
            isinstance(denominator, numbers.Number)
            or not denominator.is_potentially_variable()
        ):
            divisor = _constant_value(denominator)
            scale = 1.0 if divisor in {None, 0.0} else 1.0 / divisor
            return _Terms(
                _scaled(numerator.polynomial, scale),
                _scaled_boundaries(numerator.boundary, scale),
                dict(numerator.denominators),
                _scaled(numerator.collected, scale),
            )
        key = _expression_key(denominator)
        boundary = {
            boundary_key: dict(monomials)
            for boundary_key, monomials in numerator.boundary.items()
        }
        _add(boundary.setdefault(key, {}), numerator.polynomial)
        denominator_terms = _structural_terms(denominator)
        denominators = dict(numerator.denominators)
        denominators[key] = denominator_terms.polynomial
        return _Terms(
            boundary=boundary,
            denominators=denominators,
            collected=dict(numerator.collected),
        )
    # Intrinsic nonlinear functions are opaque: collect their contained polynomial
    # terms, but never multiply them by factors outside the function boundary.
    contained = _merge_terms(expression.args)
    collected: CoefficientMap = {}
    _add(collected, contained.polynomial)
    for monomials in contained.boundary.values():
        _add(collected, monomials)
    _add(collected, contained.collected)
    return _Terms(collected=collected)


def _add_structural(
    target: StructuralMap,
    boundary: BoundaryKey | None,
    source: CoefficientMap,
) -> None:
    for monomial, coefficient in source.items():
        key = (boundary, monomial)
        merged = target.get(key, 0.0) + coefficient
        if abs(merged) <= _CANCELLATION_TOLERANCE:
            target.pop(key, None)
        else:
            target[key] = merged


def _structural_monomials(expression) -> StructuralMap:
    terms = _structural_terms(expression)
    result: StructuralMap = {}
    _add_structural(result, None, terms.polynomial)
    for boundary, monomials in terms.boundary.items():
        _add_structural(result, boundary, monomials)
    _add_structural(result, None, terms.collected)
    return result


def _count_monomials(counts: dict[str, int], monomials: StructuralMap) -> None:
    for _, factors in monomials:
        if len(factors) < 2:
            continue
        counts["n_quadratic_terms"] += 1
        binary_names = [name for name, binary in factors if binary]
        multiplicity = Counter(binary_names)
        if any(count >= 2 for count in multiplicity.values()):
            counts["n_binary_square_terms"] += 1
        elif binary_names:
            counts["n_bilinear_binary_terms"] += 1


def structural_counts(model) -> dict[str, int]:
    """Count canonical degree-two-or-higher structures in active constraints."""
    counts = {
        "n_quadratic_terms": 0,
        "n_bilinear_binary_terms": 0,
        "n_binary_square_terms": 0,
        "n_disaggregated_vars": 0,
        "n_nonlinear_constraints": 0,
    }
    for variable in model.component_data_objects(Var, descend_into=True):
        if "disaggregatedVars" in variable.name:
            counts["n_disaggregated_vars"] += 1
    for constraint in model.component_data_objects(Constraint, active=True, descend_into=True):
        degree = constraint.body.polynomial_degree()
        if degree not in (0, 1):
            counts["n_nonlinear_constraints"] += 1
        _count_monomials(counts, _structural_monomials(constraint.body))
    return counts
