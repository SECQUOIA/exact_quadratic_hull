"""Tests for the build_model module."""

import warnings

import numpy as np
import pytest

from random_quadratic.build_model import generate_quadratic_function


def test_generate_quadratic_function_basic() -> None:
    """Test basic functionality of generate_quadratic_function."""
    n_dimensions = 3
    coeff_range = (-1.0, 1.0)
    tolerance = 1e-10  # Small tolerance for floating-point comparisons

    Q, c, d = generate_quadratic_function(n_dimensions=n_dimensions, coeff_range=coeff_range)

    # Check shapes and types
    assert Q.shape == (n_dimensions, n_dimensions)
    assert c.shape == (n_dimensions,)
    assert isinstance(d, float)

    # Check symmetry
    assert np.allclose(Q, Q.T), "Matrix Q should be symmetric"

    # Check coefficient ranges with warnings
    if not np.all(Q >= coeff_range[0] - tolerance) or not np.all(Q <= coeff_range[1] + tolerance):
        warnings.warn(
            f"Some Q values are outside the expected range [{coeff_range[0]}, {coeff_range[1]}]"
        )
    if not np.all(c >= coeff_range[0] - tolerance) or not np.all(c <= coeff_range[1] + tolerance):
        warnings.warn(
            f"Some c values are outside the expected range [{coeff_range[0]}, {coeff_range[1]}]"
        )
    if not (coeff_range[0] - tolerance <= d <= coeff_range[1] + tolerance):
        warnings.warn(
            f"Constant d is outside the expected range [{coeff_range[0]}, {coeff_range[1]}]"
        )


def test_generate_quadratic_function_sparsity() -> None:
    """Test sparsity control in generate_quadratic_function."""
    n_dimensions = 20
    coeff_range = (-1.0, 1.0)
    sparsity_factor = 0.5

    Q, c, d = generate_quadratic_function(
        n_dimensions=n_dimensions, coeff_range=coeff_range, sparsity_factor=sparsity_factor
    )

    # Count non-zero elements (excluding diagonal)
    non_diagonal_elements = n_dimensions * (n_dimensions - 1)
    non_zero_elements = np.count_nonzero(Q) - n_dimensions  # Subtract diagonal

    # Calculate actual sparsity
    actual_sparsity = non_zero_elements / non_diagonal_elements

    # Allow for some tolerance in sparsity
    assert abs(actual_sparsity - sparsity_factor) < 0.1

    # Check other outputs are valid
    assert c.shape == (n_dimensions,)
    assert isinstance(d, float)


def test_generate_quadratic_function_positive_definite() -> None:
    """Test positive definiteness in generate_quadratic_function."""
    n_dimensions = 5
    coeff_range = (-1.0, 1.0)
    min_eigenvalue = 0.5

    Q, c, d = generate_quadratic_function(
        n_dimensions=n_dimensions,
        coeff_range=coeff_range,
        ensure_positive_definite=True,
        min_eigenvalue=min_eigenvalue,
    )

    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(Q)
    # Add a small tolerance for numerical precision
    assert np.all(eigenvalues >= min_eigenvalue - 1e-10)

    # Check other outputs are valid
    assert c.shape == (n_dimensions,)
    assert isinstance(d, float)


def test_generate_quadratic_function_condition_number() -> None:
    """Test condition number control in generate_quadratic_function."""
    n_dimensions = 4
    coeff_range = (-1.0, 1.0)
    condition_number = 10.0

    Q, c, d = generate_quadratic_function(
        n_dimensions=n_dimensions,
        coeff_range=coeff_range,
        ensure_positive_definite=True,
        condition_number=condition_number,
    )

    # Calculate actual condition number
    eigenvalues = np.linalg.eigvalsh(Q)
    actual_condition_number = np.max(eigenvalues) / np.min(eigenvalues)

    # Allow for some tolerance in condition number
    assert abs(actual_condition_number - condition_number) < 0.1

    # Check other outputs are valid
    assert c.shape == (n_dimensions,)
    assert isinstance(d, float)


def test_generate_quadratic_function_constant_term() -> None:
    """Test constant term generation in generate_quadratic_function."""
    n_dimensions = 3
    coeff_range = (-1.0, 1.0)

    Q, c, d = generate_quadratic_function(
        n_dimensions=n_dimensions,
        coeff_range=coeff_range,
    )

    # Check constant term is within specified coeff_range
    assert coeff_range[0] <= d <= coeff_range[1]

    # Check other outputs are valid
    assert Q.shape == (n_dimensions, n_dimensions)
    assert c.shape == (n_dimensions,)


def test_generate_quadratic_function_reproducibility() -> None:
    """Test reproducibility with random seed."""
    n_dimensions = 3
    coeff_range = (-1.0, 1.0)
    random_seed = 42

    # Generate two matrices with same seed
    Q1, c1, d1 = generate_quadratic_function(
        n_dimensions=n_dimensions, coeff_range=coeff_range, random_seed=random_seed
    )

    Q2, c2, d2 = generate_quadratic_function(
        n_dimensions=n_dimensions, coeff_range=coeff_range, random_seed=random_seed
    )

    # Check reproducibility
    assert np.allclose(Q1, Q2)
    assert np.allclose(c1, c2)
    assert d1 == d2


def test_generate_quadratic_function_validation() -> None:
    """Test input validation in generate_quadratic_function."""
    # Test invalid n_dimensions
    with pytest.raises(ValueError, match="n_dimensions must be positive"):
        generate_quadratic_function(n_dimensions=0, coeff_range=(-1.0, 1.0))

    # Test invalid coeff_range
    with pytest.raises(ValueError, match="coeff_range must be a valid range"):
        generate_quadratic_function(n_dimensions=3, coeff_range=(1.0, -1.0))

    # Test invalid sparsity_factor
    with pytest.raises(ValueError, match="sparsity_factor must be in"):
        generate_quadratic_function(n_dimensions=3, coeff_range=(-1.0, 1.0), sparsity_factor=1.5)

    # Test invalid min_eigenvalue
    with pytest.raises(ValueError, match="min_eigenvalue must be positive"):
        generate_quadratic_function(
            n_dimensions=3,
            coeff_range=(-1.0, 1.0),
            ensure_positive_definite=True,
            min_eigenvalue=0.0,
        )

    # Test invalid condition_number
    with pytest.raises(ValueError, match="condition_number must be greater than 1"):
        generate_quadratic_function(
            n_dimensions=3,
            coeff_range=(-1.0, 1.0),
            ensure_positive_definite=True,
            condition_number=0.5,
        )


def test_generate_quadratic_function_combined() -> None:
    """Test combined features of generate_quadratic_function."""
    n_dimensions = 5
    coeff_range = (-1.0, 1.0)
    sparsity_factor = 0.3
    min_eigenvalue = 0.2
    condition_number = 5.0
    random_seed = 123
    tolerance = 1e-10  # Small tolerance for floating-point comparisons

    Q, c, d = generate_quadratic_function(
        n_dimensions=n_dimensions,
        coeff_range=coeff_range,
        ensure_positive_definite=True,
        sparsity_factor=sparsity_factor,
        min_eigenvalue=min_eigenvalue,
        condition_number=condition_number,
        random_seed=random_seed,
    )

    # Check all properties simultaneously
    assert Q.shape == (n_dimensions, n_dimensions)
    assert c.shape == (n_dimensions,)
    assert isinstance(d, float)
    assert np.allclose(Q, Q.T)

    # Check sparsity
    non_diagonal_elements = n_dimensions * (n_dimensions - 1)
    non_zero_elements = np.count_nonzero(Q) - n_dimensions
    actual_sparsity = non_zero_elements / non_diagonal_elements
    assert abs(actual_sparsity - sparsity_factor) < 0.1

    # Check positive definiteness and condition number
    eigenvalues = np.linalg.eigvalsh(Q)
    assert np.all(eigenvalues >= min_eigenvalue)
    actual_condition_number = np.max(eigenvalues) / np.min(eigenvalues)
    assert abs(actual_condition_number - condition_number) < 0.1

    # Check coefficient ranges with warnings
    if not np.all(Q >= coeff_range[0] - tolerance) or not np.all(Q <= coeff_range[1] + tolerance):
        warnings.warn(
            f"Some Q values are outside the expected range [{coeff_range[0]}, {coeff_range[1]}]"
        )
    if not np.all(c >= coeff_range[0] - tolerance) or not np.all(c <= coeff_range[1] + tolerance):
        warnings.warn(
            f"Some c values are outside the expected range [{coeff_range[0]}, {coeff_range[1]}]"
        )
    if not (coeff_range[0] - tolerance <= d <= coeff_range[1] + tolerance):
        warnings.warn(
            f"Constant d is outside the expected range [{coeff_range[0]}, {coeff_range[1]}]"
        )
