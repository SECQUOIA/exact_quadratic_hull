"""Module for building random quadratic disjunctive Pyomo models."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyomo.environ as pyo
import pyomo.gdp as gdp


def generate_quadratic_function(
    n_dimensions: int,
    coeff_range: Tuple[float, float],
    ensure_positive_definite: bool = False,
    sparsity_factor: float = 0.0,
    random_seed: Optional[int] = None,
    min_eigenvalue: float = 0.1,
    condition_number: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a random quadratic function f(x) = xᵀQx + cᵀx + d with specified properties.

    Parameters
    ----------
    n_dimensions : int
        Number of dimensions (size of Q matrix)
    coeff_range : Tuple[float, float]
        Range for coefficients in Q, c, and constant term d
    ensure_positive_definite : bool, optional
        Whether to ensure Q is positive definite, by default False
    sparsity_factor : float, optional
        Factor controlling sparsity of Q (0.0 to 1.0), by default 0.0
    random_seed : Optional[int], optional
        Random seed for reproducibility, by default None
    min_eigenvalue : float, optional
        Minimum eigenvalue when ensuring positive definiteness, by default 0.1
    condition_number : Optional[float], optional
        Desired condition number of Q (ratio of largest to smallest eigenvalue),
        by default None (no condition number control)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        Q matrix, c vector, and constant term d for the quadratic function

    Raises
    ------
    ValueError
        If n_dimensions is not positive
        If coeff_range is invalid
        If sparsity_factor is not in [0, 1]
        If min_eigenvalue is not positive
        If condition_number is not greater than 1
    """
    # Input validation
    if n_dimensions <= 0:
        raise ValueError("n_dimensions must be positive")
    if coeff_range[0] >= coeff_range[1]:
        raise ValueError("coeff_range must be a valid range (low < high)")
    if not 0 <= sparsity_factor <= 1:
        raise ValueError("sparsity_factor must be in [0, 1]")
    if ensure_positive_definite and min_eigenvalue <= 0:
        raise ValueError("min_eigenvalue must be positive when ensuring positive definiteness")
    if condition_number is not None and condition_number <= 1:
        raise ValueError("condition_number must be greater than 1")

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random symmetric matrix Q
    Q = np.random.uniform(
        low=coeff_range[0], high=coeff_range[1], size=(n_dimensions, n_dimensions)
    )
    # Make Q symmetric first
    Q = (Q + Q.T) / 2

    # Apply sparsity
    if sparsity_factor > 0:
        # Create upper triangular mask with specified sparsity
        mask = np.triu(np.random.random((n_dimensions, n_dimensions)) > (1 - sparsity_factor))
        # Make mask symmetric
        mask = np.logical_or(mask, mask.T)
        Q = Q * mask

    # Ensure positive definiteness if required
    if ensure_positive_definite:
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(Q)

        # Adjust eigenvalues to ensure positive definiteness
        min_current_eigenvalue = np.min(eigenvalues)
        if min_current_eigenvalue < min_eigenvalue:
            # Shift all eigenvalues up to ensure minimum value
            shift = min_eigenvalue - min_current_eigenvalue
            eigenvalues = eigenvalues + shift

        # Control condition number if specified
        if condition_number is not None:
            max_eigenvalue = np.max(eigenvalues)
            target_min_eigenvalue = max_eigenvalue / condition_number
            if target_min_eigenvalue < min_eigenvalue:
                raise ValueError(
                    f"Condition number {condition_number} incompatible with "
                    f"min_eigenvalue {min_eigenvalue}"
                )
            # Scale eigenvalues to achieve desired condition number
            eigenvalues = (eigenvalues - min_eigenvalue) * (
                (max_eigenvalue - target_min_eigenvalue) / (max_eigenvalue - min_eigenvalue)
            ) + target_min_eigenvalue

        # Reconstruct Q with adjusted eigenvalues
        Q = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Generate linear coefficients c
    c = np.random.uniform(low=coeff_range[0], high=coeff_range[1], size=n_dimensions)

    # Generate constant term d
    d = np.random.uniform(low=coeff_range[0], high=coeff_range[1])

    return Q, c, d


def build_model(
    n_dimensions: int = 3,
    n_disjunctions: int = 2,
    n_disjuncts_per_disjunction: int = 3,
    n_constraints_per_disjunct: int = 5,
    n_feasible_regions: int = 3,
    coeff_range: Tuple[float, float] = (-1.0, 1.0),
    constraint_margin: Tuple[float, float] = (0.0, 0.01),
    x_range: Tuple[float, float] = (-1.0, 1.0),
    ensure_positive_definite: bool = False,
    sparsity_factor: float = 0,
    random_seed: Optional[int] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> pyo.ConcreteModel:
    """
    Build a random quadratic disjunctive Pyomo model.

    Parameters
    ----------
    n_dimensions : int, optional
        Number of decision variables in the model, by default 3
    n_disjunctions : int, optional
        Number of disjunctive constraints, by default 2
    n_disjuncts_per_disjunction : int, optional
        Number of disjuncts in each disjunction, by default 2
    n_feasible_regions : int, optional
        Number of feasible regions to generate, by default 2
    coeff_range : Tuple[float, float], optional
        Range for all coefficients (quadratic, linear, and constant), by default (-1.0, 1.0)
    constraint_margin : Tuple[float, float], optional
        Range for constraint margins (controls how tight/loose constraints are),
        by default (0.0, 1.0)
    x_range : Tuple[float, float], optional
        Range for the decision variables (bounds), by default (-10.0, 10.0)
    ensure_positive_definite : bool, optional
        Whether to ensure the quadratic matrix is positive definite, by default False
    sparsity_factor : float, optional
        Factor controlling sparsity of the quadratic matrix (0.0 to 1.0), by default 0
    random_seed : Optional[int], optional
        Random seed for reproducibility, by default None
    parameters : Optional[Dict[str, Any]], optional
        Dictionary of parameters that override the default values, by default None

    Returns
    -------
    pyo.ConcreteModel
        A Pyomo model with random quadratic objective and disjunctive constraints
    """
    # Override defaults with parameters if provided
    if parameters is not None:
        if "n_dimensions" in parameters:
            n_dimensions = parameters["n_dimensions"]
        if "n_disjunctions" in parameters:
            n_disjunctions = parameters["n_disjunctions"]
        if "n_disjuncts_per_disjunction" in parameters:
            n_disjuncts_per_disjunction = parameters["n_disjuncts_per_disjunction"]
        if "n_constraints_per_disjunct" in parameters:
            n_constraints_per_disjunct = parameters["n_constraints_per_disjunct"]
        if "n_feasible_regions" in parameters:
            n_feasible_regions = parameters["n_feasible_regions"]
        if "coeff_range" in parameters:
            coeff_range = parameters["coeff_range"]
        if "constraint_margin" in parameters:
            constraint_margin = parameters["constraint_margin"]
        if "x_range" in parameters:
            x_range = parameters["x_range"]
        if "ensure_positive_definite" in parameters:
            ensure_positive_definite = parameters["ensure_positive_definite"]
        if "sparsity_factor" in parameters:
            sparsity_factor = parameters["sparsity_factor"]
        if "random_seed" in parameters:
            random_seed = parameters["random_seed"]

    # Validate that n_feasible_regions is less than or equal to n_disjuncts_per_disjunction
    assert (
        n_feasible_regions <= n_disjuncts_per_disjunction
    ), "Number of feasible regions must be less or equal to number of disjuncts per disjunction"

    # Create model
    model = pyo.ConcreteModel()

    # Create sets
    model.dimensions = pyo.RangeSet(n_dimensions)  # Dimensions set
    model.disjunctions = pyo.RangeSet(n_disjunctions)  # Disjunctions set
    model.disjuncts = pyo.RangeSet(n_disjuncts_per_disjunction)  # Disjuncts set
    model.feasible_regions = pyo.RangeSet(n_feasible_regions)  # Feasible regions set

    # Create parameters with validation
    model.coeff_range = pyo.Param(
        initialize=coeff_range, doc="Range for all coefficients (quadratic, linear, and constant)"
    )
    model.constraint_margin = pyo.Param(
        initialize=constraint_margin,
        doc="Range for constraint margins (controls how tight/loose constraints are)",
    )
    model.x_range = pyo.Param(
        initialize=x_range,
        doc="Range for the decision variables (bounds)",
    )
    model.sparsity = pyo.Param(
        initialize=sparsity_factor,
        doc="Factor controlling sparsity of the quadratic matrix (0.0 to 1.0)",
    )
    model.ensure_positive_definite = pyo.Param(
        initialize=ensure_positive_definite,
        doc="Whether to ensure the quadratic matrix is positive definite",
    )
    model.random_seed = pyo.Param(
        initialize=random_seed if random_seed is not None else 0,
        doc="Random seed for reproducibility",
    )

    # Store n_constraints_per_disjunct as a model parameter
    model.n_constraints_per_disjunct = pyo.Param(
        initialize=n_constraints_per_disjunct, doc="Number of constraints per disjunct"
    )

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random coordinates for feasible regions
    feasible_coords = {}
    for region in range(1, n_feasible_regions + 1):
        coords = np.random.uniform(low=x_range[0], high=x_range[1], size=n_dimensions)
        feasible_coords[region] = coords

    # Store feasible region coordinates in model
    model.feasible_region_coords = pyo.Param(
        model.feasible_regions,
        model.dimensions,
        initialize=lambda model, region, dimension: feasible_coords[region][dimension - 1],
        doc="Coordinates of feasible regions",
    )

    # Create variables with proper domain and bounds
    model.x = pyo.Var(
        model.dimensions,
        domain=pyo.Reals,
        bounds=x_range,  # Use x_range for bounds
        doc="Decision variables",
    )

    # Generate objective and constraints
    _generate_quadratic_objective(model=model)
    _generate_disjunctive_constraints(model=model)

    # Validate model
    if not _validate_model(model):
        raise ValueError("Generated model is invalid")

    return model


def _generate_quadratic_function(
    model: pyo.ConcreteModel,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate quadratic function coefficients using model parameters.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model containing parameters

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        Q matrix, c vector, and constant term d for the quadratic function
    """
    # Extract parameters from model
    n_dimensions = len(model.dimensions)
    coeff_range = model.coeff_range.value
    ensure_positive_definite = model.ensure_positive_definite.value
    sparsity_factor = model.sparsity.value
    random_seed = model.random_seed.value if model.random_seed.value != 0 else None

    return generate_quadratic_function(
        n_dimensions=n_dimensions,
        coeff_range=coeff_range,
        ensure_positive_definite=ensure_positive_definite,
        sparsity_factor=sparsity_factor,
        random_seed=random_seed,
    )


def _generate_quadratic_objective(
    model: pyo.ConcreteModel,
) -> None:
    """
    Generate a random quadratic objective function for the model.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model to add the objective to
    """
    # Generate quadratic function coefficients using model parameters
    Q, c, d = _generate_quadratic_function(model=model)

    # Store Q matrix
    Q_obj_dict = {(i, j): Q[i - 1, j - 1] for i in model.dimensions for j in model.dimensions}
    model.Q_objective = pyo.Param(
        model.dimensions,
        model.dimensions,
        initialize=Q_obj_dict,
        mutable=True,
        doc="Quadratic matrix for objective function",
    )

    # Store c vector
    c_obj_dict = {i: c[i - 1] for i in model.dimensions}
    model.c_objective = pyo.Param(
        model.dimensions,
        initialize=c_obj_dict,
        mutable=True,
        doc="Linear coefficients for objective function",
    )

    # Store d constant
    model.d_objective = pyo.Param(
        initialize=d, mutable=True, doc="Constant term for objective function"
    )

    # Define quadratic objective
    def quadratic_objective_rule(m: pyo.ConcreteModel) -> pyo.Expression:
        # Use the stored parameters instead of direct values
        Q_term = sum(
            m.Q_objective[i, j] * m.x[i] * m.x[j] for i in m.dimensions for j in m.dimensions
        )
        c_term = sum(m.c_objective[i] * m.x[i] for i in m.dimensions)

        return Q_term + c_term + m.d_objective

    model.obj = pyo.Objective(
        rule=quadratic_objective_rule,
        sense=pyo.minimize,
        doc="Quadratic objective function",
    )


def _generate_disjunctive_constraints(
    model: pyo.ConcreteModel,
) -> None:
    """
    Generate random disjunctive constraints for the model.

    Creates quadratic constraints in the form x^T Q x + c^T x + d ≤ 0
    for each disjunct in each disjunction. For feasible regions, ensures
    the constraint is satisfied with margin.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model to add constraints to
    """
    # Get random seed from model
    random_seed = model.random_seed.value if model.random_seed.value != 0 else None

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Extract parameters from model
    constraint_margin = model.constraint_margin.value
    n_feasible_regions = len(model.feasible_regions)
    n_constraints_per_disjunct = model.n_constraints_per_disjunct

    # Initialize counters as mutable parameters
    model.num_constraints = pyo.Param(
        initialize=0, mutable=True, doc="Counter for number of constraints"
    )
    model.num_positive_semidefinite_Q = pyo.Param(
        initialize=0, mutable=True, doc="Counter for number of positive semidefinite Q matrices"
    )
    model.num_positive_d = pyo.Param(
        initialize=0, mutable=True, doc="Counter for number of positive d values"
    )

    # Create a range set for constraints per disjunct
    model.constraints_per_disjunct = pyo.RangeSet(n_constraints_per_disjunct)

    # Create disjuncts blocks for GDP formulation
    model.disjunct_blocks = gdp.Disjunct(model.disjunctions, model.disjuncts)

    # Generate constraints for each disjunct in each disjunction
    for disjunction in model.disjunctions:
        for disjunct in model.disjuncts:
            # Determine if this disjunct should be feasible (for one of the feasible regions)
            is_feasible = False
            target_region = None

            # If disjunct index is within the number of feasible regions, make it feasible
            if disjunct <= n_feasible_regions:
                is_feasible = True
                target_region = disjunct

            # Get the feasible point coordinates if relevant
            if is_feasible:
                point = np.array(
                    [
                        model.feasible_region_coords[target_region, dimension]
                        for dimension in model.dimensions
                    ]
                )

            # Generate multiple constraints for this disjunct
            for constraint_idx in range(1, n_constraints_per_disjunct + 1):
                # Generate a random quadratic function
                Q, c, d_base = _generate_quadratic_function(model)

                # Check if Q is positive semidefinite
                eigenvalues = np.linalg.eigvalsh(Q)
                is_psd = np.all(eigenvalues >= -1e-10)  # Allow for small numerical errors

                if is_psd:
                    # Increment counter for positive semidefinite matrices
                    model.num_positive_semidefinite_Q.value += 1

                # Calculate appropriate d to ensure feasibility with margin
                if is_feasible:
                    # Evaluate x^T Q x + c^T x at the feasible point
                    quadratic_term = point.T @ Q @ point
                    linear_term = c.T @ point
                    lhs_value = quadratic_term + linear_term

                    # Set d to ensure the constraint is satisfied with margin
                    # For x^T Q x + c^T x + d ≤ 0, we want to ensure a negative margin
                    margin = np.random.uniform(constraint_margin[0], constraint_margin[1])
                    d = -lhs_value - margin  # This ensures x^T Q x + c^T x + d = -margin (negative)
                else:
                    # For non-feasible disjuncts, use the random d from _generate_quadratic_function
                    d = d_base

                # Check if d is positive
                if d > 0:
                    model.num_positive_d.value += 1

                # Increment constraint counter
                model.num_constraints.value += 1

                # Store Q, c, d for this constraint with unique names including constraint index
                Q_param_name = f"Q_constraint_{disjunction}_{disjunct}_{constraint_idx}"
                c_param_name = f"c_constraint_{disjunction}_{disjunct}_{constraint_idx}"
                d_param_name = f"d_constraint_{disjunction}_{disjunct}_{constraint_idx}"

                # Create a parameter for Q (as a flattened dict)
                Q_dict = {
                    (i, j): Q[i - 1, j - 1] for i in model.dimensions for j in model.dimensions
                }
                setattr(
                    model.disjunct_blocks[disjunction, disjunct],
                    Q_param_name,
                    pyo.Param(
                        model.dimensions,
                        model.dimensions,
                        initialize=Q_dict,
                        mutable=True,
                        doc=(
                            f"Quadratic matrix for constraint {constraint_idx} "
                            f"of disjunct {disjunct} in disjunction {disjunction}"
                        ),
                    ),
                )

                # Create a parameter for c
                c_dict = {i: c[i - 1] for i in model.dimensions}
                setattr(
                    model.disjunct_blocks[disjunction, disjunct],
                    c_param_name,
                    pyo.Param(
                        model.dimensions,
                        initialize=c_dict,
                        mutable=True,
                        doc=(
                            f"Linear coefficients for constraint {constraint_idx} "
                            f"of disjunct {disjunct} in disjunction {disjunction}"
                        ),
                    ),
                )

                # Create a parameter for d
                setattr(
                    model.disjunct_blocks[disjunction, disjunct],
                    d_param_name,
                    pyo.Param(
                        initialize=d,
                        mutable=True,
                        doc=(
                            f"Constant term for constraint {constraint_idx} "
                            f"of disjunct {disjunct} in disjunction {disjunction}"
                        ),
                    ),
                )

                # Define the quadratic constraint rule for this specific constraint
                def make_constraint_rule(
                    di: int, dj: int, ci: int
                ) -> Callable[[Any], pyo.Expression]:
                    def rule(block: Any) -> pyo.Expression:
                        # Parameters are stored on the block (disjunct), not on the main model
                        Q_param = getattr(block, f"Q_constraint_{di}_{dj}_{ci}")
                        c_param = getattr(block, f"c_constraint_{di}_{dj}_{ci}")
                        d_param = getattr(block, f"d_constraint_{di}_{dj}_{ci}")

                        # But the x variables are on the main model
                        m = block.model()
                        quadratic_term = sum(
                            Q_param[i, j] * m.x[i] * m.x[j]
                            for i in m.dimensions
                            for j in m.dimensions
                        )
                        linear_term = sum(c_param[i] * m.x[i] for i in m.dimensions)

                        return quadratic_term + linear_term + d_param <= 0

                    return rule

                # Add the quadratic constraint to the disjunct block
                constraint_name = f"quadratic_constraint_{constraint_idx}"
                setattr(
                    model.disjunct_blocks[disjunction, disjunct],
                    constraint_name,
                    pyo.Constraint(
                        rule=make_constraint_rule(disjunction, disjunct, constraint_idx)
                    ),
                )

    # Define the disjunction rule
    def disjunction_rule(m: pyo.ConcreteModel, disjunction: int) -> List[Any]:
        return [m.disjunct_blocks[disjunction, disjunct] for disjunct in m.disjuncts]

    model.logical_disjunctions = gdp.Disjunction(model.disjunctions, rule=disjunction_rule)


def _validate_model(model: pyo.ConcreteModel) -> bool:
    """
    Validate the generated model for consistency and feasibility.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model to validate

    Returns
    -------
    bool
        True if the model is valid, False otherwise
    """
    # Check if all required components exist
    if not hasattr(model, "dimensions"):
        return False
    if not hasattr(model, "disjunctions"):
        return False
    if not hasattr(model, "disjuncts"):
        return False
    if not hasattr(model, "feasible_regions"):
        return False
    if not hasattr(model, "x"):
        return False
    if not hasattr(model, "obj"):
        return False
    if not hasattr(model, "feasible_region_coords"):
        return False

    # Check if dimensions are valid
    n_dimensions = len(model.dimensions)
    if n_dimensions <= 0:
        return False

    # Check if feasible regions count is valid
    n_feasible_regions = len(model.feasible_regions)
    n_disjuncts_per_disjunction = len(model.disjuncts)
    if n_feasible_regions > n_disjuncts_per_disjunction:
        return False

    # Check if objective is quadratic
    if not isinstance(model.obj, pyo.Objective):
        return False

    return True
