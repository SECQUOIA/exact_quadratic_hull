import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import pyomo.gdp.plugins.hull_exact
# import pyomo.gdp.plugins.hull_reduced_y
# Import local modules
import solve
from pyomo.environ import value

# plugins = [pyomo.gdp.plugins.hull_exact, pyomo.gdp.plugins.hull_reduced_y]


def verify_model(model_name: str) -> Any:
    """
    Verify that the model exists and has 2 dimensions.

    Parameters
    ----------
    model_name : str
        Name of the model file

    Returns
    -------
    Any
        The loaded model
    """
    # Construct path to model
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    models_dir = os.path.join(data_dir, "models")
    model_path = os.path.join(models_dir, model_name)

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model
    model = solve.load_model(model_path)

    # Check dimensions
    if len(model.dimensions) != 2:
        raise ValueError(f"Model must have 2 dimensions, but has {len(model.dimensions)}")

    # Check for Excel results
    excel_path = os.path.join(data_dir, "results.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Results file not found: {excel_path}")

    return model


def get_solution_points(model_name: str) -> Dict[str, Tuple[float, float]]:
    """
    Get solution points from Excel results file.

    Parameters
    ----------
    model_name : str
        Name of the model file

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Dictionary mapping strategy names to solution points
    """
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    excel_path = os.path.join(data_dir, "results.xlsx")

    df = pd.read_excel(excel_path)

    # Filter for the specific model
    model_results = df[df["Model Name"] == model_name]

    if model_results.empty:
        raise ValueError(f"No results found for model {model_name}")

    solution_points = {}

    # Extract solution points for each strategy
    for _, row in model_results.iterrows():
        strategy = row["Strategy"]
        status = row["Status"]

        # Only include optimal solutions
        if status == "optimal" and isinstance(row["Optimal X Values"], str):
            # Parse the x values string
            x_values_str = row["Optimal X Values"]

            # Split by comma and extract x1, x2 values
            x_values = {}
            for part in x_values_str.split(", "):
                key, value = part.split("=")
                x_values[key] = float(value)

            # Store as (x1, x2) tuple
            if "x1" in x_values and "x2" in x_values:
                solution_points[strategy] = (x_values["x1"], x_values["x2"])

    return solution_points


def extract_objective_coefficients(model: Any) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract quadratic objective function coefficients from the model.

    Parameters
    ----------
    model : Any
        The Pyomo model

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        Q matrix, c vector, and constant term d
    """
    # Create empty matrices and vectors for objective coefficients
    n_dim = len(model.dimensions)
    Q = np.zeros((n_dim, n_dim))
    c = np.zeros(n_dim)
    d = 0.0

    # Expression for the objective is stored in model.obj.expr
    # We need to analyze this expression to extract Q, c, and d

    # Loop through all terms in the objective expression
    for term in model.obj.expr.polynomial_degree():
        if term.polynomial_degree() == 2:
            # Quadratic term (could be x[i]*x[j])
            indices = [v.index() for v in term.args]
            if len(indices) == 2:
                i, j = indices
                Q[i - 1, j - 1] = term.coefficient()
                if i != j:  # For off-diagonal terms, add symmetric counterpart
                    Q[j - 1, i - 1] = term.coefficient()
            else:
                # Handle x[i]^2 case
                i = indices[0]
                Q[i - 1, i - 1] = term.coefficient()
        elif term.polynomial_degree() == 1:
            # Linear term
            i = term.args[0].index()
            c[i - 1] = term.coefficient()
        else:
            # Constant term
            d = term

    return Q, c, d


def evaluate_objective(model: Any, x1: float, x2: float) -> float:
    """
    Evaluate the objective function at a point.

    Parameters
    ----------
    model : Any
        The Pyomo model
    x1 : float
        First coordinate
    x2 : float
        Second coordinate

    Returns
    -------
    float
        Objective function value
    """
    # Use stored parameters Q_objective, c_objective, d_objective
    x = np.array([x1, x2])

    # Extract Q, c, d from model's stored parameters
    Q = np.zeros((2, 2))
    c = np.zeros(2)

    # Get Q matrix
    for i in model.dimensions:
        for j in model.dimensions:
            Q[i - 1, j - 1] = float(value(model.Q_objective[i, j]))

    # Get c vector
    for i in model.dimensions:
        c[i - 1] = float(value(model.c_objective[i]))

    # Get d constant
    d = float(value(model.d_objective))

    # Calculate quadratic term: x^T Q x
    q_term = float(x.T @ Q @ x)

    # Calculate linear term: c^T x
    l_term = float(c.T @ x)

    # Return f(x) = x^T Q x + c^T x + d
    return float(q_term + l_term + d)


def evaluate_constraint(Q: np.ndarray, c: np.ndarray, d: float, x1: float, x2: float) -> float:
    """
    Evaluate a quadratic constraint function at a point.

    Parameters
    ----------
    Q : np.ndarray
        Quadratic matrix
    c : np.ndarray
        Linear coefficients
    d : float
        Constant term
    x1 : float
        First coordinate
    x2 : float
        Second coordinate

    Returns
    -------
    float
        Constraint function value
    """
    x = np.array([x1, x2])
    q_term = float(x.T @ Q @ x)
    l_term = float(c.T @ x)
    return float(q_term + l_term + d)


def extract_disjunctive_constraints(model: Any) -> List[Dict[str, Any]]:
    """
    Extract disjunctive constraints from the model.

    Parameters
    ----------
    model : Any
        The Pyomo model

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing constraint data
    """
    constraints = []

    # Iterate through disjunctions and disjuncts
    for disjunction in model.disjunctions:
        for disjunct in model.disjuncts:
            disjunct_block = model.disjunct_blocks[disjunction, disjunct]

            # Get the number of constraints in this disjunct
            n_constraints = model.n_constraints_per_disjunct.value

            for constraint_idx in range(1, n_constraints + 1):
                # Get parameter names for this constraint
                Q_param_name = f"Q_constraint_{disjunction}_{disjunct}_{constraint_idx}"
                c_param_name = f"c_constraint_{disjunction}_{disjunct}_{constraint_idx}"
                d_param_name = f"d_constraint_{disjunction}_{disjunct}_{constraint_idx}"

                # Get the parameters
                Q_param = getattr(disjunct_block, Q_param_name)
                c_param = getattr(disjunct_block, c_param_name)
                d_param = getattr(disjunct_block, d_param_name)

                # Convert Q_param to numpy array
                Q = np.zeros((2, 2))
                for i in model.dimensions:
                    for j in model.dimensions:
                        # Use value() to explicitly convert Pyomo numeric value to float
                        Q[i - 1, j - 1] = value(Q_param[i, j])

                # Convert c_param to numpy array
                c = np.zeros(2)
                for i in model.dimensions:
                    # Use value() to explicitly convert Pyomo numeric value to float
                    c[i - 1] = value(c_param[i])

                # Get d value - explicitly convert to float
                d = value(d_param)

                # Store constraint data
                constraints.append(
                    {
                        "disjunction": disjunction,
                        "disjunct": disjunct,
                        "constraint_idx": constraint_idx,
                        "Q": Q,
                        "c": c,
                        "d": d,
                    }
                )

    return constraints


def save_calculation_results(
    model_name: str,
    X1: np.ndarray,
    X2: np.ndarray,
    Z_obj: np.ndarray,
    feasible_mask: np.ndarray,
    solution_points: Dict[str, Tuple[float, float]],
    min_x1: float,
    max_x1: float,
    min_x2: float,
    max_x2: float,
    levels: Any,
) -> str:
    """
    Save calculation results to a file for later plotting without recalculation.

    Parameters
    ----------
    model_name : str
        Name of the model
    X1 : np.ndarray
        X1 grid coordinates
    X2 : np.ndarray
        X2 grid coordinates
    Z_obj : np.ndarray
        Objective function values at grid points
    feasible_mask : np.ndarray
        Boolean mask of feasible regions
    solution_points : Dict[str, Tuple[float, float]]
        Dictionary of solution points by strategy
    min_x1, max_x1, min_x2, max_x2 : float
        Plot bounds
    levels : Any
        Contour levels

    Returns
    -------
    str
        Path to the saved data file
    """
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    cache_dir = os.path.join(data_dir, "calculation_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a dictionary with all data
    data = {
        "model_name": model_name,
        "X1": X1,
        "X2": X2,
        "Z_obj": Z_obj,
        "feasible_mask": feasible_mask,
        "solution_points": solution_points,
        "min_x1": min_x1,
        "max_x1": max_x1,
        "min_x2": min_x2,
        "max_x2": max_x2,
        "levels": levels,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save to file
    file_path = os.path.join(cache_dir, f"{model_name.replace('.pkl', '')}_calc_data.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Calculation results saved to: {file_path}")
    return file_path


def load_calculation_results(model_name: str) -> Dict[str, Any]:
    """
    Load previously saved calculation results.

    Parameters
    ----------
    model_name : str
        Name of the model

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all saved calculation data
    """
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    cache_dir = os.path.join(data_dir, "calculation_cache")
    file_path = os.path.join(cache_dir, f"{model_name.replace('.pkl', '')}_calc_data.pkl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cached calculation data not found: {file_path}")

    # Load data
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded calculation results from: {file_path}")
    print(f"Original calculation timestamp: {data['timestamp']}")

    # Ensure we're returning a properly typed dictionary
    result: Dict[str, Any] = data
    return result


def calculate_contour_levels(z_min: float, z_max: float, n_levels: int = 15) -> np.ndarray:
    """
    Calculate logarithmically spaced contour levels between min and max values.

    Parameters
    ----------
    z_min : float
        Minimum value of the objective function
    z_max : float
        Maximum value of the objective function
    n_levels : int
        Number of contour levels to generate

    Returns
    -------
    np.ndarray
        Array of contour levels with logarithmic spacing
    """
    # Calculate the range
    z_range = z_max - z_min

    # Create logarithmically spaced values from 0 to z_range
    # Use small epsilon to avoid log(0)

    # Log space
    # epsilon = 1e-10
    # space = np.logspace(np.log10(epsilon), np.log10(z_range), n_levels)

    # Quadratic space
    space = np.linspace(0.1, z_range, n_levels)
    space = space**2

    # Cubic space
    # space = np.linspace(0.1, z_range, n_levels)
    # space = space ** 3

    # Shift by adding z_min to place in correct range
    levels = z_min + space

    return levels


def create_and_save_plot(
    X1: np.ndarray,
    X2: np.ndarray,
    Z_obj: np.ndarray,
    feasible_mask: np.ndarray,
    solution_points: Dict[str, Tuple[float, float]],
    min_x1: float,
    max_x1: float,
    min_x2: float,
    max_x2: float,
    levels: np.ndarray,
    model_name: str,
) -> str:
    """
    Create and save a plot of feasible regions, objective contours, and solution points.

    Parameters
    ----------
    X1 : np.ndarray
        X1 grid coordinates
    X2 : np.ndarray
        X2 grid coordinates
    Z_obj : np.ndarray
        Objective function values at grid points
    feasible_mask : np.ndarray
        Boolean mask of feasible regions
    solution_points : Dict[str, Tuple[float, float]]
        Dictionary mapping strategy names to solution points
    min_x1, max_x1, min_x2, max_x2 : float
        Plot bounds
    levels : np.ndarray
        Contour levels
    model_name : str
        Name of the model file for saving the plot

    Returns
    -------
    str
        Path to the saved plot file
    """
    # Print all found strategies
    print("\nSolution points found for strategies:")
    for strategy, point in solution_points.items():
        print(f"Strategy: {strategy}, Point: ({point[0]:.6f}, {point[1]:.6f})")
    print()

    # Create figure
    plt.figure(figsize=(12, 10))

    # 1. First layer: Plot feasible region with moderate transparency
    plt.imshow(
        feasible_mask,
        extent=[min_x1, max_x1, min_x2, max_x2],
        origin="lower",
        alpha=0.9,  # Slightly more opaque
        cmap="Greens",
    )

    # Add contour outline for the feasible region to make it more visible
    plt.contour(
        X1,
        X2,
        feasible_mask.astype(float),
        levels=[0.5],
        colors="darkgreen",
        linewidths=2.0,  # Thicker lines for better visibility
    )

    # Plot only contour lines with enhanced visibility
    contour = plt.contour(
        X1,
        X2,
        Z_obj,
        levels=levels,
        colors="black",  # Use black for better visibility
        linewidths=0.8,  # Thinner lines to avoid clutter with more levels
    )

    # Add contour labels with larger font
    plt.clabel(contour, inline=True, fontsize=10, fmt="%.2e")

    # 3. Plot solution points with very visible markers
    markers = {
        "gdp.bigm": "o",
        "gdp.hull": "s",
        "gdp.hull_exact": "^",
        "gdp.hull_reduced_y": "x",
        "baron_gdp.bigm": "o",
        "baron_gdp.hull": "s",
        "baron_gdp.hull_exact": "^",
        "baron_gdp.hull_reduced_y": "x",
    }
    colors = {
        "gdp.bigm": "red",
        "gdp.hull": "blue",
        "gdp.hull_exact": "purple",
        "gdp.hull_reduced_y": "orange",
        "baron_gdp.bigm": "darkred",
        "baron_gdp.hull": "darkblue",
        "baron_gdp.hull_exact": "indigo",
        "baron_gdp.hull_reduced_y": "darkorange",
    }

    # Then plot the actual solution points
    for strategy, point in solution_points.items():
        marker = markers.get(strategy, "*")
        color = colors.get(strategy, "black")
        plt.plot(
            point[0],
            point[1],
            marker=marker,
            color=color,
            markersize=10,  # Smaller markers
            markeredgecolor="black",  # Black outline
            markeredgewidth=1.0,  # Thinner outline
            alpha=0.75,  # Add transparency
            label=f"{strategy}: ({point[0]:.4f}, {point[1]:.4f})",
        )

    # Add legend, title, and labels
    plt.legend(loc="lower right", fontsize=12)
    plt.title("Feasible Regions, Objective Contours, and Solution Points", fontsize=14)
    plt.xlabel("x1", fontsize=12)
    plt.ylabel("x2", fontsize=12)
    plt.grid(True)
    plt.xlim(min_x1, max_x1)
    plt.ylim(min_x2, max_x2)

    # Ensure margins are correct and everything is visible
    plt.tight_layout()

    # Save figure
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Base filename without extension
    base_filename = model_name.replace(".pkl", "")
    output_path = os.path.join(plots_dir, f"{base_filename}_plot.png")

    # Check if file already exists and find a new name if it does
    counter = 1
    while os.path.exists(output_path):
        # File exists, create a new filename with counter
        output_path = os.path.join(plots_dir, f"{base_filename}_plot_{counter}.png")
        counter += 1

    # Save with unique filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # bbox_inches ensures nothing is cut off
    print(f"Plot saved to: {output_path}")
    # plt.show()
    
    return output_path


def plot_from_calculation_results(data: Dict[str, Any]) -> None:
    """
    Plot results using pre-calculated data, but always load solution points from Excel.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing all calculation data
    """
    # Extract data
    model_name = data["model_name"]
    X1 = data["X1"]
    X2 = data["X2"]
    Z_obj = data["Z_obj"]
    feasible_mask = data["feasible_mask"]
    min_x1 = data["min_x1"]
    max_x1 = data["max_x1"]
    min_x2 = data["min_x2"]
    max_x2 = data["max_x2"]

    # Always load fresh solution points from Excel instead of using cached ones
    solution_points = get_solution_points(model_name)

    # Instead of using saved levels, recalculate them
    # Find the min and max values in Z_obj for better contour levels
    z_min, z_max = np.min(Z_obj), np.max(Z_obj)
    print(f"Objective function range: {z_min} to {z_max}")

    # Use helper function to calculate contour levels
    levels = calculate_contour_levels(z_min, z_max)
    print(f"Using {len(levels)} quadratic contour levels: {levels}")
    
    # Use common plotting function
    create_and_save_plot(
        X1, X2, Z_obj, feasible_mask, solution_points, 
        min_x1, max_x1, min_x2, max_x2, levels, model_name
    )


def plot_feasible_regions(
    model: Any,
    constraints: List[Dict[str, Any]],
    solution_points: Dict[str, Tuple[float, float]],
    grid_size: int = 1000,
    margin: float = 0.5,
    save_calculations: bool = True,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot feasible regions, objective function contours, and solution points.

    Parameters
    ----------
    model : Any
        The Pyomo model
    constraints : List[Dict[str, Any]]
        List of constraint data
    solution_points : Dict[str, Tuple[float, float]]
        Dictionary mapping strategy names to solution points
    grid_size : int
        Size of the grid for plotting
    margin : float
        Margin to add around solution points for plotting
    save_calculations : bool
        Whether to save calculation results to a file
    model_name : Optional[str]
        Name of the model file for saving results

    Returns
    -------
    Dict[str, Any]
        Dictionary with all calculation results for potential saving
    """
    # Use fixed range from -1 to 1 for both axes as requested
    min_x1, max_x1 = -1.0, 1.0
    min_x2, max_x2 = -1.0, 1.0

    # Create grid for plotting
    x1_grid = np.linspace(min_x1, max_x1, grid_size)
    x2_grid = np.linspace(min_x2, max_x2, grid_size)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    # Calculate total number of constraint evaluations needed
    points_per_grid = grid_size * grid_size

    # Count constraints across all disjunctions and disjuncts
    disjunctions = list(model.disjunctions)
    disjuncts = list(model.disjuncts)

    # First, count total constraints
    total_constraints = 0
    constraint_per_disjunct = {}

    for disjunction in disjunctions:
        for disjunct in disjuncts:
            # Get constraints for this disjunct
            disjunct_constraints = [
                c
                for c in constraints
                if c["disjunction"] == disjunction and c["disjunct"] == disjunct
            ]

            constraint_count = len(disjunct_constraints)
            constraint_per_disjunct[(disjunction, disjunct)] = constraint_count
            total_constraints += constraint_count

    total_evaluations = points_per_grid * total_constraints

    print(f"Grid size: {grid_size}x{grid_size} = {points_per_grid} points")
    print(f"Total constraints: {total_constraints}")
    print(f"Total constraint evaluations needed: {total_evaluations:,}")

    # Evaluate objective function on the grid
    print(f"\nEvaluating objective function on {grid_size}x{grid_size} grid...")
    Z_obj = np.zeros_like(X1)
    for i in range(grid_size):
        for j in range(grid_size):
            Z_obj[i, j] = evaluate_objective(model, X1[i, j], X2[i, j])

    # Plot feasible regions for each disjunct in each disjunction
    print(f"\nEvaluating constraints on {grid_size}x{grid_size} grid...")

    # Track global progress across all constraints
    global_evaluation_count = 0
    report_interval = 1000000
    next_report = report_interval  # Report every million evaluations

    # Initialize overall feasibility mask (all points start as potentially feasible)
    feasible_mask = np.ones_like(X1, dtype=bool)

    # Early termination stats
    early_disjunct_skips = 0
    early_point_skips = 0

    # Process each disjunction
    for disjunction_idx, disjunction in enumerate(disjunctions):
        print(f"Processing disjunction {disjunction_idx+1}/{len(disjunctions)}")

        # For each grid point, track if it's feasible for any disjunct in this disjunction
        disjunction_feasibility = np.zeros_like(X1, dtype=bool)

        # Process each disjunct
        for disjunct in disjuncts:
            # Get constraints for this disjunct
            disjunct_constraints = [
                c
                for c in constraints
                if c["disjunction"] == disjunction and c["disjunct"] == disjunct
            ]

            # Create mask for the current disjunct (points start as feasible)
            disjunct_mask = np.ones_like(X1, dtype=bool)

            # Apply each constraint to refine the disjunct mask
            for constraint_idx, constraint in enumerate(disjunct_constraints):
                # Only process points that are still potentially feasible for this disjunct
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Skip if this point is already known to be feasible for this disjunction
                        # or if it's already known to be infeasible for this disjunct
                        if disjunction_feasibility[i, j] or not disjunct_mask[i, j]:
                            early_disjunct_skips += 1
                            continue

                        # Skip if this point is already known to be infeasible overall
                        if not feasible_mask[i, j]:
                            early_point_skips += 1
                            continue

                        # Evaluate constraint: x^T Q x + c^T x + d <= 0
                        constraint_value = evaluate_constraint(
                            constraint["Q"], constraint["c"], constraint["d"], X1[i, j], X2[i, j]
                        )

                        # Count the evaluation
                        global_evaluation_count += 1

                        # Update disjunct feasibility
                        if constraint_value > 0:  # Constraint is violated
                            disjunct_mask[i, j] = False

                        # Report progress
                        if global_evaluation_count >= next_report:
                            progress_pct = global_evaluation_count / total_evaluations * 100
                            print(
                                f"    Overall progress: {global_evaluation_count:,}/"
                                f"{total_evaluations:,} evaluations ({progress_pct:.1f}%)"
                            )
                            next_report = global_evaluation_count + report_interval

            # Update disjunction feasibility with points that are feasible for this disjunct
            disjunction_feasibility = np.logical_or(disjunction_feasibility, disjunct_mask)

        # Update overall feasibility mask - points must be feasible in all disjunctions
        feasible_mask = np.logical_and(feasible_mask, disjunction_feasibility)

    # Report optimization stats
    print(f"Finished all {global_evaluation_count:,} constraint evaluations")
    print(f"Skipped {early_disjunct_skips:,} evaluations due to early disjunct termination")
    print(f"Skipped {early_point_skips:,} evaluations due to early point infeasibility detection")
    theoretical_evals = total_evaluations
    print(
        f"Optimization saved approximately {theoretical_evals - global_evaluation_count:,} evals "
        f"({(theoretical_evals - global_evaluation_count)/theoretical_evals*100:.1f}%)"
    )

    # Find the min and max values in Z_obj for better contour levels
    z_min, z_max = np.min(Z_obj), np.max(Z_obj)
    print(f"Objective function range: {z_min} to {z_max}")

    # Use helper function to calculate contour levels
    levels = calculate_contour_levels(z_min, z_max)
    print(f"Using {len(levels)} quadratic contour levels: {levels}")

    # Use model_name or a default if None is provided
    if model_name is None:
        model_name = "unnamed_model.pkl"

    # Use common plotting function
    create_and_save_plot(
        X1, X2, Z_obj, feasible_mask, solution_points, 
        min_x1, max_x1, min_x2, max_x2, levels, model_name
    )

    # Create calculation data dictionary
    calculation_data = {
        "model_name": model_name,
        "X1": X1,
        "X2": X2,
        "Z_obj": Z_obj,
        "feasible_mask": feasible_mask,
        "solution_points": solution_points,
        "min_x1": min_x1,
        "max_x1": max_x1,
        "min_x2": min_x2,
        "max_x2": max_x2,
        "levels": levels,
    }

    # Save calculation results if requested
    if save_calculations:
        save_calculation_results(
            model_name,
            X1,
            X2,
            Z_obj,
            feasible_mask,
            solution_points,
            min_x1,
            max_x1,
            min_x2,
            max_x2,
            levels,
        )

    return calculation_data


def check_for_cached_calculations(model_name: str) -> bool:
    """
    Check if cached calculation results exist for this model.

    Parameters
    ----------
    model_name : str
        Name of the model

    Returns
    -------
    bool
        True if cached data exists, False otherwise
    """
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    cache_dir = os.path.join(data_dir, "calculation_cache")
    file_path = os.path.join(cache_dir, f"{model_name.replace('.pkl', '')}_calc_data.pkl")

    return os.path.exists(file_path)


def main(model_name: str, use_cache: bool = True, save_cache: bool = True) -> None:
    """
    Main function to generate plots for a 2D model.

    Parameters
    ----------
    model_name : str
        Name of the model file
    use_cache : bool
        Whether to use cached calculation results if available
    save_cache : bool
        Whether to save calculation results for future use
    """
    # Check if we can use cached results
    if use_cache and check_for_cached_calculations(model_name):
        print(f"Found cached calculation results for {model_name}")
        data = load_calculation_results(model_name)
        plot_from_calculation_results(data)
        return

    # No cache or not using cache, do full calculation
    print(f"No cached data found or not using cache. Running full calculation for {model_name}")

    # Verify and load model
    model = verify_model(model_name)

    # Get solution points from Excel results
    solution_points = get_solution_points(model_name)

    # Extract disjunctive constraints
    constraints = extract_disjunctive_constraints(model)

    # Plot feasible regions and solution points, save calculations if requested
    plot_feasible_regions(
        model, constraints, solution_points, save_calculations=save_cache, model_name=model_name
    )

    print(f"Plots created for model: {model_name}")
    print(f"Solution points: {solution_points}")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Create 2D plots for quadratic disjunctive models")
    parser.add_argument(
        "model_name",
        nargs="?",
        default="model_no_mode_2025-05-01_13-48-10_dim2_disj2_disjper2_constper2_feas2_1.pkl",
        help="Name of the model file",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable use of cached calculations"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable saving calculations to cache"
    )

    args = parser.parse_args()

    # Run main function with provided arguments
    main(model_name=args.model_name, use_cache=not args.no_cache, save_cache=not args.no_save)
