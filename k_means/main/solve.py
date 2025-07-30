import io
import json
import os
import re
import time
from contextlib import redirect_stdout
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import dill as pickle
import k_means
import pandas as pd
import pyomo.environ as pyo
import pyomo.gdp.plugins.hull_exact
import pyomo.gdp.plugins.hull_reduced_y

possible_modes = ["approximation", "exact", "reduced_power_y", "no_mode"]

plugins = [pyomo.gdp.plugins.hull_exact, pyomo.gdp.plugins.hull_reduced_y]

# Default parameters for k-means
parameters = {
    "n_dimensions": 2,
    "n_clusters": 3,
    "n_points": 10,
    "coord_range": (0.0, 10.0),
}

reformulation_strategies = ["gdp.hull", "gdp.bigm", "gdp.hull_exact", "gdp.hull_reduced_y"]


def save_model(model: pyo.ConcreteModel, directory: str, filename: str = "model.pkl") -> str:
    """
    Save a Pyomo model to a pickle file.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model to save
    directory : str
        Directory to save the model
    filename : str, optional
        Name of the file, by default "model.pkl"

    Returns
    -------
    str
        Path to the saved model file
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {file_path}")
    return file_path


def load_model(file_path: str) -> pyo.ConcreteModel:
    """
    Load a Pyomo model from a pickle file.

    Parameters
    ----------
    file_path : str
        Path to the saved model file

    Returns
    -------
    pyo.ConcreteModel
        The loaded Pyomo model
    """
    with open(file_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded from {file_path}")
    return model


def parse_root_relaxation(
    output_log_path: str, solver: str = "gurobi", subsolver: Optional[str] = None
) -> Optional[float]:
    """
    Parse the output log file to extract the root relaxation objective value.

    Parameters
    ----------
    output_log_path : str
        Path to the output log file
    solver : str, optional
        The solver used, by default "gurobi"
    subsolver : Optional[str], optional
        The subsolver used, by default None

    Returns
    -------
    Optional[float]
        The root relaxation objective value if found, None otherwise
    """
    if not os.path.exists(output_log_path):
        print(f"Warning: Output log file not found at {output_log_path}")
        return None

    try:
        with open(output_log_path, "r") as f:
            log_content = f.read()

        # Check if this is BARON output (through GAMS)
        if solver.lower() == "gams" and subsolver and subsolver.lower() == "baron":
            # For BARON, find the first iteration table and its first row
            header_match = re.search(
                r"Iteration\s+Time[^\\n]*Lower bound\s+Upper bound\s+Progress", log_content
            )

            if header_match:
                header_end = header_match.end()
                post_header = log_content[header_end:]
                lines = post_header.strip().split("\n")

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("==="):
                        continue

                    match = re.search(
                        r"\s*\S+\s+\S+\s+\S+\s+([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)\s+", line
                    )
                    if match:
                        lower_bound = float(match.group(1))
                        print(f"Found BARON root relaxation (lower bound): {lower_bound}")
                        return lower_bound

                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            lower_bound = float(parts[3])
                            print(
                                f"Found BARON root relaxation \
                                    (lower bound from column): {lower_bound}"
                            )
                            return lower_bound
                        except ValueError:
                            numeric_parts = []
                            for part in parts:
                                try:
                                    value = float(part)
                                    numeric_parts.append(value)
                                except ValueError:
                                    pass

                            if len(numeric_parts) >= 3:
                                lower_bound = numeric_parts[2]
                                print(
                                    f"Found BARON root relaxation \
                                        (lower bound from numeric extraction): {lower_bound}"
                                )
                                return lower_bound

            # Check for "Problem solved during preprocessing" case
            preprocessing_match = re.search(r"Problem solved during preprocessing", log_content)
            if preprocessing_match:
                best_possible_match = re.search(
                    r"Best possible = ([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", log_content
                )
                if best_possible_match:
                    best_possible = float(best_possible_match.group(1))
                    print(f"Found BARON root relaxation (preprocessing): {best_possible}")
                    return best_possible

                lower_bound_match = re.search(
                    r"Lower bound is\s+([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", log_content
                )
                if lower_bound_match:
                    lower_bound = float(lower_bound_match.group(1))
                    print(f"Found BARON root relaxation (lower bound): {lower_bound}")
                    return lower_bound

            print("BARON root relaxation value not found in output log")
            return None
        elif solver.lower() == "gams" and subsolver and subsolver.lower() == "ipopth":
            # For IPOPT through GAMS
            objective_match = re.search(
                r"Objective\s*\.\.\.\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", log_content
            )
            if objective_match:
                objective_value = float(objective_match.group(1))
                print(f"Found IPOPT objective value: {objective_value}")
                return objective_value

            ipopt_f_match = re.search(
                r"IPOPT\s+\d+\s+\d+\.\d+\s+\d+\.\d+\s+f\s+" r"([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)",
                log_content,
            )
            if ipopt_f_match:
                f_value = float(ipopt_f_match.group(1))
                print(f"Found IPOPT f value: {f_value}")
                return f_value

            ipopt_lines = re.findall(r"IPOPT.*", log_content)
            if ipopt_lines:
                last_ipopt_line = ipopt_lines[-1]
                numeric_values = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", last_ipopt_line)
                if numeric_values:
                    if len(numeric_values) >= 2:
                        obj_value = float(numeric_values[-2])
                        print(f"Found IPOPT objective value from iteration table: {obj_value}")
                        return obj_value
                    elif len(numeric_values) == 1:
                        obj_value = float(numeric_values[0])
                        print(f"Found single IPOPT numeric value: {obj_value}")
                        return obj_value

            print("IPOPT root relaxation value not found in output log")
            return None
        elif solver.lower() == "scip" or (
            solver.lower() == "gams" and subsolver and subsolver.lower() == "scip"
        ):
            # For SCIP
            trivial_preprocessing = re.search(
                r"problem is solved by trivial preprocessing", log_content
            )
            if trivial_preprocessing:
                obj_value_match = re.search(
                    r"objective value\s*:\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", log_content
                )
                if obj_value_match:
                    obj_value = float(obj_value_match.group(1))
                    print(f"Found SCIP root relaxation (trivial preprocessing): {obj_value}")
                    return obj_value

            lp_relax_match = re.search(
                r"LP0\s+\(\d+r,\s*\d+c\)\s*:\s*(?:opt\.|infeas\.|unbounded)\s*\[([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)",
                log_content,
            )
            if lp_relax_match:
                lp_value = float(lp_relax_match.group(1))
                print(f"Found SCIP root relaxation (LP0): {lp_value}")
                return lp_value

            dual_bound_match = re.search(
                r"root node[\s\S]*?dual bound\s*:\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", log_content
            )
            if dual_bound_match:
                dual_bound = float(dual_bound_match.group(1))
                print(f"Found SCIP root relaxation (dual bound): {dual_bound}")
                return dual_bound

            dual_bound_summary_match = re.search(
                r"Dual Bound\s*:\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", log_content
            )
            if dual_bound_summary_match:
                dual_bound = float(dual_bound_summary_match.group(1))
                print(f"Found SCIP root relaxation (dual bound summary): {dual_bound}")
                return dual_bound

            print("SCIP root relaxation not found in log")
            return None
        else:
            # For Gurobi, try multiple patterns
            match = re.search(
                r"Root relaxation: objective\s+([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", log_content
            )
            if match:
                root_relaxation_value = float(match.group(1))
                print(f"Found Gurobi root relaxation (standard): {root_relaxation_value}")
                return root_relaxation_value

            cutoff_match = re.search(r"Root relaxation: cutoff", log_content)
            if cutoff_match:
                nodes_table_match = re.search(
                    r"Nodes\s+\|\s+Current Node\s+\|\s+Objective Bounds", log_content
                )
                if nodes_table_match:
                    table_end = nodes_table_match.end()
                    table_content = log_content[table_end:]
                    lines = table_content.strip().split("\n")

                    for line in lines:
                        if re.search(r"^\s+\d+\s+\d+", line):
                            bd_match = re.search(
                                r"\|\s+.*?\s+.*?\s+\|\s+.*?\s+([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)\s+",
                                line,
                            )
                            if bd_match:
                                bound_value = float(bd_match.group(1))
                                print(
                                    f"Found Gurobi root relaxation \
                                        (cutoff table regex): {bound_value}"
                                )
                                return bound_value

                            values = []
                            for part in line.split():
                                try:
                                    values.append(float(part))
                                except ValueError:
                                    pass

                            if len(values) >= 2:
                                bound_value = values[-2]
                                print(f"Found Gurobi root relaxation (cutoff table): {bound_value}")
                                return bound_value
                            break

            single_node_match = re.search(r"Explored \d+ nodes?", log_content)
            best_bound_match = re.search(
                r"[Bb]est bound ([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", log_content
            )

            if single_node_match and best_bound_match:
                bound_value = float(best_bound_match.group(1))
                print(f"Found Gurobi root relaxation (best bound): {bound_value}")
                return bound_value

            gap_line_match = re.search(
                r"[Bb]est objective [-+]?\d*\.\d+(?:[eE][-+]?\d+)?,"
                + r" best bound ([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)",
                log_content,
            )

            if gap_line_match:
                bound_value = float(gap_line_match.group(1))
                print(f"Found Gurobi root relaxation (from gap line): {bound_value}")
                return bound_value

            print("Gurobi root relaxation not found in log")
            return None
    except Exception as e:
        print(f"Error parsing output log: {str(e)}")
        return None


def calculate_root_relaxation_gap(
    objective_value: Optional[float], root_relaxation_value: Optional[float]
) -> Optional[float]:
    """
    Calculate the relative gap between the final objective value and root relaxation value.

    Parameters
    ----------
    objective_value : Optional[float]
        Final objective value of the problem
    root_relaxation_value : Optional[float]
        Root relaxation objective value

    Returns
    -------
    Optional[float]
        Relative gap as a percentage: |objective - root_relaxation| / |objective| * 100%
        Returns None if either value is None or objective is zero
    """
    if objective_value is None or root_relaxation_value is None:
        return None

    if objective_value == 0:
        return None

    abs_gap = abs(objective_value - root_relaxation_value)
    rel_gap = abs_gap / abs(objective_value) * 100.0

    return rel_gap


def save_results(
    model: pyo.ConcreteModel,
    result: Any,
    duration: float,
    results_dir: str,
    strategy: str,
    mode: str,
    existing_model_name: Optional[str] = None,
    custom_filename: Optional[str] = None,
    solver: str = "gams",
    subsolver: Optional[str] = "gurobi",
    is_relaxation: bool = False,
    relaxation_gap: Optional[float] = None,
    absolute_gap: Optional[float] = None,
    root_relaxation_value: Optional[float] = None,
    results_cache: Optional[Dict] = None,
) -> None:
    """
    Save model parameters, solution, and performance data to a JSON file.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The solved Pyomo model
    result : Any
        The solver result
    duration : float
        Solution time in seconds
    results_dir : str
        Directory to save results
    strategy : str
        The reformulation strategy used
    mode : str
        The mode used for solving
    existing_model_name : Optional[str], optional
        Name of existing model to load, by default None
    custom_filename : Optional[str], optional
        Custom filename to use when save_only is True, by default None
    solver : str, optional
        The main solver used, by default "gams"
    subsolver : Optional[str], optional
        The subsolver used (if applicable), by default "gurobi"
    is_relaxation : bool, optional
        Whether this is a relaxed model, by default False
    relaxation_gap : Optional[float], optional
        Relative gap between original and relaxed solution (percentage), by default None
    absolute_gap : Optional[float], optional
        Absolute gap between original and relaxed solution, by default None
    root_relaxation_value : Optional[float], optional
        Root relaxation objective value from the solver output, by default None
    results_cache : Optional[Dict], optional
        Cache for storing results for gap calculation, by default None
    """
    # Extract model parameters
    model_params = {
        "n_dimensions": len(model.dimensions),
        "n_clusters": len(model.clusters),
        "n_points": len(model.points),
        "coord_range_lower": model.coord_range_lower.value,
        "coord_range_upper": model.coord_range_upper.value,
    }

    # Extract solution details
    solution: Dict[str, Any] = {}

    # First try to extract solution for optimal case
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        # Get optimal center coordinates and distances
        center_coordinates = {}
        for k in model.clusters:
            center_coordinates[f"cluster_{k}"] = {}
            for j in model.dimensions:
                center_coordinates[f"cluster_{k}"][f"dim_{j}"] = pyo.value(
                    model.center_coordinates[k, j]
                )

        distances = {str(i): pyo.value(model.distance[i]) for i in model.points}
        objective_value = pyo.value(model.obj)

        solution = {
            "status": "optimal",
            "center_coordinates": center_coordinates,
            "distances": distances,
            "objective_value": objective_value,
        }
    else:
        # For non-optimal cases, check if we can extract a solution anyway
        try:
            # Try to extract values
            center_coordinates = {}
            for k in model.clusters:
                center_coordinates[f"cluster_{k}"] = {}
                for j in model.dimensions:
                    center_coordinates[f"cluster_{k}"][f"dim_{j}"] = pyo.value(
                        model.center_coordinates[k, j]
                    )

            distances = {str(i): pyo.value(model.distance[i]) for i in model.points}
            objective_value = pyo.value(model.obj)

            # If we got this far, we have a valid solution to save
            solution = {
                "status": str(result.solver.termination_condition),
                "center_coordinates": center_coordinates,
                "distances": distances,
                "objective_value": objective_value,
            }
            print(f"Successfully extracted solution with objective value: {objective_value}")
        except Exception as e:
            print(f"Failed to extract solution: {str(e)}")
            solution = {
                "status": str(result.solver.termination_condition),
                "center_coordinates": None,
                "distances": None,
                "objective_value": None,
            }

    # Get bound from solver results
    bound = None
    try:
        if hasattr(result, "solver") and hasattr(result.solver, "dual_bound"):
            # SCIP stores the dual bound here, especially for time limit cases
            bound = result.solver.dual_bound
        elif hasattr(result, "problem") and hasattr(result.problem, "lower_bound"):
            bound = result.problem.lower_bound
        elif hasattr(result, "solver") and hasattr(result.solver, "best_objective_bound"):
            bound = result.solver.best_objective_bound
        elif hasattr(result, "solver") and hasattr(result.solver, "lower_bound"):
            bound = result.solver.lower_bound
        elif hasattr(result, "lower_bound"):
            bound = result.lower_bound
    except Exception as e:
        print(f"Warning: Could not extract bound from solver result: {str(e)}")
        bound = None

    solution["lower_bound"] = bound

    # Calculate gaps with proper error handling
    try:
        objective_value = solution.get("objective_value")
        if objective_value is not None and bound is not None:
            solution["absolute_gap"] = objective_value - bound
            if abs(bound) != 0:
                solution["relative_gap"] = solution["absolute_gap"] / abs(bound)
            else:
                solution["relative_gap"] = None
        else:
            solution["absolute_gap"] = None
            solution["relative_gap"] = None
    except Exception as e:
        print(f"Warning: Could not calculate gaps: {str(e)}")
        solution["absolute_gap"] = None
        solution["relative_gap"] = None

    # Calculate root relaxation gap if possible
    root_relaxation_gap = None
    if solution.get("objective_value") is not None and root_relaxation_value is not None:
        root_relaxation_gap = calculate_root_relaxation_gap(
            solution["objective_value"], root_relaxation_value
        )
        if root_relaxation_gap is not None:
            print(f"Root relaxation gap: {root_relaxation_gap:.2f}%")

    # Performance metrics - only include relaxation gaps for the original problem
    performance = {
        "solution_time_seconds": duration,
        "solver_status": str(result.solver.status),
        "termination_condition": str(result.solver.termination_condition),
        "solver": solver,
        "subsolver": subsolver,
        "is_relaxation": is_relaxation,
        "root_relaxation_value": root_relaxation_value,
        "root_relaxation_gap_percent": root_relaxation_gap,
    }

    # Only add relaxation gaps to original model data
    if not is_relaxation:
        performance["relaxation_gap_percent"] = relaxation_gap
        performance["absolute_gap"] = absolute_gap

    # Compile all results
    all_results = {
        "model_parameters": model_params,
        "solution": solution,
        "performance": performance,
    }

    # Save to JSON file
    problem_type = "relaxation" if is_relaxation else "original"
    results_file = os.path.join(results_dir, f"solution_data_{problem_type}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {results_file}")

    # After saving JSON, check if it's a maxTimeLimit case for original problem
    if not is_relaxation and str(result.solver.termination_condition) == "maxTimeLimit":
        try:
            relaxed_dir = os.path.join(os.path.dirname(results_dir), "relaxed")
            relaxed_file = os.path.join(relaxed_dir, "solution_data_relaxation.json")

            if os.path.exists(relaxed_file):
                # Read the relaxation data to get updated gaps
                with open(relaxed_file, "r") as f:
                    relaxed_data = json.load(f)

                # Extract the gaps if available
                relaxation_gap = relaxed_data.get("performance", {}).get("relaxation_gap_percent")
                absolute_gap = relaxed_data.get("performance", {}).get("absolute_gap")
                print(
                    f"Found existing relaxation data with gaps: {relaxation_gap}%, {absolute_gap}"
                )
        except Exception as e:
            print(f"Error reading relaxation data: {str(e)}")

    # Save to Excel file
    save_to_excel(
        model_params,
        solution,
        performance,
        strategy,
        mode,
        existing_model_name,
        custom_filename,
        solver,
        subsolver,
        is_relaxation,
        relaxation_gap,
        absolute_gap,
        root_relaxation_value,
        root_relaxation_gap,
        results_cache=results_cache,
    )


def save_to_excel(
    model_params: Dict[str, Any],
    solution: Dict[str, Any],
    performance: Dict[str, Any],
    strategy: str,
    mode: str,
    existing_model_name: Optional[str] = None,
    custom_filename: Optional[str] = None,
    solver: str = "gams",
    subsolver: Optional[str] = "gurobi",
    is_relaxation: bool = False,
    relaxation_gap: Optional[float] = None,
    absolute_gap: Optional[float] = None,
    root_relaxation_value: Optional[float] = None,
    root_relaxation_gap: Optional[float] = None,
    results_cache: Optional[Dict] = None,
) -> None:
    """
    Save results to an Excel file, creating it if it doesn't exist.
    Appends new data without deleting existing data.

    Parameters
    ----------
    model_params : Dict[str, Any]
        Dictionary of model parameters
    solution : Dict[str, Any]
        Dictionary of solution information
    performance : Dict[str, Any]
        Dictionary of performance metrics
    strategy : str
        Reformulation strategy used
    mode : str
        Mode used for solving
    existing_model_name : Optional[str], optional
        Name of existing model to load, by default None
    custom_filename : Optional[str], optional
        Custom filename to use when save_only is True, by default None
    solver : str, optional
        The main solver used, by default "gams"
    subsolver : Optional[str], optional
        The subsolver used (if applicable), by default "gurobi"
    is_relaxation : bool, optional
        Whether this is a relaxed model, by default False
    relaxation_gap : Optional[float], optional
        Relative gap between original and relaxed solution (percentage), by default None
    absolute_gap : Optional[float], optional
        Absolute gap between original and relaxed solution, by default None
    root_relaxation_value : Optional[float], optional
        Root relaxation objective value from the solver output, by default None
    root_relaxation_gap : Optional[float], optional
        Relative gap between final objective and root relaxation (percentage), by default None
    results_cache : Optional[Dict], optional
        Cache for storing results for gap calculation, by default None
    """
    # Path to Excel file
    excel_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    excel_path = os.path.join(excel_dir, "results.xlsx")

    # Ensure data directory exists
    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)

    # Prepare data for the new row
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Determine model name
    model_name = existing_model_name
    if model_name is None:
        model_name = custom_filename
    if model_name is None:
        model_name = f"model_{mode}_{current_time}.pkl"

    # Flatten center_coordinates into a single string if they exist
    center_coordinates_str = ""
    if solution.get("center_coordinates"):
        center_coords = solution["center_coordinates"]
        coord_parts = []
        for cluster_key, cluster_coords in center_coords.items():
            for dim_key, value in cluster_coords.items():
                coord_parts.append(f"{cluster_key}_{dim_key}={value:.6f}")
        center_coordinates_str = ", ".join(coord_parts)

    # Flatten distances into a single string if they exist
    distances_str = ""
    if solution.get("distances"):
        distances = solution["distances"]
        distance_parts = [f"d{k}={v:.6f}" for k, v in distances.items()]
        distances_str = ", ".join(distance_parts)

    # Check if we have more accurate gap data from cache
    if (
        results_cache
        and is_relaxation
        and "original" in results_cache
        and "relaxation" in results_cache
    ):
        try:
            orig_obj = results_cache["original"]["objective"]
            relax_obj = results_cache["relaxation"]["objective"]

            # Recalculate gaps
            abs_gap, rel_gap = calculate_gaps(orig_obj, relax_obj)

            if rel_gap is not None:
                relaxation_gap = rel_gap
                absolute_gap = abs_gap
                print(f"Using recalculated gaps from cache: {rel_gap}%, {abs_gap}")
        except Exception as e:
            print(f"Error recalculating gaps from cache: {str(e)}")

    # Create a dictionary for the new row
    new_row = {
        "Run Time": run_time,
        "Mode": mode,
        "Strategy": strategy,
        "Model Name": model_name,
        "Problem Type": "Relaxation" if is_relaxation else "Original",
        "Duration (sec)": performance["solution_time_seconds"],
        "Status": solution["status"],
        "Objective Value": solution["objective_value"],
        "Lower Bound": solution["lower_bound"],
        "Bound Absolute Gap": solution["absolute_gap"],
        "Bound Relative Gap (%)": solution["relative_gap"],
        "Root Relaxation Value": root_relaxation_value,
        "Root Relaxation Gap (%)": root_relaxation_gap,
        "Relative Gap (%)": relaxation_gap if relaxation_gap is not None else None,
        "Absolute Gap": absolute_gap if absolute_gap is not None else None,
        "Center Coordinates": center_coordinates_str,
        "Distances": distances_str,
        "Solver": solver,
        "Subsolver": subsolver if subsolver else "None",
        # Model parameters
        "n_dimensions": model_params["n_dimensions"],
        "n_clusters": model_params["n_clusters"],
        "n_points": model_params["n_points"],
        "coord_range_lower": model_params["coord_range_lower"],
        "coord_range_upper": model_params["coord_range_upper"],
    }

    # Convert to DataFrame
    df_new = pd.DataFrame([new_row])

    # Check if file exists
    if os.path.exists(excel_path):
        # File exists, read it and append
        df_existing = pd.read_excel(excel_path)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # File doesn't exist, create new
        df_updated = df_new

    # Save to Excel
    df_updated.to_excel(excel_path, index=False)
    print(f"Results appended to {excel_path}")


def calculate_gaps(
    original_obj: Optional[float], relaxed_obj: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate both absolute and relative relaxation gaps.

    Parameters
    ----------
    original_obj : Optional[float]
        Objective value of the original problem
    relaxed_obj : Optional[float]
        Objective value of the relaxed problem

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        A tuple containing (absolute_gap, relative_gap_percent)
        - absolute_gap: |original - relaxed|
        - relative_gap_percent: |original - relaxed| / |original| * 100%
    """
    if original_obj is None or relaxed_obj is None:
        return None, None

    # Calculate absolute gap
    abs_gap = abs(original_obj - relaxed_obj)

    # Calculate relative gap as a percentage
    rel_gap = None
    if original_obj != 0:
        rel_gap = abs_gap / abs(original_obj) * 100.0

    return abs_gap, rel_gap


def solve_with_solver(
    model: pyo.ConcreteModel,
    solver: str,
    subsolver: Optional[str],
    time_limit: int,
    results_dir: str,
    tee: bool = True,
) -> Tuple[Any, float]:
    """
    Solve a model with the specified solver and configuration.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The model to solve
    solver : str
        The solver to use ('gams', 'gurobi', or 'gurobi_persistent')
    subsolver : Optional[str]
        The subsolver to use if using GAMS or 'persistent' for persistent solver
    time_limit : int
        Time limit in seconds
    results_dir : str
        Directory to store results
    tee : bool, optional
        Whether to display solver output, by default True

    Returns
    -------
    Tuple[Any, float]
        Tuple containing (solver_result, duration)
    """
    if solver.lower() == "gams":
        opt = pyo.SolverFactory("gams")

        # Set up options based on subsolver
        if subsolver and subsolver.lower() == "baron":
            # BARON options through GAMS
            options_gams = ["$onecho > baron.opt", "$offecho", "GAMS_MODEL.optfile=1"]
            solver_name = "baron"
        elif subsolver and subsolver.lower() == "gurobi":
            # Default to Gurobi with GAMS
            options_gams = [
                "$onecho > gurobi.opt",
                "NonConvex 2",
                "$offecho",
                "GAMS_MODEL.optfile=1",
            ]
            solver_name = "gurobi"
        elif subsolver and subsolver.lower() == "ipopth":
            # IPOPT (interior point optimizer) through GAMS
            options_gams = [
                "$onecho > ipopt.opt",
                "max_iter 10000",
                "mu_strategy adaptive",
                "tol 1e-6",
                "$offecho",
                "GAMS_MODEL.optfile=1",
            ]
            solver_name = "ipopt"
        elif subsolver and subsolver.lower() == "scip":
            # SCIP through GAMS
            options_gams = [
                "$onecho > scip.opt",
                "limits/time = " + str(time_limit),
                "numerics/feastol = 1e-6",
                "numerics/epsilon = 1e-6",
                "numerics/sumepsilon = 1e-6",
                "display/verblevel = 4",
                "$offecho",
                "GAMS_MODEL.optfile=1",
            ]
            solver_name = "scip"
        else:
            print(f"Using unsupported GAMS subsolver: {subsolver}")
            raise ValueError(f"Unsupported GAMS subsolver: {subsolver}")

        start = time.time()
        result = opt.solve(
            model,
            solver=solver_name,
            tee=tee,
            keepfiles=True,
            tmpdir=results_dir,
            symbolic_solver_labels=True,
            add_options=[
                f"option reslim={time_limit};",
                "option threads=1;",
                "option optcr=1e-6;",
                "option optca=0;",
                *options_gams,
            ],
        )
    elif solver.lower() == "gurobi" and (not subsolver or subsolver.lower() != "persistent"):
        # Direct Gurobi solver
        opt = pyo.SolverFactory("gurobi")

        # Set Gurobi parameters
        opt.options["NonConvex"] = 2
        opt.options["TimeLimit"] = time_limit
        opt.options["Threads"] = 1
        opt.options["MIPGap"] = 1e-6
        opt.options["MIPGapAbs"] = 0

        start = time.time()
        result = opt.solve(
            model,
            tee=tee,
            symbolic_solver_labels=True,
        )
    elif solver.lower() == "gurobi" and subsolver and subsolver.lower() == "persistent":
        # Gurobi persistent solver
        print("Using Gurobi persistent solver")
        opt = pyo.SolverFactory("gurobi_persistent")

        # Load the model into the solver
        opt.set_instance(model)

        # Set Gurobi parameters
        opt.options["NonConvex"] = 2
        opt.options["TimeLimit"] = time_limit
        opt.options["Threads"] = 1
        opt.options["MIPGap"] = 1e-6
        opt.options["MIPGapAbs"] = 0

        start = time.time()
        result = opt.solve(
            tee=tee,
            save_results=True,  # This ensures results are saved to the model
            load_solutions=True,  # This ensures the solution is loaded back into the model
        )

        # Clean up the persistent solver
        opt.client_to_solver = {}
        opt.solver_model = None
    elif solver.lower() == "scip":
        # Direct SCIP solver
        opt = pyo.SolverFactory("scip")

        # Set SCIP parameters
        opt.options["limits/time"] = time_limit
        opt.options["numerics/feastol"] = 1e-6
        opt.options["numerics/epsilon"] = 1e-6
        opt.options["numerics/sumepsilon"] = 1e-6
        opt.options["display/verblevel"] = 4
        opt.options["parallel/maxnthreads"] = 1

        start = time.time()
        result = opt.solve(
            model,
            tee=tee,
            symbolic_solver_labels=True,
            keepfiles=True,
        )
    else:
        raise ValueError(f"Unsupported solver: {solver} with subsolver: {subsolver}")

    end = time.time()
    duration = end - start

    return result, duration


def solve_model(
    model: Optional[pyo.ConcreteModel],
    reformulation_strategies: List[str],
    mode: str = "approximation",
    time_limit: int = 3600,
    existing_model_name: Optional[str] = None,
    save_only: bool = False,
    custom_filename: Optional[str] = None,
    solver: str = "gams",
    subsolver: Optional[str] = "gurobi",
    calculate_relaxation_gap: bool = False,
    relaxation_only: bool = False,
) -> Optional[str]:
    """
    Solve the model using the specified solver and subsolver.

    Parameters
    ----------
    model : Optional[pyo.ConcreteModel]
        The Pyomo model to solve (can be None if existing_model_name is provided)
    reformulation_strategies : List[str]
        List of reformulation strategies to use
    mode : str, optional
        Mode for solving, by default "approximation"
    time_limit : int, optional
        Time limit in seconds, by default 3600
    existing_model_name : Optional[str], optional
        Name of existing model to load, by default None
    save_only : bool, optional
        If True, just save the model without solving, by default False
    custom_filename : Optional[str], optional
        Custom filename to use when save_only is True, by default None
    solver : str, optional
        Solver to use, by default "gams"
    subsolver : Optional[str], optional
        Subsolver to use (if using GAMS), by default "gurobi".
        Use "persistent" with "gurobi" solver to use gurobi_persistent.
    calculate_relaxation_gap : bool, optional
        Whether to calculate relaxation gap, by default False
    relaxation_only : bool, optional
        Whether to solve only the relaxation, by default False

    Returns
    -------
    Optional[str]
        If save_only is True, returns the model filename, otherwise None
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # data is parent directory in k_means\data
    results_dir_parent = os.path.join(os.path.dirname(os.getcwd()), "data")

    # Ensure data directory exists
    if not os.path.exists(results_dir_parent):
        os.makedirs(results_dir_parent)

    # Check if using existing model or creating a new one
    models_dir = os.path.join(results_dir_parent, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if existing_model_name is not None:
        # Load existing model
        model_path = os.path.join(models_dir, existing_model_name)
        if os.path.exists(model_path):
            model_for_cloning = load_model(model_path)
            print(f"Using existing model: {existing_model_name}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    else:
        # No existing model provided, use the passed model
        if model is None:
            raise ValueError("Either model or existing_model_name must be provided")
        model_for_cloning = model

        # Determine the model filename
        if save_only and custom_filename:
            # Use custom filename if provided for save_only mode
            model_filename = custom_filename
        else:
            # Use standard naming convention
            model_filename = f"model_{mode}_{current_time}.pkl"

        save_model(model_for_cloning, models_dir, model_filename)
        print(f"Model saved as: {model_filename}")

        # Track the filename for result reporting
        if existing_model_name is None:
            existing_model_name = model_filename

        # Exit if we're only saving the model
        if save_only:
            return model_filename

    # Skip solving if no reformulation strategies provided
    if not reformulation_strategies:
        return None

    for strategy in reformulation_strategies:
        # Create results directory with solver info
        # Special handling for persistent solver naming
        if solver.lower() == "gurobi" and subsolver and subsolver.lower() == "persistent":
            solver_dir = "gurobi_persistent"
        else:
            solver_dir = f"{solver}_{subsolver if subsolver else 'direct'}"

        base_results_dir = os.path.join(
            results_dir_parent, f"{solver_dir}_{strategy}", mode, current_time
        )
        if not os.path.exists(base_results_dir):
            os.makedirs(base_results_dir)

        # Get a fresh copy of the model for this strategy
        model = model_for_cloning.clone()

        # Apply the reformulation strategy
        print(f"Applying reformulation strategy: {strategy}")
        pyo.TransformationFactory(strategy).apply_to(model)

        # Create a shared data structure to store all results for this strategy
        strategy_results: Dict[str, Dict[str, Any]] = {
            "original": {
                "objective_value": None,
                "status": None,
                "root_relaxation_value": None,
                "result": None,  # Full solver result
                "duration": None,
                "model": None,  # Solved model
            },
            "relaxation": {
                "objective_value": None,
                "status": None,
                "root_relaxation_value": None,
                "result": None,  # Full solver result
                "duration": None,
                "model": None,  # Solved model
            },
            "gaps": {
                "absolute_gap": None,
                "relaxation_gap_percent": None,
            },
        }

        # Solve the original problem if not in relaxation_only mode
        if not relaxation_only:
            # Create a directory specifically for original results
            results_dir = os.path.join(base_results_dir, "original")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # Path to the file where you want to save the output
            output_file = os.path.join(results_dir, "output_log.txt")

            # Create a StringIO buffer to capture the output
            buffer = io.StringIO()

            # Redirect stdout to the buffer
            with redirect_stdout(buffer):
                print(f"Solving original (integer) problem with strategy {strategy}...")

                # Use the helper function to solve the model
                result, duration = solve_with_solver(
                    model, solver, subsolver, time_limit, results_dir
                )

                print(f"Original problem solved. Time taken: {duration} seconds")

                # Try to extract objective value
                try:
                    original_obj_value = pyo.value(model.obj)
                    print(f"Original objective value: {original_obj_value}")

                    # Store in results structure
                    strategy_results["original"]["objective_value"] = original_obj_value
                    strategy_results["original"]["status"] = str(
                        result.solver.termination_condition
                    )
                    strategy_results["original"]["result"] = result
                    strategy_results["original"]["duration"] = duration
                    strategy_results["original"]["model"] = model

                except Exception as e:
                    print(f"Could not extract original objective value: {str(e)}")

                # Write the captured output to the file
                with open(output_file, "w") as f:
                    f.write(buffer.getvalue())

                # Parse the output log for root relaxation value
                root_relaxation_value = parse_root_relaxation(output_file, solver, subsolver)
                strategy_results["original"]["root_relaxation_value"] = root_relaxation_value

                # Save original problem results - gaps will be calculated later
                save_results(
                    model,
                    result,
                    duration,
                    results_dir,
                    strategy,
                    mode,
                    existing_model_name,
                    custom_filename,
                    solver,
                    subsolver,
                    is_relaxation=False,
                    root_relaxation_value=root_relaxation_value,
                )

                # Save pretty-printed model
                save_model_pprint(model, results_dir, is_relaxation=False)

        # Solve the relaxation if requested
        if calculate_relaxation_gap or relaxation_only:
            # Create a directory specifically for relaxed results
            results_dir = os.path.join(base_results_dir, "relaxed")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # Path to the file where you want to save the output
            output_file = os.path.join(results_dir, "output_log.txt")

            # Create a StringIO buffer to capture the output
            buffer = io.StringIO()

            # Redirect stdout to the buffer
            with redirect_stdout(buffer):
                # Create a copy of the model for relaxation
                relaxed_model = model.clone()

                # Apply relaxation transformation
                print(f"Applying relaxation to model with strategy {strategy}...")
                pyo.TransformationFactory("core.relax_integer_vars").apply_to(relaxed_model)

                print(f"Solving relaxed problem with strategy {strategy}...")

                # Use the helper function to solve the relaxed model
                relaxed_result, relaxed_duration = solve_with_solver(
                    relaxed_model, solver, subsolver, time_limit, results_dir
                )

                print(f"Relaxed problem solved. Time taken: {relaxed_duration} seconds")

                # Try to extract values
                try:
                    relaxed_obj_value = pyo.value(relaxed_model.obj)
                    print(f"Relaxed objective value: {relaxed_obj_value}")

                    # Store in results structure
                    strategy_results["relaxation"]["objective_value"] = relaxed_obj_value
                    strategy_results["relaxation"]["status"] = str(
                        relaxed_result.solver.termination_condition
                    )
                    strategy_results["relaxation"]["result"] = relaxed_result
                    strategy_results["relaxation"]["duration"] = relaxed_duration
                    strategy_results["relaxation"]["model"] = relaxed_model

                except Exception as e:
                    print(f"Could not extract objective value from relaxed model: {str(e)}")

                # Write the captured output to the file
                with open(output_file, "w") as f:
                    f.write(buffer.getvalue())

                # Parse the output log for root relaxation value
                relaxed_root_relaxation_value = parse_root_relaxation(
                    output_file, solver, subsolver
                )
                strategy_results["relaxation"]["root_relaxation_value"] = (
                    relaxed_root_relaxation_value
                )

                # Calculate relaxation gap if we have both original and relaxed values
                if (
                    strategy_results["original"]["objective_value"] is not None
                    and strategy_results["relaxation"]["objective_value"] is not None
                ):
                    absolute_gap, relaxation_gap = calculate_gaps(
                        strategy_results["original"]["objective_value"],
                        strategy_results["relaxation"]["objective_value"],
                    )

                    # Store in the gaps section of our results structure
                    strategy_results["gaps"]["absolute_gap"] = absolute_gap
                    strategy_results["gaps"]["relaxation_gap_percent"] = relaxation_gap

                    if relaxation_gap is not None:
                        print(
                            f"Relaxation gap: {relaxation_gap:.2f}%, "
                            f"Absolute gap: {absolute_gap:.6f}"
                        )
                    else:
                        print(
                            f"Absolute gap: {absolute_gap:.6f}, "
                            f"Relative gap: Could not be calculated "
                            f"(division by zero)"
                        )

                    # Now save these gaps to a special summary file
                    summary_file = os.path.join(base_results_dir, "gaps_summary.json")

                    # Create a serializable copy without Pyomo objects
                    serializable_results = {
                        "original": {
                            "objective_value": strategy_results["original"]["objective_value"],
                            "status": strategy_results["original"]["status"],
                            "root_relaxation_value": (
                                strategy_results["original"]["root_relaxation_value"]
                            ),
                            "duration": strategy_results["original"]["duration"],
                        },
                        "relaxation": {
                            "objective_value": (strategy_results["relaxation"]["objective_value"]),
                            "status": strategy_results["relaxation"]["status"],
                            "root_relaxation_value": (
                                strategy_results["relaxation"]["root_relaxation_value"]
                            ),
                            "duration": strategy_results["relaxation"]["duration"],
                        },
                        "gaps": {
                            "absolute_gap": strategy_results["gaps"]["absolute_gap"],
                            "relaxation_gap_percent": (
                                strategy_results["gaps"]["relaxation_gap_percent"]
                            ),
                        },
                    }

                    # Write the serialized data to file
                    with open(summary_file, "w") as f:
                        json.dump(serializable_results, f, indent=2)

                    print(f"Saved results summary to {summary_file}")

                    # Update original model result file with gap information
                    orig_results_dir = os.path.join(base_results_dir, "original")
                    orig_results_file = os.path.join(
                        orig_results_dir, "solution_data_original.json"
                    )

                    if os.path.exists(orig_results_file):
                        try:
                            with open(orig_results_file, "r") as f:
                                orig_data = json.load(f)

                            # Add gaps to the original model's data
                            orig_data["performance"]["relaxation_gap_percent"] = relaxation_gap
                            orig_data["performance"]["absolute_gap"] = absolute_gap

                            with open(orig_results_file, "w") as f:
                                json.dump(orig_data, f, indent=2)

                            print("Updated original results file with gap information")
                        except Exception as e:
                            print(f"Error updating original results file: {str(e)}")

                # Save relaxed problem results without relaxation gaps
                # (they belong only to original)
                save_results(
                    relaxed_model,
                    relaxed_result,
                    relaxed_duration,
                    results_dir,
                    strategy,
                    mode,
                    existing_model_name,
                    custom_filename,
                    solver,
                    subsolver,
                    is_relaxation=True,
                    root_relaxation_value=relaxed_root_relaxation_value,
                )

                # Update the Excel file to ensure original model has relaxation gaps
                # and relaxed model doesn't have these gaps (to avoid redundancy)
                excel_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
                excel_path = os.path.join(excel_dir, "results.xlsx")
                if (
                    os.path.exists(excel_path)
                    and strategy_results["gaps"]["relaxation_gap_percent"] is not None
                ):
                    try:
                        # Read Excel
                        df = pd.read_excel(excel_path)

                        # Find original model rows for this strategy/model
                        original_mask = (
                            (df["Model Name"] == existing_model_name)
                            & (df["Strategy"] == strategy)
                            & (df["Mode"] == mode)
                            & (df["Problem Type"] == "Original")
                        )

                        # Find relaxation rows for this strategy/model
                        relaxation_mask = (
                            (df["Model Name"] == existing_model_name)
                            & (df["Strategy"] == strategy)
                            & (df["Mode"] == mode)
                            & (df["Problem Type"] == "Relaxation")
                        )

                        # Update gaps ONLY in original rows
                        if any(original_mask):
                            df.loc[original_mask, "Relative Gap (%)"] = strategy_results["gaps"][
                                "relaxation_gap_percent"
                            ]
                            df.loc[original_mask, "Absolute Gap"] = strategy_results["gaps"][
                                "absolute_gap"
                            ]

                            # Clear gaps from relaxation rows (they don't belong there)
                            if any(relaxation_mask):
                                df.loc[relaxation_mask, "Relative Gap (%)"] = None
                                df.loc[relaxation_mask, "Absolute Gap"] = None

                            df.to_excel(excel_path, index=False)
                            print(
                                "Updated Excel file: gaps added to original model data, "
                                "removed from relaxation"
                            )
                    except Exception as e:
                        print(f"Error updating Excel file: {str(e)}")

                # Save pretty-printed relaxed model
                save_model_pprint(relaxed_model, results_dir, is_relaxation=True)

    return None


# Keep the original function for backward compatibility
def solve_with_gurobi(
    model: Optional[pyo.ConcreteModel],
    reformulation_strategies: List[str],
    mode: str = "approximation",
    time_limit: int = 3600,
    existing_model_name: Optional[str] = None,
    save_only: bool = False,
    custom_filename: Optional[str] = None,
) -> Optional[str]:
    """
    Solve the model using Gurobi through GAMS (legacy function).

    This function is kept for backward compatibility and calls
    the more general solve_model function.
    """
    return solve_model(
        model=model,
        reformulation_strategies=reformulation_strategies,
        mode=mode,
        time_limit=time_limit,
        existing_model_name=existing_model_name,
        save_only=save_only,
        custom_filename=custom_filename,
        solver="gams",
        subsolver="gurobi",
    )


def save_model_pprint(
    model: pyo.ConcreteModel, results_dir: str, is_relaxation: bool = False
) -> None:
    """
    Save a pretty-printed representation of the model to a file.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model to save
    results_dir : str
        Directory to save the output
    is_relaxation : bool, optional
        Whether this is a relaxed model, by default False
    """
    try:
        import io
        from contextlib import redirect_stdout

        # Create a buffer to capture the output
        buffer = io.StringIO()

        # Redirect stdout to capture the pprint output
        with redirect_stdout(buffer):
            model.pprint()

        # Get the captured output
        model_str = buffer.getvalue()

        # Determine filename
        problem_type = "relaxation" if is_relaxation else "original"
        pprint_file = os.path.join(results_dir, f"model_pprint_{problem_type}.txt")

        # Write to file
        with open(pprint_file, "w") as f:
            f.write(model_str)

        print(f"Model pprint saved to {pprint_file}")
    except Exception as e:
        print(f"Error saving model pprint: {str(e)}")


if __name__ == "__main__":
    # Example 1: Create a new model and solve it
    model = k_means.build_model(**parameters)

    # Using the new general solve function with relaxation gap calculation
    solve_model(
        model,
        ["gdp.bigm"],
        mode="no_mode",
        solver="gams",
        subsolver="baron",
        calculate_relaxation_gap=False,  # Calculate the relaxation gap
        relaxation_only=False,  # Solve both original and relaxed problems
    )

    # Example 2: Using direct Gurobi
    """
    solve_model(
        model,
        ["gdp.bigm"],
        mode="no_mode",
        solver="gurobi",
        subsolver=None,
        calculate_relaxation_gap=True,  # Calculate the relaxation gap
    )
    """
