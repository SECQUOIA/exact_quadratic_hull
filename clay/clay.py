import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, SolverFactory, TransformationFactory, value

# Import GDP plugins to make them available
import pyomo.gdp.plugins.hull_exact
import pyomo.gdp.plugins.hull_reduced_y

# Add the path to import the constrained layout model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pyomo', 'examples', 'gdp', 'constrained_layout'))

try:
    from cons_layout_model import build_constrained_layout_model, constrained_layout_model_examples
except ImportError:
    print("Error: Could not import constrained layout model. Make sure cons_layout_model.py is available.")
    sys.exit(1)


def solve_with_solver(
    model: pyo.ConcreteModel,
    solver: str,
    subsolver: Optional[str],
    time_limit: int = 3600,
    tee: bool = True,
) -> Tuple[Any, float]:
    """
    Solve a model with the specified solver and configuration.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The model to solve
    solver : str
        The solver to use ('gams')
    subsolver : Optional[str]
        The subsolver to use (gurobi, baron, scip)
    time_limit : int, optional
        Time limit in seconds, by default 3600
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
            # Gurobi with GAMS
            options_gams = [
                "$onecho > gurobi.opt",
                "NonConvex 2",
                "$offecho",
                "GAMS_MODEL.optfile=1",
            ]
            solver_name = "gurobi"
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
            raise ValueError(f"Unsupported GAMS subsolver: {subsolver}")

        start = time.time()
        result = opt.solve(
            model,
            solver=solver_name,
            tee=tee,
            keepfiles=True,
            symbolic_solver_labels=True,
            add_options=[
                f"option reslim={time_limit};",
                "option threads=1;",
                "option optcr=1e-6;",
                "option optca=0;",
                *options_gams,
            ],
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    end = time.time()
    duration = end - start

    return result, duration


def extract_solution_values(model: pyo.ConcreteModel) -> Dict[str, Any]:
    """
    Extract solution values from a solved model.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The solved model

    Returns
    -------
    Dict[str, Any]
        Dictionary containing solution information
    """
    solution_values = {}
    
    # Extract rectangle positions
    if hasattr(model, 'rect_x') and hasattr(model, 'rect_y'):
        for r in model.rectangles:
            solution_values[f'rect_{r}_x'] = value(model.rect_x[r])
            solution_values[f'rect_{r}_y'] = value(model.rect_y[r])
    
    # Extract distance values
    if hasattr(model, 'dist_x') and hasattr(model, 'dist_y'):
        for r1, r2 in model.rect_pairs:
            solution_values[f'dist_x_{r1}_{r2}'] = value(model.dist_x[r1, r2])
            solution_values[f'dist_y_{r1}_{r2}'] = value(model.dist_y[r1, r2])
    
    return solution_values


def get_model_parameters(problem_name: str, metric: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model parameters for tracking.

    Parameters
    ----------
    problem_name : str
        Name of the problem (e.g., 'CLay0203')
    metric : str
        Distance metric ('l1' or 'l2')
    params : Dict[str, Any]
        Problem parameters

    Returns
    -------
    Dict[str, Any]
        Dictionary of model parameters
    """
    return {
        'problem_name': problem_name,
        'metric': metric,
        'n_rectangles': len(params['rect_lengths']),
        'n_circles': len(params['circ_xvals']),
        'rect_lengths': str(params['rect_lengths']),
        'rect_heights': str(params['rect_heights']),
        'circ_xvals': str(params['circ_xvals']),
        'circ_yvals': str(params['circ_yvals']),
        'circ_rvals': str(params['circ_rvals']),
        'sep_penalty_matrix': str(params['sep_penalty_matrix']),
    }


def save_to_excel(
    model_params: Dict[str, Any],
    solution: Dict[str, Any],
    performance: Dict[str, Any],
    strategy: str,
    solver: str,
    subsolver: Optional[str] = None,
) -> None:
    """
    Save results to an Excel file, creating it if it doesn't exist.

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
    solver : str
        The main solver used
    subsolver : Optional[str], optional
        The subsolver used, by default None
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Path to Excel file
    excel_path = os.path.join(data_dir, "clay_results.xlsx")

    # Prepare data for the new row
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format solution values string
    solution_str = ""
    if solution.get("solution_values"):
        solution_str = ", ".join([f"{k}={v:.6f}" for k, v in solution["solution_values"].items()])

    # Create a dictionary for the new row
    new_row = {
        "Run Time": run_time,
        "Problem Name": model_params["problem_name"],
        "Metric": model_params["metric"],
        "Strategy": strategy,
        "Duration (sec)": performance["solution_time_seconds"],
        "Status": solution["status"],
        "Objective Value": solution["objective_value"],
        "Lower Bound": solution.get("lower_bound"),
        "Bound Absolute Gap": solution.get("bound_absolute_gap"),
        "Bound Relative Gap (%)": solution.get("bound_relative_gap_percent"),
        "Solution Values": solution_str,
        "Solver": solver,
        "Subsolver": subsolver if subsolver else "None",
        # Model parameters
        "n_rectangles": model_params["n_rectangles"],
        "n_circles": model_params["n_circles"],
        "rect_lengths": model_params["rect_lengths"],
        "rect_heights": model_params["rect_heights"],
        "circ_xvals": model_params["circ_xvals"],
        "circ_yvals": model_params["circ_yvals"],
        "circ_rvals": model_params["circ_rvals"],
        "sep_penalty_matrix": model_params["sep_penalty_matrix"],
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


def solve_constrained_layout_problem(
    problem_name: str,
    metric: str,
    reformulation_strategies: List[str],
    solver_configs: List[Dict[str, Any]],
    time_limit: int = 3600,
) -> None:
    """
    Solve a specific constrained layout problem with multiple strategies and solvers.

    Parameters
    ----------
    problem_name : str
        Name of the problem (e.g., 'CLay0203')
    metric : str
        Distance metric ('l1' or 'l2')
    reformulation_strategies : List[str]
        List of reformulation strategies to use
    solver_configs : List[Dict[str, Any]]
        List of solver configurations
    time_limit : int, optional
        Time limit in seconds, by default 3600
    """
    print(f"\n{'='*80}")
    print(f"Solving {problem_name} with {metric} metric")
    print(f"{'='*80}")
    
    # Get problem parameters
    if problem_name not in constrained_layout_model_examples:
        print(f"Error: Problem {problem_name} not found in examples")
        return
    
    params = constrained_layout_model_examples[problem_name]
    model_params = get_model_parameters(problem_name, metric, params)
    
    # For each solver configuration
    for config in solver_configs:
        solver = config.get("solver", "gams")
        subsolver = config.get("subsolver")
        solver_str = f"{solver}" + (f" with {subsolver}" if subsolver else " direct")
        
        print(f"\nUsing solver: {solver_str}")
        
        # For each reformulation strategy
        for strategy in reformulation_strategies:
            print(f"  Applying strategy: {strategy}")
            
            try:
                # Build the model
                model = build_constrained_layout_model(params, metric)
                
                # Apply the reformulation strategy
                print(f"    Applying transformation: {strategy}")
                pyo.TransformationFactory(strategy).apply_to(model)
                
                # Solve the model
                print(f"    Solving model...")
                result, duration = solve_with_solver(
                    model, solver, subsolver, time_limit, tee=False
                )
                
                # Extract solution information
                solution = {}
                try:
                    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
                        objective_value = value(model.min_dist_cost)
                        solution_values = extract_solution_values(model)
                        
                        solution = {
                            "status": "optimal",
                            "objective_value": objective_value,
                            "solution_values": solution_values,
                        }
                        print(f"    Optimal solution found: {objective_value:.6f}")
                    else:
                        solution = {
                            "status": str(result.solver.termination_condition),
                            "objective_value": None,
                            "solution_values": None,
                        }
                        print(f"    Solution status: {result.solver.termination_condition}")
                        
                except Exception as e:
                    print(f"    Error extracting solution: {str(e)}")
                    solution = {
                        "status": "error",
                        "objective_value": None,
                        "solution_values": None,
                    }
                
                # Get bound information if available
                bound = None
                try:
                    if hasattr(result, "solver") and hasattr(result.solver, "dual_bound"):
                        bound = result.solver.dual_bound
                    elif hasattr(result, "problem") and hasattr(result.problem, "lower_bound"):
                        bound = result.problem.lower_bound
                    elif hasattr(result, "solver") and hasattr(result.solver, "best_objective_bound"):
                        bound = result.solver.best_objective_bound
                except Exception as e:
                    print(f"    Warning: Could not extract bound: {str(e)}")
                
                solution["lower_bound"] = bound
                
                # Calculate gaps if possible
                if solution.get("objective_value") is not None and bound is not None:
                    solution["bound_absolute_gap"] = solution["objective_value"] - bound
                    if bound != 0:
                        solution["bound_relative_gap_percent"] = (solution["bound_absolute_gap"] / abs(bound)) * 100
                    else:
                        solution["bound_relative_gap_percent"] = None
                else:
                    solution["bound_absolute_gap"] = None
                    solution["bound_relative_gap_percent"] = None
                
                # Performance metrics
                performance = {
                    "solution_time_seconds": duration,
                    "solver_status": str(result.solver.status),
                    "termination_condition": str(result.solver.termination_condition),
                }
                
                # Save results
                save_to_excel(
                    model_params,
                    solution,
                    performance,
                    strategy,
                    solver,
                    subsolver,
                )
                
                print(f"    Completed in {duration:.2f} seconds")
                
            except Exception as e:
                print(f"    Error solving with {solver_str} and strategy {strategy}: {str(e)}")
                continue


def run_all_problems(
    reformulation_strategies: List[str] = ["gdp.hull", "gdp.bigm", "gdp.hull_exact", "gdp.hull_reduced_y", "gdp.binary_multiplication"],
    solver_configs: Optional[List[Dict[str, Any]]] = None,
    time_limit: int = 3600,
    metrics: List[str] = ["l1", "l2"],
) -> None:
    """
    Run all constrained layout problems with specified configurations.

    Parameters
    ----------
    reformulation_strategies : List[str], optional
        List of reformulation strategies to use
    solver_configs : Optional[List[Dict[str, Any]]], optional
        List of solver configurations, by default None
    time_limit : int, optional
        Time limit in seconds, by default 3600
    metrics : List[str], optional
        List of distance metrics to use, by default ["l1", "l2"]
    """
    # Default solver configurations
    if solver_configs is None:
        solver_configs = [
            {"solver": "gams", "subsolver": "gurobi"},
            {"solver": "gams", "subsolver": "baron"},
            {"solver": "gams", "subsolver": "scip"},
        ]
    
    print(f"Running all constrained layout problems")
    print(f"Reformulation strategies: {reformulation_strategies}")
    print(f"Distance metrics: {metrics}")
    print(f"Solver configurations: {len(solver_configs)}")
    for i, config in enumerate(solver_configs, 1):
        solver = config.get("solver", "gams")
        subsolver = config.get("subsolver")
        solver_str = f"{solver}" + (f" with {subsolver}" if subsolver else " direct")
        print(f"  {i}. {solver_str}")
    
    # Get all problem names
    problem_names = list(constrained_layout_model_examples.keys())
    print(f"Problems to solve: {problem_names}")
    
    # Solve each problem with each metric
    for problem_name in problem_names:
        for metric in metrics:
            solve_constrained_layout_problem(
                problem_name,
                metric,
                reformulation_strategies,
                solver_configs,
                time_limit,
            )
    
    print("\nAll problems completed!")


if __name__ == "__main__":
    # Configuration
    reformulation_strategies = [
        "gdp.hull",
        "gdp.bigm", 
        "gdp.hull_exact",
        "gdp.binary_multiplication",
    ]
    
    solver_configs = [
        {"solver": "gams", "subsolver": "gurobi"},
        {"solver": "gams", "subsolver": "baron"},
        {"solver": "gams", "subsolver": "scip"},
    ]
    
    # Run all problems
    run_all_problems(
        reformulation_strategies=reformulation_strategies,
        solver_configs=solver_configs,
        time_limit=1800,  
        metrics=[
            "l1",
            "l2"],
    )
