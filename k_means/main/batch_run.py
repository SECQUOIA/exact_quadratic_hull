import os
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import k_means
from solve import solve_model


def generate_batch(
    n_dimensions_range: List[int] = [2],
    n_clusters_range: List[int] = [3],
    n_points_range: List[int] = [10],
    coord_range: Tuple[float, float] = (0.0, 10.0),
    batch_name: Optional[str] = None,
    mode: str = "approximation",
    solver: str = "gams",
    subsolver: Optional[str] = "gurobi",
) -> str:
    """
    Generate k-means models and save their names in a batch file.

    Parameters
    ----------
    n_dimensions_range : List[int]
        List of values for number of dimensions
    n_clusters_range : List[int]
        List of values for number of clusters
    n_points_range : List[int]
        List of values for number of points
    coord_range : Tuple[float, float]
        Range for point coordinates
    batch_name : Optional[str]
        Name for this batch (timestamp will be used if None)
    mode : str
        Mode for model solving, used in filename
    solver : str
        Solver to use, by default "gams"
    subsolver : Optional[str]
        Subsolver to use (if using GAMS), by default "gurobi"

    Returns
    -------
    str
        Path to the batch file containing model names
    """
    # Create timestamp for batch name if not provided
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if batch_name is None:
        solver_str = f"{solver}_{subsolver if subsolver else 'direct'}"
        batch_name = f"batch_{solver_str}_{mode}_{timestamp}"

    # Create directory structure
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    batches_dir = os.path.join(data_dir, "batches")

    os.makedirs(batches_dir, exist_ok=True)

    # Generate all parameter combinations
    param_combinations = list(
        product(
            n_dimensions_range,
            n_clusters_range,
            n_points_range,
        )
    )

    # Sort combinations by difficulty (sum of parameters)
    # Models with larger parameter values (more difficult) will be at the end
    param_combinations.sort(key=lambda x: sum(x))

    print(f"Creating batch with {len(param_combinations)} models...")

    # Path for the batch file
    batch_file_path = os.path.join(batches_dir, f"{batch_name}.txt")

    # Process each model
    with open(batch_file_path, "w") as f:
        for i, (n_dim, n_clusters, n_points) in enumerate(param_combinations):
            # Create parameters dict
            parameters = {
                "n_dimensions": n_dim,
                "n_clusters": n_clusters,
                "n_points": n_points,
                "coord_range": coord_range,
            }

            print(f"Building and saving model {i+1}/{len(param_combinations)}")

            # Build the model
            model = k_means.build_model(**parameters)

            # Create a unique filename for this model
            solver_str = f"{solver}_{subsolver if subsolver else 'direct'}"
            base_filename = (
                f"model_{solver_str}_{mode}_{timestamp}_dim{n_dim}_clusters{n_clusters}_"
                f"points{n_points}"
            )

            # Always add a counter to the filename
            counter = 1
            models_dir = os.path.join(data_dir, "models")

            # Find a unique filename with counter
            while True:
                model_filename = f"{base_filename}_{counter}.pkl"
                if not os.path.exists(os.path.join(models_dir, model_filename)):
                    break
                counter += 1

            # Save model using solve_model and get the filename
            saved_filename = solve_model(
                model=model,
                reformulation_strategies=[],  # Empty to skip solving, just save model
                mode=mode,
                time_limit=0,  # No solving needed
                save_only=True,  # Just save the model
                custom_filename=model_filename,  # Use our custom filename
                solver=solver,
                subsolver=subsolver,
            )

            # Write model filename to batch file
            if saved_filename:
                f.write(f"{saved_filename}\n")
            else:
                print(f"Warning: No filename returned for model {i+1}")

    print(f"Batch file created: {batch_file_path}")
    print(f"Model names saved to batch file (total: {len(param_combinations)})")

    return batch_file_path


def run_batch(
    batch_path: str,
    reformulation_strategies: List[str] = ["gdp.hull", "gdp.bigm"],
    mode: str = "approximation",
    time_limit: int = 3600,
    max_models: Optional[int] = None,
    solver_configs: Optional[List[Dict[str, Any]]] = None,
    calculate_relaxation_gap: bool = False,
    relaxation_only: bool = False,
) -> None:
    """
    Run k-means models from a batch file using specified reformulation strategies with
    multiple solver configurations.

    Parameters
    ----------
    batch_path : str
        Path to the batch file containing model names
    reformulation_strategies : List[str]
        List of reformulation strategies to use
    mode : str
        Mode for solving
    time_limit : int
        Time limit in seconds
    max_models : Optional[int]
        Maximum number of models to run (None for all)
    solver_configs : Optional[List[Dict[str, Any]]]
        List of solver configurations, each being a dictionary with 'solver' and 'subsolver' keys.
        If None, defaults to [{'solver': 'gams', 'subsolver': 'gurobi'}]
    calculate_relaxation_gap : bool
        Whether to calculate relaxation gap for each model
    relaxation_only : bool
        Whether to solve only the LP relaxation (skip solving the original MIP)
    """
    # Check if batch file exists
    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    # Read model names from batch file
    with open(batch_path, "r") as f:
        model_names = [line.strip() for line in f if line.strip()]

    # Apply max_models limit if specified
    if max_models is not None:
        model_names = model_names[:max_models]

    batch_name = os.path.basename(batch_path).replace(".txt", "")
    print(f"Running {len(model_names)} models from batch '{batch_name}'")
    print(f"Reformulation strategies: {reformulation_strategies}")
    print(f"Mode: {mode}")
    print(
        f"Relaxation settings: calculate_gap={calculate_relaxation_gap},\
              relaxation_only={relaxation_only}"
    )

    # Default solver config if none provided
    if solver_configs is None:
        solver_configs = [{"solver": "gams", "subsolver": "gurobi"}]

    print(f"Will run with {len(solver_configs)} solver configurations:")
    for i, config in enumerate(solver_configs, 1):
        solver = config.get("solver", "gams")
        subsolver = config.get("subsolver")
        solver_str = f"{solver}" + (f" with {subsolver}" if subsolver else " direct")
        print(f"  {i}. {solver_str}")

    # For each solver configuration
    for config in solver_configs:
        solver = config.get("solver", "gams")
        subsolver = config.get("subsolver")
        solver_str = f"{solver}" + (f" with {subsolver}" if subsolver else " direct")

        print(f"\n{'='*80}")
        print(f"Running with solver configuration: {solver_str}")
        print(f"{'='*80}")

        # Process each model with this solver configuration
        for i, model_name in enumerate(model_names):
            print(f"Processing model {i+1}/{len(model_names)}: {model_name} with {solver_str}")

            # Process one reformulation strategy at a time
            for strategy in reformulation_strategies:
                print(f"  Applying strategy: {strategy}")
                try:
                    # Run the model with solve_model using only one strategy
                    solve_model(
                        model=None,  # No model since we're loading by name
                        reformulation_strategies=[strategy],  # Only one strategy at a time
                        mode=mode,
                        time_limit=time_limit,
                        existing_model_name=model_name,
                        solver=solver,
                        subsolver=subsolver,
                        calculate_relaxation_gap=calculate_relaxation_gap,
                        relaxation_only=relaxation_only,
                    )
                    print(f"  Successfully completed strategy: {strategy}")

                except Exception as e:
                    print(
                        f"  Error running model {model_name} with {solver_str} and\
                              strategy {strategy}: {str(e)}"
                    )
                    # Continue with next strategy instead of stopping
                    continue

        print(f"Completed batch run of {len(model_names)} models using {solver_str}")

    print("\nAll solver configurations have been run.")


# Legacy function for backward compatibility
def run_batch_single_solver(
    batch_path: str,
    reformulation_strategies: List[str] = ["gdp.hull", "gdp.bigm"],
    mode: str = "approximation",
    time_limit: int = 3600,
    max_models: Optional[int] = None,
    solver: str = "gams",
    subsolver: Optional[str] = "gurobi",
    calculate_relaxation_gap: bool = False,
    relaxation_only: bool = False,
) -> None:
    """
    Legacy function for running models with a single solver configuration.
    """
    run_batch(
        batch_path=batch_path,
        reformulation_strategies=reformulation_strategies,
        mode=mode,
        time_limit=time_limit,
        max_models=max_models,
        solver_configs=[{"solver": solver, "subsolver": subsolver}],
        calculate_relaxation_gap=calculate_relaxation_gap,
        relaxation_only=relaxation_only,
    )


if __name__ == "__main__":
    # Example usage:

    mode = "no_mode"

    # Define multiple solver configurations to compare
    solver_configs: List[Dict[str, Any]] = [
        {"solver": "gams", "subsolver": "gurobi"},
        {"solver": "gams", "subsolver": "baron"},
        # {"solver": "gurobi", "subsolver": None},
        # {"solver": "gurobi", "subsolver": 'persistent'},
        # {"solver": "gams", "subsolver": "ipopth"},
        {"solver": "gams", "subsolver": "scip"},
        # {"solver": "scip", "subsolver": None},
    ]

    # Use an existing batch file or create a new one
    batch_path = os.path.join(
        os.path.dirname(os.getcwd()),
        "data",
        "batches",
        "k_means_96.txt",
    )

    # batch_path = None

    only_generate = False

    # Generate a batch of models
    if batch_path is None or not os.path.exists(batch_path):
        batch_path = generate_batch(
            n_dimensions_range=[i for i in range(2, 6)],
            n_clusters_range=[i for i in range(3, 6)],
            n_points_range=[i for i in range(10, 18)],
            coord_range=(-1, 1),
            mode=mode,
            solver="gams",  # For initial model generation only
            subsolver="gurobi",  # For initial model generation only
        )

    if not only_generate:
        # Run the batch with different solvers
        run_batch(
            batch_path=batch_path,
            reformulation_strategies=[
                "gdp.bigm",
                "gdp.hull",
                "gdp.hull_exact",
                # "gdp.hull_reduced_y",
                "gdp.binary_multiplication",
            ],
            mode=mode,
            time_limit=1800,
            solver_configs=solver_configs,  # Run with all specified solver configurations
            calculate_relaxation_gap=False,  # Calculate relaxation gap for each model
            relaxation_only=False,  # Solve both original and relaxed problems
        )
