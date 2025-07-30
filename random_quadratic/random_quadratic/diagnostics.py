import os
from io import StringIO
from typing import Dict, List, Optional, TextIO

import numpy as np
import pandas as pd


def analyze_results(obj_tolerance: float = 1e-4, file: Optional[TextIO] = None) -> None:
    """
    Analyze the results Excel file to identify models with different solutions across strategies.

    Parameters
    ----------
    obj_tolerance : float, optional
        Tolerance for determining if objective values are different, by default 1e-4
    file : file object, optional
        File to write output to, by default None (print to console)
    """
    # Path to Excel file
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    excel_path = os.path.join(data_dir, "results.xlsx")

    # Check if file exists
    if not os.path.exists(excel_path):
        print(f"Error: Results file not found at {excel_path}", file=file)
        return

    # Read the Excel file
    print(f"Reading results from {excel_path}...", file=file)
    df = pd.read_excel(excel_path)

    # Count total number of unique models
    total_models = df["Model Name"].nunique()
    print(f"\nTotal number of models in the results: {total_models}", file=file)

    # Group by model name to analyze different strategies for the same model
    model_groups = df.groupby("Model Name")

    # Lists to store models with different solutions
    diff_solution_models = []

    # Analyze each model group
    print("\nAnalyzing models for different solutions across strategies...", file=file)
    for model_name, group in model_groups:
        # Skip if only one strategy was applied to this model
        if len(group) <= 1:
            continue

        # Check if any objective values are different within tolerance
        # First filter out non-optimal solutions or None values
        valid_results = group[group["Status"] == "optimal"].dropna(subset=["Objective Value"])

        if len(valid_results) <= 1:
            continue

        # Check if max difference between any two objective values exceeds tolerance
        obj_values = valid_results["Objective Value"].values
        max_diff = np.max(obj_values) - np.min(obj_values)

        if max_diff > obj_tolerance:
            diff_solution_models.append(
                {
                    "model_name": model_name,
                    "strategies": valid_results["Strategy"].tolist(),
                    "objective_values": valid_results["Objective Value"].tolist(),
                    "max_diff": max_diff,
                }
            )

    # Print results for models with different solutions
    print(
        f"\nFound {len(diff_solution_models)} models with different solutions across strategies "
        f"(tolerance = {obj_tolerance}):",
        file=file,
    )

    if diff_solution_models:
        for i, model_info in enumerate(diff_solution_models, 1):
            print(f"\n{i}. Model: {model_info['model_name']}", file=file)
            print("   Strategies and objective values:", file=file)
            for strategy, obj_value in zip(
                model_info["strategies"], model_info["objective_values"]
            ):
                print(f"   - {strategy}: {obj_value:.10f}", file=file)
            print(f"   Max difference: {model_info['max_diff']:.10f}", file=file)
    else:
        print("   None - All strategies converged to the same solution within tolerance", file=file)


def analyze_performance_by_parameter(parameter: str, file: Optional[TextIO] = None) -> None:
    """
    Analyze how a specific model parameter affects solution time and convergence.

    Parameters
    ----------
    parameter : str
        The parameter to analyze (e.g., 'n_dimensions', 'n_disjunctions')
    file : file object, optional
        File to write output to, by default None (print to console)
    """
    # Path to Excel file
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    excel_path = os.path.join(data_dir, "results.xlsx")

    # Check if file exists
    if not os.path.exists(excel_path):
        print(f"Error: Results file not found at {excel_path}", file=file)
        return

    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Check if parameter exists in the dataframe
    if parameter not in df.columns:
        print(f"Error: Parameter '{parameter}' not found in results", file=file)
        return

    print(f"\nAnalyzing performance by {parameter}:", file=file)

    # Group by parameter value and strategy
    grouped = df.groupby([parameter, "Strategy"])

    # Calculate average solution time and success rate for each group
    performance_stats = grouped.agg(
        {
            "Duration (sec)": "mean",
            "Status": lambda x: (x == "optimal").mean() * 100,  # Percentage of optimal solutions
        }
    ).reset_index()

    performance_stats.columns = [parameter, "Strategy", "Avg Duration (sec)", "Success Rate (%)"]

    # Capture dataframe output to string
    output = StringIO()
    performance_stats.to_string(output, index=False)

    # Print summary
    print(output.getvalue(), file=file)


def analyze_solution_outcomes(
    time_limit: float = 1800, obj_tolerance: float = 1e-4, file: Optional[TextIO] = None
) -> None:
    """
    Analyze solution outcomes by strategy, similar to the bar plot information.

    Parameters
    ----------
    time_limit : float, optional
        Time limit in seconds, by default 1800 (30 minutes)
    obj_tolerance : float, optional
        Tolerance for determining if objective values are different, by default 1e-4
    file : file object, optional
        File to write output to, by default None (print to console)
    """
    # Path to Excel file
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    excel_path = os.path.join(data_dir, "results.xlsx")

    # Check if file exists
    if not os.path.exists(excel_path):
        print(f"Error: Results file not found at {excel_path}", file=file)
        return

    # Read the Excel file
    print(f"Reading results from {excel_path}...", file=file)
    df = pd.read_excel(excel_path)

    # Filter to include only original models (not relaxations)
    df = df[df["Problem Type"] == "Original"]

    print(
        f"\nAnalyzing solution outcomes (time limit: {time_limit}s, tolerance: {obj_tolerance}):",
        file=file,
    )

    # Get unique strategies
    strategies = df["Strategy"].unique()

    # Determine ground truth for each model
    if "Objective Value" in df.columns:
        # Only consider optimal solutions for ground truth
        optimal_solutions = df[df["Status"] == "optimal"]
        if len(optimal_solutions) > 0:
            ground_truth = optimal_solutions.groupby("Model Name")["Objective Value"].min()
        else:
            ground_truth = pd.Series(dtype=float)
    else:
        ground_truth = pd.Series(dtype=float)

    # Initialize results storage
    results_data = []
    wrong_solutions_details = []

    for strategy in strategies:
        strategy_df = df[df["Strategy"] == strategy]

        # Number of timeouts
        n_timeout = (strategy_df["Duration (sec)"] >= time_limit).sum()

        # Entries that didn't timeout
        not_timeout = strategy_df[strategy_df["Duration (sec)"] < time_limit]

        # Determine correct optimal solutions
        n_opt = 0
        if (
            "Status" in not_timeout.columns
            and "Objective Value" in not_timeout.columns
            and len(ground_truth) > 0
        ):
            gt = not_timeout["Model Name"].map(ground_truth)
            correct_opt = not_timeout[
                (not_timeout["Status"] == "optimal")
                & (np.abs(not_timeout["Objective Value"] - gt) <= obj_tolerance)
            ]
            n_opt = len(correct_opt)

        # Wrong solutions: non-timeouts that are not correct optimal
        n_wrong = len(not_timeout) - n_opt

        # Store results
        display_name = strategy.replace("gdp.", "")
        display_name = display_name.replace("bigm", "BigM")
        display_name = display_name.replace("hull_exact", "Hull Exact")
        display_name = display_name.replace("hull", "Hull(ε-approx.)")
        display_name = display_name.replace("hull_reduced_y", "Hull Reduced Y")
        display_name = display_name.replace("binary_multiplication", "Binary Mult.")

        results_data.append(
            {
                "Strategy": display_name,
                "Optimal": n_opt,
                "Timeout": n_timeout,
                "Wrong": n_wrong,
                "Total": n_opt + n_timeout + n_wrong,
            }
        )

        # Collect details about wrong solutions
        if n_wrong > 0 and len(ground_truth) > 0:
            wrong_solutions = not_timeout[
                ~(
                    (not_timeout["Status"] == "optimal")
                    & (np.abs(not_timeout["Objective Value"] - gt) <= obj_tolerance)
                )
            ]

            for _, row in wrong_solutions.iterrows():
                model_name = row["Model Name"]
                reformulation_obj = row["Objective Value"]
                ground_truth_obj = ground_truth.get(model_name, "N/A")
                solution_time = row["Duration (sec)"]
                status = row["Status"]

                wrong_solutions_details.append(
                    {
                        "Strategy": display_name,
                        "Model": model_name,
                        "Objective Value": reformulation_obj,
                        "Ground Truth": ground_truth_obj,
                        "Solution Time (s)": solution_time,
                        "Status": status,
                    }
                )

    # Print summary table
    print("\nSolution Outcomes Summary:", file=file)
    print("=" * 80, file=file)
    print(
        f"{'Strategy':<20} {'Optimal':<10} {'Timeout':<10} {'Wrong':<10} {'Total':<10}", file=file
    )
    print("-" * 80, file=file)

    for data in results_data:
        print(
            f"{data['Strategy']:<20} {data['Optimal']:<10} {data['Timeout']:<10} "
            f"{data['Wrong']:<10} {data['Total']:<10}",
            file=file,
        )

    # Print detailed information about wrong solutions
    if wrong_solutions_details:
        print(
            f"\nDetailed Information about Wrong Solutions ({len(wrong_solutions_details)} total):",
            file=file,
        )
        print("=" * 100, file=file)

        # Group by strategy for better organization
        wrong_by_strategy: Dict[str, List[Dict[str, str]]] = {}
        for detail in wrong_solutions_details:
            strategy = detail["Strategy"]
            if strategy not in wrong_by_strategy:
                wrong_by_strategy[strategy] = []
            wrong_by_strategy[strategy].append(detail)

        for strategy, details in wrong_by_strategy.items():
            print(f"\n{strategy} ({len(details)} wrong solutions):", file=file)
            print("-" * 80, file=file)
            print(
                f"{'Model':<30} {'Objective':<15} {'Ground \
                 Truth':<15} {'Time(s)':<10} {'Status':<10}",
                file=file,
            )
            print("-" * 80, file=file)

            for detail in details:
                obj_val = (
                    f"{detail['Objective Value']:.6f}"
                    if isinstance(detail["Objective Value"], (int, float))
                    else str(detail["Objective Value"])
                )
                gt_val = (
                    f"{detail['Ground Truth']:.6f}"
                    if isinstance(detail["Ground Truth"], (int, float))
                    else str(detail["Ground Truth"])
                )
                time_val = (
                    f"{detail['Solution Time (s)']:.2f}"
                    if isinstance(detail["Solution Time (s)"], (int, float))
                    else str(detail["Solution Time (s)"])
                )

                print(
                    f"{detail['Model']:<30} {obj_val:<15} {gt_val:<15} \
                        {time_val:<10} {detail['Status']:<10}",
                    file=file,
                )
    else:
        print("\nNo wrong solutions found!", file=file)


def main() -> None:
    """Main function to run all diagnostics and save to file."""
    # Define paths
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    output_file = os.path.join(os.getcwd(), "diagnostics.txt")

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Open file for writing
    with open(output_file, "w") as f:
        header = "=" * 80 + "\n" + "RESULTS DIAGNOSTICS\n" + "=" * 80
        print(header)
        print(header, file=f)

        # Analyze different solutions
        analyze_results(obj_tolerance=1e-4)
        analyze_results(obj_tolerance=1e-4, file=f)

        # Analyze solution outcomes
        analyze_solution_outcomes(time_limit=1800, obj_tolerance=1e-4)
        analyze_solution_outcomes(time_limit=1800, obj_tolerance=1e-4, file=f)

        # Analyze performance by different parameters
        for param in ["n_dimensions", "n_disjunctions", "n_disjuncts_per_disjunction"]:
            analyze_performance_by_parameter(param)
            analyze_performance_by_parameter(param, file=f)

        footer = "\nDiagnostics completed."
        print(footer)
        print(footer, file=f)

    print(f"\nDiagnostics saved to: {output_file}")


if __name__ == "__main__":
    main()
