import glob
import os
from typing import Optional

import pandas as pd
from generate_plots import (
    create_node_relaxation_comparison,
    create_performance_profile,
    create_relaxation_gap_comparison,
    create_solution_time_comparison,
)


def find_excel_file(directory: str) -> Optional[str]:
    """
    Find an Excel file in the given directory.
    Returns None if no Excel file is found or if multiple Excel files are found.
    """
    excel_files = glob.glob(os.path.join(directory, "*.xlsx"))
    if len(excel_files) == 1:
        return excel_files[0]
    return None


def process_archive_folder(archive_folder: str) -> None:
    """
    Process a single archive folder, looking for Excel files and generating plots if needed.
    """
    print(f"\nProcessing archive folder: {archive_folder}")

    # Find Excel file
    excel_file = find_excel_file(archive_folder)
    if not excel_file:
        print(f"No single Excel file found in {archive_folder}, skipping...")
        return

    print(f"Found Excel file: {excel_file}")

    # Check if plots folder exists
    plots_folder = os.path.join(archive_folder, "plots")
    if os.path.exists(plots_folder):
        print(f"Plots folder already exists in {archive_folder}, skipping...")
        return

    print(f"Creating plots folder: {plots_folder}")
    os.makedirs(plots_folder, exist_ok=True)

    # Read the Excel file
    print(f"Reading results from {excel_file}")
    df = pd.read_excel(excel_file)

    # Filter to include only original models (not relaxations)
    df = df[df["Problem Type"] == "Original"]

    print("\nData summary:")
    print(f"Total entries: {len(df)}")
    print(f"Unique models: {df['Model Name'].nunique()}")
    print(f"Strategies: {df['Strategy'].unique()}")
    print(f"Modes: {df['Mode'].unique()}")

    # Get unique solver-subsolver combinations
    df["Solver_Combo"] = df["Solver"] + "_" + df["Subsolver"].fillna("direct")
    solver_combos = df["Solver_Combo"].unique()

    print(f"Solver combinations: {solver_combos}")

    # For each solver-subsolver combination, create a separate set of plots
    for solver_combo in solver_combos:
        solver_dir = os.path.join(plots_folder, solver_combo)
        os.makedirs(solver_dir, exist_ok=True)

        print(f"\nGenerating plots for solver combination: {solver_combo}")

        # Filter data for this solver-subsolver combination
        solver_df = df[df["Solver_Combo"] == solver_combo]

        # Define the strategy comparisons to plot
        comparisons = [
            ("gdp.bigm", "gdp.hull"),
            ("gdp.hull_exact", "gdp.hull"),
            ("gdp.bigm", "gdp.hull_exact"),
            ("gdp.hull_exact", "gdp.hull_reduced_y"),
            ("gdp.hull_exact", "gdp.binary_multiplication"),
            ("gdp.bigm", "gdp.binary_multiplication"),
        ]

        # Generate comparison plots
        print("\nGenerating comparison plots...")
        for strategy1, strategy2 in comparisons:
            if (
                strategy1 in solver_df["Strategy"].values
                and strategy2 in solver_df["Strategy"].values
            ):
                create_solution_time_comparison(
                    solver_df,
                    strategy1,
                    strategy2,
                    solver_dir,
                    obj_tolerance=1e-4,
                    time_limit=1800,
                )
            else:
                print(
                    f"Warning: Strategies {strategy1} and/or {strategy2} not found \
                        in results for {solver_combo}"
                )

        # Generate performance profiles
        print("\nGenerating performance profiles...")
        create_performance_profile(
            solver_df, solver_dir, time_limit=1800, exclude_strategies=["gdp.hull_reduced_y"]
        )

        # Create subfolder for this solver's relaxation gap plots
        solver_gaps_dir = os.path.join(plots_folder, "relaxation_gaps", solver_combo)
        os.makedirs(solver_gaps_dir, exist_ok=True)

        # Check if gap data exists in the dataframe
        has_relative_gap = "Relative Gap (%)" in solver_df.columns
        has_absolute_gap = "Absolute Gap" in solver_df.columns

        if not has_relative_gap and not has_absolute_gap:
            print(f"No gap data found for {solver_combo}, skipping gap comparison plots")
        else:
            # Generate relaxation gap comparison plots
            print("\nGenerating relaxation gap comparison plots...")

            # Find all strategies with hull_exact for comparison
            strategies = solver_df["Strategy"].unique()
            base_strategies = ["gdp.hull_exact", "gdp.binary_multiplication", "gdp.bigm"]

            for base_strategy in base_strategies:
                # If hull_exact is not available, use the first strategy as base
                if base_strategy not in strategies and len(strategies) > 0:
                    base_strategy = strategies[0]
                    print(f"Note: gdp.hull_exact not found, using {base_strategy} as base strategy")

                # Compare all strategies against the base strategy
                for strategy in strategies:
                    if strategy != base_strategy:
                        # Generate relative gap comparison if data exists
                        if has_relative_gap:
                            create_relaxation_gap_comparison(
                                solver_df,
                                base_strategy,
                                strategy,
                                solver_gaps_dir,
                                gap_type="relative",
                                obj_tolerance=1e-4,
                            )

                        # Generate absolute gap comparison if data exists
                        if has_absolute_gap:
                            create_relaxation_gap_comparison(
                                solver_df,
                                base_strategy,
                                strategy,
                                solver_gaps_dir,
                                gap_type="absolute",
                                obj_tolerance=1e-4,
                            )

        # Create subfolder for this solver's node relaxation plots
        solver_node_dir = os.path.join(plots_folder, "node_relaxation", solver_combo)
        os.makedirs(solver_node_dir, exist_ok=True)

        # Check if node relaxation data exists in the dataframe
        has_node_value = "Root Relaxation Value" in solver_df.columns
        has_node_gap = "Root Relaxation Gap (%)" in solver_df.columns

        if not has_node_value and not has_node_gap:
            print(
                f"No node relaxation data found for {solver_combo}, skipping node relaxation plots"
            )
        else:
            # Generate node relaxation comparison plots
            print("\nGenerating node relaxation comparison plots...")

            # Find all strategies with hull_exact for comparison
            strategies = solver_df["Strategy"].unique()
            base_strategies = ["gdp.hull_exact", "gdp.binary_multiplication", "gdp.bigm"]

            for base_strategy in base_strategies:
                # If hull_exact is not available, use the first strategy as base
                if base_strategy not in strategies and len(strategies) > 0:
                    base_strategy = strategies[0]
                    print(f"Note: gdp.hull_exact not found, using {base_strategy} as base strategy")

                # Compare all strategies against the base strategy
                for strategy in strategies:
                    if strategy != base_strategy:
                        # Generate node relaxation value comparison if data exists
                        if has_node_value:
                            create_node_relaxation_comparison(
                                solver_df,
                                base_strategy,
                                strategy,
                                solver_node_dir,
                                comparison_type="value",
                                obj_tolerance=1e-4,
                            )

                        # Generate node relaxation gap comparison if data exists
                        if has_node_gap:
                            create_node_relaxation_comparison(
                                solver_df,
                                base_strategy,
                                strategy,
                                solver_node_dir,
                                comparison_type="gap",
                                obj_tolerance=1e-4,
                            )

        print(f"Plot generation for {solver_combo} complete!")


def main() -> None:
    # Define paths
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    archive_dir = os.path.join(data_dir, "archive")

    if not os.path.exists(archive_dir):
        print(f"Error: Archive directory not found at {archive_dir}")
        return

    # Get all subdirectories in the archive folder
    archive_folders = [d for d in glob.glob(os.path.join(archive_dir, "*")) if os.path.isdir(d)]

    if not archive_folders:
        print(f"No archive folders found in {archive_dir}")
        return

    print(f"Found {len(archive_folders)} archive folders")

    # Process each archive folder
    for archive_folder in archive_folders:
        process_archive_folder(archive_folder)

    print("\nAll archive processing complete!")


if __name__ == "__main__":
    main()
