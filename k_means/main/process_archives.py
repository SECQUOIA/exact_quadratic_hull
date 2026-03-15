import glob
import os
from typing import Optional

import pandas as pd
from generate_plots import (
    create_dolan_more_performance_profile,
    create_node_relaxation_comparison,
    create_performance_profile,
    create_relaxation_gap_comparison,
    create_root_relaxation_scatter,
    create_solution_time_comparison,
)

# Time limit for solver runs (in seconds)
TIME_LIMIT = 300


def find_excel_file(directory: str) -> Optional[str]:
    """
    Find an Excel file in the given directory.
    Returns None if no Excel file is found or if multiple Excel files are found.
    """
    excel_files = glob.glob(os.path.join(directory, "*.xlsx"))
    if len(excel_files) == 1:
        return excel_files[0]
    return None


def _normalize_hull_strategy(solver_df: pd.DataFrame) -> pd.DataFrame:
    """If 'gdp.hull' is absent but 'gdp.hull_eps_1e-4' is present, rename it."""
    available_strategies = set(solver_df["Strategy"].unique())
    if "gdp.hull" not in available_strategies and "gdp.hull_eps_1e-4" in available_strategies:
        print("Note: 'gdp.hull' not found, using 'gdp.hull_eps_1e-4' as 'gdp.hull'")
        solver_df.loc[solver_df["Strategy"] == "gdp.hull_eps_1e-4", "Strategy"] = "gdp.hull"
    return solver_df


def _expand_with_convex_variants(base_strategies: list, available: set) -> list:
    """Expand a list of base strategies to also include their convex_flag_ counterparts
    if present in the available strategies."""
    expanded = list(base_strategies)
    for s in base_strategies:
        convex = "gdp.convex_flag_" + s.replace("gdp.", "") if s.startswith("gdp.") else "convex_flag_" + s
        if convex in available:
            expanded.append(convex)
    return expanded


def _generate_plots_for_solver(
    solver_df: pd.DataFrame, solver_dir: str, plots_folder: str, combo_label: str
) -> None:
    """Generate the full set of plots for a single (possibly combined) solver DataFrame."""

    available_strategies = set(solver_df["Strategy"].unique())

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
                time_limit=TIME_LIMIT,
            )
        else:
            print(
                f"Warning: Strategies {strategy1} and/or {strategy2} not found "
                f"in results for {combo_label}"
            )

    # Generate performance profiles
    print("\nGenerating performance profiles...")
    create_performance_profile(
        solver_df, solver_dir, time_limit=TIME_LIMIT, exclude_strategies=["gdp.hull_reduced_y"]
    )

    # Generate performance profiles for specified reformulations only
    print("\nGenerating performance profiles for specific reformulations...")
    print(f"Available strategies in solver_df: {list(solver_df['Strategy'].unique())}")
    specified_reformulations = _expand_with_convex_variants(
        ["gdp.hull_exact", "gdp.hull"], available_strategies
    )
    print(f"Requested strategies: {specified_reformulations}")
    create_performance_profile(
        solver_df,
        solver_dir,
        time_limit=TIME_LIMIT,
        include_strategies=specified_reformulations,
        output_suffix="hull_exact_vs_hull",
    )

    # Generate performance profiles: BigM, General Exact Hull, Conic Exact Hull, Hull(eps)
    print("\nGenerating performance profiles for BigM + exact hulls + hull...")
    bigm_hulls_reformulations = _expand_with_convex_variants(
        ["gdp.bigm", "gdp.hull_exact", "gdp.hull_exact_conic_no_cholesky", "gdp.hull"],
        available_strategies,
    )
    create_performance_profile(
        solver_df,
        solver_dir,
        time_limit=TIME_LIMIT,
        include_strategies=bigm_hulls_reformulations,
        output_suffix="bigm_exact_hulls_hull",
    )

    # Generate performance profiles: General Exact Hull, Conic Exact Hull, Hull(eps)
    print("\nGenerating performance profiles for exact hulls + hull...")
    exact_hulls_reformulations = _expand_with_convex_variants(
        ["gdp.hull_exact", "gdp.hull_exact_conic_no_cholesky", "gdp.hull"],
        available_strategies,
    )
    create_performance_profile(
        solver_df,
        solver_dir,
        time_limit=TIME_LIMIT,
        include_strategies=exact_hulls_reformulations,
        output_suffix="exact_hulls_hull",
    )

    # Generate Dolan-Moré performance profiles
    print("\nGenerating Dolan-Moré performance profiles...")
    create_dolan_more_performance_profile(
        solver_df, solver_dir, time_limit=TIME_LIMIT, exclude_strategies=["gdp.hull_reduced_y"]
    )

    # Generate Dolan-Moré performance profiles for specified reformulations only
    print("\nGenerating Dolan-Moré performance profiles for specific reformulations...")
    create_dolan_more_performance_profile(
        solver_df,
        solver_dir,
        time_limit=TIME_LIMIT,
        include_strategies=specified_reformulations,
        output_suffix="hull_exact_vs_hull",
    )

    # Generate Dolan-Moré profiles: BigM, General Exact Hull, Conic Exact Hull, Hull(eps)
    print("\nGenerating Dolan-Moré profiles for BigM + exact hulls + hull...")
    create_dolan_more_performance_profile(
        solver_df,
        solver_dir,
        time_limit=TIME_LIMIT,
        include_strategies=bigm_hulls_reformulations,
        output_suffix="bigm_exact_hulls_hull",
    )

    # Generate Dolan-Moré profiles: General Exact Hull, Conic Exact Hull, Hull(eps)
    print("\nGenerating Dolan-Moré profiles for exact hulls + hull...")
    create_dolan_more_performance_profile(
        solver_df,
        solver_dir,
        time_limit=TIME_LIMIT,
        include_strategies=exact_hulls_reformulations,
        output_suffix="exact_hulls_hull",
    )

    # Create subfolder for this solver's relaxation gap plots
    solver_gaps_dir = os.path.join(plots_folder, "relaxation_gaps", combo_label)
    os.makedirs(solver_gaps_dir, exist_ok=True)

    # Check if gap data exists in the dataframe
    has_relative_gap = "Relative Gap (%)" in solver_df.columns
    has_absolute_gap = "Absolute Gap" in solver_df.columns

    if not has_relative_gap and not has_absolute_gap:
        print(f"No gap data found for {combo_label}, skipping gap comparison plots")
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
    solver_node_dir = os.path.join(plots_folder, "node_relaxation", combo_label)
    os.makedirs(solver_node_dir, exist_ok=True)

    # Check if node relaxation data exists in the dataframe
    has_node_value = "Root Relaxation Value" in solver_df.columns
    has_node_gap = "Root Relaxation Gap (%)" in solver_df.columns

    if not has_node_value and not has_node_gap:
        print(
            f"No node relaxation data found for {combo_label}, skipping node relaxation plots"
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

    # Generate root relaxation scatter plots (BigM vs Hull variants)
    has_root_relaxation = "Root Relaxation Value" in solver_df.columns
    if has_root_relaxation:
        solver_scatter_dir = os.path.join(plots_folder, "root_relaxation_scatter", combo_label)
        os.makedirs(solver_scatter_dir, exist_ok=True)

        print("\nGenerating root relaxation scatter plots...")
        scatter_comparisons = [
            ("gdp.bigm", "gdp.hull_exact"),
            ("gdp.bigm", "gdp.hull_exact_conic_no_cholesky"),
        ]
        strategies = solver_df["Strategy"].unique()
        for strategy_x, strategy_y in scatter_comparisons:
            if strategy_x in strategies and strategy_y in strategies:
                create_root_relaxation_scatter(
                    solver_df,
                    strategy_x,
                    strategy_y,
                    solver_scatter_dir,
                    obj_tolerance=1e-4,
                )
            else:
                print(
                    f"Skipping scatter {strategy_x} vs {strategy_y}: "
                    f"one or both strategies not present in data"
                )
    else:
        print("No Root Relaxation Value data found, skipping root relaxation scatter plots")

    print(f"Plot generation for {combo_label} complete!")


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
        solver_df = df[df["Solver_Combo"] == solver_combo].copy()

        # Normalize hull strategy name
        solver_df = _normalize_hull_strategy(solver_df)

        _generate_plots_for_solver(solver_df, solver_dir, plots_folder, solver_combo)

    # After processing individual solver combos, check for scip + scip_convex pairs
    # and create a combined folder with merged data.
    combo_list = list(solver_combos)
    scip_pairs = []
    for combo in combo_list:
        convex_combo = combo + "_convex"
        if convex_combo in combo_list:
            scip_pairs.append((combo, convex_combo))

    for base_combo, convex_combo in scip_pairs:
        combined_label = base_combo + "_combined"
        combined_dir = os.path.join(plots_folder, combined_label)
        os.makedirs(combined_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Creating combined plots: {base_combo} + {convex_combo} -> {combined_label}")
        print(f"{'=' * 60}")

        # Get data for both solver combos
        base_df = df[df["Solver_Combo"] == base_combo].copy()
        convex_df = df[df["Solver_Combo"] == convex_combo].copy()

        # Normalize hull strategy name in both
        base_df = _normalize_hull_strategy(base_df)
        convex_df = _normalize_hull_strategy(convex_df)

        # Prefix all strategies from the convex variant with "convex_flag_"
        convex_df["Strategy"] = convex_df["Strategy"].apply(
            lambda s: "gdp.convex_flag_" + s.replace("gdp.", "") if s.startswith("gdp.") else "convex_flag_" + s
        )

        # Combine into a single DataFrame
        combined_df = pd.concat([base_df, convex_df], ignore_index=True)

        print(f"Combined data: {len(combined_df)} entries")
        print(f"Combined strategies: {list(combined_df['Strategy'].unique())}")

        _generate_plots_for_solver(combined_df, combined_dir, plots_folder, combined_label)


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
