import os
from datetime import datetime
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_strategy_display_name(strategy: str) -> str:
    """Gets a display-friendly name for a strategy."""
    display_name = strategy.replace("gdp.", "")
    # Longer names must be replaced first to avoid partial replacements
    display_name = display_name.replace("hull_exact", "Hull Exact")
    display_name = display_name.replace("hull_reduced_y", "Hull Reduced Y")
    display_name = display_name.replace("binary_multiplication", "Binary Mult.")
    display_name = display_name.replace("hull", "Hull(ε-approx.)")
    display_name = display_name.replace("bigm", "BigM")
    return display_name


def create_solution_time_comparison(
    df: pd.DataFrame,
    strategy1: str,
    strategy2: str,
    output_dir: str,
    obj_tolerance: float = 1e-4,
    time_limit: float = 3600,
) -> None:
    """
    Create a scatter plot comparing solution times between two strategies.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results
    strategy1 : str
        First strategy to compare
    strategy2 : str
        Second strategy to compare
    output_dir : str
        Directory to save the plot
    obj_tolerance : float, optional
        Tolerance for determining if objective values are different, by default 1e-4
    time_limit : float, optional
        Time limit in seconds, by default 3600 (1 hour)
    """
    print(f"Creating comparison plot: {strategy1} vs {strategy2}")

    # Get display names for strategies
    display_name1 = _get_strategy_display_name(strategy1)
    display_name2 = _get_strategy_display_name(strategy2)

    # Filter data for the two strategies
    df1 = df[df["Strategy"] == strategy1]
    df2 = df[df["Strategy"] == strategy2]

    # Merge on Model Name to get matching pairs
    merged = pd.merge(df1, df2, on="Model Name", suffixes=("_1", "_2"))

    # Extract solution times
    times1 = merged["Duration (sec)_1"]
    times2 = merged["Duration (sec)_2"]

    # Extract objective values (assuming they exist in the DataFrame)
    # If these columns don't exist, you'll need to add them to your results.xlsx
    if "Objective Value_1" in merged.columns and "Objective Value_2" in merged.columns:
        obj1 = merged["Objective Value_1"]
        obj2 = merged["Objective Value_2"]

        # Check if objective values are different within tolerance
        obj_different = np.abs(obj1 - obj2) > obj_tolerance
    else:
        # If objective values don't exist, assume all are the same
        obj_different = np.zeros(len(merged), dtype=bool)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Determine max value for axis limits
    max_time = max(times1.max(), times2.max(), time_limit) * 1.2  # Add 20% margin

    # Plot diagonal line (x=y)
    plt.plot([0, time_limit], [0, time_limit], "k--", alpha=0.5, linewidth=4, label="x=y")

    # Add time limit line
    plt.axhline(
        y=time_limit,
        color="r",
        linestyle="--",
        alpha=0.7,
        linewidth=4,
        label=f"Time limit {time_limit}s",
    )
    plt.axvline(x=time_limit, color="r", linestyle="--", alpha=0.7, linewidth=4)

    # Plot data points with different colors based on objective value difference
    blue_points = ~obj_different
    red_points = obj_different

    if np.any(blue_points):
        plt.scatter(
            times1[blue_points],
            times2[blue_points],
            alpha=0.7,
            s=100,
            color="blue",
            label="Same objective",
        )

    if np.any(red_points):
        plt.scatter(
            times1[red_points],
            times2[red_points],
            alpha=0.7,
            s=100,
            color="red",
            label="Different objective",
        )

    # Add labels and title
    plt.xlabel(f"{display_name1} Solution Time (s)", fontsize=28)
    plt.ylabel(f"{display_name2} Solution Time (s)", fontsize=28)
    # plt.title(f"Solution Time Comparison\n{display_name1} vs {display_name2}", fontsize=32)

    # Add legend (always show it, regardless of whether there are different objectives)
    plt.legend(loc="lower right", fontsize=26, framealpha=0.4)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Ensure axes are equal and have the same limits
    plt.xlim(0, max_time)
    plt.ylim(0, max_time)

    # Set log scale for both axes
    plt.xscale("log")
    plt.yscale("log")

    # Ensure axes are symmetric by setting identical limits
    min_time = max(0.1, min(times1.min(), times2.min()) * 0.9)  # Avoid log(0) issues
    plt.xlim(min_time, max_time)
    plt.ylim(min_time, max_time)

    # Set tick parameters to match performance profile style
    plt.tick_params(axis="both", which="major", labelsize=24)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_dir, f"comparison_{strategy1}_vs_{strategy2}.jpg")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_file}")


def create_relaxation_gap_comparison(
    df: pd.DataFrame,
    strategy1: str,
    strategy2: str,
    output_dir: str,
    gap_type: str = "relative",
    obj_tolerance: float = 1e-4,
) -> None:
    """
    Create a scatter plot comparing relaxation gaps between two strategies.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results
    strategy1 : str
        First strategy to compare
    strategy2 : str
        Second strategy to compare
    output_dir : str
        Directory to save the plot
    gap_type : str, optional
        Type of gap to plot: "relative" for percentage gap or "absolute" for absolute gap,
        by default "relative"
    obj_tolerance : float, optional
        Tolerance for determining if objective values are different, by default 1e-4
    """
    print(f"Creating {gap_type} gap comparison plot: {strategy1} vs {strategy2}")

    # Determine column name based on gap type
    if gap_type == "relative":
        col_name = "Relative Gap (%)"
        axis_label = "Relaxation Gap (%)"
        title_prefix = "Relative"
        filename_prefix = "relative_gap"
    else:  # absolute
        col_name = "Absolute Gap"
        axis_label = "Absolute Gap"
        title_prefix = "Absolute"
        filename_prefix = "absolute_gap"

    print(title_prefix)

    # Check if the required column exists
    if col_name not in df.columns:
        print(f"Gap column '{col_name}' not found in data, skipping {gap_type} gap comparison")
        return

    # Filter data for the two strategies
    df1 = df[df["Strategy"] == strategy1]
    df2 = df[df["Strategy"] == strategy2]

    # Merge on Model Name to get matching pairs
    merged = pd.merge(df1, df2, on="Model Name", suffixes=("_1", "_2"))

    # Filter out rows where either gap is None/NaN
    valid_data = merged[pd.notna(merged[f"{col_name}_1"]) & pd.notna(merged[f"{col_name}_2"])]

    if len(valid_data) == 0:
        print(f"No valid gap data found for {strategy1} vs {strategy2}, skipping")
        return

    gaps1 = valid_data[f"{col_name}_1"]
    gaps2 = valid_data[f"{col_name}_2"]

    # Check if objective values are different
    obj_different = np.zeros(len(valid_data), dtype=bool)
    if "Objective Value_1" in valid_data.columns and "Objective Value_2" in valid_data.columns:
        obj1 = valid_data["Objective Value_1"]
        obj2 = valid_data["Objective Value_2"]
        # Check if objective values are different within tolerance
        obj_different = np.abs(obj1 - obj2) > obj_tolerance

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Find the maximum gap value for axis scaling
    max_gap = max(gaps1.max(), gaps2.max()) * 1.2  # Add 20% margin
    min_gap = min(gaps1.min(), gaps2.min()) * 0.8  # Subtract 20% margin or use 0
    min_gap = max(0, min_gap)  # Ensure non-negative

    # Plot diagonal line (x=y)
    plt.plot([0, max_gap], [0, max_gap], "k--", alpha=0.5, label="x=y")

    # Plot data points with different colors based on objective value difference
    blue_points = ~obj_different
    red_points = obj_different

    if np.any(blue_points):
        plt.scatter(
            gaps1[blue_points],
            gaps2[blue_points],
            alpha=0.7,
            s=50,
            color="blue",
            label="Same objective",
        )

    if np.any(red_points):
        plt.scatter(
            gaps1[red_points],
            gaps2[red_points],
            alpha=0.7,
            s=50,
            color="red",
            label="Different objective",
        )

    # Add labels and title
    plt.xlabel(f"{strategy1} {axis_label}", fontsize=20)
    plt.ylabel(f"{strategy2} {axis_label}", fontsize=20)
    # plt.title(f"{title_prefix} Relaxation Gap: {strategy1} vs {strategy2}", fontsize=14)

    # Add legend (only if there are different objectives)
    if np.any(red_points):
        plt.legend(loc="best")

    # Add grid
    plt.grid(True, alpha=0.3)

    # Set axis limits
    plt.xlim(min_gap, max_gap)
    plt.ylim(min_gap, max_gap)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_dir, f"{filename_prefix}_{strategy1}_vs_{strategy2}.jpg")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_file}")


def create_node_relaxation_comparison(
    df: pd.DataFrame,
    strategy1: str,
    strategy2: str,
    output_dir: str,
    comparison_type: str = "value",
    obj_tolerance: float = 1e-4,
) -> None:
    """
    Create a scatter plot comparing node relaxation values or gaps between two strategies.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results
    strategy1 : str
        First strategy to compare
    strategy2 : str
        Second strategy to compare
    output_dir : str
        Directory to save the plot
    comparison_type : str, optional
        Type of comparison: "value" for root relaxation values or "gap" for root relaxation gaps,
        by default "value"
    obj_tolerance : float, optional
        Tolerance for determining if objective values are different, by default 1e-4
    """
    print(f"Creating node relaxation {comparison_type} comparison plot: {strategy1} vs {strategy2}")

    # Determine column name based on comparison type
    if comparison_type == "value":
        col_name = "Root Relaxation Value"
        axis_label = "Root Relaxation Value"
        title_prefix = "Root Relaxation Value"
        filename_prefix = "node_relaxation_value"
    else:  # gap
        col_name = "Root Relaxation Gap (%)"
        axis_label = "Root Relaxation Gap (%)"
        title_prefix = "Root Relaxation Gap"
        filename_prefix = "node_relaxation_gap"

    print(title_prefix)

    # Check if the required column exists
    if col_name not in df.columns:
        print(
            f"Column '{col_name}' not found in data, \
                skipping node relaxation {comparison_type} comparison"
        )
        return

    # Filter data for the two strategies
    df1 = df[df["Strategy"] == strategy1]
    df2 = df[df["Strategy"] == strategy2]

    # Merge on Model Name to get matching pairs
    merged = pd.merge(df1, df2, on="Model Name", suffixes=("_1", "_2"))

    # Filter out rows where either value is None/NaN
    valid_data = merged[pd.notna(merged[f"{col_name}_1"]) & pd.notna(merged[f"{col_name}_2"])]

    if len(valid_data) == 0:
        print(f"No valid node relaxation data found for {strategy1} vs {strategy2}, skipping")
        return

    values1 = valid_data[f"{col_name}_1"]
    values2 = valid_data[f"{col_name}_2"]

    # Check if objective values are different
    obj_different = np.zeros(len(valid_data), dtype=bool)
    if "Objective Value_1" in valid_data.columns and "Objective Value_2" in valid_data.columns:
        obj1 = valid_data["Objective Value_1"]
        obj2 = valid_data["Objective Value_2"]
        # Check if objective values are different within tolerance
        obj_different = np.abs(obj1 - obj2) > obj_tolerance

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Find min and max values for axis scaling
    if comparison_type == "value":
        # For objective values, we need to handle both positive and negative values
        all_values = np.concatenate([values1, values2])
        min_val = min(all_values)
        max_val = max(all_values)

        # Add margin
        range_val = max_val - min_val
        if range_val == 0:  # Handle case where all values are the same
            range_val = abs(min_val) * 0.1 if min_val != 0 else 1.0

        min_plot = min_val - range_val * 0.1
        max_plot = max_val + range_val * 0.1

        # Plot diagonal line (x=y)
        plt.plot([min_plot, max_plot], [min_plot, max_plot], "k--", alpha=0.5, label="x=y")
    else:
        # For gap percentages, we expect non-negative values
        max_gap = max(values1.max(), values2.max()) * 1.2  # Add 20% margin
        min_gap = min(values1.min(), values2.min()) * 0.8  # Subtract 20% margin or use 0
        min_gap = max(0, min_gap)  # Ensure non-negative

        min_plot = min_gap
        max_plot = max_gap

        # Plot diagonal line (x=y)
        plt.plot([0, max_gap], [0, max_gap], "k--", alpha=0.5, label="x=y")

    # Plot data points with different colors based on objective value difference
    blue_points = ~obj_different
    red_points = obj_different

    if np.any(blue_points):
        plt.scatter(
            values1[blue_points],
            values2[blue_points],
            alpha=0.7,
            s=50,
            color="blue",
            label="Same objective",
        )

    if np.any(red_points):
        plt.scatter(
            values1[red_points],
            values2[red_points],
            alpha=0.7,
            s=50,
            color="red",
            label="Different objective",
        )

    # Add labels and title
    plt.xlabel(f"{strategy1} {axis_label}", fontsize=20)
    plt.ylabel(f"{strategy2} {axis_label}", fontsize=20)
    # plt.title(f"{title_prefix} Comparison: {strategy1} vs {strategy2}", fontsize=14)

    # Add legend (only if there are different objectives)
    if np.any(red_points):
        plt.legend(loc="best", framealpha=0.5)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Set axis limits
    plt.xlim(min_plot, max_plot)
    plt.ylim(min_plot, max_plot)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_dir, f"{filename_prefix}_{strategy1}_vs_{strategy2}.jpg")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_file}")


def create_performance_profile(
    df: pd.DataFrame,
    output_dir: str,
    time_limit: float = 3600,
    obj_tolerance: float = 1e-4,
    exclude_strategies: Optional[list] = None,
    separate_time_limit_legend: bool = False,
) -> None:
    """
    Create performance profiles showing number of instances solved within a time limit.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results
    output_dir : str
        Directory to save the plots
    time_limit : float, optional
        Time limit in seconds, by default 3600 (1 hour)
    obj_tolerance : float, optional
        Tolerance for determining if objective values are different, by default 1e-4
    exclude_strategies : list, optional
        List of strategies to exclude from combined performance plot and bar plots, by default None
    separate_time_limit_legend : bool, optional
        If True, show the time limit in a separate legend in the bottom right.
        If False, include it with the strategies in the main legend (upper left).
    """
    print("Creating performance profiles...")

    # Keep a copy of the original data for bar plot summaries
    df_orig = df.copy()

    # Filter out instances with objective values differing from ground truth beyond tolerance
    if "Objective Value" in df.columns:
        # Determine ground truth as the minimum objective value for each model
        ground_truth = df.groupby("Model Name")["Objective Value"].min()
        # Only apply filtering to data where Duration (sec) < time_limit
        mask_below_limit = df["Duration (sec)"] < time_limit
        df_below = df[mask_below_limit].copy()
        df_above = df[~mask_below_limit].copy()
        if not df_below.empty:
            df_below = df_below.assign(GroundTruth=df_below["Model Name"].map(ground_truth))
            diff = np.abs(df_below["Objective Value"] - df_below["GroundTruth"])
            valid_mask = diff <= obj_tolerance
            num_filtered = (~valid_mask).sum()
            if not valid_mask.any():
                print(
                    "No data within objective tolerance for solutions below time limit,\
                          skipping performance profiles"
                )
                return
            if num_filtered > 0:
                print(
                    f"Filtered out {num_filtered} entries with objective values outside \
                        tolerance {obj_tolerance} (for solutions below time limit)"
                )
            df_below = df_below[valid_mask]
        # Concatenate filtered below-limit and unfiltered above-limit data
        df = pd.concat([df_below, df_above], ignore_index=True)
    else:
        print("Objective Value column not found, unable to filter based on objective tolerance")

    # Get unique strategies
    strategies = df["Strategy"].unique()

    # Create individual performance profiles
    for strategy in strategies:
        print(f"  Creating profile for {strategy}")
        strategy_data = df[df["Strategy"] == strategy]

        # Sort by solution time
        solution_times = sorted(strategy_data["Duration (sec)"])

        # Create x and y arrays for plotting
        x = np.sort(solution_times)
        y = np.arange(1, len(x) + 1)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.step(x, y, where="post", linewidth=2)

        # Add time limit line (3600 seconds = 1 hour)
        plt.axvline(
            x=time_limit, color="r", linestyle="--", alpha=0.7, label=f"Time limit {time_limit}s"
        )

        plt.xlabel("Solution Time (s)", fontsize=32)
        plt.ylabel("Number of Instances Solved", fontsize=32)
        # plt.title(f"Performance Profile: {strategy}", fontsize=34)

        # Add grid
        plt.grid(True, alpha=0.3)

        # Set x-axis to log scale
        plt.xscale("log")

        # Add legend in bottom right corner
        plt.legend(loc="lower right", fontsize=24, framealpha=0.4)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the figure
        output_file = os.path.join(output_dir, f"profile_{strategy}.jpg")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved to {output_file}")

    # Create combined performance profile
    plt.figure(figsize=(12, 8))

    # Define consistent style and color mappings for the known reformulation strategies
    style_map = {
        "gdp.bigm": "-",
        "gdp.hull": "--",
        "gdp.hull_exact": "-.",
        "gdp.hull_reduced_y": ":",
        "gdp.binary_multiplication": (0, (5, 1)),
    }
    color_map = {
        "gdp.bigm": "blue",
        "gdp.hull": "brown",
        "gdp.hull_exact": "green",
        "gdp.hull_reduced_y": "purple",
        "gdp.binary_multiplication": "orange",
    }

    # Filter strategies for combined plot if exclude_strategies is provided
    if exclude_strategies is not None:
        strategies = [s for s in strategies if s not in exclude_strategies]

    # Plot each strategy with its assigned style and color
    strategy_lines = []
    strategy_labels = []
    for strategy in strategies:
        strategy_data = df[df["Strategy"] == strategy]
        solution_times = sorted(strategy_data["Duration (sec)"])
        x = np.sort(solution_times)
        y = np.arange(1, len(x) + 1)

        style = style_map.get(strategy, "-")
        color = color_map.get(strategy, "black")

        display_name = _get_strategy_display_name(strategy)

        (line,) = plt.step(
            x, y, where="post", linewidth=6, linestyle=style, color=color, label=display_name
        )
        strategy_lines.append(line)
        strategy_labels.append(display_name)

    if separate_time_limit_legend:
        # Add time limit line (no label)
        time_limit_line = plt.axvline(
            x=time_limit, color="r", linestyle="--", alpha=0.7, linewidth=4
        )
    else:
        # Add time limit line with label for main legend
        time_limit_line = plt.axvline(
            x=time_limit,
            color="r",
            linestyle="--",
            alpha=0.7,
            linewidth=4,
            label=f"Time limit {time_limit}s",
        )

    plt.xlabel("Solution Time (s)", fontsize=34)
    plt.ylabel("Number of Instances Solved", fontsize=34)
    # plt.title("Performance Profiles", fontsize=38)

    if separate_time_limit_legend:
        # First legend for strategies (upper left)
        legend1 = plt.legend(
            strategy_lines, strategy_labels, loc="upper left", fontsize=19, framealpha=0.4
        )
        plt.gca().add_artist(legend1)
        # Second legend for time limit (lower right)
        plt.legend(
            [time_limit_line],
            [f"Time limit {time_limit}s"],
            loc="lower right",
            fontsize=19,
            framealpha=0.4,
        )
    else:
        # Single legend for everything (upper left)
        plt.legend(loc="upper left", fontsize=19, framealpha=0.4)

    plt.grid(True, alpha=0.3)
    plt.tick_params(axis="both", which="major", labelsize=24)

    # Set x-axis to log scale
    plt.xscale("log")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_dir, "profile_combined.jpg")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined profile to {output_file}")

    def create_gap_performance_profile(
        df: pd.DataFrame,
        output_dir: str,
        time_limit: float,
        strategies: list,
        style_map: dict,
        color_map: dict,
    ) -> None:
        if "Bound Absolute Gap" not in df.columns:
            print("Bound Absolute Gap column not found, skipping gap profile plot.")
            return
        # Only consider data that reached the time limit
        df_gap = df[df["Duration (sec)"] >= time_limit]

        # Get all gap values (not NaN)
        all_gaps = df_gap[df_gap["Bound Absolute Gap"].notna()]["Bound Absolute Gap"]
        finite_gaps = all_gaps[np.isfinite(all_gaps)]
        if len(finite_gaps) == 0:
            print("No valid finite gap data found, skipping gap performance profile plot.")
            return
        finite_max = finite_gaps.max()
        plt.figure(figsize=(12, 8))
        strategy_lines = []
        strategy_labels = []
        n_total = df["Model Name"].nunique()
        for strategy in strategies:
            strategy_data = df_gap[df_gap["Strategy"] == strategy]
            strategy_all = df[df["Strategy"] == strategy]
            n_solved = (
                (strategy_all["Duration (sec)"] < time_limit)
                & (strategy_all["Status"] == "optimal")
            ).sum()
            gap_data = strategy_data["Bound Absolute Gap"].dropna().copy()
            gap_data[~np.isfinite(gap_data)] = finite_max
            gap_data = np.minimum(gap_data, finite_max)
            gap_data = np.sort(gap_data)
            n_gap = n_solved + np.arange(1, len(gap_data) + 1)
            # Prepend (0, 0) to the step plot
            gap_data = np.insert(gap_data, 0, 0.0)
            n_gap = np.insert(n_gap, 0, 0)
            style = style_map.get(strategy, "-")
            color = color_map.get(strategy, "black")
            display_name = _get_strategy_display_name(strategy)
            if len(gap_data) > 1:
                (line,) = plt.step(
                    gap_data,
                    n_gap,
                    where="post",
                    linewidth=6,
                    linestyle=style,
                    color=color,
                    label=display_name,
                )
                strategy_lines.append(line)
                strategy_labels.append(display_name)
        plt.xlabel("Gap", fontsize=32)
        plt.ylabel("Number of Instances", fontsize=32)
        plt.title("Performance Profile (Gap Phase)", fontsize=36)
        plt.xlim(0, finite_max)
        plt.ylim(0, n_total + 1)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis="both", which="major", labelsize=22)
        plt.legend(
            strategy_lines,
            strategy_labels,
            loc="upper left",
            fontsize=19,
            framealpha=0.4,
            title="solver",
        )
        output_file_gap = os.path.join(output_dir, "profile_combined_gap.jpg")
        plt.savefig(output_file_gap, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved gap performance profile to {output_file_gap}")

    # Create gap performance profile
    create_gap_performance_profile(df, output_dir, time_limit, strategies, style_map, color_map)

    # Create summary bar chart of solution outcomes
    print("Creating solution outcome bar plots...")
    # Use original data before filtering
    strategies_orig = df_orig["Strategy"].unique()
    # Filter strategies for bar plot if exclude_strategies is provided
    if exclude_strategies is not None:
        strategies_orig = [s for s in strategies_orig if s not in exclude_strategies]
    # Compute ground truth for original data
    if "Objective Value" in df_orig.columns:
        # Only consider optimal solutions for ground truth
        optimal_solutions = df_orig[df_orig["Status"] == "optimal"]
        ground_truth_orig = optimal_solutions.groupby("Model Name")["Objective Value"].min()
    else:
        ground_truth_orig = pd.Series(dtype=float)

    # Initialize counts and create a list to store results for text file
    counts_opt = []
    counts_timeout = []
    counts_wrong = []
    results_data = []

    for strategy in strategies_orig:
        s_df = df_orig[df_orig["Strategy"] == strategy]
        # Number of timeouts
        n_timeout = (s_df["Duration (sec)"] >= time_limit).sum()
        # Entries that didn't timeout
        s_not_timeout = s_df[s_df["Duration (sec)"] < time_limit]
        # Determine correct optimal solutions
        if "Status" in s_not_timeout.columns and "Objective Value" in s_not_timeout.columns:
            gt = s_not_timeout["Model Name"].map(ground_truth_orig)
            correct_opt = s_not_timeout[
                (s_not_timeout["Status"] == "optimal")
                & (np.abs(s_not_timeout["Objective Value"] - gt) <= obj_tolerance)
            ]
            n_opt = len(correct_opt)

            # Print details about wrong solutions
            wrong_solutions = s_not_timeout[
                ~(
                    (s_not_timeout["Status"] == "optimal")
                    & (np.abs(s_not_timeout["Objective Value"] - gt) <= obj_tolerance)
                )
            ]
            if len(wrong_solutions) > 0:
                print(f"\nWrong solutions for {strategy}:")
                for _, row in wrong_solutions.iterrows():
                    model_name = row["Model Name"]
                    reformulation_obj = row["Objective Value"]
                    ground_truth_obj = ground_truth_orig[model_name]
                    print(f"  Model: {model_name}")
                    print(f"    Reformulation objective: {reformulation_obj}")
                    print(f"    Ground truth objective: {ground_truth_obj}")
                    print(f"    Difference: {abs(reformulation_obj - ground_truth_obj)}")
                    print(f"    Status: {row['Status']}")
                    print()
        else:
            n_opt = 0
        # Wrong solutions: non-timeouts that are not correct optimal
        n_wrong = len(s_not_timeout) - n_opt
        counts_opt.append(n_opt)
        counts_timeout.append(n_timeout)
        counts_wrong.append(n_wrong)

        # Store results for text file
        display_name = _get_strategy_display_name(strategy)

        results_data.append(
            {
                "Strategy": display_name,
                "Optimal": n_opt,
                "Timeout": n_timeout,
                "Wrong": n_wrong,
                "Total": n_opt + n_timeout + n_wrong,
            }
        )

    # Save results to text file
    txt_output = os.path.join(output_dir, "solution_outcomes.txt")
    with open(txt_output, "w") as f:
        f.write("Solution Outcomes Summary\n")
        f.write("=======================\n\n")
        f.write(f"Time limit: {time_limit} seconds\n")
        f.write(f"Objective tolerance: {obj_tolerance}\n\n")

        # Write table header
        f.write(f"{'Strategy':<20} {'Optimal':<10} {'Timeout':<10} {'Wrong':<10} {'Total':<10}\n")
        f.write("-" * 60 + "\n")

        # Write data rows
        for data in results_data:
            f.write(
                f"{data['Strategy']:<20} {data['Optimal']:<10} {data['Timeout']:<10} "
                f"{data['Wrong']:<10} {data['Total']:<10}\n"
            )

    print(f"Saved solution outcomes summary to {txt_output}")

    # Plot bar chart
    x = np.arange(len(strategies_orig))
    width = 0.2
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, counts_opt, width, label="Optimal", color="green")
    plt.bar(x, counts_timeout, width, label="Timeout", color="orange")
    plt.bar(x + width, counts_wrong, width, label="Wrong Solution", color="red")
    plt.xlabel("Strategy", fontsize=30)
    plt.ylabel("Count", fontsize=30)
    plt.title("Solution Outcomes by Strategy", fontsize=32)
    display_names = [_get_strategy_display_name(s) for s in strategies_orig]
    plt.xticks(x, display_names, rotation=0, fontsize=22)
    plt.yticks(fontsize=18)
    plt.legend(loc="upper center", bbox_to_anchor=(0.32, 1), fontsize=22, framealpha=0.5)
    plt.grid(axis="y", alpha=0.3)
    bar_output = os.path.join(output_dir, "solution_outcomes_bar.jpg")
    plt.tight_layout()
    plt.savefig(bar_output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved bar plot to {bar_output}")

    def create_fraction_performance_profile(
        df: pd.DataFrame,
        output_dir: str,
        time_limit: float,
        strategies: list,
        style_map: dict,
        color_map: dict,
    ) -> None:
        if "Bound Absolute Gap" not in df.columns:
            print(
                "Bound Absolute Gap column not found, skipping fractional performance profile plot."
            )
            return
        n_total = df["Model Name"].nunique()
        min_time = max(1e-2, df["Duration (sec)"].min())
        all_gaps = df[df["Bound Absolute Gap"].notna()]["Bound Absolute Gap"]
        finite_gaps = all_gaps[np.isfinite(all_gaps)]
        if len(finite_gaps) == 0:
            print("No valid finite gap data found, skipping fractional performance profile plot.")
            return
        finite_max = finite_gaps.max()
        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0)
        ax_runtime = plt.subplot(gs[0])
        ax_gap = plt.subplot(gs[1], sharey=ax_runtime)
        strategy_lines = []
        strategy_labels = []
        for strategy in strategies:
            strategy_data = df[df["Strategy"] == strategy]
            # --- Runtime phase ---
            solved_mask = (strategy_data["Duration (sec)"] < time_limit) & (
                strategy_data["Status"] == "optimal"
            )
            solved_times = np.sort(strategy_data.loc[solved_mask, "Duration (sec)"])
            n_solved = np.arange(1, len(solved_times) + 1)
            frac_solved = n_solved / n_total
            # --- Gap phase ---
            timeout_mask = strategy_data["Duration (sec)"] >= time_limit
            gap_data = strategy_data.loc[timeout_mask, "Bound Absolute Gap"].dropna().copy()
            gap_data[~np.isfinite(gap_data)] = finite_max
            gap_data = np.minimum(gap_data, finite_max)
            gap_data = np.sort(gap_data)
            n_gap = np.arange(1, len(gap_data) + 1)
            frac_gap = n_gap / n_total
            # Prepend (0, 0) to the step plot
            gap_data = np.insert(gap_data, 0, 0.0)
            frac_gap = np.insert(frac_gap, 0, 0.0)
            style = style_map.get(strategy, "-")
            color = color_map.get(strategy, "black")
            display_name = _get_strategy_display_name(strategy)
            # Plot runtime phase, extend to time_limit only if there are timeouts
            if len(solved_times) > 0:
                has_timeout = (strategy_data["Duration (sec)"] >= time_limit).any()
                if has_timeout:
                    x_runtime = np.append(solved_times, time_limit)
                    y_runtime = np.append(frac_solved, frac_solved[-1])
                else:
                    x_runtime = solved_times
                    y_runtime = frac_solved
                (line,) = ax_runtime.step(
                    x_runtime,
                    y_runtime,
                    where="post",
                    linewidth=4,
                    linestyle=style,
                    color=color,
                    label=display_name,
                )
                strategy_lines.append(line)
                strategy_labels.append(display_name)
                y_offset = frac_solved[-1]
            else:
                y_offset = 0
            # Plot gap phase, offset y to start at last runtime fraction
            if len(gap_data) > 1:
                ax_gap.step(
                    gap_data,
                    y_offset + frac_gap,
                    where="post",
                    linewidth=4,
                    linestyle=style,
                    color=color,
                )
        # Axis settings
        ax_runtime.set_xscale("log")
        ax_runtime.set_xlim(min_time, time_limit)
        ax_gap.set_xlim(0, finite_max)
        ax_runtime.set_ylim(0, 1.01)
        ax_gap.set_ylim(0, 1.01)
        ax_runtime.set_xlabel("Runtime [s]", fontsize=22)
        ax_gap.set_xlabel("Gap", fontsize=22)
        ax_runtime.set_ylabel("Fraction of Instances", fontsize=22)
        plt.setp(ax_gap.get_yticklabels(), visible=False)
        d = 0.015
        kwargs = dict(transform=ax_runtime.transAxes, color="k", clip_on=False)
        ax_runtime.plot([1, 1], [-d, +d], **kwargs)
        ax_runtime.plot([1, 1], [1 - d, 1 + d], **kwargs)
        kwargs.update(transform=ax_gap.transAxes)
        ax_gap.plot([0, 0], [-d, +d], **kwargs)
        ax_gap.plot([0, 0], [1 - d, 1 + d], **kwargs)
        ax_runtime.legend(
            strategy_lines, strategy_labels, loc="upper left", fontsize=15, framealpha=0.4
        )
        ax_runtime.grid(True, alpha=0.3)
        ax_gap.grid(True, alpha=0.3)
        ax_runtime.tick_params(axis="both", which="major", labelsize=16)
        ax_gap.tick_params(axis="both", which="major", labelsize=16)
        fig.suptitle("Fractional Performance Profile", fontsize=26)
        output_file_fraction = os.path.join(output_dir, "profile_fraction_performance.jpg")
        plt.savefig(output_file_fraction, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved fractional performance profile to {output_file_fraction}")

    def create_absolute_performance_profile(
        df: pd.DataFrame,
        output_dir: str,
        time_limit: float,
        strategies: list,
        style_map: dict,
        color_map: dict,
    ) -> None:
        if "Bound Absolute Gap" not in df.columns:
            print(
                "Bound Absolute Gap column not found, skipping absolute performance profile plot."
            )
            return
        n_total = df["Model Name"].nunique()
        min_time = max(1e-2, df["Duration (sec)"].min())
        all_gaps = df[df["Bound Absolute Gap"].notna()]["Bound Absolute Gap"]
        finite_gaps = all_gaps[np.isfinite(all_gaps)]
        if len(finite_gaps) == 0:
            print("No valid finite gap data found, skipping absolute performance profile plot.")
            return
        finite_max = finite_gaps.max()
        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0)
        ax_runtime = plt.subplot(gs[0])
        ax_gap = plt.subplot(gs[1], sharey=ax_runtime)
        strategy_lines = []
        strategy_labels = []
        for strategy in strategies:
            strategy_data = df[df["Strategy"] == strategy]
            # --- Runtime phase ---
            solved_mask = (strategy_data["Duration (sec)"] < time_limit) & (
                strategy_data["Status"] == "optimal"
            )
            solved_times = np.sort(strategy_data.loc[solved_mask, "Duration (sec)"])
            n_solved = np.arange(1, len(solved_times) + 1)
            # --- Gap phase ---
            timeout_mask = strategy_data["Duration (sec)"] >= time_limit
            gap_data = strategy_data.loc[timeout_mask, "Bound Absolute Gap"].dropna().copy()
            gap_data[~np.isfinite(gap_data)] = finite_max
            gap_data = np.minimum(gap_data, finite_max)
            gap_data = np.sort(gap_data)
            n_gap = np.arange(1, len(gap_data) + 1)
            # Prepend (0, 0) to the step plot
            gap_data = np.insert(gap_data, 0, 0.0)
            n_gap = np.insert(n_gap, 0, 0)
            style = style_map.get(strategy, "-")
            color = color_map.get(strategy, "black")
            display_name = _get_strategy_display_name(strategy)
            # Plot runtime phase, extend to time_limit only if there are timeouts
            if len(solved_times) > 0:
                has_timeout = (strategy_data["Duration (sec)"] >= time_limit).any()
                if has_timeout:
                    x_runtime = np.append(solved_times, time_limit)
                    y_runtime = np.append(n_solved, n_solved[-1])
                else:
                    x_runtime = solved_times
                    y_runtime = n_solved
                (line,) = ax_runtime.step(
                    x_runtime,
                    y_runtime,
                    where="post",
                    linewidth=4,
                    linestyle=style,
                    color=color,
                    label=display_name,
                )
                strategy_lines.append(line)
                strategy_labels.append(display_name)
                y_offset = n_solved[-1]
            else:
                y_offset = 0
            # Plot gap phase, offset y to start at last runtime count
            if len(gap_data) > 1:
                ax_gap.step(
                    gap_data,
                    y_offset + n_gap,
                    where="post",
                    linewidth=4,
                    linestyle=style,
                    color=color,
                )

        print(fig)
        # Axis settings
        ax_runtime.set_xscale("log")
        ax_runtime.set_xlim(min_time, time_limit)
        ax_gap.set_xlim(0, finite_max)
        ax_runtime.set_ylim(0, n_total + 1)
        ax_gap.set_ylim(0, n_total + 1)
        ax_runtime.set_xlabel("Runtime [s]", fontsize=38)
        ax_gap.set_xlabel("Gap", fontsize=38)
        ax_runtime.set_ylabel("Number of Instances", fontsize=38)
        plt.setp(ax_gap.get_yticklabels(), visible=False)
        d = 0.015
        kwargs = dict(transform=ax_runtime.transAxes, color="k", clip_on=False)
        ax_runtime.plot([1, 1], [-d, +d], **kwargs)
        ax_runtime.plot([1, 1], [1 - d, 1 + d], **kwargs)
        kwargs.update(transform=ax_gap.transAxes)
        ax_gap.plot([0, 0], [-d, +d], **kwargs)
        ax_gap.plot([0, 0], [1 - d, 1 + d], **kwargs)
        ax_runtime.legend(
            strategy_lines, strategy_labels, loc="upper left", fontsize=26, framealpha=0.4
        )
        ax_runtime.grid(True, alpha=0.3)
        ax_gap.grid(True, alpha=0.3)
        ax_runtime.tick_params(axis="both", which="major", labelsize=28)
        ax_gap.tick_params(axis="both", which="major", labelsize=28)
        # fig.suptitle("Absolute Performance Profile", fontsize=26)
        plt.tight_layout()
        output_file_absolute = os.path.join(output_dir, "profile_absolute_performance.jpg")
        plt.savefig(output_file_absolute, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved absolute performance profile to {output_file_absolute}")

    # Call the new performance profile functions
    create_fraction_performance_profile(
        df, output_dir, time_limit, strategies, style_map, color_map
    )
    create_absolute_performance_profile(
        df, output_dir, time_limit, strategies, style_map, color_map
    )


def main() -> None:
    # Define paths
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    results_file = os.path.join(data_dir, "results.xlsx")
    plots_base_dir = os.path.join(data_dir, "plots")

    # Define time limit in seconds (1 hour)
    time_limit = 1800

    # Create plots base directory if it doesn't exist
    os.makedirs(plots_base_dir, exist_ok=True)

    # Create a numbered subfolder to avoid overwriting existing plots
    counter = 1
    while True:
        plots_dir = os.path.join(plots_base_dir, str(counter))
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            print(f"Created plots directory: {plots_dir}")
            break
        counter += 1

    # Create directory for relaxation gap plots
    relaxation_gaps_dir = os.path.join(plots_dir, "relaxation_gaps")
    os.makedirs(relaxation_gaps_dir, exist_ok=True)
    print(f"Created relaxation gaps directory: {relaxation_gaps_dir}")

    # Create directory for node relaxation plots
    node_relaxation_dir = os.path.join(plots_dir, "node_relaxation")
    os.makedirs(node_relaxation_dir, exist_ok=True)
    print(f"Created node relaxation directory: {node_relaxation_dir}")

    # Create a README file with timestamp and info
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(plots_dir, "README.txt"), "w") as readme:
        readme.write(f"Plots generated on: {current_time}\n")
        readme.write(f"Directory: {plots_dir}\n\n")
        readme.write("Contains:\n")
        readme.write("1. Solution time comparison plots between strategies\n")
        readme.write("2. Performance profiles for each strategy\n")
        readme.write("3. Combined performance profile\n")
        readme.write("4. Relaxation gap comparison plots\n")
        readme.write("5. Node relaxation comparison plots\n\n")
        readme.write("Generated by generate_plots.py\n")

    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return

    # Read the Excel file
    print(f"Reading results from {results_file}")
    df = pd.read_excel(results_file)

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
        solver_dir = os.path.join(plots_dir, solver_combo)
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
                    time_limit=time_limit,
                )
            else:
                print(
                    f"Warning: Strategies {strategy1} and/or {strategy2} not\
                          found in results for {solver_combo}"
                )

        # Generate performance profiles
        print("\nGenerating performance profiles...")
        create_performance_profile(
            solver_df, solver_dir, time_limit=time_limit, exclude_strategies=["gdp.hull_reduced_y"]
        )

        # Create subfolder for this solver's relaxation gap plots
        solver_gaps_dir = os.path.join(relaxation_gaps_dir, solver_combo)
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

                # Compare all strategies against the base strategy (usually hull_exact)
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
        solver_node_dir = os.path.join(node_relaxation_dir, solver_combo)
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

                # Compare all strategies against the base strategy (usually hull_exact)
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

    print("\nAll plot generation complete!")


if __name__ == "__main__":
    main()
