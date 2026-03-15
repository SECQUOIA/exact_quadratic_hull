import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mpl_ticker
import os
import glob
import numpy as np
from typing import Dict, List, Optional

# Strategies to include in all plots (leave empty list to include all available)
SELECTED_STRATEGIES = [
    "gdp.hull_exact",
    "gdp.hull",
]

TIME_LIMIT = 300.0


def _get_strategy_display_name(strategy: str) -> str:
    """Gets a display-friendly name for a strategy."""
    bare = strategy.replace("gdp.", "")

    display_name = bare
    display_name = display_name.replace("hull_exact_extra_var_inequal", "General Exact Hull Extra Var Ineq.")
    display_name = display_name.replace("hull_exact_conic_no_cholesky", "Conic Exact Hull")
    display_name = display_name.replace("hull_exact", "General Exact Hull")
    display_name = display_name.replace("hull_reduced_y", "Hull Reduced Y")
    display_name = display_name.replace("binary_multiplication", "Binary Mult.")
    display_name = display_name.replace("hull", "Hull ε-approx. (ε=1e-4)")
    display_name = display_name.replace("bigm", "BigM")
    return display_name


def _get_strategy_style_maps() -> tuple:
    """Get consistent style and color mappings for strategies."""
    style_map = {
        "gdp.bigm": "-",
        "gdp.hull": "--",
        "gdp.hull_exact": "-.",
        "gdp.hull_reduced_y": ":",
        "gdp.binary_multiplication": (0, (5, 1)),
        "gdp.hull_exact_extra_var_inequal": (0, (10, 5)),
        "gdp.hull_exact_conic_no_cholesky": (0, (3, 1, 1, 1)),
    }
    color_map = {
        "gdp.bigm": "blue",
        "gdp.hull": "brown",
        "gdp.hull_exact": "green",
        "gdp.hull_reduced_y": "purple",
        "gdp.binary_multiplication": "teal",
        "gdp.hull_exact_extra_var_inequal": "darkgreen",
        "gdp.hull_exact_conic_no_cholesky": "orange",
    }
    return style_map, color_map


def _get_strategy_color(strategy: str, base_color_map: dict):
    """Return a consistent color for a strategy, with fallback to tab20 palette."""
    if strategy in base_color_map:
        return base_color_map[strategy]

    tab20 = [plt.get_cmap("tab20")(i) for i in range(20)]
    used_colors = set(mcolors.to_rgba(c) for c in base_color_map.values())
    available = [c for c in tab20 if mcolors.to_rgba(c) not in used_colors]
    if not available:
        available = tab20

    index = abs(hash(strategy)) % len(available)
    return available[index]


def _filter_strategies(
    strategies: list,
    include_strategies: Optional[list] = None,
    exclude_strategies: Optional[list] = None,
) -> list:
    """Filter strategies based on include/exclude lists."""
    if include_strategies is not None:
        strategies = [s for s in strategies if s in include_strategies]
    if exclude_strategies is not None:
        strategies = [s for s in strategies if s not in exclude_strategies]
    return strategies


def create_dolan_more_profile(
    df: pd.DataFrame,
    output_dir: str,
    time_limit: float = TIME_LIMIT,
    strategies_filter: Optional[List[str]] = None,
    filename_suffix: str = "",
) -> None:
    strategies_all = df['strategy'].unique()
    if strategies_filter:
        strategies = [s for s in strategies_filter if s in strategies_all]
    else:
        strategies = list(strategies_all)
    if not strategies:
        print("No supported strategies found for Dolan-Moré profile, skipping.")
        return

    if 'num_reactors' not in df.columns or 'solve_time_sec' not in df.columns:
        print("Required columns not found for Dolan-Moré profile, skipping.")
        return

    style_map, color_map = _get_strategy_style_maps()

    solved_mask = (df['solve_time_sec'].notna()) & (df['solve_time_sec'] < time_limit)
    if not solved_mask.any():
        print("No solved runs found (< time_limit); cannot build Dolan-Moré profile.")
        return

    min_valid_time = df.loc[solved_mask, 'solve_time_sec'].min()
    if not np.isfinite(min_valid_time) or min_valid_time <= 0:
        print("Invalid fastest solved time for Dolan-Moré profile, skipping.")
        return
    failure_ratio = float(time_limit) / float(min_valid_time)

    per_instance_min = (
        df.loc[solved_mask].groupby('num_reactors')['solve_time_sec'].min()
    )
    if per_instance_min.empty:
        print("No instances with at least one solved run; skipping Dolan-Moré profile.")
        return

    ratios_by_strategy = {s: [] for s in strategies}
    for instance, min_time in per_instance_min.items():
        instance_slice = df[df['num_reactors'] == instance]
        for s in strategies:
            s_rows = instance_slice[instance_slice['strategy'] == s]
            if len(s_rows) == 0:
                ratios_by_strategy[s].append(failure_ratio)
                continue
            t = s_rows['solve_time_sec'].min()
            if not np.isfinite(t) or t >= time_limit or t <= 0:
                ratios_by_strategy[s].append(failure_ratio)
            else:
                ratios_by_strategy[s].append(float(t) / float(min_time))

    plt.figure(figsize=(12, 8))
    for s in strategies:
        ratios = sorted(ratios_by_strategy[s])
        if len(ratios) == 0:
            continue
        n = len(ratios)
        y = np.arange(1, n + 1) / n
        style = style_map.get(s, '-')
        color = _get_strategy_color(s, color_map)
        plt.step(
            ratios,
            y,
            where='post',
            linewidth=6,
            linestyle=style,
            color=color,
            label=_get_strategy_display_name(s),
        )

    plt.xlabel('Performance Ratio', fontsize=28)
    plt.ylabel('Fraction of Problems Solved', fontsize=28)
    plt.xscale('log')
    plt.xlim(1, failure_ratio)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=20, framealpha=0.4)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout()
    name_suffix = ("_" + filename_suffix) if filename_suffix else ""
    output_file = os.path.join(output_dir, f'dolan_more_profile{name_suffix}.jpg')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Dolan-Moré profile to {output_file}")


def create_instances_vs_time_profile(
    df: pd.DataFrame,
    output_dir: str,
    time_limit: float = TIME_LIMIT,
    strategies_filter: Optional[List[str]] = None,
    filename_suffix: str = "",
) -> None:
    strategies_all = df['strategy'].unique()
    if strategies_filter:
        strategies = [s for s in strategies_filter if s in strategies_all]
    else:
        strategies = list(strategies_all)
    if not strategies:
        print("No supported strategies found for instances-vs-time profile, skipping.")
        return

    if 'solve_time_sec' not in df.columns:
        print("Required column 'solve_time_sec' not found, skipping instances-vs-time profile.")
        return

    style_map, color_map = _get_strategy_style_maps()

    plt.figure(figsize=(12, 8))
    lines = []
    labels = []
    for s in strategies:
        s_times = df[df['strategy'] == s]['solve_time_sec'].dropna().values
        if len(s_times) == 0:
            continue
        s_times = np.sort(s_times.astype(float))
        x = s_times
        y = np.arange(1, len(x) + 1)
        style = style_map.get(s, '-')
        color = _get_strategy_color(s, color_map)
        (line,) = plt.step(
            x,
            y,
            where='post',
            linewidth=6,
            linestyle=style,
            color=color,
            label=_get_strategy_display_name(s),
        )
        lines.append(line)
        labels.append(_get_strategy_display_name(s))

    plt.axvline(x=time_limit, color='r', linestyle='--', alpha=0.7, linewidth=6, label=f'Time limit {int(time_limit)}s')

    plt.xlabel('Solution Time (s)', fontsize=34)
    plt.ylabel('Number of Instances Solved', fontsize=34)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(loc='upper left', fontsize=19, framealpha=0.4)
    plt.tight_layout()
    name_suffix = ("_" + filename_suffix) if filename_suffix else ""
    output_file = os.path.join(output_dir, f'profile_instances_vs_time{name_suffix}.jpg')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved instances-vs-time profile to {output_file}")


def create_plot(
    df,
    output_dir: str,
    filter_reactors=False,
    max_reactors=None,
    scatter_only=False,
    strategies_filter: Optional[List[str]] = None,
    filename_suffix: str = "",
):
    plt.figure(figsize=(12, 8))

    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
    style_map, color_map = _get_strategy_style_maps()

    style_idx = 0

    strategies_all = df['strategy'].unique()
    if strategies_filter:
        strategies = [s for s in strategies_filter if s in strategies_all]
    else:
        strategies = list(strategies_all)

    for strategy in strategies:
        strategy_data = df[df['strategy'] == strategy]

        if filter_reactors and max_reactors is not None:
            strategy_data = strategy_data[strategy_data['num_reactors'] <= max_reactors]

        grouped_data = strategy_data.groupby('num_reactors').agg({
            'solve_time_sec': ['mean']
        }).reset_index()

        color = _get_strategy_color(strategy, color_map)
        marker = markers[style_idx % len(markers)]
        style_idx += 1

        label = _get_strategy_display_name(strategy)

        if scatter_only:
            plt.scatter(
                grouped_data['num_reactors'],
                grouped_data[('solve_time_sec', 'mean')],
                marker=marker,
                color=color,
                label=label,
                s=100,
            )
        else:
            plt.plot(
                grouped_data['num_reactors'],
                grouped_data[('solve_time_sec', 'mean')],
                marker=marker,
                color=color,
                linestyle=style_map.get(strategy, '-'),
                linewidth=6,
                label=label,
            )

    plt.axhline(y=TIME_LIMIT, color='r', linestyle='--', alpha=0.7, label=f'Time Limit ({int(TIME_LIMIT)}s)', linewidth=4)

    if filter_reactors and max_reactors is not None:
        output_filename = f'solve_time_vs_reactors_up_to_{max_reactors}'
    else:
        output_filename = 'solve_time_vs_reactors_all_files'

    if scatter_only:
        output_filename += '_scatter'
    else:
        output_filename += '_lines'

    if filename_suffix:
        output_filename += f'_{filename_suffix}'

    plt.xlabel('Number of Reactors', fontsize=38)
    plt.ylabel('Solve Time (seconds)', fontsize=38)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.gca().xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(loc='lower right', fontsize=27, framealpha=0.6)
    plt.tight_layout()

    output_file = os.path.join(output_dir, output_filename + '.jpg')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_file}")


def write_solution_outcomes_summary(
    df: pd.DataFrame,
    output_dir: str,
    time_limit: float = TIME_LIMIT,
    output_filename: str = 'solution_outcomes.txt',
    obj_tolerance: float = 1e-4,
) -> None:
    required_cols = {'strategy', 'num_reactors', 'solve_time_sec'}
    if not required_cols.issubset(set(df.columns)):
        print("Missing required columns for summary; skipping text summary generation.")
        return

    df_local = df.copy()
    df_local['solve_time_sec'] = pd.to_numeric(df_local['solve_time_sec'], errors='coerce')

    all_instances = sorted(df_local['num_reactors'].dropna().unique())
    n_instances = len(all_instances)

    status_col = None
    objective_col = None
    for cand in ['Status', 'status']:
        if cand in df_local.columns:
            status_col = cand
            break
    for cand in ['Objective Value', 'objective', 'Objective', 'objective_value']:
        if cand in df_local.columns:
            objective_col = cand
            break

    strategies = sorted(df_local['strategy'].dropna().unique())
    if len(strategies) == 0:
        print("No strategies found; skipping text summary generation.")
        return

    summary_rows = []
    for strategy in strategies:
        s_df = df_local[df_local['strategy'] == strategy]

        present_instances = set(s_df['num_reactors'].dropna().unique())
        missing = n_instances - len(present_instances)

        n_optimal = 0
        n_wrong_optimal = 0
        n_infeasible = 0
        n_solver_error = 0
        n_timeout = 0

        if status_col is not None and objective_col is not None:
            if 'num_reactors' in df_local.columns:
                all_status = df_local[status_col].astype(str).str.lower()
                optimal_rows_all = df_local[all_status == 'optimal']
                if not optimal_rows_all.empty:
                    ground_truth = optimal_rows_all.groupby('num_reactors')[objective_col].min()
                else:
                    ground_truth = pd.Series(dtype=float)
            else:
                ground_truth = pd.Series(dtype=float)
        else:
            ground_truth = pd.Series(dtype=float)

        for instance in all_instances:
            if instance not in present_instances:
                continue

            rows = s_df[s_df['num_reactors'] == instance]
            if rows.empty:
                continue

            times = pd.to_numeric(rows['solve_time_sec'], errors='coerce')
            finite = times.notna() & np.isfinite(times)
            any_timeout = bool(((finite) & (times >= time_limit)).any())

            category = None

            if status_col is not None and objective_col is not None:
                status_series = rows[status_col].astype(str).str.lower()
                obj_series = pd.to_numeric(rows[objective_col], errors='coerce')
                gt_val = ground_truth.get(instance, np.nan)

                mask_nontime = (finite & (times < time_limit))
                if np.isfinite(gt_val):
                    opt_correct = ((status_series == 'optimal') & mask_nontime & obj_series.notna() & (np.abs(obj_series - gt_val) <= obj_tolerance)).any()
                else:
                    opt_correct = ((status_series == 'optimal') & mask_nontime).any()
                if opt_correct:
                    category = 'optimal'
                else:
                    if (status_series.str.contains('infeasible') | status_series.str.contains('unbounded')).any():
                        category = 'infeasible'
                    else:
                        if np.isfinite(gt_val):
                            wrong_opt = ((status_series == 'optimal') & mask_nontime & obj_series.notna() & (np.abs(obj_series - gt_val) > obj_tolerance)).any()
                        else:
                            wrong_opt = False
                        if wrong_opt:
                            category = 'wrong_optimal'
                        else:
                            if any_timeout:
                                category = 'timeout'
                            else:
                                category = 'solver_error'
            else:
                if (finite & (times < time_limit)).any():
                    category = 'optimal'
                elif any_timeout:
                    category = 'timeout'
                else:
                    category = 'solver_error'

            if category == 'optimal':
                n_optimal += 1
            elif category == 'infeasible':
                n_infeasible += 1
            elif category == 'wrong_optimal':
                n_wrong_optimal += 1
            elif category == 'timeout':
                n_timeout += 1
            else:
                n_solver_error += 1

        summary_rows.append({
            'Strategy': strategy,
            'Optimal': n_optimal,
            'Timeout': n_timeout,
            'Infeasible': n_infeasible,
            'Wrong_Optimal': n_wrong_optimal,
            'Solver_Error': n_solver_error,
            'Missing': int(missing),
            'Total_Instances': int(n_instances),
        })

    output_path = os.path.join(output_dir, output_filename)
    try:
        with open(output_path, 'w') as f:
            f.write("Solution Outcomes Summary\n")
            f.write("=========================\n\n")
            f.write(f"Time limit: {time_limit} seconds\n")
            f.write("Columns available: strategy, num_reactors, solve_time_sec\n\n")
            f.write("Categories:\n")
            f.write("- Solved: Finite runtime strictly below time limit\n")
            f.write("- Timeout: Finite runtime at or above time limit\n")
            f.write("- Error: Present row with NaN/inf runtime\n")
            f.write("- Missing: Instances absent for the strategy (by num_reactors)\n\n")

            f.write(
                f"{'Strategy':<20} {'Optimal':<10} {'Timeout':<10} {'Infeasible':<12} "
                f"{'Wrong_Opt':<11} {'Solver_Err':<12} {'Missing':<10} {'Instances':<10}\n"
            )
            f.write("-" * 105 + "\n")

            for row in summary_rows:
                f.write(
                    f"{row['Strategy']:<20} {row['Optimal']:<10} {row['Timeout']:<10} "
                    f"{row['Infeasible']:<12} {row['Wrong_Optimal']:<11} {row['Solver_Error']:<12} "
                    f"{row['Missing']:<10} {row['Total_Instances']:<10}\n"
                )

        print(f"Saved solution outcomes summary to {output_path}")
    except Exception as e:
        print(f"Failed to write solution outcomes summary: {e}")


def _generate_plots_for_solver(
    solver_df: pd.DataFrame,
    solver_dir: str,
    combo_label: str,
) -> None:
    """Generate the full set of plots for a single solver DataFrame."""
    available = set(solver_df['strategy'].unique())
    print(f"\nGenerating plots for {combo_label}")
    print(f"  Available strategies: {list(available)}")

    write_solution_outcomes_summary(solver_df, output_dir=solver_dir, time_limit=TIME_LIMIT)

    # Solve-time vs reactors: all strategies
    create_plot(solver_df, output_dir=solver_dir, filter_reactors=False, scatter_only=False)
    create_plot(solver_df, output_dir=solver_dir, filter_reactors=False, scatter_only=True)

    # Solve-time vs reactors: selected strategies
    selected_present = [s for s in SELECTED_STRATEGIES if s in available]
    if selected_present:
        create_plot(
            solver_df,
            output_dir=solver_dir,
            filter_reactors=False,
            scatter_only=False,
            strategies_filter=selected_present,
            filename_suffix='selected',
        )
        create_plot(
            solver_df,
            output_dir=solver_dir,
            filter_reactors=False,
            scatter_only=True,
            strategies_filter=selected_present,
            filename_suffix='selected',
        )

    # Performance profiles: all strategies
    create_dolan_more_profile(solver_df, output_dir=solver_dir, time_limit=TIME_LIMIT)
    create_instances_vs_time_profile(solver_df, output_dir=solver_dir, time_limit=TIME_LIMIT)

    # Performance profiles: selected strategies
    if selected_present:
        create_dolan_more_profile(
            solver_df,
            output_dir=solver_dir,
            time_limit=TIME_LIMIT,
            strategies_filter=selected_present,
            filename_suffix='selected',
        )
        create_instances_vs_time_profile(
            solver_df,
            output_dir=solver_dir,
            time_limit=TIME_LIMIT,
            strategies_filter=selected_present,
            filename_suffix='selected',
        )

    print(f"Plot generation for {combo_label} complete!")


def create_plots():
    excel_files = glob.glob('*.xlsx')

    all_data = []
    for file in excel_files:
        df = pd.read_excel(file)
        all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
    else:
        print("No Excel files found!")
        return

    print(f"\nData summary:")
    print(f"  Total entries: {len(combined_df)}")
    print(f"  Columns: {list(combined_df.columns)}")
    if 'strategy' in combined_df.columns:
        print(f"  Strategies: {list(combined_df['strategy'].unique())}")

    # Detect solver column: try 'solver', 'main_solver', or 'Solver'
    solver_col = None
    for cand in ['solver', 'main_solver', 'Solver']:
        if cand in combined_df.columns:
            solver_col = cand
            break

    if solver_col is not None:
        combined_df['_solver_combo'] = combined_df[solver_col].fillna('unknown')
        solver_combos = combined_df['_solver_combo'].unique()
        print(f"  Solver column: '{solver_col}'")
        print(f"  Solver combinations: {list(solver_combos)}")
    else:
        combined_df['_solver_combo'] = 'all_solvers'
        solver_combos = ['all_solvers']
        print("  No solver column detected; generating plots for all data together.")

    for solver_combo in solver_combos:
        solver_dir = solver_combo.replace(' ', '_')
        os.makedirs(solver_dir, exist_ok=True)

        solver_df = combined_df[combined_df['_solver_combo'] == solver_combo].copy()

        print(f"\n{'=' * 60}")
        print(f"Solver: {solver_combo} ({len(solver_df)} entries)")
        print(f"Output directory: {solver_dir}")
        print(f"{'=' * 60}")

        _generate_plots_for_solver(solver_df, solver_dir, solver_combo)

    print("\nAll plot generation complete!")


if __name__ == '__main__':
    create_plots()
