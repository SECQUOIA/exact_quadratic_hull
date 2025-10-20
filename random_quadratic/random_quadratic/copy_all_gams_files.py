"""
Script to copy all GAMS files from random_quadratic/data to exact_quadratic_hull/data.

This script:
1. Reads Excel results files from the random_quadratic repo
2. For each entry, finds the corresponding GAMS file using parameter matching
3. Uses name_mapping.json to rename files
4. Copies GAMS files to organized folders (conv/nonconv, by solver/strategy)
5. Naming format: rand_{conv|nonconv}_{solver}_{strategy}_{number}.gms
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def load_name_mapping(mapping_path: Path) -> Dict[str, str]:
    """
    Load the name mapping from JSON file.
    
    Parameters
    ----------
    mapping_path : Path
        Path to name_mapping.json
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping full model names to short names
    """
    with open(mapping_path, 'r') as f:
        return json.load(f)


def extract_timestamp_from_folder(folder_name: str) -> Optional[datetime]:
    """
    Extract timestamp from folder name.
    
    Parameters
    ----------
    folder_name : str
        Folder name in format YYYY-MM-DD_HH-MM-SS
        
    Returns
    -------
    Optional[datetime]
        Parsed datetime or None if parsing fails
    """
    try:
        return datetime.strptime(folder_name, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def verify_folder_match(
    excel_row: pd.Series,
    json_path: Path,
    obj_tolerance: float = 1e-4,
    time_tolerance: float = 5.0
) -> bool:
    """
    Verify that the folder matches the Excel row.
    
    Compares model parameters and solution data from JSON with Excel data.
    
    Parameters
    ----------
    excel_row : pd.Series
        Row from Excel file
    json_path : Path
        Path to solution_data.json file
    obj_tolerance : float
        Tolerance for objective value comparison
    time_tolerance : float
        Tolerance for time comparison (in seconds)
        
    Returns
    -------
    bool
        True if folder matches, False otherwise
    """
    if not json_path.exists():
        return False
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract data from JSON
        params = data.get('model_parameters', {})
        solution = data.get('solution', {})
        performance = data.get('performance', {})
        
        # Verify model parameters match
        params_match = (
            params.get('n_dimensions') == excel_row['n_dimensions'] and
            params.get('n_disjunctions') == excel_row['n_disjunctions'] and
            params.get('n_disjuncts_per_disjunction') == excel_row['n_disjuncts_per_disjunction'] and
            params.get('n_constraints_per_disjunct') == excel_row['n_constraints_per_disjunct'] and
            params.get('n_feasible_regions') == excel_row['n_feasible_regions'] and
            params.get('random_seed') == excel_row['random_seed']
        )
        
        if not params_match:
            return False
        
        # Verify objective value matches (within tolerance) - only for optimal solutions
        # For failed/timed-out solutions, objective values may not match or may be missing
        status = excel_row.get('Status', '')
        if status == 'optimal':
            obj_value = solution.get('objective_value')
            if obj_value is not None and pd.notna(excel_row['Objective Value']):
                obj_diff = abs(obj_value - excel_row['Objective Value'])
                if obj_diff > obj_tolerance:
                    return False
        
        # Verify solution time matches (within tolerance)
        # Be more lenient with time matching for non-optimal solutions
        solution_time = performance.get('solution_time_seconds')
        if solution_time is not None and pd.notna(excel_row['Duration (sec)']):
            time_diff = abs(solution_time - excel_row['Duration (sec)'])
            # Use larger tolerance for non-optimal solutions (they may have variable timing)
            effective_tolerance = time_tolerance if status == 'optimal' else time_tolerance * 3
            if time_diff > effective_tolerance:
                return False
        
        return True
        
    except Exception as e:
        print(f"Error reading JSON {json_path}: {str(e)}")
        return False


def find_gams_file(
    excel_row: pd.Series,
    data_dir: Path,
    obj_tolerance: float = 1e-4,
    time_tolerance_minutes: float = 60.0
) -> Optional[Path]:
    """
    Find the GAMS file for a given Excel row.
    
    Uses hybrid approach:
    1. Searches for folders with timestamps before or equal to Excel run time
    2. Verifies using JSON data (objective value, solution time, parameters)
    
    Parameters
    ----------
    excel_row : pd.Series
        Row from Excel file
    data_dir : Path
        Base data directory
    obj_tolerance : float
        Tolerance for objective value comparison
    time_tolerance_minutes : float
        Time window to search for folders (in minutes)
        
    Returns
    -------
    Optional[Path]
        Path to GAMS file or None if not found
    """
    solver = excel_row['Solver']
    subsolver = excel_row.get('Subsolver', None)
    strategy = excel_row['Strategy']
    mode = excel_row['Mode']
    run_time_str = excel_row['Run Time']
    
    # Parse Excel run time
    try:
        excel_run_time = datetime.strptime(run_time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print(f"Error: Could not parse run time '{run_time_str}'")
        return None
    
    # Construct search directory
    # Format: gams_{subsolver}_{strategy} or {solver}_direct_{strategy}
    if subsolver and subsolver != "None" and pd.notna(subsolver):
        solver_dir = f"{solver}_{subsolver}_{strategy}"
    else:
        solver_dir = f"{solver}_direct_{strategy}"
    
    search_dir = data_dir / solver_dir / mode
    
    if not search_dir.exists():
        print(f"Warning: Search directory does not exist: {search_dir}")
        return None
    
    # Find candidate folders
    candidates = []
    
    for folder in search_dir.iterdir():
        if not folder.is_dir():
            continue
        
        folder_time = extract_timestamp_from_folder(folder.name)
        if folder_time is None:
            continue
        
        # Calculate time difference
        # Positive values: folder is before Excel time (typical case)
        # Negative values: folder is after Excel time (can happen for long-running or failed jobs)
        time_diff_seconds = (excel_run_time - folder_time).total_seconds()
        
        # Allow folders within tolerance window both before AND after Excel time
        # For failed/timeout cases, folders may be created significantly after Excel time
        if -time_tolerance_minutes * 60 <= time_diff_seconds <= time_tolerance_minutes * 60:
            candidates.append((folder, folder_time, time_diff_seconds))
    
    if not candidates:
        print(f"Warning: No candidate folders found within time window for {excel_row['Model Name']}")
        # Fallback: Try to find ANY folder matching parameters, regardless of time
        print(f"  Attempting fallback search by parameters only...")
        all_folders = []
        for folder in search_dir.iterdir():
            if not folder.is_dir():
                continue
            folder_time = extract_timestamp_from_folder(folder.name)
            if folder_time is None:
                continue
            time_diff_seconds = (excel_run_time - folder_time).total_seconds()
            all_folders.append((folder, folder_time, time_diff_seconds))
        
        if all_folders:
            # Sort by absolute time difference
            all_folders.sort(key=lambda x: abs(x[2]))
            candidates = all_folders  # Use all folders as candidates
        else:
            return None
    else:
        # Sort by time difference (prefer closest match before Excel time)
        candidates.sort(key=lambda x: (-x[2] if x[2] >= 0 else float('inf'), abs(x[2])))
    
    # Verify candidates
    for folder, folder_time, time_diff in candidates:
        json_path = folder / "original" / "solution_data_original.json"
        gams_path = folder / "original" / "model.gms"
        
        if gams_path.exists() and verify_folder_match(excel_row, json_path, obj_tolerance):
            if abs(time_diff) > time_tolerance_minutes * 60:
                print(f"  Found match outside time window (time diff: {abs(time_diff)/60:.1f} min)")
            return gams_path
    
    # If no verified match, return the closest folder's GAMS file with a warning
    if candidates:
        gams_path = candidates[0][0] / "original" / "model.gms"
        if gams_path.exists():
            print(f"Warning: No verified match found for {excel_row['Model Name']}, using closest folder")
            return gams_path
    
    return None


def copy_gams_file(
    source_gams: Path,
    dest_base: Path,
    problem_type: str,
    solver: str,
    strategy: str,
    model_short_name: str
) -> Path:
    """
    Copy GAMS file to destination with organized structure.
    
    Parameters
    ----------
    source_gams : Path
        Source GAMS file
    dest_base : Path
        Base destination directory
    problem_type : str
        Either "conv" or "nonconv"
    solver : str
        Solver name (e.g., "gurobi", "scip", "baron")
    strategy : str
        Strategy name (e.g., "bigm", "hull", "hull_exact")
    model_short_name : str
        Short model name (e.g., "rand_conv_1")
        
    Returns
    -------
    Path
        Path to the copied file
    """
    # Extract the number from the short name
    # e.g., "rand_conv_1" -> "1", "rand_nonconv_79" -> "79"
    number = model_short_name.split('_')[-1]
    
    # Create destination directory: dest_base/conv or dest_base/nonconv
    dest_dir = dest_base / problem_type
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Create new filename: rand_{conv|nonconv}_{solver}_{strategy}_{number}.gms
    # Clean strategy name (remove "gdp." prefix if present)
    strategy_clean = strategy.replace('gdp.', '')
    new_filename = f"rand_{problem_type}_{solver}_{strategy_clean}_{number}.gms"
    
    dest_path = dest_dir / new_filename
    
    # Copy the file
    shutil.copy2(source_gams, dest_path)
    
    return dest_path


def main():
    """Main function to copy all GAMS files."""
    
    # Configuration
    source_data_dir = Path("/home/sgusev/repo/random_quadratic/data")
    dest_data_dir = Path("/home/sgusev/repo/exact_quadratic_hull/random_quadratic/data")
    name_mapping_path = Path("/home/sgusev/repo/exact_quadratic_hull/random_quadratic/data/batches/name_mapping.json")
    excel_path = Path("/home/sgusev/repo/random_quadratic/data/results.xlsx")
    
    obj_tolerance = 1e-4
    time_tolerance_minutes = 120.0  # 2 hours to handle batch runs and delayed folder creation
    
    print("="*80)
    print("COPYING ALL GAMS FILES")
    print("="*80)
    print(f"Source: {source_data_dir}")
    print(f"Destination: {dest_data_dir}")
    print(f"Excel file: {excel_path}")
    print(f"Name mapping: {name_mapping_path}")
    
    # Check if Excel file exists
    if not excel_path.exists():
        print(f"\nError: Excel file not found: {excel_path}")
        return
    
    # Load name mapping
    print(f"\nLoading name mapping...")
    name_mapping = load_name_mapping(name_mapping_path)
    print(f"Loaded {len(name_mapping)} model mappings")
    
    # Reverse mapping to easily look up by short name
    # Determine conv/nonconv from the short name
    conv_models = {k: v for k, v in name_mapping.items() if 'conv_' in v and 'nonconv' not in v}
    nonconv_models = {k: v for k, v in name_mapping.items() if 'nonconv' in v}
    
    print(f"  - Convex models: {len(conv_models)}")
    print(f"  - Non-convex models: {len(nonconv_models)}")
    
    # Read Excel file
    print(f"\n{'='*80}")
    print("READING EXCEL FILE")
    print(f"{'='*80}")
    
    df = pd.read_excel(excel_path)
    
    # Filter to original problems only (if column exists)
    if "Problem Type" in df.columns:
        df = df[df["Problem Type"] == "Original"]
    
    print(f"Total entries in Excel: {len(df)}")
    print(f"Unique models: {df['Model Name'].nunique()}")
    if 'Strategy' in df.columns:
        print(f"Strategies: {df['Strategy'].unique().tolist()}")
    if 'Solver' in df.columns:
        print(f"Solvers: {df['Solver'].unique().tolist()}")
    if 'Subsolver' in df.columns:
        print(f"Subsolvers: {df['Subsolver'].unique().tolist()}")
    
    results_summary = []
    errors = []
    not_found = []
    not_in_mapping = []
    total_excel_entries = len(df)
    
    # Process each entry
    print(f"\n{'='*80}")
    print("PROCESSING ENTRIES")
    print(f"{'='*80}")
    
    processed_count = 0
    
    for idx, row in df.iterrows():
        model_name = row['Model Name']
        strategy = row['Strategy']
        solver = row['Solver']
        subsolver = row.get('Subsolver', None)
        
        # Get short name from mapping
        if model_name not in name_mapping:
            not_in_mapping.append({
                'model_name': model_name,
                'strategy': strategy,
                'solver': solver,
                'reason': 'Not in name mapping'
            })
            continue
        
        short_name = name_mapping[model_name]
        
        # Determine problem type from short name
        problem_type = 'conv' if 'nonconv' not in short_name else 'nonconv'
        
        # Determine solver name (use subsolver if available, otherwise solver)
        solver_name = subsolver if subsolver and subsolver != "None" and pd.notna(subsolver) else solver
        
        # Find GAMS file
        gams_file = find_gams_file(row, source_data_dir, obj_tolerance, time_tolerance_minutes)
        
        if gams_file is None:
            not_found.append({
                'model_name': model_name,
                'short_name': short_name,
                'strategy': strategy,
                'solver': solver_name,
                'reason': 'GAMS file not found'
            })
            continue
        
        # Copy GAMS file
        try:
            dest_path = copy_gams_file(
                gams_file,
                dest_data_dir,
                problem_type,
                solver_name,
                strategy,
                short_name
            )
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  Processed {processed_count} files...")
            
            results_summary.append({
                'model_name': model_name,
                'short_name': short_name,
                'solver': solver_name,
                'strategy': strategy,
                'problem_type': problem_type,
                'source': str(gams_file),
                'destination': str(dest_path)
            })
            
        except Exception as e:
            errors.append({
                'model_name': model_name,
                'short_name': short_name,
                'strategy': strategy,
                'solver': solver_name,
                'error': str(e),
                'reason': 'Copy error'
            })
    
    print(f"\nProcessing completed!")
    print(f"  Files copied: {processed_count}")
    print(f"  Not found: {len(not_found)}")
    print(f"  Not in mapping: {len(not_in_mapping)}")
    print(f"  Copy errors: {len(errors)}")
    
    # Calculate statistics
    total_not_found = len(not_found) + len(errors) + len(not_in_mapping)
    
    # Save summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Excel entries processed: {total_excel_entries}")
    print(f"Successfully found and copied: {len(results_summary)}")
    print(f"NOT found (total): {total_not_found}")
    print(f"  - GAMS file not found: {len(not_found)}")
    print(f"  - Not in name mapping: {len(not_in_mapping)}")
    print(f"  - Copy errors: {len(errors)}")
    
    summary_file = dest_data_dir / "gams_copy_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("GAMS Files Copy Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {source_data_dir}\n")
        f.write(f"Destination: {dest_data_dir}\n\n")
        f.write(f"Total Excel entries processed: {total_excel_entries}\n")
        f.write(f"Successfully found and copied: {len(results_summary)} files\n")
        f.write(f"NOT found (total): {total_not_found} files\n")
        f.write(f"  - GAMS file not found: {len(not_found)}\n")
        f.write(f"  - Not in name mapping: {len(not_in_mapping)}\n")
        f.write(f"  - Copy errors: {len(errors)}\n\n")
        
        f.write("Organization:\n")
        f.write("  - conv/: Convex (PSD) problems\n")
        f.write("  - nonconv/: Non-convex problems\n")
        f.write("  - Naming: rand_{conv|nonconv}_{solver}_{strategy}_{number}.gms\n\n")
        
        if results_summary:
            # Group by problem type and solver
            by_type_solver = {}
            for result in results_summary:
                key = (result['problem_type'], result['solver'])
                if key not in by_type_solver:
                    by_type_solver[key] = []
                by_type_solver[key].append(result)
            
            f.write("Successfully Copied Files:\n")
            f.write("-"*80 + "\n")
            
            for (prob_type, solver), files in sorted(by_type_solver.items()):
                f.write(f"\n### {prob_type.upper()} - {solver.upper()} ({len(files)} files) ###\n")
                
                # Group by strategy
                by_strategy = {}
                for result in files:
                    strat = result['strategy']
                    if strat not in by_strategy:
                        by_strategy[strat] = []
                    by_strategy[strat].append(result)
                
                for strategy, strat_files in sorted(by_strategy.items()):
                    f.write(f"\n  Strategy: {strategy} ({len(strat_files)} files)\n")
                    # Just list a few examples
                    for i, result in enumerate(strat_files[:3]):
                        f.write(f"    {result['short_name']} -> {Path(result['destination']).name}\n")
                    if len(strat_files) > 3:
                        f.write(f"    ... and {len(strat_files) - 3} more\n")
        
        # Report NOT found files
        if not_found or not_in_mapping or errors:
            f.write("\n\n" + "="*80 + "\n")
            f.write("FILES NOT FOUND/COPIED\n")
            f.write("="*80 + "\n\n")
            
            if not_found:
                f.write(f"\n### GAMS Files Not Found ({len(not_found)} entries) ###\n")
                f.write("These Excel entries had no matching GAMS file in the data directory.\n\n")
                for item in not_found[:20]:
                    f.write(f"  {item['short_name']} - {item['strategy']} ({item['solver']})\n")
                    f.write(f"    Model: {item['model_name']}\n")
                if len(not_found) > 20:
                    f.write(f"\n  ... and {len(not_found) - 20} more\n")
            
            if not_in_mapping:
                f.write(f"\n### Not in Name Mapping ({len(not_in_mapping)} entries) ###\n")
                f.write("These Excel entries had models not found in name_mapping.json.\n\n")
                for item in not_in_mapping[:20]:
                    f.write(f"  {item['model_name']} - {item['strategy']} ({item['solver']})\n")
                if len(not_in_mapping) > 20:
                    f.write(f"\n  ... and {len(not_in_mapping) - 20} more\n")
            
            if errors:
                f.write(f"\n### Copy Errors ({len(errors)} entries) ###\n")
                f.write("These files were found but could not be copied due to errors.\n\n")
                for item in errors[:20]:
                    f.write(f"  {item['short_name']} - {item['strategy']} ({item['solver']})\n")
                    f.write(f"    Error: {item['error']}\n")
                if len(errors) > 20:
                    f.write(f"\n  ... and {len(errors) - 20} more\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Print detailed statistics
    conv_count = sum(1 for r in results_summary if r['problem_type'] == 'conv')
    nonconv_count = sum(1 for r in results_summary if r['problem_type'] == 'nonconv')
    
    print(f"\n{'='*80}")
    print("DETAILED STATISTICS")
    print(f"{'='*80}")
    print(f"Successfully copied: {len(results_summary)} GAMS files")
    print(f"  - Convex: {conv_count}")
    print(f"  - Non-convex: {nonconv_count}")
    
    if total_not_found > 0:
        print(f"\n{'='*80}")
        print(f"FILES NOT FOUND: {total_not_found}")
        print(f"{'='*80}")
        print(f"  - GAMS file not found: {len(not_found)}")
        print(f"  - Not in name mapping: {len(not_in_mapping)}")
        print(f"  - Copy errors: {len(errors)}")
        print(f"\nSee {summary_file} for detailed list of not found files.")
    
    print(f"\n{'='*80}")
    print("COMPLETED")
    print(f"{'='*80}")
    print(f"Total Excel entries: {total_excel_entries}")
    
    if total_excel_entries > 0:
        print(f"Files found and copied: {len(results_summary)} ({len(results_summary)/total_excel_entries*100:.1f}%)")
        print(f"Files NOT found: {total_not_found} ({total_not_found/total_excel_entries*100:.1f}%)")
    else:
        print(f"Files found and copied: {len(results_summary)}")
        print(f"Files NOT found: {total_not_found}")
        print("\nWarning: No Excel entries were processed. Please check:")
        print("  - Excel files exist in archive directories")
        print("  - Archive directories are configured correctly")
    
    print(f"\nGAMS files copied to: {dest_data_dir}")
    print(f"  - Convex: {dest_data_dir / 'conv'}")
    print(f"  - Non-convex: {dest_data_dir / 'nonconv'}")


if __name__ == "__main__":
    main()

