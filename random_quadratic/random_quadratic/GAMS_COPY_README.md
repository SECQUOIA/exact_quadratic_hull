# GAMS Files Copy Script

## Overview

The `copy_all_gams_files.py` script copies GAMS files from the `random_quadratic` repository to the `exact_quadratic_hull` repository with organized naming and folder structure.

## What It Does

1. **Reads Excel results** from archive directories in `/home/sgusev/repo/random_quadratic/data/archive/`
2. **Matches GAMS files** to Excel entries using:
   - Model parameters (dimensions, disjunctions, constraints, etc.)
   - Objective values
   - Solution times
   - Timestamps
3. **Renames files** using the name mapping from `name_mapping.json`
4. **Organizes files** into `conv/` and `nonconv/` folders
5. **Naming format**: `rand_{conv|nonconv}_{solver}_{strategy}_{number}.gms`
   - Example: `rand_nonconv_gurobi_bigm_79.gms`

## Directory Structure

### Source
```
/home/sgusev/repo/random_quadratic/data/
├── gdp.bigm/no_mode/2025-04-23_22-43-43/model.gms
├── gdp.hull/no_mode/2025-04-23_22-11-19/model.gms
└── ...
```

### Destination
```
/home/sgusev/repo/exact_quadratic_hull/random_quadratic/data/
├── conv/
│   ├── rand_conv_gurobi_bigm_1.gms
│   ├── rand_conv_gurobi_hull_1.gms
│   ├── rand_conv_baron_bigm_1.gms
│   └── ...
└── nonconv/
    ├── rand_nonconv_gurobi_bigm_1.gms
    ├── rand_nonconv_gurobi_hull_1.gms
    ├── rand_nonconv_gurobi_bigm_79.gms
    └── ...
```

## Usage

```bash
cd /home/sgusev/repo/exact_quadratic_hull/random_quadratic/random_quadratic
python3 copy_all_gams_files.py
```

## Requirements

- Python 3.6+
- pandas
- openpyxl (for reading Excel files)

Install dependencies if needed:
```bash
pip install pandas openpyxl
```

## Archives Processed

The script processes these archive directories:

### Non-convex Problems
- `non-conv_gurobi` → GAMS + Gurobi solver
- `non-conv_baron` → GAMS + BARON solver  
- `non-conv_gams_scip_rerun` → GAMS + SCIP solver

### Convex (PSD) Problems
- `psd_gurobi` → GAMS + Gurobi solver
- `psd_baron` → GAMS + BARON solver
- `psd_gams_scip_rerun` → GAMS + SCIP solver

## Strategies

The script handles these GDP reformulation strategies:
- `gdp.bigm` → Big-M reformulation
- `gdp.hull` → Convex hull reformulation
- `gdp.hull_exact` → Exact convex hull reformulation
- `gdp.hull_reduced_y` → Reduced variable convex hull
- `gdp.binary_multiplication` → Binary multiplication reformulation

## Output

The script creates:
1. **Copied GAMS files** in organized directories
2. **Summary file**: `gams_copy_summary.txt` with detailed information about:
   - Total Excel entries processed
   - Files successfully found and copied (with breakdown by problem type, solver, and strategy)
   - Files NOT found from Excel (with detailed reasons):
     - GAMS file not found in data directory
     - Model not in name mapping
     - Problem type mismatch
     - Copy errors
3. **Console output** with:
   - Progress for each archive
   - Final statistics showing found vs not found percentages
   - Detailed breakdown of reasons for not found files

## Verification

The script uses multiple verification steps to ensure correct matching:
1. **Timestamp matching**: Matches run times between Excel and folder names
2. **Parameter verification**: Checks model dimensions, disjunctions, constraints, seed
3. **Objective value matching**: Verifies objective values match (within tolerance)
4. **Solution time matching**: Verifies solution times match (within tolerance)

## Name Mapping

The script uses `/home/sgusev/repo/exact_quadratic_hull/random_quadratic/data/batches/name_mapping.json` which maps:
- Long model names (e.g., `model_no_mode_2025-04-19_17-59-27_dim3_disj3_disjper10_constper10_feas10_1.pkl`)
- To short names (e.g., `rand_nonconv_1` or `rand_conv_1`)

Model naming convention:
- `rand_conv_*`: Convex (PSD) problems (timestamp: 2025-04-20_23-19-16)
- `rand_nonconv_*`: Non-convex problems (timestamp: 2025-04-19_17-59-27)

## Example Output Names

- `rand_conv_gurobi_bigm_1.gms` → Convex model #1, Gurobi solver, Big-M strategy
- `rand_conv_baron_hull_exact_42.gms` → Convex model #42, BARON solver, Exact hull strategy
- `rand_nonconv_gurobi_bigm_79.gms` → Non-convex model #79, Gurobi solver, Big-M strategy
- `rand_nonconv_scip_hull_reduced_y_100.gms` → Non-convex model #100, SCIP solver, Reduced hull strategy

## Example Console Output

```
================================================================================
COPYING ALL GAMS FILES
================================================================================
Source: /home/sgusev/repo/random_quadratic/data
Destination: /home/sgusev/repo/exact_quadratic_hull/random_quadratic/data
Name mapping: /home/sgusev/repo/exact_quadratic_hull/random_quadratic/data/batches/name_mapping.json

Loading name mapping...
Loaded 340 model mappings
  - Convex models: 240
  - Non-convex models: 100

================================================================================
Processing: non-conv_gurobi
  Solver: gurobi
  Problem type: nonconv
================================================================================
Total entries in Excel: 500
Unique models: 100
Strategies: ['gdp.bigm', 'gdp.hull', 'gdp.hull_exact', 'gdp.hull_reduced_y', 'gdp.binary_multiplication']
  Processed 50 files...
  Processed 100 files...
  ...
  Completed: 485 files copied from non-conv_gurobi
  Not found: 10
  Not in mapping: 3
  Type mismatch: 2

================================================================================
SUMMARY
================================================================================
Total Excel entries processed: 3000
Successfully found and copied: 2850
NOT found (total): 150
  - GAMS file not found: 100
  - Not in name mapping: 30
  - Problem type mismatch: 15
  - Copy errors: 5

================================================================================
DETAILED STATISTICS
================================================================================
Successfully copied: 2850 GAMS files
  - Convex: 1900
  - Non-convex: 950

================================================================================
FILES NOT FOUND: 150
================================================================================
  - GAMS file not found: 100
  - Not in name mapping: 30
  - Problem type mismatch: 15
  - Copy errors: 5

See gams_copy_summary.txt for detailed list of not found files.

================================================================================
COMPLETED
================================================================================
Total Excel entries: 3000
Files found and copied: 2850 (95.0%)
Files NOT found: 150 (5.0%)

GAMS files copied to: /home/sgusev/repo/exact_quadratic_hull/random_quadratic/data
  - Convex: /home/sgusev/repo/exact_quadratic_hull/random_quadratic/data/conv
  - Non-convex: /home/sgusev/repo/exact_quadratic_hull/random_quadratic/data/nonconv
```

## Troubleshooting

### "Could not find GAMS file"
- Check that the source directory exists and contains GAMS files
- Verify the Excel file has correct timestamps
- Check the `time_tolerance_minutes` parameter (default: 60 minutes)

### "Model not found in name mapping"
- Check that the model name in Excel matches the name_mapping.json keys
- Verify the name_mapping.json file is up to date

### "Problem type mismatch"
- This indicates a model is incorrectly categorized in the archive
- Check if the model should be in conv or nonconv archive

## Notes

- The script creates directories automatically if they don't exist
- Existing files with the same name will be overwritten
- The script processes all entries in the Excel files, including multiple strategies per model
- Each combination of (model, solver, strategy) gets its own GAMS file

