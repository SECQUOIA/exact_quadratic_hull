# Script Comparison: copy_all_gams_files.py vs find_and_copy_failed_logs.py

## Overview

The `copy_all_gams_files.py` script is based on `find_and_copy_failed_logs.py` and follows the same approach for identifying and matching folders from Excel files.

## Key Similarities (Matching Approach)

Both scripts use the **same hybrid matching approach**:

1. **Timestamp-based search**: Find candidate folders within a time window
2. **Parameter verification**: Match model parameters (dimensions, disjunctions, constraints, seed)
3. **Solution verification**: Verify objective values and solution times
4. **JSON validation**: Read and validate `solution_data*.json` files

### Shared Functions

| Function | Purpose |
|----------|---------|
| `extract_timestamp_from_folder()` | Parse folder timestamps |
| `verify_folder_match()` | Verify folder matches Excel row using JSON data |
| `find_*_file/folder()` | Find matching folder using hybrid approach |

## Key Differences

### 1. Source Directory Structure

**find_and_copy_failed_logs.py** (exact_quadratic_hull repo):
```
data/
  └── {solver}_{subsolver}_{strategy}/
      └── {mode}/
          └── {timestamp}/
              └── original/
                  └── solution_data_original.json
```

**copy_all_gams_files.py** (random_quadratic repo):
```
data/
  └── {strategy}/
      └── {mode}/
          └── {timestamp}/
              ├── solution_data.json
              └── model.gms
```

### 2. Processing Logic

| Aspect | find_and_copy_failed_logs.py | copy_all_gams_files.py |
|--------|------------------------------|------------------------|
| **Input** | All Excel entries, then filters for failures | All Excel entries (processes all) |
| **Output** | Copies entire log folders | Copies only GAMS files |
| **Filtering** | Identifies wrong optimal & false infeasible | No filtering (copies all from Excel) |
| **Archives** | Single results.xlsx file | Multiple archive directories |

### 3. Enhanced Reporting

The new script provides **improved tracking and reporting**:

#### Tracking Categories
1. **Successfully copied** - Files found and copied
2. **GAMS file not found** - No matching folder in data directory
3. **Not in name mapping** - Model not in name_mapping.json
4. **Problem type mismatch** - Conv/nonconv mismatch
5. **Copy errors** - File found but copy failed

#### Console Output
- Progress updates per archive
- Per-archive statistics (found/not found)
- Final summary with percentages
- Detailed breakdown of not found reasons

#### Summary File Contents
- Total Excel entries processed
- Successfully copied files (grouped by type, solver, strategy)
- NOT found files section with:
  - Detailed list of each category
  - First 20 entries per category
  - Model names and strategies

## Example Output Comparison

### find_and_copy_failed_logs.py
```
Successfully processed: 15/20 models
  - Wrong optimal solutions: 8
  - False infeasible solutions: 7
Could not find logs for: 5 models
```

### copy_all_gams_files.py
```
Total Excel entries: 3000
Files found and copied: 2850 (95.0%)
Files NOT found: 150 (5.0%)
  - GAMS file not found: 100
  - Not in name mapping: 30
  - Problem type mismatch: 15
  - Copy errors: 5
```

## Verification Process

Both scripts use **identical verification steps**:

1. ✓ Parse Excel timestamp
2. ✓ Find folders within time tolerance (±60 minutes)
3. ✓ Read JSON data from folder
4. ✓ Verify model parameters match
5. ✓ Verify objective value (within tolerance)
6. ✓ Verify solution time (within tolerance)
7. ✓ Return verified match or closest candidate

## Configuration

Both scripts support the same configuration parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `obj_tolerance` | 1e-4 | Objective value matching tolerance |
| `time_tolerance_minutes` | 60.0 | Time window for folder search |

## Error Handling

Both scripts handle the same error scenarios:

- ✓ Missing Excel files
- ✓ Invalid timestamps
- ✓ Missing JSON files
- ✓ Parameter mismatches
- ✓ Missing GAMS files
- ✓ Copy errors

## Conclusion

The `copy_all_gams_files.py` script:
- ✅ Uses the **same matching approach** as `find_and_copy_failed_logs.py`
- ✅ Processes **only entries from Excel files** (not arbitrary files)
- ✅ Provides **enhanced reporting** with detailed found/not found statistics
- ✅ Adapts to **different directory structure** (random_quadratic repo)
- ✅ Copies **only GAMS files** instead of entire log folders
- ✅ Uses **name_mapping.json** for consistent naming

The script is production-ready and follows best practices from the reference implementation.

