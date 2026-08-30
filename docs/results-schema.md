# Results schema

Each planned job writes `jobs/<run_id>/result.json` atomically. A failure still writes a record. The campaign's atomic `manifest.json` records the normalized config, every planned job fingerprint, and every planned instance. Resume requires that manifest to match; individual result files mark completed jobs.

| Field | Meaning |
| --- | --- |
| `run_id` | Stable identifier derived from all model and solve inputs, including time limit and solver variant |
| `benchmark`, `instance_id`, `instance_params`, `seed` | Instance identity and generation inputs |
| `strategy`, `transformation`, `transformation_options` | Display label, registered Pyomo transformation name, and explicit options |
| `solver`, `subsolver`, `variant`, `time_limit` | GAMS interface, selected solver, solver option variant, and limit |
| `duration_sec` | Wall-clock duration for the complete job, including model construction, transformation, solve, and result extraction |
| `solver_time_sec` | GAMS `ETSOLVE` reported by Pyomo as solver user time; performance profiles use this field |
| `status` | `optimal`, `globally_optimal`, `locally_optimal`, `feasible`, `timeout`, `infeasible`, `solver_error`, `build_error`, or `transform_error` |
| `objective`, `lower_bound`, `upper_bound`, `abs_gap`, `rel_gap` | Objective and bound data when available |
| `root_relaxation` | Parsed root relaxation when available |
| `solution` | Benchmark-specific JSON payload |
| `timestamp` | UTC completion timestamp |
| `versions` | Python, Pyomo, GAMS, subsolver, and Git revision information. Subsolver version is null because the GAMS interface does not expose it reliably. |
| `error` | Failure details, otherwise null |

Ground truth uses finite objectives from only `optimal` and `globally_optimal` records, never merely local optima. Correctness uses `abs(value - truth) <= max(atol, rtol * abs(truth))`, so `atol` is the fallback near zero. Verified records outside that tolerance count as failures in performance profiles.

Reports identify a method by `(strategy, solver, subsolver, variant)`. They create a separate profile for each solver, subsolver, and variant and use strategies as curves; timings from different solvers are never pooled into one curve.

`exact-hull report` writes both the record-level `results.csv` (or `results.xlsx`) and a grouped `summary.csv`.
