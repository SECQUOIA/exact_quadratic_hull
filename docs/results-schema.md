# Results schema

Each planned job writes `jobs/<run_id>/result.json` atomically. A failure still writes a record. The campaign's atomic `manifest.json` records the normalized config, every planned job fingerprint, and every planned instance. Resume requires that manifest to match; individual result files mark completed jobs.

| Field | Meaning |
| --- | --- |
| `run_id` | Stable identifier derived from all model and solve inputs, including mode, its effective time limit, solver variant, and full solver options |
| `benchmark`, `instance_id`, `instance_params`, `seed` | Instance identity and generation inputs |
| `strategy`, `transformation`, `transformation_options` | Display label, registered Pyomo transformation name, and explicit options |
| `solver`, `subsolver`, `variant`, `mode`, `time_limit` | GAMS interface, selected solver, solver option variant, job mode, and mode-specific limit |
| `duration_sec` | Wall-clock duration for the complete job, including model construction, transformation, solve, and result extraction |
| `solver_time_sec` | GAMS `ETSOLVE` reported by Pyomo as solver user time; performance profiles use this field |
| `status` | `optimal`, `globally_optimal`, `locally_optimal`, `feasible`, `timeout`, `node_limit`, `infeasible`, `solver_error`, `build_error`, or `transform_error` |
| `objective`, `lower_bound`, `upper_bound`, `abs_gap`, `rel_gap` | Objective and bound data when available |
| `num_variables`, `num_constraints`, `num_nonzeros`, `num_discrete_variables` | GAMS model statistics returned through GDX |
| `solver_status`, `termination` | Pyomo solver status and termination condition strings |
| `solution` | Benchmark-specific JSON payload |
| `timestamp` | UTC completion timestamp |
| `versions` | Python, Pyomo, GAMS, subsolver, and Git revision information. Subsolver version is null because the GAMS interface does not expose it reliably. |
| `error` | Failure details, otherwise null |

Ground truth uses finite objectives from only `optimal` and `globally_optimal` records, never merely local optima. Correctness uses `abs(value - truth) <= max(atol, rtol * abs(truth))`, so `atol` is the fallback near zero. Verified records outside that tolerance count as failures in performance profiles.

The `solve` mode runs the complete transformed MINLP and is the only mode used for ground truth, summaries, and performance profiles. The `root` mode stops after root-node processing. The solver log markers `Node limit reached` and `node limit reached` identify `node_limit` termination whether or not a finite bound was returned; this status is not verified optimal. The `relaxation` mode applies `core.relax_integer_vars` after the GDP transformation and solves the resulting continuous model; a nonzero discrete-variable count is a `transform_error`. Bounds are read from GDX (`OBJEST` and `OBJVAL`), with solver-log parsing retained only as a fallback when GDX reports a missing bound. Non-finite values and solver infinity sentinels are stored as null.

Reports identify a method by `(strategy, solver, subsolver, variant)`. They create a separate profile for each solver, subsolver, and variant and use strategies as curves; timings from different solvers are never pooled into one curve.

`exact-hull report` writes the record-level `results.csv` (or `results.xlsx`), a solve-only grouped `summary.csv`, and `bounds.csv`. The bounds table combines modes by instance, formulation, solver, subsolver, and variant. It reports solve objective/bound/status, root bound/status, relaxation bound/value/status/certification, ground truth, and relative root and relaxation gaps. `relaxation_certified` means that the primal value and global dual bound agree within the correctness tolerance. `root_gap` is reported only for completed roots with status `node_limit`, `optimal`, or `globally_optimal`. On GEHR rows, `relaxation_matches_cehr` compares the relaxation bound with available certified CEHR variants only when GEHR and every compared CEHR row are certified; the command warns when that certified comparison is false.

`eps_relaxation.toml` shares `random_psd.toml`'s instances and seed. Because it has no `solve` records, its gaps are NaN; join its `bounds.csv` with the `random_psd` campaign's `bounds.csv` by `instance_id`.
