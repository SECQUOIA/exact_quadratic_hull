# Experiments

Use `exact-hull run CONFIG --dry-run` to inspect a campaign. Remove `--dry-run` and set `--out DIR` on a configured GAMS machine. Add `--jobs N` to run up to N solver jobs in parallel processes. Use `exact-hull report DIR --xlsx` for a workbook or omit `--xlsx` for CSV. `exact-hull plot DIR --mode solve|root|relaxation --tolerance 1e-5` incrementally refreshes `verification.csv` at the requested tolerance before writing stable-style profiles; solve is the default.

| Config | Instances | Jobs |
| --- | ---: | ---: |
| `random_psd.toml` | 36 | 1,080 |
| `random_psd_scip_convex.toml` | 36 | 360 |
| `random_psd_gurobi_auto.toml` | 36 | 216 |
| `random_psd_conic.toml` | 18 | 432 |
| `random_nonconvex.toml` | 24 | 720 |
| `eps_relaxation.toml` | 36 | 1,296 |
| `kmeans.toml` | 48 | 1,728 |
| `cstr.toml` | 9 (`NT` 2–10) | 180 |
| `clay.toml` | 6 | 216 |
| `qualification.toml` | 1 | 54 |
| `smoke.toml` | 1 | 1 |

The plan contains 6,282 jobs plus the smoke job. Total jobs equal instances × strategies × solvers × modes. Main configs contain Big-M, binary multiplication, epsilon hull, GEHR, and applicable CEHR arms; multiple Big-M is also included in the targeted nonconvex grid. `random_psd_conic.toml` is a six-encoding ablation on an 18-instance diagnostic subset and runs solve plus relaxation, not a core comparison. `eps_relaxation.toml` runs six epsilon-hull values from 1e-6 through 1e-1 in solve, root, and relaxation modes. The solve sweep captures observed GAMS/Gurobi failures: certified-wrong optima at 1e-4 and spurious infeasibility at 1e-3 and 1e-5; SCIP is correct at those values but also misbehaves at 1e-6. Shared instance IDs still support cross-campaign checks against the hull, GEHR, and CEHR rows in `random_psd.toml`. Every job rebuilds its model and solves in a private scratch directory.

The random-nonconvex campaign reports relaxation bounds where primal-dual agreement and reference consistency certify them, without attaching a hull claim. Relaxation mode is omitted from CSTR because the relaxed model is a nonconvex global problem with no attached hull claim. On convex families, GEHR relaxations remain as hull-identity checks against CEHR through `relaxation_matches_cehr`. Solver variants have dedicated configs so core grids have no dead cells. Gurobi `variant="auto"` is not run for GEHR or binary multiplication because their defining functions fail Gurobi's convexity check by construction.

For CSTR at `NT=2`, the CEHR probe recorded `n_cone_rows=0`, `n_fallback_rows=2`, and two equality fallbacks. Every disjunct quadratic is an equality, so CEHR falls back to GEHR throughout; the redundant `CEHR-mixed` arm was removed. Clay uses only the `l1` metric because `l2` changes only the objective to a nonquadratic square-root form.

Multiple Big-M uses a V1 adapter around GAMS/Gurobi because this environment exposes Gurobi through GAMS rather than a direct executable. Each M-estimation subproblem has a 30-second limit, `DualReductions 0`, and one forced thread. This adapter is used even on campaign rows whose final subsolver is SCIP, so those rows still require a working Gurobi license. The first mbigm job for an instance atomically publishes `mbigm/<instance_id>.json`; later modes and solver rows load and pass those exact values to the transformation, while missing cached values are recomputed through the same adapter. The cache validates its schema version and normalized mbigm-options fingerprint and records the instance's total estimation time. That cache file, including its provenance, is the published record of the M values used. M-estimation time is reported separately and is not part of charged solve time or the shifted geometric mean.

Run campaigns containing `gdp.mbigm` under one coordinating process with `--jobs N`, rather than independent `--shard` processes, because only the in-process gate serializes M estimation for each instance.

Gurobi option files set `FuncNonlinear 1`, so quotient-based epsilon-hull functions use nonlinear handling rather than piecewise-linear function approximation. Ordinary jobs retain `NonConvex 2`; `variant="auto"` omits it to test native convexity recognition.

`eps_relaxation.toml` shares the instances and seed from `random_psd.toml`; its gaps are populated when certified references are joined, and its `bounds.csv` can also be joined with the `random_psd` campaign by `instance_id`.

Derive references once in the `random_psd.toml` run directory, then copy its `references.json` into the `random_psd_scip_convex.toml`, `random_psd_gurobi_auto.toml`, and `eps_relaxation.toml` run directories before running `report`. These copies are inputs to `report` only: do not run `exact-hull reference` in the destination directories because agreement certificates would be re-derived from local records that lack a second subsolver and could be demoted; re-copy the file from `random_psd` instead. All four campaigns use the same 36 instance IDs.

SCIP `variant="convex"` records are diagnostics. GEHR and binary multiplication contain nonconvex defining functions under that flag, so only those rows are intentional negative controls. Big-M, epsilon hull, and CEHR are valid recognition controls. Reports conservatively exclude every convex-variant row from population ground truth and reference agreement. CSTR has nonconvex equalities and therefore has no convex-variant entry.

BARON was removed after the MPC review because of inconsistent optimality certificates on convex instances, heavy time-limit censoring, and the Gurobi license clause on comparisons with commercial solvers; the last revision with BARON support is commit d071e3b.

At campaign start, the runner atomically creates `manifest.json` with the normalized config, full plan, and Python/Pyomo/GAMS/Git provenance. Concurrent identical creation is tolerated. Resume warns on Python, Pyomo, GAMS, or Git-revision drift; `--strict-env` makes drift an error. Git drift compares HEAD commits only, so uncommitted edits are not detected; under `--strict-env`, any new commit between resume invocations is an error. A non-empty directory is otherwise accepted only with `--resume` and an exactly matching manifest.

Any config edit, including a label-only change or reordering strategies or solvers, requires a fresh output directory by design. `--resume RUN_DIR` means create or continue: using it on the first invocation makes the command idempotent. Under resume, `--limit N` means consider only the first N planned jobs, not run N additional jobs. The manifest always records the complete campaign so a later unrestricted `--resume` can finish it.

Several configs can share one worker pool. The output or resume path is then a root, and each campaign uses a directory named for its config-file stem:

```console
exact-hull run configs/random_psd.toml configs/kmeans.toml --out results --jobs 8
```

This writes `results/random_psd` and `results/kmeans`. Config stems must be unique, and `--limit` and `--shard` are unavailable in a multi-config invocation. Pass a per-stem campaign directory, such as `results/random_psd`, rather than the multi-config root to `report`, `plot`, `verify`, and `reference`.

Use `--rerun-status solver_error,timeout` with `--resume` to retry selected valid results. `--shard` remains available for multi-machine campaigns and composes with per-machine `--jobs`. For N-way execution, start shard 1 first, then point every process at the same directory:

```console
exact-hull run CONFIG --out RUN_DIR --shard 1/N --jobs J
exact-hull run CONFIG --resume RUN_DIR --shard 2/N --jobs J
```

Continue through shard N. Selection uses the full-plan zero-based index modulo N; `--limit` applies within a shard. Per-run scratch and result paths prevent collisions.

Choose `--jobs` with timing fidelity in mind. Recorded solve times and time limits are wall-clock, so CPU or memory contention inflates timings and increases timeout censoring. Stay at or below the physical core count with RAM headroom. Before a full campaign, compare `configs/qualification.toml` at `--jobs 1` and at the target level, then join the two `results.csv` files on `run_id` to check the timing effect.

## Instrumentation and certification

`exact-hull inspect CONFIG --out DIR` transforms the first instance for each strategy, writes LP artifacts where the writer supports the expression form, and writes `inspection.json` and `inspection.csv`. Add `--all-instances` for the full grid. If `gurobipy` or `pyscipopt` is importable, it also runs solver presolve; the Gurobi probe records and uses `NonConvex=2`, `FuncNonlinear=1`, and `Threads=1`. The campaign path goes through GAMS, which adds a reformulation layer, so per-job log-derived `presolved_num_*` fields are the primary presolve evidence and inspection is supplementary. The LP/MPS writers cannot export the epsilon-hull quotient, so epsilon arms have no inspection datum.

`exact-hull verify RUN_DIR --tolerance 1e-5` rebuilds each original GDP and writes `verification.csv`. It checks that every original variable has a stored finite value, agreement among duplicate binary representations, indicator integrality, bounds, global constraints, exactly-one selections, selected-disjunct constraints, logical constraints, and recomputed objectives. Outcomes are `verified_feasible`, `infeasible_point`, `fractional_indicators`, `objective_mismatch`, `no_solution`, and `not_verifiable`. `report` and `reference` accept the same `--tolerance` option and reuse only rows whose stored result timestamp and tolerance still match; new or rerun records are verified incrementally. Add `--reverify` to rebuild every row.

`exact-hull reference RUN_DIR --cap 4096` writes versioned `references.json` incrementally. It first enumerates small selection spaces and solves each fixed-disjunction subproblem globally with GAMS/SCIP; leftover logical binaries are allowed. Enumeration solves use a 300-second limit, relative gap `1e-8`, and absolute gap `1e-10`, recorded only on enumeration-route provenance. When enumeration does not certify an objective, it selects the minimum-objective agreeing pair of feasibility-verified optimal records from two different subsolvers. A verified-feasible record below that value beyond tolerance prevents certification and is recorded as conflict provenance. Enumeration certificates are authoritative and reused on rerun. Every other instance re-attempts enumeration on each invocation: raising `--cap` can upgrade an agreement certificate to an enumeration certificate, while an in-cap instance whose enumeration keeps failing re-solves its selection space each time. Use `--cap 0` to derive references from records only. Agreement certificates are then re-derived from the current records so they can be improved or falsified; demotions and improvements of previous agreement certificates are reported on stderr. Otherwise the instance remains `reference_unknown`. Existing verification is reused unless `--reverify` is given.

`exact-hull conic-bound CONFIG --out DIR` constructs the factorized, binary-free CEHR relaxation and submits its QCP through `gurobipy` only. Gurobi uses a 600-second limit and one thread. Output records status, primal objective, any available dual bound, runtime, and backend provenance; a finite objective from an optimal solve is the accepted oracle bound. An instance with any CEHR fallback row is refused because it is not a pure conic-hull oracle. JSON/CSV output is keyed by content-derived instance ID and is rewritten after each instance. Export, recognition, license, and solve failures produce a null bound plus an explanatory per-instance status and do not stop later instances. Without `gurobipy`, the command exits immediately with an actionable error; model construction remains independently testable without a solver. MOSEK support would require genuine Pyomo conic components and is not implemented.

## Qualification gate

Run `configs/qualification.toml` before a full campaign. Check that structural and path counts are populated, relaxations have no discrete variables, roots stop at the node limit or clean optimum, GDX and log bounds are coherent, verification passes, and solver version/presolve fields are populated.
