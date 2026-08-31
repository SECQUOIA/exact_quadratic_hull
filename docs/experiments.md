# Experiments

Use `exact-hull run CONFIG --dry-run` to inspect a campaign. Remove `--dry-run` and set `--out DIR` on a configured GAMS machine. Use `exact-hull report DIR --xlsx` for a workbook or omit `--xlsx` for CSV. `exact-hull plot DIR --mode solve|root|relaxation --tolerance 1e-5` incrementally refreshes `verification.csv` at the requested tolerance before writing stable-style profiles; solve is the default.

| Config | Instances | Jobs |
| --- | ---: | ---: |
| `random_psd.toml` | 240 | 14,400 |
| `random_psd_conic.toml` | 18 | 432 |
| `random_nonconvex.toml` | 100 | 2,400 |
| `eps_relaxation.toml` | 240 | 7,680 |
| `kmeans.toml` | 96 | 3,456 |
| `cstr.toml` | 9 (`NT` 2–10) | 324 |
| `clay.toml` | 12 (6 layouts × 2 metrics) | 432 |
| `qualification.toml` | 1 | 54 |
| `smoke.toml` | 1 | 1 |

Total jobs equal instances × strategies × solvers × modes. Main configs contain Big-M, binary multiplication, epsilon hull, GEHR, and applicable CEHR arms. Multiple Big-M remains in the tractable CSTR, clay, k-means, and qualification campaigns; it is omitted from the two large random grids because a large instance would require roughly 21,000 M-estimation subsolves. `random_psd_conic.toml` is a six-encoding ablation on an 18-instance diagnostic subset and runs solve plus relaxation, not a core comparison. `eps_relaxation.toml` runs root plus relaxation. Every job rebuilds its model and solves in a private scratch directory.

Multiple Big-M uses a V1 adapter around GAMS/Gurobi because this environment exposes Gurobi through GAMS rather than a direct executable. Each M-estimation subproblem has a 30-second limit, `DualReductions 0`, and one forced thread. This adapter is used even on campaign rows whose final subsolver is SCIP, so those rows still require a working Gurobi license. The first mbigm job for an instance atomically publishes `mbigm/<instance_id>.json`; later modes and solver rows load and pass those exact values to the transformation, while missing cached values are recomputed through the same adapter. The cache records and validates the normalized mbigm-options fingerprint. That cache file, including its provenance, is the published record of the M values used. M-estimation time is reported separately and is not part of charged solve time or the shifted geometric mean.

Gurobi option files set `FuncNonlinear 1`, so quotient-based epsilon-hull functions use nonlinear handling rather than piecewise-linear function approximation. Ordinary jobs retain `NonConvex 2`; `variant="auto"` omits it to test native convexity recognition.

`eps_relaxation.toml` shares the instances and seed from `random_psd.toml`; its gaps are populated when certified references are joined, and its `bounds.csv` can also be joined with the `random_psd` campaign by `instance_id`.

SCIP `variant="convex"` records are diagnostics. GEHR and binary multiplication contain nonconvex defining functions under that flag, so these are intentional negative controls. Reports label them and exclude them from population ground truth and reference agreement. CSTR has nonconvex equalities and therefore has no convex-variant entry.

BARON was removed after the MPC review because of inconsistent optimality certificates on convex instances, heavy time-limit censoring, and the Gurobi license clause on comparisons with commercial solvers; the last revision with BARON support is commit d071e3b.

At campaign start, the runner atomically creates `manifest.json` with the normalized config, full plan, and Python/Pyomo/GAMS/Git provenance. Concurrent identical creation is tolerated. Resume warns on Python, Pyomo, or GAMS drift; `--strict-env` makes drift an error. A non-empty directory is otherwise accepted only with `--resume` and an exactly matching manifest.

Any config edit, including a label-only change or reordering strategies or solvers, requires a fresh output directory by design. `--limit` restricts only the jobs executed by that invocation; the manifest always records the complete campaign so a later unrestricted `--resume` can finish it.

Use `--rerun-status solver_error,timeout` with `--resume` to retry selected valid results. For N-way execution, start shard 1 first, then point every process at the same directory:

```console
exact-hull run CONFIG --out RUN_DIR --shard 1/N
exact-hull run CONFIG --resume RUN_DIR --shard 2/N
```

Continue through shard N. Selection uses the full-plan zero-based index modulo N; `--limit` applies within a shard. Per-run scratch and result paths prevent collisions.

## Instrumentation and certification

`exact-hull inspect CONFIG --out DIR` transforms the first instance for each strategy, writes LP artifacts where the writer supports the expression form, and writes `inspection.json` and `inspection.csv`. Add `--all-instances` for the full grid. If `gurobipy` or `pyscipopt` is importable, it also runs solver presolve; otherwise transform-time counts are still emitted with a clear skip message.

`exact-hull verify RUN_DIR --tolerance 1e-5` rebuilds each original GDP and writes `verification.csv`. It checks that every original variable has a stored finite value, agreement among duplicate binary representations, indicator integrality, bounds, global constraints, exactly-one selections, selected-disjunct constraints, logical constraints, and recomputed objectives. Outcomes are `verified_feasible`, `infeasible_point`, `fractional_indicators`, `objective_mismatch`, `no_solution`, and `not_verifiable`. `report` and `reference` accept the same `--tolerance` option and reuse only rows whose stored result timestamp and tolerance still match; new or rerun records are verified incrementally. Add `--reverify` to rebuild every row.

`exact-hull reference RUN_DIR --cap 4096` writes versioned `references.json` incrementally and skips already certified instances on rerun. It first enumerates small selection spaces and solves each fixed-disjunction subproblem globally with GAMS/SCIP; leftover logical binaries are allowed. Enumeration solves use a 300-second limit, relative gap `1e-8`, and absolute gap `1e-10`, recorded only on enumeration-route provenance. Above the cap it requires feasibility-verified optimal records from two different subsolvers that agree within tolerance; agreement provenance instead identifies those source records and subsolvers. Otherwise the instance remains `reference_unknown`. Existing verification is reused unless `--reverify` is given.

`exact-hull conic-bound CONFIG --out DIR` constructs binary-free CEHR relaxations and submits their QCP form through `gurobipy`, or direct MOSEK when available, without setting Gurobi `NonConvex`. JSON/CSV output is keyed by content-derived instance ID and is rewritten after each instance. Export, recognition, license, and solve failures produce a null bound plus an explanatory per-instance status and do not stop later instances. With neither backend, the command exits immediately with an actionable error; model construction remains independently testable without a solver.

## Qualification gate

Run `configs/qualification.toml` before a full campaign. Check that structural and path counts are populated, relaxations have no discrete variables, roots stop at the node limit or clean optimum, GDX and log bounds are coherent, verification passes, and solver version/presolve fields are populated.
