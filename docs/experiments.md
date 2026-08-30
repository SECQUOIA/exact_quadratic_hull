# Experiments

Use `exact-hull run CONFIG --dry-run` to inspect a campaign. Remove `--dry-run` and set `--out DIR` on a configured GAMS machine. Use `exact-hull report DIR --xlsx` for a workbook or omit `--xlsx` for CSV. `exact-hull plot DIR` writes stable-style performance plots.

| Config | Instances |
| --- | ---: |
| `random_psd.toml` | 240 |
| `random_psd_conic.toml` | 240 |
| `random_nonconvex.toml` | 100 |
| `eps_relaxation.toml` | 240 |
| `kmeans.toml` | 96 (dimensions 2–5, clusters 3–5, points 10–17) |
| `cstr.toml` | 9 (`NT` 2–10) |
| `clay.toml` | 12 (6 layouts × 2 metrics) |
| `smoke.toml` | 1 |

Total jobs equal instances × strategies × solvers × modes. The maintained campaign configs run `solve`, `root`, and `relaxation` modes; `smoke.toml` keeps the default `solve` mode, and `eps_relaxation.toml` runs only continuous relaxations. Every job rebuilds its model and solves in a private scratch directory that the runner deletes afterwards (Pyomo's own cleanup fails on no-incumbent timeouts, where GAMS writes no primal results file). Solutions are loaded only when the solver returned one; a timed-out job without an incumbent records no objective. The runner detects Gurobi and SCIP timeouts from solver log messages and programmatic GAMS termination data. It also rejects a nominally optimal result at the time limit when the final gap is nonzero or unknown. Root node-limit termination is detected from the solver log marker (`Node limit reached` or `node limit reached`) and recorded as `node_limit` whether or not a finite bound came back; other warning, aborted, and licensing statuses are solver errors.

`eps_relaxation.toml` shares the instances and seed from `random_psd.toml`; it produces no `solve` records, so its gaps are NaN, and its `bounds.csv` is intended to be joined with the `random_psd` campaign's `bounds.csv` by `instance_id`.

BARON was removed after the MPC review because of inconsistent optimality certificates on convex instances, heavy time-limit censoring, and the Gurobi license clause on comparisons with commercial solvers; the last revision with BARON support is commit d071e3b.

At the start of a campaign, the runner atomically writes `manifest.json` with the normalized config, planned jobs, and planned instances. A non-empty output directory is accepted only with `--resume` and an exactly matching manifest. Reports use the manifest's full instance denominator and ignore result files that do not belong to the plan.

Any config edit, including a label-only change or reordering strategies or solvers, requires a fresh output directory by design. `--limit` restricts only the jobs executed by that invocation; the manifest always records the complete campaign so a later unrestricted `--resume` can finish it.
