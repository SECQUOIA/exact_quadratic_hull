# Experiments

Use `exact-hull run CONFIG --dry-run` to inspect a campaign. Remove `--dry-run` and set `--out DIR` on a configured GAMS machine. Use `exact-hull report DIR --xlsx` for a workbook or omit `--xlsx` for CSV. `exact-hull plot DIR` writes stable-style performance plots.

| Config | Instances |
| --- | ---: |
| `random_psd.toml` | 240 |
| `random_psd_conic.toml` | 240 |
| `random_nonconvex.toml` | 100 |
| `kmeans.toml` | 96 (dimensions 2–5, clusters 3–5, points 10–17) |
| `cstr.toml` | 9 (`NT` 2–10) |
| `clay.toml` | 12 (6 layouts × 2 metrics) |
| `smoke.toml` | 1 |

Total jobs equal instances × strategies × solvers. Every job rebuilds its model and solves in a private scratch directory that the runner deletes afterwards (Pyomo's own cleanup fails on no-incumbent timeouts, where GAMS writes no primal results file). Solutions are loaded only when the solver returned one; a timed-out job without an incumbent records no objective. The runner detects timeouts from Gurobi, BARON, and SCIP log messages, with GAMS solver status as an additional signal when listing text is available. It also rejects a nominally optimal result at the time limit when the final gap is nonzero or unknown. Aborted, warning, and licensing statuses are solver errors, not verified solutions.

At the start of a campaign, the runner atomically writes `manifest.json` with the normalized config, planned jobs, and planned instances. A non-empty output directory is accepted only with `--resume` and an exactly matching manifest. Reports use the manifest's full instance denominator and ignore result files that do not belong to the plan.

Any config edit, including a label-only change or reordering strategies or solvers, requires a fresh output directory by design. `--limit` restricts only the jobs executed by that invocation; the manifest always records the complete campaign so a later unrestricted `--resume` can finish it.
