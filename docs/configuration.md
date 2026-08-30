# Configuration

Experiment files use TOML and Python's standard `tomllib`. A file has one `[experiment]`, one `[instances]`, and arrays of strategy and solver tables. List-valued fields directly under `[instances]` form a Cartesian grid. Values under `[instances.fixed]` are not expanded. Fixed values and axes are merged before defaults are injected, then numeric values are normalized element by element before seeds and job identities are derived. Equivalent scalar values have the same identity whether they are an axis, fixed, or an explicit default.

```toml
[experiment]
benchmark = "random_quadratic"
base_seed = 7
time_limit = 600

[instances]
n_dimensions = [2, 3]
n_disjunctions = 1
n_disjuncts_per_disjunction = 2
n_constraints_per_disjunct = 1
n_feasible_regions = 1
ensure_positive_definite = true

[instances.fixed]
coeff_range = [-5.0, 5.0]

[[strategies]]
name = "gdp.hull"
label = "hull-eps-1e-4"
[strategies.options]
EPS = 1e-4

[[strategies]]
name = "gdp.hull_exact"
label = "GEHR"

[[solvers]]
name = "gams"
subsolver = "scip"
variant = "convex"
```

`benchmark` is one of `random_quadratic`, `kmeans`, `cstr`, or `clay`. Strategy options are explicit values passed to Pyomo; labels are presentation names only and do not affect job fingerprints. Computationally duplicate strategies and labels shared by different strategies are rejected. Unknown instance keys, transformations, subsolvers, and solver variants fail while loading the config. The only solver variant is `variant = "convex"` for SCIP; omit `variant` for Gurobi, BARON, or ordinary SCIP runs.

A deterministic seed is derived from `base_seed` and each instance parameter map. It does not depend on Python's process-randomized hash or NumPy's global random state.
