# Provenance

The original repository used full copies of Pyomo's hull transformation. At Pyomo 6.10.0, each active copy differed from upstream principally in `_transform_constraint`; the two experimental auxiliary-variable versions also changed initialization. The package now subclasses Pyomo's pinned transformation and owns one shared override. `NOTICE.md` records the BSD-3-Clause attribution.

The random-quadratic and k-means builders were consolidated from their original top-level benchmark directories. Their random generation now uses local `numpy.random.default_rng` instances. The random-quadratic builder deliberately derives a new sub-seed for each generated quadratic. The old builder reset one seed for every call and therefore repeated the same quadratic matrix across constraints; new instances are genuinely random while remaining reproducible.

`src/exact_hull/benchmarks/cstr.py` was ported from the original `cstr/gdp_reactor.py`, which was byte-identical to `reference_repositories/gdplib/gdplib/cstr_v2/gdp_reactor_gurobi.py`. The former `cstr/batch_run.py` derives from `cstr_v2/execute_v3.py`. The differently named `cstr_v2/gdp_reactor.py` is a later abandoned divergent model and was not used. Only the original GDP mode was retained. The correct source is Liñán et al. (2020), DOI 10.1016/j.compchemeng.2020.106794. For `NT=5`, gdplib reports the best-known objective 3.06181298849707.

The constrained-layout model is vendored from Pyomo-examples. Its original header is preserved; `clay.py` is only an adapter.

The transformation port keeps diagonal quadratic terms as `v**2` in GEHR, CEHR, and both equality and inequality paths of the experimental variants, matching the old GAMS `power(v,2)` emission rather than changing them to `v*v`. It intentionally corrects the old ranged-constraint lower-side sign error. The two experimental extra-variable formulations retain their original lifted equality form: a scalar `t` links the linear part and `v'Qv = t*y` links the quadratic part.
