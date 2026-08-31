# Transformations

Import `exact_hull` once before requesting a transformation through Pyomo's `TransformationFactory`.

| Registered name | Paper/experiment name | Module |
| --- | --- | --- |
| `gdp.hull_exact` | GEHR | `general.py` |
| `gdp.hull_exact_conic_no_cholesky` | CEHR, no factorization (default) | `conic.py` |
| `gdp.hull_exact_conic_original` | CEHR, only factorized | `conic.py` |
| `gdp.hull_exact_conic_no_sqrt_no_extra_var` | CEHR, squared SOC, inline | `conic.py` |
| `gdp.hull_exact_conic_no_sqrt_extra_var` | CEHR, squared SOC, auxiliary variables | `conic.py` |
| `gdp.hull_exact_conic_sqrt_no_extra_var` | CEHR, square-root SOC, inline | `conic.py` |
| `gdp.hull_exact_conic_sqrt_extra_var` | CEHR, square-root SOC, auxiliary variables | `conic.py` |
| `gdp.hull_exact_extra_var` | Experimental, equality link | `experimental.py` |
| `gdp.hull_exact_extra_var_inequal` | Experimental, inequality link | `experimental.py` |

The package does not register or replace Pyomo's deprecated `gdp.chull` alias.

CEHR emits a cone side when its quadratic matrix satisfies
`min_eigenvalue >= -1e-9 * max(1, abs(max_eigenvalue))`. Other quadratic sides and all
quadratic equalities use the GEHR homogenized fallback. The transformation records cone,
fallback, equality-fallback, and epigraph-variable counts on the model for experiment
instrumentation.

For factorized CEHR encodings, accepted negative eigenvalues are clamped to zero and every
strictly positive eigenvalue is retained in the factor. The `no_cholesky` encoding applies
the same relative PSD acceptance test but does not construct the unused factor.
