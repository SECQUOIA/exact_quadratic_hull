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

