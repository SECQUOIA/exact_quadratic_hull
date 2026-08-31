# Exact Quadratic Hull

This package implements the exact hull reformulations and four quadratic GDP benchmarks used in *Exact Hull Reformulation for Quadratically Constrained Generalized Disjunctive Programs* by Sergey Gusev and David E. Bernal Neira.

The distribution is `exact-quadratic-hull`; the Python package is `exact_hull`. Importing it registers one general exact hull (GEHR), six conic forms (CEHR), and two clearly labeled experimental forms with Pyomo. It does not modify the installed Pyomo package.

## Install

Python 3.12 or newer is required. From the repository root:

```console
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
exact-hull doctor
```

For GAMS and solver setup, see [docs/installation.md](docs/installation.md).

## Quickstart

Inspect the self-contained smoke campaign without invoking a solver:

```console
exact-hull run configs/smoke.toml --dry-run
```

Run it on a machine with GAMS and SCIP:

```console
exact-hull run configs/smoke.toml --out results/smoke --jobs 2
exact-hull report results/smoke
exact-hull plot results/smoke
exact-hull verify results/smoke
```

Every solver job builds a fresh model and writes one atomic JSON result. `--resume` creates or continues a campaign, so it can be used from the first idempotent invocation:

```console
exact-hull run configs/random_psd.toml --resume results/random-psd --jobs 8
```

Run several configs through one worker pool by giving an output root; each campaign is stored under its config-file stem:

```console
exact-hull run configs/random_psd.toml configs/kmeans.toml --out results --jobs 8
```

Use `exact-hull inspect CONFIG` for transformation/presolve instrumentation, `exact-hull reference RUN_DIR` for certified objectives, and `exact-hull conic-bound CONFIG` for the independent CEHR relaxation oracle.

See [configuration](docs/configuration.md), [results](docs/results-schema.md), [transformations](docs/transformations.md), [experiments](docs/experiments.md), and [provenance](docs/provenance.md).

## Citation

Gusev, S., & Bernal Neira, D. E. (2025). *Exact Hull Reformulation for Quadratically Constrained Generalized Disjunctive Programs.* arXiv:2508.16093. https://arxiv.org/abs/2508.16093

## License

The package is released under the MIT License. See `LICENSE` and `NOTICE.md` for third-party attributions.
