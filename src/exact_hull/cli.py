"""Command-line interface for exact-hull experiments."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pyomo
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Objective,
    SolverFactory,
    TransformationFactory,
    Var,
)
from pyomo.opt import SolverStatus

from exact_hull.analysis.tables import bounds, summary
from exact_hull.experiment.results import aggregate, read_campaign
from exact_hull.experiment.runner import expand_jobs, load_config, run
from exact_hull.transformations import TRANSFORMATIONS


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _gdx_results_available() -> bool:
    from pyomo.solvers.plugins.solvers.GAMS import gdxcc_available

    return bool(gdxcc_available)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exact-hull")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run or inspect an experiment")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--out", type=Path, default=Path("results"))
    run_parser.add_argument("--limit", type=_nonnegative_int)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--resume", type=Path)
    report = subparsers.add_parser("report", help="aggregate per-job results")
    report.add_argument("rundir", type=Path)
    report.add_argument("--xlsx", action="store_true")
    plot = subparsers.add_parser("plot", help="plot an experiment")
    plot.add_argument("rundir", type=Path)
    subparsers.add_parser("doctor", help="check runtime dependencies")
    subparsers.add_parser("list-transformations", help="list registered transformations")
    return parser


def _doctor() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"pyomo: {pyomo.version.__version__}")
    for name in TRANSFORMATIONS:
        state = "registered" if TransformationFactory(name) is not None else "missing"
        print(f"{name}: {state}")
    gams = shutil.which("gams")
    print(f"gams: {gams or 'not found'}")
    if _gdx_results_available():
        print("gams GDX results: available")
    else:
        print(
            "WARNING: Pyomo GAMS GDX results are unavailable; its dat fallback can crash "
            "on OBJVAL NA records produced by timed-out solves."
        )
    if gams:
        state = "available" if SolverFactory("gams").available(False) else "unavailable"
        print(f"gams interface: {state}")
        if state == "available":
            for subsolver in ("gurobi", "scip"):
                # A tiny MIP: every campaign subsolver is registered for MIP in
                # Pyomo's GAMS capability table, whereas SCIP is not listed for LP.
                model = ConcreteModel()
                model.x = Var(bounds=(0, None))
                model.y = Var(domain=Binary)
                model.objective = Objective(expr=model.x + model.y)
                try:
                    with tempfile.TemporaryDirectory(prefix="exact-hull-doctor-") as scratch:
                        result = SolverFactory("gams").solve(
                            model,
                            solver=subsolver,
                            tee=False,
                            keepfiles=False,
                            tmpdir=scratch,
                            add_options=["option reslim=10;"],
                        )
                    available = result.solver.status not in {
                        SolverStatus.error,
                        SolverStatus.aborted,
                    }
                    detail = "available" if available else str(result.solver.status)
                except Exception as error:  # doctor must report missing licenses without aborting
                    detail = f"unavailable ({error})"
                print(f"gams/{subsolver}: {detail}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "list-transformations":
        for name in TRANSFORMATIONS:
            print(name)
        return 0
    if args.command == "run":
        config = load_config(args.config.resolve())
        jobs = expand_jobs(config)
        if args.limit is not None:
            jobs = jobs[: args.limit]
        instance_count = len({job.instance_id for job in jobs})
        strategy_count = len({(job.strategy, job.label) for job in jobs})
        solver_count = len({(job.solver, job.subsolver, job.variant) for job in jobs})
        mode_count = len({job.mode for job in jobs})
        print(
            f"planned jobs: {len(jobs)}; instances: {instance_count}; "
            f"strategies: {strategy_count}; solvers: {solver_count}; modes: {mode_count}"
        )
        for job in jobs:
            print(
                f"{job.run_id} {job.benchmark} {job.instance_id} "
                f"{job.label} {job.solver}/{job.subsolver} {job.mode}"
            )
        if args.dry_run:
            return 0
        destination = args.resume.resolve() if args.resume else args.out.resolve()
        run(args.config.resolve(), destination, args.limit, resume=args.resume is not None)
        return 0
    if args.command == "report":
        run_directory = args.rundir.resolve()
        _, records = read_campaign(run_directory)
        print(aggregate(run_directory, xlsx=args.xlsx, records=records))
        summary_path = run_directory / "summary.csv"
        summary(records).reset_index().to_csv(summary_path, index=False)
        print(summary_path)
        bounds_path = run_directory / "bounds.csv"
        bounds_frame = bounds(records)
        bounds_frame.to_csv(bounds_path, index=False)
        print(bounds_path)
        mismatches = bounds_frame.loc[
            bounds_frame["relaxation_matches_cehr"].eq(False), "instance_id"
        ].unique()
        if len(mismatches):
            print(
                "WARNING: GEHR and CEHR relaxation bounds differ for instances: "
                + ", ".join(sorted(mismatches))
            )
        return 0
    if args.command == "plot":
        from exact_hull.analysis.plots import plot_run

        for path in plot_run(args.rundir.resolve()):
            print(path)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
