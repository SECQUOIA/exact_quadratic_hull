"""Command-line interface for exact-hull experiments."""

from __future__ import annotations

import argparse
import json
import math
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

from exact_hull.analysis.tables import bounds, censoring, methods, summary
from exact_hull.experiment.results import RUN_STATUSES, aggregate, read_campaign
from exact_hull.experiment.runner import expand_jobs, load_config, run, run_campaigns
from exact_hull.transformations import TRANSFORMATIONS


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _shard(value: str) -> tuple[int, int]:
    try:
        selected, total = (int(part) for part in value.split("/", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be K/N with positive integers") from error
    if selected < 1 or total < 1 or selected > total:
        raise argparse.ArgumentTypeError("must satisfy 1 <= K <= N")
    return selected, total


def _statuses(value: str) -> set[str]:
    statuses = {item.strip() for item in value.split(",") if item.strip()}
    unknown = statuses - RUN_STATUSES
    if not statuses or unknown:
        raise argparse.ArgumentTypeError(
            "statuses must be a comma-separated subset of " + ", ".join(sorted(RUN_STATUSES))
        )
    return statuses


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _gdx_results_available() -> bool:
    from pyomo.solvers.plugins.solvers.GAMS import gdxcc_available

    return bool(gdxcc_available)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exact-hull")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run or inspect an experiment")
    run_parser.add_argument("config", type=Path, nargs="+")
    run_parser.add_argument("--out", type=Path, default=Path("results"))
    run_parser.add_argument("--jobs", type=_positive_int, default=1)
    run_parser.add_argument("--limit", type=_nonnegative_int)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--resume", type=Path)
    run_parser.add_argument("--rerun-status", type=_statuses, default=set())
    run_parser.add_argument("--shard", type=_shard)
    run_parser.add_argument("--strict-env", action="store_true")
    report = subparsers.add_parser("report", help="aggregate per-job results")
    report.add_argument("rundir", type=Path)
    report.add_argument("--xlsx", action="store_true")
    report.add_argument("--reverify", action="store_true")
    report.add_argument("--tolerance", type=_positive_float, default=1e-5)
    plot = subparsers.add_parser("plot", help="plot an experiment")
    plot.add_argument("rundir", type=Path)
    plot.add_argument("--mode", choices=("solve", "root", "relaxation"), default="solve")
    plot.add_argument("--tolerance", type=_positive_float, default=1e-5)
    inspect = subparsers.add_parser("inspect", help="inspect transformed model structure")
    inspect.add_argument("config", type=Path)
    inspect.add_argument("--out", type=Path, default=Path("inspection"))
    inspect.add_argument("--all-instances", action="store_true")
    verify = subparsers.add_parser("verify", help="verify stored GDP solutions")
    verify.add_argument("rundir", type=Path)
    verify.add_argument("--tolerance", type=_positive_float, default=1e-5)
    reference = subparsers.add_parser("reference", help="derive certified references")
    reference.add_argument("rundir", type=Path)
    reference.add_argument("--cap", type=_nonnegative_int, default=4096)
    reference.add_argument("--reverify", action="store_true")
    reference.add_argument("--tolerance", type=_positive_float, default=1e-5)
    conic = subparsers.add_parser("conic-bound", help="run the independent CEHR oracle")
    conic.add_argument("config", type=Path)
    conic.add_argument("--out", type=Path, default=Path("conic-bounds"))
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
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "list-transformations":
        for name in TRANSFORMATIONS:
            print(name)
        return 0
    if args.command == "run":
        if args.rerun_status and args.resume is None:
            parser.error("--rerun-status requires --resume")
        if args.strict_env and args.resume is None:
            parser.error("--strict-env requires --resume")
        config_paths = [path.resolve() for path in args.config]
        if len(config_paths) > 1 and args.limit is not None:
            parser.error("--limit cannot be used with multiple configs")
        if len(config_paths) > 1 and args.shard is not None:
            parser.error("--shard cannot be used with multiple configs")
        stems = [path.stem for path in config_paths]
        if len(config_paths) > 1 and any(stem in {".", ".."} for stem in stems):
            parser.error("config file stems '.' and '..' are not allowed")
        if len(set(stems)) != len(stems):
            parser.error("config file stems must be unique")
        for config_path in config_paths:
            config = load_config(config_path)
            planned_jobs = expand_jobs(config)
            if args.shard is not None:
                selected, total = args.shard
                planned_jobs = [
                    job
                    for index, job in enumerate(planned_jobs)
                    if index % total == selected - 1
                ]
            if args.limit is not None:
                planned_jobs = planned_jobs[: args.limit]
            if len(config_paths) > 1:
                print(f"campaign: {config_path.stem}")
            instance_count = len({job.instance_id for job in planned_jobs})
            strategy_count = len({(job.strategy, job.label) for job in planned_jobs})
            solver_count = len(
                {(job.solver, job.subsolver, job.variant) for job in planned_jobs}
            )
            mode_count = len({job.mode for job in planned_jobs})
            print(
                f"planned jobs: {len(planned_jobs)}; instances: {instance_count}; "
                f"strategies: {strategy_count}; solvers: {solver_count}; modes: {mode_count}"
            )
            for job in planned_jobs:
                print(
                    f"{job.run_id} {job.benchmark} {job.instance_id} "
                    f"{job.label} {job.solver}/{job.subsolver} {job.mode}"
                )
        if args.dry_run:
            return 0
        destination = args.resume.resolve() if args.resume else args.out.resolve()
        if len(config_paths) == 1:
            run(
                config_paths[0],
                destination,
                args.limit,
                resume=args.resume is not None,
                rerun_statuses=args.rerun_status,
                shard=args.shard,
                strict_env=args.strict_env,
                jobs=args.jobs,
            )
        else:
            run_campaigns(
                config_paths,
                destination,
                resume=args.resume is not None,
                rerun_statuses=args.rerun_status,
                strict_env=args.strict_env,
                jobs=args.jobs,
            )
        return 0
    if args.command == "report":
        run_directory = args.rundir.resolve()
        manifest, records = read_campaign(run_directory)
        planned_run_ids = {job["run_id"] for job in manifest["planned_jobs"]}
        completed_run_ids = {record.run_id for record in records}
        missing = planned_run_ids - completed_run_ids
        if missing:
            print(
                f"WARNING: {len(missing)} of {len(planned_run_ids)} planned jobs have no result; "
                "aggregate statistics (including shifted geometric means) describe an "
                "incomplete campaign",
                file=sys.stderr,
            )
        from exact_hull.analysis.verify import verify_run

        verification_rows = verify_run(
            run_directory, tolerance=args.tolerance, reverify=args.reverify
        )
        verification = {
            row["run_id"]: row["verification_status"] for row in verification_rows
        }
        reference_path = run_directory / "references.json"
        references = json.loads(reference_path.read_text()) if reference_path.exists() else None
        print(
            aggregate(
                run_directory,
                xlsx=args.xlsx,
                records=records,
                references=references,
                verification=verification,
            )
        )
        summary_path = run_directory / "summary.csv"
        summary(
            records,
            references=references,
            verification=verification,
        ).reset_index().to_csv(summary_path, index=False)
        print(summary_path)
        methods_path = run_directory / "methods.csv"
        methods(
            records,
            planned_jobs=manifest["planned_jobs"],
            references=references,
            verification=verification,
        ).to_csv(methods_path, index=False)
        print(methods_path)
        bounds_path = run_directory / "bounds.csv"
        bounds_frame = bounds(records, references=references, verification=verification)
        bounds_frame.to_csv(bounds_path, index=False)
        print(bounds_path)
        censoring_path = run_directory / "censoring.csv"
        censoring(records, references, verification).to_csv(censoring_path, index=False)
        print(censoring_path)
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

        for path in plot_run(
            args.rundir.resolve(), mode=args.mode, tolerance=args.tolerance
        ):
            print(path)
        return 0
    if args.command == "inspect":
        from exact_hull.experiment.inspect import inspect_config

        _, used_backend = inspect_config(
            args.config.resolve(), args.out.resolve(), args.all_instances
        )
        print(args.out.resolve() / "inspection.json")
        print(args.out.resolve() / "inspection.csv")
        if not used_backend:
            print(
                "Solver-side presolve classification skipped: neither gurobipy nor "
                "pyscipopt is importable.",
                file=sys.stderr,
            )
        return 0
    if args.command == "verify":
        from exact_hull.analysis.verify import verify_run

        verify_run(args.rundir.resolve(), tolerance=args.tolerance, reverify=True)
        print(args.rundir.resolve() / "verification.csv")
        return 0
    if args.command == "reference":
        from exact_hull.analysis.references import derive_references

        run_directory = args.rundir.resolve()
        reference_path = run_directory / "references.json"
        try:
            previous = json.loads(reference_path.read_text()) if reference_path.exists() else {}
        except (OSError, json.JSONDecodeError, TypeError):
            previous = {}
        updated = derive_references(
            run_directory,
            cap=args.cap,
            reverify=args.reverify,
            tolerance=args.tolerance,
        )
        previous_entries = previous.get("references", {})
        updated_entries = updated.get("references", {})
        previous_agreements = {
            instance_id: entry
            for instance_id, entry in previous_entries.items()
            if entry.get("status") == "certified"
            and entry.get("provenance", {}).get("route") == "agreement"
        }
        demoted = sorted(
            instance_id
            for instance_id in previous_agreements
            if updated_entries.get(instance_id, {}).get("status") != "certified"
        )
        improved = sorted(
            instance_id
            for instance_id, entry in previous_agreements.items()
            if updated_entries.get(instance_id, {}).get("status") == "certified"
            and isinstance(entry.get("objective"), int | float)
            and isinstance(updated_entries[instance_id].get("objective"), int | float)
            and updated_entries[instance_id]["objective"] < entry["objective"]
        )
        if demoted:
            print(
                "WARNING: Agreement references demoted to reference_unknown for instances: "
                + ", ".join(demoted),
                file=sys.stderr,
            )
        if improved:
            print(
                "WARNING: Previously agreement-certified references improved for instances: "
                + ", ".join(improved),
                file=sys.stderr,
            )
        print(reference_path)
        return 0
    if args.command == "conic-bound":
        from exact_hull.experiment.conic_oracle import conic_bounds

        try:
            conic_bounds(args.config.resolve(), args.out.resolve())
        except (RuntimeError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(args.out.resolve() / "conic-bounds.csv")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
