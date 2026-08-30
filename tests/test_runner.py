from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pyomo.environ import ConcreteModel, Objective, Var
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition

from exact_hull.cli import main
from exact_hull.experiment import runner
from exact_hull.experiment.results import (
    RunRecord,
    is_valid_result,
    read_campaign,
    write_record_atomic,
)
from exact_hull.experiment.runner import Job


def _record(run_id="run"):
    return RunRecord(
        run_id=run_id,
        benchmark="kmeans",
        instance_id="instance",
        instance_params={},
        seed=1,
        strategy="exact",
        transformation="gdp.hull_exact",
        transformation_options={},
        solver="gams",
        subsolver="scip",
        variant=None,
        time_limit=10,
        duration_sec=1.5,
        solver_time_sec=1.0,
        status="optimal",
        objective=2.0,
    )


def _job(run_id="run"):
    return Job(
        run_id=run_id,
        benchmark="kmeans",
        instance_id="instance",
        params={"n_dimensions": 2, "n_clusters": 2, "n_points": 3},
        seed=1,
        strategy="gdp.hull_exact",
        label="exact",
        transformation_options={},
        solver="gams",
        subsolver="scip",
        variant=None,
        time_limit=10,
    )


def _record_for_job(job, **changes):
    record = _record(job.run_id)
    record.benchmark = job.benchmark
    record.instance_id = job.instance_id
    record.instance_params = job.params
    record.seed = job.seed
    record.strategy = job.label
    record.transformation = job.strategy
    record.transformation_options = job.transformation_options
    record.solver = job.solver
    record.subsolver = job.subsolver
    record.variant = job.variant
    record.time_limit = job.time_limit
    for name, value in changes.items():
        setattr(record, name, value)
    return record


def test_atomic_record_round_trip(tmp_path):
    path = tmp_path / "jobs" / "run" / "result.json"
    write_record_atomic(_record(), path)
    assert is_valid_result(path, "run")
    assert not path.with_name(".result.json.tmp").exists()


def test_resume_skips_existing_result(tmp_path, monkeypatch):
    config = Path(__file__).parents[1] / "configs" / "smoke.toml"
    normalized = runner.load_config(config)
    job = runner.expand_jobs(normalized)[0]
    runner._prepare_manifest(tmp_path, runner.build_manifest(normalized, [job]), resume=False)
    destination = tmp_path / "jobs" / job.run_id / "result.json"
    write_record_atomic(_record_for_job(job), destination)

    def unexpected(*args, **kwargs):
        raise AssertionError("completed job was run again")

    monkeypatch.setattr(runner, "run_job", unexpected)
    assert runner.run(config, tmp_path, resume=True) == []


def test_build_failure_returns_one_diagnostic_record(tmp_path, monkeypatch):
    class BrokenBenchmark:
        @staticmethod
        def build(case):
            raise RuntimeError("forced build failure")

    monkeypatch.setitem(runner.BENCHMARKS, "kmeans", BrokenBenchmark())
    record = runner.run_job(_job(), tmp_path, versions={})
    assert record.status == "build_error"
    assert "forced build failure" in record.error


def test_solver_failure_returns_one_diagnostic_record(tmp_path, monkeypatch):
    class TinyBenchmark:
        @staticmethod
        def build(case):
            model = ConcreteModel()
            model.x = Var()
            model.objective = Objective(expr=model.x)
            return model

        @staticmethod
        def solution(model):
            return {}

    class BrokenSolver:
        @staticmethod
        def solve(*args, **kwargs):
            raise RuntimeError("forced solver failure")

    monkeypatch.setitem(runner.BENCHMARKS, "kmeans", TinyBenchmark())
    monkeypatch.setattr(runner, "SolverFactory", lambda name: BrokenSolver())
    record = runner.run_job(_job(), tmp_path, versions={})
    assert record.status == "solver_error"
    assert "forced solver failure" in record.error


def test_solver_status_failure_retains_solver_message(tmp_path, monkeypatch):
    class TinyBenchmark:
        @staticmethod
        def build(case):
            model = ConcreteModel()
            model.x = Var(initialize=0)
            model.objective = Objective(expr=model.x)
            return model

        @staticmethod
        def solution(model):
            return {}

    result = SimpleNamespace(
        solver=SimpleNamespace(
            status=SolverStatus.aborted,
            termination_condition=TerminationCondition.licensingProblems,
            message="license checkout failed",
            user_time=None,
        ),
        problem=SimpleNamespace(lower_bound=float("-inf"), upper_bound=float("inf")),
    )

    class FailedSolver:
        @staticmethod
        def solve(*args, **kwargs):
            return result

    monkeypatch.setitem(runner.BENCHMARKS, "kmeans", TinyBenchmark())
    monkeypatch.setattr(runner, "SolverFactory", lambda name: FailedSolver())
    record = runner.run_job(_job(), tmp_path, versions={})
    assert record.status == "solver_error"
    assert record.error == "license checkout failed"


def test_real_gams_status_combinations_are_classified_safely():
    result = SimpleNamespace(
        solver=SimpleNamespace(
            status=SolverStatus.ok,
            termination_condition=TerminationCondition.optimal,
        )
    )
    assert runner._status(result, "Time limit reached", "gurobi", 10, 10, 10, 0) == "optimal"
    assert runner._status(result, "Time limit reached", "gurobi", 10, 10, 10, 1e-4) == "timeout"
    assert runner._status(result, "", "gurobi", 10, 1, 2, None) == "optimal"
    assert runner._status(result, "", "gurobi", 10, 10, 11, None) == "timeout"
    assert runner._status(result, "", "gurobi", 10, 10, 11, 0) == "optimal"
    result.solver.status = SolverStatus.aborted
    result.solver.termination_condition = TerminationCondition.licensingProblems
    result.solver.message = "license checkout failed"
    assert (
        runner._status(result, "Time limit reached", "gurobi", 10, 10, 10, 1e-4) == "solver_error"
    )
    assert runner._solver_diagnostic(result) == "license checkout failed"
    result.solver.status = SolverStatus.warning
    result.solver.termination_condition = TerminationCondition.optimal
    assert runner._status(result, "", "gurobi", 10, 1, 2, 0) == "solver_error"


def test_solver_diagnostic_ignores_undefined_pyomo_messages():
    diagnostic = runner._solver_diagnostic(SolverResults())
    assert diagnostic == "solver status=ok; termination=unknown"
    assert "<undefined>" not in diagnostic


def test_nonfinite_pyomo_bounds_are_missing():
    assert runner._as_float(float("inf")) is None
    assert runner._as_float(float("-inf")) is None
    assert runner._as_float(float("nan")) is None


def test_git_sha_resolution_does_not_use_ambient_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sha = runner._repository_git_sha()
    assert sha is None or len(sha) == 40


def test_resume_after_computational_config_change_is_rejected(tmp_path):
    root = Path(__file__).parents[1]
    original = root / "configs" / "smoke.toml"
    runner.run(original, tmp_path, limit=0)
    changed = tmp_path / "changed.toml"
    changed.write_text(original.read_text().replace("time_limit = 10", "time_limit = 11"))
    try:
        runner.run(changed, tmp_path, limit=0, resume=True)
    except ValueError as error:
        assert "config.experiment.time_limit differs" in str(error)
    else:
        raise AssertionError("changed campaign resumed against a stale manifest")


def test_campaign_persists_one_record_per_planned_job_with_failure(tmp_path, monkeypatch):
    config = tmp_path / "campaign.toml"
    config.write_text(
        """
[experiment]
benchmark = "kmeans"
time_limit = 10
[instances]
n_dimensions = 2
n_clusters = 2
n_points = [3, 4, 5]
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    )
    calls = 0

    class StubSolver:
        def solve(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("forced middle-job failure")
            return SimpleNamespace(
                solver=SimpleNamespace(
                    status=SolverStatus.ok,
                    termination_condition=TerminationCondition.optimal,
                    user_time=0.1,
                ),
                problem=SimpleNamespace(lower_bound=0.0, upper_bound=0.0),
            )

    monkeypatch.setattr(runner, "SolverFactory", lambda name: StubSolver())
    records = runner.run(config, tmp_path / "run")
    _, persisted = read_campaign(tmp_path / "run")
    assert len(records) == len(persisted) == 3
    assert [record.status for record in persisted].count("solver_error") == 1


@pytest.mark.parametrize(
    "bad_payload",
    ["{not json", json.dumps({"run_id": "placeholder"}), None],
)
def test_resume_reruns_invalid_or_mismatched_result(tmp_path, monkeypatch, bad_payload):
    config = Path(__file__).parents[1] / "configs" / "smoke.toml"
    normalized = runner.load_config(config)
    job = runner.expand_jobs(normalized)[0]
    runner._prepare_manifest(tmp_path, runner.build_manifest(normalized, [job]), resume=False)
    destination = tmp_path / "jobs" / job.run_id / "result.json"
    destination.parent.mkdir(parents=True)
    if bad_payload is None:
        write_record_atomic(_record("different-run-id"), destination)
    else:
        destination.write_text(bad_payload.replace("placeholder", job.run_id))
    calls = []

    def replacement(planned_job, output_directory, versions):
        calls.append(planned_job.run_id)
        return _record(planned_job.run_id)

    monkeypatch.setattr(runner, "run_job", replacement)
    runner.run(config, tmp_path, resume=True)
    assert calls == [job.run_id]
    assert is_valid_result(destination, job.run_id)


def test_limited_run_can_resume_full_plan(tmp_path, monkeypatch):
    config = Path(__file__).parents[1] / "configs" / "smoke.toml"
    runner.run(config, tmp_path, limit=0)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["planned_jobs"]) == 1
    assert manifest["execution"] == {"initial_limit": 0}

    monkeypatch.setattr(
        runner,
        "run_job",
        lambda job, output_directory, versions: _record(job.run_id),
    )
    records = runner.run(config, tmp_path, resume=True)
    assert len(records) == 1


def test_stale_manifest_temporary_does_not_block_initialization(tmp_path):
    (tmp_path / ".manifest.json.tmp").write_text("partial")
    config = runner.load_config(Path(__file__).parents[1] / "configs" / "smoke.toml")
    jobs = runner.expand_jobs(config)
    runner._prepare_manifest(tmp_path, runner.build_manifest(config, jobs), resume=False)
    assert (tmp_path / "manifest.json").exists()
    assert not (tmp_path / ".manifest.json.tmp").exists()


def test_invalid_output_directory_does_not_delete_manifest_temporary(tmp_path):
    temporary = tmp_path / ".manifest.json.tmp"
    temporary.write_text("partial")
    (tmp_path / "foreign.txt").write_text("foreign")
    config = runner.load_config(Path(__file__).parents[1] / "configs" / "smoke.toml")
    jobs = runner.expand_jobs(config)
    with pytest.raises(ValueError, match="Output directory is not empty"):
        runner._prepare_manifest(tmp_path, runner.build_manifest(config, jobs), resume=False)
    assert temporary.read_text() == "partial"


def test_report_writes_detail_and_summary_outputs(tmp_path):
    config = runner.load_config(Path(__file__).parents[1] / "configs" / "smoke.toml")
    job = runner.expand_jobs(config)[0]
    runner._prepare_manifest(tmp_path, runner.build_manifest(config, [job]), resume=False)
    write_record_atomic(
        _record_for_job(job),
        tmp_path / "jobs" / job.run_id / "result.json",
    )
    assert main(["report", str(tmp_path)]) == 0
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "summary.csv").exists()


def test_bad_typed_record_is_skipped_and_rerun_on_resume(tmp_path, monkeypatch):
    config_path = Path(__file__).parents[1] / "configs" / "smoke.toml"
    config = runner.load_config(config_path)
    job = runner.expand_jobs(config)[0]
    runner._prepare_manifest(tmp_path, runner.build_manifest(config, [job]), resume=False)
    destination = tmp_path / "jobs" / job.run_id / "result.json"
    payload = asdict(_record_for_job(job))
    payload["objective"] = "not-a-number"
    runner.write_json_atomic(payload, destination)

    with pytest.warns(UserWarning, match="Skipped 1 invalid"):
        assert read_campaign(tmp_path)[1] == []
    calls = []

    def replacement(planned_job, output_directory, versions):
        calls.append(planned_job.run_id)
        return _record_for_job(planned_job)

    monkeypatch.setattr(runner, "run_job", replacement)
    runner.run(config_path, tmp_path, resume=True)
    assert calls == [job.run_id]
    assert is_valid_result(destination, job.run_id, asdict(job))


def test_record_identity_mismatch_is_skipped(tmp_path):
    config = runner.load_config(Path(__file__).parents[1] / "configs" / "smoke.toml")
    job = runner.expand_jobs(config)[0]
    runner._prepare_manifest(tmp_path, runner.build_manifest(config, [job]), resume=False)
    destination = tmp_path / "jobs" / job.run_id / "result.json"
    write_record_atomic(_record_for_job(job, transformation="gdp.bigm"), destination)
    with pytest.warns(UserWarning, match="does not match planned value"):
        assert read_campaign(tmp_path)[1] == []


def test_malformed_manifest_has_clear_error(tmp_path):
    runner.write_json_atomic({"config": {}, "instances": []}, tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="Invalid campaign manifest.*planned_jobs"):
        read_campaign(tmp_path)


def test_report_with_no_valid_records_writes_normal_headers(tmp_path):
    config = runner.load_config(Path(__file__).parents[1] / "configs" / "smoke.toml")
    jobs = runner.expand_jobs(config)
    runner._prepare_manifest(tmp_path, runner.build_manifest(config, jobs), resume=False)
    assert main(["report", str(tmp_path)]) == 0
    results = pd.read_csv(tmp_path / "results.csv")
    table = pd.read_csv(tmp_path / "summary.csv")
    assert {"run_id", "objective", "ground_truth", "correct"} <= set(results.columns)
    assert {"strategy", "status", "jobs", "median_solver_time_sec"} <= set(table.columns)


def test_negative_limit_is_rejected_by_argparse(tmp_path):
    config = Path(__file__).parents[1] / "configs" / "smoke.toml"
    with pytest.raises(SystemExit) as error:
        main(["run", str(config), "--dry-run", "--limit", "-1"])
    assert error.value.code == 2
