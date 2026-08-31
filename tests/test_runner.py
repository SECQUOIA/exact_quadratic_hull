from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pyomo.environ import (
    Binary,
    BooleanVar,
    ConcreteModel,
    Constraint,
    LogicalConstraint,
    Objective,
    Var,
    maximize,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition

from exact_hull.analysis.verify import verify_record, verify_run
from exact_hull.cli import main
from exact_hull.experiment import runner
from exact_hull.experiment.results import (
    RunRecord,
    is_valid_result,
    read_campaign,
    write_json_atomic,
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
        mode="solve",
        time_limit=10,
        duration_sec=1.5,
        solver_time_sec=1.0,
        status="optimal",
        objective=2.0,
    )


def _job(run_id="run", time_limit=10, mode="solve"):
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
        mode=mode,
        time_limit=time_limit,
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
    record.mode = job.mode
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


def test_run_job_synchronizes_and_serializes_boolean_values(tmp_path, monkeypatch):
    class BooleanBenchmark:
        @staticmethod
        def build(case):
            model = ConcreteModel()
            model.x = Var(bounds=(0, 1), initialize=0)
            model.b = BooleanVar()
            model.logical = LogicalConstraint(expr=model.b)
            model.objective = Objective(expr=model.x)
            return model

        @staticmethod
        def solution(model):
            return {}

    result = SimpleNamespace(
        solver=SimpleNamespace(
            status=SolverStatus.ok,
            termination_condition=TerminationCondition.optimal,
            user_time=1.0,
        ),
        problem=SimpleNamespace(
            lower_bound=0.0,
            upper_bound=0.0,
            number_of_variables=2,
            number_of_constraints=1,
            number_of_nonzeros=1,
            number_of_integer_variables=1,
        ),
        solution=None,
    )

    class Solver:
        @staticmethod
        def solve(model, **kwargs):
            model.b.get_associated_binary().set_value(1)
            return result

    monkeypatch.setitem(runner.BENCHMARKS, "kmeans", BooleanBenchmark())
    monkeypatch.setattr(runner, "SolverFactory", lambda name: Solver())
    record = runner.run_job(_job(), tmp_path, versions={})
    assert record.solution["booleans"] == {"b": True}
    assert verify_record(record)["verification_status"] == "verified_feasible"


def test_fractional_boolean_sync_does_not_discard_numeric_payload(tmp_path, monkeypatch):
    class FractionalBenchmark:
        @staticmethod
        def build(case):
            model = ConcreteModel()
            model.x = Var(bounds=(0, 1), initialize=0)
            model.d = Disjunct([1, 2])
            model.d[1].constraint = Constraint(expr=model.x >= 0)
            model.d[2].constraint = Constraint(expr=model.x <= 1)
            model.choice = Disjunction(expr=list(model.d.values()))
            model.objective = Objective(expr=model.x)
            return model

        @staticmethod
        def solution(model):
            return {}

    result = SimpleNamespace(
        solver=SimpleNamespace(
            status=SolverStatus.ok,
            termination_condition=TerminationCondition.optimal,
            user_time=1.0,
        ),
        problem=SimpleNamespace(
            lower_bound=0.0,
            upper_bound=0.0,
            number_of_variables=4,
            number_of_constraints=3,
            number_of_nonzeros=3,
            number_of_integer_variables=2,
        ),
        solution=None,
    )

    class Solver:
        @staticmethod
        def solve(model, **kwargs):
            for disjunct in model.d.values():
                disjunct.binary_indicator_var.set_value(0.5, skip_validation=True)
            return result

    monkeypatch.setitem(runner.BENCHMARKS, "kmeans", FractionalBenchmark())
    monkeypatch.setattr(runner, "SolverFactory", lambda name: Solver())
    record = runner.run_job(_job(), tmp_path, versions={})
    assert record.solution["variables"] == {"x": 0.0}
    assert set(record.solution["indicators"].values()) == {0.5}
    assert verify_record(record)["verification_status"] == "fractional_indicators"


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
    assert runner._status(result, "", "gurobi", 10, 1, 2, 5e-7, "root") == "optimal"
    result.solver.status = SolverStatus.unknown
    assert runner._status(result, "", "gurobi", 10, 1, 2, None) == "solver_error"
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
    assert (
        runner._status(result, "Node limit reached", "gurobi", 10, 1, 2, 0.1, "root", -1.0)
        == "node_limit"
    )
    assert (
        runner._status(result, "Node limit reached", "gurobi", 10, 1, 2, None, "root", None)
        == "node_limit"
    )
    assert runner._status(result, "", "gurobi", 10, 1, 2, 0.1, "root", -1.0) == "feasible"
    assert runner._status(result, "", "gurobi", 10, 1, 2, 0, "root", -1.0) == "feasible"
    assert runner._status(result, "", "gurobi", 10, 1, 2, 5e-7, "root", -1.0) == "feasible"
    assert (
        runner._status(
            result, "Time limit reached\nNode limit reached", "gurobi", 10, 1, 2, 0.1, "root"
        )
        == "timeout"
    )
    assert (
        runner._status(result, "Time limit reached", "gurobi", 10, 1, 2, 0, "relaxation")
        == "solver_error"
    )
    result.solver.termination_condition = TerminationCondition.maxTimeLimit
    assert runner._status(result, "", "gurobi", 10, 1, 2, None, "root", -1.0) == "timeout"
    result.solver.status = SolverStatus.ok
    result.solver.termination_condition = TerminationCondition.optimal
    assert (
        runner._status(result, "node limit reached", "scip", 10, 1, 2, 0.1, "root", -1.0)
        == "node_limit"
    )
    assert runner._status(result, "", "scip", 10, 1, 2, 0.1, "root") == "feasible"
    result.solver.termination_condition = TerminationCondition.globallyOptimal
    assert runner._status(result, "", "scip", 10, 1, 2, None, "root") == "feasible"
    result.solver.termination_condition = TerminationCondition.maxIterations
    assert (
        runner._status(result, "node limit reached", "scip", 10, 1, 2, None, "root", -1.0)
        == "node_limit"
    )


def test_solver_diagnostic_ignores_undefined_pyomo_messages():
    diagnostic = runner._solver_diagnostic(SolverResults())
    assert diagnostic == "solver status=ok; termination=unknown"
    assert "<undefined>" not in diagnostic


def test_nonfinite_pyomo_bounds_are_missing():
    assert runner._as_float(float("inf")) is None
    assert runner._as_float(float("-inf")) is None
    assert runner._as_float(float("nan")) is None
    assert runner._as_bound(1e20) is None
    assert runner._as_bound(-1e19) is None


def test_gams_banner_version_parser_is_stable():
    banner = "--- Job -version Start 08/30/26 22:30:58 54.3.1 x86 64bit/Linux"
    assert runner._parse_gams_version(banner) == "54.3.1"
    assert runner._parse_gams_version("probe failed") is None


@pytest.mark.parametrize(
    ("status", "termination", "lower", "upper", "expected"),
    [
        (SolverStatus.ok, TerminationCondition.infeasible, None, None,
         TerminationCondition.infeasible),
        (SolverStatus.warning, TerminationCondition.infeasible, None, None,
         TerminationCondition.error),
        (SolverStatus.ok, TerminationCondition.feasible, 1.0, 1.0,
         TerminationCondition.optimal),
        (SolverStatus.warning, TerminationCondition.feasible, 1.0, 1.0,
         TerminationCondition.error),
    ],
)
def test_mbigm_adapter_only_trusts_clean_solver_results(
    monkeypatch, status, termination, lower, upper, expected
):
    model = ConcreteModel()
    model.x = Var(bounds=(-1, 1), initialize=0)
    model.constraint = Constraint(expr=model.x >= -1)
    model.objective = Objective(expr=model.x)
    result = SolverResults()
    result.solver.status = status
    result.solver.termination_condition = termination
    result.problem.lower_bound = lower
    result.problem.upper_bound = upper
    captured = {}

    class Solver:
        @staticmethod
        def solve(*args, **kwargs):
            captured.update(kwargs)
            return result

    monkeypatch.setattr(runner, "SolverFactory", lambda name: Solver())
    adapter = runner._GamsMbigmSolver()
    returned = adapter.solve(model)
    assert returned.solver.termination_condition == expected
    assert adapter.m_estimation_subsolves == 1
    assert any("reslim=30" in option for option in captured["add_options"])
    assert "DualReductions 0" in captured["add_options"]


def test_mbigm_cache_is_reused_and_transform_times_are_recorded(tmp_path, monkeypatch):
    def model():
        from pyomo.gdp import Disjunct, Disjunction

        instance = ConcreteModel()
        instance.x = Var(bounds=(-2, 2))
        instance.y = Var(bounds=(-2, 2))
        instance.d = Disjunct([1, 2])
        instance.d[1].constraint = Constraint(expr=instance.x + instance.y <= 1)
        instance.d[2].constraint = Constraint(expr=instance.x - instance.y <= 1)
        instance.choice = Disjunction(expr=list(instance.d.values()))
        instance.objective = Objective(expr=instance.x)
        return instance

    cache = tmp_path / "mbigm" / "instance.json"
    options = next(
        strategy["options"]
        for strategy in runner.load_config(
            Path(__file__).parents[1] / "configs" / "qualification.toml"
        )["strategies"]
        if strategy["name"] == "gdp.mbigm"
    )

    class Solver:
        @staticmethod
        def solve(*args, **kwargs):
            result = SolverResults()
            result.solver.status = SolverStatus.ok
            result.solver.termination_condition = TerminationCondition.optimal
            result.problem.lower_bound = 0.0
            result.problem.upper_bound = 0.0
            return result

    monkeypatch.setattr(runner, "SolverFactory", lambda name: Solver())
    first = model()
    first_counts, *_ = runner.transform_model(
        first, "gdp.mbigm", options, "solve", mbigm_cache_path=cache
    )
    assert cache.exists()
    assert first._exact_hull_transform_stats["m_estimation_subsolves"] > 0
    assert not first._exact_hull_transform_stats["m_estimation_cache_hit"]
    assert first._exact_hull_transform_stats["m_estimation_time_total_sec"] == pytest.approx(
        first._exact_hull_transform_stats["m_estimation_time_sec"]
    )
    cache_payload = json.loads(cache.read_text())
    assert cache_payload["m_estimation_time_total_sec"] == pytest.approx(
        first._exact_hull_transform_stats["m_estimation_time_total_sec"]
    )
    assert all(
        not row["constraint"].startswith(f"{row['disjunct']}.")
        for row in cache_payload["values"]
    )
    second = model()
    second_counts, *_ = runner.transform_model(
        second, "gdp.mbigm", options, "solve", mbigm_cache_path=cache
    )
    assert first._exact_hull_transform_stats["transform_sec"] > 0
    assert second._exact_hull_transform_stats["m_estimation_subsolves"] == 0
    assert second._exact_hull_transform_stats["m_estimation_cache_hit"]
    assert second._exact_hull_transform_stats["m_estimation_time_sec"] == 0
    assert second._exact_hull_transform_stats["m_estimation_time_total_sec"] == pytest.approx(
        first._exact_hull_transform_stats["m_estimation_time_total_sec"]
    )
    assert first_counts == second_counts
    assert sum(1 for _ in first.component_data_objects(Var)) == sum(
        1 for _ in second.component_data_objects(Var)
    )
    assert sum(1 for _ in first.component_data_objects(Constraint, active=True)) == sum(
        1 for _ in second.component_data_objects(Constraint, active=True)
    )
    assert sorted(
        str(constraint.expr)
        for constraint in first.component_data_objects(Constraint, active=True)
    ) == sorted(
        str(constraint.expr)
        for constraint in second.component_data_objects(Constraint, active=True)
    )

    payload = json.loads(cache.read_text())
    payload["schema_version"] = 2
    write_json_atomic(payload, cache)
    with pytest.raises(ValueError, match="unsupported schema version"):
        runner.transform_model(
            model(), "gdp.mbigm", options, "solve", mbigm_cache_path=cache
        )

    payload["schema_version"] = 3
    payload["values"].pop()
    write_json_atomic(payload, cache)
    truncated = model()
    runner.transform_model(
        truncated, "gdp.mbigm", options, "solve", mbigm_cache_path=cache
    )
    assert truncated._exact_hull_transform_stats["m_estimation_subsolves"] > 0

    mismatched = dict(options, use_primal_bound=not options["use_primal_bound"])
    with pytest.raises(ValueError, match="different mbigm options"):
        runner.transform_model(
            model(), "gdp.mbigm", mismatched, "solve", mbigm_cache_path=cache
        )


def test_mbigm_cache_race_accepts_a_different_same_options_m_set(tmp_path):
    path = tmp_path / "mbigm.json"
    first = {"options_fingerprint": "same", "values": [{"value": [1, 2]}]}
    second = {"options_fingerprint": "same", "values": [{"value": [3, 4]}]}
    runner._write_mbigm_cache(path, first)
    runner._write_mbigm_cache(path, second)
    assert json.loads(path.read_text()) == first


def test_mbigm_options_fingerprint_includes_estimator_identity():
    base = {"reduce_bound_constraints": True}
    gurobi = runner._mbigm_options_fingerprint({**base, "solver": "gurobi"})
    scip = runner._mbigm_options_fingerprint({**base, "solver": "scip"})
    adapter = runner._mbigm_options_fingerprint(
        {**base, "solver": runner._GamsMbigmSolver()}
    )
    assert gurobi != scip
    assert adapter == gurobi


def test_non_minimization_objective_is_a_build_error(tmp_path, monkeypatch):
    class MaximizationBenchmark:
        @staticmethod
        def build(case):
            model = ConcreteModel()
            model.x = Var(initialize=0)
            model.objective = Objective(expr=model.x, sense=maximize)
            return model

    monkeypatch.setitem(runner.BENCHMARKS, "kmeans", MaximizationBenchmark())
    record = runner.run_job(_job(), tmp_path, versions={})
    assert record.status == "build_error"
    assert "minimization objective" in record.error


@pytest.mark.parametrize("discrete_count", [0, None])
def test_relaxation_mode_removes_integrality_and_records_problem_statistics(
    tmp_path, monkeypatch, discrete_count
):
    class BinaryBenchmark:
        @staticmethod
        def build(case):
            model = ConcreteModel()
            model.x = Var(domain=Binary, initialize=0)
            model.objective = Objective(expr=model.x)
            return model

        @staticmethod
        def solution(model):
            return {"x": model.x.value}

    class InspectingSolver:
        @staticmethod
        def solve(model, *args, **kwargs):
            assert all(not variable.is_integer() for variable in model.component_data_objects(Var))
            return SimpleNamespace(
                solver=SimpleNamespace(
                    status=SolverStatus.ok,
                    termination_condition=TerminationCondition.optimal,
                    user_time=0.1,
                ),
                problem=SimpleNamespace(
                    lower_bound=0.0,
                    upper_bound=0.0,
                    number_of_variables=1,
                    number_of_constraints=0,
                    number_of_nonzeros=0,
                    number_of_integer_variables=discrete_count,
                ),
            )

    monkeypatch.setitem(runner.BENCHMARKS, "kmeans", BinaryBenchmark())
    monkeypatch.setattr(runner, "SolverFactory", lambda name: InspectingSolver())
    record = runner.run_job(_job(mode="relaxation"), tmp_path, versions={})
    assert record.status == "optimal"
    assert record.num_discrete_variables == discrete_count
    assert (record.num_variables, record.num_constraints, record.num_nonzeros) == (1, 0, 0)
    assert (record.solver_status, record.termination) == ("ok", "optimal")


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
        assert "config.experiment." in str(error) and "time_limit differs" in str(error)
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


def test_resume_reruns_only_requested_status(tmp_path, monkeypatch):
    config = Path(__file__).parents[1] / "configs" / "smoke.toml"
    normalized = runner.load_config(config)
    job = runner.expand_jobs(normalized)[0]
    runner._prepare_manifest(tmp_path, runner.build_manifest(normalized, [job]), resume=False)
    destination = tmp_path / "jobs" / job.run_id / "result.json"
    write_record_atomic(_record_for_job(job, status="solver_error"), destination)
    calls = []

    def replacement(planned_job, output_directory, versions):
        calls.append(planned_job.run_id)
        return _record_for_job(planned_job)

    monkeypatch.setattr(runner, "run_job", replacement)
    runner.run(config, tmp_path, resume=True, rerun_statuses={"solver_error"})
    assert calls == [job.run_id]


def test_shard_filters_full_plan_before_limit(tmp_path, monkeypatch):
    config = tmp_path / "shards.toml"
    config.write_text(
        """
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = 2
n_clusters = 2
n_points = [3, 4, 5, 6, 7]
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    )
    planned = runner.expand_jobs(runner.load_config(config))
    calls = []
    monkeypatch.setattr(
        runner,
        "run_job",
        lambda job, output_directory, versions: calls.append(job.run_id)
        or _record_for_job(job),
    )
    runner.run(config, tmp_path / "run", shard=(2, 3), limit=1)
    assert calls == [planned[1].run_id]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert len(manifest["planned_jobs"]) == 5


def test_strict_resume_environment_mismatch_is_an_error(tmp_path):
    config = runner.load_config(Path(__file__).parents[1] / "configs" / "smoke.toml")
    jobs = runner.expand_jobs(config)
    manifest = runner.build_manifest(config, jobs, versions={"python": "old"})
    runner._prepare_manifest(tmp_path, manifest, resume=False)
    current = runner.build_manifest(config, jobs, versions={"python": "new"})
    with pytest.raises(ValueError, match="Resume environment differs"):
        runner._prepare_manifest(tmp_path, current, resume=True, strict_env=True)


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
    assert (tmp_path / "methods.csv").exists()
    assert (tmp_path / "bounds.csv").exists()
    assert pd.read_csv(tmp_path / "methods.csv")["planned_denominator"].iloc[0] == 1


def test_verification_reuses_unchanged_rows_and_adds_or_replaces_results(
    tmp_path, monkeypatch
):
    config = runner.load_config(Path(__file__).parents[1] / "configs" / "smoke.toml")
    first_job = runner.expand_jobs(config)[0]
    second_job = replace(first_job, run_id="second")
    runner._prepare_manifest(
        tmp_path, runner.build_manifest(config, [first_job, second_job]), resume=False
    )
    first = _record_for_job(first_job)
    first.timestamp = "first-v1"
    write_record_atomic(first, tmp_path / "jobs" / first.run_id / "result.json")
    verify_run(tmp_path)

    second = _record_for_job(second_job)
    second.timestamp = "second-v1"
    write_record_atomic(second, tmp_path / "jobs" / second.run_id / "result.json")
    calls = []
    original = verify_record

    def counted(record, tolerance=1e-5, model=None):
        calls.append(record.run_id)
        return original(record, tolerance, model)

    monkeypatch.setattr("exact_hull.analysis.verify.verify_record", counted)
    rows = verify_run(tmp_path)
    assert calls == ["second"]
    assert {row["run_id"] for row in rows} == {first.run_id, second.run_id}

    calls.clear()
    first.timestamp = "first-v2"
    write_record_atomic(first, tmp_path / "jobs" / first.run_id / "result.json")
    verify_run(tmp_path)
    assert calls == [first.run_id]


def test_plot_incrementally_reverifies_rerun_records(tmp_path, monkeypatch):
    from exact_hull.analysis.plots import plot_run

    config = runner.load_config(Path(__file__).parents[1] / "configs" / "smoke.toml")
    job = runner.expand_jobs(config)[0]
    runner._prepare_manifest(tmp_path, runner.build_manifest(config, [job]), resume=False)
    record = _record_for_job(job)
    record.timestamp = "first-v1"
    write_record_atomic(record, tmp_path / "jobs" / record.run_id / "result.json")
    verify_run(tmp_path)

    record.timestamp = "first-v2"
    write_record_atomic(record, tmp_path / "jobs" / record.run_id / "result.json")
    calls = []
    original = verify_record

    def counted(current, tolerance=1e-5, model=None):
        calls.append((current.run_id, tolerance))
        return original(current, tolerance, model)

    monkeypatch.setattr("exact_hull.analysis.verify.verify_record", counted)
    assert plot_run(tmp_path, tolerance=2e-5)
    assert calls == [(record.run_id, 2e-5)]
    verification = pd.read_csv(tmp_path / "verification.csv")
    assert verification["record_timestamp"].tolist() == ["first-v2"]
    assert verification["verification_tolerance"].tolist() == [2e-5]


def test_report_and_reference_forward_verification_tolerance(tmp_path, monkeypatch):
    config = runner.load_config(Path(__file__).parents[1] / "configs" / "smoke.toml")
    job = runner.expand_jobs(config)[0]
    runner._prepare_manifest(tmp_path, runner.build_manifest(config, [job]), resume=False)
    write_record_atomic(
        _record_for_job(job), tmp_path / "jobs" / job.run_id / "result.json"
    )
    captured = {}
    original = verify_run

    def verify_with_tolerance(path, tolerance=1e-5, reverify=False):
        captured["report"] = tolerance
        return original(path, tolerance=tolerance, reverify=reverify)

    monkeypatch.setattr("exact_hull.analysis.verify.verify_run", verify_with_tolerance)
    assert main(["report", str(tmp_path), "--tolerance", "2e-5"]) == 0
    assert captured["report"] == 2e-5

    def reference_with_tolerance(path, cap=4096, reverify=False, tolerance=1e-5):
        captured["reference"] = tolerance
        return {}

    monkeypatch.setattr(
        "exact_hull.analysis.references.derive_references", reference_with_tolerance
    )
    assert main(["reference", str(tmp_path), "--tolerance", "3e-5"]) == 0
    assert captured["reference"] == 3e-5

    def plot_with_tolerance(path, mode="solve", tolerance=1e-5):
        captured["plot"] = tolerance
        return []

    monkeypatch.setattr("exact_hull.analysis.plots.plot_run", plot_with_tolerance)
    assert main(["plot", str(tmp_path), "--tolerance", "4e-5"]) == 0
    assert captured["plot"] == 4e-5


def test_bad_typed_record_is_skipped_and_rerun_on_resume(tmp_path, monkeypatch):
    config_path = Path(__file__).parents[1] / "configs" / "smoke.toml"
    config = runner.load_config(config_path)
    job = runner.expand_jobs(config)[0]
    runner._prepare_manifest(tmp_path, runner.build_manifest(config, [job]), resume=False)
    destination = tmp_path / "jobs" / job.run_id / "result.json"
    payload = asdict(_record_for_job(job))
    payload["objective"] = "not-a-number"
    write_json_atomic(payload, destination)

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
    write_json_atomic({"config": {}, "instances": []}, tmp_path / "manifest.json")
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
    assert (tmp_path / "bounds.csv").exists()


def test_negative_limit_is_rejected_by_argparse(tmp_path):
    config = Path(__file__).parents[1] / "configs" / "smoke.toml"
    with pytest.raises(SystemExit) as error:
        main(["run", str(config), "--dry-run", "--limit", "-1"])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["verify", "report", "reference", "plot"])
@pytest.mark.parametrize("tolerance", ["0", "-1", "nan", "inf"])
def test_verification_cli_rejects_nonpositive_or_nonfinite_tolerance(
    command, tolerance
):
    with pytest.raises(SystemExit) as error:
        main([command, "unused", "--tolerance", tolerance])
    assert error.value.code == 2


@pytest.mark.parametrize("option", ["--rerun-status", "--strict-env"])
def test_resume_only_options_use_argparse_error(option):
    config = Path(__file__).parents[1] / "configs" / "smoke.toml"
    arguments = ["run", str(config), "--dry-run", option]
    if option == "--rerun-status":
        arguments.append("solver_error")
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_timeout_without_incumbent_records_no_objective(tmp_path, monkeypatch):
    """GAMS reports NA levels for no-incumbent timeouts; Pyomo cannot load them."""

    class TinyBenchmark:
        @staticmethod
        def build(case):
            model = ConcreteModel()
            model.x = Var(initialize=0)
            model.objective = Objective(expr=model.x)
            return model

        @staticmethod
        def solution(model):
            return {"x": 0.0}

    result = SimpleNamespace(
        solver=SimpleNamespace(
            status=SolverStatus.ok,
            termination_condition=TerminationCondition.optimal,  # MODELSTAT overwrite
            message=None,
            user_time=6.0,
        ),
        problem=SimpleNamespace(lower_bound=-1.0, upper_bound=-0.08882106844809479),
        solution=[SimpleNamespace()],  # a solution container Pyomo cannot load
    )

    class NoIncumbentSolver:
        @staticmethod
        def solve(*args, **kwargs):
            return result

    monkeypatch.setitem(runner.BENCHMARKS, "kmeans", TinyBenchmark())
    monkeypatch.setattr(runner, "SolverFactory", lambda name: NoIncumbentSolver())
    record = runner.run_job(_job(time_limit=5), tmp_path, versions={})
    assert record.status == "timeout"
    assert record.objective is None
    assert record.solution == {}
    assert record.lower_bound == -1.0
    assert record.upper_bound is None
    assert not (tmp_path / "jobs" / record.run_id / "scratch").exists()
