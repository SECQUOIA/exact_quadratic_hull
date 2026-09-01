import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pyomo.environ import (
    Binary,
    BooleanVar,
    ConcreteModel,
    Constraint,
    Objective,
    Var,
    cos,
    sin,
    value,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverStatus, TerminationCondition

from exact_hull.analysis.references import (
    _solve_fixed_selection,
    derive_references,
    enumerate_reference,
)
from exact_hull.analysis.verify import verify_record
from exact_hull.benchmarks import BENCHMARKS
from exact_hull.benchmarks.base import BenchmarkCase
from exact_hull.experiment.conic_oracle import (
    _solve_gurobi,
    build_oracle_model,
    conic_bounds,
)
from exact_hull.experiment.inspect import _solver_presolve, inspect_config
from exact_hull.experiment.results import RunRecord
from exact_hull.experiment.runner import expand_jobs, load_config, transform_model
from exact_hull.experiment.structure import structural_counts

ROOT = Path(__file__).parents[1]


class TinyBenchmark:
    @staticmethod
    def build(case):
        model = ConcreteModel()
        model.x = Var(bounds=(0, 2))
        model.d = Disjunct([1, 2])
        model.d[1].constraint = Constraint(expr=model.x >= 0)
        model.d[2].constraint = Constraint(expr=model.x >= 1)
        model.choice = Disjunction(expr=[model.d[1], model.d[2]])
        model.objective = Objective(expr=model.x)
        return model


def _tiny_record(objective=0):
    prototype = TinyBenchmark.build(None)
    solution = {
        "variables": {
            prototype.x.name: 0.0,
        },
        "indicators": {prototype.d[1].name: 1.0, prototype.d[2].name: 0.0},
    }
    return RunRecord(
        run_id="tiny",
        benchmark="tiny",
        instance_id="tiny-1",
        instance_params={},
        seed=1,
        strategy="s",
        transformation="gdp.bigm",
        transformation_options={},
        solver="gams",
        subsolver="scip",
        variant=None,
        mode="solve",
        time_limit=10,
        duration_sec=1,
        solver_time_sec=1,
        status="optimal",
        objective=objective,
        solution=solution,
    )


def test_verify_classifies_feasibility_and_objective(monkeypatch):
    monkeypatch.setitem(BENCHMARKS, "tiny", TinyBenchmark())
    verified = verify_record(_tiny_record())
    assert verified["verification_status"] == "verified_feasible"
    assert verified["max_residual"] == 0
    mismatch = verify_record(replace(_tiny_record(), objective=1))
    assert mismatch["verification_status"] == "objective_mismatch"


@pytest.mark.parametrize("tolerance", [0, -1, float("inf"), float("nan")])
def test_verify_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="finite and positive"):
        verify_record(_tiny_record(), tolerance=tolerance)


def test_verify_rejects_missing_fractional_and_infeasible_payloads(monkeypatch):
    monkeypatch.setitem(BENCHMARKS, "tiny", TinyBenchmark())
    missing = replace(_tiny_record(), solution={"variables": {}, "indicators": {
        "d[1]": 1.0, "d[2]": 0.0,
    }})
    assert verify_record(missing)["verification_status"] == "not_verifiable"
    fractional = replace(
        _tiny_record(),
        solution={"variables": {"x": 0.0}, "indicators": {"d[1]": 0.5, "d[2]": 0.5}},
    )
    assert verify_record(fractional)["verification_status"] == "fractional_indicators"
    duplicate = replace(
        _tiny_record(),
        solution={"variables": {"x": 0.0}, "indicators": {"d[1]": 1.0, "d[2]": 1.0}},
    )
    assert verify_record(duplicate)["verification_status"] == "infeasible_point"
    infeasible = replace(
        _tiny_record(),
        solution={"variables": {"x": 0.0}, "indicators": {"d[1]": 0.0, "d[2]": 1.0}},
    )
    assert verify_record(infeasible)["verification_status"] == "infeasible_point"


class BooleanBenchmark:
    @staticmethod
    def build(case):
        model = ConcreteModel()
        model.x = Var(bounds=(0, 1))
        model.y = Var(domain=Binary)
        model.b = BooleanVar()
        model.b.associate_binary_var(model.y)
        model.objective = Objective(expr=model.x)
        return model


class IndicatorAliasBenchmark:
    @staticmethod
    def build(case):
        model = ConcreteModel()
        model.x = Var(bounds=(0, 1))
        model.d = Disjunct([1, 2])
        model.d[1].constraint = Constraint(expr=model.x >= 0)
        model.d[2].constraint = Constraint(expr=model.x >= 1)
        model.choice = Disjunction(expr=[model.d[1], model.d[2]])
        model.alias = BooleanVar()
        model.alias.associate_binary_var(model.d[1].binary_indicator_var)
        model.objective = Objective(expr=model.x)
        return model


def test_verify_derives_missing_boolean_from_stored_associated_binary(monkeypatch):
    monkeypatch.setitem(BENCHMARKS, "boolean", BooleanBenchmark())
    record = replace(
        _tiny_record(),
        benchmark="boolean",
        solution={"variables": {"x": 0.0, "y": 1.0}, "indicators": {}, "booleans": {"b": None}},
    )
    assert verify_record(record)["verification_status"] == "verified_feasible"


def test_verify_rejects_conflicting_binary_representations(monkeypatch):
    monkeypatch.setitem(BENCHMARKS, "tiny", TinyBenchmark())
    prototype = TinyBenchmark.build(None)
    conflicting_indicator = replace(
        _tiny_record(),
        solution={
            "variables": {
                "x": 0.0,
                prototype.d[1].binary_indicator_var.name: 0.0,
            },
            "indicators": {"d[1]": 1.0, "d[2]": 0.0},
            "booleans": {prototype.d[1].indicator_var.name: 1.0},
        },
    )
    assert verify_record(conflicting_indicator)["verification_status"] == "not_verifiable"

    monkeypatch.setitem(BENCHMARKS, "boolean", BooleanBenchmark())
    conflicting_boolean = replace(
        _tiny_record(),
        benchmark="boolean",
        solution={
            "variables": {"x": 0.0, "y": 1.0},
            "indicators": {},
            "booleans": {"b": 0.0},
        },
    )
    assert verify_record(conflicting_boolean)["verification_status"] == "not_verifiable"


@pytest.mark.parametrize(
    ("alias", "expected"),
    [(0.0, "not_verifiable"), (1.0, "verified_feasible")],
)
def test_verify_cross_checks_boolean_aliased_to_disjunct_indicator(
    monkeypatch, alias, expected
):
    monkeypatch.setitem(BENCHMARKS, "indicator-alias", IndicatorAliasBenchmark())
    record = replace(
        _tiny_record(),
        benchmark="indicator-alias",
        solution={
            "variables": {"x": 0.0},
            "indicators": {"d[1]": 1.0, "d[2]": 0.0},
            "booleans": {"alias": alias},
        },
    )
    assert verify_record(record)["verification_status"] == expected


def test_reference_enumeration_uses_all_tiny_selections(monkeypatch):
    monkeypatch.setitem(BENCHMARKS, "tiny", TinyBenchmark())
    calls = []

    def solve(model):
        selected = value(model.d[1].indicator_var)
        calls.append(selected)
        return 1.0 if selected else 2.0

    objective, provenance = enumerate_reference(
        "tiny", BenchmarkCase("tiny-1", {}, 1), cap=2, solve_selection=solve
    )
    assert objective == 1
    assert provenance == {"route": "enumeration", "selection_count": 2, "solved": 2}
    assert calls == [True, False]


def test_fixed_selection_solver_accepts_leftover_binary(monkeypatch):
    model = TinyBenchmark.build(None)
    model.x.set_value(0)
    model.leftover = Var(domain=Binary, initialize=0)
    model.d[1].indicator_var.fix(True)
    model.d[2].indicator_var.fix(False)
    result = SimpleNamespace(
        solver=SimpleNamespace(
            status=SolverStatus.ok,
            termination_condition=TerminationCondition.optimal,
        ),
        problem=SimpleNamespace(upper_bound=0.0),
        solution=None,
    )
    calls = []

    class Solver:
        @staticmethod
        def solve(solved_model, **kwargs):
            calls.append(solved_model.leftover.fixed)
            return result

    monkeypatch.setattr(
        "exact_hull.analysis.references.SolverFactory", lambda name: Solver()
    )
    assert _solve_fixed_selection(model) == 0
    assert calls == [False]


def test_fixed_selection_does_not_trust_non_ok_infeasibility(monkeypatch):
    model = TinyBenchmark.build(None)
    model.d[1].indicator_var.fix(True)
    model.d[2].indicator_var.fix(False)
    result = SimpleNamespace(
        solver=SimpleNamespace(
            status=SolverStatus.warning,
            termination_condition=TerminationCondition.infeasible,
        ),
        solution=None,
    )

    class Solver:
        @staticmethod
        def solve(*args, **kwargs):
            return result

    monkeypatch.setattr(
        "exact_hull.analysis.references.SolverFactory", lambda name: Solver()
    )
    assert _solve_fixed_selection(model) is None


def test_reference_provenance_is_route_specific(tmp_path, monkeypatch):
    manifest = {
        "config": {"experiment": {"benchmark": "tiny"}},
        "instances": [
            {"instance_id": name, "params": {}, "seed": 1}
            for name in ("enumerated", "agreement", "unknown")
        ],
    }
    records = [
        replace(
            _tiny_record(1),
            run_id="agree-scip",
            instance_id="agreement",
            subsolver="scip",
        ),
        replace(
            _tiny_record(1),
            run_id="agree-gurobi",
            instance_id="agreement",
            subsolver="gurobi",
        ),
    ]
    monkeypatch.setattr(
        "exact_hull.analysis.references.read_campaign",
        lambda path: (manifest, records),
    )
    monkeypatch.setattr(
        "exact_hull.analysis.references.verify_run",
        lambda path, tolerance=1e-5, reverify=False: [
            {"run_id": record.run_id, "verification_status": "verified_feasible"}
            for record in records
        ],
    )

    def enumerate_stub(benchmark, case, cap, solve_selection):
        if case.instance_id == "enumerated":
            return 2.0, {"route": "enumeration", "selection_count": 2, "solved": 2}
        return None, {"route": "enumeration", "selection_count": 5000, "skipped": True}

    monkeypatch.setattr(
        "exact_hull.analysis.references.enumerate_reference", enumerate_stub
    )
    monkeypatch.setattr("exact_hull.analysis.references._versions", lambda: {})
    entries = derive_references(tmp_path)["references"]
    assert entries["enumerated"]["provenance"]["time_limit_sec"] == 300
    assert entries["agreement"]["provenance"] == {
        "route": "agreement",
        "run_ids": ["agree-gurobi", "agree-scip"],
        "subsolvers": ["gurobi", "scip"],
    }
    assert "time_limit_sec" not in entries["unknown"]["provenance"]


def _derive_agreement_reference(tmp_path, monkeypatch, records):
    manifest = {
        "config": {"experiment": {"benchmark": "tiny"}},
        "instances": [{"instance_id": "agreement", "params": {}, "seed": 1}],
    }
    monkeypatch.setattr(
        "exact_hull.analysis.references.read_campaign",
        lambda path: (manifest, records),
    )
    monkeypatch.setattr(
        "exact_hull.analysis.references.verify_run",
        lambda path, tolerance=1e-5, reverify=False: [
            {"run_id": record.run_id, "verification_status": "verified_feasible"}
            for record in records
        ],
    )
    monkeypatch.setattr(
        "exact_hull.analysis.references.enumerate_reference",
        lambda *args, **kwargs: (
            None,
            {"route": "enumeration", "selection_count": 5000, "skipped": True},
        ),
    )
    monkeypatch.setattr("exact_hull.analysis.references._versions", lambda: {})
    return derive_references(tmp_path)["references"]["agreement"]


def test_reference_agreement_selects_the_best_agreeing_pair(tmp_path, monkeypatch):
    records = [
        replace(
            _tiny_record(10),
            run_id="worse-scip",
            instance_id="agreement",
            subsolver="scip",
        ),
        replace(
            _tiny_record(10),
            run_id="worse-gurobi",
            instance_id="agreement",
            subsolver="gurobi",
        ),
        replace(
            _tiny_record(5),
            run_id="better-scip",
            instance_id="agreement",
            subsolver="scip",
        ),
        replace(
            _tiny_record(5),
            run_id="better-gurobi",
            instance_id="agreement",
            subsolver="gurobi",
        ),
    ]

    entry = _derive_agreement_reference(tmp_path, monkeypatch, records)

    assert entry["status"] == "certified"
    assert entry["objective"] == 5
    assert entry["provenance"]["run_ids"] == ["better-gurobi", "better-scip"]


def test_reference_agreement_is_falsified_by_a_better_feasible_record(
    tmp_path, monkeypatch
):
    records = [
        replace(
            _tiny_record(10),
            run_id="agree-scip",
            instance_id="agreement",
            subsolver="scip",
        ),
        replace(
            _tiny_record(10),
            run_id="agree-gurobi",
            instance_id="agreement",
            subsolver="gurobi",
        ),
        replace(
            _tiny_record(5),
            run_id="better-feasible",
            instance_id="agreement",
            status="feasible",
        ),
    ]

    entry = _derive_agreement_reference(tmp_path, monkeypatch, records)

    assert entry["status"] == "reference_unknown"
    assert entry["objective"] is None
    assert entry["provenance"]["route"] == "agreement"
    assert entry["provenance"]["conflict"] == {
        "run_id": "better-feasible",
        "objective": 5,
        "agreed_objective": 10,
    }


def test_agreement_certificate_is_demoted_when_a_rerun_finds_a_conflict(
    tmp_path, monkeypatch
):
    records = [
        replace(
            _tiny_record(10),
            run_id=f"agree-{subsolver}",
            instance_id="agreement",
            subsolver=subsolver,
        )
        for subsolver in ("scip", "gurobi")
    ]
    first = _derive_agreement_reference(tmp_path, monkeypatch, records)
    assert first["status"] == "certified"

    records.append(
        replace(
            _tiny_record(3),
            run_id="better-feasible",
            instance_id="agreement",
            status="feasible",
        )
    )
    second = _derive_agreement_reference(tmp_path, monkeypatch, records)

    assert second["status"] == "reference_unknown"
    assert second["provenance"]["conflict"] == {
        "run_id": "better-feasible",
        "objective": 3,
        "agreed_objective": 10,
    }


def test_agreement_certificate_improves_when_a_better_pair_appears(
    tmp_path, monkeypatch
):
    records = [
        replace(
            _tiny_record(10),
            run_id=f"worse-{subsolver}",
            instance_id="agreement",
            subsolver=subsolver,
        )
        for subsolver in ("scip", "gurobi")
    ]
    first = _derive_agreement_reference(tmp_path, monkeypatch, records)
    assert first["objective"] == 10

    records.extend(
        replace(
            _tiny_record(3),
            run_id=f"better-{subsolver}",
            instance_id="agreement",
            subsolver=subsolver,
        )
        for subsolver in ("scip", "gurobi")
    )
    second = _derive_agreement_reference(tmp_path, monkeypatch, records)

    assert second["status"] == "certified"
    assert second["objective"] == 3
    assert second["provenance"]["run_ids"] == ["better-gurobi", "better-scip"]


def test_enumeration_certificate_is_reused_without_resolving(tmp_path, monkeypatch):
    manifest = {
        "config": {"experiment": {"benchmark": "tiny"}},
        "instances": [{"instance_id": "enumerated", "params": {}, "seed": 1}],
    }
    calls = []
    monkeypatch.setattr(
        "exact_hull.analysis.references.read_campaign",
        lambda path: (manifest, []),
    )
    monkeypatch.setattr(
        "exact_hull.analysis.references.verify_run",
        lambda *args, **kwargs: [],
    )

    def enumerate_once(*args, **kwargs):
        calls.append(True)
        return 2.0, {"route": "enumeration", "selection_count": 2, "solved": 2}

    monkeypatch.setattr(
        "exact_hull.analysis.references.enumerate_reference", enumerate_once
    )
    monkeypatch.setattr("exact_hull.analysis.references._versions", lambda: {})

    assert derive_references(tmp_path)["references"]["enumerated"]["objective"] == 2
    assert derive_references(tmp_path)["references"]["enumerated"]["objective"] == 2
    assert len(calls) == 1


def test_agreement_certificate_is_replaced_when_enumeration_succeeds_on_rerun(
    tmp_path, monkeypatch
):
    manifest = {
        "config": {"experiment": {"benchmark": "tiny"}},
        "instances": [{"instance_id": "agreement", "params": {}, "seed": 1}],
    }
    records = [
        replace(
            _tiny_record(10),
            run_id=f"agree-{subsolver}",
            instance_id="agreement",
            subsolver=subsolver,
        )
        for subsolver in ("scip", "gurobi")
    ]
    outcomes = [
        (None, {"route": "enumeration", "selection_count": 2, "solved": 1}),
        (2.0, {"route": "enumeration", "selection_count": 2, "solved": 2}),
    ]
    calls = []
    monkeypatch.setattr(
        "exact_hull.analysis.references.read_campaign",
        lambda path: (manifest, records),
    )
    monkeypatch.setattr(
        "exact_hull.analysis.references.verify_run",
        lambda *args, **kwargs: [
            {"run_id": record.run_id, "verification_status": "verified_feasible"}
            for record in records
        ],
    )

    def enumerate_twice(*args, **kwargs):
        calls.append(True)
        return outcomes.pop(0)

    monkeypatch.setattr(
        "exact_hull.analysis.references.enumerate_reference", enumerate_twice
    )
    monkeypatch.setattr("exact_hull.analysis.references._versions", lambda: {})

    first = derive_references(tmp_path)["references"]["agreement"]
    second = derive_references(tmp_path)["references"]["agreement"]

    assert first["provenance"]["route"] == "agreement"
    assert second["status"] == "certified"
    assert second["objective"] == 2
    assert second["provenance"]["route"] == "enumeration"
    assert len(calls) == 2


def test_agreement_rerun_attempts_over_cap_enumeration_without_selection_solves(
    tmp_path, monkeypatch
):
    manifest = {
        "config": {"experiment": {"benchmark": "tiny"}},
        "instances": [{"instance_id": "agreement", "params": {}, "seed": 1}],
    }
    records = [
        replace(
            _tiny_record(10),
            run_id=f"agree-{subsolver}",
            instance_id="agreement",
            subsolver=subsolver,
        )
        for subsolver in ("scip", "gurobi")
    ]
    (tmp_path / "references.json").write_text(
        '{"references":{"agreement":{"status":"certified","objective":10,'
        '"provenance":{"route":"agreement"}}}}'
    )
    enumeration_calls = []
    selection_calls = []
    original = enumerate_reference
    monkeypatch.setitem(BENCHMARKS, "tiny", TinyBenchmark())
    monkeypatch.setattr(
        "exact_hull.analysis.references.read_campaign",
        lambda path: (manifest, records),
    )
    monkeypatch.setattr(
        "exact_hull.analysis.references.verify_run",
        lambda *args, **kwargs: [
            {"run_id": record.run_id, "verification_status": "verified_feasible"}
            for record in records
        ],
    )

    def counted_enumeration(*args, **kwargs):
        enumeration_calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "exact_hull.analysis.references.enumerate_reference", counted_enumeration
    )
    monkeypatch.setattr("exact_hull.analysis.references._versions", lambda: {})

    entry = derive_references(
        tmp_path,
        cap=1,
        solve_selection=lambda model: selection_calls.append(model),
    )["references"]["agreement"]

    assert enumeration_calls == [True]
    assert selection_calls == []
    assert entry["status"] == "certified"
    assert entry["objective"] == 10
    assert entry["provenance"]["route"] == "agreement"


def test_agreement_pair_tie_break_is_independent_of_record_order(
    tmp_path, monkeypatch
):
    records = [
        replace(
            _tiny_record(5),
            run_id=run_id,
            instance_id="agreement",
            subsolver=subsolver,
        )
        for run_id, subsolver in (
            ("d-gurobi", "gurobi"),
            ("c-scip", "scip"),
            ("b-gurobi", "gurobi"),
            ("a-scip", "scip"),
        )
    ]
    first = _derive_agreement_reference(
        tmp_path / "a", monkeypatch, records
    )["provenance"]
    second = _derive_agreement_reference(
        tmp_path / "b", monkeypatch, list(reversed(records))
    )["provenance"]

    assert first == second
    assert first["run_ids"] == ["a-scip", "b-gurobi"]


def test_agreement_tolerance_is_independent_of_record_order(tmp_path, monkeypatch):
    low = replace(
        _tiny_record(100.0),
        run_id="low-scip",
        instance_id="agreement",
        subsolver="scip",
    )
    high = replace(
        _tiny_record(100.001000005),
        run_id="high-gurobi",
        instance_id="agreement",
        subsolver="gurobi",
    )

    forward = _derive_agreement_reference(
        tmp_path / "forward", monkeypatch, [low, high]
    )
    reverse = _derive_agreement_reference(
        tmp_path / "reverse", monkeypatch, [high, low]
    )

    assert forward["status"] == reverse["status"] == "reference_unknown"
    assert forward["provenance"] == reverse["provenance"]


def test_inspect_writes_transform_counts_without_solver_backends(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exact_hull.experiment.inspect.importlib.util.find_spec", lambda name: None
    )
    rows, used_backend = inspect_config(ROOT / "configs" / "smoke.toml", tmp_path)
    assert not used_backend
    assert rows[0]["n_quadratic_terms"] > 0
    assert rows[0]["n_fallback_rows"] > 0
    assert (tmp_path / "inspection.json").exists()
    assert (tmp_path / "inspection.csv").exists()


def test_gurobi_inspection_uses_and_records_campaign_parameters(tmp_path, monkeypatch):
    parameters = {}

    class Model:
        NumVars = 0
        NumConstrs = 0
        NumQConstrs = 0

        @staticmethod
        def setParam(name, value):
            parameters[name] = value

        @staticmethod
        def presolve():
            return Model()

        @staticmethod
        def getVars():
            return []

        @staticmethod
        def getQConstrs():
            return []

    gurobi = SimpleNamespace(
        read=lambda path: Model(),
        GRB=SimpleNamespace(BINARY="B"),
    )
    monkeypatch.setattr(
        "exact_hull.experiment.inspect.importlib.util.find_spec", lambda name: object()
    )
    monkeypatch.setitem(sys.modules, "gurobipy", gurobi)
    backend, counts = _solver_presolve(tmp_path / "model.lp")
    assert backend == "gurobipy"
    assert parameters == {"NonConvex": 2, "FuncNonlinear": 1, "Threads": 1}
    assert counts["solver_parameters"] == parameters


def test_inspect_contains_unexpected_export_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exact_hull.experiment.inspect.importlib.util.find_spec", lambda name: None
    )

    def fail_export(*args, **kwargs):
        raise RuntimeError("writer crashed")

    monkeypatch.setattr("pyomo.core.base.block.BlockData.write", fail_export)
    rows, _ = inspect_config(ROOT / "configs" / "smoke.toml", tmp_path)
    assert "writer crashed" in rows[0]["export_error"]
    assert rows[0]["model_path"] is None
    assert not list(tmp_path.glob("*.lp"))


def test_inspect_contains_temporary_file_allocation_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exact_hull.experiment.inspect.importlib.util.find_spec", lambda name: None
    )

    config = tmp_path / "inspect.toml"
    config.write_text(
        (ROOT / "configs" / "smoke.toml").read_text()
        + '\n[[strategies]]\nname = "gdp.bigm"\nlabel = "bigm"\n'
    )
    output = tmp_path / "output"
    original = __import__("tempfile").NamedTemporaryFile
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("no temporary files")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "exact_hull.experiment.inspect.tempfile.NamedTemporaryFile", fail_once
    )
    rows, _ = inspect_config(config, output)
    assert "no temporary files" in rows[0]["export_error"]
    assert rows[0]["model_path"] is None
    assert rows[1]["export_error"] is None
    assert Path(rows[1]["model_path"]).exists()
    assert len(list(output.glob("*.lp"))) == 1


def test_oracle_construction_has_cones_and_no_binary_variables():
    config = load_config(ROOT / "configs" / "qualification.toml")
    case_job = expand_jobs(config)[0]
    case = BenchmarkCase(case_job.instance_id, case_job.params, case_job.seed)
    _, descriptor = build_oracle_model(case_job.benchmark, case)
    assert descriptor["has_rotated_cone_structure"]
    assert descriptor["n_cone_rows"] > 0
    assert descriptor["n_binary_variables"] == 0


def test_binary_multiplication_structural_counts_include_cubic_products():
    def model():
        instance = ConcreteModel()
        instance.x = Var(bounds=(-2, 2))
        instance.d = Disjunct([1, 2])
        for disjunct in instance.d.values():
            disjunct.constraint = Constraint(expr=instance.x**2 <= 1)
        instance.choice = Disjunction(expr=list(instance.d.values()))
        instance.objective = Objective(expr=instance.x)
        return instance

    binmult, *_ = transform_model(model(), "gdp.binary_multiplication", {}, "solve")
    bigm, *_ = transform_model(model(), "gdp.bigm", {}, "solve")
    hull, *_ = transform_model(model(), "gdp.hull", {"EPS": 1e-4}, "solve")
    assert binmult["n_quadratic_terms"] > 0
    assert binmult["n_bilinear_binary_terms"] > 0
    assert bigm["n_quadratic_terms"] > 0
    assert bigm["n_bilinear_binary_terms"] == 0
    assert binmult["n_quadratic_terms"] == bigm["n_quadratic_terms"]
    assert hull["n_quadratic_terms"] > 0


def test_qualification_hull_structural_counts_match_bigm():
    config = load_config(ROOT / "configs" / "qualification.toml")
    job = expand_jobs(config)[0]
    case = BenchmarkCase(job.instance_id, job.params, job.seed)

    def transformed(strategy, options):
        model = BENCHMARKS[job.benchmark].build(case)
        counts, *_ = transform_model(model, strategy, options, "solve")
        return counts

    bigm = transformed("gdp.bigm", {})
    hull = transformed("gdp.hull", {"EPS": 1e-4})
    assert bigm["n_quadratic_terms"] == 36
    assert hull["n_quadratic_terms"] == bigm["n_quadratic_terms"]
    assert hull["n_bilinear_binary_terms"] > 0


def test_structural_counts_merge_algebraically_equivalent_monomials():
    def count(expression):
        model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.constraint = Constraint(expr=expression(model) <= 1)
        return structural_counts(model)

    direct = count(lambda model: model.x * model.y)
    split = count(lambda model: 0.5 * model.x * model.y + 0.5 * model.y * model.x)
    assert direct == split
    assert direct["n_quadratic_terms"] == 1


def test_structural_counts_treat_variable_quotients_as_opaque_boundaries():
    def count(expression):
        model = ConcreteModel()
        model.x = Var()
        model.denom = Var(bounds=(0.1, 2))
        model.constraint = Constraint(expr=expression(model) <= 1)
        return structural_counts(model)

    product = count(
        lambda model: model.denom
        * (model.x / model.denom)
        * (model.x / model.denom)
    )
    outside = count(lambda model: model.denom * (model.x / model.denom) ** 2)
    inside = count(lambda model: model.x**2 / model.denom)
    squared_denominator = count(lambda model: model.x**2 / model.denom**2)
    assert product == outside == inside == squared_denominator
    assert outside["n_quadratic_terms"] == 1


def test_structural_counts_combine_distinct_numerators_at_one_boundary():
    model = ConcreteModel()
    model.x = Var()
    model.y = Var()
    model.denom = Var(bounds=(0.1, 2))
    model.constraint = Constraint(
        expr=model.denom
        * (model.x / model.denom)
        * (model.y / model.denom)
        <= 1
    )
    assert structural_counts(model)["n_quadratic_terms"] == 1


def test_structural_counts_accept_literal_denominators():
    model = ConcreteModel()
    model.x = Var()
    model.y = Var()
    model.constraint = Constraint(expr=(model.x * model.y) / 2.0 <= 1)
    model.objective = Objective(expr=model.x)
    counts, *_ = transform_model(model, "gdp.bigm", {}, "solve")
    assert counts["n_quadratic_terms"] == 1


def test_structural_counts_canonicalize_linear_denominator_order():
    model = ConcreteModel()
    model.x = Var()
    model.y = Var()
    model.d = Var()
    model.e = Var()
    model.constraint = Constraint(
        expr=(model.x / (model.d + model.e))
        * (model.y / (model.e + model.d))
        <= 1
    )
    assert structural_counts(model)["n_quadratic_terms"] == 1


def test_structural_counts_distinguish_named_function_boundaries():
    model = ConcreteModel()
    model.x = Var()
    model.y = Var()
    model.d = Var()
    model.constraint = Constraint(
        expr=(model.x / sin(model.d)) * (model.y / cos(model.d)) <= 1
    )
    assert structural_counts(model)["n_quadratic_terms"] == 0


def test_structural_counts_do_not_cancel_across_distinct_boundaries():
    model = ConcreteModel()
    model.x = Var()
    model.y = Var()
    model.d1 = Var()
    model.d2 = Var()
    model.constraint = Constraint(
        expr=(model.x * model.y) / model.d1
        - (model.x * model.y) / model.d2
        <= 1
    )
    assert structural_counts(model)["n_quadratic_terms"] == 2


def test_structural_counts_remove_coefficient_cancellations():
    model = ConcreteModel()
    model.x = Var()
    model.y = Var()
    model.constraint = Constraint(expr=model.x * model.y - model.y * model.x + model.x <= 1)
    assert structural_counts(model)["n_quadratic_terms"] == 0


def test_oracle_missing_backend_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exact_hull.experiment.conic_oracle.importlib.util.find_spec", lambda name: None
    )
    with pytest.raises(RuntimeError, match="requires gurobipy; install gurobipy"):
        conic_bounds(ROOT / "configs" / "qualification.toml", tmp_path)


def test_oracle_captures_per_instance_backend_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exact_hull.experiment.conic_oracle.importlib.util.find_spec",
        lambda name: object() if name == "gurobipy" else None,
    )
    monkeypatch.setattr(
        "exact_hull.experiment.conic_oracle._solve_gurobi",
        lambda model: (_ for _ in ()).throw(RuntimeError("non-PSD model")),
    )
    rows = conic_bounds(ROOT / "configs" / "qualification.toml", tmp_path)
    assert len(rows) == 1
    assert rows[0]["oracle_bound"] is None
    assert "non-PSD model" in rows[0]["oracle_status"]
    assert (tmp_path / "conic-bounds.csv").exists()


def test_oracle_does_not_accept_nonoptimal_gurobi_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exact_hull.experiment.conic_oracle.importlib.util.find_spec",
        lambda name: object() if name == "gurobipy" else None,
    )
    monkeypatch.setattr(
        "exact_hull.experiment.conic_oracle._solve_gurobi",
        lambda model: {
            "status": "TIME_LIMIT",
            "optimal": False,
            "primal_objective": 2.0,
            "dual_bound": 1.0,
            "runtime_sec": 600.0,
        },
    )
    row = conic_bounds(ROOT / "configs" / "qualification.toml", tmp_path)[0]
    assert row["backend"] == "gurobipy"
    assert row["oracle_bound"] is None
    assert row["primal_objective"] == 2
    assert row["dual_bound"] == 1
    assert row["oracle_status"] == "TIME_LIMIT"


def test_oracle_uses_optimal_primal_as_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exact_hull.experiment.conic_oracle.importlib.util.find_spec",
        lambda name: object() if name == "gurobipy" else None,
    )
    monkeypatch.setattr(
        "exact_hull.experiment.conic_oracle._solve_gurobi",
        lambda model: {
            "status": "OPTIMAL",
            "optimal": True,
            "primal_objective": 1.5,
            "dual_bound": None,
            "runtime_sec": 2.0,
        },
    )
    row = conic_bounds(ROOT / "configs" / "qualification.toml", tmp_path)[0]
    assert row["backend"] == "gurobipy"
    assert row["oracle_bound"] == row["primal_objective"] == 1.5
    assert row["dual_bound"] is None


def test_oracle_refuses_models_with_fallback_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exact_hull.experiment.conic_oracle.importlib.util.find_spec",
        lambda name: object() if name == "gurobipy" else None,
    )
    monkeypatch.setattr(
        "exact_hull.experiment.conic_oracle.build_oracle_model",
        lambda benchmark, case: (object(), {"n_fallback_rows": 1}),
    )
    monkeypatch.setattr(
        "exact_hull.experiment.conic_oracle._solve_gurobi",
        lambda model: pytest.fail("a fallback model must not be solved"),
    )
    row = conic_bounds(ROOT / "configs" / "qualification.toml", tmp_path)[0]
    assert row["backend"] is None
    assert row["oracle_bound"] is None
    assert row["oracle_status"].startswith("refused: n_fallback_rows=1")


def test_gurobi_oracle_sets_limits_and_records_primal_dual_runtime(monkeypatch):
    parameters = {}

    class Oracle:
        NumIntVars = 0
        Status = 2
        SolCount = 1
        ObjVal = 3.0
        ObjBound = 2.5
        Runtime = 4.0

        @staticmethod
        def setParam(name, value):
            parameters[name] = value

        @staticmethod
        def optimize():
            pass

    constants = SimpleNamespace(
        LOADED=1,
        OPTIMAL=2,
        INFEASIBLE=3,
        INF_OR_UNBD=4,
        UNBOUNDED=5,
        CUTOFF=6,
        ITERATION_LIMIT=7,
        NODE_LIMIT=8,
        TIME_LIMIT=9,
        SOLUTION_LIMIT=10,
        INTERRUPTED=11,
        NUMERIC=12,
        SUBOPTIMAL=13,
        USER_OBJ_LIMIT=15,
    )
    monkeypatch.setitem(
        sys.modules,
        "gurobipy",
        SimpleNamespace(read=lambda path: Oracle(), GRB=constants),
    )

    class Model:
        @staticmethod
        def write(path, io_options):
            Path(path).touch()

    result = _solve_gurobi(Model())
    assert parameters == {"TimeLimit": 600, "Threads": 1}
    assert result == {
        "status": "OPTIMAL",
        "optimal": True,
        "primal_objective": 3.0,
        "dual_bound": 2.5,
        "runtime_sec": 4.0,
    }


def test_gurobi_oracle_accepts_optimal_value_without_dual_bound(monkeypatch):
    class MissingBoundOracle:
        NumIntVars = 0
        Status = 2
        SolCount = 1
        ObjVal = 3.0
        Runtime = 4.0

        @property
        def ObjBound(self):
            raise RuntimeError("attribute unavailable")

        @staticmethod
        def setParam(name, value):
            pass

        @staticmethod
        def optimize():
            pass

    monkeypatch.setitem(
        sys.modules,
        "gurobipy",
        SimpleNamespace(
            read=lambda path: MissingBoundOracle(),
            GRB=SimpleNamespace(OPTIMAL=2),
        ),
    )

    class Model:
        @staticmethod
        def write(path, io_options):
            Path(path).touch()

    result = _solve_gurobi(Model())
    assert result["optimal"]
    assert result["primal_objective"] == 3
    assert result["dual_bound"] is None


def test_oracle_rejects_indefinite_objective_in_fixed_config(tmp_path):
    config = tmp_path / "indefinite-objective.toml"
    config.write_text(
        (ROOT / "configs" / "qualification.toml")
        .read_text()
        .replace(
            "ensure_positive_definite = true",
            "ensure_positive_definite = true\n\n"
            "[instances.fixed]\nobjective_positive_definite = false",
        )
    )
    with pytest.raises(ValueError, match="convex-family"):
        conic_bounds(config, tmp_path / "output")


def test_content_ids_do_not_depend_on_grid_enumeration_order():
    benchmark = BENCHMARKS["random_quadratic"]
    common = {
        "n_dimensions": [2, 3],
        "n_disjunctions": [1, 2],
        "n_disjuncts_per_disjunction": 2,
        "n_constraints_per_disjunct": 1,
        "n_feasible_regions": 1,
        "ensure_positive_definite": True,
    }
    reversed_axes = {**common, "n_dimensions": [3, 2], "n_disjunctions": [2, 1]}
    first = {
        tuple(sorted(case.params.items())): case.instance_id
        for case in benchmark.cases(common, 7)
    }
    second = {
        tuple(sorted(case.params.items())): case.instance_id
        for case in benchmark.cases(reversed_axes, 7)
    }
    assert first == second
    assert all(
        instance_id.startswith("rq-") and len(instance_id) == 15
        for instance_id in first.values()
    )


def test_case_generation_rejects_truncated_hash_collision(monkeypatch):
    monkeypatch.setattr(
        "exact_hull.benchmarks.base.instance_digest", lambda *args: b"x" * 32
    )
    with pytest.raises(ValueError, match="Instance id collision"):
        BENCHMARKS["random_quadratic"].cases(
            {
                "n_dimensions": [2, 3],
                "n_disjunctions": 1,
                "n_disjuncts_per_disjunction": 2,
                "n_constraints_per_disjunct": 1,
                "n_feasible_regions": 1,
                "ensure_positive_definite": True,
            },
            7,
        )
