from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from exact_hull.analysis.outcomes import (
    correctness,
    ground_truth,
    ground_truth_with_sources,
    invalid_certificate,
    is_correct,
)
from exact_hull.analysis.plots import performance_frames, style_map
from exact_hull.analysis.profiles import dolan_more, shifted_geometric_mean
from exact_hull.analysis.tables import bounds, censoring, methods, summary
from exact_hull.experiment.results import (
    RunRecord,
    read_campaign,
    write_json_atomic,
    write_record_atomic,
)


def record(**changes):
    base = RunRecord(
        run_id="r",
        benchmark="b",
        instance_id="i",
        instance_params={},
        seed=1,
        strategy="s",
        transformation="gdp.hull_exact",
        transformation_options={},
        solver="gams",
        subsolver="scip",
        variant=None,
        mode="solve",
        time_limit=10,
        duration_sec=1,
        solver_time_sec=1,
        status="optimal",
        objective=5,
    )
    return replace(base, **changes)


def test_timed_out_low_objective_cannot_poison_ground_truth():
    records = [
        record(run_id="good", objective=5),
        record(run_id="bad", status="timeout", objective=-99),
        record(run_id="local", status="locally_optimal", objective=-100),
    ]
    truth = ground_truth(records, verification={"good": "verified_feasible"})
    assert truth == {"i": 5}


def test_convex_variant_is_excluded_from_population_ground_truth():
    records = [record(run_id="convex", variant="convex", objective=-10), record(objective=5)]
    assert ground_truth(
        records,
        verification={"convex": "verified_feasible", "r": "verified_feasible"},
    ) == {"i": 5}


def test_nonfinite_objective_cannot_be_ground_truth():
    assert ground_truth(
        [record(objective=float("nan")), record(objective=float("inf"))],
        verification={"r": "verified_feasible"},
    ) == {}


def test_non_solve_modes_are_excluded_from_ground_truth_summary_and_profiles():
    records = [
        record(run_id="solve", objective=5),
        record(run_id="root", mode="root", objective=-10, solver_time_sec=0.1),
        record(run_id="relax", mode="relaxation", objective=-20, solver_time_sec=0.1),
    ]
    assert ground_truth(records, verification={"solve": "verified_feasible"}) == {"i": 5}
    assert summary(records)["jobs"].sum() == 1
    references = {"i": {"status": "certified", "objective": 5}}
    frame = performance_frames(
        records, references=references, verification={"solve": "verified_feasible"}
    )[("gams", "scip", None)]
    assert frame.at["i", "s"] == 1


def test_bounds_combines_modes_and_checks_gehr_cehr_relaxations():
    records = [
        record(run_id="solve", objective=5, lower_bound=5),
        record(run_id="root", mode="root", status="node_limit", objective=None, lower_bound=4),
        record(
            run_id="relax-gehr",
            mode="relaxation",
            objective=3,
            lower_bound=3,
        ),
        record(
            run_id="relax-cehr",
            mode="relaxation",
            strategy="CEHR",
            transformation="gdp.hull_exact_conic_no_cholesky",
            objective=3,
            lower_bound=3,
        ),
    ]
    verification = {record.run_id: "verified_feasible" for record in records}
    table = bounds(records, verification=verification)
    gehr = table.loc[table["transformation"] == "gdp.hull_exact"].iloc[0]
    assert gehr["ground_truth"] == 5
    assert gehr["root_bound"] == 4
    assert gehr["relaxation_bound"] == gehr["relaxation_value"] == 3
    assert gehr["relaxation_certified"]
    assert gehr["root_gap"] == pytest.approx(0.2)
    assert gehr["relaxation_gap"] == pytest.approx(0.4)
    assert gehr["relaxation_matches_cehr"]


def test_feasible_relaxation_is_certified_when_primal_and_bound_agree():
    table = bounds(
        [
            record(run_id="solve", objective=5),
            record(
                run_id="relax",
                mode="relaxation",
                status="feasible",
                solver_status="ok",
                objective=3.00001,
                lower_bound=3,
            ),
        ]
    )
    assert table.iloc[0]["relaxation_certified"]


def test_uncertified_relaxations_are_not_compared_and_incomplete_roots_have_no_gap():
    table = bounds(
        [
            record(run_id="solve", objective=5, lower_bound=5),
            record(
                run_id="root",
                strategy="incomplete",
                transformation="gdp.hull",
                mode="root",
                status="timeout",
                objective=None,
                lower_bound=4,
            ),
            record(
                run_id="relax-gehr",
                mode="relaxation",
                status="timeout",
                objective=3,
                lower_bound=2,
            ),
            record(
                run_id="relax-cehr",
                mode="relaxation",
                strategy="CEHR",
                transformation="gdp.hull_exact_conic_no_cholesky",
                objective=3,
                lower_bound=3,
            ),
        ]
    )
    gehr = table.loc[table["transformation"] == "gdp.hull_exact"].iloc[0]
    incomplete = table.loc[table["strategy"] == "incomplete"].iloc[0]
    assert pd.isna(gehr["relaxation_matches_cehr"])
    assert pd.isna(incomplete["root_gap"])
    assert pd.isna(gehr["relaxation_gap"])


def test_feasible_root_bound_is_retained_and_censoring_reasons_are_reported():
    records = [
        record(run_id="solve", objective=5),
        record(run_id="root", mode="root", status="feasible", lower_bound=4),
        record(
            run_id="relax",
            mode="relaxation",
            status="timeout",
            objective=3,
            lower_bound=2,
        ),
    ]
    verification = {record.run_id: "verified_feasible" for record in records}
    assert bounds(records, verification=verification).iloc[0]["root_gap"] == pytest.approx(0.2)
    excluded = censoring(records)
    assert excluded.iloc[0]["reason"] == "unacceptable_status"
    assert excluded.iloc[0]["excluded"] == 1


def test_censoring_reports_bound_above_verified_population_truth():
    records = [
        record(run_id="solve", objective=5),
        record(
            run_id="root",
            mode="root",
            status="node_limit",
            objective=None,
            lower_bound=6,
        ),
    ]
    table = censoring(records, verification={"solve": "verified_feasible"})
    assert table.iloc[0]["reason"] == "dual_bound_exceeds_reference"


@pytest.mark.parametrize("objective", [4.0, 5.00001])
def test_relaxation_consistency_gate_accepts_values_below_or_close_to_truth(objective):
    records = [
        record(run_id="solve", objective=5),
        record(
            run_id="relax",
            mode="relaxation",
            objective=objective,
            lower_bound=objective,
        ),
    ]
    table = bounds(records, verification={"solve": "verified_feasible"})
    assert table.iloc[0]["relaxation_certified"]


def test_relaxation_consistency_gate_censors_value_above_truth():
    records = [
        record(run_id="solve", objective=5),
        record(
            run_id="relax",
            mode="relaxation",
            objective=5.001,
            lower_bound=5.001,
        ),
    ]
    verification = {"solve": "verified_feasible"}
    row = bounds(records, verification=verification).iloc[0]
    assert not row["relaxation_certified"]
    assert pd.isna(row["relaxation_gap"])
    excluded = censoring(records, verification=verification)
    assert excluded.iloc[0]["reason"] == "dual_bound_exceeds_reference"


def test_relaxation_consistency_gate_uses_the_bound_that_defines_the_gap():
    records = [
        record(run_id="solve", objective=5),
        record(
            run_id="relax",
            mode="relaxation",
            objective=5.00004,
            lower_bound=5.00009,
        ),
    ]
    verification = {"solve": "verified_feasible"}
    row = bounds(records, verification=verification).iloc[0]
    assert not row["relaxation_certified"]
    assert pd.isna(row["relaxation_gap"])
    excluded = censoring(records, verification=verification)
    assert excluded.iloc[0]["reason"] == "dual_bound_exceeds_reference"


def test_reference_invalid_certificate_and_root_gap_closed():
    references = {"i": {"status": "certified", "objective": 5}}
    bad = record(lower_bound=5.1)
    assert invalid_certificate(bad, references)
    records = [
        record(run_id="solve", objective=5, lower_bound=5),
        record(
            run_id="bigm-root",
            strategy="bigm",
            transformation="gdp.bigm",
            mode="root",
            status="node_limit",
            objective=None,
            lower_bound=3,
        ),
        record(
            run_id="gehr-root",
            mode="root",
            status="node_limit",
            objective=None,
            lower_bound=4,
        ),
    ]
    table = bounds(records, references=references)
    gehr = table.loc[table["transformation"] == "gdp.hull_exact"].iloc[0]
    assert gehr["root_gap_closed"] == pytest.approx(0.5)


def test_correctness_uses_relative_tolerance_with_absolute_near_zero():
    assert is_correct(100.0005, 100, rtol=1e-5, atol=1e-7)
    assert is_correct(5e-8, 0, rtol=1e-5, atol=1e-7)
    assert not is_correct(2e-7, 0, rtol=1e-5, atol=1e-7)


def test_dolan_more_hand_computed_table():
    times = pd.DataFrame({"a": [1, 4, 6], "b": [2, 2, 3], "c": [4, 8, 1]})
    profile = dolan_more(times, np.array([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(profile["a"], [1 / 3, 2 / 3, 2 / 3])
    np.testing.assert_allclose(profile["b"], [1 / 3, 2 / 3, 1])
    np.testing.assert_allclose(profile["c"], [1 / 3, 1 / 3, 1])


def test_dolan_more_floors_zero_solver_times():
    times = pd.DataFrame({"a": [0.0], "b": [1.0]})
    profile = dolan_more(times, np.array([1.0, 1_000_000.0]))
    np.testing.assert_allclose(profile["a"], [1, 1])
    np.testing.assert_allclose(profile["b"], [0, 1])


def test_shifted_geometric_mean_hand_value_and_timeout_charging():
    assert shifted_geometric_mean([0, 30], shift=10) == pytest.approx(10)
    table = methods(
        [
            record(run_id="solved", solver_time_sec=0),
            record(run_id="timeout", status="timeout", solver_time_sec=1, time_limit=30),
        ],
        references={"i": {"status": "certified", "objective": 5}},
        verification={"solved": "verified_feasible", "timeout": "verified_feasible"},
    )
    assert table["shifted_geometric_mean_10_sec"].iloc[0] == pytest.approx(10)
    assert table["planned_denominator"].iloc[0] == 2


def test_methods_preserves_unobserved_plan_and_unknown_references():
    planned = [
        {
            "mode": "solve",
            "label": strategy,
            "solver": "gams",
            "subsolver": "scip",
            "variant": None,
        }
        for strategy in ("s", "missing")
    ]
    table = methods([record()], planned_jobs=planned)
    assert set(table["strategy"]) == {"s", "missing"}
    observed = table.loc[table["strategy"] == "s"].iloc[0]
    missing = table.loc[table["strategy"] == "missing"].iloc[0]
    assert observed["unknown_reference_count"] == 1
    assert observed["solved_count"] == 0
    assert missing["planned_denominator"] == 1
    assert pd.isna(missing["shifted_geometric_mean_10_sec"])


def test_methods_counts_mbigm_estimation_once_per_instance():
    records = [
        record(
            run_id="first",
            instance_id="i1",
            transformation="gdp.mbigm",
            m_estimation_time_total_sec=3,
            m_estimation_cache_hit=False,
        ),
        record(
            run_id="repeat",
            instance_id="i1",
            transformation="gdp.mbigm",
            m_estimation_time_total_sec=3,
            m_estimation_cache_hit=True,
        ),
        record(
            run_id="second-instance",
            instance_id="i2",
            transformation="gdp.mbigm",
            m_estimation_time_total_sec=4,
            m_estimation_cache_hit=False,
        ),
    ]
    table = methods(records)
    assert table["m_estimation_time_total_sec"].iloc[0] == 7


def test_solver_groups_are_not_merged_and_unsolved_instances_remain_in_denominator():
    records = [
        record(run_id="a1", instance_id="i1", strategy="a", subsolver="gurobi", solver_time_sec=1),
        record(run_id="b1", instance_id="i1", strategy="b", subsolver="gurobi", solver_time_sec=10),
        record(run_id="a2", instance_id="i2", strategy="a", subsolver="gurobi", solver_time_sec=10),
        record(run_id="b2", instance_id="i2", strategy="b", subsolver="gurobi", solver_time_sec=1),
        record(run_id="a3", instance_id="i3", strategy="a", subsolver="gurobi", status="timeout"),
        record(run_id="b3", instance_id="i3", strategy="b", subsolver="gurobi", status="timeout"),
        record(run_id="a4", instance_id="i1", strategy="a", subsolver="scip", solver_time_sec=8),
        record(run_id="b4", instance_id="i1", strategy="b", subsolver="scip", solver_time_sec=2),
    ]
    references = {
        instance_id: {"status": "certified", "objective": 5}
        for instance_id in ("i1", "i2", "i3")
    }
    frames = performance_frames(
        records,
        references=references,
        verification={record.run_id: "verified_feasible" for record in records},
    )
    assert set(frames) == {("gams", "gurobi", None), ("gams", "scip", None)}
    assert frames[("gams", "gurobi", None)].shape == (3, 2)
    assert frames[("gams", "scip", None)].shape == (3, 2)
    profile = dolan_more(frames[("gams", "gurobi", None)], np.array([100.0]))
    np.testing.assert_allclose(profile.iloc[0], [2 / 3, 2 / 3])


def test_incorrect_verified_record_counts_as_failure():
    records = [
        record(run_id="truth", strategy="truth", objective=5, solver_time_sec=2),
        record(run_id="wrong", strategy="wrong", objective=6, solver_time_sec=1),
    ]
    frame = performance_frames(
        records,
        references={"i": {"status": "certified", "objective": 5}},
        verification={"truth": "verified_feasible", "wrong": "verified_feasible"},
    )[("gams", "scip", None)]
    assert frame.at["i", "truth"] == 2
    assert np.isnan(frame.at["i", "wrong"])


def test_population_fallback_truth_is_used_and_labeled_per_instance():
    records = [
        record(run_id="certified", instance_id="cert", objective=2),
        record(run_id="fallback", instance_id="fallback", objective=5),
    ]
    references = {
        "cert": {"status": "certified", "objective": 2},
        "fallback": {"status": "reference_unknown", "objective": None},
    }
    verification = {record.run_id: "verified_feasible" for record in records}
    table = summary(records, references=references, verification=verification).reset_index()
    assert set(table["ground_truth_source"]) == {"reference", "population-fallback"}
    frames = performance_frames(
        records, references=references, verification=verification
    )[("gams", "scip", None)]
    assert np.isnan(frames.at["fallback", "s"])


def test_population_fallback_requires_verified_feasible_sources_and_only_refutes():
    records = [
        record(run_id="source", objective=5),
        record(run_id="wrong", objective=6),
        record(run_id="unverified-better", objective=1),
    ]
    verification = {
        "source": "verified_feasible",
        "wrong": "verified_feasible",
        "unverified-better": "not_verifiable",
    }
    truth, sources = ground_truth_with_sources(records, verification=verification)
    assert truth == {"i": 5}
    assert sources == {"i": "population-fallback"}
    assert correctness(records[0], truth, verification, sources) is None
    assert correctness(records[1], truth, verification, sources) is False


def test_methods_labels_certified_truth_without_planned_jobs():
    current = record()
    table = methods(
        [current],
        references={"i": {"status": "certified", "objective": 5}},
        verification={current.run_id: "verified_feasible"},
    )
    assert table["ground_truth_source"].tolist() == ["reference"]


def test_missing_verification_is_unknown_not_wrong():
    assert correctness(record(), {"i": 5}, {}, {"i": "reference"}) is None


def test_truth_provenance_arguments_are_required():
    with pytest.raises(TypeError):
        ground_truth([record()])
    with pytest.raises(TypeError):
        ground_truth_with_sources([record()])
    with pytest.raises(TypeError):
        correctness(record(), {"i": 5}, {})


def test_negative_control_is_limited_to_nonconvex_defining_functions():
    transformations = {
        "bigm": "gdp.bigm",
        "binmult": "gdp.binary_multiplication",
        "hull": "gdp.hull",
        "GEHR": "gdp.hull_exact",
        "CEHR": "gdp.hull_exact_conic_no_cholesky",
    }
    records = [
        record(
            run_id=label,
            strategy=label,
            transformation=transformation,
            variant="convex",
        )
        for label, transformation in transformations.items()
    ]
    table = bounds(records)
    labeled = dict(zip(table["strategy"], table["negative_control"], strict=True))
    assert {label for label, is_negative in labeled.items() if is_negative} == {
        "binmult",
        "GEHR",
    }


def test_root_gap_closed_rejects_near_zero_denominator():
    references = {"i": {"status": "certified", "objective": 5}}
    table = bounds(
        [
            record(run_id="solve", objective=5),
            record(
                run_id="bigm",
                mode="root",
                strategy="bigm",
                transformation="gdp.bigm",
                status="node_limit",
                lower_bound=5 - 1e-7,
            ),
            record(run_id="other", mode="root", status="node_limit", lower_bound=5),
        ],
        references,
    )
    assert table["root_gap_closed"].isna().all()


def test_reference_inconsistent_root_is_excluded_everywhere():
    records = [
        record(run_id="solve", objective=5),
        record(
            run_id="bigm",
            mode="root",
            strategy="bigm",
            transformation="gdp.bigm",
            status="node_limit",
            lower_bound=3,
        ),
        record(
            run_id="bad-root",
            mode="root",
            strategy="bad",
            status="node_limit",
            lower_bound=5.1,
        ),
    ]
    references = {"i": {"status": "certified", "objective": 5}}
    table = bounds(records, references)
    assert pd.isna(table.loc[table["strategy"] == "bad", "root_gap_closed"].iloc[0])
    frame = performance_frames(records, mode="root", references=references)[
        ("gams", "scip", None)
    ]
    assert np.isnan(frame.at["i", "bad"])
    assert frame.at["i", "bigm"] == 1


def test_campaign_styles_are_unique_and_shared_across_solver_groups():
    # Repeated labels represent the same strategy in different solver groups.
    styles = style_map(["zeta", "alpha", "beta", "alpha"])
    assert len({(style["color"], style["linestyle"]) for style in styles.values()}) == 3
    assert styles["alpha"] == style_map(["zeta", "alpha", "beta"])["alpha"]


def test_manifest_keeps_missing_instances_and_skips_bad_and_foreign_records(tmp_path):
    planned_jobs = [
        {
            "run_id": f"planned-{index}",
            "benchmark": "b",
            "instance_id": f"i{index}",
            "label": "s",
            "strategy": "gdp.hull_exact",
            "solver": "gams",
            "subsolver": "scip",
            "variant": None,
            "mode": "solve",
        }
        for index in range(1, 6)
    ]
    manifest = {
        "schema_version": 1,
        "config": {},
        "planned_jobs": planned_jobs,
        "instances": [
            {"instance_id": f"i{index}", "params": {}, "seed": index} for index in range(1, 6)
        ],
    }
    write_json_atomic(manifest, tmp_path / "manifest.json")
    write_record_atomic(
        record(run_id="planned-1", instance_id="i1"),
        tmp_path / "jobs" / "planned-1" / "result.json",
    )
    write_record_atomic(
        record(run_id="foreign", instance_id="i2", objective=-100),
        tmp_path / "jobs" / "foreign" / "result.json",
    )
    malformed = tmp_path / "jobs" / "planned-2" / "result.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{not json")
    wrong_schema = tmp_path / "jobs" / "planned-3" / "result.json"
    wrong_schema.parent.mkdir(parents=True)
    wrong_schema.write_text('{"run_id": "planned-3"}')
    write_record_atomic(
        record(run_id="planned-5", instance_id="i4"),
        tmp_path / "jobs" / "planned-4" / "result.json",
    )
    with pytest.warns(UserWarning, match="Skipped 4 invalid or foreign"):
        loaded_manifest, records = read_campaign(tmp_path)
    assert [item.run_id for item in records] == ["planned-1"]
    frame = performance_frames(
        records,
        loaded_manifest["planned_jobs"],
        [item["instance_id"] for item in loaded_manifest["instances"]],
    )[("gams", "scip", None)]
    assert list(frame.index) == ["i1", "i2", "i3", "i4", "i5"]
