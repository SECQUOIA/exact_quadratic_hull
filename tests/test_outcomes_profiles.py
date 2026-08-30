from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from exact_hull.analysis.outcomes import ground_truth, is_correct
from exact_hull.analysis.plots import performance_frames, style_map
from exact_hull.analysis.profiles import dolan_more
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
        time_limit=10,
        duration_sec=1,
        solver_time_sec=1,
        status="optimal",
        objective=5,
    )
    return replace(base, **changes)


def test_timed_out_low_objective_cannot_poison_ground_truth():
    truth = ground_truth(
        [
            record(run_id="good", objective=5),
            record(run_id="bad", status="timeout", objective=-99),
            record(run_id="local", status="locally_optimal", objective=-100),
        ]
    )
    assert truth == {"i": 5}


def test_nonfinite_objective_cannot_be_ground_truth():
    assert ground_truth([record(objective=float("nan")), record(objective=float("inf"))]) == {}


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


def test_solver_groups_are_not_merged_and_unsolved_instances_remain_in_denominator():
    records = [
        record(run_id="a1", instance_id="i1", strategy="a", subsolver="gurobi", solver_time_sec=1),
        record(run_id="b1", instance_id="i1", strategy="b", subsolver="gurobi", solver_time_sec=10),
        record(run_id="a2", instance_id="i2", strategy="a", subsolver="gurobi", solver_time_sec=10),
        record(run_id="b2", instance_id="i2", strategy="b", subsolver="gurobi", solver_time_sec=1),
        record(run_id="a3", instance_id="i3", strategy="a", subsolver="gurobi", status="timeout"),
        record(run_id="b3", instance_id="i3", strategy="b", subsolver="gurobi", status="timeout"),
        record(run_id="a4", instance_id="i1", strategy="a", subsolver="baron", solver_time_sec=8),
        record(run_id="b4", instance_id="i1", strategy="b", subsolver="baron", solver_time_sec=2),
    ]
    frames = performance_frames(records)
    assert set(frames) == {("gams", "gurobi", None), ("gams", "baron", None)}
    assert frames[("gams", "gurobi", None)].shape == (3, 2)
    assert frames[("gams", "baron", None)].shape == (3, 2)
    profile = dolan_more(frames[("gams", "gurobi", None)], np.array([100.0]))
    np.testing.assert_allclose(profile.iloc[0], [2 / 3, 2 / 3])


def test_incorrect_verified_record_counts_as_failure():
    records = [
        record(run_id="truth", strategy="truth", objective=5, solver_time_sec=2),
        record(run_id="wrong", strategy="wrong", objective=6, solver_time_sec=1),
    ]
    frame = performance_frames(records)[("gams", "scip", None)]
    assert frame.at["i", "truth"] == 2
    assert np.isnan(frame.at["i", "wrong"])


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
