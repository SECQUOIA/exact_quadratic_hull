import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from exact_hull import cli
from exact_hull.benchmarks import BENCHMARKS
from exact_hull.benchmarks.base import BenchmarkCase
from exact_hull.experiment.runner import expand_jobs, load_config

ROOT = Path(__file__).parents[1]


def test_all_configs_load_and_expected_case_counts():
    expected = {
        "random_psd.toml": 240,
        "random_psd_conic.toml": 240,
        "random_nonconvex.toml": 100,
        "kmeans.toml": 96,
        "clay.toml": 12,
        "cstr.toml": 9,
        "smoke.toml": 1,
    }
    for path in (ROOT / "configs").glob("*.toml"):
        jobs = expand_jobs(load_config(path))
        assert len({job.instance_id for job in jobs}) == expected[path.name]


@pytest.mark.parametrize("name", ["random_psd.toml", "random_psd_conic.toml"])
def test_random_psd_uses_design_of_record_axes(name):
    config = load_config(ROOT / "configs" / name)
    instances = config["instances"]
    assert instances["n_dimensions"] == [3, 4, 5, 6, 7]
    assert instances["n_disjunctions"] == [3, 4, 5, 6, 7, 8, 9, 10]
    assert instances["n_disjuncts_per_disjunction"] == [10, 11, 12, 13, 14, 15]
    assert instances["n_constraints_per_disjunct"] == 10
    assert instances["n_feasible_regions"] == 10
    assert instances["ensure_positive_definite"] is True


def test_unknown_transformation_fails_at_load(tmp_path):
    config = tmp_path / "bad.toml"
    config.write_text(
        """
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = 2
n_clusters = 2
n_points = 3
[[strategies]]
name = "gdp.not_real"
[[solvers]]
subsolver = "scip"
"""
    )
    with pytest.raises(ValueError, match="Unknown transformation"):
        load_config(config)


def test_misspelled_instance_parameter_fails_at_load(tmp_path):
    config = tmp_path / "bad-parameter.toml"
    config.write_text(
        """
[experiment]
benchmark = "kmeans"
[instances]
n_dimentions = 2
n_clusters = 2
n_points = 3
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    )
    with pytest.raises(ValueError, match="n_dimentions"):
        load_config(config)


def test_fixed_table_preserves_tuple_valued_parameters(tmp_path):
    config = tmp_path / "fixed.toml"
    config.write_text(
        """
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = [2, 3]
n_clusters = 2
n_points = 3
[instances.fixed]
coord_range = [-2.0, 4.0]
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    )
    jobs = expand_jobs(load_config(config))
    assert len(jobs) == 2
    assert all(job.params["coord_range"] == [-2.0, 4.0] for job in jobs)


def test_tuple_valued_parameter_must_be_a_valid_fixed_value(tmp_path):
    config = tmp_path / "bad-range.toml"
    config.write_text(
        """
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = 2
n_clusters = 2
n_points = 3
coord_range = [-2.0, 4.0]
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    )
    with pytest.raises(ValueError, match="instances.fixed"):
        load_config(config)


@pytest.mark.parametrize(
    "value",
    ["[1.0, -1.0]", "[nan, 1.0]", "[true, false]"],
)
def test_fixed_range_requires_finite_increasing_numbers(tmp_path, value):
    config = tmp_path / "bad-fixed-range.toml"
    config.write_text(
        f"""
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = 2
n_clusters = 2
n_points = 3
[instances.fixed]
coord_range = {value}
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    )
    with pytest.raises(ValueError, match="finite, increasing"):
        load_config(config)


@pytest.mark.parametrize("subsolver", ["gurobi", "baron"])
def test_unknown_solver_variant_fails_at_load(tmp_path, subsolver):
    config = tmp_path / "bad-variant.toml"
    config.write_text(
        f"""
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = 2
n_clusters = 2
n_points = 3
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "{subsolver}"
variant = "mystery"
"""
    )
    with pytest.raises(ValueError, match="variant"):
        load_config(config)


def test_job_fingerprint_includes_time_limit_and_variant():
    config = load_config(ROOT / "configs" / "smoke.toml")
    original = expand_jobs(config)[0].run_id
    changed_time = deepcopy(config)
    changed_time["experiment"]["time_limit"] += 1
    changed_variant = deepcopy(config)
    changed_variant["solvers"][0]["variant"] = "convex"
    assert expand_jobs(changed_time)[0].run_id != original
    assert expand_jobs(changed_variant)[0].run_id != original


def test_duplicate_planned_job_is_rejected():
    config = load_config(ROOT / "configs" / "smoke.toml")
    config["solvers"].append(dict(config["solvers"][0]))
    with pytest.raises(ValueError, match="Duplicate planned job"):
        expand_jobs(config)


def test_duplicate_computational_strategy_is_rejected(tmp_path):
    config = tmp_path / "duplicate-strategy.toml"
    config.write_text(
        """
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = 2
n_clusters = 2
n_points = 3
[[strategies]]
name = "gdp.hull_exact"
label = "first"
[[strategies]]
name = "gdp.hull_exact"
label = "second"
[[solvers]]
subsolver = "scip"
"""
    )
    with pytest.raises(ValueError, match="Duplicate computational strategy"):
        load_config(config)


def test_label_cannot_name_two_computational_strategies(tmp_path):
    config = tmp_path / "duplicate-label.toml"
    config.write_text(
        """
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = 2
n_clusters = 2
n_points = 3
[[strategies]]
name = "gdp.hull_exact"
label = "same"
[[strategies]]
name = "gdp.hull"
label = "same"
[[solvers]]
subsolver = "scip"
"""
    )
    with pytest.raises(ValueError, match="refers to multiple strategies"):
        load_config(config)


def test_display_label_does_not_change_run_identity():
    config = load_config(ROOT / "configs" / "smoke.toml")
    original = expand_jobs(config)[0].run_id
    config["strategies"][0]["label"] = "renamed for display"
    assert expand_jobs(config)[0].run_id == original


def test_omitted_and_explicit_defaults_have_same_run_identity(tmp_path):
    omitted = tmp_path / "omitted.toml"
    explicit = tmp_path / "explicit.toml"
    common = """
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = 2
n_clusters = 2
n_points = 3
{fixed}
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    omitted.write_text(common.format(fixed=""))
    explicit.write_text(common.format(fixed="[instances.fixed]\ncoord_range = [-1.0, 1.0]"))
    assert (
        expand_jobs(load_config(omitted))[0].run_id == expand_jobs(load_config(explicit))[0].run_id
    )


def test_omitted_and_explicit_transformation_defaults_have_same_identity(tmp_path):
    common = """
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = 2
n_clusters = 2
n_points = 3
[[strategies]]
name = "gdp.hull_exact"
{options}
[[solvers]]
subsolver = "scip"
"""
    omitted = tmp_path / "omitted-options.toml"
    explicit = tmp_path / "explicit-options.toml"
    omitted.write_text(common.format(options=""))
    explicit.write_text(common.format(options="[strategies.options]\nEPS = 1e-4"))
    assert (
        expand_jobs(load_config(omitted))[0].run_id == expand_jobs(load_config(explicit))[0].run_id
    )


def test_explicit_default_and_omitted_option_are_duplicate_strategies(tmp_path):
    config = tmp_path / "duplicate-default-option.toml"
    config.write_text(
        """
[experiment]
benchmark = "kmeans"
[instances]
n_dimensions = 2
n_clusters = 2
n_points = 3
[[strategies]]
name = "gdp.hull_exact"
label = "omitted"
[[strategies]]
name = "gdp.hull_exact"
label = "explicit"
[strategies.options]
EPS = 1e-4
[[solvers]]
subsolver = "scip"
"""
    )
    with pytest.raises(ValueError, match="Duplicate computational strategy"):
        load_config(config)


def test_integer_and_float_range_values_have_same_seed_and_identity(tmp_path):
    common = """
[experiment]
benchmark = "kmeans"
base_seed = 7
[instances]
n_dimensions = 2
n_clusters = 2
n_points = 3
[instances.fixed]
coord_range = {coord_range}
[[strategies]]
name = "gdp.hull_exact"
[strategies.options]
EPS = {eps}
[[solvers]]
subsolver = "scip"
"""
    integers = tmp_path / "integers.toml"
    floats = tmp_path / "floats.toml"
    integers.write_text(common.format(coord_range="[-1, 1]", eps="1"))
    floats.write_text(common.format(coord_range="[-1.0, 1.0]", eps="1.0"))
    integer_job = expand_jobs(load_config(integers))[0]
    float_job = expand_jobs(load_config(floats))[0]
    assert integer_job.seed == float_job.seed
    assert integer_job.run_id == float_job.run_id


@pytest.mark.parametrize(
    ("benchmark", "fixed", "expected"),
    [
        ("cstr", "NT = 5", {"NT": 5}),
        ("clay", 'metric = "l1"', {"instance": "CLay0203", "metric": "l1"}),
        ("random_quadratic", "n_dimensions = 3", {"n_dimensions": 3}),
    ],
)
def test_non_range_fixed_parameters_do_not_overlap_defaults(tmp_path, benchmark, fixed, expected):
    path = tmp_path / f"{benchmark}.toml"
    path.write_text(
        f"""
[experiment]
benchmark = "{benchmark}"
[instances]
[instances.fixed]
{fixed}
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    )
    job = expand_jobs(load_config(path))[0]
    for key, value in expected.items():
        assert job.params[key] == value


def test_axis_fixed_and_default_placement_have_same_identity(tmp_path):
    common = """
[experiment]
benchmark = "cstr"
base_seed = 11
{instances}
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    placements = {
        "axis": "[instances]\nNT = 5",
        "fixed": "[instances]\n[instances.fixed]\nNT = 5",
        "default": "[instances]",
    }
    jobs = []
    for name, instances in placements.items():
        path = tmp_path / f"{name}.toml"
        path.write_text(common.format(instances=instances))
        jobs.append(expand_jobs(load_config(path))[0])
    assert len({job.seed for job in jobs}) == 1
    assert len({job.run_id for job in jobs}) == 1


def test_float_default_axis_is_canonicalized_elementwise_and_builds(tmp_path):
    path = tmp_path / "sparsity-axis.toml"
    path.write_text(
        """
[experiment]
benchmark = "random_quadratic"
[instances]
sparsity_factor = [0, 0.5]
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    )
    jobs = expand_jobs(load_config(path))
    assert [job.params["sparsity_factor"] for job in jobs] == [0.0, 0.5]
    for job in jobs:
        BENCHMARKS["random_quadratic"].build(BenchmarkCase(job.instance_id, job.params, job.seed))


@pytest.mark.parametrize(
    ("benchmark", "parameter"),
    [("random_quadratic", "n_dimensions"), ("cstr", "NT")],
)
def test_integral_float_instance_parameter_has_integer_identity(tmp_path, benchmark, parameter):
    common = """
[experiment]
benchmark = "{benchmark}"
[instances]
{parameter} = {value}
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    jobs = []
    for name, value in (("integer", "2"), ("float", "2.0")):
        path = tmp_path / f"{benchmark}-{name}.toml"
        path.write_text(common.format(benchmark=benchmark, parameter=parameter, value=value))
        jobs.append(expand_jobs(load_config(path))[0])
    assert jobs[0].params[parameter] == 2
    assert jobs[0].seed == jobs[1].seed
    assert jobs[0].run_id == jobs[1].run_id


@pytest.mark.parametrize("value", ["2.5", "true"])
def test_noninteger_instance_parameter_is_rejected(tmp_path, value):
    path = tmp_path / "bad-integer.toml"
    path.write_text(
        f"""
[experiment]
benchmark = "random_quadratic"
[instances]
n_dimensions = {value}
[[strategies]]
name = "gdp.hull_exact"
[[solvers]]
subsolver = "scip"
"""
    )
    with pytest.raises(ValueError, match="n_dimensions.*integers"):
        load_config(path)


def test_doctor_warns_when_gdx_results_are_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli, "_gdx_results_available", lambda: False)
    assert cli._doctor() == 0
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "OBJVAL NA" in output


@pytest.mark.parametrize("arguments", [["doctor"], ["list-transformations"]])
def test_information_cli_from_arbitrary_cwd(tmp_path, arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "exact_hull.cli", *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_smoke_dry_run_from_arbitrary_cwd(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "exact_hull.cli",
            "run",
            str(ROOT / "configs" / "smoke.toml"),
            "--dry-run",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "planned jobs: 1" in result.stdout
