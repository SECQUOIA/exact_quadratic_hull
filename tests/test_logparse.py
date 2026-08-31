from pathlib import Path

from exact_hull.experiment.logparse import (
    parse_solver_bounds,
    parse_solver_metadata,
    solver_timed_out,
)

FIXTURES = Path(__file__).parent / "fixtures" / "solver_logs"


def test_gurobi_log_excerpt():
    text = (FIXTURES / "gurobi.log").read_text()
    assert parse_solver_bounds(text, "gurobi") == (-0.5659660912989, -0.5659517803165)


def test_gurobi_no_incumbent_still_returns_the_final_bound():
    text = (FIXTURES / "gurobi_no_incumbent.log").read_text()
    assert parse_solver_bounds(text, "gurobi") == (-0.6861929638562, None)


def test_scip_log_excerpt():
    text = (FIXTURES / "scip.log").read_text()
    assert parse_solver_bounds(text, "scip") == (-0.597255461110791, -0.56595253492095)


def test_solver_versions_and_presolve_statistics():
    gurobi = parse_solver_metadata((FIXTURES / "gurobi.log").read_text(), "gurobi")
    assert gurobi == {
        "version": "13.0.2",
        "presolved_num_variables": 80,
        "presolved_num_constraints": 120,
        "presolved_num_nonzeros": 500,
        "presolved_num_soc": 3,
        "presolved_num_bilinear": 4,
        "presolved_num_quadratic": 5,
        "presolved_num_nonlinear": None,
    }
    scip = parse_solver_metadata((FIXTURES / "scip.log").read_text(), "scip")
    assert scip["version"] == "10.0.3"
    assert scip["presolved_num_variables"] == 72
    assert scip["presolved_num_constraints"] == 41
    assert scip["presolved_num_quadratic"] is None
    assert scip["presolved_num_nonlinear"] == 7


def test_solver_specific_time_limit_lines():
    assert solver_timed_out("Time limit reached", "gurobi")
    assert solver_timed_out("solving was interrupted [time limit reached]", "scip")
    assert not solver_timed_out("Optimal solution found", "gurobi")
