from pathlib import Path

from exact_hull.experiment.logparse import (
    parse_gams_solvestat,
    parse_root_relaxation,
    parse_solver_bounds,
    solver_timed_out,
)

FIXTURES = Path(__file__).parent / "fixtures" / "solver_logs"


def test_gurobi_log_excerpt():
    text = (FIXTURES / "gurobi.log").read_text()
    assert parse_root_relaxation(text, "gurobi") == 1.25
    assert parse_solver_bounds(text, "gurobi") == (2.0, 2.5)


def test_gurobi_root_falls_back_to_final_gap_line():
    text = "Best objective 2.500000e+00, best bound 2.000000e+00, gap 20%"
    assert parse_root_relaxation(text, "gurobi") == 2.0


def test_baron_log_excerpt():
    text = (FIXTURES / "baron.log").read_text()
    assert parse_root_relaxation(text, "baron") == 0.0
    assert parse_solver_bounds(text, "baron") == (2.77158991946373, 2.77158991946373)


def test_scip_log_excerpt():
    text = (FIXTURES / "scip.log").read_text()
    assert parse_root_relaxation(text, "scip") == 1.75
    assert parse_solver_bounds(text, "scip") == (2.0, 2.5)


def test_baron_preprocessing_paths_and_no_header_cross_line_capture():
    text = """Upper bound
1
Problem solved during preprocessing
Lower bound is -2.25
Best solution = 3.5
"""
    assert parse_root_relaxation(text, "baron") == -2.25
    assert parse_solver_bounds(text, "baron") == (-2.25, 3.5)


def test_baron_bounds_fall_back_to_iteration_table():
    text = """Iteration Time CPU Lower bound Upper bound Progress
1 0.01 20MB -4.5 7.0 0%
Best solution = 6.0
"""
    assert parse_solver_bounds(text, "baron") == (-4.5, 6.0)


def test_baron_root_handles_table_without_memory_column():
    text = """Iteration Time Lower bound Upper bound Progress
1 0.01 -4.5 7.0 0%
"""
    assert parse_root_relaxation(text, "baron") == -4.5


def test_scip_root_node_dual_bound_can_be_on_a_later_line():
    text = """root node processing
some intervening detail
dual bound : -1.75
"""
    assert parse_root_relaxation(text, "scip") == -1.75


def test_gams_solvestat():
    assert parse_gams_solvestat("**** SOLVER STATUS      3 Resource Interrupt\n") == 3


def test_solver_specific_time_limit_lines():
    assert solver_timed_out("Time limit reached", "gurobi")
    assert solver_timed_out("solving was interrupted [time limit reached]", "scip")
    assert solver_timed_out(" Max. allowable time exceeded", "baron")
    assert not solver_timed_out("Optimal solution found", "gurobi")
