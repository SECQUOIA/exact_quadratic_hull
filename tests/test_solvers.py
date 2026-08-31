import pytest

from exact_hull.experiment.solvers import TOLS, options_for


def test_shared_tolerance_policy_and_native_options():
    assert TOLS == {
        "rel_gap": 1e-6,
        "abs_gap": 1e-10,
        "feas": 1e-6,
        "opt": 1e-6,
        "int": 1e-5,
    }
    gurobi = "\n".join(options_for("gurobi", 60))
    assert "option optca=1e-10;" in gurobi
    assert "FuncNonlinear 1" in gurobi
    assert "NonConvex 2" in gurobi
    for name in ("MIPGapAbs", "FeasibilityTol", "OptimalityTol", "IntFeasTol"):
        assert name in gurobi
    scip = "\n".join(options_for("scip", 60))
    for name in ("limits/absgap", "numerics/feastol", "numerics/dualfeastol"):
        assert name in scip


def test_scip_convex_variant_is_an_option():
    assert "constraints/nonlinear/assumeconvex = TRUE" in options_for("scip", 60, "convex")


def test_gurobi_auto_variant_leaves_convexity_recognition_enabled():
    options = options_for("gurobi", 60, "auto")
    assert "NonConvex 2" not in options
    assert "FuncNonlinear 1" in options


@pytest.mark.parametrize("subsolver", ["gurobi", "scip"])
def test_unknown_solver_variant_is_rejected(subsolver):
    with pytest.raises(ValueError, match="variant"):
        options_for(subsolver, 60, "unknown")


def test_root_options_apply_node_limits_without_changing_tolerances():
    gurobi = options_for("gurobi", 60, mode="root")
    assert "NodeLimit 1" in gurobi
    scip = options_for("scip", 60, mode="root")
    assert {"limits/nodes = 1", "limits/totalnodes = 1", "limits/restarts = 0"} <= set(scip)
    assert options_for("gurobi", 60, mode="relaxation") == options_for("gurobi", 60)


def test_unsupported_subsolver_names_the_subsolver():
    with pytest.raises(ValueError, match="Unsupported GAMS subsolver: baron"):
        options_for("baron", 60)
