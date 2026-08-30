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
    for name in ("MIPGapAbs", "FeasibilityTol", "OptimalityTol", "IntFeasTol"):
        assert name in gurobi
    baron = "\n".join(options_for("baron", 60))
    for name in ("EpsA", "AbsConFeasTol", "RelConFeasTol"):
        assert name in baron
    scip = "\n".join(options_for("scip", 60))
    for name in ("limits/absgap", "numerics/feastol", "numerics/dualfeastol"):
        assert name in scip


def test_scip_convex_variant_is_an_option():
    assert "constraints/nonlinear/assumeconvex = TRUE" in options_for("scip", 60, "convex")


@pytest.mark.parametrize("subsolver", ["gurobi", "baron", "scip"])
def test_unknown_solver_variant_is_rejected(subsolver):
    with pytest.raises(ValueError, match="variant"):
        options_for(subsolver, 60, "unknown")
