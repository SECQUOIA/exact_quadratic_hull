"""Importing the package exposes GAMS's GDX reader under the name Pyomo imports."""

import importlib
import sys

import pytest


def test_gdxcc_alias_registered_when_gamsapi_present():
    gdx = pytest.importorskip("gams.core.gdx", reason="gamsapi wheel not installed")
    import exact_hull  # noqa: F401

    assert "gdxcc" in sys.modules
    gdxcc = importlib.import_module("gdxcc")
    for name in ("new_gdxHandle_tp", "gdxCreateD", "gdxOpenRead", "gdxDataReadRaw", "GMS_SVIDX_NA"):
        assert hasattr(gdxcc, name)
    assert gdxcc is gdx or hasattr(gdxcc, "gdxCreateD")
