"""Exact quadratic hull transformations and experiment tools."""

import sys


def _alias_legacy_gdxcc() -> None:
    """Expose GAMS's GDX reader under the legacy ``gdxcc`` name Pyomo imports.

    Pyomo's GAMS interface reads results through GDX only when ``import gdxcc``
    succeeds; otherwise it falls back to a text path that crashes on ``OBJVAL NA``
    records from timed-out solves. GAMS >= 42 ships the same SWIG module as
    ``gams.core.gdx`` (the ``gamsapi`` wheel), so register it under the old name.
    """
    if "gdxcc" in sys.modules:
        return
    try:
        import gdxcc  # noqa: F401
    except ImportError:
        try:
            from gams.core import gdx
        except ImportError:
            return
        sys.modules["gdxcc"] = gdx


_alias_legacy_gdxcc()

from exact_hull.transformations import TRANSFORMATIONS  # noqa: E402

__all__ = ["TRANSFORMATIONS"]
