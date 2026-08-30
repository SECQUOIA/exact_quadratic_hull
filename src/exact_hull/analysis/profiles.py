"""Dolan–Moré and cumulative-solved performance profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dolan_more(times: pd.DataFrame, taus: np.ndarray | None = None) -> pd.DataFrame:
    """Compute fractions within tau of each instance's best finite time."""
    if taus is None:
        taus = np.geomspace(1, 100, 101)
    floored = times.clip(lower=1e-6)
    best = floored.min(axis=1, skipna=True)
    ratios = floored.div(best, axis=0).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(
        {
            strategy: [(ratios[strategy] <= tau).sum() / len(times) for tau in taus]
            for strategy in times.columns
        },
        index=taus,
    )
