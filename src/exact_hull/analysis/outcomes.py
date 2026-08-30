"""Consistent correctness and ground-truth rules."""

from __future__ import annotations

import math
from collections.abc import Iterable

from exact_hull.experiment.results import VERIFIED_OPTIMAL_STATUSES, RunRecord


def is_correct(value: float | None, truth: float | None, rtol=1e-5, atol=1e-7) -> bool:
    """Use relative tolerance, with an absolute fallback for values near zero."""
    if value is None or truth is None or not math.isfinite(value) or not math.isfinite(truth):
        return False
    return abs(value - truth) <= max(atol, rtol * abs(truth))


def ground_truth(records: Iterable[RunRecord]) -> dict[str, float]:
    """Return the best objective per instance from verified-optimal runs only."""
    truth = {}
    for record in records:
        if (
            record.mode != "solve"
            or record.status not in VERIFIED_OPTIMAL_STATUSES
            or record.objective is None
            or not math.isfinite(record.objective)
        ):
            continue
        truth[record.instance_id] = min(truth.get(record.instance_id, math.inf), record.objective)
    return truth
