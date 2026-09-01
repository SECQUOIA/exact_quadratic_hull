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


def reference_values(references: dict | None) -> dict[str, float]:
    """Normalize only certified entries from a versioned references mapping."""
    if references is None:
        return {}
    raw = references.get("references", references)
    values = {}
    for instance_id, entry in raw.items():
        if not isinstance(entry, dict) or entry.get("status") != "certified":
            continue
        value = entry.get("objective")
        if isinstance(value, int | float) and math.isfinite(value):
            values[instance_id] = float(value)
    return values


def ground_truth(
    records: Iterable[RunRecord],
    references: dict | None = None,
    *,
    verification: dict[str, str] | None,
) -> dict[str, float]:
    """Use a certified reference per instance, else population fallback."""
    truth, _ = ground_truth_with_sources(
        records, references, verification=verification
    )
    return truth


def ground_truth_with_sources(
    records: Iterable[RunRecord],
    references: dict | None = None,
    *,
    verification: dict[str, str] | None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Return per-instance truth and whether it is certified or population-derived."""
    records = list(records)
    truth = {}
    for record in records:
        if (
            record.mode != "solve"
            or record.variant == "convex"
            or record.status not in VERIFIED_OPTIMAL_STATUSES
            or verification is None
            or verification.get(record.run_id) != "verified_feasible"
            or record.objective is None
            or not math.isfinite(record.objective)
        ):
            continue
        truth[record.instance_id] = min(truth.get(record.instance_id, math.inf), record.objective)
    sources = {instance_id: "population-fallback" for instance_id in truth}
    for instance_id, objective in reference_values(references).items():
        truth[instance_id] = objective
        sources[instance_id] = "reference"
    return truth, sources


def invalid_certificate(
    record: RunRecord,
    references: dict | None = None,
    reference_truth: dict[str, float] | None = None,
) -> bool:
    """Flag an optimality claim whose dual bound exceeds a certified objective."""
    truth = (reference_truth if reference_truth is not None else reference_values(references)).get(
        record.instance_id
    )
    return bool(
        truth is not None
        and record.mode == "solve"
        and record.status in VERIFIED_OPTIMAL_STATUSES
        and record.lower_bound is not None
        and math.isfinite(record.lower_bound)
        and record.lower_bound > truth
        and not is_correct(record.lower_bound, truth)
    )


def relaxation_certified(record: RunRecord, truth: float | None = None) -> bool:
    """Certify primal-dual agreement and consistency of the dual bound with truth."""
    acceptable = record.status in VERIFIED_OPTIMAL_STATUSES or (
        record.status == "feasible" and record.solver_status == "ok"
    )
    return bool(
        acceptable
        and record.objective is not None
        and record.lower_bound is not None
        and math.isfinite(record.objective)
        and math.isfinite(record.lower_bound)
        and is_correct(record.objective, record.lower_bound)
        and (
            truth is None
            or record.lower_bound < truth
            or is_correct(record.lower_bound, truth)
        )
    )


def root_bound_valid(
    bound: float | None, status: str | None, truth: float | None = None
) -> bool:
    """Apply the common finite/status and optional reference-consistency gate."""
    return bool(
        bound is not None
        and math.isfinite(bound)
        and status in {"node_limit", "optimal", "globally_optimal", "feasible"}
        and (
            truth is None
            or bound < truth
            or is_correct(bound, truth)
        )
    )


def correctness(
    record: RunRecord,
    truth: dict[str, float],
    verification: dict[str, str] | None,
    truth_sources: dict[str, str] | None,
) -> bool | None:
    """Return confirmed correct only against a certified reference."""
    expected = truth.get(record.instance_id)
    if expected is None:
        return None
    if verification is None or record.run_id not in verification:
        return None
    source = (truth_sources or {}).get(record.instance_id)
    if source == "population-fallback":
        return (
            False
            if record.status in VERIFIED_OPTIMAL_STATUSES
            and record.objective is not None
            and math.isfinite(record.objective)
            and record.objective > expected
            and not is_correct(record.objective, expected)
            else None
        )
    if source != "reference":
        return None
    return bool(
        record.status in VERIFIED_OPTIMAL_STATUSES
        and is_correct(record.objective, expected)
        and verification[record.run_id] == "verified_feasible"
    )


def negative_control(record: RunRecord) -> bool:
    """Return whether a convex-flag row intentionally violates solver assumptions."""
    return bool(
        record.variant == "convex"
        and record.transformation
        in {"gdp.hull_exact", "gdp.binary_multiplication"}
    )


def charged_time(record: RunRecord, confirmed_correct: bool | None) -> float:
    """Charge the full limit unless a record is independently confirmed correct."""
    if (
        confirmed_correct is True
        and record.solver_time_sec is not None
        and math.isfinite(record.solver_time_sec)
    ):
        return record.solver_time_sec
    return record.time_limit
