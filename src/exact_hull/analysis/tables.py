"""Summary and censoring tables for experiment reports."""

import math

import pandas as pd

from exact_hull.analysis.outcomes import (
    charged_time,
    correctness,
    ground_truth,
    ground_truth_with_sources,
    invalid_certificate,
    is_correct,
    reference_values,
    relaxation_certified,
    root_bound_valid,
)
from exact_hull.analysis.profiles import shifted_geometric_mean
from exact_hull.experiment.results import VERIFIED_OPTIMAL_STATUSES, RunRecord


def summary(
    records: list[RunRecord],
    references: dict | None = None,
    verification: dict[str, str] | None = None,
) -> pd.DataFrame:
    grouping = [
        "mode",
        "strategy",
        "solver",
        "subsolver",
        "variant",
        "status",
        "correct",
        "ground_truth_source",
    ]
    truth, sources = ground_truth_with_sources(records, references)
    certified_truth = reference_values(references)
    frame = pd.DataFrame(
        {
            "mode": record.mode,
            "strategy": record.strategy,
            "solver": record.solver,
            "subsolver": record.subsolver,
            "variant": record.variant,
            "status": record.status,
            "correct": correctness(record, truth, verification),
            "ground_truth_source": sources.get(record.instance_id, "unknown"),
            "solver_time_sec": record.solver_time_sec,
            "time_limit": record.time_limit,
            "run_id": record.run_id,
            "invalid_certificate": invalid_certificate(
                record, reference_truth=certified_truth
            ),
        }
        for record in records
        if record.mode == "solve"
    )
    if frame.empty:
        index = pd.MultiIndex.from_arrays([[] for _ in grouping], names=grouping)
        return pd.DataFrame(
            columns=[
                "jobs",
                "median_solver_time_sec",
            ],
            index=index,
        )
    table = frame.groupby(grouping, dropna=False).agg(
        jobs=("status", "size"), median_solver_time_sec=("solver_time_sec", "median")
    ).reset_index()
    return table.set_index(grouping)


def methods(
    records: list[RunRecord],
    planned_jobs: list[dict] | None = None,
    references: dict | None = None,
    verification: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return one solve-method row, including methods with no result records."""
    method = ["mode", "strategy", "solver", "subsolver", "variant"]
    planned_jobs = planned_jobs or [
        {
            "mode": record.mode,
            "label": record.strategy,
            "solver": record.solver,
            "subsolver": record.subsolver,
            "variant": record.variant,
            "instance_id": record.instance_id,
        }
        for record in records
    ]
    keys = {
        (job["mode"], job["label"], job["solver"], job["subsolver"], job.get("variant"))
        for job in planned_jobs
        if job["mode"] == "solve"
    }
    truth, sources = ground_truth_with_sources(records, references)
    certified_truth = reference_values(references)
    rows = []
    for key in sorted(keys, key=repr):
        group = [
            record
            for record in records
            if (
                record.mode,
                record.strategy,
                record.solver,
                record.subsolver,
                record.variant,
            )
            == key
        ]
        states = [correctness(record, truth, verification) for record in group]
        planned_group = [
            job
            for job in planned_jobs
            if (job["mode"], job["label"], job["solver"], job["subsolver"], job.get("variant"))
            == key
        ]
        source_values = {
            sources.get(job.get("instance_id"), "unknown") for job in planned_group
        }
        source = next(iter(source_values)) if len(source_values) == 1 else "mixed"
        rows.append(
            {
                **dict(zip(method, key, strict=True)),
                "ground_truth_source": source,
                "solved_count": sum(state is True for state in states),
                "planned_denominator": len(planned_group),
                "shifted_geometric_mean_10_sec": shifted_geometric_mean(
                    charged_time(record, state) for record, state in zip(group, states, strict=True)
                ),
                "invalid_certificate_count": sum(
                    invalid_certificate(record, reference_truth=certified_truth)
                    for record in group
                ),
                "unknown_reference_count": sum(
                    sources.get(record.instance_id) != "reference" for record in group
                ),
            }
        )
    return pd.DataFrame(rows)


BOUND_COLUMNS = [
    "instance_id",
    "strategy",
    "transformation",
    "solver",
    "subsolver",
    "variant",
    "ground_truth",
    "ground_truth_source",
    "objective",
    "lower_bound",
    "status",
    "root_bound",
    "root_status",
    "relaxation_bound",
    "relaxation_value",
    "relaxation_status",
    "relaxation_certified",
    "root_gap",
    "relaxation_gap",
    "relaxation_matches_cehr",
    "root_gap_closed",
    "invalid_certificate",
    "negative_control",
    "n_cone_rows",
    "n_fallback_rows",
    "n_equality_fallback_rows",
    "n_epigraph_vars",
    "n_quadratic_terms",
    "n_bilinear_binary_terms",
    "n_binary_square_terms",
    "n_disaggregated_vars",
    "n_nonlinear_constraints",
]


def bounds(records: list[RunRecord], references: dict | None = None) -> pd.DataFrame:
    """Combine solve, root, and relaxation records by formulation and solver."""
    truth, sources = ground_truth_with_sources(records, references)
    certified_truth = reference_values(references)
    grouped = {}
    for record in records:
        key = (
            record.instance_id,
            record.strategy,
            record.solver,
            record.subsolver,
            record.variant,
        )
        row = grouped.setdefault(
            key,
            {
                "instance_id": record.instance_id,
                "strategy": record.strategy,
                "transformation": record.transformation,
                "solver": record.solver,
                "subsolver": record.subsolver,
                "variant": record.variant,
                "ground_truth": truth.get(record.instance_id),
                "ground_truth_source": sources.get(record.instance_id, "unknown"),
                "invalid_certificate": False,
                "negative_control": record.variant == "convex",
            },
        )
        row["invalid_certificate"] = row["invalid_certificate"] or invalid_certificate(
            record, reference_truth=certified_truth
        )
        if record.mode == "solve":
            row.update(
                objective=record.objective,
                lower_bound=record.lower_bound,
                status=record.status,
            )
        elif record.mode == "root":
            row.update(root_bound=record.lower_bound, root_status=record.status)
        else:
            certified = relaxation_certified(record)
            row.update(
                relaxation_bound=record.lower_bound,
                relaxation_value=record.objective,
                relaxation_status=record.status,
                relaxation_certified=certified,
            )
        for name in (
            "n_cone_rows",
            "n_fallback_rows",
            "n_equality_fallback_rows",
            "n_epigraph_vars",
            "n_quadratic_terms",
            "n_bilinear_binary_terms",
            "n_binary_square_terms",
            "n_disaggregated_vars",
            "n_nonlinear_constraints",
        ):
            if getattr(record, name) is not None:
                row[name] = getattr(record, name)
    for row in grouped.values():
        ground = row["ground_truth"]
        scale = max(abs(ground), 1e-12) if ground is not None else None
        root_bound = row.get("root_bound")
        relaxation_bound = row.get("relaxation_bound")
        row["root_gap"] = (
            (ground - root_bound) / scale
            if ground is not None
            and root_bound is not None
            and root_bound_valid(root_bound, row.get("root_status"), ground)
            else float("nan")
        )
        row["relaxation_gap"] = (
            (ground - relaxation_bound) / scale
            if ground is not None
            and relaxation_bound is not None
            and row.get("relaxation_certified") is True
            else float("nan")
        )

    comparison_groups = {}
    for row in grouped.values():
        key = (row["instance_id"], row["solver"], row["subsolver"], row["variant"])
        comparison_groups.setdefault(key, []).append(row)
    for rows in comparison_groups.values():
        gehr = [
            row
            for row in rows
            if row["transformation"] == "gdp.hull_exact" and row.get("relaxation_bound") is not None
        ]
        cehr = [
            row
            for row in rows
            if row["transformation"].startswith("gdp.hull_exact_conic")
            and row.get("relaxation_bound") is not None
        ]
        if cehr and all(row.get("relaxation_certified") is True for row in cehr):
            for row in gehr:
                if row.get("relaxation_certified") is True:
                    row["relaxation_matches_cehr"] = all(
                        is_correct(row["relaxation_bound"], candidate["relaxation_bound"])
                        for candidate in cehr
                    )
        bigm = next((row for row in rows if row["transformation"] == "gdp.bigm"), None)
        if bigm is not None and root_bound_valid(
            bigm.get("root_bound"), bigm.get("root_status"), bigm.get("ground_truth")
        ):
            reference = bigm.get("ground_truth")
            denominator = (
                reference - bigm["root_bound"] if reference is not None else float("nan")
            )
            if (
                math.isfinite(denominator)
                and denominator > 1e-6 * max(1.0, abs(reference))
            ):
                for row in rows:
                    if root_bound_valid(
                        row.get("root_bound"),
                        row.get("root_status"),
                        row.get("ground_truth"),
                    ):
                        row["root_gap_closed"] = (
                            row["root_bound"] - bigm["root_bound"]
                        ) / denominator
    return pd.DataFrame(grouped.values(), columns=BOUND_COLUMNS)


def censoring(records: list[RunRecord], references: dict | None = None) -> pd.DataFrame:
    """Count excluded root and relaxation observations by method and reason."""
    rows = []
    truth = ground_truth(records, references)
    for record in records:
        reason = None
        if record.mode == "root":
            if record.lower_bound is None or not math.isfinite(record.lower_bound):
                reason = "missing_finite_dual_bound"
            elif not root_bound_valid(record.lower_bound, record.status):
                reason = "unacceptable_status"
            elif not root_bound_valid(
                record.lower_bound, record.status, truth.get(record.instance_id)
            ):
                reason = "dual_bound_exceeds_reference"
        elif record.mode == "relaxation" and not relaxation_certified(record):
            if record.objective is None or not math.isfinite(record.objective):
                reason = "missing_finite_primal"
            elif record.lower_bound is None or not math.isfinite(record.lower_bound):
                reason = "missing_finite_dual_bound"
            elif record.status not in VERIFIED_OPTIMAL_STATUSES and not (
                record.status == "feasible" and record.solver_status == "ok"
            ):
                reason = "unacceptable_status"
            else:
                reason = "primal_dual_disagreement"
        if reason is not None:
            rows.append(
                {
                    "mode": record.mode,
                    "strategy": record.strategy,
                    "solver": record.solver,
                    "subsolver": record.subsolver,
                    "variant": record.variant,
                    "reason": reason,
                }
            )
    columns = ["mode", "strategy", "solver", "subsolver", "variant", "reason", "excluded"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .groupby(columns[:-1], dropna=False)
        .size()
        .rename("excluded")
        .reset_index()
    )
