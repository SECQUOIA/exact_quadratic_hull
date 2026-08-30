"""Summary tables for experiment reports."""

import pandas as pd

from exact_hull.analysis.outcomes import ground_truth, is_correct
from exact_hull.experiment.results import VERIFIED_OPTIMAL_STATUSES, RunRecord


def summary(records: list[RunRecord]) -> pd.DataFrame:
    grouping = ["mode", "strategy", "solver", "subsolver", "variant", "status", "correct"]
    truth = ground_truth(records)
    frame = pd.DataFrame(
        {
            "mode": record.mode,
            "strategy": record.strategy,
            "solver": record.solver,
            "subsolver": record.subsolver,
            "variant": record.variant,
            "status": record.status,
            "correct": record.status in VERIFIED_OPTIMAL_STATUSES
            and is_correct(record.objective, truth.get(record.instance_id)),
            "solver_time_sec": record.solver_time_sec,
        }
        for record in records
        if record.mode == "solve"
    )
    if frame.empty:
        index = pd.MultiIndex.from_arrays([[] for _ in grouping], names=grouping)
        return pd.DataFrame(columns=["jobs", "median_solver_time_sec"], index=index)
    return frame.groupby(grouping, dropna=False).agg(
        jobs=("status", "size"), median_solver_time_sec=("solver_time_sec", "median")
    )


BOUND_COLUMNS = [
    "instance_id",
    "strategy",
    "transformation",
    "solver",
    "subsolver",
    "variant",
    "ground_truth",
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
]


def bounds(records: list[RunRecord]) -> pd.DataFrame:
    """Combine solve, root, and relaxation records by formulation and solver."""
    truth = ground_truth(records)
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
            },
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
            certified = record.status in VERIFIED_OPTIMAL_STATUSES or (
                record.status == "feasible"
                and record.solver_status == "ok"
                and is_correct(record.objective, record.lower_bound)
            )
            row.update(
                relaxation_bound=record.lower_bound,
                relaxation_value=record.objective,
                relaxation_status=record.status,
                relaxation_certified=certified,
            )
    for row in grouped.values():
        ground = row["ground_truth"]
        scale = max(abs(ground), 1e-12) if ground is not None else None
        root_bound = row.get("root_bound")
        relaxation_bound = row.get("relaxation_bound")
        row["root_gap"] = (
            (ground - root_bound) / scale
            if ground is not None
            and root_bound is not None
            and row.get("root_status") in {"node_limit", "optimal", "globally_optimal"}
            else float("nan")
        )
        row["relaxation_gap"] = (
            (ground - relaxation_bound) / scale
            if ground is not None and relaxation_bound is not None
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
    return pd.DataFrame(grouped.values(), columns=BOUND_COLUMNS)
