"""Summary tables for experiment reports."""

import pandas as pd

from exact_hull.analysis.outcomes import ground_truth, is_correct
from exact_hull.experiment.results import VERIFIED_OPTIMAL_STATUSES, RunRecord


def summary(records: list[RunRecord]) -> pd.DataFrame:
    grouping = ["strategy", "solver", "subsolver", "variant", "status", "correct"]
    truth = ground_truth(records)
    frame = pd.DataFrame(
        {
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
    )
    if frame.empty:
        index = pd.MultiIndex.from_arrays([[] for _ in grouping], names=grouping)
        return pd.DataFrame(columns=["jobs", "median_solver_time_sec"], index=index)
    return frame.groupby(grouping, dropna=False).agg(
        jobs=("status", "size"), median_solver_time_sec=("solver_time_sec", "median")
    )
