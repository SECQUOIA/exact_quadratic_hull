"""Stable report plots grouped by complete solver identity."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from exact_hull.analysis.outcomes import ground_truth, is_correct
from exact_hull.analysis.profiles import dolan_more
from exact_hull.experiment.results import VERIFIED_OPTIMAL_STATUSES, RunRecord, read_campaign

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
LINESTYLES = ("-", "--", "-.", ":")
SolverGroup = tuple[str, str, str | None]


def _solver_group(record: RunRecord) -> SolverGroup:
    return record.solver, record.subsolver, record.variant


def style_map(strategies) -> dict[str, dict[str, str]]:
    """Assign campaign-wide styles, cycling colors before line styles."""
    labels = sorted(set(strategies))
    capacity = len(COLORS) * len(LINESTYLES)
    if len(labels) > capacity:
        raise ValueError(f"At most {capacity} distinct strategy styles are supported")
    return {
        label: {
            "color": COLORS[index % len(COLORS)],
            "linestyle": LINESTYLES[index // len(COLORS)],
        }
        for index, label in enumerate(labels)
    }


def performance_frames(
    records: list[RunRecord],
    planned_jobs: list[dict] | None = None,
    planned_instances: list[str] | None = None,
) -> dict[SolverGroup, pd.DataFrame]:
    """Build one strategy matrix per solver group, retaining every planned instance."""
    records = [record for record in records if record.mode == "solve"]
    if planned_jobs is not None:
        planned_jobs = [job for job in planned_jobs if job["mode"] == "solve"]
    truth = ground_truth(records)
    if planned_jobs is None:
        groups = sorted({_solver_group(record) for record in records}, key=repr)
        strategies_by_group = {
            group: sorted({record.strategy for record in records if _solver_group(record) == group})
            for group in groups
        }
    else:
        groups = sorted(
            {(job["solver"], job["subsolver"], job.get("variant")) for job in planned_jobs},
            key=repr,
        )
        strategies_by_group = {
            group: sorted(
                {
                    job["label"]
                    for job in planned_jobs
                    if (job["solver"], job["subsolver"], job.get("variant")) == group
                }
            )
            for group in groups
        }
    if planned_instances is None:
        planned_instances = sorted({record.instance_id for record in records})
    frames = {}
    for group in groups:
        grouped = [record for record in records if _solver_group(record) == group]
        strategies = strategies_by_group[group]
        frame = pd.DataFrame(index=planned_instances, columns=strategies, dtype=float)
        for record in grouped:
            if (
                record.status not in VERIFIED_OPTIMAL_STATUSES
                or not is_correct(record.objective, truth.get(record.instance_id))
                or record.solver_time_sec is None
            ):
                continue
            current = frame.at[record.instance_id, record.strategy]
            frame.at[record.instance_id, record.strategy] = (
                record.solver_time_sec if pd.isna(current) else min(current, record.solver_time_sec)
            )
        frames[group] = frame
    return frames


def _slug(group: SolverGroup) -> str:
    return "-".join(str(value).lower().replace("_", "-") for value in group if value)


def plot_run(run_directory: Path) -> list[Path]:
    manifest, records = read_campaign(run_directory)
    planned_jobs = manifest["planned_jobs"]
    planned_instances = [instance["instance_id"] for instance in manifest["instances"]]
    styles = style_map(job["label"] for job in planned_jobs if job["mode"] == "solve")
    destinations = []
    for group, frame in performance_frames(records, planned_jobs, planned_instances).items():
        if frame.empty:
            continue
        profile = dolan_more(frame)
        figure, axis = plt.subplots()
        for strategy in profile.columns:
            axis.plot(profile.index, profile[strategy], label=strategy, **styles[strategy])
        axis.set_xscale("log")
        axis.set(xlabel="performance ratio", ylabel="fraction solved")
        axis.legend()
        destination = run_directory / f"performance-profile-{_slug(group)}.png"
        figure.savefig(destination, bbox_inches="tight", dpi=150)
        plt.close(figure)
        destinations.append(destination)
    return destinations
