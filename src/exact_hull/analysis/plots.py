"""Stable report plots grouped by complete solver identity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from exact_hull.analysis.outcomes import (
    correctness,
    ground_truth_with_sources,
    relaxation_certified,
    root_bound_valid,
)
from exact_hull.analysis.profiles import dolan_more
from exact_hull.experiment.results import RunRecord, read_campaign

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
    mode: str = "solve",
    references: dict | None = None,
    verification: dict[str, str] | None = None,
) -> dict[SolverGroup, pd.DataFrame]:
    """Build one strategy matrix per solver group, retaining every planned instance."""
    if mode not in {"solve", "root", "relaxation"}:
        raise ValueError(f"Unknown profile mode: {mode}")
    truth, _ = ground_truth_with_sources(records, references)
    records = [record for record in records if record.mode == mode]
    if planned_jobs is not None:
        planned_jobs = [job for job in planned_jobs if job["mode"] == mode]
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
            completed = correctness(record, truth, verification)
            if mode == "root":
                completed = root_bound_valid(
                    record.lower_bound, record.status, truth.get(record.instance_id)
                )
            elif mode == "relaxation":
                completed = relaxation_certified(record)
            if completed is not True or record.solver_time_sec is None:
                continue
            current = frame.at[record.instance_id, record.strategy]
            frame.at[record.instance_id, record.strategy] = (
                record.solver_time_sec
                if pd.isna(current)
                else min(current, record.solver_time_sec)
            )
        frames[group] = frame
    return frames


def _slug(group: SolverGroup) -> str:
    return "-".join(str(value).lower().replace("_", "-") for value in group if value)


def plot_run(
    run_directory: Path, mode: str = "solve", tolerance: float = 1e-5
) -> list[Path]:
    from exact_hull.analysis.verify import verify_run

    manifest, records = read_campaign(run_directory)
    planned_jobs = manifest["planned_jobs"]
    planned_instances = [instance["instance_id"] for instance in manifest["instances"]]
    reference_path = run_directory / "references.json"
    references = json.loads(reference_path.read_text()) if reference_path.exists() else None
    verification = {
        row["run_id"]: row["verification_status"]
        for row in verify_run(run_directory, tolerance=tolerance)
    }
    styles = style_map(job["label"] for job in planned_jobs if job["mode"] == mode)
    destinations = []
    for group, frame in performance_frames(
        records,
        planned_jobs,
        planned_instances,
        mode=mode,
        references=references,
        verification=verification,
    ).items():
        if frame.empty:
            continue
        profile = dolan_more(frame)
        figure, axis = plt.subplots()
        for strategy in profile.columns:
            axis.plot(profile.index, profile[strategy], label=strategy, **styles[strategy])
        axis.set_xscale("log")
        axis.set(xlabel="performance ratio", ylabel="fraction solved")
        axis.legend()
        suffix = "" if mode == "solve" else f"-{mode}"
        destination = run_directory / f"performance-profile-{_slug(group)}{suffix}.png"
        figure.savefig(destination, bbox_inches="tight", dpi=150)
        plt.close(figure)
        destinations.append(destination)
    return destinations
