"""Common benchmark case interface."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Protocol

from pyomo.environ import ConcreteModel


@dataclass(frozen=True)
class BenchmarkCase:
    instance_id: str
    params: dict[str, Any]
    seed: int


class Benchmark(Protocol):
    def cases(self, instance_config: dict[str, Any], base_seed: int) -> list[BenchmarkCase]: ...

    def build(self, case: BenchmarkCase) -> ConcreteModel: ...

    def solution(self, model: ConcreteModel) -> dict[str, Any]: ...


def grid_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand list-valued axes while preserving values in ``fixed``."""
    fixed = config.get("fixed", {})
    if not isinstance(fixed, dict):
        raise ValueError("instances.fixed must be a table")
    axes = {key: value for key, value in config.items() if key != "fixed"}
    overlap = set(axes) & set(fixed)
    if overlap:
        raise ValueError(f"Instance keys cannot be both axes and fixed: {sorted(overlap)}")
    keys = list(axes)
    values = [value if isinstance(value, list) else [value] for value in axes.values()]
    rows = []
    for combination in product(*values):
        row = dict(zip(keys, combination, strict=True))
        row.update(fixed)
        rows.append(row)
    return rows


def stable_seed(base_seed: int, params: dict[str, Any]) -> int:
    """Derive a stable per-instance seed without Python's randomized hash."""
    import hashlib
    import json

    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{base_seed}:{payload}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
