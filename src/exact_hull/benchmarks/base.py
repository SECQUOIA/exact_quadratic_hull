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


def instance_digest(benchmark: str, base_seed: int, params: dict[str, Any]) -> bytes:
    """Hash the normalized content that defines a benchmark instance."""
    import hashlib
    import json

    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{benchmark}:{base_seed}:{payload}".encode()).digest()


def stable_seed(base_seed: int, params: dict[str, Any], benchmark: str = "") -> int:
    """Derive a stable per-instance seed without Python's randomized hash."""
    return int.from_bytes(instance_digest(benchmark, base_seed, params)[:8], "big")


def content_instance_id(prefix: str, benchmark: str, base_seed: int, params: dict[str, Any]) -> str:
    """Return a short human-readable, content-addressed instance identifier."""
    return f"{prefix}-{instance_digest(benchmark, base_seed, params).hex()[:12]}"


def validate_case_ids(cases: list[BenchmarkCase]) -> list[BenchmarkCase]:
    """Reject the vanishingly unlikely case of a truncated-digest collision."""
    seen: dict[str, tuple[dict[str, Any], int]] = {}
    for case in cases:
        content = (case.params, case.seed)
        previous = seen.setdefault(case.instance_id, content)
        if previous != content:
            raise ValueError(
                f"Instance id collision for {case.instance_id}: {previous[0]} and {case.params}"
            )
    return cases
