"""Adapter for the vendored constrained-layout model."""

from pyomo.environ import value

from exact_hull.benchmarks._vendor.cons_layout_model import (
    build_constrained_layout_model,
    constrained_layout_model_examples,
)
from exact_hull.benchmarks.base import (
    BenchmarkCase,
    content_instance_id,
    grid_rows,
    stable_seed,
    validate_case_ids,
)


class ClayBenchmark:
    def cases(self, instance_config, base_seed):
        return validate_case_ids([
            BenchmarkCase(
                content_instance_id("clay", "clay", base_seed, params),
                params,
                stable_seed(base_seed, params, "clay"),
            )
            for params in grid_rows(instance_config)
        ])

    def build(self, case):
        return build_constrained_layout_model(
            constrained_layout_model_examples[case.params["instance"]],
            metric=case.params["metric"],
        )

    def solution(self, model):
        return {
            "rect_x": {str(index): value(model.rect_x[index]) for index in model.rectangles},
            "rect_y": {str(index): value(model.rect_y[index]) for index in model.rectangles},
        }


BENCHMARK = ClayBenchmark()
