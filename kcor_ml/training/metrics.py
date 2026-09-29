"""Metric aggregation independent of the training loop."""

from __future__ import annotations

from collections import defaultdict


class MeanMetrics:
    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)
        self.count = 0

    def update(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.totals[name] += value
        self.count += 1

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {name: float("nan") for name in self.totals}
        return {name: value / self.count for name, value in self.totals.items()}

