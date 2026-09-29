"""Metric aggregation independent of the training loop."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist


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

    def synchronize(self, device: torch.device) -> None:
        """Sum metric totals and counts across DDP ranks."""
        if not dist.is_available() or not dist.is_initialized():
            return
        names = sorted(self.totals)
        values = [self.totals[name] for name in names] + [float(self.count)]
        tensor = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced = tensor.cpu().tolist()
        self.totals = defaultdict(float, dict(zip(names, reduced[:-1])))
        self.count = int(reduced[-1])
