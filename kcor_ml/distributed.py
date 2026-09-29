"""Small DistributedDataParallel runtime wrapper for single-node training."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random

import numpy as np
import torch
import torch.distributed as dist


@dataclass(slots=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls, requested_device: str) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            if not requested_device.startswith("cuda"):
                raise RuntimeError("DDP requires train.device to be 'cuda'")
            if not torch.cuda.is_available():
                raise RuntimeError("DDP requested by torchrun, but CUDA is unavailable")
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            return cls(
                enabled=True,
                rank=dist.get_rank(),
                local_rank=local_rank,
                world_size=dist.get_world_size(),
                device=torch.device("cuda", local_rank),
            )

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; set train.device to 'cpu'")
        return cls(False, 0, 0, 1, torch.device(requested_device))

    def seed_everything(self, seed: int) -> None:
        rank_seed = seed + self.rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(rank_seed)

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()

