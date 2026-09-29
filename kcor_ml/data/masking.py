"""Training-only masking for self-supervised two-frame learning."""

from __future__ import annotations

import torch


class PairMasker:
    def __init__(self, fraction: float, patch_size: int, frame_drop_probability: float = 0.0):
        self.fraction = fraction
        self.patch_size = patch_size
        self.frame_drop_probability = frame_drop_probability

    def __call__(self, images: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 2:
            raise ValueError("images must have shape [B, 2, H, W]")
        batch, frames, height, width = images.shape
        patch_h = max(1, (height + self.patch_size - 1) // self.patch_size)
        patch_w = max(1, (width + self.patch_size - 1) // self.patch_size)
        coarse = torch.rand((batch, frames, patch_h, patch_w), device=images.device) < self.fraction
        mask = coarse.repeat_interleave(self.patch_size, -2).repeat_interleave(self.patch_size, -1)
        mask = mask[..., :height, :width] & valid_mask.bool()
        if self.frame_drop_probability:
            drop = torch.rand((batch, frames, 1, 1), device=images.device) < self.frame_drop_probability
            mask = mask | (drop & valid_mask.bool())
        corrupted = images.masked_fill(mask, 0.0)
        return corrupted, mask

