"""Noise2Noise training, evaluation, and single-frame inference.

Adjacent K-Cor frames (15 s apart) see the same corona with independent noise.
Training the network to predict frame B from frame A alone, with a squared
error, makes its optimum the noise-free expectation of the image. Inference
uses one frame only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ..data.dataset import KCorPairDataset, read_fits_image
from ..data.records import PairManifest
from ..distributed import DistributedContext
from .model import DenoisingUNet
from .profile import radial_noise, radial_scale, scale_at


@dataclass(slots=True)
class DenoiseConfig:
    manifest: str = "artifacts/pb2_pairs.json"
    output_dir: str = "artifacts/denoise_n2n"
    crop_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 20
    learning_rate: float = 2.0e-4
    minimum_learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    blocks_per_level: int = 2
    profile_frames: int = 100
    seed: int = 17
    device: str = "cuda"
    log_every_steps: int = 50

    @classmethod
    def load(cls, path: str | Path) -> "DenoiseConfig":
        return cls(**json.loads(Path(path).read_text()))


def build_model(config: DenoiseConfig) -> DenoisingUNet:
    return DenoisingUNet(tuple(config.channels), config.blocks_per_level)


def _augment(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Apply one random flip/transpose to every [B, C, H, W] tensor alike."""
    if random.random() < 0.5:
        tensors = tuple(t.flip(-1) for t in tensors)
    if random.random() < 0.5:
        tensors = tuple(t.flip(-2) for t in tensors)
    if random.random() < 0.5:
        tensors = tuple(t.transpose(-2, -1) for t in tensors)
    return tensors


def noise2noise_views(
    batch: dict[str, object], scale: torch.Tensor, noise: torch.Tensor, device: torch.device, augment: bool
):
    """Return normalized (input, target, input_valid, loss_weight, radius) in both directions.

    The loss weight is zero on invalid pixels and (scale / noise)^2 elsewhere,
    so squared errors in normalized units become squared errors in units of
    the local noise level. Weights depend only on position, which leaves the
    Noise2Noise optimum (the noise-free image) unchanged.
    """
    images = batch["images"].to(device, non_blocking=True).float()
    valid = batch["valid_mask"].to(device, non_blocking=True).bool()
    radius = batch["radial_coordinate"].to(device, non_blocking=True).float()
    pixel_scale = scale_at(scale, radius)
    images = images / pixel_scale
    whitening = (pixel_scale / scale_at(noise, radius)).square()
    if augment:
        images, valid, radius, whitening = _augment(images, valid, radius, whitening)
    first, second = images[:, 0:1], images[:, 1:2]
    first_valid, second_valid = valid[:, 0:1], valid[:, 1:2]
    weight = (first_valid & second_valid).float() * whitening[:, 0:1]
    return (
        torch.cat((first, second)),
        torch.cat((second, first)),
        torch.cat((first_valid, second_valid)),
        torch.cat((weight, weight)),
        torch.cat((radius[:, 0:1], radius[:, 1:2])),
    )


class Noise2NoiseTrainer:
    def __init__(self, config: DenoiseConfig, distributed: DistributedContext):
        self.config = config
        self.distributed = distributed
        self.device = distributed.device
        self.output_dir = Path(config.output_dir)
        self.manifest = PairManifest.load(config.manifest)
        self.scale, self.noise = self._radial_profiles()
        model = build_model(config).to(self.device)
        self.model = (
            DistributedDataParallel(model, device_ids=[distributed.local_rank])
            if distributed.enabled
            else model
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.minimum_learning_rate
        )
        self.amp = config.mixed_precision and self.device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def _radial_profiles(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Training pairs only, so the normalization never sees test dates.
        pairs = self.manifest.splits["train"]
        step = max(1, len(pairs) // self.config.profile_frames)
        selected = pairs[::step][: self.config.profile_frames]
        scale = radial_scale([pair.first.path for pair in selected])
        noise = radial_noise([(pair.first.path, pair.second.path) for pair in selected])
        return torch.from_numpy(scale).to(self.device), torch.from_numpy(noise).to(self.device)

    def _loader(self, split: str, training: bool) -> DataLoader:
        dataset = KCorPairDataset(self.manifest.splits[split], crop_size=self.config.crop_size, training=training)
        sampler = (
            DistributedSampler(dataset, shuffle=training, seed=self.config.seed)
            if self.distributed.enabled
            else None
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=training and sampler is None,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
            drop_last=training,
        )

    def _run(self, loader: DataLoader, training: bool, epoch: int) -> dict[str, float]:
        self.model.train(training)
        # Sums of squared error (model and identity baseline) and pixel count.
        totals = torch.zeros(3, device=self.device, dtype=torch.float64)
        context = torch.enable_grad if training else torch.no_grad
        with context():
            for step, batch in enumerate(loader, start=1):
                inputs, targets, input_valid, weight, radius = noise2noise_views(
                    batch, self.scale, self.noise, self.device, augment=training
                )
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    prediction = self.model(inputs, input_valid, radius)
                squared = (prediction.float() - targets).square() * weight
                count = (weight > 0).sum().float().clamp_min(1.0)
                loss = squared.sum() / count
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step} paths={batch['paths']}")
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                identity = ((inputs - targets).square() * weight).sum()
                totals += torch.stack((squared.sum().detach().double(), identity.detach().double(), count.double()))
                if training and self.distributed.is_main and step % self.config.log_every_steps == 0:
                    print(f"epoch={epoch} step={step}/{len(loader)} loss={loss.item():.5g}", flush=True)
        if self.distributed.enabled:
            torch.distributed.all_reduce(totals)
        model_mse, identity_mse = (totals[:2] / totals[2].clamp_min(1.0)).tolist()
        # Errors are in units of the per-frame noise variance: identity -> ~2
        # and a perfect denoiser -> ~1, so this fraction runs from 0 to ~1.
        removed = (identity_mse - model_mse) / max(identity_mse / 2.0, 1.0e-30)
        return {"mse": model_mse, "identity_mse": identity_mse, "noise_removed_fraction": removed}

    def fit(self) -> None:
        if self.distributed.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.output_dir / "radial_scale.npy", self.scale.cpu().numpy())
            np.save(self.output_dir / "radial_noise.npy", self.noise.cpu().numpy())
        train_loader = self._loader("train", training=True)
        validation_loader = self._loader("validation", training=False)
        best = float("inf")
        for epoch in range(1, self.config.epochs + 1):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            train = self._run(train_loader, training=True, epoch=epoch)
            validation = self._run(validation_loader, training=False, epoch=epoch)
            self.scheduler.step()
            if self.distributed.is_main:
                improved = validation["mse"] < best
                best = min(best, validation["mse"])
                state = {
                    "model": (self.model.module if self.distributed.enabled else self.model).state_dict(),
                    "radial_scale": self.scale.cpu(),
                    "radial_noise": self.noise.cpu(),
                    "config": asdict(self.config),
                    "epoch": epoch,
                    "validation": validation,
                }
                torch.save(state, self.output_dir / "last.pt")
                if improved:
                    torch.save(state, self.output_dir / "best.pt")
                print(
                    f"epoch={epoch} train_mse={train['mse']:.5g} "
                    f"validation_mse={validation['mse']:.5g} "
                    f"validation_identity_mse={validation['identity_mse']:.5g} "
                    f"validation_noise_removed={validation['noise_removed_fraction']:.4f} "
                    f"best={improved}",
                    flush=True,
                )
            self.distributed.barrier()


class Denoiser:
    """Load a checkpoint and denoise single full frames in physical pB."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = DenoiseConfig(**state["config"])
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = build_model(self.config).to(self.device).eval()
        self.model.load_state_dict(state["model"])
        self.scale = state["radial_scale"].to(self.device)
        noise = state.get("radial_noise")
        self.noise = None if noise is None else noise.numpy()

    @torch.no_grad()
    def __call__(self, image: np.ndarray, valid: np.ndarray, radius: np.ndarray) -> np.ndarray:
        image_t, valid_t, radius_t = (
            torch.from_numpy(np.ascontiguousarray(array))[None, None].to(self.device)
            for array in (image, valid, radius)
        )
        scale = scale_at(self.scale, radius_t.float())
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            output = self.model(image_t.float() / scale, valid_t, radius_t)
        clean = (output.float() * scale)[0, 0].cpu().numpy()
        return np.where(valid, clean, 0.0).astype(np.float32)

    def denoise_file(self, path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, valid, radius = read_fits_image(path)
        return image, self(image, valid, radius), valid
