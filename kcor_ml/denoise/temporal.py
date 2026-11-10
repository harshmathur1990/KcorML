"""Multi-frame denoising at the native 15 s cadence.

For every frame t, the network sees its temporal neighbours t-k..t-1 and
t+1..t+k (never frame t itself) and is trained to predict frame t. The noise
in frame t is independent of the neighbours, so the optimum is the noise-free
frame t: one clean output per raw frame, with no loss of cadence. Spatially
correlated noise that a single frame cannot separate from the corona averages
out across the neighbours.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ..data.dataset import crop_origin
from ..data.records import PairManifest
from ..distributed import DistributedContext
from .cache import FrameCache
from .model import DenoisingUNet
from .profile import radial_noise, radial_scale, scale_at
from .train import _augment


@dataclass(slots=True)
class TemporalConfig:
    cache_dir: str = "artifacts/cache"
    manifest: str = "artifacts/pb2_pairs.json"
    output_dir: str = "artifacts/denoise_temporal"
    neighbours: int = 4
    maximum_step_seconds: float = 20.0
    crop_size: int = 256
    batch_size: int = 8
    num_workers: int = 6
    epochs: int = 20
    learning_rate: float = 2.0e-4
    minimum_learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    blocks_per_level: int = 2
    profile_frames: int = 100
    # Probability that a training sample receives synthetic moving features.
    moving_feature_probability: float = 0.0
    maximum_feature_speed: float = 6.0  # pixels per 15 s frame (~1600 km/s)
    seed: int = 17
    device: str = "cuda"
    log_every_steps: int = 50

    @classmethod
    def load(cls, path: str | Path) -> "TemporalConfig":
        return cls(**json.loads(Path(path).read_text()))


def split_days(manifest_path: str) -> dict[str, set[str]]:
    manifest = PairManifest.load(manifest_path)
    return {
        split: {Path(pair.first.path).name[:8] for pair in pairs} for split, pairs in manifest.splits.items()
    }


def neighbour_indices(
    cache: FrameCache, center: int, neighbours: int, maximum_step_seconds: float, same_day: bool = True
) -> list[int | None]:
    """Indices of frames at offsets -k..-1, 1..k from ``center``; None when the
    frame is missing, further than offset * max step in time, or (with
    ``same_day``, which keeps date-based splits apart) on another UTC date."""
    day = cache.day(center)
    result: list[int | None] = []
    for offset in [*range(-neighbours, 0), *range(1, neighbours + 1)]:
        index = center + offset
        usable = (
            0 <= index < len(cache.paths)
            and (not same_day or cache.day(index) == day)
            and abs(cache.times[index] - cache.times[center]) <= abs(offset) * maximum_step_seconds
        )
        result.append(index if usable else None)
    return result


def add_moving_features(
    stack: np.ndarray,
    target: np.ndarray,
    offsets: list[float | None],
    noise: np.ndarray,
    rng: np.random.Generator,
    maximum_speed: float,
) -> None:
    """Add 1-3 faint anisotropic Gaussian features moving at constant velocity.

    The quasi-static corona gives the network little reason to follow motion,
    so a plain average of the neighbours is nearly optimal on real data alone.
    Each feature is drawn at the position its motion implies in every frame,
    including the target, so a model that averages over the motion is
    penalized and one that tracks it is not. ``offsets`` are frame offsets
    from the target in units of 15 s (None for missing neighbours); ``noise``
    is the per-pixel noise level of the crop.
    """
    size = target.shape[-1]
    y, x = np.indices((size, size), dtype=np.float32)
    for _ in range(rng.integers(1, 4)):
        center = rng.uniform(0, size, 2)
        speed = rng.uniform(0.0, maximum_speed)
        heading = rng.uniform(0, 2 * np.pi)
        velocity = speed * np.array([np.sin(heading), np.cos(heading)])
        sigma_along, sigma_across = rng.uniform(3.0, 12.0, 2)
        angle = rng.uniform(0, np.pi)
        amplitude = rng.uniform(0.5, 4.0) * noise[int(center[0]), int(center[1])]

        def feature(offset: float) -> np.ndarray:
            cy, cx = center + velocity * offset
            u = (y - cy) * np.cos(angle) + (x - cx) * np.sin(angle)
            v = -(y - cy) * np.sin(angle) + (x - cx) * np.cos(angle)
            return amplitude * np.exp(-0.5 * ((u / sigma_along) ** 2 + (v / sigma_across) ** 2))

        target[0] += feature(0.0)
        for slot, offset in enumerate(offsets):
            if offset is not None:
                stack[slot] += feature(offset)


class TemporalDataset(Dataset):
    """Return (neighbour stack, validity, target frame, target validity, radius) crops."""

    def __init__(
        self,
        cache: FrameCache,
        centers: list[int],
        config: TemporalConfig,
        training: bool,
        noise_profile: np.ndarray | None = None,
    ):
        self.cache = cache
        self.centers = centers
        self.config = config
        self.training = training
        self.noise_profile = noise_profile
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        center = self.centers[item]
        height, width = self.cache.radius.shape
        size = self.config.crop_size or height
        top, left = crop_origin(height, width, size, self.training, item) if size < height else (0, 0)
        window = (slice(top, top + size), slice(left, left + size))
        # NaN marks invalid pixels and missing neighbours alike.
        stack = np.full((2 * self.config.neighbours, size, size), np.nan, dtype=np.float32)
        found = neighbour_indices(self.cache, center, self.config.neighbours, self.config.maximum_step_seconds)
        for slot, index in enumerate(found):
            if index is not None:
                stack[slot] = self.cache.frames[index][window]
        target = np.array(self.cache.frames[center][window], dtype=np.float32)[None]
        if (
            self.training
            and self.noise_profile is not None
            and self.rng.random() < self.config.moving_feature_probability
        ):
            bins = np.clip((self.cache.radius[window] * len(self.noise_profile)).astype(int), 0, len(self.noise_profile) - 1)
            offsets = [
                None if index is None else (self.cache.times[index] - self.cache.times[center]) / 15.0
                for index in found
            ]
            add_moving_features(
                stack, target, offsets, self.noise_profile[bins], self.rng, self.config.maximum_feature_speed
            )
        valid = np.isfinite(stack)
        target_valid = np.isfinite(target)
        return {
            "images": torch.from_numpy(np.nan_to_num(stack, nan=0.0)),
            "valid": torch.from_numpy(valid),
            "target": torch.from_numpy(np.nan_to_num(target, nan=0.0)),
            "target_valid": torch.from_numpy(target_valid),
            "radius": torch.from_numpy(np.ascontiguousarray(self.cache.radius[window][None])),
        }


def training_centers(cache: FrameCache, days: set[str], config: TemporalConfig) -> list[int]:
    """Frames on ``days`` with at least one usable neighbour on each side."""
    centers = []
    for center in range(len(cache.paths)):
        if cache.day(center) not in days:
            continue
        found = neighbour_indices(cache, center, config.neighbours, config.maximum_step_seconds)
        before, after = found[: config.neighbours], found[config.neighbours :]
        if any(index is not None for index in before) and any(index is not None for index in after):
            centers.append(center)
    return centers


def build_model(config: TemporalConfig) -> DenoisingUNet:
    return DenoisingUNet(tuple(config.channels), config.blocks_per_level, input_frames=2 * config.neighbours)


def temporal_views(batch: dict[str, torch.Tensor], scale: torch.Tensor, noise: torch.Tensor, device, augment: bool):
    """Normalize a batch and return (inputs, valid, target, loss_weight, radius)."""
    images = batch["images"].to(device, non_blocking=True).float()
    valid = batch["valid"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True).float()
    target_valid = batch["target_valid"].to(device, non_blocking=True)
    radius = batch["radius"].to(device, non_blocking=True).float()
    pixel_scale = scale_at(scale, radius)
    images, target = images / pixel_scale, target / pixel_scale
    # Loss in units of the local noise level (see profile.py).
    weight = (pixel_scale / scale_at(noise, radius)).square()
    weight = weight * (target_valid & valid.any(dim=1, keepdim=True)).float()
    if augment:
        images, valid, target, weight, radius = _augment(images, valid, target, weight, radius)
    return images, valid, target, weight, radius


class TemporalTrainer:
    def __init__(self, config: TemporalConfig, distributed: DistributedContext):
        self.config = config
        self.distributed = distributed
        self.device = distributed.device
        self.output_dir = Path(config.output_dir)
        self.cache = FrameCache.open(config.cache_dir)
        self.days = split_days(config.manifest)
        self.scale, self.noise = self._radial_profiles()
        model = build_model(config).to(self.device)
        self.model = (
            DistributedDataParallel(model, device_ids=[distributed.local_rank]) if distributed.enabled else model
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
        # Training days only, so the normalization never sees test dates.
        pairs = PairManifest.load(self.config.manifest).splits["train"]
        step = max(1, len(pairs) // self.config.profile_frames)
        selected = pairs[::step][: self.config.profile_frames]
        scale = radial_scale([pair.first.path for pair in selected])
        noise = radial_noise([(pair.first.path, pair.second.path) for pair in selected])
        return torch.from_numpy(scale).to(self.device), torch.from_numpy(noise).to(self.device)

    def _loader(self, split: str, training: bool) -> DataLoader:
        dataset = TemporalDataset(
            self.cache,
            training_centers(self.cache, self.days[split], self.config),
            self.config,
            training,
            noise_profile=self.noise.cpu().numpy(),
        )
        sampler = (
            DistributedSampler(dataset, shuffle=training, seed=self.config.seed) if self.distributed.enabled else None
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
        # Weighted squared-error sums for the model and the plain neighbour mean, and pixel count.
        totals = torch.zeros(3, device=self.device, dtype=torch.float64)
        context = torch.enable_grad if training else torch.no_grad
        with context():
            for step, batch in enumerate(loader, start=1):
                images, valid, target, weight, radius = temporal_views(
                    batch, self.scale, self.noise, self.device, augment=training
                )
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    prediction = self.model(images, valid, radius)
                squared = (prediction.float() - target).square() * weight
                count = (weight > 0).sum().float().clamp_min(1.0)
                loss = squared.sum() / count
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step}")
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                valid_float = valid.float()
                mean = (images * valid_float).sum(1, keepdim=True) / valid_float.sum(1, keepdim=True).clamp_min(1.0)
                averaged = ((mean - target).square() * weight).sum()
                totals += torch.stack((squared.sum().detach().double(), averaged.detach().double(), count.double()))
                if training and self.distributed.is_main and step % self.config.log_every_steps == 0:
                    print(f"epoch={epoch} step={step}/{len(loader)} loss={loss.item():.5g}", flush=True)
        if self.distributed.enabled:
            torch.distributed.all_reduce(totals)
        model_mse, mean_mse = (totals[:2] / totals[2].clamp_min(1.0)).tolist()
        # In noise units a perfect prediction of frame t scores ~1 (frame t's own noise).
        return {"mse": model_mse, "neighbour_mean_mse": mean_mse}

    def fit(self) -> None:
        if self.distributed.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.output_dir / "radial_scale.npy", self.scale.cpu().numpy())
            np.save(self.output_dir / "radial_noise.npy", self.noise.cpu().numpy())
        train_loader = self._loader("train", training=True)
        validation_loader = self._loader("validation", training=False)
        if self.distributed.is_main:
            print(
                f"centers: train={len(train_loader.dataset)} validation={len(validation_loader.dataset)}",
                flush=True,
            )
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
                    f"epoch={epoch} train_mse={train['mse']:.5g} validation_mse={validation['mse']:.5g} "
                    f"validation_neighbour_mean_mse={validation['neighbour_mean_mse']:.5g} best={improved}",
                    flush=True,
                )
            self.distributed.barrier()


class TemporalDenoiser:
    """Denoise full frames from the cache (or from injected arrays) in physical pB."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda", cache_dir: str | Path | None = None):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = TemporalConfig(**state["config"])
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.config).to(self.device).eval()
        self.model.load_state_dict(state["model"])
        # The radial profiles stay those of the training data, also for new data.
        self.scale = state["radial_scale"].to(self.device)
        self.noise = state["radial_noise"].numpy()
        self.cache = FrameCache.open(cache_dir or self.config.cache_dir)
        self.radius = torch.from_numpy(self.cache.radius)[None, None].to(self.device)

    def neighbours(self, center: int) -> list[int | None]:
        # K-Cor's observing day (~17-03 UT) spans midnight UTC; at inference,
        # neighbours on the next UTC date are as valid as any other.
        return neighbour_indices(
            self.cache, center, self.config.neighbours, self.config.maximum_step_seconds, same_day=False
        )

    @torch.no_grad()
    def __call__(self, stack: np.ndarray) -> np.ndarray:
        """``stack`` is [2k, H, W] physical pB of the neighbours, NaN where invalid
        or missing; returns the denoised centre frame (NaN outside the science field)."""
        images = torch.from_numpy(np.nan_to_num(stack, nan=0.0).astype(np.float32))[None].to(self.device)
        valid = torch.from_numpy(np.isfinite(stack))[None].to(self.device)
        scale = scale_at(self.scale, self.radius)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            output = self.model(images / scale, valid, self.radius)
        clean = (output.float() * scale)[0, 0].cpu().numpy()
        science = np.isfinite(stack).any(axis=0)
        return np.where(science, clean, np.nan).astype(np.float32)

    def stack(self, center: int) -> np.ndarray:
        height, width = self.cache.radius.shape
        stack = np.full((2 * self.config.neighbours, height, width), np.nan, dtype=np.float32)
        for slot, index in enumerate(self.neighbours(center)):
            if index is not None:
                stack[slot] = self.cache.frames[index]
        return stack

    def denoise(self, center: int) -> np.ndarray:
        clean = self(self.stack(center))
        return np.where(np.isfinite(self.cache.frames[center]), clean, np.nan)
