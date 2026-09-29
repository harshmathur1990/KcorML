"""Typed project configuration.

The layout follows the configuration-first workflow used by FFNOML while
keeping runtime state out of module-level globals.  JSON is used for external
configuration so the command-line tools need no configuration dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataConfig:
    root: str = "data/pb2"
    manifest: str = "artifacts/pb2_pairs.json"
    expected_product: str = "pb2"
    max_delta_seconds: float = 15.0
    crop_size: int | None = 256
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    seed: int = 17

    def validate(self) -> None:
        if self.max_delta_seconds <= 0:
            raise ValueError("max_delta_seconds must be positive")
        if self.crop_size is not None and self.crop_size < 32:
            raise ValueError("crop_size must be at least 32 or null")
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("invalid loader configuration")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between zero and one")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test split")


@dataclass(slots=True)
class ModelConfig:
    name: str = "tf_hdn_v2"
    channels: tuple[int, ...] = (32, 64, 128, 256)
    blocks_per_level: int = 2
    group_norm_groups: int = 8
    maximum_offset_pixels: float = 4.0
    input_scale: float = 1.0e-8
    minimum_noise_scale: float = 1.0e-10
    positional_grid_size: int = 32
    time_scale_seconds: float = 15.0
    log_component_limit: float = 12.0

    def validate(self) -> None:
        if self.name != "tf_hdn_v2":
            raise ValueError(f"unsupported model: {self.name}")
        if len(self.channels) < 2 or any(value < 1 for value in self.channels):
            raise ValueError("channels must contain at least two positive values")
        if self.blocks_per_level < 1:
            raise ValueError("blocks_per_level must be positive")
        if self.input_scale <= 0 or self.minimum_noise_scale <= 0 or self.time_scale_seconds <= 0:
            raise ValueError("physical scales must be positive")
        if self.log_component_limit <= 0:
            raise ValueError("log_component_limit must be positive")


@dataclass(slots=True)
class LossConfig:
    observation_weight: float = 1.0
    normalization_bandwidth_weight: float = 1.0e-3
    noise_independence_weight: float = 1.0e-3
    common_consistency_weight: float = 1.0e-2
    radial_flatness_weight: float = 5.0e-2
    gauge_weight: float = 1.0e-2
    radial_bins: int = 32
    cme_weight: float = 0.0

    def validate(self) -> None:
        weights = (
            self.observation_weight,
            self.normalization_bandwidth_weight,
            self.noise_independence_weight,
            self.common_consistency_weight,
            self.radial_flatness_weight,
            self.gauge_weight,
            self.cme_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("loss weights must be non-negative")
        if self.radial_bins < 4:
            raise ValueError("radial_bins must be at least four")


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    minimum_learning_rate: float = 1.0e-6
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    mask_fraction: float = 0.05
    mask_patch_size: int = 8
    frame_drop_probability: float = 0.05
    device: str = "cuda"
    output_dir: str = "artifacts/training"
    resume_checkpoint: str | None = None
    log_every_steps: int = 25
    progress_bar: bool = True
    minimum_normalization_log_std: float = 1.0e-3
    maximum_log_saturation_fraction: float = 1.0e-3

    def validate(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if not 0 < self.mask_fraction < 1:
            raise ValueError("mask_fraction must be between zero and one")
        if self.mask_patch_size < 1:
            raise ValueError("mask_patch_size must be positive")
        if not 0 <= self.frame_drop_probability < 1:
            raise ValueError("frame_drop_probability must be in [0, 1)")
        if self.log_every_steps < 1:
            raise ValueError("log_every_steps must be positive")
        if self.minimum_normalization_log_std < 0:
            raise ValueError("minimum_normalization_log_std must be non-negative")
        if not 0 <= self.maximum_log_saturation_fraction <= 1:
            raise ValueError("maximum_log_saturation_fraction must be in [0, 1]")


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.loss.validate()
        self.train.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text())
        model_payload = payload.get("model", {})
        if "channels" in model_payload:
            model_payload["channels"] = tuple(model_payload["channels"])
        config = cls(
            data=DataConfig(**payload.get("data", {})),
            model=ModelConfig(**model_payload),
            loss=LossConfig(**payload.get("loss", {})),
            train=TrainConfig(**payload.get("train", {})),
        )
        config.validate()
        return config
