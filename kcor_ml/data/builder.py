"""Dataset and DataLoader construction, separated from model construction."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from ..config import DataConfig
from .dataset import KCorPairDataset
from .records import PairManifest


@dataclass(slots=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


class DataBuilder:
    def __init__(self, config: DataConfig):
        self.config = config

    def build_dataset(self, manifest: PairManifest, split: str) -> KCorPairDataset:
        if split not in manifest.splits:
            raise KeyError(f"unknown split: {split}")
        return KCorPairDataset(
            manifest.splits[split],
            crop_size=self.config.crop_size,
            training=split == "train",
        )

    def build_loader(self, dataset: KCorPairDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )

    def build(self, manifest: PairManifest) -> LoaderBundle:
        return LoaderBundle(
            train=self.build_loader(self.build_dataset(manifest, "train"), shuffle=True),
            validation=self.build_loader(self.build_dataset(manifest, "validation"), shuffle=False),
            test=self.build_loader(self.build_dataset(manifest, "test"), shuffle=False),
        )
