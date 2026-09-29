"""Data discovery, pairing, datasets, masking, and loader construction.

The record types stay importable without the optional ML dependencies. Dataset
and loader classes are imported lazily when PyTorch is available.
"""

from .records import FitsRecord, FramePair, PairManifest

__all__ = [
    "DataBuilder",
    "FitsRecord",
    "FramePair",
    "KCorPairDataset",
    "LoaderBundle",
    "PairManifest",
]


def __getattr__(name: str):
    if name in {"DataBuilder", "LoaderBundle"}:
        from .builder import DataBuilder, LoaderBundle

        return {"DataBuilder": DataBuilder, "LoaderBundle": LoaderBundle}[name]
    if name == "KCorPairDataset":
        from .dataset import KCorPairDataset

        return KCorPairDataset
    raise AttributeError(name)
