"""Command-line entry point for indexing, training, evaluation, and inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from kcor_ml.config import ExperimentConfig
from kcor_ml.data.builder import DataBuilder
from kcor_ml.data.discovery import create_manifest
from kcor_ml.data.records import PairManifest
from kcor_ml.inference import predict_pair, write_fits_product
from kcor_ml.model_builder import ModelBuilder
from kcor_ml.training.checkpoint import load_checkpoint
from kcor_ml.training.trainer import Trainer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--index", action="store_true", help="discover pb2 FITS files and build pairs")
    modes.add_argument("--train", action="store_true", help="train the configured model")
    modes.add_argument("--evaluate", action="store_true", help="evaluate a checkpoint on the test split")
    modes.add_argument("--predict", action="store_true", help="write one inference FITS product")
    parser.add_argument("--checkpoint", help="checkpoint for evaluation or inference")
    parser.add_argument("--pair-index", type=int, default=0, help="manifest test-pair index for inference")
    parser.add_argument("--output", help="inference FITS destination")
    return parser.parse_args()


def build_manifest(config: ExperimentConfig) -> PairManifest:
    manifest = create_manifest(
        config.data.root,
        expected_product=config.data.expected_product,
        maximum_delta_seconds=config.data.max_delta_seconds,
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
        seed=config.data.seed,
    )
    manifest.save(config.data.manifest)
    counts = {split: len(pairs) for split, pairs in manifest.splits.items()}
    print(f"wrote {config.data.manifest}: {counts}")
    return manifest


def require_checkpoint(arguments: argparse.Namespace) -> str:
    if not arguments.checkpoint:
        raise SystemExit("--checkpoint is required for this mode")
    return arguments.checkpoint


def main() -> None:
    arguments = parse_arguments()
    config = ExperimentConfig.load(arguments.config)
    if arguments.index:
        build_manifest(config)
        return
    manifest = PairManifest.load(config.data.manifest)
    loaders = DataBuilder(config.data).build(manifest)
    components = ModelBuilder(config).build()
    if arguments.train:
        Trainer(config, components).fit(loaders.train, loaders.validation)
        return
    checkpoint = require_checkpoint(arguments)
    load_checkpoint(checkpoint, model=components.model, map_location=components.device)
    trainer = Trainer(config, components)
    if arguments.evaluate:
        print(trainer.run_epoch(loaders.test, training=False))
        return
    dataset = loaders.test.dataset
    sample = dataset[arguments.pair_index]
    images = sample["images"].unsqueeze(0).to(components.device)
    valid = sample["valid_mask"].unsqueeze(0).to(components.device)
    delta_t = sample["delta_t"].unsqueeze(0).to(components.device)
    output = predict_pair(components.model, images, valid, delta_t)
    destination = arguments.output or f"artifacts/prediction_{arguments.pair_index:06d}.fits"
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    write_fits_product(
        destination,
        output,
        valid,
        source_paths=sample["paths"],
        checkpoint=checkpoint,
    )
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
