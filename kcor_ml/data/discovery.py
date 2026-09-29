"""Discover pb2 FITS files and create leakage-safe temporal pairs."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Iterable

from .records import FitsRecord, FramePair, PairManifest


PRODUCT_KEYS = ("PRODUCT", "PRODTYPE", "DATA_PROD", "DATAPROD")
TIME_KEYS = ("DATE-OBS", "DATE_OBS", "DATEOBS")


def _header_value(header, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = header.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _parse_time(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def discover_fits(root: str | Path, expected_product: str = "pb2") -> list[FitsRecord]:
    """Read FITS headers and return chronologically ordered pb2 records."""
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("FITS discovery requires astropy; install requirements-ml.txt") from exc

    records: list[FitsRecord] = []
    patterns = ("*.fits", "*.fit", "*.fts", "*.fits.gz", "*.fit.gz", "*.fts.gz")
    paths = sorted({path for pattern in patterns for path in Path(root).rglob(pattern)})
    for path in paths:
        header = fits.getheader(path, memmap=True)
        time_value = _header_value(header, TIME_KEYS)
        if time_value is None:
            raise ValueError(f"missing observation timestamp in {path}")
        product = _header_value(header, PRODUCT_KEYS) or expected_product
        if product.casefold() != expected_product.casefold():
            continue
        height = int(header.get("NAXIS2", 0))
        width = int(header.get("NAXIS1", 0))
        if height < 1 or width < 1:
            raise ValueError(f"invalid FITS image shape in {path}")
        records.append(FitsRecord(str(path.resolve()), _parse_time(time_value), product, (height, width)))
    return sorted(records, key=lambda record: (record.observed_at, record.path))


def build_adjacent_pairs(
    records: Iterable[FitsRecord], maximum_delta_seconds: float = 15.0
) -> list[FramePair]:
    """Pair adjacent observations without creating cross-date samples."""
    ordered = sorted(records, key=lambda record: (record.observed_at, record.path))
    pairs: list[FramePair] = []
    for first, second in zip(ordered, ordered[1:]):
        if first.observing_date != second.observing_date:
            continue
        if first.product.casefold() != second.product.casefold() or first.shape != second.shape:
            continue
        pair = FramePair(first, second)
        if 0 < pair.delta_seconds <= maximum_delta_seconds:
            pairs.append(pair)
    return pairs


def split_pairs_by_date(
    pairs: Iterable[FramePair],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[FramePair]]:
    """Assign complete dates deterministically to train, validation, or test."""
    by_date: dict[str, list[FramePair]] = defaultdict(list)
    for pair in pairs:
        by_date[pair.observing_date].append(pair)
    ranked_dates = sorted(
        by_date,
        key=lambda date: hashlib.sha256(f"{seed}:{date}".encode()).hexdigest(),
    )
    count = len(ranked_dates)
    if count < 3:
        raise ValueError("at least three observing dates are required for isolated train/validation/test splits")
    train_count = min(max(1, int(count * train_fraction)), count - 2)
    validation_count = min(max(1, int(count * validation_fraction)), count - train_count - 1)
    train_end = train_count
    validation_end = train_end + validation_count
    date_splits = {
        "train": set(ranked_dates[:train_end]),
        "validation": set(ranked_dates[train_end:validation_end]),
        "test": set(ranked_dates[validation_end:]),
    }
    return {
        split: [pair for date in sorted(dates) for pair in by_date[date]]
        for split, dates in date_splits.items()
    }


def create_manifest(
    root: str | Path,
    *,
    expected_product: str,
    maximum_delta_seconds: float,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> PairManifest:
    records = discover_fits(root, expected_product)
    pairs = build_adjacent_pairs(records, maximum_delta_seconds)
    return PairManifest(
        product=expected_product,
        maximum_delta_seconds=maximum_delta_seconds,
        splits=split_pairs_by_date(pairs, train_fraction, validation_fraction, seed),
    )
