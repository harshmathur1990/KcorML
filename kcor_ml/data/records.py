"""Serializable records used by discovery and datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FitsRecord:
    path: str
    observed_at: datetime
    product: str
    shape: tuple[int, int]

    @property
    def observing_date(self) -> str:
        return self.observed_at.date().isoformat()


@dataclass(frozen=True, slots=True)
class FramePair:
    first: FitsRecord
    second: FitsRecord

    @property
    def delta_seconds(self) -> float:
        return (self.second.observed_at - self.first.observed_at).total_seconds()

    @property
    def observing_date(self) -> str:
        return self.first.observing_date


@dataclass(slots=True)
class PairManifest:
    product: str
    maximum_delta_seconds: float
    splits: dict[str, list[FramePair]]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        def encode_record(record: FitsRecord) -> dict[str, object]:
            value = asdict(record)
            value["observed_at"] = record.observed_at.isoformat()
            value["shape"] = list(record.shape)
            return value

        payload = {
            "product": self.product,
            "maximum_delta_seconds": self.maximum_delta_seconds,
            "splits": {
                split: [
                    {"first": encode_record(pair.first), "second": encode_record(pair.second)}
                    for pair in pairs
                ]
                for split, pairs in self.splits.items()
            },
        }
        destination.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "PairManifest":
        payload = json.loads(Path(path).read_text())

        def decode_record(value: dict[str, object]) -> FitsRecord:
            return FitsRecord(
                path=str(value["path"]),
                observed_at=datetime.fromisoformat(str(value["observed_at"])),
                product=str(value["product"]),
                shape=tuple(int(item) for item in value["shape"]),
            )

        splits = {
            split: [FramePair(decode_record(item["first"]), decode_record(item["second"])) for item in pairs]
            for split, pairs in payload["splits"].items()
        }
        return cls(
            product=str(payload["product"]),
            maximum_delta_seconds=float(payload["maximum_delta_seconds"]),
            splits=splits,
        )

