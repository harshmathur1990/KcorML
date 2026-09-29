from datetime import datetime, timedelta
import unittest

from kcor_ml.data.discovery import build_adjacent_pairs, split_pairs_by_date
from kcor_ml.data.records import FitsRecord, FramePair


def record(name, time, product="pb2", shape=(64, 64)):
    return FitsRecord(name, time, product, shape)


class PairingTests(unittest.TestCase):
    def test_pairs_only_adjacent_matching_pb2_frames(self):
        start = datetime(2024, 1, 1, 12)
        records = [
            record("a.fits", start),
            record("b.fits", start + timedelta(seconds=15)),
            record("c.fits", start + timedelta(seconds=31)),
        ]
        pairs = build_adjacent_pairs(records, 15)
        self.assertEqual([(p.first.path, p.second.path) for p in pairs], [("a.fits", "b.fits")])

    def test_date_split_has_no_date_leakage(self):
        pairs = []
        for day in range(1, 11):
            start = datetime(2024, 1, day, 12)
            pairs.append(FramePair(record(f"{day}a", start), record(f"{day}b", start + timedelta(seconds=15))))
        splits = split_pairs_by_date(pairs, 0.7, 0.2, seed=3)
        dates = [{pair.observing_date for pair in split} for split in splits.values()]
        self.assertFalse(dates[0] & dates[1])
        self.assertFalse(dates[0] & dates[2])
        self.assertFalse(dates[1] & dates[2])

if __name__ == "__main__":
    unittest.main()
