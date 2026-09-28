"""Complete MLSO searches using bounded, adaptively subdivided UTC intervals."""
from calendar import monthrange
from datetime import datetime, timedelta
import re

from mlso.api import client

BASE_URL = 'https://api.mlso.ucar.edu'
# Observed service limit. A full response is treated as potentially truncated.
RESULT_LIMIT = 3000


def cadence_bucket(observed, start, cadence):
    """Global start-anchored buckets, independent of API request boundaries."""
    match = re.fullmatch(r'(\d+)(second|minute|hour|day|week|month|quarter|year)s?', cadence)
    if not match or int(match[1]) < 1:
        raise ValueError(f'Invalid cadence: {cadence}')
    count, unit = int(match[1]), match[2]
    seconds = dict(second=1, minute=60, hour=3600, day=86400, week=604800)
    if unit in seconds:
        return int((observed - start).total_seconds() // (count * seconds[unit]))
    months = count * dict(month=1, quarter=3, year=12)[unit]
    index = ((observed.year-start.year)*12 + observed.month-start.month) // months
    absolute = start.year*12 + start.month-1 + index*months
    year, month = divmod(absolute, 12)
    boundary = start.replace(year=year, month=month+1,
                             day=min(start.day, monthrange(year, month+1)[1]))
    return index - (observed < boundary)


def search_all(instrument, product, filters, *, cancelled=lambda: False,
               progress=lambda done, total, count, message: None, limit=RESULT_LIMIT):
    """Return complete, deduplicated results or raise; cancellation returns None.

    Request each UTC day, recursively split any response reaching the cap,
    and overlap boundaries to avoid dropping boundary timestamps. Never accept
    a saturated minimum-width interval as a complete result.
    """
    start = datetime.fromisoformat(filters['start-date'])
    end = datetime.fromisoformat(filters['end-date'])
    if start > end:
        raise ValueError('Start time must be before end time.')
    cadence = filters.get('every')
    base_filters = {k: v for k, v in filters.items() if k not in ('start-date', 'end-date', 'every')}
    found = {}
    requests_made = 0
    total = (end.date() - start.date()).days + 1

    def fetch(left, right, day_index):
        nonlocal requests_made
        if cancelled():
            return False
        query = dict(base_filters, **{'start-date': left.isoformat(timespec='seconds'),
                                    'end-date': right.isoformat(timespec='seconds')})
        requests_made += 1
        progress(day_index, total, len(found), f'Query {requests_made}: {query["start-date"]} → {query["end-date"]}')
        try:
            response = client.files(instrument, product, query, base_url=BASE_URL)
            records = response['files']
            if not isinstance(records, list):
                raise ValueError('API returned an invalid file list.')
        except Exception as exc:
            raise RuntimeError(f'Search incomplete at {left.isoformat()} → {right.isoformat()}: {exc}. Retry the search.') from exc
        if cancelled():
            return False
        if len(records) >= limit:
            seconds = int((right-left).total_seconds())
            if seconds <= 1:
                raise RuntimeError(f'Search incomplete: API result limit reached within {left} → {right}; cannot safely subdivide further.')
            middle = left + timedelta(seconds=seconds//2)
            return fetch(left, middle, day_index) and fetch(middle, right, day_index)
        for record in records:
            # Identity must not depend on download URL query parameters.
            key = (record.get('instrument', instrument), record.get('product', product), record['filename'])
            found[key] = record
        return True

    left = start
    for day_index in range(total):
        midnight = datetime.combine(left.date()+timedelta(days=1), datetime.min.time())
        right = min(midnight, end)
        if not fetch(left, right, day_index):
            return None
        progress(day_index+1, total, len(found), f'Completed {left.date()} • {len(found):,} unique files')
        left = midnight
    if cancelled():
        return None
    files = sorted(found.values(), key=lambda f: (f.get('date-obs', ''), f['filename']))
    if cadence:
        buckets = set()
        sampled = []
        for record in files:
            observed = datetime.fromisoformat(record['date-obs'])
            key = (record.get('instrument', instrument), record.get('product', product),
                   record.get('wave-region'), cadence_bucket(observed, start, cadence))
            if key not in buckets:
                sampled.append(record)
                buckets.add(key)
        files = sampled
    return {'files': files, 'requests': requests_made}
