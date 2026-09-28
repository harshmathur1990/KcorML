from datetime import datetime, timedelta
import unittest
from unittest.mock import patch
from mlso_search import search_all, cadence_bucket


def record(when, name=None):
    return {'date-obs': when.isoformat(), 'filename': name or when.isoformat()+'.fts',
            'instrument': 'kcor', 'product': 'pb'}


class SearchTests(unittest.TestCase):
    def run_search(self, start, end, **kwargs):
        return search_all('kcor', 'pb', {'start-date': start, 'end-date': end}, **kwargs)

    def test_four_years_include_every_day_and_leap_day(self):
        calls = []
        def api(instrument, product, filters, **kwargs):
            left = datetime.fromisoformat(filters['start-date'])
            right = datetime.fromisoformat(filters['end-date'])
            self.assertLessEqual(right-left, timedelta(days=1))
            calls.append(left.date().isoformat())
            return {'files': [record(left)]}
        with patch('mlso_search.client.files', side_effect=api):
            result = self.run_search('2019-01-01T00:00:00', '2022-12-31T23:59:59')
        self.assertEqual(len(result['files']), 1461)
        self.assertEqual(len(calls), 1461)
        self.assertIn('2020-02-29', calls)
        self.assertEqual(calls[-1], '2022-12-31')

    def test_saturated_day_splits_and_deduplicates(self):
        start = datetime(2020, 1, 1)
        source = [record(start+timedelta(seconds=i*20)) for i in range(4001)]
        def api(instrument, product, filters, **kwargs):
            left, right = (datetime.fromisoformat(filters[k]) for k in ('start-date', 'end-date'))
            return {'files': [r for r in source if left <= datetime.fromisoformat(r['date-obs']) <= right][:3000]}
        with patch('mlso_search.client.files', side_effect=api):
            result = self.run_search('2020-01-01T00:00:00', '2020-01-01T23:59:59')
        self.assertEqual(result['files'], source)
        self.assertGreater(result['requests'], 1)

    def test_midnight_overlap_and_partial_days(self):
        source = [record(datetime(2020, 1, 1, 23, 59, 59)),
                  record(datetime(2020, 1, 2)), record(datetime(2020, 1, 2, 0, 0, 1))]
        def api(instrument, product, filters, **kwargs):
            left, right = (datetime.fromisoformat(filters[k]) for k in ('start-date', 'end-date'))
            return {'files': [r for r in source if left <= datetime.fromisoformat(r['date-obs']) <= right]}
        with patch('mlso_search.client.files', side_effect=api):
            result = self.run_search('2020-01-01T23:59:59', '2020-01-02T00:00:01')
        self.assertEqual(result['files'], source)

    def test_unresolvable_cap_fails_explicitly(self):
        with patch('mlso_search.client.files', return_value={'files': [record(datetime(2020, 1, 1))]*3}):
            with self.assertRaisesRegex(RuntimeError, 'cannot safely subdivide'):
                self.run_search('2020-01-01T00:00:00', '2020-01-01T00:00:01', limit=3)

    def test_error_does_not_return_partial_results(self):
        with patch('mlso_search.client.files', side_effect=[{'files': [record(datetime(2020, 1, 1))]}, OSError('offline')]):
            with self.assertRaisesRegex(RuntimeError, 'Search incomplete at 2020-01-02'):
                self.run_search('2020-01-01T00:00:00', '2020-01-02T23:59:59')

    def test_cancel_stops_further_calls(self):
        cancelled = False
        def api(*args, **kwargs):
            nonlocal cancelled
            cancelled = True
            return {'files': []}
        with patch('mlso_search.client.files', side_effect=api) as request:
            self.assertIsNone(self.run_search('2020-01-01', '2020-01-03', cancelled=lambda: cancelled))
            self.assertEqual(request.call_count, 1)

    def test_cadence_is_global_and_filters_preserved(self):
        filters = {'start-date': '2020-01-01T00:00:00', 'end-date': '2020-01-04T23:59:59',
                   'every': '2days', 'cr': '2225', 'wave-region': '1074'}
        def api(instrument, product, query, **kwargs):
            self.assertNotIn('every', query)
            self.assertEqual(query['cr'], '2225')
            self.assertEqual(query['wave-region'], '1074')
            return {'files': [record(datetime.fromisoformat(query['start-date']))]}
        with patch('mlso_search.client.files', side_effect=api):
            result = search_all('kcor', 'pb', filters)
        self.assertEqual([r['date-obs'] for r in result['files']], ['2020-01-01T00:00:00', '2020-01-03T00:00:00'])

    def test_calendar_cadence(self):
        start = datetime(2020, 1, 31)
        self.assertEqual(cadence_bucket(datetime(2020, 2, 28), start, '1month'), 0)
        self.assertEqual(cadence_bucket(datetime(2020, 2, 29), start, '1month'), 1)
        self.assertEqual(cadence_bucket(datetime(2020, 3, 30), start, '1month'), 1)


if __name__ == '__main__':
    unittest.main()
