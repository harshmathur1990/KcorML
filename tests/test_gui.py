import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from kcor_gui import Window, client, download_one


class DownloaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = Window(auto_load=False)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def wait(self):
        deadline = time.monotonic() + 5
        while self.window.job is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertIsNone(self.window.job)

    def test_default_and_cadence(self):
        self.assertEqual(self.window.product.currentData(), 'pbavgenh')
        self.window.cadence.setValue(2)
        self.window.unit.setCurrentText('hours')
        self.assertEqual(self.window.filters()['every'], '2hours')
        self.assertNotIn('wave-region', self.window.filters())

    def test_invalid_dates(self):
        self.window.start.setDateTime(self.window.end.dateTime().addDays(1))
        with self.assertRaises(ValueError):
            self.window.filters()

    def test_search_and_empty_results(self):
        record = {'filename': 'example.fts', 'product': 'pbavgenh', 'filesize': 42}
        with patch.object(client, 'files', return_value={'files': [record]}) as request:
            self.window.search_files()
            self.wait()
            self.assertEqual(request.call_args.args[:2], ('kcor', 'pbavgenh'))
            self.assertEqual(self.window.file_model.rowCount(), 1)
            self.assertTrue(self.window.all_files.isEnabled())
        with patch.object(client, 'files', return_value={'files': []}):
            self.window.search_files()
            self.wait()
            self.assertFalse(self.window.all_files.isEnabled())

    def test_atomic_failure_preserves_existing(self):
        record = {'filename': 'example.fts', 'url': 'http://api.mlso.ucar.edu/v1/download?x=1'}
        def fail(record, directory):
            (Path(directory) / record['filename']).write_bytes(b'partial')
            raise OSError('Connection interrupted')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            target = folder / record['filename']
            target.write_bytes(b'original')
            with patch.object(client, 'download_file', side_effect=fail):
                with self.assertRaises(OSError):
                    download_one(record, folder, False)
            self.assertEqual(target.read_bytes(), b'original')
            self.assertEqual(list(folder.iterdir()), [target])

    def test_success_skip_and_https(self):
        record = {'filename': 'example.fts', 'url': 'http://api.mlso.ucar.edu/v1/download?x=1'}
        def success(record, directory):
            self.assertTrue(record['url'].startswith('https://'))
            target = Path(directory) / record['filename']
            target.write_bytes(b'data')
            return target
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(client, 'download_file', side_effect=success) as call:
                self.assertEqual(download_one(record, Path(directory)), 'Downloaded')
                self.assertEqual(download_one(record, Path(directory)), 'Skipped existing')
                self.assertEqual(call.call_count, 1)

    def test_unsafe_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                download_one({'filename': '../escape'}, Path(directory))


if __name__ == '__main__':
    unittest.main()
