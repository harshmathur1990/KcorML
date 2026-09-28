"""Qt interface for MLSO's public catalog and authenticated downloads."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote, urlsplit, urlunsplit

import requests
from mlso.api import client
from PySide6.QtCore import QDate, QDateTime, QThread, QTime, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDateTimeEdit, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QAbstractItemView,
)

BASE_URL = 'https://api.mlso.ucar.edu'
REGISTRATION = 'https://registration.hao.ucar.edu'
KCOR_PRODUCTS = [
    ('pbavgenh', 'pB enhanced intensity (averaged)'),
    ('pb', 'pB intensity'), ('pbavg', 'pB average'),
    ('pbextavg', 'pB daily average'),
    ('pbextavgenh', 'pB enhanced extended average'),
    ('nrgf', 'NRGF intensity'), ('nrgfavg', 'NRGF average'),
    ('nrgfextavg', 'NRGF extended average'),
    ('nrgfavgenh', 'NRGF enhanced average'),
    ('nrgfextavgenh', 'NRGF enhanced extended average'),
    ('pbdiff', 'pB difference'), ('nrgf+diff', 'NRGF + difference'),
    ('all', 'All products'),
]


class TimeoutSession(requests.Session):
    def request(self, method, url, **kwargs):
        kwargs.setdefault('timeout', (15, 90))
        return super().request(method, url, **kwargs)


def configure_client():
    # v1.0.0 has no timeout parameter. Adapt only its transport references;
    # all catalog, authentication and file operations still use the client.
    public = TimeoutSession()
    client.requests = SimpleNamespace(get=public.get, exceptions=requests.exceptions)
    client.session = TimeoutSession()


def safe_filename(name):
    if not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError(f'Unsafe filename received: {name!r}')
    return name


def download_one(record, destination, skip_existing=True):
    """Stage a complete response before replacing the final file."""
    name = safe_filename(record['filename'])
    target = destination / name
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise ValueError(f'Destination is not a regular file: {target}')
        if skip_existing:
            return 'Skipped existing'
    record = dict(record)
    parts = urlsplit(record['url'])
    if parts.hostname != 'api.mlso.ucar.edu' or parts.scheme not in ('http', 'https'):
        raise ValueError('Download URL is not on the MLSO API host.')
    record['url'] = urlunsplit(parts._replace(scheme='https'))
    with tempfile.TemporaryDirectory(prefix='.mlso-', dir=destination) as staging:
        downloaded = Path(client.download_file(record, staging))
        if not downloaded.is_file() or downloaded.stat().st_size == 0:
            raise ValueError('The server returned an empty file.')
        downloaded.replace(target)
    return 'Downloaded'


class Job(QThread):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int, str, str)

    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action

    def run(self):
        try:
            self.result.emit(self.action(self))
        except Exception as exc:
            self.error.emit(f'{type(exc).__name__}: {exc}')


def size_text(value):
    if not value:
        return 'Unknown'
    return f'{int(value) / 1024**2:,.2f} MiB'


class Window(QMainWindow):
    def __init__(self, auto_load=True):
        super().__init__()
        self.setWindowTitle('MLSO • K-Cor data downloader')
        self.resize(1100, 820)
        self.job = None
        self.records = []
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel('K-Cor / MLSO data downloader')
        title.setStyleSheet('font-size: 23px; font-weight: 600;')
        layout.addWidget(title)
        layout.addWidget(QLabel('White Light Level-2 pB • enhanced intensity selected by default'))

        self.options = QGroupBox('Search the archive')
        form = QFormLayout(self.options)
        self.instrument = QComboBox()
        self.instrument.addItem('COSMO K-Coronagraph (KCor)', 'kcor')
        self.product = QComboBox()
        for ident, label in KCOR_PRODUCTS:
            self.product.addItem(f'{label} [{ident}]', ident)
        self.refresh = QPushButton('Refresh catalog')
        row = QHBoxLayout()
        row.addWidget(self.instrument, 1)
        row.addWidget(self.refresh)
        form.addRow('Instrument', row)
        form.addRow('Product', self.product)
        self.description = QLabel('Enhanced pB averaged images: pbavgenh. Availability depends on date.')
        self.description.setWordWrap(True)
        form.addRow('', self.description)
        self.start = QDateTimeEdit(QDateTime(QDate.currentDate().addDays(-1), QTime(0, 0)))
        self.end = QDateTimeEdit(QDateTime(QDate.currentDate().addDays(-1), QTime(23, 59, 59)))
        for edit in (self.start, self.end):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat('yyyy-MM-dd HH:mm:ss')
        dates = QHBoxLayout()
        dates.addWidget(self.start)
        dates.addWidget(QLabel('to'))
        dates.addWidget(self.end)
        form.addRow('Observation time (UTC)', dates)
        self.cadence = QSpinBox()
        self.cadence.setRange(0, 100000)
        self.cadence.setSpecialValueText('All files')
        self.unit = QComboBox()
        self.unit.addItems(['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'quarters', 'years'])
        self.unit.setCurrentText('minutes')
        sampling = QHBoxLayout()
        sampling.addWidget(self.cadence)
        sampling.addWidget(self.unit)
        form.addRow('One file every', sampling)
        self.rotation = QSpinBox()
        self.rotation.setRange(0, 10000)
        self.rotation.setSpecialValueText('Any rotation')
        form.addRow('Carrington rotation', self.rotation)
        self.wave = QComboBox()
        self.wave.setEditable(True)
        self.wave.addItems(['', '637', '706', '789', '1074', '1079'])
        self.plan = QLineEdit()
        self.plan.setPlaceholderText('Optional: synoptic or waves')
        form.addRow('UCoMP wavelength (nm)', self.wave)
        form.addRow('UCoMP observing plan', self.plan)
        self.wave.setEnabled(False)
        self.plan.setEnabled(False)
        self.search = QPushButton('Search files')
        form.addRow(self.search)
        layout.addWidget(self.options)

        self.summary = QLabel('Refresh the catalog or search using the built-in K-Cor products.')
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['UTC observation', 'Product', 'Filename', 'Size (reported)', 'Status'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        self.download_options = QGroupBox('Download')
        download_form = QFormLayout(self.download_options)
        self.email = QLineEdit()
        self.email.setPlaceholderText('Your HAO-registered email address')
        registration = QLabel(f'<a href="{REGISTRATION}">Register with HAO</a>')
        registration.setOpenExternalLinks(True)
        email_row = QHBoxLayout()
        email_row.addWidget(self.email)
        email_row.addWidget(registration)
        download_form.addRow('Email', email_row)
        self.folder = QLineEdit(str(Path.cwd() / 'data'))
        browse = QPushButton('Browse…')
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder)
        folder_row.addWidget(browse)
        download_form.addRow('Save to', folder_row)
        self.skip = QCheckBox('Skip existing files')
        self.skip.setChecked(True)
        download_form.addRow(self.skip)
        buttons = QHBoxLayout()
        self.selected = QPushButton('Download selected')
        self.all_files = QPushButton('Download all results')
        buttons.addWidget(self.selected)
        buttons.addWidget(self.all_files)
        download_form.addRow(buttons)
        layout.addWidget(self.download_options)
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.cancel = QPushButton('Cancel')
        self.cancel.setEnabled(False)
        bottom.addWidget(self.progress)
        bottom.addWidget(self.cancel)
        layout.addLayout(bottom)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(95)
        layout.addWidget(self.log)
        self.refresh.clicked.connect(self.load_catalog)
        self.instrument.currentIndexChanged.connect(self.load_products)
        self.product.currentIndexChanged.connect(self.product_changed)
        self.search.clicked.connect(self.search_files)
        self.selected.clicked.connect(lambda: self.download(False))
        self.all_files.clicked.connect(lambda: self.download(True))
        browse.clicked.connect(self.browse)
        self.cancel.clicked.connect(self.cancel_job)
        self.set_busy(False)
        if auto_load:
            QTimer.singleShot(0, self.load_catalog)

    def set_busy(self, busy):
        self.options.setEnabled(not busy)
        self.download_options.setEnabled(not busy)
        self.selected.setEnabled(not busy and bool(self.records))
        self.all_files.setEnabled(not busy and bool(self.records))
        self.cancel.setEnabled(busy)

    def launch(self, action, callback, message):
        if self.job is not None:
            return
        self.log.appendPlainText(message)
        self.set_busy(True)
        self.progress.setRange(0, 0)
        self.job = Job(action, self)
        self.job.result.connect(callback)
        self.job.error.connect(self.show_error)
        self.job.progress.connect(self.file_progress)
        self.job.finished.connect(self.job_finished)
        self.job.start()

    def job_finished(self):
        self.job.deleteLater()
        self.job = None
        self.progress.setRange(0, 100)
        self.set_busy(False)

    def show_error(self, message):
        self.log.appendPlainText(message)
        QMessageBox.warning(self, 'MLSO request failed', message)

    def cancel_job(self):
        if self.job:
            self.job.requestInterruption()
            self.cancel.setEnabled(False)
            self.log.appendPlainText('Cancel requested; waiting for the current request/file to finish.')

    def load_catalog(self):
        def fetch(job):
            instruments = client.instruments(base_url=BASE_URL)
            products = {}
            for instrument in instruments:
                if job.isInterruptionRequested():
                    return None
                products[instrument['id']] = client.products(instrument['id'], base_url=BASE_URL)['products']
            return instruments, products
        def loaded(result):
            if result is None:
                return
            instruments, self.catalog = result
            self.instrument.blockSignals(True)
            self.instrument.clear()
            for item in instruments:
                self.instrument.addItem(item['name'], item['id'])
            index = self.instrument.findData('kcor')
            self.instrument.setCurrentIndex(max(0, index))
            self.instrument.blockSignals(False)
            self.populate_products()
            self.log.appendPlainText('Live catalog loaded.')
        self.launch(fetch, loaded, 'Loading MLSO instruments and products…')

    def load_products(self):
        self.populate_products()

    def populate_products(self):
        instrument = self.instrument.currentData()
        self.wave.setEnabled(instrument == 'ucomp')
        self.plan.setEnabled(instrument == 'ucomp')
        self.product.blockSignals(True)
        self.product.clear()
        for p in getattr(self, 'catalog', {}).get(instrument, []):
            self.product.addItem(f"{p['title']} [{p['id']}]", p['id'])
            self.product.setItemData(self.product.count()-1, p['description'], 3)
        index = self.product.findData('pbavgenh' if instrument == 'kcor' else 'l2')
        self.product.setCurrentIndex(max(0, index))
        self.product.blockSignals(False)
        self.product_changed()

    def product_changed(self):
        self.description.setText(self.product.currentData(3) or '')
        self.records = []
        self.table.setRowCount(0)
        self.summary.setText('Search to list files for this product.')
        self.selected.setEnabled(False)
        self.all_files.setEnabled(False)

    def filters(self):
        if self.start.dateTime() > self.end.dateTime():
            raise ValueError('Start time must be before end time.')
        filters = {'start-date': self.start.dateTime().toString('yyyy-MM-ddTHH:mm:ss'),
                   'end-date': self.end.dateTime().toString('yyyy-MM-ddTHH:mm:ss')}
        if self.cadence.value():
            filters['every'] = f'{self.cadence.value()}{self.unit.currentText()}'
        if self.rotation.value():
            filters['cr'] = str(self.rotation.value())
        if self.instrument.currentData() == 'ucomp':
            for key, value in [('wave-region', self.wave.currentText()), ('obs-plan', self.plan.text())]:
                if value.strip():
                    filters[key] = value.strip()
        # The upstream client concatenates values instead of URL-encoding them.
        return {key: quote(value, safe=':') for key, value in filters.items()}

    def search_files(self):
        try:
            filters = self.filters()
            instrument, product = self.instrument.currentData(), self.product.currentData()
            if not instrument or not product:
                raise ValueError('Choose an instrument and product first.')
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.records = []
        self.table.setRowCount(0)
        def fetch(job):
            result = client.files(instrument, product, filters, base_url=BASE_URL)
            return None if job.isInterruptionRequested() else result
        def loaded(result):
            if result is None:
                self.summary.setText('Search cancelled.')
                return
            self.records = result['files']
            self.table.setRowCount(len(self.records))
            for row, record in enumerate(self.records):
                values = [record.get('date-obs', ''), record.get('product', ''), record['filename'], size_text(record.get('filesize')), 'Ready']
                for col, value in enumerate(values):
                    self.table.setItem(row, col, QTableWidgetItem(value))
            total = sum(r.get('filesize', 0) or 0 for r in self.records)
            self.summary.setText(f'{instrument} / {product}: {len(self.records):,} files • reported total {size_text(total)} (sizes may be incomplete)')
            self.progress.setValue(0)
        self.launch(fetch, loaded, f'Searching {instrument}/{product}…')

    def browse(self):
        folder = QFileDialog.getExistingDirectory(self, 'Download folder', self.folder.text())
        if folder:
            self.folder.setText(folder)

    def download(self, all_results):
        rows = list(range(len(self.records))) if all_results else sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        email = self.email.text().strip()
        if not rows:
            self.show_error('Select at least one file row.')
            return
        if '@' not in email or not self.folder.text().strip():
            self.show_error('Enter your HAO-registered email and a download folder.')
            return
        destination = Path(self.folder.text()).expanduser()
        records = [(row, dict(self.records[row])) for row in rows]
        skip = self.skip.isChecked()
        def transfer(job):
            destination.mkdir(parents=True, exist_ok=True)
            client.authenticate(email, base_url=BASE_URL)
            completed = skipped = failed = 0
            for row, record in records:
                if job.isInterruptionRequested():
                    break
                job.progress.emit(row, 'Downloading…', '')
                try:
                    status = download_one(record, destination, skip)
                    completed += status == 'Downloaded'
                    skipped += status == 'Skipped existing'
                    job.progress.emit(row, status, '')
                except Exception as exc:
                    failed += 1
                    job.progress.emit(row, 'Failed', f"{record['filename']}: {exc}")
            prefix = 'Cancelled' if job.isInterruptionRequested() else 'Finished'
            return f'{prefix}: {completed} downloaded, {skipped} skipped, {failed} failed.'
        self.download_total = len(rows)
        self.download_done = 0
        self.launch(transfer, self.log.appendPlainText, f'Downloading {len(rows)} files to {destination}…')

    def file_progress(self, row, status, message):
        self.table.item(row, 4).setText(status)
        if status != 'Downloading…':
            self.download_done += 1
        self.progress.setRange(0, self.download_total)
        self.progress.setValue(self.download_done)
        if message:
            self.log.appendPlainText(message)

    def closeEvent(self, event):
        if self.job is not None:
            self.cancel_job()
            self.log.appendPlainText('Please close again after the operation has stopped.')
            event.ignore()
        else:
            event.accept()


def main():
    configure_client()
    app = QApplication(sys.argv)
    app.setApplicationName('MLSO Downloader')
    window = Window()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
