# MLSO Qt downloader

A Python/PySide6 (Qt 6) desktop GUI using `mlso-api-client`. The default is
K-Cor **White Light Level-2 pB enhanced intensity**, API product `pbavgenh`
(enhanced average image). It downloads the archive's files; it does not apply
an enhancement algorithm locally.

## Run

Python 3.10 or newer is required. From this project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python kcor_gui.py
```

On Windows, activate with `.venv\Scripts\activate` instead.

1. Let the catalog load, then choose the instrument and product. K-Cor enhanced
   pB is selected automatically; the live catalog exposes all API products,
   including standard pB, NRGF, extended averages, and UCoMP products.
2. Set your observation interval in **UTC**. Both dates are always sent; the
   optional Carrington rotation further restricts that interval.
3. Leave cadence at **All files**, or request one file per specified interval.
   UCoMP also has wavelength and observing-plan filters.
4. Click **Search files**. Select rows (Ctrl/Cmd or Shift for multiple rows),
   or use **Download all results**.
5. Enter your [HAO-registered email](https://registration.hao.ucar.edu), choose
   a destination, and download. No email is needed for catalog searches.

`pbextavgenh` is a separate enhanced extended-average product;
`nrgfavgenh` is enhanced NRGF. Product availability varies by observation date.
The API currently advertises FITS for `pbavgenh`.

Searches and downloads run in a worker thread. Cancellation waits for the
current request or file; requests have a 15-second connection timeout and a
90-second read-inactivity timeout. Completed files remain after cancellation.
Files are staged in a temporary directory before replacing the destination;
failed transfers are cleaned up. Failed rows can be selected and retried.
Existing files are skipped by filename, without validating their contents;
uncheck that option to replace them. The API's size metadata can be missing
or inaccurate and is displayed only as an estimate. Files are saved with the
names supplied by the client, without decompression or conversion.

The GUI starts with documented K-Cor products if catalog retrieval fails.
Use **Refresh catalog** to retry. Network errors appear in the log and a dialog.
Email is held only in memory for this run.

## Verification

```sh
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

Tests cover GUI query results, filters, invalid dates, safe filenames, staged
transfer failures, HTTPS download URLs, and skipping existing files. Real
authenticated downloads require your registered email.

References: [official MLSO API](https://www2.hao.ucar.edu/mlso/mlso-api),
[client documentation](https://mlso-api-client.readthedocs.io/en/v1.0.0/),
[MLSO data use](https://www2.hao.ucar.edu/mlso).
