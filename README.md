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

Searches query every UTC day in the selected interval. Any response containing
3,000 or more files is treated as potentially truncated and recursively split
into smaller time intervals. Overlapping boundaries are deduplicated by
instrument, product, and filename. If even a one-second interval hits the cap,
the search fails explicitly instead of claiming completeness. Failed or
cancelled searches discard partial results. Progress shows completed days,
unique files found so far, and the current query. Multi-year searches can take
a substantial amount of time and memory; the table uses a Qt model to avoid
allocating a widget item for every cell.

Cadence is applied after collecting metadata over the entire range: the earliest
file per instrument/product/wavelength in each interval anchored at the chosen
start time is retained. Month, quarter, and year intervals use calendar months.
This keeps cadence independent of how requests are split, but sampling does not
reduce the number of metadata requests.

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

Tests cover multi-year coverage, capped responses, interval boundaries, global
cadence, cancellation, search failures, GUI query results, filters, invalid dates,
safe filenames, staged
transfer failures, HTTPS download URLs, and skipping existing files. Real
authenticated downloads require your registered email.

References: [official MLSO API](https://www2.hao.ucar.edu/mlso/mlso-api),
[client documentation](https://mlso-api-client.readthedocs.io/en/v1.0.0/),
[MLSO data use](https://www2.hao.ucar.edu/mlso).

## Neural restoration scaffold

The `kcor_ml` package implements the initial modular scaffold described in
[`docs/ml_approach_plan.yaml`](docs/ml_approach_plan.yaml). Its responsibilities
are separated into FITS discovery and temporal pairing, datasets and loaders,
model components, model construction, probabilistic losses, training,
checkpoints, and FITS inference output. Multi-GPU training uses ordinary
PyTorch DistributedDataParallel (DDP); the code intentionally contains no FSDP.

Install the additional ML dependencies separately:

```sh
python -m pip install -r requirements-ml.txt
```

The current model is the constrained TF-HDN v2.3. It uses fixed physical scaling,
log-domain decomposition with a unit-geometric-mean gauge, annular-mean
flatness regularization on the learned clean latent, and alternating
leave-one-frame-out Noise2Noise training. It never forms a difference image or
uses NRGF as input or target.

Build the leakage-safe pair manifest once, then run the five-epoch v2.3 gate:

```sh
python pipeline.py --config configs/default.json --index
python pipeline.py --config configs/v23_smoke.json --train
```

For single-node multi-GPU training, `batch_size` is interpreted per GPU. For
example, six processes with `batch_size: 5` use a global batch size of 30:

```sh
torchrun --standalone --nproc_per_node=6 pipeline.py --config configs/v23_smoke.json --train
```

Do not start the 100-epoch run unless the smoke checkpoint reports
`checkpoint_eligible=True` and full-frame inspection confirms the separation.
Earlier checkpoints are intentionally incompatible with v2.3 and remain
reconstruction/prototype baselines only. After the gate passes, train with
`configs/default.json`; v2.3 checkpoints are written under
`artifacts/training_v23`.

Only rank zero displays the progress bar and writes `best.pt` and `last.pt`.
Each rank has its own DataLoader workers, so `num_workers: 1` creates six total
workers in the example above. Checkpoints are saved without a DDP prefix and
remain loadable by plain single-GPU inference.

Evaluate or generate a multi-extension FITS product with a checkpoint:

```sh
python pipeline.py --config configs/evaluate_full.json --evaluate --checkpoint artifacts/training_v23/best.pt
python pipeline.py --config configs/evaluate_full.json --predict --checkpoint artifacts/training_v23/best.pt --pair-index 0
python diagnose_product.py artifacts/prediction_000000.fits
```
