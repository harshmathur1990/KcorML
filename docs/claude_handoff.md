# TF-HDN v2.3 investigation handoff

## Objective

Build an end-to-end neural restoration model for temporally adjacent K-Cor
Level-2 pB frames that produces:

- denoised images in physical pB units;
- a learned low-bandwidth two-dimensional normalization field;
- a learned flattened-corona representation that retains real coronal structure;
- spatial, heteroscedastic uncertainty estimates; and
- eventually, CME probability maps.

The scientific priority is to increase faint-CME detectability without turning
outer-corona noise into false structure.

The model must not use NRGF inputs or targets, running/base/difference images,
handcrafted radial brightness corrections, or externally constructed noise
maps. It must preserve physical photometry and prevent date leakage between
the train, validation, and test splits. Faint-CME preservation must not be
claimed without labelled or simulated transient validation.

## Current model

The repository contains TF-HDN v2.3, with:

- two adjacent pB inputs;
- a shared encoder and temporal fusion;
- common, dynamic, and noise latent spaces;
- coarse normalization and noise decoders;
- a full-resolution flat-corona decoder;
- the log-domain decomposition `CLEAN_PB = NORM_FIELD * FLAT_CORONA`;
- a unit-geometric-mean gauge for `FLAT_CORONA`;
- a heteroscedastic Student-t likelihood;
- bandwidth, radial-flatness, gauge, latent-independence, and
  noise-structure losses;
- alternating leave-one-frame-out/blind-spot Noise2Noise training; and
- DistributedDataParallel support.

The CME loss is currently disabled with `cme_weight: 0`.

## Work completed

A leakage-safe manifest was generated with these pair counts:

| Split | Pairs |
| --- | ---: |
| Train | 5,023 |
| Validation | 1,183 |
| Test | 1,580 |

A five-epoch v2.3 smoke run produced
`artifacts/training_v23_smoke/best.pt`. Inference was run on test pair 0,
producing `artifacts/v23_smoke_prediction_000000.fits`.

`diagnose_product.py` reported `eligible: true` for both frames. Representative
values were:

| Metric | Observed value |
| --- | ---: |
| `normalization_log_std` | approximately 3.87-3.90 |
| `normalization_radial_log_std` | approximately 3.24-3.26 |
| `flat_log_std` | approximately 0.0068 |
| `flat_radial_rms` | approximately 0.0008 |
| `noise_structure_correlation` | approximately 0.0003 |
| `log_saturation_fraction` | 0 |

A second widely separated test pair, index 1500, was generated and visually
inspected. It reproduced essentially the same decomposition.

## Main finding

The checkpoint passes the existing numerical gate but fails the required
visual separation gate:

- `CLEAN_PB` is substantially blurred relative to `OBSERVED_PB`.
- Thin coronal morphology is lost.
- `CLEAN_PB` is nearly indistinguishable from `NORM_FIELD`.
- `NORM_FIELD` appears to carry almost the entire reconstructed signal.
- `FLAT_CORONA` is close to one, with only about 0.7% log-domain variation.
- Its independently stretched display mainly shows mottled decoder texture
  rather than convincing flattened coronal structure.
- `NOISE_SCALE` contains the same broad two-lobe morphology as the signal,
  suggesting responsibility leakage.
- Test pair 1500 shows the same behavior, making a pair-specific anomaly
  unlikely.

The current responsibility allocation appears to be:

```text
NORM_FIELD -> most broad coronal signal
FLAT_CORONA -> nearly constant plus weak texture
NOISE_SCALE -> broad signal-dependent morphology
```

This is a decomposition/responsibility collapse even though it is not a total
numerical collapse.

## Suspected masking problem

The valid-mask-aware plots still appeared to contain predictions and texture
inside the occulted center. The loader currently defines validity as:

```python
valid = np.isfinite(array) & (array != 0.0)
```

It then excludes only the geometric corners. This assumes that the occulted
disk is represented by exact zeros, which may be false for these FITS files.

This suspicion has not yet been conclusively verified. In the training
environment, check a representative product with:

```python
cy, cx = np.array(valid.shape) // 2
print("center valid:", valid[cy, cx])
print("center observed:", hdul["OBSERVED_PB_1"].data[cy, cx])

plt.figure()
plt.imshow(valid, origin="lower", cmap="gray")
plt.title("VALID_MASK")
plt.colorbar()
```

If the center is valid, derive the inner occulting mask from FITS metadata or
documented instrument geometry rather than using a guessed radius.

## Limitations of the current eligibility gate

The current gate checks finite positive output, normalization variation,
flat-corona annular radial RMS, fine-scale flat/noise correlation, and
saturation. It does not detect:

- collapse of `FLAT_CORONA` toward one;
- `CLEAN_PB` becoming nearly identical to `NORM_FIELD`;
- excessive smoothing;
- signal morphology entering `NOISE_SCALE`;
- incoherent texture in `FLAT_CORONA`; or
- standardized-residual miscalibration.

There is also a consistency issue to review: product diagnostics gate on total
`normalization_log_std`, whereas training checkpoint eligibility uses
`normalization_radial_log_std`.

## Implementation priorities

### 1. Verify and repair the science mask

- Inspect representative original FITS headers and arrays.
- Determine the correct inner occulting-disk boundary.
- Use FITS or documented instrument geometry when available.
- Add configuration only if the geometry cannot be obtained reliably.
- Add tests proving that the occulter and outside-FOV pixels are invalid while
  the scientific annulus remains valid.

### 2. Strengthen diagnostics

Add or evaluate:

- a minimum `flat_log_std`;
- clean-versus-normalization log correlation;
- noise-scale versus observed/normalization morphology;
- standardized residual statistics and coverage;
- reconstruction sharpness or high-frequency retention; and
- strict restriction of all metrics to valid science pixels.

Calibrate thresholds from data and controlled examples instead of choosing
values solely to reject the current checkpoint.

### 3. Repair component responsibility

After fixing the mask, investigate why the easiest solution places nearly all
signal in `NORM_FIELD`. Candidate experiments include:

- further restricting normalization-field spatial freedom;
- preventing non-background coronal morphology from entering normalization;
- strengthening signal/noise responsibility separation;
- rebalancing normalization-bandwidth and radial-flatness losses; and
- explicitly validating angular and transient structure retention in
  `FLAT_CORONA`.

Use small, interpretable ablations rather than changing many weights at once.

### 4. Retrain from scratch

Do not promote or reuse the existing checkpoint if it was trained with an
incorrect mask. Run a fresh five-epoch smoke experiment first.

### 5. Repeat the visual gate

Inspect representative test indices such as 0, 250, 500, 1000, and 1500. Use
valid-mask-aware displays with both identical physical/log stretches across
related panels and clearly labelled independent stretches.

### 6. Run a 15-20 epoch pilot

Run this only after the new smoke experiment shows plausible separation. Do
not naively extend the old five-epoch checkpoint by changing `epochs`: its
saved cosine scheduler was constructed with `T_max=5`.

### 7. Promote only after all gates pass

Do not start the 100-epoch run until numerical and visual gates agree.

## Promotion criteria

Before full training:

- occulted and outside-FOV pixels are correctly excluded;
- `CLEAN_PB` preserves credible coronal morphology;
- `NORM_FIELD` contains only the intended broad brightness variation;
- `FLAT_CORONA` contains coherent flattened coronal structure rather than weak
  decoder texture;
- `NOISE_SCALE` represents plausible uncertainty without copying solar
  structure;
- behavior is stable across several dates and test pairs;
- standardized residual coverage is reasonable; and
- no saturation or non-finite values occur.

## Repository status at handoff

At the time of this investigation, `main` and `origin/main` were both at
commit `c332116` (`model 2.3`), and the working tree was clean. No model or
training code was changed during the investigation.

The local development workspace did not contain the training data or generated
artifacts. A representative original FITS file and access to the training
environment are required to validate the occulter geometry.

The immediate programming task is to verify or fix the valid mask and add
regression tests before changing the model or retraining.
