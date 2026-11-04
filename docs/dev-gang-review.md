# Dev Gang review — 4 November 2026

This review revisited cleanup commit `9b454cdb98617473e20490c33d9809e1ecd7a3b8`. It used the recovered Dev Gang skill's delivery-governor, crew-role, acceptance-audit, debugging, and smoke-test workflows. Three independent reviewers examined model/scientific correctness, geospatial processing, and security/release behavior before implementation. Each fix had an explicit scope, a reproduced failure or measurement, and acceptance evidence.

The recovered skill copy lacks `env_probe.py` and `smoke_matrix.py`. As allowed by its recovery instructions, installed-version inspection, dependency-manifest review, and explicit smoke commands supplied the equivalent checks. This repository contains command-line processing code; browser, mobile, and visual acceptance checks do not apply.

## Findings and fixes

| Finding | Reproduction and resulting behavior | Regression coverage |
| --- | --- | --- |
| Valid raster values lost during postprocessing | Mosaicking an all-valid `[0, 1, 2]` tile over `[3, 3, 3]` produced `[3, 1, 2]`. Clipping and remapping could also invalidate valid zeroes. Outputs now carry internal validity masks; mosaics select the first valid pixel independently of its stored value. Missing coverage remains masked. | [Postprocessing tests](../tests/test_postprocess.py): no-nodata rasters, explicit masks overriding nodata, valid zeroes, extraction, clipping, remapping, overlap priority, gaps, offsets, and multiple output windows. The added regressions exposed 10 failures before the fix; all 22 postprocessing tests now pass. |
| Diverged training could publish a model | A tiny training run with an extreme but finite positive learning rate produced non-finite weights. Optimized prediction could still return finite scores, so checking scores alone was insufficient. Training now checks each weight tensor and the validation-score shape/finiteness before publishing. Failure retains an existing output and removes temporary checkpoints. | [Training tests](../tests/test_training_io.py) cover invalid weights/scores and output preservation. The actual divergent training reproduction was rerun and rejected with the previous output intact. |
| Uppercase model suffixes failed late or selected the wrong format | `.KERAS` and `.H5` passed validation but Keras 2.15 selected SavedModel directories rather than the intended file formats. Publication could fail after training. Only lowercase `.keras` and `.h5` are now accepted, before loading samples. | Early-failure checks in [training tests](../tests/test_training_io.py); real save/reload coverage for both accepted formats in [model tests](../tests/test_model.py). |
| Repeated prediction setup added overhead | Each patch batch created a new Keras prediction iterator. Keras models now use `predict_on_batch`; custom predictors exposing only `predict` remain supported. | [Inference tests](../tests/test_inference.py) and [real Keras parity test](../tests/test_inference_keras.py) verify dispatch, bounded patch batches, and equivalent scores/classes. |

The mosaic implementation allocates a class window and a separate occupancy mask, each at most 512 × 512 pixels, and opens one intersecting source at a time. It retains CRS, resolution, grid, dtype, and nodata compatibility checks. Sieve retains GDAL's global connected-region operation.

## Independent checks

- **20 deterministic mosaic cases** matched an independent NumPy coverage/value oracle. Each used three tiny rasters with offsets, overlaps, explicit masks, gaps, and valid zeroes. Output transforms, nodata, Rasterio masks, and GDAL 3.4 mask reads also matched.
- **5,640 patch comparisons** matched between sampling and inference across small raster shapes, multiple patch/window sizes, edges, tile seams, masked pixels, and non-finite values.
- Six generated ResNet save/reload cases covered `.keras` and `.h5`, patch sizes 9/12/15, and one/two/three bands. All retained the 13-output interface.
- A separately reloaded ResNet produced identical scores and classes through `predict` and `predict_on_batch` for batches of 1, 7, and 33 patches.
- Security/release inspection covered parameterized SQL, streamed database rows, explicit legacy-pickle opt-in, input/output path protection, temporary output publication, and read-only CI permissions. No additional actionable defect was found in those reviewed paths. This was a scoped review, not a security certification.

Warm CPU timing on a small ResNet-29 fixture measured six batches of 32 patches at approximately 0.198 seconds with `predict` and 0.092 seconds with `predict_on_batch`. These are development measurements, not a continent-scale throughput estimate or a test threshold. The native batch method executes the requested batch together; the previous `predict` path internally split it into batches of 32. Reduce `--batch-size` if activation memory is constrained.

## Validation

The complete local suite passed: **105 tests**, compared with 70 at the start of this review. It includes a tiny preparation → training → prediction command-line run. Ruff lint, Ruff formatting, shell syntax, and diff-whitespace checks also passed.

Environment: Python 3.10.12, NumPy 1.26.4, Rasterio 1.4.4, system GDAL Python bindings 3.4.1, TensorFlow CPU 2.15.1, pytest 8.4.2, Ruff 0.17.3, and psycopg2-binary 2.9.13 on Linux. Rasterio uses its bundled GDAL separately from the system Python bindings. Existing dependencies were reused; this review did not change dependency requirements.

After installing the documented dependencies, run from the repository root:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  GDAL_NUM_THREADS=1 GDAL_CACHEMAX=256 \
  TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 \
  .venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
bash -n bwcluster_bash_job.sh
git diff --check
```

The passing run emitted upstream Rasterio/Affine and Keras deprecation notices and the expected legacy-HDF5 format notice. GitHub Actions runs the full suite with CPU TensorFlow; its result should be checked on the current pull-request commit.

## Scope and remaining limits

The original ResNet architecture, 13 output indices, and class mapping remain intact. The normalization and patch-alignment changes from the initial cleanup remain documented in the [README](../README.md). No new scientific method was introduced by this review.

Tests used tiny synthetic inputs. They do **not** establish published classification accuracy or equivalence to the released European dataset. Downloaded historical model weights, full-Europe processing, GPU execution, Windows execution, live PostGIS access, and cluster job submission were not exercised. There is no live service or UI deployment in this change. No production rasters or datasets were modified.

Mask-aware consumers are required to distinguish valid class values from missing coverage. Mosaic throughput at production scale remains unmeasured. Training samples and model activations still require memory beyond the GDAL cache budget; the 256 MiB cache is not a process-memory limit.

The review fixes are delivered as a separate follow-up commit. Reverting that commit restores the initial cleanup while leaving existing datasets and outputs untouched. Existing outputs remain protected unless an explicit overwrite option is used.
