# OSM4EO: OpenStreetMap land use for Europe, 2020

Research code for land-use classification from Sentinel-2 imagery and OpenStreetMap labels using a residual convolutional neural network.

- **Paper:** [OSMlanduse: a dataset of European Union land use at 10 m resolution derived from OpenStreetMap and Sentinel-2](https://www.nature.com/articles/s41597-025-04703-8)
- **Dataset:** [Heidelberg research data repository](https://doi.org/10.11588/data/IUTCDN)
- **Map:** [osmlanduse.org](https://osmlanduse.org)
- **Existing models:** [original external model download](https://1drv.ms/f/s!AsaD57Gg8TWIlfFvgtTjDT6E0067Jg?e=kkmekb)

Images, labels, model weights, and database access are **not bundled**. The original code assumed private SDS directories and a bwcluster environment. Commands now take explicit paths, and importing a module does not start processing. The [historical country/tile table](docs/historical-results.md) is retained separately.

## Installation

The tested target is Python 3.10 on Linux, NumPy 1.26, Rasterio 1.4, GDAL 3.4, and TensorFlow CPU 2.15.1. GDAL's Python bindings must match its native library. On Ubuntu 22.04:

```bash
sudo apt-get install python3-venv python3-gdal
/usr/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install only the extras you need:

```bash
python -m pip install -r requirements-training.txt  # training and prediction, CPU
python -m pip install -r requirements-database.txt  # optional PostGIS exporter
python -m pip install -r requirements-dev.txt       # tests and lint
```

For GPU training, use a TensorFlow environment compatible with your GPU/CUDA installation instead of the CPU package. GPU execution is not covered by this repository's tests. Legacy MATLAB/dataframe helpers in `hsi_io.py` additionally need `hdf5storage`, `scipy`, or `pandas` when called.

## Prepare samples

Use a multiband Sentinel-2 raster and a single-band OSM label raster with identical CRS, transform, width, and height. Input bands must have the same meaning and order during training and prediction.

```bash
python 01_preproce_prepare_sample.py \
  --image data/S2_10m_3035.tif \
  --labels data/S2_10m_3035_label.tif \
  --train-output data/train.npz \
  --validation-output data/validation.npz \
  --samples-per-class 5000 --patch-size 12 --seed 42
```

Sampling is deterministic for the same inputs, seed, and options. It selects at most the requested number of distinct valid pixel centers per class, and splits each class using its actual sample count. Missing classes are skipped; a singleton goes to training. A split requires at least one class with two valid pixels. Neighboring patches can still overlap: this random split is **not a spatially independent evaluation**. Use geographically separate data when measuring generalization.

The numeric NPZ format stores `patches`, `labels`, and `coordinates`, plus normalization and sampling metadata. It needs no pickle. Selected patches are stored in memory; reduce `--samples-per-class` or `--patch-size` if necessary. The source image is read in bounded windows.

## Train

```bash
python 02_keras_models_single.py \
  --train data/train.npz --validation data/validation.npz \
  --output-model saved_models/model.keras \
  --depth 29 --epochs 600 --batch-size 256 --seed 42
```

Training retains the original ResNet architecture and **13 output indices (0–12)**. Classes 1–12 are land-use classes; index 0 remains reserved. The default learning rate is fixed at 0.01, matching the original effective behavior. `--learning-rate-schedule` enables the previously defined but unused schedule. The best validation checkpoint is saved to the requested model path.

Old object-array `.npy` samples are supported only with `--allow-legacy-pickle`. Use that option exclusively for files you trust; pickle can execute code. Training loads the selected samples into memory. The Moab job template `bwcluster_bash_job.sh` accepts the same arguments; configure the cluster environment and resource request before submitting it.

## Predict

```bash
python 03_keras_classification.py \
  --input data/S2_10m_3035.tif \
  --model saved_models/model.keras \
  --output results/classified.tif \
  --batch-size 256 --window-size 128
```

Prediction reads the raster in windows, uses bounded patch batches, and writes one-band `uint16` class IDs on the source grid. Invalid source pixels become output nodata 0. The model's band count, patch dimensions, and output classes are checked before publishing the result.

**Numerical changes to understand when comparing old results:**

- Training and prediction now share the same patch layout: a 12×12 patch places its target pixel at index 5, with normalized zero padding at image edges. The original inference used index 6 and symmetric padding, causing a one-pixel offset from training.
- `--legacy-patches` restores that old inference layout for comparisons with existing models. It does not reproduce the old pipeline bit for bit.
- Global min/max normalization now excludes masked and non-finite values. It uses one pair of extrema across the raster's bands, preserving the original global scaling convention. Constant valid images retain their values.
- Sampling no longer duplicates centers across the training and validation sets. Existing models remain loadable, but retraining and resulting accuracy may differ. Synthetic tests do not establish equivalence to the published dataset or validate published model accuracy.

## Postprocess

All commands provide `--help` and require explicit input/output paths. Raster outputs refuse replacement unless `--overwrite` is given, and cannot replace their input. Completed TIFFs are published from temporary files after basic validation.

```bash
# Extract class 12 (water); other pixels become 0.
python 04_postproce_extract_LU.py results/classified.tif results/water.tif --class-id 12

# Remove small connected regions using GDAL's global sieve operation.
python 05_post_sieve_filtering.py results/classified.tif results/sieved.tif --size 64 --connectivity 8

# Retain pixels where a raster mask equals 1; grids must match exactly.
python 06_post_extent_clip.py results/sieved.tif data/mask.tif results/masked.tif --mask-value 1

# Convert model indices back to the original OSM codes.
python 07_post_label_transfer.py results/masked.tif results/osm_codes.tif

# Mosaic aligned tiles without allocating the full mosaic in memory.
python 08_post_extent_merge.py results/tile1.tif results/tile2.tif --output results/mosaic.tif

# Crop/mask with every geometry in a GeoJSON (WGS84 unless specified).
python 09_post_clip_geojson.py results/mosaic.tif data/boundary.geojson results/clipped.tif
```

GeoJSON geometry is transformed to the raster CRS; use `--geometry-crs` if the file uses another CRS. Mosaics preserve class values and reject incompatible grids. Sieve uses GDAL's connected-region processing: its bookkeeping can grow with the number of regions even though pixels are not read into a whole-image NumPy array.

The class mapping is:

| Model ID | OSM code | Land-use class |
| --- | --- | --- |
| 1 | 11 | Urban fabric |
| 2 | 12 | Industrial/commercial |
| 3 | 13 | Mines/dumps/construction |
| 4 | 14 | Artificial vegetation |
| 5 | 21 | Arable land |
| 6 | 22 | Permanent crops |
| 7 | 23 | Pastures |
| 8 | 31 | Forests |
| 9 | 32 | Shrub/herbaceous vegetation |
| 10 | 33 | Open spaces |
| 11 | 41 | Inland wetlands |
| 12 | 5 | Water |

## Optional database extraction

`shp_to_lulc.py` queries `public.naturals` in a compatible PostGIS database, with `the_geom` in EPSG:3857 and an OSM `type` column. Configure connection settings through standard libpq variables (`PGHOST`, `PGDATABASE`, `PGUSER`, `PGSERVICE`) and a protected `.pgpass` or service file; credentials are not embedded in the code.

```bash
python shp_to_lulc.py data/wgs84_boundaries results/osm --tile-size 4 --batch-size 1000
```

Inputs must be WGS84 shapefiles. Each feature's bounding box is split into query tiles. The exporter clips polygons crossing tile boundaries, streams database rows, and includes the input filename in output names. Output geometries use EPSG:3857. It extracts the **bounding boxes**, not the exact input polygon outlines. Overlapping input boxes can yield duplicate or overlapping results. The historical database taxonomy also includes coastal wetlands (`4.2`); the 12-class model mapping does not. There is no automatic connection to the original Heidelberg server, and automated tests do not access a live database.

Existing shapefiles are protected. Shapefiles have multiple sidecar files, so publication of the complete set is not a single atomic operation; each successful export is retained if a later export fails.

## Checks and resource use

```bash
python -m pytest
ruff check .
ruff format --check .
```

Tests create tiny temporary rasters and geometries. They cover patch alignment, sampling splits, nodata, class remapping, grid validation, mosaics, clipping, output protection, database-export helpers, and command help/import behavior. With TensorFlow installed they also build and reload a small ResNet, and run one training epoch through the preparation → training → prediction commands on a tiny synthetic raster. Model tests are skipped when TensorFlow is absent. GitHub Actions installs the CPU model dependencies to run those checks.

Use one worker and one library thread initially on a shared server. Raster operations set a 256 MiB GDAL cache; that is **not** a total process-memory limit. On the shared WBEO host, use the existing `raster-run` wrapper for real processing jobs. Do not run a continent-wide job merely to test installation.

## License

[GNU GPL v3](LICENSE), unchanged from the original repository.
