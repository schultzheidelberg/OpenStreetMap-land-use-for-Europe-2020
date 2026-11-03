"""Windowed inference for the OSM4EO 13-output patch classifier."""

import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .common import iter_windows, normalization_stats, normalize_image, patch_padding
from .outputs import atomic_raster_output


def load_predictor(model_path):
    """Load TensorFlow only when inference is requested, with one CPU thread."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["GDAL_NUM_THREADS"] = "1"
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for prediction; install the training dependencies"
        ) from error
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf.keras.models.load_model(model_path, compile=False)


def _validate_model_shape(model, patch_size, bands):
    shape = getattr(model, "input_shape", None)
    if shape is not None:
        if isinstance(shape, list) or len(shape) != 4:
            raise ValueError("Expected a model with one channels-last patch input")
        expected = (patch_size, patch_size, bands)
        if any(actual is not None and actual != want for actual, want in zip(shape[1:], expected)):
            raise ValueError(f"Model input shape {shape} does not match patches {expected}")


def _read_window(source, window, minimum, maximum, left, right, legacy_patches):
    """Read only a tile and its halo, preserving the global image boundary."""
    row, col = int(window.row_off), int(window.col_off)
    height, width = int(window.height), int(window.width)
    top, bottom = max(0, row - left), min(source.height, row + height + right)
    start, end = max(0, col - left), min(source.width, col + width + right)
    raw = source.read(window=Window(start, top, end - start, bottom - top), masked=True)
    valid = (~np.ma.getmaskarray(raw) & np.isfinite(np.ma.getdata(raw))).all(axis=0)
    valid = valid[row - top : row - top + height, col - start : col - start + width]
    image = np.moveaxis(normalize_image(raw, minimum, maximum), 0, -1)
    padding = (
        (max(0, left - row), max(0, row + height + right - source.height)),
        (max(0, left - col), max(0, col + width + right - source.width)),
        (0, 0),
    )
    image = np.pad(image, padding, mode="symmetric" if legacy_patches else "constant")
    return image, valid


def classify_raster(
    input_path,
    output_path,
    model,
    *,
    patch_size=12,
    batch_size=256,
    window_size=128,
    legacy_patches=False,
    overwrite=False,
):
    """Classify explicit input using bounded spatial windows and patch batches.

    Scaling uses one global min/max over finite, unmasked source values, shared
    with sampling. Invalid centers receive nodata=0; invalid neighbors are zero
    after normalization. Model argmax IDs remain 0..12, without an offset.

    ``legacy_patches`` reproduces the former symmetric padding and patch origin;
    masked normalization remains corrected, so it is not a bit-for-bit legacy run.
    """
    if batch_size < 1 or window_size < 1:
        raise ValueError("Batch size and window size must be positive")
    left, right = patch_padding(patch_size)
    if legacy_patches:
        left, right = patch_size // 2, (patch_size - 1) // 2
    with atomic_raster_output([input_path], output_path, overwrite) as temporary:
        with rasterio.Env(GDAL_CACHEMAX=256 * 1024 * 1024, GDAL_NUM_THREADS="1"):
            with rasterio.open(input_path) as source:
                _validate_model_shape(model, patch_size, source.count)
                minimum, maximum = normalization_stats(source, window_size)
                profile = dict(
                    driver="GTiff",
                    width=source.width,
                    height=source.height,
                    count=1,
                    dtype="uint16",
                    nodata=0,
                    crs=source.crs,
                    transform=source.transform,
                    compress="deflate",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    BIGTIFF="IF_SAFER",
                )
                with rasterio.open(temporary, "w", **profile) as destination:
                    destination.set_band_description(
                        1, "OSM4EO class ID (0=nodata, 1..12=land use)"
                    )
                    destination.update_tags(
                        patch_size=patch_size,
                        patch_layout="legacy" if legacy_patches else "training",
                        normalization_min=minimum,
                        normalization_max=maximum,
                    )
                    for window in iter_windows(source, window_size):
                        image, valid = _read_window(
                            source, window, minimum, maximum, left, right, legacy_patches
                        )
                        result = np.zeros(valid.shape, dtype=np.uint16)
                        locations = np.flatnonzero(valid)
                        for offset in range(0, len(locations), batch_size):
                            selected = locations[offset : offset + batch_size]
                            rows, cols = np.unravel_index(selected, valid.shape)
                            patches = np.empty(
                                (len(selected), patch_size, patch_size, source.count),
                                dtype=np.float32,
                            )
                            for index, (row, col) in enumerate(zip(rows, cols)):
                                patches[index] = image[
                                    row : row + patch_size, col : col + patch_size
                                ]
                            scores = np.asarray(model.predict(patches, verbose=0))
                            if scores.shape != (len(selected), 13) or not np.isfinite(scores).all():
                                raise ValueError(
                                    "The model must return finite scores with shape (batch, 13)"
                                )
                            result[rows, cols] = scores.argmax(axis=1).astype(np.uint16)
                        destination.write(result, 1, window=window)
    return Path(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input Sentinel-2 GeoTIFF")
    parser.add_argument("--output", required=True, type=Path, help="Output class GeoTIFF (0..12)")
    parser.add_argument(
        "--model", required=True, type=Path, help="Trusted Keras model (.h5 or .keras)"
    )
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument(
        "--legacy-patches",
        action="store_true",
        help="Use historical symmetric padding/patch origin; nodata handling stays corrected",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.model.resolve() == args.output.resolve() or (
        args.model.exists() and args.output.exists() and args.model.samefile(args.output)
    ):
        parser.error("Output must differ from the input model file")
    classify_raster(
        args.input,
        args.output,
        load_predictor(args.model),
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        window_size=args.window_size,
        legacy_patches=args.legacy_patches,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
