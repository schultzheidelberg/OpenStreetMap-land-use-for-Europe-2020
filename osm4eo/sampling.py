"""Reproducible, bounded-memory sampling of aligned image and OSM label rasters."""

import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .common import (
    assert_same_grid,
    iter_windows,
    normalization_stats,
    normalize_image,
    patch_padding,
)
from .labels import LABEL_MAP


def remap_labels(values):
    """Map original OSM codes in one pass; unknown/nodata labels become zero."""
    values = np.ma.asarray(values)
    result = np.zeros(values.shape, dtype=np.uint8)
    valid = ~np.ma.getmaskarray(values)
    for source, target in LABEL_MAP.items():
        result[valid & (values.data == source)] = target
    return result


def sample_coordinates(image, labels, samples_per_class, rng, window_size=512):
    """Uniformly select unique centers using bounded per-class priority reservoirs."""
    if samples_per_class < 1:
        raise ValueError("Samples per class must be positive")
    reservoirs = {label: (np.empty(0), np.empty((0, 2), dtype=np.int64)) for label in range(1, 13)}
    for window in iter_windows(labels, window_size):
        mapped = remap_labels(labels.read(1, window=window, masked=True))
        pixels = image.read(window=window, masked=True)
        valid = np.all(~np.ma.getmaskarray(pixels) & np.isfinite(pixels.data), axis=0)
        for label in range(1, 13):
            coords = np.argwhere((mapped == label) & valid)
            if not len(coords):
                continue
            coords += np.array([int(window.row_off), int(window.col_off)])
            old_keys, old_coords = reservoirs[label]
            keys = np.concatenate((old_keys, rng.random(len(coords))))
            coords = np.concatenate((old_coords, coords))
            if len(keys) > samples_per_class:
                keep = np.argpartition(keys, samples_per_class - 1)[:samples_per_class]
                keys, coords = keys[keep], coords[keep]
            reservoirs[label] = keys, coords
    return {
        label: coords[rng.permutation(len(coords))] for label, (_, coords) in reservoirs.items()
    }


def split_coordinates(by_class, train_fraction=0.8):
    """Split actual class counts; singletons go to training and missing classes skip."""
    if not 0 < train_fraction < 1:
        raise ValueError("Train fraction must be between 0 and 1")
    train, validation = [], []
    for label, coords in sorted(by_class.items()):
        count = len(coords)
        if not count:
            continue
        train_count = max(1, int(count * train_fraction))
        if count > 1:
            train_count = min(count - 1, train_count)
        train.extend((row, col, label) for row, col in coords[:train_count])
        validation.extend((row, col, label) for row, col in coords[train_count:])
    return np.asarray(train, dtype=np.int64).reshape(-1, 3), np.asarray(
        validation, dtype=np.int64
    ).reshape(-1, 3)


def read_patch(image, row, col, size, minimum, maximum):
    """Read only one patch and pad outside the raster with normalized zeros."""
    left, _ = patch_padding(size)
    row_start, col_start = row - left, col - left
    top, start = max(0, row_start), max(0, col_start)
    bottom, end = min(image.height, row_start + size), min(image.width, col_start + size)
    window = Window(start, top, end - start, bottom - top)
    values = normalize_image(image.read(window=window, masked=True), minimum, maximum)
    patch = np.zeros((size, size, image.count), dtype=np.float32)
    patch[top - row_start : bottom - row_start, start - col_start : end - col_start] = np.moveaxis(
        values, 0, -1
    )
    return patch


def extract_samples(image, coordinates, size, minimum, maximum):
    """Allocate only selected samples, never a full padded source image."""
    patches = np.empty((len(coordinates), size, size, image.count), dtype=np.float32)
    for index, (row, col, _) in enumerate(coordinates):
        patches[index] = read_patch(image, int(row), int(col), size, minimum, maximum)
    return {
        "patches": patches,
        "labels": coordinates[:, 2].astype(np.uint8),
        "coordinates": coordinates[:, :2],
    }


def save_samples(path, samples, *, overwrite=False):
    """Publish a numeric NPZ atomically; no object arrays or pickle needed."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Sample output must use the .npz extension")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **samples)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_samples(
    image_path,
    label_path,
    train_path,
    validation_path,
    *,
    samples_per_class=5000,
    patch_size=12,
    train_fraction=0.8,
    seed=42,
    window_size=512,
    overwrite=False,
):
    """Create explicit train/validation files with no repeated pixel centers."""
    patch_padding(patch_size)
    paths = [Path(p).resolve() for p in (image_path, label_path, train_path, validation_path)]
    if len(set(paths)) != 4:
        raise ValueError("Image, labels, and both outputs must use distinct paths")
    for path in paths[2:]:
        if path.suffix.lower() != ".npz":
            raise ValueError("Sample outputs must use the .npz extension")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
    rng = np.random.default_rng(seed)
    with rasterio.Env(GDAL_CACHEMAX=256 * 1024 * 1024, GDAL_NUM_THREADS="1"):
        with rasterio.open(image_path) as image, rasterio.open(label_path) as labels:
            assert_same_grid(image, labels)
            if labels.count != 1:
                raise ValueError("Label raster must contain exactly one band")
            minimum, maximum = normalization_stats(image, window_size)
            by_class = sample_coordinates(image, labels, samples_per_class, rng, window_size)
            train, validation = split_coordinates(by_class, train_fraction)
            if not len(train) or not len(validation):
                raise ValueError(
                    "Need at least one class with two valid labeled pixels for a split"
                )
            train = train[rng.permutation(len(train))]
            validation = validation[rng.permutation(len(validation))]
            metadata = {
                "normalization_min": minimum,
                "normalization_max": maximum,
                "seed": seed,
                "patch_size": patch_size,
                "window_size": window_size,
                "train_fraction": train_fraction,
                "samples_per_class": samples_per_class,
            }
            # Release the training patches before allocating validation patches.
            save_samples(
                train_path,
                {**extract_samples(image, train, patch_size, minimum, maximum), **metadata},
                overwrite=overwrite,
            )
            save_samples(
                validation_path,
                {
                    **extract_samples(image, validation, patch_size, minimum, maximum),
                    **metadata,
                },
                overwrite=overwrite,
            )
    return len(train), len(validation)
