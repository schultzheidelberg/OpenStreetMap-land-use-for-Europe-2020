"""Shared raster grid, normalization, and patch conventions."""

import numpy as np
from rasterio.windows import Window


def patch_padding(size):
    """Return padding with the target at index 5 for the historic 12px patch."""
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("Patch size must be a positive integer")
    return (size - 1) // 2, size // 2


def iter_windows(dataset, window_size=512):
    """Iterate explicit bounded windows, including small edge windows."""
    if window_size < 1:
        raise ValueError("Window size must be positive")
    for row in range(0, dataset.height, window_size):
        for col in range(0, dataset.width, window_size):
            yield Window(
                col,
                row,
                min(window_size, dataset.width - col),
                min(window_size, dataset.height - row),
            )


def normalization_stats(dataset, window_size=512):
    """Global extrema over valid finite pixels, without loading the full raster."""
    minimum, maximum = np.inf, -np.inf
    for window in iter_windows(dataset, window_size):
        data = dataset.read(window=window, masked=True)
        valid = data.compressed()
        valid = valid[np.isfinite(valid)]
        if valid.size:
            minimum = min(minimum, float(valid.min()))
            maximum = max(maximum, float(valid.max()))
    if not np.isfinite(minimum):
        raise ValueError("Raster contains no finite, valid pixels")
    return minimum, maximum


def normalize_image(values, minimum, maximum, eps=0.1):
    """Scale valid values to [-1, 1]; retain constant images and fill invalids 0."""
    data = np.ma.asarray(values, dtype=np.float32)
    valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
    result = np.zeros(data.shape, dtype=np.float32)
    if maximum - minimum > eps:
        result[valid] = 2 * ((data.data[valid] - minimum) / (maximum - minimum)) - 1
    else:
        result[valid] = data.data[valid]
    return result


def assert_same_grid(reference, other):
    """Require pixelwise alignment, not just matching coordinate systems."""
    if (
        reference.crs != other.crs
        or reference.transform != other.transform
        or reference.width != other.width
        or reference.height != other.height
    ):
        raise ValueError("Image and labels must have identical CRS, transform, and dimensions")
