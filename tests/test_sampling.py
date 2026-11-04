"""Tiny synthetic regressions for sample selection and raster patch semantics."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from osm4eo.common import normalization_stats, patch_padding
from osm4eo.sampling import prepare_samples, read_patch, remap_labels


def write_raster(path, values, *, nodata=None, transform=None):
    if values.ndim == 2:
        values = values[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[1],
        width=values.shape[2],
        count=len(values),
        dtype=values.dtype,
        crs="EPSG:3035",
        transform=transform or from_origin(100, 200, 10, 10),
        nodata=nodata,
    ) as destination:
        destination.write(values)
    return path


def test_remap_keeps_overlapping_codes_distinct_and_masks_nodata():
    codes = np.ma.array([5, 12, 41, 11, 21, 0, 999, 13], mask=[False] * 7 + [True])
    np.testing.assert_array_equal(remap_labels(codes), [12, 2, 11, 1, 5, 0, 0, 0])


@pytest.mark.parametrize("size,target", [(3, 1), (12, 5)])
def test_odd_and_even_patches_align_target_and_zero_pad(tmp_path, size, target):
    path = write_raster(tmp_path / "image.tif", np.arange(25, dtype=np.float32).reshape(5, 5))
    with rasterio.open(path) as image:
        patch = read_patch(image, 0, 0, size, 0, 24)
    assert patch.shape == (size, size, 1)
    assert patch[target, target, 0] == -1
    assert patch[target + 1, target + 1, 0] == pytest.approx(-0.5)
    assert np.all(patch[:target] == 0)
    assert patch_padding(size) == (target, size - target - 1)


def test_sampling_splits_actual_counts_without_repeating_centers(tmp_path):
    pixels = np.arange(35, dtype=np.float32).reshape(5, 7)
    codes = np.array([11] * 10 + [12] * 3 + [5] * 2 + [13] + [0] * 19, dtype=np.uint8).reshape(5, 7)
    image = write_raster(tmp_path / "image.tif", pixels)
    labels = write_raster(tmp_path / "labels.tif", codes)
    outputs = [tmp_path / name for name in ("train.npz", "val.npz", "train2.npz", "val2.npz")]
    count = prepare_samples(
        image, labels, *outputs[:2], samples_per_class=20, patch_size=3, seed=19, window_size=2
    )
    assert count == (12, 4)
    prepare_samples(
        image, labels, *outputs[2:], samples_per_class=20, patch_size=3, seed=19, window_size=2
    )
    with np.load(outputs[0], allow_pickle=False) as train, np.load(outputs[1]) as val:
        train_centers = {tuple(point) for point in train["coordinates"]}
        val_centers = {tuple(point) for point in val["coordinates"]}
        assert len(train_centers) == 12
        assert len(val_centers) == 4
        assert train_centers.isdisjoint(val_centers)
        assert sorted(train["labels"].tolist()) == [1] * 8 + [2] * 2 + [3, 12]
        assert sorted(val["labels"].tolist()) == [1, 1, 2, 12]
        assert train["patches"].dtype == np.float32
        with np.load(outputs[2]) as repeated:
            np.testing.assert_array_equal(train["coordinates"], repeated["coordinates"])
            np.testing.assert_array_equal(train["patches"], repeated["patches"])
    with pytest.raises(FileExistsError):
        prepare_samples(image, labels, *outputs[:2], patch_size=3)


def test_sampling_excludes_invalid_image_centers(tmp_path):
    pixels = np.array([[1, 2], [3, -9999]], dtype=np.float32)
    image = write_raster(tmp_path / "image.tif", pixels, nodata=-9999)
    labels = write_raster(tmp_path / "labels.tif", np.full((2, 2), 11, dtype=np.uint8))
    outputs = [tmp_path / "train.npz", tmp_path / "val.npz"]
    assert prepare_samples(image, labels, *outputs, patch_size=3) == (2, 1)
    for output in outputs:
        with np.load(output) as data:
            assert not np.any(np.all(data["coordinates"] == [1, 1], axis=1))
            assert data["normalization_min"] == 1
            assert data["normalization_max"] == 3


def test_misaligned_labels_fail_before_output(tmp_path):
    image = write_raster(tmp_path / "image.tif", np.ones((3, 3), dtype=np.uint8))
    labels = write_raster(
        tmp_path / "labels.tif",
        np.full((3, 3), 11, dtype=np.uint8),
        transform=from_origin(110, 200, 10, 10),
    )
    with pytest.raises(ValueError, match="identical CRS"):
        prepare_samples(image, labels, tmp_path / "train.npz", tmp_path / "val.npz")
    assert not (tmp_path / "train.npz").exists()


def test_statistics_only_use_bounded_windows_and_ignore_nonfinite(tmp_path):
    pixels = np.array([[1, -9999, 4], [np.nan, 2, 3]], dtype=np.float32)
    path = write_raster(tmp_path / "image.tif", pixels, nodata=-9999)
    with rasterio.open(path) as source:

        class BoundedReader:
            width, height = source.width, source.height

            def read(self, *, window, masked):
                assert window.width <= 2 and window.height <= 2
                return source.read(window=window, masked=masked)

        assert normalization_stats(BoundedReader(), window_size=2) == (1, 4)
