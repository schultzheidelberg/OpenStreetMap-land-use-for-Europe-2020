"""Tiny synthetic regressions for geospatial and categorical correctness."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_geom

from osm4eo import postprocess


def write_raster(path, values, *, transform=None, crs="EPSG:3035", nodata=0):
    values = np.asarray(values)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=transform or from_origin(100, 200, 10, 10),
        nodata=nodata,
    ) as target:
        target.write(values, 1)
    return path


def polygon(left, bottom, right, top):
    return {
        "type": "Polygon",
        "coordinates": [
            [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
        ],
    }


def test_extract_preserves_grid_dtype_and_nodata(tmp_path, monkeypatch):
    monkeypatch.setattr(postprocess, "WINDOW_SIZE", 16)
    values = np.full((18, 19), 300, dtype="uint16")
    values[2, 3] = 12
    values[17, 18] = 12
    values[0, 0] = 65535
    source = write_raster(tmp_path / "input.tif", values, nodata=65535)
    output = tmp_path / "output.tif"
    postprocess.extract_lu(source, output, 12)
    with rasterio.open(output) as result, rasterio.open(source) as original:
        expected = np.full_like(values, 65535)
        expected[2, 3] = expected[17, 18] = 12
        np.testing.assert_array_equal(result.read(1), expected)
        assert result.transform == original.transform
        assert result.crs == original.crs
        assert result.dtypes == ("uint16",)
        assert result.nodata == 65535
        assert result.count == 1


def test_mask_rejects_same_shape_shifted_grid(tmp_path):
    source = write_raster(tmp_path / "input.tif", np.ones((2, 2), dtype="uint8"))
    mask = write_raster(
        tmp_path / "mask.tif",
        np.ones((2, 2), dtype="uint8"),
        transform=from_origin(110, 200, 10, 10),
    )
    output = tmp_path / "output.tif"
    with pytest.raises(ValueError, match="pixel grid"):
        postprocess.clip_extent(source, mask, output)
    assert not output.exists()


def test_mask_excludes_invalid_source_and_mask_pixels(tmp_path):
    source = write_raster(
        tmp_path / "input.tif", np.array([[12, 255], [3, 4]], dtype="uint8"), nodata=255
    )
    mask = write_raster(
        tmp_path / "mask.tif", np.array([[1, 1], [255, 0]], dtype="uint8"), nodata=255
    )
    output = tmp_path / "output.tif"
    postprocess.clip_extent(source, mask, output)
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read(1), [[12, 255], [255, 255]])


def test_label_transfer_is_simultaneous_and_preserves_nodata(tmp_path):
    source = write_raster(tmp_path / "input.tif", np.arange(13, dtype="uint8").reshape(1, 13))
    output = tmp_path / "output.tif"
    postprocess.transfer_labels(source, output)
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(
            result.read(1), [[0, 11, 12, 13, 14, 21, 22, 23, 31, 32, 33, 41, 5]]
        )
        assert result.nodata == 0


def test_label_transfer_moves_conflicting_nodata_marker(tmp_path):
    source = write_raster(tmp_path / "input.tif", np.array([[1, 11, 12]], dtype="uint8"), nodata=11)
    output = tmp_path / "output.tif"
    postprocess.transfer_labels(source, output)
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read(1), [[11, 0, 5]])
        assert result.nodata == 0


def test_sieve_preserves_nodata_and_removes_small_island(tmp_path):
    pytest.importorskip("osgeo.gdal")
    values = np.ones((5, 5), dtype="uint8")
    values[2, 2] = 2
    values[0, 0] = 255
    source = write_raster(tmp_path / "input.tif", values, nodata=255)
    output = tmp_path / "output.tif"
    postprocess.sieve_raster(source, output, size=2)
    with rasterio.open(output) as result:
        expected = np.ones_like(values)
        expected[0, 0] = 255
        np.testing.assert_array_equal(result.read(1), expected)
        assert result.read_masks(1)[0, 0] == 0
        assert result.nodata == 255


def test_mosaic_preserves_uint16_values_and_uses_first_overlap(tmp_path):
    first = write_raster(tmp_path / "first.tif", np.array([[300, 0], [300, 300]], dtype="uint16"))
    second = write_raster(
        tmp_path / "second.tif",
        np.full((2, 2), 400, dtype="uint16"),
        transform=from_origin(110, 200, 10, 10),
    )
    output = tmp_path / "output.tif"
    postprocess.merge_extents([first, second], output)
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read(1), [[300, 400, 400], [300, 300, 400]])
        assert result.dtypes == ("uint16",)
        assert result.transform == from_origin(100, 200, 10, 10)


def test_mosaic_rejects_subpixel_offset(tmp_path):
    first = write_raster(tmp_path / "first.tif", np.ones((2, 2), dtype="uint8"))
    second = write_raster(
        tmp_path / "second.tif",
        np.ones((2, 2), dtype="uint8"),
        transform=from_origin(105, 200, 10, 10),
    )
    with pytest.raises(ValueError, match="same pixel grid"):
        postprocess.merge_extents([first, second], tmp_path / "output.tif")


def test_geojson_clip_includes_every_feature(tmp_path):
    source = write_raster(
        tmp_path / "input.tif",
        np.ones((4, 4), dtype="uint8"),
        transform=from_origin(0, 4, 1, 1),
        crs="EPSG:4326",
    )
    geometry = tmp_path / "boundary.geojson"
    geometry.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {}, "geometry": polygon(0, 2, 1, 4)},
                    {"type": "Feature", "properties": {}, "geometry": polygon(3, 0, 4, 2)},
                ],
            }
        )
    )
    output = tmp_path / "output.tif"
    postprocess.clip_geojson(source, geometry, output)
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(
            result.read(1), [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
        )
        assert result.transform == from_origin(0, 4, 1, 1)


def test_geojson_clip_reprojects_wgs84_to_raster_crs(tmp_path):
    source = write_raster(
        tmp_path / "input.tif",
        np.array([[7, 8], [9, 10]], dtype="uint8"),
        transform=from_origin(0, 2000, 1000, 1000),
        crs="EPSG:3857",
    )
    geometry = tmp_path / "boundary.geojson"
    geometry.write_text(
        json.dumps(transform_geom("EPSG:3857", "EPSG:4326", polygon(1, 1001, 999, 1999)))
    )
    output = tmp_path / "output.tif"
    postprocess.clip_geojson(source, geometry, output)
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read(1), [[7]])
        assert result.crs.to_epsg() == 3857


def test_postprocessing_never_overwrites_input_or_existing_output(tmp_path):
    source = write_raster(tmp_path / "input.tif", np.array([[1, 2]], dtype="uint8"))
    before = source.read_bytes()
    with pytest.raises(ValueError, match="differ"):
        postprocess.extract_lu(source, source, 1, overwrite=True)
    output = tmp_path / "output.tif"
    output.write_text("keep me")
    with pytest.raises(FileExistsError):
        postprocess.extract_lu(source, output, 1)
    assert source.read_bytes() == before
    assert output.read_text() == "keep me"
