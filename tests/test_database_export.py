"""Geometry/export regressions; these tests never connect to a database."""

import pytest

import shp_to_lulc as exporter


def test_fractional_negative_bbox_is_covered_without_overlap():
    boxes = exporter.split_bbox(-4.2, -0.3, 0.1, 4.1)
    assert len(boxes) == 4
    assert boxes[-1] == pytest.approx((-0.2, 3.7, 0.1, 4.1))
    assert sum(
        (right - left) * (top - bottom) for left, bottom, right, top in boxes
    ) == pytest.approx(4.3 * 4.4)
    assert all(right > left and top > bottom for left, bottom, right, top in boxes)


def test_exact_multiple_has_no_degenerate_tiles():
    assert exporter.split_bbox(0, 0, 8, 4) == [(0, 0, 4, 4), (4, 0, 8, 4)]


@pytest.mark.parametrize("box", [(1, 0, 0, 1), (0, 0, 1, 0), (0, 0, float("nan"), 1)])
def test_bad_bbox_fails(box):
    with pytest.raises(ValueError):
        exporter.split_bbox(*box)


def test_server_cursor_and_parameterized_query(monkeypatch, tmp_path):
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def execute(self, query, params):
            self.query, self.params = query, params

        def __iter__(self):
            yield b"geometry", "forest"

    class Connection:
        def cursor(self, *, name):
            self.name = name
            self.opened = Cursor()
            return self.opened

    connection = Connection()
    calls = []
    monkeypatch.setattr(
        exporter, "export_to_shape", lambda rows, output: calls.append((list(rows), output)) or 1
    )
    output = exporter.export_extents(connection, [("tile", (-1, -2, 3, 4))], tmp_path, 7)
    assert output == [tmp_path / "tile.shp"]
    assert connection.opened.closed
    assert connection.opened.itersize == 7
    assert connection.opened.params[:4] == (-1, -2, 3, 4)
    assert connection.opened.query.count("%s") == 5
    assert calls[0][0] == [(b"geometry", "forest")]


def test_real_shape_export_preserves_projection_and_class(tmp_path):
    ogr = pytest.importorskip("osgeo.ogr")
    geometry = ogr.CreateGeometryFromWkt("MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)))")
    output = tmp_path / "forest.shp"
    assert exporter.export_to_shape([(geometry.ExportToWkb(), "forest")], output) == 1
    dataset = ogr.Open(str(output))
    layer = dataset.GetLayer()
    assert layer.GetSpatialRef().GetAuthorityCode(None) == "3857"
    feature = layer.GetNextFeature()
    assert feature["class"] == "forests"
    assert feature["code"] == "3.1"
    dataset = None
    with pytest.raises(FileExistsError):
        exporter.export_to_shape([], output)


def test_failed_export_does_not_publish_partial_files(tmp_path):
    pytest.importorskip("osgeo.ogr")
    with pytest.raises(ValueError):
        exporter.export_to_shape([(b"bad geometry", "forest")], tmp_path / "broken.shp")
    assert list(tmp_path.iterdir()) == []


def test_extent_names_include_input_stem(tmp_path):
    ogr = pytest.importorskip("osgeo.ogr")
    from osgeo import osr

    crs = osr.SpatialReference()
    crs.ImportFromEPSG(4326)
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.shp"
        paths.append(path)
        dataset = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(str(path))
        layer = dataset.CreateLayer(name, crs, ogr.wkbPolygon)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"))
        layer.CreateFeature(feature)
        dataset = None
    names = [name for name, _ in exporter.iter_extents(paths)]
    assert names == ["a_feature_1_tile_1", "b_feature_1_tile_1"]


def test_extent_partition_is_lazy_for_many_small_tiles():
    from itertools import islice

    boxes = exporter.iter_bbox(-180, -90, 180, 90, size=0.001)
    first = list(islice(boxes, 2))
    assert len(first) == 2
    assert first[0] == pytest.approx((-180, -90, -179.999, -89.999))
