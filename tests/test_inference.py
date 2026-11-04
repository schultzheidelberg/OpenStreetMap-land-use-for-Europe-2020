"""Tiny, deterministic raster regressions without downloading TensorFlow/models."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from osm4eo import inference
from osm4eo.inference import classify_raster
from osm4eo.outputs import atomic_raster_output


def write_raster(path, values, nodata=-9999):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values[np.newaxis]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=values.shape[0],
        height=values.shape[1],
        width=values.shape[2],
        dtype="float32",
        nodata=nodata,
        crs="EPSG:3035",
        transform=from_origin(10, 50, 10, 10),
    ) as dst:
        dst.write(values)


class CenterPredictor:
    input_shape = (None, 12, 12, 2)

    def __init__(self, center=5):
        self.center = center
        self.batches = []

    def predict(self, patches, verbose=0):
        self.batches.append(patches.copy())
        # Reconstruct source IDs 1..12 from the normalized center. Wrong patch
        # origins or tile seams therefore become wrong output class IDs.
        labels = np.rint((patches[:, self.center, self.center, 0] + 1) * 5.5 + 1).astype(int)
        return np.eye(13, dtype=np.float32)[labels]


def test_native_batch_prediction_preserves_patch_bound_and_labels(tmp_path):
    source, output = tmp_path / "source.tif", tmp_path / "classes.tif"
    values = np.stack([np.arange(1, 13).reshape(3, 4)] * 2)
    write_raster(source, values)

    class BatchPredictor(CenterPredictor):
        def predict_on_batch(self, patches):
            return super().predict(patches)

        def predict(self, patches, verbose=0):
            pytest.fail("Native batch predictors should not rebuild predict iterators")

    model = BatchPredictor()
    classify_raster(source, output, model, batch_size=3, window_size=2)
    with rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read(1), values[0])
    assert model.batches and all(len(batch) <= 3 for batch in model.batches)


@pytest.mark.parametrize("legacy", [False, True])
def test_windowed_prediction_alignment_masks_and_metadata(tmp_path, legacy):
    source, output = tmp_path / "source.TIFF", tmp_path / "classes.tif"
    values = np.stack([np.arange(1, 13).reshape(3, 4)] * 2).astype(np.float32)
    values[0, 1, 1] = -9999
    values[1, 2, 1] = np.nan
    write_raster(source, values)
    model = CenterPredictor(center=6 if legacy else 5)
    classify_raster(source, output, model, window_size=2, batch_size=3, legacy_patches=legacy)
    expected = np.arange(1, 13).reshape(3, 4)
    expected[1, 1] = expected[2, 1] = 0
    with rasterio.open(source) as src, rasterio.open(output) as result:
        np.testing.assert_array_equal(result.read(1), expected)
        assert result.dtypes == ("uint16",)
        assert result.nodata == 0
        assert result.crs == src.crs
        assert result.transform == src.transform
        assert result.count == 1
        assert result.tags()["patch_layout"] == ("legacy" if legacy else "training")
        np.testing.assert_array_equal(result.read_masks(1) == 0, expected == 0)
    assert all(len(batch) <= 3 and batch.dtype == np.float32 for batch in model.batches)
    # Build expected global normalization independently.
    normalized = 2 * ((values - 1) / 11) - 1
    normalized[~np.isfinite(values) | (values == -9999)] = 0
    padding = 6 if legacy else 5
    expected_patch = np.pad(
        np.moveaxis(normalized, 0, -1),
        ((padding, 6), (padding, 6), (0, 0)),
        mode="symmetric" if legacy else "constant",
    )[:12, :12]
    np.testing.assert_allclose(model.batches[0][0], expected_patch)


def test_failed_prediction_preserves_previous_output_and_cleans_temporary(tmp_path):
    source, output = tmp_path / "source.tif", tmp_path / "classes.tif"
    write_raster(source, np.arange(1, 13).reshape(3, 4))
    output.write_bytes(b"previous result")

    class BrokenPredictor:
        def predict(self, patches, verbose=0):
            raise RuntimeError("model failed")

    with pytest.raises(FileExistsError):
        classify_raster(source, output, BrokenPredictor())
    with pytest.raises(RuntimeError, match="model failed"):
        classify_raster(source, output, BrokenPredictor(), overwrite=True)
    assert output.read_bytes() == b"previous result"
    assert set(tmp_path.iterdir()) == {source, output}


def test_reject_source_overwrite_and_wrong_model(tmp_path):
    source = tmp_path / "source.tif"
    write_raster(source, np.arange(1, 13).reshape(3, 4))
    with pytest.raises(ValueError, match="differ"):
        classify_raster(source, source, CenterPredictor(), overwrite=True)
    with pytest.raises(ValueError, match="input shape"):
        classify_raster(source, tmp_path / "wrong.tif", CenterPredictor())
    assert not (tmp_path / "wrong.tif").exists()


@pytest.mark.parametrize("scores", [np.ones((1, 12)), np.full((1, 13), np.nan)])
def test_reject_invalid_model_scores(tmp_path, scores):
    source, output = tmp_path / "source.tif", tmp_path / "classes.tif"
    write_raster(source, [[1]])

    class InvalidPredictor:
        def predict(self, patches, verbose=0):
            return scores

    with pytest.raises(ValueError, match="finite scores"):
        classify_raster(source, output, InvalidPredictor())
    assert not output.exists()


def test_all_nodata_rejected_without_output(tmp_path):
    source, output = tmp_path / "source.tif", tmp_path / "classes.tif"
    write_raster(source, [[-9999]])
    with pytest.raises(ValueError, match="no finite, valid"):
        classify_raster(source, output, object())
    assert set(tmp_path.iterdir()) == {source}


def test_atomic_publication_will_not_replace_concurrent_output(tmp_path):
    output = tmp_path / "classes.tif"
    with pytest.raises(FileExistsError):
        with atomic_raster_output([], output) as temporary:
            write_raster(temporary, [[1]])
            output.write_bytes(b"other job")
    assert output.read_bytes() == b"other job"
    assert set(tmp_path.iterdir()) == {output}


def test_atomic_output_rejects_stale_destination_masks(tmp_path):
    output = tmp_path / "classes.tif"
    Path(f"{output}.msk").write_bytes(b"old mask")
    with pytest.raises(ValueError, match="sidecars"):
        with atomic_raster_output([], output, overwrite=True):
            pytest.fail("Must reject before writing")


def test_inference_import_and_cli_help_do_not_import_tensorflow():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import osm4eo.inference; assert 'tensorflow' not in sys.modules",
        ],
        cwd=root,
        check=True,
    )
    result = subprocess.run(
        [sys.executable, "03_keras_classification.py", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--model" in result.stdout
    assert "--legacy-patches" in result.stdout


@pytest.mark.parametrize("hard_link", [False, True])
def test_cli_preserves_input_model_before_loading(tmp_path, monkeypatch, capsys, hard_link):
    model = tmp_path / "model.keras"
    model.write_bytes(b"trained model")
    output = tmp_path / "alias.tif" if hard_link else model
    if hard_link:
        output.hardlink_to(model)

    def unexpected_load(path):
        pytest.fail("Must reject model replacement before loading TensorFlow/model")

    monkeypatch.setattr(inference, "load_predictor", unexpected_load)
    with pytest.raises(SystemExit) as error:
        inference.main(
            [
                "--input",
                str(tmp_path / "image.tif"),
                "--model",
                str(model),
                "--output",
                str(output),
                "--overwrite",
            ]
        )
    assert error.value.code == 2
    assert "differ from the input model" in capsys.readouterr().err
    assert model.read_bytes() == b"trained model"
