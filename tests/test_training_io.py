"""Verify safe numeric samples, legacy opt-in, and categorical raster metadata."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from hsi_io import export_data, import_data, load_samples, savetif

ROOT = Path(__file__).resolve().parents[1]


def test_numeric_samples_and_legacy_pickle_opt_in(tmp_path):
    patches = np.arange(36, dtype=np.float32).reshape(2, 3, 3, 2)
    labels = np.array([1, 12], dtype=np.uint8)
    numeric = tmp_path / "samples.npz"
    np.savez(numeric, patches=patches, labels=labels)
    loaded_patches, loaded_labels = load_samples(numeric)
    np.testing.assert_array_equal(loaded_patches, patches)
    np.testing.assert_array_equal(loaded_labels, labels)
    legacy = np.empty((2, 2), dtype=object)
    legacy[:, 0] = [patches[0], patches[1]]
    legacy[:, 1] = labels
    old_path = tmp_path / "old.npy"
    np.save(old_path, legacy)
    with pytest.raises(ValueError, match="allow-legacy-pickle"):
        load_samples(old_path)
    actual, actual_labels = load_samples(old_path, allow_legacy_pickle=True)
    np.testing.assert_array_equal(actual, patches)
    np.testing.assert_array_equal(actual_labels, labels)


@pytest.mark.parametrize("labels", [[0, 1], [1, 13], [1, 1.5]])
def test_invalid_training_labels_fail(tmp_path, labels):
    path = tmp_path / "bad.npz"
    np.savez(path, patches=np.ones((2, 3, 3, 1)), labels=labels)
    with pytest.raises(ValueError, match="class IDs 1..12"):
        load_samples(path)


def test_training_import_has_no_tensorflow_or_data_access():
    code = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('training', '02_keras_models_single.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'tensorflow' not in sys.modules
assert 'hdf5storage' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_savetif_uses_class_dtype_and_nodata_preserving_grid(tmp_path):
    reference, output = tmp_path / "image.tif", tmp_path / "classes.tif"
    transform = from_origin(100, 200, 10, 10)
    with rasterio.open(
        reference,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=3,
        dtype="float32",
        nodata=-9999,
        crs="EPSG:3035",
        transform=transform,
    ) as source:
        source.write(np.ones((3, 2, 2), dtype=np.float32))
    classes = np.array([[1, 12], [0, 7]], dtype=np.int64)
    savetif(classes, reference, output)
    with rasterio.open(output) as actual:
        assert actual.count == 1
        assert actual.dtypes == ("uint16",)
        assert actual.nodata == 0
        assert actual.transform == transform
        assert actual.crs.to_epsg() == 3035
        np.testing.assert_array_equal(actual.read(1), classes)
    with pytest.raises(FileExistsError):
        savetif(classes, reference, output)
    with pytest.raises(ValueError):
        savetif(classes, reference, reference)


def test_legacy_multiband_text_roundtrip_does_not_append(tmp_path):
    values = np.arange(12).reshape(4, 3)
    path = tmp_path / "legacy.txt"
    export_data(path, values, 3)
    export_data(path, values, 3)
    bands, rows, cols, result = import_data(path)
    assert (bands, rows, cols) == (3, 4, 1)
    np.testing.assert_array_equal(result, values)


def test_learning_schedule_is_explicit_and_retains_historic_rates():
    spec = importlib.util.spec_from_file_location("training", ROOT / "02_keras_models_single.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    assert training.lr_schedule(0) == 0.01
    assert training.lr_schedule(81) == 0.001
    assert training.lr_schedule(151) == 0.0001
    assert training.lr_schedule(201) == 0.00001
    assert training.lr_schedule(301) == 0.000005


def test_metrics_include_zero_predictions_without_missing_class_division():
    spec = importlib.util.spec_from_file_location("training", ROOT / "02_keras_models_single.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    metrics = training.classification_metrics(np.array([1, 1, 12]), np.array([1, 0, 12]))
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["cohen_kappa"] == pytest.approx(0.5)
    assert metrics["class_recall"] == {"1": 0.5, "12": 1.0}
    assert metrics["mean_present_class_recall"] == 0.75


def _training_module():
    spec = importlib.util.spec_from_file_location("training", ROOT / "02_keras_models_single.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    return training


def _training_arguments(tmp_path, **changes):
    arguments = dict(
        output_model=tmp_path / "model.keras",
        train=tmp_path / "train.npz",
        validation=tmp_path / "validation.npz",
        epochs=1,
        batch_size=1,
        learning_rate=0.01,
        depth=29,
        overwrite=False,
    )
    arguments.update(changes)
    return SimpleNamespace(**arguments)


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), float("-inf"), 0, -0.01])
def test_training_rejects_invalid_learning_rate_before_data_loading(tmp_path, rate):
    with pytest.raises(ValueError, match="finite and positive"):
        _training_module().train_model(_training_arguments(tmp_path, learning_rate=rate))


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_training_rejects_same_data_under_all_paths(tmp_path, alias):
    train, validation = tmp_path / "train.npz", tmp_path / "validation.npz"
    train.write_bytes(b"not loaded: file identity must be checked first")
    if alias == "same":
        validation = train
    elif alias == "symlink":
        validation.symlink_to(train)
    else:
        os.link(train, validation)
    with pytest.raises(ValueError, match="Training and validation files must differ"):
        _training_module().train_model(_training_arguments(tmp_path, validation=validation))
