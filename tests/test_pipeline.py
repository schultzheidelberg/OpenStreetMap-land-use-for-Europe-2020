"""Tiny CPU smoke test through the user-facing preparation/training/prediction CLI."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_train_predict_pipeline(tmp_path):
    pytest.importorskip("tensorflow")
    source, labels = tmp_path / "image.tif", tmp_path / "labels.tif"
    profile = dict(
        driver="GTiff",
        height=6,
        width=8,
        count=2,
        dtype="uint16",
        crs="EPSG:3035",
        transform=from_origin(4000000, 3000000, 10, 10),
        nodata=0,
    )
    with rasterio.open(source, "w", **profile) as target:
        target.write(np.arange(1, 97, dtype="uint16").reshape(2, 6, 8))
    with rasterio.open(labels, "w", **{**profile, "count": 1}) as target:
        target.write(np.tile(np.array([11] * 4 + [5] * 4, dtype="uint16"), (6, 1)), 1)
    train, validation = tmp_path / "train.npz", tmp_path / "validation.npz"
    model, output = tmp_path / "model.keras", tmp_path / "classes.tif"
    environment = {
        **os.environ,
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "GDAL_CACHEMAX": "256",
        "GDAL_NUM_THREADS": "1",
    }
    commands = [
        [
            "01_preproce_prepare_sample.py",
            "--image",
            source,
            "--labels",
            labels,
            "--train-output",
            train,
            "--validation-output",
            validation,
            "--samples-per-class",
            "4",
            "--seed",
            "3",
        ],
        [
            "02_keras_models_single.py",
            "--train",
            train,
            "--validation",
            validation,
            "--output-model",
            model,
            "--epochs",
            "1",
            "--depth",
            "11",
            "--batch-size",
            "2",
        ],
        [
            "03_keras_classification.py",
            "--input",
            source,
            "--output",
            output,
            "--model",
            model,
            "--batch-size",
            "8",
            "--window-size",
            "4",
        ],
    ]
    for script, *arguments in commands:
        result = subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, arguments)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    with rasterio.open(output) as predicted:
        assert predicted.shape == (6, 8)
        assert predicted.dtypes == ("uint16",)
        assert predicted.nodata == 0
        assert predicted.transform == profile["transform"]
        assert predicted.crs.to_epsg() == 3035
        assert np.all(predicted.read(1) <= 12)
    assert model.is_file()
