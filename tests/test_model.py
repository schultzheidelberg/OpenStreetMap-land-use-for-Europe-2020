"""Optional TensorFlow smoke test; no training or project data required."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("suffix", [".keras", ".h5"])
def test_original_resnet_forward_and_serialization(tmp_path, suffix):
    tf = pytest.importorskip("tensorflow")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("training", root / "02_keras_models_single.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    model = training.resnet_v2((12, 12, 3), depth=29)
    inputs = np.zeros((1, 12, 12, 3), dtype=np.float32)
    before = model(inputs, training=False).numpy()
    assert before.shape == (1, 13)
    np.testing.assert_allclose(before.sum(axis=1), 1, rtol=1e-6)
    path = tmp_path / f"model{suffix}"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(restored(inputs, training=False).numpy(), before, rtol=1e-6)
    tf.keras.backend.clear_session()
