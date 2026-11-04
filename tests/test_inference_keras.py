"""Real Keras parity check for native batch dispatch, with no timing assertions."""

import numpy as np
import pytest

from osm4eo.inference import _predict_batch


def test_native_batch_scores_match_keras_prediction():
    tf = pytest.importorskip("tensorflow")
    tf.keras.utils.set_random_seed(4)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((12, 12, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(13, activation="softmax"),
        ]
    )
    patches = np.random.default_rng(4).normal(size=(3, 12, 12, 2)).astype(np.float32)
    expected = model.predict(patches, verbose=0)
    actual = _predict_batch(model, patches)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (3, 13)
    tf.keras.backend.clear_session()
