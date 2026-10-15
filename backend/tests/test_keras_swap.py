"""Checks that a real Keras model plugs in. Skipped if TensorFlow isn't installed."""
import json
from types import SimpleNamespace

import pytest

tf = pytest.importorskip("tensorflow")

from app.preprocess import FeatureConfig, segment_to_array  # noqa: E402
from app.recognizer import KerasRecognizer, build_recognizer  # noqa: E402
from app.schemas import Frame  # noqa: E402

LABELS = ["HELLO", "SCHOOL", "MOTHER"]


def tiny_model(seq_len, n_feat, n_out):
    inp = tf.keras.Input((seq_len, n_feat))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8))(inp)
    out = tf.keras.layers.Dense(n_out, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def make_settings(tmp_path, n_feat=1662, n_out=3):
    (tmp_path / "labels.json").write_text(json.dumps(LABELS))
    (tmp_path / "model_config.json").write_text(json.dumps({"seq_len": 30}))
    tiny_model(30, n_feat, n_out).save(tmp_path / "m.keras")
    return SimpleNamespace(recognizer="keras", model_path=tmp_path / "m.keras",
                           labels_path=tmp_path / "labels.json",
                           model_config_path=tmp_path / "model_config.json", mock_script="")


def test_keras_model_loads_and_predicts(tmp_path):
    rec = build_recognizer(make_settings(tmp_path))
    arr = segment_to_array([Frame() for _ in range(17)], rec.cfg)
    top = rec.predict(arr, top_k=3)
    assert len(top) == 3 and top[0][0] in LABELS
    assert abs(sum(p for _, p in top) - 1) < 1e-4


def test_wrong_feature_size_gives_clear_error(tmp_path):
    with pytest.raises(ValueError, match="features per frame"):
        build_recognizer(make_settings(tmp_path, n_feat=225))


def test_wrong_label_count_gives_clear_error(tmp_path):
    with pytest.raises(ValueError, match="labels.json"):
        build_recognizer(make_settings(tmp_path, n_out=5))
