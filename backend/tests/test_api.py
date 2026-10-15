import json

import numpy as np
from fastapi.testclient import TestClient

from app.main import app, recognizer, settings
from app.preprocess import FeatureConfig, resample, segment_to_array
from app.schemas import Frame
from app.sentence import RuleBuilder

client = TestClient(app)


def fake_frame(with_hands=True):
    pt = lambda: [0.5, 0.5, 0.0]
    return {
        "pose": [[0.5, 0.5, 0.0, 0.9] for _ in range(33)],
        "face": [pt() for _ in range(468)],
        "left_hand": [pt() for _ in range(21)] if with_hands else None,
        "right_hand": None,
    }


def test_health_and_info():
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/v1/info").json()
    assert info["recognizer"] == "mock"
    assert info["seq_len"] == 30
    assert info["vocab_size"] > 0


def test_feature_size_matches_default_holistic_layout():
    assert FeatureConfig().frame_size == 33 * 4 + 468 * 3 + 21 * 3 + 21 * 3 == 1662


def test_segment_is_resampled_to_seq_len():
    cfg = FeatureConfig()
    for n in (1, 12, 30, 75):
        arr = segment_to_array([Frame(**fake_frame()) for _ in range(n)], cfg)
        assert arr.shape == (30, 1662)
    assert resample(np.arange(4, dtype=np.float32)[:, None], 7)[-1, 0] == 3


def test_missing_parts_become_zeros():
    arr = segment_to_array([Frame()], FeatureConfig())
    assert not arr.any()


def test_recognize_returns_word_and_confidence():
    r = client.post("/v1/recognize", json={"frames": [fake_frame() for _ in range(20)]})
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] == (body["confidence"] >= settings.confidence_threshold)
    assert (body["word"] is None) == (not body["accepted"])


def test_low_confidence_words_are_dropped():
    results = [client.post("/v1/recognize", json={"frames": [fake_frame()]}).json() for _ in range(14)]
    assert any(not r["accepted"] for r in results), "mock should sometimes give low confidence"
    assert all(r["word"] is None for r in results if not r["accepted"])


def test_sentence_rule_builder():
    assert RuleBuilder().build(["I", "GO", "SCHOOL"]) == "I go school."
    assert RuleBuilder().build([]) == ""
    r = client.post("/v1/sentence", json={"words": ["MY", "MOTHER", "TEACHER"]}).json()
    assert r["sentence"] == "My mother teacher."


def test_translate_endpoint():
    seg = {"frames": [fake_frame() for _ in range(10)]}
    r = client.post("/v1/translate", json={"segments": [seg, seg, seg]}).json()
    assert len(r["predictions"]) == 3
    assert r["words"] == [p["word"] for p in r["predictions"] if p["accepted"]]


def test_empty_segment_rejected():
    assert client.post("/v1/recognize", json={"frames": []}).status_code == 422


def test_api_key_required_when_set(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "secret123")
    seg = {"frames": [fake_frame()]}
    assert client.post("/v1/recognize", json=seg).status_code == 401
    assert client.post("/v1/recognize", json=seg, headers={"X-API-Key": "secret123"}).status_code == 200


def test_labels_file_is_valid():
    words = json.loads(settings.labels_path.read_text())
    assert len(words) == len(set(words)) == len(recognizer.labels)
