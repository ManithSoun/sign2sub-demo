"""Model 1: sign recognition (one signed word -> word + confidence).

Every recognizer has the same method:
    predict(array) -> list of (word, probability), best first

so the rest of the app never cares which model is plugged in.
"""
import itertools
import json
import random
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .preprocess import FeatureConfig
from .templates import TemplateStore, dtw_distance, match_features


def load_labels(path: Path) -> list[str]:
    """labels.json can be a list ["HELLO", ...] or a dict {"0": "HELLO", ...}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data[k] for k in sorted(data, key=lambda k: int(k))]
    return list(data)


class Recognizer(ABC):
    name = "base"

    def __init__(self, labels: list[str], cfg: FeatureConfig):
        self.labels = labels
        self.cfg = cfg

    @abstractmethod
    def predict(self, array: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        """array has shape (seq_len, frame_size)."""


class MockRecognizer(Recognizer):
    """Pretends to be the model so the demo can be built before training finishes.

    It walks through MOCK_SCRIPT word by word ("I GO SCHOOL | ..."), and every
    7th word comes back with low confidence so you can see the confidence filter working.
    """
    name = "mock"

    def __init__(self, labels, cfg, script: str):
        super().__init__(labels, cfg)
        words = [w for w in script.replace("|", " ").split() if w]
        self._words = itertools.cycle(words or labels or ["HELLO"])
        self._lock = threading.Lock()
        self._rng = random.Random(7)
        self._count = 0

    def predict(self, array, top_k=3):
        with self._lock:
            word = next(self._words)
            self._count += 1
            low = self._count % 7 == 0
        conf = round(self._rng.uniform(0.3, 0.5) if low else self._rng.uniform(0.75, 0.98), 3)
        others = [w for w in self.labels if w != word][: top_k - 1]
        rest = (1 - conf) / max(len(others), 1)
        return [(word, conf)] + [(w, round(rest, 3)) for w in others]


class KerasRecognizer(Recognizer):
    """Loads your trained Keras model. Output must be softmax over labels.json."""
    name = "keras"

    def __init__(self, labels, cfg, model_path: Path):
        super().__init__(labels, cfg)
        import tensorflow as tf  # imported here so mock mode doesn't need TensorFlow

        self.model = tf.keras.models.load_model(model_path, compile=False)
        expected = self.model.input_shape
        if expected[-1] not in (None, cfg.frame_size):
            raise ValueError(
                f"Model expects {expected[-1]} features per frame but model_config.json "
                f"gives {cfg.frame_size}. Fix 'features'/'pose_visibility' to match training."
            )
        n_out = self.model.output_shape[-1]
        if n_out != len(labels):
            raise ValueError(f"Model has {n_out} outputs but labels.json has {len(labels)} words.")

    def predict(self, array, top_k=3):
        probs = self.model.predict(array[None, ...], verbose=0)[0]
        idx = np.argsort(probs)[::-1][:top_k]
        return [(self.labels[i], float(probs[i])) for i in idx]


class TemplateRecognizer(Recognizer):
    """Matches live signing to examples recorded in the web demo ("Teach signs").

    Confidence comes from how much closer the best word is than the others,
    and drops when even the best match is far away (an unknown sign).
    """
    name = "template"

    def __init__(self, cfg, store: TemplateStore, reject_distance: float, sharpness: float):
        super().__init__([], cfg)
        self.store = store
        self.reject_distance = reject_distance
        self.sharpness = sharpness

    @property
    def labels(self):
        return list(self.store.counts().keys())

    @labels.setter
    def labels(self, _):
        pass

    def features(self, array):
        return match_features(array, self.cfg)

    def predict(self, array, top_k=3):
        items = self.store.items()
        if not items:
            return [("(no signs taught yet)", 0.0)]
        q = self.features(array)
        best = {w: min(dtw_distance(q, e) for e in ex) for w, ex in items}
        words = sorted(best, key=best.get)
        d = np.array([best[w] for w in words])
        probs = np.exp(-(d - d[0]) * self.sharpness)
        probs /= probs.sum()
        # far from every example -> not confident, whatever the ranking says
        penalty = float(np.exp(-max(0.0, d[0] - self.reject_distance) * self.sharpness))
        conf = probs * penalty
        return [(w, float(c)) for w, c in zip(words[:top_k], conf[:top_k])]


def build_recognizer(settings) -> Recognizer:
    cfg = FeatureConfig.load(settings.model_config_path)
    if settings.recognizer == "template":
        store = TemplateStore(settings.templates_path)
        return TemplateRecognizer(cfg, store, settings.template_reject_distance, settings.template_sharpness)
    labels = load_labels(settings.labels_path)
    if settings.recognizer == "keras":
        return KerasRecognizer(labels, cfg, settings.model_path)
    if settings.recognizer == "mock":
        return MockRecognizer(labels, cfg, settings.mock_script)
    raise ValueError(f"Unknown RECOGNIZER={settings.recognizer!r} (use 'mock', 'template' or 'keras')")