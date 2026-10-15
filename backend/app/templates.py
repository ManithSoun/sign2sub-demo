"""Template matching: recognise signs by comparing to examples you recorded.

No training needed. You record a few examples of each sign in the web demo
("Teach signs"), and live signing is matched to the closest example.
Good for a demo with a small, fixed set of words.

How it compares two signs:
  1. Keep only arms + hands (x, y), centred on the shoulders and scaled by
     shoulder width, so where you stand and how close you are don't matter.
  2. Dynamic Time Warping (DTW), so signing a bit faster or slower still matches.
"""
import json
import threading
from pathlib import Path

import numpy as np

from .preprocess import PART_SIZES, FeatureConfig

# Pose points used: nose, shoulders, elbows, wrists
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
MATCH_LEN = 24  # frames each sign is resampled to before comparing


def _part(vec: np.ndarray, cfg: FeatureConfig, part: str) -> np.ndarray | None:
    """Pull one body part out of a flat frame vector -> (n_points, dims)."""
    if part not in cfg.features:
        return None
    offset = 0
    for p in cfg.features:
        size = PART_SIZES[p] * cfg.dims(p)
        if p == part:
            return vec[offset: offset + size].reshape(PART_SIZES[p], cfg.dims(p))
        offset += size
    return None


def match_features(array: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """(frames, frame_size) model array -> (MATCH_LEN, 98) normalised arm+hand features."""
    out = []
    for vec in array:
        pose = _part(vec, cfg, "pose")
        if pose is None or not pose[11:13, :2].any():
            center, scale = np.array([0.5, 0.5]), 0.3
        else:
            l_sh, r_sh = pose[11, :2], pose[12, :2]
            center = (l_sh + r_sh) / 2
            scale = max(float(np.linalg.norm(l_sh - r_sh)), 1e-3)
        feats = []
        if pose is not None:
            feats.append((pose[POSE_IDX, :2] - center) / scale)
        else:
            feats.append(np.zeros((len(POSE_IDX), 2)))
        for hand in ("left_hand", "right_hand"):
            h = _part(vec, cfg, hand)
            if h is None or not h[:, :2].any():
                feats.append(np.zeros((21, 2)))  # hand not visible
            else:
                feats.append((h[:, :2] - center) / scale)
        out.append(np.concatenate(feats).flatten())
    seq = np.asarray(out, dtype=np.float32)
    # resample to a fixed length
    idx = np.linspace(0, len(seq) - 1, MATCH_LEN)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (idx - lo)[:, None]
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int = 6) -> float:
    """Average per-frame distance along the best time alignment."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2) / np.sqrt(a.shape[1])
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


class TemplateStore:
    """Recorded examples, saved to a JSON file so they survive restarts."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data: dict[str, list[np.ndarray]] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.data = {w: [np.asarray(e, dtype=np.float32) for e in ex] for w, ex in raw.items()}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {w: [e.round(4).tolist() for e in ex] for w, ex in self.data.items()}
        self.path.write_text(json.dumps(raw))

    def add(self, word: str, feats: np.ndarray) -> int:
        with self.lock:
            self.data.setdefault(word, []).append(feats)
            self._save()
            return len(self.data[word])

    def remove(self, word: str, index: int | None = None):
        with self.lock:
            if word not in self.data:
                return
            if index is None or len(self.data[word]) <= 1:
                del self.data[word]
            else:
                self.data[word].pop(index)
            self._save()

    def counts(self) -> dict[str, int]:
        return {w: len(ex) for w, ex in sorted(self.data.items())}

    def items(self):
        with self.lock:
            return [(w, list(ex)) for w, ex in self.data.items()]