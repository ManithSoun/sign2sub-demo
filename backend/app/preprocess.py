"""Turn API frames into the exact array your model was trained on.

This MUST match how you extracted keypoints during training. Edit
models/model_config.json rather than this file:

{
  "seq_len": 30,                                     # frames per word
  "features": ["pose", "face", "left_hand", "right_hand"],
  "pose_visibility": true,                           # pose has 4 values (x,y,z,vis)
  "normalize": "none"                                # or "shoulder_center"
}

The default (all parts, pose with visibility, 30 frames) gives 1662 values per
frame, the most common MediaPipe Holistic + LSTM setup.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schemas import Frame

PART_SIZES = {"pose": 33, "face": 468, "left_hand": 21, "right_hand": 21}


@dataclass
class FeatureConfig:
    seq_len: int = 30
    features: list[str] = field(default_factory=lambda: ["pose", "face", "left_hand", "right_hand"])
    pose_visibility: bool = True
    normalize: str = "none"

    @classmethod
    def load(cls, path: Path) -> "FeatureConfig":
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()

    def dims(self, part: str) -> int:
        return 4 if (part == "pose" and self.pose_visibility) else 3

    @property
    def frame_size(self) -> int:
        return sum(PART_SIZES[p] * self.dims(p) for p in self.features)


def _part_array(points, n_points: int, dims: int) -> np.ndarray:
    if not points:
        return np.zeros(n_points * dims, dtype=np.float32)
    arr = np.zeros((n_points, dims), dtype=np.float32)
    for i, p in enumerate(points[:n_points]):
        vals = list(p)[:dims]
        if len(vals) < dims:  # e.g. pose sent without visibility
            vals += [1.0 if dims == 4 else 0.0] * (dims - len(vals))
        arr[i] = vals
    return arr.flatten()


def frame_to_vector(frame: Frame, cfg: FeatureConfig) -> np.ndarray:
    parts = []
    for part in cfg.features:
        parts.append(_part_array(getattr(frame, part), PART_SIZES[part], cfg.dims(part)))
    vec = np.concatenate(parts)
    if cfg.normalize == "shoulder_center" and frame.pose and len(frame.pose) > 12:
        # Center x/y on the midpoint between the shoulders (pose points 11 and 12),
        # so position in the camera frame doesn't matter.
        cx = (frame.pose[11][0] + frame.pose[12][0]) / 2
        cy = (frame.pose[11][1] + frame.pose[12][1]) / 2
        offset = 0
        for part in cfg.features:
            d = cfg.dims(part)
            n = PART_SIZES[part]
            block = vec[offset: offset + n * d].reshape(n, d)
            present = np.any(block[:, :3] != 0, axis=1)
            block[present, 0] -= cx
            block[present, 1] -= cy
            offset += n * d
    return vec


def resample(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Stretch or squeeze a (frames, features) array to exactly target_len frames."""
    n = seq.shape[0]
    if n == target_len:
        return seq
    if n == 1:
        return np.repeat(seq, target_len, axis=0)
    src = np.linspace(0, n - 1, target_len)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None]
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)


def segment_to_array(frames: list[Frame], cfg: FeatureConfig) -> np.ndarray:
    """Returns shape (seq_len, frame_size), ready for the model (add batch dim)."""
    seq = np.stack([frame_to_vector(f, cfg) for f in frames])
    return resample(seq, cfg.seq_len)
