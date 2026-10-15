"""Request/response shapes for the API.

A "frame" is one video frame of MediaPipe Holistic landmarks.
Each part is a list of [x, y, z] (pose can also have visibility as a 4th value),
or null when that part isn't detected in the frame.
"""
from pydantic import BaseModel, Field

Point = list[float]


class Frame(BaseModel):
    pose: list[Point] | None = None        # 33 points
    face: list[Point] | None = None        # 468 points
    left_hand: list[Point] | None = None   # 21 points
    right_hand: list[Point] | None = None  # 21 points


class Segment(BaseModel):
    """One signed word: the frames between a pause and the next pause."""
    frames: list[Frame] = Field(min_length=1)


class WordPrediction(BaseModel):
    word: str | None              # None if filtered out by the confidence check
    confidence: float
    accepted: bool
    top: list[tuple[str, float]]  # top-k candidates, for debugging / the demo UI


class SentenceRequest(BaseModel):
    words: list[str]


class SentenceResponse(BaseModel):
    words: list[str]
    sentence: str
    builder: str


class TranslateRequest(BaseModel):
    segments: list[Segment] = Field(min_length=1)


class TranslateResponse(BaseModel):
    predictions: list[WordPrediction]
    words: list[str]
    sentence: str


class InfoResponse(BaseModel):
    recognizer: str
    sentence_builder: str
    confidence_threshold: float
    seq_len: int
    features: list[str]
    vocab_size: int
    api_key_required: bool
    can_teach: bool = False  # True when RECOGNIZER=template


class TeachResponse(BaseModel):
    word: str
    examples: int
    counts: dict[str, int]