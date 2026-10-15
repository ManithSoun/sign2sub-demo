"""Sign2Sub API.

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import logging
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import schemas
from .config import settings
from .preprocess import segment_to_array
from .recognizer import TemplateRecognizer, build_recognizer
from .sentence import build_sentence_builder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sign2sub")

recognizer = build_recognizer(settings)
sentence_builder = build_sentence_builder(settings)
log.info("Recognizer=%s  SentenceBuilder=%s  threshold=%.2f",
         recognizer.name, sentence_builder.name, settings.confidence_threshold)

app = FastAPI(title="Sign2Sub API", version="0.1.0",
              description="Sign language keypoints in, subtitles out.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_key(x_api_key: str | None = Header(default=None)):
    keys = settings.api_key_set
    if keys and x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


def collapse_repeats(words: list[str]) -> list[str]:
    """One sign split into several segments gives "YOU YOU" -> keep one."""
    out = []
    for w in words:
        if not out or out[-1] != w:
            out.append(w)
    return out


def recognize(segment: schemas.Segment) -> schemas.WordPrediction:
    array = segment_to_array(segment.frames, recognizer.cfg)
    top = recognizer.predict(array, top_k=3)
    word, conf = top[0]
    ok = conf >= settings.confidence_threshold
    return schemas.WordPrediction(word=word if ok else None, confidence=round(conf, 4),
                                  accepted=ok, top=[(w, round(p, 4)) for w, p in top])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/info", response_model=schemas.InfoResponse)
def info():
    """What the frontend needs to know: which landmarks to send and how many frames."""
    return schemas.InfoResponse(
        recognizer=recognizer.name,
        sentence_builder=sentence_builder.name,
        confidence_threshold=settings.confidence_threshold,
        seq_len=recognizer.cfg.seq_len,
        features=recognizer.cfg.features,
        vocab_size=len(recognizer.labels),
        api_key_required=bool(settings.api_key_set),
        can_teach=isinstance(recognizer, TemplateRecognizer),
    )


@app.get("/v1/vocab")
def vocab():
    return {"words": recognizer.labels}


@app.post("/v1/recognize", response_model=schemas.WordPrediction, dependencies=[Depends(check_key)])
def recognize_word(segment: schemas.Segment):
    """One signed word (frames between two pauses) -> word + confidence."""
    t = time.perf_counter()
    pred = recognize(segment)
    log.info("recognize %d frames -> %s (%.2f, %s) in %.0f ms", len(segment.frames), pred.top[0][0],
             pred.confidence, "kept" if pred.accepted else "dropped", (time.perf_counter() - t) * 1000)
    return pred


@app.post("/v1/sentence", response_model=schemas.SentenceResponse, dependencies=[Depends(check_key)])
def build_sentence(req: schemas.SentenceRequest):
    """List of glosses -> natural sentence."""
    words = collapse_repeats(req.words)
    return schemas.SentenceResponse(words=words, sentence=sentence_builder.build(words),
                                    builder=sentence_builder.name)


@app.post("/v1/translate", response_model=schemas.TranslateResponse, dependencies=[Depends(check_key)])
def translate(req: schemas.TranslateRequest):
    """Partner endpoint: several word segments -> full subtitle in one call."""
    preds = [recognize(s) for s in req.segments]
    words = collapse_repeats([p.word for p in preds if p.accepted])
    return schemas.TranslateResponse(predictions=preds, words=words,
                                     sentence=sentence_builder.build(words))


# ---- Teach signs (only with RECOGNIZER=template) ----

def _store():
    if not isinstance(recognizer, TemplateRecognizer):
        raise HTTPException(status_code=400, detail="Teaching only works with RECOGNIZER=template")
    return recognizer.store


@app.get("/v1/templates", dependencies=[Depends(check_key)])
def list_templates():
    """Words you've taught and how many examples each has."""
    return {"counts": _store().counts()}


@app.post("/v1/templates/{word}", response_model=schemas.TeachResponse, dependencies=[Depends(check_key)])
def add_template(word: str, segment: schemas.Segment):
    """Save one recorded example of a sign."""
    store = _store()
    word = word.strip().upper()
    if not word:
        raise HTTPException(status_code=422, detail="Word is empty")
    feats = recognizer.features(segment_to_array(segment.frames, recognizer.cfg))
    n = store.add(word, feats)
    log.info("taught %s (example %d, %d frames)", word, n, len(segment.frames))
    return schemas.TeachResponse(word=word, examples=n, counts=store.counts())


@app.delete("/v1/templates/{word}", dependencies=[Depends(check_key)])
def delete_template(word: str, last_only: bool = False):
    """Delete a word (or just its last example with ?last_only=true)."""
    store = _store()
    word = word.strip().upper()
    idx = (store.counts().get(word, 1) - 1) if last_only else None
    store.remove(word, idx)
    return {"counts": store.counts()}
