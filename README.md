# Sign2Sub Demo

Live demo of Sign2Sub: sign in front of a webcam and get subtitles.

```
Webcam ─► MediaPipe Holistic (browser) ─► keypoints ─► Sign2Sub API
                                                      ├─ Model 1: sign → word (+ confidence)
                                                      ├─ Confidence filter (drop, never guess)
                                                      └─ Model 2: words → sentence
                                         ◄─ subtitle ─┘
```

- `backend/`: FastAPI. This is the API partners would license.
- `frontend/`: React + Vite web demo.

Everything already works with a **mock model**, so you can build, style and rehearse the
demo now. When training is done, you drop in the real model. You don't need to change any code.

---

## Run it

**Backend**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

**Frontend** (new terminal)
```bash
cd frontend
npm install          # also copies MediaPipe files into public/ so it works offline
npm run dev
```
Open http://localhost:5173 and click **Start camera**.

---

## Using the demo

- **Auto mode:** just sign. A word starts when your hands appear and ends when you pause
  (hold still for a moment) or drop your hands. After about 1.5 s with no hands, the sentence
  is built and shown as a subtitle.
- **Manual mode:** a backup for stage. Hold **Space** (or the button) while signing one word,
  then release. Press **Enter** to finish the sentence.
- Grey words were below the confidence threshold, so they're left out of the subtitle.

The header shows `Model 1: MOCK` until the real model is plugged in.

Tune the word splitting (pause length, sentence gap and so on) in `frontend/src/config.js`.

---

## Plugging in the real model

### Model 1: sign recognition (Keras)

1. Put these in `backend/models/`:
   - `sign_model.keras` (or `.h5`): your trained model, softmax output
   - `labels.json`: the word list **in the same order as the model's outputs**,
     e.g. `["HELLO", "SCHOOL", ...]` (a `{"0": "HELLO", ...}` dict also works)
2. Edit `backend/models/model_config.json` so it matches **how you extracted keypoints in training**:
   ```json
   {
     "seq_len": 30,
     "features": ["pose", "face", "left_hand", "right_hand"],
     "pose_visibility": true,
     "normalize": "none"
   }
   ```
   - `seq_len`: frames per word your model expects. Segments are stretched or squeezed to fit.
   - `features`: which parts, **in the same order** you concatenated them in training.
     The default gives 33×4 + 468×3 + 21×3 + 21×3 = **1662** values per frame.
     If you only used pose + hands, set `["pose", "left_hand", "right_hand"]`.
   - `pose_visibility`: `true` if pose points had 4 values (x, y, z, visibility).
   - Missing hands/face become zeros, same as the usual training code.
3. Install TensorFlow and switch it on in `backend/.env`:
   ```bash
   pip install tensorflow
   ```
   ```
   RECOGNIZER=keras
   ```
4. Restart the backend. The header should now say `Model 1: keras`.

If the model and config don't match, the backend refuses to start and tells you why
(e.g. "Model expects 225 features per frame but model_config.json gives 1662").

> If your model uses a custom layer class, load it with `custom_objects` in
> `KerasRecognizer` (`backend/app/recognizer.py`).

### Model 2: sentence builder

Set `SENTENCE_BUILDER` in `backend/.env`:

| Value  | What it does | Needs |
|--------|--------------|-------|
| `rule` | Joins the words ("I go school.") and works offline | nothing |
| `groq` | Groq LLM turns glosses into a natural sentence | `GROQ_API_KEY` |
| `hf`   | Local Hugging Face model, optionally with a LoRA adapter | `HF_BASE_MODEL`, `HF_ADAPTER_PATH`, `pip install transformers peft torch` |

If `groq` or `hf` fails (no internet, rate limit), it falls back to `rule`, so the
demo never breaks on stage. For the `hf` option, check the input format in
`HFBuilder.build` matches what the adapter was trained on.

### Safety threshold

`CONFIDENCE_THRESHOLD=0.6` in `.env`. Words below it are dropped instead of guessed.

---

## API

| Method | Path | What |
|---|---|---|
| GET  | `/v1/info` | Model info, which landmarks to send, frames per word |
| GET  | `/v1/vocab` | Word list |
| POST | `/v1/recognize` | One word segment → word + confidence |
| POST | `/v1/sentence` | `{"words": [...]}` → sentence |
| POST | `/v1/translate` | Several segments → words + sentence in one call (for partners) |

A frame looks like:
```json
{ "pose": [[x,y,z,vis] ×33], "face": [[x,y,z] ×468],
  "left_hand": [[x,y,z] ×21] or null, "right_hand": [[x,y,z] ×21] or null }
```

Partner access: set `API_KEYS=key1,key2` and clients send `X-API-Key`. If it's empty, the API is open (demo mode).

---

## Tests
```bash
cd backend && pytest                       # API, preprocessing, real-model swap (if TF installed)
cd frontend && node test/segmenter.test.mjs   # word splitting
```

## Docker (optional)
```bash
docker compose up --build
```
Backend on :8000, web demo on :8080.

## Demo day checklist
- [ ] Real model in `backend/models/`, `RECOGNIZER=keras`, header no longer says MOCK
- [ ] Test with the actual laptop camera and the actual room lighting
- [ ] Practice in **Manual** mode as a backup
- [ ] `SENTENCE_BUILDER=rule` if the venue Wi-Fi is bad (everything else runs offline)
