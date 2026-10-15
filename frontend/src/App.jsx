import { useCallback, useEffect, useRef, useState } from "react";
import { buildSentence, deleteTemplate, getInfo, listTemplates, recognize, teach } from "./api.js";
import { SENTENCE_GAP_MS } from "./config.js";
import { drawLandmarks, handsVisible, toFrame } from "./landmarks.js";
import { Segmenter } from "./segmenter.js";
import { useHolistic } from "./useHolistic.js";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [info, setInfo] = useState(null);
  const [apiError, setApiError] = useState("");
  const [mode, setMode] = useState("auto"); // auto | manual
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [signing, setSigning] = useState(false);
  const [words, setWords] = useState([]); // current sentence: {id, word, confidence, accepted, pending}
  const [subtitle, setSubtitle] = useState("");
  const [history, setHistory] = useState([]);
  const [tab, setTab] = useState("live"); // live | teach
  const [teachWord, setTeachWord] = useState("");
  const [templates, setTemplates] = useState({}); // word -> number of examples
  const [teachMsg, setTeachMsg] = useState("");

  const featuresRef = useRef(["pose", "face", "left_hand", "right_hand"]);
  const modeRef = useRef(mode);
  const skeletonRef = useRef(showSkeleton);
  const manualRef = useRef({ recording: false, frames: [] });
  const lastHandsRef = useRef(0);
  const pendingRef = useRef(0);
  const wordsRef = useRef(words);
  const idRef = useRef(0);
  const tabRef = useRef(tab);
  const teachWordRef = useRef(teachWord);
  tabRef.current = tab;
  teachWordRef.current = teachWord;
  modeRef.current = mode;
  skeletonRef.current = showSkeleton;
  wordsRef.current = words;

  // ---- API info ----
  const loadInfo = useCallback(async () => {
    try {
      const i = await getInfo();
      setInfo(i);
      featuresRef.current = i.features;
      setApiError("");
    } catch {
      setInfo(null);
      setApiError("API offline. Start the backend.");
    }
  }, []);
  useEffect(() => { loadInfo(); }, [loadInfo]);

  // ---- teach signs (template mode) ----
  const loadTemplates = useCallback(async () => {
    try { setTemplates((await listTemplates()).counts); } catch { /* not in template mode */ }
  }, []);
  useEffect(() => { if (info?.can_teach) loadTemplates(); }, [info, loadTemplates]);

  const saveExample = useCallback(async (frames) => {
    const word = teachWordRef.current.trim().toUpperCase();
    if (!word) { setTeachMsg("Type the word first."); return; }
    if (frames.length < 5) { setTeachMsg("Too short. Hold the key for the whole sign."); return; }
    try {
      const r = await teach(word, frames);
      setTemplates(r.counts);
      setTeachMsg(`Saved ${word} (example ${r.examples}).`);
    } catch {
      setTeachMsg("Could not save. Is the backend running with RECOGNIZER=template?");
    }
  }, []);

  const removeTemplate = async (word, lastOnly) => {
    try { setTemplates((await deleteTemplate(word, lastOnly)).counts); } catch { /* ignore */ }
  };

  // ---- one word segment -> Model 1 ----
  const sendWord = useCallback(async (frames) => {
    const id = ++idRef.current;
    pendingRef.current++;
    setWords((w) => [...w, { id, pending: true }]);
    try {
      const p = await recognize(frames);
      setWords((w) => w.map((x) => (x.id === id ? { id, ...p, pending: false } : x)));
    } catch {
      setWords((w) => w.filter((x) => x.id !== id));
      setApiError("Could not reach the API.");
    } finally {
      pendingRef.current--;
    }
  }, []);

  const segmenterRef = useRef(null);
  if (!segmenterRef.current) segmenterRef.current = new Segmenter((f) => sendWord(f));

  // ---- accepted words -> Model 2 -> subtitle ----
  const finishSentence = useCallback(async () => {
    // wait (max 3 s) for words still being recognized, so none get lost
    for (let i = 0; i < 30 && pendingRef.current > 0; i++) await new Promise((r) => setTimeout(r, 100));
    const accepted = wordsRef.current.filter((w) => w.accepted).map((w) => w.word);
    if (!accepted.length) {
      setWords([]);
      return;
    }
    setWords([]);
    try {
      const r = await buildSentence(accepted);
      setSubtitle(r.sentence);
      setHistory((h) => [{ sentence: r.sentence, words: r.words, time: new Date() }, ...h].slice(0, 20));
    } catch {
      setApiError("Could not build the sentence.");
    }
  }, []);

  // Auto-finish the sentence after a pause with no hands
  useEffect(() => {
    const t = setInterval(() => {
      if (modeRef.current !== "auto") return; // manual mode: press Enter / Finish sentence
      const idle = segmenterRef.current.state === "idle";
      const quiet = Date.now() - lastHandsRef.current > SENTENCE_GAP_MS;
      const hasWords = wordsRef.current.some((w) => w.accepted);
      if (idle && quiet && hasWords && pendingRef.current === 0) finishSentence();
    }, 200);
    return () => clearInterval(t);
  }, [finishSentence]);

  // ---- every camera frame ----
  const onResults = useCallback((results) => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (skeletonRef.current) drawLandmarks(ctx, results);
      else ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    const hands = handsVisible(results);
    if (hands) lastHandsRef.current = Date.now();
    const frame = toFrame(results, featuresRef.current);

    if (tabRef.current === "teach") {
      if (manualRef.current.recording) manualRef.current.frames.push(frame);
    } else if (modeRef.current === "auto") {
      segmenterRef.current.push(frame, results);
      setSigning(segmenterRef.current.state === "signing");
    } else if (manualRef.current.recording) {
      manualRef.current.frames.push(frame);
    }
  }, []);

  const cam = useHolistic(videoRef, onResults);

  // ---- manual mode: hold Space (or the button) while signing one word ----
  const startManual = useCallback(() => {
    const canRecord = modeRef.current === "manual" || tabRef.current === "teach";
    if (!canRecord || manualRef.current.recording) return;
    manualRef.current = { recording: true, frames: [] };
    setSigning(true);
  }, []);
  const stopManual = useCallback(() => {
    if (!manualRef.current.recording) return;
    const { frames } = manualRef.current;
    manualRef.current = { recording: false, frames: [] };
    setSigning(false);
    lastHandsRef.current = Date.now();
    if (tabRef.current === "teach") saveExample(frames);
    else if (frames.length) sendWord(frames);
  }, [sendWord, saveExample]);

  useEffect(() => {
    const down = (e) => {
      if (e.target.tagName === "INPUT") return; // typing a word
      if (e.code === "Space" && !e.repeat) { e.preventDefault(); startManual(); }
      if (e.code === "Enter") finishSentence();
    };
    const up = (e) => { if (e.code === "Space" && e.target.tagName !== "INPUT") stopManual(); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [startManual, stopManual, finishSentence]);

  const switchMode = (m) => {
    segmenterRef.current.reset();
    manualRef.current = { recording: false, frames: [] };
    setSigning(false);
    setMode(m);
  };

  const clearAll = () => { setWords([]); setSubtitle(""); setHistory([]); };

  const running = cam.status === "running";

  const teaching = tab === "teach";
  const statusText = !running ? "Camera off"
    : teaching ? (signing ? `Recording ${teachWord.toUpperCase() || "…"}` : "Hold Space to record a sign")
    : signing ? "Signing" : mode === "auto" ? "Ready, start signing" : "Hold Space to sign";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="./logo.png" alt="" className="logo-img" onError={(e) => (e.currentTarget.style.display = "none")} />
          <span className="wordmark">Sign2<b>Sub</b></span>
        </div>
        <div className="meta">
          <span className={`meta-item ${info ? "ok" : "bad"}`}><i />{info ? "Connected" : "API offline"}</span>
          {!info && <button className="link" onClick={loadInfo}>Retry</button>}
        </div>
      </header>

      <main className="layout">
        {/* camera */}
        <section className="stage">
          <div className="video-wrap">
            <video ref={videoRef} playsInline muted />
            <canvas ref={canvasRef} width={1280} height={720} />

            {!running && (
              <div className="overlay">
                <h1>Sign in front of the camera</h1>
                <p>Words show up on the right. When you finish a sentence, the subtitle appears here.</p>
                <button className="btn primary big" onClick={cam.start} disabled={cam.status === "loading"}>
                  {cam.status === "loading" ? "Loading…" : "Start camera"}
                </button>
                {cam.error && <p className="err">{cam.error}</p>}
              </div>
            )}

            {running && (
              <span className={`status ${signing ? "live" : ""}`}><span className="dot" />{statusText}</span>
            )}
            <div className={`subtitle ${subtitle ? "show" : ""}`}>{subtitle}</div>
          </div>

          <div className="toolbar">
            {teaching ? (
              <button
                className={`btn ${signing ? "rec" : "primary"}`}
                onPointerDown={startManual}
                onPointerUp={stopManual}
                onPointerLeave={stopManual}
                disabled={!running || !teachWord.trim()}
              >
                {signing ? "Release to save" : `Hold to record ${teachWord.trim().toUpperCase() || "a sign"} (Space)`}
              </button>
            ) : (<>
            <div className="seg" title="Auto: words split on pauses. Manual: hold Space for each word.">
              <button className={mode === "auto" ? "on" : ""} onClick={() => switchMode("auto")}>Auto</button>
              <button className={mode === "manual" ? "on" : ""} onClick={() => switchMode("manual")}>Manual</button>
            </div>
            {mode === "manual" && (
              <button
                className={`btn ${signing ? "rec" : "primary"}`}
                onPointerDown={startManual}
                onPointerUp={stopManual}
                onPointerLeave={stopManual}
                disabled={!running}
              >
                {signing ? "Release to send" : "Hold to sign (Space)"}
              </button>
            )}
            <button className="btn" onClick={finishSentence} disabled={!words.length}>Finish sentence (Enter)</button>
            </>)}
            <span className="spacer" />
            <label className="check">
              <input type="checkbox" checked={showSkeleton} onChange={(e) => setShowSkeleton(e.target.checked)} />
              Show tracking
            </label>
            {running && <button className="btn ghost" onClick={cam.stop}>Stop camera</button>}
          </div>
          {apiError && <p className="err-bar">{apiError}</p>}
        </section>

        {/* side */}
        <aside className="side">
          {info?.can_teach && (
            <div className="tabs">
              <button className={!teaching ? "on" : ""} onClick={() => { setTab("live"); setSigning(false); }}>Live</button>
              <button className={teaching ? "on" : ""} onClick={() => { segmenterRef.current.reset(); setSigning(false); setTab("teach"); }}>
                Teach signs
              </button>
            </div>
          )}

          {teaching ? (
          <div className="card teach">
            <div className="card-top"><h2>Teach <span>signs</span></h2></div>
            <label className="field">
              <span>Word</span>
              <input
                value={teachWord}
                onChange={(e) => setTeachWord(e.target.value)}
                placeholder="e.g. HELLO"
                list="taught-words"
              />
              <datalist id="taught-words">{Object.keys(templates).map((w) => <option key={w} value={w} />)}</datalist>
            </label>
            <p className="hint">Type the word, then hold <b>Space</b> while you sign it once. Record each word 3 to 5 times.</p>
            {teachMsg && <p className="teach-msg">{teachMsg}</p>}
            <ul className="taught">
              {Object.keys(templates).length === 0 && <p className="muted">No signs yet.</p>}
              {Object.entries(templates).map(([w, n]) => (
                <li key={w} className={w === teachWord.trim().toUpperCase() ? "current" : ""}>
                  <button className="word-btn" onClick={() => setTeachWord(w)}>{w}</button>
                  <span className={`count ${n >= 3 ? "ok" : ""}`}>{n}×</span>
                  <button className="link" title="Remove last example" onClick={() => removeTemplate(w, true)}>Undo</button>
                  <button className="link danger" title="Delete word" onClick={() => removeTemplate(w, false)}>Delete</button>
                </li>
              ))}
            </ul>
          </div>
          ) : (<>
          <div className="card words">
            <div className="card-top">
              <h2>Words <span>so far</span></h2>
              {words.length > 0 && <button className="link" onClick={() => setWords([])}>Clear</button>}
            </div>
            <div className="chips">
              {words.length === 0 && <p className="muted">Each word you sign shows up here.</p>}
              {words.map((w) =>
                w.pending ? (
                  <span key={w.id} className="chip pending">…</span>
                ) : (
                  <span key={w.id} className={`chip ${w.accepted ? "" : "dropped"}`}
                    title={w.top?.map(([t, p]) => `${t} ${Math.round(p * 100)}%`).join("\n")}>
                    {w.accepted ? w.word : w.top?.[0]?.[0]}
                    <small>{Math.round(w.confidence * 100)}%</small>
                  </span>
                )
              )}
            </div>
            {words.some((w) => w.accepted === false) && <p className="hint">Crossed out = not sure, so it's left out.</p>}
          </div>

          <div className="card transcript">
            <div className="card-top">
              <h2>Your <span>transcript</span></h2>
              {history.length > 0 && <button className="link" onClick={clearAll}>Clear</button>}
            </div>
            <ul className="history">
              {history.length === 0 && <p className="muted">Finished sentences are listed here, newest first.</p>}
              {history.map((h, i) => (
                <li key={i}>
                  <p>{h.sentence}</p>
                  <small>{h.words.join(" · ")} · {h.time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small>
                </li>
              ))}
            </ul>
          </div>
          </>)}
        </aside>
      </main>
    </div>
  );
}