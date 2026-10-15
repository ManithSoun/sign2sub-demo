import { useCallback, useEffect, useRef, useState } from "react";

// Runs MediaPipe Holistic on the webcam and calls onResults for every frame.
// Holistic is loaded from public/mediapipe (see index.html), so no internet is needed.
export function useHolistic(videoRef, onResults) {
  const [status, setStatus] = useState("off"); // off | loading | running | error
  const [error, setError] = useState("");
  const holisticRef = useRef(null);
  const streamRef = useRef(null);
  const runningRef = useRef(false);
  const cbRef = useRef(onResults);
  cbRef.current = onResults;

  const stop = useCallback(() => {
    runningRef.current = false;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    holisticRef.current?.close?.();
    holisticRef.current = null;
    setStatus("off");
  }, []);

  const start = useCallback(async () => {
    try {
      setStatus("loading");
      setError("");
      if (!window.Holistic) throw new Error("MediaPipe Holistic did not load. Run npm install again.");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const base = new URL("./mediapipe/holistic/", document.baseURI).href;
      const holistic = new window.Holistic({ locateFile: (f) => base + f });
      holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        refineFaceLandmarks: false, // 468 face points, same as Python Holistic default
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      holistic.onResults((r) => cbRef.current(r));
      await holistic.initialize();
      holisticRef.current = holistic;

      runningRef.current = true;
      setStatus("running");
      const loop = async () => {
        if (!runningRef.current) return;
        if (video.readyState >= 2) {
          try {
            await holistic.send({ image: video });
          } catch (e) {
            console.error(e);
          }
        }
        requestAnimationFrame(loop);
      };
      loop();
    } catch (e) {
      console.error(e);
      setError(e.message || String(e));
      setStatus("error");
      stop();
      setStatus("error");
    }
  }, [videoRef, stop]);

  useEffect(() => stop, [stop]);

  return { status, error, start, stop };
}
