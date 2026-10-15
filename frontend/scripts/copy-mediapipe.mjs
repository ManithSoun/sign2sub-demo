// Copies MediaPipe Holistic files into public/ so the demo works offline
// (no CDN needed at the competition venue).
import { cpSync, mkdirSync } from "node:fs";
const src = "node_modules/@mediapipe/holistic";
const dst = "public/mediapipe/holistic";
mkdirSync(dst, { recursive: true });
cpSync(src, dst, { recursive: true });
console.log("MediaPipe Holistic copied to", dst);
