// All tuning knobs for the demo in one place.

export const API_URL = import.meta.env?.VITE_API_URL || "http://localhost:8000";
export const API_KEY = import.meta.env?.VITE_API_KEY || "";

export const SEGMENT = {
  // Hands must be seen this many frames in a row before a word starts
  startFrames: 3,
  // A word ends when hands are gone for this many frames...
  handsGoneFrames: 6,
  // ...or when hands are almost still for this many frames (the pause between words)
  stillFrames: 9,
  // How small a wrist movement counts as "still" (fraction of the image)
  stillThreshold: 0.006,
  // Ignore very short blips; cut very long words
  minFrames: 8,
  maxFrames: 75,
  // Keep this many frames from just before the word started
  preRollFrames: 4,
};

// No hands for this long (ms) = the sentence is finished, build the subtitle
export const SENTENCE_GAP_MS = 1600;
