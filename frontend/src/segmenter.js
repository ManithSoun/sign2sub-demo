// Splits the live frame stream into one-word segments.
//
// A word starts when hands appear, and ends when the hands drop out of view
// or stay still for a moment (the short pause signers make between words).

import { SEGMENT } from "./config.js";
import { handsVisible, movement, wrists } from "./landmarks.js";

export class Segmenter {
  constructor(onWord) {
    this.onWord = onWord;
    this.reset();
  }

  reset() {
    this.state = "idle";
    this.buffer = [];
    this.preRoll = [];
    this.seen = 0;
    this.gone = 0;
    this.still = 0;
    this.prevWrists = null;
  }

  /** Call once per video frame. frame = API frame, results = raw MediaPipe results */
  push(frame, results) {
    const hands = handsVisible(results);
    const w = wrists(results);
    const moved = movement(this.prevWrists, w);
    this.prevWrists = w;

    if (this.state === "idle") {
      this.preRoll.push(frame);
      if (this.preRoll.length > SEGMENT.preRollFrames) this.preRoll.shift();
      this.seen = hands ? this.seen + 1 : 0;
      if (this.seen >= SEGMENT.startFrames) {
        this.state = "signing";
        this.buffer = [...this.preRoll];
        this.gone = this.still = 0;
        this.stillReady = false; // must move once before a pause can end the word
      }
      return;
    }

    // signing
    this.buffer.push(frame);
    this.gone = hands ? 0 : this.gone + 1;
    if (moved > SEGMENT.stillThreshold * 2) this.stillReady = true;
    this.still = hands && moved < SEGMENT.stillThreshold ? this.still + 1 : 0;

    const ended =
      this.gone >= SEGMENT.handsGoneFrames ||
      (this.stillReady && this.still >= SEGMENT.stillFrames && this.buffer.length >= SEGMENT.minFrames) ||
      this.buffer.length >= SEGMENT.maxFrames;

    if (ended) this.finish(hands);
  }

  /** Manual mode: force-close the current word now */
  finish(handsStillUp = false) {
    // drop the trailing pause frames (hands gone or held still), keep 2 for context
    const trailing = Math.max(this.gone, this.still - 2, 0);
    const frames = trailing ? this.buffer.slice(0, -trailing) : this.buffer;
    if (frames.length >= SEGMENT.minFrames) this.onWord(frames);
    this.buffer = [];
    this.gone = this.still = 0;
    this.stillReady = false;
    // after a pause with hands still up, the next word starts right away
    this.state = handsStillUp ? "signing" : "idle";
    this.seen = 0;
  }
}
