// Quick check of word splitting with fake hand data: node test/segmenter.test.mjs
import assert from "node:assert";
const { Segmenter } = await import("../src/segmenter.js");

const words = [];
const seg = new Segmenter((f) => words.push(f.length));
const hand = (x) => ({ leftHandLandmarks: [{ x, y: 0.5 }], rightHandLandmarks: null });
const none = { leftHandLandmarks: null, rightHandLandmarks: null };
const feed = (r, n) => { for (let i = 0; i < n; i++) seg.push({}, typeof r === "function" ? r(i) : r); };

feed(none, 10);                        // nobody
feed((i) => hand(0.3 + i * 0.02), 20); // word 1: moving hand
feed(hand(0.7), 12);                   // pause, hand held still -> word 1 ends
feed((i) => hand(0.7 - i * 0.02), 15); // word 2
feed(none, 10);                        // hands down -> word 2 ends
feed(hand(0.5), 2);                    // blip, too short
feed(none, 10);

assert.equal(words.length, 2, `expected 2 words, got ${words.length}: ${words}`);
assert.ok(words.every((n) => n >= 8));
console.log("segmenter ok:", words);
