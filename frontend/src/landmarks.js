// Convert MediaPipe Holistic results into the API frame format,
// and small helpers used for word segmentation.

const r = (v) => Math.round(v * 10000) / 10000;
const xyz = (pts) => (pts ? pts.map((p) => [r(p.x), r(p.y), r(p.z)]) : null);
const xyzv = (pts) => (pts ? pts.map((p) => [r(p.x), r(p.y), r(p.z), r(p.visibility ?? 0)]) : null);

/** features = which parts the model needs (from /v1/info), so we don't send extra data */
export function toFrame(results, features) {
  const want = new Set(features);
  return {
    pose: want.has("pose") ? xyzv(results.poseLandmarks) : null,
    face: want.has("face") ? xyz(results.faceLandmarks) : null,
    left_hand: want.has("left_hand") ? xyz(results.leftHandLandmarks) : null,
    right_hand: want.has("right_hand") ? xyz(results.rightHandLandmarks) : null,
  };
}

export const handsVisible = (results) =>
  Boolean(results.leftHandLandmarks || results.rightHandLandmarks);

/** Wrist positions (point 0 of each hand), used to detect pauses */
export function wrists(results) {
  return [results.leftHandLandmarks?.[0], results.rightHandLandmarks?.[0]];
}

export function movement(prev, curr) {
  let max = 0;
  for (let i = 0; i < 2; i++) {
    if (prev?.[i] && curr?.[i]) {
      max = Math.max(max, Math.hypot(curr[i].x - prev[i].x, curr[i].y - prev[i].y));
    }
  }
  return max;
}

// Connections for drawing the skeleton overlay
const HAND = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const POSE = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]];

function drawSet(ctx, pts, edges, color, w, h, radius) {
  if (!pts) return;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.5;
  for (const [a, b] of edges) {
    if (!pts[a] || !pts[b]) continue;
    ctx.beginPath();
    ctx.moveTo(pts[a].x * w, pts[a].y * h);
    ctx.lineTo(pts[b].x * w, pts[b].y * h);
    ctx.stroke();
  }
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

export function drawLandmarks(ctx, results) {
  const { width: w, height: h } = ctx.canvas;
  ctx.clearRect(0, 0, w, h);
  const pose = results.poseLandmarks?.slice(0, 25);
  drawSet(ctx, pose, POSE, "rgba(255,255,255,0.55)", w, h, 0);
  drawSet(ctx, results.leftHandLandmarks, HAND, "#ff8a3d", w, h, 3);
  drawSet(ctx, results.rightHandLandmarks, HAND, "#5aa9ff", w, h, 3);
}
