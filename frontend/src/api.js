import { API_KEY, API_URL } from "./config.js";

async function call(path, body, method) {
  const res = await fetch(`${API_URL}${path}`, {
    method: method || (body ? "POST" : "GET"),
    headers: {
      "Content-Type": "application/json",
      ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export const getInfo = () => call("/v1/info");
export const recognize = (frames) => call("/v1/recognize", { frames });
export const buildSentence = (words) => call("/v1/sentence", { words });

// Teach signs (RECOGNIZER=template)
export const listTemplates = () => call("/v1/templates");
export const teach = (word, frames) => call(`/v1/templates/${encodeURIComponent(word)}`, { frames });
export const deleteTemplate = (word, lastOnly = false) =>
  call(`/v1/templates/${encodeURIComponent(word)}?last_only=${lastOnly}`, undefined, "DELETE");