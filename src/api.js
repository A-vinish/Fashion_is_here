export const API_BASE = "http://localhost:5000";

/**
 * Thin wrapper so every API call handles network/server errors the
 * same way, instead of repeating try/catch everywhere.
 */
async function request(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch (err) {
    throw new Error("Couldn't reach the backend. Is 'python app.py' still running?");
  }

  if (!res.ok) {
    const errorText = await res.text().catch(() => "");
    throw new Error(`Server error (${res.status}). ${errorText || "Check the backend terminal for details."}`);
  }

  return res.json();
}

export function searchByImage(file, sessionId) {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("session_id", sessionId);
  return request(`${API_BASE}/search/image`, { method: "POST", body: formData });
}

export function sendChatMessage(message, sessionId) {
  return request(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
}

export function resetSession(sessionId) {
  return request(`${API_BASE}/session/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export function getHealth() {
  return request(`${API_BASE}/health`, { method: "GET" });
}
