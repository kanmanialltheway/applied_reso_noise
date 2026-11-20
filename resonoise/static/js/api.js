// Thin client for the Reso-Noise JSON API.
export class ApiError extends Error {
  constructor(message, status, code) { super(message); this.status = status; this.code = code; }
}

async function request(method, url, body, isForm) {
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: body && !isForm ? { "Content-Type": "application/json" } : undefined,
      body: body ? (isForm ? body : JSON.stringify(body)) : undefined,
    });
  } catch (e) {
    throw new ApiError("Cannot reach the Reso-Noise server. Check that it is still running.", 0, "network");
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON body */ }
  if (!res.ok) {
    const err = data && data.error;
    throw new ApiError(err ? err.message : `Request failed (${res.status}).`, res.status, err ? err.code : "http");
  }
  return data;
}

export const api = {
  health: () => request("GET", "/api/health"),
  model: () => request("GET", "/api/model"),
  retrain: () => request("POST", "/api/model/retrain"),
  upload(fileOrBlob, name, source = "upload") {
    const fd = new FormData();
    fd.append("file", fileOrBlob, name);
    fd.append("source", source);
    return request("POST", "/api/recordings", fd, true);
  },
  sample: (kind) => request("POST", "/api/samples", { kind }),
  library: () => request("GET", "/api/recordings"),
  detail: (id) => request("GET", `/api/recordings/${id}`),
  remove: (id) => request("DELETE", `/api/recordings/${id}`),
  analyze: (id) => request("POST", `/api/recordings/${id}/analyze`),
  analysis: (id) => request("GET", `/api/recordings/${id}/analysis`),
  preview: (id, params) => request("POST", `/api/recordings/${id}/filter-preview`, params),
  process: (id, params) => request("POST", `/api/recordings/${id}/process`, params),
  label: (id, label) => request("POST", `/api/recordings/${id}/label`, { label }),
  processed: (pid) => request("GET", `/api/processed/${pid}`),
  removeProcessed: (pid) => request("DELETE", `/api/processed/${pid}`),
  waveform: (id, t0, t1, points, pid) =>
    request("GET", `/api/recordings/${id}/waveform?t0=${t0}&t1=${t1}&points=${points}${pid ? `&pid=${pid}` : ""}`),
  audioUrl: (id, download) => `/api/recordings/${id}/audio${download ? "?download=1" : ""}`,
  processedUrl: (pid, download) => `/api/processed/${pid}/audio${download ? "?download=1" : ""}`,
  reportUrl: (pid) => `/api/processed/${pid}/report`,
};
