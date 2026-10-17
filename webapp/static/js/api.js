// Thin fetch wrappers and job polling.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function parse(res) {
  const type = res.headers.get("content-type") || "";
  const body = type.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    let msg = typeof body === "string" ? body : body.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");
    throw new ApiError(msg || `HTTP ${res.status}`, res.status);
  }
  return body;
}

export async function get(url) {
  return parse(await fetch(url));
}

export async function post(url, data) {
  return parse(await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data ?? {}),
  }));
}

export async function postBlob(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data ?? {}),
  });
  if (!res.ok) await parse(res);
  const cd = res.headers.get("content-disposition") || "";
  const m = cd.match(/filename="?([^"]+)"?/);
  return { blob: await res.blob(), filename: m ? m[1] : "download" };
}

/** Uploads a file with progress (fetch has no upload progress). */
export function upload(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch (e) {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new ApiError(body?.detail || `HTTP ${xhr.status}`, xhr.status));
    };
    xhr.onerror = () => reject(new ApiError("Network error", 0));
    xhr.send(form);
  });
}

/** Polls a background job until it finishes. Resolves with job.result, rejects on error/cancel. */
export function pollJob(jobId, onProgress, interval = 400) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      let job;
      try {
        job = await get(`/api/jobs/${jobId}`);
      } catch (e) {
        reject(e);
        return;
      }
      onProgress?.(job);
      if (job.state === "done") resolve(job.result);
      else if (job.state === "error") reject(new ApiError(job.error || "Job failed", 500));
      else if (job.state === "cancelled") reject(Object.assign(new ApiError("cancelled", 0), { cancelled: true }));
      else setTimeout(tick, interval);
    };
    tick();
  });
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
