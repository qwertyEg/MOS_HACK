/*
 * Тонкая обёртка над fetch для JSON API (docs/ARCHITECTURE.md §12).
 *
 * Зачем своя обёртка, а не голый fetch на страницах:
 *  - одна точка, где 401 уводит на /login (сессия истекла посреди работы);
 *  - текст ошибки берётся из ответа FastAPI (`detail`), а не «Failed to fetch»;
 *  - загрузка файлов идёт через XHR — только он отдаёт прогресс отправки.
 */
(function () {
  class ApiError extends Error {
    constructor(message, status, body) {
      super(message);
      this.status = status;
      this.body = body;
    }
  }

  function detailText(body, status) {
    if (!body) return `Ошибка сервера (${status})`;
    const d = body.detail ?? body.error ?? body.message;
    if (typeof d === "string") return d;
    // Ошибки валидации FastAPI: [{loc, msg}] — показываем первую по-человечески.
    if (Array.isArray(d) && d.length) {
      const f = d[0];
      const where = Array.isArray(f.loc) ? f.loc.slice(1).join(".") : "";
      return where ? `${where}: ${f.msg}` : f.msg;
    }
    return `Ошибка сервера (${status})`;
  }

  function toLogin() {
    const next = encodeURIComponent(location.pathname + location.search + location.hash);
    location.href = `/login?next=${next}`;
  }

  async function request(method, path, body, opts = {}) {
    const init = { method, headers: { Accept: "application/json" }, credentials: "same-origin" };
    if (body instanceof FormData) {
      init.body = body;
    } else if (body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    if (opts.signal) init.signal = opts.signal;
    let res;
    try {
      res = await fetch(path, init);
    } catch (e) {
      if (e.name === "AbortError") throw e;
      throw new ApiError("Нет связи с сервером. Проверьте, что сервис запущен.", 0, null);
    }
    if (res.status === 401) {
      toLogin();
      throw new ApiError("Сессия истекла — войдите снова.", 401, null);
    }
    const text = await res.text();
    let data = null;
    if (text) {
      try { data = JSON.parse(text); } catch { data = text; }
    }
    if (!res.ok) throw new ApiError(detailText(typeof data === "object" ? data : null, res.status), res.status, data);
    return data;
  }

  /* Загрузка с прогрессом: onProgress(0..1) по мере отправки байтов. */
  function upload(path, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", path);
      xhr.withCredentials = true;
      xhr.setRequestHeader("Accept", "application/json");
      xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
      xhr.onload = () => {
        let data = null;
        try { data = xhr.responseText ? JSON.parse(xhr.responseText) : null; } catch { data = null; }
        if (xhr.status === 401) { toLogin(); return reject(new ApiError("Сессия истекла", 401, null)); }
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else reject(new ApiError(detailText(data, xhr.status), xhr.status, data));
      };
      xhr.onerror = () => reject(new ApiError("Нет связи с сервером", 0, null));
      xhr.send(formData);
    });
  }

  /* Опрос задания до завершения: GET /api/jobs/{id} → {state, total, done, failed, postponed, pending, errors[]}.
     state: ingesting → processing → done | postponed (провайдер не готов, кадры ждут) | failed. */
  function pollJob(jobId, onTick, { interval = 900 } = {}) {
    let stopped = false;
    const promise = new Promise((resolve, reject) => {
      const tick = async () => {
        if (stopped) return;
        try {
          const j = await request("GET", `/api/jobs/${encodeURIComponent(jobId)}`);
          onTick && onTick(j);
          if (["done", "finished", "error", "failed", "cancelled", "postponed"].includes(j.state)) return resolve(j);
        } catch (e) {
          return reject(e);
        }
        setTimeout(tick, interval);
      };
      tick();
    });
    promise.stop = () => { stopped = true; };
    return promise;
  }

  window.SV = window.SV || {};
  window.SV.ApiError = ApiError;
  window.SV.api = {
    get: (p, o) => request("GET", p, undefined, o),
    post: (p, b, o) => request("POST", p, b, o),
    put: (p, b, o) => request("PUT", p, b, o),
    patch: (p, b, o) => request("PATCH", p, b, o),
    del: (p, b, o) => request("DELETE", p, b, o),
    upload,
    pollJob,
  };
})();
