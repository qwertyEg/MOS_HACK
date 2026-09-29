/*
 * «Проверить снимок»: перетащить фото → рамки техники, чек-лист, этап и время
 * ответа (POST /api/analyze, без сохранения). Этим экраном жюри проверяет
 * распознавание на своих снимках, поэтому: вставка из буфера, пример в один
 * клик и повтор тем же файлом на другом провайдере для сравнения.
 */
document.addEventListener("alpine:init", () => {
  Alpine.data("tryPage", () => SV.mix(SV.frameMixin(), {
    provider: "local",
    file: null,
    url: null,
    busy: false,
    elapsed: 0,
    frame: null, // результат в форме кадра — переиспользуем панель разбора
    result: null,
    error: null,
    over: false,
    runs: [],

    init() {
      const app = Alpine.store("app");
      app.loadCatalog();
      const pick = () => { if (app.settings && !this.file) this.provider = app.settings.mode || "local"; };
      pick();
      this.$watch("$store.app.settings", pick);
      window.addEventListener("paste", (e) => {
        const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
        if (f) this.take(f);
      });
    },
    ready(p) { return Alpine.store("app").modeReady(p); },

    take(f) {
      if (!f) return;
      if (!f.type.startsWith("image/")) return SV.toast("Нужна картинка: jpg, png или webp", { kind: "error" });
      if (this.url) URL.revokeObjectURL(this.url);
      this.file = f;
      this.url = URL.createObjectURL(f);
      this.run();
    },
    async sample() {
      try {
        const sites = Alpine.store("app").sites || (await Alpine.store("app").loadSites());
        const s = (sites || []).find((x) => x.thumb);
        if (!s) return SV.toast("Примеров нет: создайте демо-объект на странице «Обзор»", { kind: "info" });
        const blob = await (await fetch(s.thumb, { credentials: "same-origin" })).blob();
        this.take(new File([blob], "пример.jpg", { type: blob.type || "image/jpeg" }));
      } catch (e) {
        SV.toastError(e, "Пример не загрузился");
      }
    },
    async run(provider) {
      if (provider) this.provider = provider;
      if (!this.file || this.busy) return;
      this.busy = true;
      this.error = null;
      this.frame = null;
      this.result = null;
      const t0 = performance.now();
      const tick = setInterval(() => { this.elapsed = performance.now() - t0; }, 50);
      try {
        // POST /api/analyze: provider = local | external | hybrid; annotate=0 — рамки рисуем сами (SVG).
        const fd = new FormData();
        fd.append("file", this.file);
        fd.append("provider", this.provider);
        fd.append("annotate", "0");
        const r = await SV.api.post("/api/analyze", fd);
        const total = performance.now() - t0;
        this.result = Object.assign({ client_ms: total }, r);
        const img = r.image || {};
        this.frame = Object.assign({ url: this.url, id: null, width: img.width, height: img.height,
          stage_likelihood: r.stage ? r.stage.stage_likelihood : null }, r);
        const ms = (r.timings && r.timings.total_ms) || total;
        this.runs.unshift({ provider: this.provider, name: this.file.name, ms, stage: r.stage && r.stage.front, n: (r.detections || []).length, at: new Date() });
        this.runs = this.runs.slice(0, 6);
      } catch (e) {
        this.error = e.message;
      } finally {
        clearInterval(tick);
        this.busy = false;
      }
    },
    reset() {
      this.file = null; this.frame = null; this.result = null; this.error = null;
      if (this.url) URL.revokeObjectURL(this.url);
      this.url = null;
    },
    get otherProvider() { return this.provider === "local" ? "external" : "local"; },
    get latency() { return this.result ? ((this.result.timings && this.result.timings.total_ms) ?? this.result.client_ms) : null; },
    get timings() {
      const t = (this.result && this.result.timings) || {};
      return [["Качество", t.quality_ms], ["Техника", t.detect_ms], ["Этап", t.stage_ms]].filter((x) => x[1] != null);
    },
    /* Этап из ответа модели Б: stage = {front, name, stage_likelihood, progress, substages{"3.2": "active"}}.
       Подэтап и коды работ — из справочника /api/catalog. */
    get stage() {
      const st = this.result && this.result.stage;
      /* Этап — как на площадке: чек-лист модели Б вместе с техникой на снимке (stage.fused). */
      const fu = st && st.fused && st.fused.front != null ? st.fused : null;
      const id = fu ? fu.front : st && st.front;
      if (id == null) return null;
      const cat = ((Alpine.store("app").catalog || {}).stages || []).find((x) => x.id === id) || {};
      const subs = Object.entries(st.substages || {}).filter(([k, v]) => k.startsWith(id + ".") && v === "active").map(([k]) => k);
      const sub = (cat.substages || []).find((x) => x.id === subs[0]) || null;
      const works = (cat.works || []).filter((w) => w.code && (!sub || w.substage_id === sub.id)).slice(0, 3);
      const lk = st.stage_likelihood || {};
      return { id, name: st.name || cat.name, substage: sub, works, confidence: lk[id] ?? lk[String(id)] ?? null,
        progress: (st.progress || {})[id] ?? (st.progress || {})[String(id)] ?? null, expected: cat.equipment_expected || [], forbidden: cat.equipment_forbidden || [],
        basis: fu ? fu.text : "" };
    },
    /* Правило «этап → техника» на одном снимке: чего из ожидаемого нет, что запрещено. */
    get rules() {
      const st = this.stage;
      if (!st) return null;
      const seen = new Set((this.result.detections || []).map((d) => d.class || d.cls));
      return { expected: st.expected, missing: st.expected.filter((k) => !seen.has(k)), forbidden: st.forbidden.filter((k) => seen.has(k)) };
    },
    get coldStart() {
      const t = (this.result && this.result.timings) || {};
      return (t.detect_ms || 0) > 5000 || (t.stage_ms || 0) > 8000;
    },
    get notes() { return (this.result && this.result.errors) || []; },
    get modelsLine() {
      const r = this.result;
      if (!r) return "";
      const n = (k) => (SV.META.providers[k] || {}).label || k;
      return `Модель А: ${n(r.model_a)} · модель Б: ${n(r.model_b)}` + (r.checklist && r.checklist.cost_usd ? ` · $${SV.fmt.num(r.checklist.cost_usd, 4)}` : "");
    },
    get svg() { return SV.boxesSvg(this.frame, { highlight: this.highlight }); },
    get labels() { return SV.boxLabels(this.frame, { dispW: this.dispW, conf: true }); },
    providerLabel(p) { return (SV.META.modes[p] || {}).label || p; },
  }));
});
