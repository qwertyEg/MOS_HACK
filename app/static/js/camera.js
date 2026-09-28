/*
 * Страница камеры: плеер кадров со шкалой времени, слои (рамки / маска / зоны),
 * разбор кадра, загрузка файлов с прогрессом задания, калибровка по 4+ точкам
 * и редактор зон-полигонов.
 *
 * Шкала времени — по времени съёмки, а не по номеру кадра: пропуски (камера
 * молчала, ночь без кадров) видны как пустоты. Высота штриха — сколько техники
 * работало на кадре: шкала заодно показывает ритм площадки за сутки.
 */
document.addEventListener("alpine:init", () => {
  const SIDE = ["analysis", "upload", "calib", "zones"];

  Alpine.data("cameraPage", (cameraId) => SV.mix(SV.frameMixin(), {
    cameraId,
    cam: null,
    frames: null,
    idx: -1,
    frame: null,
    cache: {},
    error: null,
    frameError: null,
    loadingFrame: false,
    side: "analysis",
    layers: { boxes: true, mask: false, zones: true },
    zones: [],

    // загрузка
    files: [],
    upOpts: { interval_min: 20, start_at: "", video_mode: "realtime", step_s: "" },
    // поток (simcam): POST /api/cameras/{id}/connect | /disconnect
    stream: { uri: "", interval: "", busy: false, info: null },
    upState: null, // {phase:'upload'|'job'|'done'|'error', p, job}
    over: false,

    // калибровка
    calib: { points: [], site: [], result: null, busy: false },
    // зоны
    draft: null, // {points:[], name, kind}

    async init() {
      const h = location.hash.replace("#", "");
      if (SIDE.includes(h)) this.side = h;
      const qs = new URLSearchParams(location.search);
      const want = qs.get("frame");
      try {
        const [cam, frames, zones] = await Promise.all([
          SV.api.get(`/api/cameras/${cameraId}`),
          SV.api.get(`/api/cameras/${cameraId}/frames?limit=500`),
          SV.api.get(`/api/cameras/${cameraId}/zones`).catch(() => []),
        ]);
        this.cam = cam;
        this.upOpts.interval_min = cam.interval_min || 20;
        this.stream.uri = cam.source_uri && /^https?:/.test(cam.source_uri) ? cam.source_uri : "";
        if (cam.site_id != null) {
          Alpine.store("app").siteId = cam.site_id;
          // Подписи «Экскаватор №1» на рамках — из единиц техники объекта.
          SV.api.get(`/api/sites/${cam.site_id}/equipment`).then((d) => { SV.setUnits(d.units); this.unitsTick++; }).catch(() => {});
        }
        Alpine.store("app").loadCatalog();
        document.title = `${cam.name} · СтройВзор`;
        const list = Array.isArray(frames) ? frames : frames.items || [];
        this.frames = list.slice().sort((a, b) => String(a.captured_at).localeCompare(String(b.captured_at)));
        this.zones = zones || [];
        if (cam.calib_points) {
          this.calib.points = (cam.calib_points.image_points || []).map((p) => [...p]);
          this.calib.site = (cam.calib_points.site_points || []).map((p) => [...p]);
        }
        if (this.frames.length) {
          const i = want ? this.frames.findIndex((f) => String(f.id) === want) : -1;
          await this.go(i >= 0 ? i : this.frames.length - 1);
        }
      } catch (e) {
        this.error = e.message;
      }
      window.addEventListener("keydown", (e) => this.onKey(e));
      // Пока у камеры есть кадры в очереди — подтягиваем свежие результаты сами.
      this._poll = setInterval(() => { if (!document.hidden && this.cam && (this.cam.pending || this.inQueue)) this.poll(); }, 6000);
    },
    destroy() { clearInterval(this._poll); },
    unitsTick: 0,
    async poll() {
      try {
        const [cam, list] = await Promise.all([
          SV.api.get(`/api/cameras/${cameraId}`),
          SV.api.get(`/api/cameras/${cameraId}/frames?limit=500`),
        ]);
        const atEnd = this.frames && this.idx === this.frames.length - 1;
        const curId = this.cur && this.cur.id;
        this.cam = Object.assign({}, this.cam, cam);
        this.frames = list.slice().sort((a, b) => String(a.captured_at).localeCompare(String(b.captured_at)));
        let i = this.frames.findIndex((f) => f.id === curId);
        if (atEnd || i < 0) i = this.frames.length - 1;
        // Кадр мог дообработаться — сбрасываем его кэш, если статус сменился.
        const f = this.frames[i];
        if (f && this.cache[f.id] && this.cache[f.id].status !== f.status) delete this.cache[f.id];
        if (i !== this.idx || !this.cache[f.id]) await this.go(i);
      } catch (e) { /* следующий опрос попробует снова */ }
    },

    onKey(e) {
      if (SV.isTyping(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      // Пока открыта модалка (снимок, подсказка), стрелки принадлежат ей.
      if ([...document.querySelectorAll("[aria-modal=true]")].some((el) => el.getClientRects().length)) return;
      const k = e.key.toLowerCase();
      if (e.key === "ArrowLeft") { e.preventDefault(); this.step(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); this.step(1); }
      else if (e.key === "Home") { e.preventDefault(); this.go(0); }
      else if (e.key === "End") { e.preventDefault(); this.go(this.frames.length - 1); }
      else if (k === "b" || k === "и") this.layers.boxes = !this.layers.boxes;
      else if (k === "m" || k === "ь") this.toggleMask();
      else if (k === "z" || k === "я") this.layers.zones = !this.layers.zones;
      else if (e.key === "Escape") { this.draft = null; }
      else if (e.key === "Enter" && this.draft) this.closeDraft();
      else if (e.key === "Backspace" && this.draft && this.draft.points.length) { e.preventDefault(); this.draft.points.pop(); }
    },

    // ------------------------------------------------------------ кадры
    async go(i) {
      if (!this.frames || !this.frames.length) return;
      i = Math.max(0, Math.min(this.frames.length - 1, i));
      this.idx = i;
      const f = this.frames[i];
      history.replaceState(null, "", `?frame=${f.id}${location.hash}`);
      this.frameError = null;
      if (this.cache[f.id]) { this.frame = this.cache[f.id]; this.prefetch(i); return; }
      this.loadingFrame = true;
      try {
        const d = await SV.api.get(`/api/frames/${f.id}`);
        this.cache[f.id] = d;
        if (this.idx === i) this.frame = d;
        this.prefetch(i);
      } catch (e) {
        this.frameError = e.message;
      } finally {
        this.loadingFrame = false;
      }
    },
    /* Соседние кадры тянем заранее — листание стрелками без мигания. */
    prefetch(i) {
      for (const j of [i - 1, i + 1]) {
        const f = this.frames[j];
        if (!f || this.cache[f.id]) continue;
        SV.api.get(`/api/frames/${f.id}`).then((d) => { this.cache[f.id] = d; new Image().src = d.url; }).catch(() => {});
      }
    },
    step(d) { this.go(this.idx + d); },
    get cur() { return this.frames && this.frames[this.idx]; },
    /* Слой маски — GET /api/cameras/{id}/mask.png (погашенный фон — полупрозрачный красный). */
    get maskUrl() { return this.cam && this.cam.mask ? `${this.cam.mask.url}?t=${encodeURIComponent(this.cam.mask.updated_at || "")}` : null; },
    toggleMask() {
      if (!this.maskUrl) return SV.toast("Маска для этой камеры ещё не построена — она копится по дневным кадрам.", { kind: "info" });
      this.layers.mask = !this.layers.mask;
    },
    async resetMask() {
      if (!confirm("Сбросить динамическую маску камеры? Она заново накопится по следующим дневным кадрам.")) return;
      try {
        await SV.api.del(`/api/cameras/${cameraId}/mask`);
        this.cam = Object.assign({}, this.cam, { mask: null });
        this.layers.mask = false;
        SV.toast("Маска сброшена — начнёт копиться заново", { kind: "success" });
      } catch (e) { SV.toastError(e, "Маска не сброшена"); }
    },

    // ------------------------------------------------------------ шкала времени
    get span() {
      if (!this.frames || this.frames.length < 2) return null;
      const a = SV.fmt.toDate(this.frames[0].captured_at).getTime();
      const b = SV.fmt.toDate(this.frames[this.frames.length - 1].captured_at).getTime();
      return { a, b: Math.max(b, a + 60000) };
    },
    tickPos(f) {
      const s = this.span;
      if (!s) return 50;
      return ((SV.fmt.toDate(f.captured_at).getTime() - s.a) / (s.b - s.a)) * 100;
    },
    /* Штрих шкалы: цвет — состояние кадра, высота — сколько техники на нём (detections_count). */
    tickClass(f) {
      if (f.status === "pending" || f.status === "processing") return "bg-info/40";
      if (f.status === "error") return "bg-bad";
      if (f.quality_ok === false || (f.weather && !["clear", "unknown"].includes(f.weather))) return "bg-warn";
      if (f.is_night) return "bg-fg-3/35";
      if (f.detections_count) return "bg-good";
      return "bg-fg-3";
    },
    tickHeight(f) { return 6 + Math.min(6, f.detections_count || 0) * 2.5; },
    get queued() { return (this.frames || []).filter((f) => f.status === "pending" || f.status === "processing").length; },
    get dayMarks() {
      const s = this.span;
      if (!s) return [];
      const out = [];
      const seen = new Set();
      for (const f of this.frames) {
        const day = SV.fmt.ymd(f.captured_at);
        if (seen.has(day)) continue;
        seen.add(day);
        const mid = new Date(`${day}T00:00:00+03:00`).getTime();
        const left = ((Math.max(mid, s.a) - s.a) / (s.b - s.a)) * 100;
        out.push({ day, left, label: SV.fmt.date(day) + (day === SV.fmt.ymd(new Date()) ? " · сегодня" : "") });
      }
      // Длинный архив (сотни дней): подписываем не каждый день, а месяцы.
      if (out.length > 14) {
        const months = [];
        let last = "";
        for (const d of out) { const m = d.day.slice(0, 7); if (m !== last) { months.push(Object.assign({}, d, { label: SV.fmt.date(d.day) })); last = m; } }
        const step = Math.ceil(months.length / 10);
        return months.filter((_, i) => i % step === 0);
      }
      return out;
    },
    scrubAt(ev) {
      const r = this.$refs.scrub.getBoundingClientRect();
      const x = (ev.clientX - r.left) / r.width;
      const s = this.span;
      if (!s) return;
      const t = s.a + x * (s.b - s.a);
      let best = 0, bd = Infinity;
      this.frames.forEach((f, i) => {
        const dd = Math.abs(SV.fmt.toDate(f.captured_at).getTime() - t);
        if (dd < bd) { bd = dd; best = i; }
      });
      if (best !== this.idx) this.go(best);
    },
    scrubDown(ev) {
      this.$refs.scrub.setPointerCapture(ev.pointerId);
      this.scrubbing = true;
      this.scrubAt(ev);
    },
    scrubbing: false,

    // ------------------------------------------------------------ оверлей
    get svg() {
      return SV.boxesSvg(this.frame, {
        showBoxes: this.layers.boxes, showZones: this.layers.zones || this.side === "zones", zones: this.zones, highlight: this.highlight,
        draft: this.draft ? this.draft.points : null,
        points: this.side === "calib" ? this.calib.points : null,
      });
    },
    get labels() { this.unitsTick; return SV.boxLabels(this.frame, { showBoxes: this.layers.boxes, dispW: this.dispW }); },
    get reprojError() { return this.cam && this.cam.calib_points ? this.cam.calib_points.reproj_error : null; },
    get pointLabels() {
      if (this.side !== "calib" || !this.frame) return [];
      const W = this.frame.width, H = this.frame.height;
      return this.calib.points.map((p, i) => ({ i, style: `left:${(p[0] / W) * 100}%;top:${(p[1] / H) * 100}%` }));
    },
    get zoneLabels() {
      if (!this.frame || !(this.layers.zones || this.side === "zones")) return [];
      const W = this.frame.width, H = this.frame.height;
      return this.zones.map((z) => {
        const pts = z.polygon || [];
        const cx = pts.reduce((a, p) => a + p[0], 0) / (pts.length || 1);
        const cy = pts.reduce((a, p) => a + p[1], 0) / (pts.length || 1);
        return { id: z.id, name: z.name, style: `left:${(cx / W) * 100}%;top:${(cy / H) * 100}%` };
      });
    },
    get picking() { return this.side === "calib" || !!this.draft; },
    onViewerClick(ev) {
      if (!this.frame || !this.picking) return;
      const p = SV.imagePoint(ev, this.$refs.img, this.frame);
      if (this.draft) {
        const pts = this.draft.points;
        if (pts.length >= 3) {
          const first = pts[0];
          const r = this.$refs.img.getBoundingClientRect();
          const scale = r.width / this.frame.width;
          if (Math.hypot((p[0] - first[0]) * scale, (p[1] - first[1]) * scale) < 12) return this.closeDraft();
        }
        pts.push(p);
      } else if (this.side === "calib") {
        this.calib.points.push(p);
        this.calib.site.push(["", ""]);
        this.calib.result = null;
      }
    },

    // ------------------------------------------------------------ загрузка
    addFiles(list) {
      const ok = [...list].filter((f) => /^(image|video)\//.test(f.type) || /\.(zip|jpe?g|png|webp|mp4|avi|mov|mkv)$/i.test(f.name));
      if (ok.length < list.length) SV.toast("Часть файлов пропущена: нужны изображения, zip или видео", { kind: "warning" });
      this.files = [...this.files, ...ok];
      this.upState = null;
    },
    get hasVideo() { return this.files.some((f) => /^video\//.test(f.type) || /\.(mp4|avi|mov|mkv)$/i.test(f.name)); },
    get filesSize() { return this.files.reduce((a, f) => a + f.size, 0); },
    fileSize(n) { return n > 1048576 ? `${SV.fmt.num(n / 1048576, 1)} МБ` : `${SV.fmt.num(n / 1024)} КБ`; },
    async upload() {
      if (!this.files.length) return;
      const fd = new FormData();
      for (const f of this.files) fd.append("files", f);
      if (this.upOpts.interval_min) fd.append("interval_min", String(this.upOpts.interval_min));
      // Время без зоны бэкенд трактует как время площадки (sites.timezone) — отдаём как ввёл пользователь.
      if (this.upOpts.start_at) fd.append("start_at", this.upOpts.start_at);
      if (this.hasVideo) {
        fd.append("video_mode", this.upOpts.video_mode);
        if (this.upOpts.step_s) fd.append("step_s", String(this.upOpts.step_s));
      }
      this.upState = { phase: "upload", p: 0, job: null };
      try {
        const r = await SV.api.upload(`/api/cameras/${cameraId}/upload`, fd, (p) => { this.upState.p = p; });
        this.upState = { phase: "job", p: 1, job: { state: "ingesting", total: 0, done: 0, errors: [] } };
        Alpine.store("app").loadQueue();
        let seen = 0;
        const job = await SV.api.pollJob(r.job_id, (j) => {
          this.upState.job = j;
          // Новые кадры появляются на шкале по мере разбора, а не в конце.
          if (j.done - seen >= 5) { seen = j.done; this.reloadFrames(false); }
        }, { interval: 1500 });
        const failed = ["error", "failed"].includes(job.state);
        this.upState.phase = failed ? "error" : "done";
        if (!failed) {
          const extra = [job.duplicates ? `дубликатов ${job.duplicates}` : null, job.skipped ? `пропущено ${job.skipped}` : null, job.postponed ? `ждут провайдера ${job.postponed}` : null].filter(Boolean).join(", ");
          SV.toast(`Разобрано кадров: ${job.done}${extra ? " (" + extra + ")" : ""}`, { kind: job.postponed ? "warning" : "success" });
          this.files = [];
          await this.reloadFrames();
        }
      } catch (e) {
        this.upState = { phase: "error", p: 0, job: { errors: [e.message] } };
      }
    },
    async reloadFrames(jump = true) {
      try {
        const [list, cam] = await Promise.all([SV.api.get(`/api/cameras/${cameraId}/frames?limit=500`), SV.api.get(`/api/cameras/${cameraId}`)]);
        this.cam = Object.assign({}, this.cam, cam);
        this.frames = (Array.isArray(list) ? list : list.items || []).slice().sort((a, b) => String(a.captured_at).localeCompare(String(b.captured_at)));
        if (this.frames.length && (jump || this.idx < 0)) await this.go(this.frames.length - 1);
      } catch (e) { /* кадры появятся при следующем обновлении */ }
    },

    // ------------------------------------------------------------ поток
    async connectStream() {
      this.stream.busy = true;
      try {
        const body = { source_uri: this.stream.uri.trim() };
        if (this.stream.interval) body.interval = Number(this.stream.interval);
        const r = await SV.api.post(`/api/cameras/${cameraId}/connect`, body);
        this.stream.info = r.camera || null;
        this.cam = Object.assign({}, this.cam, { kind: "stream", source_uri: body.source_uri });
        SV.toast("Камера подключена: кадры пойдут на /api/ingest", { kind: "success" });
      } catch (e) { SV.toastError(e, "Камера не подключилась"); } finally { this.stream.busy = false; }
    },
    async disconnectStream() {
      this.stream.busy = true;
      try {
        await SV.api.post(`/api/cameras/${cameraId}/disconnect`, {});
        this.stream.info = null;
        SV.toast("Съёмка остановлена", { kind: "success" });
      } catch (e) { SV.toastError(e, "Не удалось остановить"); } finally { this.stream.busy = false; }
    },
    copy(text) {
      try { navigator.clipboard.writeText(text); SV.toast("Скопировано", { kind: "success" }); } catch (e) { SV.toastError(e); }
    },

    // ------------------------------------------------------------ калибровка
    get calibReady() {
      return this.calib.points.length >= 4 && this.calib.site.every((p) => p[0] !== "" && p[1] !== "" && !isNaN(p[0]) && !isNaN(p[1]));
    },
    removePoint(i) { this.calib.points.splice(i, 1); this.calib.site.splice(i, 1); this.calib.result = null; },
    clearCalib() { this.calib.points = []; this.calib.site = []; this.calib.result = null; },
    async saveCalib() {
      this.calib.busy = true;
      try {
        const res = await SV.api.post(`/api/cameras/${cameraId}/calibration`, {
          image_points: this.calib.points, site_points: this.calib.site.map((p) => [Number(p[0]), Number(p[1])]),
        });
        this.calib.result = res;
        this.cam = Object.assign({}, this.cam, res.camera || { calibrated: true });
        SV.toast("Калибровка сохранена: машины с этой камеры склеиваются с соседними", { kind: "success" });
      } catch (e) {
        SV.toastError(e, "Калибровка не сохранена");
      } finally {
        this.calib.busy = false;
      }
    },
    /* PATCH /api/cameras/{id} {homography: null} — бэкенд заодно стирает calib_points. */
    async resetCalib() {
      if (!confirm("Снять калибровку камеры? Машины с неё перестанут склеиваться с соседними камерами.")) return;
      try {
        this.cam = Object.assign({}, this.cam, await SV.api.patch(`/api/cameras/${cameraId}`, { homography: null }));
        this.clearCalib();
        SV.toast("Калибровка снята", { kind: "success" });
      } catch (e) { SV.toastError(e, "Не удалось снять калибровку"); }
    },
    errVerdict(e) {
      if (e == null) return null;
      if (e <= 1) return { tone: "good", text: "точно — склейка между камерами надёжна" };
      if (e <= 3) return { tone: "warn", text: "терпимо — точки лучше разнести по кадру" };
      return { tone: "bad", text: "грубо — проверьте координаты точек" };
    },

    // ------------------------------------------------------------ зоны
    startZone() { this.draft = { points: [], name: "", kind: "work" }; this.side = "zones"; },
    closeDraft() {
      if (!this.draft || this.draft.points.length < 3) return SV.toast("Зона — минимум 3 точки", { kind: "warning" });
      this.draft.closed = true;
    },
    async saveZone() {
      const d = this.draft;
      if (!d || d.points.length < 3) return;
      try {
        const z = await SV.api.post(`/api/cameras/${cameraId}/zones`, { name: d.name.trim() || "Зона", kind: d.kind, polygon: d.points });
        this.zones = [...this.zones, z];
        this.draft = null;
        this.layers.zones = true;
        SV.toast(`Зона «${z.name}» сохранена`, { kind: "success" });
      } catch (e) {
        SV.toastError(e, "Зона не сохранена");
      }
    },
    async deleteZone(z) {
      if (!confirm(`Удалить зону «${z.name}»?`)) return;
      try {
        await SV.api.del(`/api/cameras/${cameraId}/zones/${encodeURIComponent(z.id)}`);
        this.zones = this.zones.filter((x) => x.id !== z.id);
      } catch (e) {
        SV.toastError(e, "Зона не удалена");
      }
    },
    setSide(s) {
      this.side = s;
      history.replaceState(null, "", `${location.search}#${s}`);
      if (s !== "zones") this.draft = null;
    },
  }));
});
