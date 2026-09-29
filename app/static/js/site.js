/*
 * Страница объекта: вкладки «Сводка», «Техника», «Отклонения», «Камеры», «План».
 *
 * Данные (формы — app/services/views.py бэкенда):
 *   GET /api/sites/{id}/overview    — всё для сводки одним запросом (report, stages, equipment, deviations, cameras, series);
 *   GET /api/sites/{id}/cameras     — камеры с размерами кадра, числом кадров, маской, очередью;
 *   GET /api/sites/{id}/equipment   — {units, balances}: единицы техники и «полоски» по этапам;
 *   GET /api/sites/{id}/hours       — журнал интервалов (manual=false) и ручные поправки (по умолчанию);
 *   GET /api/sites/{id}/deviations  — лента отклонений со статусами open / ack / resolved;
 *   GET /api/units/{id}             — последние детекции единицы (кроп для списка).
 *
 * «Сегодня» — не часы браузера, а report.now бэкенда: для архива 2008 года это
 * время последнего кадра, иначе просрочка на Ганте считалась бы от 2026-го.
 * Вкладка — в адресе (#equipment): ссылкой можно отправить коллеге именно «технику».
 */
document.addEventListener("alpine:init", () => {
  const TABS = ["summary", "equipment", "deviations", "cameras", "plan"];
  const DAY = 86400000;

  Alpine.data("sitePage", (siteId) => ({
    siteId,
    tab: "summary",
    ov: null,
    cams: null,
    firstFrameAt: null,
    loading: true,
    error: null,
    reprocessing: null,

    // техника
    units: null,
    balances: null,
    unitsError: null,
    unitFilter: "all",
    unitDets: {},
    intervals: null,
    heat: null,
    hoursStageSel: "now",
    corrections: null,
    fix: null, // форма ручной поправки часов {cls, hours, stage_id, note, sign}

    // отклонения
    devs: null,
    devStatus: "open",
    devSev: { critical: true, warning: true, info: true },

    // камеры
    camFrames: {},

    // гант
    ganttRange: "all",
    hover: null,

    async init() {
      const h = location.hash.replace("#", "");
      if (TABS.includes(h)) this.tab = h;
      window.addEventListener("hashchange", () => {
        const x = location.hash.replace("#", "");
        if (TABS.includes(x)) this.select(x, false);
      });
      window.addEventListener("keydown", (e) => {
        if (SV.isTyping(e) || e.metaKey || e.ctrlKey || e.altKey) return;
        if ([...document.querySelectorAll("[aria-modal=true]")].some((el) => el.getClientRects().length)) return;
        const n = Number(e.key);
        if (n >= 1 && n <= TABS.length) this.select(TABS[n - 1]);
      });
      Alpine.store("app").loadCatalog();
      await this.load();
      this.onTab();
      // Пока кадры объекта в анализе — обновляемся часто (вердикт «проявляется» на глазах), иначе раз в минуту.
      const tick = async () => {
        if (!document.hidden) await this.refresh();
        this._t = setTimeout(tick, this.pendingFrames ? 8000 : 60000);
      };
      this._t = setTimeout(tick, this.pendingFrames ? 8000 : 60000);
    },
    destroy() { clearTimeout(this._t); },

    async load(silent = false) {
      if (!silent) this.loading = true;
      try {
        const [ov, cams] = await Promise.all([
          SV.api.get(`/api/sites/${siteId}/overview`),
          SV.api.get(`/api/sites/${siteId}/cameras`).catch(() => null),
        ]);
        this.ov = ov;
        if (cams) this.cams = cams;
        if (!this.firstFrameAt && cams && cams.length) this.loadFirstFrame(cams);
        this.error = null;
        document.title = `${this.ov.site.name} · СтройВзор`;
        this.$nextTick(() => this.drawCurve());
      } catch (e) {
        if (!silent) this.error = e.message;
      } finally {
        this.loading = false;
      }
    },
    /* Начало наблюдений — самый ранний кадр объекта (по кадру на камеру, order=asc&limit=1).
       Нужен Ганту: этап, который «уже шёл на первом кадре», модель Б отдаёт без actual_start. */
    async loadFirstFrame(cams) {
      const res = await Promise.allSettled(cams.slice(0, 12).map((c) => SV.api.get(`/api/cameras/${c.id}/frames?order=asc&limit=1`)));
      const ts = res.filter((r) => r.status === "fulfilled" && r.value && r.value[0]).map((r) => r.value[0].captured_at).sort();
      if (ts.length) {
        this.firstFrameAt = ts[0];
        if (this._curve) { this._curve.update(); }
      }
    },
    get firstFrameIso() { return this.firstFrameAt ? SV.fmt.ymd(this.firstFrameAt) : null; },
    /* Фоновое обновление: сводка + то, что уже открывали. */
    async refresh() {
      await this.load(true);
      if (this.units) this.loadUnits(false);
      if (this.devs) this.loadDevs();
    },

    select(t, push = true) {
      this.tab = t;
      if (push) history.replaceState(null, "", `#${t}`);
      this.onTab();
    },
    onTab() {
      if (this.tab === "equipment") this.loadEquipmentTab();
      if (this.tab === "deviations" && !this.devs) this.loadDevs();
      // Подписи техники в отклонениях («Экскаватор №1», а не u0001) берём из списка единиц.
      if (this.tab === "deviations" && !this.units) this.loadUnits(false);
      if (this.tab === "cameras") { this.loadCamFrames(); if (!this.units) this.loadUnits(false); }
      if (this.tab === "summary") this.$nextTick(() => this.drawCurve());
    },

    // ------------------------------------------------------------ общее
    get report() { return (this.ov && this.ov.report) || {}; },
    get series() { return (this.ov && this.ov.series) || {}; },
    get vmeta() { return SV.META.verdict[this.report.verdict] || SV.META.verdict.no_data; },
    /* «Сейчас» аналитики: живая площадка — часы сервера, архив — время последнего кадра. */
    get now() { return SV.fmt.toDate(this.report.now) || new Date(); },
    get nowMs() { return this.now.getTime(); },
    get isArchive() { return Date.now() - this.nowMs > 3 * DAY; },
    get todayIso() { return this.series.today || SV.fmt.ymd(this.now); },
    get stages() { return (this.ov && this.ov.stages) || []; },
    get planStages() { return this.stages.filter((s) => s.in_plan !== false || s.status !== "not_started"); },
    get currentStage() {
      const id = this.report.current_stage;
      return this.stages.find((s) => s.id === id) || this.stages.find((s) => s.status === "active") || null;
    },
    get openDevs() { return ((this.ov && this.ov.deviations) || []).filter((d) => !d.status || d.status === "open"); },
    get devCounts() {
      const c = { critical: 0, warning: 0, info: 0 };
      for (const d of this.openDevs) c[d.severity] = (c[d.severity] || 0) + 1;
      return c;
    },
    get topDevs() {
      return [...this.openDevs].sort((a, b) => (SV.META.severity[a.severity]?.rank ?? 9) - (SV.META.severity[b.severity]?.rank ?? 9)).slice(0, 4);
    },
    get unitsNow() {
      const eq = (this.ov && this.ov.equipment) || [];
      const s = { active: 0, idle: 0, parked: 0, departed: 0 };
      for (const r of eq) { s.active += r.active || 0; s.idle += r.idle || 0; s.parked += r.parked || 0; s.departed += r.departed || 0; }
      s.total = s.active + s.idle + s.parked;
      return s;
    },
    get lastFrameAt() { return (this.ov && this.ov.site && this.ov.site.last_frame_at) || null; },
    get pendingFrames() { return ((this.ov && this.ov.cameras) || []).reduce((a, c) => a + (c.pending || 0), 0); },
    get plannedFinish() {
      return this.series.planned_finish || this.stages.map((s) => s.planned_end).filter(Boolean).sort().pop() || null;
    },
    get forecastFinish() { return this.report.forecast_finish || this.series.forecast_finish || null; },
    get finishShift() { return SV.fmt.daysBetween(this.plannedFinish, this.forecastFinish); },
    /* Время относительно «сейчас» аналитики: для архива «за 3 ч до последнего кадра», а не «18 лет назад». */
    relNow(v) {
      const d = SV.fmt.toDate(v);
      if (!d) return "—";
      if (!this.isArchive) return SV.fmt.rel(v);
      const h = (this.nowMs - d.getTime()) / 3600000;
      if (h < 0.1) return "последний кадр";
      if (h < 48) return `за ${SV.fmt.dur(h)} до последнего кадра`;
      return SV.fmt.dateTime(v);
    },
    stageMeta(s) { return SV.META.stage[s.status] || SV.META.stage.not_started; },
    stageOverdue(s) {
      return s.status !== "done" && s.planned_end && s.planned_end < this.todayIso ? SV.fmt.daysBetween(s.planned_end, this.todayIso) : 0;
    },
    stageLate(s) {
      return s.status === "not_started" && s.planned_start && s.planned_start < this.todayIso ? SV.fmt.daysBetween(s.planned_start, this.todayIso) : 0;
    },
    worksText(s) { return (s.works || []).map((w) => `${w.code} ${w.name || ""}`.trim()).join("; "); },
    camName(id) {
      const all = [...(this.cams || []), ...((this.ov && this.ov.cameras) || [])];
      const c = all.find((x) => String(x.id) === String(id));
      return c ? c.name : `Камера ${id}`;
    },
    cam(id) { return (this.cams || []).find((x) => String(x.id) === String(id)) || null; },

    /* POST /api/sites/{id}/reprocess → {job_id, frames, model_a, model_b}; прогресс — GET /api/jobs/{id}. */
    async reprocess() {
      try {
        const r = await SV.api.post(`/api/sites/${siteId}/reprocess`, {});
        this.reprocessing = { state: "processing", done: 0, total: r.frames || 0 };
        SV.toast(`Переанализ ${r.frames || ""} кадров: ${SV.META.providers[r.model_a]?.label || r.model_a} + ${SV.META.providers[r.model_b]?.label || r.model_b}. Старые результаты сохранятся.`, { kind: "info" });
        Alpine.store("app").loadQueue();
        if (r && r.job_id) {
          const j = await SV.api.pollJob(r.job_id, (x) => { this.reprocessing = x; }, { interval: 2000 });
          if (j.state === "postponed") SV.toast(`Часть кадров ждёт провайдера: ${j.postponed_reason || "провайдер не готов"}`, { kind: "warning" });
          else SV.toast("Переанализ закончен", { kind: "success" });
          await this.refresh();
        }
      } catch (e) {
        SV.toastError(e, "Переанализ не запущен");
      } finally {
        this.reprocessing = null;
      }
    },

    // ------------------------------------------------------------ Гант
    get gantt() {
      const st = this.stages.filter((s) => s.planned_start || s.actual_start);
      if (!st.length) return null;
      const all = [];
      for (const s of st) all.push(s.planned_start, s.planned_end, s.actual_start, s.actual_end);
      all.push(this.todayIso, this.forecastFinish);
      const dates = all.filter(Boolean).map((d) => SV.fmt.toDate(d).getTime());
      let a = Math.min(...dates), b = Math.max(...dates);
      if (this.ganttRange === "near") {
        const t = SV.fmt.toDate(this.todayIso).getTime();
        a = t - 75 * DAY; b = t + 105 * DAY;
      }
      const pad = (b - a) * 0.02;
      a -= pad; b += pad;
      const pos = (d) => d ? ((SV.fmt.toDate(d).getTime() - a) / (b - a)) * 100 : null;
      const clamp = (x) => Math.max(0, Math.min(100, x));
      const seg = (x, y) => {
        if (x == null || y == null) return null;
        const l = clamp(x), r = clamp(y);
        if (y < 0 || x > 100 || y < x) return null; // целиком вне окна шкалы
        return { left: Math.min(l, 99.2), width: Math.max(0.8, r - l) };
      };
      // Шкала: месяцы, а на длинных планах — кварталы или годы, чтобы подписи не слипались.
      const months = [];
      const d0 = new Date(a);
      let m = new Date(Date.UTC(d0.getUTCFullYear(), d0.getUTCMonth() + 1, 1, 12));
      const span = (b - a) / DAY;
      const step = span > 1500 ? 12 : span > 700 ? 3 : span > 400 ? 2 : 1;
      while (m.getTime() < b) {
        if (m.getUTCMonth() % step === 0) {
          const iso = m.toISOString().slice(0, 10);
          const label = new Intl.DateTimeFormat("ru-RU", { month: "short", timeZone: "UTC" }).format(m).replace(".", "");
          months.push({ left: pos(iso), label: step === 12 ? String(m.getUTCFullYear()) : m.getUTCMonth() === 0 || !months.length ? `${label} ${String(m.getUTCFullYear()).slice(2)}` : label });
        }
        m = new Date(Date.UTC(m.getUTCFullYear(), m.getUTCMonth() + 1, 1, 12));
      }
      const today = pos(this.todayIso);
      // подпись месяца под плашкой «сегодня» не читается — прячем соседнюю
      for (const mm of months) mm.hidden = today != null && Math.abs(mm.left - today) < 5;
      const first = this.firstFrameIso;
      const rows = this.stages.map((s) => {
        const plan = seg(pos(s.planned_start), pos(s.planned_end));
        // Этап уже шёл на первом кадре: факт рисуем от начала наблюдений (с «рваным» левым краем).
        const startedBefore = !s.actual_start && s.status === "active" && !!first;
        const factStart = s.actual_start || (startedBefore ? first : null);
        const factEnd = s.actual_end || (factStart ? this.todayIso : null);
        const fact = seg(pos(factStart), pos(factEnd));
        const over = this.stageOverdue(s);
        const overSeg = over && s.planned_end ? seg(pos(s.planned_end), today) : null;
        const lateEnd = s.status === "done" && s.actual_end && s.planned_end && s.actual_end > s.planned_end
          ? seg(pos(s.planned_end), pos(s.actual_end)) : null;
        const lateStart = this.stageLate(s) ? seg(pos(s.planned_start), today) : null;
        // Этап сделан, но даты факта неизвестны (был готов до первого кадра) — не рисуем «факт», только отметку.
        const doneBefore = s.status === "done" && !s.actual_start && !s.actual_end;
        const firstPos = first ? pos(first) : null;
        return { s, plan, fact, overSeg: overSeg || lateEnd, lateStart, doneBefore, startedBefore, firstPos };
      });
      return { rows, months, today: today >= 0 && today <= 100 ? today : null };
    },
    factClass(s) {
      if (s.status === "done") return "bg-good";
      if (this.stageOverdue(s)) return "bg-bad";
      return "bg-info";
    },
    hoverRow(r, ev) {
      const box = this.$refs.gantt.getBoundingClientRect();
      this.hover = { r, x: Math.max(8, Math.min(ev.clientX - box.left + 14, box.width - 280)), y: ev.clientY - box.top + 14 };
    },

    // ------------------------------------------------------------ кривая готовности
    drawCurve() {
      const el = this.$refs.curve;
      if (!el || !this.ov || !this.ov.series) return;
      const self = this;
      if (this._curve) { this._curve.update(); return; }
      this._curve = SV.charts.chart(el, () => {
        const C = SV.charts;
        const s = self.series;
        const days = s.days || [];
        if (!days.length) return null;
        const pts = (arr) => days.map((d, i) => [d, arr && arr[i] != null ? +(arr[i] * 100).toFixed(2) : null]);
        // Факт до первого кадра — не наблюдение, а экстраполяция: не рисуем.
        const first = self.firstFrameIso;
        const factPts = (arr) => pts(arr).map((p) => (first && p[0] < first ? [p[0], null] : p));
        const today = self.todayIso;
        const spanDays = days.length;
        // Прогноз: от сегодняшнего факта до 100 % в день прогноза окончания.
        let forecast = null;
        const ti = days.indexOf(today);
        const fin = self.forecastFinish;
        if (fin && ti >= 0 && s.actual && s.actual[ti] != null) forecast = [[today, +(s.actual[ti] * 100).toFixed(2)], [fin, 100]];
        const monthFmt = new Intl.DateTimeFormat("ru-RU", { month: "short", timeZone: "UTC" });
        return Object.assign(C.base(), {
          grid: { left: 4, right: 16, top: 16, bottom: 4, containLabel: true },
          tooltip: Object.assign(C.base().tooltip, {
            trigger: "axis",
            axisPointer: { type: "line", lineStyle: { color: C.t("fg-3"), width: 1 } },
            formatter(ps) {
              const d = ps[0] && ps[0].axisValue;
              const v = {};
              ps.forEach((p) => { v[p.seriesName] = p.value[1]; });
              let h = `<div style="font-weight:600;margin-bottom:4px">${SV.fmt.dateLong(new Date(d).toISOString().slice(0, 10))}</div>`;
              const row = (name, val, color, dash) => val == null ? "" :
                `<div style="display:flex;gap:10px;align-items:center;justify-content:space-between"><span style="display:flex;align-items:center;gap:6px"><span style="width:12px;height:0;border-top:2px ${dash ? "dashed" : "solid"} ${color}"></span>${name}</span><b style="font-family:JetBrains MonoVariable,monospace">${SV.fmt.num(val, 1)} %</b></div>`;
              h += row("План", v["План"], C.t("fg-3"), true);
              h += row("Факт", v["Факт"], C.t("info"));
              if (v["План"] != null && v["Факт"] != null) {
                const gap = v["Факт"] - v["План"];
                h += `<div style="margin-top:4px;color:${gap < -0.5 ? C.t("bad-fg") : C.t("fg-3")}">${gap < 0 ? "Отставание" : "Опережение"} ${SV.fmt.num(Math.abs(gap), 1)} п. п.</div>`;
              }
              return h;
            },
          }),
          xAxis: Object.assign(C.axisCommon(), {
            type: "time", splitLine: { show: false },
            axisLabel: {
              color: C.t("fg-3"), fontSize: 11, hideOverlap: true,
              formatter: (v) => {
                const d = new Date(v);
                const mo = monthFmt.format(d).replace(".", "");
                return spanDays > 300 && d.getUTCMonth() === 0 ? `${mo} ${d.getUTCFullYear()}` : spanDays > 700 ? (d.getUTCMonth() % 3 === 0 ? `${mo} ${String(d.getUTCFullYear()).slice(2)}` : "") : mo;
              },
            },
          }),
          yAxis: Object.assign(C.axisCommon(), {
            type: "value", min: 0, max: 100, interval: 25, axisLine: { show: false },
            axisLabel: { color: C.t("fg-3"), fontSize: 11, formatter: "{value}%" },
          }),
          series: [
            { name: "План", type: "line", showSymbol: false, data: pts(s.expected), lineStyle: { width: 2, type: [5, 4], color: C.t("fg-3") }, itemStyle: { color: C.t("fg-3") }, z: 2 },
            {
              name: "Факт", type: "line", showSymbol: false, connectNulls: false, data: factPts(s.actual), lineStyle: { width: 2, color: C.t("info") }, itemStyle: { color: C.t("info") }, z: 3,
              areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: C.t("info", 0.22) }, { offset: 1, color: C.t("info", 0) }]) },
              // Точка «факт на сегодня» с подписью: у короткого архива линия факта — пара дней на шкале в месяцы.
              markPoint: ti >= 0 && s.actual && s.actual[ti] != null ? {
                silent: true, symbol: "circle", symbolSize: 9, itemStyle: { color: C.t("info"), borderColor: C.t("panel"), borderWidth: 2 },
                label: { show: true, position: "left", distance: 8, color: C.t("fg"), fontSize: 11, fontWeight: 600,
                         formatter: () => `факт ${SV.fmt.pct(s.actual[ti], 1)}` },
                data: [{ coord: [today, +(s.actual[ti] * 100).toFixed(2)] }],
              } : undefined,
              markLine: {
                silent: true, symbol: "none", label: { show: true, formatter: "{b}", color: C.t("fg-2"), fontSize: 10, position: "end", distance: 3 },
                lineStyle: { color: C.t("fg-2"), type: "solid", width: 1 },
                // «Начало съёмки» — только если оно заметно раньше «сегодня», иначе подписи слипнутся.
                data: [{ xAxis: today, name: self.isArchive ? SV.fmt.date(today) : "сегодня" }].concat(
                  first && first > days[0] && SV.fmt.daysBetween(first, today) > spanDays * 0.12
                    ? [{ xAxis: first, name: "начало съёмки", lineStyle: { type: "dashed", color: C.t("fg-3") } }] : []),
              },
            },
            forecast ? { name: "Прогноз", type: "line", showSymbol: false, data: forecast, lineStyle: { width: 2, type: [2, 4], color: C.t("info", 0.7) }, itemStyle: { color: C.t("info", 0.7) }, z: 2, tooltip: { show: false } } : null,
          ].filter(Boolean),
        });
      });
    },

    // ------------------------------------------------------------ техника
    async loadEquipmentTab() {
      await Promise.all([
        !this.units ? this.loadUnits(true) : null,
        !this.intervals ? this.loadIntervals() : null,
        !this.corrections ? this.loadCorrections() : null,
      ]);
      this.$nextTick(() => this.drawHeat());
    },
    async loadUnits(withCrops = true) {
      this.unitsError = null;
      try {
        const data = await SV.api.get(`/api/sites/${siteId}/equipment`);
        this.units = data.units || [];
        this.balances = data.balances || [];
        SV.setUnits(this.units);
        if (withCrops) this.loadUnitCrops();
      } catch (e) {
        this.unitsError = e.message;
      }
    },
    /* Кроп машины в списке: последняя детекция единицы (GET /api/units/{id}). Не больше 24 запросов. */
    async loadUnitCrops() {
      const list = this.shownUnitsAll.filter((u) => !this.unitDets[u.id]).slice(0, 24);
      await Promise.all(list.map(async (u) => {
        try {
          const d = await SV.api.get(`/api/units/${u.id}`);
          const det = (d.detections || [])[0];
          if (det) this.unitDets = Object.assign({}, this.unitDets, { [u.id]: det });
        } catch (e) { /* кроп — украшение, без него строка остаётся */ }
      }));
    },
    /* Журнал интервалов работы (включая ручные поправки) — для тепловой карты и «сегодня». */
    async loadIntervals() {
      try {
        this.intervals = await SV.api.get(`/api/sites/${siteId}/hours?manual=false`);
      } catch (e) {
        this.intervals = [];
      }
      this.buildHeat();
    },
    async loadCorrections() {
      try { this.corrections = await SV.api.get(`/api/sites/${siteId}/hours`); } catch (e) { this.corrections = []; }
    },
    buildHeat() {
      const days = 14;
      const end = SV.fmt.toDate(this.todayIso);
      const dayKeys = [];
      for (let i = days - 1; i >= 0; i--) dayKeys.push(SV.fmt.ymd(new Date(end.getTime() - i * DAY)));
      const grid = dayKeys.map(() => new Array(24).fill(0));
      for (const iv of this.intervals || []) {
        if (iv.manual) continue;
        let a = SV.fmt.toDate(iv.start).getTime();
        const b = SV.fmt.toDate(iv.end).getTime();
        while (a < b) {
          const hourEnd = Math.min(b, (Math.floor(a / 3600000) + 1) * 3600000);
          const di = dayKeys.indexOf(SV.fmt.ymd(new Date(a)));
          if (di >= 0) grid[di][SV.fmt.hourOf(new Date(a))] += (hourEnd - a) / 3600000;
          a = hourEnd;
        }
      }
      const total = grid.reduce((x, r) => x + r.reduce((y, v) => y + v, 0), 0);
      this.heat = { days: dayKeys, hours: [...Array(24).keys()], values: grid, total };
    },
    drawHeat() {
      const el = this.$refs.heat;
      if (!el || !this.heat) return;
      if (this._heat) { this._heat.update(); return; }
      const self = this;
      this._heat = SV.charts.chart(el, () => {
        const h = self.heat;
        const C = SV.charts;
        const data = [];
        let max = 0.5;
        h.values.forEach((row, di) => row.forEach((v, hi) => { data.push([hi, di, +v.toFixed(2)]); max = Math.max(max, v); }));
        const dayLabel = (iso) => `${SV.fmt.weekday(iso)} ${SV.fmt.toDate(iso).getUTCDate()}`;
        return Object.assign(C.base(), {
          grid: { left: 4, right: 8, top: 4, bottom: 4, containLabel: true },
          tooltip: Object.assign(C.base().tooltip, {
            formatter(p) {
              const [hi, di, v] = p.value;
              return `<div style="font-weight:600">${SV.fmt.dateLong(h.days[di], false)}, ${String(hi).padStart(2, "0")}:00–${String(hi + 1).padStart(2, "0")}:00</div>
                      <div style="margin-top:2px">${v > 0 ? `${SV.fmt.num(v, 1)} маш.-ч работы` : "техника не работала"}</div>`;
            },
          }),
          xAxis: { type: "category", data: h.hours.map((x) => String(x).padStart(2, "0")), splitArea: { show: false }, axisLine: { show: false }, axisTick: { show: false },
                   axisLabel: { color: C.t("fg-3"), fontSize: 10, interval: 2 } },
          yAxis: { type: "category", data: h.days.map(dayLabel), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: C.t("fg-3"), fontSize: 10 } },
          visualMap: { show: false, min: 0, max, inRange: { color: [C.t("seq-0"), C.t("seq-1"), C.t("seq-2"), C.t("seq-3"), C.t("seq-4"), C.t("seq-5")] } },
          series: [{ type: "heatmap", data, itemStyle: { borderColor: C.t("panel"), borderWidth: 2, borderRadius: 3 }, emphasis: { itemStyle: { borderColor: C.t("fg"), borderWidth: 1 } } }],
        });
      });
    },
    /* Часы единицы за «сегодня» аналитики — из журнала интервалов. */
    workedToday(u) {
      if (!this.intervals) return null;
      const day = this.todayIso;
      return this.intervals.filter((iv) => iv.unit_id === u.id && SV.fmt.ymd(iv.start) === day).reduce((a, iv) => a + (iv.hours || 0), 0);
    },
    get unitCounts() {
      const c = { all: 0, active: 0, idle: 0, parked: 0, departed: 0 };
      for (const u of this.units || []) { c.all++; c[u.status] = (c[u.status] || 0) + 1; }
      return c;
    },
    get shownUnitsAll() {
      const order = { active: 0, idle: 1, parked: 2, departed: 3 };
      return (this.units || []).slice()
        .sort((a, b) => (order[a.status] ?? 9) - (order[b.status] ?? 9) || (b.worked_hours || 0) - (a.worked_hours || 0) || String(this.unitLabel(a)).localeCompare(String(this.unitLabel(b)), "ru"));
    },
    get shownUnits() { return this.shownUnitsAll.filter((u) => this.unitFilter === "all" || u.status === this.unitFilter); },
    unitLabel(u) { return u.label || `${u.name || SV.classNames[u.cls] || u.cls} ${u.uid || ""}`.trim(); },
    unitMeta(u) { return SV.META.unit[u.status] || SV.META.unit.idle; },
    unitStatusLine(u) {
      const since = (t) => SV.fmt.dur((this.nowMs - SV.fmt.toDate(t).getTime()) / 3600000);
      if (u.status === "active") return u.last_moved ? `двигалась ${this.relNow(u.last_moved)}` : "работает";
      if (u.status === "idle") return u.last_moved ? `стоит ${since(u.last_moved)}` : "стоит с появления";
      if (u.status === "parked") return u.last_moved ? `не двигалась ${since(u.last_moved)}` : "на стоянке";
      return u.last_seen ? `последний раз ${this.relNow(u.last_seen)}` : "уехала";
    },
    unitCams(u) { return (u.cameras || []).map((c) => (typeof c === "object" ? c.name : this.camName(c))); },
    /* Кроп последнего снимка единицы средствами CSS: bbox в пикселях исходного кадра,
       размер кадра — у камеры (image_w/h), картинка — превью (масштаб не важен). */
    cropStyle(u, cw = 72, ch = 52) {
      const f = this.unitDets[u.id];
      if (!f || !f.bbox) return "";
      const cam = this.cam(f.camera_id) || {};
      const W = cam.image_w || 1280, H = cam.image_h || 720;
      let [x, y, w, h] = f.bbox.map(Number);
      const px = w * 0.2, py = h * 0.2;
      x -= px; y -= py; w += 2 * px; h += 2 * py;
      const ar = cw / ch;
      if (w / h > ar) { const nh = w / ar; y -= (nh - h) / 2; h = nh; } else { const nw = h * ar; x -= (nw - w) / 2; w = nw; }
      const s = cw / w;
      return `background-image:url('${String(f.url).replace(/'/g, "%27")}');background-size:${W * s}px ${H * s}px;background-position:${-x * s}px ${-y * s}px`;
    },
    hasCrop(u) { return !!(this.unitDets[u.id] && this.unitDets[u.id].bbox); },
    openUnitFrame(u) {
      const f = this.unitDets[u.id];
      if (f) window.dispatchEvent(new CustomEvent("sv:frame", { detail: { id: f.frame_id, ids: [f.frame_id], title: this.unitLabel(u) } }));
    },

    /* «Временная полоска»: план часов, отработано, ожидаемое к «сегодня» и давность последней работы. */
    hoursRow(r) {
      const planned = Number(r.planned_hours) || 0;
      const worked = Number(r.worked_hours) || 0;
      const remaining = r.remaining_hours != null ? Number(r.remaining_hours) : Math.max(0, planned - worked);
      const expected = this.expectedHours(r);
      const sinceH = r.last_worked_at ? (this.nowMs - SV.fmt.toDate(r.last_worked_at).getTime()) / 3600000 : null;
      const idleAlert = Number(Alpine.store("app").settings?.thresholds?.analytics?.idle_alert_h) || 4;
      const onSite = (r.active || 0) + (r.idle || 0) + (r.parked || 0);
      const since = this.expectedSince(r);
      // Коротко: колонка статуса узкая. Дата — если ожидание считается с начала съёмки, а не этапа.
      const vsExpected = (x) => `${SV.fmt.pct(x)} от ожидаемого ${since ? "с " + since : "к сегодня"}`;
      let tone = "good", label = "В норме", note = "";
      if (planned <= 0) {
        tone = "neutral"; label = "Не по плану";
        note = onSite ? "на площадке, но в плане этапа не нужна" : "";
      } else if (r.detectable === false && worked <= 0) {
        // Детектор этот тип не различает (у YOLO нет асфальтоукладчика, гусеничного крана):
        // «нет на площадке» было бы неправдой — часы вносятся вручную.
        tone = "neutral"; label = "Не различается детектором"; note = "моточасы — ручной поправкой (±)";
      } else if (worked >= planned) {
        tone = "warn"; label = "Часы выработаны"; note = "проверьте, сменился ли этап";
      } else if (!onSite && worked === 0) {
        tone = "bad"; label = "Нет на площадке"; note = "по плану нужна, на снимках не видна";
      } else if (sinceH != null && sinceH > idleAlert && !(r.active > 0)) {
        tone = "bad"; label = "Простой"; note = `полоска не уменьшается ${SV.fmt.dur(sinceH)}`;
      } else if (expected > 0 && worked / expected < 0.6) {
        tone = "bad"; label = "Сильно отстаёт"; note = vsExpected(worked / expected);
      } else if (expected > 0 && worked / expected < 0.9) {
        tone = "warn"; label = "Отстаёт"; note = vsExpected(worked / expected);
      }
      if (!note && sinceH != null) note = sinceH < 0.75 ? "работает сейчас" : `работала ${this.relNow(r.last_worked_at)}`;
      const max = Math.max(planned, worked, 1);
      return {
        planned, worked, remaining, expected, since, tone, label, note,
        // Не нужна по плану: просто отработанные часы, без «перерасхода».
        fill: planned <= 0 ? (worked > 0 ? 100 : 0) : Math.min(100, (worked / max) * 100),
        over: planned > 0 && worked > planned ? ((worked - planned) / max) * 100 : 0,
        mark: expected > 0 && !(r.detectable === false && worked <= 0) ? Math.min(100, (expected / max) * 100) : null,
      };
    },
    /* Сколько часов должно быть отработано к «сегодня» — от НАЧАЛА НАБЛЮДЕНИЯ (требование 5):
       бэкенд считает по рабочим сменам с первого кадра площадки или с начала этапа, что позже
       (expected_hours). Запасной расчёт для старого отчёта — доля срока этапа с того же момента. */
    expectedHours(r) {
      if (r.expected_hours != null) return Number(r.expected_hours) || 0;
      const ids = r.stage_ids || (r.stage_id != null ? [r.stage_id] : []);
      const sts = ids.map((id) => this.stages.find((s) => s.id === id)).filter((s) => s && s.planned_start && s.planned_end);
      if (!sts.length) return 0;
      const seen = this.firstFrameAt ? SV.fmt.toDate(this.firstFrameAt).getTime() : null;
      let f = 0;
      for (const st of sts) {
        const a0 = SV.fmt.toDate(st.planned_start).getTime(), b = SV.fmt.toDate(st.planned_end).getTime() + DAY;
        const a = seen != null ? Math.max(a0, seen) : a0;
        if (b > a) f = Math.max(f, Math.max(0, Math.min(1, (this.nowMs - a) / (b - a0))));
      }
      return (Number(r.planned_hours) || 0) * f;
    },
    /* С какой даты считается ожидание, если позже начала этапа по плану (камеры начали снимать посреди этапа). */
    expectedSince(r) {
      if (!r.expected_from) return "";
      const ids = r.stage_ids || (r.stage_id != null ? [r.stage_id] : []);
      const starts = ids.map((id) => (this.stages.find((s) => s.id === id) || {}).planned_start).filter(Boolean);
      const from = SV.fmt.ymd(r.expected_from);
      if (starts.length && starts.every((d) => from <= SV.fmt.ymd(d))) return "";
      return SV.fmt.date(r.expected_from);
    },
    /* Выбор этапа для полосок: «сейчас» (сводка бэкенда по идущим этапам) или любой этап из balances. */
    get hoursStages() {
      const ids = [...new Set((this.balances || []).map((b) => b.stage_id).filter((x) => x != null))].sort((a, b) => a - b);
      return ids.map((id) => ({ id, name: (this.stages.find((s) => s.id === id) || {}).name || `Этап ${id}` }));
    },
    get hoursStage() {
      if (this.hoursStageSel !== "now") return this.stages.find((s) => s.id === Number(this.hoursStageSel)) || null;
      const eq = (this.ov && this.ov.equipment) || [];
      const ids = [...new Set(eq.flatMap((r) => r.stage_ids || []))];
      return ids.length === 1 ? this.stages.find((s) => s.id === ids[0]) : ids.length ? { name: ids.map((id) => (this.stages.find((s) => s.id === id) || {}).name).join(" + ") } : this.currentStage;
    },
    get hoursRows() {
      const order = Object.keys(SV.palette.CLASS_COLORS);
      const eq = (this.ov && this.ov.equipment) || [];
      let rows = eq;
      if (this.hoursStageSel !== "now") {
        const sid = Number(this.hoursStageSel);
        const byCls = Object.fromEntries(eq.map((r) => [r.cls, r]));
        rows = (this.balances || []).filter((b) => b.stage_id === sid).map((b) => Object.assign({}, byCls[b.cls] || { units: 0, active: 0, idle: 0, parked: 0 }, b, { stage_ids: [sid] }));
      }
      return rows.slice()
        .filter((r) => (r.planned_hours || 0) > 0 || (r.worked_hours || 0) > 0 || (r.units || 0) > 0)
        .sort((a, b) => ((b.planned_hours || 0) > 0) - ((a.planned_hours || 0) > 0) || order.indexOf(a.cls) - order.indexOf(b.cls));
    },
    get hoursTotals() {
      // Типы, которые детектор не различает и по которым нет ручных часов, выработку не занижают.
      const rows = this.hoursRows.filter((r) => (r.planned_hours || 0) > 0 && !(r.detectable === false && !(r.worked_hours > 0)));
      const p = rows.reduce((a, r) => a + (Number(r.planned_hours) || 0), 0);
      const w = rows.reduce((a, r) => a + Math.min(Number(r.worked_hours) || 0, Number(r.planned_hours) || 0), 0);
      return { planned: p, worked: w, ratio: p ? w / p : 0 };
    },

    // ручная правка часов: POST /api/sites/{id}/hours {cls, hours, stage_id?, at?, note?}
    openFix(r) {
      const sid = (r && r.stage_ids && r.stage_ids[0]) || (this.hoursStageSel !== "now" ? Number(this.hoursStageSel) : this.report.current_stage) || null;
      this.fix = { cls: r ? r.cls : "", hours: 1, sign: 1, stage_id: sid, note: "", busy: false };
    },
    async saveFix() {
      const f = this.fix;
      const hours = Math.abs(Number(f.hours) || 0) * f.sign;
      if (!f.cls || !hours) return SV.toast("Укажите тип техники и ненулевое число часов", { kind: "error" });
      f.busy = true;
      try {
        await SV.api.post(`/api/sites/${siteId}/hours`, {
          cls: f.cls, hours, stage_id: f.stage_id ? Number(f.stage_id) : null, note: f.note.trim(),
          // Для архива поправка ставится на «сейчас» аналитики, иначе она уйдёт за пределы окна отчёта.
          at: this.report.now || undefined,
        });
        SV.toast(`Поправка ${hours > 0 ? "+" : "−"}${SV.fmt.hours(Math.abs(hours), 1)} · ${SV.classNames[f.cls] || f.cls} учтена. Переанализ её не сотрёт.`, { kind: "success" });
        this.fix = null;
        await Promise.all([this.load(true), this.loadCorrections(), this.loadUnits(false)]);
      } catch (e) {
        SV.toastError(e, "Поправка не сохранена");
      } finally {
        if (this.fix) this.fix.busy = false;
      }
    },
    async deleteFix(c) {
      try {
        await SV.api.del(`/api/sites/${siteId}/hours/${c.id}`);
        this.corrections = (this.corrections || []).filter((x) => x.id !== c.id);
        SV.toast("Поправка удалена", { kind: "success" });
        await Promise.all([this.load(true), this.loadUnits(false)]);
      } catch (e) {
        SV.toastError(e, "Не удалось удалить поправку");
      }
    },
    get fixClasses() {
      const s = Alpine.store("app").settings;
      return (s && s.classes) || Object.keys(SV.classNames).map((k) => ({ key: k, name: SV.classNames[k] }));
    },

    // ------------------------------------------------------------ отклонения
    async loadDevs() {
      try {
        this.devs = await SV.api.get(`/api/sites/${siteId}/deviations?status=all&limit=500`);
      } catch (e) {
        this.devs = this.devs || [];
        SV.toastError(e, "Лента отклонений не загрузилась");
      }
    },
    get devTabCounts() {
      const c = { open: 0, ack: 0, resolved: 0, all: 0 };
      for (const d of this.devs || []) { c.all++; c[d.status || "open"] = (c[d.status || "open"] || 0) + 1; }
      return c;
    },
    get shownDevs() {
      return (this.devs || [])
        .filter((d) => this.devStatus === "all" || (d.status || "open") === this.devStatus)
        .filter((d) => this.devSev[d.severity] !== false)
        .sort((a, b) => (SV.META.severity[a.severity]?.rank ?? 9) - (SV.META.severity[b.severity]?.rank ?? 9)
          || String(b.last_seen_at || "").localeCompare(String(a.last_seen_at || "")));
    },
    sevMeta(d) { return SV.META.severity[d.severity] || SV.META.severity.info; },
    devWhen(d) {
      if (!d.started_at) return "";
      const hours = ((SV.fmt.toDate(d.last_seen_at) || this.now) - SV.fmt.toDate(d.started_at)) / 3600000;
      return hours < 0.2 ? SV.fmt.dateTime(d.started_at) : `с ${SV.fmt.dateTime(d.started_at)} · ${SV.fmt.dur(hours)}`;
    },
    devCams(d) {
      if (d.camera_name) return [d.camera_name];
      const ids = [...new Set((d.frames || []).map((f) => f.camera_id))];
      return ids.map((id) => this.camName(id));
    },
    devUnits(d) {
      const byUid = {};
      for (const u of this.units || []) byUid[u.uid] = this.unitLabel(u);
      return (d.unit_ids || []).map((x) => byUid[x] || x);
    },
    devFrames(d) {
      const fr = (d.frames || []).slice(0, 4);
      if (fr.length) return fr;
      return (d.frame_ids || []).slice(0, 4).map((id) => ({ id }));
    },
    thumb(f) { return `/api/frames/${f.id}/annotated.jpg?max_w=480`; },
    openEvidence(d, id) {
      const ids = (d.frames && d.frames.length ? d.frames.map((f) => f.id) : d.frame_ids) || [id];
      window.dispatchEvent(new CustomEvent("sv:frame", { detail: { id, ids, title: d.title } }));
    },
    async setDevStatus(d, status) {
      const prev = d.status;
      d.status = status;
      try {
        const res = await SV.api.patch(`/api/deviations/${d.id}`, { status });
        if (res && res.id) Object.assign(d, res);
        if (this.ov) this.ov.deviations = (this.ov.deviations || []).map((x) => (x.id === d.id ? Object.assign({}, x, { status }) : x));
        if (status === "ack") {
          SV.toast(`Квитировано: «${d.title}»`, { kind: "success", action: { label: "Вернуть", fn: () => this.setDevStatus(d, "open") } });
        }
      } catch (e) {
        d.status = prev;
        SV.toastError(e, "Не удалось изменить статус");
      }
    },

    // ------------------------------------------------------------ камеры
    async loadCamFrames() {
      const cams = (this.ov && this.ov.cameras) || [];
      await Promise.all(cams.map(async (c) => {
        if (!c.last_frame || this.camFrames[c.id]) return;
        try {
          const f = await SV.api.get(`/api/frames/${c.last_frame.id}`);
          this.camFrames = Object.assign({}, this.camFrames, { [c.id]: f });
        } catch (e) {
          this.camFrames = Object.assign({}, this.camFrames, { [c.id]: { error: e.message } });
        }
      }));
    },
    camSvg(c) { return SV.boxesSvg(this.camFrames[c.id]); },
    camLabels(c) { return SV.boxLabels(this.camFrames[c.id], { dispW: 620 }); },
    camInfo(c) { return this.cam(c.id) || {}; },
    newCam: { open: false, name: "", kind: "upload", interval_min: 20, busy: false },
    async createCamera() {
      this.newCam.busy = true;
      try {
        const c = await SV.api.post(`/api/sites/${siteId}/cameras`, { name: this.newCam.name.trim(), kind: this.newCam.kind, interval_min: Number(this.newCam.interval_min) || 20 });
        location.href = `/cameras/${c.id}#upload`;
      } catch (e) {
        SV.toastError(e, "Камера не добавлена");
      } finally {
        this.newCam.busy = false;
      }
    },
  }));
});
