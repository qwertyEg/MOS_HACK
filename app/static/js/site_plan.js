/*
 * Редактор плана на вкладке «План»: этапы с кодами работ xlsx и датами,
 * парк техники, плановые моточасы (авто/вручную), ручная отметка готовности,
 * импорт графика xlsx/csv и демо-план.
 *
 * План и парк сохраняются одной кнопкой (PUT /plan, PUT /fleet): так нельзя
 * получить полусохранённое состояние «даты новые, часы старые». Ручная
 * отметка этапа — отдельное действие (PATCH /stages/{id}) и применяется сразу:
 * это факт с площадки, а не черновик плана.
 */
document.addEventListener("alpine:init", () => {
  // Умолчания формулы плановых часов (ARCHITECTURE §5.6); реальные — смена объекта и
  // пороги equipment.utilization / workdays из /api/settings.
  const SHIFT_H = 10, UTIL = 0.7;

  const clone = (x) => JSON.parse(JSON.stringify(x));

  Alpine.data("sitePlan", (siteId) => ({
    plan: null,
    fleet: {},
    origPlan: "",
    origFleet: "",
    loadError: null,
    saving: false,
    warnings: [],
    importInfo: "",
    importing: false,
    dragOver: false,
    expanded: {},
    mark: null,
    markSaving: false,
    newCode: {},

    async init() {
      Alpine.store("app").loadCatalog();
      await this.loadPlan();
    },

    async loadPlan() {
      this.loadError = null;
      try {
        const [plan, fleet] = await Promise.all([
          SV.api.get(`/api/sites/${siteId}/plan`),
          SV.api.get(`/api/sites/${siteId}/fleet`).catch(() => []),
        ]);
        this.applyPlan(Array.isArray(plan) ? plan : plan.plan || [], true);
        this.fleet = {};
        for (const r of fleet || []) this.fleet[r.cls] = Number(r.count) || 0;
        this.origFleet = JSON.stringify(this.fleet);
      } catch (e) {
        this.loadError = e.message;
      }
    },
    applyPlan(rows, asOrig) {
      const norm = rows.map((r) => ({
        stage_id: r.stage_id, name: r.name || "", work_codes: [...(r.work_codes || [])],
        planned_start: r.planned_start || "", planned_end: r.planned_end || "",
        equipment: Object.assign({}, r.equipment || {}), planned_hours: Object.assign({}, r.planned_hours || {}),
        hours_manual: !!r.hours_manual, source: r.source || "",
      })).sort((a, b) => a.stage_id - b.stage_id);
      this.plan = norm;
      if (asOrig) this.origPlan = JSON.stringify(norm);
    },

    get planSource() {
      const rows = this.origPlan ? JSON.parse(this.origPlan) : [];
      return rows.length ? rows[0].source || "" : "";
    },
    /* Шаблон графика для импорта: колонки, которые понимает core.plan.importer. */
    downloadTemplate() {
      const rows = [
        ["Код", "Наименование работ", "Начало", "Окончание", "Техника"],
        ["10.1.", "Подготовка территории", "2026-04-01", "2026-04-30", "бульдозер 1, самосвал 2"],
        ["12.3.1.", "Устройство котлована", "2026-05-15", "2026-06-30", "экскаватор 2, самосвал 4"],
        ["12.4.2.", "Монолит подземной части", "2026-07-01", "2026-09-15", "автобетононасос 1, автобетоносмеситель 3"],
      ];
      const csv = "\ufeff" + rows.map((r) => r.map((x) => `"${String(x).replace(/"/g, '""')}"`).join(";")).join("\r\n");
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
      a.download = "plan_template.csv";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    },

    get dirty() {
      return this.plan && (JSON.stringify(this.plan) !== this.origPlan || JSON.stringify(this.fleet) !== this.origFleet);
    },
    get classes() {
      const s = Alpine.store("app").settings;
      const list = (s && s.classes) || Object.keys(SV.classNames).map((k) => ({ key: k, name: SV.classNames[k], tz: false }));
      return list;
    },
    className(k) { return SV.classNames[k] || k; },
    get missingStages() {
      const have = new Set((this.plan || []).map((r) => r.stage_id));
      return this.stagesRef.filter((s) => !have.has(s.id));
    },
    /* Восемь макроэтапов: состояние (статус, готовность) — из overview родителя,
       справочник работ xlsx — из /api/catalog (works с кодами, подэтапы). */
    get stagesRef() { return (this.ov && this.ov.stages) || []; },
    get catalogStages() { const c = Alpine.store("app").catalog; return (c && c.stages) || []; },
    stageState(id) { return this.stagesRef.find((s) => s.id === id) || null; },
    stageName(r) { return r.name || (this.stageState(r.stage_id) || {}).name || `Этап ${r.stage_id}`; },
    workName(code) {
      const base = String(code).split("/")[0];
      for (const s of this.catalogStages) for (const w of s.works || []) if (w.code === code || w.code === base) return code.includes("/") ? code.split("/").slice(1).join("/") : w.name;
      for (const s of this.stagesRef) for (const w of s.works || []) if (w.code === code) return w.name;
      return "";
    },
    catalogWorks(id) {
      const st = this.catalogStages.find((s) => s.id === id);
      return st ? (st.works || []).filter((w) => w.code) : [];
    },
    get shiftH() { return Number(this.ov && this.ov.site && this.ov.site.shift_hours) || SHIFT_H; },
    get util() {
      const t = (Alpine.store("app").settings || {}).thresholds || {};
      return Number((t.equipment || {}).utilization) || Number((t.analytics || {}).utilization) || UTIL;
    },
    get workdaySet() {
      const t = (Alpine.store("app").settings || {}).thresholds || {};
      const wd = (t.equipment || {}).workdays;
      return new Set(Array.isArray(wd) && wd.length ? wd : [0, 1, 2, 3, 4, 5]); // 0 = пн (как weekday() в Python)
    },

    // ---------------------------------------------------------------- даты
    calDays(r) {
      const n = SV.fmt.daysBetween(r.planned_start, r.planned_end);
      return n == null ? null : n + 1;
    },
    workdays(r) {
      if (!r.planned_start || !r.planned_end) return 0;
      const a = SV.fmt.toDate(r.planned_start), b = SV.fmt.toDate(r.planned_end);
      let n = 0;
      const set = this.workdaySet;
      for (let t = a.getTime(); t <= b.getTime(); t += 86400000) if (set.has((new Date(t).getUTCDay() + 6) % 7)) n++;
      return n;
    },
    dateError(r) { return r.planned_start && r.planned_end && r.planned_start > r.planned_end; },
    addCode(r) {
      const raw = (this.newCode[r.stage_id] || "").trim();
      if (!raw) return;
      const code = /\.$/.test(raw) || raw.includes("/") ? raw : raw + ".";
      if (!r.work_codes.includes(code)) r.work_codes.push(code);
      this.newCode[r.stage_id] = "";
    },
    removeCode(r, code) { r.work_codes = r.work_codes.filter((c) => c !== code); },
    addStage(id) {
      const s = this.stageState(Number(id));
      if (!s) return;
      const works = this.catalogWorks(s.id).filter((w) => w.status !== "stage").slice(0, 6).map((w) => w.code);
      this.plan.push({ stage_id: s.id, name: s.name, work_codes: works.length ? works : (s.works || []).map((w) => w.code), planned_start: "", planned_end: "", equipment: {}, planned_hours: {}, hours_manual: false });
      this.plan.sort((a, b) => a.stage_id - b.stage_id);
    },
    removeStage(r) {
      this.plan = this.plan.filter((x) => x !== r);
    },

    // ---------------------------------------------------------------- моточасы
    autoHours(r, cls) {
      const n = r.equipment[cls] != null ? Number(r.equipment[cls]) : Number(this.fleet[cls] || 0);
      return Math.round(n * this.workdays(r) * this.shiftH * this.util);
    },
    hoursClasses(r) {
      return [...new Set([...Object.keys(r.equipment || {}), ...Object.keys(r.planned_hours || {})])];
    },
    hoursOf(r, cls) {
      return r.hours_manual ? Number(r.planned_hours[cls] || 0) : (r.planned_hours[cls] != null && !this.touched(r) ? Number(r.planned_hours[cls]) : this.autoHours(r, cls));
    },
    /* Пока пользователь не менял строку, показываем часы с сервера (там точная формула с календарём). */
    touched(r) {
      const o = JSON.parse(this.origPlan || "[]").find((x) => x.stage_id === r.stage_id);
      return !o || JSON.stringify(o.equipment) !== JSON.stringify(r.equipment) || o.planned_start !== r.planned_start || o.planned_end !== r.planned_end;
    },
    totalHours(r) { return this.hoursClasses(r).reduce((a, c) => a + this.hoursOf(r, c), 0); },
    setHours(r, cls, v) {
      if (!r.hours_manual) {
        for (const c of this.hoursClasses(r)) r.planned_hours[c] = this.hoursOf(r, c);
        r.hours_manual = true;
      }
      r.planned_hours[cls] = Math.max(0, Number(v) || 0);
    },
    setCount(r, cls, v) {
      r.equipment[cls] = Math.max(0, Math.min(99, Number(v) || 0));
      if (!r.hours_manual) r.planned_hours[cls] = this.autoHours(r, cls);
    },
    toggleManual(r) {
      if (r.hours_manual) {
        r.hours_manual = false;
        for (const c of this.hoursClasses(r)) r.planned_hours[c] = this.autoHours(r, c);
      } else {
        for (const c of this.hoursClasses(r)) r.planned_hours[c] = this.hoursOf(r, c);
        r.hours_manual = true;
      }
    },
    addClass(r, cls) {
      if (!cls || r.equipment[cls] != null) return;
      r.equipment[cls] = 1;
      r.planned_hours[cls] = r.hours_manual ? 0 : this.autoHours(r, cls);
      this.expanded[r.stage_id] = true;
    },
    removeClass(r, cls) {
      delete r.equipment[cls];
      delete r.planned_hours[cls];
      r.equipment = Object.assign({}, r.equipment);
      r.planned_hours = Object.assign({}, r.planned_hours);
    },

    // ---------------------------------------------------------------- парк
    get fleetRows() {
      const used = new Set(Object.keys(this.fleet));
      for (const r of this.plan || []) for (const c of Object.keys(r.equipment || {})) used.add(c);
      const tz = this.classes.filter((c) => c.tz).map((c) => c.key);
      for (const k of tz) used.add(k);
      const order = Object.keys(SV.palette.CLASS_COLORS);
      return [...used].sort((a, b) => order.indexOf(a) - order.indexOf(b));
    },
    setFleet(cls, n) { this.fleet = Object.assign({}, this.fleet, { [cls]: Math.max(0, Math.min(99, Number(n) || 0)) }); },
    addFleetClass(cls) { if (cls && this.fleet[cls] == null) this.setFleet(cls, 1); },

    // ---------------------------------------------------------------- сохранение
    async save() {
      const bad = this.plan.find((r) => this.dateError(r));
      if (bad) return SV.toast(`«${this.stageName(bad)}»: начало позже окончания`, { kind: "error" });
      this.saving = true;
      try {
        const body = this.plan.map(({ source, ...r }) => Object.assign({}, r, {
          planned_start: r.planned_start || null, planned_end: r.planned_end || null,
          planned_hours: Object.fromEntries(this.hoursClasses(r).map((c) => [c, this.hoursOf(r, c)])),
        }));
        // Сначала парк: PUT /fleet пересчитывает плановые часы, PUT /plan затем отдаёт итог.
        if (JSON.stringify(this.fleet) !== this.origFleet) {
          await SV.api.put(`/api/sites/${siteId}/fleet`, Object.entries(this.fleet).map(([cls, count]) => ({ cls, count })));
          this.origFleet = JSON.stringify(this.fleet);
        }
        const saved = await SV.api.put(`/api/sites/${siteId}/plan`, body);
        this.applyPlan(Array.isArray(saved) ? saved : body, true);
        this.warnings = [];
        this.importInfo = "";
        SV.toast("План сохранён. Отклонения и прогноз пересчитаются по новым датам.", { kind: "success" });
        if (this.load) this.load(true); // обновить сводку родителя
      } catch (e) {
        SV.toastError(e, "План не сохранён");
      } finally {
        this.saving = false;
      }
    },
    revert() {
      this.applyPlan(JSON.parse(this.origPlan), false);
      this.fleet = JSON.parse(this.origFleet || "{}");
      this.warnings = [];
      this.importInfo = "";
    },
    async importFile(file) {
      if (!file) return;
      if (!/\.(xlsx|xls|csv)$/i.test(file.name)) return SV.toast("Нужен график в .xlsx или .csv", { kind: "error" });
      this.importing = true;
      try {
        // apply=0 — только разбор: план попадает в редактор, в базу — по кнопке «Сохранить план».
        const fd = new FormData();
        fd.append("file", file);
        fd.append("apply", "0");
        const res = await SV.api.post(`/api/sites/${siteId}/plan/import`, fd);
        const rows = Array.isArray(res) ? res : res.plan || [];
        this.applyPlan(rows, false);
        this.warnings = (res && res.warnings) || [];
        this.importInfo = `Разобран «${file.name}»: ${rows.length} ${SV.fmt.plural(rows.length, "этап", "этапа", "этапов")}. Проверьте даты и технику и сохраните план.`;
      } catch (e) {
        SV.toastError(e, "График не разобран");
      } finally {
        this.importing = false;
      }
    },
    async demo() {
      this.importing = true;
      try {
        // Демо-план бэкенд сразу сохраняет (source=demo) и пересчитывает сводку.
        const res = await SV.api.post(`/api/sites/${siteId}/plan/demo`, {});
        const rows = Array.isArray(res) ? res : res.plan || [];
        this.applyPlan(rows, true);
        this.warnings = (res && res.warnings) || [];
        this.importInfo = "Демо-план применён: этапы разложены по типовым срокам от даты первого кадра. Это не график заказчика — поправьте даты и сохраните.";
        if (this.load) this.load(true);
      } catch (e) {
        SV.toastError(e, "Демо-план не построен");
      } finally {
        this.importing = false;
      }
    },
    onDrop(e) {
      this.dragOver = false;
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) this.importFile(f);
    },

    // ---------------------------------------------------------------- ручная отметка
    openMark(stageId) {
      const s = this.stageState(stageId) || { status: "not_started", progress: 0 };
      this.mark = {
        stage_id: stageId, status: s.status || "not_started", progress: Math.round((s.progress || 0) * 100),
        actual_start: SV.fmt.inputDate(s.actual_start), actual_end: SV.fmt.inputDate(s.actual_end), note: s.note || "", manual: !!s.manual,
      };
    },
    markStatus(st) {
      this.mark.status = st;
      const today = this.todayIso || SV.fmt.ymd(new Date());
      if (st === "done") { this.mark.progress = 100; if (!this.mark.actual_end) this.mark.actual_end = today; }
      if (st === "not_started") { this.mark.progress = 0; this.mark.actual_start = ""; this.mark.actual_end = ""; }
      if (st === "active" && !this.mark.actual_start) this.mark.actual_start = today;
    },
    async saveMark(auto = false) {
      const m = this.mark;
      this.markSaving = true;
      try {
        const body = auto ? { manual: false } : {
          status: m.status, progress: Math.max(0, Math.min(100, Number(m.progress) || 0)) / 100,
          actual_start: m.actual_start || null, actual_end: m.status === "done" ? (m.actual_end || null) : null, note: m.note || "",
        };
        const res = await SV.api.patch(`/api/sites/${siteId}/stages/${m.stage_id}`, body);
        const st = this.stageState(m.stage_id);
        if (st) Object.assign(st, auto ? { manual: false } : body, res && typeof res === "object" ? res : {}, auto ? {} : { manual: true });
        SV.toast(auto ? "Этап снова определяется по снимкам" : "Отметка сохранена — модель её не перезапишет", { kind: "success" });
        this.mark = null;
        if (this.load) this.load(true);
      } catch (e) {
        SV.toastError(e, "Отметка не сохранена");
      } finally {
        this.markSaving = false;
      }
    },
  }));
});
