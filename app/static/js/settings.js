/*
 * Настройки: режим (локально / внешний API / гибрид) с готовностью провайдеров
 * и причинами, пороги моделей и правил, список классов техники.
 *
 * GET /api/settings → {mode, model_a, model_b, thresholds: {pipeline, stage, equipment, analytics},
 *                      classes[], providers{}, presets{}}.
 * PUT /api/settings — режим/модели сразу; пороги — кнопкой, только изменённые
 * значения: {thresholds: {группа: {порог: значение}}} (бэкенд хранит отличия от умолчаний).
 */
document.addEventListener("alpine:init", () => {
  // Известные пороги: группа бэкенда, подпись, единицы, диапазон и объяснение «зачем».
  const FIELDS = [
    { g: "equipment", key: "min_conf", sec: "a", label: "Уверенность детектора", kind: "range", min: 0.05, max: 0.95, step: 0.05, fmt: "pct",
      hint: "Рамки слабее порога отбрасываются. Ниже — больше находок в темноте и дождь, но и ложных срабатываний." },
    { g: "equipment", key: "merge_radius_m", sec: "a", label: "Радиус склейки между камерами", kind: "num", min: 0.5, max: 50, step: 0.5, unit: "м",
      hint: "Машины разных камер ближе этого на плане площадки считаются одной — чтобы не было двойного счёта." },
    { g: "equipment", key: "move_px_min", sec: "a", label: "Сдвиг «работает» от", kind: "num", min: 1, max: 100, step: 1, unit: "px",
      hint: "Меньший сдвиг рамки между кадрами — дрожание камеры, а не работа машины." },
    { g: "equipment", key: "parked_after_h", sec: "a", label: "«На стоянке», если стоит дольше", kind: "num", min: 1, max: 720, step: 1, unit: "ч",
      hint: "Стоящая дольше техника не считается «лишней» на этапе, но даёт предупреждение «вся техника стоит»." },
    { g: "equipment", key: "departed_after_h", sec: "a", label: "«Уехала», если не видна", kind: "num", min: 0.5, max: 72, step: 0.5, unit: "ч",
      hint: "Кадры раз в 20–30 минут: машину, пропавшую на один кадр, не списываем сразу." },
    { g: "stage", key: "yes_thr", sec: "b", label: "Модель Б: «да» от", kind: "range", min: 0.5, max: 0.95, step: 0.01, fmt: "num2",
      hint: "Сходство признака выше порога — ответ «да». Между порогами «нет» и «да» — «не уверен», такой ответ не голосует за этап." },
    { g: "stage", key: "no_thr", sec: "b", label: "Модель Б: «нет» до", kind: "range", min: 0.05, max: 0.5, step: 0.01, fmt: "num2",
      hint: "Сходство ниже порога — ответ «нет»." },
    { g: "stage", key: "unsure_review_ratio", sec: "b", label: "Ручная проверка при доле «не уверен»", kind: "range", min: 0.2, max: 0.9, step: 0.05, fmt: "pct",
      hint: "Кадр с большей долей «не уверен» не участвует в определении этапа и попадает в «проверить вручную»." },
    { g: "pipeline", key: "stage_every_h", sec: "b", label: "Модель Б не чаще, чем раз в", kind: "num", min: 0.25, max: 24, step: 0.25, unit: "ч",
      hint: "Этап меняется за дни, а не минуты: одного годного кадра в час на камеру достаточно (и дешевле для внешнего API)." },
    { g: "pipeline", key: "stage_mask_change", sec: "b", label: "Внеочередной вызов при сдвиге маски", kind: "range", min: 0.01, max: 0.3, step: 0.01, fmt: "pct",
      hint: "Если маска фона изменилась сильнее (подняли этаж, сняли забор) — модель Б спрашивается сразу, не дожидаясь часа." },
    { g: "analytics", key: "pair_window_h", sec: "r", label: "Окно пары техники", kind: "num", min: 0.5, max: 24, step: 0.5, unit: "ч",
      hint: "«Экскаватор без самосвалов» оценивается по окну рабочего времени, а не по одному кадру." },
    { g: "analytics", key: "idle_alert_h", sec: "r", label: "Простой после", kind: "num", min: 1, max: 72, step: 1, unit: "ч",
      hint: "Этап идёт, тип техники обязателен, а моточасы не списывались столько рабочих часов — отклонение «простой»." },
    { g: "analytics", key: "on_track_days", sec: "r", label: "«В графике» при расхождении до", kind: "num", min: 0, max: 30, step: 1, unit: "дн",
      hint: "Меньшее отставание или опережение не считается отклонением от графика." },
    { g: "analytics", key: "no_progress_days", sec: "r", label: "«Часы ушли — этап стоит» через", kind: "num", min: 1, max: 30, step: 1, unit: "дн",
      hint: "Сверка моделей А и Б: техника отработала часы, а готовность этапа по снимкам столько дней не растёт." },
    { g: "analytics", key: "utilization", sec: "r", label: "Коэффициент использования", kind: "range", min: 0.3, max: 1, step: 0.05, fmt: "num2",
      hint: "Доля смены, которую машина реально работает. Входит в плановые моточасы: единиц × дней × смена × коэффициент." },
    { g: "pipeline", key: "clock", sec: "r", label: "«Сейчас» для аналитики", kind: "select",
      options: [["auto", "Авто: живая площадка — часы, архив — последний кадр"], ["wall", "Всегда часы сервера"], ["last_frame", "Всегда время последнего кадра"]],
      hint: "Для архива 2008 года «сегодня» — дата последнего кадра, иначе всё выглядит просроченным на годы." },
  ];
  const SECTIONS = [
    { key: "a", title: "Техника · модель А" },
    { key: "b", title: "Этап · модель Б" },
    { key: "r", title: "Правила отклонений" },
  ];
  const clone = (x) => JSON.parse(JSON.stringify(x || {}));

  Alpine.data("settingsPage", () => ({
    FIELDS, SECTIONS,
    th: null,
    orig: "",
    saving: false,
    classFilter: "all",

    init() {
      const sync = () => {
        const s = Alpine.store("app").settings;
        if (!s) return;
        this.th = clone(s.thresholds);
        this.orig = JSON.stringify(this.th);
      };
      sync();
      this.$watch("$store.app.settings", (v, old) => { if (!old || !this.dirty) sync(); });
      Alpine.store("app").loadHealth();
    },
    get app() { return Alpine.store("app"); },
    get settings() { return this.app.settings; },
    get dirty() { return !!this.th && JSON.stringify(this.th) !== this.orig; },
    fieldsOf(sec) { return FIELDS.filter((f) => f.sec === sec && this.th && this.th[f.g] && f.key in this.th[f.g]); },
    val(f) { return this.th[f.g][f.key]; },
    show(f) {
      const v = this.val(f);
      if (f.kind === "select") return "";
      if (f.fmt === "pct") return SV.fmt.pct(Number(v));
      if (f.fmt === "num2") return SV.fmt.num(Number(v), 2);
      const n = Number(v);
      return `${SV.fmt.num(n, Number.isInteger(n) ? 0 : n * 10 % 1 ? 2 : 1)} ${f.unit || ""}`.trim();
    },
    /* Остальные числовые пороги модели А — для прозрачности, свёрнуты. */
    get advanced() {
      if (!this.th || !this.th.equipment) return [];
      const known = new Set(FIELDS.filter((f) => f.g === "equipment").map((f) => f.key));
      return Object.keys(this.th.equipment).filter((k) => !known.has(k) && typeof this.th.equipment[k] === "number").sort();
    },
    get thError() {
      const st = this.th && this.th.stage;
      if (st && Number(st.no_thr) >= Number(st.yes_thr)) return "Порог «нет» должен быть ниже порога «да».";
      return "";
    },
    /* Только изменённые скаляры: {группа: {порог: значение}}. */
    get patch() {
      const o = JSON.parse(this.orig || "{}");
      const out = {};
      for (const [g, vals] of Object.entries(this.th || {})) {
        for (const [k, v] of Object.entries(vals || {})) {
          if (typeof v === "object") continue;
          if (JSON.stringify(v) !== JSON.stringify((o[g] || {})[k])) {
            out[g] = out[g] || {};
            out[g][k] = typeof (o[g] || {})[k] === "number" ? Number(v) : v;
          }
        }
      }
      return out;
    },
    async saveTh() {
      if (this.thError) return SV.toast(this.thError, { kind: "error" });
      this.saving = true;
      try {
        const res = await SV.api.put("/api/settings", { thresholds: this.patch });
        if (res && res.thresholds) this.app.settings = res;
        this.th = clone(res && res.thresholds ? res.thresholds : this.th);
        this.orig = JSON.stringify(this.th);
        SV.toast("Пороги сохранены. Новые кадры анализируются с ними; «Переанализировать» на объекте пересчитает старые.", { kind: "success" });
      } catch (e) {
        SV.toastError(e, "Пороги не сохранены");
      } finally {
        this.saving = false;
      }
    },
    revert() { this.th = JSON.parse(this.orig || "{}"); },
    async setModel(which, value) {
      try {
        const res = await SV.api.put("/api/settings", { [which]: value });
        if (res && typeof res === "object" && res.mode) this.app.settings = res;
        else this.app.settings[which] = value;
        SV.toast(`Сохранено: ${which === "model_a" ? "модель А" : "модель Б"} — ${(SV.META.providers[value] || {}).label || value}`, { kind: "success" });
      } catch (e) {
        SV.toastError(e, "Не сохранилось");
      }
    },
    get providers() {
      const p = (this.app.health && this.app.health.providers) || (this.settings && this.settings.providers) || {};
      return Object.keys(SV.META.providers).map((k) => Object.assign({ key: k }, SV.META.providers[k], p[k] || { ready: null, reason: "нет данных" }));
    },
    get classes() {
      const list = (this.settings && this.settings.classes) || [];
      const order = Object.keys(SV.palette.CLASS_COLORS);
      return list.filter((c) => this.classFilter === "all" || (this.classFilter === "tz" ? c.tz : c.supported))
        .sort((a, b) => (b.tz - a.tz) || order.indexOf(a.key) - order.indexOf(b.key));
    },
    get classCounts() {
      const list = (this.settings && this.settings.classes) || [];
      return { all: list.length, tz: list.filter((c) => c.tz).length, supported: list.filter((c) => c.supported).length,
        tzSupported: list.filter((c) => c.tz && c.supported).length };
    },
    get missingTz() { return ((this.settings && this.settings.classes) || []).filter((c) => c.tz && !c.supported); },
  }));
});
