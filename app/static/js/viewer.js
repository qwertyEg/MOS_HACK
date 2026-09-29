/*
 * Общее для страниц «Камера» и «Кадр»: разбор кадра (качество, детекции,
 * чек-лист модели Б) и перевод клика по картинке в пиксели исходного кадра.
 *
 * Подмешивается в компоненты через SV.mix — он копирует геттеры как геттеры
 * (обычный spread `{...obj}` вызвал бы их один раз и заморозил значения).
 */
(function () {
  function mix(...objs) {
    const out = {};
    for (const o of objs) Object.defineProperties(out, Object.getOwnPropertyDescriptors(o));
    return out;
  }

  /* Клик по картинке → координаты в пикселях исходного кадра (без учёта масштаба вёрстки). */
  function imagePoint(ev, imgEl, frame) {
    const r = imgEl.getBoundingClientRect();
    const x = ((ev.clientX - r.left) / r.width) * (frame.width || imgEl.naturalWidth);
    const y = ((ev.clientY - r.top) / r.height) * (frame.height || imgEl.naturalHeight);
    return [Math.round(x), Math.round(y)];
  }

  function frameMixin() {
    return {
      highlight: null,
      dispW: 760,

      get quality() { return (this.frame && this.frame.quality) || {}; },
      /* Статус кадра в конвейере: pending / processing / postponed / error / done. */
      get inQueue() { return !!this.frame && ["pending", "processing"].includes(this.frame.status); },
      get qualityChips() {
        const q = this.quality;
        const f = this.frame;
        if (!f) return [];
        const chips = [];
        if (this.inQueue) chips.push({ label: "В очереди анализа", icon: "loader", tone: "info" });
        if (f.status === "postponed") chips.push({ label: "Ждёт провайдера", icon: "clock", tone: "warn" });
        if (f.status === "error") chips.push({ label: "Ошибка анализа", icon: "alert-octagon", tone: "bad" });
        chips.push(q.is_night ? { label: "Ночь", icon: "moon-star", tone: "neutral" } : { label: "День", icon: "sun", tone: "neutral" });
        const w = SV.META.weather[q.weather] || SV.META.weather.unknown;
        if (q.weather && q.weather !== "unknown") chips.push({ label: w.label, icon: w.icon, tone: q.weather === "clear" ? "neutral" : "warn" });
        if (q.quality_ok === false) chips.push({ label: "Брак кадра", icon: "alert-triangle", tone: "bad" });
        // /api/analyze отдаёт usable_for_stage; кадр из базы — stage_used (модель Б по нему спрашивалась).
        const usable = q.usable_for_stage != null ? q.usable_for_stage : null;
        if (usable === false) chips.push({ label: "Не идёт в модель Б", icon: "x", tone: "warn" });
        else if (usable === true) chips.push({ label: "Годен для этапа", icon: "check", tone: "good" });
        else if (f.stage_used) chips.push({ label: "Разобран моделью Б", icon: "check", tone: "good" });
        return chips;
      },
      get dets() { return (this.frame && this.frame.detections) || []; },
      detName(d) {
        this.unitsTick;
        const cls = SV.classNames[d.class || d.cls] || d.class || d.cls;
        const unit = SV.unitName(d.unit_id);
        // Класс этой рамки оператор задал вручную и он расходится с машиной — называем по рамке.
        if (unit && d.manual && d.manual.cls && !unit.startsWith(cls)) return cls;
        return unit || d.name || cls;
      },
      /* Подстрока рамки: класс (если в заголовке — имя единицы), зона, сдвиг с прошлого кадра. */
      detSub(d) {
        const parts = [];
        if (this.detName(d) !== this.detClass(d)) parts.push(this.detClass(d));
        const z = this.detZone(d);
        if (z) parts.push(`зона «${z}»`);
        if (d.moved_since_prev && d.displacement_px) parts.push(`сдвиг ${Math.round(d.displacement_px)} px`);
        if (!parts.length && (!d.activity || d.activity === "unknown")) parts.push(this.frame && this.frame.id ? "трек только появился" : "одиночный снимок — без трекинга");
        return parts.join(" · ");
      },
      detZone(d) {
        if (d.zone_name) return d.zone_name;
        const z = (this.zones || []).find((x) => x.id === d.zone_id);
        return z ? z.name : "";
      },
      detClass(d) { return SV.classNames[d.class || d.cls] || d.class || d.cls; },
      get detSummary() {
        const by = {};
        for (const d of this.dets) {
          const k = d.class || d.cls;
          by[k] = by[k] || { cls: k, n: 0, working: 0 };
          by[k].n++;
          if (d.activity === "working") by[k].working++;
        }
        return Object.values(by);
      },
      get checklist() { return (this.frame && this.frame.checklist) || null; },
      get checkGroups() {
        const c = this.checklist;
        if (!c) return [];
        const q = c.questions || {};
        const sc = c.scores || {};
        const items = c.items || Object.entries(c.answers || {}).map(([key, answer]) => ({
          key, answer, question: q[key] || Alpine.store("app").question(key), score: typeof sc[key] === "number" ? sc[key] : null,
        }));
        const g = { yes: [], unsure: [], no: [] };
        for (const it of items) (g[it.answer === "not_visible" ? "unsure" : it.answer] || g.unsure).push(it);
        return [
          { key: "yes", label: "Да", items: g.yes, tone: "good" },
          { key: "unsure", label: "Не уверен", items: g.unsure, tone: "warn" },
          { key: "no", label: "Нет", items: g.no, tone: "neutral" },
        ].filter((x) => x.items.length);
      },
      get unsureRatio() {
        const c = this.checklist;
        if (!c) return null;
        if (c.unsure_ratio != null) return c.unsure_ratio;
        const items = c.items || [];
        return items.length ? items.filter((i) => i.answer !== "yes" && i.answer !== "no").length / items.length : null;
      },
      get unsureThreshold() { return Number(Alpine.store("app").settings?.thresholds?.stage?.unsure_review_ratio) || 0.5; },
      get likelihood() {
        const c = this.checklist;
        const src = (c && c.stage_likelihood) || (this.frame && this.frame.stage_likelihood) || {};
        return Object.entries(src).map(([id, p]) => ({ id: Number(id), p: Number(p), name: this.stageNameById(Number(id)) }))
          .sort((a, b) => b.p - a.p).slice(0, 4);
      },
      stageNameById(id) {
        const cat = Alpine.store("app").catalog;
        const st = cat && (cat.stages || []).find((s) => s.id === id);
        if (st) return st.name;
        const names = { 1: "Подготовка территории", 2: "Ограждение котлована, шпунт, сваи", 3: "Земляные работы, котлован", 4: "Монолит подземной части",
          5: "Монолит надземной части", 6: "Кровля", 7: "Фасад и остекление", 8: "Наружные сети и благоустройство" };
        return names[id] || `Этап ${id}`;
      },
      get noChecklistReason() {
        const q = this.quality;
        const f = this.frame || {};
        if (this.inQueue) return "Кадр ещё в очереди анализа — разбор появится через несколько секунд.";
        if (f.status === "postponed") return f.note || "Провайдер модели Б не готов — кадр ждёт, анализ продолжится сам.";
        if (f.status === "error") return f.note || "Анализ кадра завершился ошибкой.";
        if (f.errors && f.errors.length) return f.errors.find((e) => /модель Б/.test(e)) || f.errors[0];
        if (q.reject_reason) return q.reject_reason;
        if (q.is_night) return "Ночной кадр — для определения этапа не используется.";
        const every = Number(Alpine.store("app").settings?.thresholds?.pipeline?.stage_every_h) || 1;
        return `Модель Б спрашивается не чаще раза в ${every === 1 ? "час" : SV.fmt.hours(every, 1)} съёмки на камеру — этот кадр пропущен, соседний разобран.`;
      },
      measure(el) {
        if (!el) return;
        const set = () => { this.dispW = el.clientWidth || 760; };
        set();
        new ResizeObserver(set).observe(el);
      },
    };
  }

  window.SV = window.SV || {};
  Object.assign(window.SV, { mix, imagePoint, frameMixin });
})();
