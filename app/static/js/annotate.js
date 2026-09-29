/*
 * Режим «Разметка» на странице кадра и камеры (требование 3: автоматически, но с ручной правкой).
 *
 * Клик по рамке → всплывающее меню: сменить класс (этой рамки или всей машины),
 * «не техника» (на этом кадре или всё это место камеры), «разделить машину с этого
 * кадра». Инструмент «+ Рамка» — протянуть прямоугольник мышью и выбрать класс.
 * «Кадр верен» — разметка кадра проверена (кадр пойдёт в датасет как проверенный).
 * Каждое действие — одна «пачка» правок на сервере: тост с «Отменить», и «Отменить
 * последнее» в панели. Кадр обновляется сразу (ответ API), техника объекта
 * перепрогоняется в фоне — пока идёт перепрогон, в панели виден индикатор.
 *
 * API (app/routers/api_annotations.py): PATCH /api/detections/{id} {cls, scope} |
 * {deleted, scope}; POST /api/frames/{id}/detections {cls, bbox}; POST /api/frames/{id}/verify;
 * POST /api/units/{id}/split {frame_id}; DELETE /api/annotations/batches/{batch}.
 *
 * Подмешивается рядом с SV.frameMixin(): ждёт от компонента frame, dets, detName(),
 * highlight. Координаты — в пикселях исходного кадра, как у рамок детектора.
 */
(function () {
  const MIN_SIDE = 8;

  function annotateMixin() {
    return {
      ann: { on: false, tool: "select", sel: null, draw: null, pending: null, cls: "", scope: "unit", busy: false,
             pos: null, replay: null, history: [], menu: null },

      // ------------------------------------------------------------ режим
      /* Клавиша E — войти/выйти из разметки; вызывается из init() страницы. */
      annInit() {
        window.addEventListener("keydown", (e) => {
          if (SV.isTyping(e) || e.metaKey || e.ctrlKey || e.altKey || this.ann.on) return;
          if ([...document.querySelectorAll("[aria-modal=true]")].some((el) => el.getClientRects().length)) return;
          const k = e.key.toLowerCase();
          if ((k === "e" || k === "у") && this.frame && this.frame.id) { e.preventDefault(); this.annToggle(true); }
        });
      },
      annToggle(force) {
        const on = force === undefined ? !this.ann.on : !!force;
        if (on && !(this.frame && this.frame.id)) return;
        this.ann.on = on;
        this.ann.tool = "select";
        this.annCancel();
        if (on) {
          this.layers && (this.layers.boxes = true);
          if (!this._annKeys) {
            this._annKeys = (e) => this.annKey(e);
            window.addEventListener("keydown", this._annKeys);
            const re = () => { if (this.ann.sel !== null || this.ann.pending) this.annPlace(); };
            window.addEventListener("resize", re);
            window.addEventListener("scroll", re, { passive: true });
          }
          Alpine.store("app").loadSettings && !Alpine.store("app").settings && Alpine.store("app").loadSettings();
        }
      },
      annTool(t) { this.ann.tool = t; this.annCancel(); },
      annCancel() { this.ann.sel = null; this.ann.pending = null; this.ann.draw = null; this.ann.pos = null; this.ann.menu = null; this.highlight = null; },
      annKey(e) {
        if (!this.ann.on || SV.isTyping(e) || e.metaKey || e.ctrlKey || e.altKey) return;
        const k = e.key.toLowerCase();
        if (e.key === "Escape") { if (this.ann.sel !== null || this.ann.pending) this.annCancel(); else this.annToggle(false); e.preventDefault(); }
        else if (k === "n" || k === "т") this.annTool(this.ann.tool === "draw" ? "select" : "draw");
        else if ((e.key === "Delete" || k === "d" || k === "в") && this.annSel) { e.preventDefault(); this.annDelete("frame"); }
        else if (k === "v" || k === "м") this.annVerify(!this.annVerified);
      },
      get annVerified() { return !!(this.frame && this.frame.annotations && this.frame.annotations.verified); },
      get annCount() { return (this.frame && this.frame.annotations && this.frame.annotations.count) || 0; },

      /* Классы для выбора: сначала восемь обязательных по ТЗ (московская стройка: КАМАЗ,
         автобетоносмеситель, «Ивановец», манипулятор…), потом остальные по алфавиту. */
      get annClasses() {
        const s = Alpine.store("app").settings;
        const list = (s && s.classes && s.classes.length) ? s.classes
          : Object.keys(SV.classNames).map((k) => ({ key: k, name: SV.classNames[k], tz: false }));
        return list.slice().sort((a, b) => (b.tz ? 1 : 0) - (a.tz ? 1 : 0) || String(a.name).localeCompare(String(b.name), "ru"));
      },

      // ------------------------------------------------------------ геометрия
      annRectStyle(b) {
        const f = this.frame || {};
        const W = f.width || 1, H = f.height || 1;
        return `left:${(b[0] / W) * 100}%;top:${(b[1] / H) * 100}%;width:${(b[2] / W) * 100}%;height:${(b[3] / H) * 100}%`;
      },
      annBoxStyle(d) { return this.annRectStyle((d.bbox || [0, 0, 0, 0]).map(Number)); },
      get annDraft() {
        const d = this.ann.draw;
        if (!d) return null;
        return [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0)];
      },
      _annPoint(ev, el) {
        const r = el.getBoundingClientRect();
        const f = this.frame;
        const x = Math.max(0, Math.min(f.width, ((ev.clientX - r.left) / r.width) * f.width));
        const y = Math.max(0, Math.min(f.height, ((ev.clientY - r.top) / r.height) * f.height));
        return [Math.round(x), Math.round(y)];
      },
      annDown(ev) {
        this._annLayer = ev.currentTarget;
        if (this.ann.tool !== "draw") { this.annCancel(); return; }
        const [x, y] = this._annPoint(ev, ev.currentTarget);
        this.ann.sel = null; this.ann.pending = null; this.ann.pos = null;
        this.ann.draw = { x0: x, y0: y, x1: x, y1: y };
        try { ev.currentTarget.setPointerCapture(ev.pointerId); } catch (e) { /* старый браузер */ }
      },
      annMove(ev) {
        if (!this.ann.draw) return;
        const [x, y] = this._annPoint(ev, ev.currentTarget);
        this.ann.draw.x1 = x; this.ann.draw.y1 = y;
      },
      annUp() {
        const b = this.annDraft;
        this.ann.draw = null;
        if (!b) return;
        if (b[2] < MIN_SIDE || b[3] < MIN_SIDE) return SV.toast("Рамка слишком мала — протяните мышью прямоугольник вокруг машины", { kind: "info" });
        this.ann.pending = { bbox: b };
        this.ann.cls = this.ann.cls || (this.annClasses[0] || {}).key || "excavator";
        this.annPlace();
      },
      annSelect(i) {
        if (!this.ann.on) this.annToggle(true);
        this.ann.pending = null;
        this.ann.sel = i;
        this.highlight = i;
        const d = this.dets[i];
        this.ann.cls = d ? (d.class || d.cls) : "";
        this.ann.scope = d && d.unit_id ? "unit" : "box";
        this.ann.menu = null;
        this.$nextTick(() => this.annPlace());
      },
      get annSel() { return this.ann.sel !== null ? this.dets[this.ann.sel] || null : null; },
      /* Меню — position:fixed рядом с рамкой: во вьюере (overflow:hidden) его бы обрезало. */
      annPlace() {
        const box = this.ann.pending ? this.ann.pending.bbox : this.annSel ? this.annSel.bbox.map(Number) : null;
        const el = this._annLayer || document.querySelector(".ann-layer");
        if (!box || !el || !this.frame) { this.ann.pos = null; return; }
        const r = el.getBoundingClientRect();
        const f = this.frame;
        const bx = r.left + (box[0] / f.width) * r.width, by = r.top + (box[1] / f.height) * r.height;
        const bw = (box[2] / f.width) * r.width, bh = (box[3] / f.height) * r.height;
        const pw = 300, ph = this.ann.pending ? 190 : 360;
        const left = Math.min(Math.max(8, bx + bw + 10 + pw < innerWidth ? bx + bw + 10 : bx - pw - 10), innerWidth - pw - 8);
        let top = by;
        if (top + ph > innerHeight - 8) top = Math.max(8, innerHeight - ph - 8);
        this.ann.pos = { left: Math.max(8, left), top: Math.max(8, top) };
      },
      get annPopStyle() {
        const p = this.ann.pos;
        return p ? `left:${p.left}px;top:${p.top}px` : "";
      },

      // ------------------------------------------------------------ подписи
      annOrigName(d) {
        const m = d && d.manual;
        if (!m) return "";
        if (m.added) return "дорисована вручную";
        const was = m.orig_cls && m.orig_cls !== (d.class || d.cls) ? `модель: ${SV.classNames[m.orig_cls] || m.orig_cls}` : "";
        return ["исправлена вручную", was].filter(Boolean).join(" · ");
      },
      get annUnitName() {
        const d = this.annSel;
        return d && d.unit_id ? (SV.unitName(d.unit_id) || "эта машина") : null;
      },

      // ------------------------------------------------------------ действия
      async _annCall(fn, okText) {
        if (this.ann.busy) return null;
        this.ann.busy = true;
        try {
          const res = await fn();
          if (res && res.frame) this.annSetFrame(res.frame);
          this.annCancel();
          if (res && res.batch) {
            this.ann.history.push(res.batch);
            SV.toast(okText(res), { kind: "success", timeout: 7000, action: { label: "Отменить", fn: () => this.annUndo(res.batch) } });
          }
          this.annFollowReplay(res && res.replay);
          window.dispatchEvent(new CustomEvent("sv:annotated", { detail: res }));
          return res;
        } catch (e) {
          SV.toastError(e, "Правка не сохранена");
          return null;
        } finally {
          this.ann.busy = false;
        }
      },
      annSetFrame(f) {
        if (!f) return;
        if (this.cache && f.id != null) this.cache[f.id] = f;
        if (!this.frame || this.frame.id === f.id) this.frame = f;
      },
      annRelabel() {
        const d = this.annSel, cls = this.ann.cls;
        if (!d || !cls) return;
        const scope = this.ann.scope === "unit" && d.unit_id ? "unit" : "box";
        return this._annCall(() => SV.api.patch(`/api/detections/${d.id}`, { cls, scope, frame_id: this.frame.id, bbox: d.bbox }),
          (r) => scope === "unit" ? `Тип машины исправлен: ${SV.classNames[cls] || cls}. Рамки и моточасы пересчитаны.` : `Класс рамки: ${SV.classNames[cls] || cls}`);
      },
      annDelete(scope) {
        const d = this.annSel;
        if (!d) return;
        return this._annCall(() => SV.api.patch(`/api/detections/${d.id}`, { deleted: true, scope, frame_id: this.frame.id, bbox: d.bbox }),
          () => scope === "camera" ? "Место камеры больше не считается техникой — на всех кадрах" : "Рамка удалена: это не техника");
      },
      annAdd() {
        const p = this.ann.pending, cls = this.ann.cls;
        if (!p || !cls) return;
        return this._annCall(() => SV.api.post(`/api/frames/${this.frame.id}/detections`, { cls, bbox: p.bbox }),
          () => `Рамка «${SV.classNames[cls] || cls}» добавлена`);
      },
      annSplit() {
        const d = this.annSel;
        if (!d || !d.unit_row_id) return;
        return this._annCall(() => SV.api.post(`/api/units/${d.unit_row_id}/split`, { frame_id: this.frame.id, uid: d.unit_id, site_id: this.frame.site_id }),
          () => "Машина разделена: с этого кадра — отдельная машина");
      },
      annVerify(flag = true) {
        if (!this.frame || !this.frame.id) return;
        return this._annCall(() => SV.api.post(`/api/frames/${this.frame.id}/verify`, { verified: !!flag }),
          () => flag ? "Кадр отмечен как проверенный — пойдёт в датасет для дообучения" : "Отметка «проверен» снята");
      },
      async annUndo(batch) {
        batch = batch || this.ann.history[this.ann.history.length - 1];
        if (!batch) return SV.toast("Отменять нечего", { kind: "info" });
        try {
          const res = await SV.api.del(`/api/annotations/batches/${encodeURIComponent(batch)}`);
          this.ann.history = this.ann.history.filter((b) => b !== batch);
          if (res && res.frame) this.annSetFrame(res.frame);
          this.annCancel();
          SV.toast("Правка отменена", { kind: "success" });
          this.annFollowReplay(res && res.replay);
          window.dispatchEvent(new CustomEvent("sv:annotated", { detail: res }));
        } catch (e) { SV.toastError(e, "Не удалось отменить"); }
      },

      /* Перепрогон техники объекта идёт в фоне: ждём и подтягиваем кадр и подписи машин. */
      annFollowReplay(state) {
        const sid = this.frame && this.frame.site_id;
        this.ann.replay = state || null;
        if (!sid) return;
        const done = () => {
          this.ann.replay = null;
          SV.api.get(`/api/frames/${this.frame.id}`).then((f) => this.annSetFrame(f)).catch(() => {});
          SV.api.get(`/api/sites/${sid}/equipment`).then((d) => { SV.setUnits(d.units); if ("unitsTick" in this) this.unitsTick++; }).catch(() => {});
        };
        if (!state || state.state === "idle") return done();
        clearTimeout(this._annPoll);
        const tick = async () => {
          try {
            const st = await SV.api.get(`/api/sites/${sid}/annotations/status`);
            this.ann.replay = st;
            if (st.state === "idle") return done();
          } catch (e) { /* следующий опрос */ }
          this._annPoll = setTimeout(tick, 1500);
        };
        this._annPoll = setTimeout(tick, 900);
      },
      get annReplayText() {
        const r = this.ann.replay;
        if (!r || r.state === "idle") return "";
        if (r.state === "queued") return "Правки приняты — техника объекта пересчитается через секунду…";
        return r.total ? `Пересчёт техники объекта: ${r.done} из ${r.total} кадров…` : "Пересчёт техники объекта…";
      },
    };
  }

  window.SV = window.SV || {};
  window.SV.annotateMixin = annotateMixin;
})();
