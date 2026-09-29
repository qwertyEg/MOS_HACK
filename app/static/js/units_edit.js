/*
 * Правка машин на вкладке «Техника» (требование 3): склеить «разные» машины в одну,
 * разделить машину с кадра, сменить тип, «это не техника»; журнал ручных правок с
 * отменой; выгрузка размеченного датасета YOLO для дообучения детектора.
 *
 * Вложенный компонент внутри sitePage: читает его units, unitLabel(), camName(),
 * loadUnits()/load()/loadIntervals(). После правки сервер перепрогоняет технику по
 * сохранённым рамкам (без детектора); небольшой объект — сразу в запросе, большой
 * архив — в фоне: тогда ждём /api/sites/{id}/annotations/status и обновляемся.
 *
 * API: POST /api/units/merge {unit_ids, target_id, cls}; POST /api/units/{id}/split {frame_id, cls};
 * PATCH /api/units/{id} {cls}; DELETE /api/units/{id}?hide_place=; GET /api/units/{id}/timeline;
 * GET /api/sites/{id}/annotations; DELETE /api/annotations/batches/{batch}; GET /api/sites/{id}/dataset.zip.
 */
document.addEventListener("alpine:init", () => {
  Alpine.data("unitTools", () => ({
    pick: [],            // id машин, отмеченных для склейки
    dlg: null,           // {kind: merge|relabel|split|delete, unit, cls, busy, items, frame_id, hide, target_id}
    menuFor: null,       // id машины с открытым меню «⋯»
    journal: null,
    journalOpen: false,
    replay: null,

    isPicked(u) { return this.pick.includes(u.id); },
    togglePick(u) { this.pick = this.isPicked(u) ? this.pick.filter((x) => x !== u.id) : [...this.pick, u.id]; },
    get pickedUnits() { return (this.units || []).filter((u) => this.pick.includes(u.id)); },
    get uClasses() {
      const s = Alpine.store("app").settings;
      const list = (s && s.classes && s.classes.length) ? s.classes
        : Object.keys(SV.classNames).map((k) => ({ key: k, name: SV.classNames[k], tz: false }));
      return list.slice().sort((a, b) => (b.tz ? 1 : 0) - (a.tz ? 1 : 0) || String(a.name).localeCompare(String(b.name), "ru"));
    },
    clsName(k) { return SV.classNames[k] || k; },

    // ------------------------------------------------------------ диалоги
    openMerge() {
      const us = this.pickedUnits;
      if (us.length < 2) return SV.toast("Отметьте флажками две и более машины, которые на деле одна", { kind: "info" });
      const target = us.slice().sort((a, b) => (b.manual ? 1 : 0) - (a.manual ? 1 : 0) || (b.worked_hours || 0) - (a.worked_hours || 0))[0];
      this.dlg = { kind: "merge", units: us, target_id: target.id, cls: target.cls, busy: false };
    },
    openRelabel(u) { this.menuFor = null; this.dlg = { kind: "relabel", unit: u, cls: u.cls, busy: false }; },
    openDelete(u) { this.menuFor = null; this.dlg = { kind: "delete", unit: u, hide: !(u.worked_hours > 0) && !u.last_moved, busy: false }; },
    async openSplit(u) {
      this.menuFor = null;
      this.dlg = { kind: "split", unit: u, cls: u.cls, items: null, frame_id: null, busy: false, error: null };
      try {
        const tl = await SV.api.get(`/api/units/${u.id}/timeline?limit=600`);
        if (this.dlg && this.dlg.kind === "split") this.dlg.items = tl.items || [];
      } catch (e) { if (this.dlg) this.dlg.error = e.message; }
    },
    /* Кадры машины по камерам — выбрать, с какого она «другая». */
    get splitGroups() {
      const d = this.dlg;
      if (!d || !d.items) return [];
      const by = {};
      for (const it of d.items) (by[it.camera_id] = by[it.camera_id] || { camera_id: it.camera_id, name: it.camera_name || this.camName(it.camera_id), items: [] }).items.push(it);
      return Object.values(by);
    },
    get splitFrom() {
      const d = this.dlg;
      return d && d.items && d.frame_id ? d.items.find((x) => x.frame_id === d.frame_id) || null : null;
    },
    get splitCount() {
      const f = this.splitFrom;
      return f ? this.dlg.items.filter((x) => x.camera_id === f.camera_id && x.captured_at >= f.captured_at).length : 0;
    },
    /* Рамка превью: выбранный кадр — ярко, кадры, которые уйдут к новой машине, — бледнее. */
    splitMark(it) {
      const f = this.splitFrom;
      if (!f) return "border-transparent";
      if (it.frame_id === f.frame_id) return "border-info";
      return it.camera_id === f.camera_id && it.captured_at >= f.captured_at ? "border-info/40" : "border-transparent";
    },
    thumbStyle(it) {
      const cam = (this.cam && this.cam(it.camera_id)) || {};
      const W = cam.image_w || 1280, H = cam.image_h || 720;
      let [x, y, w, h] = it.bbox.map(Number);
      const pad = Math.max(w, h) * 0.35;
      x -= pad; y -= pad; w += 2 * pad; h += 2 * pad;
      const cw = 88, ch = 60, ar = cw / ch;
      if (w / h > ar) { const nh = w / ar; y -= (nh - h) / 2; h = nh; } else { const nw = h * ar; x -= (nw - w) / 2; w = nw; }
      const s = cw / w;
      return `background-image:url('${String(it.url).replace(/'/g, "%27")}');background-size:${W * s}px ${H * s}px;background-position:${-x * s}px ${-y * s}px`;
    },

    // ------------------------------------------------------------ действия
    async _do(fn, okText) {
      if (!this.dlg || this.dlg.busy) return;
      this.dlg.busy = true;
      try {
        const res = await fn();
        this.dlg = null;
        this.pick = [];
        SV.toast(okText(res), { kind: "success", timeout: 8000, action: res && res.batch ? { label: "Отменить", fn: () => this.undo(res.batch) } : null });
        await this.follow(res && res.replay);
      } catch (e) {
        if (this.dlg) this.dlg.busy = false;
        SV.toastError(e, "Правка не сохранена");
      }
    },
    doMerge() {
      const d = this.dlg;
      return this._do(() => SV.api.post("/api/units/merge", { unit_ids: d.units.map((u) => u.id), unit_uids: d.units.map((u) => u.uid), site_id: this.siteId, target_id: d.target_id, cls: d.cls }),
        () => `Склеено: ${d.units.length} → одна машина «${this.clsName(d.cls)}». Моточасы и отклонения пересчитаны.`);
    },
    doRelabel() {
      const d = this.dlg;
      return this._do(() => SV.api.patch(`/api/units/${d.unit.id}`, { cls: d.cls, uid: d.unit.uid, site_id: this.siteId }),
        () => `«${this.unitLabel(d.unit)}» — теперь «${this.clsName(d.cls)}» на всех кадрах`);
    },
    doSplit() {
      const d = this.dlg;
      if (!d.frame_id) return SV.toast("Выберите кадр, с которого это другая машина", { kind: "info" });
      return this._do(() => SV.api.post(`/api/units/${d.unit.id}/split`, { frame_id: d.frame_id, cls: d.cls, uid: d.unit.uid, site_id: this.siteId }),
        () => `Машина разделена: ${this.splitCount} ${SV.fmt.plural(this.splitCount, "кадр", "кадра", "кадров")} — отдельная «${this.clsName(d.cls)}»`);
    },
    doDelete() {
      const d = this.dlg;
      return this._do(() => SV.api.del(`/api/units/${d.unit.id}?hide_place=${d.hide ? "true" : "false"}&uid=${encodeURIComponent(d.unit.uid)}&site_id=${this.siteId}`),
        () => `«${this.unitLabel(d.unit)}» убрана: не техника${d.hide ? "; место на камере больше не считается" : ""}`);
    },
    async undo(batch) {
      try {
        const res = await SV.api.del(`/api/annotations/batches/${encodeURIComponent(batch)}`);
        SV.toast("Правка отменена", { kind: "success" });
        await this.follow(res && res.replay);
      } catch (e) { SV.toastError(e, "Не удалось отменить"); }
    },

    /* Техника перепрогоняется сервером: ждём конца и обновляем вкладку целиком. */
    async follow(state) {
      this.replay = state && state.state !== "idle" ? state : null;
      while (this.replay) {
        await new Promise((r) => setTimeout(r, 1500));
        try {
          const st = await SV.api.get(`/api/sites/${this.siteId}/annotations/status`);
          this.replay = st.state === "idle" ? null : st;
        } catch (e) { this.replay = null; }
      }
      this.unitDets = {};
      await Promise.all([this.loadUnits(true), this.load(true), this.loadIntervals(), this.journalOpen ? this.loadJournal() : null]);
      this.$nextTick(() => this.drawHeat && this.drawHeat());
    },
    get replayText() {
      const r = this.replay;
      if (!r) return "";
      return r.total ? `Пересчёт техники: ${r.done} из ${r.total} кадров…` : "Пересчёт техники…";
    },

    // ------------------------------------------------------------ журнал и датасет
    async loadJournal() {
      try { this.journal = await SV.api.get(`/api/sites/${this.siteId}/annotations?limit=60`); } catch (e) { this.journal = { batches: [], stats: {}, error: e.message }; }
    },
    toggleJournal() { this.journalOpen = !this.journalOpen; if (this.journalOpen) this.loadJournal(); },
    kindIcon(k) { return { box_relabel: "pencil", box_delete: "trash", box_add: "plus", frame_verify: "check", unit_merge: "layers", unit_split: "move", unit_relabel: "pencil", unit_delete: "trash" }[k] || "hand"; },
    openBatchFrame(b) {
      if (b.frame_ids && b.frame_ids.length) window.dispatchEvent(new CustomEvent("sv:frame", { detail: { id: b.frame_ids[0], ids: b.frame_ids, title: b.kind_name } }));
    },
    datasetUrl(scope) { return `/api/sites/${this.siteId}/dataset.zip?scope=${scope}`; },
  }));
});
