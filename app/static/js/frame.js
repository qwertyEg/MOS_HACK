/*
 * Страница одного кадра (/frames/{id}): крупный кадр с рамками и полный разбор.
 * Сюда ведут ссылки из модалки доказательств — адрес можно отправить коллеге.
 */
document.addEventListener("alpine:init", () => {
  Alpine.data("framePage", (frameId) => SV.mix(SV.frameMixin(), {
    frameId,
    frame: null,
    error: null,
    zones: [],
    layers: { boxes: true, mask: false, zones: false },

    async init() {
      try {
        this.frame = await SV.api.get(`/api/frames/${frameId}`);
        if (this.frame.site_id != null) Alpine.store("app").siteId = this.frame.site_id;
        document.title = `Кадр ${SV.fmt.dateTime(this.frame.captured_at)} · СтройВзор`;
        SV.api.get(`/api/cameras/${this.frame.camera_id}/zones`).then((z) => { this.zones = z || []; }).catch(() => {});
        SV.api.get(`/api/cameras/${this.frame.camera_id}`).then((c) => { this.cam = c; }).catch(() => {});
        if (this.frame.site_id != null) SV.api.get(`/api/sites/${this.frame.site_id}/equipment`).then((d) => { SV.setUnits(d.units); this.unitsTick++; }).catch(() => {});
        Alpine.store("app").loadCatalog();
      } catch (e) {
        this.error = e.message;
      }
      window.addEventListener("keydown", (e) => {
        if (SV.isTyping(e) || !this.frame) return;
        if ([...document.querySelectorAll("[aria-modal=true]")].some((el) => el.getClientRects().length)) return;
        if (e.key === "ArrowLeft" && this.frame.prev_id) location.href = `/frames/${this.frame.prev_id}`;
        if (e.key === "ArrowRight" && this.frame.next_id) location.href = `/frames/${this.frame.next_id}`;
        const k = e.key.toLowerCase();
        if (k === "b" || k === "и") this.layers.boxes = !this.layers.boxes;
        if ((k === "m" || k === "ь") && this.maskUrl) this.layers.mask = !this.layers.mask;
        if (k === "z" || k === "я") this.layers.zones = !this.layers.zones;
      });
    },
    get svg() { return SV.boxesSvg(this.frame, { showBoxes: this.layers.boxes, showZones: this.layers.zones, zones: this.zones, highlight: this.highlight }); },
    cam: null,
    unitsTick: 0,
    get maskUrl() { return this.cam && this.cam.mask ? `${this.cam.mask.url}?t=${encodeURIComponent(this.cam.mask.updated_at || "")}` : null; },
    get providersLine() {
      const f = this.frame;
      if (!f) return "";
      const a = (f.detections || [])[0];
      const names = [a ? (SV.META.providers[a.provider] || {}).label || a.provider : null, f.checklist ? (SV.META.providers[f.checklist.provider] || {}).label || f.checklist.provider : null].filter(Boolean);
      return names.length ? "модели: " + names.join(" + ") : "";
    },
    get labels() { this.unitsTick; return SV.boxLabels(this.frame, { showBoxes: this.layers.boxes, dispW: this.dispW, conf: true }); },
  }));
});
