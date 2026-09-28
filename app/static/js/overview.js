/*
 * Обзор объектов (GET /api/sites): карточки с вердиктом, этапом, техникой в
 * работе и открытыми отклонениями. Порядок — по риску: сначала отстающие
 * и с критичными отклонениями, потому что именно их руководитель открывает первыми.
 */
document.addEventListener("alpine:init", () => {
  Alpine.data("overviewPage", () => ({
    sites: null,
    loading: true,
    error: null,
    filter: "all",
    q: "",
    seeding: false,
    createOpen: false,
    creating: false,
    form: { name: "", address: "" },

    async init() {
      await this.load();
      // Карточки живые: кадры приходят каждые 20–30 минут, обновляемся чаще.
      this._t = setInterval(() => { if (!document.hidden) this.load(true); }, 60000);
    },
    destroy() { clearInterval(this._t); },

    async load(silent = false) {
      if (!silent) this.loading = true;
      try {
        const data = await SV.api.get("/api/sites");
        const prev = Object.fromEntries((this.sites || []).map((s) => [s.id, s]));
        // Критичные отклонения карточка /api/sites не отдаёт — дотягиваем ленты
        // отклонений (сохраняем прошлое значение, чтобы число не мигало).
        this.sites = (Array.isArray(data) ? data : data.items || []).map((s) =>
          Object.assign({ critical_deviations: prev[s.id] ? prev[s.id].critical_deviations : null }, s));
        this.error = null;
        Alpine.store("app").sites = this.sites;
        this.loadCritical();
      } catch (e) {
        if (!silent) this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    /* GET /api/sites/{id}/deviations?status=open — сколько из открытых критичных.
       Только для разумного числа объектов: иначе это N запросов на каждый опрос. */
    async loadCritical() {
      const list = (this.sites || []).filter((s) => s.open_deviations > 0);
      if (list.length > 30) return;
      await Promise.all(list.map(async (s) => {
        try {
          const devs = await SV.api.get(`/api/sites/${s.id}/deviations?status=open&limit=500`);
          s.critical_deviations = devs.filter((d) => d.severity === "critical").length;
        } catch (e) { /* без числа критичных карточка всё равно полезна */ }
      }));
      for (const s of this.sites || []) if (!s.open_deviations) s.critical_deviations = 0;
    },

    get today() { return SV.fmt.dateLong(new Date(), false) + ", " + SV.fmt.weekday(new Date()); },
    get counts() {
      const c = { behind: 0, on_track: 0, ahead: 0, other: 0 };
      for (const s of this.sites || []) c[s.verdict in c ? s.verdict : "other"]++;
      return c;
    },
    get filters() {
      const c = this.counts;
      const f = [{ key: "all", label: "Все", count: (this.sites || []).length }];
      if (c.behind) f.push({ key: "behind", label: "Отстают", count: c.behind, tone: "bad" });
      if (c.on_track) f.push({ key: "on_track", label: "В графике", count: c.on_track, tone: "good" });
      if (c.ahead) f.push({ key: "ahead", label: "Опережают", count: c.ahead, tone: "info" });
      if (c.other) f.push({ key: "other", label: "Без оценки", count: c.other, tone: "neutral" });
      return f;
    },
    get maxLag() { return Math.max(0, ...(this.sites || []).filter((s) => s.verdict === "behind").map((s) => s.lag_days || 0)); },
    get unitsActive() { return (this.sites || []).reduce((a, s) => a + (s.active_units || 0), 0); },
    get devTotal() { return (this.sites || []).reduce((a, s) => a + (s.open_deviations || 0), 0); },
    get devCritical() { return (this.sites || []).reduce((a, s) => a + (s.critical_deviations || 0), 0); },
    get camerasTotal() { return (this.sites || []).reduce((a, s) => a + (Number(s.cameras) || 0), 0); },
    get shown() {
      const q = this.q.trim().toLowerCase();
      const rank = { behind: 0, no_data: 2, no_plan: 2, on_track: 3, ahead: 4 };
      return (this.sites || [])
        .filter((s) => this.filter === "all" || s.verdict === this.filter || (this.filter === "other" && !["behind", "on_track", "ahead"].includes(s.verdict)))
        .filter((s) => !q || `${s.name} ${s.address || ""}`.toLowerCase().includes(q))
        .sort((a, b) => (rank[a.verdict] ?? 2) - (rank[b.verdict] ?? 2)
          || (b.critical_deviations || 0) - (a.critical_deviations || 0)
          || (b.lag_days || 0) - (a.lag_days || 0));
    },

    vmeta(s) { return SV.META.verdict[s.verdict] || SV.META.verdict.no_data; },
    verdictText(s) {
      const m = this.vmeta(s);
      if (s.verdict === "behind" && s.lag_days) return `${m.label} · ${SV.fmt.days(s.lag_days)}`;
      if (s.verdict === "ahead" && s.lag_days) return `${m.label} · ${SV.fmt.days(s.lag_days)}`;
      return m.label;
    },
    verdictChip(s) {
      return {
        bad: "bg-bad text-white", good: "bg-good text-white", info: "bg-info text-white",
        warn: "bg-warn text-black", neutral: "bg-black/60 text-white",
      }[this.vmeta(s).tone];
    },
    stageName(s) {
      const c = s.current_stage;
      if (!c) return "не определён";
      if (typeof c === "object") return c.name || `Этап ${c.id}`;
      return s.current_stage_name || `Этап ${c}`;
    },
    fresh(s) {
      const d = SV.fmt.toDate(s.last_frame_at);
      return d && Date.now() - d.getTime() < 90 * 60000;
    },

    /* POST /api/demo/seed {} → {sites:[{id,name,cameras}], jobs:[id], warnings[]}.
       Засеваются все каталоги datasets/demo/sites; уже существующие пропускаются. */
    async seed() {
      this.seeding = true;
      try {
        const r = await SV.api.post("/api/demo/seed", {});
        const made = (r && r.sites) || [];
        if (made.length) {
          const frames = made.reduce((a, s) => a + (s.cameras || []).reduce((b, c) => b + (c.files || 0), 0), 0);
          SV.toast(`Создано ${made.length} ${SV.fmt.plural(made.length, "демо-объект", "демо-объекта", "демо-объектов")}, ${frames} ${SV.fmt.plural(frames, "кадр", "кадра", "кадров")} поставлено в анализ.`, { kind: "success" });
          if (made.length === 1) { location.href = `/sites/${made[0].id}`; return; }
        } else {
          SV.toast((r && r.warnings && r.warnings[0]) || "Демо-объекты уже созданы.", { kind: "info" });
        }
        Alpine.store("app").loadQueue();
        await this.load();
      } catch (e) {
        SV.toastError(e, "Демо-объект не создан");
      } finally {
        this.seeding = false;
      }
    },
    async create() {
      this.creating = true;
      try {
        const s = await SV.api.post("/api/sites", { name: this.form.name.trim(), address: this.form.address.trim() });
        location.href = `/sites/${s.id}`;
      } catch (e) {
        SV.toastError(e, "Объект не создан");
      } finally {
        this.creating = false;
      }
    },
  }));
});
