/*
 * Общий слой интерфейса: форматирование (всё время — по Москве), словари
 * «код → русская подпись + тон», глобальные Alpine-хранилища (тема, тосты,
 * настройки/готовность провайдеров, список объектов), шапка и модалка кадра.
 *
 * Подключается на каждой странице ДО alpine.min.js: компоненты регистрируются
 * в обработчике `alpine:init`, который Alpine вызывает перед стартом.
 *
 * Правило безопасности: данные из API выводятся только через x-text / :attr.
 * x-html используется лишь для иконок и SVG-рамок, собранных из чисел и наших
 * же констант, — имя камеры «<script>» останется текстом (урок stored XSS
 * из ветки Дениса, site.html:191).
 */
(function () {
  const TZ = "Europe/Moscow";
  const NNBSP = " "; // узкий неразрывный пробел: «86 %», «12 ч»

  // ------------------------------------------------------------ форматирование
  const isDateOnly = (s) => typeof s === "string" && /^\d{4}-\d{2}-\d{2}$/.test(s);
  function toDate(v) {
    if (!v) return null;
    if (v instanceof Date) return v;
    // Дата без времени — это календарный день, а не полночь UTC: иначе
    // в Москве «22 сентября» превращалось бы в «22-е, 03:00» и сдвигалось.
    if (isDateOnly(v)) return new Date(v + "T12:00:00Z");
    const d = new Date(v);
    return isNaN(d) ? null : d;
  }
  const dtf = (o) => new Intl.DateTimeFormat("ru-RU", Object.assign({ timeZone: TZ }, o));
  const F = {
    day: dtf({ day: "numeric", month: "short" }),
    dayY: dtf({ day: "numeric", month: "short", year: "numeric" }),
    long: dtf({ day: "numeric", month: "long", year: "numeric" }),
    longNoY: dtf({ day: "numeric", month: "long" }),
    time: dtf({ hour: "2-digit", minute: "2-digit" }),
    dt: dtf({ day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }),
    wd: dtf({ weekday: "short" }),
    ymd: dtf({ year: "numeric", month: "2-digit", day: "2-digit" }),
    hour: dtf({ hour: "2-digit", hourCycle: "h23" }),
    month: dtf({ month: "short" }),
    monthY: dtf({ month: "short", year: "2-digit" }),
  };
  const clean = (s) => s.replace(/\s*г\.$/, "").replace(" г.", "");
  const nowYear = () => Number(dtf({ year: "numeric" }).format(new Date()));

  function plural(n, one, few, many) {
    const a = Math.abs(n) % 100, b = a % 10;
    if (a > 10 && a < 20) return many;
    if (b > 1 && b < 5) return few;
    if (b === 1) return one;
    return many;
  }

  const fmt = {
    plural,
    num(v, digits = 0) {
      if (v === null || v === undefined || isNaN(v)) return "—";
      return Number(v).toLocaleString("ru-RU", { maximumFractionDigits: digits, minimumFractionDigits: digits });
    },
    pct(v, digits = 0) {
      if (v === null || v === undefined || isNaN(v)) return "—";
      return fmt.num(v * 100, digits) + NNBSP + "%";
    },
    hours(h, digits = 0) {
      if (h === null || h === undefined || isNaN(h)) return "—";
      return fmt.num(h, digits) + NNBSP + "ч";
    },
    /* Длительность в часах → «2 ч 10 мин», «3 сут 4 ч», «45 мин». */
    dur(hours) {
      if (hours === null || hours === undefined || isNaN(hours)) return "—";
      const m = Math.round(hours * 60);
      if (m < 1) return "меньше минуты";
      if (m < 60) return `${m}${NNBSP}мин`;
      const h = Math.floor(m / 60), mm = m % 60;
      if (h < 48) return mm ? `${h}${NNBSP}ч ${mm}${NNBSP}мин` : `${h}${NNBSP}ч`;
      const dd = Math.floor(h / 24), hh = h % 24;
      return hh ? `${dd}${NNBSP}сут ${hh}${NNBSP}ч` : `${dd}${NNBSP}сут`;
    },
    days(n) {
      if (n === null || n === undefined || isNaN(n)) return "—";
      const r = Math.round(Math.abs(n));
      return `${r}${NNBSP}${plural(r, "день", "дня", "дней")}`;
    },
    date(v) {
      const d = toDate(v);
      if (!d) return "—";
      const y = Number(dtf({ year: "numeric" }).format(d));
      return clean(y === nowYear() ? F.day.format(d) : F.dayY.format(d)).replace(".", "");
    },
    dateLong(v, withYear = true) {
      const d = toDate(v);
      if (!d) return "—";
      return clean((withYear ? F.long : F.longNoY).format(d));
    },
    time(v) { const d = toDate(v); return d ? F.time.format(d) : "—"; },
    /* «9 июн, 18:23»; прошлые годы — с годом: «9 июн 2020, 18:23» (архивы камер). */
    dateTime(v) {
      const d = toDate(v);
      if (!d) return "—";
      const y = Number(dtf({ year: "numeric" }).format(d));
      if (y === nowYear()) return F.dt.format(d).replace(".", "");
      return `${clean(F.dayY.format(d)).replace(".", "")}, ${F.time.format(d)}`;
    },
    weekday(v) { const d = toDate(v); return d ? F.wd.format(d) : ""; },
    hourOf(v) { const d = toDate(v); return d ? Number(F.hour.format(d)) : null; },
    /* ISO-день по Москве: группировка кадров и интервалов по календарным суткам. */
    ymd(v) {
      const d = toDate(v);
      if (!d) return "";
      const p = F.ymd.formatToParts(d);
      const g = (t) => p.find((x) => x.type === t).value;
      return `${g("year")}-${g("month")}-${g("day")}`;
    },
    rel(v) {
      const d = toDate(v);
      if (!d) return "—";
      const s = (Date.now() - d.getTime()) / 1000;
      if (s < 0) return fmt.dateTime(v);
      if (s < 60) return "только что";
      if (s < 3600) return `${Math.floor(s / 60)}${NNBSP}мин назад`;
      if (s < 6 * 3600) {
        const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
        return m ? `${h}${NNBSP}ч ${m}${NNBSP}мин назад` : `${h}${NNBSP}ч назад`;
      }
      const today = fmt.ymd(new Date());
      const yest = fmt.ymd(new Date(Date.now() - 86400000));
      if (fmt.ymd(d) === today) return `сегодня в ${F.time.format(d)}`;
      if (fmt.ymd(d) === yest) return `вчера в ${F.time.format(d)}`;
      const days = Math.floor(s / 86400);
      if (days < 7) return `${days}${NNBSP}${plural(days, "день", "дня", "дней")} назад`;
      return fmt.date(v);
    },
    /* Для <input type=date>/<input type=datetime-local> (по Москве). */
    inputDate(v) { return v ? (isDateOnly(v) ? v : fmt.ymd(v)) : ""; },
    daysBetween(a, b) {
      const x = toDate(a), y = toDate(b);
      if (!x || !y) return null;
      return Math.round((y - x) / 86400000);
    },
    toDate,
  };

  // ------------------------------------------------------------ словари
  const META = {
    verdict: {
      behind: { label: "Отставание", tone: "bad", icon: "trending-down" },
      on_track: { label: "В графике", tone: "good", icon: "check-circle" },
      ahead: { label: "Опережение", tone: "info", icon: "trending-up" },
      no_plan: { label: "Нет плана", tone: "neutral", icon: "calendar" },
      no_data: { label: "Нет данных", tone: "neutral", icon: "image" },
    },
    severity: {
      critical: { label: "Критично", tone: "bad", icon: "alert-octagon", rank: 0 },
      warning: { label: "Внимание", tone: "warn", icon: "alert-triangle", rank: 1 },
      info: { label: "К сведению", tone: "info", icon: "info", rank: 2 },
    },
    unit: {
      active: { label: "Работает", tone: "good", icon: "activity" },
      idle: { label: "Стоит", tone: "warn", icon: "pause" },
      parked: { label: "На стоянке", tone: "neutral", icon: "parking" },
      departed: { label: "Уехала", tone: "outline", icon: "log-out" },
    },
    activity: { working: "работает", idle: "стоит", unknown: "" },
    stage: {
      done: { label: "Завершён", tone: "good" },
      active: { label: "Идёт", tone: "info" },
      not_started: { label: "Не начат", tone: "neutral" },
    },
    devType: {
      equipment_missing: "Нет нужной техники",
      equipment_forbidden: "Техника не по этапу",
      pair_broken: "Разорвана пара техники",
      equipment_idle: "Простой техники",
      equipment_parked_only: "Техника на стоянке",
      outside_zone: "Техника вне зоны",
      hours_spent_no_progress: "Часы ушли — этап стоит",
      stage_late_start: "Этап начат позже плана",
      stage_overdue: "Этап просрочен",
      stage_early: "Этап раньше плана",
      stage_out_of_plan: "Этап вне плана",
      needs_review: "Проверить вручную",
      camera_issue: "Проблема с камерой",
    },
    weather: {
      clear: { label: "Ясно", icon: "sun" },
      rain: { label: "Дождь, капли", icon: "cloud-rain" },
      snow: { label: "Снег", icon: "snowflake" },
      fog: { label: "Туман", icon: "fog" },
      unknown: { label: "Погода неизвестна", icon: "cloud" },
    },
    answer: {
      yes: { label: "Да", tone: "good" },
      no: { label: "Нет", tone: "neutral" },
      unsure: { label: "Не уверен", tone: "warn" },
      not_visible: { label: "Не видно", tone: "warn" },
    },
    zoneKind: { work: "Рабочая", parking: "Стоянка", storage: "Склад", restricted: "Запретная" },
    job: {
      queued: "В очереди", pending: "В очереди", running: "Обработка", done: "Готово", finished: "Готово",
      error: "Ошибка", failed: "Ошибка", cancelled: "Отменено",
    },
    providers: {
      yolo: { label: "YOLO", what: "Модель А · локально", kind: "local" },
      siglip: { label: "SigLIP", what: "Модель Б · локально", kind: "local" },
      local_vlm: { label: "Локальная VLM", what: "Модель Б · Ollama / vLLM", kind: "local" },
      glm: { label: "GLM-4.6V", what: "Модели А и Б · z.ai", kind: "external" },
    },
    modes: {
      local: { label: "Локальные модели", short: "Локально", desc: "YOLO + SigLIP на этом компьютере, без интернета", needs: ["yolo", "siglip"], icon: "cpu" },
      external: { label: "Внешний API", short: "Внешний API", desc: "GLM-4.6V (z.ai) — нужен ключ и интернет", needs: ["glm"], icon: "cloud" },
      hybrid: { label: "Гибрид", short: "Гибрид", desc: "Техника — YOLO локально, этап — GLM-4.6V", needs: ["yolo", "glm"], icon: "layers" },
    },
  };
  const toneBadge = { good: "badge-good", warn: "badge-warn", bad: "badge-bad", info: "badge-info", neutral: "badge-neutral", outline: "badge-outline" };
  const toneText = { good: "text-good-fg", warn: "text-warn-fg", bad: "text-bad-fg", info: "text-info-fg", neutral: "text-fg-2", outline: "text-fg-3" };
  const toneBg = { good: "bg-good", warn: "bg-warn", bad: "bg-bad", info: "bg-info", neutral: "bg-fg-3", outline: "bg-fg-3" };

  // ------------------------------------------------------------ иконки
  // Пути — в partials/icons.html (один источник и для Jinja-макроса, и для JS).
  function icon(name, cls = "w-4 h-4") {
    const p = (window.SV_ICONS || {})[name] || "";
    return `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;
  }

  // ------------------------------------------------------------ тосты
  function toast(message, opts = {}) {
    const store = window.Alpine && Alpine.store("toasts");
    if (store) store.push(message, opts);
    else console.log("[toast]", message);
  }
  function toastError(e, prefix) {
    const msg = e && e.message ? e.message : String(e);
    toast(prefix ? `${prefix}: ${msg}` : msg, { kind: "error" });
  }

  // ------------------------------------------------------------ тема
  function applyTheme(dark) {
    document.documentElement.classList.toggle("dark", dark);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    try { localStorage.setItem("sv-theme", dark ? "dark" : "light"); } catch (e) { /* приватное окно */ }
    window.dispatchEvent(new CustomEvent("sv:theme", { detail: { dark } }));
  }

  /* Цвет CSS-токена для графиков: ECharts не понимает var(--x). */
  function token(name, alpha = 1) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
    if (!v) return "#888";
    const [r, g, b] = v.split(/\s+/);
    return alpha < 1 ? `rgba(${r},${g},${b},${alpha})` : `rgb(${r},${g},${b})`;
  }

  // ------------------------------------------------------------ рамки на кадре
  /* SVG-разметка рамок/зон/точек: только числа и наши константы (см. шапку). */
  function boxesSvg(frame, opts = {}) {
    if (!frame) return "";
    const W = frame.width || 1280;
    const parts = [];
    if (opts.zones && opts.showZones) {
      for (const z of opts.zones) {
        const pts = (z.polygon || []).map((p) => `${+p[0]},${+p[1]}`).join(" ");
        const c = z.kind === "parking" ? "#a1a1aa" : z.kind === "restricted" ? "#e66767" : "#3987e5";
        parts.push(`<polygon class="zone" points="${pts}" fill="${c}" fill-opacity="0.14" stroke="${c}" stroke-dasharray="6 4"/>`);
      }
    }
    if (opts.showBoxes !== false) {
      (frame.detections || []).forEach((d, i) => {
        const [x, y, w, h] = (d.bbox || [0, 0, 0, 0]).map(Number);
        const c = SV.palette.color(d.class || d.cls);
        const idle = d.activity === "idle" ? " box-idle" : "";
        const hl = opts.highlight === i ? ' stroke-width="3.5"' : "";
        parts.push(`<rect class="box-halo" x="${x}" y="${y}" width="${w}" height="${h}" rx="${W / 400}"/>`);
        parts.push(`<rect class="box${idle}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${W / 400}" stroke="${c}"${hl}/>`);
        // Рамка, исправленная или дорисованная оператором, — белый внутренний контур.
        if (d.manual) parts.push(`<rect class="box-manual" x="${x + 2}" y="${y + 2}" width="${Math.max(0, w - 4)}" height="${Math.max(0, h - 4)}" rx="${W / 400}"/>`);
      });
    }
    if (opts.draft && opts.draft.length) {
      const pts = opts.draft.map((p) => `${+p[0]},${+p[1]}`).join(" ");
      parts.push(`<polyline points="${pts}" fill="rgba(57,135,229,0.12)" stroke="#6aa8f0" stroke-width="2" vector-effect="non-scaling-stroke"/>`);
      opts.draft.forEach((p) => parts.push(`<circle cx="${+p[0]}" cy="${+p[1]}" r="${W / 180}" fill="#6aa8f0" stroke="#0b0b0c" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`));
    }
    if (opts.points && opts.points.length) {
      opts.points.forEach((p) => {
        parts.push(`<circle cx="${+p[0]}" cy="${+p[1]}" r="${W / 90}" fill="rgba(250,178,25,0.25)" stroke="#fab219" stroke-width="2" vector-effect="non-scaling-stroke"/>`);
        parts.push(`<circle cx="${+p[0]}" cy="${+p[1]}" r="${W / 400}" fill="#fab219"/>`);
      });
    }
    return parts.join("");
  }

  /* Подписи рамок — HTML поверх SVG в процентах: текст остаётся чётким при любом масштабе.
     Подписи не должны налезать друг на друга и вылезать за кадр: ширину оцениваем
     по длине текста (11px Inter ≈ 6.4px на знак) при известной ширине вьюера. */
  function boxLabels(frame, opts = {}) {
    if (!frame || opts.showBoxes === false) return [];
    const W = frame.width || 1280, H = frame.height || 1024;
    const dispW = opts.dispW || 760;
    const dispH = dispW * H / W;
    const placed = [];
    const hit = (a) => placed.some((b) => a.l < b.l + b.w && b.l < a.l + a.w && a.t < b.t + b.h && b.t < a.t + a.h);
    const dets = (frame.detections || []).map((d, i) => ({ d, i })).sort((a, b) => a.d.bbox[1] - b.d.bbox[1]);
    const out = [];
    for (const { d, i } of dets) {
      const [x, y, w, h] = (d.bbox || [0, 0, 0, 0]).map(Number);
      const cls = d.class || d.cls;
      const c = SV.palette.color(cls);
      let name = unitName(d.unit_id) || d.unit_label || d.name || (SV.classNames[cls] || cls);
      // Класс рамки задан вручную и расходится с машиной («только эта рамка») — подпись по рамке.
      if (d.manual && d.manual.cls && !String(name).startsWith(SV.classNames[cls] || cls)) name = SV.classNames[cls] || cls;
      const act = META.activity[d.activity] || "";
      const conf = opts.conf && d.conf != null ? ` · ${Math.round(d.conf * 100)}%` : "";
      const text = (d.manual ? "✎ " : "") + (act ? `${name} · ${act}` : name) + (d.manual ? "" : conf);
      // На узком экране подписи мельче, иначе закрывают саму технику.
      const small = dispW < 520;
      const lw = ((text.length * (small ? 5.6 : 6.4) + (small ? 8 : 12)) / dispW) * 100;
      const lh = ((small ? 15 : 17) / dispH) * 100;
      let l = (x / W) * 100;
      if (l + lw > 100) l = Math.max(0, ((x + w) / W) * 100 - lw);
      const top = (y / H) * 100;
      const candidates = [top - lh, top, top + lh, top + 2 * lh, top - 2 * lh];
      let t = candidates.find((tt) => tt >= 0 && tt + lh <= 100 && !hit({ l, t: tt, w: lw, h: lh }));
      if (t === undefined) t = Math.max(0, top - lh);
      placed.push({ l, t, w: lw, h: lh });
      out.push({ i, text, style: `left:${l}%;top:${t}%;background:${c};color:${SV.palette.ink(c)}${small ? ";font-size:10px;padding:2px 4px" : ""}` });
    }
    return out;
  }

  /* Подписи единиц техники: detection.unit_id — это uid единицы объекта («u0001»),
     а человеку нужно «Экскаватор №1». Страница, загрузившая /api/sites/{id}/equipment,
     кладёт сюда соответствие; uid уникальны только в пределах объекта — на странице он один. */
  const unitLabels = {};
  function setUnits(units) {
    for (const k of Object.keys(unitLabels)) delete unitLabels[k];
    for (const u of units || []) if (u.uid) unitLabels[u.uid] = u.label || `${u.name || u.cls} ${u.uid}`;
  }
  function unitName(uid) { return uid ? unitLabels[uid] || null : null; }

  /* Названия классов по-русски приходят из /api/settings (classes[]); до загрузки — ключ. */
  const classNames = {
    excavator: "Экскаватор", dump_truck: "Самосвал", bulldozer: "Бульдозер", roller: "Каток",
    concrete_mixer: "Автобетоносмеситель", truck: "Грузовик", mobile_crane: "Автокран",
    crane_manipulator: "Кран-манипулятор", tower_crane: "Башенный кран", crawler_crane: "Гусеничный кран",
    concrete_pump: "Автобетононасос", drilling_rig: "Буровая установка", pile_driver: "Копёр",
    wheel_loader: "Фронтальный погрузчик", skid_steer: "Мини-погрузчик", backhoe_loader: "Экскаватор-погрузчик",
    telehandler: "Телескопический погрузчик", grader: "Автогрейдер", asphalt_paver: "Асфальтоукладчик",
    aerial_platform: "Автовышка", facade_hoist: "Фасадный подъёмник",
  };

  window.SV = window.SV || {};
  Object.assign(window.SV, { fmt, META, icon, toast, toastError, applyTheme, token, boxesSvg, boxLabels, classNames, toneBadge, toneText, toneBg, setUnits, unitName });

  // ------------------------------------------------------------ Alpine
  document.addEventListener("alpine:init", () => {
    Alpine.magic("icon", () => (n, c) => icon(n, c));

    Alpine.store("theme", {
      dark: document.documentElement.classList.contains("dark"),
      toggle() { this.dark = !this.dark; applyTheme(this.dark); },
    });

    Alpine.store("toasts", {
      items: [],
      seq: 0,
      push(message, { kind = "info", action = null, timeout } = {}) {
        const id = ++this.seq;
        this.items.push({ id, message, kind, action });
        if (this.items.length > 4) this.items.shift();
        const t = timeout ?? (kind === "error" ? 7000 : action ? 6000 : 3800);
        if (t) setTimeout(() => this.dismiss(id), t);
      },
      dismiss(id) { this.items = this.items.filter((x) => x.id !== id); },
    });

    /* Настройки и готовность провайдеров — одна копия на страницу, шапка и
       страницы читают её отсюда, чтобы переключатель режима и «Настройки»
       не расходились. */
    Alpine.store("app", {
      settings: null,
      health: null,
      queue: null,
      catalog: null,
      sites: null,
      siteId: document.body.dataset.siteId ? Number(document.body.dataset.siteId) || document.body.dataset.siteId : null,
      settingsError: null,
      async loadSettings() {
        try {
          this.settings = await SV.api.get("/api/settings");
          this.settingsError = null;
          for (const c of this.settings.classes || []) if (c.key && c.name) classNames[c.key] = c.name;
        } catch (e) { this.settingsError = e.message; }
        return this.settings;
      },
      async loadHealth() {
        try { this.health = await SV.api.get("/api/health"); } catch (e) { this.health = { ok: false, error: e.message, providers: {} }; }
        return this.health;
      },
      /* Очередь анализа: GET /api/queue → {running, pending, busy, recompute_pending, cameras}. */
      async loadQueue() {
        try { this.queue = await SV.api.get("/api/queue"); } catch (e) { /* индикатор просто не покажется */ }
        return this.queue;
      },
      /* Справочник этапов/работ/техники/признаков чек-листа — один раз на страницу. */
      async loadCatalog() {
        if (this.catalog) return this.catalog;
        if (!this._catalogP) this._catalogP = SV.api.get("/api/catalog").then((c) => { this.catalog = c; return c; }).catch(() => null);
        return this._catalogP;
      },
      question(key) {
        const s = this.catalog && (this.catalog.signs || []).find((x) => x.key === key);
        return s ? s.question : key;
      },
      async loadSites() {
        try { this.sites = await SV.api.get("/api/sites"); } catch (e) { this.sites = this.sites || []; }
        return this.sites;
      },
      provider(key) { return (this.health && this.health.providers && this.health.providers[key]) || null; },
      /* Готов ли режим: все провайдеры пресета готовы. Возвращает причину первого неготового. */
      modeReady(mode) {
        const m = META.modes[mode];
        if (!m || !this.health) return { ready: null, reason: "" };
        for (const k of m.needs) {
          const p = this.provider(k);
          if (!p) return { ready: null, reason: `${META.providers[k]?.label || k}: нет данных о готовности` };
          if (!p.ready) return { ready: false, reason: `${META.providers[k]?.label || k}: ${p.reason || "не готов"}` };
        }
        return { ready: true, reason: "" };
      },
      async setMode(mode) {
        const prev = this.settings && this.settings.mode;
        if (this.settings) this.settings.mode = mode; // оптимистично: переключатель не «залипает»
        try {
          const res = await SV.api.put("/api/settings", { mode });
          if (res && typeof res === "object" && res.mode) this.settings = res;
          const m = META.modes[mode];
          toast(`Режим: ${m.label}. Новые кадры анализируются так; старые результаты сохранены.`, { kind: "success" });
          window.dispatchEvent(new CustomEvent("sv:mode", { detail: { mode } }));
          return true;
        } catch (e) {
          if (this.settings) this.settings.mode = prev;
          toastError(e, "Не удалось переключить режим");
          return false;
        }
      },
    });

    Alpine.data("svHeader", () => ({
      open: null, // 'sites' | 'mode' | 'menu' | 'user'
      pendingMode: null,
      query: "",
      async init() {
        const app = Alpine.store("app");
        await Promise.all([app.loadSettings(), app.loadHealth(), app.loadSites(), app.loadQueue()]);
        // Готовность провайдеров меняется сама (ключ добавили, модель загрузилась) — опрашиваем.
        this._t = setInterval(() => { if (!document.hidden) app.loadHealth(); }, 30000);
        // Очередь — чаще, пока в ней что-то есть: индикатор должен «тикать» вниз.
        const tickQueue = async () => {
          if (!document.hidden) await app.loadQueue();
          const busy = app.queue && (app.queue.pending || app.queue.busy);
          this._q = setTimeout(tickQueue, busy ? 4000 : 15000);
        };
        this._q = setTimeout(tickQueue, 4000);
      },
      destroy() { clearInterval(this._t); clearTimeout(this._q); },
      get app() { return Alpine.store("app"); },
      get mode() { return this.app.settings ? this.app.settings.mode : null; },
      get queue() {
        const q = this.app.queue;
        if (!q) return null;
        // pending — кадры в очередях камер, busy — сколько камер сейчас в работе.
        return { pending: Number(q.pending || 0), busy: Number(q.busy || 0), recompute: !!q.recompute_pending, running: q.running !== false };
      },
      get currentSite() {
        const s = this.app.sites || [];
        return s.find((x) => String(x.id) === String(this.app.siteId)) || null;
      },
      get filteredSites() {
        const q = this.query.trim().toLowerCase();
        return (this.app.sites || []).filter((s) => !q || (s.name || "").toLowerCase().includes(q) || (s.address || "").toLowerCase().includes(q));
      },
      toggle(which) { this.open = this.open === which ? null : which; this.pendingMode = null; if (which === "sites") this.$nextTick(() => this.$refs.siteSearch && this.$refs.siteSearch.focus()); },
      close() { this.open = null; this.pendingMode = null; },
      /* Переключение режима: если внешний API не готов — сначала объясняем почему. */
      async choose(mode) {
        if (mode === this.mode) return this.close();
        const r = this.app.modeReady(mode);
        if (r.ready === false && this.pendingMode !== mode) { this.pendingMode = mode; return; }
        this.close();
        await this.app.setMode(mode);
      },
      quick() {
        // Кнопка-переключатель в шапке: локально ⇄ внешний API (гибрид — в «Настройках»).
        this.choose(this.mode === "local" ? "external" : "local");
        if (this.pendingMode) this.open = "mode";
      },
    }));

    /* Модалка кадра-доказательства: открывается событием sv:frame {id, ids[], title}. */
    Alpine.data("svFrameModal", () => ({
      isOpen: false,
      ids: [],
      idx: 0,
      title: "",
      frame: null,
      loading: false,
      error: null,
      showBoxes: true,
      cache: {},
      init() {
        window.addEventListener("sv:frame", (e) => this.show(e.detail || {}));
      },
      async show({ id, ids, title }) {
        this.ids = ids && ids.length ? ids : [id];
        this.idx = Math.max(0, this.ids.findIndex((x) => String(x) === String(id)));
        this.title = title || "Снимок";
        this.isOpen = true;
        await this.load();
      },
      async load() {
        const id = this.ids[this.idx];
        this.error = null;
        if (this.cache[id]) { this.frame = this.cache[id]; return; }
        this.loading = true;
        try {
          this.frame = this.cache[id] = await SV.api.get(`/api/frames/${encodeURIComponent(id)}`);
        } catch (e) {
          this.error = e.message;
          this.frame = null;
        } finally { this.loading = false; }
      },
      step(d) {
        if (this.ids.length < 2) return;
        this.idx = (this.idx + d + this.ids.length) % this.ids.length;
        this.load();
      },
      close() { this.isOpen = false; },
      dispW: 900,
      measure(el) {
        const set = () => { this.dispW = el.clientWidth || 900; };
        set();
        if (!this._ro) { this._ro = new ResizeObserver(set); }
        this._ro.observe(el);
      },
      get svg() { return boxesSvg(this.frame, { showBoxes: this.showBoxes }); },
      get labels() { return boxLabels(this.frame, { showBoxes: this.showBoxes, dispW: this.dispW }); },
    }));

    /* Шпаргалка по клавишам: «?» из любого места, кроме полей ввода. */
    Alpine.data("svShortcuts", () => ({
      open: false,
      init() {
        window.addEventListener("keydown", (e) => {
          if (isTyping(e)) return;
          if (e.key === "?" || (e.shiftKey && e.code === "Slash")) { this.open = !this.open; e.preventDefault(); }
          if (e.key === "t" && !e.metaKey && !e.ctrlKey && !e.altKey) Alpine.store("theme").toggle();
        });
      },
    }));
  });

  function isTyping(e) {
    const t = e.target;
    return t && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));
  }
  window.SV.isTyping = isTyping;
})();
