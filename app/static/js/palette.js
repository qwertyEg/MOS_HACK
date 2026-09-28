/*
 * Единая палитра классов техники. Та же таблица HEX должна стоять в
 * core/equipment/draw.py (там — BGR для OpenCV), чтобы рамки на кадре с
 * сервера (/api/frames/{id}/annotated.jpg) и рамки SVG-оверлея в браузере
 * совпадали по цвету. Ключи — из reference/checklist.json (21 класс);
 * tests/ui/test_palette.py сверяет их со справочником.
 *
 * Восемь обязательных по ТЗ классов занимают восемь слотов проверенной
 * категориальной палитры (порядок слотов — механизм различимости при
 * дальтонизме, валидатор: соседние пары ΔE ≥ 8.4 при протанопии, ≥ 19 без неё;
 * все 8 проходят контраст ≥ 3:1 на тёмной поверхности). Остальные 13 классов
 * встречаются редко и получают дополнительные оттенки; различимость между ними
 * не гарантируется, поэтому рамка всегда подписана («Экскаватор · работает»).
 */
(function () {
  const CLASS_COLORS = {
    // обязательные по ТЗ — слоты 1..8
    excavator: "#3987e5",         // 1 синий
    dump_truck: "#d95926",        // 2 оранжевый
    bulldozer: "#199e70",         // 3 бирюзовый
    mobile_crane: "#c98500",      // 4 жёлтый
    concrete_mixer: "#d55181",    // 5 маджента
    roller: "#008300",            // 6 зелёный
    truck: "#9085e9",             // 7 фиолетовый
    crane_manipulator: "#e66767", // 8 красный
    // дополнительные
    tower_crane: "#0ea5e9",
    crawler_crane: "#14b8a6",
    concrete_pump: "#c026d3",
    drilling_rig: "#b45309",
    pile_driver: "#a8a29e",
    wheel_loader: "#84cc16",
    skid_steer: "#65a30d",
    backhoe_loader: "#6366f1",
    telehandler: "#0891b2",
    grader: "#a3a635",
    asphalt_paver: "#78716c",
    aerial_platform: "#f472b6",
    facade_hoist: "#94a3b8",
  };
  const FALLBACK = "#a1a1aa";

  function color(cls) {
    return CLASS_COLORS[cls] || FALLBACK;
  }

  // Подпись на цветной плашке: тёмный текст на светлых оттенках, белый — на тёмных.
  function ink(hex) {
    const n = parseInt(String(hex).slice(1), 16);
    const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => {
      c /= 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return L > 0.22 ? "#0b0b0c" : "#ffffff";
  }

  window.SV = window.SV || {};
  window.SV.palette = { CLASS_COLORS, color, ink, FALLBACK };
})();
