/*
 * Обёртка над ECharts: цвета берутся из CSS-токенов темы, график
 * перерисовывается при смене темы и размера контейнера.
 *
 * Правила оформления (единые для всех графиков): тонкие линии 2 px, сетка —
 * едва заметная, подписи осей приглушённые, одна ось Y, легенда своя (HTML),
 * всплывающая подсказка с перекрестием у линий и по ячейке у тепловой карты.
 */
(function () {
  const t = (n, a) => SV.token(n, a);

  function base() {
    return {
      animationDuration: 350,
      textStyle: { fontFamily: "InterVariable, Inter, system-ui, sans-serif", color: t("fg-2") },
      grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
      tooltip: {
        backgroundColor: t("panel"),
        borderColor: t("line-strong"),
        borderWidth: 1,
        padding: [8, 10],
        textStyle: { color: t("fg"), fontSize: 12 },
        extraCssText: "border-radius:10px;box-shadow:0 12px 32px -8px rgba(0,0,0,.35);",
      },
    };
  }
  function axisCommon() {
    return {
      axisLine: { lineStyle: { color: t("line-strong") } },
      axisTick: { show: false },
      axisLabel: { color: t("fg-3"), fontSize: 11 },
      splitLine: { lineStyle: { color: t("line"), width: 1 } },
    };
  }

  /* chart(el, build) — build() возвращает option; вызывается заново при смене темы. */
  function chart(el, build) {
    if (!el || !window.echarts) return null;
    let inst = echarts.init(el, null, { renderer: "canvas" });
    const render = () => {
      const opt = build();
      if (opt) inst.setOption(opt, true);
    };
    render();
    const ro = new ResizeObserver(() => inst && inst.resize());
    ro.observe(el);
    const onTheme = () => requestAnimationFrame(render);
    window.addEventListener("sv:theme", onTheme);
    return {
      inst,
      update: render,
      dispose() { ro.disconnect(); window.removeEventListener("sv:theme", onTheme); inst.dispose(); inst = null; },
    };
  }

  window.SV = window.SV || {};
  window.SV.charts = { chart, base, axisCommon, t };
})();
