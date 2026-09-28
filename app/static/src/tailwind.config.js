/*
 * Tailwind для СтройВзора. Собранный CSS лежит в git (app/static/css/app.css),
 * поэтому сервису для запуска не нужны ни Node, ни интернет.
 *
 * Пересборка после правки шаблонов/JS (standalone CLI v3.4.17, без npm; бинарник
 * tailwindcss-linux-x64 лежит на сервере в /root/mos_hack/bin/tailwindcss):
 *   tailwindcss -c app/static/src/tailwind.config.js \
 *               -i app/static/src/app.css -o app/static/css/app.css --minify
 *
 * Цвета — через CSS-переменные (см. app.css): одна разметка, две темы,
 * и прозрачность через /alpha работает для каждого токена.
 */
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;

module.exports = {
  darkMode: "class",
  content: {
    relative: true,
    files: ["../../templates/**/*.html", "../js/**/*.js"],
  },
  theme: {
    extend: {
      colors: {
        bg: v("bg"),
        panel: v("panel"),
        "panel-2": v("panel-2"),
        "panel-3": v("panel-3"),
        line: v("line"),
        "line-strong": v("line-strong"),
        fg: v("fg"),
        "fg-2": v("fg-2"),
        "fg-3": v("fg-3"),
        good: v("good"),
        warn: v("warn"),
        bad: v("bad"),
        info: v("info"),
        "good-fg": v("good-fg"),
        "warn-fg": v("warn-fg"),
        "bad-fg": v("bad-fg"),
        "info-fg": v("info-fg"),
      },
      fontFamily: {
        sans: ["InterVariable", "Inter", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains MonoVariable", "JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        "2xs": ["11px", "14px"],
        xs: ["12px", "16px"],
        sm: ["13px", "18px"],
        base: ["14px", "20px"],
      },
      borderRadius: { xl: "12px", "2xl": "16px" },
      opacity: { 8: "0.08", 12: "0.12", 14: "0.14", 35: "0.35", 45: "0.45", 55: "0.55", 65: "0.65", 85: "0.85" },
      boxShadow: {
        pop: "0 0 0 1px rgb(var(--line) / 0.10), 0 12px 32px -8px rgb(0 0 0 / 0.35), 0 4px 10px -4px rgb(0 0 0 / 0.25)",
        card: "0 1px 0 0 rgb(255 255 255 / 0.03) inset, 0 1px 2px 0 rgb(0 0 0 / 0.20)",
      },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
        "fade-in": { from: { opacity: 0, transform: "translateY(4px)" }, to: { opacity: 1, transform: "none" } },
        pulse2: { "0%,100%": { opacity: 1 }, "50%": { opacity: 0.35 } },
      },
      animation: {
        shimmer: "shimmer 1.6s infinite",
        "fade-in": "fade-in .18s ease-out both",
        pulse2: "pulse2 1.8s ease-in-out infinite",
      },
      screens: { xs: "480px" },
    },
  },
  plugins: [],
};
