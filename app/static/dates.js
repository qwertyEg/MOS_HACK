/* Поля даты в формате дд/мм/гггг.
 *
 * Штатный <input type="date"> не годится: браузер рисует его по языку своего
 * интерфейса, и на англоязычной системе оператор видит мм/дд/гггг, что бы мы
 * ни поставили в разметке. Задать формат из страницы нельзя.
 *
 * Поэтому поле обычное текстовое, а календарь — штатный, но спрятанный:
 * кнопка открывает его, выбранная дата возвращается в текст уже в нашем
 * формате. Разметка полей остаётся простой — <input data-date name="…"> —
 * скрипт достраивает остальное, в том числе для строк, добавленных на месте.
 */
(function () {
  const pad = (n) => String(n).padStart(2, '0');

  function toIso(text) {
    const m = /^(\d{2})\/(\d{2})\/(\d{4})$/.exec(text);
    if (!m) return null;
    const d = +m[1], mo = +m[2], y = +m[3];
    const t = new Date(y, mo - 1, d);
    // Date молча «исправляет» 31/02 в 03/03 — сверяем обратно.
    if (t.getFullYear() !== y || t.getMonth() !== mo - 1 || t.getDate() !== d) return null;
    return `${y}-${pad(mo)}-${pad(d)}`;
  }

  function fromIso(iso) {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
    return m ? `${m[3]}/${m[2]}/${m[1]}` : '';
  }

  /* Что бы ни вставили — 25.09.2026, 25-09-2026, 2026-09-25 — приводим к
   * дд/мм/гггг. При стирании хвостовой слэш не дописываем, иначе его нельзя
   * удалить: поле дописывало бы его обратно. */
  function format(value, deleting) {
    const iso = /^(\d{4})-(\d{1,2})-(\d{1,2})$/.exec(value.trim());
    if (iso) return `${pad(+iso[3])}/${pad(+iso[2])}/${iso[1]}`;

    const d = value.replace(/\D/g, '').slice(0, 8);
    let out = d.slice(0, 2);
    if (d.length > 2) out += '/' + d.slice(2, 4);
    if (d.length > 4) out += '/' + d.slice(4, 8);
    if (!deleting && (d.length === 2 || d.length === 4)) out += '/';
    return out;
  }

  function validate(input) {
    const v = input.value.trim();
    const ok = v === '' || toIso(v) !== null;
    input.setCustomValidity(ok ? '' : 'Дата в формате дд/мм/гггг');
    input.classList.toggle('date-invalid', !ok);
  }

  function upgrade(input) {
    if (input.dataset.dateReady) return;
    input.dataset.dateReady = '1';

    const wrap = document.createElement('span');
    wrap.className = 'relative inline-block w-full';
    input.parentNode.insertBefore(wrap, input);
    wrap.appendChild(input);

    const native = document.createElement('input');
    native.type = 'date';
    native.tabIndex = -1;
    native.setAttribute('aria-hidden', 'true');
    native.style.cssText =
      'position:absolute;right:0;bottom:0;width:1px;height:1px;opacity:0;pointer-events:none';

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.title = 'Календарь';
    btn.className = 'absolute right-1.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-700';
    btn.innerHTML =
      '<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
      'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 11h18"/></svg>';
    wrap.append(btn, native);

    input.addEventListener('input', (e) => {
      const deleting = (e.inputType || '').startsWith('delete');
      input.value = format(input.value, deleting);
      validate(input);
    });
    input.addEventListener('blur', () => {
      input.value = input.value.replace(/\/$/, '');
      validate(input);
    });

    btn.addEventListener('click', () => {
      native.value = toIso(input.value.trim()) || '';
      // showPicker есть не везде; без него остаётся ввод руками.
      try { native.showPicker(); } catch (e) { native.focus(); }
    });
    native.addEventListener('change', () => {
      input.value = fromIso(native.value);
      validate(input);
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });

    validate(input);
  }

  window.upgradeDates = (root) =>
    (root || document).querySelectorAll('input[data-date]').forEach(upgrade);

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', () => window.upgradeDates());
  else window.upgradeDates();
})();
