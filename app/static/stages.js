/* Редактор календарного плана: состав, порядок, даты.
 *
 * Порядок строк и есть порядок этапов — отдельного номера позиции нет.
 * Браузер отправляет одноимённые поля в том порядке, в каком они лежат
 * в разметке, а перетаскивание строку в разметке и двигает. Значит после
 * сохранения порядок на сервере совпадёт с тем, что видит человек, без
 * пересчёта индексов и без рассинхрона между показанным и сохранённым.
 */
(function () {
  const tbody = document.getElementById('stage-rows');
  if (!tbody) return;

  const tpl = document.getElementById('stage-row-template');
  const picker = document.getElementById('stage-picker');
  const addBtn = document.getElementById('stage-add');
  const empty = document.getElementById('stage-empty');

  let dragged = null;

  function refresh() {
    if (empty) empty.hidden = tbody.children.length > 0;
    // В списке добавления остаётся только то, чего ещё нет в плане.
    if (!picker) return;
    const used = new Set([...tbody.children].map((r) => r.dataset.macro));
    [...picker.options].forEach((o) => {
      if (o.value) o.hidden = used.has(o.value);
    });
  }

  function wire(row) {
    row.addEventListener('dragstart', (e) => {
      dragged = row;
      row.classList.add('opacity-40');
      e.dataTransfer.effectAllowed = 'move';
      // Без этого Firefox не начинает перетаскивание вовсе.
      e.dataTransfer.setData('text/plain', '');
    });
    row.addEventListener('dragend', () => {
      row.classList.remove('opacity-40');
      dragged = null;
    });
    row.addEventListener('dragover', (e) => {
      if (!dragged || dragged === row) return;
      e.preventDefault();
      const box = row.getBoundingClientRect();
      const below = e.clientY > box.top + box.height / 2;
      tbody.insertBefore(dragged, below ? row.nextSibling : row);
    });
    row.querySelector('[data-remove]')?.addEventListener('click', () => {
      row.remove();
      refresh();
    });
  }

  [...tbody.children].forEach(wire);
  refresh();

  addBtn?.addEventListener('click', () => {
    const id = picker.value;
    if (!id) return;
    const name = picker.options[picker.selectedIndex].dataset.name;
    const html = tpl.innerHTML
      .replaceAll('__KEY__', 'm:' + id)
      .replaceAll('__MACRO__', id)
      .replaceAll('__NAME__', name);
    const holder = document.createElement('tbody');
    holder.innerHTML = html.trim();
    const row = holder.firstElementChild;
    tbody.appendChild(row);
    wire(row);
    picker.value = '';
    refresh();
  });
})();
