/* Выбор папки с кадрами.
 *
 * Обзор идёт по файловой системе сервиса, а не браузера. Это не прихоть:
 * `<input type="file" webkitdirectory>` отдаёт содержимое папки, но не её
 * путь — браузер намеренно его скрывает. А прогону нужен именно путь:
 * кадры читаются с диска на месте, а не загружаются в сервис.
 *
 * Рядом с каждой папкой показано, сколько в ней кадров с распознанной меткой
 * времени. Это и есть ответ на вопрос «та ли это папка» — без него выбор
 * вслепую и ошибка обнаруживается только после подтверждения маски.
 */
(function () {
  const input = document.getElementById('source-uri');
  const openBtn = document.getElementById('source-browse');
  const modal = document.getElementById('fs-modal');
  if (!input || !openBtn || !modal) return;

  const list = document.getElementById('fs-list');
  const here = document.getElementById('fs-here');
  const upBtn = document.getElementById('fs-up');
  const pickBtn = document.getElementById('fs-pick');
  const countLabel = document.getElementById('fs-count');

  let current = '';
  let parent = '';
  let frames = 0;

  async function load(path) {
    list.innerHTML = '<div class="p-4 text-sm text-slate-500">Читаю…</div>';
    const r = await fetch('/api/fs?path=' + encodeURIComponent(path || ''));
    if (!r.ok) {
      list.innerHTML = '<div class="p-4 text-sm text-red-600">Не удалось прочитать папку</div>';
      return;
    }
    const d = await r.json();
    current = d.path;
    parent = d.parent;
    frames = d.frames;

    here.textContent = d.path;
    upBtn.disabled = !d.parent;
    upBtn.classList.toggle('opacity-40', !d.parent);

    countLabel.textContent = frames
      ? `кадров с меткой времени: ${frames}`
      : 'в этой папке кадров с меткой времени нет';
    countLabel.className = frames
      ? 'text-xs text-green-700' : 'text-xs text-slate-500';
    pickBtn.disabled = !frames;
    pickBtn.classList.toggle('opacity-40', !frames);

    if (!d.dirs.length) {
      list.innerHTML = '<div class="p-4 text-sm text-slate-500">Вложенных папок нет</div>';
      return;
    }
    list.innerHTML = '';
    d.dirs.forEach((dir) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'w-full text-left px-3 py-2 text-sm hover:bg-slate-100 ' +
                    'border-b border-slate-100 flex items-center gap-2';
      b.innerHTML = '<span class="text-slate-400">&#128193;</span>' +
                    '<span></span>';
      b.lastElementChild.textContent = dir.name;
      b.onclick = () => load(dir.path);
      list.appendChild(b);
    });
  }

  function show(on) {
    modal.hidden = !on;
    document.body.style.overflow = on ? 'hidden' : '';
  }

  openBtn.addEventListener('click', () => {
    show(true);
    load(input.value.trim());
  });
  upBtn.addEventListener('click', () => parent && load(parent));
  pickBtn.addEventListener('click', () => {
    input.value = current;
    show(false);
  });
  document.getElementById('fs-cancel').addEventListener('click', () => show(false));
  modal.addEventListener('click', (e) => { if (e.target === modal) show(false); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.hidden) show(false);
  });
})();
