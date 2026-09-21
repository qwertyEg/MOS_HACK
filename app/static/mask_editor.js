/* Редактор маски фона.
 *
 * Оператор закрашивает то, что НЕ относится к стройке: соседние дома, улицу,
 * горизонт. Закрашенное скрывается от модели. Всё незакрашенное считается
 * нашим объектом.
 *
 * Маска отправляется как PNG: белое = фон (скрыть), чёрное = объект.
 * Дальше она может только сжиматься — растущее здание отвоёвывает её обратно.
 */
(function () {
  const photo = document.getElementById('mask-photo');
  const canvas = document.getElementById('mask-canvas');
  if (!photo || !canvas) return;

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const sizeInput = document.getElementById('brush-size');
  const sizeLabel = document.getElementById('brush-size-label');
  const coverLabel = document.getElementById('cover-label');
  const form = document.getElementById('mask-form');
  const payload = document.getElementById('mask-payload');

  let mode = 'paint';
  let drawing = false;
  let last = null;
  const undo = [];

  function fit() {
    canvas.width = photo.naturalWidth;
    canvas.height = photo.naturalHeight;
    canvas.style.width = photo.clientWidth + 'px';
    canvas.style.height = photo.clientHeight + 'px';
    const existing = canvas.dataset.initial;
    if (existing) {
      const img = new Image();
      img.onload = () => { ctx.drawImage(img, 0, 0, canvas.width, canvas.height); updateCover(); };
      img.src = existing;
    } else {
      updateCover();
    }
  }

  if (photo.complete) fit(); else photo.onload = fit;
  window.addEventListener('resize', () => {
    canvas.style.width = photo.clientWidth + 'px';
    canvas.style.height = photo.clientHeight + 'px';
  });

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    const t = e.touches ? e.touches[0] : e;
    return {
      x: (t.clientX - r.left) * (canvas.width / r.width),
      y: (t.clientY - r.top) * (canvas.height / r.height),
    };
  }

  function brushRadius() {
    // Ползунок задаёт размер в экранных пикселях, рисуем в пикселях кадра.
    return (+sizeInput.value) * (canvas.width / canvas.clientWidth);
  }

  function stroke(a, b) {
    ctx.globalCompositeOperation = mode === 'paint' ? 'source-over' : 'destination-out';
    ctx.strokeStyle = 'rgba(220,38,38,0.85)';
    ctx.lineWidth = brushRadius() * 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function pushUndo() {
    undo.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    if (undo.length > 20) undo.shift();
  }

  function start(e) {
    e.preventDefault();
    pushUndo();
    drawing = true;
    last = pos(e);
    stroke(last, last);
  }
  function move(e) {
    if (!drawing) return;
    e.preventDefault();
    const p = pos(e);
    stroke(last, p);
    last = p;
  }
  function end() {
    if (!drawing) return;
    drawing = false;
    updateCover();
  }

  canvas.addEventListener('mousedown', start);
  canvas.addEventListener('mousemove', move);
  window.addEventListener('mouseup', end);
  canvas.addEventListener('touchstart', start, { passive: false });
  canvas.addEventListener('touchmove', move, { passive: false });
  canvas.addEventListener('touchend', end);

  function updateCover() {
    const d = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let on = 0;
    // Шаг 4 пикселя: точность до десятых долей процента не нужна,
    // а полный проход по 1.3 млн пикселей на каждый мазок заметно тормозит.
    for (let i = 3; i < d.length; i += 16) if (d[i] > 10) on++;
    const pct = on / (d.length / 16);
    coverLabel.textContent = (pct * 100).toFixed(1) + '%';
    coverLabel.className = pct < 0.02 || pct > 0.9
      ? 'font-medium text-amber-600' : 'font-medium text-slate-900';
  }

  document.querySelectorAll('[data-mode]').forEach((b) => {
    b.onclick = () => {
      mode = b.dataset.mode;
      document.querySelectorAll('[data-mode]').forEach((x) =>
        x.classList.toggle('ring-2', x === b));
    };
  });

  sizeInput.oninput = () => { sizeLabel.textContent = sizeInput.value; };

  document.getElementById('btn-clear').onclick = () => {
    pushUndo();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    updateCover();
  };

  document.getElementById('btn-undo').onclick = () => {
    const prev = undo.pop();
    if (prev) { ctx.putImageData(prev, 0, 0); updateCover(); }
  };

  // Частый случай: фон — это верхняя полоса кадра (небо, дома за площадкой).
  // Кнопка закрашивает всё выше текущей позиции указателя одним движением.
  let bandY = null;
  const bandBtn = document.getElementById('btn-band');
  bandBtn.onclick = () => {
    bandBtn.classList.toggle('ring-2');
    canvas.style.cursor = bandBtn.classList.contains('ring-2') ? 'crosshair' : 'default';
    bandY = bandBtn.classList.contains('ring-2') ? 0 : null;
  };
  canvas.addEventListener('click', (e) => {
    if (bandY === null) return;
    pushUndo();
    const p = pos(e);
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(220,38,38,0.85)';
    ctx.fillRect(0, 0, canvas.width, p.y);
    bandY = null;
    bandBtn.classList.remove('ring-2');
    canvas.style.cursor = 'default';
    updateCover();
  });

  form.onsubmit = () => {
    // Отдаём чёрно-белую маску: белое = фон. Полупрозрачность кисти
    // на сервере не нужна, поэтому бинаризуем здесь.
    //
    // Размер ограничен: маска всё равно пересчитывается в рабочие 480 px,
    // а раздутый data-URL упирается в лимит разбора формы на сервере.
    const MAXW = 1024;
    const k = Math.min(1, MAXW / canvas.width);
    const w = Math.round(canvas.width * k);
    const h = Math.round(canvas.height * k);

    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    tmp.getContext('2d').drawImage(canvas, 0, 0, w, h);

    const tctx = tmp.getContext('2d');
    const src = tctx.getImageData(0, 0, w, h);
    for (let i = 0; i < src.data.length; i += 4) {
      const v = src.data[i + 3] > 10 ? 255 : 0;
      src.data[i] = src.data[i + 1] = src.data[i + 2] = v;
      src.data[i + 3] = 255;
    }
    tctx.putImageData(src, 0, 0);
    payload.value = tmp.toDataURL('image/png');
    return true;
  };
})();
