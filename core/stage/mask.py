"""Динамическая маска фона камеры: что скрыть от модели Б (соседние готовые дома, улица, горизонт).

Порт `app/pipeline/mask.py` Дениса + автоматическая начальная маска. Находки Дениса
сохранены, числа — с его замеров на 319 кадрах Edinburgh Informatics Forum:

**Маска — это ФОН.** Всё остальное — наш объект; модель Б видит `~background`.

**Маска только сжимается** (после инициализации). Здание растёт и заползает на
фон — фона становится меньше; обратное физически невозможно. Направление выбрано
по цене ошибки: маска, которая растёт, в пределе скрывает всё и срывает разбор;
маска, которая сжимается, в пределе не скрывает ничего — откат к полному кадру,
хуже, чем без маски, стать не может.

**Сравнение по яркости, а не по структуре.** HOG дал F1 0.67 против 0.86 у яркости,
отрезки прямых — 0.10: фон в той сцене — деревья и перепаханная земля, структурно
они нестабильны не меньше стройки. На соседних жилых домах структура может
выиграть, но без замера на новых данных менять нельзя.

**Разность медиан половин окна, а не MAD**: стройка — ступенька (небо → стена),
MAD к ступеньке слеп по построению (рост этажа виден в одном окне из трёх).

**Освещение — по опорным клеткам**: общий сдвиг яркости (пасмурно, снег, сезон)
вычитается по самым спокойным из ещё закрашенных клеток, а не по всей маске —
когда здание отвоевало большую часть маски, медиана по ней меряет саму стройку
(полнота падала 0.91 → 0.64). Замер: F1 0.39 → 0.42 и 0.73 → 0.80.

**Счётчик со спадом**: устойчивое изменение (растущая стена) пробивает порог за
3 окна подряд, разовое (машина, облако, мокрый асфальт) откатывается.

**Проверено и отвергнуто Денисом** (не переизобретать): отложенное подтверждение
стирания по обратимости, требование связности растущей области, отбрасывание
одиночных клеток — каждое покупает точность на трудном прогоне ценой удачного.

Что добавлено и исправлено здесь:

- **Окно — по суткам, а не по кадрам** (баг D9): прогон папки у Дениса считал
  каждый кадр «днём», и при съёмке раз в 20 минут окно из 10 «дней» покрывало
  3 ч 20 мин. Здесь кадры копятся в буфер суток (по Москве), на смене суток
  сворачиваются медианой — проехавший кран и прошедший человек в каждом кадре
  в разном месте, медиана их не видит, — и медиана суток встаёт в окно.
- **Смена разрешения не роняет поток** (баг D8): кадр любого размера приводится
  к рабочей сетке маски; кадр с другим соотношением сторон пропускается, а если
  такие идут подряд (камеру заменили/перенастроили) — маска начинается заново.
  У Дениса один кадр 16:9 среди 4:3 навсегда останавливал разбор на `np.stack`.
- **Автоматическая начальная маска.** У Дениса начальную маску рисует оператор
  (кистью в UI — осталось: `set_background`). Если оператор не рисовал, после
  первых `init_days` суток клетки с малой изменчивостью дневных медиан (после
  поправки общего освещения) становятся фоном. Это НЕ «маска растёт от пустой»,
  отвергнутая Денисом: инициализация одноразовая и осторожная, дальше маска
  только сжимается. Ошибки ведут себя так же прилично, как у ручной маски:
  лишнее закрыли — стройка там изменится и откроется сама; мало закрыли — это
  то же, что работа без маски. Защита от худшего случая (закрыть саму стройку):
    * центрально-нижний эллипс кадра не маскируется автоматически никогда:
      камеру ставят смотреть на площадку, и площадка почти всегда там
      (рекомендации по установке камер) — а стройка может неделю стоять без
      видимых изменений, и одной статичности мало, чтобы назвать её фоном;
    * фоном становятся только статичные области, касающиеся края кадра:
      соседние дома, небо, улица примыкают к краю, а статичный вагончик
      посреди площадки — нет;
    * доля автоматической маски ограничена (`init_max_ratio`).
- **Снег уходит в фон со временем.** Пока в окне смешаны снежные и бесснежные
  сутки, стирание заморожено (снег на соседней крыше — не стройка); когда всё
  окно снежное, снег — новая база сравнения и сжатие продолжается.
- **Сериализация** в bytes (npz) — состояние хранится веб-слоем как есть, со
  счётчиками и окном медиан: hot_count терять нельзя, по нему выбираются
  опорные клетки освещения (Денис).

Ночные кадры и кадры с каплями в `update` передавать не надо: ИК-режим даёт
другую статистику яркости и развалит сравнение (Денис). Фильтр — `quality.assess`.
"""
from __future__ import annotations

import datetime as dt
import io
import json
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from core.contracts import Weather

MASK_FORMAT_VERSION = 1


@dataclass
class MaskConfig:
    work_width: int = 480            # маска нужна грубая, на полном разрешении считать незачем
    cell: int = 16                   # сторона клетки на рабочем разрешении
    stat_px_per_cell: int = 4        # медианы считаются на сетке 4×4 точки на клетку — экономно и так же устойчиво
    window_days: int = 10            # база сравнения: половина окна против половины
    change_threshold: float = 35.0   # порог изменения яркости клетки
    lock_windows: int = 3            # окон подряд до стирания клетки
    anchor_frac: float = 0.5         # доля самых спокойных закрашенных клеток под опору освещения
    anchor_min_cells: int = 15
    init_days: int = 5               # суток наблюдения до автоматической начальной маски
    init_static_threshold: float = 10.0  # макс. отклонение дневной медианы клетки (после поправки света)
    init_max_ratio: float = 0.7      # автоматически закрываем не больше этой доли кадра (у Дениса руками ~0.75)
    protect_center: tuple[float, float] = (0.5, 0.68)   # центр защищённого эллипса (доли ширины/высоты)
    protect_axes: tuple[float, float] = (0.3, 0.3)      # полуоси эллипса (доли ширины/высоты)
    require_border_contact: bool = True
    max_frames_per_day: int = 24     # буфер суток; дальше прореживаем равномерно
    darken: float = 0.3              # гашение фона для VLM: контекст частично сохраняется
    useful_min_ratio: float = 0.03   # скрыто меньше — честнее отдать полный кадр
    tz_offset_hours: float = 3.0     # границы суток — по Москве
    aspect_tolerance: float = 0.03
    reinit_after_mismatch: int = 12  # кадров подряд с другим соотношением сторон → камера другая, маска заново

    @classmethod
    def from_dict(cls, d: dict | None) -> "MaskConfig":
        d = dict(d or {})
        out = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("protect_center", "protect_axes"):
            if k in out and out[k] is not None:
                out[k] = tuple(out[k])
        return cls(**out)


@dataclass
class MaskUpdate:
    """Что произошло при update — веб-слою для решения «переспросить модель Б внеочередно»."""
    day_closed: bool = False
    initialized_now: bool = False
    erased_cells: int = 0
    frozen: bool = False             # окно смешанное по снегу — стирание заморожено
    skipped: str = ""                # почему кадр не учтён


def apply_background(image_bgr: np.ndarray, background: np.ndarray, mode: str = "darken",
                     darken: float = 0.3) -> np.ndarray:
    """Погасить фон на кадре. background — bool любого разрешения, True = скрыть.

    Режимы (сравнение — вопрос замера, а не рассуждения, Денис §3.4.7): darken —
    затемнение, контекст частично виден; black — заливка; blur — сильное размытие;
    crop — обрезка по габаритам видимой области без закраски.
    """
    h, w = image_bgr.shape[:2]
    bg = background
    if bg.shape[:2] != (h, w):
        bg = cv2.resize(bg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if mode == "none" or not bg.any():
        return image_bgr.copy()
    if mode == "crop":
        ys, xs = np.where(~bg)
        if len(xs) == 0:
            return image_bgr.copy()
        pad_y, pad_x = int(0.02 * h), int(0.02 * w)
        y0, y1 = max(0, ys.min() - pad_y), min(h, ys.max() + 1 + pad_y)
        x0, x1 = max(0, xs.min() - pad_x), min(w, xs.max() + 1 + pad_x)
        return image_bgr[y0:y1, x0:x1].copy()
    out = image_bgr.copy()
    if mode == "black":
        out[bg] = 0
    elif mode == "blur":
        k = max(21, (w // 15) | 1)
        out[bg] = cv2.GaussianBlur(image_bgr, (k, k), 0)[bg]
    elif mode == "darken":
        out[bg] = (out[bg].astype(np.float32) * darken).astype(np.uint8)
    else:
        raise ValueError(f"неизвестный режим маски: {mode!r} (darken | black | blur | crop | none)")
    return out


def _local_day(when: dt.datetime, offset_h: float) -> dt.date:
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return (when.astimezone(dt.timezone.utc) + dt.timedelta(hours=offset_h)).date()


class DynamicMask:
    """Состояние маски одной камеры. Живёт месяцами; хранится через dumps()/loads()."""

    def __init__(self, frame_shape: tuple[int, int], config: MaskConfig | None = None):
        self.config = config or MaskConfig()
        self._reset_to_shape(frame_shape)

    @classmethod
    def new(cls, shape_hw: tuple[int, int], config: MaskConfig | dict | None = None) -> "DynamicMask":
        cfg = config if isinstance(config, MaskConfig) else MaskConfig.from_dict(config)
        return cls(tuple(int(x) for x in shape_hw[:2]), cfg)

    # --- геометрия ---

    def _reset_to_shape(self, frame_shape: tuple[int, int]) -> None:
        cfg = self.config
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        if fh <= 0 or fw <= 0:
            raise ValueError(f"некорректный размер кадра {frame_shape}")
        self.frame_shape = (fh, fw)
        ww = cfg.work_width - cfg.work_width % cfg.cell
        wh = max(cfg.cell, round(fh * ww / fw))
        wh -= wh % cfg.cell
        self.work_shape = (wh, ww)
        self.grid = (wh // cfg.cell, ww // cfg.cell)
        spc = cfg.stat_px_per_cell
        self.stat_shape = (self.grid[0] * spc, self.grid[1] * spc)
        self.background = np.zeros(self.work_shape, dtype=bool)
        self.locked = np.zeros(self.work_shape, dtype=bool)
        self.evidence = np.zeros(self.grid, dtype=np.int16)
        self.hot_count = np.zeros(self.grid, dtype=np.int32)
        self.initialized = False
        self.source = "none"             # none | auto | manual
        self.initial_area = 0
        self.windows = 0
        self.ring: list[np.ndarray] = []     # дневные медианы на статистической сетке, uint8
        self.ring_days: list[str] = []
        self.ring_snow: list[bool] = []
        self.buffer: list[np.ndarray] = []
        self.buffer_day: dt.date | None = None
        self.buffer_seen = 0
        self.buffer_stride = 1
        self.buffer_snow = 0
        self.mismatch_streak = 0
        self.late_frames = 0
        self.frames_seen = 0

    def _cells(self, stat: np.ndarray) -> np.ndarray:
        gh, gw = self.grid
        spc = self.config.stat_px_per_cell
        return stat.reshape(stat.shape[:-2] + (gh, spc, gw, spc)).mean(axis=(-3, -1))

    def _cells_to_pixels(self, cells: np.ndarray) -> np.ndarray:
        c = self.config.cell
        return np.repeat(np.repeat(cells, c, axis=0), c, axis=1)

    def _masked_cells(self, mask: np.ndarray | None = None) -> np.ndarray:
        gh, gw = self.grid
        c = self.config.cell
        m = self.background if mask is None else mask
        return m.reshape(gh, c, gw, c).mean(axis=(1, 3)) >= 0.5

    def _stat(self, image_bgr: np.ndarray) -> np.ndarray:
        img = image_bgr
        if img.ndim == 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
        sh, sw = self.stat_shape
        return cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.uint8)

    # --- публичные свойства ---

    @property
    def masked_ratio(self) -> float:
        """Какая доля кадра сейчас скрыта."""
        return float(self.background.mean()) if self.initialized else 0.0

    @property
    def retained(self) -> float:
        """Какая доля исходной маски ещё цела. Падает по мере роста здания."""
        return float(self.background.sum()) / self.initial_area if self.initial_area else 0.0

    @property
    def useful(self) -> bool:
        """Если скрывать почти нечего, честнее сказать об этом и отдать модели полный кадр."""
        return self.initialized and self.masked_ratio > self.config.useful_min_ratio

    def visible(self, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
        """Область, которую видит модель Б (True = объект), в разрешении кадра камеры."""
        h, w = shape_hw or self.frame_shape
        if not self.initialized:
            return np.ones((h, w), dtype=bool)
        return cv2.resize((~self.background).astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    def apply(self, image_bgr: np.ndarray, mode: str = "darken") -> np.ndarray:
        """Кадр с погашенным фоном — ровно то, что уйдёт в модель Б (и в превью UI)."""
        if not self.useful:
            return image_bgr.copy()
        return apply_background(image_bgr, self.background, mode, self.config.darken)

    # --- ручная правка ---

    def set_background(self, bitmap: np.ndarray, lock: bool = False) -> None:
        """Маска, нарисованная оператором (любое разрешение, ненулевое = фон).

        lock=True — «неприкосновенная» область: её сжатие не трогает. Это то, что
        Денис назвал разумным следующим шагом: там, где оператор уверен,
        гарантия нужнее эвристики.
        """
        h, w = self.work_shape
        m = cv2.resize((np.asarray(bitmap) > 0).astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        self.background = m
        self.locked = m.copy() if lock else np.zeros_like(m)
        self.initialized = True
        self.source = "manual"
        self.initial_area = int(m.sum())
        self.evidence[:] = 0
        self.hot_count[:] = 0

    # --- поток кадров ---

    def update(self, image_bgr: np.ndarray, captured_at: dt.datetime,
               weather: Weather | None = None) -> MaskUpdate:
        """Учесть дневной пригодный кадр. На смене суток — медиана дня в окно, инициализация, сжатие."""
        res = MaskUpdate()
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            res.skipped = "пустой кадр"
            return res
        h, w = image_bgr.shape[:2]
        fh, fw = self.frame_shape
        if abs((w / h) / (fw / fh) - 1.0) > self.config.aspect_tolerance:
            self.mismatch_streak += 1
            if self.mismatch_streak < self.config.reinit_after_mismatch:
                res.skipped = (f"соотношение сторон кадра {w}×{h} не совпадает с камерой {fw}×{fh} — "
                               "кадр не учтён в маске")
                return res
            # Такие кадры идут подряд — камеру заменили или перенастроили: старая маска о другой сцене.
            self._reset_to_shape((h, w))
            res.skipped = "камера сменила формат кадра — маска начата заново"
        else:
            self.mismatch_streak = 0

        day = _local_day(captured_at, self.config.tz_offset_hours)
        if self.buffer_day is not None and day < self.buffer_day:
            self.late_frames += 1
            res.skipped = "кадр старше текущих суток — окно маски назад не пересчитывается"
            return res
        if self.buffer_day is not None and day != self.buffer_day:
            self._close_day(res)
        self.buffer_day = day

        if self.buffer_seen % self.buffer_stride == 0:
            self.buffer.append(self._stat(image_bgr))
            if len(self.buffer) >= self.config.max_frames_per_day:
                # Прореживаем равномерно: медиана должна видеть весь день, а не только утро.
                self.buffer = self.buffer[::2]
                self.buffer_stride *= 2
        snowy = (weather is Weather.SNOW) if weather is not None else _looks_snowy(image_bgr)
        self.buffer_snow += int(snowy)
        self.buffer_seen += 1
        self.frames_seen += 1
        return res

    def _close_day(self, res: MaskUpdate) -> None:
        if not self.buffer:
            return
        cfg = self.config
        median = np.median(np.stack(self.buffer), axis=0).astype(np.uint8)
        self.ring.append(median)
        self.ring_days.append(self.buffer_day.isoformat() if self.buffer_day else "")
        self.ring_snow.append(self.buffer_snow * 2 > self.buffer_seen)
        keep = max(cfg.window_days, cfg.init_days)
        self.ring, self.ring_days, self.ring_snow = self.ring[-keep:], self.ring_days[-keep:], self.ring_snow[-keep:]
        self.buffer, self.buffer_seen, self.buffer_stride, self.buffer_snow = [], 0, 1, 0
        res.day_closed = True
        if not self.initialized:
            if len(self.ring) >= cfg.init_days:
                self._auto_init()
                res.initialized_now = True
        elif len(self.ring) >= cfg.window_days:
            res.erased_cells, res.frozen = self._shrink()

    # --- автоматическая начальная маска ---

    def _site_distance(self) -> np.ndarray:
        """Эллиптическое расстояние клетки от центра площадки: ≤ 1 — внутри защищённого эллипса."""
        gh, gw = self.grid
        cx, cy = self.config.protect_center
        ax, ay = self.config.protect_axes
        ys = (np.arange(gh) + 0.5) / gh
        xs = (np.arange(gw) + 0.5) / gw
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        return np.sqrt(((xx - cx) / max(ax, 1e-6)) ** 2 + ((yy - cy) / max(ay, 1e-6)) ** 2)

    def _protected_cells(self) -> np.ndarray:
        return self._site_distance() <= 1.0

    def _auto_init(self) -> None:
        cfg = self.config
        cells = self._cells(np.stack(self.ring[-cfg.init_days:]).astype(np.float32))   # (n, gh, gw)
        med = np.median(cells, axis=0)
        # Общий свет дня (пасмурно/солнце) не должен делать неподвижное «изменчивым».
        offsets = np.median(cells - med, axis=(1, 2), keepdims=True)
        dev = np.abs(cells - offsets - med).max(axis=0)
        static = (dev < cfg.init_static_threshold) & ~self._protected_cells()
        if cfg.require_border_contact and static.any():
            n, labels = cv2.connectedComponents(static.astype(np.uint8), connectivity=8)
            border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
            border.discard(0)
            static = np.isin(labels, list(border))
        if static.mean() > cfg.init_max_ratio:
            # Статичного слишком много (выходные, простой): отдаём обратно клетки, ближайшие к
            # площадке, — ошибка «закрыли мало» дешевле, чем «закрыли стройку».
            dist = self._site_distance()
            limit = np.quantile(dist[static], 1.0 - cfg.init_max_ratio / static.mean())
            static &= dist >= limit
        self.background = self._cells_to_pixels(static)
        self.initialized = True
        self.source = "auto"
        self.initial_area = int(self.background.sum())
        self.evidence[:] = 0
        self.hot_count[:] = 0

    # --- сжатие (алгоритм Дениса) ---

    def _anchor(self, masked_cells: np.ndarray):
        ys, xs = np.where(masked_cells)
        if len(ys) < self.config.anchor_min_cells:
            return None
        order = np.argsort(self.hot_count[ys, xs], kind="stable")
        keep = max(self.config.anchor_min_cells, int(len(order) * self.config.anchor_frac))
        return ys[order[:keep]], xs[order[:keep]]

    def _shrink(self) -> tuple[int, bool]:
        """Один шаг: окно дневных медиан → сжатая маска. background после вызова не больше, чем был."""
        cfg = self.config
        snow = self.ring_snow[-cfg.window_days:]
        if any(snow) and not all(snow):
            return 0, True
        cells = self._cells(np.stack(self.ring[-cfg.window_days:]).astype(np.float32))
        half = max(1, len(cells) // 2)
        now = np.median(cells[-half:], axis=0)
        before = np.median(cells[:half], axis=0)
        masked = self._masked_cells()
        anchor = self._anchor(masked)
        offset = float(np.median((now - before)[anchor])) if anchor else 0.0
        hot = np.abs(now - before - offset) > cfg.change_threshold
        self.hot_count += hot.astype(np.int32)
        self.evidence = np.where(hot, self.evidence + 1, np.maximum(self.evidence - 1, 0)).astype(np.int16)
        erase_cells = (self.evidence >= cfg.lock_windows) & masked & ~self._masked_cells(self.locked)
        if erase_cells.any():
            self.background &= ~self._cells_to_pixels(erase_cells)
            # Неприкосновенное не трогаем и на краях клеток; locked ⊆ исходной маски, так что расти ей некуда.
            self.background |= self.locked
        self.windows += 1
        return int(erase_cells.sum()), False

    # --- сериализация ---

    def dumps(self) -> bytes:
        cfg = self.config
        meta = {
            "version": MASK_FORMAT_VERSION,
            "config": asdict(cfg),
            "frame_shape": list(self.frame_shape),
            "initialized": self.initialized,
            "source": self.source,
            "initial_area": self.initial_area,
            "windows": self.windows,
            "ring_days": self.ring_days,
            "ring_snow": self.ring_snow,
            "buffer_day": self.buffer_day.isoformat() if self.buffer_day else None,
            "buffer_seen": self.buffer_seen,
            "buffer_stride": self.buffer_stride,
            "buffer_snow": self.buffer_snow,
            "mismatch_streak": self.mismatch_streak,
            "late_frames": self.late_frames,
            "frames_seen": self.frames_seen,
        }
        sh, sw = self.stat_shape
        empty = np.zeros((0, sh, sw), dtype=np.uint8)
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            meta=np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
            background=self.background, locked=self.locked,
            evidence=self.evidence, hot_count=self.hot_count,
            ring=np.stack(self.ring) if self.ring else empty,
            buffer=np.stack(self.buffer) if self.buffer else empty,
        )
        return buf.getvalue()

    @classmethod
    def loads(cls, data: bytes, config: MaskConfig | None = None) -> "DynamicMask":
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            meta: dict[str, Any] = json.loads(bytes(z["meta"]).decode("utf-8"))
            cfg = config or MaskConfig.from_dict(meta.get("config"))
            obj = cls(tuple(meta["frame_shape"]), cfg)
            if z["background"].shape != obj.work_shape:
                # Сетка поменялась (другой конфиг) — перенести маску можно, счётчики — нет.
                obj.set_background(z["background"])
                obj.source = meta.get("source", "manual")
                return obj
            obj.background = z["background"].astype(bool)
            obj.locked = z["locked"].astype(bool)
            obj.evidence = z["evidence"].astype(np.int16)
            obj.hot_count = z["hot_count"].astype(np.int32)
            obj.ring = list(z["ring"])
            obj.buffer = list(z["buffer"])
        obj.initialized = bool(meta["initialized"])
        obj.source = meta.get("source", "none")
        obj.initial_area = int(meta.get("initial_area", 0))
        obj.windows = int(meta.get("windows", 0))
        obj.ring_days = list(meta.get("ring_days", []))
        obj.ring_snow = [bool(x) for x in meta.get("ring_snow", [])]
        bd = meta.get("buffer_day")
        obj.buffer_day = dt.date.fromisoformat(bd) if bd else None
        obj.buffer_seen = int(meta.get("buffer_seen", 0))
        obj.buffer_stride = int(meta.get("buffer_stride", 1))
        obj.buffer_snow = int(meta.get("buffer_snow", 0))
        obj.mismatch_streak = int(meta.get("mismatch_streak", 0))
        obj.late_frames = int(meta.get("late_frames", 0))
        obj.frames_seen = int(meta.get("frames_seen", 0))
        return obj


def _looks_snowy(image_bgr: np.ndarray) -> bool:
    from core.stage.quality import looks_snowy  # лениво: quality тянет только cv2, но держим модули независимыми
    return looks_snowy(image_bgr)


def masked_for_model(image_bgr: np.ndarray, context: dict | None) -> tuple[np.ndarray, bool]:
    """Кадр для классификатора с учётом маски из context (DynamicMask или bool-массив «видимое»).

    Возвращает (кадр, применена ли маска) — второе нужно, чтобы сказать VLM, что
    затемнённое — это фон.
    """
    if not context:
        return image_bgr, False
    mask = context.get("mask")
    mode = context.get("mask_mode", "darken")
    if mask is None or mode == "none":
        return image_bgr, False
    if isinstance(mask, DynamicMask):
        if not mask.useful:
            return image_bgr, False
        return mask.apply(image_bgr, mode), True
    visible = np.asarray(mask).astype(bool)
    background = ~visible
    ratio = float(background.mean()) if background.size else 0.0
    if ratio <= MaskConfig.useful_min_ratio or ratio >= 0.999:
        return image_bgr, False
    return apply_background(image_bgr, background, mode, context.get("mask_darken", MaskConfig.darken)), True


def detect_shift(reference: np.ndarray, current: np.ndarray, min_inlier_ratio: float = 0.25) -> tuple[bool, float]:
    """Сменился ли ракурс камеры (порт Дениса): (сменился, доля совпавших точек ORB+RANSAC).

    Эталон надо держать свежим (кадр прошлых суток), а не первый кадр стройки:
    «пустырь против готового здания» не совпадёт и у намертво прибитой камеры.
    Переносить маску гомографией нельзя — новая точка съёмки может показывать
    другой фон; при сдвиге честнее начать маску заново (веб-слой: CAMERA_ISSUE).
    """
    def prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        h, w = g.shape[:2]
        return cv2.resize(g, (480, max(1, round(h * 480 / w))), interpolation=cv2.INTER_AREA)

    ref, cur = prep(reference), prep(current)
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(cur, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return True, 0.0
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2), key=lambda m: m.distance)
    if len(matches) < 12:
        return True, 0.0
    good = matches[:max(12, len(matches) // 2)]
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    hmat, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if hmat is None or inliers is None:
        return True, 0.0
    ratio = float(inliers.sum()) / len(good)
    return ratio < min_inlier_ratio, ratio
