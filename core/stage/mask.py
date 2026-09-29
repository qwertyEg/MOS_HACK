"""Динамическая маска фона камеры: что скрыть от модели Б (соседние готовые дома, небо, горизонт).

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
  3 ч 20 мин. Здесь кадры копятся в буфер суток, на смене суток сворачиваются
  медианой — проехавший кран и прошедший человек в каждом кадре в разном месте,
  медиана их не видит, — и медиана суток встаёт в окно.
- **Смена разрешения не роняет поток** (баг D8): кадр любого размера приводится
  к рабочей сетке маски; кадр с другим соотношением сторон пропускается, а если
  такие идут подряд (камеру заменили/перенастроили) — маска начинается заново.
- **Автоматическая начальная маска — «дальний план»** (вторая редакция, после
  проверки на 16 камерах, из них 7 российских). Первая редакция («статичное за
  5 суток у края кадра, кроме эллипса в центре») на демо не построилась ни разу
  (съёмка 1–2 суток), на Эдинбурге закрыла грунт самой площадки (сверху камера
  видит только площадку, а в тихие первые дни грунт статичен), а в стресс-тесте
  «праздники» закрыла 70 % кадра вместе с краном и КАМАЗом. Статичность сама по
  себе не отличает соседний дом от стоящей стройки — отличает положение: соседние
  дома, горизонт и небо — ДАЛЬНИЙ план, над площадкой. Поэтому:
    * срез — медиана кадров за час (не за сутки): маска строится после
      `init_min_slices` срезов на отрезке ≥ `init_min_hours` — на коротком демо в
      первой половине дня, на суточной камере — через 6 суток; медиана среза
      убирает случайные перекрытия (проехавшая машина, стрела крана);
    * небо — ровные светлые клетки, связанные с верхним краем; горизонт — по
      нижней кромке неба; дальний план — выше горизонта + `far_depth` высоты
      кадра (и не ниже `far_max` кадра). Нет неба (камера смотрит на площадку
      сверху, как в Эдинбурге) — фоном может быть только верхняя кромка кадра;
    * фон = небо ∪ статичные клетки дальнего плана, связанные с небом или краем
      кадра. «Статична» — меняется не больше чем в `init_max_changed` доле срезов;
      «меняется» — СКО остатка после подгонки яркости и контраста клетки к
      эталону (медиане срезов): солнце и облака меняют яркость, а не рисунок;
    * техника, стоящая на площадке (точка опоры рамки модели А ниже дальнего
      плана), не закрывается никогда — вместе со стрелой, уходящей в небо. Кран на
      соседней башне опирается в дальнем плане — он фон;
    * поля кадра (letterbox) не фон — там нечего скрывать.
  Ошибки ведут себя так же прилично, как у ручной маски: лишнее закрыли — стройка
  там изменится и откроется сама; мало закрыли — это то же, что работа без маски.
  Кисть оператора (`set_background(..., lock=True)`) важнее автоматики.
- **Техника площадки открывает фон.** После инициализации рамка техники, которая
  стоит на видимой площадке, а стрелой или кузовом заходит в маску, со второго
  кадра стирает маску под собой: новый кран на своей площадке не должен быть
  «фоном». Это сжатие — маска по-прежнему не растёт.
- **История маски** — снимки при каждом изменении (не больше `history_max`):
  интерфейс показывает эволюцию, а кадр, разобранный задним числом, получает
  маску на свой момент (`visible_at`).
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
from typing import Any, Iterable

import cv2
import numpy as np

from core.contracts import Weather

MASK_FORMAT_VERSION = 2
_EPOCH = ""                      # «с начала времён»: ручная маска действует на все кадры камеры


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
    # --- автоматическая начальная маска («дальний план», см. докстринг) ---
    init_slice_minutes: int = 60     # срез — медиана кадров за этот отрезок
    init_min_slices: int = 6         # срезов до автоматической маски…
    init_min_hours: float = 4.0      # …на отрезке не короче
    init_max_slices: int = 24        # окно накопления скользит, пока маску не из чего строить
    init_resid_threshold: float = 6.0   # клетка «изменилась»: СКО остатка после подгонки яркости/контраста
    init_max_changed: float = 0.5    # статична, если менялась не больше чем в этой доле срезов
    sky_texture: float = 7.0         # небо: СКО яркости внутри клетки меньше…
    sky_value: float = 60.0          # …светлее этого и медианы своего среза…
    sky_saturation: float = 60.0     # …и мало насыщено либо голубое
    sky_min_columns: float = 0.2     # небо есть, если видно хотя бы в этой доле столбцов
    horizon_quantile: float = 0.7    # горизонт — квантиль нижней кромки неба по столбцам
    far_depth: float = 0.1           # дальний план — до этой доли высоты ниже горизонта
    far_max: float = 0.7             # и не ниже этой доли высоты кадра: низ кадра — ближний план
    top_band: float = 0.12           # нет неба — фоном может быть только верхняя кромка кадра
    init_max_ratio: float = 0.7      # потолок автоматической маски (у Дениса руками ~0.75)
    box_min_conf: float = 0.35       # рамки модели А слабее не учитываются
    box_margin_cells: int = 1        # запас вокруг рамки техники
    box_headroom: float = 0.05       # техника площадки отстаивает клетки не выше горизонта минус эта доля
    site_box_hits: int = 2           # техника площадки стирает маску под рамкой со 2-го кадра
    max_boot_boxes: int = 1500
    max_frames_per_day: int = 24     # буфер суток; дальше прореживаем равномерно
    darken: float = 0.3              # гашение фона для VLM: контекст частично сохраняется
    useful_min_ratio: float = 0.03   # скрыто меньше — честнее отдать полный кадр
    tz_offset_hours: float = 3.0     # границы суток — по Москве (веб-слой ставит пояс объекта)
    aspect_tolerance: float = 0.03
    reinit_after_mismatch: int = 12  # кадров подряд с другим соотношением сторон → камера другая, маска заново
    history_max: int = 48            # снимков маски для эволюции во времени

    @classmethod
    def from_dict(cls, d: dict | None) -> "MaskConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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
    затемнение, контекст частично виден; black — заливка; gray — заливка средним
    серым (для SigLIP чёрное небо похоже на ночь, серое — нет); blur — сильное
    размытие; crop — обрезка по габаритам видимой области без закраски.
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
    elif mode == "gray":
        out[bg] = 127
    elif mode == "blur":
        k = max(21, (w // 15) | 1)
        out[bg] = cv2.GaussianBlur(image_bgr, (k, k), 0)[bg]
    elif mode == "darken":
        out[bg] = (out[bg].astype(np.float32) * darken).astype(np.uint8)
    else:
        raise ValueError(f"неизвестный режим маски: {mode!r} (darken | black | gray | blur | crop | none)")
    return out


def _local_day(when: dt.datetime, offset_h: float) -> dt.date:
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return (when.astimezone(dt.timezone.utc) + dt.timedelta(hours=offset_h)).date()


def _utc(when: dt.datetime) -> dt.datetime:
    return (when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)).astimezone(dt.timezone.utc)


def _box_fields(box: Any) -> tuple[float, float, float, float, str, float] | None:
    """Рамка модели А в любом из видов: Detection (bbox/cls/conf), dict или кортеж (x, y, w, h, cls, conf)."""
    if box is None:
        return None
    if isinstance(box, dict):
        x, y, w, h = box.get("bbox") or (0, 0, 0, 0)
        return float(x), float(y), float(w), float(h), str(box.get("cls", "")), float(box.get("conf", 1.0))
    if hasattr(box, "bbox"):
        x, y, w, h = box.bbox
        return float(x), float(y), float(w), float(h), str(getattr(box, "cls", "")), float(getattr(box, "conf", 1.0))
    t = tuple(box)
    if len(t) < 4:
        return None
    cls = str(t[4]) if len(t) > 4 else ""
    conf = float(t[5]) if len(t) > 5 else 1.0
    return float(t[0]), float(t[1]), float(t[2]), float(t[3]), cls, conf


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
        self.site_hits = np.zeros(self.grid, dtype=np.int16)
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
        self.last_at: str = ""
        # накопление под начальную маску: срезы (медианы за час), цветные — для неба
        self.boot: list[np.ndarray] = []
        self.boot_color: list[np.ndarray] = []
        self.boot_times: list[str] = []
        self.boot_boxes: list[list[float]] = []     # [x0, y0, x1, y1] в клетках
        self.slice_buf: list[np.ndarray] = []
        self.slice_cbuf: list[np.ndarray] = []
        self.slice_key: int | None = None
        self.slice_t0: str = ""
        self.init_info: dict[str, Any] = {}
        self.protect_top: float = 0.0        # выше этой строки (клетки) техника площадки маску не открывает
        # история: снимки маски при изменениях
        self.history: list[dict[str, Any]] = []
        self.history_bits: list[np.ndarray] = []

    def _cells(self, stat: np.ndarray) -> np.ndarray:
        gh, gw = self.grid
        spc = self.config.stat_px_per_cell
        return stat.reshape(stat.shape[:-2] + (gh, spc, gw, spc)).mean(axis=(-3, -1))

    def _patches(self, stat: np.ndarray) -> np.ndarray:
        """(…, SH, SW) → (…, gh, gw, spc²): точки статистической сетки по клеткам."""
        gh, gw = self.grid
        spc = self.config.stat_px_per_cell
        lead = stat.shape[:-2]
        x = stat.reshape(lead + (gh, spc, gw, spc))
        x = np.moveaxis(x, -3, -2)          # (…, gh, gw, spc, spc)
        return x.reshape(lead + (gh, gw, spc * spc)).astype(np.float32)

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

    def _stat_color(self, image_bgr: np.ndarray) -> np.ndarray:
        img = image_bgr if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        sh, sw = self.stat_shape
        return cv2.resize(img[..., :3], (sw, sh), interpolation=cv2.INTER_AREA).astype(np.uint8)

    def _box_cells(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        fh, fw = self.frame_shape
        gh, gw = self.grid
        x, y, w, h = box
        return x / fw * gw, y / fh * gh, (x + w) / fw * gw, (y + h) / fh * gh

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

    @property
    def locked_ratio(self) -> float:
        return float(self.locked.mean())

    def bootstrap(self) -> dict[str, Any]:
        """Сколько накоплено под автоматическую маску — для честной подписи в интерфейсе."""
        cfg = self.config
        hours = 0.0
        if len(self.boot_times) >= 2:
            hours = (dt.datetime.fromisoformat(self.boot_times[-1])
                     - dt.datetime.fromisoformat(self.boot_times[0])).total_seconds() / 3600
        return {"slices": len(self.boot) + (1 if self.slice_buf else 0), "need_slices": cfg.init_min_slices,
                "hours": round(hours, 1), "need_hours": cfg.init_min_hours,
                "slice_minutes": cfg.init_slice_minutes, "frames": self.frames_seen}

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

    # --- история ---

    def _snapshot(self, event: str, since: dt.datetime | str | None) -> None:
        if isinstance(since, dt.datetime):
            since = _utc(since).isoformat()
        entry = {"from": since if since is not None else _EPOCH, "event": event,
                 "ratio": round(float(self.background.mean()), 4)}
        self.history.append(entry)
        self.history_bits.append(np.packbits(self.background.ravel()))
        cfg = self.config
        while len(self.history) > max(3, cfg.history_max):
            # Прореживаем там, где маска менялась меньше всего; первое и последнее — всегда.
            best, bi = None, None
            for i in range(1, len(self.history) - 1):
                if self.history[i]["event"] in ("manual", "auto"):
                    continue
                d = abs(self.history[i]["ratio"] - self.history[i - 1]["ratio"])
                if best is None or d < best:
                    best, bi = d, i
            if bi is None:
                bi = 1
            del self.history[bi]
            del self.history_bits[bi]

    def _unpack(self, i: int) -> np.ndarray:
        n = self.work_shape[0] * self.work_shape[1]
        return np.unpackbits(self.history_bits[i])[:n].reshape(self.work_shape).astype(bool)

    def history_index(self, when: dt.datetime | None) -> int | None:
        """Какой снимок истории действовал в момент when (None — маски тогда ещё не было)."""
        if not self.history:
            return None
        if when is None:
            return len(self.history) - 1
        t = _utc(when).isoformat()
        idx = None
        for i, e in enumerate(self.history):
            if e["from"] == _EPOCH or e["from"] <= t:
                idx = i
        return idx

    def background_at(self, when: dt.datetime | None) -> np.ndarray | None:
        """Фон (рабочее разрешение) на момент when — для кадра, разобранного задним числом."""
        if not self.initialized:
            return None
        if not self.history:
            return self.background.copy()
        i = self.history_index(when)
        if i is None:
            return None
        return self.background.copy() if i == len(self.history) - 1 else self._unpack(i)

    def history_background(self, i: int) -> np.ndarray:
        return self.background.copy() if i == len(self.history) - 1 else self._unpack(i)

    def visible_at(self, when: dt.datetime | None, shape_hw: tuple[int, int] | None = None) -> np.ndarray | None:
        """«Видимое» (True = объект) на момент when в разрешении shape_hw; None — маски тогда не было."""
        bg = self.background_at(when)
        if bg is None:
            return None
        h, w = shape_hw or self.frame_shape
        return cv2.resize((~bg).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    # --- ручная правка ---

    def set_background(self, bitmap: np.ndarray, lock: bool = False,
                       since: dt.datetime | str | None = None) -> None:
        """Маска, нарисованная оператором (любое разрешение, ненулевое = фон).

        lock=True — «неприкосновенная» область: её сжатие не трогает. Это то, что
        Денис назвал разумным следующим шагом: там, где оператор уверен,
        гарантия нужнее эвристики. Ручная маска действует на все кадры камеры
        (since=None) — история автоматики до неё больше не нужна.
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
        self.site_hits[:] = 0
        self._drop_boot()
        self.history, self.history_bits = [], []
        self._snapshot("manual", since)

    # --- поток кадров ---

    def update(self, image_bgr: np.ndarray, captured_at: dt.datetime,
               weather: Weather | None = None, boxes: Iterable[Any] | None = None) -> MaskUpdate:
        """Учесть дневной пригодный кадр. boxes — рамки модели А этого кадра (Detection, dict или
        (x, y, w, h, cls, conf) в пикселях кадра): технику площадки маска не закрывает.

        Срез часа → накопление начальной маски; смена суток → медиана дня в окно и сжатие."""
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
        if (h, w) != self.frame_shape:
            scale = (fw / w, fh / h)             # рамки — в пикселях присланного кадра
            boxes = [(b[0] * scale[0], b[1] * scale[1], b[2] * scale[0], b[3] * scale[1], b[4], b[5])
                     for b in (_box_fields(x) for x in (boxes or [])) if b is not None]

        day = _local_day(captured_at, self.config.tz_offset_hours)
        if self.buffer_day is not None and day < self.buffer_day:
            self.late_frames += 1
            res.skipped = "кадр старше текущих суток — окно маски назад не пересчитывается"
            return res
        if self.buffer_day is not None and day != self.buffer_day:
            self._close_day(res, captured_at)
        self.buffer_day = day

        stat = self._stat(image_bgr)
        if self.buffer_seen % self.buffer_stride == 0:
            self.buffer.append(stat)
            if len(self.buffer) >= self.config.max_frames_per_day:
                # Прореживаем равномерно: медиана должна видеть весь день, а не только утро.
                self.buffer = self.buffer[::2]
                self.buffer_stride *= 2
        snowy = (weather is Weather.SNOW) if weather is not None else _looks_snowy(image_bgr)
        self.buffer_snow += int(snowy)
        self.buffer_seen += 1
        self.frames_seen += 1
        self.last_at = _utc(captured_at).isoformat()

        parsed = [b for b in (_box_fields(x) for x in (boxes or [])) if b is not None
                  and b[5] >= self.config.box_min_conf and b[2] > 0 and b[3] > 0]
        if not self.initialized:
            self._feed_boot(stat, image_bgr, captured_at, parsed, res)
        elif parsed:
            res.erased_cells += self._site_equipment(parsed, captured_at)
        return res

    def _close_day(self, res: MaskUpdate, now: dt.datetime) -> None:
        if not self.buffer:
            return
        cfg = self.config
        median = np.median(np.stack(self.buffer), axis=0).astype(np.uint8)
        self.ring.append(median)
        self.ring_days.append(self.buffer_day.isoformat() if self.buffer_day else "")
        self.ring_snow.append(self.buffer_snow * 2 > self.buffer_seen)
        keep = cfg.window_days
        self.ring, self.ring_days, self.ring_snow = self.ring[-keep:], self.ring_days[-keep:], self.ring_snow[-keep:]
        self.buffer, self.buffer_seen, self.buffer_stride, self.buffer_snow = [], 0, 1, 0
        res.day_closed = True
        if self.initialized and len(self.ring) >= cfg.window_days:
            erased, res.frozen = self._shrink()
            if erased:
                res.erased_cells += erased
                self._snapshot("shrink", now)

    # --- автоматическая начальная маска ---

    def _drop_boot(self) -> None:
        self.boot, self.boot_color, self.boot_times, self.boot_boxes = [], [], [], []
        self.slice_buf, self.slice_cbuf, self.slice_key, self.slice_t0 = [], [], None, ""

    def _feed_boot(self, stat: np.ndarray, image_bgr: np.ndarray, when: dt.datetime,
                   boxes: list[tuple], res: MaskUpdate) -> None:
        cfg = self.config
        key = int(_utc(when).timestamp() // (cfg.init_slice_minutes * 60))
        if self.slice_key is not None and key != self.slice_key:
            self._close_slice()
            if self._boot_ready():
                self._auto_init(when)
                res.initialized_now = True
                if boxes:                                    # рамки этого кадра — уже в работе
                    res.erased_cells += self._site_equipment(boxes, when)
                return
        if self.slice_key != key:
            self.slice_key, self.slice_t0 = key, _utc(when).isoformat()
        self.slice_buf.append(stat)
        self.slice_cbuf.append(self._stat_color(image_bgr))
        for x, y, w, h, cls, conf in boxes:
            self.boot_boxes.append([round(v, 3) for v in self._box_cells((x, y, w, h))])
        if len(self.boot_boxes) > cfg.max_boot_boxes:
            self.boot_boxes = self.boot_boxes[-cfg.max_boot_boxes:]

    def _close_slice(self) -> None:
        if not self.slice_buf:
            return
        cfg = self.config
        self.boot.append(np.median(np.stack(self.slice_buf), axis=0).astype(np.uint8))
        self.boot_color.append(np.median(np.stack(self.slice_cbuf), axis=0).astype(np.uint8))
        self.boot_times.append(self.slice_t0)
        if len(self.boot) > cfg.init_max_slices:
            self.boot, self.boot_color = self.boot[-cfg.init_max_slices:], self.boot_color[-cfg.init_max_slices:]
            self.boot_times = self.boot_times[-cfg.init_max_slices:]
        self.slice_buf, self.slice_cbuf = [], []

    def _boot_ready(self) -> bool:
        cfg = self.config
        if len(self.boot) < cfg.init_min_slices:
            return False
        span = dt.datetime.fromisoformat(self.boot_times[-1]) - dt.datetime.fromisoformat(self.boot_times[0])
        return span >= dt.timedelta(hours=cfg.init_min_hours)

    def static_cells(self, slices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(доля срезов, где клетка изменилась; эталон клеток). «Изменилась» — СКО остатка
        после подгонки яркости и контраста клетки к эталону (медиане срезов) больше порога:
        солнце, облака и тень меняют яркость клетки, а не её рисунок."""
        x = self._patches(slices)                          # (n, gh, gw, k)
        ref = np.median(x, axis=0)
        rc = ref - ref.mean(-1, keepdims=True)
        rv = (rc ** 2).mean(-1)
        xc = x - x.mean(-1, keepdims=True)
        a = np.clip(np.where(rv > 4.0, (xc * rc).mean(-1) / np.maximum(rv, 1e-6), 1.0), 0.5, 2.0)
        rms = np.sqrt(((xc - a[..., None] * rc) ** 2).mean(-1))
        return (rms > self.config.init_resid_threshold).mean(axis=0), ref

    def sky_cells(self, colors: np.ndarray) -> np.ndarray:
        """Небо: клетки, ровные, светлые для своего среза (не темнее медианы кадра) и мало
        насыщенные либо голубые хотя бы в половине срезов, связанные с верхним краем.
        Яркость — относительно среза: на рассвете небо темнее 100, но всё равно светлее земли."""
        cfg = self.config
        colors = colors if colors.ndim == 4 else colors[None]
        votes = np.zeros(self.grid, dtype=np.float32)
        for color in colors:
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(np.float32)
            g = self._patches(gray)
            v = g.mean(-1)
            hue = self._cells(hsv[..., 0])
            sat = self._cells(hsv[..., 1])
            votes += ((g.std(-1) < cfg.sky_texture) & (v > max(cfg.sky_value, float(np.median(v))))
                      & ((sat < cfg.sky_saturation) | ((hue > 85) & (hue < 135))))
        like = votes >= 0.5 * len(colors)
        _n, lab = cv2.connectedComponents(like.astype(np.uint8), connectivity=4)
        top = [v for v in np.unique(lab[0]) if v != 0]
        return np.isin(lab, top) if top else np.zeros(self.grid, dtype=bool)

    def _auto_init(self, when: dt.datetime) -> None:
        cfg = self.config
        gh, gw = self.grid
        frac, ref = self.static_cells(np.stack(self.boot))
        static = frac <= cfg.init_max_changed
        sky = self.sky_cells(np.stack(self.boot_color))
        rows = np.arange(gh)
        # Горизонт — по нижней кромке неба. Небо, уходящее в нижнюю четверть кадра почти
        # везде, — это не небо (туман, снежное поле сливается с белым небом): без неба.
        cols = sky.any(axis=0)
        horizon = None
        if cols.mean() >= cfg.sky_min_columns:
            lowest = np.array([rows[sky[:, c]].max() for c in range(gw) if cols[c]])
            if (lowest >= 0.75 * gh).mean() <= 0.3:
                horizon = float(np.quantile(lowest, cfg.horizon_quantile)) + 1.0
        if horizon is None:
            sky = np.zeros_like(sky)
            far_rows = max(1.0, round(cfg.top_band * gh))
        else:
            far_rows = min(horizon + cfg.far_depth * gh, cfg.far_max * gh)
        far = (rows + 0.5 < far_rows)[:, None] & np.ones((1, gw), dtype=bool)
        # Выше горизонта (небо, верх соседних башен) технику площадки не отстаиваем: стрела
        # гусеничного крана из котлована иначе заслонила бы рамкой полкадра соседних домов.
        self.protect_top = max(0.0, horizon - cfg.box_headroom * gh) if horizon is not None else 0.0

        # Техника, стоящая на площадке (точка опоры — ниже дальнего плана), не фон.
        protect = np.zeros((gh, gw), dtype=bool)
        m = cfg.box_margin_cells
        top = int(self.protect_top)
        for x0, y0, x1, y1 in self.boot_boxes:
            if y1 < far_rows:
                continue                                    # опора в дальнем плане: соседняя площадка
            protect[max(top, int(y0) - m):min(gh, int(np.ceil(y1)) + m),
                    max(0, int(x0) - m):min(gw, int(np.ceil(x1)) + m)] = True
        letterbox = (ref.mean(-1) < 10) & (ref.std(-1) < 3)
        cand = (static | sky) & far & ~protect & ~letterbox
        seed = cand & (sky | _border(gh, gw))
        bg = _flood(seed, cand)
        if bg.mean() > cfg.init_max_ratio:
            # Статичного слишком много: отдаём нижние строки дальнего плана — ошибка «закрыли
            # мало» дешевле, чем «закрыли стройку». Небо остаётся.
            for r in range(gh - 1, -1, -1):
                if bg.mean() <= cfg.init_max_ratio:
                    break
                bg[r] &= sky[r]
        self.background = self._cells_to_pixels(bg)
        self.initialized = True
        self.source = "auto"
        self.initial_area = int(self.background.sum())
        self.evidence[:] = 0
        self.hot_count[:] = 0
        self.site_hits[:] = 0
        self.init_info = {"slices": len(self.boot), "sky": round(float(sky.mean()), 3),
                          "horizon": round(horizon / gh, 3) if horizon is not None else None,
                          "far": round(float(far_rows) / gh, 3), "static": round(float(static.mean()), 3),
                          "protected": round(float(protect.mean()), 3), "at": _utc(when).isoformat()}
        since = self.boot_times[0] if self.boot_times else when
        self._drop_boot()
        self._snapshot("auto", since)

    # --- техника площадки открывает фон ---

    def _site_equipment(self, boxes: list[tuple], when: dt.datetime) -> int:
        """Рамка техники, стоящей на видимой площадке, стирает маску под собой со 2-го кадра."""
        gh, gw = self.grid
        cfg = self.config
        masked = self._masked_cells()
        hit = np.zeros((gh, gw), dtype=bool)
        for x, y, w, h, cls, conf in boxes:
            x0, y0, x1, y1 = self._box_cells((x, y, w, h))
            fx = min(gw - 1, max(0, int((x0 + x1) / 2)))
            fy = min(gh - 1, max(0, int(np.ceil(y1)) - 1))
            if masked[fy, fx]:
                continue                                    # опора на фоне — соседняя площадка
            hit[max(int(self.protect_top), int(y0)):min(gh, int(np.ceil(y1))),
                max(0, int(x0)):min(gw, int(np.ceil(x1)))] = True
        hit &= masked
        if not hit.any():
            return 0
        self.site_hits = np.where(hit, self.site_hits + 1, self.site_hits).astype(np.int16)
        erase = (self.site_hits >= cfg.site_box_hits) & masked & ~self._masked_cells(self.locked)
        if not erase.any():
            return 0
        self.background &= ~self._cells_to_pixels(erase)
        self.background |= self.locked
        self.site_hits[erase] = 0
        self._snapshot("equipment", when)
        return int(erase.sum())

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
            "last_at": self.last_at,
            "boot_times": self.boot_times,
            "boot_boxes": self.boot_boxes,
            "slice_key": self.slice_key,
            "slice_t0": self.slice_t0,
            "init_info": self.init_info,
            "protect_top": self.protect_top,
            "history": self.history,
        }
        sh, sw = self.stat_shape
        empty = np.zeros((0, sh, sw), dtype=np.uint8)
        empty_c = np.zeros((0, sh, sw, 3), dtype=np.uint8)
        nbits = (self.work_shape[0] * self.work_shape[1] + 7) // 8
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            meta=np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
            background=self.background, locked=self.locked,
            evidence=self.evidence, hot_count=self.hot_count, site_hits=self.site_hits,
            ring=np.stack(self.ring) if self.ring else empty,
            buffer=np.stack(self.buffer) if self.buffer else empty,
            boot=np.stack(self.boot) if self.boot else empty,
            boot_color=np.stack(self.boot_color) if self.boot_color else empty_c,
            slice_buf=np.stack(self.slice_buf) if self.slice_buf else empty,
            slice_cbuf=np.stack(self.slice_cbuf) if self.slice_cbuf else empty_c,
            history_bits=np.stack(self.history_bits) if self.history_bits else np.zeros((0, nbits), np.uint8),
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
            files = set(z.files)
            if "site_hits" in files:
                obj.site_hits = z["site_hits"].astype(np.int16)
            if "boot" in files:
                obj.boot, obj.boot_color = list(z["boot"]), list(z["boot_color"])
                obj.slice_buf, obj.slice_cbuf = list(z["slice_buf"]), list(z["slice_cbuf"])
            if "history_bits" in files:
                obj.history_bits = list(z["history_bits"])
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
        obj.last_at = meta.get("last_at", "")
        obj.boot_times = list(meta.get("boot_times", []))
        obj.boot_boxes = [list(b) for b in meta.get("boot_boxes", [])]
        obj.slice_key = meta.get("slice_key")
        obj.slice_t0 = meta.get("slice_t0", "")
        obj.init_info = dict(meta.get("init_info", {}))
        obj.protect_top = float(meta.get("protect_top", 0.0))
        obj.history = list(meta.get("history", []))
        if len(obj.history) != len(obj.history_bits):
            obj.history, obj.history_bits = [], []
        if obj.initialized and not obj.history:
            obj._snapshot(obj.source if obj.source in ("auto", "manual") else "auto", _EPOCH)   # маска версии 1
        return obj


def _border(gh: int, gw: int) -> np.ndarray:
    b = np.zeros((gh, gw), dtype=bool)
    b[0], b[:, 0], b[:, -1] = True, True, True
    return b


def _flood(seed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Клетки allowed, связанные (8-соседство) с seed."""
    if not seed.any():
        return np.zeros_like(allowed)
    _n, lab = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=8)
    keep = [v for v in np.unique(lab[seed & allowed]) if v != 0]
    return np.isin(lab, keep)


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
