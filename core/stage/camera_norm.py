"""Норма камеры: как этот вид обычно выглядит днём — чтобы отличать помеху от сцены.

Одиночный кадр не отличает окно фасада от капли на стекле, гладкую стену от тумана,
свежий бетон на арматуре от грязи на объективе. У стационарной камеры есть сравнение
надёжнее — её собственные чистые дневные кадры. Норма — медиана «текстуры» (энергии
лапласиана на ширине 320) по последним `keep` чистым дневным кадрам, взятым не чаще
одного в `add_every_min` минут съёмки (чтобы норма покрывала несколько часов и разную
погоду, а не десять минут подряд).

Сравнение кадра с нормой (`compare`):

- **общий множитель g** — медиана отношения текстуры кадра к норме по местам, где
  норма текстурная. Облака и солнце двигают его примерно в 0.5–1.6 раза; туман,
  дождевая пелена, запотевший колпак, солнце в объективе — ниже 0.5 (на камере
  8-й очереди ЖК «Cityzen» g < 0.5 — это ежедневная засветка в 17:59 и утренний туман);
- **пелена** — насколько поднялся «тёмный канал» (He et al., 2009) нижних 2/3 кадра
  относительно нормы: туман и вода поднимают самые тёмные места, контровой свет — нет;
- **локальные провалы** — места, где норма текстурная, а кадр нет: и относительно g,
  и по абсолютной мере, и хуже «худшего нормального» уровня места (квантиль по кадрам
  нормы — дальний горизонт в утренней дымке и стрела крана, которая то есть, то нет,
  провалом не считаются). Провал считается помехой на объективе, если под ним видна та
  же сцена, только размытая (низкие частоты совпадают с нормой), тёмные места в нём
  подняты (вода рассеивает свет) и он не держится на одном месте несколько кадров
  подряд (свежий бетон на арматуре — это стройка, а не грязь). Окна фасада и белые
  бытовки в норме есть, поэтому провалом не считаются — главный источник ложных
  «капель» одиночной эвристики. Провалы под свежим снегом (место стало белым и
  ненасыщенным) не считаются: снег ложится на площадку и со временем уходит в норму;
- **сдвиг** — фазовая корреляция кадра и нормы: камеру повернуло ветром или задели.

Норма учится только на чистых дневных кадрах (`observe`). Если кадр «не похож» на
норму (сдвиг, провалы) дольше `rebase_after_h` часов съёмки подряд — вид изменился
надолго (камеру перевесили, площадка сильно поменялась), и норма собирается заново
с текущих кадров: иначе камера навсегда осталась бы «испорченной». Кадр раньше
последнего учтённого более чем на `rewind_h` часов (переанализ архива, догрузка
старых снимков) сбрасывает норму — она строится заново в порядке съёмки.

Ограничения (docs/limitations.md): ночью и в сумерках норма не сравнивается; капли на
гладком небе не видны — там нечему размываться; капля, простоявшая на одном месте
дольше `persist_min_minutes`, принимается за изменение сцены; грязь, пролежавшая на
объективе дольше `rebase_after_h`, становится нормой.
"""
from __future__ import annotations

import datetime as dt
import io
import json
from dataclasses import dataclass, field

import cv2
import numpy as np

WIDTH = 320


@dataclass
class NormConfig:
    keep: int = 9                   # сколько чистых кадров в норме
    min_frames: int = 3             # раньше сравнивать не с чем
    add_every_min: float = 50.0     # не чаще одного кадра в норму за столько минут съёмки
    min_brightness: float = 55.0    # темнее — не учим и не сравниваем (сумерки, ночь)
    texture_quantile: float = 0.4   # «текстурные» места нормы — выше этого квантиля энергии
    texture_floor: float = 2.0      # и не ниже этой энергии (гладкое небо не сравниваем)
    loss_ratio: float = 0.3         # отношение к норме (и к норме / g) ниже — провал
    low_quantile: float = 0.2       # «худший нормальный» уровень текстуры места (квантиль по кадрам нормы)
    loss_ratio_low: float = 0.45    # … и кадр хуже этого уровня во столько раз
    min_blob_frac: float = 0.0015   # провал меньше этой доли кадра — шум
    scene_lf: float = 0.45          # другое содержимое на низких частотах — сцена сменилась, не капля
    lens_min_veil: float = 0.0      # тёмные места провала не поднялись — не вода на колпаке
    persist_min_minutes: float = 45.0   # провал на том же месте в прошлых кадрах за столько минут —
    persist_frames: int = 2             # … не меньше чем в стольких — изменение сцены
    persist_overlap: float = 0.4
    recent_keep: int = 3
    rebase_after_h: float = 24.0    # «не похож на норму» дольше — норма собирается заново
    rebase_min_frames: int = 4
    rewind_h: float = 6.0           # кадр раньше последнего на столько — норма сбрасывается
    snow_value: int = 170           # провал под свежим снегом не считается помехой
    snow_saturation: int = 45


@dataclass
class NormCheck:
    """Итог сравнения кадра с нормой — сырые числа, решение принимает quality.report_from_metrics."""
    gain: float                     # общий множитель текстуры кадра к норме
    veil: float                     # подъём тёмного канала нижних 2/3 над нормой, уровни яркости
    loss_frac: float                # доля кадра в провалах «на объективе»
    loss_count: int                 # сколько таких провалов
    loss_max: float                 # самый крупный из них, доля кадра
    shift: float                    # сдвиг кадра относительно нормы, доля ширины
    shift_response: float           # уверенность пика фазовой корреляции
    frames: int                     # из скольких кадров норма
    boxes: list = field(default_factory=list)   # провалы на объективе: [x, y, w, h] в долях кадра (для UI)
    blobs: list = field(default_factory=list)   # все провалы с признаками (отладка и подбор порогов)
    loss_mask: np.ndarray | None = None          # маска всех провалов — для проверки «держится на месте»

    def as_dict(self) -> dict:
        return {"gain": round(self.gain, 3), "veil": round(self.veil, 1), "loss_frac": round(self.loss_frac, 4),
                "loss_count": self.loss_count, "loss_max": round(self.loss_max, 4), "shift": round(self.shift, 4),
                "shift_response": round(self.shift_response, 3), "frames": self.frames,
                "boxes": [[round(v, 3) for v in b] for b in self.boxes[:12]]}


def _small(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w == WIDTH:
        return img_bgr
    return cv2.resize(img_bgr, (WIDTH, max(1, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)


def _energy(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.7)
    return cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (0, 0), 1.5)


def _dark(img: np.ndarray) -> np.ndarray:
    return cv2.erode(img.min(axis=2), np.ones((7, 7), np.uint8))


def _utc(when: dt.datetime | None) -> dt.datetime | None:
    if when is None:
        return None
    return when.replace(tzinfo=dt.timezone.utc) if when.tzinfo is None else when.astimezone(dt.timezone.utc)


class CameraNorm:
    """Норма одной камеры. Не потокобезопасна: кадры камеры обрабатываются по очереди."""

    def __init__(self, config: NormConfig | None = None):
        self.cfg = config or NormConfig()
        self.frames: list[np.ndarray] = []          # BGR 320×h, uint8 — чистые дневные кадры
        self.times: list[dt.datetime | None] = []
        self.shape: tuple[int, int] | None = None   # (h, w) нормы
        self.last_at: dt.datetime | None = None      # последний учтённый кадр (любой)
        self.mismatch_since: dt.datetime | None = None
        self.mismatch_frames = 0
        self.rebased: list[str] = []                # когда норма пересобиралась (журнал)
        self.recent: list[tuple[dt.datetime | None, np.ndarray]] = []   # маски провалов прошлых кадров
        self._cache: dict | None = None

    # ------------------------------------------------------------------ состояние

    @property
    def ready(self) -> bool:
        return len(self.frames) >= self.cfg.min_frames

    def reset(self) -> None:
        self.frames, self.times, self.shape = [], [], None
        self.mismatch_since, self.mismatch_frames = None, 0
        self.recent = []
        self._cache = None

    def _ref(self) -> dict:
        if self._cache is None:
            stack = np.stack(self.frames)
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self.frames]
            energies = np.stack([_energy(g) for g in grays])
            energy = np.median(energies, axis=0).astype(np.float32)
            # «худший нормальный» уровень текстуры места: дальний горизонт в утренней дымке,
            # стрела крана, которая то есть, то нет, — у таких мест он низкий, и провал там
            # засчитывается, только если кадр хуже и его
            low = np.quantile(energies, self.cfg.low_quantile, axis=0).astype(np.float32)
            median = np.median(stack, axis=0).astype(np.uint8)
            thr = max(self.cfg.texture_floor, float(np.quantile(energy, self.cfg.texture_quantile)))
            hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
            dark = _dark(median)
            gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lp = cv2.GaussianBlur(gray, (0, 0), 3)
            self._cache = {
                "energy": energy, "energy_low": low, "gray": gray,
                "lp": (lp - lp.mean()) / max(float(lp.std()), 1.0),
                "textured": energy >= thr,
                "dark": float(np.percentile(dark[median.shape[0] // 3:], 25)),
                "dark_map": dark.astype(np.float32),
                "snowy": (hsv[..., 2] >= self.cfg.snow_value) & (hsv[..., 1] <= self.cfg.snow_saturation),
            }
        return self._cache

    # ------------------------------------------------------------------ сравнение

    def compare(self, img_bgr: np.ndarray, when: dt.datetime | None = None) -> NormCheck | None:
        """Кадр (любого размера, того же вида) → NormCheck; None — норма не готова, кадр другого
        формата или слишком тёмный. Состояние нормы не меняет — учит `observe`."""
        if not self.ready or img_bgr is None or img_bgr.size == 0:
            return None
        small = _small(img_bgr)
        if tuple(small.shape[:2]) != tuple(self.shape or ()):
            return None
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < self.cfg.min_brightness:
            return None
        ref = self._ref()
        cfg = self.cfg
        when = _utc(when)
        h, w = gray.shape

        # сдвиг: фазовая корреляция по краям (устойчиво к освещению)
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        a = cv2.Sobel(ref["gray"], cv2.CV_32F, 1, 1, ksize=3)
        b = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 1, ksize=3)
        (dx, dy), resp = cv2.phaseCorrelate(a * win, b * win)
        shift = float(np.hypot(dx, dy)) / w

        e = _energy(gray)
        tex = ref["textured"]
        eps = 1.0
        ratio = (e + eps) / (ref["energy"] + eps)
        gain = float(np.median(ratio[tex])) if tex.any() else 1.0
        rel = ratio / max(gain, 1e-3)
        # Провал — и относительно общего множителя, и по абсолютной мере: низкое утреннее
        # солнце поднимает текстуру ближнего плана вдвое (g ≈ 2), а дальний горизонт — нет,
        # и одно лишь отношение к g «теряло» городской горизонт каждое ясное утро.
        loss = tex & (np.maximum(ratio, rel) < cfg.loss_ratio) & \
            ((e + eps) / (ref["energy_low"] + eps) < cfg.loss_ratio_low)
        # свежий снег: текстура пропала, но место стало белым и ненасыщенным — это не помеха
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        snow_now = (hsv[..., 2] >= cfg.snow_value) & (hsv[..., 1] <= cfg.snow_saturation)
        loss &= ~(snow_now & ~ref["snowy"])
        loss = cv2.morphologyEx(loss.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        loss = cv2.morphologyEx(loss, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(loss, connectivity=8)
        area = float(h * w)
        lp_now = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
        lp_now = (lp_now - lp_now.mean()) / max(float(lp_now.std()), 1.0)
        dark_diff = _dark(small).astype(np.float32) - ref["dark_map"]
        recent = [(t, m) for t, m in self.recent if m.shape == loss.shape]
        blobs = []
        for i in range(1, n):
            x, y, bw, bh, a_ = (int(v) for v in stats[i])
            if a_ < cfg.min_blob_frac * area:
                continue
            m = labels == i
            lf = float(np.mean(np.abs(lp_now[m] - ref["lp"][m])))
            veil = float(np.mean(dark_diff[m]))
            # держится ли провал на месте: в прошлых сравнённых кадрах на этом месте тоже провал
            seen = [t for t, pm in recent if float(pm[m].mean()) >= cfg.persist_overlap]
            span_ok = (when is not None and seen and all(t is not None for t in seen)
                       and (when - min(seen)) >= dt.timedelta(minutes=cfg.persist_min_minutes))
            persistent = len(seen) >= cfg.persist_frames and bool(span_ok)
            blobs.append({
                "box": [x / w, y / h, bw / w, bh / h], "area": a_ / area,
                "ratio": float(np.mean(ratio[m])), "rel": float(np.mean(rel[m])), "lf": lf, "veil": veil,
                "aspect": bw / max(bh, 1),
                # под каплей видна та же сцена, только размытая; другое содержимое на низких
                # частотах — предмет уехал или появился (стрела крана повернулась), это не помеха
                "scene": lf > cfg.scene_lf, "persistent": persistent,
                "lens": lf <= cfg.scene_lf and veil >= cfg.lens_min_veil and not persistent,
            })
        blobs.sort(key=lambda b: -b["area"])
        lens = [b for b in blobs if b["lens"]]
        loss_frac = sum(b["area"] for b in lens)
        loss_max = max((b["area"] for b in lens), default=0.0)

        dark_now = float(np.percentile(_dark(small)[h // 3:], 25))
        return NormCheck(gain=gain, veil=dark_now - ref["dark"], loss_frac=loss_frac, loss_count=len(lens),
                         loss_max=loss_max, shift=shift, shift_response=float(resp),
                         frames=len(self.frames), boxes=[b["box"] for b in lens], blobs=blobs,
                         loss_mask=loss.astype(bool))

    # ------------------------------------------------------------------ обучение

    def observe(self, img_bgr: np.ndarray, clean: bool, when: dt.datetime | None,
                mismatch: bool = False, check: NormCheck | None = None) -> str | None:
        """Учесть кадр после оценки качества.

        clean — кадр дневной и без помех (годится в норму); mismatch — кадр не похож на
        норму по её собственным признакам (сдвиг, провалы) — копится для пересборки;
        check — результат `compare` этого кадра (его провалы запоминаются: провал, который
        держится на месте, — изменение сцены, а не капля).
        → событие для журнала кадра: "reset" | "rebased" | "added" | None.
        """
        if img_bgr is None or img_bgr.size == 0:
            return None
        small = _small(img_bgr)
        when = _utc(when)
        event = None
        if self.shape is not None and tuple(small.shape[:2]) != tuple(self.shape):
            self.reset()                   # другое разрешение или соотношение сторон — другая камера
            event = "reset"
        if when is not None and self.last_at is not None and \
                when < self.last_at - dt.timedelta(hours=self.cfg.rewind_h):
            self.reset()                   # переанализ архива или догрузка старых кадров — строим заново
            self.last_at = None
            event = "reset"
        if when is not None and (self.last_at is None or when > self.last_at):
            self.last_at = when
        if check is not None and check.loss_mask is not None:
            self.recent = (self.recent + [(when, check.loss_mask)])[-self.cfg.recent_keep:]

        if mismatch and self.ready:
            if self.mismatch_since is None:
                self.mismatch_since = when
            self.mismatch_frames += 1
            long = (when is not None and self.mismatch_since is not None
                    and when - self.mismatch_since >= dt.timedelta(hours=self.cfg.rebase_after_h))
            if long and self.mismatch_frames >= self.cfg.rebase_min_frames:
                stamp = when.isoformat(timespec="minutes") if when else "?"
                self.rebased = (self.rebased + [stamp])[-20:]
                self.reset()
                clean, event = True, "rebased"   # текущий кадр — первый кадр новой нормы
        elif clean:
            self.mismatch_since, self.mismatch_frames = None, 0

        if not clean:
            return event
        if float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()) < self.cfg.min_brightness:
            return event
        if self.times and when is not None and self.times[-1] is not None and event != "rebased" and \
                abs((when - self.times[-1]).total_seconds()) < self.cfg.add_every_min * 60:
            return event
        self.shape = tuple(small.shape[:2])
        self.frames.append(small.copy())
        self.times.append(when)
        if len(self.frames) > self.cfg.keep:
            self.frames, self.times = self.frames[-self.cfg.keep:], self.times[-self.cfg.keep:]
        self._cache = None
        return event or "added"

    # ------------------------------------------------------------------ хранение

    def dumps(self) -> bytes:
        iso = lambda t: t.isoformat() if t else None  # noqa: E731
        meta = {
            "v": 1, "times": [iso(t) for t in self.times], "last_at": iso(self.last_at),
            "mismatch_since": iso(self.mismatch_since), "mismatch_frames": self.mismatch_frames,
            "rebased": self.rebased, "recent_times": [iso(t) for t, _ in self.recent],
        }
        frames = np.stack(self.frames) if self.frames else np.zeros((0, 1, 1, 3), np.uint8)
        recent = np.stack([np.packbits(m, axis=None) for _, m in self.recent]) if self.recent \
            else np.zeros((0, 1), np.uint8)
        rshape = np.array(self.recent[0][1].shape if self.recent else (0, 0), np.int32)
        buf = io.BytesIO()
        np.savez_compressed(buf, frames=frames, recent=recent, rshape=rshape,
                            meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
        return buf.getvalue()

    @classmethod
    def loads(cls, data: bytes, config: NormConfig | None = None) -> "CameraNorm":
        z = np.load(io.BytesIO(data))
        meta = json.loads(bytes(z["meta"]).decode())
        parse = lambda s: dt.datetime.fromisoformat(s) if s else None  # noqa: E731
        norm = cls(config)
        frames = z["frames"]
        norm.frames = list(frames) if len(frames) else []
        norm.shape = tuple(frames.shape[1:3]) if len(frames) else None
        norm.times = [parse(t) for t in meta.get("times", [])][:len(norm.frames)]
        norm.last_at = parse(meta.get("last_at"))
        norm.mismatch_since = parse(meta.get("mismatch_since"))
        norm.mismatch_frames = int(meta.get("mismatch_frames", 0))
        norm.rebased = list(meta.get("rebased", []))
        rh, rw = (int(v) for v in z["rshape"])
        if rh and rw:
            for t, packed in zip(meta.get("recent_times", []), z["recent"]):
                mask = np.unpackbits(packed)[: rh * rw].reshape(rh, rw).astype(bool)
                norm.recent.append((parse(t), mask))
        return norm
