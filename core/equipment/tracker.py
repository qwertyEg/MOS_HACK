"""Трекер одной камеры: та же ли это машина, что на прошлом кадре, и двигалась ли она.

Кадры приходят раз в 20–30 минут, поэтому классический трекинг (Калман,
SORT, ByteTrack), рассчитанный на соседние видеокадры, здесь не работает:
за 25 минут самосвал успевает уехать, а экскаватор — сделать сотню циклов,
не сдвинув шасси. Поэтому (docs/ARCHITECTURE.md §5.3):

* Сопоставление — венгерский алгоритм по стоимости
  (1 − IoU) + расстояние центров / диагональ + штраф за класс.
  Путаемые классы (truck ↔ dump_truck) сопоставляются с небольшим штрафом:
  детектор на одной машине может выдавать то одну, то другую метку, а метка
  трека — большинство голосов за последние кадры.
* Второй проход для оставшихся: машина уехала далеко в пределах кадра.
  Сопоставляем по классу и цвету, но только если старое место действительно
  опустело (содержимое там изменилось) — иначе это пропуск детектора, и
  «переклеить» трек на другую машину было бы ошибкой.
* «Двигалась» (moved_since_prev) — смещение центра > max(8 px, 0.15·диагонали)
  ИЛИ изменение формы рамки > 0.15, подтверждённое изменением содержимого
  (рамка детектора «дышит» и без движения), ИЛИ изменение содержимого
  окрестности рамки сверх фона (поза ковша при неподвижном шасси), см.
  appearance.py.
* Сдвиг всей камеры (ветер, перевесили) компенсируется: иначе все машины
  разом «поехали бы». Большой сдвиг — ракурс сменился, движение на этом
  кадре не оцениваем.
* Первый кадр трека и кадр после большого разрыва (> max_gap_min, камера
  молчала) — Activity.UNKNOWN: сравнивать не с чем или сравнение ничего не
  доказывает (за пять часов машина могла уехать и вернуться).
* Смена режима день ↔ ночь (ИК) ломает сравнение содержимого — на таком
  кадре внешность не сравниваем, только геометрию.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from core import taxonomy
from core.contracts import Activity, Detection, FrameInfo, Weather

from . import appearance, boxes
from .config import EquipmentConfig

log = logging.getLogger(__name__)

_INF = float("inf")


@dataclass
class Track:
    track_id: str
    camera_id: str
    bbox: boxes.Box
    first_seen: dt.datetime
    last_seen: dt.datetime
    last_frame_id: int | str | None = None
    votes: deque = field(default_factory=deque)        # (класс, вес) последних кадров
    unit_id: str | None = None
    hits: int = 1
    patch: appearance.Patch | None = None              # окрестность рамки на кадре last_seen
    hist: np.ndarray | None = None                     # цвет — для второго прохода и слияния камер
    site_xy: tuple[float, float] | None = None
    # состояние для моточасов (hours/engine): серия интервалов с движением
    move_streak: int = 0
    pending: list = field(default_factory=list)
    conflicts: int = 0                                 # подряд расхождений с единицей по плану

    def vote(self, cls: str, weight: float, window: int) -> None:
        self.votes.append((cls, float(weight)))
        while len(self.votes) > window:
            self.votes.popleft()

    @property
    def label(self) -> str:
        """Метка большинства (взвешенно по уверенности); при равенстве — самая свежая."""
        score: Counter[str] = Counter()
        last_seen_at: dict[str, int] = {}
        for i, (cls, w) in enumerate(self.votes):
            score[cls] += w
            last_seen_at[cls] = i
        if not score:
            return ""
        return max(score, key=lambda c: (round(score[c], 6), last_seen_at[c]))


@dataclass
class TrackStep:
    """Итог сопоставления одной детекции текущего кадра."""
    detection: Detection
    track: Track
    prev_seen: dt.datetime | None        # когда трек видели до этого кадра; None — трек новый
    prev_frame_id: int | str | None
    judged: bool                         # было ли сравнение с прошлым кадром (не первый кадр, не разрыв)


class CameraTracker:
    """Состояние одной камеры. Вызовы — строго по порядку кадров этой камеры."""

    def __init__(self, camera_id: int | str, config: EquipmentConfig):
        self.camera_id = str(camera_id)
        self.cfg = config
        self.tracks: dict[str, Track] = {}
        self.last_frame_at: dt.datetime | None = None
        self.last_frame_id: int | str | None = None
        self.last_is_night: bool | None = None
        self._small: np.ndarray | None = None
        self._small_scale = 1.0
        self._frame_hw: tuple[int, int] | None = None
        self._counter = 0
        self._trust_content = True

    # ------------------------------------------------------------------

    def update(self, frame: FrameInfo, image_bgr: np.ndarray | None,
               detections: list[Detection]) -> tuple[list[TrackStep], list[str]]:
        t = frame.captured_at
        notes: list[str] = []

        if self.last_frame_at is not None and t <= self.last_frame_at:
            # Кадр из прошлого (догрузили папку задним числом): сравнивать его не
            # с чем честно, а состояние камеры он испортил бы. «Переанализировать»
            # прогоняет кадры заново по порядку.
            notes.append("кадр старше уже обработанного — движение техники по нему не оценивается")
            return [TrackStep(d, _detached_track(self.camera_id, d, t), None, None, False)
                    for d in detections], notes

        gray = appearance.to_gray(image_bgr) if image_bgr is not None else None
        night_switch = self.last_is_night is not None and bool(frame.is_night) != self.last_is_night
        if night_switch:
            notes.append("смена режима день/ночь — содержимое рамок не сравнивается, только положение")
        # Капли на объективе и брак кадра искажают содержимое рамок местами —
        # такое «изменение» не работа. Технику на таком кадре всё равно ведём
        # (работы идут и в дождь), но сравниваем только положение и не
        # запоминаем искажённые кропы как образец для следующего кадра.
        self._trust_content = frame.quality_ok and frame.weather != Weather.RAIN
        if gray is not None and not self._trust_content:
            notes.append("капли/брак кадра — содержимое рамок не сравнивается, только положение")

        shift, reset = self._camera_motion(gray, night_switch, notes)
        self._expire()

        live = list(self.tracks.values())
        det_hist = [appearance.color_hist(image_bgr, d.bbox) if image_bgr is not None else None
                    for d in detections]
        pairs = self._match_geometry(live, detections, shift)
        matched_t = {i for i, _ in pairs}
        matched_d = {j for _, j in pairs}
        if gray is not None and self._trust_content and not night_switch and not reset:
            pairs += self._match_far(live, detections, det_hist, gray, shift,
                                     [i for i in range(len(live)) if i not in matched_t],
                                     [j for j in range(len(detections)) if j not in matched_d])
            matched_d = {j for _, j in pairs}

        # порядок шагов — как у детекций на входе (удобно вызывающему и тестам)
        steps: list[TrackStep | None] = [None] * len(detections)
        for i, j in pairs:
            steps[j] = self._continue(live[i], detections[j], det_hist[j], frame, gray, shift,
                                      reset, night_switch)
        for j, d in enumerate(detections):
            if steps[j] is None:
                steps[j] = self._start(d, det_hist[j], frame, gray)

        self.last_frame_at = t
        self.last_frame_id = frame.frame_id
        self.last_is_night = bool(frame.is_night)
        return steps, notes

    def restore(self, last_at: dt.datetime, detections: list[Detection]) -> None:
        """Восстановить треки по последнему кадру из БД (после рестарта сервиса).

        Кропов прошлого кадра нет, поэтому первое сравнение после рестарта —
        только по геометрии; дальше всё как обычно.
        """
        self.last_frame_at = last_at
        for d in detections:
            tid = d.track_id or self._new_id()
            tr = Track(tid, self.camera_id, tuple(d.bbox), last_at, last_at, unit_id=d.unit_id,
                       site_xy=d.site_xy)
            # Несколько голосов, а не один: одна новая метка не должна сразу
            # перевернуть класс, накопленный до рестарта.
            for _ in range(3):
                tr.vote(d.cls, max(d.conf, 0.5), self.cfg.vote_window)
            self.tracks[tid] = tr
            m = re.search(r"(\d+)$", tid)
            if m:
                self._counter = max(self._counter, int(m.group(1)))

    # ------------------------------------------------------------------

    def _camera_motion(self, gray, night_switch, notes) -> tuple[tuple[float, float], bool]:
        """Сдвиг камеры относительно прошлого кадра и признак «ракурс сменился»."""
        shift, reset = (0.0, 0.0), False
        if gray is None:
            return shift, reset
        small, scale = appearance.small_frame(gray)
        hw = gray.shape[:2]
        if self._frame_hw is not None and hw != self._frame_hw:
            notes.append("изменилось разрешение кадра — движение на этом кадре не оценивается")
            reset = True
        elif self._small is not None and not night_switch:
            est = appearance.camera_shift(self._small, small, scale)
            if est is not None:
                mag = math.hypot(*est)
                if mag > self.cfg.camera_shift_reset_frac * math.hypot(hw[1], hw[0]):
                    notes.append(f"камера сдвинулась на {mag:.0f} px — движение на этом кадре не оценивается, "
                                 "проверьте калибровку и зоны")
                    reset = True
                elif mag >= 1.0:
                    shift = est
        self._small, self._small_scale, self._frame_hw = small, scale, hw
        return shift, reset

    def _expire(self) -> None:
        """Камера снимала, а трека не было дольше max_gap — трек закрыт (машину ведёт слой единиц)."""
        if self.last_frame_at is None:
            return
        limit = dt.timedelta(minutes=self.cfg.max_gap_min)
        for tid in [tid for tid, tr in self.tracks.items() if self.last_frame_at - tr.last_seen > limit]:
            del self.tracks[tid]

    def _match_geometry(self, live: list[Track], dets: list[Detection],
                        shift: tuple[float, float]) -> list[tuple[int, int]]:
        cfg = self.cfg
        if not live or not dets:
            return []
        cost = np.full((len(live), len(dets)), _INF)
        for i, tr in enumerate(live):
            pb = (tr.bbox[0] + shift[0], tr.bbox[1] + shift[1], tr.bbox[2], tr.bbox[3])
            pc, pd = boxes.center(pb), max(boxes.diag(pb), 1.0)
            label = tr.label
            for j, d in enumerate(dets):
                ov = boxes.iou(pb, d.bbox)
                dist = math.dist(pc, boxes.center(d.bbox)) / pd
                if ov == 0.0 and dist > cfg.match_max_center_diag:
                    continue
                if d.cls == label:
                    pen = 0.0
                elif taxonomy.confusable(d.cls, label):
                    pen = cfg.class_penalty_confusable
                else:
                    pen = cfg.class_penalty_other
                cost[i, j] = (1.0 - ov) + dist + pen
        return _assign(cost, cfg.match_max_cost)

    def _match_far(self, live, dets, det_hist, gray, shift, free_t, free_d) -> list[tuple[int, int]]:
        """Второй проход: машина переехала дальше, чем достаёт геометрия."""
        cfg = self.cfg
        if not free_t or not free_d:
            return []
        cost = np.full((len(free_t), len(free_d)), _INF)
        for a, i in enumerate(free_t):
            tr = live[i]
            if tr.patch is None or tr.hist is None:
                continue
            vacated = appearance.appearance_delta(tr.patch, gray, shift, cfg.appearance_min_std)
            # Старое место не опустело — это пропуск детектора, а не переезд.
            if vacated is not None and vacated < cfg.rematch_vacated_min:
                continue
            for b, j in enumerate(free_d):
                d = dets[j]
                if not taxonomy.confusable(d.cls, tr.label):
                    continue
                sim = appearance.hist_similarity(tr.hist, det_hist[j])
                if sim is None or sim < cfg.rematch_min_similarity:
                    continue
                pen = 0.0 if d.cls == tr.label else cfg.class_penalty_confusable
                cost[a, b] = pen + (1.0 - sim)
        return [(free_t[a], free_d[b]) for a, b in _assign(cost, _INF)]

    def _continue(self, tr: Track, d: Detection, hist, frame: FrameInfo, gray, shift,
                  reset: bool, night_switch: bool) -> TrackStep:
        cfg = self.cfg
        t = frame.captured_at
        gap = t - tr.last_seen
        judged = (not reset) and dt.timedelta(0) < gap <= dt.timedelta(minutes=cfg.max_gap_min)

        pb = tr.bbox
        exp_c = (boxes.center(pb)[0] + shift[0], boxes.center(pb)[1] + shift[1])
        disp = math.dist(exp_c, boxes.center(d.bbox))
        shape = (abs(d.bbox[2] - pb[2]) / max(pb[2], 1.0) + abs(d.bbox[3] - pb[3]) / max(pb[3], 1.0))
        can_compare = gray is not None and tr.patch is not None and not night_switch and self._trust_content
        app = (appearance.appearance_delta(tr.patch, gray, shift, cfg.appearance_min_std)
               if judged and can_compare else None)

        moved = False
        if judged:
            disp_moved = disp > max(cfg.move_px_min, cfg.move_diag_frac * boxes.diag(pb))
            if not cfg.shape_needs_appearance or gray is None:
                # Картинки нет совсем (переразбор по сохранённым рамкам) — как в
                # методике: изменение формы рамки само по себе.
                shape_ok = True
            else:
                # Картинка есть: форму подтверждаем содержимым. Если сравнить
                # нельзя (дождь, смена день/ночь, первый кадр после рестарта) —
                # не подтверждаем: «дыхание» рамки детектора работой не считаем.
                shape_ok = app is not None and app >= cfg.appearance_confirm_thr
            shape_moved = shape > cfg.shape_delta_thr and shape_ok
            app_moved = app is not None and app > cfg.appearance_thr
            moved = disp_moved or shape_moved or app_moved

        out = dataclasses.replace(
            d, track_id=tr.track_id, moved_since_prev=moved,
            displacement_px=float(disp) if judged else 0.0,
            bbox_shape_delta=float(shape) if judged else 0.0,
            appearance_delta=float(app) if app is not None else 0.0,
            activity=(Activity.UNKNOWN if not judged else Activity.WORKING if moved else Activity.IDLE),
            extra=dict(d.extra))
        if gap > dt.timedelta(minutes=cfg.max_gap_min):
            out.extra["gap_min"] = round(gap.total_seconds() / 60)

        prev_seen, prev_frame = tr.last_seen, tr.last_frame_id
        self._absorb(tr, d, hist, frame, gray)
        tr.hits += 1
        return TrackStep(out, tr, prev_seen, prev_frame, judged)

    def _start(self, d: Detection, hist, frame: FrameInfo, gray) -> TrackStep:
        tr = Track(self._new_id(), self.camera_id, tuple(d.bbox), frame.captured_at, frame.captured_at)
        self._absorb(tr, d, hist, frame, gray)
        self.tracks[tr.track_id] = tr
        out = dataclasses.replace(d, track_id=tr.track_id, moved_since_prev=False, displacement_px=0.0,
                                  bbox_shape_delta=0.0, appearance_delta=0.0, activity=Activity.UNKNOWN,
                                  extra=dict(d.extra))
        return TrackStep(out, tr, None, None, False)

    def _absorb(self, tr: Track, d: Detection, hist, frame: FrameInfo, gray) -> None:
        cfg = self.cfg
        tr.bbox = tuple(d.bbox)
        tr.last_seen = frame.captured_at
        tr.last_frame_id = frame.frame_id
        tr.vote(d.cls, d.conf, cfg.vote_window)
        for alt_cls, alt_conf in d.extra.get("alt", []):
            # Голос рамки, погашенной межклассовым NMS на этой же машине, —
            # вполсилы: она проиграла, но информативна при устойчивой путанице.
            tr.vote(alt_cls, 0.5 * float(alt_conf), cfg.vote_window)
        tr.patch = appearance.take_patch(gray, d.bbox) if gray is not None and self._trust_content else None
        if hist is not None:
            tr.hist = hist if tr.hist is None else (0.7 * tr.hist + 0.3 * hist).astype(np.float32)

    def _new_id(self) -> str:
        self._counter += 1
        return f"{self.camera_id}-t{self._counter}"


def _detached_track(camera_id: str, d: Detection, t: dt.datetime) -> Track:
    """Трек-заглушка для кадра вне порядка: в состояние камеры не попадает."""
    tr = Track("", camera_id, tuple(d.bbox), t, t)
    tr.vote(d.cls, d.conf, 1)
    return tr


def _assign(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """Оптимальное паросочетание минимальной стоимости; пары дороже max_cost отбрасываются."""
    if cost.size == 0 or not np.isfinite(cost).any():
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:                      # pragma: no cover — scipy есть в окружении сервиса
        return _greedy(cost, max_cost)
    big = 1e9
    rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, big))
    # Венгерский алгоритм назначает всех, кого может, — в том числе по «запрещённой»
    # (бесконечной) стоимости; такие пары выбрасываем явно (inf <= inf — истина).
    return [(int(r), int(c)) for r, c in zip(rows, cols) if np.isfinite(cost[r, c]) and cost[r, c] <= max_cost]


def _greedy(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    pairs, used_r, used_c = [], set(), set()
    for flat in np.argsort(cost, axis=None):
        r, c = np.unravel_index(flat, cost.shape)
        if r in used_r or c in used_c or not np.isfinite(cost[r, c]) or cost[r, c] > max_cost:
            continue
        pairs.append((int(r), int(c)))
        used_r.add(r)
        used_c.add(c)
    return pairs
