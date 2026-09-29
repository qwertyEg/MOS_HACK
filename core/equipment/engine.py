"""EquipmentEngine — состояние техники площадки и обработка очередного кадра.

Конвейер одного кадра (docs/ARCHITECTURE.md §4, модель А):

    детекции → postprocess (одна машина — одна рамка)
             → трекер камеры (та же машина? двигалась?)
             → проекция на план + зоны
             → единицы техники (слияние камер, номер, возврат уехавшей)
             → моточасы (интервалы с движением → ActivityInterval; самосвал под
               погрузкой и миксер у бетононасоса — тоже работа, см. _served)
             → статусы всех единиц площадки

Движок чистый: ни БД, ни HTTP. Веб-слой хранит строки, после рестарта
поднимает состояние через `restore()`, а результаты `process()` пишет в
таблицы detections / equipment_units / activity_intervals.

Потоки. Один движок на площадку. Кадры одной камеры подаются по порядку;
кадры разных камер могут идти из разных потоков — весь `process()` под
одним замком (он дешёвый: миллисекунды против сотен мс у детектора,
который вызывается ДО движка и под замок не попадает).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import math
import re
import threading
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from core import taxonomy
from core.contracts import (Activity, ActivityInterval, CameraGeometry, Detection, FrameInfo, PlanItem,
                            UnitState, UnitStatus, Zone)

from . import fusion, hours as hours_mod, postprocess
from .config import EquipmentConfig
from .status import compute_status
from .tracker import CameraTracker, TrackStep

log = logging.getLogger(__name__)

_RESTORED = "__restored__"      # «мнение» о зоне отстоя, поднятое из БД после рестарта

# Ручная разметка оператора (app/services/annotations.py) приходит полями Detection.extra:
MANUAL_UNIT = "manual_unit"            # рамка — машина, которую назвал оператор; значение — uid её единицы
MANUAL_UNIT_CLS = "manual_unit_cls"    # закреплённый оператором класс этой машины
MANUAL_CLS = "manual_cls"              # класс именно этой рамки задан вручную — классом единицы не перезаписываем
MANUAL_PREFIX = "m"                    # uid ручных единиц; свои движок пишет как u0001


def is_manual(uid: str | None) -> bool:
    """Единица создана правкой оператора (склейка, разделение, смена типа)."""
    return bool(uid) and uid.startswith(MANUAL_PREFIX)


@dataclass
class EquipmentUpdate:
    """Что изменилось после кадра.

    detections — рамки кадра с треком, единицей, зоной, «работает/стоит»;
    units — снимок ВСЕХ единиц площадки (статусы меняются и у тех, кого на
    кадре нет: уехала, встала на стоянку); intervals — новые строки журнала
    моточасов.

    Расширения контракта (ARCHITECTURE §2 разрешает добавлять поля с умолчанием):
    merged — {старый unit_id: новый}: дубль, родившийся на границе зон камер,
    склеен с настоящей единицей; веб-слою переписать unit_id в detections и
    activity_intervals и удалить старую запись equipment_units.
    notes — человекочитаемые события кадра («камера сдвинулась…»), для ленты камеры.
    """
    detections: list[Detection]
    units: list[UnitState]
    intervals: list[ActivityInterval]
    merged: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class _Unit:
    state: UnitState
    ordinal: int                                             # «Экскаватор №ordinal»
    votes: deque = field(default_factory=lambda: deque(maxlen=60))
    hist: np.ndarray | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=500))   # (время, камера, site_xy)
    credited: hours_mod.IntervalSet = field(default_factory=hours_mod.IntervalSet)
    # Камера → стоит ли единица в зоне отстоя по её мнению. Только камеры с
    # размеченными зонами: камера без зон не должна «выпускать» машину со стоянки.
    parking: dict[str, bool] = field(default_factory=dict)
    # До какого момента машину ВИДЕЛИ (стоящей или в работе): последний кадр, сравненный с
    # прошлым без разрыва. У камеры, снимающей раз в сутки, движение не оценивается вовсе,
    # и «на стоянке, ждёт вывоза» (PARKED) по времени ей не из чего вывести — «стоит».
    judged_until: dt.datetime | None = None
    # Последняя рамка по камерам: вернувшуюся «ручную» машину узнаём только на её месте кадра.
    boxes: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)


class EquipmentEngine:
    def __init__(self, config: EquipmentConfig | None = None):
        self.cfg = config or EquipmentConfig()
        self._lock = threading.RLock()
        self._reset()

    def _reset(self) -> None:
        self._cams: dict[str, CameraTracker] = {}
        self._cam_last: dict[str, dt.datetime] = {}
        self._units: dict[str, _Unit] = {}
        self._plates: dict[str, str] = {}
        self._merged: dict[str, str] = {}
        self._counter = 0
        # uid ручной единицы → класс, закреплённый оператором: голоса детектора его не меняют.
        self._locked: dict[str, str] = {}

    # ------------------------------------------------------------------
    # публичный интерфейс
    # ------------------------------------------------------------------

    def restore(self, units: list[UnitState],
                last: dict[int | str, tuple[dt.datetime, list[Detection]]]) -> None:
        """Поднять состояние после рестарта: единицы из equipment_units, по каждой
        камере — время и детекции её последнего обработанного кадра."""
        with self._lock:
            self._reset()
            for us in units:
                st = dataclasses.replace(us, cameras={str(c) for c in us.cameras})
                u = _Unit(st, ordinal=_ordinal_from_label(st.label) or self._next_ordinal(st.cls),
                          credited=hours_mod.IntervalSet(floor=st.last_seen))
                if st.status in (UnitStatus.ACTIVE, UnitStatus.PARKED):
                    u.judged_until = st.last_seen
                for _ in range(5):
                    u.votes.append((st.cls, 1.0))
                if not st.label:
                    st.label = _label(st.cls, u.ordinal)
                # PARKED раньше срока парковки — значит, стояла в зоне отстоя; зоны в
                # UnitState нет, поэтому помним это до первого кадра камеры с зонами.
                still = st.last_seen - (st.last_moved or st.first_seen)
                if st.status == UnitStatus.PARKED and still < dt.timedelta(hours=self.cfg.parked_after_h):
                    u.parking[_RESTORED] = True
                self._units[st.unit_id] = u
                if st.plate:
                    self._plates[st.plate] = st.unit_id
                # Только свои uid (u0001): у ручных единиц своя нумерация, и хвост их
                # ключа не должен сдвигать счётчик движка.
                m = re.fullmatch(r"u(\d+)", st.unit_id)
                if m:
                    self._counter = max(self._counter, int(m.group(1)))
            for cam, (ts, dets) in last.items():
                cam = str(cam)
                for d in dets:
                    if d.unit_id in self._units:
                        self._units[d.unit_id].boxes[cam] = tuple(d.bbox)
                tracker = CameraTracker(cam, self.cfg)
                tracker.restore(ts, [d if d.unit_id in self._units else dataclasses.replace(d, unit_id=None)
                                     for d in dets])
                self._cams[cam] = tracker
                self._cam_last[cam] = ts

    def units(self) -> list[UnitState]:
        with self._lock:
            return self._snapshot_all()

    def set_manual_classes(self, classes: dict[str, str]) -> None:
        """Классы, закреплённые оператором за ручными единицами (uid → класс).

        Веб-слой зовёт после `restore()`: в UnitState закрепления нет, оно живёт
        в таблице ручных правок. Пустой класс — снять закрепление."""
        with self._lock:
            for uid, cls in (classes or {}).items():
                if not cls:
                    self._locked.pop(uid, None)
                    continue
                self._locked[uid] = cls
                u = self._units.get(uid)
                if u is not None and u.state.cls != cls:
                    self._force_class(u, cls)

    def process(self, frame: FrameInfo, image_bgr: np.ndarray | None, detections: list[Detection],
                geometry: CameraGeometry | None, zones: list[Zone],
                plan: list[PlanItem]) -> EquipmentUpdate:
        with self._lock:
            return self._process(frame, image_bgr, detections, geometry, zones or [], plan or [])

    # ------------------------------------------------------------------

    def _process(self, frame, image_bgr, detections, geometry, zones, plan) -> EquipmentUpdate:
        cfg = self.cfg
        cam = str(frame.camera_id)
        t = frame.captured_at
        width = frame.width or (image_bgr.shape[1] if image_bgr is not None else 0)
        height = frame.height or (image_bgr.shape[0] if image_bgr is not None else 0)

        tracker = self._cams.get(cam)
        if tracker is None:
            tracker = self._cams[cam] = CameraTracker(cam, cfg)
        out_of_order = tracker.last_frame_at is not None and t <= tracker.last_frame_at

        dets = postprocess.clean(detections, width, height, cfg)
        steps, notes = tracker.update(frame, image_bgr, dets)

        H = geometry.homography if geometry is not None else None
        calib_size = geometry.image_size if geometry is not None else None
        cam_zones = [z for z in zones if z.camera_id is not None and str(z.camera_id) == cam]
        site_zones = [z for z in zones if z.camera_id is None]
        # Мнение о зоне отстоя: None — у камеры нет зон, судить не может.
        parking: dict[int, bool | None] = {}
        for k, s in enumerate(steps):
            d = s.detection
            xy = fusion.project(H, d.foot, calib_size, (width, height))
            d.site_xy = (round(xy[0], 2), round(xy[1], 2)) if xy else None
            d.zone_id, kind = _zone_for(d, cam_zones, site_zones)
            has_opinion = bool(cam_zones) or (bool(site_zones) and d.site_xy is not None)
            parking[k] = (kind == "parking") if has_opinion else None
            if "plate" in d.extra:
                # Номер — ключ безусловной склейки: нераспознанный («нет», «н/д»)
                # номером не считаем, сырой текст оставляем для отладки.
                raw = d.extra.pop("plate")
                plate = fusion.normalize_plate(raw)
                if plate:
                    d.extra["plate"] = plate
                elif raw:
                    d.extra["plate_raw"] = str(raw)
        if H is None and steps and not out_of_order:
            notes.append("камера не откалибрована: её техника не склеивается с другими камерами")

        if out_of_order:
            return EquipmentUpdate([s.detection for s in steps], self._snapshot_all(), [], {}, notes)

        prev_cam_t = self._cam_last.get(cam)
        self._cam_last[cam] = max(self._cam_last.get(cam, t), t)
        for s in steps:
            s.track.site_xy = s.detection.site_xy

        # Камера молчала дольше max_gap (снимает раз в сутки, ночной перерыв):
        # машины за это время переставили, трекер их не узнаёт по рамкам.
        gap = t - prev_cam_t if prev_cam_t is not None else None
        after_gap = gap if gap is not None and gap > dt.timedelta(minutes=cfg.max_gap_min) else None
        merged = self._assign_units(cam, t, steps, after_gap)

        intervals: list[ActivityInterval] = []
        for k, s in enumerate(steps):
            self._observe(self._units[s.track.unit_id], s, cam, t, parking[k])
        served = self._served(steps)
        for k, s in enumerate(steps):
            intervals += self._credit(self._units[s.track.unit_id], s, frame, plan, served.get(k))

        now = max(self._cam_last.values())
        for u in self._units.values():
            u.state.status = compute_status(u.state, now, self._cam_last, cfg, any(u.parking.values()),
                                            still_until=u.judged_until)

        out = []
        for s in steps:
            u = self._units[s.track.unit_id]
            d = s.detection
            # Класс рамки, заданный оператором («только эта рамка»), — истина для
            # разметки кадра: большинством голосов единицы его не перекрываем.
            if d.cls != u.state.cls and not d.extra.get(MANUAL_CLS):
                d.extra["raw_cls"] = d.cls
                d.cls = u.state.cls
            d.unit_id = u.state.unit_id
            d.extra["unit_label"] = u.state.label
            d.extra["unit_status"] = u.state.status.value
            out.append(d)
        return EquipmentUpdate(out, self._snapshot_all(), intervals, merged, notes)

    # ------------------------------------------------------------------
    # единицы техники
    # ------------------------------------------------------------------

    def _assign_units(self, cam: str, t: dt.datetime, steps: list[TrackStep],
                      after_gap: dt.timedelta | None = None) -> dict[str, str]:
        cfg = self.cfg
        merged: dict[str, str] = {}
        for s in steps:                                  # след прошлых склеек
            s.track.unit_id = self._resolve(s.track.unit_id)

        # Единицы, которые эта камера видит сейчас (в окне склейки). Трек, который
        # камера давно не видит, живёт до max_gap, но рамки на этом кадре не даёт —
        # иначе его единица навсегда выпадала бы из склейки с этой камерой.
        window = dt.timedelta(minutes=cfg.merge_window_min)
        mine = {tr.unit_id for tr in self._cams[cam].tracks.values()
                if tr.unit_id and abs(tr.last_seen - t) <= window}
        obs: list[fusion.Observation] = []
        for k, s in enumerate(steps):
            obs.append(fusion.Observation(frozenset({cam}), t, s.track.label or s.detection.cls,
                                          s.detection.site_xy, s.track.hist, s.detection.extra.get("plate"),
                                          s.track.unit_id, key=k))
        for uid, u in self._units.items():
            # Единицу, которую ведёт трекер этой же камеры, склеивать с новой
            # рамкой этой камеры нельзя: на одном кадре две рамки — две машины.
            if uid in mine or u.state.status == UnitStatus.DEPARTED:
                continue
            pos, cams = self._position_at(u, t, cam)
            if pos is None and not u.state.plate:
                continue
            obs.append(fusion.Observation(frozenset(cams), t, u.state.cls, pos, u.hist, u.state.plate,
                                          uid, key=uid))

        groups = fusion.cluster(obs, cfg)
        group_of = {i: g for g in groups for i in g}
        taken: set[str] = set()
        for k, s in enumerate(steps):
            tr, d = s.track, s.detection
            cls = tr.label or d.cls
            key = d.extra.get(MANUAL_UNIT)
            if key:
                # Машину назвал оператор (склейка / разделение): его слово важнее
                # склейки камер и возврата уехавших — единица задана ключом правки.
                tr.unit_id = self._manual_unit(str(key), d.cls, d.extra.get(MANUAL_UNIT_CLS), t, taken)
                tr.coloc = None
                taken.add(tr.unit_id)
                continue
            # Кандидаты — через карту склеек: предыдущая рамка этого же кадра
            # могла только что склеить одну из единиц группы с другой.
            candidates = [self._resolve(o.unit_id) for o in sorted(
                (obs[i] for i in group_of[k] if isinstance(obs[i].key, str)),
                key=lambda o: _dist(o.site_xy, d.site_xy))]
            candidates = [c for c in candidates if c in self._units and c != tr.unit_id]
            if not candidates or (tr.coloc and tr.coloc[0] != candidates[0]):
                tr.coloc = None                          # серия «в одной точке» прервалась
            plate = d.extra.get("plate")
            plate_uid = self._resolve(self._plates.get(plate)) if plate else None

            if plate_uid in self._units and plate_uid != tr.unit_id:
                self._switch(tr, plate_uid, t, merged)                  # номер — безусловно
            elif tr.unit_id is None:
                tr.unit_id = ((candidates[0] if candidates else None)
                              or self._revive(cls, cam, d.site_xy, t, busy=taken | mine, box=d.bbox)
                              or (self._resume(tr, cls, cam, d.site_xy, t, taken | mine, box=d.bbox)
                                  if after_gap is not None else None)
                              or self._new_unit(cls, t))
            elif candidates and self._young_single(tr.unit_id, t):
                self._switch(tr, candidates[0], t, merged)              # дубль с границы зон камер
            elif is_manual(tr.unit_id) and (not candidates or is_manual(candidates[0])):
                pass        # ручную единицу сам не склеиваем и не отделяем (две ручные — решил оператор)
            elif (candidates and candidates[0] not in taken and self._young_single(candidates[0], t)
                  and self._disjoint(tr.unit_id, candidates[0], t)):
                # Обратный случай: дубль родился в ДРУГОЙ камере минуту назад (она
                # снимала первой и не нашла нашу единицу в окне склейки), а наш
                # трек старый. Склеиваем дубль в нашу единицу.
                merged[candidates[0]] = tr.unit_id
                self._merge(candidates[0], tr.unit_id)
            elif candidates and self._colocated(tr, candidates[0], t):
                # Две давние единицы разных камер устойчиво стоят в одной точке
                # плана (камеру откалибровали позже, треки перепутались на
                # пересечении) — это одна машина. Оставляем старшую.
                keep, drop = sorted((tr.unit_id, candidates[0]),
                                    key=lambda u: (self._units[u].state.first_seen, u))
                if is_manual(drop):
                    keep, drop = drop, keep                  # остаётся машина, названная оператором
                merged[drop] = keep
                self._merge(drop, keep)
                tr.unit_id = keep
            elif self._drifted(tr, d, t, cam):
                log.info("трек %s отделён от единицы %s: разошлись на плане", tr.track_id, tr.unit_id)
                tr.unit_id = (self._revive(cls, cam, d.site_xy, t, exclude=tr.unit_id, busy=taken | mine, box=d.bbox)
                              or self._new_unit(cls, t))

            if tr.unit_id in taken:
                # Две рамки одного кадра не могут быть одной машиной (след
                # ошибочной прошлой склейки) — вторая получает свою единицу.
                tr.unit_id = self._new_unit(cls, t)
            taken.add(tr.unit_id)
        return merged

    def _switch(self, tr, target: str, t: dt.datetime, merged: dict[str, str]) -> None:
        """Перевести трек на другую единицу; осиротевший молодой дубль склеить с ней."""
        old = tr.unit_id
        if old is not None and old != target and self._young_single(old, t):
            merged[old] = target
            self._merge(old, target)
        tr.unit_id = target

    def _position_at(self, u: _Unit, t: dt.datetime, exclude_cam: str
                     ) -> tuple[tuple[float, float] | None, set[str]]:
        """Где была единица по другим камерам в окне ±merge_window вокруг t."""
        window = dt.timedelta(minutes=self.cfg.merge_window_min)
        latest: dict[str, tuple[dt.datetime, tuple[float, float]]] = {}
        for tt, c, xy in u.history:
            if c != exclude_cam and abs(tt - t) <= window and (c not in latest or tt > latest[c][0]):
                latest[c] = (tt, xy)
        if latest:
            pts = [xy for _, xy in latest.values()]
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)), set(latest)
        # Стоящая машина там же, где её видели последний раз, сколько бы ни прошло.
        if u.state.status != UnitStatus.ACTIVE and u.state.site_xy is not None and u.state.last_seen <= t + window:
            return u.state.site_xy, set(u.state.cameras) - {exclude_cam}
        return None, set(u.state.cameras) - {exclude_cam}

    def _young_single(self, uid: str | None, t: dt.datetime) -> bool:
        """Единица только что родилась одним треком и без номера — её можно безболезненно склеить."""
        u = self._units.get(uid) if uid else None
        if u is None or u.state.plate or is_manual(uid):
            return False
        if t - u.state.first_seen > dt.timedelta(minutes=self.cfg.merge_young_min):
            return False
        owners = sum(1 for tk in self._cams.values() for tr in tk.tracks.values() if tr.unit_id == uid)
        return owners <= 1

    def _owners(self, uid: str, t: dt.datetime) -> set[str]:
        """Камеры, чьи треки вели эту единицу в окне склейки вокруг t. Трек, который
        камера давно не видит (он живёт до max_gap), двух рамок одного кадра не даёт."""
        window = dt.timedelta(minutes=self.cfg.merge_window_min)
        return {tk.camera_id for tk in self._cams.values() for tr in tk.tracks.values()
                if tr.unit_id == uid and abs(tr.last_seen - t) <= window}

    def _disjoint(self, a: str, b: str, t: dt.datetime) -> bool:
        """Единицы ведут разные камеры: склейка не поставит две рамки одной камеры в одну машину."""
        return not (self._owners(a, t) & self._owners(b, t))

    def _colocated(self, tr, cand: str, t: dt.datetime) -> bool:
        """Трек и чужая единица ближе радиуса склейки `merge_confirm` кадров подряд (и камеры разные)."""
        if not self._disjoint(tr.unit_id, cand, t):
            tr.coloc = None
            return False
        n = tr.coloc[1] + 1 if tr.coloc and tr.coloc[0] == cand else 1
        tr.coloc = (cand, n)
        if n >= self.cfg.merge_confirm:
            tr.coloc = None
            return True
        return False

    def _drifted(self, tr, d: Detection, t: dt.datetime, cam: str) -> bool:
        """Трек устойчиво расходится с остальными камерами своей единицы — ошибочная склейка."""
        if tr.unit_id is None or d.site_xy is None or is_manual(tr.unit_id):
            tr.conflicts = 0
            return False
        u = self._units[tr.unit_id]
        window = dt.timedelta(minutes=self.cfg.merge_window_min)
        others = [xy for tt, c, xy in u.history if c != cam and abs(tt - t) <= window]
        if not others:
            tr.conflicts = 0
            return False
        if min(_dist(xy, d.site_xy) for xy in others) > self.cfg.split_radius_m:
            tr.conflicts += 1
        else:
            tr.conflicts = 0
        if tr.conflicts >= self.cfg.split_after:
            tr.conflicts = 0
            return True
        return False

    def _revive(self, cls: str, cam: str, xy, t: dt.datetime, exclude: str | None = None,
                busy: set[str] = frozenset(), box=None) -> str | None:
        """Уехавшая машина того же типа вернулась в ту же камеру/место — тот же unit_id.

        Без номера различить две одинаковые машины нельзя; зато челночные
        самосвалы не плодят «Самосвал №57» за неделю, а число единиц остаётся
        равным числу машин, которые бывают на площадке одновременно.
        """
        cfg = self.cfg
        horizon = dt.timedelta(hours=cfg.revive_within_h)
        gray = cfg.merge_radius_m * cfg.merge_gray_factor
        best = None
        for uid, u in self._units.items():
            st = u.state
            # busy — единицы, уже занятые рамками этого кадра: статус DEPARTED пересчитывается
            # только после кадра, и без этого две вернувшиеся машины получали одну и ту же
            # единицу, а вторая — новую («Самосвал №3» при двух уехавших самосвалах).
            if uid == exclude or uid in busy or st.status != UnitStatus.DEPARTED or st.plate:
                continue
            if not taxonomy.confusable(st.cls, cls) or st.last_seen > t or t - st.last_seen > horizon:
                continue
            if is_manual(uid) and not _same_place(u.boxes.get(cam), box):
                continue            # машину назвал оператор — чужую рамку того же типа ей не отдаём
            near = xy is not None and st.site_xy is not None and _dist(xy, st.site_xy) <= gray
            if cam not in st.cameras and not near:
                continue
            rank = (st.cls == cls, st.last_seen)
            if best is None or rank > best[0]:
                best = (rank, uid)
        return best[1] if best else None

    def _resume(self, tr, cls: str, cam: str, xy, t: dt.datetime, busy: set[str], box=None) -> str | None:
        """Кадр после долгого молчания камеры: машина того же типа, которую эта камера
        уже видела, — та же единица, если сейчас её не ведёт ни один трек.

        `_revive` ждёт статуса DEPARTED, а он ставится только после кадра, на котором
        машины не оказалось. У камеры, снимающей раз в сутки (архив Эдинбурга,
        Канберры) или после ночного перерыва, экскаватор к следующему снимку
        переставили — трекер его не узнаёт, а вчерашняя единица ещё «стоит». Без
        этого каждый снимок заводил новые единицы: 582 «машины» на Эдинбурге при
        парке в десяток. Горизонта нет: без номера машины одного типа не различить, и
        единиц у камеры становится столько, сколько машин она видела одновременно
        (замер на сохранённых детекциях Эдинбурга: горизонт в неделю снимков — 247
        единиц, в месяц — 90, без горизонта — 34, из них башенных кранов 5 на три
        камеры при двух настоящих). Пропуск детектора на неделю единицу не рвёт.
        Движение по такому кадру не оценивается (трекер помечает разрыв), так что
        ошибочное «та же машина» моточасов не приписывает.
        """
        window = dt.timedelta(minutes=self.cfg.merge_window_min)
        gray = self.cfg.merge_radius_m * self.cfg.merge_gray_factor
        best = None
        for uid, u in self._units.items():
            st = u.state
            if uid in busy or st.plate or cam not in st.cameras:
                continue
            if not taxonomy.confusable(st.cls, cls) or st.last_seen > t - window:
                continue
            if is_manual(uid) and not _same_place(u.boxes.get(cam), box):
                continue
            # За ночь машину могли перегнать через всю площадку — место не запрет,
            # а только порядок: сначала тот же класс, потом стоявшая рядом, потом недавняя.
            near = xy is not None and st.site_xy is not None and _dist(xy, st.site_xy) <= gray
            rank = (st.cls == cls, near, st.last_seen)
            if best is None or rank > best[0]:
                best = (rank, uid)
        if best is None:
            return None
        uid = best[1]
        # Устаревший трек этой камеры с той же единицей (его рамка на этом кадре не
        # нашлась) больше её не держит: иначе через кадр две рамки — одна машина.
        tracks = self._cams[cam].tracks
        for tid in [k for k, other in tracks.items()
                    if other is not tr and other.unit_id == uid and other.last_seen < t]:
            del tracks[tid]
        return uid

    def _new_unit(self, cls: str, t: dt.datetime, uid: str | None = None) -> str:
        if uid is None:
            self._counter += 1
            uid = f"u{self._counter:04d}"
        ordinal = self._next_ordinal(cls)
        st = UnitState(unit_id=uid, cls=cls, status=UnitStatus.IDLE, first_seen=t, last_seen=t,
                       last_moved=None, label=_label(cls, ordinal))
        self._units[uid] = _Unit(st, ordinal)
        return uid

    def _manual_unit(self, key: str, cls: str, lock: str | None, t: dt.datetime, taken: set[str]) -> str:
        """Единица, названная оператором: её uid — ключ правки; lock — закреплённый класс."""
        uid = self._resolve(key)
        if uid in taken:
            # Две рамки одного кадра с одним ключом: оператор склеил машину с её же
            # лишней рамкой. Одна машина — одна рамка, вторая остаётся отдельной.
            return self._new_unit(cls, t)
        if lock:
            self._locked[uid] = lock
        if uid not in self._units:
            self._new_unit(self._locked.get(uid) or cls, t, uid=uid)
        u = self._units[uid]
        if uid in self._locked and u.state.cls != self._locked[uid]:
            self._force_class(u, self._locked[uid])
        return uid

    def _force_class(self, u: _Unit, cls: str) -> None:
        u.ordinal = self._next_ordinal(cls)          # до смены класса — чтобы не посчитать саму себя
        u.state.cls = cls
        u.state.label = _label(cls, u.ordinal)

    def _next_ordinal(self, cls: str) -> int:
        return 1 + max((u.ordinal for u in self._units.values() if u.state.cls == cls), default=0)

    def _merge(self, src: str, dst: str) -> None:
        """Склеить единицу src в dst (дубль, родившийся на границе камер)."""
        a, b = self._units.pop(src), self._units[dst]
        b.state.first_seen = min(a.state.first_seen, b.state.first_seen)
        b.state.last_seen = max(a.state.last_seen, b.state.last_seen)
        if a.state.last_moved and (b.state.last_moved is None or a.state.last_moved > b.state.last_moved):
            b.state.last_moved = a.state.last_moved
        b.state.cameras |= a.state.cameras
        b.judged_until = max(filter(None, (a.judged_until, b.judged_until)), default=None)
        # Часы дубля переносим без пересечения с уже засчитанным у b (обе камеры
        # видели одну и ту же работу); часы дубля до рестарта движка — как есть.
        own = sum((e - s).total_seconds() / 3600 for s, e in a.credited)
        b.state.worked_hours += max(0.0, a.state.worked_hours - own)
        for s, e in a.credited:
            b.state.worked_hours += sum((pe - ps).total_seconds() / 3600 for ps, pe in b.credited.add(s, e))
        b.votes.extend(a.votes)
        b.history.extend(a.history)
        for cam, box in a.boxes.items():
            b.boxes.setdefault(cam, box)
        for cam, flag in a.parking.items():
            b.parking.setdefault(cam, flag)
        self._merged[src] = dst
        for tk in self._cams.values():
            for tr in tk.tracks.values():
                if tr.unit_id == src:
                    tr.unit_id = dst

    def _resolve(self, uid: str | None) -> str | None:
        seen = set()
        while uid in self._merged and uid not in seen:
            seen.add(uid)
            uid = self._merged[uid]
        return uid

    def _observe(self, u: _Unit, s: TrackStep, cam: str, t: dt.datetime, in_parking: bool | None) -> None:
        st, d = u.state, s.detection
        st.last_seen = max(st.last_seen, t)
        st.first_seen = min(st.first_seen, t)
        st.cameras.add(cam)
        if s.judged:
            u.judged_until = t if u.judged_until is None else max(u.judged_until, t)
        # last_moved здесь не трогаем: его ставит _credit, когда работа
        # подтверждена серией интервалов, — чтобы статус ACTIVE и полоска
        # моточасов не противоречили друг другу.
        u.votes.append((d.cls, d.conf))
        for alt_cls, alt_conf in d.extra.get("alt", []):
            u.votes.append((alt_cls, 0.5 * float(alt_conf)))
        if s.track.hist is not None:
            u.hist = s.track.hist if u.hist is None else (0.7 * u.hist + 0.3 * s.track.hist).astype(np.float32)
        u.boxes[cam] = tuple(d.bbox)
        if d.site_xy is not None:
            u.history.append((t, cam, d.site_xy))
            pos, _ = self._position_at(u, t, exclude_cam="")
            st.site_xy = (round(pos[0], 2), round(pos[1], 2)) if pos else d.site_xy
        plate = d.extra.get("plate")
        if plate and not st.plate:
            st.plate = plate
            self._plates[plate] = st.unit_id
        if in_parking is not None:
            u.parking.pop(_RESTORED, None)
            u.parking[cam] = in_parking
        self._relabel(u)

    def _relabel(self, u: _Unit) -> None:
        """Класс единицы — большинство голосов всех её камер и кадров (или класс, закреплённый оператором)."""
        locked = self._locked.get(u.state.unit_id)
        if locked:
            if u.state.cls != locked:
                self._force_class(u, locked)
            return
        score: Counter[str] = Counter()
        for cls, w in u.votes:
            score[cls] += w
        best = max(score, key=score.get) if score else u.state.cls
        if best != u.state.cls and score[best] > score.get(u.state.cls, 0.0):
            u.ordinal = self._next_ordinal(best)     # до смены класса — чтобы не посчитать саму себя
            u.state.cls = best
            u.state.label = _label(best, u.ordinal)

    # ------------------------------------------------------------------
    # моточасы
    # ------------------------------------------------------------------

    def _served(self, steps: list[TrackStep]) -> dict[int, str]:
        """Стоящие машины, которые в этом кадре обслуживает работающая: {индекс шага: кто}.

        Самосвал под экскаватором и автобетоносмеситель у бетононасоса стоят, но
        работают: машино-часы КАМАЗа под погрузкой — рабочие (путевой лист, табель).
        По одному движению рамки такой самосвал за двое суток погрузки получал 5 ч
        из 40 (карьер в Кирове) и «сильно отставал». Засчитываем, если рядом (зазор
        меньше половины большей рамки) на том же кадре работает машина-партнёр."""
        units = [self._units[s.track.unit_id] for s in steps]
        out: dict[int, str] = {}
        for k, s in enumerate(steps):
            partners = SERVED_BY.get(units[k].state.cls)
            if not partners or not s.judged or s.detection.moved_since_prev:
                continue
            for j, o in enumerate(steps):
                if j != k and units[j].state.cls in partners and o.judged and o.detection.moved_since_prev \
                        and _near(s.detection.bbox, o.detection.bbox):
                    out[k] = units[j].state.cls
                    break
        return out

    def _credit(self, u: _Unit, s: TrackStep, frame: FrameInfo, plan: list[PlanItem],
                served_by: str | None = None) -> list[ActivityInterval]:
        """Интервал с движением → строка журнала. «Работает» — движение на
        `confirm_moves` интервалах подряд: единичный сдвиг (человек прошёл
        перед машиной, её разок переставили) работой не считается. Стоящая
        машина, которую обслуживает работающая (`served_by`, см. _served), —
        тоже работа: самосвал под погрузкой."""
        cfg = self.cfg
        tr, d = s.track, s.detection
        if served_by and s.judged and not d.moved_since_prev:
            d.activity = Activity.WORKING
            d.extra["served_by"] = served_by
        if not s.judged or not (d.moved_since_prev or served_by) or s.prev_seen is None:
            tr.move_streak = 0
            tr.pending.clear()
            return []
        tr.move_streak += 1
        tr.pending.append((s.prev_seen, frame.captured_at,
                           [f for f in (s.prev_frame_id, frame.frame_id) if f is not None]))
        if tr.move_streak < cfg.confirm_moves:
            return []
        t = frame.captured_at
        u.state.last_moved = t if u.state.last_moved is None else max(u.state.last_moved, t)
        out = []
        for start, end, fids in tr.pending:
            for ps, pe in u.credited.add(*hours_mod.credit_window(start, end, cfg.max_credit_gap_min)):
                h = (pe - ps).total_seconds() / 3600
                day = hours_mod.local_date(ps + (pe - ps) / 2, cfg.timezone)
                out.append(ActivityInterval(unit_id=u.state.unit_id, cls=u.state.cls, start=ps, end=pe,
                                            hours=h, stage_id=hours_mod.stage_for(plan, u.state.cls, day),
                                            frame_ids=list(fids)))
                u.state.worked_hours += h
        tr.pending.clear()
        return out

    def _snapshot_all(self) -> list[UnitState]:
        return [_snapshot(u.state) for _, u in sorted(self._units.items())]


# --------------------------------------------------------------------------


def _snapshot(st: UnitState) -> UnitState:
    # Часы не округляем: сумма округлённых интервалов «уползает» (3 × 0.4167 ≠ 1.25) — округляет UI.
    return dataclasses.replace(st, cameras=set(st.cameras))


def _label(cls: str, ordinal: int) -> str:
    return f"{taxonomy.equipment_name(cls)} №{ordinal}"


def _ordinal_from_label(label: str) -> int | None:
    m = re.search(r"№\s*(\d+)", label or "")
    return int(m.group(1)) if m else None


# Кто кого обслуживает стоя: погрузка самосвала, выгрузка бетона в насос (см. EquipmentEngine._served).
SERVED_BY: dict[str, tuple[str, ...]] = {
    "dump_truck": ("excavator", "backhoe_loader", "wheel_loader"),
    "concrete_mixer": ("concrete_pump",),
}


def _near(a, b) -> bool:
    """Рамки рядом: зазор по обеим осям не больше половины большей стороны рамок."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    gap_x = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
    gap_y = max(0.0, max(ay, by) - min(ay + ah, by + bh))
    reach = 0.5 * max(aw, ah, bw, bh)
    return gap_x <= reach and gap_y <= reach


def _dist(a, b) -> float:
    if a is None or b is None:
        return math.inf
    return math.dist(a, b)


def _same_place(last, box) -> bool:
    """Рамка там же, где машину видели на этой камере в последний раз (в пределах её размера)."""
    if last is None or box is None:
        return False
    reach = 1.5 * max(math.hypot(last[2], last[3]), math.hypot(box[2], box[3]))
    return math.dist((last[0] + last[2] / 2, last[1] + last[3] / 2), (box[0] + box[2] / 2, box[1] + box[3] / 2)) <= reach


def _zone_for(d: Detection, cam_zones: list[Zone], site_zones: list[Zone]) -> tuple[int | None, str | None]:
    """Зона по точке контакта с землёй; при вложенных зонах — самая маленькая (самая конкретная).

    Зоны камеры (пиксели кадра) важнее зон плана площадки (метры, для них
    берём site_xy): их рисовали прямо на этом ракурсе, и площади в px² и м²
    между собой не сравнимы.
    """
    hits = [z for z in cam_zones if _inside(d.foot, z.polygon)]
    if not hits and d.site_xy is not None:
        hits = [z for z in site_zones if _inside(d.site_xy, z.polygon)]
    if not hits:
        return None, None
    z = min(hits, key=lambda z: abs(_area(z.polygon)))
    return z.id, z.kind


def _inside(p: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _area(poly: list[tuple[float, float]]) -> float:
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))
