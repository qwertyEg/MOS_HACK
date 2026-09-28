"""AnalyticsContext — снимок всего, что известно о площадке на момент пересчёта.

Веб-слой собирает его из БД (план, хронология этапов модели Б, единицы техники и
последние кадры с детекциями модели А, журнал моточасов, зоны) и отдаёт в
rules.evaluate / report.build. Ядро не ходит в БД: всё нужное лежит здесь.

Здесь же — общие для правил вычисления: наблюдения техники с зоной и статусом,
«рабочее время» площадки, текущие этапы. Правила тогда остаются короткими и
одинаково понимают, что такое «окно 2 ч рабочего времени» или «этап идёт».
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any

from core.contracts import (
    Activity, ActivityInterval, Detection, FrameInfo, HoursBalance, PlanItem,
    StageStatus, StageTimeline, UnitState, UnitStatus, Zone,
)

try:  # zoneinfo есть в stdlib, но в «тонких» докер-образах может не быть базы tzdata
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

MSK = dt.timezone(dt.timedelta(hours=3), "MSK")


@dataclass
class AnalyticsConfig:
    """Пороги аналитики. Все — в настройках (ctx.config), здесь только умолчания с обоснованием."""
    timezone: str = "Europe/Moscow"
    # Рабочее время площадки. По умолчанию 07–23 без выходных: в Москве шумные работы
    # ночью (23–07) в жилой застройке запрещены, а в выходные стройки работают. Ночные
    # смены, если они есть, всё равно видны: окно пар считается по времени работы самой техники.
    work_start_h: float = 7.0
    work_end_h: float = 23.0
    workdays: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    # окна техники
    pair_window_h: float = 2.0            # умолчание для пар без своего окна
    missing_window_h: float = 2.0         # умолчание для обязательных типов без своего окна
    missing_critical_factor: float = 4.0  # отсутствие дольше window × factor → critical
    plan_missing_window_h: float = 16.0   # тип только из плана (не обязательный по нормам) — 1 рабочий день
    max_gap_min: float = 75.0             # разрыв между наблюдениями больше — окно рвётся (не знаем, что было)
    idle_break_min: float = 60.0          # ведущая стоит дольше — «работает без пары» заканчивается
    min_frames_for_absence: int = 3       # отсутствие доказываем минимум тремя снимками
    forbidden_min_obs: int = 2            # «не по этапу» — минимум 2 снимка в работе (не одна ложная рамка)
    outside_min_obs: int = 2
    # простой и сверка моделей
    idle_alert_h: float = 4.0             # полоска не уменьшается полсмены → простой
    idle_critical_h: float = 16.0         # рабочий день → critical
    hours_warn_ratio: float = 1.0
    hours_critical_ratio: float = 1.2
    no_progress_days: int = 3             # этап не сменился столько дней после исчерпания часов
    # график
    schedule_tolerance_days: float = 3.0  # |отклонение| ≤ 3 дн. — «в срок» (как вердикт Ганта Дениса)
    schedule_critical_days: float = 14.0
    pace_window_days: int = 28            # окно «свежего» темпа для прогноза
    min_pace_days: int = 7                # короче — прогноз не даём
    # данные
    needs_review_warn: int = 5
    camera_silent_h: float = 2.0          # кадр раз в 20–30 мин: 2 ч тишины — 4–6 пропусков подряд
    camera_silent_critical_h: float = 24.0
    camera_bad_share: float = 0.5
    camera_min_frames: int = 4
    # что умеет детектор: типы вне списка не порождают «нет техники» (None — считаем, что умеет всё)
    detectable: tuple[str, ...] | None = None
    camera_names: dict[str, str] = field(default_factory=dict)
    cameras: list[dict] = field(default_factory=list)   # [{id, name, kind, last_frame_at}]

    @classmethod
    def from_dict(cls, data: dict | None) -> "AnalyticsConfig":
        """Настройки из БД: плоский словарь или {"thresholds": {...}}; лишние ключи игнорируются."""
        data = dict(data or {})
        merged = {**data, **(data.get("thresholds") or {})}
        known = {f.name for f in fields(cls)}
        kw: dict[str, Any] = {}
        for k, v in merged.items():
            if k not in known or v is None:
                continue
            if k in ("workdays", "detectable"):
                v = tuple(v)
            if k == "camera_names":
                v = {str(a): str(b) for a, b in dict(v).items()}
            kw[k] = v
        return cls(**kw)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class Obs:
    """Одна рамка техники во времени — с зоной, статусом единицы и признаком «на стоянке»."""
    frame: FrameInfo
    det: Detection
    unit: str
    status: UnitStatus | None
    zone: Zone | None

    @property
    def t(self) -> dt.datetime:
        return self.frame.captured_at

    @property
    def camera(self):
        return self.frame.camera_id

    @property
    def cls(self) -> str:
        return self.det.cls

    @property
    def working(self) -> bool:
        return self.det.activity == Activity.WORKING

    @property
    def parked(self) -> bool:
        """PARKED или в зоне отстоя: такая техника не участвует в сопоставлении с этапом (PLAN §3.9)."""
        return self.status == UnitStatus.PARKED or (self.zone is not None and self.zone.kind == "parking")

    @property
    def identified(self) -> bool:
        """Есть ли у рамки идентичность (единица после слияния камер или трек камеры)."""
        return bool(self.det.unit_id or self.det.track_id)


def count_units(obs) -> int:
    """Сколько разных машин в наблюдениях.

    Опознанные (unit_id / track_id) считаем по идентичности; неопознанные — максимумом за один
    кадр: без трекинга одна машина на пяти кадрах иначе превратилась бы в пять.
    """
    known = {o.unit for o in obs if o.identified}
    per_frame = Counter(o.frame.frame_id for o in obs if not o.identified)
    return len(known) + (max(per_frame.values()) if per_frame else 0)


def _aware(t: dt.datetime) -> dt.datetime:
    return t if t.tzinfo is not None else t.replace(tzinfo=dt.timezone.utc)


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
    return inside


@dataclass
class AnalyticsContext:
    site_id: int | str
    now: dt.datetime
    plan: list[PlanItem]
    timeline: StageTimeline
    units: list[UnitState] = field(default_factory=list)
    recent: list[tuple[FrameInfo, list[Detection]]] = field(default_factory=list)
    intervals: list[ActivityInterval] = field(default_factory=list)
    balances: list[HoursBalance] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        # контракт требует aware-время; наивное считаем UTC, чтобы сравнения с кадрами не падали
        self.now = _aware(self.now)
        self.config = dict(self.config or {})

    # ---------------------------------------------------------------- время

    @cached_property
    def cfg(self) -> AnalyticsConfig:
        return AnalyticsConfig.from_dict(self.config)

    @cached_property
    def tz(self) -> dt.tzinfo:
        if ZoneInfo is not None:
            try:
                return ZoneInfo(self.cfg.timezone)
            except Exception:
                pass
        return MSK

    def local(self, t: dt.datetime) -> dt.datetime:
        return _aware(t).astimezone(self.tz)

    @property
    def today(self) -> dt.date:
        return self.local(self.now).date()

    def day_start(self, d: dt.date) -> dt.datetime:
        return dt.datetime.combine(d, dt.time(0), tzinfo=self.tz)

    def _segments(self, day: dt.date) -> list[tuple[dt.datetime, dt.datetime]]:
        c = self.cfg
        if day.weekday() not in c.workdays:
            return []
        base = self.day_start(day)
        s, e = base + dt.timedelta(hours=c.work_start_h), base + dt.timedelta(hours=c.work_end_h)
        if c.work_end_h > c.work_start_h:
            return [(s, e)]
        # ночная смена через полночь (например, 20→08)
        return [(base, e), (s, base + dt.timedelta(hours=24))]

    def _round_the_clock(self) -> bool:
        c = self.cfg
        return c.work_start_h <= 0 and c.work_end_h >= 24 and len(set(c.workdays)) == 7

    def working_hours(self, t0: dt.datetime, t1: dt.datetime) -> float:
        """Рабочие часы площадки между t0 и t1 (ночь и выходные по настройке не считаются)."""
        t0, t1 = _aware(t0), _aware(t1)
        if t1 <= t0:
            return 0.0
        if self._round_the_clock():
            return (t1 - t0).total_seconds() / 3600
        a, b = self.local(t0), self.local(t1)
        total, day = 0.0, a.date()
        while day <= b.date():
            for s, e in self._segments(day):
                lo, hi = max(s, a), min(e, b)
                if hi > lo:
                    total += (hi - lo).total_seconds() / 3600
            day += dt.timedelta(days=1)
        return total

    def working_window_start(self, end: dt.datetime, hours: float) -> dt.datetime:
        """Момент, от которого до `end` набирается `hours` рабочих часов (начало окна)."""
        end = _aware(end)
        if self._round_the_clock() or not self.cfg.workdays:
            return end - dt.timedelta(hours=hours)
        b = self.local(end)
        left, day = hours, b.date()
        for _ in range(3660):  # не больше 10 лет назад — страховка от пустого расписания
            for s, e in reversed(self._segments(day)):
                hi = min(e, b)
                if hi <= s:
                    continue
                span = (hi - s).total_seconds() / 3600
                if span >= left:
                    return hi - dt.timedelta(hours=left)
                left -= span
            day -= dt.timedelta(days=1)
        return end - dt.timedelta(hours=hours)

    # ---------------------------------------------------------------- кадры и техника

    @cached_property
    def frames(self) -> list[FrameInfo]:
        """Кадры окна по времени (все камеры), время приведено к aware."""
        fs = [f if f.captured_at.tzinfo else _replace_time(f) for f, _ in self.recent]
        return sorted(fs, key=lambda f: f.captured_at)

    @cached_property
    def last_frame_at(self) -> dt.datetime | None:
        return _aware(self.frames[-1].captured_at) if self.frames else None

    @cached_property
    def units_by_id(self) -> dict[str, UnitState]:
        return {u.unit_id: u for u in self.units}

    @cached_property
    def zones_by_id(self) -> dict[int, Zone]:
        return {z.id: z for z in self.zones}

    def zones_of_camera(self, camera_id) -> list[Zone]:
        return [z for z in self.zones if z.camera_id is None or str(z.camera_id) == str(camera_id)]

    def zone_for(self, det: Detection, camera_id) -> Zone | None:
        """Зона детекции: из конвейера (zone_id), иначе по точке контакта с землёй."""
        if det.zone_id is not None and det.zone_id in self.zones_by_id:
            return self.zones_by_id[det.zone_id]
        x, y = det.foot
        # запретная зона важнее рабочей, если полигоны пересекаются
        hits = [z for z in self.zones_of_camera(camera_id) if len(z.polygon) >= 3 and point_in_polygon(x, y, z.polygon)]
        hits.sort(key=lambda z: z.kind != "restricted")
        return hits[0] if hits else None

    @cached_property
    def observations(self) -> list[Obs]:
        out: list[Obs] = []
        for frame, dets in self.recent:
            frame = frame if frame.captured_at.tzinfo else _replace_time(frame)
            for d in dets:
                if d.unit_id:
                    unit = d.unit_id
                elif d.track_id:
                    unit = f"{frame.camera_id}:{d.track_id}"
                else:
                    # без трекинга: «какой-то каток на камере 1» — группировка по машине грубее,
                    # но правило «работает ≥ 2 кадров» и зоны продолжают работать
                    unit = f"{frame.camera_id}:{d.cls}"
                u = self.units_by_id.get(d.unit_id) if d.unit_id else None
                out.append(Obs(frame, d, unit, u.status if u else None, self.zone_for(d, frame.camera_id)))
        out.sort(key=lambda o: o.t)
        return out

    def camera_name(self, camera_id) -> str:
        return str(self.cfg.camera_names.get(str(camera_id), camera_id))

    def unit_label(self, unit: str, cls: str | None = None) -> str:
        from core.analytics import fmt
        u = self.units_by_id.get(unit)
        if u and u.label:
            return u.label
        return fmt.eq(u.cls if u else (cls or "техника"))

    # ---------------------------------------------------------------- план и этапы

    @cached_property
    def plan_by_stage(self) -> dict[int, PlanItem]:
        """Этап → строка плана (несколько строк одного этапа сливаются: min начала, max окончания)."""
        out: dict[int, PlanItem] = {}
        for it in self.plan:
            prev = out.get(it.stage_id)
            if prev is None:
                out[it.stage_id] = it
                continue
            starts = [d for d in (prev.planned_start, it.planned_start) if d]
            ends = [d for d in (prev.planned_end, it.planned_end) if d]
            out[it.stage_id] = PlanItem(
                stage_id=it.stage_id, name=prev.name or it.name,
                planned_start=min(starts) if starts else None, planned_end=max(ends) if ends else None,
                work_codes=list(dict.fromkeys(prev.work_codes + it.work_codes)),
                equipment={k: max(prev.equipment.get(k, 0), it.equipment.get(k, 0))
                           for k in set(prev.equipment) | set(it.equipment)},
                planned_hours={**prev.planned_hours, **it.planned_hours},
                hours_manual=prev.hours_manual or it.hours_manual,
            )
        return out

    def stages_by_plan(self, day: dt.date | None = None) -> list[int]:
        day = day or self.today
        return sorted(s for s, it in self.plan_by_stage.items()
                      if it.planned_start and it.planned_end and it.planned_start <= day <= it.planned_end)

    def stage_status(self, stage_id: int) -> StageStatus:
        st = self.timeline.states.get(stage_id)
        return st.status if st else StageStatus.NOT_STARTED

    def stages_by_fact(self) -> list[int]:
        out = {s for s, st in self.timeline.states.items() if st.status == StageStatus.ACTIVE}
        cur = self.timeline.current_stage
        if cur is not None and self.stage_status(cur) != StageStatus.DONE:
            out.add(cur)
        return sorted(out)

    def current_stages(self) -> list[int]:
        """Этапы, идущие сейчас: по плану или по факту, кроме уже завершённых по факту.

        Завершённый досрочно этап из «идущих» убираем: его техника на площадке больше не нужна.
        Порядок: сначала фронт по модели Б, затем по возрастанию — для выбора этапа в тексте.
        """
        ids = set(self.stages_by_plan()) | set(self.stages_by_fact())
        ids = {s for s in ids if self.stage_status(s) != StageStatus.DONE}
        front = self.timeline.current_stage
        return sorted(ids, key=lambda s: (s != front, s))

    def has_stage_data(self) -> bool:
        """Есть ли у модели Б хоть что-то: наблюдения по дням или ручные отметки."""
        tl = self.timeline
        return bool(tl.daily_front) or any(st.manual or st.status != StageStatus.NOT_STARTED
                                           for st in tl.states.values())


def _replace_time(frame: FrameInfo) -> FrameInfo:
    """Кадр с наивным временем считаем UTC (контракт требует aware, но не падаем)."""
    from dataclasses import replace
    return replace(frame, captured_at=_aware(frame.captured_at))
