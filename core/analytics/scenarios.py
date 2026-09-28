"""Синтетические площадки: короткие сборщики контрактных типов и эталон ТЗ.

Нужны трём потребителям: тестам аналитики (без БД и моделей), засеву демо-объекта
(tools/seed_demo.py может взять `tz_reference()` как готовую сцену) и документации —
эталонный пример из ТЗ воспроизводится одной функцией и проверяется тестом.
"""
from __future__ import annotations

import datetime as dt

from core import taxonomy
from core.analytics.context import MSK, AnalyticsContext
from core.contracts import (
    Activity, ActivityInterval, Detection, FrameInfo, HoursBalance, PlanItem, StageState,
    StageStatus, StageTimeline, UnitState, UnitStatus, Weather, Zone,
)

FULL_FRAME = [(0.0, 0.0), (1920.0, 0.0), (1920.0, 1080.0), (0.0, 1080.0)]


def frame(fid, camera, t: dt.datetime, *, site=1, quality_ok: bool = True, is_night: bool = False,
          reject_reason: str = "", weather: Weather = Weather.CLEAR) -> FrameInfo:
    return FrameInfo(frame_id=fid, camera_id=camera, site_id=site, captured_at=t, width=1920, height=1080,
                     is_night=is_night, weather=weather, quality_ok=quality_ok, reject_reason=reject_reason)


def det(cls: str, unit: str | None = None, *, working: bool = True, zone: int | None = None,
        bbox=(800.0, 500.0, 220.0, 160.0), activity: Activity | None = None, track: str | None = None) -> Detection:
    act = activity or (Activity.WORKING if working else Activity.IDLE)
    return Detection(cls=cls, conf=0.9, bbox=tuple(bbox), source="test", unit_id=unit, track_id=track,
                     moved_since_prev=act == Activity.WORKING, activity=act, zone_id=zone)


def unit(uid: str, cls: str, status: UnitStatus, *, now: dt.datetime, first_seen: dt.datetime | None = None,
         last_seen: dt.datetime | None = None, last_moved: dt.datetime | None = None, label: str = "",
         cameras=("1",)) -> UnitState:
    return UnitState(unit_id=uid, cls=cls, status=status, first_seen=first_seen or now - dt.timedelta(days=3),
                     last_seen=last_seen or now, last_moved=last_moved, cameras=set(cameras), label=label)


def zone(zid: int, name: str, kind: str = "work", camera=1, polygon=None) -> Zone:
    return Zone(id=zid, name=name, kind=kind, camera_id=camera, polygon=list(polygon or FULL_FRAME))


def timeline(states: dict[int, tuple], *, front: int | None = None, daily_front=None,
             needs_review=(), outliers=()) -> StageTimeline:
    """states: этап → (статус, готовность[, фактическое начало[, окончание[, кадры-доказательства[, manual]]]])."""
    out: dict[int, StageState] = {}
    for s, spec in states.items():
        status, progress = StageStatus(spec[0]), float(spec[1])
        out[s] = StageState(stage_id=s, status=status, progress=progress,
                            actual_start=spec[2] if len(spec) > 2 else None,
                            actual_end=spec[3] if len(spec) > 3 else None,
                            evidence_frame_ids=list(spec[4]) if len(spec) > 4 else [],
                            manual=bool(spec[5]) if len(spec) > 5 else False, confidence=0.8)
    weights = {s.id: s.weight for s in taxonomy.stages().values()}
    overall = sum(weights[s] * (1.0 if st.status == StageStatus.DONE else st.progress if st.status == StageStatus.ACTIVE else 0.0)
                  for s, st in out.items()) / 100.0
    if front is None:
        active = [s for s, st in out.items() if st.status == StageStatus.ACTIVE]
        front = max(active) if active else None
    return StageTimeline(states=out, current_stage=front, overall_progress=overall,
                         daily_front=list(daily_front or []), needs_review=list(needs_review),
                         rejected_outliers=list(outliers))


def plan(spans: dict[int, tuple[dt.date, dt.date]], equipment: dict[int, dict] | None = None) -> list[PlanItem]:
    return [PlanItem(stage_id=s, planned_start=a, planned_end=b, name=taxonomy.stage_name(s),
                     equipment=dict((equipment or {}).get(s, {})))
            for s, (a, b) in sorted(spans.items())]


def context(now: dt.datetime, *, plan_items=None, tl=None, units=(), recent=(), intervals=(), balances=(),
            zones=(), config=None, site=1) -> AnalyticsContext:
    return AnalyticsContext(site_id=site, now=now, plan=list(plan_items or []),
                            timeline=tl or timeline({}), units=list(units), recent=list(recent),
                            intervals=list(intervals), balances=list(balances), zones=list(zones),
                            config=dict(config or {}))


# --------------------------------------------------------------------------
# эталон ТЗ
# --------------------------------------------------------------------------

TZ_NOW = dt.datetime(2026, 9, 28, 12, 10, tzinfo=MSK)


def excavation_plan(today: dt.date) -> list[PlanItem]:
    """План, в котором этап 3 (котлован) идёт сегодня, 1–2 уже прошли, 4–8 впереди."""
    d = dt.timedelta
    return plan({
        1: (today - d(95), today - d(60)),
        2: (today - d(70), today - d(28)),
        3: (today - d(27), today + d(17)),
        4: (today + d(10), today + d(85)),
        5: (today + d(80), today + d(320)),
        6: (today + d(310), today + d(360)),
        7: (today + d(230), today + d(400)),
        8: (today + d(350), today + d(440)),
    }, equipment={3: {"excavator": 1, "dump_truck": 3}})


def tz_reference(now: dt.datetime = TZ_NOW, *, with_trucks: bool = False, minutes: int = 130) -> AnalyticsContext:
    """Пример из ТЗ: «Устройство котлована», экскаватор работает, самосвалов нет.

    Две камеры смотрят на котлован (зона «Котлован» размечена на каждой), кадры раз в 26 минут
    со сдвигом 13 минут между камерами. Экскаватор работает `minutes` минут (по умолчанию
    2 ч 10 мин); with_trucks=True добавляет самосвал в середину окна — отклонения быть не должно.
    """
    start = now - dt.timedelta(minutes=minutes)
    today = now.astimezone(MSK).date()
    recent = []
    fid = 1000
    zones = [zone(1, "Котлован", "work", camera=1), zone(2, "Котлован", "work", camera=2)]
    t, k = start, 0
    while t <= now:
        cam = 1 if k % 2 == 0 else 2
        dets = [det("excavator", "U-EXC-1", working=True, zone=cam)]
        if with_trucks and abs((t - start) - (now - start) / 2) <= dt.timedelta(minutes=13):
            dets.append(det("dump_truck", "U-DT-1", working=True, zone=cam, bbox=(1200, 520, 260, 170)))
        recent.append((frame(fid, cam, t), dets))
        fid += 1
        k += 1
        t += dt.timedelta(minutes=13)
    # последний кадр строго в `now` на камере 1 — эпизод длится ровно `minutes`
    if recent[-1][0].captured_at != now:
        recent.append((frame(fid, 1, now), [det("excavator", "U-EXC-1", working=True, zone=1)]))
    tl = timeline({1: ("done", 1.0, today - dt.timedelta(days=92), today - dt.timedelta(days=58)),
                   2: ("done", 1.0, today - dt.timedelta(days=66), today - dt.timedelta(days=27)),
                   3: ("active", 0.6, today - dt.timedelta(days=25), None, [900, 950])},
                  front=3, daily_front=[(today - dt.timedelta(days=i), 3) for i in range(25, -1, -1)])
    units = [unit("U-EXC-1", "excavator", UnitStatus.ACTIVE, now=now, last_moved=now, label="Экскаватор №1",
                  cameras=("1", "2"))]
    if with_trucks:
        units.append(unit("U-DT-1", "dump_truck", UnitStatus.ACTIVE, now=now, last_moved=now, label="Самосвал №1"))
    intervals = [ActivityInterval("U-EXC-1", "excavator", start, now, minutes / 60, 3,
                                  [f.frame_id for f, _ in recent[-3:]])]
    balances = [HoursBalance(3, "excavator", 220.0, 120.0, now), HoursBalance(3, "dump_truck", 480.0, 190.0,
                                                                                 now - dt.timedelta(hours=20))]
    return context(now, plan_items=excavation_plan(today), tl=tl, units=units, recent=recent, intervals=intervals,
                   balances=balances, zones=zones, config={"camera_names": {"1": "1", "2": "2"}})
