"""Правила отклонений: каждый тип — позитивный и негативный случай, эталон ТЗ, стабильность ключей."""
import dataclasses
import datetime as dt

import pytest

from core.analytics import rules
from core.analytics import scenarios as sc
from core.analytics.context import MSK
from core.contracts import (
    ActivityInterval, DeviationType as DT, HoursBalance, Severity, UnitStatus,
)

NOW = dt.datetime(2026, 9, 28, 15, 0, tzinfo=MSK)
TODAY = NOW.date()
DAY = dt.timedelta(days=1)


def series(camera, start, n, step_min=20, dets=lambda k: [], fid0=0, **frame_kw):
    """n кадров камеры с шагом step_min; dets(k) — детекции k-го кадра."""
    return [(sc.frame(fid0 + k, camera, start + dt.timedelta(minutes=step_min * k), **frame_kw), dets(k))
            for k in range(n)]


def of_type(devs, t):
    return [d for d in devs if d.type == t]


def ctx_stage(stage, *, recent=(), units=(), plan_spans=None, tl=None, **kw):
    """Площадка, где идёт один этап и по плану, и по факту."""
    spans = plan_spans or {stage: (TODAY - 20 * DAY, TODAY + 30 * DAY)}
    tl = tl or sc.timeline({stage: ("active", 0.4, TODAY - 18 * DAY)}, front=stage,
                           daily_front=[(TODAY - i * DAY, stage) for i in range(18, -1, -1)])
    return sc.context(NOW, plan_items=sc.plan(spans), tl=tl, units=units, recent=recent, **kw)


def assert_explained(d):
    """Каждое отклонение объясняет себя: что видели, что ожидалось, что проверить; ключ стабилен."""
    assert d.key and d.title and d.message
    assert "Видели:" in d.message
    assert "ожидается" in d.message.lower()
    assert "Что проверить" in d.message or "Что сделать" in d.message


# --------------------------------------------------------------------------- эталон ТЗ


def test_tz_reference_pair_broken():
    """Этап «Устройство котлована», экскаватор работает, самосвалов нет 2 ч 10 мин → warning с зоной и снимками."""
    devs = rules.evaluate(sc.tz_reference())
    pb = of_type(devs, DT.PAIR_BROKEN)
    assert len(pb) == 1
    d = pb[0]
    assert d.severity == Severity.WARNING and d.stage_id == 3
    assert d.title.startswith("Возможное снижение темпа: экскаватор работает без самосвалов")
    assert d.message.startswith("Возможное снижение темпа: экскаватор работает без самосвалов 2 ч 10 мин, "
                                "зона «Котлован», камеры 1, 2")
    assert len(d.frame_ids) == 3 and d.zone_id in (1, 2) and d.unit_ids == ["U-EXC-1"]
    assert d.data["duration_h"] == pytest.approx(130 / 60, abs=0.01)
    assert_explained(d)
    # отсутствие самосвалов уже объяснено парой — отдельного «нет техники» нет
    assert not of_type(devs, DT.EQUIPMENT_MISSING)


def test_pair_key_is_stable_while_episode_continues():
    first = of_type(rules.evaluate(sc.tz_reference(NOW)), DT.PAIR_BROKEN)[0]
    later = of_type(rules.evaluate(sc.tz_reference(NOW + dt.timedelta(minutes=26), minutes=156)), DT.PAIR_BROKEN)[0]
    assert first.key == later.key
    assert later.data["duration_h"] > first.data["duration_h"]


def test_pair_not_broken_when_trucks_appear_in_window():
    assert not of_type(rules.evaluate(sc.tz_reference(with_trucks=True)), DT.PAIR_BROKEN)


def test_pair_needs_the_whole_window_not_one_frame():
    assert not of_type(rules.evaluate(sc.tz_reference(minutes=90)), DT.PAIR_BROKEN)


def test_parked_excavator_is_not_a_leader():
    ctx = sc.tz_reference()
    ctx.units[0].status = UnitStatus.PARKED
    assert not of_type(rules.evaluate(ctx), DT.PAIR_BROKEN)


def test_long_pair_break_escalates_to_critical():
    d = of_type(rules.evaluate(sc.tz_reference(minutes=400)), DT.PAIR_BROKEN)[0]
    assert d.severity == Severity.CRITICAL


def test_gap_in_observations_breaks_the_window():
    start = NOW - dt.timedelta(hours=5)
    exc = lambda k: [sc.det("excavator", "E1")]
    recent = series(1, start, 4, 20, exc) + series(1, start + dt.timedelta(hours=3, minutes=20), 5, 20, exc, fid0=10)
    ctx = ctx_stage(3, recent=recent, units=[sc.unit("E1", "excavator", UnitStatus.ACTIVE, now=NOW)])
    assert not of_type(rules.evaluate(ctx), DT.PAIR_BROKEN)


# --------------------------------------------------------------------------- позитивные случаи всех типов


def case_pair_broken():
    return sc.tz_reference()


def case_missing():
    """Котлован: 3 ч самосвалы и бульдозер на месте, экскаватора нет."""
    dets = lambda k: [sc.det("dump_truck", "T1", working=False), sc.det("bulldozer", "B1")]
    recent = series(1, NOW - dt.timedelta(hours=3), 10, 20, dets)
    return ctx_stage(3, recent=recent, units=[sc.unit("T1", "dump_truck", UnitStatus.IDLE, now=NOW)])


def case_forbidden():
    """Надземный монолит: работает каток."""
    recent = series(1, NOW - dt.timedelta(hours=1), 4, 20, lambda k: [sc.det("roller", "R1"), sc.det("tower_crane", "C1")])
    return ctx_stage(5, recent=recent, units=[sc.unit("R1", "roller", UnitStatus.ACTIVE, now=NOW),
                                              sc.unit("C1", "tower_crane", UnitStatus.ACTIVE, now=NOW)])


def case_idle():
    """Надземный монолит: башенный кран стоит 6 ч рабочего времени, полоска не уменьшается."""
    stood = NOW - dt.timedelta(hours=6)
    recent = series(1, NOW - dt.timedelta(hours=3), 10, 20,
                    lambda k: [sc.det("tower_crane", "C1", working=False), sc.det("concrete_mixer", "M1")])
    units = [sc.unit("C1", "tower_crane", UnitStatus.IDLE, now=NOW, last_moved=stood, label="Башенный кран №1")]
    intervals = [ActivityInterval("C1", "tower_crane", stood - dt.timedelta(hours=2), stood, 2.0, 5, [1])]
    balances = [HoursBalance(5, "tower_crane", 1500.0, 400.0, stood)]
    return ctx_stage(5, recent=recent, units=units, intervals=intervals, balances=balances)


def case_parked_only():
    """Котлован по плану, вся техника этапа дольше 48 ч без движения."""
    recent = series(1, NOW - dt.timedelta(hours=2), 6, 20,
                    lambda k: [sc.det("excavator", "E1", working=False), sc.det("dump_truck", "T1", working=False)])
    units = [sc.unit("E1", "excavator", UnitStatus.PARKED, now=NOW, last_moved=NOW - 4 * DAY),
             sc.unit("T1", "dump_truck", UnitStatus.PARKED, now=NOW, last_moved=NOW - 3 * DAY)]
    return ctx_stage(3, recent=recent, units=units)


def case_outside_zone():
    zones = [sc.zone(1, "Котлован", "work", polygon=[(0, 0), (960, 0), (960, 1080), (0, 1080)]),
             sc.zone(7, "Охранная зона ЛЭП", "restricted", polygon=[(1000, 0), (1920, 0), (1920, 1080), (1000, 1080)])]
    recent = series(1, NOW - dt.timedelta(hours=1), 3, 20,
                    lambda k: [sc.det("mobile_crane", "K1", bbox=(1300, 400, 300, 200)),
                               sc.det("excavator", "E1", bbox=(200, 500, 200, 150)),
                               sc.det("dump_truck", "T1", bbox=(300, 700, 200, 150))])
    return ctx_stage(3, recent=recent, zones=zones)


def case_hours_no_progress():
    """Котлован: экскаватор отработал 112 % плановых часов, этап 10 дней тот же и не завершён."""
    return ctx_stage(3, balances=[HoursBalance(3, "excavator", 160.0, 180.0, NOW - dt.timedelta(hours=3))],
                     intervals=[ActivityInterval("E1", "excavator", NOW - dt.timedelta(hours=5), NOW - dt.timedelta(hours=3),
                                                 2.0, 3, [501, 502])])


def case_late_start():
    tl = sc.timeline({3: ("active", 0.7, TODAY - 40 * DAY, None, [11, 12])}, front=3,
                     daily_front=[(TODAY - i * DAY, 3) for i in range(10, -1, -1)])
    recent = series(1, NOW - dt.timedelta(hours=1), 3, 20)
    return sc.context(NOW, plan_items=sc.plan({3: (TODAY - 45 * DAY, TODAY + 5 * DAY), 4: (TODAY - 10 * DAY, TODAY + 60 * DAY)}),
                      tl=tl, recent=recent)


def case_overdue():
    tl = sc.timeline({3: ("active", 0.8, TODAY - 50 * DAY, None, [21, 22])}, front=3,
                     daily_front=[(TODAY - i * DAY, 3) for i in range(10, -1, -1)])
    return sc.context(NOW, plan_items=sc.plan({3: (TODAY - 50 * DAY, TODAY - 6 * DAY)}), tl=tl,
                      recent=series(1, NOW - dt.timedelta(hours=1), 3, 20))


def case_early():
    tl = sc.timeline({4: ("active", 0.2, TODAY - 12 * DAY, None, [31, 32, 33])}, front=4,
                     daily_front=[(TODAY - i * DAY, 4) for i in range(12, -1, -1)])
    return sc.context(NOW, plan_items=sc.plan({4: (TODAY - 2 * DAY, TODAY + 60 * DAY)}), tl=tl)


def case_out_of_plan():
    tl = sc.timeline({2: ("active", 0.5, TODAY - 5 * DAY, None, [41, 42])}, front=2,
                     daily_front=[(TODAY - i * DAY, 2) for i in range(5, -1, -1)])
    return sc.context(NOW, plan_items=sc.plan({3: (TODAY + 10 * DAY, TODAY + 60 * DAY)}), tl=tl)


def case_needs_review():
    tl = sc.timeline({3: ("active", 0.4, TODAY - 5 * DAY)}, front=3, needs_review=[61, 62, 63, 64, 65, 66])
    return sc.context(NOW, plan_items=sc.plan({3: (TODAY - 6 * DAY, TODAY + 30 * DAY)}), tl=tl)


def case_camera_issue():
    recent = series(2, NOW - dt.timedelta(hours=9), 6, 20)
    return sc.context(NOW, recent=recent, config={"cameras": [{"id": 2, "kind": "stream"}]})


POSITIVE = {
    DT.PAIR_BROKEN: case_pair_broken,
    DT.EQUIPMENT_MISSING: case_missing,
    DT.EQUIPMENT_FORBIDDEN: case_forbidden,
    DT.EQUIPMENT_IDLE: case_idle,
    DT.EQUIPMENT_PARKED_ONLY: case_parked_only,
    DT.OUTSIDE_ZONE: case_outside_zone,
    DT.HOURS_SPENT_NO_PROGRESS: case_hours_no_progress,
    DT.STAGE_LATE_START: case_late_start,
    DT.STAGE_OVERDUE: case_overdue,
    DT.STAGE_EARLY: case_early,
    DT.STAGE_OUT_OF_PLAN: case_out_of_plan,
    DT.NEEDS_REVIEW: case_needs_review,
    DT.CAMERA_ISSUE: case_camera_issue,
}


def test_every_deviation_type_has_a_scenario():
    assert set(POSITIVE) == set(DT)


@pytest.mark.parametrize("dtype", list(POSITIVE), ids=lambda t: t.value)
def test_positive_case_is_detected_and_explained(dtype):
    devs = of_type(rules.evaluate(POSITIVE[dtype]()), dtype)
    assert devs, f"{dtype.value} не найден"
    for d in devs:
        assert_explained(d)


@pytest.mark.parametrize("dtype", list(POSITIVE), ids=lambda t: t.value)
def test_keys_are_stable_between_recomputations(dtype):
    a = {d.key for d in rules.evaluate(POSITIVE[dtype]())}
    b = {d.key for d in rules.evaluate(POSITIVE[dtype]())}
    assert a == b and len(a) == len(rules.evaluate(POSITIVE[dtype]()))


# --------------------------------------------------------------------------- детали и негативные случаи


def test_missing_key_survives_sliding_window():
    """Веб-слой передаёт «последние N часов»: окно сдвинулось на кадр — эпизод тот же, ключ тот же."""
    ctx = case_missing()
    dets = lambda k: [sc.det("dump_truck", "T1", working=False), sc.det("bulldozer", "B1")]
    later = dataclasses.replace(ctx, now=NOW + dt.timedelta(minutes=20),
                                recent=ctx.recent[1:] + series(1, NOW + dt.timedelta(minutes=20), 1, 20, dets, fid0=99))
    k1 = {d.key for d in of_type(rules.evaluate(ctx), DT.EQUIPMENT_MISSING)}
    k2 = {d.key for d in of_type(rules.evaluate(later), DT.EQUIPMENT_MISSING)}
    assert k1 == k2 and len(k1) == 1


def test_missing_names_the_type_and_shows_evidence():
    d = of_type(rules.evaluate(case_missing()), DT.EQUIPMENT_MISSING)
    assert [x.data["cls"] for x in d] == ["excavator"]
    assert d[0].severity == Severity.WARNING and len(d[0].frame_ids) == 3 and d[0].stage_id == 3


def test_missing_everything_gives_one_record():
    ctx = ctx_stage(3, recent=series(1, NOW - dt.timedelta(hours=3), 10, 20))
    d = of_type(rules.evaluate(ctx), DT.EQUIPMENT_MISSING)
    assert len(d) == 1 and d[0].key.startswith("equipment_missing:3:all:")


def test_no_missing_when_required_equipment_works():
    recent = series(1, NOW - dt.timedelta(hours=3), 10, 20,
                    lambda k: [sc.det("excavator", "E1"), sc.det("dump_truck", f"T{k % 2}")])
    assert not of_type(rules.evaluate(ctx_stage(3, recent=recent)), DT.EQUIPMENT_MISSING)


def test_parked_equipment_does_not_count_as_present():
    recent = series(1, NOW - dt.timedelta(hours=3), 10, 20,
                    lambda k: [sc.det("excavator", "E1", working=False), sc.det("dump_truck", "T1")])
    units = [sc.unit("E1", "excavator", UnitStatus.PARKED, now=NOW, last_seen=NOW - 20 * DAY)]
    d = of_type(rules.evaluate(ctx_stage(3, recent=recent, units=units)), DT.EQUIPMENT_MISSING)
    assert [x.data["cls"] for x in d] == ["excavator"]


def test_no_missing_without_enough_frames():
    """Камера молчала — отсутствие техники не доказано (это CAMERA_ISSUE, а не EQUIPMENT_MISSING)."""
    recent = series(1, NOW - dt.timedelta(hours=3), 2, 150)
    assert not of_type(rules.evaluate(ctx_stage(3, recent=recent)), DT.EQUIPMENT_MISSING)


def test_undetectable_classes_are_not_reported_missing():
    recent = series(1, NOW - dt.timedelta(hours=30), 60, 30, lambda k: [sc.det("concrete_mixer", "M1")])
    base = ctx_stage(5, recent=recent)
    assert any(d.data.get("cls") == "tower_crane" for d in of_type(rules.evaluate(base), DT.EQUIPMENT_MISSING))
    ctx = ctx_stage(5, recent=recent, config={"detectable": ["concrete_mixer", "excavator", "dump_truck"]})
    assert not of_type(rules.evaluate(ctx), DT.EQUIPMENT_MISSING)


def test_forbidden_severity_and_parallel_stage():
    d = of_type(rules.evaluate(case_forbidden()), DT.EQUIPMENT_FORBIDDEN)
    assert len(d) == 1 and d[0].severity == Severity.WARNING and d[0].unit_ids == ["R1"]
    # каток допустим на благоустройстве: если этап 8 по плану идёт параллельно — не аномалия
    ctx = case_forbidden()
    ctx = dataclasses.replace(ctx, plan=ctx.plan + sc.plan({8: (TODAY - 5 * DAY, TODAY + 40 * DAY)}))
    assert not of_type(rules.evaluate(ctx), DT.EQUIPMENT_FORBIDDEN)


def test_idle_or_parked_forbidden_equipment_is_not_an_anomaly():
    recent = series(1, NOW - dt.timedelta(hours=1), 4, 20, lambda k: [sc.det("roller", "R1", working=False)])
    assert not of_type(rules.evaluate(ctx_stage(5, recent=recent)), DT.EQUIPMENT_FORBIDDEN)
    recent = series(1, NOW - dt.timedelta(hours=1), 4, 20, lambda k: [sc.det("roller", "R1")])
    parked = [sc.unit("R1", "roller", UnitStatus.PARKED, now=NOW)]
    assert not of_type(rules.evaluate(ctx_stage(5, recent=recent, units=parked)), DT.EQUIPMENT_FORBIDDEN)
    one = series(1, NOW - dt.timedelta(minutes=10), 1, 20, lambda k: [sc.det("roller", "R1")])
    assert not of_type(rules.evaluate(ctx_stage(5, recent=one)), DT.EQUIPMENT_FORBIDDEN)


def test_excavator_on_superstructure_is_info():
    recent = series(1, NOW - dt.timedelta(hours=1), 4, 20, lambda k: [sc.det("excavator", "E1")])
    d = of_type(rules.evaluate(ctx_stage(5, recent=recent)), DT.EQUIPMENT_FORBIDDEN)
    assert d and d[0].severity == Severity.INFO


def test_idle_message_shows_time_bar_and_negative_cases():
    d = of_type(rules.evaluate(case_idle()), DT.EQUIPMENT_IDLE)
    assert len(d) == 1 and d[0].data["cls"] == "tower_crane"
    assert "осталось 1100 ч" in d[0].message and "Башенный кран №1" in d[0].message
    ctx = case_idle()
    ctx.units[0].status = UnitStatus.ACTIVE
    assert not of_type(rules.evaluate(ctx), DT.EQUIPMENT_IDLE)
    ctx = case_idle()
    fresh = NOW - dt.timedelta(hours=2)
    ctx.units[0].last_moved = fresh
    ctx.intervals[0] = ActivityInterval("C1", "tower_crane", fresh - dt.timedelta(hours=1), fresh, 1.0, 5, [1])
    ctx.balances[0] = HoursBalance(5, "tower_crane", 1500.0, 401.0, fresh)
    assert not of_type(rules.evaluate(ctx), DT.EQUIPMENT_IDLE)


def test_parked_only_negative_when_something_works():
    ctx = case_parked_only()
    ctx.units[0].status = UnitStatus.ACTIVE
    devs = rules.evaluate(ctx)
    assert not of_type(devs, DT.EQUIPMENT_PARKED_ONLY)
    d = of_type(rules.evaluate(case_parked_only()), DT.EQUIPMENT_PARKED_ONLY)[0]
    assert set(d.unit_ids) == {"E1", "T1"} and d.frame_ids


def test_outside_zone_restricted_and_negative():
    devs = of_type(rules.evaluate(case_outside_zone()), DT.OUTSIDE_ZONE)
    assert len(devs) == 1
    d = devs[0]
    assert d.zone_id == 7 and d.severity == Severity.WARNING and "Охранная зона ЛЭП" in d.message
    assert d.unit_ids == ["K1"]
    # та же техника внутри рабочей зоны — нарушения нет
    zones = [sc.zone(1, "Котлован", "work")]
    recent = series(1, NOW - dt.timedelta(hours=1), 3, 20, lambda k: [sc.det("mobile_crane", "K1")])
    assert not of_type(rules.evaluate(ctx_stage(3, recent=recent, zones=zones)), DT.OUTSIDE_ZONE)


def test_outside_all_zones_is_info():
    zones = [sc.zone(1, "Котлован", "work", polygon=[(0, 0), (500, 0), (500, 500), (0, 500)])]
    recent = series(1, NOW - dt.timedelta(hours=1), 3, 20, lambda k: [sc.det("dump_truck", "T1", bbox=(1400, 800, 200, 150))])
    d = of_type(rules.evaluate(ctx_stage(3, recent=recent, zones=zones)), DT.OUTSIDE_ZONE)
    assert len(d) == 1 and d[0].severity == Severity.INFO


def test_hours_no_progress_levels_and_negatives():
    d = of_type(rules.evaluate(case_hours_no_progress()), DT.HOURS_SPENT_NO_PROGRESS)
    assert len(d) == 1 and d[0].severity == Severity.WARNING and d[0].stage_id == 3
    assert "112 %" in d[0].message and d[0].frame_ids
    crit = ctx_stage(3, balances=[HoursBalance(3, "excavator", 160.0, 200.0, NOW)])
    assert of_type(rules.evaluate(crit), DT.HOURS_SPENT_NO_PROGRESS)[0].severity == Severity.CRITICAL
    under = ctx_stage(3, balances=[HoursBalance(3, "excavator", 160.0, 120.0, NOW)])
    assert not of_type(rules.evaluate(under), DT.HOURS_SPENT_NO_PROGRESS)
    done = ctx_stage(3, balances=[HoursBalance(3, "excavator", 160.0, 200.0, NOW)],
                     tl=sc.timeline({3: ("done", 1.0, TODAY - 30 * DAY, TODAY - DAY)}, front=4))
    assert not of_type(rules.evaluate(done), DT.HOURS_SPENT_NO_PROGRESS)
    # этап сменился вчера — рано говорить «не сменился»
    fresh = ctx_stage(3, balances=[HoursBalance(3, "excavator", 160.0, 200.0, NOW)],
                      tl=sc.timeline({3: ("active", 0.1, TODAY - DAY)}, front=3,
                                     daily_front=[(TODAY - 3 * DAY, 2), (TODAY - 2 * DAY, 2), (TODAY - DAY, 3), (TODAY, 3)]))
    assert not of_type(rules.evaluate(fresh), DT.HOURS_SPENT_NO_PROGRESS)


def test_schedule_details():
    late = {d.stage_id: d for d in of_type(rules.evaluate(case_late_start()), DT.STAGE_LATE_START)}
    assert late[4].data["days"] == 10 and "не начат" in late[4].title
    assert late[4].severity == Severity.WARNING and late[4].frame_ids    # последний снимок как доказательство
    # этап 3 начался на 5 дней позже плана и уже идёт — справка с кадром, где он впервые замечен
    assert late[3].severity == Severity.INFO and late[3].frame_ids == [11]
    over = of_type(rules.evaluate(case_overdue()), DT.STAGE_OVERDUE)[0]
    assert over.data["days"] == 6 and 22 in over.frame_ids
    early = of_type(rules.evaluate(case_early()), DT.STAGE_EARLY)[0]
    assert early.severity == Severity.INFO and early.frame_ids == [31]    # кадр, где этап впервые замечен
    oop = of_type(rules.evaluate(case_out_of_plan()), DT.STAGE_OUT_OF_PLAN)[0]
    assert oop.stage_id == 2


def test_schedule_negatives():
    on_time = sc.context(NOW, plan_items=sc.plan({3: (TODAY - 10 * DAY, TODAY + 30 * DAY)}),
                         tl=sc.timeline({3: ("active", 0.3, TODAY - 9 * DAY)}, front=3,
                                        daily_front=[(TODAY - i * DAY, 3) for i in range(9, -1, -1)]))
    devs = rules.evaluate(on_time)
    assert not [d for d in devs if d.type.value.startswith("stage_")]
    # без данных модели Б никаких «этап не начат»: это была бы ложь
    no_data = sc.context(NOW, plan_items=sc.plan({3: (TODAY - 40 * DAY, TODAY - 10 * DAY)}))
    assert not [d for d in rules.evaluate(no_data) if d.type.value.startswith("stage_")]
    # этапы, пройденные до начала плана, — не «вне плана»
    before = sc.context(NOW, plan_items=sc.plan({3: (TODAY - 5 * DAY, TODAY + 40 * DAY)}),
                        tl=sc.timeline({1: ("done", 1.0), 2: ("done", 1.0), 3: ("active", 0.1, TODAY - 4 * DAY)}, front=3,
                                       daily_front=[(TODAY - i * DAY, 3) for i in range(4, -1, -1)]))
    assert not of_type(rules.evaluate(before), DT.STAGE_OUT_OF_PLAN)


def test_out_of_plan_skips_stage_closed_by_chronology_and_always_has_a_shot():
    """План 1 и 3 (частный дом: шпунта нет). Фронт перешагнул этап 2 — «пройден» без кадров
    с его признаками: это не работа вне плана. Этап вне плана, который идёт, но кадров-доказательств
    у него нет, — показываем последние снимки: отклонение без снимка не объяснить."""
    recent = series(1, NOW - dt.timedelta(hours=2), 6)
    skipped = sc.context(NOW, plan_items=sc.plan({1: (TODAY - 9 * DAY, TODAY - 5 * DAY),
                                                  3: (TODAY - 2 * DAY, TODAY + 9 * DAY)}),
                         tl=sc.timeline({1: ("done", 1.0), 2: ("done", 1.0), 3: ("active", 0.5, TODAY - DAY, None, [7])},
                                        front=3, daily_front=[(TODAY - DAY, 3), (TODAY, 3)]), recent=recent)
    assert not of_type(rules.evaluate(skipped), DT.STAGE_OUT_OF_PLAN)
    going = sc.context(NOW, plan_items=sc.plan({8: (TODAY - 5 * DAY, TODAY + 5 * DAY)}),
                       tl=sc.timeline({1: ("active", 0.8, TODAY - DAY)}, front=1,
                                      daily_front=[(TODAY - DAY, 1), (TODAY, 1)]), recent=recent)
    (oop,) = of_type(rules.evaluate(going), DT.STAGE_OUT_OF_PLAN)
    assert oop.stage_id == 1 and "идёт" in oop.title and oop.frame_ids
    assert_explained(oop)


def test_late_start_escalates():
    tl = sc.timeline({3: ("active", 0.9, TODAY - 60 * DAY)}, front=3, daily_front=[(TODAY, 3)])
    ctx = sc.context(NOW, plan_items=sc.plan({4: (TODAY - 20 * DAY, TODAY + 60 * DAY)}), tl=tl)
    assert of_type(rules.evaluate(ctx), DT.STAGE_LATE_START)[0].severity == Severity.CRITICAL


def test_needs_review_levels():
    assert of_type(rules.evaluate(case_needs_review()), DT.NEEDS_REVIEW)[0].severity == Severity.WARNING
    few = sc.context(NOW, tl=sc.timeline({3: ("active", 0.4)}, needs_review=[1, 2], outliers=[9]))
    devs = of_type(rules.evaluate(few), DT.NEEDS_REVIEW)
    assert {d.key for d in devs} == {"needs_review:unsure", "needs_review:outliers"}
    assert all(d.severity == Severity.INFO for d in devs)
    assert not of_type(rules.evaluate(sc.context(NOW, tl=sc.timeline({3: ("active", 0.4)}))), DT.NEEDS_REVIEW)


def test_camera_issue_cases():
    d = of_type(rules.evaluate(case_camera_issue()), DT.CAMERA_ISSUE)
    assert len(d) == 1 and d[0].severity == Severity.WARNING and d[0].camera_id == 2
    # загрузка папкой — «молчание» нормально
    folder = sc.context(NOW, recent=series(2, NOW - dt.timedelta(hours=9), 6, 20),
                        config={"cameras": [{"id": 2, "kind": "folder"}]})
    assert not of_type(rules.evaluate(folder), DT.CAMERA_ISSUE)
    # живая камера, но за сутки больше половины кадров брак
    bad = series(3, NOW - dt.timedelta(hours=2), 6, 20, quality_ok=False, reject_reason="капли на объективе")
    q = of_type(rules.evaluate(sc.context(NOW, recent=bad)), DT.CAMERA_ISSUE)
    assert len(q) == 1 and q[0].key == "camera_issue:3:quality" and "капли на объективе" in q[0].message
    fresh = sc.context(NOW, recent=series(3, NOW - dt.timedelta(hours=1), 4, 20))
    assert not of_type(rules.evaluate(fresh), DT.CAMERA_ISSUE)


def test_detections_without_ids_are_not_multiplied():
    """Без unit_id/track_id одна машина на пяти кадрах — одна машина, а не пять."""
    one = series(1, NOW - dt.timedelta(hours=2), 7, 20, lambda k: [sc.det("dump_truck", working=False)])
    devs = rules.evaluate(ctx_stage(3, recent=one))
    assert not [d for d in of_type(devs, DT.PAIR_BROKEN) if d.data["pair"] == "dump_trucks_waiting"]
    two = series(1, NOW - dt.timedelta(hours=2), 7, 20,
                 lambda k: [sc.det("dump_truck", working=False), sc.det("dump_truck", working=False, bbox=(100, 600, 200, 150))])
    waiting = [d for d in of_type(rules.evaluate(ctx_stage(3, recent=two)), DT.PAIR_BROKEN)
               if d.data["pair"] == "dump_trucks_waiting"]
    assert len(waiting) == 1 and waiting[0].severity == Severity.INFO
    # «не по этапу» без треков тоже работает: работающий каток группируется по камере и классу
    roller = series(1, NOW - dt.timedelta(hours=1), 4, 20, lambda k: [sc.det("roller")])
    assert of_type(rules.evaluate(ctx_stage(5, recent=roller)), DT.EQUIPMENT_FORBIDDEN)


def test_evaluate_orders_by_severity():
    ctx = case_late_start()
    ctx = dataclasses.replace(ctx, timeline=sc.timeline({3: ("active", 0.9, TODAY - 60 * DAY)}, front=3,
                                                        daily_front=[(TODAY, 3)], needs_review=[1]))
    ranks = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
    sev = [ranks[d.severity] for d in rules.evaluate(ctx)]
    assert sev == sorted(sev, reverse=True) and len(sev) >= 2
