"""Моточасы: план по парку и датам, списание по интервалам, «полоски»."""
from __future__ import annotations

import datetime as dt
import random

import pytest

from core.contracts import ActivityInterval, PlanItem
from core.equipment import hours

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 9, 28, 8, 0, tzinfo=UTC)


def iv(start, minutes, cls="excavator", unit="u0001", stage=None):
    return ActivityInterval(unit, cls, start, start + dt.timedelta(minutes=minutes), minutes / 60, stage)


def test_credit_is_capped_by_max_gap():
    assert hours.credit_hours(T0, T0 + dt.timedelta(minutes=25)) == pytest.approx(25 / 60)
    assert hours.credit_hours(T0, T0 + dt.timedelta(hours=5)) == pytest.approx(0.75)
    assert hours.credit_hours(T0, T0) == 0.0
    assert hours.credit_hours(T0 + dt.timedelta(minutes=5), T0) == 0.0


def test_workdays_counts_mon_to_sat():
    # 2026-09-28 — понедельник; две полные недели = 12 рабочих дней пн–сб
    assert hours.workdays_in(dt.date(2026, 9, 28), dt.date(2026, 10, 11)) == 12
    assert hours.workdays_in(dt.date(2026, 10, 4), dt.date(2026, 10, 4)) == 0     # воскресенье
    assert hours.workdays_in(dt.date(2026, 10, 5), dt.date(2026, 10, 1)) == 0     # конец раньше начала
    rnd = random.Random(1)
    for _ in range(50):
        a = dt.date(2026, 1, 1) + dt.timedelta(days=rnd.randrange(300))
        b = a + dt.timedelta(days=rnd.randrange(40))
        brute = sum(1 for k in range((b - a).days + 1) if (a + dt.timedelta(days=k)).weekday() < 6)
        assert hours.workdays_in(a, b) == brute


def test_planned_hours_from_dates_and_fleet():
    """Котлован пн–вс (6 рабочих дней), на площадке 2 экскаватора и 4 самосвала."""
    plan = [PlanItem(3, dt.date(2026, 9, 28), dt.date(2026, 10, 4))]
    got = hours.planned_hours(plan, {"excavator": 2, "dump_truck": 4, "tower_crane": 1})
    assert got[3]["excavator"] == pytest.approx(2 * 6 * 10 * 0.7)
    assert got[3]["dump_truck"] == pytest.approx(4 * 6 * 10 * 0.7)
    assert "tower_crane" not in got[3], "парк общий на площадку: кран котловану не нужен и в план не идёт"


def test_plan_row_equipment_beats_fleet_and_adds_optional():
    plan = [PlanItem(3, dt.date(2026, 9, 28), dt.date(2026, 9, 30), equipment={"excavator": 1, "bulldozer": 1})]
    got = hours.planned_hours(plan, {"excavator": 3}, shift_hours=8, utilization=0.5)
    assert got[3]["excavator"] == pytest.approx(1 * 3 * 8 * 0.5)
    assert got[3]["bulldozer"] == pytest.approx(12.0)


def test_manual_plan_hours_take_priority():
    plan = [PlanItem(3, dt.date(2026, 9, 28), dt.date(2026, 10, 4),
                     planned_hours={"excavator": 16.0, "mobile_crane": 28.0}, hours_manual=True)]
    got = hours.planned_hours(plan, {"excavator": 5})
    assert got[3]["excavator"] == 16.0
    assert got[3]["mobile_crane"] == 28.0
    # парк этапа по нормам core.plan.norms (после слияния модулей — 4 самосвала на котловане)
    n = hours._default_units(3)["dump_truck"]
    assert got[3]["dump_truck"] == pytest.approx(n * 6 * 10 * 0.7), "что не правили руками — по формуле"


def test_stage_without_dates_has_no_computed_hours():
    assert hours.planned_hours([PlanItem(4, None, None)], {"concrete_mixer": 2}) == {4: {}}


def test_stage_for_prefers_stage_that_needs_the_machine():
    plan = [PlanItem(3, dt.date(2026, 9, 1), dt.date(2026, 10, 20)),
            PlanItem(4, dt.date(2026, 9, 25), dt.date(2026, 11, 30))]
    day = dt.date(2026, 10, 1)
    assert hours.stage_for(plan, "excavator", day) == 3          # обязателен на котловане
    assert hours.stage_for(plan, "concrete_pump", day) == 4      # обязателен на монолите
    assert hours.stage_for(plan, "excavator", dt.date(2026, 11, 5)) == 4   # котлован кончился
    assert hours.stage_for(plan, "excavator", dt.date(2026, 12, 5)) is None


def test_balances_three_intervals_of_25_minutes_is_1_25_hours():
    plan = [PlanItem(3, dt.date(2026, 9, 28), dt.date(2026, 10, 4), planned_hours={"excavator": 16.0})]
    ivs = [iv(T0 + dt.timedelta(minutes=25 * k), 25) for k in range(3)]
    (b,) = [b for b in hours.balances(plan, ivs) if b.cls == "excavator"]
    assert b.stage_id == 3
    assert b.worked_hours == pytest.approx(1.25)
    assert b.remaining_hours == pytest.approx(14.75)
    assert b.last_worked_at == T0 + dt.timedelta(minutes=75)


def test_balances_follow_edited_plan_dates():
    """Интервал записан на этап 3, а пользователь сдвинул даты — полоска пересчитывается по новому плану."""
    ivs = [iv(T0, 30, stage=3)]
    plan = [PlanItem(3, dt.date(2026, 10, 5), dt.date(2026, 10, 20), planned_hours={"excavator": 10.0}),
            PlanItem(8, dt.date(2026, 9, 1), dt.date(2026, 9, 30), planned_hours={"excavator": 5.0})]
    rows = {(b.stage_id, b.cls): b for b in hours.balances(plan, ivs)}
    assert rows[(8, "excavator")].worked_hours == pytest.approx(0.5)
    assert rows[(3, "excavator")].worked_hours == 0.0


def test_balances_outside_plan_go_to_none_stage_and_sort_last():
    plan = [PlanItem(3, dt.date(2026, 9, 1), dt.date(2026, 9, 10), planned_hours={"excavator": 10.0})]
    rows = hours.balances(plan, [iv(T0, 30, cls="bulldozer")])
    assert rows[-1].stage_id is None and rows[-1].cls == "bulldozer"
    assert rows[0].stage_id == 3 and rows[0].worked_hours == 0.0


def test_balances_compute_planned_when_row_is_empty():
    plan = [PlanItem(3, dt.date(2026, 9, 28), dt.date(2026, 9, 28))]
    rows = {(b.stage_id, b.cls): b for b in hours.balances(plan, [], fleet={"excavator": 1})}
    assert rows[(3, "excavator")].planned_hours == pytest.approx(7.0)


def test_interval_day_is_moscow_day():
    """22:30 UTC — это уже следующий день в Москве: часы идут на этап, который идёт тогда."""
    t = dt.datetime(2026, 9, 30, 22, 30, tzinfo=UTC)
    assert hours.local_date(t) == dt.date(2026, 10, 1)
    plan = [PlanItem(3, dt.date(2026, 9, 1), dt.date(2026, 9, 30), planned_hours={"excavator": 10.0}),
            PlanItem(4, dt.date(2026, 10, 1), dt.date(2026, 10, 30), planned_hours={"excavator": 10.0})]
    rows = {(b.stage_id, b.cls): b for b in hours.balances(plan, [iv(t, 20)])}
    assert rows[(4, "excavator")].worked_hours > 0 and rows[(3, "excavator")].worked_hours == 0


def test_interval_set_counts_overlap_once():
    s = hours.IntervalSet()
    m = lambda k: T0 + dt.timedelta(minutes=k)   # noqa: E731
    assert s.add(m(0), m(25)) == [(m(0), m(25))]
    assert s.add(m(10), m(35)) == [(m(25), m(35))]      # вторая камера — только новые 10 минут
    assert s.add(m(5), m(20)) == []
    assert s.add(m(-10), m(60)) == [(m(-10), m(0)), (m(35), m(60))]


def test_interval_set_matches_brute_force():
    rnd = random.Random(7)
    s, covered, fresh_total = hours.IntervalSet(), set(), 0
    for _ in range(300):
        a = rnd.randrange(0, 500)
        b = a + rnd.randrange(1, 60)
        fresh = s.add(T0 + dt.timedelta(minutes=a), T0 + dt.timedelta(minutes=b))
        fresh_total += sum((e - st).total_seconds() / 60 for st, e in fresh)
        covered |= set(range(a, b))
        assert fresh_total == len(covered)


def test_interval_set_floor_blocks_everything_before_restart():
    s = hours.IntervalSet(floor=T0)
    assert s.add(T0 - dt.timedelta(hours=1), T0 + dt.timedelta(minutes=10)) == [(T0, T0 + dt.timedelta(minutes=10))]


# --------------------------------------------------------------------------
# ожидаемое к «сейчас» — от начала наблюдения (требование 5)
# --------------------------------------------------------------------------

MSK = dt.timezone(dt.timedelta(hours=3))


def msk(y, mo, d, h=0, mi=0):
    return dt.datetime(y, mo, d, h, mi, tzinfo=MSK)


def test_shift_fraction_counts_working_shifts_only():
    kw = {"tz": "Europe/Moscow", "shift_start_h": 8.0, "shift_hours": 10.0}
    # понедельник 20.04.2015: смена 08–18 целиком, ночь не в счёт
    assert hours.shift_fraction(msk(2015, 4, 20, 6), msk(2015, 4, 21, 6), **kw) == pytest.approx(1.0)
    # камера начала в 13:00 — половина смены
    assert hours.shift_fraction(msk(2015, 4, 20, 13), msk(2015, 4, 20, 23), **kw) == pytest.approx(0.5)
    # воскресенье 26.04 — выходной
    assert hours.shift_fraction(msk(2015, 4, 26, 8), msk(2015, 4, 26, 18), **kw) == 0.0
    # неделя пн–вс = 6 смен, как в знаменателе плановых часов
    assert hours.shift_fraction(msk(2015, 4, 20), msk(2015, 4, 27), **kw) == pytest.approx(6.0)
    assert hours.shift_fraction(msk(2015, 4, 21), msk(2015, 4, 20), **kw) == 0.0


def test_expected_counts_from_first_frame_not_from_stage_start():
    """Песчаный карьер (Киров): этап 3 по плану 13–30.04.2015, камера снимает 20.04 08:01 – 21.04 15:24.
    От начала этапа ожидалось бы ~54 ч, и экскаватор с 15.5 ч был бы «сильно отстаёт».
    От начала съёмки — 1.74 смены × 7 ч = 12.2 ч: экскаватор в норме."""
    plan = [PlanItem(3, dt.date(2015, 4, 13), dt.date(2015, 4, 30),
                     planned_hours={"excavator": 112.0, "dump_truck": 224.0})]
    first, now = msk(2015, 4, 20, 8, 1), msk(2015, 4, 21, 15, 24)
    rows = {b.cls: b for b in hours.balances(plan, [], now=now, observed_from=first)}
    ex = rows["excavator"]
    assert ex.expected_from == first.astimezone(UTC)
    shifts = (18 - (8 + 1 / 60)) / 10 + (15.4 - 8) / 10
    assert ex.expected_hours == pytest.approx(112.0 / 16 * shifts, abs=0.05)   # 16 рабочих дней этапа
    assert 12.0 < ex.expected_hours < 12.4
    # на наблюдаемую часть этапа (20–30.04, 10 рабочих дней) план отводит 70 ч
    assert ex.planned_observed_hours == pytest.approx(70.0, abs=0.05)
    assert rows["dump_truck"].expected_hours == pytest.approx(2 * ex.expected_hours, abs=0.05)
    # без первого кадра — от начала этапа, как раньше: 6 смен первой недели + 1.74
    legacy = {b.cls: b for b in hours.balances(plan, [], now=now)}["excavator"]
    assert legacy.expected_hours == pytest.approx(7.0 * (6 + shifts), abs=0.05)
    assert legacy.planned_observed_hours == pytest.approx(112.0)


def test_expected_equals_plan_after_stage_end_and_zero_before_start():
    plan = [PlanItem(3, dt.date(2015, 4, 13), dt.date(2015, 4, 18), planned_hours={"excavator": 42.0})]
    after = {b.cls: b for b in hours.balances(plan, [], now=msk(2015, 5, 10), observed_from=msk(2015, 4, 1))}
    assert after["excavator"].expected_hours == pytest.approx(42.0)
    before = {b.cls: b for b in hours.balances(plan, [], now=msk(2015, 4, 10), observed_from=msk(2015, 4, 1))}
    assert before["excavator"].expected_hours == 0.0
    # камеры повесили после конца этапа — этап прошёл до них: ни ожидания, ни плана на период съёмки
    late = {b.cls: b for b in hours.balances(plan, [], now=msk(2015, 5, 10), observed_from=msk(2015, 4, 25))}
    assert late["excavator"].expected_hours == 0.0 and late["excavator"].planned_observed_hours == 0.0


def test_expected_absent_without_now_or_dates():
    plan = [PlanItem(3, None, None, planned_hours={"excavator": 42.0}),
            PlanItem(4, dt.date(2015, 4, 13), dt.date(2015, 4, 18), planned_hours={"tower_crane": 10.0})]
    rows = {b.cls: b for b in hours.balances(plan, [iv(T0, 30, cls="bulldozer")])}
    assert rows["excavator"].expected_hours is None
    assert rows["tower_crane"].expected_hours is None       # без «сейчас» не считаем
    rows = {b.cls: b for b in hours.balances(plan, [iv(T0, 30, cls="bulldozer")], now=msk(2015, 4, 15))}
    assert rows["excavator"].expected_hours is None         # у этапа нет дат
    assert rows["bulldozer"].expected_hours is None         # не по плану
    assert rows["tower_crane"].expected_hours == pytest.approx(10.0 / 6 * 2, abs=0.01)
