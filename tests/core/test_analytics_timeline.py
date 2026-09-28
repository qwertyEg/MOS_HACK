"""План/факт: ожидаемая готовность, отставание в днях, вердикт, прогноз, частичный план."""
import datetime as dt

import pytest

from core.analytics import scenarios as sc
from core.analytics.timeline import plan_vs_fact
from core.contracts import PlanItem, StageStatus, StageTimeline, Verdict
from core.plan.importer import demo_plan

START = dt.date(2026, 3, 2)
DAY = dt.timedelta(days=1)
PLAN = demo_plan(START)                                 # 8 этапов с перекрытиями, 510 дней
SPANS = {it.stage_id: (it.planned_start, it.planned_end) for it in PLAN}
FINISH = max(e for _, e in SPANS.values())


def facts_like_plan_at(x: dt.date, *, pace: float = 1.0) -> StageTimeline:
    """Факт, равный плану на дату x. pace < 1 — стройка всё время шла медленнее плана в 1/pace раз."""
    states = {}
    for s, (a, b) in SPANS.items():
        frac = min(1.0, max(0.0, ((x - a).days + 1) / ((b - a).days + 1)))
        act_a = START + (a - START) / pace
        act_b = START + (b - START) / pace
        if frac >= 1:
            states[s] = ("done", 1.0, act_a, act_b)
        elif frac > 0:
            states[s] = ("active", frac, act_a)
    front = max((s for s, v in states.items() if v[0] == "active"), default=None)
    return sc.timeline(states, front=front, daily_front=[(x, front or 1)])


def test_expected_progress_is_linear_within_stage():
    plan = [PlanItem(3, START, START + 9 * DAY)]
    tl = sc.timeline({3: ("active", 0.5, START)}, front=3)
    assert plan_vs_fact(plan, tl, START + 4 * DAY).expected_progress == pytest.approx(0.5)
    assert plan_vs_fact(plan, tl, START - DAY).expected_progress == 0.0
    assert plan_vs_fact(plan, tl, START + 30 * DAY).expected_progress == 1.0


@pytest.mark.parametrize("shift, verdict", [(10, Verdict.BEHIND), (-10, Verdict.AHEAD), (2, Verdict.ON_TRACK),
                                            (0, Verdict.ON_TRACK)])
def test_lag_days_and_verdict(shift, verdict):
    today = START + 200 * DAY
    pf = plan_vs_fact(PLAN, facts_like_plan_at(today - shift * DAY), today)
    assert pf.lag_days == pytest.approx(shift, abs=1.0)
    assert pf.verdict == verdict
    assert pf.lag_days > 0 if verdict == Verdict.BEHIND else True


def test_partial_plan_gives_no_false_ahead():
    """Баг api-solution: в плане только этап 5, на кадрах котлован → было «опережение −148 дн.»."""
    today = START + 60 * DAY
    tl = sc.timeline({1: ("done", 1.0), 2: ("done", 1.0), 3: ("active", 0.5, today - 20 * DAY)}, front=3,
                     daily_front=[(today, 3)])
    future5 = [PlanItem(5, today + 30 * DAY, today + 270 * DAY)]
    pf = plan_vs_fact(future5, tl, today)
    assert pf.verdict == Verdict.ON_TRACK and pf.expected_progress == 0 and pf.actual_progress == 0
    assert pf.partial_plan and pf.planned_stages == [5]
    running5 = [PlanItem(5, today - 30 * DAY, today + 210 * DAY)]
    pf = plan_vs_fact(running5, tl, today)
    assert pf.verdict == Verdict.BEHIND and pf.lag_days >= 29


def test_forecast_by_pace_relative_to_plan():
    today = START + 240 * DAY
    on_plan = plan_vs_fact(PLAN, facts_like_plan_at(today), today)
    assert on_plan.pace_ratio == pytest.approx(1.0, abs=0.05)
    assert abs((on_plan.forecast_finish - FINISH).days) <= 3
    slow = plan_vs_fact(PLAN, facts_like_plan_at(START + 120 * DAY, pace=0.5), today)
    assert slow.verdict == Verdict.BEHIND
    assert slow.pace_ratio == pytest.approx(0.5, abs=0.07)
    remaining = (FINISH - (START + 120 * DAY)).days
    assert slow.forecast_finish - today == pytest.approx(dt.timedelta(days=remaining / 0.5), abs=dt.timedelta(days=25))
    assert slow.delay_days > 200


def test_no_progress_means_no_forecast():
    today = SPANS[3][1]                                  # котлован по плану должен закончиться, а он не начат
    tl = sc.timeline({1: ("done", 1.0, START, START + 29 * DAY), 2: ("done", 1.0, START + 20 * DAY, START + 50 * DAY)},
                     front=2, daily_front=[(today - i * DAY, 2) for i in range(30, -1, -1)])
    pf = plan_vs_fact(PLAN, tl, today)
    assert pf.forecast_finish is None and "продвижения по плану нет" in pf.forecast_note
    assert pf.verdict == Verdict.BEHIND


def test_short_history_gives_no_forecast():
    today = START + 3 * DAY
    tl = sc.timeline({1: ("active", 0.1, START + DAY)}, front=1, daily_front=[(START + DAY, 1)])
    pf = plan_vs_fact(PLAN, tl, today)
    assert pf.forecast_finish is None and "меньше 7" in pf.forecast_note


def test_active_days_pace_matches_calendar_when_idle_share_is_uniform():
    today = START + 240 * DAY
    tl = facts_like_plan_at(today)
    active = {START + i * DAY for i in range(0, 241) if (START + i * DAY).weekday() != 6}
    cal = plan_vs_fact(PLAN, tl, today)
    act = plan_vs_fact(PLAN, tl, today, active_days=active, active_since=START)
    assert abs((act.forecast_finish - cal.forecast_finish).days) <= 20
    assert "активным дням" in act.forecast_note
    # журнал моточасов короче окна темпа — честный откат на календарный темп
    short = plan_vs_fact(PLAN, tl, today, active_days={today - DAY, today}, active_since=today - DAY)
    assert "журнал" in short.forecast_note and short.forecast_finish == cal.forecast_finish


def test_no_plan_and_no_data_verdicts():
    tl = sc.timeline({3: ("active", 0.5, START)}, front=3, daily_front=[(START, 3)])
    pf = plan_vs_fact([], tl, START + 10 * DAY)
    assert pf.verdict == Verdict.NO_PLAN and pf.expected_progress is None and pf.lag_days is None
    pf = plan_vs_fact(PLAN, StageTimeline({}, None, 0.0, []), START + 10 * DAY)
    assert pf.verdict == Verdict.NO_DATA and pf.lag_days is None and pf.expected_progress > 0


def test_all_planned_stages_finished():
    done = sc.timeline({s: ("done", 1.0, a, b + 5 * DAY) for s, (a, b) in SPANS.items()}, front=8,
                       daily_front=[(FINISH, 8)])
    pf = plan_vs_fact(PLAN, done, FINISH + 30 * DAY)
    assert pf.actual_finish == FINISH + 5 * DAY
    assert pf.lag_days == 5 and pf.verdict == Verdict.BEHIND and pf.forecast_finish == pf.actual_finish


def test_plan_rows_without_dates_are_ignored():
    plan = PLAN + [PlanItem(3, None, None), PlanItem(9, START, START + DAY)]
    pf = plan_vs_fact(plan, facts_like_plan_at(START + 100 * DAY), START + 100 * DAY)
    assert pf.planned_stages == list(range(1, 9)) and pf.verdict == Verdict.ON_TRACK


def test_series_for_chart():
    today = START + 200 * DAY
    pf = plan_vs_fact(PLAN, facts_like_plan_at(today - 10 * DAY), today)
    s = pf.series
    assert len(s["days"]) == len(s["expected"]) == len(s["actual"])
    assert today.isoformat() in s["days"] and s["planned_finish"] == FINISH.isoformat()
    assert s["expected"] == sorted(s["expected"])            # плановая кривая не убывает
    after = [a for d, a in zip(s["days"], s["actual"]) if d > today.isoformat()]
    assert after and all(a is None for a in after)
    stages = {r["stage_id"]: r for r in pf.stages}
    assert stages[5]["planned_status"] == "in_progress" and stages[5]["status"] == StageStatus.ACTIVE.value
