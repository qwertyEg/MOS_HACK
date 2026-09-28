"""SiteReport: вердикт, отставание, прогноз и объяснение «почему» (2–5 осмысленных фраз)."""
import dataclasses
import datetime as dt

from core.analytics import report
from core.analytics import scenarios as sc
from core.analytics.context import MSK
from core.contracts import DeviationType, Verdict

NOW = dt.datetime(2026, 9, 28, 15, 0, tzinfo=MSK)
TODAY = NOW.date()
DAY = dt.timedelta(days=1)


def test_tz_reference_report_explains_verdict_and_main_deviation():
    ctx = sc.tz_reference()
    r, pf = report.build_full(ctx)
    assert r.verdict == pf.verdict == Verdict.ON_TRACK
    assert r.current_stage == 3 and r.stage_states is ctx.timeline.states and r.hours == ctx.balances
    assert r.deviations[0].type == DeviationType.PAIR_BROKEN
    assert 2 <= len(r.explanation) <= 5
    text = " ".join(r.explanation)
    assert "по графику" in r.explanation[0] and "%" in r.explanation[0] and "28.09.2026" in r.explanation[0]
    assert "«Земляные работы, котлован»" in text
    assert "экскаватор работает без самосвалов 2 ч 10 мин" in r.explanation[-1]
    assert pf.series["days"] and r.forecast_finish == pf.forecast_finish


def test_behind_report_names_lag_and_overdue_stage():
    tl = sc.timeline({1: ("done", 1.0), 2: ("done", 1.0), 3: ("active", 0.5, TODAY - 60 * DAY, None, [5, 6])},
                     front=3, daily_front=[(TODAY - i * DAY, 3) for i in range(60, -1, -1)])
    ctx = sc.context(NOW, plan_items=sc.excavation_plan(TODAY - 40 * DAY), tl=tl)
    r = report.build(ctx)
    assert r.verdict == Verdict.BEHIND and r.lag_days > 3
    assert r.explanation[0].startswith("Отставание от графика ≈")
    assert any(d.type == DeviationType.STAGE_OVERDUE for d in r.deviations)
    assert "Главное отклонение" in r.explanation[-1]


def test_no_plan_report_asks_for_plan():
    tl = sc.timeline({3: ("active", 0.4, TODAY - 5 * DAY)}, front=3, daily_front=[(TODAY, 3)])
    r = report.build(sc.context(NOW, tl=tl))
    assert r.verdict == Verdict.NO_PLAN and r.expected_progress is None
    assert "плана нет" in r.explanation[0] and "демо-план" in r.explanation[0]
    assert r.actual_progress == round(tl.overall_progress, 4)
    assert 2 <= len(r.explanation) <= 5


def test_no_data_report_says_why():
    r = report.build(sc.context(NOW, plan_items=sc.excavation_plan(TODAY)))
    assert r.verdict == Verdict.NO_DATA
    assert "данных по этапам ещё нет" in r.explanation[0]
    assert r.explanation[-1] == "Отклонений не выявлено." or r.deviations


def test_unsure_only_report_is_not_called_no_data():
    """Модель Б кадры разобрала, но всё «не уверен»: объяснение не должно говорить «не разобрала ни одного»."""
    r = report.build(sc.context(NOW, plan_items=sc.excavation_plan(TODAY), config={"stage_observations": 7}))
    assert r.verdict == Verdict.NO_DATA
    assert "не определён" in r.explanation[0] and "7" in r.explanation[0]
    assert "ни одного дневного кадра" not in r.explanation[0]


def test_partial_plan_progress_is_comparable():
    """С частичным планом «план» и «факт» в отчёте считаются по одним этапам — и объяснение это говорит."""
    tl = sc.timeline({1: ("done", 1.0), 2: ("done", 1.0), 3: ("active", 0.5, TODAY - 10 * DAY)}, front=3,
                     daily_front=[(TODAY, 3)])
    ctx = sc.context(NOW, plan_items=sc.plan({3: (TODAY - 10 * DAY, TODAY + 9 * DAY)}), tl=tl)
    r = report.build(ctx)
    assert r.expected_progress == 0.55 and r.actual_progress == 0.5
    assert "только этапы из плана: 3" in r.explanation[0]
    assert r.verdict == Verdict.ON_TRACK


def test_explanation_mentions_forecast_delay():
    ctx = sc.tz_reference()
    slow = dataclasses.replace(ctx, timeline=sc.timeline(
        {1: ("done", 1.0, TODAY - 150 * DAY, TODAY - 90 * DAY), 2: ("done", 1.0, TODAY - 100 * DAY, TODAY - 30 * DAY),
         3: ("active", 0.1, TODAY - 25 * DAY, None, [900])},
        front=3, daily_front=[(TODAY - i * DAY, 3) for i in range(25, -1, -1)]))
    r = report.build(slow)
    assert r.verdict == Verdict.BEHIND
    assert any(p.startswith("Прогноз окончания") and "позже плана" in p for p in r.explanation)
