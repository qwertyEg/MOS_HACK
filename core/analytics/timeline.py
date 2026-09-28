"""План против факта: ожидаемая и фактическая готовность, отставание в днях, вердикт, прогноз.

Порт идей timeline.py Никиты с исправлением известных ошибок:
- никакого «автоплана по датам фото»: план растягивался на [первое фото; последнее фото],
  на дату последнего кадра ожидалось 100 %, и вердикт всегда был «отставание»;
- частичный план не даёт ложного «опережения»: ожидаемая И фактическая готовность считаются
  по одним и тем же — запланированным — этапам (раньше факт шёл по всем 8 этапам);
- пустые даты плана пропускаются, а не падают;
- допуск «в срок» — ±3 дня (как у вердикта Ганта Дениса), а не ±7.

Готовность по плану на дату = Σ вес этапа × доля прошедших дней этапа / Σ весов (линейно
внутри этапа, веса — из checklist.json). Фактическая = Σ вес × готовность этапа по модели Б.
Отставание в днях — сдвиг, при котором плановая кривая равна факту: «сколько дней назад план
ожидал сегодняшнюю готовность». Прогноз — по темпу продвижения по плану в активных днях
(PLAN.md §3.8): темп берём по свежему окну, а долю активных дней — по всей истории, чтобы
две недели дождя не превращали нормальный темп в катастрофу и наоборот.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

from core import taxonomy
from core.analytics.context import AnalyticsConfig
from core.contracts import PlanItem, StageState, StageStatus, StageTimeline, Verdict

_EPS = 1e-9


@dataclass
class PlanFact:
    """Итог сравнения. Поля до series — договор ARCHITECTURE.md; остальные — расширение для UI и отчёта."""
    expected_progress: float | None
    actual_progress: float
    lag_days: float | None                 # > 0 — отставание
    verdict: Verdict
    forecast_finish: dt.date | None
    series: dict
    planned_finish: dt.date | None = None
    actual_finish: dt.date | None = None   # когда завершены все запланированные этапы (если завершены)
    delay_days: int | None = None          # прогноз − план, > 0 — позже плана
    pace_ratio: float | None = None        # дней плана за календарный день: 1 — по плану, 0.5 — вдвое медленнее
    partial_plan: bool = False
    planned_stages: list[int] = field(default_factory=list)
    stages: list[dict] = field(default_factory=list)
    forecast_note: str = ""


def _spans(plan: list[PlanItem]) -> dict[int, tuple[dt.date, dt.date]]:
    """Этап → (начало, окончание). Этапы без одной из дат в сравнение не входят; строки одного этапа сливаются."""
    known = taxonomy.stages()
    out: dict[int, tuple[dt.date, dt.date]] = {}
    for it in plan:
        if it.stage_id not in known or not it.planned_start or not it.planned_end:
            continue
        a, b = sorted((it.planned_start, it.planned_end))
        if it.stage_id in out:
            a, b = min(a, out[it.stage_id][0]), max(b, out[it.stage_id][1])
        out[it.stage_id] = (a, b)
    return out


def _frac(d: dt.date, a: dt.date, b: dt.date) -> float:
    """Доля этапа, которая по плану пройдена к концу дня d."""
    return min(1.0, max(0.0, ((d - a).days + 1) / ((b - a).days + 1)))


def _progress_on(st: StageState | None, d: dt.date, today: dt.date) -> float:
    """Готовность этапа на прошлую дату d, восстановленная из итогового состояния.

    Хронология модели Б монотонна, поэтому внутри этапа считаем рост линейным от фактического
    начала до окончания (или до сегодня). Этап, отмеченный готовым без дат, считаем готовым
    всегда — он не влияет на темп, но и не теряется.
    """
    if st is None or st.status == StageStatus.NOT_STARTED:
        return 0.0
    if st.status == StageStatus.DONE:
        a, b = st.actual_start or st.actual_end, st.actual_end
        if a is None:
            return 1.0
        if d < a:
            return 0.0
        if b is None or d >= b:
            return 1.0
        return _frac(d, a, b)
    p = min(1.0, max(0.0, st.progress))
    a = st.actual_start
    if a is None:
        return p
    if d < a:
        return 0.0
    return p * _frac(d, a, today) if today >= a else p


class _Curve:
    """Плановая и фактическая кривые готовности по одному набору этапов."""

    def __init__(self, spans: dict[int, tuple[dt.date, dt.date]], states: dict[int, StageState], today: dt.date):
        st = taxonomy.stages()
        self.spans, self.states, self.today = spans, states, today
        self.ids = sorted(spans) if spans else sorted(st)
        self.w = {s: st[s].weight for s in self.ids}
        self.W = sum(self.w.values()) or 1.0

    def expected(self, d: dt.date) -> float:
        return sum(self.w[s] * _frac(d, *self.spans[s]) for s in self.ids) / self.W

    def actual(self, d: dt.date) -> float:
        return sum(self.w[s] * _progress_on(self.states.get(s), d, self.today) for s in self.ids) / self.W

    def plan_day_for(self, a: float, near: dt.date) -> float:
        """Дата (дробная, ординал), когда план ожидал готовность a; из плато — ближайшая к `near`."""
        lo = min(s for s, _ in self.spans.values()).toordinal() - 1
        hi = max(e for _, e in self.spans.values()).toordinal()
        n0 = near.toordinal()
        if a <= _EPS:
            return float(min(n0, lo))
        if a >= 1 - _EPS:
            return float(hi)
        vals = [(k, self.expected(dt.date.fromordinal(k))) for k in range(lo, hi + 1)]
        plateau = [k for k, v in vals if abs(v - a) <= 1e-7]
        if plateau:
            return float(min(plateau, key=lambda k: abs(k - n0)))
        for (k0, v0), (k1, v1) in zip(vals, vals[1:]):
            if v0 < a < v1:
                return k0 + (a - v0) / (v1 - v0)
        return float(hi)


def _has_data(tl: StageTimeline) -> bool:
    return bool(tl.daily_front) or any(s.manual or s.status != StageStatus.NOT_STARTED for s in tl.states.values())


def _first_fact_day(tl: StageTimeline, ids) -> dt.date | None:
    days = [st.actual_start for s, st in tl.states.items() if s in ids and st.actual_start]
    days += [d for d, _ in tl.daily_front]
    return min(days) if days else None


def _count_active(active: set[dt.date], a: dt.date, b: dt.date) -> int:
    """Активных дней в (a, b]."""
    return sum(1 for d in active if a < d <= b)


def plan_vs_fact(plan: list[PlanItem], stage_timeline: StageTimeline, today: dt.date, *,
                 active_days: set[dt.date] | None = None, active_since: dt.date | None = None,
                 config: AnalyticsConfig | dict | None = None) -> PlanFact:
    """Сравнение плана с фактом на дату `today`.

    active_days — дни, когда техника реально работала (из журнала моточасов модели А):
    по ним считается темп «в активных днях». Без них — календарный темп.
    active_since — с какой даты журнал полон (если веб-слой передал не всю историю).
    """
    cfg = config if isinstance(config, AnalyticsConfig) else AnalyticsConfig.from_dict(config)
    spans = _spans(plan)
    curve = _Curve(spans, stage_timeline.states, today)
    actual = curve.actual(today)
    has_data = _has_data(stage_timeline)
    stages = _stage_rows(spans, stage_timeline, today)

    if not spans:
        # Без плана вердикта нет, но готовность и грубый прогноз по темпу показать можно.
        active = (active_days, active_since)
        forecast, note, _ = (_forecast_no_plan(curve, stage_timeline, today, active, cfg) if has_data
                             else (None, "нет данных модели Б — прогноз невозможен", None))
        return PlanFact(None, round(stage_timeline.overall_progress, 4) if has_data else 0.0, None,
                        Verdict.NO_PLAN, forecast, _series(curve, stage_timeline, today, None, forecast),
                        stages=stages, forecast_note=note)

    planned_finish = max(e for _, e in spans.values())
    expected = curve.expected(today)
    partial = set(spans) != set(taxonomy.stages())

    if not has_data:
        pf = PlanFact(expected, 0.0, None, Verdict.NO_DATA, None, {}, planned_finish=planned_finish,
                      partial_plan=partial, planned_stages=sorted(spans), stages=stages,
                      forecast_note="нет данных модели Б — прогноз невозможен")
        pf.series = _series(curve, stage_timeline, today, planned_finish, None)
        return pf

    actual_finish = None
    if actual >= 1 - _EPS:
        ends = [stage_timeline.states[s].actual_end for s in spans
                if s in stage_timeline.states and stage_timeline.states[s].actual_end]
        # этапы отмечены готовыми без дат — когда закончили, неизвестно; «отставания» не выдумываем
        actual_finish = max(ends) if ends else min(today, planned_finish)
        lag = float((actual_finish - planned_finish).days)
    else:
        lag = today.toordinal() - curve.plan_day_for(actual, today)
    lag = round(lag, 1)
    if abs(lag) <= cfg.schedule_tolerance_days:
        verdict = Verdict.ON_TRACK
    elif lag > 0:
        verdict = Verdict.BEHIND
    else:
        verdict = Verdict.AHEAD

    forecast, note, ratio = _forecast_with_plan(curve, stage_timeline, today, actual, planned_finish,
                                                actual_finish, (active_days, active_since), cfg)
    return PlanFact(
        expected_progress=round(expected, 4), actual_progress=round(actual, 4), lag_days=lag, verdict=verdict,
        forecast_finish=forecast, series=_series(curve, stage_timeline, today, planned_finish, forecast),
        planned_finish=planned_finish, actual_finish=actual_finish,
        delay_days=(forecast - planned_finish).days if forecast else None,
        pace_ratio=round(ratio, 2) if ratio else None, partial_plan=partial, planned_stages=sorted(spans),
        stages=stages, forecast_note=note,
    )


def _pace_window(tl: StageTimeline, ids, today: dt.date, cfg: AnalyticsConfig):
    first = _first_fact_day(tl, ids)
    if first is None:
        return None, None, "нет истории факта — прогноз невозможен"
    t0 = max(today - dt.timedelta(days=cfg.pace_window_days), first)
    if (today - t0).days < cfg.min_pace_days:
        return None, first, f"истории меньше {cfg.min_pace_days} дн. — прогноз пока не даём"
    return t0, first, ""


def _active_scale(active_days, active_since, t0, first, today) -> tuple[float, float, str]:
    """(активных дней в окне, календарных на один активный за всю историю, пометка).

    Активные дни берутся из журнала моточасов. Если журнал начинается позже окна темпа
    (веб-слой передал только свежие интервалы), активные дни занижены — тогда честнее
    календарный темп, чем прогноз «через 30 лет».
    """
    span = (today - t0).days
    if not active_days:
        return float(span), 1.0, "темп по календарным дням"
    since = active_since or min(active_days)
    if since > t0:
        return float(span), 1.0, "журнал моточасов короче окна темпа — темп по календарным дням"
    n_act = _count_active(active_days, t0, today)
    hist_from = max(first, since)
    hist_act = _count_active(active_days, hist_from - dt.timedelta(days=1), today)
    if n_act == 0 or hist_act == 0:
        return float(span), 1.0, "активных дней техники в окне нет — темп по календарным дням"
    k_idle = max(1.0, ((today - hist_from).days + 1) / hist_act)
    return float(n_act), k_idle, f"темп по активным дням ({n_act} из {span}), доля простоя — по всей истории"


def _forecast_with_plan(curve: _Curve, tl, today, actual, planned_finish, actual_finish, active, cfg):
    if actual_finish is not None:
        return actual_finish, "все запланированные этапы завершены", None
    t0, first, note = _pace_window(tl, curve.ids, today, cfg)
    if t0 is None:
        return None, note, None
    a0 = curve.actual(t0)
    span = (today - t0).days
    # Прирост факта проверяем напрямую: на плато плановой кривой (разрыв между этапами)
    # одна и та же готовность соответствует многим датам, и «продвижение» получилось бы из ничего.
    advanced = curve.plan_day_for(actual, today) - curve.plan_day_for(a0, t0) if actual - a0 > _EPS else 0.0
    if advanced <= _EPS:
        return None, f"за последние {span} дн. продвижения по плану нет — прогноз невозможен (работы стоят)", None
    n_act, k_idle, how = _active_scale(*active, t0, first, today)
    ratio = advanced / n_act / k_idle          # дней плана за календарный день
    ratio = min(max(ratio, 0.2), 5.0)          # один выброс не должен дать прогноз через 30 лет
    remaining = max(0.0, planned_finish.toordinal() - curve.plan_day_for(actual, today))
    finish = today + dt.timedelta(days=math.ceil(remaining / ratio))
    short = "; окно меньше месяца — прогноз грубый" if span < 28 else ""
    return finish, how + short, ratio


def _forecast_no_plan(curve: _Curve, tl, today, active, cfg):
    t0, first, note = _pace_window(tl, curve.ids, today, cfg)
    if t0 is None:
        return None, note, None
    a1, a0 = curve.actual(today), curve.actual(t0)
    if a1 >= 1 - _EPS:
        return today, "объект готов", None
    n_act, k_idle, how = _active_scale(*active, t0, first, today)
    v = (a1 - a0) / n_act
    if v <= _EPS:
        return None, "прогресса за окно нет — прогноз невозможен", None
    days = (1 - a1) / v * k_idle
    if days > 3650:
        return None, "темп слишком мал для прогноза", None
    # без плана темп переносится линейно — ранние этапы лёгкие по весу, прогноз занижен
    return today + dt.timedelta(days=math.ceil(days)), how + "; плана нет — экстраполяция линейная, грубая", None


def _series(curve: _Curve, tl: StageTimeline, today: dt.date, planned_finish, forecast) -> dict:
    """Точки для графика «готовность: план и факт» (доли 0..1; факт — только до сегодня)."""
    starts = [a for a, _ in curve.spans.values()]
    first = _first_fact_day(tl, curve.ids)
    lo = min([d for d in starts + [first, today] if d])
    hi = max([d for d in [planned_finish, today] if d])
    span = (hi - lo).days
    step = max(1, math.ceil(span / 366))
    days = [lo + dt.timedelta(days=i) for i in range(0, span + 1, step)]
    for d in (today, hi):
        if d not in days:
            days.append(d)
    days.sort()
    return {
        "days": [d.isoformat() for d in days],
        "expected": [round(curve.expected(d), 4) for d in days] if curve.spans else None,
        "actual": [round(curve.actual(d), 4) if d <= today else None for d in days],
        "today": today.isoformat(),
        "planned_finish": planned_finish.isoformat() if planned_finish else None,
        "forecast_finish": forecast.isoformat() if forecast else None,
    }


def _stage_rows(spans, tl: StageTimeline, today: dt.date) -> list[dict]:
    """Таблица этапов план/факт для UI (даты — объекты date)."""
    rows = []
    for s, stage in sorted(taxonomy.stages().items()):
        st = tl.states.get(s)
        ps, pe = spans.get(s, (None, None))
        planned_status = None
        if ps and pe:
            planned_status = "not_started" if today < ps else "done" if today > pe else "in_progress"
        rows.append({
            "stage_id": s, "name": stage.name, "weight": stage.weight,
            "planned_start": ps, "planned_end": pe, "planned_status": planned_status,
            "status": (st.status if st else StageStatus.NOT_STARTED).value,
            "progress": round(st.progress, 3) if st else 0.0,
            "actual_start": st.actual_start if st else None, "actual_end": st.actual_end if st else None,
            "manual": bool(st and st.manual),
            "start_shift_days": (st.actual_start - ps).days if st and st.actual_start and ps else None,
            "end_shift_days": (st.actual_end - pe).days if st and st.actual_end and pe else None,
        })
    return rows
