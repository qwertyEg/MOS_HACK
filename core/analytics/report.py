"""SiteReport для дашборда: вердикт, отставание, прогноз, отклонения и объяснение «почему».

Объяснение — 2–5 коротких фраз для руководителя стройки: что по плану, что видим,
чем это грозит и какое отклонение главное. Каждая фраза опирается на число или дату,
которые UI показывает рядом, — чтобы вердикт можно было проверить, а не принять на веру.
"""
from __future__ import annotations

import datetime as dt

from core import taxonomy
from core.analytics import fmt, rules
from core.analytics.context import AnalyticsContext
from core.analytics.timeline import PlanFact, plan_vs_fact
from core.contracts import DeviationRecord, Severity, SiteReport, StageStatus, Verdict
from core.plan import catalog


def active_days(ctx: AnalyticsContext) -> set[dt.date]:
    """Дни, когда техника реально работала (есть списанные моточасы) — для темпа «в активных днях».

    Ночные смены входят: работа ночью — тоже работа (PLAN.md §3.3.1).
    """
    return {ctx.local(i.start).date() for i in ctx.intervals if i.hours > 0}


def compute_planfact(ctx: AnalyticsContext) -> PlanFact:
    """plan_vs_fact по контексту: журнал моточасов считается полным с первой его записи
    (или с даты `hours_log_since` в настройках, если веб-слой передал не всю историю)."""
    days = active_days(ctx)
    since = ctx.config.get("hours_log_since")
    if isinstance(since, str):
        since = dt.date.fromisoformat(since[:10])
    return plan_vs_fact(ctx.plan, ctx.timeline, ctx.today, active_days=days or None,
                        active_since=since or (min(days) if days else None), config=ctx.cfg)


def build_full(ctx: AnalyticsContext) -> tuple[SiteReport, PlanFact]:
    """Отчёт и подробности план/факт (series для графика, таблица этапов) одним вызовом."""
    pf = compute_planfact(ctx)
    return build(ctx, planfact=pf), pf


def build(ctx: AnalyticsContext, planfact: PlanFact | None = None) -> SiteReport:
    pf = planfact or compute_planfact(ctx)
    devs = rules.evaluate(ctx)
    # С планом «факт» — по тем же этапам, что и «план», иначе проценты несравнимы (частичный план).
    actual = pf.actual_progress if pf.expected_progress is not None else ctx.timeline.overall_progress
    return SiteReport(
        verdict=pf.verdict, lag_days=pf.lag_days, expected_progress=pf.expected_progress,
        actual_progress=round(actual, 4), forecast_finish=pf.forecast_finish,
        current_stage=ctx.timeline.current_stage, stage_states=ctx.timeline.states,
        hours=list(ctx.balances), deviations=devs, explanation=explain(ctx, pf, devs),
    )


def _verdict_phrase(ctx: AnalyticsContext, pf: PlanFact) -> str:
    as_of = getattr(pf, "fact_as_of", None)
    today = fmt.date(as_of or ctx.today)
    exp, act = fmt.pct(pf.expected_progress), fmt.pct(pf.actual_progress)
    tol = int(ctx.cfg.schedule_tolerance_days)
    scope = (f" (считаются только этапы из плана: {', '.join(map(str, pf.planned_stages))})"
             if pf.partial_plan and pf.planned_stages else "")
    if pf.verdict == Verdict.NO_PLAN:
        return (f"Календарного плана нет — вердикт не выносится; по снимкам готовность объекта "
                f"{fmt.pct(ctx.timeline.overall_progress)}. Загрузите график (CSV/XLSX) или создайте демо-план.")
    if pf.verdict == Verdict.NO_DATA:
        # Модель Б кадры разбирала, но ни на одном не ответила уверенно (всё «не уверен»):
        # это не «нет данных», а «не определить» — пользователю важно различать (отчёт UI, Киров).
        asked = int(ctx.config.get("stage_observations") or 0) or len(ctx.timeline.needs_review or [])
        if asked:
            return (f"По плану на {today} должно быть готово {exp}, но этап по снимкам не определён: модель Б "
                    f"разобрала {asked} кадр(ов) и ни на одном не ответила на признаки чек-листа уверенно "
                    "(ответы «не уверен» не голосуют). Проверьте кадры вручную, отметьте этап или "
                    "переключите модель Б на внешний API.")
        return (f"По плану на {today} должно быть готово {exp}, но данных по этапам ещё нет: модель Б не разобрала "
                "ни одного дневного кадра — вердикт появится после анализа снимков.")
    if pf.actual_finish is not None:
        how = "позже" if pf.lag_days and pf.lag_days > 0 else "раньше" if pf.lag_days and pf.lag_days < 0 else "в"
        tail = f"на {fmt.days(pf.lag_days)} {how} плана" if how != "в" else "в плановый срок"
        return f"Все запланированные этапы завершены {fmt.date(pf.actual_finish)} — {tail}{scope}."
    lag = pf.lag_days or 0.0
    if pf.verdict == Verdict.BEHIND:
        head = f"Отставание от графика ≈ {fmt.days(lag)}"
    elif pf.verdict == Verdict.AHEAD:
        head = f"Опережение графика ≈ {fmt.days(lag)}"
    else:
        head = f"Стройка идёт по графику (отклонение {lag:+.0f} дн. в пределах допуска ±{tol} дн.)"
    since = getattr(pf, "observed_since", None)
    base = (f" Съёмка идёт с {fmt.date(since)}: готовность этапа, начатого раньше, по нескольким дням снимков "
            "не измерить — за базу взят план на эту дату, вердикт — по тому, какой этап идёт." if since else "")
    if as_of:
        base += (f" Позже {today} годных для определения этапа снимков нет ({(ctx.today - as_of).days} дн.) — "
                 "сравнение с планом на эту дату; проверьте камеры или отметьте этап вручную.")
    return f"{head}: по плану на {today} должно быть готово {exp}, по снимкам — {act}{scope}.{base}"


def _plan_phrase(ctx: AnalyticsContext) -> str | None:
    now_ids = ctx.stages_by_plan()
    if now_ids:
        parts = []
        for s in now_ids[:2]:
            it = ctx.plan_by_stage[s]
            works = [catalog.label(k) for k in it.work_codes[:1]]
            parts.append(f"{fmt.stage(s)} до {fmt.date(it.planned_end)}" + (f" ({works[0]})" if works else ""))
        more = f" и ещё {len(now_ids) - 2}" if len(now_ids) > 2 else ""
        return "По плану сейчас идут: " + "; ".join(parts) + more + "."
    upcoming = sorted((it.planned_start, s) for s, it in ctx.plan_by_stage.items()
                      if it.planned_start and it.planned_start > ctx.today)
    if upcoming:
        d, s = upcoming[0]
        return f"По плану сегодня этапов нет; следующий — {fmt.stage(s)} с {fmt.date(d)}."
    return None


def _fact_phrase(ctx: AnalyticsContext) -> str | None:
    tl = ctx.timeline
    if not ctx.has_stage_data():
        return None
    done = [s for s, st in tl.states.items() if st.status == StageStatus.DONE]
    cur = tl.current_stage
    parts = []
    if cur is not None:
        st = tl.states.get(cur)
        prog = f", готовность этапа {fmt.pct(st.progress)}" if st and st.status == StageStatus.ACTIVE else ""
        manual = " (отмечено вручную)" if st and st.manual else ""
        parts.append(f"По снимкам текущий этап — {fmt.stage(cur)}{prog}{manual}")
    if done:
        parts.append(f"завершено этапов: {len(done)} из {len(taxonomy.stages())}")
    if not parts:
        return None
    # Почему этап такой — чек-лист модели Б и техника модели А (ТЗ: «этап → техника»). Для ручной
    # отметки основание модели не показываем: этап назначил человек.
    basis = getattr(tl, "basis", None) or {}
    st = tl.states.get(cur) if cur is not None else None
    why = basis.get("text") if basis.get("stage") == cur and not (st and st.manual) else None
    return ", ".join(parts) + "." + (f" {why}" if why else "")


def _forecast_phrase(pf: PlanFact) -> str | None:
    if pf.verdict in (Verdict.NO_DATA,) or pf.actual_finish is not None:
        return None
    if pf.forecast_finish is None:
        return f"Прогноз окончания не дан: {pf.forecast_note}." if pf.forecast_note else None
    if pf.planned_finish:
        d = pf.delay_days or 0
        rel = f"на {fmt.days(d)} позже плана ({fmt.date(pf.planned_finish)})" if d > 0 else \
            f"на {fmt.days(d)} раньше плана ({fmt.date(pf.planned_finish)})" if d < 0 else "в плановый срок"
        pace = f", темп {pf.pace_ratio:.2f} от планового" if pf.pace_ratio else ""
        return f"Прогноз окончания — {fmt.date(pf.forecast_finish)}, {rel}{pace}."
    return f"Прогноз окончания по темпу — {fmt.date(pf.forecast_finish)} ({pf.forecast_note})."


def _deviations_phrase(devs: list[DeviationRecord]) -> str:
    if not devs:
        return "Отклонений не выявлено."
    top = devs[0]
    crit = sum(d.severity == Severity.CRITICAL for d in devs)
    warn = sum(d.severity == Severity.WARNING for d in devs)
    head = top.data.get("headline") or top.title
    counts = f"всего {len(devs)}" + (f", критичных {crit}" if crit else "") + (f", предупреждений {warn}" if warn else "")
    return f"Главное отклонение: {head} ({counts})."


def explain(ctx: AnalyticsContext, pf: PlanFact, devs: list[DeviationRecord]) -> list[str]:
    """2–5 фраз: вердикт с цифрами → что по плану → что видим → прогноз → главное отклонение."""
    phrases = [_verdict_phrase(ctx, pf)]
    for p in (_plan_phrase(ctx) if pf.verdict != Verdict.NO_PLAN else None,
              _fact_phrase(ctx), _forecast_phrase(pf)):
        if p:
            phrases.append(p)
    phrases = phrases[:4]
    phrases.append(_deviations_phrase(devs))
    return phrases
