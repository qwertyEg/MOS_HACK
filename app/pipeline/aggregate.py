"""Хронология чек-листов → какие этапы шли на объекте и когда.

Модель Б отвечает про один кадр и только про то, что видит. Она не знает ни
про календарный план, ни про то, что было вчера. Весь вывод об этапах
делается здесь, из накопленной истории ответов.

**Почему ответ меняется во времени — это норма, а не сбой.** «Виден ли
котлован» в начале стройки «да», после того как здание поднялось — «не видно».
Именно эта смена и есть сигнал: этап шёл, а потом кончился. Если бы модель
пыталась отвечать «логически» («котлован-то был, значит да»), сигнала бы не
осталось вовсе — ответ был бы «да» до самой сдачи объекта.

**Два разных вопроса, которые легко перепутать.**

    идёт ли этап сейчас   — по свежим наблюдениям окна, без всякой памяти
    пройден ли этап       — по необратимым признакам, с памятью навсегда

Смешивать нельзя. Плита залита — это необратимо, бетон сам не исчезнет, и
после того как её перекрыли этажи, честный ответ «не видно» ничего не
отменяет. Но если считать плиту вечным подтверждением активности, этап
подземного монолита останется «активным» до конца стройки. Поэтому
необратимые признаки идут в «пройден», а активность считается только по
тому, что видно в окне прямо сейчас.

**Голосование.** У этапа есть вопросы двух полярностей: `must_have` (должно
быть видно, если этап идёт) и `must_not_have` (не должно). Ответ «не видно»
не голосует вовсе — это и есть смысл третьего значения. Доля подтверждающих
среди проголосовавших даёт P(этап активен), а доля проголосовавших от всех
вопросов — уверенность: если модель почти всё не разглядела, низкий P значит
«нечего сказать», а не «этапа нет».
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass

from app.models import Answer

# Окно сглаживания. Дневной ответ шумит: облако, ракурс, случайно
# заслонивший обзор грузовик. Этап за три дня не начинается и не кончается.
SMOOTH_DAYS = 5
ACTIVE_THRESHOLD = 0.6      # доля подтверждающих голосов, выше которой этап активен
MIN_CONFIDENCE = 0.34       # ниже — считаем, что кадр про этот этап ничего не сказал
MIN_RUN_DAYS = 3            # более короткие всплески — шум, а не этап


@dataclass(slots=True)
class DayPoint:
    day: dt.date
    p: float                # доля подтверждающих среди проголосовавших
    confidence: float       # доля вопросов, на которые вообще ответили
    votes: int


@dataclass(slots=True)
class StageCurve:
    stage_id: int
    title: str
    points: list[DayPoint]
    intervals: list[tuple[dt.date, dt.date]]
    reached: bool           # все необратимые признаки этапа когда-либо видели


def daily_answers(
    observations: list[tuple[dt.datetime, str, Answer]],
) -> dict[dt.date, dict[str, Answer]]:
    """Наблюдения по кадрам → один ответ на ключ в день.

    В сутки кадров много, и ответы по ним расходятся. Берём большинство
    среди тех, кто голосовал; если все сказали «не видно» — день по этому
    ключу молчит, и это честнее, чем выбрать da/нет монеткой.
    """
    buckets: dict[dt.date, dict[str, list[Answer]]] = defaultdict(
        lambda: defaultdict(list))
    for when, key, answer in observations:
        buckets[when.date()][key].append(answer)

    out: dict[dt.date, dict[str, Answer]] = {}
    for day, keys in buckets.items():
        resolved = {}
        for key, answers in keys.items():
            yes = sum(a is Answer.YES for a in answers)
            no = sum(a is Answer.NO for a in answers)
            if yes == 0 and no == 0:
                resolved[key] = Answer.UNSURE
            else:
                resolved[key] = Answer.YES if yes >= no else Answer.NO
        out[day] = resolved
    return out


def ever_seen(days: dict[dt.date, dict[str, Answer]], key: str) -> dt.date | None:
    """Первый день, когда признак наблюдали. None — не наблюдали ни разу."""
    seen = [d for d, ans in days.items() if ans.get(key) is Answer.YES]
    return min(seen) if seen else None


def stage_score(answers: dict[str, Answer], questions: list[dict]) -> DayPoint | None:
    """Оценка одного этапа по ответам одного дня."""
    voting = [q for q in questions if q["polarity"] in ("must_have", "must_not_have")]
    if not voting:
        return None

    votes = supporting = 0
    for q in voting:
        a = answers.get(q["key"], Answer.UNSURE)
        if a is Answer.UNSURE:
            continue
        votes += 1
        want_yes = q["polarity"] == "must_have"
        if (a is Answer.YES) == want_yes:
            supporting += 1

    if votes == 0:
        return DayPoint(day=dt.date.min, p=0.0, confidence=0.0, votes=0)
    return DayPoint(day=dt.date.min, p=supporting / votes,
                    confidence=votes / len(voting), votes=votes)


def smooth(points: list[DayPoint], window: int = SMOOTH_DAYS) -> list[DayPoint]:
    """Скользящее среднее по календарным дням, а не по индексу.

    Дни с пропусками (ночь, брак кадра, выходной без съёмки) не должны
    склеивать далеко отстоящие наблюдения в одно окно.
    """
    out = []
    for i, pt in enumerate(points):
        lo = pt.day - dt.timedelta(days=window // 2)
        hi = pt.day + dt.timedelta(days=window // 2)
        near = [q for q in points if lo <= q.day <= hi and q.votes]
        if not near:
            out.append(pt)
            continue
        out.append(DayPoint(
            day=pt.day,
            p=sum(q.p for q in near) / len(near),
            confidence=sum(q.confidence for q in near) / len(near),
            votes=max(q.votes for q in near),
        ))
    return out


def intervals(
    points: list[DayPoint],
    threshold: float = ACTIVE_THRESHOLD,
    min_confidence: float = MIN_CONFIDENCE,
    min_run: int = MIN_RUN_DAYS,
) -> list[tuple[dt.date, dt.date]]:
    """Сглаженная кривая → отрезки, на которых этап считается активным."""
    runs: list[tuple[dt.date, dt.date]] = []
    start: dt.date | None = None
    prev: dt.date | None = None

    for pt in points:
        active = pt.votes and pt.confidence >= min_confidence and pt.p >= threshold
        if active:
            if start is None:
                start = pt.day
            prev = pt.day
        elif start is not None:
            runs.append((start, prev))
            start = prev = None
    if start is not None:
        runs.append((start, prev))

    return [(a, b) for a, b in runs if (b - a).days + 1 >= min_run]


def build_curves(
    observations: list[tuple[dt.datetime, str, Answer]],
    stages: list[tuple[int, str, list[dict]]],
) -> list[StageCurve]:
    """Полный разбор: наблюдения + состав чек-листов → кривые по этапам.

    stages — список (id этапа, название, вопросы чек-листа).
    """
    days = daily_answers(observations)
    ordered = sorted(days)

    curves = []
    for stage_id, title, questions in stages:
        raw = []
        for day in ordered:
            pt = stage_score(days[day], questions)
            if pt is None:
                continue
            raw.append(DayPoint(day=day, p=pt.p, confidence=pt.confidence,
                                votes=pt.votes))

        smoothed = smooth(raw)
        latching = [q["key"] for q in questions
                    if q.get("latching") and q["polarity"] == "must_have"]
        reached = bool(latching) and all(ever_seen(days, k) for k in latching)

        curves.append(StageCurve(stage_id=stage_id, title=title,
                                 points=smoothed,
                                 intervals=intervals(smoothed),
                                 reached=reached))
    return curves


# ---------------------------------------------------------------------------
# выборка из базы
# ---------------------------------------------------------------------------

def _observations(session, site, stage_ids: list[int]):
    """Ответы по объекту вместе с камерой, с которой пришёл кадр."""
    from sqlalchemy import select

    from app.models import Checklist, ChecklistAnswer, Frame

    return session.execute(
        select(Frame.camera_id, Frame.captured_at,
               ChecklistAnswer.key, ChecklistAnswer.answer)
        .join(Checklist, Checklist.frame_id == Frame.id)
        .join(ChecklistAnswer, ChecklistAnswer.checklist_id == Checklist.id)
        .where(Checklist.site_stage_id.in_(stage_ids))
    ).all()


def _stage_defs(site) -> list[tuple[int, str, list[dict]]]:
    """Состав чек-листов берётся из снимка на `SiteStage`, а не из справочника.

    Состав заморожен в момент сохранения плана: правка CSV не должна задним
    числом менять то, по чему считалась уже накопленная история.
    """
    return [(st.id, st.title, st.questions or []) for st in site.stages]


def site_curves(session, site, camera_id: int | None = None) -> list[StageCurve]:
    """История чек-листов объекта → кривые по его этапам.

    `camera_id` сужает выборку до одной камеры. Это не мелочь: две камеры
    смотрят на площадку с разных сторон и видят разное, и вывод «этап идёт»
    по каждой из них стоит уметь посмотреть отдельно — расхождение между
    ними само по себе диагностика, а не шум.
    """
    stage_ids = [st.id for st in site.stages]
    if not stage_ids:
        return []

    rows = _observations(session, site, stage_ids)
    observations = [(when, key, answer)
                    for cam, when, key, answer in rows
                    if camera_id is None or cam == camera_id]
    if not observations:
        return []
    return build_curves(observations, _stage_defs(site))


def curves_by_camera(session, site) -> dict[int, list[StageCurve]]:
    """Кривые отдельно по каждой камере, которая что-то наблюдала."""
    stage_ids = [st.id for st in site.stages]
    if not stage_ids:
        return {}

    per_cam: dict[int, list] = defaultdict(list)
    for cam, when, key, answer in _observations(session, site, stage_ids):
        per_cam[cam].append((when, key, answer))

    defs = _stage_defs(site)
    return {cam: build_curves(obs, defs) for cam, obs in per_cam.items() if obs}
