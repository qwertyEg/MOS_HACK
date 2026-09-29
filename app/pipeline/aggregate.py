"""Хронология чек-листов → начало и активность этапов на объекте.

Модель Б отвечает только о видимых признаках одного кадра. Здесь ответы
нескольких камер сводятся по дням, а собственные физические сигналы каждого
этапа подтверждают его начало. Порядок этапов сохраняется, соседние этапы
могут идти одновременно. Сигнал подтверждается двумя положительными днями
в пределах недели; после старта следующего этапа предыдущая полоса остаётся
активной ещё 14 дней.

Прогресс этапа считается отдельно из наблюдаемых вех (`progress.py`), поэтому
отсутствие ответа на конкретный вопрос не превращается в нулевой прогресс.
Изменённые тексты вопросов не смешиваются со старыми ответами: аналитика
учитывает только ответы, чей снимок вопроса совпадает с текущим.

В модуле оставлены функции `sequential_state` и связанные с ней счётчики для
совместимости со старыми проверками. Построение текущих полос через
`build_curves` их не использует.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass

from app.models import Answer
from app.pipeline import progress as P

# Разрыв в наблюдениях, после которого отрезок обрывается. Камеру сняли,
# сервис лежал, площадку заволокло — за эти дни мы ничего не видели и
# утверждать, что этап всё это время шёл, не имеем права. Без обрыва два
# наблюдения по краям дыры склеивались бы в один отрезок через неё: на
# объекте со старой камерой это давало этап длиной в двадцать лет.
MAX_GAP_DAYS = 10

# Параметры оставлены вместе с legacy-функцией sequential_state; активный
# алгоритм использует отдельные сигналы BOUNDARY_KEYS ниже.
MAX_JUMP = 1

# Используются только legacy-функциями последовательного счёта.
MIN_STREAK_DAYS = 3
BOUNDARY_CONFIRM_DAYS = 2
OVERLAP_DAYS = 14

# Доля дней с «да», после которой признак считается постоянным и теряет право
# вето. Семь этапов на всю историю: признак одного из них физически не может
# гореть на девяти днях из десяти — значит он про стройку вообще, а не про этап.
CONSTANT_SHARE = 0.9
# Пока истории мало, «горит всегда» и «горит сейчас» — одно и то же, и
# отбирать вето не за что. Месяц съёмки — минимум, на котором различие есть.
CONSTANT_MIN_DAYS = 30


@dataclass(slots=True)
class StageCurve:
    stage_id: int
    title: str
    intervals: list[tuple[dt.date, dt.date]]
    reached: bool           # начало следующего этапа подтверждено
    progress: int = 0
    coverage: int = 0


def daily_answers(
    observations: list[tuple[dt.datetime, str, Answer]],
) -> dict[dt.date, dict[str, Answer]]:
    """Наблюдения по кадрам (с разных камер) → один ответ на ключ в день.

    Несколько камер смотрят на площадку с разных сторон, и ракурс — это не
    ошибка наблюдения, а его условие: свая, ясно видная одной камере, может
    быть у другой вовсе за кадром. Поэтому правило асимметричное — один явный
    «да» перевешивает любое число «нет» и «не видно». Симметричное
    большинство было бы неверным: оно позволило бы двум камерам, которые
    просто не смотрят в нужную сторону, отменить наблюдение третьей, которая
    смотрит прямо на признак.
    """
    buckets: dict[dt.date, dict[str, list[Answer]]] = defaultdict(
        lambda: defaultdict(list))
    for when, key, answer in observations:
        buckets[when.date()][key].append(answer)

    out: dict[dt.date, dict[str, Answer]] = {}
    for day, keys in buckets.items():
        resolved = {}
        for key, answers in keys.items():
            if any(a is Answer.YES for a in answers):
                resolved[key] = Answer.YES
            elif any(a is Answer.NO for a in answers):
                resolved[key] = Answer.NO
            else:
                resolved[key] = Answer.UNSURE
        out[day] = resolved
    return out


def constant_keys(
    days: dict[dt.date, dict[str, Answer]],
    share: float = CONSTANT_SHARE,
    min_days: int = CONSTANT_MIN_DAYS,
) -> set[str]:
    """Ключи, горящие «да» почти на всей истории, — см. шапку модуля.

    Такой ключ описывает не этап, а стройку целиком, и права задерживать
    границу прогресса у него быть не должно. На короткой истории множество
    пустое: отличить постоянный признак от идущего прямо сейчас не на чем.
    """
    if len(days) < min_days:
        return set()
    yes: defaultdict[str, int] = defaultdict(int)
    for answers in days.values():
        for key, answer in answers.items():
            if answer is Answer.YES:
                yes[key] += 1
    return {key for key, n in yes.items() if n >= share * len(days)}


def daily_feature_counts(
    days: dict[dt.date, dict[str, Answer]],
    stages: list[tuple[int, str, list[dict]]],
) -> list[tuple[dt.date, dict[int, tuple[int, int]]]]:
    """День → (все признаки этапа, временные признаки этапа) — по каждому этапу.

    Две цифры, а не одна, ради того самого разделения из шапки модуля:
    первая идёт в сравнение «кто сегодня лидирует», вторая — только
    временные (не `latching`) признаки — в проверку «этап ещё не отпустил
    свои признаки». Возвращается список, а не словарь: порядок дней важен
    дальше, а словарь его не хранит.
    """
    constant = constant_keys(days)
    out = []
    for day in sorted(days):
        answers = days[day]
        counts = {}
        for stage_id, _, questions in stages:
            must = [q for q in questions if q["polarity"] == "must_have"]
            total = sum(1 for q in must if answers.get(q["key"]) is Answer.YES)
            transient = sum(1 for q in must
                            if not q.get("latching") and q["key"] not in constant
                            and answers.get(q["key"]) is Answer.YES)
            counts[stage_id] = (total, transient)
        out.append((day, counts))
    return out


def sequential_state(
    day_counts: list[tuple[dt.date, dict[int, tuple[int, int]]]],
    order: list[int],
    max_jump: int = MAX_JUMP,
    min_streak: int = MIN_STREAK_DAYS,
) -> dict[dt.date, int]:
    """Дневные счётчики признаков → какому этапу отнесён каждый день.

    `order` — id этапов в порядке календарного плана; это и есть та самая
    последовательность, дальше которой прыгать нельзя. Возвращает день → id
    активного на тот день этапа. Граница прогресса (`frontier`, индекс в
    `order`) за один вызов только растёт — реализация «этап не открывается
    снова» не постфактум-фильтром, а тем, что состояние физически не может
    откатиться назад.

    Счётчики — пары (все признаки, временные признаки), см. `daily_feature_counts`.
    """
    out: dict[dt.date, int] = {}
    frontier = 0
    streak = 0

    for day, counts in day_counts:
        current_id = order[frontier]
        current_total, current_transient = counts.get(current_id, (0, 0))

        # Кандидат на смену — только следующий этап в окне; всё, что дальше,
        # физически не сравнивается и повлиять на решение не может.
        window = order[frontier + 1:frontier + 1 + max_jump]
        next_id = window[0] if window else None
        next_total = counts.get(next_id, (0, 0))[0] if next_id is not None else -1

        advancing = next_id is not None and next_total > current_total

        if not advancing or current_transient > 0:
            # Либо следующий этап не лидирует, либо у текущего ещё остались
            # временные признаки — переход рано, что бы ни было впереди.
            out[day] = current_id
            streak = 0
            continue

        streak += 1
        if streak >= min_streak:
            frontier += 1
            streak = 0
        out[day] = order[frontier]

    return out


def _runs_from_assignment(
    assignment: dict[dt.date, int],
    max_gap: int = MAX_GAP_DAYS,
) -> dict[int, list[tuple[dt.date, dt.date]]]:
    """День → этап, рассортированный на отрезки по этапу и по дырам в датах."""
    out: dict[int, list[tuple[dt.date, dt.date]]] = defaultdict(list)
    ordered = sorted(assignment)
    if not ordered:
        return out

    start = prev = ordered[0]
    cur = assignment[start]
    for day in ordered[1:]:
        stage = assignment[day]
        if stage != cur or (day - prev).days > max_gap:
            out[cur].append((start, prev))
            start, cur = day, stage
        prev = day
    out[cur].append((start, prev))
    return out


def build_curves(
    observations: list[tuple[dt.datetime, str, Answer]],
    stages: list[tuple[int, str, list[dict]]],
    stage_kinds: dict[int, int] | None = None,
) -> list[StageCurve]:
    """Ответы по дням → независимые полосы этапов с соседним перекрытием."""
    days = daily_answers(observations)
    order = [sid for sid, _, _ in stages]
    stage_kinds = stage_kinds or {sid: sid for sid in order}
    starts = _stage_boundaries(days, order, stage_kinds)
    intervals = _parallel_intervals(days, starts)
    progress_starts = {stage_kinds[sid]: start for sid, start in starts.items()
                       if stage_kinds.get(sid) in P.MILESTONES}
    metrics = P.stage_progress(days, progress_starts)
    latest_started = max((i for i, sid in enumerate(order) if sid in starts), default=-1)

    out = []
    for index, (sid, title, _questions) in enumerate(stages):
        metric = metrics.get(stage_kinds.get(sid, sid), P.StageProgress(0, 0, 0, 0))
        out.append(StageCurve(
            stage_id=sid, title=title, intervals=intervals.get(sid, []),
            reached=index < latest_started,
            progress=metric.percent, coverage=metric.coverage,
        ))
    return out


# Один точный признак перехода важнее победы случайной комбинации вопросов.
# Старые временные признаки («мусор», «арматура», «опалубка») больше не
# получают право задерживать весь объект.
BOUNDARY_KEYS = {
    2: {"pile_rig", "pile_stock", "pile_heads", "sheet_pile"},
    3: {"pit"},
    4: {"basement_concrete_start", "basement_walls"},
    5: {"above_grade"},
    6: {"glazing", "insulation", "cladding"},
    7: {"paving", "landscaping", "amenities"},
}


def _stage_boundaries(days: dict[dt.date, dict[str, Answer]],
                      order: list[int], stage_kinds: dict[int, int]
                      ) -> dict[int, dt.date]:
    """Первые два положительных дня сигнала этапа подтверждают его начало.

    Ищем только следующий этап после уже найденного. Этим сохраняется порядок,
    но завершённая и следующая стадии могут отображаться одновременно.
    """
    observed = sorted(days)
    if not observed or not order:
        return {}
    starts = {order[0]: observed[0]}

    for index, stage_id in enumerate(order[1:], start=1):
        previous_start = starts.get(order[index - 1])
        if previous_start is None:
            break
        stage_kind = stage_kinds.get(stage_id, stage_id)
        keys = BOUNDARY_KEYS.get(stage_kind, set())
        candidates = [day for day in observed if day >= previous_start
                      and any(days[day].get(key) is Answer.YES for key in keys)]
        confirmed = None
        for day in candidates:
            last = day + dt.timedelta(days=6)
            if sum(day <= other <= last for other in candidates) >= BOUNDARY_CONFIRM_DAYS:
                confirmed = day
                break

        # Начало подземного монолита подтверждает бетон в котловане или стены.
        # Если бетон скрыт ракурсом, совместная опалубка подземной части и
        # арматура остаются резервным сигналом; подсыпка границу не двигает.
        if stage_kind == 4 and confirmed is None:
            pairs = [day for day in observed if day >= previous_start
                     and days[day].get("formwork_basement") is Answer.YES
                     and days[day].get("rebar") is Answer.YES]
            for day in pairs:
                last = day + dt.timedelta(days=6)
                if sum(day <= other <= last for other in pairs) >= BOUNDARY_CONFIRM_DAYS:
                    confirmed = day
                    break
        if confirmed is None:
            break
        starts[stage_id] = confirmed
    return starts


def _parallel_intervals(days: dict[dt.date, dict[str, Answer]],
                        starts: dict[int, dt.date]
                        ) -> dict[int, list[tuple[dt.date, dt.date]]]:
    """Строит полосы этапов; соседние работы видны вместе 14 дней."""
    order = list(starts)
    if not order:
        return {}
    last_day = max(days)
    active_days: dict[int, list[dt.date]] = defaultdict(list)
    for index, stage_id in enumerate(order):
        start = starts[stage_id]
        end = (min(last_day, starts[order[index + 1]] + dt.timedelta(days=OVERLAP_DAYS))
               if index + 1 < len(order) else last_day)
        active_days[stage_id] = [day for day in sorted(days) if start <= day <= end]

    out = {}
    for stage_id, stage_days in active_days.items():
        if not stage_days:
            continue
        runs = []
        start = prev = stage_days[0]
        for day in stage_days[1:]:
            if (day - prev).days > MAX_GAP_DAYS:
                runs.append((start, prev))
                start = day
            prev = day
        runs.append((start, prev))
        out[stage_id] = runs
    return out


# ---------------------------------------------------------------------------
# выборка из базы
# ---------------------------------------------------------------------------

def _observations(session, site, stage_ids: list[int]):
    """Ответы по объекту вместе с камерой, с которой пришёл кадр."""
    from sqlalchemy import select

    from app.models import Checklist, ChecklistAnswer, Frame

    return session.execute(
        select(Frame.camera_id, Frame.captured_at, Checklist.site_stage_id,
               ChecklistAnswer.key, ChecklistAnswer.question, ChecklistAnswer.answer)
        .join(Checklist, Checklist.frame_id == Frame.id)
        .join(ChecklistAnswer, ChecklistAnswer.checklist_id == Checklist.id)
        .where(Checklist.site_stage_id.in_(stage_ids))
    ).all()


def _stage_defs(site) -> list[tuple[int, str, list[dict]]]:
    """Состав чек-листов берётся из снимка на `SiteStage`, а не из справочника.

    Состав заморожен в момент сохранения плана: правка CSV не должна задним
    числом менять то, по чему считалась уже накопленная история. Порядок —
    `SiteStage.order_idx` (см. `relationship(..., order_by=...)` в моделях):
    это порядок календарного плана и порядок проверки сигналов начала этапов.
    """
    return [(st.id, st.title, st.questions or []) for st in site.stages]


def site_curves(session, site, camera_id: int | None = None,
                until: dt.date | None = None) -> list[StageCurve]:
    """История чек-листов объекта → кривые по его этапам.

    `camera_id` сужает выборку до одной камеры — для отладки расхождений
    между ракурсами. В обычном разборе (`camera_id=None`) все камеры объекта
    сведены вместе ещё на входе в `daily_answers`, по правилу «да
    перевешивает нет»: так и должна работать система с несколькими камерами
    на одной площадке.
    """
    stage_ids = [st.id for st in site.stages]
    if not stage_ids:
        return []

    rows = _observations(session, site, stage_ids)
    current_questions = {
        st.id: {q["key"]: q.get("text", "") for q in (st.questions or [])}
        for st in site.stages
    }
    observations = [(when, key, answer)
                    for cam, when, stage_id, key, question, answer in rows
                    if (camera_id is None or cam == camera_id)
                    and (until is None or when.date() <= until)
                    and current_questions.get(stage_id, {}).get(key) == question]
    if not observations:
        return []
    stage_kinds = {st.id: st.macro_stage_id or 0 for st in site.stages}
    return build_curves(observations, _stage_defs(site), stage_kinds)
