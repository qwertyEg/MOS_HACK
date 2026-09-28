"""Хронология чек-листов → какие этапы шли на объекте и когда.

Модель Б отвечает про один кадр и только про то, что видит. Она не знает ни
про календарный план, ни про то, что было вчера. Весь вывод об этапах —
включая порядок и то, что этап не может начаться заново, — делается здесь.

**Стройка — это последовательность, а не независимый выбор.** Экскаватор не
выезжает копать котлован там, где уже льют фундамент. Значит и вывод должен
быть последовательным: у объекта есть ровно одна граница прогресса
(`frontier`) — самый ранний ещё не закрытый этап, — и она умеет двигаться
только вперёд. Раз закрытый этап не открывается снова: если бы «подготовка
территории» вдруг вновь возглавила счёт, пока благоустройство ещё не начато,
это железно ошибка модели Б на одном кадре, а не разворот стройки во времени.

**Кандидатов на «следующий» только два.** Сам текущий этап и тот, что сразу
за ним. Этап через один в кандидаты не попадает вовсе — не порогом
отсекается, а физически недостижим: одна шумная фотография не может
перебросить систему через три этапа разом, потому что она в принципе не
смотрит так далеко. Это и есть защита от той ошибки, которую нельзя тихо
проглотить: «благоустройство» на кадре, где ещё не закрыт котлован, —
кандидат вне окна, и мы просто ждём следующую фотографию.

**Закрыть этап можно только когда его СОБСТВЕННЫЕ признаки замолчали.** Пока
хоть одна камера показывает хоть один характерный признак текущего этапа,
переходить дальше нельзя — что бы ни показывал следующий этап. Это и
разрешает три камеры с разных ракурсов: одна не видит сваи, потому что они
за кадром, но раз другая их видит — сваи есть, и точка. Асимметрия
заложена уже в `daily_answers`: «да» перевешивает любое число «нет» и «не
видно», потому что молчание камеры не опровергает то, что увидела соседняя.

**Признак этапа — это его `must_have`-вопросы.** `must_not_have` и `context`
в счёт не идут: это про исключение и общий фон, а не про то, что характерно
именно для этого этапа. Разбор чек-листов по этому принципу будет продолжен
отдельно.

**«Молчание» текущего этапа считается не по всем его признакам, а только по
необратимым (`latching`).** Кровля, однажды закрытая кровельным покрытием,
останется видна на каждом следующем кадре — это не значит, что кровельные
работы всё ещё идут, это значит, что они закончились. Если бы такой признак
удерживал переход, застройщик застрял бы на этапе кровли навсегда: она видна
всегда, и своего признака этап «не лишится» никогда. Держат границу только
временные признаки — сваебойная установка, открытый котлован, голая
опалубка, — которые физически пропадают из кадра, когда работа кончается.
Для голосования «кто сейчас лидирует» (сравнение счёта с соседним этапом)
это разделение не нужно и не делается: там считаются все `must_have` разом.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass

from app.models import Answer

# Разрыв в наблюдениях, после которого отрезок обрывается. Камеру сняли,
# сервис лежал, площадку заволокло — за эти дни мы ничего не видели и
# утверждать, что этап всё это время шёл, не имеем права. Без обрыва два
# наблюдения по краям дыры склеивались бы в один отрезок через неё: на
# объекте со старой камерой это давало этап длиной в двадцать лет.
MAX_GAP_DAYS = 10

# Кандидатов на переход — только текущий этап и следующий: см. докстринг
# модуля. Значение больше единицы уже нарушало бы саму идею защиты.
MAX_JUMP = 1

# Переход необратим, а значит цена одной ошибочной фотографии — навсегда.
# Требуем несколько дней подряд, где старый этап молчит, а новый лидирует,
# прежде чем сдвинуть границу. Разовый всплеск — шум, не переход.
MIN_STREAK_DAYS = 3


@dataclass(slots=True)
class StageCurve:
    stage_id: int
    title: str
    intervals: list[tuple[dt.date, dt.date]]
    reached: bool           # граница прогресса уже прошла этот этап


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
    out = []
    for day in sorted(days):
        answers = days[day]
        counts = {}
        for stage_id, _, questions in stages:
            must = [q for q in questions if q["polarity"] == "must_have"]
            total = sum(1 for q in must if answers.get(q["key"]) is Answer.YES)
            transient = sum(1 for q in must if not q.get("latching")
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
) -> list[StageCurve]:
    """Полный разбор: наблюдения + состав чек-листов → кривые по этапам.

    stages — список (id этапа, название, вопросы чек-листа) **в порядке
    календарного плана**: этот порядок и есть последовательность, за
    пределы которой граница прогресса не выходит.
    """
    days = daily_answers(observations)
    order = [sid for sid, _, _ in stages]
    assignment = sequential_state(daily_feature_counts(days, stages), order)
    runs = _runs_from_assignment(assignment)

    frontier = order.index(assignment[max(assignment)]) if assignment else -1

    return [StageCurve(stage_id=sid, title=title,
                       intervals=runs.get(sid, []), reached=idx < frontier)
            for idx, (sid, title, _questions) in enumerate(stages)]


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
    числом менять то, по чему считалась уже накопленная история. Порядок —
    `SiteStage.order_idx` (см. `relationship(..., order_by=...)` в моделях):
    это порядок календарного плана, и именно он задаёт последовательность
    для `sequential_state`.
    """
    return [(st.id, st.title, st.questions or []) for st in site.stages]


def site_curves(session, site, camera_id: int | None = None) -> list[StageCurve]:
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
    observations = [(when, key, answer)
                    for cam, when, key, answer in rows
                    if camera_id is None or cam == camera_id]
    if not observations:
        return []
    return build_curves(observations, _stage_defs(site))
