"""Свёртка хронологии чек-листов в последовательность активных этапов.

Три свойства, ради которых модуль и написан:

- порядок нельзя нарушить скачком — кандидат на переход только следующий
  этап, дальний не сравнивается вовсе, что бы он ни показывал;
- закрытый этап не открывается снова — граница прогресса только растёт;
- камера, которая просто не видит признак под своим углом, не должна
  отменять наблюдение камеры, которая его видит.
"""

import datetime as dt

import pytest

from app.models import Answer
from app.pipeline import aggregate as A
from app.seed import load_checklists

START = dt.date(2026, 1, 1)
CHECKLISTS = load_checklists()


def day(offset: int) -> dt.date:
    return START + dt.timedelta(days=offset)


# ---------------------------------------------------------------------------
# daily_answers: да перевешивает нет
# ---------------------------------------------------------------------------

def obs(day_offset: int, key: str, *answers: str) -> list[tuple[dt.datetime, str, Answer]]:
    when = dt.datetime.combine(day(day_offset), dt.time(12, 0))
    return [(when + dt.timedelta(minutes=i), key, Answer(a))
            for i, a in enumerate(answers)]


def test_one_yes_beats_any_number_of_no() -> None:
    """Свая видна одной камере из трёх — значит, свая есть."""
    days = A.daily_answers(obs(0, "pit", "no", "no", "yes"))
    assert days[day(0)]["pit"] is Answer.YES


def test_no_without_any_yes_stays_no() -> None:
    days = A.daily_answers(obs(0, "pit", "no", "no"))
    assert days[day(0)]["pit"] is Answer.NO


def test_all_unsure_is_unsure_not_a_coin_flip() -> None:
    days = A.daily_answers(obs(0, "pit", "unsure", "unsure"))
    assert days[day(0)]["pit"] is Answer.UNSURE


# ---------------------------------------------------------------------------
# sequential_state: последовательность и её защита от скачков
# ---------------------------------------------------------------------------

def counts(*rows: dict[int, int]) -> list[tuple[dt.date, dict[int, tuple[int, int]]]]:
    """Однозначные синтетические счётчики: total == transient (нет latching)."""
    return [(day(i), {sid: (n, n) for sid, n in row.items()}) for i, row in enumerate(rows)]


def counts_lt(*rows: dict[int, tuple[int, int]]
             ) -> list[tuple[dt.date, dict[int, tuple[int, int]]]]:
    """Счётчики с явным разделением (total, transient) — для теста latching."""
    return [(day(i), row) for i, row in enumerate(rows)]


def test_stays_on_current_while_nothing_beats_it() -> None:
    assignment = A.sequential_state(counts({1: 3}, {1: 2}, {1: 1}), order=[1, 2, 3])
    assert set(assignment.values()) == {1}


def test_advances_after_a_streak_once_old_stage_falls_silent() -> None:
    rows = counts({1: 0, 2: 3}, {1: 0, 2: 3}, {1: 0, 2: 3})
    assignment = A.sequential_state(rows, order=[1, 2, 3], min_streak=3)
    assert [assignment[day(i)] for i in range(3)] == [1, 1, 2]


def test_single_day_spike_does_not_advance() -> None:
    """Один шумный кадр — не переход: нужно несколько дней подряд."""
    rows = counts({1: 0, 2: 5}, {1: 3, 2: 0}, {1: 0, 2: 5})
    assignment = A.sequential_state(rows, order=[1, 2, 3], min_streak=3)
    assert set(assignment.values()) == {1}, "переход случился по одному всплеску"


def test_old_stage_feature_blocks_advance_even_if_next_leads() -> None:
    """Пока у текущего этапа остался хоть один временный признак, переходить нельзя.

    Именно это разрешает разные ракурсы: пока хоть одна камера видит сваю,
    земляные работы не закрыты, даже если другая камера уже видит котлован.
    """
    rows = counts(*([{1: 1, 2: 5}] * 6))
    assignment = A.sequential_state(rows, order=[1, 2, 3], min_streak=3)
    assert set(assignment.values()) == {1}


def test_latching_evidence_does_not_block_closing() -> None:
    """Необратимый признак — это не «работа ещё идёт», а «работа сделана».

    Закрытая кровля видна на каждом следующем кадре; если бы это держало
    переход, этап кровли не закрывался бы никогда — она ведь так и не
    исчезнет из кадра. Держат границу только временные признаки.
    """
    # 1: total=3 (в т.ч. latching), transient=0 — кровля закрыта, работ не видно
    rows = counts_lt(*([{1: (3, 0), 2: (5, 5)}] * 6))
    assignment = A.sequential_state(rows, order=[1, 2], min_streak=3)
    assert [assignment[day(i)] for i in range(6)] == [1, 1, 2, 2, 2, 2]


def test_transient_evidence_still_blocks_closing() -> None:
    """Тот же счёт, но признак — временный: переход не должен случиться."""
    rows = counts_lt(*([{1: (3, 3), 2: (5, 5)}] * 6))
    assignment = A.sequential_state(rows, order=[1, 2], min_streak=3)
    assert set(assignment.values()) == {1}


def test_distant_stage_is_never_a_candidate() -> None:
    """Этап через два — не кандидат вовсе, даже если у него больше всего признаков.

    Ровно тот случай, который нельзя тихо проглотить: «благоустройство» с
    самым высоким счётом на кадре, где котлован ещё не закрыт, — не должно
    сдвинуть систему ни на шаг, сколько бы дней это ни продолжалось.
    """
    rows = counts(*([{1: 1, 2: 0, 3: 0, 4: 99}] * 10))
    assignment = A.sequential_state(rows, order=[1, 2, 3, 4], min_streak=3)
    assert set(assignment.values()) == {1}


def test_closed_stage_never_reactivates() -> None:
    """Граница только растёт: сильный сигнал старого этапа её не откатывает."""
    rows = counts({1: 0, 2: 3}, {1: 0, 2: 3}, {1: 0, 2: 3},   # переход на 2
                  {1: 9, 2: 0}, {1: 9, 2: 0}, {1: 9, 2: 0})   # 1 «ожил»
    assignment = A.sequential_state(rows, order=[1, 2], min_streak=3)
    assert [assignment[day(i)] for i in range(6)] == [1, 1, 2, 2, 2, 2]


def test_frontier_can_advance_more_than_one_stage_over_time() -> None:
    """За несколько раздельных переходов граница может уйти далеко вперёд —
    просто не одним скачком через нерассмотренный этап."""
    rows = counts(
        {1: 0, 2: 3}, {1: 0, 2: 3}, {1: 0, 2: 3},   # 1 → 2
        {1: 0, 2: 0, 3: 3}, {1: 0, 2: 0, 3: 3}, {1: 0, 2: 0, 3: 3},  # 2 → 3
    )
    assignment = A.sequential_state(rows, order=[1, 2, 3], min_streak=3)
    assert assignment[day(5)] == 3


# ---------------------------------------------------------------------------
# build_curves: сквозной разбор
# ---------------------------------------------------------------------------

# Все семь этапов подряд, по своим характерным (must_have) признакам.
# Ключ, не упомянутый в фазе, просто отсутствует в ответах дня — это не
# «нет», а «не спрошено», и на счёт (`daily_feature_counts`) влияет
# одинаково: не прибавляет ни одному этапу.
SCENE = [
    (0, 14,   {"cleared": "yes", "old_building": "yes", "debris": "yes",
               "tree_felling": "yes", "flat_ground": "yes"}),
    (15, 29,  {"pile_rig": "yes", "pile_stock": "yes", "pile_heads": "yes",
               "sheet_pile": "yes", "capping_beam": "yes"}),
    (30, 44,  {"pit": "yes", "earthwork": "yes", "soil_pile": "yes",
               "struts": "yes", "pit_bottom_bare": "yes"}),
    (45, 59,  {"pit_bottom_prepared": "yes", "formwork": "yes", "rebar": "yes",
               "basement_walls": "yes", "backfill": "yes"}),
    (60, 74,  {"above_grade": "yes", "formwork_floor": "yes",
               "unfinished_top": "yes", "masonry": "yes",
               "bare_concrete": "yes"}),
    # Кровля и фасад закрыты (cladding, glazing, roof_cover — необратимые,
    # признаны навсегда) и больше не мешают перейти к благоустройству:
    # держат границу только леса и открытый утеплитель.
    (75, 89,  {"cladding": "yes", "glazing": "yes", "scaffold": "yes",
               "insulation": "yes", "roof_cover": "yes"}),
    (90, 104, {"paving": "yes", "landscaping": "yes", "amenities": "yes",
               "asphalt_work": "yes", "no_heavy_equipment": "yes"}),
]


def observations() -> list[tuple[dt.datetime, str, Answer]]:
    """Два кадра в день, как с реальной камеры."""
    out = []
    for lo, hi, scene in SCENE:
        for offset in range(lo, hi + 1):
            when = dt.datetime.combine(day(offset), dt.time(12, 0))
            for key, value in scene.items():
                for shot in range(2):
                    out.append((when + dt.timedelta(hours=shot), key,
                                Answer(value)))
    return out


@pytest.fixture(scope="module")
def curves() -> dict[int, A.StageCurve]:
    stages = [(sid, f"этап {sid}", qs) for sid, qs in sorted(CHECKLISTS.items())]
    return {c.stage_id: c for c in A.build_curves(observations(), stages)}


def covers(curve: A.StageCurve, offset: int) -> bool:
    d = day(offset)
    return any(a <= d <= b for a, b in curve.intervals)


def test_earthwork_detected_in_its_phase(curves) -> None:
    assert covers(curves[3], 40), "котлован не опознан, когда он открыт"


def test_earthwork_ends_once_construction_moves_on(curves) -> None:
    """Признак пропал из кадра — этап закончился и назад уже не вернётся."""
    assert not covers(curves[3], 95), "котлован «идёт» через три этапа после засыпки"


def test_facade_stage_does_not_get_stuck_on_its_own_permanent_evidence(curves) -> None:
    """Ровно тот сценарий, ради которого признаки разделили на два счёта.

    Облицовка и остекление видны и после того, как фасадные работы кончились;
    держали бы они границу — этап не закрылся бы никогда, ведь фасад с дома
    не исчезнет. Держат её только леса и открытый утеплитель, а они уходят.
    """
    assert covers(curves[6], 85), "фасад не опознан в своей фазе"
    assert covers(curves[7], 100), "благоустройство не наступило — граница застряла"


def test_exactly_one_stage_active_per_day(curves) -> None:
    """Последовательность мутуально исключающая: активен ровно один этап."""
    for offset in (8, 40, 70, 85, 100):
        active = [sid for sid, c in curves.items() if covers(c, offset)]
        assert len(active) == 1, f"на день {offset} активными вышли {active}"


def test_reached_marks_stages_the_frontier_passed(curves) -> None:
    assert curves[3].reached, "котлован пройден, а граница ушла дальше"
    assert not curves[7].reached, "текущий этап не может быть отмечен пройденным"


# ---------------------------------------------------------------------------
# инвариант справочника
# ---------------------------------------------------------------------------

def test_every_stage_has_a_transient_gate() -> None:
    """У каждого этапа обязан быть признак, который замолкает по его окончании.

    Это не вкусовщина, а условие работоспособности. Граница прогресса
    сдвигается только когда у текущего этапа замолчали временные признаки.
    Этап, у которого все `must_have` необратимы, замолчать не может никогда —
    на нём граница встанет насмерть, и дальше стройка «не пойдёт» вообще.
    Проверяется на живом справочнике, а не на выдуманных данных.
    """
    for sid, questions in sorted(CHECKLISTS.items()):
        must = [q for q in questions if q["polarity"] == "must_have"]
        transient = [q["key"] for q in must if not q["latching"]]
        assert transient, (
            f"этап {sid}: все признаки необратимые — граница застрянет навсегда")


def test_stages_have_comparable_number_of_features() -> None:
    """Счёт признаков сравнивается между этапами напрямую, без нормировки.

    Значит этап с вдвое большим числом вопросов побеждал бы просто за счёт
    их количества. Разброс держим нулевым.
    """
    counts = {sid: sum(1 for q in qs if q["polarity"] == "must_have")
              for sid, qs in CHECKLISTS.items()}
    assert len(set(counts.values())) == 1, f"признаков по этапам вразнобой: {counts}"


def test_gap_in_observations_breaks_the_interval() -> None:
    """Дыра в наблюдениях не склеивается в один этап.

    Без обрыва последнее наблюдение архива и первое наблюдение сегодняшней
    съёмки оказывались бы краями одного двадцатилетнего отрезка.
    """
    rows = ([(day(i), {1: (3, 3)}) for i in range(5)]
           + [(day(i), {1: (3, 3)}) for i in range(200, 205)])
    assignment = A.sequential_state(rows, order=[1])
    runs = A._runs_from_assignment(assignment)
    assert len(runs[1]) == 2, "наблюдения через полгода склеились в один этап"
    assert runs[1][0][1] < runs[1][1][0]


# ---------------------------------------------------------------------------
# признак, горящий всегда, не имеет права вето
# ---------------------------------------------------------------------------

def always_on_scene(days_total: int, key: str) -> list[tuple[dt.datetime, str, Answer]]:
    """Сцена из двух этапов, где `key` (признак первого) горит каждый день."""
    out = []
    for offset in range(days_total):
        when = dt.datetime.combine(day(offset), dt.time(12, 0))
        scene = {key: "yes"}
        if offset >= 20:                      # второй этап давно начался
            scene |= {"pile_rig": "yes", "pile_stock": "yes",
                      "pile_heads": "yes", "sheet_pile": "yes"}
        for k, v in scene.items():
            out.append((when, k, Answer(v)))
    return out


def two_stage_curves(observations) -> dict[int, A.StageCurve]:
    stages = [(sid, f"этап {sid}", CHECKLISTS[sid]) for sid in (1, 2)]
    return {c.stage_id: c for c in A.build_curves(observations, stages)}


def test_constant_keys_finds_only_the_ever_present_one() -> None:
    days = A.daily_answers(always_on_scene(100, "debris"))
    assert A.constant_keys(days) == {"debris"}


def test_constant_keys_needs_history_before_it_judges() -> None:
    """На короткой съёмке «горит всегда» неотличимо от «горит прямо сейчас».

    Признак идущего этапа в первые дни тоже горит каждый день — отобрать у
    него вето значило бы проскочить этап, который на самом деле идёт.
    """
    days = A.daily_answers(always_on_scene(12, "debris"))
    assert A.constant_keys(days) == set()


def test_ever_present_feature_does_not_freeze_the_frontier() -> None:
    """Регрессия на реальный инцидент: прогон встал на первом этапе.

    «Видны ли кучи строительного мусора» — правда на стройке всегда, а
    числился вопрос временным признаком подготовки территории. Одного такого
    вопроса хватило, чтобы граница не сдвинулась за тринадцать месяцев
    съёмки тремя камерами.
    """
    curves = two_stage_curves(always_on_scene(100, "debris"))
    assert curves[2].intervals, "граница застряла на первом этапе"
    assert curves[1].reached, "первый этап так и не закрылся"


def test_a_real_transient_feature_keeps_its_veto() -> None:
    """Защита снимает вето только с постоянных признаков, не со всех подряд.

    `tree_felling` горит первые сорок дней и гаснет — это нормальный
    временный признак, и держать границу он обязан.
    """
    out = []
    for offset in range(100):
        when = dt.datetime.combine(day(offset), dt.time(12, 0))
        scene = {"pile_rig": "yes", "pile_stock": "yes",
                 "pile_heads": "yes", "sheet_pile": "yes"}
        if offset < 40:
            scene["tree_felling"] = "yes"
        for k, v in scene.items():
            out.append((when, k, Answer(v)))

    curves = two_stage_curves(out)
    first_day_of_stage_2 = min(a for a, _ in curves[2].intervals)
    assert first_day_of_stage_2 >= day(40), (
        "вырубка ещё шла, а граница уже ушла на сваи")
