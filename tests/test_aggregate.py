"""Свёртка хронологии чек-листов в активные этапы.

Сцена собрана так, чтобы проверить главное свойство метода: вывод об этапе
делается по смене ответов во времени, а не по одному кадру. Стройка проходит
три этапа подряд, и на каждом модель видит своё.

Отдельно проверяется случай, ради которого всё и затевалось: признак,
который был виден, а потом перестал (котлован перекрыт выросшим зданием).
Этап должен закончиться, а не «продолжаться» из-за того, что котлован
когда-то был.
"""

import datetime as dt

import pytest

from app.models import Answer
from app.pipeline import aggregate as A
from app.seed import load_checklists

START = dt.date(2026, 1, 1)
CHECKLISTS = load_checklists()

# что видно на площадке в каждый период
SCENE = [
    # (первый день, последний день, {ключ: ответ})
    (0, 40, {"pit": "yes", "soil_pile": "yes", "earthwork": "yes",
             "above_grade": "no", "cladding": "no", "glazing": "no",
             "is_construction": "yes"}),
    (41, 85, {"pit": "unsure", "soil_pile": "no", "earthwork": "no",
              "above_grade": "yes", "crane": "yes", "formwork_floor": "yes",
              "unfinished_top": "yes", "cladding": "no", "glazing": "no",
              "is_construction": "yes"}),
    (86, 130, {"pit": "unsure", "above_grade": "yes", "crane": "no",
               "formwork_floor": "no", "unfinished_top": "no",
               "scaffold": "yes", "cladding": "yes", "glazing": "yes",
               "bare_concrete": "no", "is_construction": "yes"}),
]


def observations() -> list[tuple[dt.datetime, str, Answer]]:
    """Два кадра в день, как с реальной камеры."""
    out = []
    for lo, hi, scene in SCENE:
        for offset in range(lo, hi + 1):
            when = dt.datetime.combine(START + dt.timedelta(days=offset),
                                       dt.time(12, 0))
            for key, value in scene.items():
                for shot in range(2):
                    out.append((when + dt.timedelta(hours=shot), key,
                                Answer(value)))
    return out


@pytest.fixture(scope="module")
def curves() -> dict[int, A.StageCurve]:
    stages = [(sid, f"этап {sid}", qs) for sid, qs in CHECKLISTS.items()]
    return {c.stage_id: c for c in A.build_curves(observations(), stages)}


def covers(curve: A.StageCurve, offset: int) -> bool:
    day = START + dt.timedelta(days=offset)
    return any(a <= day <= b for a, b in curve.intervals)


def test_earthwork_detected_at_start(curves) -> None:
    assert covers(curves[3], 20), "котлован не опознан, когда он открыт"


def test_earthwork_ends_when_building_rises(curves) -> None:
    """Главное: признак пропал из кадра — этап закончился.

    Если бы «котлован» считался вечным (был же когда-то), этап тянулся бы
    до конца стройки и весь график съехал бы.
    """
    assert not covers(curves[3], 110), "котлован «идёт» спустя месяцы после засыпки"


def test_frame_stage_detected_in_middle(curves) -> None:
    assert covers(curves[5], 60), "монолит надземной части не опознан"
    assert not covers(curves[5], 10), "монолит опознан там, где ещё котлован"


def test_facade_detected_at_end(curves) -> None:
    assert covers(curves[7], 110), "фасад не опознан"
    assert not covers(curves[7], 20), "фасад опознан в начале стройки"


def test_stages_do_not_all_fire_at_once(curves) -> None:
    """Мультилейбл — норма, но не все восемь этапов разом."""
    day = 60
    active = [sid for sid, c in curves.items() if covers(c, day)]
    assert 1 <= len(active) <= 3, f"на день {day} активными вышли {active}"


def test_reached_survives_occlusion(curves) -> None:
    """Котлован перекрыт, но веха пройдена — это разные вещи."""
    assert curves[3].reached, "этап не отмечен пройденным, хотя признак видели"


def test_unsure_does_not_vote() -> None:
    """Кадр, на котором ничего не разглядеть, не должен опускать оценку."""
    questions = CHECKLISTS[5]
    blind = {q["key"]: Answer.UNSURE for q in questions}
    point = A.stage_score(blind, questions)
    assert point.votes == 0
    assert point.confidence == 0.0


def test_confidence_reflects_visibility() -> None:
    questions = CHECKLISTS[7]
    half = {}
    voting = [q for q in questions if q["polarity"] != "context"]
    for i, q in enumerate(voting):
        half[q["key"]] = Answer.YES if i % 2 == 0 else Answer.UNSURE
    point = A.stage_score(half, questions)
    assert 0.0 < point.confidence < 1.0
