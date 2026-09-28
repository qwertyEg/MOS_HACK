"""Скоринг ответов чек-листа: свойства, которые наследие нарушало (баги A5, A6)."""
import random

import pytest

from core import taxonomy
from core.contracts import Answer
from core.stage import scoring
from core.stage.scoring import ACTIVE, DONE, NOT_STARTED, UNKNOWN

Y, N, U = Answer.YES, Answer.NO, Answer.UNSURE
STAGES = taxonomy.stages()
MODEL = scoring.model()
POSITIVE = sorted(MODEL.positive)
ALL_SIGNS = sorted(taxonomy.signs())


def test_unsure_does_not_vote():
    base = {"pit": Y, "soil_pile": Y, "earthwork": N}
    with_unsure = {**base, **{k: U for k in ALL_SIGNS if k not in base}}
    a, b = scoring.evaluate(base), scoring.evaluate(with_unsure)
    assert a.stage_evidence == b.stage_evidence and a.progress == b.progress and a.front == b.front
    # доля «да» среди проголосовавших, со скепсисом c = 0.5: 2 / (3 + 0.5)
    assert a.stage_evidence[3] == pytest.approx(2 / 3.5, abs=1e-3)


def test_replacing_no_by_unsure_never_lowers_evidence():
    """Раньше not_visible делил знаменатель наравне с «нет»: облицовка «да» при двух
    «не видно» давала этапу 7 всего 0.33 — ниже порога, фасад не открывался."""
    facade = {"cladding": Y, "glazing": U, "scaffold": U}
    s = scoring.evaluate(facade)
    assert s.front == 7 and s.stage_evidence[7] >= 0.5
    for k in ("glazing", "scaffold"):
        worse = scoring.evaluate({**facade, k: N})
        assert worse.stage_evidence[7] <= s.stage_evidence[7]


def test_more_yes_never_lowers_progress_property():
    """Главное свойство: превращение любого ответа на положительный признак в «да» не
    снижает ни готовность объекта, ни готовность какого-либо этапа, и фронт не падает."""
    rnd = random.Random(20260928)
    for _ in range(400):
        answers = {k: rnd.choice([Y, N, U, U]) for k in ALL_SIGNS if rnd.random() < 0.6}
        before = scoring.evaluate(answers)
        key = rnd.choice(POSITIVE)
        after = scoring.evaluate({**answers, key: Y})
        assert after.overall >= before.overall - 1e-9, (key, answers)
        for sid in STAGES:
            assert after.progress[sid] >= before.progress[sid] - 1e-9, (sid, key)
        assert (after.front or 0) >= (before.front or 0)


def test_every_substage_can_become_done_and_every_stage_reach_100():
    """Раньше 14 подэтапов не могли стать DONE, объект не бывал готов выше 99.5 %, этап 8 — никогда."""
    for sid, stage in STAGES.items():
        answers = {k: Y for k in stage.must_have}
        for sub in stage.substages:
            answers.update({k: Y for k in sub["done_when"]})
        assert scoring.evaluate(answers).progress[sid] == pytest.approx(1.0), sid
        for i, sub in enumerate(stage.substages):
            if sub["done_when"]:
                votes = scoring.answers_to_votes({k: Y for k in sub["done_when"] + sub["active_when"]})
                assert scoring.substage_status(sub, votes) == DONE, sub["id"]
            else:
                # без признака готовности подэтап засчитывается, когда начат следующий
                assert i + 1 < len(stage.substages), sub["id"]
    s = scoring.evaluate({k: Y for k in POSITIVE})
    assert s.front == 8 and s.overall == pytest.approx(1.0)


def test_done_beats_active_and_substage_order_is_respected():
    # окна стоят, но есть и пустые проёмы: подэтап 7.1 готов (DONE старше ACTIVE)
    votes = scoring.answers_to_votes({"glazing": Y, "window_openings_empty": Y})
    assert scoring.substage_statuses(votes)["7.1"] == DONE
    # увиден более поздний подэтап — предыдущий без ответа засчитан целиком
    later = scoring.substage_statuses(scoring.answers_to_votes({"curbs": Y}))
    assert later["8.3"] == ACTIVE and later["8.1"] == UNKNOWN
    assert scoring.stage_progress_from_substages(8, later) == pytest.approx((25 + 15 + 30 * 0.5) / 100)
    assert scoring.substage_statuses(scoring.answers_to_votes({"curbs": N, "asphalt_work": N}))["8.3"] == NOT_STARTED


def test_stage_is_opened_only_by_answers_not_by_shared_signs():
    """Опалубка и арматура есть и у обвязочной балки (2.4): без плиты и «ниже земли»
    монолит подземной части не открывается."""
    s = scoring.evaluate({"formwork": Y, "rebar": Y, "slab": N, "below_grade": N, "rig": Y, "pile_heads": Y})
    assert s.front == 2
    assert s.stage_evidence[4] < 0.5


def test_purely_negative_sign_penalises_its_stage():
    clad = {"cladding": Y, "glazing": Y, "scaffold": Y}
    assert scoring.evaluate(clad).front == 7
    raw = scoring.evaluate({**clad, "bare_concrete": Y})
    assert raw.stage_evidence[7] < scoring.evaluate(clad).stage_evidence[7]


def test_answers_in_any_format_are_normalised():
    s = scoring.evaluate({"pit": "yes", "soil_pile": "да", "earthwork": "not_visible", "above_grade": "no"})
    assert s.front == 3 and s.votes[3] == pytest.approx(2.0)
    assert scoring.normalize_answer("мусор") is Answer.UNSURE
    assert scoring.normalize_answer(True) is Answer.YES


def test_empty_answers_give_no_front_and_zero_progress():
    s = scoring.evaluate({})
    assert s.front is None and s.overall == 0 and s.stage_evidence == {}
    assert all(v == UNKNOWN for v in s.substages.values())


def test_day_votes_from_several_cameras_are_fractional():
    """Две камеры видят котлован, третья — нет: доказательность ниже, чем при единогласии."""
    unanimous = scoring.answers_to_votes({"pit": Y, "soil_pile": Y})
    split = scoring.add_votes(scoring.answers_to_votes({"pit": Y, "soil_pile": Y}),
                              scoring.answers_to_votes({"pit": N, "soil_pile": Y}))
    assert scoring.evaluate_votes(split).stage_evidence[3] < scoring.evaluate_votes(unanimous).stage_evidence[3]
    assert scoring.sign_state(scoring.add_votes({"pit": (1, 0)}, {"pit": (0, 1)}), "pit") is None   # ничья
