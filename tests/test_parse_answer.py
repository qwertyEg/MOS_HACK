"""Разбор ответа модели Б. Место хрупкое: модель отвечает свободным текстом,
а от результата зависит весь вердикт, поэтому проверяется явно.
"""

import pytest

from app.pipeline.model_b import Answer, parse_answer

CASES = [
    # односложные ответы — как просим в промпте
    ("Да", Answer.YES),
    ("да.", Answer.YES),
    ("ДА", Answer.YES),
    ("Нет", Answer.NO),
    ("нет,", Answer.NO),
    ("No", Answer.NO),
    # тернарность: «не уверен» не должно читаться как «нет»
    ("Не уверен", Answer.UNSURE),
    ("не уверен.", Answer.UNSURE),
    ("Не уверена", Answer.UNSURE),
    ("Я не могу определить", Answer.UNSURE),
    ("Сложно сказать, кадр засвечен", Answer.UNSURE),
    # ответ на вопрос «стройка или завершено»
    ("Стройка", Answer.YES),
    ("завершено", Answer.NO),
    # модель не послушалась и ответила развёрнуто
    ("**Да**", Answer.YES),
    ("Да, виден котлован", Answer.YES),
    ("Нет, конструкций не видно", Answer.NO),
    ("На изображении да, присутствует", Answer.YES),
    # мусор → не голосует
    ("", Answer.UNSURE),
    ("Изображение показывает строительную площадку", Answer.UNSURE),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_parse_answer(raw: str, expected: Answer) -> None:
    assert parse_answer(raw) == expected
