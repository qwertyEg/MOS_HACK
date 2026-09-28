"""Загрузка плана из файла и поведение камеры с выключенной маской.

Разбор файла пишет человек, а не программа, поэтому главное, что проверяется, —
терпимость к формату и в то же время отсутствие тихих потерь: всё, что не
удалось прочесть, обязано попасть в отчёт.
"""

import datetime as dt

import numpy as np
import pytest

from app import plan_import as P
from app.pipeline import ingest
from app.pipeline import mask as M

D = dt.date

REAL = """\
Подготовка территории: 25/09/2026 - 20/10/2026
Ограждение котлована, шпунт, сваи: 21/10/2026 - 24/01/2027
Земляные работы, котлован: 03/01/2027 - 22/03/2027
"""


# ---------------------------------------------------------------------------
# даты
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, want", [
    ("25/09/2026", D(2026, 9, 25)),
    ("25.09.2026", D(2026, 9, 25)),
    ("25-09-2026", D(2026, 9, 25)),
    ("5.9.2026", D(2026, 9, 5)),
    ("2026-09-25", D(2026, 9, 25)),
    ("  25/09/2026 ", D(2026, 9, 25)),
])
def test_dates_are_read_day_first(raw, want) -> None:
    """01/02/2026 — первое февраля, а не второе января: интерфейс день-первым."""
    assert P.parse_date(raw) == want


def test_ambiguous_date_is_day_first() -> None:
    assert P.parse_date("01/02/2026") == D(2026, 2, 1)


@pytest.mark.parametrize("raw", ["", "31/02/2026", "32.01.2026", "2026-13-01",
                                 "вчера", "25/09/26"])
def test_bad_dates_are_none_not_guessed(raw) -> None:
    assert P.parse_date(raw) is None


# ---------------------------------------------------------------------------
# разбор файла
# ---------------------------------------------------------------------------

def test_the_documented_format_parses() -> None:
    """Формат из ответа пользователю — «Название: дата - дата» — читается."""
    got = P.parse(REAL)
    assert not got.errors
    assert [(r.name, r.start, r.end) for r in got.rows] == [
        ("Подготовка территории", D(2026, 9, 25), D(2026, 10, 20)),
        ("Ограждение котлована, шпунт, сваи", D(2026, 10, 21), D(2027, 1, 24)),
        ("Земляные работы, котлован", D(2027, 1, 3), D(2027, 3, 22)),
    ]


def test_comma_inside_the_name_does_not_split_it() -> None:
    """В названии этапа запятые есть, и это не разделитель."""
    got = P.parse("Ограждение котлована, шпунт, сваи: 07.11.2005 - 10.02.2006")
    assert got.rows[0].name == "Ограждение котлована, шпунт, сваи"


def test_excel_csv_with_semicolons_and_quotes() -> None:
    got = P.parse('"Ограждение котлована, шпунт, сваи";07.11.2005;10.02.2006\n'
                  "Кровля;15.08.2007;15.12.2007")
    assert [r.name for r in got.rows] == ["Ограждение котлована, шпунт, сваи", "Кровля"]
    assert got.rows[1].end == D(2007, 12, 15)


def test_header_bom_blank_and_comment_lines_are_ignored() -> None:
    got = P.parse("﻿этап;начало;конец\n\n# черновик\nКровля: 01.01.2026 - 02.02.2026\n")
    assert len(got.rows) == 1 and not got.errors


def test_iso_dates_and_dash_variants() -> None:
    got = P.parse("Кровля: 2026-01-01 – 2026-02-02\nФасад и остекление: 01.03.2026 — 01.04.2026")
    assert [(r.start, r.end) for r in got.rows] == [
        (D(2026, 1, 1), D(2026, 2, 2)), (D(2026, 3, 1), D(2026, 4, 1))]


def test_unreadable_lines_are_reported_with_line_numbers() -> None:
    """Терпимость не должна вести к тихим потерям."""
    got = P.parse("Кровля: 01.01.2026 - 02.02.2026\n"
                  "Фасад: 32.13.2026 - 01.01.2027\n"
                  "Благоустройство: 10.02.2026 - 01.02.2026\n"
                  "Мусор 12 без дат\n")
    assert len(got.rows) == 1
    text = "\n".join(got.errors)
    assert "строка 2" in text and "нет такой даты" in text
    assert "строка 3" in text and "раньше начала" in text
    assert "строка 4" in text and "не разобрана" in text


def test_cp1251_file_from_russian_excel() -> None:
    """Excel на русской Windows сохраняет CSV в cp1251, а не в UTF-8."""
    data = "Кровля;01.01.2026;02.02.2026".encode("cp1251")
    assert P.parse(P.decode(data)).rows[0].name == "Кровля"


# ---------------------------------------------------------------------------
# сопоставление названий
# ---------------------------------------------------------------------------

KNOWN = {P.norm(n): n for n in (
    "Монолит подземной части", "Монолит надземной части",
    "Земляные работы, котлован", "Кровля")}


def test_match_ignores_case_yo_punctuation_and_spaces() -> None:
    assert P.match("ЗЕМЛЯНЫЕ  работы -- котлован", KNOWN)[1] == "exact"


def test_match_survives_a_typo_and_says_it_guessed() -> None:
    found, how = P.match("Монолит подземной чати", KNOWN)
    assert found == "Монолит подземной части" and how == "fuzzy"


def test_match_never_confuses_underground_with_aboveground() -> None:
    """«Подземной» и «надземной» отличаются двумя буквами — и это разные этапы.

    Сравнение по общей похожести строк их склеивает: строки целиком совпадают
    на девяносто процентов. Поэтому сравнение пословное, с одной опечаткой
    на слово: две правки в слове — уже другое слово.
    """
    only_above = {P.norm("Монолит надземной части"): "над"}
    assert P.match("Монолит подземной части", only_above) == (None, "")

    both = {P.norm("Монолит подземной части"): "под",
            P.norm("Монолит надземной части"): "над"}
    assert P.match("Монолит подземнои части", both) == ("под", "fuzzy")
    assert P.match("Монолит надземнои части", both) == ("над", "fuzzy")


def test_short_words_do_not_tolerate_typos() -> None:
    """В коротком слове ошибка — уже другое слово: «кров» не «кровля»."""
    assert P.match("Кров", KNOWN) == (None, "")


def test_missing_space_is_reported_rather_than_guessed() -> None:
    """Слипшиеся слова — не опечатка в слове; лучше переспросить, чем угадать."""
    assert P.match("Монолит подземнойчасти", KNOWN) == (None, "")


def test_unknown_name_is_not_matched() -> None:
    assert P.match("Покраска забора", KNOWN) == (None, "")


# ---------------------------------------------------------------------------
# камера без маски
# ---------------------------------------------------------------------------

def test_blank_state_leaves_the_frame_untouched() -> None:
    """Пустая маска ведёт себя как отсутствие маски — отдельной ветки не нужно."""
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (270, 480, 3), dtype=np.uint8)
    st = ingest.blank_state(frame)

    assert np.array_equal(M.render_masked(frame, st), frame)
    assert np.array_equal(M.render_overlay(frame, st), frame)
    assert st.masked_ratio == 0.0


def test_blank_state_is_not_useful_so_the_model_gets_the_full_frame() -> None:
    """`useful` ложно → в модель Б уходит mask=None, то есть кадр целиком."""
    frame = np.zeros((270, 480, 3), dtype=np.uint8)
    assert ingest.blank_state(frame).useful is False


def test_blank_state_matches_the_working_resolution() -> None:
    """Размер пустой маски совпадает с рабочим, иначе to_full_res поплывёт."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert ingest.blank_state(frame).shape == ingest.work_shape(frame)
