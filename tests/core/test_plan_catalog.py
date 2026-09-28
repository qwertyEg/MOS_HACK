"""Каталог работ организаторов: канон work_map.csv, починка кодов-дат, поиск строк."""
import datetime as dt
import re

import pytest

from core.plan import catalog


def test_all_rows_loaded_with_known_statuses():
    items = catalog.load()
    assert len(items) == 377
    counts = {s: sum(i.status == s for i in items) for s in catalog.STATUSES}
    assert counts == {"substage": 188, "unobservable": 122, "out_of_scope": 62, "header": 5}
    # у каждой наблюдаемой работы есть этап и подэтап — иначе UI не покажет, что сейчас идёт
    assert all(i.stage_id and i.substage_id for i in items if i.status == "substage")


def test_catalog_codes_have_no_excel_dates():
    """В xlsx 19 кодов («10.1.», «12.3.») сохранены Excel как даты; в каталоге их быть не должно."""
    for it in catalog.load():
        assert not it.code or re.fullmatch(r"\d+(\.\d+)*\.?", it.code), it.code
    assert catalog.get("10.1.").name == "Отселение домов в пятне застройки"
    assert catalog.get("12.3.").status == "header"


@pytest.mark.parametrize("raw, code", [
    (dt.datetime(2025, 1, 10), "10.1."),       # ячейка-дата в xlsx
    (dt.date(2025, 3, 12), "12.3."),
    (45667, "10.1."),                          # сериал Excel (2025-01-10) без формата даты
    ("10.01.2025", "10.1."),                   # CSV-выгрузка значения
    ("2025-03-12", "12.3."),
    ("10.01", "10.1."),                        # ведущий ноль — это дата, у организаторов нулей нет
    ("10.янв", "10.1."),
    ("10.10", "10.10."),                       # а это настоящий код 10.10.
    ("12.3.1", "12.3.1."),
    (12, "12."),
    (" 12.4.28. ", "12.4.28."),
    (None, ""),
    ("NaT", ""),
])
def test_normalize_code_repairs_excel_damage(raw, code):
    assert catalog.normalize_code(raw)[0] == code


def test_float_code_is_flagged():
    code, note = catalog.normalize_code(12.3)
    assert code == "12.3." and "проверьте" in note


def test_works_for_stage_and_substage():
    works = catalog.works_for_stage(3)
    assert works and all(w.stage_id == 3 and w.status == "substage" for w in works)
    assert "12.3.1." in {w.code for w in works}
    sub = catalog.works_for_stage(3, "3.3")
    assert sub and {w.substage_id for w in sub} == {"3.3"}
    # невидимые работы этапа 1 (геодезия, отселение) — только по флагу
    assert "10.7." not in {w.code for w in catalog.works_for_stage(1)}
    assert "10.7." in {w.code for w in catalog.works_for_stage(1, include_unobservable=True)}


def test_resolve_detail_row_and_ancestor():
    # «Обратная засыпка» внутри «12.3.7. Земляные работы» — этап 4 (пазухи), а не 3
    item, how = catalog.resolve("", "Обратная засыпка", "12.3.7.")
    assert (item.stage_id, item.substage_id, how) == (4, "4.6", "")
    assert item.key == "12.3.7./Обратная засыпка"
    assert catalog.get(item.key) is item
    # код глубже перечня — по ближайшему предку, с пояснением
    item, how = catalog.resolve("12.3.7.9.")
    assert item.code == "12.3.7." and "родительскому" in how
    assert catalog.resolve("99.9.") == (None, "")


def test_levels_and_labels():
    assert catalog.get("10.").level == 1
    assert catalog.get("10.11.1.").level == 3
    detail = catalog.get("12.3.7./Разработка грунта")
    assert detail.level == catalog.get("12.3.7.").level + 1
    assert catalog.label("12.3.1.") == "12.3.1. Устройство котлована"
    assert catalog.label("нет-такого") == "нет-такого"


def test_road_only_rows_get_fallback_stage():
    """Строки «только для дорог» не теряются при импорте графика здания: этап — по раскладке Дениса."""
    it = catalog.get("10.11.1.")
    assert it.status == "out_of_scope" and it.stage_id is None
    assert it.plan_stage_id == 1
    # внутренние работы без этапа в план/факт не входят
    assert catalog.get("12.6.2.").plan_stage_id is None


def test_stages_under_sections():
    assert catalog.stages_under("10.") == {1}
    assert {2, 3, 4} <= catalog.stages_under("12.3.")
    assert catalog.stages_under("12.7.") == {8}


def test_legacy_discrepancies_match_methodology():
    """47 строк разложены ветками по-разному — ровно столько разобрано в docs/methodology.md."""
    diff = catalog.legacy_discrepancies()
    assert len(diff) == 47
    kinds = {k: sum(d["kind"] == k for d in diff) for k in {d["kind"] for d in diff}}
    assert kinds == {"разные этапы": 15, "Никита: видно, Денис: исключено": 16,
                     "Никита: не видно с камеры, Денис: видно": 16}
    doc = (catalog.REFERENCE_DIR.parent / "docs" / "methodology.md").read_text(encoding="utf-8")
    for code in ("12.3.7./Обратная засыпка", "12.4.28.", "12.5.1.", "12.6.1.", "10.13."):
        assert code.split("/")[0] in doc
