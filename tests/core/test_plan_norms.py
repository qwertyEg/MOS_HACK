"""Нормы «этап → техника»: канон equipment_rules.json согласован со словарём, ТЗ и справочником Дениса."""
import csv

import pytest

from core import taxonomy
from core.plan import norms


def test_rules_are_consistent_with_taxonomy():
    assert norms.validate() == []


def test_every_stage_has_rules():
    assert norms.stage_ids() == list(range(1, 9))
    for s in norms.stage_ids():
        req = norms.requirement(s)
        assert req.expected and req.forbidden
        assert req.min_count, f"этап {s}: без обязательного минимума нечем проверять «нет техники»"


def test_all_tz_classes_are_covered():
    """Восемь классов ТЗ должны где-то быть ожидаемыми или допустимыми — иначе детектор их видит зря."""
    allowed = set().union(*(norms.requirement(s).allowed for s in norms.stage_ids()))
    assert set(taxonomy.TZ_EQUIPMENT) <= allowed
    expected = set().union(*(norms.requirement(s).expected for s in norms.stage_ids()))
    # кран-манипулятор и грузовик: в справочнике Дениса их не было совсем, у Никиты грузовик нигде не ожидался
    assert {"crane_manipulator", "truck"} <= expected


def test_mandatory_pairs():
    pairs = {p["id"]: p for p in norms.all_pairs()}
    must = {
        "excavator_dump_truck": ("excavator", "dump_truck"),
        "concrete_pump_mixer": ("concrete_pump", "concrete_mixer"),
        "asphalt_paver_roller": ("asphalt_paver", "roller"),
        "drilling_rig_mixer": ("drilling_rig", "concrete_mixer"),
    }
    for pid, (leader, follower) in must.items():
        assert leader in pairs[pid]["leader"] and follower in pairs[pid]["followers"]
    # эталон ТЗ: на котловане пара экскаватор↔самосвал с окном 2 ч и уровнем warning
    p3 = {p.id: p for p in norms.requirement(3).pairs}["excavator_dump_truck"]
    assert (p3.severity, p3.window_h) == ("warning", 2.0)
    assert p3.title.startswith("Возможное снижение темпа")


def test_roller_is_allowed_on_excavation():
    """Каток на котловане не аномалия: уплотнение дна и песчаного основания (подэтап 3.3)."""
    req = norms.requirement(3)
    assert "roller" in req.optional and "roller" not in req.forbidden


def test_required_types_are_allowed_and_fleet_is_a_copy():
    for s in norms.stage_ids():
        req = norms.requirement(s)
        for t in req.min_count:
            assert set(req.accepted(t)) & req.allowed
        fleet = norms.default_equipment(s)
        assert fleet and set(fleet) <= req.allowed
        fleet["excavator"] = 99
        assert norms.default_equipment(s).get("excavator") != 99


def test_unknown_stage_is_explicit_error():
    with pytest.raises(KeyError, match="этапа 42"):
        norms.requirement(42)


def test_divergence_from_legacy_dev_is_only_the_documented_one():
    """Сверка со справочником Дениса: его ожидаемая техника у нас допустима, запрещённая — запрещена,
    кроме трёх осознанных решений из docs/methodology.md."""
    path = taxonomy.REFERENCE_DIR / "legacy_dev" / "stages.csv"
    diverged = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            req = norms.requirement(int(row["id"]))
            for ru in filter(None, row["equipment_expected"].split(";")):
                assert taxonomy.RU_ALIASES[ru] in req.allowed, (row["id"], ru)
            for ru in filter(None, row["equipment_forbidden"].split(";")):
                key = taxonomy.RU_ALIASES[ru]
                if key not in req.forbidden:
                    diverged.add((int(row["id"]), key))
    assert diverged == {(2, "concrete_pump"), (3, "roller"), (4, "roller")}
