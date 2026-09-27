"""Единый словарь: техника, этапы, признаки.

Источник истины — `reference/checklist.json` (справочник Никиты, собран из
xlsx организаторов и расширен от справочника Дениса: 8 макроэтапов, 32
подэтапа, 60 признаков, 21 тип техники). Здесь только удобный доступ к нему.
Старые русские названия техники из ветки Дениса (`reference/legacy_dev/
stages.csv`) переводятся в ключи через `RU_ALIASES`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
CHECKLIST_PATH = REFERENCE_DIR / "checklist.json"

# Восемь классов, которые ТЗ (раздел 6) требует распознавать обязательно.
TZ_EQUIPMENT = (
    "dump_truck", "excavator", "roller", "crane_manipulator",
    "concrete_mixer", "bulldozer", "truck", "mobile_crane",
)

# Группы, внутри которых детектор путает классы: трекер разрешает менять
# метку трека внутри группы (итоговая метка — большинством голосов).
CONFUSABLE_GROUPS = (
    {"truck", "dump_truck", "concrete_mixer", "crane_manipulator", "concrete_pump", "mobile_crane"},
    {"excavator", "backhoe_loader"},
    {"wheel_loader", "skid_steer", "backhoe_loader", "telehandler"},
    {"drilling_rig", "pile_driver", "crawler_crane"},
)

# Русские названия из ветки Дениса → ключи.
RU_ALIASES = {
    "автобетононасос": "concrete_pump",
    "автобетоносмеситель": "concrete_mixer",
    "бетоносмеситель": "concrete_mixer",
    "автокран": "mobile_crane",
    "башенный кран": "tower_crane",
    "бульдозер": "bulldozer",
    "буровая установка": "drilling_rig",
    "гусеничный кран": "crawler_crane",
    "каток": "roller",
    "копёр": "pile_driver",
    "погрузчик": "wheel_loader",
    "подъёмник": "facade_hoist",
    "самосвал": "dump_truck",
    "экскаватор": "excavator",
    "грузовик": "truck",
    "кран-манипулятор": "crane_manipulator",
}


@dataclass(frozen=True)
class EquipmentType:
    key: str
    name: str            # по-русски, для UI
    tz: bool             # обязателен по ТЗ
    look: str = ""       # как выглядит — для промптов VLM и документации


@dataclass(frozen=True)
class Stage:
    id: int
    key: str
    name: str
    weight: float
    must_have: tuple[str, ...]
    must_not_have: tuple[str, ...]
    equipment_expected: tuple[str, ...]
    equipment_optional: tuple[str, ...]
    equipment_forbidden: tuple[str, ...]
    substages: tuple[dict, ...] = field(default_factory=tuple)
    raw: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class Sign:
    key: str
    question: str
    hint: str
    latching: bool       # однажды увиденный признак не исчезает (плита залита — навсегда)


@lru_cache(maxsize=1)
def checklist() -> dict:
    return json.loads(CHECKLIST_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def equipment() -> dict[str, EquipmentType]:
    out = {}
    for e in checklist()["equipment"]:
        out[e["key"]] = EquipmentType(e["key"], e["name"], bool(e.get("tz")), e.get("look", ""))
    return out


@lru_cache(maxsize=1)
def stages() -> dict[int, Stage]:
    out = {}
    for s in checklist()["stages"]:
        out[int(s["id"])] = Stage(
            id=int(s["id"]), key=s["key"], name=s["name"], weight=float(s.get("weight", 0)),
            must_have=tuple(s.get("must_have", [])), must_not_have=tuple(s.get("must_not_have", [])),
            equipment_expected=tuple(s.get("equipment_expected", [])),
            equipment_optional=tuple(s.get("equipment_optional", [])),
            equipment_forbidden=tuple(s.get("equipment_forbidden", [])),
            substages=tuple(s.get("substages", [])), raw=s,
        )
    return out


@lru_cache(maxsize=1)
def signs() -> dict[str, Sign]:
    out = {}
    for s in checklist()["signs"]:
        out[s["key"]] = Sign(s["key"], s["question"], s.get("hint", ""), bool(s.get("latching")))
    return out


def equipment_name(key: str) -> str:
    e = equipment().get(key)
    return e.name if e else key


def stage_name(stage_id: int | None) -> str:
    if stage_id is None:
        return "—"
    s = stages().get(int(stage_id))
    return s.name if s else f"Этап {stage_id}"


def confusable(a: str, b: str) -> bool:
    return a == b or any(a in g and b in g for g in CONFUSABLE_GROUPS)
