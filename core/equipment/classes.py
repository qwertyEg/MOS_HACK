"""Имена классов техники → канонические ключи словаря (reference/checklist.json).

Детекторы говорят на разных языках: датасеты для YOLO называют классы
«Dump Truck», «Pump truck», «Static crane» (MOCS), VLM отвечает «самосвал»
или «concrete mixer truck». Ядро знает только 21 ключ из словаря, поэтому
всё переводится здесь, в одном месте. `None` в таблице — класс осознанно
выбрасывается (люди, легковые, крюк крана).
"""
from __future__ import annotations

import re
from functools import lru_cache

from core import taxonomy

# Нормализованное имя (нижний регистр, пробелы вместо _ и -) → ключ или None.
CLASS_ALIASES: dict[str, str | None] = {
    # экскаваторы и погрузчики
    "excavator": "excavator", "digger": "excavator", "crawler excavator": "excavator",
    "wheeled excavator": "excavator", "mini excavator": "excavator", "demolition excavator": "excavator",
    "backhoe": "backhoe_loader", "backhoe loader": "backhoe_loader", "jcb": "backhoe_loader",
    "loader": "wheel_loader", "wheel loader": "wheel_loader", "front loader": "wheel_loader",
    "front end loader": "wheel_loader", "skid steer": "skid_steer", "skid steer loader": "skid_steer",
    "bobcat": "skid_steer", "mini loader": "skid_steer",
    "telehandler": "telehandler", "telescopic handler": "telehandler", "telescopic loader": "telehandler",
    # землеройные и дорожные
    "bulldozer": "bulldozer", "dozer": "bulldozer", "roller": "roller", "road roller": "roller",
    "compactor": "roller", "vibratory roller": "roller", "grader": "grader", "motor grader": "grader",
    "asphalt paver": "asphalt_paver", "paver": "asphalt_paver",
    # грузовики
    "dump truck": "dump_truck", "dumper": "dump_truck", "tipper": "dump_truck", "tipper truck": "dump_truck",
    "truck": "truck", "flatbed truck": "truck", "lorry": "truck", "trailer": "truck",
    "semi trailer": "truck", "semi truck": "truck", "heavy truck": "truck",
    "concrete mixer": "concrete_mixer", "concrete mixer truck": "concrete_mixer", "mixer truck": "concrete_mixer",
    "cement truck": "concrete_mixer", "cement mixer": "concrete_mixer", "transit mixer": "concrete_mixer",
    "concrete pump": "concrete_pump", "concrete pump truck": "concrete_pump", "pump truck": "concrete_pump",
    "boom pump": "concrete_pump",
    # краны
    "mobile crane": "mobile_crane", "truck crane": "mobile_crane", "crane": "mobile_crane",
    "autocrane": "mobile_crane", "all terrain crane": "mobile_crane",
    "crane manipulator": "crane_manipulator", "loader crane": "crane_manipulator",
    "knuckle boom crane": "crane_manipulator", "knuckle boom truck": "crane_manipulator",
    "truck with crane": "crane_manipulator", "hiab": "crane_manipulator",
    "tower crane": "tower_crane", "static crane": "tower_crane",
    "crawler crane": "crawler_crane", "lattice boom crane": "crawler_crane",
    # сваи и бурение
    "drilling rig": "drilling_rig", "drill rig": "drilling_rig", "piling rig": "drilling_rig",
    "pile driver": "pile_driver", "pile driving": "pile_driver", "pile driving machine": "pile_driver",
    "vibro hammer": "pile_driver",
    # подъёмники
    "aerial platform": "aerial_platform", "aerial work platform": "aerial_platform", "boom lift": "aerial_platform",
    "cherry picker": "aerial_platform", "scissor lift": "aerial_platform",
    "construction hoist": "facade_hoist", "facade hoist": "facade_hoist", "gondola": "facade_hoist",
    "hoist": "facade_hoist",
    # осознанно не техника
    "person": None, "worker": None, "people": None, "helmet": None, "car": None, "van": None,
    "hanging head": None, "hook": None, "other vehicle": None, "vehicle": None, "bus": None,
}


def _norm(name: str) -> str:
    s = str(name).strip().lower().replace("ё", "е")
    s = re.sub(r"[_\-/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canonical_class(name: str | None, extra: dict[str, str | None] | None = None) -> str | None:
    """Ключ словаря для имени класса или None (неизвестно или осознанно не техника).

    `extra` — карта из equipment_classes.json («класс датасета → ключ»),
    она важнее встроенной таблицы.
    """
    if name is None:
        return None
    raw = str(name).strip()
    keys = taxonomy.equipment()
    if extra:
        if raw in extra:
            return extra[raw] if extra[raw] in keys else None
        n = _norm(raw)
        for k, v in extra.items():
            if _norm(k) == n:
                return v if v in keys else None
    if raw in keys:
        return raw
    n = _norm(raw)
    as_key = n.replace(" ", "_")
    if as_key in keys:
        return as_key
    if n in CLASS_ALIASES:
        return CLASS_ALIASES[n]
    ru = _ru_names()
    if n in ru:
        return ru[n]
    return None


@lru_cache(maxsize=1)
def _ru_names() -> dict[str, str]:
    out = {_norm(k): v for k, v in taxonomy.RU_ALIASES.items()}
    for key, e in taxonomy.equipment().items():
        out[_norm(e.name)] = key
        # «Грузовик (бортовой, длинномер, трал)» → ещё и «грузовик»
        short = _norm(e.name.split("(")[0].split("/")[0])
        out.setdefault(short, key)
    return out
