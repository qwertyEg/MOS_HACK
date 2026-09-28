"""Нормы «этап → техника» — методика сопоставления из ТЗ.

Канон — `reference/equipment_rules.json` (читаемый человеком, правится без кода).
Он собран из equipment_expected/optional/forbidden справочника checklist.json,
сверен со справочником Дениса (reference/legacy_dev/stages.csv) и дополнен тем,
чего в обоих не было: пары «ведущая ↔ обслуживающая техника», обязательный минимум
с допустимыми заменами и окна, за которые отсутствие техники — уже отклонение.
Расхождения с checklist.json и их обоснование — docs/methodology.md.

Почему не читаем списки прямо из checklist.json: там expected означает «типичная»
(на этапе 1 — пять типов сразу), и требование «все ожидаемые должны быть» дало бы
ложные тревоги. Для отклонений нужен отдельный, более строгий слой — min_count.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from core import taxonomy

RULES_PATH = taxonomy.REFERENCE_DIR / "equipment_rules.json"

SEVERITIES = ("info", "warning", "critical")


@dataclass(frozen=True)
class Pair:
    """Пара «ведущая ↔ обслуживающая» техника для конкретного этапа.

    leader_state="active": ведущая работает (экскаватор копает), а обслуживающей нет рядом.
    leader_state="present": ведущие стоят в очереди (≥ leader_min_units), а обслуживающая
    не работает (самосвалы ждут погрузки).
    """
    id: str
    leader: tuple[str, ...]
    followers: tuple[str, ...]
    severity: str
    window_h: float
    escalate_after_h: float | None
    leader_state: str = "active"
    follower_state: str = "present"
    leader_min_units: int = 1
    title: str = ""
    why: str = ""
    check: str = ""


@dataclass
class StageRequirement:
    """Что этап требует от техники. Поля expected…min_count — договор из ARCHITECTURE.md;
    substitutes, missing_window_h, forbidden_info, default_fleet — расширение модуля плана."""
    stage_id: int
    expected: tuple[str, ...]
    optional: tuple[str, ...]
    forbidden: tuple[str, ...]
    pairs: tuple[Pair, ...]
    min_count: dict[str, int]
    substitutes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    missing_window_h: dict[str, float] = field(default_factory=dict)
    forbidden_info: tuple[str, ...] = ()
    default_fleet: dict[str, int] = field(default_factory=dict)
    note: str = ""

    @property
    def allowed(self) -> set[str]:
        """Техника, которая на этапе не аномалия."""
        return set(self.expected) | set(self.optional)

    def accepted(self, cls: str) -> tuple[str, ...]:
        """Типы, которые засчитываются за обязательный `cls` (сам тип + замены)."""
        return (cls, *self.substitutes.get(cls, ()))

    def stage_types(self) -> set[str]:
        """Вся «своя» техника этапа: ожидаемая и замены обязательной."""
        out = set(self.expected)
        for cls in self.min_count:
            out.update(self.accepted(cls))
        return out


@lru_cache(maxsize=4)
def _raw(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pairs_for(stage_id: int, raw: dict) -> tuple[Pair, ...]:
    out = []
    for p in raw.get("pairs", []):
        sev = p.get("stages", {}).get(str(stage_id))
        if not sev:
            continue
        out.append(Pair(
            id=p["id"], leader=tuple(p["leader"]), followers=tuple(p["followers"]), severity=sev,
            window_h=float(p.get("window_h") or 2.0),
            escalate_after_h=float(p["escalate_after_h"]) if p.get("escalate_after_h") else None,
            leader_state=p.get("leader_state", "active"), follower_state=p.get("follower_state", "present"),
            leader_min_units=int(p.get("leader_min_units", 1)),
            title=p.get("title", ""), why=p.get("why", ""), check=p.get("check", ""),
        ))
    return tuple(out)


@lru_cache(maxsize=16)
def requirement(stage_id: int) -> StageRequirement:
    """Требования этапа 1..8 к технике. Неизвестный этап — KeyError с понятным текстом."""
    raw = _raw()
    s = raw["stages"].get(str(int(stage_id)))
    if s is None:
        raise KeyError(f"нет правил техники для этапа {stage_id} (известны 1–8)")
    return StageRequirement(
        stage_id=int(stage_id),
        expected=tuple(s.get("expected", [])),
        optional=tuple(s.get("optional", [])),
        forbidden=tuple(s.get("forbidden", [])),
        pairs=_pairs_for(int(stage_id), raw),
        min_count={k: int(v) for k, v in s.get("min_count", {}).items()},
        substitutes={k: tuple(v) for k, v in s.get("substitutes", {}).items()},
        missing_window_h={k: float(v) for k, v in s.get("missing_window_h", {}).items()},
        forbidden_info=tuple(s.get("forbidden_info", [])),
        default_fleet={k: int(v) for k, v in s.get("default_fleet", {}).items()},
        note=s.get("note", ""),
    )


def stage_ids() -> list[int]:
    return sorted(int(k) for k in _raw()["stages"])


def default_equipment(stage_id: int) -> dict[str, int]:
    """Типичный парк этапа: предзаполнение плана и база плановых моточасов.

    Копия — вызывающий может править её, не портя кэш норм.
    """
    return dict(requirement(stage_id).default_fleet)


def all_pairs() -> list[dict]:
    """Пары как в JSON (для UI и документации «методика сопоставления»)."""
    return [dict(p) for p in _raw().get("pairs", [])]


def validate() -> list[str]:
    """Проверка канона на согласованность со словарём — пустой список, если всё хорошо.

    Ловит опечатки в ключах, противоречия (тип и ожидается, и запрещён), обязательный тип,
    который сам не входит в «свою» технику этапа, и пары без этапов.
    """
    errors: list[str] = []
    known = set(taxonomy.equipment())
    raw = _raw()
    for sid in stage_ids():
        req = requirement(sid)
        for fld in ("expected", "optional", "forbidden", "forbidden_info"):
            errors += [f"этап {sid}: {fld}: неизвестный тип {k}" for k in getattr(req, fld) if k not in known]
        for k in list(req.min_count) + list(req.default_fleet) + list(req.missing_window_h):
            if k not in known:
                errors.append(f"этап {sid}: неизвестный тип {k}")
        for k, subs in req.substitutes.items():
            errors += [f"этап {sid}: замена {k}→{x}: неизвестный тип" for x in subs if x not in known]
        clash = req.allowed & set(req.forbidden)
        if clash:
            errors.append(f"этап {sid}: и допустима, и запрещена: {sorted(clash)}")
        if not set(req.forbidden_info) <= set(req.forbidden):
            errors.append(f"этап {sid}: forbidden_info должно быть подмножеством forbidden")
        for k in req.min_count:
            if not set(req.accepted(k)) & req.allowed:
                errors.append(f"этап {sid}: обязательный {k} не допустим на этапе")
            if set(req.accepted(k)) & set(req.forbidden):
                errors.append(f"этап {sid}: обязательный {k} или его замена запрещены")
        for k in req.default_fleet:
            if k not in req.allowed:
                errors.append(f"этап {sid}: в типичном парке {k}, но он не допустим на этапе")
    for p in raw.get("pairs", []):
        if not p.get("stages"):
            errors.append(f"пара {p.get('id')}: не указаны этапы")
        for k in list(p.get("leader", [])) + list(p.get("followers", [])):
            if k not in known:
                errors.append(f"пара {p.get('id')}: неизвестный тип {k}")
        for sid, sev in p.get("stages", {}).items():
            if sev not in SEVERITIES:
                errors.append(f"пара {p.get('id')}: этап {sid}: уровень {sev}")
            req = requirement(int(sid))
            for k in list(p["leader"]) + list(p["followers"]):
                if k in req.forbidden:
                    errors.append(f"пара {p.get('id')}: {k} запрещён на этапе {sid}")
    return errors
