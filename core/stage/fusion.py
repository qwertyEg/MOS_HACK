"""Этап с учётом техники: довод модели А в хронологию этапов модели Б.

ТЗ §2 требует связать этап с техникой. Модель Б видит признаки этапа (котлован, опалубка,
асфальт), модель А — технику. Техника — самостоятельный довод об этапе: каток и
асфальтоукладчик работают на благоустройстве, копёр — на сваях, башенный кран с
бетононасосом — на монолите. Там, где чек-лист слаб (мелкая дальняя сцена, SigLIP «не
уверен»), этап должна подсказать техника; где чек-лист уверен, а техника спорит, —
сказать об этом в объяснении.

Модель довода — наивный Байес по нормам `reference/equipment_rules.json`:

- P(тип k задействован | фронт s): ожидаемая техника этапа (expected) 0.6, допустимая
  (optional) 0.25, не упомянутая 0.08, «не по этапу, но законна параллельно»
  (forbidden_info: экскаватор на каркасе — засыпка пазух, сети) 0.1, не по этапу 0.03;
- λ(s, k) = ln P(k|s) − ln среднее_s P(k|s). Тип, ожидаемый почти везде (автокран, грузовик),
  этапа не выдаёт (λ ≈ 0); тип одного этапа (копёр, асфальтоукладчик, каток) выдаёт сильно;
- сила довода за сутки a_k ∈ [0, 1]: работа по журналу моточасов 1 − e^(−ч/1 ч); виден, но
  активность не оценить (снимки раз в сутки — Эдинбург, Канберра) — 0.5; виден и стоит — 0.2
  (приехавшая под этап техника — слабый довод, припаркованная после этапа — не повод откатить);
- слагаемое эмиссии дня: W · clip(Σ_k a_k · λ(s, k), ±C). W = `equipment_weight` (0 — только
  чек-лист), C ограничивает один день, как term_clip у чек-листа.

Отсутствие техники не довод: камера видит не всю площадку, а детектор знает не все 21 тип
(асфальтоукладчика и гусеничного крана у YOLO нет — довод от них придёт только от GLM).
Техника различает группы этапов, а не всё: экскаватор с самосвалами ожидаемы и на
подготовке, и на котловане, и на благоустройстве — выбор внутри группы остаётся чек-листу
(котлован виден? здание выше земли?). Так и задумано: «экскаватор + самосвалы без здания →
котлован» — это техника плюс признак модели Б.

Замер на демо-объектах и почему такие числа — docs/methodology.md, раздел «Этап с учётом техники».
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping

from core import taxonomy
from core.contracts import ActivityInterval

# Короткие подписи признаков чек-листа — для объяснения «почему этап такой» (полный вопрос
# длинный, в одну фразу вердикта не помещается).
SIGN_SHORT = {
    "fence": "строительный забор", "gate_checkpoint": "въезд с КПП", "cabins": "бытовки",
    "temp_road": "временные дороги", "old_building": "здание под снос", "debris": "строительный мусор",
    "tree_felling": "вырубка деревьев", "vegetation_in_footprint": "растительность в пятне застройки",
    "cleared": "площадка расчищена", "utility_trench": "траншеи сетей", "pipes_stock": "складированные трубы",
    "rig": "буровая или сваебойная установка", "sheet_pile": "шпунтовое ограждение",
    "pile_heads": "оголовки свай", "pile_stock": "складированные сваи", "capping_beam": "обвязочная балка",
    "pit": "котлован", "soil_pile": "отвал грунта", "earthwork": "разработка грунта",
    "struts": "распорки котлована", "pit_bottom_prepared": "подготовленное дно котлована",
    "formwork": "опалубка", "rebar": "арматура", "concrete_pour": "бетонирование",
    "slab": "фундаментная плита", "below_grade": "конструкции ниже земли",
    "basement_walls": "стены подземной части", "zero_slab": "перекрытие на отметке 0",
    "waterproofing": "гидроизоляция подземной части", "backfilling": "засыпка пазух",
    "backfill": "пазухи засыпаны", "above_grade": "здание выше земли", "crane": "башенный кран у здания",
    "formwork_floor": "опалубка перекрытий", "unfinished_top": "незавершённый верхний этаж",
    "steel_frame": "металлокаркас", "masonry": "кладка стен", "bare_concrete": "голый бетонный каркас",
    "roof_work": "работы на кровле", "roof_superstructures": "надстройки на крыше",
    "roof_cover": "кровельное покрытие", "parapet": "парапеты", "scaffold": "фасадные леса",
    "window_openings_empty": "пустые оконные проёмы", "glazing": "остекление", "insulation": "утеплитель фасада",
    "facade_subframe": "подсистема фасада", "cladding": "облицовка фасада", "entrance_finished": "оформленные входы",
    "grading": "планировка территории", "curbs": "бордюры", "asphalt_work": "укладка асфальта",
    "paving": "покрытие проездов", "landscaping": "озеленение", "amenities": "детские площадки и МАФ",
    "lighting_poles": "опоры освещения", "permanent_fence": "постоянное ограждение",
    "bare_ground": "открытый грунт", "is_construction": "идёт стройка", "workers_visible": "рабочие",
}

WORKING, IDLE, UNKNOWN = "working", "idle", "unknown"


@dataclass
class FusionConfig:
    equipment_weight: float = 1.0      # W: вес довода техники против чек-листа; 0 — только чек-лист
    work_scale_h: float = 1.0          # час работы за сутки → a = 0.63, три часа → 0.95
    presence_unjudged: float = 0.5     # тип виден, работу оценить нельзя (снимки раз в сутки)
    presence_idle: float = 0.2         # тип виден и стоит на плотной съёмке
    min_conf: float = 0.5              # присутствие — только уверенные рамки (ложные «самосвалы»-бытовки 0.3–0.6)
    term_clip: float = 3.0             # C: вклад одного дня не больше, чем у чек-листа
    p_expected: float = 0.6
    p_optional: float = 0.25
    p_unlisted: float = 0.08
    p_forbidden_info: float = 0.1
    p_forbidden: float = 0.03
    parallel_decay: float = 0.5        # техника предыдущего этапа при новом фронте (доделывают параллельно)
    basis_days: int = 7                # окно объяснения «почему этап такой» — последние дни на текущем этапе

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "FusionConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Sighting:
    """Тип техники на кадре (рамка модели А после трекинга): вход для «присутствия»."""
    frame_id: int | str
    captured_at: dt.datetime
    cls: str
    conf: float
    activity: str = UNKNOWN            # working | idle | unknown (как Detection.activity)


@dataclass
class EquipmentEvidence:
    """Что модель А знает о технике площадки: журнал моточасов и рамки на кадрах."""
    intervals: list[ActivityInterval] = field(default_factory=list)
    sightings: list[Sighting] = field(default_factory=list)


@dataclass
class EquipmentDay:
    """Техника площадки за сутки (по местному времени) — довод модели А об этапе."""
    working_h: dict[str, float] = field(default_factory=dict)   # тип → моточасы, начатые в эти сутки
    seen: dict[str, int] = field(default_factory=dict)          # тип → кадров, где тип виден уверенно
    judged: dict[str, int] = field(default_factory=dict)        # тип → из них кадров с оценённой активностью
    frames: dict[str, list] = field(default_factory=dict)       # тип → кадры-доказательства (работа первой)


# --------------------------------------------------------------------------
# нормы → правдоподобия
# --------------------------------------------------------------------------


def _category(stage_id: int, cls: str) -> str:
    from core.plan import norms       # план — чистое ядро без тяжёлых зависимостей

    req = norms.requirement(stage_id)
    if cls in req.expected:
        return "expected"
    if cls in req.optional:
        return "optional"
    if cls in req.forbidden_info:
        return "forbidden_info"
    if cls in req.forbidden:
        return "forbidden"
    return "unlisted"


@lru_cache(maxsize=8)
def _table(p: tuple[float, float, float, float, float], decay: float) -> dict[str, dict[int, float]]:
    probs = dict(zip(("expected", "optional", "unlisted", "forbidden_info", "forbidden"), p))
    order = sorted(taxonomy.stages())
    out: dict[str, dict[int, float]] = {}
    for cls in taxonomy.equipment():
        own = {s: probs[_category(s, cls)] for s in order}
        # Фронт — старший НАЧАТЫЙ этап: предыдущий этап часто ещё доделывается параллельно
        # (засыпка пазух при каркасе, уплотнение основания при монтаже колонн), поэтому его
        # техника при новом фронте правдоподобна с множителем decay.
        pk = {s: max(own[s], decay * own[s - 1]) if s - 1 in own else own[s] for s in order}
        mean = sum(pk.values()) / len(pk)
        out[cls] = {s: math.log(v / mean) for s, v in pk.items()}
    return out


def likelihoods(cfg: FusionConfig | None = None) -> dict[str, dict[int, float]]:
    """λ[тип][этап] — log-отношение «этот тип при этом фронте» к среднему по этапам."""
    cfg = cfg or FusionConfig()
    return _table((cfg.p_expected, cfg.p_optional, cfg.p_unlisted, cfg.p_forbidden_info, cfg.p_forbidden),
                  float(cfg.parallel_decay))


def category(stage_id: int, cls: str) -> str:
    """expected | optional | unlisted | forbidden_info | forbidden — место типа в нормах этапа."""
    return _category(int(stage_id), cls)


# --------------------------------------------------------------------------
# сутки
# --------------------------------------------------------------------------


def _local_day(when: dt.datetime, tz_offset_hours: float) -> dt.date:
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return (when.astimezone(dt.timezone.utc) + dt.timedelta(hours=tz_offset_hours)).date()


def days(evidence: EquipmentEvidence | None, tz_offset_hours: float = 3.0,
         cfg: FusionConfig | None = None) -> dict[dt.date, EquipmentDay]:
    """Журнал моточасов и рамки → техника по суткам площадки (границы суток — как у модели Б)."""
    cfg = cfg or FusionConfig()
    out: dict[dt.date, EquipmentDay] = defaultdict(EquipmentDay)
    if evidence is None:
        return {}
    known = taxonomy.equipment()
    for iv in evidence.intervals or []:
        if iv.cls not in known or not iv.hours or iv.hours <= 0:
            continue                    # ручная поправка «−3 ч» — не довод о составе техники
        day = out[_local_day(iv.start, tz_offset_hours)]
        day.working_h[iv.cls] = day.working_h.get(iv.cls, 0.0) + float(iv.hours)
        day.frames.setdefault(iv.cls, []).extend(f for f in (iv.frame_ids or [])[:3]
                                                 if f not in day.frames.get(iv.cls, []))
    counted: set[tuple[dt.date, str, Any]] = set()
    for sg in evidence.sightings or []:
        if sg.cls not in known or sg.conf < cfg.min_conf:
            continue
        d = _local_day(sg.captured_at, tz_offset_hours)
        key = (d, sg.cls, sg.frame_id)
        if key in counted:              # две рамки одного типа на кадре — один кадр присутствия
            continue
        counted.add(key)
        day = out[d]
        day.seen[sg.cls] = day.seen.get(sg.cls, 0) + 1
        if sg.activity in (WORKING, IDLE):
            day.judged[sg.cls] = day.judged.get(sg.cls, 0) + 1
        lst = day.frames.setdefault(sg.cls, [])
        if len(lst) < 6 and sg.frame_id not in lst:
            lst.append(sg.frame_id)
    return dict(out)


def strengths(day: EquipmentDay, cfg: FusionConfig | None = None) -> dict[str, float]:
    """Тип → сила довода a_k за сутки (0..1)."""
    cfg = cfg or FusionConfig()
    out: dict[str, float] = {}
    for cls in set(day.working_h) | set(day.seen):
        h = day.working_h.get(cls, 0.0)
        a = 1.0 - math.exp(-h / max(cfg.work_scale_h, 1e-6)) if h > 0 else 0.0
        if day.seen.get(cls, 0) > 0:
            a = max(a, cfg.presence_unjudged if not day.judged.get(cls) else cfg.presence_idle)
        if a > 0:
            out[cls] = a
    return out


def stage_scores(day: EquipmentDay | None, order: Iterable[int], cfg: FusionConfig | None = None) -> dict[int, float]:
    """Этап → Σ_k a_k·λ(этап, k) без веса и ограничения: сколько техника «за» каждый фронт."""
    cfg = cfg or FusionConfig()
    order = list(order)
    if day is None:
        return {s: 0.0 for s in order}
    lam = likelihoods(cfg)
    a = strengths(day, cfg)
    return {s: sum(v * lam[k][s] for k, v in a.items()) for s in order}


def emission(day: EquipmentDay | None, order: Iterable[int], cfg: FusionConfig | None = None) -> list[float]:
    """Слагаемое log-эмиссии дня для каждого фронта из `order` (0 — техника ничего не говорит)."""
    cfg = cfg or FusionConfig()
    sc = stage_scores(day, order, cfg)
    w, clip = float(cfg.equipment_weight), float(cfg.term_clip)
    return [w * max(-clip, min(clip, sc[s])) for s in order]


# --------------------------------------------------------------------------
# объяснение
# --------------------------------------------------------------------------


def eq_short(cls: str) -> str:
    """«Грузовик (бортовой, длинномер, трал)» → «грузовик»: для фразы объяснения."""
    name = taxonomy.equipment_name(cls)
    return name.split(" (")[0].split(" / ")[0].lower()


def _hours(h: float) -> str:
    return f"{h:.1f}".replace(".", ",") + " ч"


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " и " + items[-1]


def summarize(window: list[EquipmentDay], cfg: FusionConfig | None = None) -> list[dict]:
    """Техника окна по типам: часы работы, дни присутствия, сила довода (для объяснения и UI)."""
    cfg = cfg or FusionConfig()
    agg: dict[str, dict] = {}
    for day in window:
        a = strengths(day, cfg)
        for cls, v in a.items():
            r = agg.setdefault(cls, {"cls": cls, "name": eq_short(cls), "hours": 0.0, "days": 0,
                                     "strength": 0.0, "frames": []})
            r["hours"] += day.working_h.get(cls, 0.0)
            r["days"] += 1
            r["strength"] += v
            for f in day.frames.get(cls, []):
                if len(r["frames"]) < 3 and f not in r["frames"]:
                    r["frames"].append(f)
    rows = sorted(agg.values(), key=lambda r: (-r["hours"], -r["strength"], r["cls"]))
    for r in rows:
        r["hours"] = round(r["hours"], 1)
        r["strength"] = round(r["strength"], 2)
    return rows


def relation(stage: int, avg_scores: Mapping[int, float], rows: list[dict]) -> tuple[str, list[int]]:
    """Как техника окна соотносится с этапом → (agree | contra | neutral, этапы техники).

    avg_scores — stage_scores(), усреднённые по суткам окна; rows — summarize() окна. «Этапы
    техники» — лучшие с запасом 0.5 (экскаватор с самосвалами — ничья 1/3/8). «Подтверждает» —
    этап среди лучших и на нём работает или стоит его ожидаемая техника; «спорит» — этап хуже
    лучшего на 1.0 и больше: слабый перекос от пары ложных рамок (бытовка-«самосвал») спором не считаем.
    """
    if not avg_scores:
        return "neutral", []
    best = max(avg_scores.values())
    top = sorted(s for s, v in avg_scores.items() if best > 0.3 and v >= best - 0.5)
    fit = any(category(stage, r["cls"]) == "expected" for r in rows)
    if stage in top and fit:
        return "agree", top
    if top and stage not in top and best - avg_scores.get(stage, 0.0) >= 1.0:
        return "contra", top
    return "neutral", top


def describe(stage: int, rows: list[dict], rel: str, eq_stages: list[int], checklist_stage: int | None,
             signs: list[str]) -> str:
    """Фраза «почему этап такой»: что видит чек-лист, что делает техника и как это сходится.

    stage — итоговый фронт; rows — summarize() окна; rel, eq_stages — relation(); checklist_stage —
    фронт по одному чек-листу (None — чек-лист этап не выдал); signs — видимые признаки этапа.
    """
    name = f"«{taxonomy.stage_name(stage)}»"
    working = [r for r in rows if r["hours"] > 0]
    seen_only = [r for r in rows if r["hours"] <= 0]

    def fmt_eq(rs: list[dict], limit: int = 3) -> str:
        return _join([f"{r['name']} ({_hours(r['hours'])})" if r["hours"] > 0 else r["name"] for r in rs[:limit]])

    fit = [r for r in rows if category(stage, r["cls"]) == "expected"]
    eq_phrase = (f"работают {fmt_eq(working)}" if working else f"на снимках {fmt_eq(seen_only)}") if rows else ""
    cl = _join([SIGN_SHORT.get(k, k) for k in signs[:3]])

    if not rows:
        return (f"Этап {name} — по чек-листу модели Б ({cl or 'признаки этапа на снимках'}); "
                "техники, по которой можно судить об этапе, камеры не видели.")
    if rel == "agree" and checklist_stage != stage:
        how = ("Чек-лист модели Б этап сам не выдал" if checklist_stage is None
               else f"По одному чек-листу модели Б вышел бы этап «{taxonomy.stage_name(checklist_stage)}»")
        return (f"Этап {name} определён с учётом техники: {eq_phrase}, а по нормам «этап → техника» на этом "
                f"этапе ожидаемы {_join([r['name'] for r in fit[:3]])}. {how}"
                + (f"; признаки этапа на снимках: {cl}." if cl else "."))
    if rel == "agree":
        also = [s for s in eq_stages if s != stage]
        tail = (f"; такая техника бывает и на этап{'е' if len(also) == 1 else 'ах'} "
                f"{_join([str(s) for s in also])} — {'его' if len(also) == 1 else 'их'} отличает чек-лист"
                if also else "")
        return f"Этап {name}: на снимках {cl or 'признаки этапа'}, и техника это подтверждает — {eq_phrase}{tail}."
    if rel == "contra":
        other = _join([f"«{taxonomy.stage_name(s)}»" for s in eq_stages[:2]])
        return (f"Этап {name} — по чек-листу модели Б ({cl or 'признаки этапа'}); техника спорит: "
                f"{eq_phrase} — это скорее техника этапа {other}. Проверьте кадры или отметьте этап вручную.")
    tail = f"; ожидаемая техника этапа среди них: {fmt_eq(fit)}" if fit else ""
    return (f"Этап {name} — по чек-листу модели Б ({cl or 'признаки этапа'}); {eq_phrase}{tail} — "
            "по технике этап не различить, она не противоречит.")
