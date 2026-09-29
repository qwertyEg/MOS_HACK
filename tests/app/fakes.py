"""Фейковые модули ядра для тестов бэкенда.

Реальные core.equipment / core.stage / core.plan / core.analytics пишут другие
агенты параллельно; бэкенд обязан работать с любыми реализациями контракта.
Фейки реализуют ровно интерфейс из ARCHITECTURE/задания и простую, но
осмысленную логику: детектор находит цветные прямоугольники (синий —
экскаватор, красный — самосвал), движок сопоставляет рамки с прошлым кадром
камеры и засчитывает интервалы работы, правило ловит «экскаватор без
самосвалов», хронология ставит этап 3.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np

from core import contracts as c

# BGR-диапазоны «техники» на синтетических кадрах
COLORS = {
    "excavator": ((180, 0, 0), (255, 80, 80)),      # синий
    "dump_truck": ((0, 0, 180), (80, 80, 255)),     # красный
}


# --------------------------------------------------------------------------
# core.equipment
# --------------------------------------------------------------------------

class FakeDetector:
    provider = c.Provider.LOCAL

    def __init__(self, name: str) -> None:
        self.name = name
        self.provider = c.Provider.LOCAL if name == "yolo" else c.Provider.EXTERNAL
        self.is_ready = True
        self.reason = ""
        self.fail_if: Callable[[np.ndarray, c.FrameInfo], bool] | None = None
        self.calls = 0

    def ready(self) -> tuple[bool, str]:
        return self.is_ready, self.reason

    def detect(self, image: np.ndarray, frame: c.FrameInfo) -> list[c.Detection]:
        self.calls += 1
        if self.fail_if is not None and self.fail_if(image, frame):
            raise RuntimeError("детектор упал на этом кадре")
        out = []
        for cls, (lo, hi) in COLORS.items():
            mask = cv2.inRange(image, np.array(lo, np.uint8), np.array(hi, np.uint8))
            n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                x, y, w, h, area = (int(v) for v in stats[i])
                if area >= 60:
                    out.append(c.Detection(cls=cls, conf=0.91, bbox=(float(x), float(y), float(w), float(h)),
                                           source=self.name))
        return out


@dataclasses.dataclass
class EquipmentConfig:
    move_px: float = 5.0
    max_gap_min: float = 45.0
    parked_after_h: float = 48.0
    departed_after_h: float = 3.0
    merge_radius_m: float = 5.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "EquipmentConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class EquipmentUpdate:
    detections: list
    units: list
    intervals: list


class FakeEngine:
    def __init__(self, config: EquipmentConfig | None = None) -> None:
        self.config = config or EquipmentConfig()
        self._units: dict[str, c.UnitState] = {}
        self._last: dict[Any, tuple[dt.datetime, list[c.Detection]]] = {}
        self.restored: tuple | None = None
        self.locked: dict[str, str] = {}

    def set_manual_classes(self, classes) -> None:
        self.locked.update({k: v for k, v in (classes or {}).items() if v})

    def restore(self, units, last) -> None:
        self.restored = (list(units), dict(last))
        self._units = {u.unit_id: u for u in units}
        self._last = dict(last)

    def _stage_for(self, plan, when: dt.datetime) -> int | None:
        for p in plan:
            if p.planned_start and p.planned_end and p.planned_start <= when.date() <= p.planned_end:
                return p.stage_id
        return None

    def process(self, frame, image, detections, geometry, zones, plan) -> EquipmentUpdate:
        prev_t, prev = self._last.get(frame.camera_id, (None, []))
        changed: dict[str, c.UnitState] = {}
        intervals = []
        used: set[int] = set()
        for d in detections:
            key = d.extra.get("manual_unit")          # машина, названная оператором (как в core.equipment)
            best, best_dist = None, math.inf
            for j, p in enumerate(prev):
                if j in used or (p.unit_id != key if key else p.cls != d.cls):
                    continue
                dist = math.dist(p.center, d.center)
                if dist < best_dist:
                    best, best_dist = j, dist
            if key and best is None:
                d.unit_id, d.track_id = key, f"{frame.camera_id}:{key}"
                d.activity = c.Activity.UNKNOWN
            elif best is not None and (best_dist < 150 or key):
                used.add(best)
                d.track_id, d.unit_id = prev[best].track_id, prev[best].unit_id
                d.displacement_px = best_dist
                d.moved_since_prev = best_dist > self.config.move_px
                d.activity = c.Activity.WORKING if d.moved_since_prev else c.Activity.IDLE
            else:
                n = sum(1 for u in self._units if u.startswith(d.cls)) + 1
                d.unit_id = f"{d.cls}-{n}"
                d.track_id = f"{frame.camera_id}:{d.unit_id}"
                d.activity = c.Activity.UNKNOWN
            for z in zones:
                poly = np.array(z.polygon, np.float32)
                if len(poly) >= 3 and cv2.pointPolygonTest(poly, d.foot, False) >= 0:
                    d.zone_id = z.id
            if key:
                d.unit_id = key
            u = self._units.get(d.unit_id) or c.UnitState(
                unit_id=d.unit_id, cls=d.extra.get("manual_unit_cls") or d.cls, status=c.UnitStatus.IDLE,
                first_seen=frame.captured_at, last_seen=frame.captured_at, last_moved=None,
                label=f"{d.extra.get('manual_unit_cls') or d.cls} #{d.unit_id.rsplit('-', 1)[-1]}")
            if self.locked.get(u.unit_id):
                u.cls = self.locked[u.unit_id]
            elif key and d.extra.get("manual_unit_cls"):
                u.cls = d.extra["manual_unit_cls"]
            u.last_seen = frame.captured_at
            u.cameras = set(u.cameras) | {str(frame.camera_id)}
            if d.moved_since_prev and prev_t is not None:
                hours = min((frame.captured_at - prev_t).total_seconds() / 3600, self.config.max_gap_min / 60)
                intervals.append(c.ActivityInterval(unit_id=u.unit_id, cls=u.cls, start=prev_t,
                                                    end=frame.captured_at, hours=hours,
                                                    stage_id=self._stage_for(plan, frame.captured_at),
                                                    frame_ids=[frame.frame_id]))
                u.worked_hours += hours
                u.last_moved = frame.captured_at
                u.status = c.UnitStatus.ACTIVE
            elif d.activity == c.Activity.IDLE:
                u.status = c.UnitStatus.IDLE
            self._units[u.unit_id] = u
            changed[u.unit_id] = u
        self._last[frame.camera_id] = (frame.captured_at, list(detections))
        return EquipmentUpdate(detections=list(detections), units=list(changed.values()), intervals=intervals)

    def units(self) -> list[c.UnitState]:
        return list(self._units.values())


def planned_hours(plan, fleet, shift_hours=10.0, utilization=0.7, workdays=(0, 1, 2, 3, 4, 5)):
    out: dict[int, dict[str, float]] = {}
    for p in plan:
        if not p.planned_start or not p.planned_end:
            continue
        days = sum(1 for i in range((p.planned_end - p.planned_start).days + 1)
                   if (p.planned_start + dt.timedelta(days=i)).weekday() in workdays)
        counts = dict(p.equipment) or dict(fleet)
        out[p.stage_id] = {cls: n * days * shift_hours * utilization for cls, n in counts.items() if n}
    return out


def balances(plan, intervals):
    out = []
    for p in plan:
        for cls, planned in p.planned_hours.items():
            ivs = [i for i in intervals if i.stage_id == p.stage_id and i.cls == cls]
            out.append(c.HoursBalance(stage_id=p.stage_id, cls=cls, planned_hours=planned,
                                      worked_hours=sum(i.hours for i in ivs),
                                      last_worked_at=max((i.end for i in ivs), default=None)))
    return out


def homography_from_points(image_pts, site_pts):
    src, dst = np.array(image_pts, np.float64), np.array(site_pts, np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise ValueError("точки вырождены")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H.tolist(), float(np.sqrt(((proj - dst) ** 2).sum(1)).mean())


def annotate(image, detections, labels_ru=True):
    out = image.copy()
    for d in detections:
        x, y, w, h = (int(v) for v in d.bbox)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


# --------------------------------------------------------------------------
# core.stage
# --------------------------------------------------------------------------

def assess(image, captured_at=None):
    mean = float(image.mean())
    night = mean < 40
    return c.QualityReport(quality_ok=True, is_night=night, weather=c.Weather.CLEAR, reject_reason="",
                           blur=150.0, brightness=mean, usable_for_stage=not night)


class DynamicMask:
    def __init__(self, shape) -> None:
        self.shape = tuple(int(v) for v in shape)
        self.n = 0
        self.ratio = 0.0

    @classmethod
    def new(cls, shape_hw):
        return cls(shape_hw)

    def update(self, image_bgr, captured_at) -> None:
        if tuple(image_bgr.shape[:2]) != self.shape:
            raise ValueError("размер кадра сменился")
        self.n += 1
        self.ratio = min(0.3, 0.1 * self.n)

    def apply(self, image_bgr, mode="darken"):
        return image_bgr

    def visible(self):
        v = np.ones(self.shape, bool)
        v[: int(self.shape[0] * self.ratio)] = False
        return v

    def dumps(self) -> bytes:
        return json.dumps({"shape": self.shape, "n": self.n, "ratio": self.ratio}).encode()

    @classmethod
    def loads(cls, data: bytes):
        d = json.loads(data)
        m = cls(d["shape"])
        m.n, m.ratio = d["n"], d["ratio"]
        return m

    @property
    def masked_ratio(self) -> float:
        return self.ratio


class FakeClassifier:
    def __init__(self, name: str) -> None:
        self.name = name
        self.provider = c.Provider.EXTERNAL if name == "glm" else c.Provider.LOCAL
        self.is_ready = True
        self.reason = ""
        self.calls: list = []
        self.answers = {"pit": c.Answer.YES, "earthwork": c.Answer.YES, "soil_pile": c.Answer.UNSURE,
                        "slab": c.Answer.NO}

    def ready(self):
        return self.is_ready, self.reason

    def assess(self, image_bgr, frame, keys=None, context=None):
        self.calls.append((frame.frame_id, context))
        return c.ChecklistResult(answers=dict(self.answers), scores={"pit": 0.81}, stage_likelihood={3: 0.7},
                                 model=f"fake-{self.name}", provider=self.provider, latency_ms=5.0)


@dataclasses.dataclass
class StageScores:
    stage_evidence: dict
    substages: dict
    front: int | None
    progress: dict


def evaluate(answers):
    front = 3 if answers.get("pit") == c.Answer.YES else None
    return StageScores(stage_evidence={3: 0.8}, substages={"3.1": "active"}, front=front, progress={3: 0.4})


def infer(observations, manual=None, config=None):
    states: dict[int, c.StageState] = {}
    daily = []
    if observations:
        first = min(o.captured_at for o in observations).date()
        states = {1: c.StageState(1, c.StageStatus.DONE, 1.0, actual_end=first),
                  2: c.StageState(2, c.StageStatus.DONE, 1.0, actual_end=first),
                  3: c.StageState(3, c.StageStatus.ACTIVE, 0.4, actual_start=first,
                                  evidence_frame_ids=[o.frame_id for o in observations][-3:])}
        daily = [(first, 3)]
    for k, v in (manual or {}).items():
        states[k] = v
    current = max((k for k, v in states.items() if v.status == c.StageStatus.ACTIVE), default=None)
    limit = (config or {}).get("unsure_review_ratio", 0.5)
    return c.StageTimeline(states=states, current_stage=current, overall_progress=0.2 if observations else 0.0,
                           daily_front=daily,
                           needs_review=[o.frame_id for o in observations if o.result.unsure_ratio > limit])


# --------------------------------------------------------------------------
# core.plan
# --------------------------------------------------------------------------

@dataclasses.dataclass
class WorkItem:
    code: str
    name: str
    level: int
    status: str
    stage_id: int | None
    substage_id: str | None
    object_types: list


WORKS = [
    WorkItem("10.2.", "Вынос инженерных систем", 1, "substage", 1, "1.3", ["Жильё"]),
    WorkItem("12.3.1.", "Устройство котлована", 2, "substage", 3, "3.1", ["Жильё"]),
    WorkItem("12.3.9.", "Водопонижение", 2, "unobservable", 3, None, ["Жильё"]),
]


def works_for_stage(stage_id, substage_id=None):
    return [w for w in WORKS if w.stage_id == stage_id and (substage_id is None or w.substage_id == substage_id)]


def parse(data: bytes, filename: str):
    if not filename.lower().endswith(".csv"):
        raise ValueError("фейк понимает только CSV")
    items, warnings = [], []
    for n, line in enumerate(data.decode("utf-8").splitlines(), 1):
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if n == 1 and not parts[0].isdigit():
            continue
        try:
            items.append(c.PlanItem(stage_id=int(parts[0]), planned_start=dt.date.fromisoformat(parts[1]),
                                    planned_end=dt.date.fromisoformat(parts[2]), work_codes=parts[3:]))
        except (ValueError, IndexError):
            warnings.append(f"строка {n} не разобрана")
    return items, warnings


def demo_plan(start, end=None, stage_ids=None):
    ids = list(stage_ids or range(1, 9))
    end = end or start + dt.timedelta(days=30 * len(ids))
    span = max(1, (end - start).days + 1)
    out = []
    for i, sid in enumerate(ids):
        a = start + dt.timedelta(days=span * i // len(ids))
        b = start + dt.timedelta(days=span * (i + 1) // len(ids) - 1)
        out.append(c.PlanItem(stage_id=sid, planned_start=a, planned_end=max(a, b), name=f"Этап {sid}"))
    return out


@dataclasses.dataclass
class StageRequirement:
    expected: list
    optional: list
    forbidden: list
    pairs: list
    min_count: dict


def requirement(stage_id):
    return StageRequirement(["excavator"], [], ["tower_crane"], [("excavator", "dump_truck")], {"excavator": 1})


def default_equipment(stage_id):
    return {3: {"excavator": 1, "dump_truck": 2}}.get(stage_id, {"excavator": 1})


# --------------------------------------------------------------------------
# core.analytics
# --------------------------------------------------------------------------

@dataclasses.dataclass
class AnalyticsContext:
    site_id: Any
    now: dt.datetime
    plan: list
    timeline: Any
    units: list
    recent: list
    intervals: list
    balances: list
    zones: list
    config: dict


def rules_evaluate(ctx):
    exc = [(fi, d) for fi, dets in ctx.recent for d in dets if d.cls == "excavator"]
    trucks = any(d.cls == "dump_truck" for _fi, dets in ctx.recent for d in dets)
    if not exc or trucks:
        return []
    return [c.DeviationRecord(
        key="pair_broken:excavator:dump_truck", type=c.DeviationType.PAIR_BROKEN, severity=c.Severity.WARNING,
        title="Экскаватор работает без самосвалов",
        message="Возможное снижение темпа: экскаватор есть, самосвалов в окне нет",
        stage_id=3, camera_id=exc[-1][0].camera_id, frame_ids=[fi.frame_id for fi, _ in exc][-3:],
        unit_ids=sorted({d.unit_id for _, d in exc if d.unit_id}),
        started_at=exc[0][0].captured_at, last_seen_at=exc[-1][0].captured_at)]


@dataclasses.dataclass
class PlanFact:
    expected_progress: float | None
    actual_progress: float
    lag_days: float | None
    verdict: c.Verdict
    forecast_finish: dt.date | None
    series: dict


def plan_vs_fact(plan, stage_timeline, today):
    if not plan:
        return PlanFact(None, stage_timeline.overall_progress, None, c.Verdict.NO_PLAN, None,
                        {"days": [], "expected": [], "actual": []})
    return PlanFact(0.5, stage_timeline.overall_progress, 4.0, c.Verdict.BEHIND, today + dt.timedelta(days=30),
                    {"days": [today.isoformat()], "expected": [0.5], "actual": [stage_timeline.overall_progress]})


def report_build(ctx):
    pf = plan_vs_fact(ctx.plan, ctx.timeline, ctx.now.date())
    return c.SiteReport(verdict=pf.verdict, lag_days=pf.lag_days, expected_progress=pf.expected_progress,
                        actual_progress=pf.actual_progress, forecast_finish=pf.forecast_finish,
                        current_stage=ctx.timeline.current_stage, stage_states=ctx.timeline.states,
                        hours=ctx.balances, deviations=[], explanation=["фейковый отчёт: отставание 4 дня"])


# --------------------------------------------------------------------------
# сборка
# --------------------------------------------------------------------------

class Fakes:
    """Набор фейковых модулей + доступ к экземплярам провайдеров для управления в тестах."""

    def __init__(self) -> None:
        self.detectors: dict[str, FakeDetector] = {}
        self.classifiers: dict[str, FakeClassifier] = {}
        self.engines: list[FakeEngine] = []

        def get_detector(name, **kw):
            if name not in ("yolo", "glm"):
                raise ValueError(name)
            return self.detectors.setdefault(name, FakeDetector(name))

        def get_classifier(name, **kw):
            if name not in ("siglip", "glm", "local_vlm"):
                raise ValueError(name)
            return self.classifiers.setdefault(name, FakeClassifier(name))

        def make_engine(config=None):
            e = FakeEngine(config)
            self.engines.append(e)
            return e

        self.equipment = SimpleNamespace(
            get_detector=get_detector, EquipmentConfig=EquipmentConfig, EquipmentEngine=make_engine,
            EquipmentUpdate=EquipmentUpdate,
            hours=SimpleNamespace(planned_hours=planned_hours, balances=balances),
            fusion=SimpleNamespace(homography_from_points=homography_from_points),
            draw=SimpleNamespace(annotate=annotate),
        )
        self.stage = SimpleNamespace(
            quality=SimpleNamespace(assess=assess), DynamicMask=DynamicMask, get_classifier=get_classifier,
            scoring=SimpleNamespace(evaluate=evaluate), sequence=SimpleNamespace(infer=infer),
        )
        self.plan = SimpleNamespace(
            catalog=SimpleNamespace(load=lambda: list(WORKS), works_for_stage=works_for_stage),
            importer=SimpleNamespace(parse=parse, demo_plan=demo_plan),
            norms=SimpleNamespace(requirement=requirement, default_equipment=default_equipment),
        )
        self.analytics = SimpleNamespace(
            context=SimpleNamespace(AnalyticsContext=AnalyticsContext),
            rules=SimpleNamespace(evaluate=rules_evaluate),
            timeline=SimpleNamespace(plan_vs_fact=plan_vs_fact),
            report=SimpleNamespace(build=report_build),
        )

    def install(self, providers) -> None:
        providers.override_module("core.equipment", self.equipment)
        providers.override_module("core.stage", self.stage)
        providers.override_module("core.plan", self.plan)
        providers.override_module("core.analytics", self.analytics)
