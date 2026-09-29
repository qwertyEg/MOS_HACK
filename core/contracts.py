"""Контракты между модулями СтройВзора.

Этот файл — единственная точка, через которую модули договариваются друг
с другом: модель А (техника), модель Б (этап), план, аналитика и веб-слой.
Здесь только типы данных и протоколы, никакой логики и никаких тяжёлых
зависимостей (torch, ultralytics, сеть) — его импортирует всё, в том числе
тесты на машине без GPU.

Правила:
- ядро (`core/*`) не знает про БД и HTTP: веб-слой переводит строки БД в эти
  датаклассы и обратно;
- все времена — timezone-aware `datetime` (UTC внутри, Москва в UI);
- рамки — в пикселях исходного кадра, формат XYWH (левый верх, ширина, высота),
  как в контракте модели А из PLAN.md §4.3;
- ключи техники, этапов и признаков — из `reference/checklist.json`
  (см. `core/taxonomy.py`), других названий в коде быть не должно.

Менять этот файл можно только согласованно: он общий для всех веток команды.
"""
from __future__ import annotations

import datetime as dt
import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------
# общее
# --------------------------------------------------------------------------


class Provider(str, enum.Enum):
    """Кто выполняет модель. Переключается в UI (см. docs/ARCHITECTURE.md §10)."""
    LOCAL = "local"          # свои модели: YOLO, SigLIP, локальная VLM
    EXTERNAL = "external"    # внешний API: GLM-4.6V (z.ai)


class Answer(str, enum.Enum):
    """Тернарный ответ чек-листа. UNSURE не голосует ни за, ни против этапа."""
    YES = "yes"
    NO = "no"
    UNSURE = "unsure"


class Weather(str, enum.Enum):
    CLEAR = "clear"
    RAIN = "rain"      # капли на объективе / дождь — кадр не идёт в модель Б
    SNOW = "snow"
    FOG = "fog"
    UNKNOWN = "unknown"


@dataclass
class FrameInfo:
    """Что известно о кадре до моделей. Заполняет веб-слой + core/stage/quality."""
    frame_id: int | str
    camera_id: int | str
    site_id: int | str
    captured_at: dt.datetime
    width: int
    height: int
    is_night: bool = False
    weather: Weather = Weather.UNKNOWN
    quality_ok: bool = True
    reject_reason: str = ""
    blur: float | None = None          # дисперсия лапласиана; меньше — мутнее
    brightness: float | None = None    # средняя яркость 0..255
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Результат core/stage/quality.assess(image)."""
    quality_ok: bool
    is_night: bool
    weather: Weather
    reject_reason: str = ""
    blur: float = 0.0
    brightness: float = 0.0
    usable_for_stage: bool = True      # False для ночи, дождя, брака — модель Б пропускает
    # Помехи кадра кодами (core/stage/quality.FLAGS): night, twilight, drops, fog, glare, snow_cover,
    # snowfall, occluded, shifted, low_visibility, blur, dark, overexposed, low_contrast, ir.
    # Часть — только пометка (снег на площадке, сумерки), часть исключает кадр из модели Б.
    flags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)   # сырые числа оценки — журнал кадра и UI


# --------------------------------------------------------------------------
# модель А — техника
# --------------------------------------------------------------------------


class Activity(str, enum.Enum):
    WORKING = "working"    # сместилась / изменила позу с прошлого кадра
    IDLE = "idle"          # на месте
    UNKNOWN = "unknown"    # первый кадр трека — сравнивать не с чем


class UnitStatus(str, enum.Enum):
    """Статусная машина единицы техники (PLAN.md §3.9)."""
    ACTIVE = "active"        # работала в последнем интервале
    IDLE = "idle"            # стоит меньше порога парковки
    PARKED = "parked"        # стоит дольше порога (по умолчанию 48 ч) — ждёт вывоза
    DEPARTED = "departed"    # пропала из кадра дольше порога


@dataclass
class Detection:
    """Одна рамка техники на кадре.

    Детектор заполняет cls/conf/bbox/source. Остальное дописывают трекер
    (track_id, moved_since_prev, displacement_px, bbox_shape_delta, activity),
    зоны (zone_id), слияние камер (unit_id, site_xy).
    Сериализация наружу (POST /api/detect) — `to_contract()`.
    """
    cls: str                                   # ключ из taxonomy.EQUIPMENT
    conf: float
    bbox: tuple[float, float, float, float]    # x, y, w, h в пикселях кадра
    source: str = "local"                      # имя детектора: "yolo", "glm-4.6v", ...
    track_id: str | None = None                # трек внутри одной камеры
    unit_id: str | None = None                 # единица техники на площадке (после слияния камер)
    moved_since_prev: bool = False
    displacement_px: float = 0.0
    bbox_shape_delta: float = 0.0              # |Δw/w| + |Δh/h| — работа стрелой при неподвижном центре
    appearance_delta: float = 0.0              # изменение содержимого рамки (поза ковша, стрелы)
    activity: Activity = Activity.UNKNOWN
    zone_id: int | None = None
    site_xy: tuple[float, float] | None = None # точка контакта с землёй на плане площадки, метры
    extra: dict[str, Any] = field(default_factory=dict)  # номер, эмбеддинг внешности, сырой ответ VLM

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2

    @property
    def foot(self) -> tuple[float, float]:
        """Середина нижней кромки рамки — точка, которую проецируем на план."""
        x, y, w, h = self.bbox
        return x + w / 2, y + h

    def to_contract(self) -> dict[str, Any]:
        """Формат PLAN.md §4.3 (контракт модели А для остальных модулей команды)."""
        return {
            "class": self.cls,
            "bbox": [round(v, 1) for v in self.bbox],
            "conf": round(self.conf, 3),
            "zone_id": self.zone_id,
            "moved_since_prev": self.moved_since_prev,
            "displacement_px": round(self.displacement_px, 1),
            "bbox_shape_delta": round(self.bbox_shape_delta, 3),
            "track_id": self.track_id,
            "unit_id": self.unit_id,
            "activity": self.activity.value,
        }


@runtime_checkable
class Detector(Protocol):
    """Детектор техники. Реализации: core/equipment/detect_yolo.py (LOCAL),
    core/equipment/detect_vlm.py (EXTERNAL)."""
    name: str
    provider: Provider

    def ready(self) -> tuple[bool, str]:
        """(готов ли, человекочитаемая причина если нет) — для UI настроек."""
        ...

    def detect(self, image_bgr: np.ndarray, frame: FrameInfo) -> list[Detection]:
        ...


@dataclass
class CameraGeometry:
    """Привязка камеры к плану площадки: гомография кадр → план (метры).
    None — камера не откалибрована, слияние между камерами для неё выключено."""
    camera_id: int | str
    homography: list[list[float]] | None = None   # 3×3
    image_size: tuple[int, int] | None = None     # (w, h), на котором снимались точки


@dataclass
class Zone:
    id: int
    name: str
    kind: str                                     # "work" | "parking" | "storage" | "restricted"
    camera_id: int | str | None
    polygon: list[tuple[float, float]]            # в пикселях кадра камеры


@dataclass
class UnitState:
    """Единица техники на площадке во времени (результат слияния камер + статусной машины)."""
    unit_id: str
    cls: str
    status: UnitStatus
    first_seen: dt.datetime
    last_seen: dt.datetime
    last_moved: dt.datetime | None
    worked_hours: float = 0.0
    cameras: set[str] = field(default_factory=set)
    site_xy: tuple[float, float] | None = None
    plate: str | None = None
    label: str = ""                               # «Экскаватор №2» для UI


@dataclass
class ActivityInterval:
    """Засчитанный интервал работы — строка журнала моточасов."""
    unit_id: str
    cls: str
    start: dt.datetime
    end: dt.datetime
    hours: float
    stage_id: int | None                          # этап плана, на который списаны часы (None — вне плана)
    frame_ids: list[int | str] = field(default_factory=list)  # кадры-доказательства


# --------------------------------------------------------------------------
# модель Б — этап
# --------------------------------------------------------------------------


@dataclass
class ChecklistResult:
    """Ответ модели Б по одному кадру."""
    answers: dict[str, Answer]                    # ключ признака → да/нет/не уверен
    scores: dict[str, float] = field(default_factory=dict)          # уверенность/сходство по признаку 0..1
    stage_likelihood: dict[int, float] = field(default_factory=dict)  # этап → 0..1 (если модель умеет)
    model: str = ""
    provider: Provider = Provider.LOCAL
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    equipment_hint: dict[str, int] = field(default_factory=dict)    # внешняя VLM может заодно посчитать технику
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def unsure_ratio(self) -> float:
        if not self.answers:
            return 1.0
        return sum(a == Answer.UNSURE for a in self.answers.values()) / len(self.answers)


@runtime_checkable
class StageClassifier(Protocol):
    """Модель Б. Реализации: core/stage/checklist_clip.py (LOCAL, SigLIP),
    core/stage/checklist_vlm.py (EXTERNAL GLM-4.6V или LOCAL VLM по OpenAI-API)."""
    name: str
    provider: Provider

    def ready(self) -> tuple[bool, str]:
        ...

    def assess(self, image_bgr: np.ndarray, frame: FrameInfo,
               keys: list[str] | None = None,
               context: dict[str, Any] | None = None) -> ChecklistResult:
        """keys — какие признаки спрашивать (None — все); context — что уже известно о стройке."""
        ...


@dataclass
class StageObservation:
    """ChecklistResult, привязанный ко времени — вход для sequence-вывода."""
    frame_id: int | str
    camera_id: int | str
    captured_at: dt.datetime
    result: ChecklistResult


class StageStatus(str, enum.Enum):
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    DONE = "done"


@dataclass
class StageState:
    stage_id: int
    status: StageStatus
    progress: float                               # 0..1 внутри этапа
    actual_start: dt.date | None = None
    actual_end: dt.date | None = None
    confidence: float = 0.0
    manual: bool = False                          # отмечено пользователем — модель не перезаписывает
    evidence_frame_ids: list[int | str] = field(default_factory=list)


@dataclass
class StageTimeline:
    """Результат core/stage/sequence.infer(): монотонная хронология этапов."""
    states: dict[int, StageState]
    current_stage: int | None                     # «фронт» — старший идущий этап
    overall_progress: float                       # 0..1 по весам этапов
    daily_front: list[tuple[dt.date, int]]        # день → фронт после сглаживания
    needs_review: list[int | str] = field(default_factory=list)  # кадры с долей «не уверен» выше порога
    rejected_outliers: list[int | str] = field(default_factory=list)  # кадры, противоречащие хронологии
    # Почему текущий этап такой: признаки чек-листа и техника модели А (core/stage/fusion.py).
    # {"text": фраза для UI, "decided_by": checklist | equipment | both, "equipment": [...], "signs": [...]}
    basis: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# план, моточасы, отклонения
# --------------------------------------------------------------------------


@dataclass
class PlanItem:
    """Строка календарного плана на уровне макроэтапа (детализация по кодам xlsx — work_codes)."""
    stage_id: int
    planned_start: dt.date | None
    planned_end: dt.date | None
    name: str = ""
    work_codes: list[str] = field(default_factory=list)     # коды из reference/works_catalog.xlsx
    equipment: dict[str, int] = field(default_factory=dict)  # тип → сколько единиц нужно на этапе
    planned_hours: dict[str, float] = field(default_factory=dict)  # тип → моточасы (авто или вручную)
    hours_manual: bool = False


@dataclass
class HoursBalance:
    """«Временная полоска» по типу техники на этапе.

    Ожидание к «сейчас» считается от начала НАБЛЮДЕНИЯ (первый кадр площадки),
    если камеры начали снимать после начала этапа по плану: что было до первого
    кадра, камера не видела, и считать это «отставанием» техники нельзя.
    `expected_hours` — сколько должно быть отработано к «сейчас» с `expected_from`;
    `planned_observed_hours` — сколько плановых часов этапа приходится на период
    с `expected_from` до конца этапа (с ним сверяются модели А и Б). None — не
    считалось (нет дат плана или «сейчас»).
    """
    stage_id: int | None
    cls: str
    planned_hours: float
    worked_hours: float
    last_worked_at: dt.datetime | None
    expected_hours: float | None = None
    expected_from: dt.datetime | None = None
    planned_observed_hours: float | None = None

    @property
    def remaining_hours(self) -> float:
        return max(0.0, self.planned_hours - self.worked_hours)

    @property
    def done_ratio(self) -> float:
        return 0.0 if self.planned_hours <= 0 else min(1.0, self.worked_hours / self.planned_hours)


class Severity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DeviationType(str, enum.Enum):
    # техника против этапа (ТЗ §2)
    EQUIPMENT_MISSING = "equipment_missing"          # нужной по этапу техники нет
    EQUIPMENT_FORBIDDEN = "equipment_forbidden"      # работает техника, не соответствующая этапу
    PAIR_BROKEN = "pair_broken"                      # экскаватор без самосвалов и т.п. — снижение темпа
    EQUIPMENT_IDLE = "equipment_idle"                # этап идёт, «полоска» не уменьшается
    EQUIPMENT_PARKED_ONLY = "equipment_parked_only"  # техника этапа есть, но вся стоит
    OUTSIDE_ZONE = "outside_zone"
    # сверка моделей А и Б
    HOURS_SPENT_NO_PROGRESS = "hours_spent_no_progress"  # часы отработаны, этап не сменился → задержка
    # график
    STAGE_LATE_START = "stage_late_start"
    STAGE_OVERDUE = "stage_overdue"
    STAGE_EARLY = "stage_early"
    STAGE_OUT_OF_PLAN = "stage_out_of_plan"
    # данные
    NEEDS_REVIEW = "needs_review"                    # много «не уверен» — проверить вручную
    CAMERA_ISSUE = "camera_issue"                    # камера молчит / сдвинута / кадры брак


@dataclass
class DeviationRecord:
    """Отклонение с объяснением. `key` стабилен между пересчётами —
    по нему веб-слой обновляет существующую запись вместо создания дубля."""
    key: str
    type: DeviationType
    severity: Severity
    title: str                                    # коротко, для ленты
    message: str                                  # объяснение «почему»: что видели, что ожидали по плану
    stage_id: int | None = None
    camera_id: int | str | None = None
    zone_id: int | None = None
    frame_ids: list[int | str] = field(default_factory=list)   # снимки-доказательства
    unit_ids: list[str] = field(default_factory=list)
    started_at: dt.datetime | None = None
    last_seen_at: dt.datetime | None = None
    data: dict[str, Any] = field(default_factory=dict)


class Verdict(str, enum.Enum):
    AHEAD = "ahead"          # опережение
    ON_TRACK = "on_track"    # соответствие
    BEHIND = "behind"        # отставание
    NO_PLAN = "no_plan"
    NO_DATA = "no_data"


@dataclass
class SiteReport:
    """Итог по объекту для дашборда (core/analytics/report.build)."""
    verdict: Verdict
    lag_days: float | None                        # >0 — отставание
    expected_progress: float | None               # 0..1 по плану на сегодня
    actual_progress: float                        # 0..1 по факту
    forecast_finish: dt.date | None
    current_stage: int | None
    stage_states: dict[int, StageState]
    hours: list[HoursBalance]
    deviations: list[DeviationRecord]
    explanation: list[str] = field(default_factory=list)   # 2–5 фраз «почему такой вердикт»
