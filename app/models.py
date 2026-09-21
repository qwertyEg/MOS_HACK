"""Модель данных.

Схема — это сами классы, отдельного schema.sql нет: за хакатон схема меняется
пересборкой БД, и держать два источника правды дороже, чем один.

Три вещи, которые стоит объяснить сразу, потому что они неочевидны:

1. Планирование живёт на площадке (`SiteStage`), а не на корпусе. Организаторы
   просили аналитику по объекту целиком. Корпуса (`Building`) остаются
   внутренней единицей измерения — у них своя этажность и свой вес в свёртке.

2. `CameraState` хранит накопленное состояние динамической маски. Это главная
   долгоживущая сущность конвейера: она копится месяцами и переживает
   перестановку камеры через пересчёт гомографией.

3. Ответ модели Б тернарный — да / нет / не уверена. Третье значение нужно,
   чтобы кадр, где признак не разглядеть, не голосовал вовсе, вместо того
   чтобы разбавлять статистику случайным «нет».
"""

from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import (
    Boolean, Date, DateTime, Enum, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------
# перечисления
# --------------------------------------------------------------------------

class Answer(str, enum.Enum):
    YES = "yes"
    NO = "no"
    UNSURE = "unsure"


class EquipmentStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"        # сместилась между кадрами либо изменила позу
    IDLE = "IDLE"            # стоит меньше смены
    PARKED = "PARKED"        # стоит сутками — ждёт вывоза, в этап не засчитывается
    DEPARTED = "DEPARTED"    # ушла с площадки


class ViewType(str, enum.Enum):
    SIDE = "side"            # сбоку: этажность, фасад
    TOP = "top"              # с мачты: котлован, плита, благоустройство
    REMOTE = "remote"        # с соседнего дома: общий план


class ZoneType(str, enum.Enum):
    WORK = "work"
    PARKING = "parking"      # зона отстоя: техника тут не сопоставляется с этапом
    STORAGE = "storage"


class DeviationType(str, enum.Enum):
    STAGE_BEHIND = "STAGE_BEHIND"
    STAGE_AHEAD = "STAGE_AHEAD"
    STAGE_MISMATCH = "STAGE_MISMATCH"
    STAGE_NOT_STARTED = "STAGE_NOT_STARTED"
    EQUIPMENT_MISSING = "EQUIPMENT_MISSING"
    EQUIPMENT_UNEXPECTED = "EQUIPMENT_UNEXPECTED"
    EQUIPMENT_WRONG_ZONE = "EQUIPMENT_WRONG_ZONE"
    EQUIPMENT_PARKED_ONLY = "EQUIPMENT_PARKED_ONLY"
    SITE_IDLE = "SITE_IDLE"
    ACTIVITY_DROP = "ACTIVITY_DROP"
    CREW_SHRINK = "CREW_SHRINK"
    TEMPO_DECAY = "TEMPO_DECAY"
    CAMERA_MOVED = "CAMERA_MOVED"


# --------------------------------------------------------------------------
# справочники
# --------------------------------------------------------------------------

class ObjectType(Base):
    __tablename__ = "object_types"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)


class WorkType(Base):
    """Строка исходного справочника. Храним целиком, вместе с исходным кодом
    и обоснованием отнесения к макроэтапу — чтобы решение было прослеживаемым.
    """
    __tablename__ = "work_types"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), default="")
    parent_code: Mapped[str] = mapped_column(String(32), default="")
    level: Mapped[int] = mapped_column(Integer, default=0)
    name: Mapped[str] = mapped_column(Text)
    macro_stage_id: Mapped[int | None] = mapped_column(
        ForeignKey("macro_stages.id"), nullable=True)
    excluded_reason: Mapped[str] = mapped_column(Text, default="")
    mapping_reason: Mapped[str] = mapped_column(Text, default="")
    object_types: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)


class MacroStage(Base):
    """Макроэтап — то, что различимо с камеры. 8 штук, §3.1 плана."""
    __tablename__ = "macro_stages"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    order_default: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text, default="")
    progress_metric: Mapped[str] = mapped_column(String(64), default="")
    progress_view: Mapped[str] = mapped_column(String(16), default="")

    template: Mapped["StageTemplate"] = relationship(back_populates="macro_stage",
                                                     uselist=False)


class StageTemplate(Base):
    """Шаблон чек-листа: что должно быть, чего быть не должно, что спросить."""
    __tablename__ = "stage_templates"
    id: Mapped[int] = mapped_column(primary_key=True)
    macro_stage_id: Mapped[int] = mapped_column(ForeignKey("macro_stages.id"),
                                                unique=True)
    must_have: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    must_not_have: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    equipment_expected: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    equipment_forbidden: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    measurements: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    questions: Mapped[list[dict]] = mapped_column(JSONB, default=list)

    macro_stage: Mapped[MacroStage] = relationship(back_populates="template")


class StageNorm(Base):
    """Нормативы темпа из открытых данных — лечат холодный старт прогноза."""
    __tablename__ = "stage_norms"
    id: Mapped[int] = mapped_column(primary_key=True)
    object_type_id: Mapped[int | None] = mapped_column(ForeignKey("object_types.id"))
    macro_stage_id: Mapped[int] = mapped_column(ForeignKey("macro_stages.id"))
    unit: Mapped[str] = mapped_column(String(32), default="")
    tempo_min: Mapped[float | None] = mapped_column(Float)
    tempo_median: Mapped[float | None] = mapped_column(Float)
    tempo_max: Mapped[float | None] = mapped_column(Float)
    active_days_ratio: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(Text, default="")


# --------------------------------------------------------------------------
# объект
# --------------------------------------------------------------------------

class Site(Base):
    __tablename__ = "sites"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    address: Mapped[str] = mapped_column(Text, default="")
    object_type_id: Mapped[int | None] = mapped_column(ForeignKey("object_types.id"))
    cadastral_no: Mapped[str] = mapped_column(String(64), default="")
    land_area: Mapped[float | None] = mapped_column(Float)
    permit_no: Mapped[str] = mapped_column(String(64), default="")
    permit_date: Mapped[dt.date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    server_default=func.now())

    object_type: Mapped[ObjectType | None] = relationship()
    buildings: Mapped[list["Building"]] = relationship(
        back_populates="site", cascade="all, delete-orphan")
    stages: Mapped[list["SiteStage"]] = relationship(
        back_populates="site", cascade="all, delete-orphan",
        order_by="SiteStage.order_idx")
    cameras: Mapped[list["Camera"]] = relationship(
        back_populates="site", cascade="all, delete-orphan")


class Building(Base):
    """Корпус. Внутренняя единица измерения: у него своя этажность, свой
    прогресс и свой вес при свёртке в показатели объекта (вес = площадь).
    В отчётности наружу корпуса не фигурируют — так просили организаторы.
    """
    __tablename__ = "buildings"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(128))
    floors_total: Mapped[int | None] = mapped_column(Integer)
    area: Mapped[float | None] = mapped_column(Float)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    structure_type: Mapped[str] = mapped_column(String(128), default="")
    has_underground: Mapped[bool] = mapped_column(Boolean, default=False)

    site: Mapped[Site] = relationship(back_populates="buildings")


class SiteStage(Base):
    """Этап в календарном плане объекта. Даты конкретные, не кварталы —
    организаторы указали это явно. `dates_confirmed` отличает подтверждённые
    пользователем даты от черновых, развёрнутых из вех проектной декларации.
    """
    __tablename__ = "site_stages"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    macro_stage_id: Mapped[int | None] = mapped_column(ForeignKey("macro_stages.id"))
    order_idx: Mapped[int] = mapped_column(Integer, default=0)
    custom_name: Mapped[str] = mapped_column(String(256), default="")
    planned_start: Mapped[dt.date | None] = mapped_column(Date)
    planned_end: Mapped[dt.date | None] = mapped_column(Date)
    dates_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    on_critical_path: Mapped[bool] = mapped_column(Boolean, default=True)
    lag_to_next: Mapped[int] = mapped_column(Integer, default=0)
    equipment_expected: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    equipment_forbidden: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    site: Mapped[Site] = relationship(back_populates="stages")
    macro_stage: Mapped[MacroStage | None] = relationship()

    @property
    def title(self) -> str:
        return self.custom_name or (self.macro_stage.name if self.macro_stage else "?")


class Declaration(Base):
    __tablename__ = "declarations"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    file_key: Mapped[str] = mapped_column(Text, default="")
    parsed: Mapped[dict] = mapped_column(JSONB, default=dict)
    milestones: Mapped[list] = mapped_column(JSONB, default=list)


# --------------------------------------------------------------------------
# наблюдение
# --------------------------------------------------------------------------

class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(128))
    source_type: Mapped[str] = mapped_column(String(32), default="folder")
    source_uri: Mapped[str] = mapped_column(Text, default="")
    view_type: Mapped[ViewType] = mapped_column(Enum(ViewType), default=ViewType.SIDE)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    reference_frame_key: Mapped[str] = mapped_column(Text, default="")

    site: Mapped[Site] = relationship(back_populates="cameras")
    state: Mapped["CameraState"] = relationship(
        back_populates="camera", uselist=False, cascade="all, delete-orphan")
    zones: Mapped[list["Zone"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan")


class CameraState(Base):
    """Состояние маски фона по камере. Живёт месяцами.

    `initial_mask_key` — то, что нарисовал оператор, хранится неизменным:
    по нему считается, сколько маски уже съедено растущим зданием.
    `background_key` — текущая маска, только сжимается.
    `evidence_key` — счётчик по клеткам: устойчивое изменение накапливается
    и пробивает порог, разовое откатывается назад.
    """
    __tablename__ = "camera_states"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), unique=True)
    initial_mask_key: Mapped[str] = mapped_column(Text, default="")
    background_key: Mapped[str] = mapped_column(Text, default="")
    evidence_key: Mapped[str] = mapped_column(Text, default="")
    work_w: Mapped[int] = mapped_column(Integer, default=0)
    work_h: Mapped[int] = mapped_column(Integer, default=0)
    mask_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    windows_accumulated: Mapped[int] = mapped_column(Integer, default=0)
    masked_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    retained: Mapped[float] = mapped_column(Float, default=1.0)
    mask_top_edge_px: Mapped[int | None] = mapped_column(Integer)
    px_per_floor: Mapped[float | None] = mapped_column(Float)
    last_reset_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now())

    camera: Mapped[Camera] = relationship(back_populates="state")


class Zone(Base):
    __tablename__ = "zones"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    building_id: Mapped[int | None] = mapped_column(ForeignKey("buildings.id"))
    name: Mapped[str] = mapped_column(String(128), default="")
    polygon: Mapped[list] = mapped_column(JSONB, default=list)
    zone_type: Mapped[ZoneType] = mapped_column(Enum(ZoneType), default=ZoneType.WORK)
    view_quality: Mapped[float] = mapped_column(Float, default=1.0)

    camera: Mapped[Camera] = relationship(back_populates="zones")


class Frame(Base):
    """Кадр и результаты его обработки.

    Три картинки на кадр: исходник, кадр с погашенным фоном (то, что ушло бы
    в модель Б) и наложение маски красным (чтобы оценить границы глазом).
    Хранятся в объектном хранилище, в БД только ключи.
    """
    __tablename__ = "frames"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    object_key: Mapped[str] = mapped_column(Text)
    masked_key: Mapped[str] = mapped_column(Text, default="")
    overlay_key: Mapped[str] = mapped_column(Text, default="")
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    phash: Mapped[str] = mapped_column(String(32), default="")
    is_night: Mapped[bool] = mapped_column(Boolean, default=False)
    quality_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    reject_reason: Mapped[str] = mapped_column(String(64), default="")
    masked_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    retained: Mapped[float] = mapped_column(Float, default=1.0)
    top_edge_px: Mapped[int | None] = mapped_column(Integer)
    change_pct: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (UniqueConstraint("camera_id", "captured_at",
                                       name="uq_frame_camera_time"),)


# --------------------------------------------------------------------------
# выводы моделей
# --------------------------------------------------------------------------

class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id", ondelete="CASCADE"))
    cls: Mapped[str] = mapped_column("class", String(64))
    bbox: Mapped[list] = mapped_column(JSONB)
    conf: Mapped[float] = mapped_column(Float)
    zone_id: Mapped[int | None] = mapped_column(ForeignKey("zones.id"))
    moved_since_prev: Mapped[bool] = mapped_column(Boolean, default=False)
    displacement_px: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_shape_delta: Mapped[float] = mapped_column(Float, default=0.0)
    track_id: Mapped[int | None] = mapped_column(ForeignKey("equipment_tracks.id"))


class EquipmentTrack(Base):
    """Единица техники во времени. Нужна ровно ради различения «присутствует»
    и «задействована»: организаторы предупредили, что техника неделями стоит
    на площадке в ожидании вывоза, и наличие её в кадре ничего не доказывает.
    """
    __tablename__ = "equipment_tracks"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    cls: Mapped[str] = mapped_column("class", String(64))
    first_seen: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[EquipmentStatus] = mapped_column(
        Enum(EquipmentStatus), default=EquipmentStatus.ACTIVE)
    status_since: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    last_position: Mapped[list] = mapped_column(JSONB, default=list)
    position_variance: Mapped[float] = mapped_column(Float, default=0.0)


class Checklist(Base):
    """Одно заполнение чек-листа по одному кадру. Версии копятся во времени —
    аналитика строится по всей хронологии, а не по последнему кадру.
    """
    __tablename__ = "checklists"
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id", ondelete="CASCADE"))
    site_stage_id: Mapped[int] = mapped_column(
        ForeignKey("site_stages.id", ondelete="CASCADE"))
    zone_id: Mapped[int | None] = mapped_column(ForeignKey("zones.id"))
    model_name: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    server_default=func.now())

    answers: Mapped[list["ChecklistAnswer"]] = relationship(
        back_populates="checklist", cascade="all, delete-orphan")


class ChecklistAnswer(Base):
    __tablename__ = "checklist_answers"
    id: Mapped[int] = mapped_column(primary_key=True)
    checklist_id: Mapped[int] = mapped_column(
        ForeignKey("checklists.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(64), default="")
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[Answer] = mapped_column(Enum(Answer))
    polarity: Mapped[str] = mapped_column(String(16), default="must_have")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    raw_response: Mapped[str] = mapped_column(Text, default="")

    checklist: Mapped[Checklist] = relationship(back_populates="answers")


class Measurement(Base):
    __tablename__ = "measurements"
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id", ondelete="CASCADE"))
    building_id: Mapped[int | None] = mapped_column(ForeignKey("buildings.id"))
    metric: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="")


# --------------------------------------------------------------------------
# аналитика
# --------------------------------------------------------------------------

class DailyActivity(Base):
    """Активность площадки по дням. Питает расчёт темпа в активных днях —
    связка между требованием «работает или стоит» и требованием «предсказывай
    задержку». Ночные смены считаются: работы ночью идут.
    """
    __tablename__ = "daily_activity"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    frames_total: Mapped[int] = mapped_column(Integer, default=0)
    frames_with_motion: Mapped[int] = mapped_column(Integer, default=0)
    night_frames_with_motion: Mapped[int] = mapped_column(Integer, default=0)
    workers_median: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (UniqueConstraint("site_id", "date", name="uq_activity_day"),)


class StageScore(Base):
    __tablename__ = "stage_scores"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_stage_id: Mapped[int] = mapped_column(
        ForeignKey("site_stages.id", ondelete="CASCADE"))
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    probability: Mapped[float] = mapped_column(Float, default=0.0)
    progress_pct: Mapped[float | None] = mapped_column(Float)
    unsure_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    source_frames: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (UniqueConstraint("site_stage_id", "date",
                                       name="uq_score_stage_day"),)


class Deviation(Base):
    __tablename__ = "deviations"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    site_stage_id: Mapped[int | None] = mapped_column(ForeignKey("site_stages.id"))
    type: Mapped[DeviationType] = mapped_column(Enum(DeviationType))
    severity: Mapped[str] = mapped_column(String(16), default="medium")
    detected_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                     server_default=func.now())
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    description: Mapped[str] = mapped_column(Text, default="")
    evidence_frame_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)


class Forecast(Base):
    __tablename__ = "forecasts"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_stage_id: Mapped[int] = mapped_column(
        ForeignKey("site_stages.id", ondelete="CASCADE"))
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    projected_end: Mapped[dt.date | None] = mapped_column(Date)
    delay_days: Mapped[int | None] = mapped_column(Integer)
    delay_low: Mapped[int | None] = mapped_column(Integer)
    delay_high: Mapped[int | None] = mapped_column(Integer)
    method: Mapped[str] = mapped_column(String(64), default="")
    v_recent: Mapped[float | None] = mapped_column(Float)
    v_overall: Mapped[float | None] = mapped_column(Float)
    v_plan: Mapped[float | None] = mapped_column(Float)
    k_idle: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(16), default="low")
    explanation: Mapped[str] = mapped_column(Text, default="")


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_login: Mapped[str] = mapped_column(String(64), default="")
    action: Mapped[str] = mapped_column(String(64))
    entity: Mapped[str] = mapped_column(String(64), default="")
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    server_default=func.now())
