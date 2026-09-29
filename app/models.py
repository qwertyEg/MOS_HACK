"""Схема БД (docs/ARCHITECTURE.md §11).

Типы только переносимые — `JSON`, строки, числа, даты; никаких ARRAY/JSONB:
одна и та же схема работает на SQLite (по умолчанию, ноутбук жюри) и на
PostgreSQL (docker compose). Миграций нет: `create_all` при старте.

Поверх §11 добавлено (только добавлено, ничего не убрано):
- `sites.report`/`report_at` — снимок последнего пересчёта площадки (вердикт,
  «полоски» часов, ряды план/факт), чтобы дашборд не гонял аналитику на
  каждый GET;
- `cameras.source_uri`, `created_at` — адрес камеры-потока (simcam);
- `camera_states.initial_mask_key`, `stage_mask_ratio` — под ручную правку
  маски и «внеочередной» вызов модели Б при сильном изменении маски;
- `frames.status`/`note`/`job_id`/`created_at` — понятный статус анализа
  («отложен: нет ключа ZAI_API_KEY») и прогресс заданий загрузки;
- `stage_states.evidence` — кадры-доказательства этапа;
- `activity_intervals.manual/note` (и `unit_id` допускает NULL) — ручные поправки
  моточасов оператором;
- `plan_items.position`, `deviations.note/created_at/updated_at`;
- таблица `jobs` — задания загрузки/переанализа, переживают рестарт;
- таблицы `annotations` (ручная разметка техники, переживает переанализ) и
  `raw_detections` (ответ детектора до правок — для перепрогона без детектора).
  Новые таблицы, а не колонки старых: `create_all` досоздаёт их на рабочей базе.

Все времена хранятся как UTC без зоны и отдаются как aware-UTC (`UTCDateTime`):
SQLite зону не хранит вовсе, и без этого сравнение времён расходилось бы
между SQLite и PostgreSQL.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    TypeDecorator, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class UTCDateTime(TypeDecorator):
    """Aware-datetime на входе и выходе, в БД — наивное UTC."""
    impl = DateTime(timezone=False)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC)


# --------------------------------------------------------------------------
# пользователи и настройки
# --------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    login: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(256))


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict | list | str | float | None] = mapped_column(JSON, nullable=True)


# --------------------------------------------------------------------------
# объект, камеры, зоны
# --------------------------------------------------------------------------

class Site(Base):
    __tablename__ = "sites"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    address: Mapped[str] = mapped_column(Text, default="")
    object_type: Mapped[str] = mapped_column(String(64), default="Жильё")
    floors_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), default="Europe/Moscow")
    shift_hours: Mapped[float] = mapped_column(Float, default=10.0)
    created_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, default=utcnow)
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    report_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, nullable=True)


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(16), default="upload")   # upload | folder | stream | video
    ingest_key: Mapped[str] = mapped_column(String(64), default="")
    interval_min: Mapped[int] = mapped_column(Integer, default=20)
    homography: Mapped[list | None] = mapped_column(JSON, nullable=True)      # 3×3 кадр → план, метры
    calib_points: Mapped[dict | None] = mapped_column(JSON, nullable=True)    # {image_points, site_points, reproj_error}
    image_w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_frame_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, nullable=True)
    source_uri: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, default=utcnow)


class CameraState(Base):
    """Состояние динамической маски камеры (core.stage.DynamicMask.dumps → хранилище)."""
    __tablename__ = "camera_states"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), unique=True)
    mask_key: Mapped[str] = mapped_column(String(512), default="")
    initial_mask_key: Mapped[str] = mapped_column(String(512), default="")
    counters_key: Mapped[str] = mapped_column(String(512), default="")
    windows: Mapped[int] = mapped_column(Integer, default=0)
    masked_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    retained: Mapped[float] = mapped_column(Float, default=1.0)
    stage_mask_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, nullable=True)


class Zone(Base):
    __tablename__ = "zones"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"),
                                                  nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    kind: Mapped[str] = mapped_column(String(16), default="work")   # work | parking | storage | restricted
    polygon: Mapped[list] = mapped_column(JSON, default=list)       # [[x, y], ...] в пикселях кадра


# --------------------------------------------------------------------------
# кадры и выводы моделей
# --------------------------------------------------------------------------

class Frame(Base):
    __tablename__ = "frames"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    captured_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, index=True)
    key: Mapped[str] = mapped_column(String(512))
    preview_key: Mapped[str] = mapped_column(String(512), default="")
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_night: Mapped[bool] = mapped_column(Boolean, default=False)
    weather: Mapped[str] = mapped_column(String(16), default="unknown")
    quality_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    reject_reason: Mapped[str] = mapped_column(String(256), default="")
    blur: Mapped[float | None] = mapped_column(Float, nullable=True)
    brightness: Mapped[float | None] = mapped_column(Float, nullable=True)
    stage_used: Mapped[bool] = mapped_column(Boolean, default=False)
    processed_a: Mapped[bool] = mapped_column(Boolean, default=False)
    processed_b: Mapped[bool] = mapped_column(Boolean, default=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    # pending | processing | done | postponed | error
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("camera_id", "captured_at", name="uq_frame_camera_time"),)


class EquipmentUnit(Base):
    __tablename__ = "equipment_units"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    uid: Mapped[str] = mapped_column(String(64))
    cls: Mapped[str] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(16), default="idle")
    first_seen: Mapped[dt.datetime] = mapped_column(UTCDateTime)
    last_seen: Mapped[dt.datetime] = mapped_column(UTCDateTime)
    last_moved: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, nullable=True)
    worked_hours: Mapped[float] = mapped_column(Float, default=0.0)
    cameras: Mapped[list] = mapped_column(JSON, default=list)
    plate: Mapped[str | None] = mapped_column(String(32), nullable=True)
    site_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    site_y: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (UniqueConstraint("site_id", "uid", name="uq_unit_site_uid"),)


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(32))     # ключ провайдера из настроек: yolo | glm
    cls: Mapped[str] = mapped_column(String(32))
    conf: Mapped[float] = mapped_column(Float)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    w: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)
    track_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    unit_id: Mapped[int | None] = mapped_column(ForeignKey("equipment_units.id", ondelete="SET NULL"),
                                                nullable=True, index=True)
    moved: Mapped[bool] = mapped_column(Boolean, default=False)
    displacement_px: Mapped[float] = mapped_column(Float, default=0.0)
    shape_delta: Mapped[float] = mapped_column(Float, default=0.0)
    appearance_delta: Mapped[float] = mapped_column(Float, default=0.0)
    activity: Mapped[str] = mapped_column(String(16), default="unknown")
    zone_id: Mapped[int | None] = mapped_column(Integer, nullable=True)   # без FK: удаление зоны не ломает историю
    site_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    site_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)


class ActivityInterval(Base):
    """Строка журнала моточасов. `manual` — поправка оператора («экскаватор
    работал ещё 3 ч, камера не видела»): без единицы техники, часы могут быть
    отрицательными; переанализ ручные поправки не стирает."""
    __tablename__ = "activity_intervals"
    id: Mapped[int] = mapped_column(primary_key=True)
    unit_id: Mapped[int | None] = mapped_column(ForeignKey("equipment_units.id", ondelete="CASCADE"),
                                                nullable=True, index=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    cls: Mapped[str] = mapped_column(String(32))
    stage_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start: Mapped[dt.datetime] = mapped_column(UTCDateTime)
    end: Mapped[dt.datetime] = mapped_column(UTCDateTime)
    hours: Mapped[float] = mapped_column(Float)
    frame_ids: Mapped[list] = mapped_column(JSON, default=list)
    manual: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str] = mapped_column(Text, default="")


class StageObservation(Base):
    __tablename__ = "stage_observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(32))     # siglip | local_vlm | glm
    model: Mapped[str] = mapped_column(String(128), default="")
    answers: Mapped[dict] = mapped_column(JSON, default=dict)
    scores: Mapped[dict] = mapped_column(JSON, default=dict)
    stage_likelihood: Mapped[dict] = mapped_column(JSON, default=dict)
    unsure_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, default=utcnow)


# --------------------------------------------------------------------------
# этапы, план, парк, отклонения
# --------------------------------------------------------------------------

class StageState(Base):
    __tablename__ = "stage_states"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    stage_id: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="not_started")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    actual_start: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    actual_end: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    manual: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("site_id", "stage_id", name="uq_stage_state"),)


class PlanItem(Base):
    __tablename__ = "plan_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    stage_id: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(256), default="")
    work_codes: Mapped[list] = mapped_column(JSON, default=list)
    planned_start: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    planned_end: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    equipment: Mapped[dict] = mapped_column(JSON, default=dict)
    planned_hours: Mapped[dict] = mapped_column(JSON, default=dict)
    hours_manual: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(16), default="manual")   # manual | import | demo
    position: Mapped[int] = mapped_column(Integer, default=0)


class SiteFleet(Base):
    __tablename__ = "site_fleet"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    cls: Mapped[str] = mapped_column(String(32))
    count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (UniqueConstraint("site_id", "cls", name="uq_fleet_cls"),)


class Deviation(Base):
    __tablename__ = "deviations"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(256))
    type: Mapped[str] = mapped_column(String(48))
    severity: Mapped[str] = mapped_column(String(16), default="warning")
    title: Mapped[str] = mapped_column(String(256), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    stage_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    zone_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_ids: Mapped[list] = mapped_column(JSON, default=list)
    unit_ids: Mapped[list] = mapped_column(JSON, default=list)
    started_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_seen_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="open")   # open | ack | resolved
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("site_id", "key", name="uq_deviation_key"),)


class Annotation(Base):
    """Ручная правка разметки техники (требование 3): факт от оператора, а не вывод модели.

    Одна строка — одно утверждение о рамке кадра: `relabel` (класс рамки — cls),
    `delete` (это не техника; scope=camera — и не показывать это место камеры
    дальше), `add` (дорисованная рамка), `unit` (рамка — машина unit_key, класс
    машины cls: склейка / разделение / смена типа единицы), `verify` (кадр
    проверен целиком, без рамки). Рамка правки узнаётся среди рамок детектора
    по IoU, поэтому правки переживают переанализ: конвейер накладывает их поверх
    ответа детектора (app/services/annotations.apply). `batch` — одно действие
    в интерфейсе (склейка единиц — десятки строк), отменяется целиком.
    """
    __tablename__ = "annotations"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id", ondelete="CASCADE"), index=True)
    batch: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(24))          # действие UI: box_relabel | unit_merge | …
    action: Mapped[str] = mapped_column(String(12))        # relabel | delete | add | unit | verify
    scope: Mapped[str] = mapped_column(String(8), default="frame")   # frame | camera (только delete)
    cls: Mapped[str | None] = mapped_column(String(32), nullable=True)
    orig_cls: Mapped[str | None] = mapped_column(String(32), nullable=True)
    orig_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    x: Mapped[float | None] = mapped_column(Float, nullable=True)
    y: Mapped[float | None] = mapped_column(Float, nullable=True)
    w: Mapped[float | None] = mapped_column(Float, nullable=True)
    h: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    author: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, default=utcnow)


class RawDetections(Base):
    """Ответ детектора на кадр как есть — до ручных правок и трекера.

    Нужен, чтобы правка (или её отмена) пересчитывала технику без повторного
    запуска детектора: перепрогон модели А по сохранённым рамкам — секунды,
    а не минуты YOLO на CPU (pipeline.replay_model_a)."""
    __tablename__ = "raw_detections"
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(32))
    items: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("frame_id", "provider", name="uq_raw_frame_provider"),)


class Job(Base):
    """Задание: загрузка файлов, засев демо, переанализ. Прогресс считается по кадрам."""
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    site_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kind: Mapped[str] = mapped_column(String(16), default="upload")    # upload | reprocess | seed | cli
    state: Mapped[str] = mapped_column(String(16), default="ingesting")  # ingesting | queued | failed
    total: Mapped[int] = mapped_column(Integer, default=0)        # кадров сохранено заданием
    duplicates: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, nullable=True)


Index("ix_frames_camera_status", Frame.camera_id, Frame.status)
Index("ix_detections_frame_provider", Detection.frame_id, Detection.provider)
