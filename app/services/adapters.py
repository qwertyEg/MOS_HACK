"""Строки БД ↔ датаклассы `core/contracts.py`.

Ядро не знает про БД (правило контрактов), поэтому все переводы — здесь, в
одном месте: ключи JSON в БД строковые (этап `"3"`), а в контрактах — int;
времена в БД — UTC, в контрактах — aware-datetime; enum'ы в БД — их `.value`.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import enum
from collections import defaultdict
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models as m
from core import contracts as c


# --------------------------------------------------------------------------
# общее
# --------------------------------------------------------------------------

def iso(value: dt.datetime | dt.date | None) -> str | None:
    return value.isoformat() if value is not None else None


def jsonable(obj: Any) -> Any:
    """Всё, что возвращает ядро, — в JSON для колонок JSON и ответов API."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(jsonable(k)) if not isinstance(k, str) else k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    return str(obj)


def site_tz(site: m.Site | None) -> dt.tzinfo:
    try:
        return ZoneInfo(site.timezone) if site and site.timezone else dt.UTC
    except (ZoneInfoNotFoundError, ValueError):
        return dt.UTC


def to_utc(value: dt.datetime, tz: dt.tzinfo) -> dt.datetime:
    """Наивное время (имя файла, EXIF, форма) — это время площадки."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=tz)
    return value.astimezone(dt.UTC)


def _enum(cls: type[enum.Enum], value: Any, default: enum.Enum) -> Any:
    try:
        return cls(value)
    except ValueError:
        return default


def _int_keys(d: dict | None) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _date(value: Any) -> dt.date | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


# --------------------------------------------------------------------------
# кадр, камера, зоны
# --------------------------------------------------------------------------

def frame_info(fr: m.Frame, cam: m.Camera) -> c.FrameInfo:
    return c.FrameInfo(
        frame_id=fr.id, camera_id=cam.id, site_id=cam.site_id, captured_at=fr.captured_at,
        width=fr.width or 0, height=fr.height or 0, is_night=fr.is_night,
        weather=_enum(c.Weather, fr.weather, c.Weather.UNKNOWN), quality_ok=fr.quality_ok,
        reject_reason=fr.reject_reason or "", blur=fr.blur, brightness=fr.brightness,
        meta=dict(fr.meta or {}),
    )


def apply_quality(fr: m.Frame, q: c.QualityReport) -> None:
    fr.quality_ok = bool(q.quality_ok)
    fr.is_night = bool(q.is_night)
    fr.weather = q.weather.value if isinstance(q.weather, enum.Enum) else str(q.weather)
    fr.reject_reason = q.reject_reason or ""
    fr.blur = float(q.blur) if q.blur is not None else None
    fr.brightness = float(q.brightness) if q.brightness is not None else None


def geometry(cam: m.Camera) -> c.CameraGeometry | None:
    if not cam.homography:
        return None
    size = (cam.image_w, cam.image_h) if cam.image_w and cam.image_h else None
    return c.CameraGeometry(camera_id=cam.id, homography=cam.homography, image_size=size)


def zone(z: m.Zone) -> c.Zone:
    return c.Zone(id=z.id, name=z.name, kind=z.kind, camera_id=z.camera_id,
                  polygon=[(float(p[0]), float(p[1])) for p in (z.polygon or [])])


def zones_for_camera(s: Session, cam: m.Camera) -> list[c.Zone]:
    rows = s.scalars(select(m.Zone).where(
        m.Zone.site_id == cam.site_id,
        (m.Zone.camera_id == cam.id) | (m.Zone.camera_id.is_(None)))).all()
    return [zone(z) for z in rows]


def zones_for_site(s: Session, site_id: int) -> list[c.Zone]:
    return [zone(z) for z in s.scalars(select(m.Zone).where(m.Zone.site_id == site_id)).all()]


# --------------------------------------------------------------------------
# модель А
# --------------------------------------------------------------------------

def detection(row: m.Detection, unit_uid: str | None = None) -> c.Detection:
    extra = dict(row.extra or {})
    site_xy = (row.site_x, row.site_y) if row.site_x is not None and row.site_y is not None else None
    return c.Detection(
        cls=row.cls, conf=row.conf, bbox=(row.x, row.y, row.w, row.h),
        source=extra.get("source", row.provider), track_id=row.track_id,
        unit_id=unit_uid if unit_uid is not None else extra.get("unit_uid"),
        moved_since_prev=row.moved, displacement_px=row.displacement_px,
        bbox_shape_delta=row.shape_delta, appearance_delta=row.appearance_delta,
        activity=_enum(c.Activity, row.activity, c.Activity.UNKNOWN), zone_id=row.zone_id,
        site_xy=site_xy, extra=extra,
    )


def detection_row(d: c.Detection, frame_id: int, provider: str, unit_row_id: int | None) -> m.Detection:
    x, y, w, h = (float(v) for v in d.bbox)
    extra = jsonable(dict(d.extra or {}))
    extra.setdefault("source", d.source)
    if d.unit_id is not None:
        extra["unit_uid"] = d.unit_id
    sx, sy = (float(d.site_xy[0]), float(d.site_xy[1])) if d.site_xy else (None, None)
    act = d.activity.value if isinstance(d.activity, enum.Enum) else str(d.activity)
    return m.Detection(
        frame_id=frame_id, provider=provider, cls=d.cls, conf=float(d.conf), x=x, y=y, w=w, h=h,
        track_id=str(d.track_id) if d.track_id is not None else None, unit_id=unit_row_id,
        moved=bool(d.moved_since_prev), displacement_px=float(d.displacement_px or 0.0),
        shape_delta=float(d.bbox_shape_delta or 0.0), appearance_delta=float(d.appearance_delta or 0.0),
        activity=act, zone_id=d.zone_id, site_x=sx, site_y=sy, extra=extra,
    )


def unit_state(row: m.EquipmentUnit) -> c.UnitState:
    site_xy = (row.site_x, row.site_y) if row.site_x is not None and row.site_y is not None else None
    return c.UnitState(
        unit_id=row.uid, cls=row.cls, status=_enum(c.UnitStatus, row.status, c.UnitStatus.IDLE),
        first_seen=row.first_seen, last_seen=row.last_seen, last_moved=row.last_moved,
        worked_hours=row.worked_hours or 0.0, cameras={str(x) for x in (row.cameras or [])},
        site_xy=site_xy, plate=row.plate, label=row.label or "",
    )


def apply_unit(row: m.EquipmentUnit, u: c.UnitState) -> None:
    row.cls = u.cls
    row.label = u.label or row.label or ""
    row.status = u.status.value if isinstance(u.status, enum.Enum) else str(u.status)
    row.first_seen = u.first_seen
    row.last_seen = u.last_seen
    row.last_moved = u.last_moved
    row.worked_hours = float(u.worked_hours or 0.0)
    row.cameras = sorted(str(x) for x in (u.cameras or ()))
    row.plate = u.plate
    if u.site_xy:
        row.site_x, row.site_y = float(u.site_xy[0]), float(u.site_xy[1])


def interval(row: m.ActivityInterval, uid: str) -> c.ActivityInterval:
    return c.ActivityInterval(unit_id=uid, cls=row.cls, start=row.start, end=row.end, hours=row.hours,
                              stage_id=row.stage_id, frame_ids=list(row.frame_ids or []))


def units(s: Session, site_id: int) -> list[c.UnitState]:
    rows = s.scalars(select(m.EquipmentUnit).where(m.EquipmentUnit.site_id == site_id)).all()
    return [unit_state(r) for r in rows]


def intervals(s: Session, site_id: int) -> list[c.ActivityInterval]:
    uids = dict(s.execute(select(m.EquipmentUnit.id, m.EquipmentUnit.uid)
                          .where(m.EquipmentUnit.site_id == site_id)).all())
    rows = s.scalars(select(m.ActivityInterval).where(m.ActivityInterval.site_id == site_id)
                     .order_by(m.ActivityInterval.start)).all()
    return [interval(r, uids.get(r.unit_id, str(r.unit_id)) if r.unit_id is not None else f"manual:{r.cls}")
            for r in rows]


def detections_by_frame(s: Session, frame_ids: list[int], preferred: str | None) -> dict[int, list[m.Detection]]:
    """Детекции кадров одного провайдера: текущего, если он размечал кадр, иначе
    последнего размечавшего. Смешивать провайдеров на кадре нельзя — одна машина
    получилась бы двумя рамками."""
    if not frame_ids:
        return {}
    grouped: dict[int, dict[str, list[m.Detection]]] = defaultdict(lambda: defaultdict(list))
    for chunk_start in range(0, len(frame_ids), 500):
        chunk = frame_ids[chunk_start:chunk_start + 500]
        for d in s.scalars(select(m.Detection).where(m.Detection.frame_id.in_(chunk))
                           .order_by(m.Detection.id)).all():
            grouped[d.frame_id][d.provider].append(d)
    out: dict[int, list[m.Detection]] = {}
    for fid, by_provider in grouped.items():
        if preferred and preferred in by_provider:
            out[fid] = by_provider[preferred]
        else:
            latest = max(by_provider, key=lambda p: max(d.id for d in by_provider[p]))
            out[fid] = by_provider[latest]
    return out


def unit_uids(s: Session, unit_ids: set[int]) -> dict[int, str]:
    if not unit_ids:
        return {}
    return dict(s.execute(select(m.EquipmentUnit.id, m.EquipmentUnit.uid)
                          .where(m.EquipmentUnit.id.in_(list(unit_ids)))).all())


def contract_detections(s: Session, rows: list[m.Detection]) -> list[c.Detection]:
    uids = unit_uids(s, {r.unit_id for r in rows if r.unit_id is not None})
    return [detection(r, uids.get(r.unit_id) if r.unit_id is not None else None) for r in rows]


def recent(s: Session, site_id: int, now: dt.datetime, window_h: float,
           preferred: str | None) -> list[tuple[c.FrameInfo, list[c.Detection]]]:
    """Кадры площадки с детекциями за окно до `now` (вход правил аналитики).
    Кадры без рамок тоже нужны: «самосвалов не было в окне» — это пустые кадры."""
    start = now - dt.timedelta(hours=window_h)
    rows = s.execute(
        select(m.Frame, m.Camera).join(m.Camera, m.Camera.id == m.Frame.camera_id)
        .where(m.Camera.site_id == site_id, m.Frame.processed_a.is_(True),
               m.Frame.captured_at >= start, m.Frame.captured_at <= now)
        .order_by(m.Frame.captured_at)).all()
    dets = detections_by_frame(s, [fr.id for fr, _ in rows], preferred)
    all_rows = [d for lst in dets.values() for d in lst]
    uids = unit_uids(s, {d.unit_id for d in all_rows if d.unit_id is not None})
    out = []
    for fr, cam in rows:
        lst = [detection(d, uids.get(d.unit_id) if d.unit_id is not None else None)
               for d in dets.get(fr.id, [])]
        out.append((frame_info(fr, cam), lst))
    return out


def last_detections_by_camera(s: Session, site_id: int, preferred: str | None
                              ) -> dict[int, tuple[dt.datetime, list[c.Detection]]]:
    """Для EquipmentEngine.restore: последний обработанный кадр каждой камеры."""
    out: dict[int, tuple[dt.datetime, list[c.Detection]]] = {}
    cams = s.scalars(select(m.Camera).where(m.Camera.site_id == site_id)).all()
    for cam in cams:
        fr = s.scalar(select(m.Frame).where(m.Frame.camera_id == cam.id, m.Frame.processed_a.is_(True))
                      .order_by(m.Frame.captured_at.desc()).limit(1))
        if fr is None:
            continue
        rows = detections_by_frame(s, [fr.id], preferred).get(fr.id, [])
        out[cam.id] = (fr.captured_at, contract_detections(s, rows))
    return out


# --------------------------------------------------------------------------
# модель Б
# --------------------------------------------------------------------------

def checklist_result(row: m.StageObservation) -> c.ChecklistResult:
    raw = dict(row.raw or {})
    answers = {k: _enum(c.Answer, v, c.Answer.UNSURE) for k, v in (row.answers or {}).items()}
    return c.ChecklistResult(
        answers=answers, scores={k: float(v) for k, v in (row.scores or {}).items()},
        stage_likelihood={k: float(v) for k, v in _int_keys(row.stage_likelihood).items()},
        model=row.model or "", provider=_enum(c.Provider, raw.get("provider_kind", "local"), c.Provider.LOCAL),
        latency_ms=row.latency_ms or 0.0, cost_usd=row.cost_usd or 0.0,
        equipment_hint=dict(raw.get("equipment_hint") or {}), raw=raw,
    )


def observation_row(res: c.ChecklistResult, frame_id: int, provider: str) -> m.StageObservation:
    raw = jsonable(dict(res.raw or {}))
    raw["provider_kind"] = res.provider.value if isinstance(res.provider, enum.Enum) else str(res.provider)
    if res.equipment_hint:
        raw["equipment_hint"] = jsonable(res.equipment_hint)
    return m.StageObservation(
        frame_id=frame_id, provider=provider, model=res.model or "",
        answers={k: (v.value if isinstance(v, enum.Enum) else str(v)) for k, v in res.answers.items()},
        scores=jsonable(res.scores or {}), stage_likelihood=jsonable(res.stage_likelihood or {}),
        unsure_ratio=float(res.unsure_ratio), latency_ms=float(res.latency_ms or 0.0),
        cost_usd=float(res.cost_usd or 0.0), raw=raw,
    )


def observations(s: Session, site_id: int, preferred: str | None) -> list[c.StageObservation]:
    """Наблюдения площадки по времени. Провайдера не смешиваем: у SigLIP и GLM
    разная калибровка «да/нет», хронология по смеси скакала бы."""
    rows = s.execute(
        select(m.StageObservation, m.Frame).join(m.Frame, m.Frame.id == m.StageObservation.frame_id)
        .join(m.Camera, m.Camera.id == m.Frame.camera_id)
        .where(m.Camera.site_id == site_id).order_by(m.Frame.captured_at, m.StageObservation.id)).all()
    providers = {o.provider for o, _ in rows}
    use = preferred if preferred in providers else (rows[-1][0].provider if rows else None)
    return [c.StageObservation(frame_id=fr.id, camera_id=fr.camera_id, captured_at=fr.captured_at,
                               result=checklist_result(o))
            for o, fr in rows if o.provider == use]


def stage_state(row: m.StageState) -> c.StageState:
    return c.StageState(
        stage_id=row.stage_id, status=_enum(c.StageStatus, row.status, c.StageStatus.NOT_STARTED),
        progress=row.progress or 0.0, actual_start=row.actual_start, actual_end=row.actual_end,
        confidence=row.confidence or 0.0, manual=row.manual, evidence_frame_ids=list(row.evidence or []),
    )


def apply_stage_state(row: m.StageState, st: c.StageState) -> None:
    row.status = st.status.value if isinstance(st.status, enum.Enum) else str(st.status)
    row.progress = float(max(0.0, min(1.0, st.progress or 0.0)))
    row.actual_start = _date(st.actual_start)      # реализация может отдать datetime — в колонку Date только дату
    row.actual_end = _date(st.actual_end)
    row.confidence = float(st.confidence or 0.0)
    row.evidence = jsonable(list(st.evidence_frame_ids or []))[:50]
    row.updated_at = dt.datetime.now(dt.UTC)


# --------------------------------------------------------------------------
# план и отклонения
# --------------------------------------------------------------------------

def plan_item(row: m.PlanItem) -> c.PlanItem:
    return c.PlanItem(
        stage_id=row.stage_id, planned_start=row.planned_start, planned_end=row.planned_end,
        name=row.name or "", work_codes=list(row.work_codes or []),
        equipment={k: int(v) for k, v in (row.equipment or {}).items()},
        planned_hours={k: float(v) for k, v in (row.planned_hours or {}).items()},
        hours_manual=row.hours_manual,
    )


def plan_items(s: Session, site_id: int) -> list[c.PlanItem]:
    rows = s.scalars(select(m.PlanItem).where(m.PlanItem.site_id == site_id)
                     .order_by(m.PlanItem.position, m.PlanItem.stage_id)).all()
    return [plan_item(r) for r in rows]


def plan_row_from_item(item: c.PlanItem, site_id: int, source: str, position: int) -> m.PlanItem:
    return m.PlanItem(
        site_id=site_id, stage_id=int(item.stage_id), name=item.name or "",
        work_codes=[str(x) for x in (item.work_codes or [])],
        planned_start=_date(item.planned_start), planned_end=_date(item.planned_end),
        equipment={k: int(v) for k, v in (item.equipment or {}).items()},
        planned_hours={k: float(v) for k, v in (item.planned_hours or {}).items()},
        hours_manual=bool(item.hours_manual), source=source, position=position,
    )


def apply_deviation(row: m.Deviation, rec: c.DeviationRecord) -> None:
    row.type = rec.type.value if isinstance(rec.type, enum.Enum) else str(rec.type)
    row.severity = rec.severity.value if isinstance(rec.severity, enum.Enum) else str(rec.severity)
    row.title = (rec.title or "")[:256]
    row.message = rec.message or ""
    row.stage_id = rec.stage_id
    row.camera_id = int(rec.camera_id) if isinstance(rec.camera_id, (int, str)) and str(rec.camera_id).isdigit() else None
    row.zone_id = rec.zone_id
    row.frame_ids = jsonable(list(rec.frame_ids or []))
    row.unit_ids = jsonable(list(rec.unit_ids or []))
    row.data = jsonable(dict(rec.data or {}))
