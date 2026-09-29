"""JSON-представления для UI (формы ответов — docs/ARCHITECTURE.md §12).

Поля из таблицы §12 — договор с UI: их не переименовываем и не убираем, только
добавляем. Всё тяжёлое (вердикт, «полоски» часов, ряды план/факт) берётся из
`sites.report` — снимка последнего пересчёта, а не считается на каждый GET.
"""
from __future__ import annotations

import datetime as dt
import functools
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import storage
from app.config import settings
from app.models import (
    ActivityInterval, Camera, CameraState, Detection, Deviation, EquipmentUnit, Frame, Job, PlanItem,
    Site, StageObservation, StageState, Zone,
)
from app.services import adapters, providers
from app.services import settings as settings_svc
from app.services.analysis import detection_json
from app.services.queue import frame_queue
from core import taxonomy

iso = adapters.iso
SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}


def _url(key: str | None) -> str | None:
    return storage.get().url(key) if key else None


def frame_urls(fr: Frame) -> dict[str, Any]:
    return {"url": _url(fr.preview_key or fr.key), "full_url": _url(fr.key),
            "annotated_url": f"/api/frames/{fr.id}/annotated.jpg"}


# --------------------------------------------------------------------------
# объект
# --------------------------------------------------------------------------

def _report(site: Site) -> dict[str, Any]:
    return dict(site.report or {})


def _plan_rows(s: Session, site_id: int) -> list[PlanItem]:
    return list(s.scalars(select(PlanItem).where(PlanItem.site_id == site_id)
                          .order_by(PlanItem.position, PlanItem.stage_id)))


def _last_frame(s: Session, site_id: int) -> Frame | None:
    return s.scalar(select(Frame).join(Camera, Camera.id == Frame.camera_id)
                    .where(Camera.site_id == site_id).order_by(Frame.captured_at.desc()).limit(1))


def site_card(s: Session, site: Site) -> dict[str, Any]:
    rep = _report(site)
    verdict = rep.get("verdict") or ("no_plan" if not _plan_rows(s, site.id) else "no_data")
    last = _last_frame(s, site.id)
    active_units = s.scalar(select(func.count()).select_from(EquipmentUnit).where(
        EquipmentUnit.site_id == site.id, EquipmentUnit.status == "active")) or 0
    open_devs = s.scalar(select(func.count()).select_from(Deviation).where(
        Deviation.site_id == site.id, Deviation.status == "open")) or 0
    cameras = s.scalar(select(func.count()).select_from(Camera).where(Camera.site_id == site.id)) or 0
    current = rep.get("current_stage")
    progress = rep.get("actual_progress")
    if progress is None:
        progress = rep.get("overall_progress") or 0.0
    return {
        "id": site.id, "name": site.name, "verdict": verdict, "lag_days": rep.get("lag_days"),
        "current_stage": current, "current_stage_name": taxonomy.stage_name(current) if current else None,
        "progress": progress, "active_units": active_units, "open_deviations": open_devs,
        "last_frame_at": iso(last.captured_at) if last else None,
        "thumb": _url(last.preview_key or last.key) if last else None,
        # сверх §12
        "address": site.address, "object_type": site.object_type, "floors_total": site.floors_total,
        "timezone": site.timezone, "shift_hours": site.shift_hours, "created_at": iso(site.created_at),
        "cameras": cameras, "report_at": iso(site.report_at),
    }


@functools.lru_cache(maxsize=1)
def _catalog_names() -> dict[str, tuple[str, str]]:
    """Ключ работы (как в PlanItem.work_codes) → (код, название).

    Ключ каталога — собственный код строки («12.3.1.») или «родитель/название»
    для строк-детализаций без своего кода (core.plan.catalog.WorkItem.key)."""
    catalog = providers.optional_module("core.plan.catalog")
    if catalog is None:
        return {}
    try:
        return {getattr(w, "key", w.code): (w.code, w.name) for w in catalog.load()}
    except Exception:  # noqa: BLE001
        return {}


def stage_works(stage_id: int, plan_row: PlanItem | None) -> list[dict[str, str]]:
    """Коды работ из xlsx при этапе: из плана, если там указаны, иначе — каталог."""
    if plan_row is not None and plan_row.work_codes:
        names = _catalog_names()
        out = []
        for key in plan_row.work_codes:
            code, name = names.get(key, (key, ""))
            if not code and "/" in key:      # детализация без своего кода: код родителя
                code = key.split("/", 1)[0]
            out.append({"code": code, "name": name, "key": key})
        return out
    catalog = providers.optional_module("core.plan.catalog")
    if catalog is None:
        return []
    try:
        items = catalog.works_for_stage(stage_id)
    except Exception:  # noqa: BLE001
        return []
    return [{"code": w.code, "name": w.name, "key": getattr(w, "key", w.code)} for w in items
            if getattr(w, "status", "substage") == "substage" and w.code][:40]


def _active_plan_stages(plan: list[PlanItem], today: dt.date) -> set[int]:
    return {p.stage_id for p in plan if p.planned_start and p.planned_end and p.planned_start <= today <= p.planned_end}


def _equipment_summary(s: Session, site: Site, plan: list[PlanItem], rep: dict) -> list[dict[str, Any]]:
    units = list(s.scalars(select(EquipmentUnit).where(EquipmentUnit.site_id == site.id)))
    balances = rep.get("balances") or []
    now = dt.datetime.fromisoformat(rep["now"]) if rep.get("now") else dt.datetime.now(dt.UTC)
    stages_now = _active_plan_stages(plan, now.date())
    if not stages_now and rep.get("current_stage"):
        stages_now = {rep["current_stage"]}
    selected = [b for b in balances if b.get("stage_id") in stages_now] if stages_now else balances

    by_cls: dict[str, dict[str, Any]] = {}

    def entry(cls: str) -> dict[str, Any]:
        if cls not in by_cls:
            by_cls[cls] = {"cls": cls, "name": taxonomy.equipment_name(cls), "units": 0, "active": 0, "idle": 0,
                           "parked": 0, "departed": 0, "planned_hours": 0.0, "worked_hours": 0.0,
                           "remaining_hours": 0.0, "done_ratio": 0.0, "stage_ids": [], "last_worked_at": None}
        return by_cls[cls]

    for u in units:
        e = entry(u.cls)
        status = u.status if u.status in ("active", "idle", "parked", "departed") else "idle"
        e[status] += 1
        if status != "departed":
            e["units"] += 1
    for b in selected:
        e = entry(b["cls"])
        planned, worked = float(b.get("planned_hours") or 0), float(b.get("worked_hours") or 0)
        e["planned_hours"] += planned
        e["worked_hours"] += worked
        e["remaining_hours"] += max(0.0, planned - worked)
        if b.get("stage_id") is not None and b["stage_id"] not in e["stage_ids"]:
            e["stage_ids"].append(b["stage_id"])
        if b.get("last_worked_at") and (e["last_worked_at"] is None or b["last_worked_at"] > e["last_worked_at"]):
            e["last_worked_at"] = b["last_worked_at"]
    out = []
    for e in by_cls.values():
        for k in ("planned_hours", "worked_hours", "remaining_hours"):
            e[k] = round(e[k], 2)
        e["done_ratio"] = round(min(1.0, e["worked_hours"] / e["planned_hours"]), 3) if e["planned_hours"] > 0 else 0.0
        out.append(e)
    out.sort(key=lambda e: (-e["units"], -e["planned_hours"], e["cls"]))
    return out


def overview(s: Session, site: Site) -> dict[str, Any]:
    rep = _report(site)
    plan = _plan_rows(s, site.id)
    plan_by_stage = {p.stage_id: p for p in plan}
    states = {r.stage_id: r for r in s.scalars(select(StageState).where(StageState.site_id == site.id))}

    stages = []
    for st in taxonomy.stages().values():
        p, row = plan_by_stage.get(st.id), states.get(st.id)
        stages.append({
            "id": st.id, "name": st.name, "key": st.key,
            "status": row.status if row else "not_started",
            "progress": round(row.progress, 3) if row else 0.0,
            "planned_start": iso(p.planned_start) if p else None,
            "planned_end": iso(p.planned_end) if p else None,
            "actual_start": iso(row.actual_start) if row else None,
            "actual_end": iso(row.actual_end) if row else None,
            "manual": bool(row.manual) if row else False,
            "works": stage_works(st.id, p),
            # сверх §12
            "in_plan": p is not None, "confidence": round(row.confidence, 3) if row else 0.0,
            "note": row.note if row else "", "evidence_frame_ids": list(row.evidence or []) if row else [],
            "equipment_expected": list(st.equipment_expected), "equipment_forbidden": list(st.equipment_forbidden),
        })

    cams = list(s.scalars(select(Camera).where(Camera.site_id == site.id).order_by(Camera.id)))
    state = settings_svc.get_state(s)
    cameras = []
    for cam in cams:
        last = s.scalar(select(Frame).where(Frame.camera_id == cam.id).order_by(Frame.captured_at.desc()).limit(1))
        last_a = s.scalar(select(Frame).where(Frame.camera_id == cam.id, Frame.processed_a.is_(True))
                          .order_by(Frame.captured_at.desc()).limit(1))
        units_now = 0
        if last_a is not None:
            dets = adapters.detections_by_frame(s, [last_a.id], state["model_a"]).get(last_a.id, [])
            uids = {d.unit_id for d in dets if d.unit_id is not None}
            units_now = len(uids) if uids else len(dets)
        cameras.append({
            "id": cam.id, "name": cam.name,
            "last_frame": ({"id": last.id, "url": _url(last.preview_key or last.key),
                            "captured_at": iso(last.captured_at), "full_url": _url(last.key),
                            "annotated_url": f"/api/frames/{last.id}/annotated.jpg"} if last else None),
            "units_now": units_now,
            "kind": cam.kind, "calibrated": bool(cam.homography), "pending": frame_queue.pending(cam.id),
        })

    deviations = list(s.scalars(select(Deviation).where(Deviation.site_id == site.id,
                                                        Deviation.status.in_(("open", "ack")))))
    deviations.sort(key=lambda d: (SEVERITY_RANK.get(d.severity, 3),
                                   -(d.last_seen_at.timestamp() if d.last_seen_at else 0)))
    current = rep.get("current_stage")
    return {
        "site": site_card(s, site),
        "report": {
            "verdict": rep.get("verdict") or ("no_plan" if not plan else "no_data"),
            "lag_days": rep.get("lag_days"),
            "expected_progress": rep.get("expected_progress"),
            "actual_progress": rep.get("actual_progress"),
            "forecast_finish": rep.get("forecast_finish"),
            "explanation": rep.get("explanation") or [],
            "current_stage": current, "current_stage_name": taxonomy.stage_name(current) if current else None,
            "stage_basis": rep.get("stage_basis") or {},   # почему этап такой: чек-лист + техника
            "needs_review": rep.get("needs_review") or [], "now": rep.get("now"),
            "computed_at": iso(site.report_at), "errors": rep.get("errors") or [],
        },
        "stages": stages,
        "equipment": _equipment_summary(s, site, plan, rep),
        "deviations": deviations_json(s, deviations[:100]),
        "cameras": cameras,
        "series": {"days": [], "expected": [], "actual": [], **(rep.get("series") or {})},
    }


# --------------------------------------------------------------------------
# отклонения
# --------------------------------------------------------------------------

def deviations_json(s: Session, rows: list[Deviation]) -> list[dict[str, Any]]:
    frame_ids = {int(f) for d in rows for f in (d.frame_ids or [])[:6] if str(f).isdigit()}
    frames = {f.id: f for f in s.scalars(select(Frame).where(Frame.id.in_(frame_ids)))} if frame_ids else {}
    cam_ids = {d.camera_id for d in rows if d.camera_id} | {f.camera_id for f in frames.values()}
    cams = dict(s.execute(select(Camera.id, Camera.name).where(Camera.id.in_(cam_ids))).all()) if cam_ids else {}
    zone_ids = {d.zone_id for d in rows if d.zone_id}
    zones = dict(s.execute(select(Zone.id, Zone.name).where(Zone.id.in_(zone_ids))).all()) if zone_ids else {}
    out = []
    for d in rows:
        evid = []
        for fid in (d.frame_ids or [])[:6]:
            fr = frames.get(int(fid)) if str(fid).isdigit() else None
            if fr is not None:
                evid.append({"id": fr.id, "camera_id": fr.camera_id, "captured_at": iso(fr.captured_at),
                             **frame_urls(fr)})
        out.append({
            "id": d.id, "key": d.key, "type": d.type, "severity": d.severity, "title": d.title,
            "message": d.message, "stage_id": d.stage_id,
            "stage_name": taxonomy.stage_name(d.stage_id) if d.stage_id else None,
            "camera_id": d.camera_id, "camera_name": cams.get(d.camera_id),
            "zone_id": d.zone_id, "zone_name": zones.get(d.zone_id),
            "frame_ids": list(d.frame_ids or []), "frames": evid, "unit_ids": list(d.unit_ids or []),
            "started_at": iso(d.started_at), "last_seen_at": iso(d.last_seen_at), "status": d.status,
            "data": d.data or {}, "note": d.note,
        })
    return out


# --------------------------------------------------------------------------
# камеры, зоны, кадры
# --------------------------------------------------------------------------

def camera_json(s: Session, cam: Camera) -> dict[str, Any]:
    total = s.scalar(select(func.count()).select_from(Frame).where(Frame.camera_id == cam.id)) or 0
    state = s.scalar(select(CameraState).where(CameraState.camera_id == cam.id))
    return {
        "id": cam.id, "site_id": cam.site_id, "name": cam.name, "kind": cam.kind,
        "interval_min": cam.interval_min, "ingest_key": cam.ingest_key,
        "ingest_url": settings.public_base_url.rstrip("/") + "/api/ingest",
        "homography": cam.homography, "calib_points": cam.calib_points,
        "calibrated": bool(cam.homography), "image_w": cam.image_w, "image_h": cam.image_h,
        "last_frame_at": iso(cam.last_frame_at), "source_uri": cam.source_uri,
        "frames_total": total, "pending": frame_queue.pending(cam.id),
        "mask": ({"masked_ratio": state.masked_ratio, "windows": state.windows,
                  "updated_at": iso(state.updated_at), "url": f"/api/cameras/{cam.id}/mask.png"}
                 if state is not None and state.mask_key else None),
    }


def zone_json(z: Zone) -> dict[str, Any]:
    return {"id": z.id, "site_id": z.site_id, "camera_id": z.camera_id, "name": z.name,
            "kind": z.kind, "polygon": z.polygon}


def frame_summary(fr: Frame, det_count: int = 0) -> dict[str, Any]:
    return {
        "id": fr.id, "camera_id": fr.camera_id, "captured_at": iso(fr.captured_at), **frame_urls(fr),
        "width": fr.width, "height": fr.height, "is_night": fr.is_night, "weather": fr.weather,
        "quality_ok": fr.quality_ok, "reject_reason": fr.reject_reason, "stage_used": fr.stage_used,
        "processed_a": fr.processed_a, "processed_b": fr.processed_b, "status": fr.status, "note": fr.note,
        "detections_count": det_count, "ts_source": (fr.meta or {}).get("ts_source"),
    }


def detection_counts(s: Session, frame_ids: list[int], preferred: str) -> dict[int, int]:
    dets = adapters.detections_by_frame(s, frame_ids, preferred)
    return {fid: len(lst) for fid, lst in dets.items()}


def frame_detail(s: Session, fr: Frame) -> dict[str, Any]:
    cam = s.get(Camera, fr.camera_id)
    state = settings_svc.get_state(s)
    rows = adapters.detections_by_frame(s, [fr.id], state["model_a"]).get(fr.id, [])
    dets = adapters.contract_detections(s, rows)
    det_json = []
    for row, d in zip(rows, dets):
        item = detection_json(d)
        item.update({"id": row.id, "provider": row.provider, "unit_row_id": row.unit_id,
                     "site_xy": [row.site_x, row.site_y] if row.site_x is not None else None})
        det_json.append(item)

    obs_rows = list(s.scalars(select(StageObservation).where(StageObservation.frame_id == fr.id)
                              .order_by(StageObservation.id.desc())))
    obs = next((o for o in obs_rows if o.provider == state["model_b"]), obs_rows[0] if obs_rows else None)
    review = state["thresholds"]["stage"]["unsure_review_ratio"]
    checklist = None
    if obs is not None:
        signs = taxonomy.signs()
        checklist = {
            "id": obs.id, "provider": obs.provider, "model": obs.model, "answers": obs.answers,
            "questions": {k: signs[k].question for k in obs.answers if k in signs},
            "scores": obs.scores, "stage_likelihood": obs.stage_likelihood, "unsure_ratio": obs.unsure_ratio,
            "needs_review": obs.unsure_ratio > review, "latency_ms": obs.latency_ms, "cost_usd": obs.cost_usd,
            "created_at": iso(obs.created_at),
        }
    prev_id = s.scalar(select(Frame.id).where(Frame.camera_id == fr.camera_id, Frame.captured_at < fr.captured_at)
                       .order_by(Frame.captured_at.desc()).limit(1))
    next_id = s.scalar(select(Frame.id).where(Frame.camera_id == fr.camera_id, Frame.captured_at > fr.captured_at)
                       .order_by(Frame.captured_at).limit(1))
    return {
        **frame_summary(fr, len(dets)),
        "site_id": cam.site_id if cam else None, "camera_name": cam.name if cam else None,
        "quality": {"quality_ok": fr.quality_ok, "is_night": fr.is_night, "weather": fr.weather,
                    "reject_reason": fr.reject_reason, "blur": fr.blur, "brightness": fr.brightness},
        "detections": det_json,
        "counts": dict(Counter(d.cls for d in dets)),
        "checklist": checklist,
        "observations": [{"id": o.id, "provider": o.provider, "model": o.model, "unsure_ratio": o.unsure_ratio}
                         for o in obs_rows],
        "meta": fr.meta or {}, "sha256": fr.sha256, "job_id": fr.job_id,
        "prev_id": prev_id, "next_id": next_id,
    }


# --------------------------------------------------------------------------
# техника, план, задания
# --------------------------------------------------------------------------

def unit_json(u: EquipmentUnit) -> dict[str, Any]:
    return {
        "id": u.id, "uid": u.uid, "site_id": u.site_id, "cls": u.cls, "name": taxonomy.equipment_name(u.cls),
        "label": u.label, "status": u.status, "first_seen": iso(u.first_seen), "last_seen": iso(u.last_seen),
        "last_moved": iso(u.last_moved), "worked_hours": round(u.worked_hours or 0.0, 2),
        "cameras": u.cameras or [], "plate": u.plate,
        "site_xy": [u.site_x, u.site_y] if u.site_x is not None else None,
    }


def interval_json(iv: ActivityInterval) -> dict[str, Any]:
    return {"id": iv.id, "unit_id": iv.unit_id, "cls": iv.cls, "name": taxonomy.equipment_name(iv.cls),
            "stage_id": iv.stage_id, "start": iso(iv.start), "end": iso(iv.end), "hours": round(iv.hours, 3),
            "frame_ids": iv.frame_ids or [], "manual": iv.manual, "note": iv.note}


def balances_json(site: Site) -> list[dict[str, Any]]:
    out = []
    for b in (site.report or {}).get("balances") or []:
        planned, worked = float(b.get("planned_hours") or 0), float(b.get("worked_hours") or 0)
        out.append({**b, "name": taxonomy.equipment_name(b["cls"]),
                    "stage_name": taxonomy.stage_name(b.get("stage_id")) if b.get("stage_id") else None,
                    "remaining_hours": round(max(0.0, planned - worked), 2),
                    "done_ratio": round(min(1.0, worked / planned), 3) if planned > 0 else 0.0})
    return out


def plan_json(p: PlanItem) -> dict[str, Any]:
    return {"stage_id": p.stage_id, "name": p.name or taxonomy.stage_name(p.stage_id),
            "work_codes": list(p.work_codes or []), "planned_start": iso(p.planned_start),
            "planned_end": iso(p.planned_end), "equipment": dict(p.equipment or {}),
            "planned_hours": dict(p.planned_hours or {}), "hours_manual": p.hours_manual,
            "id": p.id, "source": p.source}


def stage_state_json(row: StageState) -> dict[str, Any]:
    return {"stage_id": row.stage_id, "name": taxonomy.stage_name(row.stage_id), "status": row.status,
            "progress": row.progress, "actual_start": iso(row.actual_start), "actual_end": iso(row.actual_end),
            "confidence": row.confidence, "manual": row.manual, "note": row.note,
            "updated_at": iso(row.updated_at)}


def job_json(s: Session, job: Job) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for status, n in s.execute(select(Frame.status, func.count()).where(Frame.job_id == job.id)
                               .group_by(Frame.status)).all():
        counts[status] = n
    done, failed, postponed = counts["done"], counts["error"], counts["postponed"]
    in_work = counts["pending"] + counts["processing"]
    if job.state in ("ingesting", "failed"):
        state = job.state
    elif in_work:
        state = "processing"
    elif postponed:
        state = "postponed"
    else:
        state = "done"
    errors = list(job.errors or [])
    if failed:
        for fid, note in s.execute(select(Frame.id, Frame.note).where(Frame.job_id == job.id,
                                                                       Frame.status == "error").limit(20)).all():
            errors.append(f"кадр {fid}: {note}")
    postponed_reason = None
    if postponed:
        postponed_reason = s.scalar(select(Frame.note).where(Frame.job_id == job.id,
                                                             Frame.status == "postponed").limit(1))
    total = max(job.total, sum(counts.values()))
    return {"id": job.id, "state": state, "total": total, "done": done, "errors": errors,
            "kind": job.kind, "site_id": job.site_id, "camera_id": job.camera_id, "failed": failed,
            "postponed": postponed, "postponed_reason": postponed_reason, "pending": in_work,
            "duplicates": job.duplicates, "skipped": job.skipped, "message": job.message,
            "created_at": iso(job.created_at), "finished_at": iso(job.finished_at)}


def unit_detections(s: Session, unit: EquipmentUnit, limit: int = 50) -> list[dict[str, Any]]:
    rows = s.execute(select(Detection, Frame).join(Frame, Frame.id == Detection.frame_id)
                     .where(Detection.unit_id == unit.id).order_by(Frame.captured_at.desc()).limit(limit)).all()
    return [{"frame_id": fr.id, "camera_id": fr.camera_id, "captured_at": iso(fr.captured_at),
             "bbox": [d.x, d.y, d.w, d.h], "activity": d.activity, "moved": d.moved,
             "displacement_px": d.displacement_px, "zone_id": d.zone_id, **frame_urls(fr)} for d, fr in rows]
