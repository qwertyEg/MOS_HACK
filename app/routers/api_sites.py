"""Объекты: карточки, сводка, план, парк, этапы, техника, отклонения, переанализ."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import auth
from app.db import get_session
from app.models import ActivityInterval, Camera, Deviation, EquipmentUnit, Frame, Site, SiteFleet, utcnow
from app.routers.common import bad, get_or_404, json_body, not_found, num_field, require_obj, str_field
from app.services import pipeline, providers, sites, views
from app.services import settings as settings_svc
from app.services.queue import recomputer

router = APIRouter(prefix="/api", tags=["объекты"], dependencies=[Depends(auth.require_api_user)])

DEVIATION_STATUSES = ("open", "ack", "resolved")


def _site_fields(body: dict, partial: bool) -> dict:
    out: dict = {}
    if not partial or "name" in body:
        out["name"] = str_field(body, "name", required=True)
    if not partial or "address" in body:
        out["address"] = str_field(body, "address", max_len=1000)
    if not partial or "object_type" in body:
        out["object_type"] = str_field(body, "object_type", max_len=64, default="Жильё") or "Жильё"
    if "floors_total" in body:
        out["floors_total"] = num_field(body, "floors_total", lo=0, hi=300, integer=True, allow_none=True)
    if not partial or "timezone" in body:
        tz = str_field(body, "timezone", max_len=64, default="Europe/Moscow") or "Europe/Moscow"
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError):
            raise bad(f"timezone: неизвестный часовой пояс «{tz}» (пример: Europe/Moscow)") from None
        out["timezone"] = tz
    if "shift_hours" in body:
        out["shift_hours"] = num_field(body, "shift_hours", lo=1, hi=24)
    return out


@router.get("/sites")
def list_sites(s: Session = Depends(get_session)) -> list[dict]:
    return [views.site_card(s, site) for site in s.scalars(select(Site).order_by(Site.name))]


@router.post("/sites", status_code=201)
async def create_site(request: Request, s: Session = Depends(get_session)) -> dict:
    body = require_obj(await json_body(request))
    site = Site(**_site_fields(body, partial=False))
    s.add(site)
    s.commit()
    return views.site_card(s, site)


@router.get("/sites/{site_id}")
def get_site(site_id: int, s: Session = Depends(get_session)) -> dict:
    return views.site_card(s, get_or_404(s, Site, site_id, "объект"))


@router.patch("/sites/{site_id}")
async def patch_site(site_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    site = get_or_404(s, Site, site_id, "объект")
    fields = _site_fields(require_obj(await json_body(request)), partial=True)
    for k, v in fields.items():
        setattr(site, k, v)
    if "shift_hours" in fields:
        pipeline.compute_plan_hours(s, site)
    s.commit()
    if {"shift_hours", "floors_total", "object_type"} & fields.keys():
        recomputer.request(site.id)
    return views.site_card(s, site)


@router.delete("/sites/{site_id}")
def delete_site(site_id: int, s: Session = Depends(get_session)) -> dict:
    site = get_or_404(s, Site, site_id, "объект")
    from app.services.queue import camera_ids_for_site, frame_queue
    cams = camera_ids_for_site(site.id)
    frame_queue.cancel(cams)
    frame_queue.wait_cameras(cams, timeout=60)
    sites.delete_site(s, site.id)
    return {"ok": True, "id": site_id}


@router.get("/sites/{site_id}/overview")
def site_overview(site_id: int, s: Session = Depends(get_session)) -> dict:
    site = get_or_404(s, Site, site_id, "объект")
    if site.report is None:
        recomputer.run(site.id)
        s.refresh(site)
    return views.overview(s, site)


@router.post("/sites/{site_id}/recompute")
def site_recompute(site_id: int, s: Session = Depends(get_session)) -> dict:
    site = get_or_404(s, Site, site_id, "объект")
    recomputer.run(site.id)
    s.refresh(site)
    return views.overview(s, site)


# --------------------------------------------------------------------------
# план и парк
# --------------------------------------------------------------------------

def _plan_list(s: Session, site: Site) -> list[dict]:
    return [views.plan_json(p) for p in views._plan_rows(s, site.id)]


@router.get("/sites/{site_id}/plan")
def get_plan(site_id: int, s: Session = Depends(get_session)) -> list[dict]:
    return _plan_list(s, get_or_404(s, Site, site_id, "объект"))


@router.put("/sites/{site_id}/plan")
async def put_plan(site_id: int, request: Request, s: Session = Depends(get_session)) -> list[dict]:
    site = get_or_404(s, Site, site_id, "объект")
    try:
        items = sites.parse_plan(await json_body(request))
    except ValueError as exc:
        raise bad(str(exc)) from None
    sites.save_plan(s, site, items, "manual")
    await run_in_threadpool(recomputer.run, site.id)   # пересчёт до 0.8 с — не в цикле событий
    return _plan_list(s, site)


@router.post("/sites/{site_id}/plan/import")
async def import_plan(site_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    site = get_or_404(s, Site, site_id, "объект")
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise bad("нужен файл плана (поле file): .xlsx или .csv")
    data = await upload.read()
    if not data:
        raise bad("файл плана пустой")
    apply = str(form.get("apply", "1")).lower() not in ("0", "false", "no")
    try:
        importer = providers.module("core.plan.importer")
    except ImportError:
        raise bad("импорт плана недоступен: модуль core.plan не подключён") from None
    try:
        items, warnings = await run_in_threadpool(importer.parse, data, upload.filename or "plan.csv")
    except (ValueError, KeyError) as exc:
        raise bad(f"план не разобран: {exc}") from None
    if not items:
        raise bad("в файле не найдено ни одного этапа: " + "; ".join(warnings[:5]))
    warnings = list(warnings)
    if apply:
        warnings += sites.save_plan(s, site, list(items), "import")
        await run_in_threadpool(recomputer.run, site.id)   # пересчёт до 0.8 с — не в цикле событий
        plan = _plan_list(s, site)
    else:
        from app.services.adapters import jsonable
        plan = jsonable(list(items))
    return {"plan": plan, "warnings": warnings, "applied": apply}


@router.post("/sites/{site_id}/plan/demo")
async def demo_plan(site_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    """Демо-план под диапазон кадров объекта — с явной пометкой source=demo.
    Это не «автоплан по датам фото» Никиты: даты этапов не подгоняются под
    факт, а раскладываются по нормативным долям на период съёмки."""
    site = get_or_404(s, Site, site_id, "объект")
    body = require_obj(await json_body(request, default={}))
    first, last = s.execute(select(func.min(Frame.captured_at), func.max(Frame.captured_at))
                            .join(Camera, Camera.id == Frame.camera_id).where(Camera.site_id == site.id)).one()
    try:
        start = sites._date(body.get("start"), "start") or (first.date() if first else dt.date.today())
        end = sites._date(body.get("end"), "end") or (last.date() if last and last.date() > start else None)
    except ValueError as exc:
        raise bad(str(exc)) from None
    if end and start > end:
        raise bad("start позже end")
    stage_ids = body.get("stage_ids")
    if stage_ids is not None and (not isinstance(stage_ids, list) or not all(isinstance(x, int) for x in stage_ids)):
        raise bad("stage_ids: список номеров этапов")
    try:
        importer = providers.module("core.plan.importer")
    except ImportError:
        raise bad("демо-план недоступен: модуль core.plan не подключён") from None
    items = importer.demo_plan(start, end, stage_ids=stage_ids)
    warnings = [f"Демо-план: этапы разложены на {start}…{end or 'по нормам'}; это не реальный график — "
                "поправьте даты в редакторе плана"]
    warnings += sites.save_plan(s, site, list(items), "demo")
    await run_in_threadpool(recomputer.run, site.id)   # пересчёт до 0.8 с — не в цикле событий
    return {"plan": _plan_list(s, site), "warnings": warnings, "applied": True}


@router.get("/sites/{site_id}/fleet")
def get_fleet(site_id: int, s: Session = Depends(get_session)) -> list[dict]:
    site = get_or_404(s, Site, site_id, "объект")
    rows = s.scalars(select(SiteFleet).where(SiteFleet.site_id == site.id).order_by(SiteFleet.cls))
    return [{"cls": r.cls, "count": r.count} for r in rows]


@router.put("/sites/{site_id}/fleet")
async def put_fleet(site_id: int, request: Request, s: Session = Depends(get_session)) -> list[dict]:
    site = get_or_404(s, Site, site_id, "объект")
    try:
        sites.save_fleet(s, site, await json_body(request))
    except ValueError as exc:
        raise bad(str(exc)) from None
    await run_in_threadpool(recomputer.run, site.id)   # пересчёт до 0.8 с — не в цикле событий
    return get_fleet(site_id, s)


# --------------------------------------------------------------------------
# этапы, техника, отклонения
# --------------------------------------------------------------------------

@router.patch("/sites/{site_id}/stages/{stage_id}")
async def patch_stage(site_id: int, stage_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    site = get_or_404(s, Site, site_id, "объект")
    try:
        row = sites.patch_stage(s, site, stage_id, await json_body(request))
    except LookupError as exc:
        raise not_found(str(exc)) from None
    except ValueError as exc:
        raise bad(str(exc)) from None
    await run_in_threadpool(recomputer.run, site.id)   # пересчёт до 0.8 с — не в цикле событий
    s.refresh(row)
    return views.stage_state_json(row)


@router.get("/sites/{site_id}/equipment")
def site_equipment(site_id: int, s: Session = Depends(get_session)) -> dict:
    site = get_or_404(s, Site, site_id, "объект")
    units = s.scalars(select(EquipmentUnit).where(EquipmentUnit.site_id == site.id)
                      .order_by(EquipmentUnit.cls, EquipmentUnit.uid))
    return {"units": [views.unit_json(u) for u in units], "balances": views.balances_json(site)}


@router.get("/units/{unit_id}")
def get_unit(unit_id: int, s: Session = Depends(get_session)) -> dict:
    unit = get_or_404(s, EquipmentUnit, unit_id, "единица техники")
    intervals = s.scalars(select(ActivityInterval).where(ActivityInterval.unit_id == unit.id)
                          .order_by(ActivityInterval.start))
    return {"unit": views.unit_json(unit), "intervals": [views.interval_json(iv) for iv in intervals],
            "detections": views.unit_detections(s, unit)}


@router.get("/sites/{site_id}/hours")
def list_hours(site_id: int, manual: bool = True, s: Session = Depends(get_session)) -> list[dict]:
    """Журнал моточасов объекта; по умолчанию — ручные поправки оператора."""
    site = get_or_404(s, Site, site_id, "объект")
    q = select(ActivityInterval).where(ActivityInterval.site_id == site.id)
    if manual:
        q = q.where(ActivityInterval.manual.is_(True))
    return [views.interval_json(iv) for iv in s.scalars(q.order_by(ActivityInterval.start.desc()).limit(2000))]


@router.post("/sites/{site_id}/hours", status_code=201)
async def add_hours(site_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    """Поправить отработанные моточасы вручную: {cls, hours, stage_id?, at?, note?}."""
    site = get_or_404(s, Site, site_id, "объект")
    try:
        row = sites.add_hours_correction(s, site, await json_body(request))
    except ValueError as exc:
        raise bad(str(exc)) from None
    await run_in_threadpool(recomputer.run, site.id)   # пересчёт до 0.8 с — не в цикле событий
    return views.interval_json(row)


@router.delete("/sites/{site_id}/hours/{interval_id}")
def delete_hours(site_id: int, interval_id: int, s: Session = Depends(get_session)) -> dict:
    row = s.get(ActivityInterval, interval_id)
    if row is None or row.site_id != site_id or not row.manual:
        raise not_found(f"ручная поправка {interval_id} не найдена")
    s.delete(row)
    s.commit()
    recomputer.run(site_id)
    return {"ok": True, "id": interval_id}


@router.get("/sites/{site_id}/deviations")
def site_deviations(site_id: int, status: str = "open,ack", limit: int = 200,
                    s: Session = Depends(get_session)) -> list[dict]:
    site = get_or_404(s, Site, site_id, "объект")
    wanted = DEVIATION_STATUSES if status == "all" else tuple(x.strip() for x in status.split(",") if x.strip())
    if not wanted or any(x not in DEVIATION_STATUSES for x in wanted):
        raise bad("status: open | ack | resolved | all (можно через запятую)")
    rows = list(s.scalars(select(Deviation).where(Deviation.site_id == site.id, Deviation.status.in_(wanted))
                          .order_by(Deviation.last_seen_at.desc()).limit(max(1, min(limit, 1000)))))
    rows.sort(key=lambda d: (views.SEVERITY_RANK.get(d.severity, 3),
                             -(d.last_seen_at.timestamp() if d.last_seen_at else 0)))
    return views.deviations_json(s, rows)


@router.patch("/deviations/{deviation_id}")
async def patch_deviation(deviation_id: int, request: Request, s: Session = Depends(get_session)) -> dict:
    dev = get_or_404(s, Deviation, deviation_id, "отклонение")
    body = require_obj(await json_body(request))
    if "status" in body:
        if body["status"] not in DEVIATION_STATUSES:
            raise bad("status: open | ack | resolved")
        dev.status = body["status"]
    if "note" in body:
        dev.note = str_field(body, "note", max_len=2000)
    dev.updated_at = utcnow()
    s.commit()
    return views.deviations_json(s, [dev])[0]


@router.post("/sites/{site_id}/retry-errors", status_code=202)
def retry_errors(site_id: int, s: Session = Depends(get_session)) -> dict:
    """Кадры с ошибкой (упал провайдер, порог был NaN, процесс падал на кадре) — снова в очередь.
    Сделанное по кадру не повторяется: флаги processed_a / processed_b сохраняются."""
    site = get_or_404(s, Site, site_id, "объект")
    cams = select(Camera.id).where(Camera.site_id == site.id)
    n = 0
    for fr in s.scalars(select(Frame).where(Frame.camera_id.in_(cams), Frame.status == "error")):
        fr.status = "pending"
        fr.meta = {k: v for k, v in (fr.meta or {}).items() if k != "crashes"}
        n += 1
    s.commit()
    from app.services.queue import frame_queue
    frame_queue.recover()
    return {"frames": n}


@router.post("/sites/{site_id}/reprocess", status_code=202)
def reprocess(site_id: int, s: Session = Depends(get_session)) -> dict:
    site = get_or_404(s, Site, site_id, "объект")
    job = sites.reprocess(s, site)
    state = settings_svc.get_state(s)
    return {"job_id": job.id, "frames": job.total, "model_a": state["model_a"], "model_b": state["model_b"]}
