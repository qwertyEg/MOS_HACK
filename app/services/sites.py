"""Операции над объектом: план, парк, ручные отметки этапов, удаление, переанализ.

Вся валидация пользовательского ввода — здесь, с понятными сообщениями:
роутеры превращают ValueError в 400, а не в 500 (у Дениса битая дата плана
давала 500).
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.models import (
    ActivityInterval, Camera, CameraState, Detection, Deviation, EquipmentUnit, Frame, Job, PlanItem,
    Site, SiteFleet, StageObservation, StageState, Zone, utcnow,
)
from app.services import adapters, pipeline
from app.services import settings as settings_svc
from core import contracts as c
from core import taxonomy

STAGE_STATUSES = tuple(s.value for s in c.StageStatus)


def _date(value: Any, field: str) -> dt.date | None:
    try:
        return adapters._date(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: дата «{value}» не разобрана, нужен формат ГГГГ-ММ-ДД") from exc


def _check_equipment(value: Any, field: str, kind: type = int) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field}: ожидается объект {{тип техники: число}}")
    known = taxonomy.equipment()
    out = {}
    for cls, n in value.items():
        if cls not in known:
            raise ValueError(f"{field}: неизвестный тип техники «{cls}» (см. /api/settings → classes)")
        if isinstance(n, bool) or not isinstance(n, (int, float)) or n < 0:
            raise ValueError(f"{field}.{cls}: ожидается неотрицательное число")
        if kind is int and int(n) != n:
            raise ValueError(f"{field}.{cls}: количество техники — целое число")
        out[cls] = kind(n)
    return out


# --------------------------------------------------------------------------
# план
# --------------------------------------------------------------------------

def parse_plan(payload: Any) -> list[c.PlanItem]:
    if not isinstance(payload, list):
        raise ValueError("план: ожидается список этапов [{stage_id, planned_start, planned_end, ...}]")
    stages = taxonomy.stages()
    items: list[c.PlanItem] = []
    seen: set[int] = set()
    for i, raw in enumerate(payload):
        where = f"строка {i + 1}"
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: ожидается объект")
        try:
            stage_id = int(raw.get("stage_id"))
        except (TypeError, ValueError):
            raise ValueError(f"{where}: stage_id — номер этапа 1..{len(stages)}") from None
        if stage_id not in stages:
            raise ValueError(f"{where}: этапа {stage_id} нет в справочнике (1..{len(stages)})")
        if stage_id in seen:
            raise ValueError(f"{where}: этап {stage_id} указан дважды — один этап, одна строка плана")
        seen.add(stage_id)
        start = _date(raw.get("planned_start"), f"{where}.planned_start")
        end = _date(raw.get("planned_end"), f"{where}.planned_end")
        if start and end and start > end:
            raise ValueError(f"{where}: начало {start} позже окончания {end}")
        codes = raw.get("work_codes") or []
        if not isinstance(codes, list) or not all(isinstance(x, (str, int, float)) for x in codes):
            raise ValueError(f"{where}.work_codes: ожидается список кодов работ")
        items.append(c.PlanItem(
            stage_id=stage_id, planned_start=start, planned_end=end,
            name=str(raw.get("name") or stages[stage_id].name)[:256],
            work_codes=[str(x) for x in codes],
            equipment=_check_equipment(raw.get("equipment"), f"{where}.equipment"),
            planned_hours=_check_equipment(raw.get("planned_hours"), f"{where}.planned_hours", float),
            hours_manual=bool(raw.get("hours_manual", False)),
        ))
    return items


def save_plan(s: Session, site: Site, items: list[c.PlanItem], source: str) -> list[str]:
    """Заменить план объекта. Пустое `equipment` строки — предложение по нормам
    этапа (пользователь поправит); часы без hours_manual — пересчёт по парку."""
    warnings: list[str] = []
    from app.services import providers
    norms = providers.optional_module("core.plan.norms")
    s.execute(delete(PlanItem).where(PlanItem.site_id == site.id))
    for i, item in enumerate(items):
        if not item.equipment and norms is not None:
            try:
                item.equipment = {k: int(v) for k, v in (norms.default_equipment(item.stage_id) or {}).items()}
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"этап {item.stage_id}: нормы техники недоступны ({exc})")
        s.add(adapters.plan_row_from_item(item, site.id, source, i))
    s.flush()
    warnings.extend(pipeline.compute_plan_hours(s, site))
    s.commit()
    return warnings


# --------------------------------------------------------------------------
# парк и этапы
# --------------------------------------------------------------------------

def save_fleet(s: Session, site: Site, payload: Any) -> list[str]:
    if not isinstance(payload, list):
        raise ValueError("парк: ожидается список [{cls, count}]")
    fleet: dict[str, int] = {}
    for i, raw in enumerate(payload):
        if not isinstance(raw, dict) or "cls" not in raw:
            raise ValueError(f"строка {i + 1}: ожидается {{cls, count}}")
        fleet.update(_check_equipment({raw["cls"]: raw.get("count", 0)}, f"строка {i + 1}"))
    s.execute(delete(SiteFleet).where(SiteFleet.site_id == site.id))
    for cls, n in fleet.items():
        s.add(SiteFleet(site_id=site.id, cls=cls, count=n))
    s.flush()
    warnings = pipeline.compute_plan_hours(s, site)
    s.commit()
    return warnings


def patch_stage(s: Session, site: Site, stage_id: int, payload: Any) -> StageState:
    """Ручная отметка этапа. `manual: false` — вернуть этап модели."""
    if stage_id not in taxonomy.stages():
        raise LookupError(f"этапа {stage_id} нет в справочнике")
    if not isinstance(payload, dict):
        raise ValueError("ожидается JSON-объект")
    row = s.scalar(select(StageState).where(StageState.site_id == site.id, StageState.stage_id == stage_id))
    if row is None:
        row = StageState(site_id=site.id, stage_id=stage_id)
        s.add(row)
    if payload.get("manual") is False:
        row.manual = False
        row.note = str(payload.get("note", row.note or ""))
        row.updated_at = utcnow()
        s.commit()
        return row
    if "status" in payload:
        if payload["status"] not in STAGE_STATUSES:
            raise ValueError(f"status: одно из {', '.join(STAGE_STATUSES)}")
        row.status = payload["status"]
    if "progress" in payload:
        p = payload["progress"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise ValueError("progress: число от 0 до 1")
        row.progress = float(p)
    elif payload.get("status") == "done":
        row.progress = 1.0
    if "actual_start" in payload:
        row.actual_start = _date(payload["actual_start"], "actual_start")
    if "actual_end" in payload:
        row.actual_end = _date(payload["actual_end"], "actual_end")
    if row.actual_start and row.actual_end and row.actual_start > row.actual_end:
        raise ValueError("actual_start позже actual_end")
    if "note" in payload:
        row.note = str(payload["note"] or "")[:2000]
    row.manual = True
    row.confidence = 1.0
    row.updated_at = utcnow()
    s.commit()
    return row


def add_hours_correction(s: Session, site: Site, payload: Any) -> ActivityInterval:
    """Ручная поправка моточасов: часы по типу техники (и этапу), можно со знаком
    минус — если модель насчитала лишнее. Попадает в «полоску» как обычный интервал."""
    if not isinstance(payload, dict):
        raise ValueError("ожидается JSON-объект {cls, hours, stage_id?, at?, note?}")
    cls = payload.get("cls")
    if cls not in taxonomy.equipment():
        raise ValueError(f"cls: неизвестный тип техники «{cls}»")
    hours = payload.get("hours")
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or hours == 0 or abs(hours) > 10000:
        raise ValueError("hours: ненулевое число часов (можно отрицательное), по модулю до 10000")
    stage_id = payload.get("stage_id")
    if stage_id is not None and stage_id not in taxonomy.stages():
        raise ValueError(f"stage_id: этапа {stage_id} нет в справочнике")
    at = payload.get("at")
    when = dt.datetime.now(dt.UTC)
    if at:
        try:
            parsed = dt.datetime.fromisoformat(str(at).replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("at: время в ISO 8601") from None
        when = adapters.to_utc(parsed, adapters.site_tz(site))
    row = ActivityInterval(unit_id=None, site_id=site.id, cls=cls, stage_id=stage_id, start=when, end=when,
                           hours=float(hours), frame_ids=[], manual=True,
                           note=str(payload.get("note") or "")[:2000])
    s.add(row)
    s.commit()
    return row


# --------------------------------------------------------------------------
# удаление и переанализ
# --------------------------------------------------------------------------

def _frame_ids_subq(camera_ids):
    return select(Frame.id).where(Frame.camera_id.in_(camera_ids))


def delete_camera(s: Session, camera_id: int) -> None:
    """Порядок удаления явный: на SQLite без каскада ORM-дерево из тысяч кадров
    грузилось бы в память. Файлы в хранилище не трогаем (решение Дениса)."""
    frames = _frame_ids_subq([camera_id])
    cam = s.get(Camera, camera_id)
    if cam is not None:
        # Техника, которую видела только эта камера, уходит вместе с ней (иначе в
        # объекте оставался «фантомный» Экскаватор №2 с моточасами — отчёт UI);
        # у общих с другими камерами единиц камера просто вычёркивается.
        for unit in s.scalars(select(EquipmentUnit).where(EquipmentUnit.site_id == cam.site_id)).all():
            seen_by = [str(x) for x in (unit.cameras or [])]
            if str(camera_id) not in seen_by:
                continue
            rest = [x for x in seen_by if x != str(camera_id)]
            if rest:
                unit.cameras = rest
            else:
                s.execute(delete(ActivityInterval).where(ActivityInterval.unit_id == unit.id))
                s.execute(update(Detection).where(Detection.unit_id == unit.id).values(unit_id=None))
                s.delete(unit)
        s.flush()
    s.execute(delete(Detection).where(Detection.frame_id.in_(frames)))
    s.execute(delete(StageObservation).where(StageObservation.frame_id.in_(frames)))
    s.execute(delete(Frame).where(Frame.camera_id == camera_id))
    s.execute(delete(CameraState).where(CameraState.camera_id == camera_id))
    s.execute(delete(Zone).where(Zone.camera_id == camera_id))
    s.execute(delete(Camera).where(Camera.id == camera_id))
    s.commit()


def delete_site(s: Session, site_id: int) -> None:
    cams = list(s.scalars(select(Camera.id).where(Camera.site_id == site_id)))
    for cid in cams:
        delete_camera(s, cid)
    s.execute(delete(ActivityInterval).where(ActivityInterval.site_id == site_id))
    s.execute(delete(EquipmentUnit).where(EquipmentUnit.site_id == site_id))
    for model in (StageState, PlanItem, SiteFleet, Deviation, Zone):
        s.execute(delete(model).where(model.site_id == site_id))
    s.execute(delete(Job).where(Job.site_id == site_id))
    s.execute(delete(Site).where(Site.id == site_id))
    s.commit()
    pipeline.reset_caches(site_id, cams)


def reprocess(s: Session, site: Site) -> Job:
    """Переанализировать объект текущими провайдерами.

    Результаты других провайдеров не стираются (§10); стираются результаты
    текущих, техника и моточасы (выводятся из модели А заново), маски камер
    (иначе повторный прогон сжимал бы уже сжатую маску — грабли Дениса) и
    неручные состояния этапов. Отклонения сверятся пересчётом по key.
    """
    from app.services.ingest import new_job
    from app.services.queue import frame_queue

    state = settings_svc.get_state(s, fresh=True)
    cams = list(s.scalars(select(Camera.id).where(Camera.site_id == site.id)))
    frame_queue.cancel(cams)
    frame_queue.wait_cameras(cams, timeout=120)

    job = new_job(s, "reprocess", site_id=site.id, message=f"переанализ: {state['model_a']} + {state['model_b']}")
    frames = _frame_ids_subq(cams)
    with pipeline.site_lock(site.id):
        s.execute(delete(Detection).where(Detection.frame_id.in_(frames), Detection.provider == state["model_a"]))
        s.execute(delete(StageObservation).where(StageObservation.frame_id.in_(frames),
                                                 StageObservation.provider == state["model_b"]))
        s.execute(update(Detection).where(Detection.frame_id.in_(frames)).values(unit_id=None))
        s.execute(delete(ActivityInterval).where(ActivityInterval.site_id == site.id,
                                                 ActivityInterval.manual.is_(False)))
        s.execute(delete(EquipmentUnit).where(EquipmentUnit.site_id == site.id))
        s.execute(delete(CameraState).where(CameraState.camera_id.in_(cams)))
        s.execute(delete(StageState).where(StageState.site_id == site.id, StageState.manual.is_(False)))
        s.execute(update(Frame).where(Frame.camera_id.in_(cams)).values(
            processed_a=False, processed_b=False, stage_used=False, status="pending", note="", job_id=job.id))
        n = 0
        for fr in s.scalars(select(Frame).where(Frame.camera_id.in_(cams))):
            n += 1
            if fr.meta and "mask_done" in fr.meta:      # маска строится заново — кадр снова её кормит
                fr.meta = {k: v for k, v in fr.meta.items() if k != "mask_done"}
        job.total = n
        job.state = "queued"
        job.finished_at = utcnow()
        s.commit()
        pipeline.reset_caches(site.id, cams)
    frame_queue.recover()
    return job
