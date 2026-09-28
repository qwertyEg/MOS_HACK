"""Конвейер кадра и пересчёт площадки (docs/ARCHITECTURE.md §4).

    кадр → quality.assess → [маска] → модель А (каждый кадр, днём и ночью)
         → модель Б (только usable_for_stage, не чаще stage_every_h на камеру,
           внеочередно — при сильном изменении маски) → запись
    → пересчёт площадки с дебаунсом: sequence.infer → stage_states (ручные не
      трогаем) → hours.balances → rules.evaluate → deviations (upsert по key)
      → report.build → sites.report

Модели А и Б независимы: не готов один провайдер — второй всё равно
отрабатывает, а кадр получает статус `postponed` с причиной и будет дообработан,
когда провайдер оживёт (флаги processed_a / processed_b говорят, чего не хватает).
Ошибка одного шага не роняет кадр целиком и тем более очередь.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app import db, storage
from app.models import (
    ActivityInterval, Camera, CameraState, Detection, Deviation, EquipmentUnit, Frame, PlanItem,
    SiteFleet, Site, StageObservation, StageState, utcnow,
)
from app.services import adapters, providers
from app.services import settings as settings_svc
from app.services.providers import ProviderUnavailable, registry
from core import contracts as c
from core import taxonomy

log = logging.getLogger(__name__)

_engines: dict[int, Any] = {}          # site_id → EquipmentEngine
_masks: dict[int, Any] = {}            # camera_id → DynamicMask
_site_locks: dict[int, threading.Lock] = defaultdict(threading.Lock)
_recompute_locks: dict[int, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


def site_lock(site_id: int) -> threading.Lock:
    with _locks_guard:
        return _site_locks[site_id]


def reset_caches(site_id: int | None = None, camera_ids: list[int] | None = None) -> None:
    """Забыть состояние моделей в памяти (переанализ, смена порогов, калибровка)."""
    if site_id is None:
        _engines.clear()
        _masks.clear()
        return
    _engines.pop(site_id, None)
    for cid in camera_ids or []:
        _masks.pop(cid, None)


# --------------------------------------------------------------------------
# качество кадра
# --------------------------------------------------------------------------

def basic_quality(img: np.ndarray) -> c.QualityReport:
    """Запасная оценка, если core.stage не подключён (пороги — из ветки Дениса):
    темно/засвет/нет контраста — брак; ИК-ночь почти монохромна."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, std = float(gray.mean()), float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sat = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    night = sat < 12.0 or mean < 55
    reason = ""
    if mean < 18:
        reason = "слишком темно"
    elif mean > 243:
        reason = "засвет"
    elif std < 8:
        reason = "нет контраста"
    ok = not reason
    return c.QualityReport(quality_ok=ok, is_night=night or mean < 18, weather=c.Weather.UNKNOWN,
                           reject_reason=reason, blur=blur, brightness=mean,
                           usable_for_stage=ok and not night)


def assess_quality(img: np.ndarray, captured_at: dt.datetime) -> c.QualityReport:
    quality = providers.optional_module("core.stage.quality")
    if quality is None:
        return basic_quality(img)
    return quality.assess(img, captured_at=captured_at)


# --------------------------------------------------------------------------
# динамическая маска
# --------------------------------------------------------------------------

def _mask_key(camera_id: int) -> str:
    return f"masks/{camera_id}/mask.bin"


def load_mask(s: Session, cam: Camera) -> Any | None:
    if cam.id in _masks:
        return _masks[cam.id]
    stage = providers.optional_module("core.stage")
    if stage is None or not hasattr(stage, "DynamicMask"):
        return None
    state = s.scalar(select(CameraState).where(CameraState.camera_id == cam.id))
    if state is not None and state.mask_key:
        try:
            mask = stage.DynamicMask.loads(storage.get().get(state.mask_key))
            _masks[cam.id] = mask
            return mask
        except Exception as exc:  # noqa: BLE001 — маска восстановима наблюдением
            log.warning("маска камеры %s не читается (%s) — начинаю заново", cam.id, exc)
    return None


def _update_mask(s: Session, cam: Camera, img: np.ndarray, when: dt.datetime,
                 thresholds: dict) -> tuple[Any | None, bool]:
    """Обновить маску годным дневным кадром. → (маска, изменилась ли сильно)."""
    stage = providers.optional_module("core.stage")
    if stage is None or not hasattr(stage, "DynamicMask"):
        return None, False
    mask = load_mask(s, cam)
    if mask is None:
        mask = stage.DynamicMask.new(img.shape[:2])
    try:
        mask.update(img, when)
    except Exception as exc:  # noqa: BLE001 — камеру переставили / сменилось разрешение
        log.warning("маска камеры %s сброшена: %s", cam.id, exc)
        mask = stage.DynamicMask.new(img.shape[:2])
        mask.update(img, when)
    _masks[cam.id] = mask

    state = s.scalar(select(CameraState).where(CameraState.camera_id == cam.id))
    if state is None:
        state = CameraState(camera_id=cam.id)
        s.add(state)
    state.mask_key = storage.get().put(_mask_key(cam.id), mask.dumps(), "application/octet-stream")
    state.windows = (state.windows or 0) + 1
    state.masked_ratio = float(mask.masked_ratio)
    state.updated_at = utcnow()
    changed = (state.stage_mask_ratio is not None and
               abs(state.masked_ratio - state.stage_mask_ratio) >= thresholds["pipeline"]["stage_mask_change"])
    return mask, changed


# --------------------------------------------------------------------------
# модель А
# --------------------------------------------------------------------------

def _engine(s: Session, site: Site, thresholds: dict, provider: str) -> Any:
    engine = _engines.get(site.id)
    if engine is not None:
        return engine
    eq = providers.module("core.equipment")
    config = eq.EquipmentConfig.from_dict(thresholds.get("equipment") or {})
    engine = eq.EquipmentEngine(config)
    engine.restore(adapters.units(s, site.id), adapters.last_detections_by_camera(s, site.id, provider))
    _engines[site.id] = engine
    return engine


def _write_equipment(s: Session, fr: Frame, site: Site, update: Any, provider: str) -> int:
    s.execute(delete(Detection).where(Detection.frame_id == fr.id, Detection.provider == provider))
    uid_to_id: dict[str, int] = {}
    for u in update.units or []:
        row = s.scalar(select(EquipmentUnit).where(EquipmentUnit.site_id == site.id,
                                                   EquipmentUnit.uid == u.unit_id))
        if row is None:
            row = EquipmentUnit(site_id=site.id, uid=u.unit_id, cls=u.cls,
                                first_seen=u.first_seen, last_seen=u.last_seen)
            s.add(row)
        adapters.apply_unit(row, u)
        if not row.label:
            row.label = taxonomy.equipment_name(u.cls)
        s.flush()
        uid_to_id[u.unit_id] = row.id

    def unit_row_id(uid: str | None) -> int | None:
        if uid is None:
            return None
        if uid not in uid_to_id:
            found = s.scalar(select(EquipmentUnit.id).where(EquipmentUnit.site_id == site.id,
                                                            EquipmentUnit.uid == uid))
            if found is None:
                return None
            uid_to_id[uid] = found
        return uid_to_id[uid]

    for d in update.detections or []:
        s.add(adapters.detection_row(d, fr.id, provider, unit_row_id(d.unit_id)))
    for iv in update.intervals or []:
        uid = unit_row_id(iv.unit_id)
        if uid is None:
            continue
        exists = s.scalar(select(ActivityInterval.id).where(
            ActivityInterval.unit_id == uid, ActivityInterval.start == iv.start, ActivityInterval.end == iv.end))
        if exists is not None:
            continue
        s.add(ActivityInterval(unit_id=uid, site_id=site.id, cls=iv.cls, stage_id=iv.stage_id,
                               start=iv.start, end=iv.end, hours=float(iv.hours),
                               frame_ids=adapters.jsonable(list(iv.frame_ids or []))))
    return len(update.detections or [])


def _run_model_a(s: Session, fr: Frame, cam: Camera, site: Site, img: np.ndarray,
                 info: c.FrameInfo, state: dict) -> None:
    name = state["model_a"]
    detector = registry.require("detector", name)
    with registry.call_lock("detector", name):
        detections = detector.detect(img, info)
    with site_lock(site.id):
        engine = _engine(s, site, state["thresholds"], name)
        update = engine.process(info, img, detections, adapters.geometry(cam),
                                adapters.zones_for_camera(s, cam), adapters.plan_items(s, site.id))
        _write_equipment(s, fr, site, update, name)
        s.commit()


# --------------------------------------------------------------------------
# модель Б
# --------------------------------------------------------------------------

def _stage_due(s: Session, fr: Frame, cam: Camera, thresholds: dict, mask_changed: bool) -> bool:
    """Не чаще раза в stage_every_h съёмки на камеру — окно в обе стороны,
    чтобы кадр, догруженный задним числом, тоже не дублировал соседний вызов."""
    if mask_changed:
        return True
    every = dt.timedelta(hours=float(thresholds["pipeline"]["stage_every_h"]))
    if every.total_seconds() <= 0:
        return True
    near = s.scalar(select(func.count()).select_from(Frame).where(
        Frame.camera_id == cam.id, Frame.stage_used.is_(True), Frame.id != fr.id,
        Frame.captured_at > fr.captured_at - every, Frame.captured_at < fr.captured_at + every))
    return not near


def stage_context(s: Session, site: Site) -> dict[str, Any]:
    """Что уже известно о стройке — подсказка модели Б (контекст Никиты)."""
    states = {r.stage_id: r.status for r in s.scalars(select(StageState).where(StageState.site_id == site.id))}
    report = site.report or {}
    return {
        "site_id": site.id, "object_type": site.object_type, "floors_total": site.floors_total,
        "current_stage": report.get("current_stage"), "stage_states": states,
        "done_stages": sorted(k for k, v in states.items() if v == "done"),
    }


def _run_model_b(s: Session, fr: Frame, cam: Camera, site: Site, img: np.ndarray,
                 info: c.FrameInfo, state: dict, mask: Any | None) -> None:
    name = state["model_b"]
    classifier = registry.require("classifier", name)
    image = img
    if mask is not None and getattr(mask, "masked_ratio", 0) > 0:
        try:
            image = mask.apply(img, mode="darken")
        except Exception as exc:  # noqa: BLE001 — без маски хуже, но не повод терять кадр
            log.warning("маска не применилась к кадру %s: %s", fr.id, exc)
    context = stage_context(s, site)
    with registry.call_lock("classifier", name):
        result = classifier.assess(image, info, keys=None, context=context)
    s.execute(delete(StageObservation).where(StageObservation.frame_id == fr.id,
                                             StageObservation.provider == name))
    s.add(adapters.observation_row(result, fr.id, name))
    fr.stage_used = True
    cs = s.scalar(select(CameraState).where(CameraState.camera_id == cam.id))
    if cs is not None:
        cs.stage_mask_ratio = cs.masked_ratio


# --------------------------------------------------------------------------
# кадр целиком
# --------------------------------------------------------------------------

def claim(frame_id: int) -> bool:
    """Атомарно взять кадр в работу: защищает от двойной обработки, если кадры
    подбирает и сервер, и утилита из соседнего процесса."""
    with db.session() as s:
        res = s.execute(Frame.__table__.update()
                        .where(Frame.id == frame_id, Frame.status.in_(("pending", "postponed")))
                        .values(status="processing"))
        s.commit()
        return res.rowcount == 1


def process_frame(frame_id: int) -> str:
    """Обработать кадр. Возвращает итоговый статус. Исключения наружу — только
    если кадр вообще не удалось прочитать (очередь пометит его ошибкой)."""
    with db.session() as s:
        fr = s.get(Frame, frame_id)
        if fr is None:
            return "missing"
        cam = s.get(Camera, fr.camera_id)
        site = s.get(Site, cam.site_id) if cam else None
        if cam is None or site is None:
            return "missing"
        state = settings_svc.get_state(s)
        thresholds = state["thresholds"]

        img = storage.get().get(fr.key)
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("файл кадра не читается как изображение")

        notes: list[str] = []
        errors: list[str] = []
        postponed = False
        try:
            quality = assess_quality(img, fr.captured_at)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"оценка качества: {type(exc).__name__}: {exc}")
            quality = basic_quality(img)
        adapters.apply_quality(fr, quality)
        info = adapters.frame_info(fr, cam)
        s.commit()

        mask, mask_changed = None, False
        if quality.usable_for_stage and (fr.meta or {}).get("mask_done"):
            mask = load_mask(s, cam)        # повторный заход (отложенный кадр): маска этот кадр уже видела
        elif quality.usable_for_stage:
            try:
                mask, mask_changed = _update_mask(s, cam, img, fr.captured_at, thresholds)
                fr.meta = {**(fr.meta or {}), "mask_done": True}
                s.commit()
            except Exception as exc:  # noqa: BLE001
                s.rollback()
                errors.append(f"маска: {type(exc).__name__}: {exc}")

        if not fr.processed_a:
            try:
                _run_model_a(s, fr, cam, site, img, info, state)
                fr.processed_a = True
                s.commit()
            except ProviderUnavailable as exc:
                s.rollback()
                postponed = True
                notes.append(f"модель А ({state['model_a']}) отложена: {exc.reason}")
            except Exception as exc:  # noqa: BLE001
                s.rollback()
                log.exception("модель А упала на кадре %s", frame_id)
                errors.append(f"модель А: {type(exc).__name__}: {exc}")

        if not fr.processed_b:
            if not quality.usable_for_stage:
                fr.processed_b = True
                why = quality.reject_reason or ("ночной кадр" if quality.is_night else "кадр не годен")
                notes.append(f"модель Б пропустила кадр: {why}")
            elif not _stage_due(s, fr, cam, thresholds, mask_changed):
                fr.processed_b = True
            else:
                try:
                    _run_model_b(s, fr, cam, site, img, info, state, mask)
                    fr.processed_b = True
                    s.commit()
                except ProviderUnavailable as exc:
                    s.rollback()
                    postponed = True
                    notes.append(f"модель Б ({state['model_b']}) отложена: {exc.reason}")
                except Exception as exc:  # noqa: BLE001
                    s.rollback()
                    log.exception("модель Б упала на кадре %s", frame_id)
                    errors.append(f"модель Б: {type(exc).__name__}: {exc}")

        fr = s.get(Frame, frame_id)
        fr.status = "error" if errors else ("postponed" if postponed else "done")
        fr.note = "; ".join(errors + notes)[:2000]
        site_id = site.id
        s.commit()
        status = fr.status

    from app.services.queue import recomputer
    recomputer.request(site_id)
    return status


# --------------------------------------------------------------------------
# пересчёт площадки
# --------------------------------------------------------------------------

def site_now(s: Session, site: Site, thresholds: dict) -> dt.datetime:
    """«Сейчас» аналитики. Для живой площадки — часы, для архива (загрузили
    снимки 2006 года) — время последнего кадра, иначе всё «просрочено на 20 лет»."""
    last = s.scalar(select(func.max(Frame.captured_at)).join(Camera, Camera.id == Frame.camera_id)
                    .where(Camera.site_id == site.id))
    wall = dt.datetime.now(dt.UTC)
    mode = thresholds["pipeline"].get("clock", "auto")
    if last is None or mode == "wall":
        return wall
    if mode == "last_frame":
        return last
    gap = dt.timedelta(days=float(thresholds["pipeline"].get("live_gap_days", 3.0)))
    return wall if wall - last <= gap else last


def _empty_timeline(manual: dict[int, c.StageState]) -> c.StageTimeline:
    current = max((k for k, v in manual.items() if v.status != c.StageStatus.NOT_STARTED), default=None)
    return c.StageTimeline(states=dict(manual), current_stage=current, overall_progress=0.0, daily_front=[])


def _write_stage_states(s: Session, site_id: int, timeline: c.StageTimeline) -> None:
    rows = {r.stage_id: r for r in s.scalars(select(StageState).where(StageState.site_id == site_id))}
    for stage_id, st in (timeline.states or {}).items():
        row = rows.get(int(stage_id))
        if row is not None and row.manual:
            continue            # ручная отметка пользователя важнее модели
        if row is None:
            row = StageState(site_id=site_id, stage_id=int(stage_id))
            s.add(row)
        adapters.apply_stage_state(row, st)
        row.manual = False


def _upsert_deviations(s: Session, site_id: int, records: list[c.DeviationRecord], now: dt.datetime) -> None:
    existing = {d.key: d for d in s.scalars(select(Deviation).where(Deviation.site_id == site_id))}
    seen: set[str] = set()
    for rec in records:
        if rec.key in seen:
            continue
        seen.add(rec.key)
        row = existing.get(rec.key)
        if row is None:
            row = Deviation(site_id=site_id, key=rec.key, status="open", started_at=rec.started_at or now)
            s.add(row)
        elif row.status == "resolved":
            row.status = "open"             # вернулось — снова требует внимания
            row.started_at = rec.started_at or now
        adapters.apply_deviation(row, rec)
        row.last_seen_at = rec.last_seen_at or now
        row.updated_at = utcnow()
    for key, row in existing.items():
        if key not in seen and row.status in ("open", "ack"):
            row.status = "resolved"
            row.data = {**(row.data or {}), "resolved_at": now.isoformat(), "auto_resolved": True}
            row.updated_at = utcnow()


def _fallback_report(plan: list[c.PlanItem], timeline: c.StageTimeline) -> c.SiteReport:
    verdict = c.Verdict.NO_PLAN if not plan else c.Verdict.NO_DATA
    return c.SiteReport(verdict=verdict, lag_days=None, expected_progress=None,
                        actual_progress=timeline.overall_progress, forecast_finish=None,
                        current_stage=timeline.current_stage, stage_states=timeline.states,
                        hours=[], deviations=[],
                        explanation=["Модуль аналитики не подключён — вердикт не рассчитан."])


def _step(errors: list[str], what: str, fn, fallback):
    """Шаг пересчёта: чужой модуль может отсутствовать или упасть — пересчёт
    всё равно доводим до конца и честно пишем, чего не хватило."""
    try:
        return fn()
    except ImportError as exc:
        errors.append(f"{what}: модуль не подключён ({exc})")
    except Exception as exc:  # noqa: BLE001
        log.exception("пересчёт: %s", what)
        errors.append(f"{what}: {type(exc).__name__}: {exc}")
    return fallback() if callable(fallback) else fallback


def compute_plan_hours(s: Session, site: Site, thresholds: dict | None = None) -> list[str]:
    """Плановые моточасы для строк плана без ручных часов (§5.6)."""
    thresholds = thresholds or settings_svc.get_state(s)["thresholds"]
    rows = s.scalars(select(PlanItem).where(PlanItem.site_id == site.id)).all()
    if not rows:
        return []
    fleet = {f.cls: f.count for f in s.scalars(select(SiteFleet).where(SiteFleet.site_id == site.id))}
    try:
        hours = providers.module("core.equipment.hours")
    except ImportError as exc:
        return [f"плановые часы не рассчитаны: модуль модели А не подключён ({exc})"]
    items = [adapters.plan_item(r) for r in rows]
    table = hours.planned_hours(items, fleet, shift_hours=float(site.shift_hours or 10.0),
                                utilization=float(thresholds["analytics"].get("utilization", 0.7)))
    for r in rows:
        if not r.hours_manual:
            r.planned_hours = {k: round(float(v), 1) for k, v in (table.get(r.stage_id) or {}).items()}
    return []


def recompute_site(site_id: int) -> dict | None:
    """Пересчитать этапы, часы, отклонения и отчёт площадки. Идемпотентно."""
    with _recompute_locks[site_id]:
        with db.session() as s:
            site = s.get(Site, site_id)
            if site is None:
                return None
            state = settings_svc.get_state(s)
            thr = state["thresholds"]
            now = site_now(s, site, thr)
            errors: list[str] = []

            plan = adapters.plan_items(s, site_id)
            manual = {r.stage_id: adapters.stage_state(r) for r in
                      s.scalars(select(StageState).where(StageState.site_id == site_id, StageState.manual.is_(True)))}
            observations = adapters.observations(s, site_id, state["model_b"])
            seq_config = {**thr["stage"], **thr["pipeline"]}

            timeline = _step(errors, "хронология этапов",
                             lambda: providers.module("core.stage.sequence").infer(
                                 observations, manual=manual, config=seq_config),
                             lambda: _empty_timeline(manual))
            for stage_id, st in manual.items():         # ручное всегда в итоговой картине
                timeline.states[stage_id] = st
            _write_stage_states(s, site_id, timeline)
            s.flush()

            intervals = adapters.intervals(s, site_id)
            balances = _step(errors, "моточасы",
                             lambda: providers.module("core.equipment.hours").balances(plan, intervals), [])
            ctx = None
            ctx_mod = providers.optional_module("core.analytics.context")
            if ctx_mod is not None:
                ctx = ctx_mod.AnalyticsContext(
                    site_id=site_id, now=now, plan=plan, timeline=timeline,
                    units=adapters.units(s, site_id),
                    recent=adapters.recent(s, site_id, now, float(thr["pipeline"]["recent_window_h"]),
                                           state["model_a"]),
                    intervals=intervals, balances=balances, zones=adapters.zones_for_site(s, site_id),
                    config={**thr["analytics"], **thr["stage"], "shift_hours": site.shift_hours,
                            "floors_total": site.floors_total, "object_type": site.object_type},
                )
            else:
                errors.append("аналитика: модуль core.analytics не подключён")

            if ctx is not None:
                records = _step(errors, "правила отклонений",
                                lambda: providers.module("core.analytics.rules").evaluate(ctx), None)
                if records is not None:
                    _upsert_deviations(s, site_id, records, now)
                report = _step(errors, "отчёт", lambda: providers.module("core.analytics.report").build(ctx),
                               lambda: _fallback_report(plan, timeline))
            else:
                report = _fallback_report(plan, timeline)
            plan_fact = _step(errors, "план/факт",
                              lambda: providers.module("core.analytics.timeline").plan_vs_fact(
                                  plan, timeline, now.date()), None)

            series = {"days": [], "expected": [], "actual": []}
            if plan_fact is not None and isinstance(getattr(plan_fact, "series", None), dict):
                series.update(adapters.jsonable(plan_fact.series))
            # jsonable целиком: реализации ядра могут вернуть numpy-числа, JSON-колонка их не примет
            site.report = adapters.jsonable({
                "verdict": adapters.jsonable(report.verdict),
                "lag_days": adapters.jsonable(report.lag_days),
                "expected_progress": adapters.jsonable(report.expected_progress),
                "actual_progress": adapters.jsonable(report.actual_progress),
                "forecast_finish": adapters.jsonable(report.forecast_finish),
                "current_stage": report.current_stage if report.current_stage is not None else timeline.current_stage,
                "overall_progress": adapters.jsonable(timeline.overall_progress),
                "explanation": [str(x) for x in (report.explanation or [])],
                "needs_review": adapters.jsonable(list(timeline.needs_review or [])),
                "rejected_outliers": adapters.jsonable(list(timeline.rejected_outliers or [])),
                "daily_front": adapters.jsonable(list(timeline.daily_front or []))[-400:],
                "balances": [{"stage_id": b.stage_id, "cls": b.cls, "planned_hours": float(b.planned_hours),
                              "worked_hours": float(b.worked_hours),
                              "last_worked_at": adapters.iso(b.last_worked_at)} for b in balances],
                "series": series,
                "now": now.isoformat(),
                "errors": errors,
            })
            site.report_at = utcnow()
            s.commit()
            return site.report
