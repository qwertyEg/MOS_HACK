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
import inspect
import logging
import threading
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy import update as sa_update
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


def _params(fn: Any) -> set[str]:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def mask_class() -> Any | None:
    """DynamicMask модели Б (core.stage.mask); None — модуль не подключён."""
    for name in ("core.stage.mask", "core.stage"):
        mod = providers.optional_module(name)
        if mod is not None and hasattr(mod, "DynamicMask"):
            return mod.DynamicMask
    return None


def load_mask(s: Session, cam: Camera) -> Any | None:
    if cam.id in _masks:
        return _masks[cam.id]
    cls = mask_class()
    if cls is None:
        return None
    state = s.scalar(select(CameraState).where(CameraState.camera_id == cam.id))
    if state is not None and state.mask_key:
        try:
            mask = cls.loads(storage.get().get(state.mask_key))
            _masks[cam.id] = mask
            return mask
        except Exception as exc:  # noqa: BLE001 — маска восстановима наблюдением
            log.warning("маска камеры %s не читается (%s) — начинаю заново", cam.id, exc)
    return None


def _update_mask(s: Session, cam: Camera, img: np.ndarray, when: dt.datetime,
                 thresholds: dict, weather: Any = None) -> tuple[Any | None, bool]:
    """Обновить маску годным дневным кадром. → (маска, изменилась ли сильно).

    «Сильно» (внеочередной вызов модели Б, ARCHITECTURE §4): маска только что
    инициализировалась или стёрла клетки (стройка заползла на фон —
    `MaskUpdate.erased_cells`), либо доля маски с прошлого вызова модели Б
    сдвинулась больше порога `stage_mask_change`.
    """
    cls = mask_class()
    if cls is None:
        return None, False
    mask = load_mask(s, cam)
    if mask is None:
        mask = cls.new(img.shape[:2])
    # Погода нужна маске, чтобы снег не «стирал» соседний дом (окно замораживается).
    kw = {"weather": weather} if weather is not None and "weather" in _params(mask.update) else {}
    try:
        upd = mask.update(img, when, **kw)
    except Exception as exc:  # noqa: BLE001 — камеру переставили / сменилось разрешение
        log.warning("маска камеры %s сброшена: %s", cam.id, exc)
        mask = cls.new(img.shape[:2])
        upd = mask.update(img, when, **kw)
    _masks[cam.id] = mask

    state = s.scalar(select(CameraState).where(CameraState.camera_id == cam.id))
    if state is None:
        state = CameraState(camera_id=cam.id)
        s.add(state)
    state.mask_key = storage.get().put(_mask_key(cam.id), mask.dumps(), "application/octet-stream")
    state.windows = (state.windows or 0) + 1
    state.masked_ratio = float(mask.masked_ratio)
    state.retained = float(getattr(mask, "retained", 1.0) or 0.0)
    state.updated_at = utcnow()
    changed = bool(getattr(upd, "initialized_now", False) or (getattr(upd, "erased_cells", 0) or 0) > 0)
    if state.stage_mask_ratio is not None and \
            abs(state.masked_ratio - state.stage_mask_ratio) >= thresholds["pipeline"]["stage_mask_change"]:
        changed = True
    return mask, changed


# --------------------------------------------------------------------------
# модель А
# --------------------------------------------------------------------------

def _engine(s: Session, site: Site, thresholds: dict, provider: str) -> Any:
    engine = _engines.get(site.id)
    if engine is not None:
        return engine
    eq = providers.module("core.equipment")
    values = dict(thresholds.get("equipment") or {})
    # День плана, на который списываются моточасы, — по часовому поясу площадки;
    # смена и коэффициент использования — из карточки объекта и порогов аналитики.
    if site.timezone:
        values["timezone"] = site.timezone
    if site.shift_hours:
        values["shift_hours"] = float(site.shift_hours)
    if "utilization" in (thresholds.get("analytics") or {}):
        values["utilization"] = float(thresholds["analytics"]["utilization"])
    known = set(getattr(eq.EquipmentConfig, "__dataclass_fields__", {}) or values)
    config = eq.EquipmentConfig.from_dict({k: v for k, v in values.items() if k in known})
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

    # Склейка дубля (EquipmentUpdate.merged: {старый unit_id: новый}): единица,
    # родившаяся на стыке камер, оказалась уже известной машиной — её детекции и
    # моточасы переходят к настоящей единице, запись дубля удаляется.
    for old_uid, new_uid in (getattr(update, "merged", None) or {}).items():
        old = s.scalar(select(EquipmentUnit).where(EquipmentUnit.site_id == site.id,
                                                   EquipmentUnit.uid == old_uid))
        new_id = unit_row_id(new_uid)
        if old is None or new_id is None or old.id == new_id:
            continue
        s.execute(sa_update(Detection).where(Detection.unit_id == old.id).values(unit_id=new_id))
        s.execute(sa_update(ActivityInterval).where(ActivityInterval.unit_id == old.id).values(unit_id=new_id))
        s.delete(old)
        s.flush()

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
                 info: c.FrameInfo, state: dict) -> list[str]:
    """→ события кадра от движка («камера сдвинулась…») — в примечание кадра."""
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
    # «Камера не откалибрована» движок пишет на каждом кадре — UI и так показывает
    # это у камеры, в примечании кадра оставляем только события кадра.
    return [str(n) for n in (getattr(update, "notes", None) or []) if "не откалибрована" not in str(n)]


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


def _context_text(s: Session, site: Site, before: dt.datetime, front: int | None,
                  rows: dict[int, StageState]) -> str:
    """Текст контекста стройки для VLM — порт ContextBuilder.text() Никиты
    (api-solution/core/site.py): достигнутый этап, виденная техника, «стройка не
    идёт назад». Пустая строка, если модель Б этот объект ещё не видела."""
    n, last = s.execute(
        select(func.count(StageObservation.id), func.max(Frame.captured_at))
        .join(Frame, Frame.id == StageObservation.frame_id).join(Camera, Camera.id == Frame.camera_id)
        .where(Camera.site_id == site.id, Frame.captured_at < before)).one()
    if not n:
        return ""
    tz = adapters.site_tz(site)
    last = last if last.tzinfo else last.replace(tzinfo=dt.UTC)
    lines = [f"Контекст этой стройки по {n} предыдущим снимкам (последний — {last.astimezone(tz):%d.%m.%Y}):"]
    if front:
        since = rows.get(front).actual_start if rows.get(front) else None
        done = f"; этапы 1–{front - 1} выполнены" if front > 1 else ""
        lines.append(f"- достигнутый этап: {front} «{taxonomy.stage_name(front)}»"
                     f"{f' (с {since:%d.%m.%Y})' if since else ''}{done}.")
    seen = s.execute(select(EquipmentUnit.cls, func.count(EquipmentUnit.id))
                     .where(EquipmentUnit.site_id == site.id).group_by(EquipmentUnit.cls)).all()
    if seen:
        eq = ", ".join(f"{taxonomy.equipment_name(cls)} (до {cnt})" for cls, cnt in sorted(seen))
        lines.append(f"- техника, которую уже видели на площадке: {eq}.")
    lines.append("Стройка не идёт назад. Используй контекст, чтобы не противоречить истории, "
                 "но отвечай только по тому, что видно на этом снимке. Если снимок явно противоречит "
                 "контексту, коротко опиши это в поле context_conflict.")
    return "\n".join(lines)


def stage_context(s: Session, site: Site, before: dt.datetime | None = None,
                  mask: Any | None = None) -> dict[str, Any]:
    """Что уже известно о стройке — context классификатора модели Б.

    Ключи, которые читают реализации core.stage (build_model-b): `mask` — маска
    камеры (DynamicMask; классификатор сам гасит фон и говорит VLM, что тёмное —
    фон), `front` — текущий этап по хронологии, `text` — контекст стройки для
    GLM (Никита). Остальное — для отладки и фейков в тестах.
    """
    rows = {r.stage_id: r for r in s.scalars(select(StageState).where(StageState.site_id == site.id))}
    states = {k: r.status for k, r in rows.items()}
    report = site.report or {}
    front = report.get("current_stage")
    front = int(front) if isinstance(front, (int, float)) else None
    ctx: dict[str, Any] = {
        "site_id": site.id, "object_type": site.object_type, "floors_total": site.floors_total,
        "current_stage": front, "stage_states": states,
        "done_stages": sorted(k for k, v in states.items() if v == "done"),
        "front": front,
    }
    if mask is not None:
        ctx["mask"] = mask
    try:
        ctx["text"] = _context_text(s, site, before or dt.datetime.now(dt.UTC), front, rows)
    except Exception as exc:  # noqa: BLE001 — без контекста разбор хуже, но возможен
        log.warning("контекст стройки %s не собран: %s", site.id, exc)
    return ctx


def _run_model_b(s: Session, fr: Frame, cam: Camera, site: Site, img: np.ndarray,
                 info: c.FrameInfo, state: dict, mask: Any | None) -> None:
    name = state["model_b"]
    classifier = registry.require("classifier", name)
    # Маску не накладываем здесь: классификатор делает это сам (core.stage.mask.
    # masked_for_model) и знает, применилась ли она, — GLM тогда получает пояснение.
    context = stage_context(s, site, before=fr.captured_at, mask=mask)
    with registry.call_lock("classifier", name):
        result = classifier.assess(img, info, keys=None, context=context)
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
                mask, mask_changed = _update_mask(s, cam, img, fr.captured_at, thresholds, quality.weather)
                fr.meta = {**(fr.meta or {}), "mask_done": True}
                s.commit()
            except Exception as exc:  # noqa: BLE001
                s.rollback()
                errors.append(f"маска: {type(exc).__name__}: {exc}")

        if not fr.processed_a:
            try:
                notes.extend(_run_model_a(s, fr, cam, site, img, info, state))
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


def _tz_offset_hours(site: Site, now: dt.datetime) -> float:
    """Смещение пояса площадки на момент `now` — границы суток хронологии модели Б."""
    off = now.astimezone(adapters.site_tz(site)).utcoffset()
    return off.total_seconds() / 3600 if off else 0.0


def _detectable(model_a: str) -> list[str] | None:
    """Классы, которые умеет текущий детектор: типы вне списка не порождают
    «нет техники этапа» (у YOLO нет башенного крана сверху и асфальтоукладчика)."""
    try:
        registry.get("detector", model_a)      # дёшево: веса грузятся при первом detect()
    except ProviderUnavailable:
        return None
    return sorted(c["key"] for c in settings_svc.classes(model_a) if c["supported"])


def _sequence_config(thr: dict, site: Site, now: dt.datetime) -> dict:
    """Пороги UI → SequenceConfig модели Б (имена в UI исторические)."""
    cfg = {**thr["stage"], **thr["pipeline"]}
    if "unsure_review_ratio" in thr["stage"]:
        cfg["needs_review_ratio"] = float(thr["stage"]["unsure_review_ratio"])
    cfg["tz_offset_hours"] = _tz_offset_hours(site, now)
    return cfg


def _analytics_config(s: Session, thr: dict, site: Site, model_a: str) -> dict:
    """Пороги UI + то, что правила аналитики хотят знать о площадке (build_plan-analytics):
    пояс, что умеет детектор, камеры с типом (папка/видео не «молчат»)."""
    cfg = {**thr["analytics"], **thr["stage"], "shift_hours": site.shift_hours,
           "floors_total": site.floors_total, "object_type": site.object_type,
           "timezone": site.timezone or "Europe/Moscow"}
    if "on_track_days" in thr["analytics"]:
        cfg["schedule_tolerance_days"] = float(thr["analytics"]["on_track_days"])
    detectable = _detectable(model_a)
    if detectable is not None:
        cfg["detectable"] = detectable
    cams = s.scalars(select(Camera).where(Camera.site_id == site.id).order_by(Camera.id)).all()
    cfg["cameras"] = [{"id": cam.id, "name": cam.name, "kind": cam.kind,
                       "last_frame_at": adapters.iso(cam.last_frame_at)} for cam in cams]
    cfg["camera_names"] = {str(cam.id): cam.name for cam in cams}
    return cfg


def _balances(hours_mod: Any, plan: list[c.PlanItem], intervals: list[c.ActivityInterval], site: Site) -> list:
    if "tz" in _params(hours_mod.balances):
        return hours_mod.balances(plan, intervals, tz=site.timezone or "Europe/Moscow")
    return hours_mod.balances(plan, intervals)


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
            seq_config = _sequence_config(thr, site, now)

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
                             lambda: _balances(providers.module("core.equipment.hours"), plan, intervals, site), [])
            ctx = None
            ctx_mod = providers.optional_module("core.analytics.context")
            if ctx_mod is not None:
                ctx = ctx_mod.AnalyticsContext(
                    site_id=site_id, now=now, plan=plan, timeline=timeline,
                    units=adapters.units(s, site_id),
                    recent=adapters.recent(s, site_id, now, float(thr["pipeline"]["recent_window_h"]),
                                           state["model_a"]),
                    intervals=intervals, balances=balances, zones=adapters.zones_for_site(s, site_id),
                    config={**_analytics_config(s, thr, site, state["model_a"]),
                            "stage_observations": len(observations)},
                )
            else:
                errors.append("аналитика: модуль core.analytics не подключён")

            plan_fact = None
            if ctx is not None:
                report_mod = providers.optional_module("core.analytics.report")
                if report_mod is not None and hasattr(report_mod, "build_full"):
                    # Один проход: build_full сам вызывает rules.evaluate и plan_vs_fact
                    # с темпом в активных днях (по журналу моточасов модели А).
                    built = _step(errors, "отчёт", lambda: report_mod.build_full(ctx), None)
                    if built is not None:
                        report, plan_fact = built
                        _upsert_deviations(s, site_id, list(report.deviations or []), now)
                    else:
                        report = _fallback_report(plan, timeline)
                else:
                    records = _step(errors, "правила отклонений",
                                    lambda: providers.module("core.analytics.rules").evaluate(ctx), None)
                    if records is not None:
                        _upsert_deviations(s, site_id, records, now)
                    report = _step(errors, "отчёт", lambda: providers.module("core.analytics.report").build(ctx),
                                   lambda: _fallback_report(plan, timeline))
            else:
                report = _fallback_report(plan, timeline)
            if plan_fact is None:
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
                "planned_finish": adapters.jsonable(getattr(plan_fact, "planned_finish", None)),
                "delay_days": adapters.jsonable(getattr(plan_fact, "delay_days", None)),
                "forecast_note": str(getattr(plan_fact, "forecast_note", "") or ""),
                "stage_obs": len(observations),
                "now": now.isoformat(),
                "errors": errors,
            })
            site.report_at = utcnow()
            s.commit()
            return site.report
