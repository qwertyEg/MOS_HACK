"""Ручная разметка техники (требование 3): правки оператора поверх модели А.

Что может оператор (страница кадра и камеры — режим «Разметка», вкладка «Техника»):
  * сменить класс рамки — только этой или всей машины;
  * удалить ложную рамку — на этом кадре или «это место камеры — не техника»
    (мачта связи, контейнер, фонарный столб больше не заводят «машину»);
  * дорисовать пропущенную рамку;
  * подтвердить, что разметка кадра верна (кадр идёт в датасет как проверенный);
  * склеить две «разные» машины в одну, разделить машину с кадра, сменить тип
    машины, пометить машину «это не техника»;
  * выгрузить размеченное как датасет YOLO для дообучения детектора.

Как устроено. Правка — не изменение строк `detections` (их переписывает каждый
переанализ), а отдельный факт в таблице `annotations`: «рамка ≈(x, y, w, h) на
кадре F — автокран / не техника / машина m3f2a…». Конвейер модели А
накладывает факты поверх ответа детектора ДО трекера и движка (`apply`), поэтому
  * переанализ правки не стирает — он применяет их заново;
  * единицы, моточасы и отклонения считаются уже по исправленной картине;
  * отмена = удаление строк правки и перепрогон.
Рамка правки узнаётся среди рамок детектора по IoU ≥ 0.5: после переанализа тем
же детектором рамки совпадают до пикселя, у другого провайдера — приблизительно.

Перепрогон после правки (`pipeline.replay_model_a`) детектор не запускает: ответ
детектора на каждый кадр хранится в `raw_detections`. Кадр, на котором сделана
правка, обновляется на месте сразу (`refresh_frame`) — оператор видит результат
клика, не дожидаясь перепрогона большого архива.

Склейка / разделение / смена типа — это рамки с ключом ручной машины
(`unit_key`, он же uid её единицы, начинается с «m»): движок модели А кладёт
рамки с одним ключом в одну единицу, закрепляет её класс и сам такие единицы не
склеивает и не разделяет (core/equipment/engine.py, MANUAL_UNIT).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import uuid
import zlib
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app import storage
from app.models import Annotation, Camera, Detection, EquipmentUnit, Frame, RawDetections, Site, utcnow
from app.services import adapters, providers
from app.services import settings as settings_svc
from core import contracts as c
from core import taxonomy

log = logging.getLogger(__name__)

MATCH_IOU = 0.5        # рамка правки ↔ рамка детектора на том же кадре
PLACE_IOU = 0.6        # «место камеры — не техника»: строже, машина, вставшая рядом, пропасть не должна
SAME_BOX_IOU = 0.7     # обновление кадра на месте: та же рамка, что уже записана
MIN_SIDE_PX = 8        # дорисованная рамка меньше — промах мышью
SYNC_REPLAY_FRAMES = 400   # объект меньше — перепрогон прямо в запросе: оператор сразу видит итог склейки

# Поля Detection.extra, которые пишет движок и правки поверх ответа детектора. Имена
# совпадают с core/equipment/engine.py; не импортируем оттуда — в тестах бэкенда
# модель А подменена фейком, а смысл полей — договор между модулями.
MANUAL = "manual"
MANUAL_CLS = "manual_cls"
MANUAL_UNIT = "manual_unit"
MANUAL_UNIT_CLS = "manual_unit_cls"
MANUAL_KEYS = (MANUAL, MANUAL_CLS, MANUAL_UNIT, MANUAL_UNIT_CLS)
_DERIVED = ("unit_label", "unit_status", "unit_uid", "raw_cls", "gap_min", "box_flicker", *MANUAL_KEYS)
MANUAL_PREFIX = "m"

KINDS = {
    "box_relabel": "Класс рамки",
    "box_delete": "Удалена рамка",
    "box_add": "Дорисована рамка",
    "frame_verify": "Кадр проверен",
    "unit_merge": "Склейка машин",
    "unit_split": "Разделение машины",
    "unit_relabel": "Тип машины",
    "unit_delete": "Не техника",
}
# Действия, после которых кадр считается просмотренным человеком (датасет «проверенное»).
REVIEW_KINDS = ("box_relabel", "box_delete", "box_add", "frame_verify")
UNIT_KINDS = ("unit_merge", "unit_split", "unit_relabel", "unit_delete")


def is_manual(uid: str | None) -> bool:
    return bool(uid) and str(uid).startswith(MANUAL_PREFIX)


def new_batch() -> str:
    return uuid.uuid4().hex[:12]


def new_key() -> str:
    return MANUAL_PREFIX + uuid.uuid4().hex[:9]


# --------------------------------------------------------------------------
# геометрия
# --------------------------------------------------------------------------

def iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = max(0.0, a[2]) * max(0.0, a[3]) + max(0.0, b[2]) * max(0.0, b[3]) - inter
    return inter / union if union > 0 else 0.0


def _box(r: Annotation) -> tuple[float, float, float, float]:
    return float(r.x), float(r.y), float(r.w), float(r.h)


def _best(boxes: list[c.Detection], target, thr: float) -> int | None:
    best, j = thr, None
    for i, b in enumerate(boxes):
        v = iou(b.bbox, target)
        if v >= best:
            best, j = v, i
    return j


# --------------------------------------------------------------------------
# ответ детектора (raw_detections)
# --------------------------------------------------------------------------

def raw_item(d: c.Detection) -> dict[str, Any]:
    return {"cls": d.cls, "conf": round(float(d.conf), 4), "bbox": [round(float(v), 2) for v in d.bbox],
            "source": d.source, "extra": adapters.jsonable(dict(d.extra or {}))}


def raw_detection(item: dict) -> c.Detection:
    return c.Detection(cls=str(item["cls"]), conf=float(item.get("conf") or 0.0),
                       bbox=tuple(float(v) for v in item["bbox"]), source=str(item.get("source") or "local"),
                       extra=dict(item.get("extra") or {}))


def save_raw(s: Session, frame_id: int, provider: str, dets: list[c.Detection]) -> None:
    items = [raw_item(d) for d in dets]
    row = s.scalar(select(RawDetections).where(RawDetections.frame_id == frame_id,
                                               RawDetections.provider == provider))
    if row is None:
        s.add(RawDetections(frame_id=frame_id, provider=provider, items=items))
    else:
        row.items = items
        row.created_at = utcnow()


def raw_from_rows(rows: list[Detection]) -> list[c.Detection]:
    """Ответ детектора, восстановленный по записанным рамкам (кадры, обработанные до
    появления raw_detections): класс — до склейки с единицей и до правок."""
    known = taxonomy.equipment()
    out = []
    for r in rows:
        extra = dict(r.extra or {})
        man = extra.get(MANUAL) or {}
        if man.get("added"):
            continue
        engine_raw = extra.get("raw_cls") if extra.get("raw_cls") in known else None
        cls = man.get("orig_cls") or engine_raw or r.cls
        conf = float(man.get("orig_conf", r.conf))
        clean = {k: v for k, v in extra.items() if k not in _DERIVED and k != "source"}
        out.append(c.Detection(cls=cls, conf=conf, bbox=(r.x, r.y, r.w, r.h),
                               source=str(extra.get("source") or r.provider), extra=clean))
    return out


def raw_for_frames(s: Session, frame_ids: list[int], preferred: str | None
                   ) -> dict[int, tuple[list[c.Detection], str]]:
    """Ответ детектора по кадрам: текущего провайдера, если он размечал кадр, иначе последнего."""
    out: dict[int, tuple[list[c.Detection], str]] = {}
    for i in range(0, len(frame_ids), 500):
        chunk = frame_ids[i:i + 500]
        by_frame: dict[int, list[RawDetections]] = defaultdict(list)
        for r in s.scalars(select(RawDetections).where(RawDetections.frame_id.in_(chunk))):
            by_frame[r.frame_id].append(r)
        for fid, rows in by_frame.items():
            row = next((r for r in rows if r.provider == preferred), None) or max(rows, key=lambda r: r.id)
            out[fid] = ([raw_detection(x) for x in row.items or []], row.provider)
    return out


def ensure_raw(s: Session, site_id: int, preferred: str) -> int:
    """Кадрам, разобранным до появления raw_detections, — ответ детектора из записанных
    рамок. Делается до первой правки: потом по рамкам его уже не восстановить."""
    cams = select(Camera.id).where(Camera.site_id == site_id)
    has_raw = select(RawDetections.id).where(RawDetections.frame_id == Frame.id).exists()
    missing = list(s.scalars(select(Frame.id).where(Frame.camera_id.in_(cams), Frame.processed_a.is_(True),
                                                    ~has_raw)))
    for i in range(0, len(missing), 500):
        chunk = missing[i:i + 500]
        dets = adapters.detections_by_frame(s, chunk, preferred)
        for fid in chunk:
            rows = dets.get(fid, [])
            s.add(RawDetections(frame_id=fid, provider=rows[0].provider if rows else preferred,
                                items=[raw_item(d) for d in raw_from_rows(rows)]))
    if missing:
        s.flush()
    return len(missing)


# --------------------------------------------------------------------------
# наложение правок на ответ детектора
# --------------------------------------------------------------------------

def key_classes(s: Session, site_id: int) -> dict[str, str]:
    """Ручная машина → её класс по последней правке (склейка, смена типа)."""
    out: dict[str, str] = {}
    for key, cls in s.execute(select(Annotation.unit_key, Annotation.cls)
                              .where(Annotation.site_id == site_id, Annotation.action == "unit",
                                     Annotation.unit_key.is_not(None)).order_by(Annotation.id)).all():
        if cls:
            out[key] = cls
    return out


def places(s: Session, camera_id: int) -> list[Annotation]:
    """«Это место камеры — не техника»: действует на все кадры камеры, прошлые и будущие."""
    return list(s.scalars(select(Annotation).where(Annotation.camera_id == camera_id, Annotation.action == "delete",
                                                   Annotation.scope == "camera").order_by(Annotation.id)))


def frame_rules(s: Session, frame_ids: list[int]) -> dict[int, list[Annotation]]:
    out: dict[int, list[Annotation]] = defaultdict(list)
    for i in range(0, len(frame_ids), 500):
        for r in s.scalars(select(Annotation).where(Annotation.frame_id.in_(frame_ids[i:i + 500]),
                                                    Annotation.scope == "frame").order_by(Annotation.id)):
            out[r.frame_id].append(r)
    return out


def apply(dets: list[c.Detection], rows: list[Annotation], place_rows: list[Annotation] = (),
          key_cls: dict[str, str] | None = None) -> list[c.Detection]:
    """Ответ детектора + правки кадра → рамки для трекера и движка.

    Правки применяются по порядку: последняя правка поля побеждает (сменили класс
    рамки, потом склеили её машину с другой — класс машины из склейки). Затронутые
    рамки несут extra["manual"] (что было у детектора, какие правки) — UI рисует
    на них «вручную», датасет помечает источник."""
    key_cls = key_cls or {}
    boxes = [dataclasses.replace(d, extra=dict(d.extra or {})) for d in dets]
    state: list[dict[str, Any]] = [{} for _ in boxes]
    ordered = sorted(rows, key=lambda r: r.id)
    for r in ordered:
        if r.action == "add" and r.cls and r.x is not None:
            boxes.append(c.Detection(cls=r.cls, conf=1.0, bbox=_box(r), source="manual", extra={}))
            state.append({"added": r.id})
    for r in ordered:
        if r.action not in ("relabel", "delete", "unit") or r.x is None or r.scope != "frame":
            continue
        j = _best(boxes, _box(r), MATCH_IOU)
        if j is None:
            continue
        st = state[j]
        st.setdefault("ids", []).append(r.id)
        if r.action == "relabel" and r.cls:
            st["cls"] = (r.cls, r.id)
        elif r.action == "delete":
            st["deleted"] = r.id
        elif r.action == "unit" and r.unit_key:
            st["key"] = (r.unit_key, r.id, r.cls)
    for p in place_rows:
        pb = _box(p)
        for j, b in enumerate(boxes):
            if "added" not in state[j] and iou(b.bbox, pb) >= PLACE_IOU:
                state[j]["deleted"] = p.id

    out = []
    for b, st in zip(boxes, state):
        if not st:
            out.append(b)
            continue
        if st.get("deleted"):
            continue
        man: dict[str, Any] = {"ids": list(st.get("ids", []))}
        if "added" in st:
            man["added"] = True
            man["ids"].insert(0, st["added"])
        else:
            man["orig_cls"] = b.cls
            man["orig_conf"] = round(float(b.conf), 3)
        key, box_cls = st.get("key"), st.get("cls")
        if key:
            kc = key_cls.get(key[0]) or key[2]
            b.extra[MANUAL_UNIT] = key[0]
            man["unit"] = key[0]
            if kc:
                b.extra[MANUAL_UNIT_CLS] = kc
                b.cls = kc
        if box_cls and (not key or box_cls[1] > key[1]):
            b.cls, b.conf = box_cls[0], 1.0
            b.extra[MANUAL_CLS] = box_cls[0]
            b.extra.pop("alt", None)
            man["cls"] = box_cls[0]
        elif "added" in st:
            b.extra[MANUAL_CLS] = b.cls
            man["cls"] = b.cls
        b.extra[MANUAL] = man
        out.append(b)
    return out


def prepare(s: Session, fr: Frame, site_id: int, provider: str, detections: list[c.Detection]) -> list[c.Detection]:
    """Конвейер модели А: запомнить ответ детектора и наложить на него правки оператора."""
    save_raw(s, fr.id, provider, detections)
    rows = frame_rules(s, [fr.id]).get(fr.id, [])
    place_rows = places(s, fr.camera_id)
    if not rows and not place_rows:
        return list(detections)
    kc = key_classes(s, site_id) if any(r.action == "unit" for r in rows) else {}
    return apply(detections, rows, place_rows, kc)


def _clean(s: Session, dets: list[c.Detection], fr: Frame) -> list[c.Detection]:
    """Та же чистка кадра, что делает движок (одна машина — одна рамка, отсев мелочи)."""
    pp = providers.optional_module("core.equipment.postprocess")
    if pp is None or not hasattr(pp, "clean"):
        return dets
    try:
        cfg = providers.equipment_config(settings_svc.get_state(s)["thresholds"].get("equipment") or {})
        return pp.clean(dets, fr.width or 0, fr.height or 0, cfg)
    except Exception:  # noqa: BLE001 — чистка лишь приближает кадр к итогу перепрогона
        log.exception("чистка рамок кадра %s", fr.id)
        return dets


def refresh_frame(s: Session, fr: Frame, preferred: str) -> None:
    """Обновить рамки кадра на месте — сразу после правки, до перепрогона.

    Совпавшие рамки сохраняют трек и единицу (их пересчитает перепрогон), новые
    (дорисованные, восстановленные отменой) пишутся без единицы, лишние удаляются."""
    raw = raw_for_frames(s, [fr.id], preferred).get(fr.id)
    if raw is None:
        return
    dets, prov = raw
    site_id = s.scalar(select(Camera.site_id).where(Camera.id == fr.camera_id))
    dets = _clean(s, apply(dets, frame_rules(s, [fr.id]).get(fr.id, []), places(s, fr.camera_id),
                           key_classes(s, site_id)), fr)
    stored = list(s.scalars(select(Detection).where(Detection.frame_id == fr.id, Detection.provider == prov)
                            .order_by(Detection.id)))
    unit_cls = dict(s.execute(select(EquipmentUnit.id, EquipmentUnit.cls).where(
        EquipmentUnit.id.in_([r.unit_id for r in stored if r.unit_id is not None]))).all()) if stored else {}
    pairs = sorted(((iou(d.bbox, (r.x, r.y, r.w, r.h)), i, j) for i, d in enumerate(dets)
                    for j, r in enumerate(stored)), reverse=True)
    used_d, used_r = set(), set()
    for v, i, j in pairs:
        if v < SAME_BOX_IOU:
            break
        if i in used_d or j in used_r:
            continue
        used_d.add(i)
        used_r.add(j)
        d, row = dets[i], stored[j]
        extra = {k: v for k, v in (row.extra or {}).items() if k not in MANUAL_KEYS}
        extra.update({k: adapters.jsonable(d.extra[k]) for k in MANUAL_KEYS if k in d.extra})
        row.extra = extra
        row.conf = float(d.conf)
        row.cls = d.extra.get(MANUAL_CLS) or d.extra.get(MANUAL_UNIT_CLS) or unit_cls.get(row.unit_id) or d.cls
    for i, d in enumerate(dets):
        if i not in used_d:
            s.add(adapters.detection_row(dataclasses.replace(d, activity=c.Activity.UNKNOWN), fr.id, prov, None))
    for j, row in enumerate(stored):
        if j not in used_r:
            s.delete(row)
    s.flush()


# --------------------------------------------------------------------------
# правки
# --------------------------------------------------------------------------

@dataclass
class Change:
    """Итог одного действия оператора — для ответа API и журнала."""
    site_id: int
    batch: str
    kind: str
    note: str
    frame_ids: list[int]
    rows: int
    replay: bool           # нужен перепрогон модели А (всегда, кроме «кадр проверен»)
    sync: bool = False     # перепрогнать сразу в запросе (склейка/разделение — оператор ждёт итог)
    frame_id: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {"batch": self.batch, "kind": self.kind, "kind_name": KINDS.get(self.kind, self.kind),
                "note": self.note, "site_id": self.site_id, "rows": self.rows,
                "frame_ids": self.frame_ids[:50], "frame_id": self.frame_id}


def _check_cls(cls: Any) -> str:
    if not isinstance(cls, str) or cls not in taxonomy.equipment():
        raise ValueError(f"cls: неизвестный тип техники «{cls}» (список — /api/settings → classes)")
    return cls


def _name(cls: str | None) -> str:
    return taxonomy.equipment_name(cls) if cls else "—"


def _when(site: Site, t: dt.datetime) -> str:
    return t.astimezone(adapters.site_tz(site)).strftime("%d.%m.%Y %H:%M")


def _frame_ctx(s: Session, frame_id: Any) -> tuple[Frame, Camera, Site]:
    try:
        fid = int(frame_id)
    except (TypeError, ValueError):
        raise ValueError("frame_id: номер кадра") from None
    fr = s.get(Frame, fid)
    if fr is None:
        raise LookupError(f"кадр {fid} не найден")
    cam = s.get(Camera, fr.camera_id)
    return fr, cam, s.get(Site, cam.site_id)


def _preferred(s: Session) -> str:
    return settings_svc.get_state(s)["model_a"]


def _row(fr: Frame, site_id: int, batch: str, kind: str, action: str, author: str, note: str,
         bbox=None, **kw) -> Annotation:
    x = y = w = h = None
    if bbox is not None:
        x, y, w, h = (float(v) for v in bbox)
    return Annotation(site_id=site_id, camera_id=fr.camera_id, frame_id=fr.id, batch=batch, kind=kind,
                      action=action, author=author or "", note=note, x=x, y=y, w=w, h=h, **kw)


def _orig(det: Detection) -> str:
    return ((det.extra or {}).get(MANUAL) or {}).get("orig_cls") or det.cls


class Stale(Exception):
    """Страница показывает то, чего уже нет: перепрогон пересобрал рамки / машины (→ 409)."""


def _find_detection(s: Session, det_id: Any, body: dict) -> Detection:
    """Рамка по id — или, если перепрогон её уже переписал, по кадру и рамке (UI шлёт frame_id и bbox):
    после правки строки рамок пересоздаются, а открытая страница помнит старые id."""
    try:
        det = s.get(Detection, int(det_id))
    except (TypeError, ValueError):
        raise ValueError("id рамки — число") from None
    fid, bbox = body.get("frame_id"), body.get("bbox")
    if det is not None and (fid is None or det.frame_id == fid):
        return det
    if isinstance(fid, int) and isinstance(bbox, list) and len(bbox) == 4 \
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox):
        rows = list(s.scalars(select(Detection).where(Detection.frame_id == fid)))
        best = max(rows, key=lambda r: iou((r.x, r.y, r.w, r.h), bbox), default=None)
        if best is not None and iou((best.x, best.y, best.w, best.h), bbox) >= SAME_BOX_IOU:
            return best
    raise LookupError(f"рамка {det_id} не найдена — кадр пересчитался, обновите страницу")


def find_unit(s: Session, unit_id: Any, uid: Any = None, site_id: Any = None) -> EquipmentUnit:
    """Машина по id; если UI прислал uid, а строка с этим id уже другая машина (перепрогон
    пересобрал список), — по uid на объекте; не нашлась — Stale, а не правка чужой машины."""
    unit = None
    try:
        unit = s.get(EquipmentUnit, int(unit_id))
    except (TypeError, ValueError):
        raise ValueError("id машины — число") from None
    if uid is None or (unit is not None and unit.uid == uid):
        if unit is None:
            raise LookupError(f"машина {unit_id} не найдена — техника могла пересчитаться, обновите список")
        return unit
    sid = site_id if site_id is not None else (unit.site_id if unit is not None else None)
    alt = s.scalar(select(EquipmentUnit).where(EquipmentUnit.site_id == sid, EquipmentUnit.uid == str(uid))) \
        if sid is not None else None
    if alt is None:
        raise Stale("список техники устарел (техника пересчиталась после правки) — обновите страницу")
    return alt


def patch_detection(s: Session, det_id: Any, body: dict, author: str) -> Change:
    """Правка рамки кадра: {cls, scope: box|unit} — класс; {deleted: true, scope: frame|camera} — удалить.
    frame_id и bbox (необязательно) — найти рамку, если её id уже переписал перепрогон."""
    det = _find_detection(s, det_id, body)
    fr, cam, site = _frame_ctx(s, det.frame_id)
    preferred = _preferred(s)
    box = (det.x, det.y, det.w, det.h)
    batch = new_batch()
    if body.get("deleted"):
        scope = body.get("scope") or "frame"
        if scope not in ("frame", "camera"):
            raise ValueError("scope: frame (только этот кадр) | camera (это место камеры — не техника)")
        what = _name(det.cls)
        note = (f"{_when(site, fr.captured_at)}, «{cam.name}»: удалена рамка «{what}»"
                + (" — место камеры больше не считается техникой" if scope == "camera" else ""))
        ensure_raw(s, site.id, preferred)
        s.add(_row(fr, site.id, batch, "box_delete", "delete", author, note, box, scope=scope,
                   orig_cls=_orig(det), orig_conf=float(det.conf)))
        kind = "box_delete"
    elif "cls" in body:
        cls = _check_cls(body.get("cls"))
        scope = body.get("scope") or "box"
        if scope not in ("box", "unit"):
            raise ValueError("scope: box (только эта рамка) | unit (вся машина)")
        if scope == "unit":
            unit = s.get(EquipmentUnit, det.unit_id) if det.unit_id is not None else None
            if unit is None:
                raise ValueError("рамка ещё не привязана к машине — выберите «только эта рамка»")
            ch = relabel_unit(s, unit, {"cls": cls}, author)
            ch.frame_id = fr.id
            return ch
        note = f"{_when(site, fr.captured_at)}, «{cam.name}»: «{_name(_orig(det))}» → «{_name(cls)}»"
        ensure_raw(s, site.id, preferred)
        s.add(_row(fr, site.id, batch, "box_relabel", "relabel", author, note, box, cls=cls,
                   orig_cls=_orig(det), orig_conf=float(det.conf)))
        kind = "box_relabel"
    else:
        raise ValueError("ожидается {cls, scope: box|unit} или {deleted: true, scope: frame|camera}")
    s.flush()
    refresh_frame(s, fr, preferred)
    s.commit()
    return Change(site.id, batch, kind, note, [fr.id], 1, replay=True, frame_id=fr.id)


def _bbox_field(body: dict, fr: Frame) -> tuple[float, float, float, float]:
    b = body.get("bbox")
    if (not isinstance(b, (list, tuple)) or len(b) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in b)):
        raise ValueError("bbox: [x, y, w, h] в пикселях кадра")
    x, y, w, h = (float(v) for v in b)
    W, H = float(fr.width or 0), float(fr.height or 0)
    if W and H:
        x2, y2 = min(W, x + w), min(H, y + h)
        x, y = max(0.0, x), max(0.0, y)
        w, h = x2 - x, y2 - y
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        raise ValueError(f"рамка слишком мала или вне кадра: стороны не меньше {MIN_SIDE_PX} px")
    return round(x, 1), round(y, 1), round(w, 1), round(h, 1)


def add_box(s: Session, frame_id: Any, body: dict, author: str) -> Change:
    fr, cam, site = _frame_ctx(s, frame_id)
    cls = _check_cls(body.get("cls"))
    box = _bbox_field(body, fr)
    batch = new_batch()
    note = f"{_when(site, fr.captured_at)}, «{cam.name}»: дорисована рамка «{_name(cls)}»"
    preferred = _preferred(s)
    ensure_raw(s, site.id, preferred)
    if not fr.processed_a:
        # Модель А кадр ещё не видела: правка ляжет поверх её ответа, когда кадр дойдёт.
        s.add(_row(fr, site.id, batch, "box_add", "add", author, note, box, cls=cls))
        s.commit()
        return Change(site.id, batch, "box_add", note, [fr.id], 1, replay=False, frame_id=fr.id)
    s.add(_row(fr, site.id, batch, "box_add", "add", author, note, box, cls=cls))
    s.flush()
    refresh_frame(s, fr, preferred)
    s.commit()
    return Change(site.id, batch, "box_add", note, [fr.id], 1, replay=True, frame_id=fr.id)


def verify_frame(s: Session, frame_id: Any, body: dict, author: str) -> Change:
    """«Разметка кадра верна»: кадр — проверенный пример для дообучения."""
    fr, cam, site = _frame_ctx(s, frame_id)
    verified = body.get("verified", True)
    if not isinstance(verified, bool):
        raise ValueError("verified: true | false")
    existing = list(s.scalars(select(Annotation).where(Annotation.frame_id == fr.id, Annotation.action == "verify")))
    if not verified:
        for r in existing:
            s.delete(r)
        s.commit()
        return Change(site.id, "", "frame_verify", "отметка «кадр проверен» снята", [fr.id], len(existing),
                      replay=False, frame_id=fr.id)
    if existing:
        r = existing[0]
        return Change(site.id, r.batch, "frame_verify", r.note, [fr.id], 0, replay=False, frame_id=fr.id)
    batch = new_batch()
    note = f"{_when(site, fr.captured_at)}, «{cam.name}»: разметка кадра проверена"
    s.add(_row(fr, site.id, batch, "frame_verify", "verify", author, note))
    s.commit()
    return Change(site.id, batch, "frame_verify", note, [fr.id], 1, replay=False, frame_id=fr.id)


# ---------------------------------------------------------------- машины

def unit_from_body(s: Session, unit_id: Any, body: dict) -> EquipmentUnit:
    return find_unit(s, unit_id, body.get("uid"), body.get("site_id"))


def _unit_boxes(s: Session, unit_ids: list[int]) -> list[tuple[Detection, Frame]]:
    return [tuple(x) for x in s.execute(
        select(Detection, Frame).join(Frame, Frame.id == Detection.frame_id)
        .where(Detection.unit_id.in_(unit_ids)).order_by(Frame.captured_at, Detection.id)).all()]


def _label(u: EquipmentUnit) -> str:
    return u.label or _name(u.cls)


def _unit_rows(s: Session, site: Site, boxes, batch: str, kind: str, author: str, note: str,
               key: str, cls: str) -> int:
    for det, fr in boxes:
        s.add(_row(fr, site.id, batch, kind, "unit", author, note, (det.x, det.y, det.w, det.h),
                   unit_key=key, cls=cls, orig_cls=_orig(det), orig_conf=float(det.conf)))
    return len(boxes)


def relabel_unit(s: Session, unit: EquipmentUnit, body: dict, author: str) -> Change:
    """Сменить тип машины целиком: все её рамки и моточасы переходят к новому классу."""
    cls = _check_cls(body.get("cls"))
    site = s.get(Site, unit.site_id)
    boxes = _unit_boxes(s, [unit.id])
    if not boxes:
        raise ValueError("у машины нет рамок на кадрах — нечего исправлять")
    ensure_raw(s, site.id, _preferred(s))
    key = unit.uid if is_manual(unit.uid) else new_key()
    batch = new_batch()
    note = f"«{_label(unit)}» → «{_name(cls)}»: тип машины исправлен ({len(boxes)} рамок)"
    n = _unit_rows(s, site, boxes, batch, "unit_relabel", author, note, key, cls)
    s.commit()
    return Change(site.id, batch, "unit_relabel", note, sorted({fr.id for _, fr in boxes}), n,
                  replay=True, sync=True)


def merge_units(s: Session, body: dict, author: str) -> Change:
    """Склеить машины: оператор видит, что «Автокран №2» и «Автокран №3» — один кран.

    Если две рамки склеиваемых машин стоят на одном кадре, это две рамки одной машины
    (кран разрезан на стрелу и башню) — лишняя, меньшая, помечается «не техника»."""
    ids = body.get("unit_ids")
    if (not isinstance(ids, list) or len(ids) < 2
            or any(isinstance(x, bool) or not isinstance(x, int) for x in ids)):
        raise ValueError("unit_ids: список из двух и более id машин")
    ids = list(dict.fromkeys(ids))
    if len(ids) < 2:
        raise ValueError("unit_ids: нужны две разные машины")
    uids = body.get("unit_uids")
    if isinstance(uids, list) and len(uids) == len(ids):
        units = [find_unit(s, i, uid, body.get("site_id")) for i, uid in zip(ids, uids)]
    else:
        units = [find_unit(s, i) for i in ids]
    if len({u.id for u in units}) < 2:
        raise ValueError("unit_ids: нужны две разные машины")
    if len({u.site_id for u in units}) != 1:
        raise ValueError("склеивать можно только машины одного объекта")
    site = s.get(Site, units[0].site_id)
    target_id = body.get("target_id")
    if target_id is not None and target_id not in ids:
        raise ValueError("target_id: одна из склеиваемых машин")
    target = units[ids.index(target_id)] if target_id is not None else None
    ids = [u.id for u in units]
    boxes = _unit_boxes(s, ids)
    per_unit = defaultdict(int)
    for det, _fr in boxes:
        per_unit[det.unit_id] += 1
    if target is None:
        target = max(units, key=lambda u: (is_manual(u.uid), per_unit[u.id], u.worked_hours or 0.0, -u.id))
    cls = _check_cls(body.get("cls") or target.cls)
    ensure_raw(s, site.id, _preferred(s))
    key = target.uid if is_manual(target.uid) else new_key()
    batch = new_batch()

    by_frame: dict[tuple[int, str], list[tuple[Detection, Frame]]] = defaultdict(list)
    for det, fr in boxes:
        by_frame[(fr.id, det.provider)].append((det, fr))
    keep, extra_boxes = [], []
    for group in by_frame.values():
        group.sort(key=lambda p: (bool(((p[0].extra or {}).get(MANUAL) or {}).get("added")),
                                  p[0].w * p[0].h, p[0].conf), reverse=True)
        keep.append(group[0])
        extra_boxes.extend(group[1:])
    names = ", ".join(f"«{_label(u)}»" for u in units)
    note = f"Склеены {names} → одна машина «{_name(cls)}»"
    if extra_boxes:
        note += f"; лишних рамок той же машины на одном кадре убрано: {len(extra_boxes)}"
    n = _unit_rows(s, site, keep, batch, "unit_merge", author, note, key, cls)
    for det, fr in extra_boxes:
        s.add(_row(fr, site.id, batch, "unit_merge", "delete", author, note, (det.x, det.y, det.w, det.h),
                   scope="frame", orig_cls=_orig(det), orig_conf=float(det.conf)))
    s.commit()
    return Change(site.id, batch, "unit_merge", note, sorted({fr.id for _, fr in boxes}), n + len(extra_boxes),
                  replay=True, sync=True)


def split_unit(s: Session, unit: EquipmentUnit, body: dict, author: str) -> Change:
    """Разделить машину: с кадра frame_id (на его камере) это другая машина.

    Так разводятся машины, которые трекер слепил: утром мачта связи на горизонте,
    днём — башенный кран на том же месте кадра."""
    fr, cam, site = _frame_ctx(s, body.get("frame_id"))
    boxes = _unit_boxes(s, [unit.id])
    if not any(f.id == fr.id for _, f in boxes):
        raise ValueError("на этом кадре нет рамки этой машины — выберите кадр из её истории")
    after = [(d, f) for d, f in boxes if f.camera_id == fr.camera_id and f.captured_at >= fr.captured_at]
    before = [(d, f) for d, f in boxes if not (f.camera_id == fr.camera_id and f.captured_at >= fr.captured_at)]
    if not before:
        raise ValueError("это первый кадр машины — разделять нечего; чтобы сменить тип, используйте «Изменить тип»")
    new_cls = _check_cls(body.get("cls") or unit.cls)
    ensure_raw(s, site.id, _preferred(s))
    key_old = unit.uid if is_manual(unit.uid) else new_key()
    key_new = new_key()
    batch = new_batch()
    note = (f"«{_label(unit)}» разделена с {_when(site, fr.captured_at)} («{cam.name}»): "
            f"{len(after)} рамок — отдельная машина «{_name(new_cls)}»")
    n = _unit_rows(s, site, before, batch, "unit_split", author, note, key_old, unit.cls)
    n += _unit_rows(s, site, after, batch, "unit_split", author, note, key_new, new_cls)
    s.commit()
    return Change(site.id, batch, "unit_split", note, sorted({f.id for _, f in boxes}), n,
                  replay=True, sync=True, frame_id=fr.id)


def delete_unit(s: Session, unit: EquipmentUnit, body: dict, author: str) -> Change:
    """«Это не техника»: все рамки машины удаляются; для неподвижной (контейнер, мачта,
    столб) — ещё и место на её камерах, чтобы она не вернулась на следующем кадре."""
    site = s.get(Site, unit.site_id)
    boxes = _unit_boxes(s, [unit.id])
    if not boxes:
        raise ValueError("у машины нет рамок на кадрах — нечего удалять")
    hide = body.get("hide_place")
    if hide is None:
        hide = not (unit.worked_hours or 0) > 0 and unit.last_moved is None
    if not isinstance(hide, bool):
        raise ValueError("hide_place: true | false")
    ensure_raw(s, site.id, _preferred(s))
    batch = new_batch()
    note = (f"«{_label(unit)}» — не техника: {len(boxes)} рамок удалено"
            + ("; её место на камере больше не считается техникой" if hide else ""))
    for det, fr in boxes:
        s.add(_row(fr, site.id, batch, "unit_delete", "delete", author, note, (det.x, det.y, det.w, det.h),
                   scope="frame", orig_cls=_orig(det), orig_conf=float(det.conf)))
    n = len(boxes)
    if hide:
        last_by_cam: dict[int, tuple[Detection, Frame]] = {}
        for det, fr in boxes:
            last_by_cam[fr.camera_id] = (det, fr)
        for det, fr in last_by_cam.values():
            s.add(_row(fr, site.id, batch, "unit_delete", "delete", author, note, (det.x, det.y, det.w, det.h),
                       scope="camera", orig_cls=_orig(det), orig_conf=float(det.conf)))
            n += 1
    s.commit()
    return Change(site.id, batch, "unit_delete", note, sorted({fr.id for _, fr in boxes}), n,
                  replay=True, sync=True)


def patch_unit(s: Session, unit_id: Any, body: dict, author: str) -> Change:
    unit = unit_from_body(s, unit_id, body)
    if "cls" in body:
        return relabel_unit(s, unit, body, author)
    raise ValueError("ожидается {cls}: новый тип машины")


def undo(s: Session, batch: str) -> Change:
    """Отменить действие целиком (все строки пачки) — кадры обновить, технику перепрогнать."""
    rows = list(s.scalars(select(Annotation).where(Annotation.batch == str(batch))))
    if not rows:
        raise LookupError(f"правка {batch} не найдена — возможно, уже отменена")
    site_id = rows[0].site_id
    kind = rows[0].kind
    frame_ids = sorted({r.frame_id for r in rows})
    note = rows[0].note
    s.execute(delete(Annotation).where(Annotation.batch == str(batch)))
    s.flush()
    preferred = _preferred(s)
    if kind not in UNIT_KINDS and len(frame_ids) <= 20:
        for fid in frame_ids:
            fr = s.get(Frame, fid)
            if fr is not None and fr.processed_a:
                refresh_frame(s, fr, preferred)
    s.commit()
    return Change(site_id, str(batch), kind, f"отменено: {note}", frame_ids, len(rows),
                  replay=kind != "frame_verify", sync=kind in UNIT_KINDS,
                  frame_id=frame_ids[0] if len(frame_ids) == 1 else None)


# --------------------------------------------------------------------------
# журнал и сводка
# --------------------------------------------------------------------------

def frame_summary(s: Session, fr: Frame) -> dict[str, Any]:
    """Правки кадра — для панели разбора: проверен ли кадр, что правили."""
    rows = list(s.scalars(select(Annotation).where(Annotation.frame_id == fr.id).order_by(Annotation.id)))
    seen: dict[str, dict] = {}
    for r in rows:
        if r.batch not in seen:
            seen[r.batch] = {"batch": r.batch, "kind": r.kind, "kind_name": KINDS.get(r.kind, r.kind),
                             "note": r.note, "author": r.author, "created_at": adapters.iso(r.created_at)}
    return {
        "verified": any(r.action == "verify" for r in rows),
        "reviewed": any(r.kind in REVIEW_KINDS for r in rows),
        "count": len(rows),
        "batches": list(seen.values())[-20:],
    }


def journal(s: Session, site_id: int, limit: int = 100) -> dict[str, Any]:
    heads = s.execute(
        select(Annotation.batch, func.min(Annotation.id), func.count(Annotation.id),
               func.count(func.distinct(Annotation.frame_id)))
        .where(Annotation.site_id == site_id).group_by(Annotation.batch)
        .order_by(func.min(Annotation.id).desc()).limit(max(1, min(limit, 500)))).all()
    first = {r.id: r for r in s.scalars(select(Annotation).where(Annotation.id.in_([h[1] for h in heads])))} \
        if heads else {}
    batches = []
    for batch, first_id, n, frames in heads:
        r = first.get(first_id)
        if r is None:
            continue
        sample = list(s.scalars(select(Annotation.frame_id).where(Annotation.batch == batch)
                                .group_by(Annotation.frame_id).order_by(Annotation.frame_id.desc()).limit(4)))
        batches.append({"batch": batch, "kind": r.kind, "kind_name": KINDS.get(r.kind, r.kind), "note": r.note,
                        "author": r.author, "created_at": adapters.iso(r.created_at), "rows": n, "frames": frames,
                        "frame_ids": sample})
    by_kind = dict(s.execute(select(Annotation.kind, func.count(func.distinct(Annotation.batch)))
                             .where(Annotation.site_id == site_id).group_by(Annotation.kind)).all())
    reviewed = s.scalar(select(func.count(func.distinct(Annotation.frame_id)))
                        .where(Annotation.site_id == site_id, Annotation.kind.in_(REVIEW_KINDS))) or 0
    manual_units = s.scalar(select(func.count()).select_from(EquipmentUnit).where(
        EquipmentUnit.site_id == site_id, EquipmentUnit.uid.like(f"{MANUAL_PREFIX}%"))) or 0
    return {"batches": batches,
            "stats": {"actions": sum(by_kind.values()), "by_kind": by_kind, "reviewed_frames": reviewed,
                      "manual_units": manual_units}}


def processed_frames(s: Session, site_id: int) -> int:
    cams = select(Camera.id).where(Camera.site_id == site_id)
    return s.scalar(select(func.count()).select_from(Frame).where(Frame.camera_id.in_(cams),
                                                                  Frame.processed_a.is_(True))) or 0


# --------------------------------------------------------------------------
# датасет для дообучения
# --------------------------------------------------------------------------

EXPORT_SCOPES = ("reviewed", "all")


def _is_val(camera_id: int, day: str, ratio: float) -> bool:
    # Делим по суткам камеры, а не по кадрам: соседние кадры почти одинаковы, и
    # кадр в валидации рядом с «братом» в обучении завысил бы метрику.
    return zlib.crc32(f"{camera_id}:{day}".encode()) % 1000 < ratio * 1000


def export_dataset(s: Session, site_ids: list[int], zf: zipfile.ZipFile, scope: str = "reviewed",
                   val_ratio: float = 0.2, preferred: str | None = None) -> dict[str, Any]:
    """Размеченные кадры → zip в формате YOLO (images/, labels/, data.yaml по 21 классу словаря).

    reviewed — только кадры, которые человек правил или подтвердил («проверено»);
    all — все разобранные кадры: рамки модели как псевдоразметка плюс правки.
    Класс рамки — с учётом правок: ручной класс рамки, иначе класс её машины."""
    if scope not in EXPORT_SCOPES:
        raise ValueError(f"scope: {' | '.join(EXPORT_SCOPES)}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val: доля валидации от 0 до 0.9")
    preferred = preferred or _preferred(s)
    classes = list(taxonomy.equipment())
    index = {k: i for i, k in enumerate(classes)}
    sites = {x.id: x for x in s.scalars(select(Site).where(Site.id.in_(site_ids)))}
    if not sites:
        raise LookupError("объекты не найдены")
    cams = {cam.id: cam for cam in s.scalars(select(Camera).where(Camera.site_id.in_(list(sites))))}
    q = select(Frame).where(Frame.camera_id.in_(list(cams) or [-1]), Frame.processed_a.is_(True))
    if scope == "reviewed":
        reviewed = select(Annotation.frame_id).where(Annotation.site_id.in_(list(sites)),
                                                     Annotation.kind.in_(REVIEW_KINDS))
        q = q.where(Frame.id.in_(reviewed))
    frames = list(s.scalars(q.order_by(Frame.captured_at, Frame.id)))

    verified = set(s.scalars(select(Annotation.frame_id).where(Annotation.site_id.in_(list(sites)),
                                                              Annotation.kind.in_(REVIEW_KINDS))))
    split: dict[int, str] = {}
    for fr in frames:
        day = fr.captured_at.astimezone(adapters.site_tz(sites[cams[fr.camera_id].site_id])).date().isoformat()
        split[fr.id] = "val" if _is_val(fr.camera_id, day, val_ratio) else "train"
    parts = set(split.values())
    if val_ratio > 0 and len(frames) >= 2 and parts != {"train", "val"}:
        # Одна-две смены съёмки: по суткам не поделить — каждый k-й кадр в валидацию
        # (хотя бы один: ultralytics без валидации не учит).
        k = max(2, min(round(1 / val_ratio), len(frames)))
        for i, fr in enumerate(frames):
            split[fr.id] = "val" if i % k == k - 1 else "train"
    if val_ratio == 0:
        split = {fid: "train" for fid in split}

    stats = {"frames": 0, "boxes": 0, "manual_boxes": 0, "reviewed_frames": 0, "train": 0, "val": 0,
             "skipped": 0, "by_class": defaultdict(int)}
    manifest_frames = []
    for i in range(0, len(frames), 200):
        chunk = frames[i:i + 200]
        dets = adapters.detections_by_frame(s, [f.id for f in chunk], preferred)
        for fr in chunk:
            try:
                data = storage.get().get(fr.key)
            except Exception:  # noqa: BLE001 — файла нет в хранилище: кадр пропускаем, но честно считаем
                stats["skipped"] += 1
                continue
            W, H = fr.width, fr.height
            if not W or not H:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    stats["skipped"] += 1
                    continue
                H, W = img.shape[:2]
            cam = cams[fr.camera_id]
            part = split[fr.id]
            ext = "." + fr.key.rsplit(".", 1)[-1].lower() if "." in fr.key.rsplit("/", 1)[-1] else ".jpg"
            stem = f"s{cam.site_id}_c{cam.id}_f{fr.id}"
            lines, boxes = [], []
            for d in dets.get(fr.id, []):
                extra = d.extra or {}
                cls = extra.get(MANUAL_CLS) or d.cls
                if cls not in index:
                    continue
                x1, y1 = max(0.0, d.x), max(0.0, d.y)
                x2, y2 = min(float(W), d.x + d.w), min(float(H), d.y + d.h)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                cx, cy, bw, bh = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
                lines.append(f"{index[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                man = extra.get(MANUAL) or {}
                boxes.append({"cls": cls, "bbox": [round(d.x, 1), round(d.y, 1), round(d.w, 1), round(d.h, 1)],
                              "source": "manual" if man else "model",
                              **({"orig_cls": man["orig_cls"]} if man.get("orig_cls") and man["orig_cls"] != cls
                                 else {})})
                stats["by_class"][cls] += 1
                stats["manual_boxes"] += bool(man)
            zf.writestr(f"images/{part}/{stem}{ext}", data)
            zf.writestr(f"labels/{part}/{stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
            stats["frames"] += 1
            stats[part] += 1
            stats["boxes"] += len(lines)
            stats["reviewed_frames"] += fr.id in verified
            manifest_frames.append({"image": f"images/{part}/{stem}{ext}", "frame_id": fr.id, "site_id": cam.site_id,
                                    "camera_id": cam.id, "camera": cam.name,
                                    "captured_at": adapters.iso(fr.captured_at), "split": part,
                                    "reviewed": fr.id in verified, "width": W, "height": H, "boxes": boxes})

    names = "\n".join(f"  {i}: {k}  # {taxonomy.equipment_name(k)}" for i, k in enumerate(classes))
    zf.writestr("data.yaml", (
        "# СтройВзор: разметка техники с ручными правками оператора (YOLO)\n"
        "# Классы — 21 ключ словаря reference/checklist.json (порядок фиксирован).\n"
        "path: .\ntrain: images/train\nval: images/val\n"
        f"nc: {len(classes)}\nnames:\n{names}\n"))
    zf.writestr("classes.txt", "\n".join(classes) + "\n")
    zf.writestr("classes_ru.txt", "\n".join(f"{k}\t{taxonomy.equipment_name(k)}" for k in classes) + "\n")
    stats["by_class"] = dict(stats["by_class"])
    manifest = {"created_at": dt.datetime.now(dt.UTC).isoformat(), "scope": scope, "val_ratio": val_ratio,
                "provider": preferred, "classes": classes,
                "sites": [{"id": x.id, "name": x.name, "address": x.address} for x in sites.values()],
                "stats": stats, "frames": manifest_frames}
    zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    zf.writestr("README.txt", (
        "Датасет техники СтройВзора для дообучения детектора модели А.\n\n"
        f"Кадров: {stats['frames']} (обучение {stats['train']}, валидация {stats['val']}), "
        f"рамок: {stats['boxes']}, из них правленых вручную: {stats['manual_boxes']}.\n"
        f"Отбор: {'только кадры, проверенные оператором' if scope == 'reviewed' else 'все разобранные кадры'}.\n\n"
        "Формат YOLO: labels/*.txt — «класс cx cy w h» в долях кадра; классы — data.yaml.\n"
        "manifest.json — откуда кадр (объект, камера, время), какие рамки поставил человек\n"
        "(source=manual, orig_cls — что было у модели), какие — модель (source=model).\n\n"
        "Дообучение (ultralytics):\n"
        "  yolo detect train data=data.yaml model=models/equipment_yolo.pt epochs=30 imgsz=960\n"))
    return stats
