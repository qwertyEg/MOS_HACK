"""Статусная машина единицы техники: ACTIVE / IDLE / PARKED / DEPARTED.

Организаторы: техника неделями стоит на площадке в ожидании вывоза —
возить туда-сюда дороже. Значит, «экскаватор в кадре» ещё не значит «идут
земляные работы» (PLAN.md §3.9). Статус отвечает на вопрос «задействована ли»:

* ACTIVE   — двигалась в окне последнего наблюдения (любой из камер: если одна
             камера видит, как ходит стрела, а другой её не видно, — работает);
* IDLE     — стоит меньше `parked_after_h`;
* PARKED   — стоит дольше (или стоит в размеченной зоне отстоя) — ждёт вывоза,
             «лишней техникой» не считается, но даёт EQUIPMENT_PARKED_ONLY;
* DEPARTED — камеры, которые её видели, продолжают снимать, а её нет дольше
             `departed_after_h`. Если камера просто молчала (нет кадров),
             машина не «уезжает»: отсутствие кадров — не доказательство.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Iterable, Mapping

from core.contracts import UnitState, UnitStatus

from .config import EquipmentConfig

STATUS_RU = {
    UnitStatus.ACTIVE: "работает",
    UnitStatus.IDLE: "стоит",
    UnitStatus.PARKED: "на стоянке",
    UnitStatus.DEPARTED: "уехала",
}


def compute_status(unit: UnitState, now: dt.datetime, camera_last_frame: Mapping[str, dt.datetime],
                   config: EquipmentConfig, in_parking_zone: bool = False) -> UnitStatus:
    cfg = config
    if _absent(unit, now, camera_last_frame) > dt.timedelta(hours=cfg.departed_after_h):
        return UnitStatus.DEPARTED
    if unit.last_moved is not None and unit.last_moved >= unit.last_seen - dt.timedelta(minutes=cfg.active_window_min):
        return UnitStatus.ACTIVE
    if in_parking_zone and cfg.parking_zone_parks:
        return UnitStatus.PARKED
    # Сколько стоит — до последнего раза, когда её ВИДЕЛИ стоящей: если камера
    # молчала сутки, мы не знаем, что машина всё это время стояла.
    still_since = unit.last_moved or unit.first_seen
    if unit.last_seen - still_since >= dt.timedelta(hours=cfg.parked_after_h):
        return UnitStatus.PARKED
    return UnitStatus.IDLE


def _absent(unit: UnitState, now: dt.datetime, camera_last_frame: Mapping[str, dt.datetime]) -> dt.timedelta:
    """Сколько камеры единицы снимали, не видя её. Без камер (восстановленная запись) — по часам площадки."""
    if not unit.cameras:
        return now - unit.last_seen
    gaps = [camera_last_frame[c] - unit.last_seen for c in unit.cameras if c in camera_last_frame]
    return max(gaps) if gaps else dt.timedelta(0)


def count_by_class(units: Iterable[UnitState],
                   statuses: Iterable[UnitStatus] = (UnitStatus.ACTIVE, UnitStatus.IDLE, UnitStatus.PARKED)
                   ) -> dict[str, int]:
    """Сколько единиц каждого класса на площадке. Считаем unit_id, а не рамки:
    одна машина в поле зрения трёх камер — одна машина."""
    allowed = set(statuses)
    return dict(Counter(u.cls for u in units if u.status in allowed))
