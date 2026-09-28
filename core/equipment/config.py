"""Пороги модели А в одном месте.

Все числа, от которых зависит «работает / стоит», склейка камер и списание
моточасов, собраны здесь, а не разбросаны по коду: их редактирует страница
«Настройки» (таблица `settings`, ключ `thresholds.equipment`), и методика в
документации ссылается на эти же имена. Значения по умолчанию — из
docs/ARCHITECTURE.md §5 и PLAN.md §3.9–3.10.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class EquipmentConfig:
    # ---------------- постобработка кадра (postprocess.py) ----------------
    min_conf: float = 0.25                      # общий порог уверенности детектора
    conf_by_class: dict[str, float] = field(default_factory=dict)  # свой порог для класса
    min_box_side_px: float = 10.0               # рамка уже 10 px — шум, а не машина
    min_box_area_frac: float = 0.0001           # доля площади кадра
    edge_margin_px: float = 2.0                 # «касается края кадра»
    edge_min_area_frac: float = 0.002           # маленький обрезок у края — не считаем
    same_class_iou: float = 0.7                 # дубли одного класса (VLM любит повторять объект)
    cross_class_iou: float = 0.6                # «одна машина — одна рамка» внутри группы путаницы
    cross_class_ioa: float = 0.85               # рамка почти целиком внутри другой (манипулятор в грузовике)
    pair_iou: float = 0.85                      # пары, стоящие вплотную (насос + миксер), гасим только при почти полном совпадении
    pair_exempt: tuple[tuple[str, str], ...] = (("concrete_pump", "concrete_mixer"),)

    # ---------------- трекер одной камеры (tracker.py) ----------------
    match_max_cost: float = 1.8                 # сумма (1−IoU) + расстояние/диагональ + штраф класса
    match_max_center_diag: float = 1.5          # дальше полутора диагоналей — точно не тот же трек
    class_penalty_confusable: float = 0.3       # truck ↔ dump_truck: детектор путает, сопоставлять можно
    class_penalty_other: float = 1.0            # экскаватор ↔ бульдозер: только при почти полном совпадении рамок
    rematch_min_similarity: float = 0.35        # второй проход: машина уехала далеко в пределах кадра
    rematch_vacated_min: float = 0.08           # … и старое место правда опустело (иначе это пропуск детектора)
    max_gap_min: float = 120.0                  # разрыв больше — смещение не считаем работой, начинаем заново
    vote_window: int = 20                       # сколько последних меток трека голосуют за класс
    move_px_min: float = 8.0                    # смещение центра, px (минимум)
    move_diag_frac: float = 0.15                # … или доля диагонали рамки
    shape_delta_thr: float = 0.15               # |Δw/w| + |Δh/h| — работа стрелой при неподвижном центре
    shape_needs_appearance: bool = True         # форму рамки подтверждаем содержимым (детектор «дышит» рамкой)
    appearance_thr: float = 0.02                # доля площади рамки, где структура изменилась сверх фона (поза ковша)
    appearance_confirm_thr: float = 0.008       # минимальное изменение, чтобы поверить изменению формы рамки
    appearance_min_std: float = 3.0             # однотонный кроп (ночь, засвет) — сравнивать нечего
    camera_shift_reset_frac: float = 0.05       # сдвиг камеры больше 5 % диагонали — ракурс сменился

    # ---------------- статусы (status.py) ----------------
    active_window_min: float = 15.0             # ACTIVE, если двигалась в окне последнего наблюдения
    parked_after_h: float = 48.0                # стоит дольше — ждёт вывоза (PARKED)
    departed_after_h: float = 3.0               # камеры снимают, а машины нет дольше — уехала
    parking_zone_parks: bool = True             # в зоне отстоя неработающая машина сразу PARKED
    revive_within_h: float = 72.0               # вернулась машина того же класса — тот же unit_id

    # ---------------- слияние камер (fusion.py) ----------------
    merge_window_min: float = 15.0              # детекции разных камер ±15 мин — одно «мгновение»
    merge_radius_m: float = 5.0                 # ближе — одна машина
    merge_gray_factor: float = 1.6              # до 1.6·радиуса решает внешность
    appearance_min_similarity: float = 0.5      # сходство цветовых гистограмм для спорных случаев
    appearance_weight: float = 0.5              # вклад внешности в порядок склейки
    merge_young_min: float = 60.0               # дубль, родившийся в «серой зоне», склеиваем, пока он молодой
    split_radius_m: float = 12.0                # трек ушёл от своей единицы дальше — отделяем
    split_after: int = 2                        # … после стольких подряд расхождений

    # ---------------- моточасы (hours.py) ----------------
    max_credit_gap_min: float = 45.0            # за интервал между кадрами списываем не больше 45 мин
    confirm_moves: int = 2                      # «работает» = двигалась минимум на N интервалах подряд
    shift_hours: float = 10.0
    utilization: float = 0.7
    workdays: tuple[int, ...] = (0, 1, 2, 3, 4, 5)   # пн–сб
    timezone: str = "Europe/Moscow"             # день плана — московский, внутри всё в UTC

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EquipmentConfig":
        """Из настроек UI. Неизвестные ключи пропускаем с предупреждением (старые
        настройки не должны ронять сервис), типы приводим к типам по умолчанию."""
        cfg = cls()
        if not data:
            return cfg
        known = {f.name: f for f in dataclasses.fields(cls)}
        for key, value in data.items():
            if key not in known:
                log.warning("EquipmentConfig: неизвестный параметр %r пропущен", key)
                continue
            default = getattr(cfg, key)
            setattr(cfg, key, _coerce(key, value, default))
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["workdays"] = list(self.workdays)
        out["pair_exempt"] = [list(p) for p in self.pair_exempt]
        return out

    def validate(self) -> None:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
                raise ValueError(f"EquipmentConfig.{f.name} не может быть отрицательным: {v}")
        if not 0 < self.utilization <= 1:
            raise ValueError("EquipmentConfig.utilization должен быть в (0, 1]")
        if self.confirm_moves < 1:
            raise ValueError("EquipmentConfig.confirm_moves ≥ 1")
        if any(d not in range(7) for d in self.workdays):
            raise ValueError("EquipmentConfig.workdays — дни недели 0..6 (0 — понедельник)")

    def conf_for(self, cls: str) -> float:
        return self.conf_by_class.get(cls, self.min_conf)


def _coerce(key: str, value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "да", "on")
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if key == "workdays":
        return tuple(int(d) for d in value)
    if key == "pair_exempt":
        return tuple(tuple(p) for p in value)
    if isinstance(default, dict):
        return {str(k): float(v) for k, v in dict(value).items()}
    return value
