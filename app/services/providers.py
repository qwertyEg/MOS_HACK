"""Реестр провайдеров моделей А и Б и ленивый доступ к модулям `core.*`.

Почему лениво. Реализации моделей тянут torch/ultralytics/transformers или
ходят в сеть; веб-слой обязан подниматься и без них (кадры сохраняются всегда,
анализ — когда провайдер готов). Поэтому `core.equipment`, `core.stage`,
`core.plan`, `core.analytics` импортируются только в момент использования и
только через `module()`: тесты подменяют их фейками через `override_module`,
не трогая `sys.modules`.

Экземпляры детекторов/классификаторов кэшируются (загрузка весов YOLO и SigLIP
стоит секунд), готовность — с TTL: `ready()` локальной VLM ходит в сеть.
"""
from __future__ import annotations

import contextlib
import importlib
import threading
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

from app.config import settings

MODEL_A = ("yolo", "glm")
MODEL_B = ("siglip", "local_vlm", "glm")

_overrides: dict[str, Any] = {}
MISSING = object()   # подмена «модуля нет»: module() бросает ImportError, как при настоящем отсутствии


class ProviderUnavailable(RuntimeError):
    """Провайдер не может работать прямо сейчас (нет модуля, весов, ключа, сети)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def override_module(name: str, module: Any) -> None:
    """Подменить модуль ядра (тесты, отладка). `None` — убрать подмену."""
    if module is None:
        _overrides.pop(name, None)
    else:
        _overrides[name] = module


def clear_overrides() -> None:
    _overrides.clear()


def module(name: str) -> ModuleType | Any:
    """Модуль ядра по имени: подмена → атрибут подменённого пакета → импорт."""
    if name in _overrides:
        if _overrides[name] is MISSING:
            raise ImportError(f"No module named '{name}' (подменено в тестах)")
        return _overrides[name]
    parent, _, attr = name.rpartition(".")
    if parent in _overrides:
        if _overrides[parent] is MISSING:
            raise ImportError(f"No module named '{name}' (подменено в тестах)")
        if hasattr(_overrides[parent], attr):
            return getattr(_overrides[parent], attr)
        raise ImportError(f"в подменённом {parent} нет {attr}")
    return importlib.import_module(name)


def optional_module(name: str) -> ModuleType | Any | None:
    """То же, но `None`, если модуля нет (например, core.analytics ещё не влит)."""
    try:
        return module(name)
    except ImportError:
        return None


def _thresholds() -> dict:
    """Пороги из таблицы settings (лениво: settings импортирует этот модуль).
    БД недоступна (утилита до bootstrap) — умолчания реализаций."""
    try:
        from app import db
        from app.services import settings as settings_svc
        with db.session() as s:
            return settings_svc.get_state(s)["thresholds"]
    except Exception:  # noqa: BLE001 — пороги не должны мешать собрать провайдера
        return {}


def equipment_config(values: dict) -> Any | None:
    """EquipmentConfig модели А из порогов UI; None — модуль не подключён."""
    eq = optional_module("core.equipment")
    if eq is None or not hasattr(eq, "EquipmentConfig"):
        return None
    try:
        return eq.EquipmentConfig.from_dict(values or {})
    except Exception:  # noqa: BLE001 — кривой порог в БД не должен выключать детектор
        return eq.EquipmentConfig()


_embedders: dict[str, Any] = {}


def shared_embedder(model_name: str) -> Any | None:
    """Один SigLIP на процесс для модели Б и уточнения подтипа грузовиков (модель А).

    Обе модели по умолчанию — google/siglip2-base-patch16-224; две копии весов
    на ноутбуке 8 ГБ — лишние ~1.5 ГБ. core.equipment.refine.SiglipEmbedder
    держит модель в общем кэше по (имя, устройство) и понимает ответы и
    transformers 4, и transformers 5. None — transformers/torch не установлены
    (тогда классификатор сам объяснит это в ready()).
    """
    if model_name in _embedders:
        return _embedders[model_name]
    refine = optional_module("core.equipment.refine")
    if refine is None or not hasattr(refine, "SiglipEmbedder"):
        return None
    try:
        if not refine.transformers_available():
            return None
    except Exception:  # noqa: BLE001
        return None
    import os
    threads = os.environ.get("EQUIPMENT_THREADS", "").strip()
    emb = refine.SiglipEmbedder(model_name, threads=int(threads) if threads.isdigit() and int(threads) > 0 else None)
    _embedders[model_name] = emb
    return emb


def _construct(factory: Callable, name: str, kwargs: dict) -> Any:
    """Имена kwargs у реализаций заранее не согласованы (контракт — `**kw`),
    поэтому пробуем с нашими параметрами, а при TypeError — без них: реализации
    всё равно видят EQUIPMENT_WEIGHTS / STAGE_CLIP_MODEL в окружении."""
    if kwargs:
        try:
            return factory(name, **kwargs)
        except TypeError:
            pass
    return factory(name)


class Registry:
    def __init__(self) -> None:
        self._instances: dict[tuple[str, str], Any] = {}
        self._status: dict[tuple[str, str], tuple[float, bool, str]] = {}
        self._factories: dict[tuple[str, str], Callable[[], Any]] = {}
        self._call_locks: dict[tuple[str, str], threading.Lock] = {}
        self._lock = threading.RLock()

    # --- подмена в тестах ------------------------------------------------

    def set_factory(self, kind: str, name: str, factory: Callable[[], Any] | None) -> None:
        with self._lock:
            key = (kind, name)
            if factory is None:
                self._factories.pop(key, None)
            else:
                self._factories[key] = factory
            self._instances.pop(key, None)
            self._status.pop(key, None)

    def reset(self) -> None:
        """Забыть экземпляры и статусы (сменились пороги/ключи — пересоздать)."""
        with self._lock:
            self._instances.clear()
            self._status.clear()

    # --- экземпляры ------------------------------------------------------

    def _kwargs(self, kind: str, name: str) -> dict:
        """Параметры конструкторов реальных реализаций (core.equipment / core.stage).

        Пороги из настроек уходят в конструкторы, поэтому PUT /api/settings со
        сменой порогов делает registry.reset() — экземпляры пересобираются.
        """
        thr = _thresholds()
        if kind == "detector" and name == "yolo":
            kw: dict[str, Any] = {"weights": str(settings.path(settings.equipment_weights))}
            cfg = equipment_config(thr.get("equipment") or {})
            if cfg is not None:
                kw["config"] = cfg          # min_conf, пороги уточнения подтипа грузовиков и т.п.
            return kw
        if kind == "classifier" and name == "siglip":
            stage = thr.get("stage") or {}
            kw = {"model_name": settings.stage_clip_model}
            if "yes_thr" in stage:
                kw["yes_threshold"] = float(stage["yes_thr"])
            if "no_thr" in stage:
                kw["no_threshold"] = float(stage["no_thr"])
            emb = shared_embedder(settings.stage_clip_model)
            if emb is not None:
                kw["embedder"] = emb
            return kw
        return {}

    def get(self, kind: str, name: str) -> Any:
        key = (kind, name)
        with self._lock:
            if key in self._instances:
                return self._instances[key]
            factory = self._factories.get(key)
        try:
            if factory is not None:
                obj = factory()
            elif kind == "detector":
                obj = _construct(module("core.equipment").get_detector, name, self._kwargs(kind, name))
            elif kind == "classifier":
                obj = _construct(module("core.stage").get_classifier, name, self._kwargs(kind, name))
            else:
                raise ProviderUnavailable(f"неизвестный тип провайдера {kind}")
        except ProviderUnavailable:
            raise
        except ImportError as exc:
            raise ProviderUnavailable(f"модуль модели не установлен: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — любая ошибка сборки = провайдер не готов
            raise ProviderUnavailable(f"{name}: {type(exc).__name__}: {exc}") from exc
        with self._lock:
            self._instances.setdefault(key, obj)
            return self._instances[key]

    def detector(self, name: str) -> Any:
        return self.get("detector", name)

    def classifier(self, name: str) -> Any:
        return self.get("classifier", name)

    def peek(self, kind: str, name: str) -> Any | None:
        with self._lock:
            return self._instances.get((kind, name))

    # --- готовность ------------------------------------------------------

    def ready(self, kind: str, name: str, fresh: bool = False) -> tuple[bool, str]:
        key = (kind, name)
        now = time.monotonic()
        with self._lock:
            cached = self._status.get(key)
        if cached and not fresh and now - cached[0] < settings.provider_status_ttl_s:
            return cached[1], cached[2]
        try:
            obj = self.get(kind, name)
            ok, reason = obj.ready()
            ok, reason = bool(ok), str(reason or "")
        except ProviderUnavailable as exc:
            ok, reason = False, exc.reason
        except Exception as exc:  # noqa: BLE001
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._status[key] = (now, ok, reason)
        return ok, reason

    def call_lock(self, kind: str, name: str):
        """Локальные модели вызываем по одной: N потоков камер с YOLO/SigLIP
        одновременно на ноутбуке 8 ГБ — это своп, а не ускорение. Внешний API
        (glm) упирается в сеть, его вызовы параллелим."""
        if name == "glm":
            return contextlib.nullcontext()
        # YOLO и SigLIP — один замок: они делят процессорные потоки torch и одну
        # копию SigLIP (уточнение подтипа грузовиков + модель Б), а быстрый
        # токенизатор HF не терпит одновременных вызовов из разных потоков.
        key = ("local", "torch") if name in ("yolo", "siglip") else (kind, name)
        with self._lock:
            return self._call_locks.setdefault(key, threading.Lock())

    def require(self, kind: str, name: str) -> Any:
        """Готовый экземпляр или ProviderUnavailable с человекочитаемой причиной."""
        ok, reason = self.ready(kind, name)
        if not ok:
            raise ProviderUnavailable(reason or f"провайдер {name} не готов")
        return self.get(kind, name)

    def status(self, fresh: bool = False) -> dict[str, dict]:
        """Для /api/health и /api/settings: yolo, siglip, glm, local_vlm."""
        out: dict[str, dict] = {}
        ok, reason = self.ready("detector", "yolo", fresh)
        out["yolo"] = {"ready": ok, "reason": reason, "role": "model_a"}
        ok, reason = self.ready("classifier", "siglip", fresh)
        out["siglip"] = {"ready": ok, "reason": reason, "role": "model_b"}
        ok_a, reason_a = self.ready("detector", "glm", fresh)
        ok_b, reason_b = self.ready("classifier", "glm", fresh)
        out["glm"] = {"ready": ok_a and ok_b, "reason": reason_b or reason_a,
                      "role": "model_a+model_b", "model_a": ok_a, "model_b": ok_b}
        ok, reason = self.ready("classifier", "local_vlm", fresh)
        out["local_vlm"] = {"ready": ok, "reason": reason, "role": "model_b"}
        return out


registry = Registry()
