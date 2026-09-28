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
        if kind == "detector" and name == "yolo":
            return {"weights": str(settings.path(settings.equipment_weights))}
        if kind == "classifier" and name == "siglip":
            return {"model": settings.stage_clip_model}
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
        with self._lock:
            return self._call_locks.setdefault((kind, name), threading.Lock())

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
