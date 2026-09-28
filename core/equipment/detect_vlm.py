"""EXTERNAL-детектор техники: GLM-4.6V (z.ai) с рамками, тот же Detection, что у YOLO.

GLM-4.6V умеет grounding: по просьбе возвращает рамки объектов в
нормированных координатах 0..1000. Промпт перечисляет ключи техники с
русскими названиями и описанием внешнего вида (`look` из checklist.json) —
то же, по чему человек отличает автокран от кран-манипулятора.

Ответ модели — внешний ввод, поэтому разбор недоверчивый: неизвестные
классы выбрасываются (с пометкой в `last_report`), координаты обрезаются по
кадру, перевёрнутые рамки чинятся, масштаб 0..1 и пиксели распознаются.
Мусор вместо ответа и ЛЮБАЯ ошибка клиента (сеть, ключ, KeyError в разборе)
превращаются в VLMError с понятным текстом: веб-слой ловит именно её и
показывает пользователю, а не падает (баг M4 наследия: ловился только
VLMError, остальное обрывало прогон).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from core import taxonomy
from core.contracts import Detection, FrameInfo, Provider

from . import postprocess
from .classes import canonical_class
from .config import EquipmentConfig

try:
    import core.vlm_client as _vlm
except ModuleNotFoundError as e:      # модуль модели Б ещё не влит — см. _vlm_compat
    if e.name != "core.vlm_client":   # сломан сам модуль (нет requests и т.п.) — не прятать
        raise
    from . import _vlm_compat as _vlm

VLMError = _vlm.VLMError

log = logging.getLogger(__name__)

SYSTEM = (
    "Ты — детектор строительной техники на снимках камер видеонаблюдения стройплощадки. "
    "Находишь каждую единицу техники и обводишь её рамкой. Описываешь только то, что видно. "
    "Отвечаешь строго одним JSON-объектом, без текста до и после него."
)


def build_prompt(classes: list[str] | None = None) -> str:
    eq = taxonomy.equipment()
    keys = [k for k in (classes or list(eq)) if k in eq]
    lines = "\n".join(f"- {k} — {eq[k].name}: {eq[k].look}" for k in keys)
    return f"""Найди на снимке всю строительную технику из справочника (ключ — название: как выглядит):
{lines}

Правила:
- Одна машина — один объект. Если машина похожа на два типа, выбери один, наиболее подходящий ключ.
- Легковые автомобили, людей, бытовки, штабели материалов и технику за пределами стройплощадки не включай.
- Частично видимую машину включай, если её тип узнаётся.
- bbox — [x1, y1, x2, y2] в нормированных координатах 0..1000: (0, 0) — левый верхний угол снимка,
  (1000, 1000) — правый нижний. Рамка охватывает машину целиком, вместе со стрелой и ковшом.
- conf — твоя уверенность от 0 до 1.
- working — true, если на снимке видны признаки работы (ковш в грунте или над кузовом, поднятый кузов,
  груз на крюке, развёрнутая стрела бетононасоса); false — если машина явно стоит без дела; null — если непонятно.
- Если техники нет — верни {{"objects": []}}.

Верни JSON строго такого вида:
{{"objects": [{{"class": "<ключ из справочника>", "bbox": [x1, y1, x2, y2], "conf": 0.9, "working": null}}]}}"""


class VlmDetector:
    """Детектор на VLM. `client` — для тестов и для подмены провайдера (DI)."""

    def __init__(self, client=None, model: str | None = None, provider: str = "zai",
                 max_side: int = 1280, max_tokens: int = 2000, classes: list[str] | None = None,
                 config: EquipmentConfig | None = None):
        self._client = client
        self._model = model
        self._provider_key = provider
        self.provider = Provider.EXTERNAL if provider == "zai" else Provider.LOCAL
        self.name = "glm" if provider == "zai" else "local_vlm"
        self.max_side = max_side
        self.max_tokens = max_tokens
        self.classes = [k for k in (classes or list(taxonomy.equipment())) if k in taxonomy.equipment()]
        self.config = config or EquipmentConfig()
        self.last_report: dict[str, Any] = {}

    @property
    def model_id(self) -> str:
        c = self._client
        return str(getattr(c, "model_id", None) or getattr(c, "model", None) or self._model or self.name)

    def _get_client(self):
        if self._client is None:
            self._client = _vlm.make_client(self._provider_key, model=self._model)
        return self._client

    def ready(self) -> tuple[bool, str]:
        try:
            ok, why = self._get_client().ready()
        except Exception as e:  # noqa: BLE001 — ready() не должен ронять страницу настроек
            return False, _human(e)
        return bool(ok), str(why or "")

    def prompt(self) -> str:
        return build_prompt(self.classes)

    def detect(self, image_bgr: np.ndarray, frame: FrameInfo | None = None) -> list[Detection]:
        if image_bgr is None or getattr(image_bgr, "ndim", 0) < 2:
            raise VLMError(f"{self._title()}: нет изображения")
        h, w = image_bgr.shape[:2]
        t0 = time.monotonic()
        try:
            client = self._get_client()
            url = _vlm.image_to_data_url(image_bgr, max_side=self.max_side)
            reply = client.ask_json(SYSTEM, url, self.prompt(), self.max_tokens)
        except VLMError as e:
            raise VLMError(f"{self._title()}: {e}") from e
        except Exception as e:  # noqa: BLE001 — любая ошибка клиента наружу только как VLMError
            raise VLMError(f"{self._title()}: {_human(e)}") from e

        data = _reply_data(reply)
        try:
            dets, dropped = parse_objects(data, w, h, source=self.model_id)
        except VLMError as e:
            raise VLMError(f"{self._title()}: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise VLMError(f"{self._title()}: ответ не разобран ({_human(e)})") from e

        self.last_report = {
            "model": self.model_id,
            "latency_ms": round(float(getattr(reply, "latency_ms", 0) or (time.monotonic() - t0) * 1000)),
            "objects": len(dets),
            "dropped": dropped,
            "text": str(getattr(reply, "text", ""))[:2000],
        }
        if dropped:
            log.info("%s: отброшено %d объектов: %s", self.model_id, len(dropped), "; ".join(dropped[:5]))
        return postprocess.clean(dets, w, h, self.config)

    def _title(self) -> str:
        return f"детектор {self.model_id}"


# --------------------------------------------------------------------------
# разбор ответа
# --------------------------------------------------------------------------

_LIST_KEYS = ("objects", "detections", "equipment", "items", "machines")


def _reply_data(reply) -> dict:
    """Reply.data, а если клиент отдал только текст — вытащить JSON из текста."""
    if isinstance(reply, dict):
        return reply
    data = getattr(reply, "data", None)
    if isinstance(data, dict) and data:
        return data
    text = getattr(reply, "text", None) or getattr(reply, "raw_text", None) or ""
    try:
        return _vlm.extract_json(str(text))
    except (ValueError, TypeError) as e:
        raise VLMError(f"модель не вернула JSON: {e}; начало ответа: {str(text)[:200]!r}") from e


def parse_objects(data: Any, width: int, height: int, source: str = "glm") -> tuple[list[Detection], list[str]]:
    """JSON ответа → детекции в пикселях XYWH + список причин отброса."""
    if not isinstance(data, dict):
        raise VLMError(f"ответ модели — не JSON-объект, а {type(data).__name__}")
    objs = next((data[k] for k in _LIST_KEYS if k in data), None)
    if objs is None:
        raise VLMError(f"в ответе нет списка objects (ключи: {', '.join(map(str, data)) or 'пусто'})")
    if not isinstance(objs, list):
        raise VLMError(f"objects в ответе — не список, а {type(objs).__name__}")

    boxes_raw, dropped, broken = [], [], 0
    for i, raw in enumerate(objs):
        if not isinstance(raw, dict):
            dropped.append(f"#{i}: не объект ({str(raw)[:40]!r})")
            broken += 1
            continue
        bbox = _bbox(raw.get("bbox", raw.get("box", raw.get("bbox_2d"))))
        if bbox is None:
            dropped.append(f"#{i}: нет рамки")
            broken += 1
            continue
        name = raw.get("class", raw.get("type", raw.get("label", raw.get("name"))))
        key = canonical_class(name) if isinstance(name, str) else None
        if key is None:
            dropped.append(f"#{i}: класс {name!r} не из справочника")
            continue
        boxes_raw.append((key, bbox, raw))
    if objs and broken == len(objs):
        raise VLMError("модель вернула объекты без рамок — разобрать нечего")

    scale = _coordinate_scale([b for _, b, _ in boxes_raw], width, height)
    out = []
    for key, (x1, y1, x2, y2), raw in boxes_raw:
        sx, sy = scale
        x1, x2 = sorted((x1 * sx, x2 * sx))
        y1, y2 = sorted((y1 * sy, y2 * sy))
        x1, x2 = min(max(x1, 0.0), 1000.0), min(max(x2, 0.0), 1000.0)
        y1, y2 = min(max(y1, 0.0), 1000.0), min(max(y2, 0.0), 1000.0)
        px = (x1 / 1000 * width, y1 / 1000 * height, (x2 - x1) / 1000 * width, (y2 - y1) / 1000 * height)
        if px[2] < 1 or px[3] < 1:
            dropped.append(f"{key}: вырожденная рамка")
            continue
        extra: dict[str, Any] = {"vlm_working": _bool(raw.get("working"))}
        if isinstance(raw.get("class"), str) and raw["class"] != key:
            extra["raw_cls"] = raw["class"]
        plate = raw.get("plate") or raw.get("number")
        if isinstance(plate, str) and plate.strip():
            extra["plate"] = plate.strip()
        out.append(Detection(cls=key, conf=_conf(raw.get("conf", raw.get("confidence", raw.get("score")))),
                             bbox=tuple(round(v, 1) for v in px), source=source, extra=extra))
    return out, dropped


def _bbox(v: Any) -> tuple[float, float, float, float] | None:
    """[x1,y1,x2,y2] | [[x1,y1],[x2,y2]] | {"x1":…} | строки с числами → четыре числа."""
    if isinstance(v, dict):
        v = [v.get("x1", v.get("xmin")), v.get("y1", v.get("ymin")), v.get("x2", v.get("xmax")), v.get("y2", v.get("ymax"))]
    if isinstance(v, str):
        v = v.replace("[", " ").replace("]", " ").replace(",", " ").split()
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in v):
        v = [v[0][0], v[0][1], v[1][0], v[1][1]]
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        nums = tuple(float(x) for x in v)
    except (TypeError, ValueError):
        return None
    return nums if all(math.isfinite(x) for x in nums) else None


def _coordinate_scale(bxs: list[tuple[float, ...]], width: int, height: int) -> tuple[float, float]:
    """Множители к шкале 0..1000. GLM отвечает в 0..1000, но другие VLM — в долях или пикселях."""
    if not bxs:
        return 1.0, 1.0
    top = max(max(b) for b in bxs)
    if top <= 1.0 + 1e-6:
        return 1000.0, 1000.0                        # доли кадра
    if top > 1000.0 + 1e-6 and width and height and top <= max(width, height) * 1.05:
        return 1000.0 / width, 1000.0 / height       # пиксели исходного кадра
    return 1.0, 1.0


def _conf(v: Any) -> float:
    try:
        c = float(v)
    except (TypeError, ValueError):
        return 0.5                                   # VLM не всегда даёт уверенность; нейтральное значение
    if not math.isfinite(c):
        return 0.5
    if 1.0 < c <= 100.0:
        c /= 100.0                                   # «85» — проценты
    return min(max(c, 0.0), 1.0)


def _bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "да", "1", "работает"):
            return True
        if s in ("false", "no", "нет", "0", "стоит"):
            return False
    return None


def _human(e: Exception) -> str:
    msg = str(e).strip()
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__
