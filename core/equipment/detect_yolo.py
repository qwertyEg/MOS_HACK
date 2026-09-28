"""LOCAL-детектор техники: ultralytics YOLO, работает на ноутбуке без интернета.

Веса. Свои, дообученные на датасетах строительной техники (обучение — на
GPU-сервере, см. models/README.md): `models/equipment_yolo.pt` +
`models/equipment_classes.json` или путь из env EQUIPMENT_WEIGHTS. Если
своих весов нет — YOLO-World (zero-shot по текстовым описаниям классов):
хуже дообученной модели, но сервис работает из коробки. `ready()` честно
говорит, что именно используется и чего не хватает.

Ночь. Работы идут круглосуточно, модель А обрабатывает и ночные кадры.
Тёмный кадр перед детекцией усиливается: гамма + CLAHE по каналу яркости L
в LAB (цвет не трогаем — по цвету отличают машины при слиянии камер).

Крупные кадры. На 4K-кадре дальняя машина занимает десяток пикселей после
уменьшения до входа сети; опциональный тайлинг прогоняет кадр кусками с
перекрытием плюс целиком (крупные машины, разрезанные границей тайла).

ultralytics/torch импортируются лениво — модуль импортируется и тестируется
без них (тесты подставляют фейковую модель).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core import taxonomy
from core.contracts import Detection, FrameInfo, Provider

from . import boxes, postprocess
from .classes import canonical_class
from .config import EquipmentConfig

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "equipment_yolo.pt"
DEFAULT_CLASSES = MODELS_DIR / "equipment_classes.json"
WORLD_WEIGHTS = "yolov8s-worldv2.pt"

# Текстовые классы для YOLO-World. Короткие англ. существительные работают
# лучше длинных описаний; формулировки — по полю look из checklist.json.
WORLD_PROMPTS: dict[str, tuple[str, ...]] = {
    "excavator": ("excavator", "tracked excavator with bucket"),
    "dump_truck": ("dump truck", "tipper truck"),
    "bulldozer": ("bulldozer",),
    "roller": ("road roller", "compactor roller"),
    "concrete_mixer": ("concrete mixer truck", "cement mixer truck"),
    "truck": ("flatbed truck", "semi-trailer truck"),
    "mobile_crane": ("mobile crane", "truck crane with telescopic boom"),
    "crane_manipulator": ("truck with loader crane", "knuckle boom crane truck"),
    "tower_crane": ("tower crane",),
    "crawler_crane": ("crawler crane with lattice boom",),
    "concrete_pump": ("concrete pump truck",),
    "drilling_rig": ("drilling rig", "piling rig with mast"),
    "pile_driver": ("pile driver",),
    "wheel_loader": ("wheel loader", "front loader"),
    "skid_steer": ("skid steer loader",),
    "backhoe_loader": ("backhoe loader",),
    "telehandler": ("telehandler",),
    "grader": ("motor grader",),
    "asphalt_paver": ("asphalt paver",),
    "aerial_platform": ("boom lift", "aerial work platform"),
    "facade_hoist": ("construction hoist",),
}
# Классы-«поглотители»: без них zero-shot охотно называет легковушку грузовиком.
WORLD_DISTRACTORS = ("car", "person", "van")


def world_vocabulary() -> tuple[list[str], dict[int, str | None]]:
    """Словарь текстовых классов YOLO-World и перевод индекса обратно в ключ (None — поглотитель)."""
    vocab = [p for ps in WORLD_PROMPTS.values() for p in ps] + list(WORLD_DISTRACTORS)
    key_of = {p: k for k, ps in WORLD_PROMPTS.items() for p in ps}
    return vocab, {i: key_of.get(p) for i, p in enumerate(vocab)}


def is_dark(image_bgr: np.ndarray, threshold: float = 70.0) -> bool:
    small = cv2.resize(image_bgr, (160, 90), interpolation=cv2.INTER_AREA) if image_bgr.shape[1] > 160 else image_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return float(gray.mean()) < threshold


def enhance_low_light(image_bgr: np.ndarray, clip_limit: float = 2.0, gamma: float = 0.6) -> np.ndarray:
    """Гамма (поднимает тени) + CLAHE по яркости L в LAB (локальный контраст), цвет сохраняется."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    l = cv2.LUT(l, lut)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def load_class_map(path: str | Path) -> tuple[dict[int, str], dict[str, str | None]]:
    """equipment_classes.json → (индекс → имя класса, карта «имя датасета → ключ»).

    Поддерживаемые формы:
      {"names": {"0": "excavator", "1": "dump_truck"}}          — имена уже ключи словаря
      {"names": ["excavator", "dump_truck"]}
      {"names": {"0": "Dump Truck", "1": "Worker"},
       "map": {"Dump Truck": "dump_truck", "Worker": null}}      — датасетные имена + перевод
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("names", data) if isinstance(data, dict) else data
    if isinstance(raw, list):
        names = {i: str(n) for i, n in enumerate(raw)}
    elif isinstance(raw, dict):
        names = {int(k): str(v) for k, v in raw.items() if str(k).lstrip("-").isdigit()}
    else:
        raise ValueError(f"{path}: поле names должно быть списком или словарём")
    mapping = data.get("map", {}) if isinstance(data, dict) else {}
    return names, {str(k): (None if v is None else str(v)) for k, v in mapping.items()}


class YoloDetector:
    name = "yolo"
    provider = Provider.LOCAL

    _shared: dict[str, Any] = {}              # путь весов → загруженная модель (одна на процесс)
    _shared_lock = threading.Lock()

    def __init__(self, weights: str | Path | None = None, classes_json: str | Path | None = None,
                 conf: float = 0.25, iou: float = 0.5, imgsz: int | None = None, device: str | None = None,
                 tile: bool = False, enhance: bool | str = "auto", dark_threshold: float = 70.0,
                 model: Any = None, config: EquipmentConfig | None = None):
        self.weights = Path(weights) if weights else (Path(os.environ["EQUIPMENT_WEIGHTS"])
                                                      if os.environ.get("EQUIPMENT_WEIGHTS") else DEFAULT_WEIGHTS)
        self.classes_json = Path(classes_json) if classes_json else self._guess_classes_json()
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.tile, self.enhance, self.dark_threshold = tile, enhance, dark_threshold
        self.config = config or EquipmentConfig()
        self._model = model
        self._index_to_key: dict[int, str | None] | None = None
        self.world = False
        self._lock = threading.Lock()

    def _guess_classes_json(self) -> Path:
        env = os.environ.get("EQUIPMENT_CLASSES")
        if env:
            return Path(env)
        sibling = self.weights.with_name("equipment_classes.json")
        return sibling if sibling.exists() else DEFAULT_CLASSES

    # ------------------------------------------------------------------

    def ready(self) -> tuple[bool, str]:
        if self._model is not None:
            return True, ""
        if importlib.util.find_spec("ultralytics") is None:
            return False, "не установлен ultralytics — pip install -r requirements-ml.txt"
        if self.weights.exists():
            if not self.classes_json.exists():
                return True, f"нет {self.classes_json.name} — имена классов берутся из весов"
            return True, ""
        world_local = (MODELS_DIR / WORLD_WEIGHTS).exists()
        note = "" if world_local else "; при первом запуске скачает yolov8s-worldv2.pt (нужен интернет)"
        return True, (f"свои веса не найдены ({self.weights}) — работает YOLO-World без дообучения, "
                      f"точность ниже{note}")

    @property
    def supported_classes(self) -> list[str]:
        """Какие ключи словаря умеет этот детектор — для /api/settings.

        Без загрузки модели (torch не трогаем): по equipment_classes.json, а
        без своих весов — по словарю YOLO-World. Если json нет, а веса есть,
        список станет известен после первого detect().
        """
        self._ensure_names()
        mapping = self._index_to_key
        if mapping is None:
            if self.weights.exists() and self.classes_json.exists():
                names, extra = load_class_map(self.classes_json)
                mapping = {i: canonical_class(n, extra) for i, n in names.items()}
            elif not self.weights.exists():
                mapping = world_vocabulary()[1]
            else:
                mapping = {}
        order = list(taxonomy.equipment())
        return sorted({k for k in mapping.values() if k}, key=order.index)

    def detect(self, image_bgr: np.ndarray, frame: FrameInfo | None = None) -> list[Detection]:
        self._load()
        h, w = image_bgr.shape[:2]
        img = image_bgr
        if self.enhance is True or (self.enhance == "auto" and
                                    ((frame is not None and frame.is_night) or is_dark(image_bgr, self.dark_threshold))):
            img = enhance_low_light(image_bgr)
        if self.tile and max(h, w) > 1600:
            dets = self._tiled(img)
        else:
            dets = self._run(img, 0, 0)
        return postprocess.clean(dets, w, h, self.config)

    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            self._ensure_names()
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from ultralytics import YOLO  # тяжёлый импорт (torch) — только здесь
            except ImportError as e:
                raise RuntimeError("не установлен ultralytics — pip install -r requirements-ml.txt") from e
            key = str(self.weights) if self.weights.exists() else f"world:{WORLD_WEIGHTS}"
            with self._shared_lock:
                model = self._shared.get(key)
                if model is None:
                    if self.weights.exists():
                        model = YOLO(str(self.weights))
                    else:
                        local = MODELS_DIR / WORLD_WEIGHTS
                        model = YOLO(str(local) if local.exists() else WORLD_WEIGHTS)
                        model.set_classes(world_vocabulary()[0])
                    self._shared[key] = model
            self.world = key.startswith("world:")
            self._model = model
            self._ensure_names()

    def _ensure_names(self) -> None:
        if self._index_to_key is not None or self._model is None:
            return
        if self.world:
            self._index_to_key = world_vocabulary()[1]
            return
        names: dict[int, str] = {int(k): str(v) for k, v in dict(getattr(self._model, "names", {}) or {}).items()}
        mapping: dict[str, str | None] = {}
        if self.classes_json.exists():
            json_names, mapping = load_class_map(self.classes_json)
            names = json_names or names
        self._index_to_key = {i: canonical_class(n, mapping) for i, n in names.items()}
        unknown = [n for i, n in names.items() if self._index_to_key[i] is None]
        if unknown:
            log.info("классы детектора без ключа словаря (отбрасываются): %s", ", ".join(unknown))

    def _run(self, img: np.ndarray, ox: float, oy: float) -> list[Detection]:
        kw: dict[str, Any] = {"conf": self.conf, "iou": self.iou, "verbose": False}
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        if self.device:
            kw["device"] = self.device
        results = self._model.predict(img, **kw)
        out = []
        for r in results[:1]:
            b = r.boxes
            xyxy, confs, cls_idx = _np(b.xyxy), _np(b.conf), _np(b.cls).astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, cls_idx):
                key = (self._index_to_key or {}).get(int(k))
                if key is None:
                    continue
                out.append(Detection(cls=key, conf=float(c),
                                     bbox=boxes.xyxy_to_xywh(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                                     source="yolo-world" if self.world else "yolo"))
        return out

    def _tiled(self, img: np.ndarray, tile: int = 1280, overlap: float = 0.2) -> list[Detection]:
        h, w = img.shape[:2]
        dets = self._run(img, 0, 0)                  # целиком — крупные машины
        step = int(tile * (1 - overlap))
        for y in range(0, max(1, h - int(tile * overlap)), step):
            for x in range(0, max(1, w - int(tile * overlap)), step):
                x0, y0 = min(x, max(0, w - tile)), min(y, max(0, h - tile))
                dets += self._run(img[y0:y0 + tile, x0:x0 + tile], x0, y0)
        return _merge_same_class(dets)


def _merge_same_class(dets: list[Detection], iou: float = 0.5, ioa: float = 0.8) -> list[Detection]:
    """Склейка результатов тайлов: одна машина из соседних тайлов и из прохода целиком.
    По IoA тоже: кусок машины у границы тайла лежит внутри полной рамки."""
    out: list[Detection] = []
    for d in sorted(dets, key=lambda d: (boxes.area(d.bbox), d.conf), reverse=True):
        if any(k.cls == d.cls and (boxes.iou(k.bbox, d.bbox) >= iou or boxes.ioa_min(k.bbox, d.bbox) >= ioa)
               for k in out):
            continue
        out.append(d)
    return out


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)
