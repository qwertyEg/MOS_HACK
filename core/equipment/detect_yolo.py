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

Подтипы грузовиков. Обученный детектор не знает «грузовик» (бортовой,
длинномер) и «кран-манипулятор» — обязательные по ТЗ. Рамки группы
грузовиков уточняются zero-shot классификацией кропа (refine.py, SigLIP);
включено по умолчанию, если установлен transformers (выключить —
refine=False или EQUIPMENT_REFINE=0). Без transformers уточнение тихо
выключено, ready() и supported_classes это отражают.

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
from .refine import CropRefiner

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
      {"model": "equipment_yolo.pt", "names": {"0": "excavator", ...}, "imgsz": 640, ...}
                                                                 — как пишет обучение на сервере
      {"names": {"0": "excavator", "1": "dump_truck"}}          — имена уже ключи словаря
      {"names": ["excavator", "dump_truck"]}
      {"names": {"0": "Dump Truck", "1": "Worker"},
       "map": {"Dump Truck": "dump_truck", "Worker": null}}      — датасетные имена + перевод
    Остальные поля (метрики, источники данных) детектору не нужны.
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
                 model: Any = None, config: EquipmentConfig | None = None,
                 refine: bool | None = None, refiner: CropRefiner | None = None, refine_model: str | None = None,
                 threads: int | None = None):
        """weights — путь к .pt/.onnx, к каталогу с весами или к equipment_classes.json
        (тогда веса — его поле "model" рядом с ним); по умолчанию env EQUIPMENT_WEIGHTS,
        затем models/equipment_yolo.pt. model — готовая модель (тесты).
        threads — потоков torch на CPU (env EQUIPMENT_THREADS): ultralytics на CPU сам
        ставит «все ядра минус одно», на общей машине это мешает соседям."""
        env_weights = os.environ.get("EQUIPMENT_WEIGHTS")
        asked = Path(weights) if weights else Path(env_weights) if env_weights else DEFAULT_WEIGHTS
        self.weights, json_from_weights = _resolve_weights(asked)
        self.weights_note = ""
        if not self.weights.exists() and asked.parent.is_dir():
            # Указанного файла нет (старое имя в настройках — models/equipment.pt,
            # опечатка), а в том же каталоге лежат наши веса: они всё равно лучше
            # YOLO-World. Какие — по equipment_classes.json рядом.
            cand, cj = _resolve_weights(asked.parent)
            if cand.exists():
                self.weights_note = f"весов {asked.name} нет — взяты {cand.name} из того же каталога"
                log.warning("%s", self.weights_note)
                self.weights, json_from_weights = cand, cj
        self.classes_json = (Path(classes_json) if classes_json else json_from_weights
                             or self._guess_classes_json())
        self.conf, self.iou, self.device = conf, iou, device
        env_threads = os.environ.get("EQUIPMENT_THREADS", "").strip()
        self.threads = threads or (int(env_threads) if env_threads.isdigit() and int(env_threads) > 0 else None)
        self.imgsz = imgsz or self._meta().get("imgsz")
        self.tile, self.enhance, self.dark_threshold = tile, enhance, dark_threshold
        self.config = config or EquipmentConfig()
        self._model = model
        self._index_to_key: dict[int, str | None] | None = None
        self.world = False
        self._lock = threading.Lock()
        if refine is None:
            refine = os.environ.get("EQUIPMENT_REFINE", "1").strip().lower() not in ("0", "false", "no", "off", "нет")
        if refiner is None and refine:
            refiner = CropRefiner(config=self.config, model_name=refine_model, device=_torch_device(device),
                                  threads=self.threads)
        self.refiner = refiner

    def _guess_classes_json(self) -> Path:
        env = os.environ.get("EQUIPMENT_CLASSES")
        if env:
            return Path(env)
        sibling = self.weights.with_name("equipment_classes.json")
        return sibling if sibling.exists() else DEFAULT_CLASSES

    def _meta(self) -> dict[str, Any]:
        """Поля equipment_classes.json помимо имён (imgsz обучения и т.п.); {} — нет файла или он битый."""
        try:
            data = json.loads(self.classes_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @property
    def refine_active(self) -> bool:
        """Уточняются ли подтипы грузовиков (есть уточнитель и ему есть на чём работать)."""
        return self.refiner is not None and self.refiner.available()[0]

    # ------------------------------------------------------------------

    def ready(self) -> tuple[bool, str]:
        ok, note = self._ready_detector()
        if not ok:
            return ok, note
        extra = self._refine_note()
        return True, "; ".join(n for n in (note, extra) if n)

    def _ready_detector(self) -> tuple[bool, str]:
        if self._model is not None:
            return True, ""
        if importlib.util.find_spec("ultralytics") is None:
            return False, "не установлен ultralytics — pip install -r requirements-ml.txt"
        if self.weights.exists():
            if not self.classes_json.exists():
                return True, "; ".join(filter(None, (
                    self.weights_note, f"нет {self.classes_json.name} — имена классов берутся из весов")))
            return True, self.weights_note
        world_local = (MODELS_DIR / WORLD_WEIGHTS).exists()
        note = "" if world_local else "; при первом запуске скачает yolov8s-worldv2.pt (нужен интернет)"
        return True, (f"свои веса не найдены ({self.weights}) — работает YOLO-World без дообучения, "
                      f"точность ниже{note}")

    def _refine_note(self) -> str:
        if self.refiner is None:
            return ""
        ok, why = self.refiner.available()
        if ok:
            return f"уточнение подтипа грузовиков: {why}" if why else ""
        lost = [k for k in self.refiner.targets if k not in self._base_classes()]
        tail = f" — не распознаются: {', '.join(lost)}" if lost else ""
        return f"уточнение подтипа грузовиков выключено ({why}){tail}"

    @property
    def supported_classes(self) -> list[str]:
        """Какие ключи словаря умеет этот детектор — для /api/settings.

        Без загрузки модели (torch не трогаем): по equipment_classes.json, а
        без своих весов — по словарю YOLO-World. Если json нет, а веса есть,
        список станет известен после первого detect(). С уточнением подтипов
        (refine.py) сюда добавляются truck и crane_manipulator.
        """
        keys = self._base_classes()
        if self.refine_active and keys & set(self.refiner.refine_from):
            keys |= set(self.refiner.targets)
        order = list(taxonomy.equipment())
        return sorted(keys, key=order.index)

    def _base_classes(self) -> set[str]:
        """Классы самой сети (без уточнения подтипов)."""
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
        return {k for k in mapping.values() if k}

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
        dets = postprocess.clean(dets, w, h, self.config)
        if self.refiner is not None and not self.refiner.error:
            # После NMS: одна машина — один кроп. Уточнённый класс остаётся в той же
            # группе путаницы, повторная чистка только применит порог его класса.
            dets = self.refiner.refine(img, dets, known=self._base_classes())
            dets = postprocess.clean(dets, w, h, self.config)
        return dets

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
        # Сеть отсекает по самому низкому порогу из настроек, иначе порог класса
        # ниже общего (башенный кран 0.15 при общем 0.25) никогда не сработал бы;
        # точный порог класса применяется ниже.
        per_class = self.config.conf_by_class
        kw: dict[str, Any] = {"conf": min([self.conf, *per_class.values()]), "iou": self.iou, "verbose": False}
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        if self.device:
            kw["device"] = self.device
        results = self._model.predict(img, **kw)
        if self.threads:
            # predict() на CPU переставляет число потоков torch (select_device) — возвращаем своё.
            import torch
            torch.set_num_threads(self.threads)
        out = []
        for r in results[:1]:
            b = r.boxes
            xyxy, confs, cls_idx = _np(b.xyxy), _np(b.conf), _np(b.cls).astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, cls_idx):
                key = (self._index_to_key or {}).get(int(k))
                if key is None or float(c) < per_class.get(key, self.conf):
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


def _resolve_weights(path: Path) -> tuple[Path, Path | None]:
    """(путь к весам, equipment_classes.json, если он однозначно следует из пути)."""
    if path.suffix.lower() == ".json":
        try:
            name = json.loads(path.read_text(encoding="utf-8")).get("model") or DEFAULT_WEIGHTS.name
        except (OSError, ValueError, AttributeError):
            name = DEFAULT_WEIGHTS.name
        return path.with_name(Path(str(name)).name), path
    if path.is_dir():
        cj = path / DEFAULT_CLASSES.name
        name = DEFAULT_WEIGHTS.name
        if cj.exists():
            try:
                name = Path(str(json.loads(cj.read_text(encoding="utf-8")).get("model") or name)).name
            except (OSError, ValueError, AttributeError):
                pass
        return path / name, (cj if cj.exists() else None)
    return path, None


def _torch_device(device: str | None) -> str | None:
    """Устройство ultralytics («0», «cpu», «cuda:1») → устройство torch."""
    if device is None:
        return None
    d = str(device).strip()
    return f"cuda:{d}" if d.isdigit() else d


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)
