"""Уточнение подтипа грузовой техники по кропу: zero-shot SigLIP.

Зачем. Обученный детектор (12 классов, см. models/README.md) не знает двух
обязательных по ТЗ классов — «грузовик» (бортовой, длинномер, трал) и
«кран-манипулятор»: открытых размеченных данных для них нет, а бортовые
грузовики в обучающих датасетах сидят внутри «самосвала». Зато детектор
уверенно находит саму машину и её «семейство». Поэтому рамки группы
грузовиков (самосвал, автобетоносмеситель, автокран, автобетононасос)
дополнительно классифицируются по кропу: SigLIP сравнивает кроп с
текстовыми описаниями шести подтипов (та же модель, что у модели Б; если
веб-слой передаст общий эмбеддер, в памяти будет одна копия).

Осторожность. На своих классах детектор точнее zero-shot, поэтому класс
меняется только при уверенном перевесе (пороги — в EquipmentConfig):
вероятность нового подтипа не ниже `refine_min_prob` и выше вероятности
класса детектора на `refine_margin`; если новый класс детектор и сам умеет
(самосвал → автобетоносмеситель), перевес нужен больше —
`refine_margin_known`: детектор мог так ответить и не ответил. Исходный
класс, его уверенность и оценки подтипов остаются в
`Detection.extra["refine"]` — это видно в «Проверить снимок» и в отладке.

Модель грузится лениво, при первой рамке-грузовике; эмбеддинги текстов
считаются один раз. Любая ошибка модели (нет интернета для скачивания,
нехватка памяти) не роняет детекцию: уточнение выключается, детекции
возвращаются как есть, а `error` объясняет почему (детектор показывает это
в ready()). torch/transformers импортируются только внутри загрузки —
модуль тестируется на фейковом эмбеддере.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import logging
import os
import threading
from typing import Any, Protocol

import cv2
import numpy as np

from core.contracts import Detection

from . import boxes
from .config import EquipmentConfig

log = logging.getLogger(__name__)

DEFAULT_MODEL = "google/siglip2-base-patch16-224"

# Рамки каких классов детектора уточняем.
REFINE_FROM = ("dump_truck", "concrete_mixer", "mobile_crane", "concrete_pump")

# Подтип → описания (несколько — если у класса несколько непохожих обликов:
# бортовой и седельный тягач с полуприцепом). Оценка класса — лучшая из его
# описаний, поэтому число описаний не даёт классу форы.
#
# Формулировки и пороги подобраны на testset/equipment (фото с Wikimedia/Flickr,
# 56 рамок группы грузовиков, см. models/README.md): у первого варианта
# («truck with a knuckle-boom crane …», два шаблона фраз) автокраны уходили в
# манипуляторы (6 из 12), у этого — 0 из 12 при тех же 8 из 12 манипуляторов.
# Главное отличие — у манипулятора явно «небольшая складная установка на
# бортовом грузовике», у автокрана — «длинная телескопическая стрела».
DESCRIPTIONS: dict[str, tuple[str, ...]] = {
    "dump_truck": ("dump truck with an open tipping body for soil",),
    "truck": ("flatbed truck with a flat cargo platform", "semi-trailer truck with a long flatbed trailer"),
    "crane_manipulator": ("flatbed truck with a small folding knuckle-boom crane mounted behind the cab",),
    "concrete_mixer": ("concrete mixer truck with a rotating drum",),
    "mobile_crane": ("mobile truck crane with a long telescopic boom and outriggers",),
    "concrete_pump": ("concrete pump truck with a multi-section folding boom",),
}
# Шаблон фразы. Ансамбль из двух шаблонов («… at a construction site») на
# testset был хуже одного: второй тянул всё к «стройке», а не к облику машины.
TEMPLATES = ("a photo of a {}.",)


class Embedder(Protocol):
    """Что нужно уточнению от модели (тот же протокол, что у эмбеддера модели Б).
    Векторы — L2-нормированные строки."""
    name: str
    logit_scale: float

    def embed_images(self, images_rgb: list[np.ndarray]) -> np.ndarray: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _features(out: Any):
    """transformers 5 отдаёт из get_*_features объект с pooler_output, 4.x — тензор."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if isinstance(out, (tuple, list)):
        return out[1] if len(out) > 1 else out[0]
    return out


class SiglipEmbedder:
    """SigLIP / SigLIP2 (и CLIP) через transformers: get_image_features / get_text_features.

    Одна копия модели на процесс для каждого (имя, устройство): несколько
    детекторов (пересоздание после смены настроек) не грузят её повторно.
    """

    _shared: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
    _shared_lock = threading.Lock()

    def __init__(self, model_name: str | None = None, device: str | None = None, threads: int | None = None):
        self.name = model_name or os.environ.get("EQUIPMENT_REFINE_MODEL") or DEFAULT_MODEL
        self.device = device or os.environ.get("EQUIPMENT_REFINE_DEVICE") or None
        self.threads = threads
        self.logit_scale = 100.0
        self._model = self._proc = self._torch = None
        self._lock = threading.Lock()
        self._texts: dict[str, np.ndarray] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoProcessor

            if self.threads:
                torch.set_num_threads(int(self.threads))
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            key = (self.name, device)
            with self._shared_lock:
                if key not in self._shared:
                    model = AutoModel.from_pretrained(self.name).eval().to(device)
                    proc = AutoProcessor.from_pretrained(self.name)
                    self._shared[key] = (model, proc, torch)
                model, proc, torch_mod = self._shared[key]
            scale = getattr(model, "logit_scale", None)
            if scale is not None:
                self.logit_scale = float(scale.detach().float().exp().cpu())
            self.device, self._proc, self._torch = device, proc, torch_mod
            self._model = model

    def embed_images(self, images_rgb: list[np.ndarray]) -> np.ndarray:
        self._load()
        if self.threads:
            self._torch.set_num_threads(int(self.threads))   # его мог переставить ultralytics
        from PIL import Image

        pil = [Image.fromarray(np.ascontiguousarray(im)) for im in images_rgb]
        with self._torch.inference_mode():
            inputs = self._proc(images=pil, return_tensors="pt").to(self.device)
            feats = _features(self._model.get_image_features(**inputs))
        return _normalize(feats.float().cpu().numpy())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._texts]
        if missing:
            self._load()
            with self._torch.inference_mode():
                # SigLIP обучался на текстах, дополненных до 64 токенов, — иначе эмбеддинги хуже.
                inputs = self._proc(text=missing, padding="max_length", max_length=64, truncation=True,
                                    return_tensors="pt").to(self.device)
                feats = _features(self._model.get_text_features(**inputs))
            for t, v in zip(missing, _normalize(feats.float().cpu().numpy())):
                self._texts[t] = v
        return np.stack([self._texts[t] for t in texts])


def transformers_available() -> bool:
    return importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("torch") is not None


def weights_cached(model_name: str) -> bool | None:
    """Лежат ли веса в кэше HF (без импорта torch). None — проверить нечем."""
    if os.path.isdir(model_name):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        return isinstance(try_to_load_from_cache(model_name, "config.json"), str)
    except Exception:  # noqa: BLE001 — только подсказка для ready()
        return None


class CropRefiner:
    """Уточняет подтип рамок группы грузовиков. Потокобезопасен (замок на вызов модели)."""

    def __init__(self, embedder: Embedder | None = None, config: EquipmentConfig | None = None,
                 model_name: str | None = None, device: str | None = None, threads: int | None = None,
                 descriptions: dict[str, tuple[str, ...]] | None = None,
                 templates: tuple[str, ...] | None = None, refine_from: tuple[str, ...] = REFINE_FROM):
        self.config = config or EquipmentConfig()
        self.descriptions = dict(descriptions or DESCRIPTIONS)
        self.templates = tuple(templates or TEMPLATES)
        self.refine_from = tuple(refine_from)
        self._embedder = embedder
        self._model_name, self._device, self._threads = model_name, device, threads
        self._class_vecs: tuple[list[str], np.ndarray, np.ndarray] | None = None
        self._lock = threading.Lock()
        self.error = ""

    # ------------------------------------------------------------------

    @property
    def targets(self) -> list[str]:
        """Какие классы может выдать уточнение."""
        return list(self.descriptions)

    @property
    def model_name(self) -> str:
        if self._embedder is not None:
            return str(getattr(self._embedder, "name", "embedder"))
        # Не STAGE_CLIP_MODEL модели Б: формулировки и пороги подобраны под эту модель.
        return self._model_name or os.environ.get("EQUIPMENT_REFINE_MODEL") or DEFAULT_MODEL

    def available(self) -> tuple[bool, str]:
        """(можно ли уточнять, пояснение) — без загрузки модели."""
        if self.error:
            return False, self.error
        if self._embedder is not None:
            return True, ""
        if not transformers_available():
            return False, "не установлены transformers/torch"
        if weights_cached(self.model_name) is False:
            return True, f"веса {self.model_name} скачаются при первом грузовике в кадре (нужен интернет)"
        return True, ""

    def refine(self, image_bgr: np.ndarray, detections: list[Detection],
               known: set[str] | frozenset[str] | None = None) -> list[Detection]:
        """Новый список: рамки группы грузовиков с уточнённым классом (исходные не меняются).

        known — классы, которые детектор умеет сам: переход в них требует
        большего перевеса (`refine_margin_known`).
        """
        if self.error or image_bgr is None or not detections:
            return list(detections)
        if self._embedder is None and not transformers_available():
            return list(detections)                     # тихо выключено: ready() детектора это объясняет
        cfg = self.config
        picks, crops = [], []
        for i, d in enumerate(detections):
            if (d.cls not in self.refine_from or "refine" in d.extra
                    or min(d.bbox[2], d.bbox[3]) < cfg.refine_min_side_px):
                continue
            crop = square_crop(image_bgr, d.bbox, cfg.refine_pad_frac)
            if crop is None:
                continue
            picks.append(i)
            crops.append(crop)
        if not crops:
            return list(detections)
        try:
            with self._lock:
                probs, classes = self._probabilities(crops)
        except Exception as e:  # noqa: BLE001 — уточнение не должно ронять детекцию
            self.error = f"уточнение подтипа грузовиков выключено: {type(e).__name__}: {e}"[:300]
            log.warning("%s", self.error)
            return list(detections)

        out = list(detections)
        known = set(known or ())
        for row, i in zip(probs, picks):
            d = detections[i]
            scores = {c: round(float(p), 3) for c, p in zip(classes, row)}
            best = classes[int(np.argmax(row))]
            p_best, p_det = scores[best], scores.get(d.cls, 0.0)
            margin = cfg.refine_margin_known if best in known else cfg.refine_margin
            change = best != d.cls and p_best >= cfg.refine_min_prob and p_best - p_det >= margin
            info = {"det_cls": d.cls, "det_conf": round(float(d.conf), 3), "scores": scores,
                    "changed": change, "model": self.model_name}
            out[i] = dataclasses.replace(d, cls=best if change else d.cls, extra={**d.extra, "refine": info})
        return out

    # ------------------------------------------------------------------

    def _probabilities(self, crops_bgr: list[np.ndarray]) -> tuple[np.ndarray, list[str]]:
        emb = self._get_embedder()
        classes, text_vecs, owner = self._text_vectors(emb)
        img = emb.embed_images([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr])
        sims = _normalize(img) @ text_vecs.T                         # (кропы, описания)
        # Оценка класса — лучшее из его описаний; вероятности — softmax по классам.
        logits = np.full((sims.shape[0], len(classes)), -np.inf, dtype=np.float64)
        for j, k in enumerate(owner):
            logits[:, k] = np.maximum(logits[:, k], sims[:, j] * float(emb.logit_scale))
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True), classes

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SiglipEmbedder(self._model_name, self._device, self._threads)
        return self._embedder

    def _text_vectors(self, emb: Embedder) -> tuple[list[str], np.ndarray, list[int]]:
        """Эмбеддинг каждого описания = среднее по шаблонам. Считается один раз."""
        if self._class_vecs is None:
            classes = list(self.descriptions)
            vecs, owner = [], []
            for k, cls in enumerate(classes):
                for desc in self.descriptions[cls]:
                    phrases = [t.format(desc) for t in self.templates]
                    vecs.append(_normalize(emb.embed_texts(phrases).mean(axis=0)))
                    owner.append(k)
            self._class_vecs = (classes, np.stack(vecs), owner)
        return self._class_vecs


def square_crop(image_bgr: np.ndarray, bbox: boxes.Box, pad_frac: float = 0.06) -> np.ndarray | None:
    """Кроп рамки с полями, дополненный до квадрата серым.

    SigLIP сжимает вход в квадрат 224×224; длинномер 3:1 без дополнения
    превратился бы в «кубик» и перестал быть похож на длинномер. Дополняем
    нейтральным серым, а не соседними пикселями кадра: рядом часто стоит
    другая машина, и классифицировать надо эту.
    """
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = boxes.clip(boxes.expand(bbox, pad_frac), w, h)
    x0, y0, x1, y1 = int(x), int(y), int(round(x + bw)), int(round(y + bh))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = image_bgr[y0:y1, x0:x1]
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    canvas = np.full((side, side, 3), 127, dtype=np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return canvas
