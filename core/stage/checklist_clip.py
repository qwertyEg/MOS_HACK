"""LOCAL модель Б: чек-лист по сходству кадра с текстом (SigLIP / CLIP), без интернета и без GPU.

Для каждого признака из `reference/checklist.json` есть позитивные и негативные
английские формулировки (`core/stage/sign_prompts.json`). Вероятность «признак есть»

    p = σ(k · (s⁺ − s⁻)),

где s± — косинусное сходство эмбеддинга кадра с усреднённым эмбеддингом позитивных
/ негативных формулировок. При k = logit_scale модели это вероятность двухклассового
zero-shot-выбора между формулировками, но на реальных кадрах такой выбор слишком
самоуверен: у SigLIP2 k ≈ 113, и при порогах 0.62/0.38 «да» ставилось уже при
разности сходства 0.004 — модель отвечала «да» почти на всё (доля «не уверен» 7–10 %),
и хронология уезжала на поздние этапы (расчистка участка → «монолит», карьер → «нет
данных»). Поэтому k и пороги откалиброваны интегратором по сохранённым разностям
сходства 1896 кадров 9 демо-объектов с разметкой этапов «на глаз»: k = 35,
p ≥ 0.8 → «да» (разность ≥ 0.04), p ≤ 0.46 → «нет» (разность ≤ −0.0046), иначе
«не уверен» — такой ответ не голосует за этап (см. scoring). Доля дней, где фронт
совпал с разметкой, выросла с 0.32 до 0.68 (карьер, котлован двух камер, расчистка,
сборный каркас стали верными; асфальтирование парковки стало неверным). Выборка мала
и та же, на которой подбирали, — оценка оптимистична. `scale=None` возвращает k модели.
Пороги настраиваются в конфиге и в UI (страница «Настройки») и переопределяются на
признак (в конфиге или в sign_prompts.json).

Этап целиком: те же эмбеддинги против описаний 8 этапов → softmax → stage_likelihood.
Он только для показа и выбора кандидатов: этап открывают ответы чек-листа (scoring).

Тяжёлые зависимости (torch, transformers / open_clip) импортируются лениво внутри
эмбеддера; эмбеддер подменяется (`embedder=`), и тесты идут на фейке без torch.
Эмбеддинги текстов считаются батчем один раз и кэшируются в эмбеддере — кадр
стоит один проход image-энкодера (≈0.1–0.3 с на CPU для base-модели).

Маска применяется до классификации: `context["mask"]` (DynamicMask или bool-массив
«видимое»), режим `context["mask_mode"]` (по умолчанию darken).

Известные ограничения: SigLIP на 224×224 не различает мелочь на обзорном кадре
(арматура, оголовки свай) — для этого есть `tile_grid` (сетка фрагментов, по
признаку берётся максимум); формулировки и пороги не откалиброваны на размеченных
московских кадрах — калибровка — первое, что делать при появлении разметки.
"""
from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from core import taxonomy
from core.contracts import Answer, ChecklistResult, FrameInfo, Provider
from core.stage.mask import masked_for_model

PROMPTS_PATH = Path(__file__).resolve().parent / "sign_prompts.json"
DEFAULT_MODEL = "google/siglip2-base-patch16-224"


@dataclass
class ClipConfig:
    model_name: str = field(default_factory=lambda: os.getenv("STAGE_CLIP_MODEL", DEFAULT_MODEL))
    backend: str = field(default_factory=lambda: os.getenv("STAGE_CLIP_BACKEND", "transformers"))  # | open_clip
    device: str | None = field(default_factory=lambda: os.getenv("STAGE_CLIP_DEVICE") or None)
    yes_threshold: float = 0.8                 # калибровка по демо-объектам — см. докстринг модуля
    no_threshold: float = 0.46
    scale: float | None = 35.0                 # k; None — logit_scale модели (≈113 у SigLIP2 — слишком самоуверенно)
    stage_scale: float | None = None           # температура softmax по этапам; None — logit_scale модели
    per_sign: dict[str, tuple[float, float]] = field(default_factory=dict)  # ключ → (yes, no)
    tile_grid: int = 1                         # 1 — только кадр целиком; 2 — ещё 2×2 фрагмента
    mask_mode: str = "darken"


class Embedder(Protocol):
    """Что нужно классификатору от модели. Векторы — L2-нормированные строки."""
    name: str
    logit_scale: float

    def embed_images(self, images_rgb: list[np.ndarray]) -> np.ndarray: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _features(out: Any):
    """transformers 5 отдаёт из get_*_features объект (BaseModelOutputWithPooling), 4.x — тензор.
    Правка интегратора: на сервере (transformers 5.17) `.float()` падал на объекте, и модель Б
    не разбирала ни одного кадра (отчёты UI и модели А)."""
    if hasattr(out, "float"):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(out, (tuple, list)):
        return out[1] if len(out) > 1 else out[0]
    return out[0]


class _CachedTexts:
    """Кэш эмбеддингов текстов: формулировки не меняются, считать их на каждом кадре незачем."""

    def __init__(self):
        self._text_cache: dict[str, np.ndarray] = {}
        self._text_lock = threading.Lock()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        with self._text_lock:
            missing = list(dict.fromkeys(t for t in texts if t not in self._text_cache))
            if missing:
                vecs = _normalize(self._embed_texts_raw(missing))
                for t, v in zip(missing, vecs):
                    self._text_cache[t] = v
            return np.stack([self._text_cache[t] for t in texts])

    def _embed_texts_raw(self, texts: list[str]) -> np.ndarray:  # pragma: no cover — реализуют наследники
        raise NotImplementedError


class TransformersEmbedder(_CachedTexts):
    """SigLIP / SigLIP2 / CLIP через transformers (AutoModel: get_image_features / get_text_features)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        super().__init__()
        self.name = model_name
        self.device = device
        self.logit_scale = 100.0
        self._model = None
        self._proc = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoProcessor

            mps = getattr(torch.backends, "mps", None)
            device = self.device or ("cuda" if torch.cuda.is_available()
                                     else "mps" if mps is not None and mps.is_available() else "cpu")
            model = AutoModel.from_pretrained(self.name).eval().to(device)
            self._proc = AutoProcessor.from_pretrained(self.name)
            scale = getattr(model, "logit_scale", None)
            if scale is not None:
                self.logit_scale = float(scale.detach().float().exp().cpu())
            self.device, self._model, self._torch = device, model, torch

    def embed_images(self, images_rgb: list[np.ndarray]) -> np.ndarray:
        self._load()
        from PIL import Image

        pil = [Image.fromarray(np.ascontiguousarray(im)) for im in images_rgb]
        with self._torch.no_grad():
            inputs = self._proc(images=pil, return_tensors="pt").to(self.device)
            feats = _features(self._model.get_image_features(**inputs))
        return _normalize(feats.float().cpu().numpy())

    def _embed_texts_raw(self, texts: list[str]) -> np.ndarray:
        self._load()
        # SigLIP обучен на padding="max_length" (64 токена) — иначе эмбеддинги текста другие.
        with self._torch.no_grad():
            inputs = self._proc(text=texts, padding="max_length", max_length=64, truncation=True,
                                return_tensors="pt").to(self.device)
            feats = _features(self._model.get_text_features(**inputs))
        return feats.float().cpu().numpy()


class OpenClipEmbedder(_CachedTexts):
    """open_clip: имя вида «ViT-B-16-SigLIP:webli» или «hf-hub:timm/ViT-B-16-SigLIP»."""

    def __init__(self, model_name: str, device: str | None = None):
        super().__init__()
        self.name = model_name
        self.device = device or "cpu"
        self.logit_scale = 100.0
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import open_clip
            import torch

            arch, _, tag = self.name.partition(":") if not self.name.startswith("hf-hub:") else (self.name, "", None)
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag or None)
            model = model.eval().to(self.device)
            self._tok = open_clip.get_tokenizer(arch)
            self._pre, self._torch = preprocess, torch
            if hasattr(model, "logit_scale"):
                self.logit_scale = float(model.logit_scale.exp().item())
            self._model = model

    def embed_images(self, images_rgb: list[np.ndarray]) -> np.ndarray:
        self._load()
        from PIL import Image

        batch = self._torch.stack([self._pre(Image.fromarray(np.ascontiguousarray(im))) for im in images_rgb])
        with self._torch.no_grad():
            feats = self._model.encode_image(batch.to(self.device))
        return _normalize(feats.float().cpu().numpy())

    def _embed_texts_raw(self, texts: list[str]) -> np.ndarray:
        self._load()
        with self._torch.no_grad():
            feats = self._model.encode_text(self._tok(texts).to(self.device))
        return feats.float().cpu().numpy()


_EMBEDDERS: dict[tuple, Embedder] = {}
_EMBEDDERS_LOCK = threading.Lock()


def get_embedder(backend: str, model_name: str, device: str | None = None) -> Embedder:
    """Один эмбеддер на процесс: загрузка весов — секунды и сотни мегабайт памяти."""
    key = (backend, model_name, device)
    with _EMBEDDERS_LOCK:
        if key not in _EMBEDDERS:
            cls = OpenClipEmbedder if backend == "open_clip" else TransformersEmbedder
            _EMBEDDERS[key] = cls(model_name, device)
        return _EMBEDDERS[key]


@lru_cache(maxsize=4)
def load_prompts(path: str | None = None) -> dict:
    return json.loads(Path(path or PROMPTS_PATH).read_text(encoding="utf-8"))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class ClipChecklistClassifier:
    """StageClassifier (LOCAL): SigLIP-чек-лист."""

    name = "siglip"
    provider = Provider.LOCAL

    def __init__(self, config: ClipConfig | dict | None = None, embedder: Embedder | None = None,
                 prompts_path: str | None = None, **kw):
        if isinstance(config, dict):
            config = ClipConfig(**config)
        cfg = config or ClipConfig()
        for k, v in kw.items():           # get_classifier("siglip", yes_threshold=0.7)
            if k in ClipConfig.__dataclass_fields__:
                setattr(cfg, k, v)
        self.config = cfg
        self._embedder = embedder
        self.prompts = load_prompts(str(prompts_path) if prompts_path else None)
        self._matrices: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.config.backend, self.config.model_name, self.config.device)
        return self._embedder

    @property
    def model_id(self) -> str:
        return f"local:{self.config.model_name}"

    def ready(self) -> tuple[bool, str]:
        if self._embedder is not None:
            return True, ""
        need = ["torch", "open_clip" if self.config.backend == "open_clip" else "transformers"]
        missing = [m for m in need if importlib.util.find_spec(m) is None]
        if missing:
            return False, f"не установлены {', '.join(missing)} (pip install -r requirements-ml.txt)"
        if os.getenv("HF_HUB_OFFLINE") == "1" and self.config.backend != "open_clip":
            try:
                from huggingface_hub import try_to_load_from_cache
                if try_to_load_from_cache(self.config.model_name, "config.json") is None:
                    return False, f"веса {self.config.model_name} не скачаны (huggingface-cli download, см. models/README.md), а сеть выключена"
            except Exception:  # noqa: BLE001 — проверка кэша не должна ронять страницу настроек
                pass
        return True, ""

    # --- формулировки ---

    def _phrases(self, texts: list[str]) -> list[str]:
        tpl = self.prompts.get("template", "{}")
        return [tpl.format(t) for t in texts]

    def _sign_matrices(self, keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        """(K×d позитивные, K×d негативные) — средние эмбеддинги формулировок признака."""
        if keys not in self._matrices:
            signs = self.prompts["signs"]
            missing = [k for k in keys if k not in signs]
            if missing:
                raise KeyError(f"нет формулировок для признаков: {missing}")
            texts = []
            for k in keys:
                texts += self._phrases(signs[k]["pos"]) + self._phrases(signs[k]["neg"])
            emb = dict(zip(texts, self.embedder.embed_texts(texts)))

            def side(k: str, name: str) -> np.ndarray:
                return np.mean([emb[t] for t in self._phrases(signs[k][name])], axis=0)

            pos = _normalize(np.stack([side(k, "pos") for k in keys]))
            neg = _normalize(np.stack([side(k, "neg") for k in keys]))
            self._matrices[keys] = (pos, neg)
        return self._matrices[keys]

    def _stage_matrix(self) -> tuple[list[int], np.ndarray]:
        key = ("__stages__",)
        if key not in self._matrices:
            stages = self.prompts["stages"]
            ids = sorted(int(s) for s in stages)
            texts = [t for sid in ids for t in self._phrases(stages[str(sid)])]
            emb = dict(zip(texts, self.embedder.embed_texts(texts)))
            mat = _normalize(np.stack([np.mean([emb[t] for t in self._phrases(stages[str(sid)])], axis=0)
                                       for sid in ids]))
            self._matrices[key] = (np.array(ids), mat)
        ids, mat = self._matrices[key]
        return [int(i) for i in ids], mat

    def _thresholds(self, key: str) -> tuple[float, float]:
        if key in self.config.per_sign:
            yes, no = self.config.per_sign[key]
            return float(yes), float(no)
        spec = self.prompts["signs"].get(key, {})
        return (float(spec.get("yes_thr", self.config.yes_threshold)),
                float(spec.get("no_thr", self.config.no_threshold)))

    # --- кадр ---

    def _views(self, image_rgb: np.ndarray) -> list[np.ndarray]:
        views = [image_rgb]
        n = max(1, int(self.config.tile_grid))
        if n > 1:
            h, w = image_rgb.shape[:2]
            for i in range(n):
                for j in range(n):
                    views.append(image_rgb[i * h // n:(i + 1) * h // n, j * w // n:(j + 1) * w // n])
        return views

    def assess(self, image_bgr: np.ndarray, frame: FrameInfo, keys: list[str] | None = None,
               context: dict[str, Any] | None = None) -> ChecklistResult:
        t0 = time.perf_counter()
        ctx = dict(context or {})
        ctx.setdefault("mask_mode", self.config.mask_mode)
        image, masked = masked_for_model(image_bgr, ctx)
        rgb = np.ascontiguousarray(image[..., :3][..., ::-1]) if image.ndim == 3 else np.stack([image] * 3, axis=-1)
        keys = list(keys) if keys else list(taxonomy.signs())
        pos, neg = self._sign_matrices(tuple(keys))
        views = self._views(rgb)
        img = self.embedder.embed_images(views)                        # (views, d)
        diff = (img @ pos.T - img @ neg.T).max(axis=0)                 # по признаку — лучший фрагмент
        k = self.config.scale or float(getattr(self.embedder, "logit_scale", 100.0))
        probs = _sigmoid(k * diff)
        answers, scores = {}, {}
        for key, p in zip(keys, probs):
            yes, no = self._thresholds(key)
            answers[key] = Answer.YES if p >= yes else Answer.NO if p <= no else Answer.UNSURE
            scores[key] = round(float(p), 4)

        ids, smat = self._stage_matrix()
        # распределение по этапам — для показа и кандидатов: температура модели, а не
        # откалиброванный k чек-листа (с k = 35 softmax по 8 описаниям почти плоский)
        ks = self.config.stage_scale or float(getattr(self.embedder, "logit_scale", 100.0))
        logits = ks * (img[0] @ smat.T)
        ex = np.exp(logits - logits.max())
        likelihood = {sid: round(float(v), 4) for sid, v in zip(ids, ex / ex.sum())}

        return ChecklistResult(
            answers=answers, scores=scores, stage_likelihood=likelihood,
            model=getattr(self.embedder, "name", self.config.model_name), provider=Provider.LOCAL,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1), cost_usd=0.0,
            raw={"diff": {key: round(float(d), 5) for key, d in zip(keys, diff)}, "scale": round(k, 3),
                 "views": len(views), "masked": masked,
                 "thresholds": [self.config.yes_threshold, self.config.no_threshold]},
        )
