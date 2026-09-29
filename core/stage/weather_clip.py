"""Погода по смыслу кадра: zero-shot SigLIP «ясно / ночь / туман / дождь / снегопад…».

Эвристики по одному кадру не видят тумана на фото, где передний план резкий, и дождя
на мокром асфальте (testset/conditions: туман 0 из 10, дождь 0 из 9 до этой доработки).
SigLIP2 — та же модель, что у локальной модели Б, — различает это по смыслу кадра.
Картинка сравнивается с английскими описаниями классов (`CLASSES`, усреднённые
эмбеддинги), softmax с температурой `scale` даёт вероятности; решение — в
quality.report_from_metrics по порогам QualityConfig.clip_*.

Проверка (docs/limitations.md, tools/quality_bench.py): на 678 дневных кадрах пяти
российских камер ни одного срабатывания «туман / дождь / снегопад»; ночь без часов —
109 из 114 ночных кадров; testset/conditions: туман 9 из 10, дождь 4 из 9.
Засветку солнцем и капли-подтёки на колпаке камеры zero-shot не видит (на кадрах
камер — 0), их ловит норма камеры (camera_norm).

Стоимость — один проход image-энкодера (~0.1–0.3 с на CPU); в конвейере — только для
кадров, которые идут в модель Б (не чаще stage_every_h на камеру), и в «Проверить снимок».
"""
from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np

CLASSES: dict[str, list[str]] = {
    "clear": ["a clear daytime photo of a construction site",
              "a sharp photo of buildings under construction in daylight",
              "an overcast day at a construction site"],
    "night": ["a photo of a construction site at night", "a night photo with floodlights and a dark sky"],
    "fog": ["a construction site in thick fog", "a hazy foggy photo with low visibility"],
    "rain": ["a construction site in heavy rain", "a rainy day with wet ground and puddles"],
    "drops": ["a blurry photo taken through a camera lens with raindrops on it", "water drops on the camera glass"],
    "snowfall": ["heavy snowfall, snow is falling", "a blizzard at a construction site"],
    "snow": ["a construction site covered with snow in winter", "snow on the ground"],
    "glare": ["the sun shining directly into the camera with lens flare", "strong sun glare in the photo"],
    "dusk": ["a photo at dusk after sunset", "a sunset sky over a city"],
}


class WeatherClip:
    """Вероятности погодных классов по кадру. embedder — объект с embed_images(list[RGB]) и
    embed_texts(list[str]) → L2-нормированные строки (эмбеддер модели Б или фейк в тестах)."""

    def __init__(self, embedder: Any, scale: float = 50.0, classes: dict[str, list[str]] | None = None):
        self.embedder = embedder
        self.scale = scale
        self.classes = classes or CLASSES
        self._text: np.ndarray | None = None
        self._lock = threading.Lock()

    def _text_matrix(self) -> np.ndarray:
        with self._lock:
            if self._text is None:
                rows = []
                for name in self.classes:
                    v = np.asarray(self.embedder.embed_texts(self.classes[name]), np.float32).mean(axis=0)
                    rows.append(v / max(float(np.linalg.norm(v)), 1e-9))
                self._text = np.stack(rows)
            return self._text

    def probs_from_embedding(self, emb: np.ndarray) -> dict[str, float]:
        v = np.asarray(emb, np.float32).reshape(-1)
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        s = self._text_matrix() @ v * self.scale
        p = np.exp(s - s.max())
        p /= p.sum()
        return {k: round(float(x), 4) for k, x in zip(self.classes, p)}

    def probs(self, image_bgr: np.ndarray) -> dict[str, float]:
        from core.stage.quality import crop_borders   # чёрные поля видео 4:3 — не часть сцены

        rgb = cv2.cvtColor(crop_borders(image_bgr), cv2.COLOR_BGR2RGB)
        emb = self.embedder.embed_images([rgb])[0]
        return self.probs_from_embedding(emb)
