"""Минимальная подмена core.vlm_client на время параллельной разработки.

core/vlm_client.py пишет модуль модели Б в своей ветке. Пока его нет в
дереве, детектор на VLM должен импортироваться и тестироваться на фейковом
клиенте — для этого здесь те же имена с тем же поведением. Как только
core.vlm_client появится, detect_vlm.py возьмёт настоящий (эта подмена
используется, только если модуля физически нет). Удалить после слияния.
"""
from __future__ import annotations

import base64
import json
import re

import cv2
import numpy as np


class VLMError(RuntimeError):
    """Ошибка модели или транспорта (совпадает по смыслу с core.vlm_client.VLMError)."""


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>", "", text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("в ответе нет JSON-объекта")
    obj, _ = json.JSONDecoder().raw_decode(re.sub(r",\s*([}\]])", r"\1", text[start:]))
    if not isinstance(obj, dict):
        raise ValueError("JSON в ответе — не объект")
    return obj


def image_to_data_url(image_bgr: np.ndarray, max_side: int = 1280, quality: int = 85) -> str:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VLMError("не удалось закодировать кадр в JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def make_client(provider: str, model: str | None = None):
    raise VLMError("клиент VLM (core.vlm_client) ещё не подключён в эту сборку")
