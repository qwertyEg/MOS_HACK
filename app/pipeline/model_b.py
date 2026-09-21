"""Модель Б — VLM, заполняющая чек-лист этапа по кадру.

Сама модель вне репозитория: подключается по OpenAI-совместимому эндпоинту,
чтобы организатор мог подставить свой. Здесь только клиент и разбор ответа.

Два неочевидных решения:

1. **Ответ тернарный.** Бинарный «да/нет» заставляет модель угадывать там,
   где кадр не даёт информации — перекрыт краном, засвечен, объект далеко.
   Третье значение позволяет такой кадр не учитывать вовсе.

2. **Способ гашения фона — параметр, а не константа.** Заливка чёрным создаёт
   картинку, каких VLM в обучении не видела, и дырявое изображение может
   испортить ответ сильнее, чем убираемый шум. Какой режим лучше — вопрос
   замера, а не рассуждения, поэтому все пять вариантов реализованы и
   сравниваются через tools/vlm_bench.py.
"""

from __future__ import annotations

import base64
import enum
import io
import time
from dataclasses import dataclass

import numpy as np
import requests
from urllib.parse import urlparse
from PIL import Image, ImageFilter

from app.config import settings


class Answer(str, enum.Enum):
    YES = "yes"
    NO = "no"
    UNSURE = "unsure"


class MaskMode(str, enum.Enum):
    """Способ подачи маски модели. Сравниваются замером, см. §3.4.7 плана."""
    NONE = "none"          # без маски — базовая линия
    BLACK = "black"        # заливка чёрным
    BLUR = "blur"          # сильное размытие фона
    DARKEN = "darken"      # затемнение, контекст частично сохраняется
    CROP = "crop"          # обрезка по габаритам маски без закраски


@dataclass(slots=True)
class VlmReply:
    answer: Answer
    raw: str
    latency_ms: int


# Маркеры неуверенности ищутся по вхождению и имеют приоритет: фраза
# «не уверен» содержит и «не», и начинается не с «нет», поэтому проверять
# её надо до всего остального.
_UNSURE_MARKERS = ("не увер", "неуверен", "unsure", "unclear", "не могу",
                   "не ясно", "неясно", "непонятно", "затрудня", "сложно сказать")
_YES_WORDS = {"да", "yes", "стройка", "строится", "ведется", "ведутся"}
_NO_WORDS = {"нет", "no", "завершено", "завершен", "готово", "не"}

_TRIM = " \t\n.,!:;—-*_#«»\"'()"


def parse_answer(raw: str) -> Answer:
    """Свободный ответ модели → тернарное значение.

    Модель просят отвечать одним словом, но она не обязана слушаться, поэтому
    разбор устроен в три прохода: маркеры неуверенности, первое слово,
    вхождение куда угодно.
    """
    low = raw.strip().lower().replace("ё", "е")

    if any(m in low for m in _UNSURE_MARKERS):
        return Answer.UNSURE

    words = [w.strip(_TRIM) for w in low.split()]
    words = [w for w in words if w]
    if not words:
        return Answer.UNSURE

    first = words[0]
    if first in _YES_WORDS:
        return Answer.YES
    if first in _NO_WORDS:
        return Answer.NO

    # Ответила развёрнуто вопреки инструкции — ищем по вхождению целых слов.
    found = set(words)
    if found & _YES_WORDS:
        return Answer.YES
    if found & _NO_WORDS:
        return Answer.NO
    return Answer.UNSURE


def apply_mask(
    img: Image.Image,
    mask: np.ndarray | None,
    mode: MaskMode = MaskMode.DARKEN,
) -> Image.Image:
    """Гасит фон вне маски.

    mask — булев массив в размер изображения, True = объект (видимая область).
    """
    if mask is None or mode is MaskMode.NONE:
        return img

    if mask.shape[:2] != (img.height, img.width):
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_img = mask_img.resize((img.width, img.height), Image.NEAREST)
        mask = np.array(mask_img) > 127

    if mode is MaskMode.CROP:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return img
        return img.crop((int(xs.min()), int(ys.min()),
                         int(xs.max()) + 1, int(ys.max()) + 1))

    rgb = np.array(img.convert("RGB"))
    if mode is MaskMode.BLACK:
        bg = np.zeros_like(rgb)
    elif mode is MaskMode.BLUR:
        bg = np.array(img.convert("RGB").filter(ImageFilter.GaussianBlur(14)))
    else:  # DARKEN
        bg = (rgb * 0.3).astype(np.uint8)

    out = np.where(mask[..., None], rgb, bg)
    return Image.fromarray(out)


def encode(img: Image.Image, max_side: int = 1024) -> str:
    """JPEG data-URI. Ужимаем: на грубых бинарных вопросах разрешение выше
    1024 px качества не добавляет, а латентность растёт заметно."""
    img = img.convert("RGB")
    if max(img.size) > max_side:
        k = max_side / max(img.size)
        img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")


class ModelB:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.vlm_base_url).rstrip("/")
        self.model = model or settings.vlm_model
        self.session = requests.Session()

        # Если модель крутится локально, системный прокси надо обойти.
        # На машине разработки в окружении стоит HTTP_PROXY без no_proxy для
        # localhost, и запросы к соседнему контейнеру уходили в Squid, который
        # отвечал 503. Диагностируется мучительно: сервис «просто не отвечает».
        host = urlparse(self.base_url).hostname or ""
        if host in _LOCAL_HOSTS:
            self.session.trust_env = False

    def ask(self, data_uri: str, question: str) -> VlmReply:
        payload = {
            "model": self.model,
            "temperature": settings.vlm_temperature,
            "max_tokens": settings.vlm_max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": question},
                ],
            }],
        }
        t0 = time.perf_counter()
        r = self.session.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.vlm_api_key}"},
            json=payload,
            timeout=settings.vlm_timeout,
        )
        dt = int((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        raw = (choice["message"].get("content") or "").strip()

        # Рассуждающая модель с недостаточным бюджетом вернёт пустой content:
        # все токены ушли в reasoning. Это не «не уверена», это обрыв, и
        # молча превращать его в UNSURE нельзя — иначе тихо испортится
        # вся статистика, а причина будет неочевидна.
        if not raw and choice.get("finish_reason") == "length":
            return VlmReply(answer=Answer.UNSURE,
                            raw="<обрыв: max_tokens исчерпан рассуждением>",
                            latency_ms=dt)
        return VlmReply(answer=parse_answer(raw), raw=raw, latency_ms=dt)

    def fill_checklist(
        self,
        img: Image.Image,
        questions: list[dict],
        mask: np.ndarray | None = None,
        mask_mode: MaskMode = MaskMode.DARKEN,
    ) -> list[dict]:
        """Заполняет чек-лист одного этапа по одному кадру."""
        prepared = apply_mask(img, mask, mask_mode)
        uri = encode(prepared)
        out = []
        for q in questions:
            try:
                reply = self.ask(uri, q["text"])
            except Exception as exc:
                out.append({**q, "answer": Answer.UNSURE, "raw": f"ошибка: {exc}",
                            "latency_ms": 0})
                continue
            out.append({**q, "answer": reply.answer, "raw": reply.raw,
                        "latency_ms": reply.latency_ms})
        return out

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=5)
            return r.ok
        except Exception:
            return False
