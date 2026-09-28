"""Модель Б: определение этапа строительства по снимкам.

    quality.assess        брак / ночь / дождь / снег / туман → QualityReport
    mask.DynamicMask      динамическая маска фона камеры (порт Дениса + автоинициализация)
    get_classifier(name)  чек-лист по кадру: "siglip" (LOCAL), "glm" (EXTERNAL), "local_vlm" (LOCAL)
    scoring.evaluate      ответы → доказательность этапов, подэтапы, фронт, готовность
    sequence.infer        монотонная хронология этапов (HMM/Витерби), выбросы, needs_review

Реализации классификаторов импортируются лениво: веб-слой и тесты не тянут
transformers/torch, пока SigLIP действительно не нужен.
"""
from __future__ import annotations

from core.contracts import StageClassifier

CLASSIFIERS = ("siglip", "glm", "local_vlm")


def get_classifier(name: str, **kw) -> StageClassifier:
    """Классификатор модели Б по имени из настроек (`model_b = siglip | local_vlm | glm`)."""
    key = (name or "").strip().lower()
    if key in ("siglip", "clip", "local"):
        from core.stage.checklist_clip import ClipChecklistClassifier
        return ClipChecklistClassifier(**kw)
    if key in ("glm", "zai", "external"):
        from core.stage.checklist_vlm import GlmChecklistClassifier
        return GlmChecklistClassifier(**kw)
    if key in ("local_vlm", "vlm", "ollama"):
        from core.stage.checklist_vlm import LocalVlmClassifier
        return LocalVlmClassifier(**kw)
    raise ValueError(f"неизвестный классификатор модели Б: {name!r} (ожидается один из {CLASSIFIERS})")


__all__ = ["CLASSIFIERS", "get_classifier"]
