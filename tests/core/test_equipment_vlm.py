"""EXTERNAL-детектор на VLM — на фейковом клиенте (в сеть тесты не ходят)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from core import taxonomy
from core.contracts import Detector, Provider
from core.equipment import get_detector
from core.equipment.detect_vlm import VLMError, VlmDetector, build_prompt, parse_objects


@dataclass
class FakeReply:
    data: dict
    text: str = ""
    usage: object = None
    latency_ms: float = 12.0


@dataclass
class FakeClient:
    """Отвечает заготовленным; запоминает, что спросили."""
    answer: object = None
    error: Exception | None = None
    model_id: str = "glm-4.6v"
    calls: list = field(default_factory=list)

    def ask_json(self, system, image_url, prompt, max_tokens):
        self.calls.append((system, image_url, prompt, max_tokens))
        if self.error:
            raise self.error
        return self.answer if isinstance(self.answer, FakeReply) else FakeReply(self.answer)

    def ready(self):
        return True, ""


IMG = np.zeros((720, 1280, 3), np.uint8)


def test_normalized_coordinates_become_pixels_xywh():
    client = FakeClient({"objects": [
        {"class": "excavator", "bbox": [100, 200, 400, 600], "conf": 0.9, "working": True},
        {"class": "dump_truck", "bbox": [500, 500, 700, 800], "conf": 0.8, "working": None},
    ]})
    dets = VlmDetector(client=client).detect(IMG)
    exc = next(d for d in dets if d.cls == "excavator")
    assert exc.bbox == pytest.approx((128.0, 144.0, 384.0, 288.0))
    assert exc.source == "glm-4.6v" and exc.extra["vlm_working"] is True
    assert client.calls[0][1].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("answer", [
    {"text": "на снимке экскаватор"},                     # нет списка объектов
    {"objects": "excavator at left"},                    # не список
    {"objects": [["excavator", 1, 2, 3, 4]]},            # объекты не словари
    {"objects": [{"class": "excavator"}]},               # без рамок
    {},                                                  # пусто
])
def test_garbage_json_raises_vlm_error(answer):
    with pytest.raises(VLMError):
        VlmDetector(client=FakeClient(answer)).detect(IMG)


def test_non_json_text_reply_raises_vlm_error():
    reply = FakeReply(data={}, text="Извините, я не могу помочь")
    with pytest.raises(VLMError, match="не вернула JSON"):
        VlmDetector(client=FakeClient(reply)).detect(IMG)


def test_text_only_reply_is_parsed_like_glm_box_markup():
    text = '<|begin_of_box|>{"objects": [{"class": "roller", "bbox": [0, 0, 500, 500], "conf": 0.7}]}<|end_of_box|>'
    dets = VlmDetector(client=FakeClient(FakeReply(data={}, text=text))).detect(IMG)
    assert [d.cls for d in dets] == ["roller"]


def test_any_client_exception_becomes_vlm_error_with_readable_text():
    with pytest.raises(VLMError, match="KeyError"):
        VlmDetector(client=FakeClient(error=KeyError("choices"))).detect(IMG)
    with pytest.raises(VLMError, match="HTTP 401"):
        VlmDetector(client=FakeClient(error=VLMError("HTTP 401: неверный ключ"))).detect(IMG)


def test_unknown_classes_dropped_with_note_and_aliases_mapped():
    det = VlmDetector(client=FakeClient({"objects": [
        {"class": "car", "bbox": [0, 0, 100, 100]},
        {"class": "самосвал", "bbox": [100, 100, 400, 400], "conf": "85"},
        {"class": "Concrete Mixer Truck", "bbox": [500, 100, 900, 500], "conf": 0.6},
        {"class": "spaceship", "bbox": [0, 500, 300, 900]},
    ]}))
    dets = det.detect(IMG)
    assert {d.cls for d in dets} == {"dump_truck", "concrete_mixer"}
    assert next(d for d in dets if d.cls == "dump_truck").conf == pytest.approx(0.85)
    assert len(det.last_report["dropped"]) == 2
    assert any("spaceship" in s for s in det.last_report["dropped"])


def test_bbox_is_repaired_clipped_and_scale_detected():
    # перевёрнутая рамка и вылет за 1000
    dets, _ = parse_objects({"objects": [{"class": "bulldozer", "bbox": [600, 900, 200, 1100]}]}, 1000, 1000)
    assert dets[0].bbox == pytest.approx((200.0, 900.0, 400.0, 100.0))
    # доли кадра вместо 0..1000
    dets, _ = parse_objects({"objects": [{"class": "bulldozer", "bbox": [0.1, 0.2, 0.3, 0.4]}]}, 1280, 720)
    assert dets[0].bbox == pytest.approx((128.0, 144.0, 256.0, 144.0))
    # пиксели кадра
    dets, _ = parse_objects({"objects": [{"class": "bulldozer", "bbox": [640, 360, 1280, 720]}]}, 1280, 720)
    assert dets[0].bbox == pytest.approx((640.0, 360.0, 640.0, 360.0))
    # рамка строкой и парами точек
    dets, _ = parse_objects({"objects": [{"class": "roller", "bbox": "[100, 100, 200, 300]"},
                                         {"class": "grader", "box": [[500, 500], [600, 700]]}]}, 1000, 1000)
    assert [d.cls for d in dets] == ["roller", "grader"]


def test_duplicate_boxes_from_vlm_are_collapsed():
    dets = VlmDetector(client=FakeClient({"objects": [
        {"class": "truck", "bbox": [100, 100, 400, 400], "conf": 0.7},
        {"class": "dump_truck", "bbox": [105, 102, 398, 405], "conf": 0.6},
    ]})).detect(IMG)
    assert len(dets) == 1


def test_prompt_lists_every_class_with_russian_name_and_look():
    p = build_prompt()
    for key, e in taxonomy.equipment().items():
        assert f"- {key} — {e.name}: {e.look}" in p
    assert "0..1000" in p and '"objects"' in p


def test_protocol_ready_and_factory():
    det = get_detector("glm", client=FakeClient({"objects": []}))
    assert isinstance(det, Detector) and det.provider == Provider.EXTERNAL
    assert det.ready() == (True, "")
    assert det.detect(IMG) == []
    local = get_detector("local_vlm", client=FakeClient({"objects": []}))
    assert local.provider == Provider.LOCAL


def test_ready_without_client_explains_why():
    class Broken:
        def ready(self):
            raise VLMError("нет ZAI_API_KEY")
    ok, why = VlmDetector(client=Broken()).ready()
    assert not ok and "ZAI_API_KEY" in why


def test_unknown_detector_name():
    with pytest.raises(ValueError, match="yolo"):
        get_detector("detectron")
