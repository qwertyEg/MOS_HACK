"""Уточнение подтипа грузовиков по кропу (refine.py) — на фейковом эмбеддере, без torch и сети.

Фейк раскладывает тексты и картинки по шести осям-подтипам: описание
подтипа узнаётся по ключевому слову, кроп — по цвету центрального пикселя
(тест красит машину нужным цветом).
"""
from __future__ import annotations

import datetime as dt
import json

import cv2
import numpy as np
import pytest

from core.contracts import FrameInfo
from core.equipment import EquipmentConfig, EquipmentEngine, refine, synthetic as S
from core.equipment.detect_yolo import YoloDetector
from core.equipment.refine import CropRefiner, square_crop

AXES = list(refine.DESCRIPTIONS)          # dump_truck, truck, crane_manipulator, concrete_mixer, mobile_crane, concrete_pump
# Порядок важен: описание манипулятора содержит и «flatbed», и «folding».
KEYWORDS = {"knuckle": "crane_manipulator", "tipping": "dump_truck", "drum": "concrete_mixer",
            "telescopic": "mobile_crane", "folding": "concrete_pump", "flatbed": "truck", "semi-trailer": "truck"}


def axis_vec(**w) -> np.ndarray:
    v = np.array([w.get(k, 0.0) for k in AXES], np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    name = "fake-siglip"
    logit_scale = 10.0

    def __init__(self, colours: dict[tuple[int, int, int], np.ndarray]):
        self.colours = colours          # RGB центра кропа → вектор
        self.text_calls = 0
        self.image_calls = 0
        self.crops: list[np.ndarray] = []

    def embed_texts(self, texts):
        self.text_calls += 1
        out = []
        for t in texts:
            cls = next(c for kw, c in KEYWORDS.items() if kw in t)
            out.append(axis_vec(**{cls: 1.0}))
        return np.stack(out)

    def embed_images(self, images_rgb):
        self.image_calls += 1
        self.crops += images_rgb
        out = []
        for im in images_rgb:
            h, w = im.shape[:2]
            out.append(self.colours.get(tuple(int(c) for c in im[h // 2, w // 2]), axis_vec(dump_truck=1.0)))
        return np.stack(out)


BLUE, GREEN, RED, GREY = (0, 0, 255), (0, 255, 0), (255, 0, 0), (128, 128, 128)   # RGB


def scene(*machines) -> np.ndarray:
    """Кадр 1280×720, машины — сплошные прямоугольники цвета-кода (рамки XYWH)."""
    img = S.ground(1280, 720)
    for (x, y, w, h), rgb in machines:
        cv2.rectangle(img, (x, y), (x + w, y + h), rgb[::-1], -1)
    return img


def frame(k=0):
    return FrameInfo(k, "c1", 1, dt.datetime(2026, 9, 28, 6, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=25 * k),
                     1280, 720)


def refiner(colours, **cfg) -> tuple[CropRefiner, FakeEmbedder]:
    emb = FakeEmbedder(colours)
    return CropRefiner(embedder=emb, config=EquipmentConfig(**cfg)), emb


# --------------------------------------------------------------------------


def test_confident_flatbed_changes_dump_truck_to_truck_and_keeps_evidence():
    box = (200, 300, 300, 120)
    r, _ = refiner({BLUE: axis_vec(dump_truck=0.2, truck=1.0)})
    (d,) = r.refine(scene((box, BLUE)), [S.det("dump_truck", box, conf=0.8)], known={"dump_truck"})
    assert d.cls == "truck" and d.conf == 0.8
    info = d.extra["refine"]
    assert info["det_cls"] == "dump_truck" and info["det_conf"] == 0.8 and info["changed"] is True
    assert set(info["scores"]) == set(AXES) and info["scores"]["truck"] > 0.9
    assert sum(info["scores"].values()) == pytest.approx(1.0, abs=0.01)


def test_crane_manipulator_found_on_a_box_the_detector_called_mobile_crane():
    box = (500, 200, 260, 200)
    r, _ = refiner({GREEN: axis_vec(crane_manipulator=1.0, mobile_crane=0.3)})
    (d,) = r.refine(scene((box, GREEN)), [S.det("mobile_crane", box)], known={"mobile_crane", "dump_truck"})
    assert d.cls == "crane_manipulator" and d.extra["refine"]["det_cls"] == "mobile_crane"


def test_second_description_of_a_class_counts_too():
    """У «грузовика» два облика (бортовой, седельный с полуприцепом) — узнаётся любой."""
    class TwoLooks(FakeEmbedder):
        def embed_texts(self, texts):
            self.text_calls += 1
            vecs = []
            for t in texts:
                if "semi-trailer" in t:
                    vecs.append(np.eye(7, dtype=np.float32)[6])          # своя ось у полуприцепа
                else:
                    cls = next(c for kw, c in KEYWORDS.items() if kw in t)
                    vecs.append(np.eye(7, dtype=np.float32)[AXES.index(cls)])
            return np.stack(vecs)

    box = (200, 300, 400, 120)
    emb = TwoLooks({BLUE: np.eye(7, dtype=np.float32)[6]})
    (d,) = CropRefiner(embedder=emb).refine(scene((box, BLUE)), [S.det("dump_truck", box)])
    assert d.cls == "truck"


def test_unsure_crop_keeps_detector_class():
    box = (200, 300, 300, 120)
    r, _ = refiner({BLUE: axis_vec(dump_truck=0.97, truck=1.0)})        # truck чуть вероятнее, но без перевеса
    (d,) = r.refine(scene((box, BLUE)), [S.det("dump_truck", box)])
    assert d.cls == "dump_truck"
    assert d.extra["refine"]["changed"] is False and d.extra["refine"]["scores"]["truck"] > 0.5


def test_switch_to_a_class_the_detector_knows_needs_bigger_margin():
    """Самосвал → миксер с перевесом ~0.3: детектор сам умеет «миксер» и не сказал — не верим
    (нужен перевес refine_margin_known); детектору без такого класса — верим."""
    box = (200, 300, 300, 150)
    colours = {RED: axis_vec(dump_truck=1.0, concrete_mixer=1.09)}
    r, _ = refiner(colours)
    (d,) = r.refine(scene((box, RED)), [S.det("dump_truck", box)], known={"dump_truck", "concrete_mixer"})
    assert d.cls == "dump_truck"
    margin = d.extra["refine"]["scores"]["concrete_mixer"] - d.extra["refine"]["scores"]["dump_truck"]
    assert 0.2 <= margin < 0.4
    r2, _ = refiner(colours)
    (d2,) = r2.refine(scene((box, RED)), [S.det("dump_truck", box)], known={"dump_truck"})
    assert d2.cls == "concrete_mixer"


def test_thresholds_come_from_config():
    box = (200, 300, 300, 120)
    colours = {BLUE: axis_vec(dump_truck=0.2, truck=1.0)}
    r, _ = refiner(colours, refine_min_prob=0.9999)
    (d,) = r.refine(scene((box, BLUE)), [S.det("dump_truck", box)])
    assert d.cls == "dump_truck"
    with pytest.raises(ValueError):
        EquipmentConfig.from_dict({"refine_min_prob": 1.5})


def test_other_classes_and_tiny_boxes_are_not_touched_and_model_not_called():
    r, emb = refiner({})
    img = scene()
    dets = [S.det("excavator", (100, 100, 300, 200)), S.det("dump_truck", (700, 400, 20, 30))]
    out = r.refine(img, dets)
    assert [d.cls for d in out] == ["excavator", "dump_truck"]
    assert all("refine" not in d.extra for d in out)
    assert emb.image_calls == 0 and emb.text_calls == 0, "нечего уточнять — модель даже не грузится"


def test_input_detections_are_not_mutated_and_refine_is_idempotent():
    box = (200, 300, 300, 120)
    r, emb = refiner({BLUE: axis_vec(truck=1.0)})
    src = [S.det("dump_truck", box)]
    out = r.refine(scene((box, BLUE)), src)
    assert src[0].cls == "dump_truck" and "refine" not in src[0].extra
    again = r.refine(scene((box, BLUE)), out)
    assert again[0].cls == "truck" and emb.image_calls == 1, "уже уточнённую рамку второй раз не гоняем"


def test_text_embeddings_are_computed_once():
    box = (200, 300, 300, 120)
    r, emb = refiner({BLUE: axis_vec(truck=1.0)})
    for _ in range(3):
        r.refine(scene((box, BLUE)), [S.det("dump_truck", box)])
    assert emb.image_calls == 3
    assert emb.text_calls == sum(len(v) for v in refine.DESCRIPTIONS.values())


def test_all_truck_boxes_of_a_frame_go_in_one_batch():
    a, b = (100, 300, 300, 120), (700, 300, 300, 150)
    r, emb = refiner({BLUE: axis_vec(truck=1.0), GREEN: axis_vec(concrete_pump=1.0)})
    out = r.refine(scene((a, BLUE), (b, GREEN)), [S.det("dump_truck", a), S.det("concrete_mixer", b)])
    assert [d.cls for d in out] == ["truck", "concrete_pump"]
    assert emb.image_calls == 1 and len(emb.crops) == 2


def test_model_failure_disables_refine_without_breaking_detection():
    class Broken(FakeEmbedder):
        def embed_images(self, images_rgb):
            self.image_calls += 1
            raise OSError("нет сети: не скачать google/siglip2-base-patch16-224")

    box = (200, 300, 300, 120)
    emb = Broken({})
    r = CropRefiner(embedder=emb)
    out = r.refine(scene((box, BLUE)), [S.det("dump_truck", box)])
    assert [d.cls for d in out] == ["dump_truck"] and "refine" not in out[0].extra
    assert "нет сети" in r.error and r.available()[0] is False
    r.refine(scene((box, BLUE)), [S.det("dump_truck", box)])
    assert emb.image_calls == 1, "после ошибки модель больше не дёргаем"


def test_square_crop_pads_long_truck_with_grey_and_keeps_it_centred():
    img = scene(((100, 300, 600, 150), BLUE))
    crop = square_crop(img, (100, 300, 600, 150), pad_frac=0.0)
    assert crop.shape == (600, 600, 3)
    assert tuple(crop[0, 0]) == (127, 127, 127) and tuple(crop[300, 300]) == BLUE[::-1]
    assert square_crop(img, (1300, 800, 50, 50)) is None, "рамка за кадром — кропа нет"


# --------------------------------------------------------------------------
# в составе YoloDetector
# --------------------------------------------------------------------------


class FakeYolo:
    def __init__(self, rows):
        self.rows, self.kwargs, self.names = rows, [], {}

    def predict(self, img, **kw):
        from types import SimpleNamespace
        self.kwargs.append(kw)
        arr = np.array([r[:4] for r in self.rows], np.float32).reshape(-1, 4)
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=arr, conf=np.array([r[4] for r in self.rows]),
                                                      cls=np.array([r[5] for r in self.rows])))]


SERVER_NAMES = {"0": "excavator", "1": "dump_truck", "2": "bulldozer", "3": "roller", "4": "concrete_mixer",
                "5": "mobile_crane", "6": "tower_crane", "7": "concrete_pump", "8": "pile_driver",
                "9": "wheel_loader", "10": "backhoe_loader", "11": "grader"}


def server_models_dir(tmp_path):
    """Каталог как /root/mos_hack/models после обучения: веса + equipment_classes.json с метриками."""
    d = tmp_path / "models"
    d.mkdir()
    (d / "equipment_yolo.pt").write_bytes(b"weights")
    (d / "equipment_classes.json").write_text(json.dumps({
        "model": "equipment_yolo.pt", "names": SERVER_NAMES, "imgsz": 640, "epochs_trained": 32,
        "not_covered": ["truck", "crane_manipulator"], "val_overall": {"mAP50": 0.8646}}), encoding="utf-8")
    return d


def test_yolo_with_refine_reports_truck_and_crane_manipulator_and_uses_them(tmp_path):
    d = server_models_dir(tmp_path)
    box = (200, 300, 300, 120)
    emb = FakeEmbedder({BLUE: axis_vec(truck=1.0)})
    model = FakeYolo([(200, 300, 500, 420, 0.8, 1), (700, 200, 900, 400, 0.9, 0)])
    det = YoloDetector(weights=d / "equipment_yolo.pt", model=model, refiner=CropRefiner(embedder=emb))
    sup = det.supported_classes
    assert "truck" in sup and "crane_manipulator" in sup and "excavator" in sup
    out = det.detect(scene((box, BLUE)), frame())
    assert sorted(x.cls for x in out) == ["excavator", "truck"]
    assert model.kwargs[0]["imgsz"] == 640, "imgsz обучения из equipment_classes.json"
    assert det.ready() == (True, "")


def test_without_transformers_refine_is_quietly_off(tmp_path, monkeypatch):
    monkeypatch.setattr(refine, "transformers_available", lambda: False)
    monkeypatch.delenv("EQUIPMENT_REFINE", raising=False)
    d = server_models_dir(tmp_path)
    box = (200, 300, 300, 120)
    det = YoloDetector(weights=d / "equipment_yolo.pt", model=FakeYolo([(200, 300, 500, 420, 0.8, 1)]))
    assert det.refiner is not None and not det.refine_active
    assert "truck" not in det.supported_classes and "crane_manipulator" not in det.supported_classes
    ok, why = det.ready()
    assert ok and "уточнение подтипа грузовиков выключено" in why and "truck" in why
    (x,) = det.detect(scene((box, BLUE)), frame())
    assert x.cls == "dump_truck" and "refine" not in x.extra


def test_refine_can_be_switched_off_by_flag_or_env(tmp_path, monkeypatch):
    d = server_models_dir(tmp_path)
    assert YoloDetector(weights=d / "equipment_yolo.pt", refine=False).refiner is None
    monkeypatch.setenv("EQUIPMENT_REFINE", "0")
    det = YoloDetector(weights=d / "equipment_yolo.pt")
    assert det.refiner is None and "truck" not in det.supported_classes


def test_equipment_weights_env_accepts_models_dir_or_classes_json(tmp_path, monkeypatch):
    d = server_models_dir(tmp_path)
    monkeypatch.setenv("EQUIPMENT_REFINE", "0")
    monkeypatch.delenv("EQUIPMENT_CLASSES", raising=False)
    for target in (d, d / "equipment_classes.json", d / "equipment_yolo.pt"):
        monkeypatch.setenv("EQUIPMENT_WEIGHTS", str(target))
        det = YoloDetector()
        assert det.weights == d / "equipment_yolo.pt" and det.classes_json == d / "equipment_classes.json"
        assert det.imgsz == 640
        assert det.supported_classes[:2] == ["excavator", "dump_truck"]


def test_refined_labels_feed_the_unit_vote():
    """Уточнение иногда промахивается (truck на одном кадре из пяти) — единица остаётся одной и с классом большинства."""
    eng = EquipmentEngine()
    upd = None
    for k in range(5):
        d = S.det("truck" if k != 2 else "dump_truck", (300, 300, 200, 90), conf=0.8)
        upd = eng.process(frame(k), None, [d], None, [], [])
    assert len(upd.units) == 1 and upd.units[0].cls == "truck"
    assert upd.units[0].label == "Грузовик (бортовой, длинномер, трал) №1"


def test_old_weights_name_from_settings_falls_back_to_our_weights_in_same_dir(tmp_path, monkeypatch):
    """Веб-слой по умолчанию просит models/equipment.pt, а обучение кладёт equipment_yolo.pt."""
    monkeypatch.setenv("EQUIPMENT_REFINE", "0")
    d = server_models_dir(tmp_path)
    det = YoloDetector(weights=str(d / "equipment.pt"), model=None)
    assert det.weights == d / "equipment_yolo.pt" and det.imgsz == 640
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    ok, why = det.ready()
    assert ok and "equipment.pt" in why and "equipment_yolo.pt" in why
