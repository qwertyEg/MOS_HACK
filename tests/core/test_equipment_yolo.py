"""LOCAL-детектор YOLO — без torch/ultralytics: модель подменяется фейком."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from core.contracts import Detector, FrameInfo, Provider
from core.equipment import detect_yolo, get_detector, synthetic as S
from core.equipment.detect_yolo import YoloDetector, enhance_low_light, is_dark, load_class_map


class FakeModel:
    """Отдаёт заготовленные рамки (xyxy в координатах поданного изображения) и запоминает входы."""

    def __init__(self, boxes, names):
        self.boxes, self.names, self.inputs, self.kwargs = boxes, names, [], []

    def predict(self, img, **kw):
        self.inputs.append(img)
        self.kwargs.append(kw)
        return [_result(self.boxes)]


def _result(rows):
    arr = np.array([b[:4] for b in rows], dtype=np.float32).reshape(-1, 4)
    return SimpleNamespace(boxes=SimpleNamespace(
        xyxy=arr, conf=np.array([b[4] for b in rows], np.float32), cls=np.array([b[5] for b in rows])))


def frame(night=False, w=1280, h=720):
    import datetime as dt
    return FrameInfo(1, "c1", 1, dt.datetime(2026, 9, 28, tzinfo=dt.timezone.utc), w, h, is_night=night)


def classes_file(tmp_path, payload):
    p = tmp_path / "equipment_classes.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_canonical_names_from_classes_json(tmp_path):
    cj = classes_file(tmp_path, {"names": {"0": "excavator", "1": "dump_truck"}})
    model = FakeModel([(100, 100, 300, 250, 0.9, 0), (600, 300, 900, 500, 0.8, 1)], names={0: "x", 1: "y"})
    det = YoloDetector(weights=tmp_path / "none.pt", classes_json=cj, model=model)
    out = det.detect(S.ground(1280, 720), frame())
    assert [(d.cls, d.bbox) for d in out] == [("excavator", (100.0, 100.0, 200.0, 150.0)),
                                              ("dump_truck", (600.0, 300.0, 300.0, 200.0))]
    assert out[0].source == "yolo"
    assert model.kwargs[0]["conf"] == 0.25 and model.kwargs[0]["verbose"] is False


def test_dataset_names_with_map_and_builtin_aliases(tmp_path):
    """MOCS-подобные имена: часть переводится картой из json, часть — встроенной таблицей, люди отбрасываются."""
    cj = classes_file(tmp_path, {"names": ["Worker", "Static crane", "Pump truck", "Big Yellow Thing"],
                                 "map": {"Big Yellow Thing": "bulldozer", "Worker": None}})
    boxes = [(10, 10, 60, 120, 0.9, 0), (100, 50, 300, 600, 0.8, 1), (400, 300, 700, 500, 0.7, 2),
             (800, 300, 1000, 450, 0.6, 3)]
    det = YoloDetector(weights=tmp_path / "none.pt", classes_json=cj, model=FakeModel(boxes, {}))
    assert [d.cls for d in det.detect(S.ground(1280, 720), frame())] == ["tower_crane", "concrete_pump", "bulldozer"]
    assert det.supported_classes == ["bulldozer", "tower_crane", "concrete_pump"]   # порядок словаря


def test_names_from_weights_when_no_json(tmp_path):
    model = FakeModel([(100, 100, 300, 250, 0.9, 3)], names={3: "Excavator"})
    det = YoloDetector(weights=tmp_path / "none.pt", classes_json=tmp_path / "missing.json", model=model)
    assert [d.cls for d in det.detect(S.ground(1280, 720), frame())] == ["excavator"]


def test_class_map_formats(tmp_path):
    names, mapping = load_class_map(classes_file(tmp_path, {"names": {"0": "excavator", "2": "roller"}}))
    assert names == {0: "excavator", 2: "roller"} and mapping == {}
    names, _ = load_class_map(classes_file(tmp_path, ["a", "b"]))
    assert names == {0: "a", 1: "b"}


def test_ready_reports_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(detect_yolo.importlib.util, "find_spec", lambda name: None)
    ok, why = YoloDetector(weights=tmp_path / "none.pt").ready()
    assert not ok and "ultralytics" in why
    monkeypatch.setattr(detect_yolo.importlib.util, "find_spec", lambda name: object())
    ok, why = YoloDetector(weights=tmp_path / "none.pt").ready()
    assert ok and "YOLO-World" in why
    weights = tmp_path / "equipment_yolo.pt"
    weights.write_bytes(b"x")
    classes_file(tmp_path, {"names": {"0": "excavator"}})
    assert YoloDetector(weights=weights).ready() == (True, "")


def test_weights_path_from_env(tmp_path, monkeypatch):
    w = tmp_path / "custom.pt"
    monkeypatch.setenv("EQUIPMENT_WEIGHTS", str(w))
    assert YoloDetector().weights == w


def test_low_light_enhancement_brightens_and_keeps_shape():
    dark = S.darken(S.ground(640, 360), gain=0.2)
    out = enhance_low_light(dark)
    assert out.shape == dark.shape and out.dtype == np.uint8
    assert out.mean() > dark.mean() * 1.5
    assert is_dark(dark) and not is_dark(S.ground(640, 360))


def test_auto_enhancement_only_for_dark_or_night_frames(tmp_path):
    cj = classes_file(tmp_path, {"names": {"0": "excavator"}})
    model = FakeModel([], {})
    det = YoloDetector(weights=tmp_path / "none.pt", classes_json=cj, model=model)
    day = S.ground(640, 360)
    det.detect(day, frame(w=640, h=360))
    assert model.inputs[-1] is day, "днём кадр подаётся как есть"
    det.detect(day, frame(night=True, w=640, h=360))
    assert model.inputs[-1] is not day
    dark = S.darken(day, gain=0.2)
    det.detect(dark, frame(w=640, h=360))
    assert model.inputs[-1].mean() > dark.mean()
    off = YoloDetector(weights=tmp_path / "none.pt", classes_json=cj, model=model, enhance=False)
    off.detect(dark, frame(night=True, w=640, h=360))
    assert model.inputs[-1] is dark



def test_tiling_offsets_boxes_and_merges_duplicates(tmp_path):
    """4K-кадр кусками: рамки тайлов переводятся в координаты кадра, куски машины у границ
    тайлов и дубли из перекрытий схлопываются с рамкой прохода целиком."""
    cj = classes_file(tmp_path, {"names": {"0": "excavator"}})
    machines = [(20, 20, 220, 170), (900, 20, 1400, 170), (3000, 1500, 3300, 1700)]

    class SceneModel:
        """«Видит» машины сцены, попавшие в поданный кусок; где кусок лежит в кадре —
        закодировано в пикселях (синий = x/16, зелёный = y/16)."""
        names: dict = {}

        def __init__(self):
            self.calls = 0

        def predict(self, img, **kw):
            self.calls += 1
            x0, y0 = int(img[0, 0, 0]) * 16, int(img[0, 0, 1]) * 16
            h, w = img.shape[:2]
            rows = []
            for a, b, c, d in machines:
                ca, cb, cc, cd = max(a, x0), max(b, y0), min(c, x0 + w), min(d, y0 + h)
                if cc - ca > 10 and cd - cb > 10:
                    rows.append((ca - x0, cb - y0, cc - x0, cd - y0, 0.9, 0))
            return [_result(rows)]

    img = np.zeros((2160, 3840, 3), np.uint8)
    img[:, :, 0] = (np.arange(3840) // 16)[None, :]
    img[:, :, 1] = (np.arange(2160) // 16)[:, None]
    model = SceneModel()
    det = YoloDetector(weights=tmp_path / "none.pt", classes_json=cj, model=model, tile=True, enhance=False)
    out = det.detect(img, frame(w=3840, h=2160))
    assert model.calls == 1 + 4 * 2, "проход целиком + сетка тайлов 4×2"
    got = sorted(tuple(round(v) for v in d.bbox) for d in out)
    assert got == sorted((a, b, c - a, d - b) for a, b, c, d in machines)


def test_small_frames_are_not_tiled(tmp_path):
    cj = classes_file(tmp_path, {"names": {"0": "excavator"}})
    model = FakeModel([], {})
    YoloDetector(weights=tmp_path / "none.pt", classes_json=cj, model=model, tile=True).detect(
        S.ground(1280, 720), frame())
    assert len(model.inputs) == 1


def test_factory_and_protocol(tmp_path):
    det = get_detector("yolo", weights=tmp_path / "none.pt")
    assert isinstance(det, Detector) and det.provider == Provider.LOCAL and det.name == "yolo"
    assert get_detector("local", weights=tmp_path / "none.pt").name == "yolo"


def test_supported_classes_without_loading_model(tmp_path):
    weights = tmp_path / "equipment_yolo.pt"
    weights.write_bytes(b"x")
    cj = classes_file(tmp_path, {"names": {"0": "Excavator", "1": "Worker", "2": "Tipper"}})
    assert YoloDetector(weights=weights, classes_json=cj).supported_classes == ["excavator", "dump_truck"]
    # без своих весов — всё, что знает словарь YOLO-World
    from core import taxonomy
    assert YoloDetector(weights=tmp_path / "none.pt").supported_classes == list(taxonomy.equipment())


def test_world_vocabulary_maps_back_to_keys():
    from core import taxonomy
    vocab, index_to_key = detect_yolo.world_vocabulary()
    assert set(index_to_key.values()) - {None} == set(taxonomy.equipment())
    assert index_to_key[vocab.index("car")] is None and index_to_key[vocab.index("dump truck")] == "dump_truck"
