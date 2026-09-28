"""Одна машина — одна рамка: межклассовый NMS внутри групп путаницы и фильтры."""
from __future__ import annotations

from core.contracts import Detection
from core.equipment.config import EquipmentConfig
from core.equipment.postprocess import clean

W, H = 1280, 720


def det(cls, box, conf=0.8, **extra):
    return Detection(cls=cls, conf=conf, bbox=box, extra=dict(extra))


def test_truck_and_dump_truck_on_one_machine_collapse_to_one_box():
    dets = [det("truck", (400, 300, 200, 120), 0.61), det("dump_truck", (405, 298, 196, 124), 0.58)]
    out = clean(dets, W, H)
    assert len(out) == 1
    assert out[0].cls == "truck" and out[0].conf == 0.61
    assert out[0].extra["alt"] == [["dump_truck", 0.58]], "проигравшая метка остаётся голосом для трекера"


def test_crane_manipulator_box_inside_truck_box_is_suppressed():
    """Кран-манипулятор и «грузовик» на одной машине: меньшая рамка почти целиком внутри большей."""
    dets = [det("crane_manipulator", (400, 280, 220, 140), 0.7), det("truck", (430, 330, 150, 90), 0.5)]
    out = clean(dets, W, H)
    assert [d.cls for d in out] == ["crane_manipulator"]


def test_excavator_loading_dump_truck_keeps_both():
    """Разные группы путаницы: экскаватор законно перекрывает самосвал, который грузит."""
    dets = [det("excavator", (400, 250, 260, 200), 0.9), det("dump_truck", (520, 330, 240, 140), 0.85)]
    assert {d.cls for d in clean(dets, W, H)} == {"excavator", "dump_truck"}


def test_pump_and_mixer_side_by_side_keep_both():
    """Насос и миксер стоят вплотную и в перспективе перекрываются — это пара, а не дубль."""
    dets = [det("concrete_pump", (400, 250, 300, 180), 0.8), det("concrete_mixer", (440, 280, 220, 150), 0.75)]
    assert {d.cls for d in clean(dets, W, H)} == {"concrete_pump", "concrete_mixer"}


def test_small_and_edge_fragments_are_dropped_but_big_edge_machine_kept():
    dets = [
        det("excavator", (100, 100, 6, 30)),              # тоньше 10 px
        det("truck", (0, 400, 30, 25)),                   # маленький обрезок у левого края
        det("dump_truck", (1100, 300, 180, 150)),         # крупная машина, срезанная правым краем
    ]
    out = clean(dets, W, H)
    assert [d.cls for d in out] == ["dump_truck"]


def test_per_class_confidence_threshold():
    cfg = EquipmentConfig(conf_by_class={"tower_crane": 0.6})
    dets = [det("tower_crane", (100, 50, 200, 400), 0.5), det("excavator", (600, 300, 200, 150), 0.3)]
    out = clean(dets, W, H, cfg)
    assert [d.cls for d in out] == ["excavator"]


def test_unknown_class_dropped_and_boxes_clipped():
    dets = [det("forklift", (100, 100, 100, 100)), det("roller", (-20, 600, 200, 200))]
    out = clean(dets, W, H)
    assert len(out) == 1 and out[0].bbox == (0.0, 600.0, 180.0, 120.0)


def test_clean_is_idempotent_and_does_not_mutate_input():
    dets = [det("truck", (400, 300, 200, 120), 0.61), det("dump_truck", (405, 298, 196, 124), 0.58),
            det("excavator", (800, 300, 200, 150), 0.9)]
    once = clean(dets, W, H)
    twice = clean(once, W, H)
    assert [(d.cls, d.bbox, d.conf) for d in once] == [(d.cls, d.bbox, d.conf) for d in twice]
    assert twice[0].extra == once[0].extra
    assert all("alt" not in d.extra for d in dets)
