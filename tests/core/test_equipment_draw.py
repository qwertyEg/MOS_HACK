"""Отрисовка рамок и единая палитра классов."""
from __future__ import annotations

import numpy as np

from core import taxonomy
from core.contracts import Activity, Detection
from core.equipment import draw


def test_annotate_empty_list_returns_copy():
    img = np.full((360, 640, 3), 90, np.uint8)
    out = draw.annotate(img, [])
    assert out.shape == img.shape and out is not img
    assert np.array_equal(out, img)


def test_annotate_draws_boxes_and_does_not_touch_input():
    img = np.full((360, 640, 3), 90, np.uint8)
    dets = [Detection("excavator", 0.91, (100, 100, 200, 120), activity=Activity.WORKING),
            Detection("dump_truck", 0.7, (0, 0, 80, 60)),                 # подпись у верхнего края
            Detection("mystery", 0.5, (400, 200, 100, 100))]              # неизвестный класс — серый цвет
    out = draw.annotate(img, dets)
    assert (img == 90).all()
    assert tuple(int(v) for v in out[100 + 60, 100]) == draw.color_bgr("excavator")


def test_annotate_grayscale_input_and_without_fonts(monkeypatch):
    monkeypatch.setattr(draw, "_font", lambda size: None)       # тонкий Docker без TTF
    gray = np.full((200, 300), 128, np.uint8)
    out = draw.annotate(gray, [Detection("roller", 0.8, (50, 50, 100, 80))])
    assert out.shape == (200, 300, 3)


def test_label_text():
    d = Detection("excavator", 0.914, (0, 0, 10, 10), activity=Activity.WORKING)
    assert draw.label_for(d) == "Экскаватор · 0.91 · работает"
    d.extra["unit_status"] = "parked"
    assert draw.label_for(d) == "Экскаватор · 0.91 · на стоянке"
    assert draw.label_for(Detection("roller", 0.5, (0, 0, 1, 1)), labels_ru=False) == "roller · 0.50"
    assert draw._translit("Экскаватор · 0.91") == "Ekskavator | 0.91"


def test_palette_covers_taxonomy_with_distinct_colors():
    keys = set(taxonomy.equipment())
    assert keys <= set(draw.CLASS_COLORS)
    colors = [draw.CLASS_COLORS[k] for k in keys]
    assert len(set(colors)) == len(colors)
    tz = [np.array(draw.color_bgr(k), float) for k in taxonomy.TZ_EQUIPMENT]
    closest = min(np.linalg.norm(a - b) for i, a in enumerate(tz) for b in tz[i + 1:])
    assert closest > 60, "восемь классов ТЗ должны различаться на глаз"
