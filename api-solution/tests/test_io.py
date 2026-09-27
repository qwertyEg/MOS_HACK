"""Разбор ответа модели, даты кадров, разбор плана."""

import io
from datetime import date, datetime

import pytest
from PIL import Image

from core.glm import extract_json
from core.images import date_from_name, detect_date, prepare
from core.plan import parse_plan_file


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    'Вот ответ:\n```json\n{"a": 1}\n```',
    '<think>рассуждаю {"b": 2}</think>{"a": 1}',
    '<|begin_of_box|>{"a": 1}<|end_of_box|>',
    '{"a": 1} и ещё пояснение {не json}',
])
def test_extract_json(text):
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text,expected", [
    ('{"floors_built": <int или null>, "view": "top"}', {"floors_built": None, "view": "top"}),
    ('{"equipment": [{"type": "excavator",},], "n": 1,}', {"equipment": [{"type": "excavator"}], "n": 1}),
])
def test_extract_json_repairs_common_breakage(text, expected):
    assert extract_json(text) == expected


def test_extract_json_fails_without_object():
    with pytest.raises(ValueError):
        extract_json("не могу ответить")


@pytest.mark.parametrize("name,expected", [
    ("cam1_2025-03-14_10-30-05.jpg", datetime(2025, 3, 14, 10, 30, 5)),
    ("IMG_20250314_103005.jpg", datetime(2025, 3, 14, 10, 30, 5)),
    ("site 2025-03-14.jpg", datetime(2025, 3, 14)),
    ("14.03.2025.png", datetime(2025, 3, 14)),
    ("20250314.jpg", datetime(2025, 3, 14)),
    ("photo.jpg", None),
])
def test_date_from_name(name, expected):
    assert date_from_name(name) == expected


def _jpeg(size=(2000, 1000), exif_date=None):
    img = Image.new("RGB", size, (120, 120, 120))
    exif = Image.Exif()
    if exif_date:
        exif[306] = exif_date
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_exif_date_beats_filename():
    dt, src = detect_date(_jpeg(exif_date="2024:05:01 08:00:00"), "2025-03-14.jpg")
    assert dt == datetime(2024, 5, 1, 8, 0) and src == "exif"


def test_prepare_limits_long_side():
    out = Image.open(io.BytesIO(prepare(_jpeg((4000, 2000)))))
    assert max(out.size) == 1280


def test_parse_plan_csv():
    csv = "этап;начало;окончание\n1;01.02.2025;15.03.2025\n5;2025-06-01;2026-03-01\n".encode()
    assert parse_plan_file(csv, "plan.csv") == {
        1: (date(2025, 2, 1), date(2025, 3, 15)),
        5: (date(2025, 6, 1), date(2026, 3, 1)),
    }
