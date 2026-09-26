"""Подготовка кадра: дата съёмки, хэш, сжатие для отправки в модель."""

import base64
import hashlib
import io
import re
from datetime import datetime

from PIL import ExifTags, Image, ImageOps

from .config import IMAGE_MAX_SIDE, JPEG_QUALITY

_EXIF_DATE_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime

# Порядок важен: сначала форматы с временем, иначе «2025-03-14_10-30» распознается без него.
_NAME_PATTERNS = [
    (r"(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})[ T_-]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})", "ymdhms"),
    (r"(\d{4})[-_.](\d{2})[-_.](\d{2})", "ymd"),
    (r"(\d{2})[-_.](\d{2})[-_.](\d{4})", "dmy"),
    (r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", "ymd"),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def date_from_exif(data: bytes):
    try:
        exif = Image.open(io.BytesIO(data)).getexif()
    except Exception:
        return None
    # DateTimeOriginal лежит не в основном IFD, а во вложенном Exif IFD.
    sources = [exif, exif.get_ifd(ExifTags.IFD.Exif)]
    for tag in _EXIF_DATE_TAGS:
        for src in sources:
            value = src.get(tag)
            if value:
                try:
                    return datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    continue
    return None


def date_from_name(name: str):
    for pattern, kind in _NAME_PATTERNS:
        m = re.search(pattern, name)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        try:
            if kind == "ymdhms":
                return datetime(*g)
            if kind == "ymd":
                return datetime(g[0], g[1], g[2])
            return datetime(g[2], g[1], g[0])
        except ValueError:
            continue
    return None


def detect_date(data: bytes, name: str):
    """Возвращает (дата, источник). Источник показываем в UI, чтобы оператор видел, откуда дата."""
    dt = date_from_exif(data)
    if dt:
        return dt, "exif"
    dt = date_from_name(name)
    if dt:
        return dt, "имя файла"
    return None, ""


def prepare(data: bytes) -> bytes:
    """Поворот по EXIF, RGB, сжатие длинной стороны до IMAGE_MAX_SIDE, JPEG."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
