"""Маска фона: оператор задаёт начальную, дальше она только сжимается.

Сцена собрана так, чтобы проверить ровно то, ради чего метод и нужен:

    STAY — статичный соседний дом. Оператор его закрасил, и маска здесь
           обязана уцелеть, что бы ни делали погода и освещение.
    GROW — область, куда со временем вырастает наше здание. Оператор её тоже
           закрасил (на первом кадре там ещё небо), и маска обязана отступить.

Плюс проверяется главное свойство: маска не может вырасти ни при каких условиях.
"""

import numpy as np
import pytest

from app.pipeline import mask as M

H, W = 384, 480
GROUND_Y = 300
NEIGHBOUR = (20, 150, 120, GROUND_Y)     # x0, x1, y0, y1 — статичный дом слева
OURS_X = (300, 440)                      # наше здание растёт справа


def scene(floors: int, rng: np.random.Generator, weather: float = 0.0) -> np.ndarray:
    """Кадр: небо, земля, статичный сосед, наше здание высотой floors.

    `weather` сдвигает яркость всего кадра — имитация пасмурного дня, снега,
    другого освещения. Маска не должна на это реагировать.
    """
    img = np.zeros((H, W), dtype=np.uint8)
    img[:GROUND_Y, :] = np.clip(195 + rng.normal(0, 6, (GROUND_Y, W)) + weather, 0, 255)
    img[GROUND_Y:, :] = np.clip(95 + rng.normal(0, 4, (H - GROUND_Y, W)) + weather, 0, 255)

    x0, x1, y0, y1 = NEIGHBOUR
    img[y0:y1, x0:x1] = np.clip(135 + weather, 0, 255)      # сосед, неизменен

    if floors > 0:
        top = max(0, GROUND_Y - floors * 18)
        img[top:GROUND_Y, OURS_X[0]:OURS_X[1]] = np.clip(60 + weather, 0, 255)
    return img


def history(days: int) -> list[np.ndarray]:
    """Дневные медианы. Здание растёт на этаж каждые трое суток,
    погода гуляет синусоидой с заметной амплитудой."""
    out = []
    for d in range(days):
        rng = np.random.default_rng(d)
        weather = 30 * np.sin(d / 7.0)
        frames = [scene(d // 3, rng, weather) for _ in range(4)]
        out.append(M.daily_median(frames))
    return out


def operator_mask(shape: tuple[int, int]) -> np.ndarray:
    """Что закрасил бы оператор на первом кадре: всё выше земли —
    это небо и соседний дом, стройки там ещё нет."""
    h, w = shape
    bmp = np.zeros((H, W), np.uint8)
    bmp[:GROUND_Y, :] = 255
    return bmp


@pytest.fixture(scope="module")
def result() -> tuple[M.MaskState, np.ndarray]:
    days = history(60)
    shape = days[0].shape
    st = M.init_from_bitmap(operator_mask(shape), shape)
    initial = st.background.copy()
    win = M.WINDOW_DAYS
    for i in range(win, len(days) + 1):
        M.update(st, days[i - win:i])
    return st, initial


def _scale(st: M.MaskState) -> float:
    return st.shape[1] / W


def test_initial_mask_applied() -> None:
    days = history(12)
    st = M.init_from_bitmap(operator_mask(days[0].shape), days[0].shape)
    assert 0.6 < st.masked_ratio < 0.9, "оператор закрасил не то, что ожидалось"
    assert st.retained == 1.0


def test_mask_never_grows() -> None:
    """Главное свойство метода: маска монотонно убывает."""
    days = history(60)
    st = M.init_from_bitmap(operator_mask(days[0].shape), days[0].shape)
    prev = int(st.background.sum())
    for i in range(M.WINDOW_DAYS, len(days) + 1):
        M.update(st, days[i - M.WINDOW_DAYS:i])
        now = int(st.background.sum())
        assert now <= prev, "маска выросла — монотонность нарушена"
        prev = now


def test_static_neighbour_survives(result) -> None:
    """Статичный сосед закрашен и должен остаться закрашенным,
    несмотря на гуляющую погоду."""
    st, _ = result
    k = _scale(st)
    x0, x1, y0, y1 = NEIGHBOUR
    patch = st.background[int(y0 * k):int(y1 * k), int(x0 * k):int(x1 * k)]
    assert patch.mean() > 0.85, (
        f"маска на статичном соседе разрушена, цело только {patch.mean():.0%}")


def test_growing_building_erodes_mask(result) -> None:
    """Там, где выросло здание, маска обязана отступить."""
    st, _ = result
    k = _scale(st)
    x0, x1 = int(OURS_X[0] * k), int(OURS_X[1] * k)
    y0, y1 = int(120 * k), int(GROUND_Y * k)
    patch = st.background[y0:y1, x0:x1]
    assert patch.mean() < 0.4, (
        f"здание выросло, но маска не отступила: закрыто {patch.mean():.0%}")


def test_retained_between_zero_and_one(result) -> None:
    st, initial = result
    assert 0.0 < st.retained <= 1.0
    assert st.background.sum() <= initial.sum()


def test_top_edge_rises_as_building_grows() -> None:
    """Верхняя граница видимой области поднимается вместе со зданием —
    на этом держится измерение прогресса без запроса к модели."""
    days = history(70)
    st = M.init_from_bitmap(operator_mask(days[0].shape), days[0].shape)
    edges = []
    for i in range(M.WINDOW_DAYS, len(days) + 1):
        M.update(st, days[i - M.WINDOW_DAYS:i])
        e = st.top_edge()
        if e is not None:
            edges.append(e)
    assert len(edges) > 5
    assert edges[-1] < edges[0], f"граница не поднялась: {edges[0]} → {edges[-1]}"


def test_useful_flag() -> None:
    """Если скрывать нечего, система должна это признавать, а не
    делать вид, что фон отфильтрован."""
    shape = (384, 480)
    empty = M.MaskState(shape=shape)
    assert not empty.useful
    full = M.init_from_bitmap(np.full((H, W), 255, np.uint8), shape)
    assert full.useful


def test_render_masked_hides_background() -> None:
    shape = (384, 480)
    st = M.init_from_bitmap(operator_mask(shape), shape)
    frame = np.full((H, W, 3), 200, np.uint8)
    out = M.render_masked(frame, st, mode="black")
    assert out[:100, :].max() == 0, "фон не погашен"
    assert out[GROUND_Y + 20:, :].max() > 0, "объект погашен по ошибке"
