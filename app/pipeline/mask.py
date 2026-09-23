"""Динамическая маска фона: оператор задаёт начальную, дальше она только сжимается.

Полное описание метода — PLAN.md §3.4. Здесь то, что нужно знать, чтобы не
«упростить» код в нерабочее состояние.

**Что такое маска.** Маска — это область ФОНА, то есть то, что скрывается от
модели Б: соседние достроенные дома, улица, горизонт. Всё остальное — наш
объект. Модель видит `~background`.

**Начальное условие задаёт человек.** При заведении камеры оператор получает
первый кадр и закрашивает фон кистью. Это снимает два ограничения, которые в
автоматическом варианте были принципиальными: не нужно ждать недели, пока
маска сойдётся, и неважно, что при установке камеры уже стоят пять этажей.

**Маска только сжимается, никогда не растёт.** Здание растёт вверх и заползает
на фон — значит фона становится меньше. Обратное физически невозможно: сама
собой застройка не исчезает.

Направление монотонности выбрано не случайно, а по тому, куда ведёт ошибка.
Накопительная фиксация необратима в любом варианте, вопрос только в цене сбоя:

    маска фона растёт  →  в пределе скрыто всё, объект не виден, разбор сорван
    маска фона сжимается → в пределе не скрыто ничего, то есть откат к полному
                           кадру. Хуже, чем без маски, стать не может.

Ошибки оператора тоже ведут себя прилично. Закрасил лишнего, задев стройку —
стройка меняется, алгоритм сам её откроет, ошибка самоисправляется. Закрасил
мало — часть фона останется видимой, но это ровно то же, что работа без маски.

**Почему сравнение по яркости, а не по структуре.** Проверено замером на
319 кадрах Edinburgh Informatics Forum. Клеточные гистограммы ориентаций
градиента дают лучшее сырое разделение фона и стройки (3.31x против 2.14x),
но в сквозном тесте проигрывают: F1 0.67 против 0.86. Явные отрезки прямых
проигрывают разгромно, F1 0.10 — на линиях лежит 0.5–3% пикселей кадра,
9–51 отрезок на весь кадр, сигнала просто нет (на полном разрешении хуже,
а не лучше). Причина в том, что фон в этой сцене — деревья и перепаханная
земля, а не оконные сетки: структурно они нестабильны не меньше стройки.

Оговорка на будущее: в целевом сценарии фон другой — соседние жилые дома с
устойчивой геометрией окон, и там структурный признак может выиграть.
Но строить на нём сейчас нельзя, на имеющихся данных он измеримо хуже.

**Освещение снимается по опорным клеткам, а не по всей маске.** Пасмурный
день, снег и смена сезона двигают яркость всего кадра разом. Чтобы это не
читалось как застройка, из измеренного изменения вычитается общий сдвиг.
Считать его по всей маске нельзя: когда здание отвоевало её большую часть,
такая медиана измеряет уже стройку и вычитает сигнал сам из себя. Опора
берётся по самым спокойным из ещё закрашенных клеток — это и есть настоящий
неподвижный фон. Замер на двух прогонах Эдинбурга: F1 0.39 → 0.42 и
0.73 → 0.80, от исходной маски доживает 15% → 28% и 14% → 19%.

**Чего яркость не умеет.** На прогоне, где стройка занимает лишь пятую часть
закрашенного, точность упирается в 0.33 при любых порогах. Причина измерена:
71% фоновых клеток хоть раз дают вспышку изменения длиной в три окна —
снег, мокрый асфальт, переставленный контейнер, техника у границы площадки.
По силе и частоте изменения фон от стройки на таких данных не отделяется.

Проверено и отвергнуто, числа — на двух реальных прогонах разом:
отложенное подтверждение стирания по обратимости (точность на трудном
прогоне 0.27 → 0.31, но на удачном F1 0.80 → 0.72), требование связности
растущей области снизу вверх (0.80 → 0.60) и отбрасывание одиночных клеток
(0.80 → 0.53). Все три покупают точность на трудном прогоне ценой удачного.
Разумный следующий шаг — не признак, а право оператора пометить область
как неприкосновенную: там, где он уверен, гарантия нужнее эвристики.

**Что в расчёт не идёт.** Ночные кадры: ИК-режим даёт другую статистику
яркости, и смешивание дня с ночью развалит сравнение — половина кадров будет
«отличаться» просто из-за режима съёмки, а не потому, что там что-то построили.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import cv2
import numpy as np

# Значения подобраны замером, см. шапку модуля.
WORK_WIDTH = 480          # маска нужна грубая, на полном разрешении считать незачем
BLUR_KSIZE = 5
CELL = 16                 # сторона клетки, в которых считается изменение
WINDOW_DAYS = 10          # база сравнения: половина окна против половины
CHANGE_THRESHOLD = 35.0   # порог изменения яркости клетки
LOCK_WINDOWS = 3          # окон подряд до стирания клетки из маски
ANCHOR_FRAC = 0.5         # доля самых спокойных клеток под опору освещения


@dataclass
class MaskState:
    """Состояние маски по одной камере. Живёт месяцами."""
    shape: tuple[int, int]                        # (h, w) рабочего разрешения
    background: np.ndarray = field(default=None)  # bool, True = фон. Только убывает
    evidence: np.ndarray = field(default=None)    # int16, по клеткам
    hot_count: np.ndarray = field(default=None)   # как часто клетка менялась
    initial_area: int = 0
    windows: int = 0
    last_reset: dt.datetime | None = None

    def __post_init__(self) -> None:
        h, w = self.shape
        grid = (h // CELL, w // CELL)
        if self.background is None:
            self.background = np.zeros((h, w), dtype=bool)
        if self.evidence is None:
            self.evidence = np.zeros(grid, dtype=np.int16)
        if self.hot_count is None:
            self.hot_count = np.zeros(grid, dtype=np.int32)
        if not self.initial_area:
            self.initial_area = int(self.background.sum())

    @property
    def grid(self) -> tuple[int, int]:
        h, w = self.shape
        return h // CELL, w // CELL

    @property
    def masked_ratio(self) -> float:
        """Какая доля кадра сейчас скрыта."""
        return float(self.background.mean())

    @property
    def retained(self) -> float:
        """Какая доля исходной маски ещё цела. Падает по мере роста здания."""
        if not self.initial_area:
            return 0.0
        return float(self.background.sum()) / self.initial_area

    @property
    def useful(self) -> bool:
        """Даёт ли маска что-то вообще.

        Если скрывать почти нечего, честнее сказать об этом и отдать модели
        полный кадр, чем делать вид, что фон отфильтрован.
        """
        return self.masked_ratio > 0.03

    def top_edge(self) -> int | None:
        """Верхняя граница видимой области — растёт вместе со зданием.

        Это и есть измеритель прогресса монолита: здание тянется вверх,
        отвоёвывая маску, граница поднимается. Абсолютный счёт этажей от
        модели Б для этого не нужен, нужен только рост.
        """
        vis = ~self.background
        ys = np.where(vis.any(axis=1))[0]
        return int(ys.min()) if len(ys) else None


# ---------------------------------------------------------------------------
# подготовка кадров
# ---------------------------------------------------------------------------

def prepare(img: np.ndarray) -> np.ndarray:
    """К серому, к рабочему разрешению, лёгкое размытие против шума матрицы."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    target_h = max(1, int(h * WORK_WIDTH / w))
    # Высота кратна клетке, чтобы сетка ложилась ровно.
    target_h -= target_h % CELL
    small = cv2.resize(img, (WORK_WIDTH, target_h), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (BLUR_KSIZE, BLUR_KSIZE), 0)


def daily_median(frames: list[np.ndarray]) -> np.ndarray | None:
    """Медиана кадров одного дня.

    Медиана, а не среднее: она устойчива к выбросам, поэтому проехавший
    грузовик, прошедший человек и метущая по кадру стрела крана её не сдвигают —
    в каждом кадре они в разном месте.
    """
    if not frames:
        return None
    stack = np.stack([prepare(f) for f in frames])
    return np.median(stack, axis=0).astype(np.float32)


def init_from_bitmap(bitmap: np.ndarray, shape: tuple[int, int]) -> MaskState:
    """Маска, нарисованная оператором → состояние.

    bitmap: любое разрешение, ненулевые пиксели = фон (скрыть).
    """
    h, w = shape
    m = cv2.resize(bitmap.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return MaskState(shape=shape, background=(m > 0))


# ---------------------------------------------------------------------------
# сжатие маски
# ---------------------------------------------------------------------------

def change_map(day_medians: list[np.ndarray]) -> np.ndarray:
    """Изменение по клеткам: медиана второй половины окна против первой.

    Сравниваются две выборки, а не считается разброс внутри окна. Причина:
    строительство — это последовательность ступенек, точка была небом, стала
    стеной и дальше не меняется. Медианное абсолютное отклонение (MAD) по
    построению устойчиво к выбросам, а значит и к ступеньке: если она прошла
    не ровно посередине окна, большинство значений одинаковы и MAD около нуля.
    Замер: рост этажа детектировался в одном окне из трёх.

    Разность медиан половин гасит транспорт и людей так же хорошо (медиана
    внутри каждой половины), но ступеньку видит всегда, пока та попадает в окно.
    """
    stack = np.stack(day_medians)
    half = max(1, len(stack) // 2)
    first = np.median(stack[:half], axis=0)
    second = np.median(stack[-half:], axis=0)
    diff = np.abs(second - first)

    gh, gw = diff.shape[0] // CELL, diff.shape[1] // CELL
    return cv2.resize(diff, (gw, gh), interpolation=cv2.INTER_AREA)


def _cells(frame: np.ndarray) -> np.ndarray:
    """Кадр рабочего разрешения → сетка клеток."""
    gh, gw = frame.shape[0] // CELL, frame.shape[1] // CELL
    return cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA)


def _anchor(state: MaskState, masked_cells: np.ndarray) -> tuple:
    """Клетки, по которым меряется общая засветка кадра.

    Берутся самые спокойные из ещё закрашенных: это и есть настоящий
    неподвижный фон — стена соседнего дома, асфальт, горизонт.

    Медиану по всей маске брать нельзя. Когда здание отвоевало её большую
    часть, такая медиана измеряет уже стройку и вычитает сама себя — сигнал
    глохнет ровно там, где он есть. Замер: полнота падала с 0.91 до 0.64.
    """
    ys, xs = np.where(masked_cells)
    if len(ys) < 15:
        return None
    order = np.argsort(state.hot_count[ys, xs])
    keep = max(15, int(len(order) * ANCHOR_FRAC))
    return ys[order[:keep]], xs[order[:keep]]


def update(
    state: MaskState,
    day_medians: list[np.ndarray],
    threshold: float = CHANGE_THRESHOLD,
    lock_windows: int = LOCK_WINDOWS,
) -> MaskState:
    """Один шаг: окно дневных медиан → сжатая маска.

    background после вызова никогда не больше, чем был.
    """
    # Полуокна сравниваются только на полном окне. На огрызке в четыре дня
    # половины слишком коротки, сравнение шумит и плодит стирания на ровном
    # месте: точность на реальном прогоне падала с 0.31 до 0.28.
    if len(day_medians) < WINDOW_DAYS:
        return state

    stack = np.stack(day_medians)
    half = max(1, len(stack) // 2)
    now = _cells(np.median(stack[-half:], axis=0))
    before = _cells(np.median(stack[:half], axis=0))

    gh, gw = state.grid
    h, w = state.shape
    masked_cells = cv2.resize(state.background.astype(np.uint8), (gw, gh),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

    anchor = _anchor(state, masked_cells)
    offset = float(np.median((now - before)[anchor])) if anchor else 0.0

    hot = np.abs(now - before - offset) > threshold
    state.hot_count += hot

    # Счётчик со спадом. Устойчивое изменение (растущая стена) накапливается
    # и пробивает порог; разовое (припарковавшаяся машина, облако, мокрый
    # асфальт) откатывается назад и порога не достигает.
    state.evidence = np.where(hot, state.evidence + 1,
                              np.maximum(state.evidence - 1, 0)).astype(np.int16)

    erase_cells = (state.evidence >= lock_windows) & masked_cells
    if erase_cells.any():
        erase = cv2.resize(erase_cells.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        state.background = state.background & ~erase

    state.windows += 1
    return state


def visible_mask(state: MaskState) -> np.ndarray:
    """Область, которую видит модель Б: всё, что не закрыто фоном."""
    return ~state.background


def to_full_res(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Рабочее разрешение → полный кадр."""
    up = cv2.resize(mask.astype(np.uint8), (width, height),
                    interpolation=cv2.INTER_NEAREST)
    return up.astype(bool)


# ---------------------------------------------------------------------------
# перестановка камеры
# ---------------------------------------------------------------------------

def detect_shift(reference: np.ndarray, current: np.ndarray) -> tuple[bool, float]:
    """Сменился ли ракурс. Возвращает (сменился, доля совпавших точек).

    Важно: эталон надо держать свежим, а не брать первый кадр стройки.
    Сравнение «пустой пустырь против готового здания» не даст совпадений
    даже при намертво прибитой камере — сцена изменилась целиком, и это
    не перестановка. Эталон обновляется вместе с наблюдением.
    """
    ref, cur = prepare(reference), prepare(current)
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(ref, None)
    kp2, des2 = orb.detectAndCompute(cur, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return True, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)
    if len(matches) < 12:
        return True, 0.0

    good = matches[:max(12, len(matches) // 2)]
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or inliers is None:
        return True, 0.0
    ratio = float(inliers.sum()) / len(good)
    return ratio < 0.25, ratio


def reset(state: MaskState) -> MaskState:
    """Сброс после перестановки камеры.

    Переносить маску гомографией здесь нельзя: новая точка съёмки может
    показывать совсем другой фон. Честнее попросить оператора нарисовать
    заново, чем незаметно подсунуть неверную маску.
    """
    h, w = state.shape
    state.background = np.zeros((h, w), dtype=bool)
    state.evidence = np.zeros(state.grid, dtype=np.int16)
    state.hot_count = np.zeros(state.grid, dtype=np.int32)
    state.initial_area = 0
    state.windows = 0
    state.last_reset = dt.datetime.now(dt.UTC)
    return state


# ---------------------------------------------------------------------------
# визуализация
# ---------------------------------------------------------------------------

def render_masked(frame: np.ndarray, state: MaskState, mode: str = "darken") -> np.ndarray:
    """Кадр с погашенным фоном — ровно то, что уйдёт в модель Б."""
    h, w = frame.shape[:2]
    bg = to_full_res(state.background, w, h)
    out = frame.copy()
    if mode == "black":
        out[bg] = 0
    elif mode == "blur":
        blurred = cv2.GaussianBlur(frame, (41, 41), 0)
        out[bg] = blurred[bg]
    else:
        out[bg] = (out[bg] * 0.28).astype(np.uint8)
    return out


def render_overlay(frame: np.ndarray, state: MaskState) -> np.ndarray:
    """Кадр с маской, залитой красным — для оценки границ глазом."""
    h, w = frame.shape[:2]
    bg = to_full_res(state.background, w, h)
    out = frame.copy()
    tint = np.zeros_like(frame)
    tint[:] = (0, 0, 220)
    out[bg] = (frame[bg] * 0.5 + tint[bg] * 0.5).astype(np.uint8)
    return out
