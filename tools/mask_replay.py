#!/usr/bin/env python3
"""Прогон динамической маски по истории одной камеры с наглядным выводом.

Это не только отладка: ролик с эволюцией маски по реальной стройке — самый
понятный способ показать, что метод работает, и лучший визуал для защиты.

По каждому кадру сохраняется панель из четырёх частей:

    исходный кадр            |  карта изменений за окно
    ------------------------ | ------------------------
    накопленный объект       |  что увидит модель Б
    (зелёным, только растёт) |  (фон погашен)

Плюс общий график: покрытие кадра и верхняя граница объекта по времени.
Верхняя граница — это и есть измеритель этажности: здание растёт вверх,
граница поднимается, и никакого запроса к VLM для этого не нужно.

    python tools/mask_replay.py data/raw_noon_images/for_test --out var/replay

Дата берётся из имени файла вида `doric_2006_11_23_12_30_21.jpg`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.pipeline import mask as M
from app.pipeline.model_b import MaskMode, apply_mask
from PIL import Image

STAMP = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")


def parse_date(path: Path) -> dt.datetime | None:
    m = STAMP.search(path.name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    try:
        return dt.datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Подпись на панели, читаемая и на светлом, и на тёмном фоне."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def overlay(base: np.ndarray, mask: np.ndarray, color, alpha=0.45) -> np.ndarray:
    """Полупрозрачная заливка маски поверх кадра."""
    out = base.copy()
    tint = np.zeros_like(base)
    tint[:] = color
    m3 = np.dstack([mask] * 3)
    out = np.where(m3, (base * (1 - alpha) + tint * alpha).astype(np.uint8), out)
    return out


def panel(frame: np.ndarray, dynamic: np.ndarray, state: M.MaskState,
          visible: np.ndarray, w: int = 480) -> np.ndarray:
    h = int(frame.shape[0] * w / frame.shape[1])
    small = cv2.resize(frame, (w, h))

    def up(m):
        return cv2.resize(m.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    a = label(small, "1. исходный кадр")
    b = label(overlay(small, up(dynamic), (0, 200, 255)),
              f"2. изменения за окно  ({dynamic.mean():.1%} кадра)")
    c = label(overlay(small, up(state.object_mask), (0, 255, 0)),
              f"3. накоплено  ({state.coverage:.1%}, окон {state.windows})")

    # Четвёртая панель — ровно то, что уйдёт в модель Б.
    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    shown = apply_mask(pil, up(visible), MaskMode.DARKEN)
    d = label(cv2.cvtColor(np.array(shown), cv2.COLOR_RGB2BGR),
              "4. что увидит модель Б")

    top = np.hstack([a, b])
    bottom = np.hstack([c, d])
    return np.vstack([top, bottom])


def plot_history(hist: list[dict], out: Path, frame_h: int) -> None:
    """График покрытия и верхней границы объекта во времени."""
    if len(hist) < 2:
        return
    W, H = 1000, 340
    img = np.full((H, W, 3), 255, np.uint8)
    pad = 55
    n = len(hist)

    cov = [h["coverage"] for h in hist]
    top = [h["top_edge"] if h["top_edge"] is not None else frame_h for h in hist]

    def x(i):
        return int(pad + i * (W - 2 * pad) / max(1, n - 1))

    # покрытие кадра, левая ось
    mx = max(cov) or 1.0
    pts = [(x(i), int(H - pad - c / mx * (H - 2 * pad))) for i, c in enumerate(cov)]
    for p, q in zip(pts, pts[1:]):
        cv2.line(img, p, q, (0, 160, 0), 2, cv2.LINE_AA)

    # верхняя граница объекта, правая ось (инвертирована: вверх = выше)
    pts2 = [(x(i), int(pad + t / frame_h * (H - 2 * pad))) for i, t in enumerate(top)]
    for p, q in zip(pts2, pts2[1:]):
        cv2.line(img, p, q, (200, 60, 0), 2, cv2.LINE_AA)

    cv2.rectangle(img, (pad, pad), (W - pad, H - pad), (200, 200, 200), 1)
    cv2.putText(img, "green: coverage of frame   blue: top edge of object (higher = taller)",
                (pad, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, hist[0]["date"][:10], (pad, H - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, hist[-1]["date"][:10], (W - pad - 90, H - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("var/replay"))
    ap.add_argument("--window", type=int, default=10,
                    help="размер окна в днях для карты изменений")
    ap.add_argument("--lock", type=int, default=3,
                    help="окон подряд до фиксации точки за объектом")
    ap.add_argument("--max-growth", type=float, default=85.0,
                    help="потолок прироста одной компоненты за окно, %% кадра")
    ap.add_argument("--every", type=int, default=1,
                    help="сохранять панель каждый N-й кадр")
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N кадров")
    args = ap.parse_args()

    files = sorted(p for p in args.images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        files = files[:args.limit]
    if len(files) < args.window + 2:
        print(f"нужно хотя бы {args.window + 2} кадров, есть {len(files)}",
              file=sys.stderr)
        return 1

    panels_dir = args.out / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    print(f"кадров {len(files)}, окно {args.window} дн., фиксация после "
          f"{args.lock} окон, потолок прироста {args.max_growth}%\n")

    state: M.MaskState | None = None
    ring: list[np.ndarray] = []      # дневные медианы последних N дней
    hist: list[dict] = []
    saved = 0
    frame_h = 0

    for i, path in enumerate(files):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  пропущен нечитаемый {path.name}")
            continue
        frame_h = frame.shape[0]

        # Кадр в сутки — значит дневная медиана это он сам. На реальном потоке
        # (кадр раз в 20 минут) сюда пойдёт медиана всех дневных кадров суток.
        day = M.daily_median([frame])
        if state is None:
            state = M.MaskState(shape=day.shape)
        ring.append(day)
        if len(ring) > args.window:
            ring.pop(0)
        if len(ring) < 3:
            continue

        dynamic = M.dynamics_map(ring)
        M.update(state, ring, lock_windows=args.lock,
                 max_growth_pct=args.max_growth)
        visible = M.visible_mask(state, dynamic)

        when = parse_date(path)
        hist.append({
            "file": path.name,
            "date": when.isoformat() if when else "",
            "coverage": round(state.coverage, 5),
            "dynamic_pct": round(float(dynamic.mean()), 5),
            "top_edge": state.top_edge(),
            "windows": state.windows,
            "converged": state.converged,
        })

        if i % args.every == 0:
            img = panel(frame, dynamic, state, visible)
            stamp = when.strftime("%Y-%m-%d") if when else f"{i:04d}"
            cv2.putText(img, stamp, (img.shape[1] - 130, img.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(panels_dir / f"{i:04d}_{stamp}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved += 1

        if i % 25 == 0:
            te = state.top_edge()
            print(f"  {i:4d}/{len(files)}  {path.name[:34]:<34} "
                  f"покрытие {state.coverage:5.1%}  верх.граница "
                  f"{te if te is not None else '—'}")

    if state is None or not hist:
        print("нечего показывать", file=sys.stderr)
        return 1

    plot_history(hist, args.out / "history.png", state.shape[0])
    (args.out / "history.json").write_text(
        json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

    # Финальная маска отдельным файлом — удобно смотреть в полном размере.
    cv2.imwrite(str(args.out / "final_mask.png"),
                (state.object_mask.astype(np.uint8) * 255))

    print(f"\n{'─' * 58}")
    print(f"панелей сохранено : {saved} → {panels_dir}")
    print(f"график            : {args.out / 'history.png'}")
    print(f"числа по кадрам   : {args.out / 'history.json'}")
    print(f"итоговая маска    : {args.out / 'final_mask.png'}")
    print(f"покрытие кадра    : {state.coverage:.1%}")
    te0 = next((h["top_edge"] for h in hist if h["top_edge"] is not None), None)
    te1 = state.top_edge()
    if te0 is not None and te1 is not None:
        print(f"верхняя граница   : {te0} → {te1} px "
              f"({'поднялась' if te1 < te0 else 'не поднялась'})")
    print(f"\nролик:  ffmpeg -framerate 12 -pattern_type glob "
          f"-i '{panels_dir}/*.jpg' -c:v libx264 -pix_fmt yuv420p "
          f"{args.out / 'replay.mp4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
