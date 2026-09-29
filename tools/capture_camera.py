#!/usr/bin/env python3
"""Снимать кадры с публичной онлайн-камеры стройки раз в N минут — «реальная камера подключена».

Каждый снимок сохраняется как `<объект>_<камера>_YYYY_MM_DD_HH_MM_SS.jpg` (местное время
объекта, по умолчанию Москва) в `<каталог объекта>/<камера>/` — это формат демо-объектов
(app/services/demo.py), так что накопленную серию засевает tools/seed_demo.py. С `--push-url`
кадр ещё и сразу уходит в сервис (POST /api/ingest с ключом камеры), как от настоящей камеры.

    # один проход (для systemd-таймера раз в 20 минут)
    python tools/capture_camera.py --site-dir datasets/demo/sites/ru_cityzen_live --site-key ru_cityzen_live \\
        --camera "cam_8och=https://rtsp2.videocam.online/thumbnail?application=live&streamname=Tushino2.stream&size=1920x1080&fitmode=letterbox&format=jpg" \\
        --referer https://videocam.online/ --once

    # сам по себе, без таймера: кадр в :00, :20, :40
    python tools/capture_camera.py ... --interval-min 20

Источник — адрес снимка (JPEG по HTTP: «превью»/thumbnail видеосервера, snapshot IP-камеры)
или видеопоток (`video:` + HLS/RTSP-адрес — один кадр берётся через OpenCV). Вежливость:
один запрос на камеру за проход, свой User-Agent, при ошибке — пропуск до следующего прохода.
Кадр, не изменившийся с прошлого раза (камера зависла, поток на паузе), не сохраняется.
Журнал проходов — `<каталог объекта>/capture_log.csv`.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

USER_AGENT = "StroyVzor-demo/1.0 (LCT-2026 hackathon; one snapshot per 20 min)"
MIN_BYTES = 5_000          # меньше — это заглушка «нет сигнала», а не кадр
MIN_SIDE = 320


class CaptureError(RuntimeError):
    pass


def parse_camera(spec: str) -> tuple[str, str]:
    """`cam_1=https://...` → ("cam_1", "https://...")."""
    name, sep, url = spec.partition("=")
    name, url = name.strip(), url.strip()
    if not sep or not name or not url:
        raise argparse.ArgumentTypeError(f"камера задаётся как имя=адрес: {spec!r}")
    if any(c in name for c in "/\\ ") or name.startswith("."):
        raise argparse.ArgumentTypeError(f"имя камеры — имя каталога без пробелов и слешей: {name!r}")
    return name, url


def frame_name(site_key: str, cam: str, when: dt.datetime) -> str:
    return f"{site_key}_{cam}_{when:%Y_%m_%d_%H_%M_%S}.jpg"


def next_slot(now: dt.datetime, interval_min: int) -> dt.datetime:
    """Ближайшая следующая отметка сетки: при 20 мин — :00, :20, :40."""
    step = interval_min * 60
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    passed = (now - day).total_seconds()
    return day + dt.timedelta(seconds=(int(passed // step) + 1) * step)


def mem_available_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def fetch_http(url: str, referer: str | None, timeout: float) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            data = resp.read(20_000_000)
    except urllib.error.HTTPError as exc:
        raise CaptureError(f"HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CaptureError(f"сеть: {exc}") from None
    if ctype and not ctype.startswith("image/"):
        raise CaptureError(f"не картинка: {ctype}")
    return data


def fetch_video(url: str) -> bytes:
    """Один кадр из HLS/RTSP-потока (OpenCV тяжёлый — импорт только здесь)."""
    import cv2

    cap = cv2.VideoCapture(url)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise CaptureError("поток не отдал кадр")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise CaptureError("кадр не кодируется в JPEG")
    return buf.tobytes()


def to_jpeg(data: bytes, max_side: int) -> bytes:
    """Проверить, что это кадр, и привести к JPEG (не больше max_side по длинной стороне)."""
    from PIL import Image

    if len(data) < MIN_BYTES:
        raise CaptureError(f"слишком маленький ответ ({len(data)} Б) — похоже на заглушку")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:  # noqa: BLE001 — любой мусор вместо картинки
        raise CaptureError(f"не декодируется: {exc}") from None
    if min(im.size) < MIN_SIDE:
        raise CaptureError(f"слишком маленький кадр {im.size[0]}×{im.size[1]}")
    if im.format == "JPEG" and max(im.size) <= max_side:
        return data
    im = im.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=90)
    return out.getvalue()


def push(url: str, camera_id: int, key: str, path: Path, when_local: dt.datetime, meta: dict,
         timeout: float) -> str:
    import requests

    from simcam.camera import local_proxies

    with path.open("rb") as fh:
        resp = requests.post(url, headers={"X-Camera-Key": key},
                             files={"file": (path.name, fh, "image/jpeg")},
                             data={"camera_id": str(camera_id), "captured_at": when_local.isoformat(),
                                   "meta": json.dumps(meta, ensure_ascii=False)},
                             proxies=local_proxies(url), timeout=timeout)
    if resp.status_code >= 400:
        raise CaptureError(f"приём {resp.status_code}: {resp.text[:200]}")
    return f"{resp.status_code}"


def last_hash(cam_dir: Path) -> str | None:
    marker = cam_dir / ".last_sha1"
    return marker.read_text().strip() if marker.exists() else None


def capture_once(args: argparse.Namespace, cameras: list[tuple[str, str]], tz: ZoneInfo) -> int:
    site_dir: Path = args.site_dir
    site_dir.mkdir(parents=True, exist_ok=True)
    log_path = site_dir / "capture_log.csv"
    new_log = not log_path.exists()
    saved = 0
    free = mem_available_mb()
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if new_log:
            wr.writerow(["captured_at", "camera", "status", "file", "bytes", "sha1", "note"])
        for cam, url in cameras:
            # местное время объекта с точностью до секунды: оно и в имени файла, и в captured_at
            when = dt.datetime.now(tz).replace(microsecond=0)
            stamp = when.replace(tzinfo=None)
            row = [when.isoformat(), cam]
            if args.min_free_mb and free is not None and free < args.min_free_mb:
                wr.writerow(row + ["skipped", "", "", "", f"свободно {free} МБ < {args.min_free_mb} МБ"])
                print(f"{cam}: пропуск — свободной памяти {free} МБ < {args.min_free_mb} МБ")
                continue
            try:
                raw = fetch_video(url[6:]) if url.startswith("video:") else fetch_http(url, args.referer, args.timeout)
                data = to_jpeg(raw, args.max_side)
            except CaptureError as exc:
                wr.writerow(row + ["error", "", "", "", str(exc)])
                print(f"{cam}: ошибка — {exc}", file=sys.stderr)
                continue
            sha = hashlib.sha1(data).hexdigest()
            cam_dir = site_dir / cam
            cam_dir.mkdir(exist_ok=True)
            if sha == last_hash(cam_dir):
                wr.writerow(row + ["unchanged", "", len(data), sha, "кадр не изменился — камера зависла?"])
                print(f"{cam}: кадр не изменился, не сохраняю")
                continue
            path = cam_dir / frame_name(args.site_key, cam, stamp)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            (cam_dir / ".last_sha1").write_text(sha)
            saved += 1
            note = ""
            if args.push_url and cam in args.push:
                cam_id, key = args.push[cam]
                try:
                    note = "push " + push(args.push_url, cam_id, key, path, stamp,
                                          {"camera": cam, "source": url.split("?")[0], "capture": "capture_camera.py"},
                                          args.timeout)
                except Exception as exc:  # noqa: BLE001 — сервис лежит: кадр сохранён, отправим руками
                    note = f"push не удался: {exc}"
            wr.writerow(row + ["saved", path.name, len(data), sha, note])
            print(f"{cam}: {path.name} ({len(data) // 1024} КБ){' · ' + note if note else ''}")
    return saved


def parse_push(items: list[str]) -> dict[str, tuple[int, str]]:
    """`cam_1=5:ключ` → {"cam_1": (5, "ключ")}."""
    out = {}
    for item in items:
        cam, _, rest = item.partition("=")
        cam_id, _, key = rest.partition(":")
        if not cam or not cam_id.isdigit() or not key:
            raise SystemExit(f"--push-camera задаётся как камера=id:ключ, а не {item!r}")
        out[cam] = (int(cam_id), key)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site-dir", type=Path, required=True, help="каталог объекта (в нём подкаталоги камер)")
    p.add_argument("--site-key", required=True, help="префикс имён файлов, обычно имя каталога объекта")
    p.add_argument("--camera", action="append", type=parse_camera, required=True,
                   help="имя=адрес снимка (или video:адрес потока); можно несколько раз")
    p.add_argument("--referer", help="заголовок Referer (страница, на которой камера опубликована)")
    p.add_argument("--timezone", default="Europe/Moscow", help="часовой пояс объекта (время в именах файлов)")
    p.add_argument("--interval-min", type=int, default=20, help="шаг съёмки без --once, минут")
    p.add_argument("--once", action="store_true", help="один проход и выход (для systemd-таймера)")
    p.add_argument("--max-side", type=int, default=1920, help="уменьшать кадр до этой длинной стороны")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--min-free-mb", type=int, default=0,
                   help="не снимать, если свободной памяти меньше (общий сервер); 0 — не проверять")
    p.add_argument("--push-url", help="адрес приёма кадров сервиса, например http://127.0.0.1:8000/api/ingest")
    p.add_argument("--push-camera", action="append", default=[], help="камера=id:ключ для --push-url")
    args = p.parse_args(argv)
    args.push = parse_push(args.push_camera)
    tz = ZoneInfo(args.timezone)

    if args.once:
        capture_once(args, args.camera, tz)
        return 0
    print(f"съёмка {len(args.camera)} камер(ы) раз в {args.interval_min} мин → {args.site_dir}")
    while True:
        slot = next_slot(dt.datetime.now(tz), args.interval_min)
        time.sleep(max(0.0, (slot - dt.datetime.now(tz)).total_seconds()))
        capture_once(args, args.camera, tz)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
