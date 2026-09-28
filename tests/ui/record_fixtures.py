"""Записать фикстуры макета UI с живого бэкенда: настоящие ответы JSON API и картинки.

Макет tools/ui_mock.py отдаёт ровно то, что отдавал бэкенд, — формы ответов
не расходятся с app/services/views.py, пока фикстуры перезаписываются этим
скриптом. Запускать там, где крутится бэкенд с засеянными демо-объектами
(на сервере, не на ноутбуке):

    python tests/ui/record_fixtures.py --base http://127.0.0.1:18100 --sites 1,5

Пишет tests/ui/fixtures/api/*.json (ключ — «путь?запрос») и
tests/ui/fixtures/media/* (превью кадров-доказательств и кропов). Только stdlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "fixtures"


def key_for(path: str) -> str:
    """Имя файла фикстуры по пути с запросом (запрос — в отсортированном виде)."""
    p = urllib.parse.urlsplit(path)
    q = urllib.parse.urlencode(sorted(urllib.parse.parse_qsl(p.query)))
    raw = p.path + (f"?{q}" if q else "")
    slug = raw.strip("/").replace("/", "__").replace("?", "__q__").replace("&", "_").replace("=", "-")
    if len(slug) > 120:
        slug = slug[:100] + "_" + hashlib.sha1(raw.encode()).hexdigest()[:10]
    return slug


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def req(self, method: str, path: str, body: dict | None = None) -> bytes:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method=method,
                                   headers={"Content-Type": "application/json"} if data else {})
        with self.opener.open(r, timeout=120) as resp:
            return resp.read()

    def get(self, path: str):
        return json.loads(self.req("GET", path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18100")
    ap.add_argument("--sites", default="", help="id объектов через запятую (по умолчанию — два с отклонениями)")
    ap.add_argument("--frames-per-camera", type=int, default=10)
    ap.add_argument("--login", default="admin")
    ap.add_argument("--password", default="admin")
    args = ap.parse_args()

    c = Client(args.base)
    c.req("POST", "/api/login", {"login": args.login, "password": args.password})
    api_dir, media_dir = ROOT / "api", ROOT / "media"
    for d in (api_dir, media_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    index: dict[str, str] = {}
    media: set[str] = set()

    def scrub(x):
        # Ключи приёма кадров стенда в git не кладём.
        if isinstance(x, dict):
            return {k: ("mock-ingest-key" if k == "ingest_key" else scrub(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [scrub(v) for v in x]
        return x

    def save(path: str):
        data = scrub(c.get(path))
        k = key_for(path)
        (api_dir / f"{k}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        index[path] = k
        return data

    for p in ("/api/me", "/api/health", "/api/settings", "/api/catalog", "/api/queue", "/api/jobs"):
        save(p)
    sites = save("/api/sites")
    ids = [int(x) for x in args.sites.split(",") if x] or [
        s["id"] for s in sorted(sites, key=lambda s: -(s.get("open_deviations") or 0))[:2]]
    for s in sites:
        if s.get("thumb"):
            media.add(s["thumb"])
        # Обзор считает критичные отклонения по каждому объекту.
        save(f"/api/sites/{s['id']}/deviations?limit=500&status=open")

    for sid in ids:
        save(f"/api/sites/{sid}")
        ov = save(f"/api/sites/{sid}/overview")
        cams = save(f"/api/sites/{sid}/cameras")
        eq = save(f"/api/sites/{sid}/equipment")
        devs = save(f"/api/sites/{sid}/deviations?limit=500&status=all")
        save(f"/api/sites/{sid}/plan")
        save(f"/api/sites/{sid}/fleet")
        save(f"/api/sites/{sid}/hours")
        save(f"/api/sites/{sid}/hours?manual=false")
        frame_ids: set[int] = set()
        for cam in ov.get("cameras", []):
            if cam.get("last_frame"):
                frame_ids.add(cam["last_frame"]["id"])
                media.add(cam["last_frame"]["url"])
        for cam in cams:
            cid = cam["id"]
            save(f"/api/cameras/{cid}")
            save(f"/api/cameras/{cid}/zones")
            save(f"/api/cameras/{cid}/frames?limit=1&order=asc")
            frames = save(f"/api/cameras/{cid}/frames?limit=500")
            with_dets = [f for f in frames if f.get("detections_count")]
            pick = (with_dets[: args.frames_per_camera // 2] + frames[: args.frames_per_camera // 2])
            frame_ids.update(f["id"] for f in pick)
        for d in devs:
            for f in d.get("frames", [])[:4]:
                frame_ids.add(f["id"])
        for u in eq.get("units", [])[:24]:
            det = save(f"/api/units/{u['id']}")
            for x in det.get("detections", [])[:1]:
                media.add(x["url"])
        for fid in sorted(frame_ids):
            fr = save(f"/api/frames/{fid}")
            media.add(fr["url"])
            (media_dir / f"annotated_{fid}.jpg").write_bytes(c.req("GET", f"/api/frames/{fid}/annotated.jpg?max_w=480"))

    for url in sorted(media):
        if url and url.startswith("/media/"):
            (media_dir / key_for(url)).write_bytes(c.req("GET", url))
    (ROOT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    print(f"записано: {len(index)} ответов API, {len(media)} картинок, {size / 1e6:.1f} МБ; объекты {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
