"""Скриншоты всех страниц на живом бэкенде (или на макете tools/ui_mock.py).

Запускается в docker-контейнере playwright на сервере, не на ноутбуке:

    docker run --rm --network host --cpus=2 --memory=3g \\
      -v /root/mos_hack/shots:/shots -v <репозиторий>/tests/ui:/tools \\
      mcr.microsoft.com/playwright/python:v1.49.0-noble \\
      python /tools/shoot.py --base http://127.0.0.1:18100 --out /shots/ui/it1

Идентификаторы объекта, камеры, кадра и отклонения берутся из самого API
(первый объект с кадрами и отклонениями), поэтому скрипт работает на любой
засеянной базе. Кроме картинок печатает ошибки консоли браузера, исключения
страницы, упавшие запросы и ответы API с кодом ≥ 400 — по ним видно, что
страница не просто нарисовалась, а отработала без ошибок.

    --only site        только снимки, в имени которых есть «site»
    --mock             базовый адрес — макет (есть /__mock для пустых состояний)
    --site 2           взять конкретный объект
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# Ответы, которые ошибкой не считаются: маски у камеры может ещё не быть (404 — штатно).
EXPECTED_4XX = ("/mask.png",)


def pick_ids(req, base: str, want_site: str | None) -> dict:
    """Объект с кадрами и отклонениями, его камера с кадрами, кадр с рамками."""
    sites = req.get(f"{base}/api/sites").json()
    ids: dict = {"site": None, "camera": None, "frame": None, "night": None, "sample": None}
    if not sites:
        return ids
    cands = [s for s in sites if str(s["id"]) == want_site] if want_site else sorted(
        sites, key=lambda s: (-(s.get("open_deviations") or 0) * (1 if s.get("last_frame_at") else 0),
                              -(s.get("active_units") or 0)))
    site = cands[0]
    ids["site"] = site["id"]
    cams = req.get(f"{base}/api/sites/{site['id']}/cameras").json()
    cams = [c for c in cams if c.get("frames_total")] or cams
    if cams:
        cam = cams[0]
        ids["camera"] = cam["id"]
        frames = req.get(f"{base}/api/cameras/{cam['id']}/frames?limit=500").json()
        with_boxes = [f for f in frames if f.get("detections_count")]
        best = max(with_boxes, key=lambda f: f["detections_count"]) if with_boxes else (frames[0] if frames else None)
        if best:
            ids["frame"] = best["id"]
            ids["sample"] = best.get("full_url") or best.get("url")
        night = [f for f in frames if f.get("is_night") or (f.get("weather") not in (None, "clear", "unknown"))]
        if night:
            ids["night"] = night[0]["id"]
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--themes", default="dark,light")
    ap.add_argument("--widths", default="1440,390")
    ap.add_argument("--login", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--site", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--empty-base", default="", help="второй бэкенд с пустой базой — для пустого обзора")
    ap.add_argument("--seed-test", action="store_true",
                    help="на пустом бэкенде нажать «Создать демо-объект» и снять результат (один раз)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.base.rstrip("/")
    problems: list[str] = []
    api_calls: dict[str, set] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--lang=ru-RU"])  # даты в <input type=date> — по-русски
        # Входим один раз и переносим cookie во все контексты.
        boot = browser.new_context()
        r = boot.request.post(f"{base}/api/login", data=json.dumps({"login": args.login, "password": args.password}),
                              headers={"Content-Type": "application/json"})
        if not r.ok:
            print("вход не удался:", r.status, r.text()[:200])
            return 2
        state = boot.storage_state()
        ids = pick_ids(boot.request, base, args.site)
        sample_path = None
        if ids["sample"]:
            img = boot.request.get(base + ids["sample"])
            if img.ok:
                sample_path = out / "_sample.jpg"
                sample_path.write_bytes(img.body())
        boot.close()
        print("идентификаторы:", ids)

        S, C, F = ids["site"], ids["camera"], ids["frame"]
        shots = [("overview", "/", None)]
        if args.empty_base:
            shots.append(("overview-empty", args.empty_base.rstrip("/") + "/", "empty"))
        if S:
            shots += [(f"site-{t}", f"/sites/{S}#{t}", None) for t in ("summary", "equipment", "deviations", "cameras", "plan")]
            shots += [("modal", f"/sites/{S}#deviations", "modal")]
        if C:
            shots += [("camera", f"/cameras/{C}", None), ("camera-upload", f"/cameras/{C}#upload", None),
                      ("camera-calib", f"/cameras/{C}#calib", None), ("camera-zones", f"/cameras/{C}#zones", None)]
            if ids["night"]:
                shots.append(("camera-night", f"/cameras/{C}?frame={ids['night']}", None))
        if F:
            shots.append(("frame", f"/frames/{F}", None))
        shots += [("try", "/try", None)]
        if sample_path:
            shots.append(("try-result", "/try", "try"))
        shots += [("settings", "/settings", None), ("login", "/login", "login"), ("login-error", "/login", "login-error"),
                  ("mode-warning", "/", "mode")]

        for theme in args.themes.split(","):
            for width in map(int, args.widths.split(",")):
                mobile = width < 500
                for name, path, action in shots:
                    if args.only and args.only not in name:
                        continue
                    ctx = browser.new_context(
                        viewport={"width": width, "height": 844 if mobile else 900},
                        device_scale_factor=2 if mobile else 1, locale="ru-RU", timezone_id="Europe/Moscow",
                        storage_state=None if action in ("login", "login-error", "empty") else state)
                    ctx.add_init_script(f"try {{ localStorage.setItem('sv-theme', '{theme}') }} catch (e) {{}}")
                    page = ctx.new_page()
                    errors: list[str] = []
                    page.on("console", lambda m, e=errors: m.type == "error" and e.append(f"console: {m.text}"))
                    page.on("pageerror", lambda exc, e=errors: e.append(f"pageerror: {exc}"))
                    page.on("requestfailed", lambda rq, e=errors: e.append(f"requestfailed: {rq.url} {rq.failure}"))

                    def on_response(resp, e=errors, n=name):
                        u = resp.url
                        if "/api/" in u or "/media/" in u:
                            api_calls.setdefault(n, set()).add(resp.request.method + " " + u.split("?")[0].replace(base, ""))
                            if resp.status >= 400 and not any(x in u for x in EXPECTED_4XX):
                                e.append(f"HTTP {resp.status}: {resp.request.method} {u}")
                    page.on("response", on_response)

                    if action == "empty":
                        eb = args.empty_base.rstrip("/")
                        page.request.post(f"{eb}/api/login", data=json.dumps({"login": args.login, "password": args.password}),
                                          headers={"Content-Type": "application/json"})
                        url = path
                    else:
                        url = base + path
                    try:
                        page.goto(url, wait_until="networkidle", timeout=30000)
                    except Exception as exc:  # noqa: BLE001 — долгий опрос очереди не даёт networkidle
                        errors.append(f"goto: {exc.__class__.__name__}")
                    page.wait_for_timeout(700)
                    full = True
                    try:
                        if action == "try":
                            page.set_input_files("input[type=file]", str(sample_path))
                            page.wait_for_selector("[data-result]", timeout=60000)
                        elif action == "modal":
                            if page.locator("[data-evidence]").count():
                                page.locator("[data-evidence]").first.click()
                                page.wait_for_timeout(1200)
                                full = False
                        elif action == "login-error":
                            # неверный пароль → 401 и та же форма с ошибкой (ожидаемый ответ, не ошибка UI)
                            page.fill("#login", "admin")
                            page.fill("#password", "неверный")
                            with page.expect_navigation():
                                page.click("button[type=submit]")
                            errors[:] = [e for e in errors if "401" not in e]
                        elif action == "mode":
                            page.click("header .seg button:nth-of-type(2)")
                            page.wait_for_timeout(400)
                            full = False
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"action {action}: {exc}")
                    page.wait_for_timeout(900)
                    fn = out / f"{name}-{theme}-{width}.png"
                    page.screenshot(path=str(fn), full_page=full)
                    if action == "empty" and args.seed_test:
                        # Сценарий жюри: пустая система → «Создать демо-объект» → объект с анализом.
                        args.seed_test = False
                        try:
                            page.get_by_role("button", name="Создать демо-объект").click()
                            page.wait_for_url("**/sites/**", timeout=90000)
                            page.wait_for_timeout(4000)
                            page.screenshot(path=str(out / f"seed-start-{theme}-{width}.png"), full_page=True)
                            page.wait_for_timeout(60000)
                            page.reload(wait_until="networkidle")
                            page.wait_for_timeout(1500)
                            page.screenshot(path=str(out / f"seed-done-{theme}-{width}.png"), full_page=True)
                        except Exception as exc:  # noqa: BLE001
                            errors.append(f"seed-test: {exc}")
                    if errors:
                        problems.append(name)
                        print(f"[{name} {theme} {width}]", *sorted(set(errors)), sep="\n  ")
                    ctx.close()
        browser.close()

    (out / "_api_calls.json").write_text(json.dumps({k: sorted(v) for k, v in api_calls.items()}, ensure_ascii=False, indent=1))
    print("готово, снимков с проблемами:", len(problems), sorted(set(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
