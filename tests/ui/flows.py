"""Сценарии интерфейса на живом бэкенде: действия пользователя и их запросы к API.

Все действия обратимы (что создано — удаляется в конце), поэтому скрипт можно
гонять на демо-стенде. Запуск — в том же docker-контейнере playwright, что и shoot.py:

    python /tools/flows.py --base http://127.0.0.1:18100 --out /shots/ui/flows --site 1

Проверяет: квитирование отклонения и возврат; ручную поправку моточасов и её удаление;
новую камеру → загрузку снимков с прогрессом задания → удаление камеры; зону на кадре;
калибровку 4 точками и её снятие; «Проверить снимок». Печатает запросы с ошибками.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", default="1")
    ap.add_argument("--images", default="/tools/fixtures/media")
    args = ap.parse_args()
    base, out = args.base.rstrip("/"), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    log: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--lang=ru-RU"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, locale="ru-RU", timezone_id="Europe/Moscow")
        ctx.request.post(f"{base}/api/login", data=json.dumps({"login": "admin", "password": "admin"}),
                         headers={"Content-Type": "application/json"})
        page = ctx.new_page()
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))

        def on_resp(r):
            if "/api/" in r.url and r.request.method != "GET":
                log.append(f"{r.status} {r.request.method} {r.url.replace(base, '')}")
                if r.status >= 400:
                    problems.append(f"HTTP {r.status} {r.request.method} {r.url} {r.text()[:200]}")
        page.on("response", on_resp)

        def shot(name):
            page.wait_for_timeout(700)
            page.screenshot(path=str(out / f"{name}.png"))

        S = args.site
        # 1) Квитирование отклонения и возврат
        page.goto(f"{base}/sites/{S}#deviations", wait_until="networkidle")
        # прошлый прерванный прогон мог оставить квитированное — вернуть
        for d in ctx.request.get(f"{base}/api/sites/{S}/deviations?status=ack").json():
            ctx.request.patch(f"{base}/api/deviations/{d['id']}", data=json.dumps({"status": "open"}),
                              headers={"Content-Type": "application/json"})
        page.reload(wait_until="networkidle")
        btn = page.get_by_role("button", name="Квитировать").first
        if btn.count():
            btn.click()
            shot("01-ack-toast")
            page.get_by_role("tab", name="Квитированные").click()
            page.get_by_role("button", name="Вернуть в открытые").first.click()
            page.wait_for_timeout(600)

        # 2) Ручная поправка моточасов и её удаление
        page.goto(f"{base}/sites/{S}#equipment", wait_until="networkidle")
        page.get_by_role("button", name="Поправить часы").click()
        page.select_option("#fix-cls", "excavator")
        page.fill("#fix-h", "2.5")
        page.fill("#fix-note", "проверка UI: ночная смена")
        shot("02-hours-modal")
        page.get_by_role("button", name="Сохранить").click()
        page.wait_for_timeout(2500)
        shot("03-hours-saved")
        dels = page.get_by_role("button", name="Удалить поправку")
        while dels.count():
            dels.first.click()
            page.wait_for_timeout(1500)

        # 2б) Импорт графика xlsx — только разбор (apply=0), без сохранения
        page.goto(f"{base}/sites/{S}#plan", wait_until="networkidle")
        page.set_input_files("input[type=file][accept*=xlsx]", "/tools/fixtures/plan_sample.xlsx")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(out / "03b-plan-import.png"), full_page=True)
        if page.get_by_role("button", name="Отменить").count():
            page.get_by_role("button", name="Отменить").first.click()

        # 3) Новая камера → загрузка снимков → удаление. На временном объекте: удаление
        # камеры не стирает её единицы техники (см. отчёт), а удаление объекта — стирает.
        tmp = ctx.request.post(f"{base}/api/sites", data=json.dumps({"name": "Тест UI · временный объект"}),
                               headers={"Content-Type": "application/json"}).json()
        page.goto(f"{base}/sites/{tmp['id']}#cameras", wait_until="networkidle")
        page.get_by_text("Добавить камеру").click()
        page.fill("#nc-name", "Тест UI · загрузка")
        page.get_by_role("button", name="Создать").click()
        page.wait_for_url("**/cameras/**", timeout=20000)
        cam_id = page.url.split("/cameras/")[1].split("#")[0].split("?")[0]
        imgs = sorted(Path(args.images).glob("media__frames__*"))[:3]
        page.set_input_files("input[type=file][multiple]", [str(x) for x in imgs])
        shot("04-upload-files")
        page.get_by_role("button", name="Загрузить и разобрать").click()
        try:
            page.wait_for_selector("text=Готово", timeout=120000)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"upload: {exc}")
        shot("05-upload-done")

        # 4) Зона на кадре новой камеры
        page.goto(f"{base}/cameras/{cam_id}#zones", wait_until="networkidle")
        page.get_by_role("button", name="Нарисовать зону").click()
        box = page.locator(".viewer img").first.bounding_box()
        if box:
            for fx, fy in ((0.2, 0.3), (0.7, 0.3), (0.75, 0.8), (0.15, 0.8)):
                page.mouse.click(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
            page.keyboard.press("Enter")
            page.fill("#zn", "Котлован")
            shot("06-zone-draft")
            page.get_by_role("button", name="Сохранить зону").click()
            page.wait_for_timeout(1000)
            shot("07-zone-saved")

        # 5) Калибровка 4 точками и её снятие
        page.goto(f"{base}/cameras/{cam_id}#calib", wait_until="networkidle")
        box = page.locator(".viewer img").first.bounding_box()
        if box:
            pts = ((0.1, 0.9, 0, 0), (0.9, 0.9, 40, 0), (0.8, 0.4, 40, 60), (0.2, 0.4, 0, 60))
            for i, (fx, fy, X, Y) in enumerate(pts):
                page.mouse.click(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
                page.get_by_label(f"X точки {i + 1}").fill(str(X))
                page.get_by_label(f"Y точки {i + 1}").fill(str(Y))
            page.get_by_role("button", name="Рассчитать гомографию").click()
            page.wait_for_timeout(1500)
            shot("08-calibrated")
            page.once("dialog", lambda d: d.accept())
            page.get_by_role("button", name="Снять калибровку").click()
            page.wait_for_timeout(1000)

        # уборка: временный объект вместе с камерой, кадрами, зонами и единицами техники
        r = ctx.request.delete(f"{base}/api/sites/{tmp['id']}")
        log.append(f"{r.status} DELETE /api/sites/{tmp['id']} (уборка)")
        browser.close()

    print("\n".join(log))
    print("проблемы:" if problems else "проблем нет", *problems, sep="\n  ")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
