"""Веб-сервис: API и серверный рендер страниц в одном приложении.

Отдельного фронтенда со сборкой нет намеренно — разработка соло, и React+Vite
означал бы второй сервис, второй деплой и npm в докере ради страниц, которые
прекрасно рендерятся на сервере. Интерактивность там, где она нужна, делается
HTMX без единой строки сборки.
"""

from __future__ import annotations

import datetime as dt
import json
import secrets
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from app import auth, netutil
from app.config import settings
from app.db import get_session, init_db
from app.models import (Camera, CameraState, Deviation, Frame, MacroStage,
                        ObjectType, Site, SiteStage, StageTemplate)
from app.pipeline import aggregate, gantt, ingest, live, runner
from app.pipeline.model_b import ModelB

app = FastAPI(title="Мониторинг строительных площадок", docs_url="/api/docs")
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)
templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
def startup() -> None:
    init_db()


try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except RuntimeError:
    pass  # каталога может не быть на раннем этапе


# ---------------------------------------------------------------------------
# служебное
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz(s: Session = Depends(get_session)) -> JSONResponse:
    """Проверка живости всех внешних зависимостей разом.

    Нужна не для галочки: на защите важно за секунду понять, что именно
    отвалилось — база, хранилище или модель Б.
    """
    checks = {"db": False, "storage": False, "model_b": False}
    try:
        s.execute(select(1))
        checks["db"] = True
    except Exception:
        pass
    try:
        from app.storage import storage
        storage.exists("__healthz__")
        checks["storage"] = True
    except Exception:
        pass
    checks["model_b"] = ModelB().health()

    code = 200 if all(checks.values()) else 503
    return JSONResponse({"ok": all(checks.values()), "checks": checks,
                         "model": settings.vlm_model}, status_code=code)


@app.get("/media/{key:path}")
def media(key: str) -> Response:
    """Отдача кадров при локальном бэкенде хранилища. При S3 браузер ходит
    в MinIO напрямую по presigned-ссылке и сюда не попадает."""
    from app.storage import storage
    try:
        return Response(storage.get(key), media_type="image/jpeg")
    except Exception:
        return Response(status_code=404)


# ---------------------------------------------------------------------------
# авторизация
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
def login_submit(request: Request, login: str = Form(...), password: str = Form(...)):
    if not auth.check_credentials(login, password):
        return templates.TemplateResponse(
            request, "login.html", {"error": "Неверный логин или пароль"},
            status_code=401)
    auth.login_user(request, login)
    return auth.redirect("/")


@app.get("/logout")
def logout(request: Request):
    auth.logout_user(request)
    return auth.redirect("/login")


# ---------------------------------------------------------------------------
# страницы
# ---------------------------------------------------------------------------

# Статусы объекта. Берутся из открытых отклонений, а не из плана: план
# говорит только о намерениях, а отставание — это наблюдение. Пока разбор
# не прошёл, объект с планом честно числится «наблюдения нет».
BEHIND_TYPES = {"STAGE_BEHIND", "STAGE_NOT_STARTED", "TEMPO_DECAY",
                "SITE_IDLE", "ACTIVITY_DROP"}
AHEAD_TYPES = {"STAGE_AHEAD"}


def _site_cards(s: Session, q: str = "") -> tuple[list[dict], int]:
    """Карточки объектов со сводкой. Общее число — до фильтрации поиском."""
    sites = s.scalars(select(Site).order_by(Site.name)).all()
    total = len(sites)
    if q:
        needle = q.strip().lower()
        sites = [x for x in sites
                 if needle in x.name.lower() or needle in (x.address or "").lower()]

    open_dev = dict(s.execute(
        select(Deviation.site_id, func.count())
        .where(Deviation.resolved_at.is_(None))
        .group_by(Deviation.site_id)).all())
    frames_by_site = dict(s.execute(
        select(Camera.site_id, func.count(Frame.id))
        .join(Frame, Frame.camera_id == Camera.id)
        .group_by(Camera.site_id)).all())

    today = dt.date.today()
    cards = []
    for site in sites:
        active = [st for st in site.stages
                  if st.enabled and st.planned_start and st.planned_end
                  and st.planned_start <= today <= st.planned_end]
        cards.append({
            "site": site,
            "active": active,
            "deviations": open_dev.get(site.id, 0),
            "cameras": len(site.cameras),
            "frames": frames_by_site.get(site.id, 0),
        })
    return cards, total


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, user: str = Depends(auth.require_user),
              s: Session = Depends(get_session)):
    """Сводка по всем объектам сразу — то, с чего начинает работу куратор.

    Показывается только то, что действительно посчитано. Объект, по которому
    разбор ещё не прошёл, числится в «наблюдения нет», а не в «по графику»:
    иначе дашборд рисовал бы благополучие там, где просто нет данных.
    """
    today = dt.date.today()
    sites = s.scalars(select(Site).order_by(Site.name)).all()
    cameras = s.scalars(select(Camera)).all()

    frames_total = s.scalar(select(func.count()).select_from(Frame)) or 0
    frames_ok = s.scalar(select(func.count()).select_from(Frame)
                         .where(Frame.quality_ok.is_(True))) or 0

    open_devs = s.scalars(
        select(Deviation).where(Deviation.resolved_at.is_(None))
        .order_by(Deviation.detected_at.desc())).all()

    by_site: dict[int, list] = {}
    for d in open_devs:
        by_site.setdefault(d.site_id, []).append(d)

    # Распределение объектов по состоянию графика.
    status = {"behind": 0, "ahead": 0, "on_track": 0, "no_data": 0, "no_plan": 0}
    for site in sites:
        kinds = {d.type.value for d in by_site.get(site.id, [])}
        has_plan = any(st.enabled and st.planned_start and st.planned_end
                       for st in site.stages)
        observed = any(cam.state and cam.state.windows_accumulated
                       for cam in site.cameras)
        if not has_plan:
            status["no_plan"] += 1
        elif kinds & BEHIND_TYPES:
            status["behind"] += 1
        elif kinds & AHEAD_TYPES:
            status["ahead"] += 1
        elif observed:
            status["on_track"] += 1
        else:
            status["no_data"] += 1

    # Какие этапы идут прямо сейчас по плану — в разрезе всех объектов.
    stage_load: dict[str, int] = {}
    for site in sites:
        for st in site.stages:
            if (st.enabled and st.planned_start and st.planned_end
                    and st.planned_start <= today <= st.planned_end):
                stage_load[st.title] = stage_load.get(st.title, 0) + 1
    stage_load = dict(sorted(stage_load.items(), key=lambda kv: -kv[1]))

    dev_kinds: dict[str, int] = {}
    for d in open_devs:
        dev_kinds[d.type.value] = dev_kinds.get(d.type.value, 0) + 1
    dev_kinds = dict(sorted(dev_kinds.items(), key=lambda kv: -kv[1]))

    # Что мешает системе работать. Камера без маски не разбирается вовсе,
    # камера с исчерпанной маской отдаёт модели полный кадр — это не поломка,
    # но знать об этом надо.
    attention = []
    for cam in cameras:
        st = cam.state
        site = s.get(Site, cam.site_id)
        if st is None or not st.mask_approved:
            attention.append({"camera": cam, "site": site,
                              "what": "маска не задана",
                              "hint": "разбор кадров невозможен"})
        elif st.masked_ratio <= 0.03 and st.windows_accumulated:
            attention.append({"camera": cam, "site": site,
                              "what": "маска исчерпана",
                              "hint": f"скрыто {st.masked_ratio * 100:.0f}%, "
                                      "модель получает полный кадр"})
    for site in sites:
        if not site.cameras:
            attention.append({"camera": None, "site": site,
                              "what": "нет камер",
                              "hint": "наблюдать объект нечем"})

    return templates.TemplateResponse(request, "dashboard.html", {
        "user": user, "today": today,
        "sites_total": len(sites),
        "cameras_total": len(cameras),
        "cameras_ready": sum(1 for c in cameras if c.state and c.state.mask_approved),
        "frames_total": frames_total, "frames_ok": frames_ok,
        "status": status, "stage_load": stage_load,
        "dev_kinds": dev_kinds, "dev_total": len(open_devs),
        "recent": open_devs[:8],
        "site_of": {x.id: x for x in sites},
        "attention": attention[:8],
        "attention_total": len(attention),
    })


@app.get("/sites", response_class=HTMLResponse)
def sites_list(request: Request, q: str = "",
               user: str = Depends(auth.require_user),
               s: Session = Depends(get_session)):
    cards, total = _site_cards(s, q)
    return templates.TemplateResponse(request, "sites.html",
                                      {"cards": cards, "total": total,
                                       "q": q, "user": user})


@app.get("/sites/new", response_class=HTMLResponse)
def site_new(request: Request, user: str = Depends(auth.require_user),
             s: Session = Depends(get_session)):
    return templates.TemplateResponse(request, "site_new.html", {
        "user": user,
        "object_types": s.scalars(select(ObjectType).order_by(ObjectType.id)).all(),
        "macro_stages": s.scalars(select(MacroStage)
                                  .order_by(MacroStage.order_default)).all(),
    })


@app.post("/sites")
def site_create(request: Request, user: str = Depends(auth.require_user),
                s: Session = Depends(get_session),
                name: str = Form(...), address: str = Form(""),
                object_type_id: int = Form(...)):
    site = Site(name=name, address=address, object_type_id=object_type_id)
    s.add(site)
    s.flush()

    # Предзаполняем этапы теми, что отмечены галочкой в справочнике для этого
    # типа объекта. Даты пользователь проставляет сам — организаторы сказали,
    # что привязка этапов к датам на нашей стороне.
    otype = s.get(ObjectType, object_type_id)
    stages = s.scalars(select(MacroStage).order_by(MacroStage.order_default)).all()
    for idx, ms in enumerate(stages):
        tpl = s.scalar(select(StageTemplate)
                       .where(StageTemplate.macro_stage_id == ms.id))
        s.add(SiteStage(
            site_id=site.id, macro_stage_id=ms.id, order_idx=idx,
            equipment_expected=tpl.equipment_expected if tpl else [],
            equipment_forbidden=tpl.equipment_forbidden if tpl else [],
        ))
    s.commit()
    return auth.redirect(f"/sites/{site.id}")


def _chart(s: Session, site, today: dt.date) -> gantt.Chart | None:
    """План против факта: диаграмма Ганта на общей шкале.

    Факт строится по каждой камере отдельно. Две камеры смотрят на площадку
    с разных сторон, и то, что одна видит этап, а другая нет, — сведение,
    которое нельзя терять усреднением: чаще всего это значит, что работы
    идут с той стороны, а не что модель ошиблась.
    """
    per_cam = aggregate.curves_by_camera(s, site)
    names = {c.id: c.name for c in site.cameras}

    sources = []
    for i, cam_id in enumerate(sorted(per_cam)):
        curves = per_cam[cam_id]
        sources.append(gantt.Source(
            key=f"cam{cam_id}",
            label=names.get(cam_id, f"камера {cam_id}"),
            color=gantt.PALETTE[i % len(gantt.PALETTE)],
            intervals={c.stage_id: c.intervals for c in curves},
            reached={c.stage_id for c in curves if c.reached},
        ))

    stages = [{"id": st.id, "title": st.title,
               "planned_start": st.planned_start, "planned_end": st.planned_end}
              for st in site.stages]
    return gantt.build(stages, sources, today)


@app.get("/sites/{site_id}", response_class=HTMLResponse)
def site_detail(site_id: int, request: Request,
                user: str = Depends(auth.require_user),
                s: Session = Depends(get_session)):
    site = s.get(Site, site_id)
    if site is None:
        return HTMLResponse("Объект не найден", status_code=404)
    deviations = s.scalars(
        select(Deviation).where(Deviation.site_id == site_id)
        .order_by(Deviation.detected_at.desc()).limit(50)).all()
    used = {st.macro_stage_id for st in site.stages}
    catalog = s.scalars(select(MacroStage)
                        .order_by(MacroStage.order_default)).all()
    today = dt.date.today()
    return templates.TemplateResponse(request, "site.html", {
        "user": user, "site": site, "deviations": deviations,
        "today": today,
        "catalog": catalog,
        "available": [m for m in catalog if m.id not in used],
        "chart": _chart(s, site, today),
        "live": {c.id: live.status(c.id) for c in site.cameras},
        # Диаграмма подтягивает себя сама, только пока есть чему меняться.
        "poll": any(c.source_type == "stream" for c in site.cameras),
    })


@app.get("/sites/{site_id}/chart", response_class=HTMLResponse)
def site_chart(site_id: int, request: Request,
               user: str = Depends(auth.require_user),
               s: Session = Depends(get_session)):
    """Одна диаграмма, без остальной страницы — для опроса из HTMX."""
    site = s.get(Site, site_id)
    if site is None:
        return HTMLResponse("")
    today = dt.date.today()
    return templates.TemplateResponse(request, "_gantt.html", {
        "site": site, "chart": _chart(s, site, today), "today": today,
        "poll": any(c.source_type == "stream" for c in site.cameras),
    })


@app.post("/sites/{site_id}/stages")
async def stages_save(site_id: int, request: Request,
                      user: str = Depends(auth.require_user),
                      s: Session = Depends(get_session)):
    """Сохранение календарного плана: состав, порядок и даты разом.

    Форма разбирается вручную, а не через Form(...): число строк заранее
    неизвестно, пользователь добавляет и удаляет этапы прямо на странице.

    Порядок берётся из порядка полей `row` в теле запроса — браузер шлёт их
    в том порядке, в каком они лежат в разметке, а перетаскивание строки
    двигает саму разметку. Отдельный номер позиции хранить не нужно.

    Ключ строки: `s<id>` для уже существующего этапа, `m<id>` для только что
    добавленного из справочника. Различать обязательно — у нового этапа ещё
    нет строки в базе, а у существующего нельзя терять привязанные данные.
    """
    form = await request.form()
    keys = form.getlist("row")

    existing = {st.id: st for st in s.scalars(
        select(SiteStage).where(SiteStage.site_id == site_id)).all()}
    kept: set[int] = set()

    for idx, key in enumerate(keys):
        kind, _, raw = str(key).partition(":")
        if not raw.isdigit():
            continue
        ident = int(raw)

        if kind == "s":
            stage = existing.get(ident)
            if stage is None:
                continue
            kept.add(stage.id)
        elif kind == "m":
            stage = SiteStage(site_id=site_id, macro_stage_id=ident)
            tpl = s.scalar(select(StageTemplate)
                           .where(StageTemplate.macro_stage_id == ident))
            stage.equipment_expected = tpl.equipment_expected if tpl else []
            stage.equipment_forbidden = tpl.equipment_forbidden if tpl else []
            # Снимок вопросов берётся один раз, при добавлении этапа.
            # Дальше правка справочника на этот объект не влияет — иначе
            # накопленная история окажется посчитанной по разным чек-листам.
            stage.questions = tpl.questions if tpl else []
            s.add(stage)
        else:
            continue

        # Этап есть в плане — значит он включён. Отдельной галочки больше нет:
        # лишний этап теперь убирают из списка, а не снимают с него отметку.
        stage.enabled = True
        stage.order_idx = idx
        start = form.get(f"start_{key}") or ""
        end = form.get(f"end_{key}") or ""
        stage.planned_start = dt.date.fromisoformat(start) if start else None
        stage.planned_end = dt.date.fromisoformat(end) if end else None
        stage.dates_confirmed = bool(stage.planned_start and stage.planned_end)

    # Чего в форме не пришло — пользователь удалил со страницы.
    for stage_id, stage in existing.items():
        if stage_id not in kept:
            s.delete(stage)

    s.commit()
    return auth.redirect(f"/sites/{site_id}")


@app.post("/sites/{site_id}/cameras")
def camera_add(site_id: int, user: str = Depends(auth.require_user),
               s: Session = Depends(get_session),
               name: str = Form(...), mode: str = Form("stream"),
               source_uri: str = Form("")):
    """Заведение камеры. Два режима — поток и папка, см. `Camera`.

    Ключ приёма выдаётся сразу и живёт с камерой: он нужен уже при первом
    подключении, а придумывать отдельный шаг «сгенерировать ключ» значило бы
    добавить оператору действие, которое никогда не делается иначе.
    """
    cam = Camera(site_id=site_id, name=name,
                 source_type="folder" if mode == "folder" else "stream",
                 source_uri=source_uri.strip(),
                 ingest_key=secrets.token_urlsafe(24))
    s.add(cam)
    s.flush()
    s.add(CameraState(camera_id=cam.id))
    s.commit()
    # Сразу на карточку камеры: там рисуется маска, без неё разбор невозможен.
    return auth.redirect(f"/cameras/{cam.id}")


@app.post("/cameras/{camera_id}/delete")
def camera_delete(camera_id: int, user: str = Depends(auth.require_user),
                  s: Session = Depends(get_session)):
    """Удаление камеры вместе с накопленным состоянием и кадрами.

    Кадры уходят по внешнему ключу на стороне БД. Картинки остаются в
    хранилище: они адресуются ключом с номером камеры, новый номер их не
    переиспользует, а гонять тысячу удалений ради освобождения места,
    которое ничего не стоит, — плохой размен.
    """
    cam = s.get(Camera, camera_id)
    site_id = cam.site_id if cam else None
    if cam:
        s.delete(cam)
        s.commit()
    return auth.redirect(f"/sites/{site_id}" if site_id else "/sites")


@app.post("/sites/{site_id}/delete")
def site_delete(site_id: int, user: str = Depends(auth.require_user),
                s: Session = Depends(get_session)):
    site = s.get(Site, site_id)
    if site:
        s.delete(site)
        s.commit()
    return auth.redirect("/sites")


# ---------------------------------------------------------------------------
# камера: первый кадр, маска, прогон, результаты
# ---------------------------------------------------------------------------

def _remote(cam: Camera, path: str, method: str = "get", **kw):
    """Запрос к сервису камеры. Возвращает (ответ, текст ошибки)."""
    import requests

    if not cam.source_uri:
        return None, "у камеры не задан адрес"
    url = cam.source_uri.rstrip("/") + path
    try:
        resp = getattr(requests, method)(url, timeout=5,
                                         proxies=netutil.proxies_for(url), **kw)
    except requests.RequestException as exc:
        return None, f"камера не отвечает ({url}): {exc.__class__.__name__}"
    if resp.status_code >= 400:
        return None, f"камера ответила {resp.status_code}: {resp.text[:200]}"
    return resp, ""


def _stream_reference(s: Session, cam: Camera) -> str:
    """Опорный кадр для рисования маски — до того, как пошла съёмка.

    Маску рисуют один раз и заранее: без неё принятый кадр разбирать нечем.
    Поэтому камеру спрашивают о первом кадре отдельно, не дожидаясь, пока
    она начнёт слать серию.
    """
    from app.storage import storage

    resp, err = _remote(cam, "/api/preview")
    if resp is None:
        return err
    cam.reference_frame_key = storage.put(f"cam/{cam.id}/reference.jpg",
                                          resp.content)
    s.commit()
    return ""


@app.get("/cameras/{camera_id}", response_class=HTMLResponse)
def camera_detail(camera_id: int, request: Request, err: str = "",
                  user: str = Depends(auth.require_user),
                  s: Session = Depends(get_session)):
    from app.storage import storage

    cam = s.get(Camera, camera_id)
    if cam is None:
        return HTMLResponse("Камера не найдена", status_code=404)
    if cam.state is None:
        cam.state = CameraState(camera_id=cam.id)
        s.flush()
        s.commit()
    if not cam.ingest_key:
        cam.ingest_key = secrets.token_urlsafe(24)
        s.commit()

    first_url, first_err, total, remote = None, "", 0, None

    if cam.source_type == "stream":
        resp, first_err = _remote(cam, "/api/info")
        if resp is not None:
            remote = resp.json()
            total = remote.get("frames", 0)
        if not cam.reference_frame_key and resp is not None:
            first_err = _stream_reference(s, cam)
    else:
        # Первый кадр берётся из папки-источника: маску надо рисовать до прогона.
        folder = Path(cam.source_uri) if cam.source_uri else None
        if folder and folder.is_dir():
            items = ingest.list_frames(folder)
            total = len(items)
            if items:
                if not cam.reference_frame_key:
                    import cv2
                    img = cv2.imread(str(items[0][0]))
                    if img is not None:
                        ok, buf = cv2.imencode(".jpg", img,
                                               [cv2.IMWRITE_JPEG_QUALITY, 92])
                        cam.reference_frame_key = storage.put(
                            f"cam/{cam.id}/reference.jpg", buf.tobytes())
                        s.commit()
            else:
                first_err = "в папке нет кадров с распознаваемой меткой времени"
        elif folder:
            first_err = f"папка не найдена: {folder}"
        else:
            first_err = "у камеры не задана папка с кадрами"

    if cam.reference_frame_key:
        first_url = storage.url(cam.reference_frame_key)

    # Состояние прогона показывается, пока он идёт, и если он упал. Успешно
    # завершённый прятать обязательно: иначе страница навсегда застрянет на
    # «готово» и кнопка пересчёта больше не появится.
    running = runner.status(cam.id)
    run = running if running and (not running.finished or running.error) else None

    initial_url = (storage.url(cam.state.initial_mask_key)
                   if cam.state.initial_mask_key else None)
    frames_count = s.scalar(
        select(func.count()).select_from(Frame).where(Frame.camera_id == cam.id)) or 0

    return templates.TemplateResponse(request, "camera.html", {
        "user": user, "cam": cam, "state": cam.state,
        "first_url": first_url, "first_err": first_err, "err": err,
        "source_total": total, "initial_url": initial_url,
        "frames_count": frames_count, "remote": remote,
        "ingest_url": settings.public_base_url.rstrip("/") + "/api/ingest",
        "live": live.status(cam.id),
        "camera_id": cam.id, "run": run,
    })


@app.post("/cameras/{camera_id}/mask")
def camera_mask_save(camera_id: int, user: str = Depends(auth.require_user),
                     s: Session = Depends(get_session),
                     mask_png: str = Form(...)):
    """Принимает маску, нарисованную оператором, и делает её начальной."""
    import base64
    import cv2
    import numpy as np

    from app.pipeline import mask as M
    from app.storage import storage

    cam = s.get(Camera, camera_id)
    if cam is None or cam.state is None:
        return HTMLResponse("Камера не найдена", status_code=404)

    header, _, b64 = mask_png.partition(",")
    raw = base64.b64decode(b64)
    bitmap = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if bitmap is None:
        return HTMLResponse("Не удалось разобрать маску", status_code=400)

    ref = cv2.imdecode(
        np.frombuffer(storage.get(cam.reference_frame_key), np.uint8),
        cv2.IMREAD_COLOR)
    shape = ingest.work_shape(ref)

    st = M.init_from_bitmap(bitmap, shape)
    ingest.save_state(s, cam.state, st, initial=True)
    cam.state.mask_approved = True
    s.commit()

    # Маска подтверждена — прогонять историю больше незачем ждать команды
    # в терминале. Страница камеры сразу покажет прогресс и уведёт к кадрам.
    # Потоковую камеру это не касается: там разбор начнётся сам, как только
    # пойдут кадры, а прогонять нечего — истории ещё нет.
    if cam.source_type == "folder" and cam.source_uri and Path(cam.source_uri).is_dir():
        runner.start(camera_id)
    return auth.redirect(f"/cameras/{camera_id}")


@app.post("/cameras/{camera_id}/reset-mask")
def camera_mask_reset(camera_id: int, user: str = Depends(auth.require_user),
                      s: Session = Depends(get_session)):
    cam = s.get(Camera, camera_id)
    if cam and cam.state:
        cam.state.mask_approved = False
        cam.state.background_key = ""
        cam.state.initial_mask_key = ""
        cam.state.evidence_key = ""
        cam.state.windows_accumulated = 0
        s.commit()
    return auth.redirect(f"/cameras/{camera_id}")


@app.get("/cameras/{camera_id}/frames", response_class=HTMLResponse)
def camera_frames(camera_id: int, request: Request,
                  user: str = Depends(auth.require_user),
                  s: Session = Depends(get_session),
                  view: str = "masked", page: int = 1, per: int = 60):
    """Галерея результатов прогона — то, ради чего всё и затевалось."""
    from app.storage import storage

    cam = s.get(Camera, camera_id)
    if cam is None:
        return HTMLResponse("Камера не найдена", status_code=404)

    total = s.scalar(select(func.count()).select_from(Frame)
                     .where(Frame.camera_id == camera_id)) or 0
    pages = max(1, (total + per - 1) // per)
    page = max(1, min(page, pages))

    rows = s.scalars(
        select(Frame).where(Frame.camera_id == camera_id)
        .order_by(Frame.captured_at).offset((page - 1) * per).limit(per)).all()

    key_of = {"masked": "masked_key", "overlay": "overlay_key",
              "orig": "object_key"}.get(view, "masked_key")
    items = [{
        "frame": f,
        "url": storage.url(getattr(f, key_of) or f.object_key),
    } for f in rows]

    return templates.TemplateResponse(request, "frames.html", {
        "user": user, "cam": cam, "items": items, "view": view,
        "page": page, "pages": pages, "total": total, "per": per,
    })


@app.get("/frames/{frame_id}", response_class=HTMLResponse)
def frame_detail(frame_id: int, request: Request,
                 user: str = Depends(auth.require_user),
                 s: Session = Depends(get_session)):
    from app.storage import storage

    f = s.get(Frame, frame_id)
    if f is None:
        return HTMLResponse("Кадр не найден", status_code=404)
    cam = s.get(Camera, f.camera_id)

    prev = s.scalar(select(Frame).where(Frame.camera_id == f.camera_id,
                                        Frame.captured_at < f.captured_at)
                    .order_by(Frame.captured_at.desc()).limit(1))
    nxt = s.scalar(select(Frame).where(Frame.camera_id == f.camera_id,
                                       Frame.captured_at > f.captured_at)
                   .order_by(Frame.captured_at).limit(1))

    return templates.TemplateResponse(request, "frame.html", {
        "user": user, "cam": cam, "f": f, "prev": prev, "next": nxt,
        "orig_url": storage.url(f.object_key),
        "masked_url": storage.url(f.masked_key) if f.masked_key else None,
        "overlay_url": storage.url(f.overlay_key) if f.overlay_key else None,
    })


@app.post("/cameras/{camera_id}/run")
def camera_run(camera_id: int, user: str = Depends(auth.require_user),
               s: Session = Depends(get_session),
               model_b: str = Form("")):
    cam = s.get(Camera, camera_id)
    if cam is None or cam.state is None or not cam.state.mask_approved:
        return HTMLResponse("Сначала нужно задать маску", status_code=400)
    runner.start(camera_id, model_b=model_b == "on")
    return auth.redirect(f"/cameras/{camera_id}")


@app.get("/cameras/{camera_id}/progress", response_class=HTMLResponse)
def camera_progress(camera_id: int, request: Request,
                    user: str = Depends(auth.require_user)):
    """Кусок разметки для опроса из HTMX.

    Когда прогон закончен, отдаётся заголовок HX-Redirect — браузер сам
    уходит на страницу с разобранными кадрами, ради которой всё и делалось.
    """
    st = runner.status(camera_id)
    if st is None:
        return HTMLResponse("")

    resp = templates.TemplateResponse(request, "_progress.html",
                                      {"run": st, "camera_id": camera_id})
    if st.finished and not st.error:
        resp.headers["HX-Redirect"] = f"/cameras/{camera_id}/frames"
    return resp


# ---------------------------------------------------------------------------
# камера как поток: подключение и приём кадров
# ---------------------------------------------------------------------------

@app.post("/cameras/{camera_id}/connect")
def camera_connect(camera_id: int, user: str = Depends(auth.require_user),
                   s: Session = Depends(get_session)):
    """Сказать камере, куда слать кадры, и включить съёмку.

    Адрес приёмника сообщаем мы, а не камера его угадывает: камера живёт на
    другой машине, и «localhost» у неё свой. Ключ уходит той же командой —
    камера не хранит его между запусками, и это правильно: отозвать доступ
    должно быть можно, не заходя на камеру.
    """
    cam = s.get(Camera, camera_id)
    if cam is None:
        return HTMLResponse("Камера не найдена", status_code=404)

    def back(err: str = "") -> Response:
        return auth.redirect(f"/cameras/{camera_id}"
                             + (f"?err={quote(err)}" if err else ""))

    if cam.source_type != "stream":
        return back("камера заведена как папка, подключать нечего")
    if cam.state is None or not cam.state.mask_approved:
        return back("сначала нужно нарисовать маску: без неё кадр разбирать нечем")

    # Приёмник поднимаем до команды: первый кадр может прийти через секунду.
    live.worker(camera_id)

    resp, err = _remote(cam, "/api/start", "post", json={
        "ingest_url": settings.public_base_url.rstrip("/") + "/api/ingest",
        "camera_id": cam.id,
        "api_key": cam.ingest_key,
        "restart": True,
    })
    return back(err)


@app.post("/cameras/{camera_id}/disconnect")
def camera_disconnect(camera_id: int, user: str = Depends(auth.require_user),
                      s: Session = Depends(get_session)):
    cam = s.get(Camera, camera_id)
    if cam is None:
        return HTMLResponse("Камера не найдена", status_code=404)
    _, err = _remote(cam, "/api/stop", "post")
    return auth.redirect(f"/cameras/{camera_id}"
                         + (f"?err={quote(err)}" if err else ""))


@app.post("/cameras/{camera_id}/reset-stream")
def camera_reset_stream(camera_id: int, user: str = Depends(auth.require_user),
                        s: Session = Depends(get_session)):
    """Забыть наблюдения и вернуть маску к нарисованной — для повторного показа."""
    cam = s.get(Camera, camera_id)
    if cam is None or cam.state is None:
        return HTMLResponse("Камера не найдена", status_code=404)
    _remote(cam, "/api/stop", "post")
    live.reset(s, cam)
    return auth.redirect(f"/cameras/{camera_id}")


@app.get("/cameras/{camera_id}/live", response_class=HTMLResponse)
def camera_live(camera_id: int, request: Request,
                user: str = Depends(auth.require_user),
                s: Session = Depends(get_session)):
    """Кусок разметки для опроса из HTMX: что происходит с потоком сейчас."""
    cam = s.get(Camera, camera_id)
    if cam is None:
        return HTMLResponse("")
    resp, err = _remote(cam, "/api/info")
    frames_count = s.scalar(
        select(func.count()).select_from(Frame)
        .where(Frame.camera_id == camera_id)) or 0
    return templates.TemplateResponse(request, "_live.html", {
        "cam": cam, "camera_id": camera_id, "live": live.status(camera_id),
        "remote": resp.json() if resp is not None else None,
        "remote_err": err, "frames_count": frames_count,
        "state": cam.state,
    })


@app.post("/api/ingest")
async def api_ingest(request: Request,
                     file: UploadFile = File(...),
                     camera_id: int = Form(...),
                     captured_at: str = Form(""),
                     meta: str = Form(""),
                     s: Session = Depends(get_session)) -> JSONResponse:
    """Приём одного кадра от камеры.

    Пускает ключ, а не сессия оператора: кадры шлёт машина, у неё нет и не
    должно быть пароля пользователя. Ключ проверяется против конкретной
    камеры, а не против общего секрета, — иначе один утёкший ключ открывал
    бы приём за любую камеру системы.

    Отвечаем сразу, как только кадр лёг в очередь. Разбор занимает секунды,
    а камера в это время должна снимать, а не ждать нас.
    """
    cam = s.get(Camera, camera_id)
    key = request.headers.get("X-Camera-Key", "")
    if cam is None or not cam.ingest_key or not secrets.compare_digest(
            key, cam.ingest_key):
        return JSONResponse({"error": "неизвестная камера или ключ"},
                            status_code=403)

    data = await file.read()
    if not data:
        return JSONResponse({"error": "пустой кадр"}, status_code=400)

    try:
        when = (dt.datetime.fromisoformat(captured_at) if captured_at
                else dt.datetime.now(dt.UTC))
    except ValueError:
        return JSONResponse({"error": f"метка времени не разобрана: {captured_at}"},
                            status_code=400)
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.UTC)

    try:
        payload = json.loads(meta) if meta else {}
    except ValueError:
        # Метаданные — дело камеры, и ронять из-за них приём кадра нельзя.
        payload = {"raw": meta[:500]}
    if not isinstance(payload, dict):
        payload = {"value": payload}

    if not live.submit(camera_id, data, when, payload):
        return JSONResponse({"error": "очередь переполнена, повторите позже"},
                            status_code=503)
    return JSONResponse({"ok": True, "queued": True}, status_code=202)


@app.get("/api/fs")
def fs_browse(path: str = "", user: str = Depends(auth.require_user)) -> JSONResponse:
    """Обзор папок на машине сервиса — выбор источника кадров мышью.

    Наружу не выпускает: любой путь приводится к абсолютному и проверяется,
    что он лежит под корнем. Без этого `..` в параметре открыл бы весь диск.
    """
    root = Path(settings.fs_browse_root or Path.home()).resolve()
    try:
        here = Path(path).resolve() if path else root
        here.relative_to(root)
    except (ValueError, OSError):
        here = root
    if not here.is_dir():
        here = root

    dirs, frames_here = [], 0
    try:
        for entry in sorted(here.iterdir(), key=lambda x: x.name.lower()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": str(entry)})
            elif entry.suffix.lower() in ingest.IMAGE_EXT and \
                    ingest.parse_stamp(entry.name):
                frames_here += 1
    except PermissionError:
        pass

    parent = str(here.parent) if here != root else ""
    return JSONResponse({"path": str(here), "parent": parent, "root": str(root),
                         "dirs": dirs, "frames": frames_here})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/sites/{site_id}/stages")
def api_stages(site_id: int, user: str = Depends(auth.require_user),
               s: Session = Depends(get_session)):
    site = s.get(Site, site_id)
    if site is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "site": site.name,
        "stages": [{
            "id": st.id,
            "name": st.title,
            "planned_start": st.planned_start.isoformat() if st.planned_start else None,
            "planned_end": st.planned_end.isoformat() if st.planned_end else None,
            "dates_confirmed": st.dates_confirmed,
            "critical": st.on_critical_path,
            "enabled": st.enabled,
            "equipment_expected": st.equipment_expected,
        } for st in site.stages],
    }
