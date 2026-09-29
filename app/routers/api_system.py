"""Служебное API: здоровье, настройки провайдеров, вход, задания, очередь, демо."""
from __future__ import annotations

import threading

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app import __version__, auth, db, storage
from app.db import get_session
from app.models import Job
from app.routers.common import bad, json_body, not_found, require_obj
from app.services import demo, views
from app.services import settings as settings_svc
from app.services.providers import registry
from app.services.queue import frame_queue

public = APIRouter(prefix="/api", tags=["система"])
router = APIRouter(prefix="/api", tags=["система"], dependencies=[Depends(auth.require_api_user)])


@public.get("/health")
def health(request: Request) -> dict:
    """Открыт без входа (healthcheck Docker): без входа — только «жив ли» и версия.
    Подробности (провайдеры с адресами, очередь) — после входа: адреса внутренних
    сервисов и готовность моделей посторонним знать незачем, а проверка
    провайдеров ходит в сеть — healthcheck каждые 30 с её не дёргает."""
    db_ok = True
    try:
        with db.session() as s:
            s.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        db_ok = False
    storage_ok = storage.get().ping()
    out = {"ok": db_ok and storage_ok, "version": __version__}
    if auth.current_user(request):
        out.update({"providers": registry.status(), "db": db_ok, "storage": storage_ok,
                    "queue": {"running": frame_queue.started, "pending": frame_queue.pending()}})
    return out


@public.post("/login")
async def api_login(request: Request, s: Session = Depends(get_session)) -> dict:
    body = require_obj(await json_body(request))
    try:
        # PBKDF2 — в пуле потоков: 16 параллельных неверных паролей раньше держали цикл
        # событий, и страницы всех пользователей ждали по секунде.
        user = await run_in_threadpool(auth.login_attempt, request, s, str(body.get("login", "")),
                                       str(body.get("password", "")))
    except auth.TooManyAttempts as exc:
        raise HTTPException(429, f"слишком много неверных попыток — повторите через {exc.wait_s} с",
                            headers={"Retry-After": str(exc.wait_s)}) from None
    if user is None:
        raise HTTPException(401, "неверный логин или пароль")
    auth.login_user(request, user)
    return {"ok": True, "user": user.login}


@public.post("/logout")
def api_logout(request: Request, s: Session = Depends(get_session)) -> dict:
    auth.logout_user(request, s)
    return {"ok": True}


@router.get("/me")
def me(user: str = Depends(auth.require_api_user)) -> dict:
    return {"user": user, "version": __version__}


def _settings_json(state: dict) -> dict:
    return {**state, "classes": settings_svc.classes(state["model_a"]), "providers": registry.status(),
            "presets": {k: {"model_a": a, "model_b": b} for k, (a, b) in settings_svc.PRESETS.items()}}


@router.get("/settings")
def get_settings(s: Session = Depends(get_session)) -> dict:
    return _settings_json(settings_svc.get_state(s, fresh=True))


@router.put("/settings")
async def put_settings(request: Request, s: Session = Depends(get_session)) -> dict:
    body = require_obj(await json_body(request))
    before = settings_svc.get_state(s, fresh=True)
    try:
        state = settings_svc.update(s, body)
    except ValueError as exc:
        raise bad(str(exc)) from None
    if state["thresholds"] != before["thresholds"]:
        # Пороги уходят в конструкторы моделей и EquipmentConfig — пересобрать.
        from app.services import pipeline
        registry.reset()
        pipeline.reset_caches()
    if (state["model_a"], state["model_b"]) != (before["model_a"], before["model_b"]):
        # Отложенные кадры могли ждать именно этого провайдера. Проверка
        # готовности может ходить в сеть — не держим запрос.
        threading.Thread(target=frame_queue.recover, kwargs={"include_postponed": True}, daemon=True).start()
    return _settings_json(state)


def _pair_json(p) -> dict | list:
    """Пара «ведущая ↔ обслуживающая» (core.plan.norms.Pair) для UI и документации."""
    if isinstance(p, (list, tuple)):
        return list(p)
    return {"id": p.id, "leader": list(p.leader), "followers": list(p.followers), "severity": p.severity,
            "window_h": p.window_h, "title": p.title}


@router.get("/catalog")
def catalog() -> dict:
    """Справочник для редактора плана и подписей UI: этапы с подэтапами, кодами работ
    xlsx и правилами «этап → техника», словарь техники и признаков чек-листа."""
    from core import taxonomy

    from app.services import providers

    cat = providers.optional_module("core.plan.catalog")
    norms = providers.optional_module("core.plan.norms")
    stages = []
    for st in taxonomy.stages().values():
        works = []
        if cat is not None:
            try:
                works = [{"code": w.code, "name": w.name, "status": w.status, "substage_id": w.substage_id,
                          "key": getattr(w, "key", w.code)}
                         for w in cat.works_for_stage(st.id)]
            except Exception:  # noqa: BLE001 — справочник работ не обязателен для остального
                works = []
        rule = None
        if norms is not None:
            try:
                req = norms.requirement(st.id)
                rule = {"expected": list(req.expected), "optional": list(req.optional),
                        "forbidden": list(req.forbidden), "pairs": [_pair_json(p) for p in req.pairs],
                        "min_count": dict(req.min_count), "default_equipment": norms.default_equipment(st.id)}
            except Exception:  # noqa: BLE001
                rule = None
        stages.append({
            "id": st.id, "key": st.key, "name": st.name, "weight": st.weight,
            "equipment_expected": list(st.equipment_expected), "equipment_optional": list(st.equipment_optional),
            "equipment_forbidden": list(st.equipment_forbidden), "rule": rule,
            "substages": [{"id": x.get("id"), "name": x.get("name"), "xlsx": x.get("xlsx", [])}
                          for x in st.substages],
            "works": works,
        })
    return {
        "stages": stages,
        "equipment": [{"key": e.key, "name": e.name, "tz": e.tz} for e in taxonomy.equipment().values()],
        "signs": [{"key": x.key, "question": x.question, "latching": x.latching} for x in taxonomy.signs().values()],
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str, s: Session = Depends(get_session)) -> dict:
    job = s.get(Job, job_id)
    if job is None:
        raise not_found(f"задание {job_id} не найдено")
    return views.job_json(s, job)


@router.get("/jobs")
def list_jobs(limit: int = 20, s: Session = Depends(get_session)) -> list[dict]:
    limit = max(1, min(limit, 200))
    jobs = s.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit)).all()
    return [views.job_json(s, j) for j in jobs]


@router.get("/queue")
def queue_status() -> dict:
    return frame_queue.status()


@router.post("/demo/seed")
async def demo_seed(request: Request, s: Session = Depends(get_session)) -> dict:
    body = require_obj(await json_body(request, default={}))
    only = body.get("sites")
    if only is not None and (not isinstance(only, list) or not all(isinstance(x, str) for x in only)):
        raise bad("sites: список имён каталогов/объектов")
    try:
        # Засев читает каталоги и пишет в БД — в пуле потоков, не в цикле событий.
        result = await run_in_threadpool(demo.seed, s, only=only, replace=bool(body.get("replace", False)))
    except demo.DemoError as exc:
        raise not_found(str(exc)) from None
    return {"sites": result["sites"], "jobs": [j["job_id"] for j in result["jobs"]],
            "warnings": result["warnings"]}
