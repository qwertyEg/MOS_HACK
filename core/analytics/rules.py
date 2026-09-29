"""Отклонения по площадке — все типы DeviationType с объяснением и снимками-доказательствами.

Каждое отклонение объясняет себя одинаково: «Видели: … По плану ожидается: … Что проверить: …»,
несёт кадры-доказательства (frame_ids — UI рисует на них рамки), этап, камеру/зону и
стабильный `key`: при следующем пересчёте то же самое отклонение получает тот же ключ,
и веб-слой обновляет запись, а не плодит дубли. В ключ входит начало эпизода
(например, время начала «экскаватор без самосвалов»), поэтому новый эпизод завтра —
новая запись, а продолжение текущего — та же.

Техника против этапа оценивается по ОКНУ ВРЕМЕНИ, а не по одному кадру: самосвал мог
просто выехать из кадра на 20 минут (баг ветки api-solution — парные правила по кадру).
Учитывается только задействованная техника (ACTIVE/IDLE); PARKED и техника в зоне
отстоя в сопоставлении с этапом не участвуют (PLAN.md §3.9), но дают EQUIPMENT_PARKED_ONLY.

Пороги — core.analytics.context.AnalyticsConfig; правила «этап → техника» — core.plan.norms.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from core.analytics import fmt
from core.analytics.context import AnalyticsContext, Obs, count_units
from core.contracts import (
    Activity, DeviationRecord, DeviationType as DT, Severity, StageStatus, UnitStatus,
)
from core.plan import norms

_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


def _sev(name: str) -> Severity:
    return Severity(name)


def _bump(sev: Severity) -> Severity:
    return Severity.CRITICAL if sev != Severity.INFO else Severity.WARNING


def _stamp(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M")


def _spread(ids: list, n: int = 3) -> list:
    """До n доказательств, равномерно по эпизоду: первый, середина, последний."""
    ids = list(dict.fromkeys(ids))
    if len(ids) <= n:
        return ids
    step = (len(ids) - 1) / (n - 1)
    return list(dict.fromkeys(ids[round(i * step)] for i in range(n)))


def _latest_frames(ctx: AnalyticsContext, n: int = 3, after: dt.datetime | None = None,
                   usable: bool = False) -> list:
    """Последние кадры (по одному на камеру, пока хватает) — доказательство «на снимках этого нет»."""
    frames = [f for f in ctx.frames if after is None or f.captured_at > after]
    if usable:
        good = [f for f in frames if f.quality_ok and not f.is_night]
        frames = good or frames
    picked, seen = [], set()
    for f in reversed(frames):
        if f.camera_id not in seen:
            picked.append(f)
            seen.add(f.camera_id)
        if len(picked) >= n:
            break
    for f in reversed(frames):
        if len(picked) >= n:
            break
        if f not in picked:
            picked.append(f)
    return [f.frame_id for f in sorted(picked, key=lambda f: f.captured_at)]


def _plan_text(ctx: AnalyticsContext, stage: int) -> str:
    it = ctx.plan_by_stage.get(stage)
    if it and it.planned_start and it.planned_end:
        return f"идёт этап {fmt.stage(stage)} ({fmt.date(it.planned_start)}–{fmt.date(it.planned_end)})"
    return f"идёт этап {fmt.stage(stage)} (по снимкам; в плане его нет)"


def _plan_dates(ctx: AnalyticsContext, stage: int) -> str:
    it = ctx.plan_by_stage.get(stage)
    if it and it.planned_start and it.planned_end:
        return f"по плану {fmt.date(it.planned_start)}–{fmt.date(it.planned_end)}"
    return "в плане этапа нет"


def _stage_start(ctx: AnalyticsContext, stage: int) -> dt.datetime | None:
    """Начало этапа: плановое или фактическое — что раньше (техника нужна с первого дня работ)."""
    days = []
    it = ctx.plan_by_stage.get(stage)
    if it and it.planned_start:
        days.append(it.planned_start)
    st = ctx.timeline.states.get(stage)
    if st and st.actual_start:
        days.append(st.actual_start)
    return ctx.day_start(min(days)) if days else None


def _detectable(ctx: AnalyticsContext, types) -> list[str]:
    d = ctx.cfg.detectable
    return [t for t in types if d is None or t in d]


def _most_common_zone(obs: list[Obs]):
    zones = Counter(o.zone.id for o in obs if o.zone is not None)
    if not zones:
        return None
    zid = zones.most_common(1)[0][0]
    return next(o.zone for o in obs if o.zone is not None and o.zone.id == zid)


def _cams(obs: list[Obs]) -> list:
    return list(dict.fromkeys(o.camera for o in obs))


def _main_camera(obs: list[Obs]):
    c = Counter(o.camera for o in obs)
    return c.most_common(1)[0][0] if c else None


# --------------------------------------------------------------------------
# PAIR_BROKEN — ведущая техника без обслуживающей дольше окна
# --------------------------------------------------------------------------


@dataclass
class _Stretch:
    start: dt.datetime
    last_seen: dt.datetime
    last_work: dt.datetime
    obs: list[Obs] = field(default_factory=list)


def _stretches(ctx: AnalyticsContext, pair: norms.Pair) -> list[_Stretch]:
    """Эпизоды «ведущая есть (работает), обслуживающей нет» по всем камерам площадки.

    Появление обслуживающей техники на любой камере рвёт эпизод (единицы уже склеены
    между камерами моделью А). Разрыв в наблюдениях ведущей дольше max_gap тоже рвёт:
    что было в это время, мы не знаем. Для «работающей» ведущей эпизод начинается с кадра,
    где она работает, и заканчивается, если она простояла дольше idle_break.
    """
    cfg = ctx.cfg
    active = pair.leader_state == "active"
    events: list[tuple[dt.datetime, int, Obs]] = []
    for o in ctx.observations:
        if o.cls in pair.leader and not o.parked:
            events.append((o.t, 1, o))
        elif o.cls in pair.followers and o.status != UnitStatus.PARKED:
            if pair.follower_state == "active" and not o.working:
                continue
            events.append((o.t, 0, o))  # при равном времени обслуживающая идёт первой — пара цела
    events.sort(key=lambda e: (e[0], e[1]))
    max_gap = dt.timedelta(minutes=cfg.max_gap_min)
    idle_break = dt.timedelta(minutes=cfg.idle_break_min)
    out: list[_Stretch] = []
    cur: _Stretch | None = None
    for t, kind, o in events:
        if kind == 0:
            if cur:
                out.append(cur)
            cur = None
            continue
        if cur and t - cur.last_seen > max_gap:
            out.append(cur)
            cur = None
        if cur and active and not o.working and t - cur.last_work > idle_break:
            out.append(cur)
            cur = None
        if cur is None:
            if active and not o.working:
                continue
            cur = _Stretch(t, t, t)
        cur.obs.append(o)
        cur.last_seen = t
        if o.working or not active:
            cur.last_work = t
    if cur:
        out.append(cur)
    return out


def pair_broken(ctx: AnalyticsContext) -> list[DeviationRecord]:
    chosen: dict[str, tuple[norms.Pair, int]] = {}
    for s in ctx.current_stages():
        for p in norms.requirement(s).pairs:
            chosen.setdefault(p.id, (p, s))
    out = []
    last = ctx.last_frame_at
    for pid, (p, stage) in chosen.items():
        for st in _stretches(ctx, p):
            active = p.leader_state == "active"
            end = st.last_work if active else st.last_seen
            dur_h = (end - st.start).total_seconds() / 3600
            if dur_h + 1e-6 < p.window_h:
                continue
            units = list(dict.fromkeys(o.unit for o in st.obs))
            n_units = count_units(st.obs)
            if n_units < p.leader_min_units:
                continue
            leader_cls = Counter(o.cls for o in st.obs).most_common(1)[0][0]
            evidence_obs = [o for o in st.obs if o.working] if active else st.obs
            frame_ids = _spread([o.frame.frame_id for o in evidence_obs], 3)
            zone = _most_common_zone(st.obs)
            cams = _cams(st.obs)
            cams_txt = fmt.cameras(cams, ctx.cfg.camera_names)
            zone_txt = f"зона «{zone.name}»" if zone else "зона не размечена"
            title = p.title.format(leader=fmt.eq(leader_cls))
            headline = f"{title} {fmt.hours(dur_h)}, {zone_txt}, {cams_txt}"
            shots = fmt.count(len({o.frame.frame_id for o in st.obs}), fmt.SHOTS)
            period = f"с {fmt.moment(ctx.local(st.start))} до {fmt.moment(ctx.local(end))}"
            if active:
                seen = (f"{fmt.eq(leader_cls)} работает {period} ({shots}), обслуживающей техники "
                        f"({fmt.eq_list(p.followers)}) за это время не было ни на одной камере")
            else:
                seen = (f"на площадке {n_units} ед. ({fmt.eq(leader_cls)}) {period} ({shots}), а "
                        f"{fmt.eq_list(p.followers)} за это время не работал ни один")
            sev = _sev(p.severity)
            if p.escalate_after_h and dur_h >= p.escalate_after_h:
                sev = _bump(sev)
            ongoing = last is not None and last - end <= dt.timedelta(minutes=ctx.cfg.max_gap_min)
            out.append(DeviationRecord(
                key=f"pair_broken:{pid}:{_stamp(st.start)}",
                type=DT.PAIR_BROKEN, severity=sev, title=title,
                message=(f"{headline}. Видели: {seen}. По плану ожидается: {_plan_text(ctx, stage)} — "
                         f"{p.why}. Что проверить: {p.check}."),
                stage_id=stage, camera_id=_main_camera(st.obs), zone_id=zone.id if zone else None,
                frame_ids=frame_ids, unit_ids=units, started_at=st.start, last_seen_at=end,
                data={"pair": pid, "leader": leader_cls, "followers": list(p.followers),
                      "duration_h": round(dur_h, 2), "window_h": p.window_h, "cameras": cams,
                      "zone": zone.name if zone else None, "ongoing": ongoing, "headline": headline},
            ))
    return out


# --------------------------------------------------------------------------
# EQUIPMENT_MISSING — нужной техники нет в окне
# --------------------------------------------------------------------------


def _last_seen(ctx: AnalyticsContext, types) -> dt.datetime | None:
    types = set(types)
    ts = [o.t for o in ctx.observations if o.cls in types and not o.parked]
    ts += [u.last_seen for u in ctx.units if u.cls in types and u.status in (UnitStatus.ACTIVE, UnitStatus.IDLE)]
    return max(ts) if ts else None


def _obs_in_window(ctx: AnalyticsContext, types, start: dt.datetime, end: dt.datetime) -> list[Obs]:
    types = set(types)
    return [o for o in ctx.observations if o.cls in types and not o.parked and start <= o.t <= end]


def equipment_missing(ctx: AnalyticsContext, pair_devs: list[DeviationRecord] | None = None) -> list[DeviationRecord]:
    if not ctx.frames:
        return []
    cfg = ctx.cfg
    end = min(ctx.now, ctx.last_frame_at)
    first = ctx.frames[0].captured_at
    # Отсутствие обслуживающей техники при работающей ведущей уже объяснено парным правилом.
    explained = {f for d in (pair_devs or []) if d.data.get("ongoing") for f in d.data.get("followers", [])}
    out = []

    def absence(types, stage_start):
        last = _last_seen(ctx, types)
        ref = last or first
        if stage_start and ref < stage_start:
            ref = stage_start
        frames = [f for f in ctx.frames if ref < f.captured_at <= end]
        return last, ref, ctx.working_hours(ref, end), frames

    def episode(last, stage_start) -> str:
        # Якорь эпизода для ключа — последнее появление техники или начало этапа, а не первый кадр
        # окна: окно `recent` скользит, и ключ от него менялся бы при каждом пересчёте.
        anchor = last if last and (not stage_start or last >= stage_start) else stage_start
        return _stamp(anchor) if anchor else "open"

    for s in ctx.current_stages():
        req = norms.requirement(s)
        item = ctx.plan_by_stage.get(s)
        plan_eq = dict(item.equipment) if item else {}
        stage_start = _stage_start(ctx, s)
        required = {t: n for t, n in req.min_count.items() if _detectable(ctx, req.accepted(t))}

        # 1) на площадке нет ничего из техники этапа
        all_types = _detectable(ctx, req.stage_types())
        if required and all_types:
            win = min(req.missing_window_h.get(t, cfg.missing_window_h) for t in required)
            last, ref, absent_h, frames = absence(all_types, stage_start)
            if absent_h + 1e-6 >= win and len(frames) >= cfg.min_frames_for_absence:
                need = ", ".join(f"{fmt.eq(t)} (не менее {n})" for t, n in required.items())
                sev = Severity.CRITICAL if absent_h >= win * cfg.missing_critical_factor else Severity.WARNING
                seen_last = f"последний раз техника этапа была {fmt.moment(ctx.local(last))}" if last else \
                    "с начала наблюдений техники этапа не было"
                out.append(DeviationRecord(
                    key=f"equipment_missing:{s}:all:{episode(last, stage_start)}",
                    type=DT.EQUIPMENT_MISSING, severity=sev,
                    title=f"Нет техники этапа {fmt.stage(s)}",
                    message=(f"Видели: {fmt.hours(absent_h)} рабочего времени ({fmt.count(len(frames), fmt.SHOTS)}, "
                             f"{fmt.cameras(list(dict.fromkeys(f.camera_id for f in frames)), cfg.camera_names)}) "
                             f"на площадке нет ни одной единицы техники этапа; {seen_last}. "
                             f"По плану ожидается: {_plan_text(ctx, s)} — нужны {need}. "
                             "Что проверить: не вывезли ли технику, не остановлены ли работы, видят ли камеры рабочую зону."),
                    stage_id=s, frame_ids=_latest_frames(ctx, 3, after=ref), started_at=ref, last_seen_at=end,
                    data={"types": list(required), "absent_h": round(absent_h, 2), "window_h": win},
                ))
                continue

        # 2) нет конкретного обязательного типа / его меньше нормы
        for t, n in required.items():
            types = _detectable(ctx, req.accepted(t))
            win = req.missing_window_h.get(t, cfg.missing_window_h)
            last, ref, absent_h, frames = absence(types, stage_start)
            alt = f" (или {fmt.eq_list(types[1:])})" if len(types) > 1 else ""
            if absent_h + 1e-6 >= win and len(frames) >= cfg.min_frames_for_absence:
                if set(types) & explained:
                    continue
                sev = Severity.CRITICAL if absent_h >= win * cfg.missing_critical_factor else Severity.WARNING
                seen_last = f"последний раз — {fmt.moment(ctx.local(last))}" if last else "с начала наблюдений не появлялся"
                out.append(DeviationRecord(
                    key=f"equipment_missing:{s}:{t}:{episode(last, stage_start)}",
                    type=DT.EQUIPMENT_MISSING, severity=sev,
                    title=f"Нет техники этапа: {fmt.eq(t)}",
                    message=(f"Видели: {fmt.eq(t)}{alt} не замечен {fmt.hours(absent_h)} рабочего времени "
                             f"({fmt.count(len(frames), fmt.SHOTS)}), {seen_last}. "
                             f"По плану ожидается: {_plan_text(ctx, s)} — нужен {fmt.eq(t)}, не менее {n} ед. "
                             "Что проверить: на площадке ли техника, не вне поля зрения камер ли она работает, "
                             "не сорван ли график поставки техники."),
                    stage_id=s, frame_ids=_latest_frames(ctx, 3, after=ref), started_at=ref, last_seen_at=end,
                    data={"cls": t, "accepted": types, "absent_h": round(absent_h, 2), "window_h": win},
                ))
                continue
            need = max(n, plan_eq.get(t, 0))
            win_start = ctx.working_window_start(end, win)
            obs = _obs_in_window(ctx, types, win_start, end)
            seen = count_units(obs)
            if 0 < seen < need:
                out.append(DeviationRecord(
                    key=f"equipment_missing:{s}:{t}:few",
                    type=DT.EQUIPMENT_MISSING, severity=Severity.INFO,
                    title=f"Мало техники: {fmt.eq(t)} — {seen} из {need}",
                    message=(f"Видели: за последние {fmt.hours(win)} рабочего времени {fmt.eq(t)}{alt}: "
                             f"{seen} ед. По плану ожидается: {_plan_text(ctx, s)} — не менее {need} ед. "
                             "Что проверить: хватает ли техники на темп этапа."),
                    stage_id=s, frame_ids=_spread([o.frame.frame_id for o in obs], 3),
                    unit_ids=sorted({o.unit for o in obs if o.identified}), started_at=win_start, last_seen_at=end,
                    data={"cls": t, "seen": seen, "need": need},
                ))

        # 3) техника, которую требует план (но не нормы): мягче и с окном в рабочий день
        for t, n in plan_eq.items():
            if n <= 0 or any(t in req.accepted(r) for r in req.min_count) or t in req.forbidden:
                continue
            if not _detectable(ctx, [t]):
                continue
            last, ref, absent_h, frames = absence([t], stage_start)
            if absent_h + 1e-6 >= cfg.plan_missing_window_h and len(frames) >= cfg.min_frames_for_absence:
                out.append(DeviationRecord(
                    key=f"equipment_missing:{s}:{t}:plan:{episode(last, stage_start)}",
                    type=DT.EQUIPMENT_MISSING, severity=Severity.INFO,
                    title=f"Нет техники по плану: {fmt.eq(t)}",
                    message=(f"Видели: {fmt.eq(t)} не замечен {fmt.hours(absent_h)} рабочего времени. "
                             f"По плану ожидается: {_plan_text(ctx, s)} — в плане {n} ед. этого типа. "
                             "Что проверить: актуален ли план по технике."),
                    stage_id=s, frame_ids=_latest_frames(ctx, 3, after=ref), started_at=ref, last_seen_at=end,
                    data={"cls": t, "plan_count": n, "absent_h": round(absent_h, 2)},
                ))
    return out


# --------------------------------------------------------------------------
# EQUIPMENT_FORBIDDEN — работает техника не по этапу
# --------------------------------------------------------------------------


def equipment_forbidden(ctx: AnalyticsContext) -> list[DeviationRecord]:
    current = ctx.current_stages()
    if not current:
        return []
    allowed: set[str] = set()
    forbidden_by: dict[str, int] = {}
    for s in current:
        req = norms.requirement(s)
        allowed |= req.allowed
        for t in req.forbidden:
            forbidden_by.setdefault(t, s)
    by_unit: dict[str, list[Obs]] = defaultdict(list)
    for o in ctx.observations:
        if o.working and not o.parked:
            by_unit[o.unit].append(o)
    out = []
    for unit, obs in by_unit.items():
        cls = Counter(o.cls for o in obs).most_common(1)[0][0]
        if cls not in forbidden_by or cls in allowed or len(obs) < ctx.cfg.forbidden_min_obs:
            continue
        s = forbidden_by[cls]
        req = norms.requirement(s)
        zone = _most_common_zone(obs)
        sev = Severity.INFO if cls in req.forbidden_info else Severity.WARNING
        ok = fmt.eq_list(sorted(set(req.expected)))
        out.append(DeviationRecord(
            key=f"equipment_forbidden:{unit}:{s}",
            type=DT.EQUIPMENT_FORBIDDEN, severity=sev,
            title=f"Техника не по этапу: {fmt.eq(cls)} работает",
            message=(f"Видели: {ctx.unit_label(unit, cls)} работает — {fmt.count(len(obs), fmt.SHOTS)} "
                     f"с {fmt.moment(ctx.local(obs[0].t))} по {fmt.moment(ctx.local(obs[-1].t))}, "
                     f"{fmt.cameras(_cams(obs), ctx.cfg.camera_names)}"
                     f"{', зона «' + zone.name + '»' if zone else ''}. "
                     f"По плану ожидается: {_plan_text(ctx, s)} — {fmt.eq(cls)} на нём не нужен "
                     f"(типичная техника этапа: {ok}). Что проверить: не параллельная ли это работа, которой нет "
                     "в плане (засыпка пазух, наружные сети — тогда добавьте этап в план), не ошибка ли распознавания."),
            stage_id=s, camera_id=_main_camera(obs), zone_id=zone.id if zone else None,
            frame_ids=_spread([o.frame.frame_id for o in obs], 3), unit_ids=[unit],
            started_at=obs[0].t, last_seen_at=obs[-1].t,
            data={"cls": cls, "observations": len(obs), "current_stages": current},
        ))
    return out


# --------------------------------------------------------------------------
# EQUIPMENT_IDLE и EQUIPMENT_PARKED_ONLY — техника есть, но не работает
# --------------------------------------------------------------------------


def _required_types(ctx: AnalyticsContext, stage: int) -> dict[str, tuple[str, ...]]:
    """Обязательные типы этапа → засчитываемые типы: нормы + план + ненулевые «полоски»."""
    req = norms.requirement(stage)
    out = {t: req.accepted(t) for t in req.min_count}
    item = ctx.plan_by_stage.get(stage)
    extra = [t for t, n in (item.equipment if item else {}).items() if n > 0]
    extra += [b.cls for b in ctx.balances if b.stage_id == stage and b.planned_hours > 0 and b.remaining_hours > 0]
    for t in extra:
        if t not in out and not any(t in acc for acc in out.values()) and t not in req.forbidden:
            out[t] = (t,)
    return out


def equipment_idle(ctx: AnalyticsContext) -> list[DeviationRecord]:
    cfg = ctx.cfg
    end = min(ctx.now, ctx.last_frame_at) if ctx.last_frame_at else ctx.now
    out = []
    for s in ctx.current_stages():
        stage_start = _stage_start(ctx, s)
        for t, types in _required_types(ctx, s).items():
            if not _detectable(ctx, types):
                continue
            units = [u for u in ctx.units if u.cls in types and u.status != UnitStatus.DEPARTED]
            if not units or any(u.status == UnitStatus.ACTIVE for u in units):
                continue
            if all(u.status == UnitStatus.PARKED for u in units):
                continue  # это EQUIPMENT_PARKED_ONLY
            worked = [i.end for i in ctx.intervals if i.cls in types and i.hours > 0]
            worked += [b.last_worked_at for b in ctx.balances if b.cls in types and b.last_worked_at]
            worked += [u.last_moved for u in units if u.last_moved]
            last_worked = max(worked) if worked else None
            # Простой отсчитываем от последней работы, но не раньше начала этапа и не раньше,
            # чем техника появилась на площадке: стоять на площадке до своего появления она не могла.
            ref = max(t for t in (last_worked, stage_start, min(u.first_seen for u in units)) if t)
            unit_ids = [u.unit_id for u in units]
            obs = [o for o in ctx.observations if o.unit in set(unit_ids)]
            # Простой — то, что ВИДЕЛИ: кадры, сравненные с прошлым без разрыва (активность
            # оценена). У камеры, снимающей раз в сутки, активность не оценивается вовсе, и
            # «стоит 3909 ч» (архив Канберры) значило бы «работы не видно», а не простой.
            judged = [o.t for o in obs if o.det.activity in (Activity.IDLE, Activity.WORKING)]
            if not judged:
                continue
            seen_until = min(end, max(judged))
            idle_h = ctx.working_hours(ref, seen_until) if seen_until > ref else 0.0
            if idle_h + 1e-6 < cfg.idle_alert_h:
                continue
            sev = Severity.CRITICAL if idle_h >= cfg.idle_critical_h else Severity.WARNING
            bal = [b for b in ctx.balances if b.stage_id == s and b.cls in types]
            planned, done = sum(b.planned_hours for b in bal), sum(b.worked_hours for b in bal)
            bar = (f"по этапу отработано {done:.0f} из {planned:.0f} ч, осталось {max(0.0, planned - done):.0f} ч"
                   if planned > 0 else "плановых моточасов по этому типу нет")
            since = (f"последняя работа — {fmt.moment(ctx.local(last_worked))}" if last_worked
                     else "работы с момента появления на площадке не было")
            labels = ", ".join(ctx.unit_label(u.unit_id, u.cls) for u in units[:4])
            out.append(DeviationRecord(
                key=f"equipment_idle:{s}:{t}:{_stamp(ref)}",
                type=DT.EQUIPMENT_IDLE, severity=sev,
                title=f"Простой: {fmt.eq(t)} стоит {fmt.hours(idle_h)}",
                message=(f"Видели: {labels} на площадке, но не работает {fmt.hours(idle_h)} рабочего времени "
                         f"({since}); «полоска» моточасов не уменьшается: {bar}. "
                         f"По плану ожидается: {_plan_text(ctx, s)} — {fmt.eq(t)} должен работать. "
                         "Что проверить: поломка, нет фронта работ или материалов, простой оплачивается подрядчику."),
                stage_id=s, camera_id=_main_camera(obs), frame_ids=_spread([o.frame.frame_id for o in obs][-9:], 3),
                unit_ids=unit_ids, started_at=ref, last_seen_at=seen_until,
                data={"cls": t, "idle_h": round(idle_h, 2), "planned_hours": planned, "worked_hours": done},
            ))
    return out


def equipment_parked_only(ctx: AnalyticsContext) -> list[DeviationRecord]:
    out = []
    for s in ctx.current_stages():
        req = norms.requirement(s)
        types = req.stage_types()
        units = [u for u in ctx.units if u.cls in types and u.status != UnitStatus.DEPARTED]
        if not units or not all(u.status == UnitStatus.PARKED for u in units):
            continue
        ids = {u.unit_id for u in units}
        obs = [o for o in ctx.observations if o.unit in ids]
        since = min((u.last_moved or u.first_seen) for u in units)
        labels = ", ".join(ctx.unit_label(u.unit_id, u.cls) for u in units[:5])
        out.append(DeviationRecord(
            key=f"equipment_parked_only:{s}",
            type=DT.EQUIPMENT_PARKED_ONLY, severity=Severity.WARNING,
            title=f"Работы встали: вся техника этапа {fmt.stage(s)} на стоянке",
            message=(f"Видели: на площадке {labels} — все без движения дольше порога стоянки "
                     f"(не работают с {fmt.moment(ctx.local(since))}). "
                     f"По плану ожидается: {_plan_text(ctx, s)} — техника этапа должна работать; "
                     "визуально площадка укомплектована, но работы не идут. "
                     "Что проверить: ждёт ли техника вывоза, остановлены ли работы, нет ли фронта."),
            stage_id=s, camera_id=_main_camera(obs), frame_ids=_spread([o.frame.frame_id for o in obs][-9:], 3),
            unit_ids=sorted(ids), started_at=since, last_seen_at=max(u.last_seen for u in units),
            data={"units": sorted(ids)},
        ))
    return out


# --------------------------------------------------------------------------
# OUTSIDE_ZONE — техника в запретной зоне или вне размеченных зон
# --------------------------------------------------------------------------


def outside_zone(ctx: AnalyticsContext) -> list[DeviationRecord]:
    cfg = ctx.cfg
    restricted: dict[tuple[str, int], list[Obs]] = defaultdict(list)
    outside: dict[str, list[Obs]] = defaultdict(list)
    zoned_cams = {str(z.camera_id) for z in ctx.zones if z.kind in ("work", "parking", "storage")}
    for o in ctx.observations:
        if o.zone is not None and o.zone.kind == "restricted":
            restricted[(o.unit, o.zone.id)].append(o)   # и стоящая в запретной зоне — нарушение
        elif o.zone is None and o.status != UnitStatus.PARKED and str(o.camera) in zoned_cams:
            outside[o.unit].append(o)
    out = []
    for (unit, zid), obs in restricted.items():
        if len(obs) < cfg.outside_min_obs:
            continue
        zone = obs[0].zone
        cls = Counter(o.cls for o in obs).most_common(1)[0][0]
        out.append(DeviationRecord(
            key=f"outside_zone:{unit}:{zid}",
            type=DT.OUTSIDE_ZONE, severity=Severity.WARNING,
            title=f"Техника в запретной зоне «{zone.name}»: {fmt.eq(cls)}",
            message=(f"Видели: {ctx.unit_label(unit, cls)} в зоне «{zone.name}» (запретная) — "
                     f"{fmt.count(len(obs), fmt.SHOTS)} с {fmt.moment(ctx.local(obs[0].t))} "
                     f"по {fmt.moment(ctx.local(obs[-1].t))}, {fmt.cameras(_cams(obs), cfg.camera_names)}. "
                     "По плану ожидается: в запретной зоне техники быть не должно. "
                     "Что проверить: безопасность (охранная зона сетей, пожарный проезд), разметку зоны."),
            zone_id=zid, camera_id=_main_camera(obs), frame_ids=_spread([o.frame.frame_id for o in obs], 3),
            unit_ids=[unit], started_at=obs[0].t, last_seen_at=obs[-1].t, stage_id=None,
            data={"cls": cls, "zone": zone.name, "kind": "restricted"},
        ))
    for unit, obs in outside.items():
        if len(obs) < cfg.outside_min_obs:
            continue
        cls = Counter(o.cls for o in obs).most_common(1)[0][0]
        out.append(DeviationRecord(
            key=f"outside_zone:{unit}:none",
            type=DT.OUTSIDE_ZONE, severity=Severity.INFO,
            title=f"Техника вне рабочих зон: {fmt.eq(cls)}",
            message=(f"Видели: {ctx.unit_label(unit, cls)} вне размеченных рабочих зон и зон стоянки — "
                     f"{fmt.count(len(obs), fmt.SHOTS)}, {fmt.cameras(_cams(obs), cfg.camera_names)}. "
                     "По плану ожидается: техника работает в рабочих зонах или стоит в зоне стоянки. "
                     "Что проверить: не чужая ли это техника, не пора ли расширить рабочую зону."),
            camera_id=_main_camera(obs), frame_ids=_spread([o.frame.frame_id for o in obs], 3),
            unit_ids=[unit], started_at=obs[0].t, last_seen_at=obs[-1].t,
            data={"cls": cls, "kind": "outside"},
        ))
    return out


# --------------------------------------------------------------------------
# HOURS_SPENT_NO_PROGRESS — сверка моделей А и Б
# --------------------------------------------------------------------------


def _days_since_front_change(ctx: AnalyticsContext) -> int | None:
    days = sorted(ctx.timeline.daily_front)
    if not days:
        return None
    last_change = days[0][0]
    for (d0, f0), (d1, f1) in zip(days, days[1:]):
        if f1 != f0:
            last_change = d1
    return (ctx.today - last_change).days


def _hours_budget(b) -> float:
    """С чем сверять отработанные часы: план на НАБЛЮДАЕМУЮ часть этапа, если камеры
    начали снимать посреди этапа (что было до первого кадра, модель А не видела и
    не списала), иначе — план этапа целиком."""
    seen = getattr(b, "planned_observed_hours", None)
    if seen is not None and 0 < seen < b.planned_hours:
        return seen
    return b.planned_hours


def hours_spent_no_progress(ctx: AnalyticsContext) -> list[DeviationRecord]:
    cfg = ctx.cfg
    by_stage: dict[int, list] = defaultdict(list)
    for b in ctx.balances:
        budget = _hours_budget(b)
        if b.stage_id is not None and budget > 0 and b.worked_hours / budget >= cfg.hours_warn_ratio:
            by_stage[b.stage_id].append(b)
    unchanged = _days_since_front_change(ctx)
    out = []
    for s, bals in sorted(by_stage.items()):
        st = ctx.timeline.states.get(s)
        if st and st.status == StageStatus.DONE:
            continue
        front = ctx.timeline.current_stage
        if unchanged is not None and front == s and unchanged < cfg.no_progress_days:
            continue  # этап только что сменился — рано судить
        ratio = max(b.worked_hours / _hours_budget(b) for b in bals)
        if unchanged is None:
            sev = Severity.INFO
            fact = "данных модели Б по этапу нет — завершение не подтверждено снимками"
        else:
            sev = Severity.CRITICAL if ratio >= cfg.hours_critical_ratio else Severity.WARNING
            prog = f", готовность этапа {fmt.pct(st.progress)}" if st else ""
            status = "не начат" if not st or st.status == StageStatus.NOT_STARTED else "всё ещё идёт"
            fact = f"по снимкам этап {status}{prog}, текущий этап не менялся уже {fmt.count(unchanged, fmt.DAYS)}"
        def _spent(b) -> str:
            budget = _hours_budget(b)
            part = (f" — на период съёмки с {fmt.date(ctx.local(b.expected_from).date())}"
                    if budget < b.planned_hours and getattr(b, "expected_from", None) else "")
            return (f"{fmt.eq(b.cls)} — {b.worked_hours:.0f} из {budget:.0f} ч{part} "
                    f"({round(100 * b.worked_hours / budget)} %)")
        spent = "; ".join(_spent(b) for b in bals)
        types = {b.cls for b in bals}
        ev = [fid for i in ctx.intervals if i.stage_id == s and i.cls in types for fid in i.frame_ids]
        ev = _spread(ev[-12:], 2) + ([st.evidence_frame_ids[-1]] if st and st.evidence_frame_ids else [])
        out.append(DeviationRecord(
            key=f"hours_no_progress:{s}",
            type=DT.HOURS_SPENT_NO_PROGRESS, severity=sev,
            title=f"Моточасы израсходованы, этап {fmt.stage(s)} не завершён",
            message=(f"Видели: {spent}; {fact}. По плану ожидается: при таком расходе моточасов этап {fmt.stage(s)} "
                     f"уже должен быть завершён ({_plan_dates(ctx, s)}). Вероятна задержка: темп ниже нормы или объём "
                     "больше плана. "
                     "Что проверить: реальный объём работ, производительность техники, не списываются ли "
                     "часы на чужой этап; при необходимости поправьте плановые часы."),
            stage_id=s, frame_ids=list(dict.fromkeys(ev)),
            last_seen_at=max((b.last_worked_at for b in bals if b.last_worked_at), default=None),
            data={"ratio": round(ratio, 3), "types": sorted(types), "front_unchanged_days": unchanged},
        ))
    return out


# --------------------------------------------------------------------------
# график: STAGE_LATE_START / STAGE_OVERDUE / STAGE_EARLY / STAGE_OUT_OF_PLAN
# --------------------------------------------------------------------------


def schedule(ctx: AnalyticsContext) -> list[DeviationRecord]:
    if not ctx.has_stage_data():
        return []  # без данных модели Б «этап не начат» было бы ложью
    cfg = ctx.cfg
    tol, crit = cfg.schedule_tolerance_days, cfg.schedule_critical_days
    today = ctx.today
    last_frames = _latest_frames(ctx, 3, usable=True)
    last_shot = fmt.date(ctx.local(ctx.last_frame_at)) if ctx.last_frame_at else "—"
    out = []
    planned = {s: it for s, it in ctx.plan_by_stage.items() if it.planned_start or it.planned_end}
    for s, it in sorted(planned.items()):
        st = ctx.timeline.states.get(s)
        status = st.status if st else StageStatus.NOT_STARTED
        a_start, a_end = (st.actual_start, st.actual_end) if st else (None, None)
        first_ev = st.evidence_frame_ids[:1] if st and st.evidence_frame_ids else []
        last_ev = st.evidence_frame_ids[-1:] if st and st.evidence_frame_ids else []
        ps, pe = it.planned_start, it.planned_end
        if ps:
            late = (today - ps).days
            if status == StageStatus.NOT_STARTED and late > tol:
                out.append(DeviationRecord(
                    key=f"stage_late_start:{s}", type=DT.STAGE_LATE_START,
                    severity=Severity.CRITICAL if late > crit else Severity.WARNING,
                    title=f"Этап {fmt.stage(s)} не начат: опоздание {fmt.days(late)}",
                    message=(f"Видели: на снимках нет признаков этапа {fmt.stage(s)} (последний снимок {last_shot}). "
                             f"По плану ожидается: начало {fmt.date(ps)} — {fmt.days(late)} назад. "
                             "Что проверить: готов ли фронт (предыдущий этап), есть ли техника и бригада; "
                             "если этап идёт вне поля зрения камер — отметьте его вручную."),
                    stage_id=s, frame_ids=last_frames, data={"days": late, "planned_start": ps.isoformat()},
                ))
            elif a_start and (a_start - ps).days > tol and status == StageStatus.ACTIVE:
                d = (a_start - ps).days
                out.append(DeviationRecord(
                    key=f"stage_late_start:{s}", type=DT.STAGE_LATE_START, severity=Severity.INFO,
                    title=f"Этап {fmt.stage(s)} начат с опозданием на {fmt.days(d)}",
                    message=(f"Видели: этап {fmt.stage(s)} идёт с {fmt.date(a_start)} (первый снимок с его признаками). "
                             f"По плану ожидается: начало {fmt.date(ps)}. "
                             "Что проверить: успевает ли этап к плановому окончанию — см. прогноз."),
                    stage_id=s, frame_ids=first_ev, data={"days": d, "planned_start": ps.isoformat(),
                                                          "actual_start": a_start.isoformat()},
                ))
        if pe:
            over = (today - pe).days
            if status != StageStatus.DONE and over > tol:
                prog = f", готовность {fmt.pct(st.progress)}" if st and status == StageStatus.ACTIVE else ""
                state = "идёт" if status == StageStatus.ACTIVE else "не начат"
                out.append(DeviationRecord(
                    key=f"stage_overdue:{s}", type=DT.STAGE_OVERDUE,
                    severity=Severity.CRITICAL if over > crit else Severity.WARNING,
                    title=f"Этап {fmt.stage(s)} не завершён в срок: просрочка {fmt.days(over)}",
                    message=(f"Видели: этап {state}{prog} (последний снимок {last_shot}). "
                             f"По плану ожидается: окончание {fmt.date(pe)} — {fmt.days(over)} назад. "
                             "Что проверить: причины задержки (техника, простои, объём), сдвиг следующих этапов; "
                             "если этап завершён — отметьте это вручную."),
                    stage_id=s, frame_ids=list(dict.fromkeys(last_ev + last_frames))[:3],
                    data={"days": over, "planned_end": pe.isoformat()},
                ))
            elif status == StageStatus.DONE and a_end and (a_end - pe).days > tol and (today - a_end).days <= 30:
                d = (a_end - pe).days
                out.append(DeviationRecord(
                    key=f"stage_overdue:{s}", type=DT.STAGE_OVERDUE, severity=Severity.INFO,
                    title=f"Этап {fmt.stage(s)} завершён с опозданием на {fmt.days(d)}",
                    message=(f"Видели: этап завершён {fmt.date(a_end)}. По плану ожидается: окончание {fmt.date(pe)}. "
                             "Что проверить: сдвинулись ли из-за этого следующие этапы."),
                    stage_id=s, frame_ids=last_ev, data={"days": d, "planned_end": pe.isoformat(),
                                                         "actual_end": a_end.isoformat()},
                ))
        if ps and a_start and (ps - a_start).days > tol and status == StageStatus.ACTIVE:
            d = (ps - a_start).days
            out.append(DeviationRecord(
                key=f"stage_early:{s}", type=DT.STAGE_EARLY, severity=Severity.INFO,
                title=f"Этап {fmt.stage(s)} начат раньше плана на {fmt.days(d)}",
                message=(f"Видели: признаки этапа с {fmt.date(a_start)} (первый снимок с ними). "
                         f"По плану ожидается: начало {fmt.date(ps)}. "
                         "Что проверить: не соседняя ли стройка в кадре; если опережение настоящее — сдвиньте план."),
                stage_id=s, frame_ids=first_ev, data={"days": d, "planned_start": ps.isoformat(),
                                                      "actual_start": a_start.isoformat()},
            ))
    if planned:
        first_planned = min(planned)
        for s, st in sorted(ctx.timeline.states.items()):
            if s in planned or st.status == StageStatus.NOT_STARTED:
                continue
            # пройденные до начала плана этапы (план начинается с котлована) — не аномалия
            if st.status == StageStatus.DONE and s < first_planned:
                continue
            evidence = list(dict.fromkeys(st.evidence_frame_ids[:1] + st.evidence_frame_ids[-1:]))
            # «Пройден» без единого кадра с признаками — этап закрыт по хронологии: фронт
            # перешагнул его (на участке под частный дом после расчистки сразу котлован,
            # шпунта нет). Это не работа вне плана, а пропущенный планом этап — молчим.
            if st.status == StageStatus.DONE and not evidence and not st.manual:
                continue
            going = st.status == StageStatus.ACTIVE
            out.append(DeviationRecord(
                key=f"stage_out_of_plan:{s}", type=DT.STAGE_OUT_OF_PLAN, severity=Severity.INFO,
                title=f"Этап {fmt.stage(s)} {'идёт' if going else 'пройден'}, но его нет в плане",
                message=(f"Видели: этап {fmt.stage(s)} {'идёт' if going else 'завершён'}"
                         f"{' с ' + fmt.date(st.actual_start) if st.actual_start else ''}. "
                         "По плану ожидается: этап не запланирован. "
                         "Что проверить: добавьте этап в план с датами или, если распознано ошибочно, отметьте вручную."),
                # без кадров с признаками этапа — последние годные снимки: отклонение без снимка не объяснить
                stage_id=s, frame_ids=evidence or last_frames,
                data={"status": st.status.value},
            ))
    return out


# --------------------------------------------------------------------------
# качество данных: NEEDS_REVIEW, CAMERA_ISSUE
# --------------------------------------------------------------------------


def needs_review(ctx: AnalyticsContext) -> list[DeviationRecord]:
    out = []
    ids = list(dict.fromkeys(ctx.timeline.needs_review))
    if ids:
        n = len(ids)
        out.append(DeviationRecord(
            key="needs_review:unsure", type=DT.NEEDS_REVIEW,
            severity=Severity.WARNING if n >= ctx.cfg.needs_review_warn else Severity.INFO,
            title=f"Проверьте вручную: модель Б не уверена на {fmt.count(n, fmt.FRAMES_LOC)}",
            message=(f"Видели: на {fmt.count(n, fmt.FRAMES_LOC)} больше половины ответов чек-листа — «затрудняюсь» "
                     "(ракурс, перекрытие, туман, далеко); такие кадры не голосуют за этап. "
                     "Ожидается: уверенный ответ хотя бы на половину вопросов, иначе этап по снимкам не подтвердить. "
                     "Что сделать: откройте кадры и отметьте готовность этапов вручную — ручную отметку модель не перезаписывает."),
            frame_ids=ids[-6:], data={"count": n},
        ))
    rej = list(dict.fromkeys(ctx.timeline.rejected_outliers))
    if rej:
        out.append(DeviationRecord(
            key="needs_review:outliers", type=DT.NEEDS_REVIEW, severity=Severity.INFO,
            title=f"{fmt.count(len(rej), fmt.FRAMES).capitalize()} противоречат хронологии этапов",
            message=("Видели: на этих кадрах этап «откатывается» назад или перескакивает вперёд — вероятно, в кадр "
                     "попала соседняя стройка или случайное перекрытие; в расчёт готовности они не вошли. "
                     "Ожидается: этапы идут только вперёд и не перескакивают через один. "
                     "Что сделать: проверьте кадры; если это реальный скачок — отметьте этап вручную."),
            frame_ids=rej[-6:], data={"count": len(rej)},
        ))
    return out


def _parse_dt(value) -> dt.datetime | None:
    if value is None or isinstance(value, dt.datetime):
        return value if value is None or value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    try:
        v = dt.datetime.fromisoformat(str(value))
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def camera_issue(ctx: AnalyticsContext) -> list[DeviationRecord]:
    cfg = ctx.cfg
    cams: dict[str, dict] = {}
    for c in cfg.cameras:
        cams[str(c.get("id"))] = {"id": c.get("id"), "kind": c.get("kind"), "last": _parse_dt(c.get("last_frame_at"))}
    by_cam: dict[str, list] = defaultdict(list)
    for f in ctx.frames:
        by_cam[str(f.camera_id)].append(f)
        entry = cams.setdefault(str(f.camera_id), {"id": f.camera_id, "kind": None, "last": None})
        if entry["last"] is None or f.captured_at > entry["last"]:
            entry["last"] = f.captured_at
    out = []
    for key, c in cams.items():
        name = ctx.camera_name(c["id"])
        # загрузки папкой/видео — не живой поток: «молчание» для них нормально
        live = c["kind"] not in ("upload", "folder", "video")
        if live and c["last"] is not None:
            silent_h = (ctx.now - c["last"]).total_seconds() / 3600
            if silent_h >= cfg.camera_silent_h:
                out.append(DeviationRecord(
                    key=f"camera_issue:{key}:silent", type=DT.CAMERA_ISSUE,
                    severity=Severity.CRITICAL if silent_h >= cfg.camera_silent_critical_h else Severity.WARNING,
                    title=f"Камера {name} молчит {fmt.hours(silent_h)}",
                    message=(f"Видели: последний снимок с камеры {name} — {fmt.moment(ctx.local(c['last']))}, "
                             f"после этого кадров нет {fmt.hours(silent_h)}. Ожидается: снимок каждые 20–30 минут. "
                             "Пока камера молчит, отклонения по технике в её зоне не выявляются. "
                             "Что проверить: питание, сеть, ключ приёма камеры."),
                    camera_id=c["id"], frame_ids=[f.frame_id for f in by_cam.get(key, [])[-1:]],
                    started_at=c["last"], last_seen_at=ctx.now, data={"silent_h": round(silent_h, 2)},
                ))
        elif live and c["kind"] == "stream" and c["last"] is None:
            out.append(DeviationRecord(
                key=f"camera_issue:{key}:silent", type=DT.CAMERA_ISSUE, severity=Severity.WARNING,
                title=f"Камера {name} не прислала ни одного кадра",
                message="Видели: кадров от камеры нет. Ожидается: снимок каждые 20–30 минут. "
                        "Что проверить: адрес приёма, ключ камеры, сеть.",
                camera_id=c["id"], data={},
            ))
        day = [f for f in by_cam.get(key, []) if ctx.now - f.captured_at <= dt.timedelta(hours=24)]
        bad = [f for f in day if not f.quality_ok]
        if len(day) >= cfg.camera_min_frames and len(bad) / len(day) >= cfg.camera_bad_share:
            reasons = Counter(f.reject_reason or "брак" for f in bad).most_common(3)
            why = ", ".join(f"{r} — {n}" for r, n in reasons)
            out.append(DeviationRecord(
                key=f"camera_issue:{key}:quality", type=DT.CAMERA_ISSUE, severity=Severity.WARNING,
                title=f"Камера {name}: много брака ({len(bad)} из {len(day)} кадров за сутки)",
                message=(f"Видели: {len(bad)} из {len(day)} кадров за сутки отбракованы ({why}). "
                         "Модели на таких кадрах не работают, отклонения в зоне камеры могут быть пропущены. "
                         "Что проверить: объектив (капли, грязь, паутина), фокус, засветку, не сдвинута ли камера."),
                camera_id=c["id"], frame_ids=_spread([f.frame_id for f in bad], 3),
                started_at=bad[0].captured_at, last_seen_at=bad[-1].captured_at,
                data={"bad": len(bad), "total": len(day)},
            ))
    return out


# --------------------------------------------------------------------------


def evaluate(ctx: AnalyticsContext) -> list[DeviationRecord]:
    """Все отклонения площадки: критичные сверху, внутри уровня — свежие первыми."""
    pairs = pair_broken(ctx)
    found = (pairs + equipment_missing(ctx, pairs) + equipment_forbidden(ctx) + equipment_idle(ctx)
             + equipment_parked_only(ctx) + outside_zone(ctx) + hours_spent_no_progress(ctx)
             + schedule(ctx) + needs_review(ctx) + camera_issue(ctx))
    by_key: dict[str, DeviationRecord] = {}
    for d in found:
        prev = by_key.get(d.key)
        if prev is None or _RANK[d.severity] > _RANK[prev.severity]:
            by_key[d.key] = d

    def order(d: DeviationRecord):
        t = d.last_seen_at or d.started_at
        return -_RANK[d.severity], -t.timestamp() if t else float("inf")

    return sorted(by_key.values(), key=order)
