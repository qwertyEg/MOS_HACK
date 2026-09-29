"""Монотонная хронология этапов: наблюдения модели Б → StageTimeline.

Модель Б отвечает про один кадр и только про то, что видит. Весь вывод об этапах
делается здесь, из накопленной истории. Две ошибки наследия, которые этот модуль
закрывает:

- у Никиты прогресс — накопленный максимум по кадрам, и **один ложный кадр
  навсегда** поднимал фронт (проба: 15.8 % → 95.2 %, баг A4);
- у Дениса «пройден» защёлкивался от одного дневного «да», интервалы
  склеивались через пропуски, этапы не упорядочены (баги D3, D10).

Как устроено:

1. **Агрегация по дням.** Несколько камер и кадров за сутки (по московскому
   времени) складываются в счётчики голосов «да/нет» по признакам. «Не уверен»
   не голосует. Сутки — естественная единица: этап за час не меняется, а кадры
   одного дня с разных ракурсов дополняют друг друга.
2. **HMM по «фронту» 1..8.** Скрытое состояние — старший начатый этап. Переходы
   между соседними днями наблюдений: остаться, +1, редко +2; откат запрещён.
   Цена перехода зависит от пропуска g дней: шансы k переходов против нуля по
   Пуассону с λ = g / stage_days (k·ln λ − ln k!), но не выше нуля — априори
   переход разрешается, но никогда не поощряется: двигать фронт должны ответы.
   Так за одну ночь +1 стоит ≈ −3.8 (нужны несколько дней подтверждений), +2 —
   вдвое дороже, а после месяцев без кадров +2..+3 почти бесплатны. «Остаться»
   бесплатно всегда: иначе при редких кадрах вероятность утекала в поглощающий
   этап 8 (замер на кассете с реальными ответами GLM по снимкам раз в 2–4 месяца:
   классическая матрица переходов в степени g ставила фронт 8 с первого кадра).
   Эмиссия — из доказательности этапов (scoring): «за» — доказательность самого
   этапа-фронта, «против» — доказательность более поздних этапов; ранние этапы
   нейтральны (котлован ещё виден, когда начат монолит). Путь — Витерби.
3. **Выбросы.** Кадр, чей собственный фронт отстаёт от пути на ≥ 2 этапа
   (попытка отката, соседняя стройка на раннем этапе) или опережает путь, а
   путь туда так и не пришёл за 14 дней (галлюцинация, соседний готовый дом), —
   в `rejected_outliers`. Путь пересчитывается без них.
4. **«Идёт сейчас» и «пройдено» — разные вопросы** (формулировка Дениса из
   aggregate.py). HMM видит только текущие ответы, без памяти. Latching-признаки
   (плита залита, окна стоят) защёлкиваются для статусов подэтапов, но только
   после подтверждения ≥ 2 разными днями: единичная галлюцинация не живёт вечно.
5. **Готовность этапа** — по подэтапам, накопленным максимумом по дням ПОСЛЕ
   фильтрации выбросов, причём новый уровень засчитывается, только когда
   подтверждён вторым днём (`progress_confirm_days`). Этапы ниже фронта пути — 1.0.
6. **Ручные отметки** имеют приоритет: их состояние отдаётся как есть
   (manual=True), а даты служат якорями пути (фронт ≥ этапа с даты начала,
   < этапа до неё; «не начат» — фронт ниже этапа на всём отрезке).
7. **needs_review** — кадры, где доля «не уверен» выше порога (0.5): много
   «затрудняюсь» → предупреждение «проверьте вручную».
8. **Техника (модель А)** — второе слагаемое эмиссии дня (core/stage/fusion.py): какая техника
   работает или стоит в эти сутки и что это значит по нормам «этап → техника» (каток —
   благоустройство, копёр — сваи, башенный кран с бетоном — монолит). Вес `equipment_weight`
   (0 — только чек-лист). Выбросы ищутся по хронологии одного чек-листа: иначе техника,
   удержав путь, объявила бы выбросами все кадры модели Б и лишила её голоса. Этап без
   ответов модели Б по одной технике не назначается. Почему этап такой — `StageTimeline.basis`.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from core import taxonomy
from core.contracts import StageObservation, StageState, StageStatus, StageTimeline
from core.stage import fusion, scoring
from core.stage.fusion import EquipmentDay, EquipmentEvidence, FusionConfig
from core.stage.scoring import ScoringConfig, Votes


@dataclass
class SequenceConfig:
    needs_review_ratio: float = 0.5     # доля «не уверен» в кадре, выше которой — проверить вручную
    exclude_review_frames: bool = False  # не учитывать такие кадры в хронологии вовсе
    stage_days: float = 45.0            # характерная длительность этапа, сутки (цена перехода за ночь ≈ ln(1/45))
    support_floor: float = 0.15         # p0: эмиссия «за» при нулевой доказательности
    contra_floor: float = 0.1           # p1: эмиссия «против» при полной доказательности позднего этапа
    unknown_support: float = -0.3       # о фронте-кандидате в этот день не спрашивали: слабое «против»
    term_clip: float = 3.0              # ограничение вклада одного этапа в эмиссию дня
    latch_min_days: int = 2             # latching-признак защёлкивается со второго дня наблюдения
    progress_confirm_days: int = 2      # новый уровень готовности — со второго дня
    outlier_behind: int = 2             # кадр отстаёт от пути на столько этапов — выброс
    outlier_ahead: int = 1              # опережает на столько, а путь не пришёл за lookahead — выброс
    outlier_lookahead_days: int = 14
    tz_offset_hours: float = 3.0        # границы суток — по Москве (UTC+3, без перехода на летнее)
    equipment_weight: float = 1.0       # вес довода техники (модель А) в эмиссии; 0 — только чек-лист
    scoring: dict = field(default_factory=dict)
    fusion: dict = field(default_factory=dict)   # тонкие параметры FusionConfig (вероятности норм, силы доводов)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SequenceConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------
# вспомогательное
# --------------------------------------------------------------------------


def local_day(when: dt.datetime, tz_offset_hours: float = 3.0) -> dt.date:
    """Календарный день по местному времени площадки. Наивное время считаем UTC (контракт)."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return (when.astimezone(dt.timezone.utc) + dt.timedelta(hours=tz_offset_hours)).date()


def _log_transition(n: int, gap_days: int, stage_days: float) -> np.ndarray:
    """log-«шансы» перехода i → j за пропуск в gap_days: 0 — остаться, −inf — откат и слишком большой скачок."""
    g = max(1, int(gap_days))
    lam = g / max(stage_days, 1e-6)
    k_max = 2 + g // 30                   # между соседними днями — не дальше +2
    out = np.full((n, n), -np.inf)
    for i in range(n):
        out[i, i] = 0.0
        for k in range(1, min(k_max, n - 1 - i) + 1):
            out[i, i + k] = min(0.0, k * math.log(lam) - math.lgamma(k + 1))
    return out


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


def _emission(votes: Votes, order: tuple[int, ...], cfg: SequenceConfig, scfg: ScoringConfig) -> np.ndarray:
    """log-эмиссия дня для каждого фронта. Нули — «день ничего не говорит».

    Фронт, о признаках которого в этот день не спрашивали вовсе, получает слабое
    «против» (unknown_support): отсутствие свидетельств — слабое свидетельство
    отсутствия. Без этого внешняя VLM, которая спрашивает только 2–3 этапа-кандидата,
    оставляла поздние этапы «нейтральными», и при редких кадрах (месяцы между ними)
    путь уезжал в поглощающий этап 8 — замер на кассете с реальными ответами GLM.
    """
    sc = scoring.evaluate_votes(votes, scfg)
    em = np.zeros(len(order))
    if not sc.stage_evidence:
        return em
    p0, p1, clip = cfg.support_floor, cfg.contra_floor, cfg.term_clip
    for fi, f in enumerate(order):
        total = 0.0 if f in sc.stage_evidence else cfg.unknown_support
        for s, e in sc.stage_evidence.items():
            v = sc.votes.get(s, 0.0)
            conf = v / (v + 1.0)       # мало голосов — слабое слово
            if s == f:
                term = math.log((p0 + (1 - p0) * e) / 0.5)
            elif s > f:
                term = math.log(max(1e-9, 1 - (1 - p1) * e))
            else:
                continue               # ранние этапы нейтральны: котлован виден и при монолите
            total += max(-clip, min(clip, conf * term))
        em[fi] = total
    return em


def _viterbi(em: np.ndarray, allowed: np.ndarray, log_trans: list[np.ndarray]) -> list[int]:
    t_len, n = em.shape
    mask = np.where(allowed, 0.0, -np.inf)
    delta = em[0] + mask[0] - math.log(n)
    psi = np.zeros((t_len, n), dtype=int)
    for t in range(1, t_len):
        cand = delta[:, None] + log_trans[t]
        psi[t] = np.argmax(cand, axis=0)
        delta = cand[psi[t], np.arange(n)] + em[t] + mask[t]
    path = [int(np.argmax(delta))]
    for t in range(t_len - 1, 0, -1):
        path.append(int(psi[t][path[-1]]))
    return path[::-1]


def _last_marginal(em: np.ndarray, allowed: np.ndarray, log_trans: list[np.ndarray]) -> np.ndarray:
    """P(фронт в последний день | все наблюдения) — прямой проход; это и есть уверенность."""
    t_len, n = em.shape
    mask = np.where(allowed, 0.0, -np.inf)
    alpha = em[0] + mask[0] - math.log(n)
    for t in range(1, t_len):
        alpha = _logsumexp(alpha[:, None] + log_trans[t], axis=0) + em[t] + mask[t]
    z = _logsumexp(alpha, axis=0)
    return np.exp(alpha - z) if np.isfinite(z) else np.full(n, 1.0 / n)


def _allowed(days: list[dt.date], order: tuple[int, ...],
             manual: Mapping[int, StageState] | None) -> np.ndarray:
    """Якоря ручных отметок: какие фронты допустимы в какой день."""
    allowed = np.ones((len(days), len(order)), dtype=bool)
    if not manual or not days:
        return allowed
    idx = {s: i for i, s in enumerate(order)}
    for sid, st in manual.items():
        if int(sid) not in idx:
            continue
        si = idx[int(sid)]
        status = StageStatus(st.status)
        if status is StageStatus.NOT_STARTED:
            allowed[:, si:] = False          # на всём наблюдаемом отрезке фронт ниже этапа
            continue
        start, end = st.actual_start, st.actual_end
        anchor = start or (end if status is StageStatus.DONE else None)
        for di, d in enumerate(days):
            if anchor is None:
                if di == len(days) - 1:      # без дат — «на сегодня этап начат»
                    allowed[di, :si] = False
            elif d >= anchor:
                allowed[di, :si] = False
            elif start is not None and d < start:
                allowed[di, si:] = False
    # Противоречивые отметки не должны ронять вывод: такой день остаётся без якоря.
    empty = ~allowed.any(axis=1)
    allowed[empty] = True
    return allowed


# --------------------------------------------------------------------------
# основной вывод
# --------------------------------------------------------------------------


def _equipment_days(equipment: EquipmentEvidence | Mapping[dt.date, EquipmentDay] | None,
                    cfg: SequenceConfig, fcfg: FusionConfig) -> dict[dt.date, EquipmentDay]:
    if equipment is None or fcfg.equipment_weight <= 0:
        return {}
    if isinstance(equipment, EquipmentEvidence):
        return fusion.days(equipment, cfg.tz_offset_hours, fcfg)
    return {d: v for d, v in dict(equipment).items() if isinstance(v, EquipmentDay)}


def infer(observations: list[StageObservation], manual: dict[int, StageState] | None = None,
          config: dict | None = None,
          equipment: EquipmentEvidence | Mapping[dt.date, EquipmentDay] | None = None) -> StageTimeline:
    """equipment — техника площадки от модели А (журнал моточасов и рамки) или уже разложенная
    по суткам; None — этап только по чек-листу, как до слияния моделей."""
    cfg = config if isinstance(config, SequenceConfig) else SequenceConfig.from_dict(config)
    scfg = ScoringConfig.from_dict(cfg.scoring)
    fcfg = FusionConfig.from_dict({**cfg.fusion, "equipment_weight": cfg.equipment_weight})
    m = scoring.model()
    order = m.order
    n = len(order)
    manual = {int(k): v for k, v in (manual or {}).items()}

    obs = sorted(observations, key=lambda o: o.captured_at)
    needs_review = [o.frame_id for o in obs if o.result.unsure_ratio > cfg.needs_review_ratio]
    review_set = set(map(str, needs_review))
    used = [o for o in obs if not (cfg.exclude_review_frames and str(o.frame_id) in review_set)]

    frame_day = [local_day(o.captured_at, cfg.tz_offset_hours) for o in used]
    frame_votes = [scoring.answers_to_votes(o.result.answers) for o in used]
    frame_front = [scoring.evaluate_votes(v, scfg).front for v in frame_votes]
    obs_days = sorted(set(frame_day))

    if not obs_days:
        return _timeline_without_observations(manual, needs_review)

    # Сутки с техникой, но без ответов модели Б (ночь, дождь), тоже двигают путь — но только внутри
    # отрезка наблюдений модели Б: за его пределами техника экстраполировала бы этап в одиночку.
    eq_days = {d: v for d, v in _equipment_days(equipment, cfg, fcfg).items() if obs_days[0] <= d <= obs_days[-1]}
    days = sorted(set(obs_days) | set(eq_days))
    day_index = {d: i for i, d in enumerate(days)}

    allowed = _allowed(days, order, manual)
    log_trans = [np.zeros((n, n))] + [_log_transition(n, (days[t] - days[t - 1]).days, cfg.stage_days)
                                      for t in range(1, len(days))]
    eq_em = np.array([fusion.emission(eq_days.get(d), order, fcfg) for d in days]) if eq_days \
        else np.zeros((len(days), n))

    def day_votes(include: list[bool]) -> list[Votes]:
        out: list[Votes] = [{} for _ in days]
        for i, v in enumerate(frame_votes):
            if include[i]:
                scoring.add_votes(out[day_index[frame_day[i]]], v)
        return out

    def run(include: list[bool], with_equipment: bool) -> tuple[list[int], np.ndarray, np.ndarray, list[Votes]]:
        dv = day_votes(include)
        em_b = np.stack([_emission(v, order, cfg, scfg) for v in dv])
        em = em_b + eq_em if with_equipment else em_b
        return _viterbi(em, allowed, log_trans), em, em_b, dv

    # Первый проход — по всем кадрам и только по чек-листу; по нему ищем выбросы.
    include = [True] * len(used)
    path_i, _, _, _ = run(include, False)
    path = [order[i] for i in path_i]
    outliers: list[int] = []
    for i, ff in enumerate(frame_front):
        if ff is None:
            continue
        di = day_index[frame_day[i]]
        pf = path[di]
        if ff <= pf - cfg.outlier_behind:
            outliers.append(i)
        elif ff >= pf + cfg.outlier_ahead:
            horizon = frame_day[i] + dt.timedelta(days=cfg.outlier_lookahead_days)
            reached = max(path[j] for j, d in enumerate(days) if frame_day[i] <= d <= horizon)
            if reached < ff:
                outliers.append(i)
    for i in outliers:
        include[i] = False

    # Второй проход — без выбросов: окончательный путь по чек-листу и технике вместе.
    # Путь одного чек-листа нужен объяснению: «техника решила» или «техника подтвердила».
    path_bi, _, _, _ = run(include, False)
    path_i, em, em_b, dv = run(include, True)
    if not em_b.any():
        # Ни одного решённого ответа (всё «не уверен» или сбои): этап не определён,
        # а не «подготовка территории» по равенству шансов. Одна техника этап не назначает.
        return _timeline_without_observations(manual, needs_review)
    path = [order[i] for i in path_i]
    marginal = _last_marginal(em, allowed, log_trans)

    latched = _latched_votes(dv, days, cfg)
    progress, done_day = _progress(latched, days, path, order, cfg, scfg)
    states = _states(days, path, progress, done_day, marginal, order, used, frame_day, frame_front,
                     include, day_index)
    _add_equipment_evidence(states, days, path, eq_days)
    basis = _basis(days, path, [order[i] for i in path_bi], dv, eq_days, used, frame_day, frame_front,
                   include, order, fcfg)

    for sid, st in manual.items():
        if sid in states:
            states[sid] = StageState(
                stage_id=sid, status=StageStatus(st.status), progress=float(st.progress),
                actual_start=st.actual_start, actual_end=st.actual_end, confidence=1.0, manual=True,
                evidence_frame_ids=list(st.evidence_frame_ids) or states[sid].evidence_frame_ids,
            )

    weights = m.weights
    total_w = sum(weights.values()) or 1.0
    overall = sum(weights[s] * states[s].progress for s in order) / total_w
    return StageTimeline(
        states=states,
        current_stage=path[-1],
        overall_progress=round(overall, 4),
        daily_front=list(zip(days, path)),
        needs_review=needs_review,
        rejected_outliers=[used[i].frame_id for i in outliers],
        basis=basis,
    )


def _add_equipment_evidence(states: dict[int, StageState], days: list[dt.date], path: list[int],
                            eq_days: Mapping[dt.date, EquipmentDay]) -> None:
    """Снимки-доказательства этапа — и кадры с его техникой: этап, который выдала техника
    (благоустройство по катку), иначе остался бы без снимка, а отклонения требуют снимок."""
    if not eq_days:
        return
    for s, st in states.items():
        if st.status is StageStatus.NOT_STARTED or len(st.evidence_frame_ids) >= 5:
            continue
        extra: list = []
        for d, f in zip(reversed(days), reversed(path)):
            if f != s or d not in eq_days:
                continue
            day = eq_days[d]
            for cls in sorted(day.frames, key=lambda k: -day.working_h.get(k, 0.0)):
                if fusion.category(s, cls) != "expected":
                    continue
                extra.extend(x for x in day.frames[cls][:2] if x not in extra and x not in st.evidence_frame_ids)
            if len(extra) >= 5:
                break
        st.evidence_frame_ids = (list(st.evidence_frame_ids) + extra)[:5]


def _basis(days, path, path_b, dv, eq_days, used, frame_day, frame_front, include, order,
           fcfg: FusionConfig) -> dict:
    """Почему фронт такой: признаки чек-листа и техника последних дней на текущем этапе.

    → {"stage", "text", "decided_by": checklist | equipment | both, "checklist_stage", "equipment_stages",
       "equipment": [...], "signs": [...]}. Окно — последние `basis_days` суток на текущем этапе.
    """
    final = path[-1]
    # Окно — последние дни на текущем этапе, где было что видеть (решённые кадры модели Б или
    # техника): сетка перед объективом в последние недели не должна обнулять объяснение.
    decided_days = {frame_day[i] for i in range(len(used)) if include[i] and frame_front[i] is not None}
    on_stage = [i for i in range(len(days)) if path[i] == final]
    informative = [i for i in on_stage if days[i] in decided_days or days[i] in eq_days]
    window = (informative or on_stage)[-max(1, int(fcfg.basis_days)):]
    # чек-лист: этап по хронологии одного чек-листа (None — ни на одном кадре чек-лист этап не выдал)
    # и признаки текущего этапа, видимые в окне
    checklist_stage = path_b[-1] if decided_days else None
    votes: Votes = {}
    for i in window:
        scoring.add_votes(votes, dv[i])
    m = scoring.model()
    own = list(m.must[final]) + sorted(m.distinct[final] - set(m.must[final]))
    signs = [k for k in own if scoring.sign_state(votes, k) is True]
    # техника: средний за сутки окна довод по этапам и как он соотносится с итоговым этапом
    wdays_eq = [eq_days[days[i]] for i in window if days[i] in eq_days]
    rows = fusion.summarize(wdays_eq, fcfg)
    avg = {s: 0.0 for s in order}
    for day in wdays_eq:
        for s, v in fusion.stage_scores(day, order, fcfg).items():
            avg[s] += v / len(wdays_eq)
    rel, eq_stages = fusion.relation(final, avg, rows) if wdays_eq else ("neutral", [])
    text = fusion.describe(final, rows, rel, eq_stages, checklist_stage, signs)
    decided_by = ("equipment" if rel == "agree" and checklist_stage != final
                  else "both" if rel == "agree" else "checklist")
    return {"stage": final, "text": text, "decided_by": decided_by, "equipment_relation": rel,
            "checklist_stage": checklist_stage, "equipment_stages": eq_stages, "equipment": rows,
            "signs": signs[:5], "window": [days[window[0]].isoformat(), days[window[-1]].isoformat()]}


def _latched_votes(day_votes: list[Votes], days: list[dt.date], cfg: SequenceConfig) -> list[Votes]:
    """Голоса дня для статусов подэтапов с учётом защёлкивания.

    Latching-признак, увиденный меньше чем в latch_min_days разных днях, не считается
    вовсе — одна галлюцинация не должна поднимать готовность. Подтверждённый — виден
    с первого дня, когда его увидели (вывод пересчитывается по всей истории, так что
    дата завершения не сдвигается на день подтверждения), и каждый следующий день,
    даже если его закрыло здание.
    """
    latching = {k for k, s in taxonomy.signs().items() if s.latching}
    seen: dict[str, list[int]] = {}
    for i, votes in enumerate(day_votes):
        for k in latching:
            if scoring.sign_state(votes, k) is True:
                seen.setdefault(k, []).append(i)
    first_on = {k: idx[0] for k, idx in seen.items() if len(idx) >= cfg.latch_min_days}
    out: list[Votes] = []
    for i, votes in enumerate(day_votes):
        cur = dict(votes)
        for k in latching:
            if k in first_on and i >= first_on[k]:
                cur[k] = (1.0, 0.0)
            else:
                cur.pop(k, None)
        out.append(cur)
    return out


def _progress(latched: list[Votes], days: list[dt.date], path: list[int], order: tuple[int, ...],
              cfg: SequenceConfig, scfg: ScoringConfig) -> tuple[dict[int, float], dict[int, dt.date]]:
    """Накопленная готовность этапов: ниже фронта пути — 1.0, на фронте — по подэтапам.

    Уровень готовности засчитывается, только когда достигнут хотя бы в
    progress_confirm_days разных днях: единичный всплеск (ложное «идёт укладка
    асфальта») не поднимает накопленный максимум навсегда — баг A4 Никиты.
    Дата готовности — первый из подтвердивших дней.
    """
    confirm = max(1, int(cfg.progress_confirm_days))
    history: dict[int, list[tuple[float, dt.date]]] = {s: [] for s in order}
    progress = {s: 0.0 for s in order}
    done_day: dict[int, dt.date] = {}
    for d, votes, f in zip(days, latched, path):
        statuses = scoring.substage_statuses(votes)
        for s in order:
            if s < f:
                value, first = 1.0, d
            elif s == f:
                history[s].append((scoring.stage_progress_from_substages(s, statuses, scfg), d))
                ranked = sorted(history[s], key=lambda x: (-x[0], x[1]))
                if len(ranked) < confirm:
                    continue
                value = ranked[confirm - 1][0]
                first = min(day_ for v, day_ in ranked if v >= value)
            else:
                continue
            progress[s] = max(progress[s], value)
            if progress[s] >= 0.999 and s not in done_day:
                done_day[s] = first
    return progress, done_day


def _states(days, path, progress, done_day, marginal, order, used, frame_day, frame_front,
            include, day_index) -> dict[int, StageState]:
    final = path[-1]
    idx = {s: i for i, s in enumerate(order)}
    states: dict[int, StageState] = {}
    for s in order:
        if s < final or (s == final and progress[s] >= 0.999):
            status = StageStatus.DONE
        elif s == final:
            status = StageStatus.ACTIVE
        else:
            status = StageStatus.NOT_STARTED

        start = end = None
        if status is not StageStatus.NOT_STARTED:
            first = next(i for i, f in enumerate(path) if f >= s)
            # Этап, уже начатый к первому кадру, начался до наблюдений: дату честнее не выдумывать,
            # иначе план/факт насчитает ложное «позднее начало».
            start = days[first] if first > 0 else None
        if status is StageStatus.DONE:
            later = next((i for i, f in enumerate(path) if f > s), None)
            candidates = [d for d in (days[later] if later is not None else None, done_day.get(s)) if d]
            if path[0] > s:
                end = None                   # завершён до первого кадра
            elif candidates:
                end = min(candidates)

        if status is StageStatus.DONE:
            conf = float(marginal[idx[s] + 1:].sum()) if s < final else float(marginal[idx[s]:].sum())
        elif status is StageStatus.ACTIVE:
            conf = float(marginal[idx[s]])
        else:
            conf = float(marginal[:idx[s]].sum())

        ev = [used[i].frame_id for i in range(len(used))
              if include[i] and path[day_index[frame_day[i]]] == s and frame_front[i] == s]
        states[s] = StageState(
            stage_id=s, status=status,
            progress=1.0 if status is StageStatus.DONE else round(progress[s], 4),
            actual_start=start, actual_end=end, confidence=round(conf, 4),
            evidence_frame_ids=ev[-5:][::-1],
        )
    return states


def _timeline_without_observations(manual: dict[int, StageState], needs_review: list) -> StageTimeline:
    """Нет пригодных наблюдений: всё «не начато», кроме ручных отметок."""
    m = scoring.model()
    states = {s: StageState(stage_id=s, status=StageStatus.NOT_STARTED, progress=0.0) for s in m.order}
    for sid, st in manual.items():
        if sid in states:
            states[sid] = StageState(stage_id=sid, status=StageStatus(st.status), progress=float(st.progress),
                                     actual_start=st.actual_start, actual_end=st.actual_end, confidence=1.0,
                                     manual=True, evidence_frame_ids=list(st.evidence_frame_ids))
    started = [s for s, st in states.items() if st.status is not StageStatus.NOT_STARTED]
    total_w = sum(m.weights.values()) or 1.0
    overall = sum(m.weights[s] * states[s].progress for s in m.order) / total_w
    return StageTimeline(states=states, current_stage=max(started) if started else None,
                         overall_progress=round(overall, 4), daily_front=[],
                         needs_review=needs_review, rejected_outliers=[])
