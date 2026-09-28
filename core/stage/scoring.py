"""Ответы чек-листа → доказательность этапов, статусы подэтапов, фронт и готовность.

Порт `api-solution/core/scoring.py` Никиты. Идея сохранена: этапы здания идут по
порядку, «фронт» — самый поздний этап с достаточными признаками, всё до фронта
выполнено. Исправлено то, что нашёл разбор ветки (баги A5, A6):

1. **«Не уверен» не голосует.** Доказательность этапа — доля «да» среди
   проголосовавших признаков, а не среди всех. Раньше `not_visible` делил
   знаменатель наравне с «нет» и на фасадных кадрах (33–74 % «не видно»)
   этап 7 не открывался при облицовке «да».
2. **Нет потолков.** 14 подэтапов раньше не могли стать DONE (пустой done_when
   или ACTIVE с приоритетом над DONE), объект не бывал готов выше 99.5 %.
   Теперь DONE старше ACTIVE, а подэтап без done_when выполнен, когда начат
   более поздний подэтап того же этапа (или этап ниже фронта).
3. **Монотонность.** Больше «да» на положительные признаки никогда не снижает
   ни одну готовность: статус подэтапа только растёт (нет → идёт → готово),
   фронт только поднимается, «идущий» подэтап не получает меньше, чем
   «пропущенный при увиденном следующем». Раньше ACTIVE давал 0.5, а тот же
   подэтап без ответа — 1.0.
4. **Этап открывается только ответами.** Вероятность из разведки VLM сюда не
   входит вовсе: раньше этап вне кандидатов открывался одной вероятностью
   `stage_likelihood ≥ 0.5` без проверки чек-листом.

Штраф за противоречие (`must_not_have`) берётся только от чисто отрицательных
признаков (голый бетон у фасада, голый грунт у благоустройства). Признаки вроде
«здание выше земли» — `must_not_have` для котлована, но положительные для каркаса:
их «да» говорит за более поздний этап, и фронт поднимется сам. Штрафовать ими
ранний этап значило бы нарушить монотонность (фронт мог упасть с 3 до 2).

Доказательность с «скепсисом»: E = Σ w·b / (Σ w + c), где b — доля «да» по
признаку (один кадр — 0 или 1, день из нескольких кадров — дробь), w — вес
признака (must_have 1.0, отличительные признаки подэтапов 0.5), c = 0.5.
Одно «да» из одного голоса даёт 0.67, три из трёх — 0.86: единственный ответ
открывает этап, но не с полной уверенностью.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

from core import taxonomy
from core.contracts import Answer

DONE, ACTIVE, NOT_STARTED, UNKNOWN = "done", "active", "not_started", "unknown"
# Порядок статусов подэтапа: с ростом числа «да» статус может только расти.
STATUS_RANK = {UNKNOWN: 0, NOT_STARTED: 0, ACTIVE: 1, DONE: 2}

_YES = {"yes", "да", "true", "1", "y"}
_NO = {"no", "нет", "false", "0", "n"}

Votes = dict[str, tuple[float, float]]   # признак → (голосов «да», голосов «нет»)


@dataclass
class ScoringConfig:
    front_threshold: float = 0.5   # доказательность, с которой этап считается увиденным
    pseudo_no: float = 0.5         # c в E = Σw·b/(Σw + c): скепсис к единичному «да»
    distinct_weight: float = 0.5   # вес отличительных признаков подэтапов против must_have
    contra_weight: float = 0.5     # множитель штрафа за «да» на чисто отрицательный признак
    active_credit: float = 0.5     # вклад идущего подэтапа в готовность этапа

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "ScoringConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StageScores:
    stage_evidence: dict[int, float]      # этап → 0..1; только этапы, по признакам которых был голос
    substages: dict[str, str]             # «3.1» → done | active | not_started | unknown
    front: int | None                     # самый поздний этап с доказательностью ≥ порога
    progress: dict[int, float]            # этап → 0..1 при данном фронте
    overall: float = 0.0                  # 0..1 по весам этапов
    votes: dict[int, float] = field(default_factory=dict)  # взвешенное число голосов по признакам этапа


def normalize_answer(value: Any) -> Answer:
    """Любое представление ответа → Answer. «not_visible», «не видно», мусор → UNSURE."""
    if isinstance(value, Answer):
        return value
    if isinstance(value, bool):
        return Answer.YES if value else Answer.NO
    v = str(value).strip().lower()
    if v in _YES:
        return Answer.YES
    if v in _NO:
        return Answer.NO
    return Answer.UNSURE


def answers_to_votes(answers: Mapping[str, Any]) -> Votes:
    out: Votes = {}
    for k, v in answers.items():
        a = normalize_answer(v)
        if a is Answer.YES:
            out[k] = (1.0, 0.0)
        elif a is Answer.NO:
            out[k] = (0.0, 1.0)
    return out


def add_votes(total: Votes, more: Votes) -> Votes:
    for k, (y, n) in more.items():
        y0, n0 = total.get(k, (0.0, 0.0))
        total[k] = (y0 + y, n0 + n)
    return total


def belief(votes: Votes, key: str) -> float | None:
    """Доля «да» среди проголосовавших по признаку; None — никто не голосовал."""
    y, n = votes.get(key, (0.0, 0.0))
    return None if y + n <= 0 else y / (y + n)


def sign_state(votes: Votes, key: str) -> bool | None:
    """True — признак виден (большинство «да»), False — не виден, None — не решено (нет голосов или ничья)."""
    b = belief(votes, key)
    if b is None or b == 0.5:
        return None
    return b > 0.5


@dataclass(frozen=True)
class _Model:
    order: tuple[int, ...]
    weights: dict[int, float]
    must: dict[int, tuple[str, ...]]
    distinct: dict[int, frozenset[str]]
    contra: dict[int, tuple[str, ...]]
    substages: dict[int, tuple[dict, ...]]
    positive: frozenset[str]


@lru_cache(maxsize=1)
def model() -> _Model:
    """Структура справочника, нужная скорингу, — один раз на процесс."""
    stages = taxonomy.stages()
    order = tuple(sorted(stages))
    positive: dict[int, set[str]] = {}
    for sid in order:
        s = stages[sid]
        keys = set(s.must_have)
        for sub in s.substages:
            keys |= set(sub["active_when"]) | set(sub["done_when"])
        positive[sid] = keys
    all_positive = set().union(*positive.values())
    # Отличительные — положительные признаки этапа, которых нет у более ранних этапов.
    # Только они могут сдвинуть фронт вперёд: «идёт разработка грунта» есть и у
    # котлована (3), и у засыпки пазух (4.6) — замер Никиты на test_photos.
    distinct, earlier = {}, set()
    for sid in order:
        distinct[sid] = frozenset(positive[sid] - earlier)
        earlier |= positive[sid]
    contra = {sid: tuple(k for k in stages[sid].must_not_have if k not in all_positive) for sid in order}
    return _Model(
        order=order,
        weights={sid: stages[sid].weight for sid in order},
        must={sid: tuple(stages[sid].must_have) for sid in order},
        distinct=distinct,
        contra=contra,
        substages={sid: tuple(stages[sid].substages) for sid in order},
        positive=frozenset(all_positive),
    )


def stage_evidence(stage_id: int, votes: Votes, cfg: ScoringConfig | None = None) -> tuple[float | None, float]:
    """(доказательность этапа или None, взвешенное число голосов). UNSURE не голосует."""
    cfg = cfg or ScoringConfig()
    m = model()
    signs = [(k, 1.0) for k in m.must[stage_id]]
    signs += [(k, cfg.distinct_weight) for k in sorted(m.distinct[stage_id]) if k not in m.must[stage_id]]
    s = v = 0.0
    for k, w in signs:
        b = belief(votes, k)
        if b is not None:
            s += w * b
            v += w
    if v <= 0:
        return None, 0.0
    e = s / (v + cfg.pseudo_no)
    for k in m.contra[stage_id]:
        b = belief(votes, k)
        if b is not None:
            e *= 1.0 - cfg.contra_weight * b
    return e, v


def substage_status(sub: dict, votes: Votes) -> str:
    """done > active > not_started/unknown. DONE — все done_when видны; ACTIVE — виден любой active_when."""
    done_keys, act_keys = list(sub["done_when"]), list(sub["active_when"])
    if done_keys and all(sign_state(votes, k) is True for k in done_keys):
        return DONE
    if any(sign_state(votes, k) is True for k in act_keys):
        return ACTIVE
    if any(sign_state(votes, k) is False for k in act_keys + done_keys):
        return NOT_STARTED
    return UNKNOWN


def substage_statuses(votes: Votes) -> dict[str, str]:
    m = model()
    return {sub["id"]: substage_status(sub, votes) for sid in m.order for sub in m.substages[sid]}


def stage_progress_from_substages(stage_id: int, statuses: Mapping[str, str],
                                  cfg: ScoringConfig | None = None) -> float:
    """Готовность этапа по подэтапам: готовый = 1, идущий = active_credit, начатый более поздний
    подэтап того же этапа засчитывает все предыдущие целиком (они шли раньше)."""
    cfg = cfg or ScoringConfig()
    subs = model().substages[stage_id]
    st = [statuses.get(sub["id"], UNKNOWN) for sub in subs]
    weights = [float(sub.get("weight", 0) or 0) for sub in subs]
    if sum(weights) <= 0:
        weights = [1.0] * len(subs)
    total = 0.0
    for i, w in enumerate(weights):
        later_seen = any(s in (ACTIVE, DONE) for s in st[i + 1:])
        if st[i] == DONE or later_seen:
            credit = 1.0
        elif st[i] == ACTIVE:
            credit = cfg.active_credit
        else:
            credit = 0.0
        total += w * credit
    return total / sum(weights)


def evaluate_votes(votes: Votes, config: ScoringConfig | Mapping[str, Any] | None = None) -> StageScores:
    """То же, что evaluate(), но по счётчикам голосов — для дня из нескольких кадров и камер."""
    cfg = config if isinstance(config, ScoringConfig) else ScoringConfig.from_dict(config)
    m = model()
    evidence, vote_w = {}, {}
    for sid in m.order:
        e, v = stage_evidence(sid, votes, cfg)
        if e is not None:
            evidence[sid] = round(e, 4)
            vote_w[sid] = v
    front = max((sid for sid, e in evidence.items() if e >= cfg.front_threshold), default=None)
    statuses = substage_statuses(votes)
    progress = {}
    for sid in m.order:
        if front is None or sid > front:
            progress[sid] = 0.0
        elif sid < front:
            progress[sid] = 1.0
        else:
            progress[sid] = round(stage_progress_from_substages(sid, statuses, cfg), 4)
    total_w = sum(m.weights.values()) or 1.0
    overall = sum(m.weights[sid] * progress[sid] for sid in m.order) / total_w
    return StageScores(stage_evidence=evidence, substages=statuses, front=front,
                       progress=progress, overall=round(overall, 4), votes=vote_w)


def evaluate(answers: Mapping[str, Any], config: ScoringConfig | Mapping[str, Any] | None = None) -> StageScores:
    """Ответы одного кадра → StageScores. Отсутствующий ключ и UNSURE равнозначны: голоса нет."""
    return evaluate_votes(answers_to_votes(answers), config)
