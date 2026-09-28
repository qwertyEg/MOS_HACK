"""Тексты запросов к внешней VLM (GLM-4.6V) — порт `api-solution/core/prompts.py` Никиты.

Вопросы, подсказки и описания техники берутся из `reference/checklist.json`
(через `core.taxonomy.checklist()`) — здесь только обвязка и схема ответа.
Системный промпт общий для всех шагов по кадру: вместе с картинкой он образует
неизменный префикс, который попадает в кэш z.ai.

Тексты сохранены побайтно: по ним считается ключ кассеты с записанными ответами
GLM (`tests/fixtures/glm_cassette.json`), и любая правка буквы делает записи
недействительными. Правите промпт — перезапишите кассету и поднимите PROMPT_VERSION.
"""
from __future__ import annotations

from core import taxonomy

# Меняется вручную при правке промптов — старые ответы из кэша перестают подходить.
PROMPT_VERSION = "p3"

SYSTEM = """Ты — инженер строительного контроля. Анализируешь снимок камеры видеонаблюдения со строительной площадки здания.

Правила:
- Описывай только то, что действительно видно на снимке. Не додумывай по контексту.
- Если признак нельзя проверить (перекрыт, слишком далеко, темно, эта часть площадки вне кадра) — ответ "not_visible".
- "no" — только когда нужная часть площадки хорошо видна и признака там точно нет. Прежде чем ответить "no", проверь, попало ли это место в кадр.
- Легковые автомобили, припаркованный частный транспорт и технику за пределами стройплощадки не учитывай.
- Отвечай строго одним JSON-объектом по схеме из запроса, без текста до или после него."""

# Добавляется к запросу (после неизменной части), только если фон на кадре погашен маской.
# Денис отметил: модели не сообщали, что затемнённое — это фон, и она пыталась его разбирать.
MASK_NOTE = ("Затемнённые области кадра — фон и соседние участки за пределами нашей стройки: "
             "их не оценивай, признаки ищи только в незатемнённой части.")

MEASUREMENT_KEYS = ["workers_count", "floors_built", "floors_glazed", "facade_clad_pct", "pit_area_pct"]


def stage_sign_keys(stage_id: int, checklist: dict | None = None) -> list[str]:
    """Все признаки, нужные для разбора этапа: этапные и подэтапные, без повторов, в порядке справочника."""
    ck = checklist or taxonomy.checklist()
    stage = next(s for s in ck["stages"] if int(s["id"]) == int(stage_id))
    keys = list(stage["must_have"]) + list(stage["must_not_have"])
    for sub in stage["substages"]:
        keys += list(sub["active_when"]) + list(sub["done_when"])
    return list(dict.fromkeys(keys))


def sign_keys_for(stage_ids, checklist: dict | None = None) -> list[str]:
    keys: list[str] = []
    for sid in stage_ids:
        keys += stage_sign_keys(sid, checklist)
    return list(dict.fromkeys(keys))


def triage(checklist: dict | None = None) -> str:
    ck = checklist or taxonomy.checklist()
    eq_lines = "\n".join(f"- {e['key']}: {e['name']} — {e['look']}" for e in ck["equipment"])
    stage_lines = "\n".join(f"{s['id']}. {s['name']}: {s['summary']}" for s in ck["stages"])
    measures = {m["key"]: m for m in ck["frame_measurements"]}
    measure_lines = "\n".join(f"- {k}: {measures[k]['question']}" for k in MEASUREMENT_KEYS)
    return f"""Шаг 1. Общий разбор кадра.

Справочник строительной техники (ключ: название — как выглядит):
{eq_lines}

Этапы строительства здания:
{stage_lines}

Числовые измерения (null, если нельзя определить):
{measure_lines}

Верни JSON строго такого вида:
{{
  "quality": "good" | "night" | "fog_rain" | "obstructed" | "blurred",
  "view": "top" | "side" | "ground" | "unknown",
  "description": "1–2 предложения: что происходит на площадке",
  "equipment": [
    {{"type": "<ключ из справочника или other>", "total": <сколько единиц видно>, "working": <сколько из них работает>, "evidence": "<коротко: по какому признаку определил работу или простой>"}}
  ],
  "workers_count": <int или null>,
  "floors_built": <int или null>,
  "floors_glazed": <int или null>,
  "facade_clad_pct": <0–100 или null>,
  "pit_area_pct": <0–100 или null>,
  "latest_stage": <1–8 или null>,
  "context_conflict": <null или коротко: что на снимке противоречит контексту стройки>,
  "stage_likelihood": {{"1": <0–1>, "2": <0–1>, "3": <0–1>, "4": <0–1>, "5": <0–1>, "6": <0–1>, "7": <0–1>, "8": <0–1>}}
}}

Пояснения:
- equipment: одна запись на тип техники; пустой список, если техники нет. Тип не из справочника — "other", что это — в evidence.
- working: техника работает, если это видно на снимке — ковш в грунте или над кузовом, поднятый кузов, развёрнутая стрела бетононасоса, груз на крюке крана, люди у машины за работой. Стоящая без признаков работы — не working.
- view: top — снято с высоты (видно крышу, дно котлована или площадку сверху), side — видно фасад здания сбоку, ground — снято с земли вблизи.
- latest_stage: самый поздний этап, работы или готовый результат которого видны на снимке. Готовое здание с облицованным фасадом — это 7 (или 8, если видно благоустройство), даже если стройки вокруг не видно. null — только если на снимке нет ни стройки, ни строящегося здания.
- stage_likelihood: насколько вероятно, что работы этапа идут на площадке сейчас. Несколько этапов могут идти одновременно (например, 5 и 7)."""


def checklist_keys_step(keys: list[str], checklist: dict | None = None) -> str:
    """Шаг 2 для произвольного набора признаков (кандидатные этапы или явный список ключей)."""
    ck = checklist or taxonomy.checklist()
    signs = {s["key"]: s for s in ck["signs"]}
    lines = "\n".join(f"- {k}: {signs[k]['question']} (как выглядит: {signs[k]['hint']})" for k in keys)
    template = ", ".join(f'"{k}": "yes|no|not_visible"' for k in keys)
    return f"""Шаг 2. Чек-лист этапа. На каждый вопрос ответь "yes", "no" или "not_visible".
"not_visible" — нужное место не попало в кадр, закрыто или слишком мелкое. "no" — место хорошо видно, и признака там нет.

{lines}

Верни JSON строго такого вида:
{{"answers": {{{template}}}, "comment": "<до 25 слов: главное, что подтверждает или опровергает этап>"}}"""


def checklist_step(stage_ids, checklist: dict | None = None) -> str:
    return checklist_keys_step(sign_keys_for(stage_ids, checklist), checklist)


# --------------------------------------------------------------------------
# локальная VLM (порт app/pipeline/model_b.py Дениса)
# --------------------------------------------------------------------------

# Слова ответа выбраны под инструкцию ниже: «не видно» модель исполняет сильно лучше,
# чем «не уверена» (замер Дениса: конкретная инструкция исполняется лучше размытой).
LOCAL_CHOICES = ("да", "нет", "не видно")

LOCAL_SYSTEM = (
    "Ты отвечаешь на вопросы строго по тому, что ВИДНО на снимке "
    "строительной площадки.\n"
    "«да» — признак различим на снимке.\n"
    "«нет» — область видна, но признака в ней нет.\n"
    "«не видно» — область не попала в кадр, перекрыта или неразличима.\n"
    "Не догадывайся и не достраивай сцену по смыслу: если по этому снимку "
    "судить нельзя — отвечай «не видно». Отвечать «нет» про то, чего не "
    "видно в кадре, — ошибка."
)


def local_questions(keys: list[str], checklist: dict | None = None) -> str:
    ck = checklist or taxonomy.checklist()
    signs = {s["key"]: s for s in ck["signs"]}
    body = "\n".join(f"{k}: {signs[k]['question']} (как выглядит: {signs[k]['hint']})" for k in keys)
    return ("Ответь на каждый вопрос одним из вариантов: «да», «нет», «не видно». "
            "Ключ в ответе — ровно тот, что слева от двоеточия.\n\n" + body)


def local_schema(keys: list[str]) -> dict:
    """Схема, при которой ответ не может оказаться неполным или кривым (Денис).

    Каждый ключ — обязательное поле с перечислением значений, лишние поля запрещены.
    Ключи вместо номеров: номер не самоидентифицируется, и сдвиг ответа на единицу не поймать.
    """
    return {
        "type": "object",
        "properties": {k: {"type": "string", "enum": list(LOCAL_CHOICES)} for k in keys},
        "required": list(keys),
        "additionalProperties": False,
    }
