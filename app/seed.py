"""Наполнение справочников: типы объектов, макроэтапы, чек-листы, виды работ.

Запуск идемпотентный — можно гонять повторно.

    python -m app.seed

**Содержимое лежит в CSV, а не в коде.** Три таблицы в `reference/`:

    stages.csv              макроэтапы: название, метрика прогресса, ракурс,
                            для каких типов объектов применим
    checklist_questions.csv пул наблюдений: ключ, признак необратимости, текст
    stage_questions.csv     какой этап какими наблюдениями подтверждается
                            и в какую сторону (must_have / must_not_have)

Пул вопросов отделён от привязки к этапам намеренно. Один и тот же признак
работает на разные этапы в разные стороны: «здание выше уровня земли» — это
`must_have` для монолита надземной части и `must_not_have` для котлована.
Если хранить вопросы внутри этапа, такой признак дублируется, и модели
задаётся один и тот же вопрос дважды разными словами.

Формулировки — содержательная часть проекта, а не строки интерфейса.
Четыре правила, по которым они написаны:

1. Вопрос про *область* кадра, а не про фото целиком. Модель Б получает кадр
   с погашенным фоном, и спрашивать её надо о том, что осталось видимым.
2. Ответ тернарный. «Не видно» — полноценный ответ: кадр, где признак не
   разглядеть, не должен голосовать вовсе.
3. Вопрос — про физически наблюдаемый признак, а не про название этапа.
   «Видна ли опалубка» модель ответит, «идут ли монолитные работы» — нет.
4. Вопрос — про то, что **видно сейчас**, а не про то, что логически есть.
   «Видна ли залитая плита», а не «залита ли плита»: когда её перекроют
   этажи, честный ответ «не видно», и он не должен выглядеть ошибкой.
   Необратимость учитывается отдельно, флагом `latching`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import MacroStage, ObjectType, StageTemplate, WorkType

REFERENCE = Path("reference")

OBJECT_TYPES = [
    "Жильё", "Образование", "Здравоохранение", "Спорт", "Культура",
    "Административные здания", "ДОУ", "Офисно-деловой центр", "Дороги",
]

ANY = "*"


def _split(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(";") if x.strip()]


def _rows(name: str) -> list[dict]:
    path = REFERENCE / name
    if not path.exists():
        raise SystemExit(f"нет файла {path}")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_object_types(s) -> dict[str, int]:
    out = {}
    for name in OBJECT_TYPES:
        obj = s.scalar(select(ObjectType).where(ObjectType.name == name))
        if obj is None:
            obj = ObjectType(name=name)
            s.add(obj)
            s.flush()
        out[name] = obj.id
    return out


def seed_macro_stages(s) -> int:
    rows = _rows("stages.csv")
    for row in rows:
        sid = int(row["id"])
        st = s.get(MacroStage, sid)
        if st is None:
            st = MacroStage(id=sid)
            s.add(st)
        st.name = row["name"]
        st.order_default = sid
        st.progress_metric = row["progress_metric"]
        st.progress_view = row["view"]
        st.object_types = ([] if row["object_types"].strip() == ANY
                           else [x for x in row["object_types"].split(";") if x])
        tpl = s.scalar(select(StageTemplate)
                       .where(StageTemplate.macro_stage_id == sid))
        if tpl is None:
            tpl = StageTemplate(macro_stage_id=sid)
            s.add(tpl)
        tpl.equipment_expected = _split(row["equipment_expected"])
        tpl.equipment_forbidden = _split(row["equipment_forbidden"])
        tpl.measurements = [row["progress_metric"]]
    s.flush()
    return len(rows)


def load_checklists() -> dict[int, list[dict]]:
    """CSV → {stage_id: [вопрос, ...]} в порядке файла привязок.

    Вопросы с `stage_id = *` добавляются каждому этапу: это общий контекст,
    который спрашивается независимо от того, какую гипотезу проверяем.
    """
    pool = {r["key"]: r for r in _rows("checklist_questions.csv")}
    by_stage: dict[int, list[dict]] = {}
    shared: list[dict] = []

    for row in _rows("stage_questions.csv"):
        key = row["key"].strip()
        if key not in pool:
            raise SystemExit(f"вопрос {key!r} есть в привязке, но нет в пуле")
        q = {
            "key": key,
            "text": pool[key]["text"],
            "polarity": row["polarity"].strip(),
            "latching": pool[key]["latching"].strip() in ("1", "true", "да"),
        }
        raw = row["stage_id"].strip()
        if raw == ANY:
            shared.append(q)
        else:
            by_stage.setdefault(int(raw), []).append(q)

    for questions in by_stage.values():
        questions.extend(shared)
    return by_stage


def seed_templates(s) -> tuple[int, int]:
    by_stage = load_checklists()
    for sid, questions in by_stage.items():
        tpl = s.scalar(select(StageTemplate)
                       .where(StageTemplate.macro_stage_id == sid))
        if tpl is None:
            tpl = StageTemplate(macro_stage_id=sid)
            s.add(tpl)
        tpl.questions = questions
        tpl.must_have = [q["key"] for q in questions if q["polarity"] == "must_have"]
        tpl.must_not_have = [q["key"] for q in questions
                             if q["polarity"] == "must_not_have"]
    s.flush()
    return len(by_stage), sum(len(v) for v in by_stage.values())


def seed_work_types(s) -> int:
    """Загружает разобранный справочник. Требует предварительного запуска
    tools/parse_workbook.py — он чинит битые коды и раскладывает по этапам.
    """
    path = REFERENCE / "work_types.csv"
    if not path.exists():
        print(f"  пропущено: нет {path}, сначала tools/parse_workbook.py")
        return 0

    s.query(WorkType).delete()
    n = 0
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = int(row["macro_stage_id"])
            reason = row["reason"]
            s.add(WorkType(
                code=row["code"],
                parent_code=row["parent_code"],
                level=int(row["level"]),
                name=row["name"],
                macro_stage_id=sid if sid > 0 else None,
                excluded_reason=reason if sid == 0 else "",
                mapping_reason=reason if sid > 0 else "",
                object_types=row["object_types"].split("|") if row["object_types"] else [],
            ))
            n += 1
    return n


def main() -> None:
    init_db()
    with SessionLocal() as s:
        print("типы объектов...", end=" ")
        print(len(seed_object_types(s)))

        print("макроэтапы...", end=" ")
        print(seed_macro_stages(s))

        print("чек-листы...", end=" ")
        stages, questions = seed_templates(s)
        print(f"{stages} этапов, {questions} привязок вопросов")

        print("виды работ...", end=" ")
        print(seed_work_types(s))

        s.commit()
    print("готово")


if __name__ == "__main__":
    main()
