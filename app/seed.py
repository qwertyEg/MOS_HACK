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

from sqlalchemy import delete, select

from app.db import SessionLocal, init_db
from app.models import MacroStage, ObjectType, SiteStage, StageTemplate, WorkType

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


RETIRED = " (устарел)"


def _retire_stale_stages(s, keep: set[int]) -> None:
    """Разбирается с макроэтапами, которых больше нет в справочнике.

    Вызывается **до** обновления остальных, и это не косметика. Справочник
    ужался с восьми этапов до семи (кровля и фасад слиты), из-за чего номера
    сдвинулись: то, что раньше было этапом 8, теперь этап 7. Имя этапа в базе
    уникально, поэтому переименовать седьмой в «Благоустройство», пока
    восьмой ещё зовётся так же, нельзя — база откажет. Сначала освобождаем
    имя, потом переименовываем.

    Удалять можно только то, на что никто не ссылается. Если на старый этап
    уже завязан чей-то календарный план, это чужие данные: такой этап
    остаётся в базе, но помечается устаревшим — и из выпадающего списка
    добавления этапа исчезает по имени, и человеку видно, что он лишний.
    """
    for st in s.scalars(select(MacroStage)).all():
        if st.id in keep:
            continue

        used = s.scalars(select(SiteStage)
                         .where(SiteStage.macro_stage_id == st.id)).all()
        if not used:
            s.execute(delete(StageTemplate).where(
                StageTemplate.macro_stage_id == st.id))
            s.delete(st)
            print(f"  - удалён устаревший этап {st.id} «{st.name}»")
            continue

        if not st.name.endswith(RETIRED):
            st.name += RETIRED
        # Нулевой порядок — признак «вне плана». По нему этап исчезает из
        # предзаполнения нового объекта и из списка добавления: иначе он
        # продолжал бы всплывать там, где выбирают этап из справочника.
        st.order_default = 0
        sites = sorted({u.site_id for u in used})
        print(f"  ! этап {st.id} «{st.name}» убран из справочника, но на него "
              f"ссылаются планы объектов {sites}.")
        print("    Строки оставлены, данные целы. Планы этих объектов стоит "
              "перезалить: состав вопросов у них заморожен на момент "
              "сохранения и остался восьмиэтапным.")
    s.flush()


def seed_macro_stages(s) -> int:
    rows = _rows("stages.csv")
    _retire_stale_stages(s, {int(row["id"]) for row in rows})
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
            # «Как выглядит» — визуальный якорь для модели. Без него VLM
            # отвечает по названию признака, а не по картинке: на вопрос про
            # шпунт она ищет слово «шпунт», а не гофрированную стенку.
            "hint": (pool[key].get("hint") or "").strip(),
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
