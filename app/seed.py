"""Наполнение справочников: типы объектов, макроэтапы, чек-листы, виды работ.

Запуск идемпотентный — можно гонять повторно.

    python -m app.seed

Формулировки вопросов — содержательная часть проекта, а не строки интерфейса.
Три правила, по которым они написаны:

1. Вопрос про *область* кадра, а не про фото целиком. Модель Б получает кадр
   с погашенным фоном, и спрашивать её надо о том, что осталось видимым.
2. Ответ тернарный. «Не уверен» — полноценный ответ: кадр, где признак не
   разглядеть, не должен голосовать вовсе.
3. Вопрос — про физически наблюдаемый признак, а не про название этапа.
   «Видна ли опалубка» модель ответит, «идут ли монолитные работы» — нет.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import (MacroStage, ObjectType, StageTemplate, WorkType)

OBJECT_TYPES = [
    "Жильё", "Образование", "Здравоохранение", "Спорт", "Культура",
    "Административные здания", "ДОУ", "Офисно-деловой центр", "Дороги",
]

# id, название, метрика прогресса, ракурс для измерения
MACRO_STAGES = [
    (1, "Подготовка территории", "доля площадки освобождена и огорожена", "top"),
    (2, "Ограждение котлована, шпунт, сваи", "число погружённых свай", "any"),
    (3, "Земляные работы, котлован", "площадь и глубина выемки", "top"),
    (4, "Монолит подземной части", "доля забетонированной плиты", "top"),
    (5, "Монолит надземной части", "номер текущего этажа", "side"),
    (6, "Кровля", "доля закрытой кровли", "top"),
    (7, "Фасад и остекление", "доля закрытого фасада", "side"),
    (8, "Благоустройство", "доля благоустроенной площади", "top"),
]

TEMPLATES: dict[int, dict] = {
    1: {
        "must_have": ["строительный забор по периметру", "бытовой городок",
                      "расчищенная площадка", "временные проезды"],
        "must_not_have": ["открытый котлован", "монолитные конструкции"],
        "equipment_expected": ["экскаватор", "бульдозер", "самосвал", "погрузчик"],
        "equipment_forbidden": ["башенный кран", "автобетононасос"],
        "measurements": ["доля площадки, освобождённой и огороженной, %"],
        "questions": [
            ("fence", "must_have",
             "Виден ли сплошной строительный забор, огораживающий площадку?"),
            ("cabins", "must_have",
             "Виден ли бытовой городок — строительные бытовки или вагончики?"),
            ("cleared", "must_have",
             "Расчищена ли площадка от прежних построек и растительности?"),
            ("no_pit", "must_not_have",
             "Виден ли вырытый котлован или глубокая выемка грунта?"),
        ],
    },
    2: {
        "must_have": ["сваебойная или буровая установка", "шпунтовое ограждение",
                      "оголовки свай"],
        "must_not_have": ["монолитные стены выше уровня земли", "фасадные леса"],
        "equipment_expected": ["копёр", "буровая установка", "гусеничный кран",
                               "автокран", "самосвал"],
        "equipment_forbidden": ["автобетононасос"],
        "measurements": ["число погружённых свай", "длина шпунтового ограждения, м"],
        "questions": [
            ("rig", "must_have",
             "Видна ли сваебойная или буровая установка — машина с высокой "
             "вертикальной мачтой?"),
            ("sheet_pile", "must_have",
             "Видно ли шпунтовое ограждение — стена из вертикальных "
             "металлических профилей, забитых в грунт?"),
            ("pile_heads", "must_have",
             "Видны ли оголовки свай, торчащие из грунта?"),
            ("no_walls", "must_not_have",
             "Видны ли бетонные стены или колонны выше уровня земли?"),
        ],
    },
    3: {
        "must_have": ["открытый котлован", "откосы или крепление стенок",
                      "отвал вынутого грунта"],
        "must_not_have": ["бетонные конструкции выше уровня земли", "фасадные леса"],
        "equipment_expected": ["экскаватор", "самосвал", "бульдозер", "погрузчик"],
        "equipment_forbidden": ["автобетононасос", "каток"],
        "measurements": ["площадь вскрытого грунта, % области"],
        "questions": [
            ("pit", "must_have",
             "Виден ли открытый котлован или выемка грунта ниже уровня земли?"),
            ("soil_pile", "must_have",
             "Виден ли отвал вынутого грунта или кучи земли?"),
            ("earthwork", "must_have",
             "Видна ли землеройная техника — экскаватор или бульдозер — "
             "в котловане или у его края?"),
            ("no_concrete", "must_not_have",
             "Видны ли готовые бетонные конструкции выше уровня земли?"),
        ],
    },
    4: {
        "must_have": ["опалубка", "арматурные каркасы", "бетонная плита в котловане"],
        "must_not_have": ["этажи выше уровня земли", "фасадные панели"],
        "equipment_expected": ["автобетононасос", "автобетоносмеситель",
                               "башенный кран", "автокран"],
        "equipment_forbidden": ["каток"],
        "measurements": ["доля забетонированной площади плиты, %"],
        "questions": [
            ("formwork", "must_have",
             "Видна ли опалубка — щиты или формы для заливки бетона?"),
            ("rebar", "must_have",
             "Видны ли арматурные каркасы, сетка или прутья арматуры?"),
            ("slab", "must_have",
             "Залита ли бетонная плита на дне котлована?"),
            ("below_grade", "must_have",
             "Находятся ли основные строительные конструкции ниже "
             "уровня окружающей земли?"),
            ("no_floors", "must_not_have",
             "Возвышается ли здание на один или несколько этажей "
             "выше уровня земли?"),
        ],
    },
    5: {
        "must_have": ["этажи выше уровня земли", "башенный кран",
                      "опалубка перекрытий", "незавершённый верхний этаж"],
        "must_not_have": ["полностью закрытый фасад", "благоустроенная территория"],
        "equipment_expected": ["башенный кран", "автобетононасос",
                               "автобетоносмеситель", "подъёмник"],
        "equipment_forbidden": ["экскаватор", "каток"],
        "measurements": ["номер текущего этажа", "высота верхней границы объекта, px"],
        "questions": [
            ("above_grade", "must_have",
             "Возвышается ли здание выше уровня земли на один этаж или больше?"),
            ("crane", "must_have",
             "Виден ли башенный кран рядом со зданием?"),
            ("formwork_floor", "must_have",
             "Видна ли опалубка перекрытий или стен на верхних этажах?"),
            ("unfinished_top", "must_have",
             "Верхний этаж выглядит незавершённым — торчит арматура, "
             "открытые перекрытия, нет стен?"),
            ("no_facade", "must_not_have",
             "Закрыт ли фасад здания навесными панелями или облицовкой "
             "полностью?"),
        ],
    },
    6: {
        "must_have": ["работы на верхнем перекрытии", "кровельное покрытие",
                      "парапеты"],
        "must_not_have": ["рост этажности", "опалубка перекрытий"],
        "equipment_expected": ["башенный кран", "подъёмник", "автокран"],
        "equipment_forbidden": ["экскаватор", "копёр"],
        "measurements": ["доля закрытой кровли, %"],
        "questions": [
            ("roof_work", "must_have",
             "Ведутся ли работы на крыше здания — видны ли люди или "
             "материалы на верхнем перекрытии?"),
            ("roof_cover", "must_have",
             "Закрыта ли крыша кровельным покрытием?"),
            ("parapet", "must_have",
             "Видны ли парапеты или ограждения по периметру крыши?"),
            ("no_growth", "must_not_have",
             "Продолжает ли здание расти вверх — строится ли новый этаж "
             "над текущим верхним?"),
        ],
    },
    7: {
        "must_have": ["фасадные леса или подъёмники", "навесные панели или облицовка",
                      "установленное остекление"],
        "must_not_have": ["голые бетонные конструкции по всей высоте",
                          "опалубка перекрытий"],
        "equipment_expected": ["подъёмник", "автокран", "башенный кран"],
        "equipment_forbidden": ["экскаватор", "копёр", "бульдозер"],
        "measurements": ["доля закрытого фасада, %"],
        "questions": [
            ("scaffold", "must_have",
             "Видны ли фасадные леса, люльки или строительные подъёмники "
             "на стенах здания?"),
            ("cladding", "must_have",
             "Закрыт ли фасад навесными панелями, облицовкой или штукатуркой "
             "хотя бы частично?"),
            ("glazing", "must_have",
             "Установлены ли окна или витражное остекление?"),
            ("bare_concrete", "must_not_have",
             "Здание представляет собой голый бетонный каркас без какой-либо "
             "наружной отделки?"),
        ],
    },
    8: {
        "must_have": ["покрытие проездов и тротуаров", "озеленение",
                      "малые архитектурные формы"],
        "must_not_have": ["открытый грунт по всей территории", "башенный кран"],
        "equipment_expected": ["каток", "погрузчик", "самосвал", "экскаватор"],
        "equipment_forbidden": ["копёр", "автобетононасос"],
        "measurements": ["доля благоустроенной площади, %"],
        "questions": [
            ("paving", "must_have",
             "Уложено ли покрытие проездов или тротуаров — асфальт "
             "или тротуарная плитка?"),
            ("landscaping", "must_have",
             "Видно ли озеленение — газон, посаженные деревья или кустарники?"),
            ("amenities", "must_have",
             "Видны ли малые архитектурные формы — детские или спортивные "
             "площадки, скамейки, урны?"),
            ("bare_ground", "must_not_have",
             "Территория вокруг здания представляет собой открытый грунт "
             "без покрытия и озеленения?"),
        ],
    },
}

# Общий вопрос, задаваемый независимо от этапа. Страхует от ошибки маски:
# если модель уверенно говорит «завершено» там, где по плану идёт монолит,
# это повод показать кадр оператору, а не молча принять вывод.
GLOBAL_QUESTIONS = [
    ("is_construction", "context",
     "В этой области ведётся строительство или это завершённое здание? "
     "Ответь одним словом: стройка или завершено."),
]


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


def seed_macro_stages(s) -> None:
    for sid, name, metric, view in MACRO_STAGES:
        st = s.get(MacroStage, sid)
        if st is None:
            st = MacroStage(id=sid)
            s.add(st)
        st.name = name
        st.order_default = sid
        st.progress_metric = metric
        st.progress_view = view
    s.flush()


def seed_templates(s) -> None:
    for sid, spec in TEMPLATES.items():
        tpl = s.scalar(select(StageTemplate)
                       .where(StageTemplate.macro_stage_id == sid))
        if tpl is None:
            tpl = StageTemplate(macro_stage_id=sid)
            s.add(tpl)
        tpl.must_have = spec["must_have"]
        tpl.must_not_have = spec["must_not_have"]
        tpl.equipment_expected = spec["equipment_expected"]
        tpl.equipment_forbidden = spec["equipment_forbidden"]
        tpl.measurements = spec["measurements"]
        tpl.questions = [
            {"key": k, "polarity": p, "text": t}
            for k, p, t in list(spec["questions"]) + GLOBAL_QUESTIONS
        ]
    s.flush()


def seed_work_types(s) -> int:
    """Загружает разобранный справочник. Требует предварительного запуска
    tools/parse_workbook.py — он чинит битые коды и раскладывает по этапам.
    """
    path = Path("reference/work_types.csv")
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
        types = seed_object_types(s)
        print(len(types))

        print("макроэтапы...", end=" ")
        seed_macro_stages(s)
        print(len(MACRO_STAGES))

        print("чек-листы...", end=" ")
        seed_templates(s)
        total_q = sum(len(t["questions"]) + len(GLOBAL_QUESTIONS)
                      for t in TEMPLATES.values())
        print(f"{len(TEMPLATES)} шаблонов, {total_q} вопросов")

        print("виды работ...", end=" ")
        n = seed_work_types(s)
        print(n)

        s.commit()
    print("готово")


if __name__ == "__main__":
    main()
