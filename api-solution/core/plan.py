"""Календарный план объекта: типовой шаблон и разбор загруженного файла.

План хранится по макроэтапам (id 1–8) с конкретными датами начала и окончания.
"""

import io
from datetime import date, datetime, timedelta

import pandas as pd

# Типовой монолитный жилой дом ~17 этажей: (этап, сдвиг начала от старта объекта, длительность), дни.
# Каркас, кровля, фасад и благоустройство перекрываются — так обычно и строят.
TYPICAL = [
    (1, 0, 45),
    (2, 30, 45),
    (3, 60, 50),
    (4, 100, 80),
    (5, 170, 300),
    (6, 440, 60),
    (7, 330, 200),
    (8, 480, 90),
]


def typical_plan(start: date):
    return {sid: (start + timedelta(days=off), start + timedelta(days=off + dur)) for sid, off, dur in TYPICAL}


def plan_between(start: date, end: date):
    """Типовые пропорции этапов, растянутые на период [start, end]: этап 1 начинается
    в start, последний этап заканчивается в end, перекрытия сохраняются."""
    span = max(off + dur for _, off, dur in TYPICAL)
    scale = (end - start).days / span
    return {sid: (start + timedelta(days=round(off * scale)), start + timedelta(days=round((off + dur) * scale)))
            for sid, off, dur in TYPICAL}


_STAGE_COLS = ("stage_id", "stage", "этап", "id", "№")
_START_COLS = ("start", "начало", "дата начала", "plan_start")
_END_COLS = ("end", "окончание", "конец", "дата окончания", "plan_end")


def _find(columns, names):
    low = {str(c).strip().lower(): c for c in columns}
    for n in names:
        if n in low:
            return low[n]
    raise ValueError(f"нет колонки: одна из {', '.join(names)}")


def _to_date(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    raise ValueError(f"не разобрать дату: {text}")


def parse_plan_file(data: bytes, filename: str):
    """CSV или XLSX с колонками «этап», «начало», «окончание» (или stage_id, start, end)."""
    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
    else:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    sc, st, en = _find(df.columns, _STAGE_COLS), _find(df.columns, _START_COLS), _find(df.columns, _END_COLS)
    plan = {}
    for _, row in df.iterrows():
        sid = int(str(row[sc]).strip().split(".")[0])
        plan[sid] = (_to_date(row[st]), _to_date(row[en]))
    return plan
