"""Мониторинг стройки по фото: Streamlit-интерфейс.

Запуск: .venv/bin/streamlit run app.py
"""

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from core import config, service, timeline
from core.analyzer import STRATEGIES, Analyzer
from core.checklist import Checklist
from core.glm import GLMClient
from core.images import detect_date
from core.plan import parse_plan_file, typical_plan
from core.storage import Storage

OBJECT_TYPES = ["Жильё", "Образование", "Здравоохранение", "Спорт", "Культура",
                "Административные здания", "ДОУ", "Офисно-деловой центр"]
SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
ANSWER_ICON = {"yes": "✅ да", "no": "❌ нет", "unsure": "❔ не видно"}
STATUS_RU = {"done": "завершён", "active": "идёт", "not_started": "не начат", "unknown": "нет данных"}

st.set_page_config(page_title="Мониторинг стройки", layout="wide")


@st.cache_resource
def get_checklist():
    return Checklist()


@st.cache_resource
def get_storage():
    return Storage()


checklist = get_checklist()
storage = get_storage()
stage_name = {s["id"]: s["name"] for s in checklist.stages}
sub_name = {sub["id"]: sub["name"] for s in checklist.stages for sub in s["substages"]}


# ---------- боковая панель: объект и настройки модели ----------

with st.sidebar:
    st.header("Объект")
    objects = storage.list_objects()
    names = [o["name"] for o in objects]
    choice = st.selectbox("Объект", names + ["+ новый объект"], index=0 if names else len(names))

    if choice == "+ новый объект":
        with st.form("new_object"):
            new_name = st.text_input("Название")
            new_type = st.selectbox("Тип", OBJECT_TYPES)
            new_floors = st.number_input("Этажность по проекту (0 — неизвестно)", 0, 150, 0)
            if st.form_submit_button("Создать") and new_name.strip():
                oid = storage.create_object(new_name.strip(), new_type, new_floors or None)
                storage.save_plan(oid, typical_plan(date.today() - timedelta(days=180)))
                st.rerun()
        st.stop()

    obj = next(o for o in objects if o["name"] == choice)
    obj_type = st.selectbox("Тип", OBJECT_TYPES, index=OBJECT_TYPES.index(obj["object_type"]))
    floors = st.number_input("Этажность по проекту (0 — неизвестно)", 0, 150, obj["floors_total"] or 0)
    if obj_type != obj["object_type"] or (floors or None) != obj["floors_total"]:
        storage.update_object(obj["id"], obj_type, floors or None)
        obj["object_type"], obj["floors_total"] = obj_type, floors or None

    st.header("Модель")
    models = list(config.PRICES)
    model = st.selectbox("GLM", models, index=models.index(config.DEFAULT_MODEL)
                         if config.DEFAULT_MODEL in models else 0)
    p = config.PRICES[model]
    st.caption(f"${p.input}/{p.cached_input}/{p.output} за 1M токенов (вход/кэш/выход)")
    thinking = st.toggle("Рассуждение (thinking)", value=False,
                         help="Дороже в разы по выходным токенам. Включать для сравнения на сложных кадрах.")
    strategy = st.radio("Запросы к модели", list(STRATEGIES), format_func={
        "two_step": "Два шага: разведка + чек-лист кандидатов",
        "per_stage": "Разведка + чек-лист на каждый этап-кандидат",
    }.get)

    spend = storage.spend()
    st.metric("Потрачено всего", f"${spend['cost']:.4f}", f"{spend['calls']} вызовов", delta_color="off")
    if not config.API_KEY:
        st.error("Нет ZAI_API_KEY в .env — анализ новых фото недоступен.")

analyzer = Analyzer(checklist, storage, GLMClient(model=model, thinking=thinking), strategy)


def run_analysis(frames):
    bar = st.progress(0.0, text="Анализ…")
    spent, errors = service.analyze_frames(
        analyzer, frames,
        lambda i, n, f: bar.progress(i / n, text=f"{i + 1}/{n}: {f['filename']}"))
    # Сводка считается до вкладок, поэтому после анализа — перерисовка страницы,
    # а итог запуска переживает её через session_state.
    st.session_state["last_run"] = (len(frames), spent, errors)
    st.rerun()


frames, plan, tl = service.report(checklist, storage, analyzer, obj)

tab_upload, tab_summary, tab_frames, tab_plan = st.tabs(["Загрузка", "Сводка", "Кадры", "План"])


# ---------- загрузка ----------

with tab_upload:
    if "last_run" in st.session_state:
        n, spent, errors = st.session_state.pop("last_run")
        st.success(f"Разобрано кадров: {n - len(errors)} из {n}, потрачено ${spent:.4f}. "
                   "Результаты — на вкладках «Сводка» и «Кадры».")
        for e in errors:
            st.error(e)
    files = st.file_uploader("Фото с камер", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    if files:
        rows = []
        for f in files:
            data = f.getvalue()
            dt, src = detect_date(data, f.name)
            rows.append({"файл": f.name, "дата съёмки": dt or datetime.now().replace(microsecond=0),
                         "откуда дата": src or "не найдена — укажите"})
        table = st.data_editor(pd.DataFrame(rows), hide_index=True, disabled=["файл", "откуда дата"],
                               column_config={"дата съёмки": st.column_config.DatetimeColumn(format="DD.MM.YYYY HH:mm")})
        if st.button(f"Добавить и проанализировать ({len(files)})", type="primary"):
            added = []
            for f, (_, row) in zip(files, table.iterrows()):
                taken = pd.Timestamp(row["дата съёмки"]).to_pydatetime()
                added.append(service.ingest(storage, obj["id"], f.name, f.getvalue(), taken, row["откуда дата"]))
            todo = [f for f in storage.list_frames(obj["id"]) if f["sha256"] in added]
            run_analysis(todo)

    missing = service.pending(storage, analyzer, obj["id"])
    if missing:
        st.divider()
        st.write(f"Кадров без разбора при текущих настройках модели: {len(missing)}.")
        if st.button("Разобрать их"):
            run_analysis(missing)


# ---------- сводка ----------

with tab_summary:
    if not tl["frames"]:
        st.info("Пока нет разобранных кадров. Загрузите фото на первой вкладке.")
    else:
        st.progress(tl["overall_pct"] / 100,
                    text=f"**Готовность объекта: {tl['overall_pct']:.0f}%** (по работам, видимым с камер)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Текущий этап", f"{tl['front']}. {stage_name[tl['front']]}" if tl["front"] else "—")
        sch = tl["schedule"]
        if sch:
            lag = sch["lag_days"]
            c2.metric("График", sch["verdict"],
                      f"{'+' if lag > 0 else ''}{lag} дн. к плану" if lag else "0 дн.",
                      delta_color="inverse")
        else:
            c2.metric("График", "нет плана")
        fc = tl["forecast"]
        if fc and fc.get("finish"):
            c3.metric("Прогноз окончания", f"{fc['finish']:%d.%m.%Y}",
                      f"{fc['delay_days']:+d} дн. к плану" if fc.get("delay_days") is not None else None,
                      delta_color="inverse",
                      help=(f"Темп {fc['pace_vs_plan']}× от планового. " if fc.get("pace_vs_plan") else "")
                      + (fc["note"] or ""))
        else:
            c3.metric("Прогноз окончания", "—", help=fc["note"] if fc else "нужны кадры минимум за 2 разных дня")
        c4.metric("Простой подряд", f"{tl['idle_streak']} дн.")

        if tl["active_substages"]:
            st.write("**Сейчас идёт:** " + "; ".join(f"{s} {sub_name[s]}" for s in tl["active_substages"]))

        st.subheader("Этапы")
        plan_rows = {s["stage"]: s for s in sch["stages"]} if sch else {}
        for s in checklist.stages:
            p = tl["stage_progress"][s["id"]]
            label = f"{s['id']}. {s['name']} — {p * 100:.0f}%"
            pr = plan_rows.get(s["id"])
            if pr:
                label += f" · план {pr['plan_start']:%d.%m}–{pr['plan_end']:%d.%m.%Y}: должен быть «{pr['planned']}»"
            st.progress(p, text=label)

        st.subheader("Отклонения")
        if not tl["deviations"]:
            st.write("Не обнаружено.")
        frame_by_id = {f["id"]: f for f in frames}
        for d in reversed(tl["deviations"]):
            when = (f"{d['date_from']:%d.%m}–{d['date']:%d.%m.%Y}" if d["date_from"] != d["date"]
                    else f"{d['date']:%d.%m.%Y}")
            with st.expander(f"{SEVERITY_ICON.get(d['severity'], '')} {when} — {d['title']}"):
                st.write(d["detail"])
                if d.get("stage"):
                    st.caption(f"Этап: {d['stage']}. {stage_name[d['stage']]}")
                shots = [frame_by_id[i] for i in d["frame_ids"] if i in frame_by_id][:4]
                if shots:
                    st.image([f["image_path"] for f in shots], caption=[f["filename"] for f in shots], width=260)

        left, right = st.columns(2)
        with left:
            st.subheader("Метрики")
            st.dataframe(pd.DataFrame(tl["metrics"], columns=["метрика", "значение", "пояснение"]),
                         hide_index=True, width="stretch")
        with right:
            st.subheader("Факт и план, %")
            fact = pd.Series({pd.Timestamp(t): v for t, v in tl["series"]}).groupby(level=0).max()
            fact = fact.resample("D").max().ffill()
            chart = pd.DataFrame({"факт": fact})
            if plan:
                chart["план"] = [timeline.expected_pct(checklist, plan, d.date()) for d in chart.index]
            st.line_chart(chart)

        st.subheader("По дням")
        st.dataframe(pd.DataFrame([{
            "дата": d["date"], "кадров": d["frames"],
            "этап": d["front"], "работы": "да" if d["active"] else "простой" if d["idle"] else "?",
            "техника (в работе/всего)": ", ".join(f"{checklist.equipment_name(k)} {d['working'].get(k, 0)}/{v}"
                                                   for k, v in d["equipment"].items()) or "—",
            "рабочих": d["workers"],
        } for d in reversed(tl["days"])]), hide_index=True, width="stretch")


# ---------- кадры ----------

with tab_frames:
    scored = {f["id"]: f for f in tl["frames"]}
    for f in reversed(frames):
        s = scored.get(f["id"])
        a = f["analysis"]
        img_col, info_col = st.columns([2, 3])
        img_col.image(f["image_path"], width="stretch")
        with info_col:
            st.markdown(f"**{f['filename']}** · {f['taken_at'].replace('T', ' ')} ({f['date_source'] or 'дата вручную'})")
            if not a:
                st.warning("Не разобран.")
            else:
                t = a["triage"]
                front = s["score"]["front"]
                st.write(f"Этап по кадру: **{front}. {stage_name[front]}**" if front else "Этап не определён",
                         f"· готовность по кадру {s['score']['overall_pct']:.0f}%")
                st.caption(f"Качество: {t['quality']} · ракурс: {t['view']} · {t['description']}")
                if t["equipment"]:
                    st.dataframe(pd.DataFrame([{"техника": checklist.equipment_name(e["type"]),
                                                "всего": e["total"], "в работе": e["working"],
                                                "признак": e["evidence"]} for e in t["equipment"]]),
                                 hide_index=True, width="stretch")
                meas = {"рабочих": t["workers_count"], "этажей": t["floors_built"],
                        "остеклено этажей": t["floors_glazed"], "облицовано, %": t["facade_clad_pct"],
                        "котлован, %": t["pit_area_pct"]}
                st.caption(" · ".join(f"{k}: {v}" for k, v in meas.items() if v is not None))
                for dev in s["deviations"]:
                    st.write(f"{SEVERITY_ICON.get(dev['severity'], '')} {dev['title']}")
                with st.expander("Чек-лист и оценка этапов"):
                    st.dataframe(pd.DataFrame([{
                        "этап": sid, "вероятность (разведка)": st_["likelihood"],
                        "признаки (чек-лист)": None if st_["evidence"] is None else round(st_["evidence"], 2),
                        "готовность": f"{st_['progress'] * 100:.0f}%", "статус": STATUS_RU[st_["status"]],
                    } for sid, st_ in s["score"]["stages"].items()]), hide_index=True, width="stretch")
                    st.dataframe(pd.DataFrame([{
                        "признак": checklist.signs[k]["question"], "ответ": ANSWER_ICON[v],
                    } for k, v in a["answers"].items()]), hide_index=True, width="stretch")
                    for c in a["comments"]:
                        st.caption(c)
                u = a["usage"]
                st.caption(f"{a['model']} · {a['strategy']} · вход {u['prompt_tokens']} (кэш {u['cached_tokens']}) · "
                           f"выход {u['completion_tokens']} · ${u['cost_usd']:.4f} · {u['latency_ms'] / 1000:.1f} с")
                with st.expander("Сырой ответ модели"):
                    st.json(a["raw"])
            if st.button("Удалить кадр", key=f"del{f['id']}"):
                storage.delete_frame(f["id"])
                st.rerun()
        st.divider()


# ---------- план ----------

with tab_plan:
    st.write("Даты начала и окончания макроэтапов. Сравнение с фактом идёт по ним.")
    df = pd.DataFrame([{"этап": s["id"], "название": s["name"],
                        "начало": pd.to_datetime(plan.get(s["id"], (None, None))[0]),
                        "окончание": pd.to_datetime(plan.get(s["id"], (None, None))[1])}
                       for s in checklist.stages])
    edited = st.data_editor(df, hide_index=True, disabled=["этап", "название"], column_config={
        "начало": st.column_config.DateColumn(format="DD.MM.YYYY"),
        "окончание": st.column_config.DateColumn(format="DD.MM.YYYY"),
    })
    if st.button("Сохранить план", type="primary"):
        storage.save_plan(obj["id"], {
            int(r["этап"]): (None if pd.isna(r["начало"]) else pd.Timestamp(r["начало"]).date(),
                             None if pd.isna(r["окончание"]) else pd.Timestamp(r["окончание"]).date())
            for _, r in edited.iterrows()})
        st.rerun()

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Типовой план от даты", value=date.today() - timedelta(days=180), format="DD.MM.YYYY")
        if st.button("Заполнить типовым планом"):
            storage.save_plan(obj["id"], typical_plan(start))
            st.rerun()
    with c2:
        up = st.file_uploader("Или загрузить CSV / XLSX (колонки: этап, начало, окончание)", type=["csv", "xlsx"])
        if up and st.button("Загрузить план из файла"):
            try:
                storage.save_plan(obj["id"], parse_plan_file(up.getvalue(), up.name))
                st.rerun()
            except ValueError as e:
                st.error(str(e))
