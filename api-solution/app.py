"""Мониторинг стройки по фото: Streamlit-интерфейс.

Запуск: .venv/bin/streamlit run app.py
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from core import config, service, site, timeline
from core.analyzer import STRATEGIES, Analyzer
from core.checklist import Checklist
from core.images import detect_date
from core.plan import parse_plan_file
from core.providers import PROVIDERS, make_client
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
                # План не задаём: он построится сам по датам загруженных фото.
                storage.create_object(new_name.strip(), new_type, new_floors or None)
                st.rerun()
        st.stop()

    obj = next(o for o in objects if o["name"] == choice)
    obj_type = st.selectbox("Тип", OBJECT_TYPES, index=OBJECT_TYPES.index(obj["object_type"]))
    floors = st.number_input("Этажность по проекту (0 — неизвестно)", 0, 150, obj["floors_total"] or 0)
    if obj_type != obj["object_type"] or (floors or None) != obj["floors_total"]:
        storage.update_object(obj["id"], obj_type, floors or None)
        obj["object_type"], obj["floors_total"] = obj_type, floors or None

    st.header("Модель")
    provider = st.radio("Где работает модель", list(PROVIDERS), format_func=lambda k: PROVIDERS[k]["label"])
    spec = PROVIDERS[provider]
    models = spec["models"]
    model = st.selectbox("Модель", models, index=models.index(spec["default_model"])
                         if spec["default_model"] in models else 0)
    if model in config.PRICES and provider == "zai":
        p = config.PRICES[model]
        st.caption(f"${p.input}/{p.cached_input}/{p.output} за 1M токенов (вход/кэш/выход)")
    thinking = False
    if spec["supports_thinking"]:
        thinking = st.toggle("Рассуждение (thinking)", value=False,
                             help="Дороже в разы по выходным токенам. Включать для сравнения на сложных кадрах.")
    ready, why_not = spec["status"]()
    use_context = st.toggle("Учитывать историю стройки", value=True,
                            help="Каждый кадр разбирается с учётом подтверждённого по более ранним фото. "
                                 "Выключить — для сравнения: каждый кадр сам по себе.")
    strategy = st.radio("Запросы к модели", list(STRATEGIES), format_func={
        "two_step": "Два шага: разведка + чек-лист кандидатов",
        "per_stage": "Разведка + чек-лист на каждый этап-кандидат",
    }.get)

    spend = storage.spend()
    st.metric("Потрачено всего", f"${spend['cost']:.4f}", f"{spend['calls']} вызовов", delta_color="off")
    if not ready:
        st.error(f"Анализ новых фото недоступен: {why_not}.")

analyzer = Analyzer(checklist, storage, make_client(provider, model, thinking), strategy, use_context)


def run_analysis(n):
    bar = st.progress(0.0, text="Анализ…")
    spent, errors = service.analyze_frames(
        storage, analyzer, obj,
        lambda i, total, f: bar.progress(i / total, text=f"кадр {i + 1} из {total}: {f['filename']}"))
    # Сводка считается до вкладок, поэтому после анализа — перерисовка страницы,
    # а итог запуска переживает её через session_state.
    st.session_state["last_run"] = (n, spent, errors)
    st.rerun()


frames, plan, tl = service.report(checklist, storage, analyzer, obj)

tab_upload, tab_summary, tab_site, tab_frames, tab_plan = st.tabs(["Загрузка", "Сводка", "Стройка", "Кадры", "План"])


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
        if st.button(f"Добавить и проанализировать ({len(files)})", type="primary", disabled=not ready):
            for f, (_, row) in zip(files, table.iterrows()):
                taken = pd.Timestamp(row["дата съёмки"]).to_pydatetime()
                service.ingest(storage, obj["id"], f.name, f.getvalue(), taken, row["откуда дата"])
            run_analysis(len(service.pending(storage, analyzer, obj)))

    missing = service.pending(storage, analyzer, obj)
    if missing:
        st.divider()
        st.write(f"Ожидают разбора: {len(missing)} кадр(ов). Это новые кадры, кадры после фото, "
                 "загруженного задним числом (у них изменилась история стройки), или кадры, "
                 "разобранные другой моделью.")
        if st.button("Разобрать", disabled=not ready):
            run_analysis(len(missing))


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
                         hide_index=True)
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
        } for d in reversed(tl["days"])]), hide_index=True)


# ---------- кадры ----------

with tab_frames:
    scored = {f["id"]: f for f in tl["frames"]}
    for f in reversed(frames):
        s = scored.get(f["id"])
        a = f["analysis"]
        img_col, info_col = st.columns([2, 3])
        img_col.image(f["image_path"])
        with info_col:
            st.markdown(f"**{f['filename']}** · {f['taken_at'].replace('T', ' ')} ({f['date_source'] or 'дата вручную'})")
            if a and not f["current"]:
                st.caption("⚠ Разбор устарел (изменилась история стройки или модель) — показан последний.")
            if not a:
                st.warning("Не разобран.")
            else:
                t = a["triage"]
                front = s["score"]["front"]
                st.write(f"Этап по кадру: **{front}. {stage_name[front]}**" if front else "Этап не определён",
                         f"· готовность по кадру {s['score']['overall_pct']:.0f}%")
                st.caption(f"Качество: {t['quality']} · ракурс: {t['view']} · {t['description']}")
                if t.get("context_conflict"):
                    st.warning(f"Противоречит истории стройки: {t['context_conflict']}")
                if t["equipment"]:
                    st.dataframe(pd.DataFrame([{"техника": checklist.equipment_name(e["type"]),
                                                "всего": e["total"], "в работе": e["working"],
                                                "признак": e["evidence"]} for e in t["equipment"]]),
                                 hide_index=True)
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
                    } for sid, st_ in s["score"]["stages"].items()]), hide_index=True)
                    st.dataframe(pd.DataFrame([{
                        "признак": checklist.signs[k]["question"], "ответ": ANSWER_ICON[v],
                    } for k, v in a["answers"].items()]), hide_index=True)
                    for c in a["comments"]:
                        st.caption(c)
                u = a["usage"]
                st.caption(f"{a.get('provider', 'zai')}:{a['model']} · {a['strategy']} · вход {u['prompt_tokens']} (кэш {u['cached_tokens']}) · "
                           f"выход {u['completion_tokens']} · ${u['cost_usd']:.4f} · {u['latency_ms'] / 1000:.1f} с")
                if a.get("context"):
                    with st.expander("Контекст стройки, переданный модели"):
                        st.text(a["context"])
                with st.expander("Сырой ответ модели"):
                    st.json(a["raw"])
            if st.button("Удалить кадр", key=f"del{f['id']}"):
                service.delete_frame(storage, obj["id"], f["id"])
                st.rerun()
        st.divider()


# ---------- стройка: сводка по всем фото ----------

with tab_site:
    if not tl["frames"]:
        st.info("Пока нет разобранных кадров.")
    else:
        kb = site.knowledge(checklist, tl)
        d0, d1 = kb["period"]
        st.write(f"**{obj['name']}** · {kb['photos']} фото за {kb['days']} дн. съёмки, "
                 f"{d0:%d.%m.%Y} — {d1:%d.%m.%Y} · этап {kb['front']} · готовность {kb['overall_pct']:.0f}%")
        st.caption("Сводка собирается из разборов всех фото стройки. Каждый кадр разбирается с учётом "
                   "уже подтверждённого по более ранним фото, поэтому разборы не противоречат друг другу.")

        st.subheader("Техника за весь период")
        eq_rows = [{"техника": e["name"], "на фото": e["photos"], "дней": e["days"],
                    "максимум одновременно": e["max_at_once"], "доля в работе": e["working_share"],
                    "впервые": e["first_seen"], "последний раз": e["last_seen"],
                    "на этапах": ", ".join(map(str, e["stages"]))} for e in kb["equipment"]]
        st.dataframe(pd.DataFrame(eq_rows), hide_index=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Подтверждённые факты")
            st.dataframe(pd.DataFrame([{"факт": f["fact"], "с": f["since"], "фото": f["photo"]}
                                       for f in kb["facts"]]), hide_index=True)
        with right:
            st.subheader("Этапы")
            st.dataframe(pd.DataFrame([{"этап": f"{s['stage']}. {s['name']}", "готово, %": s["progress_pct"],
                                        "начат": s["first_seen"], "завершён": s["done_at"],
                                        "план": f"{s['plan_start']:%d.%m.%Y}–{s['plan_end']:%d.%m.%Y}"
                                        if s["plan_start"] else ""} for s in kb["stages"]]), hide_index=True)
        if kb["conflicts"]:
            st.subheader("Снимки, противоречащие истории")
            st.dataframe(pd.DataFrame(kb["conflicts"]), hide_index=True)

        st.subheader("Выгрузка CSV")
        st.caption("Разделитель «;», UTF-8 — открывается в Excel.")
        rows = site.frames_table(checklist, tl)
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("По каждому фото", site.to_csv(rows), f"{obj['name']}_фото.csv", "text/csv")
        c2.download_button("Техника", site.to_csv(eq_rows), f"{obj['name']}_техника.csv", "text/csv")
        c3.download_button("Этапы", site.to_csv(kb["stages"]), f"{obj['name']}_этапы.csv", "text/csv")
        c4.download_button("Отклонения", site.to_csv([
            {"с": d["date_from"], "по": d["date"], "уровень": d["severity"], "этап": d["stage"],
             "отклонение": d["title"], "подробно": d["detail"], "кадров": d["count"]} for d in tl["deviations"]]),
            f"{obj['name']}_отклонения.csv", "text/csv")
        st.dataframe(pd.DataFrame(rows), hide_index=True)


# ---------- план ----------

with tab_plan:
    source = storage.get_plan_source(obj["id"])
    if source == "auto":
        st.info("План построен автоматически: период от первой до последней даты фото, этапы — по типовым "
                "пропорциям (каркас дольше всего, фасад и кровля перекрываются с ним). Пересчитывается сам "
                "при добавлении и удалении фото, пока вы не отредактируете план или не загрузите файл."
                if plan else "План появится сам, когда будут фото минимум за два разных дня.")
    else:
        st.info("План задан " + ("вручную" if source == "manual" else "файлом") + " и не меняется при загрузке фото.")
        if st.button("Вернуть автоплан по датам фото"):
            storage.save_plan(obj["id"], {}, "auto")
            service.refresh_auto_plan(storage, obj["id"])
            st.rerun()

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
            for _, r in edited.iterrows()}, "manual")
        st.rerun()

    st.divider()
    up = st.file_uploader("Загрузить план из CSV / XLSX (колонки: этап, начало, окончание)", type=["csv", "xlsx"])
    if up and st.button("Загрузить план из файла"):
        try:
            storage.save_plan(obj["id"], parse_plan_file(up.getvalue(), up.name), "file")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
