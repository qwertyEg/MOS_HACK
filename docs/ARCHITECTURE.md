# СтройВзор — архитектура и контракты объединённого сервиса

> Ветка `egor-unified`. Объединяет работу команды:
> **Денис** (`dev-lamonifi`: FastAPI-каркас, камеры и поток кадров, маска фона, модель Б на локальной VLM, simcam, Гант),
> **Никита** (`api-solution`: справочник-чек-лист, скоринг этапов, правила отклонений, хронология/прогноз, клиент GLM-4.6V),
> **Егор** (модель А: детекция и классификация техники, «работает/стоит», единая техника без двойного счёта между камерами, моточасы).
>
> Этот документ — договор между модулями. Типы — в `core/contracts.py`, словарь — в `core/taxonomy.py`.
> Любое отступление от контракта модуль фиксирует в своём отчёте, а не молча.

---

## 1. Что делает сервис (одним абзацем)

Камеры площадки присылают снимки раз в 20–30 минут (или пользователь загружает папку / zip / видео).
**Модель А** находит и классифицирует технику, сопоставляет её между соседними кадрами (работает ли машина — сдвинулась/сменила позу)
и между камерами (одна машина с разных ракурсов считается один раз), ведёт статусы ACTIVE/IDLE/PARKED/DEPARTED и
**списывает отработанные моточасы из плановых** («временная полоска» по типу техники).
**Модель Б** по дневным качественным кадрам (ночь, дождь, брак отсекаются; чужие статичные здания гасит динамическая маска)
отвечает на чек-лист признаков «да / нет / не уверен» и по монотонной хронологии определяет текущий этап и его готовность;
этапы и подэтапы привязаны к кодам работ из xlsx организаторов (`reference/works_catalog.xlsx` → `reference/work_map.csv`).
**Аналитика** сверяет технику с этапом по правилам «этап → необходимая / запрещённая техника», сверяет модели А и Б между собой
(часы отработаны, а этап не сменился → задержка) и план с фактом (отставание/опережение, прогноз), и выдаёт
**объяснимые предупреждения со снимками-доказательствами и зоной**. Пользователь может вручную отметить готовность этапа,
поправить даты плана, количество техники и плановые часы. Любую модель можно переключить в UI между
**локальной** (YOLO + SigLIP, работает на ноутбуке без интернета) и **внешним API** (GLM-4.6V).

## 2. Раскладка репозитория и кто за что отвечает

```
core/                     чистая доменная логика: без БД, без HTTP, тестируется без GPU и сети
  contracts.py            ТИПЫ И ПРОТОКОЛЫ (общий, меняется только согласованно)
  taxonomy.py             словарь техники/этапов/признаков из reference/checklist.json
  equipment/              МОДЕЛЬ А
    detect_yolo.py        LOCAL: ultralytics YOLO, веса из models/; улучшение ночных кадров (CLAHE/гамма)
    detect_vlm.py         EXTERNAL: GLM-4.6V с рамками (grounding), тот же Detection
    postprocess.py        межклассовый NMS на кадре (одна машина — одна рамка), фильтр мелочи
    tracker.py            сопоставление с прошлым кадром той же камеры, moved/displacement/shape/appearance, activity
    status.py             статусная машина ACTIVE/IDLE/PARKED/DEPARTED
    fusion.py             слияние камер: гомография на план + внешность (+ номер, если прочитан) → unit_id
    hours.py              плановые моточасы из плана и парка техники; журнал ActivityInterval; HoursBalance
    engine.py             EquipmentEngine: состояние площадки, process_frame(frame, image, detections) → обновления
  stage/                  МОДЕЛЬ Б
    quality.py            брак / ночь / дождь (капли на объективе, размытие) / снег / туман → QualityReport
    mask.py               динамическая маска фона (порт mask.py Дениса + автоинициализация по статике)
    checklist_clip.py     LOCAL: SigLIP/CLIP, сходство с позитивной/негативной формулировкой → да/нет/не уверен по порогам
    checklist_vlm.py      EXTERNAL: GLM-4.6V двухшаговый разбор (порт analyzer/prompts/vlm Никиты); LOCAL VLM по OpenAI-API (порт model_b Дениса)
    scoring.py            ответы → статусы подэтапов и доказательность этапов (порт scoring Никиты с исправлениями)
    sequence.py           монотонная хронология этапов (HMM/Витерби: без отката, без перескока), выбросы, needs_review
  plan/
    catalog.py            каталог работ из xlsx: код → название → макроэтап/подэтап (work_map.csv)
    importer.py           импорт календарного плана CSV/XLSX (многоуровневые коды → макроэтапы), демо-план
    norms.py              «этап → техника»: обязательная/допустимая/запрещённая, пары, min_count; расчёт плановых часов
  analytics/
    rules.py              все DeviationRecord (см. §9)
    timeline.py           план/факт, отставание в днях, вердикт, прогноз (порт timeline Никиты без тавтологичного автоплана)
    report.py             SiteReport для дашборда
app/                      веб-сервис FastAPI (основа — каркас Дениса, разнесённый по роутерам)
  main.py config.py db.py models.py storage.py auth.py
  services/               оркестрация: очередь обработки, конвейер кадра, настройки провайдеров, адаптеры БД ↔ core
  routers/                pages.py (HTML-оболочки), api_*.py (JSON), ingest.py (/api/ingest для камер)
  templates/ static/      UI (Jinja-оболочки + Alpine.js + fetch к JSON API; ECharts; Tailwind)
models/                   веса (в git не кладём; откуда взять — models/README.md)
reference/                checklist.json, work_map.csv, works_catalog.xlsx, work_types*.csv (Денис), legacy_dev/
simcam/                   имитатор камеры (Денис)
tools/                    сбор датасета, обучение, оценка, импорт папки/видео, засев демо
tests/core/ tests/app/    pytest; без сети и GPU (внешние вызовы — фейки/кассеты)
docs/                     ARCHITECTURE.md, methodology.md, TZ_case07.md
```

Наследие веток (`api-solution/` Никиты, `app/pipeline/` Дениса) перенесено в `core/` и `app/services/`
и удалено интегратором; исходники — в истории git (слияния `23a646d`, `a6a758d`).

Владение каталогами при параллельной работе (чужие каталоги не трогать):

| Модуль | Каталоги |
|---|---|
| Модель А | `core/equipment/**`, `tests/core/test_equipment_*.py` |
| Модель Б | `core/stage/**`, `tests/core/test_stage_*.py` |
| План + аналитика | `core/plan/**`, `core/analytics/**`, новые файлы в `reference/`, `tests/core/test_plan_*.py`, `tests/core/test_analytics_*.py` |
| Бэкенд | `app/**` кроме `app/templates`, `app/static`; `tests/app/**`, `Dockerfile`, `docker-compose.yml`, `requirements*.txt`, `.env.example`, `tools/seed_demo.py`, `tools/ingest_*.py` |
| UI | `app/templates/**`, `app/static/**`, `tools/ui_mock.py`, `tests/ui/**` |

## 3. Словарь (канон)

- **Техника**: 21 ключ из `reference/checklist.json` → `core.taxonomy.equipment()`. Восемь обязательных по ТЗ — `taxonomy.TZ_EQUIPMENT`.
  Детектор может знать не все 21 класс; его классы маппятся на эти ключи, список поддерживаемых отдаётся в `/api/settings`.
- **Этапы**: 8 макроэтапов, 32 подэтапа с кодами xlsx (`substages[].xlsx`) и признаками `active_when` / `done_when`.
- **Признаки**: 60 ключей, флаг `latching` (однажды увиденное не исчезает).
- **Работы xlsx**: `reference/work_map.csv` — каждой строке перечня организаторов сопоставлен статус
  `substage | unobservable | out_of_scope | header`, макроэтап и подэтап. Этап на UI показывается вместе с кодами работ,
  которые сейчас идут («12.3.1. Устройство котлована»).
- **Правила «этап → техника»**: `equipment_expected / optional / forbidden` этапа + пары (см. `core/plan/norms.py`):
  экскаватор↔самосвал, автобетононасос↔автобетоносмеситель, асфальтоукладчик↔каток, буровая↔автобетоносмеситель.
  Расхождения со справочником Дениса (`reference/legacy_dev/stages.csv`) сведены в `docs/methodology.md`.

## 4. Конвейер обработки кадра

```
приём (upload/zip/видео/поток /api/ingest)  → Frame в БД + файл в хранилище
  → quality.assess            ночь / дождь / брак → FrameInfo
  → МОДЕЛЬ А (каждый кадр, днём и ночью):
       detector.detect → postprocess (межклассовый NMS) → tracker (с прошлым кадром камеры)
       → зоны → fusion (unit_id, site_xy) → status → hours (ActivityInterval) → запись Detection/EquipmentUnit
  → МОДЕЛЬ Б (только usable_for_stage и не чаще 1 раза в N часов на камеру; + внеочередно при сильном изменении маски):
       mask.apply → classifier.assess → запись StageObservation
  → пересчёт площадки (с дебаунсом): sequence.infer → StageState (ручные отметки не трогаем)
       → hours.balance → rules.evaluate → report.build → запись Deviation/StageState
```

Обработка идёт в фоне (очередь в процессе, по потоку на камеру, порядок кадров камеры сохраняется).
Загрузка кадров **не зависит** от готовности провайдера: кадр сохраняется всегда, анализ — когда провайдер готов.

## 5. Модель А — детали

1. **Детекция.** LOCAL — YOLO (ultralytics), дообученная на открытых датасетах строительной техники (см. `docs/methodology.md`),
   инференс на CPU ≤ 0.5 с/кадр на ноутбуке, на GPU — десятки мс. Тёмные кадры перед детекцией усиливаются (CLAHE по L в LAB).
   EXTERNAL — GLM-4.6V: промпт со списком ключей техники и описаниями `look`, ответ JSON с рамками в нормированных координатах 0..1000.
2. **Одна машина — одна рамка** на кадре: межклассовый NMS внутри групп путаницы (`taxonomy.CONFUSABLE_GROUPS`),
   оставляем рамку с большей уверенностью.
3. **Трекинг по камере** (кадры раз в 20–30 мин, классический трекинг невозможен): венгерское сопоставление с треками прошлого
   кадра по стоимости = 1−IoU + нормированное расстояние центров + штраф за несовместимый класс. Метка трека — большинство голосов.
   `moved_since_prev` = смещение центра > max(8 px, 0.15·диагональ) ИЛИ `bbox_shape_delta` > 0.15 (работа стрелой)
   ИЛИ `appearance_delta` > порога (изменение содержимого рамки при нормированной яркости — поза ковша). Первый кадр трека — UNKNOWN.
4. **Статусы.** ACTIVE — работала в последнем интервале; IDLE — стоит < `parked_after_h` (48 ч); PARKED — стоит дольше
   (не порождает «лишнюю технику», но даёт EQUIPMENT_PARKED_ONLY); DEPARTED — не видна > `departed_after_h` (3 ч).
5. **Без двойного счёта между камерами.** Камера калибруется один раз: 4+ точки на кадре ↔ те же точки на плане площадки (метры) →
   гомография. Точка контакта рамки с землёй проецируется на план. Детекции разных камер одного временного окна (±15 мин)
   совместимого класса ближе `merge_radius_m` (по умолчанию 5 м) склеиваются (union-find); спорное решает сходство внешности
   (цветовая гистограмма рамки, опционально эмбеддинг), совпавший номер — склейка безусловно. Некалиброванные камеры
   считаются независимыми, в UI — подсказка откалибровать. Количество техники = число уникальных `unit_id`, а не рамок.
6. **Моточасы.** Плановые часы по типу на этапе = число единиц (парк площадки, вводит пользователь; по умолчанию min_count из норм)
   × рабочие дни этапа × длительность смены (10 ч) × коэффициент использования (0.7). Всё редактируется.
   Факт: если единица ACTIVE в интервале между кадрами t₀→t₁, списываем min(t₁−t₀, 45 мин) на этап, идущий по плану в этот день.
   «Полоска» = план − факт по (этап, тип).

## 6. Модель Б — детали

1. **Отбор кадров**: ночь, дождь/капли, брак — не идут; не чаще раза в `stage_every_h` (1 ч) на камеру, лучший по качеству.
2. **Динамическая маска**: статичное дольше окна (дни) считается фоном (готовые дома) и гасится; медиана серии убирает случайные
   перекрытия (проехавший кран); маска только сжимается по мере роста стройки (алгоритм Дениса), начальная — автоматически по статике
   (+ ручная правка кистью в UI).
3. **Чек-лист**: LOCAL — SigLIP: для признака позитивная и негативная формулировка, `p = σ(k·(s⁺−s⁻))`; `p ≥ yes_thr` → да,
   `p ≤ no_thr` → нет, иначе «не уверен» (пороги настраиваются, по умолчанию 0.62/0.38). EXTERNAL — GLM-4.6V, двухшаговый разбор Никиты.
   Доля «не уверен» > 0.5 → кадр в `needs_review`, предупреждение «проверьте вручную».
4. **Хронология**: HMM по «фронту» 1..8 — переходы только «остаться» / «+1» / редко «+2», откат запрещён; эмиссия —
   доказательность этапов из scoring (UNSURE не голосует). Кадры, противоречащие пути Витерби, — `rejected_outliers`
   (так отсекаются соседние стройки в кадре и одиночные галлюцинации). Latching-признак подтверждается ≥ 2 днями.

## 7. План

Входы: (1) импорт CSV/XLSX календарного графика (коды перечня организаторов, многоуровневые, даты) → макроэтапы по `work_map.csv`;
(2) ручной ввод в редакторе (этапы из каталога xlsx, даты drag-n-drop); (3) демо-план под загруженные кадры (кнопка, с явной пометкой).
Автоплан «по датам фото» (тавтология из ветки Никиты) **не используется**.

## 8. Сверка моделей и план/факт

- **Часы отработаны ≥ 100 %, а этап по модели Б не завершён/не сменился N дней** → HOURS_SPENT_NO_PROGRESS (critical при > 120 %).
- **Этап по плану идёт, тип техники обязателен, моточасов 0 за последние `idle_alert_h` рабочих часов** → EQUIPMENT_IDLE.
- **Этапы меняются позже плана** → STAGE_LATE_START / STAGE_OVERDUE с отставанием в днях; вердикт по объекту: отставание / соответствие
  (|lag| ≤ 3 дн.) / опережение; прогноз окончания по темпу в активных днях.

## 9. Отклонения (`core/analytics/rules.py`)

Каждое `DeviationRecord` содержит: понятный заголовок, объяснение («видели: …, по плану ожидается: …»), этап, камеру/зону,
**снимки-доказательства** (frame_ids, UI рисует на них рамки), стабильный `key` для дедупликации между пересчётами.
Парные правила и «нет техники» оцениваются **по окну времени** (по умолчанию 2 ч рабочего времени), а не по одному кадру.
Эталон ТЗ: этап «Устройство котлована», экскаватор работает, самосвалов нет в окне → PAIR_BROKEN, warning,
«Возможное снижение темпа: экскаватор работает без самосвалов 2 ч 10 мин, зона „Котлован“, камеры 1, 2» + 3 снимка.

## 10. Провайдеры и режимы

Настройка (таблица `settings`, UI — переключатель в шапке и страница «Настройки»):
`mode = local | external | hybrid`, `model_a = yolo | glm`, `model_b = siglip | local_vlm | glm`.
Пресеты: **local** = yolo + siglip; **external** = glm + glm; **hybrid** = yolo + glm.
Результаты хранятся с именем провайдера; смена провайдера не стирает старые, «Переанализировать» пересчитывает.
Готовность: `/api/settings` отдаёт `ready` и причину для каждого провайдера (нет ключа `ZAI_API_KEY`, нет весов, не отвечает локальная VLM).

## 11. Схема БД (SQLAlchemy, переносимые типы: `JSON`, без ARRAY/JSONB; SQLite по умолчанию, PostgreSQL в compose)

| Таблица | Поля (кратко) |
|---|---|
| `users` | id, login, password_hash |
| `sites` | id, name, address, object_type, floors_total, timezone, shift_hours, created_at |
| `cameras` | id, site_id, name, kind (upload/folder/stream/video), ingest_key, interval_min, homography JSON, calib_points JSON, image_w, image_h, last_frame_at |
| `camera_states` | camera_id, mask keys, windows, masked_ratio, retained, counters key (маска Дениса) |
| `zones` | id, site_id, camera_id, name, kind, polygon JSON |
| `frames` | id, camera_id, captured_at, key, preview_key, sha256, width, height, is_night, weather, quality_ok, reject_reason, blur, brightness, stage_used, processed_a, processed_b, meta JSON; unique(camera_id, captured_at) |
| `detections` | id, frame_id, provider, cls, conf, x, y, w, h, track_id, unit_id→equipment_units, moved, displacement_px, shape_delta, appearance_delta, activity, zone_id, site_x, site_y, extra JSON |
| `equipment_units` | id, site_id, uid, cls, label, status, first_seen, last_seen, last_moved, worked_hours, cameras JSON, plate, site_x, site_y |
| `activity_intervals` | id, unit_id, site_id, cls, stage_id, start, end, hours, frame_ids JSON |
| `stage_observations` | id, frame_id, provider, model, answers JSON, scores JSON, stage_likelihood JSON, unsure_ratio, latency_ms, cost_usd, raw JSON, created_at |
| `stage_states` | id, site_id, stage_id, status, progress, actual_start, actual_end, confidence, manual, note, updated_at |
| `plan_items` | id, site_id, stage_id, name, work_codes JSON, planned_start, planned_end, equipment JSON, planned_hours JSON, hours_manual, source |
| `site_fleet` | id, site_id, cls, count  (заявленный парк техники) |
| `deviations` | id, site_id, key (unique per site), type, severity, title, message, stage_id, camera_id, zone_id, frame_ids JSON, unit_ids JSON, started_at, last_seen_at, status (open/ack/resolved), data JSON |
| `settings` | key, value JSON |

## 12. JSON API (префикс `/api`, все ответы — JSON; авторизация — сессия после `/login`, для камер — `X-Camera-Key`)

| Метод и путь | Назначение / ответ |
|---|---|
| `GET /api/health` | `{ok, version, providers:{yolo:{ready,reason}, siglip:{…}, glm:{…}, local_vlm:{…}}}` |
| `GET/PUT /api/settings` | `{mode, model_a, model_b, thresholds:{…}, classes:[{key,name,tz,supported}]}` |
| `GET/POST /api/sites`, `GET/PATCH/DELETE /api/sites/{id}` | список/создание; карточка `{id,name,verdict,lag_days,current_stage,progress,active_units,open_deviations,last_frame_at,thumb}` |
| `GET /api/sites/{id}/overview` | всё для страницы объекта: `{site, report:{verdict,lag_days,expected_progress,actual_progress,forecast_finish,explanation[]}, stages:[{id,name,status,progress,planned_start,planned_end,actual_start,actual_end,manual,works:[{code,name}]}], equipment:[{cls,name,units,active,idle,parked,planned_hours,worked_hours,remaining_hours}], deviations:[…], cameras:[{id,name,last_frame:{id,url,captured_at},units_now}], series:{days[],expected[],actual[]}}` |
| `GET/PUT /api/sites/{id}/plan` | `[{stage_id,name,work_codes,planned_start,planned_end,equipment,planned_hours,hours_manual}]` |
| `POST /api/sites/{id}/plan/import` | multipart xlsx/csv → разобранный план + предупреждения |
| `POST /api/sites/{id}/plan/demo` | демо-план под диапазон кадров |
| `GET/PUT /api/sites/{id}/fleet` | `[{cls,count}]` |
| `PATCH /api/sites/{id}/stages/{stage_id}` | ручная отметка `{status,progress,actual_start,actual_end,note}` |
| `GET /api/sites/{id}/equipment` | единицы техники со статусами и часами; `GET /api/units/{id}` — история и интервалы |
| `GET /api/sites/{id}/deviations?status=` ; `PATCH /api/deviations/{id}` | лента отклонений; квитирование |
| `GET/POST /api/sites/{id}/cameras`, `GET/PATCH/DELETE /api/cameras/{id}` | камеры |
| `POST /api/cameras/{id}/calibration` | `{image_points:[[x,y]…], site_points:[[X,Y]…]}` → гомография, ошибка репроекции |
| `GET/POST/DELETE /api/cameras/{id}/zones` | зоны |
| `POST /api/cameras/{id}/upload` | multipart: изображения / zip / видео; `interval_min`, `start_at` для видео и файлов без даты → `{job_id}` |
| `GET /api/jobs/{id}` | прогресс `{state,total,done,errors[]}` |
| `POST /api/ingest` | камера-поток (контракт Дениса): `X-Camera-Key`, file, camera_id, captured_at, meta → 202 |
| `GET /api/cameras/{id}/frames?limit&before` ; `GET /api/frames/{id}` | кадры; детали кадра: детекции, чек-лист, качество |
| `GET /media/{key}` ; `GET /api/frames/{id}/annotated.jpg` | файлы (под авторизацией); кадр с нарисованными рамками |
| `POST /api/detect` | **контракт модели А** (PLAN §4.3): файл или `{frame_id}` → `{frame_id, detections:[Detection.to_contract()]}` |
| `POST /api/analyze` | «Проверить снимок»: файл + `provider` → детекции, чек-лист, этап, время ответа (без сохранения) |
| `POST /api/sites/{id}/reprocess` | переанализировать текущими провайдерами |
| `POST /api/demo/seed` | засеять демо-объект |

## 13. UI

Шапка: название, переключатель объекта, **переключатель «Локальные модели ⇄ Внешний API»** (показывает готовность), статус очереди.
Страницы: **Обзор** (карточки объектов с вердиктом, этапом, техникой в работе, тревогами) → **Объект** (вкладки: Сводка — вердикт
с объяснением, этапы план/факт (Гант), кривая готовности; Техника — «полоски» моточасов по типам, единицы и статусы, тепловая карта
активности; Отклонения — лента с доказательными снимками; Камеры — сетка последних кадров с рамками; План — редактор этапов/дат/парка/часов,
ручные отметки) → **Камера** (плеер кадров со шкалой времени, слои рамок и маски, панель разбора кадра, загрузка, калибровка, зоны) →
**Проверить снимок** (перетащить фото → рамки + этап за секунды, выбор провайдера) → **Настройки**.
Тёмная и светлая тема; интерфейс на русском; всё, что пользователь видит, объясняет «почему».

## 14. Запуск

- Локально: `pip install -r requirements.txt` (+ `requirements-ml.txt` для локальных моделей) → `python -m app` → http://localhost:8000.
  SQLite (`var/app.db`) и файлы (`var/storage`) по умолчанию.
- Docker: `docker compose up` — app + PostgreSQL (+ MinIO по профилю).
- Внешний API: `ZAI_API_KEY` в `.env`. Без ключа всё работает на локальных моделях.

## 15. Соглашения

Комментарии по-русски и объясняют «почему». pytest; тесты не ходят в сеть и не требуют GPU/torch (тяжёлые зависимости импортируются
лениво внутри реализаций). Никаких секретов в git. Пороги — в конфиге/настройках, а не константами в глубине кода.
