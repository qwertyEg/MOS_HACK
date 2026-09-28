# models/ — веса модели А (детектор техники)

Веса в git не кладём (`.gitignore`: `models/*.pt`, `*.onnx`, `*.engine`). Этот каталог —
место, где их ищет `core/equipment/detect_yolo.py`.

## Что ожидается

| Файл | Что это |
|---|---|
| `equipment_yolo.pt` | дообученная YOLO (ultralytics) на строительную технику |
| `equipment_classes.json` | имена классов детектора, imgsz обучения и (если нужно) перевод имён в ключи словаря |
| `yolov8s-worldv2.pt` | *необязательно*: запасной zero-shot детектор YOLO-World, если своих весов нет |

## Текущие веса (обучены на GPU-сервере)

`/root/mos_hack/models/` на сервере: `equipment_yolo.pt` (YOLO11s от COCO, imgsz 640, 32 эпохи),
`equipment_classes.json`, `metrics.json`. Val (MOCS + Roboflow xc7c + kerem, 4298 кадров):
mAP50 **0.865**, mAP50-95 0.702, precision 0.865, recall 0.808. 12 классов: `excavator, dump_truck,
bulldozer, roller, concrete_mixer, mobile_crane, tower_crane, concrete_pump, pile_driver,
wheel_loader, backhoe_loader, grader`. **Не покрыты** обязательные по ТЗ `truck` и
`crane_manipulator` (размеченных открытых данных нет; бортовые грузовики MOCS «Truck» попали в
`dump_truck`) — их даёт уточнение подтипа по кропу, см. ниже.

Скопировать на машину, где работает сервис:

```bash
scp <сервер>:/root/mos_hack/models/equipment_yolo.pt <сервер>:/root/mos_hack/models/equipment_classes.json models/
```

Переменные окружения:

- `EQUIPMENT_WEIGHTS` — путь к весам (`.pt`, а также `.onnx` / `.engine` — ultralytics грузит их тем же
  `YOLO(path)`), **или** каталог с весами и `equipment_classes.json`, **или** сам
  `equipment_classes.json` (веса — его поле `"model"` рядом с ним). Параметр `weights=` у
  `get_detector("yolo", weights=...)` понимает то же самое. Если указанного файла нет, а в том же каталоге
  лежат наши веса (по `equipment_classes.json` рядом, иначе `equipment_yolo.pt`), детектор берёт их и
  пишет об этом в `ready()` — так переживается старое имя `models/equipment.pt` из настроек веб-слоя;
- `EQUIPMENT_CLASSES` — путь к `equipment_classes.json` (по умолчанию — рядом с весами, затем здесь);
- `EQUIPMENT_THREADS` — потоков torch на CPU (ultralytics на CPU сам ставит «все ядра минус одно» —
  на общей машине это стоит ограничить);
- `EQUIPMENT_REFINE=0` — выключить уточнение подтипа грузовиков;
- `EQUIPMENT_REFINE_MODEL` — модель для уточнения (по умолчанию `google/siglip2-base-patch16-224` — та же,
  что у модели Б по умолчанию; `STAGE_CLIP_MODEL` модели Б сюда намеренно не подхватывается: формулировки
  и пороги подобраны под эту модель); `EQUIPMENT_REFINE_DEVICE` — `cpu` / `cuda` / `mps`.

## Формат `equipment_classes.json`

Ключи словаря — 21 тип из `reference/checklist.json` (`core.taxonomy.equipment()`), из них восемь
обязательны по ТЗ: `dump_truck, excavator, roller, crane_manipulator, concrete_mixer, bulldozer,
truck, mobile_crane`.

Так его пишет обучение на сервере (`tools/finalize_detector.py`) — это основной формат:

```json
{"model": "equipment_yolo.pt",
 "names": {"0": "excavator", "1": "dump_truck", "2": "bulldozer", "3": "roller", "4": "concrete_mixer",
           "5": "mobile_crane", "6": "tower_crane", "7": "concrete_pump", "8": "pile_driver",
           "9": "wheel_loader", "10": "backhoe_loader", "11": "grader"},
 "imgsz": 640, "epochs_trained": 32, "not_covered": ["truck", "crane_manipulator", "..."],
 "val_overall": {"mAP50": 0.8646, "...": "..."}}
```

Детектор берёт из него `names` (индекс → ключ), `imgsz` (размер входа сети, если не задан явно) и
`model` (имя файла весов, когда `EQUIPMENT_WEIGHTS` указывает на каталог или на json); остальные поля
(метрики, источники данных) — для людей. Допустим и короткий вариант — только имена, словарём или
списком: `{"names": ["excavator", "dump_truck", ...]}`.

Если классы названы как в датасете (MOCS, Roboflow и т. п.) — добавьте карту «класс датасета → ключ»;
`null` — класс осознанно отбрасывается (люди, крюк крана, легковые):

```json
{
  "names": {"0": "Worker", "1": "Static crane", "2": "Hanging head", "3": "Crane", "4": "Roller",
            "5": "Bulldozer", "6": "Excavator", "7": "Truck", "8": "Loader", "9": "Pump truck",
            "10": "Concrete mixer", "11": "Pile driving", "12": "Other vehicle"},
  "map": {"Worker": null, "Hanging head": null, "Other vehicle": null}
}
```

Чего нет в карте, переводится встроенной таблицей `core/equipment/classes.py::CLASS_ALIASES`
(«Static crane» → `tower_crane`, «Pump truck» → `concrete_pump`, «Loader» → `wheel_loader` …) и
русскими названиями из словаря. Неизвестный класс не роняет сервис: он отбрасывается, а в лог
пишется, какие имена не нашли ключа. Если json нет совсем — имена берутся из самих весов
(`model.names`) и переводятся той же таблицей.

## Если своих весов нет

`YoloDetector.ready()` возвращает `(True, "свои веса не найдены … работает YOLO-World …")` и
детектор переключается на YOLO-World с текстовыми классами (`detect_yolo.WORLD_PROMPTS`,
формулировки по полю `look` словаря) плюс классы-«поглотители» `car / person / van`, чтобы
легковушка не становилась грузовиком. Точность заметно ниже дообученной модели — это аварийный
режим, чтобы сервис работал из коробки. При первом запуске ultralytics скачает
`yolov8s-worldv2.pt` (нужен интернет); чтобы работать офлайн, положите файл сюда заранее.

Если не установлен сам `ultralytics` — `ready()` возвращает `(False, "не установлен ultralytics …")`,
а UI предлагает внешний API (GLM-4.6V). Тяжёлые зависимости (`torch`, `ultralytics`) — в
`requirements-ml.txt`, основному сервису и тестам они не нужны.

## Как детектор используется

- Тёмные и ночные кадры перед детекцией усиливаются (гамма + CLAHE по яркости в LAB).
- Для кадров больше 1600 px можно включить тайлинг (`YoloDetector(tile=True)`): кадр идёт кусками
  1280 px с перекрытием 20 % плюс целиком, результаты склеиваются.
- После детектора всегда работает `postprocess.clean`: одна машина — одна рамка (межклассовый NMS
  внутри групп путаницы `taxonomy.CONFUSABLE_GROUPS`), мелочь и обрезки у краёв отбрасываются,
  пороги уверенности по классам — в `EquipmentConfig.conf_by_class`.

## Уточнение подтипа грузовиков по кропу (`core/equipment/refine.py`)

Детектор не знает `truck` (бортовой, длинномер) и `crane_manipulator` (кран-манипулятор): на них он
отвечает «самосвал» или «автокран». Поэтому рамки `dump_truck / concrete_mixer / mobile_crane /
concrete_pump` дополнительно классифицируются по кропу zero-shot моделью SigLIP2
(`google/siglip2-base-patch16-224`, та же, что у модели Б) между шестью текстовыми описаниями подтипов
(`refine.DESCRIPTIONS`). Кроп дополняется до квадрата серым (иначе длинномер 3:1 при сжатии в 224×224
перестаёт быть похож на длинномер). Класс меняется только при уверенном перевесе
(`EquipmentConfig.refine_min_prob = 0.5`, `refine_margin = 0.2`); в класс, который детектор умеет сам
(самосвал → миксер), — только при перевесе `refine_margin_known = 0.8`, то есть практически никогда:
на своих классах детектор точнее zero-shot. Исходный класс, его уверенность и оценки подтипов —
в `Detection.extra["refine"]`.

Включено по умолчанию, если установлены `transformers` и `torch` (`YoloDetector(refine=False)` или
`EQUIPMENT_REFINE=0` — выключить). Модель грузится при первой рамке-грузовике; если её нет в кэше HF и
нет интернета — уточнение выключается с объяснением в `ready()`, детекция продолжается. Без
`transformers` `supported_classes` не содержит `truck` и `crane_manipulator`, а `ready()` пишет, что они
не распознаются.

Проверка на `testset/equipment` (фото с Wikimedia/Flickr, класс главной машины кадра, CPU):

| Класс (кадров) | Только YOLO | YOLO + уточнение |
|---|---|---|
| `truck` (9) | 0 | **8** (девятый кадр детектор не нашёл) |
| `crane_manipulator` (12) | 0 | **8** (3 → `truck`: стрела сложена и почти не видна; 1 остался самосвалом) |
| `mobile_crane` (14) | 12 | 12 (2 кадра детектор не нашёл) |
| `dump_truck` (12) | 11 | 11 |
| `concrete_mixer` (13) | 8 | 8 |
| все рамки группы грузовиков (56) | 31 | **47** |

Формулировки выбирались из пяти вариантов на этих же 56 рамках — оценка оптимистична, выборка мала;
первый вариант уводил 6 из 12 автокранов в манипуляторы, выбранный — ни одного.

## Скорость на CPU

Сервер (8 ядер, общий, под чужой нагрузкой), 3 потока torch, кадр 1280×720, imgsz 640:
YOLO11s — 90–160 мс на кадр; уточнение SigLIP2 — 250–350 мс на рамку-грузовик (первая — дольше:
загрузка модели и эмбеддинги текстов); движок (трекинг, склейка, часы) — ~7 мс на кадр.
Кадр с одной рамкой-грузовиком целиком — 0.33–0.54 с.

Потоки: ultralytics на CPU сам ставит torch «все ядра минус одно»; на общей машине с ограничением
CPU это превращается в многократное замедление (потоки OpenMP ждут друг друга под квотой) — задайте
`EQUIPMENT_THREADS`.
