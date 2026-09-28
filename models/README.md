# models/ — веса модели А (детектор техники)

Веса в git не кладём (`.gitignore`: `models/*.pt`, `*.onnx`, `*.engine`). Этот каталог —
место, где их ищет `core/equipment/detect_yolo.py`.

## Что ожидается

| Файл | Что это |
|---|---|
| `equipment_yolo.pt` | дообученная YOLO (ultralytics) на строительную технику |
| `equipment_classes.json` | имена классов детектора и, если нужно, их перевод в ключи словаря |
| `yolov8s-worldv2.pt` | *необязательно*: запасной zero-shot детектор YOLO-World, если своих весов нет |

Обучение идёт на GPU-сервере (отдельный агент/участник). Готовые веса появляются там как
`/root/mos_hack/models/equipment_yolo.pt` и `/root/mos_hack/models/equipment_classes.json`;
скопировать сюда:

```bash
scp <сервер>:/root/mos_hack/models/equipment_yolo.pt <сервер>:/root/mos_hack/models/equipment_classes.json models/
```

Путь можно переопределить переменными окружения:

- `EQUIPMENT_WEIGHTS` — путь к весам (`.pt`, а также `.onnx` / `.engine`, если их экспортировали —
  ultralytics грузит их тем же `YOLO(path)`; для ноутбука без GPU ONNX обычно быстрее);
- `EQUIPMENT_CLASSES` — путь к `equipment_classes.json` (по умолчанию ищется рядом с весами,
  затем здесь).

## Формат `equipment_classes.json`

Ключи словаря — 21 тип из `reference/checklist.json` (`core.taxonomy.equipment()`), из них восемь
обязательны по ТЗ: `dump_truck, excavator, roller, crane_manipulator, concrete_mixer, bulldozer,
truck, mobile_crane`.

Если при обучении классы уже названы ключами словаря — достаточно списка имён:

```json
{"names": {"0": "excavator", "1": "dump_truck", "2": "bulldozer", "3": "roller",
           "4": "concrete_mixer", "5": "truck", "6": "mobile_crane", "7": "crane_manipulator"}}
```

(допустим и список: `{"names": ["excavator", "dump_truck", ...]}`).

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
