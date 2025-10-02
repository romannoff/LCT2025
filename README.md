# LCT2025 — Fetal Monitor Assistant (MVP)

**TL;DR:**
Проект — MVP веб‑платформы «цифровой ассистент» для родильных отделений. Сервис принимает потоковые и архивные данные с фетальных мониторов (частота сердечных сокращений плода — BPM, маточные сокращения — uterus), выполняет их потоковую и пакетную обработку, извлекает клинически значимые паттерны и признаки, строит краткосрочные прогнозы и предоставляет интуитивную панель/REST+WebSocket API для интеграции с оборудованием и EMR.

## Содержание репозитория

```
LCT2025/
├─ data/                   # архивные CSV-сессии и образцы
├─ models/                 # обученные/сохранённые модели (ML-core)
├─ src/                    # backend, streaming & processing logic, sensor emulator
│  ├─ app.py               # FastAPI: REST endpoints, WebSocket, ingestion, archive analysis
│  ├─ processing.py        # StreamCTGAnalyzer — потоковый CTG-анализатор
│  ├─ process_functions.py # функции извлечения признаков, детекторы паттернов
│  └─ sampler.py           # эмулятор/симулятор передачи данных (streaming)
├─ static/                 # frontend (dashboard) — HTML/CSS/JS
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
```

---

## Что реализовано

* **Приём потоковых данных:** REST‑endpoint для приёма одиночных измерений (BPM / uterus) и WebSocket для передачи обновлений клиентам в реальном времени.
* **Пакетный/архивный анализ:** endpoint для загрузки CSV (BPM и uterus) и получения итогового отчёта/анализа (аннотации событий и вероятности осложнений).
* **Потоковый анализ / детекторы:** `StreamCTGAnalyzer` и функции в `process_functions.py` извлекают признаки (вариабельность, декелерации, тахикардия/брадикардия и т.д.) в режиме стрима.
* **Эмулятор устройства:** `sampler.py` — трансляция архивных записей в поток (можно настроить скорость, chunk size) для тестирования real‑time обработки.
* **Интерфейс:** папка `static/` содержит веб dashboard (графики, автоматические аннотации) — фронтенд подключается к WebSocket для live updates.
* **API для интеграции:** REST + WebSocket, удобные для подключения оборудования и EMR.

Реализация покрывает требования MVP и основные сценарии из ТЗ; масштабирование/усиление модели и валидация требуют дополнительной работы и валидации на медицинских данных.

---

## Быстрый старт

### 1) Клонировать репозиторий

```bash
git clone https://github.com/romannoff/LCT2025.git
cd LCT2025
```

### 2) Запуск в Docker (рекомендуется, самый быстрый способ)

```bash
docker-compose up --build
```

После сборки сервис будет доступен на `http://localhost:2727` (FastAPI + frontend static). WebSocket: `ws://localhost:8000/ws/{patient_id}`.

### 3) Локальный запуск без Docker (venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# запустите uvicorn (пример)
uvicorn src.app:app --host 0.0.0.0 --port 2727 --reload
```

---

## API (основные эндпоинты)

### REST

* `POST /api/v1/measurements` — приём одного измерения (JSON)

  ```json
  {
    "patient_id": "patient_1",
    "type": "bpm",      # or "uterus"
    "value": 142.0,
    "timestamp": 1690000000.0
  }
  ```

  Возвращает `202 Accepted` и запускает фоновые worker'ы для потоковой обработки. Клиентам отправляются события через WebSocket.

* `POST /api/v1/analyze_archive/{patient_id}` — загрузка двух CSV файлов (bpm, uterus) для пакетного анализа. Формат: CSV с колонками `timestamp,value` или совместимый csv, возвращает JSON‑отчёт с обнаруженными событиями и итоговой вероятностью/рекоммендациями.

  Пример cURL:

  ```bash
  curl -X POST "http://localhost:2727/api/v1/analyze_archive/patient_1" \
    -F "bpm_file=@data/sample_bpm.csv" \
    -F "uterus_file=@data/sample_uterus.csv"
  ```

### WebSocket

* `ws://HOST:8000/ws/{patient_id}` — подписка на live‑события для конкретного пациента. Сообщения формата:

```json
{ "timestamp": 1690000000.0, "type": "update", "data": {"events": [...], "features": {...}} }
```

---

## Симулятор

* `src/sampler.py` — читает CSV из `data/` и шлёт последовательность измерений в endpoint `/api/v1/measurements` либо напрямую в очередь приложения. Параметры: скорость трансляции, интервал, patient_id. Используйте его для проверки реального времени и загрузки фронтенда.

Пример запуска:

```bash
python src/sampler.py --bpm data/sample_bpm.csv --uterus data/sample_uterus.csv --patient patient_1 --rate 4.0
```

(Параметры CLI зависят от реализации в файле.)

---

## ML / аналитический модуль

* Основная логика: `processing.py` реализует `StreamCTGAnalyzer` — потоковый анализатор CTG, который аккумулирует sliding windows, извлекает признаки и вызывает детекторы из `process_functions.py`.
* `models/` содержит обученные модели или чекпоинты (если присутствуют). Для production стоит продумать: quantization / ONNX / TensorRT для низкого потребления ресурсов и быстрого старта.

## Тестирование и валидация

* Используйте `src/sampler.py` + наборы из `data/` для тестирования.
* В случае, если необходимо протестировать работу системы на других данных, либо поместите в папку data .csv файлы с названиями bpm.csv и uterus.csv. Или загрузите данные через фронтенд
---
