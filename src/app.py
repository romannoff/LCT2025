from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import time
from collections import defaultdict
import logging
import pandas as pd
import io

from processing import StreamCTGAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# per-patient analyzers and queues
analyzers = {}
queues = defaultdict(asyncio.Queue)
clients = defaultdict(set)

class Measurement(BaseModel):
    patient_id: str
    timestamp: float
    type: str
    value: float

async def worker_loop(patient_id: str):
    """Фоновая задача для обработки данных пациента"""
    if patient_id not in analyzers:
        analyzers[patient_id] = StreamCTGAnalyzer(patient_id=patient_id)
    
    analyzer = analyzers[patient_id]
    q = queues[patient_id]

    while True:
        measurement = await q.get()
        events = analyzer.add_measurement(measurement['value'], measurement['type'])
        
        if events:
            message_to_send = {
                "timestamp": time.time(),
                "type": "update",
                "data": events
            }
            # Отправка событий всем подключенным клиентам
            for ws in list(clients[patient_id]):
                try:
                    await ws.send_json(message_to_send)
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    clients[patient_id].discard(ws)

@app.post("/api/v1/measurements")
async def ingest_measurement(measurement: Measurement, background_tasks: BackgroundTasks):
    """Эндпоинт для приема измерений"""
    if measurement.type not in ("bpm", "uterus"):
        raise HTTPException(status_code=400, detail="Type must be 'bpm' or 'uterus'")

    queues[measurement.patient_id].put_nowait(measurement.dict())
    
    if measurement.patient_id not in analyzers:
        background_tasks.add_task(worker_loop, measurement.patient_id)

    return {"status": "accepted"}

# --- НОВЫЙ ЭНДПОИНТ ДЛЯ АНАЛИЗА АРХИВА ---
@app.post("/api/v1/analyze_archive/{patient_id}")
async def analyze_archive(patient_id: str, bpm_file: UploadFile = File(...), uterus_file: UploadFile = File(...)):
    """
    Принимает архивные CSV файлы, анализирует их и возвращает итоговый отчет.
    """
    try:
        bpm_content = await bpm_file.read()
        uterus_content = await uterus_file.read()
        bpm_df = pd.read_csv(io.StringIO(bpm_content.decode('utf-8')))
        uterus_df = pd.read_csv(io.StringIO(uterus_content.decode('utf-8')))

        if 'value' not in bpm_df.columns or 'value' not in uterus_df.columns:
            raise HTTPException(status_code=400, detail="CSV файлы должны содержать колонку 'value'")

    except Exception as e:
        logger.error(f"Ошибка чтения архивных файлов: {e}")
        raise HTTPException(status_code=400, detail=f"Не удалось обработать файлы: {e}")

    analyzer = StreamCTGAnalyzer(patient_id=patient_id)
    
    min_len = min(len(bpm_df), len(uterus_df))
    # Уменьшаем количество итераций для быстрой демонстрации
    step = max(1, min_len // 1000) # Обрабатываем до 1000 точек для скорости
    for i in range(0, min_len, step):
        analyzer.add_measurement(bpm_df['value'].iloc[i], 'bpm')
        analyzer.add_measurement(uterus_df['value'].iloc[i], 'uterus')

    final_prob = analyzer._update_all_features()
    
    report = analyzer.current_features.copy()
    report["final_probability"] = final_prob
    
    # Очистка от несериализуемых объектов
    for key in ['bpm_buffer', 'uterus_buffer']:
        report.pop(key, None)

    return report

@app.websocket("/ws/{patient_id}")
async def websocket_endpoint(websocket: WebSocket, patient_id: str):
    """WebSocket для реального времени"""
    await websocket.accept()
    clients[patient_id].add(websocket)
    try:
        while True:
            # Держим соединение открытым
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients[patient_id].discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        clients[patient_id].discard(websocket)

# Статические файлы - должны быть ПОСЛЕДНИМИ
app.mount("/", StaticFiles(directory="static", html=True), name="static")