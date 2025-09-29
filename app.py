from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
import time
from collections import defaultdict
import logging

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
            events["timestamp"] = time.time()
            # Отправка событий всем подключенным клиентам
            for ws in list(clients[patient_id]):
                try:
                    await ws.send_json(events)
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    clients[patient_id].discard(ws)

@app.post("/api/v1/measurements")
async def ingest_measurement(measurement: Measurement, background_tasks: BackgroundTasks):
    """Эндпоинт для приема измерений"""
    if measurement.type not in ("bpm", "uterus"):
        raise HTTPException(status_code=400, detail="Type must be 'bpm' or 'uterus'")

    # Добавляем измерение в очередь пациента
    queues[measurement.patient_id].put_nowait(measurement.dict())
    
    # Запускаем worker если его еще нет
    if measurement.patient_id not in analyzers:
        background_tasks.add_task(worker_loop, measurement.patient_id)

    return {"status": "accepted"}

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