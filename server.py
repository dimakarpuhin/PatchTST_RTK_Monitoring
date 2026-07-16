# server.py
# FastAPI сервер для нейросетевого модуля классификации
# Реализует endpoints: /classify, /classify_batch, /health, /model/update, /classes

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import os
from datetime import datetime
import logging

from config import Config
from model import create_model
from data_loader import DataPreprocessor, RealtimeDataBuffer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="Нейросетевой модуль классификации неопределённостей РТК",
    description="Модифицированная архитектура PatchTST для классификации неопределённостей",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Добавление CORS middleware для поддержки клиента
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для хранения состояния
model = None
preprocessor = None
model_loaded = False
device = None

# Классы неопределённостей (в соответствии с главой 3)
CLASS_NAMES = {
    0: "Норма",
    1: "Скачок напряжения",
    2: "Токовая перегрузка",
    3: "Перегрев",
    4: "Электропомеха"
}

# Модели данных для API
class WindowData(BaseModel):
    """Входные данные для классификации одного окна"""
    window: List[List[float]]  # [window_len, num_channels]
    
    class Config:
        schema_extra = {
            "example": {
                "window": [[220.0, 10.2, 50.0, 35.0], [221.0, 10.3, 50.1, 34.8]]
            }
        }

class BatchData(BaseModel):
    """Входные данные для пакетной классификации"""
    windows: List[List[List[float]]]  # [batch_size, window_len, num_channels]

class ClassificationResponse(BaseModel):
    """Ответ сервера на запрос классификации"""
    class_id: int
    class_name: str
    probabilities: List[float]
    latency_ms: float
    timestamp: str

class BatchResponse(BaseModel):
    """Ответ сервера на пакетный запрос"""
    results: List[ClassificationResponse]
    total_latency_ms: float

class HealthResponse(BaseModel):
    """Ответ на запрос здоровья"""
    status: str
    model_loaded: bool
    device: str
    config: Dict[str, Any]
    timestamp: str

class UpdateModelResponse(BaseModel):
    """Ответ на запрос обновления модели"""
    status: str
    message: str
    timestamp: str


def load_model():
    """Загрузка обученной модели и статистик нормализации"""
    global model, preprocessor, model_loaded, device
    
    device = torch.device(Config.DEVICE)
    
    # Создание модели
    model = create_model(Config)
    
    # Загрузка весов, если есть
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Модель загружена из {Config.MODEL_SAVE_PATH}")
    else:
        logger.warning(f"Файл модели не найден ({Config.MODEL_SAVE_PATH})")
        logger.info("Используются случайные веса. Обучите модель через train.py")
    
    model.eval()
    model_loaded = True
    
    # Инициализация препроцессора
    preprocessor = DataPreprocessor(Config)
    
    # Загрузка статистик нормализации (ИСПРАВЛЕНО)
    stats_path = Config.MODEL_SAVE_PATH.replace('.pth', '_stats.npz')
    if os.path.exists(stats_path):
        preprocessor.load_stats(stats_path)
        logger.info(f"Статистики нормализации загружены из {stats_path}")
    else:
        logger.warning(f"Файл статистик не найден ({stats_path})")
        logger.info("Нормализация будет пропущена")
    
    logger.info(f"Сервер запущен на устройстве: {device}")
    return model


@app.on_event("startup")
async def startup_event():
    """Запуск при сервера"""
    print("\n" + "=" * 60)
    print("ЗАПУСК НЕЙРОСЕТЕВОГО СЕРВЕРА")
    print("=" * 60)
    load_model()
    Config.print_config()
    print("\nСервер готов к работе!")
    #print(f"Документация API: http://localhost:8000/docs")
    print("=" * 60 + "\n")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Проверка доступности сервера и состояния модели
    """
    config_dict = {
        "WINDOW_LENGTH": Config.WINDOW_LENGTH,
        "NUM_CHANNELS": Config.NUM_CHANNELS,
        "NUM_CLASSES": Config.NUM_CLASSES,
        "PATCH_LEN": Config.PATCH_LEN,
        "DEVICE": Config.DEVICE
    }
    
    return HealthResponse(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        device=str(device),
        config=config_dict,
        timestamp=datetime.now().isoformat()
    )


@app.get("/classes")
async def get_classes():
    """
    Получение списка классов неопределённостей
    """
    return {"classes": CLASS_NAMES}


@app.post("/classify", response_model=ClassificationResponse)
async def classify(data: WindowData):
    """
    Классификация одного временного окна
    
    Вход: список списков [window_len, num_channels]
    Выход: класс неопределённости и вероятности
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    start_time = time.time()
    
    try:
        # Преобразование входных данных в numpy array
        window = np.array(data.window, dtype=np.float32)
        
        # Проверка размерности
        expected_len = Config.WINDOW_LENGTH
        expected_channels = Config.NUM_CHANNELS
        
        if window.shape[0] != expected_len:
            raise HTTPException(
                status_code=400, 
                detail=f"Длина окна {window.shape[0]}, ожидается {expected_len}"
            )
        
        if len(window.shape) == 1:
            # Если передан одномерный массив, предполагаем один канал
            window = window.reshape(-1, 1)
        
        if window.shape[1] != expected_channels:
            raise HTTPException(
                status_code=400,
                detail=f"Число каналов {window.shape[1]}, ожидается {expected_channels}"
            )
        
        # Предобработка
        if preprocessor.stats is not None:
            window = preprocessor.normalize_zscore(window, fit=False)
        
        # Добавление размерности батча
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
        
        # Инференс
        with torch.no_grad():
            logits, embeddings = model(window_tensor, use_masking=False)
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
        
        # Получение результатов
        class_id = int(prediction.cpu().numpy()[0])
        probs = probabilities.cpu().numpy()[0].tolist()
        class_name = CLASS_NAMES.get(class_id, f"Класс {class_id}")
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ClassificationResponse(
            class_id=class_id,
            class_name=class_name,
            probabilities=probs,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при классификации: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify_batch", response_model=BatchResponse)
async def classify_batch(data: BatchData):
    """
    Пакетная классификация нескольких временных окон
    
    Вход: список окон [batch_size, window_len, num_channels]
    Выход: список результатов классификации
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    start_time = time.time()
    results = []
    
    try:
        # Преобразование входных данных
        windows = np.array(data.windows, dtype=np.float32)
        
        batch_size, window_len, num_channels = windows.shape
        
        # Проверка размерности
        if window_len != Config.WINDOW_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Длина окна {window_len}, ожидается {Config.WINDOW_LENGTH}"
            )
        
        if num_channels != Config.NUM_CHANNELS:
            raise HTTPException(
                status_code=400,
                detail=f"Число каналов {num_channels}, ожидается {Config.NUM_CHANNELS}"
            )
        
        # Предобработка для каждого окна
        windows_processed = []
        for i in range(batch_size):
            window = windows[i]
            if preprocessor.stats is not None:
                window = preprocessor.normalize_zscore(window, fit=False)
            windows_processed.append(window)
        
        windows_tensor = torch.FloatTensor(np.array(windows_processed)).to(device)
        
        # Инференс
        with torch.no_grad():
            logits, embeddings = model(windows_tensor, use_masking=False)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        # Формирование результатов
        for i in range(batch_size):
            class_id = int(predictions.cpu().numpy()[i])
            probs = probabilities.cpu().numpy()[i].tolist()
            class_name = CLASS_NAMES.get(class_id, f"Класс {class_id}")
            
            results.append(ClassificationResponse(
                class_id=class_id,
                class_name=class_name,
                probabilities=probs,
                latency_ms=0,
                timestamp=datetime.now().isoformat()
            ))
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        return BatchResponse(
            results=results,
            total_latency_ms=round(total_latency_ms, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при пакетной классификации: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/update", response_model=UpdateModelResponse)
async def update_model():
    """
    Обновление модели (дообучение)
    Загружает последнюю сохранённую модель из файла
    """
    global model, model_loaded
    
    try:
        if os.path.exists(Config.MODEL_SAVE_PATH):
            checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # ИСПРАВЛЕНО: установка режима eval
            model_loaded = True
            
            # Также обновляем статистики нормализации
            stats_path = Config.MODEL_SAVE_PATH.replace('.pth', '_stats.npz')
            if os.path.exists(stats_path) and preprocessor is not None:
                preprocessor.load_stats(stats_path)
            
            return UpdateModelResponse(
                status="success",
                message=f"Модель обновлена из {Config.MODEL_SAVE_PATH}",
                timestamp=datetime.now().isoformat()
            )
        else:
            return UpdateModelResponse(
                status="error",
                message=f"Файл модели не найден: {Config.MODEL_SAVE_PATH}",
                timestamp=datetime.now().isoformat()
            )
    except Exception as e:
        logger.error(f"Ошибка при обновлении модели: {str(e)}")
        return UpdateModelResponse(
            status="error",
            message=str(e),
            timestamp=datetime.now().isoformat()
        )


class RealtimeClassifier:
    """
    Класс для потоковой классификации в реальном времени
    Использует буфер для скользящего окна
    """
    
    def __init__(self, config, model, preprocessor):
        self.config = config
        self.model = model
        self.preprocessor = preprocessor
        self.buffer = RealtimeDataBuffer(config)
        self.device = next(model.parameters()).device
        
    def process_sample(self, sample):
        """
        Обработка одного отсчёта в реальном времени
        
        Args:
            sample: [num_channels] - новый отсчёт
        Returns:
            result: ClassificationResponse или None (если окно не готово)
        """
        sample = np.array(sample, dtype=np.float32).reshape(1, -1)
        ready, window = self.buffer.add_samples(sample)
        
        if not ready:
            return None
        
        # Предобработка окна
        if self.preprocessor.stats is not None:
            window = self.preprocessor.normalize_zscore(window, fit=False)
        
        # Инференс
        start_time = time.time()
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(window_tensor, use_masking=False)
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
        
        latency_ms = (time.time() - start_time) * 1000
        class_id = int(prediction.cpu().numpy()[0])
        
        return ClassificationResponse(
            class_id=class_id,
            class_name=CLASS_NAMES.get(class_id, f"Класс {class_id}"),
            probabilities=probabilities.cpu().numpy()[0].tolist(),
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )
    
    def reset(self):
        """Сброс буфера"""
        self.buffer.reset()


def start_server(host="0.0.0.0", port=8000):
    """Запуск сервера"""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()