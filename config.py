# config.py
# Конфигурационный файл для нейросетевого модуля классификации неопределённостей РТК

import os
import torch

class Config:
    """Гиперпараметры модели и системы"""
    
    # ========== Параметры входных данных ==========
    WINDOW_LENGTH = 512      # Длина временного окна T (отсчётов) - оптимизированный режим (для T=1024 см. диссертацию)
    NUM_CHANNELS = 4         # Количество параметров d (напряжение, ток, температура, помехи)
    NUM_CLASSES = 5          # Количество классов C (норма, скачок напряжения, токовая перегрузка, перегрев, электропомеха)
    
    # ========== Параметры патчинга ==========
    PATCH_LEN = 32           # Длина патча L
    PATCH_STRIDE = 16        # Шаг патча S (перекрытие 50%)
    # Количество патчей K = (WINDOW_LENGTH - PATCH_LEN) // PATCH_STRIDE + 1
    
    # ========== Параметры трансформера ==========
    D_MODEL = 64             # Размерность эмбеддинга d_model (должна быть кратна NUM_CHANNELS для межканального внимания)
    NUM_HEADS = 4            # Количество голов внимания H
    NUM_LAYERS = 3           # Количество слоёв трансформера L
    DROPOUT = 0.1            # Вероятность dropout
    
    # ========== Параметры регуляризации (глава 2, формула 2.8) ==========
    MASK_PROB = 0.15         # Вероятность стохастического маскирования патчей p_mask
    LAMBDA_1 = 1e-4          # Коэффициент L2-регуляризации λ₁
    LAMBDA_2 = 0.1           # Коэффициент контрастивной потери λ₂
    TEMPERATURE = 0.07       # Температура для контрастивной потери (исправлено: 0.07 — стандартное значение)
    
    # ========== Параметры обучения ==========
    BATCH_SIZE = 32          # Размер батча
    LEARNING_RATE = 1e-3     # Скорость обучения
    NUM_EPOCHS = 100         # Количество эпох обучения
    EARLY_STOPPING_PATIENCE = 10  # Остановка при отсутствии улучшений

    # ========== Параметры для синтетических данных ==========Эксперимент 1 - увеличивал количество образцов на класс с 50 до 2000, на 2000, модель обучилась 
    # точность получилась 99% что означает, что данные слишком простые и нужно их усложнить. Этим займусь в Эксперименте 2
    SYNTHETIC_SAMPLES_PER_CLASS = 2000   # количество образцов на класс для генерации, в v1.0 ставил 24 образца на класс, теперь изменил на 50, 200, 500, 1000, 2000, для повышения точности

    # ========== Параметры аугментации (глава 3) ==========
    AUGMENTATION_COUNT = 3   # Количество аугментаций на один образец R
    # Физические шумы для аугментации (таблица 3.3)
    # ИСПРАВЛЕНО: ключи изменены на числовые индексы для удобства использования
    AUG_NOISE_STD = [11.0, 0.67, 2.0, 5.0]  # [напряжение, ток, температура, помехи]
    
    # ========== Воспроизводимость (ДОБАВЛЕНО) ==========
    RANDOM_SEED = 42         # Фиксация random seed для воспроизводимости результатов
    
    # ========== Аппаратные параметры ==========
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Автоопределение GPU/CPU
    
    # ========== Пути к файлам ==========
    MODEL_SAVE_PATH = 'models/patchtst_model.pth'
    DATA_PATH = 'data/synthetic_data.csv'
    LOG_PATH = 'logs/'
    
    @classmethod
    def get_num_patches(cls):
        """Расчёт количества патчей K"""
        return (cls.WINDOW_LENGTH - cls.PATCH_LEN) // cls.PATCH_STRIDE + 1
    
    # ДОБАВЛЕНО: метод для создания необходимых директорий
    @classmethod
    def ensure_dirs(cls):
        """Создание необходимых директорий для сохранения модели и логов"""
        os.makedirs(os.path.dirname(cls.MODEL_SAVE_PATH), exist_ok=True)
        os.makedirs(cls.LOG_PATH, exist_ok=True)
        os.makedirs('data', exist_ok=True)
    
    # ДОБАВЛЕНО: метод для фиксации random seed
    @classmethod
    def set_seed(cls):
        """Фиксация random seed для воспроизводимости"""
        import random
        import numpy as np
        
        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
        torch.manual_seed(cls.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.RANDOM_SEED)
            torch.cuda.manual_seed_all(cls.RANDOM_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @classmethod
    def print_config(cls):
        """Вывод конфигурации"""
        print("=" * 50)
        print("КОНФИГУРАЦИЯ НЕЙРОСЕТЕВОГО МОДУЛЯ PATCHTST")
        print("=" * 50)
        print(f"Длина окна T:           {cls.WINDOW_LENGTH}")
        print(f"Число параметров d:     {cls.NUM_CHANNELS}")
        print(f"Число классов C:        {cls.NUM_CLASSES}")
        print(f"Длина патча L:          {cls.PATCH_LEN}")
        print(f"Шаг патча S:            {cls.PATCH_STRIDE}")
        print(f"Число патчей K:         {cls.get_num_patches()}")
        print(f"Размер эмбеддинга d_model: {cls.D_MODEL}")
        print(f"Число голов H:          {cls.NUM_HEADS}")
        print(f"Число слоёв L:          {cls.NUM_LAYERS}")
        print(f"Устройство:             {cls.DEVICE}")
        print(f"Вероятность маскирования: {cls.MASK_PROB}")
        print(f"λ₁ (L2):                {cls.LAMBDA_1}")
        print(f"λ₂ (контраст):          {cls.LAMBDA_2}")
        print(f"Температура τ:          {cls.TEMPERATURE}")
        print(f"Random seed:            {cls.RANDOM_SEED}")
        print("=" * 50)
    
    # ДОБАВЛЕНО: проверка согласованности гиперпараметров
    @classmethod
    def validate(cls):
        """Проверка согласованности гиперпараметров"""
        assert cls.D_MODEL % cls.NUM_CHANNELS == 0, \
            f"D_MODEL ({cls.D_MODEL}) должен быть кратен NUM_CHANNELS ({cls.NUM_CHANNELS})"
        assert cls.PATCH_LEN > 0 and cls.PATCH_STRIDE > 0, "Параметры патчинга должны быть положительными"
        assert cls.PATCH_STRIDE <= cls.PATCH_LEN, "Шаг патча не должен превышать длину патча"
        assert 0 <= cls.MASK_PROB <= 1, "MASK_PROB должен быть в диапазоне [0, 1]"
        print("Проверка конфигурации пройдена успешно.")


if __name__ == '__main__':
    Config.print_config()
    Config.validate()
    Config.set_seed()
    Config.ensure_dirs()
    print("Конфигурация загружена и готова к использованию.")