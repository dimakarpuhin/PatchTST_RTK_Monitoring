# synthetic_pm_like.py
# Генератор синтетических данных, максимально похожих на PM данные
# 5 каналов, 500 отсчётов, бинарная классификация

import numpy as np
from config import Config

class SyntheticPMLikeGenerator:
    """
    Генератор синтетических данных, имитирующих структуру PM данных
    - 5 каналов (температура воздуха, температура процесса, скорость, момент, износ)
    - Длина окна: 500
    - Бинарная классификация: 0 - норма, 1 - отказ
    """
    
    def __init__(self, config):
        self.config = config
        self.window_len = 500
        self.num_channels = 5
        self.fs = 1  # условная частота
        
        # Номинальные значения (как в PM данных)
        self.nominal = {
            'air_temp': 300.0,      # K
            'process_temp': 310.0,  # K
            'speed': 1500.0,        # rpm
            'torque': 40.0,         # Nm
            'tool_wear': 100.0      # min
        }
        
        # Диапазоны вариации для нормального режима
        self.normal_ranges = {
            'air_temp': (295, 305),
            'process_temp': (305, 315),
            'speed': (1200, 1800),
            'torque': (20, 60),
            'tool_wear': (50, 150)
        }
    
    def _generate_normal(self, count: int) -> np.ndarray:
        """Генерация нормального режима (класс 0)"""
        samples = []
        for _ in range(count):
            # Случайные значения в нормальных диапазонах
            air_temp = np.random.uniform(*self.normal_ranges['air_temp'])
            process_temp = np.random.uniform(*self.normal_ranges['process_temp'])
            speed = np.random.uniform(*self.normal_ranges['speed'])
            torque = np.random.uniform(*self.normal_ranges['torque'])
            tool_wear = np.random.uniform(*self.normal_ranges['tool_wear'])
            
            # Добавляем шум и тренды
            t = np.arange(self.window_len)
            
            # Каждый параметр имеет свои колебания
            air_signal = air_temp + 2 * np.sin(2 * np.pi * 0.01 * t) + np.random.randn(self.window_len) * 1.5
            process_signal = process_temp + 1.5 * np.sin(2 * np.pi * 0.008 * t) + np.random.randn(self.window_len) * 1.0
            speed_signal = speed + 50 * np.sin(2 * np.pi * 0.005 * t) + np.random.randn(self.window_len) * 20
            torque_signal = torque + 3 * np.sin(2 * np.pi * 0.01 * t + 0.5) + np.random.randn(self.window_len) * 2
            wear_signal = tool_wear + 0.5 * t / self.window_len * 50 + np.random.randn(self.window_len) * 5
            
            sample = np.column_stack([air_signal, process_signal, speed_signal, torque_signal, wear_signal])
            samples.append(sample)
        
        return np.array(samples)
    
    def _generate_failure(self, count: int) -> np.ndarray:
        """
        Генерация отказов (класс 1)
        Разные типы отказов, но все метятся как класс 1
        """
        samples = []
        failure_types = ['temp_rise', 'speed_drop', 'torque_spike', 'wear_accel']
        
        for _ in range(count):
            # Начинаем с нормального сигнала
            base = self._generate_normal(1)[0]
            
            failure_type = np.random.choice(failure_types)
            
            if failure_type == 'temp_rise':
                # Рост температуры (как при перегреве)
                start = np.random.randint(100, 200)
                end = np.random.randint(300, self.window_len)
                base[start:end, 0] += np.linspace(0, 15, end-start)
                base[start:end, 1] += np.linspace(0, 10, end-start)
                
            elif failure_type == 'speed_drop':
                # Падение скорости
                start = np.random.randint(100, 200)
                end = np.random.randint(300, self.window_len)
                base[start:end, 2] *= np.linspace(1, 0.7, end-start)
                
            elif failure_type == 'torque_spike':
                # Скачок момента
                pos = np.random.randint(100, self.window_len - 100)
                width = np.random.randint(20, 60)
                base[pos-pos//2:pos+width//2, 3] += np.random.uniform(20, 50)
                
            elif failure_type == 'wear_accel':
                # Ускоренный износ
                start = np.random.randint(100, 200)
                base[start:, 4] += np.linspace(0, 80, self.window_len - start)
            
            # Добавляем шум после аномалии
            base += np.random.randn(self.window_len, self.num_channels) * 1.0
            
            samples.append(base)
        
        return np.array(samples)
    
    def generate_dataset(self, samples_per_class: int = 500) -> tuple:
        """
        Генерация датасета с балансом классов
        samples_per_class: количество образцов каждого класса
        """
        print("=" * 60)
        print("ГЕНЕРАЦИЯ СИНТЕТИКИ, ПОХОЖЕЙ НА PM")
        print("=" * 60)
        print(f"  Образцов на класс: {samples_per_class}")
        print(f"  Всего образцов: {samples_per_class * 2}")
        print(f"  Каналов: {self.num_channels}")
        print(f"  Длина окна: {self.window_len}")
        
        # Генерация классов
        X_normal = self._generate_normal(samples_per_class)
        X_failure = self._generate_failure(samples_per_class)
        
        X = np.vstack([X_normal, X_failure])
        y = np.hstack([
            np.zeros(samples_per_class),
            np.ones(samples_per_class)
        ])
        
        # Перемешивание
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]
        
        print(f"  Готово: X.shape = {X.shape}, y.shape = {y.shape}")
        print(f"  Класс 0: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
        print(f"  Класс 1: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
        print("=" * 60)
        
        return X, y.astype(np.int64)
    
    def generate_dataset_with_channels(self, samples_per_class: int = 500, 
                                        num_channels: int = 5) -> tuple:
        """Универсальный метод для совместимости с двухэтапным обучением"""
        # Настраиваем число каналов
        self.num_channels = num_channels
        return self.generate_dataset(samples_per_class)


# ============================================================
# ПРОВЕРКА
# ============================================================

if __name__ == '__main__':
    generator = SyntheticPMLikeGenerator(Config)
    X, y = generator.generate_dataset(samples_per_class=100)
    
    print(f"\n📊 Статистика данных:")
    print(f"  min: {X.min():.4f}")
    print(f"  max: {X.max():.4f}")
    print(f"  mean: {X.mean():.4f}")
    print(f"  std: {X.std():.4f}")
    
    # Проверка корреляций (должны быть похожи на реальные)
    X_flat = X.reshape(-1, 5)
    corr = np.corrcoef(X_flat.T)
    print(f"\n📊 Корреляционная матрица (5 каналов):")
    print(corr.round(3))