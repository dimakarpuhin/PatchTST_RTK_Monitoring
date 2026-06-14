# synthetic_data.py
# Генератор синтетических данных для обучения и тестирования
# Основан на методике из главы 3 (разделы 3.1, 3.2.3)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
import os
from typing import Tuple, Optional


class SyntheticDataGenerator:
    """
    Генератор синтетических данных, имитирующих работу РТК
    Классы неопределённостей (в соответствии с главой 3):
    0 - Норма
    1 - Скачок напряжения
    2 - Токовая перегрузка
    3 - Перегрев
    4 - Электропомеха
    """
    
    def __init__(self, config):
        self.config = config
        self.window_len = config.WINDOW_LENGTH
        self.num_channels = config.NUM_CHANNELS
        self.fs = 1000  # Частота дискретизации 1000 Гц (согласно главе 3)
        
        # Номинальные значения параметров (нормальный режим)
        self.nominal = {
            'voltage': 220.0,      # В
            'current': 10.0,       # А
            'temperature': 50.0,   # °C
            'noise': 35.0          # дБ
        }
        
        # Диапазоны для аномалий (таблица 3.1 из главы 3)
        self.anomaly_ranges = {
            'voltage_surge': (0.85, 1.15),      # ±15% от номинала (187-253 В)
            'current_overload': (1.0, 2.0),      # до 200% номинала (10-20 А)
            'temperature_overheat': (20, 80),    # °C
            'noise_interference': (10, 30)       # SNR в дБ
        }
    
    def _add_noise(self, signal, channel_idx: int, noise_level: float = 0.05) -> np.ndarray:# шаг 1 увеличил значение шума в данных с 0.02 до 0.03
        #шаг 2 увеличил значение шума в данных с 0.03 до 0.04
        """
        Добавление шума к сигналу
        Args:
            signal: исходный сигнал
            channel_idx: индекс канала (0-напряжение, 1-ток, 2-температура, 3-помехи)
            noise_level: уровень шума (относительно амплитуды сигнала)
        """
        # Разный уровень шума для разных каналов
        channel_noise_levels = [0.02, 0.02, 0.01, 0.05]  # [напряжение, ток, температура, помехи]
        actual_noise = noise_level * channel_noise_levels[channel_idx]
        noise = np.random.randn(len(signal)) * np.std(signal) * actual_noise
        return signal + noise
    
    def generate_normal(self, num_samples: int) -> np.ndarray:
        """
        Генерация нормального режима работы
        X(t) = X₀(t) + ε(t), где ε(t) ~ N(0, 0.02*X₀)
        """
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            # Напряжение: синусоида 50 Гц с шумом
            voltage = self.nominal['voltage'] * (1 + 0.02 * np.random.randn())
            voltage_signal = voltage + 0.05 * voltage * np.sin(2 * np.pi * 50 * t)
            voltage_signal = self._add_noise(voltage_signal, 0, 0.02)
            
            # Ток: следует за напряжением с небольшим сдвигом
            current = self.nominal['current'] * (1 + 0.02 * np.random.randn())
            current_signal = current + 0.1 * current * np.sin(2 * np.pi * 50 * t - 0.1)
            current_signal = self._add_noise(current_signal, 1, 0.02)
            
            # Температура: медленный дрейф + шум
            temperature = self.nominal['temperature'] + 2 * np.random.randn()
            temperature_signal = temperature + 0.5 * np.sin(2 * np.pi * 0.01 * t)
            temperature_signal = self._add_noise(temperature_signal, 2, 0.01)
            
            # Уровень помех (низкий в нормальном режиме)
            noise_level = self.nominal['noise'] + 3 * np.random.randn()
            noise_signal = noise_level + np.random.randn(self.window_len) * 0.5
            noise_signal = self._add_noise(noise_signal, 3, 0.05)
            
            sample = np.column_stack([voltage_signal, current_signal, 
                                       temperature_signal, noise_signal])
            samples.append(sample)
        
        return np.array(samples)
    
    def generate_voltage_surge(self, num_samples: int) -> np.ndarray:
        """
        Генерация скачка напряжения (глава 3, раздел 3.2.3)
        U(t) = U₀·(1 + A·I(t∈[t₀, t₀+Δt])), A ∈ [0.1, 0.15]
        Диапазон: 187-253 В (соответствует ±15% от 220 В)
        """
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            # Нормальный режим
            voltage = self.nominal['voltage'] * (1 + 0.02 * np.random.randn())
            voltage_signal = voltage + 0.05 * voltage * np.sin(2 * np.pi * 50 * t)
            
            # Внезапный скачок напряжения
            surge_start = np.random.randint(int(0.3 * self.window_len), 
                                           int(0.7 * self.window_len))
            surge_duration = np.random.randint(50, 200)
            surge_amplitude = np.random.uniform(0.1, 0.15)  # 10-15% (диапазон 242-253 В)
            
            surge_end = min(surge_start + surge_duration, self.window_len)
            voltage_signal[surge_start:surge_end] *= (1 + surge_amplitude)
            
            # Добавление шума
            voltage_signal = self._add_noise(voltage_signal, 0, 0.02)
            
            # Ток следует за напряжением
            current = self.nominal['current'] * (1 + 0.02 * np.random.randn())
            current_signal = current + 0.1 * current * np.sin(2 * np.pi * 50 * t - 0.1)
            current_signal[surge_start:surge_end] *= (1 + surge_amplitude * 0.8)
            current_signal = self._add_noise(current_signal, 1, 0.02)
            
            # Температура (норма)
            temperature = self.nominal['temperature'] + 2 * np.random.randn()
            temperature_signal = temperature + 0.5 * np.sin(2 * np.pi * 0.01 * t)
            temperature_signal = self._add_noise(temperature_signal, 2, 0.01)
            
            # Уровень помех (незначительно растёт при скачке)
            noise_level = self.nominal['noise'] + 5 * np.random.randn()
            noise_signal = noise_level + np.random.randn(self.window_len) * 0.5
            noise_signal[self.window_len//3:2*self.window_len//3] += 2
            noise_signal = self._add_noise(noise_signal, 3, 0.05)
            
            sample = np.column_stack([voltage_signal, current_signal,
                                       temperature_signal, noise_signal])
            samples.append(sample)
        
        return np.array(samples)
    
    def generate_current_overload(self, num_samples: int) -> np.ndarray:
        """
        Генерация токовой перегрузки (глава 3, раздел 3.2.3)
        Ток: до 200% номинала (10-20 А)
        """
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            # Напряжение (норма)
            voltage = self.nominal['voltage'] * (1 + 0.02 * np.random.randn())
            voltage_signal = voltage + 0.05 * voltage * np.sin(2 * np.pi * 50 * t)
            voltage_signal = self._add_noise(voltage_signal, 0, 0.02)
            
            # Ток с перегрузкой
            current = self.nominal['current'] * (1 + 0.02 * np.random.randn())
            current_signal = current + 0.1 * current * np.sin(2 * np.pi * 50 * t - 0.1)
            
            # Внезапная перегрузка
            overload_start = np.random.randint(int(0.2 * self.window_len),
                                              int(0.8 * self.window_len))
            overload_duration = np.random.randint(100, 300)
            overload_factor = np.random.uniform(1.5, 2.0)  # 150-200% номинала
            
            overload_end = min(overload_start + overload_duration, self.window_len)
            current_signal[overload_start:overload_end] *= overload_factor
            
            current_signal = self._add_noise(current_signal, 1, 0.03)
            
            # ИСПРАВЛЕНО: температура растёт медленно (0.1°C/с, как в диссертации)
            temperature = self.nominal['temperature']
            # При fs=1000 Гц, dt = 0.001 с, дрейф = 0.1°C/с → 0.0001°C/отсчёт
            drift_rate = 0.0001 * (overload_factor - 1)
            temperature_signal = temperature + drift_rate * np.arange(self.window_len)
            temperature_signal += 1 * np.random.randn(self.window_len)
            temperature_signal = self._add_noise(temperature_signal, 2, 0.01)
            
            # Уровень помех
            noise_level = self.nominal['noise'] + 5 * np.random.randn()
            noise_signal = noise_level + np.random.randn(self.window_len) * 0.5
            noise_signal = self._add_noise(noise_signal, 3, 0.05)
            
            sample = np.column_stack([voltage_signal, current_signal,
                                       temperature_signal, noise_signal])
            samples.append(sample)
        
        return np.array(samples)
    
    def generate_overheat(self, num_samples: int) -> np.ndarray:
        """
        Генерация перегрева (температурный дрейф)
        Температура: 20-80°C (линейный дрейф)
        """
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            # Напряжение (норма)
            voltage = self.nominal['voltage'] * (1 + 0.02 * np.random.randn())
            voltage_signal = voltage + 0.05 * voltage * np.sin(2 * np.pi * 50 * t)
            voltage_signal = self._add_noise(voltage_signal, 0, 0.02)
            
            # Ток (норма)
            current = self.nominal['current'] * (1 + 0.02 * np.random.randn())
            current_signal = current + 0.1 * current * np.sin(2 * np.pi * 50 * t - 0.1)
            current_signal = self._add_noise(current_signal, 1, 0.02)
            
            # Температура: линейный дрейф вверх (согласно диссертации)
            start_temp = np.random.uniform(20, 40)
            end_temp = np.random.uniform(60, 80)
            temperature_signal = np.linspace(start_temp, end_temp, self.window_len)
            temperature_signal += 0.5 * np.random.randn(self.window_len)
            temperature_signal = self._add_noise(temperature_signal, 2, 0.01)
            
            # Уровень помех (незначительно растёт при нагреве)
            noise_level = self.nominal['noise'] + 8 * np.random.randn()
            noise_signal = noise_level + np.random.randn(self.window_len) * 0.5
            # Шум растёт с температурой
            temp_norm = (temperature_signal - 20) / 60
            noise_signal += 5 * temp_norm
            noise_signal = self._add_noise(noise_signal, 3, 0.05)
            
            sample = np.column_stack([voltage_signal, current_signal,
                                       temperature_signal, noise_signal])
            samples.append(sample)
        
        return np.array(samples)
    
    def generate_electromagnetic_interference(self, num_samples: int) -> np.ndarray:
        """
        Генерация электромагнитных помех (глава 3, раздел 3.2.3)
        SNR: 10-30 дБ (сильные помехи)
        Импульсные выбросы длительностью 1-5 отсчётов
        """
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            # Напряжение (норма с помехами)
            voltage = self.nominal['voltage'] * (1 + 0.02 * np.random.randn())
            voltage_signal = voltage + 0.05 * voltage * np.sin(2 * np.pi * 50 * t)
            
            # Добавление высокочастотных импульсных помех (1-5 отсчётов)
            num_spikes = np.random.randint(5, 20)
            for _ in range(num_spikes):
                spike_pos = np.random.randint(0, self.window_len)
                spike_amplitude = np.random.uniform(0.1, 0.3) * voltage
                spike_width = np.random.randint(1, 5)
                start = max(0, spike_pos - spike_width)
                end = min(self.window_len, spike_pos + spike_width)
                voltage_signal[start:end] += spike_amplitude
            
            voltage_signal = self._add_noise(voltage_signal, 0, 0.05)
            
            # Ток (аналогично)
            current = self.nominal['current'] * (1 + 0.02 * np.random.randn())
            current_signal = current + 0.1 * current * np.sin(2 * np.pi * 50 * t - 0.1)
            
            num_spikes_current = np.random.randint(3, 10)
            for _ in range(num_spikes_current):
                spike_pos = np.random.randint(0, self.window_len)
                spike_amplitude = np.random.uniform(0.1, 0.25) * current
                spike_width = np.random.randint(1, 3)
                start = max(0, spike_pos - spike_width)
                end = min(self.window_len, spike_pos + spike_width)
                current_signal[start:end] += spike_amplitude
            
            current_signal = self._add_noise(current_signal, 1, 0.05)
            
            # Температура (норма)
            temperature = self.nominal['temperature'] + 2 * np.random.randn()
            temperature_signal = temperature + 0.5 * np.sin(2 * np.pi * 0.01 * t)
            temperature_signal = self._add_noise(temperature_signal, 2, 0.01)
            
            # ИСПРАВЛЕНО: уровень помех (шум в дБ с заданным SNR)
            snr_db = np.random.uniform(10, 30)  # диапазон из диссертации (10-30 дБ)
            base_noise_level = self.nominal['noise']
            
            # Базовый шум
            noise_signal = base_noise_level + np.random.randn(self.window_len) * 3
            
            # Добавление импульсных выбросов в уровень помех
            num_spikes_noise = np.random.randint(3, 10)
            for _ in range(num_spikes_noise):
                spike_pos = np.random.randint(0, self.window_len)
                spike_amplitude = np.random.uniform(5, 15)  # дБ
                spike_width = np.random.randint(1, 3)
                start = max(0, spike_pos - spike_width)
                end = min(self.window_len, spike_pos + spike_width)
                noise_signal[start:end] += spike_amplitude
            
            # Ограничение уровня помех (не более 60 дБ)
            noise_signal = np.clip(noise_signal, 15, 60)
            noise_signal = self._add_noise(noise_signal, 3, 0.05)
            
            sample = np.column_stack([voltage_signal, current_signal,
                                       temperature_signal, noise_signal])
            samples.append(sample)
        
        return np.array(samples)
    
    def generate_dataset(self, samples_per_class: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if samples_per_class is None:
            samples_per_class = self.config.SYNTHETIC_SAMPLES_PER_CLASS
        """
        Генерация полного датасета 
        samples_per_class: количество образцов на класс (по умолчанию 24 = 120 всего)
        """
        print("=" * 50)
        print("ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОГО ДАТАСЕТА ")
        print("=" * 50)
        print(f"  Образцов на класс: {samples_per_class}")
        print(f"  Всего образцов: {samples_per_class * 5}")
        print(f"  Длина окна T: {self.window_len}")
        print(f"  Число параметров d: {self.num_channels}")
        
        # Генерация данных для каждого класса
        print("  Генерация нормального режима...")
        data_normal = self.generate_normal(samples_per_class)
        print("  Генерация скачков напряжения...")
        data_voltage = self.generate_voltage_surge(samples_per_class)
        print("  Генерация токовых перегрузок...")
        data_current = self.generate_current_overload(samples_per_class)
        print("  Генерация перегревов...")
        data_temp = self.generate_overheat(samples_per_class)
        print("  Генерация электромагнитных помех...")
        data_noise = self.generate_electromagnetic_interference(samples_per_class)
        
        # Объединение
        X = np.vstack([data_normal, data_voltage, data_current, data_temp, data_noise])
        y = np.hstack([
            np.zeros(samples_per_class),   # норма
            np.ones(samples_per_class),    # скачок напряжения
            2 * np.ones(samples_per_class), # токовая перегрузка
            3 * np.ones(samples_per_class), # перегрев
            4 * np.ones(samples_per_class)  # электропомеха
        ])
        
        # Перемешивание
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"  Готово: X.shape = {X.shape}, y.shape = {y.shape}")
        print("=" * 50)
        
        return X, y.astype(np.int64)
    
    def generate_large_dataset(self, samples_per_class: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация большого синтетического датасета (для главы 2)
        samples_per_class: количество образцов на класс (по умолчанию 2000 = 10 000 всего)
        """
        print("=" * 50)
        print("ГЕНЕРАЦИЯ БОЛЬШОГО СИНТЕТИЧЕСКОГО ДАТАСЕТА (ГЛАВА 2)")
        print("=" * 50)
        print(f"  Образцов на класс: {samples_per_class}")
        print(f"  Всего образцов: {samples_per_class * 5}")
        print(f"  Длина окна T: {self.window_len}")
        print(f"  Число параметров d: {self.num_channels}")
        
        X, y = self.generate_dataset(samples_per_class=samples_per_class)
        
        print(f"  Готово: X.shape = {X.shape}, y.shape = {y.shape}")
        print("=" * 50)
        
        return X, y
    
    def save_to_csv(self, X: np.ndarray, y: np.ndarray, filepath: str) -> pd.DataFrame:
        """
        Сохранение датасета в CSV файл
        Формат: столбцы label, t0_ch0, t0_ch1, ..., tT_chM
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Создание DataFrame
        num_samples, window_len, num_channels = X.shape
        
        # Разворачиваем временные ряды в плоскую таблицу
        data = []
        for i in range(num_samples):
            row = {'label': int(y[i])}
            for t in range(window_len):
                for c in range(num_channels):
                    row[f't{t}_ch{c}'] = X[i, t, c]
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Датасет сохранён в {filepath}")
        print(f"  Формат: {df.shape[0]} строк, {df.shape[1]} столбцов")
        return df
    
    def load_from_csv(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загрузка датасета из CSV файла
        """
        df = pd.read_csv(filepath)
        
        labels = df['label'].values
        
        # Определение размерности из данных
        # Ищем столбцы формата t0_ch0
        max_t = 0
        max_ch = 0
        for col in df.columns:
            if col.startswith('t'):
                parts = col.split('_')
                if len(parts) == 2:
                    t = int(parts[0][1:])
                    ch = int(parts[1][2:])
                    max_t = max(max_t, t)
                    max_ch = max(max_ch, ch)
        
        window_len = max_t + 1
        num_channels = max_ch + 1
        
        print(f"Загрузка данных из {filepath}")
        print(f"  Окно: {window_len} отсчётов, {num_channels} каналов")
        
        X = np.zeros((len(df), window_len, num_channels))
        
        for i in range(len(df)):
            for t in range(window_len):
                for c in range(num_channels):
                    col_name = f't{t}_ch{c}'
                    if col_name in df.columns:
                        X[i, t, c] = df.loc[i, col_name]
                    else:
                        # Если столбца нет, заполняем нулями
                        X[i, t, c] = 0.0
        
        return X, labels


class RTKDataset(Dataset):
    """PyTorch Dataset для загрузки данных"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, config, augment: bool = False):
        """
        Args:
            X: [num_samples, window_len, num_channels] - входные данные
            y: [num_samples] - метки классов
            config: объект конфигурации
            augment: применять ли аугментацию (True для обучения)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.config = config
        self.augment = augment  # флаг аугментации
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        # ИСПРАВЛЕНО: убрано self.training (у датасета нет этого атрибута)
        if self.augment:
            x = self.apply_augmentation(x)
        
        return x, y
    
    def apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Физически обоснованная аугментация (глава 3, раздел 3.2.3)
        Добавление гауссовского шума с дисперсией из таблицы 3.3
        
        В соответствии с главой 3:
        - Напряжение: σ = 11.0 В (5% от 220 В)
        - Ток: σ = 0.67 А (6.7% от 10 А)
        - Температура: σ = 2.0 °C (погрешность датчика)
        - Помехи: σ = 5.0 дБ (вариация ЭМО)
        """
        # Нормировка шума относительно номинальных значений
        '''aug_std = [
            self.config.AUG_NOISE_STD['voltage'] / 220.0,     # напряжение
            self.config.AUG_NOISE_STD['current'] / 10.0,      # ток
            self.config.AUG_NOISE_STD['temperature'] / 50.0,  # температура
            self.config.AUG_NOISE_STD['noise'] / 35.0         # помехи
        ]'''

        aug_std = [
            self.config.AUG_NOISE_STD[0] / 220.0,
            self.config.AUG_NOISE_STD[1] / 10.0,
            self.config.AUG_NOISE_STD[2] / 50.0,
            self.config.AUG_NOISE_STD[3] / 35.0
        ]
        
        noise = torch.randn_like(x)
        for c in range(self.config.NUM_CHANNELS):
            noise[:, c] *= aug_std[c]
        
        return x + noise


def create_dataloaders(config, X: np.ndarray, y: np.ndarray, 
                       train_ratio: float = 0.7, 
                       val_ratio: float = 0.15,
                       batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создание DataLoader для обучения, валидации и тестирования
    
    Args:
        config: объект конфигурации
        X: входные данные
        y: метки
        train_ratio: доля обучающей выборки
        val_ratio: доля валидационной выборки
        batch_size: размер батча (если None, используется config.BATCH_SIZE)
    """
    from sklearn.model_selection import train_test_split
    
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Разделение на train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), stratify=y, random_state=config.RANDOM_SEED
    )
    
    val_ratio_adjusted = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_adjusted), 
        stratify=y_temp, random_state=config.RANDOM_SEED
    )
    
    print(f"Разделение данных:")
    print(f"  Обучающая выборка: {len(X_train)} образцов")
    print(f"  Валидационная: {len(X_val)} образцов")
    print(f"  Тестовая: {len(X_test)} образцов")
    
    # Создание датасетов
    train_dataset = RTKDataset(X_train, y_train, config, augment=True)
    val_dataset = RTKDataset(X_val, y_val, config, augment=False)
    test_dataset = RTKDataset(X_test, y_test, config, augment=False)
    
    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Тест генератора
    config = Config
    config.set_seed()
    config.ensure_dirs()
    
    generator = SyntheticDataGenerator(config)
    
    # Тест 1: Генерация небольшого датасета (глава 3)
    print("\n" + "=" * 60)
    print("ТЕСТ 1: ГЕНЕРАЦИЯ НЕБОЛЬШОГО ДАТАСЕТА (ГЛАВА 3)")
    print("=" * 60)
    X, y = generator.generate_dataset()  # теперь использует config.SYNTHETIC_SAMPLES_PER_CLASS
    generator.save_to_csv(X, y, config.DATA_PATH)
    
    # Тест 2: Проверка через Dataset
    print("\n" + "=" * 60)
    print("ТЕСТ 2: ПРОВЕРКА DATASET")
    print("=" * 60)
    dataset = RTKDataset(X, y, config, augment=False)
    print(f"Dataset: {len(dataset)} образцов")
    sample_x, sample_y = dataset[0]
    print(f"Пример образца: x.shape = {sample_x.shape}, y = {sample_y}")
    
    # Тест 3: Проверка аугментации
    print("\n" + "=" * 60)
    print("ТЕСТ 3: ПРОВЕРКА АУГМЕНТАЦИИ")
    print("=" * 60)
    dataset_aug = RTKDataset(X, y, config, augment=True)
    sample_x_aug, _ = dataset_aug[0]
    diff = (sample_x_aug - sample_x).abs().mean().item()
    print(f"Среднее изменение после аугментации: {diff:.6f}")
    
    # Тест 4: Генерация большого датасета (глава 2)
    print("\n" + "=" * 60)
    print("ТЕСТ 4: ГЕНЕРАЦИЯ БОЛЬШОГО ДАТАСЕТА (ГЛАВА 2)")
    print("=" * 60)
    X_large, y_large = generator.generate_large_dataset(samples_per_class=2000)
    
    # Тест 5: Проверка DataLoader
    print("\n" + "=" * 60)
    print("ТЕСТ 5: ПРОВЕРКА DATALOADER")
    print("=" * 60)
    train_loader, val_loader, test_loader = create_dataloaders(config, X_large, y_large)
    batch_x, batch_y = next(iter(train_loader))
    print(f"Батч из train_loader: x.shape = {batch_x.shape}, y.shape = {batch_y.shape}")
    
    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
    print("=" * 60)