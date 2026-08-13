# synthetic_data.py
# Генератор синтетических данных с управляемыми усложнениями

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
import os
from typing import Tuple, Optional
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class SyntheticDataGenerator:
    """
    Генератор синтетических данных
    Классы: 0-Норма, 1-Скачок, 2-Перегрузка, 3-Перегрев, 4-Помехи
    """
    
    def __init__(self, config):
        self.config = config
        self.window_len = config.WINDOW_LENGTH
        self.num_channels = config.NUM_CHANNELS
        self.fs = 1000
        
        self.nominal = {
            'voltage': 220.0,
            'current': 10.0,
            'temperature': 50.0,
            'noise': 35.0
        }
        
        self.correlation_matrix = np.array([
            [1.0, 0.6, 0.3, 0.2],
            [0.6, 1.0, 0.5, 0.3],
            [0.3, 0.5, 1.0, 0.4],
            [0.2, 0.3, 0.4, 1.0]
        ])
    
    def _apply_correlations(self, data):
        """Применение корреляций между каналами"""
        for i in range(data.shape[0]):
            L = np.linalg.cholesky(self.correlation_matrix)
            sample = data[i].T
            correlated = np.dot(L, sample)
            data[i] = correlated.T
        return data
    
    def _add_noise(self, signal, channel_idx: int, level: float = 0.03):
        """Добавление гауссовского шума"""
        channel_levels = [0.03, 0.04, 0.02, 0.10]
        noise = np.random.randn(len(signal)) * np.std(signal) * channel_levels[channel_idx] * level
        return signal + noise
    
    # ============================================================
    # БАЗОВЫЕ КЛАССЫ (без усложнений)
    # ============================================================
    
    def generate_normal(self, num_samples: int) -> np.ndarray:
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            voltage = self.nominal['voltage'] * (1 + 0.02 * np.random.randn())
            voltage_signal = voltage + 0.05 * voltage * np.sin(2 * np.pi * 50 * t)
            voltage_signal = self._add_noise(voltage_signal, 0, 0.03)
            
            current = self.nominal['current'] * (1 + 0.02 * np.random.randn())
            current_signal = current + 0.1 * current * np.sin(2 * np.pi * 50 * t - 0.1)
            current_signal = self._add_noise(current_signal, 1, 0.03)
            
            temperature = self.nominal['temperature'] + 2 * np.random.randn()
            temperature_signal = temperature + 0.5 * np.sin(2 * np.pi * 0.01 * t)
            temperature_signal = self._add_noise(temperature_signal, 2, 0.03)
            
            noise_level = self.nominal['noise'] + 3 * np.random.randn()
            noise_signal = noise_level + np.random.randn(self.window_len) * 0.5
            noise_signal = self._add_noise(noise_signal, 3, 0.03)
            
            sample = np.column_stack([voltage_signal, current_signal, temperature_signal, noise_signal])
            samples.append(sample)
        return np.array(samples)
    
    def generate_voltage_surge(self, num_samples: int) -> np.ndarray:
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            voltage = self.nominal['voltage'] * (1 + 0.02 * np.random.randn())
            voltage_signal = voltage + 0.05 * voltage * np.sin(2 * np.pi * 50 * t)
            
            start = np.random.randint(int(0.3 * self.window_len), int(0.7 * self.window_len))
            duration = np.random.randint(50, 200)
            amplitude = np.random.uniform(0.1, 0.15)
            end = min(start + duration, self.window_len)
            voltage_signal[start:end] *= (1 + amplitude)
            voltage_signal = self._add_noise(voltage_signal, 0, 0.03)
            
            current = self.nominal['current'] * (1 + 0.02 * np.random.randn())
            current_signal = current + 0.1 * current * np.sin(2 * np.pi * 50 * t - 0.1)
            current_signal[start:end] *= (1 + amplitude * 0.8)
            current_signal = self._add_noise(current_signal, 1, 0.03)
            
            temperature = self.nominal['temperature'] + 2 * np.random.randn()
            temperature_signal = temperature + 0.5 * np.sin(2 * np.pi * 0.01 * t)
            temperature_signal = self._add_noise(temperature_signal, 2, 0.03)
            
            noise_level = self.nominal['noise'] + 5 * np.random.randn()
            noise_signal = noise_level + np.random.randn(self.window_len) * 0.5
            noise_signal = self._add_noise(noise_signal, 3, 0.03)
            
            sample = np.column_stack([voltage_signal, current_signal, temperature_signal, noise_signal])
            samples.append(sample)
        return np.array(samples)
    
    def generate_current_overload(self, num_samples: int) -> np.ndarray:
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            voltage = self.nominal['voltage'] * (1 + 0.02 * np.random.randn())
            voltage_signal = voltage + 0.05 * voltage * np.sin(2 * np.pi * 50 * t)
            voltage_signal = self._add_noise(voltage_signal, 0, 0.03)
            
            current = self.nominal['current'] * (1 + 0.02 * np.random.randn())
            current_signal = current + 0.1 * current * np.sin(2 * np.pi * 50 * t - 0.1)
            
            start = np.random.randint(int(0.2 * self.window_len), int(0.8 * self.window_len))
            duration = np.random.randint(100, 300)
            factor = np.random.uniform(1.5, 2.0)
            end = min(start + duration, self.window_len)
            current_signal[start:end] *= factor
            current_signal = self._add_noise(current_signal, 1, 0.03)
            
            drift = 0.0001 * (factor - 1)
            temperature_signal = self.nominal['temperature'] + drift * np.arange(self.window_len)
            temperature_signal += 1 * np.random.randn(self.window_len)
            temperature_signal = self._add_noise(temperature_signal, 2, 0.03)
            
            noise_level = self.nominal['noise'] + 5 * np.random.randn()
            noise_signal = noise_level + np.random.randn(self.window_len) * 0.5
            noise_signal = self._add_noise(noise_signal, 3, 0.03)
            
            sample = np.column_stack([voltage_signal, current_signal, temperature_signal, noise_signal])
            samples.append(sample)
        return np.array(samples)
    
    def generate_overheat(self, num_samples: int) -> np.ndarray:
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            voltage = self.nominal['voltage'] * (1 + 0.02 * np.random.randn())
            voltage_signal = voltage + 0.05 * voltage * np.sin(2 * np.pi * 50 * t)
            voltage_signal = self._add_noise(voltage_signal, 0, 0.03)
            
            current = self.nominal['current'] * (1 + 0.02 * np.random.randn())
            current_signal = current + 0.1 * current * np.sin(2 * np.pi * 50 * t - 0.1)
            current_signal = self._add_noise(current_signal, 1, 0.03)
            
            start_temp = np.random.uniform(20, 40)
            end_temp = np.random.uniform(60, 80)
            temperature_signal = np.linspace(start_temp, end_temp, self.window_len)
            temperature_signal += 0.5 * np.random.randn(self.window_len)
            temperature_signal = self._add_noise(temperature_signal, 2, 0.03)
            
            noise_level = self.nominal['noise'] + 8 * np.random.randn()
            noise_signal = noise_level + np.random.randn(self.window_len) * 0.5
            temp_norm = (temperature_signal - 20) / 60
            noise_signal += 5 * temp_norm
            noise_signal = self._add_noise(noise_signal, 3, 0.03)
            
            sample = np.column_stack([voltage_signal, current_signal, temperature_signal, noise_signal])
            samples.append(sample)
        return np.array(samples)
    
    def generate_electromagnetic_interference(self, num_samples: int) -> np.ndarray:
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            voltage = self.nominal['voltage'] * (1 + 0.02 * np.random.randn())
            voltage_signal = voltage + 0.05 * voltage * np.sin(2 * np.pi * 50 * t)
            
            num_spikes = np.random.randint(5, 20)
            for _ in range(num_spikes):
                pos = np.random.randint(0, self.window_len)
                amp = np.random.uniform(0.1, 0.3) * voltage
                width = np.random.randint(1, 5)
                start = max(0, pos - width)
                end = min(self.window_len, pos + width)
                voltage_signal[start:end] += amp
            voltage_signal = self._add_noise(voltage_signal, 0, 0.05)
            
            current = self.nominal['current'] * (1 + 0.02 * np.random.randn())
            current_signal = current + 0.1 * current * np.sin(2 * np.pi * 50 * t - 0.1)
            num_spikes = np.random.randint(3, 10)
            for _ in range(num_spikes):
                pos = np.random.randint(0, self.window_len)
                amp = np.random.uniform(0.1, 0.25) * current
                width = np.random.randint(1, 3)
                start = max(0, pos - width)
                end = min(self.window_len, pos + width)
                current_signal[start:end] += amp
            current_signal = self._add_noise(current_signal, 1, 0.05)
            
            temperature = self.nominal['temperature'] + 2 * np.random.randn()
            temperature_signal = temperature + 0.5 * np.sin(2 * np.pi * 0.01 * t)
            temperature_signal = self._add_noise(temperature_signal, 2, 0.03)
            
            base_noise = self.nominal['noise']
            noise_signal = base_noise + np.random.randn(self.window_len) * 3
            num_spikes = np.random.randint(3, 10)
            for _ in range(num_spikes):
                pos = np.random.randint(0, self.window_len)
                amp = np.random.uniform(5, 15)
                width = np.random.randint(1, 3)
                start = max(0, pos - width)
                end = min(self.window_len, pos + width)
                noise_signal[start:end] += amp
            noise_signal = np.clip(noise_signal, 15, 60)
            noise_signal = self._add_noise(noise_signal, 3, 0.05)
            
            sample = np.column_stack([voltage_signal, current_signal, temperature_signal, noise_signal])
            samples.append(sample)
        return np.array(samples)
    
    # ============================================================
    # СЛОЖНЫЕ МЕТОДЫ (с усложнениями)
    # ============================================================
    
    def _add_complex_noise(self, signal: np.ndarray, level: float = 0.03) -> np.ndarray:
        """Цветной шум"""
        noise_type = np.random.choice(['white', 'pink', 'brown'], p=[0.3, 0.4, 0.3])
        if noise_type == 'white':
            noise = np.random.randn(len(signal))
        elif noise_type == 'pink':
            from scipy.signal import lfilter
            b = [1.0]
            a = [1.0, -0.9]
            noise = lfilter(b, a, np.random.randn(len(signal)))
            noise = noise / (noise.std() + 1e-8)
        else:
            noise = np.cumsum(np.random.randn(len(signal)))
            noise = noise / (noise.std() + 1e-8)
        return signal + noise * level * np.std(signal)
    
    def _add_drift(self, signal: np.ndarray, rate: float = 0.003) -> np.ndarray:
        """Дрейф"""
        drift = np.linspace(0, rate * len(signal), len(signal))
        return signal * (1 + drift)
    
    def _add_dropouts(self, signal: np.ndarray, prob: float = 0.01) -> np.ndarray:
        """Пропуски"""
        mask = np.random.rand(len(signal)) > prob
        masked = signal.copy()
        masked[~mask] = np.nan
        x = np.arange(len(signal))
        x_valid = x[mask]
        y_valid = signal[mask]
        if len(x_valid) > 1:
            return np.interp(x, x_valid, y_valid)
        return signal
    
    def _generate_base_complex(self, base_func, count: int, use_drift: bool = True, use_dropouts: bool = True, noise_level: float = 0.03, dropout_prob: float = 0.01) -> np.ndarray:
        """Универсальная генерация сложных образцов"""
        samples = []
        for _ in range(count):
            base = base_func(1)[0]
            for c in range(self.num_channels):
                base[:, c] = self._add_complex_noise(base[:, c], noise_level)
                if use_drift:
                    base[:, c] = self._add_drift(base[:, c])
                if use_dropouts:
                    base[:, c] = self._add_dropouts(base[:, c], prob=dropout_prob )
            samples.append(base)
        return np.array(samples)
    
    def generate_complex_dataset(self, samples_per_class: int = 200, 
                                  use_drift: bool = True, 
                                  use_dropouts: bool = True, 
                                  use_overlap: bool = True,
                                  noise_level: float = 0.03,
                                  dropout_prob: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация сложного датасета с управляемыми параметрами
        """
        print("=" * 60)
        print("ГЕНЕРАЦИЯ СЛОЖНЫХ ДАННЫХ")
        print("=" * 60)
        print(f"  Образцов на класс: {samples_per_class}")
        print(f"  Дрейф: {'Да' if use_drift else 'Нет'}")
        print(f"  Пропуски: {'Да' if use_dropouts else 'Нет'}")
        print(f"  Перекрытие: {'Да' if use_overlap else 'Нет'}")
        print(f"  Уровень шума: {noise_level}")
        print("-" * 60)
        
        all_samples = []
        all_labels = []
        
        # Базовые функции для каждого класса
        base_funcs = [
            self.generate_normal,
            self.generate_voltage_surge,
            self.generate_current_overload,
            self.generate_overheat,
            self.generate_electromagnetic_interference
        ]
        
        for cls, base_func in enumerate(base_funcs):
            # Основные образцы
            samples = self._generate_base_complex(base_func, samples_per_class, 
                                                  use_drift, use_dropouts, noise_level)
            all_samples.append(samples)
            all_labels.append(np.full(samples_per_class, cls))
            
            # Перекрытие классов
            if use_overlap:
                overlap_count = int(samples_per_class * 0.1)
                for other_cls in range(5):
                    if other_cls != cls and np.random.rand() < 0.3:
                        other_base = base_funcs[other_cls]
                        s1 = self._generate_base_complex(base_func, overlap_count, use_drift, use_dropouts, noise_level)
                        s2 = self._generate_base_complex(other_base, overlap_count, use_drift, use_dropouts, noise_level)
                        weight = np.random.uniform(0.3, 0.7)
                        mixed = weight * s1 + (1 - weight) * s2
                        all_samples.append(mixed)
                        labels = np.where(np.random.rand(overlap_count) < 0.5, cls, other_cls)
                        all_labels.append(labels)
        
        X = np.vstack(all_samples)
        y = np.hstack(all_labels)
        
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]
        
        X = self._apply_correlations(X)
        
        print(f"  Готово: X.shape = {X.shape}, y.shape = {y.shape}")
        for cls in range(5):
            count = np.sum(y == cls)
            print(f"    Класс {cls}: {count} ({count/len(y)*100:.1f}%)")
        print("=" * 60)
        
        return X, y.astype(np.int64)
    
    def generate_dataset(self, samples_per_class: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Обычная генерация (без усложнений)"""
        if samples_per_class is None:
            samples_per_class = self.config.SYNTHETIC_SAMPLES_PER_CLASS
        
        print("=" * 50)
        print("ГЕНЕРАЦИЯ ПРОСТЫХ ДАННЫХ")
        print("=" * 50)
        print(f"  Образцов на класс: {samples_per_class}")
        
        data_normal = self.generate_normal(samples_per_class)
        data_voltage = self.generate_voltage_surge(samples_per_class)
        data_current = self.generate_current_overload(samples_per_class)
        data_temp = self.generate_overheat(samples_per_class)
        data_noise = self.generate_electromagnetic_interference(samples_per_class)
        
        X = np.vstack([data_normal, data_voltage, data_current, data_temp, data_noise])
        y = np.hstack([
            np.zeros(samples_per_class),
            np.ones(samples_per_class),
            2 * np.ones(samples_per_class),
            3 * np.ones(samples_per_class),
            4 * np.ones(samples_per_class)
        ])
        
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]
        
        X = self._apply_correlations(X)
        
        print(f"  Готово: X.shape = {X.shape}, y.shape = {y.shape}")
        print("=" * 50)
        
        return X, y.astype(np.int64)



    def generate_dataset_with_channels(self, samples_per_class: int = 500, num_channels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация синтетических данных с заданным числом каналов"""
        
        print("=" * 60)
        print("ГЕНЕРАЦИЯ СИНТЕТИКИ С 5 КАНАЛАМИ")
        print("=" * 60)
        print(f"  Образцов на класс: {samples_per_class}")
        print(f"  Число каналов: {num_channels}")
        
        # Сохраняем оригинальное число каналов
        original_channels = self.num_channels
        self.num_channels = num_channels
        
        # Генерируем данные
        data_normal = self.generate_normal(samples_per_class)
        data_voltage = self.generate_voltage_surge(samples_per_class)
        data_current = self.generate_current_overload(samples_per_class)
        data_temp = self.generate_overheat(samples_per_class)
        data_noise = self.generate_electromagnetic_interference(samples_per_class)
        
        # Если каналов больше, чем 4, добавляем фиктивные каналы
        if num_channels > 4:
            extra_channels = num_channels - 4
            datasets = [data_normal, data_voltage, data_current, data_temp, data_noise]
            for i, dataset in enumerate(datasets):
                for _ in range(extra_channels):
                    extra = np.random.randn(dataset.shape[0], dataset.shape[1], 1) * 0.1
                    dataset = np.concatenate([dataset, extra], axis=2)
                datasets[i] = dataset
            data_normal, data_voltage, data_current, data_temp, data_noise = datasets
        
        X = np.vstack([data_normal, data_voltage, data_current, data_temp, data_noise])
        y = np.hstack([
            np.zeros(samples_per_class),
            np.ones(samples_per_class),
            2 * np.ones(samples_per_class),
            3 * np.ones(samples_per_class),
            4 * np.ones(samples_per_class)
        ])
        
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]
        
        # Восстанавливаем число каналов
        self.num_channels = original_channels
        
        print(f"  Готово: X.shape = {X.shape}, y.shape = {y.shape}")
        print("=" * 60)
        
        return X, y.astype(np.int64)

# ============================================================
# RTKDataset
# ============================================================

class RTKDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, config, augment: bool = False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.config = config
        self.augment = augment
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            x = self.apply_augmentation(x)
        return x, y
    
    '''def apply_augmentation(self, x):
        if torch.isnan(x).any():
            col_mean = torch.nanmean(x, dim=0)
            for c in range(self.config.NUM_CHANNELS):
                x[:, c] = torch.where(torch.isnan(x[:, c]), col_mean[c], x[:, c])
        aug_std = [0.03, 0.04, 0.02, 0.10]
        noise = torch.randn_like(x)
        for c in range(self.config.NUM_CHANNELS):
            noise[:, c] *= aug_std[c] * torch.std(x[:, c])
        return x + noise'''


    def apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Физически обоснованная аугментация
        """
        # Замена nan на среднее значение
        if torch.isnan(x).any():
            col_mean = torch.nanmean(x, dim=0)
            for c in range(self.config.NUM_CHANNELS):
                x[:, c] = torch.where(torch.isnan(x[:, c]), col_mean[c], x[:, c])
        
        # Реалистичные уровни шума для аугментации
        num_channels = self.config.NUM_CHANNELS
        
        # Базовые уровни шума для первых 4 каналов (как в синтетике)
        # [напряжение, ток, температура, помехи]
        base_aug_std = [0.03, 0.04, 0.02, 0.10]
        
        # Расширяем список под количество каналов
        if num_channels <= 4:
            aug_std = base_aug_std[:num_channels]
        else:
            # Для дополнительных каналов (5, 6, ...) используем уровень 0.05
            aug_std = base_aug_std + [0.05] * (num_channels - 4)
        
        noise = torch.randn_like(x)
        for c in range(num_channels):
            noise[:, c] *= aug_std[c] * torch.std(x[:, c])
        
        return x + noise

# ============================================================
# ФУНКЦИЯ ДЛЯ СОЗДАНИЯ DATALOADER
# ============================================================

def create_dataloaders_from_arrays(X, y, config, train_ratio=0.7, val_ratio=0.15, batch_size=None):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), stratify=y, random_state=config.RANDOM_SEED
    )
    
    val_ratio_adjusted = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_adjusted),
        stratify=y_temp, random_state=config.RANDOM_SEED
    )
    
    train_dataset = RTKDataset(X_train, y_train, config, augment=True)
    val_dataset = RTKDataset(X_val, y_val, config, augment=False)
    test_dataset = RTKDataset(X_test, y_test, config, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader



# ============================================================
# ПРОВЕРКА
# ============================================================

if __name__ == '__main__':
    from config import Config
    
    Config.set_seed()
    Config.SYNTHETIC_SAMPLES_PER_CLASS = 50
    
    gen = SyntheticDataGenerator(Config)
    
    print("\n🔬 ТЕСТ: Простые данные")
    X, y = gen.generate_dataset(50)
    print(f"X: {X.shape}, y: {y.shape}")
    
    print("\n🔬 ТЕСТ: Сложные данные (только шум)")
    X, y = gen.generate_complex_dataset(50, use_drift=False, use_dropouts=False, use_overlap=False, noise_level=0.03)
    print(f"X: {X.shape}, y: {y.shape}")