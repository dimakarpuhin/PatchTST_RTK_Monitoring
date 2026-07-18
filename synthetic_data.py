# synthetic_data.py
# Генератор синтетических данных, максимально приближенных к реальным

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
    Генератор синтетических данных, имитирующих работу РТК
    С реалистичными перекрытиями классов, шумами и корреляциями
    """
    
    def __init__(self, config):
        self.config = config
        self.window_len = config.WINDOW_LENGTH
        self.num_channels = config.NUM_CHANNELS
        self.fs = 1000
        
        # ===== РЕАЛИСТИЧНЫЕ НОМИНАЛЬНЫЕ ЗНАЧЕНИЯ =====
        self.nominal = {
            'voltage': 220.0,
            'current': 10.0,
            'temperature': 50.0,
            'noise': 35.0
        }
        
        # ===== РЕАЛЬНАЯ МАТРИЦА КОРРЕЛЯЦИИ =====
        # Напряжение-Ток: 0.6 (при росте напряжения растёт ток)
        # Напряжение-Температура: 0.3 (слабая связь)
        # Ток-Температура: 0.5 (рост тока → нагрев)
        # Помехи влияют на всё
        self.correlation_matrix = np.array([
            [1.0,  0.6,  0.3,  0.2],   # U
            [0.6,  1.0,  0.5,  0.3],   # I
            [0.3,  0.5,  1.0,  0.4],   # T
            [0.2,  0.3,  0.4,  1.0]    # Noise
        ])
        
        # ===== РЕАЛИСТИЧНЫЕ ШУМЫ =====
        self.noise_levels = {
            'voltage': 0.03,   # 3% шума
            'current': 0.04,   # 4% шума
            'temperature': 0.02, # 2% шума
            'noise': 0.10      # 10% шума
        }
        
        # ===== РЕАЛИСТИЧНЫЕ ДРЕЙФЫ =====
        self.drift_rates = {
            'voltage': 0.0001,
            'current': 0.0002,
            'temperature': 0.0005,
            'noise': 0.0003
        }
    
    def _apply_correlations(self, data):
        """
        Применение корреляций между каналами
        Использует разложение Холецкого
        """
        num_samples = data.shape[0]
        original_shape = data.shape
        
        # Применяем корреляции к каждому образцу
        for i in range(num_samples):
            # Разложение Холецкого
            L = np.linalg.cholesky(self.correlation_matrix)
            
            # Транспонируем данные для умножения
            sample = data[i].T  # [channels, window_len]
            
            # Применяем корреляцию
            correlated = np.dot(L, sample)
            
            data[i] = correlated.T
        
        return data
    
    def _add_realistic_noise(self, signal, channel_idx):
        """Добавление реалистичного цветного шума"""
        noise_type = np.random.choice(['white', 'pink', 'brown'], p=[0.5, 0.3, 0.2])
        
        if noise_type == 'white':
            noise = np.random.randn(len(signal))
        elif noise_type == 'pink':
            # Розовый шум (1/f)
            from scipy.signal import lfilter
            b = np.array([1.0])
            a = np.array([1.0, -0.9])
            noise = lfilter(b, a, np.random.randn(len(signal)))
        else:
            # Коричневый шум (1/f²)
            noise = np.cumsum(np.random.randn(len(signal)))
            noise = noise / noise.std()
        
        level = list(self.noise_levels.values())[channel_idx]
        return signal + noise * np.std(signal) * level
    
    def _generate_sample_with_correlations(self, base_samples, labels):
        """
        Генерация образца с учётом корреляций
        """
        # Объединяем все каналы
        X = np.vstack(base_samples)
        y = np.hstack(labels)
        
        # Перемешиваем
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]
        
        # Применяем корреляции
        X = self._apply_correlations(X)
        
        return X, y

    def generate_normal(self, num_samples: int) -> np.ndarray:
        """Генерация нормального режима с вариациями"""
        samples = []
        for _ in range(num_samples):
            t = np.arange(self.window_len) / self.fs
            
            # ===== НАПРЯЖЕНИЕ =====
            voltage = self.nominal['voltage'] * (1 + np.random.normal(0, 0.02))
            # Добавляем 50 Гц + гармоники
            voltage_signal = voltage * (1 + 0.05 * np.sin(2 * np.pi * 50 * t))
            voltage_signal += 0.02 * voltage * np.sin(2 * np.pi * 100 * t + 0.5)
            voltage_signal = self._add_realistic_noise(voltage_signal, 0)
            # Медленные флуктуации
            voltage_signal *= (1 + 0.01 * np.sin(2 * np.pi * 0.01 * t))
            
            # ===== ТОК =====
            current = self.nominal['current'] * (1 + np.random.normal(0, 0.02))
            current_signal = current * (1 + 0.1 * np.sin(2 * np.pi * 50 * t - 0.1))
            current_signal += 0.03 * current * np.sin(2 * np.pi * 100 * t + 0.3)
            current_signal = self._add_realistic_noise(current_signal, 1)
            
            # ===== ТЕМПЕРАТУРА =====
            temp = self.nominal['temperature'] + np.random.normal(0, 2)
            temp_signal = temp + 0.5 * np.sin(2 * np.pi * 0.005 * t)
            temp_signal = self._add_realistic_noise(temp_signal, 2)
            
            # ===== ПОМЕХИ =====
            noise = self.nominal['noise'] + np.random.normal(0, 3)
            noise_signal = noise + np.random.randn(self.window_len) * 0.5
            noise_signal = self._add_realistic_noise(noise_signal, 3)
            
            sample = np.column_stack([voltage_signal, current_signal, 
                                      temp_signal, noise_signal])
            samples.append(sample)
        
        return np.array(samples)

    def generate_voltage_surge(self, num_samples: int) -> np.ndarray:
        """Генерация скачка напряжения с реалистичными эффектами"""
        samples = []
        for _ in range(num_samples):
            # Начинаем с нормального сигнала
            base = self.generate_normal(1)[0]
            
            # Выбираем случайный участок для скачка (от 20% до 80% окна)
            start = np.random.randint(int(0.2 * self.window_len), 
                                     int(0.6 * self.window_len))
            duration = np.random.randint(50, 150)
            end = min(start + duration, self.window_len)
            
            # Разная амплитуда скачка
            surge_amplitude = np.random.uniform(0.08, 0.18)
            
            # Плавный скачок (не мгновенный!)
            ramp = np.ones(self.window_len)
            ramp[start:end] = 1 + surge_amplitude * np.linspace(0, 1, end-start)
            
            base[:, 0] *= ramp
            base[:, 1] *= (1 + surge_amplitude * 0.6 * np.random.uniform(0.5, 1.0))
            
            samples.append(base)
        
        return np.array(samples)

    def generate_current_overload(self, num_samples: int) -> np.ndarray:
        """Генерация токовой перегрузки с нагревом"""
        samples = []
        for _ in range(num_samples):
            base = self.generate_normal(1)[0]
            
            start = np.random.randint(int(0.2 * self.window_len), 
                                     int(0.7 * self.window_len))
            duration = np.random.randint(100, 250)
            end = min(start + duration, self.window_len)
            
            overload = np.random.uniform(1.4, 2.0)
            base[start:end, 1] *= overload
            
            # Нагрев от перегрузки (реалистичная связь!)
            heating = (overload - 1.0) * 20 * np.linspace(0, 1, end-start)
            base[start:end, 2] += heating
            
            # Падение напряжения при перегрузке
            base[start:end, 0] *= (1 - (overload - 1.0) * 0.05)
            
            samples.append(base)
        
        return np.array(samples)

    def generate_overheat(self, num_samples: int) -> np.ndarray:
        """Генерация перегрева с ростом тока"""
        samples = []
        for _ in range(num_samples):
            base = self.generate_normal(1)[0]
            
            start = np.random.randint(50, 100)
            end = self.window_len - np.random.randint(50, 100)
            
            # Линейный рост температуры
            temp_start = np.random.uniform(20, 40)
            temp_end = np.random.uniform(60, 80)
            base[start:end, 2] = np.linspace(temp_start, temp_end, end-start)
            
            # Сопротивление растёт с температурой → ток падает
            temp_norm = (base[start:end, 2] - 20) / 60
            base[start:end, 1] *= (1 - 0.2 * temp_norm)
            
            # Напряжение немного падает
            base[start:end, 0] *= (1 - 0.05 * temp_norm)
            
            samples.append(base)
        
        return np.array(samples)

    def generate_electromagnetic_interference(self, num_samples: int) -> np.ndarray:
        """Генерация электромагнитных помех"""
        samples = []
        for _ in range(num_samples):
            base = self.generate_normal(1)[0]
            
            # Количество импульсных помех
            num_spikes = np.random.randint(5, 25)
            
            for _ in range(num_spikes):
                pos = np.random.randint(0, self.window_len)
                amplitude = np.random.uniform(0.1, 0.35)
                width = np.random.randint(1, 5)
                
                start = max(0, pos - width)
                end = min(self.window_len, pos + width)
                
                # Помехи влияют на ВСЕ каналы!
                base[start:end, :] += amplitude * np.random.randn(end-start, 4) * np.std(base[start:end, :], axis=0)
            
            # Добавляем высокочастотную модуляцию
            freq = np.random.uniform(100, 500)
            modulation = 0.1 * np.sin(2 * np.pi * freq * np.arange(self.window_len) / self.fs)
            base[:, :] *= (1 + modulation[:, np.newaxis])
            
            samples.append(base)
        
        return np.array(samples)

    def generate_mixed_sample(self, main_type: int, second_type: int) -> np.ndarray:
        """
        Генерация смешанного образца (перекрытие классов)
        """
        # Генерация основного класса
        if main_type == 0:
            main = self.generate_normal(1)[0]
        elif main_type == 1:
            main = self.generate_voltage_surge(1)[0]
        elif main_type == 2:
            main = self.generate_current_overload(1)[0]
        elif main_type == 3:
            main = self.generate_overheat(1)[0]
        else:
            main = self.generate_electromagnetic_interference(1)[0]
        
        # Генерация второго класса
        if second_type == 0:
            second = self.generate_normal(1)[0]
        elif second_type == 1:
            second = self.generate_voltage_surge(1)[0]
        elif second_type == 2:
            second = self.generate_current_overload(1)[0]
        elif second_type == 3:
            second = self.generate_overheat(1)[0]
        else:
            second = self.generate_electromagnetic_interference(1)[0]
        
        # Смешивание с разными весами
        weight = np.random.uniform(0.3, 0.7)
        mixed = weight * main + (1 - weight) * second
        
        return mixed

    def generate_dataset(self, samples_per_class: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация полного датасета с перекрытием классов"""
        if samples_per_class is None:
            samples_per_class = self.config.SYNTHETIC_SAMPLES_PER_CLASS
        
        print("=" * 60)
        print("ГЕНЕРАЦИЯ РЕАЛИСТИЧНЫХ СИНТЕТИЧЕСКИХ ДАННЫХ")
        print("=" * 60)
        print(f"  Образцов на класс: {samples_per_class}")
        
        all_samples = []
        all_labels = []
        
        # Для каждого класса
        for cls in range(5):
            # Основные образцы
            if cls == 0:
                main_samples = self.generate_normal(samples_per_class)
            elif cls == 1:
                main_samples = self.generate_voltage_surge(samples_per_class)
            elif cls == 2:
                main_samples = self.generate_current_overload(samples_per_class)
            elif cls == 3:
                main_samples = self.generate_overheat(samples_per_class)
            else:
                main_samples = self.generate_electromagnetic_interference(samples_per_class)
            
            all_samples.append(main_samples)
            all_labels.append(np.full(samples_per_class, cls))
            
            # ===== ПЕРЕКРЫТИЕ КЛАССОВ (50% от основного количества) =====
            overlap_count = max(1, samples_per_class // 2)
            
            # Смешиваем с соседними классами
            for other_cls in range(5):
                if other_cls == cls:
                    continue
                if np.random.rand() < 0.3:  # 30% вероятность смешивания
                    mixed_samples = []
                    for _ in range(overlap_count):
                        mixed = self.generate_mixed_sample(cls, other_cls)
                        mixed_samples.append(mixed)
                    
                    mixed_samples = np.array(mixed_samples)
                    all_samples.append(mixed_samples)
                    
                    # Метка: с вероятностью 50% - основной класс, 50% - второй
                    labels = np.where(np.random.rand(overlap_count) < 0.5, cls, other_cls)
                    all_labels.append(labels)
        
        # Объединение
        X = np.vstack(all_samples)
        y = np.hstack(all_labels)
        
        # Перемешивание
        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]
        
        # Применяем корреляции ко всему датасету
        X = self._apply_correlations(X)
        
        print(f"  Готово: X.shape = {X.shape}, y.shape = {y.shape}")
        print(f"  Распределение меток:")
        for cls in range(5):
            count = np.sum(y == cls)
            print(f"    Класс {cls}: {count} ({count/len(y)*100:.1f}%)")
        print("=" * 60)
        
        return X, y.astype(np.int64)

    def add_real_noise_from_file(self, X, noise_file):
        """Добавление реального шума из файла"""
        if not os.path.exists(noise_file):
            print(f"Файл шума {noise_file} не найден")
            return X
        
        noise_data = np.load(noise_file)
        noise_idx = np.random.randint(0, len(noise_data) - self.window_len)
        noise = noise_data[noise_idx:noise_idx + self.window_len]
        
        # Нормализуем шум к уровню сигнала
        noise_scale = np.std(X, axis=(0, 1)) / np.std(noise, axis=0)
        noise = noise * noise_scale[np.newaxis, :]
        
        return X + noise * 0.3  # 30% шума от уровня сигнала



# ... весь твой код SyntheticDataGenerator ...

# ============================================================
# RTKDataset - для совместимости с train.py
# ============================================================

class RTKDataset(Dataset):
    """PyTorch Dataset для загрузки данных"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, config, augment: bool = False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.config = config
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            x = self.apply_augmentation(x)
        
        return x, y
    
    def apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any():
            col_mean = torch.nanmean(x, dim=0)
            for c in range(self.config.NUM_CHANNELS):
                x[:, c] = torch.where(torch.isnan(x[:, c]), col_mean[c], x[:, c])
        
        aug_std = [0.03, 0.04, 0.02, 0.10]
        noise = torch.randn_like(x)
        for c in range(self.config.NUM_CHANNELS):
            noise[:, c] *= aug_std[c] * torch.std(x[:, c])
        
        return x + noise


def create_dataloaders_from_arrays(X, y, config, train_ratio=0.7, val_ratio=0.15, batch_size=None):
    """Создание DataLoader из массивов numpy"""
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
    
    print(f"  Обучающая: {len(train_dataset)} образцов")
    print(f"  Валидационная: {len(val_dataset)} образцов")
    print(f"  Тестовая: {len(test_dataset)} образцов")
    
    return train_loader, val_loader, test_loader



# ============================================================
# ПРОВЕРКА
# ============================================================
if __name__ == '__main__':
    from config import Config
    
    Config.set_seed()
    Config.SYNTHETIC_SAMPLES_PER_CLASS = 50
    
    generator = SyntheticDataGenerator(Config)
    X, y = generator.generate_dataset(samples_per_class=50)
    
    print(f"\n📊 Статистика данных:")
    print(f"  min: {X.min():.4f}")
    print(f"  max: {X.max():.4f}")
    print(f"  mean: {X.mean():.4f}")
    print(f"  std: {X.std():.4f}")
    
    # Проверка корреляций
    X_flat = X.reshape(-1, 4)
    corr = np.corrcoef(X_flat.T)
    print(f"\n📊 Реальная матрица корреляции:")
    print(corr.round(3))