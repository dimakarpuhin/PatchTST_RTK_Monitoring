# data_loader.py
# Загрузка данных, предобработка (нормализация, фильтрация)
# Реализация нормализации (Z-score) и медианной фильтрации из главы 3

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
import os
from scipy import signal  # ИСПРАВЛЕНО: импорт вынесен в начало файла


class DataPreprocessor:
    """
    Класс для предобработки данных (нормализация, фильтрация)
    
    Реализует методы, описанные в главе 3:
    - Z-score нормализация (раздел 3.2.2)
    - Медианная фильтрация выбросов (раздел 3.2.2)
    - Удаление дрейфа (тренда)
    """
    
    def __init__(self, config):
        self.config = config
        self.stats = None  # Для хранения статистик нормализации (mean, std)
        
    def normalize_zscore(self, data, fit=True):
        """
        Z-score нормализация (глава 3, раздел 3.2.2)
        x_tilde = (x_t - μ) / σ
        
        Args:
            data: [num_samples, window_len, num_channels] или [window_len, num_channels]
            fit: True - вычислять статистики, False - использовать сохранённые
        Returns:
            normalized: нормализованные данные
        """
        if fit:
            # Вычисление статистик (формулы из главы 3)
            if len(data.shape) == 3:
                # Многомерный случай
                num_samples, window_len, num_channels = data.shape
                means = np.mean(data.reshape(-1, num_channels), axis=0)
                stds = np.std(data.reshape(-1, num_channels), axis=0)
            else:
                # Одномерный случай
                means = np.mean(data, axis=0)
                stds = np.std(data, axis=0)
            
            self.stats = {'mean': means, 'std': stds}
        else:
            if self.stats is None:
                raise ValueError("Статистики не вычислены. Сначала вызовите normalize_zscore с fit=True")
            means = self.stats['mean']
            stds = self.stats['std']
        
        # Защита от деления на ноль
        stds[stds < 1e-8] = 1.0
        
        normalized = (data - means) / stds
        return normalized
    
    def median_filter(self, data, window_size=5, threshold=3):
        """
        Медианная фильтрация выбросов (глава 3, раздел 3.2.2)
        
        Алгоритм:
        1. Применение медианного фильтра для сглаживания
        2. Если |x_t - x̂_t| > threshold * σ_j, то x_t заменяется на x̂_t
        
        Args:
            data: [window_len, num_channels] или [num_samples, window_len, num_channels]
            window_size: размер окна медианного фильтра W (должен быть нечётным)
            threshold: порог для обнаружения выбросов (3σ по умолчанию)
        Returns:
            filtered: отфильтрованные данные
        """
        # Проверка, что window_size нечётный
        if window_size % 2 == 0:
            window_size += 1
            print(f"Предупреждение: window_size изменён на {window_size} (должен быть нечётным)")
        
        if len(data.shape) == 2:
            # Одномерный случай: [window_len, num_channels]
            filtered = np.zeros_like(data)
            for c in range(data.shape[1]):
                # Медианная фильтрация
                filtered[:, c] = signal.medfilt(data[:, c], kernel_size=window_size)
                
                # ИСПРАВЛЕНО: сравнение с σ_j исходного сигнала
                std_signal = np.std(data[:, c])
                if std_signal < 1e-8:
                    std_signal = 1.0
                diff = np.abs(data[:, c] - filtered[:, c])
                outliers = diff > threshold * std_signal
                filtered[outliers, c] = filtered[outliers, c]
        else:
            # Многомерный случай: [num_samples, window_len, num_channels]
            num_samples, window_len, num_channels = data.shape
            filtered = np.zeros_like(data)
            for i in range(num_samples):
                for c in range(num_channels):
                    filtered[i, :, c] = signal.medfilt(data[i, :, c], kernel_size=window_size)
                    
                    std_signal = np.std(data[i, :, c])
                    if std_signal < 1e-8:
                        std_signal = 1.0
                    diff = np.abs(data[i, :, c] - filtered[i, :, c])
                    outliers = diff > threshold * std_signal
                    filtered[i, outliers, c] = filtered[i, outliers, c]
        
        return filtered
    
    def remove_drift(self, data, order=1):
        """
        Удаление тренда (дрейфа) из сигнала
        
        Args:
            data: [window_len, num_channels] или [num_samples, window_len, num_channels]
            order: порядок полинома для удаления тренда (1 - линейный, 2 - квадратичный)
        Returns:
            detrended: данные без тренда
        """
        if len(data.shape) == 2:
            # Одномерный случай
            if order == 1:
                detrended = signal.detrend(data, axis=0, type='linear')
            else:
                # Полиномиальное детрендирование
                x = np.arange(data.shape[0])
                detrended = np.zeros_like(data)
                for c in range(data.shape[1]):
                    coeffs = np.polyfit(x, data[:, c], order)
                    trend = np.polyval(coeffs, x)
                    detrended[:, c] = data[:, c] - trend
        else:
            # Многомерный случай
            num_samples, window_len, num_channels = data.shape
            detrended = np.zeros_like(data)
            for i in range(num_samples):
                if order == 1:
                    detrended[i] = signal.detrend(data[i], axis=0, type='linear')
                else:
                    x = np.arange(window_len)
                    for c in range(num_channels):
                        coeffs = np.polyfit(x, data[i, :, c], order)
                        trend = np.polyval(coeffs, x)
                        detrended[i, :, c] = data[i, :, c] - trend
        
        return detrended
    
    def prepare_window(self, data, window_len, stride=None):
        """
        Разбиение длинного временного ряда на окна (для скользящего режима)
        
        Args:
            data: [total_len, num_channels] - длинный временной ряд
            window_len: длина окна T
            stride: шаг окна (по умолчанию = window_len // 2)
        Returns:
            windows: [num_windows, window_len, num_channels]
        """
        if stride is None:
            stride = window_len // 2
        
        total_len, num_channels = data.shape
        num_windows = (total_len - window_len) // stride + 1
        
        windows = []
        for i in range(num_windows):
            start = i * stride
            end = start + window_len
            windows.append(data[start:end, :])
        
        return np.array(windows)
    
    def save_stats(self, filepath):
        """
        Сохранение статистик нормализации для использования в инференсе
        
        Args:
            filepath: путь для сохранения (.npz файл)
        """
        if self.stats is None:
            raise ValueError("Статистики не вычислены. Сначала вызовите normalize_zscore с fit=True")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, mean=self.stats['mean'], std=self.stats['std'])
        print(f"Статистики нормализации сохранены: {filepath}")
    
    def load_stats(self, filepath):
        """
        Загрузка статистик нормализации для инференса
        
        Args:
            filepath: путь к .npz файлу со статистиками
        """
        stats = np.load(filepath)
        self.stats = {'mean': stats['mean'], 'std': stats['std']}
        print(f"Статистики нормализации загружены: {filepath}")
        print(f"  mean: {self.stats['mean']}")
        print(f"  std: {self.stats['std']}")


class DataLoaderFromFiles:
    """
    Загрузка данных из CSV и JSON файлов
    """
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
    
    def load_from_csv(self, filepath, has_labels=True):
        """
        Загрузка данных из CSV файла
        
        Формат CSV:
        - Если has_labels=True: первый столбец 'label', затем временные ряды
        - Формат временных рядов: t0_ch0, t0_ch1, ..., tT_chM
        
        Args:
            filepath: путь к CSV файлу
            has_labels: наличие столбца с метками
        Returns:
            X: [num_samples, window_len, num_channels]
            labels: [num_samples] или None
        """
        df = pd.read_csv(filepath)
        
        if has_labels and 'label' in df.columns:
            labels = df['label'].values
            df = df.drop('label', axis=1)
        else:
            labels = None
        
        # Определение размерности
        num_samples = len(df)
        num_features = len(df.columns)
        
        # Восстановление временных рядов
        max_t = 0
        for col in df.columns:
            if col.startswith('t'):
                t = int(col.split('_')[0][1:])
                max_t = max(max_t, t)
        
        window_len = max_t + 1
        
        # ИСПРАВЛЕНО: проверка кратности
        if num_features % window_len != 0:
            raise ValueError(
                f"Количество столбцов ({num_features}) не кратно длине окна ({window_len}). "
                f"Возможно, файл повреждён или имеет другой формат."
            )
        num_channels = num_features // window_len
        
        X = np.zeros((num_samples, window_len, num_channels))
        
        for i in range(num_samples):
            for t in range(window_len):
                for c in range(num_channels):
                    col_name = f't{t}_ch{c}'
                    if col_name in df.columns:
                        X[i, t, c] = df.loc[i, col_name]
                    else:
                        # ИСПРАВЛЕНО: предупреждение, а не молчаливое заполнение
                        print(f"Предупреждение: столбец {col_name} не найден, заполняется нулями")
                        X[i, t, c] = 0.0
        
        print(f"Загружено из {filepath}: {num_samples} образцов, "
              f"окно {window_len}, каналов {num_channels}")
        
        return X, labels
    
    def load_from_json(self, filepath):
        """
        Загрузка данных из JSON файла
        
        Формат JSON:
        {
            "sample_id": 1,
            "window_length": 512,
            "num_channels": 4,
            "data": [[[val1, val2, ...], ...], ...],
            "label": 0
        }
        или массив таких объектов
        
        Args:
            filepath: путь к JSON файлу
        Returns:
            X: [num_samples, window_len, num_channels]
            y: [num_samples]
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        samples = []
        labels = []
        
        for item in data:
            samples.append(np.array(item['data']))
            labels.append(item['label'])
        
        X = np.array(samples)
        y = np.array(labels)
        
        print(f"Загружено из {filepath}: {len(X)} образцов")
        
        return X, y
    
    def preprocess_and_save(self, X, y, output_path):
        """
        Предобработка и сохранение данных
        
        Выполняет последовательно:
        1. Удаление дрейфа
        2. Медианная фильтрация
        3. Z-score нормализация
        
        Args:
            X: [num_samples, window_len, num_channels] - исходные данные
            y: [num_samples] - метки
            output_path: путь для сохранения обработанных данных
        Returns:
            X_processed: обработанные данные
            y: метки (без изменений)
        """
        print("Предобработка данных...")
        
        # 1. Удаление дрейфа
        X = self.preprocessor.remove_drift(X)
        
        # 2. Медианная фильтрация
        X = self.preprocessor.median_filter(X, window_size=5)
        
        # 3. Нормализация
        X = self.preprocessor.normalize_zscore(X, fit=True)
        
        print(f"Предобработка завершена. X.shape = {X.shape}")
        
        # Сохранение статистик для инференса
        stats_path = output_path.replace('.csv', '_stats.npz')
        self.preprocessor.save_stats(stats_path)
        
        # Сохранение в CSV
        self._save_to_csv(X, y, output_path)
        
        return X, y
    
    def _save_to_csv(self, X, y, filepath):
        """
        Сохранение предобработанных данных в CSV
        
        Args:
            X: [num_samples, window_len, num_channels]
            y: [num_samples]
            filepath: путь для сохранения
        """
        num_samples, window_len, num_channels = X.shape
        
        data = []
        for i in range(num_samples):
            row = {'label': int(y[i])}
            for t in range(window_len):
                for c in range(num_channels):
                    row[f't{t}_ch{c}'] = X[i, t, c]
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Предобработанные данные сохранены: {filepath}")
    
    def apply_preprocessing_to_new_data(self, X, stats_path):
        """
        Применение предобработки к новым данным (для инференса)
        
        Args:
            X: [num_samples, window_len, num_channels] - новые данные
            stats_path: путь к файлу со статистиками нормализации
        Returns:
            X_processed: обработанные данные
        """
        # Загрузка статистик
        self.preprocessor.load_stats(stats_path)
        
        # 1. Удаление дрейфа
        X = self.preprocessor.remove_drift(X)
        
        # 2. Медианная фильтрация
        X = self.preprocessor.median_filter(X, window_size=5)
        
        # 3. Нормализация (с использованием сохранённых статистик)
        X = self.preprocessor.normalize_zscore(X, fit=False)
        
        return X


class RTKDataset(Dataset):
    """
    PyTorch Dataset для загрузки данных
    (Дублирует функциональность из synthetic_data.py для единообразия)
    """
    
    def __init__(self, X, y, config, augment=False):
        """
        Args:
            X: [num_samples, window_len, num_channels]
            y: [num_samples]
            config: объект конфигурации
            augment: применять ли аугментацию (True для обучения)
        """
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
            x = self._apply_augmentation(x)
        
        return x, y
    
    def _apply_augmentation(self, x):
        """
        Физически обоснованная аугментация (глава 3, раздел 3.2.3)
        Добавление гауссовского шума с дисперсией из таблицы 3.3
        """
        # Исправлено: обращение к AUG_NOISE_STD как к словарю
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


class RealtimeDataBuffer:
    """
    Буфер для потоковых данных реального времени
    Реализует скользящее окно для непрерывной классификации
    
    Используется в системе мониторинга РТК для потоковой обработки данных
    с частотой дискретизации 1000 Гц (глава 3, раздел 3.5)
    
    Принцип работы:
    - Данные поступают непрерывно (например, по 64 отсчёта за раз)
    - Буфер хранит последние buffer_size отсчётов
    - При накоплении window_len отсчётов возвращается окно для инференса
    """
    
    def __init__(self, config, buffer_size=None):
        """
        Args:
            config: объект конфигурации
            buffer_size: размер буфера (по умолчанию = 2 * window_len)
        """
        self.config = config
        self.window_len = config.WINDOW_LENGTH
        self.buffer_size = buffer_size if buffer_size else self.window_len * 2
        self.buffer = np.zeros((self.buffer_size, config.NUM_CHANNELS))
        self.write_pos = 0
        self.total_added = 0  # Общее количество добавленных отсчётов (для проверки заполненности)
        
    def add_samples(self, samples):
        """
        Добавление новых отсчётов в буфер
        
        Args:
            samples: [num_samples, num_channels] - новые данные
        Returns:
            ready: bool - готово ли окно для инференса
            window: [window_len, num_channels] - текущее окно (если готово)
        """
        num_new = len(samples)
        
        # Добавление в буфер
        end_pos = self.write_pos + num_new
        if end_pos <= self.buffer_size:
            self.buffer[self.write_pos:end_pos] = samples
        else:
            # Циклический буфер
            first_part = self.buffer_size - self.write_pos
            self.buffer[self.write_pos:] = samples[:first_part]
            self.buffer[:num_new - first_part] = samples[first_part:]
        
        self.write_pos = (self.write_pos + num_new) % self.buffer_size
        self.total_added += num_new
        
        # Проверка, достаточно ли данных для полного окна
        if self.total_added >= self.window_len:
            window = self.get_current_window()
            return True, window
        
        return False, None
    
    def get_current_window(self):
        """
        Получение текущего окна данных (последние window_len отсчётов)
        
        Returns:
            window: [window_len, num_channels]
        
        Raises:
            ValueError: если недостаточно данных в буфере
        """
        if self.total_added < self.window_len:
            raise ValueError(f"Недостаточно данных: {self.total_added} < {self.window_len}")
        
        if self.write_pos >= self.window_len:
            return self.buffer[self.write_pos - self.window_len:self.write_pos].copy()
        else:
            # Склейка хвоста и начала буфера
            tail = self.buffer[self.buffer_size - (self.window_len - self.write_pos):]
            head = self.buffer[:self.write_pos]
            return np.vstack([tail, head]).copy()
    
    def get_available_count(self):
        """
        Количество доступных отсчётов в буфере
        
        Returns:
            count: количество отсчётов
        """
        return min(self.total_added, self.buffer_size)
    
    def is_ready(self):
        """
        Проверка, готов ли буфер к инференсу
        
        Returns:
            ready: True если есть хотя бы одно полное окно
        """
        return self.total_added >= self.window_len
    
    def peek_window(self, offset=0):
        """
        Просмотр окна без извлечения (для отладки)
        
        Args:
            offset: смещение от текущей позиции (0 - последнее окно)
        Returns:
            window: [window_len, num_channels]
        """
        if self.total_added < self.window_len:
            return None
        
        # Вычисление позиции для offset
        # Для простоты реализуем только offset=0
        if offset == 0:
            return self.get_current_window()
        else:
            # Сложнее, можно добавить при необходимости
            raise NotImplementedError("Offset > 0 не реализован")
    
    def reset(self):
        """Сброс буфера (очистка всех данных)"""
        self.buffer = np.zeros((self.buffer_size, self.config.NUM_CHANNELS))
        self.write_pos = 0
        self.total_added = 0


# Функции для создания DataLoader (удобные обёртки)

def create_dataloaders_from_arrays(X, y, config, train_ratio=0.7, val_ratio=0.15):
    """
    Создание DataLoader из массивов numpy
    
    Args:
        X: [num_samples, window_len, num_channels]
        y: [num_samples]
        config: объект конфигурации
        train_ratio: доля обучающей выборки
        val_ratio: доля валидационной выборки
    Returns:
        train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split
    
    # Разделение на train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), stratify=y, random_state=config.RANDOM_SEED
    )
    
    val_ratio_adjusted = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_adjusted),
        stratify=y_temp, random_state=config.RANDOM_SEED
    )
    
    # Создание датасетов
    train_dataset = RTKDataset(X_train, y_train, config, augment=True)
    val_dataset = RTKDataset(X_val, y_val, config, augment=False)
    test_dataset = RTKDataset(X_test, y_test, config, augment=False)
    
    # DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Созданы DataLoader:")
    print(f"  Обучающая: {len(train_dataset)} образцов")
    print(f"  Валидационная: {len(val_dataset)} образцов")
    print(f"  Тестовая: {len(test_dataset)} образцов")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ DATA_LOADER.PY")
    print("=" * 60)
    
    # Тест 1: DataPreprocessor
    print("\n--- Тест 1: DataPreprocessor ---")
    config = Config
    preprocessor = DataPreprocessor(config)
    
    # Создание тестовых данных
    test_data = np.random.randn(100, config.WINDOW_LENGTH, config.NUM_CHANNELS)
    print(f"Тестовые данные: {test_data.shape}")
    
    # Нормализация
    normalized = preprocessor.normalize_zscore(test_data, fit=True)
    print(f"Нормализация: {normalized.shape}, среднее ≈ {normalized.mean():.6f}, std ≈ {normalized.std():.6f}")
    
    # Фильтрация
    filtered = preprocessor.median_filter(normalized, window_size=5)
    print(f"Фильтрация: {filtered.shape}")
    
    # Удаление дрейфа
    detrended = preprocessor.remove_drift(filtered, order=1)
    print(f"Удаление дрейфа: {detrended.shape}")
    
    # Сохранение и загрузка статистик
    preprocessor.save_stats("test_stats.npz")
    preprocessor.load_stats("test_stats.npz")
    
    # Тест 2: RealtimeDataBuffer
    print("\n--- Тест 2: RealtimeDataBuffer ---")
    buffer = RealtimeDataBuffer(config)
    print(f"Буфер создан: размер {buffer.buffer_size}, окно {buffer.window_len}")
    
    # Имитация потока данных
    total_samples = 0
    for i in range(10):
        new_data = np.random.randn(64, config.NUM_CHANNELS)
        ready, window = buffer.add_samples(new_data)
        total_samples += 64
        if ready:
            print(f"  После {total_samples} отсчётов: готово окно для инференса, форма {window.shape}")
    
    print(f"Доступно отсчётов: {buffer.get_available_count()}")
    print(f"Готов к инференсу: {buffer.is_ready()}")
    
    # Тест 3: DataLoaderFromFiles (с синтетическими данными)
    print("\n--- Тест 3: DataLoaderFromFiles ---")
    from synthetic_data import SyntheticDataGenerator
    
    generator = SyntheticDataGenerator(config)
    X_syn, y_syn = generator.generate_dataset(samples_per_class=10)
    
    loader = DataLoaderFromFiles(config)
    X_processed, y_processed = loader.preprocess_and_save(
        X_syn, y_syn, "test_processed_data.csv"
    )
    
    print(f"Обработанные данные: {X_processed.shape}")
    
    # Тест 4: create_dataloaders_from_arrays
    print("\n--- Тест 4: create_dataloaders_from_arrays ---")
    train_loader, val_loader, test_loader = create_dataloaders_from_arrays(
        X_processed, y_processed, config
    )
    
    batch_x, batch_y = next(iter(train_loader))
    print(f"Батч из train_loader: x.shape = {batch_x.shape}, y.shape = {batch_y.shape}")
    
    # Очистка тестовых файлов
    import os
    for f in ["test_stats.npz", "test_processed_data.csv"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Удалён тестовый файл: {f}")
    
    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
    print("=" * 60)