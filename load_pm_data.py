# load_pm_data.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import Config

def load_pm_data():
    """Загрузка и предобработка Predictive Maintenance Dataset (БИНАРНАЯ)"""
    
    print("=" * 60)
    print("📊 ЗАГРУЗКА PREDICTIVE MAINTENANCE DATASET (БИНАРНАЯ)")
    print("=" * 60)
    
    df = pd.read_csv(Config.PM_DATA_PATH)
    print(f"  Размер: {df.shape}")
    
    # ===== ВЫБОР ПАРАМЕТРОВ (5 каналов) =====
    feature_cols = [
        'Air temperature [K]',
        'Process temperature [K]', 
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    X_raw = df[feature_cols].values
    
    # ===== БИНАРНАЯ МЕТКА (0 - норма, 1 - отказ) =====
    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    y = np.zeros(len(df), dtype=int)
    for i, (idx, row) in enumerate(df.iterrows()):
        if row[failure_cols].sum() > 0:
            y[i] = 1  # отказ
        else:
            y[i] = 0  # норма
    
    print(f"\n  Распределение классов (бинарное):")
    print(f"    0 (No Failure): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"    1 (Failure): {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    
    # ===== НОРМАЛИЗАЦИЯ =====
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
        # ===== СОЗДАНИЕ ВРЕМЕННЫХ ОКОН (сбалансированное) =====
    window_len = Config.PM_WINDOW_LENGTH
    step = window_len // 2
    
    X = []
    y_windowed = []
    
    # 1. Окна с отказами
    failure_indices = np.where(y == 1)[0]
    if len(failure_indices) > 0:
        for idx in failure_indices:
            start = max(0, idx - window_len // 2)
            end = min(len(X_scaled), start + window_len)
            if end - start == window_len:
                window = X_scaled[start:end]
                X.append(window)
                y_windowed.append(1)
    
    # 2. Окна без отказов (столько же, сколько с отказами)
    normal_indices = np.where(y == 0)[0]
    if len(normal_indices) > 0 and len(X) > 0:
        num_normal = len(X)  # балансируем
        for _ in range(num_normal):
            start = np.random.randint(0, len(normal_indices) - window_len)
            window = X_scaled[normal_indices[start]:normal_indices[start] + window_len]
            X.append(window)
            y_windowed.append(0)
    
    X = np.array(X)
    y = np.array(y_windowed)
    
    # ===== РАЗДЕЛЕНИЕ =====
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"\n  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # ===== СОХРАНЕНИЕ =====
    os.makedirs(Config.PM_PROCESSED_PATH, exist_ok=True)
    
    np.save(f'{Config.PM_PROCESSED_PATH}/X_train.npy', X_train)
    np.save(f'{Config.PM_PROCESSED_PATH}/y_train.npy', y_train)
    np.save(f'{Config.PM_PROCESSED_PATH}/X_val.npy', X_val)
    np.save(f'{Config.PM_PROCESSED_PATH}/y_val.npy', y_val)
    np.save(f'{Config.PM_PROCESSED_PATH}/X_test.npy', X_test)
    np.save(f'{Config.PM_PROCESSED_PATH}/y_test.npy', y_test)
    
    print(f"\n💾 Данные сохранены в {Config.PM_PROCESSED_PATH}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    load_pm_data()