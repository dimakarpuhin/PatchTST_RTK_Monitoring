import pandas as pd
import numpy as np
from config import Config
from sklearn.preprocessing import StandardScaler
import os
import joblib

def prepare_real_dataset(input_file, output_prefix, window_len=None):
    if window_len is None:
        window_len = Config.WINDOW_LENGTH
    
    # Загрузка данных с метками
    df = pd.read_csv(input_file, parse_dates=['Текущие дата и время'])
    print(f"📊 Загружено {len(df)} записей")
    
    # Извлечение меток
    if 'label' in df.columns:
        labels = df['label'].values
        print(f"🏷️ Метки загружены. Уникальные классы: {np.unique(labels)}")
    else:
        print("⚠️ Колонка 'label' не найдена. Используем все метки = 0.")
        labels = np.zeros(len(df), dtype=int)
    
    # Выбранные колонки
    voltage_col = 'U_SH2:1'
    current_col = 'I_BF2:1'
    temp_col = 'T_AB1_MAX:1'
    
    print(f"📋 Используемые колонки:")
    print(f"   Напряжение: {voltage_col}")
    print(f"   Ток: {current_col}")
    print(f"   Температура: {temp_col}")
    
    # Проверяем наличие колонок
    missing = []
    for col in [voltage_col, current_col, temp_col]:
        if col not in df.columns:
            missing.append(col)
    
    if missing:
        print(f"❌ Не найдены колонки: {missing}")
        return None
    
    # Извлекаем данные
    data = df[[voltage_col, current_col, temp_col]].values
    
    # Вычисляем помехи (вариативность напряжения)
    noise = pd.Series(data[:, 0]).rolling(window=10, min_periods=1).std().values
    noise = noise.reshape(-1, 1)
    data = np.hstack([data, noise])
    
    print(f"📊 Форма данных: {data.shape}")
    
    # Проверяем на NaN
    if np.isnan(data).any():
        print("⚠️ Есть NaN в данных. Заполняем средними...")
        for i in range(data.shape[1]):
            col_mean = np.nanmean(data[:, i])
            if not np.isnan(col_mean):
                data[np.isnan(data[:, i]), i] = col_mean
            else:
                data[np.isnan(data[:, i]), i] = 0
    
    # Нормализация
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Создание окон с метками
    X = []
    y = []
    step = window_len // 2
    
    for i in range(0, len(data) - window_len, step):
        window = data[i:i+window_len]
        X.append(window)
        # Метка окна = наиболее частый класс в этом окне
        window_labels = labels[i:i+window_len]
        # Исправлено: простое вычисление моды
        unique, counts = np.unique(window_labels, return_counts=True)
        if len(unique) > 0:
            label = unique[np.argmax(counts)]
        else:
            label = 0
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Создано {len(X)} окон, форма: {X.shape}")
    print(f"🏷️ Распределение меток в окнах:")
    for cls in range(5):
        count = np.sum(y == cls)
        if count > 0:
            print(f"   Класс {cls}: {count} окон ({count/len(y)*100:.2f}%)")
    
    # Сохраняем
    X_path = f'{output_prefix}_X.npy'
    y_path = f'{output_prefix}_y.npy'
    np.save(X_path, X)
    np.save(y_path, y)
    print(f"💾 Сохранено в {X_path} и {y_path}")
    
    # Сохраняем скалер
    scaler_path = f'{output_prefix}_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"💾 Скалер сохранён в {scaler_path}")
    
    return X, y

if __name__ == '__main__':
    INPUT_FILE = 'data/real/processed/real_data_with_anomalies.csv'
    OUTPUT_PREFIX = 'data/real/processed/real_dataset'
    
    os.makedirs(os.path.dirname(OUTPUT_PREFIX), exist_ok=True)
    
    print("=" * 60)
    print("📊 ПОДГОТОВКА РЕАЛЬНЫХ ДАННЫХ С МЕТКАМИ")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл {INPUT_FILE} не найден.")
        print("   Сначала запустите: python add_anomalies_to_real.py")
    else:
        prepare_real_dataset(INPUT_FILE, OUTPUT_PREFIX)