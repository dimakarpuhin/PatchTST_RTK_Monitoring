import pandas as pd
import numpy as np
import os

def add_anomalies_to_real(input_file, output_file, anomaly_ratio=0.1):
    """
    Добавление синтетических аномалий в реальные данные
    anomaly_ratio: доля окон с аномалиями (0.1 = 10%)
    """
    # Загрузка данных
    df = pd.read_csv(input_file, parse_dates=['Текущие дата и время'])
    print(f"📊 Загружено {len(df)} записей")
    
    # Выбранные колонки
    voltage_col = 'U_SH2:1'
    current_col = 'I_BF2:1'
    temp_col = 'T_AB1_MAX:1'
    
    # Проверка наличия колонок
    for col in [voltage_col, current_col, temp_col]:
        if col not in df.columns:
            print(f"❌ Колонка {col} не найдена!")
            return None
    
    # Копируем данные
    df_anomaly = df.copy()
    labels = np.zeros(len(df), dtype=int)
    
    # Определяем окна (длина окна = 512 отсчётов, шаг = 256)
    window_len = 512
    step = 256
    num_windows = (len(df) - window_len) // step + 1  # исправлено
    
    print(f"📊 Всего окон: {num_windows}")
    print(f"📊 Добавляем аномалии в {int(anomaly_ratio * num_windows)} окон")
    
    # Выбираем случайные окна для аномалий
    anomaly_windows = np.random.choice(
        num_windows, 
        size=int(anomaly_ratio * num_windows), 
        replace=False
    )
    
    # Типы аномалий
    anomaly_types = [
        {'name': 'скачок напряжения', 'class': 1, 'factor': 1.2},
        {'name': 'токовая перегрузка', 'class': 2, 'factor': 1.5},
        {'name': 'перегрев', 'class': 3, 'factor': 1.3},
        {'name': 'помехи', 'class': 4, 'factor': 2.0}
    ]
    
    for idx in anomaly_windows:
        start = idx * step
        end = start + window_len - 1  # исправлено: end = start + window_len - 1
        
        # Выбираем тип аномалии случайно
        anomaly = np.random.choice(anomaly_types)
        class_id = anomaly['class']
        factor = anomaly['factor']
        
        # Применяем аномалию
        if class_id == 1:  # скачок напряжения
            df_anomaly.loc[start:end, voltage_col] *= factor
        elif class_id == 2:  # токовая перегрузка
            df_anomaly.loc[start:end, current_col] *= factor
        elif class_id == 3:  # перегрев
            df_anomaly.loc[start:end, temp_col] *= factor
        elif class_id == 4:  # помехи
            noise = np.random.randn(window_len) * 5
            df_anomaly.loc[start:end, voltage_col] += noise
        
        # Ставим метку для всего окна
        labels[start:end+1] = class_id  # исправлено
    
    # Сохраняем
    df_anomaly['label'] = labels
    df_anomaly.to_csv(output_file, index=False)
    
    print("\n📊 Распределение классов в размеченных данных:")
    for cls in range(5):
        count = np.sum(labels == cls)
        print(f"   Класс {cls}: {count} записей ({count/len(labels)*100:.2f}%)")
    
    print(f"\n💾 Сохранено в {output_file}")
    return df_anomaly

if __name__ == '__main__':
    INPUT_FILE = 'data/real/processed/real_data_labeled.csv'
    OUTPUT_FILE = 'data/real/processed/real_data_with_anomalies.csv'
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("=" * 60)
    print("🔧 ДОБАВЛЕНИЕ АНОМАЛИЙ В РЕАЛЬНЫЕ ДАННЫЕ")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл {INPUT_FILE} не найден.")
        print("   Сначала запустите: python label_real_data.py")
    else:
        add_anomalies_to_real(INPUT_FILE, OUTPUT_FILE, anomaly_ratio=0.1)