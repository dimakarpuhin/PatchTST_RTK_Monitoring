import pandas as pd
import numpy as np
import os

def label_real_data(input_file, output_file):
    """
    Автоматическая разметка реальных данных по пороговым значениям
    """
    # Загрузка данных
    df = pd.read_csv(input_file, parse_dates=['Текущие дата и время'])
    print(f"📊 Загружено {len(df)} записей")
    
    # Выбранные колонки (должны совпадать с prepare_real_data.py)
    voltage_col = 'U_SH2:1'
    current_col = 'I_BF2:1'
    temp_col = 'T_AB1_MAX:1'
    
    # Проверка наличия колонок
    for col in [voltage_col, current_col, temp_col]:
        if col not in df.columns:
            print(f"❌ Колонка {col} не найдена!")
            return None
    
    # Извлекаем данные
    U = df[voltage_col].values
    I = df[current_col].values
    T = df[temp_col].values
    
    # Создаём метки (по умолчанию 0 — норма)
    labels = np.zeros(len(df), dtype=int)
    
    # ===== РАЗМЕТКА =====
    # 1. Скачок напряжения (класс 1)
    labels[(U > 250) | (U < 180)] = 1
    print(f"   Класс 1 (скачок напряжения): {np.sum(labels == 1)} записей")
    
    # 2. Токовая перегрузка (класс 2)
    labels[I > 15] = 2
    print(f"   Класс 2 (токовая перегрузка): {np.sum(labels == 2)} записей")
    
    # 3. Перегрев (класс 3)
    labels[T > 60] = 3
    print(f"   Класс 3 (перегрев): {np.sum(labels == 3)} записей")
    
    # 4. Помехи (класс 4) — высокая вариативность напряжения
    # Считаем скользящее стандартное отклонение (окно 10)
    noise_std = pd.Series(U).rolling(window=10, min_periods=1).std().values
    labels[noise_std > 5] = 4
    print(f"   Класс 4 (помехи): {np.sum(labels == 4)} записей")
    
    # ===== ПРОВЕРКА =====
    print("\n📊 Распределение классов:")
    for cls in range(5):
        count = np.sum(labels == cls)
        print(f"   Класс {cls}: {count} записей ({count/len(labels)*100:.2f}%)")
    
    # Сохраняем метки
    df['label'] = labels
    df.to_csv(output_file, index=False)
    print(f"\n💾 Сохранено в {output_file}")
    
    # Также сохраняем только метки в numpy
    np.save(output_file.replace('.csv', '_labels.npy'), labels)
    print(f"💾 Метки сохранены в {output_file.replace('.csv', '_labels.npy')}")
    
    return labels

if __name__ == '__main__':
    INPUT_FILE = 'data/real/processed/real_data_full.csv'
    OUTPUT_FILE = 'data/real/processed/real_data_labeled.csv'
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("=" * 60)
    print("🏷️ РАЗМЕТКА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл {INPUT_FILE} не найден.")
        print("   Сначала запустите: python parse_real_data.py")
    else:
        label_real_data(INPUT_FILE, OUTPUT_FILE)