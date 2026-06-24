import pandas as pd
import numpy as np
import os
from glob import glob

def parse_kpa_report(file_path):
    """Парсинг одного txt файла КПА"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if line.strip()]
    
    if len(lines) < 2:
        return None
    
    # Заголовки
    headers = [h.strip() for h in lines[0].split('|') if h.strip()]
    
    # Данные
    data_rows = []
    for line in lines[1:]:
        values = [v.strip() for v in line.split('|')]
        if len(values) >= len(headers):
            data_rows.append(values[:len(headers)])
    
    if not data_rows:
        return None
    
    df = pd.DataFrame(data_rows, columns=headers)
    
    # Преобразуем числовые колонки
    for col in df.columns:
        if col != 'Текущие дата и время':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Преобразуем дату
    df['Текущие дата и время'] = pd.to_datetime(
        df['Текущие дата и время'], 
        format='%d.%m.%Y %H:%M:%S',
        errors='coerce'
    )
    
    df = df.dropna(subset=['Текущие дата и время'])
    df = df.sort_values('Текущие дата и время').reset_index(drop=True)
    
    return df

def parse_all_reports(folder_path, output_path):
    """Парсинг всех txt файлов в папке"""
    files = sorted(glob(os.path.join(folder_path, '*.txt')))
    
    if not files:
        print(f"❌ Нет txt файлов в {folder_path}")
        return None
    
    print(f"📁 Найдено {len(files)} файлов")
    
    all_dfs = []
    for file in files:
        print(f"   Обработка: {os.path.basename(file)}")
        df = parse_kpa_report(file)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        print("❌ Не удалось обработать файлы")
        return None
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values('Текущие дата и время').reset_index(drop=True)
    
    print(f"\n✅ Всего записей: {len(full_df)}")
    print(f"📅 Период: {full_df['Текущие дата и время'].min()} - {full_df['Текущие дата и время'].max()}")
    
    full_df.to_csv(output_path, index=False)
    print(f"💾 Сохранено в {output_path}")
    
    return full_df

if __name__ == '__main__':
    INPUT_FOLDER = 'data/real/raw'
    OUTPUT_FILE = 'data/real/processed/real_data_full.csv'
    
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("=" * 60)
    print("📊 ПАРСИНГ РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    
    files = glob(os.path.join(INPUT_FOLDER, '*.txt'))
    if not files:
        print(f"\n❌ В папке {INPUT_FOLDER} нет txt файлов.")
        print("   Поместите файлы КПА_отчет_*.txt в эту папку.")
    else:
        parse_all_reports(INPUT_FOLDER, OUTPUT_FILE)