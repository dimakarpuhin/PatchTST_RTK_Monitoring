# load_arc_loss.py
# Загрузка и предобработка Arc Loss Dataset

import torch
import numpy as np
import os
from config import Config

def load_arc_loss_data():
    """Загрузка Arc Loss Dataset из .pt файлов"""
    
    print("=" * 60)
    print("📊 ЗАГРУЗКА ARC LOSS DATASET")
    print("=" * 60)
    
    # Пути к файлам
    train_path = os.path.join(Config.ARC_LOSS_PATH, 'train.pt')
    val_path = os.path.join(Config.ARC_LOSS_PATH, 'val.pt')
    test_path = os.path.join(Config.ARC_LOSS_PATH, 'test.pt')
    
    # Проверка наличия файлов
    if not os.path.exists(train_path):
        print(f"❌ Файл {train_path} не найден!")
        print("   Поместите файлы в папку data/arc_loss/raw/")
        return None, None, None, None, None, None
    
    # Загрузка
    print("📥 Загрузка данных...")
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    test_data = torch.load(test_path)
    
    # Извлечение данных (структура может отличаться, уточни по README)
    # Обычно в .pt файлах словари с ключами 'X' и 'y'
    X_train = train_data['X'].numpy() if torch.is_tensor(train_data['X']) else train_data['X']
    y_train = train_data['y'].numpy() if torch.is_tensor(train_data['y']) else train_data['y']
    
    X_val = val_data['X'].numpy() if torch.is_tensor(val_data['X']) else val_data['X']
    y_val = val_data['y'].numpy() if torch.is_tensor(val_data['y']) else val_data['y']
    
    X_test = test_data['X'].numpy() if torch.is_tensor(test_data['X']) else test_data['X']
    y_test = test_data['y'].numpy() if torch.is_tensor(test_data['y']) else test_data['y']
    
    print(f"\n✅ Данные загружены:")
    print(f"  Train: {X_train.shape}, метки: {y_train.shape}")
    print(f"  Val:   {X_val.shape}, метки: {y_val.shape}")
    print(f"  Test:  {X_test.shape}, метки: {y_test.shape}")
    
    # Сохранение в .npy для быстрого доступа
    os.makedirs(Config.ARC_LOSS_PROCESSED, exist_ok=True)
    
    np.save(os.path.join(Config.ARC_LOSS_PROCESSED, 'X_train.npy'), X_train)
    np.save(os.path.join(Config.ARC_LOSS_PROCESSED, 'y_train.npy'), y_train)
    np.save(os.path.join(Config.ARC_LOSS_PROCESSED, 'X_val.npy'), X_val)
    np.save(os.path.join(Config.ARC_LOSS_PROCESSED, 'y_val.npy'), y_val)
    np.save(os.path.join(Config.ARC_LOSS_PROCESSED, 'X_test.npy'), X_test)
    np.save(os.path.join(Config.ARC_LOSS_PROCESSED, 'y_test.npy'), y_test)
    
    print(f"\n💾 Данные сохранены в {Config.ARC_LOSS_PROCESSED}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def check_data_structure():
    """Проверка структуры данных (для отладки)"""
    import torch
    
    train_path = os.path.join(Config.ARC_LOSS_PATH, 'train.pt')
    
    if not os.path.exists(train_path):
        print("❌ Файл не найден")
        return
    
    data = torch.load(train_path)
    print("Структура данных:")
    print(f"  Тип: {type(data)}")
    
    if isinstance(data, dict):
        print(f"  Ключи: {list(data.keys())}")
        for key, value in data.items():
            print(f"    {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    elif isinstance(data, (list, tuple)):
        print(f"  Длина: {len(data)}")
        for i, item in enumerate(data):
            print(f"    [{i}]: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")

if __name__ == '__main__':
    # Сначала проверь структуру
    print("=" * 60)
    print("🔍 ПРОВЕРКА СТРУКТУРЫ ДАННЫХ")
    print("=" * 60)
    check_data_structure()
    
    # Затем загрузи
    print("\n" + "=" * 60)
    load_arc_loss_data()