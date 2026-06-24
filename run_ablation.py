import pandas as pd
import time
import os
import numpy as np
from config import Config
from model import ModifiedPatchTST
from train import Trainer, prepare_data
from synthetic_data import SyntheticDataGenerator
from data_loader import create_dataloaders_from_arrays

# Создаём папку для результатов
os.makedirs('logs/ablation', exist_ok=True)

def generate_shared_data():
    """Генерация общих данных для всех экспериментов"""
    print("=" * 60)
    print("📦 ГЕНЕРАЦИЯ ОБЩИХ ДАННЫХ ДЛЯ ABLATION STUDY")
    print("=" * 60)
    
    generator = SyntheticDataGenerator(Config)
    X, y = generator.generate_dataset(samples_per_class=2000)
    
    # Сохраняем данные в файл (чтобы не генерировать каждый раз)
    np.save('data/ablation_X.npy', X)
    np.save('data/ablation_y.npy', y)
    
    print(f"Данные сохранены: X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y

def load_shared_data():
    """Загрузка общих данных"""
    X = np.load('data/ablation_X.npy')
    y = np.load('data/ablation_y.npy')
    return X, y

def run_ablation_experiment(name, config_mods, X, y):
    """
    Запуск одного эксперимента Ablation Study на общих данных
    """
    print("=" * 60)
    print(f"🧪 ЭКСПЕРИМЕНТ: {name}")
    print("=" * 60)
    
    # Сохраняем оригинальные значения
    original_values = {}
    for key in config_mods.keys():
        if hasattr(Config, key):
            original_values[key] = getattr(Config, key)
    
    # Применяем изменения
    for key, value in config_mods.items():
        setattr(Config, key, value)
        print(f"   {key} = {value}")


    # ===== ДОБАВИТЬ ОТЛАДОЧНЫЙ ВЫВОД =====
    print("\n🔍 ТЕКУЩИЕ НАСТРОЙКИ:")
    print(f"   USE_ADAPTIVE_ENCODING = {getattr(Config, 'USE_ADAPTIVE_ENCODING', 'НЕ ОПРЕДЕЛЁН')}")
    print(f"   USE_CHANNEL_ATTENTION = {getattr(Config, 'USE_CHANNEL_ATTENTION', 'НЕ ОПРЕДЕЛЁН')}")
    print(f"   MASK_PROB = {Config.MASK_PROB}")
    print(f"   LAMBDA_2 = {Config.LAMBDA_2}")
    print("=" * 60)
    # =====================================

    
    # Создаём DataLoader из общих данных
    train_loader, val_loader, _ = create_dataloaders_from_arrays(X, y, Config)
    
    # Создание модели
    model = ModifiedPatchTST(Config)
    model = model.to(Config.DEVICE)
    
    # Обучение
    trainer = Trainer(model, Config)
    start_time = time.time()
    best_acc = trainer.train(train_loader, val_loader, Config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    # Сохранение результата
    result = {
        'experiment': name,
        'best_acc': best_acc,
        'time_sec': round(elapsed, 1),
        'epochs': len(trainer.history['val_acc'])
    }
    
    # Сохраняем историю обучения
    pd.DataFrame(trainer.history).to_csv(f'logs/ablation/{name}.csv', index=False)
    
    # Восстанавливаем оригинальные значения
    for key, value in original_values.items():
        setattr(Config, key, value)
    
    return result

def run_all_ablations():
    """
    Запуск всех экспериментов Ablation Study на общих данных
    """
    # Генерируем или загружаем общие данные
    if not os.path.exists('data/ablation_X.npy'):
        X, y = generate_shared_data()
    else:
        X, y = load_shared_data()
        print(f"Загружены общие данные: X.shape = {X.shape}, y.shape = {y.shape}")
    
    results = []
    
    # 1. Полная модель (все модификации включены)
    results.append(run_ablation_experiment(
        '01_full_model', {}, X, y
    ))
    
    # 2. Без адаптивного позиционного кодирования
    results.append(run_ablation_experiment(
        '02_without_adaptive_encoding',
        {'USE_ADAPTIVE_ENCODING': False}, X, y
    ))
    
    # 3. Без межканального внимания
    results.append(run_ablation_experiment(
        '03_without_channel_attention',
        {'USE_CHANNEL_ATTENTION': False}, X, y
    ))
    
    # 4. Без контрастивной потери
    results.append(run_ablation_experiment(
        '04_without_contrastive_loss',
        {'LAMBDA_2': 0}, X, y
    ))
    
    # 5. Без маскирования
    results.append(run_ablation_experiment(
        '05_without_masking',
        {'MASK_PROB': 0}, X, y
    ))
    
    # 6. Базовый PatchTST (все модификации выключены)
    results.append(run_ablation_experiment(
        '06_baseline_patchtst',
        {
            'USE_ADAPTIVE_ENCODING': False,
            'USE_CHANNEL_ATTENTION': False,
            'LAMBDA_2': 0,
            'MASK_PROB': 0
        }, X, y
    ))
    
    # Сохраняем сводную таблицу
    df = pd.DataFrame(results)
    df.to_csv('logs/ablation/ablation_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("📊 СВОДНАЯ ТАБЛИЦА ABLATION STUDY")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    return df

if __name__ == '__main__':
    Config.set_seed()
    run_all_ablations()