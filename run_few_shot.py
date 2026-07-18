# run_few_shot.py
# Эксперимент В: Few-Shot Learning
# Сравнение моделей при разном количестве обучающих данных

import numpy as np
import torch
import pandas as pd
import time
import os
from config import Config
from model import ModifiedPatchTST, create_model
from train import Trainer
from synthetic_data import SyntheticDataGenerator
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def get_model(model_type, config):
    """Создание модели по типу"""
    if model_type == 'full':
        # Полная модель (все модификации включены)
        config.USE_ADAPTIVE_ENCODING = True
        config.USE_CHANNEL_ATTENTION = True
        config.LAMBDA_2 = 0.1
        config.MASK_PROB = 0.15
        model = create_model(config)
    elif model_type == 'baseline':
        # Базовый PatchTST (все модификации выключены)
        config.USE_ADAPTIVE_ENCODING = False
        config.USE_CHANNEL_ATTENTION = False
        config.LAMBDA_2 = 0
        config.MASK_PROB = 0
        model = create_model(config)
    elif model_type == 'no_fa':
        # Без межканального внимания
        config.USE_ADAPTIVE_ENCODING = True
        config.USE_CHANNEL_ATTENTION = False
        config.LAMBDA_2 = 0.1
        config.MASK_PROB = 0.15
        model = create_model(config)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    # Восстанавливаем настройки для следующих экспериментов
    config.USE_ADAPTIVE_ENCODING = True
    config.USE_CHANNEL_ATTENTION = True
    config.LAMBDA_2 = 0.1
    config.MASK_PROB = 0.15
    
    return model

def create_dataloaders_manual(X, y, config, val_ratio=0.2):
    """
    Ручное создание DataLoader для train/val
    """
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, stratify=y, random_state=config.RANDOM_SEED
    )
    
    # Создание датасетов
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    # Batch size: не больше размера выборки
    batch_size = min(config.BATCH_SIZE, len(X_train))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train: {len(X_train)} образцов, Val: {len(X_val)} образцов")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader

def run_few_shot_experiment(samples_per_class, model_type, config):
    """
    Запуск одного эксперимента Few-Shot Learning
    """
    print("\n" + "=" * 60)
    print(f"📊 ЭКСПЕРИМЕНТ: {model_type.upper()}, {samples_per_class} образцов/класс")
    print("=" * 60)
    
    # Генерация данных
    generator = SyntheticDataGenerator(config)
    X, y = generator.generate_dataset(samples_per_class=samples_per_class)
    
    # Ручное создание DataLoader
    train_loader, val_loader = create_dataloaders_manual(X, y, config, val_ratio=0.2)
    
    # Создание модели
    model = get_model(model_type, config)
    model = model.to(config.DEVICE)
    
    # Обучение
    trainer = Trainer(model, config)
    start_time = time.time()
    best_acc = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    # Сохранение истории
    history_df = pd.DataFrame(trainer.history)
    os.makedirs('logs/few_shot', exist_ok=True)
    history_df.to_csv(f'logs/few_shot/{model_type}_{samples_per_class}.csv', index=False)
    
    return {
        'model': model_type,
        'samples_per_class': samples_per_class,
        'best_acc': best_acc,
        'time_sec': round(elapsed, 1),
        'epochs': len(trainer.history['val_acc'])
    }

def main():
    print("=" * 70)
    print("🔬 ЭКСПЕРИМЕНТ В: FEW-SHOT LEARNING")
    print("=" * 70)
    
    Config.set_seed()
    Config.ensure_dirs()
    Config.NUM_EPOCHS = 100
    Config.EARLY_STOPPING_PATIENCE = 20
    
    # Количество образцов на класс для тестирования
    sample_sizes = [20, 50, 100, 200, 500]
    
    # Типы моделей
    model_types = ['full', 'baseline', 'no_fa']
    
    all_results = []
    
    for samples in sample_sizes:
        for model_type in model_types:
            result = run_few_shot_experiment(samples, model_type, Config)
            all_results.append(result)
    
    # Сводная таблица
    df_results = pd.DataFrame(all_results)
    
    # Создание сводной таблицы (модели × размер выборки)
    pivot = df_results.pivot(index='samples_per_class', columns='model', values='best_acc')
    pivot = pivot.round(2)
    
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА FEW-SHOT LEARNING")
    print("=" * 70)
    print(pivot.to_string())
    
    # Сохранение
    os.makedirs('logs/few_shot', exist_ok=True)
    df_results.to_csv('logs/few_shot/few_shot_results.csv', index=False)
    pivot.to_csv('logs/few_shot/few_shot_pivot.csv')
    
    print("\n💾 Результаты сохранены в logs/few_shot/")
    
    # Построение графика
    try:
        plot_few_shot_results(pivot)
    except Exception as e:
        print(f"⚠️ Не удалось построить график: {e}")

def plot_few_shot_results(pivot):
    """Построение графика зависимости точности от размера выборки"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # Цвета для моделей
        colors = {
            'full': '#2ecc71',      # зелёный
            'baseline': '#e74c3c',  # красный
            'no_fa': '#3498db'      # синий
        }
        
        labels = {
            'full': 'Модифицированный PatchTST',
            'baseline': 'Базовый PatchTST',
            'no_fa': 'PatchTST без FA'
        }
        
        for model in pivot.columns:
            plt.plot(pivot.index, pivot[model], 
                    marker='o', linewidth=2, 
                    color=colors.get(model, '#000000'),
                    label=labels.get(model, model.upper()))
        
        plt.xlabel('Образцов на класс', fontsize=12)
        plt.ylabel('Точность на валидации (%)', fontsize=12)
        plt.title('Few-Shot Learning: зависимость точности от размера выборки', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.xticks(pivot.index, pivot.index)
        plt.ylim(0, 105)
        
        plt.tight_layout()
        os.makedirs('images', exist_ok=True)
        plt.savefig('logs/few_shot/few_shot_plot.png', dpi=150)
        plt.savefig('images/few_shot_plot.png', dpi=150)
        print("📈 График сохранён: logs/few_shot/few_shot_plot.png")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Ошибка при построении графика: {e}")

if __name__ == '__main__':
    main()