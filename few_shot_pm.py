# few_shot_pm.py
# Few-Shot Learning на реальных данных PM (все модели)

import numpy as np
import torch
import pandas as pd
import time
import os
from config import Config
from model import create_model
from models_lstm import LSTMModel
from models_gru import GRUModel
from models_tcn import TCNModel
from models_transformer import TransformerModel
from train import Trainer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

def load_pm_processed():
    """Загрузка обработанных данных"""
    X_train = np.load(f'{Config.PM_PROCESSED_PATH}/X_train.npy')
    y_train = np.load(f'{Config.PM_PROCESSED_PATH}/y_train.npy')
    X_val = np.load(f'{Config.PM_PROCESSED_PATH}/X_val.npy')
    y_val = np.load(f'{Config.PM_PROCESSED_PATH}/y_val.npy')
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_model(model_type, config):
    """Создание модели по типу"""
    if model_type == 'patchtst':
        config.USE_ADAPTIVE_ENCODING = True
        config.USE_CHANNEL_ATTENTION = True
        config.LAMBDA_2 = 0.0
        config.MASK_PROB = 0.15
        model = create_model(config)
    elif model_type == 'patchtst_baseline':
        config.USE_ADAPTIVE_ENCODING = False
        config.USE_CHANNEL_ATTENTION = False
        config.LAMBDA_2 = 0.0
        config.MASK_PROB = 0.0
        model = create_model(config)
    elif model_type == 'lstm':
        model = LSTMModel(config)
    elif model_type == 'gru':
        model = GRUModel(config)
    elif model_type == 'tcn':
        model = TCNModel(config)
    elif model_type == 'transformer':
        model = TransformerModel(config)
    else:
        raise ValueError(f"Неизвестная модель: {model_type}")
    
    return model.to(config.DEVICE)

def run_few_shot_experiment(model_type, X_train, y_train, X_val, y_val, X_test, y_test, 
                            sample_ratio, config):
    """Запуск эксперимента с ограниченной выборкой"""
    
    # Уменьшаем выборку
    if sample_ratio < 1.0:
        X_train_small, _, y_train_small, _ = train_test_split(
            X_train, y_train, train_size=sample_ratio, stratify=y_train, 
            random_state=config.RANDOM_SEED
        )
    else:
        X_train_small, y_train_small = X_train, y_train
    
    print(f"\n  {model_type.upper()}: {len(X_train_small)} образцов")
    
    # DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train_small), torch.LongTensor(y_train_small))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    batch_size = min(config.BATCH_SIZE, max(16, len(X_train_small) // 2))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Модель
    model = get_model(model_type, config)
    trainer = Trainer(model, config)
    
    # Обучение
    start_time = time.time()
    best_val_acc = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    # Тестирование
    test_loss, test_acc = trainer.validate(test_loader)
    
    # F1-score
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            logits, _ = model(data, use_masking=False)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average='binary')
    
    return {
        'model': model_type,
        'sample_ratio': sample_ratio,
        'train_size': len(X_train_small),
        'val_acc': round(best_val_acc, 2),
        'test_acc': round(test_acc, 2),
        'test_f1': round(f1, 4),
        'time_sec': round(elapsed, 1)
    }

def main():
    print("=" * 70)
    print("🔬 FEW-SHOT LEARNING (ВСЕ МОДЕЛИ) НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 70)
    
    # Загрузка данных
    X_train, y_train, X_val, y_val, X_test, y_test = load_pm_processed()
    
    print(f"\n📊 Исходные данные:")
    print(f"  Train: {X_train.shape}, метки: {y_train.shape}")
    print(f"  Val:   {X_val.shape}, метки: {y_val.shape}")
    print(f"  Test:  {X_test.shape}, метки: {y_test.shape}")
    
    # Настройка конфига (оптимальная конфигурация)
    Config.WINDOW_LENGTH = X_train.shape[1]
    Config.NUM_CHANNELS = X_train.shape[2]
    Config.NUM_CLASSES = 2
    Config.D_MODEL = 30
    Config.NUM_LAYERS = 2
    Config.NUM_HEADS = 2
    Config.DROPOUT = 0.3
    Config.LAMBDA_1 = 1e-3
    Config.MASK_PROB = 0.15
    Config.BATCH_SIZE = 32
    Config.NUM_EPOCHS = 30
    Config.EARLY_STOPPING_PATIENCE = 10
    
    # Доли выборки
    sample_ratios = [1.0, 0.5, 0.25, 0.1, 0.05]
    
    # Модели для сравнения (все)
    model_types = ['patchtst', 'patchtst_baseline', 'lstm', 'gru', 'tcn', 'transformer']
    
    results = []
    
    for ratio in sample_ratios:
        print(f"\n{'='*60}")
        print(f"📊 РАЗМЕР ВЫБОРКИ: {int(ratio*100)}% ({int(ratio*len(X_train))} образцов)")
        print('='*60)
        
        for model_type in model_types:
            result = run_few_shot_experiment(
                model_type, X_train, y_train, X_val, y_val, X_test, y_test,
                ratio, Config
            )
            results.append(result)
    
    # Сводная таблица
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА FEW-SHOT LEARNING")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Сохранение
    os.makedirs('logs', exist_ok=True)
    df.to_csv('logs/pm_few_shot_results.csv', index=False)
    print("\n💾 Результаты сохранены в logs/pm_few_shot_results.csv")
    
    # Pivot таблицы для диссертации
    pivot_f1 = df.pivot(index='sample_ratio', columns='model', values='test_f1')
    pivot_acc = df.pivot(index='sample_ratio', columns='model', values='test_acc')
    
    print("\n" + "=" * 70)
    print("📋 ТАБЛИЦА ДЛЯ ДИССЕРТАЦИИ: F1-SCORE")
    print("=" * 70)
    print(pivot_f1.round(4).to_string())
    
    print("\n" + "=" * 70)
    print("📋 ТАБЛИЦА ДЛЯ ДИССЕРТАЦИИ: TEST ACCURACY (%)")
    print("=" * 70)
    print(pivot_acc.round(2).to_string())
    
    # Лучшая модель при каждом размере выборки
    print("\n" + "=" * 70)
    print("🏆 ЛУЧШАЯ МОДЕЛЬ ПРИ КАЖДОМ РАЗМЕРЕ ВЫБОРКИ")
    print("=" * 70)
    
    for ratio in sample_ratios:
        subset = df[df['sample_ratio'] == ratio]
        best = subset.loc[subset['test_f1'].idxmax()]
        print(f"  {int(ratio*100)}% ({int(ratio*len(X_train))} обр.): {best['model']} (F1 = {best['test_f1']:.4f})")

if __name__ == '__main__':
    main()