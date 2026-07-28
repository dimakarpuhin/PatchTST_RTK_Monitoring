# optimize_pm.py
# Оптимизация гиперпараметров на реальных данных PM

import numpy as np
import torch
import pandas as pd
import time
import os
from config import Config
from model import create_model
from train import Trainer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

def load_pm_processed():
    X_train = np.load(f'{Config.PM_PROCESSED_PATH}/X_train.npy')
    y_train = np.load(f'{Config.PM_PROCESSED_PATH}/y_train.npy')
    X_val = np.load(f'{Config.PM_PROCESSED_PATH}/X_val.npy')
    y_val = np.load(f'{Config.PM_PROCESSED_PATH}/y_val.npy')
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_experiment(params, train_loader, val_loader, test_loader, config):
    """Запуск одного эксперимента с заданными гиперпараметрами"""
    
    # Применяем параметры
    for key, value in params.items():
        setattr(config, key, value)
    
    # ===== ПРОВЕРКА ДЕЛИМОСТИ =====
    if config.D_MODEL % config.NUM_HEADS != 0:
        print(f"   ⚠️ D_MODEL ({config.D_MODEL}) не делится на NUM_HEADS ({config.NUM_HEADS})")
        return None
    
    if config.D_MODEL % config.NUM_CHANNELS != 0:
        print(f"   ⚠️ D_MODEL ({config.D_MODEL}) не делится на NUM_CHANNELS ({config.NUM_CHANNELS})")
        return None
    
    # Создание модели
    model = create_model(config)
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
        **params,
        'val_acc': round(best_val_acc, 2),
        'test_acc': round(test_acc, 2),
        'test_f1': round(f1, 4),
        'time_sec': round(elapsed, 1)
    }

def main():
    print("=" * 70)
    print("🔬 ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ (PM)")
    print("=" * 70)
    
    # Загрузка данных
    X_train, y_train, X_val, y_val, X_test, y_test = load_pm_processed()
    
    # Базовые настройки
    Config.WINDOW_LENGTH = X_train.shape[1]
    Config.NUM_CHANNELS = X_train.shape[2]
    Config.NUM_CLASSES = 2
    Config.LAMBDA_2 = 0.0  # без CRCE
    Config.MASK_PROB = 0.15
    Config.LAMBDA_1 = 1e-3
    Config.NUM_EPOCHS = 20
    Config.EARLY_STOPPING_PATIENCE = 10
    
    # DataLoader (фиксированный)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    # ===== ГРИД ПАРАМЕТРОВ =====
    param_grid = {
        'D_MODEL': [30, 40, 50],
        'NUM_LAYERS': [1, 2, 3],
        'NUM_HEADS': [2, 4],
        'DROPOUT': [0.2, 0.3, 0.4],
        'LEARNING_RATE': [5e-4, 1e-3],
        'BATCH_SIZE': [16, 32]
    }
    
    # Подсчёт комбинаций
    total_combinations = 0
    for d_model in param_grid['D_MODEL']:
        for num_heads in param_grid['NUM_HEADS']:
            if d_model % num_heads == 0:
                total_combinations += len(param_grid['NUM_LAYERS']) * len(param_grid['DROPOUT']) * len(param_grid['LEARNING_RATE']) * len(param_grid['BATCH_SIZE'])
    
    print(f"\n📊 Всего валидных комбинаций: {total_combinations}")
    print(f"⏱️ Примерное время: ~{total_combinations * 0.5:.1f} минут")
    
    max_experiments = 20
    print(f"🔬 Будет запущено: {max_experiments} экспериментов")
    
    results = []
    count = 0
    
    for batch_size in param_grid['BATCH_SIZE']:
        Config.BATCH_SIZE = batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        for lr in param_grid['LEARNING_RATE']:
            Config.LEARNING_RATE = lr
            
            for d_model in param_grid['D_MODEL']:
                if d_model % Config.NUM_CHANNELS != 0:
                    continue
                
                for num_layers in param_grid['NUM_LAYERS']:
                    for num_heads in param_grid['NUM_HEADS']:
                        if d_model % num_heads != 0:
                            continue
                        
                        for dropout in param_grid['DROPOUT']:
                            if count >= max_experiments:
                                break
                            
                            params = {
                                'D_MODEL': d_model,
                                'NUM_LAYERS': num_layers,
                                'NUM_HEADS': num_heads,
                                'DROPOUT': dropout,
                                'LEARNING_RATE': lr,
                                'BATCH_SIZE': batch_size
                            }
                            
                            print(f"\n🧪 Эксперимент {count+1}/{max_experiments}")
                            print(f"   {params}")
                            
                            Config.D_MODEL = d_model
                            Config.NUM_LAYERS = num_layers
                            Config.NUM_HEADS = num_heads
                            Config.DROPOUT = dropout
                            
                            result = run_experiment(params, train_loader, val_loader, test_loader, Config)
                            if result:
                                results.append(result)
                                count += 1
                            
                            if count >= max_experiments:
                                break
                        if count >= max_experiments:
                            break
                    if count >= max_experiments:
                        break
                if count >= max_experiments:
                    break
            if count >= max_experiments:
                break
        if count >= max_experiments:
            break
    
    # Сводная таблица
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("📊 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Лучший результат
    if len(df) > 0:
        best_idx = df['test_f1'].idxmax()
        best = df.iloc[best_idx]
        print("\n" + "=" * 70)
        print("🏆 ЛУЧШИЙ РЕЗУЛЬТАТ")
        print("=" * 70)
        print(f"  Test Acc:  {best['test_acc']:.2f}%")
        print(f"  Test F1:   {best['test_f1']:.4f}")
        print(f"  Val Acc:   {best['val_acc']:.2f}%")
        print(f"  Параметры:")
        for key in ['D_MODEL', 'NUM_LAYERS', 'NUM_HEADS', 'DROPOUT', 'LEARNING_RATE', 'BATCH_SIZE']:
            print(f"    {key}: {best[key]}")
    
    # Сохранение
    os.makedirs('logs', exist_ok=True)
    df.to_csv('logs/pm_optimization_results.csv', index=False)
    print("\n💾 Результаты сохранены в logs/pm_optimization_results.csv")

if __name__ == '__main__':
    main()