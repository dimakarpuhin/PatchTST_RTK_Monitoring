# run_pm_ablation.py
# Ablation Study на реальных данных Predictive Maintenance

import numpy as np
import torch
import pandas as pd
import time
import os
from config import Config
from model import create_model
from train import Trainer
from torch.utils.data import DataLoader, TensorDataset

def load_pm_processed():
    X_train = np.load(f'{Config.PM_PROCESSED_PATH}/X_train.npy')
    y_train = np.load(f'{Config.PM_PROCESSED_PATH}/y_train.npy')
    X_val = np.load(f'{Config.PM_PROCESSED_PATH}/X_val.npy')
    y_val = np.load(f'{Config.PM_PROCESSED_PATH}/y_val.npy')
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_ablation_experiment(name, config_mods):
    """Запуск одного эксперимента Ablation Study"""
    
    print("=" * 60)
    print(f"🧪 ЭКСПЕРИМЕНТ: {name}")
    print("=" * 60)
    
    # Загрузка данных
    X_train, y_train, X_val, y_val, X_test, y_test = load_pm_processed()
    
    # Настройка Config
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
    
    # Применяем изменения
    original_values = {}
    for key, value in config_mods.items():
        if hasattr(Config, key):
            original_values[key] = getattr(Config, key)
            setattr(Config, key, value)
            print(f"   {key} = {value}")
    
    # DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Модель
    model = create_model(Config)
    trainer = Trainer(model, Config)
    
    # Обучение
    start_time = time.time()
    best_acc = trainer.train(train_loader, val_loader, Config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    # Тестирование
    test_loss, test_acc = trainer.validate(test_loader)
    
    from sklearn.metrics import f1_score
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            logits, _ = model(data, use_masking=False)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average='binary')
    
    # Восстанавливаем значения
    for key, value in original_values.items():
        setattr(Config, key, value)
    
    return {
        'experiment': name,
        'val_acc': best_acc,
        'test_acc': test_acc,
        'test_f1': f1,
        'time_sec': round(elapsed, 1)
    }

def main():
    print("=" * 70)
    print("🔬 ABLATION STUDY НА РЕАЛЬНЫХ ДАННЫХ (PM)")
    print("=" * 70)
    
    results = []
    
    # 1. Полная модель
    results.append(run_ablation_experiment(
        'Full Model', {}
    ))
    
    # 2. Без Adaptive Encoding
    results.append(run_ablation_experiment(
        'Without LAPE', {'USE_ADAPTIVE_ENCODING': False}
    ))
    
    # 3. Без Channel Attention
    results.append(run_ablation_experiment(
        'Without FA', {'USE_CHANNEL_ATTENTION': False}
    ))
    
    # 4. Без Contrastive Loss
    results.append(run_ablation_experiment(
        'Without CRCE', {'LAMBDA_2': 0}
    ))
    
    # 5. Без Masking
    results.append(run_ablation_experiment(
        'Without SPM', {'MASK_PROB': 0}
    ))
    
    # 6. Базовый PatchTST
    results.append(run_ablation_experiment(
        'Baseline', {
            'USE_ADAPTIVE_ENCODING': False,
            'USE_CHANNEL_ATTENTION': False,
            'LAMBDA_2': 0,
            'MASK_PROB': 0
        }
    ))
    
    # Сводная таблица
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА ABLATION STUDY (РЕАЛЬНЫЕ ДАННЫЕ)")
    print("=" * 70)
    print(df.to_string(index=False))
    
    df.to_csv('logs/pm_ablation_results.csv', index=False)
    print("\n💾 Результаты сохранены в logs/pm_ablation_results.csv")

if __name__ == '__main__':
    main()