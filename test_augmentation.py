# test_augmentation.py
# Эксперимент: оценка вклада аугментации на малой выборке

import numpy as np
import torch
import pandas as pd
import time
import os
from config import Config
from model import create_model
from train import Trainer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def load_pm_processed():
    X_train = np.load(f'{Config.PM_PROCESSED_PATH}/X_train.npy')
    y_train = np.load(f'{Config.PM_PROCESSED_PATH}/y_train.npy')
    X_val = np.load(f'{Config.PM_PROCESSED_PATH}/X_val.npy')
    y_val = np.load(f'{Config.PM_PROCESSED_PATH}/y_val.npy')
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_dataloader(X, y, config, augment=False):
    """Создание DataLoader с возможностью отключения аугментации"""
    if augment:
        from synthetic_data import RTKDataset
        dataset = RTKDataset(X, y, config, augment=True)
    else:
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    return loader

def run_experiment_without_augmentation(X_train, y_train, X_val, y_val, X_test, y_test, config):
    """Обучение БЕЗ аугментации"""
    print("\n" + "=" * 60)
    print("🧪 ЭКСПЕРИМЕНТ: БЕЗ аугментации")
    print("=" * 60)
    
    train_loader = create_dataloader(X_train, y_train, config, augment=False)
    val_loader = create_dataloader(X_val, y_val, config, augment=False)
    test_loader = create_dataloader(X_test, y_test, config, augment=False)
    
    model = create_model(config)
    trainer = Trainer(model, config)
    
    start_time = time.time()
    best_val_acc = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    test_loss, test_acc = trainer.validate(test_loader)
    
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
        'augmentation': 'Без аугментации',
        'val_acc': round(best_val_acc, 2),
        'test_acc': round(test_acc, 2),
        'test_f1': round(f1, 4),
        'time_sec': round(elapsed, 1)
    }

def run_experiment_with_augmentation(X_train, y_train, X_val, y_val, X_test, y_test, config):
    """Обучение С аугментацией"""
    print("\n" + "=" * 60)
    print("🧪 ЭКСПЕРИМЕНТ: С аугментацией")
    print("=" * 60)
    
    train_loader = create_dataloader(X_train, y_train, config, augment=True)
    val_loader = create_dataloader(X_val, y_val, config, augment=False)
    test_loader = create_dataloader(X_test, y_test, config, augment=False)
    
    model = create_model(config)
    trainer = Trainer(model, config)
    
    start_time = time.time()
    best_val_acc = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    test_loss, test_acc = trainer.validate(test_loader)
    
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
        'augmentation': 'С аугментацией',
        'val_acc': round(best_val_acc, 2),
        'test_acc': round(test_acc, 2),
        'test_f1': round(f1, 4),
        'time_sec': round(elapsed, 1)
    }

def main():
    print("=" * 70)
    print("🔬 ЭКСПЕРИМЕНТ: ОЦЕНКА ВКЛАДА АУГМЕНТАЦИИ")
    print("=" * 70)
    
    X_train_full, y_train_full, X_val, y_val, X_test, y_test = load_pm_processed()
    
    sample_ratio = 0.25
    X_train, _, y_train, _ = train_test_split(
        X_train_full, y_train_full, train_size=sample_ratio, 
        stratify=y_train_full, random_state=42
    )
    
    print(f"\n📊 Данные:")
    print(f"  Train (25% от полной): {X_train.shape}, метки: {y_train.shape}")
    print(f"  Val: {X_val.shape}, метки: {y_val.shape}")
    print(f"  Test: {X_test.shape}, метки: {y_test.shape}")
    
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
    
    results = []
    
    results.append(run_experiment_without_augmentation(
        X_train, y_train, X_val, y_val, X_test, y_test, Config
    ))
    
    results.append(run_experiment_with_augmentation(
        X_train, y_train, X_val, y_val, X_test, y_test, Config
    ))
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА: ВЛИЯНИЕ АУГМЕНТАЦИИ")
    print("=" * 70)
    print(df.to_string(index=False))
    
    os.makedirs('logs', exist_ok=True)
    df.to_csv('logs/augmentation_effect.csv', index=False)
    print("\n💾 Результаты сохранены в logs/augmentation_effect.csv")
    
    print("\n" + "=" * 70)
    print("📋 ТАБЛИЦА ДЛЯ ДИССЕРТАЦИИ")
    print("=" * 70)
    print("\n| Эксперимент | Val Acc (%) | Test Acc (%) | F1-score |")
    print("|-------------|-------------|--------------|----------|")
    for _, row in df.iterrows():
        print(f"| {row['augmentation']} | {row['val_acc']:.2f} | {row['test_acc']:.2f} | {row['test_f1']:.4f} |")

if __name__ == '__main__':
    main()