# train_models_pm.py
# Сравнение моделей на реальных данных Predictive Maintenance

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
        # Лучшая конфигурация из Ablation Study: без CRCE
        config.USE_ADAPTIVE_ENCODING = True
        config.USE_CHANNEL_ATTENTION = True
        config.LAMBDA_2 = 0.0
        config.MASK_PROB = 0.15
        model = create_model(config)
    elif model_type == 'patchtst_baseline':
        # Базовый PatchTST (все модификации выключены)
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

def train_and_evaluate(model_type, train_loader, val_loader, test_loader, config):
    """Обучение и оценка модели"""
    print(f"\n{'='*60}")
    print(f"🚀 {model_type.upper()}")
    print('='*60)
    
    model = get_model(model_type, config)
    trainer = Trainer(model, config)
    
    start_time = time.time()
    best_val_acc = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    train_time = time.time() - start_time
    
    # Тестирование
    test_loss, test_acc = trainer.validate(test_loader)
    
    # Дополнительные метрики
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
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    
    return {
        'model': model_type,
        'val_acc': round(best_val_acc, 2),
        'test_acc': round(test_acc, 2),
        'test_f1': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'time_min': round(train_time / 60, 2),
        'params': sum(p.numel() for p in model.parameters())
    }

def main():
    print("=" * 70)
    print("🔬 СРАВНЕНИЕ МОДЕЛЕЙ НА РЕАЛЬНЫХ ДАННЫХ (PM)")
    print("=" * 70)
    
    # Загрузка данных
    X_train, y_train, X_val, y_val, X_test, y_test = load_pm_processed()
    
    print(f"\n📊 Данные:")
    print(f"  Train: {X_train.shape}, метки: {y_train.shape}")
    print(f"  Val:   {X_val.shape}, метки: {y_val.shape}")
    print(f"  Test:  {X_test.shape}, метки: {y_test.shape}")
    
    # Настройка конфига
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
    
    print(f"\n📊 Параметры модели:")
    print(f"  WINDOW_LENGTH: {Config.WINDOW_LENGTH}")
    print(f"  NUM_CHANNELS: {Config.NUM_CHANNELS}")
    print(f"  NUM_CLASSES: {Config.NUM_CLASSES}")
    print(f"  D_MODEL: {Config.D_MODEL}")
    
    # DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Модели для сравнения (добавлен patchtst_baseline)
    models = ['patchtst', 'patchtst_baseline', 'lstm', 'gru', 'tcn', 'transformer']
    results = []
    
    for model_type in models:
        result = train_and_evaluate(model_type, train_loader, val_loader, test_loader, Config)
        results.append(result)
    
    # Сводная таблица
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Сохранение
    os.makedirs('logs', exist_ok=True)
    df.to_csv('logs/pm_models_comparison.csv', index=False)
    print("\n💾 Результаты сохранены в logs/pm_models_comparison.csv")
    
    # Таблица для диссертации
    print("\n" + "=" * 70)
    print("📋 ТАБЛИЦА ДЛЯ ДИССЕРТАЦИИ")
    print("=" * 70)
    print("\n| Модель | Val Acc (%) | Test Acc (%) | F1-score | Precision | Recall | Время (мин) | Параметры |")
    print("|--------|-------------|--------------|----------|-----------|--------|-------------|-----------|")
    for _, row in df.iterrows():
        print(f"| {row['model']} | {row['val_acc']:.2f} | {row['test_acc']:.2f} | {row['test_f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['time_min']:.2f} | {row['params']} |")

if __name__ == '__main__':
    main()