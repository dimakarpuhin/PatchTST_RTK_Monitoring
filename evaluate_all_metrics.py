# evaluate_all_metrics.py
# Полная оценка всех метрик для всех моделей на PM данных

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
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def load_pm_processed():
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

def measure_inference_time(model, data_loader, config, num_runs=100):
    """Измерение времени инференса и коэффициента вариации"""
    sample = next(iter(data_loader))[0].to(config.DEVICE)
    
    # Прогрев
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample, use_masking=False)
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(sample, use_masking=False)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    cv = std_time / mean_time if mean_time > 0 else 0
    
    return mean_time, std_time, cv

def evaluate_model(model_type, test_loader, config):
    """Полная оценка одной модели"""
    print(f"\n{'='*60}")
    print(f"📊 ОЦЕНКА МОДЕЛИ: {model_type.upper()}")
    print('='*60)
    
    model = get_model(model_type, config)
    
    # Загрузка весов
    model_path = f'models/{model_type}_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✅ Загружена модель: {model_path}")
    else:
        print(f"  ⚠️ Модель {model_path} не найдена, используем случайные веса")
    
    model.eval()
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            logits, _ = model(data, use_masking=False)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Метрики
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='binary')
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
    cm = confusion_matrix(all_targets, all_preds)
    
    # Время инференса и CV
    mean_time, std_time, cv = measure_inference_time(model, test_loader, config)
    
    print(f"\n  📊 Результаты:")
    print(f"    Accuracy:   {acc*100:.2f}%")
    print(f"    F1-score:   {f1:.4f}")
    print(f"    Precision:  {precision:.4f}")
    print(f"    Recall:     {recall:.4f}")
    print(f"    ROC-AUC:    {roc_auc:.4f}")
    print(f"    Время инференса: {mean_time:.3f} мс")
    print(f"    CV:         {cv:.4f} ({cv*100:.2f}%)")
    
    return {
        'model': model_type,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'inference_time_ms': mean_time,
        'cv': cv,
        'confusion_matrix': cm
    }

def main():
    print("=" * 70)
    print("📊 ПОЛНАЯ ОЦЕНКА ВСЕХ МОДЕЛЕЙ (ВСЕ МЕТРИКИ)")
    print("=" * 70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_pm_processed()
    
    Config.WINDOW_LENGTH = X_train.shape[1]
    Config.NUM_CHANNELS = X_train.shape[2]
    Config.NUM_CLASSES = 2
    Config.D_MODEL = 30
    Config.NUM_LAYERS = 2
    Config.NUM_HEADS = 2
    Config.DROPOUT = 0.3
    Config.BATCH_SIZE = 32
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model_types = ['patchtst', 'patchtst_baseline', 'lstm', 'gru', 'tcn', 'transformer']
    results = []
    
    for model_type in model_types:
        result = evaluate_model(model_type, test_loader, Config)
        results.append(result)
    
    # Сводная таблица
    df = pd.DataFrame(results)
    
    # Преобразуем проценты
    df['accuracy'] = df['accuracy'] * 100
    
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА ВСЕХ МОДЕЛЕЙ")
    print("=" * 70)
    print(df[['model', 'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'inference_time_ms', 'cv']].to_string(index=False))
    
    # Сохранение
    os.makedirs('logs', exist_ok=True)
    df.to_csv('logs/all_models_full_metrics.csv', index=False)
    print("\n💾 Результаты сохранены в logs/all_models_full_metrics.csv")
    
    # Таблица для диссертации
    print("\n" + "=" * 70)
    print("📋 ТАБЛИЦА ДЛЯ ДИССЕРТАЦИИ")
    print("=" * 70)
    print("\n| Модель | Test Acc (%) | F1-score | Precision | Recall | ROC-AUC | Время (мс) | CV (%) |")
    print("|--------|--------------|----------|-----------|--------|---------|------------|--------|")
    for _, row in df.iterrows():
        print(f"| {row['model']} | {row['accuracy']:.2f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['roc_auc']:.4f} | {row['inference_time_ms']:.3f} | {row['cv']*100:.2f} |")

if __name__ == '__main__':
    main()