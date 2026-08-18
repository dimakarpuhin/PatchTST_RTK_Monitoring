# Анализ зависимости качества от объёма обучающей выборки

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
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

def load_pm_processed():
    X_train = np.load(f'{Config.PM_PROCESSED_PATH}/X_train.npy')
    y_train = np.load(f'{Config.PM_PROCESSED_PATH}/y_train.npy')
    X_val = np.load(f'{Config.PM_PROCESSED_PATH}/X_val.npy')
    y_val = np.load(f'{Config.PM_PROCESSED_PATH}/y_val.npy')
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_experiment_with_history(X_train, y_train, X_val, y_val, X_test, y_test, config, sample_size_pct):
    """Запуск эксперимента с сохранением истории обучения"""
    
    print(f"   Обучающая выборка: {len(X_train)} образцов")
    print(f"   Класс 0: {np.sum(y_train == 0)}, Класс 1: {np.sum(y_train == 1)}")
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    batch_size = min(config.BATCH_SIZE, len(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = create_model(config)
    trainer = Trainer(model, config)
    
    # Обучение
    best_val_acc = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    
    # Сохраняем историю
    history_df = pd.DataFrame(trainer.history)
    
    # Тестирование
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
        'history': history_df,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': f1
    }

def plot_convergence(history, sample_size, n_samples, save_dir='images/sample_size'):
    """Построение графика сходимости для одного размера выборки"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss
    ax1 = axes[0]
    if 'train_loss' in history.columns:
        ax1.plot(history['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
    if 'val_loss' in history.columns:
        ax1.plot(history['val_loss'], label='Val Loss', color='#3498db', linewidth=2)
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Функция потерь ({sample_size}%, {n_samples} образцов)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax2 = axes[1]
    if 'train_acc' in history.columns:
        ax2.plot(history['train_acc'], label='Train Accuracy', color='#e74c3c', linewidth=2)
    if 'val_acc' in history.columns:
        ax2.plot(history['val_acc'], label='Val Accuracy', color='#3498db', linewidth=2)
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Точность (%)', fontsize=12)
    ax2.set_title(f'Точность классификации ({sample_size}%, {n_samples} образцов)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{save_dir}/convergence_{sample_size}pct.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename

def main():
    print("=" * 70)
    print("🔬 АНАЛИЗ ЗАВИСИМОСТИ КАЧЕСТВА ОТ ОБЪЁМА ВЫБОРКИ")
    print("=" * 70)
    
    X_train_full, y_train_full, X_val, y_val, X_test, y_test = load_pm_processed()
    
    print(f"\n📊 Полная обучающая выборка: {len(X_train_full)} образцов")
    print(f"   Класс 0: {np.sum(y_train_full == 0)}, Класс 1: {np.sum(y_train_full == 1)}")
    
    # Настройка конфига (лучшая модель без CRCE)
    Config.WINDOW_LENGTH = X_train_full.shape[1]
    Config.NUM_CHANNELS = X_train_full.shape[2]
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
    
    # ===== РАЗМЕРЫ ВЫБОРКИ =====
    # (процент, доля от полной)
    sample_configs = [
        (100, 1.0),
        (50, 0.5),
        (25, 0.25),
        (10, 0.10),
        (5, 0.05),
        (2, 0.02)
    ]
    
    all_results = []
    convergence_plots = []
    
    for size_pct, ratio in sample_configs:
        n_samples = int(len(X_train_full) * ratio)
        
        print(f"\n📊 Размер выборки: {size_pct}% ({n_samples} образцов)")
        
        # ===== ФОРМИРОВАНИЕ ПОДВЫБОРКИ =====
        if ratio >= 1.0:
            # 100% — берём все данные
            X_train = X_train_full
            y_train = y_train_full
        else:
            # Для остальных — случайная подвыборка
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full,
                train_size=ratio,
                stratify=y_train_full,
                random_state=42
            )
        
        # Запуск эксперимента
        result = run_experiment_with_history(
            X_train, y_train, X_val, y_val, X_test, y_test, 
            Config, size_pct
        )
        
        # График сходимости
        plot_file = plot_convergence(
            result['history'], 
            size_pct, 
            n_samples
        )
        convergence_plots.append(plot_file)
        print(f"   ✅ График сохранён: {plot_file}")
        
        all_results.append({
            'sample_size_pct': size_pct,
            'n_samples': n_samples,
            'best_val_acc': result['best_val_acc'],
            'test_acc': result['test_acc'],
            'test_f1': result['test_f1'],
            'history': result['history']
        })
    
    # ============================================================
    # СВОДНАЯ ТАБЛИЦА
    # ============================================================
    
    df = pd.DataFrame([{
        'sample_size': r['sample_size_pct'],
        'n_samples': r['n_samples'],
        'best_val_acc': r['best_val_acc'],
        'test_acc': r['test_acc'],
        'test_f1': r['test_f1']
    } for r in all_results])
    
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА")
    print("=" * 70)
    print(df.round(4).to_string(index=False))
    
    os.makedirs('logs', exist_ok=True)
    df.to_csv('logs/sample_size_analysis.csv', index=False)
    
    # ============================================================
    # ГРАФИК ЗАВИСИМОСТИ F1 ОТ РАЗМЕРА ВЫБОРКИ
    # ============================================================
    
    plt.figure(figsize=(10, 6))
    
    x = df['n_samples'].values
    y = df['test_f1'].values
    
    plt.plot(x, y, marker='o', markersize=8, color='#2ecc71', linewidth=2, label='F1-score')
    plt.axhline(y=0.50, color='red', linestyle='--', linewidth=1.5, label='Порог (F1 = 0.50)')
    
    plt.xlabel('Размер обучающей выборки (образцов)', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.title('Зависимость качества от объёма обучающей выборки', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks(x, [f'{n}\n({s}%)' for n, s in zip(x, df['sample_size'])])
    
    # Аннотация минимального размера
    valid = df[df['test_f1'] >= 0.50]
    if len(valid) > 0:
        min_row = valid.iloc[0]
        plt.annotate(f'Минимальный: {int(min_row["n_samples"])} обр.\n({int(min_row["sample_size"])}%)', 
                    xy=(min_row['n_samples'], min_row['test_f1']), 
                    xytext=(min_row['n_samples']*1.5, min_row['test_f1']+0.05),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('images/sample_size_analysis.png', dpi=300)
    print("\n✅ График зависимости: images/sample_size_analysis.png")
    plt.show()
    
    # ============================================================
    # ВЫВОД
    # ============================================================
    
    print("\n" + "=" * 70)
    print("📋 ТАБЛИЦА ДЛЯ ДИССЕРТАЦИИ")
    print("=" * 70)
    
    print("\n| Размер | Образцов | Val Acc (%) | Test Acc (%) | F1-score |")
    print("|--------|----------|-------------|--------------|----------|")
    for _, row in df.iterrows():
        print(f"| {int(row['sample_size'])}% | {int(row['n_samples'])} | {row['best_val_acc']:.1f} | {row['test_acc']:.1f} | {row['test_f1']:.3f} |")
    
    print("\n" + "=" * 70)
    print("📝 ВЫВОД")
    print("=" * 70)
    
    threshold = 0.50
    valid = df[df['test_f1'] >= threshold]
    if len(valid) > 0:
        min_row = valid.iloc[0]
        print(f"\n✅ Минимальный объём для F1 ≥ {threshold}:")
        print(f"   {int(min_row['sample_size'])}% от полной ({int(min_row['n_samples'])} образцов)")
        print(f"   F1 = {min_row['test_f1']:.3f}")
    else:
        print(f"\n⚠️ При всех размерах выборки F1 < {threshold}")
    
    print("\n📂 Графики сходимости сохранены в папке images/sample_size/")
    print("=" * 70)

if __name__ == '__main__':
    main()