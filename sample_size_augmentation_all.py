# sample_size_augmentation_all.py
# Сравнение всех размеров выборки с аугментацией и без

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
from synthetic_data import RTKDataset

def load_pm_processed():
    X_train = np.load(f'{Config.PM_PROCESSED_PATH}/X_train.npy')
    y_train = np.load(f'{Config.PM_PROCESSED_PATH}/y_train.npy')
    X_val = np.load(f'{Config.PM_PROCESSED_PATH}/X_val.npy')
    y_val = np.load(f'{Config.PM_PROCESSED_PATH}/y_val.npy')
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_experiment_with_history(X_train, y_train, X_val, y_val, X_test, y_test, config, sample_size_pct, n_samples, use_augmentation=False):
    """Запуск эксперимента с сохранением истории"""
    
    if use_augmentation:
        train_dataset = RTKDataset(X_train, y_train, config, augment=True)
        aug_label = 'С аугм.'
    else:
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        aug_label = 'Без аугм.'
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    batch_size = min(config.BATCH_SIZE, len(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = create_model(config)
    trainer = Trainer(model, config)
    
    best_val_acc = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    history_df = pd.DataFrame(trainer.history)
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
        'history': history_df,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': f1,
        'aug_label': aug_label
    }

def plot_convergence(history, sample_size, n_samples, aug_label, save_dir='images/sample_size_aug_all'):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    if 'train_loss' in history.columns:
        ax1.plot(history['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
    if 'val_loss' in history.columns:
        ax1.plot(history['val_loss'], label='Val Loss', color='#3498db', linewidth=2)
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Loss ({sample_size}%, {n_samples} обр., {aug_label})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    if 'train_acc' in history.columns:
        ax2.plot(history['train_acc'], label='Train Accuracy', color='#e74c3c', linewidth=2)
    if 'val_acc' in history.columns:
        ax2.plot(history['val_acc'], label='Val Accuracy', color='#3498db', linewidth=2)
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Точность (%)', fontsize=12)
    ax2.set_title(f'Accuracy ({sample_size}%, {n_samples} обр., {aug_label})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{save_dir}/convergence_{sample_size}pct_{aug_label}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename

def main():
    print("=" * 70)
    print("🔬 СРАВНЕНИЕ ВСЕХ РАЗМЕРОВ ВЫБОРКИ (С АУГМЕНТАЦИЕЙ И БЕЗ)")
    print("=" * 70)
    
    X_train_full, y_train_full, X_val, y_val, X_test, y_test = load_pm_processed()
    
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
    
    sample_configs = [
        (2, 0.02),
        (5, 0.05),
        (10, 0.10),
        (25, 0.25),
        (50, 0.50),
        (100, 1.0),
    ]
    
    all_results = []
    
    for size_pct, ratio in sample_configs:
        n_samples = int(len(X_train_full) * ratio)
        print(f"\n📊 Размер выборки: {size_pct}% ({n_samples} образцов)")
        
        if ratio >= 1.0:
            X_train = X_train_full
            y_train = y_train_full
        else:
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full,
                train_size=ratio,
                stratify=y_train_full,
                random_state=42
            )
        
        # БЕЗ аугментации
        print(f"   🔹 Без аугментации...")
        result_no = run_experiment_with_history(
            X_train, y_train, X_val, y_val, X_test, y_test, 
            Config, size_pct, n_samples, use_augmentation=False
        )
        plot_no = plot_convergence(
            result_no['history'], size_pct, n_samples, 'Без аугм.'
        )
        print(f"      ✅ {plot_no}")
        
        # С аугментацией
        print(f"   🔸 С аугментацией...")
        result_aug = run_experiment_with_history(
            X_train, y_train, X_val, y_val, X_test, y_test, 
            Config, size_pct, n_samples, use_augmentation=True
        )
        plot_aug = plot_convergence(
            result_aug['history'], size_pct, n_samples, 'С аугм.'
        )
        print(f"      ✅ {plot_aug}")
        
        all_results.append({
            'sample_size': size_pct,
            'n_samples': n_samples,
            'no_aug_f1': result_no['test_f1'],
            'no_aug_val_acc': result_no['best_val_acc'],
            'no_aug_test_acc': result_no['test_acc'],
            'with_aug_f1': result_aug['test_f1'],
            'with_aug_val_acc': result_aug['best_val_acc'],
            'with_aug_test_acc': result_aug['test_acc'],
            'improvement': result_aug['test_f1'] - result_no['test_f1']
        })
    
    # ============================================================
    # СВОДНАЯ ТАБЛИЦА
    # ============================================================
    
    df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА")
    print("=" * 70)
    print(df.round(4).to_string(index=False))
    
    os.makedirs('logs', exist_ok=True)
    df.to_csv('logs/sample_size_aug_all.csv', index=False)
    
    # ============================================================
    # ГРАФИК: F1 vs размер выборки
    # ============================================================
    
    plt.figure(figsize=(10, 6))
    
    x = df['n_samples'].values
    y_no = df['no_aug_f1'].values
    y_aug = df['with_aug_f1'].values
    
    plt.plot(x, y_no, marker='o', markersize=8, color='#e74c3c', linewidth=2, label='Без аугментации')
    plt.plot(x, y_aug, marker='s', markersize=8, color='#2ecc71', linewidth=2, label='С аугментацией')
    
    plt.xlabel('Размер обучающей выборки (образцов)', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.title('Влияние аугментации на качество классификации', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks(x, [f'{n}\n({s}%)' for n, s in zip(x, df['sample_size'])])
    
    plt.tight_layout()
    plt.savefig('images/augmentation_all_sizes.png', dpi=300)
    print("\n✅ График: images/augmentation_all_sizes.png")
    plt.show()
    
    # ============================================================
    # ВЫВОД
    # ============================================================
    
    print("\n" + "=" * 70)
    print("📋 ТАБЛИЦА ДЛЯ ДИССЕРТАЦИИ")
    print("=" * 70)
    print("\n| Размер | Образцов | Без аугм. (F1) | С аугм. (F1) | Прирост |")
    print("|--------|----------|----------------|--------------|---------|")
    for _, row in df.iterrows():
        print(f"| {int(row['sample_size'])}% | {int(row['n_samples'])} | {row['no_aug_f1']:.3f} | {row['with_aug_f1']:.3f} | +{row['improvement']:.3f} |")
    
    print("\n" + "=" * 70)
    print("📝 ВЫВОД")
    print("=" * 70)
    
    avg_improvement = df['improvement'].mean()
    max_improvement = df['improvement'].max()
    max_row = df.loc[df['improvement'].idxmax()]
    
    print(f"\n✅ Средний прирост F1 от аугментации: +{avg_improvement:.3f}")
    print(f"✅ Максимальный прирост: +{max_improvement:.3f} при {int(max_row['sample_size'])}% выборки")
    print("\n📂 Графики сохранены в папке images/sample_size_aug_all/")

if __name__ == '__main__':
    main()