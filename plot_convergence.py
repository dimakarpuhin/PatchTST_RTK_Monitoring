# plot_convergence.py
# Построение всех графиков сходимости

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Настройки шрифта
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False

# Создаём папку для графиков
os.makedirs('images', exist_ok=True)

print("=" * 60)
print("📈 ПОСТРОЕНИЕ ВСЕХ ГРАФИКОВ СХОДИМОСТИ")
print("=" * 60)

# ============================================================
# 1. ГРАФИК: ЛУЧШАЯ МОДЕЛЬ (БЕЗ CRCE)
# ============================================================

history_file = 'logs/training_history_patchtst.csv'

if os.path.exists(history_file):
    print(f"✅ Найден файл: {history_file}")
    df = pd.read_csv(history_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1.1. Loss
    ax1 = axes[0]
    if 'train_loss' in df.columns:
        ax1.plot(df['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
    if 'val_loss' in df.columns:
        ax1.plot(df['val_loss'], label='Val Loss', color='#3498db', linewidth=2)
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Потери (Loss)', fontsize=12)
    ax1.set_title('Функция потерь (лучшая модель)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 1.2. Accuracy
    ax2 = axes[1]
    if 'train_acc' in df.columns:
        ax2.plot(df['train_acc'], label='Train Accuracy', color='#e74c3c', linewidth=2)
    if 'val_acc' in df.columns:
        ax2.plot(df['val_acc'], label='Val Accuracy', color='#3498db', linewidth=2)
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Точность (%)', fontsize=12)
    ax2.set_title('Точность классификации (лучшая модель)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/convergence_best_model.png', dpi=300)
    plt.savefig('images/convergence_best_model.pdf', format='pdf')
    print("✅ График 1: images/convergence_best_model.png")
    plt.show()
else:
    print(f"⚠️ Файл {history_file} не найден")

# ============================================================
# 2. ГРАФИК: СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ
# ============================================================

models = {
    'PatchTST (модиф.)': {
        'file': 'logs/training_history_patchtst.csv',
        'color': '#2ecc71',
        'linestyle': '-',
        'linewidth': 2.5
    },
    'PatchTST (базовый)': {
        'file': 'logs/training_history_patchtst_baseline.csv',
        'color': '#e74c3c',
        'linestyle': '--',
        'linewidth': 2.0
    },
    'LSTM': {
        'file': 'logs/training_history_lstm.csv',
        'color': '#3498db',
        'linestyle': '-.',
        'linewidth': 2.0
    },
    'GRU': {
        'file': 'logs/training_history_gru.csv',
        'color': '#f39c12',
        'linestyle': ':',
        'linewidth': 2.0
    },
    'TCN': {
        'file': 'logs/training_history_tcn.csv',
        'color': '#9b59b6',
        'linestyle': '-',
        'linewidth': 2.0
    },
    'Transformer': {
        'file': 'logs/training_history_transformer.csv',
        'color': '#1abc9c',
        'linestyle': '--',
        'linewidth': 2.0
    }
}

loaded_models = {}
for name, cfg in models.items():
    if os.path.exists(cfg['file']):
        df = pd.read_csv(cfg['file'])
        loaded_models[name] = {'df': df, 'cfg': cfg}
        print(f"✅ {name}: загружено {len(df)} эпох")
    else:
        print(f"⚠️ {name}: файл не найден")

if loaded_models:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 2.1. Loss
    ax1 = axes[0]
    for name, data in loaded_models.items():
        df = data['df']
        cfg = data['cfg']
        if 'val_loss' in df.columns:
            ax1.plot(df['val_loss'], 
                    label=name, 
                    color=cfg['color'],
                    linestyle=cfg['linestyle'],
                    linewidth=cfg['linewidth'])
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Потери на валидации (Loss)', fontsize=12)
    ax1.set_title('Сравнение сходимости: функция потерь', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2.2. Accuracy
    ax2 = axes[1]
    for name, data in loaded_models.items():
        df = data['df']
        cfg = data['cfg']
        if 'val_acc' in df.columns:
            ax2.plot(df['val_acc'], 
                    label=name, 
                    color=cfg['color'],
                    linestyle=cfg['linestyle'],
                    linewidth=cfg['linewidth'])
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Точность на валидации (%)', fontsize=12)
    ax2.set_title('Сравнение сходимости: точность', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('images/convergence_all_models.png', dpi=300)
    plt.savefig('images/convergence_all_models.pdf', format='pdf')
    print("✅ График 2: images/convergence_all_models.png")
    plt.show()

# ============================================================
# 3. ГРАФИК: ТОЛЬКО ЛУЧШИЕ МОДЕЛИ
# ============================================================

best_models = ['PatchTST (модиф.)', 'GRU', 'Transformer']
filtered_models = {k: v for k, v in loaded_models.items() if k in best_models}

if len(filtered_models) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    for name, data in filtered_models.items():
        df = data['df']
        cfg = data['cfg']
        if 'val_loss' in df.columns:
            ax1.plot(df['val_loss'], 
                    label=name, 
                    color=cfg['color'],
                    linestyle=cfg['linestyle'],
                    linewidth=cfg['linewidth'])
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Потери на валидации (Loss)', fontsize=12)
    ax1.set_title('Сравнение лучших моделей: Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for name, data in filtered_models.items():
        df = data['df']
        cfg = data['cfg']
        if 'val_acc' in df.columns:
            ax2.plot(df['val_acc'], 
                    label=name, 
                    color=cfg['color'],
                    linestyle=cfg['linestyle'],
                    linewidth=cfg['linewidth'])
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Точность на валидации (%)', fontsize=12)
    ax2.set_title('Сравнение лучших моделей: точность', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('images/convergence_best_models.png', dpi=300)
    print("✅ График 3: images/convergence_best_models.png")
    plt.show()

# ============================================================
# 4. ГРАФИК: FEW-SHOT LEARNING
# ============================================================

few_shot_file = 'logs/pm_few_shot_results.csv'
if os.path.exists(few_shot_file):
    df_fs = pd.read_csv(few_shot_file)
    print(f"✅ Найден файл Few-Shot: {few_shot_file}")
    
    plt.figure(figsize=(10, 6))
    
    models_fs = df_fs['model'].unique()
    colors_fs = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
    
    for i, model in enumerate(models_fs):
        subset = df_fs[df_fs['model'] == model]
        subset = subset.sort_values('sample_ratio')
        plt.plot(subset['sample_ratio'] * 100, subset['test_f1'], 
                marker='o', linewidth=2, color=colors_fs[i % len(colors_fs)], label=model)
    
    plt.xlabel('Размер выборки (%)', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.title('Few-Shot Learning: зависимость F1 от размера выборки', fontsize=14)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/few_shot_convergence.png', dpi=300)
    print("✅ График 4: images/few_shot_convergence.png")
    plt.show()
else:
    print(f"⚠️ Файл {few_shot_file} не найден")

# ============================================================
# СТАТИСТИКА
# ============================================================

print("\n" + "=" * 60)
print("📊 СТАТИСТИКА ПО МОДЕЛЯМ")
print("=" * 60)
print(f"{'Модель':<25} {'Эпохи':<8} {'Лучшая Val Acc (%)':<20}")
print("-" * 60)

for name, data in loaded_models.items():
    df = data['df']
    epochs = len(df)
    best_acc = df['val_acc'].max() if 'val_acc' in df.columns else '—'
    print(f"{name:<25} {epochs:<8} {best_acc:<20.2f}")
print("=" * 60)

print("\n📂 ВСЕ ГРАФИКИ СОХРАНЕНЫ В ПАПКЕ images/:")
print("   1. convergence_best_model.png")
print("   2. convergence_all_models.png")
print("   3. convergence_best_models.png")
print("   4. few_shot_convergence.png")