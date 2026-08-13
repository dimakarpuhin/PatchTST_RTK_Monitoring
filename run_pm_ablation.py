# plot_all_graphics.py
# Построение ВСЕХ графиков для диссертации в одном скрипте

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

# Настройки шрифта
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('images', exist_ok=True)

print("=" * 60)
print("📈 ПОСТРОЕНИЕ ВСЕХ ГРАФИКОВ ДЛЯ ДИССЕРТАЦИИ")
print("=" * 60)

# ============================================================
# 1. ГРАФИК: СХОДИМОСТЬ ОДНОЙ МОДЕЛИ (4 линии)
# ============================================================

def plot_single_convergence():
    print("1. Сходимость одной модели...")
    try:
        df = pd.read_csv('logs/training_history.csv')
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.plot(df['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
        ax1.plot(df['val_loss'], label='Val Loss', color='#3498db', linewidth=2)
        ax1.set_xlabel('Эпоха', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Функция потерь (модифицированный PatchTST)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(df['train_acc'], label='Train Accuracy', color='#e74c3c', linewidth=2)
        ax2.plot(df['val_acc'], label='Val Accuracy', color='#3498db', linewidth=2)
        ax2.set_xlabel('Эпоха', fontsize=12)
        ax2.set_ylabel('Точность (%)', fontsize=12)
        ax2.set_title('Точность классификации (модифицированный PatchTST)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/convergence_single_model.png', dpi=300)
        print("   ✅ images/convergence_single_model.png")
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Ошибка: {e}")

# ============================================================
# 2. ГРАФИК: СРАВНЕНИЕ ВСЕХ 6 МОДЕЛЕЙ
# ============================================================

def plot_all_models():
    print("2. Сравнение всех 6 моделей...")
    
    models = {
        'PatchTST (модиф.)': {'file': 'logs/training_history_patchtst.csv', 'color': '#2ecc71', 'linestyle': '-'},
        'PatchTST (базовый)': {'file': 'logs/training_history_patchtst_baseline.csv', 'color': '#e74c3c', 'linestyle': '--'},
        'LSTM': {'file': 'logs/training_history_lstm.csv', 'color': '#3498db', 'linestyle': '-.'},
        'GRU': {'file': 'logs/training_history_gru.csv', 'color': '#f39c12', 'linestyle': ':'},
        'TCN': {'file': 'logs/training_history_tcn.csv', 'color': '#9b59b6', 'linestyle': '-'},
        'Transformer': {'file': 'logs/training_history_transformer.csv', 'color': '#1abc9c', 'linestyle': '--'}
    }
    
    loaded = {}
    for name, cfg in models.items():
        if os.path.exists(cfg['file']):
            loaded[name] = {'df': pd.read_csv(cfg['file']), 'cfg': cfg}
    
    if not loaded:
        print("   ⚠️ Нет данных для сравнения")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    for name, data in loaded.items():
        df = data['df']
        cfg = data['cfg']
        if 'val_loss' in df.columns:
            ax1.plot(df['val_loss'], label=name, color=cfg['color'], linestyle=cfg['linestyle'], linewidth=2)
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Сравнение сходимости: Loss', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for name, data in loaded.items():
        df = data['df']
        cfg = data['cfg']
        if 'val_acc' in df.columns:
            ax2.plot(df['val_acc'], label=name, color=cfg['color'], linestyle=cfg['linestyle'], linewidth=2)
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Точность (%)', fontsize=12)
    ax2.set_title('Сравнение сходимости: точность', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('images/convergence_all_models.png', dpi=300)
    print("   ✅ images/convergence_all_models.png")
    plt.close()

# ============================================================
# 3. ГРАФИК: СРАВНЕНИЕ 3 ЛУЧШИХ МОДЕЛЕЙ
# ============================================================

def plot_best_models():
    print("3. Сравнение 3 лучших моделей...")
    
    models = {
        'PatchTST (модиф.)': {'file': 'logs/training_history_patchtst.csv', 'color': '#2ecc71'},
        'GRU': {'file': 'logs/training_history_gru.csv', 'color': '#f39c12'},
        'Transformer': {'file': 'logs/training_history_transformer.csv', 'color': '#1abc9c'}
    }
    
    loaded = {}
    for name, cfg in models.items():
        if os.path.exists(cfg['file']):
            loaded[name] = pd.read_csv(cfg['file'])
    
    if len(loaded) < 2:
        print("   ⚠️ Недостаточно данных")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    for name, df in loaded.items():
        ax1.plot(df['val_loss'], label=name, linewidth=2, color=models[name]['color'])
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Сравнение лучших: Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for name, df in loaded.items():
        ax2.plot(df['val_acc'], label=name, linewidth=2, color=models[name]['color'])
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Точность (%)', fontsize=12)
    ax2.set_title('Сравнение лучших: точность', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('images/convergence_best_models.png', dpi=300)
    print("   ✅ images/convergence_best_models.png")
    plt.close()

# ============================================================
# 4. ГРАФИК: FEW-SHOT LEARNING
# ============================================================

def plot_few_shot():
    print("4. Few-Shot Learning...")
    try:
        df = pd.read_csv('logs/pm_few_shot_results.csv')
        plt.figure(figsize=(10, 6))
        
        models = df['model'].unique()
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, model in enumerate(models):
            subset = df[df['model'] == model].sort_values('sample_ratio')
            plt.plot(subset['sample_ratio'] * 100, subset['test_f1'], 
                    marker='o', linewidth=2, color=colors[i % len(colors)], label=model)
        
        plt.xlabel('Размер выборки (%)', fontsize=12)
        plt.ylabel('F1-score', fontsize=12)
        plt.title('Few-Shot Learning: F1 vs размер выборки', fontsize=14)
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('images/few_shot_convergence.png', dpi=300)
        print("   ✅ images/few_shot_convergence.png")
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Ошибка: {e}")

# ============================================================
# 5. ГРАФИК: ABLATION STUDY (СТОЛБЧАТАЯ ДИАГРАММА)
# ============================================================

def plot_ablation():
    print("5. Ablation Study...")
    
    configs = ['Full', 'Без FA', 'Без CRCE', 'Без LAPE', 'Без SPM', 'Baseline']
    f1 = [0.569, 0.467, 0.615, 0.543, 0.543, 0.536]
    colors = ['#2ecc71', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configs, f1, color=colors, edgecolor='black')
    plt.ylabel('F1-score', fontsize=12)
    plt.title('Ablation Study: вклад модификаций', fontsize=14)
    plt.ylim(0, 0.7)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, f1):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('images/ablation_results.png', dpi=300)
    print("   ✅ images/ablation_results.png")
    plt.close()

# ============================================================
# 6. ГРАФИК: УСТОЙЧИВОСТЬ К ШУМУ (SNR)
# ============================================================

def plot_noise():
    print("6. Устойчивость к шуму...")
    
    snr = [5, 10, 15, 20, 30, 40]
    patchtst = [23, 44, 57, 72, 84, 86]
    baseline = [24, 39, 49, 67, 73, 74]
    no_fa = [20, 20, 20, 20, 20, 20]
    
    plt.figure(figsize=(10, 6))
    plt.plot(snr, patchtst, marker='o', linewidth=2, label='PatchTST (модиф.)', color='#2ecc71')
    plt.plot(snr, baseline, marker='s', linewidth=2, label='PatchTST (базовый)', color='#e74c3c')
    plt.plot(snr, no_fa, marker='^', linewidth=2, label='Без FA', color='#3498db', linestyle='--')
    
    plt.xlabel('SNR (дБ)', fontsize=12)
    plt.ylabel('Точность (%)', fontsize=12)
    plt.title('Устойчивость к шуму (SNR)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('images/noise_robustness.png', dpi=300)
    print("   ✅ images/noise_robustness.png")
    plt.close()

# ============================================================
# 7. ВЫВОД СПИСКА ВСЕХ ГРАФИКОВ
# ============================================================

print("\n" + "=" * 60)
print("📊 ВСЕ ГРАФИКИ СОЗДАНЫ")
print("=" * 60)

graphics = [
    'convergence_single_model.png',
    'convergence_all_models.png',
    'convergence_best_models.png',
    'few_shot_convergence.png',
    'ablation_results.png',
    'noise_robustness.png'
]

for g in graphics:
    if os.path.exists(f'images/{g}'):
        print(f"   ✅ {g}")
    else:
        print(f"   ❌ {g} (не найден)")

print("\n📂 Папка: images/")
print("=" * 60)

# ============================================================
# ЗАПУСК ВСЕХ ФУНКЦИЙ
# ============================================================

if __name__ == '__main__':
    plot_single_convergence()
    plot_all_models()
    plot_best_models()
    plot_few_shot()
    plot_ablation()
    plot_noise()