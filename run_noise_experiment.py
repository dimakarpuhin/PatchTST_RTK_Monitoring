# run_noise_experiment.py
# Эксперимент: Устойчивость к шуму (SNR)
# Сравнение моделей при разных уровнях шума

import numpy as np
import torch
import pandas as pd
import time
import os
from config import Config
from model import ModifiedPatchTST, create_model
from train import Trainer
from synthetic_data import SyntheticDataGenerator
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

def get_model(model_type, config):
    """Создание модели по типу"""
    if model_type == 'full':
        config.USE_ADAPTIVE_ENCODING = True
        config.USE_CHANNEL_ATTENTION = True
        config.LAMBDA_2 = 0.1
        config.MASK_PROB = 0.15
        model = create_model(config)
    elif model_type == 'baseline':
        config.USE_ADAPTIVE_ENCODING = False
        config.USE_CHANNEL_ATTENTION = False
        config.LAMBDA_2 = 0
        config.MASK_PROB = 0
        model = create_model(config)
    elif model_type == 'no_fa':
        config.USE_ADAPTIVE_ENCODING = True
        config.USE_CHANNEL_ATTENTION = False
        config.LAMBDA_2 = 0.1
        config.MASK_PROB = 0.15
        model = create_model(config)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    return model

def add_noise(X, snr_db):
    """
    Добавление гауссовского шума с заданным SNR
    """
    snr = 10 ** (snr_db / 10)
    X_noisy = X.copy()
    for i in range(X.shape[0]):
        signal_power = np.mean(X[i] ** 2)
        noise_power = signal_power / snr
        noise = np.random.randn(*X[i].shape) * np.sqrt(noise_power)
        X_noisy[i] = X[i] + noise
    return X_noisy

def train_model(X_train, y_train, X_val, y_val, model_type, config):
    """Обучение модели на данных"""
    
    # Создание DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    batch_size = min(config.BATCH_SIZE, len(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Создание модели
    model = get_model(model_type, config)
    model = model.to(config.DEVICE)
    
    # Обучение
    trainer = Trainer(model, config)
    start_time = time.time()
    best_acc = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    return trainer, best_acc, elapsed

def test_model_with_noise(model, X_test, y_test, snr_db, config):
    """Тестирование модели на зашумлённых данных"""
    # Добавляем шум
    X_noisy = add_noise(X_test, snr_db)
    
    # Конвертация в тензоры
    X_tensor = torch.FloatTensor(X_noisy).to(config.DEVICE)
    y_tensor = torch.LongTensor(y_test).to(config.DEVICE)
    
    # Инференс
    model.eval()
    with torch.no_grad():
        logits, _ = model(X_tensor, use_masking=False)
        preds = torch.argmax(logits, dim=-1)
    
    # Метрики
    acc = accuracy_score(y_test, preds.cpu().numpy())
    f1 = f1_score(y_test, preds.cpu().numpy(), average='macro')
    
    return acc, f1

def main():
    print("=" * 70)
    print("🔬 ЭКСПЕРИМЕНТ: УСТОЙЧИВОСТЬ К ШУМУ (SNR)")
    print("=" * 70)
    
    Config.set_seed()
    Config.ensure_dirs()
    Config.NUM_EPOCHS = 100
    Config.EARLY_STOPPING_PATIENCE = 20
    
    # Параметры данных
    samples_per_class = 500
    print(f"\n📊 Генерация данных: {samples_per_class} образцов на класс")
    
    # Генерация данных
    generator = SyntheticDataGenerator(Config)
    X, y = generator.generate_dataset(samples_per_class=samples_per_class)
    
    # Разделение на train/val/test (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=Config.RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=Config.RANDOM_SEED
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Типы моделей
    model_types = ['full', 'baseline', 'no_fa']
    models = {}
    
    # Обучение моделей
    for model_type in model_types:
        print(f"\n🚀 Обучение модели: {model_type.upper()}")
        trainer, best_acc, elapsed = train_model(
            X_train, y_train, X_val, y_val, model_type, Config
        )
        models[model_type] = trainer.model
        print(f"  ✅ Лучшая точность: {best_acc:.2f}%")
    
    # Тестирование при разных SNR
    snr_levels = [40, 30, 20, 15, 10, 5]
    results = []
    
    print("\n" + "=" * 60)
    print("📊 ТЕСТИРОВАНИЕ ПРИ РАЗНЫХ УРОВНЯХ SNR")
    print("=" * 60)
    
    for snr in snr_levels:
        print(f"\n🔊 SNR = {snr} дБ")
        for model_type, model in models.items():
            acc, f1 = test_model_with_noise(model, X_test, y_test, snr, Config)
            results.append({
                'model': model_type,
                'snr': snr,
                'accuracy': acc,
                'f1_macro': f1
            })
            print(f"  {model_type.upper()}: Acc = {acc:.2f}%, F1 = {f1:.4f}")
    
    # Сводная таблица
    df_results = pd.DataFrame(results)
    
    # Сводная таблица (модели × SNR)
    pivot_acc = df_results.pivot(index='snr', columns='model', values='accuracy')
    pivot_f1 = df_results.pivot(index='snr', columns='model', values='f1_macro')
    
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА: ТОЧНОСТЬ (%)")
    print("=" * 70)
    print(pivot_acc.round(2).to_string())
    
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА: F1-MACRO")
    print("=" * 70)
    print(pivot_f1.round(4).to_string())
    
    # Сохранение
    os.makedirs('logs/noise', exist_ok=True)
    df_results.to_csv('logs/noise/noise_results.csv', index=False)
    pivot_acc.to_csv('logs/noise/noise_pivot_acc.csv')
    pivot_f1.to_csv('logs/noise/noise_pivot_f1.csv')
    
    print("\n💾 Результаты сохранены в logs/noise/")
    
    # Построение графиков
    try:
        plot_noise_results(pivot_acc, pivot_f1)
    except Exception as e:
        print(f"⚠️ Не удалось построить график: {e}")

def plot_noise_results(pivot_acc, pivot_f1):
    """Построение графиков зависимости точности от SNR"""
    try:
        import matplotlib.pyplot as plt
        
        # ===== ДАННЫЕ В ПРОЦЕНТАХ =====
        # Умножаем на 100, чтобы получить проценты
        pivot_acc_percent = pivot_acc * 100
        pivot_f1_percent = pivot_f1 * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = {
            'full': '#2ecc71',      # зелёный
            'baseline': '#e74c3c',  # красный
            'no_fa': '#3498db'      # синий
        }
        
        labels = {
            'full': 'Модифицированный PatchTST',
            'baseline': 'Базовый PatchTST',
            'no_fa': 'PatchTST без FA'
        }
        
        # ===== График 1: Точность (Accuracy) =====
        ax1 = axes[0]
        for model in pivot_acc_percent.columns:
            ax1.plot(pivot_acc_percent.index, pivot_acc_percent[model], 
                    marker='o', linewidth=2.5,
                    color=colors.get(model, '#000000'),
                    label=labels.get(model, model.upper()))
        ax1.set_xlabel('SNR (дБ)', fontsize=12)
        ax1.set_ylabel('Точность (%)', fontsize=12)  # ← ПРОЦЕНТЫ
        ax1.set_title('Устойчивость к шуму: точность', fontsize=14)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 45)
        ax1.set_ylim(0, 105)  # ← 0-100% + запас
        
        # ===== График 2: F1-Macro =====
        ax2 = axes[1]
        for model in pivot_f1_percent.columns:
            ax2.plot(pivot_f1_percent.index, pivot_f1_percent[model], 
                    marker='s', linewidth=2.5,
                    color=colors.get(model, '#000000'),
                    label=labels.get(model, model.upper()))
        ax2.set_xlabel('SNR (дБ)', fontsize=12)
        ax2.set_ylabel('F1-Macro (%)', fontsize=12)  # ← ПРОЦЕНТЫ
        ax2.set_title('Устойчивость к шуму: F1-Macro', fontsize=14)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 45)
        ax2.set_ylim(0, 105)  # ← 0-100% + запас
        
        # ===== Аннотации на первом графике =====
        # Выигрыш при SNR=30
        if 'full' in pivot_acc_percent.columns and 'baseline' in pivot_acc_percent.columns:
            full_30 = pivot_acc_percent.loc[30, 'full']
            baseline_30 = pivot_acc_percent.loc[30, 'baseline']
            gain = full_30 - baseline_30
            ax1.annotate(f'Выигрыш: {gain:.0f}%', 
                        xy=(30, full_30), 
                        xytext=(32, full_30 - 15),
                        arrowprops=dict(arrowstyle='->', color='#2ecc71'),
                        fontsize=10, color='#2ecc71')
        
        plt.tight_layout()
        os.makedirs('images', exist_ok=True)
        plt.savefig('images/noise_robustness_final.png', dpi=300, bbox_inches='tight')
        plt.savefig('logs/noise/noise_plot.png', dpi=150, bbox_inches='tight')
        print("📈 График сохранён: images/noise_robustness_final.png")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Ошибка при построении графика: {e}")

if __name__ == '__main__':
    main()