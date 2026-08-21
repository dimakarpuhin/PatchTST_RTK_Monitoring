# visualize_attention_pm.py
# Визуализация весов внимания для PM данных

import numpy as np
import torch
import matplotlib.pyplot as plt
from config import Config
from model import create_model
from torch.utils.data import DataLoader, TensorDataset
import os

plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False

def load_pm_processed():
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_test, y_test

def get_attention(model, x):
    """Получение весов внимания для одного образца"""
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model.get_attention(x)
        if attn_weights is None:
            return None
        # Усредняем по головам
        if len(attn_weights.shape) == 4:
            avg_attn = attn_weights.mean(dim=1).cpu().numpy()[0]
        else:
            avg_attn = attn_weights[0].cpu().numpy()
    return avg_attn

def plot_attention(attn_matrix, title, save_path):
    """Построение тепловой карты внимания"""
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_matrix, cmap='hot', aspect='auto')
    plt.colorbar(label='Weight')
    plt.xlabel('Патчи (ключи)', fontsize=12)
    plt.ylabel('Патчи (запросы)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ {save_path}")

def main():
    print("=" * 60)
    print("👁️ ВИЗУАЛИЗАЦИЯ ВНИМАНИЯ (PM ДАННЫЕ)")
    print("=" * 60)
    
    # Загрузка данных
    X_test, y_test = load_pm_processed()
    
    # Настройка конфига
    Config.WINDOW_LENGTH = X_test.shape[1]
    Config.NUM_CHANNELS = X_test.shape[2]
    Config.NUM_CLASSES = 2
    Config.D_MODEL = 30
    Config.NUM_LAYERS = 2
    Config.NUM_HEADS = 2
    Config.DROPOUT = 0.3
    Config.USE_ADAPTIVE_ENCODING = True
    Config.USE_CHANNEL_ATTENTION = True
    Config.LAMBDA_2 = 0.0
    Config.MASK_PROB = 0.15
    
    # Загрузка модели
    print("\n📥 Загрузка модели...")
    model = create_model(Config)
    
    # Пробуем загрузить модель из разных возможных файлов
    model_paths = [
        'models/patchtst_model.pth',
        'models/patchtst_model_best.pth',
        'models/patchtst_model_fixed.pth'
    ]
    
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=Config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Модель загружена из {path}")
                loaded = True
                break
            except:
                continue
    
    if not loaded:
        print("❌ Модель не найдена. Сначала обучите модель: python train_pm.py")
        return
    
    model = model.to(Config.DEVICE)
    os.makedirs('images', exist_ok=True)
    
    # ============================================================
    # 1. Образец класса "Норма" (y = 0)
    # ============================================================
    idx_normal = np.where(y_test == 0)[0][0]
    x_normal = torch.FloatTensor(X_test[idx_normal:idx_normal+1]).to(Config.DEVICE)
    
    print("\n📊 Образец класса 'Норма'")
    attn_normal = get_attention(model, x_normal)
    if attn_normal is not None:
        plot_attention(attn_normal, 
                      'Внимание: образец "Норма"',
                      'images/attention_normal.png')
    
    # ============================================================
    # 2. Образец класса "Отказ" (y = 1)
    # ============================================================
    idx_fault = np.where(y_test == 1)[0][0]
    x_fault = torch.FloatTensor(X_test[idx_fault:idx_fault+1]).to(Config.DEVICE)
    
    print("\n📊 Образец класса 'Отказ'")
    attn_fault = get_attention(model, x_fault)
    if attn_fault is not None:
        plot_attention(attn_fault, 
                      'Внимание: образец "Отказ"',
                      'images/attention_fault.png')
    
    # ============================================================
    # 3. Сравнение на одном графике
    # ============================================================
    if attn_normal is not None and attn_fault is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        im1 = axes[0].imshow(attn_normal, cmap='hot', aspect='auto')
        axes[0].set_title('"Норма"', fontsize=14)
        axes[0].set_xlabel('Патчи (ключи)')
        axes[0].set_ylabel('Патчи (запросы)')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(attn_fault, cmap='hot', aspect='auto')
        axes[1].set_title('"Отказ"', fontsize=14)
        axes[1].set_xlabel('Патчи (ключи)')
        axes[1].set_ylabel('Патчи (запросы)')
        plt.colorbar(im2, ax=axes[1])
        
        plt.suptitle('Сравнение карт внимания', fontsize=16)
        plt.tight_layout()
        plt.savefig('images/attention_comparison.png', dpi=300)
        plt.close()
        print("   ✅ images/attention_comparison.png")
    
    print("\n" + "=" * 60)
    print("✅ ВСЕ ГРАФИКИ СОХРАНЕНЫ В ПАПКЕ images/")
    print("   - attention_normal.png")
    print("   - attention_fault.png")
    print("   - attention_comparison.png")
    print("=" * 60)

if __name__ == '__main__':
    main()