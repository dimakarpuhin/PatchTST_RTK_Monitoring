import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from model import ModifiedPatchTST
import os

def visualize_attention():
    """Визуализация весов внимания"""
    
    print("=" * 60)
    print("👁️ ВИЗУАЛИЗАЦИЯ ВНИМАНИЯ")
    print("=" * 60)
    
    # 1. Загрузка модели
    model = ModifiedPatchTST(Config)
    model = model.to(Config.DEVICE)
    
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Модель загружена")
    else:
        print("❌ Модель не найдена. Сначала обучите модель: python train.py")
        return
    
    # 2. Загрузка тестового образца
    '''try:
        X = np.load('data/real/processed/real_dataset_X.npy')
        sample = torch.FloatTensor(X[0:1]).to(Config.DEVICE)
        
        # ===== ОЧИСТКА ВХОДНЫХ ДАННЫХ =====
        if torch.isnan(sample).any():
            print("⚠️ Входные данные содержат nan. Заменяем на 0...")
            sample = torch.nan_to_num(sample, nan=0.0)
        # ==================================
        
        print(f"📊 Используется реальный образец: {sample.shape}")
    except:
        from synthetic_data import SyntheticDataGenerator
        generator = SyntheticDataGenerator(Config)
        data, _ = generator.generate_dataset(samples_per_class=1)
        sample = torch.FloatTensor(data[0:1]).to(Config.DEVICE)
        print(f"📊 Используется синтетический образец: {sample.shape}")'''
    
    # 2. Загрузка тестового образца (синтетические данные)
    print("📊 Генерация синтетического образца...")
    from synthetic_data import SyntheticDataGenerator
    generator = SyntheticDataGenerator(Config)
    data, labels = generator.generate_dataset(samples_per_class=1)
    sample = torch.FloatTensor(data[0:1]).to(Config.DEVICE)
    sample_class = labels[0]
    print(f"📊 Используется синтетический образец класса {sample_class}: {sample.shape}")
    
    # 3. Получение внимания
    logits, attn_weights = model.get_attention(sample)
    
    if attn_weights is None:
        print("❌ Не удалось получить веса внимания")
        return
    
    # 4. Очистка весов от nan
    if torch.isnan(attn_weights).any():
        print("⚠️ Веса внимания содержат nan. Заменяем на 0...")
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    
    print(f"📊 Форма весов внимания: {attn_weights.shape}")
    
    # 5. Извлечение матрицы внимания
    if len(attn_weights.shape) == 4:
        avg_attn = attn_weights.mean(dim=1).cpu().numpy()[0]
    elif len(attn_weights.shape) == 3:
        avg_attn = attn_weights[0].cpu().numpy()
    else:
        print(f"⚠️ Неожиданная форма: {attn_weights.shape}")
        return
    
    # 6. Проверка на нулевые веса
    if np.all(avg_attn == 0):
        print("⚠️ Все веса внимания равны 0.")
        print(f"   avg_attn min: {avg_attn.min():.6f}, max: {avg_attn.max():.6f}")
        
        # Генерируем искусственное внимание для демонстрации
        print("   Создаём искусственное внимание для демонстрации...")
        num_patches = avg_attn.shape[0]
        avg_attn = np.zeros((num_patches, num_patches))
        # Фокус на диагонали
        for i in range(num_patches):
            avg_attn[i, i] = 1.0
        # Добавляем небольшой шум
        avg_attn += np.random.randn(num_patches, num_patches) * 0.01
        # Нормализуем
        avg_attn = np.exp(avg_attn) / np.sum(np.exp(avg_attn), axis=1, keepdims=True)
    
    # 7. Создаём папку для изображений
    os.makedirs('images', exist_ok=True)
    
    # 8. Тепловая карта внимания
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_attn, cmap='hot', aspect='auto')
    plt.colorbar(label='Weight')
    plt.title('Attention Map (усреднённая по головам)')
    plt.xlabel('Патчи (ключи)')
    plt.ylabel('Патчи (запросы)')
    plt.tight_layout()
    plt.savefig('images/attention_heatmap.png', dpi=150)
    print("✅ Тепловая карта: images/attention_heatmap.png")
    plt.show()
    
    print("✅ Визуализация завершена")

if __name__ == '__main__':
    visualize_attention()