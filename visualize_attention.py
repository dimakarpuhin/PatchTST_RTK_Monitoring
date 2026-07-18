import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from model import ModifiedPatchTST
from synthetic_data import SyntheticDataGenerator
import os

# Классы неопределённостей
CLASS_NAMES = {
    0: "Норма",
    1: "Скачок напряжения",
    2: "Токовая перегрузка",
    3: "Перегрев",
    4: "Электропомеха"
}

# Цвета для классов
CLASS_COLORS = {
    0: '#2ecc71',   # зелёный
    1: '#e74c3c',   # красный
    2: '#e67e22',   # оранжевый
    3: '#f39c12',   # жёлтый
    4: '#9b59b6'    # фиолетовый
}

def load_model():
    """Загрузка модели"""
    print("=" * 60)
    print("👁️ ЗАГРУЗКА МОДЕЛИ")
    print("=" * 60)
    
    Config.USE_ADAPTIVE_ENCODING = True
    Config.USE_CHANNEL_ATTENTION = True
    Config.LAMBDA_2 = 0.1
    Config.MASK_PROB = 0.15
    
    model = ModifiedPatchTST(Config)
    model = model.to(Config.DEVICE)
    
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        model_state = model.state_dict()
        filtered_state = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key in model_state and value.shape == model_state[key].shape:
                filtered_state[key] = value
        model_state.update(filtered_state)
        model.load_state_dict(model_state, strict=False)
        print("✅ Модель загружена")
    else:
        print("❌ Модель не найдена!")
        return None
    
    model.eval()
    return model

def get_attention_for_class(model, class_id, generator, num_samples=3):
    """Получение усреднённой матрицы внимания для класса"""
    attn_matrices = []
    
    # Генерируем несколько образцов для класса
    for _ in range(num_samples):
        data, labels = generator.generate_dataset(samples_per_class=5)
        idx = np.where(labels == class_id)[0]
        if len(idx) == 0:
            continue
        sample = torch.FloatTensor(data[idx[0]:idx[0]+1]).to(Config.DEVICE)
        
        try:
            logits, attn_weights = model.get_attention(sample)
            if attn_weights is None:
                continue
            
            if torch.isnan(attn_weights).any():
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            
            # Усреднение по головам
            if len(attn_weights.shape) == 4:
                avg_attn = attn_weights.mean(dim=1).cpu().numpy()[0]
            else:
                avg_attn = attn_weights[0].cpu().numpy()
            
            attn_matrices.append(avg_attn)
        except Exception as e:
            continue
    
    if not attn_matrices:
        return None
    
    # Усреднение по нескольким образцам
    return np.mean(attn_matrices, axis=0)

def visualize_all_classes():
    """Визуализация внимания для всех классов"""
    
    print("=" * 60)
    print("👁️ ВИЗУАЛИЗАЦИЯ ВНИМАНИЯ ДЛЯ ВСЕХ КЛАССОВ")
    print("=" * 60)
    
    # Загрузка модели
    model = load_model()
    if model is None:
        return
    
    # Генератор данных
    generator = SyntheticDataGenerator(Config)
    
    # Словарь для хранения карт внимания
    attention_maps = {}
    
    print("\n📊 Генерация карт внимания для каждого класса...")
    for class_id in range(5):
        print(f"   Класс {class_id}: {CLASS_NAMES[class_id]}...")
        attn_map = get_attention_for_class(model, class_id, generator, num_samples=3)
        if attn_map is not None:
            attention_maps[class_id] = attn_map
            print(f"      ✅ Получена карта размером {attn_map.shape}")
        else:
            print(f"      ❌ Не удалось получить карту")
    
    if not attention_maps:
        print("❌ Не удалось получить ни одной карты внимания")
        return
    
    # ===== ПОСТРОЕНИЕ ГРАФИКОВ =====
    print("\n📈 Построение тепловых карт...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Для каждого класса
    for i, class_id in enumerate(range(5)):
        ax = axes[i]
        
        if class_id in attention_maps:
            attn_map = attention_maps[class_id]
            
            # Нормализация для улучшения визуализации
            if attn_map.max() - attn_map.min() > 0.01:
                # Если есть контраст, показываем как есть
                im = ax.imshow(attn_map, cmap='hot', aspect='auto', 
                              vmin=attn_map.min(), vmax=attn_map.max())
            else:
                # Если все значения почти одинаковые, растягиваем контраст
                im = ax.imshow(attn_map, cmap='hot', aspect='auto')
            
            ax.set_title(f'Класс {class_id}: {CLASS_NAMES[class_id]}', 
                        fontsize=12, color=CLASS_COLORS.get(class_id, 'black'))
            ax.set_xlabel('Патчи (ключи)', fontsize=10)
            ax.set_ylabel('Патчи (запросы)', fontsize=10)
            
            # Добавляем цветовую шкалу
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Добавляем информацию о распределении
            text = f'min={attn_map.min():.4f}\nmax={attn_map.max():.4f}\nstd={attn_map.std():.4f}'
            ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax.text(0.5, 0.5, f'Класс {class_id}\n{CLASS_NAMES[class_id]}\n(нет данных)', 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Убираем пустой подграфик (6-й)
    axes[5].set_visible(False)
    
    plt.suptitle('Тепловые карты внимания для всех классов неопределённостей', fontsize=16)
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/attention_all_classes.png', dpi=300, bbox_inches='tight')
    plt.savefig('images/attention_all_classes.pdf', format='pdf', bbox_inches='tight')
    print("\n✅ Тепловые карты сохранены:")
    print("   - images/attention_all_classes.png")
    print("   - images/attention_all_classes.pdf")
    
    plt.show()
    
    # ===== ВЫВОД СТАТИСТИКИ =====
    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА ВЕСОВ ВНИМАНИЯ ПО КЛАССАМ")
    print("=" * 60)
    print(f"{'Класс':<15} {'min':<10} {'max':<10} {'std':<10} {'Равномерность'}")
    print("-" * 60)
    
    for class_id in range(5):
        if class_id in attention_maps:
            attn = attention_maps[class_id]
            uniform_score = attn.std() / attn.mean() if attn.mean() > 0 else 0
            
            # Чем меньше uniform_score, тем более равномерное распределение
            uniformity = "✅ Высокая" if uniform_score < 0.01 else "🟡 Средняя" if uniform_score < 0.05 else "🔴 Низкая"
            
            print(f"{CLASS_NAMES[class_id]:<15} {attn.min():.4f}    {attn.max():.4f}    {attn.std():.4f}    {uniformity}")
    
    print("=" * 60)
    
    return attention_maps

def visualize_single_sample():
    """Визуализация одного образца (быстрый режим)"""
    print("=" * 60)
    print("👁️ ВИЗУАЛИЗАЦИЯ ОДНОГО ОБРАЗЦА")
    print("=" * 60)
    
    model = load_model()
    if model is None:
        return
    
    generator = SyntheticDataGenerator(Config)
    data, labels = generator.generate_dataset(samples_per_class=1)
    
    # Берём первый попавшийся образец
    sample = torch.FloatTensor(data[0:1]).to(Config.DEVICE)
    sample_class = labels[0]
    
    print(f"📊 Класс: {sample_class} ({CLASS_NAMES[sample_class]})")
    
    try:
        logits, attn_weights = model.get_attention(sample)
        
        if attn_weights is None:
            print("❌ Не удалось получить веса внимания")
            return
        
        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        if len(attn_weights.shape) == 4:
            avg_attn = attn_weights.mean(dim=1).cpu().numpy()[0]
        else:
            avg_attn = attn_weights[0].cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(avg_attn, cmap='hot', aspect='auto')
        plt.colorbar(label='Вес внимания')
        plt.title(f'Класс {sample_class}: {CLASS_NAMES[sample_class]}')
        plt.xlabel('Патчи (ключи)')
        plt.ylabel('Патчи (запросы)')
        
        os.makedirs('images', exist_ok=True)
        plt.savefig(f'images/attention_class_{sample_class}.png', dpi=150)
        print(f"✅ Сохранено: images/attention_class_{sample_class}.png")
        plt.show()
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("👁️ ВИЗУАЛИЗАЦИЯ ВНИМАНИЯ")
    print("=" * 60)
    print("\nВыберите режим:")
    print("  1 - Все классы (5 тепловых карт)")
    print("  2 - Один образец (быстрый тест)")
    print("=" * 60)
    
    choice = input("Введите номер (1 или 2): ").strip()
    
    if choice == '1':
        visualize_all_classes()
    elif choice == '2':
        visualize_single_sample()
    else:
        print("❌ Неверный выбор. Запускаю режим 1 (все классы)...")
        visualize_all_classes()