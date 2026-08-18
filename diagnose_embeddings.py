# diagnose_embeddings.py
# Полная диагностика эмбеддингов для модифицированной и базовой модели

import numpy as np
import torch
from config import Config
from model import create_model
from torch.utils.data import DataLoader, TensorDataset
import os

def load_pm_processed():
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_test, y_test

def diagnose_model(model_type, checkpoint_path, X_test, y_test, config):
    """Диагностика модели: проверка загрузки весов и эмбеддингов"""
    
    print(f"\n{'='*60}")
    print(f"🔍 ДИАГНОСТИКА: {model_type}")
    print('='*60)
    
    # Создание модели
    if model_type == 'modified':
        config.USE_ADAPTIVE_ENCODING = True
        config.USE_CHANNEL_ATTENTION = True
        config.LAMBDA_2 = 0.0
        config.MASK_PROB = 0.15
    else:
        config.USE_ADAPTIVE_ENCODING = False
        config.USE_CHANNEL_ATTENTION = False
        config.LAMBDA_2 = 0.0
        config.MASK_PROB = 0.0
    
    model = create_model(config)
    print(f"  Модель создана. Параметров: {sum(p.numel() for p in model.parameters())}")
    
    # Загрузка весов
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        state_dict = checkpoint['model_state_dict']
        
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        common_keys = model_keys & ckpt_keys
        
        print(f"  Ключей в модели: {len(model_keys)}")
        print(f"  Ключей в чекпоинте: {len(ckpt_keys)}")
        print(f"  Общих ключей: {len(common_keys)}")
        
        if len(common_keys) < len(model_keys):
            missing = model_keys - ckpt_keys
            print(f"  ⚠️ Отсутствуют ключи: {list(missing)[:5]}...")
        
        model.load_state_dict(state_dict, strict=False)
        print(f"  ✅ Веса загружены (strict=False)")
    else:
        print(f"  ❌ Файл {checkpoint_path} не найден")
        return None, None
    
    model = model.to(config.DEVICE)
    model.eval()
    
    # Прогон данных
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    embeddings = []
    labels = []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(config.DEVICE)
            logits, emb = model(data, use_masking=False)
            
            # Усреднение по патчам
            emb_pooled = emb.mean(dim=1)
            
            embeddings.append(emb_pooled.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    
    # Статистика эмбеддингов
    print(f"\n  📊 Статистика эмбеддингов:")
    print(f"    Форма: {embeddings.shape}")
    print(f"    Среднее: {embeddings.mean():.4f}")
    print(f"    Стандартное отклонение: {embeddings.std():.4f}")
    print(f"    min: {embeddings.min():.4f}")
    print(f"    max: {embeddings.max():.4f}")
    
    # Расстояние между классами
    emb_class0 = embeddings[labels == 0]
    emb_class1 = embeddings[labels == 1]
    
    if len(emb_class0) > 0 and len(emb_class1) > 0:
        centroid0 = emb_class0.mean(axis=0)
        centroid1 = emb_class1.mean(axis=0)
        dist = np.linalg.norm(centroid0 - centroid1)
        print(f"\n  📏 Расстояние между центроидами классов: {dist:.4f}")
        
        var0 = emb_class0.var(axis=0).mean()
        var1 = emb_class1.var(axis=0).mean()
        print(f"  📊 Внутриклассовая дисперсия:")
        print(f"    Класс 0: {var0:.4f}")
        print(f"    Класс 1: {var1:.4f}")
    
    return embeddings, labels

def main():
    print("=" * 60)
    print("🔬 ПОЛНАЯ ДИАГНОСТИКА ЭМБЕДДИНГОВ")
    print("=" * 60)
    
    # Загрузка данных
    X_test, y_test = load_pm_processed()
    print(f"\n📊 Тестовая выборка: {X_test.shape}, метки: {y_test.shape}")
    print(f"   Класс 0 (Норма): {np.sum(y_test == 0)} образцов")
    print(f"   Класс 1 (Отказ): {np.sum(y_test == 1)} образцов")
    
    # Настройка конфига
    Config.WINDOW_LENGTH = X_test.shape[1]
    Config.NUM_CHANNELS = X_test.shape[2]
    Config.NUM_CLASSES = 2
    Config.D_MODEL = 30
    Config.NUM_LAYERS = 2
    Config.NUM_HEADS = 2
    Config.DROPOUT = 0.3
    Config.BATCH_SIZE = 32
    
    # ============================================================
    # ДИАГНОСТИКА 1: МОДИФИЦИРОВАННАЯ МОДЕЛЬ
    # ============================================================
    emb_modified, labels_modified = diagnose_model(
        'modified',
        'models/patchtst_model.pth',
        X_test, y_test, Config
    )
    
    # ============================================================
    # ДИАГНОСТИКА 2: БАЗОВАЯ МОДЕЛЬ
    # ============================================================
    emb_baseline, labels_baseline = diagnose_model(
        'baseline',
        'models/patchtst_model.pth',  # используем те же веса
        X_test, y_test, Config
    )
    
    # ============================================================
    # ИТОГОВОЕ СРАВНЕНИЕ
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Общая статистика
    print(f"\n{'Показатель':<35} {'Модифицированный':<20} {'Базовый':<20}")
    print("-" * 75)
    print(f"{'Среднее эмбеддингов':<35} {emb_modified.mean():<20.4f} {emb_baseline.mean():<20.4f}")
    print(f"{'Стандартное отклонение':<35} {emb_modified.std():<20.4f} {emb_baseline.std():<20.4f}")
    
    # Расстояние между классами
    emb0_mod = emb_modified[labels_modified == 0]
    emb1_mod = emb_modified[labels_modified == 1]
    centroid0_mod = emb0_mod.mean(axis=0)
    centroid1_mod = emb1_mod.mean(axis=0)
    dist_mod = np.linalg.norm(centroid0_mod - centroid1_mod)
    
    emb0_base = emb_baseline[labels_baseline == 0]
    emb1_base = emb_baseline[labels_baseline == 1]
    centroid0_base = emb0_base.mean(axis=0)
    centroid1_base = emb1_base.mean(axis=0)
    dist_base = np.linalg.norm(centroid0_base - centroid1_base)
    
    print(f"{'Расстояние между классами':<35} {dist_mod:<20.4f} {dist_base:<20.4f}")
    
    # Внутриклассовая дисперсия
    var0_mod = emb0_mod.var(axis=0).mean()
    var1_mod = emb1_mod.var(axis=0).mean()
    var0_base = emb0_base.var(axis=0).mean()
    var1_base = emb1_base.var(axis=0).mean()
    
    print(f"{'Внутриклассовая дисперсия (класс 0)':<35} {var0_mod:<20.4f} {var0_base:<20.4f}")
    print(f"{'Внутриклассовая дисперсия (класс 1)':<35} {var1_mod:<20.4f} {var1_base:<20.4f}")
    
    # Соотношение расстояния к дисперсии (мера разделимости)
    ratio_mod = dist_mod / (var0_mod + var1_mod)
    ratio_base = dist_base / (var0_base + var1_base)
    
    print(f"\n{'Мера разделимости (dist / (var0+var1))':<35} {ratio_mod:<20.4f} {ratio_base:<20.4f}")
    
    print("\n" + "=" * 60)
    print("📝 ВЫВОД")
    print("=" * 60)
    
    if dist_mod > dist_base:
        print(f"✅ Модифицированный PatchTST имеет БОЛЬШЕЕ расстояние между классами ({dist_mod:.4f} vs {dist_base:.4f})")
    else:
        print(f"⚠️ Базовый PatchTST имеет БОЛЬШЕЕ расстояние между классами ({dist_base:.4f} vs {dist_mod:.4f})")
    
    if var0_mod < var0_base and var1_mod < var1_base:
        print(f"✅ Модифицированный PatchTST имеет МЕНЬШУЮ внутриклассовую дисперсию")
    else:
        print(f"⚠️ Внутриклассовая дисперсия сравнима или выше у модифицированной модели")
    
    if ratio_mod > ratio_base:
        print(f"✅ Модифицированный PatchTST имеет ЛУЧШУЮ меру разделимости ({ratio_mod:.4f} vs {ratio_base:.4f})")
    else:
        print(f"⚠️ Базовый PatchTST имеет ЛУЧШУЮ меру разделимости ({ratio_base:.4f} vs {ratio_mod:.4f})")

if __name__ == '__main__':
    main()