import numpy as np
import torch
import time
from config import Config
from model import ModifiedPatchTST
from train import Trainer
from data_loader import create_dataloaders_from_arrays

def run_experiment(name, config_mods):
    """Запуск одного эксперимента на реальных данных"""
    
    print("=" * 60)
    print(f"🧪 ЭКСПЕРИМЕНТ: {name}")
    print("=" * 60)
    
    # Применяем изменения к конфигурации
    for key, value in config_mods.items():
        setattr(Config, key, value)
        print(f"   {key} = {value}")
    
    # Загрузка реальных данных
    X = np.load('data/real/processed/real_dataset_X.npy')
    y = np.load('data/real/processed/real_dataset_y.npy')
    print(f"📊 Загружено: X.shape={X.shape}, y.shape={y.shape}")
    
    # Создание DataLoader
    train_loader, val_loader, _ = create_dataloaders_from_arrays(X, y, Config)
    
    # Создание модели
    model = ModifiedPatchTST(Config)
    model = model.to(Config.DEVICE)
    
    # Обучение
    trainer = Trainer(model, Config)
    trainer.optimizer.param_groups[0]['lr'] = 0.0001  # Маленький LR для дообучения
    
    start_time = time.time()
    best_acc = trainer.train(train_loader, val_loader, Config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    # Сохранение результатов
    result = {
        'experiment': name,
        'best_acc': best_acc,
        'time_sec': round(elapsed, 1)
    }
    
    print(f"✅ Результат: {best_acc:.2f}% за {elapsed:.1f} сек")
    return result

if __name__ == '__main__':
    Config.set_seed()
    Config.WINDOW_LENGTH = 512
    Config.NUM_EPOCHS = 50
    Config.EARLY_STOPPING_PATIENCE = 10
    
    results = []
    
    # 1. Базовый PatchTST (все модификации выключены)
    results.append(run_experiment(
        'Базовый PatchTST (без модификаций)',
        {
            'USE_ADAPTIVE_ENCODING': False,
            'USE_CHANNEL_ATTENTION': False,
            'LAMBDA_2': 0,
            'MASK_PROB': 0
        }
    ))
    
    # 2. Модифицированный PatchTST (все включено)
    results.append(run_experiment(
        'Модифицированный PatchTST (все включено)',
        {
            'USE_ADAPTIVE_ENCODING': True,
            'USE_CHANNEL_ATTENTION': True,
            'LAMBDA_2': 0.1,
            'MASK_PROB': 0.15
        }
    ))
    
    # Вывод итогов
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    for r in results:
        print(f"{r['experiment']:<40} {r['best_acc']:>6.2f}% ({r['time_sec']} сек)")