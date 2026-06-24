import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from config import Config
from model import ModifiedPatchTST
from train import Trainer
import os

def load_real_data():
    """Загрузка подготовленных реальных данных"""
    X = np.load('data/real/processed/real_dataset_X.npy')
    y = np.load('data/real/processed/real_dataset_y.npy')
    print(f"📊 Загружено данных: {X.shape}, меток: {y.shape}")
    return X, y

def create_dataloaders(X, y, batch_size=32, test_size=0.2):
    """Создание DataLoader для реальных данных с честным разделением"""
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"📊 Распределение меток в обучающей выборке:")
    for cls in range(5):
        count = np.sum(y_train == cls)
        if count > 0:
            print(f"   Класс {cls}: {count} образцов ({count/len(y_train)*100:.2f}%)")
    
    print(f"\n📊 Распределение меток в валидационной выборке:")
    for cls in range(5):
        count = np.sum(y_val == cls)
        if count > 0:
            print(f"   Класс {cls}: {count} образцов ({count/len(y_val)*100:.2f}%)")
    
    # Создание датасетов
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def main():
    print("=" * 60)
    print("🚀 ДООБУЧЕНИЕ МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    
    # Загрузка данных
    X, y = load_real_data()
    
    # Создание DataLoader
    train_loader, val_loader = create_dataloaders(X, y, batch_size=Config.BATCH_SIZE)
    
    # Создание модели
    model = ModifiedPatchTST(Config)
    model = model.to(Config.DEVICE)
    
    # Загрузка предобученных весов
    if os.path.exists(Config.MODEL_SAVE_PATH):
        # Загружаем только совместимые веса
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        model_state = model.state_dict()
        filtered_state = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key in model_state and value.shape == model_state[key].shape:
                filtered_state[key] = value
        model_state.update(filtered_state)
        model.load_state_dict(model_state)
        print(f"✅ Загружено {len(filtered_state)} слоёв из {len(checkpoint['model_state_dict'])}")
    else:
        print("⚠️ Предобученные веса не найдены. Обучение с нуля.")
    
    # Обучение
    trainer = Trainer(model, Config)
    trainer.optimizer.param_groups[0]['lr'] = 0.0001
    print(f"📉 Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
    
    best_acc = trainer.train(train_loader, val_loader, num_epochs=50)
    
    print(f"\n🎯 Лучшая точность на реальных данных: {best_acc:.2f}%")

if __name__ == '__main__':
    main()