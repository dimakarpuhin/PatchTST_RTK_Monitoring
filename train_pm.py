# train_pm.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from model import create_model
from train import Trainer
import os

def load_pm_processed():
    X_train = np.load(f'{Config.PM_PROCESSED_PATH}/X_train.npy')
    y_train = np.load(f'{Config.PM_PROCESSED_PATH}/y_train.npy')
    X_val = np.load(f'{Config.PM_PROCESSED_PATH}/X_val.npy')
    y_val = np.load(f'{Config.PM_PROCESSED_PATH}/y_val.npy')
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    print("=" * 60)
    print("🚀 ОБУЧЕНИЕ НА PREDICTIVE MAINTENANCE (С РЕГУЛЯРИЗАЦИЕЙ)")
    print("=" * 60)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_pm_processed()
    
    print(f"\n📊 Данные:")
    print(f"  Train: {X_train.shape}, метки: {y_train.shape}")
    print(f"  Val:   {X_val.shape}, метки: {y_val.shape}")
    print(f"  Test:  {X_test.shape}, метки: {y_test.shape}")
    
    print(f"\n📊 Распределение классов в Train:")
    for cls in range(2):
        count = np.sum(y_train == cls)
        print(f"  Класс {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # ===== НАСТРОЙКА КОНФИГА (усиленная регуляризация) =====
    Config.WINDOW_LENGTH = X_train.shape[1]
    Config.NUM_CHANNELS = X_train.shape[2]
    Config.NUM_CLASSES = 2
    Config.D_MODEL = 30      # уменьшаем модель
    Config.NUM_LAYERS = 2    # меньше слоёв
    Config.NUM_HEADS = 2     # меньше голов
    Config.DROPOUT = 0.3     # больше dropout
    Config.LAMBDA_1 = 1e-3   # сильнее L2
    Config.MASK_PROB = 0.2   # больше маскирования
    Config.BATCH_SIZE = 32
    Config.NUM_EPOCHS = 50
    Config.EARLY_STOPPING_PATIENCE = 20  # больше терпения
    
    print(f"\n📊 Параметры модели:")
    print(f"  WINDOW_LENGTH: {Config.WINDOW_LENGTH}")
    print(f"  NUM_CHANNELS: {Config.NUM_CHANNELS}")
    print(f"  NUM_CLASSES: {Config.NUM_CLASSES}")
    print(f"  D_MODEL: {Config.D_MODEL}")
    print(f"  DROPOUT: {Config.DROPOUT}")
    print(f"  LAMBDA_1: {Config.LAMBDA_1}")
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = create_model(Config)
    print(f"\n✅ Модель создана")
    
    trainer = Trainer(model, Config)
    best_acc = trainer.train(train_loader, val_loader, Config.NUM_EPOCHS)
    
    print("\n" + "=" * 60)
    print("📊 ТЕСТИРОВАНИЕ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)
    
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.2f}%")
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            logits, _ = model(data, use_masking=False)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    f1 = f1_score(all_targets, all_preds, average='binary')
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    
    print(f"\n📊 Дополнительные метрики:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)

if __name__ == '__main__':
    main()