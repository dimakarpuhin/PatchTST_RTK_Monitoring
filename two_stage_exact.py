# two_stage_exact.py
# Двухэтапное обучение с синтетикой, точно имитирующей PM

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import os
from config import Config
from model import create_model
from train import Trainer
from synthetic_pm_exact import SyntheticPMExactGenerator
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

def load_pm_processed():
    X_train = np.load(f'{Config.PM_PROCESSED_PATH}/X_train.npy')
    y_train = np.load(f'{Config.PM_PROCESSED_PATH}/y_train.npy')
    X_val = np.load(f'{Config.PM_PROCESSED_PATH}/X_val.npy')
    y_val = np.load(f'{Config.PM_PROCESSED_PATH}/y_val.npy')
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def generate_synthetic_exact(config, samples_per_class=500):
    generator = SyntheticPMExactGenerator(config)
    X, y = generator.generate_dataset(samples_per_class=samples_per_class)
    return X, y

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def adapt_classifier(model, new_num_classes):
    old_classifier = model.classifier
    new_classifier = nn.Linear(model.config.D_MODEL, new_num_classes)
    nn.init.xavier_uniform_(new_classifier.weight)
    nn.init.zeros_(new_classifier.bias)
    model.classifier = new_classifier
    model.config.NUM_CLASSES = new_num_classes
    model.classifier = model.classifier.to(next(model.parameters()).device)
    return model

def main():
    print("=" * 70)
    print("🔬 ДВУХЭТАПНОЕ ОБУЧЕНИЕ (СИНТЕТИКА С ТОЧНОЙ ФИЗИКОЙ PM → PM)")
    print("=" * 70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_pm_processed()
    
    Config.WINDOW_LENGTH = X_train.shape[1]
    Config.NUM_CHANNELS = X_train.shape[2]
    Config.NUM_CLASSES = 2
    Config.D_MODEL = 30
    Config.NUM_LAYERS = 2
    Config.NUM_HEADS = 2
    Config.DROPOUT = 0.3
    Config.LAMBDA_1 = 1e-3
    Config.MASK_PROB = 0.15
    Config.BATCH_SIZE = 32
    Config.NUM_EPOCHS = 30
    Config.EARLY_STOPPING_PATIENCE = 10
    
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, Config.BATCH_SIZE
    )
    
    results = []
    
    # ===== 1. Обучение с нуля =====
    print("\n📚 ЭКСПЕРИМЕНТ 1: Обучение с нуля")
    model = create_model(Config)
    trainer = Trainer(model, Config)
    start_time = time.time()
    best_val_acc = trainer.train(train_loader, val_loader, Config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    test_loss, test_acc = trainer.validate(test_loader)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            logits, _ = model(data, use_masking=False)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average='binary')
    
    results.append({
        'experiment': 'С нуля',
        'val_acc': round(best_val_acc, 2),
        'test_acc': round(test_acc, 2),
        'test_f1': round(f1, 4),
        'time_sec': round(elapsed, 1)
    })
    
    # ===== 2. Синтетика с точной физикой → PM =====
    print("\n📚 ЭКСПЕРИМЕНТ 2: Синтетика (точная физика) → PM")
    
    X_synth, y_synth = generate_synthetic_exact(Config, samples_per_class=500)
    synth_loader, _, _ = create_dataloaders(
        X_synth, y_synth, X_synth[:10], y_synth[:10], X_synth[:10], y_synth[:10], Config.BATCH_SIZE
    )
    
    model = create_model(Config)
    print("\n📚 Предобучение на синтетике с точной физикой...")
    trainer = Trainer(model, Config)
    trainer.train(synth_loader, val_loader, Config.NUM_EPOCHS)
    print("✅ Предобучение завершено")
    
    print("\n🔄 Адаптация классификатора...")
    model = adapt_classifier(model, 2)
    
    print("\n📚 Дообучение на PM данных...")
    trainer = Trainer(model, Config)
    trainer.optimizer.param_groups[0]['lr'] = 5e-4
    start_time = time.time()
    best_val_acc = trainer.train(train_loader, val_loader, Config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    test_loss, test_acc = trainer.validate(test_loader)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            logits, _ = model(data, use_masking=False)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average='binary')
    
    results.append({
        'experiment': 'Синтетика (точная физика) → PM',
        'val_acc': round(best_val_acc, 2),
        'test_acc': round(test_acc, 2),
        'test_f1': round(f1, 4),
        'time_sec': round(elapsed, 1)
    })
    
    # ===== 3. 50% PM → 50% PM =====
    print("\n📚 ЭКСПЕРИМЕНТ 3: 50% PM → 50% PM")
    
    half = len(X_train) // 2
    X_pretrain = X_train[:half]
    y_pretrain = y_train[:half]
    X_finetune = X_train[half:]
    y_finetune = y_train[half:]
    
    pretrain_loader, _, _ = create_dataloaders(
        X_pretrain, y_pretrain, X_val, y_val, X_test, y_test, Config.BATCH_SIZE
    )
    finetune_loader, _, _ = create_dataloaders(
        X_finetune, y_finetune, X_val, y_val, X_test, y_test, Config.BATCH_SIZE
    )
    
    model = create_model(Config)
    print("\n📚 Предобучение на 50% PM...")
    trainer = Trainer(model, Config)
    trainer.train(pretrain_loader, val_loader, Config.NUM_EPOCHS)
    print("✅ Предобучение завершено")
    
    print("\n📚 Дообучение на 50% PM...")
    trainer = Trainer(model, Config)
    trainer.optimizer.param_groups[0]['lr'] = 5e-4
    start_time = time.time()
    best_val_acc = trainer.train(finetune_loader, val_loader, Config.NUM_EPOCHS)
    elapsed = time.time() - start_time
    
    test_loss, test_acc = trainer.validate(test_loader)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            logits, _ = model(data, use_masking=False)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average='binary')
    
    results.append({
        'experiment': '50% PM → 50% PM',
        'val_acc': round(best_val_acc, 2),
        'test_acc': round(test_acc, 2),
        'test_f1': round(f1, 4),
        'time_sec': round(elapsed, 1)
    })
    
    # ===== ВЫВОД =====
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("📊 СВОДНАЯ ТАБЛИЦА (СИНТЕТИКА С ТОЧНОЙ ФИЗИКОЙ)")
    print("=" * 70)
    print(df.to_string(index=False))
    
    os.makedirs('logs', exist_ok=True)
    df.to_csv('logs/two_stage_exact_results.csv', index=False)
    print("\n💾 Результаты сохранены в logs/two_stage_exact_results.csv")
    
    print("\n📋 ТАБЛИЦА ДЛЯ ДИССЕРТАЦИИ")
    print("=" * 70)
    print("\n| Эксперимент | Val Acc (%) | Test Acc (%) | F1-score |")
    print("|-------------|-------------|--------------|----------|")
    for _, row in df.iterrows():
        print(f"| {row['experiment']} | {row['val_acc']:.2f} | {row['test_acc']:.2f} | {row['test_f1']:.4f} |")

if __name__ == '__main__':
    main()