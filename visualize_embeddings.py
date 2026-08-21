# visualize_embeddings.py
# t-SNE / UMAP визуализация эмбеддингов

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from config import Config
from model import create_model
from torch.utils.data import DataLoader, TensorDataset
import os

plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False

def load_pm_processed():
    X_train = np.load(f'{Config.PM_PROCESSED_PATH}/X_train.npy')
    y_train = np.load(f'{Config.PM_PROCESSED_PATH}/y_train.npy')
    X_val = np.load(f'{Config.PM_PROCESSED_PATH}/X_val.npy')
    y_val = np.load(f'{Config.PM_PROCESSED_PATH}/y_val.npy')
    X_test = np.load(f'{Config.PM_PROCESSED_PATH}/X_test.npy')
    y_test = np.load(f'{Config.PM_PROCESSED_PATH}/y_test.npy')
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_model(model_type, config):
    if model_type == 'modified':
        config.USE_ADAPTIVE_ENCODING = True
        config.USE_CHANNEL_ATTENTION = True
        config.LAMBDA_2 = 0.0
        config.MASK_PROB = 0.15
        model = create_model(config)
    elif model_type == 'baseline':
        config.USE_ADAPTIVE_ENCODING = False
        config.USE_CHANNEL_ATTENTION = False
        config.LAMBDA_2 = 0.0
        config.MASK_PROB = 0.0
        model = create_model(config)
    else:
        raise ValueError(f"Неизвестная модель: {model_type}")
    return model.to(config.DEVICE)

def get_embeddings(model, data_loader, config):
    """Получение эмбеддингов (усреднение по патчам)"""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(config.DEVICE)
            logits, emb = model(data, use_masking=False)
            
            # Усредняем по патчам → [batch, d_model]
            emb_pooled = emb.mean(dim=1)
            
            embeddings.append(emb_pooled.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    
    return embeddings, labels

def plot_tsne(embeddings, labels, title, save_path):
    print(f"   Выполняется t-SNE для {title}...")
    
    # Убеждаемся, что эмбеддинги 2D
    if len(embeddings.shape) > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_tsne = tsne.fit_transform(embeddings_pca)
    
    plt.figure(figsize=(10, 8))
    colors = ['#2ecc71', '#e74c3c']
    class_names = ['Норма', 'Отказ']
    
    for i in range(2):
        idx = labels == i
        plt.scatter(embeddings_tsne[idx, 0], embeddings_tsne[idx, 1],
                   c=colors[i], label=class_names[i], alpha=0.7, s=50)
    
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   ✅ {save_path}")

def main():
    print("=" * 60)
    print("📊 T-SNE ВИЗУАЛИЗАЦИЯ ЭМБЕДДИНГОВ")
    print("=" * 60)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_pm_processed()
    
    Config.WINDOW_LENGTH = X_test.shape[1]
    Config.NUM_CHANNELS = X_test.shape[2]
    Config.NUM_CLASSES = 2
    Config.D_MODEL = 30
    Config.NUM_LAYERS = 2
    Config.NUM_HEADS = 2
    Config.DROPOUT = 0.3
    Config.BATCH_SIZE = 32
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    os.makedirs('images', exist_ok=True)
    
    print("\n📊 Модели:")
    
    # 1. Модифицированный
    print("\n   Загрузка модифицированного PatchTST...")
    model_modified = get_model('modified', Config)
    checkpoint = torch.load('models/patchtst_model.pth', map_location=Config.DEVICE)
    model_modified.load_state_dict(checkpoint['model_state_dict'])
    model_modified = model_modified.to(Config.DEVICE)
    
    emb_modified, labels = get_embeddings(model_modified, test_loader, Config)
    print(f"   Эмбеддинги: {emb_modified.shape}")
    
    plot_tsne(emb_modified, labels, 
             'Модифицированный PatchTST: эмбеддинги классов',
             'images/tsne_modified.png')
    
    # 2. Базовый (создаём без модификаций)
    print("\n   Загрузка базового PatchTST...")
    model_baseline = get_model('baseline', Config)
    # Используем те же веса, но с отключёнными модификациями
    model_baseline.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model_baseline = model_baseline.to(Config.DEVICE)
    
    emb_baseline, _ = get_embeddings(model_baseline, test_loader, Config)
    print(f"   Эмбеддинги: {emb_baseline.shape}")
    
    plot_tsne(emb_baseline, labels,
             'Базовый PatchTST: эмбеддинги классов',
             'images/tsne_baseline.png')
    
    # 3. Сравнение
    print("\n   Создание общего графика...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, (emb, title, ax) in enumerate(zip(
        [emb_modified, emb_baseline],
        ['Модифицированный PatchTST', 'Базовый PatchTST'],
        axes
    )):
        if len(emb.shape) > 2:
            emb = emb.reshape(emb.shape[0], -1)
        
        pca = PCA(n_components=min(50, emb.shape[1]))
        emb_pca = pca.fit_transform(emb)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_tsne = tsne.fit_transform(emb_pca)
        
        colors = ['#2ecc71', '#e74c3c']
        class_names = ['Норма', 'Отказ']
        
        for j in range(2):
            idx = labels == j
            ax.scatter(emb_tsne[idx, 0], emb_tsne[idx, 1],
                      c=colors[j], label=class_names[j], alpha=0.7, s=50)
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/tsne_comparison.png', dpi=300)
    plt.close()
    print("   ✅ images/tsne_comparison.png")
    
    print("\n" + "=" * 60)
    print("✅ ВСЕ ГРАФИКИ СОХРАНЕНЫ В ПАПКЕ images/")
    print("   - tsne_modified.png")
    print("   - tsne_baseline.png")
    print("   - tsne_comparison.png")
    print("=" * 60)

if __name__ == '__main__':
    main()