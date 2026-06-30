import torch
import numpy as np
from config import Config
from model import ModifiedPatchTST
import os

def debug_attention():
    """Диагностика: проверка, что возвращает MultiheadAttention"""
    
    print("=" * 60)
    print("🔍 ДИАГНОСТИКА ВНИМАНИЯ")
    print("=" * 60)
    
    # Загрузка модели
    model = ModifiedPatchTST(Config)
    model = model.to(Config.DEVICE)
    
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Модель загружена")
    else:
        print("❌ Модель не найдена")
        return
    
    model.eval()
    
    # Тестовый вход
    x = torch.randn(1, Config.WINDOW_LENGTH, Config.NUM_CHANNELS).to(Config.DEVICE)
    
    # Проверка MultiheadAttention напрямую
    print("\n🔍 Проверка MultiheadAttention:")
    for i, block in enumerate(model.transformer_blocks):
        attn = block.attention
        print(f"   Блок {i}: {type(attn).__name__}")
        
        # Прямой вызов attention с need_weights=True
        with torch.no_grad():
            # Получаем вход для attention из текущего блока (эмуляция)
            patches, embeddings, patch_variance = model.patch_encoder(x)
            if model.adaptive_pos_encoding is not None:
                embeddings = model.adaptive_pos_encoding(embeddings, patch_variance)
            if model.channel_attention is not None:
                embeddings = model.channel_attention(embeddings)
            
            # Пропускаем через предыдущие блоки
            for j in range(i):
                embeddings = model.transformer_blocks[j](embeddings, return_attention=False)
            
            # Прямой вызов attention
            attn_out, attn_weights = attn(embeddings, embeddings, embeddings, need_weights=True)
            print(f"      attn_out shape: {attn_out.shape}")
            if attn_weights is not None:
                print(f"      attn_weights shape: {attn_weights.shape}")
            else:
                print(f"      attn_weights is None")
    
    print("\n✅ Диагностика завершена")

if __name__ == '__main__':
    debug_attention()