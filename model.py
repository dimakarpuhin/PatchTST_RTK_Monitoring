# model.py
# Реализация модифицированной архитектуры PatchTST
# Глава 2: формулы (2.2), (2.3), (2.4), (2.5), (2.6), (2.7)

import torch
import torch.nn as nn
import math
from config import Config
from typing import Tuple, Optional


class AdaptivePositionalEncoding(nn.Module):
    """
    Адаптивное позиционное кодирование с учётом локальной дисперсии
    Формула (2.6): E_pos^{(l)}(k) = PE(k) * α^{(l)} * σ(P_k)
    
    В данной реализации α является скалярным коэффициентом (обучаемым).
    """
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        # Базовое позиционное кодирование (как в стандартном Transformer)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]
        
        # Обучаемый скалярный коэффициент α (формула 2.6)
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, patch_variance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_patches, d_model] - эмбеддинги патчей
            patch_variance: [batch, num_patches] - локальная дисперсия каждого патча σ(P_k)
        Returns:
            [batch, num_patches, d_model] - с добавленным адаптивным кодированием
        """
        batch, num_patches, d_model = x.shape
        # Базовое кодирование для первых num_patches
        pos_encoding = self.pe[:num_patches, :].unsqueeze(0)  # [1, num_patches, d_model]
        # Адаптивная модуляция: α * σ(P_k) (размерность согласуется через broadcasting)
        adaptive_scale = self.alpha * patch_variance.unsqueeze(-1)  # [batch, num_patches, 1]
        return x + adaptive_scale * pos_encoding


class ChannelAttention(nn.Module):
    """
    Межканальный механизм внимания
    Формула (2.7): Z_channel = Z * softmax(Z^T * Z / sqrt(d))
    
    Вход: [batch, num_patches, d_model]
    Выход: [batch, num_patches, d_model]
    
    Внимание вычисляется между каналами (параметрами) через проекцию в пространство каналов.
    """
    def __init__(self, d_model: int, num_channels: int):
        super().__init__()
        self.d_model = d_model
        self.num_channels = num_channels
        self.scale = (d_model // num_channels) ** -0.5  # ИСПРАВЛЕНО: корректный scaling factor
        
        # Проекция в пространство каналов
        self.channel_proj = nn.Linear(d_model, num_channels)
        
        # Выходная проекция обратно в d_model
        self.out_proj = nn.Linear(num_channels, d_model)
        
        # LayerNorm для стабилизации
        self.norm = nn.LayerNorm(d_model)
        
    '''def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, num_patches, d_model]
        Returns:
            [batch, num_patches, d_model]
        """
       batch, num_patches, d_model = z.shape
        residual = z
        
        # Проекция в пространство каналов
        # [batch, num_patches, d_model] -> [batch, num_patches, num_channels]
        channel_features = self.channel_proj(z)
        
        # Вычисление матрицы внимания между каналами
        # channel_features: [batch, num_patches, num_channels]
        # Транспонируем для умножения: [batch, num_channels, num_patches]
        channel_features_t = channel_features.transpose(-2, -1)  # [batch, num_channels, num_patches]
        
        # Матрица внимания: [batch, num_channels, num_channels]
        attention_logits = torch.matmul(channel_features_t, channel_features_t.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_logits, dim=-1)
        
        # Применение внимания
        attended = torch.matmul(attention_weights, channel_features_t)  # [batch, num_channels, num_patches]
        attended = attended.transpose(-2, -1)  # [batch, num_patches, num_channels]
        
        # Обратная проекция
        out = self.out_proj(attended)  # [batch, num_patches, d_model]
        
        # Остаточная связь и нормализация
        out = self.norm(residual + out)
        
        return out'''


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch, num_patches, d_model = z.shape
        residual = z
        
        # Проекция в пространство каналов
        channel_features = self.channel_proj(z)
        
        # ===== ОЧИСТКА =====
        if torch.isnan(channel_features).any():
            print("   ⚠️ nan в channel_features! Заменяем на 0...")
            channel_features = torch.nan_to_num(channel_features, nan=0.0)
        # ===================
        
        channel_features_t = channel_features.transpose(-2, -1)
        
        attention_logits = torch.matmul(channel_features_t, channel_features_t.transpose(-2, -1)) * self.scale
        
        # ===== ОЧИСТКА =====
        if torch.isnan(attention_logits).any():
            print("   ⚠️ nan в attention_logits! Заменяем на 0...")
            attention_logits = torch.nan_to_num(attention_logits, nan=0.0)
        # ===================
        
        attention_weights = torch.softmax(attention_logits, dim=-1)
        
        # ===== ОЧИСТКА =====
        if torch.isnan(attention_weights).any():
            print("   ⚠️ nan в attention_weights! Заменяем на 0...")
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        # ===================
        
        attended = torch.matmul(attention_weights, channel_features_t)
        attended = attended.transpose(-2, -1)
        
        out = self.out_proj(attended)
        out = self.norm(residual + out)
        
        # ===== ОЧИСТКА =====
        if torch.isnan(out).any():
            print("   ⚠️ nan в out! Заменяем на 0...")
            out = torch.nan_to_num(out, nan=0.0)
        # ===================
        
        return out

class StochasticMasking(nn.Module):
    """
    Стохастическое маскирование патчей для регуляризации
    С вероятностью p_mask заменяет патч на обучаемый маскирующий токен
    """
    def __init__(self, d_model: int, p_mask: float = 0.15):
        super().__init__()
        self.p_mask = p_mask
        # ИСПРАВЛЕНО: обучаемый маскирующий токен (вместо 0.0)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, num_patches, d_model] - эмбеддинги патчей
        Returns:
            masked_x: [batch, num_patches, d_model] - замаскированные эмбеддинги
            mask: [batch, num_patches] - boolean маска (True = замаскирован)
        """
        batch, num_patches, d_model = x.shape
        # Генерация маски
        mask = torch.rand(batch, num_patches, 1, device=x.device) < self.p_mask
        # Замена замаскированных патчей на обучаемый токен
        masked_x = torch.where(mask, self.mask_token, x)
        return masked_x, mask.squeeze(-1)


class PatchEncoder(nn.Module):
    """
    Модуль патчинга: разбиение ряда на патчи и линейное проектирование
    Формула (2.2): E_k = P_k * W_p + b_p
    """
    def __init__(self, patch_len: int, stride: int, d_model: int, num_channels: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_channels = num_channels
        
        # Линейное проектирование патча в эмбеддинг
        patch_dim = patch_len * num_channels
        self.projection = nn.Linear(patch_dim, d_model)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, window_len, num_channels]
        Returns:
            patches: [batch, num_patches, patch_len, num_channels]
            embeddings: [batch, num_patches, d_model]
            patch_variance: [batch, num_patches] - локальная дисперсия для формулы (2.6)
        """
        batch, window_len, num_channels = x.shape
        
        # Разбиение на патчи с перекрытием
        patches = []
        for i in range(0, window_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :]  # [batch, patch_len, num_channels]
            patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # [batch, num_patches, patch_len, num_channels]
        num_patches = patches.shape[1]
        
        # Вычисление локальной дисперсии для каждого патча (формула 2.6)
        patch_variance = patches.var(dim=(2, 3))  # [batch, num_patches]
        
        # Линейное проектирование: свёртка патча в плоский вектор
        patches_flat = patches.view(batch, num_patches, -1)  # [batch, num_patches, patch_len * num_channels]
        embeddings = self.projection(patches_flat)  # [batch, num_patches, d_model]
        
        return patches, embeddings, patch_variance


class TransformerBlock(nn.Module):
    """
    Один блок трансформерного кодировщика
    Формулы (2.3) и (2.4)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    #def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Формула (2.3): Z = LayerNorm(Z + MultiHeadAttn(Z))
        #attn_out, _ = self.attention(x, x, x)
        #x = self.norm1(x + attn_out)
        
        # Формула (2.4): Z = LayerNorm(Z + FFN(Z))
        #ffn_out = self.ffn(x)
        #x = self.norm2(x + ffn_out)
        
        #return x
    # Эксперимент 8 модификация внимания 
    '''def forward(self, x, return_attention=False):
    # Self-attention с возможностью вернуть веса
        if return_attention:
            attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
            x = self.norm1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x, attn_weights
        else:
            attn_out, _ = self.attention(x, x, x, need_weights=False)
            x = self.norm1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x'''

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
            
            # ===== ОЧИСТКА =====
            if torch.isnan(attn_out).any():
                print("      ⚠️ nan в attn_out! Заменяем на 0...")
                attn_out = torch.nan_to_num(attn_out, nan=0.0)
            if attn_weights is not None and torch.isnan(attn_weights).any():
                print("      ⚠️ nan в attn_weights! Заменяем на 0...")
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            # ===================
            
            x = self.norm1(x + attn_out)
            
            # ===== ОЧИСТКА =====
            if torch.isnan(x).any():
                print("      ⚠️ nan после norm1! Заменяем на 0...")
                x = torch.nan_to_num(x, nan=0.0)
            # ===================
            
            ffn_out = self.ffn(x)
            
            # ===== ОЧИСТКА =====
            if torch.isnan(ffn_out).any():
                print("      ⚠️ nan в ffn_out! Заменяем на 0...")
                ffn_out = torch.nan_to_num(ffn_out, nan=0.0)
            # ===================
            
            x = self.norm2(x + ffn_out)
            
            # ===== ОЧИСТКА =====
            if torch.isnan(x).any():
                print("      ⚠️ nan после norm2! Заменяем на 0...")
                x = torch.nan_to_num(x, nan=0.0)
            # ===================
            
            return x, attn_weights
        else:
            attn_out, _ = self.attention(x, x, x, need_weights=False)
            
            # ===== ОЧИСТКА =====
            if torch.isnan(attn_out).any():
                print("      ⚠️ nan в attn_out! Заменяем на 0...")
                attn_out = torch.nan_to_num(attn_out, nan=0.0)
            # ===================
            
            x = self.norm1(x + attn_out)
            
            # ===== ОЧИСТКА =====
            if torch.isnan(x).any():
                print("      ⚠️ nan после norm1! Заменяем на 0...")
                x = torch.nan_to_num(x, nan=0.0)
            # ===================
            
            ffn_out = self.ffn(x)
            
            # ===== ОЧИСТКА =====
            if torch.isnan(ffn_out).any():
                print("      ⚠️ nan в ffn_out! Заменяем на 0...")
                ffn_out = torch.nan_to_num(ffn_out, nan=0.0)
            # ===================
            
            x = self.norm2(x + ffn_out)
            
            # ===== ОЧИСТКА =====
            if torch.isnan(x).any():
                print("      ⚠️ nan после norm2! Заменяем на 0...")
                x = torch.nan_to_num(x, nan=0.0)
            # ===================
            
            return x
    


class ModifiedPatchTST(nn.Module):
    """
    Модифицированный PatchTST для классификации неопределённостей РТК
    Объединяет все компоненты: патчинг, адаптивное кодирование, 
    межканальное внимание, трансформер, классификатор
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_patches = config.get_num_patches()
        
        # ДОБАВЛЕНО: проверка согласованности размерностей
        assert config.D_MODEL % config.NUM_CHANNELS == 0, \
            f"D_MODEL ({config.D_MODEL}) должен быть кратен NUM_CHANNELS ({config.NUM_CHANNELS})"
        
        # Компоненты модели
        self.patch_encoder = PatchEncoder(
            patch_len=config.PATCH_LEN,
            stride=config.PATCH_STRIDE,
            d_model=config.D_MODEL,
            num_channels=config.NUM_CHANNELS
        )
        
        # ===== Адаптивное позиционное кодирование (формула 2.6) =====
        if getattr(config, 'USE_ADAPTIVE_ENCODING', True):
            self.adaptive_pos_encoding = AdaptivePositionalEncoding(
                d_model=config.D_MODEL,
                max_len=self.num_patches
            )
        else:
            self.adaptive_pos_encoding = None
        
        # ===== Межканальное внимание (формула 2.7) =====
        if getattr(config, 'USE_CHANNEL_ATTENTION', True):
            self.channel_attention = ChannelAttention(
                d_model=config.D_MODEL,
                num_channels=config.NUM_CHANNELS
            )
        else:
            self.channel_attention = None

         # ===== Маскирование =====
        self.masking = StochasticMasking(
            d_model=config.D_MODEL,
            p_mask=config.MASK_PROB
        )
        
        # Трансформерные блоки
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.D_MODEL, config.NUM_HEADS, config.DROPOUT)
            for _ in range(config.NUM_LAYERS)
        ])
        
        # Классификатор (глобальное усреднение + линейный слой + softmax)
        # Формула (2.5): y_hat = softmax((1/K) * Σ Z_k * W_c + b_c)
        self.classifier = nn.Linear(config.D_MODEL, config.NUM_CLASSES)
        
        # LayerNorm перед классификатором
        self.final_norm = nn.LayerNorm(config.D_MODEL)
        
    def forward(self, x: torch.Tensor, use_masking: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, window_len, num_channels] - входной временной ряд
            use_masking: bool - применять ли стохастическое маскирование (только на обучении)
        Returns:
            logits: [batch, num_classes] - логиты классов
            embeddings: [batch, num_patches, d_model] - эмбеддинги патчей после трансформера
        """
        batch, window_len, num_channels = x.shape
        assert num_channels == self.config.NUM_CHANNELS, \
            f"Expected {self.config.NUM_CHANNELS} channels, got {num_channels}"
        
        # 1. Патчинг (формула 2.2)
        patches, embeddings, patch_variance = self.patch_encoder(x)
        
        # 2. Стохастическое маскирование (только на обучении)
        if use_masking and self.training:
            embeddings, mask = self.masking(embeddings)
        
        # 3. Адаптивное позиционное кодирование (формула 2.6) - ЕСЛИ ВКЛЮЧЕНО (Эксперимент 6)
        if self.adaptive_pos_encoding is not None:
            embeddings = self.adaptive_pos_encoding(embeddings, patch_variance)
        
        # 4. Межканальное внимание (формула 2.7) - ЕСЛИ ВКЛЮЧЕНО (Эксперимент 6)
        if self.channel_attention is not None:
            embeddings = self.channel_attention(embeddings)
      
        # 5. Трансформерный кодировщик (формулы 2.3, 2.4)
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        
        # 6. Глобальное усреднение по патчам и классификация (формула 2.5)
        pooled = embeddings.mean(dim=1)  # [batch, d_model]
        pooled = self.final_norm(pooled)
        logits = self.classifier(pooled)
        
        return logits, embeddings
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Метод для инференса: возвращает класс и вероятности"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, use_masking=False)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities

    # Эксперимент 8 модификация внимания 
    def get_attention(self, x):
        """
        Безопасное извлечение весов внимания с полной диагностикой.
        """
        self.eval()
        with torch.no_grad():
            print("\n🔍 ДИАГНОСТИКА get_attention:")
            
            # 1. Патчинг
            patches, embeddings, patch_variance = self.patch_encoder(x)

            # ===== ОЧИСТКА =====
            if torch.isnan(embeddings).any():
                print("   ⚠️ nan в embeddings после патчинга! Заменяем на 0...")
                embeddings = torch.nan_to_num(embeddings, nan=0.0)
                patch_variance = torch.nan_to_num(patch_variance, nan=0.0)

            # Проверка внутри патчинга
            print(f"   После патчинга: patches shape={patches.shape}, nan={torch.isnan(patches).any().item()}")
            print(f"   После патчинга: embeddings shape={embeddings.shape}, nan={torch.isnan(embeddings).any().item()}")
            print(f"   После патчинга: patch_variance shape={patch_variance.shape}, nan={torch.isnan(patch_variance).any().item()}")

            
            
            # 2. Адаптивное кодирование
            if self.adaptive_pos_encoding is not None:
                embeddings = self.adaptive_pos_encoding(embeddings, patch_variance)
                if torch.isnan(embeddings).any():
                    print("   ⚠️ nan в embeddings после adaptive_pos_encoding! Заменяем на 0...")
                    embeddings = torch.nan_to_num(embeddings, nan=0.0)
            
            # 3. Межканальное внимание
            if self.channel_attention is not None:
                embeddings = self.channel_attention(embeddings)
                print(f"   После channel_attention: nan={torch.isnan(embeddings).any().item()}")
                if torch.isnan(embeddings).any():
                    print("   ⚠️ nan в embeddings после channel_attention!")
                    embeddings = torch.nan_to_num(embeddings, nan=0.0)
            
            # 4. Проход по всем блокам
            for i, block in enumerate(self.transformer_blocks):
                embeddings = block(embeddings)
                print(f"   После блока {i}: nan={torch.isnan(embeddings).any().item()}")
                if torch.isnan(embeddings).any():
                    print(f"   ⚠️ nan в embeddings после блока {i}!")
                    embeddings = torch.nan_to_num(embeddings, nan=0.0)
            
            # 5. Классификация
            pooled = embeddings.mean(dim=1)
            pooled = self.final_norm(pooled)

            # ===== ОЧИСТКА =====
            if torch.isnan(pooled).any():
                print("   ⚠️ nan в pooled после final_norm! Заменяем на 0...")
                pooled = torch.nan_to_num(pooled, nan=0.0)
            # ===================


            logits = self.classifier(pooled)

            # ===== ОЧИСТКА =====
            if torch.isnan(logits).any():
                print("   ⚠️ nan в logits! Заменяем на 0...")
                logits = torch.nan_to_num(logits, nan=0.0)
            # ===================

            print(f"   После классификации: nan={torch.isnan(logits).any().item()}")
            
            # 6. Внимание — берём из последнего блока
            last_block = self.transformer_blocks[-1]
            attn_out, attn_weights = last_block.attention(
                embeddings, embeddings, embeddings,
                need_weights=True
            )
            # ===== ФИНАЛЬНАЯ ОЧИСТКА =====
            if attn_weights is not None:
                # Принудительно заменяем nan на 0
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                # Если все веса равны 0, создаём равномерное распределение
                if torch.all(attn_weights == 0):
                    print("   ⚠️ Все веса равны 0. Создаём равномерное распределение...")
                    num_patches = attn_weights.shape[-1]
                    attn_weights = torch.ones_like(attn_weights) / num_patches
                # Нормализуем, чтобы сумма по строкам была = 1
                attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
            # ===============================

            if attn_weights is not None:
                print(f"   После очистки: attn_weights min={attn_weights.min().item():.6f}, max={attn_weights.max().item():.6f}")
            
            print("🔍 ДИАГНОСТИКА ЗАВЕРШЕНА\n")
            
            return logits, attn_weights
        
    # Эксперимент 8
    def register_attention_hook(self):
        """
        Регистрирует forward hook для извлечения весов внимания
        """
        self.attention_weights = []
        
        def hook(module, input, output):
            # output[1] — это веса внимания (если need_weights=True)
            if isinstance(output, tuple) and len(output) > 1:
                self.attention_weights.append(output[1])
        
        # Регистрируем хук на последнем блоке
        for block in self.transformer_blocks:
            if hasattr(block, 'attention'):
                block.attention.register_forward_hook(hook)

    # Эксперимент 8
    def get_attention_with_hook(self, x):
        """
        Извлечение сырых весов внимания до softmax (через forward hook)
        """
        self.eval()
        
        # Контейнер для весов
        attention_weights = []
        
        def hook(module, input, output):
            # Вход в MultiheadAttention — это (query, key, value)
            # Мы можем вычислить внимание вручную из query и key
            q, k, v = input[0], input[1], input[2]
            
            # Вычисляем внимание вручную
            # Сначала вычисляем Q @ K^T
            q = q.transpose(0, 1)  # [seq_len, batch, dim]
            k = k.transpose(0, 1)
            
            # Масштабирование
            scale = 1.0 / (q.size(-1) ** 0.5)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Softmax
            attn_weights = torch.softmax(attn_logits, dim=-1)
            attention_weights.append(attn_weights.detach().cpu())
        
        # Вешаем хук на последний блок
        last_block = self.transformer_blocks[-1]
        
        # Сохраняем оригинальный forward
        original_forward = last_block.attention.forward
        
        def wrapped_forward(query, key, value, need_weights=False, **kwargs):
            # Вызываем оригинальный forward
            output, attn_weights = original_forward(query, key, value, need_weights=True, **kwargs)
            # Сохраняем веса
            attention_weights.append(attn_weights.detach().cpu() if attn_weights is not None else None)
            return output, attn_weights
        
        # Подменяем forward
        last_block.attention.forward = wrapped_forward
        
        with torch.no_grad():
            # Патчинг
            _, embeddings, patch_variance = self.patch_encoder(x)
            
            if self.adaptive_pos_encoding is not None:
                embeddings = self.adaptive_pos_encoding(embeddings, patch_variance)
            
            if self.channel_attention is not None:
                embeddings = self.channel_attention(embeddings)
            
            # Проход по блокам
            for i, block in enumerate(self.transformer_blocks):
                if i == len(self.transformer_blocks) - 1:
                    embeddings, _ = block(embeddings, return_attention=True)
                else:
                    embeddings = block(embeddings)
            
            # Классификация
            pooled = embeddings.mean(dim=1)
            pooled = self.final_norm(pooled)
            logits = self.classifier(pooled)
        
        # Восстанавливаем оригинальный forward
        last_block.attention.forward = original_forward
        
        if attention_weights and attention_weights[-1] is not None:
            attn_weights = attention_weights[-1]
            print(f"📊 Веса получены через hook: {attn_weights.shape}")
            return logits, attn_weights
        else:
            print("⚠️ Hook не сработал")
            return logits, None


# Функция для создания модели
def create_model(config) -> ModifiedPatchTST:
    """Создание и инициализация модели"""
    # Фиксация random seed (ДОБАВЛЕНО)
    config.set_seed()
    
    # Проверка конфигурации (ДОБАВЛЕНО)
    config.validate()
    
    model = ModifiedPatchTST(config)
    
    # Инициализация весов Xavier
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Вывод информации о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 50)
    print("Модель ModifiedPatchTST создана")
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"Устройство: {config.DEVICE}")
    print("=" * 50)
    
    return model.to(config.DEVICE)


if __name__ == '__main__':
    # Тест модели
    from config import Config
    
    Config.print_config()
    Config.validate()
    Config.set_seed()
    
    model = create_model(Config)
    
    # Тестовый проход
    batch_size = 2
    x = torch.randn(batch_size, Config.WINDOW_LENGTH, Config.NUM_CHANNELS)
    print(f"\nТестовый проход:")
    print(f"Вход: {x.shape}")
    
    logits, embeddings = model(x, use_masking=False)
    print(f"Логиты: {logits.shape}")
    print(f"Эмбеддинги: {embeddings.shape}")
    print(f"Предсказанные классы: {logits.argmax(dim=-1)}")
    
    # Проверка режима обучения с маскированием
    model.train()
    logits_masked, embeddings_masked = model(x, use_masking=True)
    print(f"\nРежим обучения с маскированием:")
    print(f"Логиты (маскирование): {logits_masked.shape}")
    
    # Проверка метода predict
    model.eval()
    preds, probs = model.predict(x)
    print(f"\nМетод predict:")
    print(f"Предсказания: {preds}")
    print(f"Вероятности: {probs.shape}")