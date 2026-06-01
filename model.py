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
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Формула (2.3): Z = LayerNorm(Z + MultiHeadAttn(Z))
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Формула (2.4): Z = LayerNorm(Z + FFN(Z))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
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
        
        self.adaptive_pos_encoding = AdaptivePositionalEncoding(
            d_model=config.D_MODEL,
            max_len=self.num_patches
        )
        
        # ИСПРАВЛЕНО: межканальное внимание теперь работает с [batch, num_patches, d_model]
        self.channel_attention = ChannelAttention(
            d_model=config.D_MODEL,
            num_channels=config.NUM_CHANNELS
        )
        
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
        
        # 3. Адаптивное позиционное кодирование (формула 2.6)
        embeddings = self.adaptive_pos_encoding(embeddings, patch_variance)
        
        # 4. Межканальное внимание (формула 2.7)
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