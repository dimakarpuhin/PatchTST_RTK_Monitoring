"""PatchTST по формулам главы 2.

Порядок: патчинг -> LAPE -> SPM -> Transformer -> CA -> классификатор.
LAPE добавляется перед каждым блоком. Если LAPE отключён, обычное
синусоидальное позиционное кодирование добавляется один раз.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class AdaptivePositionalEncoding(nn.Module):
    """LAPE по формулам (2.16)-(2.22)."""

    def __init__(
        self,
        d_model: int,
        max_len: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        if d_model <= 0 or max_len <= 0 or num_layers <= 0:
            raise ValueError("Размерности LAPE должны быть положительными")

        position = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        positional_code = torch.zeros(
            max_len, d_model, dtype=torch.float32
        )
        positional_code[:, 0::2] = torch.sin(
            position * frequencies
        )
        cosine_count = positional_code[:, 1::2].shape[1]
        positional_code[:, 1::2] = torch.cos(
            position * frequencies[:cosine_count]
        )

        self.register_buffer("positional_code", positional_code)

        # Отдельный обучаемый вектор alpha^(l) для каждого слоя.
        # При alpha=0 LAPE совпадает с обычным PE.
        self.alpha = nn.Parameter(
            torch.zeros(num_layers, d_model)
        )

    def forward(
        self,
        patch_std: torch.Tensor,
        layer_index: int,
        adaptive: bool = True,
    ) -> torch.Tensor:
        if patch_std.ndim != 2:
            raise ValueError(
                "patch_std должен иметь форму [batch, num_patches]"
            )

        number_of_patches = patch_std.shape[1]
        if number_of_patches > self.positional_code.shape[0]:
            raise ValueError(
                "Число патчей превышает длину позиционного кода"
            )

        base_code = self.positional_code[
            :number_of_patches
        ].unsqueeze(0)

        if not adaptive:
            return base_code.expand(
                patch_std.shape[0], -1, -1
            )

        if not 0 <= layer_index < self.alpha.shape[0]:
            raise IndexError(
                f"Некорректный индекс слоя: {layer_index}"
            )

        layer_alpha = self.alpha[
            layer_index
        ].view(1, 1, -1)

        # E_pos^(l)(k) = PE(k) * (1 + alpha^(l) * sigma(P_k)).
        scale = 1.0 + patch_std.unsqueeze(-1) * layer_alpha
        return base_code * scale


class StochasticMasking(nn.Module):
    """SPM по формуле (2.26)."""

    def __init__(
        self,
        d_model: int,
        probability: float,
    ) -> None:
        super().__init__()

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Вероятность маскирования должна быть в [0, 1]"
            )

        self.probability = probability
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, number_of_patches, _ = embeddings.shape

        if self.probability == 0.0:
            mask = torch.zeros(
                batch_size,
                number_of_patches,
                dtype=torch.bool,
                device=embeddings.device,
            )
            return embeddings, mask

        mask = (
            torch.rand(
                batch_size,
                number_of_patches,
                1,
                device=embeddings.device,
            )
            < self.probability
        )
        masked_embeddings = torch.where(
            mask,
            self.mask_token.to(dtype=embeddings.dtype),
            embeddings,
        )

        return masked_embeddings, mask.squeeze(-1)


class PatchEncoder(nn.Module):
    """Патчинг и линейная проекция по формуле (2.5)."""

    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        num_channels: int,
    ) -> None:
        super().__init__()

        if patch_len <= 0 or stride <= 0:
            raise ValueError(
                "Длина патча и шаг должны быть положительными"
            )

        self.patch_len = patch_len
        self.stride = stride
        self.num_channels = num_channels
        self.projection = nn.Linear(
            patch_len * num_channels,
            d_model,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                "Вход должен иметь форму "
                "[batch, window_length, num_channels]"
            )

        _, window_length, num_channels = x.shape

        if num_channels != self.num_channels:
            raise ValueError(
                f"Ожидалось каналов: {self.num_channels}; "
                f"получено: {num_channels}"
            )
        if window_length < self.patch_len:
            raise ValueError(
                "Длина окна меньше длины патча"
            )

        patches = torch.stack(
            [
                x[:, start : start + self.patch_len, :]
                for start in range(
                    0,
                    window_length - self.patch_len + 1,
                    self.stride,
                )
            ],
            dim=1,
        )

        # Формула (2.17) делит сумму на M*d.
        patch_std = patches.std(
            dim=(2, 3),
            unbiased=False,
        )

        embeddings = self.projection(
            patches.reshape(
                patches.shape[0],
                patches.shape[1],
                -1,
            )
        )

        return patches, embeddings, patch_std


class TransformerBlock(nn.Module):
    """Post-LayerNorm блок по формулам (2.10)-(2.12)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        attention_output, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        x = self.norm1(x + attention_output)
        x = self.norm2(x + self.ffn(x))

        if return_attention:
            return x, attention_weights
        return x


class ChannelAttention(nn.Module):
    """Межпризнаковое внимание по формуле (2.24)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.output_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # (Z^T Z) / sqrt(d_model), форма [B, D, D].
        gram_matrix = torch.matmul(
            z.transpose(-2, -1),
            z,
        ) / self.scale

        # Нормализация по столбцам каждой строки.
        attention_weights = torch.softmax(
            gram_matrix,
            dim=-1,
        )
        attended = torch.matmul(
            z,
            attention_weights,
        )

        return z + self.output_projection(attended)


class ModifiedPatchTST(nn.Module):
    """Модифицированная архитектура PatchTST из главы 2."""

    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.num_patches = config.get_num_patches()
        self.use_adaptive_encoding = getattr(
            config,
            "USE_ADAPTIVE_ENCODING",
            True,
        )
        self.use_channel_attention = getattr(
            config,
            "USE_CHANNEL_ATTENTION",
            True,
        )

        self.patch_encoder = PatchEncoder(
            patch_len=config.PATCH_LEN,
            stride=config.PATCH_STRIDE,
            d_model=config.D_MODEL,
            num_channels=config.NUM_CHANNELS,
        )
        self.positional_encoding = AdaptivePositionalEncoding(
            d_model=config.D_MODEL,
            max_len=self.num_patches,
            num_layers=config.NUM_LAYERS,
        )
        self.masking = StochasticMasking(
            d_model=config.D_MODEL,
            probability=config.MASK_PROB,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.D_MODEL,
                    num_heads=config.NUM_HEADS,
                    dropout=config.DROPOUT,
                )
                for _ in range(config.NUM_LAYERS)
            ]
        )
        self.channel_attention = (
            ChannelAttention(config.D_MODEL)
            if self.use_channel_attention
            else None
        )
        self.classifier = nn.Linear(
            config.D_MODEL,
            config.NUM_CLASSES,
        )

    def encode(
        self,
        x: torch.Tensor,
        use_masking: bool = False,
        return_attention: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        _, embeddings, patch_std = self.patch_encoder(x)

        mask = None
        last_attention = None

        # Обычный PE для baseline без LAPE.
        if not self.use_adaptive_encoding:
            embeddings = (
                embeddings
                + self.positional_encoding(
                    patch_std,
                    layer_index=0,
                    adaptive=False,
                )
            )

        for layer_index, block in enumerate(
            self.transformer_blocks
        ):
            # LAPE добавляется перед каждым блоком.
            if self.use_adaptive_encoding:
                embeddings = (
                    embeddings
                    + self.positional_encoding(
                        patch_std,
                        layer_index=layer_index,
                        adaptive=True,
                    )
                )

            # SPM: после первого LAPE, до первого блока.
            if (
                layer_index == 0
                and use_masking
                and self.training
            ):
                embeddings, mask = self.masking(
                    embeddings
                )

            is_last_layer = (
                layer_index
                == len(self.transformer_blocks) - 1
            )

            if return_attention and is_last_layer:
                embeddings, last_attention = block(
                    embeddings,
                    return_attention=True,
                )
            else:
                embeddings = block(embeddings)

        # CA применяется после трансформерного кодировщика.
        if self.channel_attention is not None:
            embeddings = self.channel_attention(
                embeddings
            )

        return embeddings, last_attention, mask

    def forward(
        self,
        x: torch.Tensor,
        use_masking: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, _, _ = self.encode(
            x,
            use_masking=use_masking,
            return_attention=False,
        )

        global_embedding = embeddings.mean(dim=1)
        logits = self.classifier(global_embedding)

        return logits, embeddings

    def predict(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        with torch.no_grad():
            logits, _ = self.forward(
                x,
                use_masking=False,
            )
            probabilities = torch.softmax(
                logits,
                dim=-1,
            )
            predictions = probabilities.argmax(dim=-1)

        return predictions, probabilities

    def get_attention(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Возвращает логиты и внимание последнего блока."""

        self.eval()

        with torch.no_grad():
            embeddings, attention, _ = self.encode(
                x,
                use_masking=False,
                return_attention=True,
            )
            logits = self.classifier(
                embeddings.mean(dim=1)
            )

        return logits, attention


def validate_model_config(config) -> None:
    """Проверяет ограничения архитектуры."""

    if config.D_MODEL % config.NUM_HEADS != 0:
        raise ValueError(
            "D_MODEL должен быть кратен NUM_HEADS"
        )
    if config.PATCH_LEN <= 0 or config.PATCH_STRIDE <= 0:
        raise ValueError(
            "PATCH_LEN и PATCH_STRIDE должны быть положительными"
        )
    if config.WINDOW_LENGTH < config.PATCH_LEN:
        raise ValueError(
            "WINDOW_LENGTH не может быть меньше PATCH_LEN"
        )
    if config.get_num_patches() <= 0:
        raise ValueError(
            "Модель должна содержать хотя бы один патч"
        )
    if not 0.0 <= config.MASK_PROB <= 1.0:
        raise ValueError(
            "MASK_PROB должен находиться в [0, 1]"
        )


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def create_model(config) -> ModifiedPatchTST:
    """Создаёт модель с интерфейсом текущего обучения."""

    if hasattr(config, "set_seed"):
        config.set_seed()
    else:
        seed = getattr(config, "RANDOM_SEED", 777)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    validate_model_config(config)

    model = ModifiedPatchTST(config)
    model.apply(initialize_weights)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("=" * 50)
    print("Модель Chapter2PatchTST создана")
    print(f"Всего параметров: {total_parameters:,}")
    print(
        "Обучаемых параметров: "
        f"{trainable_parameters:,}"
    )
    print(f"Устройство: {config.DEVICE}")
    print("=" * 50)

    return model.to(config.DEVICE)


if __name__ == "__main__":
    from config import Config

    Config.WINDOW_LENGTH = 16
    Config.NUM_CHANNELS = 5
    Config.NUM_CLASSES = 3
    Config.PATCH_LEN = 4
    Config.PATCH_STRIDE = 2
    Config.D_MODEL = 32
    Config.NUM_HEADS = 4
    Config.NUM_LAYERS = 2
    Config.DROPOUT = 0.1
    Config.MASK_PROB = 0.15
    Config.USE_ADAPTIVE_ENCODING = True
    Config.USE_CHANNEL_ATTENTION = True

    test_model = create_model(Config)
    test_input = torch.randn(
        2,
        Config.WINDOW_LENGTH,
        Config.NUM_CHANNELS,
        device=Config.DEVICE,
    )

    test_model.train()
    test_logits, test_embeddings = test_model(
        test_input,
        use_masking=True,
    )

    print(f"Логиты: {tuple(test_logits.shape)}")
    print(f"Эмбеддинги: {tuple(test_embeddings.shape)}")
    print(
        "Конечные значения: "
        f"{torch.isfinite(test_logits).all().item()}"
    )
