import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """Классический Transformer для временных рядов"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.D_MODEL
        num_heads = config.NUM_HEADS
        num_layers = config.NUM_LAYERS
        dropout = config.DROPOUT
        
        # Проекция входных данных в d_model
        self.input_proj = nn.Linear(config.NUM_CHANNELS, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.WINDOW_LENGTH, d_model) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, config.NUM_CLASSES)
        )
        
    def forward(self, x, use_masking=False):
        # x: [batch, window_len, channels]
        x = self.input_proj(x)  # [batch, window_len, d_model]
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)  # [batch, window_len, d_model]
        x = x.transpose(1, 2)  # [batch, d_model, window_len] для pooling
        logits = self.classifier(x)
        return logits, x