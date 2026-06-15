import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    GRU модель для классификации временных рядов
    Сравнение с модифицированным PatchTST и LSTM
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = 128
        self.num_layers = 2
        
        self.gru = nn.GRU(
            input_size=config.NUM_CHANNELS,      # 4 параметра (d)
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 64),  # *2 для bidirectional
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config.NUM_CLASSES)     # 5 классов
        )
    
    def forward(self, x, use_masking=False):
        """
        x: [batch, window_len, channels]
        возвращает logits и эмбеддинги (для совместимости с train.py)
        """
        gru_out, hidden = self.gru(x)
        # Берём последний выход последовательности
        last_out = gru_out[:, -1, :]
        logits = self.classifier(last_out)
        return logits, gru_out