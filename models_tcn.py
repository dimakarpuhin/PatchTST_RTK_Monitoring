import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    """Один блок TCN с dilated convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=(kernel_size-1)*dilation//2, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size-1)*dilation//2, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)

class TCNModel(nn.Module):
    """TCN для классификации временных рядов"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_channels = config.NUM_CHANNELS
        num_classes = config.NUM_CLASSES
        hidden_sizes = [64, 128, 128, 64]
        kernel_size = 5
        dropout = 0.1
        
        self.blocks = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(zip([num_channels] + hidden_sizes[:-1], hidden_sizes)):
            dilation = 2 ** i
            self.blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_sizes[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, use_masking=False):
        # x: [batch, window_len, channels] -> [batch, channels, window_len]
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        logits = self.classifier(x)
        return logits, x