import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Literal

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
    def forward(self, x):
        # x: (B, T, C)
        w = self.attn(x)  # (B, T, 1)
        w = torch.softmax(w, dim=1)
        return (x * w).sum(dim=1)

class ECGTransformer(nn.Module): #Pure Transformer for multi-channel ECG (without CNN/RNN) 
    def __init__(self,
                 input_channels: int,
                 seq_len: int,
                 num_classes: int,
                 d_model: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 pooling: Literal['mean','max','attn'] = 'mean',
                 positional_encoding: Literal['learnable','sin'] = 'learnable',
                 batch_first: bool = True,
                 meta_dim: int = 8):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, d_model)
        if positional_encoding == 'learnable':
            self.pos_enc = LearnablePositionalEncoding(d_model, max_len=seq_len)
        else:
            self.pos_enc = nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling
        if pooling == 'mean':
            self.pool = lambda x: x.mean(dim=1)
        elif pooling == 'max':
            self.pool = lambda x: x.max(dim=1)[0]
        elif pooling == 'attn':
            self.pool = AttentionPooling(d_model)
        else:
            raise ValueError(f'Unknown pooling: {pooling}')
        self.meta_proj = nn.Linear(meta_dim, d_model)
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x, meta): # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1,2)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.pool(x)  # (B, d_model)
        meta_emb = self.meta_proj(meta)  # (B, d_model)
        x_cat = torch.cat([x, meta_emb], dim=1)  # (B, d_model*2)
        out = self.classifier(x_cat)
        return out

# Example usage
if __name__ == '__main__':
    model = ECGTransformer(
        input_channels=12,
        seq_len=5000,
        num_classes=21,
        d_model=128,
        num_layers=4,
        num_heads=8,
        dim_feedforward=256,
        dropout=0.1,
        pooling='mean',
        positional_encoding='learnable',
        batch_first=True,
        meta_dim=8
    )
    x = torch.randn(8, 12, 5000)  # (batch, channels, seq_len)
    meta = torch.randn(8, 8)  # (batch, meta_dim)
    y = model(x, meta)
    print('Output:', y.shape)  # (8, 21) 