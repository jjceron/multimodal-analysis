from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.dropout1(attn_out)

        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class EEGFormer(nn.Module):
    def __init__(
        self,
        n_channels: int = 24,
        n_samples: int = 256,
        num_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_length: int = 64,
        pool1: int = 4,
        pool2: int = 4,
        d_model: int | None = 32,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout_eeg: float = 0.4,
        dropout_transformer: float = 0.3,
        dropout_classifier: float = 0.5,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.num_classes = num_classes

        temporal_padding = kern_length // 2
        separable_padding = 16 // 2

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kern_length),
                padding=(0, temporal_padding),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout2d(dropout_eeg),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                padding=(0, separable_padding),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout2d(dropout_eeg),
        )

        d_used = F2 if d_model is None else int(d_model)

        self.proj = (
            nn.Identity()
            if d_used == F2
            else nn.Linear(F2, d_used, bias=False)
        )

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_used,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_transformer,
        )

        self.attn_pool = AttentionPooling(d_used)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_classifier),
            nn.Linear(d_used, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}.")

        x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)

        x = x.squeeze(2)
        x = x.transpose(1, 2)

        x = self.proj(x)
        x = self.encoder(x)
        x = self.attn_pool(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits