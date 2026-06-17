from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        emb_size: int = 24,
        num_channels: int = 22,
        temporal_kernel: int = 25,
        num_filters: int = 24,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(1, temporal_kernel),
            stride=(1, 1),
        )
        self.spatial_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(num_channels, 1),
            stride=(1, 1),
        )
        self.bn = nn.BatchNorm2d(num_features=num_filters)
        self.activation = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride))
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Conv2d(
            in_channels=num_filters,
            out_channels=emb_size,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        x = x.squeeze(2).transpose(1, 2)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, seq_len: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(emb_size, 256),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        return self.net(x)


class EEGConformer(nn.Module):
    def __init__(
        self,
        emb_size: int = 32,
        num_channels: int = 22,
        n_samples: int = 256,
        temporal_kernel: int = 25,
        num_filters: int = 32,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        num_heads: int = 4,
        dim_feedforward: int = 96,
        num_layers: int = 3,
        num_classes: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            emb_size=emb_size,
            num_channels=num_channels,
            temporal_kernel=temporal_kernel,
            num_filters=num_filters,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        seq_len = (n_samples - temporal_kernel + 1 - pool_kernel) // pool_stride + 1
        seq_len = max(1, seq_len)
        self.head = ClassificationHead(
            emb_size=emb_size, seq_len=seq_len, dropout=dropout, num_classes=num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.head(x)
        return x
