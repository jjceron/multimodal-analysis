from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectGroupedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset) -> None:
        self.base = base_dataset
        self.subjects: dict[str, dict] = defaultdict(lambda: {"indices": [], "label": None})
        for i in range(len(base_dataset)):
            name = base_dataset.names[i]
            label = base_dataset.labels[i] if hasattr(base_dataset, "labels") else base_dataset.samples[i]["label"]
            self.subjects[name]["indices"].append(i)
            self.subjects[name]["label"] = label
        self.subject_names = list(self.subjects.keys())

    def __len__(self) -> int:
        return len(self.subject_names)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, int]:
        name = self.subject_names[idx]
        info = self.subjects[name]
        windows = []
        for i in info["indices"]:
            _, X, _ = self.base[i]
            windows.append(X)
        return name, torch.stack(windows), info["label"]


def collate_subjects(
    batch: list[tuple[str, torch.Tensor, int]],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    names = [item[0] for item in batch]
    windows = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    max_w = max(w.size(0) for w in windows)
    C, T = windows[0].size(1), windows[0].size(2)
    padded = torch.zeros(len(batch), max_w, C, T)
    mask = torch.zeros(len(batch), max_w, dtype=torch.bool)

    for i, w in enumerate(windows):
        n_w = w.size(0)
        padded[i, :n_w] = w
        mask[i, :n_w] = True

    return names, padded, labels, mask


class SubjectAttentionModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int = 2,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.attention = nn.Sequential(
            nn.Linear(n_classes, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Identity()
        self.n_classes = n_classes

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, W, C, T = x.shape
        logits, _ = self.backbone(x.view(B * W, C, T))
        window_logits = logits.view(B, W, self.n_classes)
        scores = self.attention(window_logits)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        alpha = F.softmax(scores, dim=1)
        agg = (alpha * window_logits).sum(dim=1)
        return agg
