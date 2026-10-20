"""A compact 1-D residual network for 12-lead ECG at 100 Hz.

Small on purpose: the study is about what happens after the score, so the
backbone only needs to be a credible, reproducible baseline.
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, k: int = 7):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity()
            if c_in == c_out and stride == 1
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.skip(x))


class ECGResNet(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5, width: int = 32, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, width, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock(width, width),
            ResBlock(width, width * 2, stride=2),
            ResBlock(width * 2, width * 4, stride=2),
            ResBlock(width * 4, width * 4, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, leads, samples) -> logits
        return self.head(self.blocks(self.stem(x)))


__all__ = ["ECGResNet"]
