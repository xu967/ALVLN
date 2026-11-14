#!/usr/bin/env python3
"""可选的较大 Torch-only 模型；不伪装成 C++ checkpoint。"""
import torch
import torch.nn as nn


class BaseCnn(nn.Module):
    """简单 RGB 导航动作模型，先用于 profile smoke 和离线训练实验。"""

    def __init__(self, in_h=120, in_w=160, num_classes=4, channels=(32, 64, 128)):
        super().__init__()
        if in_h % 8 or in_w % 8:
            raise ValueError("BaseCnn 输入高宽必须是 8 的倍数")
        c1, c2, c3 = channels
        self.in_h, self.in_w, self.num_classes = int(in_h), int(in_w), int(num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, num_classes))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, self.in_h, self.in_w):
            raise ValueError(f"BaseCnn: x 必须是 [N,3,{self.in_h},{self.in_w}]")
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.head(self.features(x))
