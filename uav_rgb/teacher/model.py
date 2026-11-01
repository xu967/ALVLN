"""PrivilegedTeacher：仅离线生成软标签的特权教师。

输入 RGB + 深度 + 位姿 + 指令 token，输出 4 类动作 logits。
结构固定，必须与 student 侧 C++/CUDA 推理口径对齐的只有输出类别数与
文本常量（kTextLen=9 / kTextVocab=22，见 src/include/uavrgb/text.hpp）。
"""
import torch
import torch.nn as nn
import torchvision.models as models

TEXT_LEN = 9        # 与 text.hpp kTextLen 一致
TEXT_VOCAB = 22     # 与 text.hpp kTextVocab 一致
VISUAL_DIM = 512
DEPTH_DIM = 64
POSE_DIM = 6        # [x, y, z, yaw, vx, vy]，由 dataset.py 组装
NUM_ACTIONS = 4


class PrivilegedTeacher(nn.Module):
    """ResNet18 + 深度编码器 + 文本 Bi-GRU/注意力 + 时序 Bi-GRU。"""

    def __init__(self, num_classes=NUM_ACTIONS, pretrained=False):
        super().__init__()
        # pretrained=True 会联网下载 ImageNet 权重；默认关闭，离线环境保持 False。
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        # 去掉最后的 fc，保留全局池化，输出 (N, 512, 1, 1)。
        self.visual = nn.Sequential(*list(resnet.children())[:-1])

        # 深度编码器：1 -> 32 -> 64，池化掉空间维，得到每帧 64 维。
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # 文本：Embedding + Bi-GRU，hidden=256 -> 双向 512，供 512 维注意力使用。
        self.text_embed = nn.Embedding(TEXT_VOCAB, 128, padding_idx=0)
        self.text_gru = nn.GRU(128, 256, batch_first=True, bidirectional=True)

        # 文本引导视觉：视觉特征做 query，文本序列做 key/value。
        self.text_attn = nn.MultiheadAttention(VISUAL_DIM, 8, batch_first=True)

        # 时序：每帧拼接 [视觉 512, 深度 64, 位姿 6] = 582。
        self.temporal_gru = nn.GRU(
            VISUAL_DIM + DEPTH_DIM + POSE_DIM,
            256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, rgb, depth, pose, text, return_feat=False):
        """rgb (B,T,3,H,W), depth (B,T,1,H,W), pose (B,T,6), text (B,L)。"""
        b, t = rgb.shape[:2]

        # 视觉/深度编码按帧展平后共享权重。
        vis = self.visual(rgb.flatten(0, 1)).flatten(1).view(b, t, VISUAL_DIM)
        dep = (self.depth_encoder(depth.flatten(0, 1)).flatten(1)
               .view(b, t, DEPTH_DIM))

        text_feat, _ = self.text_gru(self.text_embed(text))  # (B,L,512)
        # 每个时间步的视觉特征 attend 到整句指令上。
        guided, _ = self.text_attn(vis, text_feat, text_feat)  # (B,T,512)

        seq = torch.cat([guided, dep, pose], dim=-1)  # (B,T,582)
        out, _ = self.temporal_gru(seq)
        feat = out[:, -1]  # (B,512)，取最后时间步
        logits = self.fc(feat)
        if return_feat:
            return logits, feat
        return logits
