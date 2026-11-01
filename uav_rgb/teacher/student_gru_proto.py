"""Student-GRU 原型：单目 RGB 序列 + 文本 FiLM + GRU → 4 类动作 logits。

只用于低成本验证「GRU 是否有用」，不参与交付；通道数与
src/model.cpp 对齐（3→16→32→64，三次 2x2 pool 后 flatten 到 64*15*20=19200）。
"""
import torch
import torch.nn as nn

IMG_H = 120
IMG_W = 160
CNN_CHANNELS = (16, 32, 64)
GRU_HIDDEN = 128
TEXT_LEN = 9        # 与 text.hpp kTextLen 一致
TEXT_VOCAB = 22     # 与 text.hpp kTextVocab 一致
TEXT_DIM = 16
NUM_ACTIONS = 4


class StudentGRU(nn.Module):
    """RGB 序列过 3 层 conv → GRU → 文本 FiLM 调制 → fc。"""

    def __init__(self):
        super().__init__()
        c1, c2, c3 = CNN_CHANNELS
        # 120x160 → 60x80 → 30x40 → 15x20，展平为 64*15*20=19200。
        self.cnn = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flat_dim = c3 * (IMG_H // 8) * (IMG_W // 8)  # 19200
        self.gru = nn.GRU(self.flat_dim, GRU_HIDDEN, batch_first=True)

        # <pad>=0 的 embedding 不参与均值池化；FiLM 输出逐通道 γ/β。
        self.text_embed = nn.Embedding(TEXT_VOCAB, TEXT_DIM, padding_idx=0)
        self.text_film = nn.Linear(TEXT_DIM, 2 * GRU_HIDDEN)
        self.fc = nn.Linear(GRU_HIDDEN, NUM_ACTIONS)

    def _text_film(self, text):
        """忽略 <pad> 的均值池化 → Linear → (γ, β)，各 (B,128)。"""
        if text.ndim != 2 or text.shape[1] != TEXT_LEN:
            raise ValueError(f"text 形状必须是 [B,{TEXT_LEN}]，实际 {tuple(text.shape)}")
        mask = text.ne(0)  # [B,L]，0 = <pad>
        emb = self.text_embed(text) * mask.unsqueeze(-1)
        # 全是 pad 时池化为 0，避免除零。
        pooled = emb.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        gamma, beta = self.text_film(pooled).chunk(2, dim=-1)
        return gamma, beta

    def forward(self, rgb_seq, text):
        """rgb_seq (B,T,3,120,160), text (B,9) -> logits (B,4)。"""
        if rgb_seq.ndim != 5 or tuple(rgb_seq.shape[2:]) != (3, IMG_H, IMG_W):
            raise ValueError(
                f"rgb_seq 形状必须是 [B,T,3,{IMG_H},{IMG_W}]，"
                f"实际 {tuple(rgb_seq.shape)}")
        b, t = rgb_seq.shape[:2]

        # 逐帧共享 CNN 权重，再按时间展开给 GRU。
        feat = self.cnn(rgb_seq.flatten(0, 1)).flatten(1)  # (B*T,19200)
        out, _ = self.gru(feat.view(b, t, self.flat_dim))  # (B,T,128)
        h = out[:, -1]

        gamma, beta = self._text_film(text)
        h = (1.0 + gamma) * h + beta
        return self.fc(h)


if __name__ == "__main__":
    # 直接运行只做一次形状冒烟；完整前向+反向自测见 test_student_gru_proto.py。
    model = StudentGRU()
    logits = model(torch.randn(2, 5, 3, IMG_H, IMG_W), torch.zeros(2, TEXT_LEN, dtype=torch.long))
    print(f"logits {tuple(logits.shape)}, flat_dim={model.flat_dim}")
