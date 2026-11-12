#!/usr/bin/env python3
"""CnnModel 的 PyTorch 复刻（对照 src/include/uavrgb/model.hpp、src/model.cpp、src/model_dev*.cu）。

结构：3×[Conv3×3 pad1 → ReLU → MaxPool2×2]，通道 in_c→16→32→64，输入 120×160，最后 64×15×20=19200 维
→ Linear(num_classes)；100,388 个参数。在此之上三个可选段，开关与 C++ 的 make_cnn 一一对应：
  文本段   text_dim>0：embedding[22, text_dim]，每个 token 位置各取一行，按位置拼成 9·text_dim 维；
           token id 为 0（pad）或 ≥22 时该位置整段为 0
  prev 段  prev_dim>0：上一步动作 one-hot，拼在文本特征之后（FiLM 条件向量 = [文本特征 | one-hot]）
  FiLM     条件向量维 cond_dim>0 时：Linear(cond_dim→128) 输出前 64 个为 γ、后 64 个为 β，
           作用在第三个池化层的输出上：y' = (1+γ)·y + β，逐通道
  辅助头   aux_classes>0：1×1 卷积（64→aux_classes），接在 FiLM 之后的特征上；只在训练时有用，推理不算
  双帧     in_c=6：[I_t; I_{t-1}] 沿通道拼接
权重布局与 PyTorch 原生一致（conv W=[out,in,kh,kw]，Linear W=[out,in]，NCHW 展平顺序 [c][h][w]），
所以 checkpoint 的 float32 数组可以直接 copy 进来，不需要转置。

参数顺序（cpp_params）同 C++ 的 cnn_params：
  c1.W c1.b c2.W c2.b c3.W c3.b fc.W fc.b [text_emb] [film.W film.b] [aux.W aux.b]
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_io as cio  # noqa: E402

TEXT_LEN = cio.TEXT_LEN
TEXT_VOCAB = cio.TEXT_VOCAB
FILM_CH = cio.FILM_CH


def configure_determinism():
    """推理/对拍用的数值设置：关 TF32（2080 Ti 本来没有，换卡时不让它悄悄打开），cuDNN 不做自动选算法。"""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class ProtoCnn(nn.Module):
    def __init__(self, in_h=120, in_w=160, in_c=3, num_classes=4, text_dim=0, aux_classes=0, prev_dim=0):
        super().__init__()
        cio.validate_structure(in_h, in_w, in_c, num_classes, text_dim, aux_classes, prev_dim)
        self.in_h, self.in_w, self.in_c, self.num_classes = in_h, in_w, in_c, num_classes
        self.text_dim, self.aux_classes, self.prev_dim = text_dim, aux_classes, prev_dim
        self.cond_dim = cio.cond_dim(text_dim, prev_dim)
        self.c1 = nn.Conv2d(in_c, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, FILM_CH, 3, padding=1)
        self.flat_dim = FILM_CH * (in_h // 8) * (in_w // 8)
        self.fc = nn.Linear(self.flat_dim, num_classes)
        self.text_emb = nn.Parameter(torch.zeros(TEXT_VOCAB, text_dim)) if text_dim > 0 else None
        self.film = nn.Linear(self.cond_dim, 2 * FILM_CH) if self.cond_dim > 0 else None
        self.aux = nn.Conv2d(FILM_CH, aux_classes, 1) if aux_classes > 0 else None

    # ---- 与 C++ 对齐的参数视图 ----
    def structure(self):
        return dict(in_h=self.in_h, in_w=self.in_w, in_c=self.in_c, num_classes=self.num_classes,
                    text_dim=self.text_dim, aux_classes=self.aux_classes, prev_dim=self.prev_dim)

    def cpp_params(self):
        """按 cnn_params 的顺序返回参数张量（同一份存储，不是拷贝）。"""
        p = [self.c1.weight, self.c1.bias, self.c2.weight, self.c2.bias, self.c3.weight, self.c3.bias,
             self.fc.weight, self.fc.bias]
        if self.text_emb is not None:
            p.append(self.text_emb)
        if self.film is not None:
            p += [self.film.weight, self.film.bias]
        if self.aux is not None:
            p += [self.aux.weight, self.aux.bias]
        return p

    def xavier_plan(self):
        """C++ make_cnn 里 Xavier 抽样的顺序与 (fan_in, fan_out)：偏置不抽样（构造时清零）。"""
        plan = [(self.c1.weight, self.in_c * 9, 16 * 9), (self.c2.weight, 16 * 9, 32 * 9),
                (self.c3.weight, 32 * 9, FILM_CH * 9), (self.fc.weight, self.flat_dim, self.num_classes)]
        if self.text_emb is not None:
            plan.append((self.text_emb, TEXT_VOCAB, self.text_dim))
        if self.film is not None:
            plan.append((self.film.weight, self.cond_dim, 2 * FILM_CH))
        if self.aux is not None:
            plan.append((self.aux.weight, FILM_CH, self.aux_classes))
        return plan

    def init_xavier(self, uniform):
        """Xavier 均匀初始化：lim=sqrt(6/(fan_in+fan_out))，偏置清零。uniform(numel, lim) 返回长度 numel、
        取值于 (-lim, lim) 的 float32 一维数组；抽样顺序同 C++。传 cpp_rng.CppXavier(seed) 得到逐位相同的初值。"""
        with torch.no_grad():
            for p in self.cpp_params():
                if p.ndim == 1:      # 偏置（一维）清零；权重与 text_emb 都是多维，下面按 Xavier 计划覆盖
                    p.zero_()
            for w, fan_in, fan_out in self.xavier_plan():
                lim = np.float32(np.sqrt(np.float32(6.0) / np.float32(fan_in + fan_out)))   # C++: std::sqrt(6.0f / float(fan_in+fan_out))
                vals = uniform(w.numel(), lim)
                w.copy_(torch.from_numpy(np.asarray(vals, dtype=np.float32)).reshape(w.shape))

    # ---- 前向 ----
    def _cond(self, n, text, prev, device):
        parts = []
        if self.text_dim > 0:
            ids = text.long()
            valid = ((ids > 0) & (ids < TEXT_VOCAB)).unsqueeze(-1)
            emb = self.text_emb[ids.clamp(0, TEXT_VOCAB - 1)]                       # [N, 9, text_dim]
            parts.append(torch.where(valid, emb, torch.zeros_like(emb)).reshape(n, -1))
        if self.prev_dim > 0:
            onehot = torch.zeros(n, self.prev_dim, device=device, dtype=self.film.weight.dtype)
            onehot.scatter_(1, prev.long().unsqueeze(1), 1.0)
            parts.append(onehot)
        return torch.cat(parts, dim=1)

    def features(self, x, text=None, prev=None):
        """返回 FiLM 之后的 [N,64,H/8,W/8] 特征。x：uint8 或 float，形状 [N,in_c,H,W]；uint8 按 /255 缩放。"""
        if x.ndim != 4 or x.shape[1] != self.in_c or x.shape[2] != self.in_h or x.shape[3] != self.in_w:
            raise ValueError(f"forward: x 必须是 [N,{self.in_c},{self.in_h},{self.in_w}]，实际 {tuple(x.shape)}")
        n = x.shape[0]
        if self.text_dim > 0:
            if text is None or tuple(text.shape) != (n, TEXT_LEN):
                raise ValueError("forward: 模型有文本分支，必须传 [N,9] 的 text ids")
        elif text is not None:
            raise ValueError("forward: 模型没有文本分支，不接受 text ids")
        if self.prev_dim > 0:
            if prev is None or tuple(prev.shape) != (n,):
                raise ValueError("forward: 模型接 prev-action，必须传 N 个 prev id")
            if bool(((prev < 0) | (prev >= self.prev_dim)).any()):
                raise ValueError("forward: prev-action id 越界")
        elif prev is not None:
            raise ValueError("forward: 模型不接 prev-action，不接受 prev id")
        if x.dtype == torch.uint8:
            x = x.float() / 255.0       # 除法而不是乘 1/255：设备 kernel 与 C++ 读端都是 x/255.0f，逐位一致
        y = F.max_pool2d(F.relu(self.c1(x)), 2, 2)
        y = F.max_pool2d(F.relu(self.c2(y)), 2, 2)
        y = F.max_pool2d(F.relu(self.c3(y)), 2, 2)
        if self.film is not None:
            cond = self._cond(n, text, prev, x.device)
            gb = self.film(cond)
            gamma, beta = gb[:, :FILM_CH], gb[:, FILM_CH:]
            y = (1.0 + gamma)[:, :, None, None] * y + beta[:, :, None, None]
        return y

    def forward(self, x, text=None, prev=None, want_aux=False):
        y = self.features(x, text, prev)
        logits = self.fc(y.reshape(y.shape[0], -1))
        if want_aux:
            if self.aux is None:
                raise ValueError("forward: 模型没有辅助头")
            return logits, self.aux(y)
        return logits


def softmax_probs(logits):
    """float32 logits [N,C] -> float64 概率。口径同 src/infer.cpp：减最大值后 double 指数，再归一化。"""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def first_argmax(logits):
    """每行第一个最大值的下标（C++ 是 `if (logits[k] > logits[best]) best = k`，平局取小下标；np.argmax 同规则）。"""
    return np.argmax(np.asarray(logits), axis=1)


# ---- checkpoint 互转 ----
def model_from_ckpt(ck, device="cpu"):
    """Ckpt（ckpt_io.read_checkpoint 的结果）-> 载入权重的 ProtoCnn（eval 模式）。"""
    m = ProtoCnn(**ck.structure())
    with torch.no_grad():
        for p, arr in zip(m.cpp_params(), ck.params):
            p.copy_(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)))
    return m.to(device).eval()


def ckpt_from_model(model, *, adam_m=None, adam_v=None, adam_t=0, epoch=0, class_weights=None,
                    best_macro_f1=-1.0, contract=None):
    """ProtoCnn -> Ckpt。Adam 状态不给就写全零（推理用的 checkpoint 不需要它；average_ckpt 沿用最后一轮的形状）。"""
    params = [p.detach().cpu().numpy().astype(np.float32, copy=True) for p in model.cpp_params()]
    zeros = lambda: [np.zeros_like(a) for a in params]   # noqa: E731
    return cio.Ckpt(**model.structure(), params=params, adam_m=adam_m if adam_m is not None else zeros(),
                    adam_v=adam_v if adam_v is not None else zeros(), adam_t=adam_t, epoch=epoch,
                    class_weights=np.zeros(0, np.float32) if class_weights is None else np.asarray(class_weights, np.float32),
                    best_macro_f1=best_macro_f1, contract=contract if contract is not None else bytes(cio.CONTRACT_BYTES))
