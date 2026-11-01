#!/usr/bin/env python3
"""Student-GRU 原型自测：随机张量前向/反向 + pad 忽略断言，不依赖真实数据。"""
import sys

import torch
import torch.nn.functional as F

from student_gru_proto import IMG_H, IMG_W, NUM_ACTIONS, TEXT_LEN, TEXT_VOCAB, StudentGRU


def make_batch(batch, seq, device):
    # 固定 seed=0，便于复现。
    torch.manual_seed(0)
    rgb = torch.randn(batch, seq, 3, IMG_H, IMG_W, device=device)
    text = torch.randint(1, TEXT_VOCAB, (batch, TEXT_LEN), device=device)
    text[:, 3:] = 0  # 后 6 个 token 用 <pad> 填充
    labels = torch.randint(0, NUM_ACTIONS, (batch,), device=device)
    return rgb, text, labels


def check_forward_backward(model, seq, device):
    rgb, text, labels = make_batch(2, seq, device)
    model.zero_grad(set_to_none=True)
    logits = model(rgb, text)
    assert logits.shape == (2, NUM_ACTIONS), logits.shape

    loss = F.cross_entropy(logits, labels)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(param.grad).all(), f"{name} 梯度含非有限值"

    print(f"T={seq}: logits={tuple(logits.shape)} loss={loss.item():.4f} 反向梯度齐全")


def check_pad_ignored(model, device):
    """同一组非 pad token，放在不同位置 + 不同 pad 分布，输出必须一致。"""
    model.eval()
    rgb = torch.zeros(1, 1, 3, IMG_H, IMG_W, device=device)
    text_a = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0, 0]], device=device)
    text_b = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 2, 3]], device=device)
    with torch.no_grad():
        logits_a = model(rgb, text_a)
        logits_b = model(rgb, text_b)
        all_pad = model(rgb, torch.zeros(1, TEXT_LEN, dtype=torch.long, device=device))

    torch.testing.assert_close(logits_a, logits_b, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(all_pad).all(), "全 <pad> 输入产生了非有限值"
    print("pad 位置改变时输出不变；全 <pad> 输入无 NaN")


def main():
    torch.manual_seed(0)  # 固定初始化，保证自测可复现
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device}")
    model = StudentGRU().to(device)
    check_forward_backward(model, seq=1, device=device)
    check_forward_backward(model, seq=5, device=device)
    check_pad_ignored(model, device)
    print("PASS")


if __name__ == "__main__":
    sys.exit(main())
