#!/usr/bin/env python3
"""训练指标自测：macro-F1 必须能暴露「多数类把少数类吃掉」的情况。"""
from train import _accuracy, _macro_f1


def main():
    gts = [1] * 90 + [2] * 10          # 90% FORWARD，10% YAW_LEFT
    # 全猜 FORWARD：acc 很高，macro_f1 很低
    all_fwd = [1] * 100
    assert abs(_accuracy(gts, all_fwd) - 0.90) < 1e-9
    assert abs(_macro_f1(gts, all_fwd) - 0.4737) < 1e-3, _macro_f1(gts, all_fwd)

    # 完美预测：acc=1，macro_f1=1
    assert _accuracy(gts, gts) == 1.0
    assert abs(_macro_f1(gts, gts) - 1.0) < 1e-9

    # 一半 YAW 被漏判：acc 掉得少，macro_f1 掉得多
    half = [1] * 95 + [2] * 5
    assert abs(_accuracy(gts, half) - 0.95) < 1e-9
    # 少数类召回减半：acc 只掉 5 点，macro-F1 掉约 18 点
    assert abs(_macro_f1(gts, half) - 0.8198) < 1e-3, _macro_f1(gts, half)

    print("PASS")


if __name__ == "__main__":
    main()
