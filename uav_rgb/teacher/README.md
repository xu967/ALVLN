# Teacher（离线软标签教师）

## 用途与边界

`PrivilegedTeacher` 输入 RGB、深度、位姿和指令，输出 4 类动作 logits。
它**只用于离线生成软标签**（logits + 512 维特征），训练学生模型时可做蒸馏目标。

**Teacher 仅离线使用，交付模型仍是 C++/CUDA。** 本目录不进入设备端推理路径，
学生模型的自研 C++/CUDA 实现和 `src/include/uavrgb/text.hpp` 的常量才是交付口径。

## 安装

本目录需要 PyTorch/PyTorchVision（仅教师离线训练用），**不在**
`tools/requirements.txt` 里，也不要装进学生/数据工具的同一环境。请自备一个隔离
Python 环境并安装 `torch`、`torchvision`、`numpy`、`Pillow`；仓库不会自动安装。

默认 `pretrained=False`，不下载 ImageNet 权重。需要预训练 ResNet18 时加
`--pretrained`（会联网下载）。

## 数据格式与假设（需用户确认）

输入目录结构：

```
<data>/
  train/<episode_id>/episode_manifest.json + actions.jsonl + rgb_*.png
  val/...
  test/...
```

代码读取的常量已对齐现有文件：

- `sim/tools/instruction.py` 的 `VOCAB`（22 个 token）
- `src/include/uavrgb/text.hpp` 的 `kTextLen = 9`、`kTextVocab = 22`

当前采集支持 `--save-depth`，保存 `png16_mm` 深度，位姿包含机身系目标向量。
数据读取采用滑动窗口并用首帧补齐早期窗口；逐窗口预测末帧动作。
教师输入是特权信息，仅学生的 RGB+文本属于交付输入。数据与模型字段以
`dataset.py` / `model.py` 为准。蒸馏标签必须完整覆盖，使用
[`UAVSOFT1`](../tools/SOFT_TARGETS_FORMAT.md) 绑定数据/文本/概率，旧裸数组须重导出。

## 三条命令

训练（按 val macro-F1 保存 `ckpt_best.pth`）：

```bash
python3 teacher/train.py --data /root/rivermind-data/uav_rgb/<dataset> \
  --epochs 30 --batch 8 --lr 1e-3 --seq-len 8 \
  --out /root/rivermind-data/uav_rgb/runs/teacher --device cuda
```

导出软标签（`--data` 指向某个 split 目录本身）：

```bash
python3 teacher/export_soft_targets.py --data /root/rivermind-data/uav_rgb/<dataset>/test \
  --ckpt /root/rivermind-data/uav_rgb/runs/teacher/ckpt_best.pth \
  --out /root/rivermind-data/uav_rgb/runs/teacher/soft_targets.npz \
  --seq-len 8 --device cuda
```

验证 `.npz`（每个 episode 一组 `logits[4]` 与 `feat[512]`）：

```bash
python3 - <<'PY'
import numpy as np
z = np.load("/root/rivermind-data/uav_rgb/runs/teacher/soft_targets.npz")
ids = z["episode_ids"]
assert len(ids) > 0
for eid in ids:
    assert z[f"{eid}/logits"].shape == (4,), eid
    assert z[f"{eid}/feat"].shape == (512,), eid
print(f"OK: {len(ids)} episodes, logits(4) feat(512)")
PY
```

## 文件

- `model.py`：`PrivilegedTeacher`（ResNet18 + 深度编码器 + 文本 Bi-GRU/注意力 + 时序 Bi-GRU）。
- `dataset.py`：episode → 序列样本，容错跳过不完整集。
- `train.py`：训练并保存 `ckpt_best.pth`。
- `export_soft_targets.py`：导出 `{episode_id: {logits, feat}}` 到 `.npz`。
