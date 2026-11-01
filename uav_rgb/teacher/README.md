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

**以下字段是脚本的假设，当前 `sim/tools/collect.py` 并不完全满足，运行前请确认：**

1. **深度文件（最大风险）**：`collect.py` 只在内存里解码深度，**没有保存深度图**。
   `dataset.py` 按以下顺序找每步深度，找不到就跳过该集并打印警告：
   - `actions.jsonl` 行内 `depth_file` / `depth_frame`；
   - manifest 的 `depth_ref` / `depth_dir`，支持 `{step}`、`%05d` 模板或目录名；
   - 默认 `<episode>/depth_%05d.png`。
   支持 8/16 位 PNG（按 255/65535 归一化）、浮点 PNG（mode F，取原值）和 `.npy`。
2. **位姿**：假设 `actions.jsonl` 每行有 `pose_before: {x,y,z}` 和 `yaw_before`
   （角度，与 `collect.py` 一致）。模型输入为 `[x, y, z, yaw(rad), vx, vy]`，
   速度由相邻帧 `rgb_ts` 差分得到。缺这些字段的集会被跳过。
3. **样本组织**：每个 episode 取**最后 `seq_len` 帧**组成一个序列样本，
   标签取**最后一帧**的 `action_expert`（0..3）。
4. **RGB**：用行内 `frame`，否则 `rgb_%05d.png`；统一缩放到 224×224 并做
   ImageNet 归一化。
5. **指令**：取 manifest 的 `instruction`，用 `sim/tools/instruction.py`
   tokenize 后补 `<pad>` 到 9；超过 9 个 token 不截断，整集跳过。
6. 任何缺 RGB/深度/位姿、帧数不足 `seq_len`、标签越界的 episode 都会跳过并告警，
   不会中断扫描。因此**当前未补深度文件的数据集会得到 0 个样本**。

## 三条命令

训练（按 val 准确率保存 `ckpt_best.pth`）：

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
