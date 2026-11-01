# Teacher / Student-GRU 运行说明

## 环境（CPU torch，装在数据盘）

根分区空间不足，venv 放数据盘。CUDA 版 torch 需要 cuDNN 9，本机没有；
Teacher/原型先用 CPU，GPU 方案待定。

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb
python3 -m venv /root/rivermind-data/uav_rgb/teacher-venv
V=/root/rivermind-data/uav_rgb/teacher-venv/bin
$V/pip install --upgrade pip
$V/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
$V/pip install numpy pillow tqdm
```

当前实测：torch `2.15.0+cpu`、torchvision `0.29.2+cpu`，
`torch.cuda.is_available() == False`。不要安装 `nvidia-cudnn-cu12` /
`nvidia-cuda-*`（CPU wheel 不需要，且根分区放不下）。
本机 PyTorch CPU 源较慢，本次是复用 pip 下载缓存完成安装的；重装若卡在
numpy，可先单独 `$V/pip install numpy pillow tqdm` 再重跑 torch 那行。

## 原型自测

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb
/root/rivermind-data/uav_rgb/teacher-venv/bin/python teacher/test_student_gru_proto.py
```

预期末尾输出 `PASS`；覆盖 `T=1`、`T=5`、前向 `[B,4]`、全部参数有梯度、
pad 位置变化不改输出。

## Teacher 训练（当前被深度阻塞）

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb
/root/rivermind-data/uav_rgb/teacher-venv/bin/python teacher/train.py \
  --data /root/rivermind-data/uav_rgb/l1-v4-relation-500 \
  --epochs 30 --batch 8 --lr 1e-3 --seq-len 8 \
  --out /root/rivermind-data/uav_rgb/runs/teacher-cpu --device cpu
```

现在运行会以「train 集没有可用样本」退出：采集只存 `rgb_*.png`，没有深度文件。
CPU 上 404 集 × 30 epoch 也很慢，建议深度补齐后先 `--epochs 1` 冒烟。

## 当前阻塞

1. **采集不落盘深度**：`actions.jsonl` 有 `depth_ts`，但 episode 目录只有 RGB。
   `EpisodeSequenceDataset(..., require_depth=True)` 在
   `l1-v4-relation-500/train` 上 404/404 全跳过；Teacher 模型必须吃 depth，
   因此在深度导出实现前无法训练。
2. `require_depth=False` 已可用：同一 split 实测 404 个样本可读
   （RGB + text + pose，无 `depth` 键），GRU 原型/学生数据管线可先用。
3. 数据格式只验证过 `l1-v4-relation-500`：manifest 的 `instruction`、
   `actions.jsonl` 的 `pose_before`/`yaw_before`、`rgb_00000.png` 均存在；
   深度文件格式（PNG 位宽/缩放、`depth_ref` 模板）仍未验证。
4. 未下载任何预训练权重（`pretrained=False` 默认），未在真实数据上训练。
5. 分辨率未打通：`dataset.py` 输出 224×224（Teacher 口径），GRU 原型要求
   120×160；把真实数据喂给原型还需 resize/裁剪，本次只做了随机张量自测。

## 本次改动与验证

- 新增 `teacher/student_gru_proto.py`：3 层 conv(16/32/64) → flatten 19200 →
  GRU(128) → 文本 Embedding(22,16)+pad 均值池化 → FiLM → fc(4)。
- 新增 `teacher/test_student_gru_proto.py`：随机张量前向/反向 + pad 忽略自测。
- 修改 `teacher/dataset.py`：新增 `require_depth=False`，保留原深度逻辑。
- 新增 `teacher/RUN.md`（本文件）。
- 未做：深度导出、真实数据训练、CUDA/cuDNN 环境、pre-trained 权重。

| 验证项 | 结果 |
| --- | --- |
| `torch 2.15.0+cpu` 导入 | 通过，`cuda.is_available()=False` |
| `test_student_gru_proto.py` | 通过，输出 `PASS` |
| `require_depth=False` 扫真实 split | 404/404 个 episode 可用 |
| `require_depth=True` 扫同一 split | 0 个可用（无深度文件） |
| 真实数据训练 | 未跑（深度缺失 + 数据格式未确认） |
