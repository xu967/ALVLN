# 类别权重 + 碰撞样本过滤（解决 89% FORWARD 的类别不平衡）

- 时间：2026-10-29 10:00
- 范围：`src/`（加权 loss、训练器、数据集）+ `tools/preprocess.py`
- 结论：完成。**代码由 subagent 实现，它在提交报告前因 API 余额不足中断；本文档由主会话按实际构建/测试结果补写**

## 做了什么

### 1. 加权 softmax 交叉熵（CPU + CUDA）
- `src/include/uavrgb/ops.hpp` / `src/ops_cpu.cpp` / `src/softmax_cuda.cu`：新增加权版本
- 归一化定义：`L = -Σ w[y_i]·log p_i / Σ w[y_i]`，梯度 `dlogits = w[y_i]·(p - onehot) / Σ w`
- **权重全 1 时与无权版本逐位一致**（`all-ones vs unweighted: loss_bits_equal=1 grad_bits_equal=1`）

### 2. 类别权重策略 `src/class_weights.cpp` + `class_weights.hpp`
- `resolve_class_weights(spec, ds)`：`""` → 无权重；`"auto"` → 训练集**逆频率**；`"w0,w1,w2,w3"` → 显式
- `merge_resume_weights`：resume 时 CLI 显式给了且与 checkpoint 不一致就报错，没给就沿用 checkpoint 的
- `write_run_config`：把生效参数与权重写进 `<out_dir>/config.json`

### 3. 碰撞样本过滤
- `tools/preprocess.py --drop-collision`：跳过 `collision=true` 的步
- `src/dataset.cpp`：可选过滤开关（`LoadOptions`），默认不过滤保持原行为
- `uavrgb_train --drop-collision` 同步支持

### 4. CLI
```
uavrgb_train --data <dataset.uavr> --epochs N --batch B --lr L --out <run_dir>
             [--resume <ckpt>] [--seed S] [--max-grad-norm G]
             [--class-weights auto|<w0,w1,w2,w3>] [--drop-collision]
preprocess.py --out OUT [--size WxH] [--drop-collision] episodes...
```

## 实测结果（我重跑的，命令见「验证到哪一步」）

| 项 | 结果 |
| --- | --- |
| 权重全 1 vs 无权版 | **loss 与 grad 逐位相同** |
| 加权 FP64 gradcheck | **2.829e-08**（门槛 1e-5） |
| 加权 CPU/CUDA parity | loss 差 **0**，grad 差 **1.16e-10** |
| 碰撞过滤 | train 8→5、val 4→3 样本 |
| **少数类召回（核心证据）** | 不加权重 **0.0** → 加 auto 权重 **0.5** |
| auto 权重实际值 | `[0.2778, 2.5, 0, 0]`（180:20 的 90/10 不平衡） |
| ctest | **12/12 全绿**（新增 `test_weighted_loss`/`test_dataset_filter`/`test_class_weights_train`） |
| sm_75 | 只有 `sm_75` |

**核心证据解读**：在 90/10 的不平衡合成集上，不加权时模型**完全学不会少数类**（召回 0）；
加逆频率权重后召回升到 0.5，代价是多数类召回从 1.0 降到 0.733。这正是我们期望的权衡。

## 没做什么

- **没做 focal loss**（只做了加权 CE）。当前 89:4:3:3 的分布下加权够用；若加权后仍学不动再考虑。
- **没做重采样/过采样**（只做了权重与过滤）。
- **没在真实采集数据上验证**：所有实验都是合成的。真实数据落盘后要重跑一次，看加权是否让模型真的学会转向。
- **没做类别权重的自动调参**：逆频率是固定公式，没有按验证集表现调。
- **没有 subagent 的报告**：它在写报告前因 API 402 中断，所以没有它自述的「没做什么」，本文档的负面清单是我按代码和测试反推的。
- 没做混合精度、BN、dropout、数据预取。

## 待确认

- `auto` 用的是逆频率 `N/(C·n_c)`，在两个极端（某类 0 样本、某类占比 >90%）时的行为需要再看：
  测试里出现 `w=[0.2778, 2.5, 0, 0]`，说明只出现过的两个类有权重，未出现的类是 0 —— 权重 0 意味着该类样本对 loss 无贡献，
  这在「验证集有但训练集没有」的类上是合理的，但可能掩盖问题，值得复核。
- 碰撞过滤默认关闭。真实数据里碰撞帧占比若超过 5%，建议默认开启。

## 验证到哪一步

| 命令 | 结果 |
| --- | --- |
| `cmake --build src/build` | 通过（ninja: no work to do） |
| `ctest --test-dir src/build --output-on-failure` | **12/12 通过**（96.3 s） |
| `cuobjdump -lelf src/build/libuavrgb_core.a` | 仅 `sm_75` |
| `./test_weighted_loss` | PASS（bits_equal=1/1，gradcheck 2.829e-08，parity 1.16e-10） |
| `./test_dataset_filter` | PASS（train 8→5，val 4→3） |
| `./test_class_weights_train` | PASS（少数类召回 0 → 0.5） |
| `uavrgb_train` 未知参数提示 | 已列出 `--class-weights` 与 `--drop-collision` |
| `preprocess.py --help` | 含 `--drop-collision` |

## 风险与遗留

- **加权只解决「学得到」，不解决「样本够」**：少数类样本总量仍然很少（120 步里只有 9 步转向），
  权重会让模型更关注它们，但每类几十个样本撑不起泛化。真正的解法还是提高障碍密度（另一条线在做）。
- 加权后多数类召回会下降（实测 1.0 → 0.733）。上线时要在「不撞」和「不卡」之间权衡，需要闭环指标才能判断。
- `config.json` 与 `metrics.jsonl` 新增了字段，`ui/monitor.py` 未同步（它只读固定列），不影响现有显示。
