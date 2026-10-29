# C++/CUDA 审核：4 个「静默成功」P1 修复

- 时间：2026-10-29 22:15
- 范围：`src/trainer.cpp`、`src/train_main.cpp`、`src/class_weights.cpp`、`src/checkpoint.cpp`、
  `src/nn_cpu.cpp`、`src/nn_cuda.cu`、对应测试
- 结论：**完成**。全是「训练 exit 0，但产物不可信/不可用」类问题；
  每条都有修复前复现、修复后实测。

## 做了什么

1. **显式 0 类别权重不再训练中途炸**（`class_weights.cpp`）
   - 复现：`--class-weights 0,1,0,1` 被接受；纯类 0 的 batch 让 Σw=0，
     `softmax_cuda: weights 按样本求和为 0`，训练中止、无 checkpoint。
   - 修法：入口统计训练集每类样本数，给存在的类显式赋 0 直接
     `invalid_argument`；训练集不存在的类仍允许 0。
2. **results.csv / metrics.jsonl 写失败不再静默**（`trainer.cpp`）
   - 复现：把 `results.csv` 链到 `/dev/full` 后 `--resume` 运行：exit 0，
     `results.csv` 0 字节。原实现只查 open，不查 flush。
   - 修法：每次 `flush()` 后检查流状态，失败抛 `runtime_error`；表头写入也检查。
3. **非有限 dev 指标不再写 `null` / 选成 best**（`trainer.cpp`、`train_main.cpp`）
   - 复现：`--lr 1e38 --epochs 1` → exit 0，`results.csv` 出现 `null`，
     `best checkpoint` 仍选中该失效模型。
   - 修法：dev 的 ce/acc/macro_f1/balanced_acc 任一非有限 → 写 Fatal 事件并中止；
     CLI 入口把 lr 限制为 `0<lr<=1`（1e38 直接 exit 2）。
4. **checkpoint 读入/写出校验有限性 + ReLU 传播 NaN**（`checkpoint.cpp`、`nn_cpu.cpp`、`nn_cuda.cu`）
   - 复现：checkpoint 里 `c1.W[0]=NaN` 被静默加载；conv 产生 18921 个 NaN 激活，
     ReLU 把它们压成 0，loss 有限、exit 0，坏参数继续写回。
   - 修法：`save_checkpoint`/`load_checkpoint` 对每个 param/Adam m/v 校验有限性；
     CPU/CUDA ReLU 前反向遇到 NaN 继续传播，让 loss 守卫立即触发。
5. **梯度 norm 非有限时 CPU/CUDA 一致抛错**（`nn_cpu.cpp`、`nn_cuda.cu`）
   - 修复前的分叉：CPU `min(1, NaN)=1.0` 不变换；CUDA `raw > 1 ? 1 : raw` 返回 NaN。
   - 修法：两边 `!isfinite(sum)` 都抛 `runtime_error`，不让 NaN 静默进入 Adam。

## 没做什么

- **device 层 `best_macro_f1` 未接入**（`src/device/state.cpp`）：device 状态还没有
  接训练主路径，单独加字段没有端到端意义；记入遗留。
- **checkpoint 落盘 fsync**：未做断电一致性；现有 tmp+rename 能防半截，但不保证掉电持久。
- **resume RNG 状态**：未保存 mt19937，续训 shuffle 序列从头开始，不能逐位复现。
- **BatchReader 批中途失败的部分写入**、**UavrDataset 64 位乘法溢出**、
  **RunOwner unlink 后互斥失效**：均为未接入主路径/防御性 P2，未改。
- **CUDA 多 block 测试**（softmax N>256、maxpool 重叠窗口）：未补，仍只有单 block 用例。
- **`--config`**：由并行会话的 `ui/trainer_native.py` 在 UI 侧解决，`src/` 未加 JSON 解析。
- 未做真实数据长时间训练、掉电、显存耗尽注入。

## 待确认

- `--lr` 上限 1.0 是否过严（UI 也是 1.0；正常 Adam 远低于此）。
- 显式 0 权重被拒绝后，是否还需要“跳过纯零权重 batch”的语义（当前直接不允许）。

## 验证到哪一步

| 命令/场景 | 修复前 | 修复后 |
| --- | --- | --- |
| `--class-weights 0,1,0,1 --batch 1` | 训练中途 `Σw=0` 中止 | 入口 `训练失败: ... 类 0 设了 0 权重`，rc=1 |
| `results.csv` → `/dev/full`，`--resume` | exit 0，0 字节 | `训练失败: 写 results.csv 失败（磁盘满/IO 错误）`，rc=1 |
| `--lr 1e38` | 训练 exit 0，CSV 出现 `null`，坏模型成 best | `--lr 必须是 0<lr<=1 的有限数`，rc=2 |
| 含 NaN 参数的 checkpoint | 静默加载并继续训练 | `checkpoint: param[0]: 含非有限值`，`invalid_argument` |
| `relu_forward/backward(NaN)` | 返回 0 | CPU/CUDA 都返回 NaN |
| 梯度含 NaN 的 `global_grad_norm` | CPU 返回 1.0、CUDA 返回 NaN | CPU/CUDA 都抛 `runtime_error` |
| `ctest -R 'test_nn_cuda|test_class_weights_train|test_trainer|test_overfit'` | — | 全过（90.86 s，含 overfit） |

未验证：真实数据、跨版本 resume、掉电、磁盘满下的 checkpoint 保存。

## 风险与遗留

- **行为变严**：显式 0 权重、lr>1、NaN 参数现在都会失败；原先“能跑完”的脚本可能开始报错，
  这是把静默坏结果改成显式失败。
- **ReLU NaN 传播**会改变含 NaN 输入的数值结果（0 → NaN），依赖旧行为的第三方代码不存在。
- checkpoint 校验会遍历全部参数（100k 级），开销可忽略；但对超大模型需注意。
- 上面列的 P2 仍未修，尤其 device 层 best、RNG、fsync。
