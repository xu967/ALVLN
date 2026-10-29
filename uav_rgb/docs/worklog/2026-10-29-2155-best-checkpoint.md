# best-checkpoint：不再交付最后一轮模型

- 时间：2026-10-29 21:55
- 范围：`src/trainer.cpp`、`src/trainer_eval.cpp`（新增）、`src/trainer_internal.hpp`（新增）、
  `src/checkpoint.cpp`、`src/include/uavrgb/checkpoint.hpp`、`src/CMakeLists.txt`、
  `src/tests/test_trainer.cpp`
- 结论：**完成**。每 epoch 按 dev 宏 F1 保存 `ckpt_best.bin`；checkpoint 升到 v3 并兼容读 v2。
  18/18 CTest 通过，含新增的峰值一致性、best 文件读回、v2 兼容断言。

## 做了什么

- **拆文件腾出行数**：`Trainer::evaluate` 移到 `src/trainer_eval.cpp`；`slice`/`take_labels`
  移到内部头 `src/trainer_internal.hpp` 供两个 `.cpp` 共用。`trainer.cpp` 277 → 235 行（上限 280）。
- **best-checkpoint**（`src/trainer.cpp`）：dev 宏 F1 严格高于历史峰值时保存
  `<run_dir>/ckpt_best.bin`，并追加一条 `status` 事件。dev 回退到 train 时不选模型。
- **状态入盘**（`checkpoint.*`）：`TrainState.best_macro_f1` 写进 checkpoint v3。
  续训时从 checkpoint 读回，避免用更差的模型覆盖已有 `ckpt_best.bin`。
  读端接受 v2（best=-1）与 v3，仍拒绝 v1 与未知版本。
- **测试**（`test_trainer.cpp`）：
  - `state.best_macro_f1` 必须等于 `metrics.jsonl` 里 dev 宏 F1 的峰值（1e-5 容差）；
  - `ckpt_best.bin` 存在、可 `load_checkpoint`，epoch 在 1..epochs；
  - v3 往返保留 best 值；
  - 把 v3 文件原地降级成 v2（改 version、删 best 字段）后仍能读回，best=-1、参数一致。
- 文档：`README.md`、`CLAUDE.md`、`docs/HANDOFF.md`、`docs/ENGINEERING_RULES.md` 同步。

## 没做什么

- **没实现仿真时间动作窗口**（§19.3）。`--hold` 仍是墙钟秒，采集不可复现。原因：独立改造，
  必须用真实 Gazebo episode 复验；本轮没有启动仿真。
- **没改车道几何/专家策略**。原因：场景设计决策，上一轮实测已证伪“调净空参数”这条路线。
- **没在真实采集数据上训练**。测试用 `src/tests/synthetic_dataset.hpp` 的合成集。
  原因：本轮只验证训练器行为，不代表真实数据精度。
- **没做 best-checkpoint 的 UI 展示**。监控台仍显示 `results.csv` 的全部 epoch；
  best 值只在 `metrics.jsonl` 的 status 事件和 checkpoint 里。
  `ui/` 归另一会话，未改。
- **没有让旧二进制读 v3**。兼容是单向的（新读旧）；旧训练器遇到 v3 会报
  `version 不支持: 3`。原因：格式升级的必然结果，写进风险。
- **没处理 best 与 test split 的关系**：test 仍只在训练结束评估一次，不参与选模型。

## 待确认

- 选模型口径是否就用 dev 宏 F1？`EVAL_STANDARD.md` 要求报 acc/macro_f1/balanced_acc/每类召回，
  但没规定“选模型看哪个”。当前选宏观 F1，与类别不平衡的结论一致。
- 是否要在 CLI 完成行打印 best epoch/metric（现在只在 `metrics.jsonl` 里）。

## 验证到哪一步

| 命令 | 真实输出 |
| --- | --- |
| `cmake --build build/check --parallel 8` | 成功，无 warning 输出 |
| `ctest --test-dir build/check --output-on-failure` | **18/18 passed**，99.04 s |
| `./build/check/test_trainer` | `best dev macro_f1=0.136364 state=0.136364`；末轮 dev macro_f1=0.0294；`RESULT: PASS` |
| v2 兼容用例 | 同一测试内通过（参数一致、best=-1） |
| `wc -l src/trainer.cpp` | 235 行（<280） |

未验证：真实数据训练、续训跨 v2/v3 的端到端 CLI 流程、磁盘写失败时 best 保存的回滚
（`save_checkpoint` 本身是 tmp+rename）。

## 风险与遗留

- **v3 不向后兼容旧二进制**：旧构建的 `uavrgb_train` 读新 checkpoint 会直接报错。
  如需回退版本，必须先用旧代码重新训练或用新代码导出。
- **dev 集很小时 best 可能过拟合 val**。当前数据 val 只有几十到几百帧；
  正式实验仍应以 scene 级隔离的 dev 和最终 test 为准。
- `ckpt_best.bin` 与 `ckpt.bin` 都写 Adam 状态，磁盘占用翻倍（100k 参数级别可忽略）。
- 状态事件每提升一次写一条；训练结束没有“best 最终值”的独立汇总行。
