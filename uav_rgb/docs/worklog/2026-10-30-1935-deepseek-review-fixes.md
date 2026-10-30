# Codex 审核六项修复 + 上一轮 review 两项收尾

- 时间：2026-10-30 18:45–19:35
- 背景：`docs/CODE_REVIEW_DEEPSEEK_2026-10-30.md` 的 R01–R06 经独立复核全部成立
  （复核结论与级别调整见对话记录）；另修上一轮 ponytail-review 的两条。
- 验证：`bash tools/check.sh` 全量 PASS（CTest **20/20**，195.21 s；全部 Python/UI
  自检）；`--python-only` 亦 PASS（现在包含 preprocess 门禁负例）。

## 做了什么

| # | 编号 | 文件 | 修法 |
| --- | --- | --- | --- |
| 1 | R02 | `sim/tools/audit_dataset.py`、`test_audit_dataset.py` | `episode_facts` 在 `zip` 前校验 actions/oracle 行数、manifest `num_steps`；不一致返回 error → `read_episodes` FAIL。新增短 oracle、坏 `num_steps` 负例 |
| 2 | 上轮 review | `sim/tools/audit_dataset.py`、`test_audit_dataset.py` | 偏航检查拆成「没有偏航步 / 都缺 dyaw / 全是反转步」三种 SKIP，不再用误导文案；新增 10c 用例 |
| 3 | R04/R08 | `src/tests/test_conv_cuda.cpp`、`test_linear_cuda.cpp`、`test_nn_cuda.cpp`、`test_weighted_loss.cpp`、`test_softmax_cuda.cpp`；`src/nn_cpu.cpp`、`ops_cpu.cpp`、`conv_gradcheck.cpp` | 最大误差 helper/gradcheck 遇到非有限差值返回 `inf`（NaN 不再被 `std::max` 吞掉）；补 numel 不一致检查；`test_nn_cuda` 加 NaN helper 自检 |
| 4 | R05 | `src/ops_weighted.cpp`、`src/softmax_cuda.cu`、`test_weighted_loss.cpp` | CPU/CUDA 加权梯度入口校验 `Σw <= FLT_MAX`，超出抛 `invalid_argument`；加 CPU/CUDA 负例 |
| 5 | R06 | `src/nn_cpu.cpp`、`src/nn_cuda.cu`、`test_nn_cuda.cpp` | 梯度范数 sqrt 后检查能表示为 float，超出抛 `runtime_error`；加 CPU/CUDA 负例 |
| 6 | R03 | `src/trainer.cpp`、`test_class_weights_train.cpp` | `carry_resume_best`：校验源 `ckpt_best.bin` 存在且分数/类别权重一致；同目录只校验，跨目录复制到新目录；缺失/不一致直接失败。测试加「新目录必须有 best」断言 |
| 7 | R07 | `src/train_main.cpp`、`src/run_owner.cpp` | `--resume` 只有与 `--out` 同目录才允许复用非空目录；跨目录 resume 只能进空目录，避免覆盖别的 run |
| 8 | R01 | `sim/tools/collect.py`、`sim/tools/student.py`、`sim/tools/test_collect.py` | 动作窗口起点改用**派发前最新位姿**（`exec_ts`），观测位姿 `pose_before/pose_before_ts` 继续用于标签、`pose_gap_ns` 和 `pose_ts`；位移/dyaw/sim_dt/stuck 用执行起点。`StudentPolicy` 启动时发一帧全零 warmup，把 CUDA 首启开销移出第一步。新增假时钟回归（旧代码失败、新代码通过） |
| 9 | 上轮 review | `tools/check.sh` | fixture 和四组 preprocess 门禁负例移出 C++ 分支，`--python-only` 也执行；增加 PASS 行 |
| 10 | 文档 | `docs/TASKS.md`、`docs/experiments/2026-10-30-t6-train.md`、`worklog/2026-10-30-1829-t6-train.md`、`docs/experiments/2026-10-30-t7-closed-loop.md`、`worklog/2026-10-30-1929-t7-closed-loop.md` | T6/T7 验收记录；TASKS T6/T7 done |

## 验证

| 项 | 结果 |
| --- | --- |
| 全量构建 `cmake --build build/check` | exit 0 |
| 完整 CTest | **20/20**，195.21 s |
| `bash tools/check.sh` | PASS（含 C++ 构建、20 项 CTest、全部 Python/UI） |
| `bash tools/check.sh --python-only` | PASS；四组 preprocess 门禁负例现在也在其中 |
| `test_audit_dataset.py` | PASS（含 10c/12a/12b） |
| 审查附录 B 的 R02 探针 | 现在三集都报 `actions 25 行 != oracle 24 行`，`read_episodes=FAIL` |
| R01 假时钟回归 | 对旧 `collect.py` FAIL、对修复后 PASS |
| `test_student.py` | PASS（含 warmup） |
| T7 | SR 0.90 / collision 0.05 / stuck 0.00 / false_forward 0.05（20 seed） |

## 没做什么

- **没有在 R01 修复后重跑 T7**：T7 结果来自修复前 `collect.py`。修复后窗口不同，
  SR 变化方向不确定（窗口变长也可能更容易撞）；0.90 只代表修复前版本，不是
  可证明的下界。已满足 50% 验收线且失败集不是擦线，未花额外 58 min 重跑。
  实验记录 §3 已写明。
- **没有复跑审查的 CUDA 外部探针**：R05/R06 的 CUDA 路径由仓库自带的
  `test_weighted_loss`/`test_nn_cuda` 负例覆盖并通过，没有另编临时探针。
- **没有修文档漂移**（README/HANDOFF 的旧进度描述）：只更新了 TASKS 和本轮
  记录，避免大范围改共享入口文档。
- **没有 commit**：用户未授权。
