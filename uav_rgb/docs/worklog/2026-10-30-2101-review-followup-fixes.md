# ponytail review 跟进修复（CLI 回退、resume 保护、审计对齐、文档口径）

- 时间：2026-10-30 20:20–21:01
- 背景：`/ponytail:ponytail-review` 对 R01–R07 修复的复审提出 7 条
  （1 条 Must fix、2 条 Should fix、4 条 Nice to have）。
- 验证：`bash tools/check.sh` **exit 0**；CTest **20/20**；`test_audit_dataset` PASS；
  `test_collect` 15/15；`test_class_weights_train` PASS。

## 做了什么

| # | 复审项 | 文件 | 修法 |
| --- | --- | --- | --- |
| 1 | Must fix：CLI 回退仍会让动作窗口偏早 | `sim/tools/collect.py`、`gz_io.py`、`sim/README.md` | 删除 `CliTwistPublisher`；常驻 publisher 建不起来时直接 `RuntimeError`，不再回退 `gz topic -p`。CLI 只保留给 `_try_zero` 的零速安全网；同步改注释/README |
| 2 | Should fix：R07 同目录 resume 保护无测试 | `src/include/uavrgb/run_owner.hpp`、`src/run_owner.cpp`、`src/train_main.cpp`、`src/tests/test_run_owner.cpp` | 抽出 `same_run_dir(out, resume)`；加同目录/跨目录/相对路径/空 resume/软链 out 5 个用例 |
| 3 | Should fix：T7 “保守下界”无依据 | `docs/experiments/2026-10-30-t7-closed-loop.md`、`worklog/2026-10-30-1929-t7-closed-loop.md`、`worklog/2026-10-30-1935-deepseek-review-fixes.md`、`docs/TASKS.md` | 统一改成「本轮是 R01 修复前版本；窗口变化对 SR 方向不确定；严格验收需同版本重跑」 |
| 4 | Nice：R02 只查行数，不查 step 对齐 | `sim/tools/audit_dataset.py`、`test_audit_dataset.py` | `episode_facts` 在两侧都有 int `step` 时要求相等；fixture 补 step；新增 12c「行数相同、step 交换」负例 |
| 5 | Nice：NaN 守卫复制 8 份，只有一处自检 | `src/tests/test_conv_cuda.cpp`、`test_linear_cuda.cpp`、`test_weighted_loss.cpp`、`test_softmax_cuda.cpp`、`test_nn_cuda.cpp`、`test_gradcheck.cpp` | 每个测试 helper 加 NaN 负例；`test_softmax_cuda` 的内联循环提成 `max_abs_diff`；5 个库 gradcheck（linear/weighted/conv/relu/maxpool）各加一条非有限输入负例。**没有**合并成共享 header，重复仍在 |
| 6 | Nice：T6 “无梯度清零”无测量依据 | `docs/experiments/2026-10-30-t6-train.md` | 改成「train loss/指标有限（未直接测梯度）」 |
| 7 | Nice：R03 失败分支无测试 | `src/tests/test_class_weights_train.cpp` | 复制源 run 的 `ckpt.bin`、不复制 best，跨目录 resume 必须抛异常 |

## 验证

| 项 | 结果 |
| --- | --- |
| `cmake --build build/check -j 8` | exit 0 |
| `ctest -R` 受影响 7 项 | 7/7 PASS |
| `test_class_weights_train` | PASS（含新增缺 best 负例） |
| `test_audit_dataset.py` | PASS（含 12c） |
| `test_collect.py` | 15/15 PASS |
| `bash tools/check.sh` | exit 0；CTest 20/20 |
| 全文检索旧口径 | 除本 worklog 的引用外，0 处 |

## 没做什么

- **没有合并 4 份 `max_abs_err`/`max_abs_diff` 成共享 header**：复审给的是
  「共享 helper 或每份加自检」，选了后者；代码重复还在，Lean 的 -20 行没有兑现。
- **没有在真实 Gazebo 上强制触发 publisher 失败**：CLI 回退路径已删除，但没有
  做“helper 不可用 → collect 直接失败”的端到端演练；下次采集自然覆盖。
- **没有重跑 T5/T6/T7**：这些数字不受本次改动影响；T7 仍是 R01 修复前版本，
  文档已改成「方向不确定，严格验收需同版本重跑」。
- **没有 commit**：用户未授权。
