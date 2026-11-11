# 卡 W1-1：走廊判据进两条流水线的默认评测命令

- 时间：2026-11-11 23:40（`check.sh --python-only` 的结果在文末「更新」小节追加）
- 范围：`tools/e1_pipeline.py`、`tools/dagger_gz_pipeline.py`、两个流水线各自的测试文件
- 分支：`p12-w1-1`，基于 main `a714b54`
- 结论：代码与单测完成；`check.sh --python-only` 还没跑（按 11-11 23:15 的排队规矩延后），见「验证到哪一步」。

## 做了什么

- `tools/e1_pipeline.py`：`CFG` 加 `corridor_half_width`，缺省 3.0，环境变量 `E1_CORRIDOR_HALF_WIDTH` 可覆盖，设 0 关闭；`evaluate()` 在该值大于 0 时给每组评测命令追加 `--corridor-half-width 3`（`:g` 格式）。命令的其余部分与改前逐字相同。
- `tools/dagger_gz_pipeline.py`：`CFG` 加同名同环境变量的一项；`eval_cmd()` 追加同一对参数。dagger 臂、more_demos 臂、基线行、60 seed 清单的评测都经 `eval_cmd()`，一并带上。
- `tools/test_e1_pipeline.py`：新增 `test_corridor_half_width_in_eval_commands`。用子进程跑 `e1_pipeline.py --from eval --dry-run`（`CFG` 在 import 时读环境变量，只有新进程能验证默认值与覆盖），断言四组命令默认带 `--corridor-half-width 3`；`E1_CORRIDOR_HALF_WIDTH=0` 时没有，且命令与默认相比只少这两项；设 `2.5` 时按 `:g` 格式透传。
- `tools/test_dagger_gz_pipeline.py`：新增两项。`test_eval_commands_carry_corridor_half_width` 用现有的夹具跑进程内 dry-run，5 条评测命令默认带 `3`，`CFG` 设 0 后没有且只少这两项；`test_corridor_half_width_env_override` 在新进程里读 `CFG`，验证默认 `3.0`、环境变量 `0` 与 `2.5`。
- 没有改 `sim/tools/eval_closed_loop.py`，它的 `--corridor-half-width` 缺省仍是 `None`。

## 没做什么

- 没改 `sim/tools/eval_closed_loop.py` 的默认值。原因：任务卡明写保持 `None`。
- 没改 `ui/`。原因：公共规则。
- 没给 `tools/compare_datasets.py`（第 241 行起的评测命令）与 `tools/train_student_with_video.py`（`build_eval_command`）的柱阵评测加这个开关。原因：任务卡只列了两条流水线。这两处仍按旧口径出数字（`metric_version` 不带 `.corridor3m`）。
- 没给 `dagger_gz_pipeline.summary()` 加「metric_version 混合」提示。原因：超出卡的范围；`e1_pipeline.summary()` 已有这个提示。
- 没重评任何已有基线，没起 Gazebo。原因：本卡不占 gz，测试只用 dry-run 与替身进程。
- 没改 `progress.md`、`CLAUDE.md`、主仓库工作树。原因：执行规则。

## 待确认

- `dagger_gz_pipeline` 的基线行复用要求 `metric_version` 与本次 dagger 臂相同。核对结果：`$D/e1-pipeline-20261107-0417-eval4/` 下 6 个基线的 `eval.json` 都是 `sim09.v2.coast1s`，没有走廊。默认带走廊之后，这 6 行不再被复用，会在 `baselines()` 里全部重评（每行 20 个 seed，各占 1 个 gz）。要继续复用旧基线，起跑时设 `E1_CORRIDOR_HALF_WIDTH=0`，但 dagger 臂也会不带走廊。是否接受重评由主会话定。
- `compare_datasets.py` 与 `train_student_with_video.py` 是否也要默认带走廊，需要主会话决定是否另开卡。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python tools/test_e1_pipeline.py`（新测试 + 旧实现） | 红：`AssertionError: 评测命令缺 --corridor-half-width：…--policy teacher…`，rc=1（`$D/tmp/phase12-cards/w1-1/red-e1.txt`） |
| `python tools/test_e1_pipeline.py`（新实现） | 绿：`ok corridor：4 组评测命令默认带 --corridor-half-width 3，设 0 时没有，设 2.5 透传`，末行 `test_e1_pipeline: PASS`，rc=0（`green-e1.txt`） |
| `python tools/test_dagger_gz_pipeline.py -v`（新测试 + 旧实现） | 红：`FAILED (failures=2)`，`Ran 20 tests`（`red-dagger.txt`） |
| `python tools/test_dagger_gz_pipeline.py -v`（新实现） | 绿：`Ran 20 tests … OK`，rc=0（`green-dagger.txt`） |
| `e1_pipeline.py --from eval --dry-run` 的真实输出 | 默认 4 条命令都带 `--corridor-half-width 3`；`E1_CORRIDOR_HALF_WIDTH=0` 的 4 条没有；不建输出目录（`dryrun-e1-eval.txt`） |
| 对拍：main 的 `e1_pipeline.py --from eval --dry-run` 与新实现设 `E1_CORRIDOR_HALF_WIDTH=0` 的输出（路径前缀与时间戳归一后） | 4 条命令 `diff` 无差异；默认命令去掉 `--corridor-half-width 3` 后也无差异（`dry-old-main.txt`、`dry-new-off.txt`、`dry-new-default.txt`） |
| `UAV_RGB_PYTHON=$R/.venv-tools/bin/python bash tools/check.sh --python-only` | 未跑：排队规矩要求等 W1-0 第二轮结束后用守护脚本串行跑，结果见下方「更新」小节 |

草稿与日志目录：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/w1-1/`。

## 风险与遗留

- 续跑旧输出目录（`--from eval` 或同一 `--out` 重跑）时，已有的 `eval.json` 按 `pipeline_common.eval_finished` 判为完成而跳过，不会按走廊口径重评。旧目录里的评测与新评测的 `metric_version` 会不同：`e1_pipeline.summary()` 会提示混合，`dagger_gz_pipeline` 没有这个提示（读代码得出，未实测）。
- 默认值改变后，`E1_CORRIDOR_HALF_WIDTH` 在模块 import 时读取。同一进程里改环境变量不生效，要改 `CFG` 字典。
- `dagger_gz_pipeline` 里的 `corridor_half_width` 与 `e1_pipeline.CFG` 各写了一份字面量 `"3.0"`，这是任务卡的写法（两个 `CFG` 各加一行）。只改一边会让两条流水线口径不一致，测试没有锁这一点。

## 更新（2026-11-12 00:55）：`check.sh --python-only`

W1-0 第二轮 00:31:43 写出 `DONE`（rc=0）后，守护脚本按排队规矩串行等 `check.lock`，00:43:58 拿到锁。

| 项 | 结果 |
| --- | --- |
| `UAV_RGB_PYTHON=<主仓库 .venv-tools>/bin/python bash tools/check.sh --python-only`（`nice -n 10`，`PYTHONDONTWRITEBYTECODE=1`，`TMPDIR` 在数据盘；工作树 `p12-w1-1`，HEAD `c1938e7`） | rc=0；327 s（00:43:58 到 00:49:25，不含排队）；起跑时负载均值 26.27、24.59、22.18 |
| 末行 | `PASS（有 SKIP）：所选检查完成，但未检查：C++/CUDA 构建、CTest、GPU 测试、cf_sitl；模拟采集、真实训练和学生闭环须独立验收。` |
| 日志里本卡的测试 | `test_e1_pipeline: PASS`（含 `ok corridor：4 组评测命令默认带 --corridor-half-width 3，设 0 时没有，设 2.5 透传`）；`test_dagger_gz_pipeline` 为 `Ran 20 tests … OK` |

日志：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/w1-1/check-python-only.log`（1273 行），完成标记 `check-python-only.done`。
