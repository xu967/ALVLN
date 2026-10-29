# 2026-10-29 22:05 监控台接真实训练（trainer_native + .monitor 侧目录）

## 背景

上一轮评估（`docs/监控系统交接.md` §11）列出三个硬断点：trainer 不认 `--config`、
控制台文件会撞上 `RunOwner` 的空目录规则、界面没有数据集字段。
用户要求「要能完整运行的代码，不是 DEMO」，选了不动 `src/` 的封装方案。

## 做了什么

1. **新增 `ui/trainer_native.py`**（约 70 行）：接受 `--config`，读 `uavrgb.train.v1`，
   拼 `uavrgb_train --data/--epochs/--batch/--lr/--seed/--out`，`os.execv` 替换自身；
   命令框里的额外参数（`--resume` 等）原样追加；配置错误退 2，消息进 `train.log`。
2. **控制台文件挪到 `<run_dir>.monitor/`**（`ui/monitor.py`）：
   - 新增 `monitor_dir()`；`write_train_config`/`spawn_trainer`/`_start_training` 都改用侧目录；
   - `train.log` 改 `wb` 打开（新 run 不续旧日志）；
   - 回滚名单仍覆盖 `train_config.json`/`.tmp`/`train_command.txt`/`.tmp`/`train.log`，只是删在侧目录；
   - `_refresh_train_log` 优先读侧目录，找不到时回退 `<run_dir>/train.log`（兼容旧 run）。
3. **`_set_run_dir` 立刻刷新日志**：切目录后不等下一次 tick（原来最多 5 s 空白）。
4. **自检 +6 项**：native 转 CLI / 透传额外参数 / 命令框给数据集 / 缺 data 报错 /
   run 目录外放控制台文件 / 旧布局 train.log 仍可见。
5. **文档**：交接文档补 §12（用法、落盘布局、实测、仍未做）；`CLAUDE.md` 更新训练控制段；
   `docs/results.md` 追加本轮实测表。

## 没做什么

- **没改 `src/`**：`uavrgb_train` 仍不认 `--config`，`RunOwner` 也没动；封装脚本承担转换。
- **没做界面 resume**：`--resume` 要求非空目录，控制台强制空目录；目前只能命令行续跑，
  再用「选择目录」看曲线（已实测可行）。
- **没加「数据集」输入框**：`--data` 写在「训练命令」里，配置的 `data` 字段可选。
- `steps_per_epoch` 对真实 trainer 无效（步数 = 数据集/batch），保持原样。
- 没动另一会话刚改的 `tail_text`（`state["partial"]` 版本），我的两个 tail 用例对它同样通过。

## 验证到哪一步

| 项 | 真实输出 |
| --- | --- |
| 界面启动真实训练（3 epoch，`/tmp/bench_ds.uavr`） | `训练完成 rc=0`，表格 3 行、事件 26 条 |
| 启动瞬间 run 目录 / 完成后产物 | `[]` → `ckpt.bin config.json metrics.jsonl results.csv` |
| 停止（跑 4 s） | `已停止 rc=-2`，产物文件完整 |
| 命令行 resume 1 epoch | `rc=0`，`results.csv` 4 行，事件 36 条 |
| `DISPLAY=:20 python3 ui/test_monitor.py` | 100 项通过，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | 61 项通过，1 项跳过，EXIT=0 |
| `python3 ui/test_trainer_demo.py` | 89 项通过，EXIT=0 |
| `py_compile` | OK |

未验证：重建后的 trainer（含 RunOwner）从界面启动——当前二进制是 18:03 构建、还没包含 `RunOwner`；
不过启动瞬间 run 目录实测为空，重新构建后应当仍然通过。
