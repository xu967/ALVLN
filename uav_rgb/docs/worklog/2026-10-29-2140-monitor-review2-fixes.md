# 2026-10-29 21:40 监控台二审修复（默认目录 / 大日志 / 回滚名单）

## 背景

对 `ui/monitor.py` 训练控制做完 12:45 那一轮修复后再审一遍，发现 3 项问题并全部修复。
前两项先在真实 Tk 窗口里复现，改完用同一路径复验；第三项只能靠自检里的假写盘函数触发。

## 做了什么

1. **快跑结束后默认目录算回刚用过的目录**（`_default_run_dir`）
   - 复现：DEMO `epochs=1, steps=1`（约 0.2 s）。用掉的目录 `20261029-211951-demo`，
     退出后 `train_dir_var` 仍是 `20261029-211951-demo`，再点开始 →
     `启动失败: 运行目录不是空目录`。
   - 原因：`_default_run_dir` 用 `%Y%m%d-%H%M%S`，同一整秒内起停会算出同名路径；
     启动即崩（rc≠0）的 trainer 同理。
   - 修法：拼出路径后 `while os.path.exists(path)` 依次加 `-2`、`-3` 后缀。
2. **`tail_text` 在 64 KiB 处切断行/汉字**（`tail_text`）
   - 复现：3000 行 × 41 B 日志，第一块以 `...loss=0.` 结尾、第二块以
     `12345 action_acc=0.678` 开头；中文日志在切点前后各出现一个 `�`。
   - 修法：读满 `max_bytes` 且 `f.tell() < st.st_size`（后面还有内容）时，
     `chunk.rfind(b"\n")` 退到本块最后一个换行之后再记 offset。单行超过 64 KiB
     时 `rfind` 返回 -1，按原样输出，不会卡住。
3. **写盘失败留下的 `.tmp` 不在回滚名单**（`_start_training`）
   - `_write_atomic` 先写 `*.tmp` 再 `os.replace`；写盘中途失败（磁盘满/EIO）会留下
     `train_config.json.tmp` 或 `train_command.txt.tmp`，回滚只删三个固定名字，
     重试又报「运行目录不是空目录」。
   - 修法：回滚元组里加这两个 `.tmp` 名字。

测试（`ui/test_monitor.py`，+5 项）：

- 无显示层：`tail_text 按行边界切`、`tail_text 中文不被切坏`（186 KB 中文短行日志）。
- GUI 层：`GUI 结束后默认目录换新`、`GUI 默认目录跳过已存在`、`GUI 写盘失败回滚 .tmp`
  （最后一项把 `monitor.write_train_config` 换成「写一个 .tmp 再抛 OSError」）。

## 没做什么

- 没改 DEMO trainer、`--config` 合同、停止信号链、桌面快捷方式、图表与表格绘制。
- 没做增量解析 `results.csv`：实测 1 万行解析 70 ms、表格重建 110 ms、折线 2 ms，
  在 1 s 刷新下无感；10 万行才会顶到 1.7 s/tick，本项目不会跑那么多 epoch。
- 没处理「单行超过 64 KiB」和「文件末尾没有换行的半行」：正常写入都是整行 flush，
  属于病态输入，`tail_text` 保持简单。
- 没改 `docs/监控系统交接.md` 里 202 行那段与本任务无关的历史串行文字。

## 验证到哪一步

| 项 | 真实输出 |
| --- | --- |
| `DISPLAY=:20 python3 ui/test_monitor.py` | 94 项通过，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | 57 项通过，1 项跳过，EXIT=0 |
| `python3 ui/test_trainer_demo.py` | 89 项通过，EXIT=0 |
| `py_compile`（monitor/test_monitor/trainer_demo） | OK |
| 快跑复验（真实窗口） | 用掉 `20261029-213653-demo`，退出后默认 `20261029-213653-2-demo`（不存在），再点开始 → `运行中 pid=595558`，跑完 `rc=0` |
| 大日志复验（真实窗口） | 186 KB 中文日志 3 tick 读完：2000 行全部是`中文日志行：epoch 完成`，`含替换符: False` |

未验证：真实 C++/CUDA trainer（不存在）；磁盘满的真实现场（`.tmp` 回滚只用假写盘函数测过）；XFCE 桌面会话内双击快捷方式。
