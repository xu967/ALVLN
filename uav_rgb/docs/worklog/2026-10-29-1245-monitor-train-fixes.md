# 2026-10-29 12:45 本地训练控制台审核修复

## 背景

对 `ui/monitor.py` 的训练控制（2026-10-29 00:50 交付）做审核，发现 5 项问题并全部修复。
所有问题都先在真实 Tk 窗口里复现，修复后用同一路径复验。

## 做了什么

1. **启动失败后同一目录不能重试**（`ui/monitor.py` `_start_training`）
   - 复现：命令填 `/nonexistent/trainer` → 失败后目录留下 `train.log`/`train_command.txt`/`train_config.json`；
     改好命令再点，报「运行目录不是空目录（请用新目录）」。
   - 修法：在目录确认为空之后置 `rollback = True`；`except` 里只删本次写下的这三个文件，
     不动用户原有内容（「不是空目录」那条在置位前抛出，因此不会误删）。
2. **选中历史 run 时「训练输出」永远空白**（`_refresh_train_log`）
   - 复现：打开 `logs/runs/20261029-123206-demo`（`train.log` 2122 B），输出框 0 字符。
   - 修法：删掉 `if self.proc is None and self._train_state == "idle": return` 两行；
     `tail_text` 对不存在的文件本来就返回空串，空闲时多一次 `stat` 可忽略。
3. **跑完一次后运行目录被写回输入框**（`_on_train_exit`）
   - 修法：退出后 `train_dir_var` 换回 `_default_run_dir()`，下次点击即可用新目录。
4. **删除死代码**：`prepare_train_log` 唯一调用点前刚要求空目录，`train.log` 必然不存在，
   8 MiB 轮转永不触发；连同 `TRAIN_LOG_CAP` 和 `ui/test_monitor.py` 的对应自检一起删除。
   （另一会话的 `docs/worklog/2026-10-29-1205-integration-review.md` 也记录了同一问题。）
5. **`tail_text` 的 64 KiB 上限补 `shortcut:` 注释**：说明日志产出更快时会显示落后，
   接真实 trainer 观察到追不上末尾再改成循环读。

测试：`ui/test_monitor.py` 删 1 项死代码自检、加 5 项（启动失败提示 / 失败不留产物 /
同目录可重试 / 重试完成 / 历史 run 日志可见）。

## 没做什么

- 没改 DEMO trainer、`--config` 合同、停止信号链、桌面快捷方式。
- 没让"运行目录不是空目录"支持复用（仍要求新目录），只保证失败回滚不污染目录。
- 没把 `tail_text` 改成循环读到 EOF：现在 64 KiB/tick 够用，按 `shortcut:` 注释留待真实 trainer。
- 没做 resume/checkpoint、实验管理、打包。

## 验证到哪一步

| 项 | 真实输出 |
| --- | --- |
| `DISPLAY=:20 python3 ui/test_monitor.py` | 89 项通过，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | 55 项通过，1 项跳过，EXIT=0 |
| `python3 ui/test_trainer_demo.py` | 89 项通过，EXIT=0 |
| `py_compile`（monitor/trainer_demo/test_monitor） | OK |
| 失败 → 重试同目录（真实窗口） | 失败后目录 `[]`；重试 `running`，`rc=0` |
| 历史 run 的 train.log | 1887 字符可见（修复前 0） |
| 训练结束后的默认目录 | `20261029-124456-demo`（自动换新） |

未验证：真实 C++/CUDA trainer（不存在）；XFCE 桌面会话内双击快捷方式（DISPLAY 只在桌面会话注入）。
