# UI 训练监控台（原生程序）+ 桌面快捷方式

> **历史说明（2026-10-28 23:30）**：本文件上半部分记录的是 20:40 的 web 版；20:55 已按用户要求改为本地 Tkinter。`launch_ui.sh` 探 `/api/v1/health`、起 `server.py --port 8799`、开 `http://127.0.0.1:8799` 的描述**仅作历史**，不再是运行方式。当前状态见 `docs/监控系统交接.md`。

- 时间：2026-10-28 20:40（更新 20:55）
- 范围：`ui/` 监控台、`/root/桌面/` 桌面快捷方式
- 结论：完成（已按用户要求由 web 改为本地原生桌面程序）

## 做了什么

- 新增 `ui/launch_ui.sh`：幂等启动训练监控 UI（curl 探 `/api/v1/health`，未跑才起 `server.py --port 8799`，轮询等待就绪），再 `xdg-open` 打开 `http://127.0.0.1:8799`，失败回退 `firefox`。路径从脚本自身推导，仓库可迁移；不 source `env.sh`（server.py 纯标准库，无依赖）。
- 新增 `/root/桌面/uavrgb-monitor.desktop`：XFCE 桌面快捷方式，Exec 指向上述启动脚本，图标 `utilities-system-monitor`，`Terminal=false`。
- 两文件均 `chmod +x`（XFCE 对桌面 .desktop 要求可执行）。

## 没做什么

- 没把 `.desktop` 装进系统应用菜单（`/usr/share/applications/`）或 `~/.config/autostart` 开机自启。
  - 原因：用户只要求「桌面快捷方式」；自启属额外行为，未获授权。
- 没在 `.desktop` 里写死浏览器、没加自定义图标 png。
  - 原因：用系统默认 `xdg-open` 即可，图标用主题自带的系统监控图标，避免仓库里塞二进制。
- 没把 `launch_ui.sh` 做成 symlink 到 PATH 或让 `.desktop` 路径无关。
  - 原因：`.desktop` 的 `Exec` 必须绝对路径；当前写死仓库现路径，属机器本地便利项。

## 待确认

- `.desktop` 的 `Exec` 写死了仓库路径 `/root/claude-code-backup-20261028/code/uav_rgb/ui/launch_ui.sh`。若仓库迁移，需同步改这一行。
- 是否还要「开机自启」（放进 autostart）或「系统应用菜单」入口。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `bash -n ui/launch_ui.sh` | 语法 OK |
| 起 `server.py` → 探 `/api/v1/health` | `{"status":"ok","service":"uavrgb-monitor",...}` |
| `desktop-file-validate` | validate OK（仅提示 Categories 可补 System/Network，非错误） |
| 文件权限 | 两文件 `-rwxr-xr-x` |

## 风险与遗留

- 后台 server 的存活依赖 `nohup` + 父进程为 XFCE 会话；headless 测试 shell 里后台进程会被工具回收，无法验证「桌面双击后服务持续存活」，但 `nohup`+重定向是标准做法。
- `launch_ui.sh` 幂等判断只探 8799 端口；若端口被无关进程占用，会误判「已在跑」而不起服务（端口占用属少见、可自查）。

## 更新 2026-10-28 20:55（web → 原生桌面程序）

用户明确要求：训练监控台做成**本地软件**，不要 web，直接读本地，且要更丰富。据此重写。

### 做了什么

- 新增 `ui/monitor.py`（Tkinter 标准库，无 HTTP、无浏览器、无 matplotlib）：
  - 直读本地 `results.csv`（epoch/train_ce/dev_ce/dev_accuracy）+ `metrics.jsonl`（事件流）
  - 界面：汇总区（总 epochs、当前/最佳 dev_ce、dev_acc）、指标表、两张 Canvas 手绘曲线（loss 蓝=train/橙=dev、dev_accuracy）、事件日志（error/train_step/validation 人性化格式化）
  - 5s 自动刷新 + 「立即刷新」按钮 + 「打开目录」按钮；`--run-dir`/`--refresh` 参数
- 新增 `ui/test_monitor.py`：解析层自检（demo 数据 + 缺失目录 + 损坏 CSV + 坏行 jsonl，11 项断言）。
- 重写 `ui/launch_ui.sh`：直接 `exec python3 monitor.py "$@"`（透传 `--run-dir`），不再起服务/开浏览器。
- 更新 `/root/桌面/uavrgb-monitor.desktop` 的 Comment（去掉 http URL）。
- 更新 `CLAUDE.md` 的 UI 命令段（server.py → monitor.py）。
- **删除**被取代的 web 版：`ui/server.py`、`ui/test_server.py`、`ui/static/`、`ui/__pycache__`。

### 没做什么

- 没做 matplotlib 嵌入或更炫的图表库。
  - 原因：Canvas 手绘已够用，且避免引入 venv 依赖，`python3` 裸跑即可。
- 没做多 run-dir 切换下拉框、没做曲线图 hover/缩放、没做事件过滤。
  - 原因：属可选增强，用户未点名；需要时再加。
- 没删 `ui/demo/`（样例数据）。
  - 原因：仍是默认 run-dir，保留便于首次打开能看到效果。

### 待确认

- 删除 web 版是否彻底符合预期（web 版代码已不可恢复，本仓库非 git）。
- 汇总区的「当前/最佳」取值口径（当前=最后一行，最佳 dev_ce=最小值、dev_acc=最大值）是否要加「最佳 train_ce」等。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python3 -m py_compile monitor.py test_monitor.py` | OK |
| `python3 test_monitor.py` | 11 项断言全部通过 |
| `DISPLAY=:20 timeout 3 python3 monitor.py --run-dir demo` | 窗口正常构造并运行到被超时杀掉（无 traceback，exit=143=SIGTERM） |

### 风险与遗留

- GUI 冒烟只在 `DISPLAY=:20` 跑了 3 秒，未真机点开确认图表/表格渲染视觉效果（headless 无肉眼；但无 traceback 且刷新逻辑在首次构造时已执行）。
- `open_dir` 用 `xdg-open`（Linux）；Windows 需 `os.startfile`，本机不涉及。
