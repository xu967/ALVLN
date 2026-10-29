# 训练监控台本地化优化 + 代码审核

- 时间：2026-10-28 23:30 CST
- 范围：`ui/` 监控台、启动脚本、Tk 环境、监控测试、`docs/`；对全项目 C++/CUDA/sim 只做只读审核
- 结论：完成（监控台本地 Tk 程序可用；50 项自检通过；发现并记录核心/仿真问题，未改动）

## 做了什么

- 通读 `uav_rgb` 全部源码/文档：`ui/`、`src/`（C++/CUDA 头文件、实现、6+1 个测试）、`sim/`（world、teacher、collect、save_frame）、`experiments/`、`env.sh`、`CLAUDE.md`、`docs/`，以及设计规格 `TinyAerialVLN_Final_V8/references/reviews/UBUNTU_BASE_GUIDE_V1.md` 和监控相关契约 `docs/07_API_LOGGING_CONTRACT.md`、`docs/14_UI_BACKEND_OPERATION.md`。
- 判定运行模式：`nvidia-smi -L` 列出 RTX 2080 Ti → **GPU 模式**；真跑环境回归，不用“理论上能跑”代替。
- 环境修复：系统 `/usr/bin/python3` 3.12.3 原先缺 tkinter，桌面会话 PATH 不含 conda，双击快捷方式会起不来。`apt-get install -y python3-tk`（3.12.3-0ubuntu1）安装成功；`/usr/bin/python3`、`/opt/selkies-venv/bin/python3`、conda 3.11.15 均可 `import tkinter`（Tk 8.6）。
- `ui/launch_ui.sh`：增加解释器探测（`python3` → `/usr/bin/python3` → `/opt/conda/bin/python3`，找第一个能 `import tkinter` 的）；仍是纯 `exec monitor.py`，无 HTTP/浏览器/端口。
- `ui/monitor.py`：保留本地 Tkinter 架构，做以下优化：
  - `metrics.jsonl` 增量 tail（state 记录 inode/offset/partial；截断/替换 reset；首次只读末尾 1 MiB）；文件签名 `(mtime_ns,size,inode)` 未变时跳过读取/重绘。
  - 解析健壮性：UTF-8-sig、`errors="replace"`、非 object JSON 跳过、坏行跳过、空/非整数 epoch 不进入曲线、NaN/inf 过滤、csv.Error 保留已解析行。
  - 刷新异常安全：`_tick` 用 `try/except/finally`，单次异常不再停掉自动刷新；错误写状态栏 + stderr traceback。
  - Tk 安全：孤立代理字符 `_safe_text`；事件日志保留滚动位置、error 红色、暂停；启动/缩放 `<Configure>` 去抖重画；缺失点分段、单点画圆点。
  - 可用性：目录选择、暂停/继续、DEMO 标识、底部状态栏（数据年龄/缺失文件）、路径缩写、`xdg-open` 失败反馈、`--refresh` 有限且 ≥0.5 s、按 epoch 排序/最大 epoch 作为“当前”。
- `ui/test_monitor.py`：从 11 项（含 1 项恒真断言）重写为 **50 项**；覆盖 demo、缺失目录、损坏/空 epoch/BOM/非 UTF-8/超长字段、非 object/代理字符、tail 追加/半行/截断/替换/limit、断点/单点绘制、GUI 表格/日志/曲线/缩放/空 epoch/暂停/异常重排。
- `docs/results.md`：新增；记录 200k 行基准、GUI 动态验证、环境回归、旧 web 进程残留，命令与真实输出分开保存。
- `docs/监控系统交接.md`：新增；文件地图、启动方式、数据合同、行为、验证、监控台限制、全项目审核发现、环境残留、下一步。
- `docs/环境说明.md`：补充 `python3-tk` 组件行与启动器解释器说明。
- `CLAUDE.md`：更新 UI 启动/自检命令，注明 `DISPLAY=:20`、增量 tail、选择目录/暂停、解释器选择。
- `docs/worklog/2026-10-28-2040-ui-desktop-shortcut.md`：顶部加“历史说明”，标明上半部分 web 版（8799/server.py）已被 20:55 的本地 Tk 版取代。
- 真实 GUI 截图：`ui/evidence/monitor_demo.png`（980×660，DISPLAY=:20）。
- 清理：交付前删除 `ui/__pycache__`；删除 `/tmp` 临时日志/截图。
- 用两个 subagent 做独立只读审核（监控 UI、C++/CUDA/sim）；未让其修改仓库。

## 没做什么

- **没有写 NativeTrainer**（E5），没有真实训练 run；监控台只用 `ui/demo` + 合成 200k 行日志验证。
  - 原因：本次任务是监控系统；训练器/数据协议是下一阶段。
- **没有修 §8 的核心/仿真问题**（collector 动作不停止、RGB 动作后抓取、失败步骤非原子、teacher 全无效深度回前进、N=0、ActionAdapter NaN、CUDA exit、SDF 命名等）。
  - 原因：本次范围是监控台；且并行会话在审核期间持续修改 `sim/tools/collect.py`（交付时约 282 行）、在 `src/tests` 新增 `test_nn_cuda`，此时改 sim/C++ 有冲突风险。已写入交接文档 §8。
- **没有把 TinyAerialVLN 的 NavLab Studio web 前端改成本地软件**。
  - 原因：用户指向 `code/uav_rgb` 的监控系统；NavLab Studio 是另一个工程（静态 web 原型 + HTTP 服务端），若这是本意需另开任务。见交接文档 §1/§7。
- **没有做 AppImage/PyInstaller/单文件打包**。
  - 原因：当前交付是本地 Tk 程序 + 桌面快捷方式；打包会引入新依赖，未获明确要求。
- **没有做 stdout/stderr 面板、事件过滤/搜索、CSV/PNG 导出、多 run 历史/对比、run-dir 记忆**。
  - 原因：属可选增强；先把现有监控台稳定下来。
- **没有杀旧 web 监控进程 PID 133198**（`ui/server.py --port 8799`，文件已删、进程仍在）。
  - 原因：`kill -TERM 133198` 被权限分类器拒绝（“未验证身份、未获用户授权终止”）；未绕过。需要用户执行 `kill 133198`。
- **没有改 `/root/桌面/uavrgb-monitor.desktop`**，只做了 `desktop-file-validate`；没有加应用菜单/开机自启。
  - 原因：快捷方式已存在且可用；环境文档写了“修改 /root/桌面 明确禁止”，未获新授权。
- **没有跑 live Gazebo 采集**（`collect.py` 需要已启动的 `gz sim` + DISPLAY=:20）；审核 agent 也没跑完整 episode。
  - 原因：本次监控任务不需要；且 collector 正在被并行会话修改。
- **没有验证 NFS/磁盘满/1M 事件 RSS/Windows/macOS/长时间小时级运行**。
  - 原因：超出当前范围；交接文档已列风险。
- **没有安装 PyTorch/cuDNN/TensorRT/ROS 2/CUDA 13**，没有改驱动/内核。
  - 原因：项目硬约束（sm_75 + CUDA 12.8）。
- **没有删除任何用户数据/历史文档**；只删了 `ui/__pycache__` 和 `/tmp` 本次临时文件。
  - 原因：遵循“不删用户没让删的东西”。

## 待确认

- 用户说的“监控系统改成本地软件”是否也包括 TinyAerialVLN 的 NavLab Studio web 前端？本次只做了 `uav_rgb/ui`。
- 是否要终止旧 web 进程 PID 133198（8799）？需要用户授权/执行；不影响 Tk 监控台。
- 是否要加 stdout/stderr 页签、事件过滤/导出、多 run 历史、run-dir 记忆？
- 是否要现在修核心/仿真 P1（尤其 teacher 全无效深度回 `SAFE_NEUTRAL`、ActionAdapter NaN 回 `SAFE_NEUTRAL`、collector 动作停止/时间对齐）？其中 collector 需等并行会话停止修改后再动。
- 真实 trainer 的 CSV/JSONL 写入是否保证 UTF-8、每行换行、追加/原子？若不保证，需在校验层再加保护。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `DISPLAY=:20 python3 ui/test_monitor.py` | `test_monitor: 50 项通过`，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | `37 项通过，1 项跳过`，EXIT=0 |
| `DISPLAY=:20 /usr/bin/python3 ui/test_monitor.py` | 50 项通过，EXIT=0 |
| live GUI：启动/追加/暂停/继续 | 4→5 events、8→9 rows；暂停不增；继续到 6；`LIVE_GUI_OK` |
| 畸形输入 6 s 压力（空 epoch/非 UTF-8/非 object/代理字符） | EXIT=124（仍在运行），stderr 空 |
| `launch_ui.sh`（XFCE PATH，`timeout 6`） | EXIT=124（6 s 时仍在运行），stderr 空 |
| `--refresh 0` / `--refresh nan` / `--refresh -1` | argparse error，EXIT=2 |
| 200k 行/14.6 MiB 基准 | 旧全量 1.010–1.042 s；新首次 0.066–0.068 s；无新数据 0.03–0.15 ms；+1 事件 0.053 ms |
| `ctest --test-dir src/build --output-on-failure` | **7/7 通过**（会话开始 6/6，并行会话新增 test_nn_cuda） |
| `experiments/cuda_smoke/build/cuda_smoke` | `RESULT: PASS`，max abs error 0，仅 sm_75 |
| `python3 sim/tools/teacher.py` | 6/6 通过 |
| `gz sdf -p sim/worlds/uav_rgb.sdf` | EXIT=0 |
| `python3 -m py_compile`（ui + sim tools） | OK |
| GUI 截图 | `ui/evidence/monitor_demo.png`，980×660 |
| 旧 web 进程 | PID 133198 仍在 `127.0.0.1:8799`；未获授权终止 |

完整命令/输出见 `docs/results.md`。

## 风险与遗留

- 旧 web 进程 133198 仍监听 8799；需用户 `kill 133198`。这是本次唯一未清理的“非本地软件”残留。
- 并行会话正在改 `sim/tools/collect.py` 和 `src/tests`；本会话的审核行号基于 23:26 版本，交接文档已注明。`ctest` 数量从 6 变 7 是并行改动，不是本次监控改动。
- 监控台未接真实 trainer；首次打开只读末尾 1 MiB，可能少显示旧事件；表格每次重建，epoch 数极大时可能卡顿；事件只保留 200 条；没有 stdout/stderr、导出、多 run。
- `.desktop` 的 `Exec` 写死当前仓库路径；迁移后需同步改。默认打开 `ui/demo`，真实 run 需选目录/传 `--run-dir`。
- 核心/仿真审核的 P1（teacher 全无效深度回前进、ActionAdapter NaN 回前进、collector 动作不停止/时间错位等）**尚未修**；在修复并用一个短 Gazebo episode 复测前，不应把采集数据当正式训练集。
- 控制台/桌面启动时 stderr 不可见；监控台已把刷新异常写状态栏，但首次启动前的 Tk 初始化失败仍需从终端启动排查。
- `ui/__pycache__` 会在再次运行测试/程序后重新生成，可随时删除，不是源码。

## 更新 2026-10-29 00:25（审核修复）

### 做了什么

- `ui/monitor.py`：
  - `tail_events` 增加首 64 字节指纹（前缀比较），识别同 inode 的原地重写（包括写回更大文件）；缺失 `metrics.jsonl` 返回 `reset=True` 清空旧事件；`limit<=0` 返回空。
  - 单 epoch / 恒定值曲线只画一个真实 x 刻度，y 轴对称留白，不再伪造“1、2”。
  - `_set_table` 按 epoch 值恢复选中行（Tk 滚动位置本身保留）。
  - `_safe_text` 覆盖顶部路径、窗口标题、`_open_dir` 状态栏；`_set_run_dir` 先复位 tail/events/rows/sig 再碰 Tk；`_is_demo` 对 `realpath` 的 `OSError/UnicodeError` 容错。
- `ui/test_monitor.py`：50→**60** 项；新增 tail 写回更大文件（316 B→594 B）、缺失文件清空、`limit=0`、单点刻度、表格选中行、`_redraw_charts` 真实调用计数（替换原先“有 canvas item 就算重画”的弱断言）、签名未变不读文件 / 变化后重读、代理字符路径。无显示 42+1 跳过。
- `ui/launch_ui.sh`：无 `DISPLAY` 时提示并退出 1，避免桌面 `Terminal=false` 静默失败。
- `docs/监控系统交接.md`：更新 §3.2 日志重写规则、§4 表格/刻度/路径/启动行为、§5 验证数字、§6 本节修复清单。
- `docs/results.md`：追加 2026-10-29 00:25 的真实测试输出。

### 没做什么

- **没有改 `sim/` / `src/`**：仿真采集的修复是另一条工作，见 `docs/worklog/2026-10-28-2358-fix-sim-causality.md` 的 2026-10-29 更新；`teacher.py` 的 `turn_dist/emergency` 是并行会话改的。
  - 原因：本任务是监控台；并行会话正在改 sim，避免冲突。
- **没有重跑 200k 行性能基准**：首 64 字节指纹每次多一次 64 B 读，量级远小于实测 0.03–0.15 ms；没有重新造 14.6 MiB 日志。
  - 原因：改动只影响 tail 的头部读取；原基准记录在 `docs/results.md`。
- **没有更新 `ui/evidence/monitor_demo.png`**：demo 8 个 epoch，单点刻度/选中行/代理路径在 demo 上看不出视觉差异；窗口布局未变。
  - 原因：截图仍代表当前界面；如需新截图可再抓。
- **没有修改 `/root/桌面/uavrgb-monitor.desktop`**：其 `Exec` 仍写死仓库绝对路径；本次只改了启动脚本里的 DISPLAY 检查。
  - 原因：桌面文件属用户环境，未获授权改。
- **没有做** NavLab Studio web 前端转换、事件导出/多 run 历史、stdout/stderr 面板、Windows/macOS、NFS/磁盘满、小时级真实 trainer。
  - 原因：超出本轮审核修复范围。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `DISPLAY=:20 python3 ui/test_monitor.py` | `test_monitor: 60 项通过`，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | `42 项通过，1 项跳过`，EXIT=0 |
| `DISPLAY=:20 /usr/bin/python3 ui/test_monitor.py` | 60 项通过，EXIT=0（Tk 8.6） |
| `env -u DISPLAY ./ui/launch_ui.sh` | 提示无 DISPLAY，EXIT=1 |
| `DISPLAY=:20 timeout 5 ./ui/launch_ui.sh --run-dir demo` | EXIT=124（仍在运行） |
| `python3 -m py_compile ui/monitor.py ui/test_monitor.py` / `bash -n` | EXIT=0 |

### 风险与遗留

- 首 64 字节指纹只能抓“头部变了”的原地重写；若新文件前 64 字节与旧文件相同（例如只改了尾部），仍会按旧 offset 读。代码有 `shortcut:` 注释，彻底解决需全文件 hash 或要求原子替换。
- 表格选中行按 epoch 字符串恢复；重复 epoch 时只恢复第一个匹配行。
- `launch_ui.sh` 只检查 `DISPLAY` 非空；无效 DISPLAY（权限/X 转发失败）仍会在 `tk.Tk()` 处报 TclError。
- 旧的 web 监控进程 PID 133198（8799）仍是环境残留，未获授权终止，与 Tk 监控台无关。

## 更新 2026-10-29 00:50（本地训练控制台）

### 做了什么

- `ui/monitor.py` 在同一个 Tk 窗口里加“训练控制”：
  - 运行目录、训练命令、epochs / steps_per_epoch / batch_size / lr / seed；开始/停止；DEMO 徽标；Notebook「指标事件 / 训练输出」。
  - 校验参数/命令/空目录；原子写 `train_config.json` + `train_command.txt`；`Popen` 用 argv 列表（**无 shell**），`start_new_session=True`，stdout/stderr 进 `train.log`。
  - 状态机 idle/starting/running/stopping/exited；停止 SIGINT→SIGTERM→SIGKILL；关闭窗口先收子进程再 destroy；运行期 1 s 轮询，空闲恢复 `--refresh`。
  - `train.log` 增量 tail（每 tick 64 KiB、视图 2000 行、文件 8 MiB 轮转）；显示前 `_safe_text`；不阻塞 Tk。
- `ui/trainer_demo.py` + `ui/test_trainer_demo.py`（subagent 实现）：纯 stdlib DEMO trainer，`--config`，写 results.csv/metrics.jsonl，确定性模拟曲线；SIGINT 退 130；89 项自检。
- `ui/test_monitor.py` 60→**85**：训练纯函数/配置往返/日志 tail/轮转 + GUI 启动/停止/崩溃/页签/DEMO 标识（临时假 trainer）。
- 截图 `ui/evidence/monitor_control.png`（980×660，真实窗口，DEMO 运行中）。
- `docs/监控系统交接.md`、`docs/results.md`、`CLAUDE.md` 同步。

### 没做什么

- **没有真实 C++/CUDA trainer**：`ui/trainer_demo.py` 是**模拟**，曲线不代表任何模型效果；原生 trainer 必须实现同一个 `--config` 合同（`run_dir/epochs/steps_per_epoch/batch_size/lr/seed` + results.csv/metrics.jsonl），本轮没实现。
  - 原因：仓库只有算子/模型代码，没有训练循环；先把本地控制层和合同做出来。
- **没有 resume/checkpoint/device/model/optimizer/dataset/多 run 对比/实验管理/远程/打包**。
  - 原因：第一版只解决“改超参数→启动→实时看→停止”的最小闭环；合同 agent 的完整设计留作下一轮。
- **没有把训练命令经过 shell**：`source env.sh && ...` 不能用，需要包装脚本；run 目录必须为空/新目录（防覆盖旧 run）。
- **没有验证原生 trainer + GPU 的吞吐/显存/长跑**，也没测 Windows/macOS（`start_new_session`/`killpg` 是 POSIX）。
- **没有改 `ui/launch_ui.sh` 和桌面 `.desktop`**：控制面板在同一个 Tk 窗口里，启动方式不变。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `DISPLAY=:20 python3 ui/test_monitor.py` | **85 项通过**，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | 56 项通过，1 项跳过，EXIT=0 |
| `python3 ui/test_trainer_demo.py` | 89 项通过，EXIT=0 |
| live DEMO 3 epoch（控制台内启动） | `rows=3 events=35 demo_lines=11 rc=0`，状态“训练完成 rc=0” |
| live DEMO 10 epoch | `rows=10 events=112 demo_lines=32 rc=0`，run 目录含 config/command/log/CSV/JSONL |
| 假 trainer 启动/停止/崩溃 | running→停止 `已停止 rc=0`；崩溃 tab 自动切“训练输出”，rc=3 |
| 截图 | `ui/evidence/monitor_control.png`（980×660） |

### 风险与遗留

- DEMO 是模拟训练，**不能当作真实训练结果**；原生 trainer 尚未接入。
- 关闭窗口最多阻塞约 6 s（SIGINT 3 s / SIGTERM 2 s / SIGKILL 1 s）；trainer 忽略信号才会走到 SIGKILL。
- `logs/runs` 会持续增长，没有保留/清理策略；`train.log` 超过 64 KiB/s 时界面显示会落后于文件（文件本身完整）。
- 训练命令不做 shell 展开；需要环境变量的场景用包装脚本。
- run 目录非空一律拒绝；续训/覆盖不支持。
