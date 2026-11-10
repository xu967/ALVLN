# baselines 报告：开源对比算法的选型与接入设计

- 子任务名：`baselines`；时间盒 60 分钟，开始 2026-11-10 19:21:42 CST。
- 草稿目录：`/root/rivermind-data/uav_rgb/tmp/phase12-20261110/baselines/`；本报告由 `build_report.py` 把 `sections/*.md`、包装器源码、自测日志拼接而成。
- 约束遵守：没有改仓库任何文件（`R=/root/claude-code-backup-20261028/code/uav_rgb` 只读 import，`PYTHONDONTWRITEBYTECODE=1`）；没有碰 GPU 0；没有往现有 venv 装包（新建了两个数据盘 venv，见 §4.3）；新增占用约 437 MB。

## 0. 结论速览

**短名单（要跑进闭环的基线，按优先级）**

| 序 | 基线 | 理由 |
|---|---|---|
| 1 | PULP-Dronet v3 家族（320,928 参数与 2,784 参数两个尺寸），零样本 + 同架构重训 | 与我们 0.1 M 级学生同量级的公开机载网络；重训臂能隔离"监督方式与输出头"的贡献 |
| 2 | Lamberti 2023 三条带碰撞头框架（复现，仅重训） | "按区域输出碰撞概率 + 固定规则"的最近先例，与我们的 20 列 × 4 档做分解粒度消融，骨干与数据可以完全相同 |
| 3 | Depth-Anything-V2 Small 米制（室内/室外）+ 现有 `expert_action` 条带规则；加真值深度上界 | 24.8 M 的几何基础模型加同一规则，回答"几何从 RGB 里估出来能到哪" |
| 4 | DroNet 原版（零样本，加阈值重标定对照臂） | 学界默认要对比的经典基线，已就绪 |
| 5 | UniDepthV2 Small + 已知内参 + 同规则（条件项） | 只在第 3 项显示深度路线有竞争力时才做；需要新 venv，体积可能超预算 |

大型 VLN/VLA（NaVid、Uni-NaVid、NaVILA、StreamVLN、OpenFly-Agent、OpenVLA-UAV 等）只引用不跑：权重 15 至 17 GB，数据盘剩 15 GB（`df -h`）装不下；DiffPhysDrone、NavRL、YOPO、EGO-Planner 等依赖 ROS/Isaac Sim/Unity 或没有公开权重。

**已下载并跑通前向的基线**（真实 `StudentPolicy` 端到端，3 张闭环帧，CPU，原始日志见附录 C）

| 基线 | steer / p_coll（Gazebo f05，Gazebo f12，3DGS） | 映射后动作 | 单步 CPU 时延 |
|---|---|---|---|
| PULP Tiny v3 | −0.0059 / 0.657，0.0886 / 0.792，0.0168 / 0.836 | 1，0，0 | 4.54–5.18 ms |
| PULP v3 ResBlock-1.0 | 0.0246 / 0.115，0.1409 / 0.987，−0.0737 / 0.033 | 1，2，1 | 7.93–9.62 ms |
| DroNet | 0.6421 / 1.000，0.0551 / 0.999，0.0260 / 0.896 | 2，0，0 | 7.20–7.86 ms |
| TrailNet | 6 维 softmax 输出（附录 C） | 未设映射 | 19.4 / 22.6 / 98.1 ms |

**可用的 GitHub 镜像**：`https://ghfast.top/`（8 s 内下到 9.2 MB）、`https://gh-proxy.com/`（8 s 内 2.3 MB）、`https://ghproxy.net/`（慢）；直连小请求通、大文件不稳定。HF 用 `hf-mirror.com`，ModelScope API 可达。详见 §4.1。

**每个基线的预算**（"估计"表示没实测）

| 基线 | 新增磁盘 | 显存 | 单步时延 |
|---|---|---|---|
| B1 PULP v3 / Tiny | 已下 2.8 MB；重训检查点各约 1.3 MB 与 24 kB | 0（CPU） | 实测见上表 |
| B2 三条带 | 同 B1 | 0（CPU） | 骨干同 v3，未测 |
| B3 DA-V2 Small 两个米制权重 | 约 245 MB（权重 198.4 MB + 代码约 46 MB） | 估计小于 1 GB | 未测 |
| B4 DroNet | 已下 3.8 MB | 0（CPU） | 实测见上表 |
| B5 UniDepthV2 Small | 权重 136.8 MB + 新 venv（体积未测，可能超 3 GB） | 估计小于 2 GB | 未测 |

**需要用户决定的事**（详见 §6）：基线能否用 PyTorch 独立 venv 或借用 teacher-venv 的 torch；重训臂所需的距离/深度标签是否存在；要不要一个大模型数据点（需要 ≥ 25 GB 新存储与下载批准）；包装器是否并入仓库。

**两个需要先知道的风险**：零样本 DroNet 在三张闭环帧上碰撞概率都高于 0.7，大概率原地不动；映射对门槛敏感（PULP v3 在 f12 上折算 12.68°，只比 12.5° 门槛高 0.18°），敏感性扫描不能省。

报告路径：`/root/rivermind-data/uav_rgb/tmp/phase12-20261110/reports/baselines.md`。

---

## 1. 接入协议（读代码所得，只读未改仓库）

仓库根 `R=/root/claude-code-backup-20261028/code/uav_rgb`。下列行号均来自本次 `grep -n` / `sed -n` 的实际输出。

### 1.1 调用链与文件

| 位置 | 作用 |
|---|---|
| `R/sim/tools/student.py` L42 `StudentPolicy.__init__(ckpt, infer_bin=None, text_ids=None, timeout_s=30.0)` | 启动推理子进程并做 hello 校验、预热；`act()` 是每步入口 |
| `R/src/infer_main.cpp`、`R/src/infer.cpp` | C++ 端：解析 `--ckpt` / `--text`，打印 hello，读 payload，打印回复行 |
| `R/sim/tools/collect.py` L926 `elif policy == "student"`（单步取动作的分支），L1335–1342 参数 `--policy`、`--student-ckpt`、`--infer-bin`，L1623–1628 构造 `StudentPolicy(args.student_ckpt, args.infer_bin, text_ids=text_ids)` | Gazebo 单 episode 采集与评测 |
| `R/sim/tools/eval_closed_loop.py` L1034 `--infer-bin` 参数；L761 转给 `collect.py`；L777–778 `env = cp.gz_env(partition)` 与 `subprocess.Popen(cmd, env=env, ...)`；L1162 探针 `stu.StudentPolicy(args.ckpt, infer_bin)`；L1301 报告字段 `infer_bin_sha256` | 20 个冻结 seed 的闭环评测入口，每个 seed 起一个 `collect.py` 子进程 |
| `R/sim/3dgs/collect_3dgs.py` L145–215（`ForwardPolicy` / `StudentAdapter` / `DaggerPolicy`），`R/sim/3dgs/eval_3dgs.py` L160–265 | 3DGS 实景线；eval_3dgs 里同一个 StudentPolicy 复用于所有 episode |
| `R/sim/3dgs/renderer.py` | 相机 320×240，fov 60（读自代码，未改） |

行号说明：`collect.py` 与 `eval_closed_loop.py` 在 2026-11-10 19:40:48 被并行任务的合并提交 `155e377`（航向护栏）改动（`ls -l --time-style=full-iso` 与 `git log -1`），这两个文件的行号是 19:56 重新 grep 的结果，我早先读到的行号已作废。`student.py`、`infer.cpp`、`teacher.py`、两个 3DGS 脚本的修改时间都早于本任务开始（2026-11-07 至 11-09），其行号未变。

环境变量沿 `collect_parallel.gz_env`（`env = dict(os.environ)`，`R/sim/tools/collect_parallel.py` L115–116）一路继承到 `collect.py` 子进程；`student.py` L59 的 `Popen` 同一行没有 `env=` 参数。我在 `proto_selftest.py` 里用 `os.environ["BASELINE"]` 传参，包装器确实收到，说明 `StudentPolicy` 这一层继承环境。`eval_closed_loop.py` 到 `collect.py` 再到 `StudentPolicy` 的整条链没有端到端跑过（没做，原因：本任务不起 Gazebo 闭环，见「没做什么」）。保险做法是每个基线一个一行启动器，把环境变量写死在启动器里（附录 B）。

### 1.2 线协议（一次进程生命周期）

1. 调用方 `Popen([infer_bin, "--ckpt", ckpt] + (["--text", ids] 若文本条件))`，stdin/stdout 为管道。
2. 子进程在 stdout 打印一行 hello JSON：`{"in_h":..,"in_w":..,"in_c":..,"num_classes":4,"text_dim":..,"prev_action_dim":..,"aux_classes":..}`。Python 端校验（`student.py`）：`in_c` 只能是 3 或 6（L72–74）；`prev_action_dim` 只能是 0 或 4（L76–78）；`num_classes` 必须是 4（L82）；`aux_classes` 只记录（L79）；`text_dim` 读入（L87）。
3. Python 端发一个全零 payload 做预热（`_warmup`，L91–97，长度 = `in_h*in_w*in_c`，prev-action 模型再加 1 字节），并读一行回复。
4. 每个决策步（1 Hz）：写 payload，读一行 JSON 回复，以 `\n` 结尾。
5. 结束：Python 关 stdin，子进程读到 EOF 后必须以退出码 0 退出；`close()` 对非零退出码抛 `RuntimeError`（L181–199）。

payload 字节布局（`student.py` L106–128 与 `R/src/infer.cpp`）：

| 段 | 长度 | 内容 |
|---|---|---|
| 当前帧 | `in_h*in_w*3` | HWC，uint8，RGB 顺序 |
| 上一帧 | `in_h*in_w*3` | 仅 `in_c==6` 时有；`reset()` 后第一步用当前帧代替（L119–120） |
| 上一动作 | 1 | 仅 `prev_action_dim==4` 时有；值域 0..3，越界抛错（L121–125） |

回复行字段：`action`（int，0..3）、`probs`（长度 4 的数组）、`infer_ms`（float）。`act()` 只读这三个键并返回 `(action, infer_ms, probs)`（L128–133），其余键被忽略，所以包装器可以多带 `raw` 字段做事后对账。

### 1.3 图像尺寸、通道、归一化发生在哪里

| 环节 | 位置 | 具体行为 |
|---|---|---|
| 解码 | `student.py` L110–116 | `save_frame.decode(msg)` 取 gz.msgs.Image；通道数只接受 3 或 4；RGBA 转 RGB |
| 缩放 | `student.py` L29–30、L117 | Python 端 PIL `BILINEAR`，从 320×240 缩到 hello 里的 `in_w×in_h` |
| 通道顺序 | 同上 | RGB，HWC，uint8 |
| 归一化 | `R/src/infer.cpp`（读自代码） | 设备端 HWC→CHW 并 `/255`；没有 mean/std |
| 外部基线的绕法 | 本文 §2 | hello 里声明 `in_h=240, in_w=320, in_c=3`，Python 端等于不缩放；灰度化、裁剪、归一化全在包装器里按各基线原文做 |

### 1.4 上一动作与文本条件

- 上一动作：只有 `prev_action_dim==4` 的 ckpt 才收 1 字节。`prev_action` 参数是"上一步真正执行的动作"（采集脚本有安全接管时与模型输出不同，L106–110 的文档串）；传 `None` 时用模型自己上一步的输出。外部基线 `prev_action_dim=0`，该字节不存在。
- 文本条件：`--text` 传 token id 串，词表 22、长度 9（`R/src/include/uavrgb/text.hpp` 的 `kTextLen=9`、`kTextVocab=22`；分词在 `R/sim/tools/instruction.py` 的 `tokenize`）。外部基线 `text_dim=0`，不使用。要接收自然语言指令的大模型基线（NaVILA、OpenFly-Agent 等）需要别的通道（环境变量或旁路文件）传指令，协议里没有字符串字段。

### 1.5 超时与错误处理

| 情形 | 行为（`student.py`） |
|---|---|
| hello / 预热 / act 任一次交换超过 `timeout_s=30` 秒 | `_abort_start` 杀进程并抛错（L136–145） |
| 回复超过 65536 字节 | 抛错（L156） |
| 子进程在回复中途关闭 stdout | 抛错并附带 stderr（L152–154） |
| 子进程退出码非 0（关闭时） | `close()` 抛 `RuntimeError`（L198–199） |
| `--ckpt` 文件不存在、`--infer-bin` 不是可执行文件 | 调用方在启动前拒绝；`--infer-bin` 要求带 shebang 的可执行文件。`eval_closed_loop.py` 把二进制的 sha256 写进报告（`infer_bin_sha256`），包装器的哈希因此可追溯 |

两条后果：

1. 大模型基线的加载时间要挪到调用方之外：30 秒只管 hello 和每一步，包装器的 hello 必须在 30 秒内打印，真正的加载放到常驻服务端（§2.4）。
2. 没有 reset 报文。`eval_closed_loop.py` 每个 episode 起新的 `collect.py` 进程，因而新建 `StudentPolicy`；`eval_3dgs.py` 复用一个 `StudentPolicy`，`reset()` 只清 Python 端缓存，子进程收不到 episode 边界。无状态基线（第一、二档全部）不受影响；带历史记忆的视频类 VLM 在 3DGS 线上需要改 `eval_3dgs.py`（每 episode 新建进程）或扩协议，本任务不改仓库，列为待决事项。

## 2. 外部基线适配器设计

### 2.1 总体做法

不改仓库，写一个可执行的包装器 `baseline_infer.py`（带 shebang，草稿目录里已写好并测过，全文见附录 A）。三条评测入口都只认 `--infer-bin <可执行文件>` 和 `--ckpt <任意存在的文件>`，所以：

```bash
# 例：PULP-Dronet v3 零样本，Gazebo 柱阵
export CUDA_VISIBLE_DEVICES=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70 PYTHONDONTWRITEBYTECODE=1
python sim/tools/eval_closed_loop.py --task base --policy student \
    --ckpt /root/rivermind-data/uav_rgb/tmp/phase12-20261110/baselines/stub.ckpt \
    --infer-bin /root/rivermind-data/uav_rgb/tmp/phase12-20261110/baselines/launch_pulp_v3.sh  ...(其余参数与学生评测相同)
```

（`eval_closed_loop.py` 的其余参数沿用学生评测命令行，本报告不重复；我没有跑这条命令，原因见「没做什么」。）

包装器的约定：

- hello 固定声明 `in_h=240, in_w=320, in_c=3, num_classes=4, text_dim=0, prev_action_dim=0, aux_classes=0`。Python 端不缩放，基线自己做灰度、裁剪、缩放，保证"每个基线按自己原文的输入处理"。
- stdout 只写协议行：启动时 `os.dup(1)` 留一份真 stdout，再 `os.dup2(2, 1)` 把第三方库的 print 引到 stderr。
- 回复 `probs` 是 one-hot（基线没有四类概率），多带的 `raw` 字段存基线原始输出，`BASELINE_LOG=<jsonl路径>` 时每步追加一行，供事后核对映射。
- 读到 stdin EOF 即退出码 0；半截帧退出码 1，调用方会抛错。
- 每个基线一个启动器（一行 `exec <该基线的解释器> baseline_infer.py "$@"`），环境变量写在启动器里。TrailNet 需要 OpenCV 4.x 的 Caffe 导入器，解释器与别的基线不同，启动器就是为这种情况准备的（附录 B）。

### 2.2 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `BASELINE` | `pulp_tiny` | `forward` / `pulp_v3` / `pulp_tiny` / `dronet` / `trailnet` / `da2_rule` |
| `BASELINE_ROOT` | `/root/rivermind-data/uav_rgb/baselines` | 第三方代码与权重所在目录 |
| `UAV_RGB_REPO` | 仓库只读路径 | 深度规则基线只读 import `teacher.expert_action` |
| `BASELINE_DEVICE` | `cpu` | 第一档 <1.5 MB 的网络在 CPU 上毫秒级；第二档设 `cuda`，由调用方用 `CUDA_VISIBLE_DEVICES` 钉卡 |
| `BASELINE_PREP` | `native` | 灰度 200×200 的取法：`native`（各自原文的裁法）/ `resize_crop` / `squash` |
| `BASELINE_WEIGHTS` | 空 | 同架构"在我们数据上重训"臂的 `.pth` 路径，空则用官方公开权重 |
| `YAW_STEP_DEG` | 25 | 一个 YAW 动作的名义转角，公共任务书写 23–27°，取中值 |
| `HOLD_S` | 1.0 | 动作保持时间，公共任务书 |
| `STEER_SIGN` | +1 | +1 表示 steer>0 为左转（PULP-Dronet README：YawRate>0 为逆时针即左）；DroNet 符号未核实 |
| `P_CRIT` | 0.7 | 碰撞概率刹停阈值 |
| `YAW_SCALE_PULP` / `YAW_SCALE_DRONET` | 90 / 20 | steer=1 对应的偏航角速度 °/s |
| `DEPTH_TURN_DIST` / `DEPTH_EMERGENCY` | 1.5 / 0.8 | 深度规则阈值，沿用 `teacher.expert_action` 默认值 |
| `BASELINE_LOG` | 空 | 非空则每步追加一行原始输出 JSONL |
| `DA2_WEIGHTS` / `DA2_MAX_DEPTH` | 无 | `da2_rule` 专用：Depth-Anything-V2 权重路径与最大深度（室内 20，室外 80） |

### 2.3 输出到 4 个动作的映射规则与阈值依据

我们的动作：0 SAFE_NEUTRAL，1 FORWARD_SLOW（约 0.25 m），2 YAW_LEFT（约 25°），3 YAW_RIGHT；每个动作保持 1.0 s（公共任务书）。映射的总原则是"最近原语量化"：把基线想做的一秒内位移/转角，量化到最近的原语，中点就是阈值。

| 基线输出类型 | 规则 | 阈值与依据（出处） |
|---|---|---|
| 转向值 steer∈[-1,1] + 碰撞概率 p（PULP-Dronet、DroNet、TrailNet 的转向头） | `yaw_cmd = STEER_SIGN × steer × S × 1.0 s`；`abs(yaw_cmd) ≥ 12.5°` 则按符号 YAW_LEFT / YAW_RIGHT（转向优先于前进）；否则 `p ≤ 0.7` 取 FORWARD_SLOW，`p > 0.7` 取 SAFE_NEUTRAL | 12.5° 是 25° 一步转角的一半，即"转 0 步还是转 1 步"的中点。S：PULP 取 90 °/s，出自 `pulp-dronet/src_ctrl/config_dronet.h` L45 `YAW_SCALING 90.0f`（另两组配置 L67、L89 为 120）；DroNet 取 20 °/s，出自 `dronet/src_ctrl/outdoor.yaml` L31 `SpeedSettingsMaxRotationSpeedCurrent: 20.0`。p 阈值 0.7：`config_dronet.h` L55/L77/L99 `CRITICAL_PROB_COLL 0.7f`，`deep_navigation.cpp` L117 `critical_prob 0.7`（刹停条件 L58 作用在低通滤波后的速度上，不滤波时等价于 p>0.7）。DroNet 控制节点把转向值限幅到 [−1,1] 后直接当角速度指令（L65、L99–100），Bebop 驱动把它按最大旋转速度缩放，这是我的理解，驱动文档未核实。S=20 °/s 时 steer 要大于 0.625 才触发转向，S=90 °/s 时 steer 大于 0.139 即触发，这个差别在零样本臂里会直接影响结果，列入敏感性扫描（§5.3） |
| 三条带碰撞概率（Lamberti 2023 框架，复现后的网络） | 取概率最小的条带方向：中条最小则前进（中条概率 ≤T，否则 SAFE），左/右条最小则 YAW_LEFT / YAW_RIGHT；三条都大于 T 则连续发 YAW_LEFT（论文是原地转 180°±[0,30]° 的随机角，一个 25° 原语要连发约 7 步） | 规则取自论文正文 Baseline 策略（`arxiv/text_2312.08991.clean.txt` "flies towards the direction with the lowest collision probability... spins in place by 180°"）。T 的数值本次没读到，未核实；预注册取 0.5，在开发 seed 上扫 {0.3, 0.5, 0.7}，冻结 seed 只用选定值跑一次 |
| 连续速度（vx m/s、偏航角速度 °/s）：DiffPhysDrone、NavRL、YOPO 首段轨迹等 | `Δθ = yaw_rate × 1 s`；`abs(Δθ) ≥ 12.5°` 取 YAW；否则 `vx × 1 s ≥ 0.125 m` 取 FORWARD_SLOW，否则 SAFE | 0.125 m 是 0.25 m 一步前进量的一半；两个中点同一个道理。实现在包装器 `map_velocity()` |
| 中层动作（"前进 75 cm"、"左转 30°"、"stop"）：NaVILA、StreamVLN、Uni-NaVid 一类 | 动作队列：前进 d m 入 `round(d/0.25)` 个 FORWARD_SLOW；转 θ° 入 `round(θ/25)` 个 YAW；stop 入 SAFE。队列未空时不再请求模型；队列里任何一步触发碰撞即清空队列重新请求 | 0.25 m 和 25° 来自公共任务书；队列实现在包装器 `MidLevelQueue`。清空队列的规则是我的设计，没有出处，需要在公平性协议里写明 |
| 深度图（Depth-Anything-V2、UniDepth、Metric3D、ZoeDepth） | 复用仓库 `teacher.expert_action`：取中心条带（行 `[0.4h, 0.6h)`），`turn_dist=1.5 m`、`emergency=0.8 m`、`switch_margin=0.75`；深度无效时 fail-closed（取 SAFE） | 全部沿用 `R/sim/tools/teacher.py` 的现值（`BAND_LO, BAND_HI = 0.4, 0.6`），保证"换感知不换规则"。主线计划的 20 列 × 4 档可通行距离规则（`docs/reviews/2026-11-10/lit.md` D2）还没进主线，落地后用它替换，让基础模型与学生共用同一规则 |

不做的映射：不对基线输出做时间平滑（PULP/DroNet 原控制律里有的低通滤波，alpha 0.3–0.6，见 `config_dronet.h` L49–50、`deep_navigation.cpp` L114–115）。原因：我们 1 Hz 决策且动作保持 1 s，滤波常数是按原系统的帧率调的（PULP-Dronet 在 GAP8 上 19 帧/s，[abs 2407.12675]），在 1 Hz 下不可比，保留滤波会引入基线自己的相位滞后。这一条会让基线略吃亏或略占便宜，需要在论文里声明，未量化。

### 2.4 预处理与公平性注意

1. **视场与分辨率不匹配**：我们相机 320×240、水平 fov 60°（`R/sim/3dgs/renderer.py`）；Himax 相机在 Lamberti 2023 的仿真里按 87° 视场复现（`arxiv/text_2312.08991.clean.txt`，"emulates the 87° field of view of the drone's Himax camera"）。若 PULP 系列训练用的相机也是 87° 量级（PULP-Dronet v3 数据集的相机视场我没有核实），零样本臂看到的视场就窄于训练时，目标看起来更大更近；这是零样本比较的固有偏差，不能靠预处理消除。`BASELINE_PREP` 提供三种取法做敏感性：`native`（各基线原文的裁法）、`resize_crop`、`squash`。三种取法的输出差异只在 3 帧上跑过（§4.4 第 6 条的表，`prep_sensitivity.txt`），同一帧换取法动作会变；闭环敏感性没有跑（没做）。
2. **灰度**：PULP-Dronet、DroNet 吃 200×200 灰度；TrailNet 吃 BGR 彩色 320×180。包装器内部转换，hello 不声明灰度。
3. **符号约定**：`STEER_SIGN` 需要在开发 seed（不是 20 个冻结 seed）上校准一次。校准做法：取 3 个开发 seed，`STEER_SIGN` 取 +1 与 −1 各跑，选碰撞数少的，选定后写入报告并冻结。镜像帧检验（左右翻转输入，steer 应变号）我计划做但未做（§「没做什么」）。
4. **零样本与重训两个臂**：零样本臂回答"公开权重直接用能到哪"，重训臂回答"同样的网络结构给我们的数据和教师能到哪"，只有后者能隔离"网络结构 / 监督方式"的贡献。重训臂的权重通过 `BASELINE_WEIGHTS` 接入，已测通覆盖路径（§4.4）。

### 2.5 大模型基线（VLN/VLA）的瘦包装

包装器协议同上，但推理不在包装器进程内：

1. 常驻服务端（单独进程，`setsid nohup` 起，`flock` 防重复，pid/日志/DONE 标记齐全），加载权重一次，监听 unix socket。
2. 包装器启动即连 socket，hello 立刻打印（不等加载），满足 30 秒限制；每步把 320×240 帧发给服务端，等待回复，超时由服务端回 SAFE 并记录。
3. 指令文本：协议没有字符串字段，用环境变量 `BASELINE_INSTRUCTION` 或旁路文件传；`--text` 的 id 串可由包装器反查词表还原。
4. 状态：服务端按"包装器进程 pid"作为会话键，包装器进程退出即清历史；这满足 `eval_closed_loop.py` 每 episode 新进程的语义，但满足不了 `eval_3dgs.py` 复用进程的语义（§1.5）。
5. 显存与时延：见 §3.3；单卡 22 GB 只能放一个 7B 级模型的 fp16 推理（权重约 15 GB 量级）。

## 3. 候选算法核实（四档）

出处缩写（路径均在草稿目录 `/root/rivermind-data/uav_rgb/tmp/phase12-20261110/baselines/` 下）：

| 缩写 | 含义 |
|---|---|
| `[gh]` | `gh/summary.tsv`，GitHub API 取得的许可证、仓库体积（KB）、星数、最近推送；README 原文存 `gh/*.README.md` |
| `[hf]` | `hf/summary.txt`、`hf/summary_retry.txt`，hf-mirror.com 的 `/api/models/<id>` 与 `/tree/main` 返回的文件体积（MB 为 10⁶ 字节） |
| `[abs]` | arXiv 摘要页或 API 摘要原文，`arxiv/abs_*.html`、`arxiv/titles.jsonl`、`arxiv/titles2.json`；论文编号由标题检索核对过 |
| `[sel]` | 本机冒烟日志 `selftest_all.txt`、参数统计 `count_params.out` |

"未核实"表示本次没有读到第一手来源。闭环成绩只引用摘要或 README 里写明的数字，评测条件与我们的 `sim09.v2.coast1s.corridor3m` 不同，不能直接比较。

### 3.1 第一档：同量级机载小模型

| 名称 | 仓库、许可证 | 公开权重 | 输入 → 输出 | 参数、算力 | 依赖 | 论文或 README 报告的闭环结果 |
|---|---|---|---|---|---|---|
| DroNet（ResNet-8，Loquercio 2018） | https://github.com/uzh-rpg/rpg_public_dronet ，MIT，最近推送 2019-06-04 [gh] | `dronet_model.zip` 1,207,446 B（本机已下载，内含 `best_weights.h5`） | 200×200 灰度 → 转向（回归）+ 碰撞概率 | 321,634 个张量元素（含 BN 滑动统计）[sel]；与 PULP v3 ResBlock-1.0 同拓扑，按层表手算约 41 M MAC，未用工具核对 | Keras 权重，TF 1.5 时代；本机用 h5py 转 npz 后以 NumPy 前向（`venv-h5`） | 论文结果未核实（本次没取到 arXiv 原文）。控制律与常数见 `dronet/src_ctrl/deep_navigation.cpp`（v=(1−p)·vmax，p>0.7 刹停） |
| PULP-Dronet v2（Palossi 2019） | https://github.com/pulp-platform/pulp-dronet ，Apache-2.0，最近推送 2025-07-03 [gh] | `pulp-dronet-v2/model/dronet_v2_nemo_dory_original.pth` 1,300,270 B（本机已下载） | 灰度 → 转向 + 碰撞概率 | 参数与 MAC 未单独统计（没做，原因：v2 的模型文件依赖 NEMO/DORY 量化流程，FP 前向需另写桩） | PyTorch + NEMO 量化库（本机用桩替代 FP 前向所需的 `PACT_IntegerAdd`） | 摘要：在 GAP8 上 19 帧/s，权重 320 kB [abs 2407.12675] |
| PULP-Dronet v3，ResBlock depth 1.0 | 同上仓库 | `tiny-pulp-dronet-v3/model/pulp-dronet-v3-resblock-1.0.pth` 1,298,645 B（已下载） | 200×200 灰度 → 转向∈[−1,1] + 碰撞概率 | 320,928 参数，41,103,104 MAC（本机钩子实测，输入 200×200）[sel] | 同上 | 摘要：真实走廊含 180° 转弯，最大目标速度 0.5 m/s，PULP-Dronet 持续失败 [abs 2407.12675] |
| Tiny-PULP-Dronet v3，dw-pw depth 0.125 | 同上仓库 | `tiny-pulp-dronet-v3/model/tiny-pulp-dronet-v3-dw-pw-0.125.pth` 24,085 B（已下载） | 同上 | 2,784 参数，1,130,444 MAC [sel]；摘要写内存最多缩小 168 倍、降到 2.9 kB，GAP8 上最高 139 帧/s [abs 2407.12675]；另一篇 Tiny-PULP-Dronets 写参数少 50 倍、MAC 少 27 倍 [abs 2407.02405] | 同上 | 摘要：同一走廊场景 100% 成功率，0.5 m/s [abs 2407.12675]；同时开源 66k 张统一碰撞/转向数据集（链接未取） |
| TrailNet SResNet-18（NVIDIA Redtail） | https://github.com/NVIDIA-AI-IOT/redtail ，BSD-3-Clause，最近推送 2020-11-17 [gh] | `TrailNet_SResNet-18.caffemodel` 45,152,711 B + prototxt 21,761 B（本机已下载，`ls -l baselines/trailnet`） | 320×180 BGR 彩色 → 6 个输出，两组 softmax(3)，实测各组和为 1.000（朝向三类、横向偏移三类的语义按论文，本次未逐字核对） | 参数量未数；45 MB 按 fp32 估约 11 M 参数（估算，非实测） | Caffe 模型，需 OpenCV 4.x 的 `readNetFromCaffe`；teacher-venv 里的 OpenCV 5.0.0 没有该导入器（实测报错），新建 `venv-cv4`（264 MB） | 摘要：Jetson TX1 机载实时，自主飞行 1 km 林间小路 [abs 1705.02550]。任务是沿小路飞行，不是避障，只做冒烟，不入短名单 |
| Lamberti 等 2023 纳米无人机竞速框架（arXiv 2312.08991） | 正文只给补充视频（http://youtu.be/vHTAwUsj-nk），没有代码链接；GitHub 关键词搜索 `nano drone racing pulp-dronet IMAV` 返回 0 条（`gh/search_nanoracing.json`，单次搜索，证据弱）；是否存在未被正文提及的仓库：未核实 | 未发现 | 单帧灰度；把视野按竖向切三条（正文称每条 108×324 像素），每条输出碰撞概率 | 参数量未在正文读到；int8 后 317 kB，运行在 GAP8 [正文 `arxiv/text_2312.08991.clean.txt`] | 仿真数据（Bitcraze Webots，仿真相机 87° 视场）训练，41k 张，70/10/20 划分；标签：某条内属于障碍物的像素里 10% 的距离 ≤2 m 记为碰撞 | IMAV 2022 纳米机竞赛七支队伍第一，5 分钟内飞 115 m 未撞 [abs 2312.08991]。需要复现，不能零样本 |
| NanoFlowNet、2026 年的 PULP-Dronet 自动部署论文等其他纳米机网络 | NanoFlowNet [abs 2209.06918]（GAP8 上稠密光流，Crazyflie 避障）；2607.12593 摘要写 PULP-Dronet 内存缩小 2 倍、推理快 1.6 倍，自由飞最高 1.96 m/s（基线 0.5 m/s）[`arxiv/abs_2607.12593.html`]。仓库 `idsia-robotics/ultralow-power-monocular-depth-ondevice-learning`（Apache-2.0，2025-12-22 推送）[gh]；代码与权重是否可直接用、输入输出：未核实（README 未读） | 未核实 | 未核实 | 未核实 | 未核实 | 只列线索，没做 |

SG2002 的 int8 NPU 可行性：PULP 系列的算子是 3×3/1×1/深度可分离卷积、BN（可折叠）、ReLU、加法残差、全连接，都属通用算子；SG2002 工具链（TPU-MLIR）的算子清单与量化精度我没有查，结论为"未核实"。2080 Ti 与 CPU 时延：PULP v3 在 CPU 上每步 7.93–9.62 ms，Tiny 4.54–5.18 ms，DroNet 7.20–7.86 ms，TrailNet 19.41 与 22.56 ms（第三帧 98.13 ms 为离群值，原因未查）[sel 的 `selftest_all.txt`，每个基线 3 帧，线程数未控制，样本量太小只能当量级]，不需要 GPU。

### 3.2 第二档：几何基础模型 + 固定规则

| 名称 | 仓库、许可证 | 公开权重（体积） | 输入 → 输出 | 参数、延迟 | 依赖 | 闭环结果 |
|---|---|---|---|---|---|---|
| Depth-Anything-V2 Small（相对深度） | https://github.com/DepthAnything/Depth-Anything-V2 ，仓库 Apache-2.0 [gh]；Small 权重 Apache-2.0，Base/Large/Giant 为 CC-BY-NC-4.0（README L178–180） | `depth-anything/Depth-Anything-V2-Small` 的 `depth_anything_v2_vits.pth` 99.2 MB [hf] | RGB，边长取 14 的倍数 → 相对深度图（非米制） | 24.8 M（README L39）；2080 Ti 延迟没测（没下权重） | PyTorch，README 称装 `requirements.txt` | 无无人机闭环结果（论文是深度估计工作，摘要称比基于 Stable Diffusion 的模型快 10 倍以上 [abs 2406.09414]） |
| Depth-Anything-V2 Metric Small（室内 Hypersim / 室外 VKITTI2） | 同仓库 `metric_depth/` | `Depth-Anything-V2-Metric-Hypersim-Small` 与 `...-VKITTI-Small`，各 99.2 MB，Apache-2.0 [hf]；transformers 版 `Metric-Indoor-Small-hf`、`Metric-Outdoor-Small-hf` 各 99.2 MB，API 的 license 字段为空，卡片未读 | RGB → 米制深度；`max_depth` 室内 20、室外 80（`gh/DepthAnything__DA2-metric.README.md` L47） | 24.8 M；README 建议"先试更大的模型和室内版本"（L18） | 同上 | 无 |
| UniDepthV2 Small / Base | https://github.com/lpiccinelli-eth/UniDepth ，CC BY-NC 4.0（README L291–293），论文 [abs 2502.20110] | `lpiccinelli/unidepth-v2-vits14` 34,184,323 参数，单个权重文件 136.8 MB（仓库含 `pytorch_model.bin` 与 `model.safetensors` 两份，合计 273.6 MB）；`unidepth-v2-vitb14` 114,421,379 参数，457.7 MB [hf] | RGB（可选相机内参 3×3）→ 米制深度 + 预测内参 + 3D 点（README L115–154）；我们的相机内参已知（320×240，fov 60°），可直接喂入 | 延迟没测 | CUDA 11.8+，xformers，需与 torch 版本匹配（README L63–103），源码安装；比 DA-V2 难装 | 无 |
| Metric3D v2 | https://github.com/YvanYin/Metric3D ，BSD-2-Clause [gh]，论文 [abs 2404.15506] | `JUGGHM/Metric3D`：`metric_depth_vit_small_800k.pth` 150.1 MB，ConvNeXt-tiny 123.9 MB，ViT-large 1,648 MB，ViT-giant2 5,511 MB [hf]；ONNX 小模型 `onnx-community/metric3d-vit-small`，许可证 cc0-1.0，体积没取到（429） | RGB + 内参（规范焦距 1000，README L260–264）→ 米制深度与法向 | 延迟没测 | torch hub 入口 `metric3d_vit_small`（README L295–304）；其余依赖未核实 | 无 |
| ZoeDepth | https://github.com/isl-org/ZoeDepth ，MIT，GitHub 标记已归档，README 写 Intel 已停止维护（L3） | `Intel/zoedepth-nyu-kitti`，345,073,307 参数，1,380.4 MB [hf] | RGB → 米制深度 | 体积是 DA-V2 Small 的约 14 倍 | torch 1.10.1 时代（README 徽章） | 无；MonoNav 在用它 |
| Depth Pro | https://github.com/apple/ml-depth-pro ，GitHub 许可证 NOASSERTION，HF 许可证 apple-amlr [gh][hf] | `apple/DepthPro` 的 `depth_pro.pt` 1,904.4 MB；`DepthPro-hf` 951,991,330 参数 [hf] | RGB → 米制深度 + 焦距 | 对 22 GB 显存没压力，但单步延迟偏大，没测 | 未核实 | 无 |
| MonoNav 管线 | https://github.com/natesimon/MonoNav ，MIT，最近推送 2024-02-12 [gh] | 复用 ZoeDepth 权重（子模块） | RGB + **相机位姿** → TSDF 三维重建 → 运动基元规划（README L18–26，L79–81） | 未核实 | Open3D TSDF，GPU 可选（README L64） | 摘要：37 g 的 Crazyflie 在杂乱室内以 0.5 m/s 飞行 [abs 2311.14100]。需要位姿输入，与"仅 RGB 单帧"条件不同；本仿真能提供真值位姿，但那是特权信息，使用时必须在论文里写明 |

深度 → 动作的规则统一使用 §2.3 的 `teacher.expert_action` 条带规则，做到"只换感知，不换规则"。每个深度模型要跑两个版本：真值深度 + 规则（上界，隔离规则本身的能力）和估计深度 + 规则。

### 3.3 第三档：大型 VLN / VLA

| 名称 | 仓库、许可证 | 公开权重（体积） | 输入 → 输出 | 参数、延迟、显存 | 依赖 | 论文报告 |
|---|---|---|---|---|---|---|
| VLN-CE Seq2Seq / CMA（Krantz 2020） | https://github.com/jacobkrantz/VLN-CE ，MIT [gh]；论文 [abs 2004.02857] | 权重位置未核实 | RGB-D + 指令 → 低层动作 | 小模型，未核实 | Habitat-Sim 0.1.7 + Habitat-Lab 0.1.7（NaVid README 同样要求） | 未核实 |
| NaVid | https://github.com/jzhzhang/NaVid-VLN-CE ，MIT [gh]；权重仓库 `Jzzhang/NaVid`，CC-BY-NC-4.0，15,494.7 MB [hf] | 同左；另需 EVA-ViT-G 视觉编码器（体积未取） | 视频帧 + 指令 → 文字动作 | 7B 级；延迟、显存未核实 | Python 3.8，habitat-sim 0.1.7（README L34–36） | 摘要：510k 导航样本 + 763k 网络数据训练，仿真与真实均达到当时最好成绩 [abs 2402.15852] |
| Uni-NaVid | https://github.com/jzhzhang/Uni-NaVid ，MIT [gh] | `Jzzhang/Uni-NaVid`，整个仓库 16,224.2 MB（含数据）；两个权重分片 9,976.7 + 5,517.4 MB [hf] | 视频 + 指令 → 动作 | README 称在单张 A100 上约 5 Hz（使用在线 token 合并，README L110）；Vicuna-7B 底座（README L71） | flash-attn 2.5.9.post1（README L59），2080 Ti（Turing）不支持 FlashAttention-2，需改注意力后端，未验证 | 摘要：3.6M 导航样本 [abs 2412.06224]；README 表：VLN-CE R2R Val 的 NE 4.96、OS 57.4、SR 51.8、SPL 47.7，RxR Val 的 SR 56.1、SPL 44.5（`gh/jzhzhang__Uni-NaVid.README.md` L134–138；地面机器人室内导航，不是无人机） |
| NaVILA | https://github.com/AnjieCheng/NaVILA ，Apache-2.0 [gh] | `a8cheng/navila-llama3-8b-8f`，16,998.2 MB [hf] | 多帧 + 指令 → 中层动作文字（如"向前 75 cm"）；权重名带 `8f`，帧数未核实 | 8B 级，延迟未核实 | 评测环境装 flash-attn 2.5.8 预编译轮子（README L134），Turing 同样不支持 | 摘要：先输出带空间信息的中层动作，再交给底层策略 [abs 2412.04453]。**中层动作正是 §2.3 队列规则的用例** |
| StreamVLN | https://github.com/OpenRobotLab/StreamVLN （GitHub API 返回的 `full_name` 为 `InternRobotics/StreamVLN`，已改名；`gh/OpenRobotLab__StreamVLN.json`），许可证字段为空 [gh] | `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world`，8,030,345,248 参数，16,076.8 MB [hf] | 视频流 + 指令 → 动作 | 8B 级，延迟未核实 | Python 3.9，PyTorch 2.1.2，habitat-sim 0.2.4（README L51–58） | 摘要：慢快双上下文 [abs 2507.05240]；README L39：R2R 的 NE 4.90、OS 63.6、SR 56.4、SPL 50.2，RxR 的 SR 54.4、SPL 45.4（评测划分未写明，地面导航） |
| OpenFly-Agent-7B | 平台 https://github.com/SHAILAB-IPEC/OpenFly-Platform ，MIT [gh] | `IPEC-COMMUNITY/openfly-agent-7b`，MIT，7,541,237,184 参数，三个分片共 15,086.4 MB [hf] | 空中 VLN：图像 + 指令 → 空中动作（动作空间与图像帧数未核实） | 7.5B；延迟未核实 | Python 3.10；平台支持 UE、AirSim、3DGS、GTA V（README L24–95） | 摘要：100k 轨迹、18 个场景 [abs 2502.18041]；平台 README L276–292 的逐场景表：12 个场景的 SR 为 10.6% 到 43.8%，NE 为 66.9 到 178.9 m（表对应的模型与划分未写明，未核实） |
| UAV-Flow / OpenVLA-UAV | https://github.com/buaa-colalab/UAV-Flow （原名 PRIS-CV/UAV-Flow 返回 404，改名后核实），许可证字段为空；论文 [abs 2505.15725] | `wangxiangyu0814/OpenVLA-UAV`，7,541,237,184 参数，四个分片共 15,085.1 MB [hf] | 指令条件的 UAV 控制，Flask 推理服务（README L54） | 7.5B；延迟未核实 | Python 3.10，PyTorch CUDA 12.4；评测环境 UnrealZoo（README L60–82） | 摘要未写数字，没读正文 |
| IndoorUAV-Agent | 论文 [abs 2512.19024]；ModelScope `valyentine/IndoorUAV-Agent` | 只有一个 `29999.zip`，9,468,831,797 字节（`hf/modelscope_indooruav.json`） | 室内 VLN | 未核实 | 未核实 | 摘要：1,000 多个 Habitat 室内场景，16,000 多条轨迹 [abs 2512.19024] |
| See-Point-Fly（SPF） | 论文 [abs 2509.22653]；仓库未查 | 无需权重（调用外部 VLM 接口） | RGB + 指令 → 2D 航点 + 距离 → 3D 位移 | 取决于所用 VLM 接口 | 需要外网接口与密钥，本机没有 | 摘要：无需训练 [abs 2509.22653] |

本机能否跑 7–8B 模型：数据盘剩余约 15 GB（`df -h /root/rivermind-data`：295 GB 总量、280 GB 已用、15 GB 可用），而上表 7–8B 权重各 15–17 GB，装不下，且公共任务书限制单个任务新增占用不超过 3 GB、单次下载不超过 500 MB。fp16 权重本身约 15 GB，22 GB 显存放得下权重加少量激活，bf16 在 Turing 上没有原生支持，需要转 fp16。结论：第三档在本机只能引用，除非用户给出新的存储位置（§6）。

### 3.4 第四档：经典规划与 RL

共同点（来自各 README 的依赖与运行说明，逐项输入格式本次没有全部核实）：规划类需要 ROS 加深度/里程计，RL 类需要 Isaac Sim 或 Unity 一类仿真器，推理依赖状态估计与目标点，和我们的"仅 RGB 单帧"不是同条件。

| 名称 | 仓库、许可证 | 权重 | 运行条件 | 能否在无 ROS、无 GPU-GL 容器里跑 | 报告结果 |
|---|---|---|---|---|---|
| EGO-Planner | https://github.com/ZJU-FAST-Lab/ego-planner ，GPL-3.0 [gh]；论文 [abs 2008.08835] | 无（优化式规划器） | ROS1；深度 + 里程计 | 不能，只引用 | 摘要未取到数字 |
| EGO-Planner-v2（Swarm Playground） | https://github.com/ZJU-FAST-Lab/EGO-Planner-v2 ，GPL-3.0；README 仅 794 字节，指向教程 PDF [gh] | 无 | ROS | 不能，只引用 | 未核实 |
| Fast-Planner | https://github.com/HKUST-Aerial-Robotics/Fast-Planner ，GPL-3.0 [gh]；论文 [abs 1907.01531] | 无 | ROS | 不能，只引用 | 未核实 |
| YOPO | https://github.com/TJU-Aerial-Robotics/YOPO ，许可证 NOASSERTION，最近推送 2026-11-01 [gh] | 仓库内 `YOPO/saved/YOPO_1/epoch50.pth`（体积未核实） | Ubuntu 20.04 + ROS + CUDA；仿真器是自带 CUDA 随机环境（README L52–58） | 测试脚本走 ROS（README L119），不能直接跑；网络本身可能可单独调用，未核实。README：Orin NX 上 TensorRT 推理 1 ms（ResNet-14）到 5 ms（ResNet-18）（L182），RTX 3080 上 10 万样本训练 50 轮不到 1 小时（L167）；输入需要深度 + 状态 + 目标，与 RGB-only 不同 | 只引用 |
| NavRL | https://github.com/Zhefan-Xu/NavRL ，MIT [gh]；论文 [abs 2409.15634] | README 称提供预训练模型和纯 Python 演示脚本（L39–61） | 训练需 Isaac Sim 2023.1.0-hotfix.1（README L69）；部署 ROS1 Noetic / ROS2 Humble | README 称演示脚本为纯 Python（L39–61），输入形式（距离图或图像）本次未核实；训练依赖 Isaac Sim，只引用 | 摘要未取到数字 |
| DiffPhysDrone | https://github.com/HenryHuYu/DiffPhysDrone ，许可证字段为空；README 是训练代码，评测命令 `python eval.py --resume <ckpt> --target_speed 2.5`；论文 2407.10648（arXiv 标题 "Back to Newton's Laws..."，正式发表于 Nature Machine Intelligence 2025，README 的 BibTeX） | README 没列公开权重 | PyTorch 2.2.2、Python 3.11、CUDA 11.8，自带 CUDA 算子 `pip install -e src` | 纯 PyTorch + CUDA 算子，不需要 ROS 或 GL，理论上能在本机训练和评测，但要自己训练（没有权重），超出本任务预算 | 摘要：单机场景成功率 90%，此前最好方法 60%；真实林间最高 20 m/s [abs 2407.10648] |
| Agile Autonomy | https://github.com/uzh-rpg/agile_autonomy ，GPL-3.0 [gh]；论文 [abs 2110.05113] | README 称带预训练检查点（L100） | ROS Noetic + TensorFlow 2.4 + Flightmare Unity 独立可执行（README L34–95） | 需要 Unity 渲染，不能，只引用 | 摘要未取到数字 |
| MAVRL | https://github.com/tudelft/mavrl ，许可证字段为空 [gh]；论文 [abs 2402.08381] | 训练产物路径 `saved/RecurrentPPO_1/Policy/iter_*.pth`；VAE 权重在 4TU 数据站链接（README L87–92） | ROS Noetic + Unity（AvoidBench）（README L30–74） | 不能，只引用 | 摘要未取到数字 |

## 4. 本机可达性与冒烟

### 4.1 GitHub 镜像探测

探测脚本 `probe_mirrors.sh`，结果 `mirror_probe.tsv`（19:2x CST）。每个前缀各取一个 6,121 字节的 raw README 和一个 769 字节的 git 信息请求；格式为"状态码 字节数 耗时秒"。

| 前缀 | raw 文件 | git 信息请求 | 结论 |
|---|---|---|---|
| 直连 `https://github.com` | 200，6121 B，0.44 s | 200，769 B，1.55 s | 小请求通 |
| `https://ghfast.top/` | 200，6121 B，2.38 s | 200，769 B，1.00 s | 可用 |
| `https://gh-proxy.com/` | 200，6121 B，1.61 s | 200，769 B，1.01 s | 可用 |
| `https://ghproxy.net/` | 200，6121 B，5.11 s | 200，769 B，6.13 s | 可用但慢 |
| `https://github.xxlab.tech/` | 302，93 B，0.53 s | 200，769 B，1.07 s | raw 被重定向，未跟进，未核实 |
| `https://kkgithub.com/`、`https://ghproxy.cc/` | 证书已过期（curl 60） | 同左 | 不可用 |
| `https://gitclone.com/github.com/` | 502 | 502 | 不可用 |
| `https://mirror.ghproxy.com/`、`https://gh.llkk.cc/` | 连接超时 15 s | 同左 | 不可用 |
| `https://github.moeyy.xyz/`、`https://hub.gitmirror.com/` | 域名解析失败 | 同左 | 不可用 |
| `https://gh.idayer.com/` | 429 | 429 | 限流 |
| `https://gh.api.99988866.xyz/` | TLS 握手失败 | 同左 | 不可用 |

大文件速度（`speed_test.sh`，同一个 zip，各限时 8 s，`speed_test.log`，19:41）：直连 0 字节（HTTP 000）；`ghfast.top` 9,222,989 B（约 1.15 MB/s）；`gh-proxy.com` 2,284,629 B（约 286 kB/s）。19:29 的一次短窗口（约 7 s，由 `ls -l speed/` 的文件体积折算）：直连 268,314 B，`ghfast.top` 1,489,774 B，`gh-proxy.com` 1,484,107 B，`ghproxy.net` 142,755 B。两次数值差一个量级，说明带宽与其他子任务共享、波动大。

与公共任务书的出入：任务书说 GitHub 直连超时；本次小请求直连返回 200（0.44 s），大文件直连在 19:29 能收到约 268 kB、在 19:41 的 8 s 内为 0 字节，属于不稳定而非完全不通。`git clone` 直连我只试过一次，数分钟内只收到约 4 MB，被我终止，没有留日志文件（未核实的口述记录）。结论：下载走 `https://ghfast.top/<原 URL>`，次选 `https://gh-proxy.com/<原 URL>`，只取需要的文件，不整仓克隆。

其他站点：arXiv 直连可用（摘要页、PDF、HTML 全文均已取到，`arxiv/`）；`hf-mirror.com` 的 `/api/models/<id>` 与 `/tree/main` 可用，但并发 6 路时 `tree` 接口返回 429，改成串行、间隔 2.5 s 并退避后全部取到（`retry_hf_tree.py`，`hf/summary_retry.txt`）；ModelScope API（`modelscope.cn/api/v1/models/valyentine/IndoorUAV-Agent/repo/files`）返回 200。GitHub API 免密钥每小时 60 次，本次大约用掉一半。

### 4.2 已下载的基线（均在数据盘，`$D=/root/rivermind-data/uav_rgb`）

| 内容 | 路径 | 体积 | 来源 |
|---|---|---|---|
| DroNet 权重、模型结构、控制代码 | `$D/baselines/dronet/` | 3.8 MB（`dronet_model.zip` 1,207,446 B，与 HEAD 的 Content-Length 一致；转成 `dronet_best_weights.npz` 1,301,514 B） | UZH 官方 zip，经 ghfast.top 之外的原站断点续传（原站约 14 kB/s） |
| PULP-Dronet v2/v3、Tiny v3 的模型代码与权重、控制代码 | `$D/baselines/pulp-dronet/` | 2.8 MB | ghfast.top raw，只取需要的文件（8 s） |
| TrailNet | `$D/baselines/trailnet/` | 44 MB（caffemodel 45,152,711 B） | ghfast.top，19 s |
| `venv-h5`（h5py，读 `.h5`） | `$D/baselines/venv-h5/` | 117 MB | PyPI；原因：现有三个 venv 都没有 h5py，且规则禁止往里装包 |
| `venv-cv4`（opencv-python-headless 4.10.0.84，numpy<2，pillow） | `$D/baselines/venv-cv4/` | 264 MB | PyPI；原因：teacher-venv 的 OpenCV 5.0.0 没有 `readNetFromCaffe` |
| 草稿、日志、自测帧 | `/root/rivermind-data/uav_rgb/tmp/phase12-20261110/baselines/` | 6.6 MB | |

新增占用合计约 437 MB（`du -sh $D/baselines` 为 430 MB 加草稿 6.6 MB），低于 3 GB 上限。数据盘剩余 15 GB（`df -h /root/rivermind-data`）。第二、三档没有下载任何权重，只做体积探测（§3）。

### 4.3 为什么新建 venv

1. `h5py` 不在 `$D/3dgs-venv`、`$D/teacher-venv`、`$R/.venv-tools` 里，DroNet 的 `.h5` 读不了；规则禁止装进现有环境，所以新建 `venv-h5`，只用它把 `.h5` 转成 `.npz`；之后推理用 NumPy 前向，在 teacher-venv 里运行。
2. TrailNet 是 Caffe 模型，teacher-venv 的 OpenCV 5.0.0 没有 Caffe 导入器（实测 `AttributeError: module 'cv2.dnn' has no attribute 'readNetFromCaffe'`），所以新建 `venv-cv4`，由启动器 `run_trailnet.sh` 选择该解释器。
3. 第一档四个网络没有用到 PyTorch 的独立 venv：PULP 的 PyTorch 前向直接用 teacher-venv 里已有的 torch（只读使用）。重训臂、第二档深度模型（要装 torch 之外的包时）才需要新的独立 venv，是否允许见 §6。

### 4.4 冒烟：经真实 `StudentPolicy` 的端到端自测

做法：`proto_selftest.py` 只读 import 仓库的 `student.py`（`sys.path` 指到 `$R/sim/tools`，`PYTHONDONTWRITEBYTECODE=1`，`CUDA_VISIBLE_DEVICES` 为空，即只用 CPU），构造 `StudentPolicy(stub.ckpt, 包装器)`，对 3 张真实闭环帧各调一次 `act()`，再 `reset()`、`close()`。3 张帧：`frames/gz_pillars_113_f05.png`、`frames/gz_pillars_113_f12.png`（Gazebo 柱阵 seed 113），`frames/3dgs_stump_r5_f20.png`（3DGS stump 场景），均为 320×240 RGB 8 位（读 PNG 头所得），sha256 前 16 位依次为 `0efce45b10886601`、`882b014f6bf1bf2d`、`4f2b8a2e95ce67f6`。原始闭环输出里这三张帧的来源路径我没有再核对，它们是此前从已有闭环产物里复制出来的（未核实原路径）。

五个基线全部通过 hello 校验、预热、三次 `act()`、`reset()`、`close()` 且退出码 0（`selftest_all.txt` 里 `close rc ok` 共 5 行）。原始日志全文见附录 C。下表由日志整理：

| 基线 | 构造 + 预热 | 帧 f05（Gazebo） | 帧 f12（Gazebo） | 帧（3DGS） | CPU 时延 |
|---|---|---|---|---|---|
| forward（恒前进，协议对照） | 0.38 s | 动作 1 | 动作 1 | 动作 1 | 0.00 ms |
| PULP Tiny v3 | 3.25 s | steer −0.0059，p 0.6566，动作 1 | steer 0.0886，p 0.7918，动作 0 | steer 0.0168，p 0.8363，动作 0 | 4.54–5.18 ms |
| PULP v3 ResBlock-1.0 | 3.09 s | steer 0.0246，p 0.1154，动作 1 | steer 0.1409，p 0.9869，动作 2 | steer −0.0737，p 0.0332，动作 1 | 7.93–9.62 ms |
| DroNet（NumPy 前向） | 2.56 s | steer 0.6421，p 1.0000，动作 2 | steer 0.0551，p 0.9994，动作 0 | steer 0.0260，p 0.8958，动作 0 | 7.20–7.86 ms |
| TrailNet（OpenCV 4.10 dnn） | 0.53 s | 6 维输出见附录 C，动作 0（未设映射，恒 SAFE） | 同左 | 同左 | 19.41 / 22.56 / 98.13 ms（第三帧离群，原因未查） |

（各基线日志里 n=0 是预热的全零帧，不计入上表；时延是包装器自己计时的 `infer_ms`，含预处理与前向，CPU，三帧样本量太小只能当量级。两次重跑之间同一基线的时延有 ±1 ms 级波动，动作与 steer/p 不变，说明前向是确定的。）

观察与风险（每条都有日志依据）：

1. DroNet 在三张闭环帧上的碰撞概率为 1.0000、0.9994、0.8958，全部高于 0.7。映射规则里转向优先，所以第一帧（steer 0.6421，折算 12.84°，刚过 12.5° 门槛）转为 YAW_LEFT，另两帧刹停（动作 0）。零样本 DroNet 在我们的柱阵与 3DGS 场景里大概率"原地不动或原地转"，这是域差异的直接证据，不是包装器的问题（三帧样本只能提示，不能下结论）。任务卡里为此预注册了一个"阈值按开发帧中位数重标定"的对照臂。
2. PULP v3 在 f12 上 steer=0.1409，折算 `0.1409 × 90 × 1.0 = 12.68°`，刚好越过 12.5° 的转向门槛，说明映射对门槛敏感，敏感性扫描必须包含 `YAW_SCALE_PULP` 与门槛。
3. PULP Tiny 的碰撞概率在 0.66 到 0.84 之间，恰好围着 0.7 门槛，结果会随 `P_CRIT` 翻转，同样列入扫描。
4. 转向符号没有核实（只在 PULP README 里读到 YawRate>0 为逆时针即左转）；镜像帧检验没做（§「没做什么」）。
5. DroNet 的 NumPy 前向没有与 Keras 原模型做数值对拍（现有环境无 TF 1.5，且规则不允许往现有 venv 装包），层结构按 `model_struct.json` 手写（Keras 'same' 填充的不对称、BN eps 1e-3、NHWC 展平顺序已处理，但未与原模型逐层核对）：数值一致性未核实。
6. 预处理取法敏感（`BASELINE_PREP`，同样 3 帧、CPU，原始输出 `prep_sensitivity.txt`）：同一帧换取法，steer、p 与动作都会变，零样本臂的成绩会随取法浮动，闭环敏感性必须做（没做）。格式为"steer / p → 动作"。

| 基线 | 取法 | f05（Gazebo） | f12（Gazebo） | 3DGS |
|---|---|---|---|---|
| PULP v3 | native | 0.0246 / 0.115 → 1 | 0.1409 / 0.987 → 2 | −0.0737 / 0.033 → 1 |
| PULP v3 | resize_crop | −0.0004 / 0.304 → 1 | 0.1325 / 0.998 → 0 | −0.0904 / 0.576 → 1 |
| PULP v3 | squash | 0.0294 / 0.139 → 1 | 0.0079 / 0.990 → 0 | −0.0693 / 0.593 → 1 |
| PULP Tiny | native | −0.0059 / 0.657 → 1 | 0.0886 / 0.792 → 0 | 0.0168 / 0.836 → 0 |
| PULP Tiny | resize_crop | −0.0170 / 0.751 → 0 | 0.0828 / 0.894 → 0 | −0.0136 / 0.996 → 0 |
| PULP Tiny | squash | 0.0289 / 0.487 → 1 | 0.0327 / 0.851 → 0 | 0.0179 / 0.995 → 0 |
| DroNet | native | 0.6421 / 1.000 → 2 | 0.0551 / 0.999 → 0 | 0.0260 / 0.896 → 0 |
| DroNet | resize_crop | 0.1520 / 1.000 → 0 | 0.2318 / 0.998 → 0 | 0.0380 / 0.733 → 0 |
| DroNet | squash | 0.1118 / 1.000 → 0 | 0.2642 / 1.000 → 0 | 0.0944 / 0.896 → 0 |

7. `BASELINE_WEIGHTS` 覆盖路径：令其指向官方 `pulp-dronet-v3-resblock-1.0.pth` 重跑 `pulp_v3`，hello、三次 `act()`、关闭均通过（输出与默认路径相同的动作序列 1、2、1），说明重训臂的权重可经该变量接入。

## 5. 结论

### 5.1 短名单（要真正跑进闭环的基线，按优先级排序）

排序依据：先看与论文主张的贴合度（同量级小模型、输出分解方案的消融），再看单个基线的接入成本，最后看它对结论的信息量。大型 VLN/VLA 没有进短名单，原因是本机装不下权重（§3.3）。

| 序 | 基线 | 臂 | 回答的问题 | 接入成本（估） |
|---|---|---|---|---|
| 1 | PULP-Dronet v3 家族（ResBlock-1.0 与 Tiny dw-pw-0.125） | 零样本 + 同架构在我们数据上重训 | 同量级公开小模型直接用能到哪；同样的网络给我们的数据与教师能到哪 | 零样本：已完成协议自测，约半天跑完两个任务；重训：需要训练脚本与标签，约 1 至 2 天（估计，未测） |
| 2 | Lamberti 2023 三条带碰撞头框架（复现） | 仅重训，无公开权重 | 输出分解的粒度：3 条带 vs 我们的 20 列 × 4 档 | 骨干复用 PULP v3，换 3 输出头与论文规则，约 1 至 2 天（估计），前提是训练集能给出每条带的距离标签 |
| 3 | Depth-Anything-V2 Small 米制（室内、室外两个权重）+ `expert_action` 条带规则；加真值深度 + 同规则的上界 | 零样本 | 几何基础模型加固定规则的表现；规则本身的天花板 | 约半天，权重 2 × 99.2 MB，teacher-venv 已有 torch，不需新 venv |
| 4 | DroNet 原版 | 零样本，加一个阈值重标定对照臂 | 经典基线，学界默认要对比 | 已就绪，约 2 小时 |
| 5 | UniDepthV2 Small + 相机内参 + 同规则 | 零样本，条件项 | 喂入已知内参的米制深度是否优于 DA-V2 | 需要新 venv（xformers 等，体积可能超过 3 GB 预算，未测），只在第 3 项显示深度路线有竞争力时才做 |

"接入成本"一栏的天数与小时数都是我的估计，没有实测；显存与时延同理，凡写"估计"或"未测"的都没有出处。

#### 公共前置（所有任务卡共用）

1. 环境变量：`export CUDA_VISIBLE_DEVICES=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/root/rivermind-data/uav_rgb/tmp`；长任务用 `setsid nohup`，写 pid、日志、DONE 标记，用 `flock` 防重复，停止只对进程组发 `kill`（负号加 PGID，写法照公共任务书），不用 `pkill -f`。
2. 包装器与启动器：草稿目录里的 `baseline_infer.py`（附录 A）与启动器模板（附录 B）。正式使用前，由主会话决定是否把它们放进仓库的 `tools/` 之类的位置，本任务没有放。
3. 先跑协议自测：`$R/.venv-tools/bin/python proto_selftest.py <基线名>`，必须看到 `close rc ok`。
4. 评测入口、种子、成功判据、分母：见 §5.3，所有基线一致。
5. `--task base`（柱阵）与 `--task forest_e1`（森林）走 `sim/tools/eval_closed_loop.py`，3DGS 实景线走 `sim/3dgs/eval_3dgs.py`；两者都用 `--policy student --ckpt <stub.ckpt> --infer-bin <启动器>`；其余参数照抄学生评测命令（我没有核对完整命令行，执行者先对照学生评测的现行命令）。

#### 任务卡 B1：PULP-Dronet v3 家族

- 目标：两个尺寸（320,928 与 2,784 参数）各跑零样本与重训两个臂，共 4 个臂，每个臂在柱阵、森林两个任务上各 20 seed。
- 前置：权重与代码已在 `$D/baselines/pulp-dronet/src/`（已下载并通过自测，无需重下）。重现下载命令：`curl -fL -m 120 -o <目标> https://ghfast.top/https://raw.githubusercontent.com/pulp-platform/pulp-dronet/master/tiny-pulp-dronet-v3/model/pulp-dronet-v3-resblock-1.0.pth`，文件名换成 `tiny-pulp-dronet-v3-dw-pw-0.125.pth` 与 `dronet_v3.py` 即得其余文件。
- 适配器：`BASELINE=pulp_v3` 或 `pulp_tiny`，启动器模板见附录 B；输出用 §2.3 第一行规则（S=90 °/s，P_CRIT=0.7，门槛 12.5°）。
- 零样本臂：直接用官方权重。预注册的敏感性变量：`BASELINE_PREP`（三种取法）、`STEER_SIGN`（在开发 seed 上校准一次后冻结）、`P_CRIT`∈{0.5, 0.7, 0.9}。开发 seed 取与 20 个冻结 seed 不重叠的 2001 至 2010。
- 重训臂：网络结构与损失沿用 PULP 官方（转向回归 + 碰撞二分类，见 `tiny-pulp-dronet-v3/` 里的 `utility.py` 与 `config.py`）。标签从我们的训练集派生：转向标签 = 教师动作为 YAW_LEFT 取 +25/(90×1)≈+0.278，YAW_RIGHT 取 −0.278，其余取 0；碰撞标签 = 教师中心条带距离小于 `turn_dist=1.5 m` 记 1，否则 0。**训练集是否带距离或深度字段我没有核实**，没有则退一步用"教师动作∈{YAW_LEFT, YAW_RIGHT, SAFE_NEUTRAL} 记 1，FORWARD 记 0"，这会引入标签噪声，必须在论文里写明。数据量、增广、教师与学生一致；超参与早停只用开发划分，不看 20 个冻结 seed。训练环境：teacher-venv 的 torch（只读使用）；若用户要求独立 venv，见 §6。
- 资源：显存 0（CPU 推理 4.5 至 10 ms，[sel]）；磁盘新增只有重训检查点，约 1.3 MB 与 24 kB 量级；训练 GPU 时间未测，输入 200×200、41.1 M MAC，估计每个 epoch 分钟级（估算）。
- 验收：4 个臂 × 2 任务 × 20 seed 全部有结果行；`infer_bin_sha256` 写入报告；失败 episode 记原因而不是丢弃（分母 = 20）；重训臂的训练日志与检查点哈希落盘；两次独立重跑同一臂的成功数之差不超过 2/20，否则报告仿真自身方差。

#### 任务卡 B2：三条带框架复现

- 目标：复现 Lamberti 2023 的"三条竖带各输出碰撞概率 + 最小概率方向规则"，只做重训臂。
- 前置：论文原文 `arxiv/text_2312.08991.clean.txt`；没有公开代码和权重（§3.1），骨干用 PULP v3 ResBlock-1.0，把两个输出头换成三个碰撞概率输出。
- 标签：每条带内，属于障碍物的像素里有 10% 的距离 ≤2 m 记 1（论文原话，已核对）。需要逐像素深度与障碍物掩码，来自仿真真值渲染；训练集若没有，需先重渲染一批帧，这一步的工作量未估。
- 规则：§2.3 第二行；T 的论文值本次没读到，预注册在 {0.3, 0.5, 0.7} 上用开发 seed 选一个，冻结后只跑一次。
- 与我们方法的关系：骨干与数据相同、只换输出头（3 条带 vs 20 列 × 4 档 vs 直接动作行为克隆），是"分解方案消融"最干净的对照，不需要引入别的模型。
- 资源与验收：同 B1（CPU 推理，检查点约 1.3 MB）；验收额外要求给出每条带标签的正例比例统计，防止类别极端失衡。

#### 任务卡 B3：Depth-Anything-V2 Small 米制 + 条带规则

- 目标：三种深度来源各跑一遍同一条带规则：真值深度（上界）、DA-V2 Small 室内权重（`max_depth=20`）、室外权重（`max_depth=80`）。
- 下载（先 HEAD 看体积，再下；均小于 500 MB）：
  ```bash
  D=/root/rivermind-data/uav_rgb; mkdir -p $D/baselines/depth-anything-v2/weights $D/baselines/depth-anything-v2/src
  curl -sI https://hf-mirror.com/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth | grep -i -E "content-length|location"
  curl -fL -C - -o $D/baselines/depth-anything-v2/weights/depth_anything_v2_metric_hypersim_vits.pth https://hf-mirror.com/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
  curl -fL -C - -o $D/baselines/depth-anything-v2/weights/depth_anything_v2_metric_vkitti_vits.pth https://hf-mirror.com/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth
  # 代码：仓库约 46 MB（[gh] 的 size 字段 46,233 KB）；包装器期望 $D/baselines/depth-anything-v2/src/metric_depth/depth_anything_v2/dpt.py
  curl -fL -m 600 -o $D/baselines/depth-anything-v2/src.zip https://ghfast.top/https://github.com/DepthAnything/Depth-Anything-V2/archive/refs/heads/main.zip
  unzip -q $D/baselines/depth-anything-v2/src.zip -d $D/baselines/depth-anything-v2/src.tmp && mv $D/baselines/depth-anything-v2/src.tmp/Depth-Anything-V2-main/* $D/baselines/depth-anything-v2/src/
  ```
  权重各 99.2 MB（[hf]），两个共 198.4 MB。
- 适配器：`BASELINE=da2_rule BASELINE_DEVICE=cuda DA2_WEIGHTS=<权重> DA2_MAX_DEPTH=20 或 80`；包装器里 `Da2Rule` 目前是骨架（附录 A），**没有跑通**：需要补全 import 路径与 `infer_image` 输入尺寸（草稿里取 266，14 的倍数、接近 240），并用 `--task base` 的真值深度对一次。teacher-venv 已有 torch 2.14.1+cu126、torchvision 0.29.1、cv2 5.0.0（实测），DA-V2 核心代码只依赖这些，不需要装包，但未验证。
- 公平性：深度到动作的规则与学生一致（§2.3 末行）；真值深度版本用于判断瓶颈在规则还是在估计。柱阵是无纹理灰色柱体，室内与室外两个权重的米制尺度可能偏差大，所以同时报告两个。
- 资源：显存估计小于 1 GB、延迟没测（任务第一步就是测 100 帧的平均与 p95），磁盘新增约 245 MB。
- 验收：深度估计在 100 张闭环帧上与真值深度的相对误差分布（中位数、p90）写入报告；3 个深度来源 × 2 任务 × 20 seed 全有结果。

#### 任务卡 B4：DroNet 原版

- 已就绪：`BASELINE=dronet`，`STEER_SIGN` 在开发 seed 上校准，`YAW_SCALE_DRONET=20`（`outdoor.yaml` L31）。
- 臂：忠实臂（P_CRIT=0.7、S=20）；重标定臂（P_CRIT 取 200 张开发帧碰撞概率的中位数，S 取 90 与 PULP 对齐，两个改动都要在报告里写明）。
- 先做两件未完成的核对：镜像帧检验（左右翻转输入 steer 应变号），以及与 Keras 原模型的数值对拍（需要 TF 1.x，本机没有，可以改为只核对第一层卷积输出与手算，工作量小）。
- 验收：同 B1；额外写出两个臂的"前进占比"（FORWARD 步数 / 总步数），用来证明不是原地不动造成的失败。

#### 任务卡 B5（条件项）：UniDepthV2 Small + 内参

- 触发条件：B3 里 DA-V2 估计深度版本的成功率不低于真值深度版本的一半（阈值是我的设定，没有出处）。
- 前置：新 venv（`pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118`，README L74），体积可能超过 3 GB 预算（teacher-venv 实测 6.6 GB，未测新 venv），需要用户同意（§6）。权重 `lpiccinelli/unidepth-v2-vits14`，只取一个文件（136.8 MB），许可证 CC BY-NC 4.0（只能做研究用途）。
- 适配器：与 B3 相同的规则，输入时传入我们已知的内参（320×240，水平 fov 60°，焦距约 277 像素，由 `0.5×320/tan(30°)` 算得）。
- 验收：同 B3。

### 5.2 只引用、不跑的方法

| 方法 | 引用的数字与出处 | 不跑的原因 |
|---|---|---|
| DiffPhysDrone（Nature Machine Intelligence 2025，arXiv 2407.10648） | 单机场景成功率 90%，此前最好 60%；真实林间最高 20 m/s [abs 2407.10648] | README 只有训练代码、没有公开权重；要自己训练；输入与状态是特权配置；许可证字段为空 |
| NavRL（arXiv 2409.15634） | 摘要无数字 | 训练依赖 Isaac Sim 2023.1.0-hotfix.1，部署依赖 ROS；输入形式未核实 |
| YOPO（TJU） | Orin NX 上 TensorRT 推理 1 至 5 ms（README L182） | ROS + CUDA 环境，输入含深度与状态；许可证 NOASSERTION |
| EGO-Planner、EGO-Planner-v2、Fast-Planner | 无（摘要无数字） | ROS + 深度 + 里程计，经典规划器，条件不同；GPL-3.0 |
| Agile Autonomy、MAVRL | 无 | 需要 Unity 渲染（本机无 GPU-GL 容器）与 ROS Noetic |
| NaVid、Uni-NaVid、NaVILA、StreamVLN | Uni-NaVid：单张 A100 约 5 Hz（README L110），VLN-CE R2R Val SR 51.8、SPL 47.7；StreamVLN：R2R SR 56.4、SPL 50.2（README L39）。均为地面机器人室内导航，条件与我们的避障任务不同 | 权重各约 15 至 17 GB，数据盘剩 15 GB 装不下；Habitat 依赖或 FlashAttention-2（2080 Ti 不支持）；需要自然语言指令，我们的 `base`/`forest_e1` 任务没有指令；个别是带历史的有状态模型，3DGS 线协议没有 reset 报文 |
| OpenFly-Agent-7B、OpenVLA-UAV | OpenFly 100k 轨迹、18 场景 [abs 2502.18041]；平台 README 逐场景 SR 10.6% 到 43.8%（模型与划分未写明） | 权重 15 GB；平台需要 UE/AirSim 等环境；输入输出格式与我们的相机不同 |
| IndoorUAV-Agent | 1,000 多个场景、16,000 多条轨迹 [abs 2512.19024] | 单个 9.47 GB 的 zip；室内 Habitat 场景 |
| See-Point-Fly | 无需训练 [abs 2509.22653] | 需要外部 VLM 接口与密钥，本机没有；仓库未查 |
| VLN-CE Seq2Seq / CMA | 论文 [abs 2004.02857]，数字未读 | Habitat 0.1.7 与 RGB-D 输入，权重位置未核实 |
| MonoNav | Crazyflie 0.5 m/s 室内 [abs 2311.14100] | 需要相机位姿，属特权输入；可作为第二档的补充，但会改变"仅 RGB"的条件 |
| TrailNet | 1 km 林间小路飞行 [abs 1705.02550] | 任务是沿小路飞行，不是避障；只做冒烟（已通过协议自测） |
| DA-V2 Base/Large、Metric3D v2、ZoeDepth、Depth Pro | 无 | 体积与许可证（Base/Large 为 CC-BY-NC-4.0）或维护状态（ZoeDepth 已归档）不利；Small 版本足以回答"几何基础模型加规则"的问题 |
| NanoFlowNet、2607.12593 | 内存缩小 2 倍、推理快 1.6 倍、自由飞 1.96 m/s [abs 2607.12593] | 本次没读代码与权重，未核实可用性 |

### 5.3 对比实验总设计

**统一条件**：同一批 20 个冻结 seed（`1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654`，出自公共任务书 `BRIEF_COMMON.md`，共 20 个），同一判据 `sim09.v2.coast1s.corridor3m`（无碰撞、越过最远柱 x+1.0 的通过线、全程 |y| ≤ 3 m），分母 = 请求的 episode 数（崩溃与超时记失败并写原因），同一评测入口，同一观测（320×240 RGB、60° 视场、1 Hz 决策、动作保持 1.0 s），没有特权信息（第二档的真值深度臂单独标注为"上界"，不参与排名）。

**统计**：每个方法给 `成功数/20` 与 Wilson 95% 区间；与我们的学生做配对 McNemar 精确检验，报告不一致对数 b、c 与双侧 p。n=20 的检出能力很弱，按二项分布算：学生全成功、基线 14/20 时 b=6、c=0，p=2×0.5⁶=0.031；基线 15/20 时 b=5，p=0.0625，不显著；基线 10/20 时 p=0.002。所以 20 个 seed 只能区分差距很大的对手，差距小的比较要扩 seed 或标"不确定"，这一点要在论文里写明。预注册两个主比较：学生对 PULP v3 重训臂、学生对 DA-V2 估计深度加规则；其余为探索性，做 Holm 校正。

**敏感性扫描**：只在非冻结的开发 seed（建议 2001 至 2010）上做：`STEER_SIGN`、`P_CRIT`、`YAW_SCALE`、`BASELINE_PREP`、三条带阈值 T；选定的取值写进报告后，冻结 seed 上每个臂只跑一次。

**哪些比较最支持论文主张**：

1. 同量级小模型（B1 的重训臂、B4）：回答"在同一数据、同一教师、同一算力级别下，我们的输出分解与训练方式是否更好"，重训臂比零样本臂更有说服力，因为它排除了域差异。
2. 输出分解方案的消融（B2 对比我们的 20 列 × 4 档，再对比直接动作行为克隆）：骨干、数据相同，只换输出头，因果最清楚。
3. 大模型或几何基础模型的上界（B3 的估计深度与真值深度、第三档的引用数字）：说明我们的 0.1 M 参数模型与 24.8 M 深度模型加规则、7B 级 VLA 之间的距离。第三档在本机无法复现，所以只能在论文里引用并标明评测条件不同。
4. 零样本臂（B1、B4）是"公开权重直接用"的参照，不能用来证明结构优劣，只用来说明域差异本身有多大。

## 6. 需要用户决定的事

1. **基线能否使用 PyTorch**：第一档推理目前借用 `teacher-venv` 里已有的 torch（只读，未安装任何东西）。重训臂要写训练脚本，B5 要新建带 torch 与 xformers 的 venv（体积可能超过 3 GB 预算）。需要用户确认：是否允许基线放在数据盘独立 venv，是否允许重训臂使用 `teacher-venv` 的 torch。
2. **重训臂的数据与标签**：训练集是否带逐帧距离或深度、障碍物掩码（B1 的碰撞标签、B2 的条带标签都依赖它）。没有的话是退到动作派生标签（有噪声），还是重新渲染一批帧。
3. **大模型要不要有**：本机数据盘剩 15 GB，装不下任何 7B 级权重。如果论文需要一个大模型数据点，需要用户提供 ≥ 25 GB 的存储位置并批准下载（单个约 15 至 17 GB，超出 500 MB 的限制），建议候选是 NaVILA（输出中层动作，映射规则现成）；否则第三档只在论文里引用。
4. **3DGS 线上有状态基线**：`eval_3dgs.py` 复用同一个策略进程，没有 reset 报文。短名单里的基线都是无状态的，不受影响；以后要接带历史的 VLM，需要改仓库（每 episode 新进程），本任务不改。
5. **包装器放哪里**：草稿目录里的 `baseline_infer.py` 等文件是否并入仓库（例如 `sim/tools/baselines/`），由主会话决定。
6. **GitHub 直连**：公共任务书说直连超时，本次小请求直连通、大文件不稳定；下载统一走 `ghfast.top`，如果后续镜像失效需要重新探测。

## 附录 A：外部基线包装器 `baseline_infer.py`（草稿，不入库；已通过 §4.4 的自测）

路径：`/root/rivermind-data/uav_rgb/tmp/phase12-20261110/baselines/baseline_infer.py`。`da2_rule` 后端是未运行的骨架。

```python
#!/root/rivermind-data/uav_rgb/teacher-venv/bin/python
# -*- coding: utf-8 -*-
"""baseline_infer.py：外部基线适配器（草稿，不入库）。

说与 uavrgb_infer 同一套行协议，让 `--infer-bin <本脚本>` 不改评测代码就把任意模型放进闭环：
    eval_closed_loop.py --policy student --ckpt <任意存在的文件> --infer-bin <本脚本> ...
    eval_3dgs.py        --policy student --ckpt <任意存在的文件> --infer-bin <本脚本> ...
协议（见 sim/tools/student.py、src/infer_main.cpp、src/infer.cpp）：
    启动：  <bin> --ckpt <file> [--text id,id,...]
    握手：  stdout 先写一行 hello JSON：{"in_h","in_w","in_c","num_classes":4,"text_dim","prev_action_dim","aux_classes"}
    每步：  stdin 读 in_h*in_w*3*(in_c/3) 字节（HWC uint8 RGB，当前帧在前）+ prev_action_dim>0 时 1 字节；
            stdout 写一行 {"action":0..3,"probs":[4 个浮点],"infer_ms":浮点}（额外字段会被忽略）
    结束：  stdin EOF 时退出码 0（StudentPolicy.close() 对非 0 退出码抛错）
运行：BASELINE=pulp_tiny ./baseline_infer.py --ckpt /任意存在的文件
环境变量见下方配置区；默认值就在本文件里。
"""
import importlib.util
import json
import os
import sys
import time
import types

sys.dont_write_bytecode = True  # 规则：PYTHONDONTWRITEBYTECODE=1，根分区只剩 2.4 GB

# ------------------------------ 配置区（环境变量可覆盖） ------------------------------
BASELINE = os.environ.get("BASELINE", "pulp_tiny")            # forward | pulp_v3 | pulp_tiny | dronet | trailnet | da2_rule
BASE = os.environ.get("BASELINE_ROOT", "/root/rivermind-data/uav_rgb/baselines")  # 第三方代码与权重都在数据盘
REPO = os.environ.get("UAV_RGB_REPO", "/root/claude-code-backup-20261028/code/uav_rgb")  # 只读 import teacher.expert_action
DEVICE = os.environ.get("BASELINE_DEVICE", "cpu")             # 第 1 档网络 <1.5 MB，CPU 毫秒级；第 2 档用 cuda（CUDA_VISIBLE_DEVICES 由调用方钉卡）
PREP = os.environ.get("BASELINE_PREP", "native")              # 灰度 200x200 取法：native(各自原文的裁法) | resize_crop | squash
YAW_STEP_DEG = float(os.environ.get("YAW_STEP_DEG", "25"))    # 公共任务书 §1：YAW 动作每步转约 23–27°，取中值
HOLD_S = float(os.environ.get("HOLD_S", "1.0"))               # 每个动作保持 1.0 s
STEER_SIGN = float(os.environ.get("STEER_SIGN", "1"))         # +1：steer>0 = 左转（PULP-Dronet README：YawRate>0 = 逆时针 = 左）；DroNet 符号未核实，先镜像帧检验
P_CRIT = float(os.environ.get("P_CRIT", "0.7"))               # 碰撞概率紧急刹停阈值：DroNet outdoor 控制节点 critical_prob=0.7，PULP config_dronet.h CRITICAL_PROB_COLL=0.7
YAW_SCALE_PULP = float(os.environ.get("YAW_SCALE_PULP", "90"))  # PULP：yaw_rate=steer*YAW_SCALING；COLLISION_TEST 配置与 utility.normalize_yaw_rate 都是 90 °/s
YAW_SCALE_DRONET = float(os.environ.get("YAW_SCALE_DRONET", "20"))  # DroNet Bebop outdoor.yaml SpeedSettingsMaxRotationSpeedCurrent=20 °/s
DEPTH_TURN_DIST = float(os.environ.get("DEPTH_TURN_DIST", "1.5"))   # 深度规则：沿用 teacher.expert_action 默认 turn_dist
DEPTH_EMERGENCY = float(os.environ.get("DEPTH_EMERGENCY", "0.8"))   # 同上 emergency
BASELINE_LOG = os.environ.get("BASELINE_LOG", "")             # 非空：每步追加一行原始输出 JSONL（steer/p_coll 等），供事后对账
# --------------------------------------------------------------------------------------

SAFE, FWD, LEFT, RIGHT = 0, 1, 2, 3


# ------------------------------ 输出 → 4 个动作的映射 ------------------------------
def map_steer_coll(steer, p_coll, yaw_scale_deg_s, sign=STEER_SIGN):
    """转向量 + 碰撞概率 → 离散动作。

    转：本步想转的角度 = sign*steer*scale*HOLD_S；超过半个 YAW 步长就转（半步长是「转」与「不转」两个原语的最近邻分界）。
    进：碰撞概率不超过 P_CRIT 才前进（原文控制律里 v=(1-p)*vmax，p>0.7 时紧急刹停）。
    转与进冲突时先转：离散动作不能边转边进，朝向错了前进只会更近障碍。
    """
    yaw_cmd = sign * steer * yaw_scale_deg_s * HOLD_S
    if abs(yaw_cmd) >= 0.5 * YAW_STEP_DEG:
        return LEFT if yaw_cmd > 0 else RIGHT
    return FWD if p_coll <= P_CRIT else SAFE


def map_velocity(vx_mps, yaw_rate_deg_s, fwd_step_m=0.25):
    """连续速度（前向 m/s、偏航 °/s）→ 动作：把 1 s 的位移与转角量化到最近的原语（0.25 m / 25°）。"""
    dyaw, dx = yaw_rate_deg_s * HOLD_S, vx_mps * HOLD_S
    if abs(dyaw) >= 0.5 * YAW_STEP_DEG:
        return LEFT if dyaw > 0 else RIGHT
    return FWD if dx >= 0.5 * fwd_step_m else SAFE


class MidLevelQueue:
    """中层动作（「前进 75 cm」「左转 30°」）→ 微动作队列；队列空时才重新问模型（开环执行一段）。

    前进 d 米 = round(d/0.25) 次 FORWARD；转 θ° = round(θ/25) 次 YAW（至少 1 次，除非 θ<12.5°）；STOP = SAFE。
    """

    def __init__(self, fwd_step_m=0.25):
        self.q, self.fwd = [], fwd_step_m

    def push(self, kind, value):
        if kind == "forward":
            self.q += [FWD] * max(int(round(value / self.fwd)), 0)
        elif kind in ("left", "right"):
            n = int(round(abs(value) / YAW_STEP_DEG))
            self.q += [LEFT if kind == "left" else RIGHT] * n
        else:  # stop
            self.q.append(SAFE)

    def pop(self):
        return self.q.pop(0) if self.q else None


# ------------------------------ 各基线适配器 ------------------------------
class Adapter:
    in_h, in_w, in_c, prev_dim = 240, 320, 3, 0   # hello 里要原生 320x240：StudentPolicy 的 PIL resize 同尺寸即直通，各基线自己缩放

    def step(self, frames, prev_action):          # frames: [当前帧(, 上一帧)]，每个 uint8 [H,W,3]
        raise NotImplementedError


class Forward(Adapter):
    """盲前进：自检用，确认包装本身没有改变闭环结果（应与 --policy forward 逐步一致）。"""

    def step(self, frames, prev_action):
        return FWD, {}


def _gray(frame, mode, bottom_aligned=False):
    """RGB uint8 [240,320,3] → 灰度 float32 [200,200]，0..1。"""
    from PIL import Image
    import numpy as np
    im = Image.fromarray(frame).convert("L")      # ITU-R 601 亮度，与 PIL/cv2 灰度一致
    w, h = im.size
    if mode == "native":                          # 原文流水线：先到 320x240（本相机已是），再中心裁 200x200（DroNet 取底部居中）
        x0, y0 = (w - 200) // 2, (h - 200 if bottom_aligned else (h - 200) // 2)
        im = im.crop((x0, y0, x0 + 200, y0 + 200))
    elif mode == "resize_crop":                   # 高缩到 200 再中心裁宽：保留完整竖直视场
        im = im.resize((round(w * 200 / h), 200), Image.Resampling.BILINEAR)
        x0 = (im.size[0] - 200) // 2
        im = im.crop((x0, 0, x0 + 200, 200))
    elif mode == "squash":                        # 整幅压成方形：保留完整水平视场，宽高比失真
        im = im.resize((200, 200), Image.Resampling.BILINEAR)
    else:
        raise ValueError(f"BASELINE_PREP={mode}")
    return np.asarray(im, dtype=np.float32) / 255.0


class PulpDronetV3(Adapter):
    """pulp-platform/pulp-dronet tiny-pulp-dronet-v3：灰度 200x200 → (steer∈[-1,1], p_coll)。"""

    def __init__(self, variant):
        import torch
        self.torch = torch
        root = f"{BASE}/pulp-dronet/src/tiny-pulp-dronet-v3"
        # 第三方模型文件顶部 import nemo（量化库），FP 前向里 PACT_IntegerAdd 等价于 a+b；用桩替代，不改第三方源码。
        nemo, quant, pact = (types.ModuleType(n) for n in ("nemo", "nemo.quant", "nemo.quant.pact"))

        class PACT_IntegerAdd(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        pact.PACT_IntegerAdd, quant.pact, nemo.quant = PACT_IntegerAdd, pact, quant
        sys.modules.update({"nemo": nemo, "nemo.quant": quant, "nemo.quant.pact": pact})
        spec = importlib.util.spec_from_file_location("dronet_v3", f"{root}/model/dronet_v3.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if variant == "v3":      # README：ResBlock、depth_mult=1.0、bypass=True
            net, path = mod.dronet(depth_mult=1.0, block_class=mod.ResBlock, bypass=True), "pulp-dronet-v3-resblock-1.0.pth"
        else:                    # README：Depthwise、depth_mult=0.125、bypass=False
            net, path = mod.dronet(depth_mult=0.125, block_class=mod.Depthwise_Separable, bypass=False), "tiny-pulp-dronet-v3-dw-pw-0.125.pth"
        # BASELINE_WEIGHTS：同架构"在我们数据上重训"臂的权重路径（.pth，键与官方一致）；不设则用官方公开权重（零样本臂）
        ck = torch.load(os.environ.get("BASELINE_WEIGHTS") or f"{root}/model/{path}", map_location="cpu")
        net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        self.net = net.eval().to(DEVICE)
        self.n_params = sum(p.numel() for p in net.parameters())

    def step(self, frames, prev_action):
        torch = self.torch
        x = torch.from_numpy(_gray(frames[0], PREP))[None, None].to(DEVICE)
        with torch.no_grad():
            steer, coll = self.net(x)
        s, p = float(steer[0]), float(coll[0])
        return map_steer_coll(s, p, YAW_SCALE_PULP), {"steer": s, "p_coll": p}


class DroNetKeras(Adapter):
    """uzh-rpg/rpg_public_dronet：Keras 权重（h5→npz）用 torch 重写前向；灰度 200x200 → (steer, p_coll)。
    shortcut：未能与 Keras 原实现逐位对拍（本机无 TF 1.5），见报告「验证到哪一步」。"""

    def __init__(self):
        import numpy as np
        import torch
        self.torch = torch
        z = np.load(f"{BASE}/dronet/dronet_best_weights.npz")
        self.w = {k.split("|")[0] + "/" + k.split("|")[-1].split("/")[-1].replace(":0", ""): torch.from_numpy(z[k]).to(DEVICE)
                  for k in z.files}
        self.eps = 1e-3   # model_struct.json 里 BN epsilon=0.001

    @staticmethod
    def _same(x, k, s):   # TF/Keras 'same'：多出的一格补在右/下，和 torch 的对称 padding 不同
        import torch.nn.functional as F
        def tot(n):
            return max((-(-n // s) - 1) * s + k - n, 0)
        ph, pw = tot(x.shape[-2]), tot(x.shape[-1])
        return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))

    def _conv(self, x, name, k, s):
        import torch.nn.functional as F
        w = self.w[f"{name}/kernel"].permute(3, 2, 0, 1).contiguous()    # Keras [kh,kw,in,out] → torch [out,in,kh,kw]
        return F.conv2d(self._same(x, k, s), w, self.w[f"{name}/bias"], stride=s)

    def _bn(self, x, name):
        g, b = self.w[f"{name}/gamma"], self.w[f"{name}/beta"]
        m, v = self.w[f"{name}/moving_mean"], self.w[f"{name}/moving_variance"]
        sh = (1, -1, 1, 1)
        return (x - m.view(sh)) / (v.view(sh) + self.eps).sqrt() * g.view(sh) + b.view(sh)

    def step(self, frames, prev_action):
        torch = self.torch
        relu = torch.relu
        x = torch.from_numpy(_gray(frames[0], PREP, bottom_aligned=(PREP == "native")))[None, None].to(DEVICE)
        with torch.no_grad():
            x = self._conv(x, "conv2d_1", 5, 2)
            x1 = torch.nn.functional.max_pool2d(x, 3, 2)
            for bn_a, c_a, bn_b, c_b, sc in (("batch_normalization_1", "conv2d_2", "batch_normalization_2", "conv2d_3", "conv2d_4"),
                                              ("batch_normalization_3", "conv2d_5", "batch_normalization_4", "conv2d_6", "conv2d_7"),
                                              ("batch_normalization_5", "conv2d_8", "batch_normalization_6", "conv2d_9", "conv2d_10")):
                x2 = self._conv(relu(self._bn(x1, bn_a)), c_a, 3, 2)
                x2 = self._conv(relu(self._bn(x2, bn_b)), c_b, 3, 1)
                x1 = self._conv(x1, sc, 1, 2) + x2
            f = relu(x1.permute(0, 2, 3, 1).reshape(1, -1))              # Keras Flatten 是 NHWC 顺序
            steer = (f @ self.w["dense_1/kernel"] + self.w["dense_1/bias"]).item()
            coll = torch.sigmoid(f @ self.w["dense_2/kernel"] + self.w["dense_2/bias"]).item()
        s = max(-1.0, min(1.0, steer))                                   # 原控制节点对 steer 截断到 [-1,1]
        return map_steer_coll(s, coll, YAW_SCALE_DRONET), {"steer": steer, "p_coll": coll}


class TrailNet(Adapter):
    """NVIDIA-AI-IOT/redtail TrailNet_SResNet-18（Caffe）：RGB 320x180 → 6 维 softmax（视角 3 类 + 横向偏移 3 类）。
    任务是沿林间小径，不是避障；只做冒烟，不接映射（见报告）。"""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        d = f"{BASE}/trailnet"
        self.net = cv2.dnn.readNetFromCaffe(f"{d}/TrailNet_SResNet-18.prototxt", f"{d}/TrailNet_SResNet-18.caffemodel")

    def step(self, frames, prev_action):
        cv2 = self.cv2
        bgr = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)                 # Caffe 习惯 BGR 0..255，均值在网络内的 sub_mean 层做
        self.net.setInput(cv2.dnn.blobFromImage(bgr, 1.0, (320, 180), (0, 0, 0), swapRB=False, crop=False))
        out = self.net.forward().reshape(-1).tolist()
        return SAFE, {"out6": out}


class Da2Rule(Adapter):
    """第 2 档骨架：Depth Anything V2 度量深度 → 现成的 teacher.expert_action（三竖带 + 中带 [0.4h,0.6h)）。
    权重与代码按任务卡下载后才能跑；这里只写接口，本次未执行。"""

    def __init__(self):
        import torch
        self.torch = torch
        sys.path.insert(0, f"{BASE}/depth-anything-v2/src/metric_depth")
        from depth_anything_v2.dpt import DepthAnythingV2   # 官方仓库自带，只依赖 torch/cv2/numpy
        cfg = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384],
               "max_depth": float(os.environ.get("DA2_MAX_DEPTH", "20"))}   # 室内 20 m、室外 80 m（metric_depth/README）
        self.model = DepthAnythingV2(**cfg)
        self.model.load_state_dict(torch.load(os.environ["DA2_WEIGHTS"], map_location="cpu"))
        self.model = self.model.to(DEVICE).eval()
        sys.path.insert(0, f"{REPO}/sim/tools")
        from teacher import expert_action                   # 只读 import，不改仓库
        self.expert, self.prev = expert_action, None

    def step(self, frames, prev_action):
        import cv2
        bgr = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        d = self.model.infer_image(bgr, 266)                # 输入短边 266（14 的倍数，接近 240）
        d = cv2.resize(d, (160, 120), interpolation=cv2.INTER_AREA)
        a = self.expert(d.reshape(-1).tolist(), 160, 120, turn_dist=DEPTH_TURN_DIST,
                        emergency=DEPTH_EMERGENCY, prev_action=self.prev)
        self.prev = a
        return a, {"d_center_min": float(d[48:72, 53:107].min())}


REGISTRY = {"forward": Forward, "pulp_v3": lambda: PulpDronetV3("v3"), "pulp_tiny": lambda: PulpDronetV3("tiny"),
            "dronet": DroNetKeras, "trailnet": TrailNet, "da2_rule": Da2Rule}


# ------------------------------ 协议循环 ------------------------------
def main():
    args = sys.argv[1:]
    ckpt = text = None
    while args:
        k = args.pop(0)
        if k == "--ckpt" and args:
            ckpt = args.pop(0)
        elif k == "--text" and args:
            text = args.pop(0)     # 指令 token id：像素基线忽略；VLM 基线可在这里反查指令文本
        else:
            sys.stderr.write("用法: baseline_infer.py --ckpt <file> [--text id,id,...]\n")
            return 2
    if not ckpt:
        sys.stderr.write("缺 --ckpt\n")
        return 2
    # stdout 只许协议行：保留真 stdout 的副本，再把 fd 1 指向 stderr，第三方库的 print 不会污染协议。
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    try:
        ad = REGISTRY[BASELINE]()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"baseline_infer: 构造 {BASELINE} 失败: {type(e).__name__}: {e}\n")
        return 1
    hello = {"in_h": ad.in_h, "in_w": ad.in_w, "in_c": ad.in_c, "num_classes": 4, "text_dim": 0,
             "prev_action_dim": ad.prev_dim, "aux_classes": 0}
    out.write((json.dumps(hello) + "\n").encode())
    import numpy as np
    one = ad.in_h * ad.in_w * 3
    payload = one * (ad.in_c // 3) + (1 if ad.prev_dim else 0)
    log = open(BASELINE_LOG, "a") if BASELINE_LOG else None
    n = 0
    stdin = sys.stdin.buffer
    while True:
        buf = stdin.read(payload)      # BufferedReader 在管道上会读满 payload 或遇 EOF
        if not buf:
            break                      # 干净 EOF
        if len(buf) != payload:
            sys.stderr.write("baseline_infer: 帧数据不足（截断）\n")
            return 1
        frames = [np.frombuffer(buf, np.uint8, one, one * i).reshape(ad.in_h, ad.in_w, 3) for i in range(ad.in_c // 3)]
        prev = buf[-1] if ad.prev_dim else 0
        t0 = time.perf_counter()
        a, raw = ad.step(frames, prev)
        ms = (time.perf_counter() - t0) * 1e3
        probs = [1.0 if i == a else 0.0 for i in range(4)]
        out.write((json.dumps({"action": int(a), "probs": probs, "infer_ms": ms, "raw": raw}) + "\n").encode())
        if log:
            log.write(json.dumps({"n": n, "action": int(a), "ms": ms, **raw}) + "\n")
            log.flush()
        n += 1
    sys.stderr.write(f"baseline_infer[{BASELINE}]: {n} 帧\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## 附录 B：启动器与自测脚本

启动器 `launch_pulp_v3.sh`（`launch_forward.sh`、`launch_pulp_tiny.sh`、`launch_dronet.sh` 只差 `BASELINE` 一行；已用 `SELFTEST_BIN` 验证环境变量覆盖生效）：

```bash
#!/bin/bash
# 启动器：把基线名与设备写死，再交给包装器（--infer-bin 只认一个可执行文件，不能带环境变量）
export BASELINE=pulp_v3 BASELINE_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
exec /root/rivermind-data/uav_rgb/tmp/phase12-20261110/baselines/baseline_infer.py "$@"
```

TrailNet 启动器 `run_trailnet.sh`（换成 `venv-cv4` 的解释器）：

```bash
#!/bin/bash
# 启动器：TrailNet 要 OpenCV 4.x 的 Caffe 导入器（cv2 5.0 已删），用数据盘上单独的 venv-cv4 解释器跑同一个包装。
# 这也是「每个基线一个解释器」的通用写法：--infer-bin 指向这种一行启动器即可。
exec /root/rivermind-data/uav_rgb/baselines/venv-cv4/bin/python /root/rivermind-data/uav_rgb/tmp/phase12-20261110/baselines/baseline_infer.py "$@"
```

协议自测 `proto_selftest.py`（只读 import 仓库 `student.py`）：

```python
#!/usr/bin/env python3
"""用仓库的 StudentPolicy（只读 import）拉起包装，走真实协议；帧取自闭环渲染。"""
import base64, json, os, sys, time
sys.dont_write_bytecode = True
R = "/root/claude-code-backup-20261028/code/uav_rgb"
sys.path.insert(0, f"{R}/sim/tools")
from PIL import Image
import numpy as np
import student as stu
HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = ["gz_pillars_113_f05.png", "gz_pillars_113_f12.png", "3dgs_stump_r5_f20.png"]
def msg_of(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    h, w = rgb.shape[:2]
    return {"width": w, "height": h, "step": w * 3, "pixelFormatType": "RGB_INT8",
            "data": base64.b64encode(np.ascontiguousarray(rgb).tobytes()).decode()}
name = sys.argv[1]
os.environ["BASELINE"] = name
log = f"{HERE}/selftest_{name}.jsonl"
if os.path.exists(log): os.remove(log)
os.environ["BASELINE_LOG"] = log
t0 = time.time()
p = stu.StudentPolicy(f"{HERE}/stub.ckpt", os.environ.get("SELFTEST_BIN", f"{HERE}/baseline_infer.py"))
print(f"[{name}] hello ok: in_h={p.h} in_w={p.w} in_c={p.in_c} prev_dim={p.prev_dim} text_dim={p.text_dim} 构造+warmup {time.time()-t0:.2f}s")
for f in FRAMES:
    a, ms, probs = p.act(msg_of(f"{HERE}/frames/{f}"))
    print(f"[{name}] {f}: action={a} infer_ms={ms:.2f} probs={probs}")
p.reset(); p.close()
print(f"[{name}] close rc ok; raw log:")
for l in open(log): print("   ", l.strip())
```

## 附录 C：自测原始日志 `selftest_all.txt`（原样）

```text
[forward] hello ok: in_h=240 in_w=320 in_c=3 prev_dim=0 text_dim=0 构造+warmup 0.38s
[forward] gz_pillars_113_f05.png: action=1 infer_ms=0.00 probs=[0.0, 1.0, 0.0, 0.0]
[forward] gz_pillars_113_f12.png: action=1 infer_ms=0.00 probs=[0.0, 1.0, 0.0, 0.0]
[forward] 3dgs_stump_r5_f20.png: action=1 infer_ms=0.00 probs=[0.0, 1.0, 0.0, 0.0]
[forward] close rc ok; raw log:
    {"n": 0, "action": 1, "ms": 0.00224309042096138}
    {"n": 1, "action": 1, "ms": 0.002017943188548088}
    {"n": 2, "action": 1, "ms": 0.002186046913266182}
    {"n": 3, "action": 1, "ms": 0.0015809200704097748}
[pulp_tiny] hello ok: in_h=240 in_w=320 in_c=3 prev_dim=0 text_dim=0 构造+warmup 3.25s
[pulp_tiny] gz_pillars_113_f05.png: action=1 infer_ms=5.18 probs=[0.0, 1.0, 0.0, 0.0]
[pulp_tiny] gz_pillars_113_f12.png: action=0 infer_ms=4.67 probs=[1.0, 0.0, 0.0, 0.0]
[pulp_tiny] 3dgs_stump_r5_f20.png: action=0 infer_ms=4.54 probs=[1.0, 0.0, 0.0, 0.0]
[pulp_tiny] close rc ok; raw log:
    {"n": 0, "action": 1, "ms": 30.206670984625816, "steer": -0.009633742272853851, "p_coll": 0.690805196762085}
    {"n": 1, "action": 1, "ms": 5.182059947401285, "steer": -0.005889363586902618, "p_coll": 0.6565614342689514}
    {"n": 2, "action": 0, "ms": 4.668717971071601, "steer": 0.08861574530601501, "p_coll": 0.7918466329574585}
    {"n": 3, "action": 0, "ms": 4.544934956356883, "steer": 0.016752317547798157, "p_coll": 0.8363310694694519}
[pulp_v3] hello ok: in_h=240 in_w=320 in_c=3 prev_dim=0 text_dim=0 构造+warmup 3.09s
[pulp_v3] gz_pillars_113_f05.png: action=1 infer_ms=9.62 probs=[0.0, 1.0, 0.0, 0.0]
[pulp_v3] gz_pillars_113_f12.png: action=2 infer_ms=7.93 probs=[0.0, 0.0, 1.0, 0.0]
[pulp_v3] 3dgs_stump_r5_f20.png: action=1 infer_ms=8.30 probs=[0.0, 1.0, 0.0, 0.0]
[pulp_v3] close rc ok; raw log:
    {"n": 0, "action": 1, "ms": 27.639754116535187, "steer": -0.06683279573917389, "p_coll": 0.437635600566864}
    {"n": 1, "action": 1, "ms": 9.616237133741379, "steer": 0.024598784744739532, "p_coll": 0.11544162034988403}
    {"n": 2, "action": 2, "ms": 7.932533975690603, "steer": 0.1408897489309311, "p_coll": 0.9869012236595154}
    {"n": 3, "action": 1, "ms": 8.297734195366502, "steer": -0.07368950545787811, "p_coll": 0.033180367201566696}
[dronet] hello ok: in_h=240 in_w=320 in_c=3 prev_dim=0 text_dim=0 构造+warmup 2.56s
[dronet] gz_pillars_113_f05.png: action=2 infer_ms=7.70 probs=[0.0, 0.0, 1.0, 0.0]
[dronet] gz_pillars_113_f12.png: action=0 infer_ms=7.86 probs=[1.0, 0.0, 0.0, 0.0]
[dronet] 3dgs_stump_r5_f20.png: action=0 infer_ms=7.20 probs=[1.0, 0.0, 0.0, 0.0]
[dronet] close rc ok; raw log:
    {"n": 0, "action": 0, "ms": 35.99577094428241, "steer": 0.17610707879066467, "p_coll": 0.8958572149276733}
    {"n": 1, "action": 2, "ms": 7.699108915403485, "steer": 0.6421443819999695, "p_coll": 0.9999998807907104}
    {"n": 2, "action": 0, "ms": 7.860469166189432, "steer": 0.05508089065551758, "p_coll": 0.9993959665298462}
    {"n": 3, "action": 0, "ms": 7.202883018180728, "steer": 0.025959353893995285, "p_coll": 0.895844578742981}
[trailnet] hello ok: in_h=240 in_w=320 in_c=3 prev_dim=0 text_dim=0 构造+warmup 0.53s
[trailnet] gz_pillars_113_f05.png: action=0 infer_ms=19.41 probs=[1.0, 0.0, 0.0, 0.0]
[trailnet] gz_pillars_113_f12.png: action=0 infer_ms=22.56 probs=[1.0, 0.0, 0.0, 0.0]
[trailnet] 3dgs_stump_r5_f20.png: action=0 infer_ms=98.13 probs=[1.0, 0.0, 0.0, 0.0]
[trailnet] close rc ok; raw log:
    {"n": 0, "action": 0, "ms": 90.58418893255293, "out6": [0.1390797346830368, 0.7955422401428223, 0.06537798047065735, 0.14417067170143127, 0.7719128727912903, 0.08391648530960083]}
    {"n": 1, "action": 0, "ms": 19.406083971261978, "out6": [0.5599475502967834, 0.0026303771883249283, 0.4374220073223114, 0.18148082494735718, 0.7945898771286011, 0.023929303511977196]}
    {"n": 2, "action": 0, "ms": 22.557338932529092, "out6": [0.026797305792570114, 0.0025347501505166292, 0.9706678986549377, 0.27044743299484253, 0.44425660371780396, 0.28529593348503113]}
    {"n": 3, "action": 0, "ms": 98.1305141467601, "out6": [0.5853778123855591, 0.39478763937950134, 0.019834544509649277, 0.10331033170223236, 0.8746175169944763, 0.022072121500968933]}
```

参数统计 `count_params.out`（原样）：

```text
nemo imported successfully
pulp_v3: params=320928 MAC=41103104 input=(1,200,200) fp32_MB=1.284 int8_kB=320.9
nemo imported successfully
pulp_tiny: params=2784 MAC=1130444 input=(1,200,200) fp32_MB=0.011 int8_kB=2.8
dronet npz: tensors=48 total_numel(含BN滑动均值方差)=321634
```

## 附录 D：草稿目录文件索引

| 文件 | 字节 |
|---|---|
| `arxiv_2312.08991.xml` | 0 |
| `arxiv_2407.12675.xml` | 0 |
| `arxiv_titles.log` | 5 |
| `arxiv_titles.py` | 3617 |
| `baseline_infer.py` | 17747 |
| `build_report.py` | 4030 |
| `count_params.out` | 282 |
| `count_params.py` | 1763 |
| `crosscheck.py` | 1525 |
| `fetch_gh_meta.log` | 6163 |
| `fetch_gh_meta.py` | 2739 |
| `fetch_gh_meta2.log` | 1783 |
| `launch_dronet.sh` | 301 |
| `launch_forward.sh` | 302 |
| `launch_pulp_tiny.sh` | 304 |
| `launch_pulp_v3.sh` | 302 |
| `mirror_probe.tsv` | 2873 |
| `prep_sensitivity.txt` | 3709 |
| `probe_hf.log` | 3914 |
| `probe_hf.py` | 3176 |
| `probe_mirrors.log` | 0 |
| `probe_mirrors.sh` | 1578 |
| `proto_selftest.py` | 1426 |
| `retry_hf_tree.log` | 5928 |
| `retry_hf_tree.py` | 1740 |
| `run_trailnet.sh` | 415 |
| `selftest_all.txt` | 4168 |
| `selftest_dronet.jsonl` | 411 |
| `selftest_forward.jsonl` | 432 |
| `selftest_pulp_tiny.jsonl` | 432 |
| `selftest_pulp_v3.jsonl` | 430 |
| `selftest_trailnet.jsonl` | 732 |
| `speed_test.log` | 230 |
| `speed_test.sh` | 426 |
| `stub.ckpt` | 66 |
| `arxiv/abs_2312.08991.html` | 44317 |
| `arxiv/abs_2407.02405.html` | 42421 |
| `arxiv/abs_2407.12675.html` | 42765 |
| `arxiv/abs_2607.12593.html` | 44804 |
| `arxiv/html_2312.08991.html` | 194484 |
| `arxiv/pdf_2312.08991.pdf` | 522447 |
| `arxiv/pdf_2407.12675.pdf` | 413101 |
| `arxiv/text_2312.08991.clean.txt` | 49522 |
| `arxiv/text_2312.08991.txt` | 53642 |
| `arxiv/titles.jsonl` | 39330 |
| `frames/3dgs_stump_r5_f20.png` | 198738 |
| `frames/gz_pillars_113_f05.png` | 4896 |
| `frames/gz_pillars_113_f12.png` | 4554 |
| `gh/AnjieCheng__NaVILA.README.md` | 7754 |
| `gh/DepthAnything__DA2-metric.README.md` | 4621 |
| `gh/DepthAnything__Depth-Anything-V2.README.md` | 9904 |
| `gh/HKUST-Aerial-Robotics__Fast-Planner.README.md` | 13183 |
| `gh/HenryHuYu__DiffPhysDrone.README.md` | 2201 |
| `gh/NVIDIA-AI-IOT__redtail.README.md` | 6246 |
| `gh/OpenRobotLab__StreamVLN.README.md` | 15017 |
| `gh/SHAILAB-IPEC__OpenFly-Platform.README.md` | 17684 |
| `gh/TJU-Aerial-Robotics__YOPO.README.md` | 10485 |
| `gh/YvanYin__Metric3D.README.md` | 24738 |
| `gh/ZJU-FAST-Lab__EGO-Planner-v2.README.md` | 794 |
| `gh/ZJU-FAST-Lab__ego-planner.README.md` | 11027 |
| `gh/Zhefan-Xu__NavRL.README.md` | 12082 |
| `gh/apple__ml-depth-pro.README.md` | 3776 |
| `gh/buaa-colalab__UAV-Flow.README.md` | 5572 |
| `gh/gemenerik__gap8-obstacle-avoidance.README.md` | 2228 |
| `gh/isl-org__ZoeDepth.README.md` | 8385 |
| `gh/jzhzhang__NaVid-VLN-CE.README.md` | 8837 |
| `gh/jzhzhang__Uni-NaVid.README.md` | 6340 |
| `gh/lpiccinelli-eth__UniDepth.README.md` | 12096 |
| `gh/natesimon__MonoNav.README.md` | 16161 |
| `gh/summary.tsv` | 3174 |
| `gh/tudelft__mavrl.README.md` | 8052 |
| `gh/uzh-rpg__agile_autonomy.README.md` | 8660 |
| `gh/uzh-rpg__rpg_public_dronet.README.md` | 6121 |
| `hf/summary.txt` | 4200 |
| `hf/summary_retry.txt` | 6228 |
| `sections/s0_summary.md` | 4065 |
| `sections/s1_protocol.md` | 7025 |
| `sections/s2_adapter.md` | 10188 |
| `sections/s3_candidates.md` | 18337 |
| `sections/s4_local.md` | 10073 |
| `sections/s5_conclusions.md` | 17509 |
| `sections/s9_end.md` | 7689 |

（`gh/*.json`、`hf/*.json` 是各接口的原始返回，数量多，未逐个列出；`speed/` 是测速时落下的残缺 zip，未列出。）

## 做了什么

1. 通读公共任务书与调用链代码（`student.py`、`infer.cpp`、`infer_main.cpp`、`collect.py` student 分支、`eval_closed_loop.py`、`collect_3dgs.py`、`eval_3dgs.py`、`teacher.py`），写出线协议、字段、超时与错误处理（§1）。
2. 设计并写了外部基线适配器：`baseline_infer.py`（含 6 个后端：`forward`、`pulp_v3`、`pulp_tiny`、`dronet`、`trailnet`、`da2_rule` 骨架）、`proto_selftest.py`、启动器 `run_trailnet.sh`；给出输出到 4 个动作的映射与每个阈值的出处（§2）。
3. 下载第一档的代码与权重到数据盘：DroNet、PULP-Dronet v2/v3/Tiny v3、TrailNet；为 `.h5` 与 Caffe 模型新建两个数据盘 venv；DroNet 权重转 `.npz`（§4.2、§4.3）。
4. 用真实的 `StudentPolicy` 对 5 个后端做端到端自测（hello、预热、3 次 `act()`、`reset()`、`close()`），实测 PULP 两个变体的参数量与 MAC（§4.4，附录 C）。
5. 逐项核实第一至第四档候选的仓库、许可证、权重体积、输入输出、依赖与论文摘要数字：读 24 份 README（`gh/*.README.md`），用 GitHub API、hf-mirror、ModelScope 取体积，用 arXiv API 按标题核对论文编号（§3）。
6. 探测直连与 13 个 GitHub 镜像前缀，测大文件下载速度，确认 hf-mirror 与 ModelScope 可达（§4.1）。
7. 给出 5 个基线的优先级与任务卡、只引用不跑的表、对比实验总设计与统计设计（§5），列出需要用户决定的事（§6）。

## 没做什么（每条带原因）

1. 没跑任何基线的 Gazebo 或 3DGS 闭环，也没跑 `eval_closed_loop.py` → `collect.py` → `StudentPolicy` 的整条链。原因：本任务是选型与接入设计，时间盒 60 分钟，闭环评测要占用仿真与渲染资源；环境变量沿这条链继承只从 `gz_env` 的 `dict(os.environ)` 读代码推断，没有端到端验证。
2. 没下载第二、三档任何权重，只做体积与可达性探测。原因：任务书只允许第一档（单个小于 100 MB）下载，且禁止未经询问下载超过 500 MB。因此第二档的 GPU 延迟与显存没有实测。
3. 没完成并没运行 `da2_rule` 适配器，骨架里的 import 路径与输入尺寸未经执行验证。原因：DA-V2 权重与代码没有下载。
4. 没把 DroNet 的 NumPy 前向与 Keras 原模型做数值对拍。原因：本机没有 TF 1.x，规则禁止往现有 venv 装包，新建 TF venv 体积大且版本老，超出时间盒。
5. 没做转向符号的镜像帧检验（左右翻转输入看 steer 是否变号），`STEER_SIGN` 是否取 +1 仍未核实，DroNet 的符号约定尤其没核实。原因：时间；已在任务卡里改成开发 seed 上校准。
6. 没做 `BASELINE_PREP` 三种取法的闭环敏感性，只在 3 帧上跑过（§4.4 第 6 条）。原因：需要闭环。
7. 没核实 SG2002 的 TPU-MLIR 算子支持与 int8 量化精度。原因：没有查工具链文档。
8. 没读 DroNet 论文（RA-L 2018）的数字，没读 VLN/VLA 论文正文里的闭环成绩，只引用了 arXiv 摘要里写明的数字。原因：时间；摘要里没有的数字一律标"未核实"。
9. 没核实 Lamberti 2023 框架中阈值 T 的取值，也没核实是否存在正文未提及的代码仓库（只做了一次 GitHub 关键词搜索，返回 0 条，证据弱）。
10. 没读 NanoFlowNet、2607.12593、`idsia-robotics/ultralow-power-monocular-depth-ondevice-learning` 的代码与权重，没读 YOPO 以外的第四档论文数字，没取 YOPO 权重体积（GitHub API 的 tree 请求超时）。原因：时间。
11. 没核实 PULP-Dronet v2 的前向（模型文件依赖 NEMO/DORY 量化流程），只下载了文件。
12. 没验证 `BASELINE_DEVICE=cuda` 路径（第一档在 CPU 上毫秒级，没有必要）。
13. 没把包装器并入仓库。原因：任务要求不改仓库；是否并入留给主会话（§6）。
14. 没核实 3 张自测帧的原始路径（它们是此前从已有闭环产物复制来的）。
15. 没留存 `git clone` 直连的日志。原因：当时只是口头观察，我终止了该进程；§4.1 已标"未核实"。
16. 部分 hf-mirror 模型库的 `tree` 请求被限流（429），ONNX 版 Metric3D 的体积没取到；IndoorUAV 之外的 ModelScope 库没有探测。

## 验证到哪一步

| 命令或检查项 | 结果 |
|---|---|
| `$R/.venv-tools/bin/python proto_selftest.py forward / pulp_tiny / pulp_v3 / dronet`（CPU，经真实 `StudentPolicy`） | 4 个全部通过 hello 校验、预热、3 次 `act()`、`reset()`、`close()`，退出码 0（`selftest_all.txt` 中 `close rc ok` 共 5 行含 TrailNet） |
| `SELFTEST_BIN=run_trailnet.sh proto_selftest.py trailnet`（`venv-cv4`，OpenCV 4.10） | 通过；输出 6 维，两组 softmax 各组和为 1.000（附录 C） |
| `teacher-venv` 下 `cv2.dnn.readNetFromCaffe` | 失败：OpenCV 5.0.0 没有该导入器（`AttributeError`），改用 `venv-cv4` 后通过 |
| `count_params.py`（`count_params.out`） | PULP v3：320,928 参数、41,103,104 MAC；Tiny：2,784 参数、1,130,444 MAC（输入 200×200） |
| `BASELINE_PREP=resize_crop` 与 `squash` 各跑 `pulp_v3`、`pulp_tiny`、`dronet`（3 帧，CPU） | 全部通过；同一帧换取法 steer、p、动作都会变（§4.4 第 6 条，`prep_sensitivity.txt`） |
| `launch_pulp_v3.sh` 作为 `--infer-bin`，自测传入 `forward` | 日志出现 PULP v3 的 steer 字段，说明启动器写死的环境变量覆盖生效 |
| `BASELINE_WEIGHTS` 指向官方 v3 权重重跑 `pulp_v3` | 通过，动作序列 1、2、1，与默认路径一致 |
| 镜像探测 `probe_mirrors.sh`，速度 `speed_test.sh` | `ghfast.top`、`gh-proxy.com`、`ghproxy.net` 可用；`github.xxlab.tech` 的 raw 请求返回 302，未跟进；其余 9 个前缀失败或限流；`ghfast.top` 8 s 内 9,222,989 B |
| arXiv 标题检索（`arxiv_titles.py` 与二次检索 `titles2.json`） | 29 个标题中 25 个首次命中；Metric3Dv2 二次检索命中（2404.15506）；YOPO、Swarm 论文、Nano Quadcopter 轻量单目深度网络三个未命中，论文编号未核实 |
| GitHub / hf-mirror / ModelScope 元数据 | `gh/summary.tsv` 25 个仓库行；`hf/*.json` 31 个文件；ModelScope `IndoorUAV-Agent` 一个 zip 9,468,831,797 B |
| 仓库是否被本任务改动 | 本任务所有命令对仓库只读，`find $R -name __pycache__ -newermt 19:21` 为空。仓库里晚于 19:21 的改动（`sim/tools/collect.py`、`eval_closed_loop.py`、`test_collect.py`、`test_heading_guard.py`、`experiments/2026-11-10-swa-calib/*`、`.git`）来自并行任务，其中前四个文件的时间戳与合并提交 `155e377`（19:40:48）一致；我无法证明其余文件不是我写的，只能说我没有对这些路径做过写操作 |
| 数据盘占用 | 新增约 437 MB（`du -sh`），远低于 3 GB；`df` 剩余 15 GB |
| DroNet 数值对拍、转向符号、闭环成绩、第二档前向、第三档加载 | 未验证，见「没做什么」 |
| `crosscheck.py`：§3 的 MB 数字、正文 arXiv 编号、日期对原始文件 | §3 的 20 个 MB 数字都能在 `hf/` 原始返回里找到字面值；正文 25 个 arXiv 编号都在 `arxiv/` 原始文件里出现；日期里只有任务日期与文件时间戳不在 `gh/summary.tsv`。其余数字（README 行号、闭环帧输出、时延）靠写作时逐条对照命令输出，没有做自动校验 |
| 报告文字规则扫描（禁用词、破折号） | `build_report.py` 对正文与代码块各扫一遍，命中数见本节末尾"自检"一行；"每个数字都有出处"没有自动检查手段，靠人工核对 |

自检（`build_report.py`，2026-11-10 19:56:44）：正文禁用词与破折号命中 0 处；代码块内命中 0 处。行数 1045。
