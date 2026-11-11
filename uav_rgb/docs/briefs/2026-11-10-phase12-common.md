# 第十二阶段公共规则（执行任务卡之前先读完）

- 立：2026-11-10，主会话。适用于按 `docs/briefs/2026-11-10-phase12-task-cards.md` 执行任务的任何模型或人。
- 与其它文档的关系：`CLAUDE.md` 管环境与不可违反的边界，`docs/ENGINEERING_RULES.md` 管工程细则，`docs/EVAL_STANDARD.md` 管报数口径。
  本文把执行时最常用的规则集中在一处；有冲突时以那三份为准。

## 1. 项目是什么

- `uav_rgb` / TinyAerialVLN：一架约 100 g 的琛光 E1 MINI 四轴（ESP32 飞控跑开源 CF-Drone 固件，机载 LicheeRV Nano SG2002，int8 NPU），
  只靠单目 RGB（模型输入 160×120）和一句短指令，在室内与校园里避障到达目标。
- 学生模型：自研 C++/CUDA 小 CNN（约 10 万参数），输出 4 个离散高层动作：`SAFE_NEUTRAL(0)`、`FORWARD_SLOW(1)`、`YAW_LEFT(2)`、`YAW_RIGHT(3)`，
  每个动作保持 1.0 s（Gazebo 里每步前进约 0.25 m 或转约 26°）。
- 两条仿真线，同一套固件在环动力学（`sim/cf_sitl/`）：
  - Gazebo Harmonic（软件渲染，约 2 s 一步）：柱阵 `--task base`、长课程 `--task course_base --blocks K`、森林 `--task forest_e1`、带指令的 `l1` / `campus`。
    入口 `sim/tools/collect.py`、`sim/tools/eval_closed_loop.py`。
  - 3DGS（gsplat，CUDA 渲染真实场景，一集 1–3 s）：`sim/3dgs/`，场景在 `$D/3dgs-scenes/`。入口 `sim/3dgs/collect_3dgs.py`、`sim/3dgs/eval_3dgs.py`。
- 教师：特权 A*（看得到全部障碍与位姿）。学生模仿教师。方案是「可判定性分解」：图像决定得了的量（各列可通行距离）由网络学，
  图像决定不了的量（航向、目标方位、进度）走条件输入或执行器，见 `docs/PLAN_2026-11-10_phase12.md` §4。

## 2. 固定口径

- 柱阵评测：`metric_version = sim09.v2.coast1s.corridor3m`，成功 = 无碰撞、过成功线（最远柱 x + 1.0）、全程 |y| ≤ 3 m（`--corridor-half-width 3.0`）。
- 20 个冻结评测 seed：`1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654`。开发与调参只用别的 seed（建议 2001–2010 或 train 分区）。
- 主比较：20 个几何 × 每方法 3 个训练种子（60 集），配对精确 McNemar 加 Wilson 区间；长课程另报逐块存活。单次 n=20 的比较不下结论。
- 交付权重：末 10 轮权重平均加 FORWARD 先验校准（`tools/average_ckpt.py`），不是 `ckpt_best`。
- 航向护栏 `--heading-guard 2` 是评测端规则，用了就在臂名里写明；对比别的方法时，对方也加一份。
- 用了仿真真值的策略或规则，名字里必须带 `oracle`，不能当成学生的能力报告。

## 3. 路径（用户硬性要求：代码进仓库，数据与实验结果进数据盘）

- 主仓库：`R=/root/claude-code-backup-20261028/code/uav_rgb`，分支 main，推 GitHub 用 `bash tools/git_mirror_alvln.sh`（只有用户或主会话做）。
- 执行任务时不在主工作树里改文件。每张卡一个工作树：`/root/claude-code-backup-20261028/code/wt/p12-<名>`，分支同名，基于 main。
  代码、测试、实验入口脚本（`experiments/<日期>-<slug>/`）、文档都写在工作树里并在自己的分支上提交。不合并，不推送。
- 数据盘：`D=/root/rivermind-data/uav_rgb`。数据集、权重、实验输出（`$D/exp-<slug>-<日期>/`）、日志、下载物、构建目录（`$D/tmp/build-p12-<名>`）、
  草稿、venv、第三方基线的代码与权重（`$D/baselines/`）都放这里，不进仓库。
- 根分区只剩约 2.4 GB：不往 `/tmp`、`/root`、仓库目录写大文件；`TMPDIR` 指到数据盘；`PYTHONDONTWRITEBYTECODE=1`。
  数据盘只剩约 11 GB（11-11 核实）：新任务先估体积，超过 3 GB 先停下报告。不删除任何已有文件、目录、分支、worktree。

## 4. 环境

- Linux 容器。资源配置会变，每次开工先看：`nvidia-smi -L`、`cat /sys/fs/cgroup/cpu.max`（配额 ÷ 100000 = 核数）、`cat /sys/fs/cgroup/memory.max`。
  `/proc/loadavg` 反映的是宿主机（容器里什么都没跑时也有 13–25），容器自己忙不忙看 `/sys/fs/cgroup/cpu.stat` 的 `usage_usec` 与 `nr_throttled` 的增量。
  - 2026-11-10 晚：24 核，两张 RTX 2080 Ti，uav_rgb 只用 `GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70`，另一张留给别的项目。当晚的实验都在这个配置下。
  - 2026-11-11 19:23 起：12 核，内存上限 62 GB，只有一张卡 `GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740`。用户指示 uav_rgb 就放在这一张卡上跑：
    用 CUDA 的进程带 `CUDA_VISIBLE_DEVICES=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740`（按旧 UUID 钉卡时看不到设备）。别的项目的训练也可能落在这张卡上，
    起实验前看显存占用。11-10 的入口脚本里写死的旧 UUID 是冻结记录，不改；新入口脚本的 GPU 写在配置区，一行，注明日期与来历。
  - 卡的配置再变时，以 `nvidia-smi -L` 为准并向用户确认用哪一张，不要自己挑。
- Python：数据工具 `$R/.venv-tools/bin/python`（无 torch）；3DGS `$D/3dgs-venv/bin/python`（torch + gsplat）；`$D/teacher-venv` 也有 torch，
  只给对照基线与教师工具用。主线训练推理禁用 PyTorch。不往现有 venv 里装包。
- Gazebo 只能软件渲染：`export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg`。gz 实例的上限随核数定：24 核时全机不超过 7 个，12 核时不超过 3 个；起之前先看占用。
  对墙钟敏感的实验（带 `--action-latency` 的标定）独占机器：用机器锁 `$D/tmp/phase12-cards/machine.lock`（入口脚本里 `exec 8>> 锁文件; flock 8`），
  别的重实验拿同一把锁排队。
- 推理二进制快照：`$D/bin-snapshots/main-97ae04b/uavrgb_infer`。SITL 库：`$D/tmp/build-cf-sitl-lib31`（指纹 `cf_sitl_v1:1bc8fedf:lib31`，不要动这个目录）。
- 电量：SITL 每集从 50% 电量起飞，一集最多约 100 步（hold 1.0）。更长的任务要满电档（任务卡 W1-10），指纹会变，数据不与现有数据混包。
- 网络：`hf-mirror.com` 与 arXiv 可用；GitHub 大文件走 `https://ghfast.top/<原 URL>`，次选 `https://gh-proxy.com/`；`huggingface.co` 直连超时。
  下载前先 HEAD 看体积，单个超过 500 MB 的先停下报告。
- shell 里没有 `rg`，用 `grep`。不装系统包，不动驱动。

## 5. 代码规则

- 先读代码再改，不猜接口；改之前 grep 所有调用方；只改任务需要的，不顺手重构；`ui/` 不碰。
- 新开关默认关闭时行为逐位不变，用测试锁住。每条新逻辑一个先红后绿的测试，报告里贴改前失败与改后通过的真实输出。
- 复用仓库已有的函数与开关；不加依赖；不加没人要的抽象与配置。已知限制写 `shortcut: <限制>, <何时升级>` 注释。
- 注释只写代码看不出的「为什么」，中文。入口脚本的超参数集中在文件顶部配置区，一行一个，行尾写取值依据。
- C++ 单文件不超过 280 行，CUDA 不超过 360 行；只编 sm_75；模型只出 4 个离散动作；仿真真值不进学生输入。
- 改完跑相关的 `test_*.py`，再跑 `cd <工作树> && UAV_RGB_PYTHON=$R/.venv-tools/bin/python bash tools/check.sh --python-only`（约 3 分钟，必须 rc=0）。
  改了 C++ 才跑全量 `check.sh`（`UAV_RGB_BUILD_DIR=$D/tmp/build-p12-<名>`）。
- 提交：只在自己的分支上；`git add` 逐文件，不用 `git add -A`；说明用 `fix:` / `feat:` / `test:` / `docs:` 前缀，中文，一次一个主题。

## 6. 实验规则

- 超过几分钟的实验一律守护进程：`setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &`；脚本自己写 `run.pid`、追加 `run.log`、结束落 `DONE`（rc 加时间）；
  开头 `flock -n` 互斥。样板：`experiments/2026-11-10-heading-guard/run.sh`。启动后用 `ps -o pid,pgid,etime,cmd` 确认活着。
- 起跑前在入口脚本头部写预注册：预测、证伪条件、主比较（不超过 3 个）。闭环是裁判，离线指标只作旁证。参数在起跑前定死，不在评测 seed 上调。
- 不阻塞等待长实验：冒烟通过、正式实验起跑并确认前几集正常后就写报告返回，报告里写 PGID、日志、怎么看进度、怎么停、结束后跑哪条命令出汇总。
- 停进程只按 PGID：`kill -- -<PGID>`，禁止 `pkill -f`。结束前确认没有自己起的残留进程。
- 评测数字带 `metric_version`；分母用请求集数。

## 7. 交付

- 报告写中文，边做边写，防止会话中断丢结果。写作：不用「总之、综上所述、值得注意的是、让我们」；不用程度副词；不用破折号作同位语；
  数字都要有出处（命令、文件路径或 URL）；没核实的写「未核实」，没做的写「没做」并给原因。
- 报告末尾固定三节：「做了什么」「没做什么（每条带原因）」「验证到哪一步（表格：命令或检查项 / 结果）」。
- 做不到或与任务卡对不上：停下，写清现象与证据，不自行改设计、不改预注册。
