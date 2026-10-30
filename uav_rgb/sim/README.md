# sim/ —— E1 Gazebo 仿真脚手架

单目 RGB 无人机避障（uav_rgb）的 Gazebo Harmonic 最小仿真层：
一个四旋翼 + 前向 RGB 相机 + 彩色地标，可以 headless 启动、读/录相机话题、
用速度指令驱动四旋翼采轨迹。**只依赖 Gazebo 自带 CLI（`gz sim` / `gz topic` /
`gz service`）和一个 stdlib Python 脚本，不用 PyTorch / ROS 2。**

## 文件

- `worlds/uav_rgb.sdf` —— 世界：X3 等价四旋翼（inline，物理参数=官方
  OpenRobotics X3 UAV）+ 前向 RGB 相机（装于机体 `base_link`）+ 红/绿/蓝三个地标。
- `tools/save_frame.py` —— 把 `gz topic -e --json-output` 的一帧保存为 PNG（纯 stdlib）。
- `tools/scene_gen.py` —— 按 seed 生成随机障碍世界（只替换障碍段，同 seed 字节级一致）。
- `tools/collect_parallel.py` —— 多实例 `GZ_PARTITION` 隔离 + 并发 `collect.py` 的采集编排。
- `evidence/*.png` —— 实测抓帧证据。

## 0. 本容器关键事实（务必先读）

- **纯 headless 相机渲染跑不起来**。`gz sim -s -r`（不带 DISPLAY）时，
  Ogre2 传感器渲染走 EGL-PBuffer，会去初始化无驱动的 NVIDIA 设备
  `/dev/dri/card1`（容器没有 NVIDIA EGL），`eglInitialize failed` 后 segfault，
  整个 server 崩溃，`/uav/camera` 不发布。报错原文见文末「附录 A」。
- **workaround（已实测可用）**：server-only 模式显式带 `DISPLAY=:20`
  （容器里已有 Xorg 软件 GL，llvmpipe/GL 4.5）。Ogre2 在 EGL 失败后回落到
  X11/GLX 软件渲染，相机正常出图。所以命令一律写成：
  `DISPLAY=:20 gz sim -s -r <world>`。
- 物理本身（无相机）可以纯 headless 跑；只有相机传感器需要上面的 DISPLAY 工作区。
- **不要用 `gz sim -r`（GUI 模式）**：GUI 的 3D 视图渲染线程在 Ogre2 EGL 初始化
  时同样 segfault（见附录 A），整进程崩溃。用 `-s`（server-only）。

## 1. 启动世界

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb
mkdir -p /tmp/xdg && chmod 700 /tmp/xdg   # 消除 XDG_RUNTIME_DIR 报错
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg gz sim -s -r sim/worlds/uav_rgb.sdf
```

- `-s` server-only，`-r` 启动即运行（不暂停）。
- 确认加载成功：另开终端 `gz topic -l` 应看到 `/uav/camera`、`/model/x3/pose`、
  `/X3/gazebo/command/twist` 等（见文末话题表）。
- 停止：Ctrl-C；或用 `setsid` 起独立进程组后 `kill -- -PGID`。**不要用
  `pkill -f 'gz sim'`**：命令行会匹配到当前 shell 自身，可能把自己一起杀掉。
  **注意**：server 进程的 comm 是 `ruby`（`gz` 是 Ruby 调度脚本），
  `pkill -x gz` 杀不掉；核对用 `ps -eo pid,comm | grep ruby`。
  **手动调试时一次只起一个 server**：多个 gz sim 共用默认 transport 命名空间，
  会互相污染 `/model/x3/pose`、`/uav/contact` 等话题（实测：pose 话题多
  publisher、碰撞信号串台）。并行采集必须给每个实例不同的 `GZ_PARTITION`，
  见 4.6（实测分区内 25 个 topic，默认分区看不到别的实例的 `/uav/depth`）。

## 2. 读 RGB 相机（话题：`/uav/camera`，gz.msgs.Image，R8G8B8 320×240 @30Hz）

```bash
gz topic -i -t /uav/camera                    # 类型/频率
gz topic -e -t /uav/camera                    # 连续打印（含 base64 data）
gz topic -e -t /uav/camera -n 1 --json-output # 只取 1 帧，JSON（data 为 base64）
```

抓一帧存成 PNG（证据）：

```bash
gz topic -e -t /uav/camera -n 1 --json-output \
  | python3 sim/tools/save_frame.py - evidence/frame.png
# 或先存 JSON 再转： ... > /tmp/f.json && python3 sim/tools/save_frame.py /tmp/f.json out.png
```

`save_frame.py` 会在 stderr 打印 `320x240 3ch sim_time=<sec>.<nsec>`，
即观测时间戳（见下方「对齐 ObservationSpec」）。

## 3. 录 RGB（video-record 插件，已内置在相机 sensor 里）

```bash
# 开始录像
gz service -s /uav/camera/record_video \
  --reqtype gz.msgs.VideoRecord --reptype gz.msgs.Boolean --timeout 300 \
  --req 'start: true, format:"mp4", save_filename:"/tmp/fly.mp4"'
# ... 驱动四旋翼采一段轨迹 ...
# 停止录像
gz service -s /uav/camera/record_video \
  --reqtype gz.msgs.VideoRecord --reptype gz.msgs.Boolean --timeout 300 \
  --req 'stop: true'
```

也可用 gz 日志录制：`gz sim -s -r sim/worlds/uav_rgb.sdf --record-topic '/uav/camera' --record-path /tmp/logs`。

## 4. 驱动四旋翼采轨迹

速度控制器命令话题 `/X3/gazebo/command/twist`（机体坐标系 FLU：+x 前、+y 左、
+z 上；`angular.z` 为 yaw 角速度，rad/s）：

```bash
# 起飞/上升
gz topic -t "/X3/gazebo/command/twist" -m gz.msgs.Twist \
  -p "linear: {x: 0 y: 0 z: 0.5} angular: {z: 0}"
# 前进
gz topic -t "/X3/gazebo/command/twist" -m gz.msgs.Twist \
  -p "linear: {x: 0.3 y: 0 z: 0} angular: {z: 0}"
# 悬停（清空指令）
gz topic -t "/X3/gazebo/command/twist" -m gz.msgs.Twist -p " "
```

底层电机转速（跳过速度环，直接用官方 quadcopter 例子的接口）：

```bash
gz topic -t "/X3/gazebo/command/motor_speed" -m gz.msgs.Actuators \
  -p 'velocity:[700, 700, 700, 700]'
```

记录真值位姿/里程计（用于标定轨迹，**不得进入 student 输入**）：

```bash
gz topic -e -t /model/x3/pose       # gz.msgs.Pose
gz topic -e -t /model/x3/odometry   # gz.msgs.Odometry
```

注意：命令已发送 ≠ 已执行；轨迹真值以 `/model/x3/pose` 为准。相机在机头
（`frame_id=x3::base_link::camera`，随无人机运动）。

## 4.5 采集 episode（E3，碰撞真值已接）

```bash
# 专家策略采集（自动避障，通常无碰撞）
# 必须带 --scene-meta 才会用柱子真值的 carrot planner；不给会退回深度反应式专家，
# manifest 的 teacher 字段会如实记为 expert_action。
python3 sim/tools/collect.py --steps 40 --out /tmp/ep0 --scene-id 1 --seed 42 \
  --split train --scene-meta /tmp/scene_1.json
# 盲策略（恒定前进）采集：撞柱，验证 collision=True
python3 sim/tools/collect.py --steps 10 --hold 1.5 --policy forward \
  --out /tmp/ep1 --scene-id 1 --seed 43
```

输出 `episode_manifest.json`（episode_id/scene_id/seed/version/split + 相机内参）+
`actions.jsonl`（每步 timestamp_ns/action_expert/action_executed/collision/
rgb_ok/frame/rgb_ts/depth_ts/depth_rgb_gap_ns/pose_ts/pose_rgb_gap_ns/action_start_ns）+
`oracle.jsonl`（pose_gt 真值，只进 oracle）+ `rgb_%05d.png`（320×240 原图）。
碰撞=接触传感器（`/uav/contact`，稀疏话题：机体 + 四个旋翼碰撞体都挂了 contact
sensor，无接触时静默；常驻订阅按接收序号判窗口内是否收到非空 contact）或位姿
卡滞（前进指令下位移<3cm）兜底，任一触发。
**每次跑前用全新 world**（尚未实现 episode reset，残存指令会污染下一 episode）；
采集器会拒绝非空 `--out`，并在起飞前发现 z>0.5 m（world 未复位）时直接报错。
起飞按位姿闭环到 `Z_TARGET=2.1 m`，不再用固定 wall-clock。

I/O（2026-10-29 起）：`collect.py` 启动时给 camera/depth/pose/contact/camera_info
各起一个**常驻订阅**（`tools/gz_io.py`：一个 `gz topic -e --json-output` 进程 +
后台读线程 + latest-wins 环形缓冲），每步从缓冲取帧，不再每帧 spawn 一次
`gz topic -e -n 1`（本机约 250 ms/次）。每步以最新 RGB 为锚点，取 sim-time 最近的
深度、取晚于 RGB 的位姿（保证 M1），`actions.jsonl` 的 `depth_rgb_gap_ns` /
`pose_rgb_gap_ns` 是配对残差（实测 |rgb-depth| 中位数 0 ms、pose 中位数 12 ms，
旧版深度/RGB 残差 200–300 ms）。内参取自 `/uav/camera_info` 的 RGB 那条
（`frame_id=x3::base_link::camera`），manifest 里 `camera_source="gazebo_camera_info"`。

指令发布（2026-10-29 起）：`collect.py` 启动时建一个**常驻 twist publisher**
（`tools/gz_io.py` 的 `open_twist_publisher`）。当前解释器能 `import gz.transport13`
时在进程内 advertise `/X3/gazebo/command/twist`；不能时自动改用 `/usr/bin/python3`
起一个常驻 helper 子进程发布（`UAVRGB_GZ_PYTHON` 可覆盖解释器），并在 stderr
打一行警告。两者都建不起来时 `collect.py` **直接失败**，不回退 `gz topic -p` CLI：
CLI 每条固定 ~1 s，动作窗口会失真；CLI 只保留给 finally 的零速安全网。
常驻通道每条指令是内存级 publish（实测 <0.1 ms）；`--hold` 到点立刻发零速并等它
发布完成，不会因 hold 早于发布完成而丢零速。
实测 20 步（hold=0.5）：每步 wall **0.515–0.530 s**（I/O=0.015–0.029 s、
动作窗口=0.500 s、发布≈0.000 s），20/20 步 ≤0.59 s，M1 20/20。旧版 CLI 动作发布
是 1.50 s/步、发布开销 0.995 s（`gz topic -p` 在本机固定 `nanosleep(0, 800000000)`
后再发布，见 worklog）；该路径已从 `collect.py` 删除。
新鲜度门禁（D4）：depth 与 RGB 的 sim-time 差 >100 ms，或 contact 订阅进程退出，
该集直接失败；contact 是稀疏话题，无消息不等于停更。

时间口径（别混）：

- **`sim_dt`（仿真推进时间）**：两步 `/model/x3/pose` 的 sim-time 差。它随 RTF 变
  （本机 RTF 约 0.3–1.0），**不是墙钟**。旧记录里的「每步约 1.7 s」是旧版
  **sim_dt**（旧版实测 1.50–1.82 s/步），不能当成 wall 拿来和现在的数字比。
- **`wall`（墙钟）**：`collect.py` 的 `timing.total`，一步真实耗时。`--hold` 的
  单位是**仿真时间**秒（D1 起）：窗口要 `/model/x3/pose` 的 sim-time 走满 hold
  才发零速，与 RTF 无关。墙钟只作兜底（`hold / MIN_WINDOW_RTF`，RTF 低于 0.05
  判本步失败）。
- 关系：`wall ≈ sim_dt / RTF`。低 RTF 下 `wall` 变长但 `sim_dt` 稳定在 hold；
  `stuck` 阈值按 `sim_dt` 缩放，不会误报。

## 4.6 场景随机化 + 并行采集（数据量流水线）

两个纯 stdlib 脚本，不依赖 ROS 2 / PyTorch / gz-transport 直连。

### 4.6.1 按 seed 生成世界

```bash
python3 sim/tools/scene_gen.py --seed 42 --out /tmp/worlds/scene_42.sdf
python3 sim/tools/scene_gen.py --self-check          # 100 个 seed 自检
```

- 以 `worlds/uav_rgb.sdf` 为模板，**只替换**「tall pillars」注释到 X3 模型之间
  的障碍段落和 X3 的出生 pose；四旋翼 / RGB / 深度 / contact 插件和三个固定地标
  原样保留。
- 障碍 **8–9 根**，0.4×0.4×3 m，z∈[0,3] 挡住 `Z_TARGET≈2.1 m` 巡航高度；
  x∈**[1.5, 5.0]** 均匀递增；y∈**[-1.7, 1.7]** 走三条车道（中心车道 ±0.1 m、
  左右车道 ±1.6 m），按「中-左-右」蛇形循环。车道间距 1.6 m = 柱宽 0.4 +
  **缝 1.2 m**（T1 从 0.902 放宽；机体外接直径 0.711 m，单侧余量 0.2445 m）。
  同车道第 i / i+3 根 x 间距 1.3125 m（纵向缝 0.9125 m，仍 ≥ `MIN_GAP`）。
- **数量上限 9 的原因**：全对缝隙 ≥0.9 m 等价于任意两柱心的 Chebyshev 距离
  ≥1.3 m；在 3.5×3.0 m 的密布区内最多 3×3=9 根。任务要求 8–12 根，10–12 根
  必然出现一对缝隙 <0.9 m（机体外接直径 0.711 m，过不去），因此采样取 8–9 根。
- 第一根是「门柱」，中心车道 |y|≤0.25：teacher 只在画面中间带（约 ±10.9°）、
  1.5 m 内才转向；出生点 x=-0.5 时相机（x≈-0.3）到门柱约 1.6 m，**Step 0 附近
  就触发**一次左右选择，保证 episode 不会整集 FORWARD。
- 出生区 x<1.0 无遮挡（柱体最近边 x≥1.3 m）；每根柱到 y=±3 走廊边界
  ≥1.1 m。任意两柱不满足缝 ≥0.9 m 则整组重采样，最多 100 次。
- **出生位姿** `--spawn-mode default|random|safe`：default 在 x=-0.5、y=0；random 在
  门柱与侧柱的缝中心 |y|∈[0.72,0.88]、yaw∈±5° 内随机；safe 在柱区内找一个
  `plan_action` 真正判 `SAFE_NEUTRAL` 的位姿（T2 的「停」样本来源），并用
  0.47 m 机身方框 + 四个桨盘（半径 0.1）校验不撞柱；起飞阶段 z>0.5 后若出现
  contact 直接判本集失败。`collect_parallel --safe-spawn-fraction 1/N` 按固定
  stride 混入 safe episode（只接受 1/N，避免 0.4 被静默当成 0.5）。
- 同 seed 字节级一致（`--self-check` 核对两次 sha256）。同名 `.json` 写
  `scene_id / seed / geometry_version / spawn / 障碍清单 / 净空检查 / world_sha256`，
  供汇总 manifest 用。
- 没做：x>5 的更远障碍；没有利用 4 条动作以外的连续转向
  控制（teacher 的转向角由 hold 与 RTF 决定，见 4.6.2 的 RTF 说明）。

### 4.6.2 并行采集

```bash
# 默认 --steps 40 --hold 1.0（2026-10-30 起）：v3 几何 + A* 每步约 0.27 m，
# T2 实测 29–38 步过线（中位 30）；80 步会让约一半帧在柱区外原地转向。
# 旧几何（v2）曾需要 68–76 步，那时的默认 80 已不适用。
python3 sim/tools/collect_parallel.py --instances 4 --episodes-per-instance 1 \
  --out /tmp/dataset --keep-worlds
python3 sim/tools/collect_parallel.py --instances 4 --episodes-per-instance 1 \
  --steps 20 --hold 1.0 --base-seed 0 --out /tmp/demo --keep-worlds
```

- `scene_id = seed = base_seed + instance*E + episode`；**每个 episode 一个新
  world**。`collect.py` 没有 episode reset，所以 `--episodes-per-instance>1` 时
  每个 episode 都重启一次 `gz sim`（同一实例内顺序执行，实例之间并行）。
- 每个实例用 `setsid` 起独立进程组：`gz sim -s -r <world>`，环境
  `GZ_PARTITION=uavrgb_<随机 UUID>`（每个 episode 独立，避免不同采集运行串台；
  分区记录在 collection manifest 的 task 中）+ `DISPLAY=:20` +
  `XDG_RUNTIME_DIR=/tmp/xdg` + **`LP_NUM_THREADS=1`**。最后一项很关键：
  llvmpipe 默认每个 sim 进程开满渲染线程，6 实例并发时互相抢核，实测 sim
  RTF 从 1.0 掉到 **0.21**，30 s 起飞超时会误报；限成 1 线程后
  同一场景 RTF **0.50**（外部负载下实测 0.65），起飞 ~15–25 s 可完成。
- sim ready 后再稳 3 s 才启动 `collect.py`；某个 episode 失败会重启 sim
  重试 1 次（world 由 seed 确定性生成，不换场景）。这是对 `collect.py`
  常驻 publisher 偶发丢首条起飞指令的兜底，不影响数据语义。
- 先等 `/uav/camera` 和 `/uav/depth` 都真出过一帧，再并发跑 `collect.py`
  （同一 partition，`--scene-id/--seed/--split` 各不相同）。
- 收尾：对每个 sim / collect 进程组 `kill -- -PGID`（SIGTERM→超时 SIGKILL），
  并只对命令行里带本次 world 路径的 ruby 兜底；**不会**用 `pkill -f 'gz sim'`
  或全局杀 ruby。
- split 按**几何组**隔离（`split_manifest.py` 的固定 salt hash）：同几何永远同
  split，与 scene_id 无关；该策略和每集的几何 gid 写进 `collection_manifest.json`。
  旧的 `scene_id % 10` 已废弃——2000 个 seed 只有 1124 种柱布局，实测 241 组跨 split。
- 产物布局：`<out>/<split>/<scene_id>/{actions.jsonl,oracle.jsonl,rgb_*.png,
  episode_manifest.json,scene.json}`；`<out>/collection_manifest.json` 汇总
  每个 episode 的 scene/seed/split/world_sha256/validity/耗时；失败 episode
  进 `<out>/failed/`，脚本退出码非 0。`--keep-worlds` 把 SDF 留到
  `<out>/worlds/`，否则用 /tmp 临时目录并在结束时删除（`scene.json` + seed
  可字节级复现）。
- 实测（本容器，base-seed 0）：4 实例 × 1 episode × 20 步（hold=1.0）
  80/80 步、91.4 s 端到端、0.878 步/s（**常驻 publisher 改造前的数字，并行吞吐未重测**）；12 实例线性外推 2.633 步/s
  （**外推，未实测**）。4 条动作序列分别为
  `FFFFFFFFRFFFFFFFFFFF`、`RFFFFFFFFFFFFFFFFFFF`、
  `FLFFFRFFFFLLFLFLFLFL`、`RFFFFFFFFFFFFFFFFFFF`，**4/4 含非 FORWARD**，
  每条都与近组障碍交叉过（最近 0.57–1.29 m）。
- 已知限制：订阅仍走 `gz topic -e` 子进程（未做 gz-transport 订阅直连；发布已直连，
  见 4.5）；没有数据增强；`collect_parallel` 会给每集传 `--scene-meta`，
  普通集走 **A\* 全局路径 + 纯追踪**（`teacher.py:plan_action_global`），
  safe 出生集走局部 `plan_action`（保留真 SAFE 标签）；manifest 的
  `teacher`/`planner` 字段区分。第 4.6.1 节描述的纯反应式深度专家只在单独跑
  `collect.py` 且不给 `--scene-meta` 时生效。
  T1 实测（20 个 test 场景、20 个唯一几何、40 步/集）**20/20 无碰撞穿过
  x=6.0，100%**；旧局部规划在 v3 上 24 集 15 集擦柱。collision 真值会如实
  记录，训练前按需过滤。

## 5. 对齐冻结的 ObservationSpec / EpisodeSchema

- 相机 `R8G8B8` 即 RGB 字节序，与 `SimObservation.rgb`（HWC uint8）同序；
  采集保存 320×240，`tools/preprocess.py` 默认下采样到 160×120，打包为 NCHW。
  `src/include/uavrgb/observation.hpp` 的 `kObsW=160`、`kObsH=120`。
- `ObservationSpec.timestamp_ns = header.stamp.sec*1e9 + header.stamp.nsec`；
  `episode_id / scene_id / seed` 由采集器按 `EpisodeSchema` 写入，episode 按
  场景/地图隔离划分。
- 相机帧若比当前 sim time 旧超阈值，应拒绝重投（`CAMERA_FRAME_STALE`）。

## 6. 话题速查

| 话题 | 类型 | 含义 |
|---|---|---|
| `/uav/camera` | gz.msgs.Image | 前向 RGB 相机（320×240 R8G8B8） |
| `/uav/camera_info` | gz.msgs.CameraInfo | 内参 |
| `/uav/contact` | gz.msgs.Contacts | 机体 + 四旋翼接触（稀疏，只在接触时发；碰撞真值） |
| `/X3/gazebo/command/twist` | gz.msgs.Twist | 速度/偏航指令（入） |
| `/X3/gazebo/command/motor_speed` | gz.msgs.Actuators | 电机转速（入） |
| `/model/x3/pose` | gz.msgs.Pose_V | 真值位姿 |
| `/model/x3/odometry` | gz.msgs.Odometry | 真值里程计 |
| `/clock` | gz.msgs.Clock | 仿真时钟 |

## 7. 实测验证了什么（本容器，2026-10-28）

- `DISPLAY=:20 gz sim -s -r sim/worlds/uav_rgb.sdf` 正常加载，无 error。
- `/uav/camera` 发布真帧：`frame_id=x3::base_link::camera`，320×240×3，
  230400 字节，sim_time 单调递增；画面里红/绿/蓝三地标清晰可见
  （`evidence/uav_rgb_first_frame.png`，红 9556px / 蓝 6232px / 绿 3470px）。
- 发送 `linear z:0.5` 后，`/model/x3/pose` z 从 0.055 连续升到 0.354（速度环生效）。
- 纯 headless（无 DISPLAY）相机：**不通过**，Ogre2 EGL segfault（附录 A）。

## 附录 A：headless 相机失败原文（Ogre2 EGL-PBuffer）

```
[Err] [Ogre2RenderEngine.cc:1293] Unable to create the rendering window:
OGRE EXCEPTION(3:RenderingAPIException): eglInitialize failed for device
EGL_EXT_device_drm ... #0 /dev/dri/card1 in EGLSupport::getGLDisplay at
./RenderSystems/GL3Plus/src/windowing/EGL/PBuffer/OgreEglPBufferSupport.cpp (line 320)
[Err] [Ogre2RenderEngine.cc:1293] ... OpenGL 3.3 is not supported. Please update
your graphics card drivers. in GL3PlusRenderSystem::initialiseContext ...
Segmentation fault (Address not mapped to object [0x220])
```

（GUI 模式 `gz sim -r` 同类：`libMinimalScene → Ogre2RenderEngine → libEGL_mesa
→ driCreateNewScreen3 → Segmentation fault`。）

## 尚未完成的仿真验收

1. 采集循环、episode 写入及最近时间戳配对已实现；动作持续时间已改为 sim-time
   advance（D1）。仍需给深度配对残差和传感器停更设置拒绝门槛。
2. 已保存相机声明的内参；尚未完成相机外参/畸变的独立标定和 episode reset。
   当前并行编排通过每集重启世界隔离状态。
3. 学生策略闭环评测器已实现（`eval_closed_loop.py`，E6-2/E6-4），能输出成功率、
   碰撞率、误前向率和推理 p50/p95。但**还没有一次跑出可用的成绩**：旧 checkpoint
   20/20 撞柱、SR=0，原因是当时的训练数据没有 `SAFE_NEUTRAL` 类。2026-10-30 T2/T3
   已补出四类数据（52 集/2080 步）并离线过闸（test macro_f1 0.5859）；新权重
   闭环冒烟仍 SR=0（3 集全撞，碰撞点 3.4–3.7 m，0 次 SAFE，见
   `docs/worklog/2026-10-30-0530-t7-smoke.md`）。采集成功和专家自测通过不能代替此项。
