# collect.py 常驻 gz 订阅（E3 I/O 改造）

- 时间：2026-10-29 01:29
- 范围：`sim/tools/collect.py`、新增 `sim/tools/gz_io.py`、`sim/README.md` §4.5 一小节
- 结论：**部分完成** —— 常驻订阅、sim-time 配对、内参落盘、异常/收尾都完成并实测；
  但「20 步平均 < 1.0 s/步」**未达到**，实测 **1.518 s/步**（旧版 wall 2.90 s/步）。
  瓶颈不是 I/O，而是 `gz topic -p` 命令里硬编码的 0.8 s 发布前 sleep，见下。

## 做了什么

- 新增 `sim/tools/gz_io.py`：`TopicSub` 常驻订阅。
  - 一个 `gz topic -e -t <topic> --json-output` 进程 + 后台读线程，逐行 `json.loads`；
    坏行跳过；缓冲是 `deque(maxlen=buf)` 的 latest-wins（不积压）。
  - `latest()` / `latest_after(stamp_ns, timeout=0)` / `frames()` / `healthy()` / `close()`；
    超时、进程退出都返回 None，不永久阻塞；`close()` = terminate → wait(2 s) → kill 兜底。
  - `python3 gz_io.py --self-check`：假 stdout 自检坏行/环形缓冲/latest_after/EOF（无 sim）。
- 改 `sim/tools/collect.py`：
  - 启动时给 `/uav/camera`、`/uav/depth`、`/model/x3/pose`、`/uav/contact`、
    `/uav/camera_info` 各建常驻订阅（`open_subs`），`finally` 里 `close_subs`（异常/中断路径也关）。
  - 每步以**最新 RGB 为锚点**：`latest_after(prev_rgb_ts)` 取新帧；深度从缓冲里挑
    `|stamp - rgb_ts|` 最近的一帧；位姿挑 `stamp > rgb_ts` 的最近一帧（M1 靠构造保证）。
  - `actions.jsonl` 增加 `rgb_ts` / `pose_ts`（动作前那帧位姿）/ `pose_rgb_gap_ns`；
    保留原有全部字段。PNG 落盘挪到动作 publisher 起跑之后（与发布延迟重叠，RGB 仍在发指令前抓到）。
  - 动作/零速发布逻辑**原样保留**（同样两个 `gz topic -p`、hold 后立刻起零速、overtake 保护）。
  - 碰撞：contact 常驻订阅按**接收序号**判窗口内是否出现过非空 contact 列表
    （不用消息时间戳，也不会把上一步缓冲里的旧接触当本步），hold 期间每 20 ms 轮询一次。
  - 起飞 `take_off(pose_sub)` 改用常驻位姿，不再每 0.2 s spawn 一次 `gz topic -e`。
  - manifest：从 `/uav/camera_info` 取 `frame_id == x3::base_link::camera` 那条的
    `intrinsics.k` 填 `fx/fy/cx/cy`（+width/height），新增顶层 `camera_source: "gazebo_camera_info"`；
    没读到时保持 null、`camera_source: null`。
  - 结束时打印每步耗时分解 / M1 计数 / 配对残差 min·median·max（stderr），便于复验。
  - 旧函数 `echo_json/read_depth/read_rgb_png/read_pose/contact_fired/spawn_contact_listener`
    保留（`test_collect.py` 用 `echo_json`/`read_depth`/`contact_fired`，不改测试）。
- `sim/README.md` §4.5 补一段常驻订阅说明 + 新字段 + 1.0 s 发布延迟的限制。

## 没做什么

- **没做 C++ 桥**。没有写任何 C/C++ publisher/subscriber。
- **没做 sim-time hold**。hold 仍是 wall-clock sleep（结束立刻起零速），没有用
  `WorldControl` 暂停/步进，也没有用 `/clock` 对齐动作窗口。
- **没改 twist 发布方式**。仍是每步两次 `gz topic -p` 子进程（动作 + 零速），
  没有常驻 publisher、没有 LD_PRELOAD 绕过 sleep、没有 gz-transport 直连。
  - 原因：本轮铁律是「纯 stdlib Python + Gazebo CLI」；本机其实装了
    `python3-gz-transport13`，直连 publisher 可把 1.0 s 降到 ~10 ms（预计 0.55 s/步），
    但属于额外依赖，超出本轮约束，留作下一轮。
- **没跑 `collect_parallel.py`**（另一个 agent 正在改它，不并发验证）。
  但按它的调用方式用同样参数（`--steps/--hold/--out/--scene-id/--seed/--episode-id/--split`）
  真跑了 collect.py，manifest 的 `num_steps/validity/error` 字段保持兼容。
- 没改 `scene_gen.py`、`collect_parallel.py`、`ui/`、`src/`、`tools/`、`experiments/`、
  `teacher.py`、`save_frame.py`、`worlds/uav_rgb.sdf`、`test_collect.py`。
- 没做长时间稳定性/多实例并发测试（只单实例 20 步、10 步、3 步、SIGINT、无 sim 五次真跑）。
- 没有为 `/uav/contact` 的「接触结束是否发空列表」做专门实验（实测无接触时 0 msg/s，
  接地/撞柱时 ~225 msg/s；按序号+轮询检测，两种情况都能覆盖）。
- `docs/results.md` 没追加（本次按任务要求把真实输出放在本 worklog）。

## 验证到哪一步

机器状态：RTX 2080 Ti 有卡（本任务不用 GPU）；测量时另一 agent 的并行采集已结束
（`ps` 无其它 ruby，load ~12 为衰减值），新旧版本都在同样状态下测。

### 1. 旧版基线（20 步、hold=0.5、base world，`io3` 分区）

```bash
cp -a sim/tools /tmp/uavrgb_io/oldtools   # 旧 collect.py 副本（未改仓库文件）
/tmp/uavrgb_io/run_ep.sh sim/worlds/uav_rgb.sdf io3 \
  python3 /tmp/uavrgb_io/oldtools/collect.py --steps 20 --hold 0.5 \
  --out /tmp/uavrgb_io/old_ep1 --scene-id 1 --seed 42
```

```
写入 20/20 步 ... completed=True error=None
=== collect rc=0 wall=69845 ms ===
old: 20 步, 步间 sim-time 差 min/median/max = 2.87/2.90/3.00 s
old |rgb-depth| 残差 min/median/max = 200/232/300 ms
```

（用户引用的旧版 1.70 s/步在这台机器上是旧版 **sim_dt**：旧 run 的 sim_dt
逐步 1.50–1.82 s、均值 ~1.71；实际 wall 是 2.90 s/步。旧版每步 = 4 次
`gz topic -e -n 1`（各 ~250 ms）+ 动作块（hold 0.5 + 发布 1.0 s）+ contact 0.5 s
+ 动作后位姿 250 ms。）

### 2. 新版 20 步（同 world、同参数，`io7` 分区）

```bash
/tmp/uavrgb_io/run_ep.sh sim/worlds/uav_rgb.sdf io7 \
  python3 sim/tools/collect.py --steps 20 --hold 0.5 \
  --out /tmp/uavrgb_io/final_ep --scene-id 1 --seed 42 --episode-id 42 --split train
```

```
相机内参: fx=277.128 fy=277.128 cx=160.000 cy=120.000 (320x240, x3::base_link::camera)
起飞完成: z=2.12 m
每步耗时: total=1.518s  I/O=0.016s  动作窗口=0.494s  发布开销=1.008s  其他=0.000s
M1 rgb_ts<action_start_ts: 20/20
配对残差 |rgb_ts-depth_ts|: min=0.0ms median=0.0ms max=36.0ms
配对残差 pose_ts-rgb_ts: min=4.0ms median=12.0ms max=20.0ms
写入 20/20 步 ... completed=True error=None
=== collect rc=0 wall=41452 ms ===
```

manifest（节选，真输出）：

```json
"camera": {"width": 320, "height": 240, "fx": 277.1281433105469,
           "fy": 277.12815284729004, "cx": 160.0, "cy": 120.0},
"camera_source": "gazebo_camera_info",
"validity": {"camera_verified": true, "physics_verified": true, "episode_executed": true},
"num_steps": 20, "requested_steps": 20, "error": null
```

对比：

| 指标 | 旧版 | 新版 |
| --- | --- | --- |
| 20 步 end-to-end wall（含起飞/收尾） | 69.85 s | 41.45 s（1.69×） |
| 每步 wall（步循环） | 2.90 s | 1.518 s（1.91×） |
| 其中 I/O | ~1.0 s（4 次 snapshot）+ contact 0.5 s + 动作后位姿 0.25 s | 0.016 s |
| 其中动作窗口 | 0.497 s | 0.494 s |
| 其中发布开销（publisher 启动+0.8 s sleep） | ~1.0 s | 1.008 s |
| \|rgb_ts-depth_ts\| min/median/max | 200/232/300 ms | 0/0/36 ms |
| pose_ts-rgb_ts（动作前位姿 vs RGB） | 312 ms（旧 run 第 0 步） | 4/12/20 ms |
| M1 rgb_ts<action_start_ts | 旧版 20/20 | 20/20 |
| 每步 sim_dt | ~1.71 s | ~1.52 s |

**< 1.0 s/步 未达到**。根因（实测证据）：

- `gdb` 断在 `nanosleep`：`gz-transport-topic -t ... -p ...` 在
  `Publisher::Publish` **之前**调用 `nanosleep(0, 800000000)`（0.8 s），
  再加 `gz` Ruby wrapper + Node 启动 ~0.2 s，`spawn → 消息真正到达` 实测 975–1035 ms；
  与有没有订阅者无关（无订阅者的 `/foo` 同样 1.0 s），`GZ_IP=127.0.0.1` 也一样。
  这是 gz-transport13 CLI 的发现宽限期，CLI 上无开关。
- 采集每步必须两次发布（动作、零速）；动作 publisher 的 1.0 s 在关键路径上
  （观测 → spawn 动作 publisher → 1.0 s 后才生效），所以每步 ≥ 1.0 + hold = 1.5 s。
  要和 M1/因果链共存，不能把观测提前到上一个动作窗口里（否则标签对应的场景是错的）。
- 还能压的：改用直连 transport 的常驻 publisher（或 C++ 桥），把 1.0 s 变 ~10 ms，
  预计 0.55 s/步；本轮未做（见「没做什么」）。

### 3. 碰撞仍有效（`--policy forward` 10 步、hold=1.5，`io2` 分区）

```bash
/tmp/uavrgb_io/run_ep.sh sim/worlds/uav_rgb.sdf io2 \
  python3 sim/tools/collect.py --steps 10 --hold 1.5 --policy forward \
  --out /tmp/uavrgb_io/fwd_ep --scene-id 1 --seed 43
```

```
step 6: ... pos={'x': 2.5627...} dx=0.237m collision=True(contact)
step 7: ... dx=0.022m collision=True(contact+stuck)
step 8: ... dx=0.019m collision=True(contact+stuck)
step 9: ... dx=0.019m collision=True(contact+stuck)
每步耗时: total=2.510s  I/O=0.008s  动作窗口=1.499s  发布开销=1.003s  其他=0.000s
写入 10/10 步 ... completed=True error=None
```

撞的是 `pillar_center`（x=3.0, y=0, 0.4×0.4×3）；机体半宽 0.235 → 停在 x≈2.565，
与实测 2.562 一致。前 6 步无碰撞，第 6 步起 contact=true，第 7 步起 stuck 兜底也 true。
对照：teacher 20 步 `collision` 0/20（无误报）。

### 4. 异常路径（SIGINT 中断在步循环中，`io6` 分区）

```bash
timeout --preserve-status -s INT 12 python3 sim/tools/collect.py \
  --steps 20 --hold 0.5 --out /tmp/uavrgb_io/sig_ep3 --scene-id 1 --seed 44
```

```
写入 1/20 步 -> ... completed=False error=KeyboardInterrupt:
collect rc=130
pose1 0.134 0.001 2.563
pose2 0.135 0.001 2.563
pose3 0.135 0.001 2.563
--- collect 的 gz topic 子进程残留 ---
无
```

manifest：`num_steps=1`、`validity` 三项全 false、`error="KeyboardInterrupt: "`、
`camera_source="gazebo_camera_info"`；actions/oracle 各 1 行（已 flush）；
位姿 3 次采样 3 s 内不动（无人机停住）；无 `gz topic -e` 子进程残留。
（另：`rgb_00001.png` 是中断时正在写、还没轮到写 jsonl 行的孤儿 PNG，manifest 的
`num_steps` 只认 actions.jsonl，消费方按 actions 引用取帧即可。）

### 5. 无 sim（订阅等不到数据）路径

```bash
GZ_PARTITION=io_nosim python3 sim/tools/collect.py --steps 2 \
  --out /tmp/uavrgb_io/nosim_ep --scene-id 2 --seed 9 --episode-id 77
# rc=1 wall=11127ms
# RuntimeError: 常驻订阅未就绪: 10s 内没有数据: ['rgb', 'depth', 'pose']
# manifest: episode_id=77 num_steps=0 validity 全 false camera_source=null
```
无残留 `gz topic -e`。

### 6. 单元/自检

```bash
python3 sim/tools/gz_io.py --self-check     # gz_io self-check: 全部通过
python3 sim/tools/test_collect.py           # test_collect: 13 项通过
python3 sim/tools/test_save_frame.py        # test_save_frame: PASS
```

### 7. 收尾

每次跑的 sim 都是 `setsid` 起、按 `/proc/<pid>/environ` 里的 `GZ_PARTITION=io*`
认自己的 ruby 后 `kill -- -PGID`；结束时 `ps` 无 `ruby`/`gz topic -e` 残留，
`/tmp/uavrgb_io` 已删除（只保留仓库内代码和本 worklog）。

## 风险与遗留

- **每步 1.5 s 的地板**：`gz topic -p` 的 0.8 s sleep。要提速需换发布通道
  （gz-transport 直连常驻 publisher，本机已装 `python3-gz-transport13`；或 C++ 桥）。
  这会改「纯 stdlib + CLI」约束，需拍板。
- **SIGKILL collect 会漏订阅进程**：`close()` 只在 Python 正常/异常退出时执行。
  `collect_parallel.py` 用进程组 `kill -- -PGID` 收尾，订阅会一起死；
  但若只 `kill -9 <collect pid>`，5 个 `gz topic -e` 会成为孤儿（它们不会自己退出）。
- **contact 轮询窗口 20 ms + 16 帧缓冲（~70 ms）**：持续接触（撞柱）必检出；
  瞬时擦碰如果接触传感器只发 1–2 帧且恰好在两次轮询之间被覆盖，理论上可能漏，
  stuck 兜底仍在。没有做瞬时擦碰的专门实验。
- **`_pick_pose_after` 要求位姿晚于 RGB**：位姿 45 Hz 时等 4–20 ms；若位姿话题
  静默（进程活着但不出帧），该步会报错终止（`step i: pose 无晚于 RGB 的帧`），
  不会静默用旧帧。动作后位姿有 1.0 s 的补等（防止把动作前位姿当 oracle）。
- **新字段是增量的**：`actions.jsonl` 原有字段全保留，`rgb_ts/pose_ts/pose_rgb_gap_ns`
  是新增；manifest 新增 `camera_source`。没跑 collect_parallel 端到端（对方在改），
  它的 manifest 消费字段（`num_steps/validity/error`）保持。
- **孤儿 PNG**：中断发生在 PNG 已写、jsonl 行未写之间时，目录里会多一张不被引用的
  `rgb_XXXXX.png`（旧版同样存在），manifest `num_steps` 不受影响。
- 测量只做了单实例；多实例并发时 `gz topic -e` 常驻订阅每个话题多一个订阅者，
  对 sim 的序列化开销略有增加（本机未量化）。
