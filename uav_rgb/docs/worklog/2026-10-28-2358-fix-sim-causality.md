# 修复仿真采集链路：因果顺序、动作停止、异常清理、深度 fail-closed

- 时间：2026-10-28 23:58
- 范围：`sim/tools/collect.py`、`sim/tools/teacher.py`、`sim/tools/save_frame.py`、
  `sim/worlds/uav_rgb.sdf`、`sim/README.md`；本 worklog。
- 结论：**完成本轮的 M1–M5**（全部真起 Gazebo 验证）；S1/S2/S3/S4/S9 修复，
  S5/S6 部分修复；S7/S8/S10 未做（原因见下）。

## 做了什么

### M1 RGB 因果倒置（collect.py）
- 每步顺序改为：读深度 → **抓 RGB** → 读动作前位姿 → teacher 算动作 →
  起接触监听 → 发动作 → hold → 零速 → 读动作后位姿 → 收接触。
- `actions.jsonl` 的 `timestamp_ns` 改为动作前 RGB 帧时间戳；动作后位姿只进
  `oracle.jsonl`，并额外记录 `action_start_ns`（发指令前最后一次位姿观测）和
  `action_end_ns`，便于复核因果顺序。
- 深度/RGB 仍是两次独立 `gz topic -e`（不是同一帧），但两者都在发动作之前；
  按任务要求，同帧对齐/`/clock` 精确对齐留给下一轮。

### M2 动作不主动停止（collect.py）
- hold 结束后立刻起一个 `SAFE_NEUTRAL + vz=0` 的零速 publisher，再读位姿/接触。
- 实测发现 `gz topic -p` **在进程末尾才真正发布**（本机固定约 0.82 s，见下），
  若先阻塞等动作发布再 sleep(hold) 再发零速，实际动作窗口会多出约一个发布延迟。
  实现改为：动作 publisher 先起跑；`sleep(hold)` 后起零速 publisher；两者各自
  `launch+0.82s` 发布，所以有效动作窗口≈hold。另用等待线程记录动作 publisher
  的真实发布时刻，`action_wall_s` 按“零速发布 - 动作发布”计算并打印。
- 增加安全兜底：若零速比动作更早发布（`action_wall <= 0`），立即补一条同步零速
  并抛异常，避免 forward 指令残留。
- 每步 stderr 打印 `dx`（位移）、`dyaw`（转角）、`action_wall`、`sim_dt`、z 高度。

### M3 异常路径无清理（collect.py）
- `try/finally`：正常结束、深度失败、异常、Ctrl-C 都会在退出前发零速；
  动作/零速/接触 listener 子进程在 `finally` 中兜底 kill。
- `actions.jsonl` / `oracle.jsonl` 每步写完立即 `flush()`，不再最后一次性写。
- `episode_manifest.json` 在 finally 中最后写，临时文件 + `os.replace` 原子替换；
  `camera_verified` / `physics_verified` / `episode_executed` 按实际完成步数、
  RGB/位姿是否齐全填写；新增 `num_steps`、`requested_steps`、`error`。
- 未完成 episode 以退出码 1 结束（`SystemExit(1)`），不再静默返回 0。

### M4 `gz topic -e` 无超时、`cmd_twist` 不查返回码（collect.py）
- `_gz()` 统一 `timeout=5s`，超时返回 `rc=124`；`echo_json` 仍重试 3 次但有界。
- `cmd_twist`/动作 publisher 检查 returncode，失败重试一次后抛
  `RuntimeError`；零速 publisher 失败也会先等它结束再抛错。

### M5 深度失效 fail-open（teacher.py）
- `_strip_min` 返回 `(最小有效深度, 有效样本数)`。
- 中心带有效样本数 0 → `SAFE_NEUTRAL`；某侧带无有效样本 → 该侧按“堵”处理。
- 校验 `w>0 && h>0`、`len(depth) >= w*h`，不满足抛 `ValueError`（不再是 IndexError）。

### S1 小尺寸深度（teacher.py）
- 增加 `h >= 5`、`w >= 3` 下限校验，不满足抛 `ValueError`。
- `__main__` 增加 h=1/2/3、w=2、w=0、短 buffer、全 0/NaN/负数、
  中心带失效、侧带失效用例。

### S2 save_frame 行 stride（save_frame.py、collect.py）
- `decode(msg)` 按 `raw[y*step : y*step+w*nch]` 逐行切，不再把整块当紧凑缓冲。
- `nch` 由 `pixelFormatType`/`format` 推断（RGB/BGR/RGBA/BGRA），
  `step` 只用于取行；未知格式不再用 `step//w` 瞎猜。
- `collect.read_depth` 同样按 `step` 逐行解包 float32，并校验
  `len(raw) >= (h-1)*step + 4*w`。
- `collect.read_rgb_png` 复用 `save_frame.decode`/`_png`。

### S3 save_frame 非 3/4 通道/截断（save_frame.py）
- 先把 PNG 字节完整构造好，再 `open(dst,'wb')`；灰度、16-bit、未知格式、
  截断 payload 都干净报错 rc=1，不创建/不留下 0 字节文件。
- 增加 BGR/BGRA → RGB/RGBA 的通道交换；docstring 收窄到实际支持范围。

### S5 碰撞覆盖（uav_rgb.sdf、collect.py）
- 四个旋翼 `rotor_0..3` 各加一个 `type="contact"` sensor，发布到同一个
  `/uav/contact`；几何未改。
- contact listener 的生存期覆盖到“零速发布 + 读动作后位姿”之后，再收信号。
- `gz topic -i -t /uav/contact` 实测 **5 个 publisher**（base + 4 旋翼）。

### S9 SDF 左右命名（uav_rgb.sdf）
- `pillar_right` → `pillar_body_left`（仍在 y=+1.4，几何未动）。
- 注释改为：+Y 是机体左侧，y=+1.4 的柱子堵教师左带，机体右侧开 → YAW_RIGHT。
- `sim/README.md` 的 `/uav/contact` 说明改为“机体+四旋翼”，
  `/model/x3/pose` 类型从 `gz.msgs.Pose` 修正为 `gz.msgs.Pose_V`。

## 没做什么（逐条对照）

### 必须修
- **M1 已修并验证**。RGB 在发动作前抓取；10 步 teacher 每步
  `rgb_ts < action_start_ns`，领先 204–232 ms。
- **M2 已修并验证**。`action_wall` 在 hold=0.5 时实测 0.43–0.56 s；
  hold=1.5 时 1.47–1.53 s。x 位移从旧版约 0.76 m/步降到 0.084–0.126 m/步
  （hold=0.5），hold=1.5 时 0.32–0.41 m/步。
- **M3 已修并验证**。SIGINT 中断后 2 条已采行保留、manifest 三项 validity
  全 false、无人机 1.2 s 内位移 0.0000 m；无 Gazebo 时 0 步 manifest 全 false
  且退出码 1。
- **M4 已修并验证**。停掉 Gazebo 后采集 17.0 s 有界退出（旧版会永久阻塞）；
  `cmd_twist` 已检查 returncode/重试，但“publish 真的返回非 0”这条分支
  没能现场触发，只做了代码审查（见风险）。
- **M5 已修并验证**。teacher 自检新增全 0/全 NaN/全负数/中心带失效/侧带失效
  用例，全部通过。

### 应该修
- **S1 已修并验证**（h<5、w<3、短 buffer 均抛 ValueError）。
- **S2 已修并验证**：2×2 step=7、4×2 step=13、320×2 step=1024 的 padding
  用例全部逐行正确；真实 RGB step=960、深度 step=1280 也正常。
- **S3 已修并验证**：灰度/16-bit/截断/非法尺寸均 rc=1 且不产生文件。
- **S4 已修**：read_depth 按 stride 解码，合成 padded depth 单测通过；
  真实深度 step=1280=4×320，属紧凑格式，未遇到真实 padded 深度。
- **S5 部分修复**：旋翼 contact sensor 已挂（5 publisher 已确认），listener
  生存期已修；但**没有实际制造“旋翼擦碰、机体没碰”的物理场景**，旋翼 contact
  的真实触发未验证。“读失败/超时记 unknown 而不是 false”未做，超时仍记 false。
- **S6 部分修复**：actions 行已带 `rgb_ok`/`frame`，oracle 带动作起止时间，
  manifest 最后原子写、validity 按实际数据填；但 `camera` 宽高仍硬编码
  320×240，未检查输出目录旧文件，未在 manifest 里写 `rgb_pattern`/`num_rgb`。
  中断步会留下一个未被 actions 引用的孤儿 PNG（实测 `rgb_00002.png`），
  manifest 的 `num_steps=2` 不受影响；未做删除。
- **S7 未做**：按任务要求本轮不做 `/clock` sim-time 精确对齐。起飞仍是
  2.5 s wall-clock，hold 仍是 wall-clock；软件渲染时 RTF≈0.8–0.9，
  hold=0.5 的实际 sim 动作时长略小于 0.5，因此 x 位移 0.084–0.126 m，
  低于理论 0.15 m。已如实打印 `action_wall` 与 `dx`。
- **S8 未做**：按要求不改世界几何/相机/`Z_TARGET`。10 步 teacher 在
  z=1.39–1.61 时三带皆堵，从 step 3 起一直 `SAFE_NEUTRAL`，未出现 YAW；
  这是审核里 S7/S8 的低空死锁/余量问题，不是本次回归。z 仍在爬升
  1.149→1.609，但 10 步内没越过阈值。
- **S9 已修**：只改名字/注释，几何位置未动。`docs/` 里其他旧 worklog
  仍写 `pillar_right`/“左开→YAW_LEFT”，按约束未改；仓库内 `sim/` 代码
  没有引用旧模型名的脚本。
- **S10 未做**：`ui/` 是另一个会话的范围，按约束未碰。

### 其他明确没做
- 没有做 RGB/深度同帧、`/clock` sim-time 对齐、场景随机化、128×128 降采样
  （任务指定下一轮）。
- 没有做 YAW 动作的 live 转角验证：本轮 teacher 因低空一直 SAFE_NEUTRAL，
  `dyaw` 每步都是 0；零速/动作窗口机制与前进相同，但没有拿真实 YAW 步贴数字。
- 没有验证 BGR/BGRA 的真实相机帧（只用合成消息单测了通道交换）。
- 没有跑 `ui/test_monitor.py`、没有碰 `src/`、`tools/`、`experiments/`。

## 待确认

- S7/S8 的修法需要拍板：`Z_TARGET` 提到 ≥2.1、deadband 降到 ≤0.05，
  或起飞改 sim-time；本轮按约束未动。
- 是否需要把“接触信号超时”从 `collision:false` 改成 `collision:null/unknown`
  并进 manifest 计数（S5 剩余项）。
- 中断步的孤儿 RGB 是否要在 collect 里删除（本轮只保证 actions/manifest
  一致，没删文件）。
- `--hold` 目前只校验 >0；延迟隐藏依赖两个 publisher 的相对时序，虽然加了
  `action_wall<=0` 兜底，但极小的 hold（<0.1 s）仍有反转风险，是否要设下限。

## 验证到哪一步

环境：本机 RTX 2080 Ti 存在，但 Gazebo 相机无 NVIDIA EGL，按 README 用
`DISPLAY=:20` 软件 GLX；server-only。

### 1. 纯函数/静态检查

```
$ python3 sim/tools/teacher.py
  ... 20 项全 ok（含 5 条 fail-closed、6 条小尺寸/短 buffer ValueError）
  teacher: 全部通过   RC=0

$ python3 /tmp/uavrgb_saveframe_test.py        # 临时脚本，已删除
  ok 1x1 rgb / 2x2 step7 pad / 4x2 step13 pad / 320x2 step1024 pad
  ok 2x2 rgba / 1x1 bgr
  ok gray: rc=1 file=False
  ok 16bit rgb: rc=1 file=False
  ok truncated: rc=1 file=False
  save_frame selftest: PASS   RC=0

$ # collect.read_depth 合成 padded depth（临时脚本，已删除）
  padded depth: [1.0, 2.0, 3.0, 4.0] 2 2 7500000000 OK
  truncated depth: None OK

$ gz sdf -p sim/worlds/uav_rgb.sdf > /tmp/uavrgb_parsed.sdf
  RC=0, 598 行, stderr 0 字节

$ python3 -m py_compile sim/tools/{collect,save_frame,teacher}.py
  RC=0
```

真实消息格式（`gz topic -e -n 1 --json-output`）：

```
/uav/camera: width=320 height=240 step=960  pixelFormatType=RGB_INT8
/uav/depth : width=320 height=240 step=1280 pixelFormatType=R_FLOAT32
```

### 2. 真实 Gazebo：teacher 10 步（hold=0.5，fresh world）

命令：

```
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg setsid gz sim -s -r sim/worlds/uav_rgb.sdf
python3 sim/tools/collect.py --steps 10 --hold 0.5 \
  --out /tmp/uavrgb_ep_teacher --scene-id 1 --seed 42 --split train
```

逐step实测（`dx`/`dyaw`/动作窗口/z）：

```
step 0: FORWARD_SLOW dx=0.125m dyaw=0.0deg action_wall=0.52s z=1.1487
step 1: FORWARD_SLOW dx=0.084m dyaw=0.0deg action_wall=0.51s z=1.2436
step 2: FORWARD_SLOW dx=0.087m dyaw=0.0deg action_wall=0.53s z=1.3251
step 3: SAFE_NEUTRAL dx=0.035m dyaw=-0.0deg action_wall=0.50s z=1.3904
step 4: SAFE_NEUTRAL dx=0.030m dyaw=-0.0deg action_wall=0.50s z=1.4438
step 5: SAFE_NEUTRAL dx=0.029m dyaw=-0.0deg action_wall=0.56s z=1.4933
step 6: SAFE_NEUTRAL dx=0.021m dyaw=0.0deg  action_wall=0.47s z=1.5344
step 7: SAFE_NEUTRAL dx=0.019m dyaw=0.0deg  action_wall=0.50s z=1.5687
step 8: SAFE_NEUTRAL dx=0.014m dyaw=0.0deg  action_wall=0.43s z=1.5969
step 9: SAFE_NEUTRAL dx=0.001m dyaw=0.0deg  action_wall=0.52s z=1.6088
```

动作序列：`[1,1,1,0,0,0,0,0,0,0]`。
manifest：`camera_verified/physics_verified/episode_executed = true,true,true`，
`num_steps=10`，`error=null`。

**RGB vs 动作开始时间戳**（actions `timestamp_ns` vs oracle `action_start_ns`）：

```
step 0: rgb_ts=15508000000 action_start=15740000000 delta=232ms rgb<start=True
step 1: rgb_ts=18320000000 action_start=18540000000 delta=220ms rgb<start=True
step 2: rgb_ts=21108000000 action_start=21320000000 delta=212ms rgb<start=True
step 3: rgb_ts=23896000000 action_start=24100000000 delta=204ms rgb<start=True
step 4: rgb_ts=26644000000 action_start=26860000000 delta=216ms rgb<start=True
step 5: rgb_ts=29444000000 action_start=29660000000 delta=216ms rgb<start=True
step 6: rgb_ts=32276000000 action_start=32480000000 delta=204ms rgb<start=True
step 7: rgb_ts=35032000000 action_start=35240000000 delta=208ms rgb<start=True
step 8: rgb_ts=37840000000 action_start=38060000000 delta=220ms rgb<start=True
step 9: rgb_ts=40652000000 action_start=40860000000 delta=208ms rgb<start=True
ALL rgb<action_start: True
```

PNG 校验：在更早一次 6 步 teacher run（`read_rgb_png` 已与最终版相同）上
逐个解码，6 个 PNG 全部 320×240、IHDR/IDAT 完整、解码字节数正确；
最终 10 步 run 的 10 个 PNG 只确认了文件生成与 actions 引用，未再逐个解码
（`/tmp` 已清理，无法补测）。

旧版对照：审核记录每 FORWARD 步约 0.76 m、动作窗口约 2.3–2.9 s；
本轮 hold=0.5 的 FORWARD 步为 0.084–0.125 m、动作窗口（wall）0.43–0.56 s。

### 3. 真实 Gazebo：forward 撞柱（hold=1.5，fresh world）

```
python3 sim/tools/collect.py --steps 10 --hold 1.5 --policy forward \
  --out /tmp/uavrgb_ep_crash --scene-id 1 --seed 43
```

关键输出：

```
step 0: dx=0.410m action_wall=1.47s x=0.3193 collision=False
step 1: dx=0.364m action_wall=1.53s x=0.7445 collision=False
...
step 5: dx=0.330m action_wall=1.51s x=2.4010 collision=False
step 6: dx=0.076m action_wall=1.47s x=2.5630 collision=True(contact)
step 7: dx=0.009m action_wall=1.49s x=2.5634 collision=True(contact+stuck)
step 8: dx=0.007m action_wall=1.50s x=2.5633 collision=True(contact+stuck)
step 9: dx=0.008m action_wall=1.49s x=2.5635 collision=True(contact+stuck)
```

结论：碰撞真值仍有效，step 6 起 `collision=true`，x 冻结在柱面
2.8−0.235≈2.563 m；`contact+stuck` 双信号在后续步都触发。

### 4. 异常路径（M3）：SIGINT

```
timeout -s INT 12 python3 sim/tools/collect.py --steps 30 --hold 0.5 \
  --out /tmp/uavrgb_ep_sigint --scene-id 2 --seed 44
```

结果：

```
actions.jsonl 保留 2 行（step 0/1，均 timestamp/rgb_ok/frame 完整）
manifest: validity = {camera_verified:false, physics_verified:false,
                      episode_executed:false}
          num_steps=2, requested_steps=30, error="KeyboardInterrupt: "
中断后两次读位姿：moved_in_1.2s=0.0000m（无人机已停）
无残留 collect/gz topic 进程
```

同时观察到中断步留下 `rgb_00002.png`（无对应 actions 行），已在风险中记录。

### 5. 异常路径（M4/M3）：无 Gazebo server

```
python3 sim/tools/collect.py --steps 1 --no-takeoff \
  --out /tmp/uavrgb_ep_noserver
```

结果：

```
RC=1 wall=17.0s（旧版 gz topic -e 无 publisher 会永久阻塞）
stderr: step 0: depth read failed，终止
manifest: validity 全 false, num_steps=0, error="step 0: depth read failed"
```

### 6. 收尾

```
$ ps -eo pid,comm --no-headers | awk '$2=="ruby"{print}'
  (无输出)
$ ps -eo pid,comm,args --no-headers | grep -E 'gz sim|gz-transport-topic|collect.py'
  (无输出)
```

最终代码回归：加安全兜底后重跑 2 步 teacher（fresh world，hold=0.5），
`dx=0.126/0.084 m`、`action_wall=0.51/0.50 s`、`rgb<start=True`、RC=0。

## 风险与遗留

1. **动作窗口仍是 wall-clock**：`gz topic -p` 本机固定约 0.82 s 才发布；
   当前用“动作/零速两个 publisher 同时起跑、零速晚 hold 秒起跑”隐藏这段
   延迟，并记录真实 `action_wall`。这依赖两个进程发布延迟相近；已加
   `action_wall<=0` 检测 + 同步零速兜底，但 hold 极小时仍有反转风险。
   更正确的做法仍是 `/clock` sim-time 控制（下一轮）。
2. **RTF<1 时位移低于理论值**：hold=0.5 的 FORWARD 步实测 0.084–0.125 m，
   不是严格的 0.15 m；动作窗口 wall≈0.5 s，但软件渲染下 sim 时间偏短。
   数字已如实记录，不能用它反推精确 sim-time 动作时长。
3. **低空 teacher 死锁**：10 步 teacher 在 z=1.39–1.61 时三带皆堵，
   step 3 起 SAFE_NEUTRAL，未验证 YAW 动作的真实转角。属 S7/S8 遗留。
4. **碰撞覆盖**：旋翼 contact sensor 已加载（5 publisher），但旋翼单独擦碰
   未做物理复现；接触超时仍记 false，不能区分“没撞”和“没测到”。
5. **数据完整性**：中断步会留孤儿 RGB；actions/oracle 本身一致，
   manifest 的 `num_steps` 只按已 flush 的行计数。
6. **SDF 改名兼容性**：`pillar_right` 已改名 `pillar_body_left`，
   `sim/` 代码无引用，但 `docs/` 旧记录仍用旧名（按约束未改）。
7. **manifest 相机字段**仍硬编码 320×240，未从实际帧回填。
8. **cmd_twist 失败分支**未现场触发（无法人为让 publish 返回非 0），
   只做了代码审查；`gz topic -e` 超时有真实无 server 测试。

## 更新 2026-10-29 00:20（collect.py 审核跟进：takeoff / out-dir / 接触 / 深度）

> 并行会话本轮只改了 `teacher.py`（`turn_dist=1.5, emergency=0.8`）；下面是我按审核意见对 `collect.py`、`README.md`、SDF 注释和两个自检文件的修改。

### 做了什么

- `sim/tools/collect.py`：
  - `Z_TARGET` 1.8→**2.1 m**（高于 1.5 m 地标顶，S8）；`take_off` 改为按 `read_pose()` 闭环升到 `z_target`，超时 30 s，不再固定 2.5 s wall-clock（S7）。
  - 起飞前检查初始高度：`z>0.5 m` 直接报错“world 未复位”，避免复用上一 episode 的状态。
  - `--out` 非空目录直接 argparse 报错（EXIT=2），避免截断旧 `actions/oracle` 且留下旧 RGB 帧。
  - `contact_fired`：JSON 非 object（null/list）按无接触处理，不再 `AttributeError` 中断整集。
  - `read_depth`：校验 `pixelFormatType/format` 必须是 FLOAT32，防止 float64/其他格式被静默按 float32 重解释。
  - `actions.jsonl` 新增 `depth_ts` / `depth_rgb_gap_ns` / `action_start_ns`，M1 的“RGB 早于动作”和深度/RGB 时差可事后审计（不再只在控制台）。
- `sim/README.md`：盲策略撞柱示例改 `--steps 10 --hold 1.5`（旧 8 步/0.5 s 在新步长下已到不了柱子）；补充 actions 字段、非空 `--out`/未复位 world 行为、2.1 m 闭环起飞、深度/RGB 时差说明；`/uav/contact` 话题表改为“机体 + 四旋翼”。
- `sim/worlds/uav_rgb.sdf`：contact 插件注释从“仅 base_link”改为“base_link 或任一旋翼”；几何未动。
- `sim/tools/test_collect.py`（新）：13 项纯函数自检（contact 的 null/list/空/坏 JSON/超时/rc；深度紧凑、padded、float64/RGB/截断拒绝）。
- `sim/tools/test_save_frame.py`（新）：10 项自检（stride padding、BGR/RGBA、灰度/16-bit/截断/非法尺寸 rc=1 且不建文件），替代已删除的 `/tmp/uavrgb_saveframe_test.py`。

### 没做什么

- **没有做深度/RGB 同帧**：本次实测 `depth_rgb_gap_ns=168–308 ms`，已落盘供下游过滤；同帧需要一次订阅同时收两个话题，按审核结论留到下一轮。
  - 原因：改动大，且当前两步都在动作前，因果方向正确。
- **没有把 `--hold` 改成 sim-time**：动作窗口仍是 wall-clock（实测 0.47–0.59 s for hold 0.5），依赖两个 publisher 发布延迟相近；`action_wall<=0` 兜底已在上一轮。
  - 原因：需要 `/clock` 或持久 publisher，超出本轮 collect 审核修复。
- **没有消除起飞过冲**：目标 2.1 m，零速时 z=2.20 m，第一步 pose 2.636 m，随后回落到 2.31 m；仍高于 1.75 m 安全线且 teacher 不再卡死，但高度不是精确 2.1。
  - 原因：速度控制器/物理惯性；要精确需 sim-time 或提前减速，留作下一轮调参。
- **没有改地标 collision / 相机 / SDF 几何**：Z_TARGET=2.1 已让飞行高度高于 1.5 m 地标顶；审核 S8 的“地标去 collision”未做。
  - 原因：本轮按“最小改动”只改高度目标，避免改世界语义。
- **没有 live 验证旋翼单独擦碰**：SDF 的 5 个 contact publisher 已由并行会话用 `gz topic -i` 确认，但“旋翼碰、机体没碰”的物理场景未复现；接触超时仍记 `false`（S5 剩余）。
  - 原因：需要专门摆位/碰撞实验。
- **没有处理中断步的孤儿 RGB**（S6 剩余）、`camera` 宽高仍硬编码 320×240、`--hold` 未设下限（仅 >0）。
  - 原因：审核“可以改/部分修复”项，留到下一轮。
- **没有改 `teacher.py`**：`turn_dist=1.5/emergency=0.8` 是并行会话的修改；本 worklog 的实跑使用该版本。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python3 sim/tools/test_collect.py` | 13 项通过，EXIT=0 |
| `python3 sim/tools/test_save_frame.py` | PASS（10 项），EXIT=0 |
| `python3 sim/tools/teacher.py` | 22/22 通过（并行会话版本） |
| `py_compile`（collect/teacher/save_frame/两个测试） | EXIT=0 |
| `gz sdf -p sim/worlds/uav_rgb.sdf` | EXIT=0 |
| live teacher 10 步（fresh world，`--hold 0.5`） | 起飞 `z=2.20 m`；10/10；动作 8×FWD/1×LEFT/1×FWD；无 SAFE_NEUTRAL；`COLLECT_RC=0` |
| live `depth_rgb_gap_ns` | 168–308 ms，字段已写入 actions |
| 非空 `--out` 重跑 | `error: --out ... 不是空目录`，EXIT=2 |
| 复用 world 第二次采集 | 起飞前 `z=2.30 m` → `RuntimeError: world 未复位`；manifest validity 全 false，EXIT=1 |
| 收尾 | `pgrep -af 'gz sim|ruby'` 无输出；无残留 collector/sim |

### 风险与遗留

- `Z_TARGET=2.1` + 闭环起飞会过冲到 2.6 m 再回落；若后续相机/障碍高度变化，需要重新标定。
- “world 未复位”检查以 `z>0.5 m` 为阈值；`--no-takeoff` 会绕过它（有意保留给已在空中的测试）。
- 非空 `--out` 一律拒绝；想在同一目录重跑需手动清空，或换新目录。
- 动作窗口、深度/RGB 同帧、SDF 地标、旋翼接触、orphan RGB 仍是下一轮事项；本次数据可用于链路验证，不应直接当正式训练集。
