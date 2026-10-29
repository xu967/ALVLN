# E3 推进：碰撞真值（contact sensor）+ episode 文件格式落地

- 时间：2026-10-28 23:27
- 范围：`sim/worlds/uav_rgb.sdf`、`sim/tools/collect.py`、`sim/README.md`
- 结论：完成。接触传感器实测出真信号；盲策略（恒定前进）撞柱时 `collision=True` 落盘；episode 文件（manifest/actions/oracle/RGB）按冻结 `EpisodeStep`/`EpisodeMeta` 写出。**随机化未做**（任务 3 未碰）。

## 做了什么

### 1. 碰撞真值（任务 1）

- `sim/worlds/uav_rgb.sdf`：
  - 世界级加 `gz-sim-contact-system`（`gz::sim::systems::Contact`）插件。
  - `x3/base_link` 内加 `type="contact"` 传感器，绑定 `base_link_collision`，话题 `/uav/contact`，100Hz。
  - 该话题是**稀疏/触发式**的：只在发生接触时发 `gz.msgs.Contacts`，无接触时静默。
- `sim/tools/collect.py`：
  - 每步在发指令**之前**起一个 `gz topic -e -t /uav/contact -n 1 --json-output` 监听（Popen），
    动作窗口结束后 `communicate(timeout=0.5)` 收尾：收到非空 `contact` 列表=接触，超时无消息=无接触。
    实测可读（见「验证」），**不用位姿卡滞做唯一判定，而是两种真实信号并存**：
  - `contact`（接触传感器，主信号）；
  - `stuck`（位姿卡滞兜底：`FORWARD_SLOW` 指令下 hold 窗口位移 < 3cm 判被挡；
    正常 0.5s 前进约 0.15m，被柱挡住位移≈0）。
  - 任一触发即 `collision=True`；两者都触发时控制台打印 `contact+stuck`。
- 加了 `--policy forward`（盲策略：每步恒定 FORWARD_SLOW）——专家策略会主动绕柱，
  没有撞柱就不可能有碰撞样本；盲策略是 doc 02 §五 要求的恒定动作对照基线，也是
  验证碰撞真值的唯一真实途径。

### 2. episode 文件格式落地（任务 2）

- `collect.py` 新增 CLI：`--scene-id/--seed/--episode-id/--split(train|val|test)/--version`。
- 输出（与冻结 `EpisodeStep`/`EpisodeMeta` 对齐）：
  - `episode_manifest.json`：`schema/episode_id/scene_id/seed/version/split` +
    `actions_ref/oracle_ref` + `camera`（320×240 实际值，fx/fy/cx/cy 保持 null，
    未标定）+ `validity` 三标志（`camera_verified=false`、`physics_verified=true`、
    `episode_executed=true`，均为实测事实）+ `num_steps`。
  - `actions.jsonl`：每步 `step/timestamp_ns/action_expert/action_executed/collision`
    （`timestamp_ns` 取该步 RGB 帧的 `header.stamp` sim-time，与图像关联）。
  - `oracle.jsonl`：每步 `step/timestamp_ns/pose_gt`（`/model/x3/pose` 真值，
    只进 oracle 文件，不进学生 RGB 输入）。
  - `rgb_%05d.png`：320×240 原图。
- 两个文件均真实跑出（见「验证」）。scene_id/seed 由命令行传入；train/val/test
  按场景隔离（本步固定单场景，split 字段已预留）。

### 3. 过程中修的真实 bug（都实测复现）

- **`/model/x3/pose` 有多个 publisher，`-n 1` 竞态下 stdout 混入 2 个 JSON 文档**
  → `echo_json` 只取第一行 + 失败重试 3 次。
- **多次重启 gz sim 会叠出多个 server**：`gz` CLI 是 Ruby 脚本，server 进程
  comm 是 `ruby`，`pkill -x gz` 杀不掉它（`pgrep -x gz` 也永远查不到）。
  一度 3 个 server 同时在跑，把 pose/contact 话题都污染了（一次采集里读出
  `y=41,z=6.3` 这种不可能位姿、碰撞信号可能来自另一个 sim 的无人机）——
  已按 PID 全部杀掉、单 server 重跑验证，**交付证据全部来自单 server 干净重跑**。
  后续杀进程用 `ps -eo pid,comm | grep ruby` 找 PID。

## 没做什么（如实）

- **场景随机化没做**（任务 3）：柱子/地标位置仍是固定 SDF，`--seed` 只写进
  manifest，不改变场景。原因：任务优先级 3 明确排在 1/2 之后。
- **精确 sim-time 对齐没做**：深度/位姿/RGB/接触是四次独立的 `gz topic -e` 快照，
  非同一 sim-time；每步的指令实际生效时长 = 整步 wall-clock（深度 JSON 解析约
  1s + 各 topic echo 开销），`--hold 0.5` 只是下界——本机实测每步 sim-time
  前进约 2.3s、无人机每步前进约 0.75m。原因：精确对齐需照 doc 03 的
  `advance_until(applied_or_timeout)`，属后续改造。
- **128×128 降采样没做**：RGB 存 320×240 原图，留给训练预处理（doc 04）。
- **episode 间 reset/复位校验没做**：collect.py 假定每次跑前是全新 world；
  同一 server 连跑两个 episode 时，前一 episode 的末状态（含残存 twist 指令）会
  污染下一个（实测复现：上一 run 的 yaw 姿态让「前进」变成侧飞）。属 GZ1 验收项。
- **相机内参标定没做**：fx/fy/cx/cy 保持 null。
- **接触事件没有存原始 Contact 消息进 episode**：collect.py 只把接触判成 bool 写
  `collision` 字段；原始 gz.msgs.Contacts 证据存于
  `sim/evidence/contact_pillar_hit.json`（`base_link_collision` vs
  `pillar_center::link::collision`，normal=(-1,0,0)，柱面 x=2.8）。
- **深度真值没进 oracle.jsonl**（当前只有 pose_gt）：深度是特权输入，若训练需要
  可后续加 `depth_ref` 字段。

## 验证到哪一步（真实命令 + 输出）

| 项 | 命令 | 结果 |
| --- | --- | --- |
| SDF 校验 | `gz sdf -p sim/worlds/uav_rgb.sdf` | 通过（含 contact sensor + Contact 系统） |
| 单 server 启动 | `DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg gz sim -s -r sim/worlds/uav_rgb.sdf` | `/uav/contact` 话题出现，publisher 唯一 |
| 接触正样本 | 起飞→前进撞 `pillar_center`，`timeout 2 gz topic -e -t /uav/contact -n 1 --json-output` | 0.19s 内收到：`collision1=x3::base_link::base_link_collision`、`collision2=pillar_center::link::collision`、`normal=(-1,0,0)`、位置 x=2.8（柱面） |
| 接触负样本 | 悬空时同样 echo | `timeout` 2s 超时（exit 124），0 字节输出=无接触 |
| 盲策略撞柱采集 | `python3 sim/tools/collect.py --steps 8 --policy forward --out /tmp/ep_crash --scene-id 1 --seed 43` | 见下方输出 |
| 专家策略采集 | `python3 sim/tools/collect.py --steps 6 --out /tmp/ep_test --scene-id 1 --seed 42` | 见下方输出 |
| teacher 自测 | `python3 sim/tools/teacher.py` | 6/6 通过（未改动 teacher.py） |

盲策略撞柱 run（单 server，干净）关键输出：

```
step 0..3: FORWARD_SLOW x:0.07→0.84→1.62→2.40  collision=False(-)
step 4:    FORWARD_SLOW x=2.5628                collision=True(contact+stuck)
step 5..7: FORWARD_SLOW x=2.5628（冻结）        collision=True(contact+stuck)
```

- x 冻结在 2.5628：柱面 x=2.8 - 机体半宽 0.235 ≈ 2.565，物理上贴着柱子，
  与接触传感器几何一致（两种信号互相印证）。
- `actions.jsonl` 落盘：step 4-7 `"collision": true`，其余 false（负控制：自由飞行
  4 步无一误报）。
- 专家策略 run（/tmp/ep_test）：6 步动作 `[1,1,2,3,2,0]`，全程未达柱子，
  `collision` 全 false——专家主动绕开，如实为 false，不造假。

episode 文件样例（/tmp/ep_crash/episode_manifest.json）：

```json
{"schema":"navlab.episode.v1","episode_id":43,"scene_id":1,"seed":43,
 "version":"1","split":"train","actions_ref":"actions.jsonl",
 "oracle_ref":"oracle.jsonl",
 "camera":{"width":320,"height":240,"fx":null,"fy":null,"cx":null,"cy":null},
 "validity":{"camera_verified":false,"physics_verified":true,"episode_executed":true},
 "num_steps":8}
```

## 风险与遗留

- **接触是瞬时量、按步采样**：若无人机「擦碰后弹开」且弹开发生在两步之间，
  接触监听可能错过；此时靠 stuck 兜底（弹开时前进位移≈0）。极端侧滑擦过
  （持续位移但短暂接触）两种信号都可能漏——本场景柱面正对飞行方向，实测未出现。
- **命令残存**：twist 指令在下一步覆盖前一直生效；采集脚本异常退出后无人机
  会继续按最后指令飞（实测撞柱后停不下来，需 kill server）。
- **多 server 污染**：并行起多个 gz sim 会共用 transport 命名空间互相污染话题
  （pose 双 publisher 竞态、碰撞信号串台）。采集前必须确认只有一个 server。
- **专家策略的左右摆振**：本次专家 run 出现 `2,3,2`（左-右-左）抖动并伴随高度
  下滑（z 1.78→1.23），即 E2 worklog 提过的「无滞回」问题；建议正式采集前给
  teacher 加滞回，或调 min_clear。本步未改 teacher（保持 6/6 自测不动）。
- 采集吞吐低（每步约 2.8s wall-clock，RTF≈0.8 软件渲染）；大批量采集建议换
  gz-transport 二进制通道或 C++ 采集器。
