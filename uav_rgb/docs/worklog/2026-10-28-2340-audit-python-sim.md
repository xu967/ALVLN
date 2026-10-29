# 审核报告：自写 Python 工具 + 仿真世界（只读）

- 时间：2026-10-28 23:40 CST
- 范围：`ui/monitor.py`、`ui/test_monitor.py`、`ui/launch_ui.sh`、`/root/桌面/uavrgb-monitor.desktop`、`sim/tools/save_frame.py`、`sim/tools/teacher.py`、`sim/tools/collect.py`、`sim/worlds/uav_rgb.sdf`
- 方式：静态阅读 + 纯函数/纯几何测试 + `gz sdf -p`；**未启动 Gazebo、未运行 collector、未改任何源码/配置**。唯一写入是本报告（以及 `/tmp` 下的临时脚本/输出）。
- 结论先行：监控台（Tk）当前版本质量较好（50 项自检全过，之前的分母为零/空 epoch/after 异常问题已修）；但**采集链路有 P0 级数据错位/数据丢失风险，教师策略在深度失效时是 fail-open（默认前进）**，这批数据在修复前不能当正式训练集。

## 0. 审核版本（并发会话仍在改 UI/collect，行号以本表为准）

| 文件 | 行数 | mtime | sha256（前 16 位） |
|---|---:|---|---|
| `ui/monitor.py` | 531 | 2026-10-28 23:35:40 | `503df19c7d712360` |
| `ui/test_monitor.py` | 250 | 2026-10-28 23:29:51 | `8f38d4cc2412b877` |
| `ui/launch_ui.sh` | 14 | 2026-10-28 23:26:09 | `1055aaad39e8766f` |
| `/root/桌面/uavrgb-monitor.desktop` | 10 | 2026-10-28 20:53:42 | `76833962aedb1a9b` |
| `sim/tools/save_frame.py` | 44 | 2026-10-28 20:51:49 | `e7f4693d81b85a2a` |
| `sim/tools/teacher.py` | 136 | 2026-10-28 23:30:18 | `8288b0c0b176ee64` |
| `sim/tools/collect.py` | 288 | 2026-10-28 23:32:28 | `f0ffc1028177144a` |
| `sim/worlds/uav_rgb.sdf` | 399 | 2026-10-28 23:11:40 | `1fc0634af375e110` |

> **并发修改警告**：`ui/` 与 `collect.py` 在审核期间被另一个会话持续改写（monitor 从 510→531 行、collect 从 254→288 行）。上面是“我实际测到的版本”；若当前文件 hash 已变，UI/collect 的具体行号可能漂移，请以 hash 为准复核。`save_frame.py`、`teacher.py`、`uav_rgb.sdf` 在审核末段保持稳定。

## 1. 必须修（5 条）

### M1. RGB 观测在动作之后抓取，标签与图像因果倒置
- 位置：`sim/tools/collect.py:187-208`（`read_depth` 188、`expert_action` 194、`cmd_twist` 200、`sleep` 201、`read_rgb_png` 208）；文档字符串 `:10-11` 也明确写“先动作→后抓 RGB”。
- 这是什么：每步的 `action_expert` 由**动作前**的深度算出，但唯一一帧 RGB 在 hold+读位姿+接触监听之后才抓；`timestamp_ns` 用的是这帧“动作后”的 RGB。
- 失败场景：项目自己的真实 episode `/tmp/ep_test` 显示每步 sim-time 约 2.8 s；step 0 的 pose_after 在 sim 18.0 s、RGB 在 18.7 s，而动作由这帧 RGB 之前的深度决定（深度帧时间戳没有落盘，但按代码顺序在 `cmd_twist` 之前）。也就是学生看到的是“前进/左转 0.8 m 之后”的画面，却配“前进/左转”标签；YAW 步在 0.6 rad/s 下持续约 2.8 s≈96°，图像早已转到新朝向。用这批数据训练等于学 `f(结果状态)=导致它的动作`，而不是 `f(观测)=动作`。
- 最小修法：在 `cmd_twist` 之前抓 RGB（最好深度/RGB 时间戳差 ≤1 帧），用该帧时间戳写 `EpisodeStep.timestamp_ns`；动作后的状态只进 oracle（pose_gt），不要拿来配标签。

### M2. 动作指令从不主动停止；`--hold 0.5` 实际执行约 2.8 s sim-time
- 位置：`sim/tools/collect.py:200-202`（发 twist → `sleep(hold)` → 直接读 pose/contact/RGB，没有零速）；`:226` 的 `--hold` help 仍写“每动作持续时间”。
- 这是什么：速度指令会一直生效到下一步覆盖；读 pose、等 contact、抓 RGB 的 `gz topic -e` 开销全部落在同一个动作窗口里。代码自己的注释 `:57-59` 也承认“一步实际约 2.3s sim-time”。
- 失败场景：`/tmp/ep_test`/`/tmp/ep_crash` 的相邻 pose 时间戳差 2.3–2.9 s；FORWARD 步每步 x 前进约 0.76 m（0.3 m/s×2.5 s），而 `--hold 0.5` 按设计只该 0.15 m；YAW 步实际转约 90° 而不是 17°。同一场景在不同 RTF/机器负载下动作时长还会漂移（RTF 0.8→0.5 时位移/转角再变），episode 不可复现。
- 最小修法：`sleep(hold)` 后立刻发零速/`SAFE_NEUTRAL`（高度保持分量给 0），再读 pose/contact/RGB；更正确的做法是用 `/clock` 或 pose header 的 sim-time，按**目标 sim 时长**停止，而不是 wall-clock sleep。contact listener 要活到停稳之后再收。

### M3. 异常/失败路径没有 finally 清理：无人机继续飞、已采行丢失、manifest 仍写“已执行”
- 位置：`sim/tools/collect.py:238-253`（主循环）、`:255-285`（循环结束后才写文件）、`:267-268`（validity 硬编码）。
- 这是什么：没有 `try/finally`，也没有在 `break`/异常时发零速；`rows` 只在循环结束后落盘。
- 失败场景：step 5 深度读取失败（`r is None`）→ `break`；此时最后一条 twist 仍是 FORWARD_SLOW，脚本退出后无人机继续向柱子飞；step 0–4 的 actions/oracle **一行都没写**；随后仍写 `episode_manifest.json`，其中 `episode_executed:true`、`physics_verified:true`、`num_steps:0`。Ctrl-C 同理，且连 manifest 都没有。`/tmp/ep_crash` 结束后无人机就停在柱子上（x=2.5628）继续顶，下一 episode 若复用同一 server 会从柱子上开始。
- 最小修法：`try/finally` 里无条件发零速；行数据在 finally 或每步增量写；validity 由实际数据决定（0 步/失败步骤 → false）；manifest 最后写并用临时文件 `os.replace` 原子替换。

### M4. `gz topic -e` 没有超时；`cmd_twist` 不检查返回码
- 位置：`sim/tools/collect.py:90-91`（`subprocess.run` 无 `timeout`）、`:98-110`（`echo_json` 只处理非零退出/空 stdout）、`:94-95`（`cmd_twist` 丢弃 `_gz` 结果）。
- 这是什么：`gz topic -e -n 1` 在话题没有 publisher 时会**永久阻塞**，不是返回非零。
- 失败场景（已实测）：`timeout 3 gz topic -e -t /audit/none -n 1 --json-output` → `RC=124`，0 字节输出。对应到本世界：README 已记录“纯 headless 相机渲染会 segfault、`/uav/camera` 不发布”；或 topic 名写错/传感器插件加载失败时，`read_depth`/`read_pose` 会永远卡住，collector 不退出、不落盘；如果卡在第一条 twist 之后，无人机保持最后指令继续飞。`cmd_twist` 返回非零（话题名错/消息格式错）也被忽略，`action_executed` 照样写成请求动作。
- 最小修法：`_gz(..., timeout=5)` 并捕获 `TimeoutExpired`，失败按“读失败”处理；`cmd_twist` 检查 returncode，失败则中止并发零速；`action_executed` 应来自实际回执或明确标注“请求值”。

### M5. 教师策略 fail-open：全无效/全零/NaN 深度被当成“前方空旷”→ FORWARD_SLOW
- 位置：`sim/tools/teacher.py:24-33`（`_strip_min` 跳过 `d<=0`，全无效返回 `inf`）、`:44-52`（`center >= min_clear` → `FORWARD_SLOW`）。
- 这是什么：模块 docstring `:4` 自己定义“`<=0` 表示无读数/太近”，但代码把“无读数”当成“无限远/空旷”。
- 失败场景（实测输出）：`all_zero_depth -> FORWARD_SLOW`、`all_nan_depth -> FORWARD_SLOW`、`all_negative_depth -> FORWARD_SLOW`；`center_all_zero_sides_0.3 -> FORWARD_SLOW`。即深度相机没出图、格式错、被黑/反光面吞掉、或障碍物近于近裁剪面时，专家会命令前进撞上去。这是安全方向的错误默认值，比“保守停住”危险得多。
- 最小修法：每个带统计“有效样本数”；中心带有效样本为 0 → `SAFE_NEUTRAL`（fail closed）；侧带无有效样本 → 视为堵；同时校验 `w/h>=…` 与 `len(depth)==w*h`。

## 2. 应该修（10 条）

### S1. teacher 的小尺寸深度：h=1/3 采不到带，h=2 只看顶行，w<3/短 buffer 还会 IndexError
- 位置：`sim/tools/teacher.py:44-49`；`_strip_min` 索引 `:29`。
- 失败场景（实测）：h=1 → 带 `[0,0)` 为空 → 三带全 inf → `FORWARD_SLOW`；h=3 → `[1,1)` 为空 → 同样 `FORWARD_SLOW`；h=2 → 只采样 row 0（图像顶部/天空），构造“顶行 10 m、底行 0.1 m 障碍”仍返回 `FORWARD_SLOW`。若下游对深度降采样到 2–3 行（或单测/裁剪），专家直接失效。`len(depth)` 短于 `w*h` 时实测 `IndexError: list index out of range`，会打断采集且没有清理（见 M3）。w<3 时左/中带为空，中心 inf → 前进。
- 最小修法：`y1 = max(y1, y0+1)` 并 clamp 到 h；要求 `h>=4` 或按最近行采样；校验 `w>=3` 和 `len(depth)==w*h`，不满足直接拒绝/SAFE_NEUTRAL；补 h=1/2/3、w=1/2、短 buffer 的单测。

### S2. save_frame 的行 stride 用错：有 padding 时第 2 行起像素整体错位
- 位置：`sim/tools/save_frame.py:18-19`（`_png` 按 `w*nch` 紧凑切行）、`:34`（`nch = step // w`）、`:36`（`buf = raw[:h*step]`）。
- 失败场景（实测）：2×2 RGB、`step=7`（每行 1 字节 padding）时输出第 1 行像素为 `(170,10,20),(30,40,50)`，正确应为 `(10,20,30),(40,50,60)`（0xaa padding 被当作 R）；4×2、step=13 同样错位；320×2、step=1024（64 字节 padding）时第 2 行开头变成 `(204,204,204)…`（0xcc padding）。当前世界实际帧 `width=320,height=240,step=960` 是紧凑的，所以证据帧没暴露；换相机/格式/分辨率或任何带 stride 的 `gz.msgs.Image` 就会静默存错图。
- 最小修法：按行切 `raw[y*step : y*step + w*nch]`；`nch` 从 `pixelFormatType`/`format` 推断而不是 `step//w`；校验 `len(raw) >= (h-1)*step + w*nch`。

### S3. save_frame 的非 3/4 通道、截断 payload、BGR 声明
- 位置：`sim/tools/save_frame.py:18`（`{3:2,4:6}[nch]`）、`:36-38`（先 open 再算 PNG）、docstring `:4-5`。
- 失败场景（实测）：灰度 2×1、`step=2` → `nch=1` → KeyError，rc=1，但输出文件已被 `open(dst,'wb')` 创建成 0 字节；16-bit RGB、`step=6` → `nch=2` → 同样。截断 payload（2×2 只给 8 字节）→ rc=0，写出一个 PIL 报 “image file is truncated” 的坏 PNG，下游看不出失败。docstring 声称支持 B8G8R8/BGRA，但代码完全不看 `pixelFormatType`、不换 R/B（真实帧的 JSON 是 `pixelFormatType:"RGB_INT8"`；若变成 BGR，颜色会静默对调）。
- 最小修法：先构造 PNG 字节再 open/write；长度不足直接报错非 0；对 nch 非 3/4 明确报错或支持灰度/16-bit；按 `pixelFormatType` 做 BGR/BGRA→RGB/RGBA 或把 docstring 收窄到 R8G8B8/RGBA。

### S4. collect 的 RGB/深度读取复用了同样的 stride/格式假设
- 位置：`sim/tools/collect.py:128-139`（`nch=step//w` + `save_frame._png`）、`:118-125`（`n = len(raw)//4` 后 teacher 按 `y*w+x` 索引）、`:122` 只取 `w,h,step`，不看 `pixelFormatType`。
- 失败场景：深度 `step != 4*w` 时，padding 的 float 被当成像素插入，teacher 的行/列整体错位（它按紧凑 `y*w+x` 索引），本应“前方 1.5 m 有柱”可能读成别的带；RGB 则同 S2 静默错图。当前世界 RGB step=960、深度预计 4×320=1280 是紧凑的，属潜伏问题；一旦改分辨率/格式/加 padding 就触发。
- 最小修法：与 S2 共用“按 step 切行 + 格式校验”的解码函数；深度校验 `len(depth)==w*h`，不满足拒绝该步（并配合 M3 的清理）。

### S5. 碰撞真值覆盖不全：只测 base_link、监听在动作结束前被杀、stuck 只对 FORWARD 生效
- 位置：`sim/tools/collect.py:153-174`（listener 生命周期/超时/解析失败一律 False）、`:204-206`（stuck 仅 `action == FORWARD_SLOW`）；`sim/worlds/uav_rgb.sdf:204-211`（只绑 `base_link_collision`）、`:150-155`（机体盒 0.47×0.47）、`:224-285`（旋翼半径 0.1，中心 ±0.22，翼尖跨度约 0.64 m）。
- 失败场景：YAW 贴着柱子转，旋翼翼尖（距中心 0.32 m）碰到柱面，而机体半宽只有 0.235 m；contact sensor 看不到旋翼碰撞，`stuck` 又只对 FORWARD 生效 → 这一步记 `collision:false`。另外 listener 在 `pose_after` 之后最多等 0.5 s 就被 kill，而动作还继续到读 RGB（见 M2），这段时间内的碰撞也漏。超时/JSON 解析失败/`gz topic` 非零退出全部静默当“无碰撞”，无法区分“没撞”和“没测到”。
- 补充：项目自己的 `/tmp/ep_crash` 证明正面撞柱是可测的（step 4 起 `collision:true`，x 冻结在 2.5628≈柱面 2.8−机体半宽 0.235），所以不是“恒 False”，而是**覆盖有洞**。
- 最小修法：给四个旋翼 collision 也挂 contact sensor（或加一个包住整机的碰撞体）；listener 保持到零速停稳后再判；把“读失败/超时”记成 unknown（或在 manifest 里计数），不要等同 false。

### S6. episode manifest/actions 的 validity 与 RGB 关联不完整
- 位置：`sim/tools/collect.py:208-218`（`rgb_ok` 只在内存行里，`actions.jsonl` 不写）、`:231-235`（camera 宽高硬编码 320×240、无 RGB 文件模式/引用）、`:262-285`（先写 manifest 再写 JSONL）、`:267-268`（`physics_verified:true`、`episode_executed:true` 硬编码）。
- 失败场景：step 3 相机 echo 失败 → `read_rgb_png` 返回 `False,0`，但这一行仍进 `actions.jsonl`（timestamp_ns=0），`rgb_00003.png` 不存在；下游预处理若按 step 打开 PNG 会 FileNotFoundError 或错位。0 步 run 也会写 `episode_executed:true, physics_verified:true`，manifest 却比 JSONL 先落盘，崩溃时留下“看起来完整”的 manifest。
- 最小修法：`actions.jsonl` 每行带 `rgb_ok`/`frame`（或 manifest 加 `rgb_pattern` + `num_rgb`）；缺 RGB 的步要么丢弃要么标 invalid；validity 由实际数据生成；manifest 最后写、原子替换；输出目录必须为空或每次唯一，避免旧 `rgb_*.png` 残留。

### S7. 起飞/高度控制仍是 wall-clock，且目标高度只有 1.8 m，低 RTF 下前几步都在爬升
- 位置：`sim/tools/collect.py:181-184`（`take_off` 固定 `time.sleep(2.5)`，`TAKEOFF_VZ=0.5`）、`:56-62`（`Z_TARGET=1.8`、`Z_DEADBAND=0.20`、`Z_KP=0.25`）、`:80-87`。
- 失败场景：RTF=0.5 时 2.5 s wall 只等于 1.25 s sim → z≈0.15+0.625=0.775 m，而不是 1.8 m；后续每步的高度 P 控制也在 ~2.8 s 的控制周期里慢慢爬，前若干步的 RGB/深度都是低空视角，且可能被 1 m 地标挡住（见 S8），导致同一 `--scene-id/--seed` 在不同机器负载下产出不同 episode。2.5 s 从 0.15 m 以 0.5 m/s 理论只到 1.4 m，本身就够不到 1.8 m（实测 1.86 m 是控制器/时序过冲的结果，不保证）。
- 最小修法：用 `/clock` 或 pose header 的 sim-time 控制起飞，直到 z 达到 1.8±0.05 再开始采；hold 同样改用 sim-time；若坚持 wall-clock，至少按 RTF 缩放并记录实际 z。

### S8. 高度 deadband 与 1 m 地标的几何冲突；0.5 m 缝窄于 0.64 m 旋翼跨度
- 位置：`sim/tools/collect.py:56-62,80-87`；`sim/worlds/uav_rgb.sdf:76-109`（红/绿/蓝三个 1×1×1 地标，z 0.5–1.5，x 2.5–3.5）、`:111-134`（两根柱子）。
- 失败场景（纯几何 ray-cast + teacher，实测）：相机 x=0.8 m、z=1.63 m 时三带最小深度为 `1.70/1.70/1.70` → `SAFE_NEUTRAL`；z=1.68 m 时变为 `2.08/2.00/2.12` → `YAW_RIGHT`。因为 `Z_TARGET=1.8` + `Z_DEADBAND=0.2` 允许稳定在 1.60–2.00 m，若高度落在 1.60–1.65，无人机在 x≈0.6–0.8 看到三向皆堵 → SAFE_NEUTRAL，而 |err|<0.2 又让高度不再修正 → **永久卡住**。本次实测稳定在 1.86 m（高于阈值），所以是“余量只有约 5 cm 的脆弱设计”，不是每次必现。
- 另外：三个地标在低空形成 y∈[-2,-1]、[-0.5,0.5]、[1,2] 的墙，缝只有 0.5 m；机体 0.47 m 勉强过，旋翼跨度约 0.64 m 过不去，而旋翼碰撞又不在 contact sensor 覆盖内（S5）——低空试图穿缝会物理撞上但 `collision:false`。
- 最小修法：把 `Z_TARGET` 提到 ≥2.1（或 deadband 降到 ≤0.05，保证最低稳定高度 >1.75）；把 1 m 地标改成纯 visual（去掉 `<collision>`）或移出 x=3 障碍面；若地标要当障碍，缝宽要 ≥1 m 并给旋翼加接触检测。

### S9. SDF 注释/命名与实际左右相反：所谓“左开”其实是机体右侧开
- 位置：`sim/worlds/uav_rgb.sdf:111-112`（注释“center + right blocked, left open -> teacher must YAW_LEFT”）、`:124-134`（`pillar_right` 在 `y=+1.4`）、`:99-109`（`landmark_blue` 在 `y=-1.5`）。
- 失败场景：SDF/ROS 约定 +Y=左。`pillar_right` 在 y=+1.4，实际是**机体左侧**；`landmark_blue` 在 y=-1.5 是机体右侧，且在 1.8 m 巡航高度时顶部 1.5 m 低于 teacher 的中间带（见 ray-cast），所以右侧是空的。实测 ray-cast：`(0.8,1.80)` → 左 2.08/中 2.00/右 inf → `YAW_RIGHT`；项目自己的 live run 也记到 step 3=3（YAW_RIGHT），后续 worklog 写“FWD, SAFE, RIGHT, FWD×7”。任何按 SDF 注释写“必须 YAW_LEFT”的验收/单测都会失败；E2 worklog 里的“中/右堵、左开，逼专家 YAW_LEFT”同样错误。
- 最小修法：若要“左开”，把 `pillar_right` 移到 y=-1.4（并同步红/绿地标）；否则把注释/模型名改成“body-left blocked at y=+1.4, body-right open → YAW_RIGHT”，并更新 E2 worklog/README。

### S10. monitor 的增量 tail 在“同 inode 截断后写回更大文件”时丢事件
- 位置：`ui/monitor.py:85-129`，判定在 `:99`（只有 `size < offset` 才 reset），shortcut 注释 `:89-90` 只声明“同 inode 同长度”不重读。
- 失败场景（实测）：旧文件 offset=316 字节；同一 inode 用 `open(path,"w")` 截断并写入 323 字节的新日志；`tail_events` 返回 `reset=False, got=[]`，4 条新事件全丢。真实场景：trainer 重启后复用同一 `metrics.jsonl`、在 5 s 轮询间隔内写完一个比旧文件更大的新日志（或日志轮转工具原地重写）——监控台从旧偏移切进新文件中间，头部丢失并可能拼出半行，旧事件还留在 `_events` 里。
- 最小修法：state 里存文件头指纹（如前 64 字节的 hash）或首行内容；每次轮询比对，变了就 reset 从头读；或者要求 trainer 只追加/原子替换（换 inode 已支持）。补一个“截断后写更大文件”的测试。

## 3. 可以改（5 条）

### C1. monitor 文件被删除时不清空旧事件；`limit=0` 语义反直觉
- 位置：`ui/monitor.py:93-96`（stat 失败 → `reset=False`）、`:129`（`events[-limit:]`；limit=0 时 `events[-0:]` 返回**全部**）。
- 场景：`metrics.jsonl` 被删/轮转走，状态栏显示“等待 metrics.jsonl”，但日志列表仍显示上一次的 200 条旧事件，容易被当成当前事件；`load_events(..., limit=0)`（外部调用）会返回全部而不是空。当前 GUI 只用默认 limit，影响小。
- 最小修法：删除/缺失时也清空 `_events`（或明确标注“最后已知”）；`limit<=0` 直接返回 `[]`。

### C2. 单点曲线会伪造 x 轴刻度（epoch 1 显示 1 和 2）
- 位置：`ui/monitor.py:181-183`（`xmax=xmin+1`）、`:201-204`（固定两个刻度）。
- 场景：只有 1 个 epoch（例如训练刚写出第 1 行）时，x 轴标签是 1、2，图上却只有一个圆点；不是崩溃，只是读数误导。
- 最小修法：`xmax-xmin<2` 时只画一个 x 刻度（`xmin`）。

### C3. test_monitor 未覆盖 tail 的“截断后变大”与缺失文件语义
- 位置：`ui/test_monitor.py`（250 行；现有 tail 用例覆盖追加/半行/截断变小/替换 inode，见文件 125-159 附近）。
- 场景：S10 的 bug 现有 50 项测试全绿也发现不了；没有 `TAIL_WINDOW`（>1 MiB 首次只读末尾）的断言。
- 最小修法：加“同 inode 截断后写更大文件 → reset=True/不丢头”“文件删除 → 事件清空”“>1 MiB 首次 tail 只读末尾”三条用例。

### C4. launch_ui.sh/桌面快捷方式只探测 tkinter，不探测 DISPLAY；Terminal=false 会吞掉启动错误
- 位置：`ui/launch_ui.sh:7-13`；`/root/桌面/uavrgb-monitor.desktop`（`Terminal=false`，`Exec` 写死绝对路径）。
- 场景：在无 DISPLAY 的终端/SSH 里运行 launch_ui.sh，`import tkinter` 成功（找到解释器），随后 `tk.Tk()` 抛 TclError；桌面启动（Terminal=false）时用户看不到任何错误，只看到窗口没出来。仓库迁移后 Exec 的绝对路径失效同理。当前桌面会话有 :20，属边缘场景。
- 最小修法：脚本里 `[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || { echo ... >&2; exit 1; }`；或把 stderr 重定向到 `~/.cache/uavrgb-monitor.log`；`Exec` 用相对/可配置路径（需要用户授权改桌面文件）。

### C5. collect 的 CLI 没有校验 `--hold/--steps`，负值会在起飞后抛异常
- 位置：`sim/tools/collect.py:224-226`。
- 场景：`--hold -0.1` → `time.sleep(-0.1)` 在 `cmd_twist` 之后抛 `ValueError: sleep length must be non-negative`；此时已发过动作、没有 finally 清理（M3），无人机继续飞、无文件。`--steps 0` 会写一个 `episode_executed:true` 的空 episode（S6）。
- 最小修法：argparse 校验 `hold>0`、`steps>=1`（或 0 时明确拒绝）；与 M3 的 finally 一起兜底。

## 4. 实测输出（纯函数/只读，2026-10-28 23:30–23:38）

### 4.1 必跑命令

```
$ python3 sim/tools/teacher.py
  9 项用例全部 ok（含 3 条滞回），teacher: 全部通过  RC=0

$ DISPLAY=:20 python3 ui/test_monitor.py
  test_monitor: 50 项通过  RC=0
$ env -u DISPLAY python3 ui/test_monitor.py
  ... skip GUI 冒烟（无显示）; test_monitor: 37 项通过，1 项跳过  RC=0

$ gz sdf -p sim/worlds/uav_rgb.sdf >/tmp/uav_rgb_parsed.sdf
  RC=0, 566 行解析后 SDF（含 contact sensor / Contact system / 两个相机）

$ desktop-file-validate /root/桌面/uavrgb-monitor.desktop
  RC=0（仅 Categories 可补 System/Network 的 hint）
$ bash -n ui/launch_ui.sh ; RC=0
$ python3 ui/monitor.py --refresh 0|nan ; RC=2（argparse 拒绝）
```

### 4.2 save_frame 边界（Pillow 读回）

```
1x1_rgb_step3      : rc=0 mode=RGB size=(1,1) pixels=[(10,20,30)]
2x2_rgb_step6      : rc=0 pixels=[(1,2,3),(4,5,6),(10,20,30),(40,50,60)]
2x2_rgb_step7_pad  : rc=0 pixels=[(1,2,3),(4,5,6),(170,10,20),(30,40,50)]   # 行 1 错位
4x2_rgb_step13_pad : rc=0 第 2 行=(170,21,22),(23,24,25),...                  # 0xaa padding 当 R
2x2_rgba_step8     : rc=0 mode=RGBA pixels=[(1,2,3,255),(4,5,6,128),(10,20,30,255),(40,50,60,0)]
2x1_gray_step2     : rc=1, 输出文件 0 字节/坏 PNG（nch=1 -> KeyError）
1x1_rgb16_step6    : rc=1, 0 字节/坏 PNG（nch=2 -> KeyError）
2x2_truncated      : rc=0, PNG CORRUPT: image file is truncated
320x2_step1024_pad : rc=0, 第 2 行开头 (204,204,204)...（64 字节 0xcc padding 混入）
```

PNG 结构独立校验（紧凑 2×2 RGB）：`IHDR/IDAT/IEND` 三个 chunk 的 CRC 全部 `True`，IHDR=`(2,2,8,2,0,0,0)`，IEND 完整 → **紧凑 stride 下 PNG/zlib.crc32 正确**。真实帧 `/tmp/uav_frame.json`（320×240，step=960，`pixelFormatType=RGB_INT8`）→ `saved ... 320x240 3ch sim_time=4.012000000`，Pillow 读回 `RGB (320,240)`。

### 4.3 teacher 边界

```
h=1  band [0,0) 全 0.1m -> FORWARD_SLOW
h=2  顶行 10m/底行 0.1m -> FORWARD_SLOW（只采样 row 0）
h=3  band [1,1) 全 0.1m -> FORWARD_SLOW
h=4/5/10/240 全 0.1m -> SAFE_NEUTRAL（正常）
all_zero_depth / all_nan_depth / all_negative_depth -> FORWARD_SLOW
center_all_zero_sides_0.3 -> FORWARD_SLOW
center=1.999 -> YAW_LEFT; center=2.0 -> FORWARD_SLOW; center=2.001 -> FORWARD_SLOW
left==right==5, center=1 -> YAW_LEFT（平局确定）
left=0.5 center=0.5 right=5 -> YAW_RIGHT
short depth list -> IndexError: list index out of range
```

### 4.4 monitor 边界

```
单点 (1,1.0)       : ok items=11 oval=1（不再除零，画圆点）
3 点同 x           : ok items=11 oval=0（xmin==xmax -> xmax=xmin+1）
3 点同 y           : ok items=12
全 y=None / 全 x=None / 空 -> "(无数据)"，不崩
1x1 canvas         : ok items=0（winfo<60/40 early return）
空 epoch CSV       : [(2,0.5),(None,1.0)]（坏行排最后，不崩）
全空 epoch CSV     : [(None,1.0),(None,0.5)]（曲线显示“无数据”）
nan/inf CSV        : [(1,None)]（过滤为 None）
截断后写更大文件   : old_off=316 new_size=323 reset=False got=[]（4 条全丢）
文件删除后         : ([], False)（旧事件不会被清）
_tick 内 refresh 抛异常: rescheduled=1, 状态栏='刷新失败: boom'（循环不死）
load_events(limit=0): 返回全部 2 条（events[-0:] 语义）
```

### 4.5 世界几何 ray-cast（320×240 深度，60° FOV，teacher 原函数）

```
camera(x,z)      left   center  right   teacher
(0.8,1.63)       1.70   1.70    1.70    SAFE_NEUTRAL
(0.8,1.68)       2.08   2.00    2.12    YAW_RIGHT
(0.8,1.80)       2.08   2.00    inf     YAW_RIGHT
(1.1,1.63)       1.53   1.53    1.53    SAFE_NEUTRAL
(1.1,1.80)       2.08   1.70    inf     YAW_RIGHT
```

结论：所谓“左开”在 1.8 m 高度是**右侧**（body −Y）开；1.63 m 时三向皆堵 → SAFE_NEUTRAL。1 m 地标顶 z=1.5；中间带最低行向下约 4.95°，在 x=2.5 m 前面要 z≳1.70 m 才让地标落到带外（考虑不同距离略有变化）。

### 4.6 项目已有 live episode 证据（我未重跑，只读 `/tmp`）

```
/tmp/ep_test  相邻 pose dt: 2.80, 2.82, 2.76, 2.74, 2.90 s
              x: 0.06 -> 0.82 -> 1.53 -> 1.48 -> 1.31 -> 1.31
              actions: [1,1,2,3,2,0]（旧无滞回版本；每 FORWARD 步约 +0.76 m）
/tmp/ep_crash 相邻 pose dt: 2.84, 2.82, 2.82, 2.82, 2.32, 2.28, 2.38 s
              x: 0.07 -> 0.84 -> 1.62 -> 2.40 -> 2.5628（冻结）
              step 4 起 collision=true（contact+stuck）
```

这组数据直接证明：每步实际约 2.3–2.9 s sim-time，FORWARD 每步约 0.76 m；不是 `--hold 0.5`。也证明正面接触传感器是工作的（x=2.5628≈柱面 2.8−0.235）。

## 5. 已确认无问题（抽样，不是没查）

- `save_frame.py` 紧凑 stride 的 PNG 编码：IHDR/IDAT/IEND、CRC、filter 0、color type 2/6 均正确；真实 320×240 RGB 帧可读回。
- `teacher.expert_action`：`min_clear=2.0` 边界（1.999→YAW_LEFT，2.0→FORWARD_SLOW）、左右平局（→YAW_LEFT，确定）、滞回三条用例（小幅保持/明显换向/不覆盖安全）均符合预期；正常 h=240 的三带逻辑正确。
- `monitor.py`：单点/同 x/同 y/空数据不除零；`winfo_width` 未 map 时 early return + `<Configure>` 去抖重画；空/坏 epoch 不崩且排序稳定；`_tick` 异常后仍重排；tail 追加/半行/变小截断/换 inode 替换均正确；测试 50/50（有显示）/37+1 skip（无显示）。
- `collect.read_pose`：真实 `gz.msgs.Pose_V` JSON 形如 `{"pose":[{"header":{"stamp":...},"position":...}]}`，`arr[0].get("position")` 命中；live oracle 的 pose_ts 非零且单调（18.0/20.8/23.6…），Pose_V 兜底有效。边界：`pose` 为空列表时回退到 `msg.get("position")`（普通 Pose），两者都没有则返回 `(None, 0)`；当前 publisher 不产生空列表，`altitude_vz`/`_dist3` 有 None 判空，但 oracle 会写 `pose_gt:null`。`gz topic --json-output` 实测是**一行一个 JSON**，`splitlines()[0]` 对单消息成立。
- 世界 SDF：`gz sdf -p` 通过；`<contact><collision>…<topic>` 的嵌套符合 SDFormat 14 的 contact 传感器 schema（系统自带 `contact_sensor.sdf` 同写法）；`<static>true</static>` **不会**关闭碰撞检测（live 接触在 x=2.5628 触发）；两根柱 z∈[0,3] 确实覆盖 1.4–1.8 m 飞行高度；RGB/深度相机同 pose(0.2,0,0)/同 60° FOV/同 320×240；相机在机体盒（前面 0.235）内，但 near=0.05 的裁剪面在 0.25，正好不拍到自己（余量仅 0.015 m）；电机/控制器参数与系统自带 `quadcopter.sdf`/`multicopter_velocity_control.sdf` 逐项一致（motorConstant 8.54858e-06、maxRotVelocity 800、速度/姿态增益、rotor direction），T/W≈1.47、悬停约 660 rad/s < 800，未发现自洽性问题。
- `ui/launch_ui.sh`：`bash -n` 通过，能按 python3→/usr/bin/python3→/opt/conda 找带 tkinter 的解释器；`.desktop` 的 `desktop-file-validate` 通过（仅 Categories hint）。

## 6. 结论

- **发现总数 20 条**：必须修 5、应该修 10、可以改 5。另列 6 组“已确认无问题”的抽样。
- **最严重三条**：
  1. **M1（RGB 在动作后抓取，标签/图像因果倒置）**——直接决定这批数据能不能用于模仿学习；真实 episode 已显示时间差约 2.8 s。
  2. **M2（动作不停止，`--hold` 名义 0.5 s、实际约 2.8 s）**——每 FORWARD 0.76 m、每 YAW 约 90°，且随 RTF 漂移；与 M1 叠加使标签彻底失真。
  3. **M5（teacher 全无效深度 → FORWARD_SLOW）**——深度失效时专家仍命令前进，是安全方向的 fail-open；必须改成中心带无有效数据 → SAFE_NEUTRAL。
  - 紧随其后：M3（异常/失败不清理，无人机继续飞且丢行）、M4（`gz topic` 无超时，会永久挂起）。
- **数据可用性判断**：当前 collector 产出的数据**不能当正式训练集**；`/tmp/ep_test`、`/tmp/ep_crash` 只适合链路验证。修复 M1/M2/M3/M4/M5 并重跑一个短 episode 后，才谈有效样本。
- **监控台判断**：当前 `monitor.py`/`test_monitor.py` 质量明显好于 23:26 版本，之前审核关心的除零/空 epoch/after 吞异常都已修复并有测试；剩余主要是 tail 原地重写这一边缘数据一致性问题，不影响正常 append-only 训练。

## 7. 未检查

- **没有启动 Gazebo、没有运行 collector、没有 live 复测**（遵守并行会话约束）；`/tmp/ep_test`、`/tmp/ep_crash` 是项目先前 live run 的只读证据，不是我新采的。
- 没有验证当前 `collect.py`（23:32 版，带高度保持/滞回）的真实 z 轨迹/RTF/接触时序；S7/S8 的 z=1.63 死锁是**几何+控制参数推导**，live 23:34 worklog 报告稳定在 1.86 m（高于阈值），所以请按“余量脆弱”而非“必现”理解。
- 没有拿到官方 X3 UAV 的 Fuel 模型做逐项 mass/inertia 对比：本机缓存的 `x3 uav.zip` 是不完整下载（`unzip` 报 End-of-central-directory），只对比了系统自带 stock world 的电机/控制器参数；质量/惯量数值未独立核实。
- 没有验证深度图的真实 `step`/`pixelFormatType`/无命中值（没抓到深度 JSON）；S4 的 stride 问题是从代码和 RGB 真实帧推断的潜伏风险，不是当前 320×240 配置的已发生错误。
- 没有测试接触传感器的旋翼碰撞（无 live 旋翼擦碰）；S5/S8 的“旋翼撞了但 base_link 没撞”是按几何（翼尖 0.32 m vs 机体半宽 0.235 m）推导。
- 没有在 XFCE 里双击 `.desktop` 实测；只做了 `desktop-file-validate`/`bash -n`。
- 没有审 `tools/preprocess.py`、`tools/make_synthetic_episode.py`、`src/` C++/CUDA（其他审核 agent 的范围）、`experiments/`、`docs/` 其余内容；没有做监控台 200k 行性能基准（由监控会话在其 worklog 记录）。
- 没有测试 `--refresh` 的 0.5–3600 全边界、长时间运行内存增长、NFS/磁盘满等环境故障。
- 运行必跑测试会重新生成 `ui/__pycache__/*.pyc`（仓库原本已有该目录）；未改任何源码/配置。
