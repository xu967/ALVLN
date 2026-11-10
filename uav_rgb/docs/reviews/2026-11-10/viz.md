# uav_rgb 可视化代码审查报告（只读，2026-11-10，16:05 定稿）

## 结论摘要

1. `record_views.py` 三处修复都正确：零速帧 27–31% → 0，图标朝向误差中位 96–123° → 1.4–2.7°，图标离真机体中位 5.2–6.5 px → 0.9–1.3 px。残留 4 项，其中 1 项是修复带来的回归（起飞段约 7 s 面板显示 pos 与 speed）。
2. corridor 判据没有进视频：越界集终帧红字 `NOT REACHED`，面板同时写 x=6.08（goal 6.0），TOP 无走廊线，65% 的动作帧机体不在 TOP 画内；Studio 归为「未到达」，monitor 画成高过 goal 线的灰柱。
3. 「视频展示的」与「学生看到的」：像素来源一致（录像帧与决策帧 44/44 逐像素相同，训练与推理同为 PIL BILINEAR 缩到 160x120）；标注不一致（high 档把 320x240 写成 model input），3DGS low / high 档的 ONBOARD 用了与落盘决策帧不同的渲染参数。
4. 对外素材两处：短片 v2 的「学生成功」片段 seed 190 在 x=5.21 时 y 已到 3.78 m，是绕出柱阵后过线；并发会话 15:09 的保护罩改动把实心占比 6.9% 的边框件画成 8 个不透明实心圆盘，盖住机臂、电机和桨。
5. 可视化补充项排第一的是教师动作对照条：`action_expert` 已在每集 `actions.jsonl`，seed1-113 第 5–44 步教师一直给 YAW_LEFT、学生一直 FORWARD，不用重跑仿真。

## 0. 范围与证据位置

- 代码：`/root/claude-code-backup-20261028/code/uav_rgb`，main @ c85a1ff 加工作树未提交改动。审查期间工作树在变：`record_views.py` 14:42（578 行，本报告行号以它为准）、`e1_body.py` 15:09、`uav_rgb_e1.sdf` 15:27；16:03 复查这三个文件没有再变。
- 证据都在草稿目录 `S=/root/rivermind-data/uav_rgb/tmp/cursor-1110/agents/viz/`：截图 `frames/`、实测日志 `logs/`、脚本 `scripts/`、离线重合成 `resynth/viz3_worktree.mp4`。下文路径省略 `S/`。
- 真值做法：对 TOP 相机原始帧做背景中值差分，检出真机体质心，得到位置与 10 Hz 速度，再与图标、面板数值比（`scripts/top_truth.py`）。
- 数据：Gazebo 7 集（demo 教师 / 学生 seed 113；exp-viz3 教师 113；`cursor-1110/gz` 的 teacher-153、seed1-113、bc-153、trunc-153，四集都已生成并带原始帧）、森林 4103（high 档旧视频）、3DGS garden 六路线与碰撞集 route_4/ep_00001、短片 v2。
- 测试：7 套纯 Python 测试全部通过（`logs/test_*.log`：record_views、view_layout、e1_body、3dgs test_video 4 项、monitor 124 项、studio 52 项、trainer_demo 92 项）。下面的问题都没有被现有测试拦住。
- 标注：「已核实」= 有实测数字或截图；「代码结论」= 只读代码得出，没有跑出现象。

## A. 三处修复的核实

| 修复 | 结论 | 修复前（HEAD） | 修复后（工作树） |
| --- | --- | --- | --- |
| TOP 图标 yaw 度→弧度（`_yaws` 用 `math.radians`） | 正确 | 机头与轨迹切线夹角中位 96–123°，大于 90° 的帧占 51–72% | 中位 1.4–2.7°，p90 3.7–6.5°，最大 5.2–10.4° |
| 面板 speed 0.00（`_speed_at` / `_oracle_speed`） | 正确，有残留（A1、A2） | 显示 0.00 的帧占全片 27.0–31.4%，FORWARD 执行中占 34.2–37.3% | 0 帧 |
| 图标与 pos 行滞后一步（`_pos_at` 插值） | 正确，有残留（A4） | 落后中位 0.39–0.40 s、最大 0.90 s（demo 学生集 2.08 s）；离真机体中位 5.2–6.5 px、p90 14.5–16.3、最大 19.7–23.2 px | 中位 0.9–1.3 px、p90 2.5–3.0、最大 4.0–6.3 px（0.063–0.076 m），0 帧超过 10 px |

数据来源：`logs/a_recompute.txt`（5 集，HEAD 版逻辑导出在 `old/record_views_head.py`）、`logs/top_truth.txt`（4 集带原始帧）。截图：`frames/truth_viz3_oldworst.png`（同一帧修复前图标离真机体 23.2 px、修复后 1.0 px）、`frames/resynth_before_after.png`（用工作树代码对 exp-viz3 原始帧重合成的前后对比，355 帧）。并发会话日志写零速帧降到 1/272 与 1/428，我按面板 `%.2f` 的显示值统计是 0 帧，差 1 帧，不影响结论。

**A1（中，回归，已核实）起飞段面板显示 pos 与 speed**

```260:265:/root/claude-code-backup-20261028/code/uav_rgb/sim/tools/record_views.py
    if not track:
        return None
    if t <= track[0][0]:
        return track[0][1:]
    if t >= track[-1][0]:
        return track[-1][1:]
```

- 现象：首个动作之前的 70–73 帧（片头约 7 s，占全片 14–21%）面板写 `pos ... z=2.05`、`speed 0.11–0.14 m/s`，同一帧 CHASE 里机体在地面（`frames/n113_0000.png`）。0.14 是首段平均速度，来自 `_speed_at` 第 283 行的 `or (t < track[0][0] and a is track[0])`。修复前这些帧不显示这两行。同一时段 TOP 不画图标（`past` 为空），面板与 TOP 互相矛盾。
- 最小修法（3 行）：第 262–263 行改成 `if t < track[0][0]: return None`；第 283 行删掉 `or (...)` 那半句。`_draw_top` 里对 `_pos_at` 的解包在 `if pts:` 之内，不受影响。
- 验证：`check_panel_speed` 增加一条，t 取首点前 1 s，断言 `_pos_at` 与 `_speed_at` 都返回 None（现在返回首点与首段速度，红）。

**A2（中，已核实）speed 是上一步窗口末 oracle 值的阶梯保持，YAW 原地转时显示成前飞速度**

- 位置：`record_views.py:498-508`（`_oracle_speed`）、`:527-529`。
- 现象：对 TOP 帧真值，|显示 − 真实| 中位 0.02–0.04、p90 0.13–0.14、最大 0.27–0.35 m/s。YAW 步里真实速度 0.01–0.02 m/s，面板显示 0.27–0.37；YAW 之后第一个 FORWARD 步真实 0.34–0.36，面板显示 0.06–0.11。真实低于 0.08 m/s 的帧里，显示高于 0.20 的有 13/21、3/5、3/19、14/22（四集）。
- 三种算法对同一真值的误差（`logs/speed_candidates.txt`）：

| 算法 | 平均误差 m/s（四集） | 误差超过 0.15 m/s 的帧数 |
| --- | --- | --- |
| 现状：oracle `speed_mps` 阶梯保持 | 0.045–0.059 | 12–24 |
| oracle 值线性插值 | 0.020–0.044 | 1–11 |
| 只用 `_speed_at` 轨迹段差分 | 0.022–0.037 | 4–7 |

- 最小修法（净减约 14 行）：删 `_oracle_speed` 与第 527–529 行的分支，只留 `spd = _speed_at(track, t)`。
- 验证：测试里造「FORWARD（oracle speed 0.30）后接 YAW（0.8 s 位移 0.04 m）」，在 YAW 步中点断言显示值不超过 0.10（现在 0.30，红）。

**A3（低，已核实）yaw 取最近值，不插值**

- 位置：`record_views.py:302-310`。YAW 步内图标朝向误差中位 13–14°、最大 24–30°，每集 18–54 帧；图标在步界一次跳 27°（`frames/n113_yaw_strip.png`：#88–#98 保持 0°，#100 跳到 -23°，#110 跳到 -51°）。
- 回绕：现实现不做角度运算，没有回绕问题。这几集 `yaw_before` 在 -80.3..30.2°，相邻步最大跳变 30.1°，没有跨 ±180°。
- 最小修法（约 6 行）：在相邻两个 yaw 之间按最短角插值，`d = (y1 - y0 + pi) % (2*pi) - pi; yaw = y0 + f*d`。不解回绕直接线性插值的话，170°→-170° 的中点会算成 0°（机头反向）。
- 验证：0°→30° 的中点应为 15°（现在 0°，红）；170°→-170° 的中点应为 ±180°。

**A4（低，已核实）插值只用在图标与 pos 行，投影高度、轨迹末端、出画判定仍取 `past[-1]`**

- 位置：`record_views.py:414-417`、`:433-436`、`:446-454`。轨迹线末端落后图标最多一步（约 0.3 m，23 px）；片头 `past` 为空时 goal 线按地面投影在第 470 列，首个动作后跳到第 529 列（59 px）。
- 最小修法（改 4 行）：`_draw_top` 开头算一次 `cur = _pos_at(track, t)`，z、轨迹最后一个顶点、出画判定都用它。
- 验证：两点相隔 1 s、t 取中点，断言折线最后一个顶点等于图标中心（现在是旧点，红）。

**A5（已核实，无残留）** 复杂度：每帧全部线性扫描合计 40–60 µs，每集 13.6–29.9 ms，不需要优化。`track` 为空、只有一个点、相邻点同一时间戳：`_draw_top` / `_panel` 不抛异常，返回 None 或端点（实测）。

**A6（低）文字陈旧**：`record_views.py:412-413` 注释写「离 t 最近的点」，实现是插值；测试名「TOP 图标取离 t 最近的点」「面板速度：oracle 优先」同理。改 3 处文字。

**A 附（中，已核实）保护罩修复（已知问题 4）轮廓对了，实体画错了**

- 轮廓：`e1_body.py:65` 的 72 点 `GUARD_OUTLINE` 与官方 STL 外沿相差中位 1.2 mm、最大 3.9 mm（STL 外沿 0° 方向 58.1 mm、45° 方向 101.4 mm），TOP 图标按它画（`frames/resynth_before_after.png` 右列）。
- 实体：我栅格化 STL 俯视投影（0.25 mm/px），实体面积 1646 mm²，只占外轮廓包住面积 23903 mm² 的 6.9%，是细边框加每个电机两根辐条和一个电机座小环（`frames/guard_stl_topdown.png`）。工作树的 `PARTS`（`e1_body.py:46-53`）与 SDF（`uav_rgb_e1.sdf:278-317`）是 8 个 alpha=1 的实心圆柱，并集 22421 mm²，是真件的 13.6 倍，装在桨上方 z=26 mm。FOLLOW、CHASE、俯视三个角度下机臂、电机、桨都被白色圆盘盖住（`frames/guard_solid_vs_head.png`，左 HEAD、右工作树）。
- 最小修法（16 处数值）：8 个 guard 部件的 alpha 从 1.0 改到约 0.2（`PARTS` 最后一列），SDF 对应 8 个 visual 加 `<transparency>0.8</transparency>`。模板 sha 会再变一次，要赶在用 2e554bcd 模板采数据之前改。
- 验证：`test_e1_body.py` 俯视渲染后取电机座 (42, 42) mm 处像素，断言不是保护罩的白色（现在是白色，红）。

## B. 新 bug 清单

### B0. 已核实没有问题的项

| 项 | 实测结果 | 证据 |
| --- | --- | --- |
| 三路相机同步 | uav / top 10 Hz，chase 15 Hz（录制时每 100 ms 桶留一帧）；同一合成帧三格时间差中位 0–34 ms、最大 34 ms；缺帧 0，重复帧 1（片尾多出的一格，0.3%）；画面比面板 `sim t` 新 0–60 ms | `logs/sync_viz3.txt`、`logs/sync_teacher153.txt` |
| 视频时间映射 | 10 fps，1 帧 = 0.1 s 仿真时间；seed1-113 是 470 格加 15 帧定格 = 485 帧，与文件一致；3DGS 碰撞集 210 样本加 15 = 225 帧 | 同上、`frames/gs4c_0224.png` |
| 顶视投影公式与镜像 | 图右 = 世界 +X，图上 = +Y，没有镜像；左转在 TOP 里是逆时针、右转向下；按现公式图标离真机体约 1 px，上下镜像的话会差 120–335 px | `logs/top_truth.txt` |
| ONBOARD 像素来源 | 录像 uav 帧与落盘决策帧 44/44 逐像素相同；决策帧时间戳 30/30 与录像帧相同；全程 PNG、通道序一致 | `frames/onboard_vs_student_step4.png` |
| 结果字与 `eval.json` | 看过终帧的 5 集一致：trunc-153 `SUCCESS`、seed1-113 与 bc-153 `NOT REACHED`（`success=false`）、森林 4103 `COLLISION`、demo 学生 113 `SUCCESS`（旧口径 `success=true`）。超时没有单独的词，`timed_out` 的集也会写 `NOT REACHED`（代码结论，无数据）；越界见 G1 | `frames/ntr_0347.png`、`frames/n113_0484.png`、`frames/nbc_0511.png`、`frames/forest_0341.png`、`frames/demoS_0514.png` |
| 文字越界 | 结果字最长 498 px、面板第 5 行最长 482 px，面板可用 624 px | 本次 `cv2.getTextSize` 实测 |
| 3DGS TOP MAP 方向与单位 | +x 右、+y 上；1 m 比例尺与 1 m 到达圈吻合；图标朝向与 yaw 一致；面板 yaw / roll / pitch 是度，由四元数算 | `frames/gs4c_zoom.png` |

### B1. Gazebo 域

**G1（高，已核实）越界集没有专门显示，结果字与面板读数矛盾**

```942:944:/root/claude-code-backup-20261028/code/uav_rgb/sim/tools/eval_closed_loop.py
    result = ("SUCCESS" if summary.get("success") else
              "COLLISION" if summary.get("collision") else
              "STUCK" if summary.get("stuck") else "NOT REACHED")
```

- 位置：上面 3 行；`eval_closed_loop.py:953-958`（调 `compose` 不传走廊参数）；`record_views.py:321-323`、`:397-398`、`:511-512`（三个函数都没有走廊入参）。
- 现象：seed1-113（`corridor_violation=true`，`y_max_m=8.45`，`max_x=6.08`）终帧红字 `NOT REACHED`，同一面板写 `pos x= 6.08 y=-8.45`，goal 是 6.0（`frames/n113_0484.png`）；bc-153 相同（`frames/nbc_0511.png`，终点 x=6.04，y=-7.20，`y_max_m=7.33`）。面板没有越界 flag，TOP 没有走廊线，只写 `UAV OUT OF TOP VIEW`。同一种飞法在旧口径的 demo 视频里写 `SUCCESS`（`frames/demoS_0514.png`，x=6.05，y=-8.35），两段视频除了这个词以外画面相同，看不出判据变了什么。seed1-113 在 video +21.0 s 出画，+23.8 s 才越过 |y|=3 m，越界那一刻在 TOP 里看不到；两集各有 65% 的动作帧（260/397、276/424）机体不在 TOP 画内。
- 限制：TOP 相机在机体高度只看得到 |y| ≤ 2.36 m，y=±3 m 投在第 -49 行与第 409 行（图高 360），走廊线画了也在画外。
- 最小修法（约 15 行）：① `_write_video` 的结果词在 `NOT REACHED` 之前加一档 `OUT OF CORRIDOR`，并把 `args.corridor_half_width` 传给 `compose`（2 行）；② `_panel` 加一条 flag，从第一个 |y| 超过半宽的轨迹点起显示 `OUT OF CORRIDOR |y|>3.0 @ step N`（约 8 行）；③ `_draw_top` 里走廊线落在画内才画，图例加 `corridor`（约 5 行）。TOP 相机不动（改位姿会换模板 sha），走廊全貌交给 C3 的正交俯视图。
- 验证：`test_record_views.py` 造一条第 3 步 y 到 3.5 m 的轨迹，断言该步之后 `_panel` 的 flags 含 `OUT OF CORRIDOR`、之前不含；`test_eval_sim09v2.py` 断言 `{success: False, corridor_violation: True}` 的结果词是 `OUT OF CORRIDOR`。两条现在都红。

**G2（中，已核实）面板 COLLISION 比 TOP 的 X 标记早一个动作窗口**

- 位置：`record_views.py:533`、`:535`（flag 用 `action_start_ns`）；`:386-394`、`:455`、`:459`（X 用该步 oracle 的 `timestamp_ns`，即窗口末）。
- 现象：森林 4103 第 27 步，flag 从 37.105 s 起显示，X 从 38.11 s 起显示，相差 1.0 s（10 帧）。`frames/forest_0318.png` 有 flag 无 X，`frames/forest_0341.png` 两者都有。3DGS 同类：flag 从碰撞步第一个样本起（`sim/3dgs/video.py:283`），最多早 0.5 s。
- 最小修法（3 行）：`compose` 把 `hit_mark` / `stuck_mark` 的时间传给 `_panel`，flag 与 X 用同一个时刻。
- 验证：`check_contact_stuck` 取 `action_start < t < oracle ts` 的时刻，断言「flag 出现」与「X 出现」同真同假（现在一真一假，红）。

**G3（中，已核实）TOP 是透视画面，叠加层与柱子不在同一高度，贴柱飞行时读不准间隙**

- 位置：`record_views.py:409-417`（goal 线、waypoint 圈、轨迹、图标都投到机体高度）；相机 `sim/worlds/uav_rgb_e1.sdf:88`（位姿 3.25, 0, 7.0，HFOV 80°，640x360）。
- 现象：叠加层彼此一致（都在机体高度，76.3 px/m）。画面里的柱子是相机渲染出的实体，顶面在 z=3.0 m（95 px/m），柱脚在 z=0（54.5 px/m）。seed 113 的 8 根柱，顶面与机体高度截面的偏移是最小 5、中位 34、最大 47 px，折合 0.07 / 0.45 / 0.61 m（`frames/top_parallax_113.png`：绿框顶面、黄框机体高度截面、红框柱脚）。人眼看到的是顶面，图标要比的是黄框；`frames/ntr_0180.png` 里图标压在柱子侧面上，这一集没有碰撞。Gazebo TOP 没有比例尺（3DGS 的 TOP MAP 有）。
- 最小修法（约 12 行）：`_draw_top` 读 world 同名的 `scene_<seed>.json`（`pillars` 的 x / y / size 已有），把每根柱在机体高度的截面画成细线框，图例加 `obstacle @ UAV z`。
- 验证：测试里给一根柱和固定相机，断言 `polylines` 收到的四个角等于 `project_top(cam, 角点, z)`（现在没有这次调用，红）。

**G4（中，已核实）编码失败后原始帧被删，且没有回退路径**

- 位置：`sim/tools/view_layout.py:170`、`:177-186`（只有找不到 ffmpeg 才回退 cv2 mp4v）、`:193-195`、`:199-206`；`eval_closed_loop.py:962-965`。
- 现象：任务书写「libx264 失败回退 cv2 mp4v」，代码不是这样。用两个假 ffmpeg 实测（`tmp/fakeff/`）：启动即退出的，写入 0 帧后 `write()` 抛 `BrokenPipeError`；读入 3 帧后退出的，写入 3 帧后同样抛出。`release()` 抛带 ffmpeg stderr 的 `RuntimeError`，没有输出文件，没有回退。`compose` 里紧跟 Popen 的 `isOpened()` 检查拦不住：读入 3 帧才退出的那个假 ffmpeg，检查时返回 True。`_write_video` 捕获异常记 `video_error`，随后第 964–965 行在没有 `--keep-work` 时删掉 `.frames_<seed>`，这一集的视频只能重跑 Gazebo。本机 PATH 上没有 ffmpeg，用的是 imageio 自带的静态 ffmpeg，当前不会触发；触发条件是 PATH 上出现不带 libx264 的 ffmpeg（`find_ffmpeg` 里 PATH 排在 imageio 前面）或写盘失败。
- 最小修法（1 行）：`eval_closed_loop.py:964` 改成 `if not args.keep_work and "video_error" not in summary:`。帧留着就能用 `record_views.py compose` 离线重合成，不需要再写一条 cv2 回退。
- 验证：测试里把 `record_views.compose` 换成抛异常的桩，断言 `frames_dir` 仍存在且 `summary["video_error"]` 有值（现在目录被删，红）。

**G5（低，已核实）contact 与 stuck 同时出现时第二条 flag 与结果大字重叠**

- 位置：`view_layout.py:124`（flag 每条下移 34 px）、`:133`（结果字固定在 y+290）。medium 档第二条 flag 占 y 263–281，结果字占 y 267–290（合成图 `frames/panel_2flags_result.png`）。扫了数据盘 6000 集 `actions.jsonl`，stuck 为 0 集，现有视频没有出现。
- 最小修法（3 行）：flags 画在同一行，x 按前一条文字宽度右移。验证：`test_view_layout.py` 的假 cv2 记录两条 flag 的 y 相同且都小于结果字顶边（现在第二条更低，红）。

**G6（低，已核实）图例**

- `record_views.py:462`、`:481` 的 stuck 标记色 (0,165,255) 与 `view_layout.py:24` 的 YAW_RIGHT 动作色 (40,160,230) 同为橙色。修法：stuck 换一个不在 `ACTION_BGR` 里的颜色（1 行）。
- high 档 TOP 的叠加层画在 640x360 相机帧上，再被 `paste` 立方放大 2 倍（`record_views.py:367-372`），字与线发虚；3DGS 的 TOP MAP 按倍率原生画（`video.py:131`）。medium 档图例字号 0.4，约 8 px 高。动作时间条没有图例，是 11-10 VIZ3 待办 c，未完成，不算新发现。

### B2. 两域公共

**X1（高，已核实）high 档 ONBOARD 标签把 320x240 写成 model input**

```65:67:/root/claude-code-backup-20261028/code/uav_rgb/sim/tools/view_layout.py
    if display > 1:
        return f"ONBOARD {w}x{h}{fov} (model input; display {display}x)"
    return f"ONBOARD {w}x{h}{fov} camera (model input is resized)"
```

- 现象：学生输入是 160x120（`sim/tools/student.py:117` 与 `tools/preprocess.py:638` 同为 PIL BILINEAR，从 320x240 缩小；demo 用的 ckpt `two_frame=false`）。medium 档的 `... camera (model input is resized)` 属实；high 档写 `ONBOARD 320x240 HFOV 60deg (model input; display 2x)`（`frames/forest_0318.png`），教师集也这样写。这与同一函数 docstring 第 56–57 行「不冒充」相反，`test_view_layout.py:91`、`:93` 把这句话锁成了期望值。短片 v2 的森林片段是 high 档，这行字进了对外素材（`frames/reel_forest_onboard_label.png`，裁自 `stills_v2/04_gz_forest_seed4103.png`）。Studio 同类：`ui/studio.py:1371` 的格标题「第一人称视角 · RGB INPUT」、`:2416` 的注释「机载画面是模型输入」。
- 像素没有问题（见 B0）。视频展示的是 320x240 相机帧立方放大（medium 480x360，high 960x720），每步约 9 帧；学生只看其中 1 帧的 160x120 版本，画面上没有标出是哪一帧（见 C4）。
- 最小修法（2 行）：`display > 1` 时写 `ONBOARD {w}x{h}{fov} camera (model input is resized; display {display}x)`。
- 验证：先把 `test_view_layout.py:91`、`:93` 的期望改成不含 `(model input;`，此时红；改 `onboard_label` 后绿。

### B3. 3DGS 域

**S1（中，已核实）TOP MAP 没有碰撞标记**

- 位置：`sim/3dgs/video.py:179-187`（`_TopMap.draw` 只画轨迹和图标）、`:283`（只有面板 flag）。
- 现象：garden route_4/ep_00001（42 步，`truncated_on_collision=true`）终帧面板写 `COLLISION @ step 42`，TOP MAP 上没有标记（`frames/gs4c_0224.png`）。`actions.jsonl` 末步已有 `collision_point=[2.216, -1.406, 2.000]`。
- 最小修法（约 5 行）：`draw` 加 `hit_xy` 入参，`step >= hit["step"]` 时在 `collision_point` 画红色 X，图例加 `collision`。
- 验证：`video.py` 自检的 mock 集加一个碰撞步，断言终帧在 `topmap.px(collision_point)` 处是红色（现在是底色，红）。

**S2（中，已核实）只画 reach 圈，不画 goal-stop 圈，也不画规划路线**

- 位置：`video.py:157-158`；`tools/demo_3dgs_routes.py:240`（`stop_radius_m: 0.5`）；每集 manifest 的 `stop_radius: 0.5`。
- 现象：图上只有 `waypoint r=1m` 一个圈，教师集的 `SUCCESS reached` 由 0.5 m 的 goal-stop 触发（`video.py:425-426`），机体进了画出来的圈还要再飞约 0.5 m 才结束，画面没有解释。TOP MAP 只有已飞轨迹，没有 A* 规划路线。
- 最小修法（约 4 行）：`main` 把 `man.get("stop_radius")` 传进 `_TopMap`，非空时再画一个细圈，图例写 `stop r=0.5m`。验证：自检断言 `stop_radius=0.5` 时底图上有两个半径不同的圈（现在一个，红）。

**S3（中，代码结论加 medium 档实测）low / high 档 ONBOARD 的渲染参数与落盘决策帧不同**

- 位置：`video.py:256-257`（逐样本按 `renderer.width*s × height*s, ssaa=ssaa` 重渲）、`view_layout.py:29-33`（low 的 ssaa=1，high 的 scale=2）、`sim/3dgs/scene.py:24`（采集默认 `supersample=2`）。
- 现象：medium 档参数与采集相同，重渲帧在决策时刻与落盘 PNG 的平均绝对差约 4 灰度级，相邻帧之间是 15–43（`frames/gs_onboard_cmp_step21.png`）。low 档超采样是 1（采集是 2），high 档按 640x480 另渲，两档都不是学生拿到的那张图；high 档标签还写 `(model input; display 2x)`。low / high 没有用 GPU 实测。
- 最小修法（改 2 行）：ONBOARD 固定按采集参数渲染（`renderer.width × height`，不传 `ssaa`），放大交给 `paste`，`rendered_at` 入参随之删掉。
- 验证：自检的 MockRenderer 记录每次调用的宽高与 supersample，断言 ONBOARD 那次等于 `renderer.width / height` 与采集默认值（low / high 档现在不等，红）。

**S4（低，已核实）决策帧与面板步号错一个样本**：每步 5 个 trace 样本，决策帧对应上一步的最后一个样本，那一帧面板还写上一步的步号与动作；step 0 的决策帧不在视频里（`video.py:247-286`）。逐帧对照「看到什么、做了什么」会错一格。修法并入 C4（缩略图直接读落盘决策帧）。

**S5（低，代码结论）FOLLOW 的机体没有场景遮挡**：`video.py:114-115` 取 `render_view(...)[0]`，深度被丢掉，`:262-263` 把机体直接画在渲染图上。机体被树或墙挡住时仍画在最前面；跟随相机（后 1.1 m、上 0.3 m）落进几何体内部时画面是近处的高斯。出现频率没有量化。修法是用那张深度图在机体包围盒内做逐像素比较（约 8 行）；不影响学生输入，可以不做。

**S6（低）零碎**

- `video.py:85` 的 `_read_jsonl` 是死代码，被 `:441` 的同名函数覆盖，删 4 行。
- 3DGS 解释器的环境变量名不一致：`tools/demo_3dgs_routes.py:28` 用 `UAV_RGB_3DGS_PYTHON`，`tools/train_student_with_video.py:157` 用 `UAVRGB_3DGS_PYTHON`。
- 已有记录，不算新发现：图例 `spawn` / `occupancy proxy` 压在深色占据格上难读（11-07 版式日志写明不处理）；报告标题写死 Callinstrasse（已知问题 3）。

### B4. 短片合成

**R1（高，已核实）短片 v2 的「学生成功」片段 seed 190 是绕出柱阵后过线**

- 位置：`tools/make_demo_reel.py:73`（字幕「学生 · seed 190 · 成功 · 柱间蛇形通过 · 0 碰撞」，截取源视频 22–37 s）。
- 现象：源视频自己的面板读数（`frames/eval2_s190_posstrip.png`）：step 30 x=4.27 y=2.27，step 36 x=5.21 y=3.78，step 42 x=5.55 y=4.74，step 47 x=6.51 y=5.97，终点 y=7.54（`frames/eval2_s190_0816.png`）。过线前 |y| 已超过 3 m，按 corridor3m 口径是失败。截取窗口停在 step 32 附近，观众看不到它飞出柱阵。同一章的 seed 1（`:72`）在 y≈-0.9 处过线，属实；它之后漂到 y=-7.69 是旧口径过线不停造成的（`frames/eval2_s1_posstrip.png`）。10:10 审查写明没核短片字幕，这条是新发现。
- 最小修法：换素材，配置区改 1 行。现成的走廊内成功集有 `cursor-1110/gz/trunc-153`（`y_max_m=1.22`，但 ckpt 与画质档不同），或按新口径重录。
- 验证：`build` 里对 Gazebo 片段加一个门禁（约 6 行）：字幕含「成功」时读片段旁的 `eval.json`，该 seed 必须 `success=true` 且 `metric_version` 带 `corridor`，否则报错。现有配置会报错（红），换素材后通过。

**R2（中，口径已核实，变化量未知）结尾对照表是旧口径且手抄**

- 位置：`make_demo_reel.py:48-54`（`END_ROWS` 写死 20 / 10 / 13 / 4 / 2）。数字来自 `e1-pipeline-20261107-0417-eval3`，其 `metric_version` 是 `sim09.v1.coast1s`（无走廊、过线不停）。eval3 没有保留 `oracle.jsonl`，离线算不出新口径数字。参照：同口径的 eval2 里短片选的 2 个学生成功片段有 1 个越界；主会话对 eval4 的复算是 11 → 6。
- 最小修法：重评后改 `END_ROWS` 与 `END_NOTES`。让 `END_ROWS` 从各臂 `eval.json` 读（约 8 行）可以等下一次出片再做。

**R3（低）素材陈旧**：柱阵与森林片段都是三处修复之前录的（speed 有 0.00，图标朝向与位置未修），重出片时一并重录。字幕条盖住 ONBOARD 格底部 64·S px 和它的时间角标，是 `make_demo_reel.py:13` 写明的设计，不算问题。

## C. 可视化该补什么

按「帮人看懂策略为什么失败」排序。第 1 项回答「教师此刻想让它做什么」，第 2 项回答「为什么判失败」，第 3 项回答「这个策略成批地往哪里跑」，第 4 项回答「它当时看到了什么」。

| 序 | 补什么 | 数据从哪来 | 改哪里 | 约多少行 | 要不要重跑仿真 |
| --- | --- | --- | --- | --- | --- |
| 1 | 教师动作对照条：时间条下方再加一行教师动作色带，面板写 `teacher: YAW_LEFT` 与分歧步数 | `actions.jsonl` 的 `action_expert` 与 `action_executed`，Gazebo 学生集与 3DGS 集都有。seed1-113：44 步里 41 步不同 | `view_layout.panel` 加可选 `segments2`；`record_views._panel` 与 `video.py:234-241` 各补一组段 | 20 | 不用 |
| 2 | 失败原因标签与越界 flag：`OUT OF CORRIDOR` / `TIMEOUT` / 步数用尽分开写 | `eval.json` per_episode 的 `corridor_violation`、`y_max_m`、`timed_out`；越界时刻由 `oracle.jsonl` 的 y 算 | 同 G1 | 15 | 不用 |
| 3 | N 集轨迹叠加俯视图：正交、无视差，画走廊线与 goal 线，按结果着色，每臂一张 | `<out>/episodes/seed_*/oracle.jsonl`（默认保留）加 `eval.json`；柱子位置用 `scene_gen.build_scene(seed)` 重算 | 新脚本 `tools/plot_eval_tracks.py`（matplotlib 3.11 已在 `.venv-tools`） | 60 | 不用 |
| 4 | 学生输入 160x120 的真实缩略图，带「这一帧被学生看到」的标记 | 合成时 `uav.rgb` 流里 `ts == actions[i].rgb_ts` 的帧；3DGS 直接读落盘 `rgb_%05d.png` | `record_views.compose` 留住最近一张决策帧，用 `student.RESAMPLE`（PIL BILINEAR）缩小后贴进面板右侧空白；`video.py` 同理 | 15 + 6 | 不用，但只对保留了原始帧的集 |
| 5 | 最近障碍距离：面板一行，低于阈值变红 | 3DGS 的 `actions.jsonl` 已有 `clearance_m`（碰撞集末 10 步 0.49 → 0.13 m）；Gazebo 用 `scene_<seed>.json` 的 pillars 加插值位置现算 | `video.py:278-282` 加一项；`record_views._panel` 加一个距离函数 | 1 + 10 | 不用 |
| 6 | 学生四类概率条 | 现有数据没有：`collect.py:920` 把 `student.act` 返回的概率丢了 | 采集端把 4 个概率写进 `actions.jsonl`（2 行，注意审计的字段清单）；`view_layout.panel` 画 4 根条 | 17 | 要重跑评测；或对保留的决策 PNG 离线重推理 |
| 7 | 教师与学生并排对比视频 | 现成的同 seed 视频：demo 教师 / 学生 113；teacher-153 / bc-153 / trunc-153 | 新脚本，或在 `make_demo_reel.py` 加一种片段：两条视频各取 TOP 格与面板，按 `sim t` 对齐后横拼 | 40 | 不用 |
| 8 | 真值深度小图 | 模板有 depth 相机，采集也做了时间配对（`depth_ts`），但评测不落盘深度，`record` 不录深度流 | `record_views.CAMS` 加一路并做伪彩 | 15，每集多约 120 MB 临时盘 | 要 |

- 第 4 项不能用 cv2 缩放：对同一帧，cv2 的 INTER_AREA / INTER_LINEAR 与 PIL BILINEAR 有 8.9% 的像素不同，最大差 33 灰度级。
- 重合成视频需要 `.frames_<seed>` 原始帧，只有带 `--keep-work` 的集留着（现在是 exp-viz3 一集和 `cursor-1110/gz` 四集）。其余评测目录只有 `episodes/`，能做第 3 项的静态图，做不了视频。
- Studio 的二维地图（`ui/studio.py:2580-2626`）只画障碍物和 waypoint，goal 线与走廊线可以跟第 2 项一起补（约 6 行）。

## D. Studio / monitor 的确定性 bug

**D1（中，已核实）两个界面对越界集没有显示，Studio 自己发起的评测也不带走廊参数**

- 位置：`ui/studio.py:683-699`（`classify_episode`）、`:3372-3402`（逐集表）、`:2856-2858`（手动评测命令）；`ui/monitor.py:299-322`（摘要）、`:325-357`（柱状图）。
- 现象：对 seed1-113 的 `eval.json` 实测，`classify_episode` 返回「未到达」，表格同一行 max_x 是 6.08（goal 6.0），没有 y_max 列。monitor 摘要只有 `SR: 0.0% · 碰撞: 0.0% · false_forward: 0.0%`，柱状图里这一集是灰柱，柱顶高过 `goal 6` 虚线。`ui/` 下搜不到 `metric_version` 与 `corridor`，两个界面都看不出一份 SR 是哪个口径。Studio 的评测命令不传 `--corridor-half-width`（仓库里除评测脚本和它的测试外没有调用方传它），从 Studio 跑出来的 SR 仍是旧口径。
- 最小修法（约 5 行）：逐集表 max_x 单元格在 `corridor_violation` 为真时追加 `|y| 8.45 越界`（2 行）；评测摘要显示 `metric_version` 与 `corridor_violations`（Studio、monitor 各 1–2 行）。Studio 是否默认带 `--corridor-half-width 3.0` 属于口径决定，等用户定。
- 验证：`test_studio.py` 给 `_fill_episode_table` 一集越界数据，断言该行文本含「越界」；`test_monitor.py` 断言 `fmt_eval` 的输出含 `corridor`。现在都红。

**D2（中，已核实）Studio 训练曲线遇到缺测点会丢掉之前的整段**

```902:907:/root/claude-code-backup-20261028/code/uav_rgb/ui/studio.py
    for s in series:
        seg = []
        for x, y in s.get("points", []):
            if x is None or y is None:
                seg = []
                continue
```

- 现象：遇到 None 直接清空 `seg`，缺口之前的点不画。假 canvas 实测：`results.csv` 有 4 个 epoch、末尾 loss 是 nan（训练发散）时，Studio 的 loss 图两条曲线都不画；同一份数据 `monitor._line_chart` 画出 3 点和 2 点两条线（`ui/monitor.py:251-259` 先把已有段画出再清空）。训练发散时这张图是空的。现有 5 份 `results.csv` 没有缺测行，问题目前没有表现出来。
- 最小修法（净减约 5 行）：None 分支和循环结束处都调用 `monitor_io._draw_segment(canvas, seg, s["color"])`（Studio 已有 `import monitor as monitor_io`），删掉第 909–914 行自己那份画法。
- 验证：`test_studio.py` 用假 canvas 喂 `[(1, .9), (2, .8), (3, None)]`，断言画出 1 条 2 点折线（现在 0 条，红）。

其余 Studio 候选没有拿到证据，不列。11-07 报告里已记的「默认 world 是 X3」是产品决定，不重复报。

## 没读 / 没验证 / 没做

没读：

- `ui/studio.py` 第 344–520、920–1289、1400–1769、2929–3290 行；`ui/monitor.py` 第 360 行之后（`MonitorApp` 类）。D 部分因此只覆盖取数与绘图的纯函数，界面事件逻辑没有审。
- `sim/3dgs/eval_3dgs.py` 的录像段、`scene.py` / `collision.py` / `dynamics.py` 只看了与本报告相关的行；`tools/check_input_parity.py` 没读。
- 历史报告里 `2026-11-06-2103-viz-fixes.md`、`2026-11-07-0550-viz-review.md`、`VIZ_LAYOUT_20261107.md` 只做了关键词检索（model input、视差、走廊、概率、遮挡等），用来排除重复；`VIZ_20261107.md`、`2026-11-07-1255-viz-layout.md`、`agent12-viz3/PROGRESS.md` 与 `analysis/gz_timing_before.txt` 通读了。

没验证：

- 没启动 Gazebo，没用 GPU。S3 的 low / high 档差异、S5 的遮挡是代码结论。保护罩在 Gazebo 里的实际渲染没有看，只用 `e1_body.draw_shaded` 软件渲染对比（SDF 与 `PARTS` 逐项一致由 `test_e1_body.py` 保证）。
- `timed_out` 与 stuck 的显示路径没有真实数据（6000 集里 stuck 为 0），G5 用的是合成图。
- 短片 v2 只看了 5 张代表帧和 3 条 Gazebo 源视频，没有逐段核 22 段字幕；3DGS 的 16 段没有核步数与「到达」。
- `$D/tmp/main-session/s1110/frames/*.png` 与 `resynth/teacher_fixed.mp4` 没有看，改用我自己的重合成（`resynth/viz3_worktree.mp4`，355 帧）核对修复效果。草稿目录里 `demoS_*`、`gs3_*`、`nT_*` 的部分抽帧没有逐张看。
- 3DGS 占据栅格与渲染画面只核了方向和比例尺，没有做逐像素对齐（顶视渲染需要 GPU）。
- A1 之外没有逐项排查三处修复是否带来别的回归；R2 的新口径数字没有算出来。

没做：

- 没改仓库任何文件，没跑会改状态的 git 命令，没动 `$D` 下已有实验目录；全部输出在草稿目录。
- 修法都没有实施，行数是估计，「先红后绿」的测试没有写。原因：任务是只读审查。
- 没跑 `tools/check.sh`（另一个进程在跑）。
