# 3DGS 路线接 E1 MINI 固件在环（cf_sitl_proc）+ Callinstrasse 教师闭环三视角 demo

- 时间：2026-11-07 01:10
- 范围：`sim/3dgs/`（dynamics / scene / renderer / collect_3dgs / eval_3dgs / video / 新测试），demo 产物在数据盘
- 结论：完成。`--dynamics cf_sitl` 可用、默认 `cf_blackbox` 行为不变；Callinstrasse 教师闭环 3 集跑通并录了三视角视频；全部 `sim/3dgs/test_*.py` 在 3dgs-venv 下绿（含 `RUN_3DGS_REAL=1` 真 gsplat）。40 步上限内 3 集都**未到达**（差 0.16–0.35 m），补跑 48 步 3/3 到达。

## 做了什么

- `sim/3dgs/dynamics.py`：新增 `CfSitlParams`（e1_mini.json 只读视图：sha256、calibrated、camera_mount、executor 速度；`profile()="cf_sitl_v1:<sha8>"`，前缀与 `sim/tools/cf_sitl_exec.py` 一致）和 `CfSitlDynamics`：
  - 子进程 `cf_sitl_proc --params <e1_mini.json> --seed <seed>`（路径 `$CF_SITL_PROC`，缺省 `$DATA_ROOT/uav_rgb/tmp/build-cf-sitl/cf_sitl_proc`）；stderr 单独线程收尾 20 行；`_call` 用 `select` 60 s 超时，子进程退出 / 超时 / `ok=false` 都抛 `RuntimeError`，不挂死。
  - `reset(x, y, yaw_deg, episode)`：在 **z=0 地面** reset，`takeoff` 到 `spawn.z - ground_z`（2.0 m），等 `READY`；READY 时刻记为 episode 的 t=0（起飞不计入）。
  - `step(action, dt)`：`act(until_s = 当前 SITL 时刻 + dt)` + `round(dt/1ms)` 个 1 ms 步，每 0.1 s（`trace_dt`）取一次中间状态进 `self.trace`；返回 dict 与 `CfBlackbox.step` 同名字段（`t x y z vx vy vz yaw yaw_deg yaw_rate active_action path_m pose velocity`）另加 `quat roll_deg pitch_deg exec_state`；FAILSAFE 直接抛错。
  - `camera_viewmat(nav_to_world)` = 机体四元数 ⊗ `camera_mount`；`manifest()` 含 `dynamics=cf_sitl, vehicle=e1_mini, params_sha256, calibrated=false, takeoff_s, trace_dt_s, action_window, episode_clock`。
- `sim/3dgs/scene.py`：新增 `viewmat_from_quat(pos, quat, mount, nav_to_world)`（旧函数未动）；四元数只含 yaw、mount=None 时与 `viewmat_from_pose` 数值一致（单测 1e-9）。
- `sim/3dgs/renderer.py`：`GsplatRenderer`/`MockRenderer` 新增 `render_view(view, width, height)`；`render_pose` 改为算 viewmat 后调它（签名、输出不变）。
- `sim/3dgs/collect_3dgs.py`：`run_episode(..., dynamics="cf_blackbox")` + `--dynamics {cf_blackbox,cf_sitl}`；cf_sitl 分支 reset→起飞、`render_view(dyn.camera_viewmat(...))`、`finally` 里 `dyn.close()` 并写 `trace.jsonl`；manifest 加 `dynamics_backend`、`trace_ref`，`dynamics_profile/params` 改取 `dyn.profile()/manifest()`（blackbox 输出不变）；notes 按后端写。顺手修了 `pose_before.z` 原来写的是步后 z（z 固定时无差别）。run.json 加 `dynamics`。
- `sim/3dgs/eval_3dgs.py`：同样加 `--dynamics`，params 按后端选，eval.json 加 `dynamics`。
- `sim/3dgs/video.py`：新增 `compose_views_video(ep_dir, scene, renderer, mount, occ)`、`_TopMap`、`_label`、`_panel`，顶部配置区（`VIEW_FPS=10, FOLLOW_BACK_M=2.47, FOLLOW_UP_M=0.4, TOP_MARGIN_M=3, PANEL 320×240, STATUS_H=36, KEYFRAMES=4`，每项带取值依据）；CLI `python -m sim.3dgs.video --scene … --ep <ep_dir>…`；无参数仍跑 `_selftest()`（新增三视角 mock 用例）。`compose_rgb_video` 未改。
- `sim/3dgs/test_dynamics_sitl.py`（新，9 用例）：viewmat_from_quat × 3、CfSitlParams.manifest、真 proc：reset→READY（t=0、z≈2、path_m=0）、FORWARD 2 s 沿航向前进 0.3–0.8 m、YAW_LEFT 1 s 偏航 +20°–+45° 且原地、子进程被 kill 后 step 抛 RuntimeError、proc 路径不存在抛 FileNotFoundError。proc 不存在时进程用例 skip。
- demo 驱动（数据盘，不在仓库）：`/root/rivermind-data/uav_rgb/tmp/agent-3dgs-sitl/run_demo.py`（配置区：EPISODES=3, SEED0=0, STEPS=40（`DEMO_STEPS` 可覆盖）, HOLD_S=0.5, STOP_ON_COLLISION），流程 collect → video → report.json/report.md，守护进程方式启动，结束落 DONE。

### 实测事实（写进代码注释的依据）

- `cf_sitl` 噪声种子 = `--seed ^ episode`：我先用 seed==episode 探测，6 个 seed 轨迹逐位相同（异或为 0）。collect 里因此 `episode` 固定 0、`--seed` 按集递增（`seed = args.seed + ep`）。
- 起飞：地面 reset 后 6.55–8.13 s READY（seed 0–5 全部 READY，无 FAILSAFE）；若 spawn 直接给 z=2 则先自由落体再起飞，9.42 s。
- FORWARD 2 s：前进 0.49 m（稳态 0.29 m/s）；YAW_LEFT 1 s：+31.4°。
- 一次 `step n=500` 往返约 1 ms；SITL 不是瓶颈，渲染 3.6–4.0 ms/帧（p50，320×240，supersample 2）。
- 外部机位：Callinstrasse 是行人高度拍的，机位高于约 2.6 m 开始被天空壳/漂浮高斯糊住，从上往下看（4/6/8/12/20 m，max_scale 3/1/0.5 都试了）全是漂浮物（`tmp/agent-3dgs-sitl/views_probe/sheet_*.png`）。所以跟随机位只抬 0.4 m，顶视用占据代理 2D 地图 + 轨迹（任务允许的「固定顶视 + 轨迹叠加」口径，数据与碰撞判定同一份）。

## Callinstrasse 教师闭环 3 集（seed 0,1,2，hold 0.5，上限 40 步，`--stop-on-collision`）

输出：`/root/rivermind-data/uav_rgb/demo-e1-20261107/3dgs-callinstrasse-teacher/`（collect/ep_0000{0,1,2}、report.json、report.md、run.log、DONE）。窗口语义在每集 `episode_manifest.json` 的 `window_from=render_observation` 与 `dynamics_params.action_window`。

| seed | steps | 到达 | 碰撞 | 离 waypoint 最近 (m) | 末位姿 (x, y, z, yaw°) | z 范围 (m) | 动作分布 | 起飞 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 40 | 否 | 否 | 2.35 | (-8.33, -0.34, 1.99, -6.8) | 1.98–2.04 | FORWARD×40 | 6.65 |
| 1 | 40 | 否 | 否 | 2.16 | (-8.12, -0.39, 1.99, 0.8) | 1.98–2.02 | FORWARD×40 | 7.27 |
| 2 | 40 | 否 | 否 | 2.17 | (-8.10, -0.54, 1.99, -0.2) | 1.98–2.00 | FORWARD×40 | 8.13 |

到达半径 2.0 m（scene.json）。未到达的原因是算术：spawn→waypoint 8 m，需前进 ≥6 m；E1 固件执行器 0.3 m/s × 20 s 加上加速段只有 5.65–5.84 m。深度教师在这段开阔走廊里全程 FORWARD，没有触发避障。

补充（不在任务要求内，只为回答「能不能到」）：`DEMO_STEPS=48` 同配置再跑，输出 `/root/rivermind-data/uav_rgb/demo-e1-20261107/3dgs-callinstrasse-teacher-48steps/`：3/3 到达（最近 1.20 / 0.99 / 1.14 m），无碰撞，动作仍全 FORWARD。

### 视频与关键帧

每集 `views.webm`（VP8，960×276，10 fps，200 帧 = 20 s 仿真 1:1；三格 = 机载 320×240 | 跟随（同一 3DGS，机体后 2.47 m 上 0.4 m，只跟位置与偏航）| 顶视占据地图 + 轨迹；底栏 ASCII：t、step、动作、z、速度、yaw/roll/pitch，碰撞步会标红 COLLISION），关键帧 `views_key_00..03.png`：

- `/root/rivermind-data/uav_rgb/demo-e1-20261107/3dgs-callinstrasse-teacher/collect/ep_00000/views.webm` + `views_key_0{0,1,2,3}.png`
- `/root/rivermind-data/uav_rgb/demo-e1-20261107/3dgs-callinstrasse-teacher/collect/ep_00001/views.webm` + `views_key_0{0,1,2,3}.png`
- `/root/rivermind-data/uav_rgb/demo-e1-20261107/3dgs-callinstrasse-teacher/collect/ep_00002/views.webm` + `views_key_0{0,1,2,3}.png`
- 48 步补充：`/root/rivermind-data/uav_rgb/demo-e1-20261107/3dgs-callinstrasse-teacher-48steps/collect/ep_0000{0,1,2}/views.webm`（240 帧）

## 没做什么

- 没把 demo 驱动 `run_demo.py` 放进仓库：任务文件域没列新入口脚本，放在 `$DATA_ROOT/uav_rgb/tmp/agent-3dgs-sitl/`；需要的话主会话决定是否收进 `sim/3dgs/`。
- 顶视没用 3DGS 渲染：从上往下渲染全是漂浮物（见实测），改用占据代理地图；如果要「同一 3DGS」的顶视，得给渲染器加按 z 裁剪高斯的选项（改 renderer，超出最小改动）。
- 没模拟 `camera_latency`（e1_mini.json 0.10 s）与教师/学生决策延迟：现有 collect 窗口语义是渲染、决策、下发同刻，SITL 时钟只在 step 里走；observation 与 dispatch 语义在这里重合，写进了 manifest。
- 碰撞代理没随机体 roll/pitch 变：仍是圆柱包络沿直线段扫掠（collision.py v2 未动）；栅格 z 范围是 scene.json 的 [1.8, 2.2]，机体 z 掉出该范围时只查地面。本次 z 在 1.98–2.04，未出范围。
- `eval_3dgs --dynamics cf_sitl` 只做了 mock 单测链路验证（`test_collect_eval` 覆盖默认后端；cf_sitl 的 eval 没有学生权重可跑），没跑真实学生评测。
- 没改 `sim/cf_sitl/`、`sim/tools/*`、`ui/*`、`src/`；没 git commit。
- `MockRenderer.render_view` 从 viewmat 反推 yaw/pitch 画渐变，不表现 roll（mock 只用于冒烟）。

## 待确认

- 40 步上限下 3 集都差一点到达：是接受「未到达」作为 demo 结论，还是把 demo 的步数上限改成 48（已验证 3/3 到达）或把到达半径/航点改一下。
- `CF_SITL_PROC` 缺省指向 `tmp/build-cf-sitl/`（SITL lib agent 的构建目录），构建目录换了要改 `dynamics.py` 的默认值或设环境变量。
- 请求（给 SITL lib agent，不改也能用）：proc README 可以注明「噪声种子 = seed ^ episode，调用方别把两者传成同一个数」。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `3dgs-venv/bin/python sim/3dgs/test_dynamics_sitl.py`（真 proc） | `Ran 9 tests … OK` |
| `sim/3dgs/test_collect_eval.py` | `Ran 13 tests in 15.5s OK` |
| `sim/3dgs/test_collision.py` | `Ran 16 tests OK` |
| `sim/3dgs/test_dynamics.py` | `Ran 6 tests OK` |
| `RUN_3DGS_REAL=1 sim/3dgs/test_renderer.py`（真 gsplat，含 render_view 重构后的 render_pose） | `Ran 9 tests in 167s OK`（第一次含 JIT） |
| `python sim/3dgs/video.py`（自检，含三视角 mock） | `{"3dgs_video": "PASS", "frames": 3, "files": 4, "views_frames": 10}` |
| `python -m sim.3dgs.dynamics`（blackbox 自检） | `dynamics selftest PASS cf_blackbox_v1:0954b747` |
| mock 冒烟 `collect_3dgs --dynamics cf_sitl --mock-renderer --episodes 2 --steps 4` | 2/2 集成功，trace.jsonl 每步 5 行，manifest `dynamics_backend=cf_sitl` |
| demo 守护进程 `run_demo.py`（40 步） | collect 19.8 s、video 50.6 s（重跑地图版 ~60 s）、`DONE rc=0 01:00:49` |
| demo 守护进程 `DEMO_STEPS=48` | collect 24.8 s、video 90.5 s、`DONE rc=0 01:08:23` |
| 第二轮全部 `sim/3dgs/test_*.py`（最终代码，`RUN_3DGS_REAL=1`） | test_collect_eval 13 OK、test_collision 16 OK、test_dynamics 6 OK、test_dynamics_sitl 9 OK、test_renderer 9 OK（JIT 已缓存 4.9 s）；video/dynamics/collision 自检 PASS；日志 `$DATA_ROOT/uav_rgb/tmp/agent-3dgs-sitl/tests/all_tests_round2.log` |

守护进程信息：40 步 demo pid/pgid 205647（已结束），48 步 pid/pgid 211241（已结束）；日志 `<out>/run.log`，完成标记 `<out>/DONE`；停：`kill -- -<PGID>`。结束时 `pgrep -f cf_sitl_proc` 无残留。

## 风险与遗留

- `cf_sitl_proc` 一进程一实例、单线程：collect 每集起一个进程（起飞 ~7 s 仿真只要约 20 ms 墙钟），并行采集要起多个进程，没测过。
- `CfSitlDynamics.step` 的 `until_s` 用当前 SITL 时刻 + dt；若以后在 collect 里加入渲染/推理延迟模型，要把 until 改成「拍图时刻 + hold」并推进相应的空转步。
- `trace.jsonl` 每步 5 行，40 步约 65 KB/集；`tools/preprocess.py` 不读它，不影响打包。
- 跟随/顶视只是演示用；跟随格的黄圈是「机体应在的位置」，3DGS 里没有机体模型。
- 临时产物留在 `$DATA_ROOT/uav_rgb/tmp/agent-3dgs-sitl/`：`probe_proc.py`、`probe_takeoff.py`、`probe_views.py`、`views_probe/`、`synth/`、`smoke/`、`tests/*.log`、`run_demo.py`、`PROGRESS.md`（临时查看图 crop.png/lbl.png 已删）。
