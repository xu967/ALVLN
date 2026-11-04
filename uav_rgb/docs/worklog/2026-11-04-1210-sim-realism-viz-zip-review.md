# 仿真真实性 / 可视化 / 新版 zip 审查（只读）

- 时间：2026-11-04 12:10
- 范围：`uav_rgb/sim`、`uav_rgb/ui/studio.py`、`CF-Drone/`、`TinyAerialVLN_20261104.zip`、`TinyAerialVLN.md`
- 结论：审查完成；未改任何代码

## 做了什么

- 核对 zip（`TinyAerialVLN_SIM_AUDITED_20261104`）：7 个 sim/tools 文件的 sha256_before 与当前 HEAD 逐一相等，补丁基于当前 HEAD。
- 在 /tmp 副本应用 `SIM_ONLY_PATCH.diff`（`git apply -p3 --exclude='*test_sim_review_regressions.py'`，测试文件单独拷入），结果与 zip 内文件逐字节相同。
- 仿真建模审查：机体是 Gazebo 官方 X3 参数（`sim/worlds/uav_rgb.sdf:176` mass 1.5），用 MulticopterVelocityControl 做速度闭环，高度由真值 P 控制保持（`sim/tools/collect.py:98,129-137`）；真机 CF-Drone 是 118 mm 轴距（`CF-Drone/control.ino:13`），只有 STAB/ACRO 模式，MAVLink 只收 MANUAL_CONTROL / SET_ATTITUDE_TARGET（`CF-Drone/mavlink.ino:94,173`），没有速度环和定高。
- 可视化审查：`ui/studio.py:418-430` 先判 collision 再判 false_forward，而 `sim/tools/eval_closed_loop.py:193` 的 false_forward 定义必须伴随非 stuck 碰撞，所以「假前进」分类永远为 0；「闭环导航 SR」取的是 reach-ever 的 `success_rate`，分母是完成集。

## 没做什么

- 没有把 zip 补丁合入仓库。原因：合入属于代码改动，需要用户确认。
- 没有修 Studio 的 false_forward / SR 口径问题。原因：本次只做审查。
- 没有启动 Gazebo 或 GUI，也没有现场出图。原因：先给方案，等用户决定真机定高方式和接口。
- 没有核对 `CF-Drone/rc.ino`、`web_rc.ino` 的通道映射细节。原因：超出本次范围。
- 没有跑 `test_process_cleanup` 的真实 fork 压力用例以外的 GPU/CUDA 测试。原因：与本次无关。

## 待确认

- 真机定高方式：人手油门，还是给固件补 ToF 定高。这决定仿真动作接口怎么建模。
- 是否合入 zip 补丁。
- 真机实测参数（起飞重量、电机/桨、推力曲线、相机型号/内参/安装角、端到端延迟）谁来测。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 未打补丁源码 + 新测试 `test_sim_review_regressions` 等 26 项 | 失败（failures=19, errors=6），说明新测试能抓到旧缺陷 |
| 打补丁后 15 个测试模块共 38 项 `python3 -m unittest -q ...` | 通过（`Ran 38 tests in 1.119s OK`） |
| Gazebo 物理闭环 | 未跑 |

## 风险与遗留

- 当前所有闭环数字都基于 X3 + 速度控制 + 真值定高，不能外推到 CF-Drone 真机。
- Studio 历史 SR 是 reach-ever；按停住口径计，distill-base 从 0.487 降到 0.256（数据来自 subagent 对 `/root/rivermind-data/uav_rgb` 的核对，我只抽查了代码逻辑）。
- zip 不含 `sim/worlds/templates/`，如果用 zip 整包替换仓库会丢掉这些归档模板。

## 更新 2026-11-04 12:40

- 用户决定：仿真高度先固定，继续用真值定高（`sim/tools/collect.py:98-101`，2.1 m ± 0.2 m 死区），不建模油门，也不建模 ToF。
- 更正上文：`scene_gen.py:77` 的 `CRUISE_HEIGHT=2.3` 是高度带上沿（2.1 + 0.2 死区），用来检查柱子在整条高度带内都挡路（`scene_gen.py:130`），这不是矛盾。
- 本次仍未改代码。下一步（水平/偏航对齐真机）等用户确认方案和实测参数。

## 更新 2026-11-04 14:30：实现 cf_stab 动力学档

- 结论：完成（参数为占位值，等真机实测）

### 做了什么

- `sim/tools/collect.py`：新增 `--dynamics {x3_velocity,cf_stab}`，默认 `x3_velocity`，旧行为逐字不变。新增 `CfStab` 类：FORWARD_SLOW 时俯仰杆保持固定倾角，水平加速度 g·tanθ，线性阻力 k；其余动作松杆滑行，不刹车；指令延迟 0.15 s；速度在世界系积分，发指令前转回机体系；偏航沿用 ±0.6 rad/s；高度仍用真值定高。动作窗口内按位姿帧逐帧更新指令，窗口末不再发零速。卡滞判定按模型预测位移缩放。manifest 写入 `dynamics_profile`（档名里带参数）和 `dynamics_params`。
- `sim/tools/collect_parallel.py`、`sim/tools/eval_closed_loop.py`：透传 `--dynamics`，汇总 manifest 和 `eval.json` 记录 `dynamics`，每集记录 `dynamics_profile`。
- `sim/tools/audit_dataset.py`：遇到 cf_stab 集，`forward_motion` 和 `transition_lag` 记为 SKIP（这两项是定速判据）；`yaw_direction` 跳过偏航头两步。
- 测试：`test_collect.py` 新增 14 项（解析解、延迟、滑行、机体系、倾角上限、窗口内逐帧更新、不刹停、起步不误报卡滞、巡航贴柱判卡滞），`test_audit_dataset.py` 新增 cf_stab 场景。
- 结果文档：`docs/experiments/2026-11-04-cf-stab-dynamics.md`。

### 没做什么

- 没换掉 `MulticopterVelocityControl`，也没写 gz-sim C++ 插件。原因：高度固定以后，用速度曲线去驱动现有控制器就够了；代价是 X3 跟踪会有约 0.4 s 滞后，实测数字写在结果文档里。
- 没改机体尺寸、碰撞盒和 `scene_gen.py` 的 `SPAWN_BODY_HALF/ROTOR_*`。原因：缺真机桨径和外廓尺寸，改了也是猜的。
- 没改相机（噪声、畸变、内参、俯仰角）。原因：缺相机型号和标定数据。
- 没加风。原因：本次范围只做动力学。
- 没调 SDF 里的控制器增益。原因：/tmp 对照（g1）显示能减小稳态误差，但起步滞后不变；改模板会改 `template_sha256`，影响历史口径，需要你决定。
- 步间（一步结束到下一步派发之间）的指令是常量，阻力衰减要到下一步才补上。原因：当前步间隔约 0.07 s，误差很小，已用 `shortcut:` 注释标明。
- 没用 cf_stab 采正式数据、训练或做 baseline 评测。原因：参数还是占位值，做出来的数字没有意义。
- 没提交 git。原因：没有得到提交授权；`tools/preprocess.py` 的未提交改动不是本次做的，没动。
- zip 补丁没合入（`test_sim_review_regressions` 在仓库里不存在，所以不能跑）。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `sim/tools/test_*.py` 共 20 个模块（不含仓库里没有的 test_sim_review_regressions） | 全部 rc=0；`test_collect: 46 项通过` |
| Gazebo 采集 cf_stab：2 实例 × 2 集 × 60 步 | `4/4 episodes, 240/240 steps`，误判卡滞 0 |
| audit cf_stab | `yaw_direction` PASS (0/47)；`class_share` FAIL，旧档 x3 同样 FAIL，属于数据本身的问题 |
| preprocess 混合 cf_stab 与 x3 | 按 `dynamics_profile 不一致` 拒绝 |
| 闭环评测盲前进 cf_stab，2 集 | 都在 x≈1.07 m 处被接触传感器判碰撞；`eval.json` 中有 dynamics 字段 |

### 风险与遗留

- 倾角 2°、阻力 k=1.0、延迟 0.15 s 都是占位值，换成实测值后档名会变，旧的 cf_stab 数据不能和新数据混包（preprocess 会拦住）。
- cf_stab 的卡滞阈值是按 X3 跟踪滞后的实测值定的；换底层控制器或改 hold 之后要重新测误报率。
- base 任务在 cf_stab 下 SAFE_NEUTRAL 为 0，正式采集前要先确认类别分布方案。

## 更新 2026-11-04 14:35：对照评测 + 三视图视频 + 训练产物落数据盘

### 做了什么
- 6 组闭环对照（x3/cf_stab × teacher/student/盲前进，17 个同场景），结果和读数写在
  `docs/experiments/2026-11-04-cf-stab-dynamics.md`「对照评测」一节。
- 新增 `sim/tools/record_views.py`：评测时录 机载/追尾/顶视 三路相机，按仿真时间合成 1280×720 10 fps mp4，
  叠加步数/位置/速度/动作/结果、顶视轨迹投影、动作时间条。帧原始数据写在输出目录旁（数据盘），合成后删除。
  测试 `sim/tools/test_record_views.py`（不需要 Gazebo）。
- `eval_closed_loop.py` 新增 `--record-video`，每集写 `<out>/videos/seed_<seed>.mp4`。
- `teacher/train.py` 新增 `--record-video`（默认关），训练结束后用 `ckpt_best.pth` 跑闭环并录像；
  `--out` 相对路径落 `$DATA_ROOT/uav_rgb/runs/`，绝对路径不在 `$DATA_ROOT` 下直接报错。
- `ui/monitor.py` 默认 run 目录从仓库 `logs/runs` 改到 `$DATA_ROOT/uav_rgb/monitor_runs`。
- teacher venv（数据盘）装了 `opencv-python-headless`，用于合成视频。

### 验证
- 端到端：`train.py --epochs 1 --record-video --video-episodes 1` 产出 ckpt + eval.json + seed_1.mp4，全在数据盘。
- 抽帧人工核对：顶视投影圆圈落在柱顶；追尾/机载/状态栏与时间对齐。
- 测试：`sim/tools/test_*.py` 20 个模块、`teacher/test_train_metrics.py`、`ui/test_monitor.py`（117 项）通过。

### 没做什么
- 没改教师的 cf_stab 适配（下一步，见实验文档结论）。
- Studio GUI 里还不能直接播放视频，只是文件。
- 没有 ffmpeg，mp4 用 OpenCV `mp4v` 编码；部分浏览器不能内嵌播放，本地播放器可以。
- 仓库内既有的 `logs/runs` 历史目录没动（目前为空）。
