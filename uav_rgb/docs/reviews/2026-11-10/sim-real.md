# 仿真无人机为什么不像真机、怎样重新仿真（只读审查，2026-11-10）

## 结论摘要

1. 仿真里飞的是「开源固件 e7f507f 内环 + 仿真自写的外环 `executor.c`」。官方闭源固件 20261102 自带定高、定点、光流、电压补偿和控制源仲裁（B 节字符串证据）。真机若刷官方固件，仿真的外环在真机上对应一段看不到源码的固件。真机软件栈没有定，这是「不像」的第一原因，物理参数没标定排第二。
2. 实测：在能正常起飞、不中止且光流有效的变体里，24 个物理参数在各自区间内只让稳态每步前进变化 ≤2%、每步转角变化 ≤0.3%。敏感的是外环设定和时序：动作生效每晚 0.1 s，每步前进 −10%、转角 −12%；偏航设定值只比固件死区高 15%（0.600 对 0.524 rad/s），掉进死区后偏航动作整体失效；光流在 2.0 m 无效时每步前进变成 1.07 m。
3. 对学生策略影响最大的三件事：相机链路（真机图像可以裁成仿真的 60°×46.8°，仿真不用改；训练流水线没有开图像增强）、巡航高度 2.0 m 贴着 ToF 和光流的工作上限、四个动作在真机上由谁计时并换算成杆量。
4. 重新仿真的顺序：定软件栈（建议先验证「官方固件定点模式 + 外部发杆量」）→ 台架称重和测推力 → 飞四个原语并俯拍 → 把执行器当黑盒拟合到真机原语 → 过 D3 的门再置 `calibrated=true`。
5. 仿真侧新发现 10 条问题（E 节），最重的是 FAILSAFE 开环保持悬停油门：推力比假设高 34% 时 25 s 爬到 60 m。

证据标注：【实测】本次用 `$D/tmp/build-cf-sitl` 的二进制跑出的数，产物在 `$D/tmp/cursor-1110/agents/sim-real/probe/`；【扫描】`$D/exp-sim-dyn-20261110/sweep-hold1.0/table.md`（32 集 / 变体）；【文档】官方文档或官方 bin 字符串；【推算】由 assumed 值或手算得出，没有实物或仿真核对；【未知】没有任何来源。

路径缩写：`fw/` = `/root/claude-code-backup-20261028/code/CF-Drone/`；`sitl/` = `uav_rgb/sim/cf_sitl/`；`json` = `uav_rgb/sim/dynamics/e1_mini.json`；`sdf@HEAD` = `git show c85a1ff:sim/worlds/uav_rgb_e1.sdf`（工作树里并发会话改了护罩，相机段行号要 +24）；官方字符串偏移都指 `$D/tmp/research-hw/official_utf8.txt`。

## A. 仿真与真机的差距总表

| 维度 | 仿真现状 | 真机 | 对学生 sim-to-real 的影响 | 怎么消除 |
| --- | --- | --- | --- | --- |
| 1 质量与惯量 | `json:10` 0.102 kg = 文档 80 g + 估计载荷 22 g；`json:18` 惯量由组件模型估；用在 `sitl/src/plant.h:198,206-207`；`cg_offset`（`json:12`）SITL 不用 | 【文档】「重量：约80克（含标准电池）」；载荷件、护罩、惯量、重心【未知】 | 位移：低（【扫描】92–110 g 稳态前进 ±1%；惯量 ×0.5–×2 首步转角 +2.7 / −5.3%）。安全：高（110 g 悬停占空比 0.864，饱和 7.9%） | D1 的 B1、B2、B5，直接代入 |
| 2 电机推力曲线与时间常数 | `plant.h:167-168`：`T = t_full·(n·V/3.7)^1.5`，占空比一阶滞后 0.06 / 0.12 s；38 gf、指数、时间常数全是估计（`json:19-24`） | 8520 有刷 + 65–75 mm 桨（`json:16,19`），MOSFET 直驱，PWM 25 kHz 10 bit（`fw/motors.ino:14-17,60`）；推力和时间常数【未知】 | 位移：低（【扫描】推力、指数、时间常数各自区间内五个原语量变化 ≤3.1%）。安全：高（34 gf 饱和 10.6%；110 g + 34 gf + 20% 电量的组合 0/32 到 READY；≥48 gf 且前馈不改时起飞飞丢，见 E1、E2） | D1 的 B3、B4 |
| 3 电池压降 | 开路电压表 + 内阻 0.06 Ω + 功率模型（`plant.h:136-155`）；每集从 50% 电量开始（`json:117`）；固件不做电压补偿 | 500 mAh 1S、25C 以上【文档】（`json:31-32`）；内阻【未知】；官方固件有电压补偿（`0x0156a7`），开源固件没有 | 位移：低。安全：高（【扫描】0.10 Ω 时只有 2/32 到 READY 且都掉高中止，饱和 18.2%；电量 20% → 100% 悬停占空比 0.840 → 0.728） | D1 的 B6；电量按集随机 |
| 4 气动阻力 | 线性 0.040 / 0.020 N·s/m + 二次 0.003（`plant.h:177-180`，`json:26-27`）；无风 | 【未知】 | 位移：低到中（【扫描】×0.5–×2 首步前进 +4.1 / −9.7%，稳态 ±2%） | 用原语飞行里的松杆滑行段拟合 |
| 5 IMU 噪声与零偏 | `sitl/src/sensors.h:17-35`：每集抽零偏 1σ 0.1°/s、随机游走、一阶低通 184 Hz、白噪声随转速插值到 2°/s；全是估计（`json:40-46`） | MPU6500【文档】（`json:38`）；有刷电机硬连 PCB，振动水平【未知】 | 位移：中（【扫描】零偏 0.5°/s 时悬停漂移 0.86 → 5.33 cm/s，侧漂 0.97 → 3.93 cm/s）。安全：中（噪声 5°/s 饱和 21.6%） | D1 的 F1 |
| 6 姿态估计 | 固件原样：飞行中只积分陀螺，加计只在落地时修正（`fw/estimate.ino:51-56`）；执行器靠速度环积分吸收漂移，20° 限幅在 3σ 零偏下撑 69 s（`sitl/README.md:112`） | 开源固件同左；官方固件有 Mahony 滤波（`0x013b83 MAHONY_KP`），磁力计是可选件（`0x014307`、`0x013d76 MAG_YAW_W`） | 图像：低（几度姿态误差进画面）。安全：高（方案 3 下单次飞行约 1 分钟后积分饱和） | 取决于软件栈；方案 3 先给开源固件加飞行中加计修正，SITL 原样编译会自动跟上 |
| 7 控制环（角速率 / 角度 PID、混控、油门映射） | 固件 `.ino` 原样编译（`sitl/fw/fw_sitl.cpp:18-30`），1 ms 锁步；角度 P 6、角速率 PID 0.06 / 0.1 / 0.001、混控、油门杆 0.05..1 映射到占空比 0.10..0.90（`fw/control.ino:15-36,157,224-227`） | 开源固件与仿真一致（WiFi 负载下可能漏拍，`json:65`）；官方固件多出 `CTL_STICK_DZ`（`0x013ae5`）、`CTL_TRIM_LRN`（`0x013ad8`）等，增益【未知】 | 无到低：内环 0.3 s 内跟上（【实测】selftest 3.5：打杆 0.5 → 俯仰 15.0°，上升 0.32 s；偏航 150°/s，上升 0.22 s） | 方案 1 只对齐外环响应；方案 2、3 已对齐 |
| 8 RC 输入链 | 绕过：`fw_sitl.cpp:122-128` 直接写 `control*`，`sitl/src/cf_sitl.cpp:57-58` 每 1 ms 一帧、零延迟；失控保护永不触发 | CRSF 420000 → `normalizeRC()`（`fw/rc.ino:172-183`）；没做 `cr` 校准时通道号是 NAN，杆量也是 NAN，不是偏 +0.5（`rc.ino:25,178-182`）；失联 1 s → `descend()` 把推力钳到 0.5（`fw/safety.ino:55-56,74`），对 102 g 是 5.2 m/s² 下坠（`json:149`）；官方固件：`crstd` 预设（`0x01155a`）、控制源仲裁、失联自动降落（`0x0129b4`） | 位移：中（刷新率 50–250 Hz、11 bit 量化、死区）。安全：高（偏航杆 0.1146 离死区 0.1 只差满程 1.5%；【实测】设定值 0.53 rad/s 每步 30.3°，0.52 rad/s 每步 0°） | E5 的加载期校验；D4 第 7 项。方案 2、3 下可把真机 `CTL_Y_RATE_MAX` 调到 150°/s（`fw/parameters.ino:81`），偏航杆变成死区的 2.3 倍，仿真要同步改 `yaw_rate_per_stick_dps` 和 SITL 里的固件参数（后者的入口本次没查） |
| 9 定高（ToF） | 执行器 g-h 观测器 + PD + 积分（`sitl/exec/executor.c:45-66,82-90`）；ToF = 到无限大平地面的斜距 + 1 cm 白噪声，30 Hz，0.04–2.5 m（`sensors.h:39-48`，`json:48-51`） | 开源固件没有（`fw/control.ino:43` 是 stub）；官方固件：VL53L1X Medium 连续模式 + 气压计混合 + 悬停推力学习（`0x012666`、`0x015972`、`0x0159b4`），超量程报「无法维持高度」（`0x0101ac`），`ALT_TOF_MAX`（`0x013e6d`）取值【未知】 | 图像：中（高度误差改视角；仿真 z 偏差 0.04 m）。安全：高（2.0 m 离假设量程上限 0.5 m；【实测】ToF 10 Hz 时高度在 1.66–2.18 m 间振荡） | D1 的 S1；先定巡航高度（D4 第 2 项） |
| 10 水平速度与位置保持（光流） | 理想光流 `v/h − ω` + 0.05 rad/s 白噪声，100 Hz，0.08–2.5 m 有效，与地面纹理和光照无关（`sensors.h:55-66`）；执行器 PI 速度环 + P 位置环（`executor.c:97-136`） | 开源固件没有；官方固件：PMW3901 + 卡尔曼速度观测 + 定点（`0x015f09`、`0x015e86`）；10 种失效原因（`0x010609–0x0106a8`），含「高度门(无高度源或超量程)」「表面质量为0」「倾角过大」「角流超限」；要求中频随机纹理、光源不闪（`0x011913`、`0x011a09`）；`FLOW_SCALE` 要用尺子标（`0x013655`）；【推算】2 m 高时每像素约 44 mm 地面 | 位移：高（光流尺度误差 1:1 进每步位移，`executor.c:71`；【实测】2.0 m 处光流无效时每步前进 1.07 m、刹停 2.65 m、悬停漂移 100 cm/s，见 E6）。安全：高 | D1 的 S2；仿真加光流尺度误差的随机化（D4 第 4 项） |
| 11 自动起飞与降落 | 执行器自动解锁、0.5 m/s 斜坡到 2.0 m、READY 判据（`executor.c:157-178`，6.5–9.5 s）；没有降落状态，集结束靠 reset | 开源固件没有；官方固件没有自动起飞（`0x01313c`「爬升到 %.2fm 以上定高才接管，在此之前是手动油门」），有贴地提示和落地判定（`0x013180`、`0x0130a9`）、失联和低电自动降落 | 图像、位移：无（起飞段不进数据集）。安全：高 | 方案 1：人手或脚本油门起飞后交权；方案 3：执行器补 LAND |
| 12 四个动作的语义 | FORWARD = 速度环目标 0.3 m/s，YAW = 开环杆量 0.6 rad/s，SAFE = 位置保持（`executor.c:180-195`）。【实测】hold 1.0：首步 0.211 m、稳态 0.298 m、刹停滑行 0.10 m（停住 0.82 ± 0.80 s）、首步 31.7°、稳态 34.3°、FORWARD 接 YAW 滑 0.127 m；hold 0.5：首步 0.046 m、稳态 0.149 m、14.5° / 17.2°。仓库自己记的是 Gazebo 0.25 m/步、3DGS 约 0.12 m（`tools/preprocess.py:52`） | 没有定义。官方定点模式的杆量到速度的映射、`PH_VEL_MAX`（`0x013eba`）、`PH_ACC_MAX`（`0x013ece`）、`CTL_STICK_DZ` 取值【未知】 | 位移：高（标签「前进一步」对应多少米由它决定；hold 0.5 时首步只有稳态的 31%） | D1 的 P1–P4 实测，D2 拟合 |
| 13 LicheePi → ESP32 指令接口与时延 | 不存在：动作是同进程函数调用，截止时刻精确到 1 ms（`cf_sitl.cpp:116-118`） | `tools/uart_action.py` 是占位（9 字节帧、10 Hz 心跳、TTL 300 ms、波特率 115200 也是占位；`docs/uart_action.md:3-5` 自述未联调、固件没有解析器）；现成入口：CRSF 进 IO4【文档】、MAVLink UDP `MANUAL_CONTROL`（`fw/mavlink.ino:94-105`）、官方 WebSocket :81 JSON（变化时最快 20 Hz，静止 5 Hz 重发） | 位移：高（【实测】动作生效晚 0.1 s：稳态前进 −10.2%、转角 −12.1%；晚 0.2 s：−20.8% / −25.3%） | 定接口，测延迟（D1 的 C4），两条仿真线统一成显式参数（D4 第 3 项） |
| 14 相机 | 理想针孔：HFOV 60°、320×240、10 Hz、无畸变、无噪声、无运动模糊、无曝光变化、全局快门（`sdf@HEAD:307-327`；`sim/3dgs/scene.py:236,372-374`）；60° 是用户 11-07 拍板（各 `scene.json` 的 `fov_source`）；机体姿态进画面；Gazebo 里护罩和桨在近裁剪面 0.05 m 以内，3DGS 不渲染机体；延迟：3DGS 线 0，Gazebo 线约一帧 0.1 s；`camera_fov`、`camera_capture` 没有代码读，`camera_latency` 只有教师读（`sim/tools/teacher_kino.py:82`） | 型号本身是假设（`json:60`：LicheeRV Nano 配件相机 wiki 没写，暂按 GC4653）：2560×1440 卷帘、H81° / V51°、畸变 5%、720P 时 60 fps（`research-hw/maixcam_cameras.md:8-21,30`）；安装位姿、曝光策略、延迟【未知】；相机链路代码不存在；【推算】护罩外弧可能进入裁剪后画面的边角（按工作树 SDF 的护罩高度 0.026 m 和相机高度 0.015 m 算是上角，`json:63` 写的是下沿，两个高度都是 assumed） | 图像：高 | 真机图像去畸变后裁成 60°×46.8°（在 81°×51° 以内，720p 下约 865×649 px）再缩到 320×240，仿真和旧数据都不动（D4 第 1b 项）；训练开增强（D4 第 1a 项） |
| 15 机体外观与碰撞体 | 碰撞体圆柱 r 0.102 m × 0.048 m（`sdf@HEAD:189-191`），3DGS 用 `collision_envelope`（`json:17`）；外观：HEAD 是方框护罩，工作树已改成 8 个圆柱（未提交） | STL 四叶草，最大半径 101.4 mm；z 向尺寸没有来源（`json:17` 的 basis） | 图像：无（仿真里机体不入画）。碰撞判定：低（半径差 0.6 mm）。只影响第三人称录像 | 并发会话已在做 |
| 16 地面效应与近墙气流 | 没有：力只有电机推力、阻力、重力（`plant.h:163-180,198`），地面是硬点接触（`plant.h:200-212`） | 【未知】 | 巡航 2 m：无到低。起降和贴柱：低到中（安全） | 不建模；起飞段不进数据，真机通道留余量 |
| 17 环境（光照、纹理、地面对光流的影响） | Gazebo：灰柱 0.4×0.4×3 m + 均匀灰地面（`sdf@HEAD:59-74`），每个 seed 随机太阳和环境光（`sim/tools/scene_gen.py:311-343`）；3DGS：真实重建场景，从 2 m 新视角渲染；光流与地面纹理无关 | 场地【未知】；官方固件对光流地面有明确要求（第 10 行） | 图像：高（Gazebo 线）、中（3DGS 线）。位移：高（地面纹理差时光流失效） | 真机场地拍照重建一个 3DGS 评测场景；场地铺线宽 5–10 mm 量级的随机纹理（`0x011913` 的原话） |
| 18 时间同步 | 单一仿真时钟。Gazebo 线：观测帧比下发早约一帧相机，位姿戳间隔 40/40 步是 0.900 s（`$D/tmp/phase3-20261107/reports/REVIEW2_SITL_20261107.md:83,115`）；3DGS 线：渲染与下发同刻（`sim/3dgs/dynamics.py:258`）。两条线口径不同 | LicheePi（非实时 Linux）和 ESP32 两个时钟，没有同步手段；曝光时刻、推理耗时、串口抖动【未知】 | 位移：中到高（即第 13 行）。评测：中 | 动作帧带序号和 LicheePi 单调时钟并落盘；LED 对拍（D1 的 C4） |

对已知结论的补充和更正：
- 结论 1（外环在执行器里）：「真机没有」只对开源固件成立，官方固件有定高、定点和失效降级（B2）。
- 结论 2（方框护罩）：`sim/tools/e1_body.py`、`sim/worlds/uav_rgb_e1.sdf` 在工作树里已被并发会话改成四叶草近似（未提交），HEAD 仍是方框。
- 结论 3（相机）：60° 是用户拍板的口径；60°×46.8° 落在 81°×51° 以内，可以在真机侧裁出来，不必改仿真。「无延迟」只对 3DGS 线成立，Gazebo 线有约 0.1 s 的隐含延迟。
- 结论 4（绕过 `normalizeRC`）：还有两条后果。一是杆量每 1 ms 刷新、零延迟；二是固件的 RC 失联保护在仿真里永远不会触发。
- 结论 5（`altitude_assumption`）：同一个字段在 `sim/3dgs/dynamics.py:130` 也有。
- 第十阶段文档里「不跑 cr 校准时真机杆量整体偏 +0.5」不准确：通道映射默认是 NAN（`fw/rc.ino:25`），`control*` 得到的是 NAN。

## B. 真机到底跑什么飞控

### B1 事实

- 文件里没有任何一处记录用户那架真机刷的是哪个固件：`$D/real-drone/` 只有一份 README，`json:5` 的 `"firmware": "CF-Drone e7f507f"` 是仿真自己的假设，`json:150` 也写着「真机若跑官方闭源固件……本文件的 firmware 段不适用」。
- 官方文档（`research-hw/e-series_e1-mini_extended-module.html`）原文：「使用官方固件可自动识别扩展模块，安装后重新上电即可识别模块，自动根据模块类型释放定高、定点飞行模式。如果使用开源代码编译，需自行开发定高定点相关功能。」
- 开源 e7f507f 只有 RAW / ACRO / STAB / AUTO；`fw/board_config.h:139-141` 给 VL53L1X 和 PMW3901 留了引脚，没有驱动代码。MAVLink 只走 WiFi UDP，接受 `MANUAL_CONTROL`、AUTO 模式下的 `SET_ATTITUDE_TARGET` 和 `SET_ACTUATOR_CONTROL_TARGET`（`fw/mavlink.ino:94-105,173-207`）。

### B2 官方 bin 的字符串证据

来源：`official_E1_MINI_20261102.bin` 的字符串表，偏移和原文本次逐条复核过。

| 功能 | 偏移与字符串 |
| --- | --- |
| 构建 | `0x010050 'arduino-lib-builder'`、`0x010090 'v5.5.5'`。`0x010080 'Sep 16 2026'` 在 app 描述区，是 Arduino 预编译库的日期，不是固件日期 |
| 飞行模式 | `0x0152f0 '无效模式名(可选: acro/stab/althold/poshold): %s\n'`、`0x014307 '  能力: 定高=%s 光流=%s 磁力计=%s\n'`、`0x0104fb '无光流，已降级到定高'`、`0x01053f '无高度传感器，已降级到自稳'` |
| 定高 | `0x0130bf '定高已接管'`、`0x01313c '爬升到 %.2fm 以上定高才接管，在此之前是手动油门'`、`0x015645 'altState(定高子状态): %s  thrustTarget(目标推力): %.3f  hoverThrust(悬停推力): %.3f\n'`、`0x015972 'altSensorBlend(定高增益混合系数 1=ToF/0=气压计): %.2f\n'`、`0x01074a '悬停推力越界(%.3f)，已钳到[%.3f,%.3f]\n'` |
| ToF | `0x012666 '  VL53L1X: 初始化成功（continuous %ums, Medium, budget=%ums），首测 status=%d mm=%u\n'`、`0x0101ac '激光测距超出量程，无法维持高度'`、`0x015703 'failsafe(故障保护): 高度失联强制降落中\n'` |
| 光流与定点 | `0x0130e5 '定点已恢复'`、`0x0130f5 '光流失效(%s) 定点退化为手动姿态'`、`0x015e86 'posHold(定点): corrRoll(横滚修正rad)=%.3f corrPitch(俯仰修正rad)=%.3f posX(前向积分m)=%.3f posY(左向积分m)=%.3f\n'`、`0x015f09 'velEst(卡尔曼观测器 accStd=%.2f kb=%.2f): velEstX(前向m/s)=%.3f …'`（后半截略）、`0x015fa4 'posTraj(轨迹): posDesX=%.3f posDesY=%.3f errX(m)=%.3f errY(m)=%.3f\n'` |
| 光流使用条件 | `0x011913 '[ft] ⚠ 靶面必须是【中频随机纹理】：花纹地毯/迷彩布/粗记号笔乱画(线宽5~10mm)/报纸大标题；\n'`、`0x011a09 '[ft] ⚠ 光源不能闪：屏幕/PWM调光LED 与芯片约121fps 拍频会直接破坏帧间相关，用白天自然光最稳\n'`、`0x011e3c '[fc] 当前高度 %.2fm ⇒ 每像素覆盖地面 %.1fmm，视场 %.0fmm；目标纹理特征需明显大于单像素尺寸才可能被分辨\n'`、``0x013655 '[ft] 标定：拿尺子量的真实距离除以上面对应轴的值，再乘当前 %.3f，填入 `p FLOW_SCALE <值>`\n'`` |
| 控制源 | `0x010120 'SBUS'`、`0x010125 'CRSF'`、`0x01012a 'ESP-NOW'`、`0x01058c '实体遥控'`、`0x010599 'Web遥控'`、`0x0105a3 'MAVLink'`、`0x012d11 '控制源切换: %s -> %s\n'`、`0x012ce7 '%s正在控制，网页操作已被忽略'`、`0x0163ca 'MAVLink解锁被拒绝: %s'`、`0x01155a '已应用 %s 标准通道预设：CH1横滚 CH2俯仰 CH3油门 CH4偏航 CH6模式\n'` |
| 失控与电池 | `0x0129b4 '控制链路失联(%s) 自动降落'`、`0x0156a7 'batteryVoltage(电池瞬时电压V): %.3f  voltComp(补偿系数=%.1fV/滤波电压): %.3f\n'` |
| 参数 | 官方 123 个参数名，开源 87 个，官方独有 61 个：`ALT_TOF_MAX`（`0x013e6d`）、`ALT_VEL_P`（`0x013dba`）、`ALT_HOVER_TAU`（`0x013dfc`）、`PH_VEL_P`（`0x013e95`）、`PH_VEL_MAX`（`0x013eba`）、`PH_POS_P`（`0x013ec5`）、`PH_ACC_MAX`（`0x013ece`）、`FLOW_SCALE`（`0x013d4d`）、`MAHONY_KP`（`0x013b83`）、`CTL_STICK_DZ`（`0x013ae5`）等 |
| 日志 | `0x01649d–0x01669c` 共 47 个列名，含 `batteryVoltage`、`motThrOut`、`altitude`、`verticalVel`、`hoverThrust`、`measAlt`、`posX`、`posY`、`flowVelX`、`flowVelY`、`flowSqual`、`phVelEstX` |
| CLI | `0x0166ab` 菜单里有 `ft`（光流台架自检）、`fc`（光流原始帧）、`crstd`、`status`（解锁 / 状态机 / 高度 / ToF）、`acro/stab/althold/poshold`、`arm`、`disarm`、`log [dump]`，没有起飞或降落命令 |
| 网页遥控 | `0x0294d7` 内嵌页面：`new WebSocket('ws://' + location.hostname + ':81')`；杆量包 `{t:1, th, r, p, y}`（−100..100，每 50 ms 检查一次，有变化就发，静止时每 200 ms 重发）；按钮包 `{t:2, b, s}`；模式名 `['特技','自稳','定高','定点','自动降落']`；注释「ALTHOLD(2)/POSHOLD(3)：摇杆居中=定高/定点悬停（不改变高度）」 |

没有搜到：任务、航点、waypoint、mission、offboard、自动起飞、一键起飞，也没有自定义串口动作协议。官方固件的 MAVLink 接受哪些消息是二进制逻辑，字符串看不出来。

### B3 三个候选软件栈

| 方案 | 真机上跑什么 | 仿真执行器怎么处理 | 风险 |
| --- | --- | --- | --- |
| 1 官方固件 + 定点模块，外部只发杆量 | 飞控刷官方 20261102，模式 POSHOLD。上位机把动作翻译成杆量：FORWARD = 俯仰杆 x 保持 1 s，YAW = 偏航杆，SAFE = 回中。链路三选一：CRSF 串口进 IO4（`crstd` 预设，CH6 切模式）、WebSocket :81 JSON、MAVLink `MANUAL_CONTROL` | 保留，当作「官方定点模式 + 机体」的黑盒替身；`forward_speed_mps`、`vel_kp`、`vel_ki`、`yaw_rate_rps`、延迟用 D2 拟合到真机原语。固件在环的内环不再代表真机，陀螺零偏引起的漂移量按 P3 实测重设 | 闭源：杆量语义、死区、最大速度不能改，固件一升级行为可能变（oshwhub 附件是 20261021，下载页是 20261102，`research-hw/oshwhub.txt:238`）；没有自动起飞；2 m 高度下光流和 ToF 能否工作【未知】；与 `docs/uart_action.md:161-169` 的现行安全口径冲突（那里写明不能直接拿 `MANUAL_CONTROL` 或 Web RC 当四动作） |
| 2 开源固件 STAB + LicheePi 跑外环 | `executor.c` 编到 LicheePi，ToF 和光流接 LicheePi，杆量经 CRSF 发给飞控 | 保留，但 SITL 要让执行器按 LicheePi 的频率跑并加链路延迟（约 20 行），杆量走 `normalizeRC`（D4 第 7 项） | 光流去旋转要 100 Hz 陀螺（`executor.c:71-72`），CRSF 是单向的，飞控的 MAVLink 遥测只有 10 Hz 且走 WiFi（`fw/mavlink.ino:47-67`），得另加 IMU 或改固件；非实时 Linux 同时跑 CNN 和高度环，而 50% 电量时最大推重比只有约 1.10（`json:148`） |
| 3 把 `executor.c` 移植进开源固件（`sitl/README.md:88-92` 的原设计，也是 `docs/uart_action.md:23-24` 设想的「另一版固件实现解析」） | 硬件组给 e7f507f 写 VL53L1X 和 PMW3901 驱动、串口动作解析、降落、飞行中姿态修正 | 就是规格，仿真最接近真机；移植前先修 E1–E6 并补 LAND | 工作量和试飞风险最大；重复官方固件已有的功能；开源固件飞行中只积分陀螺 |

建议先花半天验证方案 1，这一步不需要 LicheePi，也不改固件：
1. 刷官方固件、插定点模块，上电输出里找「能力: 定高=… 光流=…」一行（`0x014307`，紧跟在 PMW3901 探测结果后面），再用 CLI `status` 看高度和 ToF。
2. 不装桨，在台架上摸清控制源仲裁：电脑连飞机热点，按内嵌网页的协议往 `ws://<飞机 IP>:81` 发 `{t:1, th:0, r:0, p:20, y:0}`，看「控制源切换」什么时候打印，实体遥控开着时网页输入是否被忽略（`0x012ce7`）。仲裁规则【未知】：如果实体遥控一开网页就发不进去，脚本飞行时就没有人工接管手段，要先想好怎么断电或罩网。
3. 手飞定点模式，在目标地面上 1.0 m 和 2.0 m 各悬停 30 s，看是否报「激光测距超出量程」或「光流失效」。
4. 定点悬停后由脚本发俯仰杆 x 持续 1.0 s 再回中，找出 0.3 m/s 对应的 x；偏航同理。D1 的 P1–P4 可以直接这样飞。

这只是测量手段，要成为正式飞行链路得先过 `docs/uart_action.md` 提到的安全评审。方案 1 不成立（2 m 定不住且不能降高度，或外部杆量不可靠）再退到方案 3。

仿真执行器对应真机的哪一部分：方案 1 下对应「官方固件的定高和定点 + 上位机里一张动作到杆量的表」；方案 3 下对应它自己。

## C. 未标定参数的灵敏度

| 排名 | 参数（合理区间） | 每步前进 | 每步转角 | 刹停滑行 | 悬停漂移 | 高度误差 | 油门饱和 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 光流有效高度上限 `flow_max_height`（1–3 m），相对巡航 2.0 m | ≤1.9 m 时 0.298 → 1.07–1.10 m | 不变 | 0.10 → 2.65–2.84 m | 0.9 → 100–108 cm/s | 不变 | 不变 |
| 2 | 动作生效延迟 `camera_latency`（0.05–0.20 s；3DGS 线现在是 0，Gazebo 线约 0.1） | 每 0.1 s −10.2%，首步 −18% | 每 0.1 s −12.1% | 不变 | 不变 | 不变 | 不变 |
| 3 | 偏航设定值相对固件死区（`yaw_rate_rps` 0.6，死区对应 0.524；真机偏航通道中点偏满程 1.5% 即越界） | 不变 | 0.53 时 30.3°，0.52 时 0° | 不变 | 不变 | 不变 | 不变 |
| 4 | 推力与质量（34–51 gf；92–110 g，裸机 80 g） | ≤1.3% | 首步 ≤2.2% | ≤2.1% | 不变 | z 偏差 0.022–0.046 m | 0.1–10.6%；最差组合到不了 READY |
| 5 | 悬停前馈 `hover_duty_estimate` 与对象失配（0.70–0.92） | 起飞成功的集里 ≤1.9% | ≤0.4% | −6% | 不变 | 偏高 12% 时飞丢（E2） | 不变 |
| 6 | 电池内阻与电量（0.04–0.10 Ω；20–100%） | 0.10 Ω 时 −4.4%（仅 2 集到 READY） | 不变 | 不变 | 不变 | 20% 电量 z 偏差 0.065 m；0.10 Ω 掉高中止 | 0.1–18.2% |
| 7 | ToF 帧率、噪声、量程（10–50 Hz；5–30 mm；1.3–4 m） | 不变 | 不变 | 不变 | 不变 | 10 Hz 时 ±0.26 m 振荡；20 Hz 时 0.063 m | 不变 |
| 8 | 陀螺零偏残差与光流噪声（0.03–0.5°/s；0.02–0.2 rad/s） | +1.6% | ≤0.1% | +6.1% / +15.9% | 0.86 → 5.33 / 2.68 cm/s；侧漂 0.97 → 3.93 cm/s | 不变 | 1.0 → 1.9% |

第二梯队：`drag_linear`（首步前进 +4.1 / −9.7%）、`inertia_diag`（首步转角 +2.7 / −5.3%）、`gyro_noise_std_flight` 5°/s（饱和 21.6%）。加计的四档全是 0.0%，因为飞行中固件不用加计。

说明：
- 数据来源：第 1、4–8 行用【扫描】（24 个参数两端 + 4 个组合），第 1–5、7、8 行有【实测】补充（`probe/out1` 19 个变体 × 16 集，`probe/takeoff` 10 个工况 × 4 集，`probe/gap` 10 个工况 × 8 集）。两边基线一致：0.211 / 0.298 m，31.7° / 34.4°。
- 扫描表里有 6 个参数的分数是 inf，拆开看都不是原语本身敏感：推力偏高、量程偏小、前馈偏高是起飞过冲触发 E1 和 E2，ToF 噪声是 E3，ToF 帧率是 E4，内阻偏大是推力不够。【实测】巡航高度改成 1.0 m 时，推力 45 gf 和 51 gf 都是 16/16 起飞成功，原语不变。
- 仿真里不存在、在真机上 1:1 进入原语的三个量：光流尺度（速度环闭在 `(flow + gyro)·h` 上，`executor.c:71`）、高度估计的比例误差（同一行的 `h`）、谁来计 1.0 s（上位机定时抖动 10–20 ms 对应 1–2%）。
- 序列依赖比参数依赖大：同一个 FORWARD 标签，接在 SAFE 后是 0.211 m，接在 FORWARD 后是 0.298 m，接在 YAW 后是 0.207 m。真机加速度不同时这个比例会变，所以 D1 的单步和连续序列要分开飞。
- 第 2 行的测法：每个决策周期的前 gap 秒执行器已回悬停，动作只在剩余时间有效。gap 0.05 / 0.10 / 0.20 / 0.30 s 时稳态每步前进 0.288 / 0.272 / 0.240 / 0.207 m，每步转角 32.5° / 30.2° / 25.6° / 22.0°；hold 0.5、gap 0.10 时是 0.123 m（−20.1%）和 13.0°（−24.5%）。

## D. 重新仿真的可执行方案

### D1 真机数据采集清单

目录约定（全部在 `$D/real-drone/` 下，现有 README 要重写）：

```text
hardware.json                         固件版本、扩展模块、电池、桨、巡航高度
bench/  mass.csv  cg.json  thrust_curve.csv  motor_tau.csv  inertia_bifilar.csv  battery_discharge.csv
flight/<日期>-<名字>/  fc_log.csv  actions.jsonl  overhead.mp4  overhead.json  track.csv
camera/ calib/*.png  intrinsics.json  mount.json  latency.csv  rolling_shutter.json  exposure/
sensors/ tof_static.csv  flow_translate.csv
primitives.json                       原语统计汇总，拟合和验收的输入
```

| 编号 | 测什么、怎么测 | 次数 | 文件与列 | 替换的键 |
| --- | --- | --- | --- | --- |
| B1 | 称重：0.1 g 电子秤，裸机含电池、LicheeRV Nano、相机、定点模块、护罩、线和支架、整机起飞状态 | 各 3 次 | `bench/mass.csv`：`item,mass_g,repeat` | `mass_bare`、`mass_flight`、`guard_mass` |
| B2 | 重心：悬线法，两个悬挂点各拍一张带铅垂线的照片，交点即重心 | 2 点 × 2 | `bench/cg.json`：`{x_mm,y_mm,z_mm}` | `cg_offset`（偏移 > 5 mm 再接进 SITL） |
| B3 | 推力曲线：机体用支杆架高 15 cm 以上固定在配重上放电子秤，读减重；四个电机同占空比 0.1 → 1.0，步进 0.1，每档 3 s，万用表并在电池端读电压；满电、3.8 V、3.6 V 三组。发占空比要刷开源固件，用 MAVLink `SET_ACTUATOR_CONTROL_TARGET`（`fw/mavlink.ino:195-207`，AUTO 模式）；官方 CLI 只有 `mfr/mfl/mrr/mrl` 单电机测试 | 3 组 × 10 档 × 2 遍 | `bench/thrust_curve.csv`：`trial,duty,v_bus,thrust_gf_total` | `motor_thrust_full_duty`、`thrust_duty_exponent`、`thrust_model`、`hover_duty_estimate` |
| B4 | 电机时间常数：手机贴近录音 48 kHz，占空比 0.4 ↔ 0.7 阶跃，桨通过频率 = 2 × 转速，取到 63% 的时间 | 升、降各 5 次 | `bench/motor_step_*.wav`、`bench/motor_tau.csv`：`trial,dir,tau_s` | `motor_tau_up`、`motor_tau_down` |
| B5 | 惯量：双线摆，两根等长细线吊起，绕竖直轴小角度扭摆，计 20 个周期，`I = m·g·d²·T²/(16π²·L)`；x、y、z 三个轴分别朝上 | 每轴 3 次 | `bench/inertia_bifilar.csv`：`axis,L_m,d_m,n,t_s` | `inertia_diag` |
| B6 | 电池：定点悬停从满电到固件低电告警（开源固件 L2 是 3.3 V，`json:30`），记录电压（官方日志的 `batteryVoltage`，或万用表录像）；内阻用 B3 相邻两档的压差估 | 2 块电池各 1 次 | `bench/battery_discharge.csv`：`t_s,v_bus,thrust_cmd` | `battery_r_internal`、`battery_ocv_curve`、`battery_capacity`、`endurance` |
| F1 | IMU：静置 60 s 记 MAVLink 遥测里 10 Hz 的原始 IMU（开源固件不解锁不写日志，`fw/log.ino:47`）；定点悬停后 `log dump` 取 `rates.*`。日志里是滤波后的角速度，所以用「SITL 悬停日志同一列的标准差等于真机」去定 `gyro_noise_std_flight` | 各 3 次 | `flight/…/fc_log.csv`（`log dump` 原样输出）、`flight/…/imu_static.csv` | `gyro_noise_std_static`、`gyro_noise_std_flight`、`gyro_bias_residual` |
| F2 | 内环阶跃（只在方案 2、3 需要）：STAB 手动，俯仰杆 ±0.3 保持 1 s，偏航杆 ±0.3 保持 1 s | 各 5 次 | 同上 | 拟合 `inertia_diag`、`motor_tau_*`、`torque_thrust_ratio_km` |
| C1 | 相机内参和畸变：9×6 棋盘（格 25 mm），不同姿态 25 张，`cv2.calibrateCamera`（工具环境 cv2 5.0.0 已确认可用） | 1 次，RMS > 0.5 px 重拍 | `camera/calib/*.png`、`camera/intrinsics.json`：`{width,height,fx,fy,cx,cy,dist[5],rms_px}` | `camera_model`、`camera_fov`、D4 第 1b 项的裁剪框 |
| C2 | 安装位姿与入画：卡尺量镜头相对机体中心的位置；机体放水平拍远处水平线，`pitch = atan((cy − v_地平线)/fy)`；桨转起来拍一张看护罩和桨是否入画 | 1 次 | `camera/mount.json`、`camera/props_on.png` | `camera_mount` |
| C3 | 曝光和增益范围：约 50、300、2000 lux 三种照度各录 10 s，记录曝光时间和增益 | 3 段 | `camera/exposure/*.mp4`、`camera/exposure.json` | 训练增强的模糊和噪声幅度 |
| C4 | 端到端延迟：LicheePi 用 GPIO 点亮 LED 并记单调时钟；记录首个看见亮灯的帧到达推理线程的时刻、推理结束时刻、动作帧写出时刻 | 100 次 | `camera/latency.csv`：`trial,t_led,t_frame,t_infer_done,t_cmd_out` | `camera_latency`（均值和 p95） |
| C5 | 卷帘读出：拍 1 kHz 方波驱动的 LED，数明暗条纹得行时间 | 1 次 | `camera/rolling_shutter.json`：`{line_time_us,readout_ms}` | 训练增强的剪切幅度 |
| S1 | ToF：机体架在 0.5、1.0、1.5、2.0、2.5 m（卷尺），对着目标地面各 30 s；室内灯和窗边各一组；读官方日志的 `measAlt` 列（是否要求解锁没确认），或 CLI `status` 逐次读 | 2 组 × 5 高度 | `sensors/tof_static.csv`：`t_s,range_mm,status,true_m,light` | `tof_noise_std`、`tof_range_max`、`tof_rate` |
| S2 | 光流：先按固件提示做 `ft` 标 `FLOW_SCALE`（0.20 m 高、单向直推 10 cm，`0x011890`、`0x013655`）；再手持在 1.0、1.5、2.0 m 沿卷尺平移 1 m，读 `posX` | 每高度 5 次 | `sensors/flow_translate.csv`：`height_m,true_m,posX_m,squal_mean,valid_ratio` | `flow_max_height`、`flow_noise_std`、新键 `flow_scale_err` |
| P1 | FORWARD：悬停稳定后发 FORWARD 1.0 s，再 SAFE 3 s；另做连续 5 个 FORWARD 的序列 | 单步 20 次（满电、半电各 10），序列 5 组 | `flight/…/actions.jsonl`、`overhead.mp4`、`track.csv`：`t_s,x_m,y_m,yaw_deg` | `executor.forward_speed_mps`、`vel_kp`、`vel_ki`、`drag_linear` |
| P2 | YAW：YAW_LEFT、YAW_RIGHT 各 1.0 s 后 SAFE 2 s；另做连续 5 个的序列 | 各 20 次，序列各 3 组 | 同上 | `executor.yaw_rate_rps` |
| P3 | SAFE：定点悬停 30 s | 5 次 | 同上 | `gyro_bias_residual`、`flow_noise_std` |
| P4 | 交替：FORWARD → YAW → FORWARD → SAFE | 5 组 | 同上 | 核对滑行 0.127 m 和转后首步 0.207 m |

真值怎么取：手机固定在 2.5 m 以上俯拍，1080p 60 fps；地面四角各贴一个 AprilTag 36h11（边长 150 mm）定单应，机体顶面贴一个 40 mm 的；`cv2.aruco` 在工具环境里可用（已确认）。画面覆盖 4 m 宽时约 2 mm/px。每次起飞前让一颗 LED 入镜亮一次，用来对齐 `actions.jsonl` 的时钟。

飞行日志现在能录什么：
- 开源 e7f507f：`log dump` 14 列（`t`、`rates.x/y/z`、`ratesTarget.x/y/z`、`attitude.x/y/z`、`attitudeTarget.x/y/z`、`thrustTarget`，`fw/log.ino:21-36`），100 Hz、最近 8 s、只在解锁时记（`log.ino:9-11,46-50`，`fw/board_config.h:131`）；环形缓冲按下标打印，回卷后要按 `t` 排序（`log.ino:70-77`）。MAVLink 遥测 10 Hz：姿态四元数、角速度、4 个电机输出、原始 IMU（`fw/mavlink.ino:47-67`）。缺电池电压、杆量、任何高度 / 速度 / 位置量。
- 官方固件：47 个列名（B2），够用；采样率和缓冲时长【未知】，上机执行一次 `log` 看表头和行数就能确定。

`primitives.json` 的键名与扫描脚本的指标名一致，拟合时直接比：

```json
{"schema": "uavrgb.real_primitives.v1", "firmware": "official-20261102", "mode": "POSHOLD",
 "height_m": 1.0, "hold_s": 1.0, "sticks": {"forward_pitch": 0.25, "yaw": 0.20},
 "fwd_first_m": {"n": 20, "mean": 0.0, "std": 0.0}, "fwd_ss_m": {"n": 25, "mean": 0.0, "std": 0.0},
 "brake_m": {}, "stop_s": {}, "yaw_first_deg": {}, "yaw_ss_deg": {}, "yaw_after_fwd_slide_m": {},
 "hold_drift_cms": {}, "z_dev_m": {}, "hover_duty": {}, "latency_s": {"mean": 0.0, "p95": 0.0}}
```

### D2 参数辨识

直接代入、不拟合：B1 → 质量三项；B5 → `inertia_diag`；B3 → `motor_thrust_full_duty` 和 `thrust_duty_exponent`（对 `log T = log T_full + k·log(d·V/3.7)` 做线性回归，numpy 即可，工具环境没有 scipy）；B4 → 电机时间常数；B6 → 电池三项；F1 静置段 → `gyro_noise_std_static` 和 `gyro_bias_residual`；S1、S2 → ToF 和光流的量程、帧率、噪声；C1、C2、C4 → 相机三项；实测悬停占空比 → `hover_duty_estimate`（或按 E2 改成自动算）。

要拟合的两层：
- 原语层（三种方案都要做）。变量：`executor.forward_speed_mps`、`executor.vel_kp`、`executor.vel_ki`、`executor.yaw_rate_rps`、`drag_linear.xy`、动作延迟。目标：`J = Σ_k ((仿真_k − 真机_k) / 容差_k)²`，k 取 `fwd_first_m`、`fwd_ss_m`、`brake_m`、`stop_s`、`yaw_first_deg`、`yaw_ss_deg`、`yaw_after_fwd_slide_m`、`hold_drift_cms`，容差取 D3 的门限。【实测】10 个变体 × 8 集共 5.7 s，坐标下降几十轮在几分钟内跑完。
- 内环层（只在方案 2、3）。把 F2 日志里的 `attitudeTarget` 和 `thrustTarget` 反算成杆量（`fw/control.ino:157,165`），在 SITL 里用原始杆量回放；目标是 `attitude` 与 `rates` 的加权均方差（权重 1° 和 10°/s）；变量是 `inertia_diag` 的三个比例、`motor_tau_up`、`motor_tau_down`、`torque_thrust_ratio_km`。

| 现有入口 | 现状 | 怎么用，缺多少代码 |
| --- | --- | --- |
| `cf_sitl_set_sticks_raw`（`sitl/include/cf_sitl.h:106`，`sitl/src/cf_sitl.cpp:120-125`）和 selftest 场景 3.5（`sitl/selftest/scenario_sticks.cpp`） | 库里有原始杆量旁路，只有 selftest 调；`cf_sitl_proc` 的协议只有 reset / takeoff / act / step / state / quit（`sitl/proc/README.md:23-28`） | 给 proc 加 `{"cmd":"sticks","roll":…,"pitch":…,"yaw":…,"throttle":…}`，约 12 行，协议测试加一例约 10 行；不改库行为，不用升库版本 |
| `$D/tmp/wt-sim-dyn/sim/tools/cf_sitl_sweep.py`（469 行，未入库） | 固定动作脚本 + 原语指标，键名就是 `fwd_first_m` 这一套 | 先入库；新写 `sim/tools/cf_sitl_fit.py`，直接 import 它的 `run_episode` 和 `aggregate`，约 130 行 |
| `sim/tools/cf_sil.py:583-683` 的 `--flight-log` | 自述「只做接口校验/摘要，不做系统辨识」（`cf_sil.py:581`）；要的是 PX4 ULog 或 `t,qw,…` 列，和 CF-Drone 的 `log dump` 对不上 | 不扩展。新写 `sim/tools/real_primitives.py`：解析 `log dump`（约 30 行）+ 俯拍轨迹提取（约 80 行）+ 原语统计出 `primitives.json`（约 50 行） |
| `sim/tools/cf_blackbox.py` + `sim/dynamics/cf_blackbox_v1.json` | 2D 黑盒，5 个占位参数；SAFE 只有被动滑行，没有主动刹停（`cf_blackbox.py:5`） | 现在不动。以后若要去掉固件在环，可以把它拟合到真机原语 |
| `$D/real-drone/README.md` | 要求 PX4 `flight.ulg`，本机固件没有 ULog | 按 D1 的目录重写（文档） |

内环回放脚本（方案 2、3）约 80 行。合计新增 Python 约 370 行、C++ 约 22 行。

### D3 置 `calibrated=true` 的验收门

| 门 | 量 | 门限 | 理由 |
| --- | --- | --- | --- |
| P-1 | FORWARD 首步、稳态每步位移的均值误差 | 各 ≤10%（2–3 cm） | 教师的硬余量 0.08 m、栅格 0.05 m（`sim/tools/teacher_kino.py:54,60`）；8 步视野 2.4 m（`:63`）累计误差 0.24 m，小于软余量带 0.5 m（`:55`） |
| P-2 | YAW 首步、稳态每步转角的均值误差 | 各 ≤3° | 教师航向剪枝格 10°（`teacher_kino.py:62`），连转 3 步累计 9° 不跨格 |
| P-3 | SAFE 刹停滑行距离、停住时间 | ≤0.05 m、≤0.3 s | 教师急停检查按滑行 0.118 m 建模（`teacher_kino.py:48,53`），误差要小于硬余量 |
| P-4 | 分散度 | 真机 20 次样本至少 90% 落在仿真（含域随机化）的 5–95% 区间里 | 仿真现在每步转角的集间标准差只有 0.13°，不做随机化这条必然不过；超出训练分布的位移没有标签 |
| P-5 | 定点悬停 30 s | 水平漂移 ≤0.3 m，高度 RMS 误差 ≤0.10 m | 仿真 30 s 漂 0.22 m；0.10 m 是执行器 READY 的高度容差，对 2 m 外的物体是画面里 3° 的偏移 |
| P-6 | 悬停占空比（满电、半电） | 与仿真差 ≤0.03，饱和比例 <5% | 上限 0.90、基线 0.81，余量 0.09 的三分之一；对应 11-05 评审 G2 的 ±5% |
| I-1 | 打杆阶跃的俯仰上升时间、偏航角速度上升时间、稳态倾角（方案 2、3） | 时间误差 ≤30%，倾角误差 ≤2° | 内环 0.3 s 占 1 s 窗口的 30%，再错 30% 折合位移约 3 cm，与 P-1 同量级；对应 11-05 评审 G1 |
| C-1 | 相机 | 标定 RMS ≤0.5 px；裁剪后 HFOV 60° ± 1°；延迟均值已写入参数，p95 − 均值 ≤50 ms | 1° 是 160 px 输入上的 2.7 px；50 ms 对应每步位移 5%（C 第 2 行） |
| R-1 | 回放一致性 | 真机 10 条动作序列在标定后的仿真里回放，终点误差 ≤航程的 10%，航向误差 ≤15° | 累积误差的总检查；7 m 航程对应 0.7 m，小于 1.2 m 的柱间缝（`teacher_kino.py:56`） |

全部通过后把 `json:3` 改成 `true`，同时把用到的键的 `status` 从 `assumed` 改成 `measured`，`source` 写 `$D/real-drone/` 下的文件名。

### D4 拿到真机数据之前现在就能做的事

| 排名 | 做什么 | 影响 / 工作量 | 改哪里 | 已采数据 |
| --- | --- | --- | --- | --- |
| 1a | 训练开图像增强 `--augment photo,shift`。gamma 0.8–1.25、对比度 0.8–1.2、亮度 ±0.1、平移 ±4 px 已实现（`src/trainer_dev.cpp:72-98`，`src/augment_dev.cu:26-32`），两条训练流水线都没传这个参数（`tools/e1_pipeline.py:231-233`，`tools/dagger_gz_pipeline.py:468-470`） | 高 / 各 2 行；与 `--soft-targets` 互斥（`trainer_dev.cpp:130`） | 两个流水线的 CFG 和命令行 | 不作废（只动训练） |
| 1b | 真机图像按仿真口径出图：去畸变 + 旋转校平 + 中心裁 60°×46.8° + 面积缩放到 320×240，再走现有的 bilinear 到 160×120（`tools/preprocess.py:638`，`sim/tools/student.py:117`）。【推算】720p 竖向只剩约 ±2° 的俯仰补偿余量 | 高 / 新文件约 40 行（`cv2.remap`），现在就能写，用合成畸变图自测 | 新 `tools/crop_to_sim.py`；仿真不动，`camera_profile` 不变（`sim/tools/scene_gen.py:459`） | 不作废 |
| 2 | 定巡航高度。2.0 m 离假设的 ToF 和光流上限只有 0.5 m，官方固件里 ToF 超量程会同时让定高和光流失效（A 第 9、10 行）。【实测】仿真在 1.0 m 的原语与 2.0 m 相同（0.211 / 0.299 m，31.7° / 34.3°），ToF 量程 1.3 m、光流上限 1.5 m 时照常工作 | 高 / 是决策；若改高度，`sim/tools/cf_sitl_exec.py:48`、场景 spawn 的 z、3DGS 碰撞切片都要跟着改 | 见左 | 改高度则全部重采（视角变了），所以要在放量之前定 |
| 3 | 动作生效延迟做成显式参数，两条线同一实现：每步先空 latency 秒再下发，按集抽 U(0.05, 0.20) s。教师侧已有延迟采样（`sim/tools/teacher_belief.py:60`），环境侧没有 | 高 / 3DGS 线约 8 行（`sim/3dgs/dynamics.py:247-258`），Gazebo 线把已有的约 0.1 s 记进 manifest 约 5 行 | 见左；新口径键 `action_latency` | 3DGS 新数据换口径；Gazebo 旧数据（约 0.1 s）可留 |
| 4 | 动力学按集随机化（下表） | 高 / 3DGS 线约 35 行：每集写一份参数副本给 `cf_sitl_proc --params`，profile 改成「基准值 sha + 随机化规格 sha」，抽样值写进 episode manifest；Gazebo 线约 30 行：插件的参数路径来自世界 SDF（`sitl/gz/CfSitlSystem.cc:45`） | `sim/3dgs/dynamics.py`、`sim/tools/cf_sitl_exec.py:129-131`、`sim/tools/scene_gen.py` | 新 profile，不与旧数据混包（`tools/preprocess.py:53`） |
| 5 | E1–E6 的修复攒成一次库版本升级 | 中 / 代码约 30 行，测试约 60 行 | `sim/cf_sitl` | 正常集的轨迹不变，但库版本和新增键都进 profile（lib31 → lib32），与旧数据混包要 `--allow-mixed-dynamics` |
| 6 | 执行器对齐选定的软件栈（B3） | 中 / 方案 1：0 行，等 D2 拟合；方案 2：约 20 行；方案 3：先做第 5 项并补 LAND | 见 B3 | 改执行器目标值会换 profile |
| 7 | SITL 增加经 `channels[]` → `normalizeRC()` 的杆量入口（11 bit 量化、50–150 Hz 保持） | 中 / 约 15 行 + 一条测试：停发杆量 1.1 s 后固件应进 `descend()`，现在永远不会 | `sitl/fw/fw_sitl.cpp` | 换库版本 |
| 8 | 机体外观 | 低（机体不入画）/ 并发会话已在做 | `sim/tools/e1_body.py`、SDF | 模板 sha 变，只做溯源，不是混包闸门（`preprocess.py:53-59` 的口径键里没有它） |
| 不做 | 在仿真器里加畸变、卷帘、运动模糊 | 低 / 高：要重渲全部数据 | 无 | 用 1a + 1b 代替 |

物理参数标定后原语变化 ≤2%（C 节），所以只改物理参数时旧数据在行为上仍可用；改执行器目标值、窗口延迟、相机口径或巡航高度才会让旧数据作废。

| 随机化参数 | 范围 | 粒度 | 依据 |
| --- | --- | --- | --- |
| `executor.forward_speed_mps` | ×U(0.8, 1.2) | 每集 | 光流尺度、高度估计、官方定点杆量映射的不确定度 |
| `executor.yaw_rate_rps` | ×U(0.92, 1.25) | 每集 | 下限不低于 0.55 rad/s（死区 0.524） |
| 动作延迟 | U(0.05, 0.20) s，步间抖动 ±0.02 s | 每集 + 每步 | `json:64` 的区间 |
| `drag_linear` | ×U(0.5, 2) | 每集 | `json:26` |
| `gyro_bias_residual` | 1σ 0.25°/s | 每集 | `json:42` 的区间 0.03–0.5 |
| `flow_noise_std` | U(0.02, 0.2) rad/s | 每集 | `json:57` |
| `sim.battery_soc_init` | U(0.3, 1.0) | 每集 | 一块电池的使用区间 |
| `mass_flight`、`motor_thrust_full_duty` | U(0.092, 0.110) kg、×U(0.9, 1.2) | 每集 | `json:10,19`；前提是 E2 修完，否则起飞失败 |
| `tof_noise_std` | U(0.005, 0.02) m | 每集 | 前提是 E3 修完 |
| 相机俯仰、滚转、HFOV | ±3°、±2°、±2° | 每集 | 安装误差；3DGS 侧是 `camera_mount.pitch_deg` 与 `fov_deg`，Gazebo 侧在 `scene_gen` 写相机 pose |
| 亮度、对比度、gamma、噪声、模糊 | 见 1a | 每样本（训练时） | 不进仿真器，不用重采 |

## E. 读代码时发现的仿真侧问题（按严重程度）

### E1【高】FAILSAFE 开环保持悬停油门，高度无界上升

```151:152:/root/claude-code-backup-20261028/code/uav_rgb/sim/cf_sitl/exec/executor.c
	const int flying = e->state == CF_EXEC_TAKEOFF || e->state == CF_EXEC_READY || e->state == CF_EXEC_ACTING;
	if (flying && e->airborne && in->t_s - e->last_tof_valid_s > p->tof_timeout_s) e->state = CF_EXEC_FAILSAFE;
```

```197:200:/root/claude-code-backup-20261028/code/uav_rgb/sim/cf_sitl/exec/executor.c
	case CF_EXEC_FAILSAFE:
	default:
		s.throttle = e->hover_stick; // 水平、悬停前馈；不再闭环
		break;
```

- 【实测】`probe/takeoff/summary.txt`，4 集 × 25 s。推力 51 gf、前馈不改：3.9–4.0 s 进 FAILSAFE，25 s 时 z = 60.5–63.3 m，仍以 3.0–3.2 m/s 上升。ToF 量程 1.3 m：3.0 s 进 FAILSAFE，z = 20.3–21.7 m。裸机 80 g：z 约 56 m。
- 影响：两条采集线都把 FAILSAFE 当错误（`sim/tools/cf_sitl_exec.py:382-383`，`sim/3dgs/dynamics.py:272-273`），数据不会被污染。但 `sitl/README.md:88-92` 写明这段代码要原样移植进固件，真机 ToF 丢 0.5 s 就会顶着悬停油门不放。
- 最小修复（约 5 行 + 一个新参数）：FAILSAFE 里线性收油，写法同固件 `descend()`（`fw/safety.ino:81`）。`failsafe_descend_s` 建议 2 s：【推算】按油门增益 15 m/s²/杆（`json:102`）手算，推力高 34% 的工况在 2 s 斜坡下还会再升约 2 m 后回落，照搬固件的 10 s 会再升十几米。

```c
	case CF_EXEC_FAILSAFE:
	default:
		e->hover_stick = fmaxf(0.0f, e->hover_stick - p->dt_s / p->failsafe_descend_s);
		s.throttle = e->hover_stick;
		break;
```

- 验证：新 CTest（约 30 行）：`tof_range_max = 1.3`，起飞到 2.0 m，走 25 s，断言全程 z < 3 m 且末态 z < 0.05 m。现在末态是 21 m（红）。

### E2【高】起飞前馈是手填常数，与对象参数不联动

- 读码：`executor.c:24` 用 `p->hover_duty` 换算初始悬停杆，`plant.h:94` 从 `hover_duty_estimate` 读，`json:37` 的 0.824 是离线脚本 `research-hw/calc_params.py` 算的。
- 【实测】基线起飞过冲到 2.21–2.22 m，离量程上限 2.5 m 剩 0.28 m。推力 45 gf（+18%）16/16 起飞成功；48 gf（+26%）4.3 s 进 FAILSAFE；裸机 80 g 进 FAILSAFE；满电时最高 2.51 m，ToF 短暂失效；前馈取 0.92 时 3/4 飞丢。前馈跟着改之后（51 gf 配 0.677，80 g 配 0.701）都是 16/16，原语不变。【扫描】前馈取 0.70 时 32/32 成功，READY 9.15 s：偏低无害，偏高致命。
- 影响：真机推力或称重结果第一次填进 `json` 时，只要比假设强 26% 以上，或者按裸机配置飞，仿真直接起飞失败；D4 的质量、推力、电量随机化也做不了。
- 最小修复（约 8 行，删掉一个手填量）：`plant.h` 加载时自己算，公式与 `json:139-143` 的 hover_check 相同：单电机悬停推力 `m·g/4` → 功率（`Battery::step` 同一公式）→ 母线电压（二次方程大根）→ `d = (T/t_full)^(1/k)·v_ref/V_bus`。`hover_duty_estimate` 降为只读对照项。
- 验证：`python probe/takeoff_trace.py <out>` 里的 `thrust51_ffstale`、`bare80_ffstale` 现在 4/4 进 FAILSAFE，改后应全部 READY；加一条 CTest：质量 0.080 起飞到 READY。

### E3【中】READY 判据的垂直速度容差小于 ToF 噪声带来的速度估计噪声

- 读码：`executor.c:172-173` 要求垂直速度估计的绝对值小于 `ready_vz_tol_mps`（0.1 m/s）连续 1 s；速度估计每个 ToF 样本按 `β·残差/Ts` 更新（`executor.c:62`），β/Ts = 0.08 / 0.033 = 2.4 1/s。`json:104` 自己写了 1 cm 噪声对应约 0.04 m/s 的速度噪声，按比例 30 mm 是 0.12 m/s，已经超过门限。
- 【实测】ToF 噪声 10 mm：16/16。20 mm：13/16，READY 12.0 ± 3.4 s。30 mm：0/16，一直停在 TAKEOFF，高度 1.997 m。【扫描】50 Hz、10 mm：31/32。把容差放到 0.3：30 mm 时 16/16，READY 7.24 s，原语与基线相同。
- 修复：`json` 的 `executor.ready_vz_tol_mps` 0.1 → 0.3（1 个值；value 变了 profile 的 sha 跟着变，和 D4 第 5 项一起做）。
- 验证：`python probe/probe.py <out> 16 tof_noise30mm` 从 0/16 变成 16/16。

### E4【中】高度环增益按 ToF 30 Hz 调，10 Hz 时进极限环

- 【实测】`tof_rate` 10：从不 READY，最后 8 s 高度在 1.66–2.18 m 之间振荡。20：16/16，z 偏差 0.063 m（基线 0.041）。
- 读码：观测器的 α 0.4、β 0.08 是固定值（`executor.c:59-62`），`json:104` 写明按 30 Hz 推出 ωn 8.5 rad/s；样本间隔 0.1 s 时同一公式给出 2.8 rad/s。
- 真机：官方固件启动时打印测距周期和 budget（`0x012666`），数值【未知】。
- 最小处理：这是控制设计的约束，不是一行能修的。先在 `plant.h` 加载时拒绝 `tof_rate < 20`（2 行），防止静默跑在不稳定区；真机帧率测到之后再按实际值重调 `height_kp`、`height_kd` 和 α、β。
- 验证：`sitl/tests/test_params_file.cpp` 加一例：`tof_rate = 10` 应加载失败，现在加载成功。

### E5【中】`yaw_deadzone` 读进来没人用，偏航设定值掉进固件死区时不报错

- 读码：`plant.h:97` 读，`sitl/exec/executor.h:20` 声明，`executor.c:138-141` 的 `yaw_stick()` 不检查；固件 `fw/control.ino:152` 把绝对值小于 0.1 的偏航杆置零。
- 【实测】`probe/gap/table.md`，8 集：`yaw_rate_rps` 0.60 时每步 34.3°，0.53 时 30.3°，0.52 时 −0.05°。
- 修复（2 行）：`plant.h:104-105` 已有倾角的加载期校验，旁边加一条。

```cpp
		if (e.yaw_rate_rps <= e.yaw_deadzone * e.yaw_rate_per_stick_dps * (float)(M_PI / 180))
			throw std::runtime_error("executor.yaw_rate_rps 落在固件偏航死区内（control.ino:152）");
```

- 验证：`test_params_file.cpp` 加一例：`yaw_rate_rps = 0.5` 应加载失败，现在加载成功。

### E6【中】光流失效时 FORWARD 以固定倾角开环加速，执行器不降级也不报错

```69:75:/root/claude-code-backup-20261028/code/uav_rgb/sim/cf_sitl/exec/executor.c
		e->flow_ok = in->flow_valid && e->height_valid;
		if (e->flow_ok) {
			e->flow_vel[0] = (in->flow[0] + in->gyro[1]) * e->height_est_m;
			e->flow_vel[1] = (in->flow[1] - in->gyro[0]) * e->height_est_m;
		} else {
			e->flow_vel[0] = e->flow_vel[1] = 0.0f;
		}
```

- 读码：无效帧把速度当 0，积分和位置保持冻结（`executor.c:115,130`），READY / ACTING 分支不看 `flow_ok`（`executor.c:180-196`）。FORWARD 于是一直按 0.3 m/s 的误差给 3.8° 倾角。proc 的 state 里没有光流有效位（`sitl/proc/README.md:31`），采集器看不到。
- 【实测】`flow_max_height` 1.9、巡航 2.0 m：16/16 READY、0 集中止，稳态每步前进 1.070 ± 0.232 m，刹停 2.65 m，悬停漂移 100 cm/s。官方固件在同样情况下会提示并降级（`0x0130f5`）。
- 影响：现在的名义参数下不会发生（上限 2.5 m）；一旦按真机把上限调低或做高度随机化，这种集会带着 3.6 倍的每步位移进数据集和评测，没有任何报警。
- 最小修复（约 4 行）：仿照 ToF 的超时，记 `last_flow_ok_s`，飞行中光流连续无效超过 `tof_timeout_s` 就进 FAILSAFE（配合 E1 的收油）。
- 验证：`python probe/probe.py <out> 16 flow_maxh1.9` 现在 16/16 READY、每步 1.07 m；改后应 0/16 READY（起飞爬过 1.9 m 后进 FAILSAFE）。

### E7【中】两条仿真线的「一步」口径不同，延迟只有教师知道

- 读码：Gazebo 线默认 `--window-from observation`（`sim/tools/collect.py:1284-1285`），截止时刻 = 观测位姿戳 + hold（`sim/tools/cf_sitl_exec.py:496`），观测帧比下发早约一帧相机，位姿戳间隔 0.900 s；3DGS 线渲染与下发同刻、走满窗口（`sim/3dgs/dynamics.py:258`）；`camera_latency` 只在 `teacher_kino.py:82,92-94` 用，3DGS 的教师模型把它设成 0（`teacher_kino.py:101-102`）。
- 【实测】空档每 0.1 s，稳态前进 −10.2%、转角 −12.1%（hold 0.5 时 −20.1% / −24.5%）。
- 修复：D4 第 3 项。验证：3DGS 线延迟设 0.1 s、hold 0.5 时，稳态每步前进应从 0.154 m 降到 0.123 m（本次实测值）。

### E8【中】真机数据没有入口

- 证据：`$D/real-drone/README.md` 要 PX4 `flight.ulg`；`sim/tools/cf_sil.py:581`「只做接口校验/摘要，不做系统辨识」；`cf_sitl_proc` 协议表没有原始杆量命令，`cf_sitl_set_sticks_raw` 只有 selftest 调用。
- 修复：D2 表里的三件（proc 加 `sticks` 命令、`real_primitives.py`、`cf_sitl_fit.py`）。验证：`sitl/tests/test_proc_protocol.cpp` 加一例 `sticks` 往返。

### E9【低】`uart_action.py` 走不了 CRSF（工具缺口，不是仿真 bug）

帧结构只有 const / seq / action / heartbeat / ttl / crc 六种字段（`docs/uart_action.md:112-121`），表达不了 CRSF 的 16 通道 11 bit 打包；波特率只接受 termios 常量（`tools/uart_action.py:108-110`），420000 不在其中。方案 1 走 CRSF 时要另写一个通道帧编码器（约 60 行，CRC8 多项式 0xD5，与 `fw/rc.ino` 的解析器对拍），并把飞控 `RC_BAUD` 设成 460800（`fw/parameters.ino:179`），不必写 termios2。验证：编码器输出喂给 `fw/rc.ino` 的 CRSF 解析函数（SITL 里已经原样编译了这个文件），通道值往返一致。

### E10【低】`e1_mini.json` 一处说明引用错了字符串

`json:50` 的 basis 写「超量程时官方固件回退到气压计（bin 0x01259a）」。`0x01259a` 是「VL53L1X: 初始化失败，禁用激光，回退气压计」；超量程对应 `0x0101ac`「激光测距超出量程，无法维持高度」。改一行说明文字；profile 只哈希 value（`sim/tools/cf_sitl_exec.py:121-126`），不受影响。

## 没读 / 没验证的内容

做了什么：
- 读：两份协作规范、`CLAUDE.md`、`README.md`、`progress.md` 头部、10:10 审查 worklog、CF-Drone 全部 `.ino` 和 `board_config.h`、`sim/cf_sitl` 的库、固件桩、执行器、proc、gz 插件、selftest、`e1_mini.json` 全文、`cf_sitl_exec.py`、`sim/3dgs/dynamics.py`、相机和模板相关代码、`collect.py` 的动作窗口段、`teacher_kino.py` 头部、`cf_sil.py` 的日志入口、`cf_blackbox.py` 头部、训练增强代码、`docs/uart_action.md`、官方 bin 字符串全文检索、15 份官方文档、11-05 到 11-10 的 6 份历史审查。
- 实测（全部写在草稿目录）：selftest 1 次（33/33 PASS）；`probe/out1` 19 个变体 × 16 集；`probe/takeoff` 10 个工况 × 4 集；`probe/gap` 10 个工况 × 8 集。
- 复核：初稿写完后把行号、偏移、数字对着源文件又核了一遍，改正了变体个数、光流失效原因个数、`ft` 的推距、「能力」一行的打印位置等几处。

没做什么（含原因）：
- 没改任何仓库文件，没提交，没重建 SITL：任务是只读。D 节和 E 节的修复都是建议，没有编译验证；E1 的斜坡时间和峰值高度是手算。
- 没启动 Gazebo，没跑 `tools/check.sh`：规则禁止 gz；没有改代码，跑测试没有新信息。
- 没反汇编官方 bin，只看了字符串。官方固件的默认参数值、定点模式的杆量语义、控制源仲裁规则、MAVLink 接受的消息集都不知道。
- 没读：`sim/README.md`、`docs/HANDOFF.md`、`docs/PLAN_2026-11-07_demo_and_paper.md`、`ALVLN/docs/08_CF_DRONE_SAFETY.md`（`docs/uart_action.md` 引用的安全文档）、`sim/tools/eval_closed_loop.py` 和 `record_views.py`（并发会话在改，与本题无关）、`rl_teacher/`、`teacher/`、`src/` 里除增强和训练入口之外的部分、`research-hw/` 下的 `flix_simulator.cpp`、`E1_Mini_BOM.xlsx`、`lrvnano.dts`、`maixcam_maixcam.md`（BOM 是经 `e1_mini.json` 的引用间接读的）。
- hold 0.5 的扫描目录是空的，hold 0.5 的数只有我的基线和空档两组。

验证到哪一步：
- 已核实（读码或实测）：A 表里带文件行号的仿真侧条目、B2 的全部偏移和原文、C 的数字、E1–E8 的现象。
- 推算，没有实物核对：光流每像素 44 mm（按 `json:59` 的 42° 视场和 35 像素）；720p 下裁剪框 865×649 px（按标称 81° 视场，以 C1 标定为准）；护罩进画面边角（按两个 assumed 的安装高度）。
- 真机侧全部未知：没有任何实测数据，本报告不能替代 D1。主仓库工作树有并发会话的未提交改动，`uav_rgb_e1.sdf` 的行号以 c85a1ff 为准。
