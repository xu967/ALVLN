// selftest_config.h：cf_sitl_selftest 的全部超参数（场景时长、注入量、判据阈值）。
// 改数只改这里；行尾写取值依据。运行方式见 selftest_main.cpp 文件头。
#pragma once

namespace cfg {
// ---- 通用 ----
constexpr double kDt = 0.001;              // 库接口固定 1 ms（cf_sitl.h：dt 必须 0.001）
constexpr double kTargetHeight = 2.0;      // BRIEF §4 动作语义：目标高度 2.0 m
constexpr double kTakeoffMaxS = 15.0;      // BRIEF §4 proc takeoff 缺省 max_s；超过算起飞失败
constexpr int kCsvEveryMs = 10;            // CSV 抽样间隔：1 ms 全量 30 s 是 3 万行，10 ms 画图足够

// ---- (1) 坐标系一致性（外部模式注入） ----
constexpr double kFrameStillS = 5.0;       // imu.ino:70 落地 2 s 后才开始零偏校准，滤波 τ≈1 s，5 s 收敛到 95%
constexpr double kFrameRateRps = 0.5;      // 注入角速度：1 s 转 28.6°，轴错/符号错时姿态误差 ≥ 57°，轴对时 < 1°
constexpr double kFrameSegS = 1.0;         // 每段转 1 s；段间静止 1 s 让加计修正把倾角拉回真值
constexpr double kFrameLiftAcc = 1.5;      // 转动时注入的世界系竖直加速度 m/s²：|acc|−g > 0.1 g → 固件 landed=false（estimate.ino:54），
                                           // 停掉落地期间的零偏校准（会把转速吸进零偏）与加计修正，只测陀螺链
constexpr double kFrameAttTolDeg = 2.0;    // DLPF 184 Hz + 固件 ratesFilter ≈40 Hz 滞后约 0.2°，残余零偏 0.1°/s·1 s；轴错时 ≥ 57°
constexpr double kFrameTiltDeg = 10.0;     // 静态倾斜角（测加计链）：加计修正 accWeight 0.003/步，τ≈0.33 s
constexpr double kFrameTiltSettleS = 3.0;  // ≈9 τ
constexpr double kFrameTiltTolDeg = 1.0;   // 加计噪声 0.05 m/s² ÷ g ≈ 0.3°，零偏 0.085 m/s² ≈ 0.5°

// ---- (2) 悬停 10 s ----
constexpr double kHoverS = 10.0;           // 任务要求 10 s
constexpr double kHoverTiltRmsDeg = 2.0;   // 执行器倾角上限 10°；悬停时对光流噪声（0.05 rad/s × 2 m = 0.1 m/s）的响应应远小于上限
constexpr double kHoverZRmsM = 0.10;       // 与执行器 ready_height_tol 同量级；ToF 噪声 0.01 m
constexpr double kHoverDriftM = 1.0;       // 只有速度环没有位置环，10 s 随机游走 < 1 m 才算「悬停」

// ---- (3) 执行器级 ----
constexpr double kHeightHoldS = 2.0;       // READY 后再测 2 s 的高度稳态误差
constexpr double kHeightSsTolM = 0.10;     // 同 ready_height_tol 量级
constexpr double kForwardS = 3.0;          // 任务要求 3 s
constexpr double kForwardTarget = 0.3;     // BRIEF §4：FORWARD 0.3 m/s
constexpr double kSpeedTolFrac = 0.20;     // 稳态速度 ±20%：光流速度估计噪声 ≈0.1 m/s，低通后仍有偏差
constexpr double kRiseMaxS = 2.5;          // 到 90% 用时：速度环 ωn≈2.5 rad/s、ζ≈0.8 理论 ~1 s，倾角限幅 10° 会拉长
constexpr double kStopSpeedEps = 0.05;     // 停止判据 |v_xy| < 0.05 m/s（光流噪声量级的一半）
constexpr double kStopMaxS = 5.0;          // SAFE 后最多等 5 s
constexpr double kStopDistMaxM = 1.0;      // 0.3 m/s 以 ~1 s 时间常数减速约 0.3 m，留 3 倍余量
constexpr double kYawS = 3.0;              // 任务要求 3 s
constexpr double kYawTarget = 0.6;         // BRIEF §4：±0.6 rad/s，YAW_LEFT 绕 z 正向
constexpr double kYawTolFrac = 0.20;       // 偏航杆 0.1146 刚过固件死区 0.1，速率分辨率有限
constexpr double kYawRateWinS = 0.1;       // 偏航角速度用 0.1 s 差分（真值四元数，无噪声，差分窗口只为平滑离散化）
constexpr double kSafeS = 10.0;            // 任务要求 10 s
constexpr double kSafeDriftM = 1.0;        // 同 kHoverDriftM

// ---- (3.5) 固件级打杆阶跃（cf_sitl_set_sticks_raw 旁路执行器，油门每 tick 取执行器的悬停杆） ----
constexpr double kStickStep = 0.5;          // 杆量 +0.5：倾角 = 0.5 × 30°（control.ino tiltMax），偏航 = 0.5 × 300°/s（maxRate.z）
constexpr double kStickPitchTargetDeg = 15.0;
constexpr double kStickYawTargetDps = -150.0; // 固件正杆 = 顺时针 = 绕 z 负向（FLU，z 朝上）
constexpr double kStickTolFrac = 0.05;      // ±5%（任务要求）
constexpr double kStickHoldS = 1.5;         // 15° 俯仰在 standalone 里以 g·tan15° ≈ 2.6 m/s² 加速并掉高，1.5 s 够到稳态又不至于掉太多
constexpr double kStickSsWinS = 0.3;        // 稳态 = 末 0.3 s 均值；spike 里俯仰 0.475 s 进入 ±0.75° 带、偏航 0.221 s 进入 ±7.5°/s 带
constexpr double kStickPreHoldS = 1.0;      // READY 后先让执行器再稳 1 s 再打杆
constexpr double kStickReleaseS = 2.0;      // 松杆后 2 s 内回平（任务要求）
constexpr double kLevelDeg = 1.0;           // 回平判据 |roll|,|pitch| < 1°
// 阶跃与回平的判据用固件自己的姿态估计（att_est）：STAB 环控制的是估计值；飞行中固件只积分陀螺（estimate.ino:56
// 加计修正仅 landed 时生效），真值 = 估计 + 估计误差（本模型约 1–1.5°），估计误差单独打印，不混进控制环判据
constexpr double kRecoverMaxS = 10.0;       // 俯仰阶跃后取消旁路，等执行器刹住 ~3 m/s 滑行、补回 ~0.8 m 掉高并稳住；高度环 ki=0.05 慢，留 10 s
constexpr double kSettleZM = 0.05;          // 「稳住」= |z−2| < 0.05 m 且 |vz| < 0.05 m/s 持续 0.5 s；比执行器 READY 判据（0.1/0.1/1 s）严一半，
constexpr double kSettleVzMps = 0.05;       //   否则偏航阶跃会从带着爬升速度/积分未回落的状态开始（run3：以 0.26 m/s 爬到 2.9 m，ToF 超量程 → FAILSAFE）
constexpr double kSettleHoldS = 0.5;
constexpr double kYawReleaseS = 1.0;        // 松偏航杆后看 |r| < 5°/s 用时
constexpr double kYawStopDps = 5.0;
constexpr int kStepsCsvEveryMs = 5;         // 偏航上升约 0.1 s，10 ms 抽样太粗

// ---- (4) 确定性 ----
constexpr int kDetEpisode = 3;             // 任意固定值，两进程相同
constexpr double kDetHoverS = 4.0;         // 起飞 + 悬停 4 s，再 FORWARD/YAW/SAFE 各 2 s，覆盖全部动作分支
constexpr double kDetActS = 2.0;

// ---- (5) 电池 ----
constexpr double kBatSocInit = 1.0;        // 任务要求从 SoC 1.0 开始；参数文件 sim.battery_soc_init ≠ 1.0 时写一份改过的副本再 init
constexpr double kBatMaxS = 600.0;         // 2 × 参数文件 endurance 5 min：这之前固件 L3 还没触发，说明功率模型与标称续航差一倍以上
constexpr int kBatCsvEveryMs = 100;        // 10 分钟曲线 6000 行
constexpr double kBatL1V = 3.5;            // safety.ino L1：不解锁/怠速上锁；只用于曲线标注
constexpr double kBatL3V = 3.0;            // safety.ino L3：自动降落（fw_mode → AUTO=4）
} // namespace cfg
