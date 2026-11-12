#!/usr/bin/env python3
"""E3 采集器：驱动四旋翼在 Gazebo 里跑专家策略，录 RGB/位姿/碰撞成完整 episode。

依赖一个已在运行的 gz sim（见 sim/README.md 启动命令）。teacher/forward 路径纯 stdlib；
student 路径需要 PIL 与 uavrgb_infer（见 sim/tools/student.py）；teacher-nn 路径需要
torch/PIL/numpy（见 sim/tools/teacher_policy.py，解释器建议用 teacher-venv）。

用法：
  python3 collect.py --steps 20 --out /tmp/ep0 --scene-id 1 --seed 42 --split train
  python3 collect.py --steps 8 --out /tmp/ep1 --policy forward   # 盲策略（恒定前进）基线
  python3 collect.py --steps 80 --out /tmp/ep2 --policy student \
      --student-ckpt run/ckpt.bin --stop-on-collision           # 学生闭环评测
  python3 collect.py --steps 80 --out /tmp/ep3 --policy rule-oracle --scene-meta scene.json \
      --stop-on-collision --stop-at-goal                        # 真值列距离 + 规则 v0（零训练上限，plan D3）

I/O：启动时给 camera/depth/pose/contact/camera_info 各起一个常驻订阅
（gz_io.TopicSub：一个 `gz topic -e` 进程 + 后台读线程 + latest-wins 环形缓冲），
每步从缓冲取帧，不再每帧 spawn 一个 `gz topic -e -n 1`（本机约 250 ms/次）。
发布：启动时建一个常驻 twist publisher（gz_io.open_twist_publisher：优先进程内
gz.transport13；当前解释器没有该模块时用 `/usr/bin/python3` helper 子进程），
每步 publish 是内存操作（实测 <0.1 ms）。都建不起来时**直接失败**，不回退
`gz topic -p` CLI：CLI 每条固定 ~1.0 s，动作窗口会严重失真；CLI 只保留给
finally 的零速安全网。

每步（顺序是因果链的一部分，别调换）：
  等一帧新 RGB（锚点）-> 按 sim-time 挑最近的深度、挑晚于 RGB 的位姿 ->
  teacher 得动作 -> 发动作 -> 写 PNG（落在 hold 窗口内）->
  持 hold 秒仿真时间（期间轮询 contact）-> 立刻发零速并等它发布完成 -> 取动作后位姿 ->
  增量落盘/刷盘。
RGB 仍然是**动作前**的观测（M1：rgb_ts < action_start_ts），动作窗口结束立刻发零速（M2）。
动作后的状态只进 oracle.jsonl；actions.jsonl 的 timestamp_ns 是动作前 RGB 帧。
新鲜度门禁：depth 与 RGB 的 sim-time 差 > 100 ms 判停更/不同步；contact 订阅
进程退出即失败（contact 是稀疏话题，无消息不等于停更）。

输出（与冻结 EpisodeStep/EpisodeMeta 对齐，参照 doc 02 §四 建议 schema）：
  episode_manifest.json  episode_id/scene_id/seed/version/split + 文件引用 +
                         相机内参（/uav/camera_info 的 RGB 那条）+ camera_source
  actions.jsonl          每步 step/timestamp_ns(RGB)/action_expert/action_executed/
                         collision/rgb_ok/frame/depth_ts/depth_rgb_gap_ns/pose_ts/
                         pose_rgb_gap_ns/rgb_ts/action_start_ns/pose_before/yaw_before
                         + depth_file（仅 --save-depth）
                         + probs（仅 --policy student：学生四类概率，下标 = 动作编号，保留 4 位小数；零样本基线是 one-hot）
                         + safe_gap_s（每步都有，首步 null：本步派发戳减上一步执行器截止，秒，下一动作先到时为负；W1-13 A1）
                         + obs_gap_s（每步都有，首步 null：本观测戳减上一观测戳，秒；超过 0.9 + 0.2 s 的步是停顿步；W1-13 A3）
                         + plugin_ack_sim_t（仅 cf_sitl：插件侧第一次看到本步截止的 gz 仿真秒，没确认到为 null；W1-13 A5）
  oracle.jsonl           每步 step/timestamp_ns(动作后 pose)/pose_gt + 动作起止 sim-time
  rgb_%05d.png           320x240 原图（降采样到 160x120 由 tools/preprocess.py 负责）
  depth/depth_%05d.png   --save-depth 时：16 位灰度 PNG，单位毫米（0=无回波）

碰撞真值（两种真实信号，任一触发即 collision=True）：
  1. /uav/contact 接触传感器（base_link + 四个旋翼的 contact sensor + 世界级
     gz-sim-contact-system）。该话题是稀疏/触发式的：只在接触时发布（实测无接触时
     0 msg/s，接地/碰撞时 ~225 msg/s）。常驻订阅锁存非空事件序号，连续检查从
     离地到采集结束的事件，覆盖推理和步间漂移；地面接触在离地时排除。
  2. 位姿卡滞兜底：FORWARD_SLOW 指令下动作窗口位移 < STUCK_EPS 视为被挡。
"""
import argparse
import base64
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher import (expert_action, plan_action, plan_action_global_tiebreak, plan_action_costmap,
                     TIEBREAKS, TEACHER_COST_VERSION,
                     SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT, ACTIONS,
                     teacher_version, task_of_scene)
from rule_policy import column_distances, rule_v0_entry, rule_v1_step, RULE_V0, RULE_V1, COL_DEG
import gz_io
import save_frame
import split_manifest as sm
import cf_sil
import cf_blackbox
import cf_sitl_exec
import nav_cond

TWIST_TOPIC = "/X3/gazebo/command/twist"
CAMERA_TOPIC = "/uav/camera"
CAMERA_INFO_TOPIC = "/uav/camera_info"
DEPTH_TOPIC = "/uav/depth"
POSE_TOPIC = "/model/x3/pose"
CONTACT_TOPIC = "/uav/contact"

DEPTH_DIR = "depth"
DEPTH_REF = DEPTH_DIR + "/depth_%05d.png"   # episode 内相对路径模板（downstream 按 %05d 展开）

RGB_FRAME_ID = "x3::base_link::camera"   # /uav/camera_info 上 RGB 与 depth 两个 publisher
# 载具只影响位姿话题与 camera_info 的 frame_id（相机话题/分辨率两种载具相同）。
# x3 是默认，旧命令不传 --vehicle 时行为不变；e1_mini 配 --dynamics cf_sitl。
# pose_buf：位姿订阅的 latest-wins 缓冲帧数。_pick_pose_after 在缓冲里找「晚于 RGB 的最早一帧」，
# 缓冲跨度必须盖过 RGB 帧经 gz topic/JSON 到达的延迟（实测 70~175 ms 仿真时间），否则只能拿到
# 缓冲里最旧的一帧，pose 与 RGB 的间隔就等于到达延迟。x3 位姿约 45 Hz，6 帧≈130 ms（历史值不动）；
# e1_mini OdometryPublisher 200 Hz，64 帧≈320 ms（2026-11-07 真库采集实测 6 帧时 pose_gap 25~175 ms，8/10 步超 50 ms）。
VEHICLES = {
    "x3": {"pose_topic": POSE_TOPIC, "rgb_frame_id": RGB_FRAME_ID, "pose_buf": 6},
    "e1_mini": {"pose_topic": "/model/e1_mini/pose",
                "rgb_frame_id": "e1_mini::base_link::camera", "pose_buf": 64},
}
DEFAULT_VEHICLE = "x3"


def vehicle_name(world_sdf):
    """世界 SDF 文本里带学生相机（<sensor name="camera"）的模型名；找不到回退 x3。"""
    import re
    for m in re.finditer(r'<model name="([^"]+)">', world_sdf):
        end = world_sdf.find("</model>", m.end())
        body = world_sdf[m.end():end if end >= 0 else None]
        if '<sensor name="camera"' in body and "<topic>" + CAMERA_TOPIC + "</topic>" in body:
            return m.group(1)
    return DEFAULT_VEHICLE

GZ_TIMEOUT = 5.0      # 单次 gz 命令行调用上限（publisher 卡住时不永久挂起）
STUCK_EPS = 0.03      # FORWARD_SLOW 下动作窗口位移 < 3cm 视为卡滞（被挡）
FRAME_WAIT = 2.0      # 每步等一帧新 RGB 的上限（相机 30 Hz；卡住时报错而不是阻塞）
POSE_WAIT = 0.5       # 等一帧「晚于 RGB」的位姿的上限（M1；位姿 45 Hz）
READY_TIMEOUT = 10.0  # 启动时等 camera/depth/pose 首帧的上限
DEPTH_STALE_NS = 100_000_000  # depth 与 RGB 的 sim-time 差上限；超过视为停更/不同步
CONTACT_POLL = 0.02   # 动作窗口内轮询接触锁存序号的间隔
MIN_WINDOW_RTF = 0.05 # 动作窗口的墙钟兜底：仿真时间走不动时按此 RTF 折算等待上限
# --action-latency L：动作就绪时仿真戳已超过「观测戳 + L」这么多纳秒（0.02 s）就记迟到。
# 迟到步的派发时刻不再是观测戳 + L，该集的动作生效时长不可比，评测把含迟到步的集当基础设施失败。
LATE_DISPATCH_TOL_NS = 20_000_000
# W1-13 A3：停顿步。相邻两步的观测戳（RGB 戳）间隔名义上就是步周期：相机 10 Hz、hold 1.0 s、L 0.2 s 时
# frame_dt 恒为 0.9 s（round-2 学习者臂 4,972/4,972 个相邻步对实测）。间隔超过 名义 + 0.2 s 说明这一步前主循环/相机停顿过，
# 执行器在这段时间里回 SAFE 刹停，动作生效时长与别的步不可比。阈值等价于审核 A 的「SAFE 空档 > 0.3 s」（L=0.2、hold=1.0 时
# safe_gap = obs_gap − 0.8）。只用观测戳判，不依赖 L，所以不给 --action-latency 也能记。
CAMERA_PERIOD_S = 0.1
NOMINAL_STEP_PERIOD_S = 0.9
STALL_MARGIN_S = 0.2
STALL_TOL_S = 1e-6    # 浮点容差：恰好 名义 + 0.2 s 不算停顿


def nominal_step_period_s(hold=None):
    """给定动作窗口 hold（秒；None 按默认 1.0 s），相机 10 Hz 下的名义相邻观测间隔（hold − 0.1 s）。"""
    if hold is None:
        return NOMINAL_STEP_PERIOD_S
    return max(CAMERA_PERIOD_S, round(float(hold) - CAMERA_PERIOD_S, 9))


def is_stall_gap(obs_gap_s, hold=None):
    """obs_gap_s（秒，可为 None）严格大于 名义步周期 + 余量（加 1e-6 s 容差）即停顿；None 不是停顿。"""
    return obs_gap_s is not None and obs_gap_s > nominal_step_period_s(hold) + STALL_MARGIN_S + STALL_TOL_S

# 动作 -> 速度指令。forward=+X，left=+Y，up=+Z；yaw 正 z = 左转（右手定则）。
# vz 是**高度保持**分量（见 altitude_vz）：速度控制器只吃速度指令，z:0 是「零垂直速度」
# 不是「保持高度」，实测会缓慢下沉（一次 8 步 run 从 1.78 掉到 1.23 m）。
SPEED_FWD = 0.3      # 前进速度 m/s
SPEED_YAW = 0.6      # 偏航角速度 rad/s

Z_TARGET = 2.1       # 巡航高度 m：高于 1.5 m 地标顶，避免低空三带皆堵（S8）
Z_KP = 0.25          # 高度 P 增益（(m/s)/m）。不能大：wall-clock hold 下 sim 步长会漂。
Z_DEADBAND = 0.20    # 误差小于此不修正，避免小误差反复过冲
Z_MAX = 0.3          # 垂直速度限幅 m/s
TAKEOFF_VZ = 0.5     # 起飞上升速度 m/s（make_twist 会按 Z_MAX 限到 0.3）


def twist_fields(action, vz=0.0):
    """四动作 + 高度保持分量 -> (linear.x, linear.y, linear.z, angular.z)。

    vz 属于**仿真执行器**的实现细节，不是模型输出：模型仍只吐 0..3 四动作。
    """
    z = max(-Z_MAX, min(Z_MAX, vz))
    if action == FORWARD_SLOW:
        return SPEED_FWD, 0.0, z, 0.0
    if action == YAW_LEFT:
        return 0.0, 0.0, z, SPEED_YAW
    if action == YAW_RIGHT:
        return 0.0, 0.0, z, -SPEED_YAW
    return 0.0, 0.0, z, 0.0


def _twist_text(x, y, z, wz):
    return f"linear:{{x:{x:g} y:{y:g} z:{z:.3f}}} angular:{{z:{wz:g}}}"


def make_twist(action, vz=0.0):
    """把四动作 + 高度保持分量编成 gz.msgs.Twist 文本（CLI 零速安全网用）。"""
    return _twist_text(*twist_fields(action, vz))


def altitude_vz(pose):
    """按当前高度算垂直速度指令（带死区的 P 控制）。pose 为 None 时给 0（不瞎调）。"""
    if not pose or "z" not in pose:
        return 0.0
    err = Z_TARGET - pose["z"]
    if abs(err) < Z_DEADBAND:
        return 0.0
    return Z_KP * err


# ---------------------------------------------------------------- cf_stab 动力学档
# CF-Drone 固件只有 STAB/ACRO（control.ino），没有速度环；高度按 2026-11-04 决定继续
# 用真值定高（altitude_vz）。cf_stab 把四动作当真机摇杆：FORWARD_SLOW=俯仰杆保持固定
# 倾角，水平加速度 g·tanθ，线性阻力 k；其余动作俯仰杆回中，靠阻力滑停（不刹车）。
# 速度在世界系积分（转头时动量不随机头转），发指令前转回机体系；底层仍由
# MulticopterVelocityControl 跟踪这条速度曲线。默认 x3_velocity 保持旧行为不变。
DYNAMICS = ("x3_velocity", "cf_stab", "cf_blackbox", "cf_sil", "cf_sitl")
# cf_sil 已退出 collect_parallel / eval 的默认矩阵，只在 collect.py 保留手动入口；
# 写进 manifest 的 experimental 位，提醒下游不要把它当默认训练档。
DYNAMICS_EXPERIMENTAL = ("cf_sil",)
# cf_sitl：E1 MINI 固件在环（Gazebo 内 CfSitlSystem 插件 1 kHz 跑固件），collect 只发
# 「动作 + 截止仿真秒」；必须配 --vehicle e1_mini 的世界，默认参数 sim/dynamics/e1_mini.json。
DYNAMICS_VEHICLE = {"cf_sitl": "e1_mini"}
CF_STAB_PROFILE = "cf_stab_v1"
G = 9.81
CF_MAX_TILT_DEG = 30.0  # CF-Drone/control.ino 的最大倾角
CF_FWD_TILT_DEG = 2.0   # shortcut: 占位值（终端速度≈0.34 m/s，接近旧 SPEED_FWD），等 FORWARD_SLOW 对应的真机摇杆量实测后替换
CF_DRAG_K = 1.0         # 1/s。shortcut: 占位值，等松杆滑行实测阻力后替换
CF_LATENCY_S = 0.15     # 派发→生效。shortcut: 取 ALVLN/docs/09 的 P95≤150 ms 目标，等端到端实测后替换

# cf_sil：直接驱动 4 个 MotorModel 的串级 PID 载具（sim-only）。参数档默认在
# sim/dynamics/cf_sil_v1.json；profile 带参数 sha8，换参数后旧包不能混进来。
DEFAULT_CF_SIL_PARAMS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dynamics", "cf_sil_v1.json")
# cf_blackbox 是纯 Python 模型（3DGS collector 直接 import），参数唯一来源是
# sim/dynamics/cf_blackbox_v1.json；Gazebo collect 只做参数校验，不驱动它。
DEFAULT_CF_BLACKBOX_PARAMS = cf_blackbox.DEFAULT_PARAMS_PATH


class CfStab:
    """cf_stab 状态：世界系水平速度 + 带延迟的动作队列。时间一律是仿真 ns。"""

    def __init__(self, tilt_deg=CF_FWD_TILT_DEG, drag_k=CF_DRAG_K, latency_s=CF_LATENCY_S):
        if not 0.0 <= tilt_deg <= CF_MAX_TILT_DEG:
            raise ValueError(f"cf_stab 倾角 {tilt_deg} 超出固件上限 {CF_MAX_TILT_DEG}°")
        if not (drag_k > 0 and latency_s >= 0):
            raise ValueError(f"cf_stab 参数无效: drag_k={drag_k} latency_s={latency_s}")
        self.tilt_deg, self.k, self.latency_s = tilt_deg, drag_k, latency_s
        self.accel = G * math.tan(math.radians(tilt_deg))
        self.vx = self.vy = 0.0
        self.active = SAFE_NEUTRAL
        self.pending = []        # [(生效 ns, action)]，派发时刻单调递增
        self.t_ns = None
        self.path_m = 0.0        # 模型预测的累计水平路程（卡滞判定用）

    def profile(self):
        """参数编进档名：preprocess 只比 dynamics_profile，换实测参数后旧包不能混进来。"""
        return (f"{CF_STAB_PROFILE}:tilt{self.tilt_deg:g}:k{self.k:g}"
                f":lat{self.latency_s:g}")

    def params(self):
        return {"profile": self.profile(), "fwd_tilt_deg": self.tilt_deg, "drag_k": self.k,
                "latency_s": self.latency_s, "max_tilt_deg": CF_MAX_TILT_DEG,
                "altitude": "ground_truth_hold"}

    def dispatch(self, action, t_ns):
        self.pending.append((t_ns + int(self.latency_s * 1e9), action))

    def _advance(self, dt, yaw):
        a = self.accel if self.active == FORWARD_SLOW else 0.0
        ax, ay = a * math.cos(yaw) / self.k, a * math.sin(yaw) / self.k   # 终端速度
        e = math.exp(-self.k * dt)
        # v' = a − k·v 的解析解；位移是它对时间的积分
        dx = ax * dt + (self.vx - ax) * (1 - e) / self.k
        dy = ay * dt + (self.vy - ay) * (1 - e) / self.k
        self.vx, self.vy = ax + (self.vx - ax) * e, ay + (self.vy - ay) * e
        self.path_m += math.hypot(dx, dy)

    def step(self, t_ns, yaw_deg):
        """推进到仿真时刻 t_ns，返回机体系 (vx, vy, wz)。"""
        if yaw_deg is None:
            raise ValueError("cf_stab 需要偏航真值才能把世界系速度转回机体系")
        yaw = math.radians(yaw_deg)
        if self.t_ns is None:
            self.t_ns = t_ns
        while self.pending and self.pending[0][0] <= t_ns:
            t_eff, action = self.pending.pop(0)
            if t_eff > self.t_ns:
                self._advance((t_eff - self.t_ns) / 1e9, yaw)
                self.t_ns = t_eff
            self.active = action
        if t_ns > self.t_ns:
            self._advance((t_ns - self.t_ns) / 1e9, yaw)
            self.t_ns = t_ns
        wz = {YAW_LEFT: SPEED_YAW, YAW_RIGHT: -SPEED_YAW}.get(self.active, 0.0)
        c, s = math.cos(yaw), math.sin(yaw)
        return c * self.vx + s * self.vy, -s * self.vx + c * self.vy, wz


def rl_delay_remaining(dyn, now_ns):
    """teacher-rl 观测的 delay_remaining：上一次派发的指令延迟在观测时刻还剩多少秒。

    N1：训练环境在决策时刻早已把延迟倒计时到 0（0.15 < ACTION_DT 0.2），常量 latency_s
    会让 Gazebo 侧 obs[-1]=0.3 而训练侧恒为 0。按仿真时间算：有尚未生效的派发才非零。
    """
    from pathlib import Path
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from rl_teacher.core import delay_remaining_at
    if not dyn.pending:
        return 0.0
    t_eff, _ = dyn.pending[-1]
    dispatch_s = (t_eff - int(dyn.latency_s * 1e9)) / 1e9
    return delay_remaining_at(dispatch_s, now_ns / 1e9, dyn.latency_s)


def dyn_twist(dyn, t_ns, yaw_deg, pose):
    """cf_stab 指令：水平/偏航来自摇杆模型，垂直仍是真值定高。"""
    vx, vy, wz = dyn.step(t_ns, yaw_deg)
    return vx, vy, max(-Z_MAX, min(Z_MAX, altitude_vz(pose))), wz


def _gz(*args, timeout=GZ_TIMEOUT):
    """跑一条 gz 命令；超时返回 rc=124 而不是永久挂住。"""
    cmd = ["gz", *args]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "timeout")


def spawn_twist(twist):
    return subprocess.Popen(
        ["gz", "topic", "-t", TWIST_TOPIC, "-m", "gz.msgs.Twist", "-p", twist],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def finish_twist(proc, label):
    """等 publisher 退出并检查返回码；返回发布完成的 wall-clock 时刻。"""
    try:
        _, err = proc.communicate(timeout=GZ_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"{label} twist publish timeout")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} twist publish failed rc={proc.returncode}: "
                           f"{err.strip()[:200]}")
    return time.perf_counter()


def cmd_twist(twist):
    """同步发一条 twist；失败重试一次，仍失败抛异常（由调用方 finally 兜底）。"""
    for attempt in (1, 2):
        try:
            finish_twist(spawn_twist(twist), "cmd")
            return
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(0.2)


def publish_blocking(publisher, action, vz=0.0, tries=2):
    """经发布通道发一条指令并等它真正发布完；失败重试一次（同旧 cmd_twist）。

    重试只给起飞这类低频指令用；动作/零速窗口内不重试（会破坏窗口语义）。
    """
    for attempt in range(tries):
        try:
            return publisher.wait(publisher.start(*twist_fields(action, vz)))
        except Exception:
            if attempt + 1 >= tries:
                raise
            time.sleep(0.2)


def _try_zero(publisher):
    """安全网：补一条零速（先常驻通道，再 CLI 兜底）。从不抛 Exception。"""
    if publisher is not None:
        try:
            publish_blocking(publisher, SAFE_NEUTRAL, 0.0, tries=1)
            return True
        except Exception as e:  # noqa: BLE001 - 通道坏了还要试 CLI
            print(f"警告: 常驻通道零速失败: {e}", file=sys.stderr)
    try:
        cmd_twist(make_twist(SAFE_NEUTRAL, 0.0))
        return True
    except Exception as e:  # noqa: BLE001 - 安全网自身失败只能告警
        print(f"警告: CLI 零速也失败: {e}", file=sys.stderr)
        return False


def stop_executor(publisher, sil=None):
    """安全网：cf_sil 停桨；其他档发零速。从不抛 Exception。"""
    if sil is not None:
        try:
            sil.zero_now()
            return True
        except Exception as e:  # noqa: BLE001 - 安全网自身失败只能告警
            print(f"警告: cf_sil 停桨失败: {e}", file=sys.stderr)
            return False
    return _try_zero(publisher)


def echo_json(topic):
    """一次性读一帧（旧接口，保留给 test_collect / 手动排查；主路径用 TopicSub）。"""
    # /model/x3/pose 有两个 publisher，-n 1 竞态下 stdout 可能含 2 个 JSON
    # 文档（实测），故只取第一行；偶发连接失败/坏行时重试 3 次。
    for _ in range(3):
        r = _gz("topic", "-e", "-t", topic, "-n", "1", "--json-output")
        if r.returncode != 0 or not r.stdout.strip():
            time.sleep(0.3)
            continue
        try:
            return json.loads(r.stdout.splitlines()[0])
        except ValueError:
            time.sleep(0.3)
    return None


def stamp_ns(msg):
    s = msg.get("header", {}).get("stamp", {})
    return int(s.get("sec", 0)) * 10**9 + int(s.get("nsec", 0))


def decode_depth(msg):
    """gz.msgs.Image(R_FLOAT32) -> (depth, w, h, ts)；失败打印并返回 None。"""
    try:
        w, h, step = int(msg["width"]), int(msg["height"]), int(msg["step"])
        if w <= 0 or h <= 0 or step < 4 * w:
            raise ValueError(f"bad depth size {w}x{h} step={step}")
        fmt = str(msg.get("pixelFormatType") or msg.get("format") or "").upper()
        if fmt and "FLOAT32" not in fmt:
            raise ValueError(f"unsupported depth format {fmt!r} (need FLOAT32)")
        raw = base64.b64decode(msg.get("data", ""))
        need = (h - 1) * step + 4 * w
        if len(raw) < need:
            raise ValueError(f"depth payload too short: {len(raw)} < {need}")
        depth = []
        for y in range(h):
            depth.extend(struct.unpack("<%df" % w,
                                       raw[y*step:y*step + 4*w]))
    except (ValueError, KeyError, TypeError) as e:
        print(f"decode_depth: {e}", file=sys.stderr)
        return None
    return depth, w, h, stamp_ns(msg)


def read_depth():
    """旧接口：一次性读 /uav/depth。返回 (depth,w,h,ts) 或 None。"""
    msg = echo_json(DEPTH_TOPIC)
    if not msg:
        return None
    return decode_depth(msg)


def write_rgb_png(msg, path):
    """把一帧 gz.msgs.Image 存 PNG（save_frame 负责 stride/格式），返回 (ok, sim_time_ns)。"""
    try:
        w, h, nch, rows = save_frame.decode(msg)
        png = save_frame._png(w, h, nch, rows)
    except (ValueError, KeyError, TypeError) as e:
        print(f"write_rgb_png: {e}", file=sys.stderr)
        return False, 0
    with open(path, "wb") as f:
        f.write(png)
    return True, stamp_ns(msg)


def read_rgb_png(path):
    """旧接口：一次性读 /uav/camera 并存 PNG。返回 (ok, sim_time_ns)。"""
    msg = echo_json(CAMERA_TOPIC)
    if not msg:
        return False, 0
    return write_rgb_png(msg, path)


def _depth_png16(depth, w, h):
    """深度（米，行主序 flat list）-> 16 位灰度 PNG 字节（毫米，0=无效/无回波）。

    gz 的 depth_camera 输出 R_FLOAT32（米）；PNG 存 uint16 大端样本。
    """
    if len(depth) < w * h:
        raise ValueError(f"depth too short: {len(depth)} < {w * h}")
    rows = []
    for y in range(h):
        vals = []
        for v in depth[y*w:(y+1)*w]:
            mm = int(round(v * 1000.0)) if math.isfinite(v) else 0
            vals.append(max(0, min(65535, mm)))
        rows.append(b"\x00" + struct.pack(">%dH" % w, *vals))
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)   # 16-bit 灰度
    return (b"\x89PNG\r\n\x1a\n"
            + save_frame._chunk(b"IHDR", ihdr)
            + save_frame._chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
            + save_frame._chunk(b"IEND", b""))


def write_depth_png(depth, w, h, path):
    """把解码后的深度写成 16 位 PNG；返回是否成功（I/O 错误仍上抛）。"""
    try:
        png = _depth_png16(depth, w, h)
    except (ValueError, TypeError) as e:
        print(f"write_depth_png: {e}", file=sys.stderr)
        return False
    with open(path, "wb") as f:
        f.write(png)
    return True


def _yaw_deg(q):
    """四元数 -> 偏航角（度）。缺字段时返回 None。"""
    if not isinstance(q, dict):
        return None
    try:
        x, y, z, w = (float(q.get(k, 0.0)) for k in ("x", "y", "z", "w"))
    except (TypeError, ValueError):
        return None
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _pose_parts(msg):
    """gz.msgs.Pose_V -> (position, sim_time_ns, yaw_deg)。"""
    arr = msg.get("pose")  # {"pose":[{position,orientation,header},...]}
    if isinstance(arr, list) and arr:
        return (arr[0].get("position"), stamp_ns(arr[0]),
                _yaw_deg(arr[0].get("orientation")))
    return msg.get("position"), stamp_ns(msg), _yaw_deg(msg.get("orientation"))


def _pose_stamp(msg):
    return _pose_parts(msg)[1]


def read_pose():
    """旧接口：一次性读 /model/x3/pose。返回 (position, sim_time_ns, yaw_deg)。"""
    msg = echo_json(POSE_TOPIC)
    if not msg:
        return None, 0, None
    return _pose_parts(msg)


def spawn_contact_listener():
    """旧接口：在 /uav/contact 上等一条消息。主路径改用常驻订阅。"""
    return subprocess.Popen(
        ["gz", "topic", "-e", "-t", CONTACT_TOPIC, "-n", "1", "--json-output"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def contact_fired(proc):
    """旧接口：监听窗口内是否收到非空 gz.msgs.Contacts。超时/空消息/坏行都算无接触。"""
    try:
        out, _ = proc.communicate(timeout=0.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return False
    if proc.returncode != 0 or not out.strip():
        return False
    try:
        obj = json.loads(out.splitlines()[0])
    except (ValueError, IndexError):
        return False
    contacts = obj.get("contact") if isinstance(obj, dict) else None
    return isinstance(contacts, list) and len(contacts) > 0


def _dist3(a, b):
    return ((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2 + (a["z"] - b["z"])**2) ** 0.5


def _angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- 常驻订阅

def open_subs(vehicle=DEFAULT_VEHICLE):
    """给采集用到的 5 个话题各起一个常驻订阅；半路失败也要关掉已起的。"""
    subs = {}
    try:
        subs["rgb"] = gz_io.TopicSub(CAMERA_TOPIC, buf=4)
        subs["depth"] = gz_io.TopicSub(DEPTH_TOPIC, buf=4)
        subs["pose"] = gz_io.TopicSub(VEHICLES[vehicle]["pose_topic"],
                                      buf=VEHICLES[vehicle]["pose_buf"], stamp_fn=_pose_stamp)
        subs["contact"] = gz_io.TopicSub(CONTACT_TOPIC, buf=16,
                                          event_predicate=_contact_nonempty)
        subs["camera_info"] = gz_io.TopicSub(CAMERA_INFO_TOPIC, buf=4)
        return subs
    except BaseException:
        close_subs(subs)
        raise


def close_subs(subs):
    for sub in subs.values():
        try:
            sub.close()
        except Exception as e:  # noqa: BLE001 - 收尾失败不能影响 manifest
            print(f"警告: 关闭订阅 {sub.topic} 失败: {e}", file=sys.stderr)


def wait_ready(subs, timeout=READY_TIMEOUT):
    """等 rgb/depth/pose 都出过帧。contact 可能永远静默，camera_info 单独等。

    返回 None 表示就绪，否则返回错误原因。
    """
    deadline = time.monotonic() + timeout
    need = ("rgb", "depth", "pose")
    while time.monotonic() < deadline:
        dead = [n for n in subs if not subs[n].healthy()]
        if dead:
            return f"订阅进程已退出: {dead}"
        missing = [n for n in need if subs[n].latest() is None]
        if not missing:
            return None
        time.sleep(0.05)
    missing = [n for n in need if subs[n].latest() is None]
    return f"{timeout:.0f}s 内没有数据: {missing}"


def read_camera_info(sub, timeout=3.0, frame_id=RGB_FRAME_ID):
    """从 /uav/camera_info 取 RGB 那条（frame_id=<载具>::base_link::camera）的内参。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for _, msg in sub.frames():
            cam = camera_from_info(msg, frame_id)
            if cam:
                return cam
        if not sub.healthy():
            return None
        time.sleep(0.05)
    return None


def camera_from_info(msg, rgb_frame_id=RGB_FRAME_ID):
    """gz.msgs.CameraInfo -> {width,height,fx,fy,cx,cy,frame_id}；非 RGB 相机返回 None。"""
    if not isinstance(msg, dict):
        return None
    header = msg.get("header") or {}
    frame_id = None
    for item in header.get("data") or []:
        if isinstance(item, dict) and item.get("key") == "frame_id":
            value = item.get("value")
            frame_id = value[0] if isinstance(value, list) and value else value
    if frame_id != rgb_frame_id:
        return None
    k = (msg.get("intrinsics") or {}).get("k") or []
    if len(k) < 9:
        return None
    try:
        return {"width": int(msg["width"]), "height": int(msg["height"]),
                "fx": float(k[0]), "fy": float(k[4]),
                "cx": float(k[2]), "cy": float(k[5]), "frame_id": frame_id}
    except (KeyError, TypeError, ValueError):
        return None


def _pick_depth(sub, rgb_ts):
    """缓冲里挑 sim-time 离 rgb_ts 最近的一帧。返回 (stamp_ns, msg) 或 None。"""
    best = None
    for _, msg in sub.frames():
        ts = stamp_ns(msg)
        if best is None or abs(ts - rgb_ts) < abs(best[0] - rgb_ts):
            best = (ts, msg)
    return best


def _pick_pose_after(sub, rgb_ts, timeout=POSE_WAIT):
    """挑 stamp 晚于 rgb_ts 的最近一帧位姿（M1：位姿观测必须在 RGB 之后）。"""
    deadline = time.monotonic() + timeout
    while True:
        best = None
        for _, msg in sub.frames():
            ts = _pose_stamp(msg)
            if ts > rgb_ts and (best is None or ts < best[0]):
                best = (ts, msg)
        if best is not None:
            return best
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not sub.healthy():
            return None
        sub.latest_after(rgb_ts, timeout=remaining)   # 等一帧；下一轮统一扫描


def _contact_nonempty(msg):
    contacts = msg.get("contact") if isinstance(msg, dict) else None
    return isinstance(contacts, list) and len(contacts) > 0


def _contact_seq(sub):
    """已锁存的最新非空接触序号，跨越推理和帧缓冲淘汰仍保留。"""
    if hasattr(sub, "event_seq"):
        return sub.event_seq()
    # 兼容纯函数测试中的轻量订阅器。
    return max((seq for seq, msg in sub.frames() if _contact_nonempty(msg)), default=0)


def _contact_since(sub, seq0):
    """序号 > seq0 的帧里是否出现过非空 contact 列表。"""
    return _contact_seq(sub) > seq0


def take_off(pose_sub, publisher, contact_sub=None, z_target=Z_TARGET, timeout=120.0):
    """按位姿闭环升到 z_target；位姿取常驻订阅的最新帧（S7：不用固定 wall-clock）。

    起飞前先查一次高度：明显高于地面说明 world 没有复位（上一 episode 的
    残存状态），直接报错，避免把上一集的位置/姿态当成本集起点。

    timeout=120s（旧值 30s）：8 实例软件 GL 下 RTF 可低到 0.15，0.3 m/s 升
    1.95 m 要 6.5 s 仿真 ≈ 43 s 墙钟；30s 会在 z≈1.5 处误报，触发整集重试。
    """
    def pose_now():
        msg = pose_sub.latest()
        return _pose_parts(msg) if msg else (None, 0, None)

    pos, _, _ = pose_now()
    z0 = pos.get("z") if isinstance(pos, dict) else None
    if z0 is not None and z0 > 0.5:
        raise RuntimeError(
            f"world 未复位：起飞前 z={z0:.2f} m > 0.5；请重启 gz sim 或换新 world")
    if contact_sub is not None and not contact_sub.healthy():
        raise RuntimeError("contact 订阅进程已退出")

    air_seq0 = None   # 离地后才开始查 contact：地面接触不是撞柱
    publish_blocking(publisher, SAFE_NEUTRAL, TAKEOFF_VZ)
    deadline = time.monotonic() + timeout
    last_z = None
    while time.monotonic() < deadline:
        if contact_sub is not None and not contact_sub.healthy():
            publish_blocking(publisher, SAFE_NEUTRAL, 0.0)
            raise RuntimeError("起飞阶段 contact 订阅进程已退出")
        pos, _, _ = pose_now()
        if isinstance(pos, dict):
            last_z = pos.get("z")
            if contact_sub is not None and last_z is not None and last_z > 0.5:
                if air_seq0 is None:
                    air_seq0 = _contact_seq(contact_sub)
                elif _contact_since(contact_sub, air_seq0):
                    publish_blocking(publisher, SAFE_NEUTRAL, 0.0)
                    raise RuntimeError("起飞阶段检测到 contact（spawn 太近或撞柱）")
            if last_z is not None and last_z >= z_target:
                publish_blocking(publisher, SAFE_NEUTRAL, 0.0)
                print(f"起飞完成: z={last_z:.2f} m", file=sys.stderr)
                # 保留离地基线，不能吞掉最后一次检查到零速发布之间的接触。
                return air_seq0
        time.sleep(0.1)
    publish_blocking(publisher, SAFE_NEUTRAL, 0.0)
    raise RuntimeError(f"起飞超时 {timeout:.0f}s：z={last_z}，目标 {z_target}")


def take_off_cf_sil(pose_sub, sil, contact_sub=None, z_target=None, timeout=120.0):
    """cf_sil 起飞：SAFE_NEUTRAL 的伴飞定高会自动爬升到 altitude_target。

    不发布 twist：cf_sil 的电机由后台线程直接驱动，MulticopterVelocityControl 已停用。
    """
    z_target = sil.params.altitude_target if z_target is None else z_target
    def pose_now():
        msg = pose_sub.latest()
        return _pose_parts(msg) if msg else (None, 0, None)

    pos, _, _ = pose_now()
    z0 = pos.get("z") if isinstance(pos, dict) else None
    if z0 is not None and z0 > 0.5:
        raise RuntimeError(
            f"world 未复位：起飞前 z={z0:.2f} m > 0.5；请重启 gz sim 或换新 world")
    if contact_sub is not None and not contact_sub.healthy():
        raise RuntimeError("contact 订阅进程已退出")
    if sil.error():
        raise RuntimeError(f"cf_sil 执行器故障: {sil.error()}")

    sil.set_action(SAFE_NEUTRAL)
    air_seq0 = None
    deadline = time.monotonic() + timeout
    last_z = None
    while time.monotonic() < deadline:
        if sil.error():
            sil.zero_now()
            raise RuntimeError(f"起飞阶段 cf_sil 执行器故障: {sil.error()}")
        if contact_sub is not None and not contact_sub.healthy():
            sil.zero_now()
            raise RuntimeError("起飞阶段 contact 订阅进程已退出")
        pos, _, _ = pose_now()
        if isinstance(pos, dict):
            last_z = pos.get("z")
            if contact_sub is not None and last_z is not None and last_z > 0.5:
                if air_seq0 is None:
                    air_seq0 = _contact_seq(contact_sub)
                elif _contact_since(contact_sub, air_seq0):
                    sil.zero_now()
                    raise RuntimeError("起飞阶段检测到 contact（spawn 太近或撞柱）")
            if last_z is not None and last_z >= z_target:
                print(f"起飞完成: z={last_z:.2f} m", file=sys.stderr)
                return air_seq0
        time.sleep(0.1)
    sil.zero_now()
    raise RuntimeError(f"起飞超时 {timeout:.0f}s：z={last_z}，目标 {z_target}")


def take_off_cf_sitl(pose_sub, ex, contact_sub=None):
    """cf_sitl 起飞：执行器自己发 reset/arm_takeoff 并等固件 READY；这里只管真值侧的检查。

    与 take_off_cf_sil 同样的两道门：world 必须是新鲜的（z<=0.5），离地后出现 contact
    就中止（spawn 太近或撞柱）。落地瞬间的接触不算：基线序号取 READY 时刻的值。
    """
    def pose_now():
        msg = pose_sub.latest()
        return _pose_parts(msg) if msg else (None, 0, None)

    pos, _, _ = pose_now()
    z0 = pos.get("z") if isinstance(pos, dict) else None
    if z0 is not None and z0 > 0.5:
        raise RuntimeError(
            f"world 未复位：起飞前 z={z0:.2f} m > 0.5；请重启 gz sim 或换新 world")
    if contact_sub is not None and not contact_sub.healthy():
        raise RuntimeError("contact 订阅进程已退出")
    air_seq0 = [None]

    def abort_check():
        if contact_sub is None:
            return None
        if not contact_sub.healthy():
            return "contact 订阅进程已退出"
        pos, _, _ = pose_now()
        z = pos.get("z") if isinstance(pos, dict) else None
        if z is not None and z > 0.5:
            if air_seq0[0] is None:
                air_seq0[0] = _contact_seq(contact_sub)
            elif _contact_since(contact_sub, air_seq0[0]):
                return "离地后检测到 contact（spawn 太近或撞柱）"
        return None

    ex.start(abort_check)
    pos, _, _ = pose_now()
    z = pos.get("z") if isinstance(pos, dict) else None
    print(f"起飞完成: z={'?' if z is None else f'{z:.2f}'} m（固件 READY）", file=sys.stderr)
    return _contact_seq(contact_sub) if contact_sub is not None else None


# ---------------------------------------------------------------- 单步

def heading_guard_action(net_yaw, threshold):
    """航向护栏（R1 判别实验，2026-11-10）：净转向步数 |n| 达到阈值就朝回转一步，否则 None（不干预）。

    net_yaw = 已执行的 YAW_LEFT 数减 YAW_RIGHT 数，机体自己数得出来，不是特权信息；threshold <= 0 关闭。
    用途：学生闭环一半的步视锥内没有柱子，正确动作只由「已经偏了多少」决定而图像相同；
    这条规则不重训就能判别「缺航向」是不是出阵的主因。
    """
    if threshold > 0 and abs(net_yaw) >= threshold:
        return YAW_RIGHT if net_yaw > 0 else YAW_LEFT
    return None


def collect_one(out_dir, step_idx, hold, policy, subs, publisher,
                prev_action=None, prev_rgb_ts=0, pillars=None, student=None,
                global_plan=True, window_from="dispatch", waypoint=None,
                stop_at_waypoint=False, stop_radius=0.5, save_depth=False,
                teacher_nn=None, text_ids=None, contact_seq0=None, dyn=None,
                predictive_teacher=None, rl_teacher=None, sil=None,
                tiebreak="legacy", beta=0.0, rng=None, prev_executed=None,
                costmap_opts=None, heading_guard=0, net_yaw=0, rule_hfov_deg=60.0, last_turn=None,
                rule_version="v0", rule_mem=None, action_latency_s=None, student_cond="prev-action",
                prev_until_ns=None):
    """采一步。成功返回 (row, None)，I/O 失败返回 (None, 原因)。

    dyn=None 为 x3_velocity 旧行为（定速 + 窗口末刹停）；给 CfStab 时按摇杆模型
    在窗口内逐帧更新速度，窗口末不刹停，最后一条指令一直保持到下一步派发。
    给 sil（CfSilMotorThread）时不发 twist：后台线程按动作直接驱动 4 个电机，
    窗口末也不刹停，最后一条动作保持到下一步派发。
    tiebreak：A* 教师左右近平局的决定方式（teacher.TIEBREAKS；legacy 逐位等于旧 plan_action_global）。
    beta / rng / prev_executed 只在 policy == "student" 且有柱子真值时起作用（DAgger 影子教师）：
    每步以概率 beta 改执行教师动作，prev_executed 是上一步真正执行的动作（喂给 prev-action 模型）。
    costmap_opts：--teacher-planner costmap 时给 dict（plan_action_costmap 的 tol / lookahead_steps），
    教师改为测地场代价 argmin，每行多记 q_costs（4 个浮点，步）；None = 不启用。
    rule_hfov_deg / last_turn / net_yaw 只在 policy == "rule-oracle" 时起作用：真值位姿与柱子算列距离（视场 rule_hfov_deg），
    再过 rule_policy.rule_v0；net_yaw、last_turn 由主循环按已执行动作维护。
    rule_version="v1"（计划 D6）：改走 rule_policy.rule_v1_step，rule_mem 是主循环按集清零、逐步传入的记忆点列表
    [(世界 X, 世界 Y, 产生步号)]，本函数原地更新它；该步多记 rule_mem_n 与 rule_clear。v0 路径与落盘逐位不变。
    action_latency_s（--action-latency L，W1-0）：把派发时刻固定到仿真时间。观测戳 = 与 RGB 配对的位姿帧戳
    （pose_before_ts，observation 窗口的起点，执行器截止 = 观测戳 + hold 的基准）。策略出动作后等位姿话题的
    仿真戳 ≥ 观测戳 + L 再派发，动作生效时长固定为 hold − L，与策略这一步算了多久无关；动作就绪时仿真戳已超过
    观测戳 + L + LATE_DISPATCH_TOL_NS 的步记 late_dispatch=True（立即派发，不再等）。只配 window_from="observation"。
    学生 + β=0 + 给了它时，影子教师标签挪到派发之后算（A* 的耗时落进 hold 窗口，不计进观测到派发的延迟；
    β=0 时执行的动作与标签无关）；β>0（标签可能就是执行的动作）与教师策略（A* 就是动作）不动。
    None = 旧行为：派发在最新位姿戳，不等、不记迟到、标签位置不变；row["dispatch_lag_s"]（派发戳减观测戳，秒）两种情况都记。
    student_cond（--student-cond，W1-9）：只在 policy == "student" 时起作用。"prev-action"=现状，传给学生的是 prev_executed；
    "heading-code"=传 nav_cond.heading_code(net_yaw)（net_yaw 由主循环按已执行动作维护，首帧 0 → 码 0），该步多记 cond_code。
    prev_until_ns（W1-13 A1）：上一步的执行器截止（ns，仿真戳；主循环把上一步 row["until_ns"] 带进来，首步 None）。
    row["safe_gap_s"] = 本步派发戳（exec_ts）减它，秒；首步 None；下一动作先于截止到达时为负。cf_sitl 执行器在截止后回 SAFE
    （速度参考 0 加位置保持，主动刹停），所以它就是步间刹停的时长。row["until_ns"] = 本步的执行器截止，只给主循环带给下一步，不落盘。
    row["obs_gap_s"]（W1-13 A3）= 本观测戳（RGB 戳）减 prev_rgb_ts，秒；prev_rgb_ts 为 0（首步）时 None；超过 名义步周期 + 余量
    的步是停顿步（is_stall_gap）。与 W1-3 评测侧的 max_frame_gap_s 同一口径，这里逐步落盘。
    row["plugin_ack_sim_t"]（W1-13 A5）= sil.ack_sim_t()：插件侧第一次看到本步截止的仿真秒，窗口走完后读；只有 sil 有 ack_sim_t
    （cf_sitl）才写这个键，确认不到为 None。只记录，不参与迟到判据。
    """
    if window_from not in ("dispatch", "observation"):
        raise ValueError(f"window_from 无效: {window_from!r}")
    if student_cond not in nav_cond.COND_CHOICES:
        raise ValueError(f"student_cond 无效: {student_cond!r}（{'/'.join(nav_cond.COND_CHOICES)}）")
    if action_latency_s is not None and window_from != "observation":
        raise ValueError("action_latency_s 只能配 window_from='observation'（截止时刻 = 观测戳 + hold）")
    tie_margin = None          # 只有 tiebreak=visible 的 A* 教师会填；legacy 不多算也不落盘
    q_costs = None             # 只有 costmap 教师会填：Q[SAFE, FWD, L, R]（步）
    executed_by = None         # 只有 DAgger 影子教师模式会填 "student"/"expert"，rule-oracle 填 "rule_oracle"
    cond_code = None           # 只有 --student-cond heading-code 的学生步会填：本步传给学生的航向码
    rule_cols = rule_entry = None   # 只有 rule-oracle 会填：该步的 20 列（或 27 列）距离与触发的规则条目
    rule_mem_n = rule_clear = None  # 只有 rule-oracle v1 会填：记忆点数、[clear(+α), clear(0), clear(−α)]
    label = None               # 标签（action_expert）；None = 与执行的动作相同（除影子教师外都如此）
    # --action-latency + 学生 + β=0 + 有柱子真值：影子教师标签在派发之后算（见 docstring）；其余情况标签位置不变。
    defer_label = bool(action_latency_s is not None and policy == "student" and beta == 0.0 and pillars)
    late_dispatch = None       # 只有给了 action_latency_s 才非 None
    t_step = time.perf_counter()
    if contact_seq0 is None:
        contact_seq0 = _contact_seq(subs["contact"])

    # 1) 观测：以新的 RGB 帧为锚点，depth/pose 按 sim-time 配对
    rgb_msg = subs["rgb"].latest_after(prev_rgb_ts, timeout=FRAME_WAIT)
    if rgb_msg is None:
        return None, (f"camera {FRAME_WAIT:.0f}s 内无新帧"
                      if subs["rgb"].healthy() else "camera 订阅进程已退出")
    rgb_ts = stamp_ns(rgb_msg)
    if rgb_ts <= 0:
        return None, "RGB 帧缺少 header.stamp"
    # W1-13 A3：相邻观测戳间隔（秒）；首步没有上一戳（prev_rgb_ts == 0）记 None。
    obs_gap_s = (rgb_ts - prev_rgb_ts) / 1e9 if prev_rgb_ts else None

    if not subs["depth"].healthy():
        return None, "depth 订阅进程已退出"
    depth_pick = _pick_depth(subs["depth"], rgb_ts)
    if depth_pick is None:
        return None, "depth 无帧"
    depth = decode_depth(depth_pick[1])
    if depth is None:
        return None, "depth 解码失败"
    d, w, h, depth_ts = depth
    depth_gap_ns = abs(depth_ts - rgb_ts)
    if depth_gap_ns > DEPTH_STALE_NS:
        # 话题静默时缓冲里仍是旧帧；不设上限会把旧深度当本步观测写进数据。
        return None, (f"depth 停更/不同步：与 RGB 相差 {depth_gap_ns / 1e6:.1f} ms"
                      f" > {DEPTH_STALE_NS / 1e6:.0f} ms")

    pose_pick = _pick_pose_after(subs["pose"], rgb_ts)
    if pose_pick is None:
        return None, ("pose 无晚于 RGB 的帧" if subs["pose"].healthy()
                      else "pose 订阅进程已退出")
    pose_before_ts, pose_msg = pose_pick
    pose_before, _, yaw_before = _pose_parts(pose_msg)

    infer_ms = None
    probs = None    # W1-3：学生的四类概率；只有 --policy student 才赋值，其它策略保持 None（行里不出现 probs 键）
    if policy == "teacher":
        if pillars:
            # 有柱子真值就用 carrot planner：真值只决定标签，不进学生输入。
            # 位姿缺失时不能悄悄退回深度专家——manifest 已经记成 plan_action 了，
            # 混进去的深度专家标签事后看不出来。宁可让这一步失败。
            if not pose_before:
                return None, "carrot planner 需要位姿真值，但该步 pose 缺 position"
            # safe 出生用局部规划（要保留真正的 SAFE 标签）；普通集用 A* 全局路径。
            if predictive_teacher is not None:
                if dyn is None:
                    return None, "predictive teacher 需要 cf_stab 动力学档"
                action = predictive_teacher.act(pose_before["x"], pose_before["y"],
                    yaw_before, dyn.vx, dyn.vy, active=dyn.active, latency=dyn.latency_s)
            elif costmap_opts is not None:
                # 测地场代价 argmin（BRIEF_ALGO §1）：标签是 Q 最小的动作，Q 一并落盘给软标签用。
                action, cinfo = plan_action_costmap(
                    pose_before["x"], pose_before["y"], yaw_before, pillars,
                    waypoint=waypoint, stop_at_waypoint=stop_at_waypoint,
                    stop_radius=stop_radius, **costmap_opts)
                q_costs = cinfo["q"]
            elif global_plan:
                action, tie_margin = plan_action_global_tiebreak(
                    pose_before["x"], pose_before["y"], yaw_before, pillars, tiebreak,
                    waypoint=waypoint, stop_at_waypoint=stop_at_waypoint,
                    stop_radius=stop_radius)
            else:
                action = plan_action(pose_before["x"], pose_before["y"], yaw_before,
                                     pillars)
        else:
            # 没有真值就退回深度反应式专家（传 prev_action 启用滞回）。
            action = expert_action(d, w, h, prev_action=prev_action)
    elif policy == "teacher-rl":
        if rl_teacher is None or dyn is None or not pose_before or yaw_before is None:
            return None, "teacher-rl 需要 checkpoint、cf_stab、有效位姿与 scene-meta"
        try:
            from pathlib import Path
            repo_root = str(Path(__file__).resolve().parents[2])
            if repo_root not in sys.path: sys.path.insert(0, repo_root)
            from rl_teacher.core import Phys, state_vector
            state = Phys(x=pose_before["x"], y=pose_before["y"],
                         yaw=math.radians(yaw_before), vx=dyn.vx, vy=dyn.vy,
                         action=dyn.active, last_action=dyn.active,
                         delay_remaining=rl_delay_remaining(dyn, pose_before_ts))
            import numpy as np
            obs = np.asarray(state_vector(state, (waypoint["x"], waypoint["y"]),
                                          pillars), dtype=np.float32)
            result, _ = rl_teacher.predict(obs, deterministic=True)
            action = int(result)
        except Exception as e:
            return None, f"teacher-rl 推理失败: {type(e).__name__}: {e}"
    elif policy == "student":
        if student is None:
            return None, "student 策略未初始化"
        try:
            # prev_executed：β>0 时上一步可能执行的是教师动作，prev-action 模型要看真正执行的那个；
            # β=0 时它就是学生自己上一步的输出，与不传完全等价（student.act 的缓存语义）。
            # heading-code（W1-9）：传航向码，与 preprocess --cond heading-code 写进训练侧车的是同一个函数。
            cond_in = prev_executed
            if student_cond == nav_cond.COND_HEADING_CODE:
                cond_code = cond_in = nav_cond.heading_code(net_yaw)
            action, infer_ms, probs = student.act(rgb_msg, prev_action=cond_in)
            if probs is not None:
                # W1-3：保留学生的四类概率落进行里，4 位小数。放在 try 里：概率不是数字列表就在派发前判本步失败，
                # 不留到动作发出之后才在组装行时抛异常。是学生自己的输出，与本步谁执行无关（β 混合换成教师动作、
                # 航向护栏改写动作时，probs 仍是学生看到这一帧后给的）。
                probs = [round(float(p), 4) for p in probs]
        except Exception as e:  # noqa: BLE001 - 推理进程任何失败都算本步失败
            return None, f"学生推理失败: {type(e).__name__}: {e}"
        if pillars:
            # 影子教师（DAgger，DISTILL §3.2）：学生在自己的状态分布上飞，标签仍是该状态下的真值 A* 动作。
            # 训练集全是教师轨迹时学生偏离一步就进了没见过的状态（0417 流水线 test 成功率 27% 的主因之一）。
            # 位姿缺失时宁可本步失败，不能让学生动作冒充教师标签（manifest 已记成 plan_action）。
            if not pose_before:
                return None, "影子教师需要位姿真值，但该步 pose 缺 position"
            if not defer_label:   # defer_label（--action-latency，β=0）：标签在派发之后算，见下面 sil.dispatch 之后
                label, tie_margin = plan_action_global_tiebreak(
                    pose_before["x"], pose_before["y"], yaw_before, pillars, tiebreak,
                    waypoint=waypoint, stop_at_waypoint=stop_at_waypoint, stop_radius=stop_radius)
            # β 混合与 sim/3dgs/collect_3dgs.py DaggerPolicy 同义：以概率 β 改执行教师动作。
            if beta > 0.0 and rng is not None and rng.random() < beta:
                action, executed_by = label, "expert"
            else:
                executed_by = "student"
        guard = heading_guard_action(net_yaw, heading_guard)
        if guard is not None:
            action, executed_by = guard, "heading_guard"
    elif policy == "rule-oracle":
        # 规则层上限（计划 D3）：位姿与柱子真值 → 完美列距离 → 固定规则 v0。真值只在这条 oracle 策略里用。
        # 缺柱子或位姿时本步失败，不能悄悄退回别的策略。
        if pillars is None or not pose_before or yaw_before is None:
            return None, "rule-oracle 需要柱子真值（--scene-meta）与该步位姿、航向"
        cols = column_distances(pose_before["x"], pose_before["y"], yaw_before, pillars, hfov_deg=rule_hfov_deg)
        if rule_version == "v1":
            if rule_mem is None:
                return None, "rule-oracle v1 需要主循环传入 rule_mem（记忆点列表，按集清零）"
            action, rule_entry, new_mem, info = rule_v1_step(rule_mem, step_idx, pose_before["x"], pose_before["y"],
                                                             yaw_before, cols, net_yaw, last_turn, rule_hfov_deg)
            rule_mem[:] = new_mem
            rule_mem_n, rule_clear = info["mem_n"], info["clear"]
        else:
            action, rule_entry = rule_v0_entry(cols, net_yaw, last_turn)
        rule_cols, executed_by = [round(v, 2) for v in cols], "rule_oracle"
    elif policy == "teacher-nn":
        if teacher_nn is None:
            return None, "teacher-nn 策略未初始化"
        try:
            import numpy as np
            # RGB 解码成 HWC uint8；注意别覆盖已有的 depth 的 w/h（后面写深度图要用）。
            rgb_w, rgb_h, rgb_nch, rgb_rows = save_frame.decode(rgb_msg)
            rgb_hwc = np.frombuffer(rgb_rows, dtype=np.uint8).reshape(
                rgb_h, rgb_w, rgb_nch)
            depth_hw = np.asarray(d, dtype=np.float32).reshape(h, w)
            if not isinstance(pose_before, dict) or yaw_before is None:
                raise ValueError("缺少 pose_before/yaw_before")
            if not isinstance(waypoint, dict):
                raise ValueError("teacher-nn 需要 waypoint 作为特权输入（用 --task l1）")
            pose_xyz = (pose_before["x"], pose_before["y"], pose_before["z"])
            action = teacher_nn.act(rgb_hwc, depth_hw, pose_xyz, yaw_before,
                                    (waypoint["x"], waypoint["y"]), text_ids)
        except Exception as e:  # noqa: BLE001 - 推理任何失败都算本步失败
            return None, f"teacher-nn 推理失败: {type(e).__name__}: {e}"
    else:  # policy == "forward": 盲策略（恒定前进），用于撞柱/基线对照
        action = FORWARD_SLOW
    if label is None and not defer_label:
        label = action
    t_io = time.perf_counter()

    if action_latency_s is not None:
        # 派发时刻固定到 观测戳 + L（仿真时间）。就绪时仿真戳超过 观测戳 + L + 0.02 s 记迟到并立即派发；
        # 否则等位姿话题的仿真戳走到 观测戳 + L（200 Hz 位姿，等待精度 5 ms）。这段等待不计进 timing["io"]。
        latency_ns = int(round(action_latency_s * 1e9))
        target_ns = pose_before_ts + latency_ns
        ready_msg = subs["pose"].latest()
        ready_ts = _pose_stamp(ready_msg) if ready_msg is not None else 0
        late_dispatch = ready_ts > target_ns + LATE_DISPATCH_TOL_NS
        if ready_ts < target_ns:
            wait_s = action_latency_s / MIN_WINDOW_RTF   # 仿真时间走不动时的墙钟兜底，同动作窗口
            if subs["pose"].latest_after(target_ns - 1, timeout=wait_s) is None:
                return None, ("pose 订阅进程已退出" if not subs["pose"].healthy() else
                              f"等派发时刻超时：{wait_s:.1f}s 墙钟内位姿戳没到 观测戳 + {action_latency_s:g} s"
                              "（位姿停更或 RTF 过低）")

    # 动作窗口的起点是**派发时刻**的最新位姿，不是观测帧位姿：策略推理/解码
    # 期间仿真仍在走，用观测戳会让实际执行窗口短于 hold（学生与基线不公平）。
    exec_msg = subs["pose"].latest()
    exec_ts = _pose_stamp(exec_msg) if exec_msg is not None else 0
    if exec_ts < pose_before_ts:
        exec_msg, exec_ts = pose_msg, pose_before_ts
    pose_exec, _, yaw_exec = _pose_parts(exec_msg)
    if pose_exec is None:
        pose_exec = pose_before
    if yaw_exec is None:
        yaw_exec = yaw_before
    # W1-13 A1：上一步截止到本步派发的仿真秒（SAFE 空档）；首步 None，下一动作先到则为负。
    safe_gap_s = None if prev_until_ns is None else (exec_ts - prev_until_ns) / 1e9

    # contact 是稀疏触发话题（无接触时 0 msg/s），不能把「长时间无消息」当停更；
    # 这里只拒绝订阅进程退出。话题静默/插件不发布的检测需要额外心跳，见 D4 记录。
    if not subs["contact"].healthy():
        return None, "contact 订阅进程已退出"
    action_handle = zero_handle = None
    action_wall = 0.0
    rgb_ok = False
    try:
        # dispatch=R01 语义（派发位姿起算）；observation=旧语义（观测位姿起算，
        # 推理/解码耗时从窗口里扣掉），只用于诊断/兼容旧数据。
        action_start_ns = exec_ts if window_from == "dispatch" else pose_before_ts
        pose_for_action = pose_before if window_from == "observation" else pose_exec
        t_action_start = time.perf_counter()
        try:
            if sil is not None:
                # 动作交给后台电机线程；不建 twist publisher，也不发 twist。
                # cf_sitl 的截止时刻按动作窗口起点算（observation 语义 = 观测帧 + hold），
                # 与下面 hold_end_ns 同一个起点；cf_sil 仍按派发时刻加延迟。
                sil.dispatch(action, action_start_ns
                             if getattr(sil, "until_from_window_start", False) else exec_ts)
                action_handle = None
                path0 = 0.0
            elif dyn is None:
                action_handle = publisher.start(
                    *twist_fields(action, altitude_vz(pose_for_action)))
            else:
                dyn.dispatch(action, exec_ts)
                twist0 = dyn_twist(dyn, exec_ts, yaw_exec, pose_exec)
                path0 = dyn.path_m          # 实际位移也从 pose_exec 起算
                action_handle = publisher.start(*twist0)
        except Exception:
            # 动作没发出去也要确认没有旧指令在飞；失败原因继续上抛。
            stop_executor(publisher, sil)
            raise

        if defer_label:
            # 影子教师标签（--action-latency，学生 β=0）：动作已派发，A* 的耗时落在 hold 窗口里（与下面 PNG 落盘同位置），
            # 不再计进观测到派发的延迟。同一个函数、同一组位姿与参数，标签只写进 row，不影响已派发的动作。
            try:
                label, tie_margin = plan_action_global_tiebreak(
                    pose_before["x"], pose_before["y"], yaw_before, pillars, tiebreak,
                    waypoint=waypoint, stop_at_waypoint=stop_at_waypoint, stop_radius=stop_radius)
            except Exception:
                stop_executor(publisher, sil)   # 动作已经在飞：标签失败也要刹住再上抛
                raise

        # PNG 落盘落在 hold 窗口内（RGB 已在发指令前抓到，M1 不受影响）。
        rgb_ok, _ = write_rgb_png(
            rgb_msg, os.path.join(out_dir, "rgb_%05d.png" % step_idx))
        if save_depth:
            # 深度图与 RGB 同 step 编号；写失败不能让 jsonl 指向不存在的文件。
            if not write_depth_png(d, w, h,
                                   os.path.join(out_dir, DEPTH_REF % step_idx)):
                stop_executor(publisher, sil)
                return None, "深度图写盘失败"

        contact = False
        # 动作窗口按仿真时间计：动作前的位姿戳 + hold 秒。旧实现用墙钟，RTF 一抖
        # 实际走过的仿真时间就差得远（/tmp/ds200 实测 sim_dt 变异系数 18%），
        # 标签「前进 1 秒」到底代表多少位移变得不可复现。
        hold_end_ns = action_start_ns + int(hold * 1e9)
        wall_cap = t_action_start + hold / MIN_WINDOW_RTF   # 位姿停更时不死等
        while True:
            if not subs["contact"].healthy():
                stop_executor(publisher, sil)
                return None, "动作窗口内 contact 订阅进程已退出"
            if sil is not None and sil.error():
                stop_executor(publisher, sil)
                return None, f"{getattr(sil, 'name', 'cf_sil')} 执行器故障: {sil.error()}"
            msg = subs["pose"].latest()
            if msg is not None and _pose_stamp(msg) >= hold_end_ns:
                break
            if dyn is not None and msg is not None and _pose_stamp(msg) > dyn.t_ns:
                pos_m, ts_m, yaw_m = _pose_parts(msg)
                publisher.start(*dyn_twist(dyn, ts_m, yaw_m, pos_m))
            if not subs["pose"].healthy():
                # 订阅进程已死：不必等满墙钟兜底，也能给出比「RTF 过低」准确的诊断。
                stop_executor(publisher, sil)
                return None, "动作窗口内 pose 订阅进程已退出"
            if time.perf_counter() >= wall_cap:
                stop_executor(publisher, sil)   # 动作还在生效，先刹车再报错
                return None, (f"动作窗口超时：{hold / MIN_WINDOW_RTF:.1f}s 墙钟内"
                              f"仿真时间没走满 hold={hold}s（位姿停更或 RTF 过低）")
            time.sleep(CONTACT_POLL)
            if not contact:
                contact = _contact_since(subs["contact"], contact_seq0)

        # M2：hold 到点立刻发零速，并等它真正发布完成才继续（不丢零速）。
        # cf_stab 不刹停：发窗口末的模型速度，保持到下一步派发。
        # cf_sil 也不刹停：后台电机线程继续执行最后一条动作，直到下一步 dispatch。
        # shortcut: cf_stab 步间这条指令是常量，模型的阻力衰减要到下一步派发才补上；
        # 步间隔远大于 0.1 s 仿真时间时要改成后台线程持续发布（cf_sil 已是后台线程）。
        if sil is not None:
            action_pub_wall = t_action_start
            zero_pub_wall = time.perf_counter()
        else:
            try:
                if dyn is None:
                    end_twist = twist_fields(SAFE_NEUTRAL, 0.0)
                else:
                    pos_m, ts_m, yaw_m = _pose_parts(msg)
                    end_twist = dyn_twist(dyn, ts_m, yaw_m, pos_m)
                zero_handle = publisher.start(*end_twist)
                # 先收动作的发布完成时刻：若先等零速再等动作，动作的完成时刻
                # 会被记晚而误判。
                action_pub_wall = publisher.wait(action_handle)
                zero_pub_wall = publisher.wait(zero_handle)
            except Exception:
                stop_executor(publisher, sil)
                raise
        action_wall = zero_pub_wall - action_pub_wall
        if action_wall <= 0:
            # 零速比动作先发布 => 动作可能一直生效；先补零速再失败。
            stop_executor(publisher, sil)
            raise RuntimeError("zero twist overtook action twist (hold too small)")
        t_block = time.perf_counter()

        # 动作后的位姿（oracle）：取窗口结束后缓冲里最新的一帧（位姿 45 Hz）。
        after_msg = subs["pose"].latest()
        pose_after, pose_ts, yaw_after = (_pose_parts(after_msg) if after_msg
                                          else (None, 0, None))
        if pose_ts <= action_start_ns:
            # 位姿话题静默（进程还活着但不出帧）：再等一帧，避免把动作前位姿当 oracle。
            newer = subs["pose"].latest_after(action_start_ns, timeout=1.0)
            if newer is not None:
                pose_after, pose_ts, yaw_after = _pose_parts(newer)
        if pose_ts <= action_start_ns:
            # 仍拿不到动作后的帧：这一步的 oracle 和 stuck 都不可信，不能盖 physics_verified。
            return None, "pose 停更：动作窗口后没有晚于 action_start 的位姿"
        if not subs["contact"].healthy():
            return None, "动作窗口后 contact 订阅进程已退出"
        contact_seq_end = _contact_seq(subs["contact"])
        contact = contact or contact_seq_end > contact_seq0
        t_end = time.perf_counter()
    finally:
        if publisher is not None:
            publisher.abort(action_handle)
            publisher.abort(zero_handle)

    # sim_dt 保留动作窗口合同；性能单独按派发位姿起算，避免混入推理时间。
    sim_dt = (pose_ts - action_start_ns) / 1e9 if pose_ts and action_start_ns else 0.0
    dispatch_sim_dt = (pose_ts - exec_ts) / 1e9
    dist = (_dist3(pose_exec, pose_after)
            if pose_exec and pose_after else None)
    dyaw = (_angle_diff(yaw_after, yaw_exec)
            if yaw_after is not None and yaw_exec is not None else None)
    # 低 RTF 时动作窗口的 sim 时间会变短，卡滞阈值跟着缩放，避免误报。
    if dyn is None and sil is None:
        stuck_eps = min(STUCK_EPS, 0.5 * SPEED_FWD * sim_dt)
        armed = True
    elif sil is not None:
        # cf_sil：hold < 0.5 s 时从静止起步的位移本身就小于 STUCK_EPS，不启用卡滞
        # 兜底（碰撞以 contact 为主）。长窗口按本步 FORWARD 设定点折算期望位移。
        # shortcut: 忽略步前已有速度，期望位移偏保守；换 hold/速度设定点后要重测。
        # cf_sitl（stuck_detection=False）：速度曲线由固件决定，不按 SPEED_FWD 折算，碰撞只认 contact。
        expected_m = SPEED_FWD * sim_dt if action == FORWARD_SLOW else 0.0
        stuck_eps = min(STUCK_EPS, 0.25 * expected_m)
        armed = (getattr(sil, "stuck_detection", True)
                 and sim_dt >= 0.5 and expected_m >= 2 * STUCK_EPS)
    else:
        # cf_stab：X3 底层跟踪模型曲线滞后约 0.4 s（2026-11-04 实测加速初段实际/预测
        # 低至 0.3），按 0.5×预测会把起步误判成卡滞。只在模型预测已走满 STUCK_EPS 时判，
        # 且实际不到预测的 1/4 才算被挡（贴柱时实际≈0）。
        # shortcut: 单窗口阈值，靠 X3 跟踪滞后的实测值定；换底层控制器或改 hold 后要重测误报率。
        expected_m = dyn.path_m - path0
        stuck_eps = 0.25 * expected_m
        armed = expected_m >= STUCK_EPS
    stuck = bool(action == FORWARD_SLOW and sim_dt > 0 and dist is not None
                 and armed and dist < stuck_eps)

    row = {
        "contact_seq_end": contact_seq_end,
        "step": step_idx,
        "timestamp_ns": rgb_ts,        # 动作前 RGB 帧 sim-time（EpisodeStep）
        "rgb_ts": rgb_ts,
        "action_expert": label,
        "action_executed": action,     # 发布通道已确认成功，否则不会走到这里；影子教师模式下可与标签不同
        "collision": contact or stuck,
        # 下面两项为 None 时 action_row 不落盘（默认采集的字段集与旧版完全一致）。
        "tie_margin_steps": tie_margin,
        "executed_by": executed_by,
        "q_costs": q_costs,
        "rule_cols": rule_cols,
        "rule_entry": rule_entry,
        "rule_mem_n": rule_mem_n,
        "rule_clear": rule_clear,
        "pose_gt": pose_after,
        "pose_ts": pose_ts,
        # 动作前位姿：audit 用它 + scene 柱子真值复算标签，验证「标签=该帧该做的动作」。
        "pose_before": pose_before,
        "yaw_before": yaw_before,
        "rgb_ok": rgb_ok,
        "frame": "rgb_%05d.png" % step_idx,
        "depth_file": (DEPTH_REF % step_idx) if save_depth else None,
        "depth_ts": depth_ts,
        "pose_before_ts": pose_before_ts,
        "action_start_ns": action_start_ns,
        # 派发时的仿真戳减观测戳（秒）：每步都记，不依赖 --action-latency；late_dispatch 只有给了它才非 None。
        "dispatch_lag_s": (exec_ts - pose_before_ts) / 1e9,
        "late_dispatch": late_dispatch,
        # W1-13 A1：safe_gap_s 落盘（action_row）；until_ns 只在行里，主循环带给下一步算 safe_gap_s。
        "safe_gap_s": safe_gap_s,
        "obs_gap_s": obs_gap_s,     # W1-13 A3：本观测戳减上一观测戳（秒），首步 None
        "until_ns": hold_end_ns,    # 执行器截止：动作窗口起点 + hold
        "cond_code": cond_code,      # None 时 action_row 不落盘（默认的字段集与旧版完全一致）
        "action_end_ns": pose_ts,
        "displacement_m": dist,
        "dyaw_deg": dyaw,
        "sim_dt": sim_dt,
        # 窗口末速度（m/s）：只有 cf_sitl 的 state 话题给真值速度；其余档为 None。
        # eval 的 stable_termination 用它，不用「位移/时间」的窗口均值冒充末速度。
        "speed_after_mps": (sil.speed_mps() if sil is not None and hasattr(sil, "speed_mps")
                            else None),
        "infer_ms": infer_ms,
        "action_wall_s": action_wall,
        "src": "+".join(s for s, on in (("contact", contact), ("stuck", stuck)) if on) or None,
        # 每步耗时分解（秒）：io=取帧+解码+teacher，window=动作到零速的窗口，
        # block=开始发动作到零速发布完成，tail=动作后位姿/接触/记账。
        "timing": {"io": t_io - t_step, "window": action_wall,
                   "block": t_block - t_action_start, "tail": t_end - t_block,
                   "total": t_end - t_step, "dispatch_sim_dt": dispatch_sim_dt},
    }
    if probs is not None:     # 只有学生策略才有 probs 键；其它策略的行逐键不变
        row["probs"] = probs
    # W1-13 A5：插件侧实际生效的仿真秒（执行器从 state 缓冲里回扫截止第一次变成本步截止的那一帧）。窗口走完才读，帧一定已在缓冲里。
    # 只有执行器给得出（cf_sitl 的 ack_sim_t）才有这个键，没确认到帧时为 None；x3_velocity / cf_stab / 假执行器的行逐键不变。
    if sil is not None and hasattr(sil, "ack_sim_t"):
        row["plugin_ack_sim_t"] = sil.ack_sim_t()
    return row, None


def write_manifest(path, manifest):
    """临时文件 + os.replace：崩溃时不会留下半截 manifest。"""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def write_scene_meta(path, raw):
    """原样落盘 scene_gen 的 JSON：audit 复算标签要读 pillars。"""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, path)


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def action_row(r, save_depth=False):
    """actions.jsonl 的字段白名单投影。

    坑（HANDOFF §八.7）：collect_one 返回的 row 不会自动落盘，新字段必须显式
    加到这里；save_depth=False 时字段集与旧版完全一致。
    """
    row = {k: r[k] for k in (
        "step", "timestamp_ns", "action_expert", "action_executed",
        "collision", "rgb_ok", "frame")}
    row.update({
        # 评测端按 EVAL_STANDARD §5.1 只把 contact 当碰撞，stuck 单列。
        "src": r["src"],
        "rgb_ts": r["rgb_ts"],
        "depth_ts": r["depth_ts"],
        "depth_rgb_gap_ns": r["depth_rgb_gap_ns"],
        # 配对用的观测位姿帧；动作后的 oracle 位姿只在 oracle.jsonl。
        "pose_ts": r["pose_before_ts"],
        "pose_rgb_gap_ns": r["pose_rgb_gap_ns"],
        "dyaw_deg": r["dyaw_deg"],
        "displacement_m": r["displacement_m"],
        "infer_ms": r["infer_ms"],
        "action_start_ns": r["action_start_ns"],
        # 动作前位姿：audit 的 label_recompute 靠它 + 柱子真值重跑 teacher。
        "pose_before": r["pose_before"],
        "yaw_before": r["yaw_before"],
    })
    if save_depth:
        row["depth_file"] = r["depth_file"]
    # 可选字段只在开了对应开关时出现：tie_margin_steps 来自 --tiebreak visible（|Q_L−Q_R|，步），
    # executed_by 来自 --policy student 的影子教师（本步执行者 student/expert），
    # q_costs 来自 --teacher-planner costmap（Q[SAFE, FWD, L, R]，步），
    # rule_cols / rule_entry 来自 --policy rule-oracle（该步各列距离与触发的规则条目，供事后分析失败）；
    # rule_mem_n / rule_clear 只有 --rule-version v1 才有（记忆点数与三个候选航向的通道是否畅通）；
    # probs（W1-3）只有 --policy student 才有：学生看到那一帧后的四类概率，下标 = 动作编号 0..3（src/infer.cpp：action 是
    # probs 的 argmax 下标），4 位小数；collect_one 只在学生策略下给这个键。零样本基线（baselines/baseline_infer.py）也经
    # --policy student --infer-bin 走这一支，它没有四类概率，probs 是 one-hot（原始输出在基线自己的 raw）。
    for k in ("tie_margin_steps", "executed_by", "q_costs", "rule_cols", "rule_entry", "rule_mem_n", "rule_clear",
              "probs"):
        if r.get(k) is not None:
            row[k] = r[k]
    # W1-13 A1：safe_gap_s（本步派发戳减上一步执行器截止，秒）；A3：obs_gap_s（本观测戳减上一观测戳，秒）。真 collect_one 的行每步都有
    # 这两个键（首步值为 None，落盘 null）；行里没有这个键（旧代码构造的行、合成行）就不写，旧字段集逐键不变。
    # 排在 W1-0 的两个键之前，旧键的行内顺序不变。
    for k in ("safe_gap_s", "obs_gap_s"):
        if k in r:
            row[k] = r[k]
    # W1-13 A5：plugin_ack_sim_t（插件侧实际生效的仿真秒）只有 cf_sitl 的行才有这个键；确认不到时为 null。
    if "plugin_ack_sim_t" in r:
        row["plugin_ack_sim_t"] = r["plugin_ack_sim_t"]
    # W1-0：dispatch_lag_s（派发的仿真戳减观测戳，秒）每步都记，排在最后（旧键的行内顺序不变）；
    # late_dispatch（bool）只有给了 --action-latency 才有。
    for k in ("dispatch_lag_s", "late_dispatch"):
        if r.get(k) is not None:
            row[k] = r[k]
    # W1-9：cond_code（传给学生的航向码，0/2/3）只有 --student-cond heading-code 才有；0 是合法值，所以按 is not None 判断。排在最后。
    if r.get("cond_code") is not None:
        row["cond_code"] = r["cond_code"]
    return row


def performance_summary(timings):
    """只统计已落盘步骤；RTF 按派发后的位姿时间段估算，不含启动和起飞。"""
    if not timings:
        return {"steps": 0}
    wall = sum(t["window"] for t in timings)
    sim = sum(t["dispatch_sim_dt"] for t in timings)
    return {"steps": len(timings),
            "rtf_basis": "dispatch_pose_to_after_pose",
            "mean_step_s": {k: sum(t[k] for t in timings) / len(timings)
                            for k in ("io", "window", "block", "tail", "total")},
            "action_sim_s": sim, "action_wall_s": wall,
            "action_rtf": sim / wall if wall > 0 and sim > 0 else None}


def stopped_at_waypoint(row, waypoint, stop_radius):
    """本步 SAFE 是否真的停在 waypoint 半径内。

    教师只在半径内输出 SAFE，学生可能在任何位置喊 SAFE；只看动作会把
    "3.4 m 外喊停"当成到达并提前截断 episode（2026-11-03 审计）。
    """
    pb = row.get("pose_before") or {}
    if waypoint is None or not isinstance(pb, dict) or "x" not in pb or "y" not in pb:
        return False
    return math.hypot(pb["x"] - waypoint["x"],
                      pb["y"] - waypoint["y"]) <= (stop_radius or 0.5)


# 成功线 = 最远柱 x + 1.0 m：与 EVAL_STANDARD 的 route_choice / eval_closed_loop --goal-margin /
# traverse_rate 同一口径（max_x 取 oracle pose_gt.x）。DISTILL §1.2/§3.1：0417 流水线 12000 帧里过线后的
# 2155 帧（18%）92.9% 视锥内无柱，标签 37.8% 是 YAW——是看不见的目标点的函数，学生学成「空画面就转」；
# --stop-at-goal 在采集时就不再产出它们（离线按同一条线截断重训：无柱帧误转率 22% → 5%）。
GOAL_MARGIN_M = 1.0


def goal_line_x(pillars, margin=GOAL_MARGIN_M):
    """成功线的 x 坐标。"""
    return max(float(pl["x"]) for pl in pillars) + margin


def crossed_goal_line(row, goal_x):
    """本步动作后的真值位姿是否已到/过成功线（与 eval 的 max_x ≥ goal_x 判定同口径）。"""
    pg = row.get("pose_gt")
    return isinstance(pg, dict) and pg.get("x") is not None and float(pg["x"]) >= goal_x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=sm.DEFAULT_STEPS)
    p.add_argument("--out", required=True)
    p.add_argument("--hold", type=float, default=1.0,
                   help="每动作持续时间（秒，仿真时间；走满即发零速）")
    p.add_argument("--window-from", choices=("dispatch", "observation"),
                   default="observation",
                   help="动作窗口起点：observation=观测位姿（默认，唯一被采纳基线）；"
                        "dispatch=派发位姿（R01，仅复现 T7-R，数字不可与 observation 混报）；"
                        "cf_sil 建议用 dispatch，窗口才和电机线程的派发时刻对齐")
    p.add_argument("--dynamics", choices=DYNAMICS, default="x3_velocity",
                   help="x3_velocity=旧行为（定速+每步刹停，唯一历史基线）；"
                        "cf_stab=CF-Drone 摇杆模型（倾角加速+阻力滑行+指令延迟，"
                        "高度仍真值定高；建议配 --hold 0.1~0.2；数字不可与旧档混报）；"
                        "cf_blackbox=纯 Python 黑盒（3DGS collector 直接使用；"
                        "Gazebo collect 只校验参数并明确拒绝，不拿 x3 物理冒充）；"
                        "cf_sil=直接驱动 4 个电机（串级 PID+混控，sim-only，"
                        "已退出默认矩阵，仅手动参考；参数见 --cf-sil-params）")
    p.add_argument("--cf-sil-params", default=None,
                   help="cf_sil 参数 JSON（默认 sim/dynamics/cf_sil_v1.json）；"
                        "只能和 --dynamics cf_sil 一起用")
    p.add_argument("--vehicle", choices=tuple(VEHICLES), default=None,
                   help="世界里的载具模型：x3（默认）或 e1_mini；决定位姿话题 /model/<name>/pose "
                        "与 camera_info 的 frame_id。缺省取 scene-meta 的 vehicle，再缺省 x3。"
                        "--dynamics cf_sitl 必须是 e1_mini")
    p.add_argument("--cf-sitl-params", default=None,
                   help="cf_sitl 参数 JSON（默认 sim/dynamics/e1_mini.json，与世界模板同一单源）；"
                        "只能和 --dynamics cf_sitl 一起用")
    p.add_argument("--flight-log", default=None,
                   help="cf_sil 参数辨识用 ULog(.ulg)；需要 pyulog，未安装时先导出 CSV")
    p.add_argument("--flight-log-csv", default=None,
                   help="cf_sil 参数辨识用 CSV（t,qw/qx/qy/qz,wx/wy/wz,thrust/motor,x/y/z）")
    p.add_argument("--no-takeoff", action="store_true")
    p.add_argument("--teacher-planner", choices=("astar", "predictive", "costmap"), default="astar",
                   help="--policy teacher 的 T0/T1 规划器；predictive 需要 cf_stab + scene-meta；"
                        "costmap=测地场代价 argmin（teacher.plan_action_costmap，每行多记 q_costs）")
    p.add_argument("--costmap-tol", type=float, default=25.0,
                   help="costmap 教师的航向罚死区（度）；仅 --teacher-planner costmap")
    p.add_argument("--costmap-lookahead-steps", type=int, default=1,
                   help="costmap 教师的前瞻步数（1=单步 rollout）；仅 --teacher-planner costmap")
    p.add_argument("--rl-checkpoint", default="", help="teacher-rl 离线 PPO 模型路径")
    p.add_argument("--policy", choices=("teacher", "forward", "student", "teacher-nn", "teacher-rl", "rule-oracle"),
                   default="teacher",
                   help="teacher=专家策略；forward=盲策略（恒定前进）；student=模型闭环；"
                        "teacher-nn=特权教师网络闭环；"
                        "rule-oracle=柱子真值算完美列距离 + 固定规则 v0（零训练的规则层上限，需要 --scene-meta）")
    p.add_argument("--student-ckpt", default=None, help="--policy student 的 checkpoint")
    p.add_argument("--teacher-ckpt", default=None,
                   help="--policy teacher-nn 的 PrivilegedTeacher checkpoint")
    p.add_argument("--infer-bin", default=None,
                   help="uavrgb_infer 路径（默认 UAV_RGB_BUILD_DIR 或 build/check）")
    p.add_argument("--stop-on-collision", action="store_true",
                   help="碰撞后立即结束本 episode（闭环评测用）")
    p.add_argument("--stop-at-goal", action="store_true",
                   help="真值 x 首次到/过成功线（最远柱 x + 1.0 m，与 eval 同口径）即结束本集并记"
                        " stopped_on_goal/goal_line_x；需要 --scene-meta，且不能是 waypoint 任务。"
                        "DISTILL §3.1：过线后的帧占 18%%，93%% 视锥内无柱却有 38%% 是 YAW 标签")
    p.add_argument("--tiebreak", choices=TIEBREAKS, default="legacy",
                   help="A* 教师左右近平局（|Q_L−Q_R|<0.5 步）怎么定：legacy=沿用 heap 顺序/err 符号"
                        "（逐位不变）；visible=改向学生视锥内最近柱的反侧（DISTILL §3.3）。"
                        "只对柱子真值 A* 教师（--policy teacher 的 astar 规划器，或 --policy student 的"
                        "影子教师）有意义")
    p.add_argument("--beta", type=float, default=0.0,
                   help="--policy student（DAgger）每步执行教师动作的概率，与 sim/3dgs/collect_3dgs.py"
                        " 同义；0=全由学生执行（现状）。标签始终是教师动作")
    p.add_argument("--dagger-round", type=int, default=0,
                   help="--policy student 写进 manifest 的 DAgger 轮次（溯源用，0=首轮）")
    p.add_argument("--heading-guard", type=int, default=0,
                   help="--policy student 的航向护栏（R1 判别实验）：已执行的左转数减右转数的绝对值达到 N 就朝回转一步；"
                        "0=关闭（默认，行为逐位不变）。被护栏改写的步 executed_by=heading_guard")
    p.add_argument("--student-cond", choices=nav_cond.COND_CHOICES, default=nav_cond.COND_PREV_ACTION,
                   help="--policy student 喂给条件输入（模型 prev_dim=4）的是什么：prev-action=上一步执行的动作（默认，现状）；"
                        "heading-code=航向码 nav_cond.heading_code(已执行 YAW_LEFT−YAW_RIGHT)（W1-9，与 preprocess --cond heading-code 同一个函数）。"
                        "heading-code 时 manifest 记 student_cond，每行记 cond_code；模型必须是 --prev-action 在 heading-code 数据上训的")
    p.add_argument("--action-latency", type=float, default=None, metavar="L",
                   help="把动作派发时刻固定到仿真时间（W1-0）：策略出动作后，等位姿话题的仿真戳 >= 观测戳 + L 秒再派发；"
                        "执行器截止仍是观测戳 + hold，动作生效时长固定为 hold - L，与策略这一步算了多久无关。"
                        "动作就绪时仿真戳已超过 观测戳 + L + 0.02 s 的步记迟到（行 late_dispatch、manifest late_dispatch_steps）。"
                        "开了以后每集起跑前预热 A* 栅格，学生 + beta=0 的影子教师标签挪到派发之后算，manifest 记 action_latency_s"
                        "（preprocess 不许与旧口径混包）。默认关（旧行为逐位不变）。只能配 --window-from observation，且 0 < L < --hold")
    p.add_argument("--rule-hfov-deg", type=float, default=60.0,
                   help="--policy rule-oracle 的列距离视场（度）：60=学生相机（20 列），81=真机 GC4653 的水平视场（27 列）；"
                        "列宽固定 3°，范围 [6, 180]")
    p.add_argument("--rule-version", choices=("v0", "v1"), default="v0",
                   help="--policy rule-oracle 的规则版本：v0=固定规则（无记忆，默认，行为逐位不变）；"
                        "v1=短期障碍记忆 + 候选航向选择（计划 D6，参数见 rule_policy.RULE_V1）")
    p.add_argument("--scene-id", type=int, default=0, help="EpisodeMeta.scene_id（场景隔离关键字段）")
    p.add_argument("--seed", type=int, default=0, help="EpisodeMeta.seed")
    p.add_argument("--episode-id", type=int, default=None, help="EpisodeMeta.episode_id（默认=seed）")
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--save-depth", action="store_true",
                   help="每步另存 depth/depth_%%05d.png（16 位灰度 PNG，毫米，0=无回波）")
    p.add_argument("--version", default="1", help="EpisodeMeta.version（采集器版本）")
    p.add_argument("--scene-meta", default=None,
                   help="scene_gen 产出的场景 JSON；给了就用柱子真值的 carrot planner，"
                        "否则用深度反应式专家")
    args = p.parse_args()
    # NaN <= 0 是 False，只判 <=0 会让 --hold nan 一路跑进 sleep 里挂死。
    if not (math.isfinite(args.hold) and args.hold > 0):
        p.error("--hold must be finite and > 0")
    if args.action_latency is not None:
        if not (math.isfinite(args.action_latency) and args.action_latency > 0):
            p.error("--action-latency 必须是有限正数")
        if args.action_latency >= args.hold:
            p.error(f"--action-latency 必须小于 --hold（动作生效时长 = hold - L；L={args.action_latency:g} hold={args.hold:g}）")
        if args.window_from != "observation":
            p.error("--action-latency 只能配 --window-from observation（执行器截止 = 观测戳 + hold）")
    if args.steps < 1:
        p.error("--steps must be >= 1")
    if args.policy == "student" and not args.student_ckpt:
        p.error("--policy student 需要 --student-ckpt")
    if args.policy == "teacher-nn" and not args.teacher_ckpt:
        p.error("--policy teacher-nn 需要 --teacher-ckpt")
    if args.policy == "teacher-rl" and not args.rl_checkpoint:
        p.error("--policy teacher-rl 需要 --rl-checkpoint")
    if (args.teacher_planner == "predictive" or args.policy == "teacher-rl") and (
            args.dynamics != "cf_stab" or not args.scene_meta):
        p.error("新 Teacher 必须 --dynamics cf_stab 且提供 --scene-meta")
    if args.policy == "teacher-rl" and abs(args.hold - 0.2) > 1e-6:
        p.error("teacher-rl 的训练策略要求 --hold 0.2（5Hz）；其他时长需重新训练")
    if args.policy != "student" and (args.beta != 0.0 or args.dagger_round != 0):
        p.error("--beta/--dagger-round 只能和 --policy student 一起用")
    if not (math.isfinite(args.beta) and 0.0 <= args.beta <= 1.0):
        p.error("--beta 必须在 [0, 1]")
    if args.dagger_round < 0:
        p.error("--dagger-round 必须是非负整数")
    if args.heading_guard < 0 or (args.heading_guard and args.policy != "student"):
        p.error("--heading-guard 必须是非负整数，且只能和 --policy student 一起用")
    if args.student_cond != nav_cond.COND_PREV_ACTION and args.policy != "student":
        p.error("--student-cond 只能和 --policy student 一起用")
    if args.policy != "rule-oracle" and args.rule_hfov_deg != 60.0:
        p.error("--rule-hfov-deg 只能和 --policy rule-oracle 一起用")
    if args.policy != "rule-oracle" and args.rule_version != "v0":
        p.error("--rule-version 只能和 --policy rule-oracle 一起用")
    if args.policy == "rule-oracle":
        if not args.scene_meta:
            p.error("--policy rule-oracle 需要 --scene-meta（用柱子真值算列距离）")
        if not (math.isfinite(args.rule_hfov_deg) and 2 * COL_DEG <= args.rule_hfov_deg <= 180.0):
            p.error(f"--rule-hfov-deg 必须在 [{2 * COL_DEG:g}, 180] 度内")
    if args.stop_at_goal and not args.scene_meta:
        p.error("--stop-at-goal 需要 --scene-meta（成功线由柱子真值定）")
    if args.tiebreak != "legacy" and not (args.scene_meta and (
            (args.policy == "teacher" and args.teacher_planner == "astar") or args.policy == "student")):
        p.error("--tiebreak visible 只对柱子真值 A* 教师有意义：需要 --scene-meta，"
                "且 --policy teacher（astar）或 --policy student")
    if args.teacher_planner == "costmap" and not (args.policy == "teacher" and args.scene_meta):
        p.error("--teacher-planner costmap 需要 --policy teacher 与 --scene-meta（测地场由柱子真值算）")
    if args.teacher_planner != "costmap" and (args.costmap_tol != 25.0 or args.costmap_lookahead_steps != 1):
        p.error("--costmap-tol/--costmap-lookahead-steps 只能和 --teacher-planner costmap 一起用")
    if args.teacher_planner == "costmap" and (args.costmap_lookahead_steps < 1 or args.costmap_tol < 0.0):
        p.error("--costmap-lookahead-steps ≥ 1、--costmap-tol ≥ 0")
    if args.dynamics == "cf_blackbox":
        # 纯模型档只在这里做参数校验：Gazebo collect 没有它的状态闭环，
        # 继续跑会拿 x3 物理冒充 cf_blackbox 数据。3DGS collector 直接 import 模型。
        try:
            blackbox_params = cf_blackbox.CfBlackboxParams.from_json(
                DEFAULT_CF_BLACKBOX_PARAMS)
        except (OSError, ValueError, TypeError, KeyError) as e:
            p.error(f"无法加载 cf_blackbox 参数: {e}")
        if not blackbox_params.calibrated:
            print("警告: cf_blackbox 参数 calibrated=false（占位值，不是实测标定）",
                  file=sys.stderr)
        p.error(
            f"cf_blackbox 参数 {blackbox_params.profile()} 校验通过，但该档是纯 Python "
            "模型，Gazebo collect 未接入；请让 3DGS collector 直接调用 cf_blackbox，"
            "或改用 --dynamics cf_stab/x3_velocity")
    cf_sil_params = None
    if args.dynamics == "cf_sil":
        try:
            cf_sil_params = cf_sil.CfSilParams.from_json(
                args.cf_sil_params or DEFAULT_CF_SIL_PARAMS)
        except (OSError, ValueError, TypeError, KeyError) as e:
            p.error(f"无法加载 cf_sil 参数: {e}")
        if not cf_sil_params.calibrated:
            print("警告: cf_sil 参数 calibrated=false（固件默认+占位值，不是实测标定）",
                  file=sys.stderr)
    elif args.cf_sil_params:
        p.error("--cf-sil-params 只能和 --dynamics cf_sil 一起用")
    cf_sitl_params = None
    if args.dynamics == "cf_sitl":
        try:
            cf_sitl_params = cf_sitl_exec.CfSitlParams.from_json(
                args.cf_sitl_params or cf_sitl_exec.DEFAULT_PARAMS)
        except (OSError, ValueError, TypeError, KeyError) as e:
            p.error(f"无法加载 cf_sitl 参数: {e}")
        if not cf_sitl_params.calibrated:
            print("警告: cf_sitl 参数 calibrated=false（硬件草案/假设值，不是实测标定）",
                  file=sys.stderr)
        if args.no_takeoff:
            p.error("--dynamics cf_sitl 不支持 --no-takeoff：固件必须经 arm_takeoff 进入 READY")
    elif args.cf_sitl_params:
        p.error("--cf-sitl-params 只能和 --dynamics cf_sitl 一起用")
    if args.flight_log and args.flight_log_csv:
        p.error("--flight-log 和 --flight-log-csv 只能二选一")
    log_path = args.flight_log or args.flight_log_csv
    if log_path:
        if args.dynamics != "cf_sil":
            p.error("--flight-log/--flight-log-csv 只用于 --dynamics cf_sil")
        try:
            summary = cf_sil.load_flight_log(log_path)
        except Exception as e:  # noqa: BLE001 - 参数校验失败直接退出
            p.error(f"飞行日志读取失败: {e}")
        print(f"飞行日志（仅接口校验/摘要，未做参数辨识）: {summary}",
              file=sys.stderr)
    if os.path.isdir(args.out) and os.listdir(args.out):
        p.error(f"--out {args.out} 不是空目录；请换新目录（或先清空）")
    os.makedirs(args.out, exist_ok=True)
    if args.save_depth:
        os.makedirs(os.path.join(args.out, DEPTH_DIR), exist_ok=True)
    pillars = None
    scene_meta_sha = None
    scene_gid = None
    scene_version = None
    instruction_text = target_color = relation = waypoint = task_version = None
    stop_radius = None
    scene_meta_obj = {}
    if args.scene_meta:
        with open(args.scene_meta, "rb") as f:
            raw_meta = f.read()
        scene_meta_sha = hashlib.sha256(raw_meta).hexdigest()
        scene_meta_obj = json.loads(raw_meta.decode("utf-8"))
        if args.dynamics == "cf_sitl":
            own_params = os.path.realpath(args.cf_sitl_params or cf_sitl_exec.DEFAULT_PARAMS)
            scene_params = os.path.realpath(scene_meta_obj.get("cf_sitl_params") or cf_sitl_exec.DEFAULT_PARAMS)
            if own_params != scene_params:
                p.error(f"--cf-sitl-params ({own_params}) 与 --scene-meta 里的 cf_sitl_params ({scene_params}) 不一致："
                        "世界插件与 collect.py 必须读同一份参数文件")
        pillars = scene_meta_obj["pillars"]
        scene_version = scene_meta_obj.get("geometry_version")
        instruction_text = scene_meta_obj.get("instruction")
        target_color = scene_meta_obj.get("target_color")
        relation = scene_meta_obj.get("relation")
        waypoint = scene_meta_obj.get("waypoint")
        task_version = scene_meta_obj.get("task_version")
        stop_radius = scene_meta_obj.get("stop_radius")
        if args.student_cond == nav_cond.COND_HEADING_CODE and (waypoint is not None or stop_radius is not None):
            p.error("--student-cond heading-code 只用于航向设定值 H=0 的走廊任务；带 waypoint 的任务（l1/campus）不支持")
        for k, pl in enumerate(pillars):
            if "x" not in pl or "y" not in pl:
                p.error(f"--scene-meta {args.scene_meta}: pillars[{k}] 缺 x/y")
        # 柱子真值随 episode 一起落盘：audit 的 label_recompute 要用它复算标签。
        # 只有 collect_parallel 复制过 scene.json，单跑 collect.py 的集会永远 SKIP。
        write_scene_meta(os.path.join(args.out, "scene.json"), raw_meta)
        # split 由几何组唯一决定；手传的 --split 与它不符会把同几何放进两个 split。
        scene_gid = sm.geometry_gid(pillars)
        want_split = sm.split_for_scene(scene_meta_obj)
        if args.split != want_split:
            p.error(f"--split {args.split} 与几何组 {scene_gid} 的固定 split={want_split} 不一致")
    goal_x = None
    if args.stop_at_goal:
        # waypoint 任务的成功是停在半径内（goal-stop），不是过线；两套终止条件混在一集里说不清。
        if stop_radius or waypoint is not None:
            p.error("--stop-at-goal 只用于走廊穿越任务；waypoint 任务按 stop_radius 的 goal-stop 结束")
        if not pillars:
            p.error("--stop-at-goal 需要 scene-meta 里至少一根柱（成功线 = 最远柱 x + 1.0 m）")
        goal_x = goal_line_x(pillars)
    # 载具：显式 --vehicle > scene-meta.vehicle（scene_gen --vehicle 写入）> x3。
    # cf_sitl 的固件只在 e1_mini 世界里有插件可收；e1_mini 世界也没有 X3 的速度控制器。
    vehicle = args.vehicle or scene_meta_obj.get("vehicle") or DEFAULT_VEHICLE
    if vehicle not in VEHICLES:
        p.error(f"scene-meta 的 vehicle={vehicle!r} 不在 {sorted(VEHICLES)}；请显式 --vehicle")
    need_vehicle = DYNAMICS_VEHICLE.get(args.dynamics)
    if need_vehicle and vehicle != need_vehicle:
        p.error(f"--dynamics {args.dynamics} 需要 --vehicle {need_vehicle}（当前 {vehicle}）")
    if vehicle != DEFAULT_VEHICLE and not need_vehicle:
        p.error(f"--vehicle {vehicle} 只能配 --dynamics "
                f"{[d for d, v in DYNAMICS_VEHICLE.items() if v == vehicle]}（当前 {args.dynamics}）")
    # 教师按 (载具, 任务档位) 切机身半宽/规划余量：柱阵任务走 base 档（x3 0.5/0.36、e1_mini 0.40/0.30），
    # forest_e1 走 0.20/0.15 + 圆柱圆判定；档位由 scene.json 的 family/geometry_version 推断，
    # manifest.teacher_version 也按档位写（forest_e1 与 base 的标签算法不同，版本必须能区分）。
    import teacher as _teacher
    teacher_task = task_of_scene(scene_meta_obj)
    _teacher.set_vehicle(vehicle, teacher_task)
    # 标签来自哪个教师必须进 manifest：两种策略产出的数据长得一模一样，
    # 事后只能靠这个字段区分（旧版没有任何记录）。
    if args.policy == "forward":
        teacher_kind, planner_kind = "forward", None
    elif args.policy == "rule-oracle":
        # 动作由柱子真值列距离 + 规则 v0 给出，不是教师标签；audit 的 label_recompute 不认这个 teacher，也不该认。
        teacher_kind, planner_kind = "rule_oracle", f"rule_{args.rule_version}"
    elif args.policy == "student":
        # 有柱子真值时是 DAgger 影子教师：标签口径与 --policy teacher 完全一样（plan_action/astar），
        # 只是 executed_by=student（与 sim/3dgs/collect_3dgs.py 同义）。没真值时标签只能是学生自己的输出。
        teacher_kind, planner_kind = ("plan_action", "astar") if pillars else ("student", None)
    elif args.policy == "teacher-rl":
        teacher_kind, planner_kind = "teacher_rl", "ppo_discrete4"
    elif args.policy == "teacher-nn":
        # 标签来自训练好的特权教师网络，必须与 plan_action/expert_action 区分。
        teacher_kind, planner_kind = "teacher_nn", None
    elif pillars:
        teacher_kind = "plan_action"
        planner_kind = args.teacher_planner
    else:
        teacher_kind, planner_kind = "expert_action", None
        print("警告: 没给 --scene-meta，退回深度反应式专家；manifest.teacher=expert_action",
              file=sys.stderr)

    actions_path = os.path.join(args.out, "actions.jsonl")
    oracle_path = os.path.join(args.out, "oracle.jsonl")
    actions_f = open(actions_path, "w")
    oracle_f = open(oracle_path, "w")
    subs = {}
    publisher = None
    student = None
    teacher_nn = None
    predictive_teacher = None
    rl_teacher = None
    if args.policy == "teacher" and args.teacher_planner == "predictive":
        if not pillars or not isinstance(waypoint, dict):
            p.error("predictive teacher 需要 scene-meta 的 pillars/waypoint")
        from teacher_predictive import PredictiveTeacher
        predictive_teacher = PredictiveTeacher(pillars, waypoint)
    # costmap 教师：只在 planner_kind == "costmap" 时给 collect_one 传选项（其余档位 None，行为不变）。
    costmap_opts = ({"tol": args.costmap_tol, "lookahead_steps": args.costmap_lookahead_steps}
                    if planner_kind == "costmap" else None)
    if planner_kind == "costmap":
        # manifest.teacher_version 写的是 TEACHER_COST_VERSION：规则档（B2 过柱阵后目标取线开，A* 重试 costmap 不用）
        # 按它切，采集时生效的规则与 audit 按 manifest 版本复算的规则一致（forest_e1 档的 costmap 集也一样）。
        _teacher.set_vehicle(vehicle, teacher_task, version=TEACHER_COST_VERSION)
    text_ids = None
    cam_info = None
    rows_written = 0
    all_rgb_ok = True
    all_pose_ok = True
    error = None
    completed = False
    stopped_on_collision = False
    stopped_on_goal = False
    prev_action = None
    prev_executed = None
    net_yaw = 0                # 已执行的 YAW_LEFT 数减 YAW_RIGHT 数（--heading-guard 与 rule-oracle 用）
    last_turn = None           # 最近一次执行的转向方向 YAW_LEFT / YAW_RIGHT（rule-oracle 平局时沿用）
    rule_mem = []              # rule-oracle v1 的短期障碍记忆 [(世界 X, 世界 Y, 产生步号)]；每集（每次进程）从空开始
    prev_rgb_ts = 0
    prev_until_ns = None       # W1-13 A1：上一步的执行器截止，collect_one 用它算 safe_gap_s（首步 None）
    timings = []
    late_dispatch_steps = 0    # --action-latency：动作就绪时已超过 观测戳 + L + 0.02 s 的步数
    stall_steps = 0            # W1-13 A3：停顿步数（相邻观测间隔 > 名义步周期 + 0.2 s）；默认只在行里记，manifest 只在 --action-latency 下写
    # DAgger（--policy student）：β 混合的随机源按 seed 固定，同 seed 可复现哪些步换成了教师执行。
    dagger = None
    if args.policy == "student" and pillars:
        with open(args.student_ckpt, "rb") as f:
            ckpt_sha = hashlib.sha256(f.read()).hexdigest()
        dagger = {"executed_by": "student", "dagger_round": args.dagger_round, "beta": args.beta,
                  "student_ckpt": os.path.abspath(args.student_ckpt), "student_ckpt_sha256": ckpt_sha,
                  "n_expert_executed": 0}
    rng = random.Random(args.seed) if dagger is not None else None
    dyn = CfStab() if args.dynamics == "cf_stab" else None
    sil = (cf_sil.CfSilMotorThread(cf_sil_params)
           if cf_sil_params is not None else None)
    if cf_sitl_params is not None:
        sil = cf_sitl_exec.CfSitlExecutor(
            cf_sitl_params, hold_s=args.hold,
            episode=args.episode_id if args.episode_id is not None else args.seed)
    rgb_frame_id = VEHICLES[vehicle]["rgb_frame_id"]
    try:
        subs = open_subs(vehicle)
        not_ready = wait_ready(subs)
        if not_ready:
            raise RuntimeError(f"常驻订阅未就绪: {not_ready}")
        cam_info = read_camera_info(subs["camera_info"], frame_id=rgb_frame_id)
        if cam_info:
            print(f"相机内参: fx={cam_info['fx']:.3f} fy={cam_info['fy']:.3f} "
                  f"cx={cam_info['cx']:.3f} cy={cam_info['cy']:.3f} "
                  f"({cam_info['width']}x{cam_info['height']}, {rgb_frame_id})",
                  file=sys.stderr)
        else:
            print("警告: 没读到 /uav/camera_info 的 RGB 内参，manifest 里 fx/fy/cx/cy 保持 null",
                  file=sys.stderr)
        if sil is None:
            try:
                publisher = gz_io.open_twist_publisher(TWIST_TOPIC)
                if getattr(publisher, "warning", None):
                    print(f"警告: {publisher.warning}", file=sys.stderr)
                print(f"指令发布: {publisher.mode}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001 - 任何建不起来都直接失败
                raise RuntimeError(
                    f"常驻 twist publisher 不可用（{e}）；定量采集不做 gz topic -p CLI"
                    f" 回退（每条指令约 +1.0 s，动作窗口会失真）") from e
        else:
            print(f"{args.dynamics} 执行器: {sil.params.profile()}（calibrated="
                  f"{sil.params.calibrated}，altitude={sil.params.altitude_assumption}）",
                  file=sys.stderr)

        if args.policy == "student":
            from instruction import tokenize
            from student import StudentPolicy
            print("学生策略: 启动推理进程…", file=sys.stderr)
            text_ids = tokenize(instruction_text) if instruction_text else None
            student = StudentPolicy(args.student_ckpt, args.infer_bin, text_ids=text_ids)
            cond_check = nav_cond.check_student_cond(args.student_cond, args.student_ckpt,
                                                     getattr(student, "prev_dim", 0))
            if cond_check["status"] == "mismatch":
                error = cond_check["message"]
                p.error(cond_check["message"])
            if cond_check["status"] == "unverified":
                print(f"警告: 条件输入一致性未核实（--student-cond {args.student_cond}）：{cond_check['message']}",
                      file=sys.stderr)
        elif args.policy == "teacher-rl":
            from stable_baselines3 import PPO
            rl_teacher = PPO.load(args.rl_checkpoint, device="cpu")
            print("警告: teacher-rl 是轻量物理仿真训练的实验策略，Gazebo 效果尚未验证", file=sys.stderr)
        elif args.policy == "teacher-nn":
            from instruction import tokenize
            from teacher_policy import TeacherPolicy
            print(f"teacher-nn 策略: 加载 {args.teacher_ckpt}…", file=sys.stderr)
            text_ids = tokenize(instruction_text) if instruction_text else None
            teacher_nn = TeacherPolicy(args.teacher_ckpt)

        if args.action_latency is not None and pillars and planner_kind == "astar":
            # 每集（每个 collect 进程）起跑前把整张 A* 栅格的「可走」填满：否则第一步的 A* 比后续步慢约 0.45 s（K=3），
            # 第一步必迟到。教师的 A* 是动作本身，学生的 A* 是影子标签。学生推理的首帧开销由 StudentPolicy 构造时的
            # _warmup 吃掉（每集一次，已在上面），这里不再重复。放在起飞之前：不多悬停一秒。
            t_warm = time.perf_counter()
            n_cells = _teacher.warm_free_cells(pillars, include_retries=True)
            print(f"预热 A* 栅格：{n_cells} 格，{time.perf_counter() - t_warm:.2f} s", file=sys.stderr)

        contact_seq0 = _contact_seq(subs["contact"])
        if cf_sitl_params is not None:
            print("cf_sitl: reset + arm_takeoff，等固件 READY…", file=sys.stderr)
            contact_seq0 = take_off_cf_sitl(subs["pose"], sil, subs["contact"])
        elif sil is not None:
            print("cf_sil: 停用 MulticopterVelocityControl 并启动电机线程…",
                  file=sys.stderr)
            sil.start()
            if args.no_takeoff:
                # --no-takeoff 在 cf_sil 下表示「保持当前高度」，不是关电机。
                pos0 = _pose_parts(subs["pose"].latest())[0] \
                    if subs["pose"].latest() is not None else None
                if isinstance(pos0, dict) and pos0.get("z") is not None:
                    sil.set_altitude_target(pos0["z"])
                    sil.set_action(SAFE_NEUTRAL)   # 上电并保持当前高度
                    print(f"cf_sil --no-takeoff: 保持当前高度 z={pos0['z']:.2f} m",
                          file=sys.stderr)
            else:
                print("起飞中…", file=sys.stderr)
                contact_seq0 = take_off_cf_sil(
                    subs["pose"], sil, subs["contact"],
                    z_target=sil.params.altitude_target)
        elif not args.no_takeoff:
            print("起飞中…", file=sys.stderr)
            contact_seq0 = take_off(subs["pose"], publisher, subs["contact"])

        stop_streak = 0
        goal_stop = bool(stop_radius)
        for i in range(args.steps):
            r, reason = collect_one(args.out, i, args.hold, args.policy, subs,
                                    publisher, prev_action, prev_rgb_ts, pillars, student,
                                    global_plan=True,
                                    window_from=args.window_from, waypoint=waypoint,
                                    stop_at_waypoint=goal_stop,
                                    stop_radius=stop_radius or 0.5,
                                    save_depth=args.save_depth,
                                    teacher_nn=teacher_nn, text_ids=text_ids,
                                    contact_seq0=contact_seq0, dyn=dyn,
                                    predictive_teacher=predictive_teacher,
                                    rl_teacher=rl_teacher, sil=sil,
                                    tiebreak=args.tiebreak, beta=args.beta, rng=rng,
                                    prev_executed=prev_executed, costmap_opts=costmap_opts,
                                    heading_guard=args.heading_guard, net_yaw=net_yaw,
                                    rule_hfov_deg=args.rule_hfov_deg, last_turn=last_turn,
                                    rule_version=args.rule_version, rule_mem=rule_mem,
                                    action_latency_s=args.action_latency,
                                    student_cond=args.student_cond,
                                    prev_until_ns=prev_until_ns)
            if r is None:
                error = f"step {i}: {reason}"
                print(error + "，终止", file=sys.stderr)
                break
            prev_action = r["action_expert"]   # 下一步的滞回基准
            prev_executed = r["action_executed"]
            net_yaw += (prev_executed == YAW_LEFT) - (prev_executed == YAW_RIGHT)
            if prev_executed in (YAW_LEFT, YAW_RIGHT):
                last_turn = prev_executed
            if dagger is not None and r.get("executed_by") == "expert":
                dagger["n_expert_executed"] += 1
            # 只推进到本步已检查的序号，步间漂移产生的事件归下一步。
            contact_seq0 = r["contact_seq_end"]
            prev_rgb_ts = r["rgb_ts"]
            prev_until_ns = r.get("until_ns")
            gap_ns = r["rgb_ts"] - r["depth_ts"] if r["depth_ts"] else None
            # 配对残差用**观测帧**位姿（pose_before_ts），action_start_ns 是派发时刻。
            pose_gap_ns = r["pose_before_ts"] - r["rgb_ts"]
            actions_f.write(json.dumps(
                action_row(r | {"depth_rgb_gap_ns": gap_ns,
                                "pose_rgb_gap_ns": pose_gap_ns},
                           save_depth=args.save_depth)) + "\n")
            actions_f.flush()
            oracle_f.write(json.dumps({"step": r["step"],
                                       "timestamp_ns": r["pose_ts"],
                                       "pose_gt": r["pose_gt"],
                                       "action_start_ns": r["action_start_ns"],
                                       "action_end_ns": r["action_end_ns"],
                                       # 特权通道：窗口末真值速度（只有 cf_sitl 有，其余 null）
                                       "speed_mps": r["speed_after_mps"]}) + "\n")
            oracle_f.flush()
            rows_written += 1
            late_dispatch_steps += bool(r.get("late_dispatch"))
            stall_steps += is_stall_gap(r.get("obs_gap_s"), args.hold)
            timings.append(r["timing"] | {"depth_gap_ns": gap_ns,
                                          "pose_gap_ns": pose_gap_ns,
                                          "rgb_before_start": r["rgb_ts"] < r["action_start_ns"]})
            all_rgb_ok = all_rgb_ok and r["rgb_ok"]
            all_pose_ok = all_pose_ok and r["pose_gt"] is not None
            dx = "?" if r["displacement_m"] is None else f"{r['displacement_m']:.3f}m"
            dyaw = "?" if r["dyaw_deg"] is None else f"{r['dyaw_deg']:.1f}deg"
            executed = ("" if r.get("executed_by") is None
                        else f" exec={ACTIONS[r['action_executed']]}({r['executed_by']})")
            if r.get("rule_entry") is not None:
                executed += f" rule={r['rule_entry']} dmin={min(r['rule_cols']):.2f} n={net_yaw}"
                if r.get("rule_mem_n") is not None:
                    executed += f" mem={r['rule_mem_n']} clear={''.join('1' if c else '0' for c in r['rule_clear'])}"
            print(f"step {i}: {ACTIONS[r['action_expert']]}{executed} pos={r['pose_gt']} "
                  f"dx={dx} dyaw={dyaw} action_wall={r['action_wall_s']:.2f}s "
                  f"sim_dt={r['sim_dt']:.2f}s "
                  f"collision={r['collision']}({r['src'] or '-'}) rgb={r['rgb_ok']} "
                  f"rgb_ts={r['rgb_ts']} action_start_ts={r['action_start_ns']} "
                  f"rgb<start={r['rgb_ts'] < r['action_start_ns']} "
                  f"depth_gap={'-' if gap_ns is None else f'{gap_ns/1e6:.0f}ms'} "
                  f"pose_gap={'-' if pose_gap_ns is None else f'{pose_gap_ns/1e6:.0f}ms'} "
                  f"io={(r['timing']['io']):.2f}s step={(r['timing']['total']):.2f}s",
                  file=sys.stderr)
            # goal-stop：连续 3 步 SAFE **且在 waypoint 半径内**才算停稳，提前结束，
            # 避免后 20+ 步全是 SAFE 把类别分布拉偏。
            if goal_stop and r["action_executed"] == SAFE_NEUTRAL:
                stop_streak = (stop_streak + 1
                               if stopped_at_waypoint(r, waypoint, stop_radius) else 0)
            else:
                stop_streak = 0
            if goal_stop and stop_streak >= 3:
                stopped_on_goal = True
                print(f"step {i}: goal-stop 连续 3 步 SAFE，提前结束", file=sys.stderr)
                break
            # --stop-at-goal：本步动作后真值 x 已到成功线，本步保留（它是过线的那一步），不再采下一步。
            if goal_x is not None and crossed_goal_line(r, goal_x):
                stopped_on_goal = True
                print(f"step {i}: 真值 x={r['pose_gt']['x']:.2f} ≥ 成功线 {goal_x:.2f}，"
                      f"按 --stop-at-goal 结束", file=sys.stderr)
                break
            if args.stop_on_collision and r["collision"]:
                stopped_on_collision = True
                print(f"step {i}: 碰撞，按 --stop-on-collision 结束", file=sys.stderr)
                break
        completed = rows_written == args.steps or stopped_on_goal

        if timings:
            def avg(key):
                return sum(t[key] for t in timings) / len(timings)
            gaps = [abs(t["depth_gap_ns"]) / 1e6 for t in timings
                    if t["depth_gap_ns"] is not None]
            pgaps = [t["pose_gap_ns"] / 1e6 for t in timings
                     if t["pose_gap_ns"] is not None]
            m1 = sum(1 for t in timings if t["rgb_before_start"])
            print(f"每步耗时: total={avg('total'):.3f}s  I/O={avg('io'):.3f}s  "
                  f"动作窗口={avg('window'):.3f}s  "
                  f"发布开销={avg('block') - avg('window'):.3f}s  "
                  f"其他={avg('tail'):.3f}s", file=sys.stderr)
            print(f"M1 rgb_ts<action_start_ts: {m1}/{len(timings)}", file=sys.stderr)
            if gaps:
                print(f"配对残差 |rgb_ts-depth_ts|: min={min(gaps):.1f}ms "
                      f"median={_median(gaps):.1f}ms max={max(gaps):.1f}ms",
                      file=sys.stderr)
            if pgaps:
                print(f"配对残差 pose_ts-rgb_ts: min={min(pgaps):.1f}ms "
                      f"median={_median(pgaps):.1f}ms max={max(pgaps):.1f}ms",
                      file=sys.stderr)
    except BaseException as e:
        if error is None:
            error = f"{type(e).__name__}: {e}"
        raise
    finally:
        # M3: 无论正常结束、深度失败、异常还是 Ctrl-C，先发零速/停桨再收尾
        # （不能等订阅关闭，否则中断时可能带着最后一条动作指令多飞几秒）。
        try:
            stop_executor(publisher, sil)
        finally:
            if publisher is not None:
                publisher.close()
            if sil is not None:
                sil.stop()
        close_subs(subs)
        if student is not None:
            try:
                student.close()
            except Exception as e:  # noqa: BLE001 - 收尾失败要写进 manifest，不能静默
                if error is None:
                    error = f"学生推理收尾失败: {e}"
                print(f"警告: {error}", file=sys.stderr)
        actions_f.close()
        oracle_f.close()
        episode_id = args.episode_id if args.episode_id is not None else args.seed
        camera = {"width": 320, "height": 240,
                  "fx": None, "fy": None, "cx": None, "cy": None}
        if cam_info:
            camera.update({k: cam_info[k] for k in
                           ("width", "height", "fx", "fy", "cx", "cy")})
        manifest = {
            "schema": "navlab.episode.v1",
            "episode_id": episode_id,
            "scene_id": args.scene_id,
            "seed": args.seed,
            "version": args.version,
            "split": args.split,
            # 几何组 ID：preprocess 用它拦住「同几何跨 split」，手写 split 也绕不过。
            "geometry_gid": scene_gid,
            # 几何语义版本：v2/v3 的 gid 天然不同，只查 gid 交集查不出混包。
            "geometry_version": scene_version,
            **{key: scene_meta_obj.get(key) for key in sm.PROVENANCE_KEYS},
            # 动力学由采集器决定而非场景：cf_stab/cf_sil 覆盖 scene 的 x3_v1，
            # preprocess 只比 dynamics_profile，据此拒绝混包。
            **({"dynamics_profile": dyn.profile(), "dynamics_params": dyn.params()}
               if dyn is not None else {}),
            # cf_sil 已退出 collect_parallel / eval 默认矩阵，只在 collect.py 手动可用；
            # 下游看到 experimental 就不要把它并进默认训练集。
            "dynamics_experimental": args.dynamics in DYNAMICS_EXPERIMENTAL,
            **({"dynamics_profile": sil.params.profile(),
                "dynamics_params": sil.params.manifest(),
                "altitude_assumption": sil.params.altitude_assumption}
               if sil is not None else {}),
            # 采集器选的动力学档与世界里的载具；x3_velocity/x3 是历史默认。
            "dynamics": args.dynamics,
            "vehicle": vehicle,
            # cf_sitl：参数单源 sha 与标定位提到顶层，下游不用翻 dynamics_params。
            **({"params_sha256": sil.params.sha256, "calibrated": sil.params.calibrated,
                "takeoff_failsafe_retries": sil.takeoff_failsafes}
               if cf_sitl_params is not None else {}),
            # 飞行日志只做接口校验/摘要，不参与本次控制参数；留路径供事后追溯。
            **({"flight_log": os.path.abspath(log_path)} if log_path else {}),
            # 标签由哪个教师产生：forward=盲策略对照，expert_action=深度反应式，
            # plan_action=柱子真值 carrot planner，teacher_nn=特权教师网络。
            # 缺这个字段就无法事后区分数据集。
            "teacher": teacher_kind,
            # 教师实现版本：改规划参数/前向安全门后必须能区分新旧标签。
            "teacher_version": ("uavrgb.predictive_teacher.v1" if planner_kind == "predictive"
                                else TEACHER_COST_VERSION if planner_kind == "costmap"
                                else "uavrgb.rlteacher.v1" if teacher_kind == "teacher_rl"
                                else teacher_version(teacher_task) if teacher_kind in ("plan_action",
                                                                                   "expert_action") else None),
            # plan_action 的具体实现：astar=全局路径（普通集），plan_action=局部
            # （safe 出生集，保留真 SAFE 标签），costmap=测地场代价 argmin。audit 复算必须按这个字段选。
            "planner": planner_kind,
            # costmap 教师的两个显式参数（BRIEF_ALGO §1 允许的变体：tol=0、两步前瞻）；其他 planner 不写。
            **({"costmap_tol_deg": args.costmap_tol, "costmap_lookahead_steps": args.costmap_lookahead_steps}
               if planner_kind == "costmap" else {}),
            # A* 教师近平局的决定方式（--tiebreak）：legacy 与旧标签逐位相同；visible 的标签多了
            # 「按可见柱定左右」这一步，audit 复算与训练数据溯源都要按它选。
            "teacher_tiebreak": args.tiebreak,
            # DAgger 影子教师（--policy student + scene-meta）：谁执行了动作、β、轮次、学生 ckpt。
            # 字段名与 sim/3dgs/collect_3dgs.py 一致；教师自采不写这些键，preprocess 按 teacher/0/1.0 补。
            **(dagger or {}),
            # 航向护栏只在开了的时候写键：默认 manifest 逐位不变。
            **({"heading_guard": args.heading_guard} if args.heading_guard else {}),
            # 航向码（W1-9）：学生喂的是 heading-code 时才写键（默认 prev-action 不写，manifest 逐位不变）。
            **({"student_cond": args.student_cond} if args.student_cond != nav_cond.COND_PREV_ACTION else {}),
            # rule-oracle 的视场与规则参数同样只在用到时写键（预注册的参数随每集 manifest 留证）。
            **({"rule_oracle": ({"hfov_deg": args.rule_hfov_deg, "params": RULE_V0} if args.rule_version == "v0"
                                else {"version": args.rule_version, "hfov_deg": args.rule_hfov_deg, "params": RULE_V1})}
               if args.policy == "rule-oracle" else {}),
            "rl_checkpoint": (os.path.abspath(args.rl_checkpoint) if args.policy == "teacher-rl" else None),
            "experimental_policy": args.policy == "teacher-rl" or planner_kind == "predictive",
            # hold 是**仿真时间**秒（D1 起）：窗口按位姿戳走满这个数才发零速，
            # 与 RTF 无关。同 scene 的 0.5 与 1.0 两批数据仍靠它区分。
            "hold_s": args.hold,
            "window_from": args.window_from,
            # W1-0：只在给了 --action-latency 时写键（默认 manifest 逐位不变）。preprocess 把 action_latency_s 当口径键，
            # 与旧数据、与不同 L 的数据都不能混包；late_dispatch_steps > 0 的集在评测里算基础设施失败。
            # W1-13 A3：stall_steps 同样只在 --action-latency 下写（含停顿步的集在评测里与含迟到步的集同样算基础设施失败）。
            **({"action_latency_s": args.action_latency, "late_dispatch_steps": late_dispatch_steps,
                "stall_steps": stall_steps}
               if args.action_latency is not None else {}),
            "task_version": task_version,
            "instruction": instruction_text,
            "target_color": target_color,
            "relation": relation,
            "waypoint": waypoint,
            "stop_radius": stop_radius,
            "scene_meta_sha256": scene_meta_sha,
            "actions_ref": "actions.jsonl",
            "oracle_ref": "oracle.jsonl",
            "camera": camera,
            # 内参来源：/uav/camera_info 的 RGB publisher；没读到时为 null。
            "camera_source": "gazebo_camera_info" if cam_info else None,
            # 只按实际完成情况填，失败/中断一律 false（M3）。
            "validity": {
                "camera_verified": bool(completed and rows_written > 0 and all_rgb_ok),
                "physics_verified": bool(completed and rows_written > 0 and all_pose_ok),
                "episode_executed": bool(completed and rows_written > 0),
            },
            "num_steps": rows_written,
            "performance": performance_summary(timings),
            "render_backend_requested": os.environ.get("UAVRGB_RENDER_BACKEND", "external"),
            "requested_steps": args.steps,
            # 评测集会在碰撞处提前结束；validity 仍按「跑满请求步数」严格填，
            # 下游据此区分「采集完整」与「评测终止」，不要混淆。
            "stopped_on_collision": stopped_on_collision,
            "stopped_on_goal": stopped_on_goal,
            # --stop-at-goal 的成功线 x（None = 没开）：audit 据此放行短集并核对最后一步确实过了线。
            "goal_line_x": goal_x,
            "error": error,
        }
        if args.save_depth:
            # 下游按 depth_ref 模板拼路径；depth_file 行内字段优先，两者保持一致。
            manifest["depth_ref"] = DEPTH_REF
            manifest["depth_format"] = "png16_mm"
        write_manifest(os.path.join(args.out, "episode_manifest.json"), manifest)
        print(f"写入 {rows_written}/{args.steps} 步 -> {args.out} "
              f"(episode_id={episode_id} scene_id={args.scene_id} seed={args.seed} "
              f"split={args.split} completed={completed} error={error})", file=sys.stderr)
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
