#!/usr/bin/env python3
"""cf_sil：sim-only 的 CF-Drone 风格载具骨架（不是 CF-Drone SIL）。

控制链在伴飞电脑里：四动作 -> 速度/高度设定点 -> 姿态 -> 角速率 PID -> 混控
-> 4 个电机转速，直接发布 `/X3/gazebo/command/motor_speed`（gz.msgs.Actuators）。
Gazebo 仍负责刚体物理和 4 个 MulticopterMotorModel；`MulticopterVelocityControl`
由调用方在 `/X3/enable` 上发 false 停用。

为什么不是 CF-Drone SIL：control.ino 只有 STAB/ACRO，没有速度环和位置环；
这里的高度保持用的是 Gazebo 真值 odometry 的位置 PID（altitude_assumption =
companion_hold），属于仿真假设，不能当成固件能力。

归一化约定（对齐 CF-Drone control.ino）：
  油门/电机输出 u ∈ [0,1]；本骨架按线性 ESC->转速映射发布 omega = u * max_omega
  （rad/s）。真机的油门->转速/推力曲线未实测，JSON 的 hardware.thrust_curve 为
  null，calibrated=false。
  kf  N/(rad/s)^2，thrust = kf * omega^2
  km  m，drag torque = km * thrust（MulticopterMotorModel 的 momentConstant）
  max_rpm 机械转速；发布给 Gazebo 前换算成 rad/s，并受 SDF maxRotVelocity 限制
  Actuators.velocity  rad/s（gz.msgs.Actuators 的 velocity 字段）

纯 stdlib math；不依赖 numpy。动作值 0..3 与 teacher.py 一致。
"""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time

from teacher import (SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT,
                     ACTIONS)

G = 9.81
CF_SIL_PROFILE = "cf_sil_v1"
ALTITUDE_ASSUMPTION = "companion_hold"
VALID_ACTIONS = (SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT)
AXES = ("roll", "pitch", "yaw")


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _f(d, key, default=0.0):
    try:
        v = d.get(key, default)
    except AttributeError:
        return float(default)
    return float(default if v is None else v)


def _triple(value, name):
    if isinstance(value, (int, float)):
        return (float(value),) * 3
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} 必须是数字或 3 个数字，得到 {value!r}") from e
    if len(vals) != 3:
        raise ValueError(f"{name} 必须是数字或 3 个数字，得到 {value!r}")
    return vals


def _stamp_ns(msg):
    s = (msg.get("header") or {}).get("stamp") or {}
    return int(s.get("sec", 0)) * 10**9 + int(s.get("nsec", 0))


def parse_odometry(msg):
    """gz.msgs.Odometry(JSON) -> 统一 dict。

    OdometryPublisher 的 linear/angular 都在机体系（源码 CalculateAngularVelocity
    用当前姿态的 RotateVectorReverse），所以速度/角速率误差不需要再做旋转。
    """
    if not isinstance(msg, dict):
        raise ValueError(f"odometry 不是 dict: {type(msg).__name__}")
    pose = msg.get("pose") or {}
    pos = pose.get("position") or {}
    ori = pose.get("orientation") or {}
    if "z" not in pos or "w" not in ori:
        raise ValueError("odometry 缺 pose.position/pose.orientation")
    twist = msg.get("twist") or {}
    lin = twist.get("linear") or {}
    ang = twist.get("angular") or {}
    return {
        "stamp_ns": _stamp_ns(msg),
        "position": (_f(pos, "x"), _f(pos, "y"), _f(pos, "z")),
        "orientation": tuple(_f(ori, k) for k in ("x", "y", "z", "w")),
        "linear": (_f(lin, "x"), _f(lin, "y"), _f(lin, "z")),
        "angular": (_f(ang, "x"), _f(ang, "y"), _f(ang, "z")),
    }


def quat_to_euler(q):
    """四元数 (x,y,z,w) -> (roll, pitch, yaw)，ZYX 顺序（与 collect._yaw_deg 一致）。"""
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n <= 0.0 or not math.isfinite(n):
        raise ValueError(f"非法四元数: {q!r}")
    x, y, z, w = x / n, y / n, z / n, w / n
    roll = math.atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
    pitch = math.asin(_clamp(2.0 * (w*y - z*x), -1.0, 1.0))
    yaw = math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    return roll, pitch, yaw


class _PID:
    """一维 PID；rate 环用 D-on-measurement，避免设定点阶跃造成输出尖峰。"""

    def __init__(self, kp, ki=0.0, kd=0.0, i_lim=0.0, d_on_measurement=False,
                 d_alpha=1.0):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.i_lim = abs(float(i_lim))
        self.d_on_measurement = bool(d_on_measurement)
        self.d_alpha = _clamp(float(d_alpha), 0.0, 1.0)
        self.i = 0.0
        self.prev_error = None
        self.prev_meas = None
        self.d_filt = None

    def reset(self):
        self.i = 0.0
        self.prev_error = None
        self.prev_meas = None
        self.d_filt = None

    def update(self, error, dt, measurement=None):
        p = self.kp * error
        d = 0.0
        if dt > 0.0 and math.isfinite(dt):
            if self.ki:
                self.i = _clamp(self.i + self.ki * error * dt,
                                -self.i_lim, self.i_lim)
            if self.kd:
                if self.d_on_measurement and measurement is not None:
                    if self.prev_meas is not None:
                        raw = -self.kd * (measurement - self.prev_meas) / dt
                    else:
                        raw = 0.0
                elif self.prev_error is not None:
                    raw = self.kd * (error - self.prev_error) / dt
                else:
                    raw = 0.0
                if self.d_alpha < 1.0:
                    self.d_filt = (raw if self.d_filt is None
                                   else self.d_filt + self.d_alpha * (raw - self.d_filt))
                    d = self.d_filt
                else:
                    d = raw
        self.prev_error = error
        if measurement is not None:
            self.prev_meas = measurement
        return p + self.i + d


def _gains(section, name):
    if not isinstance(section, dict):
        raise ValueError(f"pid.{name} 必须是 object，得到 {section!r}")
    if "kp" in section:
        kp = float(section["kp"])
    elif "p" in section:
        kp = float(section["p"])
    else:
        raise ValueError(f"pid.{name} 缺 kp/p")
    ki = float(section.get("ki", section.get("i", 0.0)) or 0.0)
    kd = float(section.get("kd", section.get("d", 0.0)) or 0.0)
    ilim = float(section.get("i_lim", section.get("I_LIM", 0.0)) or 0.0)
    return (kp, ki, kd, abs(ilim))


def _gains3(section, name):
    """支持 {roll:{...},pitch:{...},yaw:{...}} 或单个标量 PID。"""
    if not isinstance(section, dict):
        raise ValueError(f"pid.{name} 必须是 object，得到 {section!r}")
    if any(axis in section for axis in AXES):
        out = {}
        for axis in AXES:
            if axis not in section:
                raise ValueError(f"pid.{name}.{axis} 缺失")
            out[axis] = _gains(section[axis], f"{name}.{axis}")
        return out
    g = _gains(section, name)
    return {axis: g for axis in AXES}


class CfSilParams:
    """JSON 载具/控制参数。profile = cf_sil_v1:<sha8>，参数一变档名就变。"""

    def __init__(self, data, sha256=None, path=None):
        if not isinstance(data, dict):
            raise ValueError("cf_sil 参数必须是 JSON object")
        self.data = data
        self.path = path
        self.profile_name = str(data.get("profile_version", CF_SIL_PROFILE))
        if not self.profile_name.startswith("cf_sil"):
            raise ValueError(f"profile_version 必须是 cf_sil*，得到 {self.profile_name!r}")
        self.mass = _f(data, "mass")
        self.inertia = tuple(float(v) for v in data.get("inertia", ()))
        self.arm_length = _f(data, "arm_length")
        self.kf = _f(data, "kf")
        self.km = _f(data, "km")
        self.motor_tau_up = _f(data, "motor_tau_up")
        self.motor_tau_down = _f(data, "motor_tau_down")
        self.max_rpm = _f(data, "max_rpm")
        self.drag = _f(data, "drag")
        self.max_tilt = math.radians(_f(data, "max_tilt_deg"))
        self.max_rate = _triple(data.get("max_rate_rad_s"), "max_rate_rad_s")
        self.control_hz = _f(data, "control_hz")
        self.latency_s = _f(data, "latency_s")
        self.calibrated = bool(data.get("calibrated", False))
        self.altitude_assumption = str(
            data.get("altitude_assumption", ALTITUDE_ASSUMPTION))
        self.altitude_target = _f(data, "altitude_target_m")
        self.forward_speed = _f(data, "forward_speed_mps")
        self.yaw_rate = _f(data, "yaw_rate_rad_s")
        self.max_climb = _f(data, "max_climb_rate_mps")
        self.max_vert_accel = _f(data, "max_vertical_accel_mps2")
        self.sim_plant_source = data.get("sim_plant_source")
        self.hardware = data.get("hardware") or {}
        motor_units = str(data.get("motor_command_units", "rad/s"))
        if motor_units != "rad/s":
            raise ValueError(f"motor_command_units 只支持 rad/s，得到 {motor_units!r}")

        fw = data.get("firmware_default") or {}
        self.firmware_source = str(fw.get("source", ""))
        self.imu_hz = _f(fw, "imu_hz", self.control_hz)
        thr = fw.get("throttle") or {}
        self.throttle_min = _f(thr, "mot_thr_min", 0.10)
        self.throttle_max = _f(thr, "mot_thr_max", 0.90)
        self.arm_throttle_limit = _f(thr, "arm_throttle_limit", 0.05)
        # control.ino 的 D 低通 alpha 是在 IMU 1 kHz 下调的；换控制频率要保时间常数。
        alpha = _f(fw, "rates_d_lpf_alpha", 1.0)
        if 0.0 < alpha < 1.0 and self.control_hz > 0:
            ratio = max(1.0, self.imu_hz / self.control_hz)
            self.rate_d_alpha = 1.0 - (1.0 - alpha) ** ratio
        else:
            self.rate_d_alpha = 1.0

        for name, v in (("mass", self.mass), ("arm_length", self.arm_length),
                        ("kf", self.kf), ("km", self.km), ("max_rpm", self.max_rpm),
                        ("control_hz", self.control_hz), ("altitude_target_m", self.altitude_target)):
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"cf_sil 参数 {name} 必须 > 0，得到 {v!r}")
        if len(self.inertia) != 3 or not all(math.isfinite(v) and v > 0 for v in self.inertia):
            raise ValueError(f"inertia 必须是 3 个正数，得到 {self.inertia!r}")
        if not (0.0 < self.max_tilt < math.pi / 2):
            raise ValueError(f"max_tilt_deg 必须在 (0, 90)，得到 {data.get('max_tilt_deg')!r}")
        if not (math.isfinite(self.latency_s) and self.latency_s >= 0.0):
            raise ValueError(f"latency_s 必须 >= 0，得到 {self.latency_s!r}")
        if not (0.0 <= self.throttle_min < self.throttle_max <= 1.0):
            raise ValueError(
                f"firmware_default.throttle 必须满足 0<=min<max<=1，得到 "
                f"{self.throttle_min}/{self.throttle_max}")
        if not all(math.isfinite(v) and 0.0 < v <= math.pi * 2 for v in self.max_rate):
            raise ValueError(f"max_rate_rad_s 必须在 (0, 2π]，得到 {self.max_rate!r}")

        self.max_omega = self.max_rpm * 2.0 * math.pi / 60.0
        self.max_thrust_per_motor = self.kf * self.max_omega * self.max_omega
        self.max_total_thrust = 4.0 * self.max_thrust_per_motor
        # 仿真悬停油门（线性 ESC->转速假设）。真机悬停油门在 hardware 里是 null。
        self.hover_throttle_norm = math.sqrt(
            self.mass * G / self.max_total_thrust)

        rotors = data.get("rotors")
        if not isinstance(rotors, list) or len(rotors) != 4:
            raise ValueError("rotors 必须是长度 4 的列表")
        seen = set()
        self.rotors = []
        for r in rotors:
            try:
                pos = tuple(float(v) for v in r["position"])
                actuator = int(r.get("actuator", len(self.rotors)))
                turning = str(r["turning"]).lower()
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"rotor 配置无效: {r!r}") from e
            if len(pos) != 3 or turning not in ("cw", "ccw"):
                raise ValueError(f"rotor 配置无效: {r!r}")
            if abs(pos[0]) < 1e-9 or abs(pos[1]) < 1e-9:
                raise ValueError(f"rotor x/y 不能为 0（混控要用前后左右符号）: {r!r}")
            if actuator in seen or not 0 <= actuator < 4:
                raise ValueError(f"rotor actuator 必须是唯一的 0..3，得到 {actuator}")
            seen.add(actuator)
            self.rotors.append({
                "position": pos, "turning": turning, "actuator": actuator,
                "sign": 1.0 if turning == "ccw" else -1.0,
                "sx": 1.0 if pos[0] > 0 else -1.0,
                "sy": 1.0 if pos[1] > 0 else -1.0,
            })
        self.rotors.sort(key=lambda r: r["actuator"])

        pid = data.get("pid") or {}
        self.pid = {
            "pos": _gains(pid.get("pos"), "pos"),
            "vel": _gains(pid.get("vel"), "vel"),
            "att": _gains3(pid.get("att"), "att"),
            "rate": _gains3(pid.get("rate"), "rate"),
        }
        self.pid_source = data.get("pid_source") or {}

        if sha256 is None:
            raw = json.dumps(data, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()
            sha256 = hashlib.sha256(raw).hexdigest()
        self.sha256 = sha256
        self.sha8 = sha256[:8]

    @classmethod
    def from_json(cls, path):
        with open(path, "rb") as f:
            raw = f.read()
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            raise ValueError(f"{path}: 不是合法 JSON: {e}") from e
        return cls(data, hashlib.sha256(raw).hexdigest(), path=os.path.abspath(path))

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def profile(self):
        return f"{self.profile_name}:{self.sha8}"

    def manifest(self):
        """写进 episode_manifest.json 的 dynamics_params。"""
        return {
            "profile": self.profile(),
            "profile_name": self.profile_name,
            "sha256": self.sha256,
            "params_file": self.path,
            "calibrated": self.calibrated,
            "altitude_assumption": self.altitude_assumption,
            "motor_command_units": "rad/s",
            "gazebo_max_rot_velocity_rad_s": self.max_omega,
            "sim_plant_source": self.sim_plant_source,
            "firmware_source": self.firmware_source,
            "hardware": self.hardware,
            "hover_throttle_norm_sim": self.hover_throttle_norm,
            "mass_kg": self.mass,
            "inertia_kg_m2": list(self.inertia),
            "arm_length_m": self.arm_length,
            "rotors": [{"position_m": list(r["position"]),
                        "turning": r["turning"], "actuator": r["actuator"]}
                       for r in self.rotors],
            "kf_n_per_rad_s2": self.kf,
            "km_m": self.km,
            "motor_tau_up_s": self.motor_tau_up,
            "motor_tau_down_s": self.motor_tau_down,
            "max_rpm": self.max_rpm,
            "drag": self.drag,
            "pid": self.pid,
            "pid_source": self.pid_source,
            "max_tilt_deg": math.degrees(self.max_tilt),
            "max_rate_rad_s": list(self.max_rate),
            "control_hz": self.control_hz,
            "latency_s": self.latency_s,
            "imu_hz": self.imu_hz,
            "rate_d_lpf_alpha_effective": self.rate_d_alpha,
            "throttle_min": self.throttle_min,
            "throttle_max": self.throttle_max,
            "arm_throttle_limit": self.arm_throttle_limit,
        }


class CfSilController:
    """四动作 -> 电机转速（rad/s）。

    归一化油门/力矩沿用 CF-Drone control.ino：T 是每电机基础油门，tx/ty/tz 是
    归一化力矩指令，混控直接相加；最后按 omega = u * max_omega 发给 Gazebo。
    """

    MAX_DT = 0.1   # odom 间隔异常时不积分过头

    def __init__(self, params, altitude_target=None):
        self.p = params
        self.alt_target = float(
            params.altitude_target if altitude_target is None else altitude_target)
        self.pending = []       # [(生效 ns, action)]
        self.active = SAFE_NEUTRAL
        self.last_t_ns = None
        self.last_motors = [0.0, 0.0, 0.0, 0.0]
        self.last_telemetry = {}
        self._yaw_ref = None

        pos = params.pid["pos"]
        vel = params.pid["vel"]
        att = params.pid["att"]
        rate = params.pid["rate"]
        self._pos_z = _PID(*pos, d_on_measurement=True)
        self._vel_x = _PID(*vel)
        self._vel_y = _PID(*vel)
        self._vel_z = _PID(*vel, d_on_measurement=True)
        self._att_roll = _PID(*att["roll"])
        self._att_pitch = _PID(*att["pitch"])
        self._att_yaw = _PID(*att["yaw"])
        self._rate_roll = _PID(*rate["roll"], d_on_measurement=True,
                               d_alpha=params.rate_d_alpha)
        self._rate_pitch = _PID(*rate["pitch"], d_on_measurement=True,
                                d_alpha=params.rate_d_alpha)
        # control.ino 的 YAWRATE PID 没有 D 低通。
        self._rate_yaw = _PID(*rate["yaw"], d_on_measurement=True, d_alpha=1.0)

    def dispatch(self, action, t_ns):
        """动作派发；延迟 latency_s 后生效。越界动作直接拒绝。"""
        if isinstance(action, bool) or not isinstance(action, int) \
                or action not in VALID_ACTIONS:
            raise ValueError(
                f"cf_sil 动作必须是 {VALID_ACTIONS} 之一，得到 {action!r}")
        try:
            t = int(t_ns)
        except (TypeError, ValueError) as e:
            raise ValueError(f"cf_sil dispatch 时间戳无效: {t_ns!r}") from e
        if t < 0:
            raise ValueError(f"cf_sil dispatch 时间戳必须 >= 0，得到 {t}")
        t_eff = t + int(round(self.p.latency_s * 1e9))
        self.pending.append((t_eff, action))
        self.pending.sort(key=lambda item: item[0])

    def set_altitude_target(self, z):
        self.alt_target = float(z)

    def _on_action_change(self, old, new, yaw):
        if old in (YAW_LEFT, YAW_RIGHT) and new not in (YAW_LEFT, YAW_RIGHT):
            self._yaw_ref = yaw          # 退出偏航动作时锁住当前航向
        if old != new:
            # 设定点切换清水平积分，避免旧动作的积分windup带进新动作。
            self._vel_x.reset()
            self._vel_y.reset()

    def _mix(self, base, tx, ty, tz):
        """CF-Drone 混控：T ± roll ± pitch ± yaw；前后/左右/转向取自旋翼配置。"""
        u = []
        for r in self.p.rotors:
            u.append(base + r["sy"] * tx - r["sx"] * ty - r["sign"] * tz)
        return u

    def _desaturate(self, u):
        """CF-Drone control.ino 的 desaturate：保持平均油门，等比缩力矩分量。"""
        avg = sum(u) / 4.0
        if avg >= 1.0:
            return [1.0] * 4, True
        if avg <= 0.0:
            return [0.0] * 4, True
        mx, mn = max(u), min(u)
        sat = mx >= 1.0 or mn <= 0.0
        scale = 1.0
        if mx > 1.0:
            scale = min(scale, (1.0 - avg) / (mx - avg))
        if mn < 0.0:
            scale = min(scale, avg / (avg - mn))
        if scale < 1.0:
            u = [avg + (v - avg) * scale for v in u]
        return [_clamp(v, 0.0, 1.0) for v in u], sat

    def _motors(self, u):
        # 线性 ESC->转速映射；真机推力曲线未实测，见 JSON hardware.thrust_curve。
        return [min(self.p.max_omega,
                    max(0.0, v) * self.p.max_omega) for v in u]

    def step(self, odom, t_ns=None):
        """推进一帧。odom 可为 parse_odometry 结果或原始 gz JSON。返回 (motors, telem)。"""
        if "position" not in odom:
            odom = parse_odometry(odom)
        t_ns = odom["stamp_ns"] if t_ns is None else int(t_ns)
        if t_ns <= 0:
            t_ns = self.last_t_ns or 0

        dt = 0.0
        if self.last_t_ns is None:
            self.last_t_ns = t_ns
        elif t_ns > self.last_t_ns:
            dt = min((t_ns - self.last_t_ns) / 1e9, self.MAX_DT)
            self.last_t_ns = t_ns

        pos = odom["position"]
        vx, vy, vz = odom["linear"]
        wx, wy, wz = odom["angular"]
        roll, pitch, yaw = quat_to_euler(odom["orientation"])
        if self._yaw_ref is None:
            self._yaw_ref = yaw

        # 延迟队列：到点的动作按派发顺序生效（同一帧内多个到点取最后一个）。
        while self.pending and self.pending[0][0] <= t_ns:
            _, action = self.pending.pop(0)
            if action != self.active:
                self._on_action_change(self.active, action, yaw)
            self.active = action

        # 四动作 -> 机体系水平速度/偏航角速率设定点。
        if self.active == FORWARD_SLOW:
            vx_cmd, vy_cmd, yaw_cmd = self.p.forward_speed, 0.0, 0.0
        elif self.active == YAW_LEFT:
            vx_cmd, vy_cmd, yaw_cmd = 0.0, 0.0, self.p.yaw_rate
        elif self.active == YAW_RIGHT:
            vx_cmd, vy_cmd, yaw_cmd = 0.0, 0.0, -self.p.yaw_rate
        else:
            vx_cmd, vy_cmd, yaw_cmd = 0.0, 0.0, 0.0

        # 高度：sim-only 伴飞电脑定高（真值 z 位置 PID）。不是 CF 固件能力。
        vz_cmd = _clamp(self._pos_z.update(self.alt_target - pos[2], dt,
                                           measurement=vz),
                        -self.p.max_climb, self.p.max_climb)
        az_cmd = _clamp(self._vel_z.update(vz_cmd - vz, dt, measurement=vz),
                        -self.p.max_vert_accel, self.p.max_vert_accel)
        # 期望总推力 -> 归一化油门；按当前姿态补偿倾斜后的升力竖直分量。
        total_thrust = self.p.mass * (G + az_cmd) / max(
            0.2, math.cos(roll) * math.cos(pitch))
        base_u = math.sqrt(max(total_thrust, 0.0) / self.p.max_total_thrust)
        throttle_clamped = base_u < self.p.throttle_min or base_u > self.p.throttle_max
        base_u = _clamp(base_u, self.p.throttle_min, self.p.throttle_max)

        # 水平：机体系速度 PID -> 期望倾角（小角度近似）。
        ax = self._vel_x.update(vx_cmd - vx, dt)
        ay = self._vel_y.update(vy_cmd - vy, dt)
        max_h_acc = G * math.tan(self.p.max_tilt)
        ax = _clamp(ax, -max_h_acc, max_h_acc)
        ay = _clamp(ay, -max_h_acc, max_h_acc)
        pitch_target = _clamp(math.atan2(ax, G), -self.p.max_tilt, self.p.max_tilt)
        roll_target = _clamp(-math.atan2(ay, G), -self.p.max_tilt, self.p.max_tilt)

        # 姿态 -> 角速率设定点；偏航动作走角速率，其余动作锁航向。
        rate_x_t = self._att_roll.update(roll_target - roll, dt)
        rate_y_t = self._att_pitch.update(pitch_target - pitch, dt)
        if self.active in (YAW_LEFT, YAW_RIGHT):
            rate_z_t = yaw_cmd
        else:
            rate_z_t = self._att_yaw.update(_wrap_pi(self._yaw_ref - yaw), dt)
        rate_x_t = _clamp(rate_x_t, -self.p.max_rate[0], self.p.max_rate[0])
        rate_y_t = _clamp(rate_y_t, -self.p.max_rate[1], self.p.max_rate[1])
        rate_z_t = _clamp(rate_z_t, -self.p.max_rate[2], self.p.max_rate[2])

        # 角速率 PID -> 归一化力矩（与 control.ino 的 torqueTarget 同量纲）。
        tx = self._rate_roll.update(rate_x_t - wx, dt, measurement=wx)
        ty = self._rate_pitch.update(rate_y_t - wy, dt, measurement=wy)
        tz = self._rate_yaw.update(rate_z_t - wz, dt, measurement=wz)

        u_raw = self._mix(base_u, tx, ty, tz)
        u, mix_sat = self._desaturate(u_raw)
        motors = self._motors(u)
        self.last_motors = motors
        thrusts = [self.p.kf * w * w for w in motors]

        telem = {
            "t_ns": t_ns,
            "dt": dt,
            "active_action": self.active,
            "active_action_name": ACTIONS[self.active],
            "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
            "velocity_cmd_mps": (vx_cmd, vy_cmd, vz_cmd),
            "accel_cmd_mps2": (ax, ay, az_cmd),
            "roll_target_rad": roll_target,
            "pitch_target_rad": pitch_target,
            "rate_target_rad_s": (rate_x_t, rate_y_t, rate_z_t),
            "torque_norm": (tx, ty, tz),
            "throttle_target": base_u,
            "throttle_clamped": throttle_clamped,
            "motor_norm": tuple(u),
            "motor_rad_s": tuple(motors),
            "thrusts_n": tuple(thrusts),
            "thrust_total_n": sum(thrusts),
            "saturated": bool(mix_sat or throttle_clamped),
        }
        self.last_telemetry = telem
        return motors, telem


# ---------------------------------------------------------------- 飞行日志占位
# 只做接口校验/摘要，不做系统辨识。真机参数到位前不要用日志自动改控制参数。

def load_flight_log(path):
    """按扩展名读 .ulg（pyulog）或 .csv；返回摘要 dict。

    .ulg 需要 pyulog；未安装时明确报错，不静默降级。
    """
    if not isinstance(path, str) or not path:
        raise ValueError("飞行日志路径为空")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ulg":
        return load_ulog(path)
    if ext in (".csv", ".txt"):
        return load_flight_log_csv(path)
    raise ValueError(f"不支持的飞行日志格式 {ext!r}（支持 .ulg / .csv）")


def load_ulog(path):
    """ULog 摘要占位。pyulog 未安装时要求先导出 CSV，不自动装依赖。"""
    try:
        import pyulog
    except ImportError as e:
        raise RuntimeError(
            "读取 .ulg 需要 pyulog，或先用 --flight-log-csv 导出标准 CSV；"
            "本仓库不自动安装该依赖") from e
    if not os.path.isfile(path):
        raise ValueError(f"飞行日志不存在: {path}")
    ulog = pyulog.ULog(path)
    names = sorted(d.name for d in ulog.data_list)
    start = getattr(ulog, "start_timestamp", None)
    last = getattr(ulog, "last_timestamp", None)
    duration = ((last - start) / 1e6 if start is not None and last is not None
                and last >= start else None)
    return {
        "format": "ulog",
        "path": os.path.abspath(path),
        "message_count": len(names),
        "messages": names,
        "duration_s": duration,
        "note": "仅接口校验/摘要，尚未做系统辨识；控制参数仍来自 JSON",
    }


def load_flight_log_csv(path):
    """标准列 CSV 摘要：t, qw/qx/qy/qz, wx/wy/wz, thrust|motor, x/y/z。

    只校验表头和 t 列，不解释姿态/推力语义；留给后续参数辨识。
    """
    if not os.path.isfile(path):
        raise ValueError(f"飞行日志不存在: {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV 缺表头")
        cols = {str(name).strip().lower(): name for name in reader.fieldnames}

        def col(*names):
            for name in names:
                if name in cols:
                    return cols[name]
            return None

        tcol = col("t", "time", "timestamp")
        if tcol is None:
            raise ValueError("CSV 缺少 t/time 列")
        groups = {
            "attitude": all(col(k) for k in ("qw", "qx", "qy", "qz")),
            "rates": all(col(k) for k in ("wx", "wy", "wz")),
            "thrust": col("thrust", "motor", "throttle") is not None,
            "pos": (col("pos") is not None
                    or all(col(k) for k in ("x", "y", "z"))
                    or all(col(k) for k in ("pos_x", "pos_y", "pos_z"))
                    or all(col(k) for k in ("posx", "posy", "posz"))),
        }
        rows, bad = 0, 0
        t0 = t1 = None
        for row in reader:
            try:
                t = float(row[tcol])
                if not math.isfinite(t):
                    raise ValueError
            except (TypeError, ValueError):
                bad += 1
                continue
            rows += 1
            t0 = t if t0 is None else min(t0, t)
            t1 = t if t1 is None else max(t1, t)
    if rows == 0:
        raise ValueError("CSV 没有有效数据行")
    missing = [name for name, ok in groups.items() if not ok]
    return {
        "format": "csv",
        "path": os.path.abspath(path),
        "rows": rows,
        "invalid_rows": bad,
        "duration_s": (t1 - t0) if t0 is not None else None,
        "has_attitude": groups["attitude"],
        "has_rates": groups["rates"],
        "has_thrust": groups["thrust"],
        "has_pos": groups["pos"],
        "missing_groups": missing,
        "note": "仅接口校验/摘要，尚未做系统辨识；控制参数仍来自 JSON",
    }


class _HelperActuatorsPublisher:
    """venv 无 gz.transport13 时，用系统 Python 起 helper 发布电机/使能。"""

    def __init__(self, actuator_topic, enable_topic, connect_timeout=5.0):
        gz_python = os.environ.get("UAVRGB_GZ_PYTHON", "/usr/bin/python3")
        self._proc = subprocess.Popen(
            [gz_python, os.path.abspath(__file__), "--pub-server",
             actuator_topic, enable_topic],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        line = self._proc.stdout.readline().strip()
        if line != "ready":
            err = self._proc.stderr.read().strip()
            self._proc.kill()
            raise RuntimeError(f"cf_sil helper 启动失败: {line or err or '无输出'}")
        self._lock = threading.Lock()

    def _cmd(self, line):
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("cf_sil helper 已退出")
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
            reply = self._proc.stdout.readline().strip()
        if reply != "ok":
            raise RuntimeError(f"cf_sil helper 拒绝 {line!r}: {reply or 'EOF'}")

    def disable_velocity_control(self, timeout=5.0, enable_topic=None):
        self._cmd("enable 0")

    def publish(self, velocities):
        vals = [float(v) for v in velocities]
        if len(vals) != 4:
            raise ValueError(f"cf_sil 需要 4 个电机转速，实际 {len(vals)}")
        self._cmd("v " + " ".join(f"{v:.6f}" for v in vals))

    def close(self):
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write("q\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=1.0)
        except Exception:  # noqa: BLE001 - 收尾失败不覆盖主错误
            self._proc.kill()


class GzActuatorsPublisher:
    """发布 Actuators.velocity(rad/s) 和 /X3/enable；无 gz 时自动用 helper。"""

    def __init__(self, actuator_topic, enable_topic):
        self._helper = None
        try:
            from gz.transport13 import Node
            from gz.msgs10.actuators_pb2 import Actuators
            from gz.msgs10.boolean_pb2 import Boolean
        except Exception:  # noqa: BLE001 - venv 常见；回退系统 Python helper
            self._helper = _HelperActuatorsPublisher(actuator_topic, enable_topic)
            return
        self._Actuators = Actuators
        self._Boolean = Boolean
        self._node = Node()
        self._pub = self._node.advertise(actuator_topic, Actuators)
        self._enable_pub = self._node.advertise(enable_topic, Boolean)

    @staticmethod
    def _wait(pub, timeout, label):
        deadline = time.monotonic() + timeout
        while not pub.has_connections():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{label}: {timeout:.1f}s 内没有订阅者连接")
            time.sleep(0.02)

    def disable_velocity_control(self, timeout=5.0, enable_topic=None):
        """在 /X3/enable 发 false。cf_sil world 已删 plugin 时没有订阅者，跳过。"""
        if self._helper is not None:
            self._helper.disable_velocity_control(timeout, enable_topic)
            return
        try:
            self._wait(self._enable_pub, timeout, f"enable {enable_topic or ''}".strip())
        except RuntimeError:
            self._wait(self._pub, timeout, "motor_speed")
            return
        self._wait(self._pub, timeout, "motor_speed")
        msg = self._Boolean()
        msg.data = False
        self._enable_pub.publish(msg)

    def publish(self, velocities):
        if self._helper is not None:
            self._helper.publish(velocities)
            return
        msg = self._Actuators()
        msg.velocity.extend(float(v) for v in velocities)
        self._pub.publish(msg)

    def close(self):
        if self._helper is not None:
            self._helper.close()
            self._helper = None
            return
        self._pub = None
        self._enable_pub = None
        self._node = None


def _pub_server(actuator_topic, enable_topic):
    """helper 子进程：从 stdin 收 `enable 0/1` 和 `v w0 w1 w2 w3`。"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from gz.transport13 import Node
    from gz.msgs10.actuators_pb2 import Actuators
    from gz.msgs10.boolean_pb2 import Boolean

    node = Node()
    pub = node.advertise(actuator_topic, Actuators)
    enable_pub = node.advertise(enable_topic, Boolean)
    deadline = time.monotonic() + 5.0
    while not pub.has_connections():
        if time.monotonic() >= deadline:
            print("error: 5s 内没有电机订阅者连接", flush=True)
            return 1
        time.sleep(0.02)
    # cf_sil world 没有 VelocityControl 时 /X3/enable 无订阅者；不要求。
    enable_deadline = time.monotonic() + 1.0
    while not enable_pub.has_connections() and time.monotonic() < enable_deadline:
        time.sleep(0.02)
    print("ready", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line == "q":
            break
        try:
            if line.startswith("enable "):
                msg = Boolean()
                msg.data = bool(int(line.split()[1]))
                enable_pub.publish(msg)
            elif line.startswith("v "):
                vals = [float(x) for x in line.split()[1:]]
                if len(vals) != 4:
                    raise ValueError("需要 4 个电机转速")
                msg = Actuators()
                msg.velocity.extend(vals)
                pub.publish(msg)
            else:
                raise ValueError(f"未知命令 {line!r}")
            print("ok", flush=True)
        except Exception as e:  # noqa: BLE001 - 错误回给父进程
            print(f"error {type(e).__name__}: {e}", flush=True)
            return 1
    return 0


class CfSilMotorThread:
    """250 Hz 后台线程：读最新 odometry -> 控制器 -> 发布 4 个电机转速。

    动作由主循环 dispatch；退出/异常时 zero_now/stop 发零速停桨（best-effort）。
    """

    def __init__(self, params, odom_topic="/model/x3/odometry",
                 actuator_topic="/X3/gazebo/command/motor_speed",
                 enable_topic="/X3/enable", altitude_target=None,
                 connect_timeout=5.0, odom_timeout=10.0):
        self.params = params
        self.odom_topic = odom_topic
        self.actuator_topic = actuator_topic
        self.enable_topic = enable_topic
        self.connect_timeout = connect_timeout
        self.odom_timeout = odom_timeout
        self.controller = CfSilController(params, altitude_target)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # 启动后先停桨，等主循环显式 dispatch（起飞/--no-takeoff）再上电。
        self._zero = True
        self._error = None
        self._sub = None
        self._pub = None
        self._thread = None

    def start(self):
        import gz_io
        try:
            self._sub = gz_io.TopicSub(self.odom_topic, buf=4)
            self._pub = GzActuatorsPublisher(self.actuator_topic, self.enable_topic)
            # 先停用 MulticopterVelocityControl，再发我们自己的电机指令。
            self._pub.disable_velocity_control(self.connect_timeout, self.enable_topic)
        except Exception:
            self._close_io()
            raise
        if self._sub.latest_after(0, timeout=self.odom_timeout) is None:
            self._close_io()
            raise RuntimeError(f"{self.odom_topic}: {self.odom_timeout:.0f}s 内没有 odometry")
        self._thread = threading.Thread(target=self._loop, name="cf-sil",
                                        daemon=True)
        self._thread.start()

    def _close_io(self):
        if self._pub is not None:
            try:
                self._pub.close()
            except Exception:  # noqa: BLE001 - 收尾失败不覆盖启动错误
                pass
            self._pub = None
        if self._sub is not None:
            try:
                self._sub.close()
            except Exception:  # noqa: BLE001
                pass
            self._sub = None

    def _set_error(self, msg):
        with self._lock:
            if self._error is None:
                self._error = msg

    def error(self):
        with self._lock:
            return self._error

    def dispatch(self, action, t_ns):
        with self._lock:
            self.controller.dispatch(action, t_ns)
            self._zero = False

    def set_action(self, action):
        msg = self._sub.latest() if self._sub is not None else None
        t_ns = _stamp_ns(msg) if msg is not None else (self.controller.last_t_ns or 0)
        self.dispatch(action, t_ns)

    def set_altitude_target(self, z):
        with self._lock:
            self.controller.set_altitude_target(z)

    def zero_now(self):
        """立即发一次零速，并让线程保持零输出；从不抛异常。"""
        with self._lock:
            self._zero = True
        if self._pub is not None:
            try:
                self._pub.publish([0.0, 0.0, 0.0, 0.0])
            except Exception as e:  # noqa: BLE001 - 安全网自身失败只能告警
                print(f"警告: cf_sil 零速发布失败: {e}", flush=True)
                return False
        return True

    def _loop(self):
        period = 1.0 / self.params.control_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                zero = self._zero
            if zero:
                motors = [0.0, 0.0, 0.0, 0.0]
            else:
                msg = self._sub.latest()
                motors = [0.0, 0.0, 0.0, 0.0]
                if msg is not None:
                    try:
                        odom = parse_odometry(msg)
                        if odom["stamp_ns"] > 0:
                            with self._lock:
                                motors, _ = self.controller.step(odom)
                    except Exception as e:  # noqa: BLE001 - 线程不能静默死
                        self._set_error(f"cf_sil 控制异常: {type(e).__name__}: {e}")
                        with self._lock:
                            self._zero = True
                        motors = [0.0, 0.0, 0.0, 0.0]
            try:
                self._pub.publish(motors)
            except Exception as e:  # noqa: BLE001
                self._set_error(f"cf_sil 电机发布失败: {type(e).__name__}: {e}")
                break
            if not self._sub.healthy():
                self._set_error("cf_sil odometry 订阅进程已退出")
                break
            self._stop.wait(max(0.0, period - (time.perf_counter() - t0)))

        try:
            self._pub.publish([0.0, 0.0, 0.0, 0.0])
        except Exception:  # noqa: BLE001 - 退出零速 best-effort
            pass

    def stop(self):
        """停线程并补零速、收 IO；可重复调用，从不抛异常。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pub is not None:
            for _ in range(3):
                try:
                    self._pub.publish([0.0, 0.0, 0.0, 0.0])
                except Exception:  # noqa: BLE001
                    break
                time.sleep(0.02)
        self._close_io()


if __name__ == "__main__":
    # 内部 helper：collect.py 在 venv 里没有 gz 时，用系统 Python 起这个分支。
    if len(sys.argv) == 4 and sys.argv[1] == "--pub-server":
        raise SystemExit(_pub_server(sys.argv[2], sys.argv[3]))
    raise SystemExit("cf_sil.py 是库；--pub-server 仅供内部 helper 使用")
