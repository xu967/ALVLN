#!/usr/bin/env python3
"""cf_blackbox：四动作 -> 整机水平/偏航响应的纯 Python 黑盒状态模型。

不依赖 Gazebo / numpy，只做 2D 水平运动 + 固定高度：
  - SAFE_NEUTRAL：不主动刹车，只有线性阻力被动滑行；
  - FORWARD_SLOW：机体系 +X 固定加速度（终端速度 = 加速度 / 阻力系数）；
  - YAW_LEFT/RIGHT：偏航角速率（可带一阶滞后）。
z 固定为 z_target_m（默认 2.0 m，训练高度），vz 恒为 0。

参数唯一来源：sim/dynamics/cf_blackbox_v1.json。本模块不再内置第二份默认值；
`CfBlackbox()` 不传 params 时从该 JSON 读取。当前参数全部是占位值，
calibrated=false，等真机飞行日志（ULog）到位后再替换。

时间单位：秒（与 3DGS collector / 离线仿真一致）。collect.py 的 Gazebo 链路
仍用 cf_stab / x3_velocity，不驱动本模型；3DGS collector 直接 import 本模块。
"""
import hashlib
import json
import math
import os

from teacher import SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT

G = 9.81
PROFILE_VERSION = "cf_blackbox_v1"
VALID_ACTIONS = (SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT)
# 参数唯一来源：sim/dynamics/cf_blackbox_v1.json。本模块不内置第二份默认值。
# 注意：teacher_predictive.py 与 rl_teacher/core.py 仍有重复的占位默认
# （latency=0.15、tilt=2°、drag_k=1、yaw_rate=0.6），不在本次文件所有权内，
# 见 docs/experiments/2026-11-05-cf-blackbox.md 的「参数单源」一节。
DEFAULT_PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dynamics", "cf_blackbox_v1.json")
_EPS = 1e-9


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _num(data, key):
    """取必需数值字段；缺失/非数/非有限直接报错，不静默用默认值。"""
    if key not in data:
        raise ValueError(f"cf_blackbox 参数缺少 {key}")
    try:
        value = float(data[key])
    except (TypeError, ValueError) as e:
        raise ValueError(f"cf_blackbox 参数 {key} 不是数字: {data[key]!r}") from e
    if not math.isfinite(value):
        raise ValueError(f"cf_blackbox 参数 {key} 必须是有限数，得到 {value!r}")
    return value


def _check_action(action):
    if isinstance(action, bool) or not isinstance(action, int) or action not in VALID_ACTIONS:
        raise ValueError(f"cf_blackbox 动作必须是 {VALID_ACTIONS} 之一，得到 {action!r}")


class CfBlackboxParams:
    """cf_blackbox_v1.json -> 只读参数对象；profile 带 sha8，换参数档名必变。"""

    def __init__(self, data, sha256=None, path=None):
        if not isinstance(data, dict):
            raise ValueError("cf_blackbox 参数必须是 JSON object")
        self.data = data
        self.path = path
        self.profile_version = str(data.get("profile_version", PROFILE_VERSION))
        if not self.profile_version.startswith("cf_blackbox"):
            raise ValueError(
                f"profile_version 必须是 cf_blackbox*，得到 {self.profile_version!r}")
        source = data.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("cf_blackbox 参数必须有非空 source（参数来源/占位说明）")
        self.source = source.strip()
        self.calibrated = bool(data.get("calibrated", False))
        self.forward_accel_mps2 = _num(data, "forward_accel_mps2")
        self.drag_k = _num(data, "drag_k")
        self.latency_s = _num(data, "latency_s")
        self.yaw_rate_rad_s = _num(data, "yaw_rate_rad_s")
        self.yaw_tau_s = _num(data, "yaw_tau_s")
        self.z_target_m = _num(data, "z_target_m")
        self.max_tilt_deg = _num(data, "max_tilt_deg")

        if self.forward_accel_mps2 <= 0.0:
            raise ValueError(
                f"forward_accel_mps2 必须 > 0，得到 {self.forward_accel_mps2!r}")
        if self.drag_k <= 0.0:
            raise ValueError(f"drag_k 必须 > 0，得到 {self.drag_k!r}")
        if self.latency_s < 0.0:
            raise ValueError(f"latency_s 必须 >= 0，得到 {self.latency_s!r}")
        if self.yaw_rate_rad_s <= 0.0:
            raise ValueError(f"yaw_rate_rad_s 必须 > 0，得到 {self.yaw_rate_rad_s!r}")
        if self.yaw_tau_s < 0.0:
            raise ValueError(f"yaw_tau_s 必须 >= 0，得到 {self.yaw_tau_s!r}")
        if self.z_target_m <= 0.0:
            raise ValueError(f"z_target_m 必须 > 0，得到 {self.z_target_m!r}")
        if not (0.0 < self.max_tilt_deg < 90.0):
            raise ValueError(
                f"max_tilt_deg 必须在 (0, 90)，得到 {self.max_tilt_deg!r}")
        # 水平加速度不能超过该倾角在悬停推力下能给出的值。这是校准边界，
        # 不是额外动力学：参数写错时直接拒绝，避免训练数据悄悄失真。
        max_accel = G * math.tan(math.radians(self.max_tilt_deg))
        if self.forward_accel_mps2 > max_accel + 1e-9:
            raise ValueError(
                f"forward_accel_mps2={self.forward_accel_mps2} 超过 "
                f"max_tilt_deg={self.max_tilt_deg} 的 {max_accel:.3f} m/s²")

        if sha256 is None:
            raw = json.dumps(data, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()
            sha256 = hashlib.sha256(raw).hexdigest()
        self.sha256 = sha256
        self.sha8 = sha256[:8]

    @classmethod
    def from_json(cls, path=None):
        path = DEFAULT_PARAMS_PATH if path is None else path
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

    @property
    def terminal_speed_mps(self):
        """FORWARD_SLOW 稳态速度（阻力与推力平衡）。"""
        return self.forward_accel_mps2 / self.drag_k

    def profile(self):
        return f"{self.profile_version}:{self.sha8}"

    def manifest(self):
        """写进 episode_manifest.json 的 dynamics_params（3DGS collector 也用这里）。"""
        return {
            "profile": self.profile(),
            "profile_version": self.profile_version,
            "sha256": self.sha256,
            "params_file": self.path,
            "calibrated": self.calibrated,
            "source": self.source,
            "forward_accel_mps2": self.forward_accel_mps2,
            "terminal_speed_mps": self.terminal_speed_mps,
            "drag_k": self.drag_k,
            "latency_s": self.latency_s,
            "yaw_rate_rad_s": self.yaw_rate_rad_s,
            "yaw_tau_s": self.yaw_tau_s,
            "z_target_m": self.z_target_m,
            "max_tilt_deg": self.max_tilt_deg,
            "altitude_assumption": "fixed_z",
        }


class CfBlackbox:
    """状态 x,y,z,vx,vy,vz,yaw；z 固定，水平/偏航按四动作响应。

    `dispatch(action, t=None)` 入延迟队列；`step(action, dt)` 推进 dt 秒。
    step 的 action 只在变化时入队，所以循环里反复传同一个动作不会重复叠加延迟。
    """

    def __init__(self, params=None, x=0.0, y=0.0, yaw_deg=0.0):
        self.params = CfBlackboxParams.from_json() if params is None else params
        for name, value in (("x", x), ("y", y), ("yaw_deg", yaw_deg)):
            if not math.isfinite(float(value)):
                raise ValueError(f"cf_blackbox 初始 {name} 必须是有限数，得到 {value!r}")
        self.x = float(x)
        self.y = float(y)
        self.z = self.params.z_target_m
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = _wrap_pi(math.radians(float(yaw_deg)))
        self.yaw_rate = 0.0
        self.t = 0.0
        self.path_m = 0.0
        self.active = SAFE_NEUTRAL
        self.pending = []          # [(生效时刻 s, action)]，按生效时刻稳定排序
        self._last_command = None  # step(action) 的去重基准

    def profile(self):
        return self.params.profile()

    def manifest(self):
        return self.params.manifest()

    def dispatch(self, action, t=None):
        """把动作排到 t + latency_s 生效；t 缺省为当前模型时间。

        t 早于当前时间时按当前时间处理（调用方时钟抖动不应把动作排到过去）。
        """
        _check_action(action)
        t = self.t if t is None else float(t)
        if not math.isfinite(t):
            raise ValueError(f"cf_blackbox dispatch 时间必须是有限数，得到 {t!r}")
        t = max(t, self.t)
        last = self.pending[-1][1] if self.pending else self.active
        if action == last:
            return  # 同一动作重复派发不堆队列
        self.pending.append((t + self.params.latency_s, action))
        self.pending.sort(key=lambda item: item[0])
        self._last_command = action

    def _target_yaw_rate(self, action):
        if action == YAW_LEFT:
            return self.params.yaw_rate_rad_s
        if action == YAW_RIGHT:
            return -self.params.yaw_rate_rad_s
        return 0.0

    def _apply_due(self):
        while self.pending and self.pending[0][0] <= self.t + _EPS:
            self.active = self.pending.pop(0)[1]

    def _integrate(self, dt):
        """按当前 active 动作积分 dt 秒；yaw 用一阶滞后解析解，水平用阻力解析解。"""
        if dt <= 0.0:
            return
        target_w = self._target_yaw_rate(self.active)
        tau = self.params.yaw_tau_s
        if tau <= 0.0:
            dyaw = target_w * dt
            new_rate = target_w
        else:
            decay = math.exp(-dt / tau)
            dyaw = target_w * dt + (self.yaw_rate - target_w) * tau * (1.0 - decay)
            new_rate = target_w + (self.yaw_rate - target_w) * decay
        # 段内航向取中点，降低 yaw 变化时加速度方向的离散误差。
        yaw_mid = self.yaw + 0.5 * dyaw
        accel = (self.params.forward_accel_mps2
                 if self.active == FORWARD_SLOW else 0.0)
        ax, ay = accel * math.cos(yaw_mid), accel * math.sin(yaw_mid)
        k = self.params.drag_k
        e = math.exp(-k * dt)
        lim_x, lim_y = ax / k, ay / k
        dx = lim_x * dt + (self.vx - lim_x) * (1.0 - e) / k
        dy = lim_y * dt + (self.vy - lim_y) * (1.0 - e) / k
        self.x += dx
        self.y += dy
        self.vx = lim_x + (self.vx - lim_x) * e
        self.vy = lim_y + (self.vy - lim_y) * e
        self.yaw = _wrap_pi(self.yaw + dyaw)
        self.yaw_rate = new_rate
        self.path_m += math.hypot(dx, dy)
        # 高度是训练固定值，模型不积分 z。
        self.z = self.params.z_target_m
        self.vz = 0.0

    def step(self, action=None, dt=0.0):
        """推进 dt 秒并返回状态。

        action 非 None 且与上次不同：先按 latency_s 入队。返回 dict 含
        pose=(x,y,z)、velocity=(vx,vy,vz)、yaw（rad）、yaw_deg、yaw_rate、
        active_action、t、path_m，以及同名扁平字段。
        """
        try:
            dt = float(dt)
        except (TypeError, ValueError) as e:
            raise ValueError(f"cf_blackbox dt 不是数字: {dt!r}") from e
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError(f"cf_blackbox dt 必须有限且 >= 0，得到 {dt!r}")
        if action is not None:
            _check_action(action)
            if action != self._last_command:
                self.dispatch(action)
                self._last_command = action

        t_end = self.t + dt
        self._apply_due()
        while self.t < t_end - _EPS:
            if self.pending and self.pending[0][0] < t_end - _EPS:
                nxt = max(self.pending[0][0], self.t)
                if nxt > self.t + _EPS:
                    self._integrate(nxt - self.t)
                    self.t = nxt
                self._apply_due()
            else:
                self._integrate(t_end - self.t)
                self.t = t_end
        self.t = t_end
        return {
            "t": self.t,
            "x": self.x, "y": self.y, "z": self.z,
            "vx": self.vx, "vy": self.vy, "vz": self.vz,
            "yaw": self.yaw, "yaw_deg": math.degrees(self.yaw),
            "yaw_rate": self.yaw_rate,
            "active_action": self.active,
            "path_m": self.path_m,
            "pose": (self.x, self.y, self.z),
            "velocity": (self.vx, self.vy, self.vz),
        }
