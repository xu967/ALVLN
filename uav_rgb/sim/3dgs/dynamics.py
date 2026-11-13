#!/usr/bin/env python3
"""3DGS pilot 动力学入口：两个后端，同一 step(action, dt) -> dict 接口。

cf_blackbox（默认）：直接复用 sim/tools/cf_blackbox.py（参数单源
sim/dynamics/cf_blackbox_v1.json），固定 z=2 m、四动作、延迟/阻力/偏航一阶。
cf_sitl：E1 MINI 固件在环，经独立进程 cf_sitl_proc（stdin/stdout JSON 行，协议见
sim/cf_sitl/proc/README.md）调用，Python 不链接 GPLv3 库。参数单源
sim/dynamics/e1_mini.json；高度由固件定高，不再固定 2.0；位姿带完整四元数。
"""
import collections
import hashlib
import json
import math
import os
import select
import subprocess
import sys
import threading

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from cf_blackbox import (  # noqa: E402,F401  统一从参数单源导入
    CfBlackbox, CfBlackboxParams, DEFAULT_PARAMS_PATH,
    SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT, VALID_ACTIONS,
)
# profile 算法与 Gazebo 链路（sim/tools/cf_sitl_exec.py）同一份：cf_sitl_v1:<参数 value sha8>:lib<N>
from cf_sitl_exec import PROFILE_PREFIX as SITL_PROFILE_PREFIX, format_profile, params_values_sha256  # noqa: E402
from . import scene as scene_mod  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E1_MINI_PARAMS_PATH = os.path.join(_REPO, "sim", "dynamics", "e1_mini.json")
# cf_sitl_proc 可执行文件：SITL lib agent 的构建目录在数据盘（根分区装不下 build）。
CF_SITL_PROC = os.environ.get("CF_SITL_PROC", os.path.join(
    os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp",
    "build-cf-sitl", "cf_sitl_proc"))
SITL_DT_S = 0.001            # 固件 1 kHz 锁步，cf_sitl 只接受 1 ms 步
EXEC_STATES = {0: "DISARMED", 1: "TAKEOFF", 2: "READY", 3: "ACTING", 4: "FAILSAFE"}
EXEC_READY, EXEC_FAILSAFE = 2, 4


def make_blackbox(scene, params_path=None, params=None):
    """按场景 spawn 建飞控状态；params 传入时复用同一参数对象（manifest 一致）。"""
    params = params or CfBlackboxParams.from_json(params_path)
    spawn = scene["spawn"]
    return CfBlackbox(params, x=spawn["x"], y=spawn["y"],
                      yaw_deg=spawn.get("yaw_deg", 0.0))


def _quat_to_euler_deg(q):
    """(w,x,y,z) 机体 FLU->世界 ENU -> (roll, pitch, yaw) 度，ZYX 顺序。"""
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def probe_lib_version(proc_path, params_path):
    """起一次 cf_sitl_proc 读 state.lib_version（CF_SITL_LIB_VERSION）后 quit。没有该键 = 旧库，直接拒绝。"""
    if not os.path.isfile(proc_path):
        raise FileNotFoundError(f"cf_sitl_proc 不存在: {proc_path}（可用环境变量 CF_SITL_PROC 指定）")
    try:
        out = subprocess.run([proc_path, "--params", params_path, "--seed", "0"],
                             input='{"cmd":"state"}\n{"cmd":"quit"}\n', capture_output=True, text=True, timeout=30.0)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"cf_sitl_proc 读 state 超时: {proc_path}") from e
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if out.returncode != 0 or not lines:
        raise RuntimeError(f"cf_sitl_proc 起不来 rc={out.returncode}: {out.stderr.strip()[-500:]}")
    resp = json.loads(lines[0])
    if not resp.get("ok"):
        raise RuntimeError(f"cf_sitl_proc state 失败: {resp.get('error')}")
    v = resp["state"].get("lib_version")
    if not isinstance(v, (int, float)) or v != int(v) or int(v) < 1:
        raise RuntimeError(f"{proc_path} 的 state 没有 lib_version（库早于 CF_SITL_LIB_VERSION=31，按当前 sim/cf_sitl 重建）")
    return int(v)


class CfSitlParams:
    """e1_mini.json 的只读视图：sha、标定位、相机安装、执行器速度；manifest 写进 episode。

    profile = cf_sitl_v1:<各项 value 规范化 JSON 的 sha8>:lib<N>（AUDIT_SIM #1，与 cf_sitl_exec 同算法）：
    改注释不换 profile，改 value 或换库才换。lib_version 来自 cf_sitl_proc 的 state（from_json 默认起一次进程探测）。
    """

    def __init__(self, path, raw, obj, lib_version=None):
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: 顶层必须是对象")
        self.path, self.obj = os.path.abspath(path), obj
        self.sha256 = hashlib.sha256(raw).hexdigest()          # 文件字节 sha（追溯用，不进 profile）
        self.values_sha256 = params_values_sha256(obj)
        self.lib_version = None if lib_version is None else int(lib_version)
        self.calibrated = bool(obj.get("calibrated", False))
        self.camera_mount = self.value("camera_mount")
        if not (isinstance(self.camera_mount, dict) and len(self.camera_mount.get("xyz", ())) == 3):
            raise ValueError(f"{path}: camera_mount.value 需要 xyz[3] 与 pitch_deg")

    @classmethod
    def from_json(cls, path=None, lib_version=None, proc=None):
        """lib_version 不给时起一次 cf_sitl_proc（proc 或 $CF_SITL_PROC）读 state.lib_version。"""
        path = E1_MINI_PARAMS_PATH if path is None else path
        with open(path, "rb") as f:
            raw = f.read()
        if lib_version is None:
            lib_version = probe_lib_version(proc or CF_SITL_PROC, path)
        return cls(path, raw, json.loads(raw.decode("utf-8")), lib_version)

    def value(self, key):
        v = self.obj.get(key)
        return v.get("value") if isinstance(v, dict) and "value" in v else v

    def profile(self):
        if self.lib_version is None:
            raise RuntimeError("cf_sitl 库版本未知：CfSitlParams.from_json() 会起 cf_sitl_proc 读 state.lib_version，"
                               "直接构造时请传 lib_version")
        return format_profile(self.values_sha256, self.lib_version)

    def manifest(self):
        ex = self.value("executor") or {}
        return {"dynamics": "cf_sitl", "vehicle": "e1_mini", "profile": self.profile(),
                "params_file": self.path, "params_sha256": self.sha256,
                "params_values_sha256": self.values_sha256, "lib_version": self.lib_version,
                "calibrated": self.calibrated, "firmware": self.obj.get("firmware"),
                "camera_mount": self.camera_mount,
                "collision_envelope": self.value("collision_envelope"),
                "forward_speed_mps": ex.get("forward_speed_mps"),
                "yaw_rate_rps": ex.get("yaw_rate_rps"),
                "altitude_assumption": "firmware_height_hold",
                "sitl_proc": CF_SITL_PROC}


class CfSitlDynamics:
    """cf_sitl_proc 子进程 + 与 CfBlackbox 同形的 step(action, dt) -> dict。

    reset() 在地面 reset 再起飞到场景 spawn z 并等 READY，起飞时间不计入 episode（t 从
    READY 时刻起算）。step(action, dt)：act(until_s = 当前 t + dt，observation 语义；SITL 时钟
    只在 step 里走，渲染与下发同刻，所以也等于 dispatch 语义) + round(dt/1 ms) 个 1 ms 步，
    每 trace_dt 记一次中间状态到 self.trace（视频用，决策仍只看窗口起点那一帧）。
    """

    def __init__(self, scene, params=None, params_path=None, seed=0, proc=None,
                 trace_dt=0.1, timeout_s=60.0, takeoff_max_s=15.0):
        self.params = params or CfSitlParams.from_json(params_path)
        self.proc_path = proc or CF_SITL_PROC
        if not os.path.isfile(self.proc_path):
            raise FileNotFoundError(f"cf_sitl_proc 不存在: {self.proc_path}（可用环境变量 CF_SITL_PROC 指定）")
        self.seed = int(seed)
        self.trace_dt = float(trace_dt)
        self.timeout_s = float(timeout_s)
        self.takeoff_max_s = float(takeoff_max_s)
        self.ground_z = float(scene.get("ground_z", 0.0))
        # SITL 独立积分的地面是 z=0 平面；场景 ground_z 不为 0 时整体平移，nav z = sitl z + ground_z。
        self.takeoff_height = float(scene["spawn"]["z"]) - self.ground_z
        if not (0.0 < self.takeoff_height < 2.5):
            raise ValueError(f"spawn z 距地 {self.takeoff_height} m 超出 ToF/光流量程 (0, 2.5)")
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.yaw = self.yaw_rate = 0.0
        self.roll_deg = self.pitch_deg = 0.0
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.exec_state = 0
        self.active = SAFE_NEUTRAL
        self.t = self.t_sitl = self.t0 = 0.0
        self.takeoff_s = None
        self.path_m = 0.0
        self.step_index = 0
        self.trace = []
        self._stderr = collections.deque(maxlen=20)
        self.proc = subprocess.Popen(
            [self.proc_path, "--params", self.params.path, "--seed", str(self.seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        # stderr 单独线程收尾巴：进程崩溃时把原因带进异常，也避免它写满管道阻塞。
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        # 这个进程的库版本必须就是 params.profile() 声称的那个（params 可能用别的 proc 探测过）
        try:
            lib_version = int(self._call({"cmd": "state"})["state"].get("lib_version") or 0)
            if lib_version < 1:
                raise RuntimeError(f"{self.proc_path} 的 state 没有 lib_version（旧库，按当前 sim/cf_sitl 重建）")
            if self.params.lib_version is None:
                self.params.lib_version = lib_version
            elif self.params.lib_version != lib_version:
                raise RuntimeError(f"cf_sitl_proc lib_version={lib_version} 与 params.lib_version={self.params.lib_version} 不一致")
        except BaseException:
            self.close()
            raise

    def _pump_stderr(self):
        try:
            for line in self.proc.stderr:
                self._stderr.append(line.rstrip("\n"))
        except ValueError:
            pass

    def _call(self, req):
        """一请求一应答；子进程退出/超时/ok=false 都抛 RuntimeError，不挂死。"""
        if self.proc.poll() is not None:
            raise RuntimeError(f"cf_sitl_proc 已退出 rc={self.proc.returncode}: {' | '.join(self._stderr)}")
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise RuntimeError(f"写 cf_sitl_proc 失败（rc={self.proc.poll()}）: {e}; {' | '.join(self._stderr)}") from e
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout_s)
        if not ready:
            self.close()
            raise RuntimeError(f"cf_sitl_proc {req.get('cmd')} 超过 {self.timeout_s:g}s 无应答，已杀掉")
        line = self.proc.stdout.readline()
        if not line:
            rc = self.proc.wait(timeout=2.0)
            raise RuntimeError(f"cf_sitl_proc 已退出 rc={rc}: {' | '.join(self._stderr)}")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"cf_sitl_proc {req.get('cmd')} 失败: {resp.get('error')}")
        return resp

    def _absorb(self, st):
        x, y, z = (float(v) for v in st["pos"])
        z += self.ground_z
        self.path_m += math.hypot(x - self.x, y - self.y)
        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = (float(v) for v in st["vel"])
        self.quat = tuple(float(v) for v in st["quat"])
        self.roll_deg, self.pitch_deg, yaw_deg = _quat_to_euler_deg(self.quat)
        self.yaw = math.radians(yaw_deg)
        self.exec_state = int(st["exec_state"])
        self.t_sitl = float(st["t_s"])
        self.t = self.t_sitl - self.t0

    def reset(self, x, y, yaw_deg, episode=0):
        """地面 reset -> 起飞到 spawn z -> READY；噪声种子 = proc --seed ^ episode。"""
        self._call({"cmd": "reset", "episode": int(episode), "spawn": [float(x), float(y), 0.0],
                    "yaw": math.radians(float(yaw_deg))})
        r = self._call({"cmd": "takeoff", "height": self.takeoff_height, "max_s": self.takeoff_max_s})
        st = r["state"]
        if not r.get("ready") or int(st["exec_state"]) != EXEC_READY:
            raise RuntimeError(f"起飞 {self.takeoff_max_s:g}s 内未 READY: exec_state="
                               f"{EXEC_STATES.get(int(st['exec_state']))} z={st['pos'][2]:.2f}")
        self.t0 = float(st["t_s"])
        self.takeoff_s = self.t0
        self._absorb(st)
        self.step_index, self.path_m, self.trace = 0, 0.0, []   # 起飞段不计入 episode
        return self._state()

    def step(self, action, dt):
        if isinstance(action, bool) or not isinstance(action, int) or action not in VALID_ACTIONS:
            raise ValueError(f"cf_sitl 动作必须是 {VALID_ACTIONS} 之一，得到 {action!r}")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"cf_sitl dt 必须有限且 > 0，得到 {dt!r}")
        n_total = int(round(dt / SITL_DT_S))
        if n_total < 1 or not math.isclose(dt, n_total * SITL_DT_S, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"cf_sitl dt 必须是至少一个 1 ms tick 的整数倍，得到 {dt!r}")
        dt = n_total * SITL_DT_S
        if self.takeoff_s is None:
            raise RuntimeError("先 reset() 起飞再 step()")
        # until_s 多给半个 SITL 步：执行器的判据是 t < until_s，窗口最后一步 t == t_sitl + dt 若不多给，
        # 那一步动作已过期、执行器回 SAFE 一个 tick（速度参考归零、位置保持点重置），每个窗口末刻都来一下，
        # trace 里也会记成 action=-1 / exec_state=READY。下一窗口的 act 在继续步进之前到达，所以不会多执行。
        self._call({"cmd": "act", "action": action, "until_s": self.t_sitl + dt + 0.5 * SITL_DT_S})
        self.active = action          # 窗口内（含末刻）生效的动作
        n_chunk = max(1, int(round(self.trace_dt / SITL_DT_S)))
        yaw_prev, done = self.yaw, 0
        while done < n_total:
            n = min(n_chunk, n_total - done)
            st = self._call({"cmd": "step", "n": n})["state"]
            done += n
            self._absorb(st)
            self.trace.append({"step": self.step_index, "t": self.t, "t_sitl": self.t_sitl,
                               "pos": [self.x, self.y, self.z], "quat": list(self.quat),
                               "vel": [self.vx, self.vy, self.vz], "action": int(st.get("action", -1)),
                               "exec_state": self.exec_state})
            if self.exec_state == EXEC_FAILSAFE:
                raise RuntimeError(f"固件进入 FAILSAFE（t={self.t:.2f}s，step {self.step_index}）")
        self.yaw_rate = ((self.yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi) / dt
        self.step_index += 1
        return self._state()

    def _state(self):
        return {"t": self.t, "x": self.x, "y": self.y, "z": self.z,
                "vx": self.vx, "vy": self.vy, "vz": self.vz,
                "yaw": self.yaw, "yaw_deg": math.degrees(self.yaw), "yaw_rate": self.yaw_rate,
                "active_action": self.active, "path_m": self.path_m,
                "pose": (self.x, self.y, self.z), "velocity": (self.vx, self.vy, self.vz),
                "quat": self.quat, "roll_deg": self.roll_deg, "pitch_deg": self.pitch_deg,
                "exec_state": self.exec_state}

    def camera_viewmat(self, nav_to_world=None):
        """机载相机 world->camera：机体位姿 ⊗ e1_mini.json camera_mount（含 roll/pitch）。"""
        return scene_mod.viewmat_from_quat((self.x, self.y, self.z), self.quat,
                                           self.params.camera_mount, nav_to_world)

    def profile(self):
        return self.params.profile()

    def manifest(self):
        return {**self.params.manifest(), "seed": self.seed,
                "takeoff_height_m": self.takeoff_height, "takeoff_s": self.takeoff_s,
                "trace_dt_s": self.trace_dt,
                "action_window": "until_s = t_obs + hold_s + 0.5 ms（observation 语义；SITL 时钟只在 step 里走，"
                                 "渲染、决策、下发同刻，等于 dispatch 语义；半步余量让窗口末刻仍在动作内）",
                "episode_clock": "t=0 为 READY 时刻，起飞不计入"}

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"cmd":"quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=2.0)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                pipe.close()
            except OSError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - 析构不抛
            pass


def _selftest():
    p = CfBlackboxParams.from_json()
    assert p.calibrated is False and p.z_target_m == 2.0
    d = CfBlackbox(p, 0, 0, 0)
    for _ in range(20):
        s = d.step(FORWARD_SLOW, 0.5)
    assert s["x"] > 0.5 and abs(s["y"]) < 1e-9 and s["z"] == 2.0, s
    d2 = CfBlackbox(p, 0, 0, 0)
    for _ in range(20):
        s = d2.step(YAW_LEFT, 0.5)
    assert s["yaw_deg"] > 5 and s["z"] == 2.0, s
    print("dynamics selftest PASS", d.profile(), s["yaw_deg"])


if __name__ == "__main__":
    _selftest()
