#!/usr/bin/env python3
"""cf_sitl 执行器：collect.py 与 Gazebo 内 CfSitlSystem 插件之间的话题桥（E1 MINI 固件在环）。

与 cf_sil 的 CfSilMotorThread 同一套 duck-type 接口（start/dispatch/set_action/zero_now/
error/stop + params），collect_one 的 sil 分支逻辑不变。区别：
  * 控制闭环在 Gazebo 进程内的插件里跑（每个 1 ms 物理步一次固件循环），这里只发
    「动作 + 截止仿真秒」，插件把 Gazebo 仿真秒换算到 SITL 时钟；
  * 起飞不是 SAFE 定高爬升，而是 cmd 话题的 `reset <episode>` + `arm_takeoff <h>`，
    然后等 state 的 exec_state==READY；
  * 窗口截止 until_s = 动作窗口起点（pose 戳，仿真秒）+ hold；到期固件自动回 SAFE 悬停。

话题（与 sim/worlds/uav_rgb_e1.sdf 的插件参数一致）：
  /e1_mini/action  gz.msgs.Double_V [action, until_s_sim]
  /e1_mini/cmd     gz.msgs.StringMsg "reset <episode>" | "arm_takeoff <height_m>"
  /e1_mini/state   gz.msgs.Double_V 200 Hz；索引表见 sim/cf_sitl/gz/README.md

发布端：当前解释器有 gz.transport13 就进程内发布；没有（.venv-tools）就用
$UAVRGB_GZ_PYTHON（默认 /usr/bin/python3）跑本文件的 --pub-server 子进程，stdin 收行、
stdout 回 ack，与 gz_io 的 twist helper 同一个套路。

自检（不需要 sim）：python3 cf_sitl_exec.py --self-check
"""
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gz_io  # noqa: E402

# ---------------------------------------------------------------- 配置（改这里，别改函数体）
ACTION_TOPIC = "/e1_mini/action"   # 模板 <action_topic>
CMD_TOPIC = "/e1_mini/cmd"         # 模板 <cmd_topic>
STATE_TOPIC = "/e1_mini/state"     # 模板 <state_topic>
SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 参数单源（SITL 库 agent 维护）。与插件同一条解析规则：模板里是仓库相对路径
# sim/dynamics/e1_mini.json，按 $UAVRGB_REPO_ROOT（缺省本仓库）解析，两边永远读同一个文件。
DEFAULT_PARAMS = os.path.join(os.environ.get("UAVRGB_REPO_ROOT") or os.path.dirname(SIM_DIR),
                              "sim", "dynamics", "e1_mini.json")
PROFILE_PREFIX = "cf_sitl_v1"   # manifest.dynamics_profile = cf_sitl_v1:<参数 sha8>，换参数后旧包不混入
TAKEOFF_HEIGHT_M = 2.0   # 任务规定巡航高度；固件定高，不走 collect 的 altitude_vz
STATE_WAIT_S = 20.0      # 等 state 首帧：软件渲染下世界起来 + 插件 Configure 实测 5~10 s，留 2 倍
RESET_ACK_S = 5.0        # reset 后等 exec_state 回 DISARMED 且 SITL 时钟归零；200 Hz 下正常 <50 ms
RESET_RESEND_S = 1.0     # 这么久没确认就重发 reset：正常 <50 ms 到达，1 s 远大于抖动，又给 5 s 内 4 次重发机会
READY_TIMEOUT_S = 90.0   # arm_takeoff→READY：爬升 2 m 约 3~4 s 仿真，RTF 0.05 兜底 ≈ 80 s 墙钟
# 起飞阶段进 FAILSAFE 的兜底重试（reset→落地→再 arm），次数记进 takeoff_failsafes / manifest。
# 历史原因：机体原点离地 0.02 m 时 ToF 读数在量程下限 0.04 m 边缘，噪声偶给一个有效样本后又连续无效
# >0.5 s，执行器误判 FAILSAFE（2026-11-06 真库冒烟 2/6 次复现）；库侧已改为高度估计曾 ≥ tof_airborne_m
# =0.15 m 才算离地（e1_mini.json executor 新键）。重试只防回归，正常应为 0 次。
TAKEOFF_RETRIES = 3      # 上限 3 次：偶发问题 3 次都中的概率可忽略；再失败就是别的问题，交给调用方
SETTLE_S = 3.0           # 重试前等机体落回地面（z<0.1 m 且 |vz|<0.05）：FAILSAFE 最多漂升 0.5 s，落回 <1 s
ZERO_UNTIL_S = 3600.0    # zero_now 的 SAFE 截止相对量：远大于任何步间隔；到期后固件也仍悬停
POLL_S = 0.02            # 等 state 变化的轮询间隔：state 200 Hz，20 ms 内必有新帧
STATE_STALE_S = 3.0      # state 多久没新帧算停更（墙钟）：gz 卡死/插件崩溃时不要死等
VALID_ACTIONS = (0, 1, 2, 3)
# Double_V 索引（与 sim/cf_sitl/gz/CfSitlState.cc 一致）
IDX_T, IDX_EXEC, IDX_FW_MODE, IDX_ARMED, IDX_ACTION, IDX_UNTIL = 0, 1, 2, 3, 4, 5
IDX_HEIGHT, IDX_POS, IDX_QUAT, IDX_VEL = 18, 22, 25, 29
IDX_SIM_T, IDX_LATE, IDX_ERRORS = 32, 33, 34
IDX_STICKS = 35          # [35..38] roll pitch yaw throttle（cf_sitl.h 2026-11-06 追加）
STATE_FIELDS = 39
EXEC_STATES = {0: "DISARMED", 1: "TAKEOFF", 2: "READY", 3: "ACTING", 4: "FAILSAFE"}
EXEC_DISARMED, EXEC_TAKEOFF, EXEC_READY, EXEC_ACTING, EXEC_FAILSAFE = range(5)


class WaitTimeout(RuntimeError):
    """_wait_state 等到超时（有别于 FAILSAFE / 订阅退出 / 停更等硬错误，调用方可据此重发）。"""


def state_stamp_ns(msg):
    """state 帧的 Gazebo 仿真戳（Double_V 没有 header，仿真秒在 data[32]）。"""
    try:
        return int(round(float(msg["data"][IDX_SIM_T]) * 1e9))
    except (KeyError, TypeError, IndexError, ValueError):
        return 0


class CfSitlParams:
    """e1_mini.json 的只读视图：collect 只关心 sha、标定位与少数展示字段。"""

    altitude_assumption = "firmware_height_hold"   # 高度由固件定高，不是真值 P 控制
    altitude_target = TAKEOFF_HEIGHT_M

    def __init__(self, path, raw, obj):
        self.path, self.raw, self.obj = path, raw, obj
        self.sha256 = hashlib.sha256(raw).hexdigest()
        # 单源 JSON 里没有标定位时按未标定处理：数字全是草案/假设值，manifest 必须说清。
        self.calibrated = bool(obj.get("calibrated", False)) if isinstance(obj, dict) else False

    @classmethod
    def from_json(cls, path):
        with open(path, "rb") as f:
            raw = f.read()
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: 顶层必须是对象")
        return cls(path, raw, obj)

    def profile(self):
        return f"{PROFILE_PREFIX}:{self.sha256[:8]}"

    def _value(self, key):
        v = self.obj.get(key)
        return v.get("value") if isinstance(v, dict) and "value" in v else v

    def manifest(self):
        return {"params_path": os.path.abspath(self.path), "params_sha256": self.sha256,
                "calibrated": self.calibrated, "mass_flight": self._value("mass_flight"),
                "inertia_diag": self._value("inertia_diag"),
                "takeoff_height_m": self.altitude_target}


# ---------------------------------------------------------------- 发布端

class _InProcPublisher:
    mode = "gz.transport13（进程内）"

    def __init__(self, connect_timeout=gz_io.PUBLISHER_START_TIMEOUT):
        from gz.transport13 import Node
        from gz.msgs10.double_v_pb2 import Double_V
        from gz.msgs10.stringmsg_pb2 import StringMsg
        self._Double_V, self._StringMsg = Double_V, StringMsg
        self._node = Node()
        self._action = self._node.advertise(ACTION_TOPIC, Double_V)
        self._cmd = self._node.advertise(CMD_TOPIC, StringMsg)
        deadline = time.monotonic() + connect_timeout
        while not (self._action.has_connections() and self._cmd.has_connections()):
            if time.monotonic() >= deadline:
                raise gz_io.PublisherUnavailable(
                    f"{ACTION_TOPIC}/{CMD_TOPIC}: {connect_timeout:.0f}s 内插件没有订阅上")
            time.sleep(0.02)

    def action(self, action, until_s):
        msg = self._Double_V()
        msg.data.extend([float(action), float(until_s)])
        self._action.publish(msg)

    def cmd(self, text):
        msg = self._StringMsg()
        msg.data = text
        self._cmd.publish(msg)

    def close(self):
        self._action = self._cmd = self._node = None


class _HelperPublisher:
    """没有 gz 绑定的解释器：用 $UAVRGB_GZ_PYTHON 跑 --pub-server 子进程。"""

    def __init__(self, connect_timeout=gz_io.PUBLISHER_START_TIMEOUT):
        exe = os.environ.get(gz_io.HELPER_ENV, gz_io.DEFAULT_GZ_PYTHON)
        try:
            self._proc = subprocess.Popen(
                [exe, "-I", os.path.abspath(__file__), "--pub-server"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except OSError as e:
            raise gz_io.PublisherUnavailable(f"启动 {exe} 失败: {e}") from e
        self.mode = f"gz.transport13（helper: {exe}）"
        self._closed = False
        self._acks = queue.Queue()
        threading.Thread(target=self._pump, daemon=True, name="cf-sitl-pub").start()
        ready = self._readline(connect_timeout + 1.0)
        if ready != "ready":
            self.close()
            raise gz_io.PublisherUnavailable(f"{exe} helper 未就绪: {ready or '超时/已退出'}")

    def _pump(self):
        try:
            for line in self._proc.stdout:
                self._acks.put(line.strip())
        finally:
            self._acks.put(None)

    def _readline(self, timeout):
        try:
            return self._acks.get(timeout=timeout)
        except queue.Empty:
            return None

    def _send(self, line):
        if self._closed or self._proc.poll() is not None:
            raise RuntimeError("cf_sitl publisher helper 已退出")
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"写 cf_sitl publisher helper 失败: {e}") from e
        reply = self._readline(gz_io.PUBLISH_TIMEOUT)
        if reply != "ok":
            self.close()
            raise RuntimeError(f"cf_sitl publisher helper 发布失败: {reply or '超时/已退出'}")

    def action(self, action, until_s):
        self._send(f"a {int(action)} {float(until_s):.6f}")

    def cmd(self, text):
        self._send("c " + text)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.stdin.write("q\n")
            self._proc.stdin.flush()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


def open_publisher(connect_timeout=gz_io.PUBLISHER_START_TIMEOUT):
    try:
        from gz.transport13 import Node  # noqa: F401
    except Exception as e:  # noqa: BLE001 - 缺模块/共享库加载失败都算不可用
        inproc_err = f"当前解释器无 gz.transport13（{e}）"
    else:
        return _InProcPublisher(connect_timeout)
    try:
        return _HelperPublisher(connect_timeout)
    except Exception as e:  # noqa: BLE001
        raise gz_io.PublisherUnavailable(f"{inproc_err}；helper 不可用（{e}）") from e


def _pub_server(connect_timeout=gz_io.PUBLISHER_START_TIMEOUT):
    """helper 子进程入口：advertise 两个话题，stdin 逐行收 `a <action> <until>` / `c <text>`。"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        pub = _InProcPublisher(connect_timeout)
    except Exception as e:  # noqa: BLE001
        print(f"error {type(e).__name__}: {e}", flush=True)
        return 1
    print("ready", flush=True)
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line == "q":
            break
        if not line:
            continue
        try:
            kind, _, rest = line.partition(" ")
            if kind == "a":
                a, u = rest.split()
                pub.action(int(a), float(u))
            elif kind == "c":
                pub.cmd(rest)
            else:
                raise ValueError(f"未知指令 {kind!r}")
            print("ok", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"error {type(e).__name__}: {e}", flush=True)
    return 0


# ---------------------------------------------------------------- 执行器

class CfSitlExecutor:
    """collect_one 的 sil 后端（cf_sitl 档）。线程安全性：只在采集主线程调用。"""

    # collect_one 看这两个属性：窗口截止从动作窗口起点算；卡滞检测关闭（只认 contact）。
    until_from_window_start = True
    stuck_detection = False
    name = "cf_sitl"

    def __init__(self, params, hold_s, episode=0, takeoff_height=TAKEOFF_HEIGHT_M, log=None):
        if not (hold_s > 0):
            raise ValueError(f"hold_s 必须 > 0，得到 {hold_s!r}")
        self.params = params
        self.hold_s = float(hold_s)
        self.episode = int(episode)
        self.takeoff_height = float(takeoff_height)
        self._log = log or (lambda s: print(s, file=sys.stderr))
        self._sub = None
        self._pub = None
        self._err = None
        self._last_until = None
        self.dispatched = 0
        self.takeoff_failsafes = 0   # 起飞阶段进过几次 FAILSAFE（已 reset 重试）；写进 manifest
        self.reset_resends = 0       # reset 因首条消息丢失而重发的次数（只记日志）

    # -- 状态读取
    def state(self):
        msg = self._sub.latest() if self._sub is not None else None
        data = msg.get("data") if isinstance(msg, dict) else None
        if not isinstance(data, list) or len(data) < STATE_FIELDS:
            return None
        return data

    def exec_state(self):
        st = self.state()
        return int(st[IDX_EXEC]) if st else None

    def sim_time_s(self):
        st = self.state()
        return float(st[IDX_SIM_T]) if st else None

    def speed_mps(self):
        """最新 state 的世界系速度模长（真值，特权通道）；没有 state 时 None。"""
        st = self.state()
        if not st:
            return None
        vx, vy, vz = (float(v) for v in st[IDX_VEL:IDX_VEL + 3])
        return (vx * vx + vy * vy + vz * vz) ** 0.5

    def error(self):
        """执行器故障原因；正常返回 None。collect_one 每次轮询都会调。"""
        if self._err:
            return self._err
        if self._sub is not None and not self._sub.healthy():
            self._err = "state 订阅进程已退出"
        else:
            st = self.state()
            if st and int(st[IDX_EXEC]) == EXEC_FAILSAFE:
                self._err = "固件进入 FAILSAFE"
            elif st and st[IDX_ERRORS] > 0:
                self._err = f"插件 step_external 失败 {int(st[IDX_ERRORS])} 次"
        return self._err

    # -- 生命周期
    def _wait_state(self, pred, timeout, what, abort_check=None):
        deadline = time.monotonic() + timeout
        last_stamp, last_change = None, time.monotonic()
        while time.monotonic() < deadline:
            if not self._sub.healthy():
                raise RuntimeError(f"{what}: state 订阅进程已退出")
            st = self.state()
            if st is not None:
                stamp = st[IDX_SIM_T]
                if stamp != last_stamp:
                    last_stamp, last_change = stamp, time.monotonic()
                if int(st[IDX_EXEC]) == EXEC_FAILSAFE:
                    raise RuntimeError(f"{what}: 固件进入 FAILSAFE")
                if pred(st):
                    return st
            if time.monotonic() - last_change > STATE_STALE_S and last_stamp is not None:
                raise RuntimeError(f"{what}: state 停更 {STATE_STALE_S:.0f}s（仿真暂停或插件卡死）")
            if abort_check is not None:
                reason = abort_check()
                if reason:
                    raise RuntimeError(f"{what}: {reason}")
            time.sleep(POLL_S)
        st = self.state()
        where = (f"exec_state={EXEC_STATES.get(int(st[IDX_EXEC]), st[IDX_EXEC])} "
                 f"height={st[IDX_HEIGHT]:.2f} z={st[IDX_POS + 2]:.2f}" if st else "无 state")
        raise WaitTimeout(f"{what}超时 {timeout:.0f}s：{where}")

    def _reset_until_ack(self):
        """发 reset 直到 state 确认（DISARMED 且 SITL 时钟归零）。

        gz-transport 刚连上时首条消息可能丢（has_connections 为真不等于 ZeroMQ 链路已就绪；
        2026-11-07 真库采集 1/14 次 reset 未到达，等 5 s 超时后整集重跑）。reset 在解锁前幂等，
        所以每 RESET_RESEND_S 重发一次，直到 RESET_ACK_S 用完。返回发送次数。
        """
        deadline = time.monotonic() + RESET_ACK_S
        sends = 0
        while True:
            self._pub.cmd(f"reset {self.episode}")
            sends += 1
            try:
                self._wait_state(lambda st: int(st[IDX_EXEC]) == EXEC_DISARMED and st[IDX_T] < 0.5,
                                 min(RESET_RESEND_S, max(deadline - time.monotonic(), POLL_S)),
                                 "等 reset 生效")
            except WaitTimeout:
                if time.monotonic() >= deadline:
                    raise
                continue
            if sends > 1:
                self.reset_resends += sends - 1
                self._log(f"警告: reset 发了 {sends} 次才生效（首条消息丢失）")
            return sends

    def start(self, abort_check=None):
        """等 state → reset → arm_takeoff → READY。失败抛 RuntimeError（已尽量停桨）。"""
        self._sub = gz_io.TopicSub(STATE_TOPIC, buf=4, stamp_fn=state_stamp_ns)
        try:
            self._wait_state(lambda st: True, STATE_WAIT_S, "等 cf_sitl state 首帧")
            self._pub = open_publisher()
            self._log(f"cf_sitl 发布: {self._pub.mode}")
            for attempt in range(1 + TAKEOFF_RETRIES):
                self._reset_until_ack()   # reset 后 SITL 时钟归零、exec_state 回 DISARMED
                if attempt:   # FAILSAFE 后 reset 停桨，等机体落回地面再解锁
                    self._wait_state(lambda st: st[IDX_POS + 2] < 0.1 and abs(st[IDX_VEL + 2]) < 0.05,
                                     SETTLE_S, "等机体落地")
                self._pub.cmd(f"arm_takeoff {self.takeoff_height:g}")
                try:
                    st = self._wait_state(lambda st: int(st[IDX_EXEC]) == EXEC_READY, READY_TIMEOUT_S,
                                          "arm_takeoff→READY", abort_check)
                    break
                except RuntimeError as e:
                    if "FAILSAFE" not in str(e) or attempt == TAKEOFF_RETRIES:
                        raise
                    self.takeoff_failsafes += 1
                    self._log(f"警告: 起飞进入 FAILSAFE（第 {attempt + 1} 次，地面 ToF 假有效），reset 后重试")
            self._log(f"cf_sitl READY: height_est={st[IDX_HEIGHT]:.2f} m z={st[IDX_POS + 2]:.2f} m "
                      f"sitl_t={st[IDX_T]:.2f}s gz_t={st[IDX_SIM_T]:.2f}s"
                      + (f" 起飞重试 {self.takeoff_failsafes} 次" if self.takeoff_failsafes else ""))
        except BaseException:
            self.zero_now()
            raise

    def dispatch(self, action, t_ns):
        """动作 + 截止仿真秒 = t_ns/1e9 + hold。t_ns 是动作窗口起点（pose 戳）。"""
        if isinstance(action, bool) or not isinstance(action, int) or action not in VALID_ACTIONS:
            raise ValueError(f"cf_sitl 动作必须是 {VALID_ACTIONS} 之一，得到 {action!r}")
        t = int(t_ns)
        if t < 0:
            raise ValueError(f"cf_sitl dispatch 时间戳必须 >= 0，得到 {t}")
        if self._pub is None:
            raise RuntimeError("cf_sitl 执行器尚未 start()")
        until = t / 1e9 + self.hold_s
        self._pub.action(action, until)
        self._last_until = until
        self.dispatched += 1

    def set_action(self, action):
        now = self.sim_time_s()
        self.dispatch(action, int((now or 0.0) * 1e9))

    def zero_now(self):
        """安全网：SAFE 悬停，截止远在将来。从不抛。"""
        if self._pub is None:
            return
        try:
            now = self.sim_time_s() or 0.0
            self._pub.action(0, now + ZERO_UNTIL_S)
        except Exception as e:  # noqa: BLE001 - 安全网自身失败只能告警
            self._log(f"警告: cf_sitl zero_now 失败: {e}")

    def stop(self):
        self.zero_now()
        if self._pub is not None:
            try:
                self._pub.close()
            finally:
                self._pub = None
        if self._sub is not None:
            try:
                self._sub.close()
            finally:
                self._sub = None


def _self_check():
    p = CfSitlParams("x.json", b'{"mass_flight": {"value": 0.102}, "inertia_diag": {"value": [1, 2, 3]}}',
                     {"mass_flight": {"value": 0.102}, "inertia_diag": {"value": [1, 2, 3]}})
    assert p.profile().startswith(PROFILE_PREFIX + ":") and len(p.profile()) == len(PROFILE_PREFIX) + 9
    assert p.calibrated is False and p.manifest()["mass_flight"] == 0.102
    assert state_stamp_ns({"data": [0.0] * IDX_SIM_T + [1.25]}) == 1_250_000_000
    assert state_stamp_ns({"data": []}) == 0 and state_stamp_ns({}) == 0
    ex = CfSitlExecutor(p, hold_s=0.5, episode=7)

    class FakePub:
        sent = []
        mode = "fake"

        def action(self, a, u):
            self.sent.append((a, u))

        def cmd(self, t):
            self.sent.append(t)

        def close(self):
            pass

    ex._pub = FakePub()
    ex.dispatch(1, 2_000_000_000)
    assert FakePub.sent[-1] == (1, 2.5), FakePub.sent
    for bad in (4, -1, True, 1.0):
        try:
            ex.dispatch(bad, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝 {bad!r}")
    ex.zero_now()
    assert FakePub.sent[-1][0] == 0 and FakePub.sent[-1][1] >= ZERO_UNTIL_S
    assert ex.until_from_window_start and not ex.stuck_detection
    print("cf_sitl_exec self-check: PASS")
    return 0


if __name__ == "__main__":
    if "--pub-server" in sys.argv:
        sys.exit(_pub_server())
    if "--self-check" in sys.argv:
        sys.exit(_self_check())
    print(__doc__)
