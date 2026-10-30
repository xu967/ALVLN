#!/usr/bin/env python3
"""E3 采集器：驱动四旋翼在 Gazebo 里跑专家策略，录 RGB/位姿/碰撞成完整 episode。

依赖一个已在运行的 gz sim（见 sim/README.md 启动命令）。teacher/forward 路径纯 stdlib；
student 路径需要 PIL 与 uavrgb_infer（见 sim/tools/student.py）。

用法：
  python3 collect.py --steps 20 --out /tmp/ep0 --scene-id 1 --seed 42 --split train
  python3 collect.py --steps 8 --out /tmp/ep1 --policy forward   # 盲策略（恒定前进）基线
  python3 collect.py --steps 80 --out /tmp/ep2 --policy student \
      --student-ckpt run/ckpt.bin --stop-on-collision           # 学生闭环评测

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
  oracle.jsonl           每步 step/timestamp_ns(动作后 pose)/pose_gt + 动作起止 sim-time
  rgb_%05d.png           320x240 原图（降采样到 160x120 由 tools/preprocess.py 负责）

碰撞真值（两种真实信号，任一触发即 collision=True）：
  1. /uav/contact 接触传感器（base_link + 四个旋翼的 contact sensor + 世界级
     gz-sim-contact-system）。该话题是稀疏/触发式的：只在接触时发布（实测无接触时
     0 msg/s，接地/碰撞时 ~225 msg/s）。常驻订阅按 **接收序号** 判窗口内是否收到
     非空 contact 列表——不用消息时间戳，也不用「最新帧」（缓冲里可能还是上一步的）。
  2. 位姿卡滞兜底：FORWARD_SLOW 指令下动作窗口位移 < STUCK_EPS 视为被挡。
"""
import argparse
import base64
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher import (expert_action, plan_action, plan_action_global,
                     SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT, ACTIONS,
                     TEACHER_VERSION)
import gz_io
import save_frame
import split_manifest as sm

TWIST_TOPIC = "/X3/gazebo/command/twist"
CAMERA_TOPIC = "/uav/camera"
CAMERA_INFO_TOPIC = "/uav/camera_info"
DEPTH_TOPIC = "/uav/depth"
POSE_TOPIC = "/model/x3/pose"
CONTACT_TOPIC = "/uav/contact"

RGB_FRAME_ID = "x3::base_link::camera"   # /uav/camera_info 上 RGB 与 depth 两个 publisher

GZ_TIMEOUT = 5.0      # 单次 gz 命令行调用上限（publisher 卡住时不永久挂起）
STUCK_EPS = 0.03      # FORWARD_SLOW 下动作窗口位移 < 3cm 视为卡滞（被挡）
FRAME_WAIT = 2.0      # 每步等一帧新 RGB 的上限（相机 30 Hz；卡住时报错而不是阻塞）
POSE_WAIT = 0.5       # 等一帧「晚于 RGB」的位姿的上限（M1；位姿 45 Hz）
READY_TIMEOUT = 10.0  # 启动时等 camera/depth/pose 首帧的上限
DEPTH_STALE_NS = 100_000_000  # depth 与 RGB 的 sim-time 差上限；超过视为停更/不同步
CONTACT_POLL = 0.02   # 动作窗口内轮询接触信号的间隔（接触时 225 Hz，缓冲覆盖 ~70 ms）
MIN_WINDOW_RTF = 0.05 # 动作窗口的墙钟兜底：仿真时间走不动时按此 RTF 折算等待上限

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

def open_subs():
    """给采集用到的 5 个话题各起一个常驻订阅；半路失败也要关掉已起的。"""
    subs = {}
    try:
        subs["rgb"] = gz_io.TopicSub(CAMERA_TOPIC, buf=4)
        subs["depth"] = gz_io.TopicSub(DEPTH_TOPIC, buf=4)
        subs["pose"] = gz_io.TopicSub(POSE_TOPIC, buf=6, stamp_fn=_pose_stamp)
        subs["contact"] = gz_io.TopicSub(CONTACT_TOPIC, buf=16)
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


def read_camera_info(sub, timeout=3.0):
    """从 /uav/camera_info 取 RGB 那条（frame_id=x3::base_link::camera）的内参。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for _, msg in sub.frames():
            cam = camera_from_info(msg)
            if cam:
                return cam
        if not sub.healthy():
            return None
        time.sleep(0.05)
    return None


def camera_from_info(msg):
    """gz.msgs.CameraInfo -> {width,height,fx,fy,cx,cy,frame_id}；非 RGB 相机返回 None。"""
    if not isinstance(msg, dict):
        return None
    header = msg.get("header") or {}
    frame_id = None
    for item in header.get("data") or []:
        if isinstance(item, dict) and item.get("key") == "frame_id":
            value = item.get("value")
            frame_id = value[0] if isinstance(value, list) and value else value
    if frame_id != RGB_FRAME_ID:
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
    """当前已收到的最新一条接触消息的序号（作为动作窗口的基线）。"""
    frames = sub.frames()
    return frames[-1][0] if frames else 0


def _contact_since(sub, seq0):
    """序号 > seq0 的帧里是否出现过非空 contact 列表。"""
    return any(seq > seq0 and _contact_nonempty(msg) for seq, msg in sub.frames())


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
                return
        time.sleep(0.1)
    publish_blocking(publisher, SAFE_NEUTRAL, 0.0)
    raise RuntimeError(f"起飞超时 {timeout:.0f}s：z={last_z}，目标 {z_target}")


# ---------------------------------------------------------------- 单步

def collect_one(out_dir, step_idx, hold, policy, subs, publisher,
                prev_action=None, prev_rgb_ts=0, pillars=None, student=None,
                global_plan=True, window_from="dispatch"):
    """采一步。成功返回 (row, None)，I/O 失败返回 (None, 原因)。"""
    if window_from not in ("dispatch", "observation"):
        raise ValueError(f"window_from 无效: {window_from!r}")
    t_step = time.perf_counter()

    # 1) 观测：以新的 RGB 帧为锚点，depth/pose 按 sim-time 配对
    rgb_msg = subs["rgb"].latest_after(prev_rgb_ts, timeout=FRAME_WAIT)
    if rgb_msg is None:
        return None, (f"camera {FRAME_WAIT:.0f}s 内无新帧"
                      if subs["rgb"].healthy() else "camera 订阅进程已退出")
    rgb_ts = stamp_ns(rgb_msg)
    if rgb_ts <= 0:
        return None, "RGB 帧缺少 header.stamp"

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
    if policy == "teacher":
        if pillars:
            # 有柱子真值就用 carrot planner：真值只决定标签，不进学生输入。
            # 位姿缺失时不能悄悄退回深度专家——manifest 已经记成 plan_action 了，
            # 混进去的深度专家标签事后看不出来。宁可让这一步失败。
            if not pose_before:
                return None, "carrot planner 需要位姿真值，但该步 pose 缺 position"
            # safe 出生用局部规划（要保留真正的 SAFE 标签）；普通集用 A* 全局路径。
            if global_plan:
                action = plan_action_global(pose_before["x"], pose_before["y"],
                                            yaw_before, pillars)
            else:
                action = plan_action(pose_before["x"], pose_before["y"], yaw_before,
                                     pillars)
        else:
            # 没有真值就退回深度反应式专家（传 prev_action 启用滞回）。
            action = expert_action(d, w, h, prev_action=prev_action)
    elif policy == "student":
        if student is None:
            return None, "student 策略未初始化"
        try:
            action, infer_ms, _probs = student.act(rgb_msg)
        except Exception as e:  # noqa: BLE001 - 推理进程任何失败都算本步失败
            return None, f"学生推理失败: {type(e).__name__}: {e}"
    else:  # policy == "forward": 盲策略（恒定前进），用于撞柱/基线对照
        action = FORWARD_SLOW
    t_io = time.perf_counter()

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

    # contact 是稀疏触发话题（无接触时 0 msg/s），不能把「长时间无消息」当停更；
    # 这里只拒绝订阅进程退出。话题静默/插件不发布的检测需要额外心跳，见 D4 记录。
    if not subs["contact"].healthy():
        return None, "contact 订阅进程已退出"
    contact_seq0 = _contact_seq(subs["contact"])
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
            action_handle = publisher.start(
                *twist_fields(action, altitude_vz(pose_for_action)))
        except Exception:
            # 动作没发出去也要确认没有旧指令在飞；失败原因继续上抛。
            _try_zero(publisher)
            raise

        # PNG 落盘落在 hold 窗口内（RGB 已在发指令前抓到，M1 不受影响）。
        rgb_ok, _ = write_rgb_png(
            rgb_msg, os.path.join(out_dir, "rgb_%05d.png" % step_idx))

        contact = False
        # 动作窗口按仿真时间计：动作前的位姿戳 + hold 秒。旧实现用墙钟，RTF 一抖
        # 实际走过的仿真时间就差得远（/tmp/ds200 实测 sim_dt 变异系数 18%），
        # 标签「前进 1 秒」到底代表多少位移变得不可复现。
        hold_end_ns = action_start_ns + int(hold * 1e9)
        wall_cap = t_action_start + hold / MIN_WINDOW_RTF   # 位姿停更时不死等
        while True:
            if not subs["contact"].healthy():
                _try_zero(publisher)
                return None, "动作窗口内 contact 订阅进程已退出"
            msg = subs["pose"].latest()
            if msg is not None and _pose_stamp(msg) >= hold_end_ns:
                break
            if not subs["pose"].healthy():
                # 订阅进程已死：不必等满墙钟兜底，也能给出比「RTF 过低」准确的诊断。
                _try_zero(publisher)
                return None, "动作窗口内 pose 订阅进程已退出"
            if time.perf_counter() >= wall_cap:
                _try_zero(publisher)   # 动作还在生效，先刹车再报错
                return None, (f"动作窗口超时：{hold / MIN_WINDOW_RTF:.1f}s 墙钟内"
                              f"仿真时间没走满 hold={hold}s（位姿停更或 RTF 过低）")
            time.sleep(CONTACT_POLL)
            if not contact:
                contact = _contact_since(subs["contact"], contact_seq0)

        # M2：hold 到点立刻发零速，并等它真正发布完成才继续（不丢零速）。
        try:
            zero_handle = publisher.start(*twist_fields(SAFE_NEUTRAL, 0.0))
            # 先收动作的发布完成时刻：若先等零速再等动作，动作的完成时刻
            # 会被记晚而误判。
            action_pub_wall = publisher.wait(action_handle)
            zero_pub_wall = publisher.wait(zero_handle)
        except Exception:
            _try_zero(publisher)
            raise
        action_wall = zero_pub_wall - action_pub_wall
        if action_wall <= 0:
            # 零速比动作先发布 => 动作可能一直生效；先补零速再失败。
            _try_zero(publisher)
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
        if not contact:
            contact = _contact_since(subs["contact"], contact_seq0)
        t_end = time.perf_counter()
    finally:
        publisher.abort(action_handle)
        publisher.abort(zero_handle)

    # 位移/偏航/sim_dt 用执行起点（派发位姿）；标签用的 pose_before 保持观测帧。
    sim_dt = (pose_ts - action_start_ns) / 1e9 if pose_ts and action_start_ns else 0.0
    dist = (_dist3(pose_exec, pose_after)
            if pose_exec and pose_after else None)
    dyaw = (_angle_diff(yaw_after, yaw_exec)
            if yaw_after is not None and yaw_exec is not None else None)
    # 低 RTF 时动作窗口的 sim 时间会变短，卡滞阈值跟着缩放，避免误报。
    stuck_eps = min(STUCK_EPS, 0.5 * SPEED_FWD * sim_dt)
    stuck = bool(action == FORWARD_SLOW and sim_dt > 0 and dist is not None
                 and dist < stuck_eps)

    return {
        "step": step_idx,
        "timestamp_ns": rgb_ts,        # 动作前 RGB 帧 sim-time（EpisodeStep）
        "rgb_ts": rgb_ts,
        "action_expert": action,
        "action_executed": action,     # 发布通道已确认成功，否则不会走到这里
        "collision": contact or stuck,
        "pose_gt": pose_after,
        "pose_ts": pose_ts,
        # 动作前位姿：audit 用它 + scene 柱子真值复算标签，验证「标签=该帧该做的动作」。
        "pose_before": pose_before,
        "yaw_before": yaw_before,
        "rgb_ok": rgb_ok,
        "frame": "rgb_%05d.png" % step_idx,
        "depth_ts": depth_ts,
        "pose_before_ts": pose_before_ts,
        "action_start_ns": action_start_ns,
        "action_end_ns": pose_ts,
        "displacement_m": dist,
        "dyaw_deg": dyaw,
        "sim_dt": sim_dt,
        "infer_ms": infer_ms,
        "action_wall_s": action_wall,
        "src": "+".join(s for s, on in (("contact", contact), ("stuck", stuck)) if on) or None,
        # 每步耗时分解（秒）：io=取帧+解码+teacher，window=动作到零速的窗口，
        # block=开始发动作到零速发布完成，tail=动作后位姿/接触/记账。
        "timing": {"io": t_io - t_step, "window": action_wall,
                   "block": t_block - t_action_start, "tail": t_end - t_block,
                   "total": t_end - t_step},
    }, None


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=sm.DEFAULT_STEPS)
    p.add_argument("--out", required=True)
    p.add_argument("--hold", type=float, default=1.0,
                   help="每动作持续时间（秒，仿真时间；走满即发零速）")
    p.add_argument("--window-from", choices=("dispatch", "observation"),
                   default="dispatch",
                   help="动作窗口起点：dispatch=派发位姿（默认，R01）；"
                        "observation=观测位姿（旧语义，仅诊断/兼容旧数据）")
    p.add_argument("--no-takeoff", action="store_true")
    p.add_argument("--policy", choices=("teacher", "forward", "student"), default="teacher",
                   help="teacher=专家策略；forward=盲策略（恒定前进）；student=模型闭环")
    p.add_argument("--student-ckpt", default=None, help="--policy student 的 checkpoint")
    p.add_argument("--infer-bin", default=None,
                   help="uavrgb_infer 路径（默认 src/build/uavrgb_infer）")
    p.add_argument("--stop-on-collision", action="store_true",
                   help="碰撞后立即结束本 episode（闭环评测用）")
    p.add_argument("--scene-id", type=int, default=0, help="EpisodeMeta.scene_id（场景隔离关键字段）")
    p.add_argument("--seed", type=int, default=0, help="EpisodeMeta.seed")
    p.add_argument("--episode-id", type=int, default=None, help="EpisodeMeta.episode_id（默认=seed）")
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--version", default="1", help="EpisodeMeta.version（采集器版本）")
    p.add_argument("--scene-meta", default=None,
                   help="scene_gen 产出的场景 JSON；给了就用柱子真值的 carrot planner，"
                        "否则用深度反应式专家")
    args = p.parse_args()
    # NaN <= 0 是 False，只判 <=0 会让 --hold nan 一路跑进 sleep 里挂死。
    if not (math.isfinite(args.hold) and args.hold > 0):
        p.error("--hold must be finite and > 0")
    if args.steps < 1:
        p.error("--steps must be >= 1")
    if args.policy == "student" and not args.student_ckpt:
        p.error("--policy student 需要 --student-ckpt")
    if os.path.isdir(args.out) and os.listdir(args.out):
        p.error(f"--out {args.out} 不是空目录；请换新目录（或先清空）")
    os.makedirs(args.out, exist_ok=True)
    pillars = None
    scene_meta_sha = None
    scene_gid = None
    scene_version = None
    scene_spawn_mode = None
    if args.scene_meta:
        with open(args.scene_meta, "rb") as f:
            raw_meta = f.read()
        scene_meta_sha = hashlib.sha256(raw_meta).hexdigest()
        scene_meta_obj = json.loads(raw_meta.decode("utf-8"))
        pillars = scene_meta_obj["pillars"]
        scene_version = scene_meta_obj.get("geometry_version")
        scene_spawn_mode = (scene_meta_obj.get("spawn") or {}).get("mode")
        for k, pl in enumerate(pillars):
            if "x" not in pl or "y" not in pl:
                p.error(f"--scene-meta {args.scene_meta}: pillars[{k}] 缺 x/y")
        # 柱子真值随 episode 一起落盘：audit 的 label_recompute 要用它复算标签。
        # 只有 collect_parallel 复制过 scene.json，单跑 collect.py 的集会永远 SKIP。
        write_scene_meta(os.path.join(args.out, "scene.json"), raw_meta)
        # split 由几何组唯一决定；手传的 --split 与它不符会把同几何放进两个 split。
        scene_gid = sm.geometry_gid(pillars)
        want_split = sm.split_for_gid(scene_gid)
        if args.split != want_split:
            p.error(f"--split {args.split} 与几何组 {scene_gid} 的固定 split={want_split} 不一致")
    # 标签来自哪个教师必须进 manifest：两种策略产出的数据长得一模一样，
    # 事后只能靠这个字段区分（旧版没有任何记录）。
    if args.policy == "forward":
        teacher_kind, planner_kind = "forward", None
    elif args.policy == "student":
        teacher_kind, planner_kind = "student", None
    elif pillars:
        teacher_kind = "plan_action"
        planner_kind = "plan_action" if scene_spawn_mode == "safe" else "astar"
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
    cam_info = None
    rows_written = 0
    all_rgb_ok = True
    all_pose_ok = True
    error = None
    completed = False
    stopped_on_collision = False
    prev_action = None
    prev_rgb_ts = 0
    timings = []
    try:
        subs = open_subs()
        not_ready = wait_ready(subs)
        if not_ready:
            raise RuntimeError(f"常驻订阅未就绪: {not_ready}")
        cam_info = read_camera_info(subs["camera_info"])
        if cam_info:
            print(f"相机内参: fx={cam_info['fx']:.3f} fy={cam_info['fy']:.3f} "
                  f"cx={cam_info['cx']:.3f} cy={cam_info['cy']:.3f} "
                  f"({cam_info['width']}x{cam_info['height']}, {RGB_FRAME_ID})",
                  file=sys.stderr)
        else:
            print("警告: 没读到 /uav/camera_info 的 RGB 内参，manifest 里 fx/fy/cx/cy 保持 null",
                  file=sys.stderr)
        try:
            publisher = gz_io.open_twist_publisher(TWIST_TOPIC)
            if getattr(publisher, "warning", None):
                print(f"警告: {publisher.warning}", file=sys.stderr)
            print(f"指令发布: {publisher.mode}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - 任何建不起来都直接失败
            raise RuntimeError(
                f"常驻 twist publisher 不可用（{e}）；定量采集不做 gz topic -p CLI"
                f" 回退（每条指令约 +1.0 s，动作窗口会失真）") from e

        if args.policy == "student":
            from student import StudentPolicy
            print("学生策略: 启动推理进程…", file=sys.stderr)
            student = StudentPolicy(args.student_ckpt, args.infer_bin)

        if not args.no_takeoff:
            print("起飞中…", file=sys.stderr)
            take_off(subs["pose"], publisher, subs["contact"])

        for i in range(args.steps):
            r, reason = collect_one(args.out, i, args.hold, args.policy, subs,
                                    publisher, prev_action, prev_rgb_ts, pillars, student,
                                    global_plan=(scene_spawn_mode != "safe"),
                                    window_from=args.window_from)
            if r is None:
                error = f"step {i}: {reason}"
                print(error + "，终止", file=sys.stderr)
                break
            prev_action = r["action_expert"]   # 下一步的滞回基准
            prev_rgb_ts = r["rgb_ts"]
            gap_ns = r["rgb_ts"] - r["depth_ts"] if r["depth_ts"] else None
            # 配对残差用**观测帧**位姿（pose_before_ts），action_start_ns 是派发时刻。
            pose_gap_ns = r["pose_before_ts"] - r["rgb_ts"]
            actions_f.write(json.dumps({k: r[k] for k in (
                "step", "timestamp_ns", "action_expert", "action_executed",
                "collision", "rgb_ok", "frame")} | {
                    # 评测端按 EVAL_STANDARD §5.1 只把 contact 当碰撞，stuck 单列。
                    "src": r["src"],
                    "rgb_ts": r["rgb_ts"],
                    "depth_ts": r["depth_ts"],
                    "depth_rgb_gap_ns": gap_ns,
                    # 配对用的观测位姿帧；动作后的 oracle 位姿只在 oracle.jsonl。
                    "pose_ts": r["pose_before_ts"],
                    "pose_rgb_gap_ns": pose_gap_ns,
                    "dyaw_deg": r["dyaw_deg"],
                    "displacement_m": r["displacement_m"],
                    "infer_ms": r["infer_ms"],
                    "action_start_ns": r["action_start_ns"],
                    # 动作前位姿：audit 的 label_recompute 靠它 + 柱子真值重跑 teacher。
                    "pose_before": r["pose_before"],
                    "yaw_before": r["yaw_before"]}) + "\n")
            actions_f.flush()
            oracle_f.write(json.dumps({"step": r["step"],
                                       "timestamp_ns": r["pose_ts"],
                                       "pose_gt": r["pose_gt"],
                                       "action_start_ns": r["action_start_ns"],
                                       "action_end_ns": r["action_end_ns"]}) + "\n")
            oracle_f.flush()
            rows_written += 1
            timings.append(r["timing"] | {"depth_gap_ns": gap_ns,
                                          "pose_gap_ns": pose_gap_ns,
                                          "rgb_before_start": r["rgb_ts"] < r["action_start_ns"]})
            all_rgb_ok = all_rgb_ok and r["rgb_ok"]
            all_pose_ok = all_pose_ok and r["pose_gt"] is not None
            dx = "?" if r["displacement_m"] is None else f"{r['displacement_m']:.3f}m"
            dyaw = "?" if r["dyaw_deg"] is None else f"{r['dyaw_deg']:.1f}deg"
            print(f"step {i}: {ACTIONS[r['action_expert']]} pos={r['pose_gt']} "
                  f"dx={dx} dyaw={dyaw} action_wall={r['action_wall_s']:.2f}s "
                  f"sim_dt={r['sim_dt']:.2f}s "
                  f"collision={r['collision']}({r['src'] or '-'}) rgb={r['rgb_ok']} "
                  f"rgb_ts={r['rgb_ts']} action_start_ts={r['action_start_ns']} "
                  f"rgb<start={r['rgb_ts'] < r['action_start_ns']} "
                  f"depth_gap={'-' if gap_ns is None else f'{gap_ns/1e6:.0f}ms'} "
                  f"pose_gap={'-' if pose_gap_ns is None else f'{pose_gap_ns/1e6:.0f}ms'} "
                  f"io={(r['timing']['io']):.2f}s step={(r['timing']['total']):.2f}s",
                  file=sys.stderr)
            if args.stop_on_collision and r["collision"]:
                stopped_on_collision = True
                print(f"step {i}: 碰撞，按 --stop-on-collision 结束", file=sys.stderr)
                break
        completed = rows_written == args.steps

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
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        # M3: 无论正常结束、深度失败、异常还是 Ctrl-C，先发零速悬停再收尾
        # （不能等订阅关闭，否则中断时可能带着最后一条动作指令多飞几秒）。
        try:
            _try_zero(publisher)
        finally:
            if publisher is not None:
                publisher.close()
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
            # 标签由哪个教师产生：forward=盲策略对照，expert_action=深度反应式，
            # plan_action=柱子真值 carrot planner。缺这个字段就无法事后区分数据集。
            "teacher": teacher_kind,
            # 教师实现版本：改规划参数/前向安全门后必须能区分新旧标签。
            "teacher_version": (TEACHER_VERSION if teacher_kind in ("plan_action",
                                                                    "expert_action")
                                else None),
            # plan_action 的具体实现：astar=全局路径（普通集），plan_action=局部
            # （safe 出生集，保留真 SAFE 标签）。audit 复算必须按这个字段选。
            "planner": planner_kind,
            # hold 是**仿真时间**秒（D1 起）：窗口按位姿戳走满这个数才发零速，
            # 与 RTF 无关。同 scene 的 0.5 与 1.0 两批数据仍靠它区分。
            "hold_s": args.hold,
            "window_from": args.window_from,
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
            "requested_steps": args.steps,
            # 评测集会在碰撞处提前结束；validity 仍按「跑满请求步数」严格填，
            # 下游据此区分「采集完整」与「评测终止」，不要混淆。
            "stopped_on_collision": stopped_on_collision,
            "error": error,
        }
        write_manifest(os.path.join(args.out, "episode_manifest.json"), manifest)
        print(f"写入 {rows_written}/{args.steps} 步 -> {args.out} "
              f"(episode_id={episode_id} scene_id={args.scene_id} seed={args.seed} "
              f"split={args.split} completed={completed} error={error})", file=sys.stderr)
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
