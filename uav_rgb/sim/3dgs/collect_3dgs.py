#!/usr/bin/env python3
"""3DGS 在线采集：位姿 -> 动作 -> 动力学 -> 碰撞代理 -> gsplat 渲染 RGB。

与 sim/tools/collect.py 的 episode 格式对齐（episode_manifest.json / actions.jsonl /
oracle.jsonl / rgb_%05d.png），但**不启动 Gazebo**：位姿默认由 E1 MINI 固件在环子进程
cf_sitl_proc 推进（--dynamics cf_sitl，固件定高；可执行文件取 $CF_SITL_PROC），
--dynamics cf_blackbox 才是 sim/tools/cf_blackbox.py 的一阶黑盒、高度固定 2 m（冒烟/单测用）。
碰撞由 collision.py 的导航系 3D 占据代理沿每步位移扫掠给出（近似，不是真值）。

每步顺序（与 collect.py 的 M1 一致）：渲染动作前 RGB -> 策略出动作 -> 动力学推进
-> 碰撞 -> 落盘。actions.jsonl 的 timestamp_ns 是动作前渲染时刻。

用法：
  python -m sim.3dgs.collect_3dgs --scene $SCENES/synthetic/scene.json \
      --out $DATA_ROOT/uav_rgb/3dgs-pilot-... --steps 8 --policy teacher
  python -m sim.3dgs.collect_3dgs --scene ... --out ... --mock-renderer --dynamics cf_blackbox --steps 3
  DAgger（学生执行、astar 教师打标签；tools/dagger_pipeline.py 调用）：
  python -m sim.3dgs.collect_3dgs --scene ... --route 1 --out ... --policy student --ckpt run/ckpt_best.bin \
      --beta 0.5 --dagger-round 1 --dynamics cf_sitl --hold 0.5
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

from . import collision as collision_mod
from . import planner as planner_mod
from . import scene as scene_mod
from . import renderer as renderer_mod
from .dynamics import (CfBlackboxParams, CfSitlDynamics, CfSitlParams, FORWARD_SLOW,
                       SAFE_NEUTRAL, make_blackbox)

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from teacher import ACTIONS, TEACHER_VERSION, expert_action  # noqa: E402
import split_manifest as sm  # noqa: E402

GEOMETRY_VERSION = "3dgs.v1"
# 倒飞保护（见 run_episode 注释）：FORWARD 下窗口末**机体系**前向速度 vx_b < -ANOMALY_BACK_VX 连续 ANOMALY_BACK_WINDOWS
# 个动作窗口判执行器失控。量机体系速度而不是世界位移沿起始航向的投影（REVIEW-3DGS #3）：拐角处连续 YAW 后机体还带着
# 横向动量，世界位移沿新航向的分量可以到 -0.06 m 而机体并没有后退（r1 ep0 第 94 步 vx_b -0.066 / vy_b +0.275 被旧判据
# 隔离）。阈值 0.10 m/s：cf_sitl 悬停漂移 1–2 cm/s、拐角残留 |vx_b| ≤ 0.07 m/s 都在内；真倒飞（route 5 seed 500，t≈44 s
# 光流速度环反向积分）首窗末 vx_b 已 < -0.1 且一路加速到 -2.5 m/s，第二窗就停。连续两窗是为了放过单窗口的瞬时负值。
ANOMALY_BACK_VX = 0.10
ANOMALY_BACK_WINDOWS = 2
# 低质量帧阈值：渲染帧灰度 4 邻域拉普拉斯方差（320x240）。Callinstrasse pilot 1306 帧（2026-11-07）
# 双峰：朝未观测植被背面的"雾墙"帧 1–30（18 帧，全在 route 1/4 出生点），30–60 一帧没有，正常帧 ≥100
# （p5=150，中位 415）。取空带上沿 60：只标雾帧，不碰 60–100 的 6 帧轻微糊。只标记不丢帧，丢不丢由
# preprocess 决定；分辨率/FOV 变了阈值要重标（tmp/agent3-3dgs/lapvar_hist.py）。
FRAME_LAPVAR_MIN = 60.0


def frame_quality(rgb, depth):
    """单帧质量：lapvar（灰度 4 邻域拉普拉斯方差）、hole_frac（alpha<0.05 即 depth==0 的像素占比）、low（雾/糊）。"""
    gray = np.asarray(rgb, dtype=np.float64).mean(axis=2)
    lap = 4.0 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    lapvar = float(lap.var())
    hole = float(np.mean(np.asarray(depth) <= 0.0))
    return {"lapvar": round(lapvar, 1), "hole_frac": round(hole, 4), "low": bool(lapvar < FRAME_LAPVAR_MIN)}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_identity(scene, ply_sha256=None):
    """3DGS 场景的稳定身份（AUDIT #1），与 seed/episode 无关。

    geometry_gid = sha256("3dgs:" + PLY 指纹 + nav_to_world)[:16]：同一份几何不论
    spawn/waypoint/seed 怎么变都是同一组，整组只能在一个 split（和 Gazebo 柱阵的
    geometry_gid 一样，出生点不同不算新几何；否则同一校园的两个任务可以一个进 train
    一个进 val，图像内容却相同）。scene_id = gid 前 8 位十六进制转 int（dataset.cpp
    只认整数）。task_id 单独记 spawn/waypoint，评测报告用它区分任务版本。
    split 走 split_manifest.split_for_gid，salt = geometry_version。
    """
    ply_sha256 = ply_sha256 or _sha256(scene["ply_path"])
    m = np.asarray(scene.get("_nav_to_world", np.eye(4)), dtype=np.float64)
    geom = json.dumps({"ply_sha256": ply_sha256,
                       "nav_to_world": [[round(float(v) + 0.0, 6) for v in row] for row in m]},
                      sort_keys=True, separators=(",", ":"))
    gid = hashlib.sha256(("3dgs:" + geom).encode("utf-8")).hexdigest()[:16]
    gver = scene.get("geometry_version") or GEOMETRY_VERSION
    task = json.dumps({"spawn": scene.get("spawn"), "waypoint": scene.get("waypoint"),
                       "reach_radius_m": scene.get("reach_radius_m")},
                      sort_keys=True, separators=(",", ":"))
    return {"geometry_gid": gid, "scene_id": int(gid[:8], 16),
            "geometry_version": gver, "split_salt": gver,
            "split": sm.split_for_gid(gid, gver), "ply_sha256": ply_sha256,
            "task_id": hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]}


def _write_png(path, rgb):
    import save_frame
    h, w, nch = rgb.shape
    with open(path, "wb") as f:
        f.write(save_frame._png(w, h, nch, np.ascontiguousarray(rgb).tobytes()))


def _write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _percentile(xs, p):
    if not xs:
        return None
    ordered = sorted(xs)
    k = max(0, min(len(ordered) - 1, int(math.ceil(p / 100.0 * len(ordered))) - 1))
    return ordered[k]


class DepthTeacher:
    """深度反应式专家（teacher.expert_action）；深度来自 gsplat 期望深度。"""

    name = "expert_action"

    def __init__(self):
        self.prev = None

    def __call__(self, rgb, depth, state):
        action = expert_action(depth.reshape(-1).tolist(), depth.shape[1],
                               depth.shape[0], prev_action=self.prev)
        self.prev = action
        return action, None


class ForwardPolicy:
    """盲前进基线；不读图像，用于打通链路/对照。"""

    name = "forward"

    def __call__(self, rgb, depth, state):
        return FORWARD_SLOW, None


class StudentAdapter:
    """student.py 的常驻 C++ 推理；把渲染 RGB 包成 gz.msgs.Image dict。

    StudentPolicy 自己把 320x240 缩到模型输入（默认 160x120），与 preprocess 一致。
    """

    name = "student"

    def __init__(self, student):
        self.student = student
        # 一个 adapter 对应一集；eval_3dgs 跨集复用同一个 StudentPolicy 进程。双帧 / prev-action 模型
        # 缓存上一帧与上一步动作，不清掉的话第 k≥1 集首帧拿的是上一集末帧（REVIEW-P2 实测 max|Δp|=0.225，
        # 第一步就可能错方向）；训练口径是首帧「上一帧=自身、prev=SAFE」。单帧模型 reset 是空操作。
        student.reset()

    def __call__(self, rgb, depth, state):
        import base64
        h, w = rgb.shape[:2]
        msg = {"width": w, "height": h, "step": w * 3,
               "pixelFormatType": "RGB_INT8",
               "data": base64.b64encode(np.ascontiguousarray(rgb).tobytes()).decode()}
        action, infer_ms, _probs = self.student.act(msg)
        return action, infer_ms


class DaggerPolicy:
    """DAgger 采集（Ross 2011）：学生看图出动作并执行，占据栅格 A* 教师按特权状态给标签。

    每步以概率 beta 改执行教师动作（β 混合；第 1 轮 0.5，之后 0）。标签口径与 --policy astar
    完全一样（teacher=plan_action, planner=astar_occ），只是 executed_by=student；
    信念软标签由 sim/tools/teacher_belief.py 离线按位姿重标，不在这里算。
    """

    name = "plan_action"

    def __init__(self, student, expert, beta, rng):
        self.student, self.expert, self.beta, self.rng = StudentAdapter(student), expert, float(beta), rng
        self.planner_name = expert.planner_name
        self.version = getattr(expert, "version", None)
        self.stop_radius = expert.stop_radius
        self.last_expert = None
        self.n_expert_exec = 0

    def reset(self):
        """集边界：清学生的帧/动作缓存（StudentAdapter 只在构造时 reset，而本对象整轮复用）。"""
        self.student.student.reset()
        self.last_expert = None

    @property
    def n_plan_fail(self):
        return self.expert.n_plan_fail

    @n_plan_fail.setter
    def n_plan_fail(self, v):
        self.expert.n_plan_fail = v

    def __call__(self, rgb, depth, state):
        self.last_expert, _ = self.expert(rgb, depth, state)
        action, infer_ms = self.student(rgb, depth, state)
        if self.beta > 0.0 and self.rng.random() < self.beta:
            action = self.last_expert
            self.n_expert_exec += 1
        return action, infer_ms


def _spawn_for_episode(scene, occ, jitter_m, rng):
    s = scene["spawn"]
    x, y, yaw = float(s["x"]), float(s["y"]), float(s.get("yaw_deg", 0.0))
    if jitter_m <= 0:
        return x, y, yaw
    for _ in range(100):
        nx = x + rng.uniform(-jitter_m, jitter_m)
        ny = y + rng.uniform(-jitter_m, jitter_m)
        if not occ.collides(nx, ny, float(s["z"])):
            return nx, ny, yaw + rng.uniform(-10.0, 10.0)
    raise RuntimeError(f"spawn 抖动 {jitter_m} m 内找不到自由点")


def run_episode(scene, out_dir, renderer, occ, params, policy, steps, hold_s, dagger=None,
                split="train", seed=0, episode_id=None,
                save_depth=False, stop_on_collision=False, dynamics_path=None,
                truncate_on_collision=False,
                spawn_jitter=0.0, identity=None, split_policy=None,
                dynamics="cf_blackbox", agl=None):
    """跑一集，返回 (manifest, error)。产物写 out_dir；异常也尽量落 manifest。

    identity：scene_identity() 的结果（gid/scene_id/task_id）；None 时现算。
    agl：planner.agl_flags 的结果（tof_terrain_m / agl_model_unverified / reason，AUDIT_SIM #2 打标）；
    None 时按「不知道」写（tof_terrain_m None、agl_model_unverified False、reason []），直接调用方自己负责。
    split_policy：None -> "geometry_gid"（split 必须等于 identity["split"]）；
    调用方强行指定别的 split 时传 "diag-manual-split"，preprocess 要 --allow-diagnostic-split。
    dynamics："cf_blackbox"（params 是 CfBlackboxParams，z 固定）或 "cf_sitl"（params 是
    CfSitlParams，E1 MINI 固件在环子进程，相机位姿用完整四元数，另写 trace.jsonl）。
    碰撞（AUDIT_SIM #3 / FIXLIST D2）：3DGS 没有接触物理，撞后机体穿模继续飞，所以第一次撞之后
    的每一行都写 collision=True（src=post_collision），preprocess --drop-collision 一并丢掉。
    stop_on_collision = 闭环评测语义（manifest stopped_on_collision=true，preprocess 整集拒收）；
    truncate_on_collision = 训练采集语义：撞的那一步写完就结束本集，只有那一行被 --drop-collision 丢，
    manifest 记 truncated_on_collision=true，stopped_on_collision 保持 false。两者互斥。
    """
    if stop_on_collision and truncate_on_collision:
        raise ValueError("stop_on_collision（闭环评测）与 truncate_on_collision（训练采集）互斥")
    os.makedirs(out_dir, exist_ok=True)
    identity = identity or scene_identity(scene)
    if split_policy is None:
        split_policy = "geometry_gid"
        if split != identity["split"]:
            raise ValueError(f"split={split} 与 geometry_gid={identity['geometry_gid']} "
                             f"按几何组应属的 split={identity['split']} 不一致")
    if save_depth:
        os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    rng = np.random.default_rng(seed)
    x, y, yaw = _spawn_for_episode(scene, occ, spawn_jitter, rng)
    if dynamics == "cf_sitl":
        dyn = CfSitlDynamics(scene, params, dynamics_path, seed=seed)
        try:
            # 噪声种子 = --seed ^ episode；seed 已按集递增，episode 固定 0（两者相等时异或为 0，所有集相同）。
            s = dyn.reset(x, y, yaw, episode=0)    # 地面起飞到 READY，起飞时间不计入 episode
        except BaseException:
            dyn.close()
            raise
        x, y, yaw = s["x"], s["y"], s["yaw_deg"]   # 起飞段有漂移，episode 从实际位姿开始
    elif dynamics == "cf_blackbox":
        dyn = make_blackbox(scene, dynamics_path, params)
        dyn.x, dyn.y = x, y
        dyn.yaw = math.radians(yaw)
    else:
        raise ValueError(f"dynamics 必须是 cf_blackbox 或 cf_sitl，得到 {dynamics!r}")
    actions, oracle, render_ms = [], [], []
    rows_ok, all_rgb_ok = 0, True
    stopped, truncated, hit_once = False, False, False
    # goal-stop（与 sim/tools/collect.py 同口径）：教师带 stop_radius（planner.OccAstarTeacher）时，
    # 连续 3 步 SAFE 且动作前位姿在 waypoint 半径内就提前结束，避免后 20+ 步全是悬停 SAFE 把类别
    # 分布拉偏；cf_sitl 悬停还会漂 2–4 cm/s，窄巷里挂着不动 10 s 就漂进树篱（smoke_r3 实测）。
    stop_radius = getattr(policy, "stop_radius", None)
    wp = scene.get("waypoint") or {}
    stop_streak, stopped_on_goal = 0, False
    anomaly, back_streak = None, 0
    n_low_quality = 0
    error = None
    if hasattr(policy, "prev"):        # 每集重置教师滞回
        policy.prev = None
    if hasattr(policy, "n_plan_fail"):   # A* 教师的规划失败计数是实例级的，按集归零才能落进本集 manifest
        policy.n_plan_fail = 0
    if hasattr(policy, "n_expert_exec"):
        policy.n_expert_exec = 0
    if hasattr(policy, "reset"):         # DAgger 学生跨集复用同一个推理进程：双帧 / prev-action 缓存按集清（同 StudentAdapter）
        policy.reset()
    t_wall = time.perf_counter()

    try:
        for i in range(steps):
            if dynamics == "cf_sitl":   # 机体四元数 ⊗ camera_mount：roll/pitch 进画面
                rgb, depth, ms = renderer.render_view(dyn.camera_viewmat(scene["_nav_to_world"]))
            else:
                rgb, depth, ms = renderer.render_pose(
                    x, y, dyn.z, yaw, pitch_deg=float(
                        scene.get("camera", {}).get("pitch_deg", 0.0)))
            render_ms.append(ms)
            t_rgb = dyn.t
            quality = frame_quality(rgb, depth)
            n_low_quality += int(quality["low"])
            rgb_ok = True
            try:
                _write_png(os.path.join(out_dir, "rgb_%05d.png" % i), rgb)
                if save_depth:
                    np.save(os.path.join(out_dir, "depth", "depth_%05d.npy" % i),
                            depth)
            except OSError as e:
                rgb_ok = False
                all_rgb_ok = False
                raise RuntimeError(f"写 RGB 失败: {e}") from e

            action, infer_ms = policy(rgb, depth, dyn)
            # DAgger：执行的是学生（或 β 抽中的教师）动作，标签始终是教师的；教师/forward 策略两者相同；
            # 非 DAgger 的学生集（eval_3dgs 的 StudentAdapter）没有教师标签，action_expert 写 None，不再把学生动作记成标签
            # （preprocess 按 teacher 门禁拒收这类集，不会读到 None）。
            expert = (policy.last_expert if isinstance(policy, DaggerPolicy)
                      else None if policy.name == "student" else action)
            x0, y0, yaw0 = dyn.x, dyn.y, dyn.yaw
            z0 = dyn.z
            state = dyn.step(action, hold_s)
            x, y, yaw = state["x"], state["y"], state["yaw_deg"]
            # 沿整段位移扫掠（AUDIT #3）：只查终点时 0.69 m 一步能穿过 0.6 m 的障碍。
            collided, hit_pt = occ.sweep((x0, y0, z0), (x, y, state["z"]))
            # 撞过一次之后机体已在障碍里/穿过障碍，后续画面与标签都不可信：整段都标碰撞（src 区分真撞与穿模）。
            src = "voxel_proxy" if collided else ("post_collision" if hit_once else None)
            hit_once = hit_once or bool(collided)
            dist = math.hypot(x - x0, y - y0)
            # yaw 是 state["yaw_deg"]（度），yaw0 是 dyn.yaw（弧度）：先统一成度再求 [-180,180) 环绕差。
            # 以前直接相减再 degrees()，实测左转 9.65° 记成 553°。
            dyaw = (yaw - math.degrees(yaw0) + 180.0) % 360.0 - 180.0
            # 固件在环失控保护：FORWARD 命令下机体系前向速度连续 ANOMALY_BACK_WINDOWS 窗 < -ANOMALY_BACK_VX（判据见常量注释）。
            # route 5 seed 500 实测：飞到 t≈44 s 光流速度环反向积分，FORWARD 下倒飞加速到 2.5 m/s，
            # 18 m 外撞树才被碰撞代理停下，中间 45 步标签 FORWARD 与运动相反。这里标记并结束本集，
            # validity.physics_verified=False 让 preprocess 拒收；不改 cf_sitl。
            # state 的 vx/vy 是世界(nav)系（cf_blackbox / cf_sitl_proc 都是），投影到窗口末航向得机体系前向分量。
            vx_body = state["vx"] * math.cos(state["yaw"]) + state["vy"] * math.sin(state["yaw"])
            back_streak = back_streak + 1 if (action == FORWARD_SLOW and vx_body < -ANOMALY_BACK_VX) else 0
            if back_streak >= ANOMALY_BACK_WINDOWS:
                anomaly = {"step": i, "kind": "backward_under_forward", "windows": back_streak,
                           "vx_body_mps": round(vx_body, 3), "displacement_m": round(dist, 3)}
            ts0 = int(round(t_rgb * 1e9))
            ts1 = int(round(state["t"] * 1e9))
            actions.append({
                "step": i,
                "timestamp_ns": ts0,
                "action_expert": expert,
                "action_executed": action,
                "collision": hit_once,
                # 与 collect.py 对齐：src 区分碰撞来源；voxel_proxy = 本步扫掠撞到，post_collision = 撞后穿模的步。
                "src": src,
                "collision_point": hit_pt,
                "rgb_ok": rgb_ok,
                "frame": "rgb_%05d.png" % i,
                "rgb_ts": ts0,
                "pose_ts": ts0,
                "pose_before": {"x": x0, "y": y0, "z": z0},
                "yaw_before": math.degrees(yaw0),
                "displacement_m": dist,
                "dyaw_deg": dyaw,
                "sim_dt": hold_s,
                "infer_ms": infer_ms,
                "render_ms": ms,
                "clearance_m": occ.clearance(x, y, state["z"]),
                "frame_quality": quality,       # lapvar / hole_frac / low（阈值 FRAME_LAPVAR_MIN）；只标记不丢帧
                "depth_file": ("depth/depth_%05d.npy" % i) if save_depth else None,
                "pose_gt": {"x": x, "y": y, "z": state["z"],
                            "yaw_deg": yaw},
            })
            oracle.append({
                "step": i,
                "timestamp_ns": ts1,
                "pose_gt": {"x": x, "y": y, "z": state["z"],
                            "yaw_deg": yaw},
                "action_start_ns": ts0,
                "action_end_ns": ts1,
            })
            rows_ok += 1
            print(f"step {i}: {ACTIONS[action]} pos=({x:.2f},{y:.2f},{state['z']:.2f}) "
                  f"yaw={yaw:.1f} d={dist:.2f} collide={collided} "
                  f"render={ms:.1f}ms infer={infer_ms}", file=sys.stderr)
            if anomaly is not None:
                print(f"step {i}: 动力学异常 {anomaly}，提前结束", file=sys.stderr)
                break
            if stop_on_collision and collided:
                stopped = True
                print(f"step {i}: 碰撞，提前结束", file=sys.stderr)
                break
            if truncate_on_collision and collided:
                truncated = True
                print(f"step {i}: 碰撞，本集到此截断（这一行留给 --drop-collision）", file=sys.stderr)
                break
            if stop_radius is not None:
                at_goal = math.hypot(x0 - float(wp["x"]), y0 - float(wp["y"])) <= stop_radius
                stop_streak = stop_streak + 1 if (action == SAFE_NEUTRAL and at_goal) else 0
                if stop_streak >= 3:
                    stopped_on_goal = True
                    print(f"step {i}: goal-stop 连续 3 步 SAFE，提前结束", file=sys.stderr)
                    break
    except BaseException as e:  # noqa: BLE001 - 异常也要写 manifest
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        _write_jsonl(os.path.join(out_dir, "actions.jsonl"), actions)
        _write_jsonl(os.path.join(out_dir, "oracle.jsonl"), oracle)
        if getattr(dyn, "trace", None) is not None:   # cf_sitl：hold 窗口内 10 Hz 中间状态（视频用）
            _write_jsonl(os.path.join(out_dir, "trace.jsonl"), dyn.trace)
        if hasattr(dyn, "close"):
            dyn.close()
        try:
            with open(scene["path"], encoding="utf-8") as f:
                scene_copy = json.load(f)
            ply = scene_copy.get("ply") or scene_copy.get("ply_path")
            if isinstance(ply, str) and not os.path.isabs(ply):
                key = "ply" if "ply" in scene_copy else "ply_path"
                scene_copy[key] = os.path.abspath(os.path.join(scene["dir"], ply))
            with open(os.path.join(out_dir, "scene.json"), "w", encoding="utf-8") as f:
                json.dump(scene_copy, f, ensure_ascii=False, indent=2)
        except (OSError, ValueError) as e:
            if error is None:
                error = f"复制 scene.json 失败: {e}"
        manifest = {
            "schema": "navlab.episode.v1",
            "episode_id": episode_id if episode_id is not None else seed,
            # 身份字段（AUDIT #1）：scene_id/geometry_gid 来自 PLY 指纹，不是 seed。
            "scene_id": identity["scene_id"],
            "seed": seed,
            "version": "3dgs-pilot-2",
            "split": split,
            "split_policy": split_policy,
            "split_salt": identity["split_salt"],
            "geometry_gid": identity["geometry_gid"],
            "task_id": identity["task_id"],
            "task_version": scene.get("task_version") or "3dgs.pilot.v1",
            "geometry_version": identity["geometry_version"],
            "appearance_version": scene.get("appearance_version") or "3dgs.v1",
            "route": scene.get("route"),      # scene.apply_route 选的路线（index/name）；旧 scene 为 None
            "waypoint": scene.get("waypoint"),
            "scene": {"name": scene.get("name"),
                      "source": scene.get("source"),
                      "license": scene.get("license"),
                      "ply_sha256": identity["ply_sha256"],
                      "scene_json_sha256": _sha256(scene["path"]),
                      "spawn": scene.get("spawn"), "waypoint": scene.get("waypoint")},
            "dynamics_backend": dynamics,
            "dynamics_profile": dyn.profile(),
            "dynamics_params": dyn.manifest(),
            "trace_ref": "trace.jsonl" if getattr(dyn, "trace", None) is not None else None,
            "teacher": policy.name,
            "planner": getattr(policy, "planner_name", None),
            # 谁执行了动作：缺省按策略名（forward 对照 / student = 学生自采或 eval_3dgs / 其余教师），DAgger 由 dagger
            # 字典覆盖；β = 执行教师动作的概率，学生执行缺省 0。以前学生 eval 集缺省写 teacher、β 1.0，与 teacher=student
            # 自相矛盾（R2-3DGS #2）。preprocess 只认 teacher 标签，executed_by/dagger_round/beta 进 episodes 表供溯源。
            "executed_by": (dagger or {}).get("executed_by",
                                              {"forward": "forward", "student": "student"}.get(policy.name, "teacher")),
            "dagger_round": (dagger or {}).get("dagger_round", 0),
            "beta": (dagger or {}).get("beta", 0.0 if policy.name == "student" else 1.0),
            "student_ckpt": (dagger or {}).get("student_ckpt"),
            "student_ckpt_sha256": (dagger or {}).get("student_ckpt_sha256"),
            "n_expert_executed": getattr(policy, "n_expert_exec", None),
            "teacher_version": (TEACHER_VERSION if policy.name == "expert_action"
                                else getattr(policy, "version", None)),
            "hold_s": hold_s,
            "window_from": "render_observation",
            "actions_ref": "actions.jsonl",
            "oracle_ref": "oracle.jsonl",
            "depth_ref": "depth/depth_%05d.npy" if save_depth else None,
            "depth_format": "npy_float32_m" if save_depth else None,
            "camera": {"width": renderer.width, "height": renderer.height,
                       "fx": scene_mod.intrinsics(renderer.width, renderer.height,
                                                  renderer.fov_deg)[0, 0],
                       "fy": scene_mod.intrinsics(renderer.width, renderer.height,
                                                  renderer.fov_deg)[1, 1],
                       "cx": renderer.width / 2.0, "cy": renderer.height / 2.0,
                       # 水平 FOV 与安装：cf_sitl 用 e1_mini.json camera_mount（含 xyz 偏移、pitch），
                       # cf_blackbox 只有 scene.camera.pitch_deg；写进 manifest 便于和 Gazebo 相机对账。
                       "fov_deg": renderer.fov_deg,
                       "mount": (dyn.params.camera_mount if dynamics == "cf_sitl"
                                 else {"xyz": [0.0, 0.0, 0.0],
                                       "pitch_deg": float(scene.get("camera", {}).get("pitch_deg", 0.0))})},
            "camera_source": "gsplat",
            # 低质量帧统计（阈值见 FRAME_LAPVAR_MIN）；每帧明细在 actions.jsonl 的 frame_quality。
            "frame_quality": {"lapvar_min": FRAME_LAPVAR_MIN, "n_low": n_low_quality,
                              "low_frac": round(n_low_quality / rows_ok, 4) if rows_ok else None},
            "renderer": renderer.stats(),
            "collision_proxy": occ.summary(),
            "validity": {
                # 碰撞提前结束时也认为本集执行过；只要求每步都有帧、无异常。「跑没跑完」看 completed。
                "camera_verified": bool(all_rgb_ok and rows_ok > 0),
                "physics_verified": bool(rows_ok > 0 and error is None and anomaly is None),
                "episode_executed": bool(rows_ok > 0),
            },
            # 与 sim/tools/collect.py 同义：跑满请求步数 或 goal-stop。碰撞截断 / 评测即停 / 动力学异常 /
            # error 的集都是 false；physics_verified 不含这层意思（AUDIT_SIM Nice 7，tools/DATASET_FORMAT.md §1.5）。
            "completed": bool(rows_ok > 0 and error is None and (rows_ok == steps or stopped_on_goal)),
            "dynamics_anomaly": anomaly,
            "num_steps": rows_ok,
            "requested_steps": steps,
            "performance": {
                "steps": rows_ok,
                "render_ms": {"mean": sum(render_ms) / len(render_ms) if render_ms else None,
                              "p50": _percentile(render_ms, 50),
                              "p95": _percentile(render_ms, 95)},
                "wall_s": round(time.perf_counter() - t_wall, 3),
            },
            # ToF 平地模型打标（AUDIT_SIM #2）：飞行层以下有桌面/长椅（tof_terrain_m = 最高格顶，nav 米）或路线名
            # 含 over_* 时 agl_model_unverified=true——真机 ToF 会按它爬升，这些集不算「2 m AGL」。ToF 输入没改。
            "tof_terrain_m": (agl or {}).get("tof_terrain_m"),
            "agl_model_unverified": bool((agl or {}).get("agl_model_unverified", False)),
            "agl_model_unverified_reason": list((agl or {}).get("agl_model_unverified_reason", [])),
            "stopped_on_collision": stopped,
            "truncated_on_collision": truncated,   # 训练采集撞后截断（不是闭环评测产物，preprocess 照收）
            "stop_radius": stop_radius,            # 教师 goal-stop 半径（深度教师/forward 为 None）
            "stopped_on_goal": stopped_on_goal,
            # A* 教师 fail-closed 输出 SAFE 的步数（被围死 / waypoint 在占据里）：这些 SAFE 在画面里没有
            # 悬停依据，>0 的集不该进训练集，demo_3dgs_routes 的健康检查按它标红。非 A* 教师为 None。
            "planner_failures": getattr(policy, "n_plan_fail", None),
            "error": error,
            "notes": [
                ("3DGS 在线渲染，无 Gazebo；位姿由 E1 MINI 固件在环（cf_sitl_proc）推进，"
                 "高度由固件定高，相机位姿含 roll/pitch。" if dynamics == "cf_sitl" else
                 "3DGS 在线渲染，无 Gazebo；位姿由 cf_blackbox 推进，z 固定。"),
                "碰撞是导航系 3D 体素占据代理（不透明度筛过的高斯椭球，沿每步位移扫掠），"
                "不是真实接触真值。",
            ],
        }
        _write_json_atomic(os.path.join(out_dir, "episode_manifest.json"), manifest)
    return manifest, error


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, help="scene.json")
    p.add_argument("--out", required=True, help="输出根目录（必须为空/新目录）")
    p.add_argument("--steps", type=int, default=None, help="每集步数（默认 scene.max_steps）")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--policy", choices=("teacher", "astar", "forward", "student"), default="teacher",
                   help="teacher = 深度反应式（不知道 waypoint，开阔路线全程 FORWARD）；"
                        "astar = 占据栅格 A* + 纯追踪 + goal-stop（sim/3dgs/planner.py，多路线用）；"
                        "student = DAgger：学生执行（--ckpt），astar 教师打标签，--beta 概率改执行教师")
    p.add_argument("--ckpt", default=None, help="--policy student 的学生 checkpoint")
    p.add_argument("--infer-bin", default=None, help="uavrgb_infer 路径（默认 $UAV_RGB_BUILD_DIR/uavrgb_infer）")
    p.add_argument("--beta", type=float, default=0.0, help="DAgger β：每步改执行教师动作的概率（第 1 轮 0.5，之后 0）")
    p.add_argument("--dagger-round", type=int, default=0, help="DAgger 轮次，写进 manifest（0 = 教师数据）")
    p.add_argument("--route", type=int, default=None,
                   help="scene.routes[N]：覆盖 spawn/waypoint/max_steps（默认 0 = 顶层 spawn/waypoint）")
    p.add_argument("--hold", type=float, default=None, help="每步动作窗口秒（默认 scene.step_s）")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fov", type=float, default=None)
    p.add_argument("--mock-renderer", action="store_true",
                   help="显式 mock 后端（仅冒烟；manifest 会记录 backend=mock）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=("train", "val", "test"), default=None,
                   help="默认按几何组哈希决定（同一 PLY 只能在一个 split）；显式指定且与之"
                        "不同时记为诊断划分 diag-manual-split，打包要 --allow-diagnostic-split")
    p.add_argument("--save-depth", action="store_true")
    p.add_argument("--stop-on-collision", action="store_true",
                   help="闭环评测语义：撞到即停，manifest stopped_on_collision=true（preprocess 拒收这种集）")
    p.add_argument("--truncate-on-collision", action="store_true",
                   help="训练采集语义：撞到就结束本集，只有撞的那一行 collision=True 留给 preprocess --drop-collision；"
                        "不给时撞后继续飞，之后每行都 collision=True")
    p.add_argument("--dynamics", choices=("cf_blackbox", "cf_sitl"), default="cf_sitl",
                   help="默认 cf_sitl = E1 MINI 固件在环子进程（参数默认 sim/dynamics/e1_mini.json，"
                        "可执行文件取 $CF_SITL_PROC，缺失直接报错）；cf_blackbox = 一阶黑盒、z 固定，只用于冒烟/单测")
    p.add_argument("--dynamics-params", default=None)
    p.add_argument("--spawn-jitter", type=float, default=0.0)
    args = p.parse_args()
    if os.path.isdir(args.out) and os.listdir(args.out):
        p.error(f"--out {args.out} 不是空目录")
    scene = scene_mod.load_scene(args.scene)
    try:
        scene_mod.apply_route(scene, args.route)
    except ValueError as e:
        p.error(str(e))
    steps = args.steps if args.steps is not None else int(scene.get("max_steps", 60))
    hold = args.hold if args.hold is not None else float(scene.get("step_s", 0.5))
    if steps < 1 or not (math.isfinite(hold) and hold > 0):
        p.error("steps 必须 >=1，hold 必须有限正数")
    if args.spawn_jitter < 0 or not math.isfinite(args.spawn_jitter):
        p.error("--spawn-jitter 必须有限且 >= 0")
    if args.policy == "student" and not (args.ckpt and os.path.isfile(args.ckpt)):
        p.error("--policy student 需要存在的 --ckpt")
    if not (0.0 <= args.beta <= 1.0):
        p.error("--beta 必须在 [0,1]")
    if args.stop_on_collision and args.truncate_on_collision:
        p.error("--stop-on-collision（闭环评测）与 --truncate-on-collision（训练采集）只能选一个")
    try:
        params = (CfSitlParams.from_json(args.dynamics_params) if args.dynamics == "cf_sitl"
                  else CfBlackboxParams.from_json(args.dynamics_params))
    except (FileNotFoundError, RuntimeError) as e:   # cf_sitl_proc 缺失 / 库版本探测失败：不悄悄退回黑盒
        p.error(f"--dynamics {args.dynamics}: {e}")
    identity = scene_identity(scene)
    split, split_policy = identity["split"], "geometry_gid"
    if args.split is not None and args.split != split:
        split, split_policy = args.split, "diag-manual-split"
        print(f"警告: --split {args.split} 与几何组应属的 {identity['split']} 不同，"
              "manifest 记为 diag-manual-split（只能作诊断数据打包）", file=sys.stderr)
    # 高斯源只读一次 PLY：飞行层占据 + 飞行层以下的 ToF 地形层（AUDIT_SIM #2 打标）共用。
    gaussians = (scene_mod.read_gaussian_ply(scene["ply_path"])
                 if scene.get("collision", {}).get("source", "gaussians") == "gaussians" else None)
    occ = collision_mod.from_scene(scene, gaussians=gaussians)
    terrain = collision_mod.terrain_from_scene(scene, gaussians=gaussians)
    del gaussians
    route_name = (scene.get("route") or {}).get("name")
    renderer = renderer_mod.make_renderer(scene, args.mock_renderer, args.width,
                                          args.height, args.fov, args.device)
    student, dagger = None, None
    if args.policy in ("astar", "student"):
        occ_planner = planner_mod.OccPlanner(occ)
        policy = planner_mod.OccAstarTeacher(occ_planner, scene["waypoint"])
        check = planner_mod.route_check(occ_planner, scene["spawn"], scene["waypoint"],
                                        terrain_occ=terrain, route_name=route_name)
        if not check["feasible"]:
            p.error(f"route {scene.get('route')}: spawn/waypoint 不可达 {check}")
        agl = {k: check[k] for k in ("tof_terrain_m", "agl_model_unverified", "agl_model_unverified_reason")}
        print(f"astar 教师: route={scene.get('route')} 航程 {check['path_m']} m 转向 {check['turn_deg']}° "
              f"最小净空 {check['min_clearance_m']} m 步数预算 {check['steps_estimate']}（--steps {steps}）",
              file=sys.stderr)
        if args.policy == "student":
            from student import StudentPolicy   # sim/tools（TOOLS 已在 sys.path）；常驻 uavrgb_infer 子进程
            import random
            student = StudentPolicy(args.ckpt, args.infer_bin)
            policy = DaggerPolicy(student, policy, args.beta, random.Random(args.seed))
            dagger = {"executed_by": "student", "dagger_round": args.dagger_round, "beta": args.beta,
                      "student_ckpt": os.path.abspath(args.ckpt), "student_ckpt_sha256": _sha256(args.ckpt)}
    else:
        policy = DepthTeacher() if args.policy == "teacher" else ForwardPolicy()
        # 没有 A* 路径：沿 spawn→waypoint 直线段查地形
        agl = planner_mod.agl_flags(terrain, planner_mod.segment_points(
            (scene["spawn"]["x"], scene["spawn"]["y"]), (scene["waypoint"]["x"], scene["waypoint"]["y"]), occ.voxel),
            route_name)
    if agl["agl_model_unverified"]:
        print(f"注意: route={scene.get('route')} agl_model_unverified {agl}（飞行层以下有桌面/长椅类地形或路线名 over_*：ToF 平地"
              "模型在真机上不成立，这些集只能当外观数据，不算 2 m AGL）", file=sys.stderr)
    del terrain
    os.makedirs(args.out, exist_ok=True)
    episodes, failures = [], []
    for ep in range(args.episodes):
        ep_dir = os.path.join(args.out, "ep_%05d" % ep)
        if os.path.exists(ep_dir):
            p.error(f"{ep_dir} 已存在")
        try:
            manifest, err = run_episode(
                scene, ep_dir, renderer, occ, params, policy, steps, hold, dagger=dagger,
                split=split, seed=args.seed + ep,
                episode_id=args.seed + ep, save_depth=args.save_depth,
                stop_on_collision=args.stop_on_collision,
                truncate_on_collision=args.truncate_on_collision,
                dynamics_path=args.dynamics_params,
                spawn_jitter=args.spawn_jitter, identity=identity,
                split_policy=split_policy, dynamics=args.dynamics, agl=agl)
            episodes.append({"episode": ep, "dir": os.path.basename(ep_dir),
                             "steps": manifest["num_steps"],
                             "collision": any(a["collision"] for a in
                                              _read_jsonl(os.path.join(ep_dir, "actions.jsonl"))),
                             "error": err})
        except Exception as e:  # noqa: BLE001 - 单集失败不拖垮整轮
            failures.append({"episode": ep, "error": f"{type(e).__name__}: {e}"})
            print(f"episode {ep} 失败: {e}", file=sys.stderr)
    if student is not None:
        student.close()
    _write_json_atomic(os.path.join(args.out, "run.json"), {
        "schema": "uavrgb.3dgs_collect.v1",
        "scene": {"name": scene.get("name"), "path": scene["path"],
                  "ply": scene["ply_path"], **identity},
        "route": scene.get("route"), "spawn": scene.get("spawn"), "waypoint": scene.get("waypoint"),
        "split": split, "split_policy": split_policy,
        "policy": args.policy, "steps": steps, "hold_s": hold,
        "dagger": dagger,
        "dynamics": args.dynamics,
        "dynamics_profile": params.profile(),
        "dynamics_params": params.manifest(),
        "renderer": renderer.stats(),
        "collision_proxy": occ.summary(),
        "episodes": episodes, "failures": failures,
    })
    print(json.dumps({"out": os.path.abspath(args.out), "episodes": episodes,
                      "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


def _read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    sys.exit(main())
