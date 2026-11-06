#!/usr/bin/env python3
"""3DGS 在线采集：位姿 -> 动作 -> 黑盒动力学 -> 碰撞代理 -> gsplat 渲染 RGB。

与 sim/tools/collect.py 的 episode 格式对齐（episode_manifest.json / actions.jsonl /
oracle.jsonl / rgb_%05d.png），但**不启动 Gazebo**：位姿由 sim/tools/cf_blackbox.py
推进，高度固定 2 m，碰撞由 collision.py 的导航系 3D 占据代理沿每步位移扫掠给出
（近似，不是真值）。

每步顺序（与 collect.py 的 M1 一致）：渲染动作前 RGB -> 策略出动作 -> 动力学推进
-> 碰撞 -> 落盘。actions.jsonl 的 timestamp_ns 是动作前渲染时刻。

用法：
  python -m sim.3dgs.collect_3dgs --scene $SCENES/synthetic/scene.json \
      --out $DATA_ROOT/uav_rgb/3dgs-pilot-... --steps 8 --policy teacher
  python -m sim.3dgs.collect_3dgs --scene ... --out ... --mock-renderer --steps 3
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
from . import scene as scene_mod
from . import renderer as renderer_mod
from .dynamics import (CfBlackboxParams, CfSitlDynamics, CfSitlParams, FORWARD_SLOW,
                       make_blackbox)

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from teacher import ACTIONS, TEACHER_VERSION, expert_action  # noqa: E402
import split_manifest as sm  # noqa: E402

GEOMETRY_VERSION = "3dgs.v1"


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

    def __call__(self, rgb, depth, state):
        import base64
        h, w = rgb.shape[:2]
        msg = {"width": w, "height": h, "step": w * 3,
               "pixelFormatType": "RGB_INT8",
               "data": base64.b64encode(np.ascontiguousarray(rgb).tobytes()).decode()}
        action, infer_ms, _probs = self.student.act(msg)
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


def run_episode(scene, out_dir, renderer, occ, params, policy, steps, hold_s,
                split="train", seed=0, episode_id=None,
                save_depth=False, stop_on_collision=False, dynamics_path=None,
                spawn_jitter=0.0, identity=None, split_policy=None,
                dynamics="cf_blackbox"):
    """跑一集，返回 (manifest, error)。产物写 out_dir；异常也尽量落 manifest。

    identity：scene_identity() 的结果（gid/scene_id/task_id）；None 时现算。
    split_policy：None -> "geometry_gid"（split 必须等于 identity["split"]）；
    调用方强行指定别的 split 时传 "diag-manual-split"，preprocess 要 --allow-diagnostic-split。
    dynamics："cf_blackbox"（params 是 CfBlackboxParams，z 固定）或 "cf_sitl"（params 是
    CfSitlParams，E1 MINI 固件在环子进程，相机位姿用完整四元数，另写 trace.jsonl）。
    """
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
    stopped = False
    error = None
    if hasattr(policy, "prev"):        # 每集重置教师滞回
        policy.prev = None
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
            x0, y0, yaw0 = dyn.x, dyn.y, dyn.yaw
            z0 = dyn.z
            state = dyn.step(action, hold_s)
            x, y, yaw = state["x"], state["y"], state["yaw_deg"]
            # 沿整段位移扫掠（AUDIT #3）：只查终点时 0.69 m 一步能穿过 0.6 m 的障碍。
            collided, hit_pt = occ.sweep((x0, y0, z0), (x, y, state["z"]))
            dist = math.hypot(x - x0, y - y0)
            # yaw 是 state["yaw_deg"]（度），yaw0 是 dyn.yaw（弧度）：先统一成度再求 [-180,180) 环绕差。
            # 以前直接相减再 degrees()，实测左转 9.65° 记成 553°。
            dyaw = (yaw - math.degrees(yaw0) + 180.0) % 360.0 - 180.0
            ts0 = int(round(t_rgb * 1e9))
            ts1 = int(round(state["t"] * 1e9))
            actions.append({
                "step": i,
                "timestamp_ns": ts0,
                "action_expert": action,
                "action_executed": action,
                "collision": bool(collided),
                # 与 collect.py 对齐：src 区分碰撞来源；这里只有代理栅格。
                "src": "voxel_proxy" if collided else None,
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
            if stop_on_collision and collided:
                stopped = True
                print(f"step {i}: 碰撞，提前结束", file=sys.stderr)
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
            "teacher_version": TEACHER_VERSION if policy.name == "expert_action" else None,
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
                       "cx": renderer.width / 2.0, "cy": renderer.height / 2.0},
            "camera_source": "gsplat",
            "renderer": renderer.stats(),
            "collision_proxy": occ.summary(),
            "validity": {
                # 碰撞提前结束时也认为本集执行过；只要求每步都有帧、无异常。
                "camera_verified": bool(all_rgb_ok and rows_ok > 0),
                "physics_verified": bool(rows_ok > 0 and error is None),
                "episode_executed": bool(rows_ok > 0),
            },
            "num_steps": rows_ok,
            "requested_steps": steps,
            "performance": {
                "steps": rows_ok,
                "render_ms": {"mean": sum(render_ms) / len(render_ms) if render_ms else None,
                              "p50": _percentile(render_ms, 50),
                              "p95": _percentile(render_ms, 95)},
                "wall_s": round(time.perf_counter() - t_wall, 3),
            },
            "stopped_on_collision": stopped,
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
    p.add_argument("--policy", choices=("teacher", "forward"), default="teacher")
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
    p.add_argument("--stop-on-collision", action="store_true")
    p.add_argument("--dynamics", choices=("cf_blackbox", "cf_sitl"), default="cf_blackbox",
                   help="cf_sitl = E1 MINI 固件在环子进程（参数默认 sim/dynamics/e1_mini.json，"
                        "可执行文件取 $CF_SITL_PROC）")
    p.add_argument("--dynamics-params", default=None)
    p.add_argument("--spawn-jitter", type=float, default=0.0)
    args = p.parse_args()
    if os.path.isdir(args.out) and os.listdir(args.out):
        p.error(f"--out {args.out} 不是空目录")
    scene = scene_mod.load_scene(args.scene)
    steps = args.steps if args.steps is not None else int(scene.get("max_steps", 60))
    hold = args.hold if args.hold is not None else float(scene.get("step_s", 0.5))
    if steps < 1 or not (math.isfinite(hold) and hold > 0):
        p.error("steps 必须 >=1，hold 必须有限正数")
    if args.spawn_jitter < 0 or not math.isfinite(args.spawn_jitter):
        p.error("--spawn-jitter 必须有限且 >= 0")
    params = (CfSitlParams.from_json(args.dynamics_params) if args.dynamics == "cf_sitl"
              else CfBlackboxParams.from_json(args.dynamics_params))
    identity = scene_identity(scene)
    split, split_policy = identity["split"], "geometry_gid"
    if args.split is not None and args.split != split:
        split, split_policy = args.split, "diag-manual-split"
        print(f"警告: --split {args.split} 与几何组应属的 {identity['split']} 不同，"
              "manifest 记为 diag-manual-split（只能作诊断数据打包）", file=sys.stderr)
    occ = collision_mod.from_scene(scene)
    renderer = renderer_mod.make_renderer(scene, args.mock_renderer, args.width,
                                          args.height, args.fov, args.device)
    policy = DepthTeacher() if args.policy == "teacher" else ForwardPolicy()
    os.makedirs(args.out, exist_ok=True)
    episodes, failures = [], []
    for ep in range(args.episodes):
        ep_dir = os.path.join(args.out, "ep_%05d" % ep)
        if os.path.exists(ep_dir):
            p.error(f"{ep_dir} 已存在")
        try:
            manifest, err = run_episode(
                scene, ep_dir, renderer, occ, params, policy, steps, hold,
                split=split, seed=args.seed + ep,
                episode_id=args.seed + ep, save_depth=args.save_depth,
                stop_on_collision=args.stop_on_collision,
                dynamics_path=args.dynamics_params,
                spawn_jitter=args.spawn_jitter, identity=identity,
                split_policy=split_policy, dynamics=args.dynamics)
            episodes.append({"episode": ep, "dir": os.path.basename(ep_dir),
                             "steps": manifest["num_steps"],
                             "collision": any(a["collision"] for a in
                                              _read_jsonl(os.path.join(ep_dir, "actions.jsonl"))),
                             "error": err})
        except Exception as e:  # noqa: BLE001 - 单集失败不拖垮整轮
            failures.append({"episode": ep, "error": f"{type(e).__name__}: {e}"})
            print(f"episode {ep} 失败: {e}", file=sys.stderr)
    _write_json_atomic(os.path.join(args.out, "run.json"), {
        "schema": "uavrgb.3dgs_collect.v1",
        "scene": {"name": scene.get("name"), "path": scene["path"],
                  "ply": scene["ply_path"], **identity},
        "split": split, "split_policy": split_policy,
        "policy": args.policy, "steps": steps, "hold_s": hold,
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
