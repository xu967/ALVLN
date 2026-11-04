#!/usr/bin/env python3
"""3DGS 在线采集：位姿 -> 动作 -> 黑盒动力学 -> 碰撞代理 -> gsplat 渲染 RGB。

与 sim/tools/collect.py 的 episode 格式对齐（episode_manifest.json / actions.jsonl /
oracle.jsonl / rgb_%05d.png），但**不启动 Gazebo**：位姿由 sim/tools/cf_blackbox.py
推进，高度固定 2 m，碰撞由点云切片体素代理给出（近似，不是真值）。

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
from .dynamics import (CfBlackboxParams, FORWARD_SLOW, make_blackbox)

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from teacher import ACTIONS, TEACHER_VERSION, expert_action  # noqa: E402


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
                split="train", scene_id=0, seed=0, episode_id=None,
                save_depth=False, stop_on_collision=False, dynamics_path=None,
                spawn_jitter=0.0):
    """跑一集，返回 (manifest, error)。产物写 out_dir；异常也尽量落 manifest。"""
    os.makedirs(out_dir, exist_ok=True)
    if save_depth:
        os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    dyn = make_blackbox(scene, dynamics_path, params)
    rng = np.random.default_rng(seed)
    x, y, yaw = _spawn_for_episode(scene, occ, spawn_jitter, rng)
    dyn.x, dyn.y = x, y
    dyn.yaw = math.radians(yaw)
    actions, oracle, render_ms = [], [], []
    rows_ok, all_rgb_ok = 0, True
    stopped = False
    error = None
    if hasattr(policy, "prev"):        # 每集重置教师滞回
        policy.prev = None
    t_wall = time.perf_counter()

    try:
        for i in range(steps):
            rgb, depth, ms = renderer.render_pose(x, y, dyn.z, yaw)
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
            state = dyn.step(action, hold_s)
            x, y, yaw = state["x"], state["y"], state["yaw_deg"]
            collided = occ.collides(x, y, state["z"])
            dist = math.hypot(x - x0, y - y0)
            dyaw = math.degrees((yaw - yaw0 + 180.0) % 360.0 - 180.0)
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
                "rgb_ok": rgb_ok,
                "frame": "rgb_%05d.png" % i,
                "rgb_ts": ts0,
                "pose_ts": ts0,
                "pose_before": {"x": x0, "y": y0, "z": state["z"]},
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
            "scene_id": scene_id,
            "seed": seed,
            "version": "3dgs-pilot-1",
            "split": split,
            "geometry_gid": None,
            "task_version": scene.get("task_version") or "3dgs.pilot.v1",
            "geometry_version": scene.get("geometry_version") or "3dgs.v1",
            "appearance_version": scene.get("appearance_version") or "3dgs.v1",
            "scene": {"name": scene.get("name"),
                      "source": scene.get("source"),
                      "license": scene.get("license"),
                      "ply_sha256": _sha256(scene["ply_path"]),
                      "scene_json_sha256": _sha256(scene["path"])},
            "dynamics_profile": dyn.params.profile(),
            "dynamics_params": dyn.params.manifest(),
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
                "3DGS 在线渲染，无 Gazebo；位姿由 cf_blackbox 推进，z 固定。",
                "碰撞是点云切片体素代理（近似），不是真实接触真值。",
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
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--save-depth", action="store_true")
    p.add_argument("--stop-on-collision", action="store_true")
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
    params = CfBlackboxParams.from_json(args.dynamics_params)
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
                split=args.split, scene_id=args.seed, seed=args.seed + ep,
                episode_id=args.seed + ep, save_depth=args.save_depth,
                stop_on_collision=args.stop_on_collision,
                dynamics_path=args.dynamics_params,
                spawn_jitter=args.spawn_jitter)
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
                  "ply": scene["ply_path"]},
        "policy": args.policy, "steps": steps, "hold_s": hold,
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
