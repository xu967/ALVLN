#!/usr/bin/env python3
"""3DGS 在线评测：gsplat 渲染 320x240 RGB -> student.py 的 C++ 推理 -> 黑盒推进。

学生推理链路**完全复用** sim/tools/student.py（常驻 uavrgb_infer 子进程，自己把
320x240 缩到模型输入 160x120），本文件只负责在线环境。碰撞用点云切片体素代理；
成功 = 无碰撞且轨迹曾进入 waypoint 1.0 m（复用 eval_closed_loop.summarize_episode）。

用法：
  $DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m sim.3dgs.eval_3dgs \
      --scene $SCENES/synthetic/scene.json --out $DATA_ROOT/uav_rgb/3dgs-pilot-eval \
      --ckpt $DATA_ROOT/uav_rgb/airsim10k-run/ckpt_best.bin --episodes 1 --steps 20
  # 无学生权重时先冒烟链路：
  ... --policy forward --mock-renderer --episodes 1 --steps 3
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

from . import collision as collision_mod
from . import collect_3dgs
from . import renderer as renderer_mod
from . import scene as scene_mod
from .collect_3dgs import StudentAdapter
from .dynamics import CfBlackboxParams

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from eval_closed_loop import aggregate, summarize_episode  # noqa: E402


def _sha256(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True, help="输出根目录（必须为空/新目录）")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=None, help="每集最大步数")
    p.add_argument("--policy", choices=("student", "forward"), default="student")
    p.add_argument("--ckpt", default=None, help="学生 checkpoint（student 必填）")
    p.add_argument("--infer-bin", default=None)
    p.add_argument("--hold", type=float, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fov", type=float, default=None)
    p.add_argument("--mock-renderer", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-depth", action="store_true")
    p.add_argument("--dynamics-params", default=None)
    p.add_argument("--spawn-jitter", type=float, default=0.0)
    p.add_argument("--record-video", action="store_true",
                   help="每集把 rgb_*.png 合成 <out>/videos/seed_<seed>.mp4")
    args = p.parse_args()
    if args.policy == "student" and not args.ckpt:
        p.error("--policy student 需要 --ckpt")
    if args.ckpt and not os.path.isfile(args.ckpt):
        p.error(f"checkpoint 不存在: {args.ckpt}")
    if os.path.isdir(args.out) and os.listdir(args.out):
        p.error(f"--out {args.out} 不是空目录")
    scene = scene_mod.load_scene(args.scene)
    steps = args.steps if args.steps is not None else int(scene.get("max_steps", 60))
    hold = args.hold if args.hold is not None else float(scene.get("step_s", 0.5))
    if args.episodes < 1 or steps < 1 or not (math.isfinite(hold) and hold > 0):
        p.error("episodes/steps 必须 >=1，hold 必须有限正数")
    params = CfBlackboxParams.from_json(args.dynamics_params)
    occ = collision_mod.from_scene(scene)
    renderer = renderer_mod.make_renderer(scene, args.mock_renderer, args.width,
                                          args.height, args.fov, args.device)
    student = None
    if args.policy == "student":
        from student import StudentPolicy
        print("学生策略: 启动推理进程…", file=sys.stderr)
        student = StudentPolicy(args.ckpt, args.infer_bin)
    os.makedirs(args.out, exist_ok=True)
    results, failures = [], []
    started = time.monotonic()
    try:
        for ep in range(args.episodes):
            ep_dir = os.path.join(args.out, "ep_%05d" % ep)
            if os.path.exists(ep_dir):
                p.error(f"{ep_dir} 已存在")
            policy = (StudentAdapter(student) if student is not None
                      else collect_3dgs.ForwardPolicy())
            try:
                manifest, err = collect_3dgs.run_episode(
                    scene, ep_dir, renderer, occ, params, policy, steps, hold,
                    split="test", scene_id=args.seed, seed=args.seed + ep,
                    episode_id=args.seed + ep, save_depth=args.save_depth,
                    stop_on_collision=True,
                    dynamics_path=args.dynamics_params,
                    spawn_jitter=args.spawn_jitter)
            except Exception as e:  # noqa: BLE001 - 单集失败要进报告
                failures.append({"episode": ep, "seed": args.seed + ep,
                                 "error": f"{type(e).__name__}: {e}"})
                print(f"episode {ep} 失败: {e}", file=sys.stderr)
                continue
            actions = _read_jsonl(os.path.join(ep_dir, "actions.jsonl"))
            oracle = _read_jsonl(os.path.join(ep_dir, "oracle.jsonl"))
            wp = scene.get("waypoint")
            summary = summarize_episode(
                actions, oracle, goal_x=float(wp["x"]) if wp else 1e9,
                waypoint=wp, waypoint_radius=1.0, stop_radius=None,
                timed_out=False)
            summary.update({
                "seed": args.seed + ep, "episode": ep,
                "dir": os.path.basename(ep_dir),
                "policy": args.policy,
                "renderer_backend": manifest["renderer"]["backend"],
                "dynamics_profile": params.profile(),
                "collect_error": err,
                "wall_s": manifest["performance"]["wall_s"],
                "render_ms_p50": manifest["performance"]["render_ms"]["p50"],
                "actions": {},
            })
            for a in actions:
                k = int(a.get("action_executed", -1))
                summary["actions"][k] = summary["actions"].get(k, 0) + 1
            if args.record_video:
                from . import video as video_mod
                video_path = os.path.join(args.out, "videos",
                                          f"seed_{args.seed + ep}.mp4")
                try:
                    n = video_mod.compose_rgb_video(ep_dir, video_path, fps=5.0)
                    summary["video"] = os.path.relpath(video_path, args.out)
                    summary["video_frames"] = n
                except Exception as exc:  # noqa: BLE001 - 视频是诊断产物
                    summary["video_error"] = f"{type(exc).__name__}: {exc}"
            results.append(summary)
            print(f"[ep {ep}] steps={summary['steps']} "
                  f"collision={summary['collision']} success={summary['success']} "
                  f"max_x={summary['max_x']}", file=sys.stderr)
    finally:
        if student is not None:
            try:
                student.close()
            except Exception as e:  # noqa: BLE001 - 收尾失败要进报告
                failures.append({"episode": None, "error": f"学生收尾失败: {e}"})
    agg = aggregate(results, requested=args.episodes,
                    attempted=args.episodes, failed_count=len(failures))
    report = {
        "schema": "uavrgb.eval.v1",
        "policy": args.policy,
        "metric_version": "3dgs.pilot.v1",
        "success_criterion": "reach_waypoint_no_collision_v1",
        "scene": {"name": scene.get("name"), "path": scene["path"],
                  "ply_sha256": _sha256(scene["ply_path"]),
                  "source": scene.get("source"),
                  "license": scene.get("license")},
        "dynamics_profile": params.profile(),
        "dynamics_params": params.manifest(),
        "collision_proxy": occ.summary(),
        "renderer": renderer.stats(),
        "camera": {"width": renderer.width, "height": renderer.height,
                   "fov_deg": renderer.fov_deg},
        "ckpt": os.path.abspath(args.ckpt) if args.ckpt else None,
        "ckpt_sha256": _sha256(args.ckpt),
        "infer_bin": os.path.abspath(args.infer_bin) if args.infer_bin else None,
        "episodes_requested": args.episodes,
        "steps": steps, "hold_s": hold,
        "seeds": [args.seed + i for i in range(args.episodes)],
        "per_episode": results,
        "failed": failures,
        "wall_s": round(time.monotonic() - started, 2),
        **agg,
        "notes": [
            "3DGS 在线渲染 + student.py C++ 推理；无 Gazebo。",
            "碰撞是点云切片体素代理（近似），SR/碰撞率只能作为 pilot 链路验证，"
            "不是 Gazebo/AvoidBench 口径的正式数字。",
            "每集默认同 spawn；--spawn-jitter 可轻微扰动，但 pilot 不是统计基准。",
        ],
    }
    path = os.path.join(args.out, "eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"done-set SR={agg['success_rate']} "
          f"({sum(1 for e in results if e.get('success'))}/{agg['episodes_done']}) "
          f"collision={agg['collision_rate']} failed={len(failures)}")
    im = agg["inference_ms"]
    print(f"inference p50={im['p50']}ms p95={im['p95']}ms n_warm={im['n_warm']}")
    print(f"-> {path}")
    return 0 if len(results) == args.episodes else 1


if __name__ == "__main__":
    sys.exit(main())
