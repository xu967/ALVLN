#!/usr/bin/env python3
"""3DGS 在线评测：gsplat 渲染 320x240 RGB -> student.py 的 C++ 推理 -> 黑盒推进。

学生推理链路**完全复用** sim/tools/student.py（常驻 uavrgb_infer 子进程，自己把
320x240 缩到模型输入 160x120），本文件只负责在线环境。碰撞用 collision.py 的导航系
3D 占据代理（沿每步位移扫掠）。
成功（3dgs.pilot.v2）= 无碰撞且轨迹曾进入 waypoint 的欧氏半径 reach_radius_m
（scene.json，缺省 eval_closed_loop.REACH_RADIUS=1.5 m）；不再看 max_x（AUDIT #10）。

hold（每步动作窗口）必须显式给：--hold，或 --data 指向采集 run 目录 / episode 目录 /
.uavr 数据集，从它的 hold_s 读；两者都给且不一致则报错（AUDIT #12）。

用法：
  $DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m sim.3dgs.eval_3dgs \
      --scene $SCENES/synthetic/scene.json --out $DATA_ROOT/uav_rgb/3dgs-pilot-eval \
      --ckpt $DATA_ROOT/uav_rgb/airsim10k-run/ckpt_best.bin --episodes 1 --steps 20 \
      --data $DATA_ROOT/uav_rgb/run/ds/dataset.uavr
  # 无学生权重时先冒烟链路：
  ... --policy forward --mock-renderer --episodes 1 --steps 3 --hold 0.5
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
from eval_closed_loop import REACH_RADIUS, aggregate, summarize_episode  # noqa: E402

METRIC_VERSION = "3dgs.pilot.v2"
SUCCESS_CRITERION = "reach_waypoint_euclid_no_collision_v2"


def reach_radius_of(scene):
    r = scene.get("reach_radius_m", REACH_RADIUS)
    if not (isinstance(r, (int, float)) and math.isfinite(r) and r > 0):
        raise ValueError(f"scene.reach_radius_m 必须是正数，实际 {r!r}")
    return float(r)


def summarize_3dgs_episode(actions, oracle, scene, timed_out=False):
    """单集指标：到达 = 轨迹上任一点到 waypoint 的欧氏距离 <= reach_radius_m。

    以前传 stop_radius=None 走 route-choice 老口径（max_x >= waypoint.x 且经过 1.0 m），
    离目标同为 0.5 m 时 x=-0.5 判失败、x=+0.5 判成功（AUDIT #10）。
    """
    wp = scene.get("waypoint")
    if not isinstance(wp, dict):
        raise ValueError("scene.json 缺 waypoint")
    radius = reach_radius_of(scene)
    # stop_radius 非空 -> summarize_episode 走欧氏半径分支（x,y 平面；3DGS 里 z 固定），
    # goal_x 在该分支不参与 success 判定，只为保持调用签名。
    s = summarize_episode(actions, oracle, goal_x=float(wp["x"]), waypoint=wp,
                          goal_radius=radius, stop_radius=radius, timed_out=timed_out)
    s["reach_radius_m"] = radius
    s["metric_version"] = METRIC_VERSION
    return s


def read_hold_from_data(path):
    """--data -> (hold_s, 来源说明)。支持 .uavr / 采集 run 目录(run.json) / episode 目录。"""
    path = os.path.abspath(path)
    if os.path.isfile(path) and path.endswith(".uavr"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(TOOLS)), "tools"))
        from train_student_with_video import read_uavr_meta
        meta, src = read_uavr_meta(path), "uavr meta"
    else:
        for name in ("run.json", "episode_manifest.json"):
            cand = os.path.join(path, name)
            if os.path.isfile(cand):
                with open(cand, encoding="utf-8") as f:
                    meta, src = json.load(f), name
                break
        else:
            raise FileNotFoundError(f"--data {path}：不是 .uavr，也没有 run.json/episode_manifest.json")
    hold = meta.get("hold_s")
    if not isinstance(hold, (int, float)) or isinstance(hold, bool) or not (hold > 0):
        raise ValueError(f"--data {path} 的 {src} 没有有效 hold_s（实际 {hold!r}）")
    return float(hold), f"data {src}"


def resolve_hold(hold_arg, data_arg):
    """--hold / --data -> (hold, hold_source)；都没有或不一致都报 ValueError。"""
    if hold_arg is None and data_arg is None:
        raise ValueError("必须显式给 --hold，或 --data 指向采集/数据集以读取 hold_s"
                         "（不再静默用 scene.step_s，否则评测窗口可能与训练数据不同）")
    if data_arg is not None:
        hold, source = read_hold_from_data(data_arg)
        if hold_arg is not None and abs(hold_arg - hold) > 1e-9:
            raise ValueError(f"--hold {hold_arg} 与 --data 的 hold_s={hold} 不一致")
        return hold, source if hold_arg is None else f"cli --hold (= {source})"
    return float(hold_arg), "cli --hold"


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
    p.add_argument("--hold", type=float, default=None,
                   help="每步动作窗口秒；与 --data 至少给一个，都给必须一致")
    p.add_argument("--data", default=None,
                   help="训练数据集 .uavr / 采集 run 目录 / episode 目录：从其 hold_s 取窗口")
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
                   help="每集把 rgb_*.png 合成 <out>/videos/seed_<seed>.webm（VP8，fps=1/hold）")
    args = p.parse_args()
    if args.policy == "student" and not args.ckpt:
        p.error("--policy student 需要 --ckpt")
    if args.ckpt and not os.path.isfile(args.ckpt):
        p.error(f"checkpoint 不存在: {args.ckpt}")
    if os.path.isdir(args.out) and os.listdir(args.out):
        p.error(f"--out {args.out} 不是空目录")
    scene = scene_mod.load_scene(args.scene)
    steps = args.steps if args.steps is not None else int(scene.get("max_steps", 60))
    # AUDIT #12：hold 不再静默回退 scene.step_s；来源写进报告。
    try:
        hold, hold_source = resolve_hold(args.hold, args.data)
    except (ValueError, FileNotFoundError, OSError) as e:
        p.error(str(e))
    if args.episodes < 1 or steps < 1 or not (math.isfinite(hold) and hold > 0):
        p.error("episodes/steps 必须 >=1，hold 必须有限正数")
    params = CfBlackboxParams.from_json(args.dynamics_params)
    occ = collision_mod.from_scene(scene)
    renderer = renderer_mod.make_renderer(scene, args.mock_renderer, args.width,
                                          args.height, args.fov, args.device)
    # 身份（AUDIT #1）：scene_id/geometry_gid 来自 PLY 指纹，不随 seed 变。评测集固定写
    # split="test"，这不是按几何组哈希分的，所以 split_policy 标成 diag-eval-fixed-test
    # （split_manifest 的 diag* 约定）：preprocess 默认拒绝，评测产物不会混进训练数据。
    identity = collect_3dgs.scene_identity(scene)
    split_policy = "diag-eval-fixed-test"
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
                    split="test", seed=args.seed + ep,
                    episode_id=args.seed + ep, save_depth=args.save_depth,
                    stop_on_collision=True,
                    dynamics_path=args.dynamics_params,
                    spawn_jitter=args.spawn_jitter,
                    identity=identity, split_policy=split_policy)
            except Exception as e:  # noqa: BLE001 - 单集失败要进报告
                failures.append({"episode": ep, "seed": args.seed + ep,
                                 "error": f"{type(e).__name__}: {e}"})
                print(f"episode {ep} 失败: {e}", file=sys.stderr)
                continue
            # AUDIT #4：run_episode 内部吞掉的异常（err 非空）或物理未校验，也算失败，
            # 不进 SR 分母，并让退出码非零。以前 err 只是 summary 里一个字段，SR 照算。
            if err or not manifest.get("validity", {}).get("physics_verified", False):
                failures.append({"episode": ep, "seed": args.seed + ep,
                                 "error": err or "validity.physics_verified=false",
                                 "dir": os.path.basename(ep_dir)})
                print(f"episode {ep} 失败: {err or 'physics_verified=false'}", file=sys.stderr)
                continue
            actions = _read_jsonl(os.path.join(ep_dir, "actions.jsonl"))
            oracle = _read_jsonl(os.path.join(ep_dir, "oracle.jsonl"))
            summary = summarize_3dgs_episode(actions, oracle, scene, timed_out=False)
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
                                          f"seed_{args.seed + ep}.webm")
                try:
                    # 每帧代表一个 hold 窗口：fps=1/hold 才是仿真时间 1:1（以前固定 5 fps，
                    # hold=0.5 s 时播放快 2.5 倍）。.webm 为 VP8，浏览器可直接播放。
                    n = video_mod.compose_rgb_video(ep_dir, video_path, fps=1.0 / hold)
                    summary["video"] = os.path.relpath(video_path, args.out)
                    summary["video_frames"] = n
                    summary["video_fps"] = 1.0 / hold
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
        "metric_version": METRIC_VERSION,
        "success_criterion": SUCCESS_CRITERION,
        "reach_radius_m": reach_radius_of(scene),
        # 任务身份：同一 PLY 改了 waypoint/spawn 就是另一个任务；以前 eval.json 只有 PLY 指纹，
        # 目标从 -6 挪到 -7.5 的两次评测在报告层面完全一样（AUDIT #17）。
        "scene": {"name": scene.get("name"), "path": scene["path"],
                  "ply_sha256": identity["ply_sha256"],
                  "scene_json_sha256": _sha256(scene["path"]),
                  "geometry_gid": identity["geometry_gid"],
                  "scene_id": identity["scene_id"],
                  "task_id": identity["task_id"],
                  "split": "test", "split_policy": split_policy,
                  "waypoint": scene.get("waypoint"),
                  "spawn": scene.get("spawn"),
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
        "steps": steps, "hold_s": hold, "hold_source": hold_source,
        "seeds": [args.seed + i for i in range(args.episodes)],
        "per_episode": results,
        "failed": failures,
        "wall_s": round(time.monotonic() - started, 2),
        **agg,
        "notes": [
            "3DGS 在线渲染 + student.py C++ 推理；无 Gazebo。",
            "碰撞是导航系 3D 体素占据代理（不透明度筛过的高斯椭球沿每步位移扫掠，近似），"
            "SR/碰撞率只能作为 pilot 链路验证，不是 Gazebo/AvoidBench 口径的正式数字。",
            "到达 = 轨迹任一点到 waypoint 欧氏距离 <= reach_radius_m，与接近方向无关。",
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
    # AUDIT #4：任何一集失败都列出来并以非零退出，调用方（训练驱动/CI）不能当成功。
    for fl in failures:
        print(f"FAILED episode={fl.get('episode')} seed={fl.get('seed')}: {fl['error']}",
              file=sys.stderr)
    print(f"-> {path}")
    return 0 if (len(results) == args.episodes and not failures) else 1


if __name__ == "__main__":
    sys.exit(main())
