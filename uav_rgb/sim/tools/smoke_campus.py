#!/usr/bin/env python3
"""三族真实 Gazebo 冒烟：只回收自己启动的进程，保留失败及原始轨迹。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import collect_parallel as cp
import eval_closed_loop as ec
import scene_gen as sg
import split_manifest as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", default="0,4,8,12,16,20",
                        help="默认三族地图各一对左右任务")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--out 必须为空或不存在")
    if args.steps < 1:
        parser.error("--steps 必须为正")
    for name in ("worlds", "episodes", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)
    options = SimpleNamespace(steps=args.steps, hold=1., window_from="dispatch",
                              policy="teacher", save_depth=False,
                              sim_timeout=120., episode_timeout=600.)
    template = sg._read_template(sg.DEFAULT_TEMPLATE)
    results = []
    for seed in ec.parse_seeds(args.seeds):
        world, meta = sg.build_scene(template, seed, seed, task="campus")
        wp = out / "worlds" / f"{seed}.sdf"
        mp = wp.with_suffix(".json")
        wp.write_text(world)
        mp.write_text(json.dumps(meta, indent=2))
        task = cp.Task(0, 0, seed, seed, sm.split_for_scene(meta), str(wp), str(mp))
        cp._run_episode_once(task, options, str(out / "episodes"), str(out / "logs"))
        ep = Path(task.staging)
        actions = ec.load_jsonl(str(ep / "actions.jsonl")) if (ep / "actions.jsonl").exists() else []
        oracle = ec.load_jsonl(str(ep / "oracle.jsonl")) if (ep / "oracle.jsonl").exists() else []
        summary = ec.summarize_episode(actions, oracle, 6., meta["waypoint"],
                                       stop_radius=meta["stop_radius"], goal_radius=meta["goal_radius"])
        results.append(dict(seed=seed, family=meta["family"], split=task.split,
                            collect_ok=task.ok, error=task.error, **summary))
        print(json.dumps(results[-1]), flush=True)
        (out / "summary.json").write_text(json.dumps(results, indent=2))
    return 0 if all(r["collect_ok"] and r["success"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
