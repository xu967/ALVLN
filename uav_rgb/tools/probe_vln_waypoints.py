"""用冻结的机载图像对照教师下一动作或教师 A* 局部路标。

python -m tools.probe_vln_waypoints --manifest development.jsonl --check
python -m tools.probe_vln_waypoints --manifest development.jsonl --condition local-waypoint --out local.json
python -m tools.probe_vln_waypoints --teacher-run lamp-teacher --teacher-run cabinet-teacher --check
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


TEACHER_ACTION = {1: "C", 2: "L", 3: "R"}
DIRECTION = {0: "C", 1: "L", 7: "R"}
ALLOWED_MODES = {"baseline": ("final", "subgoal"), "local-waypoint": ("subgoal",)}


def make_grounder(condition):
    cls = importlib.import_module("sim.3dgs.grounder").VisualLanguageGrounder
    if condition == "baseline":
        return cls()
    if condition != "local-waypoint":
        raise ValueError(f"未知探针条件: {condition}")
    python = os.environ.get("UAV_RGB_VLM_PYTHON", cls.DEFAULT_PYTHON)
    return cls(worker_cmd=[python, cls.DEFAULT_WORKER, "--local-waypoint-probe"])


def _read_rgb(path):
    with Image.open(path) as image:
        frame = np.asarray(image.convert("RGB"))
    # 登记清单对解码后的 RGB 字节做哈希，两种 PNG 编码可以对应同一帧。
    return frame, hashlib.sha256(frame.tobytes()).hexdigest()


def load_pairs(manifest):
    data = manifest.read_bytes()
    first = {}
    for line in data.splitlines():
        row = json.loads(line)
        first.setdefault(row["pair"], row)
    if not first:
        raise ValueError("开发清单为空")

    samples = []
    for pair, row in first.items():
        actions, instructions = row["actions"], row["instructions"]
        if row["split"] != "development" or len(actions) != 2 or len(instructions) != 2:
            raise ValueError(f"{pair}: 需要开发集中的两条指令与教师动作")
        if any(type(a) is not int or a not in TEACHER_ACTION for a in actions) or actions[0] == actions[1]:
            raise ValueError(f"{pair}: 需要两种不同的前进、左转或右转动作")
        if any(not isinstance(t, str) or not t.strip() for t in instructions) or instructions[0] == instructions[1]:
            raise ValueError(f"{pair}: 需要两条不同的非空指令")
        path = manifest.parent / row["rgb"]
        frame, digest = _read_rgb(path)
        if digest != row["rgb_sha256"]:
            raise ValueError(f"{pair}: RGB 像素 SHA256 与登记不符 ({path})")
        queries = [{"instruction": text, "teacher": TEACHER_ACTION[action]}
                   for text, action in zip(instructions, actions)]
        sample = {k: row[k] for k in ("pair", "scene", "step", "arm", "routes", "shared_executed_state", "rgb_sha256")}
        samples.append((dict(sample, rgb=str(path), queries=queries), frame))
    return hashlib.sha256(data).hexdigest(), samples


def _source_bytes(path, hashes):
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def _teacher_frames(folder, episode, manifest, hashes):
    data = _source_bytes(folder / "actions.jsonl", hashes)
    actions = [json.loads(line) for line in data.splitlines()]
    if len(actions) != manifest["num_steps"] or len(actions) != episode["steps"]:
        raise ValueError(f"{folder}: 教师轨迹步数与清单不符")
    waypoint, radius = manifest["waypoint"], manifest["stop_radius"]
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"{folder}: 教师目标半径无效")
    covered, first = Counter(), {}
    near_steps = 0
    for step, action in enumerate(actions):
        if (action["step"] != step or action["frame"] != f"rgb_{step:05d}.png" or action["rgb_ok"] is not True
                or not (action["rgb_ts"] == action["pose_ts"] == action["timestamp_ns"])
                or action["action_executed"] != action["action_expert"]):
            raise ValueError(f"{folder} 第 {step} 步: 教师动作、RGB 或位姿记录不匹配")
        bearing, near = action["bearing_bin"], action["near_goal"]
        pos = action["pose_before"]
        distance = math.hypot(waypoint["x"] - pos["x"], waypoint["y"] - pos["y"])
        if (type(bearing) is not int or bearing not in range(9) or not math.isfinite(distance)
                or type(near) is not int or near != int(distance <= radius)):
            raise ValueError(f"{folder} 第 {step} 步: 教师方位或圈内标记无效")
        if near:
            near_steps += 1
            continue
        covered[bearing] += 1
        if bearing in DIRECTION:
            first.setdefault(bearing, action)

    samples = []
    for bearing, action in first.items():
        path = folder / action["frame"]
        if path.resolve().parent != folder:
            raise ValueError(f"图像不属于本集: {path}")
        frame, digest = _read_rgb(path)
        if frame.shape[:2] != (manifest["camera"]["height"], manifest["camera"]["width"]):
            raise ValueError(f"图像尺寸与相机清单不符: {path}")
        sample = {"key": episode["key"], "scene": episode["scene"], "seed": episode["seed"], "step": action["step"],
                  "bearing_bin": bearing, "rgb": str(path), "rgb_sha256": digest,
                  "queries": [{"instruction": episode["text"], "teacher": DIRECTION[bearing]}]}
        samples.append((sample, frame))
    coverage = {"key": episode["key"], "seed": episode["seed"], "start_pose": manifest["scene"]["spawn"],
                "steps": len(actions), "near_steps": near_steps,
                "outside_bearing_bins": dict(sorted(covered.items())), "selected_steps": [row["step"] for row, _ in samples]}
    return coverage, samples


def _teacher_episode(run, episode, hashes):
    scene, route = episode["scene"], episode["route_name"]
    folder = (run / scene / f"{route}_{episode['instruction_index']}").resolve()
    if folder.parent.parent != run or Path(episode["dir"]).parts[-2:] != folder.parts[-2:]:
        raise ValueError(f"{episode['key']}: 报告的轨迹目录与路线不符")
    manifest = json.loads(_source_bytes(folder / "episode_manifest.json", hashes))
    snapshot = json.loads(_source_bytes(folder / "scene.json", hashes))
    if (manifest["executed_by"] != "teacher" or manifest["planner"] != "astar_occ"
            or manifest["teacher_clearance"] != "braking" or manifest["dynamics_backend"] != "cf_sitl"
            or manifest["renderer"]["backend"] != "gsplat" or manifest["camera_source"] != "gsplat"):
        raise ValueError(f"{folder}: 需要真实 gsplat 图像和 cf_sitl 中执行的 A* braking 教师")
    validity = manifest.get("validity") or {}
    if not all(validity.get(field) is True for field in ("physics_verified", "camera_verified", "episode_executed")):
        raise ValueError(f"{folder}: 相机、物理或实际执行未核验")
    if not manifest["completed"] or episode.get("error") or manifest["actions_ref"] != "actions.jsonl":
        raise ValueError(f"{folder}: 教师轨迹不完整")
    originals = importlib.import_module("sim.3dgs.vln").episodes(snapshot)
    original = next((row for row in originals if row["key"] == episode["key"]), None)
    vln = manifest["vln"]
    if not original or manifest["route"]["name"] != route or manifest["scene"]["name"] != scene:
        raise ValueError(f"{folder}: 轨迹不属于登记路线")
    spawn = manifest["scene"]["spawn"]
    if (manifest["seed"] != episode["seed"] or episode.get("start_pose", spawn) != spawn
            or vln["blind_rgb"] is not False):
        raise ValueError(f"{folder}: 起点、种子或图像条件与报告不符")
    goal = snapshot["routes"][original["route_index"]]["waypoint"]
    for row in (episode, vln):
        if (any(row.get(k) != v for k, v in original.items()) or row["counterfactual"] != "correct"
                or row["supplied_text"] != original["text"] or row["supplied_instruction_key"] != original["key"]
                or row["supplied_goal_waypoint"] != goal or manifest["waypoint"] != goal):
            raise ValueError(f"{folder}: 场景原句、目标或轨迹身份不匹配")
    return _teacher_frames(folder, episode, manifest, hashes)


def load_teacher_runs(runs):
    coverage, samples, hashes = [], [], {}
    seen = set()
    for root in runs:
        run = root.resolve()
        report = json.loads(_source_bytes(run / "vln_eval.json", hashes))
        data = _source_bytes(run / "episodes.jsonl", hashes)
        episodes = [json.loads(line) for line in data.splitlines()]
        if (report["policy"] != "teacher" or report["dynamics"] != "cf_sitl" or report["counterfactual"] != "correct"
                or report["blind_rgb"] is not False or report["failures"] or report["overall"]["n_requested"] != len(episodes)):
            raise ValueError(f"{run}: 需要完整的原指令视觉教师评测")
        for episode in episodes:
            route, frames = _teacher_episode(run, episode, hashes)
            identity = (route["key"], route["seed"], tuple(sorted(route["start_pose"].items())))
            if identity in seen:
                raise ValueError(f"重复教师起点、指令与种子: {episode['key']}")
            seen.add(identity)
            coverage.append(route)
            samples.extend(frames)
    if not coverage:
        raise ValueError("教师评测清单没有轨迹")
    return {"scope": "offline_astar_local_bearing", "selection": "first_outside_step_per_route_per_present_bin_0_1_7",
            "routes": coverage, "source_sha256": hashes}, samples


def _query(request, frame, grounder, condition):
    grounder.reset_episode()
    observation = grounder.ground_step(frame, request["instruction"])
    direction = DIRECTION.get(observation.bearing_bin) if observation.goal_mode in ALLOWED_MODES[condition] else None
    result = dict(request, prediction=direction, observation=asdict(observation))
    if condition == "local-waypoint":
        probe = getattr(grounder, "last_waypoint_probe", None)
        probe = probe if isinstance(probe, dict) else None
        choice = probe.get("first_token_choice") if probe else None
        result.update(waypoint_probe=probe, first_token_prediction=choice if choice in DIRECTION.values() else None)
    return result


def query_pairs(samples, grounder, condition):
    results = []
    for pair, frame in samples:
        queries = [_query(request, frame, grounder, condition) for request in pair["queries"]]
        results.append(dict(pair, queries=queries))
    return results


def scores(pairs, always_center=False, paired=True, prediction_key="prediction"):
    by_teacher = {d: {"correct": 0, "total": 0} for d in TEACHER_ACTION.values()}
    both_correct = unknown = 0
    for pair in pairs:
        matches = []
        for request in pair["queries"]:
            teacher = request["teacher"]
            predicted = "C" if always_center else request[prediction_key]
            matches.append(predicted == teacher)
            by_teacher[teacher]["correct"] += predicted == teacher
            by_teacher[teacher]["total"] += 1
            unknown += predicted is None
        both_correct += all(matches)
    recalls = {d: c["correct"] / c["total"] if c["total"] else 0.0 for d, c in by_teacher.items()}
    total = sum(c["total"] for c in by_teacher.values())
    result = {"correct": sum(c["correct"] for c in by_teacher.values()), "requests": total,
              "unknown": unknown, "by_teacher": by_teacher, "recall_by_teacher": recalls,
              "balanced_accuracy": sum(recalls.values()) / len(recalls) if total else None}
    if paired:
        result.update(both_correct=both_correct, pairs=len(pairs))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="冻结的双指令开发集 JSONL")
    source.add_argument("--teacher-run", action="append", type=Path, help="含 episodes.jsonl 的真实教师评测目录，可重复传入")
    parser.add_argument("--condition", choices=ALLOWED_MODES, default="baseline")
    parser.add_argument("--check", action="store_true", help="核验选样和像素哈希，跳过模型推理")
    parser.add_argument("--out", type=Path, help="新建逐项报告 JSON，省略时写标准输出")
    args = parser.parse_args(argv)
    if args.out is not None and os.path.lexists(args.out):
        parser.error(f"输出已存在: {args.out}")
    paired = args.manifest is not None
    if paired:
        digest, samples = load_pairs(args.manifest.resolve())
        report = {"scope": "offline_teacher_action_proxy", "manifest": str(args.manifest.resolve()),
                  "manifest_sha256": digest, "selection": "first_row_per_pair_in_file_order"}
    else:
        report, samples = load_teacher_runs(args.teacher_run)
    pairs = [row for row, _ in samples]
    report.update(condition=args.condition, always_center=scores(pairs, always_center=True, paired=paired))
    if not args.check:
        worker = importlib.import_module("sim.3dgs.vlm_grounder_worker")
        prompt = worker.LOCAL_WAYPOINT_PROMPT if args.condition == "local-waypoint" else worker.PROMPT_TEMPLATE
        if samples:
            with make_grounder(args.condition) as model:
                pairs = query_pairs(samples, model, args.condition)
        report.update(scores=scores(pairs, paired=paired), model=worker.MODEL_PATH, max_new_tokens=worker.MAX_NEW_TOKENS,
                      prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        if args.condition == "local-waypoint":
            report["first_token_scores"] = dict(scores(pairs, paired=paired, prediction_key="first_token_prediction"),
                                                analysis="post_hoc_exploratory_conditional_argmax_L_C_R_N")
    item_key = "pairs" if paired else "frames"
    report[item_key] = pairs
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is None:
        print(payload)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.out.parent) as tmp:
        tmp.write(payload + "\n")
        tmp.flush()
        try:
            os.link(tmp.name, args.out)
        except FileExistsError:
            parser.error(f"输出已存在: {args.out}")
    print(json.dumps({k: v for k, v in report.items() if k != item_key}, ensure_ascii=False))


if __name__ == "__main__":
    main()
