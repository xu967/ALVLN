"""从两次独立的教师仿真采集核验同图双指令的局部航点。"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .vln_fork_geometry import label_goal
from .vln_fork_sources import DEFAULT_ROOT, decoded_rgb_sha, pre_action_state, require


SCENE = importlib.import_module("sim.3dgs.scene")
COLLISION = importlib.import_module("sim.3dgs.collision")
PLANNER = importlib.import_module("sim.3dgs.planner")
REACH = importlib.import_module("sim.3dgs.eval_3dgs").reach_radius_of
ROOMS = {"train": {"840139", "841012", "841150"}, "development": {"839992", "840865"}}
DESIGN_SCHEMA = "uavrgb.vln.shared_fork_design.v1"
PILOT_SCHEMA = "uavrgb.vln.shared_forks.pilot.v1"
PROTOCOL_SHA = "75bf757a4baa6a23da6bdae7a0770ae099d303d417013a71334632e5299ec108"
SOURCE_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError, EOFError)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def xyz(point):
    return tuple(point[k] for k in "xyz")


def load_room(data_root, room, joins):
    path = data_root / "3dgs-scenes" / ("sage3d_" + room) / "scene.json"
    scene = SCENE.load_scene(str(path))
    mesh = (path.parent / scene["collision"]["geometry"]).resolve()
    native, = (data_root / "pubdata/sage3d/vln" / room / "train").glob("train_trajectories_overall_*.json")
    annotations, = read(native)["scenes"]
    require(annotations["scene_id"] == room, f"Wrong native room {native}")
    params_path = Path(__file__).resolve().parents[1] / "sim/dynamics/e1_mini.json"
    return dict(scene=scene, planner=PLANNER.OccPlanner(COLLISION.from_scene(scene)),
                routes={r["name"]: r for r in scene["routes"]}, joins=joins,
                samples={int(r["trajectory_id"]): r for r in annotations["samples"]}, mesh=mesh,
                params_sha=sha(params_path), camera_mount=read(params_path)["camera_mount"]["value"],
                hashes={str(p): sha(p) for p in (path, Path(scene["ply_path"]), mesh, native)})


def load_arm(pair, design_arm, room, base):
    fixture_path, out = (base / design_arm[k] for k in ("fixture", "out"))
    require(sha(fixture_path) == design_arm["fixture_sha256"], f"Fixture SHA-256 differs: {fixture_path}")
    fixture = read(fixture_path)
    route, = fixture["routes"]
    name, target, text = (design_arm[k] for k in ("route", "target", "instruction"))
    native = room["routes"][name]
    index = design_arm["source_instruction_index"]
    task = native["sage"]["instructions"][index]
    sample = room["samples"][int(native["sage"]["trajectory_id"])]
    raw = sample["instructions"][index]
    route_key = room["scene"]["name"] + "/" + name
    start, end = (room["joins"][route_key, k] for k in ("start", "end"))
    require(task == dict(text=raw["generated_instruction"], start=raw["start"], end=raw["end"],
                         instruction_type=raw["instruction_type"]) and route["sage"]["instructions"] == [task]
            and text == task["text"] and target == task["end"] and native["sage"]["split"] == route["sage"]["split"] == "train"
            and end["target"] == target and str(end["ins_id"]) == str(design_arm["ins_id"]) and start["target"] == task["start"],
            f"Native instruction or instance differs for {name}")
    require(route["name"] == name and route["waypoint"] == native["waypoint"] == fixture["waypoint"]
            and math.dist(xyz(route["waypoint"])[:2], sample["points"][-1]["position"][:2]) < 1e-6
            and math.dist(xyz(native["spawn"]), xyz(pair["spawn"])) <= 1e-5
            and math.dist(xyz(native["spawn"])[:2], sample["points"][0]["position"][:2]) <= 1e-5
            and route["spawn"] == pair["spawn"] == fixture["spawn"], f"Wrong spawn or endpoint for {name}")
    if "source_scene_sha256" in design_arm:
        require(room["hashes"][room["scene"]["path"]] == design_arm["source_scene_sha256"], f"Native scene changed for {name}")
    if "start_instance" in design_arm:
        require(str(start["ins_id"]) == str(design_arm["start_instance"]) and native["spawn"] == design_arm["native_spawn"],
                f"Native start changed for {name}")
    ep, = jsonl(out / "episodes.jsonl")
    logs = out / fixture["name"] / (name + "_0")
    report = read(out / "vln_eval.json")
    manifest = read(logs / "episode_manifest.json")
    require(Path(ep["dir"]).resolve() == logs.resolve() and read(logs / "scene.json") == fixture
            and manifest["scene"]["scene_json_sha256"] == design_arm["fixture_sha256"], f"Wrong run snapshot for {name}")
    for source in (ep, manifest["vln"]):
        require(source["text"] == source["supplied_text"] == text and source["route_name"] == name
                and source["goal_object"] == source["supplied_goal_object"] == target and source["start_object"] == raw["start"]
                and source["supplied_goal_waypoint"] == route["waypoint"] and source["counterfactual"] == "correct",
                f"Logged task differs from registered task for {name}")
    require(report["policy"] == manifest["executed_by"] == "teacher" and report["teacher_clearance"] == manifest["teacher_clearance"] == "braking"
            and report["hold_s"] == manifest["hold_s"] == .5 and report["dynamics"] == manifest["dynamics_backend"] == "cf_sitl"
            and report["grounder"] == "none" and report["stop_mode"] == "explicit" and not report["blind_rgb"]
            and not report["failures"] and report["overall"]["n_requested"] == 1 and not manifest["vln"]["blind_rgb"]
            and manifest["planner"] == "astar_occ" and manifest["camera_source"] == manifest["renderer"]["backend"] == "gsplat"
            and all(manifest["validity"][k] for k in ("camera_verified", "physics_verified", "episode_executed")),
            f"Incomplete or incorrect teacher run for {name}")
    require(manifest["dynamics_params"]["trace_dt_s"] == .1 and manifest["requested_steps"] == pair["common_budget"]
            and ep["seed"] == report["seed"] == manifest["seed"] == manifest["dynamics_params"]["seed"] == pair["seed"]
            and manifest["depth_format"] == "npy_float32_m" and manifest["stop_on_request"]
            and manifest["scene"]["spawn"] == pair["spawn"] and manifest["scene"]["waypoint"] == route["waypoint"]
            and manifest["waypoint"] == route["waypoint"], f"Wrong teacher seed, budget, depth or endpoint for {name}")
    scene = room["scene"]
    camera, proxy = manifest["camera"], manifest["collision_proxy"]
    require(scene["nav_to_world"] == fixture["nav_to_world"] and scene["camera"] == fixture["camera"]
            and (fixture_path.parent / fixture["ply"]).resolve() == Path(scene["ply_path"]).resolve()
            and room["hashes"][scene["ply_path"]] == manifest["scene"]["ply_sha256"]
            and (fixture_path.parent / fixture["collision"]["geometry"]).resolve() == Path(proxy["geometry"]).resolve() == room["mesh"]
            and proxy["n_occupied_cells"] == room["planner"].occ.n_occupied and proxy["z_range"] == scene["collision"]["z_slice"]
            and proxy["collision_model"] == "v1" and REACH(scene) == REACH(fixture) == manifest["stop_radius"],
            f"Teacher map or reach radius differs for {name}")
    require(all(camera[k] == scene["camera"][k] for k in ("width", "height", "fov_deg"))
            and camera["mount"] == manifest["dynamics_params"]["camera_mount"] == room["camera_mount"]
            and manifest["dynamics_params"]["params_sha256"] == room["params_sha"]
            and manifest["renderer"]["near_plane_m"] == scene["render"]["near_plane_m"], f"Teacher camera differs for {name}")
    intrinsics = SCENE.intrinsics(camera["width"], camera["height"], camera["fov_deg"])
    require([camera[k] for k in ("fx", "fy", "cx", "cy")] == [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
            f"Teacher intrinsics differ for {name}")
    actions_path, trace_path = (logs / manifest[k] for k in ("actions_ref", "trace_ref"))
    actions, trace = jsonl(actions_path), jsonl(trace_path)
    require(0 < len(actions) == manifest["num_steps"] == ep["steps"] <= pair["common_budget"] and len(trace) == 5 * len(actions),
            f"Incomplete action or trace count for {name}")
    size = (camera["width"], camera["height"])
    frames, depths = [], []
    for i, action in enumerate(actions):
        require(action["step"] == i and action["action_expert"] == action["action_executed"]
                and action["stop_requested"] == action["stop_expert"] and action["sim_dt"] == .5
                and action["rgb_ts"] == action["pose_ts"] == action["timestamp_ns"], f"Bad action clock or teacher at {name}:{i}")
        window = trace[5 * i:5 * i + 5]
        require(all(t["step"] == i and t["action"] == action["action_executed"]
                    and abs(t["t"] - (action["timestamp_ns"] / 1e9 + .1 * (j + 1))) < 1e-6 for j, t in enumerate(window)),
                f"Incomplete physical trace at {name}:{i}")
        frames.append(decoded_rgb_sha(logs / action["frame"], size))
        depth = np.load(logs / action["depth_file"], allow_pickle=False)
        require(depth.shape == size[::-1] and depth.dtype == np.float32 and np.isfinite(depth).all(),
                f"Invalid sensor depth for {name}:{i}")
        depths.append(hashlib.sha256(depth.tobytes()).hexdigest())
    files = (fixture_path, out / "episodes.jsonl", out / "vln_eval.json", logs / "scene.json",
             logs / "episode_manifest.json", actions_path, trace_path)
    return dict(logs=logs, actions=actions, trace=trace, camera=camera, frames=frames, depths=depths,
                manifest=manifest, waypoint=route["waypoint"], hashes={str(p): sha(p) for p in files},
                task=dict(route=name, instruction=text, goal_object=target, goal_instance=str(design_arm["ins_id"]),
                          start_object=raw["start"], start_instance=str(start["ins_id"]), native_spawn=native["spawn"],
                          source_instruction_index=index))


def shared_steps(arms):
    """分歧步也有同一动作前观测。同动作的最后一窗照样核验已执行的飞行历史。"""
    a, b = arms
    require(a["camera"] == b["camera"], "A/B camera mismatch")
    count = min(len(a["actions"]), len(b["actions"]))
    for step in range(count):
        left, right = a["actions"][step], b["actions"][step]
        fields = ("step", "timestamp_ns", "rgb_ts", "pose_ts", "pose_before", "yaw_before")
        require(all(left[k] == right[k] for k in fields) and a["frames"][step] == b["frames"][step]
                and a["depths"][step] == b["depths"][step], f"Shared pose, RGB or depth differs at step {step}")
        different = (left["action_executed"], left["stop_requested"]) != (right["action_executed"], right["stop_requested"])
        yield step, different
        if different:
            return
        prior = slice(5 * step, 5 * (step + 1))
        require(a["trace"][prior] == b["trace"][prior], f"Shared executed history differs after step {step}")
    require(len(a["actions"]) == len(b["actions"]), "Unequal run length before any difference in executed commands")


def score_step(pair, room, arms, step):
    source = arms[0]
    a = source["actions"][step]
    row = {k: pair[k] for k in ("split", "scene", "pair", "selection_group", "heading_source", "seed", "spawn")}
    row.update(spawn_group=pair["scene"] + ":" + dump(xyz(pair["spawn"])), step=step, primary=False, verified_shared_prefix=True,
               rgb=str(source["logs"] / a["frame"]), rgb_sha256=source["frames"][step], depth_sha256=source["depths"][step],
               action_timestamp_ns=a["timestamp_ns"], pose_nav_m=xyz(a["pose_before"]), yaw_before_deg=a["yaw_before"], camera=source["camera"],
               arm_directories=[str(arm["logs"]) for arm in arms], actions=[arm["actions"][step]["action_executed"] for arm in arms],
               labels=[dict(arm["task"], original_action=arm["actions"][step]["action_executed"],
                            stop_requested=arm["actions"][step]["stop_requested"], logged_bearing_bin=arm["actions"][step]["bearing_bin"],
                            planning_status="not_scored", local_bearing_bin=None, inside_actual_image=None,
                            relative_yaw_ccw_deg=None, image_uv_same_altitude_px=None) for arm in arms])
    require(step > 0, "No measured velocity at step zero")
    previous = source["trace"][5 * step - 1]
    state = pre_action_state(a, previous)
    row.update(pre_action_velocity_xyz_mps=previous["vel"], body_quat_wxyz=previous["quat"], source_trace_time_s=previous["t"])
    actions = [arm["actions"][step] for arm in arms]
    if any(rec["collision"] for rec in actions):
        return dict(row, status="collision")
    if any(not rec["rgb_ok"] or rec["frame_quality"]["low"] for rec in actions):
        return dict(row, status="low_quality_rgb")
    scene = room["scene"]
    camera = source["camera"]
    nav = scene["_nav_to_world"]
    view = SCENE.viewmat_from_quat(xyz(a["pose_before"]), previous["quat"], camera["mount"], nav)
    for label, arm, decision in zip(row["labels"], arms, actions):
        m = arm["manifest"]
        geo = label_goal(room["planner"], state, arm["waypoint"], m["stop_radius"], camera, view, nav,
                         decision["action_expert"], m["teacher_yaw_tol_deg"], near_plane_m=m["renderer"]["near_plane_m"])
        require(geo["teacher_runtime_bin"] == decision["bearing_bin"] and geo["runtime_near"] == decision["near_goal"]
                and bool(geo["runtime_near"]) == decision["stop_requested"], f"Logged vs replayed bearing/STOP differs at step {step}")
        label.update(geo, waypoint=arm["waypoint"])
    if any(r["stop_requested"] for r in actions):
        return dict(row, status="teacher_stop")
    if any(label["planning_status"] != "astar" for label in row["labels"]):
        return dict(row, status="no_local_astar")
    bins = {label["local_bearing_bin"] for label in row["labels"]}
    if len(bins) == 1:
        return dict(row, status="same_bearing")
    if not bins <= {0, 1, 7}:
        return dict(row, status="different_outside_coarse_bins")
    visible = sum(label["inside_actual_image"] for label in row["labels"])
    return dict(row, status=("different_coarse_neither_in_image", "different_coarse_one_in_image", "different_coarse_both_in_image")[visible])


def mark_primary(observations):
    kept = set()
    for row in sorted(observations, key=lambda r: r["step"]):
        site = (row["scene"], xyz(row["spawn"]))
        if row["status"] == "different_coarse_both_in_image" and site not in kept:
            row["primary"] = True
            kept.add(site)


def summarize(pairs, observations):
    first = [{k: r[k] for k in ("scene", "pair", "step", "spawn_group", "selection_group", "heading_source", "rgb_sha256")}
             for r in observations if r["primary"]]
    counts = Counter(r["status"] for r in observations)
    return dict(requested_pairs=len(pairs), requested_arms=2 * len(pairs),
                verified_arms=sum(arm["status"] == "verified" for p in pairs for arm in p["arm_results"]),
                requested_physical_starts=len({(p["scene"], xyz(p["spawn"])) for p in pairs}),
                verified_shared_observations=len(observations), observation_status=dict(counts),
                pair_status=dict(Counter(p["status"] for p in pairs)), primary_distinct_starts=len(first),
                different_coarse_by_visible_points={str(i): counts["different_coarse_" + status] for i, status in
                    enumerate(("neither_in_image", "one_in_image", "both_in_image"))}, first_primary_per_start=first)


def registered_inputs(design_path, design, join_path):
    """协议先于采集冻结；独立核对预筛选中的路线和采集请求，避免删行后仍然计分。"""
    require(design["schema"] in {DESIGN_SCHEMA, PILOT_SCHEMA}, "Unsupported fork design schema")
    pilot = design["schema"] == PILOT_SCHEMA
    protocol_path = (design_path.parent.parent if pilot else design_path.parent) / "protocol.json"
    hashes = {str(p): sha(p) for p in (design_path, join_path, protocol_path)}
    require(hashes[str(protocol_path)] == design["protocol_sha256"] == PROTOCOL_SHA, "Frozen protocol SHA-256 differs")
    protocol = read(protocol_path)
    require(hashes[str(join_path)] == protocol["instance_join_sha256"], "Frozen instance join differs")
    pairs = design["pairs"]
    if pilot:
        require(len(pairs) == 1 and pairs[0]["pair"] == "839992_tr_118_tr_136" and pairs[0]["scene"] == "839992"
                and pairs[0]["seed"] == 3102 and pairs[0]["common_budget"] == 28 and pairs[0]["selection_group"] == "historical_regression",
                "Historical regression must contain only the 28-step registered pilot")
        return hashes
    rooms = {s: split for split, names in ROOMS.items() for s in names}
    require(design["join_sha256"] == hashes[str(join_path)] and design["split_roles"] == rooms
            and set(design["rooms"]) == set(rooms), "Formal design must report all five pre-registered rooms")
    require({s: set(protocol["roles"][s]) for s in ROOMS} == ROOMS and protocol["acquisition"]["steps"] == 8
            and design["scoring_steps"] == list(range(1, 8))
            and all(p["common_budget"] == 8 and rooms.get(p["scene"]) == p["split"] for p in pairs),
            "Formal room split or eight-step flight budget differs")
    for scene, report in design["rooms"].items():
        counts, sites = report["counts"], report["sites"]
        require(report["scene"] == scene and len(sites) == counts["eligible_sites"]
                and report["assets_sha256"][report["source_scene_path"]] == protocol["scene_sha256"][scene],
                f"Missing or incorrect room survey: {scene}")
        requested = [p for p in pairs if p["scene"] == scene]
        targets = [p["pair"] for p in requested if p["selection_group"] == "targeted"]
        selected = [s["pair"] for s in sites if s["status"] == "selected_targeted"]
        controls = [p for p in requested if p["selection_group"] == "control"]
        require(sorted(targets) == sorted(selected) and len(targets) == counts.get("selected_targeted", 0)
                and len(targets) <= protocol["design"]["cap"][scene] and len(controls) == counts["selected_control"] <= 1
                and len(requested) == len(targets) + len(controls), f"Requested pairs differ from the {scene} survey")
    return hashes


def verify(design_path, out, data_root=DEFAULT_ROOT):
    design_path, out, data_root = Path(design_path).resolve(), Path(out).resolve(), Path(data_root)
    if out.exists():
        raise FileExistsError(out)
    design = read(design_path)
    join_path = data_root / "exp-vln-stage0-pairs-20261123/interiorgs_instance_join.json"
    hashes = registered_inputs(design_path, design, join_path)
    pilot = design["schema"] == PILOT_SCHEMA
    joins = {(r["route"], r["field"]): r for r in read(join_path)["rows"]}
    pairs, rows, rooms = [], [], {}
    seen = set()
    sites = []
    groups = {"historical_regression"} if pilot else {"targeted", "control"}
    for pair in design["pairs"]:
        require(pair["pair"] not in seen and len(pair["arms"]) == 2 and pair["scene"] in ROOMS[pair["split"]]
                and pair["heading_source"] in {"designed_common", "native_common"} and pair["selection_group"] in groups,
                f"Invalid or duplicate registered pair {pair['pair']}")
        seen.add(pair["pair"])
        point = xyz(pair["spawn"])
        require(all(s != pair["scene"] or math.dist(point, pos) >= 1 for s, pos in sites),
                f"Registered physical sites are less than 1 m apart in room {pair['scene']}")
        sites.append((pair["scene"], point))
        result = {k: pair[k] for k in ("scene", "split", "pair", "seed", "spawn", "common_budget", "selection_group", "heading_source")}
        result["arm_results"] = []
        pairs.append(result)
        arms = []
        for arm in pair["arms"]:
            report = {"route": arm["route"], "out": arm["out"]}
            result["arm_results"].append(report)
            try:
                if pair["scene"] not in rooms:
                    room = load_room(data_root, pair["scene"], joins)
                    if not pilot:
                        registered = design["rooms"][pair["scene"]]["assets_sha256"]
                        require(all(registered.get(path) == digest for path, digest in room["hashes"].items()),
                                f"Native room assets differ from frozen design in {pair['scene']}")
                    rooms[pair["scene"]] = room
                    hashes.update(room["hashes"])
                data = load_arm(pair, arm, rooms[pair["scene"]], design_path.parent)
                hashes.update(data["hashes"])
                arms.append(data)
                report.update(status="verified", steps=len(data["actions"]), terminal_reason=data["manifest"]["terminal_reason"])
            except SOURCE_ERRORS as err:
                report.update(status="missing_or_unverified", reason=str(err))
        if len(arms) != 2:
            result.update(status="missing_or_unverified_arm")
            continue
        try:
            tasks = [arm["task"] for arm in arms]
            require(tasks[0]["start_instance"] == tasks[1]["start_instance"] and tasks[0]["start_object"] == tasks[1]["start_object"]
                    and tasks[0]["goal_instance"] != tasks[1]["goal_instance"] and tasks[0]["instruction"] != tasks[1]["instruction"]
                    and math.dist(*(xyz(arm["waypoint"])[:2] for arm in arms)) > 2 * arms[0]["manifest"]["stop_radius"],
                    f"Pair does not have a shared native start and two disjoint goal instances: {pair['pair']}")
            common_native = all(abs(math.remainder(task["native_spawn"]["yaw_deg"] - pair["spawn"]["yaw_deg"], 360)) <= 1e-5
                                for task in tasks)
            require((pair["heading_source"] == "native_common") == common_native, f"Wrong heading source for {pair['pair']}")
            fork = None
            local = []
            for step, differs in shared_steps(arms):
                if step:
                    local.append(score_step(pair, rooms[pair["scene"]], arms, step))
                if differs:
                    fork = step
            status = "diverged" if fork is not None else "same_until_terminal"
            if fork is None and len(local) + 1 == pair["common_budget"]:
                status = "same_through_budget"
            result.update(status=status, verified_reset=True, first_unequal_command_step=fork, shared_observations=len(local))
            rows.extend(local)
        except SOURCE_ERRORS as err:
            result.update(status="unverified_shared_history_or_replay", reason=str(err))
    mark_primary(rows)
    summary = dict(schema="uavrgb.vln.shared_forks.v1", design=str(design_path), source_sha256=hashes, pairs=pairs,
                   primary_rule="同一房间独立 XYZ 至少间隔 1 米，每个位置取第一个两条不同 0/1/7 方向都入画的已执行共享观测；步号相同按设计顺序。",
                   all=summarize(pairs, rows))
    for field in ("split", "scene", "selection_group", "heading_source"):
        summary["by_" + field] = {v: summarize([p for p in pairs if p[field] == v], [r for r in rows if r[field] == v])
                                  for v in sorted({p[field] for p in pairs})}
    if not pilot:
        for scene in design["rooms"]:
            summary["by_scene"].setdefault(scene, summarize([], []))
    for split in ROOMS:
        group = [p for p in pairs if p["split"] == split]
        summary["by_split"].setdefault(split, summarize(group, []))
        summary["by_split"][split]["by_selection_group"] = {
            v: summarize([p for p in group if p["selection_group"] == v], [r for r in rows if r["split"] == split and r["selection_group"] == v])
            for v in ("targeted", "control")}
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "verified"
        stage.mkdir()
        table = stage / "observations.jsonl"
        with table.open("w", encoding="utf-8") as file:
            for row in rows:
                print(dump(row), file=file)
        summary["observations_sha256"] = sha(table)
        (stage / "summary.json").write_text(dump(summary) + "\n", encoding="utf-8")
        stage.rename(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(dump(verify(args.design, args.out, args.data_root)["all"]))


if __name__ == "__main__":
    main()
