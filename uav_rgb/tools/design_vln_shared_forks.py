"""为同起点的两条 SAGE-3D 原生指令预选短飞分岔。筛选只读几何与固件遥测。"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from tools.vln_fork_geometry import _project


scene_io = importlib.import_module("sim.3dgs.scene")
collision = importlib.import_module("sim.3dgs.collision")
planner_mod = importlib.import_module("sim.3dgs.planner")
firmware = importlib.import_module("sim.3dgs.dynamics")
vln = importlib.import_module("sim.3dgs.vln")
reach_radius = importlib.import_module("sim.3dgs.eval_3dgs").reach_radius_of
ROOMS = {"839992": ("development", 8), "840865": ("development", 8), "840139": ("train", 2),
         "841012": ("train", 2), "841150": ("train", 2)}
BUDGET = 8
XYZ_TOL = 1e-5
HOLD = 0.5
REPO = Path(__file__).resolve().parents[1]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)


def xyz(point):
    return tuple(point[k] for k in "xyz")


def wrap(angle):
    return (angle + 180) % 360 - 180


def site_id(pos):
    return ",".join(f"{value:.5f}" for value in pos)


def site_seed(room, pos):
    return int(digest(f"vln-shared-v1|{room}|{site_id(pos)}")[:8], 16) % 1_000_000


def read_catalog(root, room, joins):
    folder = root / "3dgs-scenes" / ("sage3d_" + room)
    path = folder / "scene.json"
    source = json.loads(path.read_bytes())
    scene = scene_io.load_scene(str(path))
    sage = root / "pubdata/sage3d"
    native_path, = (sage / "vln" / room / "train").glob(f"train_trajectories_overall_*_{room}.json")
    native_name = native_path.stem.removeprefix("train_trajectories_overall_")
    labels_path = sage / "interiorgs_labels" / native_name / "labels.json"
    instance_labels = {str(r["ins_id"]): r["label"] for r in json.loads(labels_path.read_bytes())}
    native, = json.loads(native_path.read_bytes())["scenes"]
    if native["scene_id"] != room or scene["collision"]["source"] != "mesh":
        raise ValueError(f"Wrong native room or missing collision mesh: {room}")
    samples = {str(s["trajectory_id"]): s for s in native["samples"]}
    counts, routes = Counter(), []
    for r in source["routes"]:
        if r["sage"]["split"] != "train":
            continue
        counts["official_train_routes"] += 1
        key = (source["name"] + "/" + r["name"])
        start, end = joins.get((key, "start")), joins.get((key, "end"))
        if any(not v or str(v["ins_id"]) not in instance_labels for v in (start, end)):
            counts["missing_instance_join"] += 1
            continue
        sample = samples[str(r["sage"]["trajectory_id"])]
        for point, join, point_index in ((r["spawn"], start, 0), (r["waypoint"], end, -1)):
            if (math.dist(xyz(point)[:2], sample["points"][point_index]["position"][:2]) > XYZ_TOL
                    or math.dist(xyz(point)[:2], join["position_nav_xy"]) > XYZ_TOL):
                raise ValueError(f"Joined endpoint differs from raw native trajectory: {key}")
        valid = None
        for i, ins in enumerate(r["sage"]["instructions"]):
            if (ins["instruction_type"] != "Relative_Relationship" or ins["start"] != start["target"]
                    or ins["end"] != end["target"] or not ins["text"].strip()):
                continue
            raw = sample["instructions"][i]
            if (raw["scene_id"] != native_name or raw["generated_instruction"] != ins["text"]
                    or any(raw[k] != ins[k] for k in ("start", "end", "instruction_type"))):
                raise ValueError(f"Instruction differs from raw native trajectory: {key}/{i}")
            valid = dict(route=r["name"], native_route=r, text=ins["text"], instruction=ins,
                         source_instruction_index=i, start_id=str(start["ins_id"]), end_id=str(end["ins_id"]))
            break
        if valid is None:
            counts["no_matching_relative_instruction"] += 1
        else:
            routes.append(valid)
    counts["eligible_routes"] = len(routes)
    assets = (path, native_path, labels_path, Path(scene["ply_path"]), folder / scene["collision"]["geometry"])
    return scene, source, routes, counts, {str(p): file_sha(p) for p in assets}


def candidate_sites(routes, radius, counts):
    groups = {}
    for route in sorted(routes, key=lambda r: (xyz(r["native_route"]["spawn"]), r["route"])):
        pos = xyz(route["native_route"]["spawn"])
        key = next((p for p in groups if math.dist(p, pos) <= XYZ_TOL), pos)
        groups.setdefault(key, []).append(route)
    sites = {}
    for pos, members in groups.items():
        choices = []
        for a, b in itertools.combinations(sorted(members, key=lambda r: r["route"]), 2):
            counts["same_xyz_route_pairs"] += 1
            if math.dist(xyz(a["native_route"]["spawn"]), xyz(b["native_route"]["spawn"])) > XYZ_TOL:
                continue
            if a["start_id"] != b["start_id"] or a["instruction"]["start"] != b["instruction"]["start"]:
                counts["different_start"] += 1
                continue
            if a["end_id"] == b["end_id"] or a["text"] == b["text"]:
                counts["same_goal_or_sentence"] += 1
                continue
            if math.dist(xyz(a["native_route"]["waypoint"])[:2], xyz(b["native_route"]["waypoint"])[:2]) <= 2 * radius:
                counts["overlapping_goal_circles"] += 1
                continue
            choices.append((a, b))
        if choices:
            sites[pos] = choices
    counts["eligible_sites"], counts["eligible_pairs"] = len(sites), sum(map(len, sites.values()))
    return sites


def native_yaws(pair):
    return [arm["native_route"]["spawn"]["yaw_deg"] for arm in pair]


def candidate_headings(pair, headings):
    midpoint = wrap(headings[0] + wrap(headings[1] - headings[0]) / 2)
    angles = native_yaws(pair) + [midpoint + delta for delta in (0, -8, 8, -14, 14, -23, 23)]
    return list(dict.fromkeys(round(wrap(heading), 6) for heading in angles))


def choose_control(room, sites):
    options = []
    for pos, pairs in sites.items():
        for pair in pairs:
            a, b = native_yaws(pair)
            if abs(wrap(a - b)) <= 1e-5:
                key = digest(f"control|{room}|{site_id(pos)}|{pair[0]['route']}|{pair[1]['route']}")
                options.append((key, pos, pair))
    return min(options, key=lambda c: c[0])[1:] if options else None


def aim_point(planner, path, state):
    contract = vln.SUBGOAL_BEARING_CONTRACT
    look = contract["lookahead_base_m"] + contract["lookahead_speed_gain_s"] * math.hypot(state.vx, state.vy)
    return planner.aim_point(path, state.x, state.y, lookahead=look, los_max=contract["los_max_m"],
                             hard_los_max=min(look, contract["hard_los_max_m"]))


def local_view(planner, teachers, drone, camera, scene):
    view = drone.camera_viewmat(scene["_nav_to_world"])
    bearings, angles, pixels = [], [], []
    for teacher in teachers:
        aim = aim_point(planner, teacher.last_path, drone)
        if math.dist((drone.x, drone.y), aim) <= 1e-8:
            return None, "degenerate_aim"
        angle = wrap(math.degrees(math.atan2(aim[1] - drone.y, aim[0] - drone.x) - drone.yaw))
        b = vln.bearing_bin(drone.yaw, (drone.x, drone.y), aim)
        proj = _project(aim, drone.z, camera, view, scene["_nav_to_world"], scene["render"]["near_plane_m"])
        if b != teacher.last_bearing_bin:
            raise ValueError(f"Teacher bin {teacher.last_bearing_bin} and independently replayed bearing {b} differ")
        bearings.append(b)
        angles.append(angle)
        pixels.append(proj["image_uv_same_altitude_px"] if proj["inside_actual_image"] else None)
    if bearings[0] == bearings[1] or not all(b in (0, 1, 7) for b in bearings):
        return None, "same_or_unrepresentable_bin"
    if None in pixels:
        return None, "outside_image"
    margin = min(min(u / camera["width"], 1 - u / camera["width"], v / camera["height"], 1 - v / camera["height"])
                 for u, v in pixels)
    clearance = min(abs((a % 45) - 22.5) for a in angles)
    return dict(bins=bearings, relative_yaw_ccw_deg=angles, image_uv=pixels, min_image_margin=margin,
                min_bin_edge_deg=clearance), "both_inside_different"


def simulate(planner, scene, params, camera, pos, yaw, pair, seed):
    teachers = [planner_mod.OccAstarTeacher(planner, a["native_route"]["waypoint"], stop_radius=reach_radius(scene),
                                            hold_s=HOLD, clearance="braking") for a in pair]
    drone = firmware.CfSitlDynamics(dict(scene, spawn=dict(zip("xyz", pos))), params, seed=seed)
    reasons = Counter()
    prefix, witnesses = [], []
    try:
        drone.reset(*pos[:2], yaw, episode=0)
        for step in range(BUDGET):
            actions = [t(None, None, drone)[0] for t in teachers]
            if any(t.stop_requested or t.last_path is None for t in teachers):
                reasons["near_or_no_path"] += 1
                break
            if step and all(a in (1, 2, 3) for a in actions):
                metrics, reason = local_view(planner, teachers, drone, camera, scene)
                reasons[reason] += 1
                if metrics:
                    witnesses.append(dict(metrics, step=step, common_prefix_actions=list(prefix), actions=actions,
                                          pose_nav_m=[drone.x, drone.y, drone.z], body_quat_wxyz=list(drone.quat),
                                          velocity_nav_mps=[drone.vx, drone.vy, drone.vz]))
            if actions[0] != actions[1]:
                reasons["first_unequal_action"] += 1
                break
            if step + 1 == BUDGET:
                reasons["budget"] += 1
                break
            before = (drone.x, drone.y, drone.z)
            state = drone.step(actions[0], HOLD)
            if planner.occ.sweep(before, state["pose"])[0]:
                reasons["collision"] += 1
                break
            prefix.append(actions[0])
    finally:
        drone.close()
    return witnesses, reasons


def cached_planner(occ):
    planner = planner_mod.OccPlanner(occ)
    plan = planner.plan

    # OccPlanner 的 A* 只依赖起终格。瞄准点和净空仍要逐次使用动作前的真实位置。
    @lru_cache(maxsize=4096)
    def plan_cells(start_cell, goal_cell, pad):
        return plan(planner.xy(*start_cell), planner.xy(*goal_cell), pad)

    planner.plan = lambda start, goal, pad_m=6.0: plan_cells(planner.cell(*start), planner.cell(*goal), pad_m)
    return planner


def shortlist(pos, pairs, planner):
    at_spawn = SimpleNamespace(x=pos[0], y=pos[1], vx=0.0, vy=0.0)
    headings = {}
    for arm in itertools.chain.from_iterable(pairs):
        name = arm["route"]
        if name in headings:
            continue
        goal = xyz(arm["native_route"]["waypoint"])[:2]
        path = planner.plan(pos[:2], goal)
        headings[name] = None
        if path is not None:
            aim = aim_point(planner, path, at_spawn)
            if math.dist(aim, pos[:2]) > 1e-8:
                headings[name] = math.degrees(math.atan2(aim[1] - pos[1], aim[0] - pos[0]))
    candidates = []
    for a, b in pairs:
        h = (headings[a["route"]], headings[b["route"]])
        if None in h:
            continue
        gap = abs(wrap(h[1] - h[0]))
        candidates.append(((abs(gap - 40), abs(gap - 27), a["route"], b["route"]), (a, b), h))
    candidates.sort(key=lambda c: c[0])
    return candidates[:3], len(candidates)


def pair_spec(room, group, pos, pair, yaw, witness):
    native = native_yaws(pair)
    source = "native_common" if all(abs(wrap(yaw - angle)) <= 1e-5 for angle in native) else "designed_common"
    pair_name = f"{room}_{pair[0]['route']}_{pair[1]['route']}_{group}"
    return dict(pair=pair_name, scene=room, split=ROOMS[room][0], selection_group=group, heading_source=source,
                spawn=dict(zip(("x", "y", "z", "yaw_deg"), (*pos, yaw))), seed=site_seed(room, pos),
                common_budget=BUDGET, native_yaws_deg=native, cpu_witness=witness,
                waypoint_sep_m=math.dist(*(xyz(a["native_route"]["waypoint"])[:2] for a in pair)),
                arms=[dict(route=a["route"], source_instruction_index=a["source_instruction_index"],
                           instruction=a["text"], start_object=a["instruction"]["start"], start_instance=a["start_id"],
                           target=a["instruction"]["end"], ins_id=a["end_id"], native_spawn=a["native_route"]["spawn"],
                           waypoint=a["native_route"]["waypoint"], source_route_sha256=digest(dump(a["native_route"])))
                      for a in pair])


def survey_room(task):
    root, room, joins = task
    scene, source, routes, counts, assets = read_catalog(root, room, joins)
    sites = candidate_sites(routes, reach_radius(scene), counts)
    control = choose_control(room, sites)
    selected = []
    used = []
    if control:
        pos, pair = control
        selected.append(pair_spec(room, "control", pos, pair, native_yaws(pair)[0], None))
        used.append(pos)

    params = firmware.CfSitlParams.from_json(str(REPO / "sim/dynamics/e1_mini.json"))
    camera = dict(scene["camera"], mount=params.camera_mount)
    intr = scene_io.intrinsics(camera["width"], camera["height"], camera["fov_deg"])
    camera.update(fx=intr[0, 0], fy=intr[1, 1], cx=intr[0, 2], cy=intr[1, 2])
    planner = cached_planner(collision.from_scene(scene))
    locations = []
    for pos, pairs in sorted(sites.items()):
        item = dict(position_xyz=list(pos), eligible_pairs=len(pairs), attempts=[])
        locations.append(item)
        if len(selected) - int(control is not None) >= ROOMS[room][1]:
            item["status"] = "room_quota_reached"
            continue
        if any(math.dist(pos, other) < 1 for other in used):
            item["status"] = "reserved_control_or_within_1m"
            continue
        shortlist_pairs, item["pairs_with_astar_aims"] = shortlist(pos, pairs, planner)
        best, spec = None, None
        for rank, pair, headings in shortlist_pairs:
            for yaw in candidate_headings(pair, headings):
                witnesses, reasons = simulate(planner, scene, params, camera, pos, yaw, pair, site_seed(room, pos))
                counts["cpu_yaw_trials"] += 1
                item["attempts"].append(dict(routes=[a["route"] for a in pair], yaw_deg=yaw,
                                             nominal_aim_headings_deg=headings, qualifying_steps=[v["step"] for v in witnesses],
                                             outcomes=dict(reasons)))
                for hit in witnesses:
                    design = pair_spec(room, "targeted", pos, pair, yaw, hit)
                    score = (design["heading_source"] != "native_common", -hit["min_image_margin"],
                             -hit["min_bin_edge_deg"], hit["step"], rank[2:], yaw)
                    if best is None or score < best:
                        best, spec = score, design
        if spec is None:
            item["status"] = "no_qualifying_shared_window"
            continue
        selected.append(spec)
        used.append(pos)
        item.update(status="selected_targeted", pair=spec["pair"])
    counts.update(item["status"] for item in locations)
    counts["selected_control"] = int(control is not None)
    return dict(scene=room, counts=dict(counts), sites=locations, pairs=selected, assets_sha256=assets,
                source_scene_path=str(root / "3dgs-scenes" / source["name"] / "scene.json"),
                vehicle_profile=params.profile())


def write_fixtures(stage, out, root, report):
    source_path = Path(report["source_scene_path"])
    source = json.loads(source_path.read_bytes())
    routes = {r["name"]: r for r in source["routes"]}
    for pair in report["pairs"]:
        for letter, arm in zip("AB", pair["arms"]):
            original = routes[arm["route"]]
            if digest(dump(original)) != arm["source_route_sha256"]:
                raise ValueError(f"原路线在筛选后变动: {arm['route']}")
            ins = original["sage"]["instructions"][arm["source_instruction_index"]]
            route = dict(original, max_steps=BUDGET, spawn=pair["spawn"], sage=dict(original["sage"], instructions=[ins]))
            fixture = dict(source, routes=[route], multi_stage_routes=[], spawn=pair["spawn"], waypoint=route["waypoint"],
                           max_steps=BUDGET, ply=str((source_path.parent / source["ply"]).resolve()),
                           collision=dict(source["collision"], geometry=str((source_path.parent / source["collision"]["geometry"]).resolve())))
            relative = Path("fixtures") / pair["pair"] / (letter + ".json")
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dump(fixture) + "\n", encoding="utf-8")
            saved = out / relative
            dest = out / "episodes" / pair["split"] / pair["pair"] / letter
            args = [str(root / "3dgs-venv/bin/python"), "-m", "sim.3dgs.run_vln", "--scene", str(saved), "--out", str(dest),
                    "--policy", "teacher", "--grounder", "none", "--teacher-clearance", "braking", "--hold", str(HOLD),
                    "--seed", str(pair["seed"]), "--split", "train", "--all-instructions", "--limit", "1", "--videos", "0",
                    "--counterfactual", "correct", "--stop-mode", "explicit", "--save-depth"]
            arm.update(fixture=str(saved), fixture_sha256=file_sha(path), out=str(dest), argv=args,
                       source_scene_sha256=report["assets_sha256"][str(source_path)])


def prior_usage(root):
    catalog, hashes = defaultdict(list), {}
    for name in ("exp-vln-language-forks-20261124", "exp-vln-language-forks-expanded-20261124"):
        path = root / name / "registration.json"
        hashes[str(path)] = file_sha(path)
        for pair in json.loads(path.read_bytes())["pairs"]:
            for arm in pair["arms"]:
                catalog[pair["scene"]].append((arm["route"], arm["instruction"]))
    return catalog, hashes


def build(root, out, workers):
    if workers < 1:
        raise ValueError("--workers must be positive")
    protocol = out / "protocol.json"
    protocol_sha = file_sha(protocol)
    rules = json.loads(protocol.read_bytes())
    cap = {room: spec[1] for room, spec in ROOMS.items()}
    if rules["design"]["cap"] != cap or any(room not in rules["roles"][split] for room, (split, _) in ROOMS.items()):
        raise ValueError("房间分配或采样预算与预登记不符")
    for reserved in ("fixtures", "episodes", "design.json"):
        if (out / reserved).exists() or (out / reserved).is_symlink():
            raise FileExistsError(f"输出已存在: {out / reserved}")
    join_path = root / "exp-vln-stage0-pairs-20261123/interiorgs_instance_join.json"
    if file_sha(join_path) != rules["instance_join_sha256"] or file_sha(firmware.CF_SITL_PROC) != rules["acquisition"]["cpu_binary_sha256"]:
        raise ValueError("实例关联表或固件可执行文件与预登记不符")
    for room in ROOMS:
        path = root / "3dgs-scenes" / ("sage3d_" + room) / "scene.json"
        if file_sha(path) != rules["scene_sha256"][room]:
            raise ValueError(f"原场景与预登记不符: {room}")
    prior, prior_hashes = prior_usage(root)
    joins = defaultdict(dict)
    for row in json.loads(join_path.read_bytes())["rows"]:
        if row["scene"] in ROOMS:
            key = (row["route"], row["field"])
            if key in joins[row["scene"]]:
                raise ValueError(f"实例关联表出现重复键: {key}")
            joins[row["scene"]][key] = row
    tasks = [(root, room, joins[room]) for room in ROOMS]
    code = [Path(__file__), REPO / "tools/vln_fork_geometry.py", REPO / "sim/dynamics/e1_mini.json", Path(firmware.CF_SITL_PROC)]
    code += [REPO / f"sim/3dgs/{name}.py" for name in ("run_vln", "eval_3dgs", "collect_3dgs", "collision", "dynamics", "planner", "scene", "vln", "renderer")]
    doc = dict(schema="uavrgb.vln.shared_fork_design.v1", protocol_sha256=protocol_sha, rooms={}, pairs=[],
               decision="cpu_geometry_prediction_requires_independent_rgb_flights", scoring_steps=list(range(1, BUDGET)),
               split_roles={room: ROOMS[room][0] for room in ROOMS}, join_sha256=file_sha(join_path),
               prior_registration_sha256=prior_hashes, dependency_sha256={str(p): file_sha(p) for p in code})
    with TemporaryDirectory(dir=out) as tmp, ProcessPoolExecutor(max_workers=min(workers, len(ROOMS))) as pool:
        stage = Path(tmp)
        (stage / "fixtures").mkdir()
        for report in pool.map(survey_room, tasks):
            room = report["scene"]
            source = json.loads(Path(report["source_scene_path"]).read_bytes())
            old = prior[room]
            old_routes, old_texts = {r for r, _ in old}, {text for _, text in old}
            old_pos = [xyz(r["spawn"]) for r in source["routes"] if r["name"] in old_routes]
            for pair in report["pairs"]:
                pos = xyz(pair["spawn"])
                pair["used_prior_spawn_xyz"] = any(math.dist(pos, p) <= XYZ_TOL for p in old_pos)
                pair["prior_spawn_within_1m"] = any(math.dist(pos, p) < 1 for p in old_pos)
                for arm in pair["arms"]:
                    arm["used_prior_route"] = arm["route"] in old_routes
                    arm["used_prior_instruction"] = arm["instruction"] in old_texts
            write_fixtures(stage, out, root, report)
            doc["pairs"].extend(report.pop("pairs"))
            doc["rooms"][report["scene"]] = report
            print(report["scene"], dump(report["counts"]), flush=True)
        if file_sha(protocol) != protocol_sha:
            raise ValueError("筛选过程中预登记发生变化")
        (stage / "design.json").write_text(dump(doc) + "\n", encoding="utf-8")
        (stage / "fixtures").rename(out / "fixtures")
        os.link(stage / "design.json", out / "design.json")
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/rivermind-data/uav_rgb"))
    parser.add_argument("--out", type=Path, required=True, help="已含 protocol.json 的新实验根目录")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    build(args.data_root.resolve(), args.out.resolve(), args.workers)


if __name__ == "__main__":
    main()
