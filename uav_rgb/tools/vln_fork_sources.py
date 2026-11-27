"""从冻结的 SAGE-3D 同图双指令清单还原原生任务、教师轨迹和动作前状态。"""
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


scene_io = importlib.import_module("sim.3dgs.scene")
reach_radius_of = importlib.import_module("sim.3dgs.eval_3dgs").reach_radius_of
DEFAULT_ROOT = Path("/root/rivermind-data/uav_rgb")
ARCHIVE = "exp-vln-fork-counterfactual-states-20261124"
REGISTRATION_SHA = "5c20406d2b65502e214b69f29250e888dfcfa8e8a05b51901cbf43839b610226"
INDEX_SHA = "5fd400c5817df554a7f15fca024d53091ac96a9eb80713b8ec3334726980d9e4"
ROOM_SPLIT = {"840139": "train", "841012": "train", "841150": "train", "839992": "development"}
POOLS = ("train", "development", "equal_train", "equal_development")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(records, key):
    result = {record[key]: record for record in records}
    require(len(result) == len(records), f"Duplicate {key} in source records")
    return result


def pre_action_state(action, previous):
    """第 k 次决策使用上一个控制窗口末尾实测的位姿和速度。"""
    step, stamp = action["step"], action["timestamp_ns"]
    require(step > 0 and previous["step"] == step - 1, f"Missing previous trace at step {step}")
    require(action["pose_ts"] == action["rgb_ts"] == stamp and abs(previous["t"] - stamp / 1e9) < 1e-6,
            f"RGB, decision and trace clocks differ at step {step}")
    pos = [action["pose_before"][k] for k in "xyz"]
    velocity, quat = previous["vel"], previous["quat"]
    require(len(velocity) == 3 and len(quat) == 4 and all(map(math.isfinite, velocity + quat))
            and math.dist(pos, previous["pos"]) <= 1e-6, f"Invalid pre-action pose or velocity at step {step}")
    w, x, y, z = quat
    yaw = math.radians(action["yaw_before"])
    trace_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    require(abs(math.hypot(*quat) - 1) <= 1e-5 and abs(math.remainder(trace_yaw - yaw, 2 * math.pi)) <= 1e-5,
            f"Body quaternion differs from the action heading at step {step}")
    return SimpleNamespace(**action["pose_before"], yaw=yaw, vx=velocity[0], vy=velocity[1])


def decoded_rgb_sha(path, size):
    with Image.open(path) as image:
        require(image.format == "PNG" and image.size == size, f"RGB format or dimensions differ: {path}")
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def shared_fork(arms, mask, rgb):
    """相机先观测再执行动作，所以第一次执行不同动作的那一帧仍然共享。"""
    a, b = arms
    key = mask["pair"]
    fork = next((i for i, (left, right) in enumerate(zip(a["actions"], b["actions"]))
                 if left["action_executed"] != right["action_executed"]), None)
    require(mask["status"] == "strict_rgb_fork" and mask["rgb_equal_through_divergence"]
            and mask["physics_equal_through_divergence"] and fork is not None
            and fork == mask["first_unequal_executed_action_step"], f"Wrong recorded fork for {key}")
    require(a["trace"][:5 * fork] == b["trace"][:5 * fork], f"Shared flight history differs for {key}")
    for step in range(fork + 1):
        left, right = a["actions"][step], b["actions"][step]
        fields = ("step", "timestamp_ns", "rgb_ts", "pose_ts", "pose_before", "yaw_before")
        require(all(left[k] == right[k] for k in fields), f"Shared pose or clock differs at {key} step {step}")
        require(rgb(a["logs"] / left["frame"], a["camera"]) == rgb(b["logs"] / right["frame"], b["camera"]),
                f"Shared RGB differs at {key} step {step}")
    return fork


class ForkSources:
    """每行对应教师在仿真中实际看到的一张 RGB 和两条指令。hashes 保存读取的来源文件及指纹。"""

    def __init__(self, data_root=DEFAULT_ROOT):
        self.root = Path(data_root).resolve()
        self.archive = self.root / ARCHIVE
        self.hashes = {}
        self.pairs = {}
        self._masks = {}
        self._rooms = {}
        self._episodes = {}
        self._pixels = {}
        archive = self._json(self.archive / "registration.json", REGISTRATION_SHA)
        self.index = self._json(self.archive / "index.json", INDEX_SHA)
        self._vehicle_path = Path(__file__).resolve().parents[1] / "sim/dynamics/e1_mini.json"
        self._vehicle = self._json(self._vehicle_path)
        catalog_path = self.root / "exp-vln-stage0-pairs-20261123/interiorgs_instance_join.json"
        self._joins = {}
        for entry in self._json(catalog_path)["rows"]:
            key = (entry["route"], entry["field"])
            require(key not in self._joins, f"Ambiguous instance join {key}")
            self._joins[key] = entry
        for filename, digest in archive["sources"].items():
            path = self._path(filename)
            registry = self._json(path, digest)
            self._sha(catalog_path, registry["instance_join_sha256"])
            masks = self._json(path.with_name("fork_mask.json"))
            require(masks["registration_sha256"] == digest, f"Wrong fork mask for {path}")
            mask_by_pair = indexed(masks["pairs"], "pair")
            for pair in registry["pairs"]:
                key = pair["pair"]
                require(key not in self.pairs and ROOM_SPLIT.get(pair["scene"]) == pair["split"],
                        f"Duplicate or out-of-split pair {key}")
                self.pairs[key] = dict(pair, registration=str(path))
                self._masks[key] = mask_by_pair[key]
        self._tables = {}
        seen = set()
        for pool in POOLS:
            table = self._jsonl(self.archive / (pool + ".jsonl"), self.index[pool]["sha256"])
            digests = {row["rgb_sha256"] for row in table}
            require(len(table) == len(digests) == self.index[pool]["unique_rgb"] == self.index[pool]["n_rows"]
                    and seen.isdisjoint(digests), f"Repeated or missing RGB in {pool}")
            seen.update(digests)
            for row in table:
                pair = self.pairs[row["pair"]]
                require(row["scene"] == pair["scene"] and row["split"] == pair["split"] == pool.removeprefix("equal_")
                        and (row["actions"][0] == row["actions"][1]) == pool.startswith("equal_"),
                        f"Wrong room, split or action pool in {pool}: {row['pair']}")
            self._tables[pool] = table

    def rows(self):
        """按冻结清单的原始顺序遍历，source_line 从 1 开始。"""
        for pool, table in self._tables.items():
            for line, row in enumerate(table, 1):
                yield dict(row, source_pool=pool, source_line=line)

    def _path(self, filename):
        # 签名清单存的是旧数据盘的绝对路径。副本保留相对目录结构即可读取。
        return self.root / Path(filename).relative_to(DEFAULT_ROOT)

    def _sha(self, path, expected=None):
        key = str(path)
        if key not in self.hashes:
            with Path(path).open("rb") as stream:
                self.hashes[key] = hashlib.file_digest(stream, "sha256").hexdigest()
        digest = self.hashes[key]
        require(expected is None or digest == expected, f"SHA-256 mismatch: {key}")
        return digest

    def _json(self, path, expected=None):
        self._sha(path, expected)
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def _jsonl(self, path, expected=None):
        self._sha(path, expected)
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]

    def _rgb(self, path, camera):
        if path not in self._pixels:
            self._sha(path)
            self._pixels[path] = decoded_rgb_sha(path, (camera["width"], camera["height"]))
        return self._pixels[path]

    def _room(self, name):
        if name in self._rooms:
            return self._rooms[name]
        path = self.root / "3dgs-scenes" / ("sage3d_" + name) / "scene.json"
        scene = scene_io.load_scene(str(path))
        self._sha(path)
        self._sha(scene["ply_path"])
        mesh = path.parent / scene["collision"]["geometry"]
        self._sha(mesh)
        native = self.root / "pubdata/sage3d"
        files = list((native / "vln" / name / "train").glob("train_trajectories_overall_*_" + name + ".json"))
        require(len(files) == 1, f"Missing or ambiguous native trajectories in room {name}")
        trajectories, = self._json(files[0])["scenes"]
        require(trajectories["scene_id"] == name, f"Wrong native trajectory room at {files[0]}")
        native_name = files[0].stem.removeprefix("train_trajectories_overall_")
        labels_path = native / "interiorgs_labels" / native_name / "labels.json"
        room = dict(scene=scene, routes=indexed(scene["routes"], "name"), mesh=mesh, native_name=native_name,
                    samples=indexed(trajectories["samples"], "trajectory_id"), native_path=str(files[0]),
                    labels=indexed(self._json(labels_path), "ins_id"), labels_path=str(labels_path),
                    radius=reach_radius_of(scene))
        self._rooms[name] = room
        return room

    def _instruction(self, room, pair, arm, fixture):
        scene = room["scene"]
        route, = fixture["routes"]
        name = arm["route"]
        original = room["routes"][name]
        require(route["name"] == name and route["sage"]["split"] == original["sage"]["split"] == "train",
                f"Wrong SAGE route or split for {name}")
        trajectory = original["sage"]["trajectory_id"]
        sample = room["samples"][str(trajectory)]
        index = arm["source_instruction_index"]
        annotation = sample["instructions"][index]
        ins, = route["sage"]["instructions"]
        expected = dict(text=annotation["generated_instruction"], start=annotation["start"],
                        end=annotation["end"], instruction_type=annotation["instruction_type"])
        require(annotation["scene_id"] == room["native_name"] and route["sage"]["trajectory_id"] == trajectory
                and ins == expected == original["sage"]["instructions"][index]
                and ins["text"] == arm["instruction"] and ins["end"] == arm["target"],
                f"Registered instruction differs from native annotation for {name}")
        evidence = dict(route=name, source_instruction_index=index, instruction=ins["text"],
                        instruction_type=ins["instruction_type"], start_object=ins["start"], goal_object=ins["end"],
                        reference_phrase=annotation.get("reference_phrase"), waypoint=original["waypoint"],
                        native_annotation=room["native_path"], native_trajectory_id=trajectory,
                        native_split="train", native_labels=room["labels_path"])
        for field, role, point, pos in (("start", "start", 0, "spawn"), ("end", "goal", -1, "waypoint")):
            xy = [original[pos][axis] for axis in "xy"]
            require(route[pos] == original[pos] == fixture[pos] and math.dist(xy, sample["points"][point]["position"][:2]) <= 1e-6,
                    f"Native {role} position differs for {name}")
            join = self._joins.get((scene["name"] + "/" + name, field))
            ident = join["ins_id"] if join else None
            if join:
                require(join["scene"] == pair["scene"] and join["target"] == ins[field]
                        and math.dist(xy, join["position_nav_xy"]) <= 1e-6 and ident in room["labels"],
                        f"Unverified {role} instance join for {name}")
            require(role == "start" or ident == arm["ins_id"], f"Unverified goal instance for {name}")
            evidence[role + "_instance"] = ident
            evidence[role + "_instance_label"] = room["labels"][ident]["label"] if ident is not None else None
        return evidence

    def _check_camera_map(self, room, fixture, manifest):
        scene = room["scene"]
        mesh = room["mesh"]
        cam, proxy = manifest["camera"], manifest["collision_proxy"]
        same_camera = all(cam[k] == scene["camera"][k] for k in ("width", "height", "fov_deg"))
        k = scene_io.intrinsics(cam["width"], cam["height"], cam["fov_deg"])
        dyn = manifest["dynamics_params"]
        self._sha(self._vehicle_path, dyn["params_sha256"])
        require(cam["mount"] == dyn["camera_mount"] == self._vehicle["camera_mount"]["value"]
                and same_camera and [cam[x] for x in ("fx", "fy", "cx", "cy")] == [k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
                f"Teacher camera differs from native scene or vehicle: {scene['name']}")
        require(fixture["camera"] == scene["camera"] and fixture["nav_to_world"] == scene["nav_to_world"]
                and reach_radius_of(fixture) == manifest["stop_radius"] == room["radius"]
                and self._path(fixture["ply"]) == Path(scene["ply_path"])
                and manifest["renderer"]["near_plane_m"] == scene["render"]["near_plane_m"],
                f"Teacher view or goal radius differs from native scene: {scene['name']}")
        require(self._path(proxy["geometry"]) == self._path(fixture["collision"]["geometry"]) == mesh
                and proxy["source"] == scene["collision"]["source"] == "mesh" and proxy["collision_model"] == "v1"
                and proxy["z_range"] == scene["collision"]["z_slice"] and proxy["voxel_m"] == scene["collision"]["voxel_m"]
                and proxy["n_occupied_cells"] > 0, f"Teacher occupancy differs from native scene: {scene['name']}")
        self._sha(scene["ply_path"], manifest["scene"]["ply_sha256"])

    def _arm(self, room, pair, arm_no, reports):
        arm = pair["arms"][arm_no]
        mask = self._masks[pair["pair"]]["arms"][arm_no]
        fixture_path = self._path(arm["fixture"])
        self._sha(room["scene"]["path"], arm["source_scene_sha256"])
        fixture = self._json(fixture_path, arm["fixture_sha256"])
        evidence = self._instruction(room, pair, arm, fixture)
        out = self._path(arm["out"])
        logs = out / room["scene"]["name"] / (arm["route"] + "_0")
        ep, = self._jsonl(out / "episodes.jsonl")
        run = self._json(out / "vln_eval.json")
        require(run["policy"] == "teacher" and run["dynamics"] == "cf_sitl" and run["counterfactual"] == "correct"
                and not run["blind_rgb"] and not run["failures"] and run["overall"]["n_requested"] == 1
                and run["teacher_clearance"] == "braking" and run["hold_s"] == 0.5 and run["seed"] == pair["seed"],
                f"Teacher run report differs for {out}")
        manifest_path = logs / "episode_manifest.json"
        manifest = self._json(manifest_path)
        snapshot = logs / "scene.json"
        require(self._json(snapshot) == fixture and fixture["name"] == room["scene"]["name"] == manifest["scene"]["name"]
                and self._path(ep["dir"]) == self._path(mask["dir"]) == logs
                and manifest["scene"]["scene_json_sha256"] == arm["fixture_sha256"], f"Wrong scene snapshot for {logs}")
        goal, start, text = evidence["goal_object"], evidence["start_object"], evidence["instruction"]
        for source in (manifest["vln"], ep):
            require(source["route_name"] == arm["route"] and source["text"] == source["supplied_text"] == text
                    and source["goal_object"] == source["supplied_goal_object"] == goal and source["start_object"] == start
                    and source["split"] == "train" and source["counterfactual"] == "correct"
                    and source["supplied_goal_waypoint"] == evidence["waypoint"], f"Teacher input differs for {logs}")
        require(manifest["executed_by"] == "teacher" and manifest["teacher_clearance"] == "braking"
                and manifest["planner"] == "astar_occ" and manifest["dynamics_backend"] == "cf_sitl"
                and manifest["renderer"]["backend"] == manifest["camera_source"] == "gsplat"
                and not manifest["vln"]["blind_rgb"] and all(manifest["validity"][k] for k in
                    ("camera_verified", "physics_verified", "episode_executed")), f"Teacher or camera mode differs for {logs}")
        require(manifest["hold_s"] == 0.5 and manifest["dynamics_params"]["trace_dt_s"] == 0.1
                and manifest["seed"] == ep["seed"] == pair["seed"] and manifest["requested_steps"] == pair["common_budget"],
                f"Teacher timing, seed or step budget differs for {logs}")
        require(manifest["route"]["name"] == arm["route"] and manifest["scene"]["spawn"] == fixture["spawn"]
                and manifest["waypoint"] == manifest["scene"]["waypoint"] == evidence["waypoint"],
                f"Teacher route or endpoint differs for {logs}")
        self._check_camera_map(room, fixture, manifest)
        actions_path, trace_path = logs / "actions.jsonl", logs / "trace.jsonl"
        steps = manifest["num_steps"]
        for report in reports:
            counts = report["counts"][arm_no]
            replay = counts["counts"]
            require(counts["arm"] == arm_no and replay["exact_episode"]
                    and replay["action_match"] == replay["stop_match"] == replay["steps"] == steps,
                    f"Frozen action/STOP replay incomplete for {logs}")
            self._sha(actions_path, counts["actions_sha256"])
            self._sha(trace_path, counts["trace_sha256"])
        actions, trace = self._jsonl(actions_path), self._jsonl(trace_path)
        require(len(actions) == ep["steps"] == mask["steps"] == steps and len(trace) == 5 * steps
                and all(a["step"] == i for i, a in enumerate(actions))
                and all(t["step"] == i // 5 for i, t in enumerate(trace)),
                f"Incomplete teacher action/trace sequence for {logs}")
        evidence.update(fixture=str(fixture_path), teacher_manifest=str(manifest_path), teacher_scene_snapshot=str(snapshot),
                        teacher_yaw_tol_deg=manifest["teacher_yaw_tol_deg"])
        return dict(evidence=evidence, actions=actions, trace=trace, logs=logs, camera=manifest["camera"],
                    spawn=fixture["spawn"], occupied=manifest["collision_proxy"]["n_occupied_cells"])

    def _pair(self, key):
        if key in self._episodes:
            return self._episodes[key]
        pair, mask = self.pairs[key], self._masks[key]
        require(mask["split"] == pair["split"] and mask["scene"] == pair["scene"] and mask["seed"] == pair["seed"]
                and mask["budget"] == pair["common_budget"], f"Wrong mask metadata for {key}")
        room = self._room(pair["scene"])
        reports = [self._json(self.archive / group / (key + ".json")) for group in ("per_pair", "equal_per_pair")]
        arms = [self._arm(room, pair, j, reports) for j in (0, 1)]
        a, b = arms
        spawn = pair["scene"] + ":" + json.dumps(a["spawn"], sort_keys=True, separators=(",", ":"))
        routes = [arm["route"] for arm in pair["arms"]]
        ids = [arm["ins_id"] for arm in pair["arms"]]
        goals = [[arm["evidence"]["waypoint"][k] for k in "xy"] for arm in arms]
        for report in reports:
            require(report["pair"] == key and report["scene"] == pair["scene"] and report["split"] == pair["split"]
                    and report["routes"] == routes and report["spawn"] == spawn and report["pair_sep"] == pair["waypoint_sep_m"],
                    f"Per-pair source metadata differs for {key}")
        require(a["spawn"] == b["spawn"] and a["camera"] == b["camera"] and a["occupied"] == b["occupied"]
                and math.isclose(math.dist(*goals), pair["waypoint_sep_m"], abs_tol=1e-6)
                and pair["waypoint_sep_m"] > 2 * room["radius"] and len(set(ids)) == 2 and ids == mask["ids"],
                f"Ambiguous starts, goals or camera for {key}")
        fork = shared_fork(arms, mask, self._rgb)
        for arm, masked in zip(arms, mask["arms"]):
            frame = arm["logs"] / arm["actions"][fork]["frame"]
            require(self._path(masked["fork_frame"]) == frame, f"Fork image path differs at {key}")
            self._sha(frame, masked["fork_image_sha256"])
        records = [{(r["arm"], r["step"]): r for r in report["records"]} for report in reports]
        result = dict(arms=arms, room=room, fork=fork, records=records, spawn=spawn)
        self._episodes[key] = result
        return result

    def resolve(self, row):
        """核验一行已执行的仿真状态和两条原生指令，并返回该观测是否由两臂实际共享。"""
        pool, line = row["source_pool"], row["source_line"]
        table = self._tables[pool]
        key, j, step = row["pair"], row["arm"], row["step"]
        raw = {k: v for k, v in row.items() if k not in ("source_pool", "source_line")}
        require(0 < line <= len(table) and table[line - 1] == raw and j in (0, 1) and step > 0,
                f"Unregistered row in {pool}:{line}")
        source = self._pair(key)
        arms, room, fork = source["arms"], source["room"], source["fork"]
        require(source["records"][int(pool.startswith("equal_"))].get((j, step)) == raw
                and source["spawn"] == row["spawn_group"], f"Frozen row differs from pair report for {key}")
        instructions = []
        for i, data in enumerate(arms):
            evidence = data["evidence"]
            require((evidence["route"], evidence["goal_object"], evidence["goal_instance"], evidence["instruction"]) ==
                    (row["routes"][i], row["targets"][i], row["ids"][i], row["instructions"][i]),
                    f"Arm {i} differs from source row for {key}")
            instructions.append(dict(evidence, original_action=row["actions"][i]))
        arm = arms[j]
        decision = arm["actions"][step]
        previous = arm["trace"][step * 5 - 1]
        state = pre_action_state(decision, previous)
        image = arm["logs"] / decision["frame"]
        require(image == self._path(row["rgb"]) and self._rgb(image, arm["camera"]) == row["rgb_sha256"]
                and not decision["collision"] and decision["rgb_ok"] and not decision["frame_quality"]["low"],
                f"Wrong or invalid factual RGB at {pool}:{line}")
        shared = step <= fork
        require(row["shared_executed_state"] == (step == fork), f"Historical fork flag differs at {pool}:{line}")
        for i in ((0, 1) if shared else (j,)):
            logged = arms[i]["actions"][step]
            require(logged["action_expert"] == logged["action_executed"] == row["actions"][i]
                    and not logged["stop_expert"] and not logged["stop_requested"], f"Factual action/STOP differs at {pool}:{line}")
        provenance = dict(registered_shared_prefix=row["shared_executed_state"], verified_shared_prefix=shared,
                          first_unequal_executed_action_step=fork, rgb=str(image), rgb_sha256_png=self._sha(image),
                          pair_registration=self.pairs[key]["registration"], occupied_cells=arm["occupied"],
                          actions_path=str(arm["logs"] / "actions.jsonl"), trace_path=str(arm["logs"] / "trace.jsonl"),
                          action_timestamp_ns=decision["timestamp_ns"], source_trace_time_s=previous["t"],
                          pose_nav_m=[state.x, state.y, state.z], yaw_before_deg=decision["yaw_before"],
                          body_quat_wxyz=previous["quat"], pre_action_velocity_xyz_mps=previous["vel"],
                          speed_xy_mps=math.hypot(state.vx, state.vy), reach_radius_m=room["radius"],
                          goal_separation_m=self.pairs[key]["waypoint_sep_m"], factual_logged_teacher_bin=decision["bearing_bin"])
        return dict(scene=room["scene"], camera=arm["camera"], state=state, previous_trace=previous,
                    radius=room["radius"], instructions=instructions, provenance=provenance)
