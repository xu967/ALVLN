"""用真实动作前图像、InteriorGS 实例框和光学深度复核同图分岔。输出只供标注。"""
import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.vln_fork_references import lookup_reference, read_reference_spans


SCENE = importlib.import_module("sim.3dgs.scene")
ROOMS = {"839992": "development", "840865": "development",
         "840139": "train", "841012": "train", "841150": "train"}
REFERENCE_FILE = Path(__file__).with_name("data") / "vln_fork_reference_spans.json"
RAY_LIBRARY = Path("exp-vln-new-scenes-20261124/acquisition_audit/ray.so")
RAY_SHA256 = "746d9debb04846305d7490206dc2fab6aeefbd9ffee35a08f1c1e75104027250"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def read_jsonl(path):
    with path.open("rb") as stream:
        return [json.loads(line) for line in stream]


def physical_site(row):
    return (row["scene"], *(round(row["spawn"][k], 5) for k in "xyz"))


def observation_role(row):
    if row["step"] == 0:
        if not row.get("verified_reset", row["verified_shared_prefix"]):
            raise ValueError(f"复位未通过核验 {row['pair']}")
        return "reset_only"
    if not row["verified_shared_prefix"]:
        raise ValueError(f"输入含未经核实的共同前缀 {row['pair']}")
    if row["step"] < 0:
        raise ValueError("决策步不能为负")
    arms = row["labels"]
    bins = {a.get("local_bearing_bin") for a in arms}
    strict = (len(bins) == 2 and bins <= {0, 1, 7}
              and all(a.get("planning_status") == "astar" and a.get("inside_actual_image") is True for a in arms)
              and row["status"] == "different_coarse_both_in_image")
    return "primary" if strict else "flown_control"


def representatives(rows):
    """每地点选严格见证，其次选共同飞过的对照；仅有复位的地点另列。"""
    priority = {"primary": 0, "flown_control": 1, "reset_only": 2}
    selected = {}
    for row in rows:
        site, role = physical_site(row), observation_role(row)
        if ROOMS.get(row["scene"]) != row["split"] or len(row["labels"]) != 2:
            raise ValueError(f"实验房间、划分或双臂数不符 {row['pair']}")
        rank = (priority[role], row["step"])
        if site not in selected or rank < selected[site][0]:
            selected[site] = (rank, role, row)
    return [(role, row) for _, role, row in selected.values()]


def camera_rays(camera):
    u, v = np.meshgrid(np.arange(camera["width"]), np.arange(camera["height"]))
    return np.stack(((u - camera["cx"]) / camera["fx"], (v - camera["cy"]) / camera["fy"],
                     np.ones_like(u)), axis=-1)


def classify_box(corners, view, camera, depth, near, far):
    """量每个像素射线穿过 OBB 的深度区间。框内有表面并不能证明其语义身份。"""
    if corners is None:
        return {"status": "unknown", "reason": "missing_official_box"}
    box = np.asarray(corners, dtype=float)
    if box.shape != (8, 3) or not np.isfinite(box).all():
        return {"status": "unknown", "reason": "invalid_official_box"}
    box = box @ view[:3, :3].T + view[:3, 3]
    edges = (box[[1, 3, 4]] - box[0]).T
    if abs(np.linalg.det(edges)) <= 1e-10:
        return {"status": "unknown", "reason": "degenerate_official_box"}
    h, w = camera["height"], camera["width"]
    x, y, z = box.T
    u, v = camera["fx"] * x + camera["cx"] * z, camera["fy"] * y + camera["cy"] * z
    planes = np.stack([z - near, far - z, u, w * z - u, v, h * z - v])
    if (planes.max(axis=1) < 0).any():
        return {"status": "outside_image", "reason": "outside_camera_frustum", "box_pixels": 0}
    inv = np.linalg.inv(edges)
    ray = camera_rays(camera) @ inv.T
    origin = -inv @ box[0]
    parallel = np.abs(ray) < 1e-12
    valid = ~(parallel & ((origin < 0) | (origin > 1))).any(axis=-1)
    divisor = np.where(parallel, 1, ray)
    a, b = -origin / divisor, (1 - origin) / divisor
    entry = np.maximum(near, np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=-1))
    exit_depth = np.minimum(far, np.where(parallel, np.inf, np.maximum(a, b)).min(axis=-1))
    roi = valid & (exit_depth >= entry)
    yy, xx = np.nonzero(roi)
    if not len(xx):
        return {"status": "unknown", "reason": "no_pixel_center_crosses_box", "box_pixels": 0}
    result = {"box_pixels": len(xx), "box_xyxy_inclusive": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]}
    if depth is None:
        return dict(result, status="unknown", reason="no_optical_depth")
    if depth.shape != (h, w):
        raise ValueError("光学深度尺寸与相机不符")
    zz, front, back = depth[roi], entry[roi], exit_depth[roi]
    measured = np.isfinite(zz) & (zz > near) & (zz < far)
    closer = measured & (zz < front - 0.05)
    compatible = measured & (zz >= front - 0.05) & (zz <= back + 0.05)
    result.update(front_pixels=int(closer.sum()), compatible_pixels=int(compatible.sum()),
                  behind_pixels=int((measured & ~closer & ~compatible).sum()), invalid_pixels=int((~measured).sum()))
    if result["compatible_pixels"] >= 9:
        return dict(result, status="unobstructed_candidate", reason="surface_depth_inside_official_box")
    if not measured.any():
        return dict(result, status="unknown", reason="no_valid_depth_in_box")
    reason = "all_pixels_have_nearer_surface" if closer.all() else "insufficient_compatible_pixels"
    return dict(result, status="occluded_or_unconfirmed", reason=reason)


def collision_depth(root, mesh, camera, view):
    """旧图缺实测渲染深度时，复用已登记的 Embree 首交。PLY 没有语义 ID。"""
    library = root / RAY_LIBRARY
    if not library.is_file() or not mesh.is_file():
        return None
    if digest(library) != RAY_SHA256:
        raise ValueError(f"首交射线库指纹不符 {library}")
    lib = ctypes.CDLL(str(library))
    ptr, integer = ctypes.c_void_p, ctypes.c_uint
    for op, args, result in [
        ("rtcNewDevice", [ctypes.c_char_p], ptr), ("rtcNewScene", [ptr], ptr),
        ("add", [ptr, ptr, integer, ptr, integer], integer), ("commit", [ptr], None),
        ("cast", [ptr, ptr, ptr, integer, ptr, ptr], None),
        ("rtcReleaseScene", [ptr], None), ("rtcReleaseDevice", [ptr], None),
    ]:
        fn = getattr(lib, op)
        fn.argtypes, fn.restype = args, result
    geometry = importlib.import_module("sim.3dgs.collision").read_ply_vertices(mesh)
    vertices = np.ascontiguousarray(geometry["xyz"], dtype=np.float32)
    faces = np.ascontiguousarray(geometry["faces"], dtype=np.uint32)
    # 首交库的顶点是 float，光心和方向是 double，返回相机 z 深度 float。
    eye = np.ascontiguousarray(-view[:3, :3].T @ view[:3, 3], dtype=np.float64)
    rays = np.ascontiguousarray(camera_rays(camera).reshape(-1, 3) @ view[:3, :3], dtype=np.float64)
    ids, depths = np.empty(len(rays), np.uint32), np.empty(len(rays), np.float32)
    device = lib.rtcNewDevice(None)
    model = lib.rtcNewScene(device)
    try:
        lib.add(model, vertices.ctypes.data, len(vertices), faces.ctypes.data, len(faces))
        lib.commit(model)
        lib.cast(model, eye.ctypes.data, rays.ctypes.data, len(rays), ids.ctypes.data, depths.ctypes.data)
    finally:
        lib.rtcReleaseScene(model)
        lib.rtcReleaseDevice(device)
    depths[ids == np.iinfo(np.uint32).max] = np.nan
    return depths.reshape(camera["height"], camera["width"])


def reference_review(spans, text, goal):
    identity = json.dumps([text, goal], ensure_ascii=False, separators=(",", ":"))
    key = hashlib.sha256(identity.encode()).hexdigest()
    if key not in spans:
        return dict(instruction_goal_sha256=key, status="pending_text_review", phrases=None, instance_ids=None)
    return dict(instruction_goal_sha256=key, **lookup_reference(spans, text, goal))


def room_inputs(root, scene):
    folder = root / "3dgs-scenes" / ("sage3d_" + scene)
    scene_path = folder / "scene.json"
    model = read_json(scene_path)
    labels_path, = (root / "pubdata/sage3d/interiorgs_labels").glob("*_" + scene + "/labels.json")
    labels = {str(r["ins_id"]): r for r in read_json(labels_path)}
    return dict(model=model, labels=labels, labels_path=str(labels_path), labels_sha256=digest(labels_path),
                scene_sha256=digest(scene_path), mesh=folder / model["collision"]["geometry"])


def frame_evidence(root, row, room):
    rgb_path = Path(row["rgb"])
    with Image.open(rgb_path) as im:
        rgb = im.convert("RGB")
    if hashlib.sha256(rgb.tobytes()).hexdigest() != row["rgb_sha256"]:
        raise ValueError(f"动作前 RGB 指纹不符 {rgb_path}")
    manifest = read_json(rgb_path.parent / "episode_manifest.json")
    action = read_jsonl(rgb_path.parent / "actions.jsonl")[row["step"]]
    camera = dict(manifest["camera"], near_plane_m=manifest["renderer"]["near_plane_m"],
                  far_plane_m=manifest["renderer"]["far_plane_m"])
    if (action["step"] != row["step"] or rgb_path.name != action["frame"] or
            rgb.size != (camera["width"], camera["height"]) or
            [action["pose_before"][k] for k in "xyz"] != row["pose_nav_m"] or
            action["timestamp_ns"] != action["rgb_ts"] or action["rgb_ts"] != action["pose_ts"]):
        raise ValueError(f"RGB 与动作前位姿或相机不符 {rgb_path}")
    quat = manifest.get("initial_body_quat_wxyz") if row["step"] == 0 else row.get("body_quat_wxyz")
    if quat is None:
        return rgb, camera, None, None, {"type": "unknown", "reason": "missing_pre_action_quaternion"}
    view = SCENE.viewmat_from_quat(row["pose_nav_m"], quat, camera["mount"], room["model"]["nav_to_world"])
    if action.get("depth_file"):
        path = rgb_path.parent / action["depth_file"]
        depth = np.load(path, allow_pickle=False)
        sha = hashlib.sha256(depth.tobytes()).hexdigest()
        if row.get("depth_sha256", sha) != sha:
            raise ValueError(f"动作前深度指纹不符 {path}")
        return rgb, camera, view, depth, {"type": "rendered_camera_z", "path": str(path), "decoded_sha256": sha}
    depth = collision_depth(root, room["mesh"], camera, view)
    info = {"type": "unknown"}
    if depth is not None:
        info = {"type": "collision_mesh_first_surface", "path": str(room["mesh"]), "sha256": digest(room["mesh"])}
    return rgb, camera, view, depth, info


def review_frame(root, row, role, room, spans):
    rgb, camera, view, depth, surface = frame_evidence(root, row, room)
    targets = []
    for arm in row["labels"]:
        target = room["labels"].get(str(arm["goal_instance"]))
        corners = None
        if target and target.get("bounding_box"):
            corners = [[p[k] for k in "xyz"] for p in target["bounding_box"]]
        if view is None or row["status"] in ("low_quality_rgb", "reset_low_quality_rgb"):
            evidence = {"status": "unknown", "reason": "unusable_image_or_pose"}
        else:
            evidence = classify_box(corners, view, camera, depth, camera["near_plane_m"], camera["far_plane_m"])
        reference = reference_review(spans, arm["instruction"], arm["goal_object"])
        reference["objects_to_review"] = [{"phrase": phrase, "status": "pending", "instance_id": None, "box": None}
                                          for phrase in reference["phrases"] or []]
        targets.append(dict(route=arm["route"], instruction=arm["instruction"], goal_object=arm["goal_object"],
                            goal_instance=arm["goal_instance"], official_class=target["label"] if target else None,
                            target_geometry=evidence, target_rgb_review={"status": "pending", "box": None},
                            reference=reference,
                            waypoint={k: arm.get(k) for k in ("local_bearing_bin", "inside_actual_image",
                                                              "relative_yaw_ccw_deg", "image_uv_same_altitude_px")}))
    identity = {k: row[k] for k in ("scene", "split", "pair", "step", "rgb", "rgb_sha256", "selection_group", "heading_source")}
    return dict(identity, role=role, spawn=list(physical_site(row)[1:]), source_status=row["status"], labels=targets,
                camera=camera, depth_source=surface, official_labels=room["labels_path"],
                official_labels_sha256=room["labels_sha256"], scene_sha256=room["scene_sha256"]), rgb


def draw_review(rgb, record, path):
    """原图、官方目标候选框与 A* 瞄准点各占一栏；不给未标身份的语言参照画框。"""
    w, h = rgb.size
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    colors = ["#f0a330", "#43b9f4"]
    notes = []
    for i, arm in enumerate(record["labels"]):
        goal, waypoint = arm["target_geometry"], arm["waypoint"]
        notes += [f"{'AB'[i]}  {arm['goal_object']}  [ID {arm['goal_instance']}]   {goal['status']}"]
        notes += textwrap.wrap(arm["instruction"], width=max(22, (3*w-32)//9))
        where = {True: "in frame", False: "out of frame", None: "unlabelled"}[waypoint["inside_actual_image"]]
        notes.append(f"   A* waypoint: {where}     Reference text: {arm['reference']['status']}")
    sheet = Image.new("RGB", (3*w, h+72+23*len(notes)), "#161b20")
    for x in (0, w, 2*w):
        sheet.paste(rgb, (x, 34))
    draw = ImageDraw.Draw(sheet)
    for i, txt in enumerate(("Original onboard RGB", "Official OBB candidates", "A* waypoints")):
        draw.text((w*i+8, 6), txt, font=font, fill="white")
    for i, arm in enumerate(record["labels"]):
        box = arm["target_geometry"].get("box_xyxy_inclusive")
        if box:
            x, y, xx, yy = box
            draw.rectangle((w+x, 34+y, w+xx, 34+yy), outline=colors[i], width=2)
            draw.text((w+min(x+2, w-16), 34+min(y+1, h-18)), "AB"[i], font=font, fill=colors[i],
                      stroke_width=1, stroke_fill="black")
        way = arm["waypoint"]
        if way["inside_actual_image"] is True:
            x, y = way["image_uv_same_altitude_px"]
            x, y = round(x) + 2*w, round(y)+34
            draw.ellipse((x-6, y-6, x+6, y+6), outline=colors[i], width=3)
            draw.text((min(x+7, 3*w-16), y-10), "AB"[i], font=font, fill=colors[i], stroke_width=1, stroke_fill="black")
    for j, line in enumerate(notes):
        draw.text((9, h+44+23*j), line, fill="white", font=font)
    sheet.save(path)


def reset_control(pair, design_pair, source_hashes):
    """没有共同已执行窗时，从两条已核验的真实第 0 帧重建复位对照。"""
    shots, targets = [], []
    for report, arm in zip(pair["arm_results"], design_pair["arms"]):
        if report["status"] != "verified" or report["route"] != arm["route"]:
            raise ValueError(f"复位路线未通过核验 {pair['pair']}")
        episodes = Path(report["out"]) / "episodes.jsonl"
        if digest(episodes) != source_hashes[str(episodes)]:
            raise ValueError(f"复位集清单指纹不符 {episodes}")
        ep, = read_jsonl(episodes)
        logs = Path(ep["dir"])
        path = logs / "actions.jsonl"
        if digest(path) != source_hashes[str(path)]:
            raise ValueError(f"复位动作指纹不符 {path}")
        first = read_jsonl(path)[0]
        if first["step"] != 0 or first["timestamp_ns"] != first["rgb_ts"] or first["rgb_ts"] != first["pose_ts"]:
            raise ValueError(f"复位帧时序不符 {path}")
        rgb = logs / first["frame"]
        run_manifest = logs / "episode_manifest.json"
        if digest(run_manifest) != source_hashes[str(run_manifest)]:
            raise ValueError(f"复位相机清单指纹不符 {run_manifest}")
        manifest = read_json(run_manifest)
        with Image.open(rgb) as image:
            sha = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        depth = np.load(logs / first["depth_file"], allow_pickle=False)
        shots.append(dict(rgb=str(rgb), rgb_sha256=sha, depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
                          pose_nav_m=[first["pose_before"][k] for k in "xyz"], yaw=first["yaw_before"],
                          time=first["timestamp_ns"], quat=manifest.get("initial_body_quat_wxyz"),
                          camera=manifest["camera"], low=first["frame_quality"]["low"]))
        if ep["supplied_text"] != arm["instruction"] or ep["supplied_goal_object"] != arm["target"]:
            raise ValueError(f"复位原句或目标不符 {episodes}")
        targets.append(dict(route=arm["route"], instruction=arm["instruction"], goal_object=arm["target"],
                            goal_instance=str(arm["ins_id"])))
    if len(shots) != 2 or {k: v for k, v in shots[0].items() if k != "rgb"} != {k: v for k, v in shots[1].items() if k != "rgb"}:
        raise ValueError(f"两臂复位姿态、相机、RGB 或深度不同 {pair['pair']}")
    shot = shots[0]
    row = {k: pair[k] for k in ("scene", "split", "pair", "spawn", "selection_group", "heading_source")}
    return dict(row, **{k: shot[k] for k in ("rgb", "rgb_sha256", "depth_sha256", "pose_nav_m")},
                labels=targets, step=0, verified_reset=True, verified_shared_prefix=False,
                status="reset_low_quality_rgb" if shot["low"] else "reset_only_no_flown_history")


def build(observations, out, root):
    verification = read_json(observations.with_name("summary.json"))
    fingerprint = digest(observations)
    if verification["observations_sha256"] != fingerprint:
        raise ValueError("输入状态清单与实飞核验报告的指纹不符")
    rows = read_jsonl(observations)
    flown_sites = {physical_site(r) for r in rows}
    reset_pairs = [p for p in verification["pairs"] if p.get("verified_reset") and physical_site(p) not in flown_sites]
    reset_rows = []
    if reset_pairs:
        design = Path(verification["design"])
        if digest(design) != verification["source_sha256"][str(design)]:
            raise ValueError(f"复位设计清单指纹不符 {design}")
        by_pair = {p["pair"]: p for p in read_json(design)["pairs"]}
        reset_rows = [reset_control(p, by_pair[p["pair"]], verification["source_sha256"]) for p in reset_pairs]
    selected = representatives(rows + reset_rows)
    earned = {(row["scene"], row["pair"], row["step"]) for role, row in selected if role == "primary"}
    expected = {(r["scene"], r["pair"], r["step"]) for r in verification["all"]["first_primary_per_start"]}
    if earned != expected:
        raise ValueError("代表帧与实飞核验器登记的各起点首个严格见证不符")
    if out.exists():
        raise FileExistsError(f"输出目录已存在 {out}")
    rooms = {name: room_inputs(root, name) for name in sorted({row["scene"] for _, row in selected})}
    spans = read_reference_spans(REFERENCE_FILE)
    role_counts, target_counts, references = Counter(), Counter(), {}
    all_status = defaultdict(Counter)
    for row in rows:
        group = "/".join(row[k] for k in ("split", "scene", "selection_group", "heading_source"))
        all_status[group][row["status"]] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "review"
        stage.mkdir()
        with (stage / "review.jsonl").open("w", encoding="utf-8") as fp:
            for i, (role, row) in enumerate(selected):
                evidence, rgb = review_frame(root, row, role, rooms[row["scene"]], spans)
                role_counts[role] += 1
                for a in evidence["labels"]:
                    target_counts[a["target_geometry"]["status"]] += 1
                    reference = a["reference"]
                    references[reference["instruction_goal_sha256"]] = reference
                folder = stage / role
                folder.mkdir(exist_ok=True)
                path = folder / f"{i:03d}_{row['scene']}_step{row['step']}.png"
                draw_review(rgb, evidence, path)
                evidence.update(review_image=str(path.relative_to(stage)), review_image_sha256=digest(path))
                print(json.dumps(evidence, ensure_ascii=False), file=fp)
        metadata = dict(schema="uavrgb.vln.shared_fork_visibility.v1", verified_observations_sha256=fingerprint,
                        verified_summary_sha256=digest(observations.with_name("summary.json")),
                        reference_spans_sha256=digest(REFERENCE_FILE), input_path=str(observations),
                        input_counts_by_split_scene_selection_heading=all_status, requested=verification["all"],
                        requested_by_selection_group=verification.get("by_selection_group", {}),
                        requested_by_heading_source=verification.get("by_heading_source", {}),
                        source_pair_counts=verification["all"]["pair_status"], selected_sites=len(selected),
                        pairs_without_image=[p["pair"] for p in verification["pairs"] if not p.get("verified_reset")
                                             and physical_site(p) not in flown_sites],
                        representative_roles=role_counts, reviewed_arm_geometry=target_counts,
                        unique_instruction_goal_text_status=Counter(r["status"] for r in references.values()),
                        known_reference_spans=sum(len(r["phrases"] or []) for r in references.values()),
                        review_rows_sha256=digest(stage / "review.jsonl"),
                        rule="每个物理出生点一帧。优先 k>0 且两条真实共享 A* 方位不同、均可表达且双路标入画的首帧；其次最早实飞共享对照；复位单列。相同步沿用设计顺序。",
                        surface_rule="按原图逐像素计算官方 OBB 的光轴深度区间，容差 5 cm。至少 9 条射线深度相容只构成候选；每个穿框像素均有更近的有效表面才标全遮挡。无证据或缺框分列。",
                        human_review="先看原图逐目标核对可辨类别、区分同类实例，再对候选框和实际表面做标注。为目标和每个参照登记可见/部分可见/遮挡/画外/歧义、独立证据及手画框；地标身份不明时保持空。新原句先核目标、搬运物与参照。主样本与控制样本均需人工复核。")
        (stage / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        stage.rename(out)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.observations, args.out, args.data_root)["representative_roles"]))
