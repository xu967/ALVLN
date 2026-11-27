"""为同一张机载图上的两条原句重放局部航点监督。仅离线使用地图和教师记录。"""
import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .vln_fork_geometry import label_goal
from .vln_fork_references import lookup_reference, read_reference_spans
from .vln_fork_sources import ForkSources


SPLITS = ("train", "development")
COARSE_BINS = {0, 1, 7}
SCENE = import_module("sim.3dgs.scene")
COLLISION = import_module("sim.3dgs.collision")
PLANNER = import_module("sim.3dgs.planner")
VLN = import_module("sim.3dgs.vln")
REFERENCE_SPANS = Path(__file__).with_name("data") / "vln_fork_reference_spans.json"
SOURCE_FILES = (
    "tools/label_vln_forks.py", "tools/vln_fork_sources.py", "tools/vln_fork_geometry.py", "tools/vln_fork_references.py",
    "sim/3dgs/scene.py", "sim/3dgs/vln.py", "sim/3dgs/planner.py", "sim/3dgs/collision.py", "sim/3dgs/eval_3dgs.py",
)


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def label_state(row, record, planner, references):
    state, scene, camera = record["state"], record["scene"], record["camera"]
    prior = record["previous_trace"]
    view = SCENE.viewmat_from_quat((state.x, state.y, state.z), prior["quat"], camera["mount"], scene["_nav_to_world"])
    labels = []
    for arm in record["instructions"]:
        label = label_goal(planner, state, arm["waypoint"], record["radius"], camera, view, scene["_nav_to_world"],
                           arm["original_action"], arm["teacher_yaw_tol_deg"], near_plane_m=scene["render"]["near_plane_m"])
        references_for_goal = lookup_reference(references, arm["instruction"], arm["goal_object"])
        labels.append({**arm, **label, "destination_references": references_for_goal})
    if len(labels) != 2:
        raise ValueError(f"{row['pair']}: 需要两条已登记的目标指令")
    source_fields = ("split", "scene", "pair", "spawn_group", "source_pool", "source_line", "arm", "step", "rgb_sha256")
    result = {key: row[key] for key in source_fields}
    result.update(record["provenance"], original_actions=row["actions"], labels=labels)
    if labels[row["arm"]]["teacher_runtime_bin"] != result["factual_logged_teacher_bin"]:
        raise ValueError(f"教师实飞臂方位与重放不符: {row['pair']} 第 {row['step']} 步")
    return result


def counts(rows):
    labels = [label for row in rows for label in row["labels"]]
    status = Counter(label["planning_status"] for label in labels)
    valid = [label for label in labels if label["planning_status"] == "astar"]
    pairs = [row for row in rows if all(label["planning_status"] == "astar" for label in row["labels"])]
    different = [row for row in pairs if row["labels"][0]["local_bearing_bin"] != row["labels"][1]["local_bearing_bin"]]
    coarse = [row for row in different if all(label["local_bearing_bin"] in COARSE_BINS for label in row["labels"])]
    projected = [row for row in coarse if all(label["inside_actual_image"] for label in row["labels"])]
    histogram = Counter(label["local_bearing_bin"] for label in valid)
    return {
        "states": len(rows), "instruction_requests": len(labels),
        "route_pairs": len({row["pair"] for row in rows}), "spawns": len({row["spawn_group"] for row in rows}),
        "shared_prefix": sum(row["verified_shared_prefix"] for row in rows),
        "single_factual_arm": sum(not row["verified_shared_prefix"] for row in rows),
        "archived_shared_flag": sum(row["registered_shared_prefix"] for row in rows),
        "both_valid": len(pairs), "same_bin": len(pairs) - len(different), "different_bin": len(different),
        "both_coarse": sum(all(label["local_bearing_bin"] in COARSE_BINS for label in r["labels"]) for r in pairs),
        "different_both_coarse": len(coarse), "different_coarse_route_pairs": len({r["pair"] for r in coarse}),
        "different_coarse_spawns": len({r["spawn_group"] for r in coarse}),
        "different_coarse_shared": sum(row["verified_shared_prefix"] for row in coarse),
        "different_coarse_both_in_image": len(projected),
        "different_coarse_both_in_image_shared": sum(row["verified_shared_prefix"] for row in projected),
        "in_image": sum(label["inside_actual_image"] for label in valid),
        "outside_image": sum(not label["inside_actual_image"] for label in valid),
        "outside_horizontal_fov": sum(not label["inside_actual_camera_horizontal_fov"] for label in valid),
        "outside_coarse_bins": sum(label["local_bearing_bin"] not in COARSE_BINS for label in valid),
        "planning_status": dict(sorted(status.items())),
        "bearing_bins": {str(i): histogram[i] for i in range(8)},
        "actions_replayed": sum(label["replayed_teacher_action"] == label["original_action"] for label in labels),
        "coarse_action_agrees": sum(label.get("agrees_with_original_action") is True for label in valid),
        "coarse_action_differs": sum(label.get("agrees_with_original_action") is False for label in valid),
    }


def grouped_counts(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {value: counts(group) for value, group in sorted(groups.items())}


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def build(data_root, reference_path, out):
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"输出已存在: {out}")
    references, sources = read_reference_spans(reference_path), ForkSources(data_root)
    rows, planners, seen = [], {}, set()
    for row in sources.rows():
        if row["rgb_sha256"] in seen or row["split"] not in SPLITS:
            raise ValueError(f"冻结清单图像重复或划分未知: {row['pair']} 第 {row['step']} 步")
        seen.add(row["rgb_sha256"])
        record = sources.resolve(row)
        room = row["scene"]
        if room not in planners:
            planners[room] = PLANNER.OccPlanner(COLLISION.from_scene(record["scene"]))
        if planners[room].occ.n_occupied != record["provenance"]["occupied_cells"]:
            raise ValueError(f"{room} 实际占据数与飞行记录不符")
        rows.append(label_state(row, record, planners[room], references))
    used = {(arm["instruction"], arm["goal_object"]): arm["destination_references"]["status"]
            for row in rows for arm in row["labels"]}
    if len(used) != len(references):
        raise ValueError("已登记指令与参照词表覆盖不同")
    root = Path(__file__).resolve().parents[1]
    summary = {"schema": "uavrgb.vln.fork_bearings.v1", "subgoal_contract": VLN.SUBGOAL_BEARING_CONTRACT,
               "all": counts(rows), "by_split": grouped_counts(rows, "split"), "by_room": grouped_counts(rows, "scene"),
               "by_pool": grouped_counts(rows, "source_pool"), "by_spawn": grouped_counts(rows, "spawn_group"),
               "by_route_pair": grouped_counts(rows, "pair"), "reference_roles_by_instruction": dict(sorted(Counter(used.values()).items())),
               "code_sha256": {name: sha256(root / name) for name in SOURCE_FILES}, "reference_spans_sha256": sha256(reference_path),
               "sources_sha256": {**{str(p): h for p, h in sources.hashes.items()}, str(reference_path): sha256(reference_path)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    # 同盘重命名一次发布两份清单及统计。中断时用户可以用原路径重跑。
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "labels"
        stage.mkdir()
        for split in SPLITS:
            path = stage / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    if row["split"] == split:
                        print(dump(row), file=stream)
        summary["output_sha256"] = {p.name: sha256(p) for p in sorted(stage.glob("*.jsonl"))}
        (stage / "summary.json").write_text(dump(summary) + "\n", encoding="utf-8")
        stage.rename(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="含四份冻结清单和原始教师轨迹的 uav_rgb 数据根")
    parser.add_argument("--references", type=Path, default=REFERENCE_SPANS, help="原句和目标的参照短语字符区间")
    parser.add_argument("--out", type=Path, required=True, help="新建输出目录")
    args = parser.parse_args()
    summary = build(args.data_root.resolve(), args.references.resolve(), args.out.absolute())
    print(dump(summary["by_room"]))


if __name__ == "__main__":
    main()
