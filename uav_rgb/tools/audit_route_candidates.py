#!/usr/bin/env python3
"""审计 3DGS 场景的路线候选，不修改场景或候选原始文件。

输入是每个场景旁边的 ``scene.json`` 和 ``routes_check.json``。输出会保留
候选的过滤原因，并单独列出通过过滤的长航线，便于后续用同一候选集做
教师/控制器配对实验。脚本只依赖 Python 标准库。

示例（输出放数据盘）：
  python tools/audit_route_candidates.py \
    --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_839992/scene.json \
    --scene /root/rivermind-data/uav_rgb/3dgs-scenes/wanderland_1MlUucug/scene.json \
    --out /root/rivermind-data/uav_rgb/tmp/route-candidate-audit/report.json
"""
import argparse
import hashlib
import json
import math
import os


VERSION = "uavrgb.route_candidate_audit.v1"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return round(values[0], 3)
    pos = fraction * (len(values) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    value = values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return round(value, 3)


def _number(row, *names):
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def _route_row(row, scene_name, scene_meta, min_path_m):
    """把 routes_check 的一行变成稳定、可读的审计记录。"""
    name = str(row.get("name", ""))
    sage = row.get("sage") or {}
    meta = scene_meta.get(name) or {}
    path_m = _number(row, "sage_path_m")
    planner_path_m = _number(row, "astar_path_m", "path_m")
    accepted = bool(row.get("ok"))
    long_enough = path_m is not None and path_m >= min_path_m
    return {
        "name": name,
        "pair_key": f"{scene_name}/{name}",
        "route_index": meta.get("route_index"),
        "trajectory_id": row.get("trajectory_id", sage.get("trajectory_id")),
        "split": row.get("split", sage.get("split")),
        "sage_path_m": path_m,
        "astar_path_m": planner_path_m,
        "max_steps": row.get("max_steps", meta.get("max_steps")),
        "turn_deg": _number(row, "turn_deg", "turn_total_deg"),
        "min_clearance_m": _number(row, "min_clearance_m"),
        "n_instructions": row.get("n_instructions", sage.get("n_instructions")),
        "agl_model_unverified": bool(row.get("agl_model_unverified", sage.get("agl_model_unverified", False))),
        "ok": accepted,
        "in_scene": name in scene_meta,
        "reasons": list(row.get("reasons") or []),
        "long_candidate": bool(accepted and long_enough and name in scene_meta),
    }


def audit_scene(scene_path, min_path_m=10.0, top_k=5):
    """只读审计一个场景，返回可 JSON 序列化的字典。"""
    scene_path = os.path.abspath(scene_path)
    scene = _read_json(scene_path)
    check_path = os.path.join(os.path.dirname(scene_path), "routes_check.json")
    if not os.path.isfile(check_path):
        raise FileNotFoundError(f"缺少路线检查文件: {check_path}")
    checks = _read_json(check_path)
    if not isinstance(checks, list):
        raise ValueError(f"{check_path}: 顶层必须是数组")
    scene_routes = scene.get("routes") or []
    scene_name = scene.get("name") or os.path.basename(os.path.dirname(scene_path))
    scene_names = [str(route.get("name", "")) for route in scene_routes]
    scene_meta = {
        str(route.get("name", "")): {"route_index": index, "max_steps": route.get("max_steps")}
        for index, route in enumerate(scene_routes)
    }
    errors = []
    if any(not name for name in scene_names):
        errors.append("scene.routes 存在空 name")
    if len(scene_names) != len(set(scene_names)):
        errors.append("scene.routes 存在重复 name")
    check_names = [str(row.get("name", "")) for row in checks]
    if len(check_names) != len(set(check_names)):
        errors.append("routes_check.json 存在重复 name")
    check_set = set(check_names)
    missing = sorted(set(scene_names) - check_set)
    if missing:
        errors.append(f"scene.routes 缺少 routes_check 记录: {missing}")

    accepted_names = {str(row.get("name", "")) for row in checks if row.get("ok")}
    accepted_not_in_scene = sorted(accepted_names - set(scene_names))
    if accepted_not_in_scene:
        errors.append(f"routes_check 通过但未进入 scene.routes: {accepted_not_in_scene}")
    scene_not_accepted = sorted(set(scene_names) - accepted_names)
    if scene_not_accepted:
        errors.append(f"scene.routes 中路线未通过 routes_check: {scene_not_accepted}")

    rows = [_route_row(row, scene_name, scene_meta, min_path_m) for row in checks]
    accepted = [row for row in rows if row["ok"]]
    long_candidates = sorted(
        (row for row in rows if row["long_candidate"]),
        key=lambda row: (-row["sage_path_m"], row["name"]),
    )
    lengths = [row["sage_path_m"] for row in accepted if row["sage_path_m"] is not None]
    reasons = {}
    for row in rows:
        for reason in row["reasons"]:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return {
        "scene": scene_name,
        "scene_path": scene_path,
        "scene_sha256": _sha256(scene_path),
        "routes_check_path": check_path,
        "routes_check_sha256": _sha256(check_path),
        "n_scene_routes": len(scene_names),
        "n_candidates": len(rows),
        "n_accepted": len(accepted),
        "n_accepted_in_scene": sum(row["in_scene"] for row in accepted),
        "n_rejected": len(rows) - len(accepted),
        "n_long_candidates": len(long_candidates),
        "length_summary_m": {
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "max": max(lengths) if lengths else None,
        },
        "filter_reason_counts": dict(sorted(reasons.items())),
        "thresholds": {"min_sage_path_m": float(min_path_m), "top_k": int(top_k)},
        "integrity_errors": errors,
        "long_candidates": long_candidates,
        "top_long_candidates": long_candidates[:top_k],
        "candidates": rows,
    }


def audit_scenes(scene_paths, min_path_m=10.0, top_k=5):
    if not scene_paths:
        raise ValueError("至少需要一个 --scene")
    reports = [audit_scene(path, min_path_m=min_path_m, top_k=top_k) for path in scene_paths]
    integrity_errors = {
        report["scene"]: report["integrity_errors"]
        for report in reports if report["integrity_errors"]
    }
    all_long = [
        {"scene": report["scene"], **row}
        for report in reports for row in report["long_candidates"]
    ]
    all_long.sort(key=lambda row: (-row["sage_path_m"], row["scene"], row["name"]))
    return {
        "schema": VERSION,
        "scenes": reports,
        "summary": {
            "n_scenes": len(reports),
            "n_scene_routes": sum(report["n_scene_routes"] for report in reports),
            "n_candidates": sum(report["n_candidates"] for report in reports),
            "n_accepted": sum(report["n_accepted"] for report in reports),
            "n_long_candidates": sum(report["n_long_candidates"] for report in reports),
            "integrity_errors": integrity_errors,
            "long_candidates": all_long,
            "top_long_candidates": all_long[:top_k],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", required=True, help="scene.json，可重复")
    parser.add_argument("--out", required=True, help="输出 JSON；建议放在数据盘")
    parser.add_argument("--min-path-m", type=float, default=10.0, help="长航线阈值，按 SAGE 路径长度（默认 10 m）")
    parser.add_argument("--top-k", type=int, default=5, help="每个场景最多输出多少条长航线")
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_path_m) or args.min_path_m < 0:
        parser.error("--min-path-m 必须是有限非负数")
    if args.top_k < 1:
        parser.error("--top-k 必须 >= 1")
    report = audit_scenes(args.scene, args.min_path_m, args.top_k)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    summary = report["summary"]
    print(json.dumps({key: summary[key] for key in ("n_scenes", "n_scene_routes", "n_candidates", "n_accepted", "n_long_candidates")}, ensure_ascii=False))
    return 1 if summary["integrity_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
