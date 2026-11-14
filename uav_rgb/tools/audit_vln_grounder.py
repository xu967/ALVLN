#!/usr/bin/env python3
"""审计 SAGE 指令的端点词汇 grounder（诊断用途，不是正式 G1 评测）。

scene.json 没有独立的物体 catalog；本工具用所有路线的 start/end 名称作候选，
因此存在标注泄漏。输出只能回答“词汇启发式暴露了哪些问题”，不能证明地图
grounding 或端到端 VLN 成功。
"""
import argparse
import importlib.util
import json
import os
from collections import Counter


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUNDING_PATH = os.path.join(ROOT, "sim", "3dgs", "grounder.py")
_SPEC = importlib.util.spec_from_file_location("uavrgb_grounder", GROUNDING_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def audit_scene(path, min_score=0.5, min_margin=0.15):
    with open(path, encoding="utf-8") as stream:
        scene = json.load(stream)
    routes = scene.get("routes") or []
    candidates = sorted({str(name) for route in routes
                         for instruction in ((route.get("sage") or {}).get("instructions") or [])
                         for name in (instruction.get("start"), instruction.get("end")) if name})
    grounder = _MODULE.LexicalGrounder(min_score=min_score, min_margin=min_margin)
    total = accepted = hit = 0
    by_type = Counter()
    for route in routes:
        sage = route.get("sage") or {}
        for instruction in sage.get("instructions") or []:
            total += 1
            result = grounder.ground(instruction.get("text", ""), candidates)
            accepted += int(result.accepted)
            hit += int(result.accepted and result.target == instruction.get("end"))
            key = str(instruction.get("instruction_type", "unknown"))
            by_type[key] += int(result.accepted and result.target == instruction.get("end"))
    return {
        "schema": "uavrgb.vln_grounder_endpoint_diagnostic.v1",
        "scene": scene.get("name", os.path.basename(os.path.dirname(path))),
        "scene_path": os.path.abspath(path),
        "candidate_count": len(candidates),
        "instruction_count": total,
        "accepted_count": accepted,
        "accepted_rate": accepted / total if total else None,
        "accepted_target_hits": hit,
        "accepted_target_hit_rate": hit / accepted if accepted else None,
        "target_hit_rate_overall": hit / total if total else None,
        "target_hits_by_type": dict(sorted(by_type.items())),
        "leakage_warning": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    reports = [audit_scene(path) for path in args.scene]
    report = {"schema": "uavrgb.vln_grounder_endpoint_diagnostic.v1", "scenes": reports,
              "instruction_count": sum(item["instruction_count"] for item in reports),
              "accepted_count": sum(item["accepted_count"] for item in reports),
              "accepted_target_hits": sum(item["accepted_target_hits"] for item in reports),
              "leakage_warning": True}
    total = report["instruction_count"]
    accepted = report["accepted_count"]
    report["accepted_rate"] = accepted / total if total else None
    report["target_hit_rate_overall"] = report["accepted_target_hits"] / total if total else None
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
