#!/usr/bin/env python3
"""审计 OpenFly Annotation 文件，不下载数据、不修改输入文件。

示例：
  python tools/audit_openfly_annotations.py \
      --input /root/rivermind-data/uav_rgb/openfly/Annotation \
      --output /root/rivermind-data/uav_rgb/tmp/openfly-annotation-audit.json

OpenFly 的标注文件可能是 JSON 数组、单条 JSON 对象或 JSONL；这里统一按记录统计，
只依赖 Python 标准库。输出的统计结果可以用于决定是否值得下载图像/轨迹，而不是官方
OpenFly 评测分数。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SUFFIXES = {".json", ".jsonl"}
REQUIRED_LIST_FIELDS = ("action", "index_list", "pos", "yaw")


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in DEFAULT_SUFFIXES)


def _records(value: Any, source: str) -> Iterable[dict[str, Any]]:
    """把常见的 JSON 容器变成记录；未知容器明确报错，避免静默漏统计。"""
    if isinstance(value, dict):
        # 训练标注通常是数组；单条对象也允许。少数导出器把记录包在 data/records 下。
        for key in ("data", "records", "annotations", "items"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            yield value
            return
    if not isinstance(value, list):
        raise ValueError(f"{source}: 顶层必须是 JSON object、array 或 JSONL")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{source}: 记录必须是 JSON object，得到 {type(item).__name__}")
        yield item


def _load(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                yield from _records(value, f"{path}:{line_no}")
        return
    with path.open(encoding="utf-8") as f:
        yield from _records(json.load(f), str(path))


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _path_length(pos: Any) -> float | None:
    if not isinstance(pos, list) or len(pos) < 2:
        return 0.0 if isinstance(pos, list) else None
    points = []
    for p in pos:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        xy = [_number(p[0]), _number(p[1])]
        if None in xy:
            return None
        points.append(xy)
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def _split(path: Path, record: dict[str, Any]) -> str:
    for key in ("split", "dataset_split", "partition"):
        if isinstance(record.get(key), str) and record[key]:
            return record[key]
    text = "/".join(path.parts).lower()
    for name in ("unseen", "seen", "train", "val", "test"):
        if name in text:
            return name
    return "unknown"


def audit(input_path: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(input_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    files = _files(root)
    if not files:
        raise ValueError(f"没有找到 JSON/JSONL 文件: {root}")

    actions: collections.Counter[str] = collections.Counter()
    splits: collections.Counter[str] = collections.Counter()
    field_counts: collections.Counter[str] = collections.Counter()
    length_mismatches: collections.Counter[str] = collections.Counter()
    lengths: list[float] = []
    instruction_lengths: list[int] = []
    errors: list[dict[str, str]] = []
    records = valid = 0
    for path in files:
        try:
            iterator = _load(path)
            for record in iterator:
                records += 1
                for key in record:
                    field_counts[key] += 1
                instruction = record.get("gpt_instruction")
                if isinstance(instruction, str):
                    instruction_lengths.append(len(instruction.split()))
                action = record.get("action")
                if isinstance(action, list):
                    actions.update(str(x) for x in action)
                lists = [record.get(k) for k in REQUIRED_LIST_FIELDS]
                present = [v for v in lists if isinstance(v, list)]
                if present and any(len(v) != len(present[0]) for v in present[1:]):
                    length_mismatches["action/index_list/pos/yaw"] += 1
                if isinstance(action, list) and isinstance(record.get("pos"), list):
                    valid += 1
                    distance = _path_length(record["pos"])
                    if distance is not None:
                        lengths.append(distance)
                    splits[_split(path, record)] += 1
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append({"file": str(path), "error": str(exc)})

    def summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"n": 0, "min": None, "p50": None, "max": None}
        values = sorted(values)
        return {"n": len(values), "min": round(values[0], 4),
                "p50": round(values[(len(values) - 1) // 2], 4), "max": round(values[-1], 4)}

    return {
        "schema": "uavrgb.openfly_annotation_audit.v1",
        "input": str(root),
        "files": len(files),
        "records": records,
        "records_with_action_and_pos": valid,
        "parse_errors": errors,
        "split_counts": dict(sorted(splits.items())),
        "action_counts": dict(sorted(actions.items(), key=lambda item: (int(item[0]) if item[0].lstrip("-").isdigit() else 10**9, item[0]))),
        "instruction_word_count": summary([float(x) for x in instruction_lengths]),
        "trajectory_length_m": summary(lengths),
        "field_counts": dict(sorted(field_counts.items())),
        "length_mismatch_records": dict(length_mismatches),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Annotation 文件或目录")
    parser.add_argument("--output", required=True, help="输出 JSON 路径；建议放在 /root/rivermind-data")
    args = parser.parse_args()
    report = audit(args.input)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["parse_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
