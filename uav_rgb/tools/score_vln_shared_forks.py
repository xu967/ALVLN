"""用已经发表的四组固定对照评估新采集的共同分岔，每个起点的主样本只计一次。"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


PROTOCOL = Path(__file__).with_name("data") / "vln_shared_forks_protocol.json"
OLD_REGISTRATION_SHA = "14628285d47943508347ab1bb2b771daafe7d481f7fefec34bac059a6be5d709"
MODES = ("region_text", "region_blank", "global_text", "always_center")
CLASSES = ("C", "L", "R")


def checked(path, expected):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"文件与登记指纹不符 {path}")
    return path


def frozen_encoder(source):
    spec = importlib.util.spec_from_file_location("published_fork_region", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score_group(requested, observations):
    primary = [row for row in observations if row["primary"]]
    observed_sites = {row["physical_site"] for row in primary}
    if len(observed_sites) != len(primary):
        raise ValueError("主样本重复使用同一物理出生点")
    metrics = {}
    for mode in MODES:
        exact = sum(row["truth"] == row["predicted"][mode] for row in primary)
        all_exact = sum(row["truth"] == row["predicted"][mode] for row in observations)
        metrics[mode] = dict(primary_both_correct=exact, all_flown_both_correct=all_exact,
                             primary_pair_accuracy=exact / len(primary) if primary else None)
    return dict(requested_pairs=len(requested), pairs_with_flown_observation=len({r["pair"] for r in observations}),
                flown_frames=len(observations), primary_sites=len(primary), scores=metrics)


def evaluate(design_path, observations_path, old_run, source, out):
    import numpy as np
    import torch

    if out.exists():
        raise FileExistsError(f"已存在结果 {out}")
    protocol = json.loads(PROTOCOL.read_text())["prediction"]
    prior = json.loads(checked(old_run / "registration.json", OLD_REGISTRATION_SHA).read_text())
    fit_path = checked(old_run / "linear_models.json", protocol["fit_sha256"])
    config = prior["config"]
    for name, digest in prior["inputs"]["model_files_sha256"].items():
        checked(Path(config["checkpoint"]) / name, digest)
    model = frozen_encoder(checked(source / "tools/fork_region.py", protocol["feature_module_sha256"]))
    fits = json.loads(fit_path.read_text())["fits"]
    verified = json.loads(observations_path.with_name("summary.json").read_text())
    checked(observations_path, verified["observations_sha256"])
    checked(design_path, verified["source_sha256"][str(design_path.resolve())])
    pairs = json.loads(design_path.read_text())["pairs"]
    by_name = {p["pair"]: p for p in pairs}
    if len(by_name) != len(pairs):
        raise ValueError("注册的路线配对名称重复")
    rows = [json.loads(line) for line in observations_path.read_text().splitlines()]
    frames, records = [], []
    for row in rows:
        pair = by_name[row["pair"]]
        labels = row["labels"]
        if (not row["verified_shared_prefix"] or row["step"] < 1 or len(labels) != 2
                or any(row[k] != pair[k] for k in ("scene", "split", "selection_group", "heading_source"))
                or [a["instruction"] for a in labels] != [a["instruction"] for a in pair["arms"]]):
            raise ValueError(f"观测与冻结的共同前缀不符 {row['pair']} {row['step']}")
        xyz = tuple(round(float(pair["spawn"][axis]), 5) for axis in "xyz")
        current = model.Frame(row["scene"], row["pair"], str(xyz), 0, row["step"], True, Path(row["rgb"]),
                              row["rgb_sha256"], tuple(a["instruction"] for a in labels),
                              tuple(a["local_bearing_bin"] for a in labels), tuple(a["planning_status"] for a in labels),
                              tuple(a.get("inside_actual_image") for a in labels))
        truth = [CLASSES[c] if c >= 0 else None for c in current.classes]
        primary = row["primary"]
        if type(primary) is not bool or (primary and (None in truth or truth[0] == truth[1] or not all(current.inside))):
            raise ValueError(f"入选主样本的标记与方位不符 {row['pair']} {row['step']}")
        frames.append(current)
        records.append(dict(pair=row["pair"], scene=row["scene"], split=row["split"], physical_site=(row["scene"], *xyz),
                            selection_group=pair["selection_group"], heading_source=pair["heading_source"], step=row["step"],
                            rgb_sha256=row["rgb_sha256"], truth=truth, primary=primary, predicted={}))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if frames:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        features = model.extract_features(frames, model.FrozenClip(config, device), config["batch_size"])
        predictions = {m: model.predict(fits[m], features[m]).reshape(-1, 2) for m in model.MODELS}
        predictions["always_center"] = np.zeros((len(records), 2), dtype=int)
        for i, row in enumerate(records):
            row["predicted"] = {name: [CLASSES[int(v)] for v in values[i]] for name, values in predictions.items()}

    groups = {}
    for key in ("split", "scene", "selection_group", "heading_source"):
        groups[key] = {value: score_group([p for p in pairs if p[key] == value], [r for r in records if r[key] == value])
                       for value in sorted({p[key] for p in pairs})}
    for split, group in groups["split"].items():
        group["by_selection_group"] = {
            kind: score_group([p for p in pairs if (p["split"], p["selection_group"]) == (split, kind)],
                              [r for r in records if (r["split"], r["selection_group"]) == (split, kind)])
            for kind in groups["selection_group"]
        }
    summary = dict(schema="uavrgb.vln.flown_fork_transfer.v1", device=device, all=score_group(pairs, records),
                   by=groups, primary_by_room=dict(Counter(row["scene"] for row in records if row["primary"])),
                   inputs_sha256={str(p): model.sha(p) for p in (design_path, observations_path, fit_path, PROTOCOL)},
                   encoder_sha256=protocol["feature_module_sha256"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as working:
        stage = Path(working) / "scores"
        stage.mkdir()
        with (stage / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in records:
                print(json.dumps(row, ensure_ascii=False, allow_nan=False), file=stream)
        summary["predictions_sha256"] = model.sha(stage / "predictions.jsonl")
        (stage / "summary.json").write_text(model.json_text(summary), encoding="utf-8")
        stage.rename(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("design", "observations", "old-run", "source", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.design, args.observations, args.old_run, args.source, args.out)
    print(json.dumps(result["by"]["split"], ensure_ascii=False))


if __name__ == "__main__":
    main()
