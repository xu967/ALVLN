#!/usr/bin/env python3
"""训练配方核对（卡 E2E-1 第 3 步，训练之后）：新训的 run 的 config.json 与 A0 的训练配置逐键比较（A0 = e1-noise/seed0/run/config.json）。
卡要求「训练配方与 A0 相同（30 轮、batch 128、lr 0.001、类权重 auto、--drop-collision），加 --save-every-epoch」；这里把「相同」落成逐键比较，不靠眼看。

允许不同的键（每个都有来由，其余键必须全同，浮点按相对 1e-6）：
  data、dataset_sha256   数据集不同；data 必须是绝对路径（评测端一致性闸门靠 config.data 找数据集，相对路径从别的目录找不到）。
  seed                   必须等于该 run 声明的训练种子。
  source_git             UAVRGB_SOURCE_GIT 记的冻结代码提交。
  prev_action            航向码臂必须为 true，图像臂必须为 false。
  class_weights          auto 按训练集重算：同一份 demo 数据（A1 对 A0）必须与参照相同；dagger 数据集（A2、A3）只记录。
class_weights_source 必须仍是 auto。
用法：check_recipe.py --ref REF_CONFIG.json NAME=RUN_DIR:kind:seed:dataset ...    kind = code|img；dataset = demo|dagger
退出码：0 全部通过；1 有不一致；2 读不了输入。末尾 RESULT 行。
"""
import argparse
import json
import math
import os
import sys

ALLOWED = {"data", "dataset_sha256", "seed", "source_git", "prev_action", "class_weights"}


def parse_weights(v):
    """config.json 里 class_weights 是字符串 '[a, b, c, d]'，也可能已是列表。"""
    if isinstance(v, str):
        v = json.loads(v)
    return [float(x) for x in v]


def close(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return a == b
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=0.0)


def diff_config(ref, cfg, kind, seed, dataset):
    """-> 问题列表（空 = 通过）。"""
    problems = []
    for k in sorted(set(ref) | set(cfg)):
        if k in ALLOWED:
            continue
        if k not in cfg or k not in ref:
            problems.append(f"键 {k} 只在{'参照' if k in ref else '新 run'}里")
        elif not close(ref[k], cfg[k]):
            problems.append(f"键 {k}：参照 {ref[k]!r} 新 run {cfg[k]!r}")
    if cfg.get("seed") != seed:
        problems.append(f"seed 应为 {seed}，config 里是 {cfg.get('seed')!r}")
    want_prev = kind == "code"
    if cfg.get("prev_action") is not want_prev:
        problems.append(f"prev_action 应为 {want_prev}，config 里是 {cfg.get('prev_action')!r}")
    data = cfg.get("data")
    if not isinstance(data, str) or not os.path.isabs(data):
        problems.append(f"data 必须是绝对路径，config 里是 {data!r}")
    if dataset == "demo":
        try:
            if parse_weights(ref["class_weights"]) != parse_weights(cfg["class_weights"]):
                problems.append(f"class_weights：同一份 demo 数据应与参照相同，参照 {ref['class_weights']} 新 run {cfg['class_weights']}")
        except (KeyError, ValueError, TypeError) as exc:
            problems.append(f"class_weights 读不了：{exc}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("runs", nargs="+", help="NAME=RUN_DIR:kind:seed:dataset")
    args = ap.parse_args(argv)
    try:
        with open(args.ref, encoding="utf-8") as f:
            ref = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"RESULT ERROR 读不了参照 {args.ref}：{exc}")
        return 2
    bad = 0
    for spec in args.runs:
        try:
            name, rest = spec.split("=", 1)
            run_dir, kind, seed, dataset = rest.rsplit(":", 3)
            with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            problems = diff_config(ref, cfg, kind, int(seed), dataset)
        except (OSError, ValueError) as exc:
            print(f"RESULT ERROR 读不了 {spec}：{exc}")
            return 2
        status = "通过" if not problems else "不一致"
        print(f"{status}  {name}（{kind}/{dataset}，seed {seed}）：class_weights {cfg.get('class_weights')}，source_git {str(cfg.get('source_git'))[:12]}，data {cfg.get('data')}")
        for p in problems:
            print(f"    {p}")
        bad += bool(problems)
    print(f"RESULT {'OK' if not bad else 'FAIL'} 核对 {len(args.runs)} 个 run，不一致 {bad} 个（参照 {args.ref}）")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
