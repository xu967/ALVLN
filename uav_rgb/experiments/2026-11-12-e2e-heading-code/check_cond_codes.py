#!/usr/bin/env python3
"""航向码数据集的独立核对（卡 E2E-1 第 3 步，打包阶段）。

对象：tools/preprocess.py --cond heading-code 打出的 dataset.uavr + dataset.aux。
做法：不经过 preprocess 的累计逻辑，直接读每集原始的 actions.jsonl（meta.episodes[].source_dir），
按 nav_cond.heading_codes(已执行动作) 重算整集每一步的码，取 --drop-collision 之后保留下来的那些步，
与侧车 prev_action 列在该集样本区间内逐样本比对。碰撞步虽被丢，仍计入累计（它执行的动作是真实发生过的）。
保留步的判定 = 步的 collision 不为真（与 preprocess 的 --drop-collision 同义）；若 meta 的 collision_filter 没有开
drop_collision_steps，则全部步保留。每集保留步数与 meta 的 sample_count 不相等时记「映射失败」，该集不比码，也算不通过。

退出码：0 全部集一致；1 有码不一致或映射失败；2 输入读不了。
末尾 RESULT 行给出集数、样本数、非零码样本数、码直方图、不一致与映射失败的集数；--json 另存明细。
用法：check_cond_codes.py DATASET.uavr [--aux DATASET.aux] [--json OUT.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim", "tools"))
import compare_packs as cp  # noqa: E402
import nav_cond  # noqa: E402


def kept_steps(rows, drop_collision):
    """一集全部步里被打包保留的步下标。"""
    if not drop_collision:
        return list(range(len(rows)))
    return [i for i, r in enumerate(rows) if not r.get("collision")]


def expected_codes(rows, drop_collision):
    """-> (保留步下标, 这些步的期望码)。码按全部步已执行动作累计（含被丢的碰撞步）。"""
    codes = nav_cond.heading_codes([r["action_executed"] for r in rows])
    kept = kept_steps(rows, drop_collision)
    return kept, [codes[i] for i in kept]


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(data, aux):
    """-> dict：episodes / samples / nonzero / hist / mismatch_episodes（id 列表）/ mapping_failures（id 列表）。"""
    head = cp.read_head(data)
    meta = head["meta"]
    n = head["n"]
    _, column = cp.read_aux_cols(aux, n)
    drop = bool((meta.get("collision_filter") or {}).get("drop_collision_steps"))
    res = {"episodes": 0, "samples": 0, "nonzero": 0, "hist": [0, 0, 0, 0],
           "mismatch_episodes": [], "mapping_failures": [], "drop_collision": drop, "first_mismatch": None}
    for ep in meta["episodes"]:
        res["episodes"] += 1
        start, count = ep["sample_start"], ep["sample_count"]
        rows = read_rows(os.path.join(ep["source_dir"], "actions.jsonl"))
        kept, want = expected_codes(rows, drop)
        if len(kept) != count:
            res["mapping_failures"].append(ep["episode_id"])
            continue
        got = column[start:start + count]
        want = np.asarray(want, dtype=np.uint8)
        res["samples"] += count
        res["nonzero"] += int((want != 0).sum())
        for code in range(4):
            res["hist"][code] += int((want == code).sum())
        if not np.array_equal(got, want):
            res["mismatch_episodes"].append(ep["episode_id"])
            if res["first_mismatch"] is None:
                k = int(np.flatnonzero(got != want)[0])
                res["first_mismatch"] = {"episode_id": ep["episode_id"], "sample_in_episode": k,
                                         "dataset_code": int(got[k]), "recomputed_code": int(want[k])}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--aux", help="侧车路径；缺省取数据集同目录的 dataset.aux")
    ap.add_argument("--json", help="明细另存到这个路径")
    args = ap.parse_args(argv)
    aux = args.aux or os.path.join(os.path.dirname(os.path.abspath(args.data)), "dataset.aux")
    try:
        res = check(args.data, aux)
    except (OSError, ValueError, KeyError) as exc:
        print(f"RESULT ERROR 读不了输入：{exc}")
        return 2
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
            f.write("\n")
    ok = not res["mismatch_episodes"] and not res["mapping_failures"]
    if res["first_mismatch"]:
        print("首个不一致：%s" % json.dumps(res["first_mismatch"], ensure_ascii=False))
    print("RESULT %s episodes=%d samples=%d nonzero_code_samples=%d code_hist=%s mismatch_episodes=%d mapping_failures=%d drop_collision=%s"
          % ("OK" if ok else "FAIL", res["episodes"], res["samples"], res["nonzero"], res["hist"],
             len(res["mismatch_episodes"]), len(res["mapping_failures"]), res["drop_collision"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
