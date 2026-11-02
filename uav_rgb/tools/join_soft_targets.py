#!/usr/bin/env python3
"""把教师软标签按 **UAVR 全局样本序** 拼成一维扁平数组，供 C++ 训练器直接读。

为什么要这一步：`teacher/export_soft_targets.py` 产出的是按 (episode_id, step) 索引的
滑窗软标签；`.uavr` 的样本顺序由 preprocess.py 决定（按 episode 顺序、集内按 step 升序）。
C++ 侧不该解析 npz，也不该自己重建这个映射——所以在这里拼平。

输出：
  <out.bin>    float32 [n * 4]，第 k 行是第 k 个 uavr 样本的四类 logits
  <out.json>   {n, num_classes, t, episodes, missing} 供训练器校验

用法：
  python3 tools/join_soft_targets.py \
      --uavr-dir $DATA_ROOT/uav_rgb/ds-teacher400 \
      --episodes-dir $DATA_ROOT/uav_rgb/teacher-400-20261102 \
      --soft $DATA_ROOT/uav_rgb/teacher-soft.npz \
      --out $DATA_ROOT/uav_rgb/ds-teacher400/soft_targets.bin
"""
import argparse
import json
import os
import sys

import numpy as np

NUM_CLASSES = 4


def die(msg):
    print(f"错误: {msg}", file=sys.stderr)
    sys.exit(1)


def episode_steps(ep_dir):
    """返回 (全部 step 升序, 非碰撞 step 升序)；与 preprocess.py 的排序/过滤口径一致。"""
    path = os.path.join(ep_dir, "actions.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    steps = sorted(int(r["step"]) for r in rows)
    clean = sorted(int(r["step"]) for r in rows if not r.get("collision"))
    return steps, clean


def find_episode(episodes_dir, episode_id):
    """collect 把 episode 放在 <root>/<split>/<id>/；这里在 train/val/test 下按 id 找。"""
    for split in ("train", "val", "test"):
        d = os.path.join(episodes_dir, split, str(episode_id))
        if os.path.isdir(d):
            return d
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uavr-dir", required=True, help="含 instructions.jsonl 的目录")
    p.add_argument("--episodes-dir", required=True,
                   help="采集根目录（<root>/<split>/<episode_id>/actions.jsonl）")
    p.add_argument("--soft", required=True, nargs="+",
                   help="一个或多个 export_soft_targets.py 产出的 .npz（train/val 分开导时可给多个）")
    p.add_argument("--out", required=True, help="输出 .bin（同名 .json 写元信息）")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="教师 logits 除以 T 再 softmax（T=1 = 教师原始分布；C++ 端要求每行和为 1）")
    args = p.parse_args()

    sidecar = os.path.join(args.uavr_dir, "instructions.jsonl")
    if not os.path.isfile(sidecar):
        die(f"找不到 {sidecar}")
    rows = [json.loads(line) for line in open(sidecar, encoding="utf-8") if line.strip()]
    rows.sort(key=lambda r: int(r["sample_start"]))
    n = sum(int(r["sample_count"]) for r in rows)
    if rows and int(rows[-1]["sample_start"]) + int(rows[-1]["sample_count"]) != n:
        die("sidecar 的 sample_start/sample_count 不连续，无法拼成扁平序")

    soft = {}
    seq_len = None
    for path in args.soft:
        z = np.load(path)
        if seq_len is None and "seq_len" in z:
            seq_len = int(z["seq_len"])
        for key in z.files:
            if key.endswith("/logits"):
                soft[key] = z[key]
    out = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    filled = np.zeros(n, dtype=bool)
    missing = 0

    for r in rows:
        eid = str(r["episode_id"])
        start, count = int(r["sample_start"]), int(r["sample_count"])
        ep = find_episode(args.episodes_dir, eid)
        if ep is None:
            die(f"episode {eid} 在 {args.episodes_dir} 下找不到")
        all_steps, clean_steps = episode_steps(ep)
        # .uavr 可能用也可能没用 --drop-collision；两种都支持，用长度判定。
        if len(all_steps) == count:
            steps = all_steps
        elif len(clean_steps) == count:
            steps = clean_steps
        else:
            die(f"episode {eid}: uavr 存了 {count} 步，但 actions.jsonl 有 "
                f"{len(all_steps)} 步（非碰撞 {len(clean_steps)}），对不上")
        for k, step in enumerate(steps):
            key = f"{eid}/{step}/logits"
            if key not in soft:  # type: ignore[operator]
                missing += 1
                continue
            out[start + k] = soft[key]
            filled[start + k] = True

    # 教师导出的是裸 logits，C++ 端要求每行是概率分布（和为 1），这里按温度做 softmax。
    # 缺失的样本保持全 0 -> softmax 后是均匀分布（不提供任何信息，等价于只学硬标签）。
    t = float(args.temperature)
    if not t > 0:
        die(f"--temperature 必须为正（实际 {t}）")
    z = out / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    frac = float(filled.mean())
    print(f"拼接 {n} 条样本，覆盖 {filled.sum()} 条（{frac:.1%}），缺 {missing} 条，"
          f"温度 T={t}", file=sys.stderr)
    if n and filled.sum() == 0:
        die("一条都没匹配上；检查 --episodes-dir 的 episode_id 是否与软标签一致")
    if missing:
        print("警告: 缺失的样本软标签为均匀分布，"
              "训练前请确认 dataset 与软标签来自同一次采集", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.tofile(args.out)
    meta = {"n": int(n), "num_classes": NUM_CLASSES,
            "filled": int(filled.sum()), "missing": int(missing),
            "seq_len": seq_len, "temperature": t}
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"写出 {args.out}（{out.nbytes} 字节）", file=sys.stderr)


if __name__ == "__main__":
    main()
