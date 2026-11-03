#!/usr/bin/env python3
"""把教师软标签按 **UAVR 全局样本序** 拼成一维扁平数组，供 C++ 训练器直接读。

为什么要这一步：`teacher/export_soft_targets.py` 产出的是按 (episode_id, step) 索引的
滑窗软标签；`.uavr` 的样本顺序由 preprocess.py 决定（按 episode 顺序、集内按 step 升序）。
C++ 侧不该解析 npz，也不该自己重建这个映射——所以在这里拼平。

输出：
  <out.bin>    UAVSOFT1 120 字节身份头 + 小端 float32 [n * 4] 概率
  <out.json>   覆盖率、温度、UAVR/侧车/payload SHA256，供人工追溯

用法：
  python3 tools/join_soft_targets.py \
      --uavr-dir $DATA_ROOT/uav_rgb/ds-teacher400 \
      --episodes-dir $DATA_ROOT/uav_rgb/teacher-400-20261102 \
      --soft $DATA_ROOT/uav_rgb/teacher-soft.npz \
      --out $DATA_ROOT/uav_rgb/ds-teacher400/soft_targets.bin
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile

import numpy as np

NUM_CLASSES = 4
HEADER = struct.Struct("<8sQII32s32s32s")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").digest()


def check_spans(rows, dataset):
    """侧车顺序必须与 UAVR 内嵌的 episode/sample 区间完全一致。"""
    with open(dataset, "rb") as f:
        header = f.read(48)
        if len(header) != 48 or header[:8] != b"UAVR\x01\x00\x30\x00":
            die("dataset.uavr 文件头非法")
        n = struct.unpack_from("<I", header, 8)[0]
        meta_offset, meta_len = struct.unpack_from("<QI", header, 36)
        if os.fstat(f.fileno()).st_size != meta_offset + meta_len:
            die("dataset.uavr 长度与头不符")
        f.seek(meta_offset)
        meta = json.loads(f.read(meta_len))
    spans = sorted(meta["episodes"], key=lambda r: int(r["sample_start"]))
    if len(rows) != len(spans) or not n:
        die("instructions 与 UAVR episode 数量不符或数据为空")
    cursor = 0
    for row, span in zip(rows, spans):
        start, count = int(row["sample_start"]), int(row["sample_count"])
        if (start != cursor or count <= 0 or start != int(span["sample_start"])
                or count != int(span["sample_count"])
                or str(row["episode_id"]) != str(span["episode_id"])):
            die("instructions 样本顺序/episode 与 UAVR 不匹配")
        cursor += count
    if cursor != n:
        die("instructions 样本区间未完整覆盖 UAVR")
    return n


def die(msg):
    print(f"错误: {msg}", file=sys.stderr)
    sys.exit(1)


def episode_steps(ep_dir):
    """返回 (全部 step 升序, 非碰撞 step 升序)；与 preprocess.py 的排序/过滤口径一致。"""
    path = os.path.join(ep_dir, "actions.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    steps = sorted(int(r["step"]) for r in rows)
    if len(set(steps)) != len(steps):
        die(f"{path} 含重复 step")
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

    protected = [os.path.join(args.uavr_dir, name) for name in
                 ("dataset.uavr", "instructions.jsonl")] + args.soft
    meta_path = os.path.splitext(args.out)[0] + ".json"
    protected_paths = {os.path.realpath(path) for path in protected}
    if (os.path.realpath(args.out) in protected_paths
            or os.path.realpath(meta_path) in protected_paths | {os.path.realpath(args.out)}):
        die("输出不能覆盖输入数据或教师标签")

    sidecar = os.path.join(args.uavr_dir, "instructions.jsonl")
    if not os.path.isfile(sidecar):
        die(f"找不到 {sidecar}")
    dataset = os.path.join(args.uavr_dir, "dataset.uavr")
    data_hash, text_hash = file_sha256(dataset), file_sha256(sidecar)
    rows = [json.loads(line) for line in open(sidecar, encoding="utf-8") if line.strip()]
    rows.sort(key=lambda r: int(r["sample_start"]))
    n = check_spans(rows, dataset)

    soft = {}
    seq_len = None
    for path in args.soft:
        with np.load(path, allow_pickle=False) as z:
            if "seq_len" in z:
                if seq_len is not None and seq_len != int(z["seq_len"]):
                    die("软标签 seq_len 不一致")
                seq_len = int(z["seq_len"])
            for key in z.files:
                if key.endswith("/logits"):
                    if key in soft:
                        die(f"软标签重复键: {key}")
                    values = z[key]
                    if values.shape != (NUM_CLASSES,) or not np.isfinite(values).all():
                        die(f"软标签 {key} 必须是 4 个有限 logits")
                    soft[key] = values
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

    if missing:
        die(f"缺失 {missing}/{n} 条教师 logits，拒绝生成软标签；请导出全部样本")
    # 缺失项不能替换为均匀分布：alpha>0 时这会改变硬标签目标。
    t = float(args.temperature)
    if not np.isfinite(t) or not t > 0:
        die(f"--temperature 必须为正（实际 {t}）")
    z = out.astype(np.float64) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out = (e / e.sum(axis=1, keepdims=True)).astype("<f4")
    if not np.isfinite(out).all():
        die("温度导致概率非有限，请使用可表示的温度")

    frac = float(filled.mean())
    print(f"拼接 {n} 条样本，覆盖 {filled.sum()} 条（{frac:.1%}），缺 {missing} 条，"
          f"温度 T={t}", file=sys.stderr)
    if file_sha256(dataset) != data_hash or file_sha256(sidecar) != text_hash:
        die("拼接期间 UAVR 或文本侧车被改写，拒绝输出")
    payload = out.tobytes()
    payload_hash = hashlib.sha256(payload).digest()
    header = HEADER.pack(b"UAVSOFT1", n, NUM_CLASSES, 1, data_hash, text_hash, payload_hash)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".soft_targets.", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(header)
            f.write(payload)
        os.replace(tmp, args.out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    meta = {"schema": "uavrgb.soft_targets.v1", "n": int(n), "num_classes": NUM_CLASSES,
            "filled": int(filled.sum()), "missing": int(missing),
            "seq_len": seq_len, "temperature": t,
            "uavr_sha256": data_hash.hex(), "instructions_sha256": text_hash.hex(),
            "payload_sha256": payload_hash.hex()}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"写出 {args.out}（{HEADER.size + out.nbytes} 字节）", file=sys.stderr)


if __name__ == "__main__":
    main()
