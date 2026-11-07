#!/usr/bin/env python3
"""把教师软标签按 **UAVR 全局样本序** 拼成一维扁平数组，供 C++ 训练器直接读。

为什么要这一步：`teacher/export_soft_targets.py`（滑窗教师）和 `sim/tools/teacher_belief.py`
（信念边缘化教师）产出的都是按 (episode_id, step) 索引的软标签；`.uavr` 的样本顺序由
preprocess.py 决定（按 episode 顺序、集内按 step 升序）。C++ 侧不该解析 npz，也不该自己
重建这个映射——所以在这里拼平。

输出：
  <out.bin>    UAVSOFT1 120 字节身份头 + 小端 float32 [n * 4] 概率
  <out.json>   覆盖率、温度、UAVR/侧车/payload SHA256、教师来源，供人工追溯

episode 区间：有 instructions.jsonl 侧车（文本任务）时以侧车为准并与 UAVR 内嵌区间核对
（flags=1）；没有侧车（纯视觉数据，3DGS pilot / DAgger）时直接用 UAVR 元数据里的 episodes
区间（flags=0，侧车哈希全 0；C++ 读端要求目录里确实没有侧车）。
episode 目录：先按 --episodes-dir/<split>/<id>/ 找（Gazebo collect 布局），找不到再用 UAVR
元数据记录的 source_dir（3DGS 采集按 route_N/ep_xxxxx 放，没有 split 层）。

用法：
  python3 tools/join_soft_targets.py \
      --uavr-dir $DATA_ROOT/uav_rgb/ds-teacher400 \
      --episodes-dir $DATA_ROOT/uav_rgb/teacher-400-20261102 \
      --soft $DATA_ROOT/uav_rgb/teacher-soft.npz \
      --out $DATA_ROOT/uav_rgb/ds-teacher400/soft_targets.bin
  信念教师（logits = log p，温度必须给 1.0 才还原 p）：
  python3 tools/join_soft_targets.py --uavr-dir <ds> --soft belief.npz --temperature 1.0 --out <ds>/soft_targets.bin
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
# 软标签 npz 里的教师来源键（export_soft_targets.py / teacher_belief.py 写入）；多份 npz 必须一致。
PROVENANCE_KEYS = ("teacher_ckpt_sha256", "select_split", "train_data_sha256",
                   "teacher_version", "k", "alpha_steps", "aggregate", "seed")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").digest()


def die(msg):
    print(f"错误: {msg}", file=sys.stderr)
    sys.exit(1)


def read_uavr(dataset):
    """返回 (样本数 n, 按 sample_start 排好的 episodes 区间)。"""
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
    if not n or not spans:
        die("UAVR 为空或没有 episodes 元数据")
    return n, spans


def check_spans(rows, spans, n):
    """侧车顺序必须与 UAVR 内嵌的 episode/sample 区间完全一致。"""
    if len(rows) != len(spans):
        die("instructions 与 UAVR episode 数量不符")
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


def episode_steps(ep_dir):
    """返回 (全部 step 升序, 非碰撞 step 升序)；与 preprocess.py 的排序/过滤口径一致。"""
    path = os.path.join(ep_dir, "actions.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    steps = sorted(int(r["step"]) for r in rows)
    if len(set(steps)) != len(steps):
        die(f"{path} 含重复 step")
    clean = sorted(int(r["step"]) for r in rows if not r.get("collision"))
    return steps, clean


def find_episode(episodes_dir, episode_id, source_dir=None):
    """collect 把 episode 放在 <root>/<split>/<id>/；找不到再用 UAVR 记录的 source_dir。"""
    if episodes_dir:
        for split in ("train", "val", "test"):
            d = os.path.join(episodes_dir, split, str(episode_id))
            if os.path.isdir(d):
                return d
    if source_dir and os.path.isfile(os.path.join(source_dir, "actions.jsonl")):
        return source_dir
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uavr-dir", required=True, help="含 dataset.uavr（可选 instructions.jsonl）的目录")
    p.add_argument("--episodes-dir", default=None,
                   help="采集根目录（<root>/<split>/<episode_id>/actions.jsonl）；不给则用 UAVR 记录的 source_dir")
    p.add_argument("--soft", required=True, nargs="+",
                   help="一个或多个 export_soft_targets.py / teacher_belief.py 产出的 .npz")
    p.add_argument("--out", required=True, help="输出 .bin（同名 .json 写元信息）")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="教师 logits 除以 T 再 softmax（T=1 = 教师原始分布；C++ 端要求每行和为 1）")
    args = p.parse_args()

    dataset = os.path.join(args.uavr_dir, "dataset.uavr")
    sidecar = os.path.join(args.uavr_dir, "instructions.jsonl")
    has_sidecar = os.path.isfile(sidecar)
    protected = [dataset] + ([sidecar] if has_sidecar else []) + args.soft
    meta_path = os.path.splitext(args.out)[0] + ".json"
    protected_paths = {os.path.realpath(path) for path in protected}
    if (os.path.realpath(args.out) in protected_paths
            or os.path.realpath(meta_path) in protected_paths | {os.path.realpath(args.out)}):
        die("输出不能覆盖输入数据或教师标签")

    n, spans = read_uavr(dataset)
    data_hash = file_sha256(dataset)
    if has_sidecar:
        text_hash = file_sha256(sidecar)
        rows = [json.loads(line) for line in open(sidecar, encoding="utf-8") if line.strip()]
        rows.sort(key=lambda r: int(r["sample_start"]))
        check_spans(rows, spans, n)
        flags = 1
    else:
        text_hash, rows, flags = bytes(32), spans, 0
        if sum(int(r["sample_count"]) for r in rows) != n:
            die("UAVR episodes 区间未完整覆盖样本数")

    soft = {}
    seq_len = None
    provenance = {}
    for path in args.soft:
        with np.load(path, allow_pickle=False) as z:
            if "seq_len" in z:
                if seq_len is not None and seq_len != int(z["seq_len"]):
                    die("软标签 seq_len 不一致")
                seq_len = int(z["seq_len"])
            for key in PROVENANCE_KEYS:
                if key not in z.files:
                    continue
                value = str(z[key].item())
                if key in provenance and provenance[key] != value:
                    die(f"教师来源元信息不一致: {key}（{provenance[key]} vs {value}）")
                provenance[key] = value
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
        ep = find_episode(args.episodes_dir, eid, r.get("source_dir"))
        if ep is None:
            die(f"episode {eid} 在 {args.episodes_dir or '(未给 --episodes-dir)'} 与 source_dir "
                f"{r.get('source_dir')} 下都找不到")
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
          f"温度 T={t}，侧车 {'有' if has_sidecar else '无（flags=0）'}", file=sys.stderr)
    if file_sha256(dataset) != data_hash or (has_sidecar and file_sha256(sidecar) != text_hash):
        die("拼接期间 UAVR 或文本侧车被改写，拒绝输出")
    payload = out.tobytes()
    payload_hash = hashlib.sha256(payload).digest()
    header = HEADER.pack(b"UAVSOFT1", n, NUM_CLASSES, flags, data_hash, text_hash, payload_hash)
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
            "seq_len": seq_len, "temperature": t, "flags": flags,
            "uavr_sha256": data_hash.hex(),
            "instructions_sha256": text_hash.hex() if has_sidecar else None,
            "payload_sha256": payload_hash.hex(),
            # 教师来源（npz 没有这些键时为 null）；student config 会读回再落盘。
            "teacher_ckpt_sha256": provenance.get("teacher_ckpt_sha256"),
            "teacher_select_split": provenance.get("select_split"),
            "teacher_train_data_sha256": provenance.get("train_data_sha256"),
            # 信念教师（teacher_belief.py）：版本、假设数、温度（步数单位）、聚合方式、seed
            "teacher_version": provenance.get("teacher_version"),
            "belief_k": provenance.get("k"),
            "belief_alpha_steps": provenance.get("alpha_steps"),
            "belief_aggregate": provenance.get("aggregate"),
            "belief_seed": provenance.get("seed"),
            "soft_files": [os.path.abspath(s) for s in args.soft]}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"写出 {args.out}（{HEADER.size + out.nbytes} 字节）", file=sys.stderr)


if __name__ == "__main__":
    main()
