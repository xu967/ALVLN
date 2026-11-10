#!/usr/bin/env python3
"""末 k 轮权重平均 + FORWARD_SLOW 类先验校准，写出可交付的 ckpt（计划 B4；闭环证据 experiments/2026-11-10-swa-calib/RESULT.md）。

用法（仓库根）：python tools/average_ckpt.py --run <run 目录> --data <dataset.uavr> --out <输出 ckpt> [--last-k 10] [--no-calibrate] [--infer-bin <uavrgb_infer>]
run 目录要有 --save-every-epoch 训出的 ckpt_epNN.bin；GPU 由调用方用 CUDA_VISIBLE_DEVICES 钉；<out> 已存在则拒绝；产物旁写 <out>.json（来源与 sha256、δ、占比、二进制 sha256）。
平均：末 k 轮的全部参数张量按 float64 累加取平均再转 float32，头与 Adam 状态沿用最后一轮（模型没有归一化层，不用重估统计量）。
校准：真实 uavrgb_infer 在 val 分区（剔除碰撞步）逐帧推理，二分求 δ，使 FORWARD 的 logit 加 δ 后预测为 FORWARD 的占比等于 val 标签里 FORWARD 的占比，再 fc.b[FORWARD] += δ。
--data 的 sha256 必须等于 ckpt 训练契约里记的数据集 sha256：传错数据集，目标占比就不是训练分布的，δ 会静默偏掉。
只平均不校准比 ckpt_best 还差（闭环 20/60 对 27/60），两步必须一起用；--no-calibrate 只给对拍与消融。
"""
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

# ───────────────────────── 配置区 ─────────────────────────
LAST_K = 10                   # 末 10 轮：11-10 闭环只测过 k=10（走廊内 39/60 对 ckpt_best 27/60，RESULT.md）；草稿还生成过 k=5/15/20（swa_make.log），没进闭环
FORWARD = 1                   # FORWARD_SLOW 的类下标（src/include/uavrgb/action.hpp）
FC_B = 7                      # fc.b 在参数表里的位置（顺序同 cnn_params：c1.W c1.b c2.W c2.b c3.W c3.b fc.W fc.b，FiLM/aux 只追加在后面）
DELTA_RANGE = (-30.0, 30.0)   # δ 二分区间（nat）；与草稿 analyze2.calibrate 同，三个训练种子实测 δ 在 +1.56..+1.70
BISECT_STEPS = 60             # 区间宽 60，二分 60 次后剩约 5e-17，超出 float64 对 log 概率的分辨；与草稿同，改了就复现不了 11-10 的权重
VERIFY_FRAMES = 3             # 校准后 FORWARD 预测帧数与目标帧数的容许差：二分的 δ 落在阈值两侧之一（差 1 帧），真实推理与离线平移再差 ≤1 帧（草稿实测 1/1458）；超过说明 δ 触到区间端点或概率异常
# ──────────────────────────────────────────────────────────

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INFER = os.path.join(os.environ.get("UAV_RGB_BUILD_DIR") or os.path.join(REPO, "build", "check"), "uavrgb_infer")  # 同 sim/tools/student.py
CK_HEAD = struct.Struct("<8sII4i3I")         # magic、version、param_count、in_h in_w in_c num_classes、adam_t、epoch、weight_count（44 B）
DS_HEAD = struct.Struct("<4sHHIHHBBBBQQQI")  # tools/DATASET_FORMAT.md §1.1
SAME = ("param_count", "in_h", "in_w", "in_c", "num_classes", "text_dim", "aux_classes", "prev_dim")   # 末 k 轮的模型结构必须一致
# 数据集读法不 import select_ckpt.Uavr：它经 student 拉进 PIL 与 teacher，本工具只依赖 numpy 与标准库。


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ckpt(path):
    """UAVRCK01 v6（布局见 src/include/uavrgb/checkpoint.hpp，文件没有校验和）；版本、长度、非有限值任何不符都报错，不猜。
    shortcut: 只认 v6；v2..v5 要照 checkpoint.cpp 的 read_header/read_text_dim/read_v6 逐版本补，用到再加。"""
    with open(path, "rb") as f:
        b = f.read()
    head = CK_HEAD.unpack_from(b, 0) if len(b) >= CK_HEAD.size else (b"", 0, 0, 0, 0, 0, 0, 0, 0, 0)
    magic, ver, pc, h, w, c, ncls, _adam_t, epoch, nw = head
    if magic != b"UAVRCK01" or ver != 6 or nw not in (0, ncls) or min(h, w, c, ncls) <= 0:
        raise ValueError(f"{path}: 不是合法的 UAVRCK01 v6（magic={magic!r} version={ver}）")
    off = CK_HEAD.size + 4 * nw + 4                      # class_weights、best_macro_f1
    text_dim, = struct.unpack_from("<i", b, off)
    off += 4 + 113                                       # text_dim、v5 训练契约（定长 113 B，开头 32 B 是训练数据集 sha256）
    contract = b[off - 113:off]
    aux, prev = struct.unpack_from("<2i", b, off)
    off += 8
    params, spans = [], []
    for _ in range(pc):
        nd, = struct.unpack_from("<I", b, off)
        if nd > 8:
            raise ValueError(f"{path}: ndim 异常 {nd}")
        dims = struct.unpack_from(f"<{nd}q", b, off + 4)
        off += 4 + 8 * nd
        n = int(np.prod(dims, dtype=np.int64))
        if n <= 0 or off + 4 * n > len(b):
            raise ValueError(f"{path}: 参数数据不足")
        params.append(np.frombuffer(b, "<f4", n, off).reshape(dims))
        spans.append((off, n))
        off += 4 * n
    if off + 8 * sum(n for _, n in spans) != len(b):     # 之后恰好是 Adam m、v 两段，不许多也不许少
        raise ValueError(f"{path}: 长度与布局不符（Adam 段缺失或尾部多余）")
    if not all(np.isfinite(p).all() for p in params):
        raise ValueError(f"{path}: 参数含非有限值")
    return {"raw": b, "param_count": pc, "in_h": h, "in_w": w, "in_c": c, "num_classes": ncls, "epoch": epoch, "text_dim": text_dim,
            "aux_classes": aux, "prev_dim": prev, "contract": contract, "params": params, "spans": spans}


def write_ckpt(src, params, path):
    """以 src 为模板只换参数张量（头、契约、Adam m/v 原样），先写临时文件再 rename，同 save_checkpoint。"""
    buf = bytearray(src["raw"])
    for (off, n), p in zip(src["spans"], params):
        if p.size != n or not np.isfinite(p).all():
            raise ValueError("参数形状不符或含非有限值")
        buf[off:off + 4 * n] = np.ascontiguousarray(p, dtype="<f4").tobytes()
    with open(path + ".tmp", "wb") as f:
        f.write(buf)
    os.replace(path + ".tmp", path)


def load_window(run, k):
    """run 里最大的 ckpt_epNN 为末轮，取 ep(末-k+1)..ep(末)；缺任何一个、头里 epoch 与文件名不符、结构或形状不一致都报错。
    shortcut: 末轮不与 config.json 的 epochs 对账（续训时 epochs 的含义不同），被中途杀掉的 run 会平均到不完整的窗口；接流水线时由训练的 DONE 保证。"""
    run = os.path.abspath(run)
    found = {int(m.group(1)): os.path.join(run, n) for n in os.listdir(run) if (m := re.fullmatch(r"ckpt_ep(\d+)\.bin", n))}
    if not found:
        raise ValueError(f"{run} 里没有 ckpt_epNN.bin（训练要加 --save-every-epoch）")
    window = range(max(found) - k + 1, max(found) + 1)
    missing = [f"ckpt_ep{e:02d}.bin" for e in window if e not in found]   # k 比 run 的轮数大时 e<=0 也算缺
    if k < 1 or missing:
        raise ValueError(f"末 {k} 轮（ep{window.start}..ep{window.stop - 1}）取不全：缺 " + "、".join(missing or ["（k 须 >= 1）"]))
    files = [(e, found[e]) for e in window]
    cks = [read_ckpt(p) for _, p in files]
    for (e, p), c in zip(files, cks):
        if c["epoch"] != e:
            raise ValueError(f"{p}: 头里 epoch={c['epoch']} 与文件名的 ep{e} 不符（文件被改名或拷错？）")
        if [c[x] for x in SAME] != [cks[0][x] for x in SAME]:
            raise ValueError(f"{p}: 模型结构 {SAME} 与 {files[0][1]} 不同")
        for i, (x, y) in enumerate(zip(c["params"], cks[0]["params"])):
            if x.shape != y.shape:
                raise ValueError(f"{p}: 参数 {i} 形状 {x.shape} 与 {files[0][1]} 的 {y.shape} 不一致")
    return files, cks


def average(param_lists):
    """float64 按轮序累加后除以 k 再转 float32；累加顺序固定，保证逐位可复现。"""
    acc = [p.astype(np.float64) for p in param_lists[0]]
    for ps in param_lists[1:]:
        acc = [a + p for a, p in zip(acc, ps)]
    return [(a / len(param_lists)).astype(np.float32) for a in acc]


def shift_forward_bias(params, delta):
    out = list(params)
    out[FC_B] = params[FC_B].copy()
    out[FC_B][FORWARD] += np.float32(delta)   # float32 加 float32：与草稿 calib.py 同一次舍入
    return out


def forward_frac(probs, delta=0.0):
    z = np.log(np.maximum(probs, 1e-300))     # 概率只有 6 位有效数字，下限截断同草稿
    z[:, FORWARD] += delta
    return float((z.argmax(1) == FORWARD).mean())


def calibrate(probs, target):
    """二分求 δ：占比随 δ 单调不减，找使 forward_frac >= target 的最小 δ；做法同草稿 analyze2.calibrate。"""
    lo, hi = DELTA_RANGE
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if forward_frac(probs, mid) < target else (lo, mid)
    delta = 0.5 * (lo + hi)
    got = forward_frac(probs, delta)
    if abs(got - target) * len(probs) > VERIFY_FRAMES:
        raise ValueError(f"δ={delta:.3f} 处 FORWARD 占比 {got:.4f} 仍偏离目标 {target:.4f}（触到区间端点？概率异常？）")
    return delta


def read_val(path):
    """dataset.uavr 的 val 分区（tools/DATASET_FORMAT.md §1–§2）→ (images NCHW memmap, 样本下标, 标签)。
    剔除碰撞步与 src/dataset.cpp 的 drop_collision 同义：collision_flags 缺失、长度≠N、含 0/1 以外字符都报错，不当成无碰撞。"""
    with open(path, "rb") as f:
        magic, ver, hs, n, h, w, c, idt, ldt, _r, io, lo, mo, ml = DS_HEAD.unpack(f.read(DS_HEAD.size))
        if (magic, ver, hs, c, idt, ldt) != (b"UAVR", 1, 48, 3, 1, 1):
            raise ValueError(f"{path}: 不是 UAVR v1 的 RGB uint8 数据集")
        if (io, lo, mo) != (48, 48 + n * c * h * w, 48 + n * c * h * w + n) or os.path.getsize(path) != mo + ml:
            raise ValueError(f"{path}: 头里的偏移与文件长度不自洽")
        f.seek(mo)
        meta = json.loads(f.read(ml).decode("utf-8"))
    flags = meta.get("collision_flags")
    if not isinstance(flags, str) or len(flags) != n or set(flags) - {"0", "1"}:
        raise ValueError(f"{path}: meta 缺 collision_flags 或格式不符（长度须等于 N、只含 0/1）；不当成无碰撞")
    idx = np.asarray([i for ep in meta["episodes"] if ep["split"] == "val"
                      for i in range(ep["sample_start"], ep["sample_start"] + ep["sample_count"]) if flags[i] == "0"], dtype=np.int64)
    if not idx.size:
        raise ValueError(f"{path}: val 分区剔除碰撞步后没有样本")
    return np.memmap(path, np.uint8, "r", io, (n, c, h, w)), idx, np.fromfile(path, dtype=np.uint8, count=n, offset=lo)[idx]


def infer_probs(binary, ckpt, images, idx):
    """起 uavrgb_infer 子进程，按 sim/tools/student.py 的行协议逐帧喂 HWC RGB（写一帧读一行，不会撑满管道）→ probs[len(idx), 4]。
    shortcut: 读回复没有超时，进程挂死会卡住；接流水线时由外层守护脚本兜（student.py 有 30 s 超时，但要 PIL）。"""
    h, w = images.shape[2:]
    out = np.zeros((len(idx), 4))
    with tempfile.TemporaryFile() as err, subprocess.Popen([binary, "--ckpt", ckpt], stdin=subprocess.PIPE,
                                                         stdout=subprocess.PIPE, stderr=err) as p:
        def reply():
            line = p.stdout.readline()
            if not line:
                p.wait()
                err.seek(0)
                raise RuntimeError(f"uavrgb_infer 提前退出 rc={p.returncode}: {err.read().decode(errors='replace').strip()}")
            return json.loads(line)
        try:
            hello = reply()
            if (hello["in_h"], hello["in_w"], hello["in_c"], hello["num_classes"]) != (h, w, 3, 4):
                raise RuntimeError(f"推理进程 hello 与数据集不符: {hello}")
            for k, g in enumerate(idx):
                p.stdin.write(np.ascontiguousarray(images[g].transpose(1, 2, 0)).tobytes())   # NCHW → HWC，同 student.py 发的图
                p.stdin.flush()
                out[k] = reply()["probs"]
            p.stdin.close()
            if p.wait() != 0:
                raise RuntimeError(f"uavrgb_infer rc={p.returncode}")
        except BaseException:
            p.kill()
            raise
    return out


def calibrate_and_write(args, last, params, out):
    """平均后的 ckpt 真实推理 val → δ → 写 fc.b[FORWARD]+δ 的 ckpt，再对写出的文件真实推理核对占比，通过才落到 out。返回 json 字段。
    shortcut: 只支持单帧纯视觉 ckpt。文本指令（逐集 --text）与双帧/prev-action（要读 dataset.aux 的 prev_index/prev_action）行协议还原不出来，
    宁可报错也不算出一个错的 δ；需要时按 student.py / select_ckpt.predict_indices 的写法补。"""
    if (last["in_c"], last["text_dim"], last["prev_dim"]) != (3, 0, 0):
        raise ValueError(f"暂不支持校准：ckpt 带 in_c={last['in_c']} text_dim={last['text_dim']} prev_dim={last['prev_dim']}，"
                         "这些输入无法从数据集经行协议还原（只平均请加 --no-calibrate）")
    images, idx, y = read_val(args.data)
    if images.shape[2:] != (last["in_h"], last["in_w"]):
        raise ValueError(f"数据集 {images.shape[2:]} 与 ckpt 输入 {(last['in_h'], last['in_w'])} 不符")
    data_sha, trained_on = sha256_file(args.data), last["contract"][:32].hex()   # 契约开头 32 B = trainer 记的训练数据集 sha256
    if data_sha != trained_on:
        raise ValueError(f"--data 的 sha256 {data_sha[:12]} 与 ckpt 训练契约里的 {trained_on[:12]} 不同：校准只在训练用的那份数据的 val 上做")
    with tempfile.TemporaryDirectory(prefix=".average_ckpt_", dir=os.path.dirname(out)) as tmp:
        write_ckpt(last, params, f"{tmp}/avg.bin")
        probs = infer_probs(args.infer_bin, f"{tmp}/avg.bin", images, idx)
        target = float((y == FORWARD).mean())
        delta = calibrate(probs, target)
        write_ckpt(last, shift_forward_bias(params, delta), f"{tmp}/cal.bin")
        after = float((infer_probs(args.infer_bin, f"{tmp}/cal.bin", images, idx).argmax(1) == FORWARD).mean())
        if abs(after - target) * len(idx) > VERIFY_FRAMES:
            raise ValueError(f"写出的 ckpt 真实推理 FORWARD 占比 {after:.4f} 偏离目标 {target:.4f}，不落盘")
        os.replace(f"{tmp}/cal.bin", out)
    return {"data": os.path.abspath(args.data), "data_sha256": data_sha,
            "split": "val", "n_frames": int(len(idx)), "forward_target": target, "forward_frac_before": forward_frac(probs), "delta": float(delta),
            "forward_frac_after_real": after, "infer_bin": os.path.abspath(args.infer_bin), "infer_bin_sha256": sha256_file(args.infer_bin)}


def run(args):
    out = os.path.abspath(args.out)
    if os.path.exists(out) or os.path.exists(out + ".json"):
        raise ValueError(f"{out}（或 .json）已存在，不覆盖")
    files, cks = load_window(args.run, args.last_k)
    last, params = cks[-1], average([c["params"] for c in cks])
    meta = {"tool": "tools/average_ckpt.py", "run": os.path.abspath(args.run), "last_k": args.last_k, "last_epoch": files[-1][0],
            "template": files[-1][1], "inputs": [{"file": p, "epoch": e, "sha256": sha256_file(p)} for e, p in files],
            "calibrated": not args.no_calibrate, "out": out}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.no_calibrate:
        write_ckpt(last, params, out)
    else:
        meta.update(calibrate_and_write(args, last, params, out))
    meta["out_sha256"] = sha256_file(out)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"average_ckpt: ep{files[0][0]}..ep{files[-1][0]}（k={args.last_k}）→ {out}  sha256={meta['out_sha256']}")
    if meta["calibrated"]:
        print(f"average_ckpt: val {meta['n_frames']} 帧，FORWARD 占比 {meta['forward_frac_before']:.4f} → {meta['forward_frac_after_real']:.4f}"
              f"（目标 {meta['forward_target']:.4f}），δ={meta['delta']:+.6f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="--save-every-epoch 训出的 run 目录")
    ap.add_argument("--data", help="dataset.uavr（与训练同一份）；校准必需")
    ap.add_argument("--out", required=True, help="输出 ckpt；已存在则拒绝")
    ap.add_argument("--last-k", type=int, default=LAST_K)
    ap.add_argument("--no-calibrate", action="store_true", help="只平均，不校准（给对拍与消融）")
    ap.add_argument("--infer-bin", default=DEFAULT_INFER)
    args = ap.parse_args(argv)
    if not args.no_calibrate and not args.data:
        ap.error("校准需要 --data（只平均请加 --no-calibrate）")
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as e:
        sys.exit(f"average_ckpt: {e}")


if __name__ == "__main__":
    sys.exit(main())
