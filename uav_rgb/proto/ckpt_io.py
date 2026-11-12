#!/usr/bin/env python3
"""C++ checkpoint（UAVRCK01 v6）的读写，只用 numpy。

格式的唯一出处是 src/include/uavrgb/checkpoint.hpp 与 src/checkpoint.cpp，这里逐段复刻：
  "UAVRCK01" | u32 version=6 | u32 param_count | i32 in_h,in_w,in_c,num_classes | u32 adam_t | u32 epoch
  | u32 weight_count + f32[weight_count] | f32 best_macro_f1 | i32 text_dim | 契约 113 B
  | i32 aux_classes | i32 prev_dim | 逐参数 (u32 ndim, i64 dims[ndim], f32 data) | 逐参数 Adam m | 逐参数 Adam v
读端的校验项与 load_checkpoint 一致：magic、version、结构元信息、参数个数、逐参数 dims、
非有限值、尾部多余数据；写端先写临时文件再 rename，写出前同样拒绝非有限值。

不经过 torch：推理包装器要在 import torch 之前读文件头，把 hello 行发出去（torch 冷启动要几秒）。

只支持 v6（训练器当前写出的版本）。v2..v5 的差异只是字段缺省，没有样本文件可验证，所以没有补；
遇到旧版本直接报错，需要时按 checkpoint.cpp 的 read_header 版本分支补。
"""
import os
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"UAVRCK01"
VERSION = 6
CONTRACT_BYTES = 113   # v5 训练契约定长段（checkpoint.cpp kContractBytes）
TEXT_LEN = 9           # src/include/uavrgb/text.hpp kTextLen
TEXT_VOCAB = 22        # kTextVocab
MAX_TEXT_DIM = 4096    # kMaxTextDim
MAX_PREV_DIM = 64      # model.hpp kMaxPrevDim
MAX_AUX_CLASSES = 254  # model.hpp kMaxAuxClasses
FILM_CH = 64           # c3 的输出通道数，FiLM 给每个通道一个 γ 和一个 β

# 契约布局：32B dataset sha256 | u8 flags | 32B soft sha256 | f32 distill_alpha | f32 lr | u32 seed | i32 batch | 32B split sha256
_CONTRACT_FMT = "<32sB32sffIi32s"
assert struct.calcsize(_CONTRACT_FMT) == CONTRACT_BYTES


class CkptError(ValueError):
    """checkpoint 读写失败（对应 C++ 的 std::invalid_argument "checkpoint: ..."）。"""


def make_contract(dataset_sha256, drop_collision, lr, seed, batch, split_scheme_sha256=None,
                  soft_targets_sha256=None, distill_alpha=0.5):
    """按 trainer 的 make_contract 拼 113 B。sha 参数接受 32 字节或 64 位十六进制串；缺省为全 0。"""
    def sha(v):
        if v is None:
            return bytes(32)
        if isinstance(v, str):
            v = bytes.fromhex(v)
        if len(v) != 32:
            raise CkptError("sha256 必须是 32 字节")
        return bytes(v)
    return struct.pack(_CONTRACT_FMT, sha(dataset_sha256), 1 if drop_collision else 0, sha(soft_targets_sha256),
                       float(distill_alpha), float(lr), int(seed) & 0xFFFFFFFF, int(batch), sha(split_scheme_sha256))


def parse_contract(raw):
    """113 B -> dict；sha 以十六进制串给出，全 0 的 sha 给空串（= 未记录）。"""
    if len(raw) != CONTRACT_BYTES:
        raise CkptError(f"契约长度 {len(raw)} != {CONTRACT_BYTES}")
    ds, flags, soft, alpha, lr, seed, batch, split = struct.unpack(_CONTRACT_FMT, raw)

    def hexs(b):
        return "" if b == bytes(32) else b.hex()
    return {"dataset_sha256": hexs(ds), "drop_collision": bool(flags & 1), "soft_targets_sha256": hexs(soft),
            "distill_alpha": alpha, "lr": lr, "seed": seed, "batch": batch, "split_scheme_sha256": hexs(split)}


def cond_dim(text_dim, prev_dim):
    """FiLM 条件向量维：[9·text_dim 文本特征 | prev one-hot]（model.hpp cond_dim）。"""
    return TEXT_LEN * text_dim + prev_dim


def validate_structure(in_h, in_w, in_c, num_classes, text_dim=0, aux_classes=0, prev_dim=0):
    """与 make_cnn 的入参检查一致。"""
    if in_h <= 0 or in_w <= 0 or num_classes <= 0:
        raise CkptError("in_h/in_w/num_classes 必须为正")
    if in_c not in (3, 6):
        raise CkptError(f"in_c 只能是 3（单帧）或 6（双帧），实际 {in_c}")
    if not 0 <= text_dim <= MAX_TEXT_DIM:
        raise CkptError(f"text_dim 必须在 0..{MAX_TEXT_DIM}")
    if not 0 <= aux_classes <= MAX_AUX_CLASSES or not 0 <= prev_dim <= MAX_PREV_DIM:
        raise CkptError("aux_classes 须在 0..254、prev_dim 须在 0..64")
    if in_h % 8 or in_w % 8:
        raise CkptError("输入 H/W 必须是 8 的倍数（三次 2x2 pool）")


def expected_shapes(in_h, in_w, in_c, num_classes, text_dim=0, aux_classes=0, prev_dim=0):
    """参数形状，顺序同 cnn_params：c1.W c1.b c2.W c2.b c3.W c3.b fc.W fc.b [text_emb] [film.W film.b] [aux.W aux.b]。
    conv W = [out, in, kh, kw]，Linear W = [out, in]，与 PyTorch 的 weight 布局相同。"""
    validate_structure(in_h, in_w, in_c, num_classes, text_dim, aux_classes, prev_dim)
    flat = FILM_CH * (in_h // 8) * (in_w // 8)
    shapes = [(16, in_c, 3, 3), (16,), (32, 16, 3, 3), (32,), (FILM_CH, 32, 3, 3), (FILM_CH,),
              (num_classes, flat), (num_classes,)]
    if text_dim > 0:
        shapes.append((TEXT_VOCAB, text_dim))
    cdim = cond_dim(text_dim, prev_dim)
    if cdim > 0:
        shapes += [(2 * FILM_CH, cdim), (2 * FILM_CH,)]
    if aux_classes > 0:
        shapes += [(aux_classes, FILM_CH, 1, 1), (aux_classes,)]
    return shapes


@dataclass
class Ckpt:
    """一份 checkpoint 的全部内容。字段含义见 checkpoint.hpp；params/adam_m/adam_v 按 cnn_params 顺序。"""
    in_h: int
    in_w: int
    in_c: int
    num_classes: int
    params: list
    adam_m: list
    adam_v: list
    text_dim: int = 0
    aux_classes: int = 0
    prev_dim: int = 0
    adam_t: int = 0
    epoch: int = 0
    class_weights: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    best_macro_f1: float = -1.0
    contract: bytes = bytes(CONTRACT_BYTES)

    def structure(self):
        return dict(in_h=self.in_h, in_w=self.in_w, in_c=self.in_c, num_classes=self.num_classes,
                    text_dim=self.text_dim, aux_classes=self.aux_classes, prev_dim=self.prev_dim)


@dataclass
class Meta:
    """只读文件头得到的结构信息（read_checkpoint_meta）。"""
    in_h: int
    in_w: int
    in_c: int
    num_classes: int
    text_dim: int
    aux_classes: int
    prev_dim: int


class _Reader:
    def __init__(self, buf):
        self.buf = memoryview(buf)
        self.pos = 0

    def take(self, n, what):
        if n < 0 or self.pos + n > len(self.buf):
            raise CkptError(f"checkpoint: 读取不足: {what}")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt, what):
        """总是返回 tuple。"""
        return struct.unpack(fmt, self.take(struct.calcsize(fmt), what))

    def one(self, fmt, what):
        return self.unpack(fmt, what)[0]

    def floats(self, count, what):
        # 拷贝成可写数组：memoryview 背后是只读 bytes，训练续跑、平均要就地改
        return np.frombuffer(self.take(4 * count, what), dtype="<f4").astype(np.float32, copy=True)


def _read_header(r):
    """magic/version/结构元信息；read_meta 与 read_checkpoint 共用（对应 checkpoint.cpp read_header）。"""
    if bytes(r.take(8, "magic")) != MAGIC:
        raise CkptError("checkpoint: magic 不符（不是 UAVRCK01）")
    version = r.one("<I", "version")
    if version != VERSION:
        raise CkptError(f"checkpoint: version {version} 不支持（本模块只实现 v{VERSION}）")
    param_count = r.one("<I", "param_count")
    in_h, in_w, in_c, num_classes = r.unpack("<4i", "结构元信息")
    if in_h <= 0 or in_w <= 0 or in_c <= 0 or num_classes <= 0:
        raise CkptError("checkpoint: 结构元信息非正数")
    if in_c > 6:
        raise CkptError("checkpoint: in_c 非法（只支持 3/6）")
    return param_count, in_h, in_w, in_c, num_classes


def _read_mid(r, num_classes):
    """头之后到 prev_dim 的定长段：adam_t/epoch/类别权重/best/text_dim/契约/aux_classes/prev_dim。"""
    adam_t, epoch, nw = r.unpack("<3I", "adam_t/epoch/weight_count")
    if nw not in (0, num_classes):
        raise CkptError("checkpoint: class_weights 数量不符")
    weights = r.floats(nw, "class_weight")
    if nw and not (np.isfinite(weights).all() and (weights >= 0).all()):
        raise CkptError("checkpoint: class_weights 含非有限/负值")
    best = r.one("<f", "best_macro_f1")
    if not np.isfinite(best) or best < -1.0:
        raise CkptError("checkpoint: best_macro_f1 非法")
    text_dim = r.one("<i", "text_dim")
    if not 0 <= text_dim <= MAX_TEXT_DIM:
        raise CkptError("checkpoint: text_dim 非法")
    contract = bytes(r.take(CONTRACT_BYTES, "契约"))
    parsed = parse_contract(contract)
    if not (np.isfinite(parsed["distill_alpha"]) and np.isfinite(parsed["lr"])):
        raise CkptError("checkpoint: 训练契约含非有限值")
    aux_classes, prev_dim = r.unpack("<2i", "aux_classes/prev_dim")
    if not 0 <= aux_classes <= MAX_AUX_CLASSES or not 0 <= prev_dim <= MAX_PREV_DIM:
        raise CkptError("checkpoint: aux_classes/prev_dim 非法")
    return adam_t, epoch, weights, float(best), text_dim, contract, aux_classes, prev_dim


def read_meta(path):
    """只读头部：in_h/in_w/in_c/num_classes/text_dim/aux_classes/prev_dim。不读参数区。"""
    with open(path, "rb") as f:
        buf = f.read(1 << 16)   # 头到 prev_dim 不到 0.5 KiB（类别权重段随类数增长）；不读参数区
    r = _Reader(buf)
    _, in_h, in_w, in_c, num_classes = _read_header(r)
    _, _, _, _, text_dim, _, aux_classes, prev_dim = _read_mid(r, num_classes)
    return Meta(in_h, in_w, in_c, num_classes, text_dim, aux_classes, prev_dim)


def _check_finite(name, arr):
    if not np.isfinite(arr).all():
        idx = int(np.flatnonzero(~np.isfinite(arr.reshape(-1)))[0])
        raise CkptError(f"checkpoint: {name}: 含非有限值（索引 {idx}）")


def read_checkpoint(path):
    """读整份 checkpoint 并按 load_checkpoint 的口径校验；任何不符直接抛 CkptError，不做部分加载。"""
    with open(path, "rb") as f:
        buf = f.read()
    r = _Reader(buf)
    param_count, in_h, in_w, in_c, num_classes = _read_header(r)
    adam_t, epoch, weights, best, text_dim, contract, aux_classes, prev_dim = _read_mid(r, num_classes)
    try:
        shapes = expected_shapes(in_h, in_w, in_c, num_classes, text_dim, aux_classes, prev_dim)
    except CkptError as exc:
        raise CkptError(f"checkpoint: 模型结构不符（{exc}）") from None
    if param_count != len(shapes):
        raise CkptError(f"checkpoint: 参数个数不符（文件 {param_count}，结构要求 {len(shapes)}）")
    params = []
    for i, shape in enumerate(shapes):
        ndim = r.one("<I", "ndim")
        if ndim > 8:
            raise CkptError(f"checkpoint: param: ndim 异常 {ndim}")
        dims = r.unpack(f"<{ndim}q", "dims") if ndim else ()
        if dims != shape:
            raise CkptError(f"checkpoint: param[{i}]: 形状与模型不符（文件 {dims}，结构要求 {shape}）")
        arr = r.floats(int(np.prod(shape)), "param").reshape(shape)
        _check_finite(f"param[{i}]", arr)
        params.append(arr)
    adam_m, adam_v = [], []
    for dst, name in ((adam_m, "adam.m"), (adam_v, "adam.v")):
        for shape in shapes:
            arr = r.floats(int(np.prod(shape)), name).reshape(shape)
            _check_finite(name, arr)
            dst.append(arr)
    if r.pos != len(buf):
        raise CkptError("checkpoint: 文件尾部有多余数据")
    return Ckpt(in_h=in_h, in_w=in_w, in_c=in_c, num_classes=num_classes, params=params, adam_m=adam_m,
                adam_v=adam_v, text_dim=text_dim, aux_classes=aux_classes, prev_dim=prev_dim, adam_t=adam_t,
                epoch=epoch, class_weights=weights, best_macro_f1=best, contract=contract)


def encode_checkpoint(ck):
    """Ckpt -> bytes（v6）。写出前校验形状与有限性，规则同 save_checkpoint。"""
    shapes = expected_shapes(ck.in_h, ck.in_w, ck.in_c, ck.num_classes, ck.text_dim, ck.aux_classes, ck.prev_dim)
    for label, arrs in (("params", ck.params), ("adam_m", ck.adam_m), ("adam_v", ck.adam_v)):
        if len(arrs) != len(shapes):
            raise CkptError(f"checkpoint: {label} 个数 {len(arrs)} 与结构要求 {len(shapes)} 不符")
        for i, (a, s) in enumerate(zip(arrs, shapes)):
            if tuple(a.shape) != s:
                raise CkptError(f"checkpoint: {label}[{i}] 形状 {tuple(a.shape)} 与结构要求 {s} 不符")
    weights = np.asarray(ck.class_weights, dtype="<f4").reshape(-1)
    if len(weights) not in (0, ck.num_classes):
        raise CkptError("checkpoint: class_weights 长度与 num_classes 不符")
    if len(ck.contract) != CONTRACT_BYTES:
        raise CkptError(f"checkpoint: 契约必须是 {CONTRACT_BYTES} B")
    out = [MAGIC, struct.pack("<II", VERSION, len(shapes)),
           struct.pack("<4i", ck.in_h, ck.in_w, ck.in_c, ck.num_classes),
           struct.pack("<3I", ck.adam_t, ck.epoch, len(weights)), weights.tobytes(),
           struct.pack("<f", ck.best_macro_f1), struct.pack("<i", ck.text_dim), ck.contract,
           struct.pack("<2i", ck.aux_classes, ck.prev_dim)]
    for i, a in enumerate(ck.params):
        _check_finite(f"param[{i}]", a)
        out.append(struct.pack("<I", a.ndim) + struct.pack(f"<{a.ndim}q", *a.shape))
        out.append(np.ascontiguousarray(a, dtype="<f4").tobytes())
    for name, arrs in (("adam.m", ck.adam_m), ("adam.v", ck.adam_v)):
        for a in arrs:
            _check_finite(name, a)
            out.append(np.ascontiguousarray(a, dtype="<f4").tobytes())
    return b"".join(out)


def write_checkpoint(path, ck):
    """先写 <path>.tmp 再 rename，中途失败不会留下半截文件覆盖旧的。"""
    data = encode_checkpoint(ck)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
    os.replace(tmp, path)
