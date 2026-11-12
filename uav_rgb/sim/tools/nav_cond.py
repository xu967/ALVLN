#!/usr/bin/env python3
"""航向码条件输入（任务卡 W1-9）：把「已经偏了多少」压成 4 值码，走现成的 prev-action 通路（训练器 --prev-action、
推理协议里的 1 字节、模型 prev_dim=4 的 one-hot），不改 C++。

n = 该步之前已执行的 YAW_LEFT 数 − YAW_RIGHT 数 − H（航向设定值；柱阵与长课程任务 H 恒为 0）。
  码 0：|n| ≤ 1；码 2：n ≥ 2（偏左了，该右转）；码 3：n ≤ −2（偏右了，该左转）；码 1 不用；首帧 n = 0 → 码 0。
这个量机体自己数得出来（真机上对应飞控偏航估计相对起飞航向的差），不是特权信息。

打包（tools/preprocess.py --cond heading-code）、采集与闭环（sim/tools/collect.py --student-cond heading-code）都 import 这里的
同一个函数，不许各写一份。同一个文件还放一致性闸门：模型训练时用的是哪种条件，评测时必须喂同一种，否则等于喂分布外输入。

自检：python sim/tools/nav_cond.py
"""
import hashlib
import json
import math
import os
import struct
import sys

# 与 src/include/uavrgb/action.hpp、sim/tools/collect.py 一致（test_nav_cond_wiring 锁住）
FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = 1, 2, 3

COND_PREV_ACTION = "prev-action"
COND_HEADING_CODE = "heading-code"
COND_CHOICES = (COND_PREV_ACTION, COND_HEADING_CODE)
# 数据集 meta 顶层 "cond" 的取值（只有 --cond heading-code 的包才有这个键；没有 = prev-action 语义）
DATASET_COND_TAG = "heading_code.v1"
_TAG_TO_COND = {DATASET_COND_TAG: COND_HEADING_CODE}

_UAVR_HEAD = struct.Struct("<4sHHIHHBBBBQQQI")   # 同 tools/preprocess.py 的 HEADER（48 B）：meta_off 在 36、meta_len 在 44
_CKPT_HEAD = struct.Struct("<8sIIiiiiIII")       # 同 src/checkpoint.cpp：magic + 9×4B 定长头（共 44 B）
_CKPT_CONTRACT_BYTES = 113                       # v5 训练契约固定 113 B，开头 32 B 为 dataset_sha256


def heading_code(n, setpoint=0):
    """净转向数 n（已执行 YAW_LEFT − YAW_RIGHT）→ 4 值码。setpoint = 航向设定值 H，柱阵任务恒为 0。"""
    m = n - setpoint
    if m >= 2:
        return 2
    if m <= -2:
        return 3
    return 0


def heading_codes(executed, setpoint=0):
    """一集的已执行动作序列 → 每步的码（取执行**前**的 n：第 t 步的码只用第 0..t−1 步的动作）。"""
    codes, n = [], 0
    for action in executed:
        codes.append(heading_code(n, setpoint))
        n += (action == YAW_LEFT) - (action == YAW_RIGHT)
    return codes


# ---------------------------------------------------------------- 一致性闸门

def read_dataset_meta(path):
    """dataset.uavr 尾部的 meta JSON；不是 UAVR 文件抛 ValueError，读不了抛 OSError。"""
    with open(path, "rb") as f:
        head = f.read(_UAVR_HEAD.size)
        if len(head) != _UAVR_HEAD.size:
            raise ValueError(f"{path}: 不是 UAVR 数据集（文件头不足 {_UAVR_HEAD.size} 字节）")
        magic, _ver, header_size, _n, _h, _w, _c, _dt, _ldt, _r, _io, _lo, meta_off, meta_len = _UAVR_HEAD.unpack(head)
        if magic != b"UAVR" or header_size != _UAVR_HEAD.size:
            raise ValueError(f"{path}: 不是 UAVR 数据集（magic/头长度不符）")
        f.seek(meta_off)
        raw = f.read(meta_len)
    if len(raw) != meta_len:
        raise ValueError(f"{path}: meta 被截断")
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: meta 不是合法 JSON：{exc}") from exc


def dataset_cond_tag(path):
    """数据集 meta 顶层的 cond 标签；没有这个键返回 None。"""
    return read_dataset_meta(path).get("cond")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt_dataset_sha256(path):
    """读 UAVRCK01 v5+ 权重头内训练契约记录的 dataset_sha256（64 位小写 hex）。
    对齐 src/checkpoint.cpp:read_v5_contract：文件不存在、非 v5+ 权重或全零占位契约返回 None。"""
    try:
        with open(path, "rb") as f:
            raw = f.read(_CKPT_HEAD.size)
            if len(raw) != _CKPT_HEAD.size:
                return None
            magic, ver, _np, _h, _w, _c, _nc, _at, _ep, nw = _CKPT_HEAD.unpack(raw)
            if magic != b"UAVRCK01" or ver < 5 or ver > 6 or nw > 64:
                return None
            f.seek(4 * nw + 8, os.SEEK_CUR)      # class_weights + v3 best_macro_f1 + v4 text_dim
            contract = f.read(_CKPT_CONTRACT_BYTES)
    except OSError:
        return None
    if len(contract) != _CKPT_CONTRACT_BYTES:
        return None
    distill_alpha, lr, _seed, _batch = struct.unpack_from("<ffIi", contract, 65)
    ds_sha = contract[:32]
    if not (math.isfinite(distill_alpha) and math.isfinite(lr)) or ds_sha == b"\x00" * 32:
        return None
    return ds_sha.hex()


def find_train_config(ckpt):
    """ckpt 的训练配置（先看同目录续训写出的 config.resume.json，再看首次训练的 config.json；
    先查 ckpt 同目录，再查 tools/average_ckpt.py 在 <ckpt>.json 里记的 run 目录）。返回 (路径, dict)；都没有 (None, None)。"""
    dirs = [os.path.dirname(os.path.abspath(ckpt))]
    sidecar = _read_json(ckpt + ".json")
    if isinstance(sidecar, dict) and isinstance(sidecar.get("run"), str):
        dirs.append(sidecar["run"])
    for d in dirs:
        for name in ("config.resume.json", "config.json"):
            path = os.path.join(d, name)
            cfg = _read_json(path)
            if isinstance(cfg, dict) and isinstance(cfg.get("data"), str):
                return path, cfg
    return None, None


def check_student_cond(requested, ckpt, prev_dim, explicit=False):
    """起跑前闸门：模型训练时的条件类型 vs 评测要喂的 --student-cond。

    返回 {"status": ok|mismatch|unverified, "requested", "trained": none|prev-action|heading-code|None, "source", "message"}。
    - prev_dim == 0（推理进程 hello 的 prev_action_dim）：模型没有条件输入，trained = none；
      要求 heading-code，或 explicit=True 时显式要求 prev-action，均判 mismatch（explicit 默认 False 保持旧调用方兼容）。
    - prev_dim > 0：优先读 <ckpt>.json 的 dataset_cond/data，否则读 config.resume.json/config.json → 数据集 meta 的 cond；
      同时核对 ckpt 头内嵌训练契约的 dataset_sha256 与配置/侧车一致，不一致直接判 mismatch。
    """
    prev_dim = prev_dim if isinstance(prev_dim, int) and not isinstance(prev_dim, bool) else 0

    def result(status, trained, source, message):
        return {"status": status, "requested": requested, "trained": trained, "source": source, "message": message}

    if prev_dim <= 0:
        if requested == COND_HEADING_CODE:
            return result("mismatch", "none", "ckpt prev_dim=0",
                          f"--student-cond {COND_HEADING_CODE}，但 ckpt 没有条件输入（prev_dim=0），传进去的码不会生效；"
                          "要评测航向码学生请用 --prev-action 训出的 ckpt")
        if explicit and requested == COND_PREV_ACTION:
            return result("mismatch", "none", "ckpt prev_dim=0",
                          f"显式指定 --student-cond {COND_PREV_ACTION}，但 ckpt 没有条件输入（prev_dim=0），"
                          "传进去的上一步动作不会生效；评测纯视觉模型请不传 --student-cond，或改用 --prev-action 训出的 ckpt")
        return result("ok", "none", "ckpt prev_dim=0", "ckpt 没有条件输入")

    def finish_with_tag(tag, source):
        if tag is None:
            trained = COND_PREV_ACTION
        elif tag in _TAG_TO_COND:
            trained = _TAG_TO_COND[tag]
        else:
            return result("unverified", None, source, f"数据集 meta 的 cond={tag!r} 认不得（本版本只认 {sorted(_TAG_TO_COND)}）")
        if trained != requested:
            return result("mismatch", trained, source,
                          f"--student-cond {requested} 与模型训练时的条件类型 {trained} 不一致；"
                          "评测端必须喂训练时的同一种条件，否则等于喂了分布外的输入")
        return result("ok", trained, source, f"训练条件 {trained} 与 --student-cond 一致")

    ck_sha = ckpt_dataset_sha256(ckpt)

    # I2-5 第 1 点：优先使用 average_ckpt 在 <ckpt>.json 记录的 dataset_cond 与绝对路径 data，
    # 避免因训练时 --data 为相对路径或原 .uavr 已被归档挪走而退化成 unverified。
    sidecar_path = ckpt + ".json"
    sidecar = _read_json(sidecar_path)
    if isinstance(sidecar, dict):
        if isinstance(sidecar.get("out_sha256"), str) and sidecar["out_sha256"] and os.path.isfile(ckpt):
            if _sha256_file(ckpt) != sidecar["out_sha256"]:
                return result("mismatch", None, sidecar_path,
                              f"{ckpt} 的 sha256 与 {sidecar_path} 的 out_sha256 不一致（权重文件被替换过）")
        sc_want_sha = sidecar.get("data_sha256")
        if ck_sha is not None and isinstance(sc_want_sha, str) and sc_want_sha and sc_want_sha.lower() != ck_sha:
            return result("mismatch", None, sidecar_path,
                          f"{ckpt} 头内训练契约的 dataset_sha256（{ck_sha[:12]}）与 {sidecar_path} 的 "
                          f"data_sha256（{sc_want_sha[:12]}）不一致")
        has_sc_cond, sc_tag = False, None
        if "dataset_cond" in sidecar:
            has_sc_cond, sc_tag = True, sidecar["dataset_cond"]
        elif isinstance(sidecar.get("cond"), dict) and "dataset_cond" in sidecar["cond"]:
            has_sc_cond, sc_tag = True, sidecar["cond"]["dataset_cond"]
        if has_sc_cond:
            sc_src = f"{sidecar_path} -> {sidecar.get('data') or '(dataset_cond)'}"
            return finish_with_tag(sc_tag, sc_src)
        if isinstance(sidecar.get("data"), str) and os.path.isfile(sidecar["data"]):
            sc_data = sidecar["data"]
            sc_src = f"{sidecar_path} -> {sc_data}"
            try:
                sc_tag = dataset_cond_tag(sc_data)
                disk_sha = _sha256_file(sc_data)
                if ck_sha is not None and disk_sha != ck_sha:
                    return result("mismatch", None, sc_src,
                                  f"{ckpt} 头内训练契约的 dataset_sha256（{ck_sha[:12]}）与数据集 {sc_data} 的 "
                                  f"sha256（{disk_sha[:12]}）不一致")
                if sc_want_sha and disk_sha != str(sc_want_sha).lower():
                    return result("unverified", None, sc_src,
                                  f"数据集 {sc_data} 的 sha256 与 {sidecar_path} 记的 {str(sc_want_sha)[:12]} 不同（文件被换过）")
                return finish_with_tag(sc_tag, sc_src)
            except (OSError, ValueError):
                pass

    cfg_path, cfg = find_train_config(ckpt)
    if cfg is None:
        return result("unverified", None, "无",
                      f"{ckpt} 旁边（及 .json 记的 run 目录）没有训练 config.resume.json / config.json，拿不到训练时的条件类型")
    want_sha = cfg.get("dataset_sha256")
    if ck_sha is not None and isinstance(want_sha, str) and want_sha and want_sha.lower() != ck_sha:
        return result("mismatch", None, cfg_path,
                      f"{ckpt} 头内训练契约的 dataset_sha256（{ck_sha[:12]}）与 {cfg_path} 的 "
                      f"dataset_sha256（{want_sha[:12]}）不一致（权重并非该配置训出）")
    data = cfg["data"]
    source = f"{cfg_path} -> {data}"
    try:
        tag = dataset_cond_tag(data)
        disk_sha = _sha256_file(data)
        if ck_sha is not None and not want_sha and disk_sha != ck_sha:
            return result("mismatch", None, source,
                          f"{ckpt} 头内训练契约的 dataset_sha256（{ck_sha[:12]}）与数据集 {data} 的 "
                          f"sha256（{disk_sha[:12]}）不一致")
        if want_sha and disk_sha != str(want_sha).lower():
            return result("unverified", None, source,
                          f"数据集 {data} 的 sha256 与训练时记的 {str(want_sha)[:12]} 不同（文件被换过），meta 里的条件类型不可信")
    except (OSError, ValueError) as exc:
        return result("unverified", None, source, f"读不到训练数据集的 meta：{exc}")
    return finish_with_tag(tag, source)


# ---------------------------------------------------------------- 自检

def self_check():
    """纯函数自检（任务卡 W1-9 的验收例 + 边界）；失败抛 AssertionError。"""
    def need(cond, what):
        if not cond:
            raise AssertionError(f"nav_cond 自检失败：{what}")

    got = heading_codes([FORWARD_SLOW, YAW_LEFT, YAW_LEFT, FORWARD_SLOW, YAW_RIGHT])
    need(got == [0, 0, 0, 2, 2], f"F L L F R 的码应为 0 0 0 2 2，得到 {got}")
    for n, want in ((0, 0), (1, 0), (-1, 0), (2, 2), (5, 2), (-2, 3), (-5, 3)):
        need(heading_code(n) == want, f"heading_code({n}) 应为 {want}")
    need(heading_code(4, setpoint=3) == 0 and heading_code(5, setpoint=3) == 2, "setpoint 要从 n 里减掉")
    need(heading_codes([]) == [] and heading_codes([YAW_LEFT]) == [0], "首帧 n=0 → 码 0")
    need(1 not in {heading_code(n) for n in range(-50, 51)}, "码 1 不用")
    need(heading_codes([YAW_RIGHT] * 3) == [0, 0, 3], "右偏 2 步以上是码 3")


if __name__ == "__main__":
    self_check()
    print("nav_cond: self-check ok")
    sys.exit(0)
