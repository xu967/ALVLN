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


def find_train_config(ckpt):
    """ckpt 的训练配置（trainer 的 config.json，有 data 路径与 dataset_sha256）：先看 ckpt 同目录，
    再看 tools/average_ckpt.py 在 <ckpt>.json 里记的 run 目录。返回 (路径, dict)；都没有 (None, None)。"""
    candidates = [os.path.join(os.path.dirname(os.path.abspath(ckpt)), "config.json")]
    sidecar = _read_json(ckpt + ".json")
    if isinstance(sidecar, dict) and isinstance(sidecar.get("run"), str):
        candidates.append(os.path.join(sidecar["run"], "config.json"))
    for path in candidates:
        cfg = _read_json(path)
        if isinstance(cfg, dict) and isinstance(cfg.get("data"), str):
            return path, cfg
    return None, None


def check_student_cond(requested, ckpt, prev_dim):
    """起跑前闸门：模型训练时的条件类型 vs 评测要喂的 --student-cond。

    返回 {"status": ok|mismatch|unverified, "requested", "trained": none|prev-action|heading-code|None, "source", "message"}。
    - prev_dim == 0（推理进程 hello 的 prev_action_dim）：模型没有条件输入，trained = none；要求 heading-code 就是 mismatch。
    - prev_dim > 0：条件类型只能从训练数据读（trainer 的 config.json → 数据集 meta 的 cond）：有 heading_code.v1 标签 = heading-code，
      没有 cond 键 = prev-action（旧打包就是这个语义）。读不到 / 数据集被换过（sha256 不符）/ 标签认不得 = unverified，不猜。
    """
    prev_dim = prev_dim if isinstance(prev_dim, int) and not isinstance(prev_dim, bool) else 0

    def result(status, trained, source, message):
        return {"status": status, "requested": requested, "trained": trained, "source": source, "message": message}

    if prev_dim <= 0:
        if requested == COND_HEADING_CODE:
            return result("mismatch", "none", "ckpt prev_dim=0",
                          f"--student-cond {COND_HEADING_CODE}，但 ckpt 没有条件输入（prev_dim=0），传进去的码不会生效；"
                          "要评测航向码学生请用 --prev-action 训出的 ckpt")
        return result("ok", "none", "ckpt prev_dim=0", "ckpt 没有条件输入")
    cfg_path, cfg = find_train_config(ckpt)
    if cfg is None:
        return result("unverified", None, "无",
                      f"{ckpt} 旁边（及 .json 记的 run 目录）没有训练 config.json，拿不到训练时的条件类型")
    data = cfg["data"]
    source = f"{cfg_path} -> {data}"
    try:
        tag = dataset_cond_tag(data)
        want_sha = cfg.get("dataset_sha256")
        if want_sha and _sha256_file(data) != want_sha:
            return result("unverified", None, source,
                          f"数据集 {data} 的 sha256 与训练时记的 {str(want_sha)[:12]} 不同（文件被换过），meta 里的条件类型不可信")
    except (OSError, ValueError) as exc:
        return result("unverified", None, source, f"读不到训练数据集的 meta：{exc}")
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
