#!/usr/bin/env python3
"""由 sim/dynamics/e1_mini.json 派生一份参数变体（任务卡 W1-10：满电档）。

读源参数，按 --set 的点路径改值，写到 --out，打印改动与新旧 dynamics_profile，最后用采集侧的读法读回自检。
副本只该放数据盘（$D/params/），不进仓库。副本按源文件的文本原样复制，只替换被改的那几个值的文本（`diff 源 副本`
只有改动行），不加任何追溯键；追溯靠 eval.json 的 cf_sitl_params（路径、文件 sha256、values_sha256）与
manifest 的 params_sha256。

用法（仓库根）：
  .venv-tools/bin/python tools/make_params_variant.py \
      --out /root/rivermind-data/uav_rgb/params/e1_mini_soc100.json --set sim.battery_soc_init=1.0
  .venv-tools/bin/python tools/make_params_variant.py --self-check

点路径规则：
  - 找不到的键报错，不新建（打错键名不能静默造出一个没人读的新键）。键名里带 "." 的项
    （如 motor_thrust_full_by_state 的 full_4.20V_oc）不能用点路径寻址。
  - e1_mini.json 的项多是 {"value": ..., "unit": ..., "basis": ...}：路径写到项名就落在它的 value 上；
    项名后面的键在项本身里找不到时，穿过 value 再找。所以 sim.battery_soc_init 与 sim.value.battery_soc_init 等价。
  - 新值是 JSON。类型要与原值一致（数字对数字、字符串对字符串、布尔对布尔、数组长度不变、不接受 null / NaN / Infinity）；
    原值是浮点而新值写成 1 时存成 1.0：JSON 里 1 与 1.0 的指纹不同，同一个数不该有两个指纹。
  - 目标是对象（如 sim、drag_linear）时报错，要写到具体的键。

指纹：与采集侧同一个函数（cf_sitl_exec.params_values_sha256）：各项 value 的规范化 JSON 的 sha256，注释项不算；
dynamics_profile = cf_sitl_v1:<sha8>:lib<N>，N 取 sim/cf_sitl/include/cf_sitl.h 的 CF_SITL_LIB_VERSION。
改 sim.battery_soc_init 0.5→1.0：1bc8fedf → 06e5f017（任务卡 W1-10 给的数；测试锁定）。
"""
import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import cf_sitl_exec  # noqa: E402  指纹只有这一份实现

DEFAULT_SRC = os.path.join(REPO, "sim", "dynamics", "e1_mini.json")
LIB_HEADER = os.path.join(REPO, "sim", "cf_sitl", "include", "cf_sitl.h")
LIB_VERSION_RE = re.compile(r"^\s*#\s*define\s+CF_SITL_LIB_VERSION\s+(\d+)\b", re.M)
OUT_MODE = 0o644   # mkstemp 建的临时文件是 0600；副本要让别的用户（gz 进程可能换用户）也读得到


# ---------------------------------------------------------------- 指纹

def values_sha256(obj):
    return cf_sitl_exec.params_values_sha256(obj)


def read_lib_version(header=LIB_HEADER):
    """库版本号；头文件读不到返回 None（profile 里写 lib?，与采集侧 start() 之前一致）。"""
    try:
        with open(header, encoding="utf-8") as f:
            m = LIB_VERSION_RE.search(f.read())
    except OSError:
        return None
    return int(m.group(1)) if m else None


def profile_of(obj, lib_version):
    return cf_sitl_exec.format_profile(values_sha256(obj), lib_version)


# ---------------------------------------------------------------- 点路径

def _parse_json_value(raw, item):
    def reject_constant(name):
        raise ValueError(f"--set {item!r}: 新值不接受 {name}")
    try:
        return json.loads(raw, parse_constant=reject_constant)
    except json.JSONDecodeError as e:
        raise ValueError(f"--set {item!r}: 新值 {raw!r} 不是 JSON（{e.msg}）；字符串要带引号") from e


def _resolve(obj, path):
    """点路径 → (容器, 键, 规范化后的键列表)。规则见文件头。"""
    parts = path.split(".")
    if not path or any(not p for p in parts):
        raise ValueError(f"--set 的路径 {path!r} 有空段")
    node, trail = obj, []
    container = key = None
    for part in parts:
        if not isinstance(node, dict):
            raise ValueError(f"路径 {path}：{'.'.join(trail)} 不是对象，没有 {part!r}")
        if part not in node:
            inner = node.get("value")
            if isinstance(inner, dict) and part in inner:
                node = inner
                trail.append("value")
            else:
                where = ".".join(trail) or "顶层"
                have = ", ".join(sorted(k for k in node if k != "value")) or "（无）"
                raise ValueError(f"路径 {path}：{where} 下没有 {part!r}；有：{have}")
        container, key = node, part
        trail.append(part)
        node = node[part]
    if isinstance(node, dict) and "value" in node:   # 终点是 {"value": ...} 项：落在它的 value 上
        container, key = node, "value"
        trail.append("value")
    return container, key, trail


def _coerce(old, new, where):
    """新值类型必须与原值一致；返回要写入的值。"""
    if isinstance(old, bool):
        if isinstance(new, bool):
            return new
    elif isinstance(old, (int, float)):
        if isinstance(new, (int, float)) and not isinstance(new, bool):
            if not math.isfinite(new):
                raise ValueError(f"{where}：新值 {new!r} 不是有限数")
            return float(new) if isinstance(old, float) else new
    elif isinstance(old, str):
        if isinstance(new, str):
            return new
    elif isinstance(old, list):
        if isinstance(new, list) and len(new) == len(old):
            return [_coerce(o, n, f"{where}[{i}]") for i, (o, n) in enumerate(zip(old, new))]
        if isinstance(new, list):
            raise ValueError(f"{where}：数组长度 {len(new)} 与原值的 {len(old)} 不同")
    else:
        raise ValueError(f"{where}：目标是 {type(old).__name__}（对象或 null），写到具体的键，例如 drag_linear.xy")
    raise ValueError(f"{where}：原值 {json.dumps(old, ensure_ascii=False)[:60]}（{type(old).__name__}），"
                     f"新值 {json.dumps(new, ensure_ascii=False)[:60]}（{type(new).__name__}）类型不符")


def _apply(obj, sets):
    """返回 (改后的新对象, [(键元组, 旧值, 新值)])。不改传入的对象；按给定顺序逐条应用。"""
    new = copy.deepcopy(obj)
    changes = []
    for item in sets:
        path, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set {item!r}：要写成 路径=JSON值，例如 sim.battery_soc_init=1.0")
        value = _parse_json_value(raw, item)
        container, key, trail = _resolve(new, path.strip())
        old = container[key]
        try:
            fixed = _coerce(old, value, ".".join(trail))
        except ValueError as e:
            raise ValueError(f"--set {item!r}：{e}") from e
        container[key] = fixed
        changes.append((tuple(trail), old, fixed))
    return new, changes


def apply_sets(obj, sets):
    """返回 (改后的新对象, [(规范点路径, 旧值, 新值)])；同一路径写两次两条都记。"""
    new, changes = _apply(obj, sets)
    return new, [(".".join(t), o, n) for t, o, n in changes]


# ---------------------------------------------------------------- 文本定位：只替换被改的值，其余字节原样

def _skip_ws(text, i):
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _scan_string(text, i):
    """text[i] 是开引号；返回闭引号之后的位置。"""
    i += 1
    while text[i] != '"':
        i += 2 if text[i] == "\\" else 1
    return i + 1


def _scan_value(text, i, path, spans):
    """从 text[i] 起扫一个 JSON 值，返回结束位置。path 不是 None 时，把对象成员的 (起, 止) 记进 spans[path + (键,)]；
    数组里面的东西不记（数组按叶子整体替换）。文本已经过 json.loads 校验，这里不再查语法。"""
    i = _skip_ws(text, i)
    c = text[i]
    if c == "{":
        i = _skip_ws(text, i + 1)
        if text[i] == "}":
            return i + 1
        while True:
            end_key = _scan_string(text, i)
            key = json.loads(text[i:end_key])
            i = _skip_ws(text, _skip_ws(text, end_key) + 1)   # 跳过冒号
            end = _scan_value(text, i, None if path is None else path + (key,), spans)
            if path is not None:
                spans[path + (key,)] = (i, end)
            i = _skip_ws(text, end)
            if text[i] == "}":
                return i + 1
            i = _skip_ws(text, i + 1)                         # 跳过逗号
    if c == "[":
        i = _skip_ws(text, i + 1)
        if text[i] == "]":
            return i + 1
        while True:
            i = _skip_ws(text, _scan_value(text, i, None, spans))
            if text[i] == "]":
                return i + 1
            i = _skip_ws(text, i + 1)
    if c == '"':
        return _scan_string(text, i)
    j = i
    while j < len(text) and text[j] not in ",]} \t\r\n":
        j += 1
    return j


def value_spans(text):
    """{键元组: (起, 止)}：每个对象成员的值在 text 里的文本区间。"""
    spans = {}
    _scan_value(text, 0, (), spans)
    return spans


def splice_values(text, src_obj, changes):
    """把 changes（_apply 的返回）里每个路径的最终值写进 text 的对应区间；写回原值的路径不动。返回新文本。"""
    spans = value_spans(text)
    final = {}
    for trail, old, new in changes:
        final.setdefault(trail, [old, None])[1] = new
    edits = []
    for trail, (old, new) in final.items():
        if _same(old, new):
            continue
        start, end = spans[trail]
        if not _same(json.loads(text[start:end]), old):
            raise RuntimeError(f"定位错位：{'.'.join(trail)} 的文本 {text[start:end][:40]!r} 解析后不是原值 {old!r}")
        edits.append((start, end, json.dumps(new, ensure_ascii=False)))
    for start, end, rep in sorted(edits, reverse=True):
        text = text[:start] + rep + text[end:]
    return text


# ---------------------------------------------------------------- 写出与读回自检

def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _same(a, b):
    """叶子相等且类型相同：1 与 1.0 的指纹不同，算不同。"""
    return json.dumps(a, sort_keys=True, ensure_ascii=True) == json.dumps(b, sort_keys=True, ensure_ascii=True)


def _leaf_diffs(a, b, prefix=""):
    """逐键比较两个 JSON 对象：[(路径, a 的值, b 的值)]。对象逐层往下，数组与标量当叶子。"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{prefix}{k}", a.get(k, "<缺>"), b.get(k, "<缺>")))
            else:
                out += _leaf_diffs(a[k], b[k], f"{prefix}{k}.")
        return out
    return [] if _same(a, b) else [(prefix[:-1], a, b)]


def verify_written(path, src_obj, new_obj, changes):
    """用采集侧的读法（CfSitlParams.from_json）读回 path：内容与内存里的新对象一致、指纹一致，
    与源对象逐叶比较的差别正好是 changes 里每个路径的「最终值」。失败抛 RuntimeError。changes 是 apply_sets 的返回形式。"""
    p = cf_sitl_exec.CfSitlParams.from_json(path)
    if p.obj != new_obj:
        raise RuntimeError(f"读回的内容与内存里的新对象不同：{path}")
    if p.values_sha256 != values_sha256(new_obj):
        raise RuntimeError(f"读回的 values_sha256 {p.values_sha256[:8]} 与内存 {values_sha256(new_obj)[:8]} 不同")
    final = {}
    for canon, old, new in changes:
        final.setdefault(canon, [old, None])[1] = new
    expect = sorted((c, o, n) for c, (o, n) in final.items() if not _same(o, n))
    got = sorted(_leaf_diffs(src_obj, p.obj))
    if got != expect:
        raise RuntimeError(f"读回后与源文件的差别 {got} 不等于指定的改动 {expect}")
    return p


def make_variant(src, out, sets, force=False, lib_version="auto"):
    """读 src，应用 sets，写 out（先写同目录临时文件，读回自检通过再改名）。返回信息字典。
    out 不能是 src（含符号链接指回）；out 已存在时要 force。"""
    if lib_version == "auto":
        lib_version = read_lib_version()
    raw = _read_bytes(src)
    text = raw.decode("utf-8")
    src_obj = json.loads(text)
    if not isinstance(src_obj, dict):
        raise ValueError(f"{src}: 顶层必须是对象")
    new_obj, trail_changes = _apply(src_obj, sets)
    changes = [(".".join(t), o, n) for t, o, n in trail_changes]
    if os.path.realpath(out) == os.path.realpath(src):
        raise ValueError(f"--out 与源文件是同一个文件：{src}")
    if os.path.lexists(out) and not force:
        raise ValueError(f"--out 已存在：{out}（要覆盖加 --force）")
    out_text = splice_values(text, src_obj, trail_changes)
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".mpv-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(out_text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, OUT_MODE)
        p = verify_written(tmp, src_obj, new_obj, changes)
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return {
        "src": os.path.abspath(src), "out": os.path.abspath(out), "changes": changes,
        "src_sha256": hashlib.sha256(raw).hexdigest(), "sha256": p.sha256,
        "src_values_sha256": values_sha256(src_obj), "values_sha256": p.values_sha256,
        "src_profile": profile_of(src_obj, lib_version), "profile": profile_of(new_obj, lib_version),
        "lib_version": lib_version,
    }


# ---------------------------------------------------------------- 自检（不读仓库里的参数文件）

def self_check():
    """合成一份小参数，走完 点路径 → 文本替换 → 写出 → 读回 → 指纹。返回 0；任一条不成立抛 AssertionError。"""
    src_text = (
        '{\n'
        '  "schema": "synthetic.v1",\n'
        '  "mass": {"value": 0.1, "unit": "kg", "basis": "注释 \\"带引号\\" {大括号}, 逗号: 冒号 \\\\"},\n'
        '  "curve": {"value": [[1.0, 4.2], [0.5, 3.8]], "unit": "V"},\n'
        '  "drag": {"value": {"xy": 0.04, "z": 0.02}},\n'
        '  "rate": {"value": 25},\n'
        '  "sim": {\n'
        '    "value": {"dt_s": 0.001, "battery_soc_init": 0.5},\n'
        '    "unit": "-"\n'
        '  },\n'
        '  "note": "没有 value 的顶层项不进指纹"\n'
        '}\n')
    base = json.loads(src_text)
    new, ch = apply_sets(base, ["sim.battery_soc_init=1"])
    assert ch == [("sim.value.battery_soc_init", 0.5, 1.0)] and isinstance(new["sim"]["value"]["battery_soc_init"], float)
    assert base["sim"]["value"]["battery_soc_init"] == 0.5, "输入对象被改了"
    assert apply_sets(base, ["sim.value.battery_soc_init=1.0"])[0] == new, "两种写法应等价"
    assert apply_sets(base, ["drag.xy=0.05"])[1] == [("drag.value.xy", 0.04, 0.05)]
    rate30 = apply_sets(base, ["rate=30"])[0]["rate"]["value"]
    assert rate30 == 30 and isinstance(rate30, int), "原值是整数就保持整数"
    assert apply_sets(base, ["curve=[[1.0,4.2],[0.5,3.9]]"])[0]["curve"]["value"][1][1] == 3.9
    for bad in ("sim.nope=1", "nope=1", "sim.battery_soc_init=true", 'sim.battery_soc_init="x"', "sim.battery_soc_init=NaN",
                "sim.battery_soc_init=[1]", "sim=1", "drag=1", "curve=[[1.0]]", "mass=null", "mass=", "mass", "..=1"):
        try:
            apply_sets(base, [bad])
        except ValueError:
            continue
        raise AssertionError(f"应当拒绝：--set {bad}")
    commented = copy.deepcopy(base)
    commented["mass"]["basis"] = "换一句注释"
    commented["note"] = "换一句"
    assert values_sha256(commented) == values_sha256(base), "注释不该进指纹"
    assert values_sha256(new) != values_sha256(base), "改了 value 指纹应变"
    sets = ["sim.battery_soc_init=1.0", "rate=30", "curve=[[1.0,4.2],[0.5,3.9]]"]
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "src.json"), os.path.join(d, "out.json")
        with open(src, "w", encoding="utf-8", newline="") as f:
            f.write(src_text)
        before = _read_bytes(src)
        info = make_variant(src, out, sets, lib_version=31)
        assert _read_bytes(src) == before, "源文件被改了"
        expect_text = (src_text.replace('"battery_soc_init": 0.5', '"battery_soc_init": 1.0')
                       .replace('"value": 25}', '"value": 30}')
                       .replace("[[1.0, 4.2], [0.5, 3.8]]", "[[1.0, 4.2], [0.5, 3.9]]"))
        assert _read_bytes(out).decode("utf-8") == expect_text, "除被改的三处值，其余字节应与源文件相同"
        assert info["values_sha256"] == values_sha256(apply_sets(base, sets)[0])
        assert info["profile"] == f"cf_sitl_v1:{values_sha256(apply_sets(base, sets)[0])[:8]}:lib31"
        for args in ((src, src, []), (src, out, [])):
            try:
                make_variant(*args)
            except ValueError:
                continue
            raise AssertionError("应当拒绝覆盖源文件 / 已有输出")
        bad_out = os.path.join(d, "bad.json")
        try:
            make_variant(src, bad_out, ["sim.nope=1"])
        except ValueError:
            pass
        assert not os.path.exists(bad_out) and sorted(os.listdir(d)) == ["out.json", "src.json"], "失败不该留文件"
        noop = os.path.join(d, "noop.json")
        info0 = make_variant(src, noop, [])
        assert _read_bytes(noop) == before and info0["sha256"] == info0["src_sha256"], "没有改动的副本应与源文件逐字节相同"
    print("make_params_variant self-check PASS")
    return 0


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=DEFAULT_SRC, help="源参数文件（缺省 sim/dynamics/e1_mini.json）")
    p.add_argument("--out", help="输出路径（放数据盘 $D/params/，不进仓库）")
    p.add_argument("--set", action="append", default=[], metavar="PATH=JSON",
                   help="点路径=JSON 值，可重复；例如 sim.battery_soc_init=1.0")
    p.add_argument("--force", action="store_true", help="--out 已存在时覆盖（源文件仍不许覆盖）")
    p.add_argument("--self-check", action="store_true", help="合成数据自检后退出（不读仓库参数文件）")
    args = p.parse_args(argv)
    if args.self_check:
        try:
            return self_check()
        except AssertionError as e:
            print(f"make_params_variant self-check FAIL: {e}", file=sys.stderr)
            return 1
    if not args.out:
        p.error("缺 --out")
    try:
        info = make_variant(args.src, args.out, args.set, force=args.force)
    except (ValueError, OSError, RuntimeError) as e:
        p.error(str(e))
    print(f"源文件  : {info['src']}")
    print(f"  文件字节 sha256 = {info['src_sha256']}")
    print(f"  dynamics_profile = {info['src_profile']}")
    print(f"改动    : {len(info['changes'])} 项")
    for canon, old, new in info["changes"]:
        print(f"  {canon}: {json.dumps(old, ensure_ascii=False)} -> {json.dumps(new, ensure_ascii=False)}")
    print(f"输出    : {info['out']}")
    print(f"  文件字节 sha256 = {info['sha256']}")
    print(f"  dynamics_profile = {info['profile']}")
    print("自检    : PASS（按采集侧读法读回：内容与内存一致，与源文件的差别只有上述改动，指纹一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
