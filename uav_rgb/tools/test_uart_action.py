#!/usr/bin/env python3
"""tools/uart_action.py 自测：CRC、帧编码、动作越界、心跳/seq、dry-run、坏配置、串口缺失。

不打开真实串口：串口路径用例只用不存在的设备，或 mock os.open 抛 PermissionError。
"""
import json
import os
import re
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().with_name("uart_action.py")
EXAMPLE_CONFIG = ROOT / "config" / "uart_protocol.example.json"
CONFIG = EXAMPLE_CONFIG  # main() 中指向旧版 9B 配置夹具，保证既有单帧编码测试向量断言逐字不变
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uart_action as ua  # noqa: E402

LEGACY_V1_CONFIG = {
    "placeholder": True,
    "device": "/dev/ttyS1",
    "baud": 115200,
    "heartbeat_hz": 10,
    "ttl_ms": 300,
    "heartbeat_action": 0,
    "action_encoding": [0, 1, 2, 3],
    "crc": {"algorithm": "crc8", "poly": 7, "init": 0, "xorout": 0, "reflect": False},
    "frame": [
        {"name": "magic", "kind": "const", "bytes": [170, 85]},
        {"name": "version", "kind": "const", "size": 1, "value": 1},
        {"name": "seq", "kind": "seq", "size": 1, "endian": "little"},
        {"name": "action", "kind": "action", "size": 1, "endian": "little"},
        {"name": "heartbeat", "kind": "heartbeat", "size": 1, "endian": "little"},
        {"name": "ttl", "kind": "ttl", "size": 2, "endian": "little"},
        {"name": "crc", "kind": "crc", "size": 1, "endian": "little"},
    ],
}

EXPECTED = {
    0: "aa 55 01 00 00 00 2c 01 2f",
    1: "aa 55 01 00 01 00 2c 01 39",
    2: "aa 55 01 00 02 00 2c 01 03",
    3: "aa 55 01 00 03 00 2c 01 15",
}
HEARTBEAT_SEQ0 = "aa 55 01 00 00 01 2c 01 44"
ACTION1_SEQ5 = "aa 55 01 05 01 00 2c 01 d4"


def run_cli(args, stdin=""):
    return subprocess.run([sys.executable, str(SCRIPT), *args], input=stdin,
                          capture_output=True, text=True, timeout=20)


def run_cli_open(args, stdin_bytes=b""):
    """保持 stdin 管道打开直到子进程因 --frames 退出（模拟常驻上游连接）。"""
    p = subprocess.Popen(
        [sys.executable, str(SCRIPT), *args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        if stdin_bytes:
            os.write(p.stdin.fileno(), stdin_bytes)
        out, err = p.stdout.read(), p.stderr.read()
        rc = p.wait(timeout=20)
    finally:
        for stream in (p.stdin, p.stdout, p.stderr):
            if stream:
                stream.close()
    return subprocess.CompletedProcess(p.args, rc, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"))


def raw_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def expect_bad(temp, mutate, needle):
    cfg = raw_config()
    mutate(cfg)
    path = temp / "bad.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    try:
        ua.load_config(path)
    except ua.ProtocolError as exc:
        assert needle in str(exc), (needle, str(exc))
    else:
        raise AssertionError(f"坏配置未被拒绝: {needle}")


def test_crc():
    # 标准校验向量，防止 CRC 实现自洽但算法错。
    assert ua.crc8(b"123456789", 0x07, 0x00, 0x00, False) == 0xF4
    assert ua.crc16(b"123456789", 0x1021, 0xFFFF, 0x0000, False) == 0x29B1
    assert ua.crc16(b"123456789", 0x1021, 0x0000, 0x0000, False) == 0x31C3


def test_default_config_and_encode():
    cfg = ua.load_config(CONFIG)
    assert [f["name"] for f in cfg["frame"]] == [
        "magic", "version", "seq", "action", "heartbeat", "ttl", "crc"]
    assert cfg["placeholder"] is True
    for action, expected in EXPECTED.items():
        frame = ua.encode_frame(cfg, action, 0, False)
        assert frame.hex(" ") == expected, (action, frame.hex(" "))
        assert frame[-1] == ua.crc8(frame[:-1], 0x07, 0, 0, False)
    assert ua.encode_frame(cfg, 1, 5, False).hex(" ") == ACTION1_SEQ5
    # 心跳帧：heartbeat=1，action 字段固定写 heartbeat_action，不沿用调用参数。
    heartbeat = ua.encode_frame(cfg, 3, 0, True)
    assert heartbeat.hex(" ") == HEARTBEAT_SEQ0
    assert heartbeat[4:6] == bytes([0, 1])


def test_seq_helpers():
    assert [ua.next_seq(0, 1), ua.next_seq(1, 1), ua.next_seq(255, 1)] == [1, 2, 0]
    assert ua.next_seq(65535, 2) == 0
    for bad in (-1, 4, 99):
        try:
            ua.encode_frame(ua.load_config(CONFIG), bad, 0, False)
        except ua.ProtocolError as exc:
            assert "动作越界" in str(exc)
        else:
            raise AssertionError(f"动作越界未拒绝: {bad}")
    for bad in ("4", "-1", "left", ""):
        try:
            ua.parse_action(bad)
        except ua.ProtocolError:
            pass
        else:
            raise AssertionError(f"非法动作输入未拒绝: {bad!r}")


def test_crc16_frame():
    cfg = raw_config()
    cfg["crc"] = {"algorithm": "crc16", "poly": 0x1021, "init": 0xFFFF,
                  "xorout": 0, "reflect": False}
    cfg["frame"][-1].update({"size": 2, "endian": "big"})
    cfg = ua.validate_config(cfg)
    frame = ua.encode_frame(cfg, 2, 7, False)
    assert frame[-2:] == ua.crc16(frame[:-2], 0x1021, 0xFFFF, 0, False).to_bytes(2, "big")


def test_bad_config(temp):
    expect_bad(temp, lambda c: c["frame"].pop(5), "缺少 ttl")
    expect_bad(temp, lambda c: c["action_encoding"].pop(), "4 个元素")
    expect_bad(temp, lambda c: c["frame"][3].update(kind="pwm"), "不支持")
    expect_bad(temp, lambda c: c["crc"].update(algorithm="crc32"), "crc8 或 crc16")
    expect_bad(temp, lambda c: c["frame"][-1].update(size=2), "CRC 位宽")
    expect_bad(temp, lambda c: c.update(baud=12345), "不支持波特率")
    expect_bad(temp, lambda c: c.update(ttl_ms=70000), "ttl_ms 必须是")
    expect_bad(temp, lambda c: c.update(heartbeat_hz=0), "正数")
    expect_bad(temp, lambda c: c["frame"][0]["bytes"].__setitem__(0, 256), "0..255")

    def crc_not_last(c):
        crc = c["frame"].pop()
        c["frame"].insert(4, crc)
    expect_bad(temp, crc_not_last, "最后一个字段")

    path = temp / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    try:
        ua.load_config(path)
    except ua.ProtocolError as exc:
        assert "合法 JSON" in str(exc)
    else:
        raise AssertionError("坏 JSON 未被拒绝")


def test_cli_dry_run(temp):
    p = run_cli(["--dry-run", "--config", str(CONFIG), "--action", "1"])
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert p.stdout.splitlines() == [EXPECTED[1]]
    assert "占位帧协议" in p.stderr

    # dry-run 必须完全跳过串口：设备不存在也应成功。
    cfg = raw_config()
    cfg["device"] = str(temp / "no-such-uart")
    path = temp / "dry.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    p = run_cli(["--dry-run", "--config", str(path), "--action", "2"])
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert p.stdout.splitlines() == [EXPECTED[2]]

    p = run_cli(["--dry-run", "--config", str(CONFIG), "--action", "4"])
    assert p.returncode == 2 and "动作越界" in p.stderr
    p = run_cli(["--dry-run", "--config", str(CONFIG), "--frames", "0"])
    assert p.returncode == 2 and "--frames" in p.stderr


def test_cli_heartbeat():
    p = run_cli_open(["--dry-run", "--config", str(CONFIG), "--frames", "3"])
    assert p.returncode == 0, (p.stdout, p.stderr)
    frames = [bytes.fromhex(line) for line in p.stdout.splitlines()]
    assert len(frames) == 3
    assert [f[3] for f in frames] == [0, 1, 2]       # seq 逐帧 +1
    assert [f[4] for f in frames] == [0, 0, 0]       # 心跳动作固定 SAFE_NEUTRAL
    assert [f[5] for f in frames] == [1, 1, 1]       # heartbeat 标志
    for frame in frames:
        assert frame[-1] == ua.crc8(frame[:-1], 0x07, 0, 0, False)


def test_cli_stdin_actions():
    p = run_cli_open(["--dry-run", "--config", str(CONFIG), "--frames", "5"], stdin_bytes=b"1\n3\n")
    assert p.returncode == 0, (p.stdout, p.stderr)
    frames = [bytes.fromhex(line) for line in p.stdout.splitlines()]
    assert len(frames) == 5
    assert [f[3] for f in frames] == [0, 1, 2, 3, 4]
    actions = [f[4] for f in frames if f[5] == 0]
    assert actions == [1, 3], actions

    p = run_cli(["--dry-run", "--config", str(CONFIG), "--frames", "100"], stdin="7\n")
    assert p.returncode == 2 and "动作越界" in p.stderr


def test_serial_errors(temp):
    cfg = raw_config()
    cfg["device"] = str(temp / "definitely-no-such-uart")
    path = temp / "serial.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    p = run_cli(["--config", str(path), "--action", "0"])
    assert p.returncode == 2, (p.stdout, p.stderr)
    assert "串口不存在" in p.stderr
    assert "Traceback" not in p.stderr

    cfg = ua.load_config(CONFIG)
    with mock.patch.object(ua.os, "open", side_effect=PermissionError(13, "Permission denied")):
        try:
            ua.open_serial(cfg)
        except ua.ProtocolError as exc:
            assert "串口无权限" in str(exc)
        else:
            raise AssertionError("权限不足未报错")

    cfg = ua.load_config(CONFIG)
    cfg["device"] = ""
    try:
        ua.open_serial(cfg)
    except ua.ProtocolError as exc:
        assert "缺少 device" in str(exc)
    else:
        raise AssertionError("缺少 device 未报错")

    p = run_cli(["--dry-run", "--config", str(temp / "missing.json"), "--action", "0"])
    assert p.returncode == 2 and "读取配置失败" in p.stderr


def _read_hex_frames_until(fd, predicate, deadline_s):
    """非阻塞读子进程 stdout hex 行，直到 predicate(frames) 为真或到达 deadline_s。"""
    end = time.monotonic() + deadline_s
    buf = bytearray()
    frames = []
    while not predicate(frames):
        rem = end - time.monotonic()
        if rem <= 0:
            break
        ready, _, _ = select.select([fd], [], [], min(rem, 0.05))
        if not ready:
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        buf.extend(chunk)
        while (nl := buf.find(b"\n")) >= 0:
            line = bytes(buf[:nl]).decode("utf-8", "replace").strip()
            del buf[:nl + 1]
            if line:
                frames.append(bytes.fromhex(line))
    return frames


def test_n1a_partial_line_heartbeat():
    """N-1a：上游写出无换行的半行 b'2' 时心跳不能停摆，补写 b'\\n' 后才发出动作 2。"""
    cfg = ua.load_config(CONFIG)

    # 1) 注入时钟确定性检验：半行不触发动作、不阻塞心跳；拼成 '12\\n' 时按整行拒绝而非拆成 1 和 2；超长无换行拒绝
    t_now = [0.0]
    events = [(0.02, b"1"), (0.35, b"2\n")]
    emitted = []

    def fake_poll(timeout, _open):
        if events and events[0][0] <= t_now[0] + timeout + 1e-9:
            ev_t, data = events.pop(0)
            t_now[0] = max(t_now[0], ev_t)
            return data
        t_now[0] += timeout
        return None

    try:
        ua.run_stream(cfg, emitted.append, max_frames=10, clock_fn=lambda: t_now[0], poll_fn=fake_poll)
    except ua.ProtocolError as exc:
        assert "越界" in str(exc), exc
    else:
        raise AssertionError("分两次到达的 '1' + '2\\n' 拼成 '12' 未被拒绝")
    assert all(f[5] == 1 for f in emitted) and len(emitted) >= 3, emitted

    try:
        ua.run_stream(cfg, [].append, max_frames=2, clock_fn=lambda: 0.0,
                      poll_fn=lambda _t, _o: b"1" * (ua.MAX_LINE_BYTES + 1))
    except ua.ProtocolError as exc:
        assert "缺少换行符" in str(exc), exc
    else:
        raise AssertionError("超长无换行残行未被拒绝")

    # 2) 真实子进程管道复现（探针 Case 4，放宽余量防负载抖动）
    p = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--dry-run", "--config", str(CONFIG)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    os.set_blocking(p.stdout.fileno(), False)
    try:
        init = _read_hex_frames_until(p.stdout.fileno(), lambda fs: len(fs) >= 1, 5.0)
        assert len(init) >= 1, "启动后未收到首帧心跳"
        os.write(p.stdin.fileno(), b"2")
        mid = _read_hex_frames_until(
            p.stdout.fileno(),
            lambda fs: sum(1 for f in fs if f[5] == 1) >= 2,
            1.5,
        )
        hb_mid = [f for f in mid if f[5] == 1]
        act_mid = [f for f in mid if f[5] == 0]
        assert len(hb_mid) >= 2, f"半行阻塞了心跳调度，1.5s 内仅收到 {len(hb_mid)} 帧心跳"
        assert act_mid == [], f"半行尚未结束就提前发出了动作帧: {act_mid}"
        os.write(p.stdin.fileno(), b"\n")
        after = _read_hex_frames_until(
            p.stdout.fileno(),
            lambda fs: any(f[5] == 0 for f in fs),
            3.0,
        )
        acts = [f[4] for f in after if f[5] == 0]
        assert acts == [2], f"补齐换行后期望发出动作 [2]，实得 {acts}"
    finally:
        p.kill()
        p.wait()


def test_n1b_multiline_burst_in_order():
    """N-1b：常驻管道一次写入多行（不关闭 stdin）时，同一批所有动作按到达顺序立即全发，不被滞留到后续心跳周期。"""
    cfg = ua.load_config(CONFIG)

    # 1) 注入时钟确定性检验：t=0.02s 一次到达 b'1\\n3\\n2\\n'（远早于下次心跳 0.10s），三帧动作必须在 t=0.02s 当拍按序全发
    t_now = [0.0]
    events = [(0.02, b"1\n3\n2\n")]
    timeline = []

    def fake_poll(timeout, _open):
        if events and events[0][0] <= t_now[0] + timeout + 1e-9:
            ev_t, data = events.pop(0)
            t_now[0] = max(t_now[0], ev_t)
            return data
        t_now[0] += timeout
        return None

    ua.run_stream(
        cfg,
        lambda f: timeline.append((round(t_now[0], 3), f[3], f[4], f[5])),
        max_frames=5,
        clock_fn=lambda: t_now[0],
        poll_fn=fake_poll,
    )
    act_events = [(t, act) for t, _seq, act, hb in timeline if hb == 0]
    assert [a for _, a in act_events] == [1, 3, 2], f"多行动作未按到达顺序全发: {timeline}"
    assert all(t == 0.02 for t, _ in act_events), f"多行动作被滞留到后续心跳周期才发出: {timeline}"

    # 2) 真实子进程常驻管道复现（探针 Case 1；设 heartbeat_hz=0.5 即周期 2.0s，留足 1.2s 余量防满载抢核误报）
    slow_cfg = raw_config()
    slow_cfg["heartbeat_hz"] = 0.5
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(slow_cfg, tf)
        slow_path = tf.name
    p = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--dry-run", "--config", slow_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    os.set_blocking(p.stdout.fileno(), False)
    try:
        init = _read_hex_frames_until(p.stdout.fileno(), lambda fs: len(fs) >= 1, 5.0)
        assert len(init) >= 1
        os.write(p.stdin.fileno(), b"1\n3\n2\n")
        got = _read_hex_frames_until(
            p.stdout.fileno(),
            lambda fs: sum(1 for f in fs if f[5] == 0) >= 3,
            1.2,
        )
        acts = [f[4] for f in got if f[5] == 0]
        assert acts == [1, 3, 2], f"常驻管道一次写入三行期望立即按序发出 [1, 3, 2]，实得 {acts}"
    finally:
        p.kill()
        p.wait()
        os.unlink(slow_path)


def test_n1c_eof_and_upstream_timeout():
    """N-1c：上游 EOF 时发完已缓冲行、补发一帧 SAFE_NEUTRAL 后立即退出（不继续发心跳屏蔽飞控 TTL）；支持 --upstream-timeout-ms。"""
    # 1) 一次性灌入几行后立即 EOF（不传 --frames 也不会死循环）：按序发出 1, 3，补一帧 SAFE_NEUTRAL(0) 后 rc=0 退出
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--config", str(CONFIG)],
        input="1\n3\n", capture_output=True, text=True, timeout=3,
    )
    assert p.returncode == 0, (p.stdout, p.stderr)
    frames = [bytes.fromhex(line) for line in p.stdout.splitlines() if line.strip()]
    act_frames = [f[4] for f in frames if f[5] == 0]
    assert act_frames == [1, 3, 0], f"EOF 前应发完 [1, 3] 并补发一帧 SAFE_NEUTRAL(0)，实得动作帧 {act_frames}"
    assert frames[-1][4] == 0 and frames[-1][5] == 0, f"末帧必须是 EOF 补发的 SAFE_NEUTRAL 动作帧，实得 {frames}"

    # 2) 半行处异常断开（无换行即 EOF）：仍须先发一帧 SAFE_NEUTRAL 保安全，再以退出码 2 报错
    p_half = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--config", str(CONFIG)],
        input="2", capture_output=True, text=True, timeout=3,
    )
    assert p_half.returncode == 2, (p_half.stdout, p_half.stderr)
    half_frames = [bytes.fromhex(line) for line in p_half.stdout.splitlines() if line.strip()]
    half_acts = [f[4] for f in half_frames if f[5] == 0]
    assert half_acts == [0] and half_frames[-1][5] == 0, half_frames

    # 3) --upstream-timeout-ms（注入时钟确定性检验）：上游不关管道但超时未发新动作，主动发 SAFE_NEUTRAL 并写告警；恢复后再超时仍生效
    cfg = ua.load_config(CONFIG)
    t_now = [0.0]
    events = [(0.05, b"2\n"), (0.60, b"1\n"), (0.95, b"")]
    timeline = []
    warnings = []

    def fake_poll(timeout, _open):
        if events and events[0][0] <= t_now[0] + timeout + 1e-9:
            ev_t, data = events.pop(0)
            t_now[0] = max(t_now[0], ev_t)
            return data
        t_now[0] += timeout
        return None

    rc = ua.run_stream(
        cfg,
        lambda f: timeline.append((round(t_now[0], 3), f[4], f[5])),
        upstream_timeout_ms=250,
        clock_fn=lambda: t_now[0],
        poll_fn=fake_poll,
        warn_fn=warnings.append,
    )
    assert rc == 0
    acts_timed = [(t, act) for t, act, hb in timeline if hb == 0]
    # t=0.05 动作 2 -> t=0.30 超时发 SAFE(0) -> t=0.60 恢复动作 1 -> t=0.85 再次超时发 SAFE(0) -> t=0.95 EOF 发 SAFE(0)
    assert acts_timed == [(0.05, 2), (0.3, 0), (0.6, 1), (0.85, 0), (0.95, 0)], acts_timed
    assert len(warnings) == 2 and all("250 ms" in w for w in warnings), warnings

    p_bad = run_cli(["--dry-run", "--config", str(CONFIG), "--upstream-timeout-ms", "-1"], stdin="")
    assert p_bad.returncode == 2 and "upstream-timeout-ms" in p_bad.stderr


def test_n1d_no_same_round_heartbeat_and_duration(temp):
    """N-1d：动作帧发出后同轮不再追发心跳并重置心跳计时；支持可选动作有效时长字段 duration。"""
    cfg = ua.load_config(CONFIG)

    # 1) 注入时钟检验（探针 Case 3）：t=0.0 已有动作 '1\\n'，同轮绝不紧跟心跳帧，心跳推迟整整一个周期（t=0.10s）
    t_now = [0.0]
    events = [(0.0, b"1\n"), (0.16, b"2\n")]
    timeline = []

    def fake_poll(timeout, _open):
        if events and events[0][0] <= t_now[0] + timeout + 1e-9:
            ev_t, data = events.pop(0)
            t_now[0] = max(t_now[0], ev_t)
            return data
        t_now[0] += timeout
        return None

    ua.run_stream(
        cfg,
        lambda f: timeline.append((round(t_now[0], 3), f[3], f[4], f[5])),
        max_frames=4,
        clock_fn=lambda: t_now[0],
        poll_fn=fake_poll,
    )
    # 期望：t=0.0 发动作1(hb=0) -> t=0.1 发心跳(hb=1) -> t=0.16 发动作2(hb=0) -> t=0.26 发心跳(hb=1)
    assert timeline == [
        (0.0, 0, 1, 0),
        (0.1, 1, 0, 1),
        (0.16, 2, 2, 0),
        (0.26, 3, 0, 1),
    ], f"动作帧后未推迟心跳或同轮双发: {timeline}"

    # 一次性灌入 '1\\n3\\n' 后 EOF：全程不应夹杂任何同轮心跳帧
    p = run_cli(["--dry-run", "--config", str(CONFIG)], stdin="1\n3\n")
    assert p.returncode == 0
    oneshot = [bytes.fromhex(l) for l in p.stdout.splitlines() if l.strip()]
    assert [(f[4], f[5]) for f in oneshot] == [(1, 0), (3, 0), (0, 0)], oneshot

    # 2) 示例配置带 duration 字段：动作帧编码有效时长（默认值见 test_n1e），心跳帧 duration 写 0；坏 duration_ms 拒绝
    ex_cfg = ua.load_config(EXAMPLE_CONFIG)
    assert any(f["kind"] == "duration" for f in ex_cfg["frame"]), "示例配置应含 duration 字段"
    f_act = ua.encode_frame(ex_cfg, 2, 7, False)
    f_hb = ua.encode_frame(ex_cfg, 2, 8, True)
    # 定位 duration 字段偏移
    off = 0
    dur_size = 0
    dur_endian = "little"
    for fld in ex_cfg["frame"]:
        if fld["kind"] == "duration":
            dur_size = fld["size"]
            dur_endian = fld.get("endian", "little")
            break
        off += len(fld["bytes"]) if "bytes" in fld else fld["size"]
    assert int.from_bytes(f_act[off:off + dur_size], dur_endian) == ex_cfg["duration_ms"]
    assert int.from_bytes(f_hb[off:off + dur_size], dur_endian) == 0
    f_custom = ua.encode_frame(ex_cfg, 1, 9, False, duration_ms=850)
    assert int.from_bytes(f_custom[off:off + dur_size], dur_endian) == 850

    bad = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    bad["duration_ms"] = 0
    bad_path = temp / "bad_dur.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    try:
        ua.load_config(bad_path)
    except ua.ProtocolError as exc:
        assert "duration_ms" in str(exc), exc
    else:
        raise AssertionError("duration_ms=0 未被拒绝")


def test_n1e_example_duration_is_nominal_period_plus_guard():
    """卡 W1-13 第三轮定的真机语义：每条动作收到之后有效 P + guard_s，新动作一到就取代。
    P 取采集端的名义步周期（sim/tools/collect.py 的 NOMINAL_STEP_PERIOD_S，0.9 s），
    guard_s 取主会话在那张卡里定的 0.05 s，所以示例配置的 duration_ms 是 950；P 变了这里要跟着红。"""
    src = (ROOT / "sim" / "tools" / "collect.py").read_text(encoding="utf-8")
    m = re.search(r"^NOMINAL_STEP_PERIOD_S\s*=\s*([0-9.]+)", src, re.M)
    assert m, "sim/tools/collect.py 里找不到 NOMINAL_STEP_PERIOD_S"
    nominal_ms = round(float(m.group(1)) * 1000)
    guard_ms = 50
    cfg = ua.load_config(EXAMPLE_CONFIG)
    assert cfg["duration_ms"] == nominal_ms + guard_ms, \
        f"示例 duration_ms={cfg['duration_ms']}，应为名义步周期 {nominal_ms} ms 加保持余量 {guard_ms} ms"


def test_n1g_duration_ms_without_duration_field_is_rejected(temp):
    """配置里写了 duration_ms 而帧里没有 duration 字段：有效时长发不到接收端，必须拒绝，不能悄悄忽略；
    不带 duration_ms 的旧 9 字节配置照常加载。"""
    cfg = json.loads(json.dumps(LEGACY_V1_CONFIG))
    ua.validate_config(json.loads(json.dumps(cfg)))
    cfg["duration_ms"] = 950
    path = temp / "dur_no_field.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    try:
        ua.load_config(path)
    except ua.ProtocolError as exc:
        assert "duration_ms" in str(exc) and "duration 字段" in str(exc), exc
    else:
        raise AssertionError("帧里没有 duration 字段却配了 duration_ms，没有被拒绝")
    p = run_cli(["--dry-run", "--config", str(path), "--action", "1"])
    assert p.returncode == 2 and "duration_ms" in p.stderr, (p.returncode, p.stderr)


def test_n1h_time_spans_limited_to_receiver_domain():
    """ttl_ms 与 duration_ms 的上限是 min(字段容量, 2^31 - 1)：接收端用 uint32 毫秒时钟做回绕比较，
    跨度超过 2^31 - 1 的帧它必然丢弃；4 字节字段下发送端在加载配置时就拒绝，不发出必被丢弃的帧。"""
    def widened(ttl_ms, duration_ms):
        raw = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        for field in raw["frame"]:
            if field["kind"] in ("ttl", "duration"):
                field["size"] = 4
        raw["ttl_ms"], raw["duration_ms"] = ttl_ms, duration_ms
        return raw

    cfg = ua.validate_config(widened(0x7FFFFFFF, 0x7FFFFFFF))        # 上限本身可以
    assert cfg["ttl_ms"] == 0x7FFFFFFF and cfg["duration_ms"] == 0x7FFFFFFF
    for name, ttl, dur in (("ttl_ms", 0x80000000, 1000), ("duration_ms", 300, 0x80000000),
                           ("ttl_ms", 0xFFFFFFFF, 1000), ("duration_ms", 300, 0xFFFFFFFF)):
        try:
            ua.validate_config(widened(ttl, dur))
        except ua.ProtocolError as exc:
            assert name in str(exc), exc
        else:
            raise AssertionError(f"{name} 超过 2^31-1 却没有被拒绝（ttl={ttl:#x} duration={dur:#x}）")
    try:
        ua.encode_frame(cfg, 1, 0, False, duration_ms=0x80000000)
    except ua.ProtocolError as exc:
        assert "duration_ms" in str(exc), exc
    else:
        raise AssertionError("encode_frame 的 duration_ms 覆盖值超过 2^31-1 却没有被拒绝")
    assert getattr(ua, "MAX_SPAN_MS", None) == 0x7FFFFFFF, "uart_action.MAX_SPAN_MS 应为 0x7FFFFFFF"


def main():
    global CONFIG
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        legacy_path = temp / "legacy_v1.json"
        legacy_path.write_text(json.dumps(LEGACY_V1_CONFIG), encoding="utf-8")
        CONFIG = legacy_path
        test_crc()
        test_default_config_and_encode()
        test_seq_helpers()
        test_crc16_frame()
        test_bad_config(temp)
        test_cli_dry_run(temp)
        test_cli_heartbeat()
        test_cli_stdin_actions()
        test_serial_errors(temp)
        test_n1a_partial_line_heartbeat()
        test_n1b_multiline_burst_in_order()
        test_n1c_eof_and_upstream_timeout()
        test_n1d_no_same_round_heartbeat_and_duration(temp)
        test_n1e_example_duration_is_nominal_period_plus_guard()
        test_n1g_duration_ms_without_duration_field_is_rejected(temp)
        test_n1h_time_spans_limited_to_receiver_domain()
    print("PASS: uart_action 帧编码/CRC/心跳/seq/dry-run/坏配置/串口错误 全部通过")


if __name__ == "__main__":
    main()
