#!/usr/bin/env python3
"""tools/uart_action.py 自测：CRC、帧编码、动作越界、心跳/seq、dry-run、坏配置、串口缺失。

不打开真实串口：串口路径用例只用不存在的设备，或 mock os.open 抛 PermissionError。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().with_name("uart_action.py")
CONFIG = ROOT / "config" / "uart_protocol.example.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uart_action as ua  # noqa: E402

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
    p = run_cli(["--dry-run", "--config", str(CONFIG), "--frames", "3"], stdin="")
    assert p.returncode == 0, (p.stdout, p.stderr)
    frames = [bytes.fromhex(line) for line in p.stdout.splitlines()]
    assert len(frames) == 3
    assert [f[3] for f in frames] == [0, 1, 2]       # seq 逐帧 +1
    assert [f[4] for f in frames] == [0, 0, 0]       # 心跳动作固定 SAFE_NEUTRAL
    assert [f[5] for f in frames] == [1, 1, 1]       # heartbeat 标志
    for frame in frames:
        assert frame[-1] == ua.crc8(frame[:-1], 0x07, 0, 0, False)


def test_cli_stdin_actions():
    p = run_cli(["--dry-run", "--config", str(CONFIG), "--frames", "5"], stdin="1\n3\n")
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


def main():
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        test_crc()
        test_default_config_and_encode()
        test_seq_helpers()
        test_crc16_frame()
        test_bad_config(temp)
        test_cli_dry_run(temp)
        test_cli_heartbeat()
        test_cli_stdin_actions()
        test_serial_errors(temp)
    print("PASS: uart_action 帧编码/CRC/心跳/seq/dry-run/坏配置/串口错误 全部通过")


if __name__ == "__main__":
    main()
