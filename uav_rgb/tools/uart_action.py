#!/usr/bin/env python3
"""LicheePi → ESP32 四动作 UART 发送骨架（占位协议，未在任何固件上验证）。

只做帧编码和串口发送；不做模型推理，不修改 CF-Drone。字段顺序、动作码、CRC
全部由 JSON 配置给出。默认帧是占位格式，等 real-drone/uart_protocol.md 替换。

串口用 Python 标准库 termios 配置为 8N1 raw，不依赖 pyserial。
"""
import argparse
import json
import os
import select
import sys
import termios
import time
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "uart_protocol.example.json"
FIELD_KINDS = {"const", "seq", "action", "heartbeat", "ttl", "crc"}


class ProtocolError(ValueError):
    """配置或协议不合法，CLI 统一转成退出码 2。"""


def _reflect(value, bits):
    out = 0
    for _ in range(bits):
        out = (out << 1) | (value & 1)
        value >>= 1
    return out


def crc8(data, poly=0x07, init=0x00, xorout=0x00, reflect=False):
    crc = init
    for byte in data:
        if reflect:
            byte = _reflect(byte, 8)
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    if reflect:
        crc = _reflect(crc, 8)
    return crc ^ xorout


def crc16(data, poly=0x1021, init=0xFFFF, xorout=0x0000, reflect=False):
    crc = init
    for byte in data:
        if reflect:
            byte = _reflect(byte, 8)
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    if reflect:
        crc = _reflect(crc, 16)
    return crc ^ xorout


def _int(value, name, low, high):
    # bool 是 int 的子类，JSON 里的 true 不能当数字用。
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ProtocolError(f"{name} 必须是 {low}..{high} 的整数")
    return value


def _endian(field, name):
    endian = field.get("endian", "little")
    if endian not in ("little", "big"):
        raise ProtocolError(f"{name}.endian 只能是 little 或 big")
    return endian


def _crc_params(cfg):
    crc = cfg["crc"]
    if crc["algorithm"] == "crc8":
        return crc8, 8
    return crc16, 16


def _validate_crc(cfg):
    crc = cfg.get("crc")
    if not isinstance(crc, dict):
        raise ProtocolError("配置缺少 crc 对象")
    algorithm = crc.get("algorithm")
    if algorithm not in ("crc8", "crc16"):
        raise ProtocolError("crc.algorithm 只能是 crc8 或 crc16")
    bits = 8 if algorithm == "crc8" else 16
    crc["poly"] = _int(crc.get("poly"), "crc.poly", 1, (1 << bits) - 1)
    crc["init"] = _int(crc.get("init", 0), "crc.init", 0, (1 << bits) - 1)
    crc["xorout"] = _int(crc.get("xorout", 0), "crc.xorout", 0, (1 << bits) - 1)
    reflect = crc.get("reflect", False)
    if not isinstance(reflect, bool):
        raise ProtocolError("crc.reflect 必须是 true/false")
    crc["reflect"] = reflect


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ProtocolError("配置顶层必须是 JSON 对象")

    if "placeholder" in cfg and not isinstance(cfg["placeholder"], bool):
        raise ProtocolError("placeholder 必须是 true/false")
    device = cfg.get("device", "")
    if not isinstance(device, str):
        raise ProtocolError("device 必须是字符串")

    baud = _int(cfg.get("baud"), "baud", 1, 10_000_000)
    if getattr(termios, f"B{baud}", None) is None:
        raise ProtocolError(f"当前平台不支持波特率 {baud}")

    hz = cfg.get("heartbeat_hz")
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not hz > 0:
        raise ProtocolError("heartbeat_hz 必须是正数")
    cfg["heartbeat_hz"] = float(hz)

    _validate_crc(cfg)

    fields = cfg.get("frame")
    if not isinstance(fields, list) or not fields:
        raise ProtocolError("frame 必须是非空数组")
    names = set()
    kinds = set()
    action_size = ttl_size = None
    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            raise ProtocolError(f"frame[{index}] 必须是对象")
        name = field.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError(f"frame[{index}].name 必须是非空字符串")
        if name in names:
            raise ProtocolError(f"frame 字段名重复: {name}")
        names.add(name)
        kind = field.get("kind")
        if kind not in FIELD_KINDS:
            raise ProtocolError(f"{name}.kind 不支持: {kind!r}")
        if kind in ("seq", "action", "heartbeat", "ttl", "crc"):
            if kind in kinds:
                raise ProtocolError(f"frame 中 {kind} 字段重复")
            kinds.add(kind)
        if kind == "const":
            has_bytes = "bytes" in field
            has_value = "value" in field
            if has_bytes == has_value:
                raise ProtocolError(f"{name} 必须且只能有 bytes 或 value 之一")
            if has_bytes:
                raw = field["bytes"]
                if not isinstance(raw, list) or not raw:
                    raise ProtocolError(f"{name}.bytes 必须是非空数组")
                for byte in raw:
                    _int(byte, f"{name}.bytes", 0, 255)
                if "size" in field and _int(field["size"], f"{name}.size", 1, 64) != len(raw):
                    raise ProtocolError(f"{name}.size 与 bytes 长度不一致")
            else:
                size = _int(field.get("size"), f"{name}.size", 1, 8)
                _int(field["value"], f"{name}.value", 0, (1 << (8 * size)) - 1)
                _endian(field, name)
            continue
        size = _int(field.get("size"), f"{name}.size", 1, 4)
        _endian(field, name)
        if kind == "action":
            action_size = size
        elif kind == "ttl":
            ttl_size = size
        elif kind == "crc":
            _, bits = _crc_params(cfg)
            if size * 8 != bits:
                raise ProtocolError(f"{name}.size 必须等于 CRC 位宽 {bits} 位")
            if index != len(fields) - 1:
                raise ProtocolError("crc 字段必须是 frame 的最后一个字段")

    for required in ("seq", "action", "heartbeat", "ttl", "crc"):
        if required not in kinds:
            raise ProtocolError(f"frame 缺少 {required} 字段")

    ttl = cfg.get("ttl_ms")
    cfg["ttl_ms"] = _int(ttl, "ttl_ms", 1, (1 << (8 * ttl_size)) - 1)

    encoding = cfg.get("action_encoding")
    if not isinstance(encoding, list) or len(encoding) != 4:
        raise ProtocolError("action_encoding 必须是 4 个元素的数组（动作 0..3）")
    limit = 1 << (8 * action_size)
    for action, wire in enumerate(encoding):
        _int(wire, f"action_encoding[{action}]", 0, limit - 1)
    cfg["heartbeat_action"] = _int(cfg.get("heartbeat_action", 0), "heartbeat_action", 0, 3)
    return cfg


def load_config(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(f"读取配置失败: {path}: {exc}") from exc
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"配置不是合法 JSON: {path}: {exc}") from exc
    return validate_config(cfg)


def _pack(value, size, endian):
    return int(value).to_bytes(size, endian)


def encode_frame(cfg, action, seq, heartbeat):
    """action 0..3；heartbeat=True 时 action 字段写 heartbeat_action。"""
    if not 0 <= action < 4:
        raise ProtocolError(f"动作越界: {action}（只允许 0..3）")
    wire_action = cfg["heartbeat_action"] if heartbeat else action
    wire_action = cfg["action_encoding"][wire_action]
    body = bytearray()
    crc_field = None
    for field in cfg["frame"]:
        kind = field["kind"]
        if kind == "crc":
            crc_field = field
        elif kind == "const":
            if "bytes" in field:
                body += bytes(field["bytes"])
            else:
                body += _pack(field["value"], field["size"], _endian(field, field["name"]))
        elif kind == "seq":
            body += _pack(seq, field["size"], _endian(field, field["name"]))
        elif kind == "action":
            body += _pack(wire_action, field["size"], _endian(field, field["name"]))
        elif kind == "heartbeat":
            body += _pack(1 if heartbeat else 0, field["size"], _endian(field, field["name"]))
        elif kind == "ttl":
            body += _pack(cfg["ttl_ms"], field["size"], _endian(field, field["name"]))
        else:  # validate_config 已拦截，这里防御未来新增 kind 漏处理。
            raise ProtocolError(f"未知字段类型: {kind}")
    crc_fn, _ = _crc_params(cfg)
    params = cfg["crc"]
    crc = crc_fn(bytes(body), params["poly"], params["init"], params["xorout"], params["reflect"])
    return bytes(body) + _pack(crc, crc_field["size"], _endian(crc_field, crc_field["name"]))


def next_seq(seq, size):
    return (seq + 1) % (1 << (8 * size))


def parse_action(text):
    try:
        value = int(text, 10)
    except ValueError:
        raise ProtocolError(f"动作必须是 0..3 的整数: {text!r}") from None
    if not 0 <= value < 4:
        raise ProtocolError(f"动作越界: {value}（只允许 0..3）")
    return value


def open_serial(cfg):
    device = cfg.get("device", "")
    if not device:
        raise ProtocolError("配置缺少 device（串口设备路径）")
    speed = getattr(termios, f"B{cfg['baud']}", None)
    if speed is None:
        raise ProtocolError(f"当前平台不支持波特率 {cfg['baud']}")
    try:
        fd = os.open(device, os.O_WRONLY | os.O_NOCTTY)
    except FileNotFoundError as exc:
        raise ProtocolError(f"串口不存在: {device}") from exc
    except PermissionError as exc:
        raise ProtocolError(f"串口无权限: {device}（检查用户是否在 dialout 组）") from exc
    except OSError as exc:
        raise ProtocolError(f"无法打开串口 {device}: {exc.strerror or exc}") from exc
    try:
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0  # iflag
        attrs[1] = 0  # oflag
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL  # 8N1，无硬件流控
        attrs[3] = 0  # lflag: raw
        attrs[4] = speed
        attrs[5] = speed
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCOFLUSH)
    except (termios.error, OSError) as exc:
        os.close(fd)
        raise ProtocolError(f"配置串口 {device} 失败（可能不是串口设备）: {exc}") from exc
    return fd


def write_frame(fd, frame):
    view = memoryview(frame)
    while view:
        try:
            written = os.write(fd, view)
        except OSError as exc:
            raise ProtocolError(f"串口写入失败: {exc.strerror or exc}") from exc
        if written <= 0:
            raise ProtocolError("串口写入返回 0 字节")
        view = view[written:]


def _emit(fd, frame, dry_run):
    if dry_run:
        print(frame.hex(" "))
        sys.stdout.flush()
    else:
        write_frame(fd, frame)


def run(args):
    cfg = load_config(args.config)
    if cfg.get("placeholder"):
        print("uart_action: 警告：使用占位帧协议，等 real-drone/uart_protocol.md 替换后再上机",
              file=sys.stderr)
    if args.frames is not None and args.frames <= 0:
        raise ProtocolError("--frames 必须 > 0")

    if args.action is not None:
        action = parse_action(str(args.action))
        fd = None if args.dry_run else open_serial(cfg)
        try:
            _emit(fd, encode_frame(cfg, action, 0, False), args.dry_run)
        finally:
            if fd is not None:
                os.close(fd)
        return 0

    fd = None if args.dry_run else open_serial(cfg)
    seq_size = next(f["size"] for f in cfg["frame"] if f["kind"] == "seq")
    seq = 0
    sent = 0
    period = 1.0 / cfg["heartbeat_hz"]
    next_tick = time.monotonic()
    stdin_open = True
    try:
        while args.frames is None or sent < args.frames:
            timeout = max(0.0, next_tick - time.monotonic())
            ready = []
            if stdin_open:
                ready, _, _ = select.select([sys.stdin], [], [], timeout)
            else:
                time.sleep(timeout)
            if ready:
                line = sys.stdin.readline()
                if line == "":
                    stdin_open = False  # EOF 后继续发心跳；进程存活即持续保活
                elif line.strip():
                    action = parse_action(line.strip())
                    _emit(fd, encode_frame(cfg, action, seq, False), args.dry_run)
                    seq = next_seq(seq, seq_size)
                    sent += 1
            if args.frames is not None and sent >= args.frames:
                break
            now = time.monotonic()
            if now >= next_tick:
                _emit(fd, encode_frame(cfg, cfg["heartbeat_action"], seq, True), args.dry_run)
                seq = next_seq(seq, seq_size)
                sent += 1
                next_tick += period
                if next_tick <= now:  # 落后时不补发突发帧
                    next_tick = now + period
    finally:
        if fd is not None:
            os.close(fd)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="LicheePi → ESP32 四动作 UART 发送（占位协议，默认持续心跳）")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help=f"协议配置 JSON（默认 {DEFAULT_CONFIG}）")
    parser.add_argument("--action", type=int, help="发送一个动作 0..3 后退出")
    parser.add_argument("--dry-run", action="store_true", help="只打印 hex，不打开串口")
    parser.add_argument("--frames", type=int, help="最多发送 N 帧后退出；默认持续发送")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ProtocolError as exc:
        print(f"uart_action: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
