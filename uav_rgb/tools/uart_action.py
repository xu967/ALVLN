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
FIELD_KINDS = {"const", "seq", "action", "heartbeat", "ttl", "duration", "crc"}
MAX_LINE_BYTES = 256  # 动作行仅含单个数字与空白；限制无换行残行长度，防异常上游耗尽板端内存
SAFE_NEUTRAL = 0
MAX_SPAN_MS = 0x7FFFFFFF  # 接收端（uart_codec.py）用 uint32 毫秒时钟做回绕比较，单个 ttl、duration 跨度不得超过 2^31 - 1
# 上游静默超时默认值（占位）：动作窗口 hold 1000 ms 加 200 ms 推理/调度余量；0 表示关闭上游超时检查。
# 对照：采集端停顿步门限是名义步周期 900 ms 加 200 ms = 1100 ms（sim/tools/collect.py），两者差 100 ms，是否对齐待主会话定；
# 接收端按帧里的 duration（示例 950 ms）自行到期回 SAFE_NEUTRAL，早于这里的 1200 ms，这个超时只负责发显式 SAFE 帧和告警。
DEFAULT_UPSTREAM_TIMEOUT_MS = 1200


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
    action_size = ttl_size = duration_size = None
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
        if kind in ("seq", "action", "heartbeat", "ttl", "duration", "crc"):
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
        elif kind == "duration":
            duration_size = size
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
    cfg["ttl_ms"] = _int(ttl, "ttl_ms", 1, min((1 << (8 * ttl_size)) - 1, MAX_SPAN_MS))
    if duration_size is not None:
        cfg["duration_ms"] = _int(cfg.get("duration_ms"), "duration_ms", 1,
                                  min((1 << (8 * duration_size)) - 1, MAX_SPAN_MS))
    elif "duration_ms" in cfg:
        raise ProtocolError("配置了 duration_ms，但 frame 里没有 duration 字段，有效时长发不到接收端；"
                            "加上 duration 字段或删掉 duration_ms")

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


def encode_frame(cfg, action, seq, heartbeat, duration_ms=None):
    """action 0..3；heartbeat=True 时仅作链路保活（duration 写 0，接收端忽略动作与时长字段）。"""
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
        elif kind == "duration":
            if heartbeat:
                dur = 0
            elif duration_ms is None:
                dur = cfg["duration_ms"]
            else:
                dur = _int(duration_ms, "duration_ms", 1, min((1 << (8 * field["size"])) - 1, MAX_SPAN_MS))
            body += _pack(dur, field["size"], _endian(field, field["name"]))
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


class _LineBuffer:
    """基于原始字节块的增量行缓冲：避免 TextIOWrapper.readline 在无换行半行上阻塞主循环。"""

    def __init__(self):
        self.buf = bytearray()
        self.eof = False

    def feed(self, chunk):
        if not chunk:
            self.eof = True
            return
        self.buf.extend(chunk)
        if b"\n" not in self.buf and len(self.buf) > MAX_LINE_BYTES:
            raise ProtocolError(f"单行输入超过 {MAX_LINE_BYTES} 字节且缺少换行符")

    def has_complete_line(self):
        return b"\n" in self.buf

    def pop_line(self):
        nl = self.buf.find(b"\n")
        if nl < 0:
            return None
        raw = bytes(self.buf[:nl])
        del self.buf[:nl + 1]
        return raw.decode("utf-8", "replace")


def _default_poll(in_fd):
    def _poll(timeout, stdin_open):
        if stdin_open:
            ready, _, _ = select.select([in_fd], [], [], timeout)
            return os.read(in_fd, 4096) if ready else None
        time.sleep(timeout)
        return None
    return _poll


def _default_warn(msg):
    print(msg, file=sys.stderr, flush=True)


def run_stream(cfg, emit_fn, *, max_frames=None, upstream_timeout_ms=DEFAULT_UPSTREAM_TIMEOUT_MS,
               in_fd=None, clock_fn=time.monotonic, poll_fn=None, warn_fn=_default_warn):
    """持续流主循环；poll_fn 与 clock_fn 可注入，便于无抖动测试半行、心跳与超时时序。"""
    if isinstance(upstream_timeout_ms, bool) or not isinstance(upstream_timeout_ms, int) or upstream_timeout_ms < 0:
        raise ProtocolError("--upstream-timeout-ms 必须是 >= 0 的整数（0 表示关闭）")
    if in_fd is None:
        in_fd = sys.stdin.fileno()
    if poll_fn is None:
        poll_fn = _default_poll(in_fd)
    seq_size = next(f["size"] for f in cfg["frame"] if f["kind"] == "seq")
    seq = 0
    sent = 0
    period = 1.0 / cfg["heartbeat_hz"]
    upstream_timeout_s = upstream_timeout_ms / 1000.0
    start_t = clock_fn()
    next_tick = start_t
    last_action_at = start_t
    upstream_stalled = False
    reader = _LineBuffer()
    while max_frames is None or sent < max_frames:
        next_deadline = next_tick
        if upstream_timeout_s > 0.0 and not upstream_stalled:
            next_deadline = min(next_deadline, last_action_at + upstream_timeout_s)
        timeout = 0.0 if reader.has_complete_line() else max(0.0, next_deadline - clock_fn())
        chunk = None if reader.has_complete_line() else poll_fn(timeout, not reader.eof)
        if chunk is not None:
            reader.feed(chunk)
        emitted_action = False
        while (max_frames is None or sent < max_frames) and (line := reader.pop_line()) is not None:
            if line.strip():
                action = parse_action(line.strip())
                emit_fn(encode_frame(cfg, action, seq, False))
                seq = next_seq(seq, seq_size)
                sent += 1
                last_action_at = clock_fn()
                upstream_stalled = False
                emitted_action = True
        if max_frames is not None and sent >= max_frames:
            break
        if chunk is not None and len(chunk) >= 4096 and not reader.eof:
            if emitted_action:
                next_tick = clock_fn() + period
            continue  # 单次读满 4 KiB 时立即继续排空内核管道，不等心跳周期
        if reader.eof:
            # 上游关闭管道：先发一帧 SAFE_NEUTRAL 取消当前转向/前飞，再立即退出把链路失联超时交还飞控 TTL
            emit_fn(encode_frame(cfg, SAFE_NEUTRAL, seq, False))
            seq = next_seq(seq, seq_size)
            sent += 1
            if bytes(reader.buf).strip():
                raise ProtocolError("标准输入在未换行的半行处关闭")
            break
        now = clock_fn()
        if upstream_timeout_s > 0.0 and not upstream_stalled and now - last_action_at >= upstream_timeout_s - 1e-12:
            upstream_stalled = True
            warn_fn(f"uart_action: 警告：上游超过 {upstream_timeout_ms} ms 未发新动作，已下发 SAFE_NEUTRAL")
            emit_fn(encode_frame(cfg, SAFE_NEUTRAL, seq, False))
            seq = next_seq(seq, seq_size)
            sent += 1
            emitted_action = True
            if max_frames is not None and sent >= max_frames:
                break
        if emitted_action:
            # 动作帧本身已刷新飞控侧链路存活计时，本轮不再紧跟心跳帧
            next_tick = now + period
        elif now >= next_tick:
            emit_fn(encode_frame(cfg, cfg["heartbeat_action"], seq, True))
            seq = next_seq(seq, seq_size)
            sent += 1
            next_tick += period
            if next_tick <= now:  # 落后时不补发突发帧
                next_tick = now + period
    return 0


def run(args):
    cfg = load_config(args.config)
    if cfg.get("placeholder"):
        print("uart_action: 警告：使用占位帧协议，等 real-drone/uart_protocol.md 替换后再上机",
              file=sys.stderr)
    if args.frames is not None and args.frames <= 0:
        raise ProtocolError("--frames 必须 > 0")
    upstream_timeout_ms = getattr(args, "upstream_timeout_ms", DEFAULT_UPSTREAM_TIMEOUT_MS)
    if upstream_timeout_ms < 0:
        raise ProtocolError("--upstream-timeout-ms 必须 >= 0")

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
    try:
        return run_stream(
            cfg,
            lambda frame: _emit(fd, frame, args.dry_run),
            max_frames=args.frames,
            upstream_timeout_ms=upstream_timeout_ms,
        )
    finally:
        if fd is not None:
            os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="LicheePi → ESP32 四动作 UART 发送（占位协议，默认持续心跳）")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help=f"协议配置 JSON（默认 {DEFAULT_CONFIG}）")
    parser.add_argument("--action", type=int, help="发送一个动作 0..3 后退出")
    parser.add_argument("--dry-run", action="store_true", help="只打印 hex，不打开串口")
    parser.add_argument("--frames", type=int, help="最多发送 N 帧后退出；默认持续发送")
    parser.add_argument("--upstream-timeout-ms", type=int, default=DEFAULT_UPSTREAM_TIMEOUT_MS,
                        help=f"上游超过该毫秒数无新动作则下发 SAFE_NEUTRAL 并告警（0=关闭，默认 {DEFAULT_UPSTREAM_TIMEOUT_MS}）")
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
