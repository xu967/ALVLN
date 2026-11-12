#!/usr/bin/env python3
"""tools/uart_codec.py 与全链路伪终端故障注入测试（卡 DEP1 / 审核 N-7）。

覆盖：
1. 示例配置升级为 CRC-16 + 2 字节 seq + duration，且旧 9 字节 CRC-8 配置仍兼容解析；
2. 零系统时钟依赖（时钟全部由调用方以 ms 传入）与定长缓冲无动态扩容；
3. 随机比特翻转、粘包拆包（含假 0xAA 0x55 帧头重同步与半帧超时）、丢帧、乱序/重放、
   8 位/16 位序号回绕、心跳保活不改动作与截止、动作到期回 SAFE_NEUTRAL、断流 350 ms 触发失联；
4. 真实伪终端（os.openpty）串联 uart_action.py 真实串口路径（termios 8N1）与接收状态机。
各类故障注入均断言「接收端接受的坏帧数 == 0」。
"""
import inspect
import json
import os
import random
import select
import subprocess
import sys
import tempfile
import time
import tty
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = Path(__file__).resolve().parent
SCRIPT = TOOLS / "uart_action.py"
EXAMPLE_CONFIG = ROOT / "config" / "uart_protocol.example.json"
sys.path.insert(0, str(TOOLS))
# 等伪终端数据的上限：只防挂死，不是完成判据（完成判据都是事件）；变异检查时可调小
WIRE_TIMEOUT_S = float(os.environ.get("UART_TEST_WIRE_TIMEOUT_S", "30"))

import uart_action as ua  # noqa: E402
import uart_codec as uc  # noqa: E402
from test_uart_action import LEGACY_V1_CONFIG  # noqa: E402


def test_example_config_upgraded_and_no_sys_clock():
    cfg = ua.load_config(EXAMPLE_CONFIG)
    assert cfg["crc"]["algorithm"] == "crc16"
    seq_field = next(f for f in cfg["frame"] if f["kind"] == "seq")
    crc_field = next(f for f in cfg["frame"] if f["kind"] == "crc")
    dur_field = next(f for f in cfg["frame"] if f["kind"] == "duration")
    assert seq_field["size"] == 2
    assert crc_field["size"] == 2
    assert dur_field["size"] == 2

    # 接收状态机源码不得 import time，确保可逐行映射为固件纯状态机
    src = inspect.getsource(uc)
    assert "import time" not in src and "monotonic(" not in src

    rx = uc.UartReceiver(cfg)
    buf_id = id(rx.rx_buf)
    buf_cap = len(rx.rx_buf)
    assert buf_cap == uc.MAX_FRAME_SIZE
    rx.feed_bytes(b"\x00\xaa\x55" + ua.encode_frame(cfg, 1, 0, False), now_ms=100)
    assert id(rx.rx_buf) == buf_id and len(rx.rx_buf) == buf_cap, "接收缓冲发生了动态重分配"


def test_bit_flips_zero_bad_frames():
    """随机比特翻转（固定种子，500 帧中 35% 注入 1..8 比特翻转 + 随机杂散字节）：坏帧接受数必须为 0。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)
    rng = random.Random(20261112)
    rx = uc.UartReceiver(cfg)

    intact_frames = set()
    expected_intact_seqs = []
    accepted_raw = []
    accepted_seqs = []

    def on_frame(raw_bytes, parsed):
        accepted_raw.append(raw_bytes)
        accepted_seqs.append(parsed["seq"])

    stream = bytearray()
    for seq in range(500):
        hb = (seq % 3 == 1)
        act = 0 if hb else (seq % 4)
        frame = ua.encode_frame(cfg, act, seq, hb)
        if rng.random() < 0.35:
            bad = bytearray(frame)
            n_flips = rng.randint(1, 8)
            for _ in range(n_flips):
                idx = rng.randrange(len(bad))
                bit = 1 << rng.randrange(8)
                bad[idx] ^= bit
            if bytes(bad) != frame:
                stream.extend(bad)
                continue
        intact_frames.add(frame)
        expected_intact_seqs.append(seq)
        stream.extend(frame)
        if rng.random() < 0.15:
            # 夹入含假 magic 字节的噪声
            stream.extend(bytes([0xAA, 0x55, rng.randrange(256)]))

    rx.feed_bytes(bytes(stream), now_ms=1000, on_frame=on_frame)
    bad_frames_accepted = sum(1 for raw in accepted_raw if raw not in intact_frames)
    assert bad_frames_accepted == 0, f"接受了坏帧: {bad_frames_accepted}"
    assert accepted_seqs == expected_intact_seqs, (len(accepted_seqs), len(expected_intact_seqs))
    assert (rx.crc_errors + rx.header_drops + rx.field_errors) > 100


def test_sticky_fragmented_resync_and_partial_timeout():
    """粘包、逐字节拆包、假 0xAA 0x55 帧头滑窗重同步、半帧超时清零。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)
    rx = uc.UartReceiver(cfg, byte_timeout_ms=50)
    intact = []
    got_seqs = []

    def on_frame(raw_bytes, parsed):
        assert raw_bytes in intact, "滑窗重同步拼出了非原始帧"
        got_seqs.append(parsed["seq"])

    # 构造：假帧头 0xAA 0x55 0x01 + 截断残片 + 5 个紧挨粘包真帧
    blob = bytearray(b"\x99\xaa\x55\x01\xff\xaa\x55")
    for seq in range(10, 15):
        f = ua.encode_frame(cfg, seq % 4, seq, False)
        intact.append(f)
        blob.extend(f)

    # 按随机 1..4 字节拆包喂入（同一时刻，纯靠滑窗重同步找回全部 5 帧）
    rng = random.Random(7)
    pos = 0
    while pos < len(blob):
        step = rng.randint(1, 4)
        rx.feed_bytes(blob[pos:pos + step], now_ms=200, on_frame=on_frame)
        pos += step
    assert got_seqs == [10, 11, 12, 13, 14], got_seqs

    # 半帧截断后静默超过 byte_timeout_ms=50ms：残片应在下一帧到来前被超时清零
    f_next = ua.encode_frame(cfg, 2, 15, False)
    intact.append(f_next)
    rx.feed_bytes(f_next[:5], now_ms=300, on_frame=on_frame)
    assert rx.rx_len == 5
    rx.feed_bytes(f_next, now_ms=360, on_frame=on_frame)
    assert rx.partial_timeouts == 1
    assert got_seqs[-1] == 15


def test_drop_out_of_order_and_wraparound():
    """丢帧、乱序、重复重放，以及 16 位（65535->0）与旧版 8 位（255->0）序号回绕。"""
    # 1) 16 位序号：丢帧 + 乱序 + 重放 + 跨 65535->0 回绕
    cfg16 = ua.load_config(EXAMPLE_CONFIG)
    rx16 = uc.UartReceiver(cfg16)
    accepted = []

    def send16(seq, act=1, hb=False, t=100):
        raw = ua.encode_frame(cfg16, act, seq, hb)
        ok = rx16.feed_bytes(raw, now_ms=t, on_frame=lambda _r, p: accepted.append(p["seq"]))
        return ok

    assert send16(65530, act=1, t=100) == 1
    assert send16(65530, act=2, t=110) == 0          # 重复帧拒收
    assert rx16.current_action == 1                  # 重复帧不改动作
    assert send16(65534, act=2, t=120) == 1          # 中间丢了 65531..65533，正常接受
    assert send16(65532, act=3, t=130) == 0          # 迟到的乱序旧帧拒收
    assert rx16.current_action == 2
    assert send16(65535, act=2, t=140) == 1
    assert send16(0, act=3, t=150) == 1              # 65535 -> 0 回绕接受
    assert send16(3, act=1, t=160) == 1              # 0 -> 3 跳跃接受
    assert send16(65535, act=0, t=170) == 0          # 回绕后的旧帧 65535 拒收
    assert accepted == [65530, 65534, 65535, 0, 3], accepted
    assert rx16.seq_drops == 3

    # 2) 旧版 8 位序号 + CRC-8 配置兼容：跨 255->0 回绕与半窗口边界
    cfg8 = ua.validate_config(json.loads(json.dumps(LEGACY_V1_CONFIG)))
    rx8 = uc.UartReceiver(cfg8)
    acc8 = []
    for seq in (253, 254, 255, 0, 1, 128):
        rx8.feed_bytes(ua.encode_frame(cfg8, 1, seq, False), now_ms=200,
                       on_frame=lambda _r, p: acc8.append(p["seq"]))
    # 1 -> 128 差值 127 < 128 接受；再发 0（差值 128 >= 半窗口 128）必须拒收
    rx8.feed_bytes(ua.encode_frame(cfg8, 2, 0, False), now_ms=210,
                   on_frame=lambda _r, p: acc8.append(p["seq"]))
    assert acc8 == [253, 254, 255, 0, 1, 128], acc8
    assert rx8.seq_drops == 1


def _seq_width_cfg(size):
    """示例配置里只把 seq 字段改成 size 字节（帧长、CRC 位置随之后移）。"""
    raw = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    for field in raw["frame"]:
        if field["kind"] == "seq":
            field["size"] = size
    return ua.validate_config(raw)


def _seq_probe(cfg, last, new):
    """序号锚在 last（动作 1）之后喂序号 new（动作 2）；返回 new 帧是否被接受，同时核对拒收时状态原封不动。"""
    rx = uc.UartReceiver(cfg)
    assert rx.feed_bytes(ua.encode_frame(cfg, 1, last, False, duration_ms=1000), now_ms=100) == 1, \
        f"锚点帧 seq={last} 没有被接受"
    got = rx.feed_bytes(ua.encode_frame(cfg, 2, new, False, duration_ms=1000), now_ms=110)
    if got:
        assert rx.current_action == 2 and rx.last_seq == new, f"last={last} new={new}：被接受的帧没有更新动作与序号"
    else:
        assert rx.current_action == 1 and rx.last_seq == last and rx.seq_drops == 1, \
            f"last={last} new={new}：被拒收的帧改动了状态，或没有计入 seq_drops"
    return got == 1


def test_half_window_boundary_for_all_seq_widths():
    """半窗口序号比较在 1、2、3、4 字节序号上各锁一次：差值 1..H-1 接受，差值正好等于半窗口 H、H+1、
    0（重复）与 -1（回退一格）拒收；起点覆盖 0、H-1、H 与回绕前后。1 字节宽度对 (last, new) 做枚举判据对拍。"""
    for size in (1, 2, 3, 4):
        cfg = _seq_width_cfg(size)
        window = 1 << (8 * size)
        half = window >> 1
        layout = uc.FrameLayout(cfg)
        assert layout.seq_half == half and layout.seq_mask == window - 1, f"seq={size}B 的半窗口或掩码与宽度不符"
        for last in (0, 1, half - 1, half, window - 2, window - 1):
            for diff, want in ((1, True), (half - 1, True), (half, False), (half + 1, False),
                               (0, False), (window - 1, False)):
                new = (last + diff) % window
                assert _seq_probe(cfg, last, new) is want, \
                    f"seq={size}B last={last} 差值={diff}（半窗口 {half}）应{'接受' if want else '拒收'}"

    # 1 字节：对一批 last 与全部 256 个 new，用「后面 127 个序号」的枚举集合做判据，与接收端对拍
    cfg1 = _seq_width_cfg(1)
    rng = random.Random(20261112)
    for last in [0, 1, 127, 128, 129, 254, 255] + [rng.randrange(256) for _ in range(9)]:
        newer = {(last + k) % 256 for k in range(1, 128)}
        for new in range(256):
            assert _seq_probe(cfg1, last, new) is (new in newer), f"last={last} new={new}"

    # 2、3、4 字节：固定种子随机对拍，判据换成有符号距离写法 d = ((new - last + H) mod 2^N) - H，d > 0 才算新
    for size in (2, 3, 4):
        cfg = _seq_width_cfg(size)
        window = 1 << (8 * size)
        half = window >> 1
        for _ in range(300):
            last, new = rng.randrange(window), rng.randrange(window)
            signed = ((new - last + half) % window) - half
            assert _seq_probe(cfg, last, new) is (signed > 0), f"seq={size}B last={last} new={new}"


def test_heartbeat_semantics_action_expiry_and_link_loss():
    """心跳只续期链路不改动作与截止；新动作立即取代；动作到期回 SAFE；断流 350ms 进 LINK_LOST。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)  # ttl_ms=300；下面的动作帧都显式写 duration_ms=1000，不依赖示例的默认值
    rx = uc.UartReceiver(cfg)
    assert rx.link_state == uc.LINK_IDLE and rx.current_action == 0

    # t=100ms 收到 YAW_RIGHT(3)，有效时长 1000ms（截止 1100ms），链路截止 400ms
    rx.feed_bytes(ua.encode_frame(cfg, 3, 0, False, duration_ms=1000), now_ms=100)
    assert rx.link_state == uc.LINK_ACTIVE
    assert rx.current_action == 3
    assert rx.action_deadline_ms == 1100 and rx.link_deadline_ms == 400

    # 故意构造一个把 action 字段写成 YAW_LEFT(2) 的心跳帧：接收端必须忽略心跳帧里的 action！
    malicious_cfg = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    malicious_cfg["heartbeat_action"] = 2
    malicious_cfg = ua.validate_config(malicious_cfg)
    for step_idx, t in enumerate(range(200, 1050, 100), start=1):
        rx.feed_bytes(ua.encode_frame(malicious_cfg, 2, step_idx, True), now_ms=t)
        assert rx.link_state == uc.LINK_ACTIVE
        assert rx.current_action == 3, f"t={t} 心跳帧篡改了当前动作: {rx.current_action}"
        assert rx.action_deadline_ms == 1100, f"t={t} 心跳帧篡改了动作截止: {rx.action_deadline_ms}"

    # t=1100ms：链路仍由心跳保活（上次心跳 t=1000，链路截止 1300），但动作有效期到点 -> 自动回 SAFE_NEUTRAL(0)
    rx.feed_bytes(ua.encode_frame(cfg, 0, 10, True), now_ms=1100)
    assert rx.link_state == uc.LINK_ACTIVE
    assert rx.current_action == 0 and rx.action_expirations == 1

    # t=1150ms 新动作 FORWARD_SLOW(1, dur=1000 -> 2150)，期间正常 10Hz 心跳保活，t=1800ms 新动作 YAW_LEFT(2) 直接无空档取代
    rx.feed_bytes(ua.encode_frame(cfg, 1, 11, False, duration_ms=1000), now_ms=1150)
    assert rx.current_action == 1 and rx.action_deadline_ms == 2150
    for idx, hb_t in enumerate(range(1250, 1800, 100), start=12):
        rx.feed_bytes(ua.encode_frame(cfg, 0, idx, True), now_ms=hb_t)
        assert rx.link_state == uc.LINK_ACTIVE and rx.current_action == 1
    rx.feed_bytes(ua.encode_frame(cfg, 2, 20, False, duration_ms=1000), now_ms=1800)
    assert rx.current_action == 2 and rx.action_deadline_ms == 2800 and rx.failsafe_count == 0

    # 断流 350ms（> ttl_ms=300ms）：t=2150ms 触发失联保护，动作强制回 SAFE_NEUTRAL(0)
    rx.tick(now_ms=2099)
    assert rx.link_state == uc.LINK_ACTIVE and rx.current_action == 2
    rx.tick(now_ms=2150)
    assert rx.link_state == uc.LINK_LOST
    assert rx.current_action == 0
    assert rx.failsafe_count == 1

    # 发送端进程重启后从 seq=0 重新发帧：LINK_LOST 后允许重新锚定 seq
    rx.feed_bytes(ua.encode_frame(cfg, 1, 0, False), now_ms=2500)
    assert rx.link_state == uc.LINK_ACTIVE and rx.current_action == 1


U32 = 1 << 32


def _widened_cfg(ttl_ms, duration_ms):
    """ttl 与 duration 都放宽到 4 字节的示例配置变体（用来碰 2^31 - 1 的跨度上限；超过上限的值发送端会拒绝加载）。"""
    raw = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    for field in raw["frame"]:
        if field["kind"] in ("ttl", "duration"):
            field["size"] = 4
    raw["ttl_ms"] = ttl_ms
    raw["duration_ms"] = duration_ms
    return ua.validate_config(raw)


def test_u32_wrap_link_loss():
    """固件的毫秒计数是 uint32（约 49.7 天回绕）：截止比较要写成 (int32)(now - deadline) >= 0。
    失联倒计时跨回绕：回绕前 100 ms 收到最后一帧，ttl 300 ms，第 300 ms 起失联（回绕后第 200 ms）。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)       # ttl_ms=300
    for shift in (0, 7 * U32):                  # 调用方传 64 位毫秒时只有低 32 位有意义
        rx = uc.UartReceiver(cfg)
        t0 = U32 - 100
        rx.feed_bytes(ua.encode_frame(cfg, 1, 0, False, duration_ms=60000), now_ms=shift + t0)
        assert rx.link_deadline_ms == (t0 + 300) % U32, "链路截止没有按 uint32 存"
        for dt in (0, 100, 150, 299):
            rx.tick(shift + (t0 + dt) % U32)
            assert rx.link_state == uc.LINK_ACTIVE and rx.current_action == 1, f"dt={dt} 提前失联"
        rx.tick(shift + (t0 + 300) % U32)
        assert rx.link_state == uc.LINK_LOST, "跨 uint32 回绕后失联保护没有按期触发"
        assert rx.current_action == 0 and rx.failsafe_count == 1


def test_u32_wrap_action_expiry():
    """动作有效期跨回绕：回绕前 500 ms 收到 dur=1000 的动作，心跳保活；第 1000 ms 动作回 SAFE、链路仍在。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)
    rx = uc.UartReceiver(cfg)
    t1 = U32 - 500
    rx.feed_bytes(ua.encode_frame(cfg, 3, 0, False, duration_ms=1000), now_ms=t1)
    # 截止时刻与固件里一样存成 uint32：回绕后的值落在 [0, 2^32)
    assert rx.link_deadline_ms == (t1 + 300) % U32 and rx.action_deadline_ms == (t1 + 1000) % U32, \
        "链路截止与动作截止没有按 uint32 存"
    for idx, dt in enumerate(range(100, 1000, 100), start=1):
        rx.feed_bytes(ua.encode_frame(cfg, 0, idx, True), now_ms=(t1 + dt) % U32)
        assert rx.link_state == uc.LINK_ACTIVE and rx.current_action == 3, f"dt={dt} 动作提前到期"
    rx.tick((t1 + 999) % U32)
    assert rx.current_action == 3
    rx.tick((t1 + 1000) % U32)
    assert rx.current_action == 0 and rx.action_expirations == 1, "跨回绕后动作有效期没有按期到期"
    assert rx.link_state == uc.LINK_ACTIVE


def test_u32_wrap_partial_frame_timeout():
    """半帧超时跨回绕：byte_timeout_ms=50，严格大于 50 ms 才清残片。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)
    rx = uc.UartReceiver(cfg, byte_timeout_ms=50)
    t2 = U32 - 10
    rx.feed_bytes(ua.encode_frame(cfg, 1, 0, False)[:5], now_ms=t2)
    rx.tick((t2 + 50) % U32)
    assert rx.rx_len == 5 and rx.partial_timeouts == 0
    rx.tick((t2 + 51) % U32)
    assert rx.rx_len == 0 and rx.partial_timeouts == 1, "跨回绕后半帧超时没有触发"


def test_u32_span_limit_for_ttl_and_duration():
    """跨度上限 2^31 - 1：回绕安全的比较只在截止离现在不到 2^31 毫秒时成立，
    ttl、duration 超过这个上限的帧按字段取值域错误丢弃。"""
    ok = _widened_cfg(ttl_ms=0x7FFFFFFF, duration_ms=0x7FFFFFFF)
    rx = uc.UartReceiver(ok)
    assert rx.feed_bytes(ua.encode_frame(ok, 2, 0, False), now_ms=U32 - 5) == 1, \
        "跨度恰为 2^31-1 的合法帧被接收端丢弃"
    rx.tick((U32 - 5 + 0x7FFFFFFE) % U32)
    assert rx.link_state == uc.LINK_ACTIVE and rx.current_action == 2, "跨度 2^31-1 的截止提前触发"
    rx.tick((U32 - 5 + 0x7FFFFFFF) % U32)
    assert rx.link_state == uc.LINK_LOST, "跨度 2^31-1 的截止跨回绕后没有按期触发"
    assert ua.MAX_SPAN_MS == uc.MAX_SPAN_MS, "发送端与接收端的跨度上限常量不一致"
    good = _widened_cfg(ttl_ms=300, duration_ms=1000)
    for bad_ttl, bad_dur, hb in ((0x80000000, 1000, True), (300, 0x80000000, False)):
        # 发送端加载配置时就拒绝这两个值（见 test_uart_action 的 N-1h），接收端构造时也会再校验一遍配置；
        # 所以用合法配置建接收端，另拷一份字典改成越界值只用来编码，单独验证接收端对线上字节的取值域检查
        bad = dict(good)
        bad["ttl_ms"], bad["duration_ms"] = bad_ttl, bad_dur
        frame = ua.encode_frame(bad, 1, 0, hb)
        rx = uc.UartReceiver(good)
        got = rx.feed_bytes(frame, now_ms=0)
        assert got == 0 and rx.field_errors >= 1 and rx.link_state == uc.LINK_IDLE, \
            f"ttl={bad_ttl} duration={bad_dur} 超过 2^31-1 的帧没有被丢弃"


def test_pty_end_to_end_fault_injection_and_link_loss(temp):
    """真实发送进程 -> 伪终端（termios 8N1）-> UartReceiver。
    完成判据都是事件：先等接收端收下 2 个动作帧与至少 1 个心跳帧，关上游管道后等末帧 SAFE_NEUTRAL 被收下，
    不用「静默多久就当读完」；超时只是防挂死的上限。接收端时钟用虚拟毫秒，断流 350 ms 触发失联。"""
    master_fd, slave_fd = os.openpty()
    os.set_blocking(master_fd, False)
    slave_name = os.ttyname(slave_fd)

    cfg = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    cfg["device"] = slave_name
    cfg["heartbeat_hz"] = 20
    cfg["ttl_ms"] = 300
    cfg_path = temp / "pty_uart.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    valid_cfg = ua.load_config(cfg_path)
    rx = uc.UartReceiver(valid_cfg)
    accepted_actions = []
    heartbeats = []

    def on_f(_raw, parsed):
        if parsed["heartbeat"] == 0:
            accepted_actions.append(parsed["action"])
        else:
            heartbeats.append(parsed["seq"])

    # 关掉上游超时：本测试只看 EOF 补发 SAFE_NEUTRAL，不让真实时间长短在流里多插一帧 SAFE_NEUTRAL
    p = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--config", str(cfg_path), "--upstream-timeout-ms", "0"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    def pump(until, what):
        give_up = time.monotonic() + WIRE_TIMEOUT_S
        while not until():
            # 上限每圈都查：发送进程的心跳不停，数据源源不断时也要能失败退出
            assert time.monotonic() < give_up, \
                f"等{what}超过 {WIRE_TIMEOUT_S:g} s：动作 {accepted_actions}，心跳 {len(heartbeats)} 个"
            ready, _, _ = select.select([master_fd], [], [], 0.5)
            if ready:
                rx.feed_bytes(os.read(master_fd, 4096), now_ms=1000, on_frame=on_f)
                continue
            rc_now = p.poll()
            assert rc_now in (None, 0), f"发送进程异常退出 rc={rc_now}: {p.stderr.read().decode('utf-8', 'replace')}"

    try:
        # 真实串口字节流之前先放一段假帧头残片，考接收端的滑窗
        rx.feed_bytes(b"\xaa\x55\xff", now_ms=1000)
        os.write(p.stdin.fileno(), b"1\n2\n")
        pump(lambda: len(accepted_actions) >= 2 and len(heartbeats) >= 1, "两个动作帧与至少一个心跳帧")

        # 关闭上游管道：子进程补发一帧 SAFE_NEUTRAL(0) 后立即退出
        p.stdin.close()
        pump(lambda: len(accepted_actions) >= 3, "EOF 末帧 SAFE_NEUTRAL")
        rc = p.wait(timeout=30.0)
        assert rc == 0, p.stderr.read().decode("utf-8", "replace")

        assert accepted_actions == [1, 2, 0], f"动作帧应为 [1, 2, EOF 末帧 SAFE_NEUTRAL 0]，实得 {accepted_actions}"
        assert rx.crc_errors == 0 and rx.field_errors == 0 and rx.seq_drops == 0
        assert rx.link_state == uc.LINK_ACTIVE

        # 停止输入满 350 ms（> ttl_ms=300 ms）：接收端进入 LINK_LOST 失联保护
        rx.tick(now_ms=1299)
        assert rx.link_state == uc.LINK_ACTIVE
        rx.tick(now_ms=1350)
        assert rx.link_state == uc.LINK_LOST
        assert rx.current_action == 0
        assert rx.failsafe_count == 1
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        for pipe in (p.stdin, p.stdout, p.stderr):
            pipe.close()
        os.close(master_fd)
        os.close(slave_fd)


class _PtyWire:
    """伪终端当串口线用：往 master 写，接收端从 slave 读（termios raw 8N1，和固件 UART 一样不改字节）。
    send() 先核对 slave 读回的字节与写入的逐字节相同，再喂接收状态机。
    完成判据是「读满写入的字节数」，不靠静默时间；30 s 只是防挂死的上限。"""

    def __enter__(self):
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)
        return self

    def __exit__(self, *_exc):
        os.close(self.master)
        os.close(self.slave)

    def send(self, rx, chunk, now_ms, on_frame=None):
        assert 0 < len(chunk) <= 1024, "单次写入不超过 1024 字节，免得写满伪终端缓冲"
        os.write(self.master, chunk)
        got = bytearray()
        give_up = time.monotonic() + WIRE_TIMEOUT_S
        while len(got) < len(chunk):
            ready, _, _ = select.select([self.slave], [], [], 0.5)
            if ready:
                got += os.read(self.slave, len(chunk) - len(got))
            assert time.monotonic() < give_up, f"伪终端 {WIRE_TIMEOUT_S:g} s 内没有送出写入的字节"
        assert bytes(got) == chunk, "伪终端改动了字节"
        return rx.feed_bytes(bytes(got), now_ms, on_frame=on_frame)


def _wire_run(cfg, steps, rx=None, byte_timeout_ms=50):
    """steps = [(虚拟毫秒, 字节)]：每段都先经伪终端送达，再按该虚拟时刻喂接收状态机。
    返回 (rx, accepted, trace)：accepted 是 [(seq, 心跳, 动作, 原始帧)]，trace 是每段送完后的 current_action。"""
    rx = rx or uc.UartReceiver(cfg, byte_timeout_ms=byte_timeout_ms)
    accepted, trace = [], []

    def on_frame(raw, parsed):
        accepted.append((parsed["seq"], parsed["heartbeat"], parsed["action"], raw))

    with _PtyWire() as wire:
        for now_ms, chunk in steps:
            wire.send(rx, chunk, now_ms, on_frame=on_frame)
            trace.append(rx.current_action)
    return rx, accepted, trace


def _wf(cfg, seq, hb=False, act=None):
    """故障注入用的帧：动作帧默认动作为 1..3 轮换（不取 0，免得和 SAFE_NEUTRAL 混淆），心跳帧只保活。"""
    return ua.encode_frame(cfg, 0 if hb else (1 + seq % 3 if act is None else act), seq, hb)


def test_pty_wire_random_bit_flips():
    """故障 1：随机比特翻转。固定种子，400 帧里约 35% 被翻 1..8 个比特，另夹假帧头残片；
    判据是「接收端接受的帧都是原样的好帧、好帧一个不少、顺序不变」，不是只数一数。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)
    rng = random.Random(20261112)
    steps, intact_seqs, intact_raw = [], [], set()
    for seq in range(400):
        frame = _wf(cfg, seq, hb=(seq % 3 == 1))
        if rng.random() < 0.35:
            bad = bytearray(frame)
            for bit in rng.sample(range(len(bad) * 8), rng.randint(1, 8)):
                bad[bit // 8] ^= 1 << (bit % 8)
            steps.append((20 * seq, bytes(bad)))
            continue
        steps.append((20 * seq, frame))
        intact_seqs.append(seq)
        intact_raw.add(frame)
        if rng.random() < 0.15:
            steps.append((20 * seq, bytes([0xAA, 0x55, rng.randrange(256)])))
    rx, accepted, _ = _wire_run(cfg, steps)
    assert all(raw in intact_raw for *_head, raw in accepted), "接收端接受了被翻转过的坏帧"
    assert [a[0] for a in accepted] == intact_seqs, "好帧丢了、多了或乱序"
    assert rx.failsafe_count == 0, "帧间隔 20 ms，不该触发失联"
    assert rx.crc_errors + rx.header_drops + rx.field_errors > 50, "故障没有真的注入进去"


def test_pty_wire_sticky_fragmented_and_partial_timeout():
    """故障 2：粘包拆包。开头放一遍 0..255 全部字节值（同时核对伪终端不改字节），再接假帧头残片与 40 个紧挨的帧，
    随机拆成 1..5 字节小包并夹 40、90 字节的粘包大块；最后一个半帧静默 60 ms 后被超时清零，后面的完整帧照收。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)
    rng = random.Random(7)
    blob = bytearray(bytes(range(256)) + b"\x99\xaa\x55\x01\xff\xaa\x55")
    for seq in range(40):
        blob += _wf(cfg, seq)
    steps, pos, now = [], 0, 1000
    while pos < len(blob):
        size = rng.choice((1, 2, 3, 4, 5, 5, 5, 40, 90))
        steps.append((now, bytes(blob[pos:pos + size])))
        pos += size
        now += 1
    steps.append((now, _wf(cfg, 40)[:5]))                    # 半帧
    steps.append((now + 60, _wf(cfg, 41)))                   # 60 ms > byte_timeout_ms=50
    rx, accepted, _ = _wire_run(cfg, steps)
    assert [a[0] for a in accepted] == list(range(40)) + [41], [a[0] for a in accepted]
    assert all(raw == _wf(cfg, seq) for seq, _hb, _act, raw in accepted), "滑窗重同步拼出了非原始帧"
    assert rx.partial_timeouts == 1, "半帧静默 60 ms 后没有被超时清零"


def test_pty_wire_dropped_frames():
    """故障 3：丢帧。按固定规律丢掉约四分之一的帧（含相邻两帧），帧间隔 20 ms；
    幸存的帧全部按序接受，丢帧本身不算乱序，也不触发失联。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)
    dropped = {s for s in range(120) if s % 7 == 3 or s % 11 == 5}
    assert any(s in dropped and s + 1 in dropped for s in range(119)), "夹具里应有相邻两帧同时被丢"
    survivors = [s for s in range(120) if s not in dropped]
    rx, accepted, _ = _wire_run(cfg, [(20 * s, _wf(cfg, s)) for s in survivors])
    assert [a[0] for a in accepted] == survivors, f"幸存的帧没有全部按序接受：{[a[0] for a in accepted]}"
    assert rx.seq_drops == 0, "丢帧被当成了乱序"
    assert rx.failsafe_count == 0 and rx.crc_errors == 0, "丢帧不该触发失联，也不该出 CRC 错"
    assert rx.current_action == accepted[-1][2]


def test_pty_wire_out_of_order_and_duplicates():
    """故障 4：乱序与重复。60 帧里三处相邻对调、帧 10 重复两次、帧 30 在帧 31 之后重放；
    期望按「后面不到半窗口才算新」逐帧推演，被拒的旧帧不改当前动作，也不改序号锚点。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)
    order = list(range(60))
    for a in (5, 20, 40):
        order[a], order[a + 1] = order[a + 1], order[a]
    order[11:11] = [10, 10]                                  # 帧 10 之后再来两个重复
    order.insert(order.index(31) + 1, 30)                    # 帧 31 之后重放帧 30
    expected, last, cur, expected_trace = [], None, 0, []
    for seq in order:
        if last is None or 0 < (seq - last) % 65536 < 32768:
            expected.append(seq)
            last, cur = seq, 1 + seq % 3
        expected_trace.append(cur)
    rx, accepted, trace = _wire_run(cfg, [(20 * i, _wf(cfg, seq)) for i, seq in enumerate(order)])
    assert [a[0] for a in accepted] == expected, f"接受的帧与逐帧推演不一致：{[a[0] for a in accepted]}"
    assert trace == expected_trace, "被拒收的旧帧或重复帧改动了当前动作"
    assert rx.seq_drops == len(order) - len(expected) and rx.seq_drops >= 6


def test_pty_wire_seq_wraparound_16bit_and_legacy_8bit():
    """故障 5：序号回绕。默认 16 位（65524 起跨 65535->0）与旧 8 位 CRC-8 配置（248 起跨 255->0）各一遍，
    回绕之后重放一个回绕之前的旧帧必须被拒。"""
    legacy = ua.validate_config(json.loads(json.dumps(LEGACY_V1_CONFIG)))
    for cfg, start, window in ((ua.load_config(EXAMPLE_CONFIG), 65524, 65536), (legacy, 248, 256)):
        seqs = [(start + i) % window for i in range(24)]
        assert seqs[0] > seqs[-1], "夹具要真的越过回绕点"
        order = seqs[:14] + [seqs[5]] + seqs[14:]            # 越过回绕点之后重放回绕之前的旧帧
        rx, accepted, _ = _wire_run(cfg, [(20 * i, _wf(cfg, seq)) for i, seq in enumerate(order)])
        assert [a[0] for a in accepted] == seqs, f"window={window}"
        assert rx.seq_drops == 1, f"window={window}"


def test_pty_wire_stall_350ms_link_loss_and_recovery():
    """故障 6：断流 350 ms。动作帧与心跳帧交替发完之后停发：299 ms 还在线、350 ms 进 LINK_LOST 且动作回 SAFE_NEUTRAL；
    发送端重启从 seq=0 起重新发帧（也经伪终端）被收下，链路恢复。"""
    cfg = ua.load_config(EXAMPLE_CONFIG)                     # ttl_ms=300
    rx, accepted, trace = _wire_run(cfg, [(20 * s, _wf(cfg, s, hb=(s % 2 == 1))) for s in range(30)])
    last_t = 20 * 29
    assert trace[-1] == 1 + 28 % 3, "心跳帧不该改动当前动作"
    assert rx.link_state == uc.LINK_ACTIVE and rx.failsafe_count == 0, "帧间隔 20 ms，不该触发失联"
    rx.tick(last_t + 299)
    assert rx.link_state == uc.LINK_ACTIVE and rx.current_action == 1 + 28 % 3, "断流 299 ms 就失联了"
    rx.tick(last_t + 350)
    assert rx.link_state == uc.LINK_LOST, "断流 350 ms 没有进入 LINK_LOST"
    assert rx.current_action == 0 and rx.failsafe_count == 1, "失联后动作没有回 SAFE_NEUTRAL"
    _rx, again, _ = _wire_run(cfg, [(last_t + 400, _wf(cfg, 0, act=1))], rx=rx)
    assert [a[0] for a in again] == [0], "失联之后从 seq=0 重新发帧没有被收下"
    assert rx.link_state == uc.LINK_ACTIVE and rx.current_action == 1


def test_sender_open_serial_is_transparent_for_all_byte_values():
    """发送端 open_serial 配出的串口在伪终端上不改任何字节（OPOST/ONLCR、ICRNL、XON/XOFF 都关掉）：
    0..255 全部字节值各两遍原样到达，其中 0x0A、0x0D、0x11、0x13、0x03、0x1A、0x7F 都是 tty 默认会处理的。"""
    master, slave = os.openpty()
    try:
        cfg = ua.load_config(EXAMPLE_CONFIG)
        cfg["device"] = os.ttyname(slave)
        fd = ua.open_serial(cfg)
        try:
            payload = bytes(range(256)) * 2
            ua.write_frame(fd, payload)
        finally:
            os.close(fd)
        got = bytearray()
        give_up = time.monotonic() + WIRE_TIMEOUT_S
        while len(got) < len(payload):
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                got += os.read(master, len(payload) - len(got))
            assert time.monotonic() < give_up, f"伪终端 {WIRE_TIMEOUT_S:g} s 内没有送出写入的字节"
        assert bytes(got) == payload, "串口配置改动了字节"
    finally:
        os.close(master)
        os.close(slave)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        test_example_config_upgraded_and_no_sys_clock()
        test_bit_flips_zero_bad_frames()
        test_sticky_fragmented_resync_and_partial_timeout()
        test_drop_out_of_order_and_wraparound()
        test_half_window_boundary_for_all_seq_widths()
        test_heartbeat_semantics_action_expiry_and_link_loss()
        test_u32_wrap_link_loss()
        test_u32_wrap_action_expiry()
        test_u32_wrap_partial_frame_timeout()
        test_u32_span_limit_for_ttl_and_duration()
        test_pty_end_to_end_fault_injection_and_link_loss(temp)
        test_pty_wire_random_bit_flips()
        test_pty_wire_sticky_fragmented_and_partial_timeout()
        test_pty_wire_dropped_frames()
        test_pty_wire_out_of_order_and_duplicates()
        test_pty_wire_seq_wraparound_16bit_and_legacy_8bit()
        test_pty_wire_stall_350ms_link_loss_and_recovery()
        test_sender_open_serial_is_transparent_for_all_byte_values()
    print("PASS: uart_codec 滑窗重同步/CRC16/半窗口回绕/心跳语义/动作有效期/PTY故障注入 全部通过")


if __name__ == "__main__":
    main()
