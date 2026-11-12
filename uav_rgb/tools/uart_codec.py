#!/usr/bin/env python3
"""四动作 UART 接收端参考状态机（卡 DEP1 / 审核 N-7）。

设计约束（便于逐行移植为 ESP32 固件侧无动态分配的 C 代码）：
- 不读取任何系统时钟，所有时刻均由调用方以单调毫秒整数 now_ms 传入；
- 接收缓冲固定为长度 MAX_FRAME_SIZE 的预分配字节数组，运行期零扩容，接收与校验路径不做切片分配
  （on_frame 回调是测试钩子，会拷出一份 bytes，固件里没有这一步）；
- 校验失败时按单字节起滑窗重同步（与 CF-Drone/rc.ino 的 memmove 滑窗同构），不漏粘包后的真帧；
- 序号判新采用无符号半窗口模运算：0 < ((seq - last_seq) & seq_mask) < seq_half；
- 心跳帧（heartbeat == 1）只刷新链路失联截止时刻，绝不修改当前动作及其动作有效截止时刻；
- 毫秒时钟按固件的 uint32_t 处理：截止时刻取 (now + 跨度) mod 2^32，到期判据写成
  (int32_t)(now - deadline) >= 0（见 _reached），因此帧里的 ttl、duration 跨度不得超过 2^31 - 1，
  超过的帧按字段取值域错误丢弃。
"""
import uart_action as ua

MAX_FRAME_SIZE = 32
SAFE_NEUTRAL = 0

U32_MASK = 0xFFFFFFFF
U32_HALF = 0x80000000
MAX_SPAN_MS = 0x7FFFFFFF

LINK_IDLE = 0
LINK_ACTIVE = 1
LINK_LOST = 2


def _read_uint_buf(buf, off, size, little_endian):
    """从定长字节缓冲 [off, off+size) 按端序无分配解出无符号整数（对应 C 移位拼接）。"""
    val = 0
    if little_endian:
        for i in range(size - 1, -1, -1):
            val = (val << 8) | buf[off + i]
    else:
        for i in range(size):
            val = (val << 8) | buf[off + i]
    return val


def _reached(now_ms, deadline_ms):
    """对应 C：(int32_t)(now_ms - deadline_ms) >= 0；now_ms、deadline_ms 都已取低 32 位。"""
    return ((now_ms - deadline_ms) & U32_MASK) < U32_HALF


def _crc_prefix(buf, length, bits, poly, init, xorout, reflect):
    """对 buf[0..length) 计算 CRC-8 或 CRC-16（无临时 bytes 拷贝）。"""
    crc = init
    if bits == 8:
        for idx in range(length):
            byte = ua._reflect(buf[idx], 8) if reflect else buf[idx]
            crc ^= byte
            for _ in range(8):
                crc = ((crc << 1) ^ poly) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
        if reflect:
            crc = ua._reflect(crc, 8)
        return crc ^ xorout
    for idx in range(length):
        byte = ua._reflect(buf[idx], 8) if reflect else buf[idx]
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    if reflect:
        crc = ua._reflect(crc, 16)
    return crc ^ xorout


class FrameLayout:
    """编译期常量帧描述符（对应固件侧只读常量结构体）。"""

    __slots__ = (
        "frame_size", "lead_byte", "const_mask", "const_val",
        "seq_off", "seq_size", "seq_le", "seq_mask", "seq_half",
        "act_off", "act_size", "act_le", "action_encoding",
        "hb_off", "hb_size", "hb_le",
        "ttl_off", "ttl_size", "ttl_le",
        "dur_off", "dur_size", "dur_le",
        "crc_off", "crc_size", "crc_le",
        "crc_bits", "crc_poly", "crc_init", "crc_xorout", "crc_reflect",
    )

    def __init__(self, cfg):
        cfg = ua.validate_config(dict(cfg))
        self.const_mask = bytearray(MAX_FRAME_SIZE)
        self.const_val = bytearray(MAX_FRAME_SIZE)
        self.dur_off = 0
        self.dur_size = 0
        self.dur_le = True

        off = 0
        for field in cfg["frame"]:
            kind = field["kind"]
            le = field.get("endian", "little") == "little"
            if kind == "const":
                if "bytes" in field:
                    raw = bytes(field["bytes"])
                else:
                    raw = int(field["value"]).to_bytes(field["size"], "little" if le else "big")
                size = len(raw)
                if off + size > MAX_FRAME_SIZE:
                    raise ua.ProtocolError(f"帧总长超过接收缓冲上限 {MAX_FRAME_SIZE} 字节")
                for i in range(size):
                    self.const_mask[off + i] = 1
                    self.const_val[off + i] = raw[i]
                off += size
                continue
            size = field["size"]
            if off + size > MAX_FRAME_SIZE:
                raise ua.ProtocolError(f"帧总长超过接收缓冲上限 {MAX_FRAME_SIZE} 字节")
            if kind == "seq":
                self.seq_off, self.seq_size, self.seq_le = off, size, le
                self.seq_mask = (1 << (8 * size)) - 1
                self.seq_half = 1 << (8 * size - 1)
            elif kind == "action":
                self.act_off, self.act_size, self.act_le = off, size, le
            elif kind == "heartbeat":
                self.hb_off, self.hb_size, self.hb_le = off, size, le
            elif kind == "ttl":
                self.ttl_off, self.ttl_size, self.ttl_le = off, size, le
            elif kind == "duration":
                self.dur_off, self.dur_size, self.dur_le = off, size, le
            elif kind == "crc":
                self.crc_off, self.crc_size, self.crc_le = off, size, le
            off += size

        self.frame_size = off
        self.lead_byte = self.const_val[0] if self.const_mask[0] else -1
        self.action_encoding = tuple(int(v) for v in cfg["action_encoding"])
        crc_cfg = cfg["crc"]
        self.crc_bits = 8 if crc_cfg["algorithm"] == "crc8" else 16
        self.crc_poly = crc_cfg["poly"]
        self.crc_init = crc_cfg["init"]
        self.crc_xorout = crc_cfg["xorout"]
        self.crc_reflect = crc_cfg["reflect"]


class UartReceiver:
    """无动态分配风格的四动作 UART 接收状态机。"""

    __slots__ = (
        "layout", "byte_timeout_ms",
        "rx_buf", "rx_len", "has_byte_ts", "last_byte_ms",
        "link_state", "current_action",
        "has_seq", "last_seq",
        "link_deadline_ms", "has_action_deadline", "action_deadline_ms",
        "accepted_actions", "accepted_heartbeats",
        "crc_errors", "header_drops", "field_errors", "seq_drops",
        "partial_timeouts", "action_expirations", "failsafe_count",
    )

    def __init__(self, cfg, byte_timeout_ms=50):
        self.layout = cfg if isinstance(cfg, FrameLayout) else FrameLayout(cfg)
        if isinstance(byte_timeout_ms, bool) or not isinstance(byte_timeout_ms, int) or byte_timeout_ms < 0:
            raise ua.ProtocolError("byte_timeout_ms 必须是 >= 0 的整数")
        self.byte_timeout_ms = byte_timeout_ms
        self.rx_buf = bytearray(MAX_FRAME_SIZE)
        self.reset()

    def reset(self):
        self.rx_len = 0
        self.has_byte_ts = False
        self.last_byte_ms = 0
        self.link_state = LINK_IDLE
        self.current_action = SAFE_NEUTRAL
        self.has_seq = False
        self.last_seq = 0
        self.link_deadline_ms = 0
        self.has_action_deadline = False
        self.action_deadline_ms = 0
        self.accepted_actions = 0
        self.accepted_heartbeats = 0
        self.crc_errors = 0
        self.header_drops = 0
        self.field_errors = 0
        self.seq_drops = 0
        self.partial_timeouts = 0
        self.action_expirations = 0
        self.failsafe_count = 0

    def _slide_resync(self):
        """丢弃当前首字节并在定长缓冲内原地前移，寻找下一个候选帧头字节。"""
        n = self.rx_len
        lead = self.layout.lead_byte
        shift = 1
        if lead >= 0:
            while shift < n and self.rx_buf[shift] != lead:
                shift += 1
        rem = n - shift
        for i in range(rem):
            self.rx_buf[i] = self.rx_buf[i + shift]
        self.rx_len = rem

    def _decode_action(self, wire_action):
        enc = self.layout.action_encoding
        for act_id in range(4):
            if enc[act_id] == wire_action:
                return act_id
        return -1

    def tick(self, now_ms):
        """推进超时状态（调用方按飞控主循环节拍传入单调毫秒整数 now_ms）。"""
        now_ms = int(now_ms) & U32_MASK
        if self.link_state == LINK_ACTIVE and _reached(now_ms, self.link_deadline_ms):
            self.link_state = LINK_LOST
            self.current_action = SAFE_NEUTRAL
            self.has_action_deadline = False
            # 失联后清空序号锚点，允许上位机进程重启后从 seq=0 重新建立同步
            self.has_seq = False
            self.failsafe_count += 1
        elif (self.link_state == LINK_ACTIVE and self.has_action_deadline
              and _reached(now_ms, self.action_deadline_ms)):
            self.current_action = SAFE_NEUTRAL
            self.has_action_deadline = False
            self.action_expirations += 1

        if (self.rx_len > 0 and self.byte_timeout_ms > 0
                and self.has_byte_ts and ((now_ms - self.last_byte_ms) & U32_MASK) > self.byte_timeout_ms):
            self.rx_len = 0
            self.partial_timeouts += 1

    def feed_byte(self, byte_val, now_ms, on_frame=None):
        """喂入单个字节；若完成并接受一帧有效新帧则返回 True，否则返回 False。"""
        now_ms = int(now_ms) & U32_MASK
        self.tick(now_ms)
        self.last_byte_ms = now_ms
        self.has_byte_ts = True

        L = self.layout
        b = byte_val & 0xFF
        if self.rx_len == 0 and L.lead_byte >= 0 and b != L.lead_byte:
            self.header_drops += 1
            return False

        self.rx_buf[self.rx_len] = b
        self.rx_len += 1
        if self.rx_len < L.frame_size:
            return False

        # 1) 固定常量字段（magic / version 等）校验
        for i in range(L.frame_size):
            if L.const_mask[i] and self.rx_buf[i] != L.const_val[i]:
                self.header_drops += 1
                self._slide_resync()
                return False

        # 2) CRC 校验
        calc_crc = _crc_prefix(
            self.rx_buf, L.crc_off, L.crc_bits,
            L.crc_poly, L.crc_init, L.crc_xorout, L.crc_reflect,
        )
        wire_crc = _read_uint_buf(self.rx_buf, L.crc_off, L.crc_size, L.crc_le)
        if calc_crc != wire_crc:
            self.crc_errors += 1
            self._slide_resync()
            return False

        # 3) 字段取值域校验
        seq = _read_uint_buf(self.rx_buf, L.seq_off, L.seq_size, L.seq_le)
        wire_act = _read_uint_buf(self.rx_buf, L.act_off, L.act_size, L.act_le)
        hb = _read_uint_buf(self.rx_buf, L.hb_off, L.hb_size, L.hb_le)
        ttl = _read_uint_buf(self.rx_buf, L.ttl_off, L.ttl_size, L.ttl_le)
        dur = _read_uint_buf(self.rx_buf, L.dur_off, L.dur_size, L.dur_le) if L.dur_size > 0 else 0

        if hb not in (0, 1) or ttl < 1 or ttl > MAX_SPAN_MS:
            self.field_errors += 1
            self._slide_resync()
            return False

        act_id = SAFE_NEUTRAL
        if hb == 0:
            act_id = self._decode_action(wire_act)
            if act_id < 0 or (L.dur_size > 0 and (dur < 1 or dur > MAX_SPAN_MS)):
                self.field_errors += 1
                self._slide_resync()
                return False

        # 语法与 CRC 均合法，整帧消费完毕（不再对合法帧内部滑窗）
        raw_accepted = bytes(self.rx_buf[:L.frame_size]) if on_frame is not None else None
        self.rx_len = 0

        # 4) 半窗口无符号模运算判新：拒绝重复帧（diff == 0）与乱序旧帧（diff >= seq_half）
        if self.has_seq:
            diff = (seq - self.last_seq) & L.seq_mask
            if diff == 0 or diff >= L.seq_half:
                self.seq_drops += 1
                return False
        self.has_seq = True
        self.last_seq = seq

        # 5) 更新链路与动作状态：心跳只续期链路，不改当前动作与动作截止
        self.link_state = LINK_ACTIVE
        self.link_deadline_ms = (now_ms + ttl) & U32_MASK
        if hb == 1:
            self.accepted_heartbeats += 1
        else:
            self.current_action = act_id
            if L.dur_size > 0:
                self.has_action_deadline = True
                self.action_deadline_ms = (now_ms + dur) & U32_MASK
            else:
                self.has_action_deadline = False
            self.accepted_actions += 1

        if on_frame is not None:
            on_frame(raw_accepted, {
                "seq": seq,
                "heartbeat": hb,
                "action": -1 if hb == 1 else act_id,
                "ttl_ms": ttl,
                "duration_ms": dur,
            })
        return True

    def feed_bytes(self, data, now_ms, on_frame=None):
        """批量喂入字节流，返回本批被接受的有效新帧数量。"""
        accepted = 0
        for b in data:
            if self.feed_byte(b, now_ms, on_frame=on_frame):
                accepted += 1
        return accepted
