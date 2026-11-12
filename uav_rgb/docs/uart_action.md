# LicheePi → ESP32 四动作 UART 收发参考骨架

> 状态（2026-11-12）：**占位协议，未联调、未上机**。`tools/uart_action.py`（下行发送端）与 `tools/uart_codec.py`（接收端状态机的参考实现，写成可逐行移植为固件 C 代码的风格）只在宿主机上验证过：管道、`os.openpty()` 伪终端、注入时钟。ESP32 固件侧解析代码、上行回执帧版式、真实物理串口（引脚、电平、误码率）都没有做，等硬件事实确认后再补。当前开源 `CF-Drone` 固件没有四动作 UART 解析器，直接接线不会让飞机执行动作。

优先顺序：串口（先用串口调试助手验证字节流）→ 不行再考虑 WiFi。两者当前都没有可用的四动作接收端，见 §8。

## 1. 固件事实（只读核对 `../CF-Drone`，2026-11-12 对照源码复核）

| 文件 | 事实 |
| --- | --- |
| `CF-Drone.ino:26` | `Serial.begin(115200)`，USB 串口是明文 CLI 控制台 |
| `cli.ino` | `handleInput()`（`:230`）逐字符读 `Serial`，遇 `\n` 就把累积的文本交给 `doCommand()`（`:105`）。`doCommand()` 执行 `arm`（`:147`）、`disarm`、`auto`（`:159`）、单电机测试 `mfr`/`mfl`/`mrr`/`mrl` 等命令，不认识的回 `Invalid command`（`:226`）；**不能把二进制动作帧发到这里** |
| `board_config.h`、`rc.ino` | `BOARD_RC_SERIAL`：ESP32 与 ESP32-S3 是 `Serial2`，ESP32-C3 是 `Serial1`（`board_config.h:111,68,21`），被 CRSF(420000)/SBUS(100000) 遥控占用。`Serial1`、`Serial2` 在固件里只通过这个宏出现 |
| `mavlink.ino` | 整个文件在 `#if WIFI_ENABLED` 内（`:4`）；`WIFI_ENABLED` 取自 `BOARD_WIFI_ENABLED`（`CF-Drone.ino:12`），ESP32-C3 为 `0`（`board_config.h:49`，注释写明为释放单核 CPU），ESP32 与 S3 为 `1`。能收 `MANUAL_CONTROL`（`:94`）、`SET_ATTITUDE_TARGET`（`:173`）、`SET_ACTUATOR_CONTROL_TARGET`（`:195`）等，**没有四动作消息** |
| `web_rc.ino` | 整个文件在 `#if WEB_RC_ENABLED` 内（`:3`）；WiFi HTTP 摇杆、按钮和控制台，输出是连续控制量，不是四动作；ESP32-C3 为 `0`（`board_config.h:50`） |

因此：

- 现有固件的 UART 只有 CLI 和 RC 两种用途，没有「收 0..3 动作」的解析器。
- 把动作帧接到 USB `Serial` 会被当成 CLI 文本。乱码通常只触发 `Invalid command`，但同一入口也能执行 `arm`、`auto`、单电机测试，所以**禁止**把动作帧接到 CLI。
- 把动作帧接到 RC 串口会破坏 CRSF/SBUS 解析，**禁止**。
- 真机动作链路需要另一版固件实现解析：按 `ALVLN/docs/08_CF_DRONE_SAFETY.md` §5 的 PR-F1（解析、CRC、ACK、超时、日志，先无桨台架）。本仓库不改固件。

## 2. 最小物理链路与待确认的硬件事实

| LicheePi SG2002 | ESP32 | 说明 |
| --- | --- | --- |
| UART TX（3.3V TTL） | 选定 UART 的 RX | 交叉连接 |
| UART RX（3.3V TTL） | 选定 UART 的 TX | 可选。下行工具只发不收，上行回执要用它 |
| GND | GND | 必须共地 |
| 不要接 5V | 无 | 3.3V TTL；电平与引脚以实物丝印为准 |

接线前缺一项就不要接线：

1. **LicheeRV Nano 的串口设备名与引脚复用**（`/dev/ttyS*`）。审核意见 `docs/reviews/2026-11-12-review-round2/review2-n.md:94` 引用厂商手册 `rvnano_5_peripheral.md` 说：`UART0` 默认挂 Linux 控制台与内核日志；带 WiFi 的 W/WE 版 `GPIOP18..23` 被 SDIO 占用，`UART3` 不可用；候选串口是 `UART1`（`/dev/ttyS1`，`GPIOA28/29`，未焊蓝牙排阻时可用）。**未核实**：该手册不在本仓库和数据盘里，这几条没有对照过原文。上板后用逻辑分析仪看上电期间有没有杂散输出。
2. **ESP32 子型号与固件路线**：飞控是 ESP32、ESP32-S3 还是 ESP32-C3，尚未由实物确认（`ALVLN/docs/08_CF_DRONE_SAFETY.md` §1）。按现有固件的用法，C3 上控制台 `Serial` 与 RC 的 `Serial1` 都已占用；ESP32 与 S3 的 `Serial1` 在固件里没有被用到，引脚是否空闲要看实板。「ESP32-C3 只有两个硬件 UART」出自审核意见 `review2-n.md:94`，**未核实**。真机用官方闭源固件还是改造开源 `CF-Drone` 固件，也没有定。
3. **电气连接**：两端共地、3.3V TTL、波特率一致。默认 `115200` 与 `/dev/ttyS1` 只是占位值。

## 3. 发送端用法、EOF 行为与退出码

默认读 `config/uart_protocol.example.json`；所有帧字段都由该 JSON 决定。

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb

# 只编码单帧并退出（seq=0，不打开串口）
python3 tools/uart_action.py --dry-run --action 1
# aa 55 01 00 00 01 00 2c 01 b6 03 19 b4

# 管道灌入两行动作后关闭（EOF）：按序发 1、3，再补一帧 SAFE_NEUTRAL(0)，退出码 0
printf '1\n3\n' | python3 tools/uart_action.py --dry-run
# aa 55 01 00 00 01 00 2c 01 b6 03 19 b4
# aa 55 01 01 00 03 00 2c 01 b6 03 8a 78
# aa 55 01 02 00 00 00 2c 01 b6 03 1f 7e

# 只发心跳（拿字节流对照串口调试助手）：上游必须保持打开，标准输入一关就按 EOF 处理。
# 这里用 sleep 1 撑住 1 s，4 帧在约 0.4 s 内发完（10 Hz）
python3 tools/uart_action.py --dry-run --frames 4 < <(sleep 1)

# 常驻流模式：推理进程每行输出一个动作 ID，保持管道打开
<推理进程> | python3 tools/uart_action.py --config config/uart_protocol.example.json

# 真实串口（设备必须存在、有权限；当前固件仍不会解析）：发一帧 SAFE_NEUTRAL 后退出
python3 tools/uart_action.py --config config/uart_protocol.example.json --action 0
```

运行期行为（卡 DEP1 修订，测试清单见 §9）：

- **`--action ID`**：发一帧动作后退出，`seq=0`，不发心跳，不补 `SAFE_NEUTRAL`。
- **增量行缓冲与多行排空**（N-1a、N-1b）：从标准输入按字节块增量读入。每个完整行（以 `\n` 结尾、内容是 0..3 的整数，空行忽略）立即发一帧动作，同一批到达的多行按到达顺序在同一轮全部发出。没有换行的半行先暂存，不阻塞心跳，也不提前当动作发；无换行的残行超过 256 字节按错误处理。
- **心跳**：按 `heartbeat_hz` 发 `heartbeat=1` 的帧，action 字段写 `heartbeat_action`（默认 0，即 `SAFE_NEUTRAL`），duration 字段写 0。发出动作帧本身已刷新接收端的链路存活计时，所以同一轮不再追发心跳，下一次心跳顺延一个周期（N-1d）。
- **序号**：动作帧、心跳帧和补发的 `SAFE_NEUTRAL` 帧共用一个计数，从 0 起，按 seq 字段宽度回绕。
- **`ttl_ms`、`duration_ms`** 是配置里的常量，写进每帧。本工具不递减，接收端按本地接收时刻起算。
- **上游关闭管道（EOF）**（N-1c）：排空缓冲里的完整行后，如果还没达到 `--frames` 上限，补发一帧 `SAFE_NEUTRAL(0)` 的动作帧并立即退出，不再发心跳，把后续的链路失联判定交给接收端的 `ttl`。干净 EOF 退出码 0；缓冲里留有未换行的半行时，同样先补发 `SAFE_NEUTRAL`，再以 2 退出。标准输入接 `/dev/null` 等于立刻 EOF：只发一帧 `SAFE_NEUTRAL` 就退出，不会持续发心跳。
- **上游静默超时** `--upstream-timeout-ms`（默认 1200，0 关闭）：管道没关，但超过这个时间没有新动作行，就补发一帧 `SAFE_NEUTRAL` 并向 stderr 告警；之后心跳照常，收到新动作后恢复监测。默认值是占位，来源是动作窗口 hold 1000 ms 加 200 ms 余量。采集端的停顿门限是名义步周期 900 ms 加 200 ms，即 1100 ms（`sim/tools/collect.py` 的 `NOMINAL_STEP_PERIOD_S`、`STALL_MARGIN_S`），两者差 100 ms，是否对齐待定。接收端按帧里的 `duration`（示例 950 ms）自行到期回 `SAFE_NEUTRAL`，早于这个超时，所以这个超时只负责发显式 SAFE 帧和告警。
- **退出码**：
  - `0`：正常完成（`--action` 发完、`--frames` 发满，或标准输入在行边界干净 EOF 并补发 `SAFE_NEUTRAL` 后退出）。
  - `2`：配置非法、动作行非法或越界、半行 EOF、单行超 256 字节、串口打开或写入失败（stderr 有中文原因）。
  - `130`：`Ctrl-C`。
  - 除半行 EOF 外，退出码 2 与 130 的路径**不补发** `SAFE_NEUTRAL`。此时最后一个动作由接收端兜底：最后一帧之后没有心跳续期，示例配置下 `ttl_ms=300`，最迟 300 ms 进入链路失联并回 `SAFE_NEUTRAL`。
- `--frames N` 统计所有发出的帧（动作、心跳、补发的 `SAFE_NEUTRAL`）。

串口参数：`termios` raw、8N1、无软硬流控（`cflag = CS8|CREAD|CLOCAL`，`iflag`、`oflag`、`lflag` 全清零），只写打开（`O_WRONLY|O_NOCTTY`），不发任何模型数据。`--dry-run` 完全不调用 `os.open`，设备不存在也能跑。

## 4. 占位帧格式与配置

**以下每个值都是占位，等 `real-drone/uart_protocol.md` 提供真实协议后替换。**

默认示例配置（`config/uart_protocol.example.json`，定长 13 字节，CRC-16 加 2 字节序号）：

```text
offset  field      bytes  value
0..1    magic      2      0xAA 0x55
2       version    1      0x01
3..4    seq        2      无符号小端，每帧加 1，0xFFFF 后回绕到 0x0000
5       action     1      action_encoding[0..3] = 0,1,2,3
6       heartbeat  1      0 = 动作帧，1 = 心跳帧
7..8    ttl        2      ttl_ms = 300，小端（0x2C 0x01），链路保活超时
9..10   duration   2      动作帧写 duration_ms = 950，小端（0xB6 0x03）；心跳帧写 0
11..12  crc        2      CRC-16/CCITT-FALSE（poly 0x1021，init 0xFFFF，xorout 0，reflect false），小端，覆盖 byte 0..10
```

动作帧例：action=1、seq=0 → `aa 55 01 00 00 01 00 2c 01 b6 03 19 b4`，这是 `--dry-run --action 1` 的实际输出，CRC 用标准库 `binascii.crc_hqx(data, 0xFFFF)` 独立复算一致。心跳帧：action 字段写 `heartbeat_action`，heartbeat=1，duration=0。

向后兼容：旧版配置（1 字节 seq、CRC-8 poly 0x07、没有 `duration` 字段的 9 字节帧）仍能加载，单帧编码逐字节不变，例如 action=1、seq=0 → `aa 55 01 00 01 00 2c 01 39`（`tools/test_uart_action.py` 里 `EXPECTED` 向量的既有断言，原文没有改动；这些测试通过 `main()` 里的旧版 9 字节配置夹具运行）。

### duration 字段（动作有效时长）

- **含义**：该动作帧收到之后有效的毫秒数，到期自动回 `SAFE_NEUTRAL`；新动作一到就取代旧动作，不留空档。心跳帧写 0，接收端忽略。
- **示例值 950 ms** = 名义步周期 P（900 ms，`sim/tools/collect.py` 的 `NOMINAL_STEP_PERIOD_S`）加保持余量 `guard_s`（50 ms）。语义来自卡 W1-13 第三轮（主分支提交 `847d860`，`docs/briefs/2026-11-10-phase12-task-cards.md`）：每条动作收到之后有效 P + `guard_s`，新动作一到就取代；`guard_s = 0.05 s` 由主会话定。真机的 P 与最终的 `guard_s` 还没有量过，950 是占位。`tools/test_uart_action.py` 的 N-1e 测试把示例值钉在「`NOMINAL_STEP_PERIOD_S` 加 50 ms」上，采集端改了 P，这条测试会变红，提醒同步。
- **单位与范围**：毫秒，取值 1..min(字段容量, 2^31-1)。示例的 2 字节字段容量是 65535 ms。字段放不下，或超过 2^31-1，发送端加载配置时以退出码 2 拒绝（N-1h）；`encode_frame(..., duration_ms=)` 的覆盖值受同样限制。`ttl_ms` 的范围规则相同。
- **上限 2^31-1 的原因**：接收端的毫秒时钟是 uint32，截止时刻按 `(now + 跨度) mod 2^32` 保存，到期判据 `(int32_t)(now - deadline) >= 0` 只在跨度小于 2^31 时成立；超过上限的帧，接收端按字段取值域错误丢弃。
- **没有 `duration` 字段的协议**：配置里不能写 `duration_ms`，写了加载时以退出码 2 拒绝（N-1g），否则有效时长发不到接收端却被悄悄忽略。接收端不设动作截止，动作保持到被取代或链路失联。

### 顶层配置字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `placeholder` | 否 | `true` 时每次运行都在 stderr 警告；换真实协议后改 `false` |
| `device` | 实发必填 | 串口路径；`--dry-run` 可缺省 |
| `baud` | 是 | 1..10,000,000，且当前平台的 `termios` 有对应的 `B<baud>` 常量 |
| `heartbeat_hz` | 是 | 心跳频率，正数 |
| `ttl_ms` | 是 | 写入 ttl 字段的常量，1..min(ttl 字段容量, 2^31-1) |
| `duration_ms` | 帧里有 `duration` 字段时必填，没有该字段时不许写 | 写入动作帧 duration 字段的默认值，1..min(duration 字段容量, 2^31-1) |
| `heartbeat_action` | 否 | 心跳帧里 action 字段对应的动作 ID，0..3，默认 0 |
| `action_encoding` | 是 | 长度 4 的数组，把动作 ID 0..3 映射成线上数值，每项要放得进 action 字段 |
| `crc` | 是 | `algorithm` 为 `crc8` 或 `crc16`，`poly` 必填，`init`、`xorout` 默认 0，`reflect` 默认 `false` |
| `frame` | 是 | 有序字段数组，决定线上字节顺序；`name` 不能重复 |

### `frame[]` 字段类型

| `kind` | 作用 | 参数 |
| --- | --- | --- |
| `const` | 固定字节、版本号 | `bytes`（非空数组）或 `value` + `size`（1..8），二选一，可带 `endian` |
| `seq` | 帧序号 | `size` 1..4，`endian` |
| `action` | 动作码 | `size` 1..4，`endian` |
| `heartbeat` | 心跳标志 | `size` 1..4，`endian` |
| `ttl` | TTL 常量 | `size` 1..4，`endian` |
| `duration` | 动作有效时长，可选 | `size` 1..4，`endian` |
| `crc` | 校验 | `size` 必须等于 CRC 位宽（`crc8` 为 1，`crc16` 为 2）；必须是最后一个字段 |

`seq`、`action`、`heartbeat`、`ttl`、`crc` 各必须出现一次，`duration` 至多一次，`endian` 缺省为 `little`。可配置项覆盖字段顺序、大小端、magic 与 version、动作码、seq 宽度、ttl 与 duration 宽度、CRC-8 或 CRC-16 及参数。接收端参考实现另限整帧不超过 32 字节。**不支持**变长帧、转义、ACK 与重传、request_id；真实协议若需要这些，脚本要改，不只是改配置。

## 5. 接收端参考实现（`tools/uart_codec.py`）

`UartReceiver` 是接收端状态机的参考实现，写成可逐行移植为固件 C 代码的风格：不读取系统时钟，时刻全部由调用方以毫秒整数 `now_ms` 传入；接收缓冲是固定 32 字节（`MAX_FRAME_SIZE`）的预分配数组，运行期不扩容；帧布局由 `FrameLayout` 从发送端同一份配置生成，对应固件里的只读常量结构体。`on_frame` 回调是测试钩子，会拷出一份 bytes，固件里没有这一步。

调用方式：每收到一个字节调用 `feed_byte(b, now_ms)`（或批量的 `feed_bytes`），飞控主循环每拍调用 `tick(now_ms)`；读 `current_action`（动作 ID 0..3，`SAFE_NEUTRAL=0`）与 `link_state`（`LINK_IDLE`、`LINK_ACTIVE`、`LINK_LOST`）。

### 5.1 收帧流水线与重同步

缓冲为空时，首字节不是帧头首字节就直接丢弃（`header_drops`）。一帧攒满 `frame_size` 字节后依次检查，任何一步不过都丢弃并重同步：

1. 常量字段 magic、version，不过计入 `header_drops`。
2. CRC（`crc_errors`）。
3. 字段取值域（`field_errors`）：heartbeat 只能是 0 或 1；`ttl` 在 1..2^31-1；动作帧的 action 必须是 `action_encoding` 里的值，有 `duration` 字段时它在 1..2^31-1。心跳帧不检查 action 与 duration。
4. 序号判新（`seq_drops`），见 5.4。
5. 全部通过才更新状态。

重同步：校验失败时丢掉缓冲首字节，并在定长缓冲内原地前移到下一个与帧头首字节相同的位置，不丢粘包后面的真帧，与 `CF-Drone/rc.ino:61-79` 的 `memmove` 单字节滑窗同构。半帧超时：缓冲里有残片、距上一个字节超过 `byte_timeout_ms`（默认 50 ms）就清空（`partial_timeouts`）。

### 5.2 心跳与动作严格分离

- **动作帧**（`heartbeat == 0`）：立即更新 `current_action`，新动作一到就取代旧动作，没有空档；有 `duration` 字段时设置 `action_deadline = now + duration`，没有该字段时动作保持到被取代或链路失联；同时刷新 `link_deadline = now + ttl`。
- **心跳帧**（`heartbeat == 1`）：只刷新 `link_deadline = now + ttl`，忽略帧内的 action 与 duration，不改 `current_action` 与 `action_deadline`。
- 接收端信任帧内的 `ttl` 与 `duration`，自己不设上限，只受上面的取值域检查约束。固件实现是否另设本地上限，待主会话定。

### 5.3 两级到期保护

- **动作到期**：链路 `LINK_ACTIVE` 期间，`now >= action_deadline` 且没有新动作，`current_action` 回 `SAFE_NEUTRAL(0)`，链路保持 `LINK_ACTIVE`（`action_expirations`）。之后的心跳不会把旧动作恢复回来。
- **链路失联**：`now >= link_deadline`，即示例 `ttl_ms=300` 下断流满 300 ms，状态转 `LINK_LOST`，`current_action` 回 `SAFE_NEUTRAL(0)`，`failsafe_count` 加 1，并清除序号锚点 `has_seq`，所以上位机进程重启后可以从 `seq=0` 重新同步。失联后收到合法帧，链路回到 `LINK_ACTIVE`。
- 同一拍两者都到期，按链路失联处理。

### 5.4 序号判新：半窗口无符号模运算

序号字段是 1..4 字节，位宽 W = 8、16、24 或 32，`MASK = 2^W - 1`，`HALF = 2^(W-1)`。2 字节序号下 `MASK = 65535`、`HALF = 32768`；10 Hz 心跳下约 109 分钟回绕一次（65536 / 10 Hz = 6553.6 s，动作帧也占序号，实际更快）。

- 首帧（`has_seq` 为假，上电后或失联后）接受任意合法帧，记 `last_seq = seq`。
- 之后算 `diff = (seq - last_seq) & MASK`：`0 < diff < HALF` 判为新帧，接受并更新 `last_seq`；`diff == 0`（重复、重放）或 `diff >= HALF`（乱序迟到的旧帧，包括回绕后迟到的回绕前的帧，例如 `0` 之后迟到的 `65535`）一律拒收，不刷新动作与 TTL。`diff` 恰好等于 `HALF` 也拒收，这个边界在 1、2、3、4 字节序号下都有测试（N-7c）。

### 5.5 毫秒时钟按 uint32 处理（N-7b）

固件的毫秒计数是 `uint32_t`，2^32 ms 约 49.7 天回绕。`uart_codec.py` 把 `now_ms` 取低 32 位，截止时刻按 `(now + 跨度) mod 2^32` 保存，到期判据 `_reached(now, deadline)` 等价于 C 的 `(int32_t)(now - deadline) >= 0`，半帧超时的时间差也按 uint32 取模。因此帧里的 `ttl`、`duration` 跨度不得超过 2^31-1，发送端加载配置时已拒绝更大的值（§4，N-1h）。

### 5.6 CRC-16 的误检量级

- `CRC-16/CCITT-FALSE`（生成多项式 x^16 + x^12 + x^5 + 1，含因子 x + 1）对 13 字节（104 比特）的定长帧，必然检出所有单比特错、双比特错、奇数个比特错，以及长度不超过 16 比特的突发错误。
- 对任意均匀随机的多比特翻转，纯 CRC-16 的理论漏检概率是 2^-16 = 1/65536 ≈ 1.53e-5；旧 CRC-8 是 2^-8 ≈ 3.91e-3，是前者的 256 倍。随机字节流凑出一个合法窗口的概率：3 字节常量头 2^-24 乘 CRC 2^-16 = 2^-40 ≈ 9.1e-13（每个窗口位置），字段取值域与序号检查只会让它更低。
- 以上是理论值加测试用例（500 帧、固定种子、35% 的帧注入 1..8 比特翻转，坏帧接受数为 0）。真机串口的实际误码率没有测，需要在有刷电机换向的干扰下实测。

## 6. 与 `real-drone/uart_protocol.md` 对接

该文件目前**不存在**（用户尚未提供）。拿到后按下面顺序核对，不要凭默认占位帧上机：

1. **帧边界**：magic 值、长度、是否定长；总字节数是否与默认的 13 字节一致；整帧是否不超过接收端参考实现的 32 字节上限。
2. **字段**：字段顺序、各自宽度与大小端；seq 的位宽与回绕规则（接收端用半窗口判新，见 5.4）。
3. **动作码**：0..3 的语义和线上编码，填 `action_encoding`。
4. **心跳**：心跳帧是否就是「action=0 加 heartbeat=1」；心跳频率与超时；是否要单独的消息类型。
5. **TTL 与动作有效时长**：单位（ms 还是帧数）、接收端本地时钟判定；本工具只写常量，不递减。真实协议有没有 `duration` 这类字段，决定接收端能否不依赖下一帧就让动作自己到期（见 §4）。
6. **CRC**：算法（CRC-8 或 CRC-16）、poly、init、xorout、reflect、覆盖范围、CRC 字节的大小端。当前脚本固定「覆盖 CRC 之前的所有字节，且 CRC 在最后」。
7. **ACK 与状态**：真实协议若有 ACK 或 EXEC_STATUS，本骨架未实现，必须另加，且不能把 ACK 当执行完成（`ALVLN/docs/08_CF_DRONE_SAFETY.md` §3）。上行回执帧版式等硬件事实确认后再定。

若以上都能映射到现有 schema：只改 `config/uart_protocol.example.json`（或复制一份新配置），并把 `placeholder` 设为 `false`。否则先改 `tools/uart_action.py` 与 `tools/uart_codec.py`，不要改 CF-Drone。

## 7. 串口调试助手与 WiFi

### 7.1 串口调试助手（优先）

用途只限验证字节流，不是飞行链路：

1. 用 `--dry-run` 得到 hex（只发心跳的写法见 §3）。
2. 用 USB-TTL 回环或逻辑分析仪，在串口调试助手里按 hex 发送，确认收发字节、波特率、CRC 一致。
3. 只有 ESP32 已有四动作解析固件、且通过无桨台架的 CRC、重放、乱序、TTL 注入测试之后，才把线接到 ESP32 的动作 UART。
4. **不要把调试助手接到 ESP32 的 USB `Serial`**（CLI，含 arm 与电机命令）。

### 7.2 WiFi（串口不可行时）

现有固件的 WiFi 通路没有四动作消息：

- `mavlink.ino` 的 `MANUAL_CONTROL` 是连续摇杆量，`SET_ATTITUDE_TARGET` 是姿态与推力目标；`web_rc.ino` 的 HTTP 接口同样是连续摇杆、按钮和控制台命令入口。把 0..3 动作映射成它们，会绕开「四个离散安全动作」的语义，也与 `ALVLN/docs/08_CF_DRONE_SAFETY.md` §2 的「ESP32 唯一控制仲裁、无 ACK 不判成功」冲突。
- 该 fallback 不在本骨架范围内，未实现、未验证。要做需要单独的安全评审和固件或桥接实现，不能直接拿 `MANUAL_CONTROL` 或 Web RC 当四动作。

## 8. 当前未实现项与安全边界

未实现：

- **上行回执与偏航遥测**：`uart_action.py` 只做单向下行（`O_WRONLY`），没有 ESP32 → LicheeRV Nano 的执行回执帧与偏航上行帧。板端条件码需要的上行协议，要与固件改造一起定稿，帧版式等硬件事实确认后再定。
- **固件端解析与真实串口验证**：ESP32 侧解析代码没有写，没有在真实开发板串口上收发验证。
- **真机时间参数**：`duration_ms=950`、`ttl_ms=300`、`--upstream-timeout-ms` 默认 1200、`heartbeat_hz=10` 都是占位值，没有在真机上量过 P、`guard_s` 和串口延迟。

安全边界：

- 不装桨，先台架；串口调试助手通过只代表字节对，不代表飞控能安全执行。飞行规程是无桨台架 → 系绳 → 网内 → 自由飞（`docs/ROADMAP.md:41`）。上桨前的断链演练（拔掉串口 TX 线、`kill -9` 板端推理进程、注入连续 CRC 坏帧、拨动实体 RC 开关夺权）出自审核意见 `docs/reviews/2026-11-12-review-round2/review2-n.md:137`，还不是定稿规程。
- ESP32 飞控保留最终否决权；模型与本工具只产生有界意图。
- `SAFE_NEUTRAL` 不等于已验证的安全悬停：`ALVLN/docs/08_CF_DRONE_SAFETY.md` §1 写明定高 `ALTHOLD` 缺少完整控制实现，开环下降依赖未经充分标定的悬停推力常量；心跳也不是急停。
- 链路失效依赖接收端的 `ttl` 与 `duration` 保护；本工具无法证明对端会执行该策略。
- 本工具无 ACK，发送成功不等于飞控接收或执行成功。
- 真机验证门禁见 `ALVLN/docs/08_CF_DRONE_SAFETY.md` §4 与 `ALVLN/docs/09_SG2002_DEPLOYMENT.md` 的 E6。

## 9. 验证

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb
.venv-tools/bin/python tools/test_uart_action.py
.venv-tools/bin/python tools/test_uart_codec.py
```

`tools/check.sh`（含 `--python-only`）在 `tools/test_*.py` 循环里自动收集这两个文件（`tools/check.sh:190-193`）。旧文档写的「`check.sh` 只扫 `sim/tools/test_*.py`，本测试要单独运行」与脚本不符，已更正。

`tools/test_uart_action.py`（发送端，16 个用例）：

- CRC-8 与 CRC-16 标准向量、默认帧逐字节期望值、旧 9 字节配置的单帧编码逐字节不变、动作越界、seq 递增与回绕、心跳与动作帧的区别、stdin 动作、dry-run 不碰硬件、坏配置、串口不存在或权限不足（mock，不打开真实串口）。
- N-1a 半行不阻塞心跳；N-1b 多行按序全发；N-1c EOF 补发 `SAFE_NEUTRAL` 与上游静默超时；N-1d 动作帧同轮不追发心跳与 duration 编码；N-1e 示例 duration 与采集端名义步周期同源；N-1g 有 `duration_ms` 无 `duration` 字段被拒绝；N-1h ttl、duration 上限收紧到 2^31-1。

`tools/test_uart_codec.py`（接收端，18 个用例）：

- 零系统时钟依赖与定长缓冲不扩容；500 帧固定种子随机比特翻转，坏帧接受数为 0；假 magic、粘包、拆包的重同步与半帧超时；丢帧、乱序、重放，8 位与 16 位序号跨零回绕；半窗口边界在 1..4 字节序号下的接受与拒收（N-7c）；心跳保活不改动作、动作到期回 `SAFE_NEUTRAL`、断流失联；uint32 毫秒时钟回绕下的失联倒计时、动作有效期、半帧超时，以及 ttl、duration 的跨度上限（N-7b）。
- 真实伪终端（`os.openpty()` 加 `termios` 8N1）：发送进程到接收端的全链路，六种故障（随机比特翻转、粘包拆包与半帧超时、丢帧、乱序与重复、16 位与 8 位序号回绕、断流 350 ms 失联并恢复）的逐项注入，以及发送端 `open_serial` 对 0..255 全部字节值原样透传（N-7d、N-7e）。伪终端用例的完成判据都是事件（读满字节数、收到某帧），超时只是防挂死的上限，默认 30 s，可用环境变量 `UART_TEST_WIRE_TIMEOUT_S` 调整。

实机串口与 ESP32 解析**未验证**。
