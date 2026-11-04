# LicheePi → ESP32 四动作 UART 发送骨架

> 状态（2026-11-05）：**占位协议，未联调、未上机**。本文和 `tools/uart_action.py`
> 只解决「把 0..3 动作编码成可配置帧并发到串口」，不做模型推理、不修改 CF-Drone 固件。
> 当前 CF-Drone 固件**没有四动作 UART 解析器**，直接接串口不会让飞机执行动作。

## 1. 固件事实（只读核对 `../CF-Drone`）

| 文件 | 事实 |
| --- | --- |
| `CF-Drone.ino:26` | `Serial.begin(115200)`，USB 串口是 CLI 控制台 |
| `cli.ino` | `handleInput()` 逐行读 `Serial`，`doCommand()` 执行 `arm`/`disarm`/`auto`/马达测试等命令；**不能把二进制动作帧发到这里** |
| `rc.ino` | `BOARD_RC_SERIAL`（classic/S3 = `Serial2`，C3 = `Serial1`）被 CRSF(420000)/SBUS(100000) 遥控占用 |
| `mavlink.ino` | 仅 `WIFI_ENABLED` 时编译；WiFi 上收发 MAVLink。可收 `MANUAL_CONTROL`、`SET_ATTITUDE_TARGET`、`SET_ACTUATOR_CONTROL_TARGET` 等，**没有四动作消息或 NAV_INTENT** |
| `web_rc.ino` | 仅 `WEB_RC_ENABLED` 时编译；WiFi HTTP 摇杆/按钮/控制台，输出同样是连续 `control*` 量，不是四动作 |

因此：

- 现有固件的 UART 只有 CLI 和 RC 两种用途，没有「收 0..3 动作」的解析器。
- 把动作帧接到 USB `Serial` 会被当成 CLI 文本；乱码最多触发 `Invalid command`，
  但同一入口也能执行 `arm`/`auto`/马达测试，**禁止**把导航帧接到 CLI。
- 把动作帧接到 RC 串口会破坏 CRSF/SBUS 解析，**禁止**。
- 真机动作链路必须先有另一版固件实现解析（按 `ALVLN/docs/08_CF_DRONE_SAFETY.md`
  的 PR-F1：解析/CRC/ACK/超时/日志，先无桨台架），本任务不改固件。

优先顺序：**串口（先用串口调试助手验证字节流）→ 不行再考虑 WiFi**。两者当前都
没有可用的四动作接收端，见 §6。

## 2. 最小物理链路（待硬件确认）

| LicheePi SG2002 | ESP32 | 说明 |
| --- | --- | --- |
| UART TX（3.3V TTL） | 选定 UART 的 RX | 交叉连接 |
| UART RX（3.3V TTL） | 选定 UART 的 TX | 可选，本工具只发不收 |
| GND | GND | 必须共地 |
| 不要接 5V | — | 3.3V TTL；电平/引脚以实物丝印为准 |

前置确认（缺一项就不要接线）：

1. LicheePi 上实际串口设备名（`/dev/ttyS*`）与引脚复用；SG2002 的 pinmux/设备树
   不在本仓库范围内。
2. ESP32 型号（classic/S3/C3）和空闲 UART 实例。按 `board_config.h`，现有 RC 已占用
   `Serial1`/`Serial2`；要用 UART 动作通道必须另选实例或改固件。
3. 两端共地、3.3V TTL、波特率一致。默认配置的 `115200` 只是占位。

## 3. 工具用法

默认读 `config/uart_protocol.example.json`；所有帧字段均由该 JSON 决定。

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb

# 只编码、不碰硬件：打印一帧 hex
python3 tools/uart_action.py --dry-run --action 1
# aa 55 01 00 01 00 2c 01 39

# 心跳模式：10 Hz，发 3 帧后退出（便于 dry-run / 串口调试助手对照）
python3 tools/uart_action.py --dry-run --frames 3

# 从 stdin 读动作 ID（每行一个），同时按 heartbeat_hz 发心跳
printf '1\n3\n' | python3 tools/uart_action.py --dry-run --frames 5

# 真实串口（设备必须存在、有权限；当前固件仍不会解析）
python3 tools/uart_action.py --config config/uart_protocol.example.json
```

行为约定：

- `--action ID`：发一帧动作后退出，seq=0。
- 不给 `--action`：进入心跳模式；stdin 每读到一行 0..3 就立即发一帧动作。
- 心跳模式在 stdin EOF 后**继续发心跳**，直到 Ctrl-C 或 `--frames` 用尽；生产管线
  应保持 stdin 打开，进程退出后由接收端 TTL 兜底。
- 每帧 seq 递增，按 seq 字段位宽回绕；心跳帧和动作帧共用同一 seq 计数。
- TTL 是常量，写在每帧里；本工具不做递减，接收端必须用本地接收时间判过期。
- 退出码：`0` 正常，`2` 配置/协议/串口错误（stderr 有中文原因），`130` Ctrl-C。
- `--dry-run` 完全不调用 `os.open`，设备不存在也能跑。

串口参数：`termios` raw、8N1、无软硬流控，不发任何模型数据。

## 4. 默认占位协议

**以下每个值都是占位，等 `real-drone/uart_protocol.md` 提供真实协议后替换。**

```text
offset  field      bytes  value
0..1    magic      2      0xAA 0x55
2       version    1      0x01
3       seq        1      每帧 +1，0xFF 后回 0x00
4       action     1      action_encoding[0..3] = 0,1,2,3
5       heartbeat  1      1=心跳帧，0=动作帧
6..7    ttl        2      300 ms，小端（0x2C 0x01）
8       crc        1      CRC-8 poly=0x07 init=0 xorout=0，覆盖 byte 0..7
```

动作帧例：action=1、seq=0 → `aa 55 01 00 01 00 2c 01 39`。
心跳帧：action 字段写 `heartbeat_action`（默认 0 = SAFE_NEUTRAL），heartbeat=1。

`config/uart_protocol.example.json` 顶层字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `placeholder` | 否 | `true` 时脚本每次运行都在 stderr 警告；换真实协议后改 `false` |
| `device` | 实发必填 | 串口路径；`--dry-run` 可缺省 |
| `baud` | 是 | termios 支持的波特率 |
| `heartbeat_hz` | 是 | 心跳频率（>0） |
| `ttl_ms` | 是 | 写入 ttl 字段的常量，必须能放进 ttl 字段 |
| `heartbeat_action` | 否 | 心跳帧里 action 字段的值，默认 0 |
| `action_encoding` | 是 | 长度 4 的数组，把动作 ID 0..3 映射成线上数值 |
| `crc` | 是 | `algorithm`=`crc8`/`crc16`，以及 `poly`/`init`/`xorout`/`reflect` |
| `frame` | 是 | 有序字段数组，决定线上字节顺序 |

`frame[]` 字段类型：

| `kind` | 作用 | 必填参数 |
| --- | --- | --- |
| `const` | 固定字节/版本号 | `bytes` 或 `value`+`size`(+`endian`) |
| `seq` | 帧序号 | `size` 1..4，`endian` |
| `action` | 动作码 | `size` 1..4，`endian` |
| `heartbeat` | 心跳标志 | `size` 1..4，`endian` |
| `ttl` | TTL 常量 | `size` 1..4，`endian` |
| `crc` | 校验 | `size` 必须等于 CRC 位宽；必须是最后一个字段 |

可配置项覆盖字段顺序、大小端、magic/version、动作码、seq 宽度、TTL 宽度、
CRC-8/CRC-16 及参数。**不支持**变长帧、转义、ACK/重传、request_id；真实协议
若需要这些，脚本要改，不只是改配置。

## 5. 与 `real-drone/uart_protocol.md` 对接

该文件目前**不存在**（用户尚未提供）。拿到后按下面顺序核对，不要凭默认占位帧上机：

1. **帧边界**：magic 值、长度、是否定长；总字节数是否与默认 9 B 一致。
2. **字段**：字段顺序、各自宽度/大小端；seq 回绕规则。
3. **动作码**：0..3 的语义和线上编码，填 `action_encoding`。
4. **心跳**：心跳帧是否就是「action=0 + heartbeat=1」；心跳频率/超时；是否要
   单独的消息类型。
5. **TTL**：单位（ms/帧数）、接收端本地时钟判定；本工具只写常量，不递减。
6. **CRC**：算法（CRC-8/CRC-16）、poly/init/xorout/reflect、覆盖范围、CRC 字节
   大小端。当前脚本固定「覆盖 CRC 之前的所有字节且 CRC 在最后」。
7. **ACK/状态**：真实协议若有 ACK/EXEC_STATUS，本骨架未实现，必须另加，且不能
   把 ACK 当执行完成（`ALVLN/docs/08_CF_DRONE_SAFETY.md §3`）。

若以上都能映射到现有 schema：只改 `config/uart_protocol.example.json`（或复制一份
新配置）并把 `placeholder` 设为 `false`。否则先改 `tools/uart_action.py`，
不要改 CF-Drone。

## 6. 串口调试助手 / WiFi fallback

### 6.1 串口调试助手（优先）

用途仅限**验证字节流**，不是飞行链路：

1. `--dry-run` 得到 hex。
2. 用 USB-TTL 回环或逻辑分析仪，在串口调试助手里按 hex 发送，确认收发字节、
   波特率、CRC 一致。
3. 只有在 ESP32 已有四动作解析固件、且通过无桨台架的 CRC/重放/乱序/TTL 注入
   测试后，才把线接到 ESP32 的动作 UART。
4. **不要把调试助手接到 ESP32 的 USB `Serial`**（CLI，含 arm/马达命令）。

### 6.2 WiFi（串口不可行时）

现有固件的 WiFi 通路没有四动作消息：

- `mavlink.ino` 的 `MANUAL_CONTROL` 是连续摇杆量，`SET_ATTITUDE_TARGET` 是姿态/推力
  目标；`web_rc.ino` 的 HTTP 接口同样是连续摇杆/按钮（含控制台命令入口）。把 0..3
  动作映射成它们会绕开「四个离散安全动作」的语义，也与
  `ALVLN/docs/08_CF_DRONE_SAFETY.md §2` 的「ESP32 唯一控制仲裁、无 ACK 不判成功」
  冲突。
- 该 fallback **不在本骨架范围内，未实现、未验证**。要做需要单独的安全评审和
  固件/桥接实现，不能直接拿 `MANUAL_CONTROL`/Web RC 当四动作。

## 7. 安全边界

- 不装桨、先台架；串口调试助手通过只代表字节对，不代表飞控能安全执行。
- ESP32 飞控保留最终否决权；模型/本工具只产生有界意图。
- `SAFE_NEUTRAL` 不等于已验证的安全悬停：`ALVLN/docs/08_CF_DRONE_SAFETY.md` 明确
  当前 `ALTHOLD`/开环下降未充分验证；心跳也不是急停。
- 链路失效依赖接收端 TTL；本工具无法证明对端会执行该策略。
- 本工具无 ACK，发送成功不等于飞控接收/执行成功。
- 真机验证门禁见 `ALVLN/docs/08_CF_DRONE_SAFETY.md §4` 与
  `ALVLN/docs/09_SG2002_DEPLOYMENT.md` E6。

## 8. 验证

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb
python3 tools/test_uart_action.py
# PASS: uart_action 帧编码/CRC/心跳/seq/dry-run/坏配置/串口错误 全部通过
```

测试覆盖：CRC-8/CRC-16 标准向量、默认帧逐字节期望值、动作越界、seq 递增/回绕、
心跳与动作帧区别、stdin 动作、dry-run 不碰硬件、坏配置、串口不存在/权限不足
（mock，不打开真实串口）。实机串口与 ESP32 解析**未验证**。

注意：`tools/check.sh` 的 Python 测试循环只扫 `sim/tools/test_*.py`，本测试需按上面
命令单独运行；本次未改 `check.sh`。
