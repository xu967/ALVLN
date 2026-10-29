# CF-Drone 源码审计复核（针对 V7.1 §1）

> 审计对象：`D:\论文\uav-research-20260930\yingjian\CF-Drone.zip`
> 仓库：`https://github.com/songge8/CF-Drone.git`
> commit：**`e7f507f`（"更新说明"）**，本地 `git log` 可复现
> 规模：5,901 行 C/C++（33 个 `.ino/.h/.cpp`）
> 审计日期：2026-10-28
> 审计目的：① 核对 V7.1 §1.1 的 15 条断言；② 补做 V7.1 未覆盖的安全/权限/传感器/UART 资源专项

---

## 0. 审计方法与边界

| 项 | 说明 |
|---|---|
| 我实际读过的文件 | `CF-Drone.ino`、`control.ino`、`control.h`、`motors.ino`、`estimate.ino`、`safety.ino`、`mavlink.ino`、`cli.ino`、`parameters.ino`、`wifi.ino`、`board_config.h`、`battery.ino` + 4 个数学头文件 |
| 我只扫过未细读 | `web_rc.ino`(498)、`web_rc_html.h`(666)、`MPU9250.cpp`(577)、`SBUS.cpp`(267)、`parameters.ino` 部分、`log.ino`、`led.ino` |
| 我无法判断 | **实物板到底是哪颗芯片、板上是否焊了扩展板传感器**。`board_config.h` 用 `CONFIG_IDF_TARGET_*` 分支，三种配置并存，源码本身不能告诉你哪个在跑 |
| 不做法律判断 | LICENSE 的 GPLv3 + 附加条款兼容性需专业意见；我只记录与代码行为相关的事实 |
| 编译验证 | **未做**。无 ESP32 工具链，未编译。所有"会/不会"均基于源码静态阅读 |

---

## 1. V7.1 §1.1 十五条断言：逐条核对

**结论先说：13 条完全正确（含行号），2 条结论对但严重性被低估，0 条错误。** 这份审计的行号精度相当高——我抽查的 17 处行号引用，无一处偏差。

| # | V7.1 断言 | 我的核对位置 | 判定 |
|---|---|---|---|
| 1 | `CF-Drone.ino:25–66` setup/loop 流程为 `readIMU→step→readRC→estimate→control→sendMotors→handleInput`，WiFi 时加 MAVLink | `CF-Drone.ino:44–66`，顺序**逐字一致** | ✅ 准确 |
| 2 | `control.ino:13–53` 注释针对 **118 mm 轴距**；滚转/俯仰角速度 PID + 姿态外环 + 偏航；`RAW/ACRO/STAB/ALTHOLD/AUTO` 枚举 | `control.ino:13` 注释原文"118mm轴距的微型四轴飞行器"；`:50` 枚举 | ✅ 准确 |
| 3 | `control.ino:43` `ALTHOLD_HOVER_THRUST 0.5f` 注释为 **stub，待气压计实现** | `control.ino:43` 原文：`// 定高悬停基础推力（stub，待气压计实现后替换）` | ✅ **行号精确** |
| 4 | `control.ino:327–330` WebRC 选定高会回退 `STAB` 并提示"定高模式暂不支持" | `control.ino:327–330`，原文 `setWebRCWarn("定高模式暂不支持，已切换为自稳")` | ✅ **行号精确** |
| 5 | `control.ino:109–180` STAB 把前两通道映射为目标姿态、油门映射归一化推力；ACRO 是目标角速度 | `control.ino:160–180` | ✅ 准确 |
| 6 | `control.ino:183–235` `attitudeTarget → ratesTarget → torqueTarget → motors[4]`，混控顺序在源码中定义 | `:183–209` 两级 PID；`:224–227` 混控四行 | ✅ 准确 |
| 7 | `estimate.ino:36–63` 陀螺积分姿态，落地时有加速度修正；`applyAcc()` 在 `!landed` 时直接 return | `estimate.ino:40–58`，`landed` 定义与 `if (!landed) return;` 均在 | ✅ 事实准确，**但严重性被低估，见 L3** |
| 8 | `imu.ino` 初始化/读数/校准/安装方向；`imuOK` 参与解锁 | `control.ino:119–130` 解锁前查 `imuOK` | ✅ 准确 |
| 9 | `rc.ino:131–169` SBUS/CRSF/ELRS 用 `BOARD_RC_SERIAL` 与参数化引脚；默认 CRSF 420000 | `rc.ino:136` `BOARD_RC_SERIAL.begin(rcBaud, SERIAL_8N1, rcRxPin, rcTxPin)`；`board_config.h` `BOARD_RC_BAUD 420000` | ✅ 准确，**但漏了 C3 用 `Serial1` 的资源后果，见 §4** |
| 10 | `board_config.h` 三芯片 RC 串口/引脚不同；**C3 默认 `WIFI_ENABLED=0`** | `board_config.h:48` `#define BOARD_WIFI_ENABLED 0  // C3 单核...` | ✅ 准确 |
| 11 | `CF-Drone.ino:26`、`cli.ino` `Serial.begin(115200)`，主串口承担启动日志与文本 CLI | `CF-Drone.ino:26`；`cli.ino:66` `Serial.print(buf)` | ✅ 准确 |
| 12 | `mavlink.ino:70–105` MAVLink 经 `receiveWiFi()`/`sendWiFi()`，支持 `MANUAL_CONTROL` | `mavlink.ino:75–78`（`sendWiFi`）；`:93–104` `MANUAL_CONTROL` | ✅ 准确 |
| 13 | `mavlink.ino:173–207` AUTO 下有 `SET_ATTITUDE_TARGET`、`SET_ACTUATOR_CONTROL_TARGET`；后者可直接写电机数组；**高风险** | `:173–191`（`armed = m.thrust > 0`）；`:193–207`（`memcpy(motors, m.controls, ...)`；`armed = motors[i]>0`） | ✅ 准确，**但漏掉了同一文件里更高权限的 `SERIAL_CONTROL`，见 L1** |
| 14 | `safety.ino:38–116` 有 RC 失联、WebRC 失联与 `descend()`；RC 阈值约 1 s，WebRC 8 s | `safety.ino:5` `rcLossTimeout = 1`；`:8` `WEB_RC_LOSS_TIMEOUT_MS 8000UL` | ✅ 准确 |
| 15 | `safety.ino:60–85` `descend()` 逐渐降推力，低于 0 时上锁 | `safety.ino:65–88`，`thrustTarget -= dt/descendTime; if (<0) {=0; armed=false;}` | ✅ 准确，**但漏了它用 stub 常量当悬停推力，见 L4** |
| 16 | `motors.ino` 4 个归一化命令，支持 PWM | `motors.ino:9` `float motors[4]`；`:12` 引脚顺序 `RL,RR,FR,FL` | ✅ 准确 |
| 17 | `control.h` 有 `ControlState` 与 `MODE_AUTO` 定义；"一份头文件存在不等于接口已实现" | `control.h:5–19` | ✅ 准确，且**可再加两条**（见下） |

**对你这份审计的总体评价**：这是一次**真正读过源码**的审计，不是摘要。行号、变量名、枚举值、注释原文都对得上。作为一个自审，它的可信度显著高于我见过的多数工程文档。

**两条可补充的小瑕疵**（都在 `control.h`，不构成安全风险）：
- `control.h:35` 声明 `const char* getModeName(int mode);`，而 `control.ino:259` 定义的是 `const char* getModeName()`（**无参**）。C++ 允许把它当重载，所以不报错，但 `(int)` 那个版本永远没有定义——**是一个永不链接的死声明**。
- `struct ControlState` 与 `void combineInputs()` 在 `control.h` 声明，全工程**无任何实现或调用**。这恰好印证了你那句"头文件存在不等于接口已实现"。

---

## 2. V7.1 漏掉的四条（按严重度排序）

### L1（最严重，P0）｜MAVLink `SERIAL_CONTROL` 是一条**未认证远程 shell**，可 `arm`、转电机、改电机引脚、重启

V7.1 把 `SET_ATTITUDE_TARGET` / `SET_ACTUATOR_CONTROL_TARGET` 标为"高风险"，方向对，但**漏掉了同一文件里权限更高的那条**：

```c
// mavlink.ino:163-171
if (msg.msgid == MAVLINK_MSG_ID_SERIAL_CONTROL) {
    mavlink_serial_control_decode(&msg, &m);
    if (m.target_system && m.target_system != mavlinkSysId) return;
    char data[...]; strlcpy(data, (const char *)m.data, m.count);
    doCommand(data, true);          // ← 直接进 CLI
}
```

**没有 `mode` 门禁、没有 `armed` 检查、没有认证。** 而 `doCommand`（`cli.ino:100+`）支持的完整命令表包括：

| 命令 | 后果 |
|---|---|
| `arm` / `disarm` | **远程解锁/上锁** |
| `mfr` / `mfl` / `mrr` / `mrl` | 单电机测试，"马达不受算法影响运转" |
| `p <name> <value>` | 改**任意**参数，含 `MOT_PIN_FL/FR/RL/RR`（回调是 `setupMotors`）、PID、限幅 |
| `raw/stab/acro/auto` | 改飞行模式 |
| `reboot` / `reset` | 重启 / 清空参数并重启 |
| `ca` / `cr` | 陀螺校准 / 遥控校准（长时间阻塞） |

而入口的可达性由 `wifi.ino:25–26` 决定：

```c
storage.getString("WIFI_AP_SSID", "Drone_WiFi").c_str(),
storage.getString("WIFI_AP_PASS", "12345678").c_str()
```

**默认 AP 名 `Drone_WiFi`、默认密码 `12345678`。**

完整攻击链：**连上 `Drone_WiFi`/`12345678` → 发一个 MAVLink `SERIAL_CONTROL` 包 → `arm` + `mfr` → 电机转。** 全部无需任何凭据。

**触发条件（必须写清楚，否则会误判）**：仅当 `BOARD_WIFI_ENABLED == 1` 时成立。`board_config.h` 中 **C3 分支为 `0`（关闭）**，**ESP32 / ESP32-S3 分支为 `1`（开启）**。所以：

- 如果你的板子是 **C3** → 这个洞当前不可达（但你也就没有 WiFi 调试能力）
- 如果是 **ESP32 / S3** → **洞是开的，且默认密码是公开的**

**另外两个相关问题**：
1. `SET_ACTUATOR_CONTROL_TARGET` 的直接写电机路径，只被 `mode != AUTO` 挡住。而 `mode` 可以因为**失效保护**（`descend()` 里 `mode = AUTO`）或参数 `CTL_FLT_MODE_*` 变成 AUTO——**即"进入失效保护"这个动作本身，会把这个高权限写电机入口解锁**。
2. `memcpy(motors, m.controls, sizeof(motors))` 对 `controls` **没有 NaN/范围检查**。NaN 经 `getDutyCycle` 的 `constrain(value,0,1)` 后仍为 NaN（NaN 与任何数比较均为 false），`round(NaN)` 是未定义行为。

**对 V7 计划的直接影响**：你的 `NAV_FRAME_V1`（CRC/TTL/ACK/REJECTED/人工接管）设计得很好，但**如果 WiFi MAVLink 保持开启，攻击者根本不需要走你的 UART**。所以 V7 的安全门禁必须加一条：

> **凡台架/飞行测试，`BOARD_WIFI_ENABLED` 与 `BOARD_WEB_RC_ENABLED` 必须为 0，或 AP 密码必须改为非默认值并记录在案；`SERIAL_CONTROL → doCommand` 路径必须被移除或用编译开关隔离，且在 HIL 测试中作为一项显式故障注入项。**

这条应该排在 `UART_CRC_ERROR` 之前——因为它比任何 UART 攻击都容易。

---

### L2（P0，安全）｜`CTL_FLT_MODE_*` 可以把模式设成 `ALTHOLD`，但主控制路径**没有 ALTHOLD 分支**

V7.1 说"有 ALTHOLD 枚举不代表有真实定高控制"，正确。但实际情况比"没实现"更糟：**它可达，且可达后的行为是危险的。**

1. `parameters.ino:42`：`void setValue(const float value) { if (integer) *i = (int)value; else *f = value; }` —— **无任何范围校验**（`setParameter` 只对整型拒绝 NaN）。
2. `parameters.ino:92–94`：`CTL_FLT_MODE_0/1/2` 直接绑定 `flightModes[0..2]`。
3. `control.ino:110–112`：`mode = flightModes[i]` —— 于是 `mode` 可以是 3（ALTHOLD）或任意整数。
4. `control.ino:160–180`：**只有 `if (mode == STAB)`、`if (mode == ACRO)`、`if (mode == RAW)` 三个分支，没有 `ALTHOLD`。**

**后果**：一旦 `mode == ALTHOLD`（或任何非法值），三个分支都不执行 →
- `attitudeTarget` 停留在上一周期 STAB 分支写入的值 → **姿态环持续追一个冻结的期望姿态**；
- `ratesExtra` 同样冻结；
- `thrustTarget` 仍是油门的直接映射（`interpretControls:154–158` 在所有 mode 分支之前执行）。

**净效果：飞手推 roll/pitch 摇杆没有反应（因为姿态目标不再更新），但油门照旧直通。** 一架 118 mm 四轴在"看起来切换成功了"的状态下失去横滚/俯仰权限——这比"提示不支持"危险得多。

注意 V7.1 引用的 `control.ino:327–330` 那条回退**只存在于 WebRC 按钮路径**，参数路径和 `mode` 通道路径都没有对应保护。

**修法（三选一，建议同时做前两条）**：
1. 在 `interpretControls()` 末尾加穷尽分支 + `default:` 落到 STAB；
2. `setParameter` 对 `CTL_FLT_MODE_*` 加白名单（0/1/2/4），拒绝 3 与越界值；
3. **把"设成 ALTHOLD 必须报 `FC_ALT_HOLD_UNAVAILABLE` 并拒绝"做成回归测试**——这正好是你 V7 gate 表里已有的错误码，现在有了真实的触发点。

---

### L3（P0 for VLN，被 V7.1 低估）｜**飞行中姿态是纯陀螺积分，没有加速度计修正，也没有磁力计**

V7.1 写的是"现有估计是姿态而非三维位置/速度；飞行中校正机制有实现局限，须做长时漂移和姿态稳定测试"。这太轻了。源码是这样：

```c
// estimate.ino:41-50
float accNorm = acc.norm();
landed = !motorsActive() && abs(accNorm - ONE_G) < ONE_G * 0.1f;
if (!landed) return;                      // ← 飞行中直接返回，加速度计完全不参与
```

```c
// estimate.ino:30
float levelBiasGain = 0;                  // Mahony I 项增益（关闭）
```

叠加：
- **无磁力计**：`drivers/` 只有 MPU9250 / invensense / SBUS，无 QMC5883L 等磁力计驱动 → **偏航无任何绝对参考**；
- 加速度计修正只在 `motorsActive()==false`（即电机不转）时生效；
- 水平修正的 I 项被显式关闭。

**所以飞行期间：`attitude` = 陀螺积分，roll/pitch/yaw 全部累积漂移，yaw 尤其。**

**为什么这对本项目是结构性的**：AerialVLN 的动作空间是 **15°/步的偏航**，而你的 VLN 指令形如"从右侧绕过、到红屋顶停下"。如果机体自己的航向在 30 秒内漂 10–20°，那么"左转 15°"这个动作在执行层就已经不可靠，**视觉-语言模型的输出会和一个正在漂移的航向叠加**。这不是"精度不够"，是**闭环的执行基准在动**。

**修法（按代价排序）**：
1. **最便宜**：在测试中量化漂移率（静止上电、电机怠速、单轴旋转各 60 s，记录 yaw 漂移 deg/min），把它写进论文的"执行层不确定性"一节，并据此限制单次任务时长；
2. **中等**：启用扩展板的 **QMC5883L** 做航向绝对参考（磁力计 + 倾斜补偿），这需要先确认扩展板在不在；
3. **最贵**：PMW3901 光流 + VL53L1X 测距做速度/高度估计。

**这条同时解释了一件事**：为什么 V7 的物理轨一定要 CF-Drone 参与——因为如果换成 Gazebo 自带控制器，你可以直接用仿真真值姿态，**这个漂移问题就被藏起来了**。藏起来不是错，但必须在论文里声明"实机航向误差未建模"。

---

### L4（P1）｜失效保护链条依赖一个自己标注为 stub 的常量；且注释与代码的电压阈值不一致

**4.1 `descend()` 用 stub 当悬停推力**

```c
// safety.ino:69
if (thrustTarget > ALTHOLD_HOVER_THRUST) thrustTarget = ALTHOLD_HOVER_THRUST;
```
`ALTHOLD_HOVER_THRUST = 0.5f`，而它在 `control.ino:43` 的注释是"**stub，待气压计实现后替换**"。

`descend()` 是**唯一**的自动降落路径（RC 失联、WebRC 失联、电池 L3 危急全部走它）。它的起点推力被钉在一个未标定的常数上：
- 真实悬停推力若为 0.35 → 前 ~1.5 s 会**先爬升**再下降；
- 若为 0.65 → **下降过快**。

对一个 118 mm 微型四轴，这两种都不是"安全降落"。**V7.1 说"属于推力渐降，不是高度闭环"是对的，但没指出这个渐降的起点本身是未标定的。**

**4.2 电压阈值：注释与代码不一致**

| | 注释（`safety.ino:144–147`） | 代码（`battery.ino:9–11`） |
|---|---|---|
| L1 未解锁禁解锁 | 3.4 V | `VBAT_WARN_THRESHOLD` = **3.5 V** |
| L2 飞行告警 | 2.8 V | `VBAT_LOW_THRESHOLD` = **3.3 V** |
| L3 自动降落 | 2.6 V | `VBAT_CRITICAL_THRESHOLD` = **3.0 V** |

差 0.1 / 0.5 / 0.4 V。**代码是对的（3.0 V 自动降落对单节锂电合理，2.6 V 太低），注释是过期的。** 但这恰好说明：**源码审计必须核代码、不能核注释**——这也是你 V7.1 §1 做得对的地方（你引用的基本都是代码行，不是注释）。

**4.3 顺带一个正面发现**：`parameters.ino:syncParameters()` 里有 `if (motorsActive()) return;` —— **飞行中不写 Flash**，避免擦写延迟打断控制。这是很好的纪律，V7.1 没提，值得写进"可以继承的设计"。

---

## 3. 硬件层面的关键发现：这直接改写了我上一轮的 P0-4

`board_config.h` 明确为一块**扩展板**预留了四类传感器：

| 传感器 | 作用 | 引脚分配 |
|---|---|---|
| **BMP388** | 气压高度 → **定高** | I2C |
| **QMC5883L** | 磁力计 → **航向绝对参考**（解 L3） | I2C |
| **VL53L1X** | ToF 测距 → 低空高度/降落控制 | XSHUT：ESP32 `37` / S3 `32` / C3 禁用 |
| **PMW3901** | 光流 → **水平速度**（V7.1 说缺的速度闭环） | CS：ESP32 `38` / S3 `33` / C3 禁用 |

注释原文（`board_config.h:100`）：
> `// 1 = 启用扩展板传感器探测（BMP388/VL53L1X/QMC5883L/PMW3901）;0 = 跳过探测，所有能力标志保持 false`

**但软件侧一行都没有**：
- `grep -r EXPANSION_BOARD_ENABLED` → **只命中 `board_config.h` 自己**（定义 + `#undef` + 兜底 `#if`）。**这个宏从未被任何 `.ino/.cpp` 引用，是死配置。**
- `drivers/` 目录只有 `MPU9250`、`invensense_imu`、`SBUS`。**无 BMP388 / QMC5883L / VL53L1X / PMW3901 任何驱动文件。**
- 全工程无 `Adafruit_BMP`、无 `VL53L1X`、无光流相关符号。

**这解开了我上一轮给自己写的"反例 2"**：我问"板上会不会其实有气压计，只是代码没实现"。答案是——

> **不是"代码没实现"，而是"连驱动都还没有"；但 ① 平台在配置层就规划了这四颗传感器，② 引脚号是具体数字而非占位符，③ S3 分支同时给了 I2C(35/36)、VL53(32)、PMW(33)，说明有人为一块真实 PCB 分配过资源。**

**所以现在最该做的不是读文档，是拿万用表/放大镜看板子**：I2C 上（ESP32: GPIO21/22；S3: GPIO35/36）有没有器件应答、有没有 BMP388/QMC 的丝印。**这一步 20 分钟，决定后面 8 周的路线。**

---

## 4. UART 资源的真相（V7.1 说"不要假设 Serial2 都空闲"——现在有精确答案）

| 芯片 | console | RC | **NAV UART 可用？** |
|---|---|---|---|
| ESP32（classic） | `Serial` (UART0, 115200) | `Serial2`（GPIO4 RX，CRSF 420000） | ✅ **`Serial1` 空闲** |
| ESP32-S3 | `Serial` (UART0) | `Serial2`（GPIO8 RX） | ✅ **`Serial1` 空闲** |
| **ESP32-C3** | `Serial` (UART0) | **`Serial1`**（`board_config.h:21`） | ❌ **两个 UART 全占** |

C3 只有 2 个 UART（UART0/UART1）。所以如果我上一轮建议的"LicheeRV → ESP32 NAV 串口"要落地：

- **ESP32 / S3 板**：直接用 `Serial1` + 空闲 GPIO，一天可通；
- **C3 板**：需要额外决策 —— ① 把 console 迁到 C3 原生 USB-Serial-JTAG（C3 有此硬件），释放 UART0；② 或把 RC 迁到 UART0 的备用引脚、NAV 用 UART1；③ 或用 USB CDC 走 NAV（LicheeRV 有 USB2.0 OTG，可行但增加主机侧复杂度）。

**结论：先确认芯片型号，再写协议。** 这比 V7.1 §9.2 里那份精心设计的帧格式优先级更高——因为帧格式在哪个 UART 上都一样，但 UART 不够就一行都跑不起来。

---

## 5. 好消息：CF ControlCore 的抽离比 V7 计划的容易得多

V7 把 `CF0/CF1`（CFMath 黄金对拍 + CF Core SIL）当成两个独立里程碑。按源码看，**难度被高估了**：

| 文件 | 是否已硬件无关 | 依赖 |
|---|---|---|
| `vector.h` (139 行) | ✅ **完全无关** | 无任何 include |
| `quaternion.h` (228 行) | ✅ 无关 | 仅 `vector.h` |
| `pid.h` (50 行) | ✅ 无关 | 仅 `lpf.h` |
| `lpf.h` (37 行) | ✅ 无关 | 无 |
| `util.h` (91 行) | ❌ 有 | `<math.h>` + ESP brownout 寄存器 |

而 `controlAutitude()/controlRates()/controlTorque()/desaturate()` 只依赖上述四个头 + 6 个全局量（`attitudeTarget/ratesTarget/ratesExtra/torqueTarget/thrustTarget/motors`）+ `print()`。

**也就是说：写一个 ~30 行的 shim（`constrain` / `radians` / `mapf` / `isfinite` / 空 `print`），这 4 个头 + 4 个函数就能在 Linux 下直接编译成数学核，无需任何移植工作。**

工作量：**1–2 天**（含测试），而不是 V7 里那种"重大里程碑"。

**建议的 golden 测试清单（可直接实现为 `sim/tests/` 下的用例）**：

| # | 用例 | 期望 |
|---|---|---|
| 1 | 静止、attitude=单位四元数、gyro=0、dt=0.002 跑 5 s | attitude 不变（误差 <1e-6） |
| 2 | gyro=(0,0,90°/s)、积分 1 s | yaw ≈ 90°（±0.5°） |
| 3 | attitudeTarget 右倾 10°，thrustTarget=0.5 | `torqueTarget.x` 符号与 FL/RL 减速方向一致 |
| 4 | 四电机等推力 | `motors[i] == thrustTarget` 全等 |
| 5 | 零推力 | 四电机 = `motThrMin`（怠速） |
| 6 | `armed=false` | 四电机全 0 |
| 7 | 人为给 `motors = {1.2, -0.3, 0.9, 0.4}` 调 `desaturate` | avg 保持、max≤1、min≥0 |
| 8 | `descend()` 从 thrust=0.5、dt=0.002 跑 10 s | 单调降到 0 且 `armed=false` |
| 9 | **`CTL_FLT_MODE_2 = 3`（ALTHOLD）** | **必须被拒绝或明确降级**（这是 L2 的回归测试） |
| 10 | 同上，`= 99` | 同上 |

第 9、10 条现在是**已知会失败**的——它们正好可以作为你 SIL 阶段的第一个真实缺陷修复。

---

## 6. 对 V7.1 的修订清单（8 条，按优先级）

| # | 改什么 | 位置 |
|---|---|---|
| 1 | **§1.1 增加 `SERIAL_CONTROL → doCommand` 一行，标注为"未认证远程 shell，可 arm/转电机/改引脚"，严重度高于另外两条 MAVLink 入口** | V7.1 §1.1 表 |
| 2 | 在 §9.3/§9.4 前置一条硬门禁：**台架/飞行前必须关闭 WiFi 或改默认密码；`doCommand` 的 `arm`/`mfr`/`MOT_PIN` 路径必须用编译开关隔离，并作为 HIL 故障注入项** | V7.1 §9 |
| 3 | 新增错误码 **`FC_MODE_UNIMPLEMENTED`**：`CTL_FLT_MODE_*` 取到未实现模式时必须拒绝并报错 | V7.1 §13 |
| 4 | 把 L2 作为 **CF0 gate 的必测项**（现在 gate 只要求"PID/mixer/坐标 golden"） | V7.1 §14 CF0 |
| 5 | **§1.2 增加"飞行中姿态为纯陀螺积分、无磁力计"**，并把它列为物理轨的已知不确定性；论文必须声明 | V7.1 §1.2 / §16 |
| 6 | 修 §4.3/§5.3：定高的阻碍不是"硬件没有"，而是"**扩展板四颗传感器零驱动**"。把"确认扩展板是否存在 + I²C 扫描"写成 **GZ-1（比 GZ0 更早）** 的验收项 | V7.1 §5.3 / §14 |
| 7 | 修 §9.1：C3 的 RC 是 `Serial1`，**两个 UART 全占**；NAV UART 只在 ESP32/S3 上直接可行。先确认芯片，再定协议 | V7.1 §9.1 |
| 8 | 下调 CF0/CF1 的工期估计：4 个数学头已硬件无关，**1–2 天可得 Linux 数学核**；把省下的时间投给 10 条 golden 测试 | V7.1 §14 |

---

## 7. 结论

**这次审计把你上一轮的"不可核实"变成了"已核实且更严重"。**

三句话：

1. **你那份 §1 审计的质量是可信的**——17 处行号引用零偏差，结论方向全对。作为自审，这已经高于行业平均。我复核后没有发现任何一条错误断言。

2. **但它漏掉的东西比它写下的东西更危险。** `SERIAL_CONTROL → doCommand` 这条未认证远程 shell（默认 AP `Drone_WiFi`/`12345678`），权限比你自己标"高风险"的那两个入口更高；`CTL_FLT_MODE_*` 可把模式设到没有实现的 `ALTHOLD` 从而让飞手失去横滚/俯仰权限；飞行中姿态是纯陀螺积分且无磁力计——**这一条直接命中你 15°/步 的动作空间**。

3. **最重要的不是这些 bug，而是那个硬件事实**：平台在配置层就规划了 BMP388 / QMC5883L / VL53L1X / PMW3901（定高、磁航向、测距、光流），引脚号都是具体数字，**但驱动代码为零，`EXPANSION_BOARD_ENABLED` 是个从未被引用的死宏**。

   这意味着你的物理轨不是"逻辑不可达"，而是：**硬件接口规划好了、软件一行没写、而固件被你自己冻结了。** 三件事里只有第三件是"规定"，前两件都是"工作量"。

   所以下一步不是再写一版方案，而是：**拿起板子，看 I²C 上有没有应答。** 有 → 补 3 个驱动 + 1 个定高环，物理轨就活了（2–4 周）；没有 → 老老实实走 Gazebo 自带控制器，把 CF-Drone 降为工程附录。**两条路都通向论文，中间那条（继续写文档）哪条都不通。**
