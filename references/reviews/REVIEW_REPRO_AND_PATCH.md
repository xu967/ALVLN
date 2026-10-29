# TinyAerialVLN V7.1 独立评审报告（第二版 · 源码实测）

> **评审日期**：2026-10-28
> **评审对象**：`V7_1_RESEARCH_PLAN.md`（973 行）、`CODE_RULES.md`（202 行）、`docs/V7_1_INDEPENDENT_AI_REVIEW.md`，以及工程 ZIP 内全部 49 个文件。
> **评审环境**：Ubuntu 24.04 容器，Python 3.12.3，g++ 13.3。**没有** gz、nvcc、GPU、SG2002、ESP32，所以本报告里所有硬件相关结论都只能是"未验证"。
> **配套文件**：
> - `review_v2_repro.py`：复现脚本，只用标准库。在原版 V7.1 上 21 项全部复现。
> - `review_v2.patch`：最小补丁。改 8 个文件、新增 392 行有效代码，符合 CODE_RULES 的单次 PR 上限。打补丁后，21 项中有 18 项消失。
> - 两者的全文分别附在附录 C 和附录 B，本文件可以单独阅读。

**证据等级说明**

| 标记 | 含义 |
|---|---|
| 【实测】 | 本次在容器里实际执行并复现 |
| 【源码】 | 可以直接在代码或文档中定位 |
| 【文献】 | 来自公开资料，链接见 §11 |
| 【推测】 | 需要你用实验或源码来证伪 |

---

## 0. 阅读覆盖与对第一版的更正

第一版报告只读到了评审包，读不到方案、代码规则和 ZIP。这一版全部读完，并实际运行了测试和故障注入。第一版有几处判断需要修正：

| 第一版判断 | 第二版结论 | 依据 |
|---|---|---|
| CF-Drone 可能是 ESP-Drone / Crazyflie 系，建议用 CrazySim | **撤回**。CF-Drone 是用 Arduino 写的 ESP32 飞控，高度疑似 Flix 派生。对应的现成 SIL 是 Flix 自带的 Gazebo 仿真器，见 P1-1 | 方案 §1.1；Flix 公开仓库 |
| 看板可能存在 XSS | **不成立**。`dashboard.html` 没有 innerHTML，动态内容全部用 createElement + textContent 写入，并且设置了 CSP | 【源码】 |
| 离散 5 m / 15° 动作直接下发飞控，列为 P0 | **降为 P1**。方案 §5.4 已要求双 head 或 adapter，§13 有 `ACTION_PROTOCOL_MISMATCH` 错误码。剩余缺口是 NavIntent 的单位没有定义（P1-7） | 【源码】 |
| MiniWorld / BabyAI 撑不起空中结论，列为 P0 | **降为 P2**。方案 §6.1 已限定 D0/D1 只做功能验证 | 【源码】 |
| 文件数上限会逼出巨型单文件 | **目前不成立**。`code/src`、`code/cuda`、`sim/src` 都是 0 个源文件。真正的问题是 24 个文件名已被预先分配完（P1-3） | 【源码】 |
| Host 校验缺失、嵌套 NaN、`ready` 误导 | **全部坐实，而且比预想的严重**（B01–B07） | 【实测】 |
| 物理飞行安全列为 P0 | **降为 P1**。方案的流程门禁写得好（禁止自由飞行、无桨 HIL）。新增供电、载荷、退化方式三类风险（P1-5） | 方案 §9.3–9.4 |

---

## 1. 结论摘要

方案的工程纪律是认真的。双轨评测、学生在线输入合同、分层门禁、禁止自由飞行，都比一般课题计划严谨。问题集中在三个层面：

1. **研究层**：主方法 ECPT 只有名字，没有定义（P0-1）。新颖性空间已被 L2Stop、Self-Monitoring、X-Fly 挤压（P1-6）。
2. **数据层**：现有场景生成器让任务退化成"直飞约 16 米然后停下"，语言和视觉都可能是多余的（P0-2）。
3. **工程层**：
   - 已实现的 Python/C++ 工具有 16 个可复现缺陷（B01–B16）。
   - 日志契约在方案与代码之间不一致（P1-2）。
   - 研究链路本身还是 0 行代码（P0-3）。

另外有一件需要尽快确认的事：CF-Drone 高度疑似由开源项目 Flix 派生（P1-1）。如果属实，CF0/CF1 可以复用上游的仿真思路，少写很多代码；同时许可证来源要先理清。

| 编号 | 级别 | 一句话 |
|---|---|---|
| P0-1 | P0 | ECPT 未定义，H-progress 不可证伪 |
| P0-2 | P0 | 场景生成器：直线永远畅通、终点固定、无地标、目标对学生相机可见 |
| P0-3 | P0 | 训练器、学生、仿真闭环均为 0 行代码；看板状态是硬编码的 |
| P1-1 | P1 | CF-Drone ≈ Flix：上游有运行原始 Arduino 代码的 Gazebo 仿真；另有 1 kHz 主频与许可来源问题 |
| P1-2 | P1 | JSONL 契约漂移：方案自己的示例被看板 100% 拒收；SR 缺少 split、n、是否闭环 |
| B01–B07 | P1 | 看板：一行坏日志拖垮整个 run、DNS rebinding、损坏清单仍显示 ready、静默截断 |
| P1-3 | P1 | 24 个文件预算已被预先分配完，漏掉了 YAML、图像解码、ONNX、相机采集 |
| P1-4 | P1 | 统计功效：SR 从 30% 到 40% 每组约需 356 条 episode，还要乘以场景聚类的设计效应 |
| P1-5 | P1 | 载荷、供电掉压、开环下降的风险没有测量计划 |
| P1-6 | P1 | 独立 STOP 与进度估计都已有先例 |
| P1-7 | P1 | 协议、数据契约文档一份都没有，UI 文档却有三份 |
| B08–B16 | P2 | 启动器、头文件、预检、测试的契约与健壮性问题 |

---

## 2. 已复现的代码缺陷（B01–B16）

**定级**：B01–B07 合计定为 P1，理由是可用性下降、状态误导和本机信息外泄。B08–B16 定为 P2。

**复现方式**：复现脚本对每一条给出"存在 / 已修复"。原版的完整输出见附录 A。

| 编号 | 位置 | 现象 | 证据 | 打补丁后 |
|---|---|---|---|---|
| B01 | `tools/dashboard.py:69, 180, 191` | 只检查顶层的 NaN。日志里混入一行 `{"event":"train_step","extra":{"grad_norm":NaN}}` 后，序列化抛出的 ValueError 被当成 404 处理，**整个 run 的指标全部消失**；前端每 4 秒刷新一次，每次都显示"接口连接异常" | 【实测】 | 跳过这一行并给出告警，其余事件照常显示 |
| B02 | `dashboard.py:60–62` | 一行 3 万层的嵌套数组会触发 RecursionError，没有被捕获，连接直接断开、无任何响应；一行 5000 位的整数超过 Python 的整数位数上限，返回 404 | 【实测】 | 跳过这一行 |
| B03 | `dashboard.py:88–95` | 只要有一个根节点为 `[]` 的场景清单，就会触发 AttributeError，`/api/scenes`、`/api/status`、`/api/audit` 三个接口全部断连，前端整页报错 | 【实测】 | 标记为 `manifest_invalid` |
| B04 | `dashboard.py:193–194` | 工程目录名带"工程"二字。OSError 的消息里含中文路径，`send_error` 把它写进按 latin-1 编码的 HTTP 状态行时失败，连接无响应。UI 设计文档 §4 写的"I/O 问题返回 500"，在默认解压路径下根本不会出现 | 【实测】 | 状态行只写 ASCII 错误码，正常返回 500 |
| B05 | `dashboard.py:159` 起 | 不校验 Host 头。伪造 `Host: evil.example` 仍返回 200 和完整的 audit JSON，恶意网页可以通过 DNS rebinding 在你的浏览器里把它读走 | 【实测】 | 只接受白名单内的 Host，其余返回 421 |
| B06 | `dashboard.py:110` | 清单全部损坏时，scenes 状态仍是 `ready`，文案显示"已发现 2 份"。这正是 CODE_RULES V7.1 第 4 条禁止的"文件存在就算就绪" | 【实测】 | 只统计有效清单，并核对 sha256 |
| B07 | `dashboard.py:36–42` | 最多只列 80 个 run（按名字倒序），且没有任何提示。有 89 个 run 时 run_count 显示 80，名字排在后面的基线 run 返回 RUN_NOT_FOUND。做一组 8 个消融 × 5 个 seed 的实验就会碰到 | 【实测】 | 报告真实总数；未列出的 run 仍可读取；按修改时间倒序 |
| B08 | `launcher.py:46` 对比 `dashboard.py:23` | 启动器用 `str.isalnum()` 校验，接受 `实验一`、全角 `ｒｕｎ１` 这类 run ID；看板的正则只收 ASCII，这些 run 永远显示不出来 | 【实测】 | 两处共用同一条正则，并且先校验输入 |
| B09 | `launcher.py:49–56` | 按方案 §11 先 `train` 再 `eval` 同一个 run-id，eval 报 RUN_EXISTS（退出码 4）；eval 也没有指定 checkpoint 的参数 | 【实测】 | eval 写到 `runs/<id>/eval_<时间戳>/`，并传入 `--run-dir` |
| B10 | `launcher.py:83` | 方案 §11 列出的 `board doctor`、`uart dry-run` 得到的是 argparse 报错（退出码 2），没有打印 FEATURE_NOT_IMPLEMENTED，违反 CODE_RULES V7 第 9 条 | 【实测】 | 返回退出码 3 并打印 FEATURE_NOT_IMPLEMENTED |
| B11 | `launcher.py:16–33` | doctor 只凭 `which` 就标 `[OK]`：一个直接 exit 1 的假 nvcc 也显示 `[OK] nvcc`。这与方案 §11.2"禁止根据文件名存在就标绿"矛盾，也没有检查 RTX 5090 需要的 CUDA ≥ 12.8 | 【实测】 | 改标为 `[FOUND]`，并实际运行版本探针 |
| B12 | `training_monitor.hpp:48–52` | 调用 `Step()` 之后，`std::cerr` 永久停在 fixed + precision(0) 状态，之后打印 `lr=2e-4` 只会显示 `0` | 【实测】（已编译运行） | 改用私有的 ostringstream 格式化 |
| B13 | `training_monitor.hpp:28–44` | 日志流没有固定为 C locale。训练器如果设置了带千分位或逗号小数点的全局 locale（例如为了输出中文调用 `std::locale::global(std::locale(""))`），就会写出 `"step":26.000,"loss":0,75`，整行变成非法 JSON，被看板全部拒收 | 【实测】（已编译运行） | `imbue(std::locale::classic())`，并用 17 位有效数字 |
| B14 | `training_monitor.hpp:35–40` | 遇到 NaN 直接抛异常，不写错误事件，看板上看不到训练为什么停了（方案 §12.1 要求"保存错误上下文"）。ETA 从对象构造时开始计时，断点续训后会严重低估 | 【实测】/【源码】 | 先写一条 `error` 事件再抛出；构造函数接收 `resume_step` |
| B15 | `sim_preflight.py:48` | 用 `'multicopter' in filename` 做子串判断。一个只装了 `MulticopterVelocityControl` 插件的 world 也能通过"电机模型"检查 | 【实测】 | 解析 XML，要求至少 4 个 MulticopterMotorModel |
| B16 | `.gitignore`；`code/tests/test_infrastructure.py:38–42`；`sim/tests/test_v7_contract.py:9` | ① `artifacts/` 被 git 忽略，全新 clone 后 code/tests 出 1 个 error、sim/tests 出 1 个 fail，所以"13+2 PASS"只在 ZIP 状态下成立；② 训练器一旦编译出来，`test_refuses_nonexistent_training_binary` 会真的启动训练、创建 `runs/forbidden_test` 然后失败；③ `.gitignore` 没有忽略 `build/`，编译产物会被提交 | 【实测】 | 测试自建临时目录；训练器存在时跳过该测试；忽略 `build/` |

---

## 3. P0 阻断问题

### P0-1 主方法 ECPT 没有定义，研究假设不可证伪

**问题**

ECPT 在方案里只以名字出现：

- 方案：§14 的 METHOD 门禁、§15.1 基线表里的"提出的证据校准记忆"、§16.2、§17 第 9 条；
- CODE_RULES：第 61、135、137、139 行，还为它预留了 `model/ecpt.cc`。

没有任何地方回答下面五个问题：

- (a) "证据"是什么：哪些特征，如何与当前子指令对齐；
- (b) "校准"用什么方法、看什么指标：温度缩放还是 conformal，ECE 还是覆盖率；
- (c) 进度标签从哪里来：损失函数里的 `weak_event_label` 没有定义，AerialVLN 这一侧有没有子指令对齐标注也没有说明；
- (d) STOP 如何由进度和证据导出；
- (e) 参数量、FLOPs 预算，以及能否部署到板端。

结果是 §16.1 的 H-progress 无法证伪，"等容量配对消融"也没有可比的对象。

**证据**

【源码】`grep -n ECPT V7_1_RESEARCH_PLAN.md CODE_RULES.md`

**争议 / 反例**

如果定义写在没有打包的文档里（例如 V6 方案），本条降为"文档缺失"级别的 P2。但评审包没有包含它，外部评审者同样看不到。

**最小改动**

写一页《ECPT 定义》，建议放在 `docs/ECPT_SPEC.md`，占用"docs ≤5"的一个名额（可以合并掉一份 UI 文档）。内容包括：

- 公式与标签来源；
- 推理时允许使用的输入；
- 参数量与 FLOPs；
- 与 Self-Monitoring、L2Stop、X-Fly 的逐项差异表。

经 ADR 冻结之后，再开始写 `ecpt.cc`。

**放弃条件**

定义写完后，如果它在结构上等价于"Self-Monitoring 进度头 + 独立 STOP head + 温度缩放"，就换研究问题，不要靠追加 ROI 或 VLM 来包装。

**改进后实验**

| 项目 | 设定 |
|---|---|
| 对比组 | TinyVLN-GRU+Progress（Self-Monitoring 式）、TinyVLN-STOP、TinyVLN-ECPT |
| 容量控制 | 参数量差 < 1%，FLOPs 相同，数据相同 |
| 重复 | ≥ 5 个 seed |
| 统计 | val_unseen 上按场景聚类 bootstrap |
| 主指标 | SR、NE、OSR−SR（漏停）、过早停止率、ECE |
| 调参纪律 | 阈值只在 val_seen 上选，val_unseen 只跑一次 |

### P0-2 场景生成器让任务退化，语言和视觉都可能是多余的

**问题**

`scene_tool.py:39–43` 会把所有障碍排除在中央通道之外。我统计了 4 个 preset × 2000 个 seed、每个 seed 采样 200 个障碍的情况，起点 (-8,0) 到终点 (8,0) 的直线与任一障碍表面的最小距离如下：

| preset | 最小净空 |
|---|---|
| forest / mixed | 1.15 m |
| corridor | 1.48 m |
| columns | 2.25 m |

此外还有五个问题：

- 第 39 行注释说留出 "1.4 m 通道"，但代码实际留出的净空至少 2.3 m 宽。
- 起点和终点对所有 seed 固定不变，清单里永远是 (-8,0,1) 和 (8,0,1)。
- `mixed` preset 走的是 `else` 分支，输出与 `forest` 字节级完全相同。
- 所有障碍都是同一种材质颜色（`.42 .36 .27`）。方案 §6.2 的示例指令 "Pass the blue pillar, turn right at the yellow gate…" 里的地标，根本生成不出来。
- 终点红球是一个普通 visual（z=0.15，半径 0.15），机载相机能直接看到，违反方案 §4.3"目标标记必须与学生所见相机分离"；而清单里写的终点高度是 z=1。

后果是：一个只会"直飞约 16 米然后停下"的盲策略，就可能拿到接近满分。在这样的场景上，任何 SR 都无法说明模型用到了语言或视觉。

**证据**

【实测】复现脚本中的 P0-2、P0-2b 两项。

**争议 / 反例**

如果这些场景只打算作为 S0 难度（方案 §6.4 的"空地 + 单个颜色标记"），它本身没有错。问题是现在 4 个 preset 都只能生成 S0，S1–S3 需要的转弯、相似地标、遮挡都没有生成器支持。

**最小改动**

1. 在可通行区域内随机化终点，并让至少一半场景的直线被挡住，必须绕行。
2. 加入颜色和形状可区分的地标与干扰物；任务按"地标 → 指令 → 参考路径"的方向反向约束生成（方案 §6.2 已有这条要求）。
3. 用 visibility flags 和 mask 让目标标记对学生相机不可见。gz-sim 8 提供了对应组件，需要在 Harmonic 上实测是否生效。
4. 修掉 `mixed` 等同于 `forest`、终点高度不一致这两个问题。
5. 补丁里加了 `test_P0_2_scenes_require_more_than_flying_straight`，目前标为 expectedFailure。生成器修好后删掉这个装饰器，它就成为一道回归门禁。

**放弃条件**

如果在新场景集上，盲基线（RGB 置零 + 空指令）的 SR 仍高于 20%（建议阈值），这个场景集作废，重做任务生成。

**改进后实验**

每个场景集发布前跑四个探针：盲策略、只看视觉（空指令）、只看语言（RGB 置零）、打乱指令。按场景聚类报告 SR 及其 95% CI。

### P0-3 研究链路是 0 行代码，看板也反映不了将来的进展

**问题**

- `code/src/*`、`code/cuda`、`sim/src` 里都只有 `.gitkeep`。
- `CMakeLists.txt.planned` 和两份 YAML 都是一行占位。
- 唯一的 C++ 是 76 行的 `training_monitor.hpp`。
- 15 个测试全部针对 Python 工具。

看板对 physics、cf_sil、edge、firmware、dashboard 这 5 项是硬编码状态（`dashboard.py:109–128`），与文件头部 docstring（第 4–5 行）"状态均由本地文件推导"的说法矛盾。也就是说，即使 GZ0 已经通过，界面仍会显示"未实现"。

**证据**

【源码】`find code/src code/cuda sim/src -type f`

**争议 / 反例**

如果训练器原型在别的仓库里已经存在，请把它连同 gradcheck 结果一起放进下一版评审包。

**最小改动**

1. 门禁状态改为读取证据文件，例如 `artifacts/gates/GZ0.json`，里面记录 trace 文件路径和 sha256。看板只核对文件存在和哈希一致，不负责判断是否通过。补丁先做了第一步：每个状态都带上 `derived_from: hardcoded / file_probe` 标注。
2. 第一阶段只交付两样东西：GZ0 的 trace，以及最小训练器的四项数值门禁——FP64 gradcheck、独立 oracle、CPU/CUDA parity、续训一致性。第一版报告 P0-2 里的这些门禁仍然有效。

**放弃条件**

如果 4 周内训练器的数值门禁不能全部通过，就把训练端切到 PyTorch（需要 ADR），C++ 只保留推理端。

**改进后实验**

同一份数据、同一个模型，自研训练器与参考实现各跑 3 个 seed，最终 val loss 和 SR 的差异应落在 seed 方差之内。

---

## 4. P1 风险

### P1-1 CF-Drone 高度疑似 Flix 派生——上游已有运行原始 Arduino 代码的 Gazebo 仿真

**对照**

| 特征 | CF-Drone（方案 §1.1 所述） | Flix（公开仓库） |
|---|---|---|
| 主循环 | readIMU → step → readRC → estimate → control → sendMotors → handleInput，开 Wi-Fi 时再加 MAVLink | 单线程 Arduino 主循环，运行在 1000 Hz |
| 电机顺序 | 注释为 RL, RR, FR, FL | 0 号左后、1 号右后、2 号右前、3 号左前 |
| 控制链 | attitudeTarget → ratesTarget → torqueTarget → motors[4] | 同名的控制目标，motors 为 [0,1] 的浮点数组 |
| 定高 | ALTHOLD 是桩函数，"待气压计实现" | 气压计暂未使用，位置控制在开发中 |
| 芯片 | ESP32 / C3 / S3 各有不同引脚 | 支持 ESP32-S3 / C3 |
| 通信 | Wi-Fi 收发 MAVLink；AUTO 模式下接受 SET_ATTITUDE_TARGET | 通过 Wi-Fi 或 ESP-NOW 收发 MAVLink；提供 Python 脚本库 |

**影响**

1. **可以复用上游仿真。** Flix 的仿真器由三部分组成：机体模型 `flix.sdf`；把虚拟传感器数据交给 Arduino 代码的 Gazebo 插件 `simulator.cpp`；部分实现 Arduino API 的 `Arduino.h`。这正是 CF0/CF1 计划从零写的东西。
   - 限制：它基于 Gazebo 11，推荐 Ubuntu 20.04。Gazebo Classic 已于 2025 年 1 月停止支持，Ubuntu 24.04 上没有它；Harmonic 支持到 2028 年 9 月。
   - 可选路线：在 20.04 容器里跑原版，作为 golden oracle；或者把 `simulator.cpp + Arduino.h` 的思路移植成 gz-sim 8 的 System 插件。`flix.sdf` 也比 X3 更接近 118 mm 的微型机。
2. **频率不一致。** Flix 主循环是 1 kHz，而方案 §4.8 设定物理 500 Hz、CF 内环 200–500 Hz。如果 CF 沿用了按每步固定系数的滤波器，SIL 行为会偏离真机。建议物理步长 ≤ 1 ms，或者让控制核按 1 kHz 子步运行。【推测，需看 CF 源码确认】
3. **坐标约定。** Flix 从 v1.0 起改用前-左-上（FLU）约定。CF 的坐标约定要在单测里显式断言（CODE_RULES 第 6 条已有此要求）。
4. **许可证来源。** CF-Drone 自称 GPLv3，外加商业限制和品牌条款；而上游仓库页面侧栏没有显示许可证，HelloGitHub 也标为 None。在发表或分发任何派生代码之前，先弄清 CF-Drone 与上游的关系和授权情况。这只是风险提示，不是法律意见。

**证据**

【文献】Flix 的 README、`docs/firmware.md`、`gazebo/README.md`。派生关系属于【推测】。

**争议 / 反例**

把 CF-Drone 的 `control.ino`、`estimate.ino`、`motors.ino` 与 Flix 对应文件做 diff。如果差异超过一半，或者结构明显不同，本条撤回。

**最小改动**

先做 diff，不改固件。然后再决定 SIL 路线。CF0 的 golden vectors 直接由"原版仿真器 + 原始代码"生成。

**放弃条件**

如果原版仿真器在容器里也跑不起来，就回到方案的路线 B，但以 `flix.sdf` 的机体参数作为起点。

### P1-2 日志与指标契约在方案和代码之间漂移，SR 无法审计

| 来源 | 字段名 | 事件类型 |
|---|---|---|
| 方案 §12.3 | `kind`、`loss_total` | sim_step、eval_episode、hardware 等 |
| 代码（头文件、dashboard.py、HTML） | `event`、`loss` | 只认 train_step、validation |

【实测】把方案 §12.3 的 4 行示例喂给看板：0 条显示，4 条 `LOG_EVENT_INVALID`。

其他缺口：

- 方案 §12.1 要求记录的 loss_action、loss_stop、梯度范数、吞吐、显存，头文件一项都不写。
- validation 事件只有 `sr` 和 `stop_f1`，没有 split、episode 数、seed，也没有"是否闭环"的标记。界面上的"Val Success Rate"可能只是 teacher-forcing 下的动作匹配率。

**最小改动**

选定一份契约（建议以代码为准，因为已经有测试覆盖），写进 `docs/EPISODE_SCHEMA.md`。补丁已经做了两件事：

- 头文件写出的每一行都带 `"schema":"tinyvln.metrics.v1"`；
- `Validation` 扩展为 `(step, sr, stop_f1, split, n_episodes, closed_loop)`。

HTML 还需要在 SR 卡片下方显示 split 和 n，补丁没有改 HTML。复现脚本里的 P1-2 这一项，要等你决定以哪份契约为准后才会消失。

### P1-3 代码预算与依赖规则互相冲突

CODE_RULES §2.1 已经把 24 个文件名全部分配完（core 3 + ops 7 + model 6 + data 2 + train 2 + export 1 + board 3），余量为零。总量上限是 24 × 280 = 6720 有效行；CUDA 是 6 × 360 = 2160 行。

下列必需功能都没有分到文件：

- (a) 训练器要读 `--config config/experiment.yaml`，C++ 里需要 YAML 解析；
- (b) episode 图像的解码；
- (c) 在一个 ≤ 280 行的 `onnx_export.cc` 里手写 ONNX protobuf；
- (d) 板端的相机采集与预处理——board 目录只有 infer_runtime、uart_bridge、safety_fsm 三个文件；
- (e) SR、NE、nDTW、SDTW 的计算。

同时，CODE_RULES §10 规定 tools/ 只能用 Python 标准库，于是 MiniWorld/BabyAI 的 adapter 和 `onnx` 导出包都无法使用。

**最小改动**

写一份 ADR，包含三点：

- 允许把 header-only 的第三方库（如 stb_image、单头文件的 YAML/JSON 库）放进 third_party；
- 允许一个隔离的 Python 导出环境——onnx/onnxruntime 只用于导出和 parity 检查，不进入训练进程；
- 把板端的相机采集列入预算。

**放弃条件**

如果 ADR 不通过，就把 ONNX 导出改为"权重文件 + 固定图模板"，由隔离的 Python 工具生成。

### P1-4 统计功效不足以支撑"等容量配对消融"

80% 功效、α = 0.05、非配对时，每组需要的 episode 数：

| 想检测的差异 | 每组 episode 数 |
|---|---|
| SR 10% → 20% | 约 199 |
| SR 30% → 40% | 约 356 |
| SR 30% → 35% | 约 1376 |

另外两个数字：

- n = 100、SR = 30% 时，Wilson 95% CI 为 [21.9%, 39.6%]，宽 17.7 个百分点。
- episode 嵌套在场景里。每个场景 10 条、ICC 为 0.1–0.3 时，设计效应是 1.9–3.7，有效样本量要相应缩小。

方案 §6.5 规划了训练集的规模，但测试集规模写的是"由功效分析决定"，却没有给出任何数字。

**最小改动**

- 预先登记 val_unseen 的规模和主要对比；
- 在同一批 episode 上做成对比较（McNemar 检验）；
- 报告按场景聚类的 bootstrap CI；
- 成功率很低时，以 NE、nDTW、进度误差作为主指标。

### P1-5 硬件可行性与退化方式未评估

- **载荷与供电**：在刷式微型机上加装 LicheeRV Nano 和相机之后，推重比还剩多少余量、1S 电池如何升压到 5 V、电机电流尖峰会不会造成掉压重启，方案都没有测量计划。Flix 自己也把升压模块列为"为了更稳定供电"的可选件。
- **退化方式**：方案 §1.1 记录的唯一退化手段 `descend()` 是开环的推力渐降。空中断链时，下降速度不受控制。

**最小改动**

在无桨台架上测三件事：

1. 全机重量，以及单个电机的静推力–PWM 曲线；
2. 满油门瞬态下，SG2002 供电电压的最低值（用示波器测）；
3. 断链后 `descend()` 实际输出的推力曲线。

**放弃条件**

如果满载推重比小于 2，或者掉压能让 SG2002 复位，就不进入系留测试。

### P1-6 新颖性空间（承接第一版）

- L2Stop 已经有独立的 STOP 模块；
- Self-Monitoring 已经有进度估计；
- X-Fly 在空中 VLN 里用了独立的 Bernoulli STOP head。

所以"独立 STOP"本身不能作为贡献。可能的增量只剩一个组合：SG2002 级别的端侧部署 + 单目 RGB + 带统计保证的三态终止。这需要由 P0-1 的定义和 P1-4 的统计设计来证明。不能把"没找到先例"写成"首篇"。

### P1-7 协议与数据契约文档缺位，优先级倒置

- NavSetpoint 的 value_1 / value_2 写着"单位见协议表"，但协议表不存在。
- 方案 §10 规划的 ACTION_PROTOCOL、EPISODE_SCHEMA、CF_INTEGRATION、SIM_VALIDATION 四份文档，一份都没有。
- CODE_RULES 提到 ADR 大约 11 次，却没有 ADR 模板。

与此同时，UI 和评审相关的文档已经有 3 份，"docs ≤5"的名额被占掉了一大半。建议下一轮把 UI 文档合并成 1 份，把名额让给契约文档。

---

## 5. P2 建议

| 位置 | 问题 | 建议 |
|---|---|---|
| `sim/README.md:3`、方案第 662 行 | 引用的是 `V7_RESEARCH_PLAN.md`，实际文件名为 `V7_1_RESEARCH_PLAN.md` | 改正引用 |
| `CODE_RULES.md:87` | 引用的 V6 方案 §7.1.1 不在评审包里 | 把公共头文件和 CUDA 候选清单搬进 CODE_RULES |
| CODE_RULES "docs ≤5" | 已有 3 份，另规划 4 份，合计 7 份 | 合并 UI 文档 |
| `code/tests/` | Python 工具的测试放在 code/ 下，违反 CODE_RULES §1"code/ 只放 C++/CUDA" | 移到 `tools/tests/` |
| 场景清单 | 没有记录生成器版本或 hash、Python 版本；`random.choice` 不保证跨 Python 版本可复现 | 清单里记录 `generator_sha256` 和 `python_version` |
| `/api/audit` | 没有 git commit、文件 hash、测试结果，作为评审证据价值有限 | 补丁已加入文件 sha256；git commit 由导出脚本补上 |
| 退出码 | launcher 的 3 表示 BINARY_NOT_BUILT，sim_preflight 的 3 表示 FEATURE_NOT_IMPLEMENTED | 统一退出码表（补丁已实现，见附录 B） |
| `launcher.py:53` | 进程启动失败也会留下空的 run 目录，用同名重试会被 RUN_EXISTS 拦住 | 失败时清理或做标记 |
| HTML 曲线 | 断点续训会产生重复的 step，折线会往回折 | 按 step 去重 |
| MiniWorld | 方案已限定其用途。补充：Farama 已宣布停止维护（见第一版报告） | 锁定版本，或换用其他环境 |

---

## 6. 评审包断言 C01–C10 的核验

| 断言 | 核验结论 | 证据 |
|---|---|---|
| C01 `run.sh` 是唯一入口 | 成立，且使用了 `set -Eeuo pipefail` | 【源码】 |
| C02 只绑定 127.0.0.1 | 成立，但见 B05 | 【实测】 |
| C03 GET 只读取本地文件 | 成立，但见 B01–B04、B07 | 【实测】 |
| C04 POST 返回 405 | 成立；PUT、DELETE、OPTIONS、HEAD 返回 501 | 【实测】 |
| C05 五个页面 | 成立 | 【源码】 |
| C06 拒绝 NaN 和损坏行 | **部分不成立**：嵌套 NaN 会让整个 run 返回 404（B01、B02） | 【实测】 |
| C07 现有场景是静态 SDF | 成立，collision 与 visual 一致；但存在 P0-2 的问题 | 【实测】 |
| C08 拒绝没有电机插件的 world | 成立，但只是子串匹配（B15） | 【实测】 |
| C09 camera / cf-sil 返回未实现 | 成立（退出码 3） | 【实测】 |
| C10 未打包 CF-Drone 固件 | 成立 | 【源码】 |
| "13 + 2 个测试通过" | ZIP 状态下成立；全新 git clone 后不成立（B16） | 【实测】 |

---

## 7. 评审包 18 个问题逐条回答

1. **任务定义是否仍是空中 VLN？** 文档的定义仍然是 VLN。但在现有场景生成器下，语言不是必要信息（P0-2）。在加入盲基线和语言干预门禁之前，只能称作"计划中的空中 VLN"。
2. **学生输入是否泄漏真值？** 目前还没有学生代码，无法检查。数据层的隔离规则（CODE_RULES §10 中 SceneSpec 与 EpisodeObservation 分离）写得对。新发现的风险有两个：目标标记对学生相机可见（P0-2）；场景清单含全部障碍坐标，数据加载器必须禁止读取它。建议加输入 schema 的白名单测试，以及对隐状态做线性探针。
3. **单目、无里程计下哪些状态不可观测？** 绝对距离、绝对高度、累计位移、被遮挡的地标是否已经经过，都只能弱估计。"不确定"应定义为"校准后的预测集包含多个动作"（conformal 方法，可参考 KnowNo），而不是一个手调的阈值。
4. **与先例的实质差异？** 见 P1-6。ECPT 还没有定义，目前无法回答。
5. **STOP head 是否优于 8 路 softmax？** 方案 §15.1 已经列了 TinyVLN-STOP，并要求"同参数、同数据、同阈值调优规则"，但没写阈值在哪个 split 上选。建议：阈值和温度只在 val_seen 上选，val_unseen 只跑一次；softmax 版本补上等量参数；报告 OSR−SR 和过早停止率；至少 5 个 seed，用 McNemar 检验。
6. **是否依赖稀有成功？** 是。据第一版报告引用的 LookasideVLN 结果，AerialVLN-S val_unseen 上 Seq2Seq 的 SR 只有 2.3%，CMA 只有 3.2%。功效数字见 P1-4。应以 NE、nDTW、进度误差为主指标，并按场景聚类做统计。
7. **BabyAI / MiniWorld 的合法用途？** 方案 §6.1 已限定为功能验证，这是合理的。补充一点：MiniWorld 已停止维护。
8. **程序化 SDF 的碰撞与相机坐标系？** 生成器的障碍物 collision 与 visual 一致【源码】。但场景里没有机体、相机和插件。相机 optical frame 的变换，需要用方案 §4.3 的红柱测试来验证。
9. **X3 的 rad/s 与 CF 归一化混控如何标定？** 方案 §4.5 的思路是对的。补充：优先用 `flix.sdf` 加实测推力台数据，不要用 X3（P1-1）。
10. **各种周期与仿真时间如何统一？** 方案 §4.8 的锁步设计是对的。补充：CF 内环 1 kHz 与物理 500 Hz 不一致（P1-1）。
11. **IMU 的职责分离是否合理？** 合理，方案 §4.7 已写明。论文里需要说明：学生不融合 IMU，会带来尺度不可观测的问题。
12. **5 m / 15° 是否被错用到飞控？** 方案 §5.4 已经禁止直接映射，这是合理的。剩余的缺口是 NavIntent 的单位和语义没有定义（P1-7）。
13. **24 / 6 个文件的上限是否够用？** 目前是 0 个源文件。问题在于预算已被预先分配完，而且漏掉了若干必需功能（P1-3）。
14. **训练器有没有 oracle、gradcheck、续训、parity？** 方案 §8.2 和 CODE_RULES §7 都写了要求，但代码为零（P0-3）。
15. **TPU-MLIR 与 TOPS→FPS？** 方案和 HTML 都正确声明了"不用 TOPS 推算 FPS"。仍然需要"先把随机权重的 cvimodel 跑上板"这一探针（第一版 P1-4）。
16. **未来 ESP32 修改如何分阶段？** 方案 §9 的协议草案（seq、TTL、CRC、ACK、EXEC_DONE、按本地到达时间判过期）质量不错。建议分 4 个 PR：协议回显 → 心跳与 TTL 失联保护 → 定高（需要先有传感器）→ HIL 故障注入测试集。动手之前，先与上游 Flix 做 diff（P1-1）。
17. **状态等级能否防止"工具存在即 PASS"？** 不能。清单损坏时仍显示 ready（B06），9 项中有 5 项是硬编码（P0-3）。补丁加上了 `derived_from` 标注。
18. **HTTP 风险是否可接受？** 打补丁前，B01–B05 不可接受；打补丁后，作为单机开发工具可以接受。

---

## 8. 保留 / 修改 / 删除

| 模块 | 判定 | 代价 | 复现门禁 |
|---|---|---|---|
| `dashboard.py` | 修改 | 约 1 人日（补丁已提供） | 补丁内 10 个回归测试通过；复现脚本中 B01–B07 全部为"已修复" |
| `dashboard.html` | 小改 | 约 0.5 人日 | SR 卡片显示 split 和 n；续训重复的 step 去重 |
| `training_monitor.hpp` | 修改 | 补丁已提供 | B12–B14 为"已修复" |
| `launcher.py` | 修改 | 补丁已提供 | B08–B11 为"已修复" |
| `sim_preflight.py` | 修改 | 补丁已提供 | B15 为"已修复" |
| `scene_tool.py` 的任务生成部分 | 重写 | 3–5 人日 | P0-2 门禁去掉 expectedFailure 后能通过；盲基线 SR < 20% |
| 从零自研 CF-ControlCore SIL | **删除**，改为复用或移植 Flix 仿真器 | diff 约 0.5 人日 | 用原版仿真器生成 golden vectors |
| 用 X3 验证 CF 的 PID | **删除**（X3 只用于 GZ0 管线验证） | — | — |
| ECPT | 冻结，直到给出定义 | 写定义约 1–2 人日 | ADR 通过 |
| 知识蒸馏 / 大型教师 | 暂缓（方案已如此安排） | — | 教师的闭环表现优于几何专家 |
| UI 新功能 | 冻结（方案 V7.1-G 已如此安排） | — | — |
| 物理飞行 | 维持禁止 | — | P1-5 的台架三项 + HIL |

---

## 9. 三项分别评分

| 维度 | 分数 | 理由 |
|---|---|---|
| 研究发表创新性 | 3/10 | ECPT 没有定义，无从论证；独立 STOP 和进度估计都已有先例 |
| 物理飞行安全性 | 3/10 | 流程设计好（门禁、HIL、禁止自由飞行）；但没有任何硬件证据，供电、载荷、开环下降这几类风险未覆盖 |
| UI 可信度 | 5/10 | 加分项：如实显示空状态、textContent、CSP、不伪造数据。扣分项：一行坏日志拖垮整个 run、DNS rebinding、硬编码状态、损坏清单仍显示 ready、契约漂移。打补丁后预计可到 7/10 |

---

## 10. 接下来两周

| 时间 | 动作 | 验收标准 |
|---|---|---|
| 第 1 周 | 合入补丁，或自行修复 B01–B16 | 复现脚本里"存在"只剩 P0-2、P0-2b、P1-2 三项 |
| 第 1 周 | 写 ECPT 定义、EPISODE_SCHEMA、ACTION_PROTOCOL，统一 JSONL 契约 | ADR 通过；P1-2 这一项消失 |
| 第 1 周 | CF-Drone 与 Flix 做 diff，确认来源和许可 | 产出 diff 报告 |
| 第 2 周 | 重写 scene_tool 的任务生成，加入盲基线门禁 | 去掉 expectedFailure 后测试通过 |
| 第 2 周 | 在 5090 上跑 GZ0（官方 X3） | 拿到 motor→pose trace，并锁定版本 |
| 之后 | 最小训练器的数值门禁；SG2002 随机权重上板 | gradcheck、oracle、parity 通过；实测 p95 延迟 |

---

## 11. 文献与外部事实核查

| 项目 | 结论 | 链接 |
|---|---|---|
| AerialVLN（ICCV 2023） | 存在。动作单位为水平 5 m、垂直 2 m、转向 15°，成功阈值 20 m；训练集中 stop 只占约 1% | https://arxiv.org/pdf/2308.06735 、https://openaccess.thecvf.com/content/ICCV2023/supplemental/Liu_AerialVLN_Vision-and-Language_Navigation_ICCV_2023_supplemental.pdf 、https://arxiv.org/pdf/2503.11091 |
| LookasideVLN | 存在（arXiv 2604.17190） | https://arxiv.org/abs/2604.17190 |
| AirAnchor | 存在（arXiv 2609.08442），零样本，基于 MLLM | https://arxiv.org/abs/2609.08442 |
| LightVLN | 存在（arXiv 2610.05024），语言骨干约 0.5B | https://arxiv.org/abs/2610.05024 |
| QAD | **有歧义**：可能是 "Query, Align, and Distill"（VLN 蒸馏），也可能是 Quantization-Aware Distillation。请提供准确引用 | https://arxiv.org/html/2609.33097 、https://www.alphaxiv.org/abs/2601.20088v1 |
| X-Fly（评审包未列出，强相关） | 空中 VLN 中的独立 Bernoulli STOP head + recovery DAgger | https://arxiv.org/html/2610.21450 |
| Self-Monitoring（ICLR 2019） | 进度监控器 | https://github.com/chihyaoma/selfmonitoring-agent |
| L2Stop（Findings of EMNLP 2020） | 独立的 STOP 模块 | https://aclanthology.org/2020.findings-emnlp.62/ |
| KnowNo（CoRL 2023） | 用 conformal prediction 决定何时求助 | https://arxiv.org/abs/2307.01928 |
| MiniWorld / BabyAI | 2D 平面图、无文本指令、已停止维护 / 2D 网格、合成语言 | https://miniworld.farama.org/content/design/ 、https://arxiv.org/pdf/1810.08272v4 |
| Gazebo X3 示例 | maxRotVelocity 800、motorConstant 8.54858e-06，属于公斤级机体 | https://github.com/gazebosim/gz-sim/blob/gz-sim8/examples/worlds/quadcopter.sdf |
| gz-sim 已知问题 | #2637（推力计算）、#2657（Harmonic 中偏航方向反了） | https://github.com/gazebosim/gz-sim/issues/2637 、https://github.com/gazebosim/gz-sim/issues/2657 |
| gz-sim 8 可见性组件 | 视觉实体有 VisibilityFlags，相机传感器有 VisibilityMask | https://gazebosim.org/api/sim/8/Visibility_8hh.html |
| Gazebo Classic 停止支持 | 2025 年 1 月停止支持，Ubuntu 24.04 上没有；Harmonic 支持到 2028 年 9 月 | https://discourse.openrobotics.org/t/gazebo-classic-end-of-life/48748 、https://discourse.openrobotics.org/t/gazebo-classic-and-citadel-end-of-life/48152 |
| Flix | ESP32 Arduino 飞控；Gazebo 11 仿真器运行原始代码；主循环 1000 Hz；v1.0 起使用 FLU 约定 | https://github.com/okalachev/flix 、https://github.com/okalachev/flix/blob/master/docs/firmware.md 、https://github.com/okalachev/flix/blob/master/gazebo/README.md 、https://www.github.com/okalachev/flix/releases |
| Flix 许可证 | 仓库侧栏没有显示许可证；HelloGitHub 标为 None | https://hellogithub.com/en/repository/okalachev/flix |
| SG2002 / LicheeRV Nano | 1 TOPS INT8，256 MB DDR3 | https://wiki.sipeed.com/hardware/en/lichee/RV_Nano/1_intro 、https://wiki.sipeed.com/maixpy/en/index.html |
| TPU-MLIR CV18xx | 只支持 BF16 和对称 INT8；LayerNorm 默认留在浮点 | https://tpumlir.org/quick_start_en/Appx.02_cv18xx_guide.html |
| RTX 5090 | sm_120，需要 CUDA ≥ 12.8 | https://discuss.pytorch.org/t/is-there-a-pytorch-build-that-supports-nvidia-rtx-5090-compute-capability-12-0-sm-120/223536 |
| http.server 与 DNS rebinding | 官方文档说明不建议用于生产；防御方法是 Host 白名单 | https://docs.python.org/3/library/http.server.html 、https://github.com/nccgroup/singularity/wiki/Preventing-DNS-Rebinding-Attacks |

---

## 附录 A. 复现脚本：用法与本次输出

```bash
python3 review_v2_repro.py /path/to/TinyAerialVLN_V7_1_工程
```

- 脚本会把工程复制到临时目录，**不修改你的原目录**。
- 需要 Python 3.9+；如果有 g++ 和 git，会多跑 B12–B14 和 B16a 两组检查。
- 脚本会在临时副本里创建假的 `build/tinyvln_*`，用来测试启动器契约，不会触碰任何硬件。

**原版 V7.1 的输出：**

```text
[P0-2 ] 存在  起点→终点直线在所有 preset/seed 下都畅通  —— 最小净空 1.15 m
[P0-2b] 存在  mixed preset 与 forest 输出完全相同
[B16c ] 存在  .gitignore 未忽略 build/
[B05  ] 存在  伪造 Host 头仍可读取 /api/audit（DNS rebinding）  —— HTTP 200
[B01  ] 存在  一行嵌套 NaN 让整个 run 的指标接口失效  —— HTTP 404
[B02  ] 存在  深层嵌套 / 超长整数行让接口断连或 404  —— deep→无响应, bigint→404
[P1-2 ] 存在  方案 §12.3 的 JSONL 契约示例被看板全部拒收  —— 显示 0/4 条（补丁不改契约，需你决定以哪份为准）
[B03  ] 存在  根为 [] 的场景 manifest 让 /api/status 断连  —— HTTP 无响应
[B06  ] 存在  manifest 全部损坏时 scenes 仍为 ready  —— state=ready
[B07  ] 存在  超过 80 个 run 时计数与列表被静默截断  —— run_count=80，实际 89
[B04  ] 存在  OSError 消息含中文路径时连接无响应（文档声称 500）  —— HTTP 无响应
[B11  ] 存在  doctor 只凭 which 标 [OK]（坏掉的 nvcc 也算 OK）  —— exit=0
[B10  ] 存在  方案 §11 的 board/uart 子命令不报 FEATURE_NOT_IMPLEMENTED  —— exit=2
[B08  ] 存在  launcher 接受看板永远不显示的非 ASCII run ID  —— exit=0
[B09  ] 存在  按方案 §11 先 train 再 eval 同一 run-id 会失败  —— eval exit=4 [ERR][RUN_EXISTS] /tmp/tinyvln_review_g4
[B16b ] 存在  训练器编译后，现有测试会真的启动训练并失败  —— 已创建 runs/forbidden_test
[B15  ] 存在  只有速度控制插件的 world 通过“电机模型”检查  —— exit=127
[B13  ] 存在  全局 locale 带千分位/逗号小数时写出非法 JSON  —— {"event":"train_step","step":26.000,"total_steps":52.000,"lo
[B12  ] 存在  Step() 之后 std::cerr 被改成 fixed+precision(0)  —— 0
[B14  ] 存在  NaN 时先抛异常、不写错误事件
[B16a ] 存在  全新 git clone（artifacts/ 被忽略）后测试失败  —— code/tests: FAILED (errors=1); sim/tests: FAILED (failures=1)

汇总：存在 21，已修复 0，跳过 0
```

**打补丁后的输出：**

剩下的 3 项是设计层面的决定，补丁有意没有改动。

```text
[P0-2 ] 存在  起点→终点直线在所有 preset/seed 下都畅通  —— 最小净空 1.15 m
[P0-2b] 存在  mixed preset 与 forest 输出完全相同
[B16c ] 已修复  .gitignore 未忽略 build/
[B05  ] 已修复  伪造 Host 头仍可读取 /api/audit（DNS rebinding）  —— HTTP 421
[B01  ] 已修复  一行嵌套 NaN 让整个 run 的指标接口失效  —— HTTP 200
[B02  ] 已修复  深层嵌套 / 超长整数行让接口断连或 404  —— deep→200, bigint→200
[P1-2 ] 存在  方案 §12.3 的 JSONL 契约示例被看板全部拒收  —— 显示 0/4 条（补丁不改契约，需你决定以哪份为准）
[B03  ] 已修复  根为 [] 的场景 manifest 让 /api/status 断连  —— HTTP 200
[B06  ] 已修复  manifest 全部损坏时 scenes 仍为 ready  —— state=blocked
[B07  ] 已修复  超过 80 个 run 时计数与列表被静默截断  —— run_count=89，实际 89
[B04  ] 已修复  OSError 消息含中文路径时连接无响应（文档声称 500）  —— HTTP 500
[B11  ] 已修复  doctor 只凭 which 标 [OK]（坏掉的 nvcc 也算 OK）  —— exit=0
[B10  ] 已修复  方案 §11 的 board/uart 子命令不报 FEATURE_NOT_IMPLEMENTED  —— exit=3
[B08  ] 已修复  launcher 接受看板永远不显示的非 ASCII run ID  —— exit=4
[B09  ] 已修复  按方案 §11 先 train 再 eval 同一 run-id 会失败  —— eval exit=0 
[B16b ] 已修复  训练器编译后，现有测试会真的启动训练并失败
[B15  ] 已修复  只有速度控制插件的 world 通过“电机模型”检查  —— exit=4
[B13  ] 已修复  全局 locale 带千分位/逗号小数时写出非法 JSON  —— {"schema":"tinyvln.metrics.v1","event":"train_step","step":2
[B12  ] 已修复  Step() 之后 std::cerr 被改成 fixed+precision(0)  —— 0.37
[B14  ] 已修复  NaN 时先抛异常、不写错误事件
[B16a ] 已修复  全新 git clone（artifacts/ 被忽略）后测试失败  —— code/tests: OK (expected failures=1); sim/tests: OK

汇总：存在 3，已修复 18，跳过 0
```

---

## 附录 B. 最小补丁 `review_v2.patch`

**应用方式**（在工程根目录执行）：

```bash
patch -p1 < review_v2.patch      # 或 git apply review_v2.patch
python3 -m unittest discover -s code/tests -p 'test_*.py'   # 23 个测试，OK（其中 1 个 expected failure 是 P0-2 门禁）
python3 -m unittest discover -s sim/tests -p 'test_*.py'    # 4 个测试，OK
```

**改动范围**：8 个文件，新增 392 行有效代码，删除 91 行。没有改动任何 `.ino`、PID、引脚或飞控协议。

**流程上的注意**：按你的 CODE_RULES V7.1 第 10 条，本轮只允许改 `dashboard.py`、`launcher.py`、HTML 和一个测试文件。所以这个补丁应作为下一轮变更合入，或者附一份 ADR。

| 文件 | 改动 |
|---|---|
| `tools/dashboard.py` | Host 白名单；递归检查非有限值并限制嵌套深度；单行解析失败只跳过该行；清单必须是 JSON 对象并核对 sha256；状态行只写 ASCII 错误码；兜底的 500；报告真实 run 总数，按修改时间倒序；`O_NOFOLLOW` 打开文件；每个状态带 `derived_from`；audit 附带文件 sha256；端口被占用时给出可读的错误码 |
| `tools/launcher.py` | run ID 与看板共用同一条正则，并在探测环境之前先校验；eval 写入子目录并传 `--run-dir`；`board` / `uart` 返回 FEATURE_NOT_IMPLEMENTED；doctor 改标 `[FOUND]` 并运行版本探针；统一退出码表 |
| `tools/sim_preflight.py` | 解析 XML，统计 MulticopterMotorModel 插件数量，要求至少 4 个 |
| `code/include/tinyvln/training_monitor.hpp` | 固定 C locale，用 17 位有效数字；不再改动 `std::cerr` 的格式状态；先写 error 事件再抛出；支持 `resume_step`；每行带 `schema` 字段；`Validation` 增加 split、n_episodes、closed_loop（目前没有调用方，不会造成兼容问题） |
| `code/tests/test_infrastructure.py` | 自动创建 `artifacts/`；训练器存在时跳过"拒绝启动"测试 |
| `sim/tests/test_v7_contract.py` | 场景改为现场生成，不依赖被 git 忽略的文件；新增"只有速度控制插件"和"4 个电机插件"两个用例 |
| `code/tests/test_review_regressions.py`（新增） | B01–B16 的回归测试，外加标为 expectedFailure 的 P0-2 门禁。在原版上 10 个用例中有 9 个失败或报错 |
| `.gitignore` | 增加 `build/` |

**统一退出码**（launcher 与 sim_preflight 共用）

| 退出码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | 用法错误（argparse） |
| 3 | FEATURE_NOT_IMPLEMENTED |
| 4 | 输入或配置错误（RUN_ID_INVALID、CONFIG_NOT_READY、RUN_EXISTS、RUN_NOT_FOUND、GZ_WORLD_*、端口不可用） |
| 5 | 子进程启动失败 |
| 6 | BINARY_NOT_BUILT |
| 127 | 依赖缺失（GZ_NOT_INSTALLED、PYTHON_NOT_FOUND） |
| 130 | 用户中断 |

**补丁未处理的部分**，需要你来做决定：

- 场景生成器的重新设计（P0-2）；
- JSONL 以哪份契约为准（P1-2）；
- HTML 中显示 split 和 n。

**补丁全文：**

````diff
diff -ruN a/.gitignore b/.gitignore
--- a/.gitignore
+++ b/.gitignore
@@ -13,3 +13,4 @@
 *.mp4
 *.zip
 .DS_Store
+build/
diff -ruN a/code/include/tinyvln/training_monitor.hpp b/code/include/tinyvln/training_monitor.hpp
--- a/code/include/tinyvln/training_monitor.hpp
+++ b/code/include/tinyvln/training_monitor.hpp
@@ -1,6 +1,7 @@
 #pragma once
 // Real training hook: progress on stderr and structured metrics in metrics.jsonl.
 // Include from a FUTURE trainer; this header does not train a model by itself.
+// Every line carries "schema":"tinyvln.metrics.v1"; numbers are always written in the C locale.
 #include <algorithm>
 #include <chrono>
 #include <cmath>
@@ -8,10 +9,15 @@
 #include <fstream>
 #include <iomanip>
 #include <iostream>
+#include <limits>
+#include <locale>
+#include <sstream>
 #include <stdexcept>
 #include <string>
 
 namespace tinyvln {
+inline constexpr const char* kMetricsSchema = "tinyvln.metrics.v1";
+
 inline std::string JsonEscape(const std::string& s) {
   std::string out;
   for (char c : s) {
@@ -25,52 +31,77 @@
 
 class TrainingMonitor {
  public:
-  explicit TrainingMonitor(const std::string& jsonl_path)
-      : file_(jsonl_path, std::ios::app), begin_(Clock::now()) {
+  // resume_step: steps already completed when this process started (0 for a fresh run),
+  // so that ETA after a checkpoint resume is not underestimated.
+  explicit TrainingMonitor(const std::string& jsonl_path, std::size_t resume_step = 0)
+      : file_(jsonl_path, std::ios::app), begin_(Clock::now()), resume_step_(resume_step) {
     if (!file_) throw std::runtime_error("METRICS_OPEN_FAILED: " + jsonl_path);
+    file_.imbue(std::locale::classic());  // never "26,000" or "0,75" inside JSON
+    file_ << std::setprecision(std::numeric_limits<double>::max_digits10);  // lossless round trip
   }
 
   void Step(std::size_t step, std::size_t total, double loss,
             double accuracy, double lr) {
     if (!std::isfinite(loss) || !std::isfinite(accuracy) || !std::isfinite(lr))
-      throw std::runtime_error("TRAIN_NONFINITE_METRIC: loss/accuracy/lr");
+      Fail(step, "TRAIN_NONFINITE_METRIC", "loss/accuracy/lr is NaN or Inf");
     if (total == 0 || step > total || accuracy < 0 || accuracy > 1)
-      throw std::runtime_error("TRAIN_METRIC_RANGE: step/total/accuracy");
+      Fail(step, "TRAIN_METRIC_RANGE", "step/total/accuracy out of range");
     const double elapsed = std::chrono::duration<double>(Clock::now() - begin_).count();
-    const double eta = step ? elapsed * (static_cast<double>(total - step) / step) : 0;
-    file_ << "{\"event\":\"train_step\",\"step\":" << step
+    const std::size_t done = step > resume_step_ ? step - resume_step_ : 0;
+    const double eta = done ? elapsed * (static_cast<double>(total - step) / done) : 0;
+    file_ << "{\"schema\":\"" << kMetricsSchema << "\",\"event\":\"train_step\",\"step\":" << step
           << ",\"total_steps\":" << total << ",\"loss\":" << loss
           << ",\"action_accuracy\":" << accuracy << ",\"lr\":" << lr
           << ",\"elapsed_sec\":" << elapsed << ",\"eta_sec\":" << eta << "}\n";
     file_.flush();
+    // Format in a private stream: std::cerr's flags/precision are never modified.
+    std::ostringstream bar;
+    bar.imbue(std::locale::classic());
     constexpr int kWidth = 28;
     const int filled = static_cast<int>(kWidth * static_cast<double>(step) / total);
-    std::cerr << '\r' << "[TRAIN] [" << std::string(filled, '#')
-              << std::string(kWidth - filled, '-') << "] " << step << '/' << total
-              << " loss=" << std::fixed << std::setprecision(4) << loss
-              << " acc=" << std::setprecision(1) << accuracy * 100 << '%'
-              << " ETA=" << std::setprecision(0) << eta << "s" << std::flush;
-    if (step == total) std::cerr << '\n';
+    bar << '\r' << "[TRAIN] [" << std::string(filled, '#')
+        << std::string(kWidth - filled, '-') << "] " << step << '/' << total
+        << " loss=" << std::fixed << std::setprecision(4) << loss
+        << " acc=" << std::setprecision(1) << accuracy * 100 << '%'
+        << " ETA=" << std::setprecision(0) << eta << "s";
+    if (step == total) bar << '\n';
+    std::cerr << bar.str() << std::flush;
   }
 
-  void Validation(std::size_t step, double sr, double stop_f1) {
+  // split: e.g. "val_unseen"; n_episodes: episodes behind `sr`;
+  // closed_loop=false marks teacher-forced / offline proxies so the UI cannot pass them off as SR.
+  void Validation(std::size_t step, double sr, double stop_f1, const std::string& split,
+                  std::size_t n_episodes, bool closed_loop) {
     if (!std::isfinite(sr) || sr < 0 || sr > 1 || !std::isfinite(stop_f1)
-        || stop_f1 < 0 || stop_f1 > 1)
-      throw std::runtime_error("VAL_METRIC_INVALID");
-    file_ << "{\"event\":\"validation\",\"step\":" << step
-          << ",\"sr\":" << sr << ",\"stop_f1\":" << stop_f1 << "}\n";
+        || stop_f1 < 0 || stop_f1 > 1 || n_episodes == 0)
+      Fail(step, "VAL_METRIC_INVALID", "sr/stop_f1 outside [0,1] or n_episodes == 0");
+    file_ << "{\"schema\":\"" << kMetricsSchema << "\",\"event\":\"validation\",\"step\":" << step
+          << ",\"sr\":" << sr << ",\"stop_f1\":" << stop_f1
+          << ",\"split\":\"" << JsonEscape(split) << "\",\"n_episodes\":" << n_episodes
+          << ",\"closed_loop\":" << (closed_loop ? "true" : "false") << "}\n";
     file_.flush();
   }
 
   void Status(const std::string& event, const std::string& message) {
-    file_ << "{\"event\":\"" << JsonEscape(event) << "\",\"message\":\""
-          << JsonEscape(message) << "\"}\n";
+    file_ << "{\"schema\":\"" << kMetricsSchema << "\",\"event\":\"" << JsonEscape(event)
+          << "\",\"message\":\"" << JsonEscape(message) << "\"}\n";
     file_.flush();
   }
 
  private:
   using Clock = std::chrono::steady_clock;
+
+  // Record why training stopped before throwing, so the dashboard can show the cause.
+  [[noreturn]] void Fail(std::size_t step, const char* code, const char* message) {
+    file_ << "{\"schema\":\"" << kMetricsSchema << "\",\"event\":\"error\",\"severity\":\"FATAL\""
+          << ",\"code\":\"" << code << "\",\"step\":" << step
+          << ",\"message\":\"" << JsonEscape(message) << "\"}\n";
+    file_.flush();
+    throw std::runtime_error(std::string(code) + ": " + message);
+  }
+
   std::ofstream file_;
   Clock::time_point begin_;
+  std::size_t resume_step_;
 };
 }  // namespace tinyvln
diff -ruN a/code/tests/test_infrastructure.py b/code/tests/test_infrastructure.py
--- a/code/tests/test_infrastructure.py
+++ b/code/tests/test_infrastructure.py
@@ -16,6 +16,7 @@
 class InfrastructureTest(unittest.TestCase):
     def test_scene_deterministic_and_well_formed(self):
         from xml.etree import ElementTree as ET
+        (ROOT / 'artifacts').mkdir(exist_ok=True)  # artifacts/ is gitignored: absent in a fresh clone
         with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as d:
             output = Path(d)
             first = scene_tool.generate('forest', 123, 30, output)
@@ -36,6 +37,8 @@
             scene_tool.generate('forest', 42, -1, ROOT / 'artifacts' / 'scenes')
 
     def test_refuses_nonexistent_training_binary(self):
+        if (ROOT / 'build' / 'tinyvln_train').exists():
+            self.skipTest('trainer is built; this test must never launch real training')
         p = subprocess.run([str(ROOT / 'run.sh'), 'train', '--run-id', 'forbidden_test'],
                            text=True, capture_output=True)
         self.assertNotEqual(p.returncode, 0)
diff -ruN a/code/tests/test_review_regressions.py b/code/tests/test_review_regressions.py
--- a/code/tests/test_review_regressions.py
+++ b/code/tests/test_review_regressions.py
@@ -0,0 +1,169 @@
+"""Regression tests for the V7.1 review (v2) findings B01–B16 and gate P0-2."""
+import http.client
+import json
+from pathlib import Path
+import shutil
+import socket
+import subprocess
+import sys
+import tempfile
+import threading
+import time
+import unittest
+from unittest.mock import patch
+
+ROOT = Path(__file__).resolve().parents[2]
+sys.path.insert(0, str(ROOT / 'tools'))
+import dashboard  # noqa: E402
+import scene_tool  # noqa: E402
+
+GOOD = b'{"event":"train_step","step":1,"total_steps":2,"loss":0.5}\n'
+
+
+def free_port():
+    with socket.socket() as s:
+        s.bind(('127.0.0.1', 0))
+        return s.getsockname()[1]
+
+
+def start_server():
+    port = free_port()
+    threading.Thread(target=dashboard.serve, args=(port,), daemon=True).start()
+    for _ in range(50):
+        try:
+            socket.create_connection(('127.0.0.1', port), timeout=0.1).close()
+            return port
+        except OSError:
+            time.sleep(0.05)
+    raise RuntimeError('dashboard did not start')
+
+
+def get(port, path, host=None):
+    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
+    conn.request('GET', path, headers={'Host': host} if host else {})
+    resp = conn.getresponse()
+    return resp.status, resp.read()
+
+
+class MetricsRobustness(unittest.TestCase):
+    def run_with(self, payload):
+        tmp = tempfile.TemporaryDirectory()
+        self.addCleanup(tmp.cleanup)
+        base = Path(tmp.name)
+        (base / 'r1').mkdir()
+        (base / 'r1' / 'metrics.jsonl').write_bytes(GOOD + payload + GOOD)
+        with patch.object(dashboard, 'RUNS', base):
+            return dashboard.metrics_for('r1')
+
+    def test_B01_nested_nonfinite_line_is_skipped_not_fatal(self):
+        data = self.run_with(b'{"event":"train_step","extra":{"grad_norm":NaN}}\n')
+        self.assertEqual(len(data['events']), 2)
+        self.assertTrue(any('LOG_NONFINITE' in w for w in data['warnings']))
+
+    def test_B02_deep_nesting_and_huge_int_are_skipped(self):
+        for bad in (b'{"event":"x","v":' + b'[' * 30000 + b']' * 30000 + b'}\n',
+                    b'{"event":"x","v":' + b'9' * 5000 + b'}\n'):
+            self.assertEqual(len(self.run_with(bad)['events']), 2)
+
+
+class SceneManifests(unittest.TestCase):
+    def test_B03_B06_invalid_manifests_never_break_or_count_as_ready(self):
+        with tempfile.TemporaryDirectory() as tmp:
+            base = Path(tmp)
+            (base / 'a.json').write_text('[]')
+            (base / 'b.json').write_text('{broken')
+            with patch.object(dashboard, 'SCENES', base):
+                scenes = dashboard.scene_list()
+                self.assertTrue(all(s['status'] == 'manifest_invalid' for s in scenes))
+                check = {c['key']: c for c in dashboard.status()['checks']}['scenes']
+                self.assertEqual(check['state'], 'blocked')
+
+
+class RunListing(unittest.TestCase):
+    def test_B07_total_is_reported_beyond_listing_cap(self):
+        with tempfile.TemporaryDirectory() as tmp:
+            base = Path(tmp)
+            for i in range(85):
+                (base / f'r{i:03}').mkdir()
+                (base / f'r{i:03}' / 'metrics.jsonl').write_bytes(GOOD)
+            with patch.object(dashboard, 'RUNS', base):
+                self.assertEqual(dashboard.status()['run_count'], 85)
+                self.assertEqual(len(dashboard.runs_list()), dashboard.MAX_RUNS_LISTED)
+                dashboard.metrics_for('r000')  # listed or not, a valid run stays readable
+
+
+class HttpLayer(unittest.TestCase):
+    @classmethod
+    def setUpClass(cls):
+        cls.port = start_server()
+
+    def test_B05_foreign_host_header_is_rejected(self):
+        self.assertEqual(get(self.port, '/api/audit', host='evil.example:%d' % self.port)[0], 421)
+        self.assertEqual(get(self.port, '/api/audit')[0], 200)
+
+    def test_B04_io_error_with_non_ascii_path_still_answers_500(self):
+        def boom(_name):
+            raise PermissionError(13, 'Permission denied', '/home/u/项目_工程/metrics.jsonl')
+        with patch.object(dashboard, 'metrics_for', boom):
+            self.assertEqual(get(self.port, '/api/metrics?run=x')[0], 500)
+
+
+class LauncherContract(unittest.TestCase):
+    def run_sh(self, *args):
+        return subprocess.run([str(ROOT / 'run.sh'), *args], capture_output=True, text=True)
+
+    def test_B08_non_ascii_run_id_rejected_like_dashboard(self):
+        for bad in ('实验一', 'ｒｕｎ１'):
+            r = self.run_sh('train', '--run-id', bad)
+            self.assertIn('RUN_ID_INVALID', r.stderr)
+
+    def test_B10_planned_subcommands_report_not_implemented(self):
+        for args in (('board', 'doctor'), ('uart', 'dry-run')):
+            r = self.run_sh(*args)
+            self.assertEqual(r.returncode, 3)
+            self.assertIn('FEATURE_NOT_IMPLEMENTED', r.stderr)
+
+
+@unittest.skipUnless(shutil.which('g++'), 'g++ not available')
+class TrainingMonitorHeader(unittest.TestCase):
+    PROGRAM = r'''
+#include "tinyvln/training_monitor.hpp"
+#include <locale>
+struct Comma : std::numpunct<char> {
+  char do_decimal_point() const override { return ','; }
+  char do_thousands_sep() const override { return '.'; }
+  std::string do_grouping() const override { return "\3"; }
+};
+int main(int, char** argv) {
+  std::locale::global(std::locale(std::locale::classic(), new Comma));
+  tinyvln::TrainingMonitor m(argv[1]);
+  m.Step(26000, 52000, 0.75, 0.5, 2e-4);
+  std::cerr << "\nprobe=" << 0.37 << "\n";
+  try { m.Step(26001, 52000, 0.0 / 0.0, 0.5, 2e-4); } catch (const std::exception&) { std::cerr << "THROWN\n"; }
+}
+'''
+
+    def test_B12_B13_B14_locale_stream_state_and_error_event(self):
+        with tempfile.TemporaryDirectory() as tmp:
+            src, exe, log = Path(tmp) / 't.cpp', Path(tmp) / 't', Path(tmp) / 'm.jsonl'
+            src.write_text(self.PROGRAM)
+            subprocess.run(['g++', '-std=c++17', '-I', str(ROOT / 'code/include'), str(src), '-o', str(exe)],
+                           check=True, capture_output=True)
+            r = subprocess.run([str(exe), str(log)], capture_output=True, text=True)
+            events = [json.loads(line) for line in log.read_text().splitlines()]  # B13: valid JSON
+            self.assertEqual(events[0]['step'], 26000)
+            self.assertIn('probe=0.37', r.stderr)  # B12: std::cerr state untouched
+            self.assertIn('THROWN', r.stderr)
+            self.assertEqual(events[-1]['code'], 'TRAIN_NONFINITE_METRIC')  # B14: logged before throw
+
+
+class SceneGate(unittest.TestCase):
+    @unittest.expectedFailure  # P0-2: remove the decorator once the generator stops reserving a straight corridor
+    def test_P0_2_scenes_require_more_than_flying_straight(self):
+        blocked = sum(any(abs(o['y']) - o['r'] < 0.3 for o in scene_tool.create_obstacles('forest', seed, 60))
+                      for seed in range(200))
+        self.assertGreater(blocked / 200, 0.5)
+
+
+if __name__ == '__main__':
+    unittest.main()
diff -ruN a/sim/tests/test_v7_contract.py b/sim/tests/test_v7_contract.py
--- a/sim/tests/test_v7_contract.py
+++ b/sim/tests/test_v7_contract.py
@@ -1,18 +1,51 @@
 import pathlib
 import subprocess
+import sys
+import tempfile
 import unittest
 
 ROOT = pathlib.Path(__file__).resolve().parents[2]
+sys.path.insert(0, str(ROOT / 'tools'))
+import scene_tool  # noqa: E402
+
+MOTOR = ('<plugin filename="gz-sim-multicopter-motor-model-system" '
+         'name="gz::sim::systems::MulticopterMotorModel"/>')
+VELOCITY_ONLY = ('<plugin filename="gz-sim-multicopter-control-system" '
+                 'name="gz::sim::systems::MulticopterVelocityControl"/>')
+
+
+def smoke(world):
+    return subprocess.run([str(ROOT / 'run.sh'), 'sim', 'smoke', '--world', str(world), '--dry-run'],
+                          capture_output=True, text=True)
+
 
 class V7Contract(unittest.TestCase):
     def test_static_scene_rejected_as_flight(self):
-        scene = ROOT / 'artifacts/scenes/forest_42.sdf'
-        r = subprocess.run([str(ROOT/'run.sh'), 'sim', 'smoke', '--world', str(scene), '--dry-run'], capture_output=True, text=True)
-        self.assertNotEqual(r.returncode,0)
+        (ROOT / 'artifacts').mkdir(exist_ok=True)  # do not depend on gitignored, pre-generated files
+        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as d:
+            r = smoke(scene_tool.generate('forest', 42, 24, pathlib.Path(d)))
+        self.assertNotEqual(r.returncode, 0)
         self.assertIn('GZ_MOTOR_MODEL_MISSING', r.stderr)
+
+    def test_velocity_controller_alone_is_not_motor_dynamics(self):
+        with tempfile.TemporaryDirectory() as d:
+            world = pathlib.Path(d) / 'w.sdf'
+            world.write_text(f'<sdf version="1.9"><world name="w">{VELOCITY_ONLY}</world></sdf>')
+            r = smoke(world)
+        self.assertIn('GZ_MOTOR_MODEL_MISSING', r.stderr)
+
+    def test_four_motor_plugins_pass_the_static_check(self):
+        with tempfile.TemporaryDirectory() as d:
+            world = pathlib.Path(d) / 'w.sdf'
+            world.write_text(f'<sdf version="1.9"><world name="w">{MOTOR * 4}</world></sdf>')
+            r = smoke(world)
+        self.assertNotIn('GZ_MOTOR_MODEL_MISSING', r.stderr)
+
     def test_unimplemented_camera_cannot_false_pass(self):
-        r = subprocess.run([str(ROOT/'run.sh'), 'sim', 'camera'],capture_output=True,text=True)
-        self.assertNotEqual(r.returncode,0)
+        r = subprocess.run([str(ROOT / 'run.sh'), 'sim', 'camera'], capture_output=True, text=True)
+        self.assertNotEqual(r.returncode, 0)
         self.assertIn('FEATURE_NOT_IMPLEMENTED', r.stderr)
 
-if __name__ == '__main__': unittest.main()
+
+if __name__ == '__main__':
+    unittest.main()
diff -ruN a/tools/dashboard.py b/tools/dashboard.py
--- a/tools/dashboard.py
+++ b/tools/dashboard.py
@@ -1,50 +1,84 @@
 #!/usr/bin/env python3
 """V7.1 read-only local research cockpit. No training or simulator actions are executed here.
 
-All status values are derived from local files / binary presence. Dashboard does NOT
+Every status check carries `derived_from`: `file_probe` (computed from local files / binaries)
+or `hardcoded` (a hand-maintained label until a gate-evidence file exists). Dashboard does NOT
 claim that an available binary, world file or metric proves a validated flight.
 """
 from __future__ import annotations
 
 import argparse
+from collections import deque
 from datetime import datetime, timezone
+import hashlib
 from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
 import json
 import math
+import os
 from pathlib import Path
 import re
 import shutil
+import sys
 from urllib.parse import urlparse, parse_qs
 
 ROOT = Path(__file__).resolve().parents[1]
 RUNS = ROOT / 'artifacts' / 'runs'
 SCENES = ROOT / 'artifacts' / 'scenes'
 HTML = ROOT / 'assets' / 'dashboard.html'
-RUN_ID = re.compile(r'^[A-Za-z0-9_]{1,70}$')
+RUN_ID = re.compile(r'^[A-Za-z0-9_]{1,70}$')  # tools/launcher.py uses the same rule
 MAX_LOG_TAIL_BYTES = 2 * 1024 * 1024
 MAX_EVENTS = 2000
 MAX_LINE_BYTES = 64 * 1024
+MAX_RUNS_LISTED = 80
+MAX_MANIFEST_BYTES = 1024 * 1024
+MAX_JSON_DEPTH = 32
+PROVENANCE_FILES = ('tools/dashboard.py', 'tools/launcher.py', 'tools/scene_tool.py',
+                    'tools/sim_preflight.py', 'assets/dashboard.html')
 
 
 def utc_now():
     return datetime.now(timezone.utc).isoformat(timespec='seconds')
 
 
-def runs_list():
+def all_finite(value, depth=0):
+    """True only if every number in a bounded-depth JSON value is finite."""
+    if depth > MAX_JSON_DEPTH:
+        return False
+    if isinstance(value, float):
+        return math.isfinite(value)
+    if isinstance(value, dict):
+        return all(all_finite(v, depth + 1) for v in value.values())
+    if isinstance(value, list):
+        return all(all_finite(v, depth + 1) for v in value)
+    return True
+
+
+def run_ok(path):
+    metrics = path / 'metrics.jsonl'
+    return bool(RUN_ID.fullmatch(path.name) and path.is_dir() and not path.is_symlink()
+                and metrics.is_file() and not metrics.is_symlink())
+
+
+def all_runs():
     if not RUNS.is_dir():
         return []
-    return sorted((p.name for p in RUNS.iterdir() if p.is_dir() and not p.is_symlink()
-                   and RUN_ID.fullmatch(p.name) and (p / 'metrics.jsonl').is_file()
-                   and not (p / 'metrics.jsonl').is_symlink()), reverse=True)[:80]
+    runs = [p for p in RUNS.iterdir() if run_ok(p)]
+    runs.sort(key=lambda p: (p / 'metrics.jsonl').stat().st_mtime, reverse=True)  # newest first
+    return [p.name for p in runs]
+
+
+def runs_list():
+    return all_runs()[:MAX_RUNS_LISTED]
 
 
 def metrics_for(name):
-    if not RUN_ID.fullmatch(name) or name not in runs_list():
+    if not RUN_ID.fullmatch(name or '') or not run_ok(RUNS / name):
         raise ValueError('RUN_NOT_FOUND')
     path = RUNS / name / 'metrics.jsonl'
-    rows, warnings = [], []
-    size = path.stat().st_size
-    with path.open('rb') as f:
+    rows, warnings = deque(maxlen=MAX_EVENTS), []
+    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))  # no symlink swap after the check
+    with os.fdopen(fd, 'rb') as f:
+        size = os.fstat(f.fileno()).st_size
         offset = max(0, size - MAX_LOG_TAIL_BYTES)
         f.seek(offset)
         if offset:
@@ -59,20 +93,18 @@
                 continue
             try:
                 event = json.loads(raw.decode('utf-8'))
-            except (UnicodeDecodeError, json.JSONDecodeError):
+            except (UnicodeDecodeError, ValueError, RecursionError):
+                # ValueError also covers the int-digit limit; one bad line must never fail the run
                 warnings.append(f'LOG_JSON_INVALID: near_tail_line={number}')
                 continue
             if not isinstance(event, dict) or not isinstance(event.get('event'), str):
                 warnings.append(f'LOG_EVENT_INVALID: near_tail_line={number}')
                 continue
-            # Prevent NaN, Infinity, unexpected object graphs leaking into UI.
-            if any(isinstance(v, float) and not math.isfinite(v) for v in event.values()):
-                warnings.append(f'LOG_NONFINITE: near_tail_line={number}')
+            if not all_finite(event):  # recursive: nested NaN/Inf or over-deep objects
+                warnings.append(f'LOG_NONFINITE_OR_TOO_DEEP: near_tail_line={number}')
                 continue
             rows.append(event)
-            if len(rows) > MAX_EVENTS:
-                rows.pop(0)
-    return {'run_id': name, 'events': rows, 'warnings': warnings[-15:],
+    return {'run_id': name, 'events': list(rows), 'warnings': warnings[-15:],
             'truncated': size > MAX_LOG_TAIL_BYTES, 'source': 'real_jsonl_only'}
 
 
@@ -85,56 +117,81 @@
             continue
         sdf = path.with_suffix('.sdf')
         try:
+            if path.stat().st_size > MAX_MANIFEST_BYTES:
+                raise ValueError('manifest larger than 1 MiB')
             spec = json.loads(path.read_text(encoding='utf-8'))
+            if not isinstance(spec, dict):
+                raise ValueError('manifest root must be a JSON object')
+            sha_ok = None
+            if sdf.is_file() and isinstance(spec.get('sha256'), str):
+                sha_ok = hashlib.sha256(sdf.read_bytes()).hexdigest() == spec['sha256']
             scenes.append({'name': path.stem, 'preset': str(spec.get('preset', 'unknown')),
                            'seed': spec.get('seed'), 'obstacles': spec.get('placed_obstacles', spec.get('requested_obstacles')),
-                           'sdf_exists': sdf.is_file(), 'kind': 'static_geometry_only',
+                           'sdf_exists': sdf.is_file(), 'sdf_sha256_ok': sha_ok, 'kind': 'static_geometry_only',
                            'status': 'not_a_flight_simulator'})
-        except (OSError, ValueError, TypeError) as exc:
+        except (OSError, ValueError, TypeError, RecursionError) as exc:
+            detail = type(exc).__name__ if isinstance(exc, OSError) else str(exc)[:160]  # no local paths
             scenes.append({'name': path.stem, 'status': 'manifest_invalid',
-                           'error_code': 'SCENE_MANIFEST_INVALID', 'detail': str(exc)[:180]})
+                           'error_code': 'SCENE_MANIFEST_INVALID', 'detail': detail})
     return scenes
 
 
-def state_check(key, label, state, detail, command=None):
-    value = {'key': key, 'label': label, 'state': state, 'detail': detail}
+def state_check(key, label, state, detail, command=None, derived_from='file_probe'):
+    value = {'key': key, 'label': label, 'state': state, 'detail': detail, 'derived_from': derived_from}
     if command:
         value['command'] = command
     return value
 
 
 def status():
-    found_runs = runs_list()
+    found_runs = all_runs()
+    scenes = scene_list()
+    valid = sum(1 for s in scenes if s.get('status') == 'not_a_flight_simulator')
+    invalid = len(scenes) - valid
+    gz = shutil.which('gz')
+    trainer = (ROOT / 'build/tinyvln_train').is_file()
+    config = (ROOT / 'config/experiment.yaml').is_file()
     checks = [
-        state_check('dashboard', '本地指标看板', 'ready', '只读服务；只显示已记录的 JSONL', './run.sh dashboard --port 8765'),
-        state_check('scenes', '程序化静态场景', 'ready' if scene_list() else 'blocked',
-                    f'已发现 {len(scene_list())} 份 scene manifest；这不是 RGB 飞行仿真',
+        state_check('dashboard', '本地指标看板', 'ready', '只读服务；只显示已记录的 JSONL',
+                    './run.sh dashboard --port 8765', derived_from='hardcoded'),
+        state_check('scenes', '程序化静态场景', 'ready' if valid else 'blocked',
+                    f'有效 manifest {valid} 份，损坏 {invalid} 份；这不是 RGB 飞行仿真',
                     './run.sh scene --preset forest --seed 42 --count 24'),
-        state_check('gazebo', 'Gazebo CLI', 'available_unverified' if shutil.which('gz') else 'blocked',
-                    '已发现 gz CLI，但动力学/电机/RGB 尚未验证' if shutil.which('gz') else '当前机器找不到 gz 命令',
+        state_check('gazebo', 'Gazebo CLI', 'available_unverified' if gz else 'blocked',
+                    '已发现 gz CLI，但动力学/电机/RGB 尚未验证' if gz else '当前机器找不到 gz 命令',
                     './run.sh sim doctor'),
         state_check('physics', '四旋翼物理闭环 GZ0–GZ3', 'not_implemented',
-                    '尚无通过测试的四电机→动力学→RGB 运行链路'),
+                    '尚无通过测试的四电机→动力学→RGB 运行链路', derived_from='hardcoded'),
         state_check('cf_sil', 'CF-Drone 软件在环', 'not_implemented',
-                    '尚未移植 PID 控制核；原 ESP32 固件未改动'),
-        state_check('trainer', '自研 C++/CUDA 学生训练器',
-                    'available_unverified' if (ROOT / 'build/tinyvln_train').is_file() else 'not_implemented',
-                    '发现训练文件但尚未通过训练验证' if (ROOT / 'build/tinyvln_train').is_file() else '尚无已编译训练器；不提供模拟曲线'),
-        state_check('model_config', '训练配置', 'ready' if (ROOT / 'config/experiment.yaml').is_file() else 'blocked',
-                    '已提供训练配置' if (ROOT / 'config/experiment.yaml').is_file() else '仅有 .planned 占位，不得启动训练'),
+                    '尚未移植 PID 控制核；原 ESP32 固件未改动', derived_from='hardcoded'),
+        state_check('trainer', '自研 C++/CUDA 学生训练器', 'available_unverified' if trainer else 'not_implemented',
+                    '发现训练文件但尚未通过训练验证' if trainer else '尚无已编译训练器；不提供模拟曲线'),
+        state_check('model_config', '训练配置', 'ready' if config else 'blocked',
+                    '已提供训练配置（未校验内容）' if config else '仅有 .planned 占位，不得启动训练'),
         state_check('edge', 'SG2002 NPU 部署', 'not_implemented',
-                    '未提供设备实测 cvimodel / camera→UART 端到端结果'),
+                    '未提供设备实测 cvimodel / camera→UART 端到端结果', derived_from='hardcoded'),
         state_check('firmware', 'ESP32 固件', 'frozen',
-                    '本次只读参考 CF-Drone.zip；没有修改、烧录或验证新固件'),
+                    '本次只读参考 CF-Drone.zip；没有修改、烧录或验证新固件', derived_from='hardcoded'),
     ]
     return {'version': '7.1', 'generated_at': utc_now(), 'checks': checks,
-            'run_count': len(found_runs), 'scene_count': len(scene_list()),
+            'run_count': len(found_runs), 'runs_listed': min(len(found_runs), MAX_RUNS_LISTED),
+            'scene_count': valid, 'scene_invalid_count': invalid,
             'read_only': True, 'no_fake_metrics': True}
 
 
+def provenance():
+    hashes = {}
+    for rel in PROVENANCE_FILES:
+        try:
+            hashes[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
+        except OSError:
+            hashes[rel] = None
+    return {'file_sha256': hashes}
+
+
 def audit():
     return {'schema': 'tinyvln.audit.v1', 'generated_at': utc_now(),
-            'project': 'TinyAerialVLN V7.1', 'status': status(),
+            'project': 'TinyAerialVLN V7.1', 'status': status(), 'provenance': provenance(),
             'evidence_levels': {'ready': '文件或工具存在；不等于物理闭环验证',
                                 'available_unverified': '仅发现资源，缺运行验收',
                                 'blocked': '缺前置依赖',
@@ -147,6 +204,7 @@
                 '尚无 SG2002 端到端推理实测',
                 '没有修改 CF-Drone 的 ESP32 固件',
                 '静态障碍场景不等于可飞行的模拟器',
+                'derived_from=hardcoded 的状态由人工维护，不是自动验收结果',
                 '本页不支持开始训练或修改设备配置'
             ],
             'verification_commands': ['python3 -m unittest discover -s code/tests -p "test_*.py"',
@@ -155,8 +213,17 @@
 
 
 def serve(port):
+    allowed_hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}  # blocks DNS-rebinding reads
+
     class Handler(BaseHTTPRequestHandler):
+        def fail(self, code, error_code):
+            # The reason phrase goes into the latin-1 status line: keep it an ASCII error code.
+            self.send_error(code, error_code)
+
         def do_GET(self):
+            if self.headers.get('Host', '') not in allowed_hosts:
+                self.fail(421, 'HOST_NOT_ALLOWED')
+                return
             path = urlparse(self.path)
             try:
                 if path.path in ('/', '/index.html'):
@@ -164,7 +231,10 @@
                 elif path.path == '/api/health':
                     body, ctype, code = {'status': 'ok', 'read_only': True}, 'application/json', 200
                 elif path.path == '/api/runs':
-                    body, ctype, code = {'runs': runs_list()}, 'application/json', 200
+                    total = all_runs()
+                    body = {'runs': total[:MAX_RUNS_LISTED], 'total': len(total),
+                            'truncated': len(total) > MAX_RUNS_LISTED}
+                    ctype, code = 'application/json', 200
                 elif path.path == '/api/metrics':
                     name = parse_qs(path.query).get('run', [''])[0]
                     body, ctype, code = metrics_for(name), 'application/json', 200
@@ -175,7 +245,7 @@
                 elif path.path == '/api/audit':
                     body, ctype, code = audit(), 'application/json', 200
                 else:
-                    self.send_error(404, 'NOT_FOUND'); return
+                    self.fail(404, 'NOT_FOUND'); return
                 if not isinstance(body, bytes):
                     body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
                 self.send_response(code)
@@ -184,14 +254,23 @@
                 self.send_header('X-Content-Type-Options', 'nosniff')
                 self.send_header('Content-Security-Policy',
                                  "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
-                                 "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'")
+                                 "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
+                                 "frame-ancestors 'none'")
                 self.send_header('Content-Length', str(len(body)))
                 self.end_headers()
                 self.wfile.write(body)
             except ValueError as exc:
-                self.send_error(404, str(exc))
+                if str(exc) == 'RUN_NOT_FOUND':
+                    self.fail(404, 'RUN_NOT_FOUND')
+                else:
+                    print(f'[DASHBOARD][DASHBOARD_INTERNAL_ERROR] {exc!r}', file=sys.stderr)
+                    self.fail(500, 'DASHBOARD_INTERNAL_ERROR')
             except OSError as exc:
-                self.send_error(500, f'DASHBOARD_IO_ERROR: {exc}')
+                print(f'[DASHBOARD][DASHBOARD_IO_ERROR] {exc!r}', file=sys.stderr)
+                self.fail(500, 'DASHBOARD_IO_ERROR')
+            except Exception as exc:  # never drop a connection without a response
+                print(f'[DASHBOARD][DASHBOARD_INTERNAL_ERROR] {exc!r}', file=sys.stderr)
+                self.fail(500, 'DASHBOARD_INTERNAL_ERROR')
 
         def do_POST(self):
             self.send_error(405, 'READ_ONLY_DASHBOARD')
@@ -199,7 +278,11 @@
         def log_message(self, fmt, *args):
             print('[DASHBOARD]', fmt % args)
 
-    httpd = ThreadingHTTPServer(('127.0.0.1', port), Handler)
+    try:
+        httpd = ThreadingHTTPServer(('127.0.0.1', port), Handler)
+    except OSError as exc:
+        print(f'[ERR][DASHBOARD_PORT_UNAVAILABLE] 127.0.0.1:{port} 无法监听：{exc.strerror}', file=sys.stderr)
+        raise SystemExit(4)
     print(f'[OK] TinyAerialVLN V7.1 看板: http://127.0.0.1:{port}/')
     print('[INFO] 本地只读，不执行训练、仿真或固件改动。Ctrl+C 退出。')
     try:
diff -ruN a/tools/launcher.py b/tools/launcher.py
--- a/tools/launcher.py
+++ b/tools/launcher.py
@@ -5,12 +5,26 @@
 from datetime import datetime
 import os
 from pathlib import Path
+import re
 import shutil
 import subprocess
 import sys
 
 ROOT = Path(__file__).resolve().parents[1]
 RUNS = ROOT / 'artifacts' / 'runs'
+RUN_ID = re.compile(r'^[A-Za-z0-9_]{1,70}$')  # identical to tools/dashboard.py
+# Exit codes shared with tools/sim_preflight.py
+EXIT_USAGE, EXIT_NOT_IMPLEMENTED, EXIT_INPUT, EXIT_PROCESS, EXIT_NOT_BUILT = 2, 3, 4, 5, 6
+
+
+def probe(cmd: list[str]) -> str:
+    """Run a version probe; FOUND in PATH is not evidence that the tool works."""
+    try:
+        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
+    except (OSError, subprocess.TimeoutExpired) as exc:
+        return f'probe failed: {type(exc).__name__}'
+    lines = (out.stdout or out.stderr).strip().splitlines()
+    return (lines[-1] if lines else '(no output)') + ('' if out.returncode == 0 else f' [exit={out.returncode}]')
 
 
 def doctor() -> int:
@@ -20,9 +34,15 @@
     missing = []
     for cmd, optional in commands:
         found = shutil.which(cmd)
-        print(f'{"[OK]" if found else "[WARN]"} {cmd}: {found or "未安装/不在 PATH"}')
+        print(f'{"[FOUND]" if found else "[WARN]"} {cmd}: {found or "未安装/不在 PATH"}')
         if not found and not optional:
             missing.append(cmd)
+    probes = {'nvcc': ['nvcc', '--version'], 'gz': ['gz', 'sim', '--versions'],
+              'nvidia-smi': ['nvidia-smi', '--query-gpu=name,driver_version,compute_cap', '--format=csv,noheader']}
+    for cmd, argv in probes.items():
+        if shutil.which(cmd):
+            print(f'[PROBE] {cmd}: {probe(argv)}')
+    print('[INFO] FOUND 只表示可执行文件在 PATH 中；RTX 5090 (sm_120) 需要 CUDA ≥ 12.8，请核对 PROBE 行。')
     for key in ['artifacts', 'tools', 'code', 'assets']:
         print(f'{"[OK]" if (ROOT / key).exists() else "[ERR]"} project/{key}')
     if not (ROOT / 'config' / 'experiment.yaml').exists():
@@ -33,33 +53,49 @@
     return 0 if not missing else 2
 
 
-def gated_binary(name: str, run_id: str | None) -> int:
+def gated_binary(name: str, run_id: str | None, mode: str = 'train') -> int:
+    if mode == 'train':
+        run_id = run_id or datetime.now().strftime('run_%Y%m%d_%H%M%S')
+    elif not run_id:
+        print('[ERR][RUN_ID_REQUIRED] eval 必须用 --run-id 指定一个已有的训练 run。', file=sys.stderr)
+        return EXIT_INPUT
+    if not RUN_ID.fullmatch(run_id):  # validate input before probing the environment
+        print('[ERR][RUN_ID_INVALID] run ID 只能用 ASCII 字母、数字和下划线，1–70 字符（与看板一致）。', file=sys.stderr)
+        return EXIT_INPUT
     exe = ROOT / 'build' / name
     cfg = ROOT / 'config' / 'experiment.yaml'
     if not exe.is_file() or not os.access(exe, os.X_OK):
         print(f'[ERR][BINARY_NOT_BUILT] {exe} 不存在或不可执行。先按开发门禁实现并编译。', file=sys.stderr)
-        return 3
+        return EXIT_NOT_BUILT
     if not cfg.is_file():
         print(f'[ERR][CONFIG_NOT_READY] 缺少 {cfg}，不得使用 .planned 占位文件训练。', file=sys.stderr)
-        return 4
-    run_id = run_id or datetime.now().strftime('run_%Y%m%d_%H%M%S')
-    if not run_id.replace('_', '').isalnum() or len(run_id) > 70:
-        print('[ERR][RUN_ID_INVALID] run ID 只能用字母、数字和下划线，≤70字。', file=sys.stderr)
-        return 4
-    out = RUNS / run_id
-    if out.exists():
-        print(f'[ERR][RUN_EXISTS] {out} 已存在，禁止覆盖旧实验。', file=sys.stderr)
-        return 4
+        return EXIT_INPUT
+    run_dir = RUNS / run_id
+    extra: list[str] = []
+    if mode == 'train':
+        if run_dir.exists():
+            print(f'[ERR][RUN_EXISTS] {run_dir} 已存在，禁止覆盖旧实验。', file=sys.stderr)
+            return EXIT_INPUT
+        out = run_dir
+    else:  # eval never overwrites the training run: it writes a sub-directory per evaluation
+        if not run_dir.is_dir():
+            print(f'[ERR][RUN_NOT_FOUND] {run_dir} 不存在，无法评测。', file=sys.stderr)
+            return EXIT_INPUT
+        out = run_dir / datetime.now().strftime('eval_%Y%m%d_%H%M%S')
+        if out.exists():
+            print(f'[ERR][RUN_EXISTS] {out} 已存在，请稍后重试。', file=sys.stderr)
+            return EXIT_INPUT
+        extra = ['--run-dir', str(run_dir)]
     out.mkdir(parents=True)
-    print(f'[INFO] 启动 {name}; run_id={run_id}; Ctrl+C 请求终止。')
+    print(f'[INFO] 启动 {name}; run_id={run_id}; output={out}; Ctrl+C 请求终止。')
     try:
-        return subprocess.call([str(exe), '--config', str(cfg), '--output', str(out)], cwd=ROOT)
+        return subprocess.call([str(exe), '--config', str(cfg), *extra, '--output', str(out)], cwd=ROOT)
     except KeyboardInterrupt:
         print('\n[WARN][USER_INTERRUPT] 请求中止；检查是否生成完整 checkpoint。', file=sys.stderr)
         return 130
     except OSError as exc:
         print(f'[ERR][PROCESS_START_FAILED] {exc}', file=sys.stderr)
-        return 5
+        return EXIT_PROCESS
 
 
 def menu() -> int:
@@ -74,13 +110,15 @@
             '2': lambda: subprocess.call([sys.executable, str(ROOT / 'tools/scene_tool.py')], cwd=ROOT),
             '3': lambda: subprocess.call([sys.executable, str(ROOT / 'tools/dashboard.py')], cwd=ROOT),
             '4': lambda: gated_binary('tinyvln_train', None),
-            '5': lambda: gated_binary('tinyvln_eval_bridge', None)}.get(
+            '5': lambda: gated_binary('tinyvln_eval_bridge', input('评测哪个 run ID: ').strip(), 'eval')}.get(
                 selection, lambda: print('[ERR][MENU_OPTION_INVALID] 请输入 0..5。') or 2)()
 
 
 def main() -> int:
     p = argparse.ArgumentParser(description='TinyAerialVLN CLI (no fake training)')
-    p.add_argument('command', nargs='?', choices=['doctor', 'scene', 'dashboard', 'ui', 'train', 'eval'], default=None)
+    p.add_argument('command', nargs='?', default=None,
+                   choices=['doctor', 'scene', 'dashboard', 'ui', 'train', 'eval', 'board', 'uart'])
+    p.add_argument('rest', nargs='*', help='board/uart 的子命令（尚未实现）')
     p.add_argument('--run-id', default=None)
     p.add_argument('--port', type=int, default=8765)
     p.add_argument('--preset', choices=['forest', 'corridor', 'columns', 'mixed'], default='forest')
@@ -89,6 +127,12 @@
     args = p.parse_args()
     if args.command is None:
         return menu()
+    if args.command in ('board', 'uart'):  # listed in plan §11, gated by EDGE/HIL
+        print(f'[ERR][FEATURE_NOT_IMPLEMENTED] ./run.sh {args.command} {" ".join(args.rest)} 需在 EDGE/HIL 门禁实现后提供。',
+              file=sys.stderr)
+        return EXIT_NOT_IMPLEMENTED
+    if args.rest:
+        p.error(f'unexpected arguments: {" ".join(args.rest)}')
     if args.command == 'doctor':
         return doctor()
     if args.command in ('dashboard', 'ui'):
@@ -98,7 +142,7 @@
                                '--seed', str(args.seed), '--count', str(args.count)], cwd=ROOT)
     if args.command == 'train':
         return gated_binary('tinyvln_train', args.run_id)
-    return gated_binary('tinyvln_eval_bridge', args.run_id)
+    return gated_binary('tinyvln_eval_bridge', args.run_id, 'eval')
 
 
 if __name__ == '__main__':
diff -ruN a/tools/sim_preflight.py b/tools/sim_preflight.py
--- a/tools/sim_preflight.py
+++ b/tools/sim_preflight.py
@@ -11,6 +11,10 @@
 import sys
 import xml.etree.ElementTree as ET
 
+MOTOR_PLUGIN_FILENAMES = {'gz-sim-multicopter-motor-model-system', 'libgz-sim-multicopter-motor-model-system.so',
+                          'ignition-gazebo-multicopter-motor-model-system'}
+MOTOR_PLUGIN_NAMES = {'gz::sim::systems::MulticopterMotorModel', 'ignition::gazebo::systems::MulticopterMotorModel'}
+
 
 def main() -> int:
     parser = argparse.ArgumentParser(description="TinyAerialVLN Gazebo bring-up diagnostic")
@@ -44,9 +48,13 @@
     if doc.tag != 'sdf' or doc.find('world') is None:
         print("[ERR][SDF_WORLD_INVALID] Expected an SDF root containing a world.", file=sys.stderr)
         return 4
-    # Only a smoke launcher, but fail if given the old V6.1 static geometry world.
-    if not any('multicopter' in (p.get('filename','').lower()) for p in doc.findall('.//plugin')):
-        print("[ERR][GZ_MOTOR_MODEL_MISSING] SDF has no multicopter system plugin. Static scene != flight dynamics.", file=sys.stderr)
+    # Count real rotor-force plugins; a velocity controller or a static scene is not motor dynamics.
+    motors = [p for p in doc.iter('plugin')
+              if p.get('filename', '') in MOTOR_PLUGIN_FILENAMES or p.get('name', '') in MOTOR_PLUGIN_NAMES]
+    if len(motors) < 4:
+        print(f"[ERR][GZ_MOTOR_MODEL_MISSING] found {len(motors)} MulticopterMotorModel plugin(s), need >= 4. "
+              "Static scene or velocity-control plugin != flight dynamics. If the plugins live inside an "
+              "<include>d model file, pass an expanded world.", file=sys.stderr)
         return 4
     if not gz:
         print("[ERR][GZ_NOT_INSTALLED] Install Gazebo Harmonic and retry.", file=sys.stderr)
````

---

## 附录 C. 复现脚本全文 `review_v2_repro.py`

````python
#!/usr/bin/env python3
"""TinyAerialVLN V7.1 评审（第二版）复现脚本 —— 只用 Python 标准库（3.9+）。

用法:  python3 review_v2_repro.py /path/to/TinyAerialVLN_V7_1_工程
脚本把工程复制到临时目录（目录名保留“工程”二字，用于复现 B04），绝不修改原目录。
每条结果为：存在 / 已修复 / 跳过。对原始 V7.1 与打补丁后的工程各跑一次即可对照。
"""
import contextlib
import http.client
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

GOOD = b'{"event":"train_step","step":1,"total_steps":2,"loss":0.5}\n'
PLAN_CONTRACT = [  # V7_1_RESEARCH_PLAN.md §12.3 的四行示例（时间戳占位已替换）
    '{"kind":"train","run_id":"bc_001","step":1200,"loss_total":0.73,"lr":0.0002,"timestamp_utc":"2026-10-28T00:00:00Z"}',
    '{"kind":"sim_step","episode_id":"s1_042","step":34,"sim_time_ns":1700000000,"camera_ts_ns":1666000000,'
    '"policy_action":"turn_right","collision":false}',
    '{"kind":"eval_episode","episode_id":"s1_042","success":false,"collision":true,"stop_reason":"collision",'
    '"path_length_m":12.4}',
    '{"kind":"hardware","seq":125,"camera_to_uart_ms":62.3,"ack_ok":true,"ttl_expired":false}',
]
FAKE_BINARY = '''#!/usr/bin/env bash
out=""; while [[ $# -gt 0 ]]; do [[ "$1" == "--output" ]] && out="$2"; shift; done
echo '{"event":"train_step","step":1,"total_steps":1,"loss":1.0}' > "$out/metrics.jsonl"
'''
MONITOR_PROGRAM = r'''
#include "tinyvln/training_monitor.hpp"
#include <locale>
struct Comma : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};
int main(int, char** argv) {
  std::locale::global(std::locale(std::locale::classic(), new Comma));
  tinyvln::TrainingMonitor m(argv[1]);
  m.Step(26000, 52000, 0.75, 0.5, 2e-4);
  std::cerr << "\nprobe=" << 0.37 << "\n";
  try { m.Step(26001, 52000, 0.0 / 0.0, 0.5, 2e-4); } catch (const std::exception&) { std::cerr << "THROWN\n"; }
}
'''
RESULTS = []


def report(code, title, present, note=''):
    state = '存在' if present else '已修复'
    RESULTS.append(state)
    print(f'[{code:5s}] {state}  {title}' + (f'  —— {note}' if note else ''))


def skip(code, title, why):
    RESULTS.append('跳过')
    print(f'[{code:5s}] 跳过  {title}  —— {why}')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_port(port, seconds=8.0):
    end = time.time() + seconds
    while time.time() < end:
        try:
            socket.create_connection(('127.0.0.1', port), timeout=0.2).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


def http_get(port, path, host=None):
    """Return (status, body); status is None when the server closes without any response."""
    try:
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        conn.request('GET', path, headers={'Host': host} if host else {})
        resp = conn.getresponse()
        return resp.status, resp.read()
    except (OSError, http.client.HTTPException) as exc:
        return None, repr(exc).encode()


def run(args, cwd, env=None, timeout=120):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(root, name, payload):
    d = root / 'artifacts' / 'runs' / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'metrics.jsonl').write_bytes(payload)


def check_scenes(root):
    st = load_module('scene_tool_review', root / 'tools' / 'scene_tool.py')
    clearance = min(min(abs(o['y']) - o['r'] for o in obs)
                    for preset in st.PRESETS for seed in range(300)
                    for obs in [st.create_obstacles(preset, seed, 200)] if obs)
    report('P0-2', '起点→终点直线在所有 preset/seed 下都畅通', clearance > 0.3, f'最小净空 {clearance:.2f} m')
    same = st.make_sdf(st.create_obstacles('forest', 7, 40), 7) == st.make_sdf(st.create_obstacles('mixed', 7, 40), 7)
    report('P0-2b', 'mixed preset 与 forest 输出完全相同', same)
    ignored = [line.strip() for line in (root / '.gitignore').read_text().splitlines()]
    report('B16c', '.gitignore 未忽略 build/', not any(x in ('build/', 'build', '/build/') for x in ignored))


def check_dashboard_http(root):
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(root / 'tools' / 'dashboard.py'), '--port', str(port)],
                            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_port(port):
            skip('B01-7', '看板 HTTP 检查', '看板没有启动')
            return
        status, _ = http_get(port, '/api/audit', host=f'evil.example:{port}')
        report('B05', '伪造 Host 头仍可读取 /api/audit（DNS rebinding）', status == 200, f'HTTP {status}')

        write_run(root, 'r_nested', GOOD + b'{"event":"train_step","extra":{"grad_norm":NaN}}\n')
        status, _ = http_get(port, '/api/metrics?run=r_nested')
        report('B01', '一行嵌套 NaN 让整个 run 的指标接口失效', status != 200, f'HTTP {status}')

        write_run(root, 'r_deep', GOOD + b'{"event":"x","v":' + b'[' * 30000 + b']' * 30000 + b'}\n')
        write_run(root, 'r_bigint', GOOD + b'{"event":"x","v":' + b'9' * 5000 + b'}\n')
        s1, _ = http_get(port, '/api/metrics?run=r_deep')
        s2, _ = http_get(port, '/api/metrics?run=r_bigint')
        report('B02', '深层嵌套 / 超长整数行让接口断连或 404', s1 != 200 or s2 != 200,
               f'deep→{s1 or "无响应"}, bigint→{s2 or "无响应"}')

        write_run(root, 'plan_contract', ('\n'.join(PLAN_CONTRACT) + '\n').encode())
        status, body = http_get(port, '/api/metrics?run=plan_contract')
        shown = len(json.loads(body)['events']) if status == 200 else 0
        report('P1-2', '方案 §12.3 的 JSONL 契约示例被看板全部拒收', shown == 0,
               f'显示 {shown}/4 条（补丁不改契约，需你决定以哪份为准）')

        scenes = root / 'artifacts' / 'scenes'
        scenes.mkdir(parents=True, exist_ok=True)
        bad = scenes / 'zz_listroot.json'
        bad.write_text('[]')
        status, _ = http_get(port, '/api/status')
        report('B03', '根为 [] 的场景 manifest 让 /api/status 断连', status != 200, f'HTTP {status or "无响应"}')
        bad.unlink()

        backup = root / 'artifacts' / '_manifest_backup'
        backup.mkdir(exist_ok=True)
        for f in scenes.glob('*.json'):
            f.rename(backup / f.name)
        (scenes / 'a.json').write_text('{broken')
        (scenes / 'b.json').write_text('{broken')
        status, body = http_get(port, '/api/status')
        state = {c['key']: c for c in json.loads(body)['checks']}['scenes']['state'] if status == 200 else None
        report('B06', 'manifest 全部损坏时 scenes 仍为 ready', state == 'ready', f'state={state}')
        for f in ('a.json', 'b.json'):
            (scenes / f).unlink()
        for f in backup.glob('*.json'):
            f.rename(scenes / f.name)
        backup.rmdir()

        for i in range(85):
            write_run(root, f'bulk_{i:03}', GOOD)
        actual = sum(1 for p in (root / 'artifacts' / 'runs').iterdir() if (p / 'metrics.jsonl').is_file())
        status, body = http_get(port, '/api/status')
        count = json.loads(body)['run_count'] if status == 200 else None
        report('B07', '超过 80 个 run 时计数与列表被静默截断', count != actual, f'run_count={count}，实际 {actual}')
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(root / 'artifacts' / 'runs', ignore_errors=True)


def check_io_error_status_line(root):
    if all(ord(ch) < 256 for ch in str(root)):
        skip('B04', 'OSError 消息含中文路径时无响应', '临时目录名不含非 latin-1 字符')
        return
    dash = load_module('dashboard_review', root / 'tools' / 'dashboard.py')

    def boom(_name):
        raise PermissionError(13, 'Permission denied', str(dash.ROOT / 'artifacts' / 'runs' / 'x' / 'metrics.jsonl'))
    dash.metrics_for = boom
    port = free_port()
    sink = io.StringIO()  # keep the expected server traceback out of the report
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        threading.Thread(target=dash.serve, args=(port,), daemon=True).start()
        wait_port(port)
        status, _ = http_get(port, '/api/metrics?run=x')
        time.sleep(0.2)
    report('B04', 'OSError 消息含中文路径时连接无响应（文档声称 500）', status != 500, f'HTTP {status or "无响应"}')


def check_launcher(root):
    env = dict(os.environ)
    fakebin = root.parent / 'fakebin'
    fakebin.mkdir(exist_ok=True)
    (fakebin / 'nvcc').write_text('#!/bin/sh\nexit 1\n')
    (fakebin / 'nvcc').chmod(0o755)
    env['PATH'] = f'{fakebin}{os.pathsep}{env.get("PATH", "")}'
    r = run([str(root / 'run.sh'), 'doctor'], root, env=env)
    report('B11', 'doctor 只凭 which 标 [OK]（坏掉的 nvcc 也算 OK）', '[OK] nvcc' in r.stdout, f'exit={r.returncode}')

    r = run([str(root / 'run.sh'), 'board', 'doctor'], root)
    report('B10', '方案 §11 的 board/uart 子命令不报 FEATURE_NOT_IMPLEMENTED',
           'FEATURE_NOT_IMPLEMENTED' not in r.stderr, f'exit={r.returncode}')

    build, config = root / 'build', root / 'config'
    build.mkdir(exist_ok=True)
    for name in ('tinyvln_train', 'tinyvln_eval_bridge'):
        (build / name).write_text(FAKE_BINARY)
        (build / name).chmod(0o755)
    (config / 'experiment.yaml').write_text('seed: 1\n')

    r = run([str(root / 'run.sh'), 'train', '--run-id', '实验一'], root)
    report('B08', 'launcher 接受看板永远不显示的非 ASCII run ID', r.returncode == 0, f'exit={r.returncode}')

    run([str(root / 'run.sh'), 'train', '--run-id', 'bc_x'], root)
    r = run([str(root / 'run.sh'), 'eval', '--run-id', 'bc_x'], root)
    report('B09', '按方案 §11 先 train 再 eval 同一 run-id 会失败', r.returncode != 0,
           f'eval exit={r.returncode} {r.stderr.strip()[:40]}')

    r = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'code/tests', '-p', 'test_infrastructure.py',
             '-k', 'test_refuses_nonexistent_training_binary'], root)
    launched = (root / 'artifacts' / 'runs' / 'forbidden_test').exists()
    report('B16b', '训练器编译后，现有测试会真的启动训练并失败', launched or 'FAILED' in r.stderr,
           '已创建 runs/forbidden_test' if launched else '')
    shutil.rmtree(build, ignore_errors=True)
    (config / 'experiment.yaml').unlink()


def check_sim_preflight(root):
    world = root.parent / 'velocity_only.sdf'
    world.write_text('<sdf version="1.9"><world name="w"><plugin filename="gz-sim-multicopter-control-system" '
                     'name="gz::sim::systems::MulticopterVelocityControl"/></world></sdf>')
    r = run([str(root / 'run.sh'), 'sim', 'smoke', '--world', str(world), '--dry-run'], root)
    report('B15', '只有速度控制插件的 world 通过“电机模型”检查', 'GZ_MOTOR_MODEL_MISSING' not in r.stderr,
           f'exit={r.returncode}')


def check_monitor_header(root):
    if not shutil.which('g++'):
        skip('B12-14', 'training_monitor.hpp', '未找到 g++')
        return
    work = root.parent / 'hpp'
    work.mkdir(exist_ok=True)
    (work / 't.cpp').write_text(MONITOR_PROGRAM)
    build = run(['g++', '-std=c++17', '-I', str(root / 'code' / 'include'), str(work / 't.cpp'),
                 '-o', str(work / 't')], root)
    if build.returncode:
        skip('B12-14', 'training_monitor.hpp', '编译失败：' + build.stderr[:80])
        return
    log = work / 'm.jsonl'
    r = run([str(work / 't'), str(log)], root)
    lines = log.read_text().splitlines()
    valid = True
    try:
        json.loads(lines[0])
    except ValueError:
        valid = False
    report('B13', '全局 locale 带千分位/逗号小数时写出非法 JSON', not valid, lines[0][:60])
    report('B12', 'Step() 之后 std::cerr 被改成 fixed+precision(0)', 'probe=0.37' not in r.stderr,
           r.stderr.split('probe=')[-1].split('\n')[0] if 'probe=' in r.stderr else '')
    report('B14', 'NaN 时先抛异常、不写错误事件', not any('TRAIN_NONFINITE_METRIC' in line for line in lines))


def check_fresh_clone(source):
    if not shutil.which('git'):
        skip('B16a', '全新 git clone 后测试失败', '未找到 git')
        return
    base = Path(tempfile.mkdtemp(prefix='tinyvln_clone_'))
    repo = base / 'repo'
    shutil.copytree(source, repo, ignore=shutil.ignore_patterns('__pycache__', 'build', 'runs'))
    git = ['git', '-c', 'user.name=review', '-c', 'user.email=review@example.invalid', '-c', 'init.defaultBranch=main']
    run(git + ['init', '-q'], repo)
    run(git + ['add', '-A'], repo)
    run(git + ['commit', '-q', '-m', 'snapshot'], repo)
    run(['git', 'clone', '-q', str(repo), str(base / 'clone')], base)
    clone = base / 'clone'
    (clone / 'run.sh').chmod(0o755)
    a = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'code/tests', '-p', 'test_*.py'], clone)
    b = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'sim/tests', '-p', 'test_*.py'], clone)
    broken = 'FAILED' in a.stderr or 'FAILED' in b.stderr
    report('B16a', '全新 git clone（artifacts/ 被忽略）后测试失败', broken,
           f"code/tests: {a.stderr.strip().splitlines()[-1]}; sim/tests: {b.stderr.strip().splitlines()[-1]}")
    shutil.rmtree(base, ignore_errors=True)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    source = Path(sys.argv[1]).resolve()
    if not (source / 'tools' / 'dashboard.py').is_file():
        print(f'[ERR] {source} 不是 TinyAerialVLN 工程根目录')
        return 2
    tmp = Path(tempfile.mkdtemp(prefix='tinyvln_review_'))
    root = tmp / 'TinyAerialVLN_V7_1_工程'
    shutil.copytree(source, root, ignore=shutil.ignore_patterns('__pycache__', 'build', 'runs'))
    (root / 'run.sh').chmod(0o755)
    print(f'被测工程: {source}\n临时副本: {root}\n')
    check_scenes(root)
    check_dashboard_http(root)
    check_io_error_status_line(root)
    check_launcher(root)
    check_sim_preflight(root)
    check_monitor_header(root)
    check_fresh_clone(source)
    print(f"\n汇总：存在 {RESULTS.count('存在')}，已修复 {RESULTS.count('已修复')}，跳过 {RESULTS.count('跳过')}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
````

