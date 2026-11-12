# 第二轮交叉审核报告 N：部署链与真机接口（板端推理路径、串口动作协议、量化与导出）

- 审核基准：主仓库 `/root/claude-code-backup-20261028/code/uav_rgb` 分支 `main`，开工提交 `ada0ca5`（2026-11-12 08:00 CST），收尾复核至 `43fe70a`（本角度涉及的 `src/`、`tools/uart_action.py`、`docs/uart_action.md`、`sim/tools/student.py`、`sim/tools/nav_cond.py` 在此期间无变更；另外只读核对了分支 `p12-e2e-p0` 的 `proto/`、分支 `p12-e2e-1` 的 `dc8e5ff` 与固件原件 `CF-Drone` 提交 `e7f507f`）。
- 起止时刻：2026-11-12 08:00 CST 至 2026-11-12 08:56 CST。
- 草稿与复现产物：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-n/`（含 `uart_probe.py`、`uart_probe.out`、`quant_probe.py`、`quant_probe.out`）。只读审查：未修改任何被 git 跟踪的文件，未起 Gazebo，未训练，未占 GPU。

## 1. 结论

从训练好的 `ckpt.bin` 到琛光 E1 MINI（ESP32 跑 CF-Drone 固件 + 机载 LicheeRV Nano SG2002 int8 NPU）真机闭环飞行，仓库目前只有 PC 侧 CUDA 常驻推理进程 `uavrgb_infer` 与单向占位串口脚本 `tools/uart_action.py`，导出、量化、板端前处理、板端推理、双向串口与固件解析均为空白（`$D/real-drone/` 仅有一份要求 PX4 ULog 的旧 `README.md`）。本角度下最要紧的 3 件事：

1. **现有串口发送骨架在常驻管道模式下存在吞动作、屏蔽飞控失联保护、心跳覆盖动作三处实测缺陷，且单向无回执协议支撑不了板端条件码**（`N-1`、`N-6`）：`tools/uart_action.py` 用 `select.select` 搭配带缓冲的 `sys.stdin.readline()`，上游一次写入两行动作时第 2 行被用户态缓冲吞没并延迟到下一周期才发出；上游推理进程崩溃关闭管道（EOF）后，脚本继续无限发心跳，使飞控侧 300 ms TTL 失联保护失效；动作帧发出后同轮循环立即追加 `action=0` 的心跳帧。同时脚本以 `O_WRONLY` 只写打开串口，既拿不到「已执行动作回执」（算动作计数历史码必需），也拿不到「飞控偏航四元数与起飞边沿」（算角度分档航向码必需）。
2. **SG2002 对称 int8 量化对主干卷积温和，但直接量化 FiLM 乘法分支与大动态范围输出层会让转向帧翻转 4.4%–11.4%；把图切在 `p3`（NPU 跑三层卷积 int8，CPU 跑 FiLM + 全连接 FP32）可根治**（`N-2`）：实测 `cal_swa_s1_k10.bin` 的 `SAFE` 类 logit 低至 `-58.47`（`e2e-p0.md` 在 2,000 帧上测得最大绝对值 `65.4`），输出层若按对称 int8 量化，步长 `0.4151` nat/LSB，预测转向帧翻转率从 `0.00%` 升至 `4.40%`；带 FiLM 的航向码权重 `a2-code-dagger-s1` 若将 FiLM 逐通道乘法一并放入对称 int8，预测转向帧翻转达 `8.59%`（含 int8 输出层达 `11.36%`）。而 `FiLM + 19,200→4` 全连接仅 `76,800` MAC，留在 SG2002 的 1 GHz C906 CPU 上用 FP32 算耗时不足 `0.2` ms，预测转向帧翻转率即压回 `1.52%`（单帧模型 `0.00%`）。
3. **条件字节语义不进 checkpoint 与推理握手行，客户端默认回落与 3DGS 硬编码会把航向码模型喂成死码 `1`，实测翻转 14.0%–21.9% 的动作**（`N-3`、`N-5`）：`prev_dim=4` 无法区分「上一步动作」与「三值航向码」，`sim/tools/student.py` 缺省回落到模型上一步输出，`sim/3dgs/collect_3dgs.py` 硬编码喂上一步执行的动作（`FORWARD_SLOW=1`）。航向码训练集里 `code=1` 样本数为 `0`，对应未受监督的随机初值列；在 `code=0` 帧上喂入 `1` 会使 `a1-code-demo-s1` 翻转 `13.95%`、`a2-code-dagger-s1` 翻转 `21.88%`（226 个转向预测被全部压成前进）。板端没有训练配置与 `.uavr` 数据集，现有 `nav_cond.check_student_cond` 在板端恒退化为 `unverified`。

---

## 2. 发现清单（按严重程度排序）

### N-1：`tools/uart_action.py` 常驻管道吞没排队动作、上游崩溃后持续发心跳屏蔽飞控 TTL、同轮循环心跳覆盖动作

- **位置**：`tools/uart_action.py:260`、`:331-357`（提交 `c54070d`）；`tools/test_uart_action.py:163-174`（提交 `c54070d`）；`docs/uart_action.md:70-75`、`:89-96`（提交 `c54070d`）。
- **现象**：
  1. `select.select([sys.stdin], ...)` 检查底层文件描述符可读后，第 339 行调用带用户态缓冲的 `sys.stdin.readline()`。常驻推理进程向管道连续写入 `b"1\n3\n"`（不关闭管道）时，Python `TextIOWrapper` 一次 `read` 把两行都吸进内部缓冲区，只返回 `"1\n"`；下一轮循环底层 fd 已无未读字节，`select` 一直阻塞到心跳超时，`"3\n"` 被吞在缓冲区里发不出去，直到 0.35 s 后上游再写 `"2\n"` 时才吐出过期的动作 `3`，而新动作 `2` 继续被吞（`uart_probe.out` Case 1：前 0.35 s 仅发出 `[(0, 1)]`，0.35 s 后写 `2\n` 吐出 `[(5, 3), (6, 2)]`）。
  2. 第 340–341 行在 `line == ""`（上游进程崩溃或退出关闭 stdout）时仅设 `stdin_open = False`，主循环转入 `time.sleep(timeout)` 并继续以 10 Hz 发送心跳帧（`uart_probe.out` Case 2：关闭 stdin 0.55 s 后进程仍存活，继续发出 5 帧心跳）。
  3. 第 343–356 行处理完 stdin 动作帧后没有 `continue` 也没有推迟 `next_tick`，紧接着执行 `if now >= next_tick:`，在同一轮循环内微秒级背靠背再发一帧 `action=0, heartbeat=1` 的心跳帧（`uart_probe.out` Case 3：`seq=0, action=1, hb=0` 后立即紧跟 `seq=1, action=0, hb=1`）。
  4. 上游若因卡顿写出半行 `b"1"`（未带 `\n`），`sys.stdin.readline()` 同步死等换行符，整个主循环停摆，连心跳帧也断流（`uart_probe.out` Case 4：0.45 s 内发出 0 帧）。
- **为什么是缺陷及影响**：不影响当前仿真闭环数字（仿真走 `cf_sitl`，不经过 `uart_action.py`）。一旦上台架或真机用管道把常驻推理器接到 `uart_action.py`：(1) 缓冲吞行会让飞机执行晚一拍的陈旧转向指令；(2) 推理进程段错误或 OOM 退出后，`uart_action.py` 替死掉的推理器继续喂心跳，飞控侧依赖 300 ms 无帧超时的 TTL 失联保护永远不会触发；(3) 心跳帧编码了 `action=0` 且 `ttl_ms=300`，若飞控固件按帧更新动作则前一条非零动作在几百微秒内被清零为悬停，若飞控固件忽略心跳帧的 `action` 字段而只续期上一动作，则无法实现仿真新协议「动作有效 0.8 s 后回 `SAFE_NEUTRAL` 刹停 0.1 s」的时序（第一轮 G5）。已有单测 `test_cli_stdin_actions` 用 `subprocess.run(..., input="1\n3\n")` 一次性灌入并立即触发 EOF，掩盖了常驻流下的上述问题。
- **怎样复现或验证**：运行草稿目录探针 `nice -n 15 /root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python /root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-n/uart_probe.py`（纯 `--dry-run` 管道，耗时 < 2 s），输出见 `uart_probe.out`。
- **建议的最小修法**：(1) 弃用 `sys.stdin.readline()`，改对 `os.set_blocking(0, False)` 的 fd `0` 用 `os.read(0, 4096)` 自维护 `bytearray` 行缓冲，按 `\n` 切出所有完整行逐一处理，半行留在缓冲中不阻塞心跳；(2) `os.read` 返回空字节串（EOF）时默认立即退出（或立刻发一帧 `SAFE_NEUTRAL` 后退出），把链路超时交还飞控 TTL；另加 `--upstream-timeout-ms`（例如 1200 ms 无新动作即主动转发 `SAFE_NEUTRAL` 告警）；(3) 协议语义拆清：心跳帧只保活链路、不携带动作码（或携带当前正在执行的动作及其剩余有效毫秒数 `rem_ms`）。
- **状态**：已复现。
- **归属建议**：**现在就能在仿真一侧修的**（修改仓库主机侧文件 `tools/uart_action.py`、`tools/test_uart_action.py`、`config/uart_protocol.example.json`，无需硬件，建议单独立一张部署前准备小卡）。

### N-2：SG2002 对称 int8 下输出层大动态范围与 FiLM 逐通道乘法引发 4.4%–11.4% 转向翻转；`p3` 切图混合执行可降至 0%–1.5%

- **位置**：`src/model.cpp:63-80`、`:188-243`（提交 `ed25f52`）；`src/model_dev_film.cu:47-56`（提交 `ed25f52`）；`src/infer.cpp:93-102`（提交 `ed25f52`）。实测权重：`$D/exp-swa-calib-20261110/ckpt/cal_swa_s1_k10.bin`、`$D/exp-e2e-heading-code-20261112/runs/a1-code-demo-s1/ckpt_best.bin`、`a2-code-dagger-s1/ckpt_best.bin`。
- **现象**：模型无归一化层，含 `19,200 → 4` 全连接与 `p3' = (1+γ)·p3 + β` 逐通道调制。用 256 帧训练集校准、在评估集上做 SG2002/CV181x TPU 对称有符号 int8（`[-127, 127]`）伪量化试验（`quant_probe.out`）：
  1. **输出层大动态范围挤压有效动作分辨率**：`cal_swa_s1_k10.bin` 在 `val+test` 1,458 帧上，`p1/p2/p3` 最大值为 `1.316 / 2.297 / 5.270`，19,200 维 `p3` 的 L1 范数中位数 `506.0`、最大 `1990.7`；因为训练时 `SAFE(0)` 类权重高达 `36.63` 且极少正样本，`SAFE` 的 logit 落在 `[-58.47, +2.28]`（`e2e-p0.md` 在 2,000 帧上测得最大绝对值 `65.4`），而真正竞争的 `FWD(1)`、`LEFT(2)`、`RIGHT(3)` 分布在 `[-17.14, +14.69]`。若 `.cvimodel` 把输出 logit 也按对称 int8 输出（校准阈值 `52.72`，步长 `0.4151` nat/LSB），评估集中有 `2.81%` 的帧 `|logit_FWD - max_YAW| < 1 LSB`，导致 `1.37%` 的帧发生 top-2 整数平局，FP32 预测为转向的帧有 `4.40%` 被翻转；保持输出层 FP32 时，逐通道权重量化的总翻转率仅 `0.34%`、转向帧翻转 `0.00%`。另外，激活若用 `99.9%` 分位截断（`p3` 从 `5.270` 截到 `2.137`），最大 logit 误差飙升至 `8.477` nat，预测转向帧翻转 `3.14%`。
  2. **FiLM 逐通道乘法在对称 int8 下的双输入量化损耗**：航向码模型 `a2-code-dagger-s1`（val 1,249 帧）中，`1+γ` 范围 `[+0.021, +2.495]`（步长约 `0.020`），与 `p3` 在对称 int8 下相乘后再重缩放进 `19,200` 维内积，误差沿通道累加；即便校准集完整覆盖各条件码且输出保留 FP32，整网进 NPU 时总翻转率仍达 `3.60%`、FP32 预测转向帧翻转率达 `8.59%`（若输出层同为 int8 则升至 `11.36%`，top-2 平局率 `2.48%`）。
  3. **切图验证**：若 NPU 只跑 `c1..p3` 三层卷积池化（`52.53M` MAC，占全网 `99.85%` 算力），输出 `64×15×20 = 19,200` 字节 int8 特征给 CPU，由 CPU 按 `scale_p3` 还原并在 FP32 下执行 `FiLM + fc`（仅 `76,800` MAC），`a2-code-dagger-s1` 的总翻转率从 `3.60%` 降至 `1.04%`，预测转向帧翻转率从 `8.59%` 降至 `1.52%`，`max|Δlogit|` 从 `0.535` 降至 `0.231` nat。
- **为什么是缺陷及影响**：第一轮 `src.md` §E 的 int8 模拟（翻转 0.2%–0.6%）只测了无 FiLM 的单帧模型且未量化输出层，低估了端到端路线（带 FiLM 条件输入）整网塞进 int8 NPU 时在转向关键帧上的精度退化。`8.59%–11.36%` 的转向帧翻转足以改变闭环撞柱率；而把 `FiLM + fc` 留在 CPU 还能顺带绕开 TPU-MLIR 多输入动态分支编译与校准集按条件码配平的复杂度。
- **怎样复现或验证**：运行 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 /root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python /root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-n/quant_probe.py`，结果已落盘于 `quant_probe.out`。
- **建议的最小修法**：(1) 部署切图架构定为「NPU 跑视觉主干到 `p3`（激活必须按 max 校准，禁用分位截断；若整网进 NPU 则 `.cvimodel` 末层必须指定 `--quant_output` 关闭、以 FP32/BF16 吐出 logit），CPU 跑 FP32 的 FiLM 调制与末层小线性头」；(2) 导出工具附带离线 int8 门禁：不仅卡总动作一致率 `≥ 99%`，必须单独卡「转向帧（标签或 FP32 预测为 YAW）动作翻转率 `≤ 2%`」。
- **状态**：已复现（numpy 对称 int8 伪量化；TPU-MLIR 硬件定点移位舍入待工具链核对）。
- **归属建议**：**现在就能在仿真一侧修的**（在仓库 `tools/` 下新建 ONNX 导出与 int8 离线对拍门禁脚本时直接按 `p3` 切图实现，无需等真机或卡 E2E-C1；结构侧降低 `fc` 维度的列结构头已在卡 E2E-P1 探索）。

### N-3：条件字节语义缺失叠加客户端回落与 3DGS 硬编码，将航向码模型喂入未训练死码 `1`，实测翻转 14.0%–21.9% 动作

- **位置**：`src/checkpoint.cpp:174-175`（提交 `ed25f52`）；`src/infer_main.cpp:69-72`（提交 `ed25f52`）；`sim/tools/student.py:106-125`（提交 `ed25f52`）；`sim/3dgs/collect_3dgs.py:167`、`:189-191`（提交 `5387747`）；`sim/tools/nav_cond.py:34-41`、`:108-140`（提交 `d816d2c`）。
- **现象**：W1-9 的航向码复用 `prev_dim=4` 通路，取值集合只有 `{0, 2, 3}`（`nav_cond.py:34-41` 明写「码 1 不用」），而 `checkpoint` v6 与 `uavrgb_infer` 握手行只报 `"prev_action_dim": 4`，不记录该字节是 `prev-action` 还是 `heading-code`。这在三条调用路径上造成错配：
  1. `sim/3dgs/collect_3dgs.py:167`（卡 S1a 刚合入的代码）写死 `self.student_cond = nav_cond.COND_PREV_ACTION if getattr(student, "prev_dim", 0) else None`，并在第 189 行把 `self.last_executed` 当条件喂给任何 `prev_dim=4` 的模型，且 `eval_3dgs.py` 未接入 `check_student_cond`。
  2. `sim/tools/student.py:122` 在调用方未传 `prev_action` 时静默取 `self._last_action`（模型自身上一步输出，80% 以上为 `FORWARD_SLOW=1`）。
  3. Gazebo 侧新增的 `nav_cond.check_student_cond`（`nav_cond.py:94-140`）依赖读取 `ckpt` 旁或 `<ckpt>.json` 指向的 `config.json` 再追查原始 `.uavr` 的 `meta["cond"]` 与整包 `sha256`；到了板端（或数据集移走/训练传相对路径时）直接返回 `unverified` 并放行。
  用刚训出的真实权重核实后果（`quant_probe.out`）：`a1-code-demo-s1` 与 `a2-code-dagger-s1` 训练集里 `code=1` 的样本数均为 `0`，`film.W[:, 1]` 完全停留在未受任何梯度约束的 Xavier 随机初值（L2 范数 `1.4937`，`β` 在 `[-0.26, +0.23]`）。在 val 集原本 `code=0` 的帧上，若按上述回落喂入上一步动作 `1`（`FORWARD_SLOW`）：`a1-code-demo-s1`（559 帧）有 `13.95%` 动作翻转，预测直方图从 `[1, 466, 42, 50]` 变成 `[0, 544, 2, 13]`；`a2-code-dagger-s1`（1,065 帧）有 `21.88%` 动作翻转，预测直方图从 `[7, 832, 68, 158]` 变成 `[0, 1065, 0, 0]`，所有转向预测被彻底抹平为前进。
- **为什么是缺陷及影响**：这是第一轮 C3、G7 在真实航向码权重上的定量证实，且卡 S1a（`5387747`）修 C2 时在 `StudentAdapter` 新引入了硬编码 `COND_PREV_ACTION` 的半截逻辑。不修此项，任何带航向码的模型一进 3DGS 评测或上板端脱离 `.uavr` 运行，就会在直飞一步后持续吃进死码 `1` 变成盲目直冲器，且进程退出码为 `0`。
- **怎样复现或验证**：运行草稿目录 `quant_probe.py` 查看末尾 `[错喂死码1实验]` 行；或将 `a2-code-dagger-s1/ckpt_best.bin` 传给 `StudentPolicy` 后连续两步不传 `prev_action` 调用 `act()`，第二步自动喂入 `1` 而不报错。
- **建议的最小修法**：(1) `checkpoint` 升 v7，写入 `cond_kind`（无 / `prev_action` / `heading_code_steps` / `heading_code_deg`）与合法取值掩码 `valid_cond_mask`（航向码为 `0b1101`，推理端收到死码 `1` 立即抛异常拒收）；(2) `uavrgb_infer` 握手行与板端导出包 `deploy.json` 同步输出 `cond_kind`；(3) `StudentPolicy.act` 在 `prev_dim > 0` 且 `prev_action is None` 时直接抛 `ValueError`，删掉回落到 `self._last_action` 的隐式分支。
- **状态**：已复现（真实权重数值已复现，代码路径已核实）。
- **归属建议**：**要等卡 E2E-C1 一起改的**（C++ `checkpoint` 升版、握手行扩展与 `sim/tools/student.py` 删隐式回落并入卡 `E2E-C1`；3DGS 侧接 `check_student_cond` 已并入卡 `S2`；`<ckpt>.json` 记条件类型已并入卡 `W1-15c`）。

### N-4：板端推理路径与训练路径存在 5 处算子及输入契约断层，仓库无 CPU 参考前向与 ONNX 导出器

- **位置**：`src/augment_dev.cu:20-23`（提交 `6cad3b7`）；`src/infer.cpp:23-38`、`:68-83`、`:93-102`（提交 `ed25f52`）；`src/trainer_eval.cpp:47-53`（提交 `ed25f52`）；`sim/tools/student.py:90-125`（提交 `ed25f52`）；`tools/check_input_parity.py:55-64`、`:120-135`（提交 `6cad3b7`）；`wt/p12-e2e-p0/proto/export.py:1-130`。
- **现象**：逐行比对训练路径（`preprocess.py` → `trainer_dev.cpp` → `DeviceCnn`）、PC 闭环路径（`student.py` → `uavrgb_infer`）与 SG2002 板端预期路径，共有 5 处断层，且全仓没有任何不依赖 CUDA 的整网前向或导出脚本（`proto/export.py` 只把 `.pt` 转回自定义 `UAVRCK01` v6，不导出 ONNX）：
  1. **输入缩放与 `/255` 量化台阶**：训练与 PC 闭环均用 `PIL.Image.resize((160, 120), BILINEAR)`（带抗锯齿半核支撑的连续双线性核），再用 `static_cast<float>(u8) / 255.0f` 映射到 `[0, 1]`（共 256 个离散台阶）。SG2002 侧若用 CVI VPSS 硬件缩放，坐标对齐方式不同（第一轮 G4 测得 `cv2.INTER_LINEAR` 翻转 `0.7%`，最近邻翻转 `1.9%`）；同时 TPU-MLIR 把 `scale=1/255` 融合进对称有符号 int8 输入张量时，非负区间只占 `[0, +127]` 共 128 个台阶，输入动态分辨率减半（暗部每 2 个 uint8 灰度级合并为 1 级）。
  2. **双帧堆叠与集首边界无法走 NPU 图像前处理**：`in_c=6` 时，`infer.cpp:68-79` 接收两张 HWC uint8 图并按 `[当前帧 R,G,B ; 上一帧 R,G,B]` 摆成 `6×120×160` NCHW，集首第 0 步由 `student.py:120` 把当前帧复制两份且 `prev=0`。TPU-MLIR 的硬件图像预处理（CSC/FusePreprocess）只支持单张 3 通道图像，双帧模型在板端必须关闭硬件预处理融合，由 CPU 维护 `3×120×160` 上一帧环形缓冲并在每次起飞解锁时显式 `reset()`。
  3. **文本分支绑定进程生命周期，且空文本不是中性元**：`infer_main.cpp:52-54` 只在启动进程时通过 `--text` 接收一条指令，`infer.cpp:33-36` 在 `text_dim > 0` 且未传 `--text` 时填全 `0`（`<pad>`）。虽然 `<pad>` 的嵌入行恒为 `0`，但 `film` 线性层偏置 `film.b` 非零，全 `<pad>` 会产生一组特定的通道调制 `(1 + film.b[:64])·p3 + film.b[64:]`；板端常驻守护进程若想不重启切换指令，现有协议不支持按集更新 token。
  4. **`argmax` 严格大于的平局偏置**：`infer.cpp:100`（`if (logits[k] > logits[best]) best = k;`）与 `trainer_eval.cpp:50` 一致采用严格大于号，平局时恒取下标最小的类（优先级 `SAFE(0) > FWD(1) > LEFT(2) > RIGHT(3)`）。FP32 下平局概率极低，但在 int8 输出或定点累加下（`N-2` 实测平局率 `0.52%–2.48%`），若板端 C 代码手写成 `>=`，平局帧会从 `FWD(1)` 翻成 `RIGHT(3)`。
  5. **输入对拍工具漏测双帧与条件**：`tools/check_input_parity.py:57-64` 用 `__new__` 绕过 `__init__` 硬编码 `in_c = 3`、`prev_dim = 0`，没有覆盖双帧拼接、条件字节与校准偏置 `fc.b`；`src/tests/test_infer_layout.cpp` 只测了 `run_infer_loop` 内部，没覆盖板端导出模型。
- **为什么是缺陷及影响**：没有 CPU 独立参考前向与 ONNX 导出器，部署链第 1–4 环停在纸面；上述第 2、4 条一旦板端手写 C 代码与 `student.py` / `infer.cpp` 口径不一，会在首帧或平局帧静默偏航。
- **怎样复现或验证**：读上述文件行号即可确认；运行 `.venv-tools/bin/python tools/check_input_parity.py --help` 可见仅支持单张 PNG 与 `in_c=3`。
- **建议的最小修法**：(1) 新增纯 numpy 参考前向 `tools/ref_forward.py`（约 70 行，草稿 `quant_probe.py` 的 `backbone_fp32` 已验证与 C++ 一致）与无 PyTorch 依赖的 `tools/export_onnx.py`（直接按 protobuf 或 `onnx.helper` 把 `UAVRCK01` v6 拼成 `c1..p3` ONNX 图，旁写含 `film/fc` 权重与 `argmax` 规则的侧车）；(2) 扩展 `tools/check_input_parity.py` 覆盖 `in_c=6` 与 `prev_dim > 0`，并导出 20 帧金标向量（输入 uint8 + 条件码 → 期望 `p3` 特征与 4 类 logit）供板端 C 程序自检。
- **状态**：读代码确认。
- **归属建议**：**现在就能在仿真一侧修的**（修改 `tools/check_input_parity.py`，新增 `tools/ref_forward.py` 与 `tools/export_onnx.py`，纯主机侧 Python/numpy 即可落地，不依赖硬件或卡 E2E-C1）。

### N-5：条件维度硬编码为 4，且推理流缺少集边界复位标志，阻断 E2 九档码与有状态模型的部署

- **位置**：`src/trainer_dev.cpp:150-151`（提交 `ed25f52`）；`src/aux_sidecar.cpp:104`（提交 `ed25f52`）；`sim/tools/student.py:76-78`、`:90-104`（提交 `ed25f52`）；`tools/average_ckpt.py:210`（main 提交 `55cba8d`）及分支 `p12-e2e-1` `tools/average_ckpt.py:40,249`（提交 `dc8e5ff`）；`src/infer.cpp:61-88`（提交 `ed25f52`）。
- **现象**：
  1. 计划 §10.8 与任务卡 `E2E-P1` 已确定 E2 条件码改用净转向计数 $n$ 的 9 档（`±4`，加上条件置空「未知」共 10 档）。模型头 `src/include/uavrgb/model.hpp:47` 虽留了 `kMaxPrevDim = 64`，但 `trainer_dev.cpp:151` 写死 `cfg_.prev_action ? classes : 0`（恒为 4），`aux_sidecar.cpp:104` 校验 `v >= num_classes` 就报错，`student.py:77` 写死 `if self.prev_dim not in (0, 4): _abort_start`，分支 `p12-e2e-1` 新改的 `average_ckpt.py:40` 也写死 `PREV_DIM = 4`。
  2. 推理进程行协议 `src/infer.cpp:51,61-83` 每步固定读 `one * images + (prev_dim > 0 ? 1 : 0)` 字节，没有任何字节表示「新的一集开始（reset）」或「本帧是 warmup 假帧」。`student.py:90-99` 在启动时往管道喂一帧全零图像跑 `_warmup()`，换集时 `student.reset()` 只清 Python 侧缓存，不通知子进程。
- **为什么是缺陷及影响**：当前无状态 CNN 在 `infer.cpp` 里每帧独立覆盖显存缓冲，全零 warmup 不会残留；但计划 §10.7 第 11 条与 `E2E-P1` 候选含「按已执行偏航平移的列特征缓存」与「小循环单元」，且板端推理守护进程在多次试飞（arm → disarm → 再 arm）之间常驻不重启。行协议与板端接口若无显式 `reset` 信号，warmup 全零帧与上一集末态会直接污染下一集第 0 步的内部状态（第二轮 `L-3` 在原型包装器上也指出了同一隐患）。
- **怎样复现或验证**：读 `student.py:77` 与 `infer.cpp:51` 确认：给握手行传入 `"prev_action_dim": 9` 时 `StudentPolicy` 立即抛 `RuntimeError`（`test_student.py:138` 甚至把拒绝非 0/4 当成回归断言锁死了）。
- **建议的最小修法**：(1) 训练器加 `--cond-dim K`（缺省跟 `num_classes` 走，保持旧命令行逐位不变），`aux_sidecar`、`student.py`、`average_ckpt.py` 统一按模型头声明的 `prev_dim`（`1..64`）校验；(2) 行协议与未来板端推理 API 增加集首复位语义（例如条件字节最高位 `0x80` 标 `episode_start`，或支持带 `reset` 标记的控制帧，warmup 后自动复位）。
- **状态**：读代码确认。
- **归属建议**：**要等卡 E2E-C1 一起改的**（并入卡 `E2E-C1` 的条件维度可配与握手协议升级，与第二轮 `L-3` 主线侧一并处理）。

### N-6：板端计算条件码所需的飞控回执与偏航遥测在串口与固件中均无通路，且双端物理 UART 存在引脚冲突与误触发风险

- **位置**：`tools/uart_action.py:19`、`:252-283`（提交 `c54070d`）；`docs/uart_action.md:10-46`、`:124`（提交 `c54070d`）；固件 `CF-Drone/CF-Drone.ino:26,58`、`cli.ino:147-200,239-248`、`rc.ino:131-158`、`mavlink.ino:47-67`、`board_config.h:20-56,68-103,111-142`（提交 `e7f507f`）；硬件资料 `$D/tmp/research-hw/rvnano_5_peripheral.md:14-125,351-355`。
- **现象**：端到端线规划了两种机载条件码：方案甲「按已执行动作累计净转向步数 $n$」（计划 §10.3，`nav_cond.py`），方案乙「按飞控姿态估计偏航角相对起飞航向的差值分档」（计划 §10.7 第 6 条，应对第一轮 G2、G3）。对照现有串口代码、固件原件与板卡手册，两条路都走不通：
  1. **串口工具只写不读，协议无上行帧**：`tools/uart_action.py:260` 用 `os.O_WRONLY | os.O_NOCTTY` 打开设备，没有串口读路径，`FIELD_KINDS` 只有下行 6 个字段。
  2. **方案甲缺执行回执与否决状态**：真机上动作可能因 CRC 错丢包、飞控未解锁、RC 摇杆接管（`safety.ino:89-101`）、低压保护（`safety.ino:149`）或落入 `0.1` 偏航杆死区（`control.ino:152`）而未执行或提前中止；没有来自 ESP32 的 `(seq, executed_action, actual_ms, veto_reason)` 回执，板端盲目按「已发出动作」累加 $n$，漏一拍即整集恒偏 `26°`。
  3. **方案乙缺串口姿态下行与起飞航向锁存触发**：开源固件 `CF-Drone` 的姿态估计 `attitude`（`estimate.ino:48`，纯陀螺积分，无磁力计）只在 `WIFI_ENABLED` 时经 WiFi UDP 以 10 Hz 发 `MAVLINK_MSG_ID_ATTITUDE_QUATERNION`（`mavlink.ino:47-51`），串口上没有任何遥测输出；且若飞控主控是 `ESP32-C3`，`board_config.h:49` 硬编码 `BOARD_WIFI_ENABLED = 0`，连 WiFi MAVLink 也被编译期裁掉。此外，计算相对偏航 $\Delta\psi = \text{wrap}(\psi(t) - \psi_0)$ 必须知道「起飞解锁进入自主模式」的精确时刻以锁存 $\psi_0$，单靠板端自己是不知道飞手何时拨杆解锁的。
  4. **双端物理 UART 资源冲突与安全隐患**：ESP32 侧，`Serial`（UART0）跑无校验的逐字符明文 `cli.ino:239-248`，二进制帧一旦误接进去，包含换行符的随机字节流随时可能匹配短命令（如 `arm`、`mfr` 单电机全速盲转）；`BOARD_RC_SERIAL`（ESP32/S3 为 `Serial2`，C3 为 `Serial1`）已被 ELRS/CRSF `420000` 波特率占用，且 ESP32-C3 总共只有 `UART0` 和 `UART1` 两个硬件串口，没有空闲第三路硬件 UART。LicheeRV Nano 侧，`UART0` 默认挂着 Linux 控制台与内核 `printk`（`rvnano_5_peripheral.md:14-72`），不改 `/boot/uEnv.txt` 开机日志就会灌进飞控；带 WiFi 的 W/WE 版 `GPIOP18..23` 被 SDIO WiFi 占用导致 `UART3` 不可用（`:351`），全板唯一默认干净可用的是 `UART1`（`/dev/ttyS1`，`GPIOA28/29`，未焊蓝牙排阻时可用，`:353`）。
- **为什么是缺陷及影响**：直接决定真机接口的物理可行性与串口协议定稿。若不先把单向动作协议改成全双工「下行动作帧 + 上行状态/偏航回执帧」，计划 §10.7 第 6 条「E2 起条件码优先按角度定义」在硬件上拿不到输入。
- **怎样复现或验证**：核对 `tools/uart_action.py:260`、`CF-Drone/board_config.h:21,49` 与 `rvnano_5_peripheral.md:351-355`；上板后用逻辑分析仪测 LicheeRV Nano `GPIOA28/29`（`/dev/ttyS1`）上电期间是否有杂散输出，并在 ESP32 CLI 输入 `sys` 核对芯片型号（ESP32 / S3 / C3）。
- **建议的最小修法**：(1) 硬件确认 ESP32 具体型号（若是 C3 需评估软件串口/SPI 或与 CRSF 复用方案；若是 ESP32/S3 则用空闲 `UART1` 分配独立引脚）；LicheeRV Nano 固定用 `/dev/ttyS1`；(2) 在定稿 `real-drone/uart_protocol.md` 时定义为全双工协议：下行帧带 `(magic, ver, seq, action, duration_ms, crc16)`，ESP32 以 20–50 Hz 回传定长遥测/回执帧 `(magic, ver, ack_seq, exec_action, exec_ms, mode_flags, yaw_cdeg_i16, vbat_mv, crc16)`，一次同时满足方案甲（`ack_seq + exec_action + exec_ms`）与方案乙（`mode_flags` 解锁边沿锁存 $\psi_0$ + `yaw_cdeg_i16` 算相对偏航角）。
- **状态**：读代码确认（仓库与固件源码已核实；真机芯片子型号与板卡版本待实物确认）。
- **归属建议**：**只有拿到真机或板子才能核实的**（需在真机上核实：ESP32 芯片子型号与空闲引脚、刷官方闭源固件还是改开源固件、LicheeRV Nano `/dev/ttyS1` 上电电平与双向误码率；协议文档定稿后，上位机全双工收发器可在无硬件下用伪终端先写）。

### N-7：串口链路在丢包、粘包、回绕乱序与无回执失步下的行为缺少接收状态机参考实现与仿真故障注入测试

- **位置**：`tools/uart_action.py:238-240`、`:285-296`（提交 `c54070d`）；`tools/test_uart_action.py:52-220`（提交 `c54070d`）；`sim/tools/collect.py:1854-1858`（提交 `d816d2c`）。
- **现象**：
  1. `tools/uart_action.py` 只实现了发送侧字节拼接，`test_uart_action.py` 也只校验单帧静态 hex，没有任何接收端滑动窗口重同步（假 `0xAA 0x55` 载荷字节粘包）、半帧截断超时、CRC 错帧丢弃、序号判重与乱序拒绝的参考实现与测试。
  2. 默认 `seq` 字段仅 1 字节（`0..255`，`uart_action.py:238`），与 10 Hz 心跳共用计数器时每 `25.6` s 回绕一次；若接收端简单按 `seq > last_seq` 判新帧而未用半窗口模运算 `(uint8_t)(seq - last_seq) - 1 < 127`，第 25.6 秒回绕到 `0` 后飞控会拒收后续全部帧直至 TTL 超时坠落。
  3. 默认校验为 `CRC-8`（`poly=0x07`，误判漏检率 `1/256 ≈ 0.39%`），有刷电机 25 kHz PWM 大电流换向与 WiFi 天线贴近排针的场景下，突发多比特干扰的漏检风险高于 `CRC-16-CCITT`。
  4. 仿真侧 `collect.py:1855-1856` 假定 `action_executed` 永远与派发动作相等（无丢帧、无否决），尚未测试「串口丢弃 `2%–5%` 动作帧而板端无回执照常累计 `net_yaw`」时航向码学生的越界退化幅度。
- **为什么是缺陷及影响**：`docs/uart_action.md:155` 写明上机前必须通过「CRC/重放/乱序/TTL 注入测试」，但仓库里连对端解析器参考模型与伪终端回环测试都没有；若把未定清回绕规则的协议直接交给固件团队各自实现，极易在 `seq` 回绕处出事故。
- **怎样复现或验证**：读 `tools/test_uart_action.py` 全文确认无 `openpty`、无接收端解析状态机、无丢包/粘包/回绕注入用例。
- **建议的最小修法**：(1) 默认配置切到已支持的 `crc16`（`poly=0x1021`）并将 `seq` 扩为 2 字节，在 `tools/uart_action.py`（或同目录 `uart_codec.py`）写出与固件严格同构的无动态分配接收状态机参考实现（含 `magic` 滑窗重同步、半窗口序号回绕比较、TTL 倒计时）；(2) 在 `tools/test_uart_action.py` 中用 `os.openpty()` 跑通「10% 随机比特翻转 + 粘包拆包 + 丢帧 + 乱序 + seq 255→0 回绕 + 断流 350 ms 触发 TTL」闭环单测；(3) 在 `sim/tools/collect.py` 增加可选诊断参数（默认 `0` 不变），模拟下发丢包变 `SAFE` 而历史码失步，测出 E2E-1 模型的容错上限。
- **状态**：读代码确认。
- **归属建议**：**现在就能在仿真一侧修的**（修改 `tools/uart_action.py`、`tools/test_uart_action.py`、`config/uart_protocol.example.json` 及 `sim/tools/collect.py` 诊断开关，纯软件即可完成，不依赖真机硬件或卡 E2E-C1）。

### N-8：真机相机档位视场裁剪、安装俯仰余量与前处理契约缺失，板端直接缩放会触发 9%–14% 动作翻转

- **位置**：`sim/tools/student.py:113-118`（提交 `ed25f52`）；`tools/preprocess.py:642-645`、`:728-745`（提交 `e41703e`）；`src/infer_main.cpp:69-72`（提交 `ed25f52`）；硬件资料 `$D/tmp/research-hw/maixcam_cameras.md:8-22`、`sim/dynamics/e1_mini.json:60-64`。
- **现象**：
  1. 仿真两条线固定为 `320×240`（`4:3`）、水平视场 `60°`（垂直视场 `46.8°`）、无畸变理想针孔，再双线性缩放到 `160×120`。真机候选相机 `GC4653` 原生分辨率 `2560×1440`（`16:9`），标称视场 `H81° × V51°`，镜头畸变 `5%`（`maixcam_cameras.md:10-21`）。第一轮 G4 已复现：把 `16:9` 直接缩成 `4:3`（`160×120`）翻转 `14.1%` 动作，视场仅偏 `2.2°` 即翻转 `8.7%` 动作。
  2. `GC4653` 手册写「30fps@2K, 60fps@720P」：若 SG2002 CVI ISP 驱动的 `1280×720@60fps` 档采用传感器中心开窗（crop）而非全画幅 `2×2` binning，水平视场会从 `81°` 腰斩至约 `46°`，低于仿真的 `60°`，物理上裁不出与仿真一致的画面，只能退回 `2560×1440@30fps`（帧周期增至 `33.3` ms 且占满 256 MB DDR3 中的 ION 视频缓冲）。
  3. 即使全画幅可用，`V51°` 相对仿真 `V46.8°` 上下合计只剩 `4.2°`（`±2.1°`）余量：相机支架安装俯仰误差或前进加速时的机体俯仰角（约 `3.8°`）一旦叠加，去畸变加中心裁切就会触及图像上下黑边；且四叶草护罩前缘是否进入实机广角镜头畸变区仍未实测。
  4. 仓库既没有按标定内参生成板端 `cv::remap` 查找表的工具 `crop_to_sim.py`，`uavrgb_infer` 握手行与 `checkpoint` 也不带任何 `hfov_deg`、`aspect`、`crop_spec` 元数据（第一轮 G7）。
- **为什么是缺陷及影响**：板端前处理不是简单的 `resize(160, 120)`，核心是「去畸变 + 相机安装角对齐 + 精确 `60°×46.8°` 视锥重投影」。漏掉这一步或选错传感器读出模式，模型在真机第 0 步就会看到拉伸失真的几何而失效。
- **怎样复现或验证**：第一轮 G4 探针（`$D/tmp/phase12-cards/review-g/resize_probe.out`）已复现视场与宽高比敏感度；仓库内检索 `crop_to_sim` 或 `undistort` 结果为 0 个文件。
- **建议的最小修法**：(1) 拿到 LicheeRV Nano 与摄像头后，第一时间运行 `/mnt/system/usr/bin/sensor_test` 分别在 `2K@30fps` 与 `720P@60fps` 下同机位拍同一张棋盘格，对比两档标定出的归一化焦距 `fx / width` 判定是否为中心开窗，并实拍电机旋转态确认护罩是否入画及室内照度曝光时长；(2) 软件侧先写好 `tools/crop_to_sim.py`（输入相机 `K, dist, R_mount`，输出 `160×120` 预计算 `int16` 定点 `remap` 表，板端每帧只需一次查表即可同时完成去畸变、安装俯仰补偿、视场裁切与双线性缩放到 `160×120`），并在卡 `E2E-C1` 的部署侧车里写入目标视场契约（`hfov=60.0, vfov=46.826`）。
- **状态**：读代码确认（视场敏感度第一轮已复现；传感器 720P 读出模式为推断，待板测）。
- **归属建议**：**只有拿到真机或板子才能核实的**（必须在板上实测：720P@60fps 档是否为全视场 binning、镜头内参畸变、支架安装俯仰角、护罩入画情况与室内卷帘曝光时间；定点 `remap` 表生成脚本可先在工具侧备好，视场契约字段随卡 `E2E-C1` 写入）。

### N-9：板端单步推理超时降级、进程守护与飞控失控保护的两级联动尚未定义，开源固件与 SITL 现有失控分支存在安全缺陷

- **位置**：`sim/tools/student.py:42`、`:136-162`（提交 `ed25f52`）；`tools/uart_action.py:331-359`（提交 `c54070d`）；`sim/cf_sitl/exec/executor.c:151-152`、`:197-200`（提交 `1bc8fed`）；固件 `CF-Drone/safety.ino:49-86`（提交 `e7f507f`）；硬件资料 `$D/tmp/research-hw/rvnano_1_intro.md:100`。
- **现象**：
  1. PC 评测客户端 `student.py:42` 默认 `timeout_s = 30.0`，超时直接 `kill()` 子进程并抛异常终止整集。板端飞行时决策周期约 `0.9–1.0` s（计划目标曝光到发令端到端 p95 `< 150` ms，`ROADMAP.md:40`），SG2002 运行非实时 Buildroot/Debian Linux（只有 256 MB DDR3，且内测版 70405 板卡 NPU 跑在 1.0 TOPS 下硬件不稳定、只能降频跑 0.5 TOPS，`rvnano_1_intro.md:100`）；一旦页面回收、ISP 掉帧或 NPU 驱动阻塞超过 `150` ms，板端主控循环缺乏「当拍立即下发 `SAFE_NEUTRAL(0)` 悬停并标记本步超时」的软看门狗规范。
  2. 飞控侧第二级保护：若采用方案 3 将 `sim/cf_sitl/exec/executor.c` 移植进开源固件，`CF_EXEC_FAILSAFE`（`:197-200`）当前开环锁死 `hover_stick` 不收油（第一轮 `sim-real.md` E1 实测推力偏高时 25 s 爬升到 60 m）；而原版 `CF-Drone/safety.ino:55-86` 的 `descend()` 假定存在 `ALTHOLD_HOVER_THRUST`（实际把推力钳到 `0.5`，对 102 g 整机产生约 `5.2 m/s²` 近自由落体下坠），且只在 `controlTime != 0` 后断流超过 `1.0` s 才触发。
- **为什么是缺陷及影响**：没有把「板端一级超时（>150 ms 发 `SAFE_NEUTRAL` 定点刹停）→ 串口断链二级超时（>300 ms 飞控自动转定点悬停 2 s，仍无恢复或 ToF 失效则闭环匀速缓降上锁）→ 实体遥控器最高优先级一票否决接管」串成闭环状态机并在无桨台架逐项演练，首飞一旦遇到 Linux 抖动或推理进程退出就会失控撞顶或砸地。
- **怎样复现或验证**：读 `executor.c:197-200` 与 `CF-Drone/safety.ino:61-86` 确认现有失控保护代码；第一轮 `sim-real.md` §E1 已在 SITL 复现 60 m 飞丢。
- **建议的最小修法**：(1) 制定两级超时契约写入真机安全规程：板端推理循环用 `CLOCK_MONOTONIC` 设 `150` ms 硬截止，超时当拍发 `SAFE_NEUTRAL` 且不更新动作计数；板端进程开硬件看门狗 `/dev/watchdog`；(2) 先在 SITL 修掉 `executor.c` 的 E1（闭环降落或受控线性收油）；(3) 上浆叶前必须在无桨台架完成四项断链演练：拔掉串口 TX 线、`kill -9` 板端推理进程、注入连续 CRC 坏帧、拨动实体 RC 模式开关强行夺权。
- **状态**：读代码确认（SITL 飞丢由上轮实测确认，板端延迟与台架响应待实测）。
- **归属建议**：**只有拿到真机或板子才能核实的**（需在 LicheeRV Nano 实板上确认 PCB 版本 70405/70415、实测 ISP+NPU+CPU 全链路 p50/p99 延迟分布，并在 ESP32 无桨台架上实测断链降落与 RC 强行夺权；SITL 侧 `executor.c` 的 FAILSAFE 修复可先在仿真侧单独立卡）。

---

## 2b. 从 `ckpt.bin` 到真机起飞的全链路现状与缺口总表

| 步骤 | 现在仓库里有什么（文件与测试） | 缺什么 | 缺的部分里哪些不用硬件就能先做 |
| --- | --- | --- | --- |
| 1. 导出 | `src/checkpoint.cpp`（`UAVRCK01` v6 读写，`test_checkpoint_versions`）；`tools/average_ckpt.py`（末 10 轮平均加 `fc.b[1]` 校准，`test_average_ckpt.py`）；分支 `p12-e2e-p0` 的 `proto/ckpt_io.py`、`proto/export.py`（仅 `.pt ↔ .bin`，无 ONNX） | `UAVRCK01` → ONNX 导出脚本；无 CUDA 的整网独立参考前向；随模型输出的 `deploy.json`（含输入视场、裁剪、条件种类与维度、`argmax` 规则） | **全部可先做**：写纯 Python `tools/ref_forward.py` 与 `tools/export_onnx.py`（按 `p3` 切图导出主干 ONNX + 侧车 `film/fc` 权重），生成 20 帧金标输入输出供板端回归对拍 |
| 2. 量化 | 无任何仓库内代码或 TPU-MLIR 环境记录（仅第一轮 `src.md` 草稿与本轮 `quant_probe.py` 做了离线伪量化模拟） | 校准集抽取脚本（覆盖各条件码）；SG2002（`cv181x`）TPU-MLIR 转换流水线（`model_transform` → `run_calibration` → `model_deploy` 出 `.cvimodel`）；FP32 对 `.cvimodel` 的转向帧翻转率门禁 | **校准抽取与离线门禁可先做**：写校准集抽取器与对称 int8 伪量化门禁（卡死 max 校准、禁用输出层 int8、检验转向帧翻转率 `≤ 2%`）；若在主机 Docker 配好算能 `tpu-mlir`（支持 x86 仿真跑 `.cvimodel`），连 `.cvimodel` 编译与逐帧数值对拍都不用真板即可跑通 |
| 3. 板端前处理 | PC 侧 `tools/preprocess.py:load_frame`、`sim/tools/student.py:113-118`（PIL `BILINEAR` `320×240 → 160×120`）；`tools/check_input_parity.py`（仅单帧无条件 RGB）；`src/tests/test_infer_layout.cpp` | CVI VI/VPSS 取帧代码；按真机相机内参、畸变与安装俯仰角裁出 `60°×46.8°`（`4:3`）并缩放到 `160×120` 的重映射工具 `crop_to_sim.py`；双帧与条件的对拍测试 | **重映射工具与测试可先做**：写 `tools/crop_to_sim.py`（输入任意标定 JSON，产出板端可直接加载的 `160×120` `int16` 定点 `remap` 查表并用合成畸变图单测），扩充 `check_input_parity.py` 覆盖双帧与条件；真机 720P 是否开窗及真实 `K/dist` 需等板测 |
| 4. 板端推理 | PC 侧 CUDA 常驻进程 `src/infer_main.cpp`、`src/infer.cpp`（测试：`test_infer_io`、`test_infer_layout`）；客户端 `sim/tools/student.py`（`test_student.py`） | SG2002 板端 C/C++ 推理程序（调 `cviruntime` 跑 `c1..p3` int8，CPU FP32 跑 `FiLM + fc` + 严格 `>` 的 `argmax`）；双帧环形缓冲与起飞 `reset()`；单步 `150` ms 超时回落 `SAFE` | **CPU 后半段与状态机可先做**：板端 `FiLM + fc + argmax + 双帧缓存 + reset + 150ms 超时降级` 是纯标准 C/C++ 逻辑，可在 x86 上先写好并用第 1 步导出的 20 帧 `p3` 金标跑通 CTest，上板只剩填 `CVI_NN_Forward` 约 30 行胶水代码 |
| 5. 条件输入怎么算 | `sim/tools/nav_cond.py`（三值航向码 `heading_code` 及依赖训练文件的 `check_student_cond`，测试：`test_nav_cond.py`、`test_nav_cond_wiring.py`） | (1) `checkpoint` 与 `deploy.json` 自身携带 `cond_kind`、`prev_dim`（支持 9 档码）与合法码掩码；(2) 板端根据飞控上行回执累计 `net_yaw`（按步数）或根据飞控上行偏航角与解锁边沿算 `Δψ` 分档（按角度） | **契约与纯函数计算器可先做**：随卡 `E2E-C1` 升版 `checkpoint` 写进 `cond_kind` 并放开 `prev_dim`；用 C/Python 写好板端「回执驱动的动作计数器」与「解锁锁存 $\psi_0$ 的角度分档器」及丢回执/跨 `±180°` 折回单测 |
| 6. 动作下发 | `tools/uart_action.py`、`config/uart_protocol.example.json`、`docs/uart_action.md`（测试：`tools/test_uart_action.py`，已进 `check.sh`） | 修掉 `N-1` 四个常驻发送 bug；补动作持续时长字段 `duration_ms`（表达「有效 `0.8` s 后刹停」或新持续协议）；选定双端不冲突的硬件串口；ESP32 固件侧四动作解析器（开源 `CF-Drone` 当前为 0 行） | **上位机编解码器与固件解析模块可先做**：修 `uart_action.py`，定稿二进制帧规格，把 ESP32 端无动态分配的 `CRC-16` 滑窗解析状态机写成独立纯 C 文件（同 `sim/cf_sitl/exec/executor.c` 方式接入 SITL 跑 CTest），无需真机即可测透粘包、乱序与 `seq` 回绕 |
| 7. 回执 | **无**（`uart_action.py:260` 只写 `O_WRONLY`；固件串口无结构化回传，`mavlink.ino` 仅在 WiFi 开启时发 UDP 遥测） | ESP32 → LicheeRV Nano 上行帧定义与固件发送代码（含 `ack_seq`、`executed_action`、`exec_ms`、`mode_flags`、`yaw_cdeg_i16`、`vbat_mv`）；上位机非阻塞串口读取与对齐逻辑 | **双工协议与伪终端联调可先做**：在主机用 `os.openpty()` 把板端发送/接收线程与第 6 步的纯 C 飞控解析/回执模拟器连成闭环，注入 10% 丢包与否决事件验证条件码不失步 |
| 8. 失效保护 | `student.py` 30 s 杀进程超时；`uart_action.py` 帧内写静态 `ttl_ms=300`；`CF-Drone/safety.ino`（RC 1 s 失联 `descend`、134° 倒置停机、2.6 V 低压降落）；`sim/cf_sitl/exec/executor.c`（ToF 0.5 s 超时进 `FAILSAFE`） | 修 `uart_action.py` EOF 继续发心跳屏蔽 TTL 的漏洞（`N-1`）；修 `executor.c` 的 `FAILSAFE` 锁死悬停油门飞丢（上轮 E1）；板端推理超时降级与 `/dev/watchdog`；飞控两级超时（先定点刹停再闭环缓降）及 RC 强行接管台架验证 | **软件与 SITL 侧可先做**：修 `uart_action.py` EOF 退出行为，在 `sim/cf_sitl/` 里修好 `FAILSAFE` 闭环缓降与动作 TTL 到期状态转换并加 CTest；物理拔线与 RC 夺权必须等无桨台架实测 |

### 附：板端条件码对串口协议的精确需求与 INT8 量化实测摘要

1. **板端算条件码需要串口给什么（现在的协议给不了）**：
   - 若用**按步数计的历史码**（计划 §10.3，`n = 已执行左转 − 已执行右转`）：板端不能按自己发出的动作计数，必须由飞控在每个动作窗口结束或被中断时回传 `(ack_seq, executed_action, actual_ms, veto_flag)`，只有确认飞控在自主模式下实际执行超过有效阈值（例如 `actual_ms ≥ 500` ms 且未掉入死区）才累加 `n`；否则一次丢包或飞手拨杆接管就会让后续整集航向码永久错位。
   - 若用**按角度计的航向码**（计划 §10.7 第 6 条，固件偏航估计相对起飞航向分档）：它是无累积漏计误差的更优方案，但需要串口以 `≥ 10–20` Hz 持续上行 `(armed_auto_flag, yaw_est_cdeg)`。板端在 `armed_auto_flag` 由 `0 → 1` 的上升沿锁存起飞基准航向 $\psi_0$，每拍取图像曝光时刻最近的 $\psi(t)$ 计算 $\Delta\psi = \text{wrap180}(\psi(t) - \psi_0)$ 再分档（注意开源 `CF-Drone` 无磁力计，`attitude` 偏航纯靠陀螺积分，前 80 s 漂移可控，超长程仍会慢漂）。现有 `uart_action.py`（只写单向）与开源固件（串口仅明文 CLI 或单向 RC 输入）对这两组字段均无支持。
2. **INT8 量化风险数值汇总（出处：`quant_probe.out`，对称 int8 `[-127, 127]`，256 帧校准）**：
   - `cal_swa_s1_k10.bin`（单帧无条件，1,458 帧评估）：无 BN 未造成卷积激活爆炸（`p1/p2/p3` 峰值 `1.32/2.30/5.27`），`19,200→4` 全连接逐通道与逐张量权重量化在 FP32 输出下总翻转仅 `0.34%` 与 `0.21%`，转向帧翻转均为 `0.00%`；真正危险的是两处配置陷阱：一是**输出层一并对称 int8 量化**（因 `SAFE` 类负向极值 `-58.47` 拉大步长至 `0.415` nat/LSB，转向帧翻转跳至 `4.40%`，top-2 整数平局 `1.37%`），二是**激活按 `99.9%` 分位截断**（`p3` 长尾被削，转向帧翻转跳至 `3.14%`，`max|Δlogit|=8.48` nat）。
   - `a1-code-demo-s1` / `a2-code-dagger-s1`（带 FiLM 航向码，574 / 1,249 帧评估）：FiLM 的 `(1+γ)·p3 + β` 进对称 int8 后，`a2` 总翻转升至 `3.60%`、FP32 预测转向帧翻转 `8.59%`（连输出层一起 int8 则达 `11.36%`）；改为 **NPU 算 `c1..p3` int8 + CPU 算 `FiLM + fc` FP32** 后，`a2` 总翻转即回落至 `1.04%`、预测转向帧翻转 `1.52%`。另外，`a1` 因示范集里 `code∈{2,3}` 仅占 `5%`，随机抽 256 帧校准时 `code=2` 仅抽中 4 帧；若校准脚本漏喂侧车条件（全按 `code=0` 校准），`p3mod` 阈值会从 `7.89` 低估到 `6.42`。

---

## 3. 第一轮清单相关条目核对

核对范围：任务书指定的 `review-g.md` 之 `G2`、`G7` 与 `review-c.md` 之 `C3`，以及直接关联的 `C1`、`C2/G1`、`C4`、`G4`、`G5`、`F2`。

- **修对了的**：
  - `F2`（部署推理路径 `run_infer_loop` 的输入布局单测）：卡 `T1`（提交 `ceac416`）新增了 `src/tests/test_infer_layout.cpp`，用 `16×24` 非方形输入与独立参考前向锁住了 `HWC → CHW` 下标、RGB 通道序、双帧 `[当前帧 ; 上一帧]` 顺序及 4 个 `prev` 取值，对调两帧会明确变红。
- **修了一半的**：
  - `C3`（条件字节语义不在 checkpoint 与握手行里）：卡 `W1-9`（提交 `d816d2c`）在 `sim/tools/nav_cond.py` 加了 `check_student_cond` 并在 `eval_closed_loop.py:1415` 接入，但存在三处缺口：(1) `collect.py` 与 3DGS 全线（`collect_3dgs.py`、`eval_3dgs.py`）未接该闸门（第二轮 `I2-2`、`J-5` 已报）；(2) 依赖外部 `config.json` 与原始 `.uavr` 文件存在，板端与独立分发权重恒退化为 `unverified`；(3) `student.py` 仍保留缺省回落 `_last_action`，`collect_3dgs.py`（卡 `S1a` 提交 `5387747` 修 `C2/G1` 时）又把任意 `prev_dim>0` 模型硬编码当 `prev-action` 喂入死码 `1`（见本报告 `N-3`）。
  - `C1`（交付配方 `average_ckpt.py` 不支持带条件、双帧、文本模型）：`main`（`ada0ca5`）上仍未合入；分支 `p12-e2e-1` 的 `dc8e5ff` 只补了单帧无文本 `prev_dim=4`，仍不支持双帧（`in_c=6`）、文本（`text_dim>0`）与 `prev_dim != 4`，且 `<ckpt>.json` 未记数据集条件类型（第二轮 `I2-5`）。
  - `G5`（仿真一步固定时序与文档、占位串口协议不一致）：公共规则 §1 的文字描述已更正，卡 `W1-13` 第一、二部分（提交 `e40c693`）合入了 `safe_gap_s` 与 `plugin_ack_sim_t` 记账，但第三部分「动作持续到下一次派发」及 `uart_action.py` 侧的动作时长表达尚未改动。
- **没修的（均在排期或待硬件清单中，代码未动）**：
  - `G2`（真机无已执行动作回执，仿真无下发丢失模型）：`tools/uart_action.py` 与 `docs/uart_action.md` 自 `c54070d` 以来 0 行改动；本轮进一步查出常驻管道吞行与 EOF 心跳屏蔽 TTL 缺陷（`N-1`）及双端物理串口冲突（`N-6`）。
  - `G7`（握手行与 checkpoint 缺前处理与条件契约）：`src/checkpoint.cpp` 与 `src/infer_main.cpp` 自 `ed25f52` 以来 0 行改动，排在卡 `E2E-C1`。
  - `C4`（条件维度写死 4、翻转镜像写死 2↔3）：未改，排在卡 `E2E-C1`（本报告 `N-5` 补充了 `average_ckpt`、`student.py` 与行协议无 `reset` 标志的联动阻塞）。
  - `G4`（`crop_to_sim` 内参视场裁剪工具与训练视场随机化）：未实现（见本报告 `N-8`）。

---

## 3b. 上真机之前必须先有的测试清单

1. **无需硬件、在主机 CI 上必须先过的 5 项测试**：
   - **串口常驻管道与双工伪终端故障注入测试**（扩充 `tools/test_uart_action.py`）：用 `os.openpty()` 连接上位机收发器与纯 C 飞控解析状态机，覆盖同批多行写入不吞行、半行不阻塞心跳、上游 EOF 立即停心跳并在 `350` ms 内触发对端 TTL、10% 随机比特翻转（CRC-16 全拦）、粘包拆包滑窗重同步、`seq` 跨 `65535→0` 回绕、回执驱动的条件码不失步。
   - **导出模型与 `uavrgb_infer` 的金标对拍测试**：同一份交付权重（含单帧、双帧、FiLM 9 档码），`tools/ref_forward.py`、导出的 ONNX 图、`uavrgb_infer` 三者在 200 帧真实样本（含集首帧与平局边界帧）上概率最大绝对差 `< 1e-5`、动作 `100%` 一致。
   - **INT8 转向帧专项门禁测试**：校准后模型（或 `p3` 切图混合模型）对 FP32 参照，不仅总动作一致率 `≥ 99%`，且真值/预测为 `YAW_LEFT/RIGHT` 的子集上动作翻转率 `≤ 2.0%`、死码拒收生效。
   - **`crop_to_sim` 几何重投影闭环测试**：对已知 `60°` 仿真图施加合成桶形畸变（`k1=-0.08`）、`3°` 安装俯仰旋转与 `16:9` 扩边，经 `crop_to_sim` 定点 `remap` 还原出 `160×120` 后喂给 `uavrgb_infer`，与原图动作一致率 `≥ 98%`。
   - **板端 C 推理状态机宿主机单测**：在 x86 上编译板端推理核心（桩掉 `CVI_NN_Forward`，喂入导出的 `p3` 金标），验证双帧缓存、起飞 `reset()` 清零、严格 `>` 平局规则、单拍超时 `>150` ms 强制输出 `SAFE_NEUTRAL(0)`。
2. **装桨起飞前、在真板与无桨台架上必须过的 4 项测试**：
   - **相机模式与视场实测门**：在 LicheeRV Nano 上实测 `2K@30fps` 与 `720P@60fps` 的棋盘格内参，确认裁剪后水平视场落在 `60° ± 1°`、四叶草护罩在桨叶旋转时不入有效区。
   - **LED 端到端延迟与抖动台架门**：GPIO 点亮 LED 入镜 200 次，实测「曝光 → ISP/remap → NPU `p3` → CPU `FiLM+fc` → 串口发出 → ESP32 收到并回执」全链路延迟 p50 与 p99（要求 p95 `< 150` ms，连续 10 分钟无内存增长、无 NPU 挂死）。
   - **串口抗干扰与引脚安全台架门**：无桨解锁电机转至 `80%` 油门（引入 25 kHz 有刷换向电磁干扰），双工连续收发 10,000 帧，统计 CRC 错帧率 `< 0.1%`、误接受坏帧 `0` 次；LicheeRV Nano 冷重启期间飞控不误解锁、电机不抖动。
   - **三级失效保护无桨演练门**：在无桨运行态下依次执行：(a) `kill -9` 板端推理进程；(b) 拔掉串口线；(c) 阻塞推理线程 `300` ms；(d) 拨动实体遥控器接管开关。逐项核实飞控日志均按预期在规定毫秒内进入定点刹停、闭环缓降或 RC 接管，无油门锁死爬升或直接掉电停桨。

---

## 4. 查过但没有发现问题的地方

1. **输入分辨率与文本常量的跨语言一致性门禁**：`tools/check.sh:15-38` 自动比对 `src/include/uavrgb/observation.hpp`（`160×120`）与 `tools/preprocess.py`（`DEFAULT_SIZE = (160, 120)`），以及 `src/include/uavrgb/text.hpp`（`kTextLen=9, kTextVocab=22`）与 `preprocess.py` / `sim/tools/instruction.py`；当前两边完全一致。
2. **PC 侧单帧图像从采集 PNG 到训练与闭环的像素级对齐**：`tools/check_input_parity.py` 经无损 PNG 与 `StudentPolicy.act` 两条路径解码 `RGB_INT8` / `RGBA_INT8` / `BGR_INT8`，到达 `src/augment_dev.cu:21`（`x / 255.0f`）前的 `uint8` 张量逐字节一致；辅助深度头在 `src/infer.cpp:58-59` 中不参与前向，不增加推理开销。
3. **FORWARD 先验校准偏置的落盘方式对板端透明**：`tools/average_ckpt.py:128-132` 直接把校准标量 $\delta$ 加进 `fc.b[1]`（`float32` 原地写回 `checkpoint`），板端推理只需照常算线性层加 `argmax`，无需额外维护后处理阈值参数。
4. **`tools/uart_action.py` 的单帧编码与 CRC 算术**：`crc8` 与 `crc16` 均通过标准测试向量 `'123456789'`（`0xF4`、`0x29B1`、`0x31C3`）校验；`validate_config` 正确拦截了 `bool` 伪装成 `int`（`isinstance(value, bool)`）、CRC 非末字段、波特率不在 `termios` 表内及动作码越界；`tools/check.sh:187-190` 实际上已把 `tools/test_uart_action.py` 纳入自动回归（仅 `docs/uart_action.md:194` 的旧说明写着未纳入）。
5. **Gazebo 线的条件喂入顺序与集间复位**：`sim/tools/collect.py:966-969,1855-1856` 在 `student` 策略下正确按「上一步实际执行动作」维护 `prev_executed` 与 `net_yaw`，首步传 `0`，与 `tools/preprocess.py:328-370` 的打包口径严格对齐。
6. **开源固件避免把二进制流误文档化的安全声明**：`docs/uart_action.md:10-25` 对 `CF-Drone` 的 `Serial`（CLI）与 `BOARD_RC_SERIAL`（CRSF/SBUS）做了准确预警，明确禁止直接将占位二进制帧灌入现有固件串口。

---

## 5. 没有查的地方与原因

- **没有跑算能官方 `tpu-mlir` / `cviruntime` 工具链**：本机容器未安装算能 TPU-MLIR 编译器与 `cv181x` 运行时，因此 `N-2` 的量化实验采用纯 numpy 对称有符号 int8（`[-127, 127]`）伪量化，硬件层面的移位定点乘子（multiplier + rshift）舍入细节未核实。
- **没有打开真实串口设备（`/dev/ttyS*`）或操作硬件引脚**：当前环境为服务器容器，无 LicheeRV Nano 与 ESP32 实物；`N-1` 的串口工具复现通过 `--dry-run` 与匿名管道完成。
- **没有反汇编官方闭源固件 `official_E1_MINI_20261102.bin`**：仅沿用第一轮已核实的字符串表结论与开源 `CF-Drone`（`e7f507f`）源码，官方固件内部的定点杆量映射曲线与控制源抢占状态机未核实。
- **没有启动 Gazebo、没有跑训练、没有占用 GPU、没有运行 `tools/check.sh`**：严格遵守任务书第 3 节只读与不抢占实验资源的规定。
- **没有深入审查 `sim/cf_sitl/exec/executor.c` 的内部控制律细节与文档全量过时表述**：前者属角度 `K`，后者属角度 `O`，本报告仅在涉及失控保护耦合（`N-9`）与 `docs/uart_action.md` 处点到为止。

---

## 附录：过程记录（只读审查）

### 做了什么
- 按任务书顺序读完个人规范、`2026-11-12-review-round2-brief.md`、`CLAUDE.md`、`2026-11-10-phase12-common.md` §1–§4、`PLAN_2026-11-10_phase12.md` §0 与 §10、`FIXLIST.md`、`FIXLIST2.md`、第一轮 `review-g.md` / `review-c.md` / `review-h.md`、上一轮 `src.md` / `sim-real.md` 及 `e2e-p0.md`。
- 全仓检索量化、导出、NPU 与串口相关文件，逐行审读 `tools/uart_action.py`、`tools/test_uart_action.py`、`docs/uart_action.md`、`config/uart_protocol.example.json`、`src/infer.cpp`、`src/infer_main.cpp`、`src/checkpoint.cpp`、`src/model.cpp`、`src/model_dev_film.cu`、`src/augment_dev.cu`、`sim/tools/student.py`、`sim/tools/nav_cond.py`、`sim/3dgs/collect_3dgs.py`、`sim/3dgs/eval_3dgs.py`、`tools/average_ckpt.py`（main 与 `p12-e2e-1`）、`wt/p12-e2e-p0/proto/export.py`、`$D/real-drone/README.md`、`$D/tmp/research-hw/` 板卡与相机手册及 `CF-Drone` 固件源码。
- 在草稿目录 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-n/` 编写并单进程低优先级（`nice -n 15`，纯 CPU）运行 `uart_probe.py` 与 `quant_probe.py`，取得常驻管道故障复现与 3 个真实权重（`cal_swa_s1_k10`、`a1-code-demo-s1`、`a2-code-dagger-s1`）的 int8 量化及错喂死码实测数据，完成本报告 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/review2-n.md`。

### 没做什么（及原因）
- 未修改仓库内任何文件、未写 `docs/worklog/` 与 `progress.md`、未提交或建分支：原因，任务书第 3 节硬性规定只读审查。
- 未跑 TPU-MLIR 编译 `.cvimodel`、未测真实串口与相机硬件：原因，容器内无工具链与真机硬件。
- 未启动 Gazebo、3DGS 渲染、模型训练或 `tools/check.sh`：原因，遵守任务书禁止项与 12 核配额保护规则。

### 验证到哪一步

| 检查项 / 命令 | 结果 |
| --- | --- |
| `nice -n 15 .venv-tools/bin/python .../review2-n/uart_probe.py` | 通过（复现常驻管道吞第 2 行动作、EOF 后持续发心跳、同轮双发、半行阻塞 4 项现象，落盘 `uart_probe.out`） |
| `OMP_NUM_THREADS=1 nice -n 15 .venv-tools/bin/python .../review2-n/quant_probe.py` | 通过（完成 `cal_swa_s1_k10`、`a1-code-demo-s1`、`a2-code-dagger-s1` 激活统计、对称 int8 各配置翻转率、`p3` 切图混合精度与死码 `1` 翻转实验，落盘 `quant_probe.out`） |
| `git -C /root/claude-code-backup-20261028/code/uav_rgb status --short` | 空（主仓库与各工作树保持只读干净，无残留后台进程） |
