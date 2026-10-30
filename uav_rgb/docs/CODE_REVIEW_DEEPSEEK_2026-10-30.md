# UAV RGB 代码审核问题清单与独立复核材料

> 用途：把本文件连同仓库代码交给 DeepSeek，逐项独立复核本次审核结论。
> 审核者：Codex 主 agent + 3 个只读审核 agent；本文不代表 DeepSeek 已认可这些结论。
> 原审核时间：2026-10-30 17:13–17:49，Asia/Shanghai。
> 整理时间：2026-10-30 18:10 起。
> 状态：6 项正式发现，初评均为 P2；本次没有实施修复。

## 0. 给 DeepSeek 的复核任务

请独立审核下列六项，不要因为本文已有结论就直接同意。优先核实实际代码、调用链、输入是否合法，以及复现是否足以支持声称的影响。

请对每项分别回答：

1. **结论**：成立 / 部分成立 / 不成立 / 证据不足；不要将尚未复跑写成已经验证。
2. **证据**：具体函数、关键语句、真实调用方和触发输入；如行号变化，用函数名定位。
3. **影响**：普通工作流问题、异常数据处理、测试可信度问题，还是极端数值边界；是否影响当前数据/训练，需要单独论证。
4. **复现质量**：是否调用真实实现，替身替换了哪些部分，是否存在构造不合法、跳过入口校验、浮点比较或时钟模型的问题。
5. **严重性**：是否认同 P2；若调整，说明现实使用场景和理由。
6. **最小修复**：明确改哪里、哪些调用方必须同步改、有哪些兼容性影响；不要只写“增加校验”。
7. **验收**：给出修复前应失败、修复后应通过的回归断言，并指出需不需要 GPU / Gazebo。

本文中的修法是建议，未经过实施验证。请重点找出修法本身是否会改变标签与 RGB 的对应关系、错误地选择 best、破坏既有数值合同，或扩大改动范围。

如果只能看到本 MD、看不到完整代码，请明确区分“依据摘录可判断的行为”和“需要仓库才能核实的调用/合同”，不要假装读过未提供文件。

## 1. 审核输入与项目约束

### 1.1 代码版本和证据

- 工作区：`/root/claude-code-backup-20261028/code/`。
- 项目根：`/root/claude-code-backup-20261028/code/uav_rgb/`。
- HEAD：`e0d2e4e6ae65dfaa1f2e2f55f81cd44a86fa6d53`，**另有未提交改动**；不能只 checkout 此提交后声称已复核本文版本。
- 原审核针对当时的整个工作树；外部 `source-before.json` 保存 214 份输入文件的 SHA256、大小、mtime。
- 18:10 整理时再次核对六项发现涉及的 13 份关键源码，均与原审核 hash 相同。关键文件 hash 见附录 F。
- 原始证据目录：`/root/rivermind-data/uav_rgb/current-code-review-20261030-1713/`。
- 两份原始记录：[审核过程](worklog/2026-10-30-1713-current-code-review.md)、[实验结果](experiments/2026-10-30-current-code-review.md)。本文件已内嵌关键代码、复现和输出，可单独转交阅读。

本文所有“当前进度”指上述审核快照，不自动代表之后其他会话继续修改的状态。

### 1.2 与判断相关的合同

| 项目 | 合同/实际环境 |
| --- | --- |
| 学生输入 | 仅单目 RGB；数据集 NCHW，仿真观测 HWC，由适配层转换 |
| 学生输出 | 4 个离散动作：0 SAFE_NEUTRAL、1 FORWARD_SLOW、2 YAW_LEFT、3 YAW_RIGHT |
| 特权信息 | 深度、几何、位姿真值只供 teacher/oracle，不能加入学生输入 |
| 数值实现 | 自写 C++/CUDA；禁止 PyTorch、cuBLAS、cuDNN、TensorRT、ROS 2 |
| GPU | 原审核实际检测到 RTX 2080 Ti，sm_75；CUDA 12.8、GCC 13.3 |
| 仿真 | Gazebo Harmonic，软件 GL；hold 已定义为仿真时间秒 |
| 数据划分 | 按场景/几何组隔离；不能按同场景帧随机拆分 |
| 离线指标 | 固定 4 类口径，同时报 acc、macro_f1、balanced_accuracy、每类 recall；无样本类 recall/F1 为 0 |
| 模型选择 | 按 val/dev 宏 F1 选择 best；test 不参与模型选择 |
| 产物位置 | 数据集、权重、run、实验日志在 `/root/rivermind-data/uav_rgb/`，不能写入 `code/` |
| 代码规模 | C++ 单文件 ≤280 行、CUDA ≤360 行；修复应遵守既有文件边界 |

进一步核对时，每个 shell 先 `source uav_rgb/env.sh`（从工作区根运行）；运行 GPU 前重新 `nvidia-smi -L`。原审核时服务器另有训练和 GPU 进程，所以本次耗时只记录功能验证，不作加速结论。

## 2. 实现进度：避免把基础设施交付当成并行训练完成

| 模块 | 原审核确认的实现程度 | 判断依据 |
| --- | --- | --- |
| 训练主路径 | CNN forward/backward、加权 CE、裁剪、Adam 已可训练 | `src/trainer.cpp`、`model.cpp` 和 CUDA 包装 |
| checkpoint/best | v3，保存 `best_macro_f1`，按 dev 选择 best | `src/checkpoint.cpp`、`trainer.cpp` |
| RunOwner | 已接训练 CLI 和 eval-only CLI | `src/train_main.cpp:94、129` |
| eval-only/推理 | 独立 checkpoint 复评与常驻推理进程已接入 | `trainer_eval.cpp`、`infer_main.cpp` |
| 数据与闭环入口 | v3 几何、A* 教师、SAFE 出生、打包/审计与学生闭环入口已实现 | `sim/tools/`、`tools/preprocess.py` |
| T1–T7 | 审核快照中 T1–T5 done，T6 doing，T7 todo | 当时任务板及实际 T6 进程；不是本文件整理时重新验收 |
| device 基础层 | DeviceBuffer/PinnedBuffer、Context/Event、参数/Adam 状态导入和独立 snapshot 已实现 | `src/device/`、相关测试 |
| BatchReader | 已实现独立 pread，但没有接入训练热路径 | Trainer 仍用 `UavrDataset::load_images` |
| resident 设备链 | Conv1→ReLU→Pool、整网 forward/backward、CE/norm/Adam 整步和 Trainer backend 尚未实现 | 没有对应生产执行器；设备状态层只有导入/导出 |
| 梯度归约并行 | db/dW 两阶段分块归约尚未实现 | 旧 kernel 每个权重/通道线程串行遍历 N/H/W |
| 数据流水线 | 两个 pinned slot、copy stream 与 compute stream 重叠尚未实现 | 独立 reader 测试不能代替训练流水线 |

GPU 空闲问题的代码依据仍在：旧 CUDA 包装逐层 malloc、上传输入和权重、同步、回传 host Tensor；dW/db 没有把 N/H/W 归约维展开为足够多的 block。这些是**已登记的未完成实现**，没有混进下面六项新发现计数。

本次没有重新跑 Nsight 或性能基准，不能由上述静态事实量化“当前 GPU 空闲百分比”或“预计必然加速几倍”。

## 3. 问题总览与证据强度

P2 在本文表示有明确触发条件、需要修复的正确性或测试问题；不等于所有输入都会出错，也不表示已造成当前 T5/T6 产物损坏。

| 编号 | 初评 | 问题 | 已完成的验证 | 尚未证明 |
| --- | --- | --- | --- | --- |
| R01 | P2 | 推理前观测时戳被用于推理后的动作 hold 起点 | 真实 `collect_one` + 内存假订阅/发布者复现 | 真实 Gazebo 下具体缩短量及对导航成绩的影响 |
| R02 | P2 | actions/oracle 用 zip 截短，坏尾行漏审仍全 PASS | 真实 `episode_facts/audit` + 内存文件替身，11/11 PASS | 当前 T5 已损坏；额外清点反而未发现条数/step 不一致 |
| R03 | P2 | 跨目录续训继承 best 分数但不迁移 best 权重 | 真实 CLI、真实 GPU 小训练、源 best 先复评确认 | 所有 resume 都出错；同目录且 best 完好并非本例 |
| R04 | P2 | 数值对拍误差函数忽略 NaN | 编译执行相同表达式，得到零误差 | 当前生产 kernel 已经输出 NaN |
| R05 | P2 | 有限大类别权重让加权梯度全零 | 生产 CPU/CUDA 函数均实测 | 正常 auto 权重/T6 已触发 |
| R06 | P2 | 有限梯度总 norm 转 float 溢出，裁剪清零 | 生产 CPU/CUDA 函数均实测 | 当前 T6 实际出现如此大的梯度 |

## 4. R01：策略计算时间被算入动作执行窗口

### 4.1 位置、调用和当前行为

- 主位置：`sim/tools/collect.py::collect_one`，原行 582、594、612。
- 关联统计：原行 666–674 的 sim_dt、displacement、dyaw 和 stuck。
- 关联写盘：原行 862、872 的 `pose_rgb_gap_ns` 和 `pose_ts`。
- 可达调用：`eval_closed_loop.py::run_episode → collect.py --policy student → main → collect_one → student.act`。
- `policy=forward` 无学生推理开销；teacher 也有决策开销，程度取决于实现耗时。

关键执行顺序是：先按 RGB 配对得到 `pose_before_ts`，再计算学生动作，最后发布动作。但 hold 终点仍由最早的观测时戳计算。

```python
# collect.py:582，发生在前面已经取得 pose_before_ts 之后
action, infer_ms, _probs = student.act(rgb_msg)

# collect.py:594，发生在 student.act 返回之后
action_start_ns = pose_before_ts
action_handle = publisher.start(*twist_fields(action, altitude_vz(pose_before)))

# collect.py:612
hold_end_ns = action_start_ns + int(hold * 1e9)
while True:
    msg = subs["pose"].latest()
    if msg is not None and _pose_stamp(msg) >= hold_end_ns:
        break
    # 下方还有健康检查、墙钟兜底与 contact 轮询
```

以上摘录省略了 PNG 写入和错误处理，仅用于呈现真实语句顺序。

### 4.2 触发条件、预期和实测

1. RGB 匹配的位姿时戳为 10.01 s。
2. `student.act` 执行期间仿真继续运行，最新位姿来到 11.5 s。
3. 设置 `hold=1.0`；代码算出的 hold 终点是 11.01 s。
4. 实际开始发动作时已经是 11.5 s，所以第一次检查就认为窗口完成，马上发零速。

**预期**：按动作执行语义，推理完成后派发的新动作应获得约 1 s 的仿真执行窗口；若发布/状态不可确认，应显式失败，不能把推理期间当成已执行。

**内存复现实际输出**：动作与零速均记录在 `11500000000 ns`，返回 `error=None`，`action_start_ns=10010000000`、`sim_dt=1.49`。`action_wall_s` 约 0.00008 s。

若策略计算期间仿真推进量小于 hold，窗口也会相应缩短；归零是极端但清楚的构造。该内存测试没有模拟 Gazebo 接收与执行消息的全部延迟，不能用它声称真实物理执行精确为零。

### 4.3 影响边界

- 学生与盲前进实际接受的动作窗口可能因计算时延不同而不同；会干扰闭环比较。
- 记录的位移、偏航、sim_dt、stuck 也混入观测到派发之间的时间。
- 本发现不等于模型学得好或坏，不证明此前闭环失败完全由此引起。
- 这里只证明时钟语义错误，尚未进行慢推理的真实 Gazebo 注入实验。

### 4.4 建议修复及需要同步修改的地方

将观测基准与执行基准分开，保留两组事实：

| 基准 | 应保留/使用的内容 | 消费方 |
| --- | --- | --- |
| RGB 观测 | 原 `pose_before/yaw_before/pose_before_ts` | teacher 标签复算、RGB/pose 配对误差 |
| 动作执行 | 策略返回后、派发前最新有效位姿与仿真时戳；必要时与发布确认语义衔接 | hold 终点、执行位移、dyaw、sim_dt、stuck |

原 `collect.py:862、872` 直接用 `action_start_ns` 写配对位姿时戳。新增执行起点后，此处应明确改回观测时戳，不能把推理延迟变成传感器配对误差。

不要用“重新取位姿再重算标签”作为一行替换，这会让标签对应另一时刻的几何，而 RGB 还是旧帧。发布前 latest 也不自动等于控制器真正收到动作的时间；修复时需说明常驻通道与 CLI 回退的发布确认边界。

### 4.5 回归与请复核的问题

- 无 Gazebo 单测：让 `student.act` 返回期间推进假仿真时钟，断言派发后不会因旧时戳立即清零。
- 验证固定 hold 和两个不同推理耗时，执行窗口一致，观测配对时戳保持原值。
- 验证位姿停更和发布失败依旧零速兜底、非成功返回。
- 修复后补真实 Gazebo 慢推理注入，分开报告 dispatch、发布确认、动作停止的仿真/墙钟时刻。
- 请 DeepSeek 核实：项目 hold 合同是否确实指动作执行时间；本例是否足以证明当前代码混淆了观测与执行；建议的两组位姿是否还有漏改的消费者。

完整最小复现见附录 A。

## 5. R02：审计器静默按较短文件截断

### 5.1 位置和真实调用

- `sim/tools/audit_dataset.py::episode_facts`，原行 69–71 读 manifest/actions/oracle，原行 91 `zip(rows, oracle)`。
- `episode_facts` 返回的 `num_steps` 为 `len(steps)`，已经是截短后的数量。
- `main → episode_facts → audit` 使用截短数据做 safe_collision、time_pairing、label_recompute 等检查。
- `tools/preprocess.py::load_episode` 原行 123 读取完整 actions，原行 133 校验 actions 与 manifest 的 num_steps；它没有按 oracle 长度裁掉这些样本。

```python
# audit_dataset.py:69–71
man = load_json(os.path.join(ep_dir, "episode_manifest.json"))
rows = load_jsonl(os.path.join(ep_dir, man["actions_ref"]))
oracle = load_jsonl(os.path.join(ep_dir, man["oracle_ref"]))

# audit_dataset.py:91
for i, (row, orc) in enumerate(zip(rows, oracle)):
    dt = (orc["action_end_ns"] - orc["action_start_ns"]) / 1e9
    # 其余检查事实也从当前 row/orc 提取

# 返回对象里的计数来自截断后的 steps
"num_steps": len(steps)
```

### 5.2 触发输入和实测

构造 train/val/test 三个内存 episode，每个：

- manifest：`num_steps=25`、`requested_steps=25`，三项 validity 都为 true。
- actions：25 行，前 24 行是 `[0,2,1,1,3,1] ×4`；各动作标签用当前 teacher 真正复算验证。
- oracle：只有 24 行。
- 最后一行 actions：SAFE，`collision=True`，`depth_rgb_gap_ns=999999999`。

**预期**：文件条数不同，无法确认逐步配对，应读入失败或审计 FAIL；不能把不完整的 oracle 当作正常审计输入。

**实际**：共 75 行 actions 只审 72 行；没有 FAIL 或 SKIP，11 项全部 PASS；`safe_collision=0/12`，`time_pairing max_gap=0`，`label_recompute=0/72`。

这不是直接构造 audit 的最终 facts 来绕过加载校验：复现替换的是文件读函数，调用了真正的 `episode_facts`。

### 5.3 影响、反证及当前数据检查

损坏/截断文件能被盖上完整通过的结论。打包器与审计器可能消费不同的样本集；使用 drop-collision 时会过滤明确标记的碰撞步，但这不能补回未审计样本的全部时间、标签和 oracle 检查。

**没有认定当前 T5 已损坏。** 主 agent 另行只读清点真实 T5 的全部有效数据，结果：599 集，actions=23960、oracle=23960，manifest 数量及逐行 step 均一致，`problems=[]`。这条额外检查是对本发现实际影响范围的约束。

本复现验证文件不完整时的拒绝能力，不要求正常采集流程每次都会产生截断文件。

### 5.4 建议修复与回归

在 zip 前明确检查：

1. actions/oracle 的有效行数一致；不一致返回 error，并带两边数量。
2. manifest 有 num_steps 时，其类型/数值合法且等于有效行数；不要用新 `len(steps)` 覆盖后丢失原声明。
3. 两侧 step 是否唯一、是否按期望顺序对应；仅行数相同不能证明逐步配对正确。
4. requested_steps 与实际步数的关系按用途处理：完整训练采集应满足执行完整性；评测碰撞即停属于合法提前终止，不能一概用 requested_steps 强制相等。

应新增 oracle 短一行、长一行、相同步数但 step 错位、坏尾行、manifest 计数错误的负例；正常完整集和合法提前终止集按各自合同验收。

请 DeepSeek 特别复核：这些约束应放 episode_facts、audit 还是共享加载处；preprocess 是否也需要消费完整性结果；不要因为本例当前 T5 数量正常，就把校验器本身的漏洞判成不存在。

完整最小复现见附录 B。

## 6. R03：跨目录续训保留 best 分数，却没有 best 权重文件

### 6.1 代码与合同

- `src/trainer.cpp::Trainer::run` 原行 82–85：load_checkpoint 继承 `state_.best_macro_f1`。
- 原行 97 附近：best 路径始终指向新的 `cfg_.out_dir/ckpt_best.bin`。
- 原行 223–225：只有新 dev 宏 F1 严格大于历史值，才写该文件。
- `src/train_main.cpp:129` 对显式 resume 放行输出目录；没有要求 source 与 out 是同一目录。
- `test_class_weights_train.cpp` 已有新目录 resume 用法，但只检查类别权重，未断言历史 best 产物。

```cpp
if (!cfg_.resume.empty()) {
  load_checkpoint(cfg_.resume, model_, state_);
  class_weights_source_ = merge_resume_weights(cfg_.class_weights, class_weights_,
                                               state_.class_weights);
}

// 随后每轮末尾
if (!dev_is_train && dev_stats.macro_f1 > static_cast<double>(state_.best_macro_f1)) {
  state_.best_macro_f1 = static_cast<float>(dev_stats.macro_f1);
  save_checkpoint(best_ckpt_path, model_, state_);
}
save_checkpoint(ckpt_path, model_, state_);
```

### 6.2 触发与已执行的确定性复现

源目录有真实 `ckpt.bin` 和 `ckpt_best.bin`。用一个合法 8×8、4 样本数据集（2 train、2 val，标签全部为 0）构造输出类别 0 的模型。

固定 4 类宏指标时，此验证集只含类别 0；预测全对，acc=1，macro_f1=balanced_accuracy=0.25，recall `[1,0,0,0]`。先通过生产 eval-only 确认源 best，避免任意填一个虚假的历史分数。

执行真实 CLI：

```text
uavrgb_train --data dataset.uavr --resume source/ckpt.bin
            --out resumed --epochs 1 --batch 2 --lr 0.001
```

实际子命令退出 0，续训 dev 宏 F1 仍为 0.25；随后检查：

```text
resumed_best=0.25 last_exists=1 best_exists=0 source_best=1
```

输入分数是可复算的实际 best，不依赖伪造 `best_macro_f1=1`；核心条件是“新目录 + 本轮未严格创新高”。

### 6.3 影响与修复选项

新的输出目录不能独立交付它宣称保存的历史 best。后续 `--eval-only resumed/ckpt_best.bin` 或 T7 选择 best 的流程会找不到文件。

请先明确跨目录 resume 的合同，再选最小实现：

- 若跨目录 resume 被支持：定位并验证历史 best，然后迁移到新输出目录。不能仅猜测 source 的兄弟文件一定就是对应 best；需要核对结构、状态来源和分数关联的可验证边界。
- 若不能可靠携带历史 best：明确拒绝这种续训方式，或要求显式提供历史 best。
- 若产品希望定义“新阶段最佳”：应清楚改合同和命名/记录，不能把重置分数后的较差模型仍称为全程 best。

源 `ckpt.bin` 是末轮参数时，通常不包含历史最佳那一轮的参数；**不能简单把 source ckpt.bin 复制成 ckpt_best.bin**。也不应无条件把 resume 当作允许覆盖任意其他 run 的授权。

### 6.4 建议验收与复核重点

- 新目录续训未创新高：best 文件可取回且对应源历史 best，或入口明确拒绝。
- 创新高：best 替换为新模型，记录分数/epoch 与参数一致。
- 同目录且 best 已存在：保留既有行为；同目录 best 缺失时也要明确处理。
- 缺源 best、结构错误、来源不符：不能训练成功后才静默丢交付物。
- 请 DeepSeek 判断：这是已支持 workflow 的 bug，还是需要收紧合同的缺口；即使选择“不支持跨目录”，当前是否缺少拒绝逻辑。

完整 C++ 数据/状态构造与 CLI 顺序见附录 C。

## 7. R04：CPU/CUDA 对拍的最大误差函数会吞掉 NaN

### 7.1 已核对的位置

| 文件 | 函数/原行 |
| --- | --- |
| `src/tests/test_conv_cuda.cpp` | `max_abs_err`，41 |
| `src/tests/test_linear_cuda.cpp` | `max_abs_err`，41 |
| `src/tests/test_nn_cuda.cpp` | `max_abs_err`，26 |
| `src/tests/test_weighted_loss.cpp` | `max_abs_diff`，45 |

示例真实写法：

```cpp
double max_abs_err(const uavrgb::Tensor& a, const uavrgb::Tensor& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.numel(); ++i) {
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return m;
}
```

### 7.2 原理与实测

有限参考 0 对 NaN 结果，差值的 fabs 为 NaN。对于 `std::max(m, NaN)`，第二个操作数不大于第一个，比较不会选择它，返回已有 m。初始 m 为 0，后面没有其他误差时最终就是 0。

本轮编译执行相同表达式，真实输出：

```text
test_parity_helper expected=0 actual=NaN reported_max_error=0
```

证明的是**验收函数无法可靠拒绝非有限退化**，没有修改或伪造生产 kernel 来声称已经出现此输出。其他显式 `isnan` 测试只覆盖其自己的场景，不自动修复这些最大误差函数。

### 7.3 修复及回归

- 对期待有限输出的 parity，先检查 shape 和双方每个值均有限，再计算最大误差；发现 NaN/Inf 就失败，给出索引。
- 不要只写 `if (diff > tolerance)`，NaN 与阈值比较也为 false。
- 专门测试 NaN 传播的用例应显式断言 `isnan`；不能把它套进“必须全有限”的通用数值对拍。
- 用参考有限、候选中间某项 NaN 的完整 Tensor 验证比较器确实拒绝；如设计共享 helper，可给共享 helper 这一条有效回归，不必为每份镜像测试重复大量样板。

请 DeepSeek 核实这四处 helper 的具体断言消费者，判断目前哪些测试会被绕过；不要把“测试存在假绿路径”扩大成“现有 20 项全是假的”。

## 8. R05：正有限的大类别权重让加权梯度归零

### 8.1 接口、输入合法性与位置

- CUDA：`src/softmax_cuda.cu::softmax_ce_grad_weighted_cuda` 原行 253。
- CPU：`src/ops_weighted.cpp::softmax_ce_grad_weighted` 原行 83。
- CUDA kernel 原行 115 的参数 `wsum` 也是 float；原行 127 做除法。
- CLI `class_weights.cpp::parse_explicit/resolve_class_weights` 接受有限、非负且训练类非零的显式权重；`3e38f` 本身可表示且为正有限。
- 原接口 `softmax_cuda.hpp` 合同要求 weights 长度 C、有限非负、样本权重和 >0，没有写“总和必须可表示为 float”。

```cpp
// double 求和之后，在梯度入口窄化
const float wsum = static_cast<float>(check_weight_sum(labels, weights, c));

// kernel 和 CPU 参考使用同类表达式
g[idx] = weights[labels[i]] * (p - (j == labels[i] ? 1.0f : 0.0f)) / wsum;
```

### 8.2 数学性质与已测结果

加权平均交叉熵定义：

```text
D = sum_i w[y_i]
L = sum_i w[y_i] * CE_i / D
dL/dz[i,j] = (w[y_i] / D) * (p[i,j] - 1[j=y_i])
```

所有类别权重共同乘同一个正数，归一化后的 loss 和 gradient 应保持不变（允许正常浮点舍入误差）。

输入为 logits 全零 `[2,4]`、labels `[0,1]`。权重从全 1 改成全 `3e38f`：

- double 样本权重和约为 `6e38`，有限。
- 转 float 后为 Inf。
- 分子仍有限，除以 Inf 得到 ±0。
- loss 的归一化使用 double 分母，仍约为 log(4)，所以现有 loss 有限检查不会终止。

真实 CPU 与 CUDA 输出完全相同：

```text
scale=1     loss=1.38629 grad=[-0.375,0.125,0.125,0.125,0.125,-0.375,0.125,0.125]
scale=3e+38 loss=1.38629 grad=[-0,0,0,0,0,-0,0,0]
```

### 8.3 影响边界和建议

这是明确的极端数值边界，不是当前 auto 类别权重或 T6 出错的证据。已有 Adam 动量仍可能造成参数变化，所以准确影响是“本 batch 的新学习信号被静默清零”，不能笼统说所有训练状态完全不动。

建议保留 double 分母计算归一化比例，或采用有数学依据的公共缩放；只改 host 局部变量为 double、但 kernel 参数仍 float，不算完整修复。CPU 与 CUDA 都要改。

补性质回归：正常权重与大常数等比例缩放后的 loss/grad 一致；检查所有梯度有限、并至少有应非零的项。单纯 CPU/CUDA parity 会让两边相同错误一起通过。

请 DeepSeek 核实乘除的中间精度和 kernel 参数，判断最小稳定表达式；不要用未经说明的权重上限绕过原输入合同。

生产函数复现代码与全部输出见附录 C/D。

## 9. R06：有限梯度的全局 norm 转 float 溢出，裁剪变成清零

### 9.1 位置与调用

- CUDA：`src/nn_cuda.cu::global_grad_norm_cuda` 原行 310；`clip_grad_norm_cuda` 原行 316–318。
- CPU：`src/nn_cpu.cpp::global_grad_norm` 原行 188；`clip_grad_norm` 原行 194–198。
- Trainer 在 backward 后直接调用 CUDA clip，然后更新所有参数的 Adam。

```cpp
// sum 是 double，只检查了平方和
if (!std::isfinite(sum)) {
  throw std::runtime_error("global_grad_norm_cuda: 梯度含非有限值");
}
return static_cast<float>(std::sqrt(sum));

// clip 中
const float norm = global_grad_norm_cuda(views);
const float raw = max_norm / (norm + 1e-6f);
const float coef = raw > 1.0f ? 1.0f : raw;
```

### 9.2 构造与实测

输入每项均为有限 float：`g=[3e38f,3e38f]`，`max_norm=5`。

```text
double sum ≈ 1.8e77，有限
double norm ≈ 4.24264e38，有限，但超过 float 最大值
转 float 后 norm = Inf
coef = 5 / Inf = 0
g *= 0 → [0,0]
```

CPU 和 CUDA 真实输出都是：

```text
norm=inf coef=0 values=[0,0]
```

若按数学裁剪，两个输出应约为 `5/sqrt(2)=3.5355339`，而不是零。

### 9.3 合同、修复选择与验收

旧 API 明确返回 float norm；它未明确承诺对超出 float 范围的 norm 必须抛错，所以不要仅凭未来 resident 方案把“没有抛错”当作违反旧合同。正式发现是：clip 对有限输入静默产生错误的零梯度。

可接受修复方向：

- 内部以 double norm 计算稳定的缩放系数，并说明返回 float norm/coef 的边界；或
- 转换后立刻检查有限性，明确拒绝该次 clip，避免修改梯度并继续 Adam。

计划 `PARALLEL_TRAINING_PLAN.md §33.3` 对未来 resident 路径已要求 float norm 转出 Inf 判失败；旧 CPU/CUDA 若采用同一拒绝语义应写清行为变化。

回归需覆盖“每个分量有限、整体 norm 溢出 float”，不能只测直接输入 NaN/Inf。选择拒绝方案时断言异常发生且梯度未被部分修改；选择稳定裁剪方案时断言输出有限、非零、范数约 5。正常小梯度与普通裁剪仍须通过。

请 DeepSeek 复核：sum 的 double 精度确实足以承受本例，溢出发生在哪次转换；建议修法是否会受极小系数或 CUDA 浮点设置影响。

## 10. 额外观察、已登记欠债与文档漂移

### 10.1 ReLU backward 的 NaN 合同观察：未计入六项正式发现

`nn_cpu.cpp:82`、`nn_cuda.cu:67` 都使用：

```cpp
dx[i] = (x[i] > 0.0f || isnan(x[i])) ? dy[i] : 0.0f;
```

实测 `x=NaN,dy=1 → dx=1`。注释写 NaN 继续传播，但现有 NaN 回归同时给 x/dy 置 NaN，无法区分“由 x 传播”还是“直接复制 dy”。

完整训练的前向/loss 有其他非有限守卫，未证明该分支导致正常训练静默成功。请独立判断应补合同还是列缺陷；不能据这一个算子观察就推断当前模型坏了。

### 10.2 已登记问题，避免重复当作新发现

- resume 缺数据签名、RNG 精确恢复；device snapshot 未保存完整 best 元信息。
- checkpoint fsync/唯一临时文件提交协议；数据 reader 的部分溢出防御。
- `.run.lock` 删除重建后的互斥边界、采集同输出目录的并发 TOCTOU。
- depth/contact 停更门禁、多 block parity 覆盖、硬件部署合同。
- resident 训练及双槽流水线尚未接入。

这些可以按已有工单继续审查，不应与本文件新发现重复计数。

### 10.3 文档与代码冲突

- `PARALLEL_TRAINING_IMPLEMENTATION.md` 仍把 RunOwner 接 CLI、v3 checkpoint 列为未做，实际主路径已有；sampler/fsync 仍未做。
- README 进度表还写闭环“尚无实现”、几何去重与仿真时间窗口未做；不能据它否定已有代码。
- HANDOFF 的旧 19 项测试、best 尚未评估与后续 T4 更新并存；原审核实际注册并通过 20 项。
- TASKS 的 T7“跑前补 src”过时，`collect.py:867` 已写盘。

本次没有替其他会话批量改写这些共享入口。请复核时按具体代码和证据确认状态，不把过时段落当成事实。

## 11. 已完成验证和明确没做的验证

原审核实际跑过独立构建与完整 `tools/check.sh`，退出 0：20/20 CTest 通过；Python 自检和非 GUI UI 自检通过；GUI 一项因没有 DISPLAY 跳过。原始总测试时间为 183.65 s，**不是性能基准**。

本文件整理没有重新跑这一整套检查，引用的是原审核日志；核心源码 hash 已复核一致。

| 项目 | 本文的证据范围 |
| --- | --- |
| GPU 功能回归 | 原审核真实 GPU，20/20；没有把历史 17/17 当当前结果 |
| R01 | 内存订阅/发布者复现；未做真实 Gazebo 慢推理注入 |
| R02 | 内存文件替身复现；额外只读清点真实 T5 数量与 step |
| R03 | 真实 CLI/真实 GPU 小训练，源 best 用 eval-only 验证 |
| R04 | 编译执行相同误差表达式；未故意修改生产 kernel |
| R05/R06 | 生产 CPU/CUDA 函数均复现 |
| 当前导航能力 | 本次没有新训练/真实闭环验收，不能给出新的 SR 或泛化结论 |
| 并行加速 | 没有新 Nsight、occupancy、吞吐基准或 memcheck |
| 业务修改 | 未修业务代码；原始未提交改动未回滚、暂存或提交 |

建议复核顺序：R01/R02 → R03 → R04 → R05/R06。若继续实施性能改造，先修可信对拍，再按设备链、整网 forward、backward、整步、Trainer 接入、db/dW 归约、双槽预取逐步验收。

## 12. 建议 DeepSeek 返回的复核表

| 编号 | 成立/部分成立/不成立/证据不足 | 复跑环境与输出 | 调整后的级别 | 同意/修正的根因 | 最小修复和回归 |
| --- | --- | --- | --- | --- | --- |
| R01 | 待复核 | 待填 | 待填 | 待填 | 待填 |
| R02 | 待复核 | 待填 | 待填 | 待填 | 待填 |
| R03 | 待复核 | 待填 | 待填 | 待填 | 待填 |
| R04 | 待复核 | 待填 | 待填 | 待填 | 待填 |
| R05 | 待复核 | 待填 | 待填 | 待填 | 待填 |
| R06 | 待复核 | 待填 | 待填 | 待填 | 待填 |

请另列：无法取得的文件/日志、没有跑的验证、不同意本文的判断，以及新增发现。复核阶段先出判断，不要求直接修改生产代码。

## 附录 A：R01 完整内存复现及原始输出

从工作区根 `/root/claude-code-backup-20261028/code` 运行。此命令不启动 Gazebo/GPU，不写真实数据文件，使用仓库已有测试替身。

以下代码从原审核保存的源码逐字导出，没有为了整理报告改写它的控制流。断言的含义是“确认旧漏洞能出现”，修复后应改用第 4 节定义的正确行为回归，不能为了让这段漏洞探针继续过而保留错误。

```bash
source uav_rgb/env.sh
"$UAV_RGB_PYTHON" - <<'PY'
import sys,struct
from unittest.mock import patch
sys.path.insert(0,'uav_rgb/sim/tools')
import collect as c
from test_collect import FakeSub,pose_msg,image_msg,depth_msg
before=pose_msg(10.01);after=pose_msg(11.5)
rgb=image_msg(4,6,10)
dep=depth_msg(4,6,16,'R_FLOAT32',struct.pack('<24f',*([5.]*24)))
dep['header']={'stamp':{'sec':10,'nsec':0}}
pose=FakeSub(frames=[(1,before)],latest=before)
subs={'rgb':FakeSub(after=rgb),'depth':FakeSub(frames=[(1,dep)]),'pose':pose,'contact':FakeSub()}
class Student:
 def act(self,msg):
  pose._latest=after
  return 1,1490.,[0.,1.,0.,0.]
class Pub:
 def __init__(self):self.sent=[]
 def start(self,*args):
  self.sent.append((c._pose_stamp(pose.latest()),args))
  return c.time.perf_counter()
 def wait(self,handle):return handle
 def abort(self,handle):pass
pub=Pub()
with patch.object(c,'write_rgb_png',return_value=(True,None)):
 row,err=c.collect_one('/memory',0,1.,'student',subs,pub,student=Student())
print('error=',err)
print('publish records (sim-time ns, twist fields)=',pub.sent)
print('reported action_start_ns=',row['action_start_ns'],'action_end_ns=',row['action_end_ns'],'sim_dt=',row['sim_dt'],'action_wall_s=',row['action_wall_s'])
print('actual action and zero sim-time difference=',pub.sent[1][0]-pub.sent[0][0])
assert err is None
assert pub.sent[1][0]-pub.sent[0][0]==0 and row['sim_dt']>=1.
PY
```

原审核真实输出：

```text
error= None
publish records (sim-time ns, twist fields)= [(11500000000, (0.3, 0.0, 0.0, 0.0)), (11500000000, (0.0, 0.0, 0.0, 0.0))]
reported action_start_ns= 10010000000 action_end_ns= 11500000000 sim_dt= 1.49 action_wall_s= 8.041597902774811e-05
actual action and zero sim-time difference= 0
```

## 附录 B：R02 完整内存复现及原始输出

同样从工作区根运行，不启动 Gazebo/GPU、不落真实 episode 文件。`patch` 只替换读文件函数与路径存在判断，实际解析和审计循环仍来自生产代码。

```bash
source uav_rgb/env.sh
"$UAV_RGB_PYTHON" - <<'PY'
import sys
from unittest.mock import patch
sys.path.insert(0,'uav_rgb/sim/tools')
import audit_dataset as ad
import split_manifest as sm
facts=[]
for split,px in [('train',1.5),('val',3.5),('test',5.5)]:
 pillars=[{'name':'p','x':px,'y':0.,'z':1.5,'size':[.4,.4,3.]}]
 poses={0:({'x':px,'y':0.,'z':2.},0.),1:({'x':-10.,'y':0.,'z':2.},0.),2:({'x':-10.,'y':0.,'z':2.},-45.),3:({'x':-10.,'y':0.,'z':2.},45.)}
 rows=[]
 for i,a in enumerate([0,2,1,1,3,1]*4):
  pb,yaw=poses[a]
  assert ad.plan_action(pb['x'],pb['y'],yaw,pillars)==a
  rows.append({'step':i,'action_expert':a,'action_executed':a,'collision':False,'depth_rgb_gap_ns':0,'pose_rgb_gap_ns':0,'displacement_m':0.25 if a==1 else 0.,'dyaw_deg':{2:10.,3:-10.}.get(a,0.),'pose_before':pb,'yaw_before':yaw})
 rows.append(dict(rows[0],step=24,collision=True,depth_rgb_gap_ns=999999999))
 oracle=[{'step':i,'action_start_ns':i*10**9,'action_end_ns':(i+1)*10**9,'pose_gt':{'x':i*.25,'y':0.,'z':2.}} for i in range(24)]
 manifest={'split':split,'teacher':'plan_action','planner':'plan_action','geometry_version':sm.SPLIT_SALT,'teacher_version':ad.TEACHER_VERSION,'actions_ref':'actions.jsonl','oracle_ref':'oracle.jsonl','num_steps':25,'requested_steps':25,'validity':{'camera_verified':True,'physics_verified':True,'episode_executed':True}}
 with patch.object(ad,'load_json',side_effect=lambda p:{'pillars':pillars} if p.endswith('scene.json') else manifest),patch.object(ad,'load_jsonl',side_effect=lambda p:rows if p.endswith('actions.jsonl') else oracle),patch.object(ad.os.path,'exists',return_value=True):
  facts.append(ad.episode_facts('/memory/'+split))
checks=ad.audit(facts)
print('3 memory episodes, each actions=25/oracle=24/manifest num_steps=25')
print('facts steps=',sum(f['num_steps'] for f in facts),'expected actions=75, discarded unsafe tail rows=3')
for item in checks:print(item['name'],item['status'],item['detail'])
print('all checks pass=',all(item['status']=='PASS' for item in checks),'checks=',len(checks))
assert sum(f['num_steps'] for f in facts)==72
assert len(checks)==11 and all(item['status']=='PASS' for item in checks)
PY
```

原审核真实输出：

```text
3 memory episodes, each actions=25/oracle=24/manifest num_steps=25
facts steps= 72 expected actions=75, discarded unsafe tail rows=3
manifest_validity PASS 0/3 集 validity 不全 true（含已排除 0 集）
time_pairing PASS max_gap=0.0 ms，超 50 ms 的步 0/72
action_window PASS sim_dt mean=1.000s min=1.000 max=1.000 cv=0.0%
forward_motion PASS 0/12 个稳态前进步位移不足 0.12×dt（0.0%，阈值 1%）
transition_lag PASS 偏航接前进的 24 步，位移中位数为标称的 83%（阈值 20%）
yaw_direction PASS 0/24 个偏航步转向与标签相反或过小（跳过 0 个反向接续步；另有 0 步 dyaw=null 未验）
class_share PASS 每类占比低于 5%：无；split 内整类缺席：无
safe_collision PASS 0/12 个 SAFE 步带 collision
geometry_version PASS uavrgb.geom.v3
split_geometry PASS 交集为 0（train=1, val=1, test=1）
label_recompute PASS 0/72 步复算标签与原标签不一致（0.0%，阈值 0%；跳过 0 步（缺位姿 0、缺真值 0、旧几何版本 0）；另有 0 步来自非 plan_action 的集未参与）
all checks pass= True checks= 11
```

## 附录 C：R03/R04/R05/R06 使用的完整 C++ 探针

下面是原审核成功编译并运行的完整源码，直接调用仓库 CPU/CUDA 函数。`numeric` 模式验证 R04/R05/R06 和 ReLU 观察；`setup/inspect` 配合真实训练 CLI 验证 R03。`setup` 仅在临时目录写合成数据，不修改项目样本。

标签全部改为 0 是为了让验证集 macro_f1=0.25 可确定复现；它不是模型学习实验。源 best 是否有这个分数会在 CLI eval-only 阶段实际核对。

<!-- REVIEW_CPP_PROBE_BEGIN -->
```cpp
#include "synthetic_dataset.hpp"
#include "uavrgb/trainer.hpp"
#include "uavrgb/ops.hpp"
#include "uavrgb/softmax_cuda.hpp"
#include "uavrgb/nn.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
using namespace uavrgb;
int main(int argc,char**argv){
 const std::string mode=argv[1];
 if(mode=="numeric"){
  Tensor l=Tensor::zeros({2,4});const std::vector<int> y{0,1};
  for(bool gpu:{false,true}){
   for(float scale:{1.f,3e38f}){
    std::vector<float>w(4,scale);auto g=gpu?softmax_ce_grad_weighted_cuda(l,y,w):softmax_ce_grad_weighted(l,y,w);
    float loss=gpu?softmax_ce_loss_weighted_cuda(l,y,w):softmax_ce_loss_weighted(l,y,w);
    std::cout<<"weighted backend="<<(gpu?"CUDA":"CPU")<<" scale="<<scale<<" loss="<<loss<<" grad=[";
    for(size_t i=0;i<g.numel();++i)std::cout<<(i?",":"")<<g[i];std::cout<<"]\n";
   }
   Tensor a=Tensor::from_flat({2},{3e38f,3e38f});float norm=gpu?global_grad_norm_cuda({&a}):global_grad_norm({&a});
   float coef=gpu?clip_grad_norm_cuda({&a},5.f):clip_grad_norm({&a},5.f);
   std::cout<<"clip backend="<<(gpu?"CUDA":"CPU")<<" norm="<<norm<<" coef="<<coef<<" values=["<<a[0]<<","<<a[1]<<"]\n";
   Tensor nx=Tensor::from_flat({1},{std::nanf("")}),dy=Tensor::from_flat({1},{1.f});auto dx=gpu?relu_backward_cuda(nx,dy):relu_backward(nx,dy);
   std::cout<<"relu_backward backend="<<(gpu?"CUDA":"CPU")<<" x=NaN dy=1 result="<<dx[0]<<"\n";
  }
  double err=std::max(0.0,std::fabs(0.0-std::nan("")));
  std::cout<<"test_parity_helper expected=0 actual=NaN reported_max_error="<<err<<"\n";return 0;
 }
 namespace fs=std::filesystem;const fs::path r=argv[2];
 if(mode=="setup"){
  fs::create_directories(r/"source");auto ds=uavrgb_test::write_synthetic_uavr((r/"dataset.uavr").string(),4,8,8,2,7);
  std::fstream f(ds.path,std::ios::binary|std::ios::in|std::ios::out);f.seekp(48+4*3*8*8);char labels[4]={0,0,0,0};f.write(labels,4);f.close();
  auto m=make_cnn(8,8,3);for(Tensor*p:cnn_params(m))std::fill(p->data(),p->data()+p->numel(),0.f);m.fc.b[0]=10.f;
  auto s=make_train_state(m,0.001f);s.best_macro_f1=.25f;s.epoch=1;save_checkpoint((r/"source/ckpt.bin").string(),m,s);save_checkpoint((r/"source/ckpt_best.bin").string(),m,s);
  std::cout<<"setup true_best_macro_f1=0.25 source_best_exists="<<fs::exists(r/"source/ckpt_best.bin")<<"\n";
 }else if(mode=="inspect"){
  auto m=make_cnn(8,8,3);auto s=make_train_state(m,0.001f);load_checkpoint((r/"resumed/ckpt.bin").string(),m,s);
  std::cout<<"resumed_best="<<s.best_macro_f1<<" last_exists="<<fs::exists(r/"resumed/ckpt.bin")<<" best_exists="<<fs::exists(r/"resumed/ckpt_best.bin")<<" source_best="<<fs::exists(r/"source/ckpt_best.bin")<<"\n";
 }
}
```
<!-- REVIEW_CPP_PROBE_END -->

### C.1 独立构建和重新执行命令

下面是为复核整理的执行入口，原审核用的是独立 `build/audit-20261030-1713`；此处改成新 `build/deepseek-review`，避免用到历史二进制。**本次整理没有重新执行下面的新入口**，原始成功命令和输出仍见附录 D/E。

运行前遵守仓库约定：采集窗口独占，不在真实采集进行时叠加构建/CTest/GPU 任务。CUDA 复现必须以当下 `nvidia-smi -L` 的结果决定能否运行；没有 GPU 时明确标“未跑 CUDA”，不要用 CPU 输出代替。

```bash
source uav_rgb/env.sh
nvidia-smi -L
cmake -S "$UAV_RGB_ROOT/src" -B "$UAV_RGB_ROOT/build/deepseek-review"   -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$UAV_RGB_ROOT/build/deepseek-review" --parallel 8
```

保留本 MD 在仓库的同名位置，然后运行以下入口。它从本文件提取上述 C++，在仓库外创建一次性工作目录；输出日志保留，临时源码/二进制/合成 run 自动清理。请逐行核对实际结果，脚本退出 0 仅表示命令运行成功，不自动表示六项结论全部成立。

```bash
source uav_rgb/env.sh
"$UAV_RGB_PYTHON" - <<'PY'
from pathlib import Path
from datetime import datetime
import os
import re
import subprocess
import tempfile

root = Path(os.environ['UAV_RGB_ROOT'])
report = root / 'docs/CODE_REVIEW_DEEPSEEK_2026-10-30.md'
match = re.search(
    r'<!-- REVIEW_CPP_PROBE_BEGIN -->\n```cpp\n(.*?)\n```',
    report.read_text(encoding='utf-8'), re.S,
)
if match is None:
    raise RuntimeError('报告中的 C++ 探针代码块没有找到')
build = root / 'build/deepseek-review'
archive = (Path(os.environ['DATA_ROOT']) / 'uav_rgb' /
           ('deepseek-recheck-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
archive.mkdir(parents=True)
commands = []

def run(args, name):
    result = subprocess.run(args, capture_output=True, text=True)
    output = result.stdout + result.stderr
    (archive / (name + '.log')).write_text(output, encoding='utf-8')
    commands.append({'name': name, 'argv': [str(x) for x in args],
                     'exit_code': result.returncode})
    print(name, 'EXIT=', result.returncode)
    print(output)
    result.check_returncode()
    return result

with tempfile.TemporaryDirectory(prefix='probe-', dir=archive) as directory:
    temporary = Path(directory)
    source = temporary / 'probe.cpp'
    executable = temporary / 'probe'
    source.write_text(match.group(1), encoding='utf-8')
    run(['g++', '-std=c++17', '-O2', '-I' + str(root / 'src/include'),
         '-I' + str(root / 'src/tests'), str(source),
         str(build / 'libuavrgb_core.a'), '-L/usr/local/cuda-12.8/lib64',
         '-lcudart', '-ldl', '-lpthread', '-lrt', '-o', str(executable)], 'build')
    run([str(executable), 'numeric'], 'numeric')
    case = temporary / 'resume-case'
    train = str(build / 'uavrgb_train')
    run([str(executable), 'setup', str(case)], 'setup')
    run([train, '--data', str(case / 'dataset.uavr'), '--eval-only',
         str(case / 'source/ckpt_best.bin'), '--out', str(case / 'source-eval'),
         '--batch', '2', '--split', 'val'], 'source-best-eval')
    run([train, '--data', str(case / 'dataset.uavr'), '--resume',
         str(case / 'source/ckpt.bin'), '--out', str(case / 'resumed'),
         '--epochs', '1', '--batch', '2', '--lr', '0.001'], 'resume-cli')
    run([str(executable), 'inspect', str(case)], 'inspect')
    print('resume case best exists:', (case / 'resumed/ckpt_best.bin').exists())

import json
(archive / 'commands.json').write_text(
    json.dumps(commands, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('日志保留在：', archive)
print('临时源码、二进制、合成数据和 run 目录已自动清理。')
PY
```

## 附录 D：C++ 数值探针的原始输出

以下来自原审核 `numeric-probe.log`，生产 CPU/CUDA 路径都实际运行过。完整 probe 编译参数保存在外部 `probe-command-and-source.json`。没有使用 `-ffast-math` 重新解释 NaN 行为。

```text
weighted backend=CPU scale=1 loss=1.38629 grad=[-0.375,0.125,0.125,0.125,0.125,-0.375,0.125,0.125]
weighted backend=CPU scale=3e+38 loss=1.38629 grad=[-0,0,0,0,0,-0,0,0]
clip backend=CPU norm=inf coef=0 values=[0,0]
relu_backward backend=CPU x=NaN dy=1 result=1
weighted backend=CUDA scale=1 loss=1.38629 grad=[-0.375,0.125,0.125,0.125,0.125,-0.375,0.125,0.125]
weighted backend=CUDA scale=3e+38 loss=1.38629 grad=[-0,0,0,0,0,-0,0,0]
clip backend=CUDA norm=inf coef=0 values=[0,0]
relu_backward backend=CUDA x=NaN dy=1 result=1
test_parity_helper expected=0 actual=NaN reported_max_error=0
```

## 附录 E：跨目录 resume CLI 的原始输出

以下来自 `resume-best-probe.log`，包含源 best 复评、续训与检查的退出码。临时路径是当时真实路径，验证后已清理；重新执行请使用附录 C 创建的新临时目录，不要期待旧路径仍存在。

```text
setup EXIT=0
setup true_best_macro_f1=0.25 source_best_exists=1

source-best-eval EXIT=0
{"schema":"tinyvln.metrics.v1","event":"validation","step":0,"loss":0.000136246934,"action_accuracy":1,"split":"val","macro_f1":0.25,"balanced_accuracy":0.25,"confusion":[2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
eval-only val: ce=0.0001 acc=1.0000 macro_f1=0.2500 bal_acc=0.2500 recall=[1.000 0.000 0.000 0.000] n=2

resume-cli EXIT=0
epoch 1 dev   ce=0.0001 acc=1.0000 macro_f1=0.2500 bal_acc=0.2500 recall=[1.000 0.000 0.000 0.000]
完成：2 样本，0.8 s，2.4 样本/秒，输出 /root/rivermind-data/uav_rgb/current-code-review-20261030-1713/tmp/review-probe-kcklfaxa/resume-case/resumed

inspect EXIT=0
resumed_best=0.25 last_exists=1 best_exists=0 source_best=1

REPRO_CONFIRMED: successful cross-directory resume preserved verified best score 0.25 but did not create best checkpoint
```

## 附录 F：关键源码 SHA256 与证据索引

下列 hash 属于原审核工作树。整理时对这些条目再次与磁盘内容比较，全部一致；未来修复后 hash 改变属于正常现象，应据 diff 判断问题是否已修，不能继续机械引用原行号。

| 文件（相对项目根） | SHA256 |
| --- | --- |
| `sim/tools/collect.py` | `75cb92c795ee7993f251f3b3aad4afd8dc0c8083016a4f4adabcb996642be186` |
| `sim/tools/audit_dataset.py` | `34c24f5bda851a651ee887d1d4448029b9da34ccd0c9ff0fc81c580f11ac3a99` |
| `tools/preprocess.py` | `c88d89ee4e0e623c24ac515871aef1fe5a16ffadcfda7513fcc547fe1f7f9aab` |
| `src/trainer.cpp` | `f5a846c863e88124773a0384bfd2519d8e206f53aa6c3f7c81fd3443e297086a` |
| `src/train_main.cpp` | `f321806319efb08fa62b02ec3b99fc97b21de90b102b06f7e58ed98285dde1ea` |
| `src/softmax_cuda.cu` | `83cca38de4d3ec821ec45abce6424d4690e0689e551ff023e284f98d30e80f7f` |
| `src/ops_weighted.cpp` | `7cb17b7adf731ad3ff2d412dc981c1fb3df6c233ded2dd4d2ef806f909d5ca86` |
| `src/nn_cuda.cu` | `842b4068212f3e6c5e7a9c4fc9ec191847a1731661e0e1ae9ccb027a94f89182` |
| `src/nn_cpu.cpp` | `ba89d7db305ddef9e9e23705e70cea018c335b93fb3eb9d5efe56b6065532728` |
| `src/tests/test_conv_cuda.cpp` | `86827809374df438ee43e1131cc704d11a377527edf379238b0bddf359f39e91` |
| `src/tests/test_linear_cuda.cpp` | `86ae3150cb60b2bc242bc78b13429e4252094f3a8f6038ed0e678e36dce22f81` |
| `src/tests/test_nn_cuda.cpp` | `9a04970e17531151a5e6fb12587dca4d66da795d03a8340342bb73c9de4d1a1c` |
| `src/tests/test_weighted_loss.cpp` | `b7a27db0ed7b9bc1d5bac46e3da14321deaf8a1ca801b73075b401cd56d92a2a` |

外部目录 `/root/rivermind-data/uav_rgb/current-code-review-20261030-1713/` 中的主要证据：

| 文件 | 用途 |
| --- | --- |
| `source-before.json`、`source-after-check.json` | 原审核输入快照与内容未变化核对 |
| `git-status-before.txt`、`implementation-before.diff` | 当时未提交改动与代码 diff |
| `check.log` | 独立构建、20 项 CTest 和 Python/UI 自检原文 |
| `python-probes-command-and-source.json` | R01/R02 完整源码、命令和退出码 |
| `collect-delayed-inference.log` | R01 内存复现输出 |
| `audit-truncated-oracle.log` | R02 内存复现输出 |
| `probe-command-and-source.json` | C++ 完整源码及实际编译/执行 argv |
| `numeric-probe.log` | R04/R05/R06 与 ReLU 观察的 CPU/CUDA 输出 |
| `resume-best-probe.log` | R03 CLI 的逐步输出 |
| `t5-count-integrity.json` | 599 集实际数据的条数/step 清点 |
| `binary-sha256.json`、`evidence-sha256.json` | 原审核二进制与证据文件校验 |
| `probe-initial-build.log` | 最初临时探针少传 lr 导致的编译错误，后已纠正探针；不是产品 bug |

## 13. 本次整理做了什么 / 没做什么

- **做了什么**：将六项发现合并为一份可转交文档；补充独立复核任务、真实合同、源码摘录、完整复现、原始输出、修复连带影响、回归验收和源码 hash。
- **没做什么**：没有修改业务源码、正式测试、共享任务板或真实数据；原因是用户本次要给 DeepSeek 独立复核的材料。
- **没重新跑什么**：没有新跑 GPU/训练/Gazebo/性能检查；原因是整理现有已保存证据，本文对新执行入口仅做静态核对，不把历史运行冒充本次重跑。
- **未代替用户做什么**：没有把代码或文档发送给 DeepSeek 或任何外部服务，没有 commit/push；本文件交由用户自行转交。
- **风险与遗留**：六项仍为本次审核者的结论和初评，建议 DeepSeek 依据真实代码逐项同意、修正或否定；R01 的真实 Gazebo 影响量、R03 的跨目录合同选择仍需后续验证/明确。
