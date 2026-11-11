# 交叉审核 C：C++/CUDA 训练与推理栈

- 对象：主仓库 `/root/claude-code-backup-20261028/code/uav_rgb` main @ `97c4228`，只读。审核人：Claude Opus 5.5，2026-11-12。
- 范围：`src/`、`src/include/`，以及与它直接相接的条件输入链（`sim/tools/student.py`、`tools/preprocess.py` 的侧车、`tools/average_ckpt.py`、`sim/3dgs/collect_3dgs.py` 与 `guard.py` 的学生适配层）。
- 草稿与输出：`$D/tmp/phase12-cards/review-c/`（`$D = /root/rivermind-data/uav_rgb`）。验证只用了今晚的构建目录 `$D/tmp/build-check-main-1111` 里现成的二进制。
- 记号：【已复现】= 跑出了结果；【已核实·读码】= 读到了确定的代码路径，没有运行；【推断】= 需要条件成立，写了怎样验证。

## 0. 结论

1. 交付配方（末 10 轮平均加 FORWARD 先验校准，公共规则 §2）对端到端路线要用的每一种条件模型都不可用：`tools/average_ckpt.py` 遇到 `prev_dim>0`、`in_c=6`、`text_dim>0` 一律拒绝校准。卡 E2E-1 的第二步只计划补 `prev_dim>0`，双帧与文本没有计划；另外，带条件的模型只用一个全局 δ 校准，可能把各条件值下的先验校歪（推断）。
2. 3DGS 线的学生适配层从不把「实际执行的动作」交给 prev-action 模型：DAgger 的 β 混合与路线护栏改写动作之后，推理端拿到的仍是学生自己的上一步输出，训练侧车记的却是执行动作。E3、E4 计划在 3DGS 上做带条件的 DAgger，这会让 prev-action（以及以后的航向码）在训练与推理之间静默错位。
3. 条件字节的语义（上一步动作，还是航向码）不进 checkpoint，也不进推理进程的 hello；客户端默认喂「模型自己的上一步输出」。正确喂法只能靠 checkpoint 外面的一串文件（`config.json` → 数据集路径 → meta → sha256）去推，3DGS 线没有这道闸门，板端也没有这些文件。

C++ 核心本身这一轮没有发现会算错的地方：双帧加 prev-action 的「训练评估路径」与「推理行协议路径」在真实 checkpoint 上逐格一致（§3 第 1 条）；compute-sanitizer 的 memcheck、racecheck、initcheck 在覆盖文本 FiLM、双帧、prev-action、辅助头的两个测试上都是 0 错误。`src/` 自上一轮审查（`c85a1ff`）以来没有任何提交，上一轮 D2–D11 全部仍未修（`git log c85a1ff..97c4228 -- src` 为空）；D1 由 `tools/average_ckpt.py`（`55cba8d`）在工具层补上，训练器本身没变。

## 1. 发现清单（按严重程度）

### C1【高】交付配方对条件模型、双帧模型、文本模型全部不可用；单一 δ 对条件模型可能不对

- 位置：`tools/average_ckpt.py:207-212`（`(in_c, text_dim, prev_dim) != (3, 0, 0)` 直接报错）；`tools/average_ckpt.py:128-132`（δ 加在 `fc.b[FORWARD]` 上，对所有条件值相同）；公共规则 `docs/briefs/2026-11-10-phase12-common.md` §2「交付权重：末 10 轮权重平均加 FORWARD 先验校准」。
- 现象：端到端路线 E1（航向码走 `--prev-action` 通路）、E2（双帧、历史码加宽）、E5（文本 token）训出来的模型，都过不了 `average_ckpt.py` 的校准；只能 `--no-calibrate`，而工具自己的文件头写着「只平均不校准比 ckpt_best 还差（闭环 20/60 对 27/60）」。
- 为什么是缺陷：上一轮已证明不校准时末期权重的类先验在摆（`src.md` B.4），交付权重的固定口径就是这两步。端到端的每一个臂如果退回 `ckpt_best` 或只平均，与基线比的就不是同一种交付配方，E1 的 2×2 与 E2 的判据会混进「交付配方不同」这个因素。
- 计划覆盖情况：卡 E2E-1 第二步只写了「支持 `prev_dim > 0`」。双帧（`in_c=6`，需要按侧车 `prev_index` 取上一帧）与文本（需要逐集 `--text`，而推理进程一次只能带一条指令，`infer_main.cpp:48-58`）没有计划。
- 单一 δ 的问题【推断】：带航向码时，码 2/3（已偏了两步以上）的帧标签里 YAW 占比与码 0 的帧差别很大；一个全局 δ 只对齐总体 FORWARD 占比，可能让码 0 的帧更爱前进、码 2/3 的帧该转不转。成立条件：各条件值下的 FORWARD 标签占比差异大，且模型在各条件值下的偏差方向不同。验证：校准后按条件值分组，分别比较预测 FORWARD 占比与标签 FORWARD 占比；差超过约 0.05 就说明需要逐条件校准（可解为对 `film.b` 的 β 段加一组偏移，或换成逐条件的阈值）。
- 最小修法：E2E-1 第二步实现时，把「按侧车取 prev_action 字节」与「按侧车 `prev_index` 取上一帧」一起做（双帧的行协议是先当前帧后上一帧，见 §3 第 1 条的对拍脚本，可直接复用）；校准完输出逐条件值的占比表进 `<out>.json`。文本模型暂时显式不支持并在文档写明。
- 状态：拒绝路径【已核实·读码】；单一 δ 的影响【推断】。

### C2【高】3DGS 线：prev-action 模型拿到的是学生自己的输出，不是实际执行的动作

- 位置：`sim/3dgs/collect_3dgs.py:169-176`（`StudentAdapter.__call__` 调 `self.student.act(msg)`，不传 `prev_action`）；`sim/3dgs/collect_3dgs.py:211-217`（`DaggerPolicy` 以概率 β 把动作换成教师动作，学生缓存不知道）；`sim/3dgs/guard.py:104-115`（`OracleGuardedPolicy` 改写动作，内层学生缓存不知道）；`sim/tools/student.py:122`（`prev_action=None` 时用 `_last_action`，即模型自己的输出）。对照：训练侧车的 prev-action 取上一步的 `action_executed`（`tools/preprocess.py:239-263`），3DGS 采集记的 `action_executed` 是改写之后的动作（`collect_3dgs.py:362`）。Gazebo 线是对的：`sim/tools/collect.py:957` 传 `prev_action=prev_executed`，`:1816` 取的是护栏改写后的执行动作。
- 现象：在 3DGS 上，β>0 的 DAgger 步与护栏改写步之后的下一步，prev-action 模型的条件输入与「该步数据进训练集时的条件值」不同。
- 为什么是缺陷：E3（带闸门的 DAgger）与 E4（3DGS 室内的条件模仿）都计划在 3DGS 上用带条件输入的学生（计划 §10.4）。这类错位不报错，只会让闭环数字变差，而且变差的程度随 β 与护栏触发率变化，臂之间不可比。W1-9 的航向码若接到 3DGS，同一个口子会把错误的航向码喂进去（航向码按「已执行动作」累计）。
- 复现：单帧、无条件模型不受影响（`prev_dim=0` 时缓存不进 payload）。最小用例：在 `sim/3dgs/test_*` 里用一个假的 `StudentPolicy`（记录每次 `act` 收到的 `prev_action`），包进 `DaggerPolicy(beta=1.0)` 跑两步，断言第二步收到的是第一步的教师动作；现在会收到 `None`（随后由缓存换成学生输出）。
- 最小修法：`StudentAdapter` 存一份 `last_executed`，由 `run_episode` 在派发后回写（或让 `DaggerPolicy`、`OracleGuardedPolicy` 在改写后调用 `student.note_executed(a)`），`act(msg, prev_action=last_executed)`；集首为 `None`。约 10 行加上面的测试。
- 状态：【已核实·读码】，没有运行（3DGS 模块的导入链要 torch 与 gsplat，没有为此起进程）。

### C3【中高】条件字节的语义不在 checkpoint 与 hello 里，客户端默认喂「上一步动作」

- 位置：`src/checkpoint.cpp:174-175`（v6 只记 `prev_dim` 一个整数）；`src/infer_main.cpp:69-72`（hello 只报 `prev_action_dim`）；`src/trainer_dev.cpp:150-151`（`--prev-action` 时 `prev_dim = num_classes`，条件列直接取侧车 `prev_action`，不管它装的是什么）；`sim/tools/student.py:106-125`（不传值就喂模型自己的上一步输出）；`src/class_weights.cpp:242`（`config.json` 的 `data` 按命令行原样写，相对路径不转绝对，例：`$D/tmp/agent4-student-p2/run-tp10-two_prev_depth/config.json` 里是 `agent4-student-p2/ds-depth/dataset.uavr`）。
- 现象：卡 W1-9 把航向码写进侧车的 prev-action 列，训练器与 checkpoint 看到的仍是 `prev_dim=4`。拿到一个 `prev_dim=4` 的 checkpoint，无法从它本身知道该喂上一步动作还是航向码。
- 现有的缓解：W1-9 的工作树（`wt/p12-w1-9`，未提交，未合入 main）在 `sim/tools/nav_cond.py:94-151` 加了闸门，沿 `config.json`（或 `average_ckpt.py` 的 `<ckpt>.json` 里的 run 目录）→ 数据集 meta 的 `cond` → sha256 去推，读不到时只告警（`unverified`）。这条链断的情形：训练时 `--data` 给了相对路径；数据集被挪走或删掉（数据盘 94% 已用，清理是迟早的事）；3DGS 线（`eval_3dgs.py`、`collect_3dgs.py`）没有接这道闸门；板端部署只有权重文件。
- 为什么是缺陷：喂错条件不会报错，网络照样出动作。E1 的预注册主比较 M1、M2 看的正是「有无航向码」，喂错一侧就会把预测证伪的方向弄反。
- 最小修法：checkpoint 升 v7，在 `prev_dim` 后追加 `u8 cond_kind`（0 = 上一步动作，1 = `heading_code.v1`，其余保留），训练器从数据集 meta 的 `cond` 取值写入；hello 追加 `"cond_kind"`；`student.py` 在 `cond_kind != 0` 且调用方没传值时报错，不再回落到 `_last_action`。旧 v6 读回 0，行为不变。测试：`test_checkpoint_versions` 加 v6 读入与 v7 往返；`test_student` 加「cond_kind=1 且 `prev_action=None` 时报错」。`config.json` 的 `data` 改写成绝对路径（1 行）。
- 状态：缺字段【已核实·读码】；会被喂错【推断】，条件是评测或采集走了没有闸门的路径，或闸门返回 `unverified` 而运行者没看告警。

### C4【中】条件码的维度与镜像规则写死成「4 个动作」

- 位置：`src/trainer_dev.cpp:150-151`（`prev_dim = cfg_.prev_action ? classes : 0`，不可配）；`src/aux_sidecar.cpp:104`（侧车 prev-action 值必须小于 `num_classes`）；`src/trainer_dev.cpp:47-59`（翻转增强把条件值 2↔3 互换，规则就是 `YAW_LEFT`↔`YAW_RIGHT`）；`src/infer.cpp:51,80-83`（每步 1 字节）；`sim/tools/student.py:76-78`（只接受 `prev_action_dim` 为 0 或 4）。`model.hpp:47` 的上限 `kMaxPrevDim = 64` 与这几处不一致。
- 现象：计划 §10.4 的 E2 写「历史码加宽（条件维度可配，对应第十一阶段 C6）」。按现在的代码，加宽必须同时改训练器、侧车校验、翻转规则、推理协议与客户端，不是配置。
- 为什么是缺陷：最容易出错的一处是翻转。航向码 v1 碰巧与动作同号（码 2 = 偏左，码 3 = 偏右，镜像后互换正好对），所以 W1-9 开 `--augment flip` 不会错；码一加宽（比如多档偏角、或加进度位），翻转仍只交换 2 与 3，其余档位不镜像，训练数据会静默出现「画面镜像了、条件没镜像」的样本。
- 最小修法：E2 动手前，在 `TrainConfig` 加 `prev_dim`（缺省等于类数，逐位不变）与一张镜像表（从数据集 meta 读，缺省为动作的 2↔3），`aux_sidecar.cpp:104` 改为与 `prev_dim` 比较，`student.py` 的白名单改为与 hello 对齐。开 flip 而 meta 没有镜像表时拒绝启动。
- 状态：【已核实·读码】。

### C5【中】深度辅助头接在 FiLM 之后，辅助损失会把条件调制往「与条件无关」的方向拉

- 位置：`src/include/uavrgb/model.hpp:37-39`（辅助头读 FiLM 之后的第三层特征）；`src/model_dev_aux.cu:53,79-82`（前向用 `fc_input()`，即 `p3mod_`；梯度叠进 `dflat_` 后再经 FiLM 反传）；`src/model.cpp:246-251,263-272`（host 路径相同）。
- 现象：深度分带目标只由图像决定。辅助头看到的是 `(1+γ(cond))·y + β(cond)`，同一张图在不同条件值下特征不同，深度损失的梯度会经 FiLM 反传，压 γ、β 随条件变化的幅度，或者逼辅助头学会抵消它。
- 为什么是缺陷【推断】：E2 计划把「历史码加宽」与「几何辅助头」一起上，且判据不带护栏。辅助损失与条件通路争同一组 FiLM 参数（`--aux-lambda` 默认 0.5），可能让条件码的作用变弱，结论会被读成「历史码没用」。成立条件：λ 不小、条件值在训练集里分布不均。验证：同 seed 训两臂，辅助头分别接 FiLM 之前（`p3_`）与之后，比较「只改条件字节时动作翻转的帧比例」（§3 第 1 条的脚本把 prev 字节换掉即可）与 `film.W` 各列的范数差。
- 最小修法：辅助头改读 `p3_`，梯度在 `film_backward` 之后叠到对 `p3` 的梯度上（`model_dev.cu:261-262` 两行换序，`model_dev_aux.cu` 改输入指针，host 路径同步），加结构开关并写进 v7 checkpoint；缺省保持现状，逐位不变。
- 状态：【推断】。

### C6【中低】平均后的权重做 `--eval-only` 时，碰撞过滤静默回落到命令行缺省；`average_ckpt.py` 自己总是剔除碰撞，不看契约

- 位置：`src/train_main.cpp:200-218`（ckpt 同目录没有 `config.json` 时只告警，按命令行值，缺省 `drop_collision=false`）；`tools/average_ckpt.py:154-171`（`read_val` 无条件剔除碰撞步，不读契约里的 `drop_collision` 位）；契约里本来就记着这一位（`src/checkpoint.cpp:167`）。
- 现象：`average_ckpt.py` 的输出放在 run 目录之外（它的用法就是这样）。拿它跑 `uavrgb_train --eval-only` 而忘了带 `--drop-collision`，val/test 会包含碰撞步，与训练时、与工具校准时的样本集不同，只有一行 stderr 告警。卡 E2E-1 的第二步正要拿 `--eval-only` 与工具逐帧对拍。
- 最小修法：`train_main.cpp` 读不到 `config.json` 时改读 checkpoint 契约的 `drop_collision`（v5 起都有），与命令行不一致时拒绝；`average_ckpt.py` 的 `read_val` 按契约位决定是否剔除。
- 状态：【已核实·读码】。

### C7【低】打包期 `--drop-collision` 会造出「上一帧 = 自身、prev ≠ SAFE」的训练专有组合

- 位置：`tools/preprocess.py:258-263`（prev-action 从包含被丢步的完整 `actions.jsonl` 取）；`tools/preprocess.py:858-861`（被丢步之后的那一帧 `prev_index` 指向自身）。
- 现象：被丢的碰撞步后面那一帧，双帧输入是「当前帧复制两份」（首帧口径），条件却是真实的上一步动作。推理时首帧永远是 prev = SAFE，这个组合不会出现。
- 影响：只在打包时加 `--drop-collision` 才发生（训练器的 `--drop-collision` 不受影响，上一帧照常现读）。帧数少，影响小。修法：这种帧的 `prev_action` 同时置 0，或干脆把它也标成集首。
- 状态：【已核实·读码】。

### C8【低】辅助头的分带语义与文本词表不进 checkpoint

- 位置：`src/include/uavrgb/checkpoint.hpp:4-24`（只有 `aux_classes`，没有分带上界与包络半径）；`src/model_dev_aux.cu:46-51`（只校验目标小于 `aux_classes`）；`src/include/uavrgb/text.hpp:3-11`（只约定词表大小 22 与长度 9，词到 id 的映射在 `sim/tools/instruction.py`，不在任何指纹里）。
- 影响：用分带上界不同的数据集做 `--eval-only` 或续训，深度指标与损失按另一套语义算，不报错；词表重排后旧文本模型静默换义。辅助头部署时丢弃，词表在 E5 之前不动，所以排在低。修法：随 C3 的 v7 一起追加分带上界（7 个 f32）与词表 sha256（32 B）。
- 状态：【已核实·读码】。

### C9【低，测试缺口】双帧顺序与 prev 字节没有「训练路径 = 推理路径」的回归测试

- 位置：`src/tests/test_infer_io.cpp:112-118` 只断言「换上一帧或换 prev 字节会改变输出」，把当前帧与上一帧在推理端对调，这条测试照样通过；`tools/check_input_parity.py` 只覆盖单帧（`student.py:35-40` 的注释）。
- 我做的对拍（§3 第 1 条）证明现在两边一致，而且对调后结果会变，所以这个测试有区分力，值得固定下来。修法：把草稿 `p2_parity.py` 的逻辑写成 `test_eval_only` 里的一个用例：合成 8×8 双帧加 prev-action 小数据集，训 1 轮，`--eval-only` 的混淆矩阵与按行协议喂 `run_infer_loop` 的混淆矩阵相同。
- 状态：一致性【已复现】；缺测试【已核实·读码】。

## 2. 上一轮报告各项的状态

`src/` 自 `c85a1ff` 没有提交（`git log --oneline c85a1ff..97c4228 -- src` 输出为空）。

| 上一轮编号 | 状态 |
| --- | --- |
| D1 权重不停稳 | 工具层已补（`tools/average_ckpt.py`，`55cba8d`）；训练器没变；对条件模型不可用，见 C1 |
| D2 不记梯度范数 | 未修（`model_dev.cu:301` 返回值仍在 `trainer.cpp:555` 被丢弃） |
| D3 warmup/余弦/AdamW 无数值测试 | 未修 |
| D4 种子共用、增强种子固定、RNG 不入盘 | 未修（`model.cpp:82`、`trainer.cpp:507`） |
| D5 末批不满照常更新 | 未修（`trainer.cpp:518-519`） |
| D6 `--max-grad-norm` 续训不继承 | 未修（`trainer.cpp:215-252` 仍不读） |
| D7 没有不依赖 CUDA 的整网前向 | 未修；卡 E2E-P0 的 PyTorch 复刻会部分替代（不在 main） |
| D8 未接入的设备层与 `BatchReader` | 未修（`CMakeLists.txt:37,51-58` 仍在） |
| D9 conv 的 dx 没有差分检查 | 未修 |
| D10、D11 架构与行数 | 未变 |

## 3. 查过、没有发现问题的地方

1. 双帧加 prev-action 的训练与推理口径一致【已复现】。用 P2 冒烟 run 的 `ckpt_best.bin`（`$D/tmp/agent4-student-p2/run-tp10-two_prev_depth/`，in_c=6、prev_dim=4、aux=8）与它的数据集 `ds-depth`，草稿 `review-c/p2_parity.py` 按行协议（先当前帧、后侧车 `prev_index` 指向的上一帧，再 1 字节侧车 prev-action）逐帧喂 `build-check-main-1111/uavrgb_infer`，test 146 帧的混淆矩阵 `[16,8,0,0,0,99,0,0,1,6,3,1,0,5,1,6]` 与训练器结束时用同一权重写进 `metrics.jsonl` 的 test 混淆矩阵逐格相同（`review-c/p2_parity_test.out`）。把两帧对调后变成 `[17,7,0,0,2,97,0,0,0,9,1,1,0,7,1,4]`（`p2_parity_test_swap.out`），说明对拍能区分顺序。旁记：对调只改了约 10/146 帧的动作，这个 3 轮小模型几乎不用上一帧（属于角度 H 的问题，样本太小，不下结论）。
2. 内存与同步【已复现】：compute-sanitizer 12.8 在 `test_infer_io`（memcheck 加 leak-check、initcheck）与 `test_device_train_step`（memcheck、racecheck、initcheck）上都是 0 错误、0 泄漏、0 竞争；后者覆盖文本 FiLM、双帧、prev-action、辅助头的 host↔device 对拍（日志 `review-c/san_*.log`）。
3. FiLM 前反向：host（`model.cpp:197-226,273-313`）与设备（`model_dev_film.cu`）公式一致；dγ = Σ dy'·y、dβ = Σ dy'、dy = dy'·(1+γ)；`cond_gather_kernel` 的下标在 `text_dim=0` 或 `prev_dim=0` 时不读未上传的缓冲（initcheck 也没报）；`text_demb_kernel` 按 token 顺序求和，确定。
4. 反向顺序：fc → 辅助头梯度叠进对 FiLM 输出的梯度 → FiLM 反传 → 池化与卷积（`model_dev.cu:257-278`），与辅助头读 FiLM 输出的前向一致。
5. 增强：每样本参数按 `nc / in_c` 取（`augment_dev.cu:25`），双帧两张图共用同一组平移、翻转与光度；翻转同时互换标签、prev-action、深度目标的列（`trainer_dev.cpp:47-65,90-95`）；文本模型禁用翻转。
6. 推理进程：`istream::read` 读满或 EOF，截断帧报错；prev 字节越界报错（`infer.cpp:61-83`）；文本 id 越界在 `infer_main.cpp:36` 与 `infer.cpp:28-31` 两处拦。
7. checkpoint 读取：文件里的维度只拿来与模型比，不用于分配（`checkpoint.cpp:43-50,286-292`）；非有限值、尾部多余数据、版本越界都拒绝；`average_ckpt.py:read_ckpt` 有同样的长度检查。
8. 侧车读取：长度、两个 sha256、`prev_index` 与 prev-action 的范围逐项校验（`aux_sidecar.cpp:77-109`）。
9. 下标与溢出：conv 与 linear 的 kernel 用 int64 算元素下标并在 host 侧按 INT_MAX 设上限；conv 反向工作区（batch 128 时第三层约 470 万个 float）按形状算；`DeviceCnn` 构造中途分配失败会释放已分配的段（`model_dev.cu:97-105`）。
10. 确定性：卷积 dW/db 固定顺序归约、池化反传窗口不重叠、FiLM 与文本梯度固定顺序；唯一的乱序归约是全局范数的 double `atomicAdd`（`nn_cuda.cu:90-95`），对 10 万个平方和的 double 累加，舍入差传到 float 范数的概率按量级估在 1e-8 以下，上一轮实测的逐位复现与此相容。
11. Gazebo 线的条件输入：`collect.py:957` 传执行动作，`:1816` 取护栏改写后的动作；每集新起 `StudentPolicy`，集首缓存为「上一帧 = 自身、prev = 0」，与打包口径一致。
12. 图像预处理：Gazebo 训练与闭环都是 PIL `BILINEAR`（`preprocess.py:645`、`student.py:117`）；3DGS 训练帧是无损 PNG，闭环把同一个渲染数组交给 `student.py` 走同一次缩放；归一化只有 `/255.0f`，训练与推理共用 `augment_dev.cu:21`。
13. `--eval-only` 的结构以 checkpoint 为准（`trainer_eval.cpp:86-103`），不会被命令行的结构开关改掉。

## 4. 没有查的地方与原因

- `ops_soft.cpp`、`soft_targets.cpp`、`ops_spatial.cpp` 的 CPU 参考：上一轮查过，这一轮源码未变，没有重读。
- `run_owner.cpp`、`metrics_json.cpp`、`event.hpp`、`src/device/`、`batch_reader.cpp`：与训练推理数值无关，或是上一轮 D8 已列的未接入代码。
- `src/tests/` 只读了 `test_infer_io.cpp` 与 `test_device_train_step.cpp` 的相关段；测试质量是角度 F。
- `proto/` 与卡 E2E-P0：不在 main 上。W1-9 只读了工作树里 `nav_cond.py` 的闸门函数，其余改动没有审（未提交，不在本次范围）。
- 没有训练，没有起 Gazebo 与 3DGS；C2 没有运行复现（3DGS 模块的导入需要 torch 与 gsplat 环境）；C5 的实验没有做（需要训练两臂）。
- 板端 int8 与 NPU：上一轮 E 节已盘点，这一轮没有新信息。

## 做了什么

- 读了任务书、`CLAUDE.md`、公共规则 §1–§4、计划 §0 与 §10、上一轮 `docs/reviews/2026-11-10/src.md`；读了 `src/` 的模型、设备路径、FiLM、辅助头、训练器、评估、推理、checkpoint、数据集与侧车读取、CUDA 算子，以及条件输入链上的 Python 文件。
- 写了 1 个对拍脚本（`review-c/p2_parity.py`，35 行），跑了 2 次；用 compute-sanitizer 跑了 2 个现成测试共 5 次。每次单进程、`nice -n 15`、钉卡 `GPU-b544b42e-…`，都在 10 秒内结束。

## 没做什么（含原因）

- 没改仓库、没提交、没写仓库里的 worklog 与 `progress.md`：任务要求只读，只写本报告与草稿。
- 没训练、没跑 `tools/check.sh`、没重新构建：任务硬性边界。
- C2、C5 没有运行复现：前者需要 3DGS 环境起进程，后者需要训练，都超出本次允许的验证规模；给了最小用例与实验设计。

## 验证到哪一步

| 命令或检查项 | 结果 |
| --- | --- |
| `git log --oneline c85a1ff..97c4228 -- src` | 空（src 未变） |
| `p2_parity.py … run-tp10-two_prev_depth/ckpt_best.bin … test` | 混淆矩阵与训练器 test 记录逐格相同 |
| 同上加 `swap` | 混淆矩阵改变（对拍有区分力） |
| `compute-sanitizer --tool memcheck --leak-check full test_infer_io` | PASS，0 错误，0 泄漏 |
| `compute-sanitizer --tool initcheck test_infer_io` | PASS，0 错误 |
| `compute-sanitizer --tool memcheck / racecheck / initcheck test_device_train_step` | 均 PASS，0 错误，0 竞争 |
| C2 的 3DGS 适配层错位 | 未运行，读码确认 |
| C5 辅助头位置的影响 | 未运行，推断 |
