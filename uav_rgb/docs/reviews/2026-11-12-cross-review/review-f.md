# 交叉审核 F：测试的质量（main `97c4228`，只读）

- 审核者：Claude Opus 5.5（subagent），2026-11-12 00:37 起。任务书 `docs/briefs/2026-11-11-cross-review-brief.md` §1 的 F 行。
- 草稿：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/review-f/`（变异副本 `mut/`、扫描脚本、提议用例 `proposed/`）。
- 对照的上一轮报告：`docs/reviews/2026-11-10/arch.md`（B3-d 门禁）、`src.md`（D2、D3、D6、D9）。已写过的问题只记「仍未修」，不重复展开。
- 「已复现」= 我在草稿副本里跑出了现象；「读代码」= 由代码直接可判定、没有运行；「推断」= 写明成立条件。

## 1. 结论

1. **3DGS 线上，喂给 prev-action 学生的「上一步动作」不是真正执行的动作（已复现，且没有任何测试）。** 路线护栏或 DAgger β 改写动作后，
   `StudentAdapter` 不传 `prev_action`，学生缓存的是自己上一步的输出；训练打包取的是 `action_executed`。Gazebo 线是对的。
   W1-9 的航向码与 E4 的路线方位码都走这一列，端到端学生一进 3DGS 闭环就会带着这个训练与推理不一致。
2. **部署路径 `run_infer_loop` 没有一条带参考值的测试（读代码）。** 唯一的测试用 8×8 方形输入，只查确定性与「换输入输出会变」，
   HWC→CHW 写成转置、双帧顺序写反都不会红；`check_input_parity.py` 把 C++ 这一步用 Python 重写了一遍，没有执行 C++，也不在门禁里。
3. **打包的「标签 = 教师标签」与「图像 i ↔ rgb_<step>.png」在工作树门禁里没人守（已复现）。** 把标签改成 `action_executed`、
   把帧号错一位，`--python-only` 门禁里的全部 Python 测试照样通过；DAgger 数据（学生执行、教师标注）正是端到端路线 E1、E3 的训练数据。
   我写了一个用例，在 HEAD 上通过，在这两个变异上都变红。

## 2. 发现清单（按严重程度）

### F1【会让数据与闭环悄悄不一致，已复现】3DGS 学生的 prev-action 在护栏 / DAgger 改写后取错

- 位置：`sim/3dgs/collect_3dgs.py:169-176`（`StudentAdapter.__call__` 调 `self.student.act(msg)`，不传 `prev_action`）；
  `sim/3dgs/guard.py:98-116`（`OracleGuardedPolicy` 改写内层动作）；`sim/3dgs/collect_3dgs.py:211-216`（`DaggerPolicy` 以概率 β 改执行教师动作，
  `--beta` 见 `:561`）；`sim/tools/student.py:122,133`（不传时用自己上一步的输出）。
  对照：训练口径 `tools/preprocess.py:263` 取上一行 `action_executed`；Gazebo 线 `sim/tools/collect.py:957` 显式传 `prev_action=prev_executed`。
- 现象：内层学生恒出 FORWARD、护栏把它改成 YAW_LEFT 时，真正执行 `[2, 2, 2]`，下一步喂给学生的 prev 是 `[1, 1]`（应为 `[2, 2]`）；
  DAgger β=1、教师恒 YAW_RIGHT 时，执行 `[3, 3, 3]`，喂的是 `[1, 1]`。
- 为什么是缺陷：prev-action 模型在训练里看到的是执行动作，在 3DGS 闭环里看到的是自己的输出，两者在护栏触发或 β 生效的每一步之后都不同。
  `test_student.py:122-124` 只测了「调用方显式给 prev_action 时以它为准」，没有测 3DGS 的调用方给没给。
- 影响：已有结论不受影响。11-10 的 3DGS 护栏实验（`experiments/2026-11-10-bearing-guard-3dgs/run.sh:49`）用的是 calswa 权重，
  `tools/average_ckpt.py:210` 只放行 `in_c=3, text_dim=0, prev_dim=0` 的 ckpt，所以那批学生没有 prev 输入（由代码推出，未逐个读 ckpt 头核实）。
  受影响的是后续：W1-9 的航向码复用 prev 这一列（任务卡 W1-9 第 2 条）；E4 的「图像 + 路线方位码」若同样走这一列；3DGS DAgger 第 1 轮 β=0.5。
- 复现：`cd review-f && PYTHONDONTWRITEBYTECODE=1 TMPDIR=$PWD/tmpdir <venv>/python proposed/test_3dgs_student_prev_action.py`，
  两个用例在 HEAD 上都红（`[1, 1] != [2, 2]`、`[1, 1] != [3, 3]`）。最小脚本 `repro_3dgs_prev_action.py` 输出同样的结论。
- 最小修法：`StudentAdapter` 记下上一步真正执行的动作（`run_episode` 拿到最终动作后回写，或让 `OracleGuardedPolicy` / `DaggerPolicy`
  在改写后调用 `adapter.note_executed(a)`），下一步 `act(msg, prev_action=...)`。航向码落地时同理传 `heading_code(net_yaw)`，与 Gazebo 线同一个函数。
  上面的提议用例转绿即可。

### F2【训练与推理一致性没有带参考值的测试，读代码】`run_infer_loop` 的 HWC→CHW、帧序、prev 字节

- 位置：`src/infer.cpp:63-80`（逐帧 HWC→CHW，CHW 里通道 0..2 是当前帧、3..5 是上一帧）；`src/tests/test_infer_io.cpp:29-137`；
  `tools/check_input_parity.py:58-63`；`tools/check.sh` 全文没有调用 `check_input_parity.py`。
- 现象：`test_infer_io` 的模型是 `make_cnn(8, 8, ...)`，断言只有：同输入两次输出一致、action 在 0..3、probs 有 4 个、文本不改变输出、截断抛异常、
  换上一帧或 prev 字节输出会变。没有一处把输出与参考前向比。`check_input_parity.py` 的 B 路在 `:63` 用 `hwc.transpose(2, 0, 1)` 代替 infer.cpp，
  `--infer-bin` 不给时不执行 C++；它在 HEAD 上通过（0.6 s，合成帧 RGB/RGBA/BGR 三种全一致），但不在门禁里。
- 为什么是缺陷：部署只有这一个网络，这一步是训练分布到部署输入的唯一接口。下列错误都能通过现有测试：
  `infer.cpp:74` 的源下标写成转置（8×8 方形下形状不变）；`:68` 的帧序写反（「上一帧变了输出会变」仍成立）；通道顺序写成 BGR。
  训练侧的拼法在 `src/trainer_dev.cpp:36-45`（`[I_t ; I_{t-1}]`），两边一致是靠注释约定，没有测试把它们绑在一起。
- 复现或验证：提议用例 A（`proposed/cpp_cases.cpp` 第一段）：16×8 非方形、双帧 + prev，host `forward`（`model.hpp:97`）作参考，probs 差 < 1e-4；
  再把两帧对调，断言输出与参考不同。接入后把 `infer.cpp:74` 改成转置下标、或把 `:68` 的帧序反过来，应当变红。我没有编译运行（审核规则不构建、不占 GPU）。
- 最小修法：加用例 A；`check.sh` 在两个 Python 循环之后加一行跑 `tools/check_input_parity.py`（全量模式下再带 `--infer-bin`、`--ckpt` 跑一次真推理对拍）。

### F3【会让训练数据悄悄变坏，已复现】打包的标签来源与图像对齐在 `--python-only` 门禁里没人守

- 位置：`tools/preprocess.py:251`（`action = int(row["action_expert"])`）、`:285`（`png = rgb_%05d.png % step`）；
  `tools/make_synthetic_episode.py:68-69`（合成集 `action_executed == action_expert`）；`tools/check.sh:45-66`（`read_dataset --selfcheck` 在 `--python-only` 分支之外）。
- 现象：在草稿副本里做两处变异，跑 `tools/test_preprocess_split.py`、`test_dagger_gz_pipeline.py`、`test_dagger_pipeline.py`、`test_e1_pipeline.py`：
  - 标签改成 `row.get("action_executed", row["action_expert"])`：全部通过（SURVIVED）。
  - 帧号改成 `step + 1`（下一帧存在时）：全部通过（SURVIVED）。
- 为什么是缺陷：DAgger 集（`executed_by=student`）里执行动作与教师标签不同，标签取错就是「学生学自己」，DAgger 聚合的结论（11-10 聚合臂 1/20、E1 的 2×2）都建立在
  这一行上。标签来源的变异在全量门禁里同样不会红：全量门禁唯一回读标签的是 `read_dataset --selfcheck`，而合成集的执行动作等于标签（推断，未跑全量）。
  帧号错位在全量门禁的 `--selfcheck` 里会红（逐样本像素约定，`read_dataset.cpp:131-160`），但公共规则 §5 要求工作树里的卡只跑 `--python-only`，
  错位会一直留到主会话合并后的全量检查。
- 复现：`python mutate.py pre_label_executed`、`python mutate.py pre_png_off_by_one`（草稿目录，约 45 s 一个）。
- 最小修法：加提议用例 `proposed/test_preprocess_label_alignment.py`（约 90 行，1 s）：把 ep1 改成学生执行、执行动作逐步与标签不同，打包后逐样本回读 UAVR，
  断言标签等于 `action_expert`、像素等于 `load_frame(rgb_<step>.png)`。实测：HEAD 上 OK；`pre_label_executed` 上红（`1 != 0 : 样本 0：标签必须是教师标签`）；
  `pre_png_off_by_one` 上红（`样本 0：像素必须来自 rgb_00000.png`）。放进 `tools/` 就自动进门禁的 `tools/test_*.py` 循环。

### F4【会让训练悄悄变坏，读代码】水平翻转增强的标签与 prev-action 互换没有测试

- 位置：`src/trainer_dev.cpp:75-97`（`make_aug`：翻转样本 YAW_LEFT↔YAW_RIGHT）、`:48-65`（`flip_side_inputs`：prev-action 同样互换、深度目标按列镜像）；
  `src/tests/test_augment_dev.cpp:71-84`（只查像素 kernel，`c=3`）。`grep -n flip src/tests/*.cpp` 除 test_augment_dev 的像素检查外没有命中。
- 为什么是缺陷：这两个函数在匿名命名空间里，删掉其中任何一处互换，全部 CTest 照样通过（读代码判定：没有测试经过它们的标签输出）。
  `--augment flip` 在仓库里出现 3 次、`--augment all` 1 次（`grep -rhoE -- '--augment[ =]+[a-z,]+' experiments tools sim docs`）。
  W1-9 把航向码写进 prev 这一列（码 2 = 偏左、码 3 = 偏右），翻转时 2↔3 恰好也是航向码该做的镜像，这个正确性是巧合，没有测试锁住。
- 验证：提议用例 B（`proposed/cpp_cases.cpp` 第二段）：「左半亮 → YAW_LEFT、右半亮 → YAW_RIGHT」的无噪声集，开 `--augment flip` 训练，训练准确率应 > 0.95；
  互换被删时同一种画面带两种标签，准确率封顶约 0.5。prev 版：画面全灰、标签 = prev（2/3 交替），开 flip 加 `--prev-action` 应能拟合。未编译。
- 最小修法：加用例 B；或把 `make_aug` / `flip_side_inputs` 移出匿名命名空间做单元测试。

### F5【会让训练悄悄变坏，读代码】双帧训练读的是不是 `prev_index` 那一帧，没有测试

- 位置：`src/trainer_dev.cpp:43`；`src/tests/test_overfit.cpp:98-111`（双帧变体的标签由当前帧亮度决定）；`test_device_train_step.cpp:105-172`（直接喂张量，不经数据读取）。
- 为什么是缺陷：`:43` 把 `ds.aux.prev_index.at(idx[i])` 写成 `idx[i]`（上一帧 = 当前帧），`test_overfit` 照样 > 0.95，没有别的测试经过这一行。
  双帧是 E2 的候选输入（计划 §10.4）。
- 验证：提议用例 C：图像类别取随机序列，标签改写成上一帧的类别，噪声置 0；双帧模型应能拟合，单帧对照不超过「按当前类取多数」的上限加 0.05。未编译。
- 最小修法：加用例 C。

### F6【门禁覆盖，读代码 + 已运行】`check.sh` 不跑的测试

`tools/check.sh:157-169` 只遍历 `sim/tools/test_*.py`、`tools/test_*.py` 与 3 个 `ui/` 测试。没进门禁的：

| 测试 | 规模 | 现状 |
| --- | --- | --- |
| `sim/3dgs/test_*.py` | 9 个文件、160 个 `def test_`（AST 计数） | arch.md B3-d 已报，仍未修。E4 的到点与碰撞判据（`test_collect_eval.py`）在这里，端到端线越往 3DGS 走越要紧 |
| `experiments/*/test_*.py` | 9 个文件、152 个 | 冻结实验的汇总脚本；计划 §0 第 11 条记 11-11 合并后人工跑过全过 |
| `teacher/test_*.py`、`rl_teacher/test_core.py` | 6 个文件 | arch.md B3-d 已提 |
| `tools/check_input_parity.py` | 不叫 test_，0.6 s | 本次在 HEAD 上跑通（见 F2） |
| `tools/test_event.cpp` | EventSchema 自检，文件头写「不接入 CMake」 | 本次用 g++ 编到草稿目录跑通，`RESULT: PASS` |
| `sim/cf_sitl/gz/tests/test_flatten_state.cc` | gz 插件状态展平 | `sim/cf_sitl/CMakeLists.txt:51` 的 `foreach(sub proc selftest tests)` 不含 `gz`，check.sh 构建不到；全量日志里 cf_sitl 的 CTest 是 9 项，不含它 |

另外两条与门禁有关：
- `--python-only` 不回读 UAVR（见 F3）；工作树里的卡只跑这一档。
- 全量日志（`$D/tmp/phase12-20261110/main/full-check.log`，HEAD `af5292f`，与 `97c4228` 只差文档）共 336 s，其中 CTest 34 项 32 s；把 `sim/3dgs` 的测试加进来，arch.md 实测约 65 s。

### F7【健壮性，已复现】`check.sh` 的「几何 split 门禁」负例测的不是它声称的那道门

- 位置：`tools/check.sh:106-121`（两集改成同一个 `geometry_gid`，期望打包失败）；`tools/preprocess.py:188-190`（split 复算门）与 `:388-390`（同 gid 跨 split 检查）。
- 现象：按 check.sh 的改法造数据跑 preprocess，报错是 `:188` 的「split=val 与 geometry_gid=check.shared_geometry 按记录策略应属的 split=train 不一致」，
  根本没走到 `:388`。把 `:388` 的检查整段关掉（变异 `pre_no_gid_uniqueness`），`test_preprocess_split.py` 也全部通过。
- 影响：`:388` 只在 split 复算被跳过时才是唯一防线（诊断 manifest，例如 `collect_3dgs` 调用方强行指定 split 时写的 `diag-manual-split`）；
  同一几何通常同一个 `scene_id`，`:382` 的 scene_id 检查多半也能挡住，所以排在后面。
- 最小修法：负例改用诊断策略（两集 `split_policy` 设成诊断名、带 `--allow-diagnostic-split`），并断言 stderr 含「几何组」。

### F8【低，已复现】闭环判据的边界没有锁

- 位置：`sim/tools/eval_closed_loop.py:509-510`（走廊 `y_max > half_width`）、`:545`（过线 `max_x >= goal_x`）。
- 现象：把 `>` 改 `>=`、`>=` 改 `>`，`test_eval_metrics`、`test_eval_summary`、`test_eval_requested_set`、`test_eval_sim09v2`、`test_eval_provenance` 全部通过。
- 影响：位姿是连续量，恰好落在边界上的概率低，排最后。与 `collect.py` 的 goal-stop 判据是否同为 `>=` 我没有核对（角度 A 的范围）。
- 修法：`test_eval_metrics` 加两行：`y_max == 3.0` 不算越界、`max_x == goal_x` 算过线。

### 上一轮仍未修的测试缺口（不重复展开）

`src.md` 的 D2（train_step 不记梯度范数，`test_trainer.cpp` 里 `grad_norm` 出现 0 次）、D3（warmup / 余弦 / AdamW 衰减没有数值测试）、
D6（`--max-grad-norm` 续训不继承，`trainer.hpp` 的 `RunConfigSnapshot` 仍无此字段）、D9（conv 的 dx 没有任何差分检查，
`test_conv_cuda.cpp:70` 只做 CPU 对 CUDA 互拍）。这几条在 `97c4228` 上都还在。

## 3. 最要紧的 5 个函数与缺的用例

| 函数 | 缺的用例 | 文件 | 状态 |
| --- | --- | --- | --- |
| `collect_3dgs.StudentAdapter`（与 `OracleGuardedPolicy`、`DaggerPolicy` 的组合） | 改写后的下一步，prev = 执行动作 | `proposed/test_3dgs_student_prev_action.py` | 已运行：HEAD 上 2/2 红（暴露 F1） |
| `src/infer.cpp: run_infer_loop` | 非方形、双帧 + prev，与 host 参考前向逐值对拍；帧序对调必须改变输出 | `proposed/cpp_cases.cpp` A 段 | 未编译 |
| `tools/preprocess.py: load_episode`（经 `write_dataset` 落盘） | DAgger 集标签 = `action_expert`；样本 i 的像素 = `rgb_<step>.png` | `proposed/test_preprocess_label_alignment.py` | 已运行：HEAD 绿，两个变异红 |
| `src/trainer_dev.cpp: make_aug / flip_side_inputs` | 开 flip 后「哪半边亮决定左右」仍可拟合；prev 版同理 | `proposed/cpp_cases.cpp` B 段 | 未编译 |
| `src/trainer_dev.cpp: read_u8_batch` | 标签只由上一帧决定时双帧可拟合，单帧不行 | `proposed/cpp_cases.cpp` C 段 | 未编译 |

## 4. 查过但没有发现问题的地方

- 142 个 Python 测试文件按脚本方式运行时都会执行自己的用例：AST 扫描所有顶层函数，没有「定义了但没人调用」的测试函数
  （唯一命中是 `test_scene_course.py` 的 `tearDownModule`，那是 unittest 钩子）。脚本 `review-f/dead_tests.py`。
- 没有恒真断言：没有 `assert (x, msg)`、`assert 真常量`、吞掉 `AssertionError` 的 `except`。`assertTrue(x, y)` 第二参数不是字符串字面量的 44 处逐条看过，
  都是把变量当失败消息，不是误写的 `assertEqual`。脚本 `review-f/weak_asserts.py`。
- `src/tests/` 的 34 个 `.cpp` 全部在 `src/CMakeLists.txt` 注册，全量日志里 34/34 通过；`ctest --no-tests=error` 防住了空注册表。
- 闭环判据的主路径：去掉成功判据里的走廊条件被 `test_eval_sim09v2` 抓到；Wilson 的 z 改成 1.645 被 `test_eval_metrics` 抓到；
  stuck 与 contact 的区分、requested / attempted / skipped 三个分母、SPL 的截断都有逐值断言。
- prev-action 取执行动作（打包侧）：`test_preprocess_split.py:262-280` 把执行动作改成与标签不同并断言侧车里的值，变异 `pre_prev_expert` 的设计我没跑，但这条断言直接覆盖它。
- `student.py` 的双帧 payload 顺序、首帧口径、`reset()`、越界 prev 拒绝：`test_student.py:93-133` 逐字节断言。
- `average_ckpt.py`：只改 `fc.b[FORWARD]` 的逐字节断言、二分精度、数据集 sha 绑定、不支持的输入拒绝，都有测试；参考权重复现是可选项（`UAVRGB_REPRO_AVGCKPT=1`）。
- `test_teacher_policy.py` 的 torch 冒烟在全量日志里真的跑了（日志 1044 行「torch smoke: PASS」），不是静默 SKIP。

## 5. 没有查的地方与原因

- C++ 侧的变异没有实际编译运行：审核规则不构建、不占 GPU。F2、F4、F5 的结论来自读代码，提议用例 A、B、C 未编译；接入时每条按段尾的「变异核对」先红后绿。
- `sim/3dgs/test_*.py` 与 `experiments/*/test_*.py` 在 `97c4228` 上没有跑（计划 §0 第 11 条记 11-11 合并后全过，本次未核实）。
- `test_collect.py` 的动作窗口、时间戳、迟到判定语义没有逐条评估断言强度：属于角度 A 的核心逻辑，避免重复。
- `ui/` 的测试、`sim/cf_sitl` 测试的判据阈值没有看。
- 全量 `check.sh` 没有跑（规则禁止）；关于全量门禁能否抓到某个变异的说法，都标了「推断」。
- 个人规范要求的仓库内 `docs/worklog/` 过程文档与 `progress.md` 没有写：本次只读，不改仓库任何文件。

## 6. 验证到哪一步

| 命令或检查项 | 结果 |
| --- | --- |
| `review-f/dead_tests.py`（AST 扫 142 个测试文件） | 无死测试函数 |
| `review-f/weak_asserts.py` | 无恒真断言；44 处 `assertTrue(x, 变量)` 人工核过 |
| `mutate.py`（基线，草稿副本） | 9 个相关测试文件通过；`test_course_summary.py` 依赖 `experiments/`，副本里缺，移出 |
| `mutate.py pre_label_executed` | SURVIVED |
| `mutate.py pre_png_off_by_one` | SURVIVED |
| `mutate.py pre_no_gid_uniqueness` | SURVIVED |
| `mutate.py eval_success_no_corridor` | KILLED（`test_eval_sim09v2`） |
| `mutate.py eval_wilson_z` | KILLED（`test_eval_metrics`） |
| `mutate.py eval_corridor_ge`、`eval_goal_gt` | SURVIVED |
| `proposed/test_preprocess_label_alignment.py` | HEAD：OK（1 s）；两个变异上：FAILED |
| `proposed/test_3dgs_student_prev_action.py` | HEAD：FAILED 2/2（`[1, 1] != [2, 2]`、`[1, 1] != [3, 3]`） |
| `tools/check_input_parity.py`（无 `--infer-bin`） | PASS，0.6 s |
| `g++ tools/test_event.cpp` 并运行 | `RESULT: PASS (all checks)` |
| check.sh 几何 split 负例的实际拒绝原因 | 由 `preprocess.py:188` 的 split 复算门拒绝，不是 `:388` |
| 所有验证进程 | 串行、`nice -n 15`、`PYTHONDONTWRITEBYTECODE=1`、`TMPDIR` 在草稿目录；仓库 `git status` 无改动 |
