# 无人机视觉语言导航审计与执行记录

- 时间：2026-11-26 21:23 CST
- 范围：`code/uav_rgb` 主线、`code/ALVLN` GitHub 镜像、`code/geoecpt_vln` 独立工程、数据盘已有 VLN 证据
- 11 月 26 日完成只读审计；11 月 27 日的源码修复、真实模型与固件在环实验见文末执行更新。

## 做了什么

- 读取 `README.md`、`progress.md`、`docs/HANDOFF.md`、`docs/PLAN_2026-11-13_vln.md`、`docs/EVAL_STANDARD.md`、最新实验计划和证据闭环。
- 沿 `sim/3dgs/run_vln.py`、`grounder.py`、`collect_3dgs.py`、`vln.py`、`video.py`、`ui/studio.py` 和 `train.py` 追踪语言、学生、控制、安全层、评测和录像入口。
- 核对 GitHub 镜像。`uav_rgb` HEAD 为 `2c772f6`，`ALVLN` `main` 为 `bf1983d`，远端 `origin/main` 对齐，镜像指针为 `2c772f6f88cd1ca95f802f8c9c71390773776676`。已提交主线内容逐文件一致。
- 核对工作树。当前存在 375 个已跟踪删除、39 个修改、11 个未跟踪文件，累计约 `+2,035/-88,769` 行。它们属于既有工作状态，本轮没有重置、清理、暂存、提交或推送。
- 复现语言短语解析、阶段圆末点相交和交换汇总字段问题，并读取已有逐集 JSON/CSV、视频和控制结果。
- 使用 RTX 2080 Ti 环境完成 Python 3DGS 相关回归、C++ 构建和 CTest。

## 11 月 26 日审计结论

- 任务合同应是“输入一条自然语言指令，无人机只用指令、RGB、允许的历史和机载状态，找到指定目标实例或区域，按指令阶段飞行，无碰撞并自主 STOP”。拿取、放置等操作任务另列，不混入当前导航成功率。
- 现行学生并不直接消费原始语言。`run_vln.py` 的 VLM worker 读取指令和最新 RGB，输出离散 bearing、阶段、near-goal 和词语，`StudentAdapter` 再将其作为 15 维条件送入学生。R2 checkpoint `text_dim=0`，因此只能称分层语言条件局部控制诊断。
- `VlmBeliefTracker` 保存四帧，但 `VisualLanguageGrounder._encode_latest_frame_b64` 只编码最后一帧，worker 收到的 `frames_b64` 长度始终为 1。文本 memory 是先前输出的摘要，不是视觉历史。
- 非 VLM 更新窗在 `run_vln.py:98-101` 可能把方位强制为居中。已有 `839934` 记录显示 42 次 VLM 更新、40 个间隔窗被置为居中，导致持续前进、碰撞或超时。
- `grounder.py` 对“Fly to the wall lamp and stop/hover”仍会把结束词带入目标匹配，中文“飞到壁灯并悬停”也不能稳定得到壁灯目标。目标解析必须先于新训练修复。
- VLM 模式有 grounding 时，`StudentAdapter` 用 `confirmed_target` 覆盖 C++ 独立 STOP head。STOP head 的输出只被记录，不参与执行，当前不能宣称它已验证。
- `train.py:737-748` 汇总只输出原始 `goal_object`，省略 `supplied_goal_object` 和 supplied metrics。交换任务和 zero 任务的汇总会误导读者，底层 `episodes.jsonl` 才是当前可信来源。
- 交换入口默认先 `one_per_route`，在过早截断的池中会漏掉同起点、同预算、同句式的异目标。旧报告也无法让读者看清缺少搭档和飞行失败各占多少。全请求失败计 0 是既有评测约定。
- 当前语言闭环最新证据是 `839934` 正确/交换四集 `0/4` 成功、`3/4` 碰撞、`1/4` 错误 STOP。最新裁图和控制闭环仍为 `0/4`，不能宣称算法提升。
- Studio 的 3DGS 训练后入口仍调用 `eval_3dgs` 并读取 `eval.json`，没有语言 goal function 和 `stop_on_request`。VLN 主线使用 `run_vln.py` 和 `vln_eval.json`，两条入口不能混为同一结果。
- 当前录像 HUD 没有 fresh/inherited/search/inferred 来源、目标证据、执行目标和 STOP 来源。终态长文本在 640 px 格子中会被裁切。跟随镜头的平滑不能用于证明机体控制平顺。
- 目前没有真实飞行的姿态、角速度、推力或电机日志。不能把抖动归因于 CF-Drone PID。已有仿真倾角 RMS 约 2 度，证据更支持动作窗内策略切换和 stale bearing 问题。
- `code/geoecpt_vln` 是独立的 AerialVLN/AeroBelief 工程，其 benchmark `DONE` 为 `rc=1`，主模型 Val-Seen-Eval SR 为 0，不能并入室内 3DGS 主线结果。

## 后续执行顺序

1. **P0 合同和入口闸门**：修正目标短语解析，建立 `LanguageTask` 的目标、参照物、关系、阶段和终止动作字段。swap 必须预检同起点、同 split、同句式、同阶段数、同预算和异目标，zero 的 supplied goal 必须为 null。汇总同时显示原目标、输入目标和各自指标。为每个问题先补失败回归。
2. **P1 证据链**：让每个决策步绑定 step、仿真时间、RGB 时间戳、原始指令、当前阶段、目标/参照证据、可见性、遮挡、方位、置信度、证据来源、动作提议、执行动作、安全层改写和 STOP 来源。VLM 更新缺失只继承带 TTL 的有效证据，不伪造 fresh center。
3. **P2 局控门禁**：固定语言模块，先对 direct bearing、真值局部子目标、地图教师和盲飞做同 RGB/深度/位姿/动力学配对。分别量首转时刻、前进率、动作切换/米、左右反转、累计偏航、速度、碰撞前窗和 STOP。先比较 0.25/0.5 秒动作窗，局部安全结果不替代完整路线成功。
4. **P3 目标证据和停止**：用目标/参照区域、跨帧实例关联和可见性状态建立 SEARCH、CANDIDATE、TRACKED、NEAR、STOP_READY、LOST 状态。STOP 同时要求正确实例证据、接近证据、连续确认和安全状态。将视觉证据 STOP、学生 STOP head 和安全 SAFE 做独立消融。
5. **P4 数据与训练**：在局控门禁通过后采集同一初始观测的 correct/swap/zero，加入画外、遮挡、低纹理、错参照和同类实例 hard negatives。先 DAgger 纠偏，再比较瞬时 RGB、短历史、证据门控、独立 STOP 和安全层，冻结场景、种子、预算和 checkpoint 规则。
6. **P5 正式比较**：训练、开发、留出按场景和路线簇隔离，至少三种子。主表按实际输入目标实例报告 SR、OSR、NE、SPL、nDTW、SDTW、碰撞、超时、错误 STOP 和停止来源，控制附表报告动作切换/米、反转、偏航、姿态和 jerk。公开方法先做同接口本地适配，官方数字单列背景。
7. **P6 可视化和清理**：Studio 的 VLN 入口统一到 `run_vln.py`/`vln_eval.json`。录像显示原始/有效方位、来源、证据、提议/执行/改写和终态，按格宽换行。清理前生成删除 manifest，逐项全仓引用搜索、Git 可恢复核验、定向测试和摘要哈希，禁止批量删除当前工作树。

## 没做什么

- 没有修改 Python、C++、CUDA、固件、训练参数、评测逻辑或可视化实现。
- 没有启动训练、长时间闭环、VLM GPU 任务、数据采集或真机飞行。
- 没有删除实验结果、日志、脚本、缓存、权重、Markdown 或当前工作树中的既有删除项。
- 没有提交或推送 Git。用户明确要求本阶段先读代码和整理想法，后续提交必须按主题精确暂存。
- 没有把 teacher、G0、reactive、OpenFly、AeroBelief 或旧 M1 结果写成当前端到端语言收益。

## 验证到哪一步

| 项 | 结果 |
|---|---|
| `python -B -m unittest -q sim.3dgs.test_vln sim.3dgs.test_grounder sim.3dgs.test_video` | 122 项通过 |
| `python -B -m unittest -q sim.3dgs.test_collect_eval sim.3dgs.test_eval_split` | 38 项通过 |
| `python -B -m unittest -q sim.3dgs.test_student_inputs sim.3dgs.test_3dgs_student_prev_action sim.3dgs.test_renderer` | 运行 33 项，31 项通过、2 项跳过 |
| `cmake --build build/check -j4` | 56/56 目标完成 |
| `ctest --test-dir build/check --output-on-failure -j4` | 42/42 通过，24.12 s |
| GitHub 镜像和远端核对 | `ALVLN/main` 与 `origin/main` 干净，镜像源指向 `uav_rgb` HEAD |

## 风险与遗留

- 当前 Python 回归没有覆盖 VLM 多帧实际传输、非 fresh 方位来源、合法 swap 预检、supplied 汇总、终点圆相交和 HUD 超长换行。
- 多阶段指标同时保留无序命中和按序成功，论文不能只报 `stage_sr`。
- 当前工作树的 C++ 构建和测试是绿的，但大规模未提交差异仍未经过主题审查，不能把工作树视为可发布快照。

## 执行更新 2026-11-27

### 做了什么

- 语言目标解析保留中英文终止动作前的目的地，区分目标与参照物，并保留对否定和降落的拒绝。单帧沿用原提示。四帧为显式消融，按旧到新输入，要求以最新帧判断。
- 修正多阶段航线中线段进入目标圆的几何判断。交换指令先在完整候选池预检，再启动 VLM 与渲染器。原有全请求口径包含配对失败，另报合法配对的输入目标成绩；其运行异常仍计入分母，空指令没有输入目标。
- G3 改用 840865 的已登记路线 tr_74/7、tr_87/8。同一起点与朝向，各 106 步，两端相距 3.164 m。用原始场景重建的文件与 11 月 23 日登记逐字节一致，SHA-256 为 `e44572862284b326506d68ee098ab962841d4bde713e4b49ad89fef4390450dd`。六个登记场景的 12 条有向配对键逐项一致。这是跨房间开发诊断，两个原始 annotation 均为 train。
- 移除 tracker 和 reactive 控制器里“转过就认定目标在正前”的两处推断。每步记录 RGB、仿真时间、原始及有效方位、证据年龄、动作提议和实际执行。暂选最多继承一个动作窗，继承的近距判断不能触发停机。阶段切换清上一个目标的视觉历史；DAgger 接管也记录真正的停止来源。
- 当前录像在可用证据旁标出本步或上次模型报告的地标，旧稀疏记录缺步显示 `unrecorded`，长终态按格宽缩放。[四帧八步录像](</root/rivermind-data/uav_rgb/exp-vln-frame-history-20261127-mdRWGA/flight-four-prefix/sage3d_839934_tr_42_tr_69/tr_42_0/evidence.mp4>)

### 验证到哪一步

- 冻结教师机载图像的四条非空请求，每个输入窗各跑一次，另有两个空指令检查。可见物体的匹配原句在两张图上，单帧报最终目标 2/2，四帧 1/2。模型推理都已在 GPU0 实际运行。原始探针中的延迟从编码之后开始计；当前代码把 JPEG 编码也计入调用延迟，两种日志不可混算。
- 839934 已登记壁灯起点，R2、种子 421、125 个决策步。实际单帧闭环结果为 SR 0/1、OSR 0/1、无碰撞、未停止、末端误差 3.627 m。125 步全是右转，累计绝对偏航 2143.57°，63 次新观测、62 步继承。动作未切换，仍然在原地转圈；评估“抖”应同时看动作、姿态和净位移。
- 旧策略在同一起点第 8 步碰撞，复用这段时长比较动作。当前单帧与四帧均为右转八步，空指令为左四、右四。三臂实际首帧 SHA-256 均为 `37993dd8362650a9fc9c6f52f79c44134c22619613d95c9c82fc5a99098d7a9a`；空臂八步均为 search，未调用视觉模型，供给目标指标为空。[逐项结果与下一步](../experiments/2026-11-21-vln-paper-plan.md#11-月-27-日单帧与四帧开发对照) · [完整运行目录](</root/rivermind-data/uav_rgb/exp-vln-frame-history-20261127-mdRWGA/>)

同一起点局控检查已经跑完。地图教师 34 步停靠，真值 A* 局部路标的 R2 24 步停靠，真值直线方位的 R2 第 3 步碰撞。三臂首帧与视觉臂逐字节一致；各臂的逐步日志、命令和结果已在论文计划中列明。

用于发布的独立快照启用 `RUN_3DGS_REAL=1`，八个 3DGS/VLN 模块共 207 项全过，共享视频面板自检通过。现有 C++ 构建运行 CTest 42/42 通过。包含通用训练入口的扩展检查为 212/213，唯一失败为 `tools.test_train_entry.TrainEntryGuardTests.test_presets_set_expected_modes` 的 OpenFly 预检，缺少本机资产目录的 `asset_inventory.json` 和六场景 `manifest.json`。在基线 `2c772f6` 实际复现相同的两条缺失路径。 [VLN 测试日志](</root/rivermind-data/uav_rgb/exp-vln-frame-history-20261127-mdRWGA/release-vln-tests.log>) · [含 OpenFly 的扩展检查](</root/rivermind-data/uav_rgb/exp-vln-frame-history-20261127-mdRWGA/verified-tests.log>)

### 复现本轮飞行

在仓库根目录执行。给每次运行使用一个新的输出目录。完整路线使用登记 JSON；复核短前缀时将场景换成下列产物中的 `first-eight.json`，四帧使用 `--vlm-history-frames 4`，空指令使用 `--counterfactual zero`。R2 文件 SHA-256 为 `513e95a7a26709ab38a0157453138ea28ef0dac663e8f267919c2c64e397e667`。

```bash
D=/root/rivermind-data/uav_rgb
E="$D/exp-vln-frame-history-20261127-mdRWGA"
CUDA_VISIBLE_DEVICES=0 "$D/3dgs-venv/bin/python" -B -m sim.3dgs.run_vln \
  --scene "$D/exp-vln-acquired-approach-839934-20261124/tr_42_to_tr_42.json" \
  --policy student --ckpt "$D/exp-s1-dagger-r2-cleanaug-20261123/train_r2_subgoal/model/ckpt_best.bin" \
  --grounder vlm --vlm-history-frames 1 --vlm-every 2 --seed 421 \
  --hold 0.5 --stop-mode explicit --videos 1 --out "$D/replicate-wall-lamp"
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "$D/3dgs-venv/bin/python" -B "$E/probe.py" \
  "$D/exp-vln-acquired-approach-839934-20261124/registration.json" "$E/probe-replica.jsonl"
```

要输入自己的指令，在上面的 `run_vln` 命令中加入 `--instruction '飞到候诊区椅子上方的壁灯并悬停'` 并使用另一个输出目录。这个入口将原句送入视觉语言模型，按自由演示记录飞行。该句没有登记实例 ID 和终点，因此没有 SR；要评分，先登记指令和目标，再运行路线模式。

### 没做什么

- 当前 STOP 仍依赖两次新鲜、居中、近距且名称匹配的模型观测。它缺少实例区域、跨帧 ID 和独立的距离核验。同名异物、房间或区域目标仍需分别标注。模型的 search 也没有独立区分画外、遮挡和漏检。
- 尚未调飞控 PID、接入真机遥测、训练新模型、运行公开方法同协议主表或检验从未见过的路线。现有结果支持先比较局部真值路标与视觉方位，再冻结候选实例与停机证据。Studio 的普通 3DGS 评测仍读取另一种 `eval.json`；这次生成的有语言评测和录像来自 `run_vln`。
- 当前工作树的大批既有删除、旧实验、日志及用户的训练入口改动留给独立清理。后续先查每个产物的登记、引用、哈希和可恢复来源。只把本次行为和对应测试从原提交提取并发布。

## 更新 2026-11-27 午间 · 跨房间复验

### 做了什么

- 从发布镜像 `a2d0e37` 执行八集真实 `gsplat + cf_sitl` 匹配指令对照。采用原登记的橱柜起点和教师轨迹中的窗户起点，把模型、固件、场景、两条原句、四臂参数与输入指纹冻结到数据盘。橱柜教师在本次机器台账前已作为预检执行，计入八集。GPU0 的 UUID 为 `GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740`。
- 用冷启动实际首帧、相机外参、官方实例框及原生窗网格复核 #122。将飞行日志中的真实位姿、动作、偏航及终止原因汇成距离曲线，只发布派生曲线。八集的逐步数据、原始文件指纹和可独立复现的评分代码在 [scores.json](</root/rivermind-data/uav_rgb/exp-vln-local-waypoint-transfer-20261127/scores.json>)。
- 修复截断任务误用完整示范的 nDTW/SDTW 评分。内嵌和原始 SAGE 参考路径使用相同的 XY 首尾校验。实际 671 条原生单段路径逐点复核通过。原始报告保留，复核成绩在论文计划中列出。正确指令与交换指令各自使用对应参考路径。
- 实现可复用的离线图文路标探针。相同模型、JPEG 和像素分别询问现有问题或局部可通行方向。每条原句前清记忆。探针支持同图双指令的教师动作、真实飞行的 A* 路标两种标签，输出逐条预测、每类分母和恒选正前的对照。局部路标不能触发 STOP。[探针与选样](../../tools/probe_vln_waypoints.py)

### 验证到哪一步

两条任务模板分别指定诊室衣柜旁的橱柜、卧室二采自然光的窗。地图教师和两种特权学生各自停靠 2/2，视觉语言学生 0/2，无碰撞。窗户起飞时另一窗体也有可见像素，只凭目标区域入画无法验收实例辨认。离线登记、各项得分、视觉见证、下一轮选择条件见[论文计划](../experiments/2026-11-21-vln-paper-plan.md#11-月-27-日跨房间局部路标复验)。

同图双指令的严格局部路标读数为 0/10 对，17/20 条生成文本无法按原协议解析。额外记下原始文本和首个词元，复算得到动作 10/20、整对 1/10，三条教师路线的真路标为 5/6。这是事后诊断。两次运行的严格预测对 20 条分岔和 6 个真路标逐项相同，提示词和像素保持一致。另从动作前速度、位姿和地图重放同一张图的两个局部方向；20/20 个教师动作可复现，只有 2/10 对确实需要不同且可表达的方向。两对的原问法与候选首词元均为整对 0/2。[逐条真值与重放方法](</root/rivermind-data/uav_rgb/exp-vln-local-waypoint-transfer-20261127/fork_bearing_labels.json>)

独立发布快照启用 `RUN_3DGS_REAL=1`，8 个模块的 Python 测试 248/248 通过，耗时 75.169 s。`ctest --test-dir build/check --output-on-failure -j4` 使用本机现有构建，42/42 通过，耗时 33.50 s。用评分 JSON 内的复核代码重读八集原始日志与派生图片，逐项核验通过。Git 含候选、评分源码、测试、数表和只用距离生成的图；原机载图片、权重及三维资产在登记的数据盘位置。[Python 测试输出](</root/rivermind-data/uav_rgb/exp-vln-local-waypoint-transfer-20261127/release-python-final.log>) · [CTest 输出](</root/rivermind-data/uav_rgb/exp-vln-local-waypoint-transfer-20261127/release-ctest.log>)

### 复现

从 `code/uav_rgb` 根目录运行。闭环登记中的 `defaults.common_argv` 替换场景和新输出目录，再接 `arms` 中的参数。为避免镜像滚动，复现原始闭环请使用 `fd63117` 的检出；探针和新路径评分使用本次提交。下例核对离线样本，随后运行局部路标条件。

```bash
D=/root/rivermind-data/uav_rgb
E="$D/exp-vln-local-waypoint-transfer-20261127"
PY="$D/3dgs-venv/bin/python"
FORK="$D/exp-vln-fork-counterfactual-states-20261124/development.jsonl"
TEACHERS=(--teacher-run "$D/exp-vln-frame-history-20261127-mdRWGA/control-teacher"
          --teacher-run "$E/cabinet-teacher" --teacher-run "$E/window-teacher")
"$PY" -B -m tools.probe_vln_waypoints --manifest "$FORK" --check
"$PY" -B -m tools.probe_vln_waypoints "${TEACHERS[@]}" --check
CUDA_VISIBLE_DEVICES=0 "$PY" -B -m tools.probe_vln_waypoints --manifest "$FORK" \
  --condition local-waypoint --out "$E/fork-replica.json"
CUDA_VISIBLE_DEVICES=0 "$PY" -B -m tools.probe_vln_waypoints "${TEACHERS[@]}" \
  --condition local-waypoint --out "$E/bearing-replica.json"
```

`--condition baseline` 使用原来的单帧问题。`local-waypoint` 的同一次运行可核对原始生成文本、严格评分和单词元事后诊断。完整八次飞行的独立核验命令如下，运行目录须包含原始日志。

```bash
cd /root/rivermind-data/uav_rgb/exp-vln-local-waypoint-transfer-20261127
/opt/conda/bin/python -B -c 'import json,sys; exec(json.load(open(sys.argv[1]))["reproduction"]["source"])' scores.json verify
```

### 没做什么

- 本轮是圈外途中起飞的匹配指令复验。两夹具没有符合合同的交换伙伴，不能据此报告端到端的语言因果、整程任务成功、目标实例正确率或未见场景成绩。正式留出房间 840818 继续封存。
- 未调真机 PID。两次超时都能由策略持续下达同一转向指令解释，尚无新的真机遥测。局部路标候选只参加离线对照，是否进入闭环按预先写下的判据决定。
- 未追加公开榜单对比或清理历史实验。公开方法的同协议比较依赖可用的指令策略；旧日志同时是这次首帧和反事实动作可追溯的来源。

## 更新 2026-11-27 下午 · 同图双指令与局部方向

### 做了什么

- 并行复核了四份冻结状态清单、两份路线登记、原生 SAGE-3D 指令、80 条教师飞行和相机记录。旧清单在首次执行动作分歧时才置共享标志。新的读取器从两条实飞轨迹、动作前位姿、四元数、速度和解码 RGB 核验了完整共享前缀。每行记录实飞臂；分歧之后，另一条指令使用同一张照片、同一状态重放。原始记录及标志均保留。[来源读取器](../../tools/vln_fork_sources.py)
- 离线重放沿用运行时 A* 子目标和制动教师，分别保存连续方位、八方向桶、下一动作和相机内外参投影。近终点、无路径、退化向量有独立状态。原句的起终点、实例关联来自原数据；逐句人工复核的目的地参照以联合摘要和字符区间保存，这样 Git 中可以复现短语，不需要分发源图文。[几何重放](../../tools/vln_fork_geometry.py) · [制标入口](../../tools/label_vln_forks.py) · [终点参照区间](../../tools/data/vln_fork_reference_spans.json)
- 在独立工作树给 `train.py` 增加可选的 `fork_region`。固定 CLIP 同时提取整图和左中右三个视觉区域的表示。三种输入独立拟合，训练端只从三个训练房间取可表达的标签。训练和开发使用独立房间、起点、路线对和图像。原工作树 `train.py` 有既有 +453/−179 行改动，实验分支从已发布源码构建。[实验分支](https://github.com/xu967/ALVLN/tree/experiment/20261127-region-bearing/uav_rgb)

### 复现

在 `uav_rgb` 根目录执行。标签来源包含 `3dgs-scenes`、`pubdata`、11 月 23 日的实例关联目录，以及 11 月 24 日的两份分岔教师采集和反事实清单三个目录。复制数据时保留相对层级和文件字节。还原十对旧回归另需上一轮的 `fork_bearing_labels.json`。标签输出路径必须是新目录。

```bash
D=/root/rivermind-data/uav_rgb
E="$D/exp-vln-fork-bearings-20261127"
PY="$D/3dgs-venv/bin/python"
JOB="$E/labels-replica-job"
mkdir "$JOB"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 setsid nohup bash -c '
  "$1" -B -m tools.label_vln_forks --data-root "$2" --out "$3"
  rc=$?
  printf "rc=%d finished_at=%s\n" "$rc" "$(date -Iseconds)" > "$4/DONE"
  exit "$rc"
' _ "$PY" "$D" "$E/labels-replica" "$JOB" > "$JOB/run.log" 2>&1 < /dev/null &
echo "$!" > "$JOB/run.pid"
```

本机首次逐状态重放实际耗时 4 分 11 秒。单房间的 32 张图低估了其他房间的 A* 开销。复跑可预留约 5 分钟；完成后读 `$JOB/DONE`，核对 `rc=0` 再执行真实数据回归。

```bash
UAVRGB_FORK_LABEL_DIR="$E/labels-replica" "$PY" -B -m unittest -q \
  tools.test_label_vln_forks tools.test_vln_fork_sources tools.test_vln_fork_geometry \
  tools.test_vln_fork_references tools.test_probe_vln_waypoints
```

训练命令在 GitHub 的 `experiment/20261127-region-bearing` 分支的 `uav_rgb` 目录运行。已完成的实验可先查分母、模型权重、逐条预测和运行登记；复跑指定新的输出名。实验分支读取数据盘原先登记的 `$E/labels`。在相同目录布局的备份机上恢复时，两个登记文件分别取自 Git 的 `tools/data/vln_fork_bearings_registration.json`、`tools/data/vln_fork_region_registration.json`，对应 `$E/registration.json`、`$E/region_registration.json`。派生 JSONL 含可回查的绝对证据路径；原登记 SHA 对应此机的数据根，迁移到不同绝对路径后仍可重放标签，不能将新 JSONL 当作原指纹启动这次固定协议的训练。

```bash
UAVRGB_NAME=exp-vln-fork-region-replica python3 train.py fork_region --check
CUDA_VISIBLE_DEVICES=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740 \
  UAVRGB_NAME=exp-vln-fork-region-replica python3 train.py fork_region
```

### 没做什么

- 没有给终点参照词分配假想实例或像素框。旧框标注 558 张图与这批 1,118 张图的解码像素哈希交集为零。相机内的瞄准点仅有几何投影，目标实体可见性及沿射线的遮挡需要分别标注。
- 没有更改 R2 飞行权重、飞控增益或真机配置。既有固件在环日志能解释持续右转；本批输入是同一动作前状态，不能估计真实机体抖动或停机成功率。公开方法同任务闭环与正式留出房间 840818 仍按论文计划安排。已登记教师日志是重放来源，保留供复查。

### 验证到哪一步

- 真实数据回归将四房间来源与划分逐行对齐。再次根据位姿和瞄准点独立计算，2,236 个连续角度和桶均匹配；旧十对的两条原句、两条连续方向、相机投影和教师动作也逐项复现。标签清单 SHA-256 为训练 `24a9d3ae740b69f9ebf9d4022f61ea59de38cb29e851fcd560bb7de4dc23b8c0`、开发 `0c79a8079e84919c4355a5da3e6efed1321b1204aac143aed08e46c7d36c3566`。来源路径、代码指纹与其余输入由汇总文件保存。[标签汇总](</root/rivermind-data/uav_rgb/exp-vln-fork-bearings-20261127/labels/summary.json>)
- 制标、参照、旧探针及真实十对回归的 27 项测试全部通过。再加 `sim.3dgs.test_teacher_clearance`、`test_teacher_tol`、`test_collision`、`test_collision_model`，在独立导出的发布快照中共 **181/181 通过，37.288 秒**。原工作树中两个既有测试文件有差异，相同命令在那里是 172/172，40.006 秒。候选分支的 `tools.test_fork_region` 五项通过。旧训练入口七个预设逐项比对版本前后预检的输出和退出码，OpenFly 的旧资产缺失在两版均会触发；本次候选使用了完整的真实权重与图像。[发布快照测试日志](</root/rivermind-data/uav_rgb/exp-vln-fork-bearings-20261127/release-regression.log>)
- 物理 GPU0（NVIDIA GeForce RTX 2080 Ti）于 17:54 完成全部候选及对照，入口记录 19.09 秒，`DONE` 为 `rc=0`。独立从 286 对逐条预测重算四组、三类、32 对异向、九个路线对及起点分布；均与发布指标一致。双入画异向 0/5、真实共享异向 0/1。区域候选未过三项条件，本次实验停在离线阶段。[逐条预测与指标入口](</root/rivermind-data/uav_rgb/exp-vln-fork-region-20261127/fork_region/summary.json>) · [数据盘运行日志](</root/rivermind-data/uav_rgb/exp-vln-fork-region-20261127/run.log>)
- 开发集的配对图按预先选样规则出图并人工检查；从配套 JSON 的复图代码得到字节相同的 PNG，SHA-256 为 `b3181d8f60ade7766ab7b835fb94909614ff7e50d98cb4d71f3aeea715477e6e`。含机载 RGB 的图与原始日志留在数据盘，Git 收录方法、文字区间、登记、结果表与复现命令。[配对图与复图代码](</root/rivermind-data/uav_rgb/exp-vln-fork-bearings-20261127/paired_waypoints.json>)

## 更新 2026-11-27 晚间 · 补采共同分岔

### 做了什么

- 先冻结房间、五房原文件与 CF-SITL 可执行文件指纹、8 步预算、原句选择、独立起点和主样本规则。补查了符合旧条件的轨迹库，核对旧 40 对与其共同前缀。使用原生 XYZ，在每一地点选一条共同航向，可还原每条路线原始的朝向。富集样本的 CPU 预演在两位教师发出相同动作时才推进固件。[设计与预演](../../tools/design_vln_shared_forks.py)
- 原 `sim.3dgs.run_vln` 收到两份不同的原生 SAGE-3D 任务。A/B 用同一个 seed 和出生位姿分别飞，写出调试工具可直接读取的真实相机、深度、动作和轨迹。核验器打开两份产物核对整个共同前缀，包括各窗的 RGB、深度及 10 Hz 固件轨迹。复位图不进入主样本，首次动作分歧当下的前图仍属共同前缀。[飞行核验](../../tools/verify_vln_shared_forks.py)
- 第一个表列的是局部瞄准点。目标实体另用 InteriorGS 原生实例盒和同期深度检查，输出候选区域及人工可复核的画面，参照词按原句与目标的精确摘要连接。再以已经发布的固定参数评分，每个主起点只计一次，推理端只读取 RGB 和实际原句。[目标可见性复核](../../tools/vln_fork_visibility.py) · [冻结模型评分](../../tools/score_vln_shared_forks.py)

### 验证到哪一步

复飞历史对照 `839992_tr_118_tr_136`，A/B 两进程共用 GPU0，合计 25.353 秒，两端均正常结束。完整 28+28 个 RGB、140+140 条遥测各自与旧飞行一致；第 26 步前图解码 SHA-256 为 `94b4c4329164b0b91fc5ad504fe45e35549f7e012f77f6cabd3e60b92b327d5e`。审核器验收了 26 个已经飞过共同前缀的状态，其中 25 个同向，1 个异向且单侧路标入画。没有把这对重复路线算作新增地点。[历史复飞对账](</root/rivermind-data/uav_rgb/exp-vln-shared-prefix-20261127/pilot/historical_comparison.json>)

第 26 步两条原句的真方向依次为正前、右前。区域加原句给出左前／左前，区域加空句同样给出左前／左前，整图加原句给出右前／右前，恒前给出正前／正前。该步仅一个路标入画，严格主样本的分母为零。另对照原图、官方实例盒以及渲染深度，壁画 #157 的 85 条穿框射线与盒内深度相容，饰物 #86 的 475 条射线都先遇到更近的表面。盒内深度相容还需结合原图核对实例身份。[原图、目标框与局部路标](</root/rivermind-data/uav_rgb/exp-vln-shared-prefix-20261127/historical_visibility_smoke.png>) · [冻结模型逐窗预测](</root/rivermind-data/uav_rgb/exp-vln-shared-prefix-20261127/pilot/frozen-scores/predictions.jsonl>)

设计、飞行证据、可见性、冻结模型、既有制标和真实历史回归共 43 项通过。覆盖了改动预算、漏登记房间、缺一臂、只共享复位图、相同动作直到时限及同一地点多帧等情形。正式批次按此次提交的干净源码运行。

### 复现与接手

数据根记为 `D=/root/rivermind-data/uav_rgb`，实验根为 `E=$D/exp-vln-shared-prefix-20261127`。固定训练产物来自 `$D/exp-vln-fork-region-20261127/fork_region`，该批冻结特征代码位于同目录的兄弟目录 `source`，源提交 `4503dcb`。另建输出目录复跑时，先将 Git 的 `tools/data/vln_shared_forks_protocol.json` 放入新目录作 `protocol.json`。使用相同的固件和五房原生资产；运行命令如下。

```bash
export CF_SITL_PROC="$D/tmp/build-cf-sitl/cf_sitl_proc"
export TORCH_EXTENSIONS_DIR="$D/torch-cache/extensions-sm75" TORCH_CUDA_ARCH_LIST=7.5
export CUDA_VISIBLE_DEVICES=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
PY="$D/3dgs-venv/bin/python"
"$PY" -B -m tools.design_vln_shared_forks --out "$E" --workers 3
```

设计器发布 `design.json` 后，按 `pairs[].arms[].argv` 的完整参数数组从干净源码运行每个独立教师进程。每个退出码单列，保留失败和提前终止。其余步骤读同一登记。

```bash
"$PY" -B -m tools.verify_vln_shared_forks --design "$E/design.json" --out "$E/verified" --data-root "$D"
"$PY" -B -m tools.vln_fork_visibility --observations "$E/verified/observations.jsonl" --out "$E/visibility" --data-root "$D"
"$D/envs/vlm-venv/bin/python" -B -m tools.score_vln_shared_forks --design "$E/design.json" \
  --observations "$E/verified/observations.jsonl" --old-run "$D/exp-vln-fork-region-20261127/fork_region" \
  --source "$D/exp-vln-fork-region-20261127/source" --out "$E/frozen-scores"
```

### 没做什么

本批没有训练新权重、用富集样本估计自然路线分岔频率或调整飞行增益。几何入画、渲染深度和实例盒用于辅助审图，未获实例证据的参照保留为待复核。短窗结果不能衡量从原生起点到终点的导航成功率；840818 继续封存。即使 CPU 预演满足两点入画，也只有实采通过双臂核验才能增加覆盖数。
