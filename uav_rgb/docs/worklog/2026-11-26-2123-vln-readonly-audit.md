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
