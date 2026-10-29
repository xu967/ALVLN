# 接手会话：环境核验、三路代码审核、best-checkpoint、数据门禁与本地监控

- 时间：2026-10-29 22:10
- 范围：`uav_rgb/` 全仓（`src/`、`sim/tools/`、`tools/`、`ui/`、`docs/`）+ 本机环境；
  未改 `TinyAerialVLN_Final_V8/`、`CF-Drone/`、`idea/`
- 结论：**完成**。GPU 环境全绿；本地监控台确认无 web 依赖，并在同一会话补齐真实训练接入；
  best-checkpoint、3 项数据完整性 P1、4 项 C++/CUDA 静默失败 P1 已修；
  三路审核 P0 均为零。

## 做了什么

### 1. 环境与仓库现状（实测）

- `nvidia-smi -L` → RTX 2080 Ti；`doctor.py --require-gpu --require-sim` 全部 PASS；
  `tools/check.sh` EXIT=0：18/18 CTest（95.56 s）、Python 自检全过、产物仅 `sm_75`。
- `git log`/树对比：本地 `uav_rgb` HEAD `70bf088` 的 192 个文件与远端 ALVLN `f5aeeaf`
  的 `uav_rgb/` 子树逐文件哈希一致；本轮改动全部未提交、未推送。
- 真实 Gazebo 冒烟：1 实例 × 3 步，起飞 z=2.12 m，3/3 步写入，EXIT=0，无残留进程。

### 2. 三路只读审核（子 agent）

- **sim/tools + tools**：3 个 P1 + 10 个 P2。P1 全部已修，见下；P2 记录了
  depth/contact 停更、carrot 全堵（已修）、actions.jsonl 含位姿派生量、provenance 丢失、
  SIGTERM 清理等。
- **ui/**：P0 为零；本地化目标达成（无 web/端口、快捷方式可启动、无隐藏网络）。
  P1 三条：孤儿训练进程（已修：SIGINT/SIGTERM → `_request_quit`）、真实 trainer 启动合同
  （已由并行会话 `ui/trainer_native.py` + `.monitor` 侧目录修好）、`tail_text` 半行
  （已修：`state["partial"]`）。P2 记录了 FIFO/超大文件、默认目录在仓库内、文档过期等。
- **src/**：P0 为零；4 个 P1 + 12 个 P2，全是「训练 exit 0 但产物不可信」类型。
  P1 已全部修复：显式 0 权重导致中途除零、CSV/JSONL 写失败静默、非有限 dev 指标写
  `null` 并选成 best、NaN 参数被 checkpoint/ReLU 掩盖。另有 8 个 P2 记录未修
  （device 层 best、RNG、fsync、64 位溢出等）。详见
  `2026-10-29-2215-src-silent-failure-fixes.md`。

### 3. 代码改动

- **best-checkpoint**（本会话）：`Trainer::evaluate` 拆到 `src/trainer_eval.cpp`；
  每 epoch 按 dev 宏 F1 保存 `ckpt_best.bin`；checkpoint v3 记录峰值、兼容读 v2。
  详见 `2026-10-29-2155-best-checkpoint.md`。
- **数据完整性**（本会话）：`-0.0` gid 归一 + salt v2；collect/preprocess 几何 split 门禁；
  位姿停更 fail-closed；teacher 全堵回 `SAFE_NEUTRAL`。详见
  `2026-10-29-2200-data-integrity-fixes.md`。
- **监控台**（并行会话主导 + 本会话尾修）：`ui/trainer_native.py` 把 UI 配置转成真实 trainer
  CLI；控制台文件移到 `<run_dir>.monitor/`；`tail_text` 半行缓存；SIGTERM 清理；
  浏览非空历史 run 后训练输出目录换新。详见 `2026-10-29-2205-real-trainer-integration.md`。
- **C++/CUDA 静默失败**（本会话）：显式 0 类别权重入口拒绝；CSV/JSONL 写失败抛错；
  dev 非有限指标中止 + lr 上限 1.0；checkpoint 读/写校验有限性；ReLU 传播 NaN；
  非有限梯度 norm CPU/CUDA 一致抛错。详见
  `2026-10-29-2215-src-silent-failure-fixes.md`。

### 4. 文档

- `docs/HANDOFF.md` §十一：本轮结论、下一任优先级、Git 状态。
- `docs/EVAL_STANDARD.md`：split 改成几何组口径、salt v2、旧 `scene_id % 10` 标为废弃。
- `docs/监控系统交接.md`：旧 web 进程已消失、真实训练接入、`ckpt_best.bin`。
- `tools/DATASET_FORMAT.md`：`geometry_gid` 字段。
- `docs/results.md`：本轮所有实测输出。

## 没做什么

- **没有提交或推送任何改动**。原因：规范要求用户明确说「提交/推送」；且工作区包含
  并行会话的改动，需逐文件确认归属。
- **没有实现 §19.3 仿真时间动作窗口**。原因：独立改造，必须用真实 Gazebo 多实例复验；
  `--hold` 仍是墙钟秒，采集不可复现。
- **没有加深度/contact 停更阈值**。原因：历史 gap 到过 168–308 ms，阈值要真实多轮标定。
- **没有重采 v1 数据**。原因：salt 升 v2 后必须新数据版本；重采需要先修动作窗口/车道几何。
- **没有实现学生闭环 SimulatorEval、车道几何、SAFE_NEUTRAL 数据补齐**。
  原因：这些是研究主线，超出本轮「核验 + 审核 + 安全修复」范围。
- **没有转换 TinyAerialVLN_Final_V8 的 NavLab Studio web 前端**。原因：用户要求代码落在
  `uav_rgb/`；uav_rgb 的监控台已是本地 Tk。V8 仍是独立 web 原型。
- **没有在真实采集数据上训练到收敛**。原因：本轮只验证训练器/监控链路；
  合成集与 5 帧真实集都不能作为泛化结论。

## 待确认

- salt v2 之前采集的所有 episode（含 `/root/rivermind-data` 里的历史冒烟）如何处理：
  重采、单独标 v1，还是只保留作调试数据。
- 位姿停更改为整集失败后，真实 8 实例采集的成功率/吞吐是否可接受。
- 车道缝 0.902 m 是否放宽（场景几何决策，需要定目标穿越率）。
- 是否要把 `displacement_m`/`dyaw_deg` 移出 actions.jsonl（当前 preprocess 不读，
  但契约上属动作后位姿派生）。

## 验证到哪一步

| 项 | 真实输出 |
| --- | --- |
| `bash tools/check.sh`（DISPLAY=:20） | **EXIT=0** |
| CTest | **18/18 passed**，95.56 s（最终全量） |
| `cuobjdump -lelf build/check/uavrgb_train` | 仅 `sm_75` |
| 真实 Gazebo 采集 | 1 实例 3 步，3/3，EXIT=0 |
| `test_monitor.py`（GUI） | 102 项通过 |
| `test_trainer_demo.py` | 89 项通过 |
| `test_collect.py` | 14 项通过 |
| `test_nn_cuda`（NaN / 非有限 norm 新增断言） | 通过 |
| `test_split_manifest.py` | 9 tests OK |
| `teacher.py` / `scene_gen.py` / `gz_io.py` | 全部通过 / 100 seeds PASS / 全部通过 |
| 几何 split 负例（check.sh） | 同 gid 跨 train/val 被 preprocess 拒绝 |
| 界面真实训练（并行会话） | 3 epoch rc=0；停止 rc=-2；resume rc=0 |
| 界面 wrapper + 新 trainer（本会话） | rc=0，产出 ckpt_best.bin |
| SIGTERM 冒烟 | 进程 rc=0，无残留 |
| `git diff --check` / `py_compile` | 干净 / OK |

未验证：小时级真实训练、8 实例复采、真实训练收敛、板端/NPU、Windows/macOS、
Gazebo GUI、v1→v2 数据重标。

## 风险与遗留

- **v1 数据与 v2 salt 不兼容**：混用会得到错误的 split 标签。
- **checkpoint v3 旧二进制读不了**：回退版本需重新训练或导出。
- **位姿停更 fail-closed 会降低采集成功率**：这是正确性换来的代价，需实采确认。
- **best-checkpoint 在极小 val 上可能过拟合 val**：正式实验仍以 scene 级隔离和 test 为准。
- **工作区未提交**：并发会话改动与本次改动混在一起，提交前需逐文件确认。
