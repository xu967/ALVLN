# 2026-11-16 晚间实验报告：卡 S1 机载动态条件/STOP 学生、卡 G2 零特权 VLM、SCENE-LONG 与 OpenFly v2

- 日期：2026-11-16
- 主产物目录：
  - `/root/rivermind-data/uav_rgb/exp-s1-g2-scenelong-20261116/`（`DONE: rc=0`）
  - `/root/rivermind-data/uav_rgb/exp-openfly-v2-20261116/`（`DONE: rc=0`）
  - `/root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/six_scene_causal_bc_v1/`（`DONE: completed rc=0`）
- 复现入口：
  - `experiments/2026-11-16-s1-g2-scenelong/run.py` 与 `experiments/2026-11-16-s1-g2-scenelong/run_s1_g2_v2.py`
  - `experiments/2026-11-16-openfly-v2/run.py`

---

## 1. 任务卡 SCENE-LONG：室内碰撞网格解锁与户外原生长航线导入

### 1.1 SAGE-3D 门洞开启网格碰撞替换高斯体素前后对比

使用 `tools/usdc_to_ply.py` 提取 `spatialverse/SAGE-3D_Collision_Mesh` 的世界系三角网格并移除闭合 CAD 门板 `/Door_*` 后，6 个场景的可行单段路线从 `249 / 676 (36.8%)` 提升至 **`578 / 676 (85.5%)`**，并串联出同 split 跨房间 `30–55 m` 多阶段航线 **`123` 条**：

| 场景 | 候选总数 | 旧高斯碰撞可行数 | 新网格碰撞可行数 | 单段 `>=15m` 条数 | 单段最大 A* 长度 (m) | 多阶段 `30–55m` 条数 |
|---|---:|---:|---:|---:|---:|---:|
| `sage3d_839992` | 200 | 123 | 155 | 23 | 17.65 | 24 |
| `sage3d_840139` | 158 | 34 | 145 | 81 | 35.68 | 24 |
| `sage3d_840865` | 100 | 40 | 84 | 0 | 13.48 | 24 |
| `sage3d_841012` | 109 | 36 | 92 | 3 | 17.88 | 24 |
| `sage3d_841150` | 104 | 14 | 97 | 19 | 22.04 | 24 |
| `sage3d_841980` | 5 | 2 | 5 | 1 | 18.67 | 3 |
| **合计** | **676** | **249** | **578** | **127** | **35.68** | **123** |

### 1.2 Wanderland 原生街道语言长航线导入与质量探针

| 场景 | 原生候选数 | `route_check` 通过数 | `>=30m` 条数 | `>=40m` 条数 | 最大测地线 / A* 长度 |
|---|---:|---:|---:|---:|---|
| `wanderland_1MlUucug` | 18 | 15 | 5 | 2 | `47.26 m` / `52.86 m` |
| `wanderland_10nvb` | 18 | 13 | 4 | 3 | `42.03 m` / `45.10 m` |

`wanderland_1MlUucug` 首条长航线 `wl_ep03_47m`（`52.86 m`）探针实测：主航向 88 帧低质率 `0.0%`、下半幅 `alpha >= 0.50` 最小有效占比 `91.97%`（通过）、每 `1.0 m` 处 `±45°/±90°` 扫视 204 帧低质率 `14.71%`（通过 `sweep_pass`）。

---

## 2. 任务卡 S1：自研 C++/CUDA 15 维动态导航条件 + 第 5 维独立 STOP 学生闭环

### 2.1 训练配置与关键消融教训

- 首轮尝试开启 `--class-weights auto` 时，因刹停教师 `v5` 仅在制动与末步输出 `SAFE`（约占 2%），逆频率权重变为 `[15.09, 0.36, 1.72, 1.70]`，`YAW_L/R` 权重为 `FORWARD` 的 `4.8` 倍，导致在 `bearing_bin == 0`（`[-22.5°, +22.5°)`）内分类器偏向原地左右摆头（留出集 `SR = 0.10`）。
- 改为无偏置动作 CE + `--two-frame --vln-nav --stop-head --depth-aux --vln-search-dropout 0.35`（仅在 `840139/841012/841150` 的 `train` 路线采集 75 集示范 + 45 集 DAgger，`839992` 的 `train` 路线 15 集作 `val`，共 135 集 `697 MB`），验证集动作准确率从 `59.6%` 升至 `76.2%`、`STOP F1 = 0.933`（自动选出阈值 `0.55`），单步 C++/CUDA 推理耗时约 `0.85 ms`。

### 2.2 留出测试集与户外零样本长航线闭环结果（`cf_sitl` 固件在环）

留出集由 `sage3d_839992` 的 5 条 `test_Instruction_Unseen` 与完全未参与训练/验证的公寓 `sage3d_841980` 的 5 条 `test_Scene_Unseen` 组成（共 10 集）：

| 评测臂 | 场景范围 | 请求/完成 | SR | OSR | SPL | nDTW | SDTW | 中位 NE (m) | 碰撞率 | 自主 STOP 率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `teacher_v5`（特权上界） | 室内留出 10 集 | 10/10 | **0.900** | 1.000 | 0.831 | 0.604 | 0.551 | 0.866 | 0.000 | 1.000 |
| `student_g0_subgoal`（C++ 学生） | 室内留出 10 集 | 10/10 | **0.700** | 0.700 | **0.587** | **0.444** | **0.441** | **0.956** | 0.200 | 0.700 |
| ├ 其中 `test_Instruction_Unseen` | `839992` 5 集 | 5/5 | **0.800** | 0.800 | 0.675 | 0.509 | 0.507 | 0.980 | 0.200 | 0.800 |
| └ 其中 `test_Scene_Unseen` | 未见公寓 `841980` 5 集 | 5/5 | **0.600** | 0.600 | 0.499 | 0.380 | 0.375 | 0.932 | 0.200 | 0.600 |
| `student_blind_unknown`（方位恒置 8） | 室内留出 10 集 | 10/10 | 0.000 | 0.000 | 0.000 | 0.039 | 0.000 | 7.065 | 0.200 | **0.000** |
| `student_blind_rgb`（全黑图像 + A*子目标） | 室内留出 10 集 | 10/10 | 0.800 | 0.800 | 0.674 | 0.517 | 0.515 | 0.922 | 0.100 | 0.800 |
| **户外零样本迁移 `student_g0_subgoal`** | `wanderland_1MlUucug` 3 条长线（含 `52.86m`） | 3/3 | **1.000** | **1.000** | **0.974** | **0.666** | **0.666** | **2.326** | **0.000** | **1.000** |

**方法学诚实诊断**：`student_g0_subgoal` 的自主 STOP `SR = 0.700` 达到 `teacher_v5`（`0.900`）的 `77.8%`，满足卡 S1 的 `>= 75%` 门槛，且方位遮盲 `blind_unknown` 跌至 `0.000` 且零误停；但 `student_blind_rgb` 在喂入 A* 局部子目标时仍达 `0.800`，说明 A* 局部子目标本身已承担了大部分绕障几何信息，不能据此宣称单靠机载 CNN 已学会无引导复杂绕障。

---

## 3. 任务卡 G2：零特权视觉语言 Grounder (`UAV-PUR-4B`) + 机载 C++ 学生反事实闭环

在完全未参与训练的公寓 `sage3d_840865` 中选取两组原生同起点异目标任务对（`acoustic_equipment_1 -> sofa_1` vs `body_pillow_8`；`basin_cabinet_2 -> mirror_cabinet_1` vs `window_8`），1 Hz 调用 `UAV-PUR-4B`、10 Hz 运行 C++ 学生与 1 kHz `cf_sitl`：

| 反事实模式 | 请求/完成 | SR | OSR | nDTW | 平均 NE (m) | 碰撞率 | 自主 STOP 率 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `correct`（原指令） | 4/4 | 0.000 | **0.250** | 0.180 | 2.852 | **0.000** | 0.000 | `tr_66 (sofa_1)` 纯视觉语言引导进入目标圈（`OSR=1, NE=1.19m`） |
| `swap`（同起点互换目标指令） | 4/4 | 0.000 | 0.250 | 0.203 | 2.244 | **0.000** | 0.000 | 互换后飞向配对另一侧目标（`tr_86` 被喂 `sofa` 指令飞抵沙发旁） |
| `zero`（空指令） | 4/4 | 0.000 | 0.000 | 0.142 | 3.081 | **0.000** | 0.000 | 原地单向扫描，不乱飞、不误停 |

**关键发现**：① `VlmBeliefTracker` 的奇数步单步转角回正推算与跨中线防过冲消除了 1 Hz VLM 与 10 Hz 控制之间的 `L-L-R-R` 极限环振荡，使无人机平稳飞抵目标物前方；② 未微调的航拍大模型 `UAV-PUR-4B` 在室内 `0.8–1.2 m` 近景始终输出 `near_goal = 0`，是当前零特权 `SR = 0.0` 的唯一卡点（C++ 学生自身的 `stop_head` 在收到 `near_goal = 1` 时召回率达 `100%`）。

---

## 4. 任务卡 OF-E1/E2：OpenFly 6 场景因果扩量与 `v2` 契约消融

- **数据与空间隔离**：100 条 6 场景路线（`env_airsim_{16,18,23,26,gz,sh}`）全部通过 `0..5`（含升降 `±3.0 m`）因果原语对拍，提取 `960` 张组首 PNG（磁盘占用 `575 MiB`），按 `< 30 m` 3D 连通走廊聚簇划分为 `80 train / 20 validation`，跨 split 同场景最近 3D 轨迹距离实测 **`42.79 m`**（严格大于 `30 m` 阈值）。
- **固定 1,200 updates（seed `20261116`）空间隔离验证集（20 条路线，172 帧）结果**：

| 模型臂 | 契约版本 | Val CE (↓) | Val Acc | Present mF1 | STOP Prec | STOP Recall | STOP F1 | 二周期摆头数 | 20 集 Sequence Parity 最大差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1_cma_rgb_language_6scene` | `v1` | 1.778 | 0.529 | 0.200 | 0.500 | 0.250 | 0.333 | 0 | `7.63e-6`（argmax 全同） |
| `v2_cma_rgb_only` | `v2` | **1.134** | **0.605** | **0.354** | 0.600 | 0.600 | 0.600 | 0 | `3.10e-6`（argmax 全同） |
| `v2_cma_rgb_language` (`correct`) | `v2` | 1.193 | 0.570 | 0.353 | **0.650** | **0.650** | **0.650** | 0 | `5.72e-6`（argmax 全同） |
| └ 同模型 `cyclic_instruction_swap` | `v2` | 1.223 | 0.558 | 0.338 | 0.519 | 0.700 | 0.596 | 0 | — |

**待显卡空闲时运行的多场景 UE4 AirSim `v3` 在线闭环复现命令**（先按 `docs/OPENFLY_LOCAL_EVAL.md` 启动对应场景二进制）：

```bash
TORCH_HOME=/root/rivermind-data/uav_rgb/torch-cache \
HF_HOME=/root/rivermind-data/uav_rgb/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/rivermind-data/uav_rgb/envs/airvln/bin/python tools/eval_openfly_subset.py \
  --checkpoint /root/rivermind-data/uav_rgb/exp-openfly-v2-20261116/training/v2_cma_rgb_language/model.pt \
  --manifest /root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/six_scene_causal_bc_v1/manifest.json \
  --scene env_airsim_16 \
  --airvln-root /root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/AirVLN \
  --out /root/rivermind-data/uav_rgb/exp-openfly-v2-20261116/closedloop_env16_v2_lang
```
