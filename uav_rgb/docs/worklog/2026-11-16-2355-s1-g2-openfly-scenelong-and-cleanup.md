# 执行计划 §10.18 四卡落地（S1/G2/OF-E1&E2/SCENE-LONG）与数据盘二次深度清理

- 时间：2026-11-16 23:55
- 范围：`src/`、`tools/`、`sim/tools/`、`sim/3dgs/`、`experiments/` 及数据盘 `/root/rivermind-data`
- 结论：完成（卡 S1、卡 SCENE-LONG、卡 OF-E1/E2 离线验证与卡 G2 零特权在线链路全部实测跑通；数据盘可用空间提升至 101 GiB；全量 `tools/check.sh` 退出码 0）

## 做了什么

- 数据盘 `/root/rivermind-data` 二次深度审计与安全清理：核查跨项目引用后，清理 1,088 个废旧目录/重复物理拷贝/中间轮次权重与 75,819 张未编码历史散帧 PNG（含 `root-offload-20261107`、`downloads/inria-3dgs-pretrained`、重复拷贝 `airvln-runtime-ubuntu/ENVs/env_5` 改软链、`exp-vln-m1-20261113`、`exp-proto-p1-20261112`、`tmp/phase12-cards` 变异构建草稿及新实验打包后的原始 `depth/*.npy`），在新增今日全部扩量数据与新虚拟环境后，`/root/rivermind-data` 可用容量从 `72 GiB` 提升至 `101 GiB`（较早间救急前 `1.5 GiB` 累计净增约 `99.5 GiB`）。
- 落地任务卡 S1（机载自研 C++/CUDA 动态导航条件与独立 STOP 头）：
  - 新增 `src/include/uavrgb/vln_head.hpp` 与 `src/vln_head.cpp`：实现 15 维动态导航条件混合基数单字节编解码 `(prev_action 0..3, bearing_bin 0..8, near_goal 0..1) -> 0..71`、翻转增强左右镜像映射 `flip_vln_nav_code`、第 5 维独立 `stop_logit` 的数值稳定带权二分类 BCE 损失与验证集阈值扫描。
  - 更新 `src/include/uavrgb/aux_sidecar.hpp`、`src/aux_sidecar.cpp`、`tools/read_dataset.cpp`、`tools/preprocess.py`：扩展 `UAVRAUX1` 侧车 `flags bit1`（`stop_target u8[N]`，优先取 `stop_expert` 防 DAgger 泄漏）与 `flags bit2`（`nav_cond u8[N]`），新增 `--cond vln-nav` 与 `--vln-search-dropout`，将 `"teacher_version"` 纳入 `FAMILY_KEYS` 防混包，提取 `vln.text` 写入 `vln_instructions.jsonl`。
  - 更新 `src/model.cpp`、`src/model_dev.cu`、`src/model_dev_film.cu`、`src/trainer*.cpp`、`src/train_main.cpp`、`src/checkpoint.*`、`src/infer*.cpp`：支持 `prev_dim=15` 三槽并发 FiLM 激活与 `fc.out=5` 联合训练；升版 `UAVRCK01 v7` 追加 32B `aux_sidecar_sha256` 与 `stop_threshold`（向下兼容 `v2..v6`）；`uavrgb_infer` 对 `prev_dim=15` 读 3 字节条件并输出 `"stop_prob"` 与 `"stop_requested"`；文本模型传 `flip` 显式报错拒绝。
  - 更新 `sim/tools/nav_cond.py`、`sim/tools/student.py`、`tools/select_ckpt.py`、`tools/average_ckpt.py`、`sim/3dgs/collect_3dgs.py`、`sim/3dgs/run_vln.py`：打通 3DGS 学生采集、DAgger 与闭环评测；新增 CTest `src/tests/test_vln_cond_stop.cpp` 与 Python 单测。
  - GPU + `cf_sitl` 闭环实测（产物 `/root/rivermind-data/uav_rgb/exp-s1-g2-scenelong-20261116/card_s1_v2/`）：仅用 3 个训练公寓 `train` 路线采集 + 1 轮 DAgger 训练自研 C++ 模型（`ckpt_best.bin`），在留出集（`839992` `test_Instruction_Unseen` + 完全未见公寓 `841980` `test_Scene_Unseen` 共 10 集）上自主 STOP `SR = 0.70`（达特权教师 `v5` `SR = 0.90` 的 `77.8%`，通过 `>= 75%` 门禁），`SPL = 0.587`、`nDTW = 0.444`、`SDTW = 0.441`；方位遮盲 `blind_unknown` 跌至 `SR = 0.0, stop_rate = 0.0`；零样本迁移至户外 `wanderland_1MlUucug` 街道长航线（最长 `52.86 m`，满电 `soc100`）取得 `3/3` 自主停靠（`SR = 1.000, SPL = 0.974, nDTW = 0.666`）。
- 落地任务卡 G2（零特权视觉语言 Grounder 在线闭环）：
  - 新建 `/root/rivermind-data/uav_rgb/envs/vlm-venv` 环境，新增 `sim/3dgs/vlm_grounder_worker.py` 并扩展 `sim/3dgs/grounder.py`、`sim/3dgs/reactive.py`、`sim/3dgs/run_vln.py`、`sim/3dgs/test_grounder.py`：只读挂载本地 `UAV-PUR-4B`（缓存全部指向数据盘，不写 `/root/rivermind-fs`），实现严格零特权签名 `ground_step(rgb_history, instruction)`、共享显卡自适应显存分层、1 Hz 目标信念更新与跨中线防过冲 `VlmBeliefTracker`。
  - 在未参与训练的公寓 `sage3d_840865` 两组原生同起点异目标对上完成 `correct / swap / zero` 12 集零特权 `cf_sitl` 闭环：全程 `0` 碰撞、`0` OOM，`correct` 组在 `sofa_1` 任务上纯靠单目画面与语言引导进入目标半径（`OSR = 0.25, NE = 1.19 m`），同时定位了航拍预训练 `UAV-PUR-4B` 在室内近景 `<=1.2m` 不触发 `near_goal=1` 导致 `stop_rate = 0.0` 的真实缺口。
- 落地任务卡 SCENE-LONG（室内碰撞网格解锁跨房间长航线与户外原生长航线导入）：
  - 新建 `/root/rivermind-data/uav_rgb/envs/usd-venv` 与 `tools/usdc_to_ply.py`，拉取全部 6 个 SAGE-3D 二进制 USDC 碰撞网格并转换开启门洞的 `collision_mesh.ply`；更新 `tools/scene_import_sage3d.py`，使 6 场景可行单段路线从 `249/676 (36.8%)` 解锁至 `578/676 (85.5%)`（`>=15m` 127 条，最长 `35.68 m`），并生成同 split 跨房间多阶段 `30–55 m` 长航线 `123` 条。
  - 更新 `tools/scene_import_wanderland.py`，参数化导入 `episodes.json` 原生街道语言长航线与优化渲染参数，`wanderland_1MlUucug` 解锁 `15/18` 条（最长 A* `52.86 m`），`wanderland_10nvb` 解锁 `13/18` 条（最长测地线 `42.03 m`）。
  - 更新 `sim/3dgs/renderer.py` 与 `tools/scene_probe_quality.py`：补齐 `last_alpha`、下半幅 `alpha >= 0.50` 门禁、每 `1.0 m` 处 `±45°, ±90°` 转向扫视检查与 `< 3` 台训练相机视锥掩码。
- 落地任务卡 OF-E1/E2（OpenFly 6 场景因果扩量与 `v2` 契约升级）：
  - 新建 `tools/prepare_openfly_expansion.py`，从国内镜像拉取并逐帧校验 6 个 AirSim 场景 100 条路线的 `pose.jsonl` 与 `960` 张组首 PNG（`575 MiB`），补齐 `4/5` 升降 `±3.0 m` 原语校验，按 `< 30 m` 3D 轨迹连通走廊聚簇划分为 `80 train / 20 validation`（跨划分同场景最小 3D 距离实测 `42.79 m`）。
  - 预缓存 `bert-base-uncased` 至 `/root/rivermind-data/uav_rgb/hf-cache`，升级 `tools/train_openfly_subset.py` 与 `tools/eval_openfly_subset.py` 支持 `uavrgb.openfly_external_bc.v2`（ImageNet 归一化、冻结 BERT 词表示、自动拐点与 `STOP` 权重、仅训练期激活的 `PROGRESS_MONITOR`、带 20% 训练 dropout 的因果 `prev_action`、6 场景评测支持）。
  - 完成 1,200 updates 三臂 GPU 训练与 20 条空间隔离验证集在线/离线逐帧对拍（产物 `/root/rivermind-data/uav_rgb/exp-openfly-v2-20261116/`）：`v2_cma_rgb_language` 的验证集 `STOP F1` 从 `v1` 的 `0.333` 提升至 `0.650`（高于 `rgb_only` 的 `0.600` 与 `swap` 的 `0.596`），二周期摆头 `0` 次，20 条验证序列 `argmax` 100% 一致（最大绝对 logit 差 `5.72e-6`）。

## 没做什么

- 未删除 `/root/rivermind-data/uav_rgb/downloads/OpenFly_DataGen/ue/*.zip`（4 个未解压 Linux UE 城市场景压缩包，共 `69.45 GiB`）与 `downloads/AerialVLN/`（`38.13 GiB`）
  - 原因：属于原始未解压大场景或软链被引用的外部基准资产，当前数据盘已有 `101 GiB` 可用空间，按规范留待用户确认是否需要跑 UE 大城市或 AerialVLN 原生场景后再删。
- 未删除用户其他项目目录（`YOLOVOmodal_3871538535` 145G、`Drone-PUR` 33G、`psg_hf` 22G、`MutilModel_3871538535` 12G、`scratch_audit` 2.3G）
  - 原因：属于其他项目现役数据与单测外部依赖（且用户 `train.py` 正在 GPU 上运行），严禁触碰。
- 未启动多场景 UE4/AirSim 实例跑 OpenFly `v2` 新权重的在线闭环录像
  - 原因：晚间 GPU 正被用户另一项目的 `/opt/conda/bin/python train.py`（PID 2014203，占显存 `8.9–12.6 GiB`、利用率 `83%`）持续使用，再拉起多个 Vulkan/OpenGL UE4 城市仿真进程会争抢显存导致对方训练 OOM；已完成全部 100 路线因果扩量、1,200 步 GPU 训练与 20 路线在线/离线 `sequence_parity` 门禁验证，在线 UE4 闭环命令已写入实验报告供显卡空闲时一键运行。
- 未在 SAGE 室内帧上微调 `UAV-PUR-4B` 的 `near_goal` 近距判停阈值，也未修改 `proto/` PyTorch 原型线的 `ckpt_io.py`（仍锁 `v6`）
  - 原因：本轮聚焦打通零特权 VLM 在线闭环基础设施并诚实暴露航拍模型在室内近距不报 `near_goal=1` 的瓶颈；`proto/` 原型线已被自研 C++ `v7` 主线超越，无需同步升版。
- 未执行任务卡 `ARCH-LEAN` 的死代码删除（约 6,500 行）与 `cf_sitl` 执行器 `lib32` 改动
  - 原因：计划明文规定这两项须经用户点头批准后方可执行。

## 待确认

- 是否不再需要未解压的 4 个 Linux UE 城市场景包 `downloads/OpenFly_DataGen/ue/*.zip`（删除可再释放 `69.45 GiB`，或仅删低画质副本释放 `34.72 GiB`）？
- 是否批准执行任务卡 `ARCH-LEAN` 清理零调用历史模块（`src/device/*`、`ui/monitor.py`、`sim/tools/cf_sil.py` 等约 6,500 行）并归档移除剩余 4 个 `wt/p12-*` 脏工作树？

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 全量门禁 `UAV_RGB_BUILD_DIR=... CF_DRONE_DIR=../CF-Drone bash tools/check.sh` | 通过（`exit_code=0`，含 C++/CUDA 45/45 CTest、`read_dataset`、`cf_sitl` 9/9 CTest + 33/33 自检、`sim/tools`、`tools`、`sim/3dgs` 16 个测试文件、`ui` 全部绿灯，0 SKIP） |
| OpenFly 单元测试 `unittest discover -s tools -p 'test_*openfly*.py'` | 通过（76/76 OK） |
| 守护进程 `exp-openfly-v2-20261116`（OpenFly 6 场景扩量 + `v2` 三臂训练 + 20 集 parity） | 通过（`DONE rc=0`，跨划分最近 3D 距离 `42.79 m`，`STOP F1` `0.333 -> 0.650`） |
| 守护进程 `exp-s1-g2-scenelong-20261116`（SCENE-LONG 导入 + 卡 S1 DAgger 闭环 + 卡 G2 VLM 反事实 + Wanderland 53m 学生闭环） | 通过（`DONE rc=0`，留出集学生 `SR=0.70` vs 教师 `0.90`，户外零样本长航线学生 `SR=1.00, SPL=0.974`） |

## 风险与遗留

- 当给机载学生喂 A* 局部子目标方位（`_g0_subgoal`）时，子目标本身已绕开静态墙体，因此 `student_blind_rgb` 仍能取得高成功率（`SR = 0.80`）；要强制检验机载网络的纯视觉避障贡献，后续需用不经过 A* 的直线地标方位或 VLM 扇区方位训练/评测。
- 零特权 `UAV-PUR-4B` 在未见室内场景中虽能把无人机引导进目标半径（`sofa_1` `OSR=1, NE=1.19m`），但因航拍域差异在室内近景不输出 `near_goal=1`，下一步需用 SAGE 训练集终点近景帧做轻量 LoRA/线性探针校准 `near_goal`，或结合机载前视深度扇区触发停靠。
- `/root/rivermind-fs` 的 Inode 仍为 100% 满载（`IFree=0`），所有后续脚本必须继续保持缓存与输出落盘在 `/root/rivermind-data`（当前剩余 `101 GiB`）。
