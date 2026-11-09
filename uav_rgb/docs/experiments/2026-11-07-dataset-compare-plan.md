# 两个数据集对比：Gazebo 柱阵→森林（A）vs 3DGS 校园转弯路线（B）（2026-11-07 04:40 立）

- 用户要求：「柱阵 → 森林三档」为一个数据集、「3DGS → 转弯路线」为一个数据集，做对比。两条线都要三视角视频。
- 载具统一 E1 MINI 固件在环（`--dynamics cf_sitl`），动作四元，observation 语义，hold：Gazebo 1.0 s、3DGS 0.5 s（历史口径不同，见「待确认」）。
- 口径：`docs/EVAL_STANDARD.md`；E1 数字与历史 X3 不混报。

## 1. 数据集定义

| | A：Gazebo 柱阵 + 森林 | B：3DGS Callinstrasse 转弯路线 |
| --- | --- | --- |
| 场景 | `scene_gen --task base`（柱阵，几何 v5）+ `--task forest_e1 --density sparse/medium/dense`（随机圆柱森林，只收教师能过的档位） | `3dgs-scenes/callinstrasse` 真实校园街景，≥4 条带转弯/避障的路线（route 0 = 原直线） |
| 采集器 | `collect_parallel.py`（4 实例，软件渲染，RTF 锁 1.0） | `sim.3dgs.collect_3dgs`（GPU gsplat，3 ms/帧） |
| 教师 | 柱子真值 A*（E1 余量 0.40/0.30 + 脱困 v2） | 体素占据 A*（同一套 teacher.plan_action_global，占据代理当柱子） |
| 规模 | base 300 集（进行中）+ 森林 ≥150 集 | 先 20 集 pilot（进行中），过检后放到 ≥300 集（4 路线 × 出生扰动 × seed） |
| 观测 | 320×240 RGB，10 Hz 相机 | 320×240 RGB，3DGS 渲染（SH3 + 2× 超采样） |
| 测试集 | 20 个冻结 seed = `tools/eval_seeds.EVAL_SEEDS_BASE_20`（20 独立几何，与两个学生的训练几何零重叠；2026-11-09 起唯一来源，计划初稿写的 `180,206,…,1282` 按 gid 只有 11 个几何、已淘汰）；森林 `FOREST_SPARSE_SEEDS_20` | 留出路线/留出出生扰动（route 列表里标 split=test） |
| 产物目录 | `$DATA_ROOT/uav_rgb/e1-pipeline-20261107-0417/`（base）+ forest agent 的目录 | `$DATA_ROOT/uav_rgb/datasets/e1-3dgs-callinstrasse-*` |

两者 episode 目录格式都是 `collect.py` 口径（`episode_manifest.json` + `actions.jsonl` + `rgb_*.png`），`tools/preprocess.py` 直接打包；B 的 manifest 多出 3DGS 身份字段，preprocess 已支持（HANDOFF_3DGS §五 3）。

## 2. 对比矩阵（学生 = 同一结构同一超参，只换数据）

| 学生 | 训练集 | 在 A 测试集闭环（Gazebo 20 seed） | 在 B 测试集闭环（3DGS 留出路线） |
| --- | --- | --- | --- |
| 教师（上界） | — | 跑 | 跑 |
| 盲前进（下界） | — | 跑 | 跑 |
| 旧 X3 学生 | X3 599 集（历史） | 跑（载具迁移差距） | 跑 |
| S_A | A | **域内** | 跨域 |
| S_B | B | 跨域 | **域内** |
| S_AB | A ∪ B | 混合 | 混合 |

报：SR（分母 = 请求集）、碰撞率、stuck、false_forward（E1 coast 口径）、推理 p50/p95、动作分布，n 与 Wilson 95% 区间。看的三个问题：① 合成几何（A）学到的避障能不能迁到真实外观（B）；② 真实外观但路线少（B）会不会过拟合路线；③ 混合是否两边都不掉。

## 3. 工具

- A 侧：`tools/e1_pipeline.py`（已在跑 base；forest agent 加 `COLLECT_SETS`）。
- B 侧：3DGS agent 产出路线定义 + `collect_3dgs --route`；评测 `sim.3dgs.eval_3dgs --policy student --ckpt --infer-bin`（已支持 student/forward；教师组用 collect_3dgs --policy teacher 的到达/碰撞统计）。
- 对比入口（待写，等两个 agent 的 CLI 定下来）：`tools/compare_datasets.py`，配置区列 DATASETS={A,B,AB} 与 EVAL_DOMAINS={gazebo,3dgs}，顺序：打包 ×3 → 训练 ×3（设备常驻，每个 ≈5 min）→ 评测 2 域 × 6 组 → `summary.md`；守护进程跑。

## 4. 时间

| 步 | 预计 |
| --- | --- |
| A base 300 集采集（进行中，04:25 起） | ≈ 65 min |
| A 森林接入 + 三档教师实测（agent） | 1–2 h |
| B 路线 + 20 集 pilot（agent） | 1–2 h |
| B 放量 ≥300 集（3 ms/帧，CPU 侧 cf_sitl_proc） | ≈ 30–60 min |
| 打包 + 训练 3 个学生 | ≈ 20 min |
| 评测 2 域 × 6 组 | Gazebo ≈ 40 min（4 并行），3DGS ≈ 20 min |

## 5. 待确认

- hold：Gazebo 线 1.0 s、3DGS 线 0.5 s 是历史口径；跨域评测时学生面对的「每步位移」不同（0.25 m vs 0.12 m）。建议 B 放量时也用 hold 1.0，两边一致；否则表里要标注。
- B 的教师是占据代理 A*，代理有误差（voxel 0.2 m、inflate 0.15 m）；教师组 SR 的「上界」含代理误差。
- 3DGS 画质（见 worklog 04:40 更新）：Callinstrasse 是街面手持采集，2M 高斯铺一整条街，偏离采集路径/高度后发糊；是否换/增场景（H3DGS Campus/SmallCity 预训练、自采校园）由用户定。

## 6. 这个项目现在是不是 VLN（给老师看时怎么定位）

对照 AerialVLN（ICCV 2023：8,446 条轨迹、25 个城市场景、平均航程 662 m / 204 步、指令平均 83 词、成功阈值 20 m，基线 unseen SR 1–4%）、CityNav、OpenUAV、OpenFly（100K 轨迹、18 场景，含 3DGS 渲染场景）这些基准：

- 我们现在做的是**微型无人机（100 g）单目 RGB 的语言条件反应式导航**：四动作、航程 10–25 m、指令是 22 词模板（颜色 + 关系词 → waypoint，L1 任务），学生是 ~150k 参数的 CNN（为 SG2002 NPU）。按基准定义，它还**不算 VLN**：没有自然语言路线描述、没有地标级 grounding、没有长程记忆、指标口径（1.5 m 成功半径）不可比。
- 但它是 VLN 的**必要底座**，且有基准工作没有的东西：真实固件在环的动力学、真机可部署的模型尺寸、sim-to-real 的完整链路、3DGS 真实场景（OpenFly 也用 3DGS 当环境，说明路线对）。`docs/ROADMAP.md` 阶段 3 的门禁写得对：「正确指令相对打乱指令的 SR 增益显著，才叫语言导航」。
- 从现在到「能叫 VLN」最短路径：① B 数据集的路线配上地标指令（「朝红顶楼飞，到停车场前左转」），词表 ≤200；② 评测三组对照（正确指令 / 打乱指令 / 盲前进）+ SR/OSR/NE/SPL；③ 学生加短时序（堆帧或 GRU）解决「停得准」；④ 对外定位为 **TinyAerialVLN：面向微型无人机的小模型 VLN 与 sim-to-real**，与 AerialVLN-S 在缩小尺度上对照，而不是硬比它们的城市级数字。
