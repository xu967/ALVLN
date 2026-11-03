# Gazebo 场景真实性与学习影响评估（改造提案）

- 日期：2026-11-03
- 性质：只做分析与方案，不改代码、不启动 Gazebo、不跑训练/评测/编译
- 判断对象：`sim/tools/scene_gen.py`、`sim/worlds/uav_rgb.sdf`、`sim/tools/collect*.py`、`sim/tools/split_manifest.py`
- 结论先行：**当前场景确实离真实世界很远，但“不够照片级”不是最该先修的问题。** 最影响学习的是三点：
  1. **几何族只有 1 个**：所有场景都是同一套三车道蛇形方柱阵，模型可能记住车道结构/动作节奏，跨场景泛化没有被证明；
  2. **模板里 3 个固定纯色 1 m 方块每集都在**：它们不在 `scene_gen` 的 pillars、`geometry_gid` 和教师真值里，却出现在学生 RGB/深度画面中，是捷径和师生不一致；
  3. **视觉自由度极低**：地面/柱/天空无纹理、单一方向光、无可见阴影、无噪声/模糊/曝光变化，图像梯度与颜色数比照片级参照低 1–2 个数量级，小 CNN 只需学颜色/边缘/位置。
- 建议顺序：**先做不改 gid 的“外观现实主义 v1”（含删除固定地标）并用配对闭环实验证明它影响策略；通过后再做几何族扩展（v6，全量重采约 7.4 h 起）。** 不建议直接上 Fuel/PBR/HDRI 照片级资产。

## 0. 硬约束核对

| 约束 | 本方案的遵守方式 |
| --- | --- |
| Gazebo Harmonic 8.15，软件 GL（llvmpipe，`DISPLAY=:20`） | 所有方案都先给渲染成本估计和 10/20 步 pilot 门限；不假设有 GPU EGL |
| 禁止 ROS 1/2、Unity、Flightmare、PyTorch/TensorRT 主栈 | 方案只用 SDF、stdlib/Pillow 生成纹理、现有 `collect*.py`/`eval_closed_loop.py`；不引入新运行时 |
| 学生输入 160×120 单目 RGB，小 CNN（3→16→32→64，约 10 万参数） | 不改输入分辨率、不改网络；场景改动只针对“学得对” |
| 几何决定 gid→split，改几何=旧数据作废 | 方案 A/D 不动 pillars 几何，`GEOMETRY_VERSION`/`SPLIT_SALT` 不变；方案 B 明确 bump 版本、全量重采；方案 C 视是否加碰撞体而定 |
| 本机离线/弱网，Fuel 是否可下载/多大/能否离线复现 | 已实测 Fuel 可达、64 KB range 约 45 KB/s；缓存里 X3 UAV 5.4 MB、Construction Cone 0.7 MB；方案 A/B 完全不用网络，方案 C 只允许把 5–20 MB 小资产 vendoring 后离线用 |
| 当前有训练在跑（`/tmp/train_students_1000_b.log`） | 第一阶段所有改动必须 opt-in，默认 `legacy` 输出字节级不变；新数据只写 `$DATA_ROOT/uav_rgb/`，不动现有 1000 集 |

## 1. 现状差距（逐项，带行号）

### 1.1 几何与布局

| # | 现状 | 证据（文件:行） | 与真实世界的差距 |
| --- | --- | --- | --- |
| G1 | 只有一族布局：三车道蛇形，柱 i 与 i+3 同车道 | `scene_gen.py:141-170`；`split_manifest.py:34` | 真实走廊/柱阵不会只有一种拓扑；模型可能记住“中-左-右”节奏 |
| G2 | 柱数只取 8–9 根 | `scene_gen.py:64` | 真实密度连续变化；当前密度的上限由“全对缝隙 ≥0.9 m”推出，不是真实约束 |
| G3 | 柱截面固定 0.4×0.4 m、高固定 3.0 m、无旋转 | `scene_gen.py:56-58`、`226-240` | 真实障碍有圆柱、墙、箱子、斜放物、不同高度；当前全是同一方柱 |
| G4 | x 位置由 `[1.5, 5.0]` 等距生成，只随 n 变 | `scene_gen.py:155-164` | x 只有 8/9 个固定网格；真实布局是连续/不规则 |
| G5 | y 只有三条车道（中心 ±0.1 m、左右 ±1.6 m），第二根随机左/右 | `scene_gen.py:62-63`、`152-154` | 横向自由度只有 0.2 m 连续 + 一个镜像位；真实横向偏移连续 |
| G6 | 走廊边界 y=±3 只存在于校验/A* 代码，不是物理墙 | `scene_gen.py:61`、`123-127` | 世界是 100×100 无限地面，飞出去没有物理反馈；教师假设的走廊学生看不到 |
| G7 | 出生点只有 default 或 random（y≈±0.72–0.88，yaw ±5°） | `scene_gen.py:191-209` | 真实起飞点/朝向更分散；当前随机量很小 |
| G8 | 模板里 3 个固定纯色 1 m 方块（红/绿/蓝，x=3，y=0/±1.5，z=1）每集都在 | `uav_rgb.sdf:78-110` | 固定位置、纯色、每场景重复；真实场景不会有“永远在同一坐标的纯色地标” |
| G9 | 上述固定方块不在 `scene_gen` 的 pillars/`geometry_gid`/教师真值里 | `scene_gen.py:82-83`、`251-268`；`split_manifest.py:51-57`；`collect.py:907` | 学生看得到、教师不知道；`clearance`、A*、`label_recompute` 都不包含它们，是师生不一致 |
| G10 | 固定方块有 collision（1×1×1 m），但巡航高度 2.3 m，不挡飞行 | `uav_rgb.sdf:82-87`、`scene_gen.py:76` | 不挡巡航，但仍在 RGB/深度视野下方；删除/改色主要影响视觉，不影响 A* 标签（需实测确认） |

### 1.2 材质、纹理、光照、天空

| # | 现状 | 证据（文件:行） | 与真实世界的差距 |
| --- | --- | --- | --- |
| M1 | 所有材质只有 ambient/diffuse/specular，没有 texture、PBR、normal map、roughness | `uav_rgb.sdf:60-110`；`scene_gen.py:226-240` | 真实表面有纹理、粗糙度、法线细节；当前是纯色塑料/灰块 |
| M2 | 柱子的 specular 直接等于 diffuse（灰柱 0.6，彩色柱也用同一 RGB） | `scene_gen.py:228-237` | 非物理高光；真实混凝土/金属的镜面反射与固有色不同 |
| M3 | 地面 100×100 m 纯灰平面，specular 0.8 | `uav_rgb.sdf:60-75` | 真实地面有沥青/水泥/草地/地砖纹理、接缝、污渍、标线；0.8 镜面也不物理 |
| M4 | 单一方向光，diffuse 0.8、specular 0.2，方向固定 | `uav_rgb.sdf:46-58` | 无时间/天气/色温变化；真实光照有方向、强度、颜色、环境光变化 |
| M5 | 没有 `<scene>`/`<sky>`/fog/background 配置 | `uav_rgb.sdf:29-58`（全文无 `<scene>`） | 背景是默认均匀浅灰；真实场景有天空、墙、天花板、远景 |
| M6 | `cast_shadows=true`，但实测帧里看不到可辨识投影 | `uav_rgb.sdf:47`；`$DATA_ROOT/uav_rgb/previews/preview_700302.png` | 缺阴影会丢失接触感/深度线索；需确认是缺 `<scene>` 配置还是 llvmpipe 限制 |
| M7 | 没有地面杂物、墙、天花板、低矮道具、动态物体 | `uav_rgb.sdf:112-136` | 真实环境有锥桶、箱子、标线、管线、行人/车辆；当前只有柱子和 3 个固定方块 |
| M8 | 颜色是纯 sRGB 常量，没有色彩管理/白平衡/曝光 | `landmarks.py:18-23`；`uav_rgb.sdf:78-110` | 真实相机在不同色温/曝光下，红/黄/蓝的像素值会漂移；L1 依赖颜色，这是 sim2real 的关键风险 |

### 1.3 相机、运动、物理

| # | 现状 | 证据（文件:行） | 与真实世界的差距 |
| --- | --- | --- | --- |
| C1 | 相机 60° HFOV、320×240、30 Hz、R8G8B8、near 0.05/far 50 | `uav_rgb.sdf:160-180` | 真实无人机相机常见 84°–120° HFOV；60° 是为 L1 可见性选的窄视场，不应作为“真实感”改动 |
| C2 | 无噪声、无镜头畸变、无运动模糊、无曝光/白平衡/滚动快门 | `uav_rgb.sdf:160-180`（无 `<noise>`/`<distortion>`） | 真实图像有传感器噪声、压缩、运动模糊、畸变；当前是理想针孔 |
| C3 | 训练时 320×240 双线性降到 160×120 | `tools/preprocess.py:56-57` | 真实板端也会降采样，但会带压缩/噪声；当前降采样后几乎是纯色块 |
| C4 | 相机装在机体前方 0.2 m，水平朝前，无俯仰/振动 | `uav_rgb.sdf:160-161` | 真实相机有安装角、机体振动、姿态抖动；当前图像完全稳定 |
| C5 | 速度控制、无风、无电机振动、无气动扰动 | `uav_rgb.sdf:31-34`、`413-428` | 真实飞行有风扰/振动/控制误差；当前物理是理想化 |
| C6 | 深度相机与 RGB 同姿态/同 FOV，仅教师使用 | `uav_rgb.sdf:184-201` | 深度是特权真值，不进学生输入；这是设计约束，不是缺陷 |
| C7 | 场景静态，无动态障碍/其他飞行器 | `uav_rgb.sdf:112-136` | 真实环境有动态物体；当前任务不要求，但会影响鲁棒性 |

### 1.4 图像统计证据（量级对比）

本仓库 L1 v4 帧（30 张，320×240）与 `$DATA_ROOT/uav_rgb/pubdata/extract-airsim10k` 的 30 张照片级合成帧对比。AirSim 10k 是**合成**数据（户外公园），不是实拍；只用作“有纹理/光照/阴影”的参照，不是真实世界基准。

| 指标 | 本仓库当前场景 | AirSim 10k 参照 | 说明 |
| --- | ---: | ---: | --- |
| 灰度标准差 | 51.9 ± 22.4 | 74.2 ± 9.2 | 当前对比度高但来自“白天空 vs 深灰柱”，不是纹理细节 |
| 平均梯度幅值 | **1.20 ± 0.70** | **19.72 ± 3.46** | 当前高频细节少约 16 倍 |
| 平均饱和度 | 0.094 ± 0.115 | 0.382 ± 0.077 | 当前大部分像素是灰/白，只有少量纯色柱 |
| RGB 量化唯一颜色数 | **35.5 ± 27.5** | **2154 ± 354** | 当前颜色多样性低约 60 倍 |
| 最大单一颜色占比 | 0.545 ± 0.136 | 0.355 ± 0.112 | 当前画面一半以上是同一种灰/白 |
| 灰度均值 | 163.7 ± 31.6 | 77.8 ± 21.1 | 当前整体偏亮、偏平 |

参照帧：`$DATA_ROOT/uav_rgb/l1-v4-relation-500/test/900101/rgb_00005.png`（当前场景）、`$DATA_ROOT/uav_rgb/previews/preview_700302.png`（14 步序列）、`sim/evidence/uav_rgb_first_frame.png`；AirSim 帧：`$DATA_ROOT/uav_rgb/pubdata/extract-airsim10k/.../rgb/009410.png`、`008022.png`。

**结论**：当前不是“照片级 vs 写实”的差距，而是“纯色块世界 vs 有纹理世界”的差距；但这条只证明观感差距，是否影响策略必须用第 2 节的实验判定。

## 2. 哪些差距真的影响学习（带判据）

### 2.1 判据总则

- **闭环优先**：离线 `acc`/`macro_f1` 高不能证明泛化（`docs/EVAL_STANDARD.md:169`）；场景改动是否值得，以闭环 `SR`/碰撞/失败集为主，离线只作辅助。
- **配对设计**：同一批 seed/场景、同一 checkpoint、同一教师，只改场景或外观；否则无法把场景效应和训练噪声分开。
- **统计功效**：n=20 的配对闭环最小可检测差异约 30 pp；n=40 约 15 pp；n=60 约 10 pp（`docs/worklog/2026-11-03-distill-round-retrospective.md`）。场景 A/B 至少 40 集，建议 60 集；报 McNemar 精确 p 和 Wilson 区间，报 failed 数。
- **先有非零基线**：如果冻结学生在 legacy 场景上闭环 SR 接近 0，先修策略/训练，不要用场景改动掩盖。不要直接引用旧 T7 的 SR 0.90（旧 observation 语义/旧几何），先在当前 v5 几何上复测 20 集建立基线。

### 2.2 逐项判断

| 差距 | 是否影响学习 | 机制 | 可执行判据 | 优先级 |
| --- | --- | --- | --- | --- |
| G8/G9 固定纯色方块 | **很可能，且是捷径** | 固定位置/纯色/每集重复；学生可用绝对颜色/位置锚点，教师真值不含它们，形成师生不一致 | 同一批位姿，渲染“有/无固定方块”两版，冻结学生离线重放：动作改变率 >20% 说明依赖；再跑 40 集配对闭环看 SR | P0 |
| G1–G7 单一几何族 | **很可能** | 车道网格/固定递进/固定形状，模型可能记住结构；`family_holdout=false` 已写在 `docs/EVAL_STANDARD.md:114`、`docs/HANDOFF.md:772` | 族留出：训练 A+B 族，测 C 族（未见布局），n≥40；与同族留出 seed 比，SR 下降 ≤15 pp 才说明现有族足够；>30 pp 说明必须扩族 | P0（但改动最贵） |
| M1–M3 无纹理/材质非物理 | **待验证，很可能** | CNN 只需颜色/边缘/位置就能分区；真实纹理/高光会改变特征分布 | 同几何、只换纹理/地面材质，冻结学生离线重放 + 40 集闭环；动作改变率 <5% 且 ΔSR <5 pp → 不是当前瓶颈 | P1 |
| M4/M6 恒定光照/无阴影 | **待验证** | 固定光照让模型不学颜色恒常性；阴影是深度/接触线索，但也可能干扰颜色分割 | 3 组光照（太阳方向/强度/色温）同几何，冻结学生闭环；ΔSR >10 pp 说明需要光照随机化 | P1 |
| M5 天空/背景纯灰 | 主要观感，可能影响特征多样性 | 缺少远景/背景 distractor；但当前任务只在前景柱阵 | 与 M1 同一次外观 A/B 一起测；若单独做，优先级低于纹理 | P2 |
| C2 相机理想（无噪声/模糊/畸变/曝光） | **待验证** | 真实图像噪声/模糊会造成域偏移；适度噪声也可能正则化 | 在预处理/评测图像上加噪声、模糊、伽马、JPEG 压缩，冻结学生闭环；ΔSR >10 pp 说明需要增强 | P1 |
| C1 FOV/分辨率 | **不要改** | 60° HFOV 是 L1 可见性/关系词判定的前提（`landmarks.py:15-24`、`HALF_FOV_DEG=30`）；改 FOV 会改 `assign_landmarks` 的可见柱对，等于改任务 | 任何 FOV/分辨率改动都必须重跑 landmark 可见性自检并重采；不作为“真实感”改动 | 禁止（本阶段） |
| C4/C5 相机/机体理想运动 | 低，可能影响鲁棒性 | 无振动/风扰；0.3 m/s 下运动模糊有限 | 加相机抖动/模糊后闭环；若 ΔSR <5 pp，不做 | P2 |
| G6 走廊无物理墙 | 待验证 | 教师假设 y∈[-3,3]，学生可能飞出；但当前 waypoint 在走廊内 | 统计闭环轨迹 |y|>3 的比例；若 <5%，不加墙 | P2 |
| M7 无杂物/动态 | 主要观感 | 真实环境有 distractor；但新增 3D 物体会进深度相机，可能改教师标签 | 若加，只加在巡航高度以下且不在教师 A* 路径；先验证 `label_recompute` 不变 | P2 |
| M8 颜色无色彩管理 | **可能影响 L1** | L1 靠颜色选目标；真实白平衡/曝光会让色值漂移 | 用反事实指令对（`make_counterfactual.py`）测颜色判别；在白平衡/伽马变化下看目标色选择是否翻转 | P1 |

### 2.3 已有证据与不能下的结论

- **颜色可见性成立**：40 个默认出生集 × 40 帧，目标色 40/40 出现，step 0 有 33/40 可见（`docs/experiments/2026-10-31-l1-data-health.md:53-60`）。这只证明“任务视觉前提成立”，**不证明**模型没有用固定地标/绝对颜色。
- **场景族单一已被文档承认**：`docs/EVAL_STANDARD.md:114`、`:169`；`sim/tools/split_manifest.py:65` 会打印 `family_holdout=false` 警告。任何“高准确率”都不能当作跨族泛化。
- **当前策略模型可能先在训练/时序上失败**：`docs/worklog/2026-11-03-distill-round-retrospective.md` 记录过教师模型在训练过的 train split 上闭环 SR 0.1、0 次转向（根因是闭环窗口 warm-up，不是场景）。做场景 A/B 前必须先确认冻结 checkpoint 在 legacy 上有非零 SR；否则场景效应测不出来。
- **没有找到实拍帧**：`$DATA_ROOT/uav_rgb/` 里只有 Gazebo 帧和 AirSim 10k（合成）；AirSim 不能替代真实世界验证。真实相机参数（FOV/噪声/曝光）也没有硬件规格（`CLAUDE.md` 说 LicheePi/ESP32 协议未确认）。

## 3. 方案对比

### 方案 A：外观现实主义 v1（不动 gid，推荐先做）

**目标**：在不改变 pillars 几何、不改变 `geometry_gid`/split 的前提下，去掉固定地标捷径，加入可控的纹理、光照、天空、地面和轻量图像域噪声，让模型不能只靠“纯色块 + 固定位置”。

**改哪些文件**

- `sim/worlds/uav_rgb.sdf`：加 `<scene>`（ambient/background/shadows）；把地面/固定地标材质参数化；保留 X3/相机/深度/contact 不动。
- `sim/tools/scene_gen.py`：加 `--appearance legacy|v1`，默认 `legacy`；v1 时替换光照/天空/地面材质，并删除或灰化 3 个固定方块；新增 `appearance` 元数据（纹理 sha256、光照参数）。**不改** `PILLAR_*`、`LANE_SEP`、`SPAWN_*`、`GEOMETRY_VERSION`。
- 新增 `sim/worlds/textures/`（4–6 张 256/512 px 程序化 PNG，总计 <1 MB）和可选 `sim/tools/make_textures.py`（stdlib/Pillow，固定 seed，输出 sha256）。也可以直接把生成好的 PNG vendoring 进仓库，避免采集时依赖 Pillow。
- `sim/tools/split_manifest.py`：只在 manifest 里记录 `appearance_version`，**不改** `SPLIT_SALT`/`split_for_gid`。
- `tools/preprocess.py`（可选）：加 `--augment photometric`（亮度/对比度/伽马/白平衡/噪声/模糊/JPEG），默认关闭；记录在 dataset meta。
- `sim/tools/collect*.py`、`eval_closed_loop.py`：把 `--appearance` 透传给 `scene_gen`；不改采集协议。
- `sim/tools/test_spawn.py`：补 legacy 输出 sha256 与 v1 确定性用例。

**几何/gid/旧数据**：pillars 的 x/y/z/size 全部不变，`geometry_gid` 不变，`GEOMETRY_VERSION`/`SPLIT_SALT` 不变，`tools/preprocess.py:104-108` 的版本检查不受影响；旧数据仍可打包/训练。但模板 `template_sha256`/`family_id` 会变（`split_manifest.py:61-78`），且旧图像没有新外观；**为干净 A/B，旧/新外观不要混在同一次训练里**，用 `appearance_version` 标记，新数据单独目录。

**渲染成本估计**：只改材质/光照，不增加几何/绘制调用；320×240 下地面约 40–60% 像素，1–2 层纹理采样 + 简单光照，估计每步墙钟 +5–20%；阴影 pass 对 <200 三角形场景估计 +5–15%；合计目标 ≤1.25× legacy。若 PBR/法线/1024² 纹理导致 >1.5×，降级为纯色 + 程序化噪声，不上 PBR。

**可复现性**：程序化纹理固定 seed，输出 sha256 写进 `scene.json`；不依赖网络/Fuel；PNG 总量 <1 MB，可随仓库分发。建议服务器上生成一次后 vendoring，采集时不跑 Pillow，避免 Pillow 版本差异。

**怎么证明值得做**

1. **渲染成本 pilot**：1 实例 × 10 步、4 实例 × 20 步，legacy vs v1，记录 `wall/step`、RTF、失败/超时；门限 ≤1.25×，否则降级。
2. **离线重放**：同一批位姿，legacy/v1 各渲染一次，冻结学生逐帧推理；动作改变率 >20% 说明模型对固定地标/外观敏感。
3. **配对闭环**：40–60 个相同 seed，冻结学生在 legacy 与 v1 上各跑一次；McNemar 精确检验。ΔSR ≥15 pp（n=40）或 ≥10 pp（n=60）→ 外观/捷径真的影响策略；ΔSR <5 pp → 外观不是瓶颈，转方案 B。
4. **训练侧验证（可选）**：用 v1 采 100–150 集，训练同配方学生；在 legacy 和 v1 上互测。若 v1 训练后 legacy SR 不降、v1 SR 明显升，说明外观随机化有正则化收益。

**结论**：成本最低、可逆、不废数据，适合第一周落地；但它**不解决几何单一性**，不能把结果说成“泛化提升”。

### 方案 B：几何族扩展（v6，全量重采）

**目标**：从“1 个车道族”扩展到 3–4 个布局族，并做族留出，真正测跨布局泛化。

**改哪些文件**

- `sim/tools/scene_gen.py`：新增 `FAMILY` 参数和采样器；候选族：
  - F0 当前 lane3_serpentine（保留作对照）；
  - F1 随机矩形/圆柱混合，尺寸 0.2–0.8 m，高 2–4 m，带 ±15° 旋转和位置抖动；
  - F2 走廊/隔墙/开口（墙厚 0.2–0.4 m，留可通行门洞）；
  - F3 近距障碍开局/密集簇（为 SAFE 类提供真实近障，而不是靠贴柱面）。
- `sim/tools/split_manifest.py`：**split 改成按 family 分**，否则同一族的 train/test 会混在一起，`family_holdout` 只是报告字段（`split_manifest.py:145`）。`GEOMETRY_VERSION` bump 到 `uavrgb.geom.v6`，`SPLIT_SALT` 随之变。
- `sim/tools/landmarks.py`：`VISIBLE_X_MAX`/`LATERAL_MIN`/`LATERAL_SPAN_MAX`/`HALF_FOV_DEG` 需要按新族重算；`waypoint_for` 的 0.8 m standoff 和 A* 余量要复测。
- `sim/tools/teacher.py`/`collect*.py`：A* 栅格/clearance/`ray_clear` 对非方柱、墙、旋转体要重新验证；`audit_dataset.py` 的 `GEOM_CONST` 要跟着改。
- 测试：`test_spawn.py`、`test_landmarks.py`、`test_split_manifest.py`、`test_collect*.py` 全量更新。

**几何/gid/旧数据**：全部变。`GEOMETRY_VERSION`/`SPLIT_SALT` bump 后，`tools/preprocess.py:104-108` 会拒绝旧 manifest，`audit_dataset.py:577-579` 也会 FAIL；现有 1000 集（`ds-teacher1000`）**全部作废**。全量重采按 4 实例 135 集/h 约 7.4 h（`docs/experiments/2026-10-30-t5-scaleup.md:53-54`）；多族通常需要 2000–3000 集，约 15–22 h，另加打包/训练/评测。旧 checkpoint 不能与新数据直接比较。

**渲染成本估计**：几何/绘制调用增加；圆柱 32 三角形、墙/箱子几十个；颜色 pass 和深度 pass 都变重；阴影 pass 可能翻倍。估计每步墙钟 +20–50%，吞吐 1.5→0.75–1.2 步/s；4 实例 RTF 可能从 0.25–0.5 掉到 0.2–0.4，接近起飞 120 s 超时边缘（`collect_parallel.py:12-15`、`88-95`）。必须先用 20 步 pilot 测。

**可复现性**：全程序化，无外部资产；固定 seed + `scene_gen --self-check`；版本号/常量写进 `scene.json`/manifest。比方案 C 更容易离线复现。

**怎么证明值得做**：族留出实验。训练 F0+F1，测试 F2（未见族），n≥40；同时测 F0 的同族留出 seed。判据：族留出 SR 下降 ≤15 pp 说明现有族多样性够；>30 pp 说明必须扩族。若下降主要来自训练/数据量而不是族，先加数据再扩族。注意：当前 split 按 gid 不按族，必须先改 split，否则“族留出”是假的。

**结论**：这是解决“太单一”的正解，但成本最高、旧数据全废；应作为方案 A 之后的门控阶段，不要在当前训练/数据周期里直接切。

### 方案 C：照片级资产（Fuel/PBR/HDRI）

**目标**：用 Fuel 模型、PBR 材质、HDRI 天空、阴影/环境光遮蔽，让画面接近照片。

**改哪些文件**：`sim/worlds/uav_rgb.sdf`、`scene_gen.py`、新增 `sim/worlds/models/`/`textures/`/`hdri/`、`GZ_SIM_RESOURCE_PATH` 配置；若加碰撞体，还要改 `scene_gen`/`teacher`/`audit`。

**几何/gid/旧数据**：只替换现有表面材质 → gid 不变、旧数据可用；一旦加入 Fuel 道具/墙/碰撞体 → gid 变、旧数据作废。多数“照片级”方案会引入新几何，实际接近方案 B 的数据代价。

**资源与可复现性**：Fuel 可达；本次实测 64 KB range 约 45 KB/s；`~/.gz/fuel` 已缓存 X3 UAV 5.4 MB、Construction Cone 0.7 MB；`sim/evidence/camera_sensor_first_frame.png` 里已有 cone 渲染证据。5–20 MB 的小资产集可接受（理论 2–7 min，实际预留 20–60 min），100 MB 以上不现实。**离线复现必须 vendoring 到仓库**（`sim/worlds/models/`，记录 license/sha256），不能依赖 `~/.gz` 或运行时下载。

**渲染成本估计**：PBR + 法线/粗糙度贴图 + HDRI + 阴影 + 更多 draw call，软件 GL 下 RGB pass 显著变重；估计每步墙钟 +50–150%，吞吐可能 0.5–0.9 步/s，4 实例 RTF 可能 <0.3，触发起飞超时。**先做 20 步 pilot；>1.5× 直接否决。**

**怎么证明值得做**：先只替换 1–2 个材质/一个天空，测渲染成本；再做 20–40 集闭环，与方案 A 比。若方案 A 已经能带来大部分收益，C 的额外收益不值得采集吞吐损失。结论：**不建议作为第一步**；学习收益不确定，渲染成本最高。

### 方案 D：图像域增强（零渲染成本，补充）

**目标**：不改 Gazebo，只在预处理/训练侧加光度/噪声/模糊/压缩增强，教模型对真实相机变化不变。

**改哪些文件**：`tools/preprocess.py` 或训练数据加载器；新增 `--augment photometric` 配置；记录到 dataset meta。不改 `scene_gen`/SDF。

**几何/gid/旧数据**：不变；旧数据可复用；零 Gazebo 渲染成本（CPU 预处理每帧毫秒级）。

**怎么证明值得做**：同一数据集，训练“无增强 vs 有增强”两组；在 legacy、方案 A 的 v1、以及人工噪声/模糊测试集上闭环。若增强能缩小 legacy→v1 的 SR 差距，说明部分“真实感”问题可以在数据侧解决，不必全押在场景上。

**结论**：不是场景改造的替代，但应作为方案 A 的配套；成本几乎为零，建议直接纳入 A/B 的一个对照臂。

### 方案对比表

| 方案 | 主要改动 | 几何/gid/旧数据 | 渲染成本估计 | 资源/复现 | 证明实验 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| A 外观现实主义 v1 | 删固定地标；纹理/光照/天空/地面；可选图像增强 | gid 不变；旧数据工具上仍合法；新旧外观不要混训 | +5–25%（目标 ≤1.25×） | 程序化 PNG <1 MB，无网络 | 40–60 集冻结学生配对闭环；渲染 pilot | **推荐先做** |
| B 几何族扩展 v6 | 多族布局/形状/墙/密度；族级 split | gid 全变；旧 1000 集全废；全量重采 ≥7.4 h | +20–50% | 全程序化，可复现 | 族留出：训练 A+B，测 C；n≥40 | 战略正解，第二阶段 |
| C 照片级资产 | Fuel/PBR/HDRI/道具 | 看是否加碰撞体；多数会废数据 | +50–150%，可能 RTF<0.3 | 5–20 MB vendoring；45 KB/s 实测 | 20 步渲染 pilot + 与 A 比 | 不建议第一步 |
| D 图像域增强 | 预处理/训练侧光度增强 | 不变；旧数据可用 | ≈0 | 无 | 训练有无增强对比；跨 legacy/v1 闭环 | 作为 A 的配套 |

## 4. 推荐方案与第一周第一步

### 4.1 推荐

**推荐方案 A，方案 D 作为 A 的配套；方案 B 作为 A 通过后的第二阶段；方案 C 不做。**

一句话：**先删掉固定纯色地标、把地面/柱/天空/光照做成确定性的“外观 v1”，用 40–60 集冻结学生的配对闭环证明外观/捷径真的影响策略；通过后再为几何族扩展付全量重采的代价。**

为什么不是先做 B：B 会让现有 1000 集和当前训练全部作废（重采 ≥7.4 h，多族更像 15–22 h），而 A 可逆、不碰 gid、能在一周内给出“值不值得”的答案。A 的 null result 不是“场景不重要”，而是“当前任务的瓶颈不在外观，应转 B 的族留出 pilot”。

### 4.2 第一周落地步骤

**第 1 步（半天）**：在 `sim/tools/scene_gen.py` 加 `--appearance legacy|v1`，默认 `legacy`；保证 seed 0..99 的 `world_sha256` 与当前字节级一致（先跑 `scene_gen --self-check --check-seeds 100` 和 sha256 对照）。v1 先做两件事：删除模板 3 个固定纯色方块（或按 task 只保留 base），在 `uav_rgb.sdf` 加 `<scene>`/光照参数。

**第 2 步（1 天）**：加 4–6 张程序化纹理（地面沥青/水泥、柱面混凝土/金属、墙/天空），写 vendoring 的 `sim/worlds/textures/` 和 sha256；v1 按 seed 随机选纹理/光照方向/强度/色温/环境光；记录 `appearance_version`、纹理 sha256、光照参数到 `scene.json`。**不碰** `PILLAR_*`/`LANE_SEP`/`GEOMETRY_VERSION`。

**第 3 步（半天）**：渲染成本 pilot。1 实例 × 10 步 legacy vs v1；4 实例 × 20 步 legacy vs v1。门限：`wall/step` ≤1.25× legacy、无起飞超时、无相机 segfault、RTF 不低于 0.2。超过门限就砍阴影/PBR，只保留程序化纯色 + 光照。

**第 4 步（1 天）**：配对数据。用同一批 40–60 个 seed，采 legacy 与 v1 各一轮（或复用已有 legacy 子集 + 新采 v1）；`--appearance v1` 透传到 `collect_parallel.py`/`eval_closed_loop.py`。新数据写 `$DATA_ROOT/uav_rgb/sim-realism-2026-11/`，不写仓库。

**第 5 步（1 天）**：冻结学生（用 `$DATA_ROOT/uav_rgb/distill-1000-20261103/student-base/ckpt_best.bin` 或当前已完成的 best；不要用正在训练的 a50 目录），在 legacy/v1 上各跑 40–60 集闭环；先确认 legacy SR 非零。报 SR、collision、stuck、failed、McNemar；同时做离线同姿态重放的动作改变率。

**第 6 步（半天）**：决策。
- ΔSR ≥15 pp（n=40）或 ≥10 pp（n=60）且渲染 ≤1.25× → 扩 v1 到 300–500 集，重训并复测；
- ΔSR <5 pp → 外观不是瓶颈，停止扩外观，转方案 B 的 3 族 × 100–150 集族留出 pilot；
- 介于 5–15 pp → 加 n 到 60–80，或把 D（图像增强）作为低成本补充再测。

### 4.3 建议的 pilot 命令（本轮不执行）

```bash
# 1) legacy 必须字节不变
python3 sim/tools/scene_gen.py --self-check --check-seeds 100
python3 sim/tools/scene_gen.py --seed 0 --out /tmp/w_legacy.sdf --appearance legacy
python3 sim/tools/scene_gen.py --seed 0 --out /tmp/w_v1.sdf     --appearance v1
sha256sum /tmp/w_legacy.sdf /tmp/w_v1.sdf

# 2) 渲染成本 pilot（20 步，4 实例）
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 sim/tools/collect_parallel.py \
  --instances 4 --episodes-per-instance 1 --steps 20 --hold 1.0 \
  --base-seed 0 --appearance v1 --out /tmp/pilot_v1

# 3) 配对闭环（示例；--appearance/--seeds 需先透传）
python3 sim/tools/eval_closed_loop.py --policy student \
  --ckpt "$DATA_ROOT/uav_rgb/distill-1000-20261103/student-base/ckpt_best.bin" \
  --episodes 40 --seeds 0:40 --appearance legacy --out /tmp/eval_legacy
python3 sim/tools/eval_closed_loop.py --policy student \
  --ckpt "$DATA_ROOT/uav_rgb/distill-1000-20261103/student-base/ckpt_best.bin" \
  --episodes 40 --seeds 0:40 --appearance v1 --out /tmp/eval_v1
```

## 5. 风险与未查证项

| 项 | 影响 | 处置/验证 |
| --- | --- | --- |
| 渲染成本只是估算，未在本机实测 | 软件 GL 下可能超 1.25×，采集变慢 | 第 3 步 10/20 步 pilot；>1.5× 降级为纯色/无阴影 |
| SDF 1.6 的 `<pbr>`/纹理标签在 Harmonic 8.15 + Ogre2/llvmpipe 下是否可用未验证 | v1 可能渲染不出纹理 | 先做 1 个材质 smoke；失败就用纯色/程序化顶点色，不硬上 PBR |
| `cast_shadows=true` 但当前看不到阴影，原因未定 | 加 `<scene>` 后可能仍无阴影，或成本超预期 | 单独做一个有/无 `<scene>` 的对照帧；确认是配置还是 llvmpipe 限制 |
| 删除固定方块会改变深度相机画面 | `expert_action` 的深度标签可能变；l1 A* 用 GT pillars 应不受影响 | 用 5 集跑 `label_recompute`；只对 `expert_action` 集做门控；l1 先用 A* 验证 |
| 外观改动可能对闭环 SR 无显著影响 | 白花一周 | 这是 A/B 的目的；ΔSR <5 pp 就停，转 B |
| 外观改动可能伤害 L1 颜色判别 | 纹理/光照让红/黄/蓝混淆，SR 反降 | 颜色板固定为 `landmarks.COLOR_RGB` 附近；光照随机化设上限；用反事实指令对测颜色选择 |
| 统计功效不足 | n=20 测不出 15 pp | n≥40（MDE 15 pp），建议 60（MDE 10 pp）；报 McNemar/Wilson |
| 方案 B 的族级 split 改动会影响旧 gid | 旧数据全废 | 只在 v6 做；旧数据不原地重标；全量重采预算 ≥7.4 h，多族 15–22 h |
| `template_sha256`/`family_id` 会变 | 报告里的 family 变化；不影响 split（`split_for_gid` 只用 gid） | 在 manifest 记录 `appearance_version`；不要用 family_id 做 split |
| Fuel 下载慢/资产许可 | C 方案不可复现 | 只允许 vendoring 5–20 MB 并记录 license/sha256；不在运行时下载 |
| 没有实拍数据 | 无法证明 sim2real | AirSim 10k 只是照片级合成；真实飞行验证另立任务 |
| 当前训练正在跑 | 改默认模板/数据会污染正在跑的实验 | 所有改动 opt-in；默认 legacy 字节不变；新数据只写 `$DATA_ROOT`，不动现有 run |
| 固定方块不在 `geometry_gid` | 删除后旧数据仍能通过 preprocess，可能被误混 | 加 `appearance_version`；A/B 训练集不混旧/新外观；审计报告里显式写 |

### 未查证项（明确未做）

- 没有启动 Gazebo，所以**没有实测**纹理/阴影/多几何对 RTF、step 墙钟、起飞超时的真实影响；第 3 步 pilot 是必做门限。
- 没有验证 SDF 1.6 `<pbr>` 在 Ogre2/llvmpipe 的实际支持；没有验证 `<scene>` 能否让当前阴影出现。
- 没有验证删除固定方块后 `expert_action` 的 `label_recompute` 是否全过；只从代码推断 l1 A* 不受影响。
- 没有查证一套完整的 Fuel 小资产清单/许可/总大小；只看了缓存里的 X3 UAV 5.4 MB、Construction Cone 0.7 MB，并用 64 KB range 测了约 45 KB/s。
- 没有实拍帧，无法给出真实相机 FOV/噪声/曝光参数；没有做 sim2real 验证。
- 没有运行方案 A/B 的闭环，所有 ΔSR 门限是统计设计值，不是实测结果。
- 没有改代码；文中出现的 `--appearance` 是提案接口，当前 `scene_gen.py`/`collect_parallel.py`/`eval_closed_loop.py` 还没有这个参数。
