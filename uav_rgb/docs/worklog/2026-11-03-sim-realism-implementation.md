# 场景逼真化落地实施方案（外观 v2：纹理 / 光照 / 成像 / 内容）

- 日期：2026-11-03
- 性质：只做设计，不新建代码/资产，不启动 Gazebo，不跑训练/评测/编译
- 承接：`docs/worklog/2026-11-03-sim-scene-realism-proposal.md`（现状评估与方案 A/B/C/D 取舍）
- 已完成：`docs/worklog/2026-11-03-0840-appearance-v1.md`（删除 3 个固定纯色地标 + `<scene>`/随机太阳，legacy 字节不变，实测渲染成本约 0.98×）
- 一句话现状：捷径已删、光照已动，但地面/柱子/背景仍是纯色块；本文给出“在不改 `geometry_gid` 的前提下，把画面做成有纹理、有噪声、有遮挡关系的场景”的可执行方案。

## 0. 硬约束核对

| 约束 | 本方案的遵守方式 |
| --- | --- |
| Gazebo Harmonic 8.15，软件 GL（llvmpipe），`LP_NUM_THREADS=1`，4–6 实例并发 | 只加 1 次 albedo 采样、少量三角面/绘制调用；阴影/天空/道具分别做 10/20 步成本 pilot；每项都给降级路径 |
| 禁止 ROS/Unity/Flightmare/PyTorch/TensorRT | 只用 SDF、Pillow 生成并 vendoring 的 PNG/OBJ、现有 `scene_gen.py`/`collect*.py`/`preprocess.py` |
| 几何决定 `geometry_gid` → split，改几何旧数据作废 | 纹理/光照/成像/无碰撞装饰都不改 `pillars`，gid 不变；明确列出“唯一会废数据”的改动 |
| 离线/弱网，不下载模型库 | 全部程序化资产 <5 MB，随仓库 vendoring；不访问 Fuel |
| 机器可能在跑训练与 Gazebo pilot | 本文所有实验都写成“待训练结束后执行”，本轮不启动任何进程 |
| 学生输入 160×120、4 动作小 CNN | 所有视觉参数按“320×240 渲染 → 双线性降到 160×120”后的效果验收 |

## 1. 分层方案

### 1.1 材质与纹理

#### 1.1.1 技术选型（按本机已安装的二进制定）

本机是 sdformat14 + gz-sim 8.15 + gz-rendering 8.3 + Ogre2，SDF 声明是 1.6。对 `/usr/share/sdformat14/1.6/material.sdf`、`/lib/x86_64-linux-gnu/libgz-sim8-rendering.so.8`、`libgz-rendering8-ogre2.so` 的检查结论：

| 写法 | 是否可用 | 依据 / 注意 |
| --- | --- | --- |
| `<material><script><uri>...<name>...` | **不可用** | gz-sim8 二进制内含 `Custom material scripts are not supported.` / `Gazebo does not support Ogre material scripts.`。不要写 Ogre `.material`。 |
| `<material><pbr><metal>` + `albedo_map` / `roughness_map` / `metalness_map` | **可用** | `libgz-sim8-rendering` 的 `LoadMaterial` 读取 `PbrWorkflow::AlbedoMap/RoughnessMap/MetalnessMap`，用 `gz::common::findFile` 解析路径。只支持 **metal workflow**，二进制明确写 “PBR material: currently only metal workflow is supported.” |
| `<pbr><specular>`（specular/glossiness workflow） | **忽略** | 同上，SceneManager 只走 metal。 |
| `normal_map` / `environment_map` | SDF 语法支持，但 gz-sim 很可能忽略，**不建议** | `libgz-sim8-rendering` 里只看到 `AlbedoMap/RoughnessMap/MetalnessMap` 的读取；法线还需切线空间和第二次采样，环境反射在 llvmpipe 上很慢。160×120 下收益低。 |
| `<material><diffuse>` 等纯色 | 可用（现状） | 作为纹理失败时的降级路径。 |

**选择：只上 albedo map + 常数 roughness/metalness，不上 normal/roughness/metalness map、不上 environment map、不上脚本材质。**

#### 1.1.2 资产清单（程序化生成，vendoring，固定 seed）

新增（允许新建的资产目录，本轮不创建）：

```
sim/worlds/textures/
  ground_asphalt_a_512.png      # 沥青 A：灰底 + 细噪声 + 裂纹 + 油渍
  ground_asphalt_b_512.png      # 沥青 B：带白色虚线/停车线/接缝
  ground_concrete_512.png       # 水泥地：接缝 + 污渍
  ground_tile_512.png           # 地砖：4×4 砖 + 砖缝
  pillar_concrete_256x512.png   # 混凝土柱面：竖向流痕 + 模板缝
  pillar_metal_256x512.png      # 拉丝金属：竖向细条纹 + 锈点
  pillar_paint_256x512.png      # 旧漆面：小面积剥落
sim/worlds/meshes/
  ground_uv.obj                 # 100×100 m 两块三角面，UV 0..25（4 m 重复一次）
sim/worlds/textures/manifest.json  # 每个文件的 sha256、尺寸、生成参数、seed
```

- 尺寸/体积估算：512² RGB PNG 带噪声约 0.4–0.8 MB/张，4 张约 2–3 MB；256×512 柱面 3 张约 0.3–0.8 MB；OBJ <2 KB。**总计 <5 MB**。
- 生成方式：新增 `sim/tools/make_textures.py`（只允许用 `.venv-tools` 的 Pillow/numpy），固定 `random.Random(0xC0FFEE)` / `np.random.default_rng(0xC0FFEE)`：
  - 底噪：逐像素高斯噪声 σ≈6–10（0–255）。
  - 低频污渍：生成 32×32 随机噪声，用 `Image.resize(..., BICUBIC)` 放大到 512，叠加 5–10% 亮度。
  - 裂纹/接缝：`ImageDraw.line` 的随机游走，颜色比底色暗 15–25%。
  - 标线：在 `ground_asphalt_b` 上画 0.15 m 宽白线（512 px/4 m ≈ 128 px/m，线宽 ~19 px）+ 磨损缺口。
  - 柱面：竖向 `ImageDraw.line` 流痕 + 每 1 m 一道水平模板缝。
- **确定性策略**：PNG 一旦生成就提交进仓库，采集时**不运行 Pillow**；`manifest.json` 记录 sha256。生成脚本只用于以后有意改纹理时重跑。这样不受 Pillow/zlib 版本差异影响。
- 不允许在采集时联网或从 Fuel 下载；现有 `~/.gz/fuel` 的 X3 5.4 MB / cone 0.7 MB 只作对照，不进本方案。

#### 1.1.3 地面 UV 覆盖问题（关键，决定方案是否有效）

SDF `<plane>` 的 UV 是 0..1。100×100 m 平面贴一张 512² 图 = 5 px/m，在 160×120 下等于没有纹理；把平面缩小到 24 m 又会在前方露出地面边缘。**不能直接给 `<plane>` 贴图。**

采用 **视觉 mesh + 碰撞仍用 plane**：

```xml
<model name="ground_plane">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
    </collision>
    <visual name="visual">
      <geometry><mesh><uri>meshes/ground_uv.obj</uri><scale>1 1 1</scale></mesh></geometry>
      <material>
        <ambient>1 1 1 1</ambient>
        <diffuse>1 1 1 1</diffuse>
        <pbr><metal>
          <albedo_map>textures/ground_asphalt_a_512.png</albedo_map>
          <roughness>0.92</roughness>
          <metalness>0.0</metalness>
        </metal></pbr>
      </material>
    </visual>
  </link>
</model>
```

`ground_uv.obj` 内容（两块三角面，法线 +Z，UV 0..25，即 100 m / 4 m = 25 次重复）：

```obj
v -50 -50 0
v  50 -50 0
v  50  50 0
v -50  50 0
vn 0 0 1
vt 0 0
vt 25 0
vt 25 25
vt 0 25
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
```

- 4 m 重复一次、512² → **128 px/m**；近处（相机高 2.3 m）足够看出沥青颗粒/标线，远处由 mipmap 收敛，不会闪烁。
- 视觉 mesh 2 个三角形，与 plane 相同；碰撞体仍是 plane，物理/教师真值完全不变。
- 如果担心 OBJ 的 UV>1 在 Ogre2 下不重复（未验证），降级为：512² 图内预先生成 4×4 自身重复图案 + UV 0..6.25（25 m 重复 4 次），或直接 2048² 拉伸覆盖 100 m（texel 4.9 cm，能看出但偏糊）。

#### 1.1.4 柱面材质（保住 L1 颜色语义）

- **灰柱**（`task="base"` 全部、`task="l1"` 的未染色柱）：加 albedo map。

```xml
<material>
  <ambient>0.6 0.6 0.6 1</ambient>
  <diffuse>1 1 1 1</diffuse>
  <pbr><metal>
    <albedo_map>textures/pillar_concrete_256x512.png</albedo_map>
    <roughness>0.88</roughness>
    <metalness>0.0</metalness>
  </metal></pbr>
</material>
```

- **L1 彩色柱（2 根/集）**：**保持纯色，不贴 albedo map**，只把 `specular` 从“等于 diffuse”改成中性低值（0.10–0.15），避免高光把红/黄/蓝洗白。理由：颜色是 L1 的唯一视觉线索（`landmarks.COLOR_RGB`，`landmarks.py:29-34`），纹理/光照最容易在这里造成回退；柱子总面积不大，不贴纹理不影响整体观感。
- `diffuse 1 1 1` + albedo map 是刻意的：gz-sim 的 metal workflow 是否把 texture 与 diffuse 相乘**未验证**；用白色 diffuse 可保证纹理颜色不被二次调暗。彩色柱不贴图，所以两种路径不会互相干扰。
- 如果 smoke 帧发现贴图后柱面整体变亮/变暗，只调 `ambient`/`diffuse`，不动 `landmarks.COLOR_RGB`。

#### 1.1.5 资源路径

gz-sim 用 `gz::common::findFile` 解析 `<albedo_map>` 和 `<mesh><uri>`。生成 world 在 `/tmp/uavrgb_worlds_*`（`collect_parallel.py:491-493`），相对路径不会自动相对 SDF 文件，因此：

- 在 `collect_parallel.py::gz_env()`（`sim/tools/collect_parallel.py:86-95`）和任何直接跑 `collect.py`/`eval_closed_loop.py` 的入口，给子进程加：

```python
env["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
    [os.path.join(SIM_DIR, "worlds"), env.get("GZ_SIM_RESOURCE_PATH", "")]).rstrip(os.pathsep)
```

- world 里写相对路径 `textures/xxx.png`、`meshes/ground_uv.obj`，保持 world 字节与机器路径无关，`world_sha256` 仍可复现。
- 验证方式：gazebo 日志里不能出现 `Unable to find texture` / fallback 到纯色；出一帧后检查地面有高频纹理。
- 备选（不推荐）：`scene_gen` 写绝对路径。简单，但 world 字节随安装路径变化，golden/复现性变差。

#### 1.1.6 llvmpipe 下会失效/极慢的特性

| 特性 | 结论 | 处置 |
| --- | --- | --- |
| Ogre material script | **失效**（gz-sim8 直接不支持） | 不用 |
| `<scene><fog>` | **失效/被忽略**（gz-rendering8 的 `Scene` 没有 fog API，SceneManager 也没有下传） | 不做雾；要“空气感”用背景色/远景墙 |
| `<sky>` 的 time/clouds | **很可能被忽略**：SceneManager 只调用 `SetSkyEnabled`，二进制里没有 time/cloud 的下传 | 只把天空当“固定天幕开/关”；颜色变化仍用 `<background>` |
| roughness / metalness map | 会读取，但每多一张图就多一次采样 | 不做（用常数 roughness/metalness） |
| normal / environment map | 很可能被忽略；即使生效，normal 要切线空间、env 要 cubemap，llvmpipe 上很慢 | 不做 |
| PBR specular workflow | 忽略 | 用 metal |
| 阴影 shadow map | 可用；llvmpipe 是 CPU 光栅，阴影 pass 是最大不确定项 | v1 pilot 已含阴影，成本约 0.98×；v2 仍需用边际法复测 |
| 各向异性过滤 / MSAA | 无法从 SDF 控制；llvmpipe 逐样本，代价高 | 不主动加；靠 mipmap 控制 |
| 大纹理（2048²+） | CPU 内存/采样缓存压力，4–6 实例叠加 | 单张 ≤512²，总量 <5 MB |
| CPU 高斯噪声 | 不是渲染，320×240×3 每帧 <1 ms | 可用 |

#### 1.1.7 纹理不可用时的降级路径（按顺序）

1. **只保留地面 albedo**（地面占画面 40–60%），柱面回到纯色；纹理采样减半。
2. **不用 OBJ**：改回 `<plane>` + 2048²/1024² 拉伸纹理（接受 2–5 cm texel 的模糊），或缩小视觉平面到 40×40 m 并接受地平线边缘。
3. **完全不用 PBR**：纯色 + 每 seed 换调色板（地面深灰/水泥/砂色，柱面冷灰/暖灰/锈色）+ 1.3 的成像增强。仍比现状好，因为至少去掉了固定纯色地标。
4. 如果连纹理采样都导致 >1.35×，停止场景外观工作，转几何族（方案 B）或只做成像增强（几乎零成本）。

### 1.2 光照与大气

#### 1.2.1 v1 参数的问题与 v2 建议值

v1 的 `appearance_params()`（`scene_gen.py:252-271`）已经随机化，但有几个非物理耦合：

- 方位角固定 `[-160°, -20°]`：太阳永远在机体右侧（`direction.y = -cos(el)sin(az)`，sin<0 → 光向 +Y，太阳在 -Y），阴影方向单一。
- 强度、色温、环境光、天空色彼此独立采样；真实场景里“太阳低 → 暖、弱、环境暗、天偏橙”是相关的。
- `specular = 0.10–0.30` 对漫反射为主的场景偏高。

v2 建议：

| 参数 | v1 | v2 | 理由 |
| --- | --- | --- | --- |
| 太阳方位 az | `[-160,-20]` | `[-160,-25] ∪ [25,160]` | 两侧都出现，避免 `|az|<25°` 的正面逆光（柱面变黑，伤害 L1 颜色） |
| 太阳高度 el | `[25,70]` | `[22,72]` | 低角度给长阴影和暖色；高角度给顶光 |
| 强度 scale | `[0.60,0.95]` 独立 | `0.65 + 0.30*h`，h=sin(el) 归一化 | 太阳越高越亮 |
| 暖色 warm | `[-0.04,0.04]` | `0.10*(1-h) + N(0,0.02)`，裁到 `[0,0.12]` | 低太阳偏暖；不再出现“偏蓝的暖太阳” |
| 环境光 ambient | `[0.25,0.45]` | `0.42 - 0.20*h` | 太阳越高，环境光占比越低 |
| 镜面 specular | `[0.10,0.30]` | `[0.08,0.20]` | 漫反射场景，降低非物理高光 |
| 天空/背景色 | RGB 独立 `[0.45,0.72]/[0.55,0.78]/[0.70,0.92]` | 低太阳 `(0.62,0.66,0.72)` → 高太阳 `(0.50,0.66,0.92)` 线性插值 | 地平线偏灰白、天顶偏蓝；只有一个 `<background>` 颜色，不能再做渐变 |
| 纹理变体 | 无 | 每种表面 3–4 个变体，按 seed 选 | 同一套纹理重复所有集会变成新的“固定外观”捷径 |

**太阳色温是否随时间变化**：做，但**不用 `<sky><time>` 驱动**（见 §1.2.3，time 很可能不生效），而是把“太阳高度 h”当作时间的代理：h 低 → 暖/弱/环境暗，h 高 → 中性/亮/环境亮。这样既有物理相关性，又不依赖未验证的天空时间接口；每集的 h 由 seed 采样，等效于“不同时段的飞行”。

太阳方向仍用 v1 的公式（`<direction>` 是光的传播方向）：

```python
direction = [-cos(el)*cos(az), -cos(el)*sin(az), -sin(el)]
```

#### 1.2.2 阴影

- 现状：模板 `cast_shadows=true`（`uav_rgb.sdf:47`），v1 又加了 `<shadows>true</shadows>`；但**没有出帧确认阴影是否真的可见**（v1 worklog 的“待确认”）。
- 成本：v1 的 1×10 / 4×20 pilot 是 36.68 vs 35.73 s、54.05 vs 53.01 s，阴影+`<scene>` 没有可测成本。但这两种 run 都被起飞（约 30 s）主导，5% 的渲染成本看不出来。
- v2 做法：**保留阴影**，但用“边际步耗时”复测（见 §3.1）。如果阴影单独 >1.15×，关掉 `<scene><shadows>` 和 `<light><cast_shadows>`，用下面两条廉价替代：
  - 柱面纹理底部画 10–15% 的竖向变暗（烘焙接触阴影）；
  - 柱脚不另加透明贴片（透明排序在 llvmpipe 上更贵）。
- 判据：出一帧同时含柱子和地面，柱子背光侧地面出现可辨识的暗区；若没有，先怀疑 `<scene>` 没生效或 llvmpipe 阴影路径，不要继续加成本。

#### 1.2.3 天空

- `<sky>` 在 SDF 1.6 scene 里存在，但 gz-sim8 的 SceneManager 只看到 `SetSkyEnabled`；`<time>/<clouds>` 没有被下传的证据。**不要按“时间会驱动天空”设计。**
- 建议：v2 默认**不开 `<sky>`**，继续用 `<background>` 颜色（可控、可按 seed 变、无额外 pass）。做一次“开/关天空”的单帧对照：如果开启后画面明显更像户外、成本 <3%，再作为默认；否则永久关闭。
- 云层不做：云 compositor 在 llvmpipe 上是最贵的一类全屏效果，且会随时间变化，破坏 scene 的确定性。

#### 1.2.4 雾与环境光遮蔽

- 雾：见 §1.1.6，gz-rendering 8 没有实现；不要在 SDF 里写 `<fog>` 然后期待效果。
- SSAO/GI：不做（需 GPU 特性/多 pass，llvmpipe 上不可接受）。
- 替代：
  - 远景墙/背景色充当“大气透视”；
  - 柱脚/墙脚的暗部烘焙进 albedo；
  - 光照随机化本身已经提供了明暗变化，比固定 AO 更有利于泛化。

### 1.3 相机与成像

#### 1.3.1 哪些能在 SDF 做，哪些不能

| 效果 | 位置 | 结论 |
| --- | --- | --- |
| 高斯传感器噪声 | SDF `<camera><noise>` | **可用**：`libgz-sensors8-camera` 有 `ImageGaussianNoiseModel`，二进制明确 “The camera sensor only supports Gaussian noise.” |
| 镜头畸变 (Brown k1/k2/p1/p2) | SDF `<camera><distortion>` | **可用但不建议**：`ImageBrownDistortionModel` 存在，CPU 后处理；会改像素几何，威胁 L1 的 `HALF_FOV_DEG=30 / FOV_MARGIN=2` 可见性契约 |
| 鱼眼/广角镜头 | SDF `<camera><lens>` | 可用，但会改 FOV/任务语义，禁止 |
| 曝光 / 白平衡 / 伽马 | SDF 无字段 | 必须放 `tools/preprocess.py`（训练侧）或采集后处理 |
| 运动模糊 / 滚动快门 | SDF 无字段 | 0.3 m/s + 30 Hz 下可忽略，不做 |
| JPEG/压缩伪影 | 相机输出 R8G8B8，collect 写 PNG | 放 `tools/preprocess.py`（内存里 JPEG 往返） |
| 暗角/渐晕 | 无 | 放 preprocess（径向增益） |

#### 1.3.2 推荐：SDF 只加“轻噪声”，重活放 preprocess

- SDF 噪声（可选，v2 再做）：在 RGB 相机 `<camera>` 内、`<clip>` 之后插：

```xml
<noise>
  <type>gaussian</type>
  <mean>0.0</mean>
  <stddev>0.015</stddev>
</noise>
```

  - 取值理由：stddev 是归一化到 [0,1] 的标准差。320×240 渲染后双线性降到 160×120，独立噪声方差约降为 1/4，即有效 σ≈0.0075（约 2 DN）——在 160×120 上可见但不淹没柱色。
  - 代价：CPU 逐像素，<1 ms/帧；但会让帧级结果不可位复现（当前物理/渲染本来也不是位级可复现）。
  - 实现注意：只能插到 RGB 传感器（`name="camera" type="camera"`），不要碰深度相机；X3 在模板 `END_MARK` 之后，`_apply_appearance` 目前只处理前缀，需要单独的 `_apply_camera_appearance(world, seed)` 用字符串定位（同 `_apply_spawn` 的做法），且只对 `--appearance v2` 生效。

- preprocess 光度增强（训练侧，推荐先做，零渲染成本）：
  - 只对 `split=="train"` 生效；val/test 保持干净，否则指标不可比。
  - **每集固定**参数（真实相机一次飞行的曝光/白平衡是固定的）：`gain∈[0.88,1.12]`、`gamma∈[0.93,1.07]`、白平衡 `(R,B)∈[0.95,1.05]/[0.93,1.07]`、暗角 `[0.85,1.0]`、JPEG 质量 `{70,80,90}`。
  - **每帧**噪声：在 resize 到 160×120 **之后**加，σ∈[0,0.02]（0–5 DN），seed = `(episode_seed, step)`。
  - 顺序：320×240 先做 JPEG 编码/解码（模拟采集压缩）→ 白平衡/增益/伽马 → resize 到 160×120 → 暗角 → 加噪。先 JPEG 后缩放，是因为真实链路先压缩再降采样更常见；如果只想省事，也可以缩放后 JPEG，但块效应会集中在小图上。
  - 实现位置：`tools/preprocess.py::load_frame()`（`preprocess.py:330-356`）增加 `augment` 参数；`build_meta()` 写 `"augment": "photo.v1"`；CLI 加 `--augment none|photo`，默认 `none`，保证现有打包字节不变。
  - 成本：每帧 JPEG 往返约 1–3 ms，160×120 噪声 <0.5 ms；4 万帧约 1–3 min，相对采集 7.4 h 可忽略。

- 不建议畸变：
  - `HALF_FOV_DEG=30`、`FOV_MARGIN_DEG=2` 是 L1 可见性契约；畸变会把边缘柱心推出画幅或改变左右角度。
  - 160×120 上 `|k1|≤0.03` 的位移约 1–2 px，视觉收益接近零，风险却是任务级。
  - 如果以后一定要做，先跑 `landmarks.assign_landmarks` 的 500 seed 可见性自检，再限制 `|k1|≤0.03`，并且 RGB/深度使用同一畸变参数。

- 不建议运动模糊：0.3 m/s、30 Hz 下位移 1 cm/帧，在 160×120 上不足 1 px；做了也看不出来。

### 1.4 场景内容

#### 1.4.1 新增内容一览（全部 visual-only、无碰撞）

| 内容 | 数量/布局 | 三角面/个 | 碰撞 | 改 gid？ | 教师/标签 | 成本 |
| --- | --- | --- | --- | --- | --- | --- |
| 地面标线/油渍 | 烘焙进地面 albedo，不建几何 | 0 | 无 | 否 | 无 | 0 |
| 侧墙（矮墙/围挡） | y=±3.6，x∈[-2,10]，高 1.2 m，厚 0.15 m，2 个 box | 12 | **无** | 否 | 无 | 2 draw call |
| 木箱/货箱 | 4–6 个，0.5–0.8 m，放在 `|y|∈[2.0,2.8]` 或 x>6 | 12 | **无** | 否 | 无 | 4–6 draw call |
| 锥桶 | 4–6 个，x∈[6,9] 或 `|y|>2.2` | 32–64（16 段圆柱/圆锥） | **无** | 否 | 无 | 4–6 draw call |
| 远景墙/厂房剪影 | x≈25，宽 20 m、高 6 m，1 个 box | 12 | 无 | 否 | 无（深度 far=20 看不到） | 1 draw call |
| 天花板 | 不做 | — | — | — | — | — |

- 总增量约 300–450 三角形、15–20 个 draw call；当前场景约 700 三角形、15–20 draw call。llvmpipe 是填充率/调用开销敏感，不是三角形敏感，估计 +2–8%，必须 pilot。
- 所有装饰**禁止 `<collision>`**。原因：教师的 `plan_action`/A*、`clearance`、`audit_dataset` 的 label_recompute 都只知道 `pillars`（`scene_gen.py:150-179`、`audit_dataset.py:693-716`）；一个有碰撞但不在 `pillars` 里的箱子会让教师给出错误标签，而且 `geometry_gid` 不会变，静默污染数据。
- 装饰必须在巡航高度 2.3 m 以下或走廊 `|y|>2.0` 之外；锥桶/箱子不进入 A* 的 0.5 m 余量。
- 远景墙只影响 RGB（`far=50`），深度相机 `far=20` 看不到，不进入深度图。

#### 1.4.2 会不会改 `geometry_gid`（必须写清楚）

- `geometry_gid = sha256(canonical(pillars))`（`split_manifest.py:45-58`）；`split_for_gid` 只用 gid 和 `SPLIT_SALT`。
- 本方案的纹理、光照、天空、相机噪声、无碰撞装饰、地面视觉 mesh **都不在 `pillars` 里，gid 不变，`GEOMETRY_VERSION`/`SPLIT_SALT` 不变**。
- 旧 1000 集（`ds-teacher1000`）仍然能通过 `preprocess.py` 的 `geometry_version` 检查；旧 checkpoint 也能在新外观上直接评测（这正是 A/B 需要的）。
- **但不能把新旧外观混在同一次训练里**。当前 `appearance_version` 只写在 `scene.json`，`collect.py` 没有把它写进 `episode_manifest.json`（`collect.py:1117-1135` 只写了 `geometry_version`），`preprocess.py` 也没有拦截。v2 必须补：
  - `collect.py` 从 scene meta 读 `appearance_version` 写进 episode manifest；
  - `preprocess.py` 像 `task_version` 一样检查：缺字段视为 `legacy`（旧 episode 不带该字段，不能因此拒绝旧数据），但“有字段”与“缺字段”混包、或不同 `appearance_version` 混包，直接 `die`；
  - `build_meta()` 写 `appearance_version`。
- **唯一会废数据的改动**：改 `PILLAR_SIZE/HEIGHT/X_RANGE/Y_RANGE/LANE_SEP/COUNT`、给装饰加碰撞、把装饰放到巡航高度或走廊内。那时必须 bump `GEOMETRY_VERSION`，旧 1000 集全部作废，全量重采 ≥7.4 h（4 实例 135 集/h），多族 15–22 h。本方案不碰这些。

#### 1.4.3 明确不做

- 天花板/室内顶灯：会挡住天空、增加阴影 pass，且当前任务没有室内语义。
- 动态物体（行人/其他无人机）：会改变碰撞/标签，属于任务变更。
- Fuel 模型/HDRI/照片纹理：离线、体积、许可、渲染成本都不划算。
- 碰撞侧墙：虽然能防止飞出走廊，但会改变旧策略的闭环行为语义；要加必须单独实验并 bump 版本。

## 2. 落地顺序与门控

原则：**先零成本、再低风险、最后内容**；每步都 opt-in，`legacy`/`v1` 字节不变；每步都有“停止条件”，不通过就不进入下一步。

### 第 1 步：资产与管线骨架（约 0.5 天，风险低）

**改哪些文件**

- 新增 `sim/tools/make_textures.py`、`sim/worlds/textures/*`、`sim/worlds/meshes/ground_uv.obj`、`manifest.json`。
- `sim/tools/scene_gen.py`：新增 `APPEARANCE_V2="v2"`；`_apply_appearance` 支持 v2；用现有字符串标记定位地面块（`'    <model name="ground_plane">'` 到其后第一个 `'    </model>'`）替换材质/几何；柱子 XML 增加 `appearance` 参数（灰柱贴图、彩色柱保持纯色）；
- `sim/tools/collect_parallel.py`、`collect.py`、`eval_closed_loop.py`：`gz_env` 加 `GZ_SIM_RESOURCE_PATH`；episode manifest 写 `appearance_version`。
- `tools/preprocess.py`：统一 `appearance_version` 拦截（此步只加字段检查，不做增强）。
- `sim/tools/test_spawn.py`：legacy golden 不变；v1 确定性不变；v2 确定性、gid 与 legacy 相同、装饰无 `<collision>`、引用的纹理存在且 sha256 匹配。

**怎么证明**

- `scene_gen.py --self-check --check-seeds 100`：legacy/v1/v2 全部确定性；v2 的 `pillars` 与 legacy 完全一致；v2 world 里出现 `albedo_map`、不出现 `landmark_`。
- 资产：PNG 总量 <5 MB；重新生成一次 `sha256` 与 `manifest.json` 一致。
- **此步不改渲染**，不跑 Gazebo；只做静态检查。

**停止条件**：任何 legacy/v1 字节变化、v2 的 pillars 与 legacy 不一致、装饰里出现 `<collision>`。

### 第 2 步：地面 + 灰柱 albedo（约 1 天，收益最大）

**改哪些文件**：第 1 步的 scene_gen 材质注入；`uav_rgb.sdf` 不动（legacy 字节必须不变）；`test_spawn.py` 增加 v2 材质断言。

**怎么证明（成本 + 视觉 + 颜色三件事）**

1. 渲染成本（必须独占 CPU，等训练结束）：
   - 1 实例 × 10 步 vs 1 实例 × 40 步，用边际斜率 `(wall40-wall10)/30` 比较 legacy 与 v2；4 实例 × 20 步作并发对照。
   - 门限：边际 `wall/step` **≤1.25× legacy**；4 实例 RTF 不低于 0.2；无起飞超时/相机 segfault。
2. 视觉统计（30 帧，320×240 渲染 → 160×120）：
   - 唯一颜色数（RGB 5 bit 量化）从当前约 35 提升到 **≥300**；
   - 平均梯度幅值从约 1.2 提升到 **≥5**；
   - 最大单色占比从约 0.55 降到 **≤0.35**。
   - 这三项是“有没有纹理”的硬指标；达不到说明 UV/贴图没生效，不要继续。
3. L1 颜色：
   - 彩色柱保持纯色；用 `landmarks.assign_landmarks` 的 500 seed 自检 + 在渲染帧上采样目标色像素，确认红/黄/蓝/绿仍可区分（简单 RGB 距离 ≥60/441，色相顺序不变）。
   - 用 `make_counterfactual.py` 的指令对做冻结学生推理，目标色选择翻转率不应显著上升。

**停止/降级**：

- 边际成本 >1.35× → 去掉柱面贴图，只留地面；仍超 → 2048² 单张地面；仍超 → 回到纯色 + 第 3 步。
- 出现 `Unable to find texture` 或唯一颜色数没变 → 先修资源路径/UV，不要调其他参数。
- L1 颜色区分失败 → 彩色柱不仅不贴图，连光照随机化也收紧（强度下限 0.7、暖色上限 0.08）。

### 第 3 步：成像增强（约 0.5 天，零渲染成本）

**改哪些文件**：`tools/preprocess.py` 加 `--augment photo`（默认 none）；可选 `scene_gen` v2 注入 SDF `<noise>`；`test_preprocess` 增加确定性/颜色距离断言。

**怎么证明**

- 渲染成本：≤1.02×（只加 CPU 噪声；preprocess 不计入采集成本）。
- 颜色：抽 100 组随机增强参数，把 4 个 `landmarks.COLOR_RGB` 过一遍变换，最小 RGB 距离 ≥60/441；否则收紧 gain/WB 范围。
- 离线：同一份干净 val，训练时开/关增强，干净 val 的 macro F1 下降 **≤2 pp**。
- 闭环：冻结学生在“干净 legacy”上 SR 下降 **≤5 pp**（增强不应破坏已有能力）。
- 可选的强证据：在评测侧对同一批图像人工加噪/JPEG（只改评测输入，不改训练），若增强组的 SR 下降明显小于对照组，说明增强有效。

**停止条件**：红/黄或蓝/绿混淆、干净 val 掉 >2 pp、闭环 SR 掉 >5 pp。此时把 `--augment photo` 保持默认关闭，只保留 SDF 轻噪声。

### 第 4 步：天空 / 阴影 / 太阳时间相关化（约 0.5–1 天）

**改哪些文件**：`scene_gen.py::appearance_params` 换成 §1.2.1 的相关化采样；`<scene>` 增加可选 `<sky>`；阴影开关做成 v2 参数。

**怎么证明**

- 单帧对照：`shadows on/off`、`sky on/off` 各一帧，看柱脚是否有暗区、天空是否可见、是否出现 `Unable to find skybox material`。
- 成本：默认阴影 + 背景色；如果阴影边际成本 >1.15×，关闭阴影。
- 光照多样性：抽 20 个 seed，统计太阳方位覆盖左右两侧、阴影方向不单一；用同一批位姿渲染，冻结学生的动作改变率应 >10%（说明光照真的进入了观测），但 L1 目标色选择不能翻转。

**停止条件**：阴影不可见且无法用 `<scene>` 修好（说明是 llvmpipe/实现限制）→ 关闭阴影，不做替代几何；`<sky>` 成本 >1.05× 或画面异常 → 用背景色。

### 第 5 步：无碰撞装饰与远景（约 1 天）

**改哪些文件**：`scene_gen.py` 的生成段落（`lines` 中 pillars 之后）加侧墙/箱子/锥桶/远景墙；颜色用纯色 + roughness，不贴图；`test_spawn.py` 断言无碰撞、gid 不变。

**怎么证明**

- `geometry_gid` 与 legacy 完全相同（逐 seed 对比）。
- 5 集 `expert_action` 的 `label_recompute` 全过；深度/RGB 对齐不变；`audit_dataset` 的几何常量项不变。
- 成本 ≤1.3×；遮挡/可见性检查：装饰不进入 `landmarks` 的可见柱对判定（纯几何函数，不受装饰影响，但要确认渲染帧里没有大块遮挡目标柱）。
- 40–60 集配对闭环：冻结学生在“有/无装饰”两版上跑，动作改变率 >20% 说明模型在看装饰；若 ΔSR <5 pp，外观不是瓶颈。

**停止条件**：任何标签/几何变化、成本 >1.3×、装饰遮挡目标柱导致 L1 不可解 → 只保留远景墙，删除近处装饰。

### 第 6 步：决策（不是新改动）

- 第 2–5 步都通过且配对闭环 ΔSR ≥15 pp（n=40）或 ≥10 pp（n=60）→ 扩到 300–500 集重训，再复测。
- ΔSR <5 pp → **停止外观投入**，按上一份提案转几何族（方案 B）的族留出 pilot。
- 5–15 pp → 加 n 到 60–80，或把第 3 步的增强作为独立变量再测。

## 3. 成本与风险

### 3.1 渲染成本怎么测（关键：不能用总墙钟）

现有 pilot 的 `collection_wall_s` 被起飞（约 30 s）主导：1×10 步 36.7 s 里渲染只占几秒。正确做法：

1. 同一 `--hold/--steps` 参数下，测 `N1=10` 与 `N2=40` 两次，边际步耗时：
   `marginal = (wall(N2) - wall(N1)) / (N2 - N1)`。
2. 对 legacy / v1 / v2 各测一遍，比较 `marginal_v2 / marginal_legacy`。
3. 4 实例 × 20 步只作并发/RTF/超时对照，不用来算单步成本。
4. 每次 pilot 必须独占 CPU（等 `/tmp/train_students_1000_b.log` 结束后）；否则训练和渲染互相污染。
5. 记录：`wall/step`、RTF、失败/超时、`gz sim` 日志中的纹理/天空/阴影 warning。

**估算**（未实测，按 llvmpipe 经验）：

| 改动 | 估计边际成本 | 依据 |
| --- | --- | --- |
| 地面 + 灰柱 albedo | +5–15% | 每像素多一次纹理采样；地面占 40–60% 像素；mipmap 控制缓存 |
| 地面 OBJ（2 三角形） | ≈0 | 与 plane 同面数 |
| SDF 高斯噪声 | <1% | CPU 逐像素，320×240 |
| 天空 on | +0–5% | 全屏天幕；无云 |
| 阴影（已有） | v1 实测未检出 | 1×10/4×20 比值 0.98；需边际法确认 |
| 装饰 15–20 draw call | +2–8% | llvmpipe 调用开销 + 深度 pass |
| 合计目标 | **≤1.25×** | 超过就按 §1.1.7 降级 |

### 3.2 数据与重采成本

- 外观 v2 **不改 gid**：旧 1000 集仍合法，但新旧外观不能混训。
- 要训练 v2 学生：200 集约 1.5 h，500 集约 3.7 h（4 实例 135 集/h）；加打包/审计/训练另算。
- 一旦改几何/加碰撞：`GEOMETRY_VERSION` bump，旧 1000 集全部作废，≥7.4 h 重采；多族 15–22 h。
- 旧 checkpoint 可以直接在 v2 场景上评测（这是第 2–5 步 A/B 的基础），不需要重训。

### 3.3 风险表

| 风险 | 影响 | 处置 |
| --- | --- | --- |
| PBR albedo 在 llvmpipe/Ogre2 下贴不出来 | 第 2 步白做 | 先做 1 个材质 smoke；失败按 §1.1.7 降级 |
| 相对路径找不到纹理 | 静默回退纯色 | `GZ_SIM_RESOURCE_PATH` + 日志断言；scene_gen 生成时校验文件 sha256 |
| albedo 与 diffuse 相乘/覆盖行为未验证 | 颜色偏暗/偏亮 | 灰色 diffuse 用白；彩色柱不贴图；出帧后只调 ambient |
| 纹理重复太明显 | 新的固定外观捷径 | 每种表面 3–4 个变体 + 4 m 地面 tile + 每 seed 选择 |
| 阴影在 llvmpipe 上成本高/不可见 | 采集变慢或没有接触感 | 边际法测；>1.15× 关闭，用纹理烘焙暗部 |
| SDF 噪声让帧不可复现 | 无法做逐帧 golden | 只做 scene/world 级 golden；帧级本来就受物理时序影响 |
| preprocess 增强伤颜色 | L1 目标色混淆 | 只对 train、参数范围受限、颜色距离断言 |
| 新旧外观混训 | 指标不可归因 | episode manifest + preprocess 双重 `appearance_version` 拦截 |
| 装饰被误加碰撞 | 教师不知道，标签静默错误 | 生成器自检 `assert "<collision>" not in decor`；label_recompute |
| 训练与 pilot 并发污染成本测量 | 门限误判 | 独占 CPU；用边际斜率；记录训练进程状态 |

## 4. 未查证项（明确没做）

- 没有启动 Gazebo：纹理在 llvmpipe/Ogre2 下的实际观感、UV 重复、阴影可见性、天空开关、成本，全部**未实测**；文中所有百分比都是估计。
- 没有验证 `<pbr><metal><albedo_map>` 与 `diffuse` 的相乘/覆盖关系；没有验证 OBJ 的 UV>1 是否按 wrap 重复；没有验证 `GZ_SIM_RESOURCE_PATH` 对 `<albedo_map>` 的解析（只从 `findFile` 符号和 gz-sim 二进制推断）。
- 没有验证 `<sky>` 的 `<time>/<clouds>` 是否被忽略（只看到 `SetSkyEnabled`，没看到 time/cloud 下传）；没有验证开启 sky 后的成本。
- 没有验证 `<camera><noise>` 的 stddev 单位与降采样后的实际噪声强度；没有验证 `<distortion>` 对 L1 可见性的具体影响（因此建议不用）。
- v1 pilot 的 0.98× 是总墙钟比值，不能证明阴影成本为零；必须用 §3.1 的边际法复测。
- 没有实拍帧，无法证明任何“逼真化”能提升 sim2real；AirSim 10k 只是合成参照。没有硬件相机规格（FOV/噪声/曝光）。
- 没有评估 Fuel 之外的资产许可；本方案完全不使用外部资产。
- 没有改任何代码/资产；文中出现的 `--appearance v2`、`make_textures.py`、`GZ_SIM_RESOURCE_PATH` 改动都是待实施项。
