# 02｜轻量数据集、程序化场景、语义任务与反捷径评测

## 一、为何不再只依赖 AerialVLN-S

AerialVLN 的公开轻量文件主要是标注，官方闭环需要大型 UE/AirSim 场景包。其标注文件体积小不等于模拟成本低。我们的工程分为：D0 单元数值/文本与动作协议；D1 低成本 RGB 场景功能验证；D2 真实连续物理四旋翼训练；D3 公开空中 VLN 正式对照。不能用 D0/D1 的结果冒充 D3。

| 来源 | 输入与任务 | 优点 | 缺陷 | 正确定位 |
|---|---|---|---|---|
| BabyAI/MiniGrid | 程序化文字指令+离散动作 | 很小，适合证明“语言是否起作用” | 不是空中视角和真实动力学 | 算子/记忆/分句/动作/STOP 早期测试 |
| MiniWorld | 第一视角 RGB、可自建任务 | 轻量，方便调整纹理与房间 | 不是现成无人机 VLN；上游维护状态须锁定 | 单目与语言视觉闭环功能试验 |
| Gazebo Harmonic 自建 | RGB、可控制动力学、实体场景 | 可对接四旋翼物理、控制与离线真值 | 语言标注、场景合法性、教师与评测全须自建 | 主研究物理轨 |
| AerialVLN-S | 标准空中 VLN 指令/轨迹/评测 | 公认基准、方便文献对照 | 独立 UE/AirSim 场景资源、旧环境兼容 | 标准 benchmark，单列协议 |
| LANI/NavDrone | Unity 的地面/低空场景，4 类动作 | 已有历史代码控制器可重用 | 不代表真实飞行，无人机域差距 | 验证旧代码与数据适配，不能做空中结论 |
| OpenFly | 更大型空中数据与仿真工具链 | 扩展长程导航与实景域参考 | 高依赖与大资源投入 | 后续扩展，非 MVP 阻断 |

不在文档中凭空提供“数据集下载已完成”或“镜像大小保证”。需真实下载、许可、manifest、哈希核验。

## 二、当前静态生成器真实能力

`./run.sh scene --seed 42 --difficulty S1` 写入 `artifacts/scenes/scene_S1_42.sdf` 和 `...json`：

- SDF 包含地面、立柱障碍、至少一组彩色语义地标，所有主要实体有明确碰撞或视觉几何。
- S1/S2 使用一个必经直线上的阻挡障碍，再随机增加障碍，A* 按机体膨胀半径寻找可达参考路径。最多 120 次重采样；失败返回 `SCENE_UNSOLVABLE`。
- 数据写入**字节级** SHA256（跨平台不受文本模式 CRLF 翻译影响），固定 Python `random.Random(seed)`；记录场景难度、所有物体、轨迹、指令和实际状态标志。
- 生成的标注中 `oracle_path`、`obstacles` 等是**教师特权信息**。在线学生输入必须来自相机与合法指令，不能偷读这个 JSON。
- **不足**：当前语言由几何路径自动产生，但“pass near” 的可见性与地标实际通过语义未经过动态验证；终点还需要更强的语言消歧、真实相机/动作轨迹与可复核 episode。它仍然是 `static_world_only`，不是可用于论文的 RGB VLN 数据集。

## 三、场景和语言任务必须共同设计

`SceneSpec` 含 `scene_id`, `seed`, `difficulty`, `start_pose`, `landmarks`, `obstacles`, `goal_candidates`, `camera_intrinsics`, `physics_profile`。`EpisodeSpec` 含完整 `instruction`, `reference_route`, `task_id`, `start_pose`, `goal_region`, `allowed_actions`, `timeout`, `generation_version`。`StudentObservation` **只含** `rgb_timestamp`, `rgb_pixels`, `instruction_tokens`, `previous_action`, `memory`；绝对不能含 `goal_region`, `oracle_path`, `depth_gt`, `pose_gt`, `collision_oracle`。

场景必须能实现多步指令：

- S0：单地标/简单 stop（只作模型初跑，不作为主要论文结果）。
- S1：两处可区分地标、至少一次真实转向；相同图像不同指令应导致不同目标动作。
- S2：多个相似地标、遮挡与背景干扰、路径分叉，需要短期历史判断经过了哪个地标。
- S3：随机布局/光照/纹理/机体动力学扰动，独立 held-out 场景，不能仅把原场景旋转当“泛化”。

生成流程：先采样带几何真值的场景→生成候选 waypoint/地标序列→规划器检查路径间隙/动力学可行性→按 waypoint 生成语言片段→验证语言唯一性/遮挡/相似物干扰→渲染 RGB→采集训练样本→进行盲策略测试。出现无解、目标穿透、碰撞体丢失、地标被完全遮挡时，有限重试并保存失败原因。

**不要写死 “到红球直飞 16m 停止”**。至少一半评测场景的直线路径必须被障碍挡住；起终点、障碍密度、颜色、地标相对顺序和纹理应按独立 seed 随机化。

## 四、数据格式（建议 schema，应在真实采集器完成后锁定）

```json
{
  "schema":"navlab.episode.v1",
  "scene_id":"scene_S2_0042",
  "episode_id":"episode_00042",
  "split":"train",
  "seed":42,
  "instruction":"Pass near the blue pillar, then ...",
  "camera":{"width":384,"height":224,"fx":null,"fy":null,"cx":null,"cy":null},
  "observations_ref":"observations/",
  "actions_ref":"actions.jsonl",
  "oracle_ref":"oracle/private_teacher_only.json",
  "validity":{"camera_verified":false,"physics_verified":false,"episode_executed":false},
  "hashes":{}
}
```

保留 `null` 直到真正完成相机标定/执行。采集器应写 `frame_id`, `sim_time_ns`, `camera_time_ns`, `command_seq`, `action_start_time`, `action_done_time`, `rgb_sha256`, `pose_gt`（放入私有 oracle 文件）、`collision`、`stop_reason`、`reset_generation`。训练样本的学术评测属性不能通过手工改 CSV 假造。

## 五、初始数据规模与统计功效

先生成 16 条调试、64 条小样本过拟合、100–300 条真实 episode 做可行性检查，再根据**实际每 episode 墙钟成本**设计 500–2000 条训练数据。测试集应按**场景组**划分，参考目标 >=300 个配对 episode 用于检测数量级为 10 个百分点的 SR 改善；具体样本量必须随基线 SR、同场景相关性、有效轨迹数及配对分歧率重新计算。不要把帧数当作统计独立样本数。

- 每个策略使用同一批 `pair_id={scene,seed,start_pose,instruction,weather,physics_profile}`。
- 同一场景下的多条语言表述视为聚类样本；用场景 cluster bootstrap CI。
- 不要只报告 SR，还要报 collision rate、NE、nDTW/SDTW、OSR、Early STOP、Missed STOP、是否到达可行目标区域。
- 任何方法需与 **RGB 置零、语言打乱、恒定动作、只看语言、只看视觉** 对照；若盲策略在某子集接近模型成绩，拒绝把该子集当作语言理解的证据。
- 规划器计算的可通行性不得作为学生真实在线输入；用 privileged labels 是允许的，但必须列出训练/部署信息差。

## 六、仿真吞吐预算

如果每 episode 60 个策略 step、策略 step 映射 2 秒仿真时间，则 2000 条轨迹约 240000 秒仿真时间，RTF=1 时约 67 小时墙钟（不含重置、渲染与存储开销）。这只是**示例估算**，非本机实测。必须通过 10/50/100 条 episode 的 real-time factor、RGB 帧率、磁盘写入、GPU 占用与故障率实测，再决定并行实例数。不可默认 5090 一定高并行。
