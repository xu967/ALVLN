# 场景尺度修复：障碍拉进 20 步可达范围

- 时间：2026-10-29 01:02
- 范围：`sim/tools/scene_gen.py`、`sim/tools/collect_parallel.py`、`sim/README.md`
- 结论：完成。4 实例 × 1 episode × 20 步真跑，4/4 episode 含非 FORWARD 动作，
  每条轨迹都与障碍近距交叉；同 seed 确定性、`gz sdf -p`、进程收尾均通过。

## 做了什么

- `sim/tools/scene_gen.py`：
  - 障碍 x 从单一 `X_RANGE=(2.5, 9.0)` 改为两组：近组 **[1.5, 4.0]** 放 2–3 根、
    远组 **[4.5, 6.0]** 放 0–2 根，总数仍 2–5；20 步 × hold=1.0（约 4–5.5 m，
    随 RTF 波动）能覆盖近组全部、远组大部分。
  - 近组第一根固定为「门柱」：y∈[-0.25, 0.25]。teacher 只在画面中间带
    （约 ±10.9°）且 1.5 m 内才转向，y 偏得多的柱子会被径直飞过；门柱保证
    每条 episode 至少触发一次避障。
  - 出生区净空 `SPAWN_CLEAR_X` 2.0 → **1.0**（柱体最近边 1.5-0.2=1.3 m，
    x<1.0 仍无遮挡）；校验巡航高度 2.1 → **2.3**（柱顶 3.0 m，仍满足）。
  - `.json` meta 增加 `pillar_x_groups` / `gate_y_abs_max_m`，其余 schema 不变。
  - 缝隙契约、重采样上限、模板替换逻辑、RNG 调用顺序均保持确定性。
- `sim/tools/collect_parallel.py`：
  - 默认 `--hold` 0.5 → **1.0**（标称每步约 0.275 m；本次 4 实例实测约
    0.18–0.23 m/步，首步 0.11–0.14 m，20 步仍覆盖近组）；`--steps` 保持 20。
  - 实例分区 `GZ_PARTITION=p<i>` → **`dq<i>`**，避开另一个 agent 的 `io` 前缀。
- `sim/README.md`：更新 4.6.1/4.6.2 的 x 范围、门柱、默认参数、分区名与实测数字。

## 没做什么

- 没改 `sim/tools/collect.py`、`teacher.py`、`save_frame.py`、`sim/worlds/uav_rgb.sdf`。
  - 原因：任务明令禁止（collect.py 另有人并行修改；teacher/SDF 不在允许范围）。
- 没改 teacher 的反应式策略（中间带 ±10.9°、`turn_dist=1.5`、`emergency=0.8`）。
  - 原因：teacher.py 不可改；场景侧只能通过摆放降低其失效概率。
- 没做 sim-time 对齐、episode reset、数据增强、gz-transport 直连。
  - 原因：本次只修场景尺度，这些是既有遗留项。
- 没跑 12 实例实测。
  - 原因：任务只要求 4 实例；README 中 2.633 步/s 是线性外推，已标注未实测。
- 没统计 >4 条 episode 的碰撞率。
  - 原因：验收只跑 4 条；1/4 碰撞只能作为个案，不能当总体比例。
- 没写 `docs/results.md`。
  - 原因：本次是数据质量修复，真实输出直接写在本 worklog；未改允许范围外的文档。
- 没杀另一个 agent 的 `io1` gz/ruby 进程（收尾时唯一残留 `ruby` pid=389859）。
  - 原因：不是本会话拉起，按约定不动别人的进程。

## 待确认

- 「门柱 |y|≤0.25」是新增的场景分布约束，超出原先「y 纯随机」；
  是否作为长期设计保留？
- 近组放 2–3 根，但 20 步内实际进入 1.5 m 触发范围的障碍是 1–2 根
  （转向后航向改变，其余柱子可能偏出视野）。「遇到 2–3 个」是否要求
  实际近距经过 ≥2 根？当前只保证 ≥1 根。
- scene 2 的碰撞/摆振（见下）是否需要后续加场景间距规则或改进 teacher？
  本次未处理。

## 验证到哪一步

### 1. scene_gen 自检（确定性 + 契约）

```bash
$ python3 sim/tools/scene_gen.py --self-check --check-seeds 100
scene_gen self-check: 100/100 seeds OK, pillar count distribution {2: 29, 3: 38, 4: 27, 5: 6}
  seed 0: sha256 a=cb57666d3448039789a6d2ba36ed3913123c6ee4789e679a29d0f8acd70d810e
          sha256 b=cb57666d3448039789a6d2ba36ed3913123c6ee4789e679a29d0f8acd70d810e  match=True pillars=2
scene_gen self-check: PASS（同 seed 字节级一致，障碍 2–5 根，缝隙契约全过）
```

两次独立 CLI 生成同 seed 0 的 world 也逐字节一致：

```bash
$ sha256sum /tmp/uavrgb_det_{a,b}/scene_0.sdf
cb57666d3448039789a6d2ba36ed3913123c6ee4789e679a29d0f8acd70d810e  .../uavrgb_det_a/scene_0.sdf
cb57666d3448039789a6d2ba36ed3913123c6ee4789e679a29d0f8acd70d810e  .../uavrgb_det_b/scene_0.sdf
DETERMINISTIC_OK
```

### 2. 真跑验收（4 实例 × 1 episode × 20 步，hold=1.0）

```bash
$ DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg \
  python3 sim/tools/collect_parallel.py --instances 4 --episodes-per-instance 1 \
  --steps 20 --hold 1.0 --base-seed 0 --out /tmp/uavrgb_accept --keep-worlds
== 采集完成: 4/4 episodes, 80/80 steps
   并行窗口 4 实例 x 1 ep: world_gen=0.2s collect=91.1s end_to_end=91.4s
   吞吐: 0.878 步/s （4 实例端到端）；单实例均值 0.225 步/s
```

4 条 episode 的 `validity` 均为
`camera_verified/physics_verified/episode_executed = true`。

动作序列（F=FORWARD_SLOW，L=YAW_LEFT，R=YAW_RIGHT，S=SAFE_NEUTRAL）、
末位姿、障碍坐标（来自各自 `scene.json`）：

| scene | 动作序列（20 步） | 非 F 步数 | collision 步数 | 末位姿 (x,y,z) | 障碍坐标 (x,y) |
|---|---|---|---|---|---|
| 0 | `FFFFFFFFRFFFFFFFFFFF` | 1 | 0 | (4.396, -1.694, 2.264) | (3.745, 0.171), (2.851, 1.608) |
| 1 | `RFFFFFFFFFFFFFFFFFFF` | 1 | 0 | (4.260, -2.588, 2.299) | (1.640, -0.037), (2.925, 1.406), (5.257, -1.141) |
| 2 | `FLFFFRFFFFLLFLFLFLFL` | 8 | 12 | (1.608, 0.259, 2.306) | (2.026, -0.128), (2.557, 1.446), (4.548, 0.824), (4.758, -0.523) |
| 3 | `RFFFFFFFFFFFFFFFFFFF` | 1 | 0 | (4.190, -2.597, 2.275) | (3.010, -0.065), (1.664, 0.478), (5.756, -1.850) |

- **非全 FORWARD 比例：4/4**（核心指标要求 ≥3/4，通过）。
- 「至少遇到 1 个障碍」用轨迹与障碍位置交叉验证：每条最近距离分别为
  1.150 / 0.825 / 0.567 / 1.285 m，全部进入 teacher 的 1.5 m 触发范围。
- 逐障碍最近距离（m，@step）：
  - scene 0: pillar_0 1.150@13，pillar_1 1.816@8
  - scene 1: pillar_0 0.825@6，pillar_1 2.721@7，pillar_2 1.757@19
  - scene 2: pillar_0 0.567@9，pillar_1 1.424@9，pillar_2 2.909@9，pillar_3 3.190@9
  - scene 3: pillar_0 1.534@10，pillar_1 1.285@5，pillar_2 1.735@19
- scene 2 的异常分析：pillar_0 (2.026,-0.128) 与 pillar_1 (2.557,1.446)
  在 x 上仅隔 0.53 m、分居航线两侧；step 5 时 pillar_1 占左带、pillar_0
  已近到 0.7 m 左右，teacher 转向右后撞上 pillar_0（step 8 起 12 步
  `collision=true`，在 x≈1.6 处左右摆振）。这是 teacher 纯反应式 + 近距
  双柱布局的交互问题，不是 x 范围本身失效；collision 真值已如实落盘。

### 3. `gz sdf -p` 解析生成的 world

```bash
$ for s in 0 1 2 3; do gz sdf -p /tmp/uavrgb_accept/worlds/scene_$s.sdf >/dev/null; echo $?; done
0
0
0
0    # 4 个 world 均 rc=0，stderr 0 行
```

模板障碍段确实被替换：生成的 world 中 `tall pillars` 计数 0、
`scene_gen:` 注释计数 1（原有 `pillar_center` 等模板柱没有残留）。

### 4. 收尾

- `collect_parallel.py` 自己打印：`残留 ruby 进程: 1 [389859]`。
  复核 `/proc/389859/environ` 为 `GZ_PARTITION=io1`，命令行是另一个 agent 的
  `gz sim -s -r .../sim/worlds/uav_rgb.sdf`，**不是本次采集的 dq0–dq3**，未杀。
- 本次 4 个 `dq` 分区的 gz sim / collect 进程组均已退出；没有 `dq` 前缀残留。
- 本次 /tmp 临时数据（`/tmp/uavrgb_accept`、`/tmp/uavrgb_scene_check`、
  `/tmp/uavrgb_det_a|b`、`/tmp/gzsdf_err_*.txt`、运行日志）已在交付前删除。

## 风险与遗留

- **collision 数据**：4 条中 1 条从 step 8 起碰撞并摆振，训练前建议按
  `collision=true` 过滤；本次未改 teacher 或场景间距来消除它。
- **只保证 1 次避障**：门柱保证至少一次转向；20 步内第二根障碍是否能
  进视野取决于转向后的航向与随机 y，不保证「实际遇到 2–3 根」。
- **y 分布改变**：门柱 |y|≤0.25 使场景不再是全 y 均匀随机，可能降低
  侧向障碍的多样性；如需恢复多样性需同时延长 episode 或改 teacher。
- **RTF 依赖**：hold 是 wall-clock，真实前进量随负载变化（本次实测约
  0.19–0.23 m/步，低于之前 0.275 m/步的标定值）；20 步仍能覆盖近组，
  但 12 实例并发时 RTF 可能继续下降，建议后续改 sim-time 闭环。
- **未实测 12 实例**：2.633 步/s 是线性外推。
