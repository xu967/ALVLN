# 场景随机化 + 并行采集流水线（scene_gen + collect_parallel）

- 时间：2026-10-29 00:39
- 范围：新增 `sim/tools/scene_gen.py`、`sim/tools/collect_parallel.py`；增补 `sim/README.md`
- 结论：完成。`scene_gen.py` 12100 个 seed 自检通过；4 实例真跑 48/48 步，
  4 个 world sha256 互异、4 条动作序列/轨迹互异；收尾后无 gz/ruby 残留。

## 做了什么

### 1. `sim/tools/scene_gen.py`（新增）

- 纯 stdlib。`random.Random(seed)` 采样 2–5 根 0.4×0.4×3 m 柱子：
  中心 x∈[2.5, 9]、y∈[-1.9, 1.9]（要求是 y∈[-3, 3]，实际采样先扣掉到 y=±3
  走廊边界的 0.9 m 净空），z=1.5，柱顶 3 m 挡住 `Z_TARGET≈2.1 m` 巡航高度；
  柱体最近边 x≥2.3 m > 出生区 2.0 m。
- 缝隙契约：任意两柱在 x 或 y 方向 `max(gap_x, gap_y) ≥ 0.9 m`；每根柱到
  y=±3 边界也各留 ≥0.9 m。整组不满足则重采样，最多 100 次（实测 1000 个
  seed 平均 2.54 次、最多 14 次，0 失败）。
- 模板替换：只替换 `<!-- tall pillars` 到 `<!-- X3-equivalent quadrotor`
  之间的障碍段落；四旋翼 / RGB / 深度 / contact 插件与三个固定地标原样保留。
- 输出 `<out>.sdf` + 同名 `.json`：scene_id / seed / 障碍清单 / 净空检查 /
  template_sha256 / world_sha256 / sampling_attempts。
- `--self-check [--check-seeds N]`：同 seed 生成两次比 sha256 + 数量/缝隙校验。

### 2. `sim/tools/collect_parallel.py`（新增）

- 纯 stdlib（threading / subprocess / dataclasses / concurrent.futures）。
- 流程：先为所有 (instance, episode) 生成 world（失败则不起任何 sim）；每个
  实例一个 worker，顺序跑自己的 episodes；**每个 episode 新起一个**
  `gz sim -s -r <world>`（collect.py 没有 episode reset）；实例之间并发。
- `scene_id = seed = base_seed + instance*E + episode`。
- 隔离：每实例 `GZ_PARTITION=p<i>` + `DISPLAY=:20` +
  `XDG_RUNTIME_DIR=/tmp/xdg`；先等 `/uav/camera` 和 `/uav/depth` 都真出过一帧，
  再起 `collect.py`（同一 partition、不同 scene_id/seed/split）。
- 进程管理：`subprocess.Popen(start_new_session=True)`（setsid）+
  `os.killpg(PGID, SIGTERM→SIGKILL)`；注册表兜底；只对命令行里带本次 world
  路径的 ruby 兜底，**没有**用 `pkill -f 'gz sim'`，也**没有**全局杀 ruby。
- split：`scene_id % 10 < 8 -> train`，`== 8 -> val`，`== 9 -> test`；
  策略与每个场景的划分写进 `<out>/collection_manifest.json`。
- 汇总：`<out>/<split>/<scene_id>/{actions.jsonl,oracle.jsonl,rgb_*.png,
  episode_manifest.json,scene.json}`；失败 episode 进 `<out>/failed/`，
  脚本退出码非 0；`--keep-worlds` 保留 `<out>/worlds/`，否则用 /tmp 临时目录
  并在结束时删除（`scene.json` + seed 可字节级复现）。
- 打印每实例 sim ready / collect 起止 / collect 的逐步日志，以及总吞吐与
  12 实例线性外推。

### 3. `sim/README.md`

- 文件列表加入两个新脚本。
- 修正“停止”段：明确 `pkill -f 'gz sim'` 会命令行自匹配、可能杀掉当前 shell，
  改为 setsid + `kill -- -PGID`；说明手动调试仍一次一个 server，并行必须用
  `GZ_PARTITION` 隔离。
- 新增 4.6 节：生成契约、并行流程、split、产物布局、进程收尾、实测数字与限制。

## 没做什么

- **没做 sim-time 对齐**：深度与 RGB 仍是 `collect.py` 里两次独立
  `gz topic -e`，只记录 `depth_ts` / `depth_rgb_gap_ns`，未改成同一帧。
- **没做 gz-transport 直连**：所有订阅/发布仍走 `gz topic` CLI 子进程
  （本机固定约 0.8 s 发布延迟）。
- **没做数据增强**。
- 没做 episode reset / world 复位：`--episodes-per-instance>1` 时每个 episode
  重启一次 gz sim，启动/起飞开销计入时长。
- 没实现可选的第二组障碍（x∈[6,12]）。
- 没改 `collect.py` / `teacher.py` / `save_frame.py` / `uav_rgb.sdf` 几何；
  没碰 `ui/`、`src/`、`tools/`、`experiments/`。
- 没实测 12 实例：只按 4 实例端到端吞吐线性外推；且本次演示 hold=1.5，
  不能直接与历史 hold=0.5 的 2.59 步/s 比较。
- 没做模型训练/评估。

## 验证到哪一步

### 1. scene_gen 自检与同 seed 确定性

```bash
$ python3 sim/tools/scene_gen.py --self-check
scene_gen self-check: 100/100 seeds OK, pillar count distribution {2: 49, 3: 33, 4: 16, 5: 2}
  seed 0: sha256 a=806818fa4e6c92f5290c90c7969e07a97f964a2d5096c4206eb985be8425aefc
          sha256 b=806818fa4e6c92f5290c90c7969e07a97f964a2d5096c4206eb985be8425aefc  match=True pillars=3
scene_gen self-check: PASS（同 seed 字节级一致，障碍 2–5 根，缝隙契约全过）

$ time python3 sim/tools/scene_gen.py --self-check --check-seeds 12100
scene_gen self-check: 12100/12100 seeds OK, pillar count distribution {2: 6049, 3: 3708, 4: 1744, 5: 599}
real 0m6.912s
```

同 seed 42 写两个真实文件后比对：

```bash
$ python3 sim/tools/scene_gen.py --seed 42 --out /tmp/uavrgb_check/a/scene_42.sdf
$ python3 sim/tools/scene_gen.py --seed 42 --out /tmp/uavrgb_check/b/scene_42.sdf
$ sha256sum /tmp/uavrgb_check/a/scene_42.sdf /tmp/uavrgb_check/b/scene_42.sdf
626c642a1b496f4358e37813b41c56a72ce23d6b8b47ed5fc529b79932e635b4  .../a/scene_42.sdf
626c642a1b496f4358e37813b41c56a72ce23d6b8b47ed5fc529b79932e635b4  .../b/scene_42.sdf
```

生成 world 能被 Gazebo 真实加载：`gz sim -s -r /tmp/uavrgb_check/a/scene_42.sdf`
→ 进程存活，`/uav/camera` + `/uav/depth` 存在，分区内 25 个 topic。

### 2. 4 实例 × 1 episode × 12 步（真跑）

```bash
$ python3 sim/tools/collect_parallel.py --instances 4 --episodes-per-instance 1 \
    --steps 12 --hold 1.5 --base-seed 12078 --out /tmp/uavrgb_demo --keep-worlds
...
== 采集完成: 4/4 episodes, 48/48 steps
   并行窗口 4 实例 x 1 ep: world_gen=0.2s collect=64.8s end_to_end=65.1s
   吞吐: 0.741 步/s （4 实例端到端）；单实例均值 0.186 步/s
   12 实例线性外推: 2.222 步/s（未实测）
   输出: /tmp/uavrgb_demo
     train: scene_id=[12080, 12081]
     val: scene_id=[12078]
     test: scene_id=[12079]
   manifest: /tmp/uavrgb_demo/collection_manifest.json
```

4 个 world 的 sha256（互不相同）：

| scene_id | split | pillars | world sha256 |
|---|---|---|---|
| 12078 | val | 5 | `6b297b2c94d82e1f8efaf537881c713481e394fa9a01c9c14b7f0920afb7bfaf` |
| 12079 | test | 4 | `0a85195173baf7d3a0a6fc5239e6ab90a28ce930f5aa8d6258dc17691c9966e0` |
| 12080 | train | 4 | `4f9d6adadad2fb88af8bb8b322716ac7ad66167afa0e116653263249224cc4ae` |
| 12081 | train | 3 | `c194d47881fa5e04c6624f79f1066f9fb77ee6ba7753fddd6b2d5547a2380b13` |

用 `scene_gen.py --seed <id>` 重新生成这 4 个 world，sha256 与
`/tmp/uavrgb_demo/worlds/` 下的文件逐一相同（seed + scene.json 可复现，
不依赖 `--keep-worlds`）。

每个 episode 都产出 `actions.jsonl` 12 行、`oracle.jsonl` 12 行、
`rgb_00000.png`–`rgb_00011.png` 12 张；4 个 `episode_manifest.json` 的
`camera_verified/physics_verified/episode_executed` 全为 true。

动作序列（F=FORWARD_SLOW，L=YAW_LEFT，R=YAW_RIGHT）与末位真值位姿：

```
val/12078     FFFFFFLRLRLR      x=2.498 y=0.019 z=2.282
test/12079    FFFFFFRFFFFF      x=3.710 y=-1.404 z=2.251
train/12080   FFFFFLFFFFFF      x=3.599 y=1.948 z=2.243
train/12081   FFFFLFFFFFFF      x=3.474 y=2.290 z=2.242
```

首帧 RGB 的 sha256 也互不相同（`1a3f7462…` / `dc8ea8d7…` / `9b7762bb…` /
`9f8e202d…`）。动作、轨迹、图像三者都不同 → 实例之间没有互相污染。

收尾后：

```bash
$ ps -eo pid,comm --no-headers | awk '$2=="ruby"{print $1}' | wc -l
0
$ ps -eo pid,comm,args --no-headers | grep -E 'gz sim|ruby' | grep -v grep
（无输出）
```

### 3. 多 episode 与失败收尾

- 多 episode：`--instances 1 --episodes-per-instance 2 --steps 2 --hold 0.5`
  → `2/2 episodes`，日志显示 partition `p0` 两次 sim ready（每个 episode 新 sim），
  目录 `val/12078`、`test/12079`，收尾 ruby=0。
- 失败收尾：`--episode-timeout 5` 强制 collect 超时 →
  `failed/scene_12078`、`PYTHON_EXIT=1`，`ps` 无 gz/ruby 残留。
  （第一次通过管道看 `$?` 得到 0 是 `tail` 的返回码；去掉管道后确认 Python
  真实退出码为 1。）
- 这两次测试都没加 `--keep-worlds`，产物树里没有 `worlds/`，说明临时 world
  目录已按预期删除。

### 4. 吞吐与 12 实例外推

- 实测：4 实例 × 1 episode × 12 步，48 步 / 65.1 s（端到端，含 world 生成、
  每集 sim 启动、起飞）= **0.741 步/s**；单实例均值 0.186 步/s。
- 12 实例线性外推：`0.741 × 12/4 = 2.222 步/s`，**只是外推，未实测**。
- 该数字是 hold=1.5 的短 episode；历史 6→1.47、12→2.59 步/s 是 hold=0.5
  的稳态数字，不能直接比较。40 步正式采集会把每集启动/起飞开销摊薄，
  单集步/s 预期会更高（未实测）。

## 风险与遗留

1. **短 horizon + 窄中带**：teacher 只取深度图中带（±10°）；12 步 @ hold=0.5
   实际只前进约 0.8–1.0 m，够不到 x≥2.5 的柱子，很多随机场景会全程
   `FORWARD_SLOW`。本次演示用 hold=1.5（每步约 0.33 m）+ base-seed 12078
   （特意选了 4 个场景都有一根柱子落在 x≈3.6–4.2、|y|≤0.27 的飞行走廊附近）
   才让 4 条动作序列都不同。**这是演示选择，不是场景生成器的普遍保证**；
   正式采集应增加 `--steps`（40 步约 6 m）或 hold。
2. **吞吐不可直接与历史数字比**：见上，hold 和 episode 长度都不同。
3. **无 episode reset**：每个 episode 重启 gz sim，启动/起飞开销固定；
   `--episodes-per-instance` 越大，这部分开销摊得越薄，但总时长仍按集数增长。
4. **GZ_PARTITION 复用 p<i>**：同一实例的多个 episode 复用分区，收尾会等进程组
   清空再起下一个；若未来 gz 把子进程 setsid 到新会话，组杀可能漏掉。当前实测
   `gz sim` 是单个 ruby 进程（PGID=PID），并对 world 路径做了 ruby 兜底。
5. **teacher 在 scene 12078 出现 L/R 交替**（`FFFFFFLRLRLR`）：这是既有
   teacher 滞回行为在窄柱前的表现，不是本次改动引入；标签可能偏噪。未改
   teacher。
6. **固定地标保留**：随机柱可能与 x=3 的 1 m 地标在几何上相交（两者都 static，
   无物理影响；地标在 z≤1.5，巡航高度看柱体）。若后续要求画面更干净，可在
   生成时删除地标；本次按“只替换障碍”保留。
7. **未做 sim-time 对齐**：RGB 与深度仍可能差一帧，`actions.jsonl` 只记录 gap；
   训练前需按 `depth_rgb_gap_ns` 过滤或补对齐。
8. **临时文件**：本次验证产生的 `/tmp/uavrgb_*`（demo/multi/fail/check/smoke/cal
   与日志）已在记录后清理；本文件保留了关键输出，可据此复现。

## 验证产物与清理

- 仓库新增文件只有：
  - `sim/tools/scene_gen.py`
  - `sim/tools/collect_parallel.py`
  - `docs/worklog/2026-10-29-0039-parallel-scene-collection.md`
  - `sim/README.md`（修改）
- 所有 /tmp 临时目录与日志已删除；`/tmp/xdg` 是仓库文档既有的共享运行时目录，
  未删除。
