# 2026-11-14 v3 控制基线与 Wanderland 长航线跟进

## 结论

SAGE 六场景的 `3dgs.vln.v3` teacher / forward 同请求配对对照已完成：两臂各 249 条、请求 key 完全一致、均无仿真异常。teacher 是已知 waypoint 与全局占据图的特权控制上界，forward 是盲飞下界；这不是语言策略实验。Wanderland 缺失图像归档已补齐并校验。另一个 evaluation-ready 场景 `10nvbQnL3fUaY8zb5Orh_t3mk7wnal26i` 有 18 条参考 episode、最长 48.26 m，核心 8 文件现已下载并校验；隔离导入后有 30–43m 路线通过画质筛查。首条 41.8m 路线的半电 SITL smoke 已实际运行并因推力预算耗尽触地；满电独立 profile 的悬停门禁、单集 smoke 与 3 seed 配对均已完成，但只属于特权目标点控制诊断。

## 实验一：显式 STOP v3 配对控制基线

- 时间：2026-11-14（Asia/Shanghai）
- 代码：`d2ee68681c4f94684a0ef86db89f6188fcf2f5e6`
- 协议：`3dgs.vln.v3`，`cf_sitl` / `cf_sitl_v1:1bc8fedf:lib31`，`hold=0.5s`，默认显式 STOP，`seed=0`，每航线一条指令，`videos=0`
- 场景与请求：六个 SAGE-3D 场景，249 请求；规范化请求清单 SHA-256 `0c47d09bd75550c098222551487e070ef05940a9d710038d8e77e5047f4e34e5`；六个 scene JSON 清单 SHA-256 `e86c7fe47cc9303c3613108870b8df6a045edd85935f8b206703c39768b266ae`
- GPU：NVIDIA GeForce RTX 2080 Ti，运行时 UUID `GPU-4910bb04-2d00-ca81-f64a-2031568758ad`
- 输出：`/root/rivermind-data/uav_rgb/exp-vln-v3-m1-20261114/{teacher,forward}`，各有 `vln_eval.json`、`episodes.jsonl` 和逐集目录；日志在同一父目录。

| 策略 | 请求/异常 | SR | OSR | SPL | NE 中位数 | 碰撞率 | STOP |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher（目标点+A*占据图特权上界） | 249 / 0 | 0.8434 | 0.8554 | 0.7003 | 0.358 m | 0.1486 | 0.8434 |
| forward（盲目前进） | 249 / 0 | 0 | 0.2410 | 0 | 3.381 m | 0.9639 | 0 |

逐集请求 key 完全相同。teacher 有 210 成功、37 碰撞、2 步数耗尽；forward 有 240 碰撞、9 步数耗尽。teacher 每集动作切换中位数 11 次（1.907 次/m），forward 为 0（它始终前进，不能据此说平顺）。60 条 teacher 成功请求的 forward 轨迹曾进入目标半径；150 条 teacher 成功而 forward 从未进入目标半径。该差异确认目的地控制的价值，但并未测语言理解。

teacher 的场景成功数为：839992 110/123、840139 27/34、840865 31/40、841012 29/36、841150 13/14、841980 0/2。未见场景仍只有 2 条请求，不能据此报泛化。新版 SPL 的起点修正使其与 v1 的 SPL 不同；SR/碰撞接近历史结果不代表算法退化或提升。`run_vln.make_policy()` 仍使用默认 `clearance=ray`；本实验没有验证 swept 净空或抖动修复。

运行命令（teacher 与 forward 使用完全相同场景、seed 和预算，串行）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=GPU-4910bb04-2d00-ca81-f64a-2031568758ad PYTHONPATH=. \
  /root/rivermind-data/uav_rgb/3dgs-venv/bin/python -m sim.3dgs.run_vln \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_839992/scene.json \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_840139/scene.json \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_840865/scene.json \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_841012/scene.json \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_841150/scene.json \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_841980/scene.json \
  --policy teacher --out /root/rivermind-data/uav_rgb/exp-vln-v3-m1-20261114/teacher --videos 0 --seed 0
```

第二臂将 `--policy teacher` / `teacher` 输出路径换成 `forward`。真实运行由启动 agent 串行执行；teacher shell 会话句柄过期，无法取回退出码，但进程结束、249 条 episodes 与完整 `vln_eval.json` 均存在；forward 退出码 0。

## 数据下载状态

数据盘本轮开始时 243 GB 可用；当前下载与实验产物均写入 `/root/rivermind-data`。

| 数据 | 核验结果 | 状态 |
|---|---|---|
| 旧 Wanderland 场景 1MlUucug 的 `images.tar.gz` | 457,635,844 B；SHA-256 `34dc41fea6537cfa6e40a65260393faf7330079359777065cdf9a57ac272e20b` | 下载完成，tar 可完整遍历 |
| `images_mask.tar.gz` | 72,472 B；SHA-256 `ae8c5a28266ff1c61acd78671e7b8f0f8359b5600b95d7028cfc2ddc956c03c3` | 下载完成，tar 可完整遍历 |
| `fisheye.tar.gz` | 113,581,152 B；SHA-256 `53bc25d4feef2705b638ec275b7c59983d20490636369eeb96b386f81d6b3e13` | 下载完成，tar 可完整遍历 |
| `fisheye_mask.tar.gz` | 61,348 B；SHA-256 `bfa32093984740442052dda1c3c0f715f4347e481c3e17b7260acddeebb8c966` | 下载完成，tar 可完整遍历 |
| 上述四包合计 | 571,330,816 B，包含 563 个普通文件 | 未解包；原场景目录补齐但没有改写 3DGS 场景 |
| 新长路线候选 `10nvb...` | 官方 evaluation-ready metadata 有 18 条 episode，最长 geodesic/reference path 48.2638 m，其中 9 条 ≥30 m；Apache-2.0 | 核心 8 文件完成；大小全匹配，6 个 LFS SHA-256 全匹配 |

新候选关键资产的上游大小：`3dgs.ply` 475,445,426 B（已经完成）；`raw_pcd.ply` 919,533,541 B（已完成）；`scene.usdz` 328,759,062 B；`transforms.json` 627,983 B；`episodes.json` 162,806 B；COLMAP sparse 文件另计。下载/候选清单位于 `/root/rivermind-data/uav_rgb/downloads/wanderland/_dl-20261114/`。HF 官方域名在当前容器直连超时，实际用公开镜像 `hf-mirror.com`，未改系统代理；按官方 tree bytes 与 HF LFS OID 校验，最终完整性标记为 `_dl-20261114/CORE_OK_10nvb`。USDZ 单流过慢，最终用 HTTP Range 分段补齐，拼接后 SHA-256 与 LFS OID 完全一致；旧日志中的 rc=143 是本轮切换下载方式留下的中止记录，不能当成最终失败。

## 长路线实验门禁与下一步

不能直接对新候选运行 `tools/scene_import_wanderland.py`：它把 scene ID、路线、场景目录硬编码为旧场景，会污染已有导入物。须先在隔离目录参数化/驱动 importer：3DGS/点云/USDZ+COLMAP 文件完整性和 PLY bounds → COLMAP 相机与 LiDAR 对齐及 nav frame → 将 48.26 m 路线做坐标变换 → scene 内 LiDAR 占据 `route_check` → 0.5 m 采样深度有效率/锐度 → 查看全路线接触表和转弯视角。探针门槛未校准，仅作筛查；场景长并不代表视角覆盖合格。任何一步失败都保留诊断、排除此路线，不用截图插帧补视觉。

门禁全部通过后先做单条路线、同一动力学的一集 teacher smoke，确认显式 STOP、碰撞检测和末端位置；再做至少 3 seeds 的 paired teacher / forward（同 route/起点/预算）。teacher 为 oracle 控制诊断，不报语言成功；原始 Wanderland 指令不能用其 `goals.position`、`reference_path`、`gt_locations` 构建语言 catalog。之后若要测语言，需在未看测试答案前基于场景资产独立标注地标、冻结 hash，再做正确/错配/空/关系反事实，全部请求保留在分母。


## 更新 2026-11-14 22:40：完整下载、坐标纠错、真实长路线 smoke

### 完整性与隔离场景

核心资产共 1,724,595,584 B（8文件；以逐文件 expected TSV 为准）。完整核验：2,014,593 个 Gaussian，34,056,790 个 LiDAR 点，657 COLMAP 相机记录，18 episode；PLY 声明的数据长度吻合。USDZ ZIP 校验通过，`scene.usda` 可读；3DGS、LiDAR、USDZ 和三个 sparse bin 的六个 SHA-256 均匹配官方 LFS OID。新场景原图/鱼眼 tar 未下载；旧场景四包不代表新场景原图齐备。

隔离目录为 `/root/rivermind-data/uav_rgb/3dgs-scenes/wanderland_10nvb`；原始目录按完整 scene ID 区分。复用 importer 临时副本，仅改 ID、目标目录、REPO 和路线，源码尚未参数化。convert/align 成功；657/657 相机脚下有地面点，up 残差 1.46°、局部地面平面倾角 2.261°。固定 nav z=2m 相对地面 p10/50/90 为 1.394/2.000/2.677m；当前 SITL ToF 仍测水平面，这不是地形跟随验证。

初次临时路线直接拷贝 world 的 x/z，是本轮驱动错误；其 34.5m A*、空洞探针和13.4m子段探针均无效，不能归因数据集、不能纳入成绩。当前精确变换 `nav = R.T @ (world - t)`，本场景等价 `[-world_x, -world_z, 0.6156-world_y]`，飞行 z 再固定2m。正确映射后逐条检查18个起终点；episode 0/1/2 的目标在2m占据层不可用，未强行移开终点。

### 修正后路线与画质筛查

| 原 episode | A* 探针路径 | 帧数 | 低锐度帧 | 最大无效深度 | 结果 |
|---|---:|---:|---:|---:|---|
| 4 | 41.83m | 72 | 0% | 1.39% | PASS；已看接触表 |
| 6 | 45.04m | 77 | 2.60% | 12.39% | FAIL；不跑控制实验 |
| 9 | 43.33m | 74 | 0% | 1.39% | PASS；已看接触表 |
| 10 | 30.55m | 52 | 0% | 0.41% | PASS；未人工复核 |

命令：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. DATA_ROOT=/root/rivermind-data /root/rivermind-data/uav_rgb/3dgs-venv/bin/python tools/scene_probe_quality.py --scene /root/rivermind-data/uav_rgb/3dgs-scenes/wanderland_10nvb/scene.json --out /root/rivermind-data/uav_rgb/tmp/10nvb-import/probe_corrected`。总退出1是因ep6失败，其余三条PASS；阈值仍未校准。ep4/9接触表能连续辨认步道、白凳、灯柱、树和垃圾桶，仍有近处树冠重影/浮点。结果与图在该目录，采样通过不等于全部姿态通过。

### 半电真实烟雾实验失败与因果诊断

实际运行 `run_episode(stop_on_request=True)`，route0=ep4，seed141113、hold0.5、max_steps381、评测radius3m；teacher内部STOP半径仍0.5m。GPU RTX2080Ti、同 `cf_sitl_v1:1bc8fedf:lib31`，没有修改PID或电池参数。产物 `/root/rivermind-data/uav_rgb/exp-wanderland-10nvb-20261114`：`execution_recipe.json`、`smoke_config.json`、完整params、二进制哈希、动作、oracle、trace、247张PNG、质量记录、抽帧图、`smoke_results.json`及退出码1。

247步、33.10m，在step246触地，SR0、OSR0、NE7.992m、SPL0、动作切换36次/1.087每米。不是瞬间掉高：READY后119.4s开始明显下降，123.5s触地。飞行画质最低有效深度98.13%、低锐度1.21%；最后两帧随掉高进入地下/模糊，不能将汇总PASS写成全段适飞。门禁失败后没有启动原计划6集配对。

控制 agent CPU-only 对照（未改文件）：原参数原seed重放247动作，各窗口末XYZ与原记录误差0；同参数原地SAFE悬停也在SITL130.78s触地；仅内存副本SoC0.5→1.0、重放同动作，末z1.993m、最低z1.955m、无油门饱和。原运行READY Vbus3.432V，step237油门饱和1.0，末步Vbus2.998V；高度估计跟随真值，执行器保持ACTING、固件STAB已解锁，ToF仅触地末步无效。主因是默认半电+102g载荷的电池/推力预算耗尽；结论只适用于当前未标定模型。

满电诊断使用独立 profile 与结果目录，CPU 覆盖完整190.5s预算、GPU smoke 和同seed teacher/forward 配对均已完成。不把换电量称为算法提升。当前 `physics_verified=true` 不覆盖高度健康，不能把它单独当飞行有效性证据。


## 更新 2026-11-14 23:15：满电门禁与同 seed 配对完成

### 实验二：SoC=1.0 控制诊断

- 配置：`sim.battery_soc_init=1.0` 的独立参数文件，profile `cf_sitl_v1:06e5f017:lib31`；场景 `wanderland_10nvb` route0=`ep4_long_checked`，评测半径3m，显式 STOP，未接语言模型。完整产物在 `/root/rivermind-data/uav_rgb/exp-wanderland-10nvb-20261114/soc100/`。
- CPU SAFE 悬停门禁：READY 后190.5s，通过；高度1.955–2.001m，末值1.988m，最大绝对垂向速度0.037m/s。
- 单集 smoke（seed141113）：343步、42.016m，NE0.347m，SPL0.896，显式 STOP，碰撞0，质量探针通过；高度1.953–2.018m，最低电压3.390V。
- 同 seed 配对（141114/141115/141116，三组初始 RGB/位姿/yaw/scene/task/profile 一致）：

| 策略 | n | SR/OSR | 碰撞率 | STOP率 | SPL | NE均值 | 路径均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher（目标点+A*特权） | 3 | 1.000/1.000 | 0 | 1.000 | 0.8961 | 0.318m | 41.97m |
| blind-forward（不读目标） | 3 | 0/0 | 1.000 | 0 | 0 | 29.378m | 4.67m |

teacher 动作切换均值62.67次（1.491次/m），blind-forward为0；三集质量探针均通过。这个结果是单路线、满电、特权目标点控制诊断，不能与半电 profile 合并，不能写成语言理解、算法泛化或真机有效性。

- paired integrity 已验证三组初始 RGB/位姿/yaw、scene/task/params/profile 一致。外部 GPU 训练进程存在，因此不使用 wall-clock 推断延迟。
- `execution_recipe.json` 的 paired 状态字段仍是生成时的 `not started`，以 `paired_summary.json`、`paired_results.json` 和退出码为准；后续应修复生成器，避免状态漂移。

### 当前结果边界

这些结果只证明数据、渲染、碰撞、SITL 动力学和显式 STOP 在满电单路线下可以复现；它们没有证明语言 grounding、视觉语言学生、未知场景泛化、真实飞行或公开榜单排名。正式 VLN 评测必须先冻结独立目标 catalog，再按 `predicted_target_control_*` 与真正 E2E success 分开记分。
