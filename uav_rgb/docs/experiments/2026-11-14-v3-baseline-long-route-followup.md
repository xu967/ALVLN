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


## 更新 2026-11-15 00:40：AerialVLN / OpenFly 下载与自动门禁

- AerialVLN 标注（train 16,386、val_seen 1,818、val_unseen 2,310、test 4,830 episodes）大小、JSON、清单 SHA/LFS 或 Git blob 校验全部通过；审计 JSON：`/root/rivermind-data/uav_rgb/tmp/aerialvln-annotation-audit-20261115.json`。
- OpenFly Annotation 三个 split 共 103,226 条记录，审计无 JSON 错误和数组长度不一致；另按 HF tree manifest 完成 SHA256/Git blob 校验。审计：`/root/rivermind-data/uav_rgb/tmp/openfly-annotation-audit-20261115.json`，完整性：`/root/rivermind-data/uav_rgb/tmp/openfly-annotation-integrity-20261115.json`。
- AerialVLN simulator bundle 共 277 文件、40,541,074,951 B，持续后台下载；自动 watcher `/root/rivermind-data/uav_rgb/downloads/after_download_runner.py` 将等全部文件完整且哈希通过后，完成标注审计并触发同一 SAGE 场景的 1 集 teacher/forward smoke。此 smoke 只看控制诊断，不是 AerialVLN 或语言成绩。
- 本机 HF 登录已验证可访问公开镜像；OpenFly_DataGen gated API 仍返回 403，账号尚未批准访问。数据清单约 115.83 GiB，README 为非商业使用；下载前等待 HF 授权，并保留 AerialVLN 完整空间。
- AirVLN 官方 train/eval 目前不能直接启动：AirSim 场景路径、Habitat 预训练权重、AirVLN checkpoint 和兼容环境未就绪；单卡 smoke 需要先执行官方收集流程并 batch=1，不能拿缺 checkpoint 的 eval 目录模式启动。

## 更新 2026-11-15 01:40：AerialVLN 门禁与 SAGE 单集复核

AerialVLN 的 277 个模拟器文件（40,541,074,951 B）已全部下载并逐文件校验；`train/val_seen/val_unseen/test` 标注也通过大小、JSON 和 SHA/Git blob 校验。OpenFly Annotation 三个公开 split 共 103,226 条记录，通过解析、字段和数组长度审计。

下载门禁触发的首轮 SAGE smoke 没有真正启动，因为自动脚本指向了旧的 lib30 以前 `cf_sitl_proc`。修正为 lib31 后，使用相同场景 `sage3d_841150`、seed `20261115`、limit=1、`cf_sitl_v1:1bc8fedf:lib31`，得到以下仅用于控制诊断的结果：

| 策略 | 动作数 | SR | OSR | SPL | NE | 碰撞率 | STOP率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher（特权目标/规划） | 84 | 1.0 | 1.0 | 0.775948 | 0.321754 m | 0.0 | 1.0 |
| forward（盲目前进） | 12 | 0.0 | 0.0 | 0.0 | 3.075232 m | 1.0 | 0.0 |

teacher 日志显示末端显式 STOP，forward 在第 11 步碰撞。这个配对说明当前渲染、SITL、碰撞和停止协议能够产生可复现结果，同时也说明 forward 并没有语言目标或视觉目标估计能力。两条结果的 `grounder=none`，不能标作 VLN 成绩，也不能和 AirVLN benchmark 比较。完整 JSON 在 `/root/rivermind-data/uav_rgb/auto-after-download-20261115-retry1/{teacher,forward}/vln_eval.json`。

自动门禁脚本已固定使用数据盘的 lib31 proc，后续下载完成不会再因旧二进制在首步前失败。`sim.3dgs.test_dynamics_sitl` 13/13 通过。

OpenFly_DataGen 仍未下载：本机 HF token 和 gated manifest API 可用，但实际数据文件端点继续返回 HTTP 403；后台下载器保持每 10 分钟重试。AirVLN Python 3.8 环境已创建，Gibson backbone、25 个场景和标注路径已就位，CUDA wheel/requirements 仍在数据盘临时目录安装；没有 AirVLN policy checkpoint，因此官方 eval 尚未运行。

## 更新 2026-11-15 02:40：AirVLN 单路线真实采集 smoke

AirVLN 依赖已在数据盘 Python 3.8 环境完成，torch `2.1.2+cu118`、AirSim `1.7.0`、Gibson 权重和 ResNet-50 缓存均通过导入/初始化检查。由于 Unreal 二进制拒绝 root，使用 `ubuntu` 用户和数据盘隔离 runtime；没有修改原始 `vendor/` 权限。

无 `DISPLAY` 时，UE 虽启动并开放 RPC，第一次 `simGetImages` 会超时并触发 SIGSEGV；继承已有 `DISPLAY=:20`、`QT_X11_NO_MITSHM=1`、`SDL_HINT_CUDA_DEVICE=0` 后，RGB `224x224x3` 与深度 `256x256` 探针通过。隔离 server 保留 UE 子进程并 drain stdout，真实 collector 正常完成：场景 `env_5`，单个 trajectory `39KV3A5D2G5VV16NHAJOVU94299S7W` 的 3 条语言指令，两个 190 步 minibatch，LMDB 写入 3 条 episode feature、190 条 RGB、190 条 depth。产物在 `/root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/DATA/img_features/collect/airvln-single-route-20261115-r3/`。

这只是 teacher-forcing 数据采集链 smoke，不是训练后自主导航，也没有 checkpoint、eval 或公开 VLN 分数；OpenFly_DataGen 实际文件仍返回 HTTP 403，后台下载器继续等待授权。

## 更新 2026-11-15 16:57：下载源和并发实测

这次是 CPU 网络/文件验证，不是 VLN 算法实验。当前 `nvidia-smi -L` 能列出 RTX 2080 Ti，但没有使用 GPU；所有请求 `Session.trust_env=False`，不走代理，不记录 token 或签名 URL。

### 来源与等价性

- ModelScope `SimpleNav/OpenFly_env` 递归清单为 627 文件 / 50,256,260,200 B，包含 Linux 场景，无 `.pcd`、`.zip`、`.ply`。从本地已下载的 6 个 HF AirSim ZIP 流式读出主 `.pak`，SHA-256 全部匹配魔搭对应文件。UE 只核对目录、大小、魔搭侧哈希，未证明与 HF 原包内部逐文件相同。
- HF 镜像当前的原始 DataGen 清单为 20 个资产 / 124,369,388,594 B；13 个完整文件合计 9,057,576,681 B。恢复前的断点为 bigcity PCD 1,421,869,056 B、bigcity ZIP 58,720,256 B；完整剩余字节 113,831,222,601 B，磁盘可用 152,686,338,048 B。

### 短窗口测速

方法：使用 `requests` 流式 Range 请求，每路最多 32 MiB；HF 并发对比每路另设 20 秒上限，读取不同区段后关闭响应，不写测试载荷。请求内容均返回 HTTP 206。样本大小和时间有限，不能用不同文件的数值推断整库加速倍数。

| 测量对象 | 字节数 / 时间 | 实测吞吐 |
| --- | --- | --- |
| 魔搭 bigcity `.ucas` | 32 MiB / 2.69 s | 12.49 MB/s |
| 魔搭 smallcity `.ucas` | 32 MiB / 1.87 s | 17.93 MB/s |
| HF 同一 bigcity ZIP，2 路 | 48,234,496 B / 20.582 s | 2.344 MB/s |
| HF 同一 bigcity ZIP，4 路 | 93,323,264 B / 20.730 s | 4.502 MB/s |

HF 本次 4 路短窗口比 2 路约快 92%，据此选择 4 路恢复。魔搭虽更快，缺少本任务部分原始文件，未用不等价内容替换，不新增约 50 GB 重复下载。

### 实际续传验证

启动配置：`config/openfly_download.supervisor.conf`，实际命令 `/opt/conda/bin/python -u tools/download_openfly_datagen.py --workers 4`；监督进程为 `openfly-datagen`。2026-11-15 16:56:22 读取清单，16:56:35 已重新校验 13 个完整文件，随后 4 路继续写入旧目录。

16:57:02–16:57:32 对 `airsim/pcd_map/ue` 所有文件总字节两次采样（包含正式文件与 part，避免改名影响差值）：30.002 s 增加 198,180,864 B，约 6.606 MB/s；累计载荷 10,871,613,161 B，剩余磁盘 152,352,886,784 B。这是实际写盘证据，全量下载尚未完成。

下载器的 HTTP/续传测试 12/12 通过；所有完成文件仍需最终哈希验证才计为 complete。没有解压全量、运行 OpenFly 官方 eval、得出新 VLN 成绩或估计固定完成时间。


## 更新 2026-11-15 19:50：并行缺陷修复与额外实验

时间均为 +0800。代码基线 `dc4fa8820cd0c6994d87c0e32606fa7a5b397a97` 加本轮 diff；GPU 为 RTX 2080 Ti，UUID `GPU-e9d01956-88db-b317-ea53-be3fb6cf7190`。主产物 `/root/rivermind-data/uav_rgb/exp-parallel-debug-20261115/`，CPU 响应另在 `exp-control-response-20261115/`。不使用墙钟推断速度提升；收尾时另有外部 GPU Python 进程，未动它。

### A. 语言解析缺陷与反事实回归

旧解析器取最后一个 `to`，会把 `Go to the chair next to the door` 错误当成去 door；裸子串还把 outdoor 匹配成 door。首轮改为首介词后，交叉审核发现起点 `Start at ...`、副词和多句动作仍可绕过；最终按运动动词边界处理单目标，多个运动子句或未支持关系拒绝。保留三轮结果，避免只展示最后一轮成功。

| 检查 | 基线 | 最终修复 |
|---|---:|---:|
| 定向合成缺陷案例 | 15/32 通过 | 32/32 通过 |
| SAGE 6 场景请求 | 4,429 | 4,429 |
| 正确接受 | 575 | 558 |
| 错误接受 | 183 | 138 |
| 拒绝 | 3,671 | 3,733 |

这不是 held-out 语言成绩：32 案例用于定向回归；SAGE 候选从端点获取，含答案泄漏。误接受少 45 的同时正确接受少 17，不能称总体导航提升。独立 reviewer 的 17 组起点/缺候选/slowly/多句/关系/词边界检查均通过。产物 `language/grounder_counterfactual_and_scene_diagnostic.json` 中 `latest_round=reviewed_motion_clause_fix`；最终日志 `language/regression-final.log`。

复现回归：

```bash
PYTHONDONTWRITEBYTECODE=1 /root/rivermind-data/uav_rgb/3dgs-venv/bin/python -m unittest   sim.3dgs.test_grounder tools.test_audit_vln_grounder sim.3dgs.test_vln -v
```

真实输出 61 tests / OK。词法解析仍只支持有限英文单目标，不支持开放多阶段语言任务，也不是视觉 grounder。

### B. 标注与依赖门禁

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_openfly_annotations.py   --input /root/rivermind-data/uav_rgb/downloads/OpenFly/Annotation   --output /root/rivermind-data/uav_rgb/tmp/openfly-annotation-gate-20261115-debug.json
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tools.test_audit_openfly_annotations -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tools.test_download_openfly_datagen -v
```

全量标注 103,226 条（train 100,226 / seen 1,800 / unseen 1,200），`ok=true`，解析错误、无效记录、并行长度错误均 0。原 CLI 对缺指令/长度不等仍 exit0，现 exit2；三个测试方法含 13 个 CLI 子场景通过，下载器 12 项通过。门禁只证明语言存在与并行字段结构，不证明动作/位姿元素数值、视觉可用或指令语义正确。

首轮全仓 Python 门禁停在 `ModuleNotFoundError: No module named requests`：下载器单测此前用系统 Python，漏掉项目 `.venv-tools` 的依赖声明。本轮补 `requests>=2.32,<3`，直连 PyPI 安装到现有 venv，数据盘 TMPDIR/cache；安装实际 requests 2.34.2。完整重跑 `PYTHONDONTWRITEBYTECODE=1 DATA_ROOT=/root/rivermind-data bash tools/check.sh --python-only` 退出 0。原失败 `python-check.log`、重跑 `python-check-retry1.log/.exitcode` 均保留；日志汇总 72 个 unittest batch 共报告 1,475 项（包含跳过项，另有 assert/selfcheck），不是 1,475 个无跳过通过。GUI/可选依赖等 SKIP 不隐藏；没有重建 C++/CUDA。

### C. 3DGS 长路线视角压力检查

复用 `tools.scene_probe_quality.sample_poses/probe_poses/summarize/contact_sheet`，对 `wanderland_10nvb` 的 route 0/2/3（ep4/ep9/ep10）各测试中心线、偏航 −20°/+20°、横移左/右 0.3m。沿现有 A* 路径以工具默认 0.5m 采样；横移为 `x -= lateral*sin(yaw); y += lateral*cos(yaw)`，高度和相机 pitch 不变。执行入口为项目 3dgs-venv `python -u -`，由 stdin 驱动现有函数，没有新增持久 runner。完整参数/scene hash 在 `view_coverage/config.json`，逐帧和 renderer 元数据在 `view_coverage/results.json`。

| 路线 | A* 路长 | 每变体帧数 | 五变体低锐度帧 | 最大深度无效率 |
|---|---:|---:|---:|---:|
| ep4 | 41.83m | 72 | 0 | 1.6771% |
| ep9 | 43.33m | 74 | 0 | 1.6771% |
| ep10 | 30.55m | 52 | 0 | 0.8945% |

共 990 帧，15/15 通过当前未校准筛查阈值（lapvar≥60 的低质比例≤10%，每帧 depth_valid≥90%），全局最低 lapvar=86.2，所有采样姿态在当前占据代理中无碰撞。这里只检查离散位姿，不是全轨迹扫掠或训练视角覆盖证明。

agent 目视查看 `route0_left_0p3m.png`、`route3_center.png`、`route3_yaw_plus20.png`：步道、长椅和灯柱可辨，树冠局部仍有明显浮点、重影与拉伸，尤其 ep10 第15–24抽样附近。**自动 PASS 不代表视觉质量通过独立人工验收**。不能据此宣布数据问题或渲染问题已解决。

另执行真实 GPU 渲染回归：

```bash
RUN_3DGS_REAL=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/root/rivermind-data/uav_rgb/tmp   /root/rivermind-data/uav_rgb/3dgs-venv/bin/python -m unittest sim.3dgs.test_renderer -v
```

16/16 通过，包含默认跳过的真实 gsplat 两项；日志 `renderer-real.log`，退出码 0。

### D. 54 组固件响应实验：转向/SAFE 不等于瞬间停住

用现有 `CfSitlDynamics.reset/step`，每组 reset 后前进 `{0.5,2,4}s`，执行 `{SAFE_NEUTRAL,YAW_LEFT,YAW_RIGHT}` 一个 `{0.5,1.0}s` 窗口，再 SAFE 恢复 3s；seeds `{0,1,2}`，共 54 组。CPU 真实 `cf_sitl_proc` lib31、20ms trace，满电独立 profile `cf_sitl_v1:06e5f017:lib31`，`calibrated=false`；无占据/渲染/语言输入。命令为 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /root/rivermind-data/uav_rgb/3dgs-venv/bin/python -u -` stdin 调用既有动力学，精确过程/参数/二进制哈希在 `exp-control-response-20261115/response_recipe.json`。

- 54/54 完成；响应前速度 0.240–0.321m/s。SAFE/转向后最大前滑 0.15596m，各臂中位约 0.102m。
- hold0.5 的转向窗口前滑中位约 0.101m、最大 0.133m；hold1 的偏航约 ±32°。
- 高度范围 1.9458–1.9943m，最低电压 3.7761V。详细 `response_results.json` 和 `response_traces.jsonl` 保留全部样本。

这证明当前仿真中的惯性/位置保持有延续运动，不能由 SAFE 名称认定已经刹停。不能把满电2m结果直接用于半电1.2m，更不能改真机 PID；下一步按实际配置测四动作轨迹保护。

### E. 真实 GPU 净空配对与修复前后隔离

安全缺陷的确定性复现：同一预测碰撞状态旧 swept 输出 `wait,wait,forward`，第三步 `_swept_blocked=True` 仍前进。修复后连续20次仍否决、净空恢复放行；DAgger beta0 专家标签保持安全，8步 SAFE 不产生显式 STOP。swept版本 v3→v4，默认 ray 和 hold 不变。

真实配对入口复用 `collect_3dgs.run_episode`，固定 `sage3d_841150`/`sage3d_841980` 各 route0，seeds151101–151103，`hold_s=1.0`（swept只在此窗口有位移系数），步骤预算 `ceil(scene.max_steps*0.5/hold)`=48/62，默认半电 profile `cf_sitl_v1:1bc8fedf:lib31`，`stop_on_collision=True, stop_on_request=True, split_policy=diag-clearance-paired`。每臂用 `OccAstarTeacher(..., clearance=ray/swept)`；先前 run_vln CLI 仍默认 ray，本次没有扩 CLI。轨迹/指标来自相同 `vln.episode_metrics` 和 `route_shortest_path_m`，显式 STOP。

| 臂 | 请求/异常 | 成功 | 碰撞 | 超时 | SPL | NE均值 |
|---|---:|---:|---:|---:|---:|---:|
| ray v2 | 6/0 | 4/6 | 1/6 | 1/6 | 0.5823 | 1.2053m |
| swept v4 | 6/0 | 5/6 | 0/6 | 1/6 | 0.7166 | 0.9625m |
| 修复前 swept v3 | 6/0 | 5/6 | 0/6 | 1/6 | 0.7166 | 0.9625m |

共18集。六组初始位姿/yaw逐值相同；旧 v3 从 `git show dc4fa88:sim/3dgs/planner.py` 内存加载，不回写工作树。**v3/v4六组动作/碰撞/位姿逐行相同，这些请求未触发旧的第三次等待放行，ray与swept差异不是本次修复收益**。证据在 `clearance_pair/results.json`、`clearance_before/results.json`、`verification_summary.json`；实际代码补丁与 hash 在 pair config/source.patch。两场景各一条路线样本很小，只是特权控制诊断，无语言、无泛化结论。

剩余 swept 超时：841980/seed151103，共62步（36转、24前进、2SAFE），NE4.001m。根据实录逐步重建61/61动作一致；step29–49左右转振荡21秒、20次相邻反转，净移0.0748m。`turn_blocked`与`turn_err`在约−44°/−71°来回冲突；静态可用角窗约[−69°,−50°]，一步反转约27°可跳过。只有两次孤立wait_swept，不是持续等待导致；最低电压3.339V、油门最大0.935，无触地/饱和证据。详细 `clearance_pair/timeout_decisions.json`。

限制：当前 collision_model=v1、0.1m高斯占据代理；teacher STOP半径0.5m、场景评测半径1m；未完整保护转弯/SAFE残余运动。841980已经用于诊断，后续不能称盲测调参集。没有扩大预算覆盖失败。

### 下一步执行门槛

继续原计划 §10.11，先固定失败状态分离转向量化与净空冲突；匹配半电1.2m条件测四动作响应；离线比较v1/v2碰撞代理。每项同seed/预算只改一个因素，保留拒绝、碰撞和超时分母。然后落实原E0条件+独立STOP和D独立catalog；当前诊断结果不能替代真正语言导航目标。


## 更新 2026-11-15 21:40：OpenFly env18 真实环境与标注位姿门禁

本次不是SAGE或本项目SITL实验，而是刚下载的 **OpenFly_DataGen env_airsim_18**。基线源仓库 `c872baf2a69c1ba5e3c10db06a3c8ccdebdae04a`；没有改算法源码。GPU RTX2080Ti UUID `GPU-e9d01956-88db-b317-ea53-be3fb6cf7190`，启动时外部进程占约7.6GB显存，未停止或修改。产物统一在 `/root/rivermind-data/uav_rgb/exp-openfly-scene-20261115/`。

### 下载与资源门禁

21:36:52 +0800：官方manifest20文件/124,369,388,594 B；本地载荷123,544,493,857 B（约99.34%），19文件完成，最后 `ue/env_ue_smallcity_windows.zip.part` 仍未完，余约824.9MB；Supervisor PID23826活跃，日志含ChunkedEncodingError自动续传。以当前文件与status为准，不用过往下载速度承诺完成时间。

本轮独立重算实际使用的 `airsim/env_airsim_18.zip`（285,821,005 B）SHA-256：`292c751e50f79f3cb4113a53e7e73d409a7bb0a97bf19a44672a8348ac7863b8`，匹配官方manifest。只解压26项、377,676,858 B，新增结果/运行目录共约384.8MB。数据盘剩约23.6GB，未全量解压。完整核对快照 `download_integrity_snapshot.json`。

### 环境/短动作实验

复用 `/root/rivermind-data/uav_rgb/envs/airvln/bin/python`、AirSim和已有X显示。以ubuntu隔离运行，RPC41458，RGB/DepthPlanar320×240、FOV90°、无LiDAR。不是本项目60°相机或cf_sitl；不能混合评分。

成功启动命令（精确绝对路径见recipe.json）：

```text
runuser -u ubuntu -- env DISPLAY=:20 QT_X11_NO_MITSHM=1 SDL_HINT_CUDA_DEVICE=0 <runtime/env_airsim_18/LinuxNoEditor/AirVLN/Binaries/Linux/AirVLN-Linux-Shipping> -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter=0 --settings <settings.json>
```

初次窗口启动4秒内Vulkan swapchain `VkResult=-13` / SIGSEGV，未到RPC；改用已有离屏参数后RPC3.003s就绪。客户端硬超时100s、整体预算180s，finally仅回收自有进程组。每次reset后依次hover1s、0.3m/s前进1s、−15°/s偏航1s，并记录碰撞/位移；3次reset重复，不声称随机seed。

| 项目 | 结果 |
|---|---|
| 三次reset/动作 | 3/3完整，共9个动作、0碰撞 |
| 悬停窗口位移 | 0.06061 / 0.06083 / 0.06095m |
| 前进窗口位移 | 0.10313 / 0.10311 / 0.10310m |
| 偏航变化 | 约−11.26649°，同时位移约0.12312m |
| 图像 | 7组320×240 RGB与float-depth保存；目视可见楼群/道路 |
| 客户端/退出清理 | rc0；42.902s完成；UE随后按计划TERM，rc−15并非运行失败 |

`results.json`、`launch_status.json`、`recipe.json`、`summary.json`与失败尝试日志均保留。这里验证SimpleFlight/RPC/渲染动作链，官方eval采用位姿设置动作，不能把本轮物理短动作当作官方语言导航或动作重放成绩。

### 参考位姿取图与时序风险

取本地官方 `eval_test.json` 的env18 demo index30，前三个pos/yaw，沿官方映射 `AirSim=(x,-y,-z), yaw=-yaw`。预设位置误差门槛2cm。三种时序都实际运行：

| 设置后读图方式 | 三个位置误差 | 判断 |
|---|---|---|
| 物理不停，等待0.2s | 0.19353 / 1.00147 / 1.30331m | 失败：SimpleFlight继续推进，z漂移 |
| 先pause再set，不推进 | 1015.18 / 1013.89 / 1012.60m | 失败：位姿尚未在仿真tick应用 |
| pause→set→simContinueForTime(0.02)→读pose/RGB/depth | 0.00241 / 0.00518 / 0.00961m | 3/3通过2cm诊断门槛 |

不能把前两次失败归因标注损坏；它们揭示仿真更新时序。最终门禁只证明三个静态参考位姿能在有限误差下应用并取图，没有改官方bridge，也不证明整条轨迹动作对齐或指令正确。图与三种尝试JSON保留，最终 `annotation_pose_results.json`。

### env18语言/动作契约审计

只读检查已下载标注：train9,731条、seen203条、unseen0条；官方演示配置30条。train163/seen3条存在xyz/xyzyaw混用，四维第四项全部等于独立yaw，数值有限，属于schema差异，不能称坏数据。train1,097/seen22条含−1/−2；生成脚本定义为up/down，本地官方eval动作解码只支持0..9。不得静默把未知token改为STOP。

对官方demo30条的428个相邻位姿转移，直接套本地eval运动函数：action[i]精确匹配231个，action[i+1]匹配235个。该诊断未建立完整采样/动作契约，不能推导其余都是错误标注，更不能直接生成教师训练标签。报告 `env18_annotation_contract.json`、来源与样例 `env18_annotation_examples.json`。

当前官方快照的动作尺度为前进3/6/9m、偏航±30°、横移/升降3m，eval以设置位姿并忽略碰撞方式推进，20m成功半径与直线起终点SPL分子；与本机小型无人机四动作SITL不同。在线eval可从场景实时取图，不需先下全库历史PNG；仍需有效策略/processor/statistics及推理环境。训练还需图像与RLDS/TFDS字段转换。未装大模型、未开训练/正式评测。

### 后续交接

1. 等最后Windows归档hash/改名完成更新下载状态；不影响已完成Linux场景实验。
2. 显式冻结action token、状态前后时序、xyz/xyzyaw、STOP和图像对应，再写转换/回归；保留原标注及未知token拒绝。
3. 在独立桥接实现中固定仿真暂停/步进/取图契约，增加实际位姿误差检查；目前20ms只对3样本测试，不能未经验证写成全场景默认。
4. 接入受训语言导航策略后才跑正式SR/OSR/NE/SPL；与项目SITL及本次物理smoke分报。
