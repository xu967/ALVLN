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

## 更新 2026-11-15 22:10：六场景解压与 AirVLN 语言反传

### OpenFly Linux 资产

统一路径 `/root/rivermind-data/uav_rgb/openfly-runtime-ubuntu/envs/airsim/`。env16/18/23/26/gz/sh 全部就绪；env18 为既有目录软链接，未复制第二份。六个 ZIP SHA-256 匹配官方快照，125 个普通文件 CRC 通过，六场景启动脚本及 ELF 以 ubuntu UID1000 检查可读可执行。env26/gz/sh 的原 start.sh 缺执行位，只补 `+x`。

总展开 8,366,552,206 B，本轮新增 7,988,875,348 B；2026-11-15 22:03:48 +0800 完成时数据盘剩 15,288,889,344 B，最后 Windows 包仍余 671,802,641 B。解压过程每大文件前保留 12 GiB 加剩余下载空间；无 staging 残留。未解压 UE/Windows，其他五个新场景尚未实跑渲染。`extraction_manifest.json` 与 `asset_inventory.json` 位于上述 runtime 根目录。

### AirVLN 真实语言训练门禁

GPU RTX2080Ti 22528MiB，Py3.8 / torch2.1.2+cu118，FP32；本轮启动前无其他 compute 进程。使用已存在的 3 条语言改写、同一物理路线的 collect LMDB，每条 190 步，冻结预训练视觉特征。未将它们当作 OpenFly 数据。

```bash
cd /root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/AirVLN
PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=/root/rivermind-data/uav_rgb/torch-cache OMP_NUM_THREADS=4 \
  /root/rivermind-data/uav_rgb/envs/airvln/bin/python -u src/vlnce_src/train.py \
  --run_type train --policy_type seq2seq --collect_type TF \
  --name airvln-single-route-20261115-r3 --batchSize 1 --dagger_it 1 \
  --epochs 1 --lr 0.00025 --trainer_gpu_device 0
```

真实退出码 0，1 epoch / 3 batch / 3 次优化更新；逐 batch action CE 为 2.0794103、2.0699828、2.0535865。这三次输入不同，不将它们解释成评测提升。模型总参数35,587,869，可训练2,961,429。

权重 `airvln-runtime-ubuntu/DATA/output/airvln-single-route-20261115-r3/train/checkpoint/20261115-220042-118222/ckpt.LAST.pth` 为166,461,946 B，SHA-256 `c34c89018d49c1c4389397f8a234489d2cbac818db6e504903d16d133a32beb3`；严格重载 missing/unexpected 均空，15个优化器state的step均为3。相同seed初始化对照，语言embedding/LSTM权重最大变化分别7.4816e-4/7.5104e-4。

固定第一帧RGB/depth与零隐藏状态，三条指令的logits最大差异为3.20e-4–4.03e-4，语言置零差异1.37e-3，但三者argmax均为FORWARD。只证明语言连接与反传有效，不证明语义理解、泛化或闭环成功。日志、命令、重载结果在 `exp-parallel-debug-20261115/airvln-language-training/`；22:03检查GPU已释放。未装包、未重新下载视觉权重。

## 更新 2026-11-15 22:23：真实 OpenFly RGB+语言离线 BC

### 数据、模型和监督口径

数据根 `/root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/env18_offline_bc_v1/`，来自 `IPEC-COMMUNITY/OpenFly` 固定revision `a12316d56a4e35a32ad626fb725ed7089937a1c4`。直连hf-mirror，复用已有下载器的fetch/断点/SHA检查；没有开梯子或重新拉整个库。166文件99,404,391 B，其中150张原始853×480 RGB及16份pose.jsonl（后者仅对账，不入模型）。

选择官方train中env18/medium_short、4–20步、含转向、动作仅0..9且唯一末尾STOP的完整路线，按 `sha256(20261115:route_id)` 排序选前16个，排除重复route和完全相同pos/yaw签名；前12条训练116帧，后4条内部验证34帧。两组都来自官方train，不能称官方seen/unseen或独立场景。空间邻近/走廊重叠未排除。

`manifest.json` SHA-256 `5bf8a200fd1768ea8e95346307bcc3d6ab1d603a3fa976d43c44151469661181`；原train.json SHA-256 `1a4c5a5ed1561ea5b3b79b6c25da7648ce4085b7d8489d62022a9bd1244100df`。`download_manifest.json`保留远端path/expected hash/选择原记录；`independent_data_audit.json`记录主代理独立验证。所有原图SHA、PIL解码、图片完整路线+index_list匹配、语言/动作原字段匹配均通过。目视一张原图城市建筑正常，不当作全数据语义质量验收。

复用官方AirVLN的 `VLNCETrainer` / seq2seq，自然语言词表10,038、上限300tokens；本子集694词中17 OOV，无截断。冻结ResNet50在eval/no_grad下提取RGB特征，预处理PIL bilinear→224×224→float32/255，遵循该encoder原实现不另加ImageNet归一化。RNN逐路线reset，输入白名单仅instruction/rgb_features；禁用depth/progress/prev_action分支。没有输入pos、目标、参考路径或未来图像。模型35,588,895参数，其中2,962,455可训练。

保持原十类输出，监督直接为 `image_path/index_list[i] → action[i]`，不重映射成四动作，不复制末帧或STOP。此处仅复现公布的同索引监督；官方压缩8/9与位姿转换还未严格对齐，不能拿离线结果声称动力学或闭环正确。全子集动作计数：0=16、1=10、2=14、3=11、8=10、9=89，4/5/6/7无样本。

### 复现命令与执行

以下从源repo运行；输出目录必须不存在。两次分别取 `EPOCHS=5` / `50`，对应out为 `openfly-env18-rgb-language-bc-v1` / `openfly-env18-rgb-language-bc-50ep-v1`，均在 `exp-parallel-debug-20261115/` 下。

```bash
PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=/root/rivermind-data/uav_rgb/torch-cache OMP_NUM_THREADS=4 \
  /root/rivermind-data/uav_rgb/envs/airvln/bin/python -u tools/train_openfly_subset.py \
  --manifest /root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/env18_offline_bc_v1/manifest.json \
  --airvln-root /root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/AirVLN \
  --out /root/rivermind-data/uav_rgb/exp-parallel-debug-20261115/openfly-env18-rgb-language-bc-v1 \
  --epochs 5 --seed 20261115
```

RTX2080Ti 22528MiB、Py3.8/torch2.1.2+cu118、FP32、Adam默认lr0.00025，每完整episode一次更新，无额外类别加权。5轮60更新在22:18完成exit0；随后固定50轮600更新也exit0，不按验证集挑选checkpoint。两次代码SHA同为 `52bdd1445591945aed9491e112fba3b251f4bcc3bd9166d5651819a29ce95245`，前5轮metrics含混淆矩阵逐字段完全一致；比较记录在50轮`-launch/comparison.json`。

### 实际结果与失败结论

| 固定预算 | train CE | train正确 | train STOP | 内部val CE | 内部val正确 | 内部val STOP |
|---|---:|---:|---:|---:|---:|---:|
| 训练前 | 2.302153 | 47/116 | 0/12 | 未记录 | 未记录 | 未记录 |
| 5轮 | 1.303069 | 70/116 | 0/12 | 1.421098 | 19/34 | 0/4 |
| 50轮 | 0.466842 | 102/116 | 12/12 | 2.131480 | 19/34 | 2/4 |

5轮train/val所有预测均为9，即多数类前进，55.88%验证准确率等于常数多数类基线。50轮能拟合训练，但val CE恶化；val动作1/2/3/8全部0正确、9为17/19，另外2个STOP正确。训练集拟合和保存权重已验证，泛化没有成立，不部署、不报告导航成功率。

50轮固定语言消融（不会重新训练）：

| 语言输入 | train CE /正确率 | 内部val CE /正确率 |
|---|---|---|
| 原指令 | 0.466842 /87.93% | 2.131480 /55.88% |
| 编码器输出置零 | 0.503526 /86.21% | 2.148843 /55.88% |
| 路线间循环换指令 | 0.506535 /83.62% | 2.228877 /55.88% |

语言embedding最大变化5轮0.0072407、50轮0.0416946；输出受语言影响，但不证明语义正确。后续优先扩独立路线/动作覆盖并比较训练加权，不能只增加epoch。

两份 `model.pt` 均164,345,178 B，新trainer strict重载logits最大差均0。SHA-256：5轮 `b50f77595b6faff4ad0a87cfdb190de5ecff1532a4609d16eca687a8446bc5b7`；50轮 `ae7e9f7dae16a00d290b422029be62387fe2ce3aa45602bfc9b0e5dc146a2db8`。`contract.json`绑定数据/词表/实现源码hash、输入/动作与预处理；`metrics.jsonl`和`result.json`保留原始数值，各自`-launch`目录保留recipe、train.log、exit.json。训练进程退出、GPU释放。

### 回归与未完成项

- 新增回归9/9在airvln训练环境通过；`.venv-tools`结果6通过/3因缺torch跳过。覆盖标签/STOP、来源/hash、错路线同stem图像、重复route/几何、因果prev/reset、输入白名单、语言消融hook清理、拒绝覆盖。`tools/check.sh`自动枚举tools/test_*.py，无需再改测试清单。
- 未跑全仓构建/回归、正式seen/unseen、跨场景/三seed、闭环或官方7B。原始压缩token与动力学尚有协议缺口；本轮完整路线均来自单一场景的小子集，不能形成公开排名。
- 22:23 DataGen仍19/20、载荷123,787,763,489/124,369,388,594 B，Windows smallcity余581,625,105 B，Supervisor仍续传；数据盘剩14,769,565,696 B。快照保存在 `exp-parallel-debug-20261115/openfly-training-download-snapshot.json`，Linux训练不依赖这个Windows包。

## 更新 2026-11-15 23:00：OpenFly 实景反馈闭环与视频

### 协议、相机与数值门禁

- 运行环境：RTX2080Ti（UUID `GPU-e9d01956-88db-b317-ea53-be3fb6cf7190`）、airvln Python3.8 / torch2.1.2+cu118、FP32、OMP4；开始前无其他compute任务。使用前述50epoch checkpoint SHA `ae7e9f7dae16a00d290b422029be62387fe2ce3aa45602bfc9b0e5dc146a2db8` 和原16路线manifest，**无新增训练**。
- 最终 evaluator SHA `3aceb266dab37d582a3b44694cbade2907bf81c9a5bc2e19a9c4bae4d2bafdbc`；correct_v2/swap_v2/zero_v2三组均该代码，默认30动作/300m/120s上限。此前correct是首次门禁试跑，仅缺协议错误退出计数；结果保留，不混入配对表。
- env18实际非root、`-RenderOffscreen -NoSound -NoVSync -GraphicsAdapter=0`启动，端口41458、drone_1、front_custom；853×480/FOV90/TargetGamma1.5/相机NED Z=1。启动命令/cwd/设置/日志保存在产物根 `launch.json`、`settings.json`。新Xvfb尝试因display20已有占用退出，未影响实际offscreen渲染；未操作已有显示服务。
- 位姿协议：OpenFly `(x,y,z,yaw)` 转AirSim `(x,-y,-z,-yaw)`；每集reset后暂停，设置请求位姿、推进0.02s仿真、等暂停状态再等0.1s取图。动作使用官方eval的3/6/9m前进、±30°转向、3m升降/平移、0 STOP；**忽略碰撞响应的位姿跳转，不是物理控制或扫掠安全验证**。
- 当前压缩token仍有原样例歧义：首内部验证路线token8，但当前图像到下一保留图像仅3m；故不将闭环失败全归模型，不把诊断结果当正式OpenFly评分。
- 初始相机门禁：首验证route位置误差0.001853m、yaw -59.999992°（请求-60°），真实RGB853×480。原图与实景同位姿MAE10.3296、交换RB后18.1826，目视同建筑同视角；亮度/渲染细节仍有差异，未宣称像素完全同分布。证据 `camera_gate.json`、`camera_comparison.png`。
- 真实checkpoint首验证8帧逐帧/整段对拍：CPU最大logits差6.67572e-6，零语言4.76837e-6；GPU2.86102e-6；动作均相同。零语言 `ablate_instruction=True` 与训练encoder输出置零CPU对拍最大差0。GPU证据 `sequence_parity_gpu.json`；CPU为独立代理实际执行输出，未另存机器文件。

### 固定三组12集结果

产物根：`/root/rivermind-data/uav_rgb/exp-openfly-closed-loop-20261115`。策略只接当前RGB/指令，持续RNN hidden；初始位姿仅reset，终点仅作诊断度量/字幕，未用目标或参考动作指导模型。正确、路线循环换指令、语言编码清零各4集，所有集自行预测STOP；程序、协议、录像错误均0。

| 语言条件 | 集数 | STOP且距目标<20m（仅诊断） | 平均终点距离 | 相比正确指令动作完全相同 |
|---|---:|---:|---:|---:|
| correct | 4 | 3/4 | 13.7342m | — |
| swap | 4 | 3/4 | 15.8070m | 3/4 |
| zero | 4 | 3/4 | 13.7342m | 4/4 |

| 路线序号（固定manifest顺序） | correct终点距离 | correct动作序列 | swap/zero变化 |
|---|---:|---|---|
| 0 | 27.0256m | 9,9,9,9,9,9,2,0 | swap多走一个9后转向停止，距离35.3166m；zero相同 |
| 1 | 3.1059m | 9,9,9,9,9,9,0 | 两组动作相同 |
| 2 | 9.0000m | 9,9,9,1,0 | 两组动作相同 |
| 3 | 15.8054m | 9,9,9,9,9,9,9,0 | 两组动作相同 |

12集共85个真实决策观测，每集RGB hash不同且每个已执行动作后的时间戳严格增加；最大位置误差0.0306003m，最大yaw误差0.00002128°，小于固定0.2m/1°门禁。endpoint碰撞报告未触发，**不能据此称沿途无碰撞**。比较机器记录为 `comparison.json`、各组 `summary.json` 和逐集 `trace.jsonl`。

**结论：** 反馈与停止链路已连通，模型却在零语言时保持4/4相同动作；目前行为主要是长直行后停止，尚无语义任务依赖证据。3/4的20m内停止只是宽松几何诊断，绝不能写成VLN成功率75%或公开算法提升。本轮只含同场景四条已查看路线，无泛化/多seed结论。

### 视频与开关实测

- `index.html` 为本地观看页面，按路线并排展示correct/swap/zero；不需额外网络服务。正确指令视频在 `correct_v2/episode_000..003/decisions.mp4`，对应另外两目录为对照。
- 12段全部实际H264解码：85帧、固定2fps、宽1280、高758/784/862/888（随完整指令字幕增高），总7,814,587 B；逐集帧数与trace一致。完整编码/尺寸/sha验证清单 `video_validation.json`，首末图已抽看。
- 一帧就是一个真实观测，不作平滑插帧或捏造跟随视角；顶部明确标注位姿步进/非实时物理飞行，字幕含原指令、当前动作、评估距离和终止原因。跳动来自离散pose-step，不构成硬件抖动证据。
- 默认录像开启，显式 `--record-video`；`--no-record-video` 真实运行1集（目录 `video_off`，exit0），没有MP4但有首末PNG、预览、trace和结果。
- `sim.3dgs.run_vln --videos N` 原默认0保留；修复编码异常丢掉评测汇总，新增失败仍继续下一回合/落盘回归。未改变collect原始onboard采样：其简拼只含动作前RGB，四格trace重渲包含终态，二者不能混说。

### 可复现命令与验证记录

先按产物 `launch.json` 用既有env18启动并加载同目录 `settings.json`；不要复用忙碌端口或覆盖输出。以下在仓库根运行，另外两组只改language和新输出目录：

```bash
PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=/root/rivermind-data/uav_rgb/torch-cache OMP_NUM_THREADS=4 \
  /root/rivermind-data/uav_rgb/envs/airvln/bin/python -u tools/eval_openfly_subset.py \
  --checkpoint /root/rivermind-data/uav_rgb/exp-parallel-debug-20261115/openfly-env18-rgb-language-bc-50ep-v1/model.pt \
  --manifest /root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/env18_offline_bc_v1/manifest.json \
  --airvln-root /root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/AirVLN \
  --out /root/rivermind-data/uav_rgb/exp-openfly-closed-loop-20261115/correct_v2 \
  --port 41458 --limit 4 --max-steps 30 --record-video --language correct

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
  /root/rivermind-data/uav_rgb/envs/airvln/bin/python -m unittest \
  tools.test_train_openfly_subset tools.test_eval_openfly_subset -v
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv-tools/bin/python -m unittest \
  sim.3dgs.test_vln sim.3dgs.test_video -v
```

- 正式配对各自 `*_v2_launch.json` 记录准确命令、开始/结束时刻、源码hash和returncode，完整日志为对应 `.log`。上面输出已经存在，复现实验必须另取新路径。
- OpenFly回归21/21（2.575s），3DGS回归57/57（5.483s）；测试均禁用GPU，实际闭环另用真实GPU。独立代理也跑过H264/VP8回读与训练录像包装器。
- 初次将四个suite全部放到airvln Python3.8执行失败：既有grounder的 `str | None` 需要较新Python；保留 `regression.log`，拆到原环境后通过，未更改grounder来规避环境差异。
- 未跑全仓构建/主线物理闭环/官方seen-unseen/公开比较；没有调整训练权重、动作大小或判定半径来改善本轮数值。
- 23:00盘余14,522,527,744 B，整个本轮目录约49.35MB；DataGen最后Windows包仍余387,638,545 B，Supervisor仍运行。本轮env18/GPU已停止并清理，证据 `cleanup.json`、`final_snapshot.json`。


## 更新 2026-11-15 23:20 +0800：OpenFly 三种策略、多种子对照预登记

已冻结12条训练/4条内部开发验证路线；seed=20261115/20261116/20261117，两种学习模式均从头训练50epoch、固定最终权重。闭环比较盲前进+固定STOP、单独训练rgb_only、rgb_language；后者追加swap/zero，每组保存视频。配置存数据盘 exp-openfly-comparison-20261115/preregistered.json。当前只是预登记，数值以完成记录为准；不把推理去语言冒充纯视觉训练。


### 固定预算训练与独立公平性审计

本轮产物根为 /root/rivermind-data/uav_rgb/exp-openfly-comparison-20261115。六次训练在2026-11-15 23:24:03–23:27:12 +0800实际完成，均退出0；没有增加训练轮数或按验证结果挑选权重。三个seed下每模式50轮、600次更新，逐配对初始权重hash、50轮样本顺序hash、数据/词表/源码hash完全一致；重载logits差0。训练期间保留其他项目的YOLO进程，UAV作业串行运行；使用RTX2080Ti、airvln Python3.8、torch2.1.2+cu118、FP32、OMP4。

| 模式 | 验证动作准确率（3seed均值） | 有支持类macro recall | STOP precision | STOP recall | CE均值 |
|---|---:|---:|---:|---:|---:|
| RGB+语言，从头训练 | 55.88% | 23.25% | 35.56% | 50.00% | 2.05506 |
| RGB-only，从头训练 | 57.84% | 26.32% | 41.11% | 50.00% | 2.13863 |
| 始终预测训练多数类9，仅离线参照 | 55.88% | 16.67% | 无预测STOP | 0% | 不适用 |

每个seed都评估同4条路线/34帧，不能称12条独立验证路线。macro recall仅平均有支持的0/1/2/3/8/9六类；4/5/6/7没有样本，不能把这些类的recall写成测得0。三seed的逐项值和sample std保存在offline_comparison.json/csv，独立复算证据offline_metrics_independent_audit.json。语言模型动作准确率未超过多数类常数，但STOP行为不同；不能只用总准确率判断导航能力。纯视觉略高也不构成显著性或泛化结论。

两个模式模型总参数同为35,588,895，requires_grad名义参数2,962,455；实际获得梯度为语言2,699,670、纯视觉2,105,610。纯视觉语言embedding变化严格0，训练与验证correct/swap/zero指标完全一致；语言模式embedding真实更新。独立证据training_mode_cpu_gate.json及training_pair_independent_audit.json。

训练源码SHA256：57849b2f7a37e7dd83f8fa6f9931ced4b67c2effacd813255eecdfb30aea044e。评测源码SHA256：a2c559520cd752fed623a519a2a3bb579ce2ad07f9d25ebd607d7425a7a66c58。每次启动的完整参数、环境、起止时刻和返回码保存在launches，权重及contract在training，两种源码在本轮开始后冻结。

### 原始动作时序审计

16份原始pose.jsonl均与来源manifest哈希/大小一致；338个动作中322次相邻转移全部满足当前位姿执行当前原始动作到下一位姿（最大位置误差1.14e-13m）。连续go straight贪心按最多3个分组，组长1/2/3分别为1/8/9；150条公布压缩标注的图像、位置、yaw全部选取组内最后一个原始动作的前置观测。99条宏动作比组首位置提前3m或6m。

完整宏动作从组首到下一组首134/134对齐；从现有保留组末观测到下一组末仅74/134对齐。这是当前线上“观测后执行完整宏动作”与保留帧监督不等价的证据，尚未找到官方压缩生成实现，不能宣称官方意图或官方数据集bug。压缩标签不在本轮分数出现后改写，动作距离不调小。

下一版可保留原宏动作标签、改用逐组验证的组首PNG和起点，但必须建立独立因果观测manifest、保留v1证据并重训所有配对。51张组首图已经本地可用，另外99张在固定revision远端存在，合计67,146,546 B（64.04MiB）；仅查询metadata，本轮未下载或混入训练。逐图官方LFS SHA和原始索引、每个转移证据见protocol_audit.json，SHA256 f112ab8d7b48818d172eddefea8fc29e7c7f8bd5c363f31b3ea1b06976d56275。

当前4条验证路线与训练原始路径最小距离分别约170.66/309.90/177.42/367.39m，未发现这些路径紧邻泄漏，但仍是单场景已查看开发集。PULP-Dronet v3、Tiny-PULP、DroNet本轮均实际CPU单帧前向成功；这些四动作避障模型没有OpenFly的语言和十类STOP契约，不将其分数拼成同协议排名。

### 52回合闭环完成与对照结果（2026-11-15 23:47 +0800）

13个评测臂在23:30:17–23:41:18 +0800全部实际完成：三seed×（独立RGB-only、RGB+正确语言、RGB+交换语言、RGB+零语言）×4路线，再加盲基线4路线。13次命令全部退出0，程序/协议/录像错误均0；碰撞是有效任务失败，不能混为程序失败或从分母删除。

| 策略 | 回合/独立路线 | 平均终点欧氏距离↓ | seed均值标准差 | STOP且<5m | STOP且<10m | STOP且<20m | 端点碰撞终止 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 盲前进+固定STOP | 4/4 | 30.1855m | 不适用 | 0/4 | 0/4 | 1/4 | 1/4 |
| 独立训练纯视觉 | 12/4 | 16.0471m | 3.1783m | 4/12 | 6/12 | 9/12 | 2/12 |
| RGB+正确语言 | 12/4 | 13.7326m | 1.0364m | 5/12 | 6/12 | 9/12 | 0/12 |
| RGB+交换语言 | 12/4 | 18.8374m | 4.6775m | 3/12 | 4/12 | 9/12 | 0/12 |
| RGB+零语言 | 12/4 | 14.4251m | 1.1967m | 3/12 | 6/12 | 9/12 | 0/12 |

20m是事前沿用的宽松几何诊断，不能写作官方SR；5/10m只是附加描述性门槛，不用它们选择模型或声称显著改善。seed标准差为三个seed各自四路线均值的样本标准差，不是置信区间。重复路线只有4条，不是12条独立路线。

| seed | 正确语言NE | 独立纯视觉NE | 交换语言NE | 零语言NE | swap动作相同 | zero动作相同 |
|---|---:|---:|---:|---:|---:|---:|
| 20261115 | 13.7342m | 18.3978m | 15.8070m | 13.7342m | 3/4 | 4/4 |
| 20261116 | 12.6954m | 12.4311m | 24.2245m | 13.7342m | 2/4 | 3/4 |
| 20261117 | 14.7681m | 17.3125m | 16.4806m | 15.8070m | 3/4 | 3/4 |

语言模型平均NE比独立纯视觉低2.3145m，但逐seed改善分别为4.6635m、−0.2643m、2.5444m；各模式20m内停止均9/12。效果并非每seed一致，不能用这四条开发路线宣布可靠优势。交换指令8/12动作序列不变，清零语言10/12不变；语言数值连接存在，任务语义利用仍弱。盲基线第0路线在第8次前进后报告碰撞，未执行后续固定STOP；独立纯视觉在seed15/17第0路线也碰撞终止。碰撞终帧真实误差最高1.3925m，按碰撞优先终止，不能说所有观测都通过0.2m姿态门禁。

同路线的起点/终点评估/预算/相机/源码均配对；不同reset首RGB哈希均不同，起点实际位置最大pair差约0.528mm。故闭环动作差是配对描述，不能排除渲染变化；额外固定观测实验用于隔离语言因素。

### 完全相同图像下的语言诊断

六模型冻结CNN权重逐张量完全一致，复用同34张官方验证PNG特征，在CPU以相同hidden reset和禁用previous-action条件计算所有语言组；没有新训练、没有未来帧/目标输入。GPU原结果与CPU accuracy完全一致、CE差<1e-4。

| 模式/seed | 换指令改变动作帧数 | 清零语言改变动作帧数 | 平均绝对归一化logit差：swap/zero |
|---|---:|---:|---:|
| RGB+语言 20261115 | 1/34 | 1/34 | 0.160451 / 0.153084 |
| RGB+语言 20261116 | 2/34 | 1/34 | 0.339341 / 0.281017 |
| RGB+语言 20261117 | 1/34 | 0/34 | 0.167004 / 0.158294 |
| RGB-only 每个seed | 0/34 | 0/34 | 严格0 / 0 |

变化均在第3条路线的前进/STOP选择，未产生左右转。结论应是“语言会改变部分输出，但动作依赖仍弱”，不能沿用上一轮单seed说所有语言条件绝对无影响，也不能称已学会按语义选择目的地。固定观测证据fixed_observation_language_audit.json及.log。

### 视频、验证与复现

本轮本地观看页面：/root/rivermind-data/uav_rgb/exp-openfly-comparison-20261115/index.html。按路线并排显示盲基线、纯视觉、正确/交换/零语言，seed下拉选择三组模型，默认第一个seed，提供0.5/1倍播放；52段原MP4和逐步trace均有链接。没有开启公网服务。逐回合CSV、完整JSON、离线CSV/JSON也在产物根。

独立代理完整解码52段/390帧，H264 avc1、2fps，总35,748,807B；5,030项trace/视频/配对断言全通过。首末RGB逐像素哈希匹配trace，时间戳严格递增，动作到下一requested pose对齐。根代理目视盲基线碰撞终帧与长指令交换组终帧，字幕完整可读，无伪造平滑插帧。证据video_trace_independent_audit.json、closed_loop_independent_audit.json。

本轮CPU回归：训练11项+评测18项，共29/29通过（root合并运行2.584s）；真实GPU六训练/52闭环另行通过。未重复未改动的3DGS57项或全仓构建；这些不作为本轮新测成绩。eval和train保留输出目录拒绝覆盖、严格hash、错误非零退出。

复现直接调用原入口，示例见docs/OPENFLY_LOCAL_EVAL.md：两种训练模式均--epochs 50 --seed <上述seed>，使用固定manifest与新--out；闭环--policy student --checkpoint <对应新权重> --language correct|swap|zero --limit 4 --max-steps 30 --max-distance 300 --timeout 120 --record-video。rgb_only从checkpoint识别，盲基线--policy blind_forward且不传checkpoint/airvln-root。每次准确argv/env/cwd/起止/rc在launches/train_*.json及eval_*.json，仿真启动在launch_simulator.json/settings.json。

23:43已关闭本任务仿真，端口41458释放；原有其他项目YOLO进程未动。23:45数据盘余10,824,851,456B，系统盘余1,107,648,512B；本轮产物约1.12GB（六checkpoint为主体）。DataGen仍19/20，最后Windows包余149,611,793B，下载继续，不写“已经全部完成”。

### 本轮范围与下一步

已做同模型结构/同数据预算的独立纯视觉和语言模型对照、盲行为基线、语言消融、三seed、完整视频与协议审计。尚未完成官方OpenFly-Agent/AerialVLN公开seen/unseen排名，也没有训练新的原创算法或验证真机/连续物理控制。PULP等CPU可用性检查不算同协议VLN成绩。

优先按计划§10.14另建因果组首观测版本，下载缺失64.04MiB图片并对拍，再做同预算重训；之后扩独立路线/场景及任务分歧。不能凭当前表宣称正式泛化或仅继续加epoch。

### 更新 2026-11-16：优化约束的独立只读复核

本轮没有新训练或闭环分数。三个 agent 并行重读源码与既有产物，运行方式是现有 Python 环境内的只读 JSON/原始 pose 统计；GPU可见但未被本任务使用，未留下临时脚本。后续实施约束见 PLAN §10.15。

- 数据代理重新解析并核验16份原pose SHA：338原动作（297直行、25转向、16 STOP）→150压缩组；150/150保留索引为组末，134/134组首宏转移对齐，最大位置差约2.38e-13m；组末仅74/134对齐。与旧审计结论一致，99张偏移图与60次不匹配转移是不同统计量。99张新组首图仍未在当前subset下载。
- 语言代理核对随机可训练词嵌入、冻结视觉特征、GRU状态和 STOP：三个语言模型现有开发左右转 recall 均0。现有策略不是没有语言接口，而是尚无可靠语言任务辨别证据。CMA无条件前动作和不同空间缓存使直接切换配置不构成单因素注意力消融。
- 控制代理核对现有52条结果：最长wall耗时约20.076s，没有time_budget终止；源码中录像写入计入wall timeout。它尚未造成这批结果的超时，但未来模型时延不能用整回合wall_s代替。2fps视频是决策序列，不是真实运动采样。
- 根代理直接读取 `comparison.json` 断言52行与三项错误计数0；计划路径/阶段齐全，diff空白检查通过。未复跑上一轮29项测试，不把旧测试算本轮验证。

### 更新 2026-11-16 03:11 +0800：并行审核、负例修复与真实回归

产物根 `/root/rivermind-data/uav_rgb/exp-code-audit-20261116`；本轮从源提交 `8e04728` 开始。`scope.json` 事前限定只修可复现问题、单seed两模式重训、受控真实闭环、固定观测/重复reset；不改数据时序或网络结构。环境：RTX2080Ti UUID `GPU-e9d01956-88db-b317-ea53-be3fb6cf7190`，AirVLN Python3.8/torch2.1.2，项目工具Python3.12；原生CUDA架构sm75。三agents按训练/审计、评测/录像、3DGS/SITL划分所有权并交叉审核；根负责C++/CUDA全套和GPU/UE。

#### 实际测试与失败记录

| 检查 | 本轮真实结果 | 证据 |
|---|---|---|
| C++/CUDA重新配置、增量构建 | 当前src构建0退出，89个build步骤、33.789s | `native/0.log`、`1.log`、`results.json` |
| 初次44项CTest | 43通过、1个CPU占用比例断言失败；隔离重跑仍失败 | `native/2.log`、`blocking_isolated.log` |
| 修正测试口径后完整CTest | 44/44，通过47.13s | `native/ctest_final.log/json/xml` |
| OpenFly train/audit/eval/download最终集成 | 58/58，通过10.801s，无skip；分别13+5+28+12 | `python_final_verified.log` |
| 3DGS/SITL最终11模块 | 306/306，通过104.994s，无skip；包含真实SITL进程测试 | `scene_control_audit/final_atomic_cpu_tests.log` |
| 官方Annotation兼容 | 3文件103,226记录通过新字段校验；原−1/−2 token、xyz+yaw仍接受 | `training_audit/real_annotation_audit.json` |
| 当前训练manifest | 16路线/150图来源与hash通过，未改manifest | `training_audit/real_manifest_gate.json` |
| 数据下载 | 下载器00:12:25日志全20资产verified；当前20文件合计124,369,388,594B，0个.part | 原下载器status/log；本轮只重新核文件数/大小，未重新hash全124GB |

408项是44+58+306三套最终用例总数，不再把内部定向重复测试相加。测试名称、确切命令、环境和各阶段红绿日志在各audit子目录。训练/审计新增负例先在原版复现19个subcase失败，评测原版27项套件有18个subcase失败+2error；控制非有限负例先42个失败，原子更新再2fail+3error。这些失败是缺陷证据，未删日志。

本轮修复集中于：Annotation审计输出和.tmp不能覆盖实际输入（包括目录内链接指向的外部文件）；非法元素/非有限pos/yaw拒绝；物理路线按实际xyz/yaw数值去重，整数/小数或冗余第四维不能绕过；策略logits/hidden/反馈有限性、首时间戳和预算校验；推理/编码超预算不派发额外移动；RPC/录像初始化失败仍记录结果；SITL完整解析遥测后才提交状态和时基，无效位姿不触发STOP或护栏转向。

两个测试口径修正必须明说：

- `test_cuda_blocking_sync` 原来要求全进程CPU占用 `<0.35×off`；隔离运行off0.9857核、cli0.3955核、env1 0.3869核，四训练均rc0且权重逐字节一致。短进程的CUDA初始化/I/O不是调度功能本身。最终增加独立子进程直接调用既有helper并读CUDA flags（实际12，调度mask为BlockingSync=4），保留四个真实CLI/env训练、权重一致和非法env拒绝；CPU比例仅报告，不换更宽阈值。没有在每个exec子进程读取flags，不声称那一层直接测过；生产CUDA代码未改。
- 最终Python集成首次57/58：下载中断后Python3.8保存1,048,608B，测试硬编码要求1,048,576B；Python3.12原测试通过。不同requests/urllib3对尾块交付不同。测试改为记录真实落盘offset和前缀内容，严格断言第二次Range等于该offset，再核完整内容/SHA/无.part。两个解释器各12/12通过，最终58/58。生产下载器未改。证据 `download_retry`、`python_final.log` 与最终日志。

#### 实际GPU重训与权重对拍

直接调用既有 `tools/train_openfly_subset.py --epochs 50 --seed 20261115 --training-mode rgb_language|rgb_only`，原v1 manifest与模型配置不变，分别600更新。02:09:18–02:10:12两运行退出0，耗时29.935/24.323s；每个模式的499个state张量与上一轮同seed权重逐项完全相同，权重状态SHA分别 `3997fd2f282ca3c048fa5c1b57ddf239defa65147d0ed6cd273d80a491486f81`、`3841af44bbc390950ff910e3d81059ddbea097b61ed49378bc78cfca485e8ec5`。

这证明新增输入校验没有改变合法数据的训练结果，**不是模型性能提升**。训练源码SHA `dcb37806e477adf5934502819e7cd99d9a712d1df45240c429f6e98550185d28`，评测源码SHA `63797a3c8745370e9d92be9d9b86467f55030dbbff6fb610a0e1ccde42eba8a4`；旧checkpoint不绕过源码hash，真实闭环使用本轮重训产物。命令/env/cwd/起止/rc在 `launches/train_*.json`，结果在 `training/`，对拍在 `training_parity.json`。

#### 21个真实回合与20视频：有1次协议失败

固定原相机/起点/30步/300m/120s/2fps，5臂×4路线，加正确语言关闭录像1路线。通过程序运行不等于任务成功；下面所有请求均保留：

| 臂 | 回合 | 终止分布 | 命令退出码 |
|---|---:|---|---:|
| blind_forward | 4 | 3 fixed_stop，1 collision_reported | 0 |
| rgb_only_correct | 4 | 3 predicted_stop，1 collision_reported | 0 |
| rgb_language_correct | 4 | 3 predicted_stop，1 pose_application_error | **1** |
| rgb_language_swap | 4 | 4 predicted_stop | 0 |
| rgb_language_zero | 4 | 4 predicted_stop | 0 |
| rgb_language_no_video | 1 | predicted_stop | 0 |

正确语言episode_000第5帧实际位置仍为上一步，requested向前9m，误差9.00003053m；图像timestamp也未增。门禁在策略调用前终止，不把此帧继续喂模型。`errors=0`只表示未抛Python异常，`protocol_errors=1`和退出码1才是完整状态。不能把这个回合当有效NE样本做模型均值；4次advance返回只说明发送命令，commanded_distance36m不等于实际反馈路程（约27m）。失败未用重跑覆盖。

20MP4全部独立OpenCV完整解码、共149帧、2fps；全部与逐集trace/result帧数一致。无录像回合8帧trace，合计157；21×首末PNG解码RGB哈希均精确匹配trace，off无MP4且首末图/preview完整。六个launch退出码与summary逐一一致。`video_audit.json`明确完整性通过、闭环并非全通过。根目视失败终帧，字幕`predicted=NO ACTION`、`end=pose_application_error`及完整指令清晰。新的本地 `index.html` 标红失败，链接全部21回合，不展示混合成功均值。

#### 同步、渲染与历史的进一步诊断

- 4起点×5reset共20次，最大两次起点差0.528336mm；各路线平均成对像素绝对差约1.221/3.319/2.367/1.112（0–255尺度），log-prob最大差0.0232/0.1349/0.7815/0.0748。所有首动作correct/swap/zero均为9，只证明这20个起始观测的动作稳定，不代表整条路线利用语言。
- 4条验证序列共34帧，逐帧与整段argmax完全一致，最大绝对logit差1.049e-5；使用现有`atol=rtol=1e-5`组合容差通过，不能误写绝对误差都≤1e-5。相同原图下每帧清空hidden仅改变1/34动作；这是seed20261115的状态依赖诊断，未证明GRU无用或新记忆有效。
- 冻结原失败四动作前缀，独立10次重放2次协议失败。trial1：physics已到新位置，render和camera仍落后9m，即使timestamp已增长；trial7：render/physics位置更新但timestamp不增长。逐RPC monotonic时间、pause、image camera位置、physics位置/速度/时间都在 `feedback_diagnostic/report.json` / `rpc_calls.json`。因此仅等pause、只看位置或只看新timestamp均不足。
- 独立试验固定`simContinueForFrames(2)`、单次setPose无重发，10/10前缀出现timestamp不增，不合入生产代码；它也改变推进语义，不能拿来与原20ms分数混报。证据 `frame_step_candidate/report.json`。
- 再做10次单次请求后的有界被动观测（最长2s，无额外tick/位姿命令），这批没有遇到初始无效观测，故不能证明等待能修好故障；同样未合入。证据 `passive_wait_diagnostic/report.json`。此次不猜UE内部根因，不删除拒绝门禁。

#### 真实SITL分支与超时故障

原seed151103故障前缀step28/29各SAFE/LEFT/RIGHT，共6种条件，每次重新reset并重放完整动作前缀。最终版本1710个历史采样位置/速度/四元数/时间与原记录误差0，60个分支响应与修补原子校验前逐值一致。step28 SAFE仍位移0.158650m、末速0.061530m/s；step29 SAFE位移0.016271m、末速0.040154m/s，左右转约±31.6°。均只有1s局部分支且旧v1碰撞代理无碰，不是长航线安全/停稳结论。

对本次自建SITL proc注入SIGSTOP后，0.02s读超时触发回收，实际约2.021s结束（含回收等待），进程rc−9；零新增trace、状态不变，重试拒绝，不强制前进。既有pose-only到点STOP契约保持，不新增未定义速度要求。`scene_control_audit/audit.json`与commands/trace保存完整证据。

#### 范围、失败和清理

未实现组首因果数据版本、未下载99图、未改变模型/超参数、未修复既有控制振荡/真机PID、未做官方seen/unseen或新场景排名。新同步候选未通过，不宣称适配器稳定；数据与执行两项P0是后续优先级。没有新训练入口或长期队列。

两次诊断命令因Python字符串/JS封装语法错误在执行前失败，修正后仅正式执行一次；无数据或源码误写。首次ffprobe不可用的失败保留，使用已有OpenCV解码而未安装新依赖。03:02仅停止本轮UE组686099/686100，端口41458释放；已有桌面及后出现的llama-server未动。当前系统盘约720MiB、数据盘约8.9GiB，未解压大型UE。新产物/权重/视频全部留数据盘，不进入Git。


## 更新 2026-11-16：因果监督与Seq2Seq/CMA固定预算对照

时间：2026-11-16 +0800。本轮源基点4235d7e，未提交diff只改既有OpenFly入口/测试及文档；实际GPU为RTX2080Ti 22,528MiB（UUID e9d01956-88db-b317-ea53-be3fb6cf7190），AirVLN环境Py3.8/torch2.1.2/cu118。原始证据根目录 R=/root/rivermind-data/uav_rgb/exp-openfly-optimization-20261116。既有llama-server占约15.5GiB，不停止他人服务。

### 数据、训练与公平性

- 99张缺失组首PNG共67,146,546B已按固定revision从hf-mirror直连补齐。ModelScope探测返回500，不据此声称平台没有该数据；HF主站当时连接失败，镜像返回真实PNG。未用代理、未新增认证或全库下载。150图总100,125,636B全部哈希/解码/853×480 RGB检查通过。
- 新manifest：downloads/OpenFly/subsets/env18_causal_bc_v2/manifest.json，SHA256 69ec0c13319e4eb42cbe8e457768e0152c5cebf968007e59771ef3ed9c151482；源revision a12316d56a4e35a32ad626fb725ed7089937a1c4。相对v1恰好99张图/index变，路线、12/4划分、指令和原动作不变。150组、134宏转移、322原语转移、末尾唯一STOP全部核验；旧v1/原图/raw pose保留。
- 训练预注册：R/preregistered_comparison.json，SHA f4d24d32adf32b7b5f25059d504620e087e1d112d4ac9550b77dde0bec2e86f3。两架构 × 两输入模式 × 三seed20261115/16/17；每次50轮/600更新，固定最后checkpoint，不按结果挑epoch。十二次命令在training_execution.json，合计485.31s，退出均0。
- 两架构共用同原PNG、224双线性缩放、/255和冻结ResNet50权重；Seq2Seq特征2048×1×1，CMA为2048×4×4并有可训练位置嵌入/交叉注意力，不能称同一特征或单模块消融。深度、previous-action、进度辅助均关闭；CMA整个前动作嵌入输出清零，batch也不再含上一GT动作。
- 同架构/seed的初始state和全部50轮训练顺序hash一致；12模型Adam step均600，前动作参数无optimizer state；6个RGB-only的整个语言编码器无optimizer state。模型重载logits差0，全部CPU与GPU confusion一致，最大CE差9.82e-7。

命令模板（输出目录须新建；本次12个精确命令见JSON）：

    PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=/root/rivermind-data/uav_rgb/torch-cache OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=GPU-e9d01956-88db-b317-ea53-be3fb6cf7190       /root/rivermind-data/uav_rgb/envs/airvln/bin/python tools/train_openfly_subset.py       --manifest /root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/env18_causal_bc_v2/manifest.json       --airvln-root /root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/AirVLN       --out <新目录> --policy-type <seq2seq或cma> --training-mode <rgb_language或rgb_only> --epochs 50 --seed <固定seed>

| 三seed均值；仅4开发路线34帧 | 准确率 | 已支持类别macro recall | STOP recall |
|---|---:|---:|---:|
| Seq2Seq RGB+语言 | 52.94% | 20.18% | 33.33% |
| Seq2Seq RGB-only | 52.94% | 17.98% | 16.67% |
| CMA RGB+语言 | 52.94% | 28.90% | 33.33% |
| CMA RGB-only | 61.76% | 31.07% | 58.33% |

开发动作9占19/34，always9描述性准确率55.88%，并不等于闭环成功率。固定同图交换指令后Seq2Seq/CMA语言模型分别4/102、19/102帧改argmax；置零分别3/102、31/102。全部纯视觉模型对换语言/置零精确不变，前动作扰动全部差0。三seed仍只有4条独立路线，不能把102帧当独立样本；CMA更敏感不等于任务更好。详见baseline_audit/paired_offline_analysis.json及data_audit/training12_*。

### v2执行协议与200回合：保留的失败实验

v2每动作固定一次setPose、time(.02)、frames(2)，不重试、不放宽0.2m/1°/递增stamp门禁。短前缀、10动作和碰撞门禁通过后统一运行开发100回合+官方seen100回合，30arms合计4500.61s。100开发是4路线的策略/seed重复，100seen是20任务×5臂，不能称200条路线。

| 开发v2，3seed×4路线 | STOP<20 | 有效观测OSR | 可信末NE m（有效分母） |
|---|---:|---:|---:|
| 盲前进固定STOP | 2/4 | 4/4 | 19.96（3/4） |
| Seq2Seq RGB-only | 6/12 | 12/12 | 46.99（12/12） |
| Seq2Seq 正确语言 | 6/12 | 12/12 | 59.22（12/12） |
| Seq2Seq 交换/置零语言 | 各7/12 | 各12/12 | 各50.09（12/12） |
| CMA RGB-only | 5/12 | 12/12 | 74.09（12/12） |
| CMA 正确语言 | 6/12 | 12/12 | 33.36（12/12） |
| CMA 交换语言 | 4/12 | 12/12 | 50.78（12/12） |
| CMA 置零语言 | 2/12 | 12/12 | 111.33（11/12） |

开发集OSR已饱和；CMA语言对照有方向性而Seq2Seq没有。48对语言闭环起始RGB hash均不同，不能将所有轨迹变化归于文字；固定图像测试只能证实接线。缺反馈的NE是条件均值，不以不同分母比较排名。

| 官方seen子集v2，仅固定seed20261115 | STOP<20 /20 | 有效OSR /20 | 原协议终止 /20 | 碰撞 /20 |
|---|---:|---:|---:|---:|
| 盲前进 | 1 | 19 | 0 | 5 |
| Seq2Seq RGB-only | 5 | 14 | 0 | 2 |
| Seq2Seq RGB+语言 | 3 | 15 | 0 | 2 |
| CMA RGB-only | 4 | 17 | 2 | 2 |
| CMA RGB+语言 | 4 | 17 | 8 | 3 |

10个协议失败均为64–95动作后向下0.20002–0.22064m偏差；0 stale、0程序错误、0录像错误。另11个碰撞终帧也超pose门槛（seen9、dev2），代码按碰撞优先正确终止，不假改成新的protocol终止；全200共21反馈无效终帧，不能以原NE进行导航排名。所有失败都保留分母和原trace。

全部200段视频经FFmpeg完整解码，3923帧逐集匹配；400首末原PNG哈希、400预览原RGB区域、动作转移/原始起终点/距离/30contract/12checkpoint全通过。审计记录data_audit/evaluation200_independent_audit.json。首次审计一次性命令的帧数解析失误已保留并完整重解码，不是实验重跑。

### 官方任务和作者结果的比较范围

seen任务从原seen.json冻结20条env18 medium_short，任务清单SHA abc1418e1352623b35ef5039a7ff725b8fdf49c014423bd7811c9c57e3b6929a。源seen文件SHA c768d457dd91e638e95e1d7382e3e26c60ca7b9ff099ac8d7a909e9b9697d2cb。源row/hash重建起终点，训练清单只接受train.json，二者不可混用。排除完整轨迹/ID重叠，但3条与train/dev空间接近；4条起点已在20m内。OSR含起点，另存动作后OSR。总量20/203 env18、20/1800seen，unseen未评。

官方OpenFly-Platform c0750754 README场景表报告env18 NE77.4m、SR39.4%、OSR65.5%、SPL27.0%。这是作者表，未明确对应checkpoint和分母，不能与我们20条加安全门禁的结果混排。官方eval原实现最终3D距离<20即SR，不要求STOP；OSR取动作后命令位姿；legacy SPL=直线起终点距/(.001+命令累计路程)，没有标准SPL的max分母，可能大于1。它没有本地的300m/120s/碰撞终止和反馈门禁。我们只报本地诊断，未提交远程榜单。

公开源码审计见R/baseline_audit.json及official_sources，访问2026-11-16：OpenFly-Agent镜像元数据权重15,082,600,824B，未本地加载；LightVLN公开pre-fix快照不含权重；AeroBrain代码/权重仍coming soon；SimpleNav需要另一套Qwen/torch/FA依赖，未复现。这些是可运行性核查，不是性能失败。AirVLN Seq2Seq/CMA已在本机重训适配，不能冒充官方预训练导航模型。

主来源：[OpenFly固定README](https://github.com/SHAILAB-IPEC/OpenFly-Platform/blob/c075075497a7122bad82f5b76b9be926ad5a81b3/README.md#evaluation)、[官方eval实现](https://github.com/SHAILAB-IPEC/OpenFly-Platform/blob/c075075497a7122bad82f5b76b9be926ad5a81b3/train/eval.py)、[AirVLN](https://github.com/AirVLN/AirVLN)、[LightVLN](https://github.com/zhaoyiming/LightVLN)、[AeroBrain](https://github.com/Ideal-ljl/AeroBrain)、[SimpleNav](https://github.com/OpenBMB/SimpleNav)。本轮工具检索空响应时，以保存的官方API/原文HTTP200和SHA复核，不编造论文表或提交接口。

### v3修复验证，正式模型重测单独记录

四次固定v2长前缀重放有2次在51/53步超0.2m，2次走完仍偏0.1426/0.1472m；实际nav vz为-7.47至-8.18m/s，physics/render位姿几乎相同，被动等.35s全部状态不变。根因是setPose保留速度造成累积下落，不能靠继续等帧修好。

v3每动作一次simSetKinematics原子设位姿并清零线/角速度和加速度；固定API time(.02)+frames(2)保留，0.2m/1°/fresh及碰撞优先不变。明确属于无惯性运动学导航诊断，不是物理飞行改进或官方原协议。候选4×100动作、10动作、3碰撞全部通过，生产接口另100动作/101帧通过，最大误差0.003177399m。38评测单测通过，旧v1/v2保留；最终相关工具74/74通过（tools_final.log）。

已冻结v3 evaluator SHA78fcbeea43b1b1769a02cbb09a5703a61c68ce77c946f9f89d7aa90b4eacb3c5及R/evaluation_v3_freeze.json，五臂在evaluation_v3/official_seen完整100集统一重测。模型、任务、预算、种子不变；结果追加后不得与旧v2合并成同协议成绩。完整失败根因/原命令见sync_audit/final_protocol_failure_review.json/md。


### 视觉复核：目的性与停止问题

人工查看4个实际失败案例共8张首末preview，根代理另复看两张；图像有完整楼体细节，无黑帧，不能把这些任务失败简单归于渲染损坏。CMA语言episode014初图是与灰色砖立面/拱窗属性相符的楼体，前进12m后距目标约0.0099m，接着68个转向；xy不动，是原地摆头，不是绕楼飞行。相同任务Seq2Seq也曾经过目标，却继续飞离，在169.90m处对白色矩形窗立面STOP。另CMA错误STOP时近景主要为草地/天空，距目标208.20m。

这些可见属性仍不能证明目标实例唯一，重复白楼/矩形窗需要结合指令阶段和空间关系。下一步应先扩充真实目标/STOP样本及任务辨别诊断，不以加attention命名新方法。此批为UE AirSim图像，不能拿来评价3DGS重建质量。证据baseline_audit/v2_visual_failure_review.json/.md，图片SHA及完整动作保留。


### 最终推理对拍与下一批数据盘点

12个实际checkpoint×4开发路线共48对、408帧完成逐帧carry与整段前向对拍；全部通过atol=rtol=1e-5，argmax差异0/408。全局绝对差2.002716e-5，其中CMA语言两帧超过纯绝对1e-5但符合联合相对判据，不宣称bitwise一致。每条reset后hidden/prev/mask清零，首帧logits/hidden与原首帧精确相同。8组真实PNG特征SHA与先前审计一致；证据baseline_audit/trained_sequence_parity.json。

E1只读盘点六个已解压AirSim场景：官方train61,004行，其中53,712条有效0..9唯一末尾STOP、6,670条medium_short；筛前进+转向并排除已有冻结集后池5,006条。按seed20261116和场景配额给出100候选，env16/18各16，env23/26/gz/sh各17。960公开保留帧标签的0..9支持为[100,77,79,72,21,21,0,0,89,501]，5条含升降。六场景所有合法原生标注均无6/7侧移，不可宣称十类覆盖。

候选不等于训练ready：未验证raw因果组首、未做地理簇train/dev划分，本地所需新图命中0，缺图字节未知；100条固定revision元数据API名单已给出，未请求/下载。不能用exact去重冒充空间独立，扩训练前应核完整raw的距离、共享走廊与连通路径，再计算压缩/解压/缓存峰值。data_expansion_candidate.json SHA176a5240cd98ec36ad0b9da5c2cf13f8045b8af370feaaf2575c951c59c14df2。

### 最终 v3 seen100结果（2026-11-16 08:02 +0800完成）

| 策略；同20任务/seed20261115 | STOP<20 | 有效OSR（含起点） | 可信末NE m（n/20） | 碰撞 | 协议错误 |
|---|---:|---:|---:|---:|---:|
| 盲前进+固定STOP | 1/20 | 19/20 | 35.41（17/20） | 5/20 | 0/20 |
| Seq2Seq RGB-only | 5/20 | 14/20 | 78.99（20/20） | 1/20 | 0/20 |
| Seq2Seq RGB+语言 | 4/20 | 15/20 | 88.27（18/20） | 2/20 | 0/20 |
| CMA RGB-only | 5/20 | 17/20 | 97.60（20/20） | 1/20 | 0/20 |
| CMA RGB+语言 | 3/20 | 17/20 | 77.95（18/20） | 3/20 | 0/20 |

五臂全部退出0，共2750.78s；0协议/程序/录像/过期时间戳错误。100视频完整解码2767帧，200首末PNG和200预览原RGB区域均匹配，动作/来源/hash/预算全量独立核验通过。审计SHA a0f10c92af3bbb2860f8c82eeb298ac5544d79d2e859a7aa385ad5296e86e5b3。12碰撞全部保留，其中7终帧反馈越界，故可信NE分母合计93/100；不把条件均值作排名。非碰撞最大位姿误差0.003181912m。

v3通过本轮长序列验证，解决的是残余速度累积导致的评测漂移；它没有解决任务目的/STOP。两种架构语言版STOP成功均没有超过各自纯视觉版：Seq2Seq4/20 vs5/20，CMA3/20 vs5/20。不能宣称增加语言或注意力已改善泛化，更不能按条件NE靠近作者表就声称达到官方水平。CMA语言7/20步数耗尽、4/20距离耗尽；持续摆头/走过目标是下一步失败分析重点。20任务、单seed、同一seen场景的限制不变。

盲基线有效OSR19/20但仅1/20固定STOP在半径内，说明经过目标附近与完成任务差别巨大。4/20起点已在20m内；动作后OSR分别为盲19、SeqRGB13、Seq语言14、CMARGB16、CMA语言17。即使半径宽松，本轮结果仍不足完成无人机VLN要求。

历史v2的200回合和新v3的100回合共300段视频、6690帧，分别保留协议；没有把一次失败静默重跑成成功，也没有把同20任务重复当100独立路线。汇总comparison_v3.json，精确命令evaluation_v3_execution.json，逐片证据data_audit/evaluation_v3_independent_audit.json。观看index.html共300条，按协议/路线/seed过滤；视频开关仍为原入口 --record-video / --no-record-video。


最终独立重算与根代理汇总一致，closedloop_v3_interpretation.json SHA b151523696849dc416140b0c06d24ba17f43d058cc201c3b5a0e3b412c0a0c64。CMA语言同一episode014在v3中已不发生累积下落，却在距目标0.00133m处连续98次原地转向，直到100步预算终止。这将评测漂移与目标停止缺陷区分开：v3修好了本轮执行门禁，没有让模型学会任务完成。

收尾只对记录的本任务UE进程组835450发送SIGTERM，已确认无残留PID、41458无监听；llama-server793082与桌面281546保持运行。最终数据盘约5.7GiB、系统盘约706MiB、另一挂载约68GiB可用；不据总盘容量自动下载大模型。独立审计与source snapshots保留全部原失败；没有其他进程清理。
