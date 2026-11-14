# VLN 实验启动、数据补齐与算法复核

- 时间：2026-11-14 22:00（Asia/Shanghai；后续状态追加在本文件）
- 范围：`sim/3dgs` SAGE v3 控制对照、Wanderland 下载与长路线候选、算法/控制/渲染/数据语义审查
- 结论：部分完成；SAGE v3 对照、四个缺失归档、新长路线核心下载/隔离导入/画质筛查、满电门禁和 3 seed 配对均完成；语言学生与独立 catalog 仍未实现

## 做了什么

- 用并行 agents 分别审查了 VLN 算法/学生接口、控制和抖动、3DGS/导入与可视化、Wanderland 指令语义/泄漏风险。完整结论和后续门禁见 `docs/PLAN_2026-11-13_vln.md §10.8` 与 `docs/experiments/2026-11-14-v3-baseline-long-route-followup.md`。
- 在 GPU 空闲后启动并完成 SAGE 六场景显式 STOP v3 teacher/forward 配对控制基线：各 249 请求、key 完全一致、0 异常；teacher SR 0.8434 / OSR 0.8554 / SPL 0.7003 / collision 0.1486，forward SR 0 / OSR 0.2410 / collision 0.9639。逐集结果和准确口径在实验结果文档。
- 从公开 Wanderland HF 清单补下旧场景缺失的四个图像/鱼眼 tar：合计 571,330,816 B，四包大小与官方 tree 完全一致、SHA-256 已记录、tar 完整遍历各包成功；未解包，未覆盖既有 scene 或删改用户数据。
- 从公开 evaluation-ready 清单筛出新场景 `10nvbQnL3fUaY8zb5Orh_t3mk7wnal26i`：18 条指令路线、最长参考距离 48.2638 m、9 条 ≥30 m、Apache-2.0。核心 3DGS/点云/USDZ/位姿/COLMAP 资产已完整下载到数据盘独立原始目录并校验。
- 固定 v3 本次代码 commit、数据/请求 manifest SHA 与 GPU UUID；下载和实验均写数据盘。初始盘余 243 GB，未改代理配置。

## 没做什么

- 没有训练语言学生、视觉 grounder 或新 VLN 网络。原因：现有 `run_vln` 没有接入学生；`StudentAdapter` 无目标条件且 STOP 固定 false，先需要实现与单测其完整输入/停止契约。
- 没有把 v3 teacher 成绩标作语言导航成功，也没有把 forward 的零动作切换写成平顺改进。原因：teacher 直接读取 waypoint/全局占据图；forward 只盲目前进。
- 没有修改真机 PID、ESP32 飞控、动作空间或执行器参考斜坡。原因：现有 SITL profile `calibrated=false`，证据优先指向策略切换时未考虑残余速度，不能据此外推真实飞控参数。
- 没有解包旧场景四个原图 tar、未下新场景原图；没有用候选的 `goals.position`、参考轨迹或 `gt_locations` 建语言 catalog。原因：渲染核心资产已足够；这些轨迹/目标字段是评测答案，只能用于明确的oracle控制诊断。
- 没有宣称整个新场景适飞或算法完成。原因：当前只筛出3条通过单一路径探针的路线，45m路线被拒；满电 smoke 与配对虽已完成，仍只有单路线控制诊断，半电失败仍是真实结果。
- 没有跑真机、公开榜单对比、独立 language grounding 指标。原因：本轮先做已授权的控制对照、补数据和可执行验证门禁；当前数据/学生/独立标注条件不足。
- 没有删除实验数据或批量整理目录。原因：目录用途和引用关系不全，且本轮目标是恢复实验并保留可追溯产物。

## 待确认

- ep10 是否补人工接触表、画质探针阈值如何校准；当前只将其列为自动筛查 PASS，不把上游路线长度当结论。
- 正式地标 catalog 的独立标注来源和标注流程需要在查看测试目标前冻结；本轮尚未创建目录。

## 验证到哪一步

| 项 | 结果 |
|---|---|
| v3 teacher `vln_eval.json`、249 episodes、0 failure | 完整文件存在；teacher wrapper 的 shell 句柄已失效，无法读取退出码，未伪称拿到退出码 |
| v3 forward `vln_eval.json`、249 episodes、0 failure | 完成，退出码 0 |
| 两臂请求 key 对齐 | 249/249 完全相同 |
| Wanderland 四个缺失 tar | 官方大小、SHA-256、tar 遍历通过；未解包 |
| 新场景核心资产 | 8文件大小匹配、6个LFS SHA匹配；PLY/JSON/COLMAP/USDZ解析通过 |
| GPU/算力 | 运行前检查 RTX 2080 Ti UUID `GPU-4910bb04-2d00-ca81-f64a-2031568758ad`；v3 实验结束后无仿真 GPU 进程 |
| 新代码测试 | `test_dynamics_sitl.py` 13、`test_dynamics.py` 6、`test_video.py` 13，32项通过；真实proc测试覆盖同刻遥测及缺字段null |
| Git | 实验收尾并review后一起提交/镜像推送；最终状态追加在下方 |

## 风险与遗留

- 当前已证实项目没有完整 VLN 策略闭环。先接语言/视觉目标估计、置信度拒绝、短历史和独立 STOP；不能继续用 teacher/control 分数代替语言算法。
- 抖动优先按“固定动作动力学响应”和“策略切换/刹车净空”分开配对测量；没有真实动力学标定前，不调整真机 PID。
- 长路线场景必须隔离导入。现有 `tools/scene_import_wanderland.py` 硬编码旧场景 ID/目标目录，直接运行会写入已有场景路径。
- 镜像只解决当前 HF 官方域名连接问题；保留每个文件的预期字节数、LFS OID/SHA 与失败日志，核心资产全数校验后才允许 `CORE_OK`。


## 更新 2026-11-14 22:50

### 做了什么

- 新场景核心8文件全量校验，隔离转换/对齐完成；正确world→nav变换后筛出3条30–43m路线通过当前深度+锐度探针。ep4/9人工看图仍有局部树冠拉伸，但无大范围崩坏；ep10尚未人工复核，探针阈值也未校准。
- 纠正本轮临时驱动直接拷贝world x/z的错误；初次错误坐标下所有路径/质量结果作废。最终scene只采用 `R.T@(world-t)` 变换，旧场景未覆盖。
- 真跑41.8m半电smoke并定位触地原因；CPU精确重放、原地悬停和单改SoC重放确认是未标定电池/推力预算耗尽，不是ToF或场景地形先失效。
- `sim/3dgs/dynamics.py` 最小追加proc已有的电压、ToF、高度估计、sticks/motors到trace，缺失值null；`test_dynamics_sitl.py` 增加真实进程与缺字段兼容测试。未改控制逻辑、默认profile或真机PID。
- 独立SoC1.0诊断profile由现有make_params_variant生成到数据盘；CPU完整预算悬停PASS，GPU路线smoke成功；3seed配对已完成。精确结果/命令/限制见实验文档。
- 计划§10.9按B/C/D/E现有任务卡补充下一位实现者的输入契约、文件范围、验收与消融；没有另起一套训练/评测架构。

### 没做什么及原因

- 没有更换默认50%电量、改电池容量/质量/PID或固件，也未把满电称算法改进；先保留两profile可对账。
- 没有修订 `physics_verified` 含义或新增低电终止；本轮先补诊断证据，后续需单独约定失效/终止协议和测试。
- 没有实现语言学生网络、独立catalog或VLM；现有输入/STOP/标注权限缺口已列出，不能在无独立标签时训练出虚假成功。
- 没有批量删除旧文档和实验；本轮保留复现实验产物和错坐标诊断记录，避免丢证据。临时下载脚本、importer副本及运行驱动在用完后清理。

### 验证与风险

- 32项相关动力学/视频测试PASS；`git diff --check`通过。没有重跑全仓测试（控制算法未改，实验真跑已覆盖相关链）。
- 半电失败和满电成功在独立目录/profile分报；配对时存在外部GPU训练，锁步物理步长不受墙钟影响，但不将本轮wall time用于推理速度比较。
- 恒定nav高度不是恒定离地高度；当前SITL ToF是水平面，不能由这些结果推出真机可飞或地形跟随能力。


## 更新 2026-11-14 23:15

### 做了什么

- 收尾并核对 SoC=1.0 独立 profile 的 CPU SAFE 门禁、单集 route0 smoke 和 141114–141116 三组 teacher/forward 配对；结果与 `paired_summary.json`、`paired_results.json`、integrity 记录一致。
- 将实验文档更新为真实完成状态，明确半电失败与满电成功不可合并；补充 teacher/forward 数值、profile、同 seed 一致性和外部 GPU 不用于延迟结论。
- 修正计划的指标契约：route-waypoint 与预测目标控制代理不等于 E2E VLN；按实现写清 SR/OSR、SPL、NE、碰撞率、STOP 率分母；明确四动作+独立 STOP、SAFE 与 STOP 的边界。
- 保留动力学遥测改动及 32 项相关测试结果；trace 中 ToF/高度估计是 proc 原始 frame，`pos.z` 可能含 `ground_z`，已写入交接说明。

### 没做什么

- 没有训练或接入视觉语言学生、视觉 grounder、独立人工 catalog，也没有报告真正 E2E VLN success；原因：现有输入/标签权限和 STOP 输出契约尚未实现。
- 没有修改默认半电 profile、PID、固件、动作空间或真机参数；原因：本轮只做可追溯诊断，未完成动力学标定。
- 没有把单路线满电 teacher/forward 当算法优越性、泛化或公开榜单成绩；原因：teacher 读取特权目标点和全局 A*，forward 只是盲飞对照。
- 没有删除旧实验、日志、文档或批量整理目录；原因：引用关系和证据链尚未逐项确认，避免误删可复现产物。

### 待确认

- 是否为 ep10 补人工接触表并校准画质探针阈值；当前只将其列为自动筛查 PASS。
- 独立 catalog 的标注人、来源文件和目标语义是否先冻结，再开始语言反事实实验。

### 验证到哪一步

| 项 | 结果 |
|---|---|
| `test_dynamics_sitl.py` | 13/13 通过 |
| `test_dynamics.py` | 6/6 通过 |
| `test_video.py` | 13/13 通过 |
| SoC=1.0 CPU SAFE 190.5s | 通过，高度1.955–2.001m |
| SoC=1.0 route0 smoke | 通过，343步、42.016m、无碰撞 |
| SoC=1.0 paired 3 seeds | 通过，teacher 3/3；forward 3/3 碰撞 |
| `git diff --check` | 通过（本轮文档和代码 diff 均检查） |

### 风险与遗留

- 满电配对只覆盖一条路线和一个未校准动力学 profile；不能和半电失败结果合并，也不能外推真机。
- `execution_recipe.json` 的 paired 状态字段仍是旧生成值 `not started`，以实际结果 JSON 和退出码为准；需要后续修复生成器。
- 当前代码仍无真正语言→视觉→控制→自主停止闭环；下一步必须先完成独立目标数据和 E0/D1/D2 门禁。

## 更新 2026-11-15 00:35：公开数据下载门禁与自动触发

### 做了什么

- 恢复 AerialVLN `train.json` 的截断下载；文件大小 `303,722,815` B、SHA-256 和 JSON 解析均通过。`val_seen`、`val_unseen`、`test` 也完成 JSON/大小核对，四个 split 的 episode 结构可读。
- 完成 OpenFly `Annotation` 的公开子集下载：`train/seen/unseen` 共 103,226 条记录、3 个文件均可解析；现有审计报告无 parse error、动作/位姿数组无长度不一致。
- 保留 AerialVLN 模拟器的单个断点下载进程，不重复启动；其清单为 277 个文件、`40,541,074,951` B（约 37.76 GiB），当前仍在续传 4 个大 PAK。
- 在数据盘挂起 `downloads/after_download_runner.py`：等待 simulator manifest 全部达到预期字节数后，逐文件校验清单 hash，写 AerialVLN/OpenFly 审计结果，再串行跑 1 集现有 SAGE-3D `teacher` 与 `forward` smoke。输出位于 `/root/rivermind-data/uav_rgb/auto-after-download-20261115/`。
- 克隆官方 AirVLN 源码到数据盘 `vendor/AirVLN`，用于后续路径/依赖审计；未宣称官方评测已运行。

### 没做什么及原因

- 没有下载 OpenFly 全库约 2.97 TB；当前数据盘无法容纳。用户完成 HF 登录后，已安全验证本机 token 可用（不输出凭证）；OpenFly_DataGen 的文件请求仍返回 403，账号尚未获准访问，因此不绕过门禁。DataGen 清单为约 115.83 GiB、主要是仿真环境/地图资产，README 限非商业用途；先完成 AerialVLN 下载，再等 HF 页面审批或条款确认后按需下载。
- 没有启动 AirVLN 官方 eval：当前缺少 AirVLN checkpoint、Habitat `gibson-2plus-resnet50.pth`、正确的 `ENVs/` sibling 路径和兼容运行环境；官方脚本默认 8 GPU，而本机只有 1 张 RTX 2080 Ti。无 checkpoint 时不能启动会持续等待的 eval。
- 没有把现有 SAGE teacher/forward smoke 当作语言导航成绩；它只验证已有控制器，正式语言闭环仍缺独立、冻结且不泄漏的地标 catalog。

### 验证与待办

| 项 | 当前结果 |
|---|---|
| AerialVLN train SHA/JSON | 通过；train 16,386 episodes |
| AerialVLN val/test JSON | 通过；val_seen 1,818、val_unseen 2,310、test 4,830 |
| OpenFly Annotation audit | 通过；103,226 records，0 parse error |
| AerialVLN simulator | 121/277 文件完整（记录时点）；4 个大 PAK 仍下载 |
| 自动实验触发 | 已挂起，等 simulator manifest 完整后执行 |
| OpenFly_DataGen | 未下载；等待用户在主机完成 HF login 后再评估 |

自动触发完成后，再把实际 smoke 退出码和审计摘要补入本工作日志及实验文档；没有新的真实语言导航指标前，不会编造公开基准成绩。

## 更新 2026-11-15 01:40：完整模拟器门禁、SITL 修复与单集配对

### 做了什么

- AerialVLN 模拟器下载完成：277/277 文件、40,541,074,951 B；逐文件内容哈希全部通过。四个标注 split 继续通过大小、JSON、SHA/Git blob 校验。
- 自动 watcher 首轮确实被触发，但使用了旧的 `build-cf-sitl-v2-20261107/cf_sitl_proc`，其 state 没有 `lib_version`，因此 teacher/forward 在首个 episode 前退出。将 watcher 改为当前 `tmp/build-cf-sitl/cf_sitl_proc`（lib31），并单独重跑相同场景、seed、limit=1 的配对。
- 真实 SAGE teacher 单集完成：84 个动作，显式 STOP，SR=1、OSR=1、SPL=0.775948、NE=0.321754 m、碰撞率=0、STOP率=1。
- 真实 SAGE forward 单集完成：12 个动作后碰撞，SR=0、OSR=0、SPL=0、NE=3.075232 m、碰撞率=1、STOP率=0。两者都使用 `cf_sitl_v1:1bc8fedf:lib31`，同场景和同 seed。
- 使用修正后的 proc 跑 `sim.3dgs.test_dynamics_sitl`，13/13 通过。产物保存在数据盘 `/root/rivermind-data/uav_rgb/auto-after-download-20261115-retry1/`。
- 将自动 watcher 的 SITL 路径改为稳定的 lib31 构建，避免下次下载门禁完成后再次在启动前失败。

### 没做什么及原因

- 没有把这组 SAGE 数值写成语言导航成绩。该入口 `grounder=none`，teacher 读取特权目标/规划，forward 只是盲目前进；它们只证明渲染、动力学、碰撞和 STOP 链可运行。
- 没有宣称 AirVLN 官方 benchmark 已运行。官方环境尚在安装依赖，且没有 AirVLN policy checkpoint；当前只准备数据、路径和 Gibson backbone。
- 没有把 OpenFly_DataGen 记为已下载。账号 token 和 gated manifest API 可读，但实际文件 GET/HEAD 仍返回 HTTP 403；后台下载器每 10 分钟重试，没有绕过访问控制，也没有生成空的假文件。
- 没有修改导航算法、真机 PID 或可视化代码。当前结果足以定位“控制可运行但没有语言闭环”，继续改算法前仍需独立、冻结且不泄漏的目标 catalog。

### 验证与风险

| 项 | 结果 |
|---|---|
| AerialVLN simulator | 277/277，逐文件哈希通过 |
| AerialVLN annotations | train 16,386；val_seen 1,818；val_unseen 2,310；test 4,830，审计通过 |
| OpenFly Annotation | 103,226 条，0 parse error，字段长度审计通过 |
| SAGE teacher smoke | 1/1 成功；仅 controller diagnostic |
| SAGE forward smoke | 1/1 完成但碰撞；仅盲飞下界 |
| SITL protocol/dynamics tests | 13/13 通过 |
| OpenFly_DataGen gated assets | 实际内容仍 403，未下载 |
| AirVLN Python 3.8 env | 已创建；torch/requirements 尚未完成安装 |

AirVLN 依赖安装曾把大 wheel 的临时文件写入 30GB 系统盘并报 `No space left on device`；后续安装已改用数据盘 `TMPDIR` 与 `PIP_CACHE_DIR`。当前 CUDA wheel 仍需完成后才能做官方单场景 collect smoke，不能用等待时间伪造实验结果。

## 更新 2026-11-15 02:40：AirVLN 非 root 单路线采集 smoke

### 做了什么

- 完成 AirVLN Python 3.8 环境和官方依赖安装：torch `2.1.2+cu118`、torchvision `0.16.2+cu118`、AirSim `1.7.0`、`pytorch-transformers`，并确认 CUDA、Gibson 权重和 ResNet-50 缓存可读。模型初始化 smoke 正常，但它本身没有启动场景。
- 发现官方 Unreal 二进制拒绝 root 启动；没有修改原始 `vendor/` 权限，而是在数据盘建立 `airvln-runtime-ubuntu/` 隔离运行目录，由 `ubuntu` 用户运行。`env_5` 完整复制到该目录，标注只保留同一 `trajectory_id=39KV3A5D2G5VV16NHAJOVU94299S7W` 的 3 条语言指令，模型/词表/其余数据只读链接。
- 定位并记录了无显示环境下的渲染崩溃：UE 能启动并开放 RPC，但第一次 `simGetImages` 会超时后 SIGSEGV。继承已有 `DISPLAY=:20`、`QT_X11_NO_MITSHM=1`、`SDL_HINT_CUDA_DEVICE=0` 后，AirSim 直接返回 RGB `224x224x3` 和深度 `256x256`，图像尺寸契约通过。
- 在隔离副本中保留 UE 子进程、增加 stdout drain，避免官方 server 在 readiness banner 后杀掉 shell 导致端口失效；原仓库和原始官方副本未改。
- 真实运行官方 `collect`：场景 `env_5`，`seq2seq`、teacher forcing、batch=1、`maxAction=220`。采集完成 2 个 minibatch、每个 190 步，共写入 3 条 episode feature、190 条 RGB 和 190 条 depth LMDB 记录；日志以 `END data_it: 0` 正常收尾。

### 没做什么及原因

- 没有启动官方 eval 或报告 AirVLN benchmark 分数：当前只有随机初始化/未训练的 policy，没有 checkpoint；本次是数据、渲染、AirSim RPC 和 LMDB 写入 smoke。
- 没有把 collector 的动作成功率当 VLN 成绩；采集使用 teacher forcing，结果验证的是采集链而不是自主语言导航。
- 没有把隔离 runtime 的 server 生命周期补丁直接合并到仓库：它针对外部官方源码，改动只服务本次可回溯 smoke；后续若长期使用，应在 vendoring 方案中单独维护并加进程组清理测试。
- 没有停止或修改其他 GPU 训练进程；AirVLN 使用剩余显存完成 smoke，结束后检查确认没有残留 UE/server 进程。

### 验证与风险

| 项 | 结果 |
|---|---|
| AirVLN imports/CUDA | 通过；torch `2.1.2+cu118`，CUDA 可见 |
| 无 DISPLAY 的首次采集 | 失败；图像请求超时并触发 UE SIGSEGV，已定位为渲染显示环境 |
| `DISPLAY=:20` AirSim 图像探针 | 通过；RGB 224×224×3，Depth 256×256×65536 |
| 官方 collect smoke | 通过；3 episode LMDB、190 RGB、190 depth |
| collector 输出 | `/root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/DATA/img_features/collect/airvln-single-route-20261115-r3/` |
| OpenFly_DataGen | 仍未下载；实际文件端点 HTTP 403，后台重试保留 |

该 smoke 仍不是 E2E VLN：语言只进入官方 collector 的训练样本，policy 没有训练，且没有自主停止、公开评测或与 SAGE 指标的可比性。长期运行 AirVLN 还需固定显示/渲染启动契约、维护 server 子进程回收，并在 checkpoint 产生后单独做 eval 门禁。
