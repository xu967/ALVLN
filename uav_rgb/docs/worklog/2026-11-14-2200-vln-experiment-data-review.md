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
