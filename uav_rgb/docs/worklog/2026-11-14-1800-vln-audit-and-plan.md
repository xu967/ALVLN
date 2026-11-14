# VLN 全量审查与交接

- 时间：2026-11-14 18:00
- 范围：`uav_rgb` 的 VLN 任务层、3DGS、grounder、仿真控制、原型训练配置
- 结论：部分完成

## 做了什么

- 并行审查了 `sim/3dgs/run_vln.py`、`sim/3dgs/vln.py`、grounder、3DGS 场景链路和飞控仿真。
- 确认当前指令文本仍只用于任务组织/视频字幕；teacher 使用 waypoint，forward 盲飞，尚不能宣称语言理解。
- 确认 3DGS 主要问题是训练相机覆盖不足；短航线和视角外发糊不能靠渲染参数单独解决。
- 确认抖动主要来自 20 ms 离散动作直接改位姿，以及停-转-走动作切换；真机效果仍需实测校准。
- 保留并验证现有 G1 可解释词法 grounder、原型训练配置化改动和对应单测。
- 形成后续实施顺序：M2 G0 方位条件学生 → M3 G1/G2 语言落地 → M4 多段/长航线 → M5 数据扩容与真机协议。

## 没做什么

- 没有把字幕或 waypoint 偷换成语言理解结果；原因是这会污染 VLN 指标。
- 没有直接更换公开数据集；原因是 AerialVLN 等尺度与当前 E1 微型无人机不匹配，应先做覆盖度门禁和长航线小规模验证。
- 没有修改飞控硬件参数或宣称真机稳定；原因是当前只有仿真证据，缺少称重、推力曲线和传感器实测。
- 没有删除实验产物和废弃文档；原因是仓库/数据盘归属与保留清单尚未逐项确认，误删不可逆。
- 没有执行完整 CUDA/C++ 构建；本轮先完成 Python 逻辑回归，GPU/CUDA 长测留给 M2 实现后的专项验证。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python -m unittest sim/3dgs/test_grounder.py tools/test_audit_vln_grounder.py proto/test_config.py` | 通过，13 项（1 项跳过） |
| 上轮 `bash tools/check.sh --python-only` | 更正：仅取得中途输出，不能判定通过；本次重新运行 |
| `nvidia-smi -L` | 发现 1 张 RTX 2080 Ti，UUID `GPU-4910bb04-2d00-ca81-f64a-2031568758ad` |

## 待确认

- 是否批准增加 500 GB 数据盘并下载 Wanderland/InteriorGS 场景。
- 真机 LicheePi 型号、ESP32 协议、成功半径和 STOP 语义。
- 删除历史实验目录/废弃 md 的明确清单。

## 风险与遗留

- 当前指标只能作为 M1 任务层上界/下界，不能作为语言模型成绩。
- 下一步必须实现 M2 bearing 条件学生，并单独报告 STOP、碰撞、SPL、nDTW/sDTW。

## 更新 2026-11-14：按 §10 继续执行

### 做了什么
- 重新核对个人 skill 和现行计划：卡 A 已完成，B/C/D 是当前任务，E 必须等依赖验收。
- 卡 B：探针将深度有限且大于零的比例独立纳入门禁，避免锐利棋盘配零深度也通过；报告升 v2，0.9 默认阈值明确为未标定诊断值。
- 卡 D：词法 grounder 对不支持语种、否定、多目标、升降返回明确 unsupported，不默默忽略指令。
- 卡 C：复算历史 240 集、9174 步位移数据；真实 Wanderland GPU 探针与 Python 全量回归见实验记录。
- 恢复标准镜像流程。四份镜像未跟踪文件经逐字节核对与已提交源文件一致后移除重复副本；上一轮 c934b01 已推送为 GitHub e67badf。

### 没做什么
- B 未完成回程/扫视/多高度覆盖与验证集阈值标定，不能宣布长航线合格。
- C 未更改 PID、执行器或教师版本；历史 hold=1.0 统计不能当成 hold=0.5 标定。
- D 未接正式语言闭环；本地没有独立地图候选坐标表，禁止用测试终点构造答案。
- E/F 未启动训练，按计划依赖顺序推进。
- 更正上轮“完整阅读”和确定抖动根因的表述：只完成了部分入口审查，尚未完成全仓逐文件验证；既有 swept 和显式 STOP 已实现，不能重复当作待开发。

### 验证到哪一步
- 改动相关 unittest 20/20，退出码 0；历史位移复算退出码 0。完整结果另记实验文档。

## 用户要求正式训练：已启动

做了什么：新增 `experiments/2026-11-14-l1-language-formal/run.sh`，启动 C++/CUDA L1 模板语言条件基线，三个种子1234/1235/1236顺序各20轮，text_dim=16、batch=64、lr=0.0005、drop_collision。输出 `/root/rivermind-data/uav_rgb/exp-l1-language-formal-20261114`，记录数据/指令/二进制哈希及PID身份；冻结二进制，保留真实日志。

没做什么：没有用纯视觉Torch训练冒充VLN；没有启动未实现的3DGS条件学生；没有宣称训练完成或泛化提升。旧L1数据缺split_scheme/split_salt，训练器只校验scene_id隔离，不能据此宣称新场景泛化。

验证：bash -n通过；训练进程809293已使用GPU 578MiB，输出config.json/metrics.jsonl/results.csv，等待正式epoch结果。停止使用 `bash tools/stop_daemon.sh /root/rivermind-data/uav_rgb/exp-l1-language-formal-20261114`。

补充数据审计：L1 train14496、val1688、test0；没有独立测试集，不能将val数字报告为test。子代理最初“trunc不存在”的判断有误，另一代理实读证实存在；本轮未采用该纯视觉数据。

## 更新 2026-11-14 15:15

- 用户要求充分利用租用 GPU 后，核实 RTX 2080 Ti 22,528 MiB；原语言基线三 seed 已完成，`DONE=0`。
- 新增 `experiments/2026-11-14-l1-language-formal/run_parallel.sh`，启动三个相同预算的 `--vision-only` 对照并发运行；当前每个进程约 572 MiB，训练仍在运行。
- 核实公开数据：SAGE-3D VLN、Wanderland 扩展、AirVLN、MatrixCity 均未下载完成；仅有空目录/元数据。已下载可训练数据是内部 `l1-v4-cf`，其 test split 为空。
- 本轮没有冒充完成正式闭环评测：评测若与训练抢同一张卡会显存分配失败；已有历史旧数据指标与本轮数据严格分开。
