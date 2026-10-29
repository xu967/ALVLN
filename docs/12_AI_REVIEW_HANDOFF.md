# 12｜供其他 AI / 代码审阅者独立审核的任务书

请严格执行“**源文件核对优先于方案文字；实测优先于推算；证据等级必须分开**”。审核本文时不要把此次 `run.sh test` 通过当作真实四旋翼可飞。

## 可独立运行的检查

```bash
./run.sh doctor
./run.sh test
./run.sh cpp-test
./run.sh scene --difficulty S1 --seed 42
./run.sh sim-probe
./run.sh ui
curl -fsS http://127.0.0.1:8765/api/v1/health
curl -fsS 'http://127.0.0.1:8765/api/v1/simulation/capabilities?task=lani_v1'
```

预期：本包测试通过；`sim-probe` 在未安装 Gazebo 的机器上**非零退出**是合规行为；API 默认无训练 run、无可用真正 simulator profile；浏览器未连接 API 的 `DEMO` 曲线不是研究成绩。

## 重点审核清单（请提供反证）

1. 研究方法 ECPT 是否真与 Self-Monitoring/L2Stop/LightVLN/AirAnchor 有机制差异？还是常见 GRU/attention/STOP 拼接？
2. 单目 RGB+长指令是否已出现可被语义任务逼迫使用的闭环成绩？请分别检查盲策略、图像置零、语言打乱。
3. 场景生成器是否仍存在固定起终点、固定直线/目标颜色捷径？学生是否可能读取 oracle JSON？
4. GZ0 的 motor→pose、GZ1 的真实 RGB 帧和 GZ3 的 episode trace 是否真的存在？单纯 SDF/插件标签不算。
5. API 是否可以通过篡改 HTTP `run_id`/`argv`/路径/Host/Origin 触发不受控外部程序？检查 profile 白名单与子进程所有权。
6. 训练 JSONL 与原始 stdout/stderr 是否一致？一个嵌套 NaN 或深度异常 JSON 是否拖垮全部 run？
7. C++ Tensor/Conv/GRU 的 forward/backward 有没有外部数值 oracle？目前唯一已测的是 CPU Linear-CE gradcheck，必须承认缺失。
8. AerialVLN 的 val_seen/val_unseen 是否与真实源协议一致？有没有把自建高清 RGB 与官方 224² 混报？
9. 用户是否在未做真值泄漏检查时将 privileged geometry 喂给 student？
10. SG2002 的 BF16/INT8 backend 是否真支持所有使用的算子？是否有完整相机→UART P95 实测？
11. CF-Drone 的高权限 MAVLink 通道、未实现 ALTHOLD、姿态漂移、开环失效保护如何阻断了实体飞行？有没有误把“有气压计扩展引脚”说成已能定高？
12. 并行研发有没有主次颠倒，继续美化没有实测训练/物理轨迹的 UI？
13. 上游归档的 LICENSE/商业限制/Git 提交是否会影响派生源码分发？
14. 代码文件数限制是否阻断必需的单元测试、图像/数据协议、日志与安全适配？
15. 任何 `POST` 是否可在没有管理员注册 profile 时产生伪造 success run？
16. `/api/v1/simulation-jobs` 目前只是控制平面；是否有误将它当作已经实现官方 Unity/UE Supervisor？

## 审核建议表

| 字段 | 必填内容 |
|---|---|
| ID / priority | P0/P1/P2，能复现为 P0 的必要条件 |
| Evidence | `path:line`, commit/hash, 实际运行命令及 stdout/stderr 摘录 |
| Confidence | `source_observed`, `locally_tested`, `official_documented`, `inference`, `not_verified` |
| Failure | 可产生的错误结果，是否有安全影响 |
| Fix | 最小更改，不要为了代码量凑代码 |
| Counterexample | 一项可能推翻评审者结论的测试 |
| Gate | 修复后要跑的自动化测试和失败退出码 |

## 重要否证条件

如果审核者提出“已经完成自动仿真”，必须要求真实 Gazebo/AirSim 执行 trace、图像 SHA256、控制器/相机/引擎版本、退出码与 evaluator 源实现；仅网页画面或模拟静态 SDF 不能接受。若提出“模型已在 5090 完成训练”，要求真实 loss CSV、checkpoint、训练日志、C++/CUDA 核心源码、样本出处和设备信息。
