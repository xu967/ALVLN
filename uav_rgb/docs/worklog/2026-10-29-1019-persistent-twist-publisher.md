# 常驻 twist publisher：采集指令发布去 CLI 化

- 时间：2026-10-29 10:19
- 范围：`sim/tools/collect.py`、`sim/tools/gz_io.py`、`sim/README.md`
- 结论：完成

## 做了什么

- `sim/tools/gz_io.py` 新增常驻指令发布：
  - `open_twist_publisher(topic)`：当前解释器能 `import gz.transport13` 时返回
    `_InProcTwistPublisher`（进程内 `Node.advertise`，等 `has_connections()` 后可用，
    实测 publish 6–100 µs）；不能时启动 `_HelperTwistPublisher`（`/usr/bin/python3
    -I gz_io.py --pub-server <topic>` 常驻子进程，stdin/stdout 协议，逐条回 ack，
    忽略 SIGINT，父进程 EOF 即退出；`UAVRGB_GZ_PYTHON` 可覆盖解释器）；都失败抛
    `PublisherUnavailable`。
  - `start(x,y,z,wz) / wait(handle) / abort(handle) / close()` 统一接口；helper
    超时或报错时关闭通道，后续 publish 立即抛错，交给调用方兜底。
  - `gz_io.py --self-check` 增加 helper 协议测试（假解释器验证 ready/ok/error/close/
    幂等/关闭后再 publish 报错），不需要 sim。
- `sim/tools/collect.py`：
  - `twist_fields()` 作为四动作 + 高度保持的唯一编码；`make_twist()` 只给 CLI 回退用。
  - 新增 `CliTwistPublisher`（`gz topic -p`，接口与常驻通道一致）和
    `publish_blocking()`（起飞用，失败重试一次，同旧 `cmd_twist`）；
    `_try_zero()` 安全网（先常驻通道单次尝试，失败再 CLI）。
  - `take_off()` 与 `collect_one()` 改走 publisher；动作窗口仍是 `hold` wall 秒，
    到点**立刻**发零速并等 `wait()` 完成（M2），不再依赖「两个 CLI publisher 同延迟」。
  - publish 异常：先 `_try_zero()` 补零速再抛错；零速发布异常同样先补零。
  - `main` 建 publisher 失败时打一行警告并回退 CLI；`finally` 改成**先发零速**、
    再关 publisher、再关订阅（避免订阅收尾卡住时带着最后一条动作指令多飞）。
  - `timing` 口径修正：动作完成时刻取 `publisher.wait()` 返回值；CLI 回退时先等动作
    再等零速，否则「观察到进程退出」的时刻会被记晚，误报 `zero twist overtook`。
  - `actions.jsonl` 字段未增未减；M1/M3/碰撞/sim_dt/stuck 逻辑未改。
- `sim/README.md`：4.5 补常驻 publisher / helper / CLI 回退与新耗时；新增
  `sim_dt` 与 `wall` 的口径说明（旧记录「每步约 1.7 s」是旧版 sim_dt，不是 wall）；
  修正 4.6.2 中已过时的「没有 sim-time 对齐 / 没有 gz-transport 直连」描述，并标注
  并行吞吐是改造前数字、未重测。

## 没做什么

- **没做 C++ 桥**（gz-transport C++ publisher / 自定义可执行）。
  - 原因：Python `gz.transport13` 在本机 publish 实测 6–100 µs，已满足 <0.6 s/步；
    再引入编译产物和部署依赖没有收益。
- **没做 sim-time hold**。`--hold` 仍是 wall 秒，动作窗口按墙钟计。
  - 原因：改成按 `/clock` 推进 sim-time 会改变 `--hold` 的语义和现有数据分布，
    且低 RTF 下会让每步 wall 不可控；超出本次「只换发布通道」的范围。
- **没做订阅通道直连**：5 个话题仍是 `gz topic -e` 子进程（常驻、内存取帧）。
  - 原因：本次只要求发布侧；订阅侧 I/O 已降到 0.015 s。
- **没改** `collect_parallel.py`、`scene_gen.py`、`ui/`、`src/`、`tools/`、
  `experiments/`、`teacher.py`、`save_frame.py`、world 文件。
  - 原因：任务铁律；`collect_parallel.py` 由另一 agent 修改。解释器问题在
    `collect.py`/`gz_io.py` 内部用 helper 解决，未要求外部改调用方式。
- **没有重测并行吞吐**：另一 agent 的 `collect_parallel.py --instances 6` 运行期间
  6/6 都自动选中了 helper，但该次 6/6 起飞超时（见「风险与遗留」），不能作为吞吐证据。
- **没做 episode reset**，也没清理中断时「PNG 已写、行未提交」的孤儿帧
  （`rgb_%05d.png` 可能比 `num_steps` 多 1 个）。
  - 原因：旧行为如此，不在本次范围；数据加载按 `actions.jsonl` 引用不会误用。
- **没验证父进程 SIGKILL 时 helper 的退出**（只验证了 SIGINT；helper 设计为 stdin
  EOF 退出，但未单独打 SIGKILL 实测）。
- **没在别的 Gazebo / 无 gz 机器上实测**：本机用 `UAVRGB_GZ_PYTHON=/nonexistent`
  模拟 helper 不可用，验证 CLI 回退能采完。
- **没改** `docs/HANDOFF.md` 等其它文档里旧的「1.7 s」说法（不在允许修改范围）。
- **没做**零速心跳/重复发送：正常只发一次零速，失败才走安全网。

## 待确认

- helper 解释器默认硬编码 `/usr/bin/python3`（可用 `UAVRGB_GZ_PYTHON` 覆盖）。
  是否需要自动探测多个候选解释器？当前未做。
- 中断时已写未提交的孤儿 PNG 是否要删？当前保留（旧行为）。
- `--hold` 是否最终要改成 sim-time 语义？当前明确为 wall（README 已写清）。
- 高并发低 RTF 下，`take_off()` 的 30 s wall 上限可能到不了 2.1 m（他人 6 实例
  运行实测 z≈1.4）。是否改成按 sim-time / 高度比例放宽，需要采集编排方决定；
  本次未改该逻辑，也未改 `collect_parallel.py`。

## 验证到哪一步

真实输出（本机 2026-10-29，`DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg`，GZ_PARTITION
逐次隔离；另有其它 agent 的 sim 在跑，主数据在空闲窗口采集）：

| 项 | 结果 |
| --- | --- |
| `python3 sim/tools/gz_io.py --self-check`（conda 与 `/usr/bin/python3` 各一次） | 通过（helper 假进程协议） |
| `python3 sim/tools/test_collect.py` | 13 项通过 |
| 20 步 hold=0.5，`/usr/bin/python3`，进程内 publisher | **total=0.530 s/步**（另一次空闲窗口 0.519）；I/O=0.015–0.029、动作窗口=0.500、发布=0.000–0.004、其他=0.000；20/20 步 ≤0.59 s；M1 20/20；manifest 20/20 validity 全 true；`actions.jsonl` 旧 13 字段齐全 |
| 撞柱：scene 42（门柱 x=1.5,y=0.067,0.4×0.4×3），`--policy forward` 20 步 hold=0.5 | step 9 起 `collision=True(contact+stuck)`；最终 x=1.06287、y=0.00613；柱面 x=1.3，base_link 半宽 0.235 m → 前脸距柱面约 **2 mm**；y 在柱宽 [-0.133,0.267] 内 |
| conda `python3`（无 gz 模块）3 步 hold=0.5 | 自动选中 helper：`警告: 当前解释器无 gz.transport13（No module named 'gz'），改用 helper 子进程发布` + `指令发布: gz.transport13（helper: /usr/bin/python3）`；total=0.518 s/步；3/3；manifest valid |
| conda `python3` + `UAVRGB_GZ_PYTHON=/nonexistent`（强制 helper 失败）3 步 | `警告: 常驻 publisher 不可用（当前解释器无 gz.transport13（No module named 'gz'）；helper 不可用（启动 /nonexistent 失败: [Errno 2] ...））；回退 gz topic -p CLI（每条指令约 +1.0 s）`；total=1.504 s（I/O=0.015、窗口=0.493、发布开销=0.995）；3/3 采完；manifest valid |
| SIGINT（`/usr/bin/python3`，最终代码，第 2 步后对进程组发 SIGINT） | collect + 5 个订阅进程全部退出（组内 `ps` 为空）；manifest `num_steps=2/20`、validity 全 false、`error=KeyboardInterrupt`；已采 2 行 flush、M1 2/2；中断后 1.2 s 内滑行 2.8 cm（减速），稳定后 1 s 位移 **0.0000 m** |
| SIGINT（conda + helper，改 `finally` 顺序前） | 组内 collect + 5 订阅 + helper 全部消失；manifest 2/20、validity 全 false；后续 1 s 位移 0.0000 m |
| 收尾 | 本任务自建的 sim 全部 kill；`ps` 无本任务的 `uav_rgb.sdf` / `--pub-server` / `collect.py`；`/tmp/pp_*` 已清理（他人 `/tmp/uavrgb_*` 未动） |

新旧对比（同参数 hold=0.5、20 步）：

| 指标 | 旧（`gz topic -p`） | 新（常驻 publisher） |
| --- | --- | --- |
| 每步 wall（`timing.total`） | 1.518 s | **0.519–0.530 s**（本机有他人 sim 时单步最差 0.59 s） |
| I/O | 0.016 s | 0.015–0.029 s |
| 动作窗口 | 0.494 s | 0.500 s |
| 发布开销 | 1.008 s | 0.000–0.004 s（多数步 0.000–0.001） |
| 其他 | 0.000 s | 0.000 s |
| 20 步步循环 | ~30.4 s | ~10.4–10.6 s（2.9×） |
| M1 | 20/20 | 20/20 |
| 回退 CLI | — | 1.504 s/步（与旧版一致，发布开销 0.995 s） |

## 风险与遗留

- publish 是 transport 层 fire-and-forget（`Publisher.publish` 返回即认为已交给
  gz-transport，没有端到端 ack）。证据是 `has_connections()` 后才开跑，且实测
  前进位移与零速停住；但没有做逐条消息的接收确认。
- helper 一旦出现超时/错误会关闭通道，**当前 episode 会报错退出**（同时先用 CLI
  补零速），不会静默降级继续飞。若希望单条失败后自动重连 helper，需要另做。
- 低 RTF / CPU 争抢时，hold 末尾的 `time.sleep` 可能过冲：本机同时有他人 6 个 sim
  时实测一步窗口 0.57 s、总 0.59 s（仍 <0.6，但余量只有 10 ms 级）；publish 本身
  不是瓶颈。
- 只读观察（**未计入验收**，另一 agent 的 `collect_parallel.py` 运行，本任务未改其
  文件、未干预）：
  - 两次运行共 12/12 个采集进程都自动选中 helper，说明不改 `collect_parallel.py`
    也能用上常驻发布；其中一次 4/6 实例采集成功，`--hold 1.0` 时 **1.01–1.02 s/步**
    （I/O≈0.02、窗口 1.00、发布≈0），manifest 正常。
  - 同一次运行另有 2/6 起飞超时（30 s 后 `z=0.055`，停在地面）；更早的 6 实例运行
    6/6 起飞超时（`z≈1.4`，RTF≈0.15）。失败原因**未定位**：可能是 6 路软件渲染/
    高并发下 sim 不推进或首条指令时序问题，也可能是资源争抢；没有旧 CLI 版本在同
    负载下的基线可对比，因此不能断定与本改动无关。`take_off()` 的 30 s wall 上限和
    重试逻辑本次未改（仅把发布通道换成常驻），单实例运行起飞 100% 成功。
  - 并行吞吐需要在编排方排查后再测；本报告不宣称并行场景的可靠性。
- 订阅仍是每实例 5 个 `gz topic -e` 子进程，未做 gz-transport 订阅直连；进一步提速
  需另立任务。
- 未做长时间（数千步）运行的内存/句柄泄漏测试，也未在非 Linux / 非 Harmonic 环境
  验证。
