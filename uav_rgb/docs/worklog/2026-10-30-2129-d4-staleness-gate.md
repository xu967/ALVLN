# D4：depth/contact 停更门禁

- 时间：2026-10-30 21:05–21:29
- 范围：`sim/tools/collect.py`、`sim/tools/test_collect.py`、`sim/README.md`
- 背景：`docs/TASKS.md` 待排第一条 D4。旧行为：depth 话题停更时
  `_pick_depth` 仍从缓冲取旧帧；contact 订阅进程退出后 `_contact_since`
  一直返回 False，碰撞会静默漏记。

## 做了什么

| # | 项 | 修法 |
| --- | --- | --- |
| 1 | depth 停更 | `collect_one` 先查 `depth.healthy()`；解码后要求 `abs(depth_ts-rgb_ts) <= DEPTH_STALE_NS=100 ms`，超出该步失败（旧帧不再当本步观测） |
| 2 | contact 进程退出 | 步首、动作窗口内每轮、动作窗口后各查一次 `contact.healthy()`；窗口内退出先补零速再失败 |
| 3 | 起飞阶段 | `take_off` 起飞前查 contact 订阅；起飞过程中退出先补零速再失败 |
| 4 | 测试 | `test_collect` 新增 5 例：depth 旧帧、depth 订阅退出、contact 步首退出、contact 窗口内退出、起飞前 contact 退出；共 20 项 |
| 5 | 文档 | `collect.py` 模块说明、`sim/README.md`、`TASKS.md`、`ENGINEERING_RULES.md`、`HANDOFF.md` 同步 |

阈值说明：audit 的 `time_pairing` 仍是 50 ms（数据质量门禁）；采集侧的 100 ms
是「停更」门禁，给正常配对抖动留余量。T5 全量 max gap 36 ms，1 集 smoke max 32 ms。

## 验证

| 项 | 结果 |
| --- | --- |
| `python3 sim/tools/test_collect.py` | **20/20 PASS** |
| `bash tools/check.sh` | **exit 0**，CTest 20/20 |
| 真实 Gazebo 冒烟（1 实例 × 1 集 × 8 步，seed 200000） | 8/8 步完成，depth gap max 32 ms，无 D4 误杀 |
| 冒烟产物 | `/root/rivermind-data/uav_rgb/d4-smoke-20261030`（1 集 / 8 步） |

## 没做什么

- **没有做真实停更注入**（停插件/SIGSTOP）：单测用 FakeSub 注入旧帧/退出；
  真实注入成本高，等 T7 复跑自然覆盖健康路径。
- **contact 只能检测订阅进程退出，不能检测「进程活着但话题不再发布」**：
  contact 是稀疏触发话题，正常飞行 0 msg/s，没有心跳就无法区分停更与无接触。
  这是已知缺口，未在本轮加心跳/发布者存活检查。
- **100 ms 阈值只用了 T5/冒烟的 gap 统计，没有多轮真机标定**；后续若出现误杀，
  先看 `depth_rgb_gap_ns` 分布再调。
- **没有 commit**：用户未授权。
