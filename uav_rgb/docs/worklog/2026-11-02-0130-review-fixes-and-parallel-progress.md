# Week1 Day1 下午：独立审核 + 并行修复 + 环境准备

- 时间：2026-11-02 00:50–01:40
- 方式：4 个并行子代理（审核 1 + 实现 3），文件域互不重叠
- 范围：`sim/tools/`、`src/`、`docs/`、`teacher/`

## 做了什么

### 1. 独立代码审核（只读，9 条 finding）

审核固定到 commit `7c26471`，不预设结论。9 条全部带 file:line 与失败场景。
其中 2 条 P1 我独立复现确认：

| # | 严重性 | 结论 |
| --- | --- | --- |
| 1 | P1 | L1 闸门没验证「同色柱对」和「waypoint 在目标柱外侧」 |
| 2 | P1 | **`l1-v4-cf` 审计实际 12 PASS + 5 FAIL**，与 HANDOFF「17/17」不符 |
| 3 | P2 | `provenance_uniform` 在 base 数据上恒 SKIP |
| 4 | P2 | eval 默认 dispatch，与唯一基线 observation 不一致 |
| 5 | P2 | HANDOFF「shuffle SR>0 就说明读懂指令」推理错误 |
| 6 | P2 | 文档/注释写 mean-pool+concat，实现是 FiLM；param_count 实为 11 |
| 7 | P2 | 深度专家把 +inf 当有效距离 → fail-open |
| 8 | P2 | 部分行缺 `text_ids` 会静默丢弃整个侧车 |
| 9 | P2 | `--task l1` 漏传 `--spawn-mode random` 无参数校验 |

审核同时确认**没有**问题的部分：FiLM 反向梯度、checkpoint v4 读写偏移、
eval 分母与去重键、几何边界（最小间距 3.2297 m）、学生无数据泄漏路径。

### 2. 修复（分 4 次提交）

- `bf99b02` audit：同色柱对 + `waypoint_side`；反事实集 6 项改 SKIP；
  `provenance_uniform` 分字段判定
- `d210158` src：部分缺 `text_ids` 硬报错（带行号）；三处注释与 FiLM 对齐；
  `param_count` 9 → 11
- `5f22992` eval/collect：窗口语义默认统一 observation；l1 强制 random；
  `_strip_min` 用 `math.isfinite`
- `7eedda7` docs：FiLM 描述、shuffle 推断、17/17 说法

### 3. 采集落盘深度（`8a28402`，本轮早前）

`--save-depth`：16 位 PNG，单位毫米，`depth/depth_%05d.png`；
白名单投影抽成 `action_row(r, save_depth)`。关闭时行为完全不变。

### 4. 审计第 18 条 `window_semantics`（`7f357af`，本轮早前）

全一致 PASS / 混用 FAIL / 部分缺 FAIL / 全缺 SKIP（旧数据不误杀）。

### 5. Teacher 脚手架与环境（`b37af93`）

- venv 在 `$DATA_ROOT/uav_rgb/teacher-venv`（torch 2.15.0+cpu）——
  根分区只有 30 G 且本机没有 cuDNN，CUDA 版 torch 起不来
- `student_gru_proto.py`：单目 RGB + 文本 + GRU，随机张量前向/反向自测 PASS
- `dataset.py` 支持 `require_depth=False`，v4 train **404/404 集可用**
- 修了深度单位错位：`png16_mm` 除以 1000 得米（原按 /65535 归一化，差 65 倍），
  新增 `test_depth_units.py`

### 6. 数据清理（用户授权后执行）

`/root/rivermind-data/uav_rgb` 从 **17 G → 11 G**。只删作废的 ds/raw 大文件，
保留全部 `run-*/`、`eval-*/`、`audit.json`：

- `pubdata/airsim10k.zip`（2.0 G，`extract-airsim10k/` 保留作重建源）
- `l1-full-400-0803` 的 ds/train/val/test/logs
- `l1-stop-500-173307` 的 ds*/cf/dec/train/val/test/logs（**6 组 L2 ckpt 全部保留**）
- `l1-pilot-120-0642` 的 ds/train/val/test/logs/tmp
- `l1-smoke-0615`、`l1-pilot-52-0620`、`l1-stop-50-*`、`airsim10k-eval-ckpt*`
- `trash-20261030-2230`

关键产物核验：v4 ckpt、三组 eval.json、L2 六组 ckpt、t6/v3/airsim10k 权重均在。

## 没做什么（含原因）

1. **没重跑 v4 三组闭环**。需要独占 Gazebo 约 4 h（3 组 × 80 min），
   当前机器有并发会话；且需先确认 `--window-from observation` 已是默认（本轮已改）。
2. **没真采一集带深度的数据**。`--save-depth` 只有 fake 订阅覆盖，
   真实 `gz topic` → 解码 → PNG 链路未端到端验证。
3. **没训练 Teacher**。CPU torch 上跑 404 集不现实；GPU 方案需先决定是否
   在 venv 里装 cuDNN（本机系统层故意没有）。
4. **没把 GRU 原型接真实数据**。`dataset.py` 输出 224×224，原型要 120×160，
   resize/裁剪口径未统一。
5. **没跑 `tools/check.sh`**。并行期间它会和其他任务抢 `build/check`；
   改用了 `cmake --build src/build` + 全量 CTest（21/21）与逐个 Python 自测。
6. **没改 `collect.py` 内部函数 `collect_one(window_from="dispatch")` 的默认值**。
   主入口已显式传参，改内部默认只会扩大影响面。

## 验证到哪一步（真实输出）

```
cmake --build src/build        → 成功（33/33）
ctest --test-dir src/build     → 100% tests passed, 0 failed out of 21（134.98 s）
test_audit_dataset.py          → PASS
test_save_depth.py             → PASS
test_collect.py                → 21 项通过
test_eval_closed_loop_args.py  → PASS
teacher.py                     → 全部通过（含「全 inf 深度 -> SAFE_NEUTRAL」）
test_student_gru_proto.py      → PASS
test_depth_units.py            → PASS

audit l1-v4-relation-500       → 19/19 PASS
audit l1-v4-cf                 → 12 PASS / 7 SKIP / 0 FAIL，结论 PASS
df /root/rivermind-data        → 116 G 可用（清理前 110 G）
```

## 风险与遗留

1. **`waypoint_side` 会把 v3 及更早的 L1 数据判 FAIL**（waypoint 朝走廊中心）。
   这是设计意图，但审旧数据集时要知道这是新闸门，不是数据坏了。
2. **混合包（反事实 + 普通）只有代码路径，没有造数据实测**。
3. **深度落盘未在真实仿真验证**；若某版本 gz 的深度不是米，毫米换算会整体偏移。
4. **CPU torch 训练 Teacher 不可行**，GPU 方案未定。
5. 根分区仍只有 30 G（已用 94%），后续装包要放数据盘。
