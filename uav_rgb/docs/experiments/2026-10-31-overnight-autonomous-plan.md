# 夜间自主执行计划（2026-10-30 23:55 – 2026-10-31 07:30）

- 目标：在用户离开的 ~7 小时内，自主完成 T7-R 定稿、执行语义 A/B、决定 baseline
  合同，并尽可能推进 L0（指令数据通路）；所有实验与代码改动留痕、提交、推送。
- 约束：同一时刻只允许一个 Gazebo 任务；采集/评测期间不跑 CTest/训练；
  长任务用 `setsid` + `--keep-work`；产物只写 `$DATA_ROOT`；不删用户数据。
- 当前状态：T7-R 进行中（`eval-student-r01b`）；旧版 SR 0.90 是 R01 前语义；
  已提交 `2bfaf57` 并 push（ALVLN `d25e242`）。B1/B4/D-GEODUP 已写入 TASKS。

## 0. 时间线（决策树）

| 时间 | 动作 | 产物/判据 |
| --- | --- | --- |
| 00:00–00:30 | T7-R 收尾 + 失败分类 | `eval.json`、`docs/experiments/2026-10-31-t7-r01b.md` |
| 00:30–01:00 | 实现 `--window-from` 诊断开关 + 2 seed A/B | `ab-observation` / `ab-dispatch` 两个 eval |
| 01:00–01:15 | 判定 R01 窗口是否为主因 | 见 §2 判据 |
| 01:15–05:30 | 分支 A：冻结 v1 + L0；或分支 B：重采/方差 | 见 §3 |
| 05:30–06:30 | L0 测试/文档；若重采完成则打包+训练 | `tools/check.sh`、train run |
| 06:30–07:30 | 评测（若模型已出）/ 汇总 / commit+push | 早间报告 |

**硬止损**：06:30 之后不再启动新的 Gazebo/训练任务；已在跑的用 `setsid` 留着并写状态。

## 1. Phase 0：T7-R 收尾（≤30 min）

1. 等 `eval-student-r01b/eval.json` 出现；若进程退出但无 JSON，按 HANDOFF §4.1
   的 fallback 换新目录重跑，不覆盖。
2. 对比旧 `eval-student/eval.json`：SR / collision / stuck / false_forward /
   action_distribution / inference p50/p95 / failed 数。
3. 逐集失败分类（`.work/ep_<seed>/actions.jsonl` + `oracle.jsonl`）：
   - 早碰撞：collision step < 20 且 max_x < 3；
   - 过线碰撞：max_x > goal_x 但 collision；
   - SAFE 悬停：SAFE 步占比 > 50% 且 max_x < goal_x；
   - false_forward：碰撞步 action_executed=1 且 src 含 contact。
4. 写 `docs/experiments/2026-10-31-t7-r01b.md`（命令、环境、逐集表、结论），
   更新 HANDOFF §4.1/§4.3、TASKS T7-R。**0.90 与本次数字都标明适用语义版本。**

## 2. Phase 1：执行语义 A/B（30 min）

### 2.1 代码改动（最小）
- `sim/tools/collect.py`：新增 `--window-from {dispatch,observation}`，默认
  `dispatch`（当前 R01 语义）。`observation` 时 `action_start_ns = pose_before_ts`；
  写进 manifest 便于追溯。
- `sim/tools/eval_closed_loop.py`：透传 `--window-from` 给 collect。
- `sim/tools/test_collect.py`：加两种窗口的假时钟回归；默认必须是 dispatch。
- 不改 `student.py`；不改动作窗口的标签口径（标签始终用观测位姿）。

### 2.2 运行（顺序，不并发）
```bash
BASE=/root/rivermind-data/uav_rgb/t6-full-1640
for wf in observation dispatch; do
  setsid nohup env DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg TMPDIR="$BASE/tmp" \
    "$UAV_RGB_PYTHON" sim/tools/eval_closed_loop.py --policy student \
    --ckpt "$BASE/run/ckpt_best.bin" --steps 80 --hold 1.0 --split test \
    --seeds 49,57 --window-from "$wf" --keep-work \
    --out "$BASE/ab-$wf" > "$BASE/ab-$wf.log" 2>&1 < /dev/null &
  wait
done
```

### 2.3 判据
- **窗口主因**：observation 2/2 成功、dispatch 0/2 成功（或差距 ≥2 集）。
- **不确定**：两者相同；进入 Phase 2B（方差/资源核查）。
- **dispatch 反而更好**：T7-R 的低 SR 可能是资源竞争/随机性；复跑 3 次 seed 49
  量化方差后再决定。

## 3. Phase 2：分支

### 3A（推荐）窗口主因 → 冻结 v1 baseline + L0
1. 文档：定义 **v1 baseline = observation-window 语义**（旧 0.90），
   dispatch-window 是**新执行合同**，需要重采重训，列为 P1，不在今晚阻塞。
   HANDOFF/TASKS/CLAUDE 同步写清；禁止把 0.90 与 dispatch 数字混报。
2. 立即实现 **L0**（预计 3–4 h，纯 CPU，可与后台采集并行）：
   - `collect.py`：episode manifest 增加 `instruction`（L1 才生成；L0 允许空串）。
   - `tools/preprocess.py`：读每集 instruction，写 `<out>/instructions.jsonl`
     （`episode_id, instruction, sample_start, sample_count`）；缺/多/错必须失败。
   - `audit_dataset.py`：新增 `instruction_coverage`：每集非空、无坐标泄漏、
     与 scene 目标色一致（L1 字段）；缺字段 SKIP/FAIL 规则写清。
   - `src/infer_main.cpp` / `infer.cpp`：协议增加 `TEXT <len> <id...>` 头；
     L0 只解析并校验，文本向量先置零；`student.py` 发送；
     `src/tests/test_infer_io.cpp` 加合法/畸形头用例。
   - `tools/check.sh`：把新测试接入；全量 PASS。
3. 若 3A 开始后仍有时间（≥03:30），**后台启动** T5b 重采（dispatch 语义）：
   ```bash
   DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg setsid nohup \
     "$UAV_RGB_PYTHON" sim/tools/collect_parallel.py \
     --instances 4 --episodes-per-instance 150 --steps 40 --hold 1.0 \
     --base-seed 200000 --out "$DATA_ROOT/uav_rgb/t5b-dispatch-<date>" \
     > "$DATA_ROOT/uav_rgb/t5b-dispatch-<date>.log" 2>&1 < /dev/null &
   ```
   599 集约 4.5 h；**采集期间只做 L0 文档/代码，不跑 CTest/训练**。
   **采集期间禁止编辑 `collect.py`/`collect_parallel.py`/`scene_gen.py`/`student.py`**：
   `collect_parallel` 每个 episode 重新 spawn `collect.py`，编辑会改变后续集行为。
   只改 `preprocess.py`/`audit_dataset.py`/`infer_main.cpp`/`test_*`/docs；
   `collect.py` 的 instruction 字段等 T5b 结束后再改。
   若 06:00 前采完：preprocess → 训练 10 轮（约 1.8 h，可能超出 7 h）→
   用 `setsid` 留着，早间报告状态，不强行等完。

### 3B 不确定 → 方差/资源核查（≤1.5 h）
1. seed 49 用 dispatch 连跑 3 次（3×2 min），记录 SR/max_x/动作序列；
   若 3 次结果分叉，说明 Gazebo 随机性/资源竞争是主因。
2. 查 `.work/ep_49/../logs` 的 RTF、`eval.json inference_ms p50/p95`；
   p95 > 200 ms 或 RTF < 0.2 → 资源竞争，重跑前清 `gz sim`/`ruby` 残留。
3. 若方差大：T7 结论写「20 seed 单次不可区分 0.9 vs 0.5」，
   建议后续固定 3 次重复或扩到 50 seed；不改模型。
4. 剩余时间做 L0（同 3A 第 2 步）。

## 4. Phase 3：L1 pilot（仅当 L0 完成且 baseline 已定，最多 1 h）

- `scene_gen.py`：加 `color` 字段与 2 根彩色柱（红/蓝），保证同一场景 ≥2 根
  彩色柱；目标由 instruction 决定，目标颜色随机且不总是最近。
- `collect.py`：按目标色生成 1 个模板指令
  `fly to the {color} pillar and stop`，写入 manifest。
- `audit_dataset.py`：目标色存在、指令与目标一致、无坐标泄漏、同场景反事实对可生成。
- 跑 50 集 pilot（4 实例，约 25 min），audit 全 PASS；**不训练**。
- 若时间不足，只提交代码 + 自测，pilot 留给下一天。

## 5. 提交 / 推送规则

- 每个阶段结束：`git status --short` → `git diff --check` → 精确 `git add <path>`
  → 中文提交主题 + `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- 测试通过才提交；失败实验只写文档，不提交半成品。
- push 走 `ALVLN` 镜像：rsync `--dry-run` → rsync → `git add uav_rgb` →
  commit → `git push origin main` → `git status --short --branch` +
  `git ls-remote --heads origin` 核对。
- 不 `--force`；不提交数据/权重/日志/密钥。

## 6. 止损与回滚

- 评测进程死：看日志尾 + `ps` 残留；换新目录重跑，不覆盖。
- 磁盘：数据盘 < 5 G 时停止新采集；不删用户数据，只清本次临时目录。
- Gazebo 残留：`ps -eo pid,comm` 找 `ruby`/`gz sim`，用 `setsid` 的 PGID kill；
  禁止 `pkill -f 'gz sim'`（会杀自己）。
- `check.sh` 失败：修或回退本次改动，不提交失败状态。
- 06:30 后不启动新长任务；已在跑的保留并写状态。

## 7. 早间交付（07:30 前写进 `docs/worklog/2026-10-31-0730-overnight.md`）

1. T7-R 最终数字 + 失败分类 + 与旧 0.90 的差异解释。
2. A/B 结果与 baseline 合同决定（v1 observation vs 新 dispatch）。
3. L0 完成度：改了哪些文件、测试结果、未做什么。
4. 若启动了 T5b 重采/训练：进度、产物路径、预计完成时间。
5. 提交/推送哈希；下一步建议。
