# 自主执行任务编排（无固定截止，2026-10-30 起）

- 目标：把 T7 baseline 钉死 → 确定执行语义合同 → 建立语言条件 VLN 的最小闭环。
- 原则：**按依赖编排，不设人为时间截止**；同一时刻只跑一个 Gazebo/GPU 任务；
  长任务可跨小时/跨天，每过一个 gate 汇报、提交、推送。
- 模型：**禁止用 DeepSeek Pro**；审核/规划/改代码用 flash 档或当前会话主模型
  （见 personal-skill）。
- 当前状态：T7-R 进行中（`eval-student-r01b`）；旧 SR 0.90 是 R01 前语义；
  `2bfaf57`/`e654d93` 已提交，ALVLN `0499f0f` 已 push。B1/B4/D-GEODUP 已入任务板。

## 1. 资源车道与互斥

| 车道 | 内容 | 规则 |
| --- | --- | --- |
| **G** Gazebo/GPU | 闭环评测、A/B、采集、训练、闭环复评 | 同一时刻只一个；采集/评测期间不跑 CTest/训练/第二个 Gazebo |
| **C** CPU 代码/文档 | L0/L1 代码、audit/preprocess、测试、文档 | 可与 G 并行；**不得编辑 G 正在读取的源文件** |
| **V** 验证 | `bash tools/check.sh`、CTest | 只在 G 空闲时跑 |

### 文件锁（G 在跑时）

- T5b 采集：禁改 `collect.py`、`collect_parallel.py`、`scene_gen.py`、`teacher.py`、`gz_io.py`。
  `collect_parallel` 每个 episode 重新 spawn `collect.py`，编辑会改变后续集行为。
- 训练：禁改 `src/`、`tools/preprocess.py`、数据集文件。
- 闭环评测：禁改 `eval_closed_loop.py`、`collect.py`、`student.py`、`scene_gen.py`、`worlds/*.sdf`。
- 可安全并行：`docs/`、`audit_dataset.py`、`preprocess.py`（采集期间未使用）、
  `src/infer_main.cpp`/`infer.cpp`（采集 teacher 不用 infer）、测试文件、新脚本。

## 2. 任务依赖图

```
T0 T7-R 收尾
  └─ T1 --window-from 开关 + 测试
       └─ T2 A/B（seed 49/57 × observation/dispatch）
            ├─ D1 窗口主因 → T4 冻结 v1 → T5 T5b 全量重采（G，长）
            │                                  ├─ T6 L0 实现（C，并行，避开文件锁）
            │                                  └─ T7 T5b 打包+训练+复评（G）
            │                                       └─ D3 新合同是否采纳
            └─ D2 不确定 → T3b 方差/资源核查 → 再决定 T4/T5
T8 L0 测试 + check.sh（G 空闲时）
T9 L1 场景/指令代码（T5b 结束后，改 scene_gen/collect）
T10 L1 pilot 50 集 + audit（G）
T11 L2 文本编码+融合+消融（G/C，多日）
T12 每个 gate：文档 + commit + push
```

## 3. 各任务详情

### T0（G）T7-R 收尾

- 等 `eval-student-r01b/eval.json`；若进程退出无 JSON，换新目录重跑，不覆盖。
- 对比旧 `eval-student/eval.json`：SR/collision/stuck/false_forward/动作分布/
  p50/p95/failed。
- 逐集失败分类（`.work/ep_<seed>/actions.jsonl` + `oracle.jsonl`）：
  早碰撞（step<20 且 max_x<3）、过线碰撞（max_x>goal_x 但 collision）、
  SAFE 悬停（SAFE>50% 且 max_x<goal_x）、false_forward。
- 写 `docs/experiments/2026-10-31-t7-r01b.md`；更新 HANDOFF §4.1/§4.3、TASKS T7-R。
- 注意：seed 49/57 是近重复几何（差 5 mm），动作逐位相同；结论里写明
  20 seed 的独立场景数被高估（D-GEODUP）。

### T1（C）`--window-from` 诊断开关

- `sim/tools/collect.py`：`--window-from {dispatch,observation}`，默认 `dispatch`
  （R01 语义）。observation 时 `action_start_ns = pose_before_ts`；写进 manifest。
- `sim/tools/eval_closed_loop.py`：透传参数。
- `sim/tools/test_collect.py`：两种窗口的假时钟回归；默认必须是 dispatch。
- 验收：`test_collect.py` PASS；不改标签口径（标签始终用观测位姿）。

### T2（G）A/B

```bash
BASE=/root/rivermind-data/uav_rgb/t6-full-1640
for wf in observation dispatch; do
  env DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg TMPDIR="$BASE/tmp" \
    "$UAV_RGB_PYTHON" sim/tools/eval_closed_loop.py --policy student \
    --ckpt "$BASE/run/ckpt_best.bin" --steps 80 --hold 1.0 --split test \
    --seeds 49,57 --window-from "$wf" --keep-work \
    --out "$BASE/ab-$wf" > "$BASE/ab-$wf.log" 2>&1
done
```

- **D1 窗口主因**：observation 2/2 成功、dispatch 0/2 成功（或差距 ≥2 集）。
- **D2 不确定**：两者相同 → T3b。

### T3b（G）不确定时的方差/资源核查

- seed 49 用 dispatch 连跑 3 次（每次换新 out），记录 SR/max_x/动作序列。
- 查 `.work` 的 RTF、`eval.json` 的 p50/p95；p95>200 ms 或 RTF<0.2 → 资源竞争。
- 若方差大：结论写「20 seed 单次无法区分」；改用 3 次重复/50 seed；
  用 observation 语义重跑完整 20 seed，建立当前环境的 v1 基线。
- 不改模型；修环境后重做 T2。

### T4（C）冻结 baseline 合同

- 若 D1 成立：**v1 baseline = observation-window 语义**（历史 0.90），
  dispatch-window 是新的执行合同，需重采重训；两者数字永不混报。
- 更新 HANDOFF/CLAUDE/TASKS/EVAL_STANDARD；写明 p50/p95 与窗口语义。
- 若 T3b 发现方差大：v1 基线改为「当前环境 observation 语义 3 次重复的均值」。

### T5（G，长）T5b 全量重采（dispatch 语义）

```bash
BASE5=/root/rivermind-data/uav_rgb/t5b-dispatch-20261031
mkdir -p "$BASE5"
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg setsid nohup \
  "$UAV_RGB_PYTHON" sim/tools/collect_parallel.py \
  --instances 4 --episodes-per-instance 150 --steps 40 --hold 1.0 \
  --base-seed 200000 --out "$BASE5/collection" \
  > "$BASE5/collection.log" 2>&1 < /dev/null &
```

- 与 T5 同几何/seeds，只改执行窗口语义；**不覆盖** `t5-full-1155`。
- 采集期间只做 T6（C），遵守文件锁。
- 验收：599 集左右、`audit_dataset.py` 全 PASS、几何 split 与 T5 一致。

### T6（C，与 T5 并行）L0 指令数据通路

只改采集期间未被使用的文件：

- `tools/preprocess.py`：读每集 `instruction`，写 `<out>/instructions.jsonl`
  （`episode_id, instruction, sample_start, sample_count`）；缺/多/错失败。
- `sim/tools/audit_dataset.py`：`instruction_coverage`：每集非空、无坐标泄漏、
  与 scene 目标色一致（L1 字段）；缺字段 SKIP/FAIL 规则写清。
- `src/infer_main.cpp` / `infer.cpp`：加 `--text <id,id,...>` CSV（单集一条；
  多指令再升 TEXT 头）；L0 只解析/校验，文本向量先置零；`student.py` 发送；
  `src/tests/test_infer_io.cpp` 加带 text_ids 的回归；`test_collect.py` 加字段回归。
- **`collect.py` 的 instruction 字段等 T5b 结束后再改**（文件锁）。
- 验收：`test_audit_dataset`/`test_collect`/`test_infer_io` PASS；
  G 空闲后 `bash tools/check.sh` 全量 PASS。

### T7（G/GPU）T5b 打包 + 训练 + 复评

- 按 `docs/experiments/2026-10-30-t6-train.md` 的命令，把数据目录换成
  `$BASE5/collection`，打包到 `$BASE5/ds`，训练 10 轮到 `$BASE5/run`，
  eval-only 评 best，闭环 20 seed 到 `$BASE5/eval-student`。
- 避免覆盖：所有输出新目录；训练失败用 `--resume` 同目录或新目录。
- **D3**：T5b 闭环 SR ≥ v1 − 5 pp → 采纳 dispatch 为新的训练/评测合同；
  否则先查失败模式，不急着改模型。

### T8（V）L0 验证

- G 空闲后 `bash tools/check.sh`；CTest 20/20 + Python/UI 全过。
- 提交前 `git diff --check`；按主题提交；同步 ALVLN + push。

### T9（C）L1 场景/指令代码（T5b 结束后）

- `scene_gen.py`：加 `color` 字段与 2 根彩色柱（红/蓝），同一场景 ≥2 根彩色柱；
  目标由 instruction 决定，颜色随机且目标不总是最近。
- `collect.py`：按目标色生成 1 个模板 `fly to the {color} pillar and stop`，
  写入 manifest；`actions.jsonl` 白名单同步。
- `audit_dataset.py`：目标色存在、指令与目标一致、无坐标泄漏、同场景反事实对可生成。
- 自测 + `scene_gen --self-check`；不改 T5b 已采数据。

### T10（G）L1 pilot

- 50 集（4 实例，约 25 min）到新目录；`audit_dataset.py` 全 PASS。
- 不训练；记录类别/指令分布；若失败，先修数据设计。

### T11（G/C）L2（多日）

- 文本编码（embedding+mean-pool 或 tiny GRU，CPU 侧）+ FiLM/concat 融合；
  4 动作；<1M 参数；训练 + 闭环 + 文本置空/打乱/同图换指令三组消融。
- 阈值先冻结（建议 +10pp / −10pp / 一致率 ≥60%）。
- 不达标先改任务设计，不开 ECPT；ECPT 仍是条件项。

## 4. 长任务恢复

- 评测/采集死：看日志尾 + `ps` 残留 `gz sim`/`ruby`；用 setsid 的 PGID kill，
  禁止 `pkill -f 'gz sim'`；换新目录重跑，不覆盖。
- 采集中断：按 seed 范围补采缺失集；不往旧目录追加半成品。
- 训练中断：同目录 `--resume`，或新目录；不覆盖半成品。
- 磁盘：数据盘 <5 G 时停止新采集；只清本次临时目录，不删用户数据。

## 5. 提交 / 推送

- 每个 gate：`git status --short` → `git diff --check` → 精确 `git add` →
  中文提交主题 + `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- 测试通过才提交；失败实验只写文档，不提交半成品。
- push：rsync `--dry-run` → rsync → ALVLN `git add uav_rgb` → commit → push →
  `git status --short --branch` + `git ls-remote --heads origin` 核对。
- 不 `--force`；不提交数据/权重/日志/密钥。

## 6. 汇报节奏

- 每个 gate 写一段状态：数字、证据路径、下一步、风险。
- 不设「几点必须停」；只在 G 任务未结束时避免再开第二个 G 任务。
- 用户回来后先看 `docs/worklog/2026-10-31-*.md` 和 `docs/experiments/`。
