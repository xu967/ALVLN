# 场景升级：同色柱对 + 关系词指令（几何 v5 / l1.v3）

- 时间：2026-11-01 04:20–05:00
- 范围：`sim/tools/`（landmarks / scene_gen / collect / audit_dataset /
  eval_closed_loop / make_counterfactual / make_decisive / 三个测试）、
  `src/`（text.hpp、test_text_branch）、`tools/preprocess.py`、
  `sim/README.md`、`docs/EVAL_STANDARD.md`、`docs/HANDOFF.md`
- 结论：完成（代码与文档）；**未重采、未重训、未跑闭环**

## 为什么改

上一轮闭环（49 个 test 任务）SR 34.7%、碰撞 30.6%、false_forward 22.4%，而
离线 flip 只有 45.5%，说明文本仍然没进决策。根因是任务本身：
**两根目标柱异色，图像里目标色唯一，"看图选颜色"就够了**，模型完全有理由
忽略指令（实测换词只改变 2e-4 概率、argmax 0 次变化）。

改成两根**同色**柱后，图像只给出「前方有两个候选」，选哪根只能靠指令里的
关系词。这就是语言「非用不可」的最小场景。

## 做了什么

- `sim/tools/landmarks.py`：`assign_landmarks` 重写。选 **两根同色** 柱，
  必须分居出生时刻机头轴线两侧（`LATERAL_MIN=0.3`）且横向跨距 ≤2.4 m
  （`LATERAL_SPAN_MAX`，60° 视场装得下的上限）；`relation` 由 seed 奇偶决定
  保证 left/right 平衡；新增公开函数 `lateral(p, spawn)`（按 spawn 的 yaw 折算
  机身左向偏移）。**选不出合格柱对时直接抛 ValueError**，不静默降级。
- `sim/tools/scene_gen.py`：`GEOMETRY_VERSION` v4 → **v5**；`task_version`
  `l1.v2` → **`l1.v3`**；meta 的 `distractor_color` 换成 `relation`。
- `sim/tools/collect.py`：episode manifest 增加 `relation` 字段。
- `sim/tools/audit_dataset.py`：读 `scene.relation`；`landmark_balance`
  （目标最近/非最近）改成 **`relation_balance`**——同色下两根柱都是目标色，
  旧检查恒为 PASS 假绿；新检查要求两根彩色柱真的分居左右且 left/right 都有。
  `target_color` 检查同时核对指令里的关系词与 scene 一致。
- `sim/tools/eval_closed_loop.py`：`override_instruction` 的 shuffle 从
  「换颜色」改成 **left↔right 取反**；`dedup_key` 加入 `relation`——同色柱对的
  left/right 两个任务彩色柱完全相同，不加这一项会被去重成一个。
- `sim/tools/make_counterfactual.py`、`make_decisive.py`：`alt_target` 返回
  （反向关系词，另一根柱的 waypoint）；反事实里 `target_color` 不变，只有
  `relation`/`instruction`/`waypoint` 变。`alt_target` 顺带改用真正的
  `waypoint_for`（旧版把柱子中心当 waypoint，和原集不一致）。
- `src/include/uavrgb/text.hpp`：`kTextLen` 8 → **9**（关系模板正好 9 token）。
- `tools/preprocess.py`：`TEXT_LEN` 8 → **9**。
- `src/tests/test_text_branch.cpp`：「9 token 必须被拒」改成「10 token 必须被拒」，
  夹具换成 10 token 的指令。
- 测试：`test_landmarks.py` 重写（同色、分居左右、关系词一致、左右平衡、
  default 出生必须报错）；`test_eval_closed_loop_args.py` 改成同色柱对 +
  关系词消融；`test_make_decisive.py` 夹具改成同色柱对并断言 relation 互换。
- 文档：`sim/README.md` 出生位姿段（顺带删掉已废弃的 safe 模式说明）、
  `docs/EVAL_STANDARD.md` §5 消融与关系词定义、`docs/HANDOFF.md` 几何常量表 +
  L1 段落 + S 组说明。

## 没做什么

- **没重采、没重训、没跑闭环。** 只改生成路径和口径，数据要下一轮采。
  - 原因：重采 500 集约 3 小时、训练与 49 任务闭环另算；这轮先把任务定义和
    门禁改对，否则采了也是同一批学不到文本的数据。等用户决定什么时候开跑。
- **没改相机视场（仍是 60° 水平）。** 关系任务靠的是「出生在 y≈±0.8 的缝里，
  左右两根候选各偏 0.8 m」，这个几何在 60° 下够用（实测 4 个 seed 的两根柱
  分别落在 ±21°、±15°）。改成 90° 会更宽松，但会同时改变观测分布，
  和「任务升级」混在一起就说不清是哪个起的作用。
  - 代价：`--task l1` 从此**必须配 `--spawn-mode random`**，default（y=0）会报错。
- 没动 `traverse_rate.py`、`_spawn_collides`、C++ 模型结构（`film` 的形状由
  `kTextLen` 推导，代码本身不用改）。
- 没改历史 worklog / experiments 里的记录（T2、T5、l1-pilot、l2-film-finetune）。
- 没提交、没推送：用户未授权。

## 待确认

- **左右两个目标的难度不对称**：目标是侧车道柱时，waypoint 恰好落在出生点的
  横向延长线上，几乎直飞；目标是门柱时，waypoint 在另一侧 1.6 m 外。seed 奇偶
  保证两类各半，但没有做等难度设计。如果这轮训完发现「直飞那半边」学得快、
  「横移那半边」学不会，说明是任务难度问题而不是文本问题。
- **`LATERAL_SPAN_MAX=2.4` 是手算的**：60° 水平视场、出生点距彩色柱约 2 m，
  单侧偏移 ≤1.2 m。没有用实际渲染的像素验证过「两根柱真的同时在画面里」——
  要等重采后抽帧看。
- 门禁口径改了（`landmark_balance` → `relation_balance`），如果还有别的脚本或
  文档按旧名字读 `audit.json`，需要一起改；本轮只查了本仓库。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python -m py_compile`（10 个改动文件） | PASS |
| `sim/tools/test_landmarks.py` | PASS（20 seeds：同色、分居左右、关系词一致、左右各半；单柱与 y=0 出生都抛 ValueError） |
| `sim/tools/test_instruction.py` | PASS |
| `sim/tools/test_eval_closed_loop_args.py` | PASS（left/right 去重键不同、shuffle 取反、缺 relation 报错） |
| `sim/tools/test_make_decisive.py` | PASS（反事实 relation 互换、waypoint 换到另一根柱） |
| `sim/tools/scene_gen.py --self-check` | PASS（100 seeds 字节级一致） |
| `bash tools/check.sh` | **PASS**（21/21 CTest，含 `test_text_branch` 按 kTextLen=9 重跑；全部 Python 自检） |
| `scene_gen --task l1 --spawn-mode random` 实跑 | PASS，seed 7 出 `l1.v3 / relation=right / 'pass the blue pillar on your right and stop' / geom.v5` |
| `scene_gen --task l1`（default 出生） | 按设计报错：`出生点 y=0.00 处找不到分居左右、跨距 ≤2.4 m 的两根柱` |
| 旧文本 ckpt 能否加载 | **不能**：`checkpoint: param[9]: 形状与模型不符`，rc=1（干净报错，不是静默加载） |
| 旧纯视觉 ckpt 能否加载 | 能：`text_dim=0` 的 ckpt 正常加载（`film` 尺寸与 kTextLen 无关） |
| 重采 / 重训 / 闭环 | 未跑（见「没做什么」） |

## 风险与遗留

- **所有 `text_dim > 0` 的 checkpoint 全部作废**：`film.W` 从 `[8*16, 2*c3.out]`
  变成 `[9*16, 2*c3.out]`，加载时报形状不符。这是有意的（任务换了，权重本来
  也不该复用），但意味着离线 flip / 闭环基线必须重训才有数。
  纯视觉 checkpoint（`text_dim=0`，如 T6 的 `t2-v3c`）不受影响，仍可直接复评。
- **v4 及更早的数据集不能与 v5 混包**：`GEOMETRY_VERSION` 进 `SPLIT_SALT`，
  `preprocess`/`audit` 会按 `geometry_version` 拒绝。已打好的 `ds/`、`ds-cf/`、
  `ds-dec/` 仍可训练/评测旧模型，但不能和新数据合。
- 60° 视场这个前提没有渲染验证（见「待确认」）。如果重采后抽帧发现两根柱不是
  同时可见，那就得回到「扩视场」或「缩小 `LATERAL_SPAN_MAX`」，两者都要再
  bump 一次几何版本。
- 「提前停车 34.7%」那个假设本轮没有触碰，仍然未验证。

## 更新 2026-11-01 05:20：加视场门禁（冒烟后发现）

重采前的 4 集冒烟暴露出一个此前只做了手算的问题：两根目标柱虽然都在画面里，
但贴在左右边缘；按 500 个 seed 实测，**15 个（3%）最外侧的柱心水平角超过 30°**
（最大 32.2°），那一根只剩一条边在画幅内。

- `sim/tools/landmarks.py`：新增 `h_angle(p, spawn)`（柱心相对相机光轴的水平偏角，
  相机按 SDF 的 `<pose>0.2 0 0</pose>` 前移 0.2 m、按 spawn yaw 折算）与常量
  `HALF_FOV_DEG=30` / `FOV_MARGIN_DEG=2` / `CAMERA_FORWARD_M=0.2`。
  `assign_landmarks` 在候选柱对里**优先选两根都 ≤28°** 的；一组都不满足时退回
  全部候选（不提高失败率）。
- 复测 500 个 seed：最外侧柱心角 p50=20.9° / p99=27.9° / **max 27.9°，超过 30° 的
  0/500**（改前 15/500）。

| 项 | 结果 |
| --- | --- |
| `test_landmarks.py` | PASS |
| `bash tools/check.sh --python-only` | PASS |
| 冒烟 4 集 / 8 集 / 20 集实采 | 全部 `stopped_on_goal=true`；20 集 315 步分布 SAFE 19.4% / FORWARD 69.5% / YAW_L 5.7% / YAW_R 5.4%，left 10 / right 10 |
| 8 集冒烟的 `class_share` FAIL | 小样本假阳性（val/test 只有 3–4 集，整类缺席）；20 集样本该条已 PASS |
