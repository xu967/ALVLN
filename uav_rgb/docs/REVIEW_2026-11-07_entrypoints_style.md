# 入口脚本「可读性 / 改超参数便利性」审查（uav_rgb，HEAD 9d764d9 快照）

- 时间：2026-11-06 22:30
- 审查对象：`/root/rivermind-data/uav_rgb/tmp/wt-baseline-20261106/`（干净 worktree，HEAD 9d764d9）。下文 `file:line` 都相对这个快照；主仓库正在被 4 个 agent 改，行号可能已漂移。
- 方法：只读；跑了各入口的 `--help`（输出留在 `/root/rivermind-data/uav_rgb/tmp/agent-style/help_*.txt`）；用 ast 统计函数长度；用 `split_manifest.split_for_seed` 核对了 20 个冻结 seed 的 split。没有起 Gazebo、没有用 GPU、没有改任何仓库文件。
- 评分依据：`/root/.claude/skills/personal-skill/SKILL.md` §6.10 七条；样板 `/root/rivermind-data/YOLOVOmodal_3871538535/train.py`。
- 没看的部分放在文末。

## 结论摘要

1. 现状：跑一套「采 300 集 / 40 步 / hold 1.0 → 训 30 轮 batch 128 lr 1e-3 → 评 20 seed」要敲 5 个脚本、8 条命令、约 45 个 flag 参数 + 3 个环境变量；hold 要打 5 次、steps 5 次、seed 列表 4 次（§1.3）。同一个量的重复定义：hold 9 处、动作枚举 11 份、巡航高度 4 个值 5 处（§1.4）。
2. 对照 §6.10：库级常量区（`collect.py:85-104`、`scene_gen.py:51-103`、`eval_closed_loop.py:43-47`）已是样板水平；差距在用户真正要改的「实验级」参数——集数、步数、hold、轮数、batch、seed 列表——没有一处集中，4 个主入口（`collect_parallel.py`、`eval_closed_loop.py`、3DGS 两个脚本、`train_ppo.py`）在「①集中」上不达标，默认值散在 argparse、别的文件、函数签名和函数体里（§2）。
3. 最值得先改的三个：`collect_parallel.py`（~45 行，§3.1）、`eval_closed_loop.py`（~40 行，§3.9）、`collect.py` + `teacher.py`（~25 + ~15 行，§3.2/§3.12）。全部是「默认值搬到文件顶 + `default=CFG[...]`」，flag 名不变、`sim/tools/test_*.py` 不动，合计约 190 行（§3.14）。
4. `tools/e1_pipeline.py` 草稿 299 行已在 §4.2，`py_compile` + `--dry-run` 通过；跑之前有 10 个待确认点（§4.4），最硬的三个：`--vehicle e1_mini` / `--dynamics cf_sitl` flag 是否已存在、cf_sitl 要不要显式 `--template`、`audit_dataset.py:604` 对 cf_sitl 的 `forward_motion` 误判。
5. 用户给的 20 个「冻结 test seed」（`HANDOFF.md:915`）在当前 `uavrgb.geom.v5` salt 下 17 个是 train，不能直接用；草稿改为第一组扫 test split、其余三组复用 `attempted_seeds`。
6. 会导致改错参数的可读性问题：5 个 300+ 行的函数、`scene_gen.py:16-18` 过期的「默认 80 步」注释、`HANDOFF.md:898` 漏写 `UAVRGB_RENDER_BACKEND=software`、`smoke_campus.py:28-30` 已经跑不起来（§5）。

## 0. 样板 `train.py` 好在哪（107 行）

| 样板里的做法 | 位置 |
| --- | --- |
| 超参数是字面量，直接写在调用处：`imgsz=640` `workers=8` `lr0=1e-3` `close_mosaic=10` | train.py:68-78、89-105 |
| 「不显然」的值行尾/行上一句为什么：batch 32 ≈ 10–11 GiB 显存、cache 默认关是因为 41 GiB 缓存 ×1.5 安全系数超过 cgroup 58 GiB 会被静默拒绝 | train.py:69、72-76 |
| 环境变量可覆盖，但默认值就写在文件里：`os.environ.setdefault("YOLOVO_EPOCHS", "100")`；覆盖写法一行放注释里：`YOLOVO_BATCH=16 python train.py`、`YOLOVO_CACHE=1 python train.py` | train.py:44-47、69、76 |
| 运行方式一行：`python train.py v2`；想跑旧行为：`YOLOVO_RELATE_DATA= python train.py` | train.py:41-42 |
| 留 1–2 行注释掉的备选项当提示：`# lrf=0.01,` `# optimize='AdamW',` `# fraction=0.4,` | train.py:96-98、104 |
| 从上到下就是流程：环境修补 → 默认配置 → 选字段 → import → 训练调用；没有配置类、没有 YAML、没有 argparse | 全文 |
| 失败教训直接写进注释（TTY 0x0 让 tqdm 不渲染；cuDNN 必须在 import torch 前进 LD_LIBRARY_PATH） | train.py:4-10、15-37 |

一句话：打开文件 30 秒内知道改哪一行、改成什么、为什么现在是这个值。

## 1. 现状表

### 1.1 用户实际要敲的命令（从 `docs/HANDOFF.md` §十二、`README.md`、`sim/README.md`、`docs/HANDOFF_3DGS_20261105.md`、`tools/check.sh`、各脚本 docstring 抓的）

| # | 命令 | 出处 |
| --- | --- | --- |
| 1 | `bash tools/check.sh [--python-only]` | HANDOFF.md:886-887 |
| 2 | `"$UAV_RGB_PYTHON" sim/tools/collect_parallel.py --instances 4 --episodes-per-instance 1 --steps 40 --hold 1.0 --base-seed 0 --out <dir>`（v4 主线再加 `--task l1 --spawn-mode random --window-from observation`） | HANDOFF.md:901-902、364-366；README.md:44-46 |
| 3 | `sim/tools/audit_dataset.py --dataset <dir>` | HANDOFF.md:910、368 |
| 4 | `tools/preprocess.py [--drop-collision] --out <ds> <dir>/{train,val,test}/*/` | HANDOFF.md:905、371-372；README.md:47-48 |
| 5 | `uavrgb_train --data <ds>/dataset.uavr --epochs 10 --batch 128 --lr 1e-3 --class-weights auto --out <run>`（v4：`--text-dim 16 --epochs 20 --batch 64 --lr 5e-4 --seed 1234 --drop-collision`） | HANDOFF.md:906-907、379-381；README.md:49-52 |
| 6 | `uavrgb_train --data ... --eval-only <ckpt> --split test --batch 128 --out <dir>` | HANDOFF.md:384-386 |
| 7 | `sim/tools/eval_closed_loop.py --policy student --ckpt <ckpt> --steps 80 --hold 1.0 --split test --seeds 23,25,…,180 --out <dir>`；盲前进改 `--policy forward` | HANDOFF.md:913-917 |
| 8 | `tools/train_student_with_video.py --data <uavr> --out <run> [uavrgb_train 参数]` | train_student_with_video.py:4-7 |
| 9 | `python -m sim.3dgs.collect_3dgs --scene <scene.json> --out <dir> --steps 8 --policy teacher` | HANDOFF_3DGS_20261105.md:97 |
| 10 | `python -m sim.3dgs.eval_3dgs --scene ... --out ... --ckpt ... --episodes 1 --steps 20` | HANDOFF_3DGS_20261105.md:115 |
| 11 | `sim/tools/scene_gen.py --seed 42 --out <sdf>` / `--self-check` | sim/README.md:201-202 |
| 12 | `teacher/train.py --data <dir> --epochs 30 --batch 8 --lr 1e-3 --seq-len 8 --out <run> --device cuda` | teacher/train.py:4-6 |
| 13 | `python -m rl_teacher.train_ppo --out <path> [--steps 100000 --n-envs 4]` | rl_teacher/train_ppo.py:62 |
| 14 | `tools/train_students.sh --data ... --out ... --groups base,a0.5 -- --text-dim 16 --epochs 20` | train_students.sh:7-9 |
| 15 | `ui/launch_studio.sh`（内部走 `ui/trainer_native.py --config`） | README.md:70 |

注意两处文档已经和代码对不上（用户照抄会错）：
- HANDOFF.md:900「并行采集（必须 DISPLAY=:20）」：`collect_parallel.gz_env` 默认 `UAVRGB_RENDER_BACKEND=gpu`（collect_parallel.py:93）并且会**删掉** `DISPLAY`（:106-108）；本容器 gpu 后端段错误（BRIEF §2），正确的是 `UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg`。
- HANDOFF.md:906、384 用 `src/build/uavrgb_train`，README.md:49、check.sh:6 用 `build/check/`；`train_student_with_video.py:218` 默认 `src/build`，`student.py:22-23`、`trainer_native.py:20-23`、`train_students.sh:16` 默认 `build/check`。同一台机器上两个构建目录、两套默认。

### 1.2 每个入口的超参数散在哪

| 入口（行数） | argparse flag 数 | 环境变量 | JSON / 别的文件 | 文件内常量 | 「看不见」的默认值（埋在别处） |
| --- | --- | --- | --- | --- | --- |
| `sim/tools/collect_parallel.py`（781） | 23（:450-488） | `UAVRGB_RENDER_BACKEND`（默认 `"gpu"` 埋在 `gz_env` :93，不在顶部）、`UAVRGB_XDG_RUNTIME_DIR`（:52）、`GZ_SIM_RESOURCE_PATH`（:100）、`DISPLAY`（:115） | 透传 8 个 flag 给 collect.py（:292-306） | `DISPLAY_VALUE` `SIM_SETTLE_S` `DYNAMICS_CHOICES` `SPLIT_POLICY`（:51-60） | `--steps` 默认 = `split_manifest.DEFAULT_STEPS`（**一个 split 哈希模块里放采集步数**，split_manifest.py:36-39）；`--window-from` 默认 `None` 再按 task 解析（:490-491）；`--instances 4` 没有取值依据（cgroup 12 核、软件渲染 3 s/步都在 BRIEF 不在代码）；`--episode-timeout 600` `--sim-timeout 120` 无注释；没有「总集数」flag，用户要自己算 `instances × episodes-per-instance` |
| `sim/tools/collect.py`（1583） | 23（:1119-1167） | `UAVRGB_RENDER_BACKEND`（只记录，:1562）、`UAV_RGB_BUILD_DIR`（经 student.py） | `--cf-sil-params` 默认 `sim/dynamics/cf_sil_v1.json`（:160-162）；cf_blackbox 参数文件（:165） | 顶部 31 个常量（:73-104、147-165），**大多数有行尾为什么**，是仓库里最像样板的一段 | `--steps` 同上；`--hold 1.0`（:1121）与 collect_parallel/eval 各一份；`stop_radius or 0.5` 写了两次（:1114、:1386）；相机 320×240 写死在 manifest 兜底（:1494）；`take_off(timeout=120.0)` 在签名里（:616）；`collect_one` 有 20 个参数（:714-716）、322 行 |
| `tools/preprocess.py`（684） | 7（:584-599） | 无 | 无 | `TEXT_LEN=9`（:47）、`DEFAULT_SIZE=(160,120)`（:63，有为什么）、`HEADER` | 基本没有；`TEXT_LEN`/`DEFAULT_SIZE` 是 C++ 常量的手抄副本，靠 check.sh:12-36 用 sed 对齐 |
| `src/train_main.cpp`（`uavrgb_train`，211） | 17（:46-97） | 无 | 无 | 无（都在 `TrainConfig`） | **全部默认值在 `src/include/uavrgb/trainer.hpp:21-25,32,39`**：epochs 10 / batch 32 / lr 1e-3 / max_grad_norm 5.0 / seed 1234 / text_dim 0 / distill_alpha 0.5，`usage()`（:22-31）不打印默认值，没有注释说明为什么是这些数；文档里三套不同数字（README:51 `1/5`，HANDOFF:906 `10/128/1e-3`，HANDOFF:379 `20/64/5e-4`，HANDOFF:220 `10/64/1e-3`） |
| `ui/trainer_native.py`（74） | 1 + 透传 | `UAV_RGB_TRAIN_BIN` / `UAVRGB_TRAIN_BIN` / `UAV_RGB_BUILD_DIR`（:20-23、60） | 读 `train_config.json`（studio.py:1333-1340 写） | `SCHEMA` | 没有超参数，纯适配器；达标 |
| `tools/train_student_with_video.py`（280） | 4 + 透传（:202-208） | `UAV_RGB_BUILD_DIR`（默认 **`src/build`**，:218）、`UAVRGB_3DGS_PYTHON`、`DATA_ROOT`（:156-159）、`UAV_RGB_PYTHON`（:196） | 从 `.uavr` 尾部 meta 推断 task/split/dynamics/hold/window（:126-149） | `UAVR_HEADER` | 录像 `episodes=1` 写死（:147、:173）；推断规则只在 docstring（:15-20） |
| `teacher/train.py`（286） | 13（:152-174） | `DATA_ROOT`（:32） | 无 | `RUNS_ROOT` | 默认值全在 `build_parser` 里可见（epochs 30 / batch 8 / lr 1e-3 / seq-len 8），只有 `--class-weights` 有为什么（YAW 占 8.8%）；batch 8、seq-len 8 没有记录依据 |
| `rl_teacher/train_ppo.py`（63） | 5（:10-14） | 无 | `reward_v1.json`（:14） | 无 | PPO 超参 **inline 字面量无注释**：`n_steps=256,batch_size=256,learning_rate=3e-4`（:46）；seed 池 `range(800)`（:39）、`seeds[:64]`（:41）；物理档字符串 `'cf_stab_approx_v1_dt0.04_latency0.15_k1_tilt2'`（:55）是手抄的第四套动力学常数（C 报告 §1 第 6 行） |
| `sim/tools/eval_closed_loop.py`（861） | 27（:550-603） | 经 collect_parallel 的 `gz_env` | `--protocol <json>` 可选 | `WAYPOINT_RADIUS=1.0`（:42）、`REACH_RADIUS=1.5`（:43-47，为什么写得很好）、`CAMERA_PROFILE`（:51） | `--steps 80`（:562）与采集的 40 不同、没写为什么；`--hold 1.0`（:563）第三份；`--episodes None→3`（:165-170）；`--max-seeds 500`（:587）；`--goal-margin 1.0` 的解释在 HANDOFF.md:996-998 不在代码；campus `goal_radius 0.5` 字面量（:786）；**20 个冻结 test seed 只存在于 HANDOFF.md:915，仓库里没有任何文件/常量**；teacher-rl 的 hold 0.2 约束在三处各写一遍（collect.py:1183、collect_parallel.py:511-512、eval_closed_loop.py:670-673） |
| `sim/3dgs/collect_3dgs.py`（380）/ `eval_3dgs.py`（207） | 18 / 18 | `--device cuda` 字面量；TORCH 缓存路径只在 BRIEF | `--steps`/`--hold` 默认 `None` → `scene.json` 的 `max_steps`/`step_s` → 再兜底 `scene.py:194-195` 的 60 / 0.5（**三层**，collect_3dgs.py:323-324、eval_3dgs.py:83-84）；`--dynamics-params` 默认 `cf_blackbox_v1.json`（cf_blackbox.py:31-33） | 无 | `waypoint_radius=1.0` 字面量（eval_3dgs.py:124），与 eval_closed_loop 的 `WAYPOINT_RADIUS` 是两份；`split="test"` 写死（eval_3dgs.py:109）；hold 3DGS 默认 0.5 vs Gazebo 1.0，没有解释；一半 flag 没有 help（:307-318、:62-72）；不能 `python sim/3dgs/collect_3dgs.py` 直接跑（相对 import，:26-29），只能 `python -m` |
| `sim/tools/scene_gen.py`（598） | 10 | 无 | 模板 `sim/worlds/uav_rgb.sdf` | 顶部 25 个常量（:51-103），多数有为什么 | docstring 过时：:17-18 说「默认 --steps 因此是 80」（现在 40）；:16 说巡航 z≈2.3 而 collect 定高 2.1；`CRUISE_HEIGHT=2.3`（:80）无注释 |
| `sim/tools/campus.py`（162） | 库 | 无 | 纹理目录 | `VERSION` `FAMILIES` `RADIUS=0.5`（:12-15） | `RADIUS 0.5` 无为什么；「每 8 个 seed 一张图」的 8 出现在 :31 和 :52 两处，它是 eval 里「20 集只有 3 张图」（eval_closed_loop.py:815-817）的根源 |
| `sim/tools/teacher.py`（548） | 库 | 无 | 无 | `BAND_LO/HI`、`Y_GAIN`、`TEACHER_VERSION`（:19-34，有为什么） | **规划几何藏在 5 个函数签名里**：`expert_action(turn_dist=1.5, emergency=0.8, switch_margin=0.75)`（:56-57）、`ray_clear(half=0.2, samples=33)`（:137）、`plan_action(lookahead=1.6, clearance=0.36, y_max=30.0, tol=15.0, turn_weight=0.25, grid=5)`（:174-175）、`astar_path(clearance=0.5, step=0.1)`（:230）、`plan_action_global(clearance=0.5, step=0.1, lookahead=0.7, tol=25.0)`（:311-312）；前向安全门 `0.4 m` / `0.36` 字面量（:214、:354）。换 E1 机体要改的正是这些（C 报告 §1 第 5 行） |
| `sim/dynamics/cf_blackbox_v1.json`（22）/ `cf_sil_v1.json`（97） | — | — | 自身 | 有 `notes[]` 和 `calibrated:false`，风格对 | 与 collect.py 常量重复：`forward_speed_mps 0.3`（cf_sil:81）、`yaw_rate_rad_s 0.6`（:82）、`altitude_target_m 2.1`（:85）；cf_blackbox 的偏航 0.785 rad/s（:18）、高度 2.0（:20）与 cf_sil 不同 |
| `config/` | — | — | 只有 `uart_protocol.example.json`（占位） | — | 与实验无关 |
| `ui/studio.py`（3060） | GUI 字段 | 多个 | 写 `train_config.json` | `ACTION_NAMES` 和 `ACTIONS` **同一内容定义两次**（:62-63） | 手动评测不传 dynamics/hold/window/steps，只写死 `--text-mode keep`（:2412-2415）；live demo 不传 dynamics/window（:2266-2269）（C 报告 N2） |

### 1.3 改一套实验参数要碰几处

目标：采 300 集、每集 40 步、hold 1.0、训 30 轮 batch 128 lr 1e-3、评 20 seed（教师 / 旧 X3 学生 / 新 E1 学生 / 盲前进）。

| 步骤 | 文件 | 命令 | 用户要写的 flag 实参 | 备注 |
| --- | --- | --- | --- | --- |
| 采集 | collect_parallel.py | 1 | `--instances 4 --episodes-per-instance 75 --steps 40 --hold 1.0 --base-seed 0 --out …`（新载具再加 `--dynamics cf_sitl --vehicle e1_mini [--template …]`）= 7–9 | 300 集要自己除成 4×75；还要 3 个环境变量 `UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg` |
| 审计 | audit_dataset.py | 1 | 2 | |
| 打包 | preprocess.py | 1 | 3 + 手写 shell glob `…/{train,val,test}/*/` | |
| 训练 | uavrgb_train | 1 | `--data --epochs 30 --batch 128 --lr 1e-3 --seed 1234 --class-weights auto --drop-collision --out` = 8 | 二进制路径两个候选 |
| 评测 | eval_closed_loop.py | 4 | 每条 `--policy … [--ckpt …] --task base --split test --steps 80 --hold 1.0 --window-from … --dynamics … --seeds <90 字符> --out …` ≈ 9–10 | 同一个 seed 列表粘 4 次；hold 1.0 第 2–5 次 |
| 合计 | **5 个脚本** | **8 条命令** | **约 45 个 flag 实参 + 3 个环境变量** | `hold` 敲 5 次、`steps` 两个不同值敲 5 次、seed 列表敲 4 次；没有任何文件记录「这次实验用的是哪套」，只能事后翻各产物的 manifest |

### 1.4 同一个参数在几处重复定义

| 参数 | 位置 → 值 | 备注 |
| --- | --- | --- |
| `hold` 默认 | collect.py:1121、collect_parallel.py:455、eval_closed_loop.py:563 → **1.0**；collect_3dgs.py:324、eval_3dgs.py:84、sim/3dgs/scene.py:195 → **0.5**；smoke_campus.py:28 → 1.；teacher-rl/predictive 必须 0.2：collect.py:1183、collect_parallel.py:511-512、eval_closed_loop.py:670-673 | 9 处；Gazebo 与 3DGS 默认不同且没写原因 |
| `steps` 默认 | split_manifest.py:39 → **40**（collect.py:1119、collect_parallel.py:453 引用）；eval_closed_loop.py:562 → **80**；smoke_campus.py:17 → 80；sim/3dgs/scene.py:194 → 60；scene_gen.py:17-18 docstring 写 80（过时） | 采集 40 / 评测 80 是有意的（评测给足步数），但没有任何注释说 |
| 模型输入 160×120 | src/include/uavrgb/observation.hpp:15-16（合同）；tools/preprocess.py:63；tools/airsim_to_uavr.py:238；teacher/student_gru_proto.py:9-10；src/model.cpp:20（注释）；4 个测试夹具 | check.sh:12-22 只对齐 hpp 与 preprocess 两处 |
| 相机 320×240 | collect.py:1494 兜底；eval_closed_loop.py:51 `CAMERA_PROFILE`；sdf；collect_3dgs/eval_3dgs 的 `--width/--height` 默认 None→scene | |
| 动作编号/名字 | src/include/uavrgb/action.hpp:8；teacher.py:19-20；preprocess.py:64；eval_closed_loop.py:41；studio.py:62-63（两份）；record_views.py:31；airsim_to_uavr.py:35；cf_sil.py:40；cf_blackbox.py:26；config/uart_protocol.example.json `action_encoding` | 11 份；Python 侧只有 teacher.py 那份被别处 import |
| `SPEED_FWD` 0.3 m/s | collect.py:97；audit_dataset.py:79（注释「与 collect.py 一致」，靠人工）；cf_sil_v1.json:81；collect.py:154 `CF_FWD_TILT_DEG=2.0` 注释「终端速度≈0.34」；cf_blackbox_v1.json:15-16 `1.3787/1.0` → 终端速度 **1.38 m/s**（不同）；BRIEF §4 0.3 | |
| `SPEED_YAW` 0.6 rad/s | collect.py:98；cf_sil_v1.json:82；cf_blackbox_v1.json:18 → **0.785**（45°/s，不同）；rl_teacher/core.py（C 报告） | |
| 巡航高度 | collect.py:100 `Z_TARGET=2.1`；scene_gen.py:80 `CRUISE_HEIGHT=2.3`；cf_sil_v1.json:85 2.1；cf_blackbox_v1.json:20 2.0；BRIEF §4 目标 2.0 | **四个值五个地方**；scene_gen 用 2.3 校验柱子挡不挡，collect 实际飞 2.1 |
| 机身半宽 | scene_gen.py:76 `SPAWN_BODY_HALF=0.245`；teacher.py:115-116 `_body_hits_box` 的 half 由调用方传（docstring 0.47 m）；teacher.py:137 `ray_clear(half=0.2)`；rl_teacher/core.py 0.356/0.38（C 报告） | |
| `stop_radius` 兜底 0.5 | collect.py:1114、:1386；eval_closed_loop.py:786（campus goal_radius）；campus.py:15 `RADIUS=0.5` | |
| 到达半径 | eval_closed_loop.py:42 `WAYPOINT_RADIUS=1.0`、:47 `REACH_RADIUS=1.5`；eval_3dgs.py:124 `waypoint_radius=1.0` 字面量 | |
| 20 个冻结 test seed | 只在 docs/HANDOFF.md:915 | **实测：在当前 `SPLIT_SALT=uavrgb.geom.v5` 下这 20 个 seed 是 17 train / 2 val / 1 test**（`split_for_seed`），`--split test --seeds 23,…` 会跳过 19 个。它们是 geom.v3 时代冻结的，已失效（详见 §4 待确认） |
| `uavrgb_train` 默认 | trainer.hpp:21-25 → 10 / 32 / 1e-3 / 5.0 / 1234 | CLI 不显示 |
| 构建目录 | `build/check`（README:49、check.sh:6、student.py:22、trainer_native.py:20、train_students.sh:16）vs `src/build`（HANDOFF:890、906、384；train_student_with_video.py:218） | |

## 2. 对照 §6.10 逐条打分

✓ 达标 △ 部分 ✗ 不达标。七条：①超参数集中在顶部配置区 ②写字面量 ③行尾一句取值依据 ④环境变量可覆盖但默认值在文件里可见 ⑤运行方式一行写在文件头 ⑥从上到下读就是流程 ⑦注释写为什么。

| 入口 | ①集中 | ②字面量 | ③行尾理由 | ④env 默认可见 | ⑤运行一行 | ⑥顺序即流程 | ⑦写为什么 | 一句话 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| collect_parallel.py | ✗ 23 个 flag 的默认值散在 :450-488，一半在别的文件 | △ steps 引用 split_manifest | △ help 写语义不写依据；只有 `SIM_SETTLE_S`（:54）有 | △ `RENDER_BACKEND` 默认 `"gpu"` 在 `gz_env` 函数体里（:93） | ✓ :4-6，但缺本容器必需的 3 个 env 前缀 | ✓ docstring 五步 + main 里 1)–5) 注释 | ✓ | 流程注释写得好，参数找不到 |
| collect.py | △ 31 个常量在顶部（:73-165），但 argparse 23 个 flag 在 1117 行以后 | ✓ | ✓ :85-104 是全仓库范例（每个值一句为什么） | ✓ | ✓ :7-12 | ✗ `main` 463 行、`collect_one` 322 行 20 个参数，流程被 cf_sil / blackbox / teacher-rl / student 四个分支切碎 | ✓ | 常量区达标，主体读不动 |
| preprocess.py | ✓ | ✓ | ✓（:62） | n/a | ✓ | ✓ | ✓ | 达标，不动 |
| uavrgb_train（C++） | ✗ 默认值在 trainer.hpp，CLI 不显示 | ✓（在 hpp） | ✗ trainer.hpp:21-25 没有一句为什么 | n/a | ✓ train_main.cpp:1-8 | ✓ | △ 只有校验有为什么 | 不改 C++；Python 侧由流水线托住 |
| trainer_native.py | n/a（无超参） | — | — | ✓ :10-12 写了 env 覆盖 | ✓ | ✓ | ✓ | 达标 |
| train_student_with_video.py | △ 推断规则在 docstring | ✓ | △ | ✗ `UAV_RGB_BUILD_DIR` 默认 `src/build`（:218）与别处 `build/check` 不同 | ✓ | ✓ | ✓ | 一行修默认目录 |
| teacher/train.py | ✓ `build_parser` 一处 | ✓ | △ 只有 class-weights 有依据 | ✓ `DATA_ROOT`（:32） | ✓ :4-6 | ✓ | △ | 补 3 条注释就达标 |
| rl_teacher/train_ppo.py | ✗ PPO 超参 inline（:46）、seed 池 800/64 inline（:39,41） | ✓ | ✗ | n/a | △ 只在文件尾（:62） | ✓ | ✗ 全文无「为什么」 | 无空格压行写法（`p=argparse…`）本身就难读 |
| eval_closed_loop.py | ✗ 27 个 flag + 顶部 4 常量 + 兜底 0.5 在 collect.py | △ | △ `REACH_RADIUS` 的为什么是范例（:43-47）；steps 80 / goal-margin 1.0 / max-seeds 500 没有 | ✓ | ✓ :8-19 | △ `main` 311 行，其中参数校验约 100 行（:604-703） | ✓ | 补配置区 + 冻结 seed 常量 |
| collect_3dgs.py / eval_3dgs.py | ✗ 默认在 flag(None) → scene.json → scene.py 三层 | ✗ `default=None` | ✗ 一半 flag 无 help | △ `--device cuda` 字面量；TORCH 缓存路径只在 BRIEF | ✓ | ✓ | △ | 把兜底数字提到顶部 |
| scene_gen.py | ✓ :51-103 | ✓ | ✓ 多数 | n/a | ✓ | ✓ | ✓ | 达标，但 docstring :16-18 过时会误导 |
| campus.py | ✓ | ✓ | △ `RADIUS`、「8 seed/图」无依据 | n/a | 库 | 库 | △ | 补 2 条注释 |
| teacher.py | △ 顶部 3 个有，规划几何 12 个在 5 个签名里 | ✓ | △ `tol=25` 有（:316），clearance 0.36/0.5、lookahead 0.7/1.6、0.4 m 没有 | n/a | ✓ | ✓ | ✓ | 把签名默认提成常量 |
| sim/dynamics/*.json | ✓ | ✓ | ✓ `notes[]` + `calibrated` | n/a | n/a | n/a | ✓ | 风格对；问题是与 collect.py 重复 |

总评：**常量区写法**（collect.py:85-104、scene_gen.py:51-103、eval_closed_loop.py:43-47）已经是样板水平；差距集中在三点：(1) 用户要改的「实验级」参数（集数、步数、hold、轮数、batch、seed 列表）没有一个地方集中放，散在 argparse 默认 + 文档；(2) 默认值跨文件引用或埋在函数签名 / 函数体里；(3) 同一数字多处手抄（§1.4）。

## 3. 具体改法（每个入口 20–60 行，测试不动，不加配置类 / YAML / 依赖）

统一手法：在 import 之后、第一个函数之前放一个 `CFG = dict(...)`（或一组模块常量），每个参数一行、字面量、行尾为什么；`argparse` 的 `default=CFG["x"]`，flag 名、choices、help 全部不变，所以现有命令行和测试（`test_eval_provenance.py` 用 argv 调 `main()`、`test_collect_parallel.py` 用 `SimpleNamespace` 调内部函数）不受影响。行尾注释里的数字都来自仓库已有注释 / 文档 / 实验记录，来源写在后面；找不到依据的写「依据未记录」而不是编一个。

### 3.1 `sim/tools/collect_parallel.py`（约 45 行：+28 配置区，−/~17 处 argparse 默认值）

放在 :69（`import asset_campus` 之后）：

```python
# ───── 配置区：用户常改的实验参数。命令行同名 flag 可覆盖；默认值只在这里改 ─────
# 运行：UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg \
#       "$UAV_RGB_PYTHON" sim/tools/collect_parallel.py --out <新目录>
# 本容器 gpu 渲染后端段错误，必须 software（BRIEF §2）；别的机器默认 gpu 不变。
CFG = dict(
    instances=4,              # 并行 gz sim 数。cgroup 只有 12 核；软件渲染每实例 LP_NUM_THREADS=1 约 3 s/步；历史实测 4 实例 0.878 步/s（sim/README §4.6.2）
    episodes_per_instance=1,  # 总集数 = instances × 这个数；每集重启一次 gz sim（collect.py 没有 episode reset）
    steps=40,                 # v3 几何 + A* 每步约 0.27 m，T2 实测 29–38 步过线（中位 30）；80 步会让一半帧在柱区外原地转（原 split_manifest.py:36-38）
    hold=1.0,                 # 每动作仿真时间秒，与 RTF 无关。x3_velocity 基线 1.0；cf_stab / teacher-rl 必须 0.2（5 Hz）。评测端必须用同一个值
    window_from=None,         # None=按任务：campus 用 dispatch，其余 observation（唯一被采纳基线；dispatch 的 SR 只有 0.55，HANDOFF §4.5）
    dynamics="x3_velocity",   # 旧基线。cf_stab=摇杆模型；cf_sitl=E1 MINI 固件在环（另一 agent 正在加）
    task="base",              # base=柱阵；l1=同色柱对+关系词（必须 spawn_mode="random"）；campus / campus_assets
    spawn_mode="default",     # random=y/yaw 随机出生，l1 必选
    appearance="v1",          # v1 删固定地标+随机光照；复现 2026-11-04 前的旧数据才用 legacy
    policy="teacher",         # forward=盲前进对照；teacher-rl 需 --rl-checkpoint
    base_seed=0,              # scene_id = seed = base_seed + 序号；同 seed 字节级复现世界
    episode_timeout_s=600.0,  # 软件渲染 3 s/步 × 80 步 + 起飞 ≈ 4–5 min，留一倍余量
    sim_timeout_s=120.0,      # 等相机/深度首帧；8 实例软件 GL 下 RTF 可低到 0.15（collect.py:622 的教训）
)
RENDER_BACKEND_DEFAULT = "gpu"   # gz_env() 读 UAVRGB_RENDER_BACKEND 时的兜底；本容器要显式 software
# SIM_SETTLE_S / DISPLAY_VALUE / XDG_RUNTIME_DIR 留在原处（:51-54），它们不是实验参数
```

对应改动：
- `gz_env` :93 `env.get("UAVRGB_RENDER_BACKEND", "gpu")` → `env.get("UAVRGB_RENDER_BACKEND", RENDER_BACKEND_DEFAULT)`（仍在调用时读 `env`，`test_rendering.py` 的 patch 照常生效）。
- :450-488 每个 `default=` 换成 `default=CFG["..."]`（`--steps` 不再引用 `sm.DEFAULT_STEPS`）。
- `split_manifest.py:36-39` 的 `DEFAULT_STEPS` 搬到 `collect.py` 顶部常量区（它只被 collect.py:1119 和这里用；`split_manifest` 是 split 哈希模块，放采集步数是放错了地方）。collect_parallel 永远显式传 `--steps`（:293），所以 collect.py 那份只管单跑调试；两处都写 40，注释互相指向。
- 不动：`Task`、`run_episode`、`_popen_collect`、manifest 字段、退出码。

### 3.2 `sim/tools/collect.py`（约 25 行，全在顶部常量区和 argparse 默认值）

常量区 :85-104 已经达标，只补齐散落的几个，放在 :104 之后：

```python
# ───── 单跑 collect.py 的实验默认值（collect_parallel / eval_closed_loop 都显式传 flag 覆盖） ─────
DEFAULT_STEPS = 40          # 从 split_manifest 搬来：v3 几何 + A* 每步约 0.27 m，29–38 步过线；80 步一半帧在柱区外原地转
DEFAULT_HOLD_S = 1.0        # 仿真时间秒；x3_velocity 基线。cf_stab/teacher-rl 必须 0.2
DEFAULT_WINDOW_FROM = "observation"   # 唯一被采纳基线；dispatch 只复现 T7-R
STOP_RADIUS_DEFAULT = 0.5   # scene.json 没给 stop_radius 时的兜底；L1 训练目标「0.5 m 内停」（EVAL_STANDARD §5.2）
CAMERA_W, CAMERA_H = 320, 240   # /uav/camera_info 读不到时写进 manifest 的兜底；与 sdf 一致，模型输入 160×120 由 preprocess 下采样
```

对应改动：:1119 `default=DEFAULT_STEPS`；:1121 `default=DEFAULT_HOLD_S`；:1124 `default=DEFAULT_WINDOW_FROM`；:1114 和 :1386 的 `stop_radius or 0.5` → `or STOP_RADIUS_DEFAULT`；:1494 `{"width": 320, "height": 240, ...}` → `{"width": CAMERA_W, "height": CAMERA_H, ...}`。去掉 `import split_manifest` 里对 `DEFAULT_STEPS` 的依赖即可，`sm` 其它用法不变。

不拆 `collect_one`（322 行）和 `main`（463 行）：拆了就是重构；它们不是用户改参数的地方。

### 3.3 `tools/preprocess.py`

达标，不改。`TEXT_LEN` / `DEFAULT_SIZE` 的手抄副本由 check.sh:12-36 把关，保留。

### 3.4 `uavrgb_train`（`src/train_main.cpp`，不改 C++）

Python 侧不另写 wrapper：训练超参数放进 §4 流水线的 `CFG`（`epochs / batch / lr / seed / class_weights / text_dim`），流水线的 `train()` 一步拼命令行，用户改数字只改 CFG。单独手跑时继续敲 `uavrgb_train` 的 flag。看不见的默认值（trainer.hpp:21-25：epochs 10、batch 32、lr 1e-3、max-grad-norm 5.0、seed 1234、text-dim 0、distill-alpha 0.5）请求主会话抄进 `docs/HANDOFF.md` §十二 那条训练命令的注释里，并把那里的 `src/build/` 改成 `build/check/`（与 README/check.sh 一致）。

### 3.5 `ui/trainer_native.py`

不改。74 行适配器，没有超参数，env 覆盖写法已在文件头（:10-12）。

### 3.6 `tools/train_student_with_video.py`（1 行）

:218 `REPO_ROOT / "src" / "build"` → `REPO_ROOT / "build" / "check"`，与 `student.py:22-23`、`trainer_native.py:20-23`、`train_students.sh:16`、`check.sh:6` 对齐。`test_train_student_with_video.py:110-122` 自己设了 `UAV_RGB_BUILD_DIR`，不受影响。docstring :9 的「默认 `<repo>/src/build/uavrgb_train`」同步改。

### 3.7 `teacher/train.py`（约 6 行注释）

`build_parser`（:152-174）本身就是配置区，只缺依据。补：

```python
    p.add_argument("--epochs", type=int, default=30)     # RUN.md：404 集 × 30 epoch 在 CPU 上很慢，冒烟先 --epochs 1
    p.add_argument("--batch", dest="batch_size", type=int, default=8)   # 依据未记录（ResNet18 + 8 帧序列的显存）；待实测后补
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=8,
                   help="每个 episode 取最后 T 帧")   # 依据未记录
```

`test_train_metrics.py:20-25` 只调 `build_parser().parse_args`，默认值不变即可。

### 3.8 `rl_teacher/train_ppo.py`（约 20 行）

顶部加配置区，:46 的 inline 字面量改引用；顺手把 `p=argparse.ArgumentParser()` 这种无空格写法按 PEP 8 加空格（纯格式，不改行为）：

```python
CFG = dict(
    total_steps=100_000,   # PPO 总步数；依据未记录
    n_envs=4,              # DummyVecEnv 并行环境数（纯 Python 物理，CPU 侧）
    seed=1234,             # 与 uavrgb_train 默认 seed 一致
    n_steps=256,           # 每环境每轮采样；依据未记录
    batch_size=256,
    learning_rate=3e-4,    # SB3 PPO 默认值
    seed_pool=800,         # asset_campus 的 seed 扫描上限；每 8 个 seed 一张图（campus.py:30-31）
    env_check_seeds=64,    # check_env 只抽前 64 个 train seed
)
```

`model=PPO('MlpPolicy', env, seed=args.seed, n_steps=CFG["n_steps"], batch_size=CFG["batch_size"], learning_rate=CFG["learning_rate"], ...)`；`range(800)` → `range(CFG["seed_pool"])`；`seeds[:64]` → `seeds[:CFG["env_check_seeds"]]`。:55 的物理档字符串是手抄的第四套常数，先不动，报到 C 报告的那条（SITL 参数单源）里一起收。

### 3.9 `sim/tools/eval_closed_loop.py`（约 40 行）

放在 :63（`DYNAMICS_PROFILES = cp.DYNAMICS_CHOICES`）之后：

```python
# ───── 配置区：闭环评测默认值；命令行同名 flag 可覆盖 ─────
# 运行：UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg \
#       "$UAV_RGB_PYTHON" sim/tools/eval_closed_loop.py --policy student --ckpt <bin> --out <新目录>
CFG = dict(
    steps=80,            # 比采集的 40 多一倍：评测要给学生飞完全程的余量（T7 起一直 80）；集在碰撞/到达时提前结束
    hold=1.0,            # 必须等于训练数据的 hold_s；cf_stab/teacher-rl 数据是 0.2。不一致时 --protocol 会标 ood=true
    episodes=None,       # None：给了 --seeds 用其个数，否则 3（冒烟量）
    split="test",        # 评测只许 test；val 用来看，test 用来报（EVAL_STANDARD §4）
    base_seed=0,
    max_seeds=500,       # test 约占几何组 10%，扫 500 个 seed 够凑 20–50 集
    seeds="",            # 固定对照集：第一组跑完后把它 eval.json 的 attempted_seeds 粘进来，四组同场景。HANDOFF §十二 的 23..180 是 geom.v3 时代冻结的，在 v5 salt 下 17/20 落在 train，已失效
    task="base", appearance="v1", dynamics="x3_velocity", window_from=None,   # window None=按任务（campus→dispatch，其余 observation）
    text_mode="keep",
    goal_margin=1.0,     # base 任务成功线 goal_x = 最远柱 x + 1.0 m（HANDOFF §十三「其他符号」）
    sim_timeout_s=120.0, episode_timeout_s=600.0,   # 同 collect_parallel
)
```

:550-603 各 `default=` 换成 `default=CFG[...]`；其余不动（`--window-from` 的 None 语义保留，:605-606）。`WAYPOINT_RADIUS` / `REACH_RADIUS` / `CAMERA_PROFILE` 留在 :42-51（它们是口径合同不是实验参数）。campus 的 `goal_radius 0.5`（:786）→ `campus.RADIUS`（已 import，:59），去掉一处手抄。

### 3.10 `sim/3dgs/collect_3dgs.py` / `eval_3dgs.py`（各约 15 行）

两个文件顶部各加（import 之后）：

```python
# ───── 配置区。None 表示「跟 scene.json 走」，scene.json 没写时的兜底数字也写在这里 ─────
# 运行：$DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m sim.3dgs.collect_3dgs --scene <scene.json> --out <新目录>
#      （必须 python -m 从仓库根跑：文件用相对 import；首次渲染 gsplat JIT 约 170 s，先 unset TORCH_CUDA_ARCH_LIST）
CFG = dict(
    steps=None,          # None=scene.max_steps；scene 也没写时 60（sim/3dgs/scene.py:194）
    hold=None,           # None=scene.step_s；兜底 0.5 s（scene.py:195）。Gazebo 侧默认 1.0，两边不同的原因仓库里没有记录
    episodes=1,
    policy="teacher",    # eval 侧默认 student
    device="cuda",
    seed=0,              # 第 ep 集 seed = seed + ep
    spawn_jitter_m=0.0,  # 0=固定出生点；>0 在 scene.spawn 周围随机
    dynamics_params=None,  # None=sim/dynamics/cf_blackbox_v1.json（calibrated=false 占位值）
)
SCENE_FALLBACK_STEPS, SCENE_FALLBACK_HOLD_S = 60, 0.5   # :323-324 / :83-84 的两个字面量提到这里
```

`:323-324`（eval :83-84）改成 `scene.get("max_steps", SCENE_FALLBACK_STEPS)` / `scene.get("step_s", SCENE_FALLBACK_HOLD_S)`；argparse `default=CFG[...]`；给 :307-318（eval :62-72）里没有 help 的 flag 补一句 help。`eval_3dgs.py:124` 的 `waypoint_radius=1.0` → `from eval_closed_loop import WAYPOINT_RADIUS`（该文件已从 eval_closed_loop import `summarize_episode`，同一行加一个名字）。`test_collect_eval.py` 走 `--mock-renderer` 和显式 `--steps`，不受影响。

### 3.11 `sim/tools/scene_gen.py`（3 行）/ `sim/tools/campus.py`（2 行注释）

- scene_gen.py:16-18 docstring 改成「挡住 2.1 m 巡航高度（collect.py `Z_TARGET`）；x∈[1.5,5.0]，A* 每步约 0.27 m，默认 `--steps 40` 见 collect.py」；:80 `CRUISE_HEIGHT = 2.3` 加注释 `# 校验柱子是否挡住巡航层：取 collect.py Z_TARGET 2.1 + 0.2 定高死区上沿；比它低的柱子不算障碍`（这是从 Z_DEADBAND=0.20 推的，请主会话确认，确认不了就写「依据未记录」）。
- campus.py:15 `RADIUS = 0.5` 加 `# 可见彩柱前目标区半径（EVAL_STANDARD 顶部 visible_goal_reach_v1）`；:31 的 `seed // 8` 加 `# 每 8 个 seed 同一张图：4 色 × 左右；eval 的 unique_maps 按此折算`。

### 3.12 `sim/tools/teacher.py`（约 15 行）

把 5 个签名里的几何默认值提成顶部常量，签名改为引用常量（参数名、顺序、默认值不变，调用方和 `teacher.py` 自测、`test_audit_dataset.py` 的 `label_recompute` 都不用动）。放在 :34（`Y_GAIN`）之后：

```python
# ───── 规划几何（全部按 X3：0.47 m 方盒、外接直径 0.711 m）。换 E1 MINI 载具只改这里 ─────
RAY_BODY_HALF = 0.2        # ray_clear 采样时的机身半宽；0.235+0.01 是碰撞盒，这里略小留给 clearance 承担
CLEAR_LOCAL = 0.36         # plan_action 射线净空（柱面到机身中心）；依据未记录
CLEAR_ASTAR = 0.5          # A* 栅格膨胀 = 外接半径 0.356 + 约 0.14 余量
LOOKAHEAD_LOCAL = 1.6      # 局部规划射线长度；依据未记录
LOOKAHEAD_ASTAR = 0.7      # 纯追踪前视；每步漂约 0.27 m，取 2.5 步
TOL_LOCAL_DEG = 15.0
TOL_ASTAR_DEG = 25.0       # 偏航一步约 34°，容差小于半步会左右过冲（T1 门柱振荡根因）
FRONT_CHECK_M = 0.4        # 前向安全门：下一步 0.3 m/s × 1 s 前进量 + 余量；hold 变了要重定
TURN_DIST_M, EMERGENCY_M, SWITCH_MARGIN_M = 1.5, 0.8, 0.75   # 深度反应式专家的三个阈值（expert_action）
```

对应：`:56-57 expert_action(..., turn_dist=TURN_DIST_M, emergency=EMERGENCY_M, ..., switch_margin=SWITCH_MARGIN_M)`；`:137 ray_clear(..., half=RAY_BODY_HALF, ...)`；`:174-175 plan_action(..., lookahead=LOOKAHEAD_LOCAL, clearance=CLEAR_LOCAL, ..., tol=TOL_LOCAL_DEG, ...)`；`:230 astar_path(..., clearance=CLEAR_ASTAR, ...)`；`:311-312 plan_action_global(..., clearance=CLEAR_ASTAR, ..., lookahead=LOOKAHEAD_ASTAR, tol=TOL_ASTAR_DEG, ...)`；`:214`、`:354` 的 `0.4` → `FRONT_CHECK_M`，`:354` 的 `0.36` → `CLEAR_LOCAL`。注意 `TEACHER_VERSION`（:25）的注释要求「改动作选择必须 bump」：这次只是等值搬家，不 bump。

### 3.13 `sim/dynamics/*.json`、`config/`

不改文件。`cf_sil_v1.json:81-85` 的 0.3 / 0.6 / 2.1 与 `collect.py:97-100` 重复是因为 cf_sil 是独立执行器；e1_mini.json 落地后 E1 的数字以它为单源（BRIEF §4）。请求：`collect.py:97-100` 四行注释里各加一句「仅 x3_velocity 档；cf_sil 见 cf_sil_v1.json，E1 见 e1_mini.json」，防止有人改了 collect.py 以为 SITL 也跟着变。

### 3.14 改动量汇总

| 入口 | 估计行数 | 测试 |
| --- | --- | --- |
| collect_parallel.py | ~45 | 不动 |
| collect.py（含搬 `DEFAULT_STEPS`，split_manifest 删 4 行） | ~25 | 不动 |
| eval_closed_loop.py | ~40 | 不动 |
| collect_3dgs.py + eval_3dgs.py | ~30 | 不动 |
| teacher.py | ~15 | 不动（自测 `python teacher.py` 仍应 PASS） |
| train_ppo.py | ~20（含格式） | 无测试 |
| teacher/train.py | ~6 注释 | 不动 |
| train_student_with_video.py | 1 | 不动 |
| scene_gen.py / campus.py | ~5 | 不动 |
| 合计 | ~190 行，全部是默认值搬家和注释 | `bash tools/check.sh --python-only` 应原样通过 |

## 4. 流水线入口设计稿：`tools/e1_pipeline.py`

### 4.1 设计取舍（为什么长这样）

- **一个文件、六个函数、一个 CFG dict**，照 `YOLOVOmodal_3871538535/train.py` 的样子：配置区在最上面，每个字面量带行尾理由；函数名就是步骤名，`STEPS` 元组的顺序就是执行顺序，`--from` 只是 `STEPS.index()` 切片。没有 class、没有 YAML、不依赖仓库里任何模块（只 `subprocess` 调现有脚本），所以不受 4 个并行 agent 改 Python 内部签名的影响——**只依赖 CLI flag 名和两个 JSON 文件的字段名**（见 4.4 表）。
- **不吞现有脚本的退出码语义，但也不被它们卡死**：
  - `collect_parallel.py:773` 一集失败就 `rc=1`，300 集里丢 1 集是常态（发布层偶发丢首条指令，`collect_parallel.py:15`），所以用 `collection_manifest.json` 的 `episodes_ok/episodes_requested`（`:693-694`）算成功率，低于 `collect_min_ok=0.95` 才停。
  - `audit_dataset.py:1019` 任一 FAIL 即 `rc=1`，流水线默认遵守（坏数据不进训练）；但 `audit_dataset.py:604-605` 的 `forward_motion` 只豁免 `cf_stab` 前缀，cf_sitl 数据会按 X3 定速 0.3×0.4 m/s 判成 FAIL（C 报告 §2 行 20 同一结论），所以给了 `audit_ignore=("forward_motion",)` 白名单，并写明审计脚本修好后改回 `()`。
  - `eval_closed_loop.py:857` 有失败集 `rc=1` 但 `eval.json` 已写；四组对照要全部跑完才有意义，所以评测步只检查 `eval.json` 存在，不因 rc 停。
- **20 个 test seed 的来源**：用户给的 23…180 是 `docs/HANDOFF.md:915` 冻结的 geom.v3 时代列表，在当前 `split_manifest.SPLIT_SALT = "uavrgb.geom.v5"`（`split_manifest.py:35`）下 17 个落 train、2 个 val、1 个 test，`eval_closed_loop.py:404-405` 的 split 过滤会把它们跳掉。所以第一组（teacher）用 `--split test --base-seed 100000 --max-seeds 500 --episodes 20` 扫出 20 个 test 场景，其余三组把它的 `attempted_seeds`（`eval_closed_loop.py:796`）回传 `--seeds`——这正是 `eval_closed_loop.py:17` 文档说的 T7 做法。要换成人工冻结列表只需填 `CFG["eval_seeds"]="a,b,c"`。
- **四组共用 `vehicle_flags()`**：`--dynamics/--hold/--window-from/--vehicle/--template` 在采集和四组评测里来自同一个函数，训练数据和评测协议不可能错开（`eval_closed_loop.py` 的 `ood` 判定依赖 `window_from/hold_s/dynamics` 一致，`:52-53`）。旧 X3 学生也在 E1 动力学下评——这是对照的本意（旧模型在新机体上掉多少），不是在它自己的 x3_velocity 下评；要后者再加一组改 `vehicle_flags`。
- **守护进程约定按 §6.9**：docstring 第一行就是 `setsid nohup … & echo $! > run.pid`；每步前后写 `status.json`；`finally` 里无论成败都落 `DONE`（`rc= 时间 last_step=`），外层等待脚本只看这个文件；`kill -- -<PGID>` 能连子进程一起杀，因为 `collect_parallel.py` 自己也 `setsid` 并在 finally 里 `kill -- -PGID`（`:18`）。
- **不做的事**：不加 GPU 空闲检查（训练前跑 `nvidia-smi` 是运行者的事，脚本里加就得处理 CUDA 之外的情况）；不做断点内续跑（`--from train` 粒度够用，训练内续跑让 `uavrgb_train --resume` 管）；不并行跑四组评测（12 核已被 4 个 gz 实例占满，串行更稳）。

### 4.2 完整草稿（299 行；放 `tools/e1_pipeline.py`，不进本次 4 个 agent 的改动范围）

```python
#!/usr/bin/env python3
"""E1 MINI 流水线：采集 → 审计 → 打包 → 训练 → 20 seed 闭环四组对照 → 汇总。守护进程跑。

运行（仓库根，先 source env.sh）：
  OUT=$DATA_ROOT/uav_rgb/e1-pipeline-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
  setsid nohup "$UAV_RGB_PYTHON" tools/e1_pipeline.py --out "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
  echo $! > "$OUT/run.pid"; ps -o pid,pgid,etime,cmd -p "$(cat "$OUT/run.pid")"   # 停：kill -- -<PGID>
看进度：tail -f $OUT/run.log；每步状态 $OUT/status.json；结束落 $OUT/DONE（rc + 时间 + 最后一步）。
续跑：--from train（跳过 collect/audit/preprocess，要求它们的产物已在 OUT 里）。
只打印命令不执行：--dry-run。改参数只改 CFG，不改函数。
预计耗时（软件渲染 3 s/步）：采集 300 集 × 40 步 ÷ 4 实例 ≈ 2.5 h；训练 30 轮 ≈ 1–2 h；评测 4 组 × 20 集 ≤ 80 步 ≈ 2–5 h。
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
BUILD = os.environ.get("UAV_RGB_BUILD_DIR", os.path.join(REPO, "build", "check"))  # check.sh 的目录；src/build 是旧的
PY = sys.executable   # 用 $UAV_RGB_PYTHON 启动本脚本，子进程继承它（preprocess/eval 需要 PIL、numpy）

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
CFG = dict(
    # 采集
    episodes=300,            # 总集数，取 instances 的整数倍。T6 学生用了 599 集 / 23960 步；300 集 ≈ 12000 步
    steps=40,                # 每集步数：v3 几何 + A* 每步约 0.27 m，T2 实测 29–38 步过线；80 步会让一半帧在柱区外原地转
    hold=1.0,                # 每动作仿真时间秒。先沿用 X3 基线 1.0 便于对比；SITL 下杆量保持到下一次 dispatch，是否要缩短待实测
    instances=4,             # 并行 gz sim。cgroup 12 核、软件渲染每实例约 1 核 3 s/步；多于 6 个互相拖慢
    base_seed=0,             # scene_id = seed = base_seed + 序号。split 由几何哈希决定，不由 seed 段决定
    task="base",             # 柱阵。l1 要同时改 spawn_mode=random、text_dim=16
    vehicle="e1_mini",       # 【待确认】新 flag --vehicle，gz agent 正在加；None=不传（X3）
    dynamics="cf_sitl",      # 【待确认】新档位 --dynamics cf_sitl，SITL lib agent 正在加；复现旧数字用 "x3_velocity"
    window_from="dispatch",  # 【待确认】C 报告 §6：cf_sitl 强制 dispatch（窗口从下发时刻算）；X3 基线是 observation
    template=None,           # 【待确认】E1 世界模板 sdf 绝对路径（gz agent 产出）；None=默认 sim/worlds/uav_rgb.sdf（X3）
    collect_min_ok=0.95,     # 偶发 1–2 集丢首条起飞指令属正常（collect_parallel 自带重试一次）；低于 95% 说明环境坏了，停
    # 审计
    audit_ignore=("forward_motion",),   # 【待确认】audit_dataset.py:604 只豁免 cf_stab，按 X3 定速判 cf_sitl 必 FAIL；
                                        # 审计脚本加了 cf_sitl 豁免后改成 ()。其余任何 FAIL 仍停
    # 打包
    drop_collision=True,     # 专家撞柱后的标签本身是错的；T6 起一直开。训练端同步传 --drop-collision
    # 训练（uavrgb_train；没列的 flag 用 trainer.hpp 默认：max-grad-norm 5.0）
    epochs=30,               # 用户本轮指定。T6 best 落在 epoch 8，20 轮后 dev F1 不再涨；30 轮给 best-ckpt 余量
    batch=128,               # 160×120 输入，4080 SUPER 32 GB 远不到上限；HANDOFF §十二 示例值。T6 实际用 64
    lr=1e-3,                 # Adam；HANDOFF §十二 示例值。l1/FiLM 用 5e-4（v4）
    seed=1234,               # trainer.hpp 默认，与历史 run 一致
    class_weights="auto",    # 训练集逆频率。FORWARD 占 ~75%，不加权 SAFE/YAW 召回归零（EVAL_STANDARD §0）
    text_dim=0,              # base 任务无指令。l1 用 16；数据带 instructions.jsonl 而 text_dim=0 时会自动传 --vision-only
    # 评测（四组同一批 test 场景）
    eval_episodes=20,        # T7 口径 20 集
    eval_steps=80,           # 评测给足全程余量（T7 起 80）；碰撞/到达会提前结束
    eval_seed_base=100000,   # 第一组从这里起扫 test split；与采集 seed 段分开只为目录好认
    eval_max_seeds=500,      # test 约占几何组 10%，扫 500 个凑 20 个够
    eval_seeds=None,         # None=第一组自动扫、其余三组复用它的 attempted_seeds；或写 "a,b,c"。
                             # HANDOFF §十二 的 23..180 是 geom.v3 时代冻结的，v5 salt 下 17/20 落在 train，不能用
    old_x3_ckpt=f"{DATA_ROOT}/uav_rgb/t6-full-1640/run/ckpt_best.bin",   # T6 best（epoch 8）：X3 / x3_velocity / hold 1.0 训的
    episode_timeout=600.0,   # 同 collect_parallel 默认：3 s/步 × 80 步 + 起飞，留一倍余量
    sim_timeout=120.0,       # 等相机/深度首帧；软件 GL 多实例时 RTF 可低到 0.15
)
EVAL_GROUPS = (   # (目录名, --policy, ckpt 的 CFG 键)。顺序即执行顺序；第一组负责扫 seed，其余复用
    ("teacher", "teacher", None),              # 柱子真值 A* 专家：新动力学下的可解性上界
    ("student-x3-old", "student", "old_x3_ckpt"),
    ("student-e1-new", "student", "new_ckpt"),
    ("forward", "forward", None),              # 盲前进基线
)
GZ_ENV = {"UAVRGB_RENDER_BACKEND": "software", "DISPLAY": ":20",
          "XDG_RUNTIME_DIR": "/tmp/xdg"}       # 本容器 gpu 后端段错误（BRIEF §2）；换机器删掉这三项即回到默认 EGL
STEPS = ("collect", "audit", "preprocess", "train", "eval", "summary")
# ──────────────────────────────────────────────────────────────────────────

DRY = False


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def run(cmd, env_extra=None):
    """跑一条命令，stdout/stderr 直接进 run.log；返回退出码，由调用方决定要不要停。"""
    log("$ " + " ".join(cmd))
    if DRY:
        return 0
    return subprocess.run(cmd, env=dict(os.environ, **(env_extra or {}))).returncode


def sh(cmd, env_extra=None):
    """非零退出就抛异常 → 流水线停。"""
    rc = run(cmd, env_extra)
    if rc != 0:
        raise RuntimeError(f"退出码 {rc}: {os.path.basename(cmd[0])} {cmd[1] if len(cmd) > 1 else ''}")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def mark(out, step, state, **extra):
    """每步落一次 status.json：新会话一看就知道跑到哪。"""
    if DRY:
        return
    path = os.path.join(out, "status.json")
    status = read_json(path) if os.path.isfile(path) else {}
    status[step] = dict(state=state, at=time.strftime("%F %T"), **extra)
    write_json(path, status)


def need(path, hint):
    if not DRY and not os.path.exists(path):
        raise RuntimeError(f"缺 {path}（{hint}）")


def vehicle_flags():
    """采集和评测共用的动力学/载具参数，保证四组评测和训练数据是同一套。"""
    flags = ["--dynamics", CFG["dynamics"], "--hold", str(CFG["hold"]), "--window-from", CFG["window_from"]]
    if CFG["vehicle"]:
        flags += ["--vehicle", CFG["vehicle"]]       # 【待确认】flag 名
    if CFG["template"]:
        flags += ["--template", CFG["template"]]
    return flags


# ───────────────────────────── 六个步骤，按顺序 ─────────────────────────────
def collect(out):
    col = os.path.join(out, "collection")
    if os.path.isdir(col) and os.listdir(col):
        raise RuntimeError(f"{col} 非空：collect_parallel 拒绝复用目录。删掉它或换 --out；想跳过采集用 --from audit")
    per = math.ceil(CFG["episodes"] / CFG["instances"])
    # collect_parallel 有一集失败就 rc=1（:773），不能拿 rc 当成败；按 manifest 里的成功率判
    run([PY, f"{REPO}/sim/tools/collect_parallel.py",
         "--instances", str(CFG["instances"]), "--episodes-per-instance", str(per),
         "--steps", str(CFG["steps"]), "--base-seed", str(CFG["base_seed"]),
         "--task", CFG["task"], "--keep-worlds", "--out", col, *vehicle_flags()],
        env_extra=GZ_ENV)
    if DRY:
        return
    man = os.path.join(col, "collection_manifest.json")
    need(man, "collect_parallel 没写出汇总，看 collection/logs/")
    m = read_json(man)
    ratio = m["episodes_ok"] / max(m["episodes_requested"], 1)
    log(f"采集 ok {m['episodes_ok']}/{m['episodes_requested']}，"
        f"split train/val/test = {[len(m['splits'][s]) for s in ('train', 'val', 'test')]}")
    if ratio < CFG["collect_min_ok"]:
        raise RuntimeError(f"采集成功率 {ratio:.2%} < {CFG['collect_min_ok']:.0%}，见 {col}/failed/ 与 logs/")


def audit(out):
    col = os.path.join(out, "collection")
    need(os.path.join(col, "collection_manifest.json"), "先跑 collect")
    report = os.path.join(col, "audit.json")
    run([PY, f"{REPO}/sim/tools/audit_dataset.py", "--dataset", col, "--out", report])   # rc=1 表示有 FAIL（:1019）
    if DRY:
        return
    need(report, "audit_dataset 没写出 audit.json")
    failed = [c["name"] for c in read_json(report)["checks"] if c["status"] == "FAIL"]
    ignored = [n for n in failed if n in CFG["audit_ignore"]]
    if ignored:
        log(f"审计 FAIL 但按 CFG.audit_ignore 放行: {ignored}")
    if [n for n in failed if n not in CFG["audit_ignore"]]:
        raise RuntimeError(f"审计 FAIL: {failed}，见 {report}")   # 不让坏数据进训练


def preprocess(out):
    col, ds = os.path.join(out, "collection"), os.path.join(out, "ds")
    eps = sorted(d for s in ("train", "val", "test") for d in glob.glob(f"{col}/{s}/*/"))
    if not DRY and not eps:
        raise RuntimeError(f"{col} 下没有 train/val/test 集目录")
    cmd = [PY, f"{REPO}/tools/preprocess.py", "--out", ds]
    if CFG["drop_collision"]:
        cmd.append("--drop-collision")
    sh(cmd + (eps or [f"{col}/{{train,val,test}}/*/"]))   # 不依赖 shell 展开，集目录在 Python 里枚举


def train(out):
    ds, run_dir = os.path.join(out, "ds"), os.path.join(out, "run")
    need(os.path.join(ds, "dataset.uavr"), "先跑 preprocess")
    need(f"{BUILD}/uavrgb_train", "先 bash tools/check.sh 构建，或设 UAV_RGB_BUILD_DIR")
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise RuntimeError(f"{run_dir} 非空：uavrgb_train 的 RunOwner 拒绝复用（run_owner.cpp:42）；删掉或换 --out")
    cmd = [f"{BUILD}/uavrgb_train", "--data", f"{ds}/dataset.uavr", "--out", run_dir,
           "--epochs", str(CFG["epochs"]), "--batch", str(CFG["batch"]), "--lr", repr(float(CFG["lr"])),
           "--seed", str(CFG["seed"]), "--class-weights", CFG["class_weights"]]
    if CFG["drop_collision"]:
        cmd.append("--drop-collision")
    if CFG["text_dim"] > 0:
        cmd += ["--text-dim", str(CFG["text_dim"])]
    elif os.path.isfile(f"{ds}/instructions.jsonl"):
        cmd.append("--vision-only")   # 数据带指令却要纯视觉基线时 trainer 要求显式声明（trainer.cpp:370-373）
    sh(cmd)
    need(f"{run_dir}/ckpt_best.bin", "训练结束但没有 best ckpt，看 run/metrics.jsonl")


def evaluate(out):
    ckpts = dict(old_x3_ckpt=CFG["old_x3_ckpt"], new_ckpt=os.path.join(out, "run", "ckpt_best.bin"))
    for key, path in ckpts.items():
        need(path, f"{key} 不存在")
    seeds = CFG["eval_seeds"]
    for name, policy, ckpt_key in EVAL_GROUPS:
        edir = os.path.join(out, f"eval-{name}")
        done = os.path.join(edir, "eval.json")
        if os.path.isfile(done):
            log(f"跳过 {name}：已有 eval.json")
        else:
            cmd = [PY, f"{REPO}/sim/tools/eval_closed_loop.py", "--policy", policy,
                   "--task", CFG["task"], "--split", "test", "--steps", str(CFG["eval_steps"]),
                   "--episodes", str(CFG["eval_episodes"]), "--out", edir,
                   "--sim-timeout", str(CFG["sim_timeout"]), "--episode-timeout", str(CFG["episode_timeout"]),
                   *vehicle_flags()]
            if ckpt_key:
                cmd += ["--ckpt", ckpts[ckpt_key], "--infer-bin", f"{BUILD}/uavrgb_infer"]
            if seeds:
                cmd += ["--seeds", seeds]
            else:   # 第一组：从 eval_seed_base 起扫 test split 凑 eval_episodes 个
                cmd += ["--base-seed", str(CFG["eval_seed_base"]), "--max-seeds", str(CFG["eval_max_seeds"])]
            # eval_closed_loop 有失败集时 rc=1 但 eval.json 已写（:857）；四组对照要全跑完再一起看，不因 rc 停
            run(cmd, env_extra=GZ_ENV)
            need(done, f"{name} 没写出 eval.json，看 run.log")
        if not seeds:
            # 后三组用同一批场景（含失败集，分母一致）；写进 status 便于复现
            seeds = ("<eval-teacher attempted_seeds>" if DRY else
                     ",".join(str(s) for s in read_json(done)["attempted_seeds"]))
            mark(out, "eval_seeds", "ok", seeds=seeds)
            log(f"四组对照共用 seeds: {seeds}")


def summary(out):
    rows = {}
    for name, _policy, _k in EVAL_GROUPS:
        path = os.path.join(out, f"eval-{name}", "eval.json")
        if DRY and not os.path.isfile(path):
            continue
        e = read_json(path)
        rows[name] = {k: e.get(k) for k in (
            "success_rate", "requested_success_rate", "collision_rate", "stuck_rate",
            "false_forward_rate", "episodes_done", "failed_count", "skipped_count",
            "success_criterion", "dynamics", "hold_s", "steps", "ood")}
        rows[name]["wilson95"] = (e.get("wilson_ci") or {}).get("success")
    if not DRY:
        write_json(os.path.join(out, "summary.json"), {"cfg": CFG, "groups": rows})
    log("%-16s %6s %6s %6s %5s %5s" % ("group", "SR", "coll", "stuck", "done", "fail"))
    for name, r in rows.items():
        log("%-16s %6s %6s %6s %5s %5s" % (name, r["success_rate"], r["collision_rate"],
                                           r["stuck_rate"], r["episodes_done"], r["failed_count"]))
    log("口径：SR 分母 episodes_done；四组同一批 seed；详见 docs/EVAL_STANDARD.md")


RUN = dict(collect=collect, audit=audit, preprocess=preprocess, train=train, eval=evaluate, summary=summary)


def main():
    global DRY
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="输出根目录（数据盘，如 $DATA_ROOT/uav_rgb/e1-pipeline-<时间>）")
    p.add_argument("--from", dest="start", choices=STEPS, default="collect", help="从哪一步开始（续跑）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令")
    a = p.parse_args()
    DRY = a.dry_run
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    if not DRY:
        write_json(os.path.join(out, "config.json"),   # 这次实验到底用了哪套参数，事后不用翻 manifest
                   {"cfg": CFG, "groups": EVAL_GROUPS, "gz_env": GZ_ENV, "argv": sys.argv,
                    "repo": REPO, "build": BUILD, "python": PY})
        if os.path.exists(os.path.join(out, "DONE")):
            os.remove(os.path.join(out, "DONE"))   # 续跑时先摘掉旧标记，免得外面的等待脚本误判
    rc, last = 0, None
    try:
        for step in STEPS[STEPS.index(a.start):]:
            last = step
            log(f"==== {step} 开始")
            mark(out, step, "running")
            t0 = time.monotonic()
            RUN[step](out)
            mark(out, step, "ok", seconds=round(time.monotonic() - t0, 1))
            log(f"==== {step} 完成 {time.monotonic() - t0:.0f}s")
    except Exception as e:   # noqa: BLE001 - 任一步失败就停，原因进 status/DONE
        rc = 1
        mark(out, last or "init", "failed", error=f"{type(e).__name__}: {e}")
        log(f"!!!! {last} 失败：{e}")
    finally:
        if not DRY:
            with open(os.path.join(out, "DONE"), "w", encoding="utf-8") as f:
                f.write(f"rc={rc} {time.strftime('%F %T')} last_step={last}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
```

### 4.3 自检（在 `tmp/agent-style/` 用 `.venv-tools` 跑，没碰仓库、没起 Gazebo）

- `python -m py_compile` 通过；`--help` 能看到 docstring 里的运行方式。
- `--dry-run` 从 collect 走到 summary 打出 8 条命令（1 采集 + 1 审计 + 1 打包 + 1 训练 + 4 评测），flag 名逐条和 `help_gazebo.txt` 里 `collect_parallel.py / audit_dataset.py / preprocess.py / eval_closed_loop.py --help` 及 `src/train_main.cpp:46-77` 对过；dry-run 不写任何文件（只 `mkdir` 输出目录）。日志存 `tmp/agent-style/dryrun.log`。
- 采集命令实际会打成：`collect_parallel.py --instances 4 --episodes-per-instance 75 --steps 40 --base-seed 0 --task base --keep-worlds --out <OUT>/collection --dynamics cf_sitl --hold 1.0 --window-from dispatch --vehicle e1_mini`；训练命令：`uavrgb_train --data <OUT>/ds/dataset.uavr --out <OUT>/run --epochs 30 --batch 128 --lr 0.001 --seed 1234 --class-weights auto --drop-collision`。

### 4.4 待确认（跑之前要主会话或对应 agent 拍板）

| # | 待确认点 | 现状依据 | 草稿里怎么处理 |
|---|---|---|---|
| 1 | `--vehicle e1_mini` 这个 flag 是否存在、叫什么 | 快照三处 argparse 都没有 `--vehicle`（`collect_parallel.py:450-488`、`collect.py:1117-1200`、`eval_closed_loop.py:547-600`）；C 报告 §6 行 114 写的是「车辆名收成常量」而不是新 flag | `CFG["vehicle"]=None` 就不传；只在 `vehicle_flags()` 一处拼 |
| 2 | `--dynamics cf_sitl` 是否已进 `collect_parallel.DYNAMICS_CHOICES`（`:57`）和 `eval_closed_loop.DYNAMICS_PROFILES`（`:63`） | 快照两处白名单只有 `x3_velocity/cf_stab/cf_blackbox`；`test_collect.py:538-542` 还断言 `cf_sil` 不在里面——新档位名叫 `cf_sitl` 才不会撞这条测试 | 不传就被 argparse 拒绝，第一步即停，不会白跑 |
| 3 | cf_sitl 是否要求显式 `--template`（E1 世界 sdf） | C 报告 §6 行 115：「cf_sitl 档必须显式给 --template」；快照默认模板是 X3 的 `sim/worlds/uav_rgb.sdf`（`collect_parallel.py:484`） | `CFG["template"]` 留 None 并标待确认；gz agent 给出路径后填绝对路径 |
| 4 | `window_from="dispatch"` + `hold=1.0` 是否是 cf_sitl 的正确组合 | C 报告 §6 行 116：cf_sitl 强制 dispatch 并要求显式 hold；但 hold 1.0 是 X3 基线值，SITL 杆量保持 1 s 是否合理没人测过（`collect.py:1131` 对 cf_stab 的建议是 0.1–0.2） | 两个值都显式传；eval 的 `ood` 会在协议不一致时报警 |
| 5 | 审计 `forward_motion` 对 cf_sitl 的误判 | `audit_dataset.py:604-605` 只豁免 `dynamics_profile.startswith("cf_stab")` | `audit_ignore=("forward_motion",)`；审计 agent 加了豁免后改 `()` |
| 6 | 20 个 seed 不能用 HANDOFF 的 23…180 | `HANDOFF.md:915` 列表在 `SPLIT_SALT=uavrgb.geom.v5` 下只有 1 个 test | 第一组扫 test、其余回传 `attempted_seeds`；要固定列表就填 `eval_seeds` |
| 7 | 旧 X3 学生（`t6-full-1640/run/ckpt_best.bin`）在 E1 动力学下评还是在它自己的 x3_velocity 下评 | 用户要的是「旧 X3 学生」做对照组；草稿按「同一协议四组」做 | 要加第五组「旧学生 + x3_velocity + X3 模板」，复制 `EVAL_GROUPS` 一行并给 `vehicle_flags` 加个 override 参数（约 6 行） |
| 8 | `teacher` 策略在 cf_sitl 下能不能跑 | `eval_closed_loop.py` 的 `--policy teacher` 走 `collect.py --scene-meta` 的 A* 专家，C 报告 §6 行 114 说 cf_sitl 档要「关掉卡滞判定」，专家在新执行器下的表现没数据 | 放在第一组：它失败整条线立即停，省掉后面几小时 |
| 9 | 300 集的 train/val/test 比例不可控 | split 由几何哈希定（`collect_parallel.py:21-24`），300 个 seed 大约 80/10/10 但有波动；`preprocess.py` 要求三个 split 都非空 | collect 步把三个 split 的集数打进日志；为空时 preprocess 会报错停下 |
| 10 | `.venv-tools` 还是 `$UAV_RGB_PYTHON` | `PY = sys.executable`，谁启动就用谁；docstring 写的是 `$UAV_RGB_PYTHON`（`env.sh` 导出） | 用错解释器会在 preprocess 缺 PIL 时停，不会污染数据 |

### 4.5 草稿依赖的外部接口（这些变了草稿才要跟着改）

| 接口 | 位置 | 草稿用到的 |
|---|---|---|
| `collection_manifest.json` | `collect_parallel.py:692-694` | `splits`、`episodes_requested`、`episodes_ok` |
| `audit.json` | `audit_dataset.py:992-1005` | `checks[].name/.status` |
| `eval.json` | `eval_closed_loop.py:766-817` | `attempted_seeds`、`success_rate`、`requested_success_rate`、`collision_rate`、`stuck_rate`、`false_forward_rate`、`episodes_done`、`failed_count`、`skipped_count`、`success_criterion`、`dynamics`、`hold_s`、`steps`、`ood`、`wilson_ci.success`（全部在 `demo-baseline-20261106/gazebo-l1v4-student/eval.json` 里实测存在） |
| 退出码 | `collect_parallel.py:773`、`audit_dataset.py:1019`、`eval_closed_loop.py:857` | 见 4.1 |
| 非空目录拒绝 | `collect_parallel.py:524-525`、`src/run_owner.cpp:42` | collect/train 步进入前先自检，给出可操作的报错 |
| 训练 flag | `src/train_main.cpp:46-77` | `--data --out --epochs --batch --lr --seed --class-weights --drop-collision --text-dim --vision-only` |
| `--vision-only` 规则 | `src/trainer.cpp:370-373` | 数据带 `instructions.jsonl` 而 `text_dim=0` 时必须显式传 |

## 5. 顺手发现的可读性问题（会直接导致改错参数的那种）

函数长度用 ast 统计（`end_lineno - lineno + 1`），文件行数用 `wc -l`，都在快照上跑。

### 5.1 超过 300 行的函数：改一个默认值要在一个函数里翻几百行

| 函数 | 位置 | 行数 | 为什么会改错 |
|---|---|---|---|
| `collect.py` `main` | `sim/tools/collect.py:1117-1579` | 463 | argparse（:1118-1150）、cf_sil 校验（:1201-1217）、规划器选择（:1273-1289）、动力学实例化（:1306）、manifest 字段（:1517）全在一个函数里。`--hold` 默认 1.0 在 :1121，cf_stab 对 hold 0.2 的硬要求在 :1183，隔 60 行，改一个漏一个 |
| `collect.py` `collect_one` | `sim/tools/collect.py:714-1035` | 322 | 采集主循环；`stop_radius or 0.5` 这种默认值埋在 :1114 和 :1386 两处函数体里 |
| `collect_parallel.py` `main` | `sim/tools/collect_parallel.py:447-777` | 331 | argparse :450-488 → `window_from` 二次解析 :490-491 → 组合校验 :505-521 → 调度 → manifest :680-742。`--steps` 的 help 把常量名写死（:454「默认见 split_manifest.DEFAULT_STEPS=40」），常量挪位置 help 就说谎 |
| `eval_closed_loop.py` `main` | `sim/tools/eval_closed_loop.py:547-857` | 311 | 同上结构；`--hold` 默认 1.0（:563）与新 Teacher 的 hold 0.2 硬校验（:670-673）相隔 110 行 |
| `audit_dataset.py` `audit` | `sim/tools/audit_dataset.py:369-923` | 555 | 十几个检查串成一段；判据常量一半在文件顶（:79-85），一半在函数体中间（`STAB_SKIP` :608），改豁免规则（cf_sitl 要用）得先读完 240 行找到 :604 |
| 次级 | `sim/3dgs/collect_3dgs.py:131-295` `run_episode` 165 行；`tools/preprocess.py:86-250` `load_episode` 165 行 | | 可读但没到改错参数的程度 |

不拆函数（拆了 `test_*.py` 大概率要跟着动）。§3 的改法——默认值全部上提到文件顶 `CFG`/常量区——已经让「改参数」不再需要进这些函数；函数长度本身留着。

### 5.2 一个文件干不相关的事

- `sim/tools/collect.py`（1583 行）同时是：Gazebo 采集器、闭环评测的执行器（`--policy student --stop-on-collision`，:1150 附近）、cf_sil 的参数辨识入口（`--flight-log/--flight-log-csv` :1139-1141，:1213-1217 还有它们的互斥校验）。改采集默认值的人要在辨识用的 flag 之间找 `--hold`。
- `ui/studio.py`（3060 行）里三处各自手拼 collect/eval 命令：训练配置 :1333-1340、live demo :2266-2269、手动评测 :2412-2415。三处对 `--dynamics/--hold/--window-from` 的处理各不相同，C 报告的 N2（手动评测按 X3 档跑 cf_stab 模型且 `ood=false`）就是这么来的。
- `sim/tools/split_manifest.py:36-39`：一个管「几何哈希 → split」的模块里放了采集默认步数 `DEFAULT_STEPS = 40`，注释说「collect.py / collect_parallel.py 共用，改一处即可」，但 `eval_closed_loop.py:562` 的 80 和 `scene_gen.py:16-18` 的 80 都不跟它走。§3.2 把它搬回 `collect.py` 顶部。

### 5.3 注释 / 文档与代码不一致（照注释改会改错）

| 位置 | 写的 | 实际 | 后果 |
|---|---|---|---|
| `sim/tools/scene_gen.py:16-18` | 「实测每步仅前进 0.105 m，穿完要 68–76 步；默认 --steps 因此是 80」 | 现行默认 40、每步约 0.27 m（`split_manifest.py:36-39`） | 读 scene_gen 的人会把 steps 改回 80 |
| `docs/HANDOFF.md:898-899` | 「并行采集（必须 DISPLAY=:20）」且命令只带 `DISPLAY` / `XDG_RUNTIME_DIR` | 代码默认 `UAVRGB_RENDER_BACKEND=gpu`（`collect_parallel.py:93`）走 EGL，不读 DISPLAY；本容器要显式 `software` 才用得上 DISPLAY | 照抄命令在本容器段错误（BRIEF §2） |
| `docs/HANDOFF.md:914-915` | 20 个「冻结 test seed」 | 在 `SPLIT_SALT=uavrgb.geom.v5` 下 17 train / 2 val / 1 test | `--split test --seeds …` 会把 19 个跳掉，只剩 1 集 |
| `sim/tools/audit_dataset.py:79` | `SPEED_FWD = 0.3  # 与 collect.py 一致` | 靠人手同步；`collect.py:97` 改了这里不会跟 | `forward_motion` 判据（:619）悄悄失效 |
| `sim/tools/collect_parallel.py:454` | help「默认见 split_manifest.DEFAULT_STEPS=40」 | 常量值硬编码进 help 字符串 | 改常量不改 help，`--help` 就报错数 |
| `sim/tools/collect_parallel.py:463`、`eval_closed_loop.py:571`、`collect.py:1131` | 「cf_stab 建议配 --hold 0.1~0.2」出现三次 | `--hold` 默认仍是 1.0（:563 / :1121），没有任何地方按 dynamics 换默认值 | 用 cf_stab 的人要自己记得改 hold，忘了就得到 ood 数据 |

### 5.4 误导性名字

- `ui/studio.py:62-63`：`ACTION_NAMES` 和 `ACTIONS` 是同一个元组写两遍，读者会以为一个是显示名一个是枚举值；全仓库这个四元组另有 9 份拷贝（§1.4）。
- `cf_sil`（`collect.py:1201` 现有档位、`--cf-sil-params`）与本轮要加的 `cf_sitl` 只差一个字母。`test_collect.py:538-542` 还断言 `cf_sil` 不在 `collect_parallel.DYNAMICS_CHOICES` 里——新档位的 help 字符串里要写清「不是 cf_sil」，否则 grep / tab 补全 / 口头交流都会串。
- `--window-from` 的 argparse `default=None`（`collect_parallel.py:488-491`）：`--help` 文本写「observation=默认」，真正的默认在 `main` 里按 task 分支（campus → dispatch）。`eval_closed_loop.py` 同一模式。用 campus 任务的人看 `--help` 以为是 observation，实际是 dispatch，两者数字按规定不能混报。
- `split_manifest.DEFAULT_STEPS`：名字像 split 的参数，实际是采集步数（见 5.2）。

### 5.5 直接会踩的坑

- `sim/tools/smoke_campus.py:28-30`：手写的 `SimpleNamespace(steps, hold, window_from, policy, save_depth, sim_timeout, episode_timeout)` 没有 `dynamics` / `teacher_planner`，而 `collect_parallel._popen_collect` 读 `args.dynamics`（:298）和 `args.teacher_planner`（:301）→ 一跑就 `AttributeError`。这是「参数不集中」的直接代价：`_popen_collect` 加一个字段，所有手拼 Namespace 的调用方都要跟。§3.1 的 `CFG` dict 可以直接 `SimpleNamespace(**CFG, …)` 补齐。
- `sim/3dgs/collect_3dgs.py:26-29` 相对导入：`python sim/3dgs/collect_3dgs.py --help` 直接 `ImportError`，只能 `python -m sim.3dgs.collect_3dgs`；脚本自己的 docstring 没写，只有 `docs/HANDOFF_3DGS_20261105.md` 写了。`eval_3dgs.py` 同。

## 没看的部分

- `ui/studio.py` 只看了 :62-63、:1333-1340、:2266-2269、:2412-2415 四段，其余 3000 行没读；`ui/monitor.py` 没看。
- `sim/tools/audit_dataset.py` 只看了常量区、`forward_motion` 段（:600-625）和 `main`；其余 check 的判据没逐条核。
- `sim/tools/cf_sil.py`、`sim/tools/record_views.py`、`rl_teacher/core.py`、`sim/3dgs/renderer.py`/`dynamics.py` 内部没看；只看了它们被入口调用的方式。
- C++ 只看了 `src/train_main.cpp`、`trainer.hpp` 的默认值、`trainer.cpp:370-373`、`run_owner.cpp:42`；trainer 内部没看。
- 没跑任何 `test_*.py`（§3 说的「测试不动」是按测试源码里引用的名字推断的，不是跑出来的）；没跑 `teacher/train.py --help`（.venv-tools 无 torch）；没起 Gazebo、没用 GPU。
- 主仓库正在被 4 个 agent 改的内容完全没看；本报告全部 `file:line` 相对 HEAD 9d764d9 快照。
- 流水线草稿只做了 `py_compile` + `--dry-run`，没有在真实数据上跑过任何一步。

