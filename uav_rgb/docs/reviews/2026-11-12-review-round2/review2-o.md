# 第二轮交叉审核报告：角度 O（文档与代码的一致性）

- 审核对象：主仓库 `/root/claude-code-backup-20261028/code/uav_rgb`（开工时 `main` 为 `ada0ca5`，落盘前复核至 `2798e4a`）；原型线工作树 `/root/claude-code-backup-20261028/code/wt/p12-e2e-p0`（`1e27bc4`）；清理对账分支 `p12-cleanup`（`5f157a6`）。
- 起止时刻：2026-11-12 08:00 CST 开工（08:20 完成代码与命令行核对后会话中断，09:21 CST 续写至 09:32 CST 收工）。
- 审核范围：`README.md`、`CLAUDE.md`、`docs/HANDOFF.md`、`docs/HANDOFF_3DGS_20261105.md`、`docs/EVAL_STANDARD.md`、`docs/ENGINEERING_RULES.md`、`docs/ROADMAP.md`、`tools/DATASET_FORMAT.md`、`docs/uart_action.md`、`proto/README.md`（及入口直接指向的 `sim/README.md`、`docs/TASKS.md`）对照生产代码。

## 1. 结论

代码本身在 11-10 至 11-12 经历了密集的口径收紧与工具改造（固件在环 E1 载具、走廊边界、固定派发延迟、按几何聚类检验、3DGS 场景划分门禁、交付权重校准、PyTorch 原型线），但主仓库的入口文档大面积停留在 10-30 至 11-05，导致新读者或新 agent 照文档操作会在多处直接报错、跑错口径或破坏环境。最要紧的三件事：第一，`docs/HANDOFF_3DGS_20261105.md` 的第 4、5、6 步命令均跑不通，其中第 4 步还暴露出一个生产代码缺陷（`tools/train_student_with_video.py` 与 `ui/studio.py` 共用的 `build_eval_command` 调 `eval_3dgs` 时漏传 `--hold` 或 `--data`，提交 `8399e32` 补 `--allow-train-scene` 时因单测只查 `--dry-run` 字符串而未发现，真跑必以 `rc=2` 失败）；第二，`CLAUDE.md` 同文并存新旧两个互斥的 GPU UUID（照第 17、22 行钉 `GPU-dbc6435f-…` 会让 CUDA 看不到任何设备），且 `CLAUDE.md`、`HANDOFF.md`、`proto/README.md` 的多条命令用裸 `python`/`python3`、相对 `.venv-tools` 或写往仅剩 2.8 GB 的根分区 `/tmp`；第三，SITL 默认二进制路径与数据盘清理存在耦合隐患（`collect_parallel.py` 与 `sim/3dgs/dynamics.py` 默认找 `tmp/build-cf-sitl`，不含公共规则冻结的 `tmp/build-cf-sitl-lib31`，清理数据盘若删掉前者会让 Gazebo 与 3DGS 双线断链），同时 `EVAL_STANDARD.md` 与 `DATASET_FORMAT.md` 漏记了走廊判据、停顿失败判据、`split_policy=geometry_gid` 及 C++ meta 子串扫描约束。

---

## 2. 发现清单（按严重程度排序）

### O-1：3DGS 入口文档第 4、5、6 步命令全部报错退出，且牵出 `train_student_with_video.py` 漏传 `--hold`/`--data` 的生产缺陷

- **位置**：`docs/HANDOFF_3DGS_20261105.md:25-26,95-97,127-143`；生产代码 `tools/train_student_with_video.py:165-178,222`（含提交 `8399e32`）、`sim/3dgs/eval_3dgs.py:129-137,202`、`tools/splat_to_ply.py:141-146`。
- **现象**：
  1. **第 4 步训练加自动录像（生产代码缺陷）**：`HANDOFF_3DGS_20261105.md:127-130` 调 `.venv-tools/bin/python tools/train_student_with_video.py --data .../dataset.uavr --out .../train ...`。首先，`train_student_with_video.py:222` 默认从 `REPO_ROOT / "src" / "build"` 找 `uavrgb_train`（与 `README.md:38` 全仓统一 `build/check` 矛盾，从零克隆未设 `UAV_RGB_BUILD_DIR` 直接 `rc=2`）。其次，即使指定了 `UAV_RGB_BUILD_DIR`，`build_eval_command`（`:171-177`，`ui/studio.py` 共用）构造的 `sim.3dgs.eval_3dgs` 命令行只传了 `--scene`、`--out`、`--ckpt`、`--episodes 1`、`--record-video`、`--allow-train-scene`（`8399e32` 刚加），**既没传 `--hold` 也没传 `--data`**。真跑完训练后，`eval_3dgs.resolve_hold`（`eval_3dgs.py:129`）必定报错 `ValueError: 必须显式给 --hold，或 --data 指向采集/数据集以读取 hold_s`（`rc=2`）。`tools/test_train_student_with_video.py:214-234` 只断言了 `--dry-run` 打印的字符串，漏检了这一必败路径。
  2. **第 5 步单独评测（三处必败）**：`HANDOFF_3DGS_20261105.md:133-138` 的 `eval_3dgs` 命令：① 同样没传 `--hold` 或 `--data`（`rc=2`）；② `--infer-bin src/build/uavrgb_infer` 指向已废弃的旧目录；③ 审核期间提交 `e32714b` 在 `:132` 注释里补了 `--allow-train-scene` 的文字说明，但代码块里的命令仍未带该参数，而示例场景 `callinstrasse/scene.json`（`gid=86b0f5cb953fbb03`）与 `make-synthetic --seed 0`（`gid=4213c4a93d807214`）实测均落在 `train` 分区，照敲仍 `rc=2`。此外，文档 `:25-26,95-97` 写 `eval_3dgs --record-video` 输出 `.mp4`，实际代码（`eval_3dgs.py:202,320`）输出 `.webm`（VP8）。
  3. **第 6 步 `.splat` 转换（已删参数）**：`HANDOFF_3DGS_20261105.md:141-143` 写 `splat_to_ply.py in.splat out.ply --flip-height --scale 2.0 --z-offset 0.516`。`--flip-height` 已被删除，`argparse` 直接返回 `rc=2`。
- **为什么是缺陷及影响**：`HANDOFF_3DGS_20261105.md` 自称「这条线的唯一入口」，但后三步核心命令无一能跑通；`train_student_with_video.py` 的漏参还会让桌面台 `ui/studio.py` 的 3DGS 训练后自动录像静默失败（被包装脚本降级为警告）。
- **怎样复现或验证**：
  - `nice -n 15 .venv-tools/bin/python tools/splat_to_ply.py in.splat out.ply --flip-height` → 输出 `error: --flip-height 已删除：它只翻 z，是行列式 -1 的镜像。y-down 的源用 --up -y`，`rc=2`。
  - `nice -n 15 .venv-tools/bin/python -c 'import tools.train_student_with_video as t; print(t.build_eval_command("/py", "/ev", "/ck", "/out", {"renderer":"3dgs","data":"/d/dataset.uavr","hold_s":1.0}))'` → 返回列表不含 `--data` 与 `--hold`，传给 `eval_3dgs` 即触发 `resolve_hold` 退出 `rc=2`。
- **建议的最小修法**：
  1. `tools/train_student_with_video.py`：默认构建目录改为 `REPO_ROOT / "build" / "check"`（与 `student.py:27`、`average_ckpt.py:32` 对齐）；`build_eval_command` 的 3DGS 分支在 `data` 存在时追加 `["--data", str(data)]`（否则若有 `hold_s` 追加 `["--hold", repr(float(params["hold_s"]))]`），单测真调 `eval_3dgs` 参数解析。
  2. `docs/HANDOFF_3DGS_20261105.md`：第 5 步命令补 `--data $DATA_ROOT/uav_rgb/run/ds/dataset.uavr --allow-train-scene` 并删掉 `--infer-bin src/build/uavrgb_infer`（默认走 `build/check`）；第 6 步 `--flip-height` 改为 `--up=-y`（及 `:30,:173` 同步）；`.mp4` 改为 `.webm`。
- **状态**：已复现（其中第 5 步 `--allow-train-scene` 注释由 `e32714b` 审核期间部分修补，其余未修）。
- **归属建议**：生产代码部分并入卡 `W1-15c`，文档修补并入入口文档统一整理卡。

---

### O-2：`CLAUDE.md` 同文并存互斥的 GPU UUID，且三份主文档的「无例外禁 PyTorch」与 `proto/` 原型线冲突

- **位置**：`CLAUDE.md:11,17,20,22,26,115`；`docs/HANDOFF.md:40,56`；`docs/ENGINEERING_RULES.md:23`；对照 `proto/README.md`（分支 `p12-e2e-p0` `1e27bc4`）与 `docs/briefs/2026-11-10-phase12-common.md:50,82`。
- **现象**：
  1. `CLAUDE.md:17` 写「所有用 CUDA 的进程都带 `CUDA_VISIBLE_DEVICES=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70`」，`:20` 写「`2026-11-11 19:23` 起本机只有一张卡 `GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740`，`GPU-dbc6435f-…` 不在机器上」，紧接着 `:22` 的英文正文又写「The machine has two NVIDIA GeForce RTX 2080 Ti ... since 2026-11-10; uav_rgb may use only `GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70`」。
  2. `CLAUDE.md:11,26,115`、`docs/HANDOFF.md:40,56`、`docs/ENGINEERING_RULES.md:23` 均把「禁止 PyTorch」列为无例外的不可违反红线（且 `CLAUDE.md:26` 称本机未安装 PyTorch），未反映 2026-11-12 用户批准的原型线规则（允许在 `proto/` 目录下借用 `$D/teacher-venv` 的 PyTorch 做结构探索，主线 `src/ sim/ tools/` 不 import，结果标 `proto`）。
- **为什么是缺陷及影响**：`CLAUDE.md` 是所有会话自动加载的系统级指令。英文段（`:22`）位于中文补丁段（`:20`）之后，新会话或新脚本若按 `:17` 或 `:22` 钉 `GPU-dbc6435f-…`，`uavrgb_train`、`uavrgb_infer` 与 `gsplat` 会直接报无可用 CUDA 设备；而未读 `phase12-common.md` 的审查或开发会话会把 `proto/` 与 `$D/teacher-venv` 误判为违规产物。
- **怎样复现或验证**：实测 `nvidia-smi -L` 仅输出 `GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740)`。
- **建议的最小修法**：删掉 `CLAUDE.md:17` 与 `:22` 中已失效的双卡指令，只保留「每次开工以 `nvidia-smi -L` 实测为准，当前单卡 `GPU-b544b42e-…`」一条；在 `CLAUDE.md:11` 与 `ENGINEERING_RULES.md:23` 补半句「交付训练与推理栈禁 PyTorch；结构探索仅限 `proto/` 原型线（不进主线，数字标 `proto`）」。
- **状态**：已复现。
- **归属建议**：主会话合并 `p12-e2e-p0` 前后直接修 `CLAUDE.md` 与 `ENGINEERING_RULES.md`。

---

### O-3：`CLAUDE.md`、`HANDOFF.md`、`README.md` 的核心采集/训练/评测命令用裸 `python3`、写根分区 `/tmp`、跑旧载具 `x3` 且产出空 `val`

- **位置**：`CLAUDE.md:75-87`；`docs/HANDOFF.md:364-392,589-595,931-948`；`README.md:58-69`；`sim/README.md:148-151,249-252,306-308`。
- **现象**：
  1. **裸 `python3` 缺依赖报错**：`env.sh` 只导出 `UAV_RGB_PYTHON="$UAV_RGB_ROOT/.venv-tools/bin/python"`，不改 `PATH`。未手动 `activate` `.venv-tools` 时，`CLAUDE.md:75,80` 与 `HANDOFF.md:931-948` 的裸 `python3` 会落到 `/opt/conda/bin/python` 或 `/usr/bin/python3`。实测 `/usr/bin/python3 -c "import PIL"` 报 `ModuleNotFoundError: No module named 'PIL'`，`tools/preprocess.py`（`:101`）与 `sim/tools/student.py`（`:21`）立即退出；而 `/opt/conda/bin/python -c "import torch"` 报 `ImportError: libcudnn.so.9`。
  2. **写往根分区 `/tmp`**：`CLAUDE.md:76,80,84` 与 `HANDOFF.md:932-947` 把采集、打包与训练输出指向 `/tmp/dataset`、`/tmp/ds`、`/tmp/run1`（`sim/README.md:308` 指 `/tmp/e1-demo`）。本机根分区仅剩 2.8 GB（91% 已用），直接违反 `CLAUDE.md:15,126` 与 `ENGINEERING_RULES.md:27`「禁止写根分区，一律放 `$DATA_ROOT/uav_rgb`」。
  3. **默认旧载具与旧评测口径**：`CLAUDE.md:75`、`README.md:60`、`HANDOFF.md:931,944` 不带 `--dynamics cf_sitl --vehicle e1_mini`，默认跑已淘汰的 `x3_velocity`；`HANDOFF.md:944-947` 的 `eval_closed_loop.py` 示例不带 `--corridor-half-width 3.0` 与 `--action-latency 0.20`，且直接评 `ckpt_best.bin` 而非 `tools/average_ckpt.py` 交付权重，产出的数字不能进第十二阶段任何对比表。
  4. **单集/少集示例 `val=0` 导致无 `ckpt_best.bin`**：实测 `scene_gen.py` 下 `README.md:61` 的 `seed 42` 属于 `train`，`CLAUDE.md:76` 的 `seed 0..3` 属于 `train, test, train, train`（且 `CLAUDE.md:80` 只打包 `/tmp/dataset/train/*/`）。打包后 `splits.val == 0`；`uavrgb_train` 回退 `dev_is_train=true` 并拒绝写出 `ckpt_best.bin`（`src/trainer.cpp:632`），后续照跑推理或评测 `--ckpt .../ckpt_best.bin` 必报文件不存在。
- **为什么是缺陷及影响**：新人按主入口文档从头敲一遍，每一步都会踩中缺包、写满根盘、跑错载具或找不到 `ckpt_best.bin`。
- **怎样复现或验证**：
  - `/usr/bin/python3 tools/preprocess.py --out /tmp/x /tmp/y` → `需要 numpy/Pillow：No module named 'PIL'（用 .venv-tools/bin/python 运行）`。
  - 对 `seed 42` 与 `seed 0..3` 调 `scene_gen.py` + `split_manifest.split_for_scene`（见草稿 `notes.md` 第 9 条）→ 无一落在 `val`。
- **建议的最小修法**：把 `CLAUDE.md` 与 `HANDOFF.md` 命令里的裸 `python3` 统一改成 `"$UAV_RGB_PYTHON"`，`/tmp/...` 改成 `"$DATA_ROOT/uav_rgb/tmp/..."`；Gazebo 主线示例加上 `--dynamics cf_sitl --vehicle e1_mini --corridor-half-width 3.0 --action-latency 0.20` 与 `average_ckpt.py` 步骤，并注明单 `train` split 冒烟只产 `ckpt.bin`、不产 `ckpt_best.bin`。
- **状态**：已复现。
- **归属建议**：新立入口文档同步卡（或主会话收尾时统一改）。

---

### O-4：`proto/README.md` 的裸 `python` 与相对 `.venv-tools` 在 worktree 下失效，且守护进程停法存在重启后误杀风险

- **位置**：工作树 `/root/claude-code-backup-20261028/code/wt/p12-e2e-p0`（`1e27bc4`）的 `proto/README.md:65-73,104,122`、`proto/runs/common.sh:5,32`。
- **现象**：
  1. `proto/README.md:65-66` 写 `python proto/export.py --run <run 目录>`。实测本机 `which python` 指向 `/opt/conda/bin/python`，其 `import torch` 因缺少 `libcudnn.so.9` 直接抛 `ImportError`；`export.py` 在 `load_pt` 导入 `torch` 时崩溃。
  2. `proto/README.md:71` 写 `.venv-tools/bin/python tools/average_ckpt.py ...`，`:122` 写 `cd proto && ../.venv-tools/bin/python -m unittest test_ckpt_io`，`:73` 写裸 `uavrgb_train --eval-only ...`。原型线代码在独立 worktree `/root/claude-code-backup-20261028/code/wt/p12-e2e-p0` 中运行，实测该目录下没有 `.venv-tools`（只在主仓库 `$R/.venv-tools` 存在），`uavrgb_train` 也不在 `PATH` 中，照敲会报 `No such file or directory` 或 `command not found`。
  3. `proto/README.md:104` 教读者用 `kill -- -$(cat $OUT/run.pid)` 停止守护进程，配套的 `proto/runs/common.sh:32` 也只写 `echo $$ > "$OUT/run.pid"`。按交叉审核 E-4 与公共规则 §6（及刚合入 `main` 的 `f2c1107` `tools/stop_daemon.sh`），容器频繁重启后 PID 会被重分给无关进程，不校验进程启动时刻与命令行直接 `kill -- -PGID` 会误杀别的任务。
- **为什么是缺陷及影响**：卡 `E2E-P1` 及其后续执行者在原型线 worktree 里照 `proto/README.md` 导出权重、校准或停进程时，会直接触发解释器报错或误杀进程组。
- **怎样复现或验证**：
  - `nice -n 15 /opt/conda/bin/python -c "import torch"` → `ImportError: libcudnn.so.9: cannot open shared object file`。
  - `ls /root/claude-code-backup-20261028/code/wt/p12-e2e-p0/.venv-tools` → `No such file or directory`。
- **建议的最小修法**：`proto/README.md` 里所有需要 `torch` 的命令统一写 `$D/teacher-venv/bin/python`，主线工具统一写 `$R/.venv-tools/bin/python`，C++ 二进制写 `$D/tmp/build-check-main-1111/uavrgb_train`（或 `$UAV_RGB_BUILD_DIR/uavrgb_train`）；`proto/runs/common.sh` 接入 `tools/exp_lib.sh` 的三字段 `run.pid` 格式，文档停进程命令改为 `bash $R/tools/stop_daemon.sh $OUT`。
- **状态**：已复现。
- **归属建议**：并入卡 `P1b`（原型线的更正）。

---

### O-5：`sim/README.md` 未交代 `CF_SITL_BUILD_GZ=OFF`，且代码默认 SITL 目录不含冻结的 `build-cf-sitl-lib31`（与数据盘清理耦合）

- **位置**：`sim/README.md:300-309`；`README.md:18-22`；对照 `sim/cf_sitl/CMakeLists.txt:22,56`、`sim/tools/collect_parallel.py:64-68`、`sim/3dgs/dynamics.py:36-38`、`tools/check.sh:70-80`、`docs/briefs/2026-11-10-phase12-common.md:86`、`docs/briefs/2026-11-12-cleanup-brief.md:14`。
- **现象**：
  1. **构建步骤缺失开关**：`README.md:18-22` 让读者设 `CF_DRONE_DIR` 后跑 `bash tools/check.sh`，但 `check.sh:73-75` 构建 `sim/cf_sitl` 时不传 `-DCF_SITL_BUILD_GZ=ON`，而 `sim/cf_sitl/CMakeLists.txt:22` 默认 `option(CF_SITL_BUILD_GZ ... OFF)`，因此跑完 `check.sh` 根本不会产出 Gazebo 插件 `libCfSitlSystem.so`。`sim/README.md:300-304` 虽提到再按 `cf_sitl/gz/README.md` 编插件，但没说明顶层加 `-DCF_SITL_BUILD_GZ=ON` 即可一次编齐。
  2. **默认搜索路径与清理保护名单错位**：公共规则 §4（`:86`）与清理任务书 §1（`:14`）保护的冻结 SITL 目录是 `$D/tmp/build-cf-sitl-lib31`（实测含 `cf_sitl_proc` 与 `gz/libCfSitlSystem.so`，`lib_version=31`）。然而主线代码里，Gazebo 侧 `collect_parallel.py:64-68` 的 `DEFAULT_PLUGIN_DIRS` 只列了 `$D/tmp/build-cf-sitl/gz`、`$D/tmp/build-cf-sitl`、`$D/tmp/agent-gz/build-gz-stub`；3DGS 侧 `sim/3dgs/dynamics.py:36-38` 的 `CF_SITL_PROC` 默认也是 `$D/tmp/build-cf-sitl/cf_sitl_proc`。两个默认路径都指向未列入清理任务书 §1 显式保留名单的 `$D/tmp/build-cf-sitl`（11-09 21:24 构建，早于第十二阶段分界线 11-10 19:00）。
- **为什么是缺陷及影响**：一是新人跑完 `check.sh` 后照 `sim/README.md:305` 不设 `UAVRGB_CF_SITL_PLUGIN_DIR` 直接跑 E1 采集，在干净环境里找不到 `libCfSitlSystem.so`；二是若本次数据盘清理把早于第十二阶段的 `$D/tmp/build-cf-sitl` 当作旧构建目录删掉，所有未显式设 `UAVRGB_CF_SITL_PLUGIN_DIR` 或 `CF_SITL_PROC` 的命令与单测都会立刻报错退出。
- **怎样复现或验证**：读 `collect_parallel.py:64-68` 与 `sim/3dgs/dynamics.py:36-38`；实测 `sha256sum` 显示 `build-cf-sitl/gz/libCfSitlSystem.so` 与 `build-cf-sitl-lib31/gz/libCfSitlSystem.so` 同为 `2811ed50df0d…`（`lib_version=31`），但代码默认只找前者。
- **建议的最小修法**：① 数据盘清理清单里必须把 `$D/tmp/build-cf-sitl` 标为「留」（或把 `build-cf-sitl-lib31` 加入 `DEFAULT_PLUGIN_DIRS` 首项及 `CF_SITL_PROC` 回退候选）；② `README.md` 与 `sim/README.md:300` 写明一键构建命令 `cmake -S sim/cf_sitl -B $D/tmp/build-cf-sitl -G Ninja -DCF_DRONE_DIR=$R/../CF-Drone -DCF_SITL_BUILD_GZ=ON`。
- **状态**：读代码与目录实测确认。
- **归属建议**：数据盘保留项立即同步给主会话审批 `cleanup` 清单时核对；默认搜索路径与文档说明并入 `W1-15c` 或文档整理卡。

---

### O-6：`docs/EVAL_STANDARD.md` 漏记第十二阶段主表的 5 项硬判据与交付口径，且残留默认 EGL 的错误说法

- **位置**：`docs/EVAL_STANDARD.md:92,100-108,125,338-433`；对照 `sim/tools/eval_closed_loop.py:58-70,442-528,770-788,1072-1145`（提交 `25e4b28`、`dd98f89`、`bbc6132`、`0d7f3b5`、`e40c693`）与 `tools/average_ckpt.py`（提交 `8fadc83`）。
- **现象**：`EVAL_STANDARD.md` §5.6 在 11-12 07:42（`0d7f3b5`）刚更新了几何聚类符号翻转检验与 `stats_report.py` 的协议一致性键，但整份约束文件漏记了以下已在 `main` 生效的主表规则：
  1. **走廊横向边界（`--corridor-half-width 3.0`）**：全文无一处出现 `corridor`。`:101` 与 §5.4 仍写 route_choice 成功只需「无碰撞且 `max_x ≥ goal_x`」，而现行主表口径 `sim09.v2.coast1s.corridor3m` 要求全程 `|pose_gt.y| ≤ 3.0 m`（`eval_closed_loop.py:608,644`），越界判 `corridor_violation=True`、`success=False`。
  2. **停顿步（stall）基础设施失败判据**：§5.5（`:366-369`）只写了 `late_dispatch` 步使该集进 `failed[]`；`e40c693`（W1-13）合入后，带 `--action-latency` 时相邻观测间隔 `obs_gap_s > (hold - 0.1) + 0.2 s` 的停顿集同样判基础设施失败（记入 `failed[]` 与顶层 `stall_episodes`，在请求集分母中计 0），每集新增 `safe_gap_s_median`、`safe_gap_s_max`、`obs_gap_s_max`、`stall_steps`；旧协议数据在新协议下评测另写顶层 `protocol_notes`。
  3. **长课程逐块指标与百米碰撞数**：W1-4（`dd98f89`）新增的 `course_base` 逐集 `blocks_passed`、顶层 `block_survival`（分母 `episodes_requested`）及全任务 `collisions_per_100m`（`eval_closed_loop.py:473-504,770-788`）完全未写入标准。
  4. **断帧监控字段**：W1-3（`bbc6132`）新增的逐集 `max_frame_gap_s` 与 `frame_gap_flag`（相邻 `rgb_ts` 差 `> 1.2 × hold`）未写入。
  5. **模型交付与渲染环境过时**：`:125` 仍写「best-checkpoint 只按 dev 宏 F1 选择」，未写明现行学生交付权重统一采用末 10 轮权重平均加 val 前进先验校准（`tools/average_ckpt.py`）；`:92` 仍写「当前默认 NVIDIA EGL」。
- **为什么是缺陷及影响**：`CLAUDE.md:143` 与 `HANDOFF.md:1045` 规定「报任何数字前先读 `EVAL_STANDARD.md`」。标准文件漏掉走廊判据、停顿失败判据和交付配方，会让后续写实验脚本或核对报表的 agent 漏传 `--corridor-half-width 3.0`、误交 `ckpt_best.bin` 或读不懂 `stall_episodes` 导致的成功数下降。
- **怎样复现或验证**：`grep -nE 'corridor|blocks_passed|block_survival|collisions_per_100m|stall_episodes|safe_gap_s|average_ckpt' docs/EVAL_STANDARD.md` 输出为空。
- **建议的最小修法**：在 `EVAL_STANDARD.md` §5.4 补 `--corridor-half-width`、`blocks_passed`/`block_survival`、`collisions_per_100m`、`max_frame_gap_s` 的定义与分母；在 §5.5 补 `stall_episodes`、`safe_gap_s`、`protocol_notes`（可随卡 `W1-13` 第三轮写时序时一并补齐）；在 §4 注明交付权重用 `average_ckpt.py`，把 `:92` 的默认 EGL 改为本容器软件 GL。
- **状态**：读代码与文档确认。
- **归属建议**：并入卡 `W1-13`（第三轮本来就要改 `EVAL_STANDARD.md`，见 `FIXLIST.md` G5）或文档整理卡。

---

### O-7：`tools/DATASET_FORMAT.md` 的 `split_policy` 合同与代码矛盾，且漏记最新过滤开关与 C++ 子串扫描硬约束

- **位置**：`tools/DATASET_FORMAT.md:58,69,89-95,125,204-205,240-245`；`docs/HANDOFF.md:800`；对照 `tools/preprocess.py:559-602,770-777,999-1005,1244-1320`（提交 `f383584`、`e40c693`）与 `src/dataset.cpp:142-210`。
- **现象**：
  1. **`split_policy` 合同过时**：`DATASET_FORMAT.md:89` 写「`split_policy` 恒为 `scene_id`」，`HANDOFF.md:800` 写「`.uavr` 的 C++ reader 硬性要求 `split_policy == "scene_id"`」。实际现行 `preprocess.py:582-601` 对带 `geometry_gid` 的柱阵/森林/3DGS 数据集统一写出 `"split_policy": "geometry_gid"` 与 `"split_scheme": "geometry_gid"`（校园写出 `"campus_map_v1"` 或 `"campus_assets_map_v1"`）；`src/dataset.cpp:193-209` 仅在旧包缺 `split_scheme` 时才要求 `split_policy == "scene_id"`，新包要求 `split_policy == split_scheme` 且属于四种受支持方案之一。
  2. **C++ meta 扫描器的非标准 JSON 约束未入格式规范**：`DATASET_FORMAT.md:58` 写「关键字段只允许追加」，`:125` 写「trainer 可用任意 JSON 库解析」。但 `src/dataset.cpp` 实际用朴素子串查找（`meta.find("\"episodes\"")` 与 `value_pos` 找第一次出现），不解析 JSON 树。`preprocess.py:770-777` 专门有防呆断言：`"episodes":` 在整段 meta JSON 中必须恰好出现 1 次，且 `"split_policy"`、`"split_scheme"`、`"split_salt"`、`"collision_flags"` 四个键必须出现在 `"episodes":` 数组之前（11-07 曾因嵌套字典含 `"episodes"` 键导致 C++ 读端报 `episodes[] 为空`）。任何外部脚本或原型导出工具若照文档用普通 `json.dumps(sort_keys=True)` 写 meta（`'collision_flags'` 字典序排在 `'episodes'` 前，但 `'split_*'` 排在 `'episodes'` 后），C++ 读端若在 `episodes[]` 项内遇到同名子串就会读错。
  3. **漏记新增打包过滤开关**：§6.2/§6.3 只描述了 `--drop-collision`；未记录 `--exclude-failed-teacher`（W1-7 `f383584`，写 `collision_filter.exclude_failed_teacher`）与 `--drop-late-and-stall`（W1-13 `e40c693`，丢弃 `late_dispatch=true` 或 `obs_gap_s` 超阈值的步，写 `collision_filter.drop_late_and_stall`，同样使下一步 `prev_index` 指向自身而 `prev_action` / `heading_code` 保留真实已执行历史）。
- **为什么是缺陷及影响**：若有新工具（例如卡 `PK1`、板端校准集导出器或 `proto/` 新数据生成脚本）按 `DATASET_FORMAT.md` 字面构造或校验 `.uavr` meta，要么写死 `split_policy="scene_id"` 触发 `split_policy 与 split_scheme 不一致` 报错，要么因 `sort_keys=True` 触发子串扫描错位。
- **怎样复现或验证**：读 `src/dataset.cpp:180-210` 与 `tools/preprocess.py:770-777`；检查任意现行 `.uavr` 的尾部 meta，`split_policy` 均为 `"geometry_gid"`。
- **建议的最小修法**：在 `DATASET_FORMAT.md` §1.4 更正 `split_policy` / `split_scheme` 的四种合法值，显式写出 C++ 子串扫描器的两条硬约束（`"episodes":` 唯一，四个顶层键必须在 `"episodes":` 之前）；在 §6.3 补一行 `--drop-late-and-stall` 与 `--exclude-failed-teacher`。
- **状态**：读代码确认。
- **归属建议**：并入卡 `PK1`（打包的健壮性）或文档整理卡。

---

### O-8：多份入口文档的「当前项目状态、唯一基线、目录结构、未修缺陷」已与代码事实相反

- **位置**：`CLAUDE.md:7,90,121-123`；`README.md:5,36`；`docs/HANDOFF.md:3-18,114-121,380-392,448-450,740-742,825-837`；`docs/ENGINEERING_RULES.md:29,85-93`；`sim/README.md:363-373`；`docs/TASKS.md:1-60`；对照 `p12-cleanup` 分支提交 `5f157a6`。
- **现象**：
  1. **项目阶段与唯一基线失真**：`CLAUDE.md:7` 与 `HANDOFF.md:114-121,146` 仍声明「T7 v1 observation（X3 载具，SR 0.90）是唯一 baseline」「当前主线：L1 v4（2026-11-01），只有这组数能当现状」；`README.md:5` 写「当前交付模型是单帧 RGB + 有限词表 FiLM」。它们均未指向 `docs/PLAN_2026-11-10_phase12.md` §10 与 `docs/briefs/2026-11-10-phase12-common.md`（现行主线是 E1 MINI `cf_sitl` + 条件历史码端到端模型）。
  2. **构建目录与渲染后端前后打架**：`HANDOFF.md:8-10` 顶部「当前状态」宣称默认仿真已改为 NVIDIA EGL 无界面渲染（本容器 EGL 必段错误）；`HANDOFF.md:17` 说统一用 `build/check`，同文件 `:380,385` 与 `:740-742` 却让读者去编和跑 `src/build/uavrgb_train`。另需注意：本机仓库根下残留着历史构建目录 `$R/build/check`（46 MB）与 `$R/src/build`（33 MB），若新人照 `CLAUDE.md:64` 或 `README.md:22` 不设 `UAV_RGB_BUILD_DIR=$D/tmp/...` 直接构建，会在仅剩 2.8 GB 的根分区内新建编译产物。
  3. **已修完的缺陷仍列在「未修 / 禁用」里**：
     - `CLAUDE.md:90` 写「传感器停更/陈旧位姿处理、学生闭环评测仍缺」（与同文件 `:7` 自相矛盾）；
     - `HANDOFF.md:448-450,831` 警告「`carry_resume_best` 跨数据集冻结 best、resume 静默继承旧 `class_weights`，下次用 `--resume` 前必须先修」，实际 `src/trainer.cpp:184-299`（v5 训练契约 113 B + `merge_resume_config` + 卡 C1a `45bb34c`）早已修完；
     - `ENGINEERING_RULES.md:93` 列「`collect_parallel` 对 SIGTERM 无清理」，实际 `collect_parallel.py:291`（`request_stop`）与 `eval_closed_loop.py:1445` 早已处理；
     - `sim/README.md:369-373` 末尾仍写「闭环评测还没有一次跑出可用的成绩：SR=0」；
     - `README.md:36` 与 `HANDOFF.md:75` 写 **CTest 30/30**、`audit_dataset.py` 11 或 18 条闸门，实际当前 `src/CMakeLists.txt` 注册了 **31 项** `add_test`（卡 T1 第二批 `b706b2f` 加了 `test_infer_layout`、`test_flip_augment`、`test_two_frame_prev_index`），`audit_dataset.py` 为 19 项（校园另有 5 项专用检查）。
  4. **与 `p12-cleanup` 的目录说明冲突**：清理分支提交 `5f157a6` 刚删掉了空占位目录 `data/`、`models/`、`logs/`、`third_party/`；一旦合入 `main`，`CLAUDE.md:121-123` 与 `ENGINEERING_RULES.md:29` 对这四个仓库内目录的描述即告失实。
- **为什么是缺陷及影响**：`CLAUDE.md:7` 与 `README.md:6,37` 均在首屏命令读者「接手先读 `docs/HANDOFF.md`，任务以 `docs/TASKS.md` 为准」，把读者直接引向早于第十一阶段的历史口径与错误警告。
- **怎样复现或验证**：逐行对照上述文档行号、`src/CMakeLists.txt` 的 `add_test` 计数（31 个）及 `git show 5f157a6 --stat`。
- **建议的最小修法**：在 `README.md`、`CLAUDE.md`、`docs/HANDOFF.md`、`docs/TASKS.md` 顶部显著置顶一段「2026-11-12 第十二阶段现行入口：先读 `docs/briefs/2026-11-10-phase12-common.md` 与 `docs/PLAN_2026-11-10_phase12.md` §0、§10，下方 11-05 及更早章节仅供历史溯源」；合并 `p12-cleanup` 时同步删改 `CLAUDE.md:121-123` 与 `ENGINEERING_RULES.md:29,85-93`；在构建示例前加 `export UAV_RGB_BUILD_DIR=$DATA_ROOT/uav_rgb/tmp/build-check`。
- **状态**：读代码与 git 记录确认。
- **归属建议**：主会话合并 `p12-cleanup` 时顺手更新顶部导引与目录说明。

---

### O-9：守护进程食谱已在审核期间修复，但四份入口文档的 GitHub 推送说明仍教手工 `rsync`，会破坏镜像游标与逐卡 diff

- **位置**：`docs/HANDOFF.md:471-475,784-787`；`docs/HANDOFF_3DGS_20261105.md:193-199`；`CLAUDE.md:140-141`；`docs/ENGINEERING_RULES.md:60-63`；对照 `tools/git_mirror_alvln.sh:13-44`、`tools/stop_daemon.sh`。
- **现象**：
  1. **守护进程起停食谱（审核期间已修）**：开工时（`ada0ca5`），`docs/HANDOFF.md:784-787` 教读者用 `> $OUT/run.log ... & echo $! > $OUT/run.pid` 起跑、用裸 `kill -- -$(cat $OUT/run.pid)` 停进程（存在第二实例截断日志及重启后误杀进程组的缺陷）。审核期间主会话提交 `e32714b` 已将该处改为 `>> $OUT/run.log`（由脚本自写含启动时刻的 `run.pid`）与 `bash tools/stop_daemon.sh $OUT`。但 `CLAUDE.md:92` 仍只写了 `kill -- -PGID`，未提及 `tools/stop_daemon.sh` 与实验模板 `experiments/_template/run.sh`。
  2. **GitHub 镜像食谱全部过时（未修）**：`CLAUDE.md:140`、`HANDOFF.md:471-475`、`HANDOFF_3DGS_20261105.md:193-199`、`ENGINEERING_RULES.md:61` 四处全部指导用手工 `rsync -a --delete ... uav_rgb/ ALVLN/uav_rgb/` 再 `git commit && git push`，无一处提到现行唯一镜像脚本 `bash tools/git_mirror_alvln.sh`。`git_mirror_alvln.sh` 依赖 `ALVLN/uav_rgb/.mirror_src_commit` 记录上次同步到的 `uav_rgb` 提交号，并沿 `--first-parent` 把每张卡的合并提交一对一重放成单个 GitHub 提交。
- **为什么是缺陷及影响**：任何人或主会话若照四份入口文档手工 `rsync` 提交到 `../ALVLN`，一来把多个合并压成一团，违反用户「方便逐条看 diff」的要求；二来不更新 `.mirror_src_commit`，下次跑 `tools/git_mirror_alvln.sh` 会把已同步的提交重放一遍或因工作树/历史不一致报错。
- **怎样复现或验证**：读 `tools/git_mirror_alvln.sh:13-44` 对照 `HANDOFF_3DGS_20261105.md:193-199` 与 `CLAUDE.md:140`。
- **建议的最小修法**：把四份入口文档的推送说明统一替换为「推送由用户或主会话执行 `bash tools/git_mirror_alvln.sh`（维护 `.mirror_src_commit` 游标，勿手工 `rsync`）」；`CLAUDE.md:92` 补一句 `tools/stop_daemon.sh`。
- **状态**：第 1 款审核期间已修（`e32714b`），第 2 款读代码确认未修。
- **归属建议**：文档整理卡（或主会话直接改）。

---

### O-10：`docs/uart_action.md` 编码细节与代码完全一致，但未提示「10 Hz 心跳动作码 + 300 ms TTL」与仿真步时序及条件码的冲突

- **位置**：`docs/uart_action.md:69-77,85-96,124-140`；对照 `tools/uart_action.py:206-236,330-357`、`config/uart_protocol.example.json`、`sim/tools/nav_cond.py`。
- **现象**：
  1. 实跑核对确认 `tools/uart_action.py` 的 CLI 参数、CRC-8/CRC-16、默认 9 字节帧布局及示例输出（`--dry-run --action 1` 输出 `aa 55 01 00 01 00 2c 01 39`）与 `docs/uart_action.md` 逐字一致。
  2. 但时序语义存在两处断层（与第一轮审核 G2、G5 呼应）：① `uart_action.py:330-356` 在等待 stdin 下一行动作期间，每 `1 / heartbeat_hz = 100 ms` 发送一帧 `heartbeat=1, action=heartbeat_action(0=SAFE_NEUTRAL), ttl_ms=300`；文档 `:95,134` 未规定固件收到 `heartbeat=1` 时是「仅续期保活、保持上一动作」还是「切回 `SAFE_NEUTRAL`」，若固件团队按帧内 `action=0` 执行，1.0 s 的动作窗口在 100 ms 后即被心跳掐断；即便忽略心跳的 `action`，固定 `ttl_ms=300` 也无法表达仿真里的 0.8 s 动作有效窗（或持续到下次派发）。② 文档 `:139` 虽注明无 ACK，但未提醒第十二阶段端到端模型的条件输入（航向码）若在板端按「已发送动作」计数，一旦串口丢帧或固件超时拒收，整集条件码会永久偏移。
- **为什么是缺陷及影响**：硬件/固件团队将来对照 `docs/uart_action.md` §5 制定 `real-drone/uart_protocol.md` 时，若照搬心跳与 TTL 语义或省略执行回执，真机动作持续时长与条件输入会与仿真完全脱节。
- **怎样复现或验证**：跑 `printf '1\n' | .venv-tools/bin/python tools/uart_action.py --dry-run --frames 4` 可见第 1 帧发 `action=1` 后，100 ms 起连续发 `action=0, heartbeat=1` 帧。
- **建议的最小修法**：在 `docs/uart_action.md` §5 对接清单新增两条硬性要求：① 明确心跳保活与动作持续时长（`until` 或单独保活不改当前动作）的语义，匹配仿真步周期；② 若板端策略使用动作历史条件码，协议必须回传飞控实际接纳的动作序号或偏航角估计（对应新立卡 `DEP1`）。
- **状态**：读代码与 dry-run 确认。
- **归属建议**：并入刚立的卡 `DEP1`（串口工具与接收端参考实现）。

---

## 3. 专题：新人从零做一遍会卡在哪

照 `README.md` 与 `CLAUDE.md` 给出的步骤，从拿到仓库到跑完一集闭环评测，逐环节只读核对文档说明与代码前置条件如下：

| 步骤 | 文档写的做法 | 实际会发生什么 / 漏交代的前置条件 | 卡点等级 |
| --- | --- | --- | --- |
| **1. 拿到仓库与准备环境** | `source env.sh && bash tools/setup_env.sh`（`README.md:14-15`、`HANDOFF.md:64`） | ① `HANDOFF.md:64` 写「`setup_env.sh` 只装 numpy/Pillow，`check.sh` 还要手装 `opencv-python-headless`」，实际 `tools/requirements.txt` 已含 `opencv-python-headless` 与 `imageio-ffmpeg`（两处说法不一）；② 干净克隆环境若无外网 PyPI，`setup_env.sh` 调 `pip install` 没有离线镜像说明；③ 未交代必须先设 `export TMPDIR=$DATA_ROOT/uav_rgb/tmp PYTHONDONTWRITEBYTECODE=1`（根分区仅剩 2.8 GB）。 | 低（文案过期，有网能过） |
| **2. 跑通 `tools/check.sh --python-only`** | `bash tools/check.sh --python-only`（`README.md:32`） | `check.sh` 在 `f2c1107`（E-13）已自动设 `PYTHONDONTWRITEBYTECODE=1` 与数据盘 `TMPDIR`，`.venv-tools` 就绪时可 `rc=0` 跑通。**但有隐性漏检**：若 `$DATA_ROOT/uav_rgb/3dgs-venv/bin/python` 未单独按 `HANDOFF_3DGS_20261105.md:50-56` 手工搭建，3DGS 的全部测试文件会被 `SKIP`（退出码仍为 0，仅末行提示 `PASS（有 SKIP）`）。`setup_env.sh` 不建 `3dgs-venv`。 | 低（能 rc=0，但缺 `3dgs-venv` 时跳过 3DGS 段） |
| **3. 编译 C++/CUDA 与 SITL 二进制** | `bash tools/check.sh` 或 `cmake -S src -B build/check ...`（`README.md:19-22`、`CLAUDE.md:64-66`） | **三处卡点**：① 不设 `UAV_RGB_BUILD_DIR=$DATA_ROOT/uav_rgb/tmp/build-check` 时，默认在仓库根 `build/check` 编译，直接占用剩 2.8 GB 的根分区；② 固件源码 `CF-Drone` 不在本仓库内，`check.sh` 默认找 `../CF-Drone`（从 `ALVLN` 克隆的新人需手工解压 `../ALVLN/references/snapshots/CF-Drone.zip` 并设 `CF_DRONE_DIR`）；③ **最隐蔽**：即使设了 `CF_DRONE_DIR` 跑通全量 `check.sh`，`sim/cf_sitl/CMakeLists.txt:22` 默认 `CF_SITL_BUILD_GZ=OFF`，**不会编出 Gazebo 插件 `libCfSitlSystem.so`**，后面跑 E1 闭环会直接报错（见 O-5）。 | **高（必卡在缺 `libCfSitlSystem.so` 或写根盘）** |
| **4. 采/打一个小数据包** | Gazebo 短采（`README.md:60-64`、`CLAUDE.md:75-80`）或合成夹具（`README.md:77-78`） | ① **显示服务前置条件未写清**：软件渲染要求 `:20` 上已有运行中的 Xorg/Xvfb（llvmpipe），文档只写 `DISPLAY=:20`，没写新机器/重启后 `:20` 不在时怎么起显示服务；② `CLAUDE.md:75,80` 用裸 `python3`（系统 Python 缺 `PIL` 即崩，见 O-3）且写 `/tmp`；③ `README.md:60` 与 `CLAUDE.md:75` 都没加 `--dynamics cf_sitl --vehicle e1_mini`，采出的是旧 `x3` 数据，后续与 E1 评测不兼容；④ 单集 `seed 42`（或 `seed 0..3` 的 `train/*/`）打包后 `val=0`。 | **高（裸 `python3` 缺 `PIL`、缺显示服务说明、采错载具）** |
| **5. 训一轮并拿到可评测权重** | `./build/check/uavrgb_train ...`（`README.md:65-68,79-81`、`CLAUDE.md:84`） | **三处卡点**：① 若照 `CLAUDE.md:17,22` 钉旧 UUID `GPU-dbc6435f-…`，CUDA 找不到设备直接失败（见 O-2）；② 用上一步 `seed 42`（`val=0`）训完只生成 `ckpt.bin`，**不生成 `ckpt_best.bin`**，而 `README.md:81` 紧接着敲 `uavrgb_infer --ckpt .../ckpt_best.bin`（若用于单集包会找不到文件；合成夹具 `ep1..3` 恰好含 `ep3` 为 `val` 才能过）；③ `README.md` 与 `CLAUDE.md` 完全没提现行交付必须加 `--save-every-epoch` 再跑 `tools/average_ckpt.py` 做末 K 轮平均加校准。 | **高（钉错 GPU UUID、单集包无 `ckpt_best.bin`、漏校准步骤）** |
| **6. 跑一集闭环评测** | `README.md` 无命令；`CLAUDE.md` 无命令；跳去 `HANDOFF.md:944-947` 或 `HANDOFF_3DGS_20261105.md:133-138` | ① `README.md` 与 `CLAUDE.md` 两份顶层入口居然**都没有给出哪怕一条 `eval_closed_loop.py` 命令**；② 读者跳到 `HANDOFF.md:944` 拿到的是裸 `python3`（缺 `PIL` 报错）、写 `/tmp/eval1`、跑旧 `x3`、缺软件渲染三件套（触发 Ogre2 EGL 段错误 `rc=-11`）、缺 `--corridor-half-width 3.0` 与 `--action-latency 0.20`；③ 若走 3DGS 线照 `HANDOFF_3DGS_20261105.md:133` 跑，因缺 `--hold`/`--data`、指错 `src/build/uavrgb_infer`、缺 `--allow-train-scene` 直接 `rc=2`（见 O-1）。 | **阻塞（照任何一份入口文档敲闭环评测均无法一次跑通）** |

---

## 4. 建议删除或标成历史的文档（供与 `p12-cleanup` 对账）

清理分支 `p12-cleanup`（截至 `5f157a6`）按「从根集合出发传递 1 层有引用即保留」的规则删掉了无引用文档。以下文档**正因为被入口文档引用而逃过了自动清理**，但内容已整体过时，继续挂在入口处会误导读者，建议主会话在合并 `p12-cleanup` 后做如下处理：

| 文档路径 | 现状与过时理由 | 当前谁在引用它（根集合内） | 建议处理方式 |
| --- | --- | --- | --- |
| `docs/TASKS.md` | 宣称是「唯一任务来源」，但全表停在 10-31 至 11-04（T1–T7、L1 v4、SIM-E1 doing），与第十二阶段唯一任务源 `docs/briefs/2026-11-10-phase12-task-cards.md` 直接冲突。 | `README.md:37`、`CLAUDE.md:7`、`docs/HANDOFF.md:80,545`、`docs/ENGINEERING_RULES.md:80`、`docs/ROADMAP.md:4` | **移入 `docs/archive/` 或在顶部加粗标为历史归档**，将上述 5 处入口引用全部改指 `docs/briefs/2026-11-10-phase12-task-cards.md`。 |
| `docs/realism_rl_v1/README.md` | 11-04 的写实 Gazebo + PPO Teacher 支线说明，未训练权重、未跑闭环，已不在端到端路线内（同目录 `VALIDATION.md` 已被 `p12-cleanup` 的 `e02cd21` 删除）。 | 仅 `README.md:130` 一处引用 | **删除或移入 `docs/archive/`**，同步删掉 `README.md:128-135` 末节。 |
| `docs/archive/PARALLEL_TRAINING_PLAN.md` 及 `docs/worklog/2026-10-29-*.md` 四份首日审查记录 | 10-29 立项首日的审查与未实施方案（`DeviceTrainState` 未接入主路径），无现行操作价值，仅因 `README.md` 底部硬编码链接而保留。 | `README.md:118-122`（「交接与证据」节） | **从 `README.md:116-126` 删掉这 5 条链接**；4 份 10-29 worklog 即可按清理规则归入删除类。 |
| `docs/CAMPUS_BASELINE.md`（及 `docs/experiments/2026-11-03-campus-baseline.md`） | 11-03 旧载具低模校园 pilot（闭环 0/80），现行真实场景线已迁至 3DGS（InteriorGS / Callinstrasse）。因代码仍有 `--task campus` 分支，规范仍有溯源价值，但不宜放在 `README.md` 首屏。 | `README.md:8`、`docs/HANDOFF.md:19`、`docs/EVAL_STANDARD.md:11`、`docs/TASKS.md:3` | **保留文件但顶部标「历史低模试点」**，从 `README.md:7-8` 首屏撤下，改放历史说明节。 |
| `docs/TEACHER_LINE_PLAN.md` 与 `docs/PLAN_2026-11-10_phase11.md` | 11-06 教师线规划（其中软标签蒸馏已证伪）与 11-10 第十一阶段计划（模块化子目标执行器方案已被第十二阶段 §10 端到端路线取代）。 | `ROADMAP.md:61` 引前者；`PLAN_2026-11-10_phase12.md:9` 引后者 | **保留文件供溯源，各自顶部加一行「方案已由 `PLAN_2026-11-10_phase12.md` §10 更新」**。 |
| `docs/HANDOFF.md` 与 `docs/ROADMAP.md` | 根集合核心保留项，但 `HANDOFF.md` 95% 内容是 11-04 前的 X3 / L1 v4 记录，`ROADMAP.md` 是 11-06 阶段 0 立项前的半年设想（表里写 E1 数据为零、CTest 25 项）。 | `README.md`、`CLAUDE.md`、`ENGINEERING_RULES.md` 等多处 | **保留文件，在两文顶部置顶指向 `docs/briefs/2026-11-10-phase12-common.md` 与 `PLAN_2026-11-10_phase12.md`**，将 `HANDOFF.md` 旧章节明确标为历史。 |

---

## 5. 第一轮清单（`FIXLIST.md`）中与本角度相关条目的复核

本角度上一轮无独立报告，以下核对第一轮清单中涉及入口文档、评测口径说明与操作食谱的条目：

- **修对了的**：
  - **D2、D4、D5、D7、A4（卡 W1-14，`0d7f3b5`）**：`docs/EVAL_STANDARD.md` §5.3、§5.6 的统计文字与 `tools/stats_report.py`、`tools/paired_stats.py` 对齐（请求集分母、失败集记 0、按 `geometry_gid` 整簇重采样区间、`EXACT_MAX_CLUSTERS=20` 精确符号翻转与 `100000` 次蒙特卡洛回退、`--allow-mixed-protocol` 五个键的表述一致）。
  - **E-13（卡 W1-15a，`4b43a9e`）**：`tools/check.sh:6-8` 已补 `PYTHONDONTWRITEBYTECODE=1` 与数据盘 `TMPDIR` 自动回退，与 `README.md` 的无参数调用方式一致。
  - **F7（卡 T1，`54ae6bb`）**：`tools/check.sh:110-146` 的几何 split 两个负例报错断言与 `tools/preprocess.py` 的两道门文案严格对应。
- **修了一半的**：
  - **D1（卡 S1a `5387747`、W1-15b `8399e32`、`e32714b`）**：`eval_3dgs.py` 代码与文档头部 docstring 已同步，`e32714b` 给 `HANDOFF_3DGS_20261105.md:132` 加了注释提示 `--allow-train-scene`，但代码块内的示例命令仍缺 `--allow-train-scene` 且缺 `--hold`/`--data`；`8399e32` 给 `train_student_with_video.py` 加了 `--allow-train-scene`，却遗漏了原本就缺 `--hold`/`--data` 导致真跑 `rc=2` 的问题（见 O-1）。
  - **E-3、E-4（卡 W1-15a/b、`e32714b`）**：`docs/HANDOFF.md:784-787` 的流水线起跑与停进程食谱已改为 `>>` 与 `tools/stop_daemon.sh`，但 `proto/README.md:104` 与 `CLAUDE.md:92` 仍是旧的 `kill -- -PGID` 写法（见 O-4、O-9）。
- **尚未修的（按排期等待）**：
  - **G5（步时序写进 `EVAL_STANDARD.md`）**：`docs/briefs/2026-11-10-phase12-common.md:13` 已更正，但 `docs/EVAL_STANDARD.md` 仍未写入步间刹停、停顿集失败与动作有效时长（排在卡 `W1-13` 第三轮，见 O-6）。

---

## 6. 查过但没有发现问题的地方

1. **`UAVR` v1 与 `UAVRAUX1` 二进制字节布局完全一致**：逐字段核对了 `tools/DATASET_FORMAT.md` §1.1（48 B 头：`<4sHHIHHBBBBQQQI`）与 §6.1（128 B 侧车头：`<8sQIHHBBH` + 两个 32 B SHA256 + `<7f` 分带上界 `0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 4.0` + `<fI` 包络半径 `0.102`），与 `tools/preprocess.py:79-86,108,1168-1174`、`src/dataset.cpp:473-501`、`src/aux_sidecar.cpp:45-98`、`tools/read_dataset.cpp:254-290` 完全吻合，无任何偏移或端序出入。
2. **`DATASET_FORMAT.md` §6.2 的新增条件码与解析深度描述准确**：`--cond heading-code`（`cond="heading_code.v1"`，累计已执行 `YAW_LEFT - YAW_RIGHT`，`|n|<=1 -> 0, n>=2 -> 2, n<=-2 -> 3`）及 `--depth-source analytic`（`e1_mini` 相机安装前 `0.045 m`、上 `0.015 m`，3×3 子射线，无 255 未知值，`depth_format` 区分 `png16_mm` 的 0→第 8 带与 `npy_float32_m` 的 0→255）与 `preprocess.py`、`sim/tools/nav_cond.py` 逐项一致。
3. **四动作编号、输入分辨率与文本词表常量全仓对齐**：`0=SAFE_NEUTRAL, 1=FORWARD_SLOW, 2=YAW_LEFT, 3=YAW_RIGHT`、`160×120`（4:3，PIL bilinear）、`kTextLen=9`、`kTextVocab=22` 在 `src/include/uavrgb/{action,observation,text}.hpp`、`tools/preprocess.py`、`sim/tools/{student,instruction}.py`、`proto/ckpt_io.py` 及各文档间完全一致（且有 `check.sh:15-38` 守护）。
4. **`UAVRCK01` v6 checkpoint 布局与行协议对齐**：`proto/README.md`、`proto/ckpt_io.py`、`tools/average_ckpt.py`、`src/include/uavrgb/checkpoint.hpp` 对 44 B 头、类别权重、`best_macro_f1`、`text_dim`、113 B 训练契约、`aux_classes`、`prev_dim` 的字节定义一致；`uavrgb_infer` 与 `proto/infer_wrap.py` 的启动握手 JSON 行及每步二进制输入（HWC uint8 + 可选上一帧 + 可选 1 字节条件码）一致。
5. **`tools/uart_action.py` 与 `docs/uart_action.md` 帧编码逐字节吻合**：实跑 `--dry-run --action 1` 输出 `aa 55 01 00 01 00 2c 01 39`，与文档 §3、§4 完全一致；`config/uart_protocol.example.json` 全部字段与文档表格一致。
6. **18 个 CLI 入口 `--help` 核对全部通过**：逐一实跑了 `tools/{doctor,preprocess,splat_to_ply,uart_action,train_student_with_video,stats_report,average_ckpt}.py`、`sim/tools/{collect,collect_parallel,eval_closed_loop,scene_gen,audit_dataset}.py`、`ui/monitor.py`、`sim.3dgs.{collect_3dgs,eval_3dgs,renderer,scene}`、`proto/{train,export}.py` 的 `--help`，退出码全为 0。
7. **入口文档本地 Markdown 链接无死链**：用脚本扫描了本角度全部 10 份入口文档的所有 Markdown 链接与相对路径（含 `../ALVLN/references/snapshots/CF-Drone.zip`、`../ALVLN/docs/08_CF_DRONE_SAFETY.md`、`../CF-Drone`），在当前 `main` 上目标文件全部存在（注：若后续合并 `p12-cleanup` 的 `5f157a6` 删掉 `data/ models/ logs/ third_party/`，需同步更新 `CLAUDE.md:121-123`，已列在 O-8）。

---

## 7. 没有查的地方与原因

1. **没有真跑 Gazebo 采集/闭环、3DGS 批量渲染、模型训练与全量 `tools/check.sh`**：遵守任务书第 3 节只读与不抢占 12 核配额 / GPU / 池锁的硬性规定；涉及运行行为的结论均通过读代码、单步参数解析或百毫秒级纯 Python 计算得出。
2. **没有逐篇核对 `docs/worklog/`（189 份）、`docs/experiments/`（55 份）与 `experiments/*/` 内部 README 的历史实验数字**：不在本角度规定的 10 份入口文档范围内，且清理 agent 正在分支 `p12-cleanup` 上处理无引用旧文档。
3. **没有审查 `ui/studio.py` 与 `ui/README_STUDIO.md` 的 GUI 交互细节**：仅核对了 `README.md`、`CLAUDE.md` 提及的启动脚本 `ui/launch_studio.sh`、`ui/monitor.py --help` 及与 `train_student_with_video.py` 共用的 `build_eval_command` 接口。
4. **没有核对硬件实机串口与板端 SG2002 工具链**：`$D/real-drone` 下尚无真实协议文档 `uart_protocol.md`，板端导出代码尚未建立（属角度 N 与卡 `DEP1`/`DEP2` 范围）。

---

## 附：交付自检

### 做了什么

- 按任务书第 1–4 节读完了全部前置材料与本角度范围内的 10 份入口文档（外加直接关联的 `sim/README.md`、`docs/TASKS.md`），逐条比对了 C++/CUDA 与 Python 生产代码。
- 实跑核对了 18 个 CLI 的 `--help`、`uart_action.py --dry-run`、`splat_to_ply.py` 已删参数报错、Gazebo 与 3DGS 示例场景的几何组 `split` 归属、系统/conda/worktree 解释器可用性、SITL 插件目录与版本、入口文档全部链接存在性，中间记录落盘于 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-o/notes.md`。
- 在落盘前对 `ada0ca5..2798e4a` 期间新合入的提交（含 `e32714b`、`8399e32`）及 `p12-cleanup` 分支（`5f157a6`）做了二次复核，产出了 10 条分级发现、「新人从零卡点」专题与「建议删除或标成历史的文档」对账表，写入 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/review2-o.md`。

### 没做什么

- 未修改主仓库或任何 worktree 中受 git 跟踪的文件，未写 `docs/worklog/` 与 `progress.md`（原因：任务书第 3 节硬性要求只读审核，不改仓库、不提交）。
- 未实际启动 `gz sim`、`uavrgb_train`、`eval_closed_loop.py`、`eval_3dgs.py` 或 `tools/check.sh`（原因：任务书第 3 节禁止启动仿真、训练与门禁脚本，避免干扰正在运行的实验池）。

### 验证到哪一步

| 检查项 | 结果 |
| --- | --- |
| 18 个 Python / 模块入口 `--help` | 18/18 通过（`rc=0`） |
| `tools/uart_action.py --dry-run --action 1` | 输出 `aa 55 01 00 01 00 2c 01 39`，与 `docs/uart_action.md` 逐字节一致 |
| `tools/splat_to_ply.py in.splat out.ply --flip-height` | 已复现报错 `rc=2`（参数已删除） |
| 3DGS 示例场景（`callinstrasse`、合成走廊 `seed=0`）split 复算 | 两者均属 `train` 分区，不带 `--allow-train-scene` 调 `eval_3dgs` 必 `rc=2` |
| Gazebo 示例 `seed 42` 与 `seed 0..3` split 复算 | 无一落在 `val`，训练不产 `ckpt_best.bin` |
| `/usr/bin/python3` 导 `PIL`、`/opt/conda/bin/python` 导 `torch` | 均复现 `ModuleNotFoundError` / `ImportError: libcudnn.so.9` |
| 10 份入口文档 Markdown 链接与文件存在性扫描 | 当前 `main` 上全部存在（0 死链） |
| 报告禁用词与仓库 `git status --short` 只读检查 | 见落盘后终端校验（干净） |
