# REFACTOR-TOOLS：gsplat 缓存目录与 dagger_gz 重复 helper 收成一处

- 时间：2026-11-09 23:00
- 范围：`tools/pipeline_common.py`、`tools/dagger_pipeline.py`、`tools/demo_3dgs_routes.py`、`tools/scene_import_mipnerf.py`、
  `tools/dagger_gz_pipeline.py` 及对应 `tools/test_*.py`（phase9 BRIEF §3）
- 结论：完成（check.sh 结果见「验证到哪一步」）。报告：`$DATA_ROOT/uav_rgb/tmp/phase9-20261109/reports/REFACTOR_TOOLS_20261109.md`

## 做了什么

- `tools/pipeline_common.py`：加模块常量 `DATA_ROOT` 与 `torch_extensions_dir(capability=None, python=None)`：
  目录 `$DATA_ROOT/uav_rgb/torch-cache/extensions-sm<major><minor>`；capability 缺省用 python 问
  `torch.cuda.get_device_capability()`（120 s 超时）；问不到返回 None。函数体取自 dagger_pipeline 原版，逐行不变。
- `tools/dagger_pipeline.py`：`torch_extensions_dir` 转调 pc，默认 python 仍是 `PY3DGS`；问不到仍返回 None，main 不设 `TORCH_EXTENSIONS_DIR`。
- `tools/demo_3dgs_routes.py`：import pipeline_common；`torch_extensions_dir` 转调 pc（python 用 `PY`），pc 返回 None 时抛 `RuntimeError`。
- `tools/scene_import_mipnerf.py`：import pipeline_common；`torch_ext_dir` 先看环境变量 `TORCH_EXTENSIONS_DIR`，
  再转调 pc（用 `PY` 问 torch；原来问 nvidia-smi），问不到回退 `extensions-smunknown`。
- `tools/dagger_gz_pipeline.py`：`log` / `read_json` / `write_json` 改用 pipeline_common 的（三者逐字相同）；
  `run` / `sh` / `mark` / `need` 语义不同，保留；原 `shortcut:` 注释换成两行，写明为什么不能换。
- 测试（先红后绿）：`test_pipeline_common.test_torch_extensions_dir`（新）与 `test_shared_helpers`（加 dagger_gz 三个 helper
  是同一份）；`test_dagger_pipeline.test_torch_extensions_dir_uses_pipeline_common`（新）；
  `test_demo_3dgs_routes.test_torch_cache_dir_shared_rule_raises_without_gpu`（新）；`test_scene_import_mipnerf.TestTorchExtDir`（新）。

## 没做什么

- `tools/scene_import_wanderland.py:76` 的第四份 `torch_extensions_dir` 没改。
  - 原因：不在 BRIEF §3 文件域。它与 demo 原实现逐字相同，可照 demo 改成转调 pc、None 时报错。
- dagger_gz 的 `run` / `sh` / `mark` / `need` / `train_done` / `eval_done`、`evaluate` 里挪半成品目录与并行 `Popen` 没换成 pc 的。
  - 原因：语义不同（进程组与 SIGTERM 转发、status.json 合并、dry-run 提醒行、partial 判定、改名格式），换了外部行为会变；逐个原因见报告 §3。
- 没在 3DGS venv 里真问一次 torch 算力。
  - 原因：用户限定 3DGS venv 只用来确认 import；GPU 上有 DAgger 正式跑的进程。替代证据见「验证到哪一步」第 4 行。
- 没跑任何真实 3DGS / gz 流水线，没起渲染或 JIT。
  - 原因：本任务只动目录名解析与 helper 来源，不需要。
- 没测 pc 里的 120 s 超时分支。
  - 原因：要真等 120 s；这段代码逐行取自 dagger_pipeline 原版。
- 没保留 demo 原来的异常类型（`CalledProcessError` / `FileNotFoundError`）。
  - 原因：pc 问不到只返回 None（dagger_pipeline 需要这个语义）；demo 仍然报错、DONE 仍 rc=1，变的只是异常类型与文字。
- 没在 `test_dagger_gz_pipeline.py` 里加测试（仍 18 项）。
  - 原因：「helper 是 pc 那一份」的断言按现有写法放在 `test_pipeline_common.test_shared_helpers`（e1 / compare / dagger_pipeline 的同类断言都在那里）。
- 没 commit、没 push；没动主仓库工作树、`tmp/wt-eval4` 与别的 worktree；没动任何缓存目录。

## 待确认

- wanderland 的第四份是否在合并时一起收。
- `scene_import_mipnerf` 改用 torch 问算力（与另两处同源）是否接受；若要保留 nvidia-smi，就回到两种问法。
- dagger_gz `eval_done` 不看 `partial`：恰好在两集之间被中断时，partial 的 eval.json 续跑会被当成完成、该臂跳过重评
  （`tmp/wt-eval4` 正在跑的副本同样如此）。是否换成 `pc.eval_finished`（会改续跑判定）。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 红测：改前跑 5 条新测试 | 全红：`AttributeError: module 'pipeline_common' has no attribute 'torch_extensions_dir'`（pc 与 dagger_pipeline 两条）；`AssertionError: dagger_gz_pipeline.log 不是 pipeline_common 的那一份`；`AttributeError: module 'demo_3dgs_routes' has no attribute 'pc'`；`AttributeError: module 'scene_import_mipnerf' has no attribute 'pc'` |
| 改后 5 个测试文件（`.venv-tools`） | 全过：`test_pipeline_common: PASS`；`test_dagger_pipeline: PASS`；`test_dagger_gz_pipeline` `Ran 18 tests … OK`；`test_demo_3dgs_routes` `Ran 5 tests … OK`；`test_scene_import_mipnerf` `Ran 9 tests … OK` |
| `--dry-run` 改前改后 diff：dagger_gz 默认 / smoke / `--stop-after-preflight`，dagger_pipeline 默认 / `--collect-round0 --rounds 1` | 5 组 `diff <(cut -c21- before) <(cut -c21- after)` 全空（rc=0；行数 83/42/4/95/68 不变；每行都有 20 字符时间戳前缀，原样 diff 的差异行全是时间戳） |
| 三脚本缓存目录改前改后对拍（`tmp/agent11-refactor-tools/compare_ext_dir.py`，算力问假 python） | 问得到（打印 `7 5`）：三者改前改后都是 `/root/rivermind-data/uav_rgb/torch-cache/extensions-sm75`。问不到：dagger 改前改后 None；demo 改前抛 CalledProcessError / FileNotFoundError，改后抛 RuntimeError；mipnerf 改前走 nvidia-smi 不看 PY 得 sm75，改后回退 extensions-smunknown。环境变量优先：改前改后都返回 `/given/ext` |
| 本机算力两种问法 | `nvidia-smi --query-gpu=name,compute_cap` → `NVIDIA GeForce RTX 2080 Ti, 7.5`；torch 问法的旁证：`datasets/e1-3dgs-pilot-20261109`（run_pilots.sh 调 demo_3dgs_routes）20:42:49 开跑，DONE `rc=0 2026-11-09 21:24:21`，`torch-cache/extensions-sm75/gsplat_cuda` mtime 21:24:00；nvidia-smi 问法：mipnerf 19:41–20:01 四次运行日志都是 `TORCH_EXTENSIONS_DIR=/root/rivermind-data/uav_rgb/torch-cache/extensions-sm75` |
| 3DGS venv import | `3dgs-venv/bin/python` import pipeline_common / demo_3dgs_routes / scene_import_mipnerf 通过，两者的 `pc` 是同一个模块，torch 未导入 |
| `check.sh --python-only`（22:59 起，23:02:35 结束，rc=0） | 最后两行：`test_train_student_with_video: PASS` / `PASS（有 SKIP）：所选检查完成，但未检查：C++/CUDA 构建、CTest、GPU 测试、cf_sitl；模拟采集、真实训练和学生闭环须独立验收。` |

## 风险与遗留

- demo 问不到算力：DONE 仍 rc=1，run.log 的 `FAILED:` 行从 `CalledProcessError: …` 变成 `RuntimeError: 问不到 GPU 算力…`；原来没有超时，现在 120 s。
- mipnerf 算力来源换成 PY 里的 torch：单卡且 PY 正常时同值；PY 坏了、torch 看不到卡或问超时，改前 nvidia-smi 仍给 sm75，
  改后回退 extensions-smunknown（torch 看不到卡时 gsplat 本来跑不了；偶发超时后若 probe 步能用卡，会在新目录重编约 4 min）；
  多卡且 `CUDA_VISIBLE_DEVICES` 不是 0 号卡时，改后跟 torch 可见的第一张卡走；启动多一次 torch import 子进程（几秒）。
- dagger_gz `eval_done` 的 partial 问题照旧（见「待确认」）。
- pc 新增模块级 `DATA_ROOT`（import 时读环境变量），与各调用方原来的取值时机相同。
