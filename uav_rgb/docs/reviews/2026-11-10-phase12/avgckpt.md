# avgckpt 报告（B4：tools/average_ckpt.py）

状态：完成（2026-11-10 20:20）。分支 `p12-avgckpt`，提交 `55cba8d`，工作树 `/root/claude-code-backup-20261028/code/wt/p12-avgckpt`。

## 结论

- `tools/average_ckpt.py`（276 行）与 `tools/test_average_ckpt.py`（17 项，16 通过、1 跳过；跳过的是需要 GPU 的复现性验收）入库。
- 用工具重新生成 11-10 闭环实验的 6 个权重，sha256 与 `exp-swa-calib-20261110/ckpt/` 的参考文件逐位相同。
  草稿推理二进制与闭环用的 `main-97ae04b` 快照二进制各跑两遍以上，每次 6/6 相同。
- `check.sh --python-only` rc=0。
- 流水线没接，接入点与待补测试见 §4。
- 本分支 4 个文件（`tools/average_ckpt.py`、`tools/test_average_ckpt.py`、实验 README 一行、worklog）与主仓库 155e377..ef68141 的改动无文件级交集。

## 0. 先核实的事实（读草稿与 src 的结果）

- 工作树 `/root/claude-code-backup-20261028/code/wt/p12-avgckpt`，分支 `p12-avgckpt`，基于 main 155e377，开工时工作区干净。
- ckpt 格式：`src/include/uavrgb/checkpoint.hpp` 文件头注释与 `src/checkpoint.cpp` 的 `save_checkpoint` 一致。写端版本号 `kVersion = 6`，读端接受 v2..v6；**文件没有校验和**（注释写明「无校验和」，v5 契约里的三个 SHA256 是数据集/软标签/split 方案的指纹，不是文件校验）。
  `ck.py` 的 `load` 无条件按 v6 布局解析（读 text_dim、113 B 契约、aux/prev），对 v2..v5 会错位且不报错。新工具只认 v6，其它版本直接报错。
- `ck.py` 对 trainer 写出的真实文件成立：`ckpt_ep21.bin`、`ckpt_ep30.bin` 的 version=6、param_count=8、头里的 epoch 等于文件名里的轮次、text_dim/aux/prev 全为 0、in_c=3。
- `src/dataset.cpp` 的 split 与碰撞口径：`episodes[]` 的 `sample_start/sample_count/split` 选区间，`drop_collision` 时按 `collision_flags` 的 `'1'` 剔除；字段缺失或长度不等于 N 或含 0/1 以外字符时报错，不当成无碰撞。`run_val.py` 在字段缺失时当成无碰撞，新工具按 C++ 的口径报错。
- 行协议（`src/infer_main.cpp`、`src/infer.cpp`、`sim/tools/student.py`）：启动先出一行 hello JSON（in_h、in_w、in_c、num_classes、text_dim、prev_action_dim、aux_classes），之后每帧写 `in_c/3` 张 HWC RGB（再加 prev-action 1 字节），回一行 `{"action","probs","infer_ms"}`。`probs` 用 `std::ostream` 默认精度输出，只有 6 位有效数字。
- 草稿的校准（`analyze2.calibrate`）：对 val 帧的 `log(max(p,1e-300))`，在 FORWARD 列加 δ，二分 60 次，区间 [-30, 30]，`mean(argmax==1) < target` 时抬下界；返回 `0.5*(lo+hi)`；`target` = val 标签里类 1 的占比（剔除碰撞步）。写 ckpt 时 `params[7][1] += np.float32(δ)`，以平均后的 ckpt 为模板。
- 草稿的平均（`swa.py`）：按轮序升序，float64 累加，除以 k，转 float32；模板是 `ckpt_ep30.bin`。
- 参考 sha256（`exp-swa-calib-20261110/inputs.txt`）与盘上文件 `sha256sum` 一致（9 个文件逐个核对过）；草稿目录 `tmp/cursor-1110/agents/src/ckpt/` 下的同名文件 sha256 也一致。
- 数据集 `tmp/agent6-distill/trunc/ds/dataset.uavr`：N=9663，val 574 帧（18 集）、test 884、train 8205；`collision_flags` 全为 `'0'`（打包时已丢 182 个碰撞步）。val 标签 FORWARD 501 / L 40 / R 33 / SAFE 0。
- 两个推理二进制 sha256 不同：草稿用的 `tmp/cursor-1110/build-check/uavrgb_infer` 是 667407c5…，闭环用的 `bin-snapshots/main-97ae04b/uavrgb_infer` 是 b5458338…（大小同为 1493736 B）。
- `RESULT.md` 的闭环数字（走廊内成功，60 个配对集）：现行 ckpt_best 27，末 10 轮平均加校准 39，ckpt_best 加校准 36，末 10 轮平均不校准 20。文件头里「只平均比 ckpt_best 还差」引的是这一行。

## 1. 设计决定

1. **只认 v6。** 读端支持 v2..v6，草稿对 v2..v5 会错位不报错；新工具遇到其他版本直接报错，`read_ckpt` 的 `shortcut:` 注释写了升级路径（照 `checkpoint.cpp` 逐版本补）。
2. **窗口取法。** run 里编号最大的 `ckpt_epNN.bin` 为末轮，取 ep(末−k+1)..ep(末)。缺任何一个（含 k 大于轮数）、文件头 epoch 与文件名不符、结构字段或形状不一致，都报错。
   没有与 `config.json` 的 epochs 对账（续训时 epochs 的含义不同），`shortcut:` 已记。
3. **平均。** 与草稿 `swa.py` 同：升序、float64 累加、除 k、转 float32；头、契约、Adam 取末轮。k 个相同文件逐位不变（k·a 在 float64 里精确），测试锁住。
4. **校准二分。** 与草稿 `analyze2.calibrate` 同：区间 [-30,30]、60 步、`frac < target` 抬下界、返回 `0.5*(lo+hi)`。同一份概率下 δ 逐位相同（§3）。改这些常量就复现不了 11-10 的权重，配置区注释写明。
5. **容差 `VERIFY_FRAMES = 3`。** 二分结束时 lo、hi 是相邻 float64，返回值可能落在阈值任一侧（差 1 帧）；真实推理与离线平移再差不超过 1 帧。第一版用绝对容差 0.01，在 76 帧的合成数据上差 1 帧（1/76 = 0.0132）就超限而误报，改成按帧数。seed2 实测比目标少 1 帧（0.8711 对 0.8728）。
6. **成品先真实推理核对再落盘。** 写出的 cal ckpt 再跑一遍 val，占比偏离目标超过 3 帧则不落盘。成品在 `<out 目录>/.average_ckpt_XXXX/` 里生成，通过后 `os.replace` 到 `<out>`；失败时临时目录自动清掉（测试检查了无残留）。
7. **`--data` 必须是训练用的那份。** ckpt 的训练契约（v5 起）前 32 B 是训练数据集 sha256，三个种子实测都等于 `sha256(dataset.uavr)`（改检查之前的 `<out>.json` 同时记了两个值，相同）。不同则报错；通过时 `<out>.json` 的 `data_sha256` 就是契约里的值。
   这是任务书没列的检查：dagger 流水线每轮有自己的 `ds`，传错的可能性真实存在，传错会让目标占比不是训练分布的，δ 静默偏掉。代价是不能拿别的数据集的 val 校准；现有用法里没有这种需求。要放开可以改成警告，见「待主会话拍板」。
8. **`collision_flags` 口径同 `src/dataset.cpp`。** 缺失、长度不等于 N、含 0/1 以外字符都报错。
9. **`<out>.json` 最后写；`<out>` 或 `<out>.json` 已存在时拒绝覆盖。** 流水线判完成用 `.json`（`<out>` 单独存在是半成品）。
10. **不钉 GPU。** 调用方用 `CUDA_VISIBLE_DEVICES` 钉；本次验收钉在 `GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70`，没碰 GPU 0。
11. **不 import `select_ckpt`。** 它经 `student` 拉进 PIL 与 teacher；本工具只依赖 numpy 与标准库，数据集按 `tools/DATASET_FORMAT.md` 自己读（头、偏移与文件长度自洽检查，不自洽报错）。
12. **不支持的输入报错，不猜。** ckpt 的 `in_c != 3`、`text_dim > 0`、`prev_dim > 0` 时，校准报「暂不支持」并提示 `--no-calibrate`；只平均不受影响。`calibrate_and_write` 的 `shortcut:` 注释写了升级条件（按 `student.py` / `select_ckpt.predict_indices` 补文本与 prev-action）。

## 2. 先红后绿

首次运行（实现前），`avgckpt/red.txt`：

```
Traceback (most recent call last):
  File ".../tools/test_average_ckpt.py", line 25, in <module>
    import average_ckpt as ac  # noqa: E402
ModuleNotFoundError: No module named 'average_ckpt'
```

后加的「数据集 sha256 不符」一项，在去掉该检查的副本上跑，`avgckpt/red_guard.txt`：

```
FAIL: test_data_must_be_the_one_the_ckpt_was_trained_on
AssertionError: SystemExit not raised
Ran 1 test in 0.961s
FAILED (failures=1)
```

最终（`python -W error::ResourceWarning tools/test_average_ckpt.py`）：

```
Ran 17 tests in 0.892s
OK (skipped=1)
```

开发中撞到并修掉的错误：

- `np.fromfile(path, np.uint8, n, lo)` 的第 4 个位置参数是 sep：`TypeError: fromfile() argument 4 must be str, not int`，改成关键字 `offset=`。
- 合成数据的 e2e 里，绝对容差 0.01 误报（真实推理 FORWARD 占比 0.6974 对目标 0.7105，差 1 帧，即 0.0131），改成 `VERIFY_FRAMES`。
- 测试里 `open(...).read()` 触发 ResourceWarning，改成 `slurp/spit/load_json` 辅助函数，`-W error::ResourceWarning` 下通过。

## 3. 复现性验收

命令（仓库根；`UAVRGB_REPRO_OUT` 是父目录，测试在里面新建目录）：

```
UAVRGB_REPRO_AVGCKPT=1 UAVRGB_REPRO_OUT=<父目录> CUDA_VISIBLE_DEVICES=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70 \
  $R/.venv-tools/bin/python tools/test_average_ckpt.py TestReproduction
# 换推理二进制：加 UAVRGB_REPRO_INFER=<路径>（默认是草稿 tmp/cursor-1110/build-check/uavrgb_infer）
```

输入：`exp-student-select-20261109/e1-noise/seed{0,1,2}/run` 的末 10 轮（ep21..ep30，`last_epoch=30`）；数据集 `tmp/agent6-distill/trunc/ds/dataset.uavr`（sha256 534ca5904ea0…，等于三个 ckpt 训练契约里的数据集 sha256）。
共跑 5 次：草稿二进制 3 次（`repro_run1.txt`、`repro_run2_draftbin.txt`、`repro_final_draftbin.txt`），快照二进制 2 次（`repro_run3_snapbin.txt`、`repro_final_snapbin.txt`），文件都在 `avgckpt/`。每次 6/6 sha256 与参考相同；最终代码的两次各约 25 s。

### 3.1 sha256 对照

两个推理二进制下的结果相同，也等于 `exp-swa-calib-20261110/inputs.txt` 的参考值：

| 文件 | 生成方式 | sha256（本工具 = 参考） |
|---|---|---|
| swa_s0_k10.bin | `--no-calibrate` | 15570258caf075802f1054482ad359c3fcfefff000db5407765d7f6b4eba6534 |
| swa_s1_k10.bin | `--no-calibrate` | b00ddf1756a514229f4739a09f6598739d2ee0b054d05e0aca70b15c68080e2c |
| swa_s2_k10.bin | `--no-calibrate` | 5ac4aa9aead3f16c609cb1e3bcd275032c5a9cbaf46de84a6f2631a32d6fe436 |
| cal_swa_s0_k10.bin | 平均 + 校准 | 2fb25b30b337589999ad2385e2372bb38ae1ca6e9b0b28d6889d25bf4a327928 |
| cal_swa_s1_k10.bin | 平均 + 校准 | 5cfe51f58f4f4a1d1afb4f79db0f11e777fd67271b9605f722b6bdf8372a6900 |
| cal_swa_s2_k10.bin | 平均 + 校准 | abcae7cf0d09a475f68356535e14c0afeaf201aefa0fd5f0375ceff7d4b17734 |

### 3.2 δ 与 FORWARD 占比对照

「草稿 δ」是用草稿自己保存的逐帧概率（`tmp/cursor-1110/agents/src/probs/swa_s{s}_k10.npy` 的前 574 行）和草稿 `analyze2.calibrate` 的二分重算的；脚本 `avgckpt/delta_compare.py`，输出 `avgckpt/delta_compare.txt`。

| 种子 | δ（本工具，float64） | 草稿 δ | 参考两个 ckpt 的 `fc.b[1]` 之差（float32） | FORWARD 占比 校准前 → 后（真实推理） | 目标 |
|---|---|---|---|---|---|
| 0 | 1.6995256849869436 | 逐位相同 | 1.6995257139205933 | 0.7387 → 0.8728 | 0.8728（501/574） |
| 1 | 1.5575871746157457 | 逐位相同 | 1.5575871467590332 | 0.7526 → 0.8728 | 0.8728 |
| 2 | 1.6185366028066048 | 逐位相同 | 1.6185365915298462 | 0.7648 → 0.8711 | 0.8728 |

- δ 与 float32 之差在 3e-8 以内，来自 `fc.b[1]` 加 δ 时的 float32 舍入（输出文件逐位相同，所以舍入路径与草稿一致）。
- 本工具这次推理的 574×4 个概率与草稿保存的概率逐位相同（最大绝对差 0），两个二进制都是。说明这两个二进制对这三个 ckpt 的推理输出相同；对其他 ckpt 是否也相同，未核实。
- 两个二进制 sha256 不同却输出相同的原因，未核实（没有比对二进制或构建输入）。
- 同一个二进制重复跑，输出逐位相同，GPU 推理在这个口径下可复现。

## 4. 流水线接入（本轮没接，给后续任务卡）

### 4.1 `tools/e1_pipeline.py`

- `train(out)`（222–241 行）：`uavrgb_train` 命令（231–233 行）加 `--save-every-epoch`（`src/train_main.cpp` 用法行已有；每轮另存 `<run_dir>/ckpt_epNN.bin`，NN 与 `src/trainer.cpp:647` 的 `%02d` 一致）。计划 B4 里同行的 `--augment photo`、类权重属于另一张卡。
- 训练步之后新增一步（`STEPS`，121 行，`train` 与 `eval` 之间加 `average`；或并进 `train()` 末尾），命令行：

  ```
  [PY, f"{REPO}/tools/average_ckpt.py", "--run", run_dir, "--data", f"{ds}/dataset.uavr",
   "--out", f"{run_dir}/ckpt_swa_cal.bin", "--last-k", "10", "--infer-bin", f"{BUILD}/uavrgb_infer"]
  ```

  `CUDA_VISIBLE_DEVICES` 沿用流水线进程的环境（由启动者钉卡）；需要 GPU，每次 574 帧推理约 4 s、一次运行推理两遍，复现测试 3 个种子共约 25 s。
- `evaluate(out)`（246 行）：`new_ckpt` 由 `run/ckpt_best.bin` 改指向 `run/ckpt_swa_cal.bin`。
- 续跑：`<out>.json` 在则跳过；`<out>` 在而 `.json` 不在是半成品，要么删掉重跑，要么报错让人处理（工具自己拒绝覆盖）。
- 开关：CFG 里加一项（如 `average_last_k`，0 关闭，默认关闭），关闭时命令列表与现状逐位相同。

### 4.2 `tools/dagger_gz_pipeline.py`

- `train_into(ds, run_dir)`（462–479 行）：命令（468–470 行）加 `--save-every-epoch`；`need(ckpt_best)` 与 `train_done` 检查之后，新增 `average_into(ds, run_dir)`，同样的命令行，日志 `log_path=run_dir + ".average.log"`。
- `run_round`（479–494 行）里在 `round_{k}.train` 之后加 `step(out, f"round_{k}.average", average_into, ...)`；返回值（494 行）由 `ckpt_best.bin` 改成 `ckpt_swa_cal.bin`，它是下一轮学生自采的 `--student-ckpt` 与各臂评测的 ckpt。
- `run_more_demos`（497–519 行）同样处理，返回值在 519 行。
- `step()` 只在 `status.json` 记 running/ok/failed，续跑语义靠各函数自己判断，所以 `average_into` 要自己判断 `.json` 在则跳过。

### 4.3 要补的测试

- `tools/test_e1_pipeline.py`：
  - 默认关闭：`train` 的命令列表与现状逐位相同（锁住「新开关默认关闭时行为不变」）。
  - 开启：训练命令含 `--save-every-epoch`，其后紧跟上面的 `average_ckpt.py` 命令；`evaluate` 的 `new_ckpt` 指向 `ckpt_swa_cal.bin`。
  - 续跑：`<out>.json` 在则不再起 `average_ckpt.py`（仿 132–141 行 `test_train_resume_by_metrics_done` 的写法）；`<out>` 在而 `.json` 不在时的行为。
  - 失败传播：`average_ckpt.py` rc≠0 时流水线停下，不拿旧的 `ckpt_best.bin` 去评测。
- `tools/test_dagger_gz_pipeline.py`：同样四项；现有 297 行的断言（`--student-ckpt {o}/round_1/run/ckpt_best.bin`）和 302 行的训练命令断言锁的是默认口径，开关打开时要另写一份。
- 单独一张卡做真实验收：在一次真实流水线的 run 目录上跑 `average_ckpt.py`，对比 `<out>.json` 的 δ 与占比。

## 做了什么

- 新增 `tools/average_ckpt.py`：末 k 轮权重平均（float64 累加）加 FORWARD 先验校准（真实 `uavrgb_infer` 推理 val、二分求 δ、`fc.b[1] += δ`、成品再推理核对），旁文件 `<out>.json`，只认 v6，缺文件/形状不符/版本不符/数据集 sha256 不符/碰撞标志不合法/不支持的输入都报错。
- 新增 `tools/test_average_ckpt.py`：17 项，先红后绿；`check.sh --python-only` 会收到；复现性验收用环境变量开关。
- 在 `experiments/2026-11-10-swa-calib/README.md` 末尾加一行，指向正式工具。
- 新增 `docs/worklog/2026-11-10-2014-avgckpt.md`（过程文档，跟代码一起提交）。
- 复现性验收：6 个权重 sha256 逐位相同（两个二进制各跑过），δ 与草稿逐位相同。
- 提交一次：`55cba8d`（分支 `p12-avgckpt`）。

## 没做什么（每条带原因）

- 没接 `e1_pipeline.py`、`dagger_gz_pipeline.py`。原因：任务书写明本轮只写接入点（§4）。
- 没加 `--augment photo`、类权重这些 B4 的另一半。原因：不属于这张卡。
- 没在文档里找到可追加的「工具清单」。原因：grep `select_ckpt` 命中的是 `progress.md`、`docs/worklog/`、`docs/experiments/2026-11-09-student-select.md`、`docs/PLAN_2026-11-10_phase11.md`、`docs/reviews/2026-11-10/` 的快照，都不是现行的工具清单；`README.md` 没有工具清单一节。
- 没改 `progress.md`。原因：任务书没要求；多个分支并行，各改同一个文件会在合并时冲突（推断，未核实其他分支是否在改），进度以本报告和 worklog 为准，由主会话汇总。
- 没支持文本条件、prev-action、双帧 ckpt 的校准。原因：行协议要逐集 `--text`、读 `dataset.aux` 的 prev 索引，还原不出来就会算出错的 δ；本轮没有这类 ckpt 要用。遇到时报「暂不支持」，`shortcut:` 注释写了升级条件。
- 没支持 v2..v5 ckpt。原因：当前训练器只写 v6。
- 没给推理子进程加读超时。原因：接流水线时由外层守护脚本兜（`student.py` 有 30 s 超时，但它依赖 PIL）；`shortcut:` 已记。
- 没把末轮与 `config.json` 的 epochs 对账。原因：续训时 epochs 的含义不同；被中途杀掉的 run 会平均到不完整窗口，接流水线时靠训练的 done 标记保证；`shortcut:` 已记。
- 没把文件压回 220 行。原因：276 行里函数合计 209 行；压到 220 只能丢逐项校验（v6 解析、窗口一致性、val 读取自洽检查）。
- 没跑全量 `check.sh`、C++ 构建、CTest。原因：没改 C++；任务书规定只改 Python 时跑 `--python-only`。
- 没跑 Gazebo 与闭环。原因：任务书禁止；闭环数字沿用 `RESULT.md`，本轮证明的是权重逐位相同。
- 没用别的 k 或别的数据集验收。原因：参考权重只有 k=10 的三份，没有别的参考可对。

## 验证到哪一步

| 命令或检查项 | 结果 |
|---|---|
| `python tools/test_average_ckpt.py`（首次，无实现） | 红：`ModuleNotFoundError: No module named 'average_ckpt'`（`avgckpt/red.txt`） |
| 数据集 sha256 检查的测试，在去掉检查的副本上 | 红：`SystemExit not raised`（`avgckpt/red_guard.txt`） |
| `python -W error::ResourceWarning tools/test_average_ckpt.py`（最终代码） | 17 项，16 通过，1 跳过（复现性验收，默认跳过） |
| `UAVRGB_REPRO_AVGCKPT=1 … TestReproduction`，草稿二进制（667407c5…） | 通过：6/6 sha256 与参考相同（共 3 次） |
| 同上，`UAVRGB_REPRO_INFER=…/bin-snapshots/main-97ae04b/uavrgb_infer`（b5458338…） | 通过：6/6 sha256 与参考相同（共 2 次） |
| `avgckpt/delta_compare.py` | 概率与草稿保存的逐位相同（最大绝对差 0）；δ 三个种子逐位相同 |
| `cd <工作树> && UAV_RGB_PYTHON=$R/.venv-tools/bin/python bash tools/check.sh --python-only` | rc=0（2026-11-10 20:15:41 结束，输出「PASS（有 SKIP）」；未检查项：C++/CUDA 构建、CTest、GPU 测试、cf_sitl）；日志 `avgckpt/check.log` |
| 流水线接入、真实流水线 run 上的验收 | 没做（见 §4 与「没做什么」） |
| 闭环（Gazebo） | 没做（任务书禁止） |
| 残留进程 | 我起的都已结束：`check.sh` 用守护脚本跑完（`avgckpt/check.DONE` rc=0），其余命令在前台结束。`ps` 里能看到 `exp-long-course-*` 与 `main-97ae04b/uavrgb_infer --ckpt …exp-swa-calib…` 的进程（PGID 1372765、1396877、1397036、1397388），不是我起的，没动 |
| 合并冲突 | 主仓库此时已在 ef68141（p12-longcourse 已合入）；`git diff --name-only 155e377 ef68141` 的 25 个文件与我的 4 个文件无交集 |

## 待主会话拍板

- `--data` 与训练契约 sha256 不符时报错（设计决定 7）。要放开成警告还是保持报错。
- 文件 276 行对任务书的「约 220 行」：接受，还是要压。
- `VERIFY_FRAMES = 3` 的容差与「`<out>` 已存在则拒绝覆盖」的策略，接入流水线的任务卡按这两条写续跑逻辑。
