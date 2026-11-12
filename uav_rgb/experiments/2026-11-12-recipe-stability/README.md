# 训练配方的稳定性筛选与工作点扫描（任务卡 E2E-R1）

任务卡 `docs/briefs/2026-11-10-phase12-task-cards.md` §4b「卡 E2E-R1」。起因：交叉审核 H（`docs/reviews/2026-11-12-cross-review/review-h.md`）用已有的逐轮预测复现了一件事：
训练种子之间、相邻轮之间的不稳定，大部分是「前进对转向」这一个工作点在漂。这里问两件事：训练器现成的开关能不能在配方层把漂移消掉（第一部分，离线）；
这个工作点在闭环上放在哪最好（第二部分，闭环）。结果看 `RESULT.md`；过程与「没做什么」看 `docs/worklog/` 里同名的过程文档。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run_offline.sh` | 第一部分入口（守护进程）：复训闸门（参照配方种子 0）→ R1、R2、R3（或替代臂 R2e、R3e）各 3 个训练种子 → 逐轮 val 预测 → 末 10 轮平均与平均加校准。预注册与「起跑前发现」写在文件头 |
| `gate_check.py` | 闸门对账：复训与参照 run 的 results.csv、metrics.jsonl、config.json、逐轮权重逐项比 |
| `avg_preds.py` | 末 10 轮平均与平均加校准的 ckpt（调 `tools/average_ckpt.py`），真实推理落盘 val 预测 |
| `stability.py` | 第一部分的指标：相邻轮翻转率（原始 / 每轮校准）、三种子全一致率、校准量 δ 的范围、val 无权 CE 曲线、平均权重的 macro-F1 与每类召回、过线判决；`repro-h1` 子命令复现审核 H 的数字 |
| `select_val_geoms.py` | 第二部分的 20 个 val 几何：按几何组 ID 排除后挑选，并用 `scene_gen` 真生成核对；`--frozen-check` 只核对冻结清单 |
| `make_shifted_ckpts.py` | 第二部分的权重：现成 `cal_swa_s{0,1,2}_k10.bin` 的 FORWARD 偏置平移 ±0.5 nat（调 `tools/average_ckpt.py` 的 `shift_forward_bias`），逐字节核对 |
| `run_closed_loop.sh` | 第二部分入口（守护进程）：排整机锁，2 条车道，三档 × 3 训练种子 × 20 几何（有过线的配方臂时再加一个臂），含迟到步的集重跑一次。预注册写在文件头 |
| `summarize_sweep.py` | 第二部分的汇总：三档对比、两两比较（几何聚类符号翻转精确检验 + bootstrap 区间，逐集 McNemar 并列）、预注册判定、逐几何表、迟到与重跑记录 |
| `stub_eval.py` | 演练用的评测桩（不起 Gazebo），只用来走通 `run_closed_loop.sh` 的流程，数字没有含义 |
| `test_stability.py`、`test_sweep.py`、`test_geoms_ckpts.py` | 测试；运行 `cd experiments/2026-11-12-recipe-stability && <解释器> -m unittest test_stability test_sweep test_geoms_ckpts` |
| `results/` | 入库的结果：`stability.json`、`stability.md`、`ce_curves.csv`、`curves.png`（第一部分），`val_geoms.json`（几何清单的出处），第二部分汇总 |
| `RESULT.md` | 结果与读法 |

## 运行（要点；完整的起跑条件、DONE 的 rc、停与续跑见两个入口脚本的文件头）

```bash
# 第一部分：训练池锁、GPU 的 UUID 在脚本里；输出 $D/exp-recipe-stability-20261112/
setsid nohup bash experiments/2026-11-12-recipe-stability/run_offline.sh > /dev/null 2>&1 < /dev/null &
# 指标（读第一部分的输出；--result-dir 是入库目录）
.venv-tools/bin/python experiments/2026-11-12-recipe-stability/stability.py report \
  --out-dir $D/exp-recipe-stability-20261112 --ref $D/exp-student-select-20261109 \
  --result-dir experiments/2026-11-12-recipe-stability/results
# 第二部分：第一部分的结果提交之后才能起；输出 $D/exp-recipe-stability-20261112/closed-loop/
mkdir -p $D/exp-recipe-stability-20261112/closed-loop && cd $D/exp-recipe-stability-20261112/closed-loop && \
  setsid nohup bash <仓库>/experiments/2026-11-12-recipe-stability/run_closed_loop.sh > /dev/null 2>&1 < /dev/null &
```

## 与任务卡的差别（详见 `RESULT.md` 与过程文档）

- 训练器拒绝卡里 R2、R3 的显式 0 权重（训练集里 SAFE 类有样本：训练器状态行写 8205 个训练样本，参照 run `config.json` 的 auto 权重按 N/(4·n) 反推 SAFE 为 56 个；反推过程在 `run_offline.sh` 文件头），两个臂没有按卡里的写法跑成；另跑了标明「替代」的 R2e、R3e（SAFE 权重 1e-6），它们不进卡的过线判决。
- 20 个几何按几何组 ID 挑，不按 seed 区间挑：base 任务只有 804 个不同几何，高位 seed 重复低位几何。
- 第二部分的起跑条件沿用 `run_r2.sh` 的 GPU、配额、机器锁，去掉「容器 CPU < 1.5 核」，原因写在 `run_closed_loop.sh` 条款 P7。
