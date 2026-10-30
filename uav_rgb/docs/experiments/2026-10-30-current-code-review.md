# 当前代码审核：验证与最小复现

- 时间：2026-10-30 17:13–17:49（Asia/Shanghai）。
- 性质：代码功能验证、缺陷复现和数据文件一致性清点；不是模型性能排名或导航成绩。
- 环境：Linux、真实 NVIDIA GeForce RTX 2080 Ti、CUDA 12.8、GCC 13.3；有其他 GPU 进程及现有 T6 训练，任何耗时均不解释为吞吐变化。
- 输入版本：Git HEAD `e0d2e4e` 加工作树；完整 hash 以外部 `source-before.json` 为准。
- 结论/修法：[审核过程与六项发现](../worklog/2026-10-30-1713-current-code-review.md)。
- 证据目录：`/root/rivermind-data/uav_rgb/current-code-review-20261030-1713/`。

## 1. 独立构建与全量检查

真实执行命令：

```bash
source uav_rgb/env.sh
nvidia-smi -L
export UAV_RGB_BUILD_DIR="$UAV_RGB_ROOT/build/audit-20261030-1713"
export UAV_RGB_BUILD_JOBS=8
export TMPDIR=/root/rivermind-data/uav_rgb/current-code-review-20261030-1713/tmp
bash "$UAV_RGB_ROOT/tools/check.sh" \
  > /root/rivermind-data/uav_rgb/current-code-review-20261030-1713/check.log 2>&1
```

首次运行前已创建证据目录与 `tmp/`。`TMPDIR` 让测试合成数据落在代码目录之外；`check.sh` 自动清理自己的临时目录。构建不复用 `src/build` 或其他会话的 `build/check`。

关键真实输出：

```text
CHECK_EXIT=0
100% tests passed, 0 tests failed out of 20
Total Test time (real) = 183.65 sec
test_monitor: 74 项通过，1 项跳过
test_trainer_demo: 89 项通过
PASS: 所选检查完成；模拟采集、真实训练和学生闭环须独立验收。
```

20 项包含 trainer、overfit、eval-only、加权损失、数据划分/过滤、device 状态与故障收尾、BatchReader、RunOwner。Python 数据格式联检、各自检和 `sim/tools/test_*.py` 均通过；GUI 一项因没有 DISPLAY 跳过，未声称图形界面已跑。完整逐项输出保存在 `check.log`。

没有因此断言所有错误边界已覆盖。后述复现都是既有测试缺少的输入。

## 2. 审计截断漏验：生产函数 + 内存文件替身

命令形态：先 `source uav_rgb/env.sh`，再用 `"$UAV_RGB_PYTHON" -c '<复现源码>'`。完整源码、argv、退出码保存在 `python-probes-command-and-source.json` 的 `audit-truncated-oracle` 条目；输出为 `audit-truncated-oracle.log`。

输入：三个 train/val/test 内存 episode，各有 25 行 actions、24 行 oracle、manifest `num_steps=requested_steps=25`。前 24 步四类标签用当前 `plan_action` 实际生成；尾行注入 `SAFE_NEUTRAL + collision=True + depth_rgb_gap_ns=999999999`。只替换读文件函数，实际调用 `episode_facts → audit`。

真实输出节选：

```text
3 memory episodes, each actions=25/oracle=24/manifest num_steps=25
facts steps= 72 expected actions=75, discarded unsafe tail rows=3
manifest_validity PASS 0/3 集 validity 不全 true（含已排除 0 集）
time_pairing PASS max_gap=0.0 ms，超 50 ms 的步 0/72
safe_collision PASS 0/12 个 SAFE 步带 collision
geometry_version PASS uavrgb.geom.v3
split_geometry PASS 交集为 0（train=1, val=1, test=1）
label_recompute PASS 0/72 步复算标签与原标签不一致
all checks pass= True checks= 11
```

该探针退出 0 表示成功证明漏洞，不表示坏数据应被接受；没有写入或修改真实采集数据。

## 3. 慢推理动作窗口：生产函数 + 假订阅/发布者

执行方式与上项相同，完整文本在同一 JSON 的 `collect-delayed-inference` 条目，输出在 `collect-delayed-inference.log`。

复用现有 `test_collect.py` 的 `FakeSub/pose_msg/image_msg/depth_msg`。RGB/位姿观测约在 10 s，`FakeStudent.act()` 返回期间将订阅的最新仿真时刻推进至 11.5 s。publisher 仅记录派发时戳；PNG 写入用返回成功的替身，调用真实 `collect_one(hold=1,policy='student')`。

真实输出：

```text
error= None
publish records (sim-time ns, twist fields)= [(11500000000, (0.3, 0.0, 0.0, 0.0)), (11500000000, (0.0, 0.0, 0.0, 0.0))]
reported action_start_ns= 10010000000 action_end_ns= 11500000000 sim_dt= 1.49 action_wall_s= 8.041597902774811e-05
actual action and zero sim-time difference= 0
```

输出证明执行起点错误地包含策略计算期；没有启动 Gazebo，不外推为当前真实闭环的误差大小。

## 4. CPU/CUDA 数值边界

先确认 `nvidia-smi -L` 有卡，再编译临时 C++ 探针并链接本轮构建的 `libuavrgb_core.a`：

```bash
g++ -std=c++17 -O2 \
  -I "$UAV_RGB_ROOT/src/include" -I "$UAV_RGB_ROOT/src/tests" \
  <临时目录>/probe.cpp "$UAV_RGB_ROOT/build/audit-20261030-1713/libuavrgb_core.a" \
  -L/usr/local/cuda-12.8/lib64 -lcudart -ldl -lpthread -lrt \
  -o <临时目录>/probe
<临时目录>/probe numeric
```

完整 C++、实际 argv 和退出码在 `probe-command-and-source.json`；真实输出在 `numeric-probe.log`。临时文件由 `TemporaryDirectory` 自动删除，JSON 中保留源码文本便于重建，没有留下临时脚本或可执行文件。

真实输出：

```text
weighted backend=CPU scale=1 loss=1.38629 grad=[-0.375,0.125,0.125,0.125,0.125,-0.375,0.125,0.125]
weighted backend=CPU scale=3e+38 loss=1.38629 grad=[-0,0,0,0,0,-0,0,0]
clip backend=CPU norm=inf coef=0 values=[0,0]
relu_backward backend=CPU x=NaN dy=1 result=1
weighted backend=CUDA scale=1 loss=1.38629 grad=[-0.375,0.125,0.125,0.125,0.125,-0.375,0.125,0.125]
weighted backend=CUDA scale=3e+38 loss=1.38629 grad=[-0,0,0,0,0,-0,0,0]
clip backend=CUDA norm=inf coef=0 values=[0,0]
relu_backward backend=CUDA x=NaN dy=1 result=1
test_parity_helper expected=0 actual=NaN reported_max_error=0
```

各输入与判断：

| 探针 | 输入 | 正确性质/本轮判断 |
| --- | --- | --- |
| 加权 CE | 零 logits `[2,4]`、labels `[0,1]`，四个权重一起由 1 改为 `3e38f` | 归一化后梯度应不变，实际归零；CPU/CUDA 同错 |
| 裁剪 | 有限梯度 `[3e38f,3e38f]`、max_norm=5 | 正常裁剪约 `[3.5355,3.5355]`，或明确拒绝；实际静默清零 |
| 对拍 helper | `std::max(0, fabs(0-NaN))` | 有限参考对 NaN 应判失败，实际报误差 0；未伪造 kernel 结果 |
| ReLU backward | x=NaN、dy=1 | 实际返回 1；仅列合同/测试观察，未当作完整训练一定失效的证明 |

记录一次失败：最初探针编译时把 `make_train_state(model, lr)` 少写了 lr，编译退出 1，原文在 `probe-initial-build.log`。修正探针调用后编译退出 0，业务源码未改；这次编译错误不是产品缺陷，不冒充修复前失败。

## 5. 跨目录 resume 不生成 best：真实 CLI

同一临时 C++ 程序的 `setup` 模式写 8×8、4 样本的合法 UAVR（2 train / 2 val，四类接口；样本真值均为类别 0）。源模型输出类别 0，状态记录 `best_macro_f1=0.25`，同时确实存在源 `ckpt.bin/ckpt_best.bin`。

先运行生产 eval-only 验证该 best 分数，再用真实 CLI 续训：

```bash
<临时目录>/probe setup <case>
"$UAV_RGB_ROOT/build/audit-20261030-1713/uavrgb_train" \
  --data <case>/dataset.uavr --eval-only <case>/source/ckpt_best.bin \
  --out <case>/source-eval --batch 2 --split val
"$UAV_RGB_ROOT/build/audit-20261030-1713/uavrgb_train" \
  --data <case>/dataset.uavr --resume <case>/source/ckpt.bin \
  --out <case>/resumed --epochs 1 --batch 2 --lr 0.001
<临时目录>/probe inspect <case>
```

上述每个子命令均退出 0。关键真实输出：

```text
setup true_best_macro_f1=0.25 source_best_exists=1
eval-only val: ce=0.0001 acc=1.0000 macro_f1=0.2500 bal_acc=0.2500 recall=[1.000 0.000 0.000 0.000] n=2
epoch 1 dev   ce=0.0001 acc=1.0000 macro_f1=0.2500 bal_acc=0.2500 recall=[1.000 0.000 0.000 0.000]
resumed_best=0.25 last_exists=1 best_exists=0 source_best=1
REPRO_CONFIRMED: successful cross-directory resume preserved verified best score 0.25 but did not create best checkpoint
```

这是合成回归用例，四类指标只用于验证源 best 的真实性，不是模型学习成绩。完整记录在 `resume-best-probe.log`；所有输入、run 目录和临时二进制在核对后清理。

## 6. 真实 T5 的数量与 step 对齐检查

执行只读 Python 清点：递归 `/root/rivermind-data/uav_rgb/t5-full-1155/`，跳过 `failed/`，读取每个 manifest 指向的 actions/oracle；比较条数、manifest 的 num_steps/requested_steps 及对应 step。

真实输出：

```json
{"totals":{"episodes":599,"actions":23960,"oracle":23960},"problems":[]}
```

逐集清单保存在 `t5-count-integrity.json`。另读已有 `audit.json`，其记录为 599 集/23960 步、11 项 PASS、label_recompute 覆盖 23960 步且无不一致。
**已有 audit 是此前结果，本轮没有重新执行全部 A* 标签复算。** 本轮清点证明没有发现上述 actions/oracle 截断问题，不代表所有数据语义和传感器状态都在本轮重新验证。

## 7. 输入、输出与验证限制

- 输入快照覆盖源码、相关工具、配置及文档共 214 份文件，记录 SHA256、大小和 mtime；审核结束前重核，没有检测到原有文件内容在本轮变化。最终核对见 `source-after-check.json`。
- 读取了真实 Git 状态和 diff；有大量其他会话/既有未提交改动，本轮未暂存、提交、回滚或覆盖它们。
- 外部证据包括 `git-status-before.txt`、`git-diff-stat-before.txt`、`implementation-before.diff`、上述各 log/JSON；本轮二进制和证据 hash 另存 JSON。
- `tmp/` 已检查为空，临时 C++、可执行文件、合成数据和一次性进程均已退出/删除；保留的是验证记录、构建产物及源码文本的 JSON 证据。
- 没有新性能基准、Nsight、memcheck、真实 Gazebo 或完整训练；不能据此声称 GPU 利用率改善、吞吐提升或 T7 达标。
