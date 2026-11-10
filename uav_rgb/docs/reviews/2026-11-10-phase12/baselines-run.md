# baselines-run：外部基线包装器入库、离线核对、零样本实验排队

- 时间盒：2026-11-10 20:04 起 50 分钟；初稿写于 20:36，起跑前修订（主会话 20:41 批准）后于 20:53 更新臂表、守护进程与占位符，状态以文末「验证到哪一步」为准
- 分支 `p12-baselines`（基于 main 155e377，已含航向护栏），worktree `/root/claude-code-backup-20261028/code/wt/p12-baselines`，没有推送，没有碰 main
- 提交：57cf134 `feat(baselines)`；8991e00 `feat(experiments)`；96774f0 `docs(worklog)`（第三个是过程文档，派工只要求两次提交，多出的一次放过程文档）；4926b36 `fix(experiments)`（起跑前修订：去 `dronet-guard2`、等两个 `DONE`、5 车道，同步 `summarize.py`、README、worklog）
- 草稿与离线产物：`/root/rivermind-data/uav_rgb/tmp/phase12-20261110/baselines-run/`
- 实验输出目录：`/root/rivermind-data/uav_rgb/exp-baselines-zeroshot-20261110/`

## 一、结论

1. 包装器、5 个启动器、README、测试已入库（57cf134）。`check.sh --python-only` rc=0（20:31 一次，起跑前修订后 20:49 复跑一次）。没有改 `collect.py`、`eval_closed_loop.py`、`scene_gen.py`。
2. 转向符号：PULP v3 与 Tiny-PULP 实测为 +1（柱在裁剪视场右侧时 steer 更大，95% 区间不含 0）；DroNet 判不出（裁剪视场内左右两组均值差 −0.0002，区间 [−0.064, 0.073]），沿用 +1。判不出本身说明 DroNet 在我们的画面上没有有效转向信号。
3. 碰撞概率：DroNet 饱和，中位数 0.9994、99.97% 的帧大于 0.7，忠实映射下单帧 FORWARD 占比 0.000（SAFE 0.871，左转 0.127），闭环里会原地不动。PULP v3 与 Tiny 单帧 FORWARD 占比 0.563 与 0.636，会前进，不会停在原地。
4. 重标定臂（阈值取中位数、S=90）单帧左转占比 0.735，预计大多在原地左转；它排在队尾；主会话 20:41 已确认保留。
5. 起跑前按主会话 20:41 的意见修订了预注册（此时没有任何闭环数据，改动记在 `run.sh` 文件头）：去掉 `dronet-guard2`，臂数 7 变 6；起跑条件改为同时等 `exp-rule-oracle-20261110/DONE` 与 `exp-long-course-20261110/DONE`；车道 2 变 5；臂序 pulp_v3、pulp_tiny、pulp_v3-guard2、pulp_tiny-guard2、dronet、dronet-recal。旧守护进程（PGID 1433245）20:50:01 已按 PGID 停掉，`ps` 里无残留；新守护进程 PGID 1490902（20:50:08 开始，6 个臂预检通过，20:50:29 进入等待）。预计：long-course 在 21:40 到 22:00 结束（主会话估计，未核实），届时起跑；PULP 的 4 个臂约 25 到 30 分钟出数（主会话估计；我按 heading-guard 实验每臂 18 到 20 分钟推算，未核实）；整个队列约在起跑后 1 到 1.2 小时结束（我的估算，未核实：`dronet-recal` 要等第一个车道空出才起，两个 DroNet 臂每集跑满 80 步）。等待上限 3 小时从 20:50:29 起算，到约 23:50。
6. 闭环结果还没有。预注册预测：零样本走廊内成功不超过 6/20 且低于学生加护栏；证伪线 12/20。

## 二、交付物

| 路径（仓库内，worktree 下） | 作用 |
| --- | --- |
| `sim/tools/baselines/baseline_infer.py` | 包装器：说 `uavrgb_infer` 的行协议；配置区（环境变量可覆盖、每参数一行带出处）、映射纯函数 `map_steer_coll` / `map_velocity`、`PulpDronetV3`（v3、tiny）与 `DroNet` 适配器。`da2_rule` 没有放入（草稿里只有骨架、没跑通），要用时按任务卡 B 新写 |
| `sim/tools/baselines/launch_{pulp_v3,pulp_tiny,dronet,dronet_recal,forward}.sh` | 一行启动器，环境变量写死，teacher-venv 的 python |
| `sim/tools/baselines/README.md` | 协议、映射规则与每个阈值的出处、符号证据、怎么加基线、权重下载与转换命令、`stub.ckpt` 一行命令 |
| `sim/tools/test_baseline_infer.py` | 9 个测试函数（`check.sh` 按 `sim/tools/test_*.py` 平铺收集，所以放这一层） |
| `experiments/2026-11-10-baselines-zeroshot/{run.sh,summarize.py,README.md}` | 实验入口（预注册在 `run.sh` 头部）、汇总、说明 |
| `experiments/2026-11-10-baselines-zeroshot/{offline_check.py,check_h5_vs_npz.py}` | 离线核对与 DroNet 权重对拍 |
| `docs/worklog/2026-11-10-2035-baselines-zeroshot.md` | 过程文档 |

第三方源码与权重留在 `/root/rivermind-data/uav_rgb/baselines/`，仓库里没有二进制，`stub.ckpt` 由 `run.sh` 现场生成（`.gitignore` 已挡 `*.ckpt`，`git check-ignore -v` 的结果见「验证到哪一步」）。

## 三、离线核对（不起 Gazebo）

数据：`/root/rivermind-data/uav_rgb/e1-pipeline-20261107-0417/collection/train`，255 集、10,200 帧，全取不抽样（`offline_check.py meta`，`EP_STRIDE=1`）。教师动作占比 SAFE/FORWARD/左/右 = 0.021/0.779/0.099/0.101。每帧用 `scene.json` 与 `pose_before`、`yaw_before` 算最近柱的方位角与距离。基线对每帧原图与左右翻转图各推理一次（`offline_check.py infer <基线>`，CPU，2 线程）。全部数值在 `offline/offline_check.json`，日志 `offline/analyze_final.log`；自助法固定 seed 0、2,000 次、按集重采样，所以复跑得到同一组数。

### 3.1 转向符号与镜像检验

分组：最近柱（3 m 内）在画面左半或右半，全画面内 1,101 / 1,122 帧；裁剪视场内（native 裁剪只保留水平 ±19°）297 / 352 帧，那是网络真正看到的范围。指标：右组 steer 均值减左组，正值 = 柱在右时网络更偏向左转（朝远离柱的方向），与「steer>0 = 左转」的约定一致。

| 基线 | 裁剪视场内 右−左（95% 区间） | 全画面内 右−左（95% 区间） | corr(steer, 翻转输入的 steer) | 翻转后变号的帧占比（n） | 奇部 std / 偶部 std / 偶部均值 | 教师转向帧上 +1 / −1 的方向一致率（n=2,041） | STEER_SIGN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pulp_v3 | +0.162（0.129, 0.194） | +0.070（0.052, 0.087） | +0.093 | 0.335（6,812） | 0.086 / 0.095 / 0.004 | 0.523 / 0.477 | +1，实测 |
| pulp_tiny | +0.193（0.176, 0.211） | +0.083（0.074, 0.092） | −0.637 | 0.360（7,342） | 0.058 / 0.027 / 0.028 | 0.536 / 0.464 | +1，实测 |
| dronet | −0.0002（−0.064, 0.073） | +0.055（0.027, 0.087） | +0.379 | 0.121（9,574） | 0.154 / 0.230 / 0.307 | 0.506 / 0.494 | 判不出，沿用 +1 |

读法：
- 镜像不翻号的原因：steer 里有一块与图像左右无关的成分（偶部）。PULP v3 的偶部与奇部同量级（std 0.095 对 0.086），Tiny 以奇部为主（0.058 对 0.027，corr −0.64，最接近镜像等变），DroNet 以偶部为主（偶部均值 0.307，steer 总均值 0.313，几乎是一个常数偏置）。
- 教师方向一致率接近 0.5：教师的转向由 A* 路径决定，不等于「最近的柱在哪一侧」，所以这一列只作旁证，不作符号依据；符号依据是前两列。
- DroNet 全画面分组的差 +0.055 区间不含 0，裁剪视场内为 0。裁剪视场外的柱不在网络的输入里，全画面差的成因推测是裁剪视场内另有与最近柱同侧的可见柱，未核实，不当作转向能力。

### 3.2 碰撞概率分布与默认映射下的动作占比

「近处正前方有柱」= 最近柱距离 ≤ 1.2 m 且方位角绝对值 ≤ 15°（74 帧）；「无柱」= 3 m 内没有柱（3,891 帧）。AUROC 用秩和法，近处正前方有柱 vs 无柱。动作占比用忠实映射（PULP S=90，DroNet S=20，阈值 0.7，门槛 12.5°，符号 +1）；符号翻转只会把左右占比对调。

| 基线 | p_coll 中位数 / p10 / p90 | p_coll > 0.7 | 近处有柱 vs 无柱的中位数 | AUROC | steer 均值 / std | SAFE / FORWARD / 左 / 右 | 教师转向帧上 同向 / 反向 / 不转 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pulp_v3 | 0.353 / 0.079 / 0.981 | 0.342 | 0.941 vs 0.236 | 0.944 | 0.004 / 0.130 | 0.205 / 0.563 / 0.110 / 0.122 | 0.150 / 0.101 / 0.749 |
| pulp_tiny | 0.532 / 0.147 / 0.937 | 0.352 | 0.870 vs 0.264 | 0.973 | 0.028 / 0.064 | 0.309 / 0.636 / 0.037 / 0.017 | 0.090 / 0.018 / 0.893 |
| dronet | 0.99945 / 0.99676 / 0.9999995 | 0.9997 | 0.99968 vs 0.99916 | 0.626 | 0.313 / 0.276 | 0.871 / 0.000 / 0.127 / 0.001 | 0.042 / 0.071 / 0.887 |
| dronet-recal（P_CRIT=0.9994456，S=90） | 同 dronet | 阈值已改，按定义约 0.5 的帧高于阈值 | 同 dronet | 同 dronet | 同 dronet | 0.066 / 0.156 / 0.735 / 0.043 | 未算 |
| 教师（参照） |  |  |  |  |  | 0.021 / 0.779 / 0.099 / 0.101 |  |

读法：
- 零样本臂会不会原地不动：DroNet 会（FORWARD 0.000，87% 的帧 SAFE，13% 的帧左转，是常数偏置 0.313 乘 S=20 后只在 steer 高于 0.625 的尾部触发转向）；PULP v3 与 Tiny 不会（FORWARD 0.56 与 0.64）。
- PULP 的碰撞头在我们的画面上有区分度（AUROC 0.944 与 0.973），但绝对水平偏高：无柱的帧中位数 0.24 与 0.26，大于 0.7 的帧占 34% 与 35%，会无故刹停；Tiny 的转向幅度小（steer std 0.064，只有 5% 的帧过 12.5° 门槛），预期转不开。
- DroNet 的碰撞头饱和，区分度弱：AUROC 0.626，近处有柱与无柱的中位数 0.99968 对 0.99916；复核用的 60 帧里碰撞 logit 最大到 17.0（`dronet_verify.json`）。
- 重标定臂（阈值取中位数 0.9994456、S=90）：在偏置 0.313 上乘 90 得 28°，高于 12.5° 门槛，73.5% 的帧左转，预计在原地转圈。派工把它定为次要臂，原样放进队列，风险见第五节。

### 3.3 DroNet 前向的独立复核

- 权重：`dronet_best_weights.npz` 与原 Keras `best_weights.h5` 逐数据集对拍：48 个数据集 vs 48 个键，缺 0、多 0、数值不等 0（`offline/h5_vs_npz.txt`，`check_h5_vs_npz.py`，`venv-h5` 执行）。
- 前向：包装器的 torch 前向与一份独立的 NumPy 层图解释器（按 `model_struct.json`，TF same 填充，BN eps 1e-3）在 60 帧上逐层对拍，最大绝对差：conv2d_1 7.4e-7、add_1 1.3e-6、add_2 2.3e-6、add_3 2.6e-6、steer 1.8e-6、碰撞 logit 5.0e-6、p_coll 7.4e-8（`offline/dronet_verify.json`；对应激活最大绝对值 2.5、3.9、4.8、6.4、1.0、17.0）。
- 没覆盖：与 Keras/TF 原模型的对拍。本机没有 TF 1.x，所以两份实现共享「我对 Keras 层语义的理解」这一前提（same 填充、BN、Flatten 的 NHWC 顺序、sigmoid）；这个前提没有被第三方实现验证过。

### 3.4 据此定下的启动器参数

| 启动器 | STEER_SIGN | 其他 |
| --- | --- | --- |
| `launch_pulp_v3.sh` | +1 | 默认 S=90、阈值 0.7 |
| `launch_pulp_tiny.sh` | +1 | 同上 |
| `launch_dronet.sh` | +1（判不出，沿用约定） | 默认 S=20、阈值 0.7 |
| `launch_dronet_recal.sh` | +1 | `P_CRIT=0.9994456`（校准帧中位数 0.99944562 取 7 位小数），`YAW_SCALE_DRONET=90` |

## 四、闭环实验

### 4.1 设计

入口 `experiments/2026-11-10-baselines-zeroshot/run.sh`（预注册、预测、证伪线、主比较在文件头）。评测命令与 heading-guard 实验一致：`--task base --split test`，20 个冻结 seed `1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654`，`--steps 80 --hold 1.0 --window-from observation --dynamics cf_sitl --vehicle e1_mini --corridor-half-width 3.0`，`--policy student --ckpt stub.ckpt --infer-bin <启动器>`，口径 `sim09.v2.coast1s.corridor3m`，分母 = 请求集数 20。代码用 `git archive 57cf134` 冻结到 `$D/tmp/phase12-20261110/baselines-run/code-57cf134`。

| 序 | 臂 | 启动器 | 护栏 |
| --- | --- | --- | --- |
| 1 | pulp_v3 | launch_pulp_v3.sh | 无 |
| 2 | pulp_tiny | launch_pulp_tiny.sh | 无 |
| 3 | pulp_v3-guard2 | launch_pulp_v3.sh | `--heading-guard 2` |
| 4 | pulp_tiny-guard2 | launch_pulp_tiny.sh | `--heading-guard 2` |
| 5 | dronet | launch_dronet.sh | 无 |
| 6 | dronet-recal（次要，队尾） | launch_dronet_recal.sh | 无 |

起跑前修订前是 7 个臂（多一个 `dronet-guard2`，臂序为 pulp_v3、pulp_tiny、dronet、三个护栏臂、dronet-recal）。

参照不重跑，`summarize.py` 直接读已有 `eval.json`：学生 calswa 39/60、calswa + 护栏 50/60（已用 `summarize.py` 复算，与 `exp-heading-guard-20261110/summary.md` 一致）；学生 calswa-s1 + 护栏 17/20 作配对参照；教师 20/20 与盲前进 0/20 取 11-07 eval4（口径 `sim09.v2.coast1s`，没有走廊判据）。参数量：320,928 / 2,784 / DroNet 321,634 个张量元素（含 BN 统计，约 32 万）；我们的学生约 10 万取自派工消息，未独立核实。

预注册要点：预测零样本臂走廊内成功不超过 6/20 且低于学生加护栏；证伪线 12/20；主比较 3 个（各臂对 6/20 与 12/20 两条线；零样本最好臂对学生 calswa-s1 + 护栏的配对 McNemar；护栏臂对无护栏臂的配对 McNemar，修订后只对 PULP 的两个基线）。n=20 不下确定结论。离线核对给出的臂级预期也写在文件头：DroNet FORWARD 占比 < 5% 且 0/20，失败以未到线为主。

起跑前修订（2026-11-10 20:44，主会话批准，当时没有任何闭环数据）改了三处：去掉 `dronet-guard2`（离线单帧 FORWARD 占比 0.000，护栏没有作用空间，每集空耗 80 步）；起跑条件加上 long-course 的 `DONE`；车道 5。预测、证伪线、主比较 ① ② 不变；原文留在 `run.sh` 文件头，修订块写明时间与理由。冻结代码不变：修订只改 `experiments/` 与 `docs/`，`git diff --stat 57cf134 HEAD -- sim tools` 为空，仍用 `code-57cf134`。

### 4.2 守护进程

- 启动命令：`cd /root/rivermind-data/uav_rgb/exp-baselines-zeroshot-20261110 && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &`
- PGID：1490902（`run.pid`），确认命令 `ps -o pid,pgid,etime,cmd -p $(cat run.pid)`：`1490902 1490902 00:23 bash run.sh`（20:50:31 实测，PID 与 PGID 相同，进程在预检之后的等待循环里，另有一个 `sleep 60` 子进程）。旧守护进程 PGID 1433245（20:33:28 起，只做过预检与等待，没有起过任何臂）20:50:01 用 `kill -- -1433245` 停掉：该进程组在 `ps` 里无残留，`$OUT/.lock` 的 flock 已释放；旧的 `inputs.txt`、`run.sh`、`summarize.py` 留作 `*.rev0-pgid1433245`，没有删任何文件
- 日志：`$OUT/run.log`（进度），每臂 `$OUT/<臂>.log` 与 `$OUT/<臂>/eval.json`，闭环里基线每步输出 `$OUT/<臂>.baseline.jsonl`（按 `pid` 切开各集，`n=0` 是预热帧）
- 停：`kill -- -1490902`（只发给进程组，不用 `pkill -f`）；续跑：同一条启动命令，评完的臂跳过，半成品挪到 `<臂>.partial-<时刻>`
- 汇总：`cd $OUT && /root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py`，结束时 `run.sh` 自己跑并落 `summary.md` 与 `DONE`
- 起跑条件：`exp-rule-oracle-20261110/DONE` 与 `exp-long-course-20261110/DONE` 两个文件都出现；每 60 秒查一次，每 10 分钟在 `run.log` 写一行（列出还缺哪个）；等待上限 3 小时（从 20:50:29 起，到约 23:50），超时落 `DONE`（rc=5，写明还缺哪个文件）并退出，不起任何臂
- 车道与臂序：5 条车道，前 5 个臂（pulp_v3、pulp_tiny、pulp_v3-guard2、pulp_tiny-guard2、dronet）同时起，`dronet-recal` 在第一个空出的车道上起；届时规则层 v1 实验占 2 个 gz，合计 7 个，到全机上限
- 预计时间：2026-11-10 20:50:08 开始，6 个臂预检 20:50:11 到 20:50:29 通过后进入等待。起跑 = 两个 `DONE` 里较晚的那个：rule-oracle 约 20:50、long-course 21:40 到 22:00（均为主会话估计，未核实），所以约 21:40 到 22:00 起跑。PULP 的 4 个臂并行约 25 到 30 分钟出数（主会话估计），即约 22:05 到 22:30；整个队列约起跑后 1 到 1.2 小时结束，即约 22:40 到 23:10（我的估算：`dronet-recal` 在第一个车道空出后才起，两个 DroNet 臂每集跑满 80 步，未核实）

### 4.3 起跑后第一集怎么验证，失败怎么查

第一个臂的第一集是端到端验证（真实 Gazebo、真实 `StudentPolicy`、真实启动器）。预检已经在起跑前验过：参数解析、启动器与解释器、权重加载、协议握手与预热（`eval_closed_loop.py` 的探针路径，不起 gz）。
起跑后看这几处：
1. `run.log` 出现 `pulp_v3 开始`，随后 `<臂>.log` 里有 `[seed 1] ...` 行；
2. `<臂>.baseline.jsonl` 在几十秒内出现带新 `pid` 的行（每集一个 pid），字段 `steer`、`p_coll`、`action`、`ms`；
3. `<臂>/episodes/seed_1/actions.jsonl` 里 `action_executed` 是基线的动作，`action_expert` 是影子教师的标签，`infer_ms` 有值；
4. 失败原因：`StudentPolicy` 把推理进程的 stderr 并进异常文本，写进 `<臂>.log` 与 `eval.json` 的失败记录；包装器退出时在 stderr 写 `baseline_infer[<基线>]: N 帧`。

## 五、风险与需要主会话决定的事

1. **dronet-recal 预计原地左转**：单帧左转占比 0.735。主会话 20:41 已确认保留，排在队尾，占一条车道约 30 到 40 分钟（估算）。要删就从 `run.sh` 的 `ARMS` 去掉一行，起跑前改，先 `kill -- -<PGID>` 再重起（预检会再跑一遍）。
2. **DroNet 两个臂（dronet、dronet-recal）预计前进很少**：dronet 单帧 FORWARD 占比 0.000，每集会跑满 80 步，比成功的集慢，整体耗时偏长；护栏对只会 SAFE 的网络没有作用空间，所以 `dronet-guard2` 在起跑前去掉。
3. **零样本只量域差异**：native 裁剪只保留 60° 视场里的中间 ±19°，柱阵的侧擦在视场外发生时网络看不到；报告不能把零样本的失败写成结构劣势。`BASELINE_PREP=resize_crop` / `squash` 的敏感性没做（见「没做什么」）。
4. **CPU 争用**：机器负载约 24（`uptime`），gz 软件渲染占 CPU，基线闭环里用 1 个线程。离线 2 线程、同样的负载下，整遍耗时除以 20,400 次前向（原图加翻转图，含读图与预处理）：PULP v3 约 46 ms（942 s）、DroNet 约 13 ms（259 s）、Tiny 约 6 ms（129 s），均远低于协议的 30 秒超时；闭环里 1 线程的耗时没测过。
5. **DroNet 符号判不出**：+1 是约定，不是实测；闭环里的左右行为不能解释成「学到了避障」。
6. **没有 Keras/TF 原模型对拍**：DroNet 的数值正确性只靠两份我写的实现互相核对，以及 h5 与 npz 逐元素一致。
7. 合并：本分支只新增文件（`sim/tools/baselines/`、`sim/tools/test_baseline_infer.py`、`experiments/2026-11-10-baselines-zeroshot/`、一份 worklog），没有修改任何已有文件，和别的 worktree 不会冲突。`progress.md` 没改（见「没做什么」）。
8. **gz 配额与等待上限靠排期保证**：5 条车道的前提是起跑时全机只剩规则层 v1 的 2 个 gz；`run.sh` 起跑前不数全机 gz（车道数按主会话的排期写死，改成按 `pgrep gz` 动态调车道超出这次修订的范围），届时若还有别的实验在跑会超过 7 个。等待上限从 20:50:29 起算，到约 23:50，比 long-course 预计结束（21:40 到 22:00，未核实）晚约 2 小时；long-course 拖过 23:50 则本实验落 rc=5、不起任何臂。

## 六、后续任务卡

### 卡 A：同结构在我们数据上重训（PULP v3 两个尺寸 + 三条带碰撞头）

目的：把「公开权重的域差异」和「结构本身的能力」分开。本轮零样本臂只量前者。重训臂用同一批训练数据、同一套评测协议，回答：320,928 与 2,784 参数的 PULP 结构在我们数据上能走多远，以及「三条纵带各出一个碰撞概率」的输出头（Lamberti 等 2023，arXiv 2312.08991）对比「转向回归 + 一个碰撞概率」的两头。

前置与已有：
- 包装器 `BASELINE_WEIGHTS=<.pth>` 已经能给 `pulp_v3`、`pulp_tiny` 加载同结构权重（`state_dict` 或含 `state_dict` 键的字典），映射与启动器不变；复制 `launch_pulp_v3.sh` 另起一个启动器写死权重路径即可。三条带头需要新适配器（见下），不在本轮。
- 实验入口照 `experiments/2026-11-10-baselines-zeroshot/run.sh` 复制改臂表，评测协议与本轮逐项相同，才能与零样本臂、学生做配对比较。

数据来源（已核对布局）：`/root/rivermind-data/uav_rgb/e1-pipeline-20261107-0417/collection/{train,val,test}/<seed>/`，每集有 `rgb_%05d.png`（320×240 RGB）、`actions.jsonl`（`action_expert` 教师动作 0..3、`pose_before`、`yaw_before` 弧度、`collision`）、`oracle.jsonl`（真值位姿）、`scene.json`（`pillars`）。train 255 集 10,200 帧，val、test 分区同结构。训练只用 train，超参与早停只用 val，不看 20 个冻结评测 seed（它们在 test 分区）。更多数据（DAgger 各轮、more_demos）在 `/root/rivermind-data/uav_rgb/exp-dagger-gz-20261109/`；我们的学生所用训练集的确切组成**未核实**，动手前先从该实验的配置确认，重训臂必须与学生同一份数据才是公平对照。注意：episode 目录里没有逐帧深度图文件（`actions.jsonl` 有 `depth_ts` 字段，说明采集时有深度，但没落盘，是否存在于别处**未核实**）。

标签派生：
- 转向标签：教师动作为 YAW_LEFT 取 +25/(90×1.0)≈+0.278，YAW_RIGHT 取 −0.278，FORWARD 与 SAFE 取 0（一个周期内按 S=90 °/s 折算的 25° 步长）。这是三值标签，比 PULP 原文的真机转向标注粗，要在报告里写明。
- 碰撞标签（两头版）：直接复用 `experiments/2026-11-10-baselines-zeroshot/offline_check.py` 的 `build_meta`（逐帧最近柱的方位角与距离，来自 `scene.json` 与位姿）；标签 = 1 当最近柱距离 ≤ 1.2 m 且方位角绝对值 ≤ 15°。该定义是本轮离线核对里算 AUROC 的设定，不是 PULP 原文的定义（原文用真机标注，未核实），阈值要在 val 上看正例比例后再定，正例过少（本轮 train 里只有 74 帧近处正前方有柱）就放宽距离。
- 三条带版：把 60° 水平视场按列切成三条（约 [−30°, −10°]、[−10°, 10°]、[10°, 30°]），每条一个碰撞标签。Lamberti 的定义是「条内 10% 的像素距离 ≤ 2 m」，需要逐像素深度，现有数据没有。退一步用几何近似：条内存在最近柱表面距离 ≤ 2 m 记 1（需读 `scene.json` 的柱半径字段，字段名**未核实**）；或者用 `eval_closed_loop.py --save-depth`（该选项在 `eval_closed_loop.py` 的参数表里）重采一批帧，`collect.py` 是否有同名选项**未核实**。三条带头的决策规则：转向朝碰撞概率最小的条带，三条都大于阈值 T 则 SAFE；T 在 val 上从 {0.3, 0.5, 0.7} 选一个冻结，论文值本次没读到。

训练脚本位置与环境：
- 新建 `sim/tools/baselines/train_pulp.py`，不改第三方源码。PULP 官方训练代码在数据盘 `/root/rivermind-data/uav_rgb/baselines/pulp-dronet/src/tiny-pulp-dronet-v3/{classes.py,utility.py,config.py,scripts.sh}`，读 Udacity 与碰撞数据集；损失与训练循环可以参照，数据加载要换成上面的标签，损失权重与 epoch 数以 `config.py` 为准（本任务没有逐项核对）。
- 解释器：`/root/rivermind-data/uav_rgb/teacher-venv/bin/python`（有 torch 与 CUDA，只读使用，不装包）；只用 GPU `GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70`，`CUDA_VISIBLE_DEVICES` 按 UUID 钉。两个网络都很小（41.1 M MAC 与 1.13 M MAC，`count_params.out`），CPU 也可训，训练时长**未测**。
- 守护进程方式、`flock`、`DONE`、frozen code 目录照本轮 `run.sh`；检查点放 `$D/<实验目录>/ckpt/`，记 sha256，单任务新增不超过 3 GB（检查点约 1.3 MB 与 24 kB 量级）。

臂：`pulp_v3` 两头、`pulp_tiny` 两头、`pulp_v3` 三条带（可选 `pulp_tiny` 三条带），各自带与不带 `--heading-guard 2`；每臂 20 个冻结 seed。至少 2 个训练种子，以便估训练方差。

验收标准：
1. 离线（val 分区）：steer 与教师方向的一致率、碰撞头 AUROC，与本报告 3.1、3.2 的零样本数同口径逐项对比；
2. 闭环：每臂 20 集全有结果行，失败集写原因，分母 20；报告 FORWARD 占比与护栏改写占比；
3. 对零样本臂与学生 calswa-s1 + 护栏做同 seed 配对精确 McNemar，报 b、c 与 p；n=20 不下确定结论；
4. 两个训练种子的闭环成功数差不超过 2/20，超过则如实报告训练方差；
5. 训练日志、检查点哈希、`inputs.txt`（提交号与文件哈希）落盘；预注册（预测、证伪线、≤3 个主比较）先写进入口脚本头部再起跑。

### 卡 B：Depth-Anything-V2 Small 米制深度 + 规则

目的：测「单目米制深度基础模型 + 固定规则」这条无训练路线在我们的柱阵里能到哪，并和「真值深度 + 同一条规则」的上界对比，判断瓶颈在估计还是在规则。

规则：同一条规则就是 `exp-rule-oracle-20261110` 用的规则 v0（`sim/tools/rule_policy.py`，参数 `RULE_V0` 在文件顶部，在分支 `p12-ruleoracle` 提交 18f2ec0，尚未合入 main；规则内容见该实验 `run.sh` 头部：通道内有柱则 blocked，朝左右两半 Σmin(d, sum_clip) 较大的一侧转，|n|≥n_max 朝回转）。它的输入是「每列距离 d_j 与方位角 φ_j」（60° 视场 20 列，81° 视场 27 列）。真值深度版本就是该实验本身（柱子真值算出的列距离），所以不用新建真值臂；该实验的结果在 `exp-rule-oracle-20261110/summary.md`，起跑时可能还没出，先确认。

数据源与标签：本卡不训练。需要的数据是闭环里的 RGB 帧与对应的真值列距离：
- 估计误差评估：在 `e1-pipeline-20261107-0417/collection/train` 的帧上（每集抽若干帧，共 100 帧左右），用 `scene.json` 与位姿算真值列距离（`offline_check.py` 的 `build_meta` 已能算最近柱方位与距离，列距离需要在它基础上按柱的几何展开，`rule_policy.py` 里有真值列距离的现成实现，复用它，不重写）；
- DA-V2 输出的米制深度图 → 20 列距离：每列取柱体所在的行带（水平飞行、相机朝前，柱子是竖直灰色圆柱，地面在下半幅）内的低分位数（候选 p10 与最小值，在 train 帧上选一个冻结），再按列的方位角换成沿机体前向与侧向的分量。
- 米制尺度：Hypersim 室内权重（`max_depth=20`）与 VKITTI 室外权重（`max_depth=80`）都跑；柱阵是无纹理灰色圆柱，两个权重的尺度可能差很多，所以都报。

资产与下载（均小于 500 MB，先 HEAD 看体积再下，走镜像）：

```bash
D=/root/rivermind-data/uav_rgb; mkdir -p $D/baselines/depth-anything-v2/weights $D/baselines/depth-anything-v2/src
curl -fL -C - -o $D/baselines/depth-anything-v2/weights/depth_anything_v2_metric_hypersim_vits.pth https://hf-mirror.com/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
curl -fL -C - -o $D/baselines/depth-anything-v2/weights/depth_anything_v2_metric_vkitti_vits.pth https://hf-mirror.com/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth
curl -fL -m 600 -o $D/baselines/depth-anything-v2/src.zip https://ghfast.top/https://github.com/DepthAnything/Depth-Anything-V2/archive/refs/heads/main.zip
```

下载命令摘自 `reports/baselines.md` 的任务卡 B3，解压与目录整理的命令也在那里；这些命令本轮没有执行，权重尚未下载，体积数字（各 99.2 MB）来自该报告，未重新核实。

适配器位置与环境：在 `sim/tools/baselines/baseline_infer.py` 新增 `Da2Rule` 适配器并登记进 `REGISTRY`，加一个 `launch_da2_rule.sh`（`DA2_WEIGHTS`、`DA2_MAX_DEPTH` 写死）。没有训练脚本。解释器 `teacher-venv`（torch 2.14.1+cu126、torchvision 0.29.1、cv2 5.0.0，版本取自 `reports/baselines.md`，本轮没有重新核对）；DA-V2 的核心代码只依赖这些，不装包，但**未验证**能直接 import。输入边长取 14 的倍数（草稿取 266，接近 240 高度），延迟第一步先测 100 帧的均值与 p95。适配器要把深度图送进规则，规则在 `p12-ruleoracle` 分支，要等它合入 main 或把本卡的分支建在它之上。

验收标准：
1. 深度估计在 100 张 train 帧上相对真值列距离的相对误差分布（中位数、p90）写入报告，室内与室外两个权重各一份；
2. 闭环：每个权重 × `--task base` 20 个冻结 seed 全有结果行，分母 20，失败集写原因；报告走廊内成功、撞柱、FORWARD 占比；
3. 与「真值深度 + 同规则」（rule-oracle 实验，同 seed 同口径）配对比较，写出估计误差造成的成功率差；
4. 延迟（100 帧均值与 p95）写入报告；
5. 预注册先行：预测（例如「估计深度版本不低于真值版本的一半」，这个阈值是我的设定，没有出处）、证伪线与 ≤3 个主比较写进入口脚本头部。

## 做了什么

1. 把草稿包装器整理入库：配置区（环境变量覆盖、每参数一行带出处）、映射纯函数、PULP v3 / Tiny / DroNet 适配器；`BASELINE` 无默认值，漏设报错；去掉了 `da2_rule`、`trailnet` 与队列规则的骨架；加了 `pid` 字段便于按集切开 `BASELINE_LOG`。
2. 5 个一行启动器，README（协议、映射与阈值出处、符号证据、怎么加基线、下载与 h5 转 npz 命令、`stub.ckpt` 一行命令）。
3. 测试 9 个函数：映射纯函数、forward 经真实 `StudentPolicy`、探针适配器字节级往返、EOF 与截断帧退出码、启动器静态检查，权重在时的真实启动器往返。
4. 离线核对：mirror、符号、碰撞概率分布、动作占比、DroNet NumPy 复核与 h5 对拍（第三节）；据此写定三个启动器的符号与重标定阈值。
5. 实验入口 `run.sh`（预注册、预检、排队、frozen code、`inputs.txt`；起跑前修订后为 6 个臂、等两个 `DONE`、5 车道）与 `summarize.py`；预检用正式命令行跑 `eval_closed_loop.py` 到非空 `--out` 报错为止，不起 gz。
6. `check.sh --python-only`、提交、启动守护进程并确认存活。
7. 两张后续任务卡。
8. 起跑前修订（主会话 20:41 的要求）：改 `run.sh` 文件头（修订时间 20:44、理由、原文保留）、臂表、等待与车道，同步 `summarize.py`（臂清单、护栏配对去掉 dronet 一对）、实验 README、worklog，一个 `fix(experiments)` 提交 4926b36；按 PGID 停旧守护进程并确认无残留；用新脚本重起，6 个臂预检通过、进入等待；报告里残留的占位符（提交哈希、PGID、`check.sh` 返回码等）全部填成实际值，复查时报告、实验 README、worklog 三个文件里已无占位符记号。

## 没做什么（每条带原因）

- 没做 `BASELINE_PREP` 的敏感性（`resize_crop`、`squash`）闭环臂与离线对比：派工的臂表里没有，时间盒内先保证 6 个臂的预检与排队。README 与包装器已支持这两个取法，加臂即可。
- 没做 DroNet 与 Keras/TF 原模型的数值对拍：本机没有 TF 1.x，装它超出这次的体积与时间预算。
- 没跑任何 Gazebo 闭环：派工要求冒烟不起 Gazebo，且 gz 配额已被 rule-oracle 与 long-course 占满，守护进程在排队。第一集端到端验证由守护进程起跑后完成，结果不在本报告里。
- 没实现 `da2_rule`：留给卡 B，草稿骨架没跑通，入库未验证的代码不如不入。
- 没实现三条带适配器、没做重训：留给卡 A，派工明确是后续任务卡。
- 没改 `collect.py`、`eval_closed_loop.py`、`scene_gen.py`：派工禁止，另一个 agent 在改。
- 没给 DroNet 保留护栏臂：按主会话 20:41 的意见去掉，原因是离线单帧 FORWARD 占比 0.000，护栏没有作用空间；起跑后若 DroNet 的实际行为与离线预测不符，再按需补跑。
- 没重新冻结代码：修订只改 `experiments/` 与 `docs/`，`git diff --stat 57cf134 HEAD -- sim tools` 为空，仍用 `code-57cf134`，重新冻结只会多出一个内容相同的目录。
- 没改仓库 `progress.md`：多个 worktree 并行，同一文件的追加会在合并时冲突；本轮的进度、决策、待办写在 worklog 与本报告里，合并时由主会话汇入。
- 没推送、没动 main、没删任何已有文件或分支。
- 没写 `RESULT.md`：闭环结果还没有，写了就是编的。
- 没把 `inf_*.npz`、`meta.npz` 入库：数据盘产物，体积与再生成成本见各自命令，不进仓库。

## 验证到哪一步

| 项 | 命令或来源 | 结果 |
| --- | --- | --- |
| 测试文件 | `$R/.venv-tools/bin/python sim/tools/test_baseline_infer.py` | PASS；含 forward、pulp_v3、pulp_tiny、dronet、dronet_recal 五个真实启动器经 `StudentPolicy` 的往返（各 3 行日志） |
| 红到绿（真实） | 开发中的两次失败输出 | ① 探针适配器测试：`ModuleNotFoundError: No module named 'torch'`（`main()` 对所有非 forward 基线都 import torch），把线程设置移进 `_import_torch()` 后通过；② 启动器静态检查：`ValueError: could not convert string to float: '0.9994456：DroNet'`（正则命中了注释行），改成只解析 `export` 行后通过 |
| 变异检查 | `baselines-run/mutation_check.py`，输出 `mutation.log` | 5 个故意改坏的版本（门槛 12.5° 改 25°、左右对调、`<=` 改 `<`、去掉非有限保护、连续速度门槛 0.125 m 改 0.25 m）全部被测试抓到；原文件与还原后均通过；存活 0 个 |
| DroNet 权重 | `check_h5_vs_npz.py` | 48 数据集一致，缺 0、多 0、不等 0 |
| DroNet 前向 | `offline_check.py dronet-verify`（60 帧） | torch 前向对 NumPy 解释器最大绝对差 ≤ 5.0e-6（逐层数值见 3.3）；与 Keras/TF 对拍没做 |
| 离线核对 | `offline_check.py analyze`，10,200 帧 | 见第三节；pulp_v3 的整遍推理 942 s |
| 预检（不起 gz） | 正式命令行 + 非空 `--out` | `eval_closed_loop.py` 报「--out ... 非空」，rc=2，3.8 s；`BASELINE_LOG` 里写入 1 行预热帧（探针经过启动器）；命令只跑 3.8 s 且在参数检查处退出，没有起 gz（命令前后 `pgrep -c gz` 为 41 与 42，多出的 1 个来自同机其他实验，未逐个核实） |
| `run.sh` 控制流 | 把 `PY` 换成假 python、两个起跑文件换成假文件的副本（`baselines-run/dryrun2/`，起跑前修订后重跑；修订前的 `dryrun/` 保留） | A 路径：6 个臂预检通过；两个文件缺一个时 `run.log` 只列缺的那个；都到之后 5 个臂同一秒起、`dronet-recal` 在第一个车道空出后起；`summarize.py` 跑通，`DONE` 写 `评完 6/6`；护栏参数只出现在两个 guard 臂的正式命令行（`pulp_v3-guard2`、`pulp_tiny-guard2`），`dronet` 为 0。B 路径（等待上限缩成 6 s，文件不出现）：`DONE` 写 rc=5 并列出缺的两个文件，起了 0 个臂 |
| `summarize.py` | 对参照臂与 3 个借来的真实 eval 目录跑 | 参照复算：calswa 39/60、calswa + 护栏 50/60、calswa-s1 + 护栏 17/20、教师 20/20、盲前进 0/20；配对 McNemar 对手算值 b=1、c=3 得 p=0.625 一致；各臂分支只用借来的数据验过，没有用本实验自己的 eval.json 验过。修订后的臂清单重跑：表里 6 个臂，配对表里没有 dronet 的护栏一对 |
| 仓库检查 | `tools/check.sh --python-only`（初次 `baselines-run/check_python_only.log`，20:31；起跑前修订后 `check_python_only2.log`，20:49:35，约 3 分钟，含 `test_baseline_infer: PASS`） | 两次 rc=0。中间有一次起得不对：没设 `UAV_RGB_PYTHON`，worktree 下没有 `.venv-tools`，rc=127，日志留作 `check_python_only2-envfail.log`，不算结果 |
| `.gitignore` | `git check-ignore -v <路径>/stub.ckpt` | `.gitignore:15:*.ckpt` 同时命中 `sim/tools/baselines/stub.ckpt` 与 `experiments/2026-11-10-baselines-zeroshot/stub.ckpt` |
| 守护进程（新） | `ps -o pid,pgid,etime,cmd -p $(cat run.pid)` | `1490902 1490902 00:23 bash run.sh`（20:50:31 实测，PID 与 PGID 相同；该进程组另有一个 `sleep 60`）；`run.log`：20:50:08 开始，20:50:11 到 20:50:29 六个臂预检通过，20:50:29 起「等待中，仍缺：rule-oracle 与 long-course 的 `DONE`」 |
| 旧守护进程已停 | `kill -- -1433245`（20:50:01）后 `ps -eo pid,pgid,etime,cmd` 按 PGID 与 `baseline_infer`、`launch_*`、实验目录名过滤 | 均为空；`flock -n` 取 `$OUT/.lock` 成功，说明锁已释放；停之前两个 `DONE` 都不存在，旧进程没有起过任何臂（`run.log` 无「开始」臂行） |
| 冻结代码未变 | `git diff --stat 57cf134 HEAD -- sim tools` | 输出为空；`57cf134..HEAD` 只动 `experiments/` 与 `docs/`；`inputs.txt` 记 `commit 57cf134`、`repo_head 4926b36` 与 `run.sh`、`summarize.py` 的 sha256 |
| 数据盘新增 | `du -sh`（20:51） | 草稿目录 13 MB（含冻结代码 8 MB 与离线 npz），实验目录 128 KB；根分区剩余 2.4 GB，未变 |
| 闭环第一集、各臂成功率 | 守护进程起跑后 | 未验证：还没起跑，在等两个 `DONE`（20:50 实测二者都不存在） |
