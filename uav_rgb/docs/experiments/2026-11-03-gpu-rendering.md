# GPU 仿真渲染对照（2026-11-03）

结论：共享 Gazebo 启动入口改为 NVIDIA EGL 后，两个 campus 场景平均每步墙钟
由 4.23–4.29 秒降为 1.02 秒，约 4.2 倍。仅指本机两个场景的步骤吞吐，
不包含启动/起飞，不代表整体训练速度、复杂场景或多实例吞吐。

## 环境与产物

- 2026-11-03 13:39–13:52，Ubuntu 24.04、Gazebo 8.15、RTX 2080 Ti；
  本轮 `nvidia-smi -L` 确认有卡，未安装或替换驱动。
- 原始数据根：`/root/rivermind-data/uav_rgb/gpu-render-20261103/`。
- GPU：`gpu/`；有效软件对照：`software-legacy/`；学生短测：`student/`。
- 两次串行仿真真实 Ogre 日志分别复制为 `gpu-ogre2.log`、`software-ogre2.log`。
  前者 `GL_VENDOR = NVIDIA Corporation`、`GL_RENDERER = NVIDIA GeForce RTX 2080 Ti/PCIe/SSE2`；
  后者 `GL_VENDOR = Mesa`、`GL_RENDERER = llvmpipe (LLVM 20.1.2, 256 bits)`。
  日志仅证明本次串行运行；Ogre 默认日志共享，不能拿它证明并行各实例的后端。

## 命令与结果

在仓库根执行，下列输出目录已存在，复跑必须另取新目录：

```bash
source env.sh
UAVRGB_RENDER_BACKEND=gpu "$UAV_RGB_PYTHON" sim/tools/smoke_campus.py \
  --out /root/rivermind-data/uav_rgb/gpu-render-20261103/gpu --steps 40 --seeds 0,16
UAVRGB_RENDER_BACKEND=software "$UAV_RGB_PYTHON" sim/tools/smoke_campus.py \
  --out /root/rivermind-data/uav_rgb/gpu-render-20261103/software-legacy --steps 40 --seeds 0,16
```

两组均退出 0。相同 seed 的 SDF 字节哈希一致；RGB/depth 都是 320×240、30 Hz，
hold=1 仿真秒、dispatch，real_time_factor=1，未降低图像质量或改变物理参数。
成功后连续三个 SAFE 提前结束；不同渲染速度下异步采样使轨迹/步数略有不同，
所以比较平均每步耗时，不把完成整集用时直接作为等长轨迹加速比。

| seed / 场景 | 软件步数 | GPU 步数 | 软件秒/步 | GPU 秒/步 | 步耗时加速 | 软件窗口 RTF | GPU 窗口 RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / 建筑通道 | 24 | 27 | 4.2275 | 1.0165 | 4.16× | 0.2378 | 1.0072 |
| 16 / 广场 | 22 | 23 | 4.2918 | 1.0238 | 4.19× | 0.2343 | 1.0053 |

数据来自每集 `episode_manifest.json.performance`。RTF 为已完成动作窗口
`sum(sim_dt)/sum(action_wall_s)`，位姿采样与发布边界不完全重合，约 1% 偏差不代表突破
实时上限。软件每步约 99.5% 耗时在动作等待；取帧/解码/策略约 22–24 ms，
GPU 后为 22–26 ms。该证据支持先修渲染，而非优先优化 PNG 或学生推理。
两模式教师均 2/2 到达且无碰撞，仅为小样本验证。

```bash
"$UAV_RGB_PYTHON" sim/tools/audit_dataset.py \
  --dataset /root/rivermind-data/uav_rgb/gpu-render-20261103/gpu/episodes \
  --out /root/rivermind-data/uav_rgb/gpu-render-20261103/gpu-audit.json
"$UAV_RGB_PYTHON" sim/tools/audit_dataset.py \
  --dataset /root/rivermind-data/uav_rgb/gpu-render-20261103/software-legacy/episodes \
  --out /root/rivermind-data/uav_rgb/gpu-render-20261103/software-audit.json
```

- GPU 审计 PASS，标签复算 0/50 不一致，RGB/depth 最大残差 36 ms，
  动作时长 1.000–1.020 秒，均值 1.002 秒。
- 软件审计 FAIL 一项：`class_share` 的 YAW_RIGHT 占比低于 5%；
  标签复算 0/46 不一致，时间配对、动作窗口、目标到达等检查通过。
  保留门限；该双 seed 对照不作为合格训练集。

## 学生与渲染共享 GPU 的短程验证

```bash
UAVRGB_RENDER_BACKEND=gpu "$UAV_RGB_PYTHON" sim/tools/eval_closed_loop.py \
  --task campus --policy student \
  --ckpt /root/rivermind-data/uav_rgb/campus-implementation-20261103/train-smoke/ckpt_best.bin \
  --split test --seeds 16 --steps 3 --hold 1 --window-from dispatch --keep-work \
  --out /root/rivermind-data/uav_rgb/gpu-render-20261103/student
```

退出 0，1/1 完成，执行失败 0，3 步均 FORWARD；无碰撞，**成功 0/1**，
目标距离 4.0705 m。每步 1.031 秒；warm 推理 p50=8.07018 ms、p95=8.437 ms，
仅 n=2；首帧 9.634 ms。只验证 GPU 渲染和 CUDA 推理能同时工作，
不证明模型导航表现，也不足以评估推理长尾。

## 失败与回归

- 首次软件对照 `software/` 强制 Mesa EGL vendor，第二次 `software-fixed/`
  保留 `LIBGL_ALWAYS_SOFTWARE=1`；两次均在初始化时 `rc=-11`，堆栈为 Mesa
  `driCreateNewScreen3`。已移除这些覆盖参数，使用旧 `DISPLAY` 路径后通过。
  失败目录/日志保留；不把失败计作完成的对照样本。
- `bash tools/check.sh --python-only` 退出 0；最终软件分支修订后单独重跑
  `test_rendering.py`（5 项）和 `test_collect_parallel.py`（2 项）通过。
- 本轮没有改 C++/CUDA，因此未重跑完整 CTest；此前校园基线的 25/25 结果不冒充本轮执行。
- 本机检查 Ogre 源码的网络请求超时，改用已安装库的反汇编确认日志位置；未下载依赖。

## 后续优化顺序

1. 固定地图/相机/hold，分别测 1、2、4 实例：报告成功集/小时、有效帧/秒、
   各集 RTF、CPU/显存、推理 p95、审计失败率；不使用线性外推。
2. 若 CPU 花在图像 JSON/base64 上，再将常驻 CLI 订阅改为原生 protobuf，
   保留时间戳、接触锁存和有界缓冲。当前 RGB+float depth 原始流量约
   16.1 MB/仿真秒，base64 约 21.5 MB/仿真秒，随实例数增加。
3. 再单独研究解除 1× 实时限速。必须监控动作结束超时、观测年龄、碰撞、标签复算；
   异步推理与 20 ms 轮询在高 RTF 下会放大为更多仿真延迟。不能只调倍率就宣称等价。
4. 高精细资产/光照和更高相机分辨率按因素测算成本。GPU 渲染解决吞吐，
   不自动提供更真实的气动、风、相机标定或 sim-to-real 保证。

## 2026-11-03 14:06 审计复验

本轮 `nvidia-smi -L` 确认 RTX 2080 Ti；不重新启动仿真。磁盘重算两场景平均步耗时
比值为 4.159030504968337、4.191871044733657，GPU/软件同 seed 世界字节相等。
保存的 Ogre renderer、四集累计仿真时长与报告一致。原始 student/eval.json 顶层和
per_episode 均没有 `performance` / `render_backend_requested`。

回归命令（仓库根）：

```bash
.venv-tools/bin/python sim/tools/test_rendering.py
.venv-tools/bin/python sim/tools/test_collect_parallel.py
.venv-tools/bin/python sim/tools/test_collect.py
.venv-tools/bin/python sim/tools/test_eval_summary.py
```

输出分别为 5 tests OK、2 tests OK、26 项通过、PASS。这些通过的旧测试未覆盖以下负例。

计时复现：复用 `test_collect` 的 FakeSub/ClockPoseSub/pose_msg/image_msg，
让假墙钟和假仿真钟同步从 10.2 秒开始，Student.act 推进两钟 0.8 秒，
用 sleep 推进两钟相同时间，publisher 返回假墙钟。调用真实
`collect_one(... hold=1, policy='student', window_from='observation')`，
将 `row['timing'] | {'sim_dt': row['sim_dt']}` 送入真实 `performance_summary`。
得到以下输出，**证明统计区间错误，不是实测仿真变成 5×**：

```json
{
  "actual_sim_wall_rate": 1.0,
  "observation_age_before_dispatch_s": 0.8,
  "action_sim_s": 1.0,
  "action_wall_s": 0.19999999999999574,
  "action_rtf": 5.000000000000107
}
```

清理复现命令如下，信号函数均 mock，不会操作真实进程：

```python
import sys
from unittest.mock import Mock, patch
sys.path.insert(0, 'sim/tools')
import collect_parallel as cp

proc = Mock(pid=31337)
proc.wait.return_value = 0
with patch.object(cp.os, 'killpg') as killpg, \
     patch.object(cp.time, 'monotonic', side_effect=[0, 0, 6]), \
     patch.object(cp.time, 'sleep'), patch.object(cp, '_unregister') as unregister:
    clean = cp.stop_group(proc, 'audit-mock')
    print(clean, [int(c.args[1]) for c in killpg.call_args_list], unregister.called)
# 实际输出：False [15, 0] True —— 未发 SIGKILL，残留组却被取消登记。

with patch.object(cp.subprocess, 'run', return_value=Mock(
        stdout='424242 1 /usr/lib/gz-transport-topic -e -t /uav/camera\n')), \
     patch.object(cp.os, 'kill') as kill:
    cp._sweep_orphan_topics()
    print(kill.call_args)
# 实际输出：call(424242, <Signals.SIGKILL: 9>) —— 未核验本次任务归属。
```

通过 `git show HEAD:sim/tools/collect_parallel.py` 核对，两个清理缺陷在 GPU 改动前已存在。
审计发现及修复建议见对应 worklog 更新；本轮未修改实现，缺陷仍待修复。

## 2026-11-03 14:57 审计修复验收

本节为后续修复结果，上节“待修复”指 14:06 的审计时点。环境仍为 RTX 2080 Ti，
未安装驱动或新依赖。新产物根：`/root/rivermind-data/uav_rgb/gpu-audit-fixes-20261103/`。

### 回归

- `test_process_cleanup.py`：6/6；真实创建独立进程组，组长退出、子进程忽略 TERM，
  新实现升级 KILL；测试以 subreaper 回收自己的孤儿，并恢复设置。
- `test_eval_provenance.py`：4/4，实际运行汇总及默认目录删除流程，进程调用 mock。
- `test_rendering.py`：6/6，慢推理 observation 的 RTF 恢复 1×；dispatch 也是 1×，
  推理已吃完窗口且没有仿真推进时返回 null。旧 `sim_dt` 和 action_start 仍符合原合同。
- `source env.sh` 后执行 `bash tools/check.sh --python-only`，退出 0；完整输出 `check.log`。
  该检查与短程仿真有运行重叠，本次耗时不作为独立性能基准。

### 真实 GPU 评测，默认删除工作目录

```bash
source env.sh
UAVRGB_RENDER_BACKEND=gpu "$UAV_RGB_PYTHON" sim/tools/eval_closed_loop.py \
  --task campus --policy student \
  --ckpt /root/rivermind-data/uav_rgb/campus-implementation-20261103/train-smoke/ckpt_best.bin \
  --split test --seeds 16 --steps 3 --hold 1 --window-from dispatch \
  --out /root/rivermind-data/uav_rgb/gpu-audit-fixes-20261103/student-default-cleanup
```

退出 0，`.work` 确认不存在。最终 `eval.json.per_episode[0]` 保留以下实际输出：

```json
{
  "render_backend_requested": "gpu",
  "performance": {
    "steps": 3,
    "rtf_basis": "dispatch_pose_to_after_pose",
    "mean_step_s": {
      "io": 0.058578103159864746,
      "window": 1.0144005337109168,
      "block": 1.0147483944892883,
      "tail": 0.000044341819981733956,
      "total": 1.073452743080755
    },
    "action_sim_s": 3.0,
    "action_wall_s": 3.0432016011327505,
    "action_rtf": 0.9858038977382668
  }
}
```

1/1 集执行完成，失败 0，成功 **0/1**，碰撞 0；三步均 FORWARD。
总墙钟 28.8 秒，包含启动等成本。warm 推理 p50=12.2137 ms、p95=14.773 ms，仅 n=2。
只验证修复后真实链路及元数据留存，不推断导航性能或速度波动。
串行 Ogre 日志复制到 `student-ogre2.log`，14:53:25 的 GL_VENDOR=NVIDIA Corporation，
GL_RENDERER=NVIDIA GeForce RTX 2080 Ti/PCIe/SSE2。进程核查未发现本次仿真残留。

本轮未运行完整 C++/CUDA CTest，也未重跑软件/GPU 加速对照；此前约 4.2× 是上一节独立实验结果。
