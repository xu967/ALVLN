# 并行审计验证记录（2026-11-05）

- 环境：Linux，RTX 2080 Ti；Asia/Shanghai；主线 9d764d9。
- 类型：现有测试、只读产物核对、内存故障复现；不是新的导航性能实验。
- 本轮没有启动长训练、真实场景渲染或完整闭环性能评测。

## 1. 环境与已有构建

在 `code/uav_rgb` 执行并先 `source ./env.sh`：

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
ctest --test-dir build/check -N
ninja -C build/check -n
"$UAV_RGB_PYTHON" tools/doctor.py --require-gpu --require-sim
```

真实关键输出：

```text
NVIDIA GeForce RTX 2080 Ti
16150 MiB / 22528 MiB, utilization 51% （检查时快照）
Total Tests: 25
ninja: no work to do.
doctor: result=PASS mode=GPU
cmake 3.28.3; ninja 1.11.1; GCC 13.3.0
gazebo 8.15.0; gz.transport13 / gz.msgs10 imports OK; tkinter 8.6
data_root=/root/rivermind-data
```

doctor 结果只表明命令、导入和路径检查通过，不代表相机渲染或真实闭环通过。因存在其他 GPU 作业，采用串行注册测试，没有另开渲染和训练实验。

## 2. C++/CUDA 注册测试

```bash
ctest --test-dir build/check --output-on-failure --no-tests=error -j 1 --timeout 120
```

退出 0，真实输出：

```text
100% tests passed, 0 tests failed out of 25
Total Test time (real) = 186.52 sec
```

包含数值算子、训练器、过拟合、checkpoint 相关、软标签、文本分支、设备状态、BatchReader 和 RunOwner 已注册测试。使用现有构建；没有声称在干净机器重新构建。

## 3. Python 主检查与独立检查

```bash
PYTHONDONTWRITEBYTECODE=1 bash tools/check.sh --python-only
PYTHONDONTWRITEBYTECODE=1 "$UAV_RGB_PYTHON" tools/test_uart_action.py
PYTHONDONTWRITEBYTECODE=1 "$UAV_RGB_PYTHON" tools/test_splat_to_ply.py
PYTHONDONTWRITEBYTECODE=1 "$UAV_RGB_PYTHON" -m rl_teacher.test_core
```

各命令退出 0。关键输出：

```text
PASS（python-only）：C++/CUDA 构建、CTest、GPU 未验证；模拟采集、真实训练和学生闭环须独立验收。
PASS: uart_action 帧编码/CRC/心跳/seq/dry-run/坏配置/串口错误 全部通过
splat_to_ply: PASS
rl_teacher core PASS
```

第一条的“GPU 未验证”是该命令自身范围；本轮另执行了第 2 节 CTest，不把两者混成一次完整 check.sh 执行。没有执行无参数 `bash tools/check.sh`，因此未跑其中仅全量模式提供的 Python 打包→C++ read_dataset 编译检查和 join 的 native-check 分支。

主检查中 `ui/test_studio.py` 的一个用例出现 `ResourceWarning: unclosed file ... eval.log`，17 项仍为 OK；记录该警告，不将其写成零警告。临时测试目录由现有测试自行管理。

## 4. 3DGS CPU/mock 子集（仿真审计 agent 执行）

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
.venv-tools/bin/python -B -m unittest \
  sim.3dgs.test_dynamics \
  sim.3dgs.test_renderer.TestMockRenderer \
  sim.3dgs.test_renderer.TestGeometry.test_viewmat_forward \
  sim.3dgs.test_renderer.TestGeometry.test_yaw_left_turns_camera \
  sim.3dgs.test_renderer.TestGeometry.test_intrinsics \
  sim.3dgs.test_collision.TestCollision.test_inside_and_inflation \
  sim.3dgs.test_collision.TestCollision.test_ground_and_out_of_slice \
  sim.3dgs.test_collision.TestCollision.test_clearance \
  sim.3dgs.test_collision.TestCollision.test_empty
```

真实输出：`Ran 15 tests in 0.059s / OK`。未运行完整 discover 或 RUN_3DGS_REAL=1；没有把 15 项写成全部 20 项。

## 5. 已有数据与指标核对

主代理使用 `PYTHONDONTWRITEBYTECODE=1 "$UAV_RGB_PYTHON" -` 的只读脚本，读取下列 JSON，并逐张计算 PNG SHA256；没有改动数据文件：

- `/root/rivermind-data/uav_rgb/3dgs-campus-pilot-20261105-v2/collect-train/ep_00000/`
- 同根 `collect-val/ep_00000/`
- 同根 `run/closed_loop_eval_20261105T053248629351Z/eval.json`
- 同根 `run/eval-success/eval.json`、`run/eval-success2/eval.json`
- `/root/rivermind-data/uav_rgb/campus-paired-20261103/eval-cl-full/eval.json`

结果：

```text
3DGS train scene_id=0, seed=0, geometry_gid=None
3DGS val   scene_id=1, seed=1, geometry_gid=None
RGB equal: 10/10
场景：同一 PLY、同 spawn=(-14,0,2)、同采集目标=(-6,0,2)
三次 eval：同 ckpt sha256 a7436a153707e0c0ae2a76b7c9111c416572c4cec9619559da5481bc0d000ca7
三次均 steps=12, actions={1:12}, max_x=-7.309334483030743, collision=false
waypoint.x=-6.0 => SR=0
waypoint.x=-7.0 => SR=0
waypoint.x=-7.5 => SR=1
Gazebo campus: episodes_requested=80, episodes_done=80, success=0,
collision=4, failed=[], unique geometry_gid=10
```

训练 agent 另直接读取既有 UAVR 数据块，并调用 preprocess.scan/build_meta：`packed_train_equals_val=True`，train/val 类别分布均 `[0,10,0,0]`，`split_groups=[]`，打包扫描放行。以上是旧产物复核，不是本轮重跑获得的性能。

## 6. 缺陷复现证据（agent 内存脚本／dry-run）

各审计 agent 用 `python -B -` 内联脚本调用实际函数，对磁盘写入、进程或渲染依赖使用内存替身。脚本未持久化，以下保留输入、调用点和真实关键输出，避免误称正式回归测试。

| 复现 | 输入／调用 | 实际输出 |
| --- | --- | --- |
| 坐标系 | nav_to_world 平移 10 m，实际 collision.from_scene 与 viewmat 逻辑 | 导航 x=1 false，x=11 true |
| 穿障 | run_episode，hold=.5，障碍 x=1.05，端点 .839960→1.388678 | 占据点 true，但所有动作 collision=false |
| 返回错误 | 注入 scene.json 复制失败，实际 eval 主流程 | rc=0，physics_verified=false，collect_error 非空，done=1，failed=0，SR=1 |
| 转角 | 实际左转 9.6545588° | dyaw_deg=553.1654708° |
| 方向成功判据 | summarize_episode，同距目标 .5m，左右两侧 | x=-.5 false；x=.5 true |
| 补 seed 指标 | aggregate，3 成功、4 attempted、1 failed、requested=3 | requested SR=1，execution=1，completion=.75 |
| 时长丢失 | auto_eval_params + build_eval_command，hold_s=.2 | params 保留 .2，命令无 --hold |
| UI 串 run | 原 _attach_eval_to_run，A 评测、当前 B | B 摘要包含 A；B 可被写 finished/0 |
| monitor 回滚 | 原 _start_training，已有辅助文件、坏引号 | 本次写入0次，旧文件仍进入删除路径 |
| 启动误恢复 | 原恢复方法，starting、无 PID | 写 incomplete_unobserved 摘要 |
| UART 半行 | 真实子进程 tools/uart_action.py --dry-run --frames 4；第一帧后输入不带换行的1 | .701秒无输出；补换行后恢复并rc=0 |

UART 没有打开串口；UI 没有启动真实训练；没有实际删除用户日志。CF-Drone 两项采用源码路径和算术模型核对，不列为执行固件通过或失败。

## 7. 仓库与产物状态

```bash
git -C uav_rgb status --short --branch
git -C ALVLN status --short --branch
git -C CF-Drone diff --ignore-space-at-eol --stat
diff -qr -x .git -x build -x .venv-tools -x __pycache__ -x .ipynb_checkpoints uav_rgb ALVLN/uav_rgb
```

在新增审计文档之前，主线、镜像状态干净，忽略生成目录后的内容比较无差异。CF-Drone 保留既有大量行尾修改，忽略行尾后无差异。本轮仅新增报告、结果和过程三个 Markdown 文件。

根目录旧 `download_colmap.rc=18`、旧 ZIP 不完整；文档 agent 14:00 检查数据盘另一下载中的包为 65,138,688 B 且继续增长，ZIP 尚不完整。没有控制下载进程，不能把检查时快照当作最终下载结果。

## 8. 未验证

未运行真实 3DGS/Gazebo 闭环、完整模型训练、gsplat 真实 GPU 用例、全部 3DGS discover、干净容器构建、固件编译、实机/HIL、真实双窗口竞态和服务器网络安全检查。未新增正式缺陷回归测试，因为本轮是审计而非修复。
