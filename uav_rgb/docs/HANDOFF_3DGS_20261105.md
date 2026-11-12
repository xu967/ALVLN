# 交接：3DGS 在线仿真 + 默认视频 + UART 骨架（2026-11-05）

本文件是 2026-11-05 这条线的唯一入口。历史交接仍看 `docs/HANDOFF.md`，
指标口径看 `docs/EVAL_STANDARD.md`，实验细节看
`docs/experiments/2026-11-05-3dgs-pilot.md`、`2026-11-05-cf-blackbox.md`。

## 一、当前边界（已由用户确认）

- 视觉模型只输出 4 个动作（0=SAFE_NEUTRAL、1=FORWARD_SLOW、2=YAW_LEFT、3=YAW_RIGHT）。
- 部署目标：LicheePi SG2002 只做视觉→4 动作，通过 UART 把动作 ID 发给 ESP32；
  **ESP32 飞控不改**，负责姿态/电机；板端 NPU/量化/摄像头 runtime 不在本仓库范围。
- 仿真：3DGS 在线渲染 + 黑盒“4 动作→整机响应”，训练高度**固定 2 m**；
  3DGS 只负责视觉，碰撞用点云/体素代理。
- 数据、日志、视频、模型、venv 一律放 `$DATA_ROOT/uav_rgb/`；代码在 `code/uav_rgb/`。
- 真机参数/ULog/UART 帧格式尚未提供；当前一切黑盒参数 `calibrated=false`。

## 二、代码地图（本轮新增/改动）

| 路径 | 作用 |
| --- | --- |
| `sim/3dgs/renderer.py` | gsplat 渲染 RGB + expected depth；`MockRenderer` 供单测 |
| `sim/3dgs/scene.py` | PLY 读写、`nav_to_world`、`viewmat_from_pose`、合成场景 |
| `sim/3dgs/collision.py` | 点云高度切片体素占据 + 地面；**近似代理** |
| `sim/3dgs/dynamics.py` | 薄包装，直接 import `sim/tools/cf_blackbox.py`（参数单源） |
| `sim/3dgs/collect_3dgs.py` | 在线采集：位姿→4 动作→黑盒→碰撞→渲染→episode |
| `sim/3dgs/eval_3dgs.py` | 学生 C++ 在线闭环评测；`--record-video` 出 mp4 |
| `sim/3dgs/video.py` | `rgb_*.png` → mp4（cv2） |
| `sim/tools/cf_blackbox.py` | 4 动作黑盒状态模型；固定 z=2.0、延迟/阻力/偏航一阶 |
| `sim/dynamics/cf_blackbox_v1.json` | 黑盒参数唯一来源（占位，`calibrated=false`） |
| `tools/splat_to_ply.py` | `.splat → 3DGS PLY`；y-up→z-up、`--flip-height`、`--scale`、`--z-offset` |
| `tools/train_student_with_video.py` | 学生训练后自动 1 集闭环录像；读 `.uavr` meta 推断 task/split/dynamics/renderer |
| `tools/uart_action.py` | LicheePi 侧 4 动作 UART 发送；可配置帧/CRC/心跳；默认帧是占位 |
| `config/uart_protocol.example.json` | 串口帧配置示例（占位） |
| `teacher/train.py` | `--record-video` 默认开、1 集、失败只告警、时间戳目录 |
| `ui/studio.py` | 训练面板默认勾选“训练后保存视频”；训练后自动评测；3DGS 数据走 `eval_3dgs` |
| `sim/tools/collect_parallel.py` / `eval_closed_loop.py` | `cf_sil` 已退出默认矩阵；`cf_blackbox` 在 Gazebo 链路显式拒绝 |

## 三、环境

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb
source env.sh
```

- 3DGS venv（**只在数据盘**）：
  `$DATA_ROOT/uav_rgb/3dgs-venv`，torch 2.7.1+cu128、gsplat 1.5.3、
  numpy/scipy/pillow/opencv-python-headless。仓库里没有它的 requirements，从零搭建按
  `docs/experiments/2026-11-05-3dgs-pilot.md` 的安装段：

```bash
# 2026-11-05 实际用过的命令（本轮未重装验证；pip 缓存/JIT 产物按 pilot 文档放数据盘）
/usr/bin/python3 -m venv $DATA_ROOT/uav_rgb/3dgs-venv
$DATA_ROOT/uav_rgb/3dgs-venv/bin/pip install \
  'https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
$DATA_ROOT/uav_rgb/3dgs-venv/bin/pip install gsplat==1.5.3 numpy scipy pillow opencv-python-headless ninja jaxtyping rich packaging
unset TORCH_CUDA_ARCH_LIST   # 不要写死架构：gsplat 按本机 GPU JIT（显卡换过两次，型号以 nvidia-smi -L 为准），带着旧变量会编错架构
```

- 场景数据也只在数据盘。没有 Callinstrasse（下载地址见 pilot 文档 §场景表，MIT）时可先用仓库自带的
  合成走廊走通链路：`python -m sim.3dgs.scene make-synthetic --out <dir> --seed 0` 生成 `scene.json`，
  再把下面 §五 命令里的 `--scene` 换成它（2026-11-07 从零克隆实测：生成 + 单帧渲染 5 s，
  `collect_3dgs --episodes 1 --steps 10 --policy teacher` 7 s，首次 gsplat JIT 另计）。
- 学生 C++ 推理：`src/build/uavrgb_infer`（`build/check` 也可，用 `UAV_RGB_BUILD_DIR`）。
- 数据工具 venv：`.venv-tools`（已装 opencv-python-headless）。
- 首次 gsplat 会 JIT 编译 CUDA 扩展，按本机 GPU 编（2080 Ti 实测约 **228–235 s**）；
  之后走 `$TORCH_EXTENSIONS_DIR/gsplat_cuda`。流水线脚本（`tools/demo_3dgs_routes.py`、`tools/scene_import_wanderland.py`）
  把缓存目录定成 `$DATA_ROOT/uav_rgb/torch-cache/extensions-sm<算力>`（`torch.cuda.get_device_capability()`），换卡后
  自动落到新目录重新 JIT，不会拿别的架构的 cubin 跑。
- 下载/安装前建议：

```bash
export PIP_CACHE_DIR=$DATA_ROOT/uav_rgb/pip-cache
export TMPDIR=$DATA_ROOT/uav_rgb/tmp
export HF_HOME=$DATA_ROOT/uav_rgb/hf-cache
export TORCH_HOME=$DATA_ROOT/uav_rgb/torch-cache
```

## 四、场景与实测

### 场景目录（全部在数据盘）

| 场景 | 路径 | Gaussians | 许可 |
| --- | --- | ---: | --- |
| 合成走廊 | `3dgs-scenes/synthetic/` | 45,497 | 自建 |
| Mip-NeRF 360 bicycle | `3dgs-scenes/bicycle/bicycle_real.json` | 567,714 | HF 转换**未声明**，仅本地 |
| Callinstrasse 校园 | `3dgs-scenes/callinstrasse/scene.json` | 2,062,370 | **MIT** |
| HF 室内（room/kitchen/playroom/counter/bonsai） | `3dgs-scenes/hf/` | — | 未声明，仅本地 |

Callinstrasse PLY 为 511,469,292 B，16 路分段下载 + 大小校验后拼接；
COLMAP 位姿包（86 MB）后台下载中，尚未用于外参/尺度二次校验。

### 实测（RTX 2080 Ti，320×240）

| 场景 | 单帧渲染 | 采集 | 学生评测 | 视频 |
| --- | --- | --- | --- | --- |
| 合成走廊 | 0.92 ms | 8 步 | 8 步 | `3dgs-pilot-20261105/run7/.../seed_0.mp4` |
| bicycle | 2–13 ms | 12 步，第 9 步碰撞 | 10 步，第 9 步碰撞 | `3dgs-real-pilot-20261105/run/.../seed_0.mp4` |
| Callinstrasse | 4–14 ms | train 10 + val 10，无碰撞 | 12 步，**SR=1.0 (1/1)** | `3dgs-campus-pilot-20261105-v2/run/eval-success2/videos/seed_0.mp4` |

Callinstrasse 的 spawn=(-14,0,2)、waypoint=(-7.5,0,2)、pitch=0、
collision slice z=[1.8,2.2]、voxel 0.2 m、inflate 0.15 m。
**单场景 1 集，SR 只证明链路，不是统计结论。**

## 五、常用命令（都在 `code/uav_rgb` 下）

```bash
# 1) 单帧渲染检查
$DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m sim.3dgs.renderer \
  --scene $DATA_ROOT/uav_rgb/3dgs-scenes/callinstrasse/scene.json \
  --pose=-14,0,2,0,0 --width 320 --height 240 --fov 60 \
  --out /tmp/frame.png --depth-out /tmp/depth.npy --bench 10

# 2) 在线采集（teacher 标签）。--dynamics 默认 cf_sitl（2026-11-09 起，要 $CF_SITL_PROC 指向
#    按当前 sim/cf_sitl 构建的 cf_sitl_proc）；只想冒烟链路可加 --dynamics cf_blackbox。
$DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m sim.3dgs.collect_3dgs \
  --scene $DATA_ROOT/uav_rgb/3dgs-scenes/callinstrasse/scene.json \
  --out $DATA_ROOT/uav_rgb/run/collect-train \
  --episodes 1 --steps 10 --policy teacher --split train --seed 0 --truncate-on-collision

# 3) 打包（会复制 scene.json 到数据集根，自动绝对化 PLY 路径）
.venv-tools/bin/python tools/preprocess.py --drop-collision \
  --out $DATA_ROOT/uav_rgb/run/ds \
  $DATA_ROOT/uav_rgb/run/collect-train/ep_00000 \
  $DATA_ROOT/uav_rgb/run/collect-val/ep_00000

# 4) 学生训练 + 自动 1 集闭环录像（3DGS 数据自动走 eval_3dgs）
.venv-tools/bin/python tools/train_student_with_video.py \
  --data $DATA_ROOT/uav_rgb/run/ds/dataset.uavr \
  --out $DATA_ROOT/uav_rgb/run/train \
  --epochs 2 --batch 8 --lr 0.001 --seed 1

# 5) 单独评测（11-12 起：场景按几何哈希属于 train 或 val 分区时默认拒绝评测，rc=2；自采场景上做诊断要加 --allow-train-scene，eval.json 会标 split_policy=diag-train-scene）
$DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m sim.3dgs.eval_3dgs \
  --scene $DATA_ROOT/uav_rgb/run/ds/scene.json \
  --out $DATA_ROOT/uav_rgb/run/eval \
  --ckpt $DATA_ROOT/uav_rgb/run/train/ckpt_best.bin \
  --infer-bin src/build/uavrgb_infer \
  --episodes 1 --steps 12 --record-video

# 6) .splat → PLY
$DATA_ROOT/uav_rgb/3dgs-venv/bin/python tools/splat_to_ply.py \
  in.splat out.ply --flip-height --scale 2.0 --z-offset 0.516

# 7) UART 骨架（占位协议，dry-run 不碰硬件）
python3 tools/uart_action.py --dry-run --action 1
```

`--pose` 必须用 `=`（负坐标会被 argparse 当选项）。

## 六、测试与验收

```bash
bash tools/check.sh                     # 全量：构建 + CTest + Python
bash tools/check.sh --python-only       # 不碰 C++/GPU
$DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m unittest discover -s sim/3dgs -p 'test_*.py'
python3 tools/test_uart_action.py
python3 tools/test_train_student_with_video.py
.venv-tools/bin/python tools/test_splat_to_ply.py
```

本轮（2026-11-05）结果：`tools/check.sh` PASS；CTest 25/25；`sim/3dgs` 20 项（1 个真实
gsplat 用例默认 skip，`RUN_3DGS_REAL=1` 开启）；其余单测全过。
2026-11-07 从零克隆复跑：CTest 30/30；`sim/3dgs` 65 项 OK（2 skip，23 s）。

## 七、已知坑

1. **gsplat 首次 JIT 慢**（约 4 分钟）。不要并发两个 JIT 进程到同一个
   `TORCH_EXTENSIONS_DIR`，会 ninja 竞态。编好的缓存**不能复制到别的目录名下复用**：torch 的 JIT 版本哈希含
   `build_directory`（`torch/utils/_cpp_extension_versioner.py`），换了目录名第一次用必定整目录重编（2026-11-09
   `extensions-sm75-20261109`、`extensions-sm75` 各重编过一次）。新目录先单进程跑一集，编完再并行。
2. **坐标系**：gsplat viewmat 是 COLMAP/OpenCV 约定（相机 +z 朝前）。
   真实 3DGS 常是 y-up；用 `splat_to_ply.py --flip-height --scale --z-offset`
   对齐后，先用 `renderer` 看一帧再采集。`room`/`kitchen` 尚未校准好。
3. **spawn 必须在空域**：Callinstrasse 的 (0,0) 在建筑/树代理内，采集立刻碰撞；
   开阔路面从 (-14,0) 起飞。换场景先用 `collision.Occupancy` 扫一遍。
4. **成功判据**：`eval_3dgs` 复用 route-choice 分支，
   `success = 无碰撞 且 max_x >= waypoint.x 且 到 waypoint <= 1.0 m`。
   航向朝 +x 时 waypoint 不能设得比实际到达点更远。
5. **碰撞是代理**：点云切片 + 膨胀；薄树叶、玻璃、背面会漏检，
   不能当 Gazebo contact 或真实碰撞真值。
6. **数据身份**：3DGS 数据 `geometry_version=3dgs.v1`，与 Gazebo 数据不可混包；
   `cf_blackbox` 固定 z=2.0，Gazebo 采集仍是 `Z_TARGET=2.1`，不能混报。
7. **Callinstrasse 下载**：单连接 ~12 KB/s 且会 503；必须分段并发 + 最终
   大小校验，不能把 503 HTML 拼进 PLY。
8. **许可**：Callinstrasse 是 MIT；HF `dylanebert/3dgs` 转换未声明许可，
   只放数据盘本地验证，不随仓库分发。

## 八、Git 与推送

- `uav_rgb` 本地无 remote；本轮最新提交以 `git log --oneline -1` 为准。
- 推送走 `ALVLN` 镜像：

```bash
cd /root/claude-code-backup-20261028/code
rsync -a --delete --exclude=.git --exclude=build --exclude=src/build \
  --exclude=.venv-tools --exclude=.ipynb_checkpoints --exclude=__pycache__ \
  --exclude='*.pyc' uav_rgb/ ALVLN/uav_rgb/
cd ALVLN && git status --short && git add uav_rgb && git commit -m "..." && git push origin main
```

## 九、下一步

1. Callinstrasse COLMAP 位姿包下完后，做尺度/外参二次校验；
   再考虑多集、多种子评测（当前 1/1 不是统计结论）。
2. 校准 `room`/`kitchen` 的朝向/尺度，作为第二个真实场景。
3. 等用户提供：
   - `real-drone/uart_protocol.md`（帧头/长度/动作编码/波特率/心跳/CRC）
   - `real-drone/flight.ulg`（PX4/外部，含 local_position/attitude/actuator）
   - BOM/质量/桨/电池/悬停油门
   到位后标定 `cf_blackbox_v1.json`，把 `calibrated` 改为 true。
4. 真机前：无桨台架 + `tools/uart_action.py` 字节流验证 + 失效注入
   （停发/CRC 错/乱序）；CF-Drone 仓库固件没有 4 动作 UART 解析器，
   需确认实飞固件版本或走串口调试助手/WiFi。
